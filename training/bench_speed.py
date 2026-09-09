"""Control-vs-opt timing: inference (t11-shaped) and train (t10-shaped).

    python -m training.bench_speed --what infer --modes control,bc
    python -m training.bench_speed --what train --modes control,opt

Infer:

* ``control`` = ``GS_PREFIX_KV=0`` (leftover left-pad). Parity stays off.
* ``bc`` = production B+C defaults.
* ``parity`` (optional third column) = leftover left-pad + ``GS_PAD_PARITY=1``.
  Not the control.
* ``sglang`` is added on the ``infer-sglang`` branch.

Train: one token-fenced corpus, then vary only the compute flags.

* ``control`` = ``GS_CHUNKED_KD=0 GS_TRAIN_PREFIX_KV=0``.
* ``opt`` = chunked KD + train prefix-KV.
Fence drop counts print once from materialize, not by comparing two corpora.

Pretty-prints a table. GPU + (for infer) NAMS. Remote only.

Infer also samples VRAM: ``peak_smi_GiB`` is ``nvidia-smi`` memory.used
(the box topline -- Gemma + MiniLM + fragmentation); ``peak_torch_GiB``
is this process's caching allocator. The previous infer bench did not
record either; train already printed ``peak_GiB``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def nvidia_smi_used_mib() -> int | None:
    """Box-wide used MiB, or None if nvidia-smi is missing/unparseable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    total = 0
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            total += int(line)
        except ValueError:
            return None
    return total


class VramWatch:
    """Sample nvidia-smi while a block runs; also torch allocator peak."""

    def __init__(self, interval_s: float = 2.0):
        self.interval_s = interval_s
        self.peak_smi_mib = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "VramWatch":
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        mib = nvidia_smi_used_mib()
        if mib is not None:
            self.peak_smi_mib = mib
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="vram-watch", daemon=True,
        )
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            mib = nvidia_smi_used_mib()
            if mib is not None:
                self.peak_smi_mib = max(self.peak_smi_mib, mib)
            self._stop.wait(self.interval_s)

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def snapshot(self) -> dict[str, float | None]:
        import torch
        torch_gib = None
        if torch.cuda.is_available():
            torch_gib = round(torch.cuda.max_memory_allocated() / 2**30, 1)
        end = nvidia_smi_used_mib()
        peak = self.peak_smi_mib or end
        return {
            "peak_smi_GiB": (
                None if peak is None else round(peak / 1024.0, 1)
            ),
            "end_smi_GiB": None if end is None else round(end / 1024.0, 1),
            "peak_torch_GiB": torch_gib,
        }


def _set_flags(mode: str, what: str) -> None:
    if what == "infer":
        os.environ.pop("INFER_BACKEND", None)
        if mode == "control":
            os.environ["GS_PREFIX_KV"] = "0"
            os.environ.pop("GS_PAD_PARITY", None)
        elif mode == "bc":
            os.environ["GS_PREFIX_KV"] = "1"
            os.environ.pop("GS_PAD_PARITY", None)
        elif mode == "parity":
            os.environ["GS_PREFIX_KV"] = "0"
            os.environ["GS_PAD_PARITY"] = "1"
        elif mode == "sglang":
            raise SystemExit(
                "sglang mode is not on this branch -- check out "
                "infer-sglang and use --modes control,bc,sglang"
            )
        else:
            raise SystemExit(f"unknown infer mode {mode!r}")
        return
    if what == "train":
        if mode == "control":
            os.environ["GS_CHUNKED_KD"] = "0"
            os.environ["GS_TRAIN_PREFIX_KV"] = "0"
        elif mode == "opt":
            os.environ["GS_CHUNKED_KD"] = "1"
            os.environ["GS_TRAIN_PREFIX_KV"] = "1"
        else:
            raise SystemExit(f"unknown train mode {mode!r}")
        return
    raise SystemExit(f"unknown --what {what!r}")


def _print_table(rows: list[dict], keys: list[str]) -> None:
    widths = {k: max(len(k), *(len(str(r.get(k, ""))) for r in rows))
              for k in keys}
    print("  ".join(k.ljust(widths[k]) for k in keys))
    print("  ".join("-" * widths[k] for k in keys))
    for r in rows:
        print("  ".join(str(r.get(k, "")).ljust(widths[k]) for k in keys))


def _run_infer(modes: list[str]) -> list[dict]:
    from training.selftest import (
        _T11_GAMES,
        _T11_MOVES,
        _T11_PARALLEL,
        _warmed_timed_datagen,
    )

    extra = [
        "--games", str(_T11_GAMES),
        "--max-moves", str(_T11_MOVES),
        "--seed", "11",
    ]
    out: list[dict] = []
    for mode in modes:
        _set_flags(mode, "infer")
        from agent.model import get_model
        model = get_model()
        model._prefix_kv.clear()
        model._prefix_kv_len.clear()
        if hasattr(model, "verify_pad_parity"):
            model._verify_pad_parity = None
        label = f"bench_infer_{mode}"
        with VramWatch() as watch:
            summary = _warmed_timed_datagen(
                label, parallel=_T11_PARALLEL, extra_args=extra,
            )
        vram = watch.snapshot()
        out.append({
            "mode": mode,
            "warmup_s": summary.get("warmup_s"),
            "steady_s/gen": summary.get("steady_s_per_gen"),
            "blended_s/gen": summary.get("seconds_per_generation"),
            "3000_h": summary.get("epoch_3000_hours"),
            "wall_s": summary.get("wall_seconds"),
            "gens": summary.get("generations"),
            "peak_smi_GiB": vram["peak_smi_GiB"],
            "peak_torch_GiB": vram["peak_torch_GiB"],
        })
    return out


def _run_train(modes: list[str]) -> list[dict]:
    """Materialize once; time the same batches under each flag set."""
    import math
    import random as pyrandom
    import tempfile
    import time

    import torch
    from PIL import Image
    from transformers import AutoProcessor

    from agent.config import CONFIG
    from agent.model import ADAPTERS, spec_for
    from training.external_data import sources_from_manifest
    from training.game_traces import (
        AnalystTraceSource,
        GameTraceSource,
        PlayerAnchorSource,
    )
    from training.selftest import _T10_BATCHES_PER_SOURCE, _T10_LABEL
    from training.token_fence import measure_image_soft_tokens
    from training.train import (
        Collator,
        PrefixKVRuntime,
        TrainConfig,
        attach_neftune,
        build_model,
        epoch_batches,
        materialize,
        resolve_terminator_id,
        weighted_loss,
    )

    traces_dir = REPO_ROOT / "data_game" / _T10_LABEL
    if not (traces_dir / "traces.jsonl").is_file():
        raise SystemExit(
            f"bench train needs {traces_dir}/traces.jsonl "
            "(or set T10_DATAGEN_LABEL)"
        )
    sources = [
        GameTraceSource(traces_dir / "traces.jsonl"),
        PlayerAnchorSource(traces_dir / "traces.jsonl"),
        AnalystTraceSource(traces_dir / "analyst_traces.jsonl"),
        *sources_from_manifest(),
    ]
    cfg = TrainConfig(label="bench_speed")
    src_weights = {s.name: s.weight for s in sources}
    spec = spec_for(cfg.architecture or CONFIG.model_key)
    fence_proc = AutoProcessor.from_pretrained(
        spec.hf_id, token=CONFIG.hf_token,
        trust_remote_code=spec.trust_remote_code,
    )
    with tempfile.TemporaryDirectory(prefix="bench_fence_") as td:
        img = Path(td) / "m.png"
        Image.new("RGB", (32, 32), (18, 18, 18)).save(img)
        image_soft = measure_image_soft_tokens(fence_proc, str(img))
    by_source = materialize(
        sources, cfg, processor=fence_proc, image_soft_tokens=image_soft,
    )
    print(
        "fence drops are the materialize WARNING lines above "
        "(same corpus for every train mode)",
        flush=True,
    )

    model, processor, _, _ = build_model(spec, cfg, CONFIG.hf_token)
    attach_neftune(model, cfg.neftune_alpha)
    terminator = resolve_terminator_id(
        model, getattr(processor, "tokenizer", processor)
    )
    collator = Collator(processor, ADAPTERS[spec.family], terminator,
                        compute_dtype=torch.bfloat16, device=cfg.device)
    import bitsandbytes as bnb
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(
        params, lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    model.train()
    rng = pyrandom.Random(0)
    planned = {
        name: epoch_batches(list(exs), cfg.micro_batch, rng)
        for name, exs in sorted(by_source.items())
    }

    def _one(exs: list, prefix_state: PrefixKVRuntime) -> float:
        torch.cuda.synchronize()
        t = time.perf_counter()
        built = collator.build_batch(exs)
        loss = weighted_loss(
            model, built["model_inputs"], built["weights"],
            loss_kind=exs[0].loss,
            prefix_kv=prefix_state,
            prefix_len=exs[0].prefix_n_tokens,
            prefix_hash=exs[0].prefix_hash,
            kd_chunk=cfg.kd_lm_head_chunk,
        )
        (loss / cfg.grad_accum).backward()
        torch.cuda.synchronize()
        return time.perf_counter() - t

    rows: list[dict] = []
    first = next(iter(planned.values()))[0]
    for mode in modes:
        _set_flags(mode, "train")
        prefix_state = PrefixKVRuntime()
        _one(first, prefix_state)
        optimizer.zero_grad(set_to_none=True)
        prefix_state.cache = None
        prefix_state.key = None
        peak = 0.0
        batch_s: list[float] = []
        opt_s: float | None = None
        for name, batches in planned.items():
            torch.cuda.reset_peak_memory_stats()
            times = [
                _one(b, prefix_state)
                for b in batches[:_T10_BATCHES_PER_SOURCE]
            ]
            batch_s.extend(times)
            peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
            torch.cuda.synchronize()
            t = time.perf_counter()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step_s = time.perf_counter() - t
            opt_s = step_s if opt_s is None else min(opt_s, step_s)
            prefix_state.opt_steps += 1
            prefix_state.cache = None
            prefix_state.key = None
        mean_s = sum(batch_s) / len(batch_s)
        n_epoch = 0
        for name, exs in by_source.items():
            eff = min(cfg.micro_batch, exs[0].batch_cap or cfg.micro_batch)
            n_epoch += math.ceil(
                max(1, round(src_weights.get(name, 1.0) * len(exs))) / eff
            )
        train_s = n_epoch * mean_s
        train_s += (n_epoch / cfg.grad_accum) * (opt_s or 0.0)
        rows.append({
            "mode": mode,
            "steady_s/batch": round(mean_s, 2),
            "peak_GiB": round(peak, 1),
            "epoch_h": round(train_s / 3600, 2),
            "n_epoch_batches": n_epoch,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--what", choices=("infer", "train"), required=True)
    p.add_argument(
        "--modes", default=None,
        help="comma-separated; infer default control,bc; train default "
             "control,opt",
    )
    args = p.parse_args(argv)
    os.chdir(REPO_ROOT)
    if args.what == "infer":
        modes = [m.strip() for m in (args.modes or "control,bc").split(",")
                 if m.strip()]
        rows = _run_infer(modes)
        keys = ["mode", "warmup_s", "steady_s/gen", "blended_s/gen",
                "3000_h", "wall_s", "gens", "peak_smi_GiB",
                "peak_torch_GiB"]
    else:
        modes = [m.strip() for m in (args.modes or "control,opt").split(",")
                 if m.strip()]
        rows = _run_train(modes)
        keys = ["mode", "steady_s/batch", "peak_GiB", "epoch_h",
                "n_epoch_batches"]
    print()
    _print_table(rows, keys)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
