"""Train-epoch wall estimate. Production flags only. Last line is the number.

    python -m training.probe_train_time
    T10_DATAGEN_LABEL=aug27_big_step_iter1 python -m training.probe_train_time

Same stack as t10 / weekend train (materialize + fence + 4-bit LoRA +
chunked KD + train prefix-KV), but one mode and fewer timed batches so
it stays under an hour. Needs a REAL datagen corpus -- a 32-gen bench
label will not estimate a 3000-gen epoch.

Does not save a checkpoint. Remote GPU. No NAMS.
"""

from __future__ import annotations

import argparse
import math
import os
import random as pyrandom
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_MIN_PLAYER_RECORDS = 500


def _count_jsonl(path: Path) -> int:
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--label", default=None,
        help="data_game/<label>/ (default T10_DATAGEN_LABEL or "
             "overnight_iter1)",
    )
    p.add_argument(
        "--batches", type=int, default=2,
        help="timed micro-batches per source after one warmup (default 2)",
    )
    args = p.parse_args(argv)
    if args.batches < 1:
        raise SystemExit("--batches must be >= 1")
    os.chdir(REPO_ROOT)
    os.environ.setdefault("GS_CHUNKED_KD", "1")
    os.environ.setdefault("GS_TRAIN_PREFIX_KV", "1")

    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")

    from PIL import Image
    import torch
    from transformers import AutoProcessor

    from agent.config import CONFIG
    from agent.model import ADAPTERS, spec_for
    from training.bench_speed import VramWatch
    from training.external_data import sources_from_manifest
    from training.game_traces import (
        AnalystTraceSource,
        GameTraceSource,
        PlayerAnchorSource,
    )
    from training.selftest import _T10_LABEL
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

    label = args.label or _T10_LABEL
    traces_dir = REPO_ROOT / "data_game" / label
    traces = traces_dir / "traces.jsonl"
    analyst = traces_dir / "analyst_traces.jsonl"
    if not traces.is_file():
        raise SystemExit(
            f"No {traces}. Point --label or T10_DATAGEN_LABEL at a "
            "3000-gen datagen directory, not a 32-gen bench folder."
        )
    n_player = _count_jsonl(traces)
    print(
        f"corpus {label}: {n_player} player records, "
        f"analyst_jsonl={'yes' if analyst.is_file() else 'MISSING'}",
        flush=True,
    )
    if n_player < _MIN_PLAYER_RECORDS:
        raise SystemExit(
            f"{n_player} player records < {_MIN_PLAYER_RECORDS}. "
            "This estimate would be a tiny-corpus mix (replay-dominated). "
            "Use a weekend-sized traces.jsonl."
        )

    print(
        "production train flags "
        f"(GS_CHUNKED_KD={os.environ.get('GS_CHUNKED_KD')} "
        f"GS_TRAIN_PREFIX_KV={os.environ.get('GS_TRAIN_PREFIX_KV')}); "
        f"{args.batches} timed batch(es)/source. "
        "Target well under 1 h (materialize is the long pole).",
        flush=True,
    )

    sources = [
        GameTraceSource(traces),
        PlayerAnchorSource(traces),
        AnalystTraceSource(analyst) if analyst.is_file() else None,
        *sources_from_manifest(),
    ]
    sources = [s for s in sources if s is not None]
    cfg = TrainConfig(label="probe_train_time")
    src_weights = {s.name: s.weight for s in sources}
    spec = spec_for(cfg.architecture or CONFIG.model_key)

    t_setup0 = time.perf_counter()
    fence_proc = AutoProcessor.from_pretrained(
        spec.hf_id, token=CONFIG.hf_token,
        trust_remote_code=spec.trust_remote_code,
    )
    with tempfile.TemporaryDirectory(prefix="probe_fence_") as td:
        img = Path(td) / "m.png"
        Image.new("RGB", (32, 32), (18, 18, 18)).save(img)
        image_soft = measure_image_soft_tokens(fence_proc, str(img))
    print("materializing (fence + noise + manifest)...", flush=True)
    by_source = materialize(
        sources, cfg, processor=fence_proc, image_soft_tokens=image_soft,
    )
    if not by_source:
        raise SystemExit("materialize produced no sources")
    print(
        f"materialized {sum(len(v) for v in by_source.values())} examples "
        f"from {len(by_source)} source(s)",
        flush=True,
    )

    print("loading 4-bit model + LoRA...", flush=True)
    model, processor, _, _ = build_model(spec, cfg, CONFIG.hf_token)
    attach_neftune(model, cfg.neftune_alpha)
    terminator = resolve_terminator_id(
        model, getattr(processor, "tokenizer", processor)
    )
    collator = Collator(
        processor, ADAPTERS[spec.family], terminator,
        compute_dtype=torch.bfloat16, device=cfg.device,
    )
    import bitsandbytes as bnb
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(
        params, lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    model.train()
    setup_s = time.perf_counter() - t_setup0
    print(f"setup {setup_s / 60:.1f} min; timing batches...", flush=True)

    rng = pyrandom.Random(0)
    planned = {
        name: epoch_batches(list(exs), cfg.micro_batch, rng)
        for name, exs in sorted(by_source.items())
    }
    prefix_state = PrefixKVRuntime()

    def _one(exs: list) -> float:
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
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss on {exs[0].source}")
        (loss / cfg.grad_accum).backward()
        torch.cuda.synchronize()
        return time.perf_counter() - t

    first = next(iter(planned.values()))[0]
    _one(first)
    optimizer.zero_grad(set_to_none=True)
    prefix_state.cache = None
    prefix_state.key = None

    per_source: list[tuple[str, float, int, float]] = []
    opt_s: float | None = None
    with VramWatch() as watch:
        for name, batches in planned.items():
            if not batches:
                continue
            n_time = min(args.batches, len(batches))
            times = [_one(b) for b in batches[:n_time]]
            torch.cuda.synchronize()
            t = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step_s = time.perf_counter() - t
            opt_s = step_s if opt_s is None else min(opt_s, step_s)
            prefix_state.opt_steps += 1
            if (cfg.prefix_kv_refresh_steps > 0
                    and prefix_state.opt_steps % cfg.prefix_kv_refresh_steps
                    == 0):
                prefix_state.cache = None
                prefix_state.key = None
            mean_s = sum(times) / len(times)
            exs = by_source[name]
            eff = min(cfg.micro_batch, exs[0].batch_cap or cfg.micro_batch)
            n_epoch = math.ceil(
                max(1, round(src_weights.get(name, 1.0) * len(exs))) / eff
            )
            peak = torch.cuda.max_memory_allocated() / 2**30
            per_source.append((name, mean_s, n_epoch, peak))
            print(
                f"  {name}: {mean_s:.2f}s/batch  "
                f"{n_epoch} batches/epoch  torch_peak={peak:.1f} GiB",
                flush=True,
            )

    train_s = sum(mean_s * n_epoch for _, mean_s, n_epoch, _ in per_source)
    n_epoch_total = sum(n for _, _, n, _ in per_source)
    train_s += (n_epoch_total / cfg.grad_accum) * (opt_s or 0.0)
    epoch_h = (setup_s + train_s) / 3600
    vram = watch.snapshot()

    print()
    print(
        f"setup {setup_s / 60:.1f} min + {n_epoch_total} micro-batches "
        f"+ opt every {cfg.grad_accum}. "
        f"Held-out eval at save_steps={cfg.save_steps} is NOT in this "
        f"number (two hooks on a ~300-step epoch; t10: measure one, ×2).",
        flush=True,
    )
    print(
        f"VRAM peak nvidia-smi={vram['peak_smi_GiB']} GiB  "
        f"torch={vram['peak_torch_GiB']} GiB",
        flush=True,
    )
    # Last line: the only number the weekend-fit question needs.
    print(f"TRAIN EPOCH ESTIMATE: {epoch_h:.2f} hours", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
