"""Fast VRAM probe for parallel infer -- not a timing bench.

    python -m training.probe_infer_vram
    python -m training.probe_infer_vram --parallel 10

Loads the 12B once, opens ``--parallel`` NAMS sessions (MiniLM on the
same GPU -- that is what killed p16), then one player ``generate_batch``
and one analyst ``generate_batch`` at that width. Samples nvidia-smi
every 2 s.

Budget: one model load + two short batches. About 3-8 min, not an hour.
Does not play 8×4 games. For the 3000-gen clock use
``python -m training.bench_speed --what infer``.

``peak_smi_GiB`` is the box topline. On 96 GiB, p12 OOM'd at T~7k and
p16 died CUBLAS; this probe is how you see headroom before bumping
``--parallel``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _smi_line(watch, label: str) -> None:
    snap = watch.snapshot()
    print(
        f"{label}:  smi_now={snap['end_smi_GiB']} GiB  "
        f"smi_peak={snap['peak_smi_GiB']} GiB  "
        f"torch_peak={snap['peak_torch_GiB']} GiB",
        flush=True,
    )


def _stretch(model, prompts: list, min_tokens: int) -> None:
    """Grow the last text part until the longest encode is >= min_tokens."""
    if min_tokens <= 0:
        return
    import copy

    def _len(msgs: list) -> int:
        return int(model.encode_messages(msgs)["input_ids"].shape[1])

    longest = max(prompts, key=_len)
    if _len(longest) >= min_tokens:
        return
    msgs = copy.deepcopy(longest)
    part = None
    for m in reversed(msgs):
        for p in reversed(m.get("content") or []):
            if isinstance(p, dict) and p.get("type") == "text":
                part = p
                break
        if part is not None:
            break
    if part is None:
        raise RuntimeError("no text part to stretch")
    while _len(msgs) < min_tokens:
        part["text"] += " again" * 16
    # Replace the longest row so the batch actually has that T.
    idx = max(range(len(prompts)), key=lambda i: _len(prompts[i]))
    prompts[idx] = msgs
    print(
        f"stretched row {idx} to {_len(msgs)} tokens "
        f"(--min-prompt-tokens {min_tokens})",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--parallel", type=int, default=8,
                   help="batch width and NAMS session count (default 8)")
    p.add_argument("--checkpoint", default=None)
    p.add_argument(
        "--min-prompt-tokens", type=int, default=0,
        help="if >0, stretch one row to at least this many tokens "
             "(late-game prefill stress; encode-first)",
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=32,
        help="decode length (default 32 -- peak is usually prefill + KV)",
    )
    args = p.parse_args(argv)
    if args.parallel < 2:
        raise SystemExit("--parallel must be >= 2 (need a real batch)")
    if args.checkpoint:
        os.environ["MODEL_CHECKPOINT"] = args.checkpoint
    os.chdir(REPO_ROOT)

    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
    os.environ["GS_PREFIX_KV"] = "1"
    os.environ.pop("GS_PAD_PARITY", None)
    os.environ["MODEL_DO_SAMPLE"] = "0"

    from agent.model import get_model, set_default_checkpoint
    from training.bench_speed import VramWatch
    from training.infer_situations import (
        _build_analyst_prompts,
        _build_player_prompts,
        _import_runtime,
        _make_session,
    )

    if args.checkpoint:
        set_default_checkpoint(args.checkpoint)

    print(
        f"VRAM probe: parallel={args.parallel}  "
        f"max_new={args.max_new_tokens}  "
        f"min_prompt_tokens={args.min_prompt_tokens or 'natural'}",
        flush=True,
    )
    print("Not a timing bench. Target 3-8 min after the 12B starts loading.",
          flush=True)

    with VramWatch() as watch:
        print("loading model...", flush=True)
        model = get_model()
        original = model._sampling_kwargs
        model._sampling_kwargs = lambda: {"do_sample": False}
        _smi_line(watch, "after 12B load")

        print(f"opening {args.parallel} NAMS sessions (MiniLM)...", flush=True)
        rt = _import_runtime()
        sessions = [
            _make_session(rt, f"vram_p{args.parallel}_{i}", model)
            for i in range(args.parallel)
        ]
        try:
            _smi_line(watch, "after NAMS x" + str(args.parallel))
            print("building prompts...", flush=True)
            player = _build_player_prompts(
                sessions[0], model, rt, n=args.parallel,
            )
            analyst = _build_analyst_prompts(
                sessions[0], model, rt, n=args.parallel,
            )
            _stretch(model, player, args.min_prompt_tokens)
            _stretch(model, analyst, args.min_prompt_tokens)
            plens = [
                int(model.encode_messages(m)["input_ids"].shape[1])
                for m in player
            ]
            alens = [
                int(model.encode_messages(m)["input_ids"].shape[1])
                for m in analyst
            ]
            print(f"player token lengths: {plens}", flush=True)
            print(f"analyst token lengths: {alens}", flush=True)

            print(f"player generate_batch n={args.parallel}...", flush=True)
            model.generate_batch(
                [{"messages": m} for m in player],
                max_new_tokens=args.max_new_tokens,
                stop_regex=rt["game_io"].PLAYER_STOP_PATTERN,
            )
            _smi_line(watch, "after player batch")

            print(f"analyst generate_batch n={args.parallel}...", flush=True)
            model.generate_batch(
                [{"messages": m} for m in analyst],
                max_new_tokens=args.max_new_tokens,
            )
            _smi_line(watch, "after analyst batch")
        finally:
            model._sampling_kwargs = original
            for s in sessions:
                s.close()

        snap = watch.snapshot()
        print()
        print(
            f"PEAK  nvidia-smi={snap['peak_smi_GiB']} GiB  "
            f"torch={snap['peak_torch_GiB']} GiB  "
            f"end_smi={snap['end_smi_GiB']} GiB",
            flush=True,
        )
        print(
            "Read peak nvidia-smi against 96 GiB. "
            "p8 finished the hour bench; this is the number for p10/p12.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
