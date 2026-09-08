#!/usr/bin/env python3
"""Greedy HF vs SGLang on a few prompts (sequential — not two 12B copies).

    INFER_BACKEND=hf python scripts/compare_infer_greedy.py --write /tmp/hf.json
    INFER_BACKEND=sglang SGLANG_MODEL_PATH=weights/.../merged_... \
        python scripts/compare_infer_greedy.py --against /tmp/hf.json

Mismatch is a hard fail (template drift or backend bug), not a silent
difference. Run PEFT-vs-merged under HF first
(``export_merged_checkpoint.py --compare``) before blaming SGLang.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")


def _prompts(img: Path) -> list[list[dict]]:
    return [
        [{"role": "user", "content": [
            {"type": "image", "url": str(img)},
            {"type": "text", "text":
             "In one short sentence, what colors do you see?"},
        ]}],
        [{"role": "user", "content": [
            {"type": "image", "url": str(img)},
            {"type": "text", "text":
             "Answer briefly: is the grid mostly empty?"},
        ]}],
        [{"role": "user", "content": [
            {"type": "text", "text": "Reply with the single word: ok"},
        ]}],
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--write", type=Path, help="write greedy replies to JSON")
    p.add_argument("--against", type=Path,
                   help="compare this run to a previously written JSON")
    p.add_argument("--max-new-tokens", type=int, default=48)
    args = p.parse_args(argv)
    if bool(args.write) == bool(args.against):
        raise SystemExit("pass exactly one of --write or --against")

    from PIL import Image

    from agent.model import get_model
    from agent.speed_flags import infer_backend

    backend = infer_backend()
    model = get_model()
    model._sampling_kwargs = lambda: {"do_sample": False}
    with tempfile.TemporaryDirectory(prefix="cmp_infer_") as td:
        img = Path(td) / "board.png"
        Image.new("RGB", (64, 64), (20, 90, 20)).save(img)
        prompts = _prompts(img)
        replies = [
            model.generate(p, max_new_tokens=args.max_new_tokens)
            for p in prompts
        ]

    payload = {
        "backend": backend,
        "replies": replies,
        "n": len(replies),
    }
    if args.write:
        args.write.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {len(replies)} {backend} replies to {args.write}")
        return 0

    other = json.loads(args.against.read_text(encoding="utf-8"))
    other_replies = other["replies"]
    if len(other_replies) != len(replies):
        raise SystemExit(
            f"reply count {len(replies)} vs {len(other_replies)}"
        )
    mism = [
        (i, other_replies[i], replies[i])
        for i in range(len(replies))
        if other_replies[i] != replies[i]
    ]
    if mism:
        print(
            f"{other.get('backend')} vs {backend} MISMATCH "
            "(hard fail -- template or backend drift):",
            flush=True,
        )
        for i, a, b in mism:
            print(f"  [{i}] {other.get('backend')}={a!r}\n"
                  f"       {backend}={b!r}", flush=True)
        return 1
    print(
        f"{other.get('backend')} vs {backend} greedy match "
        f"on {len(replies)} prompt(s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
