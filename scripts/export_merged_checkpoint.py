#!/usr/bin/env python3
"""Merge a PEFT adapter (incl. modules_to_save: embed_vision) into one HF folder.

SGLang and vLLM cannot load ``modules_to_save: embed_vision``. Inference
today is bf16 + adapter, not 4-bit. This writes a standalone folder that
those stacks can serve as the base.

    python scripts/export_merged_checkpoint.py \
        --architecture gemma-4-12b \
        --checkpoint weekend_iter1_step400 \
        --out weights/gemma-4-12b/merged_weekend_iter1_step400

``--compare`` greedy-compares the unmerged PEFT load vs the merged folder
under HF (same process, sequential: PEFT first, then unload, then merged).
Do this BEFORE blaming SGLang for output drift -- the merge is its own
suspect.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--architecture", default=None,
                   help="MODEL_REGISTRY key (default: MODEL_KEY / gemma-4-12b)")
    p.add_argument("--checkpoint", required=True,
                   help="adapter folder name under weights/<architecture>/")
    p.add_argument("--out", required=True, help="destination HF folder")
    p.add_argument("--compare", action="store_true",
                   help="greedy PEFT vs merged under HF after export")
    p.add_argument("--max-new-tokens", type=int, default=48)
    args = p.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    from agent.config import CONFIG
    from agent.model import checkpoint_dir, spec_for

    arch = args.architecture or CONFIG.model_key
    spec = spec_for(arch)
    ckpt = checkpoint_dir(spec.key, args.checkpoint, CONFIG)
    if not (ckpt / "adapter_config.json").is_file():
        raise SystemExit(f"no adapter_config.json under {ckpt}")
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    adapter_cfg = json.loads(
        (ckpt / "adapter_config.json").read_text(encoding="utf-8")
    )
    saved = adapter_cfg.get("modules_to_save") or []
    print(f"merging {ckpt} onto {spec.hf_id} (modules_to_save={saved})",
          flush=True)

    dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(
        spec.hf_id, token=CONFIG.hf_token,
        trust_remote_code=spec.trust_remote_code,
    )
    base = AutoModelForMultimodalLM.from_pretrained(
        spec.hf_id,
        dtype=dtype,
        device_map="cpu",
        token=CONFIG.hf_token,
        trust_remote_code=spec.trust_remote_code,
    )
    peft = PeftModel.from_pretrained(base, str(ckpt), is_trainable=False)
    merged = peft.merge_and_unload()
    merged.save_pretrained(out)
    processor.save_pretrained(out)
    meta = {
        "source_architecture": spec.key,
        "source_hf_id": spec.hf_id,
        "source_checkpoint": args.checkpoint,
        "modules_to_save": saved,
        "note": "merged LoRA + modules_to_save for SGLang / non-PEFT serve",
    }
    (out / "merge_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"wrote merged model to {out}", flush=True)

    if not args.compare:
        return 0
    return _compare_peft_vs_merged(
        spec, args.checkpoint, out, args.max_new_tokens,
    )


def _compare_peft_vs_merged(
    spec, checkpoint: str, merged_dir: Path, max_new_tokens: int,
) -> int:
    """Sequential greedy: PEFT VLModel, unload, merged-as-base VLModel."""
    import tempfile
    from dataclasses import replace

    from PIL import Image

    from agent.model import VLModel, reset_default

    prompts: list[list[dict]] = []
    with tempfile.TemporaryDirectory(prefix="merge_cmp_") as td:
        img = Path(td) / "board.png"
        Image.new("RGB", (64, 64), (30, 80, 30)).save(img)
        q = "In one short sentence, what do you see?"
        prompts.append([{"role": "user", "content": [
            {"type": "image", "url": str(img)},
            {"type": "text", "text": q},
        ]}])
        prompts.append([{"role": "user", "content": [
            {"type": "text", "text": "Reply with the single word: ok"},
        ]}])

        reset_default()
        peft = VLModel(spec, checkpoint=checkpoint).load()
        peft._sampling_kwargs = lambda: {"do_sample": False}
        peft_replies = [
            peft.generate(p, max_new_tokens=max_new_tokens) for p in prompts
        ]
        peft.unload()
        reset_default()

        merged_spec = replace(spec, hf_id=str(merged_dir))
        merged = VLModel(merged_spec, checkpoint=None).load()
        merged._sampling_kwargs = lambda: {"do_sample": False}
        merged_replies = [
            merged.generate(p, max_new_tokens=max_new_tokens) for p in prompts
        ]
        merged.unload()
        reset_default()

    mism = [
        (i, a, b)
        for i, (a, b) in enumerate(zip(peft_replies, merged_replies))
        if a != b
    ]
    if mism:
        print("PEFT vs merged MISMATCH (do not blame SGLang yet):", flush=True)
        for i, a, b in mism:
            print(f"  [{i}] peft={a!r}\n       merged={b!r}", flush=True)
        return 1
    print(
        f"PEFT vs merged greedy match on {len(prompts)} prompt(s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
