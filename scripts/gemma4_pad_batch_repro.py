"""Reproduction: Gemma 4 Unified corrupts a LEFT-PADDED row inside a
multi-row multimodal batch, while the identical padded row alone (batch=1)
and the unpadded rows of the same batch are fine.

Self-contained: draws a synthetic image with PIL, builds two chat prompts of
different token lengths around it, and compares next-token prefill logits
across three conditions:

  A. each prompt solo (batch=1, no padding)          -> baseline
  B. short prompt left-padded, still batch=1
  C. short prompt left-padded, stacked in a batch of
     2 with the long prompt

Expected: B and C stay within bf16 wobble (<~1 max |dLogit|) of A.
Observed: corrupted conditions show ~40 max |dLogit| with the argmax
flipping to a modality special token ('<audio|>') on an image prompt.
B sweeps pad lengths because the corruption appears to depend on the
pad length itself (multiples of 8 were observed benign, pad=7 corrupt),
not on batch size.

Every sequence-aligned tensor (input_ids, attention_mask, token type ids)
is padded in lockstep, position_ids follow the attention mask (same
convention generate() uses), and the padded row's suffix is asserted
byte-identical to its solo encoding -- so condition C's divergence is not a
collation artifact. A greedy generate() comparison at the end shows the
user-visible symptom (the padded row degenerates to caption-style text).

Usage:  python gemma4_pad_batch_repro.py [--model google/gemma-4-12B-it]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForMultimodalLM, AutoProcessor

SHORT_Q = "In one short sentence, what colors do you see?"
LONG_Q = ("Answer briefly: is the grid mostly empty? Explain in one "
          "sentence why you think so.")


def make_image(path: Path, size: int = 96) -> Path:
    """Deterministic grid of colored squares on light gray."""
    img = Image.new("RGB", (size, size), (235, 235, 235))
    draw = ImageDraw.Draw(img)
    cell = size // 6
    colors = [(200, 40, 40), (40, 160, 40), (220, 180, 30), (60, 60, 200)]
    for i, (gx, gy) in enumerate([(0, 1), (2, 3), (3, 0), (4, 4), (1, 5),
                                  (5, 2), (2, 0), (0, 4)]):
        draw.rectangle(
            [gx * cell, gy * cell, (gx + 1) * cell - 1, (gy + 1) * cell - 1],
            fill=colors[i % 4],
        )
    img.save(path)
    return path


def encode(processor, image_path: Path, text: str) -> dict:
    messages = [{"role": "user", "content": [
        {"type": "image", "url": str(image_path)},
        {"type": "text", "text": text},
    ]}]
    return processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )


def left_pad(enc: dict, pad_len: int, pad_token_id: int) -> dict:
    """Left-pad EVERY sequence-aligned tensor in lockstep with input_ids."""
    seq_len = enc["input_ids"].shape[1]
    out = {}
    for k, v in enc.items():
        if not isinstance(v, torch.Tensor):
            continue
        if (not v.dtype.is_floating_point and v.dim() == 2
                and v.shape[1] == seq_len):
            fill = pad_token_id if k == "input_ids" else 0
            out[k] = torch.cat([v.new_full((1, pad_len), fill), v], dim=1)
        else:
            out[k] = v
    # The padded row's suffix must be byte-identical to the solo encoding.
    assert torch.equal(out["input_ids"][0, pad_len:], enc["input_ids"][0])
    assert out["attention_mask"][0, :pad_len].sum() == 0
    return out


def last_logits(model, enc: dict) -> torch.Tensor:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs = {}
    for k, v in enc.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            inputs[k] = v.to(dtype) if v.dtype.is_floating_point else v
    # Same position_ids convention generate() uses for left-padded prefill.
    mask = inputs["attention_mask"]
    inputs["position_ids"] = (mask.long().cumsum(-1) - 1).clamp(min=0)
    with torch.inference_mode():
        return model(**inputs).logits[:, -1].float().cpu()


def top5(tok, logits: torch.Tensor) -> str:
    vals, ids = logits.topk(5)
    return "  ".join(f"{tok.decode([int(i)])!r}:{float(v):.2f}"
                     for v, i in zip(vals, ids))


def report(tok, label: str, solo: torch.Tensor, other: torch.Tensor) -> bool:
    delta = (other - solo).abs()
    flipped = int(other.argmax()) != int(solo.argmax())
    print(f"{label:<28s} max|dLogit|={float(delta.max()):8.4f} "
          f"mean={float(delta.mean()):.5f} argmax_flipped={flipped}")
    if flipped:
        print(f"    solo top5: {top5(tok, solo)}")
        print(f"    this top5: {top5(tok, other)}")
    return flipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-12B-it")
    args = ap.parse_args()

    import transformers
    print(f"transformers=={transformers.__version__}  "
          f"torch=={torch.__version__}")

    processor = AutoProcessor.from_pretrained(args.model, padding_side="left")
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype="auto", attn_implementation="sdpa",
        device_map="auto",
    ).eval()
    tok = getattr(processor, "tokenizer", processor)
    pad_id = tok.pad_token_id

    with tempfile.TemporaryDirectory() as tmp:
        img = make_image(Path(tmp) / "grid.png")
        enc_s = encode(processor, img, SHORT_Q)
        enc_l = encode(processor, img, LONG_Q)
    len_s, len_l = enc_s["input_ids"].shape[1], enc_l["input_ids"].shape[1]
    assert len_l > len_s, (len_s, len_l)
    print(f"prompt lengths: short={len_s} long={len_l} tokens")

    solo_s = last_logits(model, enc_s)[0]
    solo_l = last_logits(model, enc_l)[0]
    padded_s = left_pad(enc_s, len_l - len_s, pad_id)

    print(f"\nsolo short top5: {top5(tok, solo_s)}")
    # B: same row, alone in the batch, over a sweep of pad lengths --
    # the corruption tracks the pad length (alignment), not batch size.
    corrupted = False
    for pad_len in (1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 63, 64):
        corrupted |= report(
            tok, f"B. padded, batch=1, pad={pad_len}",
            solo_s, last_logits(model, left_pad(enc_s, pad_len, pad_id))[0],
        )

    # C: same padded row, batch of 2 with the (unpadded) long prompt.
    stacked = {k: torch.cat([padded_s[k], enc_l[k]], dim=0)
               for k in padded_s if isinstance(padded_s[k], torch.Tensor)}
    both = last_logits(model, stacked)
    corrupted |= report(tok, "C. padded, batch=2", solo_s, both[0])
    report(tok, "   unpadded control row", solo_l, both[1])

    # User-visible symptom: greedy generate, solo vs the same batch of 2.
    gen_kwargs = dict(max_new_tokens=40, do_sample=False)
    device = next(model.parameters()).device

    def to_dev(d):
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in d.items()}

    with torch.inference_mode():
        g_solo = model.generate(**to_dev(enc_s), **gen_kwargs)
        g_batch = model.generate(**to_dev(stacked), **gen_kwargs)
    solo_text = processor.decode(g_solo[0][len_s:], skip_special_tokens=True)
    batch_text = processor.decode(g_batch[0][len_l:], skip_special_tokens=True)
    print(f"\ngreedy solo   : {solo_text.strip()!r}")
    print(f"greedy batched: {batch_text.strip()!r}")

    print("\nExpected: every condition within max|dLogit| <~ 1 of solo "
          "(bf16 wobble). BUG: a padded condition shows max|dLogit| in "
          "the tens with the argmax flipped to a modality special token, "
          "and the batched greedy text diverges from solo.")
    return 1 if corrupted else 0


if __name__ == "__main__":
    sys.exit(main())
