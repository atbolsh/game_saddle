"""Diagnose the Gemma 4 left-pad divergence: precision noise or mask bug?

Background (see TO_TEST.md stage-6 note): left-padded variable-length
multimodal decode diverges from solo generate even when every collated
tensor is suffix-identical to the solo encoding, so ``generate_batch`` only
true-batches equal-length rows. The open question is WHY the padded path
diverges. Two competing theories with different fingerprints:

  * bf16/SDPA kernel noise -- next-token logits differ by ~1e-2, deltas do
    NOT grow with pad length, argmax flips (if any) land on near-tie
    runner-up tokens.
  * a real attention-mask / position defect in the padded multimodal
    prefill -- deltas are large (order of the logit scale), grow or persist
    with pad length, and the top tokens shift systematically (observed
    t6 failures flipped to caption-style 'a ...' openings on BOTH padded
    rows, which uncorrelated noise cannot explain).

One prompt, one prefill forward pass per condition: solo, then the same row
left-padded by 8 / 64 / 256 positions inside a batch-1 tensor (padding with
the tokenizer's pad token; attention_mask zeros; every sequence-aligned int
tensor padded in lockstep, exactly like the removed generate-path collate).
Also runs a text-only prompt to isolate the vision-block mask. Compares the
last-position logits.

Run on the REMOTE box:  python -m training.probe_pad_divergence
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _left_pad_row(enc: dict[str, Any], pad_len: int,
                  pad_token_id: int) -> dict[str, Any]:
    """Left-pad one batch-1 encoding by pad_len positions (probe-only)."""
    import torch

    seq_len = enc["input_ids"].shape[1]
    out: dict[str, Any] = {}
    for k, v in enc.items():
        if not isinstance(v, torch.Tensor):
            continue
        if (not v.dtype.is_floating_point and v.dim() == 2
                and v.shape[1] == seq_len):
            fill = pad_token_id if k == "input_ids" else 0
            pad = v.new_full((1, pad_len), fill)
            out[k] = torch.cat([pad, v], dim=1)
        else:
            out[k] = v
    assert out["attention_mask"][0, :pad_len].sum() == 0
    assert (out["input_ids"][0, pad_len:] == enc["input_ids"][0]).all()
    return out


def _last_logits(model: Any, enc: dict[str, Any]) -> Any:
    import torch

    inputs = model._move_inputs_to_model(enc)
    # Mirror generate's prefill: position_ids from the attention mask (pad
    # positions clamped to 0, real tokens 0..L-1). A bare forward would
    # default to arange over the padded length instead.
    mask = inputs["attention_mask"]
    inputs["position_ids"] = (mask.long().cumsum(-1) - 1).clamp(min=0)
    with torch.inference_mode():
        out = model.model(**inputs)
    return out.logits[:, -1].float().cpu()


def _describe(tok: Any, logits: Any, k: int = 5) -> str:
    vals, ids = logits.topk(k)
    parts = [
        f"{tok.decode([int(i)])!r}:{float(v):.3f}"
        for v, i in zip(vals, ids)
    ]
    return "  ".join(parts)


def run_probe() -> int:
    from agent.model import get_model
    from training.selftest import _tiny_png

    model = get_model()
    tok = getattr(model.processor, "tokenizer", model.processor)
    pad_id = tok.pad_token_id
    assert pad_id is not None, "tokenizer has no pad token"

    with tempfile.TemporaryDirectory(prefix="pad_probe_") as tmp:
        img = _tiny_png(Path(tmp) / "board.png", seed=9)
        cases = {
            "image+text": [{"role": "user", "content": [
                {"type": "image", "url": str(img)},
                {"type": "text",
                 "text": "In one short sentence, what colors do you see?"},
            ]}],
            "text-only": [{"role": "user", "content": [
                {"type": "text",
                 "text": "Reply with a single word: what color is the sky "
                         "on a clear day?"},
            ]}],
        }
        suspicious = False
        for name, messages in cases.items():
            enc = model.encode_messages(messages)
            solo = _last_logits(model, enc)[0]
            print(f"\n=== {name} (prompt {enc['input_ids'].shape[1]} tok) ===")
            print(f"solo   top5: {_describe(tok, solo)}")
            for pad_len in (8, 64, 256):
                padded = _last_logits(
                    model, _left_pad_row(enc, pad_len, pad_id)
                )[0]
                delta = (padded - solo).abs()
                flipped = int(padded.argmax()) != int(solo.argmax())
                print(
                    f"pad={pad_len:<4d} max|dLogit|={float(delta.max()):.4f} "
                    f"mean={float(delta.mean()):.5f} "
                    f"argmax_flipped={flipped}"
                )
                if flipped or float(delta.max()) > 0.5:
                    suspicious = True
                    print(f"         top5: {_describe(tok, padded)}")

        # ---- the t6 configuration: padded row inside a REAL batch of 2.
        # Batch-1 padding above showed only bounded wobble; if THIS row
        # diverges hard (t6 flipped its first token to 'a', far below the
        # solo top-5), the defect is in the batched multimodal path
        # (e.g. image-feature scatter with left-padded rows), not padding
        # per se.
        import torch

        q_short = "In one short sentence, what colors do you see?"
        q_long = ("Answer briefly: is the grid mostly empty? Explain in "
                  "one sentence why you think so.")
        encs = {}
        for label, q in (("short", q_short), ("long", q_long)):
            encs[label] = model.encode_messages(
                [{"role": "user", "content": [
                    {"type": "image", "url": str(img)},
                    {"type": "text", "text": q},
                ]}]
            )
        len_s = encs["short"]["input_ids"].shape[1]
        len_l = encs["long"]["input_ids"].shape[1]
        assert len_l > len_s, (len_s, len_l)
        solo_s = _last_logits(model, encs["short"])[0]
        solo_l = _last_logits(model, encs["long"])[0]
        padded_s = _left_pad_row(encs["short"], len_l - len_s, pad_id)
        stacked = {
            k: torch.cat([padded_s[k], encs["long"][k]], dim=0)
            for k in padded_s
            if isinstance(padded_s[k], torch.Tensor)
        }
        both = _last_logits(model, stacked)
        print(f"\n=== batch of 2 (short {len_s} tok left-padded to {len_l}, "
              "long unpadded) ===")
        for label, solo, row in (("short/padded", solo_s, both[0]),
                                 ("long/control", solo_l, both[1])):
            delta = (row - solo).abs()
            flipped = int(row.argmax()) != int(solo.argmax())
            print(
                f"{label:<13s} max|dLogit|={float(delta.max()):.4f} "
                f"mean={float(delta.mean()):.5f} argmax_flipped={flipped}"
            )
            if flipped or float(delta.max()) > 0.5:
                suspicious = True
                print(f"    solo  top5: {_describe(tok, solo)}")
                print(f"    batch top5: {_describe(tok, row)}")

    print(
        "\nVERDICT GUIDE: max|dLogit| ~1e-2 flat across pad lengths and no "
        "argmax flips => bf16 kernel noise. Deltas >~0.5, growing with pad "
        "length, or flips toward caption-style tokens ('a', 'an') => real "
        "mask/position defect. If batch-1 padding is tame but the "
        "batch-of-2 padded row diverges hard, the bug is in the batched "
        "multimodal path (image-feature scatter / mask across rows) -- "
        "report upstream; equal-length-only batching stays mandatory "
        "either way."
    )
    return 1 if suspicious else 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run_probe()


if __name__ == "__main__":
    sys.exit(main())
