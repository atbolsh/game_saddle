"""Stage 1 of the torch MRE for transformers#47651: capture the failing
SDPA call from a real Gemma 4 prefill.

Background (https://github.com/huggingface/transformers/issues/47651):
left-padded Gemma 4 prefill corrupts next-token logits whenever the padded
total length is 1 mod 32. zucchini-nlp bisected it to torch's SDPA
EFFICIENT_ATTENTION backend (eager and SDPA-MATH are clean; images
irrelevant) and found that dummy-initialized weights do NOT reproduce --
the corruption depends on the actual tensor values. A synthetic
random-tensor MRE therefore cannot fire; the MRE must ship the REAL
tensors.

This script (GPU box, needs transformers + the model):

  1. sanity-checks zucchini-nlp's backend claim at the model level
     (poisoned prefill under MATH vs default backend selection);
  2. intercepts every torch.nn.functional.scaled_dot_product_attention
     call during one poisoned prefill (total length 1 mod 32) and one
     control prefill (total length 2 mod 32), recording args on CPU;
  3. replays every captured call in isolation under EFFICIENT_ATTENTION
     vs MATH and ranks the divergence -- the poisoned run must contain at
     least one materially diverging call (if none diverges in isolation
     the composition matters and this MRE approach fails; the script says
     so LOUDLY and exits 2);
  4. saves the worst poisoned call + the same-index control call into a
     weights_only-loadable bundle (sdpa_repro.pt) for the standalone,
     torch-only replay script (torch_sdpa_repro.py) that gets attached to
     the pytorch issue.

Usage:
    python scripts/torch_sdpa_capture.py \
        [--model google/gemma-4-12B-it] [--out sdpa_repro.pt]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemma4_pad_batch_repro import encode, left_pad, make_image  # noqa: E402

#: A call counts as "materially diverging" only if EFFICIENT vs MATH
#: disagrees this much in absolute terms AND this many times worse than
#: the same-index control call. bf16 backend wobble is ~1e-2 on
#: attention outputs; the bug's logit-level effect was ~40.
ABS_THRESHOLD = 0.5
REL_THRESHOLD = 50.0


def build_inputs(model, enc: dict) -> dict:
    """Mirror gemma4_pad_batch_repro.last_logits's input prep (device,
    dtype, and generate()'s left-pad position_ids convention)."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs = {}
    for k, v in enc.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            inputs[k] = v.to(dtype) if v.dtype.is_floating_point else v
    mask = inputs["attention_mask"]
    inputs["position_ids"] = (mask.long().cumsum(-1) - 1).clamp(min=0)
    return inputs


def _to_cpu(v):
    return v.detach().to("cpu").clone() if isinstance(v, torch.Tensor) else v


def capture_calls(model, inputs: dict) -> tuple[torch.Tensor, list[dict]]:
    """One prefill with F.scaled_dot_product_attention spied on. Returns
    (last-position logits, list of captured calls in execution order)."""
    calls: list[dict] = []
    real = F.scaled_dot_product_attention

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        calls.append({
            "args": [_to_cpu(a) for a in args],
            "kwargs": {k: _to_cpu(v) for k, v in kwargs.items()},
            "out": _to_cpu(out),
        })
        return out

    F.scaled_dot_product_attention = spy
    try:
        with torch.inference_mode():
            logits = model(**inputs).logits[:, -1].float().cpu()
    finally:
        F.scaled_dot_product_attention = real
    if not calls:
        raise RuntimeError(
            "prefill ran but ZERO scaled_dot_product_attention calls were "
            "intercepted -- this transformers version reaches the kernel "
            "through a different symbol than "
            "torch.nn.functional.scaled_dot_product_attention, and the "
            "capture hook must be moved there. Do NOT trust any output "
            "of this script until that is fixed."
        )
    return logits, calls


def replay(call: dict, backend: SDPBackend, fp32: bool = False):
    """Re-run one captured call in isolation under a forced backend."""
    def prep(v):
        if not isinstance(v, torch.Tensor):
            return v
        v = v.to("cuda")
        return v.float() if (fp32 and v.dtype.is_floating_point) else v

    args = [prep(a) for a in call["args"]]
    kwargs = {k: prep(v) for k, v in call["kwargs"].items()}
    with torch.inference_mode(), sdpa_kernel([backend]):
        return F.scaled_dot_product_attention(*args, **kwargs).float().cpu()


def divergence(call: dict) -> float:
    eff = replay(call, SDPBackend.EFFICIENT_ATTENTION)
    math = replay(call, SDPBackend.MATH)
    return float((eff - math).abs().max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-12B-it")
    ap.add_argument("--out", default="sdpa_repro.pt")
    args = ap.parse_args()

    import transformers
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    print(f"transformers=={transformers.__version__}  "
          f"torch=={torch.__version__}  "
          f"gpu={torch.cuda.get_device_name(0)}")

    processor = AutoProcessor.from_pretrained(args.model,
                                              padding_side="left")
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype="auto", attn_implementation="sdpa",
        device_map="auto",
    ).eval()
    tok = getattr(processor, "tokenizer", processor)
    pad_id = tok.pad_token_id

    with tempfile.TemporaryDirectory() as tmp:
        enc = encode(processor, make_image(Path(tmp) / "grid.png"),
                     "In one short sentence, what colors do you see?")
    length = enc["input_ids"].shape[1]
    poison_pad = (1 - length) % 32 or 32     # padded total == 1 mod 32
    control_pad = poison_pad + 1             # padded total == 2 mod 32
    print(f"prompt {length} tokens; poison pad {poison_pad} "
          f"(total {length + poison_pad}), control pad {control_pad} "
          f"(total {length + control_pad})")

    # ---- step 1: model-level backend check (zucchini-nlp's finding)
    solo = build_inputs(model, enc)
    poisoned = build_inputs(model, left_pad(enc, poison_pad, pad_id))
    with torch.inference_mode():
        ref = model(**solo).logits[:, -1].float().cpu()[0]
        with sdpa_kernel([SDPBackend.MATH]):
            math_logits = model(**poisoned).logits[:, -1].float().cpu()[0]
    print(f"[model-level] poisoned prefill under FORCED MATH backend: "
          f"max|dLogit| vs solo = {float((math_logits - ref).abs().max()):.4f} "
          "(expect <~1: MATH is clean, confirming a backend bug)")

    # ---- step 2: capture both prefills
    poisoned_logits, poison_calls = capture_calls(model, poisoned)
    d = float((poisoned_logits[0] - ref).abs().max())
    print(f"[capture] poisoned prefill (default backend): max|dLogit| "
          f"vs solo = {d:.4f} (expect tens: the bug fired)")
    _, control_calls = capture_calls(
        model, build_inputs(model, left_pad(enc, control_pad, pad_id)))
    assert len(poison_calls) == len(control_calls), (
        len(poison_calls), len(control_calls))
    print(f"[capture] {len(poison_calls)} SDPA calls per prefill")

    # ---- step 3: replay each call in isolation, rank divergence
    ranked = sorted(
        ((divergence(c), i) for i, c in enumerate(poison_calls)),
        reverse=True,
    )
    print("\ntop 5 EFFICIENT-vs-MATH divergences (poisoned prefill):")
    for dv, i in ranked[:5]:
        q = poison_calls[i]["args"][0]
        print(f"  call {i:3d}  max|d|={dv:10.4f}  q shape "
              f"{tuple(q.shape)} dtype {q.dtype}")

    worst_d, worst_i = ranked[0]
    control_d = divergence(control_calls[worst_i])
    print(f"\nworst call {worst_i}: poisoned max|d|={worst_d:.4f}, "
          f"same-index control max|d|={control_d:.4f}")
    if not (worst_d > ABS_THRESHOLD
            and worst_d > REL_THRESHOLD * max(control_d, 1e-6)):
        print(
            "\n*** ISOLATION FAILED: no single SDPA call diverges "
            "materially between backends, yet the model-level logits do. "
            "The corruption only appears in composition; a single-call "
            ".pt bundle would NOT reproduce the bug and this MRE "
            "approach needs rethinking. NOT writing a bundle. ***"
        )
        return 2

    # ---- step 4: save the weights_only-loadable bundle
    def pack(call):
        return {"args": call["args"], "kwargs": call["kwargs"],
                "out_original_run": call["out"]}

    bundle = {
        "meta": {
            "source": "https://github.com/huggingface/transformers/"
                      "issues/47651",
            "model": args.model,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "call_index": worst_i,
            "n_calls_per_prefill": len(poison_calls),
            "poison_kv_len": length + poison_pad,
            "control_kv_len": length + control_pad,
            "note": "poisoned kv_len % 32 == 1; control differs by ONE "
                    "left-pad token, same layer, same prompt",
        },
        "poisoned": pack(poison_calls[worst_i]),
        "control": pack(control_calls[worst_i]),
    }
    torch.save(bundle, args.out)
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"\nwrote {args.out} ({size_mb:.1f} MB)")
    print("next: python scripts/torch_sdpa_repro.py", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
