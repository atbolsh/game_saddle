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
  3. finds the guilty call by comparing each call's CAPTURED ORIGINAL
     OUTPUT against an fp32-MATH recomputation on the same captured
     inputs (ground truth: only the corrupted call disagrees with its own
     reference), then replays that call under every SDPA backend (MATH /
     EFFICIENT / CUDNN / FLASH) to identify WHICH backend reproduces the
     corruption. Lesson from the first run of this script (2026-08-03):
     ranking EFFICIENT-vs-MATH replay disagreement found nothing, because
     the backend the model actually used was neither -- on Blackwell +
     head_dim 512 the default selector can pick the cuDNN backend, which
     the EFFICIENT/MATH comparison never exercises;
  4. saves the guilty poisoned call + the same-index control call into a
     weights_only-loadable bundle (sdpa_repro.pt) for the standalone,
     torch-only replay script (torch_sdpa_repro.py) that gets attached to
     the pytorch issue. If the corruption cannot be reproduced by ANY
     backend replay on the captured inputs (i.e. it is stateful, not a
     pure function of the inputs), the script says so LOUDLY and exits 2.

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

#: A call counts as "materially corrupted" only if its captured output
#: disagrees with the fp32-MATH reference this much in absolute terms AND
#: this many times worse than the same-index control call. bf16 backend
#: wobble is ~1e-1 on these attention outputs; the bug's logit-level
#: effect was ~40.
ABS_THRESHOLD = 0.5
REL_THRESHOLD = 50.0

#: Every selectable SDPA backend; the guilty one is whichever reproduces
#: the captured (corrupted) output. Backends that reject these inputs
#: (e.g. FLASH with an arbitrary mask / head_dim 512) are reported as
#: unsupported and skipped.
BACKENDS = [
    ("MATH", SDPBackend.MATH),
    ("EFFICIENT", SDPBackend.EFFICIENT_ATTENTION),
    ("CUDNN", SDPBackend.CUDNN_ATTENTION),
    ("FLASH", SDPBackend.FLASH_ATTENTION),
]


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


def corruption(call: dict) -> float:
    """max|captured original output - fp32-MATH recomputation on the SAME
    captured inputs|. Ground truth for 'this call went wrong in the real
    run', regardless of which backend the run's selector picked."""
    ref = replay(call, SDPBackend.MATH, fp32=True)
    return float((call["out"].float() - ref).abs().max())


def backend_table(call: dict) -> dict[str, dict[str, float] | None]:
    """Replay one call under every backend; per backend, max|d| vs the
    fp32-MATH reference and vs the captured original output (a backend
    matching the corrupted original IS the guilty backend)."""
    ref = replay(call, SDPBackend.MATH, fp32=True)
    orig = call["out"].float()
    table: dict[str, dict[str, float] | None] = {}
    for name, be in BACKENDS:
        try:
            out = replay(call, be)
        except Exception:
            table[name] = None  # backend rejects these inputs
            continue
        table[name] = {"vs_fp32_ref": float((out - ref).abs().max()),
                       "vs_original": float((out - orig).abs().max())}
    return table


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

    # ---- step 3: find the guilty call by ORIGINAL-OUTPUT corruption
    # (not by backend-vs-backend replay disagreement: the 2026-08-03 run
    # proved the guilty backend can be one the replay pair didn't cover)
    ranked = sorted(
        ((corruption(c), i) for i, c in enumerate(poison_calls)),
        reverse=True,
    )
    print("\ntop 5 corrupted calls (captured output vs fp32-MATH "
          "reference, poisoned prefill):")
    for dv, i in ranked[:5]:
        q = poison_calls[i]["args"][0]
        print(f"  call {i:3d}  max|d|={dv:10.4f}  q shape "
              f"{tuple(q.shape)} dtype {q.dtype}")

    worst_d, worst_i = ranked[0]
    control_d = corruption(control_calls[worst_i])
    print(f"\nworst call {worst_i}: poisoned max|d|={worst_d:.4f}, "
          f"same-index control max|d|={control_d:.4f}")
    if not (worst_d > ABS_THRESHOLD
            and worst_d > REL_THRESHOLD * max(control_d, 1e-6)):
        print(
            "\n*** NO CORRUPTED CALL FOUND: every captured SDPA output "
            "matches its own fp32-MATH recomputation, yet the "
            "model-level logits diverge under the default backend and "
            "not under forced MATH. That should be impossible if the "
            "corruption lives inside F.scaled_dot_product_attention on "
            "these inputs -- suspect the capture is not faithful (a "
            "different kernel symbol?) or the bug is stateful. NOT "
            "writing a bundle. ***"
        )
        return 2

    # ---- step 3b: which backend reproduces the corrupted output?
    print(f"\nper-backend replay of call {worst_i} (poisoned):")
    table = backend_table(poison_calls[worst_i])
    guilty = []
    for name, row in table.items():
        if row is None:
            print(f"  {name:<10s} unsupported on these inputs")
            continue
        print(f"  {name:<10s} vs fp32 ref: {row['vs_fp32_ref']:10.4f}   "
              f"vs captured original: {row['vs_original']:10.4f}")
        # Guilty = wrong vs the reference AND matching the corrupted
        # original run (i.e. this backend is what the run executed).
        if (row["vs_fp32_ref"] > ABS_THRESHOLD
                and row["vs_original"] < row["vs_fp32_ref"] / REL_THRESHOLD):
            guilty.append(name)
    if not guilty:
        print(
            "\n*** CORRUPTED CALL FOUND BUT NOT REPRODUCIBLE: call "
            f"{worst_i}'s captured output is wrong, yet NO backend "
            "replay on the captured inputs reproduces it. The corruption "
            "is not a pure function of the SDPA inputs (stateful kernel "
            "bug / workspace reuse?) -- a .pt bundle alone will not "
            "serve as an MRE. Report this finding on the torch issue "
            "verbatim instead. NOT writing a bundle. ***"
        )
        return 2
    print(f"\nGUILTY BACKEND(S): {guilty} -- corrupted vs the fp32 "
          "reference AND bit-matching the captured original run")

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
            "guilty_backends": guilty,
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
