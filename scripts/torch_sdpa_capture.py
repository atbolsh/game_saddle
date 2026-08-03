"""Stage 1 of the torch MRE for transformers#47651: capture the failing
SDPA call from a real Gemma 4 prefill, WITH ITS EXACT MEMORY LAYOUT.

Background (https://github.com/huggingface/transformers/issues/47651):
left-padded Gemma 4 prefill corrupts next-token logits whenever the padded
total length is 1 mod 32. Maintainer triage bisected it to a non-MATH SDPA
backend (forcing MATH fixes it) and found dummy weights do NOT reproduce.

Lessons from two failed capture attempts on 2026-08-03, baked in here:

  * v1 ranked calls by EFFICIENT-vs-MATH replay disagreement -> nothing.
  * v2 ranked by captured-output-vs-fp32 reference -> found corrupted
    calls (max|d| ~ 15-19, controls ~0.09) BUT no backend replay of the
    CPU-round-tripped tensors reproduced the corruption. Meanwhile the
    replay warnings proved cuDNN (head_dim > 128) and FLASH (non-null
    attn_mask) are ineligible for these inputs, so the original run can
    only have used EFFICIENT -- whose replay on the same VALUES is clean.
    Conclusion: the bug depends on the tensors' MEMORY LAYOUT (storage
    offsets, strides, buffers shared between k/v views), which
    ``.to("cpu").clone()`` silently normalizes.

So this version:

  1. sanity-checks the backend claim at model level (poisoned prefill
     under forced MATH must be clean);
  2. spies on every torch.nn.functional.scaled_dot_product_attention call
     during a poisoned prefill (total length 1 mod 32) and a control
     prefill (one pad token longer). The spy computes ``d_live`` -- the
     call's real output vs a forced-MATH recompute on the LIVE, in-place
     GPU tensors -- which is the round-trip-free guilty-call detector.
     It also packs each argument's ENTIRE underlying storage as raw
     bytes (shared storages saved once, preserving k/v adjacency) plus
     (shape, stride, storage_offset), so the replay can rebuild
     bit-identical views via ``Tensor.set_``;
  3. replays the guiltiest call from the packed storages under every
     backend, printing each backend's delta vs an fp32-MATH reference
     and vs the captured (corrupted) original output;
  4. writes sdpa_repro.pt (weights_only-loadable) for the standalone
     torch_sdpa_repro.py IF a backend replay reproduces the corruption;
     otherwise explains LOUDLY what was and wasn't established (the
     d_live numbers alone are already reportable evidence) and exits 2.

Usage:
    python scripts/torch_sdpa_capture.py \
        [--model google/gemma-4-12B-it] [--out sdpa_repro.pt]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemma4_pad_batch_repro import encode, left_pad, make_image  # noqa: E402

# Repo-root .env -> os.environ (HF_TOKEN for the model download; see the
# env-secrets .cursor rule). python-dotenv is in requirements.txt.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

#: A call counts as corrupted only if its live output disagrees with the
#: live forced-MATH recompute this much in absolute terms AND this many
#: times worse than the same-index control call. bf16 backend wobble on
#: these attention outputs is ~1e-1; the observed corruption is ~15-20.
ABS_THRESHOLD = 0.5
REL_THRESHOLD = 50.0

BACKENDS = [
    ("MATH", SDPBackend.MATH),
    ("EFFICIENT", SDPBackend.EFFICIENT_ATTENTION),
    ("CUDNN", SDPBackend.CUDNN_ATTENTION),
    ("FLASH", SDPBackend.FLASH_ATTENTION),
]


# ==================================================== layout-exact packing

def pack_call(args: tuple, kwargs: dict) -> dict:
    """Pack SDPA call arguments preserving EXACT memory layout.

    Each distinct underlying storage is dumped ONCE as raw bytes (so
    tensors that are views into the same buffer -- e.g. k and v sliced
    from one fused projection -- stay adjacent on rebuild), and every
    tensor is recorded as (storage id, dtype, shape, stride, offset).
    Everything is weights_only-serializable.
    """
    storages: dict[str, torch.Tensor] = {}

    def pack(v):
        if not isinstance(v, torch.Tensor):
            return v
        stor = v.untyped_storage()
        sid = str(stor.data_ptr())
        if sid not in storages:
            u8 = torch.empty(0, dtype=torch.uint8, device=v.device)
            u8.set_(stor)  # 1-D uint8 view of the WHOLE storage
            storages[sid] = u8.cpu().clone()
        return {
            "__packed__": True,
            "sid": sid,
            "dtype": str(v.dtype).split(".")[-1],
            "shape": tuple(v.shape),
            "stride": tuple(v.stride()),
            "offset": v.storage_offset(),
        }

    return {
        "storages": storages,
        "args": [pack(a) for a in args],
        "kwargs": {k: pack(v) for k, v in kwargs.items()},
    }


def unpack_call(packed: dict, device: str = "cuda") -> tuple[list, dict]:
    """Rebuild the call's tensors with their original layout (offsets,
    strides, shared storages) on ``device``."""
    live = {sid: u8.to(device) for sid, u8 in packed["storages"].items()}

    def unpack(p):
        if not (isinstance(p, dict) and p.get("__packed__")):
            return p
        t = torch.empty(0, dtype=getattr(torch, p["dtype"]), device=device)
        t.set_(live[p["sid"]].untyped_storage(),
               p["offset"], p["shape"], p["stride"])
        return t

    return ([unpack(a) for a in packed["args"]],
            {k: unpack(v) for k, v in packed["kwargs"].items()})


def layout_lines(packed: dict) -> list[str]:
    out = []
    names = [f"arg{i}" for i in range(len(packed["args"]))]
    items = list(zip(names, packed["args"]))
    items += [(k, v) for k, v in packed["kwargs"].items()]
    for name, p in items:
        if isinstance(p, dict) and p.get("__packed__"):
            contig = torch.empty(p["shape"]).stride() == tuple(p["stride"])
            out.append(
                f"  {name:<10s} {p['dtype']:<9s} shape {p['shape']} "
                f"stride {p['stride']} storage_offset {p['offset']} "
                f"contiguous={bool(contig)} "
                f"storage_bytes={packed['storages'][p['sid']].numel()}"
            )
        else:
            out.append(f"  {name:<10s} = {p!r}")
    return out


# ========================================================== capture + replay

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


def capture_calls(model, inputs: dict) -> tuple[torch.Tensor, list[dict]]:
    """One prefill with F.scaled_dot_product_attention spied on.

    Per call the spy records: the packed (layout-exact) arguments, the
    original output, and ``d_live`` = max|original output - forced-MATH
    recompute on the SAME live tensors| -- computed before anything is
    moved or copied, so it cannot be fooled by layout normalization.
    """
    calls: list[dict] = []
    real = F.scaled_dot_product_attention

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        with sdpa_kernel([SDPBackend.MATH]):
            out_math = real(*args, **kwargs)
        calls.append({
            "packed": pack_call(args, kwargs),
            "out": out.detach().cpu().clone(),
            "d_live": float((out - out_math).abs().max()),
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
            "through a different symbol; move the capture hook there. Do "
            "NOT trust any output of this script until that is fixed."
        )
    return logits, calls


def replay(packed: dict, backend: SDPBackend, fp32: bool = False):
    """Re-run one packed call in isolation under a forced backend, from
    layout-faithful rebuilt tensors."""
    args, kwargs = unpack_call(packed)
    if fp32:
        args = [a.float() if isinstance(a, torch.Tensor)
                and a.dtype.is_floating_point else a for a in args]
        kwargs = {k: (v.float() if isinstance(v, torch.Tensor)
                      and v.dtype.is_floating_point else v)
                  for k, v in kwargs.items()}
    with torch.inference_mode(), sdpa_kernel([backend]), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return F.scaled_dot_product_attention(*args, **kwargs).float().cpu()


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

    # ---- step 1: model-level backend check
    solo = build_inputs(model, enc)
    poisoned = build_inputs(model, left_pad(enc, poison_pad, pad_id))
    with torch.inference_mode():
        ref = model(**solo).logits[:, -1].float().cpu()[0]
        with sdpa_kernel([SDPBackend.MATH]):
            math_logits = model(**poisoned).logits[:, -1].float().cpu()[0]
    print(f"[model-level] poisoned prefill under FORCED MATH backend: "
          f"max|dLogit| vs solo = {float((math_logits - ref).abs().max()):.4f} "
          "(expect <~1: MATH is clean, confirming a backend bug)")

    # ---- step 2: capture both prefills (d_live computed in-flight)
    poisoned_logits, poison_calls = capture_calls(model, poisoned)
    d = float((poisoned_logits[0] - ref).abs().max())
    print(f"[capture] poisoned prefill (default backend): max|dLogit| "
          f"vs solo = {d:.4f} (expect tens: the bug fired)")
    _, control_calls = capture_calls(
        model, build_inputs(model, left_pad(enc, control_pad, pad_id)))
    assert len(poison_calls) == len(control_calls), (
        len(poison_calls), len(control_calls))
    print(f"[capture] {len(poison_calls)} SDPA calls per prefill")

    # ---- step 3: rank by d_live (live output vs live forced-MATH; no
    # round trip involved, so layout games cannot hide the guilty call)
    ranked = sorted(((c["d_live"], i) for i, c in enumerate(poison_calls)),
                    reverse=True)
    print("\ntop 5 live divergences (default backend vs forced MATH on "
          "the LIVE tensors, poisoned prefill):")
    for dv, i in ranked[:5]:
        print(f"  call {i:3d}  d_live={dv:10.4f}")
    worst_d, worst_i = ranked[0]
    control_d = control_calls[worst_i]["d_live"]
    print(f"\nworst call {worst_i}: poisoned d_live={worst_d:.4f}, "
          f"same-index control d_live={control_d:.4f}")
    if not (worst_d > ABS_THRESHOLD
            and worst_d > REL_THRESHOLD * max(control_d, 1e-6)):
        print(
            "\n*** NO GUILTY CALL EVEN ON LIVE TENSORS: the default and "
            "forced-MATH backends agree inside every call, yet the "
            "model-level logits diverge. The corruption then happens "
            "OUTSIDE F.scaled_dot_product_attention (or forcing MATH "
            "changes something else entirely). NOT writing a bundle; "
            "bring this output back for a rethink. ***"
        )
        return 2

    print(f"\nmemory layout of call {worst_i}'s arguments (the part a "
          "CPU round trip would normalize):")
    for line in layout_lines(poison_calls[worst_i]["packed"]):
        print(line)

    # ---- step 4: layout-faithful replay under every backend
    packed = poison_calls[worst_i]["packed"]
    fp32_ref = replay(packed, SDPBackend.MATH, fp32=True)
    orig = poison_calls[worst_i]["out"].float()
    print(f"\nper-backend replay of call {worst_i} from packed storages "
          "(layout-exact):")
    guilty = []
    for name, be in BACKENDS:
        try:
            out = replay(packed, be)
        except Exception:
            print(f"  {name:<10s} unsupported on these inputs")
            continue
        d_ref = float((out - fp32_ref).abs().max())
        d_orig = float((out - orig).abs().max())
        print(f"  {name:<10s} vs fp32 ref: {d_ref:10.4f}   "
              f"vs captured original: {d_orig:10.4f}")
        if d_ref > ABS_THRESHOLD and d_orig < d_ref / REL_THRESHOLD:
            guilty.append(name)
    if not guilty:
        print(
            "\n*** STILL NOT REPRODUCIBLE IN ISOLATION: the guilty call "
            f"is real (d_live={worst_d:.2f} on live tensors, control "
            f"{control_d:.2f}) but even a layout-exact single-call "
            "replay is clean. Remaining suspects: allocator state / "
            "adjacent-memory contents (an out-of-bounds read would pick "
            "up different neighbors in replay) or some stream/workspace "
            "state. The d_live evidence + the layout dump above are "
            "still reportable: torch devs can be told the corruption "
            "fires in-place during the model forward but not on rebuilt "
            "tensors. NOT writing a bundle. ***"
        )
        return 2
    print(f"\nGUILTY BACKEND(S): {guilty} -- corrupted vs the fp32 "
          "reference AND matching the captured original run")

    # ---- step 5: save the weights_only-loadable bundle
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
                    "left-pad token, same layer, same prompt. Tensors "
                    "are packed as raw storages + (shape, stride, "
                    "offset) because the bug is layout-sensitive.",
        },
        "poisoned": {"packed": packed,
                     "out_original_run": poison_calls[worst_i]["out"]},
        "control": {"packed": control_calls[worst_i]["packed"],
                    "out_original_run": control_calls[worst_i]["out"]},
    }
    torch.save(bundle, args.out)
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"\nwrote {args.out} ({size_mb:.1f} MB)")
    print("next: python scripts/torch_sdpa_repro.py", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
