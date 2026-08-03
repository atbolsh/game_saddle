"""Torch-only MRE: an SDPA backend corrupts its output for a specific
real-world input (kv sequence length == 1 mod 32, bf16, CUDA), while the
MATH backend on the SAME tensors is correct. The guilty backend on the
reporting machine is recorded in the bundle's ``meta["guilty_backends"]``;
this script replays ALL backends so the result is meaningful anywhere.

Context: found via Gemma 4 batched generation
(https://github.com/huggingface/transformers/issues/47651). Two things
make this repro unusual, both established during triage:

  * the corruption is VALUE-dependent -- randomly initialized tensors of
    the same shapes do not reproduce -- so the bundle ships the exact
    tensors captured from the failing model forward;
  * it is also LAYOUT-dependent -- a plain ``.cpu().clone()`` round trip
    of the same values replays clean -- so the bundle ships each
    argument's ENTIRE underlying storage as raw bytes plus its
    (shape, stride, storage_offset), and this script rebuilds the exact
    views with ``Tensor.set_`` (storages shared between arguments, e.g.
    k/v views of one buffer, are restored shared).

The bundle holds two captured calls from the SAME layer of the SAME
prompt, differing only by ONE extra left-pad token:

  * "poisoned": kv_len % 32 == 1  -> the guilty backend diverges
    massively from the fp32-MATH reference
  * "control":  kv_len % 32 == 2  -> every backend agrees to bf16 wobble

Only torch is imported. Loads with weights_only=True. Exit 1 = bug fired.

Usage:  python torch_sdpa_repro.py sdpa_repro.pt
"""

from __future__ import annotations

import sys
import warnings

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

BACKENDS = [
    ("MATH_bf16", SDPBackend.MATH),
    ("EFFICIENT", SDPBackend.EFFICIENT_ATTENTION),
    ("CUDNN", SDPBackend.CUDNN_ATTENTION),
    ("FLASH", SDPBackend.FLASH_ATTENTION),
]


def unpack_call(packed: dict, device: str = "cuda") -> tuple[list, dict]:
    """Rebuild the call's tensors with their original memory layout."""
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


def run(packed: dict, backend: SDPBackend, fp32: bool = False):
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
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    print(f"torch=={torch.__version__}  "
          f"cuda={torch.version.cuda}  gpu={torch.cuda.get_device_name(0)}")
    bundle = torch.load(sys.argv[1], weights_only=True)
    print("bundle meta:", bundle["meta"], "\n")

    fired = False
    for name in ("poisoned", "control"):
        packed = bundle[name]["packed"]
        orig = bundle[name]["out_original_run"].float()
        q_meta = packed["args"][0]
        kv_len = packed["args"][1]["shape"][-2]
        ref = run(packed, SDPBackend.MATH, fp32=True)  # fp32 ground truth
        d_orig = float((orig - ref).abs().max())
        print(f"[{name}] q shape {q_meta['shape']} kv_len {kv_len} "
              f"(% 32 = {kv_len % 32}) dtype {q_meta['dtype']}")
        print(f"  captured original run vs fp32 ref: {d_orig:10.4f}")
        for bname, be in BACKENDS:
            try:
                out = run(packed, be)
            except Exception as exc:
                print(f"  {bname:<10s} unsupported on these inputs "
                      f"({type(exc).__name__})")
                continue
            d = float((out - ref).abs().max())
            print(f"  {bname:<10s} vs fp32 ref: {d:10.4f}")
            # MATH_bf16's delta is honest bf16 wobble; a backend an order
            # of magnitude past 0.5 on the poisoned case is the bug.
            if name == "poisoned" and bname != "MATH_bf16" and d > 0.5:
                fired = True
        print()

    print("Expected: every backend matches the fp32-MATH reference to "
          "bf16 wobble (the MATH_bf16 row) on both cases. BUG: on the "
          "poisoned case (kv_len % 32 == 1) at least one backend "
          "diverges orders of magnitude past wobble -- matching the "
          "'captured original run' row, which is what the full model "
          "forward produced -- while the control (ONE extra kv position, "
          "otherwise the same tensors) is clean everywhere.")
    return 1 if fired else 0


if __name__ == "__main__":
    sys.exit(main())
