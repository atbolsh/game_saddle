"""Torch-only MRE: an SDPA backend corrupts its output for a specific
real-world input (kv sequence length == 1 mod 32, bf16, CUDA), while the
MATH backend on the SAME tensors is correct. The guilty backend on the
reporting machine is recorded in the bundle's ``meta["guilty_backends"]``;
this script replays ALL backends so the result is meaningful on any
hardware.

Context: found via Gemma 4 batched generation
(https://github.com/huggingface/transformers/issues/47651). The
corruption is value-dependent -- randomly initialized tensors of the same
shapes do NOT reproduce -- so this script replays the EXACT q/k/v/mask
tensors captured from the failing model forward (sdpa_repro.pt, attached
to the issue; produced by torch_sdpa_capture.py in the same repo).

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

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

BACKENDS = [
    ("MATH_bf16", SDPBackend.MATH),
    ("EFFICIENT", SDPBackend.EFFICIENT_ATTENTION),
    ("CUDNN", SDPBackend.CUDNN_ATTENTION),
    ("FLASH", SDPBackend.FLASH_ATTENTION),
]


def run(case: dict, backend: SDPBackend, fp32: bool = False):
    def prep(v):
        if not isinstance(v, torch.Tensor):
            return v
        v = v.to("cuda")
        return v.float() if (fp32 and v.dtype.is_floating_point) else v

    args = [prep(a) for a in case["args"]]
    kwargs = {k: prep(v) for k, v in case["kwargs"].items()}
    with torch.inference_mode(), sdpa_kernel([backend]):
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
        case = bundle[name]
        q, k = case["args"][0], case["args"][1]
        ref = run(case, SDPBackend.MATH, fp32=True)  # fp32 ground truth
        print(f"[{name}] q {tuple(q.shape)} kv_len {k.shape[-2]} "
              f"(% 32 = {k.shape[-2] % 32}) dtype {q.dtype} -- "
              "max|backend output - MATH(fp32) reference|:")
        for bname, be in BACKENDS:
            try:
                d = float((run(case, be) - ref).abs().max())
            except Exception as exc:
                print(f"  {bname:<10s} unsupported on these inputs "
                      f"({type(exc).__name__})")
                continue
            print(f"  {bname:<10s} {d:10.4f}")
            # MATH_bf16's delta is honest bf16 wobble; a backend an order
            # of magnitude past ABS 0.5 on the poisoned case is the bug.
            if name == "poisoned" and bname != "MATH_bf16" and d > 0.5:
                fired = True
        print()

    print("Expected: every backend matches the fp32-MATH reference to "
          "bf16 wobble (the MATH_bf16 row) on both cases. BUG: on the "
          "poisoned case (kv_len % 32 == 1) at least one backend "
          "diverges orders of magnitude past wobble, while the control "
          "(ONE extra kv position, otherwise the same tensors) is clean "
          "everywhere.")
    return 1 if fired else 0


if __name__ == "__main__":
    sys.exit(main())
