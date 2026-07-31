"""Torch-only MRE: SDPA EFFICIENT_ATTENTION backend corrupts its output
for a specific real-world input (kv sequence length == 1 mod 32, bf16,
CUDA), while the MATH backend on the SAME tensors is correct.

Context: found via Gemma 4 batched generation
(https://github.com/huggingface/transformers/issues/47651). The
corruption is value-dependent -- randomly initialized tensors of the same
shapes do NOT reproduce -- so this script replays the EXACT q/k/v/mask
tensors captured from the failing model forward (sdpa_repro.pt, attached
to the issue; produced by torch_sdpa_capture.py in the same repo).

The bundle holds two captured calls from the SAME layer of the SAME
prompt, differing only by ONE extra left-pad token:

  * "poisoned": kv_len % 32 == 1  -> EFFICIENT vs MATH diverge massively
  * "control":  kv_len % 32 == 2  -> all backends agree to bf16 wobble

Only torch is imported. Loads with weights_only=True. Exit 1 = bug fired.

Usage:  python torch_sdpa_repro.py sdpa_repro.pt
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


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
        eff = run(case, SDPBackend.EFFICIENT_ATTENTION)
        math_bf16 = run(case, SDPBackend.MATH)
        math_fp32 = run(case, SDPBackend.MATH, fp32=True)
        d_backends = float((eff - math_bf16).abs().max())
        d_eff_ref = float((eff - math_fp32).abs().max())
        d_math_ref = float((math_bf16 - math_fp32).abs().max())
        print(f"[{name}] q {tuple(q.shape)} kv_len {k.shape[-2]} "
              f"(% 32 = {k.shape[-2] % 32}) dtype {q.dtype}")
        print(f"  max|EFFICIENT - MATH(bf16)|  = {d_backends:10.4f}")
        print(f"  max|EFFICIENT - MATH(fp32)|  = {d_eff_ref:10.4f}")
        print(f"  max|MATH(bf16) - MATH(fp32)| = {d_math_ref:10.4f}"
              "   <- honest bf16 wobble for these tensors")
        if name == "poisoned" and d_backends > 10 * max(d_math_ref, 1e-6):
            fired = True

    print("\nExpected: EFFICIENT_ATTENTION matches MATH to bf16 wobble on "
          "both cases. BUG: on the poisoned case (kv_len % 32 == 1) "
          "EFFICIENT_ATTENTION diverges orders of magnitude past wobble, "
          "while the control (ONE extra kv position, otherwise the same "
          "tensors) is clean.")
    return 1 if fired else 0


if __name__ == "__main__":
    sys.exit(main())
