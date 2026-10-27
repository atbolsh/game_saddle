"""Random-tensor check for the mem-efficient SDPA residue bug.

transformers#47651 / pytorch#191984: with 16 query heads, 1 KV head, and
head_dim 512, ``sdpa_attention_forward`` returns a wrong last position
when ``L % 32 == 1`` and K/V stay a stride-0 ``repeat_kv`` view. Torch
>= 2.14 fixes that kernel. This script does not load Gemma weights.

Exit 1 if CUDA is missing, or if any ``L % 32 == 1`` row's last-position
error against an fp64 reference exceeds 0.1.
"""

from __future__ import annotations

import sys


def main() -> int:
    import torch
    import transformers
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    if not torch.cuda.is_available():
        print(
            "CUDA is required; this check does not run on CPU",
            file=sys.stderr,
        )
        return 1

    print(
        f"transformers {transformers.__version__}, torch {torch.__version__}, "
        f"{torch.cuda.get_device_name()}"
    )
    torch.manual_seed(0)
    query_heads, head_dim = 16, 512

    class GlobalAttn(torch.nn.Module):
        num_key_value_groups = query_heads
        is_causal = True

    lengths = (32, 33, 34, 64, 65, 66, 96, 97, 98)
    print(
        "dtype      L  L%32 | sdpa_attention_forward vs fp64 reference: "
        "last position  other positions"
    )
    failures = 0
    for dtype in (torch.float32, torch.bfloat16):
        for length in lengths:
            query = torch.randn(
                1, query_heads, length, head_dim, device="cuda", dtype=dtype,
            )
            key = torch.randn(
                1, 1, length, head_dim, device="cuda", dtype=dtype,
            )
            value = torch.randn(
                1, 1, length, head_dim, device="cuda", dtype=dtype,
            )
            mask = torch.ones(
                length, length, dtype=torch.bool, device="cuda",
            ).tril()[None, None]
            out, _ = sdpa_attention_forward(
                GlobalAttn(), query, key, value, mask,
            )
            scores = (
                query.double() @ key.double().transpose(-1, -2)
            ) * head_dim ** -0.5
            ref = (
                scores.masked_fill(~mask, float("-inf")).softmax(-1)
                @ value.double()
            ).transpose(1, 2)
            delta = (out.double() - ref).abs()
            last = delta[:, -1].max().item()
            rest = delta[:, :-1].max().item()
            wrong = length % 32 == 1 and last > 0.1
            if wrong:
                failures += 1
            print(
                f"{str(dtype)[6:]:9s} {length:3d} {length % 32:4d} | "
                f"{last:52.2e} {rest:16.2e}"
                f"{'  <-- WRONG' if wrong else ''}"
            )
    if failures:
        print(
            f"FAIL: {failures} length(s) with L % 32 == 1 exceeded 0.1 "
            "at the last position. This GPU's SDPA kernel is still wrong "
            "at that width.",
            file=sys.stderr,
        )
        return 1
    print("CLEAN: every L % 32 == 1 row stayed within 0.1 of the fp64 reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
