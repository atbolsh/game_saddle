# DRAFT: pytorch/pytorch issue (SDPA mem-efficient kernel, kv_len % 32 == 1)

Status: **fill the two `<<< >>>` blocks from a remote run, then file.**

- Where to file: <https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml>
  (the "Bug report" form; sections below map onto its fields)
- Suggested title:
  `SDPA EFFICIENT_ATTENTION returns wrong output when kv sequence length % 32 == 1 (bf16 + attn_mask, CUDA Blackwell; MATH backend correct on identical tensors)`
- Attach: `sdpa_repro.zip` containing `sdpa_repro.pt` + `torch_sdpa_repro.py`
  (GitHub takes zips up to 25 MB; the bundle is two SDPA calls' worth of
  bf16 tensors, well under that)

---

## 🐛 Describe the bug

`F.scaled_dot_product_attention` with the **EFFICIENT_ATTENTION** backend
returns badly corrupted output for a specific real-world input whose **kv
sequence length is ≡ 1 mod 32** (bf16, CUDA, batch 1, with a 4-D
`attn_mask`). The **MATH** backend on the *identical* tensors is correct
(matches an fp32 reference to bf16 wobble), and the *same* tensors with
**one extra kv position** (kv_len ≡ 2 mod 32) are clean on every backend.

Found via Gemma 4 batched generation in transformers
(huggingface/transformers#47651): left-padding a prompt so the padded
total hits 1 mod 32 flips the next-token argmax to an unrelated special
token (logit deltas ~40). Maintainer triage there ruled transformers out:
eager attention and SDPA-with-MATH are clean, the presence of images is
irrelevant, and — important for reproduction — **the bug does not fire
with randomly initialized weights of the same architecture**. The
corruption is value-dependent, so this MRE ships the exact q/k/v/mask
tensors captured from the failing forward instead of synthetic ones.

Repro (only `torch` imported; the `.pt` loads with `weights_only=True`):

```bash
unzip sdpa_repro.zip
python torch_sdpa_repro.py sdpa_repro.pt
```

The script replays both captured calls under EFFICIENT_ATTENTION,
MATH (bf16), and MATH (fp32 reference) and prints max abs differences.
Output on the reporting machine:

```
<<< PASTE torch_sdpa_repro.py OUTPUT HERE >>>
```

The two calls come from the same attention layer of the same prompt and
differ only by one left-pad token, so the control isolates the kv_len
alignment as the trigger. The capture harness (how the tensors were
extracted from the model forward, plus a sweep showing the 1-mod-32
pattern across many lengths) is in the transformers issue linked above.

### Expected behavior

EFFICIENT_ATTENTION matches MATH to numerical wobble at every sequence
length, as it already does for the control case and for every other
length in the sweeps on the transformers side (the pattern held across
prompt lengths ~280–2900: corruption iff total kv_len % 32 == 1).

## Versions

```
<<< PASTE `python -m torch.utils.collect_env` OUTPUT HERE >>>
```

Observed on: torch 2.12.0+cu130, NVIDIA RTX PRO 6000 Blackwell
Workstation Edition, bf16, single GPU. Not yet tried on non-Blackwell
hardware (the transformers maintainer who reproduced the pattern can
confirm their hardware in the linked issue).

cc @drisspg (SDPA / attention)

---

# DRAFT: follow-up comment for huggingface/transformers#47651

> Filed the torch-side MRE: pytorch/pytorch#NNNNN
>
> Since dummy weights don't reproduce (the corruption is
> value-dependent), the MRE captures the exact q/k/v/attn_mask tensors
> from the worst-diverging SDPA call of the failing prefill (kv_len 289 =
> 1 mod 32) plus the same-layer call from a control prefill one pad token
> longer (kv_len 290), and replays them through bare
> `F.scaled_dot_product_attention` — no transformers code involved.
> EFFICIENT_ATTENTION vs MATH diverges by ~`<max|d|>` on the poisoned
> length and stays at bf16 wobble (~`<control>`) on the control. Tensor
> bundle + replay script are attached on the torch issue.
>
> @chavalasantosh in light of @zucchini-nlp's triage (kernel bug, not
> mask construction), please don't ship a transformers-side mask change
> for this; if a temporary mitigation is wanted at all, the tightly-
> scoped kind you described (affected backend/dtype only, warn once,
> removable) is the right shape.
