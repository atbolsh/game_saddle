# DRAFT: pytorch/pytorch issue (SDPA mem-efficient kernel, kv_len % 32 == 1)

Status: **READY TO FILE** (both output blocks filled from the 2026-08-03
remote run; fix the issue number in the follow-up comment after filing).

- Where to file: <https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml>
  (the "Bug report" form; sections below map onto its fields)
- Suggested title:
  `SDPA EFFICIENT_ATTENTION returns wrong output for broadcasted (stride-0) k/v when kv length % 32 == 1 (bf16 + attn_mask, CUDA; MATH correct on identical tensors)`
- Attach: `sdpa_repro.zip` containing `sdpa_repro.pt` + `torch_sdpa_repro.py`
  (GitHub takes zips up to 25 MB; the bundle is two SDPA calls' worth of
  bf16 tensors, well under that)

---

## 🐛 Describe the bug

`F.scaled_dot_product_attention` with the **EFFICIENT_ATTENTION** backend
returns badly corrupted output (max abs error ~19.5, vs ~0.09 honest bf16
wobble) for a specific real-world input whose **kv sequence length is
≡ 1 mod 32** (bf16, CUDA, batch 1, 4-D boolean `attn_mask`). The **MATH**
backend on the *identical* tensors is correct (matches an fp32 reference
to bf16 wobble), and the *same* tensors with **one extra kv position**
(kv_len ≡ 2 mod 32) are clean on every backend. The attached repro
replays all four selectable backends, so the comparison is complete on
any hardware (CUDNN and FLASH decline these inputs: head_dim 512,
non-null mask).

**The memory layout matters as much as the values.** The failing call
(a Gemma 4 KV-shared/GQA attention layer) looks like:

```
q         bf16 shape (1, 16, 289, 512) stride (2367488, 512, 8192, 1)  # non-contiguous [B,S,H,D] permute
k, v      bf16 shape (1, 16, 289, 512) stride (0, 0, 512, 1)           # ONE kv head expand()ed to 16 heads
attn_mask bool shape (1, 1, 289, 289)  contiguous
dropout_p=0.0  scale=1.0  is_causal=False
```

Materializing the same values contiguously (a `.cpu().clone()` round
trip) makes EFFICIENT_ATTENTION **correct** — the bug requires the
broadcasted stride-0 k/v (and/or the strided q). It is also
value-dependent: randomly initialized weights did not reproduce during
transformers-side triage. The repro therefore ships the exact captured
tensors as raw storages + `(shape, stride, storage_offset)` and rebuilds
the precise views with `Tensor.set_`; this layout-exact replay reproduces
the corrupted full-model output **bit-exactly** (max abs diff 0.0 vs the
captured forward).

Found via Gemma 4 batched generation in transformers
(huggingface/transformers#47651): left-padding a prompt so the padded
total hits 1 mod 32 flips the next-token argmax to an unrelated special
token (logit deltas ~40). Maintainer triage there ruled transformers out:
eager attention and SDPA-with-MATH are clean and the presence of images
is irrelevant.

Repro (only `torch` imported; the `.pt` loads with `weights_only=True`):

```bash
unzip sdpa_repro.zip
python torch_sdpa_repro.py sdpa_repro.pt
```

The script replays both captured calls under every selectable backend
(MATH bf16, EFFICIENT, CUDNN, FLASH) against an fp32-MATH reference and
prints max abs differences. Output on the reporting machine:

```
torch==2.12.0+cu130  cuda=13.0  gpu=NVIDIA RTX PRO 6000 Blackwell Workstation Edition
bundle meta: {'source': 'https://github.com/huggingface/transformers/issues/47651', 'model': 'google/gemma-4-12B-it', 'torch': '2.12.0+cu130', 'transformers': '5.14.1', 'gpu': 'NVIDIA RTX PRO 6000 Blackwell Workstation Edition', 'call_index': 41, 'n_calls_per_prefill': 48, 'poison_kv_len': 289, 'control_kv_len': 290, 'guilty_backends': ['EFFICIENT'], 'note': 'poisoned kv_len % 32 == 1; control differs by ONE left-pad token, same layer, same prompt. Tensors are packed as raw storages + (shape, stride, offset) because the bug is layout-sensitive.'} 

[poisoned] q shape (1, 16, 289, 512) kv_len 289 (% 32 = 1) dtype bfloat16
  captured original run vs fp32 ref:    19.5110
  MATH_bf16  vs fp32 ref:     0.0625
  EFFICIENT  vs fp32 ref:    19.5110
  CUDNN      unsupported on these inputs (RuntimeError)
  FLASH      unsupported on these inputs (RuntimeError)

[control] q shape (1, 16, 290, 512) kv_len 290 (% 32 = 2) dtype bfloat16
  captured original run vs fp32 ref:     0.0865
  MATH_bf16  vs fp32 ref:     0.0625
  EFFICIENT  vs fp32 ref:     0.0865
  CUDNN      unsupported on these inputs (RuntimeError)
  FLASH      unsupported on these inputs (RuntimeError)

Expected: every backend matches the fp32-MATH reference to bf16 wobble (the MATH_bf16 row) on both cases. BUG: on the poisoned case (kv_len % 32 == 1) at least one backend diverges orders of magnitude past wobble -- matching the 'captured original run' row, which is what the full model forward produced -- while the control (ONE extra kv position, otherwise the same tensors) is clean everywhere.
```

The two calls come from the same attention layer of the same prompt and
differ only by one left-pad token, so the control isolates the kv_len
alignment as the trigger. The capture harness (how the tensors were
extracted from the model forward, plus a sweep showing the 1-mod-32
pattern across many lengths) is in the transformers issue linked above.

### Expected behavior

Every backend matches MATH to numerical wobble at every sequence length,
as they already do for the control case and for every other length in
the sweeps on the transformers side (the pattern held across prompt
lengths ~280–2900: corruption iff total kv_len % 32 == 1).

## Versions

```                      
PyTorch version: 2.12.0+cu130             
Is debug build: False                        
CUDA used to build PyTorch: 13.0                     
ROCM used to build PyTorch: N/A                      
                                                             
OS: Ubuntu 24.04.4 LTS (x86_64)                      
GCC version: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 
Clang version: Could not collect                     
CMake version: version 3.28.3                        
Libc version: glibc-2.39                             
                                                             
Python version: 3.12.13 | packaged by conda-forge | (main, Mar  5 2026, 16:50:00) [GCC 14.3.0] (64-bit runtime)
Python platform: Linux-6.17.0-35-generic-x86_64-with-glibc2.39
Is CUDA available: True                                                                                                   
CUDA runtime version: 13.0.88                                                                                             
CUDA_MODULE_LOADING set to:                                                                                               
GPU models and configuration: GPU 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition                                                                                                                                                              
Nvidia driver version: 580.159.03                    
```

(collect_env output trimmed for length; full printout available on
request.)

Observed on: torch 2.12.0+cu130, NVIDIA RTX PRO 6000 Blackwell
Workstation Edition, bf16, single GPU. Not yet tried on non-Blackwell
hardware (the transformers maintainer who reproduced the pattern can
confirm their hardware in the linked issue).

**AI disclosure** (per this repo's AI policy): the repro scripts and the
first draft of this report were written with AI assistance (Claude, via
Cursor). I ran every experiment myself on the hardware above — all
numbers in this report are from my own runs — and the
layout-sensitivity finding came from iterating on two earlier capture
designs whose replays failed to reproduce the corruption until the
exact storages/strides were preserved. I have reviewed the scripts and
the analysis, understand them, and take responsibility for the report.

cc @drisspg (SDPA / attention)

---

# DRAFT: follow-up comment for huggingface/transformers#47651

> Filed the torch-side MRE: pytorch/pytorch#NNNNN
>
> Since dummy weights don't reproduce (the corruption is
> value-dependent), the MRE captures the exact q/k/v/attn_mask tensors
> from the corrupted SDPA call of the failing prefill (kv_len 289 =
> 1 mod 32) plus the same-layer call from a control prefill one pad token
> longer (kv_len 290), and replays them through bare
> `F.scaled_dot_product_attention` under all four backends — no
> transformers code involved. The guilty backend is EFFICIENT_ATTENTION,
> confirming @zucchini-nlp's mem-efficient diagnosis: it diverges ~19.5
> from an fp32-MATH reference on the poisoned length and reproduces the
> captured model forward bit-exactly, while MATH stays at bf16 wobble
> (~0.06) and the control length is clean everywhere.
>
> One extra finding worth recording: the corruption is LAYOUT-dependent,
> not just value-dependent. The failing call has GQA-broadcast k/v
> (`stride (0, 0, 512, 1)` — one kv head `expand()`ed to 16) and a
> non-contiguous q; materializing the same values contiguously makes the
> EFFICIENT backend correct. That's presumably (part of) why dummy-weight
> repro attempts and naive tensor-dump replays came back clean. Tensor
> bundle (raw storages + shape/stride/offset, rebuilt via `Tensor.set_`)
> and the replay script are attached on the torch issue.
>
> @chavalasantosh in light of @zucchini-nlp's triage (kernel bug, not
> mask construction), please don't ship a transformers-side mask change
> for this; if a temporary mitigation is wanted at all, the tightly-
> scoped kind you described (affected backend/dtype only, warn once,
> removable) is the right shape.
