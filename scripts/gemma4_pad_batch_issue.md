# Final HF issue draft — Gemma 4 Unified left-pad corruption

READY TO FILE. Steps: https://github.com/huggingface/transformers/issues/new/choose
→ "Bug report", paste the sections below, and insert the current contents of
`scripts/gemma4_pad_batch_repro.py` at the single marker.

---

**Title:** Gemma 4 Unified: left-padding a multimodal prompt by a specific
amount (pad=7 here) corrupts prefill logits — argmax flips to `<audio|>`;
neighboring pad lengths are fine

### System Info

- `transformers` version: 5.14.1 (output below is from 5.13.1; the two
  versions produce **bit-identical** results for this script, corruption
  included)
- Platform: Linux-6.17.0-35-generic-x86_64-with-glibc2.39
- Python version: 3.12.13
- Huggingface_hub version: 1.18.0
- Safetensors version: 0.8.0
- Accelerate version: 1.14.0
- Accelerate config: not found
- DeepSpeed version: not installed
- PyTorch version (accelerator?): 2.12.0+cu130 (CUDA)
- Using distributed or parallel set-up in script?: No
- Using GPU in script?: Yes (single GPU, `device_map="auto"`)
- GPU type: NVIDIA RTX PRO 6000 Blackwell Workstation Edition

### Who can help?

@zucchini-nlp

### Reproduction

With `Gemma4UnifiedForConditionalGeneration` (`google/gemma-4-12B-it`, bf16,
SDPA), left-padding a multimodal prompt (282 tokens, one image) by exactly
7 positions corrupts the prefill logits: max next-token logit delta vs the
unpadded forward jumps from ~0.5 (bf16 wobble) to **40**, and the argmax
flips from `'The'`/`'I'` to the **audio** special token `<audio|>` — on an
image prompt. Every other pad length in the sweep (1–6, 8, 9, 15, 16, 63,
64) stays within wobble, so this is not gradual numerical noise; it is a
discrete failure at a specific offset:

```text
B. padded, batch=1, pad=6    max|dLogit|=  0.5000 mean=0.13692 argmax_flipped=False
B. padded, batch=1, pad=7    max|dLogit|= 40.1875 mean=11.57746 argmax_flipped=True
    solo top5: 'I':27.25  'The':27.25  'You':22.62  'There':21.12  'In':21.00
    this top5: '<audio|>':25.25  'a':11.50  '-':9.69  'I':8.94  'T':8.81
B. padded, batch=1, pad=8    max|dLogit|=  0.5312 mean=0.12108 argmax_flipped=False
```

Notes:

- Batch size is irrelevant: batch=1 with pad=7 corrupts identically to a
  batch of 2 where the shorter row is padded by 7 (the standard left-pad
  collation for variable-length batched `generate`). The unpadded row of
  that batch stays clean (max delta 0.55).
- Not a collation artifact: every sequence-aligned tensor (`input_ids`,
  `attention_mask`, token type ids) is padded in lockstep, the padded row's
  suffix is asserted byte-identical to its solo encoding, and
  `position_ids` follow the attention mask.
- Text-only prompts are unaffected at every pad length tried (max delta
  ~0.5–0.66); the image input is required.
- Reproduces bit-identically on transformers 5.13.1 and 5.14.1.
- Ambiguity we did not resolve: for this prompt, pad=7 is also the pad that
  makes the total padded length 289 — we have not varied the prompt to
  distinguish "pad length 7" from "total length 289" as the trigger. The
  script below makes that a two-minute experiment.
- User-visible symptom: decoding strips the special token, so greedy
  `generate` on a padded-by-7 row silently returns degraded caption-style
  text instead of an instruction-following reply — it looks like sampling
  variance, not a broken forward:

```text
greedy solo   : 'The image contains red, yellow, green, and blue squares on a light gray background.'
greedy batched: 'a yellow rectangle, a red rectangle, a green rectangle, and a blue rectangle.'
```

Standalone script (no external assets; draws its own image, sweeps pad
lengths at batch=1, then runs the batch-of-2 and greedy comparisons; exits
1 when the bug fires):

```python
<-- PASTE scripts/gemma4_pad_batch_repro.py HERE -->
```

Full output (transformers 5.13.1; 5.14.1 output is bit-identical):

```text
transformers==5.13.1  torch==2.12.0+cu130
prompt lengths: short=282 long=289 tokens

solo short top5: 'I':27.25  'The':27.25  'You':22.62  'There':21.12  'In':21.00
B. padded, batch=1, pad=1    max|dLogit|=  0.5156 mean=0.13077 argmax_flipped=False
B. padded, batch=1, pad=2    max|dLogit|=  0.5625 mean=0.12874 argmax_flipped=False
B. padded, batch=1, pad=3    max|dLogit|=  0.5820 mean=0.14670 argmax_flipped=False
B. padded, batch=1, pad=4    max|dLogit|=  0.5469 mean=0.12501 argmax_flipped=False
B. padded, batch=1, pad=5    max|dLogit|=  0.4453 mean=0.10723 argmax_flipped=False
B. padded, batch=1, pad=6    max|dLogit|=  0.5000 mean=0.13692 argmax_flipped=False
B. padded, batch=1, pad=7    max|dLogit|= 40.1875 mean=11.57746 argmax_flipped=True
    solo top5: 'I':27.25  'The':27.25  'You':22.62  'There':21.12  'In':21.00
    this top5: '<audio|>':25.25  'a':11.50  '-':9.69  'I':8.94  'T':8.81
B. padded, batch=1, pad=8    max|dLogit|=  0.5312 mean=0.12108 argmax_flipped=False
B. padded, batch=1, pad=9    max|dLogit|=  0.6875 mean=0.07407 argmax_flipped=False
B. padded, batch=1, pad=15   max|dLogit|=  0.7031 mean=0.12159 argmax_flipped=False
B. padded, batch=1, pad=16   max|dLogit|=  0.4648 mean=0.08500 argmax_flipped=False
B. padded, batch=1, pad=63   max|dLogit|=  0.5000 mean=0.08344 argmax_flipped=False
B. padded, batch=1, pad=64   max|dLogit|=  0.4062 mean=0.07283 argmax_flipped=False
C. padded, batch=2           max|dLogit|= 40.1875 mean=11.60843 argmax_flipped=True
    solo top5: 'I':27.25  'The':27.25  'You':22.62  'There':21.12  'In':21.00
    this top5: '<audio|>':25.25  'a':11.25  '-':9.38  'I':9.00  'T':8.88
   unpadded control row      max|dLogit|=  0.5469 mean=0.16452 argmax_flipped=False

greedy solo   : 'The image contains red, yellow, green, and blue squares on a light gray background.'
greedy batched: 'a yellow rectangle, a red rectangle, a green rectangle, and a blue rectangle.'
```

### Expected behavior

A left-padded multimodal row should produce (numerically close to) the same
logits as the identical row unpadded, at every pad length — as it already
does at 12 of the 13 pad lengths tried and for text-only inputs at all of
them. A discrete corruption at one specific offset, flipping the argmax to
an *audio* token on an image prompt, suggests an alignment/stride
assumption in the padded multimodal path (vision-block attention mask or
image-feature placement). Practical impact: standard left-padded batched
`generate` over variable-length multimodal prompts silently degrades
whichever rows happen to land on a bad pad offset.
