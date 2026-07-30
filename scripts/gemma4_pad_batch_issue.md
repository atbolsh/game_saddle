# Draft HF issue — Gemma 4 Unified left-pad corruption

**ON HOLD — do not file yet.** The "5.14 regression" framing below is
suspect: the supposed 5.13.1 rerun produced bit-identical output to the
5.14.1 run, and every benign batch=1 pad ever observed was a multiple of 8
(8/64/256) while every corrupt case used pad=7. The repro script now
sweeps pad lengths and prints its own transformers version; rerun it and
rewrite the Reproduction section around whichever story the sweep tells
(pad-length alignment vs version regression).

File at: https://github.com/huggingface/transformers/issues/new/choose → "Bug report".

---

**Title:** Gemma 4 Unified: left-padded multimodal input produces corrupted
prefill logits (argmax flips to `<audio|>`); regressed further in 5.14

### System Info

- `transformers` version: 5.14.1 (see version matrix below; partially
  present on 5.13.1)
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
SDPA), left-padding a multimodal row corrupts its prefill logits. Measured
max next-token logit delta versus the solo (unpadded, batch=1) forward of
the same prompt:

| condition | 5.13.1 | 5.14.1 |
|---|---|---|
| left-padded row, batch=1 | 0.41–0.66, argmax unchanged (pads 8/64/256) *(TODO: confirm with this exact script)* | **40.19, argmax flips to `<audio\|>`** |
| left-padded row, batch=2 (with unpadded longer row) | **40.0, argmax flips to `<audio\|>`** | **40.19, argmax flips to `<audio\|>`** |
| unpadded control row, same batch=2 | 0.86, unchanged | 0.55, unchanged |

So on 5.13.1 the corruption required a multi-row batch; on 5.14.1 a single
left-padded multimodal row is enough. Text-only prompts are unaffected
(max delta ~0.5–0.66 at pads 8/64/256, no flips).

It is not a collation artifact: every sequence-aligned tensor
(`input_ids`, `attention_mask`, token type ids) is padded in lockstep, the
padded row's suffix is asserted byte-identical to its solo encoding, and
`position_ids` follow the attention mask. The corruption is invisible in
normal use because decoding strips the special token, so greedy `generate`
just returns caption-style text that looks like sampling variance (see
output below) rather than a broken forward.

Standalone script (draws its own image; prints the table and a greedy
generate comparison; exits 1 when the bug fires):

```python
<-- PASTE scripts/gemma4_pad_batch_repro.py HERE -->
```

Output on the system above (transformers 5.14.1):

```text
prompt lengths: short=282 long=289 tokens

solo short top5: 'I':27.25  'The':27.25  'You':22.62  'There':21.12  'In':21.00
B. padded, batch=1           max|dLogit|= 40.1875 mean=11.57746 argmax_flipped=True
    solo top5: 'I':27.25  'The':27.25  'You':22.62  'There':21.12  'In':21.00
    this top5: '<audio|>':25.25  'a':11.50  '-':9.69  'I':8.94  'T':8.81
C. padded, batch=2           max|dLogit|= 40.1875 mean=11.60843 argmax_flipped=True
    solo top5: 'I':27.25  'The':27.25  'You':22.62  'There':21.12  'In':21.00
    this top5: '<audio|>':25.25  'a':11.25  '-':9.38  'I':9.00  'T':8.88
   unpadded control row      max|dLogit|=  0.5469 mean=0.16452 argmax_flipped=False

greedy solo   : 'The image contains red, yellow, green, and blue squares on a light gray background.'
greedy batched: 'a yellow rectangle, a red rectangle, a green rectangle, and a blue rectangle.'
```

### Expected behavior

A left-padded multimodal row should produce (numerically close to) the same
logits as the identical row unpadded — as text-only inputs already do. The
flip to an *audio* special token on an image prompt suggests the padded
path misroutes multimodal features (or misbuilds the vision-block mask)
when real tokens are offset by left padding. Practical impact: any batched
`generate` over variable-length multimodal prompts (standard left-pad
serving) silently degrades the shorter rows.
