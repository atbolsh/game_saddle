# Model candidates for replacing Gemma 4 E4B

Research snapshot: **2026-07-23**. Constraints: vision input, free/open
weights, fine-tunable on a single A100 (80 GB). Soft cap ~20B parameters.

> **Now implemented:** every model below except the "For completeness" tier
> is a switchable entry in `agent.model.MODEL_REGISTRY` (registry key in
> parentheses in the tables/lists below), selectable via the notebooks'
> model dropdown or the `MODEL_KEY` env var. All HF repo ids were verified
> against the HF API on 2026-07-23; the only correction was **Ovis2.5-9B**,
> whose repo moved from `AIDC-AI` to **`ATH-MaaS/Ovis2.5-9B`**.
> `trust_remote_code` (from each repo's `auto_map`) is needed for Step3-VL,
> Phi-4-Reasoning-Vision, Kimi-VL, and InternVL3.5; none of the repos are
> gated except the usual Google/Gemma license acknowledgment handled by
> `HF_TOKEN`.

Feasibility framing for one A100 80 GB:

- **Full fine-tune**: realistic up to ~8-12B (bf16 + gradient checkpointing +
  8-bit optimizer).
- **LoRA/QLoRA**: up to ~30B total parameters. Perfectly adequate for our
  task; borderline MoE models (3-4B active) are included on this basis.

## Top tier — try these first

| Model (registry key) | Size | License | Notes |
|---|---|---|---|
| Qwen3-VL-8B-Thinking (`qwen3-vl-8b-thinking`; Instruct: `qwen3-vl-8b-instruct`) | ~9B dense | Apache 2.0 | Best all-around candidate |
| Gemma 4 12B Unified (`gemma-4-12b`) | 12B dense | Apache 2.0 | Lowest migration cost from E4B |
| GLM-4.1V-9B-Thinking (`glm-4.1v-9b-thinking`) | 9B dense | MIT | RL reasoner, far above its weight |
| Step3-VL-10B (`step3-vl-10b`) | 10B | open weights | Perception/reasoning leader <=10B |

**Qwen3-VL-8B-Thinking** — first recommendation. Deepest fine-tuning
ecosystem (LLaMA-Factory, Unsloth, TRL first-class), Apache 2.0, 128K+
context. The Qwen VL line is RL-trained on *grounding* (pixel coordinates,
GUI targets) — our player's failure mode is precise spatial perception of a
simple synthetic scene, and grounding-RL'd models are optimized for exactly
"where is the thing, in numbers". Thinking variant adds reasoning RL. At 9B,
full fine-tune is feasible on the A100; LoRA trivial.

**Gemma 4 12B Unified** (released 2026-07-02 — newer than our E4B; its
`gemma4_unified` architecture needs **transformers >= 5.10.0**, enforced at
load time via the registry's `min_transformers`). Two
arguments: (1) the whole harness (chat template, image token budgeting,
stop-string behavior, RegexStopCriteria) already speaks Gemma, so switching
cost is near zero; (2) it is the encoder-free "unified" design — raw image
patches go straight into the decoder, so fine-tuning trains the WHOLE
perception path in one pass, no frozen-encoder question. Built-in thinking
mode, 256K context. Risk: brand-new architecture, fine-tuning recipes less
battle-tested than Qwen's.

**GLM-4.1V-9B-Thinking** — beat Qwen2.5-VL-72B on 29/42 benchmarks via
curriculum RL (RLCS), MIT license, strong on the STEM/grounding/GUI mix that
correlates with our task. Less tooling support than Qwen, more than Kimi.

**Step3-VL-10B** (StepFun) — newest of the four; heavy RLVR + RLHF
post-training, explicitly strong on counting, grounding, geometry-flavored
perception. Most capability per parameter; ecosystem maturity is the open
question. Verified against the repo (2026-07-24): **always-thinking** (the
chat template auto-opens `<think>`; vendor deployments use the DeepSeek-R1
reasoning parser), loads via `AutoModelForCausalLM` + the model card's
`key_mapping`, and the official example decodes **greedily** (its
`generation_config` is unfiltered 1.0/1.0/0) — all encoded in the registry
spec. Reasoning runs long (benchmarks use up to 64K tokens), so raise
`MODEL_MAX_NEW_TOKENS` if replies truncate mid-think.

## Solid second tier

- **Phi-4-Reasoning-Vision-15B** (`phi-4-reasoning-vision-15b`)
  (Microsoft, MIT, 2026-03) — reasoning-first
  VLM with explicit think-blocks it can invoke or skip per task. Catch:
  **16K context** — debrief/self-eval conversations with interleaved frames
  could bump into it. Fine for the player role, tight for the analyst role.
- **Kimi-VL-A3B-Thinking-2506** (`kimi-vl-a3b-thinking-2506`)
  (Moonshot, MIT) — 16B total / 2.8B active
  MoE, 128K context, native-resolution vision encoder (nice for 768x768
  frames — no tiling artifacts). Reasoning-RL'd. Downsides: DeepSeek-V3-style
  MLA/MoE internals mean thinner fine-tuning support, and 2.8B active is a
  small brain for subtle geometry — instinct says our task wants dense.
- **MiMo-VL-7B-RL** (`mimo-vl-7b-rl`) (Xiaomi, Apache 2.0) — 7B, mixed on-policy RL;
  outperformed Qwen2.5-VL-7B on 35/40 tasks, excellent GUI grounding. The
  budget pick; very cheap to iterate on.
- **InternVL3.5-8B / -14B** (`internvl3.5-8b` / `internvl3.5-14b`)
  (Shanghai AI Lab, Apache 2.0) — Cascade RL
  (offline then online) with real reasoning gains; the 14B is one of the
  strongest dense models under 20B. Good HF support.
- **Ovis2.5-9B** (`ovis2.5-9b`) (Apache 2.0) — strong 9B with thinking mode and
  native-resolution ViT; less community mindshare, consistently good numbers.
  Repo id is now `ATH-MaaS/Ovis2.5-9B` (moved from `AIDC-AI`).

## Borderline over the limit (MoE; QLoRA only)

- **Qwen3-VL-30B-A3B-Thinking** (`qwen3-vl-30b-a3b-thinking`) — 30B total /
  3B active; ~60 GB bf16 weights, so QLoRA-or-nothing on one A100, but
  near-32B-class quality at 3B-active inference speed.
- **Gemma 4 26B-A4B** (`gemma-4-26b-a4b`) — same story inside the Gemma
  family (26B total / 4B active). The next step after Gemma 4 12B without
  leaving the harness.
- **InternVL3.5-30B-A3B** (`internvl3.5-30b-a3b`) — MoE sibling of the
  InternVL line, same caveats.

## For completeness (skip)

- **Llama 3.2 11B Vision** — only Llama that fits; Sept 2024, bolt-on
  cross-attention adapter, no reasoning post-training. Llama 4 multimodal
  starts at 109B total. Would be a downgrade from Gemma 4 E4B on reasoning.
- **Pixtral 12B** (Mistral, Apache 2.0) — fine in 2024, no reasoning line
  since; superseded by the top tier.
- **Molmo 7B** (AI2) — fully open *data* as well as weights (matters for
  reproducibility of the future self-training platform), but trails tier one
  on reasoning.
- **DeepSeek-VL2** — MoE, Dec 2024, no reasoning refresh since.

## Concrete plan

Bake-off between **Qwen3-VL-8B-Thinking** and **Gemma 4 12B Unified**, using
the interactive self-eval notebook as the benchmark: same boards, same
default questions; compare OBS-line accuracy and analyst ratings. Qwen tests
the "better grounding DNA" hypothesis; Gemma 4 12B tests "same family, more
capacity, unified architecture" with minimal harness changes. If neither
moves the needle on perception, add **Step3-VL-10B** or
**GLM-4.1V-9B-Thinking** as the third contender before the MoE borderliners.

## Practical warning for any swap

The adaptation layer now exists in `agent/model.py`: per-family adapters
strip think blocks before parsing/persisting, generation stopping is gated
so a move token inside a think block never halts generation, and a reply
that forgets its think-close tag stays fully visible (an intended move is
still honored) but is flagged as a FORMAT ERROR. Two caveats remain for the
first REMOTE run of a new family: (1) the custom-code repos (Step3-VL,
Phi-4-Reasoning-Vision, Kimi-VL, InternVL3.5, Ovis2.5) register their model
classes under `AutoModelForCausalLM` only, so the registry loads them
through that Auto class (`ModelSpec.loader="causal"`, verified against each
repo's `auto_map`); their *processor*/chat-template behavior is the
remaining unverified surface and fails loudly with the model key + repo id
if incompatible; (2) chat-template/token differences can shift stop
behavior, so watch the first few generations in the run logs.
