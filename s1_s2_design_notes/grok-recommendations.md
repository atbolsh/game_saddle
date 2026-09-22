# Grok recommendations (2026-09-21)

Locked answers to the questions in `INSTRUCTIONS.md`, plus the S1
shape agreed after that note. Sharing S1's convolutional backbone
with Gemma is a later option, written at the end of section 1.
Amended the same day: the checkpoint decision in section 2 is his
call, recorded disagree-and-commit; section 4 (S2 vision placement)
is new.

## 1. S1

A small classifier, not a Helix decoder. Helix's 80M transformer
exists to turn a language latent into dozens of continuous joint
commands. This S1 emits one of four classes per tick.

- **Backbone.** A convolutional net on the order of ResNet-18 (~11M),
  owned by S1, initialized from image pretraining. The board is
  768×768; the red eye is about 31 pixels across. Gemma 4 12B's
  `embed_vision` mixes each 48×48 patch into one vector and has no
  attention inside the patch, so it is the wrong front-end for
  reading which way the sprite faces (Gemma Team, arXiv:2607.02770;
  the 12B replaces the 550M / 27-layer encoder used by the other
  Gemma 4 sizes).
- **Keep the spatial map.** Do not globally pool it away. Project the
  target point `(x, y)` — a vector, not a word — up to the feature
  width and use it as one query over the cells. One attention pool,
  then a linear layer to four logits and a softmax.
- **Actions.** noop, CLOCK, ANTICLOCK, FORWARD. One primitive per
  tick. A counted turn is many ticks. No action chunk: a chunk cannot
  abort when the frame changes (that is how π0 buys 50 Hz; Black et
  al., arXiv:2410.24164).
- **First SFT.** Rendered board, fake target floats, the oracle's next
  primitive. Facing is only in the picture, so the frame is in every
  batch. `s` may be passed later as another vector into the same
  query. Do not build a float-only MLP and bolt vision on afterwards.

The pool is a few hundred thousand parameters. The backbone is the
bulk. This is the Helix idea of a fast net with its own eyes (Figure,
"Helix", 20 Feb 2025: 80M S1, own conv backbone, 200 Hz, conditioned
by a stale S2 latent) cut down to four logits.

`neural_net/s1.py` keeps every ResNet-18 stage (stride 4, 8, 16, and
32 on a 768 frame) and pools each map with the target query. The four
pooled vectors are concatenated into the 4-way decision. ImageNet
weights are an initialization option only; after that the whole
module is a `state_dict`. `s` is not an input.

### Later option: one backbone, both nets

Not part of the first S1. When it is worth doing, run the
convolutional backbone once per frame. S1 reads the map every tick.
S2, on its slower turn, reads the cached map through a linear
projection into the 12B hidden size, spliced in where the image
soft-tokens sit.

Gemma will not understand those vectors on day one. Its pretrained
eye is the pairing of `embed_vision` with the existing weights. The
finetune has to make the new features necessary: drop the native
image tokens on a fraction of steps, so the loss can only be reduced
by reading the convolutional features.

Keep the other schedules available, including for tasks that should
still use Gemma's pretrained visual recognition:

- **Native tower only.** Convolutional features not fed to S2.
  `embed_vision` carries the image, as today.
- **Convolutional tower only.** Native image tokens dropped. S2 has
  to read the shared map.
- **Both disabled.** Text only, for a task that should not see a
  frame.
- **Both on**, with the native tokens dropped on a fraction of steps,
  is the mixture that teaches S2 to use the shared map without
  deleting the pretrained path.

Until that finetune holds, S2 keeps receiving ordinary pixels.

## 2. Which checkpoint

**Decision (his call): start from the aug27 weights** —
`aug27_big_step_iter1_step313`, or `aug27_big_step_iter2_step332`,
the later adapter in the same Sep 3 dump. Either is a LoRA on
Gemma 4 12B, not a new model. The bet: the adapter has learned more
latents about the board than the fresh base has, and the tics can be
edited out in SFT. The original recommendation here was the fresh
base; recorded as disagree-and-commit. The evidence below stays as
the caveat list the SFT has to plan for.

Counts below use non-empty replies only. iter1 and iter2
`traces.jsonl` are mostly pruned tombstones.

| Source | Player starts with `OBS:` | Player emits `[ANALYST]` / `RATING:` | Analyst emits `RATING:` | Analyst emits a literal `[ANALYST]` tag |
|---|---|---|---|---|
| July 24 play, bare `gemma-4-12B-it` | 199/206 (97%) | 0 | — | — |
| aug27 smoke1 (after iter1) | 248/263 (94%); `OBS:` appears in 263/263 | 0 | 262/262 | 122/262 (47%) |
| aug27 smoke2 (after iter2) | 270/293 (92%); `OBS:` in all 293 | 0 | 289/289 | 160/289 (55%) |
| Sep 10 control, aug18 | 337/339 (99%) | 0 | 338/338 | 131/338 (39%) |

The player does not pick up the analyst's `RATING` / `WRONG` /
`[ANALYST]` habit. What it does is a fixed script: `OBS: I am at …;
my eye points toward … o'clock`, then `REASON`, then one move token.
Replies that do not start with `OBS:` almost all start with
`[REMEMBER target: …]` and then the same `OBS:` line. Prompt B in the
September smoke pushed that REMEMBER-first opening to about 94% on
both aug18 and step313. Prompt C deleted `REASON` (0 of 397 step313
replies contain it). The template follows the prompt. The July base
already obeyed `OBS:` before this training.

The adapter's scar is the analyst voice and the vertical words. About
half of analyst replies insert `[ANALYST]` as a section marker, almost
all open with "The player's response is analyzed…", and `TARGET:` plus
`RATING:` are essentially mandatory. The grader agrees with the
flipped map. In `sep10_smoke_pC_aug27s313`, an analyst writes that
"I am at the bottom right" is correct for agent `(0.605, 0.814)`. In
this engine, larger `y` is higher on the screen. `0.814` is the top of
the board. The July base was still saying "upper-left" and
"lower-right".

Caveats that stand under the aug27 decision:

- **The tics differ in depth.** `[ANALYST]`, the mandatory `TARGET:` /
  `RATING:`, and the "The player's response is analyzed…" opening are
  surface habits; clean SFT data that never contains them should wear
  them away. The up/down swap is not a surface habit — the grader
  *agrees* with the flipped map — so it must be actively
  counter-trained, not just left unreinforced. The `s` head is a
  lever here: its regression labels carry the true y-up convention on
  every token.
- **The move-token policy is not a reason to keep these weights.** It
  is the adapter's main earned skill, and it migrates to S1; S2 stops
  emitting move tokens. Whatever value aug27 has must come from board
  latents, not from the policy.
- **The bet is cheaply testable.** Train the section-3 coordinate
  head with the trunk frozen, once on aug27 and once on the fresh
  base. If aug27 really carries more board information in its hidden
  states, the frozen-trunk probe shows it directly, before any SFT is
  spent.

The fresh base remains the fallback if the scars resist editing: it
keeps the bake-off reasoning, drops the `[ANALYST]` tic and the
trained up/down swap, and the same system prompt puts `OBS:` back
immediately.

## 3. How to emit `s`, `v`, `v_bar`

Add a second linear head on the same vector that feeds `lm_head` (the
last hidden state, hidden size 2304). Six outputs, about 14k
parameters. Train it with a regression loss. Leave the 262,144-way
softmax alone.

Hijacking six vocabulary rows fails because `tie_word_embeddings` is
true, so those rows are also token embeddings. Their pre-softmax
values are scaled to compete with a quarter-million other logits.
Masking them out of the softmax removes them from the word
distribution and does not turn them into coordinates. `v_bar` has to
be able to leave the square, so do not put a sigmoid on all six. An
unbounded linear head is the right constraint. A sigmoid only on `s`
is optional.

Call that head on every new token. The extra matmul is nothing next
to the tied 2304×262144 head. For the first training pass, put the
loss on `s` only, one `(x, y)` broadcast across the reply — valid
while the agent does not move during a reply; once S1 acts mid-reply,
labels pin to the frame S2 saw (section 4). Leave `v` and `v_bar` at
loss weight 0 until there are labels. An untrained head will still
produce numbers. Do not hand those to S1.

## 4. S2 vision: stale frame at the end of the prompt

Initial approach: one frame per S2 turn, stale for the whole reply,
with the image tokens placed at the **end** of the prompt, right
before generation. KV caches invalidate from the changed token
onward, so everything before the image — system prompt, history —
stays cached across frame swaps; only the image and the reply so far
ever need recomputing. At 768×768 and 48-px patches the image is
16×16 = 256 tokens.

Optional later optimization: refresh the frame every N tokens
mid-reply. A refresh at reply position `t` is a prefill of
~`256 + t` tokens, so the FLOPs overhead is `(256 + t)/N` extra
token-forwards per emitted token. At N = 1 (every token, t̄ ≈ 100)
that is ~300× decode — out of the question. At N = 32 it is ~10× in
FLOPs, but wall clock is much kinder: decode is memory-bandwidth-bound
(the whole model streams per token) while a few-hundred-token prefill
is one compute-bound pass costing a handful of decode steps, so
N = 32 lands near ~10% wall-clock overhead. Every refresh re-runs the
multimodal prefill path — i.e. the #47651 left-pad workaround — so
that path stays in the loop's tests.

Stale vision makes the emitted `s` stale too: the head reports the
agent position in the frame S2 saw, not the live engine state, and
the training labels must be pinned to the seen frame — a label from
the live position would ask the head to predict motion it cannot
observe. If `s` is ever an input to S1, handle the staleness
explicitly: S1 sees a fresh frame every tick, and a stale `s` must
not override it.

## 5. Gemma 4 as S2

Gemma 4 12B stays the S2. The bake-off picked it over the other open
vision-language models for analyst and debrief quality, and those
requirements have not changed. The encoder-free setup fits a
coordinate head: the hidden state that predicts the next word has
already attended to the raw patches.

The vocabulary is huge, so decode stays heavy no matter how small the
coordinate head is. The left-pad multimodal prefill bug
(`transformers` #47651, residue 1 mod 32) is still in this stack.
Qwen3-VL-8B was the closest alternative, and it lost that bake-off.
E4B is faster and has a real vision tower. It is a worse decider. Do
not change families to make the six floats easier.

## 6. Same repo

Stay in this repo. Put the new loop in `neural_net/` and call the engine,
the renderer, and the oracle from there. A new repo would mean
re-paying for the geometry, the Gemma loader, and the pad workaround.
The debt that hurts is writing this inside `interactive.py` and
`modes.py`, where every ask inherits `OBS:`, NAMS, and the analyst
round. Do not do that.

## 7. Two constraints on the handoff

The loop in `neural_net/` updates S1's target on every new token: S1
steps on the live game until the next token's forward returns (or
until an optional step cap). An untrained `v` will send it after head
noise; that is what the head's training is for. Do not hand an
untrained head's numbers to a run you mean to keep.

`s` in the text and `s` in the head will diverge on purpose at first.
The word loss never touches the new head. The September analyst
already says "bottom" and then prints `y = 0.814` and marks itself
correct. The head is how that sentence stops being graded against
itself.
