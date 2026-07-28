# Training overview — the self-training roadmap and the `train.py` contract

This is the umbrella document for the self-training project: what the stages
are, what `training/train.py` promises to every future data source, and the
exact training recipe with its justifications. Everything training-related —
this code and these docs — lives in `training/`; the remote-verification
checklist for the current stage is [TO_TEST.md](TO_TEST.md). Companion
documents:

- [TRAINING_GAME_TRACES.md](TRAINING_GAME_TRACES.md) — the standard use of
  game traces (the self-eval loop as a data generator).
- [TRAINING_EXTRA_DATASETS.md](TRAINING_EXTRA_DATASETS.md) — replay mixing
  against catastrophic forgetting, and the early-warning suite.
- [TRAINING_TRACE_EXTRAS.md](TRAINING_TRACE_EXTRAS.md) — non-standard uses of
  game traces: planted-error data, prompt internalization.

A convention used throughout these docs: where the project owner's plan
overrides the assistant's recommendation, the recommendation is preserved in a
**"Disagree and commit"** subsection — what was recommended, why, and what
failure signal would justify revisiting it. The committed plan is what gets
built; the alternatives are the pre-approved fallbacks if it stalls.

The guiding design bias (owner's call, applied everywhere): this repo is a
**springboard for general self-training**, not a containerized solver for one
game. Where a choice trades task-specific performance against generality —
engine-derived supervision, hand-built baselines, role-specialized adapters —
the general/simple option wins by default, and the specialized option is
recorded here as a crutch to add only if training fails without it.

## Stages

| Stage | Contents | Status |
|---|---|---|
| 1 | These docs; bare-bones `train.py`; `weights/` checkpoint convention; checkpoint loading in every script and notebook | this commit |
| 2 | Concrete `DataSource`s: game traces from self-eval sessions (incl. image augmentation), planted-error generator | next |
| 3 | Replay `DataSource`s (arithmetic, reasoning, general instruction, AV) + the early-warning probe suite wired into `train.py`'s eval hooks | after 2 |
| 4 | First real training iteration: generate → grade → train → eval → repeat | after 3 |
| 5+ | Possible: analyst training under a verified metric, prompt internalization, image-decoder graft (see [IMAGE_DECODER_GRAFT.md](../IMAGE_DECODER_GRAFT.md)) | discussion |

## The `train.py` contract

`training/train.py` is a **library, not a run**: the generic loop
(`run_training(sources, config)`) plus a `TrainConfig` dataclass holding
every knob, with the recipe below as its defaults. Each concrete training
run is a short separate script that picks the data sources and the config
deviations and calls the loop — copy
[run_first_iteration.py](run_first_iteration.py) per run/iteration;
`train.py` itself should not change between runs. A generic CLI front-end
(`python -m training.train --data ...`, every flag mapping 1:1 onto a
`TrainConfig` field) is kept for ad-hoc runs.

The loop is deliberately source-agnostic. Everything it knows about data
comes through two small types:

- **`TrainingExample`** — chat `messages` (the same HF content-list format
  `agent/model.py` uses, text and image parts), a `target_text` string (the
  assistant reply to be trained on), and optional `span_weights`: a list of
  `(char_start, char_end, weight)` triples over `target_text`.
- **`DataSource`** — a named iterable of `TrainingExample`s with a mixture
  weight. Game traces, arithmetic replay, planted-error batches, video/audio
  comprehension sets: each is just another `DataSource` plugged into the same
  loop. Stage 1 ships only `JsonlSource` (examples serialized as JSON lines)
  so the loop is testable end to end; real sources arrive in stages 2–3.

### One loss for both label types

Every example is trained with **weighted token cross-entropy**: prompt tokens
contribute nothing, and each target token contributes
`weight * -logprob(token)`.

- Plain SFT is the special case where every target-token weight is 1.0.
- RL-style annotation ("these tokens were good, these were bad") is the
  general case: arbitrary per-token weights, including negative weights for
  tokens to suppress (an advantage-weighted / unlikelihood-flavored update).

Sources express weights in *character* space over `target_text` (that is what
annotators — the analyst's `WRONG:` spans, a rating, a programmatic checker —
naturally produce); the collator maps them onto tokens via the tokenizer's
offset mapping. Per-example loss is normalized by the **sum of absolute token
weights**, so a heavily-annotated example and a plain SFT example arrive at
comparable gradient scale, and long examples do not dominate short ones.

### Sequence construction mirrors inference exactly

The trained sequence is `prompt_ids + content_ids + [eos]` where `prompt_ids`
come from `apply_chat_template(messages, add_generation_prompt=True)` — i.e.
byte-for-byte the same prefix the model sees in `VLModel.generate` — and
`content_ids` are the tokenization of `target_text`. This is exactly the
sequence a successful generation would have produced, so there is no
train/inference template mismatch, and the offset mapping over `target_text`
comes for free.

### Images are first-class

The collator runs the model's own `AutoProcessor` over the full messages, so
every batch carries `pixel_values` (cast to compute dtype, as in
`VLModel.generate`) and the forward pass is the multimodal forward: the loss
lands on text tokens that *attend to the image*. An example that declares an
image part but produces no pixel tensor is a hard error, never a silent
text-only fallback — understanding the frame is most of the point of this
training.

The vision tower stays frozen; the **multimodal projector trains** alongside
the language-model LoRA (via PEFT `modules_to_save`, so it rides inside the
adapter checkpoint). That is the standard low-risk lever for improving visual
grounding without destabilizing the encoder.

Micro-batching (Intermission): the default is `micro_batch=4` with
`grad_accum=4`, so the **effective batch stays 16** (`micro_batch x
grad_accum`); both remain independent knobs. Batches are formed inside
buckets keyed by `(loss kind, image count, length bin)` — loss kind because
KD needs the extra teacher forward, image count because pixel tensors must
stack, length bin to bound padding waste — then the batch list is shuffled
so sources still interleave. The collator right-pads (training never
generates), padded positions carry weight 0, and `weighted_loss` normalizes
each example by its OWN absolute-weight sum before averaging the batch, so
a micro-batch of 4 is exactly 4 batch-1 passes averaged (selftest t4
asserts the parity). `--micro-batch 1` restores the old behavior.

## The recipe, with justifications

QLoRA on a single A100 (the 12B in 4-bit NF4 leaves ample room for
activations at our sequence lengths):

| Knob | Value | Why |
|---|---|---|
| Quantization | 4-bit NF4, double quant, bf16 compute | The QLoRA recipe (Dettmers et al. 2023); matches full-bf16 fine-tuning quality in their ablations |
| LoRA placement | all linear layers of the language model | QLoRA ablation: all-linear beats attention-only; vision tower excluded (frozen), projector trained fully |
| LoRA rank / alpha | r=32, alpha=64 | Literature range for narrow-domain adaptation is r=16–64 with diminishing returns above; alpha=2r is the common scaling |
| LoRA dropout | 0.05 | QLoRA default for <13B models |
| Optimizer | paged 8-bit AdamW | QLoRA default; the paging absorbs activation spikes |
| Peak LR | 1e-4 | LoRA wants ~10x the full-fine-tune LR; 1e-4–2e-4 is the replicated sweet spot for 7–13B |
| Schedule | cosine decay to ~10% of peak, ~3% warmup | Boring and robust; `--scheduler constant` kept as the ablation alternative |
| Weight decay | 0.0 on LoRA params | QLoRA convention — decay fights the low-rank update; exposed as a flag for ablation |
| Grad clip | max-norm 1.0 | Standard stabilization, matters with negative-weight tokens |
| Effective batch | ~16 (micro-batch 4 x grad-accum 4) | Small-data SFT regime; smaller effective batches track the fresh on-policy data better; micro-batch 4 buys the GPU-utilization win without changing the math (bucketed padding, per-example normalization) |
| Epochs per iteration | 1–2 | Self-generated data is only on-policy the first pass; re-epoching amplifies the model's own quirks; instruction-tuning literature sees memorization at 3+ |
| NEFTune noise | alpha ~5, on by default | Jain et al. 2023: uniform embedding noise consistently improves small-data SFT; doubles as a mild regularizer against overfitting one visual domain |
| Seed | fixed, logged | Reproducibility; every run logs config + seed + git rev |

Deliberately **omitted**, recorded so future-us knows they were considered:
label smoothing (distorts move-token confidence, which we grade on), sequence
packing (incompatible with per-example images and span weights), adapter EMA
(overkill at this scale).

### Data volumes (provenance marked)

- **LoRA hyperparameters** — literature-backed: LoRA (Hu et al. 2021) rank
  ablations, QLoRA (Dettmers et al. 2023) recipe, LIMA (Zhou et al. 2023) for
  the small-data-SFT regime.
- **1–5k player generations per iteration across ~50–100 games** —
  engineering judgment, not a citation. Sized by (a) generation wall-clock on
  one A100 (~2–7 h per batch, one full cycle ≈ a day), (b) the task's low
  intrinsic dimensionality (the move decision is ~3 classes over a few
  informative variables; 5k boards tile the bearing space densely), (c) the
  expert-iteration principle that several small fresh batches beat one large
  stale one. Expect roughly half to survive grading/filtering.
- **~10–50k cumulative filtered examples** to saturate the narrow task across
  5–15 iterations — judgment, anchored loosely on expert-iteration papers
  (ReST, ReST-EM, STaR) which use 10k–100k+ per iteration for much broader
  task distributions.
- **First-iteration learning-curve diagnostic** — do not trust the numbers
  above: train on 500 / 1.5k / 5k subsets of the first batch, run the fixed
  eval on each, and let the slope set the next batch size.

## Checkpoints: the `weights/` convention

- A checkpoint is a **PEFT adapter folder**, never a full model copy:
  `weights/<architecture-key>/<checkpoint-name>/` containing
  `adapter_config.json`, `adapter_model.safetensors` (LoRA + projector via
  `modules_to_save`), and `train_meta.json` (config, source mix, metrics, git
  rev). `<architecture-key>` is the `agent.model.MODEL_REGISTRY` key
  (`gemma-4-12b`, `gemma-4-e4b`).
- Base weights always come from HuggingFace. Loading a checkpoint =
  HF base + adapter. `[default]` everywhere means "bare HF, no adapter".
- `weights/` is gitignored and created by `scripts/setup_env.sh`. (On the
  owner's local editing box it is a symlink onto removable storage; remote
  boxes just use the plain directory.)
- Every entry point can select a checkpoint: `MODEL_CHECKPOINT` in `.env`,
  `--checkpoint` on `agent.runner` and `train.py` (`--resume-checkpoint`),
  and the architecture + checkpoint dropdowns in the notebooks.

## Rollback and the early-warning hook

`train.py` maintains an **eval-hook** interface: after every checkpoint save
it runs each registered probe (stage 1 ships one placeholder — held-out loss
on a reserved slice of the training data) and compares guarded metrics
against the best value seen. On regression past the threshold it logs at
ERROR and, per `--on-regression {warn,rollback,abort}` (default `rollback`),
restores the best adapter and continues, or aborts the run. Rollbacks are
capped by `--max-rollbacks` (default 3): rollback restores weights but not
the schedule position, so a persistently regressing run would otherwise
oscillate silently — past the cap the run aborts naming the best
checkpoint. The real probe suite — per-capability benchmarks, planted-error
analyst miss rate — is specified in
[TRAINING_EXTRA_DATASETS.md](TRAINING_EXTRA_DATASETS.md) and lands in
stage 3.

Eval cost is capped (Intermission): the held-out slice takes
`--holdout-fraction` of each source but at most `--holdout-cap` examples
per source (default 100 — the fraction alone would make every eval a
~6k-forward, hour-long affair on the full manifest), the held-out pass runs
micro-batched, and the exact-match probes generate at most 192 tokens per
item (answers are short; ramblers that hit the cap score wrong, which is
itself signal).

## Logging

Every run writes `logs/train_<label>_<timestamp>/` (same convention as the
agent's `run_logging`): `config.json` (full resolved config + seed + git
rev + the discovered LoRA target-module list), `train_log.jsonl` +
human-readable `train_log.txt` (per-step loss, per-source loss, LR, grad
norm, token/weight counts), `events.jsonl` (saves, eval results, rollbacks,
warnings). A training run you cannot reconstruct afterwards is a training
run you cannot debug.
