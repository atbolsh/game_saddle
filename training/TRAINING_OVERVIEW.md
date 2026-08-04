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
| 5+ | Possible: analyst training under a verified metric, prompt internalization + instinct-move default (staged plan in [TRAINING_TRACE_EXTRAS.md](TRAINING_TRACE_EXTRAS.md)), image-decoder graft (see [IMAGE_DECODER_GRAFT.md](../IMAGE_DECODER_GRAFT.md)) | discussion |

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

### The weekend run (multi-epoch, unattended)

`python -m training.run_weekend` chains **3 expert-iteration epochs**
unattended: for each epoch k it runs datagen
(`generate_game_traces --label weekend_iter<k> --parallel 16`, on the
previous epoch's adapter), trains on that epoch's traces (reward-weighted
CE + the `PlayerAnchorSource` trust region anchored to the previous
adapter — see TRAINING_GAME_TRACES.md) + the analyst anchor + the manifest
replay sources, resumed from the previous adapter, and finishes with a
**post-train smoke eval**: 8 real games with the fresh checkpoint (label
`<prefix>_smoke<k>`, never trained on) whose win rate / mean rating /
degeneracy fraction / gold-distance deltas are logged (`grep smoke_eval`)
and stored under `smoke` in the state file — every checkpoint gets a
game-performance reading even when nothing runs after it. Datagen is
parallel by default: the Gemma 4 left-pad prefill bug
([huggingface/transformers#47651](https://github.com/huggingface/transformers/issues/47651))
has a verified workaround (stage-6 notes in [TO_TEST.md](TO_TEST.md)),
and measured scaling (2026-07-30, 96 GB box) never inverts: serial
24.1 s/gen, `--parallel 10` 8.5, `--parallel 24` 6.4. The default 16 is
set by VRAM comfort — 24 ran close to the limit. `--parallel 1` restores
the fully serial conservative path.

Launch (remote box, NAMS up, external data downloaded, from repo root):

```bash
nohup python -m training.run_weekend > weekend.log 2>&1 &
```

**Fitting the window.** One epoch costs roughly
`--max-generations x s/gen` of datagen (~6 h at default parallelism,
~20 h serial, per the measurements above) plus one train stage
(measured ~2 h + setup on the 3000-generation overnight corpus, selftest
t10 2026-07-31; pad ~10–15% for save-time eval hooks);
`--max-generations` (default 3000) is the knob. `--epochs`, `--games`,
`--parallel`, `--start-checkpoint`, and `--prefix` are also flags. Before
the real launch, rehearse the whole orchestration cheaply (TO_TEST.md,
"weekend rehearsal"): a tiny budget plus `--train-max-steps` exercises
subprocess chaining, checkpoint hand-off, and the state file in under an
hour.

**Weights survive every epoch by construction:** each epoch's train stage
saves under `weights/<arch>/weekend_iter<k>_step<N>/` (periodic +
final saves), and labels differ per epoch, so nothing is ever
overwritten. The final adapter name is logged at the end and recorded in
the state file.

**Crashes and restarts.** Every stage is a subprocess, retried once on
failure. A partially-crashed datagen resumes via `--append` with the
remaining generation budget; a twice-failed train stage carries the
previous epoch's checkpoint forward (always logged at ERROR, never
silently). `data_game/weekend_state.json` records finished stages, so
rerunning the same command after a box reboot skips completed work; to
force a stage to rerun, delete its entry from that file (and for datagen,
the `data_game/weekend_iter<k>/` directory).

**Stop-on-poison** is the one exception to "keep marching": if datagen or
a smoke eval exits with code 3, the degeneracy fuse tripped
(`DEGENERACY_FUSE` in `generate_game_traces.py`: 25 consecutive
generations with no parseable rating or move = a collapsed checkpoint).
That is deterministic, so the orchestrator records the broken checkpoint
under `poisoned` in the state file and STOPS the whole run instead of
retrying or training later epochs on garbage.

**Monitoring:** `tail -f weekend.log`; per-epoch
`data_game/weekend_iter<k>/generation_stats.json` (wall clock,
`seconds_per_generation`) and `logs/train_weekend_iter<k>_<stamp>/`
(steps, evals, rollbacks). The orchestrator also samples topline GPU
memory once a minute (stage-tagged, `data_game/weekend_vram.jsonl`) and
ends the log with a **VRAM summary** (peak, mean during datagen, mean
during training) plus total datagen/train hours — check those before
raising `--parallel`.

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
  general case: arbitrary per-token weights. The MECHANISM supports
  negative weights (unlikelihood-flavored suppression), but the game
  reward mapping deliberately never emits them — weighted CE with net-
  negative weights is unbounded below and collapsed the 2026-08-01 run
  (postmortem in TRAINING_GAME_TRACES.md); suppression now comes from the
  bounded player trust region instead.

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

Any dedicated vision/audio **tower** stays frozen (excluded from LoRA by
name); the **multimodal vision embedder** trains alongside the language-model
LoRA via PEFT `modules_to_save` (so it rides inside the adapter checkpoint).
On Gemma 4 / Gemma 4 Unified that module is `embed_vision`
(`model.embed_vision`) — there is no `multi_modal_projector`. For the 12B
Unified architecture this embedder *is* the whole vision path (encoder-free
patch→LM projection), which is the intended trainable lever. Under 4-bit
QLoRA the embedder/towers + `lm_head` are kept in bf16 via
`BitsAndBytesConfig.llm_int8_skip_modules` with **full path prefixes**
(`model.embed_vision`, not bare `embed_vision` — transformers only matches
a skip key as a prefix at the start of the module name). PEFT cannot
`requires_grad` on quantized packed weights; a post-load assertion fails
loudly if the skip list missed anything under the `modules_to_save` target.

Micro-batching (Intermission): the default is `micro_batch=4` with
`grad_accum=4`, so the **effective batch stays 16** (`micro_batch x
grad_accum`); both remain independent knobs. Batches are formed inside
buckets keyed by `(loss kind, image count, length bin)` — loss kind because
KD needs the extra teacher forward, image count because pixel tensors must
stack, length bin to bound padding waste — then the batch list is shuffled
so sources still interleave. The collator right-pads (training never
generates), padded positions carry weight 0, and `weighted_loss` normalizes
each example by its OWN absolute-weight sum before averaging the batch, so
a micro-batch of 4 is the mean of the same per-example losses as 4
batch-1 passes (selftest t4 asserts parity within bf16/SDPA noise).
`--micro-batch 1` restores the old behavior.

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

`train.py` maintains an **eval-hook** interface: a **baseline eval at
step 0** (checkpoint saved + every probe scored BEFORE the first optimizer
step — added after the 2026-08-01 collapse hid entirely in front of the
first periodic eval), then after every checkpoint save it runs each
registered probe (held-out loss on a reserved slice, reported per source)
and compares guarded metrics against the best value seen. Regressions are
**two-tier**, sized so noise never triggers weight surgery (the collapse
was a 70x blowup, not a 10% wobble): a SOFT regression (worse than
best-ever by `--regression-tolerance`, default 10%) only logs a WARNING —
unless the same metric stays soft-regressed 3 evals in a row
(`--soft-streak-limit`), which is persistent drift and escalates; a HARD
regression (worse by `--hard-multiplier`, default 2x, or past a guard's
absolute ceiling — the analyst KD anchor's is 5.0) triggers
`--on-regression {warn,rollback,abort}` (default `rollback`). Exception
(2026-08-04): the KD-anchor sources' guards are **ceiling-only**
(`guard_relative = False` — player anchor ceiling 1.0, analyst 5.0),
because an anchor's held-out loss is a drift meter pinned to its step-0
floor: the best-ever multiplier read "any learning at all" as a hard
regression and rolled a healthy retest back to base twice
(TRAINING_GAME_TRACES.md). Rollback restores `last_good_ckpt` — the
newest checkpoint whose own eval had no hard regression — never the save
that just regressed; `run_weekend` likewise hands the NEXT epoch the
`last_good_checkpoint` from the trainer's `done` event, not the
highest-step directory (the final save lands BEFORE the final eval, so
after a terminal rollback the newest directory is exactly the rejected
one). Rollbacks are capped by `--max-rollbacks` (default 3): rollback
restores weights but not the schedule position, so a persistently
regressing run would otherwise oscillate silently — past the cap the run
aborts naming the last good checkpoint. The real probe suite — per-capability benchmarks, planted-error
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
