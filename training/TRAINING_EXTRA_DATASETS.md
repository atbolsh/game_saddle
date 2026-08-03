# Extra datasets and their use — replay mixing and the early-warning suite

The game data is narrow; the model must stay general. This document records
what gets mixed into every training run besides game traces, and how
degradation is detected early. See
[TRAINING_OVERVIEW.md](TRAINING_OVERVIEW.md) for how any of this plugs into
`train.py` (each replay set is just another `DataSource` with a mixture
weight).

## What must not be forgotten

Capabilities to preserve, in the owner's priority order (French can go):

1. **Reasoning** (chain-of-thought, math/logic);
2. **Image understanding outside this game**;
3. **Video and audio comprehension**;
4. **Agentic behavior** (tool calls, multi-step instructions);
5. **General English** (instruction following, prose quality).

Arithmetic is listed twice on purpose: besides being a capability to
preserve, arithmetic-specific data is the committed compensation for
analyst drift (the analyst's job is mostly small trigonometry and clock
arithmetic — see [TRAINING_GAME_TRACES.md](TRAINING_GAME_TRACES.md), item 3).

## Replay mixing

Every training iteration mixes replay `DataSource`s alongside the game-trace
source, at a combined **~20–50% of each batch** (engineering judgment; the
LoRA-forgetting literature consistently shows even ~10–30% replay largely
preserves held-out capabilities, and LoRA itself — base weights frozen — is
already the strongest single protection). Retune the split from the
early-warning suite, not from taste.

### The manifest (implemented — phase 2)

The single source of truth is **[datasets.json](datasets.json)**: both
`scripts/setup_env.sh` (via `python -m training.download_external`) and the
run scripts (via `training.external_data.sources_from_manifest()`) read it.
Each entry's `examples_per_epoch` is an ABSOLUTE per-epoch quota (the
`ExternalSource` weight is `examples_per_epoch / n_materialized`), so the
mixture never shifts when a cap changes. `enabled: false` turns a dataset
off everywhere at once. All HF ids verified 2026-07; per-row schemas are a
remote TO_TEST item.

| Dataset | HF id / generator | Cap (rows) | Per epoch | Loss | Probe | Role |
|---|---|---|---|---|---|---|
| `gsm8k` | `openai/gsm8k` (main) | all 7.5k | 200 | CE | 100 test items, exact-match | Arithmetic — the analyst-drift compensator |
| `metamathqa` | `meta-math/MetaMathQA` | 20k | 200 | CE | — | Augmented math word problems |
| `orca_math` | `microsoft/orca-math-word-problems-200k` | 20k | 200 | CE | — | More arithmetic breadth |
| `numinamath_cot` | `AI-MO/NuminaMath-CoT` | 20k | 150 | CE | — | Competition-style CoT depth |
| `navigation` | `training/synth_navigation.py` (local, seeded) | 10k | 300 | CE | 100 items, exact-match | Clock/compass/bearing/shortest-rotation in THIS repo's conventions — the geometry failure modes, directly |
| `openthoughts` | `open-thoughts/OpenThoughts-114k` | 8k | 200 | KD | — | General reasoning retention |
| `slimorca` | `Open-Orca/SlimOrca` | 20k | 150 | KD | — | Instruction following / English tone |
| `cauldron_vqav2` | `HuggingFaceM4/the_cauldron` (vqav2) | 6k | 250 | KD | — | Broad non-game VQA |
| `cauldron_cocoqa` | `HuggingFaceM4/the_cauldron` (cocoqa) | 6k | 200 | KD | — | Non-game vision |
| `cauldron_ai2d` | `HuggingFaceM4/the_cauldron` (ai2d) | all 2.4k | 150 | KD | — | Diagram QA — closest proxy to board-like images |
| `sharegpt4video` | placeholder, `enabled: false` | — | 0 | KD | — | Video replay — sidelined, see below |
| `audio_qa` | placeholder, `enabled: false` | — | 0 | KD | — | Audio replay — sidelined, see below |

**Why video/audio replay is sidelined (2026-07).** A priority call, not a
technical dependency — KD replay needs no connection to the game harness.
The risk of drift is low because LoRA never touches the vision/audio towers
(`train.build_model` excludes them from adapter placement), so those
modalities can degrade only through the language side's handling of tower
embeddings. Meanwhile the cost of enabling them now is real: both entries
lack converters (and `audio_qa` lacks even a dataset choice; the downloader
refuses loudly if enabled without a converter); video decodes to
many-frames-per-example, i.e. long-sequence KD — the exact kept-logit OOM
class profiled in selftest t10, so enabling it means a `micro_batch_cap`
and a fresh t10 VRAM/time profile; and whether the deployed 12B checkpoint
even accepts audio input needs a remote check (docs only promise audio for
E4B). Revisit after the first full training round.

Replay volume with these counts: **~2,000 replay examples/epoch, of which
arithmetic + navigation = 1,050 (~52% of replay)** — arithmetic-heavy on
purpose (see above). When the ~2.5k-example game source joins, replay is
~45% of a batch, inside the 20–50% band. Coverage note: LLaVA-OneVision
data from the original candidate list is deliberately substituted by the
Cauldron configs — same purpose (broad non-game VQA), one clean download
code path instead of a multi-repo layout.

**CE vs KD, the rule:** CE ("strengthen") trains on the dataset's own
targets and is used exactly where we WANT the model to move — arithmetic and
navigation, the analyst-drift compensators. KD ("preserve") ignores the
dataset's authorship style and instead matches the student's logits to the
**untrained base model** over the same target tokens (soft cross-entropy,
target positions only; the teacher is the same process via PEFT
`disable_adapter()`, no second copy of the weights) — used everywhere the
goal is *don't drift*, because "what the base already does" is precisely the
thing being protected. Per-line `loss` lives in the materialized jsonl, so
one file can mix kinds if ever needed.

### Self-distillation replay (implemented alternative to KD)

The generation-based variant of the same idea: have the **base model answer
the replay prompts itself**, then train on those outputs with plain CE —
"keep producing exactly what you would have produced" expressed as ordinary
SFT data instead of a logit-matching loss.
`python -m training.generate_self_distill --dataset <name>` writes
`data_selfdistill.jsonl` beside a materialized dataset (originals never
touched); swap it in with a plain `JsonlSource`. The utility forces
`set_default_checkpoint(None)`, so the generating model is the pristine
base even when `MODEL_CHECKPOINT` is set in `.env`.

The trade-off, and the decision rule: KD costs nothing to prepare but is
rigid at train time (it pins the student's distribution at every position
of *someone else's* text, and needs two forwards per example);
self-distillation costs one offline generation pass per dataset plus the
extra jsonl on disk, after which training is a single ordinary CE forward
on text in the base model's own voice. **Run with KD first.** Switch a
dataset to self-distilled targets only on evidence from the early-warning
suite: KD sources still drifting (per-source held-out guards), a broken
`disable_adapter()` teacher path (selftest t3's KD-equals-teacher-entropy
check), or KD memory pressure that `micro_batch_cap` can't tame (the
stage-10 note in [TO_TEST.md](TO_TEST.md) — long-target sources like
openthoughts already run capped).

### The analyst KD anchor (implemented — in-domain replay)

The manifest sets anchor *generic* chat/reasoning/vision behavior, but the
analyst task — privileged frame + settings + grading a player reply — is
far from all of those distributions, and it is the one shared-weight
capability the plan protects without ever training it. The analyst anchor
is the in-domain replay set that closes that gap: datagen records every
analyst exchange (the EXACT analyst prompt + the analysis it produced) to
`data_game/<label>/analyst_traces.jsonl`, and `AnalystTraceSource`
([game_traces.py](game_traces.py)) feeds those contexts back as
uniform-weight **KD examples** — soft cross-entropy against the **frozen
base** model on the analyst's own contexts.

Design decisions, recorded:

- **The teacher is always the frozen base** (`disable_adapter()`),
  regardless of which checkpoint generated the trace — so the anchor pins
  analyst behavior to Gemma-as-shipped and never chases its own drift,
  even when later iterations' traces come from trained checkpoints. This
  is why KD, not CE-on-recorded-text, is the loss: CE would anchor to one
  sampled (possibly drifted) trajectory; KD needs no stored teacher at
  all, only contexts.
- **No reward enters.** It is an anchor, not RL: records with
  `rating: null` are kept. Records whose analysis quoted *unverified*
  (hallucinated) `WRONG:` spans are dropped loudly by default — the one
  analyst failure the harness detects for free is not worth anchoring on.
  Truncated search-call generations are never recorded at all (counted as
  `analyst_skipped_search` in the datagen stats).
- **Quota, not proportion:** `examples_per_epoch` (default 150, the same
  convention as the manifest sets) — the per-epoch mix stays fixed while
  the trace corpus grows across datagen runs.
- **The gate is the standard one:** `run_training` auto-guards
  `heldout_loss/analyst_<label>`, and for a KD source that held-out loss
  *is* drift-from-base on analyst contexts — an analyst-drift meter logged
  to `eval_log.jsonl` at every save alongside the other per-source
  metrics, with the usual rollback on a >10% relative regression.
- **Relation to the planted-error probe** ([TRAINING_TRACE_EXTRAS.md](TRAINING_TRACE_EXTRAS.md)):
  the probe is *measurement*, this anchor is *prevention*; the probe's
  corrupted replies must never enter training, or the tripwire stops
  being evidence.
- **Player/analyst separation stays structural:** `analyst_traces.jsonl`
  and `traces.jsonl` are never read by the same source, and the datagen
  leak tripwire continues to guard player records only.

### Download strategy and disk

`setup_env.sh` measures free disk at the repo root ONCE and writes
`data_external/settings.json`: `download_mode: "full"` by default (whole
datasets land in the HF cache and stay there — no network dependence
afterwards), `"stream"` when less than **60 GB** is free (only the consumed
shards ever hit disk). Both modes produce byte-identical materialized
output under `data_external/<name>/`. Approximate figures: **materialized
data ~2–2.5 GB with the default caps; full-mode HF cache ~20 GB across the
manifest (vqav2 alone is ~13.5 GB)** — see the README's disk-budget
paragraph for the box-level totals.

## The early-warning suite

A fixed battery of probes, run by `train.py`'s eval-hook interface after
every checkpoint save. Design rules:

- **Fixed** — the same items every run, never regenerated, so scores are
  comparable across iterations and across weeks.
- **Small** — minutes, not hours; it runs at every save.
- **Per-capability** — one score per protected capability, not one blended
  number that can hide a collapse.

What is implemented now (phase 2):

- **Per-source held-out loss** — the held-out slice is reported and GUARDED
  per dataset (`heldout_loss/<name>`), each example scored with its own
  loss kind; for KD sources the held-out KD loss *is* the drift-from-base
  measure.
- **Exact-match probes** ([probes.py](probes.py), built by
  `download_external.py`): `exact_match/gsm8k` (100 fixed test items) and
  `exact_match/navigation` (100 fixed synthetic items — the direct
  instrument for the clock/compass failure modes). Greedy generation,
  ~3–5 min per save on an A100; guarded higher-is-better.
- **Per-task eval history** — every evaluation appends one FLAT row to the
  run's `eval_log.jsonl` with every metric as its own key, so any metric's
  trajectory graphs in one line (`pandas.read_json(..., lines=True)`).

Probe set (planned additions, next stages):

| Probe | Measures | Source |
|---|---|---|
| Held-out game boards: move accuracy + OBS format compliance | The skill being trained | Fixed set of ~100 boards with known-good moves, frozen at suite creation |
| Planted-error analyst miss rate | Analyst quality / sycophancy drift | The planted-error generator of [TRAINING_TRACE_EXTRAS.md](TRAINING_TRACE_EXTRAS.md) |
| Instruction-following slice | General English/agentic retention | Small fixed prompt set, scored by exact-match/format checks where possible |

Each guarded metric carries a soft threshold (relative, default 10% —
WARNING only) and a hard one (2x worse than best, an absolute ceiling
where declared — the analyst anchor's is 5.0 — or 3 consecutive soft
breaches); a hard breach triggers `train.py`'s regression path —
ERROR-level logging and, by default, rollback to the last GOOD checkpoint
(`--on-regression warn|rollback|abort`; details in
TRAINING_OVERVIEW.md's rollback section). The analyst miss-rate probe is
the designated tripwire for the shared-network drift risk accepted in
[TRAINING_GAME_TRACES.md](TRAINING_GAME_TRACES.md).

## Memory (NAMS) hygiene during training epochs

Not a dataset matter, recorded here so it is not lost: NAMS accumulates
messages without limit, and the current plan is to **reset episodic memory
every ~100 full games (~200 messages)** during data generation, keeping the
seeded semantic model (`scripts/reset_semantics.sh` is the existing tool).
This changes later, when stored memories become sophisticated enough to be
worth carrying across training epochs.
