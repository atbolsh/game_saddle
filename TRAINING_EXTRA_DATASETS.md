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
already the strongest single protection). Suggested starting split of the
replay share: arithmetic/math ~40%, general multimodal ~30%, general
instruction ~20%, AV ~10%; retune from the early-warning suite, not from
taste.

Candidate public datasets (verify exact HF ids on the remote box before
wiring them in — the no-fuzzy-fallbacks rule applies to dataset names too):

| Capability | Candidates | Notes |
|---|---|---|
| Arithmetic / math | GSM8K (`openai/gsm8k`), MetaMathQA, orca-math-word-problems | Short, verifiable, cheap to mix; the analyst-drift compensator |
| Reasoning | OpenThoughts / OpenR1-style distilled reasoning sets | Prefer sets distilled from models at or above Gemma 4's level |
| General multimodal | The Cauldron (`HuggingFaceM4/the_cauldron`), LLaVA-OneVision data | Broad VQA/captioning/document mixtures; keeps non-game vision alive |
| General instruction | A SlimOrca-class instruction set | Keeps English/agentic tone intact |
| Video / audio | ShareGPT4Video-class captions; audio QA sets | Only if the deployed checkpoint actually exercises AV; otherwise probe-only |

### Self-distillation replay (recorded option, recommended default)

Instead of training on a public dataset's *own* targets (whose style may
differ from Gemma 4's and thus cause needless drift), sample prompts from
the public sets and have the **base model generate the targets**, then train
on those. The replay data is then exactly "what the base model already
does", which is the thing being protected. Costs one generation pass per
replay batch; can be produced on the same box between iterations. Use plain
targets first if simplicity wins; switch to self-distilled targets if the
early-warning suite shows style/capability drift that replay mixing alone
does not stop.

## The early-warning suite

A fixed battery of probes, run by `train.py`'s eval-hook interface after
every checkpoint save (mechanism shipped in stage 1 with a placeholder
held-out-loss hook; the real probes land in stage 3). Design rules:

- **Fixed** — the same items every run, never regenerated, so scores are
  comparable across iterations and across weeks.
- **Small** — minutes, not hours; it runs at every save.
- **Per-capability** — one score per protected capability, not one blended
  number that can hide a collapse.

Probe set (initial):

| Probe | Measures | Source |
|---|---|---|
| Held-out game boards: move accuracy + OBS format compliance | The skill being trained | Fixed set of ~100 boards with known-good moves, frozen at suite creation |
| Planted-error analyst miss rate | Analyst quality / sycophancy drift | The planted-error generator of [TRAINING_TRACE_EXTRAS.md](TRAINING_TRACE_EXTRAS.md) |
| Arithmetic slice (~100 GSM8K-class items) | Reasoning/arithmetic retention | Public eval split, fixed subset |
| Non-game VQA slice (~100 items) | General vision retention | Public eval split, fixed subset |
| Instruction-following slice | General English/agentic retention | Small fixed prompt set, scored by exact-match/format checks where possible |

Each guarded metric carries a threshold (e.g. "no more than 3 points below
the best checkpoint's score"); a breach triggers `train.py`'s regression
path — ERROR-level logging and, by default, rollback to the best checkpoint
(`--on-regression warn|rollback|abort`). The analyst miss-rate probe is the
designated tripwire for the shared-network drift risk accepted in
[TRAINING_GAME_TRACES.md](TRAINING_GAME_TRACES.md).

## Memory (NAMS) hygiene during training epochs

Not a dataset matter, recorded here so it is not lost: NAMS accumulates
messages without limit, and the current plan is to **reset episodic memory
every ~100 full games (~200 messages)** during data generation, keeping the
seeded semantic model (`scripts/reset_semantics.sh` is the existing tool).
This changes later, when stored memories become sophisticated enough to be
worth carrying across training epochs.
