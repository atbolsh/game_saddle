# Standard use of game traces — the self-eval loop as data generator

How gameplay becomes training data. This is the committed plan (the owner's
design); where the assistant recommended something else, the alternative is
preserved in a "Disagree and commit" subsection with its reasons and the
failure signal that would justify revisiting it. See
[TRAINING_OVERVIEW.md](TRAINING_OVERVIEW.md) for the loss/recipe mechanics.

## The committed plan

1. **Data comes from the self-eval loop.** A player generation over a live
   board, immediately debugged by the analyst (same conversation, privileged
   settings). The pair (context, player reply, analyst verdict) is one unit
   of raw data.
2. **Backprop through player tokens only.** The player's reply is the
   training target; analyst tokens are never LM targets (weight 0 /
   excluded). The analyst's verdict enters training only as *annotation* —
   an example-level rating and per-span `WRONG:` marks that become the
   per-token weights of [TRAINING_OVERVIEW.md](TRAINING_OVERVIEW.md)'s
   unified loss.
3. **One shared network for player and analyst.** No role-specialized
   adapters. The hypothesis: sycophantic drift is driven by backpropagating
   through judge tokens selected for agreement; since analyst tokens are
   never targets, that gradient pathway simply does not exist. Residual
   drift through shared representations is compensated with extra
   non-game data — especially arithmetic (see
   [TRAINING_EXTRA_DATASETS.md](TRAINING_EXTRA_DATASETS.md)) — and watched
   via the planted-error miss-rate probe (see
   [TRAINING_TRACE_EXTRAS.md](TRAINING_TRACE_EXTRAS.md)).
4. **No engine-derived baselines.** Neither the player's data selection nor
   the analyst's verdicts are checked against engine-computed ground truth.
   The grading signal is the analyst itself. This is the "springboard"
   principle: in the environments this repo is meant to grow into, no oracle
   engine will be available, so the loop must not learn to depend on one.

The exact mapping from analyst verdict to per-token weights (e.g. rating as
a global scale, `WRONG:` spans down-weighted or negative, move token
up-weighted) is a stage-2 decision — it belongs to the game-trace
`DataSource`, not to `train.py`, which only sees `(char_start, char_end,
weight)` spans.

### Disagree and commit: separate LoRA adapters per role

**Recommended:** two adapters on the shared frozen base — one trained for the
player role, one (initially untrained = the initial product by construction)
for the analyst — swapped at inference by the harness, which already knows
which role is speaking.

**Reasons:** (a) player-role training shifts shared representations that the
analyst rides on, so "player tokens only" removes the *direct* drift pathway
but not the indirect one; (b) independent evaluation and independent rollback
per role; (c) the guarantee the owner wants — "the end product is as good an
analyst as the initial product" — holds trivially when the analyst's weights
literally do not change.

**Committed instead:** one shared network. Rationale: simplicity, the
springboard principle (role-adapter plumbing is one more crutch), and the
expectation that analyst quality is protected well enough by (no analyst
backprop + arithmetic replay + the miss-rate probe).

**Revisit if:** the planted-error miss rate of the trained network degrades
versus the base model across iterations while player metrics improve — that
is the signature of indirect drift, and role adapters are the direct fix.

### Disagree and commit: engine-verified analyst training (STaR for the critic)

**Recommended:** because the analyst receives privileged settings, most of
its claims are objectively checkable arithmetic (clock hour of facing,
bearing to gold, shorter rotation direction, move correctness, `WRONG:` span
verbatim-ness). Generate analyst outputs, keep only those whose checkable
claims all verify against the engine, and train the analyst on the survivors
— filtered SFT where the training signal is "were your claims true", so the
sycophancy gradient never exists. This is the one known-safe way to make the
analyst *improve* rather than merely not degrade.

**Committed instead:** no analyst backprop at all for now, and no engine
verification (springboard principle). Recorded as the pre-approved mechanism
for a later "train the analyst" stage if one happens.

### Disagree and commit: engine-derived baselines for player data

**Recommended:** the engine's settings admit trivial oracles — a
settings-to-optimal-move function (compute relative bearing, threshold into
FORWARD/CLOCK/ANTICLOCK), and trajectory-length comparison against an
automated solver over full games. Used as a *data filter* (drop or
down-weight player examples whose move contradicts the oracle) they are
invisible at inference and cost nothing at runtime; used as *metrics* they
give an analyst-independent measure of player progress.

**Committed instead:** the analyst is the only grader (springboard
principle: no oracle in future environments). Recorded as the first crutch
to add if training on analyst-graded data alone fails to move the fixed
eval — an oracle filter cleanly separates "the data was bad" from "the
training didn't take".

### Disagree and commit: supervised perception data from the engine

**Recommended:** perception (misreading where the gold is, which way the
agent faces) has been the dominant player failure mode, and it does not need
to be learned from graded rollouts: the engine can emit unlimited supervised
(frame → ground-truth OBS line) pairs for free, at any scale, in minutes.
Mixing 10–50k such pairs into training would carry the perception burden and
let the self-play batches carry only decision/format signal.

**Committed instead:** perception is learned from the same analyst-graded
self-eval data as everything else (springboard principle — programmatic
captions are exactly the hard-wired engine information being avoided).
Recorded as the second crutch: if the learning curve shows decision-making
improving while OBS-line accuracy stalls, this is the targeted fix.

## Data volumes (committed working numbers)

Recorded from the planning discussion; provenance and reasoning in
[TRAINING_OVERVIEW.md](TRAINING_OVERVIEW.md):

- **1–5k player generations per iteration**, across **~50–100 games**;
  expect roughly half to survive grading/filtering (500–3k training
  examples per iteration).
- **~10–50k cumulative filtered examples** over 5–15 iterations to saturate
  the narrow task.
- **First-iteration learning curve** (train on 500 / 1.5k / 5k subsets,
  compare on the fixed eval) decides whether the next batch scales up or
  down. These numbers are starting points, not commitments.

## Visual generalization — regularize the images

Worry, recorded as a requirement on the stage-2 game-trace `DataSource`:
image-interpretation skills learned on this one renderer (one palette, one
agent sprite, one board style) may not generalize. The data source that
feeds frames into training must therefore apply **label-safe image
regularization** — augmentations sampled per example at load time:

- Gaussian pixel noise;
- blur (mild);
- brightness / contrast / color jitter;
- JPEG compression artifacts;
- slight random crops / rescales (small enough not to cut off the agent or
  the gold).

**NOT label-safe here, do not use naively:** horizontal/vertical flips and
rotations. They invert or shift the clock/bearing semantics that the OBS
line and the move token are graded on; using them would require rewriting
the text labels to match, which is possible but is its own project.

A second, independent augmentation axis is **engine-side variation**: board
and agent colors, agent/gold sizes, render size. That path changes what the
"true" frame looks like rather than degrading it, and needs nothing but
engine parameters — worth wiring into the data generator when convenient.
