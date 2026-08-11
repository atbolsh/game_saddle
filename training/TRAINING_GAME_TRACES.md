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
   **PARTIALLY SUSPENDED 2026-08-05:** after two failed runs the
   engine-oracle crutch was activated for move tokens (see "Disagree and
   commit: engine-derived baselines" below, and the crutch block in
   [game_traces.py](game_traces.py)). The principle stands as the end
   state; the crutch is explicitly temporary.

The exact mapping from analyst verdict to per-token weights lives in the
game-trace `DataSource` ([game_traces.py](game_traces.py)), not in
`train.py`, which only sees `(char_start, char_end, weight)` spans. The
implemented mapping is in "The reward scheme" below.

## The implemented loop

`python -m training.generate_game_traces --label iterN [--checkpoint ...]`
drives the EXACT self-eval machinery of the interactive notebook
(`InteractiveSelfEvalSession`, default player and analyst questions, same
prompts, same memory screening) headlessly: per round it asks the player,
asks the analyst once, then ends the round so the move propagates. **A game
is formally: the gold eaten, or `--max-moves` player rounds** (default 50),
whichever comes first -- each round costs two generations, and early
wandering traces carry little signal per round, so short games are the
cheap default; raise the cap as the player gets good enough to need it.

**Perception-question rounds** (`--question-rate`, default 0.15): with this
probability a round asks a perception question ("Are you facing the gold?",
"Is the gold to your left?", ...) instead of the default move request —
targeted pressure on the identified weak point (reading the frame), with
fully gradeable ground truth since the analyst checks the answer against
the exact settings it already receives. The pool
(`PERCEPTION_QUESTION_GROUPS` in the generator) is **direction-balanced**:
questions are grouped with their mirrored variants and sampling is
group-uniform then variant-uniform, so "to your left?" and "to your right?"
(and above/below, top/bottom) are asked with identical probability and the
data never teaches a directional prior. A correct answer is prose with NO
move token; a mistakenly emitted token still propagates (the game contract:
an emitted token is executed) and the analyst prompt explicitly calls an
unrequested move token a format mistake. Question rounds don't advance the
game, so at rate q a `--max-moves` cap yields ~q fewer move rounds per
game. `meta.question` stores the exact question asked, which is how
perception records are distinguished downstream; the run summary counts
them as `perception_rounds`.

One record per player generation lands in `data_game/<label>/traces.jsonl`:

- `messages` — the EXACT prompt of the accepted player generation (system
  prompt, screened NAMS context, accumulated search notes, frame), with the
  frame's url rewritten to a stable copy in `data_game/<label>/images/`
  (byte-identical to what the player saw; the live `memory_images/` copy
  does not survive NAMS resets);
- `target_text` — the raw player reply, the ONLY trainable tokens;
- `meta` — RAW annotations only: analyst rating, harness-verified `WRONG:`
  spans, action, game/move indices, the round's `question`, `game_won`,
  `moves_from_end`, and the round's agent-to-gold distances
  (`dist_to_gold_before` / `dist_to_gold_after`, normalized board units;
  `null` when no gold remains). The distances are a rating-INDEPENDENT
  quality signal — "did this move close in on the gold" — used to
  cross-check the analyst's rating distribution after a run; they are not
  (yet) a training input. Rewards are computed at TRAINING time from these
  fields, so every ratio below stays tunable without regenerating data.

A second file, `data_game/<label>/analyst_traces.jsonl`, records the
analyst side of every round: the EXACT analyst prompt (`messages`,
privileged — settings, unscrubbed context — with the frame url pointing at
the SAME stable copy as the player record) and the analysis as
`target_text`, plus `meta` (rating, verified/unverified spans, questions,
indices, search-call count). It feeds `AnalystTraceSource`, the KD-vs-base
analyst anchor documented in
[TRAINING_EXTRA_DATASETS.md](TRAINING_EXTRA_DATASETS.md). Rounds whose
accepted analyst generation was a truncated search call are skipped and
counted (`analyst_skipped_search`). The player/analyst separation is
structural — no loader reads both files.

Housekeeping baked into the generator:

- **Parallel sessions** (`--parallel`, default 16): N sessions play N games
  concurrently, one worker thread each, sharing the ONE loaded model
  through `agent/parallel_gen.py` — concurrent generations merge into
  batched decode calls (batch-1 decode is memory-bandwidth-bound, so
  extra rows are cheap; measured 2026-07-30: serial 24.1 s/gen,
  `--parallel 10` 8.5, `--parallel 24` 6.4 — diminishing but
  never-inverting returns, default set by VRAM headroom). Caveat for Gemma 4
  Unified: a left-padded multimodal prefill is corrupted at specific
  widths (upstream transformers#47651), so `generate_batch` pads
  mixed-length rows around the poisoned widths and parity-checks every
  padded row's prefill before decoding (KNOWN TRANSFORMERS BUG WORKAROUND
  banner in `agent/model.py`); rows that fail the check fall back to
  equal-length cohorts that decode sequentially inside the call.
  Requests batch only with identical stop signatures,
  so player-phase and analyst-phase generations never truncate each other.
  A concurrent session's analyst text is exactly as visible to another
  player as a PAST session's (the intended cross-game learning channel);
  current-session screening is untouched. `--parallel 1` restores the
  sequential loop.
- **NAMS reset every ~100 games** (`--reset-every`): episodic memory is
  wiped back to the seeded semantic model; tips are `Preference` nodes and
  survive. Prevents unbounded memory growth and keeps retrieved context
  representative of a fresh box. With parallel workers the reset happens at
  block boundaries: all workers drain their current games, one resets, all
  restart — no game ever loses its memory mid-flight.
- **Inference-time image noise** (`--no-noise` to disable): every stored
  snapshot is degraded in place (strength 0.5, see below) BEFORE the player,
  the analyst, NAMS, or the training copy sees it — one invariant, one set
  of bytes.
- **Analyst-leak tripwire:** every analysis generated in the current session
  is matched against each record's serialized player context; a hit aborts
  the run. The load-side screening (`exclude_analyst` + `exclude_session`)
  should make a hit impossible — the tripwire is there for the day that
  regresses.
- Records whose analyst forgot the `RATING:` line are written with
  `rating: null` and counted; `GameTraceSource` DROPS them loudly at load
  time (never train on a guessed reward).
- **Degeneracy fuse** (`DEGENERACY_FUSE = 25`): 25 CONSECUTIVE generations
  with no parseable rating AND no parseable move mean the checkpoint is
  collapsed, not unlucky. All workers drain and the run exits with code 3
  (`EXIT_POISONED`); `run_weekend` treats that as fatal for the whole loop
  (stop-on-poison). Added after the 2026-08-01 collapse burned ~15h
  generating gibberish (postmortem below).
- **End-of-run stats + plots:** counters land in
  `data_game/<label>/generation_stats.json` and, via
  `training/datagen_plots.py`, as figures in
  `logs/datagen_stats_<label>_<stamp>/` — rating histogram, rolling mean
  rating, rounds per game with wins marked, verified-`WRONG:`-span rate.
  Read the rating histogram BEFORE training: grades compressed into a
  narrow positive band with rare WRONG spans mean a nearly uniform reward,
  i.e. the batch will train like plain self-SFT.

## The reward scheme

**The algorithm, by name: single-sample offline REINFORCE with a shaped
per-token advantage — equivalently, reward-weighted regression.** The
weighted-CE loss of [TRAINING_OVERVIEW.md](TRAINING_OVERVIEW.md) *is* the
REINFORCE surrogate `-w * log p(token)` with the per-token weight playing
the role of the advantage; there is no critic, no importance ratio, no
clipping — which is also why each batch is trained for a SINGLE epoch
(after one gradient step the data is off-policy and this estimator has no
correction for that).

**The 2026-08-05 shape/scale split.** A real bug forced a restructuring:
`weighted_loss` normalizes each example by its own sum of |weights|, which
makes the loss **scale-invariant in the span weights** — any uniform
reward multiplier on a reply's weights cancels top and bottom (only the
terminator's fixed 1.0 kept it from cancelling exactly; a ×0.1 novelty
"downweight" changed the gradient ~8%, not 10×). Every reply-wide reward
knob of the aug4 runs — rating base, win boost, action balance, novelty —
was therefore a near-no-op, and the run cloned its corpus at effectively
uniform strength. The reward now splits into two carriers:

**SCALE — `TrainingExample.example_weight`** ("how much this reply
matters"), applied by the loss AFTER normalization, so it actually
reaches the gradient. Multiplicative chain, per reply:

1. **base = exp-advantage** (`rating_advantage`): 0 at `r ≤ -0.5` (hard
   floor: confirmed-bad replies teach *nothing*), else
   `min(3.0, exp((r - r_bar) / 0.3))` where `r_bar` is the corpus mean
   rating (pre-pass at source load, logged). The analyst's **ranking** is
   the signal we trust — its absolute calibration drifts and saturates
   (aug4: 77% of iter2 moves rated exactly +1.0), and under any fixed
   affine mapping a saturated corpus trains as uniform behavior cloning.
   The exponential restores contrast wherever the ratings sit: ~2× per
   +0.2 of rating above the mean, symmetric below, capped at 3.0 so one
   reply cannot own its batch;
2. **+ win boost, won games only:** `1.0 * 0.95^d` (`d` = rounds from the
   winning move), ADDED to the base. The win is the loop's ONLY
   ground-truth reward and is sized to dominate the analyst — it even
   rescues an analyst-floored reply near a win. The ~13.5-round half-life
   still pays the closing approach far more than a lucky game's wandering
   prefix;
3. **× action balance — TEMPORARY HACK** (`action_balance_multipliers`,
   screaming block in [game_traces.py](game_traces.py)):
   `mean_count / count(action)` over the same corpus, clamped to
   `[1/4, 4]` (a binding cap logs a WARNING — the corpus mix is
   degenerate). Equal total cloning mass per move TYPE; closed-loop, so a
   turn-heavy corpus can no longer amplify itself. REMOVE once a
   principled per-move signal exists;
4. **× novelty ("boredom") decay — OFF BY DEFAULT, ON in weekend runs
   since 2026-08-11** (`NoveltyTracker`; `GameTraceSource(...,
   novelty=True)` at the `run_weekend.train_one_epoch` call site; class
   default stays False): `max(0.1, 0.9^k)` on the k-th consecutive
   identical move (perception rounds skipped, not reset). The
   "re-enable only on observed repetition degeneracy" condition fired:
   the aug6 run's turn runs were 97% self-continuing and 11 epochs of
   training *deepened* the commitment (flip rate 0.10 → 0.03);
5. **× 0.25 when the engine oracle contradicts the move**
   (`ORACLE_WRONG_SCALE`, crutch below) — the rationalization of a wrong
   move is suspect end to end;
6. **× 2.0 on ray-hit rounds** (`TRANSITION_BOOST`, 2026-08-11) — the
   transition moves that gate winning are a handful per game and were
   drowning in continuation moves; see the tightening note in the
   oracle section below.

A scale of exactly 0 (floored rating, no win boost) **skips the record**
(logged) — zero-scaled forwards teach nothing and cost real GPU time.

**SHAPE — `span_weights`** (relative emphasis WITHIN the reply,
`build_span_weights`):

1. **base 1.0** over the whole reply — every token of a reply is the same
   "move", so they share its reward;
2. **verified `WRONG:` spans → −0.5** (`WRONG_SPAN_WEIGHT`): **bounded
   unlikelihood** in the loss (`-log(1 - p)`, |w| as emphasis — train.py,
   NEGATIVE WEIGHTS) — verified-bad text is actively suppressed again,
   with a floored objective this time (postmortem below). Unverified
   spans never reach training. The win boost does NOT soften these
   anymore: a verified-wrong claim stays wrong in a won game;
3. **the move token gets the oracle's verdict** (crutch block in
   [game_traces.py](game_traces.py)): `[MOVE]` span → **1.5** when it
   matches the engine oracle, **−0.5** (unlikelihood) when it contradicts
   it, untouched on "neutral" (defensible under the instructed 45° cone)
   or "unknown" (no oracle meta — pre-2026-08-05 corpora; counted and
   logged).

### The 2026-08-04 spin-bot (why the balance, oracle, and contrast exist)

The first collapse-proofed retest produced no gibberish — but got WORSE at
the game each epoch. The analyst rated 77% of iter2's moves exactly +1.0
(each individual "rotate toward the gold" step looks locally sensible), so
reward-weighted regression degenerated into plain behavior cloning of the
corpus action mix, which was already turn-heavy; each epoch amplified it
(smoke evals: 28% FORWARD / 40% ANTICLOCK → 6.5% / 69%, mean rating
0.785 → 0.348, a bot that spins in place forever). The 2026-08-05 defense
stack: the exp-advantage base restores contrast among the saturated
ratings; the action balance equalizes total cloning mass per move type,
so a turn-heavy corpus can no longer amplify itself; the engine oracle
punishes wrong-way turns directly; and the RL-scale LR (3e-6, clip 0.1 —
TRAINING_OVERVIEW recipe) shrinks how far any one epoch can push. The
novelty decay stays available behind its toggle if repetition degeneracy
survives all of that. Only the smoke eval caught this failure — every
offline held-out loss improved while the policy degraded.

### Postmortem: the 2026-08-01 collapse (the unbounded objective)

The original mapping used the rating directly (weights in `[-1, 1]`, with a
`negative_scale` knob defaulting to symmetric REINFORCE). Negative-weight
CE (`w * -log p`) is **unbounded below with a gradient that GROWS as
p → 0**: the model is paid ever more to make those tokens ever less
likely, and the cheapest descent direction is to destroy the whole
distribution. The first weekend run did exactly that: epoch 2's corpus
skewed negative, training loss fell 0.11 → -37.5 in ~300 steps, the
checkpoint fled into `<unused...>` token gibberish, and epochs 3–7 burned
~15h of datagen on unparseable output. The guards missed it because (a) no
eval ran before step 200, so the first post-collapse eval BECAME the
baseline, and (b) the rollback target tracked the newest checkpoint with
ANY improving metric — i.e. the regressed save itself.

**The corrected lesson (2026-08-05):** the problem was never negative
signal per se — it was the unbounded FORM. Bounded unlikelihood
(`-log(1 - p)`) also pushes a bad token's probability down, but the loss
floors at 0 and its gradient **vanishes** as p → 0: there is no collapse
well. The interim all-non-negative mapping (`max(0, (r+0.5)/1.5)`) was
safe but overcorrected — masking verified-bad text to 0 suppresses
nothing, and pairing it with a saturated analyst produced the aug4
uniform-cloning failure. The current scheme (above) restores negative
signal in the bounded form. The rest of the defense stack: the step-0
baseline eval + two-tier guards + `last_good_ckpt` rollback (train.py),
the player trust region (below), the datagen degeneracy fuse
(`DEGENERACY_FUSE` in the generator: 25 consecutive rating-less AND
move-less generations → exit 3), and the orchestrator's stop-on-poison
(run_weekend.py).

### The player trust region (kd_anchor)

`PlayerAnchorSource` ([game_traces.py](game_traces.py)) replays the SAME
player traces with uniform weight and `loss="kd_anchor"`: soft
cross-entropy against the **parent checkpoint** — the adapter this epoch
resumed from, loaded as a second frozen PEFT adapter
(`TrainConfig.anchor_checkpoint`, wired to `resume` by
`run_weekend.train_one_epoch`). RLHF-style "don't drift too far from where
you started".

- **Parent, never base:** the point of the loop is to SURPASS the base
  model, so a base anchor would fight the objective. The parent anchor only
  bounds how far a SINGLE epoch can move; it advances every epoch, so
  cumulative improvement is unbounded. (Epoch 1 from base has no parent
  checkpoint and falls back to the base teacher — which IS its parent.)
- **Every parseable record is kept** — including `rating: null` and
  `r ≤ -0.5` records that the reward mapping drops or zeroes: the tokens
  cloning is NOT pulling on are exactly the unconstrained directions a
  collapse escapes through.
- **Mass:** `PLAYER_ANCHOR_WEIGHT = 0.25` — a quarter of the player corpus
  per epoch, ~half an hour of the train stage.
- **Ceiling-only guards for BOTH anchors** (`guard_relative = False`,
  2026-08-04 fix): an anchor's held-out KD loss is a drift meter — it
  starts at its minimum (student == teacher at step 0) and can only rise,
  so a best-ever multiplier guard means "at most one entropy of drift
  ever" and hard-rolled a healthy retest back to base at 0.15 nats/token
  (normal RLHF territory: healthy fine-tuning sits at 0.05–0.3 nats/token;
  outputs go weird past ~1–2). The player anchor's absolute ceiling is
  `PLAYER_ANCHOR_CEILING = 1.0` (an order of magnitude above healthy
  drift, ~6x below the observed collapse at ~6 nats/token); the analyst
  anchor — unchanged, still pinned to the FROZEN BASE, never chasing its
  own drift — keeps its ceiling of 5.0 (healthy ≈ 1; the collapse hit 63).

### Deferred: the analyst revamp

Considered and deferred (a multi-day project): adding `CORRECT:` span marks
beside `WRONG:`, and/or per-span ratings instead of one message-level
rating, giving denser and more surgical per-token weights. **Trigger to
revisit:** perception/move quality stalls across iterations while analyst
ratings stay high — that is the signature of the message-level rating being
too blunt an instrument. Until then the message-level rating + WRONG spans
carry the signal. (Also listed in [FUTURE_GOALS.md](../FUTURE_GOALS.md).)

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

### Disagree and commit: engine-derived baselines for player data — **ACTIVATED 2026-08-05**

**Recommended:** the engine's settings admit trivial oracles — a
settings-to-optimal-move function (compute relative bearing, threshold into
FORWARD/CLOCK/ANTICLOCK), and trajectory-length comparison against an
automated solver over full games. Used as a *data filter* (drop or
down-weight player examples whose move contradicts the oracle) they are
invisible at inference and cost nothing at runtime; used as *metrics* they
give an analyst-independent measure of player progress.

**Originally committed instead:** the analyst as the only grader
(springboard principle), with this recorded as the first crutch to add if
training on analyst-graded data alone failed to move the fixed eval.

**That failure signal arrived** (two runs: the 2026-08-01 collapse, then
the aug4 spin-bot under a saturated analyst), so the crutch is now LIVE —
deliberately, and marked ULTRAVIOLET in the code. Datagen stamps the raw
facts per move (`_oracle_meta`: `oracle_rel_bearing`, `oracle_ray_hit`,
`oracle_move` — exact, since bare levels have no internal walls);
`game_traces.oracle_verdict` classifies at train time:

| verdict | geometry | effect |
|---|---|---|
| `correct` | matches `oracle_move` (or any turn when the gold is ≥170° behind) | move-token span **1.5** |
| `neutral` | defensible under the instructed 45° cone (FORWARD in-cone without a ray hit) | none — the analyst's rating stands |
| `wrong` | turn away from the shorter rotation; FORWARD outside the cone; **any turn under a ray hit** (missed forward — tightened 2026-08-11) | move-token span **−0.5** (unlikelihood) + example scale **×0.25** |
| `unknown` | no oracle meta (old corpora) / no move | none; counted + logged |

**2026-08-11 tightening + transition boost:** the aug6 11-epoch run showed
missed forwards were the biggest leak — ~900 of them slipped through as
"fine-tuning neutral" while the analyst rated 42% of them ≥ +0.8, so
FORWARD-on-ray-hit compliance sat at ~50% for 11 epochs. A turn under a
ray hit is now `wrong`, and every RAY-HIT round's `example_weight` is
multiplied by `TRANSITION_BOOST` (2.0): taking the FORWARD at alignment is
cloned at 2×, missing it nets 0.25 × 2 = 0.5× with a sharpened −0.5 span
on the turn token. The same run also turned novelty decay ON at the
`run_weekend` call site (`GameTraceSource(..., novelty=True)`, class
default still False) to tax the 97%-self-continuing turn runs, and the
shared analyst grading block (`_BLOCK_GRADING_TOLERANCE` in
[agent/modes.py](../agent/modes.py)) gained explicit must-be-negative
rules for missed forwards (<~10° off dead-ahead) and wrong-direction
turns, plus a "rating must follow your own geometry" consistency clause.

Facts live in the data, thresholds live train-side — retuning never
requires regenerating traces. The springboard principle still holds as the
end state: this oracle hard-wires "good play = greedy geodesic to the
nearest gold", which future environments (internal walls, competing goals)
will make actively wrong — remove it or demote it to an analyst-visible
feature by then (crutch block in [game_traces.py](game_traces.py),
[FUTURE_GOALS.md](../FUTURE_GOALS.md)).

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

Worry: image-interpretation skills learned on this one renderer (one
palette, one agent sprite, one board style) may not generalize. Implemented
in [image_noise.py](image_noise.py) and applied at BOTH ends — at datagen
inference (strength 0.5, in place on the stored snapshot, so player /
analyst / training all see the same bytes) and again at training time
(strength 1.0, fresh per-run noised copies made by `GameTraceSource`) —
**label-safe image regularization**, sampled per image:

- Gaussian pixel noise, plus per-pixel speckle (multiplicative);
- mild blur OR JPEG compression artifacts (one of the two);
- brightness / contrast / color jitter;
- 2–6 BIG discolored patches (semi-transparent tints, 8–25% of the image
  side each; placed uniformly at random, so partially covering the agent or
  the gold is allowed and intended — tints, not dropout);
- one whole-image color-drift tint (a full-frame translucent rectangle,
  weaker than the patches);
- slight random crops / rescales (≤4% per edge — small enough not to cut
  off the agent or the gold).

**10% of frames skip everything** (`_SKIP_PROB`, 2026-08-04) and pass
through completely clean, independently at each end: the network must also
see uncorrupted boards, or it ends up miscalibrated on the un-noised
frames it meets outside the datagen harness. (So ~10% of stored snapshots
are pristine, ~10% of training copies add nothing on top of the stored
frame, and ~1% of training images are pristine end to end.)

Magnitudes are tuned by eye in `notebooks/noise_tuner.ipynb` (sliders over a
live board frame; a "Regenerate" button rerolls the board and every random
draw).

**NOT label-safe here, do not use naively:** horizontal/vertical flips and
rotations. They invert or shift the clock/bearing semantics that the OBS
line and the move token are graded on; using them would require rewriting
the text labels to match, which is possible but is its own project.

A second, independent augmentation axis is **engine-side variation**: board
and agent colors, agent/gold sizes, render size. That path changes what the
"true" frame looks like rather than degrading it, and needs nothing but
engine parameters — worth wiring into the data generator when convenient.
