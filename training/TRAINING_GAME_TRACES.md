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

One record per player generation lands in `data_game/<label>/traces.jsonl`:

- `messages` — the EXACT prompt of the accepted player generation (system
  prompt, screened NAMS context, accumulated search notes, frame), with the
  frame's url rewritten to a stable copy in `data_game/<label>/images/`
  (byte-identical to what the player saw; the live `memory_images/` copy
  does not survive NAMS resets);
- `target_text` — the raw player reply, the ONLY trainable tokens;
- `meta` — RAW annotations only: analyst rating, harness-verified `WRONG:`
  spans, action, game/move indices, `game_won`, `moves_from_end`. Rewards
  are computed at TRAINING time from these, so every ratio below stays
  tunable without regenerating data.

Housekeeping baked into the generator:

- **Parallel sessions** (`--parallel`, default 3): N sessions play N games
  concurrently, one worker thread each, sharing the ONE loaded model
  through `agent/parallel_gen.py` — concurrent generations merge into
  batched decode calls (batch-1 decode is memory-bandwidth-bound, so a
  batch of 3 costs barely more than a batch of 1; expected ~2–2.5x on the
  datagen wall-clock). Requests batch only with identical stop signatures,
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

Per token of the player reply, `GameTraceSource` builds the weight as:

1. **base = analyst rating** `r ∈ [-1, 1]` over the whole reply — every
   token of a reply is the same "move", so they share its reward;
2. **verified `WRONG:` spans override the base with -1.0** — the one
   token-level signal the analyst gives; unverified spans never reach
   training;
3. **win boost, won games only:** `b = 0.2 * 0.9^d` (`d` = rounds from the
   winning move) is added UNIFORMLY to every token of the message —
   including WRONG spans (they become `-1.0 + b`). The win is the loop's one
   ground-truth signal; on a won trajectory even flagged text gets its
   penalty softened rather than trusted less. Magnitudes: the boost peaks at
   0.2 on the winning move and fades with ~10-round half-life, so analyst
   grades (|r| up to 1.0) stay the dominant signal and long wandering
   prefixes of a lucky game get almost nothing;
4. **clamp to [-1, 1]**, then multiply still-negative weights by
   `negative_scale` — the committed default is **1.0 (symmetric REINFORCE,
   negatives unlearn at full strength)**; `0.5` is the gentler first knob if
   training destabilizes, `0.0` is filtered behavior cloning (strictly
   positive rewards, maximally stable, learns nothing from mistakes).

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

Worry: image-interpretation skills learned on this one renderer (one
palette, one agent sprite, one board style) may not generalize. Implemented
in [image_noise.py](image_noise.py) and applied at BOTH ends — at datagen
inference (strength 0.5, in place on the stored snapshot, so player /
analyst / training all see the same bytes) and again at training time
(strength 1.0, fresh per-run noised copies made by `GameTraceSource`) —
**label-safe image regularization**, sampled per image:

- Gaussian pixel noise;
- mild blur OR JPEG compression artifacts (one of the two);
- brightness / contrast / color jitter;
- 2–6 small discolored patches (semi-transparent tints, a few percent of
  the image side each — mild local discoloration, not dropout);
- slight random crops / rescales (≤4% per edge — small enough not to cut
  off the agent or the gold).

**NOT label-safe here, do not use naively:** horizontal/vertical flips and
rotations. They invert or shift the clock/bearing semantics that the OBS
line and the move token are graded on; using them would require rewriting
the text labels to match, which is possible but is its own project.

A second, independent augmentation axis is **engine-side variation**: board
and agent colors, agent/gold sizes, render size. That path changes what the
"true" frame looks like rather than degrading it, and needs nothing but
engine parameters — worth wiring into the data generator when convenient.
