# Extra uses of game traces — planted errors, prompt internalization

Non-standard uses of gameplay data: everything that is *derived from* traces
rather than being a straight (context, reply, grade) training example. The
standard pipeline is [TRAINING_GAME_TRACES.md](TRAINING_GAME_TRACES.md).

## Planted-error data

Take a player reply that is **correct** (as graded, or hand-picked) and
programmatically corrupt it, producing a reply with a known, labeled
mistake. **The generator is implemented:** [planted_errors.py](planted_errors.py)
(`scramble_player_reply(text, rng)` plants exactly ONE corruption per reply
and returns the change log; nothing imports it yet — usage stays deferred as
below). The generation techniques — deliberately cheap:

- **Clock-word shift:** grep the reply for clock words (`N o'clock`,
  matching the compass/clock convention the prompts use) and replace one
  chosen hour with a random different hour 1–12 — with the circular shift
  size recorded in the label, since a 1–2 hour shift is *within* the
  grading tolerance and must be labeled "acceptable", not "error".
- **Direction-word swap:** swap one direction word with its opposite
  (left/right, up/down, upper/lower, top/bottom, above/below,
  clockwise/counter-clockwise), case-preserving; compounds ("upper left")
  fall out of the word-level swap. A narrow guard skips `right` when it
  means "correct" ("the right move", "all right") — every swap lands in
  the change log, so residual misfires stay auditable.
- **Move-token scramble:** replace the final move token with another of the
  three, strip its brackets (the known bare-word format error), or delete
  it outright when a move was requested.

Each corrupted reply carries machine-readable ground truth: what was
changed, where (char span), and whether it is inside or outside grading
tolerance; an unchanged return means no target was found and the caller
must skip that reply. Exact usage is decided later. The two candidate
uses, in order of commitment:

1. **Measure analyst quality (committed).** Feed corrupted replies through
   the analyst phase and score: miss rate (planted error not flagged),
   false-positive rate (unchanged spans flagged), tolerance compliance
   (within-tolerance shifts not punished). This is the sycophancy/drift
   tripwire in the early-warning suite
   ([TRAINING_EXTRA_DATASETS.md](TRAINING_EXTRA_DATASETS.md)) — it works
   even though the analyst is never trained, because the *shared* network
   underneath it is.
2. **Analyst backprop under a different metric (open).** If the analyst is
   ever trained, planted-error catches are the natural training signal:
   the label is programmatic (no engine query at inference, no human), and
   "did you catch the planted error" cannot reward agreement. Whether and
   when to do this is an explicitly deferred decision.

Note the relationship to the engine-verification idea recorded in
[TRAINING_GAME_TRACES.md](TRAINING_GAME_TRACES.md): planted errors need *no
engine oracle at all* — the corruption itself is the ground truth — which is
why they survived the "no engine-derived baselines" rule.

## Prompt internalization (future discussion)

The system prompts are long: game rules, geometry conventions, move-token
protocol, grading calibration. Once traces exist in volume, the same data
can be used to **train the prompt away**: train on examples whose context
carries a *shortened* system prompt (or none) while the targets remain the
behavior produced under the *full* prompt. The model internalizes the
rules; the context budget and per-call latency shrink; the harness stops
being the only place the conventions live.

Recorded considerations for when this is discussed properly:

- Do it **late**, after behavior is stable — internalizing a prompt that is
  still being edited bakes in stale rules that can no longer be fixed by
  editing a string.
- Internalize in **layers** (game rules first, grading calibration last),
  keeping the ability to override by prompt at inference.
- The planted-error and fixed-board probes are the regression harness for
  it: internalization that degrades either is rejected.
- This is also the cheapest rehearsal for the repo's long-term goal
  (self-training toward environments where behavior cannot be
  prompt-engineered per task).

## Other derived uses (parking lot)

Ideas noted during planning, not scheduled: distilling debrief-mode search
strategies into examples (turning good multi-tool debrief transcripts into
training data for tool use); mining traces for hard boards (boards where
the player failed repeatedly) to oversample in later iterations; using
reflection messages as reasoning-style replay data.
