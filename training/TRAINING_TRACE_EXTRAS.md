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
   underneath it is. Datagen now records every analyst exchange's exact
   context to `analyst_traces.jsonl` (it feeds the analyst KD anchor of
   TRAINING_EXTRA_DATASETS.md), which is also the prerequisite for this
   probe — corrupted replies slot into recorded analyst prompts — and for
   any future analyst training.
2. **Analyst backprop under a different metric (open).** If the analyst is
   ever trained, planted-error catches are the natural training signal:
   the label is programmatic (no engine query at inference, no human), and
   "did you catch the planted error" cannot reward agreement. Whether and
   when to do this is an explicitly deferred decision.

Note the relationship to the engine-verification idea recorded in
[TRAINING_GAME_TRACES.md](TRAINING_GAME_TRACES.md): planted errors need *no
engine oracle at all* — the corruption itself is the ground truth — which is
why they survived the "no engine-derived baselines" rule.

## Prompt internalization and instinct moves (staged plan, not scheduled)

The end state (project owner's goal): **player behavior is the default** —
what the agent does with no system prompt at all — and moves are
**instinctive**: frame + terse request in, bare move token out, no OBS/
REASON prose unless observations are asked for. Normal Gemma chat and
analyst behavior survive in other circumstances.

This is two separate compressions with different risk profiles, and they
must not be conflated:

1. **Internalize the prompt** (context distillation): train on examples
   whose context carries a *shortened* system prompt (or none) while the
   targets remain the behavior produced under the *full* prompt. Behavior
   targets don't change at all — the low-risk half. The context budget and
   per-call latency shrink; the harness stops being the only place the
   conventions live.
2. **Internalize the reasoning** (implicit chain-of-thought): same
   decision, shorter output — just the move token. This is the risky half:
   the OBS/REASON tokens are causally useful forward passes spent reading
   the frame before committing, and stripping them costs accuracy until
   the network learns to do that work silently. It needs its own
   measurement at every step.

Note the mechanism is behavior distillation, NOT prompt recitation:
training the model to memorize the prompt *text* is neither necessary nor
sufficient for following it. What gets trained is the conditional
distribution "given this reduced context, produce the reply the full
prompt produced".

**The trigger contract.** "Default" needs a signal the network can
condition on, and the frame image is it: frame present + terse request →
player instinct; no frame → normal Gemma assistant. The analyst also sees
frames, so analyst mode stays *prompt-selected* (it only ever runs inside
the harness, which always supplies its prompt). This gives every stage two
standing tripwire probes: a text-only chat must NEVER emit a move token,
and an analyst-prompted call must still produce the RATING/WRONG
structure.

**The stages** (each ends with a probe pass and a human judgment; stages 2
and 3 train on *recorded* traces, so every ordinary datagen iteration run
before them silently builds their training set):

- **Stage 0 — gate.** Nothing below starts until the player is competent
  under the full prompt and the prompts have stopped changing:
  internalizing a prompt that is still being edited bakes in stale rules
  that can no longer be fixed by editing a string.
- **Stage 1 — instinct as a *prompted* behavior.** Add a move-request
  variant to datagen ("Your move. Emit only the move token, no
  commentary.") alongside the perception questions. The analyst grades
  those rounds like any other, so token-only data arrives on-policy and
  graded through the existing loop — zero new machinery — and the scary
  question ("how much worse are the moves without the reasoning?") gets
  measured per checkpoint before anything becomes a default.
- **Stage 2 — prompt internalization, layered.** A derived `DataSource`
  re-reads `traces.jsonl`, rewrites the `messages` context to drop or
  shorten the system prompt, and keeps `target_text` unchanged (this is
  exactly why records store the full exact `messages` — no regeneration
  needed). Internalize in layers — game rules first, geometry/move
  protocol next, grading calibration last or never (it belongs to the
  analyst) — mixed with heavy external replay, keeping the ability to
  override by prompt at inference.
- **Stage 3 — flip the default.** No system prompt, frame, terse request →
  token-only targets taken from stage-1/2 rounds that graded well (or won
  games only). The only stage where the *default* changes, done last, when
  board competence is at its peak and the regression baseline is longest.
  Contingency if move quality drops: one more datagen+train loop at the
  new default before judging.

Implementation notes recorded now so they aren't rediscovered: plain
weighted CE on recorded targets fits the current `train.py` contract with
no changes and is the starting point; KD in the "full-prompt teacher →
short-prompt student" form would need teacher and student to see
*different contexts*, which the current same-batch `disable_adapter()`
teacher path does not do — a real code change, only worth it if CE
disappoints. The planted-error and fixed-board probes are the regression
harness throughout: internalization that degrades either is rejected.
This is also the cheapest rehearsal for the repo's long-term goal
(self-training toward environments where behavior cannot be
prompt-engineered per task).

## Other derived uses (parking lot)

Ideas noted during planning, not scheduled: distilling debrief-mode search
strategies into examples (turning good multi-tool debrief transcripts into
training data for tool use); mining traces for hard boards (boards where
the player failed repeatedly) to oversample in later iterations; using
reflection messages as reasoning-style replay data.
