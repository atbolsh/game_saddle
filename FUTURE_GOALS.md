# Future Goals

This file tracks work that is intentionally out of scope for the current
milestone (the Gemma 4 E4B + NAMS game agent) but is a real objective for
later. Items are not commitments; they are reminders. The README's
"Notes / limitations" section points here.

Status legend: **Not started** / **Exploring** / **In progress**.

## 1. Automatic finetuning dataset generation — *Not started*

Combine mode 1 (game-playing traces, with the game image as context) and
mode 3 (self-evaluation verdicts over Conversations + Reasoning traces,
with the Settings dict available) to produce supervised-finetuning
datasets for Gemma 4 E4B. Each training example will be assembled from the
recorded `(:Message)-[:CAPTURED_STATE]->(:GameSnapshot)` pairs and the
linked reasoning traces, with the mode-3 verdict used as a quality
signal / label filter.

Concrete first steps when picked up:

* a Cypher export that, for a given session, returns the ordered
  `(before_image_path, question, action, gold_collected, verdict)` tuples;
* a `dataset.py` builder that materialises these into HF `datasets` rows
  (image + text instruction + target action);
* a quality filter that drops turns the mode-3 evaluator scored below a
  threshold.

## 2. Interior walls and multi-gold levels — *Exploring*

Generalise the level generator past `random_bare_settings` (4 boundary
walls + 1 gold piece). Targets:

* `discreteGame.random_settings(...)` with `num_extra_walls > 0`,
* multiple gold pieces (`typical_max_gold_num > 1`),
* the angle-restricted variant (`restrict_angles=True`) for cleaner
  autoencoder pretraining.

This requires the agent to handle obstacles (wall-avoidance) and
sequential gold collection. The `ACTION_MAP` and per-move recording
machinery in `agent/game_io.py` and `agent/modes.py` already generalise;
only interior walls (and an engine-native walk-out terminal) remain.

**2026-08-12 notebook-only cut:** `agent/game_io.new_multi_gold_game` +
`boundary_openings` (non-AI oracle of boundary gaps) +
`MultiGoldSelfEvalSession` (`notebooks/multi_gold_eval.ipynb`) exercise
0–3 golds, sealed vs open rooms, target commitment (`TARGET:` line),
walking out an opening, and `[END_GAME]` when the room is sealed and
empty. **2026-08-25:** training datagen uses that factory (`--multi-gold`);
weekend smoke stays sealed one-gold. See goal 10.

## 3. Audio and video modalities of Gemma 4 — *Exploring*

**Measurement harness written (2026-08-12), first run pending.**
`training/eval_av.py` scores a named checkpoint against the frozen
Gemma 4 12B base on LibriSpeech test-clean (WER) and NExT-QA multiple
choice (letter accuracy), using the standard processor chat-template
parts `{"type": "audio", "path": ...}` / `{"type": "video", "path": ...}`
(video via `num_frames` / `do_sample_frames`). Whether the 12B accepts
both modalities through this path is settled by the script's first
remote run — a processor rejection there is a finding, not something to
route around.

Still future work:

* feed short audio instructions (e.g. a spoken "turn right") to the agent
  in mode 1, and record the audio as a message attachment;
* feed a video roll of the game (a sequence of frames) instead of a
  single still, and let the agent reason about motion;
* record those modalities into NAMS as `GameSnapshot`-style media nodes
  (audio/video bytes on disk, path + small preview on the node, mirroring
  the image-storage approach in `agent/image_store.py`);
* enable the video/audio KD **replay** placeholders in
  `training/datasets.json` — independent of all of the above (replay needs
  no game-harness connection). Deliberately sidelined for the first
  training rounds: the vision/audio towers are LoRA-frozen, so drift risk
  is low, and enabling costs converters + a fresh VRAM profile (rationale
  in `training/TRAINING_EXTRA_DATASETS.md`). The eval script does not
  unblock that.

Currently only image + text are used **in the agent loop**; audio/video
comprehension is measurable off to the side.

## 4. Analyst revamp: CORRECT spans + per-span ratings — *Not started*

Today the analyst emits one message-level `RATING:` plus verbatim `WRONG:`
span quotes; the reward mapping (`training/game_traces.py`) turns that into
per-token weights. The revamp would make the signal denser and more
surgical:

* a `CORRECT:` span mark beside `WRONG:` (praise specific reasoning, not
  just flag mistakes);
* per-span ratings instead of (or refining) the single message-level
  rating;
* an **arousal / novelty axis** beside quality (see goal 5): the analyst
  rates "how surprising or consequential was this moment?" separately
  from "was this a good move?", so a locally-sensible step inside an
  endless loop stops earning full marks. The 2026-08-04 retest showed
  exactly this failure: each individual "rotate toward the gold" move
  looked reasonable to a per-move rater, so a spin-loop was rated +1.0
  move after move.

Deferred as a multi-day project (new analyst prompt, new parser, new
verification harness, new weight mapping, regenerated data). **Trigger to
pick it up:** perception/move quality stalls across training iterations
while analyst ratings stay high — the signature of the message-level
rating being too blunt (see the reward-scheme section of
`training/TRAINING_GAME_TRACES.md`).

## 5. Novelty / arousal reward shaping — *Exploring*

A first, deliberately crude cut ships in `training/game_traces.py`
(`NoveltyTracker`, marked WORK IN PROGRESS): the k-th consecutive
identical move keeps only `0.9^k` of its cloning weight (floored at 0.1,
"unrewarded" rather than "unlearned"), so a repeated-move loop stops
feeding on itself. Its companion `ACTION_BALANCE` (per-action
inverse-frequency reweighting) is an explicitly TEMPORARY hack
(screaming comments in situ) and is NOT part of this goal — it must be
removed, not refined.

The biological framing worth building toward: animals carry a general
**arousal** mechanism — vagus-nerve-mediated fear responses, and the
analogous surge after large positive rewards — that sharpens learning for
whatever happened *shortly before* a highly surprising, threatening, or
rewarding event. A survived near-miss or a big find is consolidated far
more thoroughly than a routine step; conversely, "boredom" keeps us from
getting excited about every step we take. Both directions belong in the
loop, not just the decay:

* **positive arousal**: upweight the moves leading into gold pickups,
  wins, and narrow escapes (the win boost in `example_scale` is a
  primitive version of this — it could become event-triggered and
  magnitude-scaled);
* **state-aware repeat detection**: the current tracker only catches
  identical *consecutive* moves; alternating CLOCK/ANTICLOCK oscillation
  or a revisited `(position, heading)` sails through — hash the state,
  not just the last action;
* **per-game action-entropy bonuses** as a softer alternative to streak
  decay;
* **data-sampling side**: rather than (only) reweighting after the fact,
  datagen-time rejection sampling / deduplication of low-novelty records
  shapes what enters the corpus at all — a natural companion to the
  analyst revamp's arousal axis (goal 4), which would let the rater judge
  novelty from inside the trace;
* **rare-move rewards**: a graded, novelty-integrated bonus for actions
  underrepresented in the current corpus — the principled successor to
  the blunt equal-mass `ACTION_BALANCE` reweighting, which assumes all
  move types deserve identical total mass and cannot survive into
  environments where they genuinely differ in importance.

**2026-08-11 finding (aug6 11-epoch run): the residual failure is
TRANSITIONS, and it is measurable.** The move-repetition attractor
survived the reward rework in a milder form: within-game turn direction
is 97% self-continuing by the late epochs (flip rate fell 0.10 → 0.03
across the run — training *deepened* the commitment; p90 same-direction
runs of ~40 turns = full orbits), FORWARD runs are 83% self-continuing
with ~90% of aborts geometrically justified, and both continuation
behaviors are locally right. What the model gets wrong is *when to
switch*: given a ray-hit (oracle says FORWARD), it starts walking only
~50% of the time, spinning straight past the alignment it just earned —
the single biggest leak between its ~10% win rate and actually winning.
Those decisive transition moves are a handful per game, drowned in
~2,600 correct-continuation moves per corpus. Concrete next cut:
**up-weight transition moves** (first FORWARD of a run on a ray-hit
round, first turn after aim is lost — the oracle meta already labels
both) via `example_scale`, and switch the novelty decay ON so the k-th
consecutive identical turn stops earning full cloning weight. This slots
under the existing arousal framing: a transition IS the surprising,
consequential moment worth consolidating.

## 6. Trust region as a CONSTRAINT: additive per-token KL — *Not started*

Today the trust region is a *dataset*: `PlayerAnchorSource` replays player
traces at weight 0.25 with `loss="kd_anchor"` (teacher = the parent
checkpoint). That bounds drift only as strongly as its share of the
gradient mass — it competes with the task loss instead of constraining it.
The industry-standard form is an **additive per-token KL penalty computed
on the same batch**:

    loss = task_loss + lambda * KL(student || parent)

evaluated on the game batch's own tokens via the existing
`_base_model_logits(teacher="anchor")` machinery — no extra data pass, and
the anchor becomes a constraint on *every* update. Start at
`lambda ≈ 0.05–0.1`, or use an adaptive controller targeting a KL budget
(trl's adaptive KL targets a few nats per response). Two cautions,
recorded when this was deferred (2026-08-05): the CARE-style result that
KL anchors can *backfire* by pinning the model to whatever the parent
already does (checkpoint selection — already in place — is the
complementary defense), and lambda interacts with the LR (retune after
the 3e-6 drop, not before).

## 7. GRPO / group-relative on-policy RL — *Not started*

The current trainer is single-sample offline REINFORCE: one graded reply
per position, weights = shaped advantage, no importance correction, one
epoch per batch. The natural upgrade once the loop is stable is
**GRPO-style group sampling**: generate K replies per position, compute
advantages *relative to the group* (no learned critic needed — the group
mean is the baseline), and update with ratio clipping. What it buys here:

* the group baseline solves analyst saturation *structurally* — even if
  every reply gets 0.9+, the within-group ranking still carries signal
  (the exp-advantage-vs-corpus-mean mapping in `game_traces.py` is the
  poor man's version of exactly this);
* on-policy sampling kills the off-policy staleness caveat entirely;
* published Gemma-family recipes exist to copy (Google Tunix GRPO at
  lr 3e-6 — the same scale the trainer now uses).

Cost: K× the datagen per position and a harness rework (the dispatcher
already batches, but the trace format assumes one reply per round).
Prerequisite: a reward cheaper than the full analyst call per sample, or
K analyst calls accepted as the price.

## 8. NAMS entity-extraction pollution — *Done* (2026-08-25)

**2026-08-13 finding (July session dumps + aug11/aug12 reset censuses):**
NAMS ran spaCy NER over every message, and spaCy on geometry/clock-face
prose produced steady junk entities: `~118 degrees` as a Person, `CLOCK` /
`CLOCKWISE` / `~4:20` as Organizations, `\approx` / `OBS` / `AI` as
Geopolitical Locations, `radians` as an Organization+Group, LaTeX sliced
mid-expression (`\text{atan2}(-0.0204` as an Organization), clock-position
phrases (`4:18 o'clock`) as Event+Time. Volume: ~6,000 junk entities per
60-game datagen epoch (~3.5–4k `Entity+Object`, ~1.9k `Entity+Event+Time`,
~130 Person / ~150 Organization / ~110 Location), roughly 2 per
generation. Character: almost all were single-mention orphans (~1
`MENTIONS` edge each — unique numeric strings, not shared references),
and where strings DID repeat, resolution failed to merge case/label
variants (4 `agent` nodes shadowing the seeded semantic `Agent`; `OBS`
under two label sets). Everything sat at spaCy's default ~0.85
confidence, so thresholding was not a lever.

**Done:** `make_memory_settings` sets `extractor_type=ExtractorType.NONE`
(and `enable_llm_fallback=False`). Messages are stored without auto-NER;
the long-term graph is the five seeded entities plus Preference rows.
spaCy/GLiNER weights still download via `scripts/setup_env.sh` so a later
re-enable does not stall mid-run. `reset_memory_to_seed` still purges
leftover extracted `Entity` nodes from older graphs.

Do not turn extraction back on as the discovery path for novel in-game
objects — that is goal 12. Reasoning traces are unchanged.

## 9. Debrief notepad injection — *Not started*

`_BLOCK_REVIEW_WHOLE_REPLY` (shared by the scene analyst and debrief) now
tells the reviewer to check `[REMEMBER ...]` lines against "the notepad
shown in your context". `build_scene_analyst_messages` injects that
notepad; `build_debrief_messages` does not. In debrief the instruction
is therefore dangling. Harmless today (debrief reviews recorded play,
not live `[REMEMBER]` spam), but if debrief verdicts ever complain
about a missing notepad, either inject the recorded notepad into the
debrief builder or gate that one sentence out of the debrief
composition.

## 10. Align training with multi-gold / target-exit play — *In progress*

The 2026-08-19 mode unification put target commitment, openings, and
`[END_GAME]` into every player/analyst prompt. **2026-08-25 training
datagen cut:** `run_weekend` datagen passes `--multi-gold`
(`MultiGoldSelfEvalSession`, `n_gold=None` → uniform {0,1,2,3},
`opening="any"`, `end_on_clear=False`). Eating gold does not end the
game. A win is a correct `[END_GAME]` on a sealed empty board, or walking
out so the agent disc contacts the unit-square boundary with no gold
remaining (`discreteGame.agent_exited`; datagen queries it after
`end_round()`). The oracle aims at remaining gold, then at opening
centers, then `END_GAME`, using the engine-validated `TARGET:` line;
`oracle_verdict` grades a quit as correct/wrong instead of unknown.
Analyst prompts already include the `openings` dict entry; player context
still strips it.

Weekend **smoke** is unchanged: sealed one-gold, eat-gold win, so those
win rates stay comparable to earlier runs. t5/t8/t9 stay on that default
too.

Still open:

* action-balance still pins `[END_GAME]` at 1.0 (inverse-frequency would
  amplify a handful of quits);
* walk-out is `discreteGame.agent_exited` (disc contact with the
  unit-square boundary; sealed rooms cannot fire it because boundary
  walls are thicker than `agent_r`);
* drop the player-facing "END_GAME is not available in this mode" notice
  on the sealed smoke path (still correct there).

## 11. Agent-written core tips (the "always after" plan) — *Not started*

A future tool will let the player and the analyst write their own core
tips into the same substrate the system prompts load from
(`core_player_*` / `core_analyst_*` Preference rows, assembled by
category-sort). The TOOL, never the agent, assigns the category name:
role prefix (`core_player_` / `core_analyst_`) + a number in the reserved
500+ range + a short slug. Because assembly is category-sort, agent-
written tips therefore always land AFTER every seeded block (seeded
numbers stop at 140; t1 enforces < 500), in a deterministic, reproducible
position, with zero code changes. Landing before the seeded blocks
(numbers below 010) is deliberately possible under the same sort
mechanism but not the default.

Analyst-authored tips get `tag_analyst_text` applied by the tool, so both
gates (category prefix and `[ANALYST]` text scrub) hold; player-authored
tips stay untagged.

`get_core_tips` already keeps well-formed extras in the 500+ range.
The remaining work is the writing tool itself.

Trigger to pick it up: player-written `tip_learned_*` tips seeing real
use (2026-08-19 discussion).

## 12. Deliberate agent-saved memory — *Not started*

Auto-NER is off (goal 8). The session notepad (`[REMEMBER key: …]`) is
session-scoped, exact-keyed, and unembedded — it never enters
`get_context` similarity recall. Future tasks will introduce unseen
in-game objects; the player must be able to **commit** a fact to
preserve it, and those writes should be gradeable and schedulable into
training.

Do not turn spaCy/GLiNER extraction back on as the discovery path.
Novel objects appear on the **frame** first; NAMS extraction only reads
**message text**, and a malleable GLiNER schema only lists *kinds*, not
instances. The house pattern is already `[REMEMBER target:]`: a
parseable token in the reply, the harness writes an exact row, the
analyst grades it against Settings.

When picked up:

* a parseable write token in the same family as `[REMEMBER]` (the
  **tool**, not the model, assigns the graph key — same rule as goal 11
  for tips);
* persist via `long_term.add_entity` when the fact should survive
  sessions and be `[SEARCH]`-able; keep `SessionNote` for this-game
  scratchpad. Preferences stay tips, not objects;
* type catalog stays code-seeded (new *kinds* are a seed/schema edit);
  new *instances* are saves;
* training is the REMEMBER path: the write is in the reply, the analyst
  has Settings/oracle, WRONG/RATING on the span, traces become the
  rows. Unverified NER nodes must not become CE/KD data.
