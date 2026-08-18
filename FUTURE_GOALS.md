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
only the level-creation call and the stop condition
(`gold_remaining == 0`) need widening.

**2026-08-12 notebook-only cut:** `agent/game_io.new_multi_gold_game` +
`boundary_openings` (non-AI oracle of boundary gaps) +
`MultiGoldSelfEvalSession` (`notebooks/multi_gold_eval.ipynb`) exercise
0–3 golds, sealed vs open rooms, target commitment (`TARGET:` line),
walking out an opening, and `[END_GAME]` when the room is sealed and
empty. The existing self-eval / datagen critical path is unchanged
(`[END_GAME]` is not in `game_io.ACTIONS`). Candidate future datagen
mode once the notebook behavior looks right — do not wire it into
`generate_game_traces` until then.

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

## 8. NAMS entity-extraction pollution + retrieval value — *Not started*

**2026-08-13 finding (July session dumps + aug11/aug12 reset censuses):**
NAMS runs spaCy NER over every message, and spaCy on geometry/clock-face
prose produces steady junk entities: `~118 degrees` as a Person, `CLOCK` /
`CLOCKWISE` / `~4:20` as Organizations, `\approx` / `OBS` / `AI` as
Geopolitical Locations, `radians` as an Organization+Group, LaTeX sliced
mid-expression (`\text{atan2}(-0.0204` as an Organization), clock-position
phrases (`4:18 o'clock`) as Event+Time. Volume: ~6,000 junk entities per
60-game datagen epoch (~3.5–4k `Entity+Object`, ~1.9k `Entity+Event+Time`,
~130 Person / ~150 Organization / ~110 Location), roughly 2 per
generation. Character: almost all are single-mention orphans (~1
`MENTIONS` edge each — unique numeric strings, not shared references),
and where strings DO repeat, resolution fails to merge case/label
variants (4 `agent` nodes shadowing the seeded semantic `Agent`; `OBS`
under two label sets). Everything sits at spaCy's default ~0.85
confidence, so thresholding is not a lever. The datagen run-start hygiene
reset already deletes all of it (census lines in the weekend logs), so
this is contained-per-run pollution, not accumulating pollution.

When picked up, in order of leverage:

* **kill it at the source, not at reset time**: disable or restrict
  NAMS's NER extraction for game-session messages (NAMS extraction
  config — not this repo's code);
* **measure retrieval value first**: the pollution's practical cost is
  the entity tier serving `~4:20 (Organization)` to a semantic query
  about clock bearings mid-run. Datagen traces already record every
  `[SEARCH]` call and its results (`searches` in the trace meta), so
  "is NAMS pulling its weight" is answerable offline before changing
  anything — that reading should gate how much effort the rest gets;
* **resolution case/label merging** is real but minor (dozens of nodes,
  not thousands).

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
