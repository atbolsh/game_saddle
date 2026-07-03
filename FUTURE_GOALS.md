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

## 2. Interior walls and multi-gold levels — *Not started*

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

## 3. Audio and video modalities of Gemma 4 — *Not started*

Gemma 4 E4B natively supports audio input, and the Gemma 4 family
supports video. Future work:

* feed short audio instructions (e.g. a spoken "turn right") to the agent
  in mode 1, and record the audio as a message attachment;
* feed a video roll of the game (a sequence of frames) instead of a
  single still, and let the agent reason about motion;
* record those modalities into NAMS as `GameSnapshot`-style media nodes
  (audio/video bytes on disk, path + small preview on the node, mirroring
  the image-storage approach in `agent/image_store.py`).

Currently only image + text are used.
