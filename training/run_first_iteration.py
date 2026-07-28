"""The planned FIRST self-training iteration -- a run script, not a library.

``training/train.py`` is deliberately generic: the loop and all its knobs
(:class:`training.train.TrainConfig`) know nothing about any particular run.
Each concrete run is a short script like this one -- pick the data sources,
pick the config, call :func:`run_training`. Copy this file per run/iteration
(``run_iter2.py``, ``run_replay_ablation.py``, ...); train.py itself should
not change between runs.

STATUS: template. The game-trace ``DataSource`` and the graded jsonl batches
it produces arrive in stage 2; until they exist the path below is a
placeholder and this script fails loudly at ``JsonlSource`` construction.

Run from the repo root::

    python -m training.run_first_iteration
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.train import (  # noqa: E402
    DataSource,
    JsonlSource,
    TrainConfig,
    configure_logging,
    run_training,
)

# --------------------------------------------------------------- the data
# Graded player generations from the self-eval loop (stage 2 produces these;
# TRAINING_GAME_TRACES.md has the volume rationale: 1-5k generations per
# iteration, roughly half surviving grading). Replay sources join in stage 3
# (TRAINING_EXTRA_DATASETS.md), e.g.:
#   JsonlSource("data/replay_arithmetic.jsonl", weight=0.4),
SOURCES: list[DataSource] = [
    JsonlSource("data/game_traces_iter1.jsonl"),
]

# ------------------------------------------------------------- the config
# Only deviations from the TRAINING_OVERVIEW.md recipe defaults belong here;
# everything unspecified stays at the documented default.
CONFIG = TrainConfig(
    label="iter1",
    # architecture=None -> MODEL_KEY from .env (gemma-4-12b).
    epochs=1,  # self-generated data is only on-policy the first pass
)

if __name__ == "__main__":
    configure_logging()
    raise SystemExit(run_training(SOURCES, CONFIG))
