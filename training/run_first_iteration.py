"""The planned FIRST self-training iteration -- a run script, not a library.

``training/train.py`` is deliberately generic: the loop and all its knobs
(:class:`training.train.TrainConfig`) know nothing about any particular run.
Each concrete run is a short script like this one -- pick the data sources,
pick the config, call :func:`run_training`. Copy this file per run/iteration
(``run_iter2.py``, ``run_replay_ablation.py``, ...); train.py itself should
not change between runs.

Prerequisites (both on the remote box):

    python -m training.download_external          # or bash scripts/setup_env.sh
    python -m training.generate_game_traces --label iter1

Run from the repo root::

    python -m training.run_first_iteration
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.external_data import sources_from_manifest  # noqa: E402
from training.game_traces import AnalystTraceSource, GameTraceSource  # noqa: E402
from training.probes import build_probe_hooks  # noqa: E402
from training.train import (  # noqa: E402
    DataSource,
    TrainConfig,
    configure_logging,
    run_training,
)

# --------------------------------------------------------------- the data
# The graded self-eval game traces are the POINT of the iteration (reward
# scheme + volume rationale: TRAINING_GAME_TRACES.md); the manifest replay
# sources ride along to keep general capabilities from washing out
# (TRAINING_EXTRA_DATASETS.md documents each dataset's role and loss kind).
# The analyst KD anchor pins analyst behavior to the frozen base while the
# player RL moves the shared weights; its auto-guarded per-source held-out
# KD loss (heldout_loss/analyst_iter1) is the analyst-drift meter.
SOURCES: list[DataSource] = [
    GameTraceSource("data_game/iter1/traces.jsonl"),
    AnalystTraceSource("data_game/iter1/analyst_traces.jsonl"),
    *sources_from_manifest(),
]

# ------------------------------------------------------------ the probes
# Exact-match capability probes (GSM8K + synthetic navigation) with
# higher-is-better regression guards; run after every checkpoint save.
PROBE_HOOKS, PROBE_GUARDS = build_probe_hooks()

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
    raise SystemExit(run_training(
        SOURCES, CONFIG,
        extra_hooks=PROBE_HOOKS, extra_guards=PROBE_GUARDS,
    ))
