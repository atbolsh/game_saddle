"""The planned FIRST self-training iteration -- a run script, not a library.

``training/train.py`` is deliberately generic: the loop and all its knobs
(:class:`training.train.TrainConfig`) know nothing about any particular run.
Each concrete run is a short script like this one -- pick the data sources,
pick the config, call :func:`run_training`. Copy this file per run/iteration
(``run_iter2.py``, ``run_replay_ablation.py``, ...); train.py itself should
not change between runs.

STATUS: template. The external replay sources and probes (phase 2) are live;
the game-trace ``DataSource`` and the graded jsonl batches it produces
arrive in the next stage -- until then the game line below stays commented
out, and the script as-is runs a REPLAY-ONLY smoke pass (useful for
verifying the phase-2 plumbing end to end on the remote box).

Prerequisite: ``python -m training.download_external`` (or a full
``bash scripts/setup_env.sh``) so data_external/ is materialized.

Run from the repo root::

    python -m training.run_first_iteration
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.external_data import sources_from_manifest  # noqa: E402
from training.probes import build_probe_hooks  # noqa: E402
from training.train import (  # noqa: E402
    DataSource,
    TrainConfig,
    configure_logging,
    run_training,
)

# --------------------------------------------------------------- the data
# Replay: every enabled dataset from training/datasets.json, each weighted
# so it contributes exactly its manifest examples_per_epoch
# (TRAINING_EXTRA_DATASETS.md documents each dataset's role and loss kind).
SOURCES: list[DataSource] = [
    *sources_from_manifest(),
    # Graded player generations from the self-eval loop (next stage;
    # TRAINING_GAME_TRACES.md has the volume rationale: 1-5k generations
    # per iteration, roughly half surviving grading). Uncomment when the
    # game-trace DataSource exists:
    # JsonlSource("data/game_traces_iter1.jsonl"),
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
