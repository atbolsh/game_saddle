"""Unattended weekend self-training: N serial epochs of datagen -> train.

One "epoch" here is one full expert-iteration cycle. For epoch k (1-based):

  1. datagen   ``python -m training.generate_game_traces --label
     <prefix>_iter<k> --parallel 1 --checkpoint <previous epoch's adapter>``
  2. train     ``python -m training.run_weekend --train-iter <k>`` -- the
     epoch's GameTraceSource + AnalystTraceSource plus the manifest replay
     sources, resumed from the previous epoch's adapter (epoch 1 starts
     from bare HF weights unless --start-checkpoint is given).

Each stage is a SUBPROCESS, deliberately: the inference model is a
process-wide singleton (agent/model.py) and training builds its own
quantized PEFT copy, so one process cannot host both cleanly -- and a
crashed stage must not take the whole weekend down with it.

WEIGHTS SURVIVE EVERY EPOCH by construction: each train stage saves PEFT
adapters under ``weights/<arch>/<prefix>_iter<k>_step<N>/`` (periodic
save_steps saves plus a final save), and later epochs use fresh labels, so
nothing ever overwrites an earlier epoch's checkpoints.

Crash policy -- the run must survive an unattended weekend:

* A crashed datagen stage is retried once with ``--append`` and the
  REMAINING generation budget (player generations already on disk are
  counted from traces.jsonl -- one record per generation). Training then
  runs on whatever traces exist; an epoch with zero traces skips training
  and carries the previous checkpoint forward (logged at ERROR).
* A crashed train stage is retried once; if it fails again, the next
  epoch's datagen uses the PREVIOUS epoch's checkpoint (logged at ERROR,
  never silently -- see the no-fuzzy-fallbacks rule).
* Orchestrator restart: ``data_game/<prefix>_state.json`` records finished
  stages and per-epoch checkpoints; rerunning the same command skips
  completed work and finishes partial datagen via the --append path.

The exit code is the number of failed stages (0 = clean weekend).

Run on the remote box (NAMS up, repo root, inside tmux or nohup)::

    nohup python -m training.run_weekend > weekend.log 2>&1 &

Budget arithmetic before launching: selftest t8 measures the real serial
seconds-per-generation; one epoch costs about ``--max-generations`` x that,
plus the train stage. Trim ``--max-generations`` (and ``--games``) so
``epochs x (datagen + train)`` fits the window you have. Details in
TRAINING_OVERVIEW.md ("The weekend run").
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("run_weekend")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_GAME_DIR = REPO_ROOT / "data_game"

#: Attempts per stage (1 initial + 1 retry). A transient crash (NAMS
#: hiccup, CUDA OOM on an unlucky batch) should not cost the weekend; a
#: deterministic crash should not burn it in a loop either.
MAX_ATTEMPTS = 2


# ==================================================================== state

def _state_path(prefix: str) -> Path:
    return DATA_GAME_DIR / f"{prefix}_state.json"


def _load_state(prefix: str) -> dict:
    path = _state_path(prefix)
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        logger.info("resuming from %s: %s", path, state)
        return state
    return {"done": [], "checkpoints": {}}


def _save_state(prefix: str, state: dict) -> None:
    path = _state_path(prefix)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ================================================================== helpers

def _count_lines(path: Path) -> int:
    """Player generations already on disk: traces.jsonl has exactly one
    record per player generation."""
    if not path.is_file():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _latest_checkpoint(label: str) -> str | None:
    """Newest (highest-step) adapter checkpoint saved for ``label``.

    Exact, label-scoped lookup under ``weights/<arch>/``: train.py names
    every save ``<label>_step<N>`` and the final save is always the highest
    step, so max-N IS the run's final checkpoint (post-rollback if the
    guard machinery fired).
    """
    from agent.config import CONFIG  # torch-free

    root = Path(CONFIG.weights_dir) / CONFIG.model_key
    pattern = re.compile(rf"{re.escape(label)}_step(\d+)$")
    best_step, best_name = -1, None
    if root.is_dir():
        for d in root.iterdir():
            m = pattern.fullmatch(d.name)
            if m and (d / "adapter_config.json").is_file():
                if int(m.group(1)) > best_step:
                    best_step, best_name = int(m.group(1)), d.name
    return best_name


def _run_stage(cmd: list[str], stage: str) -> bool:
    """One subprocess attempt; True on exit 0. Output inherits our
    stdout/stderr so everything lands in the one weekend log."""
    logger.info("[%s] starting: %s", stage, " ".join(cmd))
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    hours = (time.perf_counter() - t0) / 3600
    if proc.returncode == 0:
        logger.info("[%s] finished in %.2fh", stage, hours)
        return True
    logger.error("[%s] FAILED (exit %d) after %.2fh",
                 stage, proc.returncode, hours)
    return False


# =================================================================== stages

def _datagen(k: int, checkpoint: str | None,
             args: argparse.Namespace) -> bool:
    """Datagen for epoch k, serial, resumable. True if any traces exist
    afterwards (training can proceed)."""
    label = f"{args.prefix}_iter{k}"
    traces = DATA_GAME_DIR / label / "traces.jsonl"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        done_gens = _count_lines(traces)
        remaining = args.max_generations - done_gens
        if done_gens and remaining <= 0:
            logger.info("[datagen%d] budget already spent (%d gens on "
                        "disk); nothing to do", k, done_gens)
            return True
        cmd = [
            sys.executable, "-m", "training.generate_game_traces",
            "--label", label,
            "--parallel", "1",
            "--games", str(args.games),
            "--max-generations", str(remaining),
            # Distinct noise/question streams per epoch AND per resume
            # attempt (a replayed stream would revisit the same frames).
            "--seed", str(args.seed + 100 * k + attempt),
        ]
        if checkpoint:
            cmd += ["--checkpoint", checkpoint]
        # Existence, not record count: a crash can leave an empty
        # traces.jsonl, and generate_game_traces refuses to run over an
        # existing file without --append.
        if traces.exists():
            cmd += ["--append"]
            logger.warning("[datagen%d] resuming: %d gens on disk, "
                           "%d remaining", k, done_gens, remaining)
        if _run_stage(cmd, f"datagen{k} attempt {attempt}"):
            return True
    n = _count_lines(traces)
    if n:
        logger.error("[datagen%d] gave up after %d attempts but %d "
                     "generations exist -- training on the partial epoch",
                     k, MAX_ATTEMPTS, n)
        return True
    logger.error("[datagen%d] gave up with ZERO traces -- skipping this "
                 "epoch's training", k)
    return False


def _train(k: int, resume: str | None, args: argparse.Namespace) -> str | None:
    """Train stage for epoch k (subprocess of this same module). Returns
    the new checkpoint name, or None if the stage ultimately failed."""
    label = f"{args.prefix}_iter{k}"
    cmd = [
        sys.executable, "-m", "training.run_weekend",
        "--train-iter", str(k), "--prefix", args.prefix,
    ]
    if resume:
        cmd += ["--resume-checkpoint", resume]
    if args.train_max_steps:
        cmd += ["--train-max-steps", str(args.train_max_steps)]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if _run_stage(cmd, f"train{k} attempt {attempt}"):
            ckpt = _latest_checkpoint(label)
            if ckpt:
                logger.info("[train%d] checkpoint: %s", k, ckpt)
                return ckpt
            # Exit 0 with no checkpoint on disk would mean train.py's save
            # contract broke -- do not paper over it, but do not stop the
            # weekend either.
            logger.error("[train%d] exited 0 but no %s_step* checkpoint "
                         "found under weights/ -- treating as failure",
                         k, label)
    logger.error("[train%d] gave up after %d attempts -- next epoch will "
                 "reuse checkpoint %r", k, MAX_ATTEMPTS, resume)
    return None


def train_one_epoch(k: int, prefix: str, resume: str | None,
                    max_steps: int | None = None) -> int:
    """The in-process train stage (child mode, --train-iter). Mirrors
    run_first_iteration.py: the epoch's game + analyst traces, the manifest
    replay sources, the capability probes, one pass over the data."""
    from training.external_data import sources_from_manifest
    from training.game_traces import AnalystTraceSource, GameTraceSource
    from training.probes import build_probe_hooks
    from training.train import TrainConfig, configure_logging, run_training

    configure_logging()
    label = f"{prefix}_iter{k}"
    sources = [
        GameTraceSource(f"data_game/{label}/traces.jsonl"),
        AnalystTraceSource(f"data_game/{label}/analyst_traces.jsonl"),
        *sources_from_manifest(),
    ]
    hooks, guards = build_probe_hooks()
    cfg = TrainConfig(
        label=label,
        epochs=1,  # self-generated data is only on-policy the first pass
        resume_checkpoint=resume,
        max_steps=max_steps,  # None = the full single pass
    )
    return run_training(sources, cfg, extra_hooks=hooks, extra_guards=guards)


# ============================================================= orchestrator

def orchestrate(args: argparse.Namespace) -> int:
    state = _load_state(args.prefix)
    checkpoint = args.start_checkpoint
    failures = 0

    for k in range(1, args.epochs + 1):
        # Replay this epoch's recorded outcome if it already finished.
        if f"train{k}" in state["done"]:
            recorded = state["checkpoints"].get(str(k))
            checkpoint = recorded or checkpoint
            logger.info("epoch %d already complete (checkpoint %r); "
                        "skipping", k, checkpoint)
            continue

        logger.info("=== epoch %d/%d (starting checkpoint %r) ===",
                    k, args.epochs, checkpoint)

        if f"datagen{k}" in state["done"]:
            logger.info("[datagen%d] already complete; skipping", k)
            have_traces = True
        else:
            have_traces = _datagen(k, checkpoint, args)
            if have_traces:
                state["done"].append(f"datagen{k}")
                _save_state(args.prefix, state)
            else:
                failures += 1

        new_ckpt = None
        if have_traces:
            new_ckpt = _train(k, checkpoint, args)
            if new_ckpt:
                checkpoint = new_ckpt
            else:
                failures += 1
        # train<k> is marked done even on failure: retrying it Monday by
        # hand beats burning the remaining weekend re-crashing.
        state["done"].append(f"train{k}")
        state["checkpoints"][str(k)] = new_ckpt
        _save_state(args.prefix, state)

    logger.info("weekend run complete: %d epoch(s), %d failed stage(s), "
                "final checkpoint %r", args.epochs, failures, checkpoint)
    return failures


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m training.run_weekend",
        description="Serial multi-epoch self-training loop "
                    "(datagen -> train per epoch, subprocess per stage)",
    )
    p.add_argument("--epochs", type=int, default=3,
                   help="expert-iteration cycles (default 3: Friday "
                        "afternoon -> Monday morning at ~20h/epoch serial "
                        "datagen, t8 2026-07-30; see TRAINING_OVERVIEW.md)")
    p.add_argument("--games", type=int, default=60,
                   help="games per datagen epoch (generate_game_traces "
                        "default)")
    p.add_argument("--max-generations", type=int, default=3000,
                   help="player-generation budget per datagen epoch; THE "
                        "knob for fitting the weekend (epoch datagen "
                        "hours ~= this x t8's s/gen / 3600)")
    p.add_argument("--seed", type=int, default=17,
                   help="base seed; each epoch and resume attempt derives "
                        "a distinct noise/question stream from it")
    p.add_argument("--prefix", default="weekend",
                   help="label prefix: data_game/<prefix>_iter<k>/, "
                        "checkpoints <prefix>_iter<k>_step<N>, state file "
                        "data_game/<prefix>_state.json")
    p.add_argument("--start-checkpoint", default=None,
                   help="adapter under weights/<arch>/ for epoch 1's "
                        "datagen + train to start from (default: bare HF "
                        "weights)")
    p.add_argument("--train-max-steps", type=int, default=None,
                   help="cap each train stage at N optimizer steps "
                        "(default: full pass); use a small N to smoke-test "
                        "the whole orchestration cheaply before the real "
                        "weekend")
    p.add_argument("--train-iter", type=int, default=None,
                   help="INTERNAL (child mode): run epoch k's train stage "
                        "in this process and exit")
    p.add_argument("--resume-checkpoint", default=None,
                   help="INTERNAL (child mode): adapter the train stage "
                        "resumes from")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    if args.train_iter is not None:
        return train_one_epoch(args.train_iter, args.prefix,
                               args.resume_checkpoint,
                               max_steps=args.train_max_steps)
    return orchestrate(args)


if __name__ == "__main__":
    sys.exit(main())
