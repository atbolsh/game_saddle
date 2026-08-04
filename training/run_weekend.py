"""Unattended weekend self-training: N sequential epochs of datagen -> train.

One "epoch" here is one full expert-iteration cycle. For epoch k (1-based):

  1. datagen   ``python -m training.generate_game_traces --label
     <prefix>_iter<k> --parallel <--parallel, default 16> --checkpoint
     <previous epoch's adapter>`` (``--parallel 1`` restores the fully
     serial datagen path)
  2. train     ``python -m training.run_weekend --train-iter <k>`` -- the
     epoch's GameTraceSource + PlayerAnchorSource (the trust region,
     anchored to the previous epoch's adapter) + AnalystTraceSource plus
     the manifest replay sources, resumed from the previous epoch's
     adapter (epoch 1 starts from bare HF weights unless
     --start-checkpoint is given).
  3. smoke eval  8 real games with the FRESH checkpoint through the same
     datagen path (label ``<prefix>_smoke<k>``, never trained on): win
     rate, mean/min rating, degeneracy fraction, and mean gold-distance
     delta are logged and stored under ``smoke`` in the state file --
     every checkpoint gets a game-performance reading even when no
     further datagen follows it, and a poisoned checkpoint surfaces in
     ~15 min. ``grep smoke_eval`` on the run log for the morning review.

Each stage is a SUBPROCESS, deliberately: the inference model is a
process-wide singleton (agent/model.py) and training builds its own
quantized PEFT copy, so one process cannot host both cleanly -- and a
crashed stage must not take the whole weekend down with it.

WEIGHTS SURVIVE EVERY EPOCH by construction: each train stage saves PEFT
adapters under ``weights/<arch>/<prefix>_iter<k>_step<N>/`` (periodic
save_steps saves plus a final save), and later epochs use fresh labels, so
nothing ever overwrites an earlier epoch's checkpoints. The checkpoint the
next epoch RESUMES FROM is the ``last_good_checkpoint`` from the train
stage's ``done`` event -- NOT the highest-step directory on disk: train.py
saves the final checkpoint BEFORE the final eval, so when that eval
hard-regresses and rolls back, the rejected save is the highest step on
disk (the 2026-08-04 retest handed exactly such a rejected checkpoint to
epoch 2; ``_train_result_checkpoint``).

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
* STOP-ON-POISON is the one deliberate exception to "keep marching": a
  datagen or smoke-eval stage exiting with code 3 means the degeneracy
  fuse tripped (generate_game_traces: the checkpoint emits gibberish with
  no parseable ratings or moves). That is deterministic, not transient --
  every later epoch would train on garbage -- so the orchestrator logs
  the broken checkpoint at ERROR, records it under ``poisoned`` in the
  state file, and STOPS the entire run. No retry, no later epochs.

The exit code is the number of failed stages (0 = clean weekend; a
poisoned stop adds 1).

MONITORING: the orchestrator samples topline GPU memory (``nvidia-smi``,
summed over GPUs) every minute, tagged with the running stage, appending
to ``data_game/<prefix>_vram.jsonl``; the final log prints a VRAM summary
(peak overall, mean during datagen, mean during training) plus total
datagen and train hours. If ``nvidia-smi`` is unusable the monitor warns
ONCE and stops -- monitoring must never take down the weekend.

Run on the remote box (NAMS up, repo root, inside tmux or nohup)::

    nohup python -m training.run_weekend > weekend.log 2>&1 &

Budget arithmetic before launching: one datagen epoch costs about
``--max-generations`` x the seconds-per-generation of your ``--parallel``
setting, plus the train stage. Measured 2026-07-30: serial 24.1 s/gen
(selftest t8), ``--parallel 10`` 8.5, ``--parallel 24`` 6.4 -- so the
default 3000-generation epoch is ~20 h serial but ~6 h at the default
parallelism. The train stage on that corpus measured ~2 h + setup
(selftest t10, 2026-07-31; pad ~10-15% for save-time eval hooks). Trim
``--max-generations`` (and ``--games``) so ``epochs x (datagen + train)``
fits the window you have. Details in TRAINING_OVERVIEW.md ("The weekend
run").
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.generate_game_traces import EXIT_POISONED  # noqa: E402

logger = logging.getLogger("run_weekend")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_GAME_DIR = REPO_ROOT / "data_game"

#: Attempts per stage (1 initial + 1 retry). A transient crash (NAMS
#: hiccup, CUDA OOM on an unlucky batch) should not cost the weekend; a
#: deterministic crash should not burn it in a loop either.
MAX_ATTEMPTS = 2

#: Seconds between VRAM samples (plus one extra sample at every stage
#: boundary, so even sub-minute smoke-test stages appear in the log).
VRAM_SAMPLE_INTERVAL_S = 60.0


# ============================================================ VRAM monitor

class VramMonitor:
    """Every-minute topline GPU-memory samples, from the orchestrator.

    The orchestrator never touches the GPU itself (both stages are
    subprocesses), so samples come from ``nvidia-smi`` and reflect the
    box's true topline: ``memory.used`` summed over all GPUs, in MiB.
    Each sample is appended immediately to
    ``data_game/<prefix>_vram.jsonl`` (one JSON object per line:
    ``{"t": unix_time, "stage": "datagen1 attempt 1"|"train2 ..."|null,
    "mib": N}``; stage null = between stages) so a crashed run still has
    its trace. :meth:`finish` logs the summary: peak overall, mean during
    datagen, mean during training.

    If ``nvidia-smi`` is missing or unparseable the monitor logs ONE
    warning and stops sampling -- monitoring must never take down the
    weekend, and a silent zero would be worse than no data
    (visible-fallback rule).
    """

    def __init__(self, prefix: str):
        self.path = DATA_GAME_DIR / f"{prefix}_vram.jsonl"
        self.samples: list[tuple[str | None, int]] = []
        self._stage: str | None = None
        self._warned = False
        self._file_dead = False
        self._stopping = False
        self._wake = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="vram-monitor", daemon=True
        )
        self._thread.start()

    def set_stage(self, stage: str | None) -> None:
        self._stage = stage
        self._wake.set()  # immediate boundary sample

    def _read_mib(self) -> int | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode != 0:
                raise RuntimeError(out.stderr.strip()
                                   or f"exit {out.returncode}")
            return sum(int(line.strip())
                       for line in out.stdout.splitlines() if line.strip())
        except Exception as exc:
            if not self._warned:
                self._warned = True
                logger.warning("VRAM monitor DISABLED (nvidia-smi failed: "
                               "%s) -- no VRAM log for this run", exc)
            return None

    def _loop(self) -> None:
        while not self._stopping:
            mib = self._read_mib()
            if mib is None:
                return
            stage = self._stage
            self.samples.append((stage, mib))
            if not self._file_dead:
                # A dead trace file (disk full, permissions) must not
                # stop in-memory sampling: the end-of-run summary still
                # works. One WARNING, then stop writing.
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "t": round(time.time(), 1),
                            "stage": stage, "mib": mib,
                        }) + "\n")
                except OSError as exc:
                    self._file_dead = True
                    logger.warning("VRAM trace file %s unwritable (%s) -- "
                                   "keeping in-memory samples only",
                                   self.path, exc)
            self._wake.wait(timeout=VRAM_SAMPLE_INTERVAL_S)
            self._wake.clear()

    def finish(self) -> None:
        """Stop sampling and log the summary."""
        self._stopping = True
        self._wake.set()
        self._thread.join(timeout=15)
        if not self.samples:
            logger.warning("VRAM summary: no samples recorded")
            return

        def _mean_gib(kind: str) -> float | None:
            vals = [m for s, m in self.samples if s and s.startswith(kind)]
            return (sum(vals) / len(vals) / 1024.0) if vals else None

        peak_stage, peak = max(self.samples, key=lambda sm: sm[1])
        parts = [f"peak {peak / 1024.0:.1f} GiB "
                 f"(during {peak_stage or 'idle'})"]
        for kind in ("datagen", "train"):
            mean = _mean_gib(kind)
            parts.append(f"mean {kind} {mean:.1f} GiB" if mean is not None
                         else f"no {kind} samples")
        trace = ("(trace file was unwritable; in-memory samples only)"
                 if self._file_dead else f"full trace: {self.path}")
        logger.info("VRAM summary (%d samples, 1/min): %s -- %s",
                    len(self.samples), "; ".join(parts), trace)


#: Set by orchestrate(); _run_stage tags the monitor + stage-hours ledger.
_MONITOR: VramMonitor | None = None

#: Total wall hours per stage kind, across epochs AND retry attempts
#: (time spent is time spent, success or not). Logged at the end of the
#: run next to the VRAM summary.
_STAGE_HOURS: dict[str, float] = {"datagen": 0.0, "train": 0.0}


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


def _train_result_checkpoint(label: str) -> str | None:
    """The checkpoint a finished train stage actually stands behind: the
    ``last_good_checkpoint`` field of the ``done`` event in the newest
    ``logs/train_<label>_*/events.jsonl``.

    WHY NOT the highest-step directory under weights/: train.py saves the
    final checkpoint BEFORE the final eval. When that eval hard-regresses,
    the guard rolls the in-memory weights back to last_good, but the
    rejected save stays on disk as the highest step -- the 2026-08-04
    retest promoted exactly such a rejected checkpoint into epoch 2's
    datagen. The done event is the trainer's verdict; a healthy final
    eval makes ``last_good_checkpoint`` the final save anyway.

    A missing run dir / events file / done event / checkpoint field
    returns None (the caller logs and treats the stage as failed) --
    never fall back to a highest-step guess (no-fuzzy-fallbacks)."""
    # Timestamp suffixes (%Y-%m-%d_%H-%M-%S) sort lexicographically ==
    # chronologically, so the last glob match is the newest attempt.
    run_dirs = sorted((REPO_ROOT / "logs").glob(f"train_{label}_*"))
    if not run_dirs:
        logger.error("no logs/train_%s_* run directory found -- cannot "
                     "read the train stage's verdict", label)
        return None
    events_path = run_dirs[-1] / "events.jsonl"
    done: dict | None = None
    try:
        with open(events_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("kind") == "done":
                    done = record
    except (OSError, json.JSONDecodeError) as exc:
        # A stage failure (loud, retried/carried by the caller), never an
        # orchestrator crash -- the weekend must survive a mangled log.
        logger.error("cannot read %s (%s)", events_path, exc)
        return None
    if done is None:
        logger.error("%s has no 'done' event -- the train stage never "
                     "finished cleanly", events_path)
        return None
    ckpt = done.get("last_good_checkpoint")
    if not ckpt or ckpt == "None":
        logger.error("done event in %s has no usable last_good_checkpoint "
                     "(got %r)", events_path, ckpt)
        return None
    return Path(ckpt).name


def _run_stage(cmd: list[str], stage: str) -> int:
    """One subprocess attempt; returns the exit code (0 = success; the
    callers care about EXIT_POISONED = 3 specifically). Output inherits
    our stdout/stderr so everything lands in the one weekend log. Also
    brackets the VRAM monitor's stage tag and books the wall time into
    the per-kind hours ledger (smoke evals run the datagen path and book
    as datagen)."""
    logger.info("[%s] starting: %s", stage, " ".join(cmd))
    kind = ("datagen" if stage.startswith(("datagen", "smoke"))
            else "train")
    if _MONITOR is not None:
        _MONITOR.set_stage(stage)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT)
    finally:
        hours = (time.perf_counter() - t0) / 3600
        _STAGE_HOURS[kind] += hours
        if _MONITOR is not None:
            _MONITOR.set_stage(None)
    if proc.returncode == 0:
        logger.info("[%s] finished in %.2fh", stage, hours)
    else:
        logger.error("[%s] FAILED (exit %d) after %.2fh",
                     stage, proc.returncode, hours)
    return proc.returncode


def _summarize_traces(label: str) -> dict:
    """Game-performance summary of ``data_game/<label>/traces.jsonl``:
    games/wins (a game = one distinct (session_id, game_index)), mean/min
    analyst rating, the rating-null fraction (the degeneracy meter -- a
    healthy run sits near 0), and the mean per-move gold-distance delta
    (positive = the player closes in on the gold; rating-independent
    quality cross-check, recorded by generate_game_traces)."""
    path = DATA_GAME_DIR / label / "traces.jsonl"
    games: dict[tuple, bool] = {}
    ratings: list[float] = []
    deltas: list[float] = []
    n_records = 0
    n_rating_null = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            meta = json.loads(line)["meta"]
            n_records += 1
            key = (meta.get("session_id"), meta.get("game_index"))
            games[key] = games.get(key, False) or bool(meta.get("game_won"))
            rating = meta.get("rating")
            if rating is None:
                n_rating_null += 1
            else:
                ratings.append(float(rating))
            before = meta.get("dist_to_gold_before")
            after = meta.get("dist_to_gold_after")
            if before is not None and after is not None:
                deltas.append(before - after)
    wins = sum(games.values())
    return {
        "generations": n_records,
        "games": len(games),
        "wins": wins,
        "win_rate": round(wins / len(games), 3) if games else None,
        "mean_rating": (round(sum(ratings) / len(ratings), 3)
                        if ratings else None),
        "min_rating": min(ratings) if ratings else None,
        "rating_null_fraction": (round(n_rating_null / n_records, 3)
                                 if n_records else None),
        "mean_dist_delta": (round(sum(deltas) / len(deltas), 4)
                            if deltas else None),
    }


# =================================================================== stages

#: Post-train smoke eval size: 8 real games (~100-200 generations,
#: ~10-20 min at --parallel 16) -- enough for a win-rate/rating/degeneracy
#: reading on every fresh checkpoint, cheap enough to run unconditionally.
SMOKE_GAMES = 8
SMOKE_MAX_GENERATIONS = 200


def _smoke_eval(k: int, checkpoint: str | None,
                args: argparse.Namespace) -> str:
    """Post-train sanity check on epoch k's fresh checkpoint: a tiny run
    through the REAL datagen path (so the degeneracy fuse and distance
    recording apply) under the separate label ``<prefix>_smoke<k>`` --
    smoke traces never enter a training corpus. Returns ``"ok"`` /
    ``"poisoned"`` (fuse tripped -> stop the run) / ``"failed"`` (crash:
    a missing reading, logged at ERROR, but NOT a failed epoch -- the
    checkpoint may still be fine)."""
    label = f"{args.prefix}_smoke{k}"
    traces = DATA_GAME_DIR / label / "traces.jsonl"
    cmd = [
        sys.executable, "-m", "training.generate_game_traces",
        "--label", label,
        "--parallel", str(args.parallel),
        "--games", str(SMOKE_GAMES),
        "--max-generations", str(SMOKE_MAX_GENERATIONS),
        "--seed", str(args.seed + 100 * k + 51),
    ]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    if traces.exists():
        cmd += ["--append"]
    rc = _run_stage(cmd, f"smoke{k}")
    if rc == EXIT_POISONED:
        return "poisoned"
    if rc != 0:
        return "failed"
    return "ok"


def _datagen(k: int, checkpoint: str | None,
             args: argparse.Namespace) -> str:
    """Datagen for epoch k, resumable. Returns one of:

    * ``"ok"``       -- traces exist; training can proceed;
    * ``"failed"``   -- gave up with zero traces (skip this epoch's train);
    * ``"poisoned"`` -- the degeneracy fuse tripped (EXIT_POISONED):
      ``checkpoint`` produces un-trainable gibberish. NOT retried -- a
      collapsed checkpoint is deterministic, and the orchestrator must
      stop the whole run (stop-on-poison, module docstring)."""
    label = f"{args.prefix}_iter{k}"
    traces = DATA_GAME_DIR / label / "traces.jsonl"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        done_gens = _count_lines(traces)
        remaining = args.max_generations - done_gens
        if done_gens and remaining <= 0:
            logger.info("[datagen%d] budget already spent (%d gens on "
                        "disk); nothing to do", k, done_gens)
            return "ok"
        cmd = [
            sys.executable, "-m", "training.generate_game_traces",
            "--label", label,
            "--parallel", str(args.parallel),
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
        rc = _run_stage(cmd, f"datagen{k} attempt {attempt}")
        if rc == 0:
            return "ok"
        if rc == EXIT_POISONED:
            return "poisoned"
    n = _count_lines(traces)
    if n:
        logger.error("[datagen%d] gave up after %d attempts but %d "
                     "generations exist -- training on the partial epoch",
                     k, MAX_ATTEMPTS, n)
        return "ok"
    logger.error("[datagen%d] gave up with ZERO traces -- skipping this "
                 "epoch's training", k)
    return "failed"


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
        if _run_stage(cmd, f"train{k} attempt {attempt}") == 0:
            ckpt = _train_result_checkpoint(label)
            if ckpt:
                logger.info("[train%d] checkpoint (last good per the "
                            "trainer's done event): %s", k, ckpt)
                return ckpt
            # Exit 0 with no readable done-event verdict would mean
            # train.py's logging contract broke -- do not paper over it,
            # but do not stop the weekend either.
            logger.error("[train%d] exited 0 but no last_good_checkpoint "
                         "verdict could be read for label %s -- treating "
                         "as failure", k, label)
    logger.error("[train%d] gave up after %d attempts -- next epoch will "
                 "reuse checkpoint %r", k, MAX_ATTEMPTS, resume)
    return None


def train_one_epoch(k: int, prefix: str, resume: str | None,
                    max_steps: int | None = None) -> int:
    """The in-process train stage (child mode, --train-iter). Mirrors
    run_first_iteration.py: the epoch's game + analyst traces, the player
    trust-region anchor, the manifest replay sources, the capability
    probes, one pass over the data."""
    from training.external_data import sources_from_manifest
    from training.game_traces import (
        AnalystTraceSource,
        GameTraceSource,
        PlayerAnchorSource,
    )
    from training.probes import build_probe_hooks
    from training.train import TrainConfig, configure_logging, run_training

    configure_logging()
    label = f"{prefix}_iter{k}"
    sources = [
        GameTraceSource(f"data_game/{label}/traces.jsonl"),
        # Trust region over the SAME player traces: kd_anchor to the parent
        # checkpoint (= resume; falls back to the base when resume is None,
        # which IS epoch 1's parent). Rationale in game_traces.py.
        PlayerAnchorSource(f"data_game/{label}/traces.jsonl"),
        AnalystTraceSource(f"data_game/{label}/analyst_traces.jsonl"),
        *sources_from_manifest(),
    ]
    hooks, guards = build_probe_hooks()
    cfg = TrainConfig(
        label=label,
        epochs=1,  # self-generated data is only on-policy the first pass
        resume_checkpoint=resume,
        anchor_checkpoint=resume,  # the trust region's frozen teacher
        max_steps=max_steps,  # None = the full single pass
    )
    return run_training(sources, cfg, extra_hooks=hooks, extra_guards=guards)


# ============================================================= orchestrator

def orchestrate(args: argparse.Namespace) -> int:
    global _MONITOR
    _MONITOR = VramMonitor(args.prefix)
    state = _load_state(args.prefix)
    checkpoint = args.start_checkpoint
    failures = 0
    poisoned = False

    def _stop_on_poison(stage: str, bad_ckpt: str | None) -> None:
        """STOP-ON-POISON: a tripped degeneracy fuse means ``bad_ckpt``
        produces un-trainable gibberish -- every later epoch would train
        on garbage, so the whole run stops HERE, loudly, with the broken
        checkpoint on record for the morning autopsy."""
        nonlocal poisoned
        poisoned = True
        state["poisoned"] = {"stage": stage, "checkpoint": bad_ckpt}
        _save_state(args.prefix, state)
        logger.error(
            "STOPPING THE ENTIRE RUN: checkpoint %r is POISONED (the "
            "degeneracy fuse tripped during %s -- consecutive generations "
            "with no parseable rating or move). No retry, no later "
            "epochs; recorded under 'poisoned' in %s. Inspect the "
            "checkpoint and data_game/ output by hand before relaunching.",
            bad_ckpt or "<bare HF weights>", stage, _state_path(args.prefix),
        )

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
            outcome = "ok"
        else:
            outcome = _datagen(k, checkpoint, args)
            if outcome == "ok":
                state["done"].append(f"datagen{k}")
                _save_state(args.prefix, state)
            else:
                failures += 1
        if outcome == "poisoned":
            _stop_on_poison(f"datagen{k}", checkpoint)
            break

        new_ckpt = None
        if outcome == "ok":
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

        # Post-train smoke eval (STANDARD, every fresh checkpoint): 8 real
        # games -> win rate / ratings / degeneracy / distance deltas,
        # logged AND stored in the state file -- so even a run whose last
        # stage is training (no following datagen) ends with a
        # game-performance reading, and a poisoned checkpoint surfaces in
        # ~15 min instead of at the next epoch's multi-hour datagen.
        if new_ckpt:
            smoke = _smoke_eval(k, new_ckpt, args)
            if smoke == "ok":
                summary = _summarize_traces(f"{args.prefix}_smoke{k}")
                logger.info("[smoke_eval%d] checkpoint %s: %s",
                            k, new_ckpt, json.dumps(summary))
                state.setdefault("smoke", {})[str(k)] = summary
                _save_state(args.prefix, state)
            elif smoke == "poisoned":
                _stop_on_poison(f"smoke{k}", new_ckpt)
                break
            else:
                logger.error("[smoke_eval%d] crashed -- no performance "
                             "reading for %s (the checkpoint itself may "
                             "still be fine; run the smoke eval by hand)",
                             k, new_ckpt)

    if poisoned:
        logger.error("weekend run STOPPED ON POISON after %d failed "
                     "stage(s); last checkpoint %r (see 'poisoned' in the "
                     "state file)", failures, checkpoint)
    else:
        logger.info("weekend run complete: %d epoch(s), %d failed "
                    "stage(s), final checkpoint %r",
                    args.epochs, failures, checkpoint)
    logger.info("stage time: datagen %.2fh (smoke evals included), train "
                "%.2fh (all epochs + retries)",
                _STAGE_HOURS["datagen"], _STAGE_HOURS["train"])
    _MONITOR.finish()
    return failures + (1 if poisoned else 0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m training.run_weekend",
        description="Unattended multi-epoch self-training loop "
                    "(datagen -> train per epoch, subprocess per stage)",
    )
    p.add_argument("--epochs", type=int, default=3,
                   help="expert-iteration cycles (default 3; fits Friday "
                        "afternoon -> Monday morning even serially at "
                        "~20h/epoch, t8 2026-07-30 -- with lots of slack "
                        "at the default parallelism)")
    p.add_argument("--games", type=int, default=60,
                   help="games per datagen epoch (generate_game_traces "
                        "default)")
    p.add_argument("--max-generations", type=int, default=3000,
                   help="player-generation budget per datagen epoch; THE "
                        "knob for fitting the window (epoch datagen hours "
                        "~= this x s/gen / 3600; measured s/gen in the "
                        "module docstring)")
    p.add_argument("--parallel", type=int, default=16,
                   help="concurrent datagen sessions per epoch (passed to "
                        "generate_game_traces); 1 = the fully serial "
                        "conservative path; default 16 is set by VRAM "
                        "headroom on a 96 GB box")
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
