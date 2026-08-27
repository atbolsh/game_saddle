"""Unattended weekend self-training: N sequential epochs of datagen -> train.

One "epoch" here is one full expert-iteration cycle. For epoch k (1-based):

  1. datagen   ``python -m training.generate_game_traces --label
     <prefix>_iter<k> --parallel <--parallel, default 8> --checkpoint
     <previous epoch's adapter> --multi-gold`` (0–3 golds, openings any;
     eating gold does not end the game). ``--parallel 1`` restores the
     fully serial datagen path
  2. train     ``python -m training.run_weekend --train-iter <k>`` -- the
     epoch's GameTraceSource + PlayerAnchorSource (the trust region,
     anchored to the previous epoch's adapter) + AnalystTraceSource plus
     the manifest replay sources, resumed from the previous epoch's
     adapter (epoch 1 starts from bare HF weights unless
     --checkpoint is given).
  3. smoke eval  8 real **sealed one-gold** games with the FRESH
     checkpoint through the same generator without ``--multi-gold``
     (label ``<prefix>_smoke<k>``, never trained on): eat-gold win
     rate, mean/min rating, degeneracy fraction, and mean gold-distance
     delta are logged and stored under ``smoke`` in the state file --
     comparable to earlier weekends. A poisoned checkpoint surfaces in
     ~15 min. ``grep smoke_eval`` on the run log for the morning review.
  4. prune  after a SUCCESSFUL train, the consumed corpus is tombstoned
     down to a keepsake sample (one won game if any, plus one other
     random game): every other record in traces.jsonl /
     analyst_traces.jsonl becomes ``{"pruned": true, "meta": ...}`` and
     its frames are deleted, so win-rate/rating stats stay computable
     forever while disk is reclaimed (``_prune_datagen``; summary under
     ``pruned`` in the state file). Smoke dirs are never pruned. A
     failed epoch keeps its full corpus for the retry-by-hand path.
     ``--prune <label>`` (repeatable) runs the same pruning standalone
     on any already-trained-on corpus and exits -- for dirs that predate
     this step.

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

To continue from an existing adapter, pass ``--checkpoint NAME`` (epoch 1
datagen + train start from that adapter; later epochs follow last-good).
``--start-checkpoint`` and ``--resume-checkpoint`` are rejected: they used
to be two flags on this parser with different meanings, and the latter was
ignored by the parent (the 2026-08-16 weekend started from bare HF).
``--resume-checkpoint`` belongs to ``python -m training.train``.

Omit ``--seed`` for a fresh board stream (a random base seed is drawn,
logged at INFO as ``base seed N (...)``, and stored in the state file).
Pass that logged value as ``--seed N`` to replay the same run; a mismatch
with a stored ``base_seed`` is a hard error.

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
import random
import secrets
import shutil
import subprocess
import sys
import tempfile
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


def _resolve_seed(args: argparse.Namespace, state: dict | None,
                  persist_prefix: str | None) -> None:
    """Fill ``args.seed`` once per process: explicit flag, reused from
    the state file (crash-resume of this prefix), or a fresh random
    draw. A mismatch between ``--seed`` and a stored ``base_seed`` is
    a hard error -- mixing two streams silently is the failure mode
    this exists to prevent. The resolved value is logged at INFO and
    persisted so the run is replayable."""
    explicit = args.seed
    stored = None if state is None else state.get("base_seed")
    if stored is not None:
        stored = int(stored)

    if explicit is not None:
        if stored is not None and stored != explicit:
            where = (str(_state_path(persist_prefix))
                     if persist_prefix else "the state file")
            raise SystemExit(
                f"--seed {explicit} disagrees with base_seed {stored} "
                f"in {where}; pass the stored seed to resume this run, "
                f"or a new --prefix to start a fresh stream"
            )
        args.seed = explicit
        source = "explicit"
    elif stored is not None:
        args.seed = stored
        source = "reused from state"
    else:
        n = secrets.randbelow(2 ** 31 - 1)
        if n == 0:
            n = 1
        args.seed = n
        source = ("resolved, stored in state" if persist_prefix
                  else "resolved")

    if persist_prefix is not None and state is not None:
        if state.get("base_seed") != args.seed:
            state["base_seed"] = args.seed
            _save_state(persist_prefix, state)
    logger.info("base seed %d (%s)", args.seed, source)


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
    if done.get("ended_early"):
        # The trainer stopped itself (e.g. the consecutive-rollback stop,
        # train.py ROLLBACK) and stands behind last_good -- a valid
        # hand-off, but the weekend log must say so loudly.
        logger.error("[%s] trainer ENDED EARLY: %s -- handing the next "
                     "epoch its last good checkpoint %s",
                     label, done["ended_early"], ckpt)
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


#: Noised-frame temp dirs created by training/game_traces.py
#: (_make_noise_dir): game_<label>_noise_*, player_anchor_<label>_noise_*,
#: analyst_<label>_noise_*.
_NOISE_DIR_GLOBS = ("game_*_noise_*", "player_anchor_*_noise_*",
                    "analyst_*_noise_*")


def _sweep_noise_dirs() -> None:
    """Delete leaked noised-frame temp dirs (2026-08-11): game_traces
    removes its noise dirs at interpreter exit, but a train stage that
    dies hard (OOM kill, box reboot) leaks ~6 GiB of frame copies per
    stage into the system temp dir -- the aug6 run leaked ~65 GiB this
    way. The orchestrator runs stages strictly sequentially, so at an
    epoch boundary no train process is alive and every matching dir is
    stale by construction."""
    tmp = Path(tempfile.gettempdir())
    n_dirs = 0
    n_bytes = 0
    for pattern in _NOISE_DIR_GLOBS:
        for d in tmp.glob(pattern):
            if not d.is_dir():
                continue
            n_bytes += sum(f.stat().st_size
                           for f in d.rglob("*") if f.is_file())
            shutil.rmtree(d, ignore_errors=True)
            n_dirs += 1
    if n_dirs:
        logger.info("swept %d stale noised-frame temp dir(s), %.1f GiB, "
                    "from %s", n_dirs, n_bytes / 2 ** 30, tmp)


def _prune_datagen(label: str, seed: int) -> dict | None:
    """Disk hygiene for a corpus that has been TRAINED ON (2026-08-06):
    tombstone every game except a keepsake sample -- one WON game (when
    any exist) plus one other game, both chosen deterministically from
    ``seed``.

    Kept games keep their full records (messages, target text, frames).
    Every other record is replaced IN PLACE, in both ``traces.jsonl``
    and ``analyst_traces.jsonl``, by ``{"pruned": true, "meta": <the
    original meta>}``. meta is the only thing stats consumers read
    (:func:`_summarize_traces`, the datagen plots), so win rates,
    ratings, and distance deltas stay computable from a pruned corpus
    forever; only the bulky contexts go. Frames under ``images/`` not
    referenced by a kept record are deleted.

    A pruned corpus can never be silently retrained: GameTraceSource
    hard-fails on the first tombstone (no ``messages`` key), which is
    the desired loud failure -- two games quietly posing as a full
    corpus would be far worse. Each file is rewritten via a temp file +
    atomic replace, so a crash mid-prune never loses records. Returns a
    summary dict, or None when the corpus is missing or already pruned
    (idempotent under manual re-runs)."""
    out_dir = DATA_GAME_DIR / label
    traces_path = out_dir / "traces.jsonl"
    analyst_path = out_dir / "analyst_traces.jsonl"
    if not traces_path.is_file():
        logger.error("[prune %s] no traces.jsonl -- nothing to prune", label)
        return None

    # Pass 1: the game list and each game's won status. A game is one
    # distinct (session_id, game_index) -- same keying as
    # _summarize_traces (game_index alone repeats across --append
    # resume attempts).
    games: dict[tuple, bool] = {}
    with open(traces_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("pruned"):
                logger.info("[prune %s] already pruned; skipping", label)
                return None
            meta = obj["meta"]
            key = (meta.get("session_id"), meta.get("game_index"))
            games[key] = games.get(key, False) or bool(meta.get("game_won"))

    rng = random.Random(seed)
    won = sorted(k for k, w in games.items() if w)
    keep: set[tuple] = set()
    if won:
        keep.add(rng.choice(won))
    rest = sorted(k for k in games if k not in keep)
    if rest:
        keep.add(rng.choice(rest))

    kept_images: set[str] = set()
    counts = {"records_kept": 0, "records_pruned": 0}

    def rewrite(path: Path) -> None:
        if not path.is_file():
            return
        tmp = path.with_name(path.name + ".prune_tmp")
        with open(path, encoding="utf-8") as src, \
                open(tmp, "w", encoding="utf-8") as dst:
            for line in src:
                if not line.strip():
                    continue
                obj = json.loads(line)
                meta = obj["meta"]
                key = (meta.get("session_id"), meta.get("game_index"))
                if key in keep:
                    counts["records_kept"] += 1
                    for m in obj.get("messages") or []:
                        content = m.get("content")
                        if not isinstance(content, list):
                            continue
                        for part in content:
                            if (isinstance(part, dict)
                                    and part.get("type") == "image"):
                                kept_images.add(Path(part["url"]).name)
                    dst.write(line if line.endswith("\n") else line + "\n")
                else:
                    counts["records_pruned"] += 1
                    dst.write(json.dumps(
                        {"pruned": True, "meta": meta}, ensure_ascii=False
                    ) + "\n")
        tmp.replace(path)

    rewrite(traces_path)
    rewrite(analyst_path)

    images_dir = out_dir / "images"
    n_deleted = 0
    bytes_freed = 0
    if images_dir.is_dir():
        for img in images_dir.iterdir():
            if not img.is_file() or img.name in kept_images:
                continue
            bytes_freed += img.stat().st_size
            img.unlink()
            n_deleted += 1

    summary = {
        "games_total": len(games),
        "games_kept": [
            {"session_id": sid, "game_index": gi, "won": games[(sid, gi)]}
            for sid, gi in sorted(keep, key=str)
        ],
        **counts,
        "images_deleted": n_deleted,
        "mib_freed": round(bytes_freed / 2 ** 20, 1),
    }
    logger.info("[prune %s] %s", label, json.dumps(summary))
    return summary


# =================================================================== stages

#: Post-train smoke eval size: 8 real games x 50 gens (400 generations,
#: ~65 min at measured 9.9 s/gen) -- a full-length game so the player
#: has time to finish, cheap enough to run unconditionally. Poisoned
#: checkpoints still surface in ~15 min (degeneracy fuse trips after 25
#: consecutive bad generations, early in the smoke).
#:
#: Smoke stays sealed one-gold (no --multi-gold) so eat-gold win rates
#: stay comparable to earlier weekends. Training datagen passes this.
DATAGEN_ROOM_FLAG = ["--multi-gold"]
SMOKE_GAMES = 8
SMOKE_MAX_GENERATIONS = 400
SMOKE_QUESTION_RATE = 0.075  # half of datagen's default 0.15


def _smoke_eval(k: int, checkpoint: str | None,
                args: argparse.Namespace) -> str:
    """Post-train sanity check on epoch k's fresh checkpoint: a tiny run
    through the REAL generator under the separate label ``<prefix>_smoke<k>``
    -- smoke traces never enter a training corpus, and they stay sealed
    one-gold eat-to-win so those win rates stay comparable to earlier
    weekends. Returns ``"ok"`` / ``"poisoned"`` (fuse tripped -> stop
    the run) / ``"failed"`` (crash: a missing reading, logged at ERROR,
    but NOT a failed epoch -- the checkpoint may still be fine)."""
    label = f"{args.prefix}_smoke{k}"
    traces = DATA_GAME_DIR / label / "traces.jsonl"
    cmd = [
        sys.executable, "-m", "training.generate_game_traces",
        "--label", label,
        "--parallel", str(args.parallel),
        "--games", str(SMOKE_GAMES),
        "--max-generations", str(SMOKE_MAX_GENERATIONS),
        "--seed", str(args.seed + 100 * k + 51),
        "--question-rate", str(SMOKE_QUESTION_RATE),
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
        ] + DATAGEN_ROOM_FLAG
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
        cmd += ["--checkpoint", resume]
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
        # novelty=True (2026-08-11): the boredom decay taxes blind
        # continuation -- the aug6 run's turn runs were 97%
        # self-continuing and training DEEPENED the commitment (flip
        # rate 0.10 -> 0.03 over 11 epochs). Class default stays False
        # (t1 asserts a default source must not decay).
        GameTraceSource(f"data_game/{label}/traces.jsonl", novelty=True),
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
        # 16k chars (~4k tok at the old 4-chars/token proxy) now drops
        # essentially every multi-gold player/analyst row (T~7k). 128 KiB
        # still fences pathological KD dumps. Weekend trains on this GPU
        # were never VRAM-bound (aug18/aug21 logs).
        max_example_chars=131072,
    )
    return run_training(sources, cfg, extra_hooks=hooks, extra_guards=guards)


# ============================================================= orchestrator

def orchestrate(args: argparse.Namespace) -> int:
    global _MONITOR
    _MONITOR = VramMonitor(args.prefix)
    state = _load_state(args.prefix)
    _resolve_seed(args, state, persist_prefix=args.prefix)
    checkpoint = args.checkpoint
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
        _sweep_noise_dirs()

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

        # Disk hygiene (2026-08-06): the corpus has been consumed, so
        # tombstone it down to a keepsake sample (one won game + one
        # other; stats stay computable from the metas left behind).
        # Only after a SUCCESSFUL train -- a failed epoch keeps its full
        # corpus for the retry-by-hand path above. Smoke dirs are never
        # pruned (tiny, and the user curates them). A pruning crash is
        # not worth killing an unattended weekend over: log it loudly
        # and move on.
        if new_ckpt:
            try:
                pruned = _prune_datagen(f"{args.prefix}_iter{k}",
                                        seed=args.seed + 9000 + k)
            except Exception:
                logger.exception("[prune iter%d] FAILED -- full corpus "
                                 "left on disk; prune by hand if disk "
                                 "runs low", k)
            else:
                if pruned:
                    state.setdefault("pruned", {})[str(k)] = pruned
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


class _RejectedCheckpointFlag(argparse.Action):
    """Trap for the two names that used to mean different things here.

    ``--start-checkpoint`` was the parent dest (epoch 1's adapter).
    ``--resume-checkpoint`` was a child-only dest the parent never read,
    so ``python -m training.run_weekend --resume-checkpoint X`` started
    from bare HF (the 2026-08-16 weekend). Both collapsed to
    ``--checkpoint``, used by parent and child. Accepting either old
    name again would re-open the silent-ignore hole.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string == "--resume-checkpoint":
            parser.error(
                "--resume-checkpoint is a python -m training.train flag, "
                "not a run_weekend flag. The 2026-08-16 weekend started "
                "from bare HF because this name was accepted and ignored. "
                f"Pass --checkpoint {values} (epoch 1 starts from that "
                "adapter; later epochs follow the previous epoch's "
                "last-good checkpoint)."
            )
        parser.error(
            f"{option_string} was renamed to --checkpoint (same meaning). "
            f"Pass --checkpoint {values}."
        )


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
    p.add_argument("--parallel", type=int, default=8,
                    help="concurrent datagen sessions per epoch (passed to "
                        "generate_game_traces); 1 = the fully serial "
                        "conservative path; default 8 after 12 OOM'd on "
                        "50-move multi-gold contexts (T~7k) on a 96 GB box "
                        "(2026-08-26). 16 died with CUBLAS on shorter "
                        "sealed games (2026-08-17)")
    p.add_argument("--seed", type=int, default=None,
                   help="base seed; each epoch and resume attempt derives "
                        "a distinct noise/question stream from it. Omit "
                        "to draw a random seed (logged and stored in the "
                        "state file so the run is replayable); pass the "
                        "logged value to replay")
    p.add_argument("--prefix", default="weekend",
                   help="label prefix: data_game/<prefix>_iter<k>/, "
                        "checkpoints <prefix>_iter<k>_step<N>, state file "
                        "data_game/<prefix>_state.json")
    p.add_argument("--checkpoint", default=None, metavar="NAME",
                   help="adapter under weights/<arch>/ for this process to "
                        "start from. Parent: epoch 1's datagen + train "
                        "(default: bare HF weights). Child (--train-iter): "
                        "the adapter this train stage resumes from. Later "
                        "epochs follow the previous epoch's last-good "
                        "checkpoint automatically. --start-checkpoint and "
                        "--resume-checkpoint are rejected: they used to "
                        "mean different things on this parser, and "
                        "--resume-checkpoint was ignored by the parent "
                        "(the 2026-08-16 weekend started from bare HF)")
    p.add_argument("--start-checkpoint", "--resume-checkpoint",
                   dest="_rejected_checkpoint",
                   action=_RejectedCheckpointFlag, metavar="NAME",
                   help=argparse.SUPPRESS)
    p.add_argument("--train-max-steps", type=int, default=None,
                   help="cap each train stage at N optimizer steps "
                        "(default: full pass); use a small N to smoke-test "
                        "the whole orchestration cheaply before the real "
                        "weekend")
    p.add_argument("--prune", metavar="LABEL", action="append",
                   default=None,
                   help="STANDALONE tool mode: prune data_game/<LABEL> "
                        "down to the keepsake sample (one won game if any "
                        "+ one other; every other record tombstoned to "
                        "its meta, unreferenced frames deleted) and exit "
                        "-- no epochs run. Repeatable. Only for corpora "
                        "that have ALREADY been trained on: a pruned "
                        "corpus cannot be retrained. Does not touch the "
                        "state file; idempotent on already-pruned dirs. "
                        "The keepsake pick derives from --seed.")
    p.add_argument("--train-iter", type=int, default=None,
                   help="INTERNAL (child mode): run epoch k's train stage "
                        "in this process and exit. Pass the adapter as "
                        "--checkpoint (same flag as the parent)")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    if args.train_iter is not None:
        return train_one_epoch(args.train_iter, args.prefix,
                               args.checkpoint,
                               max_steps=args.train_max_steps)
    if args.prune:
        # Standalone prune mode: exit code counts labels with no
        # traces.jsonl (already-pruned dirs log INFO and are fine).
        _resolve_seed(args, state=None, persist_prefix=None)
        missing = 0
        for label in args.prune:
            if not (DATA_GAME_DIR / label / "traces.jsonl").is_file():
                missing += 1
            _prune_datagen(label, seed=args.seed)
        return missing
    return orchestrate(args)


if __name__ == "__main__":
    sys.exit(main())
