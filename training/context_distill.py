"""Reusable context-distillation runner: CE finetune + per-epoch gate/smoke.

Each epoch is a complete, separately saved, separately measured unit
(train.py ``save_steps`` defaults to 400; one distill epoch is only ~127
optimizer steps, so a single 2-epoch ``run_training`` call would skip the
epoch-1 boundary and the per-epoch gate/smoke). Parent and child are
separate processes -- the inference singleton and the train copy cannot
share one.

    python -m training.context_distill \\
        --data data_game/<corpus>/traces_distill.jsonl \\
        --checkpoint <start_adapter> \\
        --label <run_label> --lr 1e-5 --epochs 2

Per epoch ``e`` (child label ``<label>_e<e>``):

  1. Train child: player CE + analyst CE (clone the recorded analyses)
     + the usual ``AnalystTraceSource`` KD leash (~150/epoch vs frozen
     base). Final save is the epoch checkpoint, unconditional, before
     gate/smoke.
  2. Analyst gate on that checkpoint. Fail does NOT skip smoke and does
     NOT stop the loop unless ``--stop-on-gate-fail``.
  3. Sealed one-gold smoke (same knobs as the sep10 arms).

State file: ``data_game/<label>_state.json`` (skip-done resume).

Logs (always on disk, nohup not required for persistence):

* ``data_game/<label>_orchestrator.log`` -- parent INFO/ERROR
* ``data_game/<label>_vram.jsonl`` -- same 1/min ``nvidia-smi`` monitor
  as ``run_weekend``
* ``data_game/<label>_stages/<stage>.log`` -- teed child stdout+stderr
* ``data_game/<label>_stages/<stage>.exit.json`` -- exit / signal
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.game_traces import (  # noqa: E402
    ANALYST_BATCH_CAP,
    _TraceFileSource,
    _make_noise_dir,
)
from training.image_noise import TRAINING_STRENGTH  # noqa: E402
from training.train import TrainingExample  # noqa: E402

logger = logging.getLogger("context_distill")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_GAME_DIR = REPO_ROOT / "data_game"

SMOKE_GAMES = 8
SMOKE_MAX_GENERATIONS = 400
SMOKE_QUESTION_RATE = 0.075
SMOKE_PARALLEL = 12
DEFAULT_SMOKE_SEED = 20260910
DEFAULT_LR = 1e-5
DEFAULT_EPOCHS = 2

#: CE fence (player + analyst clone). 8192 dropped 2351/2924 analyst-CE
#: rows on the frankenstein corpus -- the long TARGET/WRONG writeups.
#: 12288 is the old analyst cap; it OOMed only as KD at B=4 / T~12k
#: (94.8 GiB). Fused CE does not build [T x 262k], so 12k CE on 96 GiB
#: is the safe "keep the analyses" number.
DISTILL_CE_TOKEN_CAP = 12288
#: KD leash stays at the weekend fence. AnalystTraceSource already
#: batch_cap=2; do not feed it 12k rows.
DISTILL_KD_TOKEN_CAP = 8192

#: Set by orchestrate(); _run_stage tags the VRAM monitor and tees
#: children under ``data_game/<label>_stages/``.
_MONITOR = None
_STAGE_LOG_DIR: Path | None = None
_LAST_STAGE_REPORT: dict | None = None


class DistillTraceSource(_TraceFileSource):
    """Plain-CE clone of every record in a (usually preprocessed) trace file.

    Uniform ``example_weight=1.0``, no span weights, no reward machinery.
    Records with ``rating: null`` are KEPT -- rating is irrelevant here.
    Standard noised-frame handling is inherited from ``_TraceFileSource``.

    ``name`` defaults to ``distill_<stem>`` so player
    (``traces_distill``) and analyst (``analyst_traces_distill``) in the
    same directory stay distinct held-out meters. ``drop_unverified``
    skips analyst records whose WRONG spans failed substring checks.
    """

    def __init__(
        self,
        path: str | Path,
        noise_strength: float = TRAINING_STRENGTH,
        noise_seed: int | None = None,
        name: str | None = None,
        drop_unverified: bool = False,
        batch_cap: int | None = None,
    ):
        super().__init__(path, noise_strength, noise_seed)
        self.name = name or f"distill_{self.path.stem}"
        self.weight = 1.0
        self.drop_unverified = drop_unverified
        self.batch_cap = batch_cap

    def examples(self) -> Iterator[TrainingExample]:
        rng = random.Random(self.noise_seed)
        noise_dir = _make_noise_dir(f"{self.name}_noise_")
        n_yielded = 0
        n_dropped = 0
        for lineno, messages, target_text, meta in self._iter_records():
            if self.drop_unverified and meta.get("unverified_spans"):
                n_dropped += 1
                continue
            self._rewrite_frames(messages, rng, noise_dir, lineno)
            yield TrainingExample(
                messages=messages,
                target_text=target_text,
                span_weights=None,
                loss="ce",
                source=self.name,
                meta=meta,
                example_weight=1.0,
                batch_cap=self.batch_cap,
            )
            n_yielded += 1
        logger.info(
            "%s: yielded %d record(s) as uniform-CE distill examples "
            "(rating-null kept; no span weights)",
            self.name, n_yielded,
        )
        if n_dropped:
            logger.warning(
                "%s: DROPPED %d/%d record(s) whose analysis quoted "
                "unverified WRONG spans (drop_unverified=True).",
                self.name, n_dropped, n_dropped + n_yielded,
            )


def _state_path(label: str) -> Path:
    return DATA_GAME_DIR / f"{label}_state.json"


def _load_state(label: str) -> dict:
    path = _state_path(label)
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        logger.info("resuming from %s: %s", path, state)
        return state
    return {"done": [], "checkpoints": {}}


def _save_state(label: str, state: dict) -> None:
    path = _state_path(label)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _attach_orchestrator_log(label: str) -> Path:
    """Parent file log. Independent of the tty / nohup."""
    path = DATA_GAME_DIR / f"{label}_orchestrator.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
    ))
    logging.getLogger().addHandler(fh)
    logger.info("orchestrator log: %s", path)
    return path


def _exit_report(rc: int) -> dict:
    """Decode subprocess.wait status. Negative rc is -signal."""
    if rc < 0:
        sig = -rc
        try:
            name = signal.Signals(sig).name
        except ValueError:
            name = f"SIG{sig}"
        return {
            "exit": rc,
            "ok": False,
            "signaled": True,
            "signal": sig,
            "signal_name": name,
            "note": (
                f"child killed by {name} ({sig}); no Python traceback. "
                "SIGKILL is the OOM killer or an external kill; "
                "SIGTERM is pkill/timeout. CUDA OOM is a Python "
                "exception (exit 1) and does not appear in dmesg."
            ),
        }
    return {
        "exit": rc,
        "ok": rc == 0,
        "signaled": False,
        "signal": None,
        "signal_name": None,
        "note": None if rc == 0 else (
            "child exited with a process status; the stage .log has "
            "stdout+stderr (CUDA OOM is typically exit 1 + "
            "RuntimeError and does not appear in dmesg)."
        ),
    }


def _run_stage(cmd: list[str], stage: str) -> int:
    """Run a child, tee stdout+stderr to disk, record exit/signal.

    VRAM monitor is tagged for the duration (same ``VramMonitor`` as
    ``run_weekend``). ``PYTHONUNBUFFERED=1`` so the child's prints
    reach the tee before a crash.
    """
    global _LAST_STAGE_REPORT
    logger.info("[%s] starting: %s", stage, " ".join(cmd))
    if _MONITOR is not None:
        _MONITOR.set_stage(stage)
    t0 = time.perf_counter()
    child_log = None
    if _STAGE_LOG_DIR is not None:
        _STAGE_LOG_DIR.mkdir(parents=True, exist_ok=True)
        child_log = _STAGE_LOG_DIR / f"{stage}.log"
        logger.info("[%s] child log: %s", stage, child_log)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    rc = 1
    child: subprocess.Popen[str] | None = None
    try:
        if child_log is None:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
            rc = proc.returncode
        else:
            header = (
                f"=== {stage} start "
                f"{_dt.datetime.now().isoformat(timespec='seconds')}\n"
                f"cwd={REPO_ROOT}\n"
                f"cmd={' '.join(cmd)}\n\n"
            )
            with open(child_log, "a", encoding="utf-8") as lf:
                lf.write(header)
                lf.flush()
                child = subprocess.Popen(
                    cmd, cwd=REPO_ROOT, env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert child.stdout is not None
                for line in child.stdout:
                    lf.write(line)
                    lf.flush()
                    sys.stdout.write(line)
                    sys.stdout.flush()
                rc = child.wait()
    except BaseException:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
        raise
    finally:
        hours = (time.perf_counter() - t0) / 3600
        if _MONITOR is not None:
            _MONITOR.set_stage(None)
    report = _exit_report(rc)
    report.update({
        "stage": stage,
        "hours": round(hours, 4),
        "cmd": cmd,
        "log": str(child_log) if child_log else None,
    })
    _LAST_STAGE_REPORT = report
    if child_log is not None:
        exit_path = child_log.with_suffix(".exit.json")
        exit_path.write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8",
        )
        report["exit_path"] = str(exit_path)
    if rc == 0:
        logger.info("[%s] finished in %.2fh", stage, hours)
    else:
        logger.error(
            "[%s] FAILED after %.2fh: exit=%s signaled=%s "
            "signal=%s log=%s -- %s",
            stage, hours, report["exit"], report["signaled"],
            report["signal_name"], report["log"], report["note"],
        )
    return rc


def train_one_epoch(
    data: Path,
    label: str,
    epoch: int,
    resume: str | None,
    lr: float,
    analyst_ce: Path | None,
    analyst_kd: Path | None,
) -> int:
    """In-process train child (``--train-epoch``)."""
    from training.game_traces import AnalystTraceSource
    from training.probes import build_probe_hooks
    from training.train import TrainConfig, configure_logging, run_training

    configure_logging()
    sources: list = [DistillTraceSource(data)]
    if analyst_ce is not None:
        sources.append(DistillTraceSource(
            analyst_ce, drop_unverified=True,
            batch_cap=ANALYST_BATCH_CAP,
        ))
    if analyst_kd is not None:
        sources.append(AnalystTraceSource(analyst_kd))
    hooks, guards = build_probe_hooks()
    child_label = f"{label}_e{epoch}"
    cfg = TrainConfig(
        label=child_label,
        epochs=1,
        lr=lr,
        resume_checkpoint=resume,
        max_example_tokens=DISTILL_CE_TOKEN_CAP,
        max_example_tokens_analyst=DISTILL_KD_TOKEN_CAP,
    )
    return run_training(sources, cfg, extra_hooks=hooks, extra_guards=guards)


def _train_epoch(e: int, resume: str | None,
                 args: argparse.Namespace) -> str | None:
    from training.run_weekend import _train_result_checkpoint

    child_label = f"{args.label}_e{e}"
    cmd = [
        sys.executable, "-m", "training.context_distill",
        "--train-epoch", str(e),
        "--data", str(args.data),
        "--label", args.label,
        "--lr", str(args.lr),
    ]
    if resume:
        cmd += ["--checkpoint", resume]
    if args.no_analyst_ce:
        cmd.append("--no-analyst-ce")
    elif args.analyst_data:
        cmd += ["--analyst-data", str(args.analyst_data)]
    if args.no_analyst_anchor:
        cmd.append("--no-analyst-anchor")
    elif args.analyst_anchor:
        cmd += ["--analyst-anchor", str(args.analyst_anchor)]
    rc = _run_stage(cmd, f"train{e}")
    args.last_train_exit = rc
    args.last_train_report = _LAST_STAGE_REPORT
    if rc != 0:
        return None
    ckpt = _train_result_checkpoint(child_label)
    if ckpt:
        logger.info("[train%d] checkpoint (last good per the trainer's "
                    "done event): %s", e, ckpt)
        return ckpt
    logger.error("[train%d] exited 0 but no last_good_checkpoint for "
                 "label %s", e, child_label)
    return None


def _gate(e: int, checkpoint: str, args: argparse.Namespace) -> dict:
    out = DATA_GAME_DIR / f"{args.label}_analyst_gate{e}.json"
    cmd = [
        sys.executable, "-m", "training.analyst_gate",
        "--checkpoint", checkpoint,
        "--out", str(out),
    ]
    rc = _run_stage(cmd, f"analyst_gate{e}")
    verdict: dict = {"passed": rc == 0, "exit": rc, "verdict_path": str(out)}
    if out.is_file():
        try:
            loaded = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("[analyst_gate%d] could not parse %s: %s", e, out, exc)
        else:
            if isinstance(loaded, dict):
                loaded["passed"] = rc == 0
                loaded["exit"] = rc
                loaded["verdict_path"] = str(out)
                verdict = loaded
    return verdict


def _smoke(e: int, checkpoint: str, args: argparse.Namespace) -> dict | None:
    from training.run_weekend import _summarize_traces

    smoke_label = f"{args.label}_smoke{e}"
    traces = DATA_GAME_DIR / smoke_label / "traces.jsonl"
    cmd = [
        sys.executable, "-m", "training.generate_game_traces",
        "--label", smoke_label,
        "--parallel", str(args.parallel),
        "--games", str(SMOKE_GAMES),
        "--max-generations", str(SMOKE_MAX_GENERATIONS),
        "--seed", str(args.smoke_seed),
        "--question-rate", str(SMOKE_QUESTION_RATE),
        "--checkpoint", checkpoint,
    ]
    if traces.exists():
        cmd.append("--append")
    rc = _run_stage(cmd, f"smoke{e}")
    if rc != 0:
        return None
    return _summarize_traces(smoke_label)


def _sibling_analyst(data: Path) -> Path | None:
    """Prefer the prepped analyst file, fall back to the raw traces."""
    parent = Path(data).parent
    for name in ("analyst_traces_distill.jsonl", "analyst_traces.jsonl"):
        path = parent / name
        if path.is_file():
            return path
    return None


def _resolve_analyst_path(
    explicit: str | None, *, skip: bool, data: Path, flag: str,
) -> Path | None:
    if skip:
        return None
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise SystemExit(f"{flag} is not a file: {path}")
        return path
    return _sibling_analyst(data)


def orchestrate(args: argparse.Namespace) -> int:
    global _MONITOR, _STAGE_LOG_DIR
    from training.run_weekend import VramMonitor

    data = Path(args.data)
    if not data.is_file():
        raise SystemExit(f"--data is not a file: {data}")
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required (the start adapter)")
    if not args.label:
        raise SystemExit("--label is required")

    args.analyst_data = _resolve_analyst_path(
        args.analyst_data, skip=args.no_analyst_ce,
        data=data, flag="--analyst-data",
    )
    args.analyst_anchor = _resolve_analyst_path(
        args.analyst_anchor, skip=args.no_analyst_anchor,
        data=data, flag="--analyst-anchor",
    )
    logger.info("player CE: %s", data)
    logger.info("analyst CE: %s", args.analyst_data or "skipped")
    logger.info("analyst KD leash: %s", args.analyst_anchor or "skipped")

    _STAGE_LOG_DIR = DATA_GAME_DIR / f"{args.label}_stages"
    _STAGE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    _MONITOR = VramMonitor(args.label)
    logger.info("VRAM trace: %s (1/min nvidia-smi, same as weekend)",
                _MONITOR.path)

    state = _load_state(args.label)
    state.setdefault("done", [])
    state.setdefault("checkpoints", {})
    state["started"] = state.get("started") or _dt.datetime.now().isoformat(
        timespec="seconds",
    )
    state["pid"] = os.getpid()
    state["phase"] = "starting"
    state["player_ce"] = str(data)
    state["analyst_ce"] = str(args.analyst_data) if args.analyst_data else None
    state["analyst_kd"] = (
        str(args.analyst_anchor) if args.analyst_anchor else None
    )
    state["orchestrator_log"] = str(
        DATA_GAME_DIR / f"{args.label}_orchestrator.log"
    )
    state["vram_trace"] = str(_MONITOR.path)
    state["stage_logs"] = str(_STAGE_LOG_DIR)
    _save_state(args.label, state)
    logger.info("state file: %s", _state_path(args.label))

    resume = args.checkpoint
    failures = 0
    try:
        return _orchestrate_epochs(args, state, resume, failures)
    finally:
        if _MONITOR is not None:
            _MONITOR.finish()
            _MONITOR = None


def _orchestrate_epochs(
    args: argparse.Namespace, state: dict, resume: str, failures: int,
) -> int:
    from training.run_weekend import _sweep_noise_dirs

    for e in range(1, args.epochs + 1):
        logger.info("=== distill epoch %d/%d (resume %r) ===",
                    e, args.epochs, resume)
        state["phase"] = f"train{e}"
        _save_state(args.label, state)
        _sweep_noise_dirs()

        recorded = state["checkpoints"].get(str(e)) if (
            f"train{e}" in state["done"]
        ) else None
        if recorded:
            resume = recorded
            logger.info("[train%d] already complete (checkpoint %r)",
                        e, resume)
            ckpt = recorded
        else:
            if f"train{e}" in state["done"]:
                logger.error("[train%d] was marked done with no "
                             "checkpoint -- retrying", e)
                state["done"] = [x for x in state["done"] if x != f"train{e}"]
                state["checkpoints"].pop(str(e), None)
                _save_state(args.label, state)
            ckpt = _train_epoch(e, resume, args)
            state["last_child"] = getattr(args, "last_train_report", None)
            if not ckpt:
                failures += 1
                report = getattr(args, "last_train_report", None)
                state["phase"] = f"train{e}_failed"
                state["last_train_exit"] = getattr(args, "last_train_exit", None)
                state["last_child"] = report
                _save_state(args.label, state)
                logger.error(
                    "STOPPING: train%d produced no checkpoint. "
                    "exit=%s signal=%s. Inspect: %s ; "
                    "logs/train_%s_e%d_*/{heartbeat.json,last_step.json,"
                    "crash.txt,events.jsonl,train_log.txt} ; "
                    "%s_vram.jsonl",
                    e,
                    (report or {}).get("exit"),
                    (report or {}).get("signal_name"),
                    (report or {}).get("log"),
                    args.label, e,
                    args.label,
                )
                return failures
            state["done"].append(f"train{e}")
            state["checkpoints"][str(e)] = ckpt
            _save_state(args.label, state)
            resume = ckpt

        if f"analyst_gate{e}" not in state["done"]:
            state["phase"] = f"analyst_gate{e}"
            _save_state(args.label, state)
            verdict = _gate(e, ckpt, args)
            state.setdefault("analyst_gate", {})[str(e)] = verdict
            state["done"].append(f"analyst_gate{e}")
            _save_state(args.label, state)
            if verdict.get("exit") == 1:
                logger.error("[analyst_gate%d] FAIL on %s -- smoke still "
                             "runs; loop continues unless "
                             "--stop-on-gate-fail", e, ckpt)
            elif verdict.get("exit") not in (0, 1):
                logger.error("[analyst_gate%d] crashed (exit %s)",
                             e, verdict.get("exit"))
        else:
            verdict = (state.get("analyst_gate") or {}).get(str(e), {})
            logger.info("[analyst_gate%d] already complete: %s",
                        e, {"passed": verdict.get("passed"),
                            "exit": verdict.get("exit")})

        if f"smoke{e}" not in state["done"]:
            state["phase"] = f"smoke{e}"
            _save_state(args.label, state)
            summary = _smoke(e, ckpt, args)
            if summary is None:
                failures += 1
                logger.error("[smoke%d] crashed -- no performance reading "
                             "for %s", e, ckpt)
            else:
                logger.info("[smoke%d] checkpoint %s: %s",
                            e, ckpt, json.dumps(summary))
                state.setdefault("smoke", {})[str(e)] = summary
            state["done"].append(f"smoke{e}")
            _save_state(args.label, state)
        else:
            logger.info("[smoke%d] already complete: %s",
                        e, state.get("smoke", {}).get(str(e)))

        if args.stop_on_gate_fail and verdict.get("exit") == 1:
            logger.error("STOPPING: --stop-on-gate-fail and epoch %d "
                         "gate failed (checkpoint %r)", e, ckpt)
            return failures + 1

        if ckpt:
            resume = ckpt

    state["phase"] = "done"
    _save_state(args.label, state)
    logger.info("distill run complete: %d epoch(s), %d failed stage(s), "
                "final checkpoint %r", args.epochs, failures, resume)
    return failures


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m training.context_distill",
        description="Per-epoch context-distillation orchestrator "
                    "(train -> gate -> smoke)",
    )
    p.add_argument(
        "--data", required=True,
        help="path to traces_distill.jsonl (never hardcoded)",
    )
    p.add_argument(
        "--checkpoint", default=None,
        help="start adapter (parent) or resume adapter (child)",
    )
    p.add_argument("--label", required=True, help="run label")
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument(
        "--analyst-data", default=None,
        help="analyst jsonl for CE clone of recorded analyses; default = "
             "sibling analyst_traces_distill.jsonl, else analyst_traces.jsonl",
    )
    p.add_argument(
        "--no-analyst-ce", action="store_true",
        help="skip analyst CE (KD leash can still run)",
    )
    p.add_argument(
        "--analyst-anchor", default=None,
        help="analyst jsonl for the KD-to-base leash; same sibling default "
             "as --analyst-data",
    )
    p.add_argument(
        "--no-analyst-anchor", action="store_true",
        help="skip the analyst KD leash even if a sibling file exists",
    )
    p.add_argument(
        "--smoke-seed", type=int, default=DEFAULT_SMOKE_SEED,
        help="sealed-smoke board seed (default 20260910 = sep10 arms)",
    )
    p.add_argument(
        "--parallel", type=int, default=SMOKE_PARALLEL,
        help="smoke --parallel (default 12)",
    )
    p.add_argument(
        "--stop-on-gate-fail", action="store_true",
        help="stop the loop after an epoch whose gate failed "
             "(default: record and continue)",
    )
    p.add_argument(
        "--train-epoch", type=int, default=None,
        help="INTERNAL (child mode): run epoch e's train stage and exit",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    if args.train_epoch is None:
        _attach_orchestrator_log(args.label)
    if args.train_epoch is not None:
        if not args.checkpoint:
            raise SystemExit("--train-epoch requires --checkpoint")
        data = Path(args.data)
        analyst_ce = _resolve_analyst_path(
            args.analyst_data, skip=args.no_analyst_ce,
            data=data, flag="--analyst-data",
        )
        analyst_kd = _resolve_analyst_path(
            args.analyst_anchor, skip=args.no_analyst_anchor,
            data=data, flag="--analyst-anchor",
        )
        return train_one_epoch(
            data, args.label, args.train_epoch,
            args.checkpoint, args.lr, analyst_ce, analyst_kd,
        )
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
