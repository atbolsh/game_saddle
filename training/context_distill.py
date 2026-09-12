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
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.game_traces import (  # noqa: E402
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
    ):
        super().__init__(path, noise_strength, noise_seed)
        self.name = name or f"distill_{self.path.stem}"
        self.weight = 1.0
        self.drop_unverified = drop_unverified

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


def _run_stage(cmd: list[str], stage: str) -> int:
    logger.info("[%s] starting: %s", stage, " ".join(cmd))
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    hours = (time.perf_counter() - t0) / 3600
    if proc.returncode == 0:
        logger.info("[%s] finished in %.2fh", stage, hours)
    else:
        logger.error("[%s] FAILED (exit %d) after %.2fh",
                     stage, proc.returncode, hours)
    return proc.returncode


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
    if _run_stage(cmd, f"train{e}") != 0:
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
    from training.run_weekend import _sweep_noise_dirs

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
    _save_state(args.label, state)
    logger.info("state file: %s", _state_path(args.label))

    resume = args.checkpoint
    failures = 0

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
            if not ckpt:
                failures += 1
                state["phase"] = f"train{e}_failed"
                _save_state(args.label, state)
                logger.error(
                    "STOPPING: train%d produced no checkpoint. The "
                    "child died or left no last_good_checkpoint. Do not "
                    "advance to the next epoch. Inspect "
                    "logs/train_%s_e%d_*/ (train_log.txt, events.jsonl) "
                    "and the parent stderr for the exit.",
                    e, args.label, e,
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
