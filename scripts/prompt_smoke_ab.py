#!/usr/bin/env python3
"""One-off 5-arm smoke A/B for the Sep 10 player-prompt cut.

Checks out each arm's branch from origin (no manual fetch beforehand),
runs the sealed one-gold smoke (same knobs as run_weekend._smoke_eval),
and always returns the working tree to master.

    python scripts/prompt_smoke_ab.py

NAMS must be up. Remote GPU. Skip an arm if
``data_game/<label>/generation_stats.json`` already exists.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_GAME = REPO_ROOT / "data_game"

#: Shared seed so all five arms see the same 8 boards.
SEED = 20260910
PARALLEL = 12
GAMES = 8
MAX_GENERATIONS = 400
QUESTION_RATE = 0.075

ARMS: list[tuple[str, str, str]] = [
    ("sep10_smoke_ctrl_aug18r5", "master",
     "aug18_restart_iter5_step292"),
    ("sep10_smoke_pB_aug18r5", "prompt-b-core-verbatim",
     "aug18_restart_iter5_step292"),
    ("sep10_smoke_pB_aug27s313", "prompt-b-core-verbatim",
     "aug27_big_step_iter1_step313"),
    ("sep10_smoke_pC_aug18r5", "prompt-c-procedure",
     "aug18_restart_iter5_step292"),
    ("sep10_smoke_pC_aug27s313", "prompt-c-procedure",
     "aug27_big_step_iter1_step313"),
]


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True,
        capture_output=True,
    )


def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _checkout_origin(branch: str) -> None:
    fetch = _git("fetch", "origin", branch)
    if fetch.returncode != 0:
        raise RuntimeError(
            f"git fetch origin {branch} failed:\n"
            f"{fetch.stderr or fetch.stdout}"
        )
    co = _git("checkout", "-B", branch, f"origin/{branch}")
    if co.returncode != 0:
        raise RuntimeError(
            f"git checkout -B {branch} origin/{branch} failed:\n"
            f"{co.stderr or co.stdout}"
        )
    print(f"  checked out origin/{branch}", flush=True)


def _run_arm(label: str, checkpoint: str) -> int:
    traces = DATA_GAME / label / "traces.jsonl"
    cmd = [
        sys.executable, "-m", "training.generate_game_traces",
        "--label", label,
        "--parallel", str(PARALLEL),
        "--games", str(GAMES),
        "--max-generations", str(MAX_GENERATIONS),
        "--seed", str(SEED),
        "--question-rate", str(QUESTION_RATE),
        "--checkpoint", checkpoint,
    ]
    if traces.exists():
        cmd.append("--append")
    print("  " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    return int(proc.returncode)


def main() -> int:
    rows: list[tuple[str, str, int | str, int]] = []
    try:
        for label, branch, checkpoint in ARMS:
            stats = DATA_GAME / label / "generation_stats.json"
            print(f"\n=== {label}  branch={branch}  ckpt={checkpoint} ===",
                  flush=True)
            if stats.is_file():
                n = _count_jsonl(DATA_GAME / label / "traces.jsonl")
                print(f"  skip: {stats} exists ({n} traces)", flush=True)
                rows.append((label, "skipped", 0, n))
                continue
            try:
                _checkout_origin(branch)
            except RuntimeError as exc:
                print(f"  CHECKOUT FAILED: {exc}", flush=True)
                rows.append((label, "checkout", 1, 0))
                continue
            rc = _run_arm(label, checkpoint)
            n = _count_jsonl(DATA_GAME / label / "traces.jsonl")
            print(f"  exit {rc}  traces={n}", flush=True)
            rows.append((label, "ran", rc, n))
    finally:
        back = _git("checkout", "master")
        if back.returncode != 0:
            print(
                "WARNING: could not return to master:\n"
                f"{back.stderr or back.stdout}",
                flush=True,
            )
        else:
            print("\nreturned to master", flush=True)

    print("\nlabel                          status    rc  traces")
    print("-" * 56)
    for label, status, rc, n in rows:
        print(f"{label:30s} {status:8s} {rc!s:>3}  {n}")
    failed = sum(1 for _, status, rc, _ in rows
                 if status != "skipped" and rc != 0)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
