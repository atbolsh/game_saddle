#!/usr/bin/env python3
"""One-off t1 composition check across the Sep 10 prompt-cut branches.

Checks out each branch from origin (no manual fetch beforehand), runs
``python -m training.selftest t1``, and always returns the working tree
to master. Pure units: no GPU, no NAMS. Remote python env.

    python scripts/prompt_t1_ab.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

BRANCHES = (
    "master",
    "prompt-b-core-verbatim",
    "prompt-c-procedure",
)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True,
        capture_output=True,
    )


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


def main() -> int:
    rows: list[tuple[str, str, int]] = []
    try:
        for branch in BRANCHES:
            print(f"\n=== t1  branch={branch} ===", flush=True)
            try:
                _checkout_origin(branch)
            except RuntimeError as exc:
                print(f"  CHECKOUT FAILED: {exc}", flush=True)
                rows.append((branch, "checkout", 1))
                continue
            cmd = [sys.executable, "-m", "training.selftest", "t1"]
            print("  " + " ".join(cmd), flush=True)
            rc = int(subprocess.run(cmd, cwd=REPO_ROOT).returncode)
            print(f"  exit {rc}", flush=True)
            rows.append((branch, "ran", rc))
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

    print("\nbranch                         status    rc")
    print("-" * 44)
    for branch, status, rc in rows:
        print(f"{branch:30s} {status:8s} {rc!s:>3}")
    failed = sum(1 for _, _, rc in rows if rc != 0)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
