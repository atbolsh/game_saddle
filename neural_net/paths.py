"""Repo-root paths for the S1/S2 package."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def weights_root() -> Path:
    """``weights/`` anchored at the repo root when the config path is relative."""
    from agent.config import CONFIG

    path = Path(CONFIG.weights_dir)
    return path if path.is_absolute() else REPO_ROOT / path
