"""Game level generation, rendering, and Settings serialization.

We **wrap** the existing ``game.discreteEngine`` package; we do not modify
``discreteEngine.py`` or ``game/levels/skeleton.py``. Only "bare" levels are
generated for now: 4 boundary walls + 1 gold piece near the agent, via
``discreteGame.random_bare_settings``.

Moves exposed to the agent:
  - ``CLOCK``    -> ``swivel_clock``     (turn clockwise)
  - ``ANTICLOCK``-> ``swivel_anticlock`` (turn counter-clockwise)
  - ``FORWARD``  -> ``stepForward``      (advance one step)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# The game package is imported via ``from game.discreteEngine import *`` in
# game/__init__.py, so ``discreteGame`` and ``Settings`` are top-level names
# after importing the package. We import explicitly for clarity.
from game.discreteEngine import discreteGame, Settings  # noqa: F401
from game.levels.skeleton import Settings as SettingsClass  # noqa: F401

# Map agent-facing action names to engine method names.
ACTION_MAP: dict[str, str] = {
    "CLOCK": "swivel_clock",
    "ANTICLOCK": "swivel_anticlock",
    "FORWARD": "stepForward",
}
ACTIONS = list(ACTION_MAP.keys())

# Keys we serialise on a Settings object. ``walls`` and ``gold`` are lists
# of lists of floats; everything else is a scalar.
_SETTINGS_FIELDS = [
    "gameSize",
    "direction",
    "agent_x",
    "agent_y",
    "agent_r",
    "gold_r",
]


def new_bare_game(gameSize: int = 64) -> discreteGame:
    """Create a fresh bare discrete game (env mode, no GUI window)."""
    game = discreteGame(envMode=True)
    # ``random_bare_settings`` lives on the engine instance.
    bare = game.random_bare_settings(gameSize=gameSize)
    return discreteGame(settings=bare, envMode=True)


def settings_to_dict(s: Settings) -> dict[str, Any]:
    """Serialise a Settings object to a plain dict (JSON-safe)."""
    out: dict[str, Any] = {k: getattr(s, k) for k in _SETTINGS_FIELDS}
    out["gold"] = [list(g) for g in s.gold]
    out["walls"] = [list(w) for w in s.walls]
    return out


def settings_from_dict(d: dict[str, Any]) -> Settings:
    """Inverse of :func:`settings_to_dict`."""
    return Settings(
        gameSize=int(d["gameSize"]),
        direction=float(d["direction"]),
        agent_x=float(d["agent_x"]),
        agent_y=float(d["agent_y"]),
        agent_r=float(d["agent_r"]),
        gold_r=float(d["gold_r"]),
        gold=[list(g) for g in d.get("gold", [])],
        walls=[list(w) for w in d.get("walls", [])],
    )


def game_to_settings_dict(game: discreteGame) -> dict[str, Any]:
    return settings_to_dict(game.settings)


def game_from_settings_dict(d: dict[str, Any]) -> discreteGame:
    return discreteGame(settings=settings_from_dict(d), envMode=True)


def render_frame_array(game: discreteGame) -> np.ndarray:
    """Return the current frame as a uint8 HxWx3 RGB array."""
    arr = game.getData()  # float in [0,1], shape (W, H, 3) per pygame surfarray
    return (arr * 255).astype("uint8")


def render_frame_png(game: discreteGame, path: str | os.PathLike) -> tuple[int, int]:
    """Render the current frame to a PNG. Returns (width, height)."""
    arr = render_frame_array(game)
    # pygame surfarray is (width, height, 3); PIL wants (height, width, 3).
    img = Image.fromarray(np.transpose(arr, (1, 0, 2)), mode="RGB")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return img.size  # (width, height)


def apply_action(game: discreteGame, action_name: str) -> int:
    """Apply an agent action to the game; return gold collected this step.

    Raises ``ValueError`` for unknown actions.
    """
    if action_name not in ACTION_MAP:
        raise ValueError(f"Unknown action: {action_name!r}")
    method = getattr(game, ACTION_MAP[action_name])
    collected = method()
    return int(collected or 0)


def gold_remaining(game: discreteGame) -> int:
    return len(game.settings.gold)


def parse_action(text: str) -> str | None:
    """Find the first occurring action keyword in ``text`` (case-insensitive,
    word-boundary aware). Returns one of ACTIONS or None."""
    import re

    pattern = r"\b(" + "|".join(ACTIONS) + r")\b"
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def dump_settings_json(d: dict[str, Any]) -> str:
    return json.dumps(d)
