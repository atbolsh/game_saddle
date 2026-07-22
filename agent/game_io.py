"""Game level generation, rendering, and Settings serialization.

Wraps the ``game.discreteEngine`` package. Only "bare" levels are generated
for now: 4 boundary walls + 1 gold piece near the agent, via
``discreteGame.random_bare_settings``.

COORDINATE CONVENTION (single source of truth, engine and prompts agree):
  - The world is y-UP: larger y = higher on the presented screen. The engine
    draws on a y-down pygame surface internally and flips ONCE at
    presentation (``getData``), so the Settings numbers match the picture.
  - ``direction`` (theta) is a COMPASS BEARING: theta=0 points straight up
    (12 o'clock) and theta increases CLOCKWISE as seen on screen. The facing
    vector in world coordinates is the standard bearing idiom
    ``(sin theta, cos theta)``, and the agent faces a target when
    ``theta ~= atan2(x_target - x_agent, y_target - y_agent)`` (x-difference
    FIRST -- ``bearing = atan2(east, north)``, no sign flips anywhere).
  - Wall caveat: a wall's ``[x, y, w, h, angle]`` anchor is its display
    bottom-left corner with ``h`` extending up-screen, but a nonzero wall
    ``angle`` still appears ANTICLOCKWISE on screen (the wall-drawing math
    predates the flip). All bare-game walls have angle 0, so this never
    surfaces in current levels.

Moves exposed to the agent map directly onto same-named engine methods:
  - ``CLOCK``    -> ``swivel_clock``     (turn clockwise on screen)
  - ``ANTICLOCK``-> ``swivel_anticlock`` (turn counter-clockwise on screen)
  - ``FORWARD``  -> ``stepForward``      (advance one step)
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# The game package is imported via ``from game.discreteEngine import *`` in
# game/__init__.py, so ``discreteGame`` and ``Settings`` are top-level names
# after importing the package. We import explicitly for clarity.
from game.discreteEngine import discreteGame, Settings  # noqa: F401
from game.levels.skeleton import Settings as SettingsClass  # noqa: F401

# Map agent-facing action names to engine method names (identity: engine
# method names match their on-screen effect).
ACTION_MAP: dict[str, str] = {
    "CLOCK": "swivel_clock",
    "ANTICLOCK": "swivel_anticlock",
    "FORWARD": "stepForward",
}
ACTIONS = list(ACTION_MAP.keys())

# Wire format the model emits to make a move: distinctive bracketed tokens
# (e.g. ``[FORWARD]``). They never collide with ordinary prose and tokenize
# cleanly, so we can use them as generation stop strings: the model's turn is
# a loop of "reason -> emit one move token -> we stop, apply it, re-render,
# generate again on the new frame". Ending the turn needs no special token --
# the model just finishes its message (Gemma's native ``<end_of_turn>``) without
# emitting a move token.
MOVE_STOP_STRINGS = [f"[{a}]" for a in ACTIONS]  # ["[CLOCK]", "[ANTICLOCK]", "[FORWARD]"]
_MOVE_RE = re.compile(r"\[(" + "|".join(ACTIONS) + r")\]", re.IGNORECASE)

# A move word WITHOUT brackets (e.g. plain 'ANTICLOCK'). Never a move -- but
# when a reply contains one and no bracketed token, the model almost
# certainly INTENDED a move and fumbled the format, which the harness should
# describe loudly rather than mislabel as a prose answer.
BARE_MOVE_RE = re.compile(r"\b(" + "|".join(ACTIONS) + r")\b", re.IGNORECASE)

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


# Minimum agent<->gold separation for a freshly generated bare game, in
# normalised board units ([0,1] square). The engine's default places the gold
# within ~0.1 of the agent ("almost on top of it"); we want a real gap so the
# agent has to navigate.
MIN_GOLD_DISTANCE = 0.6


def new_bare_game(
    gameSize: int = 768,
    min_gold_distance: float = MIN_GOLD_DISTANCE,
) -> discreteGame:
    """Create a fresh bare discrete game (env mode, no GUI window).

    The engine places the single gold piece within ``max_agent_offset`` of the
    agent (default ~0.1), which lands it almost on top of the agent. We instead
    require the gold to be at least ``min_gold_distance`` away (normalised board
    units). Since the reachable interior is only ~0.9 wide, a central agent
    leaves little room for a far gold, so we re-roll the whole level (agent +
    walls + gold) until a valid far placement is found, keeping the best-found
    layout as a fallback. The engine itself is never modified.
    """
    engine = discreteGame(envMode=True)

    best_settings = None
    best_dist = -1.0
    # Outer loop re-rolls agent/walls; inner loop searches for a far gold that
    # is also wall-valid, using the engine's own coordinate sampler.
    for _ in range(64):
        # ``max_agent_offset`` large so the engine's initial gold can be anywhere;
        # we override it below regardless.
        bare = engine.random_bare_settings(gameSize=gameSize, max_agent_offset=1.0)
        ax, ay = bare.agent_x, bare.agent_y
        for _ in range(200):
            gx, gy = engine.random_valid_coords(bare.walls, engine.typical_gold_r)
            dist = math.hypot(gx - ax, gy - ay)
            if dist > best_dist:
                best_dist = dist
                bare.gold = [(gx, gy)]
                best_settings = bare
            if dist >= min_gold_distance:
                return discreteGame(settings=bare, envMode=True)

    # Fallback: no layout hit the target after many tries -- use the farthest
    # gold placement we saw (still a valid, non-overlapping position).
    return discreteGame(settings=best_settings, envMode=True)


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
    """Return the engine action for the move token in ``text`` (one of
    ``ACTIONS``), or ``None`` if the model emitted no move token.

    Only the bracketed tokens (``[CLOCK]`` / ``[ANTICLOCK]`` / ``[FORWARD]``)
    count as moves -- plain prose that merely mentions "forward" does not. When
    generation is stopped via :data:`MOVE_STOP_STRINGS` the move token sits at
    the tail, so we take the last match to be safe."""
    matches = _MOVE_RE.findall(text)
    if not matches:
        return None
    return matches[-1].upper()


def find_bare_move(text: str) -> str | None:
    """Return the LAST bare (unbracketed) move word in ``text`` -- e.g.
    'ANTICLOCK' without brackets -- or ``None``.

    Only meaningful when :func:`parse_action` found no bracketed token: a
    bare word is never applied as a move, but its presence means the model
    probably intended one and got the format wrong, and callers should say
    so explicitly instead of treating the reply as plain prose."""
    if parse_action(text) is not None:
        return None
    matches = BARE_MOVE_RE.findall(text)
    if not matches:
        return None
    return matches[-1].upper()


def dump_settings_json(d: dict[str, Any]) -> str:
    return json.dumps(d)
