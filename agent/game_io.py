"""Game level generation, rendering, and Settings serialization.

Wraps the ``game.discreteEngine`` package. Datagen uses only "bare" levels
(4 boundary walls + 1 gold piece) via :func:`new_bare_game`. A notebook-only
multi-gold factory (:func:`new_multi_gold_game`) and a boundary-openings
oracle (:func:`boundary_openings`) live alongside it.

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
import random
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
# the model just finishes its message (its native end-of-turn/eos token)
# without emitting a move token.
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

# Side-wall thickness in the engine (discreteEngine.side_wall_width = 50/800).
# boundary_openings uses 1.5x this as the "this rect is a boundary wall" band.
_SIDE_WALL_WIDTH = 50 / 800
_OPENING_BAND = 1.5 * _SIDE_WALL_WIDTH
_OPENING_MIN_WIDTH = 0.01
_MULTI_GOLD_TRIES = 200


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


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted((min(a, b), max(a, b)) for a, b in intervals)
    out = [ordered[0]]
    for lo, hi in ordered[1:]:
        last_lo, last_hi = out[-1]
        if lo <= last_hi:
            out[-1] = (last_lo, max(last_hi, hi))
        else:
            out.append((lo, hi))
    return out


def _complement(covered: list[tuple[float, float]],
                lo: float = 0.0, hi: float = 1.0) -> list[tuple[float, float]]:
    """Gaps in ``[lo, hi]`` not covered by the merged intervals."""
    gaps: list[tuple[float, float]] = []
    cursor = lo
    for a, b in _merge_intervals(covered):
        a = max(a, lo)
        b = min(b, hi)
        if a > cursor:
            gaps.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < hi:
        gaps.append((cursor, hi))
    return gaps


def boundary_openings(settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Continuous stretches of the unit-square boundary NOT covered by any
    wall -- the exits.

    Each opening is ``{"side": "left"|"right"|"top"|"bottom",
    "from": [x, y], "to": [x, y], "center": [x, y], "width": float}``.
    Coordinates are the same y-up system as the rest of the settings dict
    (larger y = higher on the presented screen).

    Algorithm, per side independently (corners are therefore counted on
    both adjacent sides -- documented, not a bug): collect axis-aligned
    (``angle == 0``) walls whose rect lies in that side's boundary band
    (inner face within 1.5 x the side-wall thickness of the edge); project
    each onto the edge axis; merge covered intervals; complement within
    ``[0, 1]``; drop slivers narrower than 0.01. A wall in the boundary
    band with ``angle != 0`` raises ``ValueError`` -- rotated boundary
    geometry is not guessed at.

    LIMITATION: band membership is decided from the UNROTATED extents
    ``[x, x+w] x [y, y+h]``, so a rotated wall anchored OUTSIDE the band
    that sweeps into it is silently ignored rather than raising. Safe
    today because every caller feeds ``random_side_walls`` output, whose
    side walls are always axis-aligned (``wall_theta = 0`` hardcoded); if
    interior/rotated walls (FUTURE_GOALS goal 2) ever reach this function,
    the band test must be redone with rotated extents first.
    """
    walls = settings.get("walls") or []
    band = _OPENING_BAND
    sides = {
        "left":   {"axis": "y", "edge": 0.0, "is_low": True},
        "right":  {"axis": "y", "edge": 1.0, "is_low": False},
        "bottom": {"axis": "x", "edge": 0.0, "is_low": True},
        "top":    {"axis": "x", "edge": 1.0, "is_low": False},
    }
    openings: list[dict[str, Any]] = []
    for side, spec in sides.items():
        covered: list[tuple[float, float]] = []
        for wall in walls:
            if len(wall) < 5:
                raise ValueError(
                    f"wall {wall!r} is not [x, y, w, h, angle]"
                )
            x, y, w, h, angle = (float(wall[0]), float(wall[1]),
                                 float(wall[2]), float(wall[3]),
                                 float(wall[4]))
            x0, x1 = min(x, x + w), max(x, x + w)
            y0, y1 = min(y, y + h), max(y, y + h)
            if spec["axis"] == "y":
                # left/right: wall must sit in the x-band of that edge
                if spec["is_low"]:
                    in_band = x0 <= band and x1 >= 0.0
                else:
                    in_band = x1 >= 1.0 - band and x0 <= 1.0
                proj = (y0, y1)
            else:
                if spec["is_low"]:
                    in_band = y0 <= band and y1 >= 0.0
                else:
                    in_band = y1 >= 1.0 - band and y0 <= 1.0
                proj = (x0, x1)
            if not in_band:
                continue
            if abs(angle) > 1e-9:
                raise ValueError(
                    f"boundary-band wall on the {side} side has nonzero "
                    f"angle {angle} (wall={wall!r}); openings are only "
                    f"defined for axis-aligned boundary walls"
                )
            covered.append(proj)
        for a, b in _complement(covered):
            width = b - a
            if width < _OPENING_MIN_WIDTH:
                continue
            mid = (a + b) / 2.0
            if side == "left":
                fr, to, center = [0.0, a], [0.0, b], [0.0, mid]
            elif side == "right":
                fr, to, center = [1.0, a], [1.0, b], [1.0, mid]
            elif side == "bottom":
                fr, to, center = [a, 0.0], [b, 0.0], [mid, 0.0]
            else:
                fr, to, center = [a, 1.0], [b, 1.0], [mid, 1.0]
            openings.append({
                "side": side, "from": fr, "to": to,
                "center": center, "width": width,
            })
    return openings


def new_multi_gold_game(
    gameSize: int = 768,
    n_gold: int | None = None,
    opening: str = "require",
    min_gold_separation: float = 0.15,
    min_gold_distance: float = MIN_GOLD_DISTANCE,
) -> discreteGame:
    """A room that may hold 0–3 golds and may or may not have a boundary
    opening. ``new_bare_game`` is untouched; this factory is the multi-gold
    notebook's constructor.

    ``n_gold`` None -> uniform random in {0,1,2,3}. ``opening`` is one of
    ``"require"`` (at least one exit), ``"forbid"`` (sealed -- the
    ``[END_GAME]``-correct empty-room case), or ``"any"``. Golds (when any)
    sit at least ``min_gold_distance`` from the agent and
    ``min_gold_separation`` from each other. Does NOT route through
    ``random_bare_settings`` (that function indexes ``gold[0]``).
    """
    if opening not in ("require", "forbid", "any"):
        raise ValueError(
            f"opening must be 'require', 'forbid', or 'any'; got {opening!r}"
        )
    if n_gold is not None and n_gold not in (0, 1, 2, 3):
        raise ValueError(f"n_gold must be 0..3 or None; got {n_gold!r}")

    engine = discreteGame(envMode=True)
    target_n = n_gold

    for _ in range(_MULTI_GOLD_TRIES):
        walls = engine.random_walls(num_extra_walls=0)
        openings = boundary_openings({"walls": walls})
        if opening == "require" and not openings:
            continue
        if opening == "forbid" and openings:
            continue
        ax, ay = engine.random_valid_coords(walls, engine.typical_agent_r)
        n = target_n if target_n is not None else random.randint(0, 3)
        golds: list[tuple[float, float]] = []
        placed = True
        for _g in range(n):
            found = None
            for _try in range(200):
                gx, gy = engine.random_valid_coords(
                    walls, engine.typical_gold_r)
                if math.hypot(gx - ax, gy - ay) < min_gold_distance:
                    continue
                if any(math.hypot(gx - px, gy - py) < min_gold_separation
                       for px, py in golds):
                    continue
                found = (gx, gy)
                break
            if found is None:
                placed = False
                break
            golds.append(found)
        if not placed:
            continue
        settings = Settings(
            gameSize=gameSize,
            agent_r=engine.typical_agent_r,
            gold_r=engine.typical_gold_r,
            walls=walls,
            gold=list(golds),
            agent_x=ax,
            agent_y=ay,
            direction=random.uniform(0, 2 * math.pi),
        )
        return discreteGame(settings=settings, envMode=True)

    raise ValueError(
        f"new_multi_gold_game: no board matched n_gold={n_gold!r} "
        f"opening={opening!r} after {_MULTI_GOLD_TRIES} tries"
    )


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


def truncate_at_first_move_token(text: str) -> str:
    """Keep ``text`` through the first bracketed move token, drop the rest.

    Notebook human-takeover: a hand-typed ``[FORWARD]`` / ``[CLOCK]`` /
    ``[ANTICLOCK]`` ends the move the same way a generate stop-string
    would. No token -> the string is unchanged. Uses the same
    :data:`_MOVE_RE` as :func:`parse_action`."""
    m = _MOVE_RE.search(text)
    return text[: m.end()] if m else text


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
