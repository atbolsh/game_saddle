"""Point-target oracle and the S1 pretraining target distribution.

The bearing matches ``training/generate_game_traces.py`` ``_oracle_meta``:
``atan2(dx, dy)``, theta a compass bearing, positive relative bearing means
the target is clockwise of the facing. A point already inside the agent
disc is ``noop``. Otherwise ``FORWARD`` when the facing ray comes within
``agent_r`` of the point and the point is in front; else the shorter turn.
"""

from __future__ import annotations

import math
import random
from typing import Any

from agent.game_io import boundary_openings, settings_to_dict

ACTIONS: tuple[str, ...] = ("noop", "CLOCK", "ANTICLOCK", "FORWARD")
ACTION_INDEX = {name: i for i, name in enumerate(ACTIONS)}

_INSIDE_P = 0.25
_OPENING_P = 0.20
_GOLD_P = 0.20


def point_oracle(settings: Any, x: float, y: float) -> str:
    """Next primitive that walks the agent disc onto ``(x, y)``."""
    ax = float(settings.agent_x)
    ay = float(settings.agent_y)
    ar = float(settings.agent_r)
    dx = float(x) - ax
    dy = float(y) - ay
    if math.hypot(dx, dy) <= ar:
        return "noop"
    theta = float(settings.direction)
    rel = (math.atan2(dx, dy) - theta + math.pi) % (2 * math.pi) - math.pi
    fx, fy = math.sin(theta), math.cos(theta)
    ray_hit = dx * fx + dy * fy > 0 and abs(dx * fy - dy * fx) <= ar
    if ray_hit:
        return "FORWARD"
    return "CLOCK" if rel > 0 else "ANTICLOCK"


def sample_target(game: Any, rng: random.Random) -> tuple[float, float]:
    """One target coordinate for ``game``.

    One uniform draw:

    * 25% — uniform inside the agent disc;
    * 20% — an opening center plus Gaussian jitter (sigma = width / 4),
      or the free-space case when the board has no opening;
    * 20% — a gold center plus Gaussian jitter (sigma = ``gold_r``),
      or the free-space case when the board has no gold;
    * otherwise — uniform in the unit square, rejected while the point
      lies inside a wall.

    The opening draw does not fall through into the gold draw. Jittered
    opening and gold points are kept even if they land in a wall or
    outside the square; the oracle labels whatever point came out.
    """
    settings = game.settings
    u = rng.random()
    if u < _INSIDE_P:
        ang = rng.random() * 2 * math.pi
        rad = math.sqrt(rng.random()) * float(settings.agent_r)
        return (
            float(settings.agent_x) + rad * math.cos(ang),
            float(settings.agent_y) + rad * math.sin(ang),
        )
    if u < _INSIDE_P + _OPENING_P:
        openings = boundary_openings(settings_to_dict(settings))
        if openings:
            opening = openings[rng.randrange(len(openings))]
            cx, cy = opening["center"]
            sigma = float(opening["width"]) / 4.0
            return (
                float(cx) + rng.gauss(0.0, sigma),
                float(cy) + rng.gauss(0.0, sigma),
            )
        return _sample_free(game, rng)
    if u < _INSIDE_P + _OPENING_P + _GOLD_P:
        golds = list(settings.gold)
        if golds:
            gx, gy = golds[rng.randrange(len(golds))]
            sigma = float(settings.gold_r)
            return (
                float(gx) + rng.gauss(0.0, sigma),
                float(gy) + rng.gauss(0.0, sigma),
            )
        return _sample_free(game, rng)
    return _sample_free(game, rng)


def _sample_free(game: Any, rng: random.Random) -> tuple[float, float]:
    """Uniform in the unit square, outside every wall. ``agent_r=0`` so
    the test is the point itself, not a disc."""
    for _ in range(10000):
        x, y = rng.random(), rng.random()
        if game.full_wall_check(x, y, agent_r=0):
            return x, y
    raise RuntimeError(
        "sample_target: no wall-free point in the unit square after 10000 draws"
    )
