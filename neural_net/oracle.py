"""Point-target oracle and the S1 pretraining target distribution.

The bearing matches ``training/generate_game_traces.py`` ``_oracle_meta``:
``atan2(dx, dy)``, theta a compass bearing, positive relative bearing means
the target is clockwise of the facing. ``noop`` when the point is within
70% of ``agent_r`` (a buffer inside the sprite). Otherwise ``FORWARD``
inside the zone of allowance, and the shorter turn outside it.

The zone is the beam the sprite sweeps by pressing ``FORWARD``, then a
cone. One ``FORWARD`` step is 1/16 of the board, the ``stepForward``
default. Out to 1.2 of those steps the beam is straight: the point is in
front of the center and the facing ray comes within ``agent_r`` of it.
Past that line the beam's two edges open, and the angle between the
edges is 12° (6° each side of the beam, one turn step). A point in front
at forward distance ``along`` is inside when its perpendicular distance
is at most ``agent_r`` for ``along`` ≤ 1.2/16, and at most
``agent_r + (along - 1.2/16) * tan(6°)`` further out. The picture is
``s1_s2_design_notes/zone_of_allowance.svg``.

The old mixture (inside / opening / gold / free space) is scaled by 0.8.
The other 20% is a shell just outside the sprite: 10% anywhere, 10% with
the agent center within 1.5 sprite diameters of a wall. The inside draw
is uniform in the noop disc only, so every one of those targets is a
``noop``.
"""

from __future__ import annotations

import math
import random
from typing import Any

from agent.game_io import boundary_openings, settings_to_dict

ACTIONS: tuple[str, ...] = ("noop", "CLOCK", "ANTICLOCK", "FORWARD")
ACTION_INDEX = {name: i for i, name in enumerate(ACTIONS)}

#: ``noop`` inside this fraction of the agent radius. The rest of the
#: disc is still a move, so a target on the rim is not "arrived".
#: Inside-agent training draws use this same radius.
NOOP_RADIUS_FRAC = 0.70

#: One ``FORWARD`` primitive. ``discreteGame.stepForward`` default.
FORWARD_STEP = 1.0 / 16
#: The straight beam runs this many ``FORWARD`` steps. The cone opens
#: after that.
ALLOWANCE_STRAIGHT_STEPS = 1.2
#: Half of the 12° opening. 6° is one ``CLOCK`` step (``pi/30``).
_CONE_HALF_RAD = math.pi / 30
_CONE_TAN = math.tan(_CONE_HALF_RAD)
_STRAIGHT_REACH = ALLOWANCE_STRAIGHT_STEPS * FORWARD_STEP

#: ``discreteEngine.draw_agent`` draws the eye at ``0.4 * agent_r``.
EYE_RADIUS_FRAC = 0.4
#: Nearby targets sit at most this many eye-radii off the sprite surface.
NEAR_SURFACE_EYE_RADII = 0.5
#: Near-wall draws place the agent center at most this many sprite
#: diameters from a wall. One diameter is ``2 * agent_r``.
NEAR_WALL_DIAMETERS = 1.5

_NEAR_P = 0.10
_NEAR_WALL_P = 0.10
_LEGACY_SCALE = 1.0 - _NEAR_P - _NEAR_WALL_P
_INSIDE_P = 0.25 * _LEGACY_SCALE
_OPENING_P = 0.20 * _LEGACY_SCALE
_GOLD_P = 0.20 * _LEGACY_SCALE


def _allowance_half_width(along: float, agent_r: float) -> float:
    """Half-width of the zone at forward distance ``along``.

    ``along`` is the projection onto the facing vector. Up to
    ``_STRAIGHT_REACH`` the beam is as wide as the sprite. Past it the
    edges leave the beam at 6° each, so a slice perpendicular to facing
    widens by ``tan(6°)`` per unit of extra distance.
    """
    extra = along - _STRAIGHT_REACH
    if extra <= 0.0:
        return agent_r
    return agent_r + extra * _CONE_TAN


def point_oracle(settings: Any, x: float, y: float) -> str:
    """Next primitive that brings the agent disc onto ``(x, y)``.

    ``noop`` inside ``NOOP_RADIUS_FRAC * agent_r``. ``FORWARD`` inside
    the zone of allowance. Outside it, ``CLOCK`` when the relative
    bearing is positive (clockwise) and ``ANTICLOCK`` otherwise.
    """
    ax = float(settings.agent_x)
    ay = float(settings.agent_y)
    ar = float(settings.agent_r)
    dx = float(x) - ax
    dy = float(y) - ay
    if math.hypot(dx, dy) <= NOOP_RADIUS_FRAC * ar:
        return "noop"
    theta = float(settings.direction)
    rel = (math.atan2(dx, dy) - theta + math.pi) % (2 * math.pi) - math.pi
    fx, fy = math.sin(theta), math.cos(theta)
    along = dx * fx + dy * fy
    perp = dx * fy - dy * fx
    if along > 0.0 and abs(perp) <= _allowance_half_width(along, ar):
        return "FORWARD"
    return "CLOCK" if rel > 0 else "ANTICLOCK"


def sample_target(game: Any, rng: random.Random) -> tuple[float, float]:
    """One target coordinate for ``game``.

    One uniform draw:

    * 10% — just outside the sprite, at most half an eye radius off
      the surface;
    * 10% — the same shell, after the agent is moved so its center is
      at most 1.5 sprite diameters from a wall (the board's walls and
      gold stay put);
    * 20% — uniform inside the noop disc (70% of ``agent_r``);
    * 16% — an opening center plus Gaussian jitter (sigma = width / 4),
      or the free-space case when the board has no opening;
    * 16% — a gold center plus Gaussian jitter (sigma = ``gold_r``),
      or the free-space case when the board has no gold;
    * otherwise — uniform in the unit square, rejected while the point
      lies inside a wall.

    The opening draw does not fall through into the gold draw. Jittered
    opening, gold, and near-surface points are kept even if they land in
    a wall or outside the square; the oracle labels whatever point came
    out. The 20/16/16/28 split is the previous 25/20/20/35 mixture
    scaled by 0.8.
    """
    settings = game.settings
    u = rng.random()
    if u < _NEAR_P:
        return _sample_near(settings, rng)
    if u < _NEAR_P + _NEAR_WALL_P:
        _move_agent_near_wall(game, rng)
        return _sample_near(game.settings, rng)
    if u < _NEAR_P + _NEAR_WALL_P + _INSIDE_P:
        return _sample_inside(settings, rng)
    if u < _NEAR_P + _NEAR_WALL_P + _INSIDE_P + _OPENING_P:
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
    if u < _NEAR_P + _NEAR_WALL_P + _INSIDE_P + _OPENING_P + _GOLD_P:
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


def _sample_inside(settings: Any, rng: random.Random) -> tuple[float, float]:
    """Uniform in the noop disc: radius ``NOOP_RADIUS_FRAC * agent_r``."""
    ang = rng.random() * 2 * math.pi
    rad = math.sqrt(rng.random()) * (NOOP_RADIUS_FRAC * float(settings.agent_r))
    return (
        float(settings.agent_x) + rad * math.cos(ang),
        float(settings.agent_y) + rad * math.sin(ang),
    )


def _sample_near(settings: Any, rng: random.Random) -> tuple[float, float]:
    """Uniform angle, gap uniform in ``[0, 0.5 * eye radius]`` past the surface."""
    ar = float(settings.agent_r)
    eye_r = EYE_RADIUS_FRAC * ar
    gap = rng.random() * (NEAR_SURFACE_EYE_RADII * eye_r)
    rad = ar + gap
    ang = rng.random() * 2 * math.pi
    return (
        float(settings.agent_x) + rad * math.cos(ang),
        float(settings.agent_y) + rad * math.sin(ang),
    )


def _move_agent_near_wall(game: Any, rng: random.Random) -> None:
    """Move the agent to a legal center within ``NEAR_WALL_DIAMETERS``."""
    settings = game.settings
    limit = NEAR_WALL_DIAMETERS * (2.0 * float(settings.agent_r))
    for _ in range(10000):
        x, y = rng.random(), rng.random()
        if not game.full_wall_check(x, y):
            continue
        if _nearest_wall_distance(game, x, y) <= limit:
            settings.agent_x = float(x)
            settings.agent_y = float(y)
            return
    raise RuntimeError(
        "sample_target: no agent position within "
        f"{NEAR_WALL_DIAMETERS} sprite diameters of a wall after 10000 draws"
    )


def _nearest_wall_distance(game: Any, x: float, y: float) -> float:
    walls = list(game.settings.walls)
    if not walls:
        raise RuntimeError("sample_target: board has no walls")
    return min(_distance_to_wall(game, x, y, wall) for wall in walls)


def _distance_to_wall(game: Any, x: float, y: float, wall: Any) -> float:
    """Distance from ``(x, y)`` to the wall rectangle.

    Same wall frame as ``discreteGame._contact_normal``: ``backRot``, then
    the closest point on the axis-aligned rectangle.
    """
    wall_x, wall_y, wall_w, wall_h, wall_theta = (float(v) for v in wall[:5])
    ax, ay = game.backRot(x, y, wall_theta)
    left, top = game.backRot(wall_x, wall_y, wall_theta)
    right = left + wall_w
    bottom = top + wall_h
    cx = min(max(ax, left), right)
    cy = min(max(ay, top), bottom)
    return math.hypot(ax - cx, ay - cy)


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
