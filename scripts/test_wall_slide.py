#!/usr/bin/env python3
"""Wall-slide scenarios for discreteGame. Run on the remote box:

    python scripts/test_wall_slide.py

Assertions are in each wall's own frame (via backRot), never raw x/y.
Exits nonzero on any failure.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from game.discreteEngine import discreteGame  # noqa: E402
from game.levels.skeleton import Settings  # noqa: E402

GAME_SIZE = 768
MIN_STEP = 1.0 / GAME_SIZE
AGENT_R = 0.05
DEFAULT_LIM = 1.0 / 16
WALL_W = 0.0625
WALL_H = 0.5

failures: list[str] = []
passes = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passes
    if cond:
        print(f"PASS  {name}")
        passes += 1
    else:
        msg = f"FAIL  {name}" + (f"  ({detail})" if detail else "")
        print(msg)
        failures.append(msg)


def make_game(agent_x, agent_y, direction, walls) -> discreteGame:
    settings = Settings(
        gameSize=GAME_SIZE,
        direction=direction,
        agent_x=agent_x,
        agent_y=agent_y,
        agent_r=AGENT_R,
        gold_r=1.0 / 64,
        gold=[],
        walls=walls,
    )
    return discreteGame(settings, envMode=True)


def to_wall_frame(game, wall, x, y):
    return game.backRot(x, y, wall[4])


def from_wall_frame(game, wall, wx, wy):
    return game.backRot(wx, wy, -wall[4])


def wall_limits(game, wall):
    left, top = game.backRot(wall[0], wall[1], wall[4])
    return left, top, left + wall[2], top + wall[3]


def facing_from_vec(vx, vy) -> float:
    return math.atan2(vx, vy)


def dist_to_rect(game, wall, x, y) -> float:
    ax, ay = to_wall_frame(game, wall, x, y)
    left, top, right, bot = wall_limits(game, wall)
    cx = min(max(ax, left), right)
    cy = min(max(ay, top), bot)
    return math.hypot(ax - cx, ay - cy)


def _axis_wall(theta, ul=(0.4, 0.25), w=WALL_W, h=WALL_H):
    return [ul[0], ul[1], w, h, theta]


def _place_flush_left(game, wall, along=0.5, gap=2 * MIN_STEP):
    """World pose flush outside the wall's left face (wall-frame -x)."""
    left, top, right, bot = wall_limits(game, wall)
    wx = left - AGENT_R - gap
    wy = top + along * (bot - top)
    return from_wall_frame(game, wall, wx, wy)


# ------------------------------------------------------------------ cases


def test_diagonal_grazing():
    gap = 2 * MIN_STEP
    tilt = math.radians(30)
    expected_tangent = DEFAULT_LIM * math.cos(tilt)
    for theta in (0.0, math.pi / 6, math.pi / 4, 1.0):
        wall = _axis_wall(theta)
        g = make_game(0.5, 0.5, 0.0, [wall])
        sx, sy = _place_flush_left(g, wall, along=0.5, gap=gap)
        g.settings.agent_x, g.settings.agent_y = sx, sy
        # Into the wall (+x in wall frame) and along +y, 30 deg off tangent.
        vwx, vwy = math.sin(tilt), math.cos(tilt)
        vx, vy = from_wall_frame(g, wall, vwx, vwy)
        # from_wall_frame on a direction: backRot is linear, origin-centered.
        g.settings.direction = facing_from_vec(vx, vy)
        start = (g.settings.agent_x, g.settings.agent_y)
        start_w = to_wall_frame(g, wall, *start)
        g.stepForward()
        end = (g.settings.agent_x, g.settings.agent_y)
        end_w = to_wall_frame(g, wall, *end)
        outward = start_w[0] - end_w[0]  # left face: -x is outward
        tangent = end_w[1] - start_w[1]
        tag = f"grazing theta={theta:.3f}"
        check(
            f"{tag} still free",
            g.full_wall_check(end[0], end[1]),
            f"pos=({end[0]:.4f},{end[1]:.4f})",
        )
        check(
            f"{tag} normal displacement",
            -gap - MIN_STEP <= outward <= MIN_STEP,
            f"outward={outward:.6f} bound=[{-gap - MIN_STEP:.6f},{MIN_STEP:.6f}]",
        )
        check(
            f"{tag} tangent displacement",
            abs(tangent - expected_tangent) <= 5 * MIN_STEP,
            f"tangent={tangent:.6f} expected={expected_tangent:.6f} "
            f"tol={5 * MIN_STEP:.6f}",
        )


def test_head_on():
    gap = 2 * MIN_STEP
    for theta in (0.0, math.pi / 6, math.pi / 4, 1.0):
        wall = _axis_wall(theta)
        g = make_game(0.5, 0.5, 0.0, [wall])
        sx, sy = _place_flush_left(g, wall, along=0.5, gap=gap)
        g.settings.agent_x, g.settings.agent_y = sx, sy
        vx, vy = from_wall_frame(g, wall, 1.0, 0.0)  # inward normal
        g.settings.direction = facing_from_vec(vx, vy)
        start = (g.settings.agent_x, g.settings.agent_y)
        g.stepForward()
        end = (g.settings.agent_x, g.settings.agent_y)
        start_w = to_wall_frame(g, wall, *start)
        end_w = to_wall_frame(g, wall, *end)
        total = math.hypot(end[0] - start[0], end[1] - start[1])
        tangential = abs(end_w[1] - start_w[1])
        tag = f"head-on theta={theta:.3f}"
        check(
            f"{tag} total displacement",
            total <= gap + MIN_STEP,
            f"total={total:.6f} cap={gap + MIN_STEP:.6f}",
        )
        check(
            f"{tag} tangential displacement",
            tangential <= 2 * MIN_STEP,
            f"tangential={tangential:.6f}",
        )


def test_right_angle_hold():
    # Inner L-corner at (0.4625, 0.4625): walls occupy -x and -y of that point.
    wa = [0.4, 0.4, WALL_W, 0.4, 0.0]
    wb = [0.4, 0.4, 0.4, WALL_W, 0.0]
    gap = 2 * MIN_STEP
    ax = 0.4 + WALL_W + AGENT_R + gap
    ay = 0.4 + WALL_W + AGENT_R + gap
    g = make_game(ax, ay, facing_from_vec(-1.0, -1.0), [wa, wb])
    start = (g.settings.agent_x, g.settings.agent_y)
    g.stepForward()
    mid = (g.settings.agent_x, g.settings.agent_y)
    d1 = math.hypot(mid[0] - start[0], mid[1] - start[1])
    slack = gap * math.sqrt(2) + MIN_STEP
    check(
        "right-angle first step slack",
        d1 <= slack + 2 * MIN_STEP,
        f"d1={d1:.6f} slack={slack:.6f}",
    )
    g.stepForward()
    end = (g.settings.agent_x, g.settings.agent_y)
    d2 = math.hypot(end[0] - mid[0], end[1] - mid[1])
    check(
        "right-angle second step hold",
        d2 <= 2 * MIN_STEP,
        f"d2={d2:.6f}",
    )
    check(
        "right-angle still free",
        g.full_wall_check(end[0], end[1]),
    )


def test_acute_hold():
    # 60-degree empty funnel: thin A along +x from the vertex, thin B at
    # -60 deg. Start flush on A's empty face, sliding toward the vertex;
    # first step reaches the standoff, second must hold (old-vs-new tangent).
    wa = [0.5, 0.5, 0.4, 0.02, 0.0]
    wb = [0.5, 0.5, 0.4, 0.02, -math.pi / 3]
    tilt = math.radians(15)
    vx, vy = -math.cos(tilt), math.sin(tilt)
    x, y = 0.67, 0.5 - AGENT_R - 2 * MIN_STEP
    g = make_game(x, y, facing_from_vec(vx, vy), [wa, wb])
    check("acute start free", g.full_wall_check(x, y))
    start = (g.settings.agent_x, g.settings.agent_y)
    g.stepForward()
    mid = (g.settings.agent_x, g.settings.agent_y)
    d1 = math.hypot(mid[0] - start[0], mid[1] - start[1])
    check(
        "acute first step reached junction (moved)",
        d1 > MIN_STEP,
        f"d1={d1:.6f} mid=({mid[0]:.4f},{mid[1]:.4f})",
    )
    g.stepForward()
    end = (g.settings.agent_x, g.settings.agent_y)
    d2 = math.hypot(end[0] - mid[0], end[1] - mid[1])
    check(
        "acute second step hold",
        d2 <= 3 * MIN_STEP,
        f"d2={d2:.6f} end=({end[0]:.4f},{end[1]:.4f})",
    )
    check("acute still free", g.full_wall_check(end[0], end[1]))


def test_obtuse_redirect():
    # Free region left of B (y ~ 0.36) sliding +y along B into A at 135°.
    wa = [0.25, 0.45, 0.5, WALL_W, 0.0]
    wb = [0.25, 0.45, 0.5, WALL_W, -math.pi / 4]
    g = make_game(0.18, 0.36, 0.0, [wa, wb])  # facing +y
    check("obtuse start free", g.full_wall_check(0.18, 0.36))
    g.stepForward(lim=0.125)
    end = (g.settings.agent_x, g.settings.agent_y)
    end_b = to_wall_frame(g, wb, *end)
    b_left, b_top, b_right, b_bot = wall_limits(g, wb)
    along_b = max(abs(end_b[0] - b_left), abs(end_b[1] - b_top))
    check(
        "obtuse still free",
        g.full_wall_check(end[0], end[1]),
        f"pos=({end[0]:.4f},{end[1]:.4f})",
    )
    check(
        "obtuse past junction along B",
        along_b >= 3 * MIN_STEP,
        f"along_b={along_b:.6f} end_b=({end_b[0]:.4f},{end_b[1]:.4f}) "
        f"B_ul=({b_left:.4f},{b_top:.4f})",
    )


def test_corner_slip_and_resume():
    tilt = math.radians(20)
    long_lim = 0.125
    # Face contact a short way from the TOP end so a 20-degree-into-wall
    # heading hits the face first, then the corner, then should resume.
    end_offset = 0.03
    for theta in (0.0, math.pi / 6):
        wall = _axis_wall(theta, ul=(0.4, 0.3), w=WALL_W, h=0.3)
        g = make_game(0.5, 0.5, 0.0, [wall])
        left, top, right, bot = wall_limits(g, wall)
        wx = left - AGENT_R - 2 * MIN_STEP
        wy = top + end_offset
        sx, sy = from_wall_frame(g, wall, wx, wy)
        # Into wall (+x) and toward the TOP end (-y).
        vwx, vwy = math.sin(tilt), -math.cos(tilt)
        vx, vy = from_wall_frame(g, wall, vwx, vwy)
        direction = facing_from_vec(vx, vy)
        v0 = (math.sin(direction), math.cos(direction))

        def _run(lim):
            gg = make_game(sx, sy, direction, [wall])
            gg.stepForward(lim=lim)
            return gg.settings.agent_x, gg.settings.agent_y, gg

        long_x, long_y, long_g = _run(long_lim)
        tag = f"slip-resume theta={theta:.3f}"
        check(
            f"{tag} still free",
            long_g.full_wall_check(long_x, long_y),
        )
        clearance = dist_to_rect(long_g, wall, long_x, long_y)
        check(
            f"{tag} cleared the wall",
            clearance > AGENT_R + MIN_STEP,
            f"dist={clearance:.6f} need>{AGENT_R + MIN_STEP:.6f}",
        )
        disp = (long_x - sx, long_y - sy)
        along_v0 = disp[0] * v0[0] + disp[1] * v0[1]
        check(
            f"{tag} kept 0.75 of budget on original heading",
            along_v0 >= 0.75 * long_lim,
            f"along_v0={along_v0:.6f} need>={0.75 * long_lim:.6f}",
        )
        # Short run: stop just past the corner (end_offset plus a corner-radius).
        short_lim = end_offset + AGENT_R + 8 * MIN_STEP
        short_x, short_y, _ = _run(short_lim)
        extra = (long_x - short_x, long_y - short_y)
        extra_mag = math.hypot(*extra)
        if extra_mag < 4 * MIN_STEP:
            check(
                f"{tag} last-leg extra displacement",
                False,
                f"extra mag {extra_mag:.6f} too small to judge heading",
            )
        else:
            extra_u = (extra[0] / extra_mag, extra[1] / extra_mag)
            dot = extra_u[0] * v0[0] + extra_u[1] * v0[1]
            angle = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
            check(
                f"{tag} last leg along original heading",
                angle <= 10.0,
                f"angle={angle:.2f} deg",
            )


def test_open_space():
    g = make_game(0.5, 0.5, 0.0, walls=[])
    g.stepForward()
    dx = g.settings.agent_x - 0.5
    dy = g.settings.agent_y - 0.5
    # direction 0 -> facing (0, 1): +y
    check(
        "open-space x unchanged",
        abs(dx) <= MIN_STEP,
        f"dx={dx:.6f}",
    )
    check(
        "open-space full step along facing",
        abs(dy - DEFAULT_LIM) <= MIN_STEP,
        f"dy={dy:.6f} expected={DEFAULT_LIM:.6f}",
    )


def main() -> int:
    test_diagonal_grazing()
    test_head_on()
    test_right_angle_hold()
    test_acute_hold()
    test_obtuse_redirect()
    test_corner_slip_and_resume()
    test_open_space()
    print()
    print(f"{passes} passed, {len(failures)} failed")
    if failures:
        for f in failures:
            print(" ", f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
