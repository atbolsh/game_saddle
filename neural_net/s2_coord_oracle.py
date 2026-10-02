"""Clock, direction, and yes/no scoring for the coord-embed phase.

No torch and no pygame. The trainer samples boards; this module only
decides whether a sample has an unambiguous label and what a reply is
worth. Rewards stay here. They are not written into the prompt.
"""

from __future__ import annotations

import math

GOLD_R = 1.0 / 64.0
AGENT_R = 0.05
REPLY_JITTER_SIGMA = GOLD_R / 5.0
S_SIGMA = 0.5 * AGENT_R / 3.0
S_RADIUS = 0.5 * AGENT_R
AXIS_MARGIN = 0.05
HOUR_BOUNDARY_DEG = 3.0

KIND_UPDOWN = "updown"
KIND_LEFTRIGHT = "leftright"
KIND_HOUR = "hour"
KIND_LOOKING = "looking"
KIND_MOVING = "moving"
KINDS = (
    KIND_UPDOWN, KIND_LEFTRIGHT, KIND_HOUR, KIND_LOOKING, KIND_MOVING,
)

LINE_UP = "Looking: up"
LINE_DOWN = "Looking: down"
LINE_LEFT = "Looking: left"
LINE_RIGHT = "Looking: right"


def gaze_radians(sx: float, sy: float, vx: float, vy: float) -> float:
    """Clockwise from 12 o'clock. 12 o'clock is +y."""
    return math.atan2(vx - sx, vy - sy)


def clock_hour(sx: float, sy: float, vx: float, vy: float) -> int:
    """Nearest hour, 1 through 12. Ties are the caller's job to avoid."""
    hours = gaze_radians(sx, sy, vx, vy) / (math.pi / 6.0)
    nearest = int(round(hours)) % 12
    return 12 if nearest == 0 else nearest


def degrees_from_hour_boundary(sx: float, sy: float, vx: float, vy: float) -> float:
    """Degrees from the nearest boundary between two hours.

    Hour centers are 30 degrees apart, so a boundary sits 15 degrees
    from each center. Zero means the gaze is exactly halfway.
    """
    deg = math.degrees(gaze_radians(sx, sy, vx, vy)) % 360.0
    into = deg % 30.0
    return abs(into - 15.0)


def hour_is_unique(sx: float, sy: float, vx: float, vy: float) -> bool:
    return degrees_from_hour_boundary(sx, sy, vx, vy) >= HOUR_BOUNDARY_DEG


def circular_hour_distance(a: int, b: int) -> int:
    delta = abs(int(a) - int(b)) % 12
    return min(delta, 12 - delta)


def hour_reward(true_hour: int, said_hour: int) -> float:
    """1, 1/2, 1/4, or 0. Three hours is 90 degrees and scores 0."""
    distance = circular_hour_distance(true_hour, said_hour)
    if distance == 0:
        return 1.0
    if distance == 1:
        return 0.5
    if distance == 2:
        return 0.25
    return 0.0


def gaze_vbar(
    sx: float, sy: float, vx: float, vy: float, *, looking: bool,
) -> tuple[float, float]:
    """Looking reflects the target through the agent. Moving uses the target."""
    if looking:
        return (2.0 * sx - vx, 2.0 * sy - vy)
    return (vx, vy)


def direction_label(kind: str, sx: float, sy: float, vx: float, vy: float) -> str | None:
    """``up`` / ``down`` / ``left`` / ``right``, or None when the axis is a tie."""
    if kind == KIND_UPDOWN:
        if abs(vy - sy) < AXIS_MARGIN:
            return None
        return "up" if vy > sy else "down"
    if kind == KIND_LEFTRIGHT:
        if abs(vx - sx) < AXIS_MARGIN:
            return None
        return "right" if vx > sx else "left"
    raise ValueError(f"not a direction kind: {kind!r}")


def correct_line(kind: str, expected: str | int) -> str:
    if kind == KIND_UPDOWN:
        if expected == "up":
            return LINE_UP
        if expected == "down":
            return LINE_DOWN
    elif kind == KIND_LEFTRIGHT:
        if expected == "left":
            return LINE_LEFT
        if expected == "right":
            return LINE_RIGHT
    elif kind == KIND_HOUR:
        hour = int(expected)
        if 1 <= hour <= 12:
            return f"Hour: {hour}"
    elif kind in (KIND_LOOKING, KIND_MOVING):
        if expected in ("yes", "no"):
            return f"Answer: {expected}"
    raise ValueError(f"no reply line for {kind!r} {expected!r}")


def parse_reply(kind: str, text: str) -> str | int | None:
    """The reply's label, or None when the text is not the one required line."""
    line = text.strip()
    if kind == KIND_UPDOWN:
        if line == LINE_UP:
            return "up"
        if line == LINE_DOWN:
            return "down"
        return None
    if kind == KIND_LEFTRIGHT:
        if line == LINE_LEFT:
            return "left"
        if line == LINE_RIGHT:
            return "right"
        return None
    if kind == KIND_HOUR:
        prefix = "Hour: "
        if not line.startswith(prefix):
            return None
        body = line[len(prefix):]
        if not body.isdigit():
            return None
        hour = int(body)
        if body != str(hour) or not 1 <= hour <= 12:
            return None
        return hour
    if kind in (KIND_LOOKING, KIND_MOVING):
        if line == "Answer: yes":
            return "yes"
        if line == "Answer: no":
            return "no"
        return None
    raise ValueError(f"unknown kind {kind!r}")


def reply_reward(kind: str, text: str, expected: str | int) -> float:
    got = parse_reply(kind, text)
    if got is None:
        return 0.0
    if kind == KIND_HOUR:
        return hour_reward(int(expected), int(got))
    return 1.0 if got == expected else 0.0
