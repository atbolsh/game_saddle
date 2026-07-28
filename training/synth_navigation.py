"""Synthetic navigation dataset: clock / compass / bearing arithmetic.

Generates the one dataset no public source covers: conversions and
shortest-rotation problems in EXACTLY this repo's conventions (see
agent/game_io.py and the geometry prompt blocks):

  * bearings are measured CLOCKWISE from 12 o'clock (north);
  * one clock hour = 30 degrees; larger theta = more clockwise;
  * radians: theta_rad = theta_deg * pi / 180, pi ~= 3.14159.

Problem families: clock hour -> degrees, degrees -> nearest hour,
degrees <-> radians, 8-point compass <-> bearing, signed bearing difference,
shortest rotation direction (between clock hours and between bearings), and
signed hour distance. These are pure math trivia with programmatic ground
truth -- no game engine involved, so the "no engine-derived baselines" rule
(TRAINING_GAME_TRACES.md) is untouched.

Every target ends with a final line ``ANSWER: <value>`` (the probes'
exact-match extraction keys on it); questions state the expected answer
format explicitly, so exact matching is fair.

Deterministic: everything derives from the caller's seed. Used by
training/download_external.py via external_data.GENERATORS; no network.
"""

from __future__ import annotations

import math
import random
from typing import Any, Iterator

#: 8-point compass, in repo convention (bearing clockwise from north).
COMPASS_POINTS = [
    ("N", 0), ("NE", 45), ("E", 90), ("SE", 135),
    ("S", 180), ("SW", 225), ("W", 270), ("NW", 315),
]

_CONVENTION = (
    "Use this convention: bearings are measured clockwise from 12 o'clock "
    "(north), one clock hour = 30 degrees."
)


def _wrap_signed(deg: float) -> float:
    """Wrap a difference into (-180, 180]."""
    d = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def _q(text: str, answer_format: str) -> str:
    return f"{text} {_CONVENTION} End your reply with 'ANSWER: {answer_format}'."


# Each maker: rng -> (question, solution_text, answer_str). Solutions are
# short worked derivations so CE training teaches the METHOD, not answer
# memorization.

def _hour_to_degrees(rng: random.Random):
    h = rng.randint(1, 12)
    deg = (h % 12) * 30
    q = _q(f"A gaze points toward {h} o'clock. What is its bearing in "
           "degrees?", "<integer>")
    sol = (f"Each clock hour is 30 degrees clockwise from north, so "
           f"{h} o'clock is {h % 12} x 30 = {deg} degrees.\nANSWER: {deg}")
    return q, sol, str(deg)


def _degrees_to_hour(rng: random.Random):
    h = rng.randint(1, 12)
    jitter = rng.randint(-14, 14)
    deg = ((h % 12) * 30 + jitter) % 360
    q = _q(f"A bearing is {deg} degrees. Which clock hour is it closest to "
           "(1-12)?", "<integer 1-12>")
    hour = round(deg / 30) % 12 or 12
    sol = (f"{deg} / 30 = {deg / 30:.2f}, which rounds to {round(deg / 30) % 12}"
           f" -- that is {hour} o'clock.\nANSWER: {hour}")
    return q, sol, str(hour)


def _degrees_to_radians(rng: random.Random):
    deg = rng.randint(0, 359)
    rad = round(deg * math.pi / 180.0, 3)
    q = _q(f"Convert a bearing of {deg} degrees to radians (round to 3 "
           "decimals, pi ~= 3.14159).", "<number with 3 decimals>")
    sol = (f"radians = {deg} x pi / 180 = {deg} x 0.0174533 ~= {rad:.3f}."
           f"\nANSWER: {rad:.3f}")
    return q, sol, f"{rad:.3f}"


def _radians_to_degrees(rng: random.Random):
    deg = rng.randint(0, 359)
    rad = round(deg * math.pi / 180.0, 3)
    q = _q(f"Convert a bearing of {rad:.3f} radians to degrees (round to "
           "the nearest integer, pi ~= 3.14159).", "<integer>")
    sol = (f"degrees = {rad:.3f} x 180 / pi ~= {rad:.3f} x 57.2958 ~= {deg}."
           f"\nANSWER: {deg}")
    return q, sol, str(deg)


def _compass_to_degrees(rng: random.Random):
    point, deg = rng.choice(COMPASS_POINTS)
    q = _q(f"What bearing in degrees is compass point {point}?", "<integer>")
    sol = (f"On the 8-point compass measured clockwise from north, {point} "
           f"is {deg} degrees.\nANSWER: {deg}")
    return q, sol, str(deg)


def _degrees_to_compass(rng: random.Random):
    point, base = rng.choice(COMPASS_POINTS)
    deg = (base + rng.randint(-20, 20)) % 360
    q = _q(f"A bearing is {deg} degrees. Which 8-point compass direction "
           "(N, NE, E, SE, S, SW, W, NW) is it closest to?", "<compass point>")
    sol = (f"The nearest multiple of 45 to {deg} is {base}, which is "
           f"{point}.\nANSWER: {point}")
    return q, sol, point


def _bearing_difference(rng: random.Random):
    a = rng.randint(0, 359)
    d = rng.choice([x for x in range(-179, 180) if x != 0])
    b = (a + d) % 360
    q = _q(f"You face bearing {a} degrees; the target is at bearing {b} "
           "degrees. What is the signed shortest rotation in degrees "
           "(positive = clockwise, negative = counter-clockwise)?",
           "<signed integer>")
    sol = (f"Raw difference: {b} - {a} = {b - a}; wrapped into (-180, 180] "
           f"that is {int(_wrap_signed(b - a))}.\nANSWER: {int(_wrap_signed(b - a))}")
    return q, sol, str(int(_wrap_signed(b - a)))


def _rotation_direction_hours(rng: random.Random):
    h1 = rng.randint(1, 12)
    # skip 0 (no turn) and 6 (ambiguous 180-degree turn)
    d = rng.choice([x for x in range(-5, 6) if x != 0])
    h2 = (h1 + d - 1) % 12 + 1
    ans = "clockwise" if d > 0 else "counter-clockwise"
    q = _q(f"You face {h1} o'clock and want to face {h2} o'clock. Is the "
           "shortest rotation clockwise or counter-clockwise?",
           "<clockwise|counter-clockwise>")
    sol = (f"From {h1} to {h2} o'clock is {d:+d} hours the short way "
           f"({abs(d)} x 30 = {abs(d) * 30} degrees); "
           f"{'positive means clockwise' if d > 0 else 'negative means counter-clockwise'}."
           f"\nANSWER: {ans}")
    return q, sol, ans


def _rotation_direction_degrees(rng: random.Random):
    a = rng.randint(0, 359)
    # avoid |diff| near 0 (trivial) and near 180 (ambiguous)
    d = rng.choice([x for x in range(-170, 171) if abs(x) >= 10])
    b = (a + d) % 360
    ans = "clockwise" if d > 0 else "counter-clockwise"
    q = _q(f"You face bearing {a} degrees; the gold is at bearing {b} "
           "degrees. Should you rotate clockwise or counter-clockwise "
           "(shortest way)?", "<clockwise|counter-clockwise>")
    sol = (f"Wrapped difference: {b} - {a} wrapped into (-180, 180] is "
           f"{int(_wrap_signed(b - a))} degrees; "
           f"{'positive = clockwise' if d > 0 else 'negative = counter-clockwise'}."
           f"\nANSWER: {ans}")
    return q, sol, ans


def _hours_between(rng: random.Random):
    h1 = rng.randint(1, 12)
    d = rng.choice([x for x in range(-5, 6) if x != 0])
    h2 = (h1 + d - 1) % 12 + 1
    q = _q(f"How many clock hours is the shortest rotation from facing "
           f"{h1} o'clock to facing {h2} o'clock (signed: positive = "
           "clockwise)?", "<signed integer>")
    sol = (f"({h2} - {h1}) mod 12 = {(h2 - h1) % 12}; taking the short way "
           f"gives {d:+d} hours.\nANSWER: {d:+d}")
    return q, sol, f"{d:+d}"


_MAKERS = [
    _hour_to_degrees,
    _degrees_to_hour,
    _degrees_to_radians,
    _radians_to_degrees,
    _compass_to_degrees,
    _degrees_to_compass,
    _bearing_difference,
    _rotation_direction_hours,
    _rotation_direction_degrees,
    _hours_between,
]


def _make(rng: random.Random) -> tuple[str, str, str, str]:
    maker = rng.choice(_MAKERS)
    q, sol, ans = maker(rng)
    return q, sol, ans, maker.__name__.lstrip("_")


def generate_records(n: int, seed: int) -> Iterator[dict[str, Any]]:
    """Training records in the canonical external-data shape (no 'loss' --
    the materializer stamps the manifest entry's loss on every line)."""
    rng = random.Random(seed)
    for _ in range(n):
        q, sol, _ans, kind = _make(rng)
        yield {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": q}]}
            ],
            "target_text": sol,
            "meta": {"kind": kind},
        }


def generate_probe(n: int, seed: int) -> Iterator[dict[str, Any]]:
    """Probe records: prompt + gold final answer for exact-match scoring.
    Uses seed+1 internally so probe items never collide with training items
    generated from the same manifest seed."""
    rng = random.Random(seed + 1)
    for _ in range(n):
        q, _sol, ans, kind = _make(rng)
        yield {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": q}]}
            ],
            "answer": ans,
            "meta": {"kind": kind},
        }
