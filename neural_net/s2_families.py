"""The weekend trainer's task families, in one list.

Everything that decides *what* the joint S2 run trains on lives here:

* :data:`FAMILIES` is the list of families. Edit a ``weight`` to change a
  family's share of the training steps (weights are renormalised, so
  they need not sum to one); set it to ``0.0`` to exclude the family
  from training. A family's eval still runs when its weight is zero, so
  a dropped skill stays visible in ``eval/step_*/summary.json``.
  ``--mix A=0.3,G=0`` on the command line overrides these defaults for
  one run.
* The ``SHIFT_*`` tables below are family G's own knobs: the size of
  "a little", the size of a bare "look left", the share of each
  sub-case, which named places anchor the shift, and the wording.

The builders and the eval functions themselves live in
:mod:`neural_net.s2_weekend_eval`; this module only wires them up.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from neural_net.s2_weekend_eval import (
    Batch,
    _eval_begin,
    _eval_gaze,
    _eval_gaze_game,
    _eval_look,
    _eval_retarget,
    _eval_shift,
    build_begin,
    build_gaze,
    build_gaze_game,
    build_look_move,
    build_retarget,
    build_shift,
)

Builder = Callable[[random.Random, Path, int, Callable[[], dict[str, Any]]], Batch]
Evaluator = Callable[[Any, random.Random, Path], list[dict[str, Any]]]


@dataclass(frozen=True)
class Family:
    """One task family of the weekend run.

    ``key``
        The letter used in logs, ``--mix``, and the per-family loss parts.
    ``stage``
        The VRAM / heartbeat stage name.
    ``sample``
        The ``family`` label of eval samples and the eval summary group.
    ``weight``
        Default share of training steps; ``0.0`` excludes the family.
    ``inject``
        What the reply tokens carry as the coordinate code during
        training. ``"oracle"``: the true ``(s, v)`` with jitter.
        ``"feedback"``: the localizer's own reads one token behind, as
        the game injects them (recorded with the model in eval mode).
        Feedback is the honest choice whenever the label could otherwise
        be copied from the code.
    ``prompt_readout``
        Also supervise the localizer at the last prompt token, where the
        game takes its first read with no code on it. The label there is
        the first reply token's ``v``.
    ``build``
        ``(rng, tmp, step, place_image) -> Batch``; ``None`` for the
        anchor family, which trains on text examples instead of boards.
    ``eval``
        ``(trainer, rng, tmp) -> samples``; ``None`` when the family has
        no board eval (the anchor holdout CE is computed separately).
    """

    key: str
    stage: str
    sample: str
    weight: float
    inject: str
    prompt_readout: bool
    build: Builder | None
    eval: Evaluator | None


def _build_a(rng, tmp, step, place_image):
    return build_gaze(rng, place_image, mode="train")


def _build_b(rng, tmp, step, place_image):
    # One place, as in the eval: "Look at X." and the gaze question in one
    # user turn, a look line and the answer in one reply.
    return build_gaze_game(rng, tmp, step, n_turns=1, question=None)


def _build_c(rng, tmp, step, place_image):
    return build_look_move(rng, tmp, step, force_kind=None)


def _build_d(rng, tmp, step, place_image):
    return build_begin(rng, tmp, step, board=None, limit_one=False)


def _build_e(rng, tmp, step, place_image):
    return build_retarget(rng, tmp, step, sub=None, board=None)


def _build_g(rng, tmp, step, place_image):
    return build_shift(rng, tmp, step, case=None)


# ---------------------------------------------------------------- the list
# key, stage, sample label, weight, inject, prompt_readout, build, eval
FAMILIES: tuple[Family, ...] = (
    Family("A", "gaze", "gaze/oracle", 0.20, "oracle", False, _build_a, _eval_gaze),
    Family("B", "gaze_game", "gaze/game", 0.20, "feedback", True, _build_b, _eval_gaze_game),
    Family("C", "look_move", "look_move", 0.20, "oracle", False, _build_c, _eval_look),
    Family("D", "begin", "begin", 0.15, "oracle", False, _build_d, _eval_begin),
    Family("E", "retarget", "retarget", 0.15, "oracle", False, _build_e, _eval_retarget),
    Family("F", "anchor", "anchor", 0.10, "oracle", False, None, None),
    # G: "look a little to the left" = move the gaze by SHIFT_LITTLE from the
    # place it is on. Feedback injection, as the game plays it; the localizer
    # is also read at the last prompt token (the first reply token's gaze).
    Family("G", "shift", "shift", 0.12, "feedback", True, _build_g, _eval_shift),
)

# -------------------------------------------------------- family G: shifts
# The step one instruction moves the gaze, in picture units (the room is
# the unit square, a localizer cell is 1/16 = 0.0625).
SHIFT_LITTLE = 0.05       # "a little", "a bit", "slightly"
SHIFT_BARE = 0.05         # "look left" with no size word; same as little for now
# A shifted point must stay this far inside the picture edge. Exits sit on
# the edge (x or y = 0 / 1) and corners at 0.1 / 0.9, so with 0.03 a step
# from an exit lands in the wall band (0.95) and a step from a corner towards
# the wall is allowed; directions that would leave the picture are not drawn.
SHIFT_INNER = 0.03

# Sub-case shares within family G (renormalised). Each line is one prompt
# shape; ``v`` is the final gaze target, written next to the shape.
SHIFT_CASES: dict[str, float] = {
    # prior "Look at X." round in the Recent block; "Now, look a little
    # to the left." -> one look line, v = X + d
    "two_turn": 0.25,
    # "Look at X. Now, look a little to the left." in one user turn ->
    # two look lines; line 1 looks at X, line 2 at X + d
    "one_turn": 0.15,
    # two shifts in one user turn -> three look lines, X, X + d1, X + d1 + d2
    "stack_msg": 0.15,
    # prior look and a prior shift in the block; the new shift stacks on
    # the previous one: one look line, v = X + d1 + d2
    "stack_turns": 0.15,
    # "a little more / further to the left" after a prior left shift:
    # one more step the same way, v = X + 2 d
    "more": 0.15,
    # "Now, look left." (no size word): v = X + SHIFT_BARE * unit
    "bare": 0.15,
}

# Which named place the shift starts from (renormalised over the kinds the
# board has). Corners and exits are included on purpose: the skill is
# "relative to the current gaze", not "relative to the gold".
SHIFT_TARGETS: dict[str, float] = {"gold": 0.50, "exit": 0.25, "corner": 0.25}

# Directions in picture coordinates (x right, y up).
SHIFT_DIRECTIONS: dict[str, tuple[float, float]] = {
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "up": (0.0, 1.0),
    "down": (0.0, -1.0),
}
# How each direction is said. Bare forms may drop "to the".
SHIFT_DIRECTION_WORDS: dict[str, tuple[str, ...]] = {
    "left": ("to the left", "left"),
    "right": ("to the right", "right"),
    "up": ("up",),
    "down": ("down",),
}
# Templates; ``{dir}`` is one of the direction words above. The first
# template of each kind is the canonical wording used in the tests.
SHIFT_TEMPLATES: dict[str, tuple[str, ...]] = {
    "little": (
        "Now, look a little {dir}.",
        "Look a little {dir}.",
        "Now look a bit {dir}.",
        "Shift your gaze slightly {dir}.",
    ),
    "more": (
        "Now, look a little more {dir}.",
        "A little further {dir}.",
        "Now, a bit more {dir}.",
    ),
    "bare": (
        "Now, look {dir}.",
        "Look {dir}.",
    ),
}


# --------------------------------------------------------------- lookups
def family(key: str) -> Family:
    for spec in FAMILIES:
        if spec.key == key:
            return spec
    raise KeyError(f"no family {key!r}; families are {family_keys()}")


def family_keys() -> tuple[str, ...]:
    return tuple(spec.key for spec in FAMILIES)


def stage_names() -> dict[str, str]:
    return {spec.key: spec.stage for spec in FAMILIES}


def default_mix() -> dict[str, float]:
    """Default weights from :data:`FAMILIES`, before renormalisation."""
    return {spec.key: float(spec.weight) for spec in FAMILIES}


def parse_mix(text: str | None) -> dict[str, float]:
    """``--mix`` override. A zero drops that family; the rest are renormalised."""
    mix = default_mix()
    if text:
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            key, sep, raw = part.partition("=")
            if not sep:
                raise SystemExit(f"--mix entry {part!r} is not KEY=WEIGHT")
            key = key.strip()
            if key not in mix:
                raise SystemExit(
                    f"--mix key {key!r} is not one of {','.join(family_keys())}"
                )
            try:
                mix[key] = float(raw)
            except ValueError as exc:
                raise SystemExit(f"--mix weight {raw!r} is not a float") from exc
    if any(value < 0.0 for value in mix.values()):
        raise SystemExit("--mix weights must be >= 0")
    kept = {key: mix[key] for key in family_keys() if mix[key] > 0.0}
    total = sum(kept.values())
    if total <= 0.0:
        raise SystemExit("--mix dropped every family")
    return {key: kept[key] / total for key in kept}


def build_train_family(
    key: str,
    rng: random.Random,
    tmp: Path,
    step: int,
    place_image: Callable[[], dict[str, Any]],
) -> Batch:
    """One training image for family ``key``. ``place_image`` is the
    coord-tf four-direction deal (family A)."""
    spec = family(key)
    if spec.build is None:
        raise ValueError(f"family {key!r} has no board builder")
    return spec.build(rng, tmp, step, place_image)


def mix_help() -> str:
    defaults = ",".join(f"{spec.key}={spec.weight:.2f}" for spec in FAMILIES)
    return (
        f"Default {defaults}. A zero weight drops that family and the rest "
        "are renormalized. The defaults live in neural_net/s2_families.py."
    )
