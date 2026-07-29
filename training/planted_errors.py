"""Planted-error generation: corrupt a correct player reply, keep the label.

Implements the committed techniques of TRAINING_TRACE_EXTRAS.md ("Planted-
error data"): take a player reply that is correct (as graded, or hand-
picked) and programmatically plant ONE known, labeled mistake in it. The
corruption itself is the ground truth -- no engine oracle, no human -- which
is what makes these usable as an analyst miss-rate probe (does the analyst
flag the planted error? does it stay quiet about unchanged spans? does it
respect the 2-hour grading tolerance?) and, if the analyst is ever trained,
as a training signal that cannot reward agreement.

Three corruption kinds, matching the reply shape observed in the real logs
("OBS: ... my eye points toward 12 o'clock; the gold is at the top right,
toward 1 o'clock of me. REASON: ... [CLOCK]"):

  * ``move_token_*``  -- the final bracketed move token is replaced with a
    different one of the three, stripped of its brackets (the known
    bare-word format error), or deleted outright;
  * ``clock_shift``   -- one ``N o'clock`` phrase gets a random new hour
    (1-12); the change records the circular shift and whether it is within
    the 2-hour grading tolerance (a within-tolerance shift must be labeled
    "acceptable", NOT "error");
  * ``direction_swap``-- one direction word is swapped with its opposite
    (left/right, up/down, upper/lower, top/bottom, above/below,
    clockwise/counter-clockwise), case-preserving; compounds like
    "top right" or "lower-left" fall out of the word-level swap.

Every corruption returns a machine-readable change log; callers that get an
unchanged reply back (no applicable target) should SKIP it, never guess.

NOT hooked up anywhere yet by design -- usage (the analyst-quality probe in
the early-warning suite, possible analyst training) is an explicitly
deferred decision; see TRAINING_TRACE_EXTRAS.md.
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import game_io

#: Change-log record: kind, char span IN THE CORRUPTED TEXT, the exact old
#: and new substrings, and whether the change is within grading tolerance
#: (only clock shifts of <= ``TOLERANCE_HOURS`` qualify; every other planted
#: error is a real mistake).
Change = dict

#: The grading calibration the analyst prompts promise: direction estimates
#: within 2 clock hours of the truth are NOT mistakes.
TOLERANCE_HOURS = 2

#: ``N o'clock`` / ``N:MM o'clock`` phrases, as seen in real replies
#: ("12 o'clock", "4:18 o'clock", "between 3 and 4 o'clock" -- the last
#: yields two separate single-hour matches).
CLOCK_RE = re.compile(r"\b(\d{1,2})(?::\d{2})?\s*o'?clock\b", re.IGNORECASE)

#: Direction words and their opposites. Longest alternatives first in the
#: regex so "counter-clockwise" wins over "clockwise" and "upper" over "up".
#: The bracketed move tokens ([CLOCK] etc.) are safe: none of these words
#: match their spelling.
_OPPOSITE = {
    "counter-clockwise": "clockwise",
    "counterclockwise": "clockwise",
    "anticlockwise": "clockwise",
    "clockwise": "counter-clockwise",
    "left": "right",
    "right": "left",
    "upper": "lower",
    "lower": "upper",
    "top": "bottom",
    "bottom": "top",
    "above": "below",
    "below": "above",
    "up": "down",
    "down": "up",
}
_DIRECTION_RE = re.compile(
    r"\b(" + "|".join(sorted(_OPPOSITE, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

#: Guard for 'right' meaning "correct", not the direction: skip the swap
#: when 'right' is directly followed by one of these nouns ("the right
#: move", "turning in the right direction") or sits in "all right" /
#: "that's right". A heuristic, deliberately narrow -- every swap lands in
#: the change log, so residual misfires stay auditable.
_RIGHT_NOUN_RE = re.compile(
    r"^\s+(move|moves|choice|choices|call|answer|thing|decision|direction|"
    r"way|track|spot|idea)\b",
    re.IGNORECASE,
)
_RIGHT_PRECEDER_RE = re.compile(r"(?:\ball|that'?s|\bis|\bwas|\bare)\s+$",
                                re.IGNORECASE)


def _match_case(replacement: str, original: str) -> str:
    """Shape ``replacement``'s capitalization like ``original``'s."""
    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def _semantic_right(text: str, m: re.Match) -> bool:
    """True when this 'right' match means "correct" (see _RIGHT_NOUN_RE)."""
    if m.group(0).lower() != "right":
        return False
    return bool(_RIGHT_NOUN_RE.match(text[m.end():])
                or _RIGHT_PRECEDER_RE.search(text[: m.start()]))


# ============================================================ the scramblers

def scramble_move_token(text: str, rng: random.Random) -> tuple[str, list[Change]]:
    """Corrupt the FINAL bracketed move token (the one that would be
    executed): replace it with a different token of the three, strip its
    brackets, or delete it. No token -> unchanged."""
    last = None
    for last in game_io._MOVE_RE.finditer(text):
        pass
    if last is None:
        return text, []
    old = last.group(0)
    action = last.group(1).upper()
    mode = rng.choice(("replace", "strip", "delete"))
    if mode == "replace":
        new = "[" + rng.choice([a for a in game_io.ACTIONS if a != action]) + "]"
    elif mode == "strip":
        new = action
    else:
        new = ""
    corrupted = text[: last.start()] + new + text[last.end():]
    return corrupted, [{
        "kind": f"move_token_{mode}",
        "span": [last.start(), last.start() + len(new)],
        "old": old,
        "new": new,
        "within_tolerance": False,
    }]


def scramble_clock(text: str, rng: random.Random) -> tuple[str, list[Change]]:
    """Replace ONE randomly chosen ``N o'clock`` phrase with a different
    random hour 1-12 (minutes, if any, are dropped -- the corrupted phrase
    is a clean whole-hour claim). Records the circular hour shift and
    whether it is within the grading tolerance. No valid phrase ->
    unchanged."""
    matches = [m for m in CLOCK_RE.finditer(text)
               if 1 <= int(m.group(1)) <= 12]
    if not matches:
        return text, []
    m = rng.choice(matches)
    old_hour = int(m.group(1))
    new_hour = rng.choice([h for h in range(1, 13) if h != old_hour])
    shift = min((new_hour - old_hour) % 12, (old_hour - new_hour) % 12)
    new = f"{new_hour} o'clock"
    corrupted = text[: m.start()] + new + text[m.end():]
    return corrupted, [{
        "kind": "clock_shift",
        "span": [m.start(), m.start() + len(new)],
        "old": m.group(0),
        "new": new,
        "shift_hours": shift,
        "within_tolerance": shift <= TOLERANCE_HOURS,
    }]


def scramble_directions(text: str, rng: random.Random) -> tuple[str, list[Change]]:
    """Swap ONE randomly chosen direction word with its opposite,
    case-preserving. 'right' meaning "correct" is skipped (see
    _RIGHT_NOUN_RE). No candidate -> unchanged."""
    candidates = [m for m in _DIRECTION_RE.finditer(text)
                  if not _semantic_right(text, m)]
    if not candidates:
        return text, []
    m = rng.choice(candidates)
    old = m.group(0)
    new = _match_case(_OPPOSITE[old.lower()], old)
    corrupted = text[: m.start()] + new + text[m.end():]
    return corrupted, [{
        "kind": "direction_swap",
        "span": [m.start(), m.start() + len(new)],
        "old": old,
        "new": new,
        "within_tolerance": False,
    }]


# ========================================================= full-reply entry

_SCRAMBLERS = {
    "move_token": scramble_move_token,
    "clock": scramble_clock,
    "direction": scramble_directions,
}

#: The corruption kinds ``scramble_player_reply`` draws from, by default.
KINDS = tuple(_SCRAMBLERS)


def scramble_player_reply(
    text: str,
    rng: random.Random,
    kinds: tuple[str, ...] | None = None,
) -> tuple[str, list[Change]]:
    """Plant exactly ONE labeled error in a player reply.

    Tries the corruption kinds in random order and applies the first one
    that finds a target -- a single localized error keeps the label clean
    (the analyst either catches THE mistake or misses it). Returns
    ``(corrupted_text, changes)``; ``(text, [])`` means no kind applied and
    the caller should skip this reply, never fabricate a label."""
    order = list(kinds or KINDS)
    unknown = [k for k in order if k not in _SCRAMBLERS]
    if unknown:
        raise ValueError(
            f"scramble_player_reply: unknown kind(s) {unknown} "
            f"(known: {sorted(_SCRAMBLERS)})"
        )
    rng.shuffle(order)
    for kind in order:
        corrupted, changes = _SCRAMBLERS[kind](text, rng)
        if changes:
            return corrupted, changes
    return text, []
