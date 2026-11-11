"""Stored gaze: the coordinate codes a reply was written with, replayed on recall.

While S2 writes a reply, every token it is fed carries a coordinate
code: the Localizer's ``(s, v)`` read one token earlier (the first reply
token carries the read at the last prompt token). Once the reply is text
in NAMS that is gone, and a later turn that recalls the message sees the
words but not where the eyes were.

This module keeps the reads with the message and puts them back.

* :func:`reply_trace` turns one generated reply into a small JSON-able
  trace: the prefill read, one read per generated token, and each
  token's character span in the stored text. ``S2GameSession`` stores it
  in the message's NAMS metadata (``metadata["gaze"]``).
* :func:`plan_prompt_codes` takes the messages a turn recalled and the
  encoded prompt, finds each recalled reply inside the
  Recent-conversation block, and says which prompt tokens carry which
  ``(s, v)``. ``prefill_step`` arms the ``CoordEmbedder`` with them for
  the prompt forward, the same way a reply token is armed in decode.

Convention, mode ``full``:

* a recalled reply token carries the code it carried when it was
  generated (the read one token earlier; the prefill read on the first);
* the token right after the reply carries the read at the reply's last
  content token: where the eyes were left. That is the slot the
  terminator had. In the block it is the newline before the next
  message, or the first token of whatever follows the block.

Tokens are matched by character span, so a different split inside the
block (``" Yes"`` for ``"Yes"``) does not matter. A message with no
trace, or one whose text no longer matches its trace, carries nothing.

Modes: ``full`` (every recalled reply), ``anchor`` (only the slot after
the most recent reply), ``off``. The default is ``full``; the
``S2_GAZE_REPLAY`` environment variable or the session argument
overrides it.
"""

from __future__ import annotations

import bisect
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

TRACE_VERSION = 1
MODES = ("off", "anchor", "full")
ENV_MODE = "S2_GAZE_REPLAY"
DEFAULT_MODE = "full"
# The CoordEmbedder's admissible input range. A stored read is a value
# the game already fed back, so it is inside; the clamp is a guard.
_COORD_LOW = -1.0
_COORD_HIGH = 2.0
_DECIMALS = 5


def resolve_mode(mode: str | None = None) -> str:
    """``mode``, else ``$S2_GAZE_REPLAY``, else ``full``. Raises on a typo."""
    chosen = mode if mode is not None else (os.environ.get(ENV_MODE) or DEFAULT_MODE)
    if chosen not in MODES:
        raise ValueError(f"gaze replay mode {chosen!r} is not one of {MODES}")
    return chosen


@dataclass
class GazeReplay:
    """What one play turn recalled, and how to replay it.

    ``messages`` are the block's messages, oldest first, as
    ``agent.memory.load_played_messages`` returns them: dicts with
    ``role``, ``content`` and ``gaze`` (a trace or None).
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    mode: str = DEFAULT_MODE


def _four(values: Any) -> list[float] | None:
    try:
        row = [float(item) for item in values]
    except (TypeError, ValueError):
        return None
    if len(row) != 4 or not all(math.isfinite(item) for item in row):
        return None
    return row


def reply_trace(
    tokenizer: Any,
    token_ids: list[int],
    reads: list[list[float]],
    prefill: list[float],
    text: str,
) -> dict[str, Any] | None:
    """The trace of one generated reply, or None when it cannot be aligned.

    ``token_ids`` are the generated ids, terminator included. ``reads[i]``
    is ``(sx, sy, vx, vy)`` read at generated token ``i``; ``prefill`` is
    the read at the last prompt token. ``text`` is the reply as it will
    be stored (special tokens dropped, stripped). ``spans[i]`` is token
    ``i``'s character span in ``text``; a token that adds no character
    (the terminator, stripped whitespace) has an empty span.
    """
    n = len(token_ids)
    if n == 0 or len(reads) != n or not text:
        return None
    first = _four(prefill)
    rows = [_four(row) for row in reads]
    if first is None or any(row is None for row in rows):
        return None
    pieces = tokenizer.batch_decode(
        [[int(tid)] for tid in token_ids], skip_special_tokens=True,
    )
    raw = "".join(pieces)
    if raw.strip() != text:
        # Per-token pieces do not rebuild the stored text (a character
        # split across tokens, a tokenizer that cleans up spaces). No
        # trace is better than a shifted one.
        return None
    lead = len(raw) - len(raw.lstrip())
    limit = len(text)
    spans: list[list[int]] = []
    cursor = 0
    for piece in pieces:
        start = min(max(cursor - lead, 0), limit)
        cursor += len(piece)
        end = min(max(cursor - lead, 0), limit)
        spans.append([start, end])
    return {
        "v": TRACE_VERSION,
        "chars": limit,
        "prefill": [round(item, _DECIMALS) for item in first],
        "spans": spans,
        "reads": [[round(item, _DECIMALS) for item in row] for row in rows],
    }


def valid_trace(trace: Any, content: str) -> bool:
    """True when ``trace`` is a version-1 trace of exactly ``content``."""
    if not isinstance(trace, dict) or trace.get("v") != TRACE_VERSION:
        return False
    if trace.get("chars") != len(content) or not content:
        return False
    spans = trace.get("spans")
    reads = trace.get("reads")
    if not isinstance(spans, list) or not isinstance(reads, list):
        return False
    if not spans or len(spans) != len(reads):
        return False
    if _four(trace.get("prefill")) is None:
        return False
    for span, row in zip(spans, reads):
        if _four(row) is None:
            return False
        if (
            not isinstance(span, (list, tuple)) or len(span) != 2
            or not 0 <= int(span[0]) <= int(span[1]) <= len(content)
        ):
            return False
    return any(int(span[1]) > int(span[0]) for span in spans)


def carried_codes(trace: dict[str, Any]) -> list[list[float]]:
    """The code each generated token carried: the read one token earlier."""
    reads = [_four(row) for row in trace["reads"]]
    return [_four(trace["prefill"])] + reads[:-1]  # type: ignore[list-item]


def final_read(trace: dict[str, Any]) -> list[float]:
    """The read at the last content token: where the reply left the eyes."""
    last = 0
    for index, span in enumerate(trace["spans"]):
        if int(span[1]) > int(span[0]):
            last = index
    return _four(trace["reads"][last])  # type: ignore[return-value]


def _clamp(row: list[float]) -> list[float]:
    return [min(max(item, _COORD_LOW), _COORD_HIGH) for item in row]


def plan_prompt_codes(
    tokenizer: Any,
    input_ids: list[int],
    replay: GazeReplay | None,
) -> tuple[list[int], list[list[float]]]:
    """Which prompt tokens carry which ``(sx, sy, vx, vy)``.

    Returns ``(positions, coords)``, positions ascending, both empty when
    nothing is replayed. ``input_ids`` is one encoded prompt. The block
    is found by its header and each message by its ``[role] text`` line,
    in order; a message that is not found verbatim is skipped.
    """
    from agent.memory import RECENT_CONVERSATION_HEADER

    if replay is None or replay.mode == "off" or not replay.messages:
        return [], []
    if replay.mode not in MODES:
        raise ValueError(f"gaze replay mode {replay.mode!r} is not one of {MODES}")
    assistants = [
        index for index, item in enumerate(replay.messages)
        if item.get("role") == "assistant"
    ]
    if not assistants:
        return [], []
    wanted = set(assistants) if replay.mode == "full" else {assistants[-1]}
    wanted = {
        index for index in wanted
        if valid_trace(replay.messages[index].get("gaze"),
                       str(replay.messages[index].get("content") or ""))
    }
    if not wanted:
        return [], []
    pieces = tokenizer.batch_decode(
        [[int(tid)] for tid in input_ids], skip_special_tokens=False,
    )
    starts: list[int] = []
    total = 0
    for piece in pieces:
        starts.append(total)
        total += len(piece)
    text = "".join(pieces)
    cursor = text.find(RECENT_CONVERSATION_HEADER)
    if cursor < 0:
        logger.warning("gaze replay: the prompt has no Recent-conversation block")
        return [], []
    cursor += len(RECENT_CONVERSATION_HEADER)
    by_position: dict[int, list[float]] = {}
    for index, item in enumerate(replay.messages):
        content = str(item.get("content") or "")
        tag = f"[{item.get('role')}] "
        at = text.find("\n" + tag + content, cursor)
        if at < 0:
            if index in wanted:
                logger.warning(
                    "gaze replay: recalled reply %d is not in the prompt verbatim",
                    index,
                )
            continue
        begin = at + 1 + len(tag)
        end = begin + len(content)
        cursor = end
        if index not in wanted:
            continue
        trace = item["gaze"]
        # First token that overlaps the reply, first token after it.
        first = max(bisect.bisect_right(starts, begin) - 1, 0)
        after = bisect.bisect_left(starts, end)
        if after >= len(pieces):
            continue
        if replay.mode == "full":
            codes = carried_codes(trace)
            spans = trace["spans"]
            for position in range(first, after):
                lo = starts[position] - begin
                hi = lo + len(pieces[position])
                best = -1
                best_overlap = 0
                for token, span in enumerate(spans):
                    overlap = min(hi, int(span[1])) - max(lo, int(span[0]))
                    if overlap > best_overlap:
                        best, best_overlap = token, overlap
                if best >= 0:
                    by_position[position] = _clamp(codes[best])
        by_position[after] = _clamp(final_read(trace))
    positions = sorted(by_position)
    return positions, [by_position[position] for position in positions]
