"""Failure analyst: one thread over every flagged memory.

The system prompt is loaded from NAMS (``core_fa_`` tips). The loaded
set, the scratchpad, and the analyst conversation persist until
:meth:`FailureAnalyst.restart`. Pictures are capped. Gaze is the stored
trace of the BEFORE message, printed at three decimals.

The old mode-1 grader lives in :mod:`agent.modes_legacy_analyst`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import memory as mem
from .config import CONFIG, AgentConfig
from .game_io import fmt3, settings_json_with_openings
from . import run_logging
from neural_net.gaze_replay import valid_trace

logger = logging.getLogger(__name__)

PARTS = ("bq", "bm", "bi", "bs", "aq", "am", "ai", "as", "gaze")
DEFAULT_PARTS = ("bq", "bm", "bi", "bs", "aq", "am")
PART_ORDER = ("bq", "bm", "bi", "bs", "aq", "am", "ai", "as", "gaze")

# The continuation the tool prompt promises: the result is shown, then
# the model is asked to continue. No tool token ends the turn.
ANALYST_CONTINUE = "Continue."

# Same placeholder shape as the other missing BEFORE parts.
NO_PRIOR_GAZE = "[ERROR, no prior gaze]"

_PART_LABEL = {
    "bq": "before question",
    "bm": "before message",
    "bs": "before Settings",
    "aq": "after question",
    "am": "after message",
    "as": "after Settings",
}

ANALYST_TOOL_PATTERN = (
    r"(?i)\[\s*(?:"
    r"LOAD\s+(?P<load_n>\d+)\s*(?::\s*(?P<load_parts>[a-z][a-z ]*))?"
    r"|DROP\s+(?P<drop_all>ALL)"
    r"|DROP\s+(?P<drop_n>\d+)\s*(?::\s*(?P<drop_parts>[a-z][a-z ]*))?"
    r"|GAZE\s+(?P<gaze_n>\d+)\s*(?::\s*(?P<gaze_spec>-\d+|\d+\s*-\s*\d+))?"
    r"|NOTE\s+(?P<note_key>[^:\[\]\n]+?)\s*:\s*(?P<note_text>[^\[\]]*?)"
    r"|FORGET\s+(?P<forget_key>[^\[\]\n]+?)"
    r"|SEARCH\b[:\s]*(?P<search_query>[^\[\]]+?)"
    r"|LIST"
    r")\s*\]"
)
ANALYST_TOOL_RE = re.compile(ANALYST_TOOL_PATTERN)


def parse_analyst_call(text: str) -> tuple[dict[str, Any] | None, str]:
    """``(call, text)`` for the first analyst tool token, text cut after it.

    ``call`` is None when the reply has no complete token. Parts are a
    list of part names, or None when the token names none (``[LOAD n]``
    means the default set; ``[DROP n]`` means the whole memory).
    """
    match = ANALYST_TOOL_RE.search(text)
    if not match:
        return None, text
    truncated = text[: match.end()]
    if match.group("load_n") is not None:
        return {
            "tool": "LOAD",
            "n": int(match.group("load_n")),
            "parts": _part_list(match.group("load_parts")),
        }, truncated
    if match.group("drop_all") is not None:
        return {"tool": "DROP", "n": None, "parts": None}, truncated
    if match.group("drop_n") is not None:
        return {
            "tool": "DROP",
            "n": int(match.group("drop_n")),
            "parts": _part_list(match.group("drop_parts")),
        }, truncated
    if match.group("gaze_n") is not None:
        return {
            "tool": "GAZE",
            "n": int(match.group("gaze_n")),
            "window": _gaze_spec(match.group("gaze_spec")),
        }, truncated
    if match.group("note_key") is not None:
        return {
            "tool": "NOTE",
            "key": match.group("note_key").strip(),
            "text": (match.group("note_text") or "").strip(),
        }, truncated
    if match.group("forget_key") is not None:
        return {
            "tool": "FORGET",
            "key": match.group("forget_key").strip(),
        }, truncated
    if match.group("search_query") is not None:
        return {
            "tool": "SEARCH",
            "query": match.group("search_query").strip(),
        }, truncated
    return {"tool": "LIST"}, truncated


def _part_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [part.lower() for part in raw.split() if part]


def _gaze_spec(raw: str | None) -> tuple[str, int, int | None] | None:
    if raw is None:
        return None
    text = raw.strip()
    if text.startswith("-"):
        return ("tail", int(text[1:]), None)
    left, right = re.split(r"\s*-\s*", text, maxsplit=1)
    return ("range", int(left), int(right))


def image_names(loaded: dict[int, set[str]]) -> list[tuple[int, str]]:
    """``(memory, "before"|"after")`` for every loaded picture, in order."""
    names: list[tuple[int, str]] = []
    for n in sorted(loaded):
        parts = loaded[n]
        if "bi" in parts:
            names.append((n, "before"))
        if "ai" in parts:
            names.append((n, "after"))
    return names


def image_cap_refusal(
    loaded: dict[int, set[str]],
    n: int,
    parts: set[str],
    cap: int = 4,
) -> str | None:
    """The refusal sentence, or None when the load stays within ``cap``.

    The sentence lists the pictures already loaded, not the one refused.
    """
    trial = {key: set(value) for key, value in loaded.items()}
    trial.setdefault(n, set()).update(parts)
    if len(image_names(trial)) <= cap:
        return None
    current = image_names(loaded)
    listing = ", ".join(f"{number} {side}" for number, side in current)
    return f"{len(current)} images are loaded: {listing}. Drop one first."


def index_line(n_flags: int, loaded: list[int]) -> str:
    """The index line. ``loaded`` is the memory numbers, oldest-number order."""
    nums = ", ".join(str(n) for n in loaded)
    if nums:
        tail = f"Loaded now: {nums}."
    else:
        tail = "Loaded now:."
    return (
        f"{n_flags} flagged memories, numbered 0..{n_flags - 1} "
        f"oldest first. {tail}"
    )


def memory_header(n: int, session_id: str, timestamp: Any) -> str:
    return f"Memory {n} (game {str(session_id)[:8]}, {_stamp(timestamp)})"


def _stamp(value: Any) -> str:
    if value is None:
        return ""
    iso = getattr(value, "iso_format", None)
    if callable(iso):
        return str(iso())
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        seconds = float(value)
        if seconds > 10**11:
            seconds /= 1000.0
        return datetime.fromtimestamp(
            seconds, tz=timezone.utc,
        ).strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def _read4(row: Any) -> list[float]:
    if not isinstance(row, (list, tuple)) or len(row) != 4:
        raise RuntimeError(f"gaze read is not four floats: {row!r}")
    return [float(item) for item in row]


def _summary_sv(read: list[float]) -> str:
    sx, sy, vx, vy = (fmt3(item) for item in read)
    return f"s=({sx}, {sy})  v=({vx}, {vy})"


def _summary_row(label: str, read: list[float]) -> str:
    return f"{label:<22} {_summary_sv(read)}"


def _lines_of(text: str) -> list[tuple[int, int, str]]:
    """``(start, end_exclusive, line)`` for each line, newline excluded."""
    rows: list[tuple[int, int, str]] = []
    pos = 0
    parts = text.split("\n")
    for index, line in enumerate(parts):
        rows.append((pos, pos + len(line), line))
        pos += len(line)
        if index != len(parts) - 1:
            pos += 1
    return rows


def _piece(text: str, span: Any) -> str:
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        raise RuntimeError(f"gaze span is not a pair: {span!r}")
    start, end = int(span[0]), int(span[1])
    if not 0 <= start <= end <= len(text):
        raise RuntimeError(
            f"gaze span {start}:{end} outside a message of {len(text)}"
        )
    return text[start:end]


def _final_bracket(text: str, spans: list[Any]) -> int | None:
    """Index of the last token whose piece ends with ``]``.

    None when no piece does. The caller then uses the last content token.
    """
    found = None
    for index, span in enumerate(spans):
        if _piece(text, span).endswith("]"):
            found = index
    return found


def _last_content(text: str, spans: list[Any]) -> int:
    found = None
    for index, span in enumerate(spans):
        piece = _piece(text, span)
        if piece:
            found = index
    if found is None:
        raise RuntimeError("gaze trace has no content token")
    return found


def format_gaze_summary(n: int, text: str, trace: dict[str, Any]) -> str:
    """The five-or-six-line gaze summary of one BEFORE message."""
    if not valid_trace(trace, text):
        raise RuntimeError(f"memory {n} BEFORE gaze does not match its text")
    reads = [_read4(row) for row in trace["reads"]]
    spans = trace["spans"]
    lines = [
        f"gaze, memory {n}, BEFORE message ({len(reads)} tokens):",
        _summary_row("  prefill", _read4(trace["prefill"])),
    ]
    for number, (start, end, line) in enumerate(_lines_of(text), start=1):
        chosen = None
        for index, span in enumerate(spans):
            stop = int(span[1])
            if start < stop <= end:
                chosen = index
        if chosen is None:
            continue
        tail = line[-12:]
        label = f'  line {number} "{tail}"'
        lines.append(_summary_row(label, reads[chosen]))
    bracket = _final_bracket(text, spans)
    if bracket is None:
        bracket = _last_content(text, spans)
    sx, sy, vx, vy = (fmt3(item) for item in reads[bracket])
    lines.append(
        f"last ']' read: s=({sx}, {sy}) v=({vx}, {vy}). "
        "On a move, S1 walked toward that v."
    )
    return "\n".join(lines)


def resolve_gaze_window(
    n_tokens: int, spec: tuple[str, int, int | None],
) -> tuple[int, int, bool]:
    """Inclusive ``(start, end, clipped)``. ``end < start`` means nothing."""
    kind, left, right = spec
    if n_tokens <= 0:
        return 0, -1, True
    if kind == "tail":
        start = n_tokens - left
        clipped = start < 0
        return max(0, start), n_tokens - 1, clipped
    if right is None:
        raise RuntimeError(f"gaze window {spec!r} has no end")
    clipped = left < 0 or right >= n_tokens or left > right
    start = max(0, left)
    end = min(n_tokens - 1, right)
    if start > end:
        return start, end, True
    return start, end, clipped


def format_gaze_window(
    n: int,
    text: str,
    trace: dict[str, Any],
    spec: tuple[str, int, int | None],
) -> str:
    """One row per token for a window of the BEFORE message."""
    if not valid_trace(trace, text):
        raise RuntimeError(f"memory {n} BEFORE gaze does not match its text")
    reads = [_read4(row) for row in trace["reads"]]
    spans = trace["spans"]
    total = len(reads)
    start, end, clipped = resolve_gaze_window(total, spec)
    if end < start:
        asked = _asked_span(spec, total)
        return (
            f"gaze, memory {n}, BEFORE message, tokens {asked} of {total}:\n"
            "clipped to nothing; that range is past the end."
        )
    header = (
        f"gaze, memory {n}, BEFORE message, tokens {start}-{end} of {total}:"
    )
    rows = [
        header,
        "   i  piece          s                 v",
    ]
    for index in range(start, end + 1):
        quoted = '"' + _piece(text, spans[index]).replace("\n", "\\n") + '"'
        sx, sy, vx, vy = (fmt3(item) for item in reads[index])
        s = f"({sx}, {sy})"
        v = f"({vx}, {vy})"
        rows.append(f"{index:4d}  {quoted:<15}{s}    {v}")
    if clipped:
        rows.append(f"clipped to {start}-{end} of {total}.")
    return "\n".join(rows)


def _asked_span(spec: tuple[str, int, int | None], total: int) -> str:
    kind, left, right = spec
    if kind == "tail":
        start = max(0, total - left)
        return f"{start}-{total - 1}"
    return f"{left}-{right}"


def _settings_text(side: dict[str, Any]) -> str:
    if "error" in side:
        settings = side.get("settings")
        return settings if isinstance(settings, str) else ""
    raw = side.get("settings")
    if not isinstance(raw, str) or not raw:
        return ""
    if raw.startswith("[ERROR"):
        return raw
    opened = settings_json_with_openings(raw)
    if not opened:
        return ""
    return json.dumps(fmt3(json.loads(opened)), indent=2)


def _side_text(side: dict[str, Any], key: str) -> str:
    if "error" in side and key == "image":
        return str(side.get("error") or "")
    return ""


class FailureAnalyst:
    """Browses every flagged memory and talks to the user about patterns.

    Construct once per notebook. :meth:`ask` may run several generations
    when the reply ends in a tool token. :meth:`restart` starts a new
    analyst conversation and clears the loaded set.
    """

    def __init__(
        self,
        cfg: AgentConfig | None = None,
        load_model: bool = True,
        enable_logging: bool = True,
        log_label: str | None = None,
    ):
        self.cfg = cfg or CONFIG
        self.logger = (
            run_logging.new_run_logger(label=log_label or "analyst")
            if enable_logging
            else None
        )
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="nams-analyst-loop", daemon=True,
        )
        self._thread.start()
        self.client = self._run(mem.connect(self.cfg))
        self.model = None
        if load_model:
            self.load_inkling()
        self.session_id = "analyst-" + uuid.uuid4().hex
        self._loaded: dict[int, set[str]] = {}
        self._windows: dict[int, tuple[str, int, int | None]] = {}
        self._flags: list[dict[str, Any]] = []
        self._marked = False
        self._note_round = 0
        self._system = ""
        self.refresh_flags()

    def _run(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def close(self) -> None:
        try:
            if self.client is not None:
                self._run(self.client.close())
        except Exception as exc:  # pragma: no cover - best-effort teardown
            logger.debug("client.close() failed: %s", exc)
        finally:
            self.client = None
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)

    def __enter__(self) -> "FailureAnalyst":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def restart(self) -> dict[str, Any]:
        """A new analyst conversation. The loaded set is cleared.

        NAMS keeps the previous analyst thread. Use
        :meth:`reset_to_pre_analyst` to delete those writes.
        """
        self.session_id = "analyst-" + uuid.uuid4().hex
        self._loaded = {}
        self._windows = {}
        self._marked = False
        self._note_round = 0
        self._system = ""
        self.refresh_flags()
        return {"session_id": self.session_id}

    def reset_to_pre_analyst(self) -> dict[str, Any]:
        """Delete every analyst write, then start a fresh thread.

        Play conversations, flagged messages, snapshots, entities, and
        preferences stay. The loaded model stays. A move in the flagged
        or snapshot count raises after the deletes, because those nodes
        are not analyst data.
        """
        before = self._play_counts()
        self._run(self._wipe_analyst_graph())
        after = self._play_counts()
        if before != after:
            raise RuntimeError(
                "reset_to_pre_analyst changed play data "
                f"(flagged {before['flagged']}->{after['flagged']}, "
                f"snapshots {before['snapshots']}->{after['snapshots']})"
            )
        info = self.restart()
        info["flagged"] = after["flagged"]
        info["snapshots"] = after["snapshots"]
        return info

    def _play_counts(self) -> dict[str, int]:
        return {
            "flagged": self._count(
                "MATCH (m:Message) WHERE m.metadata CONTAINS '\"failure\"' "
                "RETURN count(m) AS c"
            ),
            "snapshots": self._count(
                "MATCH (g:GameSnapshot) RETURN count(g) AS c"
            ),
        }

    def _count(self, query: str) -> int:
        rows = self._run(self.client.graph.execute_write(query, {})) or []
        if not rows:
            raise RuntimeError(f"count query returned no row: {query}")
        value = dict(rows[0]).get("c")
        if value is None:
            raise RuntimeError(f"count query returned no c: {rows[0]!r}")
        return int(value)

    async def _wipe_analyst_graph(self) -> None:
        # Do not match the bare substring "analyst": flag text contains it.
        queries = (
            "MATCH (c:Conversation) "
            "WHERE c.kind = 'analyst' OR c.session_id STARTS WITH 'analyst-' "
            "OPTIONAL MATCH (c)-[:HAS_MESSAGE]->(m) "
            "DETACH DELETE m, c",
            "MATCH (t:ReasoningTrace) "
            "WHERE t.session_id STARTS WITH 'analyst-' "
            "DETACH DELETE t",
            "MATCH (n:SessionNote) "
            "WHERE n.session_id STARTS WITH 'analyst-' "
            "DETACH DELETE n",
            "MATCH (m:Message) "
            "WHERE m.metadata CONTAINS '\"kind\": \"analyst\"' "
            "OR m.metadata CONTAINS '\"kind\":\"analyst\"' "
            "DETACH DELETE m",
        )
        for query in queries:
            await self.client.graph.execute_write(query, {})

    def refresh_flags(self) -> list[dict[str, Any]]:
        """Every flagged memory in NAMS, oldest first.

        Called when the analyst is created, on restart, and at the start
        of each ask, so the list is present before the user types.
        """
        self._flags = self._run(mem.load_all_flagged(self.client))
        return self._flags

    def catalog(self) -> str:
        """The index line plus one line per flagged memory."""
        listing = self._list()
        if not listing:
            return self.loaded_line()
        return self.loaded_line() + "\n" + listing

    def switch_model(
        self,
        key: str,
        purge_others: bool = False,
        checkpoint: str | None = None,
    ) -> dict[str, Any]:
        from .model import switch_session_model
        return switch_session_model(self, key, purge_others, checkpoint)

    def load_analyst_model(self, key: str) -> dict[str, Any]:
        """Load one catalog model. No play checkpoint, no process default.

        Frees whatever trunk this analyst was holding. Does not install
        itself as the process-wide default, so a play session's Gemma
        is left alone unless it was this object's previous model.
        """
        from . import model as model_mod
        from .inkling import build_analyst_model

        previous = self.model
        if previous is not None:
            previous.unload()
        if model_mod._DEFAULT is previous:
            model_mod._DEFAULT = None
        self.model = build_analyst_model(key, self.cfg).load()
        return {
            "key": self.model.key,
            "label": self.model.label,
            "model_id": self.model.model_id,
            "hf_id": self.model.model_id,
        }

    def load_inkling(self) -> dict[str, Any]:
        """Load the unfinetuned Inkling 2 Mini."""
        return self.load_analyst_model("inkling-2-mini")

    def load_gemma_dir(self, path: str) -> dict[str, Any]:
        """Load the Gemma trunk from a play-notebook checkpoint directory.

        A ``gemma_meta.json`` slice is loaded the way play loads a full
        save's ``gemma/`` folder. A bare adapter directory (the parts
        Gemma dropdown, or a coord checkpoint) is stacked on the base
        model. The localizer and S1 are not loaded. The process-wide
        model is this trunk, so a second copy is not kept on the GPU.
        """
        from . import model as model_mod

        folder = Path(path)
        meta_path = folder / "gemma_meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            fmt = meta.get("format")
            key = meta.get("model_key")
            if not key or fmt not in ("peft", "state_dict"):
                raise RuntimeError(
                    "gemma_meta.json must have model_key and format "
                    f"peft|state_dict, got {meta!r}"
                )
        elif (folder / "adapter_config.json").is_file():
            fmt = "peft"
            key = self.cfg.model_key
            meta = None
        else:
            raise RuntimeError(f"no Gemma weights under {folder}")
        self._unload_model()
        self.model = model_mod.VLModel(
            model_mod.spec_for(key), self.cfg, checkpoint=None,
        ).load()
        if fmt == "peft":
            from peft import PeftModel
            self.model.model = PeftModel.from_pretrained(
                self.model.model, str(folder), is_trainable=False,
            )
        else:
            weights = folder / "gemma.pt"
            if not weights.is_file():
                raise FileNotFoundError(f"missing {weights}")
            import torch
            state = torch.load(weights, map_location="cpu", weights_only=True)
            self.model.model.load_state_dict(state)
        self.model.model.eval()
        model_mod._DEFAULT = self.model
        return {"key": key, "path": str(folder), "format": fmt}

    def _unload_model(self) -> None:
        from . import model as model_mod
        current = self.model if self.model is not None else model_mod._DEFAULT
        if current is not None:
            current.unload()
        self.model = None
        model_mod._DEFAULT = None

    def loaded_line(self) -> str:
        return index_line(len(self._flags), self._loaded_numbers())

    def loaded_images(self) -> list[dict[str, Any]]:
        """Pictures in the loaded set: ``n``, ``side``, ``path``."""
        found: list[dict[str, Any]] = []
        for n, side in image_names(self._loaded):
            item = self._flags[n]
            block = item["before"] if side == "before" else item["after"]
            path = block.get("path") if isinstance(block, dict) else None
            if not path:
                continue
            found.append({"n": n, "side": side, "path": str(path)})
        return found

    def notepad(self) -> str:
        notes = self._run(mem.get_session_notes(self.client, self.session_id))
        return mem.format_notepad(notes)

    def ask(
        self,
        question: str,
        on_step: Any = None,
    ) -> dict[str, Any]:
        """One user turn. Tool tokens keep the turn going, up to the cap."""
        if self.model is None:
            raise RuntimeError("FailureAnalyst has no model; pass load_model=True")
        from . import modes
        self._system = self._run(modes.load_failure_analyst_prompt(self.client))
        self.refresh_flags()
        user_msg = self._run(
            self.client.short_term.add_message(
                session_id=self.session_id, role="user", content=question,
                metadata={"kind": "analyst"},
            )
        )
        self._mark()
        trace = self._run(
            mem.start_turn_trace(
                self.client, self.session_id, task=question,
                triggered_by_message_id=user_msg.id,
            )
        )
        replies: list[dict[str, Any]] = []
        tool_calls = 0
        end_text = question
        cap = self.cfg.analyst_max_tool_calls
        try:
            while True:
                messages = self._messages(end_text)
                raw = self.model.generate(
                    messages,
                    max_new_tokens=self.cfg.max_new_tokens,
                    stop_regex=ANALYST_TOOL_PATTERN,
                )
                call, text = parse_analyst_call(raw)
                self._run(
                    self.client.short_term.add_message(
                        session_id=self.session_id, role="assistant",
                        content=text, metadata={"kind": "analyst"},
                    )
                )
                result = {"text": text, "tool_call": call}
                replies.append(result)
                if on_step is not None:
                    on_step(dict(result))
                if call is None:
                    break
                tool_calls += 1
                note = self._apply(call)
                self._run(
                    self.client.short_term.add_message(
                        session_id=self.session_id, role="user",
                        content=note if note else "\n",
                        metadata={"kind": "analyst"},
                    )
                )
                result["tool_result"] = note
                if on_step is not None:
                    on_step({"tool_result": note, "loaded": self.loaded_line()})
                if tool_calls >= cap:
                    limit = f"{cap} tool calls reached. This turn ends."
                    self._run(
                        self.client.short_term.add_message(
                            session_id=self.session_id, role="user",
                            content=limit, metadata={"kind": "analyst"},
                        )
                    )
                    replies.append({"text": limit, "tool_call": None})
                    if on_step is not None:
                        on_step({"text": limit, "tool_call": None})
                    break
                end_text = ANALYST_CONTINUE
        finally:
            self._run(
                mem.complete_turn_trace(
                    self.client, trace,
                    outcome=(
                        f"analyst turn ended after {len(replies)} "
                        f"generation(s), {tool_calls} tool call(s)"
                    ),
                    success=True,
                )
            )
        return {
            "session_id": self.session_id,
            "question": question,
            "replies": replies,
            "tool_calls": tool_calls,
            "loaded": self.loaded_line(),
            "notepad": self.notepad(),
            "images": self.loaded_images(),
        }

    def _loaded_numbers(self) -> list[int]:
        return sorted(n for n, parts in self._loaded.items() if parts)

    def _mark(self) -> None:
        if self._marked:
            return
        self._run(
            self.client.graph.execute_write(
                "MATCH (d:Conversation {session_id: $sid}) "
                "SET d.kind = 'analyst'",
                {"sid": self.session_id},
            )
        )
        self._marked = True

    def _messages(self, end_text: str) -> list[dict[str, Any]]:
        recent = self._run(
            mem.get_recent_messages(
                self.client, self.session_id,
                self.cfg.recent_messages_window, scrub=False,
            )
        )
        suffix = f"[user] {end_text}"
        if recent.endswith(suffix):
            recent = recent[: -len(suffix)].rstrip("\n")
        preamble = [self.catalog()]
        preamble.append(self.notepad())
        if recent:
            preamble.append(recent)
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "\n\n".join(preamble) + "\n\n"},
        ]
        for n in self._loaded_numbers():
            self._append_memory(content, self._flags[n], self._loaded[n])
        content.append({"type": "text", "text": end_text})
        return [
            {"role": "system", "content": [{"type": "text", "text": self._system}]},
            {"role": "user", "content": content},
        ]

    def _append_memory(
        self,
        content: list[dict[str, Any]],
        item: dict[str, Any],
        parts: set[str],
    ) -> None:
        n = int(item["n"])
        blocks: list[str] = [
            memory_header(n, str(item.get("session_id") or ""), item.get("timestamp")),
        ]

        def flush() -> None:
            if not blocks:
                return
            content.append({"type": "text", "text": "\n".join(blocks) + "\n"})
            blocks.clear()

        def text_part(name: str, value: str) -> None:
            if name not in parts:
                return
            blocks.append(f"{_PART_LABEL[name]}:")
            blocks.append(value)

        text_part("bq", str(item.get("before_question") or ""))
        text_part("bm", str(item.get("before_message") or ""))
        if "bi" in parts:
            self._image_part(content, blocks, item["before"], "before", n, flush)
        text_part("bs", _settings_text(item["before"]))
        text_part("aq", str(item.get("after_question") or ""))
        text_part("am", str(item.get("after_message") or ""))
        if "ai" in parts:
            self._image_part(content, blocks, item["after"], "after", n, flush)
        text_part("as", _settings_text(item["after"]))
        if "gaze" in parts:
            blocks.append(self._gaze_text(item))
        flush()

    def _image_part(
        self,
        content: list[dict[str, Any]],
        blocks: list[str],
        side: dict[str, Any],
        which: str,
        n: int,
        flush: Any,
    ) -> None:
        if "error" in side:
            blocks.append(str(side.get("error") or ""))
            return
        path = side.get("path")
        if not path:
            raise RuntimeError(f"memory {n} {which} image has no path")
        blocks.append(f"[image] {which}, memory {n}")
        flush()
        content.append({"type": "image", "url": str(Path(path).resolve())})

    def _gaze_text(self, item: dict[str, Any]) -> str:
        n = int(item["n"])
        text = str(item.get("before_message") or "")
        gaze = item.get("gaze")
        if not isinstance(gaze, dict):
            return NO_PRIOR_GAZE
        summary = format_gaze_summary(n, text, gaze)
        spec = self._windows.get(n)
        if spec is None:
            return summary
        return summary + "\n" + format_gaze_window(n, text, gaze, spec)

    def _apply(self, call: dict[str, Any]) -> str:
        tool = call["tool"]
        if tool == "LIST":
            return self._list()
        if tool == "SEARCH":
            return self._run(self._search(call["query"]))
        if tool == "NOTE":
            return self._note(call["key"], call["text"])
        if tool == "FORGET":
            return self._forget(call["key"])
        if tool in ("LOAD", "DROP", "GAZE"):
            return self._edit_loaded(call)
        raise RuntimeError(f"unknown analyst tool {tool!r}")

    def _edit_loaded(self, call: dict[str, Any]) -> str:
        tool = call["tool"]
        if tool == "DROP" and call["n"] is None:
            self._loaded = {}
            self._windows = {}
            return "dropped all"
        n = int(call["n"])
        if n < 0 or n >= len(self._flags):
            return f"no memory {n}."
        if tool == "DROP":
            if call["parts"] is None:
                self._loaded.pop(n, None)
                self._windows.pop(n, None)
                return f"dropped {n}"
            unknown = _unknown(call["parts"])
            if unknown:
                return unknown
            current = self._loaded.get(n, set())
            current.difference_update(call["parts"])
            if "gaze" not in current:
                self._windows.pop(n, None)
            if current:
                self._loaded[n] = current
            else:
                self._loaded.pop(n, None)
            return "dropped " + str(n) + ": " + " ".join(call["parts"])
        if tool == "LOAD":
            parts = list(DEFAULT_PARTS if call["parts"] is None else call["parts"])
        else:
            parts = ["gaze"]
        unknown = _unknown(parts)
        if unknown:
            return unknown
        wanted = set(parts)
        refusal = image_cap_refusal(
            self._loaded, n, wanted, self.cfg.analyst_max_images,
        )
        if refusal:
            return refusal
        self._loaded.setdefault(n, set()).update(wanted)
        if tool == "GAZE" and call["window"] is not None:
            self._windows[n] = call["window"]
            return self._gaze_text(self._flags[n])
        if tool == "GAZE":
            return self._gaze_text(self._flags[n])
        return "loaded " + str(n) + ": " + " ".join(parts)

    def _list(self) -> str:
        lines = []
        for item in self._flags:
            note = str(item.get("failure") or "").replace("\n", " ")
            sid = str(item.get("session_id") or "")[:8]
            lines.append(f"{item['n']}  {sid}  {note[:80]}")
        return "\n".join(lines)

    async def _search(self, query: str) -> str:
        ids = {str(item["id"]): int(item["n"]) for item in self._flags}
        msgs = await self.client.short_term.search_messages(
            query, session_id=self.session_id, limit=50,
        )
        lines = []
        for msg in msgs or []:
            mid = str(getattr(msg, "id", "") or "")
            if mid not in ids:
                continue
            content = str(getattr(msg, "content", "") or "")
            first = content.split("\n", 1)[0] if content else ""
            lines.append(f"memory {ids[mid]}: {first}")
            if len(lines) >= 8:
                break
        return "\n".join(lines)

    def _note(self, key: str, text: str) -> str:
        if not key:
            return "no note key."
        self._note_round += 1
        self._run(mem.set_session_note(
            self.client, self.session_id, key, text, self._note_round,
        ))
        return f"noted {key}"

    def _forget(self, key: str) -> str:
        if not key:
            return "no note key."
        self._run(mem.delete_session_note(self.client, self.session_id, key))
        return f"forgot {key}"


def _unknown(parts: list[str]) -> str:
    for part in parts:
        if part not in PARTS:
            return f"unknown part {part}."
    return ""
