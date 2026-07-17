"""Stateful, notebook-friendly privileged debrief session (mode 4).

:class:`DebriefSession` is the mode-4 counterpart of
:class:`agent.interactive.InteractiveSession`: an interactive chat over any
**recorded** play conversation, with full privileged access to ground truth --
the saved snapshot images AND the exact Settings JSON the player never saw.

One :meth:`ask` is a *multi-generation turn*, mirroring play mode's structure:
a cursor points at one of the player's recorded messages (the "current
message"), whose text + the ONE frame the player saw + exact settings are
always in context, alongside the user instruction the player was answering.
The model may end a reply with a ``[SHOW n]`` / ``[NEXT]`` / ``[BACK]`` tool
token to move the cursor; we stop generation at the token (generation-time
regex stop -- see :class:`agent.model.RegexStopCriteria`), swap the current
message, and generate again. The turn ends when a reply carries no tool
token, or the per-turn tool budget (``cfg.debrief_max_tool_calls``) is
exhausted.

Persistence: every debrief is a real NAMS conversation with session id
``debrief-<uuid>``. After its first message exists, the ``Conversation`` node
is marked ``kind='debrief'`` / ``debrief_of=<play sid>`` and linked to the
analyzed play conversation via a ``DEBRIEF_OF`` edge -- so debriefs are
excluded from the play-conversation picker (:meth:`list_conversations`) and
visible as their own threads in the graph. Each :meth:`ask` records one
reasoning trace (task = the question) with cursor moves as tool calls;
fetched snapshots are linked to the assistant message with
``CAPTURED_STATE {role:'observation'}``.

The debrief conversation itself is quarantined from the playing agent (mode-1
context is session-scoped). The ONLY way debrief content reaches the play
conversation is the explicit :meth:`save_self_eval`, which distills the
debrief into a mode-3-format verdict (``kind='self_evaluation'`` message +
reasoning trace on the play conversation).

Async bridge: same pattern as ``InteractiveSession`` -- a dedicated asyncio
loop on a background thread runs all NAMS calls; public methods are plain sync
calls safe to wire to ipywidgets callbacks. The heavy ``model.generate`` runs
on the calling thread, not the loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable

from .config import AgentConfig, CONFIG
from . import game_io
from . import image_store
from . import memory as mem
from . import modes
from . import run_logging
from .model import get_model

logger = logging.getLogger(__name__)


def _msg_kind(m: dict[str, Any]) -> str | None:
    """The ``kind`` from a Message node's metadata. NAMS stores metadata as a
    JSON string property; tolerate an already-parsed dict too."""
    meta = m.get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except ValueError:
            logger.warning("Unparseable message metadata: %.80r", meta)
            return None
    return (meta or {}).get("kind")


class DebriefSession:
    """A persistent, privileged mode-4 debrief session over recorded games.

    Construct once per notebook (connects NAMS, loads Gemma), then call
    :meth:`list_conversations` / :meth:`select` / :meth:`ask` /
    :meth:`save_self_eval` from UI callbacks and :meth:`close` when done.
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
            run_logging.new_run_logger(label=log_label or "debrief")
            if enable_logging
            else None
        )

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="nams-debrief-loop", daemon=True
        )
        self._thread.start()

        self.client = self._run(mem.connect(self.cfg))
        self.model = get_model(self.cfg) if load_model else None

        # Analysis target state (set by select()).
        self.play_session_id: str | None = None
        self.debrief_session_id: str | None = None
        self._conversation_marked = False
        # Index of the player's recorded (assistant) messages: n -> dict with
        # kind, action, content, the ONE frame the player saw, and the user
        # instruction that was live at that message.
        self._msg_index: list[dict[str, Any]] = []
        self._trace_block: str = "(no player messages recorded)"
        self._snapshots: list[dict[str, Any]] = []
        # Navigation cursor: the message the reviewer is currently looking at
        # ('Current message' in the prompt). [SHOW n] jumps it, [NEXT]/[BACK]
        # move it by one. Persists across ask() turns; reset by select().
        self._cursor: int | None = None

    # ------------------------------------------------------------------ bridge
    def _run(self, coro: Any) -> Any:
        """Run a coroutine on the background loop and block for its result."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    # ------------------------------------------------------------------ public
    def list_conversations(self) -> list[dict[str, Any]]:
        """All PLAY conversations (debriefs excluded), newest first, with
        message counts -- feeds the notebook's conversation picker."""
        rows = self._run(
            self.client.query.cypher(
                "MATCH (c:Conversation) "
                "WHERE c.kind IS NULL OR c.kind <> 'debrief' "
                "OPTIONAL MATCH (c)-[:HAS_MESSAGE]->(m:Message) "
                "RETURN c.session_id AS session_id, c.created_at AS created_at, "
                "count(m) AS n_messages "
                "ORDER BY c.created_at DESC",
                {},
            )
        )
        convos = [dict(r) for r in (rows or [])]
        run_logging.log_db_retrieval(
            function="DebriefSession.list_conversations",
            arguments={},
            result=convos,
        )
        return convos

    def select(self, play_session_id: str) -> dict[str, Any]:
        """Set the analysis target and start a NEW debrief conversation.

        Prefetches the play session's message+snapshot index: ALL assistant
        messages in chronological order -- moves, reflections and answers --
        define the message numbering that ``[SHOW n]`` refers to. Each entry
        carries the ONE frame the player saw when writing it (the 'before'
        snapshot; 'after' / any linked snapshot as fallbacks) and the user
        instruction that was live at that point. The previous debrief
        conversation (if any) simply remains stored in NAMS; this session
        stops appending to it.
        """
        rows = self._run(
            image_store.fetch_messages_with_snapshots(self.client, play_session_id)
        )
        if not rows:
            raise ValueError(
                f"No messages found for play session {play_session_id!r}; "
                "cannot debrief an empty conversation."
            )

        self.play_session_id = play_session_id
        self.debrief_session_id = "debrief-" + mem.new_session_id()
        self._conversation_marked = False
        self._msg_index = []
        self._snapshots = self._run(
            image_store.fetch_session_snapshots(self.client, play_session_id)
        )

        instruction = "(no user instruction recorded)"
        for row in rows:
            m = row.get("message") or {}
            content = str(m.get("content", ""))
            role = m.get("role")
            if role == "user":
                # The most recent user message is the instruction the player
                # was answering from here on -- it 'hangs' over every
                # subsequent player message, exactly as it did during play.
                instruction = content
                continue
            if role != "assistant":
                continue
            snaps = row.get("snapshots") or []
            frame = (
                next((s for s in snaps if s.get("label") == "before"), None)
                or next((s for s in snaps if s.get("label") == "after"), None)
                or (snaps[0] if snaps else None)
            )
            self._msg_index.append({
                "n": len(self._msg_index),
                "kind": _msg_kind(m),
                "action": game_io.parse_action(content),
                "content": content,
                "frame": frame,
                "instruction": instruction,
            })

        if self._msg_index:
            moves = [e["action"] for e in self._msg_index if e["action"]]
            n = len(self._msg_index)
            self._trace_block = (
                f"{n} recorded player messages (0..{n - 1}); "
                f"the player's moves in order: {modes._summarize_actions(moves)}"
            )
            self._cursor = 0
        else:
            self._trace_block = "(no player messages recorded)"
            self._cursor = None

        latest_frame = self._resolve_path(self._snapshots[-1]["path"]) if self._snapshots else None
        info = {
            "play_session_id": play_session_id,
            "debrief_session_id": self.debrief_session_id,
            "n_messages": len(rows),
            "n_player_messages": len(self._msg_index),
            "n_moves": sum(1 for e in self._msg_index if e["action"]),
            "n_snapshots": len(self._snapshots),
            "latest_frame_path": latest_frame,
        }
        run_logging.log_db_retrieval(
            function="DebriefSession.select",
            arguments={"play_session_id": play_session_id},
            result=info,
        )
        logger.info(
            "Debrief target selected: play=%s debrief=%s "
            "(%d player messages, %d snapshots)",
            play_session_id, self.debrief_session_id,
            len(self._msg_index), len(self._snapshots),
        )
        return info

    def ask(
        self,
        question: str,
        on_step: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """One multi-generation debrief turn (play-mode style).

        The model may end each reply with ``[SHOW n]`` / ``[NEXT]`` /
        ``[BACK]``; we move the cursor, swap the current message (text +
        frame + settings) into context, and generate again, until a reply
        with no tool token ends the turn or ``cfg.debrief_max_tool_calls``
        moves have been made. ``on_step`` (if given) fires after every
        generation with the reply text, any tool call, and the frame fetched
        for it.
        """
        if self.debrief_session_id is None:
            raise ValueError("No play conversation selected; call select() first.")

        user_msg = self._run(
            self.client.short_term.add_message(
                session_id=self.debrief_session_id, role="user", content=question,
                metadata={"kind": "debrief"},
            )
        )
        self._ensure_conversation_marked()

        trace = self._run(
            mem.start_turn_trace(
                self.client, self.debrief_session_id, task=question,
                triggered_by_message_id=user_msg.id,
            )
        )

        replies: list[dict[str, Any]] = []
        current = (
            self._message_block(self._cursor) if self._cursor is not None else None
        )
        # The question itself is already the newest entry of the recency
        # window (persisted above), so don't repeat it verbatim here.
        prompt_text = "Answer the newest user message in the debrief conversation above."
        tool_calls = 0
        try:
            while True:
                recent = self._run(
                    mem.get_recent_messages(
                        self.client, self.debrief_session_id,
                        self.cfg.recent_messages_window, scrub=False,
                    )
                )
                messages = modes.build_debrief_messages(
                    self._trace_block + "\n" + self._cursor_line(),
                    recent, current, prompt_text,
                )
                raw = self.model.generate(
                    messages,
                    max_new_tokens=self.cfg.gemma_max_new_tokens,
                    stop_regex=modes.DEBRIEF_TOOL_PATTERN,
                )
                call, text = modes.parse_debrief_call(raw)

                assistant_msg = self._run(
                    self.client.short_term.add_message(
                        session_id=self.debrief_session_id, role="assistant",
                        content=text, metadata={"kind": "debrief"},
                    )
                )
                step_node = self._run(
                    self.client.reasoning.add_step(trace.id, thought=text)
                )

                moved = False
                if call is not None:
                    target, reason = self._resolve_tool_target(call)
                    found = target is not None
                    self._run(
                        self.client.reasoning.record_tool_call(
                            step_node.id, call["tool"],
                            {k: v for k, v in call.items() if k != "tool"},
                            result={"found": found, "message": target},
                        )
                    )
                    tool_calls += 1
                    if found:
                        moved = True
                        self._cursor = target
                        current = self._message_block(target)
                        if current.get("snapshot_id"):
                            self._run(
                                image_store.link_snapshot_to_message(
                                    self.client, assistant_msg.id,
                                    current["snapshot_id"], role="observation",
                                )
                            )
                        prompt_text = (
                            f"(Continuing your reply to: {question}\n"
                            f"Message {target} is now the current message: "
                            "its recorded text, frame, and exact settings are "
                            "in your context above. Continue your analysis; "
                            "emit another tool token ([SHOW n], [NEXT], "
                            "[BACK]) to inspect a different message, or "
                            "finish your answer.)"
                        )
                    else:
                        prompt_text = (
                            f"(Continuing your reply to: {question}\n"
                            f"{reason} Remember: [SHOW n] retrieves recorded "
                            "message n's text, frame, and exact settings. "
                            "Continue your analysis; emit a valid tool token "
                            "or finish your answer.)"
                        )

                result = {
                    "kind": "debrief_generation",
                    "generation": len(replies),
                    "raw": raw,
                    "text": text,
                    "tool_call": call,
                    "cursor": self._cursor,
                    "frames": (
                        [current["path"]]
                        if moved and current and current.get("path") else []
                    ),
                }
                replies.append(result)
                if on_step is not None:
                    on_step(result)

                if call is None:
                    break
                if tool_calls >= self.cfg.debrief_max_tool_calls:
                    logger.warning(
                        "Debrief tool budget exhausted (%d tool calls); "
                        "ending the turn with the last reply.",
                        tool_calls,
                    )
                    break
        finally:
            self._run(
                mem.complete_turn_trace(
                    self.client, trace,
                    outcome=(
                        f"debrief turn ended after {len(replies)} "
                        f"generation(s), {tool_calls} tool call(s)"
                    ),
                    success=True,
                )
            )

        return {
            "play_session_id": self.play_session_id,
            "debrief_session_id": self.debrief_session_id,
            "question": question,
            "replies": replies,
            "num_generations": len(replies),
            "tool_calls": tool_calls,
            "trace_id": str(trace.id),
        }

    def save_self_eval(self) -> dict[str, Any]:
        """Distill the debrief conversation into a structured verdict and store
        it on the analyzed PLAY conversation (mode-3 convention:
        ``kind='self_evaluation'`` message + reasoning trace). Raises if
        nothing has been discussed yet."""
        if self.debrief_session_id is None:
            raise ValueError("No play conversation selected; call select() first.")
        return self._run(
            modes.persist_debrief_verdict(
                self.client, self.play_session_id, self.debrief_session_id,
                self.model, cfg=self.cfg,
            )
        )

    def dump_db(self, name: str | None = None, include_embeddings: bool = False) -> dict[str, Any]:
        """Dump the current DB status (all nodes + relationships) to a ``.dump``
        JSON file for offline inspection. Reads over the live bolt connection --
        it does NOT stop Neo4j, so it is safe to call mid-session.

        The file lands in this run's log directory (or a fresh ``logs/`` file if
        logging is disabled). Returns ``{path, nodes, relationships}``."""
        path = run_logging.resolve_dump_path(self.logger, name)
        return self._run(
            mem.dump_database_to_file(self.client, path, include_embeddings=include_embeddings)
        )

    def close(self) -> None:
        """Close the memory client and stop the background loop."""
        try:
            if self.client is not None:
                self._run(self.client.close())
        except Exception as exc:  # pragma: no cover - best-effort teardown
            logger.debug("client.close() failed: %s", exc)
        finally:
            self.client = None
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)

    def __enter__(self) -> "DebriefSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------------------------------------------------------------- internal
    def _resolve_path(self, path: str) -> str:
        """Snapshot nodes store repo-relative paths; resolve for the model."""
        return str(Path(path).resolve())

    def _ensure_conversation_marked(self) -> None:
        """Mark the debrief Conversation node (kind / debrief_of / DEBRIEF_OF
        edge) once its node exists, i.e. after the first message was added."""
        if self._conversation_marked:
            return
        self._run(
            self.client.graph.execute_write(
                "MATCH (d:Conversation {session_id: $dsid}) "
                "SET d.kind = 'debrief', d.debrief_of = $psid "
                "WITH d MATCH (p:Conversation {session_id: $psid}) "
                "MERGE (d)-[:DEBRIEF_OF]->(p)",
                {"dsid": self.debrief_session_id, "psid": self.play_session_id},
            )
        )
        self._conversation_marked = True

    def _message_block(self, n: int) -> dict[str, Any]:
        """The current-message context block for message ``n``: header,
        recorded text, the ONE frame the player saw (before-the-move
        snapshot), its settings, and the instruction live at that message."""
        e = self._msg_index[n]
        if e["action"]:
            desc = f"move: {e['action']}"
        elif e["kind"] == "reflection":
            desc = "reflection, no move"
        else:
            desc = f"{e['kind'] or 'commentary'}, no move"
        frame = e["frame"]
        return {
            "header": f"Current message under inspection -- message {n} ({desc}):",
            "content": e["content"],
            "instruction": e["instruction"],
            "path": self._resolve_path(frame["path"]) if frame else None,
            "settings_json": frame.get("settings_json") if frame else None,
            "snapshot_id": frame.get("id") if frame else None,
        }

    def _cursor_line(self) -> str:
        """The 'Current message' line for the prompt context."""
        if self._cursor is None:
            return "Current message: (none -- no player messages recorded)"
        return (
            f"Current message: {self._cursor} "
            f"(valid messages: 0..{len(self._msg_index) - 1})"
        )

    def _resolve_tool_target(self, call: dict[str, Any]) -> tuple[int | None, str]:
        """Map a parsed tool call to a message number. Returns ``(n, '')`` on
        success or ``(None, reason)`` explaining why the call is invalid."""
        n_msgs = len(self._msg_index)
        if n_msgs == 0:
            return None, (
                "This session has no recorded player messages, so there is "
                "nothing to show."
            )
        tool = call["tool"]
        if tool == "SHOW":
            n = call["step"]
            if 0 <= n < n_msgs:
                return n, ""
            return None, (
                f"Message {n} does not exist -- valid messages are "
                f"0..{n_msgs - 1}."
            )
        if self._cursor is None:
            # Unreachable when messages exist (select() sets the cursor), but
            # fail loudly rather than guess a position.
            return None, (
                "There is no current message to navigate from; use "
                "[SHOW n] first."
            )
        n = self._cursor + 1 if tool == "NEXT" else self._cursor - 1
        if 0 <= n < n_msgs:
            return n, ""
        edge = "last" if tool == "NEXT" else "first"
        return None, (
            f"[{tool}] failed: you are already at the {edge} message "
            f"(current message: {self._cursor}, valid messages: "
            f"0..{n_msgs - 1})."
        )
