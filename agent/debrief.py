"""Stateful, notebook-friendly privileged debrief session (mode 4).

:class:`DebriefSession` is the mode-4 counterpart of
:class:`agent.interactive.InteractiveSession`: an interactive chat over any
**recorded** play conversation, with full privileged access to ground truth --
the saved snapshot images AND the exact Settings JSON the player never saw.

One :meth:`ask` is a *multi-generation turn*, mirroring play mode's structure:
the model may end a reply with a ``[SHOW n]`` tool token to pull up any
recorded step's frames (before/after images + exact settings); we stop
generation at the token (generation-time regex stop -- see
:class:`agent.model.RegexStopCriteria`), fetch the frames, and generate again
on the enriched context. The turn ends when a reply carries no tool token, or
the per-turn tool budget (``cfg.debrief_max_tool_calls``) is exhausted.

Persistence: every debrief is a real NAMS conversation with session id
``debrief-<uuid>``. After its first message exists, the ``Conversation`` node
is marked ``kind='debrief'`` / ``debrief_of=<play sid>`` and linked to the
analyzed play conversation via a ``DEBRIEF_OF`` edge -- so debriefs are
excluded from the play-conversation picker (:meth:`list_conversations`) and
visible as their own threads in the graph. Each :meth:`ask` records one
reasoning trace (task = the question) with ``[SHOW]`` fetches as tool calls;
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
        # Per-move index of the analyzed session: step number -> dict with
        # action, content, before/after frame paths + settings.
        self._move_index: list[dict[str, Any]] = []
        self._move_listing: str = "(no moves recorded)"
        self._snapshots: list[dict[str, Any]] = []

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

        Prefetches the play session's message+snapshot index: assistant move
        messages in chronological order define the step numbering that
        ``[SHOW n]`` refers to (snapshots do not reliably carry a step
        property). The previous debrief conversation (if any) simply remains
        stored in NAMS; this session stops appending to it.
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
        self._move_index = []
        self._snapshots = self._run(
            image_store.fetch_session_snapshots(self.client, play_session_id)
        )

        for row in rows:
            m = row.get("message") or {}
            if m.get("role") != "assistant":
                continue
            action = game_io.parse_action(str(m.get("content", "")))
            if not action:
                continue  # answers / reflections are not moves
            snaps = row.get("snapshots") or []
            before = next((s for s in snaps if s.get("label") == "before"), None)
            after = next((s for s in snaps if s.get("label") == "after"), None)
            self._move_index.append({
                "step": len(self._move_index),
                "action": action,
                "content": str(m.get("content", "")),
                "before": before,
                "after": after,
            })

        if self._move_index:
            self._move_listing = "\n".join(
                f"step {e['step']}: {e['action']}" for e in self._move_index
            )
            summary = modes._summarize_actions(
                [e["action"] for e in self._move_index]
            )
            self._move_listing += f"\n(summary: {summary})"
        else:
            self._move_listing = "(no moves recorded)"

        latest_frame = self._resolve_path(self._snapshots[-1]["path"]) if self._snapshots else None
        info = {
            "play_session_id": play_session_id,
            "debrief_session_id": self.debrief_session_id,
            "n_messages": len(rows),
            "n_moves": len(self._move_index),
            "n_snapshots": len(self._snapshots),
            "latest_frame_path": latest_frame,
        }
        run_logging.log_db_retrieval(
            function="DebriefSession.select",
            arguments={"play_session_id": play_session_id},
            result=info,
        )
        logger.info(
            "Debrief target selected: play=%s debrief=%s (%d moves, %d snapshots)",
            play_session_id, self.debrief_session_id,
            len(self._move_index), len(self._snapshots),
        )
        return info

    def ask(
        self,
        question: str,
        on_step: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """One multi-generation debrief turn (play-mode style).

        The model may end each reply with ``[SHOW n]``; we fetch that step's
        frames and generate again, until a reply with no tool token ends the
        turn or ``cfg.debrief_max_tool_calls`` fetches have been made.
        ``on_step`` (if given) fires after every generation with the reply
        text, any tool call, and the frames fetched for it.
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
        frames = self._default_frames()
        prompt_text = f"User question: {question}"
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
                    self._move_listing, recent, frames, prompt_text
                )
                raw = self.model.generate(
                    messages,
                    max_new_tokens=self.cfg.gemma_max_new_tokens,
                    stop_regex=modes.SHOW_CALL_PATTERN,
                )
                show_step, text = modes.parse_show_call(raw)

                assistant_msg = self._run(
                    self.client.short_term.add_message(
                        session_id=self.debrief_session_id, role="assistant",
                        content=text, metadata={"kind": "debrief"},
                    )
                )
                step_node = self._run(
                    self.client.reasoning.add_step(trace.id, thought=text)
                )

                frames = []
                if show_step is not None:
                    found = 0 <= show_step < len(self._move_index)
                    self._run(
                        self.client.reasoning.record_tool_call(
                            step_node.id, "SHOW", {"step": show_step},
                            result={"found": found},
                        )
                    )
                    tool_calls += 1
                    if found:
                        frames = self._frames_for_step(show_step)
                        for f in frames:
                            self._run(
                                image_store.link_snapshot_to_message(
                                    self.client, assistant_msg.id,
                                    f["snapshot_id"], role="observation",
                                )
                            )
                        prompt_text = (
                            f"(Continuing your reply to: {question}\n"
                            f"The frames for step {show_step} are attached "
                            "above. Continue your analysis; emit another "
                            "[SHOW n] if you need another step, or finish "
                            "your answer.)"
                        )
                    else:
                        prompt_text = (
                            f"(Continuing your reply to: {question}\n"
                            f"Step {show_step} does not exist -- valid steps "
                            f"are 0..{len(self._move_index) - 1}. Continue "
                            "your analysis; emit a valid [SHOW n] or finish "
                            "your answer.)"
                        )

                result = {
                    "kind": "debrief_generation",
                    "generation": len(replies),
                    "raw": raw,
                    "text": text,
                    "show_step": show_step,
                    "frames": [f["path"] for f in frames],
                }
                replies.append(result)
                if on_step is not None:
                    on_step(result)

                if show_step is None:
                    break
                if tool_calls >= self.cfg.debrief_max_tool_calls:
                    logger.warning(
                        "Debrief tool budget exhausted (%d [SHOW] calls); "
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
                        f"generation(s), {tool_calls} [SHOW] call(s)"
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

    def _default_frames(self) -> list[dict[str, Any]]:
        """The last ``cfg.debrief_max_frames`` saved snapshots (chronological;
        the final one is the session's current state)."""
        tail = self._snapshots[-self.cfg.debrief_max_frames:]
        frames = []
        n = len(tail)
        for i, s in enumerate(tail):
            is_last = i == n - 1
            caption = (
                f"Attached frame {i + 1}/{n} (label={s.get('label')})"
                + (" -- the MOST RECENT saved frame, i.e. the session's "
                   "current state:" if is_last else ":")
            )
            frames.append({
                "snapshot_id": s.get("id"),
                "path": self._resolve_path(s["path"]),
                "caption": caption,
                "settings_json": s.get("settings_json"),
            })
        return frames

    def _frames_for_step(self, step: int) -> list[dict[str, Any]]:
        """Before/after frames (image + settings) for move ``step``."""
        entry = self._move_index[step]
        frames = []
        for label, snap in (("BEFORE", entry["before"]), ("AFTER", entry["after"])):
            if snap is None:
                continue
            frames.append({
                "snapshot_id": snap.get("id"),
                "path": self._resolve_path(snap["path"]),
                "caption": (
                    f"[SHOW {step}] result: step {step} "
                    f"({entry['action']}) -- screen {label} the move:"
                ),
                "settings_json": snap.get("settings_json"),
            })
        return frames
