"""Stateful, notebook-friendly interactive self-eval session.

:class:`InteractiveSelfEvalSession` drives the "interactive self-eval"
notebook: a two-phase loop over one persistent game and ONE conversation
thread, alternating between a scene-scoped player and a privileged analyst.

Each round:

  1. **Player phase** (:meth:`ask_player`). The user asks about the CURRENT
     scene -- usually "what's the best move?", sometimes a general question
     ("are you facing the gold?"). The player answers in a SINGLE generation
     under ``SYSTEM_PROMPT_SCENE_PLAY`` (same memory context as play mode:
     recency window + semantic search, settings scrubbed; ``[SEARCH]`` over
     the semantic + reasoning tiers allowed). If the reply ends in a move
     token, the move is parsed but **deliberately NOT applied** -- it is held
     as the pending action so the analyst can judge the decision before its
     outcome exists.

  2. **Analyst phase** (:meth:`ask_analyst`, repeatable). Control returns to
     the user, who may edit the default analysis question or submit it as-is,
     and then go BACK AND FORTH with the analyst as long as they like (each
     follow-up is another :meth:`ask_analyst` call in the same round). The
     analyst reviews ONLY the player's latest reply, under
     ``SYSTEM_PROMPT_SCENE_ANALYST``, with privileged access: the exact frame
     the player saw AND its Settings JSON (which the player never sees), plus
     ``[SEARCH]`` over all memory tiers. No [SHOW]/[NEXT]/[BACK] navigation
     exists in this mode -- there is exactly one message to review. Each
     verdict is stored in the SAME conversation (assistant message,
     ``kind='analysis'``, content prefixed "(analyst)") so the player's
     recency window naturally includes past analyses. Any ``WRONG: "..."``
     error spans in a verdict are verified against the recorded player reply
     by exact substring match (:func:`agent.modes.parse_wrong_spans`).

  3. **End of round** (:meth:`end_round`). When the user is satisfied with
     the analysis, the pending move is propagated to the game, the 'after'
     frame rendered, the round's trace completed, and the phase flips back
     to "player".

The player's memory access works exactly as in play mode -- and because the
analyst writes into the same session, the player sees the analyst's feedback
in later rounds. The privacy asymmetry holds: analyst messages are prose
(ratings + reasoning), and the recency window / semantic search remain
settings-scrubbed for the player.

:meth:`reset_game` swaps in a brand-new random bare board mid-conversation
and records that fact as a message, so the agent's memory never silently
jumps between unrelated boards.

Inherits the async bridge, model, logging, dump/close machinery from
:class:`agent.interactive.InteractiveSession`; the multi-move :meth:`ask` of
the parent is disabled (this mode is strictly one generation per scene).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from . import game_io
from . import image_store
from . import memory as mem
from . import modes
from .interactive import InteractiveSession

logger = logging.getLogger(__name__)


#: Pre-filled into the player text box in the notebook; the user may edit it
#: or submit it unchanged.
DEFAULT_PLAYER_QUESTION = "Please make the right move for this position."

#: Pre-filled into the analyst text box in the notebook; the user may edit it
#: or submit it unchanged.
DEFAULT_ANALYST_QUESTION = (
    "Analyze the player's performance, using newly available privileged "
    "information. Go through the response part by part -- the OBS line, the "
    "reasoning, and the move token -- and say which parts were correct and "
    "which were incorrect.\n"
    "For EVERY mistaken phrase, add a line of the form:\n"
    "WRONG: \"<the exact words copied from the player's reply>\"\n"
    "(one line per mistake, copied verbatim so the words can be found in "
    "the reply). Then give a final rating from -1.0 to 1.0."
)


class InteractiveSelfEvalSession(InteractiveSession):
    """A persistent player/analyst self-eval session over one conversation.

    Construct once per notebook (connects NAMS, loads Gemma), then call
    :meth:`ask_player` / :meth:`ask_analyst` / :meth:`reset_game` /
    :meth:`restart` from UI callbacks and :meth:`close` when done. The
    ``phase`` attribute ("player" or "analyst") tells the UI which input to
    enable.
    """

    DEFAULT_PLAYER_QUESTION = DEFAULT_PLAYER_QUESTION
    DEFAULT_ANALYST_QUESTION = DEFAULT_ANALYST_QUESTION

    def __init__(self, *args: Any, log_label: str | None = None, **kwargs: Any):
        super().__init__(*args, log_label=log_label or "self_eval", **kwargs)

    # ------------------------------------------------------------------ state
    def restart(self) -> dict[str, Any]:
        """Full reset: new bare game AND a new conversation thread. Also
        clears the phase machine back to the player phase."""
        self.phase: str = "player"
        # Everything the analyst needs about the player's latest reply, plus
        # the not-yet-applied move and the still-open turn trace.
        self._pending: dict[str, Any] | None = None
        return super().restart()

    def reset_game(self, record: bool = True) -> dict[str, Any]:
        """Swap in a brand-new random bare board, SAME conversation.

        With ``record=True`` (default) the reset is written to the
        conversation as a message, so the agent's memory carries an explicit
        marker that everything before it happened on a DIFFERENT board.
        """
        if self.phase != "player":
            raise ValueError(
                "Cannot reset the game mid-round: the analyst has not run "
                "and a move may be pending. Finish the round first."
            )
        self.game = game_io.new_bare_game(gameSize=self.cfg.game_size)
        if record:
            self._run(
                self.client.short_term.add_message(
                    session_id=self.session_id, role="user",
                    content=(
                        "(The game was reset: a brand-new random board was "
                        "generated. Previous scenes, moves, and analyses "
                        "refer to a DIFFERENT board and no longer describe "
                        "what you see.)"
                    ),
                    metadata={"kind": "game_reset"},
                )
            )
        logger.info("Game reset (session %s kept).", self.session_id)
        return {
            "session_id": self.session_id,
            "frame_path": self.current_frame_path(),
            "gold_remaining": game_io.gold_remaining(self.game),
        }

    def ask(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError(
            "InteractiveSelfEvalSession has no multi-move ask(); use "
            "ask_player() / ask_analyst()."
        )

    # ----------------------------------------------------------------- player
    def ask_player(self, question: str) -> dict[str, Any]:
        """Player phase: one single-generation answer about the CURRENT scene.

        A move token in the reply is parsed and stored as the pending action
        but NOT applied -- propagation happens in :meth:`end_round`, after
        the analyst exchanges. Flips ``phase`` to "analyst".
        """
        if self.phase != "player":
            raise ValueError(
                "A player reply is already awaiting analysis; run "
                "ask_analyst() first."
            )

        # 1. Snapshot the current ('before') frame -> disk + GameSnapshot node.
        snapshot_before_id = image_store.snapshot_id()
        settings_before = game_io.game_to_settings_dict(self.game)
        before_path, before_props = self._run(
            image_store.store_snapshot(
                self.client, self.session_id, snapshot_before_id, self.game,
                settings_before, cfg=self.cfg, label="before",
            )
        )

        # 2. Memory context, exactly as in play mode (settings scrubbed).
        query = modes._retrieval_query(
            question, 0, None, game_io.gold_remaining(self.game)
        )
        ctx = self._run(
            mem.get_game_context(
                self.client, self.session_id, query=query,
                recent_window=self.cfg.recent_messages_window,
            )
        )

        # 3. Single generation under the scene-play prompt, with the same
        #    [SEARCH] loop as play mode (semantic + reasoning tiers, scrubbed).
        search_notes: list[str] = []
        searches: list[dict[str, str]] = []
        while True:
            messages = modes._build_game_messages(
                modes.SYSTEM_PROMPT_SCENE_PLAY, before_path, ctx, question,
                search_results="\n\n".join(search_notes) or None,
            )
            over_budget = len(searches) >= self.cfg.memory_search_max_calls
            raw = self.model.generate(
                messages,
                max_new_tokens=self.cfg.gemma_max_new_tokens,
                stop_strings=game_io.MOVE_STOP_STRINGS,
                stop_regex=None if over_budget else modes.SEARCH_TOOL_PATTERN,
            )
            kind, payload, text = modes.classify_move_or_search(raw)
            if kind != "search" or over_budget:
                break
            results = self._run(
                mem.search_memory(
                    self.client, payload, tiers=("semantic", "reasoning"),
                    top_k=self.cfg.memory_search_top_k, scrub=True,
                )
            )
            search_notes.append(modes.format_search_note(payload, results))
            searches.append({"query": payload, "results": results, "thought": text})
            if len(searches) >= self.cfg.memory_search_max_calls:
                search_notes.append(modes.SEARCH_BUDGET_NOTE)
            logger.info("player: [SEARCH %s]", payload)
        action = game_io.parse_action(raw) if kind == "move" else None

        # 4. Persist the scene: user question + player reply, both anchored to
        #    the 'before' frame. NO 'after' snapshot yet -- the move has not
        #    been applied.
        user_msg = self._run(
            mem.add_user_question(
                self.client, self.session_id, question,
                snapshot_id=snapshot_before_id,
            )
        )
        self._run(
            image_store.link_snapshot_to_message(
                self.client, user_msg.id, snapshot_before_id, role="before"
            )
        )
        assistant_msg = self._run(
            mem.add_assistant_message(
                self.client, self.session_id, raw,
                kind="move" if action else "answer",
            )
        )
        self._run(
            image_store.link_snapshot_to_message(
                self.client, assistant_msg.id, snapshot_before_id, role="before"
            )
        )

        # 5. One turn trace for the whole round; searches recorded now, the
        #    player step (with the move's actual outcome) at propagation time,
        #    completion at the end of the analyst phase.
        trace = self._run(
            mem.start_turn_trace(
                self.client, self.session_id, task=question,
                triggered_by_message_id=user_msg.id,
            )
        )
        for s in searches:
            self._run(
                modes.record_search_tool_call(
                    self.client, trace, s["thought"], s["query"], s["results"]
                )
            )

        self._pending = {
            "question": question,
            "raw": raw,
            "action": action,
            "before_path": before_path,
            "settings_json": before_props.get("settings_json"),
            "assistant_msg_id": str(assistant_msg.id),
            "trace": trace,
            "n_analyses": 0,
        }
        self.phase = "analyst"
        # A bare move word (e.g. 'ANTICLOCK' without brackets) is never
        # applied, but it is a format fumble worth surfacing loudly rather
        # than mislabeling the reply as a prose answer.
        bare_move = game_io.find_bare_move(raw) if action is None else None
        logger.info(
            "player replied (action=%s%s); awaiting analysis.",
            action or "none",
            f", bare move word {bare_move!r}" if bare_move else "",
        )
        return {
            "session_id": self.session_id,
            "question": question,
            "raw": raw,
            "action": action,
            "bare_move": bare_move,
            "before_path": before_path,
            "searches": searches,
            "phase": self.phase,
        }

    # ---------------------------------------------------------------- analyst
    def ask_analyst(
        self,
        question: str | None = None,
        on_step: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Analyst phase: one privileged review exchange about the player's
        latest reply. Repeatable -- the round stays open (``phase`` stays
        "analyst") until :meth:`end_round` propagates the pending move.

        On the FIRST call of a round an empty ``question`` defaults to
        :data:`DEFAULT_ANALYST_QUESTION`; on follow-up calls an empty
        question is rejected so a stray click cannot burn a generation.
        ``on_step`` (if given) fires after every analyst generation (search
        calls included).
        """
        if self.phase != "analyst" or self._pending is None:
            raise ValueError("No player reply awaiting analysis; ask the player first.")
        pending = self._pending
        question = (question or "").strip()
        if not question:
            if pending["n_analyses"] > 0:
                raise ValueError(
                    "Empty follow-up question; type one, or press 'Back to "
                    "player' to end the round."
                )
            question = DEFAULT_ANALYST_QUESTION
        trace = pending["trace"]

        # The analysis request lives in the same conversation; the "(to
        # analyst)" prefix keeps it distinguishable from scene questions in
        # the player's recency window.
        self._run(
            self.client.short_term.add_message(
                session_id=self.session_id, role="user",
                content="(to analyst) " + question,
                metadata={"kind": "analysis_request"},
            )
        )

        # Analyst [SEARCH] loop: privileged -- all tiers, unscrubbed.
        search_notes: list[str] = []
        replies: list[dict[str, Any]] = []
        n_searches = 0
        while True:
            recent = self._run(
                mem.get_recent_messages(
                    self.client, self.session_id,
                    self.cfg.recent_messages_window, scrub=False,
                )
            )
            messages = modes.build_scene_analyst_messages(
                pending["question"], pending["raw"], pending["action"],
                pending["before_path"], pending["settings_json"],
                recent, question,
                search_results="\n\n".join(search_notes) or None,
            )
            over_budget = n_searches >= self.cfg.memory_search_max_calls
            raw = self.model.generate(
                messages,
                max_new_tokens=self.cfg.gemma_max_new_tokens,
                stop_regex=None if over_budget else modes.SEARCH_TOOL_PATTERN,
            )
            q, text = modes.parse_search_call(raw)
            step_result = {
                "kind": "analyst_generation",
                "generation": len(replies),
                "raw": raw,
                "text": text,
                "search_query": q,
            }
            replies.append(step_result)
            if on_step is not None:
                on_step(step_result)
            if q is None or over_budget:
                break
            results = self._run(
                mem.search_memory(
                    self.client, q, tiers=mem.SEARCH_TIERS,
                    top_k=self.cfg.memory_search_top_k, scrub=False,
                )
            )
            search_notes.append(modes.format_search_note(q, results))
            self._run(
                modes.record_search_tool_call(self.client, trace, text, q, results)
            )
            n_searches += 1
            if n_searches >= self.cfg.memory_search_max_calls:
                search_notes.append(modes.SEARCH_BUDGET_NOTE)
            logger.info("analyst: [SEARCH %s]", q)
        analysis = raw

        # Store the analysis in the SAME conversation so the player's recency
        # window carries it into later rounds; the "(analyst)" prefix is how
        # the player recognizes the privileged reviewer's voice.
        self._run(
            mem.add_assistant_message(
                self.client, self.session_id, "(analyst) " + analysis,
                kind="analysis",
            )
        )
        pending["n_analyses"] += 1

        # Harness-verified error spans: exact substring match against the
        # recorded player reply. Unverified spans are surfaced, never kept.
        wrong_spans = modes.parse_wrong_spans(analysis, pending["raw"])

        logger.info(
            "analyst exchange %d done (%d verified / %d unverified WRONG "
            "spans); round still open.",
            pending["n_analyses"],
            len(wrong_spans["verified"]), len(wrong_spans["unverified"]),
        )
        return {
            "session_id": self.session_id,
            "question": question,
            "analysis": analysis,
            "replies": replies,
            "wrong_spans": wrong_spans,
            "player_raw": pending["raw"],
            "n_analyses": pending["n_analyses"],
            "phase": self.phase,
        }

    def end_round(self) -> dict[str, Any]:
        """End the analyst phase: propagate the pending move, record the
        player step with its actual outcome, complete the round's trace, and
        flip ``phase`` back to "player"."""
        if self.phase != "analyst" or self._pending is None:
            raise ValueError("No round is open; ask the player first.")
        pending = self._pending
        trace = pending["trace"]

        action = pending["action"]
        gold_collected = game_io.apply_action(self.game, action) if action else 0
        self._run(
            mem.add_reasoning_step(
                self.client, trace, thought=pending["raw"], action=action,
                gold_collected=gold_collected,
            )
        )
        after_path = None
        if action:
            snapshot_after_id = image_store.snapshot_id()
            settings_after = game_io.game_to_settings_dict(self.game)
            after_path, _ = self._run(
                image_store.store_snapshot(
                    self.client, self.session_id, snapshot_after_id, self.game,
                    settings_after, cfg=self.cfg, label="after",
                    extra={"action": action, "gold_collected": gold_collected},
                )
            )
            self._run(
                image_store.link_snapshot_to_message(
                    self.client, pending["assistant_msg_id"], snapshot_after_id,
                    role="after",
                )
            )

        gold_remaining = game_io.gold_remaining(self.game)
        n_analyses = pending["n_analyses"]
        outcome = (
            f"scene round: action={action or 'none'}; "
            f"gold_collected={gold_collected}; "
            f"analyst_exchanges={n_analyses}; "
            f"gold_remaining={gold_remaining}"
        )
        self._run(
            mem.complete_turn_trace(
                self.client, trace, outcome=outcome,
                success=(gold_remaining == 0) if action else True,
            )
        )

        self._pending = None
        self.phase = "player"
        logger.info(
            "round ended after %d analyst exchange(s); move %s propagated "
            "(gold_collected=%d).",
            n_analyses, action or "none", gold_collected,
        )
        return {
            "session_id": self.session_id,
            "action": action,
            "gold_collected": gold_collected,
            "gold_remaining": gold_remaining,
            "after_path": after_path,
            "frame_path": self.current_frame_path(),
            "n_analyses": n_analyses,
            "phase": self.phase,
        }
