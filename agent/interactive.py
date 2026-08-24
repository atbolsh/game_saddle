"""Stateful, notebook-friendly interactive game session (mode 1).

:class:`InteractiveSession` wraps one **persistent** game and one conversation
thread so you can ask the agent question after question and watch its
reactions, exactly as it sees them. It is the interactive counterpart to
:func:`agent.modes.mode_game`, which instead spins up a fresh game on every
call.

One :meth:`ask` is **one generation**: the agent sees the current live frame
plus its (settings-stripped) memory context, may run ``[SEARCH]`` loops, and
emits at most one move token. Generation stops at the token; CLOCK /
ANTICLOCK / FORWARD are applied immediately. ``[END_GAME]`` is a stop
string so generation halts there, but ``game_io.parse_action`` returns
None for it -- no board change, session continues.

:meth:`restart` re-initializes the env (a brand new bare game) and starts a
new conversation thread (a fresh ``session_id``), reusing the already-loaded
model and the already-connected memory client.

**Async bridge.** NAMS is async and the Neo4j async driver is bound to the
event loop it was created on, but ipywidgets button callbacks are synchronous.
So we run a dedicated asyncio loop in a background thread, create/connect the
``MemoryClient`` there, and marshal every coroutine onto it via
``run_coroutine_threadsafe``. All public methods here are therefore plain sync
calls, safe to wire straight to a button's ``on_click``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .config import AgentConfig, CONFIG
from . import game_io
from . import image_store
from . import memory as mem
from . import modes
from . import run_logging
from .model import get_model, switch_session_model

logger = logging.getLogger(__name__)


class InteractiveSession:
    """A persistent, single-game interactive mode-1 session.

    Construct it once per notebook (it connects NAMS and loads Gemma), then
    call :meth:`ask` / :meth:`restart` from your UI callbacks and
    :meth:`close` when done.
    """

    def __init__(
        self,
        cfg: AgentConfig | None = None,
        load_model: bool = True,
        enable_logging: bool = True,
        log_label: str | None = None,
    ):
        self.cfg = cfg or CONFIG

        # Per-run logging (LLM calls + DB retrievals) is on by default; it lands
        # in a fresh logs/<label>_<timestamp>/ directory and captures every
        # generate call and memory retrieval for this session. Pass
        # enable_logging=False to turn it off.
        self.logger = (
            run_logging.new_run_logger(label=log_label or "play")
            if enable_logging
            else None
        )

        # Background event loop for all async NAMS calls.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="nams-loop", daemon=True
        )
        self._thread.start()

        self.client = self._run(mem.connect(self.cfg))
        self._system_prompts = self._run(modes.load_scene_prompts(self.client))
        self.model = get_model(self.cfg) if load_model else None

        self.game: Any = None
        self.session_id: str = ""
        self.restart()

    def _new_game(self) -> Any:
        """Construct the board for this session. Subclasses override to
        change the factory.

        Legacy training/play board -- sealed one-gold room; eating gold
        still wins; openings / ``[END_GAME]`` exist in the prompt for the
        multi-gold path but this factory does not spawn them.
        """
        return game_io.new_bare_game(gameSize=self.cfg.game_size)

    # ------------------------------------------------------------------ bridge
    def _run(self, coro: Any) -> Any:
        """Run a coroutine on the background loop and block for its result."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    # ------------------------------------------------------------------ public
    def restart(self) -> dict[str, Any]:
        """Re-initialize the env (new bare game) and start a new conversation
        thread (new ``session_id``). Returns the new session id + the path to
        the freshly rendered starting frame."""
        self.game = self._new_game()
        self.session_id = mem.new_session_id()
        self.round_no = 0
        logger.info("Interactive session restarted: session_id=%s", self.session_id)
        return {
            "session_id": self.session_id,
            "frame_path": self.current_frame_path(),
            "gold_remaining": game_io.gold_remaining(self.game),
        }

    def switch_model(
        self,
        key: str,
        purge_others: bool = False,
        checkpoint: str | None = None,
    ) -> dict[str, Any]:
        """Switch to registry model ``key`` (see ``agent.model.MODEL_REGISTRY``),
        optionally with a trained adapter ``checkpoint`` from
        ``weights/<key>/`` (None = bare HuggingFace weights).

        With ``purge_others=True`` ("save only one set of weights at a
        time"), the conversation is restarted first and every other registry
        model's cached weights are deleted from disk before the new ones are
        downloaded. Without it, the conversation continues under the new
        model."""
        return switch_session_model(self, key, purge_others, checkpoint)

    def current_frame_path(self) -> str:
        """Render the current game frame to disk (no DB write) and return its
        absolute path. Handy for previewing the board between turns."""
        rel = Path(self.cfg.image_dir) / self.session_id / "current.png"
        abs_path = rel.resolve()
        game_io.render_frame_png(self.game, abs_path)
        return str(abs_path)

    def ask(
        self,
        question: str,
        on_step: Callable[[dict[str, Any]], None] | None = None,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Take one generation against the persistent game.

        ``max_steps`` is accepted for call-site compatibility and ignored:
        one ``ask`` is always one generation (plus any ``[SEARCH]`` loops).
        ``on_step`` (if given) is called with the generation's result dict.

        ``[END_GAME]`` is a stop string so generation halts at the token;
        ``parse_action`` returns None for it, so nothing is applied.
        """
        del max_steps  # one generation per ask; kept so callers need not change
        # 1. Snapshot the current ('before') frame -> disk + GameSnapshot node.
        snapshot_before_id = image_store.snapshot_id()
        settings_before = game_io.game_to_settings_dict(self.game)
        before_path, _ = self._run(
            image_store.store_snapshot(
                self.client, self.session_id, snapshot_before_id, self.game,
                settings_before, cfg=self.cfg, label="before",
            )
        )

        # 2. Memory context (settings stripped -- mode-1 privacy invariant).
        query = modes._retrieval_query(
            question, 0, None, game_io.gold_remaining(self.game)
        )
        ctx = self._run(
            mem.get_game_context(
                self.client, self.session_id, query=query,
                recent_window=self.cfg.recent_messages_window,
                exclude_analyst=True,
            )
        )
        notes = self._run(
            mem.get_session_notes(self.client, self.session_id)
        )
        notepad = mem.format_notepad(notes)

        # 3. Single generation under the unified scene-play prompt.
        #    [END_GAME] stops generation; parse_action yields None for it.
        play_stop = game_io.MOVE_STOP_STRINGS + [modes._TOK_END_GAME]
        search_notes: list[str] = []
        searches: list[dict[str, str]] = []
        while True:
            messages = modes._build_game_messages(
                self._system_prompts["scene_play"], before_path, ctx, question,
                search_results="\n\n".join(search_notes) or None,
                notepad=notepad,
            )
            over_budget = len(searches) >= self.cfg.memory_search_max_calls
            raw = self.model.generate(
                messages,
                max_new_tokens=self.cfg.max_new_tokens,
                stop_strings=play_stop,
                stop_regex=None if over_budget else modes.SEARCH_TOOL_PATTERN,
            )
            kind, payload, text = modes.classify_move_or_search(raw)
            if kind != "search" or over_budget:
                break
            results = self._run(
                mem.search_memory(
                    self.client, payload, tiers=("semantic", "reasoning"),
                    top_k=self.cfg.memory_search_top_k, scrub=True,
                    exclude_analyst=True,
                )
            )
            search_notes.append(modes.format_search_note(payload, results))
            searches.append({"query": payload, "results": results, "thought": text})
            if len(searches) >= self.cfg.memory_search_max_calls:
                search_notes.append(modes.SEARCH_BUDGET_NOTE)
            logger.info("[SEARCH %s]", payload)

        new_notes = game_io.parse_remember_notes(raw)
        for k, v in new_notes:
            self._run(
                mem.set_session_note(
                    self.client, self.session_id, k, v, self.round_no,
                )
            )
        self.round_no += 1
        action = game_io.parse_action(raw) if kind == "move" else None
        gold_collected = game_io.apply_action(self.game, action) if action else 0

        turn = self._run(
            modes._record_step(
                self.client, self.session_id, self.cfg, self.game, question, raw,
                action, gold_collected, snapshot_before_id, before_path,
                include_user_message=True,
            )
        )

        step_result = {
            "session_id": self.session_id,
            "step": 0,
            "question": question,
            "raw": raw,
            "action": action,
            "bare_move": game_io.find_bare_move(raw) if action is None else None,
            "gold_collected": gold_collected,
            "gold_remaining": game_io.gold_remaining(self.game),
            "before_path": turn["snapshot_before_path"],
            "after_path": turn["snapshot_after_path"],
            "user_msg_id": turn["user_msg_id"],
            "searches": searches,
            "notes": new_notes,
        }

        trace: Any = None
        try:
            trace = self._run(
                mem.start_turn_trace(
                    self.client, self.session_id, task=question,
                    triggered_by_message_id=step_result["user_msg_id"],
                )
            )
            for s in searches:
                self._run(
                    modes.record_search_tool_call(
                        self.client, trace, s["thought"], s["query"],
                        s["results"],
                    )
                )
            self._run(
                mem.add_reasoning_step(
                    self.client, trace, thought=step_result["raw"],
                    action=step_result["action"],
                    gold_collected=step_result["gold_collected"],
                )
            )
            if on_step is not None:
                on_step(step_result)
        finally:
            outcome, success = modes._turn_trace_outcome(
                [step_result], game_io.gold_remaining(self.game)
            )
            self._run(
                mem.complete_turn_trace(
                    self.client, trace, outcome=outcome, success=success
                )
            )

        return {
            "session_id": self.session_id,
            "question": question,
            "steps": [step_result],
            "num_steps": 1,
            "gold_remaining": game_io.gold_remaining(self.game),
            "solved": game_io.gold_remaining(self.game) == 0,
            "trace_id": str(trace.id) if trace else None,
            "success": success,
        }

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

    def reset_memory_to_seed(self) -> dict[str, int]:
        """Wipe all episodic memory (conversations, messages, game snapshots,
        reasoning traces/steps) and keep ONLY the seeded semantic model:
        the ``_SEMANTIC_MODEL_ENTITIES`` (matched by name -- the same match
        ``add_semantic_relationships`` relies on) plus every ``Preference``
        node (seed prefs AND learned tips).

        EXTRACTED entities are deleted too (2026-08-11): NAMS's entity
        extraction mints ``Entity`` nodes from every stored message, and
        the old blanket ``n:Entity`` exemption let them pile up through
        every reset -- the aug6 11-epoch run accumulated ~33k junk
        entities (vs 5 seed ones) that competed with the seed model in
        semantic retrieval and never got cleared.

        This restores the graph to the "semantic seeding only" state -- the
        status quo ante of a fresh box right after ``seed`` + ``link``. Use it
        to clean up after a failed/experimental conversation. Returns a dict of
        ``{label: count}`` deleted. Note: it clears EVERY conversation, not just
        the current one.

        After the wipe, re-heals ``core_player_*`` / ``core_analyst_*``
        Preference rows from the code seed (:func:`memory.ensure_core_tips`)
        so a graph that never had them (or drifted) still gets the current
        crop of prompt tips.

        This does NOT touch on-disk images or start a new conversation; call
        :meth:`restart` afterwards for a fresh thread + board.
        """
        return self._run(self._reset_memory_to_seed())

    async def _reset_memory_to_seed(self) -> dict[str, int]:
        seed_names = [name for name, _, _ in mem._SEMANTIC_MODEL_ENTITIES]
        rows = await self.client.graph.execute_write(
            "MATCH (n) "
            "WHERE NOT ((n:Entity AND n.name IN $seed_names) OR n:Preference) "
            "WITH n, labels(n) AS l DETACH DELETE n RETURN l",
            {"seed_names": seed_names},
        )
        counts: Counter = Counter()
        for r in rows:
            labels = dict(r).get("l") or []
            counts["+".join(labels) or "(none)"] += 1
        await mem.ensure_core_tips(self.client)
        return dict(counts)

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

    def __enter__(self) -> "InteractiveSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
