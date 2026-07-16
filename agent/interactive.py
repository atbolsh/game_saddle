"""Stateful, notebook-friendly interactive game session (mode 1).

:class:`InteractiveSession` wraps one **persistent** game and one conversation
thread so you can ask the agent question after question and watch its
reactions, exactly as it sees them. It is the interactive counterpart to
:func:`agent.modes.mode_game`, which instead spins up a fresh game on every
call and takes a single turn.

One :meth:`ask` is a *multi-move turn*: making a move does not end the turn.
Each step:

  1. Render the *current* game frame to disk + a ``GameSnapshot`` node.
  2. Retrieve NAMS context with the Settings dict stripped out -- the mode-1
     privacy invariant is preserved: the model never sees exact coordinates.
  3. Build the multimodal prompt (system + memory context + image + question)
     and call Gemma 4 E4B (with a high ``max_new_tokens`` ceiling).
  4. Generation is stopped early the instant a move token (``[FORWARD]`` etc.)
     appears; we apply that move, re-render, and loop back to step 1 feeding
     the *updated* view.
  5. Persist each step to NAMS (user message on the first step only, then
     assistant message + reasoning trace + before/after snapshots).

The turn ends when the model replies without emitting a move token (Gemma's
native end-of-turn), collects the gold, or a safety step cap is hit. An optional
``on_step`` callback fires after every step so a UI can show the board and the
agent's move live as the turn unfolds.

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
from .model import get_model

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
        self.model = get_model(self.cfg) if load_model else None

        self.game: Any = None
        self.session_id: str = ""
        self.restart()

    # ------------------------------------------------------------------ bridge
    def _run(self, coro: Any) -> Any:
        """Run a coroutine on the background loop and block for its result."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    # ------------------------------------------------------------------ public
    def restart(self) -> dict[str, Any]:
        """Re-initialize the env (new bare game) and start a new conversation
        thread (new ``session_id``). Returns the new session id + the path to
        the freshly rendered starting frame."""
        self.game = game_io.new_bare_game(gameSize=self.cfg.game_size)
        self.session_id = mem.new_session_id()
        logger.info("Interactive session restarted: session_id=%s", self.session_id)
        return {
            "session_id": self.session_id,
            "frame_path": self.current_frame_path(),
            "gold_remaining": game_io.gold_remaining(self.game),
        }

    def current_frame_path(self) -> str:
        """Render the current game frame to disk (no DB write) and return its
        absolute path. Handy for previewing the board between turns."""
        rel = Path(self.cfg.image_dir) / self.session_id / "current.png"
        abs_path = rel.resolve()
        game_io.render_frame_png(self.game, abs_path)
        return str(abs_path)

    def _step(
        self,
        question: str,
        step: int,
        recent_actions: list[str] | None = None,
        reflection: str | None = None,
    ) -> dict[str, Any]:
        """Run one step of a turn: snapshot -> context -> model -> apply move
        -> persist (messages + snapshots). The turn-level reasoning trace is
        managed by :meth:`ask`, not here. Returns a per-step result dict."""
        # 1. Snapshot the current ('before') frame -> disk + GameSnapshot node.
        snapshot_before_id = image_store.snapshot_id()
        settings_before = game_io.game_to_settings_dict(self.game)
        before_path, _ = self._run(
            image_store.store_snapshot(
                self.client, self.session_id, snapshot_before_id, self.game,
                settings_before, cfg=self.cfg, label="before",
            )
        )

        # 2. Memory context (settings stripped -- mode-1 privacy invariant):
        #    a guaranteed recency window of the latest messages + the general
        #    semantic search across all memory tiers. The search is queried with
        #    the *situational* state (recent moves + gold progress), not the
        #    static instruction, so trace/tip recall is relevant to this move.
        query = modes._retrieval_query(
            question, step, recent_actions, game_io.gold_remaining(self.game)
        )
        ctx = self._run(
            mem.get_game_context(
                self.client, self.session_id, query=query,
                recent_window=self.cfg.recent_messages_window,
            )
        )

        # 3. Build the multimodal prompt and call the model (sync, on this
        #    thread -- the heavy GPU work does not need the async loop). On
        #    continuation steps, nudge the model with the updated-view protocol.
        if step == 0:
            prompt_text = question
        else:
            prompt_text = (
                f"{question}\n\n(Continuing your turn -- you have made {step} "
                f"move(s). This is the updated screen after your last move. "
                f"Emit your next move token, or finish your reply without a "
                f"move token to end your turn.)"
            )
        messages = modes._build_game_messages(
            modes.SYSTEM_PROMPT_GAME, before_path, ctx, prompt_text,
            reflection=reflection,
        )
        # Stop generation the instant a move token appears; apply it, then the
        # next step re-generates on the freshly rendered frame. A reply with no
        # move token means the model ended its turn (Gemma's native end-of-turn).
        raw = self.model.generate(
            messages,
            max_new_tokens=self.cfg.gemma_max_new_tokens,
            stop_strings=game_io.MOVE_STOP_STRINGS,
        )

        # 4. Parse the move (if any) and apply it to the persistent game.
        action = game_io.parse_action(raw)
        gold_collected = game_io.apply_action(self.game, action) if action else 0

        # 5. Persist the step's messages + snapshots (user message on step 0
        #    only). The reasoning trace spans the whole turn and is handled in
        #    :meth:`ask`.
        turn = self._run(
            modes._record_step(
                self.client, self.session_id, self.cfg, self.game, question, raw,
                action, gold_collected, snapshot_before_id, before_path,
                include_user_message=(step == 0),
            )
        )

        return {
            "session_id": self.session_id,
            "step": step,
            "question": question if step == 0 else None,
            "raw": raw,
            "action": action,
            "gold_collected": gold_collected,
            "gold_remaining": game_io.gold_remaining(self.game),
            "before_path": turn["snapshot_before_path"],
            "after_path": turn["snapshot_after_path"],
            "user_msg_id": turn["user_msg_id"],
        }

    def _reflect(
        self, question: str, trace: Any, steps: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Run one reflection pause (generative-agents style, arXiv:2304.03442):
        render the CURRENT frame, ask the model to re-examine the situation
        (was I ever facing the gold? am I facing it now? am I still turning the
        right way?) with NO move this step, and persist the reflection to memory
        (assistant message kind='reflection' + a reasoning step on the turn
        trace). Returns a result dict with ``kind='reflection'`` suitable for
        the same ``on_step`` callback as ordinary move steps."""
        frame_path = self.current_frame_path()
        actions = [s["action"] for s in steps if s.get("action")]
        messages = modes.build_reflection_messages(frame_path, question, actions)
        raw = self.model.generate(
            messages, max_new_tokens=self.cfg.gemma_max_new_tokens
        )
        reflection = raw.strip()
        self._run(
            modes.persist_reflection(self.client, self.session_id, trace, reflection)
        )
        logger.info("reflection after %d move(s): %s", len(actions), reflection)
        return {
            "kind": "reflection",
            "session_id": self.session_id,
            "step": len(steps),
            "raw": raw,
            "reflection": reflection,
            "action": None,
            "frame_path": frame_path,
        }

    def ask(
        self,
        question: str,
        on_step: Callable[[dict[str, Any]], None] | None = None,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Take one multi-move interactive turn against the persistent game.

        Making a move does NOT end the turn: after each move the board is
        re-rendered and fed back to the model so it can keep going (e.g.
        [CLOCK], [CLOCK], [FORWARD], ...). The turn ends when the model replies
        without emitting a move token (its native end-of-turn / a plain answer),
        the gold is collected, or ``max_steps`` (default ``cfg.max_solve_steps``)
        is hit.

        ``on_step`` (if given) is called with each step's result dict as it
        happens -- wire it to your UI to watch the board update live.

        Returns a summary dict: ``steps`` (list of per-step dicts), ``num_steps``,
        ``gold_remaining`` and ``solved``.
        """
        max_steps = max_steps or self.cfg.max_solve_steps
        steps: list[dict[str, Any]] = []
        # One reasoning trace for the whole turn (task = the user's question);
        # each step becomes a reasoning step within it. Opened after step 0's
        # user message exists and completed once the turn ends.
        trace: Any = None
        # Reflection bookkeeping (generative-agents style): every applied move
        # accrues cfg.reflection_points_per_move importance points; at
        # cfg.reflection_threshold the agent pauses to reflect (default: every
        # 30 moves = one 180-degree sweep of rotations) and the total resets.
        # The latest reflection is injected into every subsequent prompt.
        reflection_points = 0
        last_reflection: str | None = None

        try:
            for i in range(max_steps):
                step_result = self._step(
                    question, i, modes._recent_actions(steps),
                    reflection=last_reflection,
                )
                steps.append(step_result)

                # Open the turn trace once the first user message exists, then
                # record this generation (and every later one) as a step.
                if i == 0:
                    trace = self._run(
                        mem.start_turn_trace(
                            self.client, self.session_id, task=question,
                            triggered_by_message_id=step_result["user_msg_id"],
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

                # Stop conditions: the model emitted no move token (it ended its
                # turn / answered), or the gold has been collected.
                if step_result["action"] is None:
                    break
                if step_result["gold_remaining"] == 0:
                    break

                # Reflection pause: enough importance points have accrued.
                reflection_points += self.cfg.reflection_points_per_move
                if reflection_points >= self.cfg.reflection_threshold:
                    reflection_result = self._reflect(question, trace, steps)
                    last_reflection = reflection_result["reflection"]
                    reflection_points = 0
                    if on_step is not None:
                        on_step(reflection_result)
        finally:
            outcome, success = modes._turn_trace_outcome(
                steps, game_io.gold_remaining(self.game)
            )
            self._run(
                mem.complete_turn_trace(self.client, trace, outcome=outcome, success=success)
            )

        return {
            "session_id": self.session_id,
            "question": question,
            "steps": steps,
            "num_steps": len(steps),
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
        reasoning traces/steps) and keep ONLY the seeded semantic model
        (``Entity`` + ``Preference`` nodes and their relationships).

        This restores the graph to the "semantic seeding only" state -- the
        status quo ante of a fresh box right after ``seed`` + ``link``. Use it
        to clean up after a failed/experimental conversation. Returns a dict of
        ``{label: count}`` deleted. Note: it clears EVERY conversation, not just
        the current one.

        This does NOT touch on-disk images or start a new conversation; call
        :meth:`restart` afterwards for a fresh thread + board.
        """
        return self._run(self._reset_memory_to_seed())

    async def _reset_memory_to_seed(self) -> dict[str, int]:
        rows = await self.client.graph.execute_write(
            "MATCH (n) WHERE NOT (n:Entity OR n:Preference) "
            "WITH n, labels(n) AS l DETACH DELETE n RETURN l",
            {},
        )
        counts: Counter = Counter()
        for r in rows:
            labels = dict(r).get("l") or []
            counts["+".join(labels) or "(none)"] += 1
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
