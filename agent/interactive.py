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
from .model import get_model

logger = logging.getLogger(__name__)


class InteractiveSession:
    """A persistent, single-game interactive mode-1 session.

    Construct it once per notebook (it connects NAMS and loads Gemma), then
    call :meth:`ask` / :meth:`restart` from your UI callbacks and
    :meth:`close` when done.
    """

    def __init__(self, cfg: AgentConfig | None = None, load_model: bool = True):
        self.cfg = cfg or CONFIG

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

    def _step(self, question: str, step: int, triggered_by_id: str | None) -> dict[str, Any]:
        """Run one step of a turn: snapshot -> context -> model -> apply move
        -> persist. Returns a per-step result dict."""
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
        ctx = self._run(
            mem.get_game_context(self.client, self.session_id, query=question)
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
            modes.SYSTEM_PROMPT_GAME, before_path, ctx, prompt_text
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

        # 5. Persist the step (user message on the first step only).
        turn = self._run(
            modes._record_turn(
                self.client, self.session_id, self.cfg, self.game, question, raw,
                action, gold_collected, snapshot_before_id, before_path,
                include_user_message=(step == 0),
                triggered_by_message_id=triggered_by_id,
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
        triggered_by_id: str | None = None

        for i in range(max_steps):
            step_result = self._step(question, i, triggered_by_id)
            if i == 0:
                triggered_by_id = step_result["user_msg_id"]
            steps.append(step_result)
            if on_step is not None:
                on_step(step_result)

            # Stop conditions: the model emitted no move token (it ended its
            # turn / answered), or the gold has been collected.
            if step_result["action"] is None:
                break
            if step_result["gold_remaining"] == 0:
                break

        return {
            "session_id": self.session_id,
            "question": question,
            "steps": steps,
            "num_steps": len(steps),
            "gold_remaining": game_io.gold_remaining(self.game),
            "solved": game_io.gold_remaining(self.game) == 0,
        }

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
