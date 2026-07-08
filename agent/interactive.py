"""Stateful, notebook-friendly interactive game session (mode 1).

:class:`InteractiveSession` wraps one **persistent** game and one conversation
thread so you can ask the agent question after question and watch its
reactions, exactly as it sees them. It is the interactive counterpart to
:func:`agent.modes.mode_game`, which instead spins up a fresh game on every
call and takes a single turn.

Per turn (:meth:`ask`):

  1. Render the *current* game frame to disk + a ``GameSnapshot`` node.
  2. Retrieve NAMS context with the Settings dict stripped out -- the mode-1
     privacy invariant is preserved: the model never sees exact coordinates.
  3. Build the multimodal prompt (system + memory context + image + question)
     and call Gemma 4 E4B.
  4. If the reply contains a move keyword, apply it to the persistent game.
  5. Persist the whole turn to NAMS (user/assistant messages, reasoning trace,
     before/after snapshots) via :func:`agent.modes._record_turn`.

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
from pathlib import Path
from typing import Any

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

    def ask(self, question: str) -> dict[str, Any]:
        """Take one interactive turn against the persistent game.

        Returns a dict with the raw model reply, the parsed move (if any),
        gold counters, and the absolute paths of the ``before`` frame (the
        exact image the model saw) and the ``after`` frame (``None`` if the
        turn was a pure Q&A with no move).
        """
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
        #    thread -- the heavy GPU work does not need the async loop).
        messages = modes._build_game_messages(
            modes.SYSTEM_PROMPT_GAME, before_path, ctx, question
        )
        raw = self.model.generate(messages)

        # 4. Parse a move and apply it to the persistent game.
        action = game_io.parse_action(raw)
        gold_collected = game_io.apply_action(self.game, action) if action else 0

        # 5. Persist the whole turn (messages + trace + before/after snapshots).
        turn = self._run(
            modes._record_turn(
                self.client, self.session_id, self.cfg, self.game, question, raw,
                action, gold_collected, snapshot_before_id, before_path,
            )
        )

        return {
            "session_id": self.session_id,
            "question": question,
            "raw": raw,
            "action": action,
            "gold_collected": gold_collected,
            "gold_remaining": game_io.gold_remaining(self.game),
            "before_path": turn["snapshot_before_path"],
            "after_path": turn["snapshot_after_path"],
        }

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
