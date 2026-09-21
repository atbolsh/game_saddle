"""Multi-gold / no-gold / no-end room sessions.

:class:`MultiGoldRoomMixin` holds the shared factory (``n_gold``,
``opening``, ``_new_game``), ``session_state``, ``force_end``, and the
``[END_GAME]`` freeze hook. Two concrete classes use it:

* :class:`MultiGoldSelfEvalSession` -- player/analyst (notebook + datagen).
* :class:`MultiGoldPlaySession` -- player only (``notebooks/play.ipynb``).
"""

from __future__ import annotations

import logging
from typing import Any

from . import game_io
from . import memory as mem
from .interactive import InteractiveSession
from .self_eval_session import InteractiveSelfEvalSession

logger = logging.getLogger(__name__)


class MultiGoldRoomMixin:
    """Shared multi-gold room mechanics. Bind fields before
    ``InteractiveSession.__init__`` (that constructor calls ``restart``
    which dispatches to :meth:`_new_game`)."""

    END_ON_CLEAR = False

    def _bind_room(
        self,
        n_gold: int | None,
        opening: str,
        end_on_clear: bool = False,
    ) -> None:
        self.n_gold = n_gold
        self.opening = opening
        self.END_ON_CLEAR = end_on_clear
        self.session_state: str = "active"

    def _new_game(self) -> Any:
        return game_io.new_multi_gold_game(
            gameSize=self.cfg.game_size,
            n_gold=self.n_gold,
            opening=self.opening,
        )

    def _on_player_end_game(self) -> None:
        self.session_state = "ended_by_player"

    def restart(self) -> dict[str, Any]:
        self.session_state = "active"
        return super().restart()

    def reset_game(self, record: bool = True) -> dict[str, Any]:
        self.session_state = "active"
        return super().reset_game(record=record)

    def force_end(self, reason: str = "user") -> dict[str, Any]:
        """User-initiated end (notebook button). Not a player move.

        Abandons an open self-eval round if one is in flight. Play has no
        pending round -- the ``_pending`` cleanup is a no-op there.
        """
        pending = getattr(self, "_pending", None)
        if pending is not None:
            outcome = (
                f"scene round abandoned: ended_by_{reason}; "
                f"pending_action={pending.get('action')}"
            )
            self._run(
                mem.complete_turn_trace(
                    self.client, pending["trace"], outcome=outcome,
                    success=False,
                )
            )
            self._pending = None
        if hasattr(self, "phase"):
            self.phase = "player"
        self.session_state = "ended_by_user"
        logger.info("session force-ended (reason=%s).", reason)
        return {
            "session_id": self.session_id,
            "session_state": self.session_state,
            "gold_remaining": game_io.gold_remaining(self.game),
            "frame_path": self.current_frame_path(),
            "phase": getattr(self, "phase", "player"),
        }


class MultiGoldSelfEvalSession(MultiGoldRoomMixin, InteractiveSelfEvalSession):
    """Player/analyst self-eval over a room that may hold 0–3 golds.

    ``n_gold`` None = random 0..3. ``opening`` is ``"require"`` / ``"forbid"``
    / ``"any"``. ``end_on_clear`` defaults False: eating the last gold does
    not end the session -- the player walks out an opening or emits
    ``[END_GAME]``.
    """

    END_ON_CLEAR = False

    def __init__(
        self,
        *args: Any,
        n_gold: int | None = None,
        opening: str = "require",
        end_on_clear: bool = False,
        **kwargs: Any,
    ):
        self._bind_room(n_gold, opening, end_on_clear)
        log_label = kwargs.pop("log_label", None)
        super().__init__(*args, log_label=log_label or "multi_gold_eval", **kwargs)

    def end_round(self) -> dict[str, Any]:
        if self.phase != "analyst" or self._pending is None:
            raise ValueError("No round is open; ask the player first.")
        pending = self._pending
        action = pending["action"]
        if action != "END_GAME":
            result = super().end_round()
            if (self.END_ON_CLEAR
                    and result["gold_remaining"] == 0
                    and result.get("action")):
                self.session_state = "cleared"
            return result

        # [END_GAME]: do NOT apply a board action; the analyst has already
        # graded the reply. Record the final thought (the base end_round
        # records a reasoning step for every round, including action-None
        # ones -- END_GAME rounds must not be the lone exception), then
        # complete the trace and freeze the session.
        trace = pending["trace"]
        n_analyses = pending["n_analyses"]
        gold_remaining = game_io.gold_remaining(self.game)
        self._run(
            mem.add_reasoning_step(
                self.client, trace, thought=pending["raw"],
                action="END_GAME", gold_collected=0,
            )
        )
        outcome = (
            f"scene round: action=END_GAME; gold_collected=0; "
            f"analyst_exchanges={n_analyses}; gold_remaining={gold_remaining}"
        )
        self._run(
            mem.complete_turn_trace(
                self.client, trace, outcome=outcome, success=True,
            )
        )
        self.round_no += 1
        self._pending = None
        self.phase = "player"
        self.session_state = "ended_by_player"
        logger.info(
            "round ended after %d analyst exchange(s); [END_GAME] -- "
            "session ended by player.",
            n_analyses,
        )
        return {
            "session_id": self.session_id,
            "action": "END_GAME",
            "gold_collected": 0,
            "gold_remaining": gold_remaining,
            "after_path": None,
            "frame_path": self.current_frame_path(),
            "n_analyses": n_analyses,
            "phase": self.phase,
            "session_state": self.session_state,
        }


class MultiGoldPlaySession(MultiGoldRoomMixin, InteractiveSession):
    """Mode-1 play over a multi-gold room. One ``ask`` applies the move
    immediately. ``[END_GAME]`` freezes ``session_state``."""

    def __init__(
        self,
        *args: Any,
        n_gold: int | None = None,
        opening: str = "require",
        end_on_clear: bool = False,
        **kwargs: Any,
    ):
        self._bind_room(n_gold, opening, end_on_clear)
        log_label = kwargs.pop("log_label", None)
        super().__init__(*args, log_label=log_label or "play", **kwargs)
