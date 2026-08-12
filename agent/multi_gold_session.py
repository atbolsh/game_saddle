"""Multi-gold / no-gold / no-end self-eval session.

A subclass of :class:`InteractiveSelfEvalSession` that leaves the existing
one-gold critical path untouched. New prompt blocks, a multi-gold factory,
a boundary-openings oracle on the analyst's settings JSON, and an
``[END_GAME]`` token that the base parser never sees.
"""

from __future__ import annotations

import logging
from typing import Any

from . import game_io
from . import memory as mem
from . import modes
from .self_eval_session import InteractiveSelfEvalSession

logger = logging.getLogger(__name__)


class MultiGoldSelfEvalSession(InteractiveSelfEvalSession):
    """Player/analyst self-eval over a room that may hold 0–3 golds.

    ``n_gold`` None = random 0..3. ``opening`` is ``"require"`` / ``"forbid"``
    / ``"any"``. ``end_on_clear`` defaults False: eating the last gold does
    not end the session -- the player walks out an opening or emits
    ``[END_GAME]``.
    """

    PLAYER_SYSTEM_PROMPT = modes.SYSTEM_PROMPT_SCENE_PLAY_MULTI
    ANALYST_SYSTEM_PROMPT = modes.SYSTEM_PROMPT_SCENE_ANALYST_MULTI
    PLAYER_STOP_STRINGS = game_io.MOVE_STOP_STRINGS + [modes._TOK_END_GAME]
    END_ON_CLEAR = False

    def __init__(
        self,
        *args: Any,
        n_gold: int | None = None,
        opening: str = "require",
        end_on_clear: bool = False,
        **kwargs: Any,
    ):
        # Set before super().__init__: InteractiveSession.restart() runs
        # during construction and dispatches to _new_game.
        self.n_gold = n_gold
        self.opening = opening
        self.END_ON_CLEAR = end_on_clear
        self.session_state: str = "active"
        super().__init__(*args, **kwargs)

    def _new_game(self) -> Any:
        return game_io.new_multi_gold_game(
            gameSize=self.cfg.game_size,
            n_gold=self.n_gold,
            opening=self.opening,
        )

    def _analyst_settings_dict(self) -> dict[str, Any]:
        d = super()._analyst_settings_dict()
        d["openings"] = game_io.boundary_openings(d)
        return d

    def _parse_player_action(self, raw: str, kind: str) -> str | None:
        if modes._TOK_END_GAME in raw:
            return "END_GAME"
        return super()._parse_player_action(raw, kind)

    def restart(self) -> dict[str, Any]:
        self.session_state = "active"
        return super().restart()

    def reset_game(self, record: bool = True) -> dict[str, Any]:
        self.session_state = "active"
        return super().reset_game(record=record)

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

    def force_end(self, reason: str = "user") -> dict[str, Any]:
        """User-initiated end (notebook button). Not a player move -- no
        analyst grading. Abandons an open round if one is in flight."""
        if self._pending is not None:
            pending = self._pending
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
        self.phase = "player"
        self.session_state = "ended_by_user"
        logger.info("session force-ended (reason=%s).", reason)
        return {
            "session_id": self.session_id,
            "session_state": self.session_state,
            "gold_remaining": game_io.gold_remaining(self.game),
            "frame_path": self.current_frame_path(),
            "phase": self.phase,
        }
