"""One S2 game.

Notes live in a dict. The conversation lives in NAMS: a turn loads the
last 7 messages of this session plus up to 2 older ones from semantic
search, when any remain outside those 7. Every turn is written, look
or move. A turn shows S2 the current frame and
the question, and wakes S1 only when the reply ends with ``[MOVE]``.
A reply that ends with ``[END_GAME]``, or an S1 window that walks
out, ends the game. ``new_game`` opens
``weights/s1s2_runs/<stamp>/turns.jsonl`` and every ``turn`` appends
one record, including the S1 action list. ``replay`` rebuilds a move
window from the recorded actions. ``play`` runs the default question
sequence.
"""

from __future__ import annotations

import asyncio
import os
import random
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import numpy as np
from PIL import Image

from agent.config import CONFIG
from agent.game_io import (
    compose_s2_question,
    game_to_settings_dict,
    gold_remaining,
    new_multi_gold_game,
    parse_remember_notes,
    sealed_empty,
)
from agent.memory import (
    add_assistant_message,
    add_user_question,
    connect,
    format_notepad,
    load_played_context,
    new_session_id,
)
from agent.modes import (
    S2_BEGINNING_MOVE_USER,
    S2_MOVE_TARGET_USER,
    SYSTEM_PROMPT_S2,
    _build_game_messages,
    ends_with_end_game,
    ends_with_move,
)
from game.discreteEngine import discreteGame
from neural_net.loop import MoveRecord, S1S2, new_turns_path, replay_frames
from neural_net.render import canonical_frame

TokenHook = Callable[[dict[str, Any]], None]


@dataclass
class TurnResult:
    text: str
    tokens: list[tuple]
    move: MoveRecord | None
    gold_remaining: int
    exited: bool
    user_text: str
    frame: np.ndarray
    ended: bool
    outcome: dict[str, Any] | None


class S2GameSession:
    """Standalone harness. ``new_game`` deals a board; ``turn`` plays one reply."""

    def __init__(
        self,
        net: S1S2,
        *,
        n_gold: int | None = 1,
        opening: str = "require",
        seed: int | None = None,
        game_size: int | None = None,
    ) -> None:
        self.net = net
        self.n_gold = n_gold
        self.opening = opening
        self.seed = seed
        self.game_size = CONFIG.game_size if game_size is None else int(game_size)
        self.game: discreteGame | None = None
        self.notes: dict[str, tuple[str, int]] = {}
        self.last_move: dict[str, Any] | None = None
        self.outcome: dict[str, Any] | None = None
        self.round = 0
        self.turns_path = None
        self.session_id = ""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="s2-nams", daemon=True,
        )
        self._thread.start()
        try:
            self.client = self._run(connect(CONFIG))
        except Exception:
            self.client = None
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            raise
        self.new_game()

    def _run(self, coro: Any) -> Any:
        """Run a NAMS coroutine on the background loop."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def close(self) -> None:
        """Close the memory client and stop the background loop."""
        try:
            if self.client is not None:
                self._run(self.client.close())
        finally:
            self.client = None
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)

    def new_game(self) -> discreteGame:
        """Deal a fresh board and clear notes, the round, the report, and the outcome."""
        state = random.getstate()
        if self.seed is not None:
            random.seed(self.seed)
        try:
            self.game = new_multi_gold_game(
                gameSize=self.game_size,
                n_gold=self.n_gold,
                opening=self.opening,
            )
        finally:
            if self.seed is not None:
                random.setstate(state)
        self.notes = {}
        self.last_move = None
        self.outcome = None
        self.round = 0
        self.session_id = new_session_id()
        self.turns_path = new_turns_path()
        return self.game

    @property
    def over(self) -> bool:
        """True after ``[END_GAME]`` or after the agent walks out."""
        return self.outcome is not None

    def next_question(self) -> str:
        """The default user line for the turn that has not been played yet."""
        if self.round == 0:
            return S2_BEGINNING_MOVE_USER
        return S2_MOVE_TARGET_USER

    def notepad(self) -> str:
        rows = [
            {"key": key, "value": value, "updated_round": updated}
            for key, (value, updated) in self.notes.items()
        ]
        return format_notepad(rows)

    def turn(
        self,
        question: str,
        *,
        on_token: TokenHook | None = None,
        max_new_tokens: int | None = None,
    ) -> TurnResult:
        if self.game is None:
            raise RuntimeError("S2GameSession has no game")
        if self.over:
            raise RuntimeError("S2GameSession is over")
        self.round += 1
        board_sealed_empty = sealed_empty(game_to_settings_dict(self.game))
        frame = np.array(canonical_frame(self.game), copy=True)
        user_text = compose_s2_question(question, self.last_move)
        self.last_move = None
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        tokens: list[tuple] = []

        def _hook(record: dict[str, Any]) -> None:
            item = (
                record["piece"],
                record["s"],
                record["v"],
                bool(record["eos"]),
                bool(record["move_close"]),
                bool(record["end_close"]),
            )
            tokens.append(item)
            if on_token is not None:
                on_token(record)

        try:
            Image.fromarray(frame, mode="RGB").save(path)
            context = self._run(load_played_context(
                self.client, self.session_id, user_text,
            ))
            messages = _build_game_messages(
                SYSTEM_PROMPT_S2, path, context, user_text,
                notepad=self.notepad(),
            )
            result = self.net.generate(
                self.game,
                messages,
                max_new_tokens=max_new_tokens,
                log_path=self.turns_path,
                on_token=_hook,
                describe=lambda _row, generated: self._settle(
                    generated, user_text, board_sealed_empty,
                ),
            )
        finally:
            if os.path.exists(path):
                os.unlink(path)
        self._record(user_text, result.text)
        return TurnResult(
            text=result.text,
            tokens=tokens,
            move=result.move,
            gold_remaining=gold_remaining(self.game),
            exited=bool(self.game.agent_exited()),
            user_text=user_text,
            frame=frame,
            ended=bool(result.ended),
            outcome=self.outcome,
        )

    def _settle(
        self,
        result,
        user_text: str,
        board_sealed_empty: bool,
    ) -> dict[str, Any]:
        """Notes, the move report, and the outcome. Returns the turn's log fields."""
        for key, value in parse_remember_notes(result.text):
            self.notes[key] = (value, self.round)
        move = result.move
        if move is not None:
            self.last_move = {
                "collected": move.collected,
                "gold_remaining": gold_remaining(self.game),
                "n_steps": move.n_steps,
                "stopped_by": move.stopped_by,
            }
        outcome: dict[str, Any] | None = None
        if result.ended:
            outcome = {
                "reason": "end_game",
                "won": board_sealed_empty,
                "round": self.round,
            }
        elif move is not None and move.exited:
            outcome = {
                "reason": "exit",
                "won": gold_remaining(self.game) == 0,
                "round": self.round,
            }
        self.outcome = outcome
        return {"round": self.round, "user_text": user_text, "outcome": outcome}

    def _record(self, user_text: str, reply: str) -> None:
        """Write this turn, user line and reply. Looks included."""
        if self.client is None:
            raise RuntimeError("S2GameSession is closed")
        if ends_with_move(reply):
            kind = "move"
        elif ends_with_end_game(reply):
            kind = "end"
        else:
            kind = "look"
        self._run(add_user_question(self.client, self.session_id, user_text))
        self._run(add_assistant_message(
            self.client, self.session_id, reply, kind=kind,
        ))

    def play(
        self,
        max_turns: int,
        on_turn: Callable[[TurnResult], None] | None = None,
    ) -> Iterator[TurnResult]:
        """Run ``next_question`` until the game is over or ``max_turns`` replies."""
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")
        for _ in range(max_turns):
            if self.over:
                break
            result = self.turn(self.next_question())
            if on_turn is not None:
                on_turn(result)
            yield result

    def replay(self, move: MoveRecord) -> Iterator[np.ndarray]:
        return replay_frames(move)
