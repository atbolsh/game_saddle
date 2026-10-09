"""One S2 game, no memory graph.

Notes live in a dict. A turn shows S2 the current frame and the
question, and wakes S1 only when the reply ends with ``[MOVE]``.
``replay`` rebuilds that window from the recorded actions.
"""

from __future__ import annotations

import os
import random
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import numpy as np
from PIL import Image

from agent.config import CONFIG
from agent.game_io import (
    compose_s2_question,
    gold_remaining,
    new_multi_gold_game,
    parse_remember_notes,
)
from agent.memory import format_notepad
from agent.modes import SYSTEM_PROMPT_S2, _build_game_messages
from game.discreteEngine import discreteGame
from neural_net.loop import MoveRecord, S1S2, replay_frames
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
        self.round = 0
        self.new_game()

    def new_game(self) -> discreteGame:
        """Deal a fresh board and clear notes, the round, and the last report."""
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
        self.round = 0
        return self.game

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
        self.round += 1
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
            )
            tokens.append(item)
            if on_token is not None:
                on_token(record)

        try:
            Image.fromarray(frame, mode="RGB").save(path)
            messages = _build_game_messages(
                SYSTEM_PROMPT_S2, path, "", user_text, notepad=self.notepad(),
            )
            result = self.net.generate(
                self.game,
                messages,
                max_new_tokens=max_new_tokens,
                on_token=_hook,
            )
        finally:
            if os.path.exists(path):
                os.unlink(path)
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
        return TurnResult(
            text=result.text,
            tokens=tokens,
            move=move,
            gold_remaining=gold_remaining(self.game),
            exited=bool(self.game.agent_exited()),
            user_text=user_text,
            frame=frame,
        )

    def replay(self, move: MoveRecord) -> Iterator[np.ndarray]:
        return replay_frames(move)
