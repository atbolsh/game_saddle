"""One S2 game.

Notes live in a dict. The conversation lives in NAMS: a turn loads the
last 7 messages of this session plus up to 2 older ones from semantic
search, when any remain outside those 7. Every turn is written, look
or move, and each reply is written with its gaze trace: the ``(s, v)``
the Localizer read at every token. When a later turn recalls the reply,
its tokens in the prompt carry those codes again, and the token right
after it carries where the eyes were left (:mod:`neural_net.gaze_replay`;
``gaze_replay`` is ``full``, ``anchor`` or ``off``). A turn shows S2 the
current frame and the question, and wakes S1 only when the reply ends
with ``[MOVE]``.
A reply that contains ``[FLAG failure: ...]`` is stored like any other
reply, gaze included. That message also gets two snapshots, ``before``
and ``after``: the Settings dict and the board of the attempt. The
attempt is this turn when the reply moved, otherwise the previous
attempt the session held. A look leaves the board unchanged, so both
labels are that same state. The default load returns the flag line with the
message and does not return the dicts or the images.
A reply that ends with ``[END_GAME]``, or an S1 window that walks
out, ends the game. ``new_game`` opens
``weights/s1s2_runs/<stamp>/turns.jsonl`` and every ``turn`` appends
one record, including the S1 action list. ``replay`` rebuilds a move
window from the recorded actions. ``play`` runs the default question
sequence.
"""

from __future__ import annotations

import asyncio
import logging
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
    parse_failure_flags,
    parse_remember_notes,
    sealed_empty,
)
from agent.image_store import (
    link_snapshot_to_message,
    snapshot_id,
    store_snapshot,
)
from agent.memory import (
    add_assistant_message,
    add_user_question,
    connect,
    format_notepad,
    load_played_messages,
    new_session_id,
    played_block,
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
from neural_net.gaze_replay import GazeReplay, resolve_mode
from neural_net.loop import MoveRecord, S1S2, new_turns_path, replay_frames
from neural_net.render import canonical_frame

TokenHook = Callable[[dict[str, Any]], None]

logger = logging.getLogger(__name__)


@dataclass
class _Attempt:
    """One turn's board before the reply and after S1.

    A look does not move the game, so ``before`` and ``after`` are the
    same settings dict and the same frame.
    """

    before: dict[str, Any]
    after: dict[str, Any]
    before_frame: np.ndarray
    after_frame: np.ndarray


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
    # ``(sx, sy, vx, vy)`` read at the last prompt token.
    prefill: list[float] | None = None
    # How many prompt tokens carried a gaze replayed from recalled replies.
    n_replayed: int = 0


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
        gaze_replay: str | None = None,
    ) -> None:
        self.net = net
        self.n_gold = n_gold
        self.opening = opening
        self.seed = seed
        # ``full``, ``anchor`` or ``off``; None reads $S2_GAZE_REPLAY, then
        # ``full``. May be reassigned between turns.
        self.gaze_replay = resolve_mode(gaze_replay)
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
        """Deal a fresh board and clear notes, the held attempt, the round, the report, and the outcome."""
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
        self._attempt: _Attempt | None = None
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
        before_settings = game_to_settings_dict(self.game)
        board_sealed_empty = sealed_empty(before_settings)
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
            recalled = self._run(load_played_messages(
                self.client, self.session_id, user_text,
            ))
            messages = _build_game_messages(
                SYSTEM_PROMPT_S2, path, played_block(recalled), user_text,
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
                gaze_replay=GazeReplay(recalled, resolve_mode(self.gaze_replay)),
            )
        finally:
            if os.path.exists(path):
                os.unlink(path)
        moved = result.move is not None
        if moved:
            after_settings = game_to_settings_dict(self.game)
            after_frame = np.array(canonical_frame(self.game), copy=True)
        else:
            after_settings = before_settings
            after_frame = frame
        this = _Attempt(before_settings, after_settings, frame, after_frame)
        notes = [note for _key, note in parse_failure_flags(result.text) if note]
        if notes and (moved or self._attempt is None):
            chosen: _Attempt | None = this
        elif notes:
            chosen = self._attempt
        else:
            chosen = None
        self._attempt = this
        self._record(
            user_text, result.text, result.gaze,
            failure="\n".join(notes) if notes else None,
            attempt=chosen,
        )
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
            prefill=result.prefill,
            n_replayed=result.n_replayed,
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

    def _record(
        self,
        user_text: str,
        reply: str,
        gaze: dict[str, Any] | None = None,
        failure: str | None = None,
        attempt: _Attempt | None = None,
    ) -> None:
        """Write this turn, user line and reply. Looks included.

        ``gaze`` is the reply's gaze trace; it is stored with the reply
        whatever ``gaze_replay`` is, so a later session can replay it.
        ``failure`` is the flag note. The reply text, including the
        ``[FLAG failure: ...]`` line, is stored unchanged. When the note
        is set, ``attempt`` supplies the before and after Settings dicts
        and frames, written as two snapshots on this message.
        """
        if self.client is None:
            raise RuntimeError("S2GameSession is closed")
        if ends_with_move(reply):
            kind = "move"
        elif ends_with_end_game(reply):
            kind = "end"
        else:
            kind = "look"
        self._run(add_user_question(self.client, self.session_id, user_text))
        message = self._run(add_assistant_message(
            self.client, self.session_id, reply, kind=kind, gaze=gaze,
            failure=failure,
        ))
        if failure is None:
            return
        if attempt is None:
            raise RuntimeError("failure flag has no before/after board")
        message_id = getattr(message, "id", None)
        if message_id is None and isinstance(message, dict):
            message_id = message.get("id")
        if not message_id:
            raise RuntimeError(
                "assistant message has no id; failure boards were not stored"
            )
        message_id = str(message_id)
        before_path = self._run(self._write_board(
            message_id, "before", attempt.before, attempt.before_frame,
        ))
        after_path = self._run(self._write_board(
            message_id, "after", attempt.after, attempt.after_frame,
        ))
        line = (
            f"failure flag stored on message {message_id} "
            f"(before {before_path}, after {after_path})"
        )
        logger.info("%s", line)
        print(line)

    async def _write_board(
        self,
        message_id: str,
        role: str,
        settings: dict[str, Any],
        frame: np.ndarray,
    ) -> str:
        """One snapshot: Settings dict on the node, ``frame`` as the PNG."""
        sid = snapshot_id()
        path, _props = await store_snapshot(
            self.client, self.session_id, sid, None, settings,
            label=role, frame=frame,
        )
        await link_snapshot_to_message(
            self.client, message_id, sid, role=role,
        )
        return path

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
