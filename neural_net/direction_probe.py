"""One question about one board.

The menu is three base models, then every assembled full save under
``weights/full/``. A full save answers through :func:`generate_s2_reply`,
so the Localizer's ``s`` and ``v`` are added to the next token the way
play adds them. A base model answers with no Localizer: no ``s``, no
``v``, nothing stored. S1 is not loaded.

The system prompt is the S2 prompt loaded from NAMS. One question ends
the game; the next question needs a new board.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from agent.config import CONFIG
from agent.game_io import new_multi_gold_game
from agent.inkling import ANALYST_MODELS, build_analyst_model
from agent.modes import _build_game_messages, load_s2_prompt_sync
from neural_net.loop import generate_s2_reply
from neural_net.render import canonical_frame
from neural_net.s2 import GemmaS2
from neural_net.weight_catalog import label, list_assembled

#: Dropdown order above the full saves.
DIRECTION_KEYS = ("inkling-2-mini", "gemma-4-31b", "gemma-4-12b")
_FULL = "full:"


def direction_menu_options(root: Path | None = None) -> list[tuple[str, str]]:
    """Base models, then full saves, newest first.

    A full-save value is ``full:`` plus the directory ``GemmaS2.load_all``
    opens. ``s1.pt`` in that directory is not a menu row and is not loaded.
    """
    options = [
        (ANALYST_MODELS[key].label, key) for key in DIRECTION_KEYS
    ]
    seen = {value for _text, value in options}
    for path in list_assembled(root):
        value = _FULL + str(path)
        if value in seen:
            raise RuntimeError(
                f"full save {value!r} collides with a base-model key"
            )
        seen.add(value)
        options.append((label(path, root), value))
    return options


class DirectionProbe:
    """One loaded trunk and one board. The board allows one question."""

    def __init__(self) -> None:
        self.base: Any = None
        self.s2: GemmaS2 | None = None
        self.game: Any = None
        self.system = ""
        self.used = False
        self.choice = ""
        self.label = ""

    def load(self, choice: str) -> dict[str, Any]:
        """Load one menu row. The previous trunk is freed first."""
        self._unload()
        self.choice = choice
        try:
            if choice in DIRECTION_KEYS:
                self.base = build_analyst_model(choice).load()
                self.label = self.base.label
                return {
                    "label": self.label,
                    "model_id": self.base.model_id,
                    "localizer": False,
                }
            if not choice.startswith(_FULL):
                raise RuntimeError(f"unknown direction model {choice!r}")
            folder = Path(choice[len(_FULL):])
            self.s2 = GemmaS2("gemma-4-12b")
            self.s2.load_all(folder)
            self.label = label(folder)
            return {
                "label": self.label,
                "model_id": str(folder),
                "localizer": True,
            }
        except Exception:
            self._unload()
            raise

    def new_game(self, n_gold: int | None, opening: str) -> dict[str, Any]:
        """Deal a board and load the S2 prompt. The question slot opens."""
        if self.base is None and self.s2 is None:
            raise RuntimeError("Load a model first.")
        self.system = load_s2_prompt_sync()
        self.game = new_multi_gold_game(
            gameSize=CONFIG.game_size,
            n_gold=n_gold,
            opening=opening,
        )
        self.used = False
        return {
            "n_gold": len(self.game.settings.gold),
            "opening": opening,
            "frame": canonical_frame(self.game),
        }

    def ask(self, question: str) -> str:
        """One reply. A second call raises until :meth:`new_game`."""
        text = question.strip()
        if not text:
            raise RuntimeError("The question is empty.")
        if self.game is None or not self.system:
            raise RuntimeError("Start a game first.")
        if self.used:
            raise RuntimeError(
                "This game already had its one step. Start a new game."
            )
        frame = canonical_frame(self.game)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            Image.fromarray(frame, mode="RGB").save(path)
            messages = _build_game_messages(self.system, path, "", text)
            if self.s2 is not None:
                reply = generate_s2_reply(self.s2, self.game, messages)
            elif self.base is not None:
                reply = self.base.generate(messages)
            else:
                raise RuntimeError("Load a model first.")
        finally:
            if os.path.exists(path):
                os.unlink(path)
        self.used = True
        return reply

    def _unload(self) -> None:
        if self.base is not None:
            self.base.unload()
            self.base = None
        s2 = self.s2
        self.s2 = None
        if s2 is None:
            return
        if s2.coord_embed is not None:
            s2.coord_embed.uninstall()
        if s2._tap is not None:
            s2._tap.remove()
            s2._tap = None
        s2.localizer = None
        s2.coord_embed = None
        s2.vl.unload()
