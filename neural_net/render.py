"""Framebuffer export and a wall-layer cache for the live S1 loop.

``discreteGame.draw_walls`` allocates and rotates a surface per wall per
frame. Walls do not move during a turn, so the live loop snapshots that
layer once and redraws only the agent and the gold.

Export matches ``getData``: pygame's surface is y-down, and one vertical
flip makes a larger world-y higher in the image. The array is HxWx3 RGB.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame

from game.discreteEngine import discreteGame


def export_surface(surface: pygame.Surface) -> np.ndarray:
    """HxWx3 uint8 RGB, y-up, from an unflipped pygame surface."""
    raw = pygame.surfarray.array3d(surface)  # width, height, 3
    return np.transpose(raw[:, ::-1, :], (1, 0, 2))


def canonical_frame(game: discreteGame) -> np.ndarray:
    """One full engine draw, then :func:`export_surface`."""
    game.draw()
    return export_surface(game.windowSurface)


class WallCache:
    """Reuse the white-and-walls layer across S1 ticks on one game."""

    def __init__(self) -> None:
        self._surface: pygame.Surface | None = None
        self._sig: tuple | None = None

    def render(self, game: discreteGame) -> np.ndarray:
        sig = (
            int(game.settings.gameSize),
            tuple(tuple(float(v) for v in wall) for wall in game.settings.walls),
        )
        if self._surface is None or self._sig != sig:
            game.draw(ignore_agent=True, ignore_gold=True)
            self._surface = game.windowSurface.copy()
            # Exact white is the board color, so the blit covers walls only.
            # Draw order matches ``draw``: agent, then walls, then gold.
            self._surface.set_colorkey(game.WHITE)
            self._sig = sig
        game.windowSurface.fill(game.WHITE)
        game.draw_agent()
        game.windowSurface.blit(self._surface, (0, 0))
        game.draw_gold()
        return export_surface(game.windowSurface)


def apply_primitive(game: discreteGame, action: str) -> int:
    """One S1 primitive without the full wall redraw ``universal_update`` does.

    ``FORWARD`` uses the same default step as ``stepForward`` (1/16 of the
    board) and the same slide and gold-pickup code. Turns add or subtract
    one ``pi/30`` step, matching ``swivel_clock`` / ``swivel_anticlock``.
    """
    if action == "noop":
        return 0
    if action == "CLOCK":
        game.settings.direction = game.mod2pi(
            game.settings.direction + math.pi / 30
        )
        return 0
    if action == "ANTICLOCK":
        game.settings.direction = game.mod2pi(
            game.settings.direction - math.pi / 30
        )
        return 0
    if action == "FORWARD":
        theta = game.settings.direction
        game._slide_move(math.sin(theta), math.cos(theta), 1.0 / 16)
        return int(game.gold_update() or 0)
    raise ValueError(f"unknown S1 action {action!r}")


def pose(game: discreteGame) -> dict[str, float]:
    settings = game.settings
    return {
        "x": float(settings.agent_x),
        "y": float(settings.agent_y),
        "theta": float(settings.direction),
    }
