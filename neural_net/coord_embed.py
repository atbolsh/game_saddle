"""Gated Fourier code added to token embeddings. Inert in learn-to-look.

Nothing here trains in the learn-to-look phase. The gates and
projections receive gradient only in the later
learn-to-tell-where-you-are-looking phase, after a Localizer hits the
floors. That phase sets ``enabled = True``, teacher-forces label
``v``, ``v_bar`` at reply positions, later mixes in the Localizer's
own detached predictions, and adds a read-back loss in which a later
position's Localizer output must reproduce an earlier position's
injected coordinates, with LoRA at a real learning rate and the KD
anchor on.

The output is ``tanh(gate) * proj(features)``. The gate starts at 0, so
the added vector is exactly 0. The projection is random (std 0.02), not
zero: a zero projection would make the gate's gradient zero (it
multiplies ``proj(features)``) and the projection's gradient zero (it
multiplies ``tanh(0)``), and neither could ever move. ``tanh`` is used
rather than ``sigmoid`` (which is 0.5 at init) or ``ReLU`` (zero
gradient at 0).

Ablation, not built: replace the Fourier path for ``v`` with bilinear
interpolation into ``embed_vision.pos_embedding``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

COORD_INPUT_LOW = -1.0  # v_bar = 2 s - v, s and v in [0, 1]^2, lies in [-1, 2]
COORD_INPUT_HIGH = 2.0
COORD_FOURIER_BANDS = 8
COORD_CODES = ("v", "v_bar")
_FEATURES = 4 * COORD_FOURIER_BANDS


class CoordEmbedder(nn.Module):
    """Add a code for ``(v, v_bar)`` on top of selected token embeddings.

    ``s`` is not an input: it is already visible in the picture.
    ``enabled`` is a plain attribute, persisted in the save file, and
    stays False until the later phase turns it on.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.enabled = False
        self.proj_v = nn.Linear(_FEATURES, self.hidden_size)
        self.proj_vbar = nn.Linear(_FEATURES, self.hidden_size)
        nn.init.normal_(self.proj_v.weight, std=0.02)
        nn.init.normal_(self.proj_vbar.weight, std=0.02)
        nn.init.zeros_(self.proj_v.bias)
        nn.init.zeros_(self.proj_vbar.bias)
        self.gate_v = nn.Parameter(torch.zeros(()))
        self.gate_vbar = nn.Parameter(torch.zeros(()))
        self._handle: Any = None
        self._pending: tuple[torch.Tensor, torch.Tensor] | None = None

    def featurize(self, x: torch.Tensor) -> torch.Tensor:
        """``[..., 2]`` -> ``[..., 4 * bands]``. Raise outside ``[-1, 2]``.

        ``x_unit = (x + 1) / 3`` maps the admissible range onto ``[0, 1]``.
        Band ``k = 0`` is ``cos(pi * x_unit)``, strictly monotone on that
        interval, so the lowest band alone is injective: no two admissible
        inputs share a code. Higher bands wrap on purpose.
        """
        below = bool((x < COORD_INPUT_LOW).any())
        above = bool((x > COORD_INPUT_HIGH).any())
        if below or above:
            raise ValueError(
                f"coordinate {float(x.min())}..{float(x.max())} outside "
                f"[{COORD_INPUT_LOW}, {COORD_INPUT_HIGH}]"
            )
        span = COORD_INPUT_HIGH - COORD_INPUT_LOW
        x_unit = (x - COORD_INPUT_LOW) / span
        parts: list[torch.Tensor] = []
        for k in range(COORD_FOURIER_BANDS):
            angle = math.pi * (2 ** k) * x_unit
            parts.append(torch.sin(angle))
            parts.append(torch.cos(angle))
        return torch.cat(parts, dim=-1)

    def code(self, coords: torch.Tensor) -> torch.Tensor:
        """``[B, P, 4]`` ``(v_x, v_y, v_bar_x, v_bar_y)`` -> ``[B, P, H]``."""
        if coords.shape[-1] != 4:
            raise ValueError(f"code expects 4 columns, got {coords.shape[-1]}")
        v = torch.tanh(self.gate_v) * self.proj_v(self.featurize(coords[..., 0:2]))
        vbar = torch.tanh(self.gate_vbar) * self.proj_vbar(
            self.featurize(coords[..., 2:4])
        )
        return v + vbar

    def install(self, model: nn.Module) -> None:
        """Hook ``get_input_embeddings()``, after Gemma's ``sqrt(d)`` scale."""
        if self._handle is not None:
            raise RuntimeError("CoordEmbedder.install called twice")
        emb = model.get_input_embeddings()
        if emb is None:
            raise RuntimeError("model.get_input_embeddings() returned None")

        def _hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> torch.Tensor:
            # The payload stays until clear_pending. Under gradient
            # checkpointing the hook fires again on the recompute pass
            # and must see the same coordinates.
            if self._pending is None:
                return output
            coords, positions = self._pending
            added = self.code(coords).to(dtype=output.dtype)
            out = output.clone()
            batch, n_pos = positions.shape
            for b in range(batch):
                for p in range(n_pos):
                    out[b, int(positions[b, p]), :] += added[b, p]
            return out

        self._handle = emb.register_forward_hook(_hook)

    def uninstall(self) -> None:
        """Drop the hook so ``install`` can run on a reloaded trunk."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def set_pending(self, coords: torch.Tensor, positions: torch.Tensor) -> None:
        """Store a payload. A no-op while ``enabled`` is False.

        A second call while a payload is live raises: the caller clears
        it in a ``finally`` after the step.
        """
        if not self.enabled:
            return
        if self._pending is not None:
            raise RuntimeError("CoordEmbedder already has a live payload")
        if coords.shape[-1] != 4:
            raise ValueError(f"set_pending coords have {coords.shape[-1]} columns")
        if positions.shape[:2] != coords.shape[:2]:
            raise ValueError(
                f"positions {tuple(positions.shape)} != coords {tuple(coords.shape[:2])}"
            )
        self._pending = (coords, positions)

    def clear_pending(self) -> None:
        self._pending = None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "enabled": self.enabled,
                "bands": COORD_FOURIER_BANDS,
                "input_low": COORD_INPUT_LOW,
                "input_high": COORD_INPUT_HIGH,
                "hidden_size": self.hidden_size,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        blob = torch.load(Path(path), map_location="cpu", weights_only=True)
        for key, expected in (
            ("bands", COORD_FOURIER_BANDS),
            ("input_low", COORD_INPUT_LOW),
            ("input_high", COORD_INPUT_HIGH),
            ("hidden_size", self.hidden_size),
        ):
            got = blob.get(key)
            if got != expected:
                raise RuntimeError(
                    f"coord_embed {key} is {got}, this module has {expected}"
                )
        self.load_state_dict(blob["state_dict"])
        self.enabled = bool(blob["enabled"])
