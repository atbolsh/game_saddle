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

Phase 2 calls ``set_magnitude`` and freezes the gates and the Linears.
Only the LoRA trains. ``s`` is a third channel, same Fourier and same
``tanh``; a 4-column payload still means ``(v, v_bar)``.

Ablation, not built: replace the Fourier path for ``v`` with bilinear
interpolation into ``embed_vision.pos_embedding``.
"""

from __future__ import annotations

import contextvars
import logging
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn

logger = logging.getLogger("coord_embed")

COORD_INPUT_LOW = -1.0  # v_bar = 2 s - v, s and v in [0, 1]^2, lies in [-1, 2]
COORD_INPUT_HIGH = 2.0
COORD_FOURIER_BANDS = 8
COORD_CODES = ("s", "v", "v_bar")
_FEATURES = 4 * COORD_FOURIER_BANDS
_S_KEYS = ("gate_s", "proj_s.weight", "proj_s.bias")
#: ``tanh(gate)`` for the phase-2 probe and trainer. Shared so the two
#: scripts cannot drift apart.
MAGNITUDES = {"barely": 0.05, "half": 0.5, "recommended": 0.25}
_MUTED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "coord_embed_muted", default=False,
)


def resolve_magnitude(value: str) -> float:
    """``barely`` / ``half`` / ``recommended``, or a float in ``(0, 1)``."""
    if value in MAGNITUDES:
        return MAGNITUDES[value]
    try:
        magnitude = float(value)
    except ValueError as exc:
        raise ValueError(
            f"magnitude {value!r} is not barely, half, recommended, or a float"
        ) from exc
    if not 0.0 < magnitude < 1.0:
        raise ValueError(
            f"magnitude must be in (0, 1), got {magnitude}"
        )
    return magnitude


@contextmanager
def muted_coord_embed() -> Iterator[None]:
    """KD teacher forwards add no coordinate code.

    The student forward in the same step keeps its payload. The flag is
    a context variable so the teacher path in ``weighted_loss`` can mute
    without holding the embedder.
    """
    token = _MUTED.set(True)
    try:
        yield
    finally:
        _MUTED.reset(token)


class CoordEmbedder(nn.Module):
    """Add a code for ``s``, ``v``, and ``v_bar`` on selected token embeddings.

    Four columns are ``(v, v_bar)``, the path ``decode`` already uses.
    Six columns are ``(s, v, v_bar)``. ``enabled`` stays False until a
    caller turns it on. ``set_magnitude`` freezes every gate and Linear:
    phase 2 trains the LoRA, not this module.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.enabled = False
        self.proj_s = nn.Linear(_FEATURES, self.hidden_size)
        self.proj_v = nn.Linear(_FEATURES, self.hidden_size)
        self.proj_vbar = nn.Linear(_FEATURES, self.hidden_size)
        for proj in (self.proj_s, self.proj_v, self.proj_vbar):
            nn.init.normal_(proj.weight, std=0.02)
            nn.init.zeros_(proj.bias)
        self.gate_s = nn.Parameter(torch.zeros(()))
        self.gate_v = nn.Parameter(torch.zeros(()))
        self.gate_vbar = nn.Parameter(torch.zeros(()))
        self._handle: Any = None
        self._pending: tuple[torch.Tensor, torch.Tensor] | None = None
        self._muted = False

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

    def _channel(
        self, gate: nn.Parameter, proj: nn.Linear, point: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tanh(gate) * proj(self.featurize(point))

    def code(self, coords: torch.Tensor) -> torch.Tensor:
        """``[B, P, 4]`` is ``(v, v_bar)``. ``[B, P, 6]`` is ``(s, v, v_bar)``."""
        width = int(coords.shape[-1])
        if width == 4:
            return self._channel(self.gate_v, self.proj_v, coords[..., 0:2]) + (
                self._channel(self.gate_vbar, self.proj_vbar, coords[..., 2:4])
            )
        if width == 6:
            return (
                self._channel(self.gate_s, self.proj_s, coords[..., 0:2])
                + self._channel(self.gate_v, self.proj_v, coords[..., 2:4])
                + self._channel(self.gate_vbar, self.proj_vbar, coords[..., 4:6])
            )
        raise ValueError(f"code expects 4 or 6 columns, got {width}")

    def set_magnitude(self, magnitude: float) -> None:
        """Set every gate so ``tanh(gate) == magnitude``, then freeze.

        The Linears stay at their std-0.02 draw. A later load of this
        module must use the saved file: a new draw is a different code.
        """
        if not 0.0 < magnitude < 1.0:
            raise ValueError(f"magnitude must be in (0, 1), got {magnitude}")
        gate = math.atanh(magnitude)
        with torch.no_grad():
            self.gate_s.fill_(gate)
            self.gate_v.fill_(gate)
            self.gate_vbar.fill_(gate)
        self.freeze_readout()

    def freeze_readout(self) -> None:
        """Gates and Linears do not train."""
        for param in (
            self.gate_s, self.gate_v, self.gate_vbar,
            *self.proj_s.parameters(),
            *self.proj_v.parameters(),
            *self.proj_vbar.parameters(),
        ):
            param.requires_grad_(False)

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
            # and must see the same coordinates. A muted teacher forward
            # adds nothing even if that payload is live.
            if (
                _MUTED.get() or self._muted or not self.enabled
                or self._pending is None
            ):
                return output
            coords, positions = self._pending
            added = self.code(coords).to(dtype=output.dtype)
            out = output.clone()
            batch, n_pos = positions.shape
            width = int(out.shape[1])
            for b in range(batch):
                for p in range(n_pos):
                    idx = int(positions[b, p])
                    if idx < 0 or idx >= width:
                        raise RuntimeError(
                            f"coord position {idx} outside sequence length {width}"
                        )
                    out[b, idx, :] += added[b, p]
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
        if coords.shape[-1] not in (4, 6):
            raise ValueError(
                f"set_pending coords have {coords.shape[-1]} columns; expected 4 or 6"
            )
        if positions.shape[:2] != coords.shape[:2]:
            raise ValueError(
                f"positions {tuple(positions.shape)} != coords {tuple(coords.shape[:2])}"
            )
        self._pending = (coords, positions)

    def clear_pending(self) -> None:
        self._pending = None

    @contextmanager
    def muted(self) -> Iterator[None]:
        """This embedder adds nothing until the block exits."""
        previous = self._muted
        self._muted = True
        try:
            yield
        finally:
            self._muted = previous

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
        incoming = blob["state_dict"]
        current = self.state_dict()
        missing = [key for key in current if key not in incoming]
        unexpected = [key for key in incoming if key not in current]
        if set(missing) == set(_S_KEYS) and not unexpected:
            logger.warning(
                "coord_embed %s has no s channel; initializing s at gate 0 "
                "and proj std 0.02",
                path,
            )
            self.load_state_dict(incoming, strict=False)
        elif missing or unexpected:
            raise RuntimeError(
                f"coord_embed {path} does not match this module: "
                f"missing {missing}, unexpected {unexpected}"
            )
        else:
            self.load_state_dict(incoming)
        self.enabled = bool(blob["enabled"])
