"""Spatial-softmax readout over one Gemma decoder layer's image grid.

The trunk stays frozen. This module is the only thing that trains in
the learn-to-look phase. It reads layer ``KEPT_LAYER`` (of 48), not the
final hidden state: the last layer is the next-token distribution, and
the image geometry is easier to read before that.

A three-layer mix is deliberately not built. The shelved ablation is a
softmax over ``KEPT_LAYERS = (16, 24, 32)`` with a learned per-head
temperature (ELMo-style), one buffer of ~1500 images, and the choice
rule "the shallowest layer that reaches the text-only floor". Change
``KEPT_LAYER`` (or pass ``--kept-layer``) to try one layer at a time.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

KEPT_LAYER = 24
# Shelved, not built: KEPT_LAYERS = (16, 24, 32), softmax mix, ~1500-image
# buffer, keep the shallowest layer that reaches the text-only floor.
LOCALIZER_DIM = 512
LOCALIZER_REFINE_LAYERS = 2
LOCALIZER_HEADS = 8
LOCALIZER_MLP_DIM = 2048
GRID_SIDE = 16
GRID_CELLS = GRID_SIDE * GRID_SIDE
NO_IMAGE_COORD = 0.5
COORD_ORDER = ("s", "v")
LOCALIZER_COORDS = COORD_ORDER
N_COORD_HEADS = len(COORD_ORDER)


def cell_centers(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Centre of each raster cell, row 0 at the top. [GRID_CELLS, 2].

    ``x = (col + 0.5) / 16``, ``y = 1 - (row + 0.5) / 16``. Same corner
    as ``S1._pool``: row 0 is world ``y = 1``.
    """
    row = torch.arange(GRID_SIDE, device=device)
    col = torch.arange(GRID_SIDE, device=device)
    yy, xx = torch.meshgrid(row, col, indexing="ij")
    x = (xx.reshape(-1).to(dtype) + 0.5) / GRID_SIDE
    y = 1.0 - (yy.reshape(-1).to(dtype) + 0.5) / GRID_SIDE
    return torch.stack((x, y), dim=-1)


def cell_index(points: torch.Tensor) -> torch.Tensor:
    """Raster index of each point, or ``-1`` outside ``[0, 1]^2``.

    ``points`` is ``[..., 2]``. An endpoint of exactly 1 lands in the
    last cell; anything outside the closed unit square is ``-1``.
    """
    x = points[..., 0]
    y = points[..., 1]
    inside = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
    col = (x * GRID_SIDE).long().clamp(0, GRID_SIDE - 1)
    row = ((1.0 - y) * GRID_SIDE).long().clamp(0, GRID_SIDE - 1)
    index = row * GRID_SIDE + col
    return torch.where(inside, index, torch.full_like(index, -1))


class Localizer(nn.Module):
    """Two query heads over a refined 16x16 grid. fp32 throughout.

    ``s`` and ``v`` each attend the same keys. The point is the
    attention-weighted cell centre plus a zero-initialised offset, so
    step 0 is exactly the softmax expectation.
    """

    def __init__(
        self,
        hidden_size: int,
        dim: int = LOCALIZER_DIM,
        refine_layers: int = LOCALIZER_REFINE_LAYERS,
        heads: int = LOCALIZER_HEADS,
        mlp_dim: int = LOCALIZER_MLP_DIM,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.dim = int(dim)
        self.refine_layers = int(refine_layers)
        self.heads = int(heads)
        self.mlp_dim = int(mlp_dim)
        self.in_norm = nn.LayerNorm(self.hidden_size)
        self.in_proj = nn.Linear(self.hidden_size + 2, self.dim)
        layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=self.heads,
            dim_feedforward=self.mlp_dim,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.refine = nn.TransformerEncoder(layer, num_layers=self.refine_layers)
        self.q_norm = nn.LayerNorm(self.hidden_size)
        self.q_proj = nn.Linear(self.hidden_size, N_COORD_HEADS * self.dim)
        self.offset = nn.ModuleList(
            nn.Linear(self.dim, 2) for _ in range(N_COORD_HEADS)
        )
        for linear in self.offset:
            nn.init.zeros_(linear.weight)
            nn.init.zeros_(linear.bias)
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "Localizer hidden=%d dim=%d refine=%d heads=%d params=%d",
            self.hidden_size, self.dim, self.refine_layers, self.heads, n_params,
        )

    def forward(
        self, grid: torch.Tensor, query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``grid [B, 256, H]``, ``query [B, Q, H]`` -> coords, cell logits.

        Returns ``coords [B, Q, 4]`` and ``cell_logits [B, Q, 2, 256]``,
        both fp32. Column order is ``COORD_ORDER`` (``s``, then ``v``).
        """
        if grid.shape[1] != GRID_CELLS:
            raise ValueError(
                f"grid length {grid.shape[1]}, expected {GRID_CELLS}"
            )
        if grid.shape[-1] != query.shape[-1]:
            raise ValueError(
                f"grid hidden {grid.shape[-1]} != query hidden {query.shape[-1]}"
            )
        if grid.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden {grid.shape[-1]} != Localizer hidden {self.hidden_size}"
            )
        g = self.in_norm(grid.float())
        centers = cell_centers(g.device, g.dtype)
        g = torch.cat(
            (g, centers.expand(g.shape[0], -1, -1)), dim=-1,
        )
        keys = self.refine(self.in_proj(g))
        q = self.q_proj(self.q_norm(query.float()))
        q = q.view(q.shape[0], q.shape[1], N_COORD_HEADS, self.dim)
        scale = self.dim ** 0.5
        cell_logits = torch.einsum("bqhd,bkd->bqhk", q, keys) / scale
        attn = torch.softmax(cell_logits, dim=-1)
        expect = attn @ centers.to(dtype=attn.dtype)
        attended = attn @ keys
        points = []
        for head in range(N_COORD_HEADS):
            points.append(
                expect[..., head, :] + self.offset[head](attended[..., head, :])
            )
        coords = torch.cat(points, dim=-1)
        return coords, cell_logits


def no_image_coords(batch: int, n_query: int, device: torch.device) -> torch.Tensor:
    """``[batch, n_query, 4]`` filled with ``NO_IMAGE_COORD`` (``s``, then ``v``)."""
    return torch.full(
        (batch, n_query, 2 * N_COORD_HEADS),
        NO_IMAGE_COORD, dtype=torch.float32, device=device,
    )


def find_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """The unique ``language_model.layers`` ModuleList, or raise."""
    found: list[tuple[str, nn.ModuleList]] = []
    seen: set[int] = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList):
            continue
        if not name.endswith("language_model.layers") or id(module) in seen:
            continue
        seen.add(id(module))
        found.append((name, module))
    if len(found) != 1:
        names = [name for name, _ in found]
        raise RuntimeError(
            "expected exactly one language_model.layers ModuleList, "
            f"found {len(found)}: {names}"
        )
    layers = found[0][1]
    if len(layers) <= KEPT_LAYER:
        raise RuntimeError(
            f"language_model.layers has {len(layers)} layers; "
            f"KEPT_LAYER is {KEPT_LAYER}"
        )
    return layers


class LayerTap:
    """Forward hook that keeps one decoder layer's output tensor."""

    def __init__(self, model: nn.Module, index: int) -> None:
        layers = find_decoder_layers(model)
        if index < 0 or index >= len(layers):
            raise RuntimeError(
                f"LayerTap index {index} outside 0..{len(layers) - 1}"
            )
        self.index = int(index)
        self._captured: torch.Tensor | None = None

        def _hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            self._captured = hidden

        self._handle = layers[index].register_forward_hook(_hook)

    def take(self) -> torch.Tensor:
        captured = self._captured
        self._captured = None
        if captured is None:
            raise RuntimeError(
                f"LayerTap at layer {self.index} captured nothing"
            )
        return captured

    def remove(self) -> None:
        self._handle.remove()


class StopAfterLayer(Exception):
    """Raised from a temporary hook to unwind a truncated forward."""


def truncated_forward(
    model: nn.Module, inputs: dict[str, Any], tap: LayerTap,
) -> torch.Tensor:
    """Run until ``tap``'s layer and return that layer's output.

    The caller owns ``inputs["past_key_values"]``. ``StopAfterLayer``
    unwinds before the model returns, so the filled cache is the object
    the caller passed in, not a value on the model's output. The stop
    hook is registered after the tap, so the tap stores the hidden
    state before the exception fires.

    The model must be in eval. Training mode with gradient checkpointing
    drops ``past_key_values`` inside Gemma 4.
    """
    if model.training:
        raise RuntimeError("truncated_forward requires model.eval()")
    layers = find_decoder_layers(model)
    forwarded = dict(inputs)
    pos = forwarded.get("position_ids")
    from neural_net.s2 import _wants_cache_position

    if not _wants_cache_position(model):
        forwarded.pop("cache_position", None)
    elif "cache_position" not in forwarded and pos is not None:
        # Shared across the batch. A caller that already set
        # cache_position (the stage-B append index) keeps it; RoPE
        # still reads the per-row position_ids.
        forwarded["cache_position"] = pos[0]

    def _stop(_module: nn.Module, _inputs: Any, _output: Any) -> None:
        raise StopAfterLayer()

    handle = layers[tap.index].register_forward_hook(_stop)
    try:
        with torch.no_grad():
            model(**forwarded, use_cache=True)
    except StopAfterLayer:
        pass
    finally:
        handle.remove()
    return tap.take()


def grid_positions(
    model_inputs: dict[str, Any],
    tokenizer: Any = None,
) -> torch.Tensor:
    """``[B, 256]`` long indices of the image tokens in each row.

    Image tokens are where ``mm_token_type_ids`` (else ``token_type_ids``)
    is nonzero. Anything other than exactly 256 per row raises, with the
    per-row counts, the token-type values seen, and the decoded text
    from 4 tokens before the first nonzero to 4 after the last.
    """
    if "mm_token_type_ids" in model_inputs:
        mask = model_inputs["mm_token_type_ids"]
    elif "token_type_ids" in model_inputs:
        mask = model_inputs["token_type_ids"]
    else:
        raise KeyError(
            "grid_positions: neither mm_token_type_ids nor token_type_ids"
        )
    if "input_ids" not in model_inputs:
        raise KeyError("grid_positions: model_inputs has no input_ids")
    ids = model_inputs["input_ids"]
    positions: list[torch.Tensor] = []
    bad: list[str] = []
    values = sorted({int(v) for v in mask.unique().tolist()})
    for row in range(int(mask.shape[0])):
        idx = (mask[row] != 0).nonzero(as_tuple=False).flatten()
        if int(idx.numel()) != GRID_CELLS:
            window = ""
            if idx.numel() > 0:
                lo = max(0, int(idx[0]) - 4)
                hi = min(int(ids.shape[1]), int(idx[-1]) + 5)
                span = [int(t) for t in ids[row, lo:hi].tolist()]
                if tokenizer is not None:
                    window = tokenizer.decode(span, skip_special_tokens=False)
                else:
                    window = str(span)
            bad.append(
                f"row {row}: {int(idx.numel())} image tokens, window={window!r}"
            )
            continue
        positions.append(idx)
    if bad:
        raise RuntimeError(
            "grid_positions: expected exactly "
            f"{GRID_CELLS} image tokens per row; token-type values {values}; "
            + "; ".join(bad)
        )
    return torch.stack(positions, dim=0)


def gather_grid(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """``hidden [B, T, H]``, ``positions [B, 256]`` -> ``[B, 256, H]``."""
    if positions.shape[0] != hidden.shape[0]:
        raise ValueError(
            f"positions batch {positions.shape[0]} != hidden batch {hidden.shape[0]}"
        )
    index = positions.to(device=hidden.device).unsqueeze(-1).expand(
        -1, -1, hidden.shape[-1],
    )
    return hidden.gather(1, index)
