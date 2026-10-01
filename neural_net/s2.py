"""Gemma 4 with a Localizer readout on one decoder layer.

The Localizer reads the image-token states at ``KEPT_LAYER`` and the
hidden state of the token being generated, and emits
``(s_x, s_y, v_x, v_y, v_bar_x, v_bar_y)``. A prompt with no image
yields ``NO_IMAGE_COORD`` (0.5) for every coordinate.

``CoordEmbedder`` is installed on the token embedding and stays
disabled. It does nothing until a later phase sets ``enabled``.

Save / load slices:

* save_all / load_all — Gemma slice plus localizer.pt and coord_embed.pt
* save_gemma / load_gemma — Gemma only
* save_snapshot / load_snapshot — the Localizer bundle

A PEFT adapter (the aug27 checkpoints) is saved with save_pretrained.
A bare HuggingFace Gemma is saved as a full state_dict; that file is
the whole 12B and is large on purpose.

A sep24 snapshot that only has ``coord_head.pt`` is rejected by name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from agent.model import VLModel, spec_for
from neural_net.coord_embed import CoordEmbedder
from neural_net.localizer import (
    KEPT_LAYER,
    LayerTap,
    Localizer,
    gather_grid,
    grid_positions,
    no_image_coords,
)


def _wants_cache_position(model: nn.Module) -> bool:
    """Current transformers places a cached step with cache_position.
    Pass it when the forward signature names it or accepts **kwargs."""
    import inspect

    params = inspect.signature(model.forward).parameters
    if "cache_position" in params:
        return True
    return any(
        item.kind == inspect.Parameter.VAR_KEYWORD for item in params.values()
    )


def _hidden_size(model: nn.Module) -> int:
    cfg = model.config
    for obj in (cfg, getattr(cfg, "text_config", None)):
        size = getattr(obj, "hidden_size", None) if obj is not None else None
        if size:
            return int(size)
    raise RuntimeError(
        "Gemma config has no hidden_size (checked config and text_config)"
    )


def _find_lm_head(model: nn.Module) -> nn.Module:
    found: list[tuple[str, nn.Module]] = []
    seen: set[int] = set()
    for name, module in model.named_modules():
        if name.split(".")[-1] != "lm_head" or id(module) in seen:
            continue
        seen.add(id(module))
        found.append((name, module))
    if len(found) != 1:
        names = [name for name, _ in found]
        raise RuntimeError(
            f"expected exactly one lm_head, found {len(found)}: {names}"
        )
    return found[0][1]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class GemmaS2:
    """Loaded lazily. Call load, load_gemma, or load_all before prefill."""

    def __init__(
        self,
        model_key: str = "gemma-4-12b",
        checkpoint: str | None = None,
        kept_layer: int = KEPT_LAYER,
    ) -> None:
        self.model_key = model_key
        self.vl = VLModel(spec_for(model_key), checkpoint=checkpoint)
        self.kept_layer = int(kept_layer)
        self.localizer: Localizer | None = None
        self.coord_embed: CoordEmbedder | None = None
        self._tap: LayerTap | None = None
        self._grid: torch.Tensor | None = None

    @property
    def hidden_size(self) -> int:
        if self.vl.model is None:
            raise RuntimeError("hidden_size called before the model was loaded")
        return _hidden_size(self.vl.model)

    def load(self) -> GemmaS2:
        """Load the HuggingFace base plus this instance's checkpoint, and
        a freshly initialized Localizer."""
        self.vl.load()
        self._build_readout()
        return self

    def _build_readout(self) -> None:
        if self.vl.model is None:
            raise RuntimeError("_build_readout called before the model was loaded")
        device = next(self.vl.model.parameters()).device
        size = self.hidden_size
        previous_loc = self.localizer
        previous_emb = self.coord_embed
        if self._tap is not None:
            self._tap.remove()
            self._tap = None
        if previous_emb is not None:
            previous_emb.uninstall()
        self.localizer = Localizer(size).to(device=device, dtype=torch.float32)
        self.coord_embed = CoordEmbedder(size).to(device=device, dtype=torch.float32)
        if previous_loc is not None and previous_loc.hidden_size == size:
            self.localizer.load_state_dict(previous_loc.state_dict())
        if previous_emb is not None and previous_emb.hidden_size == size:
            self.coord_embed.load_state_dict(previous_emb.state_dict())
            self.coord_embed.enabled = previous_emb.enabled
        self._tap = LayerTap(self.vl.model, self.kept_layer)
        self.coord_embed.install(self.vl.model)
        self._grid = None

    def prefill(self, encoded: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """Cached prefill. Returns (logits [B, V], coords [B, 6], past)."""
        self._require_loaded()
        inputs = self.vl._move_inputs_to_model(encoded)
        mask = inputs["attention_mask"]
        inputs["position_ids"] = (mask.long().cumsum(-1) - 1).clamp(min=0)
        logits, past = self._forward(inputs)
        hidden = self._tap.take()
        batch = int(hidden.shape[0])
        device = hidden.device
        has_mask = (
            "mm_token_type_ids" in inputs or "token_type_ids" in inputs
        )
        mask_tensor = inputs.get("mm_token_type_ids", inputs.get("token_type_ids"))
        if not has_mask or mask_tensor is None or not bool((mask_tensor != 0).any()):
            self._grid = None
            coords = no_image_coords(batch, 1, device)[:, 0, :]
        else:
            positions = grid_positions(inputs)
            self._grid = gather_grid(hidden, positions).to(torch.bfloat16)
            coords = self.localizer(self._grid, hidden[:, -1:, :])[0][:, 0, :]
        return logits, coords, past

    def decode(
        self,
        token_ids: torch.Tensor,
        past: Any,
        seq_len: int,
        prev_coords: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """One cached decode step.

        token_ids is [B], the tokens just produced. seq_len is how many
        real tokens are already in the cache; the new token sits at that
        index. ``prev_coords`` is the previous step's ``[B, 6]``; while
        the embedder is disabled, passing it changes nothing.
        """
        self._require_loaded()
        if token_ids.dim() != 1:
            raise ValueError(
                f"decode token_ids must be [B], got {tuple(token_ids.shape)}"
            )
        device = next(self.vl.model.parameters()).device
        batch = int(token_ids.shape[0])
        inputs = {
            "input_ids": token_ids.to(device).view(batch, 1),
            "attention_mask": torch.ones(
                batch, seq_len + 1, dtype=torch.long, device=device
            ),
            "position_ids": torch.full(
                (batch, 1), seq_len, dtype=torch.long, device=device
            ),
            "past_key_values": past,
        }
        if prev_coords is not None:
            self.coord_embed.set_pending(
                prev_coords[:, 2:6].unsqueeze(1).to(device),
                positions=torch.zeros(batch, 1, dtype=torch.long, device=device),
            )
        try:
            logits, past_out = self._forward(inputs)
            hidden = self._tap.take()
            if self._grid is None:
                coords = no_image_coords(batch, 1, device)[:, 0, :]
            else:
                coords = self.localizer(self._grid, hidden)[0][:, 0, :]
        finally:
            self.coord_embed.clear_pending()
        return logits, coords, past_out

    def _forward(self, inputs: dict[str, Any]) -> tuple[torch.Tensor, Any]:
        pos = inputs.get("position_ids")
        if pos is not None and _wants_cache_position(self.vl.model):
            inputs = dict(inputs)
            inputs["cache_position"] = pos[0]
        with torch.inference_mode():
            out = self.vl.model(**inputs, use_cache=True, logits_to_keep=1)
        logits = out.logits
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        elif logits.dim() != 2:
            raise RuntimeError(f"logits rank {logits.dim()}, expected 2 or 3")
        past = getattr(out, "past_key_values", None)
        if past is None:
            raise RuntimeError("Gemma forward returned no past_key_values")
        return logits, past

    def _require_loaded(self) -> None:
        if (
            self.vl.model is None
            or self.localizer is None
            or self.coord_embed is None
            or self._tap is None
        ):
            raise RuntimeError(
                "GemmaS2 is not loaded; call load(), load_gemma(), or load_all()"
            )

    def _require_model(self) -> None:
        if self.vl.model is None:
            raise RuntimeError("Gemma is not loaded")

    def save_gemma(self, path: str | Path) -> None:
        """Write the Gemma slice and gemma_meta.json.

        PEFT adapters use save_pretrained. A bare base model writes
        gemma.pt, the full state dict.
        """
        self._require_model()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        from peft import PeftModel

        meta: dict[str, Any] = {
            "hf_id": self.vl.spec.hf_id,
            "model_key": self.vl.spec.key,
        }
        if isinstance(self.vl.model, PeftModel):
            self.vl.model.save_pretrained(path)
            meta["format"] = "peft"
        else:
            torch.save(self.vl.model.state_dict(), path / "gemma.pt")
            meta["format"] = "state_dict"
        _write_json(path / "gemma_meta.json", meta)

    def load_gemma(self, path: str | Path) -> None:
        """Replace Gemma weights from a directory written by save_gemma.

        A readout that already matches the hidden size is kept.
        """
        path = Path(path)
        meta_path = path / "gemma_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"load_gemma: no gemma_meta.json under {path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fmt = meta.get("format")
        key = meta.get("model_key")
        if not key or fmt not in ("peft", "state_dict"):
            raise RuntimeError(
                "load_gemma: gemma_meta.json must have model_key and "
                f"format peft|state_dict, got {meta!r}"
            )
        previous_loc = self.localizer
        previous_emb = self.coord_embed
        self.model_key = key
        self.vl = VLModel(spec_for(key), checkpoint=None)
        self.vl.load()
        if fmt == "peft":
            from peft import PeftModel

            self.vl.model = PeftModel.from_pretrained(
                self.vl.model, str(path), is_trainable=False
            )
        else:
            weights = path / "gemma.pt"
            if not weights.is_file():
                raise FileNotFoundError(f"load_gemma: missing {weights}")
            state = torch.load(weights, map_location="cpu", weights_only=True)
            self.vl.model.load_state_dict(state)
        self.vl.model.eval()
        self.localizer = previous_loc
        self.coord_embed = previous_emb
        self._build_readout()

    def _save_readout(self, path: Path) -> None:
        if self.localizer is None or self.coord_embed is None:
            raise RuntimeError("save called before the readout exists")
        torch.save(
            {
                "state_dict": self.localizer.state_dict(),
                "kept_layer": self.kept_layer,
                "dim": self.localizer.dim,
                "refine_layers": self.localizer.refine_layers,
                "heads": self.localizer.heads,
                "mlp_dim": self.localizer.mlp_dim,
                "hidden_size": self.localizer.hidden_size,
            },
            path / "localizer.pt",
        )
        self.coord_embed.save(path / "coord_embed.pt")

    def load_localizer(self, path: str | Path) -> GemmaS2:
        """Read the Localizer bundle onto an already-loaded trunk.

        The directory holds ``localizer.pt``, ``coord_embed.pt``, and
        ``train_meta.json``. Gemma weights are not in this directory;
        ``load_snapshot`` chooses the trunk first, then calls this.
        """
        self._require_model()
        path = Path(path)
        for name in ("localizer.pt", "coord_embed.pt", "train_meta.json"):
            if not (path / name).is_file():
                raise FileNotFoundError(f"load_localizer: missing {path / name}")
        self._load_readout_files(path)
        return self

    def _load_readout_files(self, path: Path) -> None:
        blob = torch.load(path / "localizer.pt", map_location="cpu", weights_only=True)
        self.kept_layer = int(blob["kept_layer"])
        self._build_readout()
        assert self.localizer is not None
        if int(blob["hidden_size"]) != self.localizer.hidden_size:
            raise RuntimeError(
                f"localizer hidden_size {blob['hidden_size']} != "
                f"model {self.localizer.hidden_size}"
            )
        for key in ("dim", "refine_layers", "heads", "mlp_dim"):
            if int(blob[key]) != getattr(self.localizer, key):
                raise RuntimeError(
                    f"localizer {key} is {blob[key]}, "
                    f"this build has {getattr(self.localizer, key)}"
                )
        self.localizer.load_state_dict(blob["state_dict"])
        self.coord_embed.load(path / "coord_embed.pt")
        device = next(self.vl.model.parameters()).device
        self.localizer.to(device=device, dtype=torch.float32)
        self.coord_embed.to(device=device, dtype=torch.float32)

    def save_all(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.save_gemma(path / "gemma")
        self._save_readout(path)
        _write_json(path / "train_meta.json", {
            "trainer": "save_all",
            "gemma_base": None,
            "kept_layer": self.kept_layer,
            "model_key": self.model_key,
        })

    def load_all(self, path: str | Path) -> None:
        path = Path(path)
        self.load_gemma(path / "gemma")
        self.load_localizer(path)

    def save_snapshot(
        self,
        path: str | Path,
        *,
        gemma_base: str | None,
        trainer: str,
        extra: dict | None = None,
    ) -> None:
        """Write the Localizer bundle. Gemma weights only when unnamed."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._save_readout(path)
        meta = {
            "trainer": trainer,
            "gemma_base": gemma_base,
            "kept_layer": self.kept_layer,
            "model_key": self.model_key,
        }
        if extra:
            meta.update(extra)
        _write_json(path / "train_meta.json", meta)
        if gemma_base is None:
            self.save_gemma(path / "gemma")

    def load_snapshot(self, path: str | Path) -> GemmaS2:
        """Load one Localizer snapshot and leave the model in eval.

        Precedence for the trunk, with no fallback between branches:
        ``gemma/gemma_meta.json``, else ``adapter_config.json`` in the
        snapshot, else ``train_meta.json``'s ``gemma_base`` under
        ``weights/<model_key>/``. A directory that only has
        ``coord_head.pt`` is the sep24 linear head and is rejected.
        """
        path = Path(path)
        if (path / "coord_head.pt").is_file() and not (path / "localizer.pt").is_file():
            raise RuntimeError(
                f"{path} is a sep24 linear-head snapshot (coord_head.pt); "
                "the Localizer format needs localizer.pt"
            )
        for name in ("localizer.pt", "coord_embed.pt", "train_meta.json"):
            if not (path / name).is_file():
                raise FileNotFoundError(f"load_snapshot: missing {path / name}")
        meta = json.loads((path / "train_meta.json").read_text(encoding="utf-8"))
        if (path / "gemma" / "gemma_meta.json").is_file():
            self.load_gemma(path / "gemma")
        elif (path / "adapter_config.json").is_file():
            if self.vl.model is None:
                self.load()
            self._unwrap_adapter()
            from peft import PeftModel

            self.vl.model = PeftModel.from_pretrained(
                self.vl.model, str(path), is_trainable=False
            )
            self.vl.model.eval()
        elif meta.get("gemma_base"):
            from neural_net.paths import weights_root

            base = weights_root() / self.model_key / str(meta["gemma_base"])
            if not (base / "adapter_config.json").is_file():
                raise FileNotFoundError(
                    f"load_snapshot: gemma_base {meta['gemma_base']!r} "
                    f"has no adapter_config.json under {base}"
                )
            if self.vl.model is None:
                self.load()
            self._unwrap_adapter()
            from peft import PeftModel

            self.vl.model = PeftModel.from_pretrained(
                self.vl.model, str(base), is_trainable=False
            )
            self.vl.model.eval()
        else:
            raise RuntimeError(
                f"load_snapshot: {path} has no gemma/, no adapter_config.json, "
                "and train_meta.json has no gemma_base"
            )
        self.load_localizer(path)
        return self

    def _unwrap_adapter(self) -> None:
        """Drop a previously applied PEFT wrapper. Base weights stay as loaded."""
        from peft import PeftModel

        model = self.vl.model
        if not isinstance(model, PeftModel):
            return
        if self.coord_embed is not None:
            self.coord_embed.uninstall()
        if self._tap is not None:
            self._tap.remove()
            self._tap = None
        self.vl.model = model.get_base_model()
        self.vl.model.eval()
