"""Gemma 4 with a 6-float coordinate head on the lm_head input.

The head reads the same last-token hidden state that feeds lm_head and
emits (s_x, s_y, v_x, v_y, v_bar_x, v_bar_y) with no sigmoid. Head math
is fp32. The trunk stays on whatever dtype it was loaded with.

Save / load slices:

* save_all / load_all — Gemma slice plus the head
* save_gemma / load_gemma — Gemma only
* save_head / load_head — the linear layer only

A PEFT adapter (the aug27 checkpoints) is saved with save_pretrained.
A bare HuggingFace Gemma is saved as a full state_dict; that file is
the whole 12B and is large on purpose.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from agent.model import VLModel, spec_for


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
    ) -> None:
        self.model_key = model_key
        self.vl = VLModel(spec_for(model_key), checkpoint=checkpoint)
        self.coord_head: nn.Linear | None = None
        self._hook: Any = None
        self._last_hidden: torch.Tensor | None = None

    @property
    def hidden_size(self) -> int:
        if self.vl.model is None:
            raise RuntimeError("hidden_size called before the model was loaded")
        return _hidden_size(self.vl.model)

    def load(self) -> GemmaS2:
        """Load the HuggingFace base plus this instance's checkpoint, and
        a freshly initialized coordinate head."""
        self.vl.load()
        self._build_head()
        return self

    def _build_head(self) -> None:
        if self.vl.model is None:
            raise RuntimeError("_build_head called before the model was loaded")
        device = next(self.vl.model.parameters()).device
        size = self.hidden_size
        previous = self.coord_head
        self.coord_head = nn.Linear(size, 6).to(device=device, dtype=torch.float32)
        if previous is not None and previous.in_features == size:
            self.coord_head.load_state_dict(previous.state_dict())
        self._install_hook()

    def _install_hook(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        if self.vl.model is None:
            return
        head = _find_lm_head(self.vl.model)

        def _capture(_module: nn.Module, args: tuple, _output: Any) -> None:
            hidden = args[0]
            if hidden.dim() == 3:
                hidden = hidden[:, -1]
            elif hidden.dim() != 2:
                raise RuntimeError(
                    f"lm_head input rank {hidden.dim()}, expected 2 or 3"
                )
            self._last_hidden = hidden.detach()

        self._hook = head.register_forward_hook(_capture)

    def prefill(self, encoded: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """Cached prefill. Returns (logits [B, V], coords [B, 6], past)."""
        self._require_loaded()
        inputs = self.vl._move_inputs_to_model(encoded)
        mask = inputs["attention_mask"]
        inputs["position_ids"] = (mask.long().cumsum(-1) - 1).clamp(min=0)
        return self._forward(inputs)

    def decode(
        self,
        token_ids: torch.Tensor,
        past: Any,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """One cached decode step.

        token_ids is [B], the tokens just produced. seq_len is how many
        real tokens are already in the cache; the new token sits at that
        index.
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
        return self._forward(inputs)

    def _forward(self, inputs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, Any]:
        assert self.coord_head is not None
        self._last_hidden = None
        pos = inputs.get("position_ids")
        if pos is not None and _wants_cache_position(self.vl.model):
            inputs = dict(inputs)
            inputs["cache_position"] = pos[0]
        with torch.inference_mode():
            out = self.vl.model(**inputs, use_cache=True, logits_to_keep=1)
        if self._last_hidden is None:
            raise RuntimeError(
                "lm_head hook did not fire; the coordinate head has no input"
            )
        logits = out.logits
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        elif logits.dim() != 2:
            raise RuntimeError(f"logits rank {logits.dim()}, expected 2 or 3")
        coords = self.coord_head(self._last_hidden.float())
        past = getattr(out, "past_key_values", None)
        if past is None:
            raise RuntimeError("Gemma forward returned no past_key_values")
        return logits, coords, past

    def _require_loaded(self) -> None:
        if self.vl.model is None or self.coord_head is None or self._hook is None:
            raise RuntimeError(
                "GemmaS2 is not loaded; call load(), load_gemma(), or load_all()"
            )

    def _require_model(self) -> None:
        if self.vl.model is None:
            raise RuntimeError("Gemma is not loaded")

    def save_head(self, path: str | Path) -> None:
        if self.coord_head is None:
            raise RuntimeError("save_head called before the coordinate head exists")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.coord_head.state_dict(), path)

    def load_head(self, path: str | Path) -> None:
        self._require_model()
        if self.coord_head is None:
            self._build_head()
        assert self.coord_head is not None
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        self.coord_head.load_state_dict(state)
        device = next(self.vl.model.parameters()).device
        self.coord_head.to(device=device, dtype=torch.float32)

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

        A coordinate head that already matches the hidden size is kept.
        Otherwise a new random head is allocated; follow with load_head.
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
        previous = self.coord_head
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
        self.coord_head = previous
        self._build_head()

    def save_all(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.save_gemma(path / "gemma")
        self.save_head(path / "coord_head.pt")

    def load_all(self, path: str | Path) -> None:
        path = Path(path)
        self.load_gemma(path / "gemma")
        self.load_head(path / "coord_head.pt")
