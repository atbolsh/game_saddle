"""Gemma 4 E4B multimodal model wrapper.

Loads ``google/gemma-4-E4B-it`` via HuggingFace ``transformers`` using
``AutoModelForMultimodalLM`` + ``AutoProcessor``. Inputs follow the HF
chat format with content lists supporting text and image parts, e.g.::

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "..."}]},
        {"role": "user", "content": [
            {"type": "image", "url": "/path/to/frame.png"},
            {"type": "text", "text": "Make the best move."},
        ]},
    ]

The model is loaded once per process and shared across modes.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from .config import AgentConfig, CONFIG

logger = logging.getLogger(__name__)


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "auto": "auto",
}


class Gemma4E4B:
    """Thin async-friendly (sync-under-the-hood) wrapper around Gemma 4 E4B."""

    def __init__(self, cfg: AgentConfig | None = None):
        self.cfg = cfg or CONFIG
        self.model: Any = None
        self.processor: Any = None
        self._loaded = False

    def load(self) -> "Gemma4E4B":
        if self._loaded:
            return self
        model_id = self.cfg.gemma_model_id
        dtype = _DTYPE_MAP.get(self.cfg.gemma_dtype.lower(), "auto")
        logger.info("Loading Gemma 4 E4B: %s (dtype=%s)", model_id, dtype)
        kwargs: dict[str, Any] = {
            "dtype": dtype,
            "attn_implementation": "sdpa",
        }
        if self.cfg.gemma_device == "auto":
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = {"": self.cfg.gemma_device}
        if self.cfg.hf_token:
            kwargs["token"] = self.cfg.hf_token
        self.processor = AutoProcessor.from_pretrained(model_id, token=self.cfg.hf_token or None)
        self.model = AutoModelForMultimodalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        self._loaded = True
        logger.info("Gemma 4 E4B loaded.")
        return self

    def _resolve_image_url(self, url: str) -> str:
        """Allow ``url`` to be a local filesystem path; HF processor accepts
        paths directly. We also tolerate a ``file://`` prefix."""
        if url.startswith("file://"):
            return url[len("file://"):]
        return url

    def generate(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        if not self._loaded:
            self.load()
        # Normalise image URLs (paths) in content lists.
        norm_messages: list[dict] = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                new_content = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        part = {**part, "url": self._resolve_image_url(part["url"])}
                    new_content.append(part)
                norm_messages.append({**m, "content": new_content})
            else:
                norm_messages.append(m)
        inputs = self.processor.apply_chat_template(
            norm_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Move to model device/dtype.
        target_device = next(self.model.parameters()).device
        inputs = inputs.to(target_device)
        # Pixel values / image inputs should be cast to model dtype for the
        # vision encoder; text inputs keep long.
        try:
            model_dtype = next(self.model.parameters()).dtype
            for k, v in list(inputs.items()):
                if v.dtype.is_floating_point:
                    inputs[k] = v.to(model_dtype)
        except StopIteration:
            pass
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.cfg.gemma_max_new_tokens,
                do_sample=False,
            )
        in_len = inputs["input_ids"].shape[-1]
        gen = out[0][in_len:]
        return self.processor.decode(gen, skip_special_tokens=True).strip()


_DEFAULT: Gemma4E4B | None = None


def get_model(cfg: AgentConfig | None = None) -> Gemma4E4B:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Gemma4E4B(cfg).load()
    return _DEFAULT
