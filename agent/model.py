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
from . import run_logging

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

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int | None = None,
        stop_strings: list[str] | None = None,
    ) -> str:
        """Run one generation. If ``stop_strings`` is given, generation halts as
        soon as any of those strings is emitted (HF ``StopStringCriteria``); the
        stop string is included at the tail of the returned text. Gemma's native
        ``<end_of_turn>`` / ``<eos>`` still terminate generation on their own
        (they are in ``model.config.eos_token_id``), so a reply that emits no
        stop string simply ends the turn."""
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
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.cfg.gemma_max_new_tokens,
            "do_sample": self.cfg.gemma_do_sample,
        }
        if self.cfg.gemma_do_sample:
            gen_kwargs["temperature"] = self.cfg.gemma_temperature
            gen_kwargs["top_p"] = self.cfg.gemma_top_p
            gen_kwargs["top_k"] = self.cfg.gemma_top_k
        if stop_strings:
            # StopStringCriteria requires the tokenizer to be passed to generate.
            gen_kwargs["stop_strings"] = stop_strings
            gen_kwargs["tokenizer"] = getattr(self.processor, "tokenizer", self.processor)

        raw_out: str | None = None
        err: str | None = None
        try:
            with torch.inference_mode():
                out = self.model.generate(**inputs, **gen_kwargs)
            in_len = inputs["input_ids"].shape[-1]
            gen = out[0][in_len:]
            raw_out = self.processor.decode(gen, skip_special_tokens=True).strip()
            return raw_out
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            # Log every generate call (input + output), on by default when a run
            # logger is active. Best-effort: never let logging break generation.
            rendered = None
            try:
                rendered = self.processor.decode(
                    inputs["input_ids"][0], skip_special_tokens=False
                )
            except Exception:
                rendered = None
            run_logging.log_llm_call(
                model=self.cfg.gemma_model_id,
                kind="generate",
                request={"messages": messages, "rendered_prompt": rendered},
                params={
                    "max_new_tokens": gen_kwargs.get("max_new_tokens"),
                    "do_sample": gen_kwargs.get("do_sample"),
                    "temperature": gen_kwargs.get("temperature"),
                    "top_p": gen_kwargs.get("top_p"),
                    "top_k": gen_kwargs.get("top_k"),
                    "stop_strings": stop_strings,
                },
                response=None if raw_out is None else {"raw": raw_out},
                error=err,
            )


_DEFAULT: Gemma4E4B | None = None


def get_model(cfg: AgentConfig | None = None) -> Gemma4E4B:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Gemma4E4B(cfg).load()
    return _DEFAULT
