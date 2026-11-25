"""Analyst-only loader for the unfinetuned Inkling 2 Mini.

Play and training stay on the Gemma registry in :mod:`agent.model`.
This class is not in ``MODEL_REGISTRY`` and does not replace the
process-wide default model.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import torch

from . import run_logging
from .config import CONFIG, AgentConfig
from .model import FamilyAdapter, RegexStopCriteria

logger = logging.getLogger(__name__)

HF_ID = "thinkingmachines/Inkling-2-Mini"
KEY = "inkling-2-mini"
# inkling2_loop landed in transformers 5.24.0. 5.24.1 is the patch on
# that release. Older pins in this repo do not contain the architecture.
MIN_TRANSFORMERS = "5.24.1"

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "auto": "auto",
}


class InklingAnalystModel:
    """Base Inkling 2 Mini. No adapter, no S1, no localizer.

    ``generate`` matches the call :meth:`agent.analyst.FailureAnalyst.ask`
    already makes: messages, ``max_new_tokens``, ``stop_regex``, and a
    decoded string back.
    """

    def __init__(self, cfg: AgentConfig | None = None):
        self.cfg = cfg or CONFIG
        self.hf_id = HF_ID
        self.key = KEY
        self.model: Any = None
        self.processor: Any = None
        self._loaded = False
        self._images = FamilyAdapter()

    def load(self) -> "InklingAnalystModel":
        if self._loaded:
            return self
        import transformers
        from packaging.version import Version

        installed = transformers.__version__
        if Version(installed) < Version(MIN_TRANSFORMERS):
            raise RuntimeError(
                f"Inkling 2 Mini ({HF_ID}) requires transformers "
                f">= {MIN_TRANSFORMERS} (its inkling2_loop architecture is "
                f"not in older releases), but {installed} is installed. "
                f"Run: pip install -U 'transformers>={MIN_TRANSFORMERS},<5.25'"
            )
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        dtype = _DTYPE_MAP.get(self.cfg.model_dtype.lower(), "auto")
        kwargs: dict[str, Any] = {"dtype": dtype}
        if self.cfg.model_device == "auto":
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = {"": self.cfg.model_device}
        if self.cfg.hf_token:
            kwargs["token"] = self.cfg.hf_token
        logger.info("Loading Inkling %s (dtype=%s)", HF_ID, dtype)
        try:
            self.processor = AutoProcessor.from_pretrained(
                HF_ID,
                token=self.cfg.hf_token or None,
            )
            self.model = AutoModelForMultimodalLM.from_pretrained(HF_ID, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Inkling 2 Mini ({HF_ID}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self.model.eval()
        self._loaded = True
        logger.info("Inkling %s loaded.", HF_ID)
        return self

    def unload(self) -> None:
        if self.model is not None:
            logger.info("Unloading Inkling %s.", HF_ID)
        self.model = None
        self.processor = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int | None = None,
        stop_regex: str | None = None,
    ) -> str:
        """One reply. ``stop_regex`` halts on the first analyst tool token."""
        if not self._loaded:
            self.load()
        norm = self._images.prepare_messages(messages)
        encoded = self.processor.apply_chat_template(
            norm,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = self._move_inputs(encoded)
        prompt_len = int(inputs["input_ids"].shape[-1])
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.cfg.max_new_tokens,
            "do_sample": self.cfg.do_sample,
        }
        if self.cfg.do_sample:
            if self.cfg.temperature is not None:
                gen_kwargs["temperature"] = self.cfg.temperature
            if self.cfg.top_p is not None:
                gen_kwargs["top_p"] = self.cfg.top_p
            if self.cfg.top_k is not None:
                gen_kwargs["top_k"] = self.cfg.top_k
        if stop_regex:
            from transformers import StoppingCriteriaList

            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([
                RegexStopCriteria(stop_regex, tokenizer, prompt_len=prompt_len)
            ])

        reply: str | None = None
        err: str | None = None
        try:
            with torch.inference_mode():
                out = self.model.generate(**inputs, **gen_kwargs)
            reply = self.processor.decode(
                out[0][prompt_len:], skip_special_tokens=True,
            ).strip()
            return reply
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            rendered = None
            try:
                rendered = self.processor.decode(
                    inputs["input_ids"][0], skip_special_tokens=False,
                )
            except Exception:
                rendered = None
            run_logging.log_llm_call(
                model=f"{KEY} ({HF_ID})",
                kind="generate",
                request={"messages": messages, "rendered_prompt": rendered},
                params={
                    "max_new_tokens": gen_kwargs.get("max_new_tokens"),
                    "do_sample": gen_kwargs.get("do_sample"),
                    "temperature": gen_kwargs.get("temperature"),
                    "top_p": gen_kwargs.get("top_p"),
                    "top_k": gen_kwargs.get("top_k"),
                    "stop_regex": stop_regex,
                },
                response=None if reply is None else {"raw": reply},
                error=err,
            )

    def _move_inputs(self, encoded: Any) -> dict[str, Any]:
        device = next(self.model.parameters()).device
        out: dict[str, Any] = {}
        for key, value in encoded.items():
            if isinstance(value, torch.Tensor):
                out[key] = value.to(device)
            else:
                out[key] = value
        return out
