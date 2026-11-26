"""Analyst-only model catalog.

Play and training stay on the Gemma registry in :mod:`agent.model`.
Nothing here is added to ``MODEL_REGISTRY`` and nothing here replaces
the process-wide default model.

Local entries load base Hugging Face weights. API entries call OpenAI
or Anthropic with the key already in the repo-root ``.env``.
"""

from __future__ import annotations

import base64
import gc
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from . import run_logging
from .config import CONFIG, AgentConfig
from .model import FamilyAdapter, RegexStopCriteria

logger = logging.getLogger(__name__)

# 5.24.1 is the first patch that has inkling2_loop. That release also
# still has qwen3_5 (Qwen3.8-27B) and gemma4 (Gemma 4 31B). 5.8.0 is an
# earlier release: it has qwen3_5 and gemma4, and it does not have
# inkling2_loop, which landed in 5.24.0.
MIN_TRANSFORMERS = "5.24.1"
_TRANSFORMERS_SPEC = f"transformers>={MIN_TRANSFORMERS},<5.25"

_IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "auto": "auto",
}


@dataclass(frozen=True)
class AnalystModelSpec:
    """One row of the analyst dropdown."""

    key: str
    label: str
    kind: str
    model_id: str
    env_key: str | None = None


def _spec(
    key: str,
    label: str,
    kind: str,
    model_id: str,
    env_key: str | None = None,
) -> AnalystModelSpec:
    return AnalystModelSpec(
        key=key, label=label, kind=kind, model_id=model_id, env_key=env_key,
    )


#: Insertion order is the top of the analyst dropdown. The first row is
#: the default. Weight checkpoints are not in this dict; they are
#: appended by :func:`analyst_menu_options`.
ANALYST_MODELS: dict[str, AnalystModelSpec] = {
    spec.key: spec for spec in (
        _spec(
            "inkling-2-mini", "Inkling 2 Mini", "local",
            "thinkingmachines/Inkling-2-Mini",
        ),
        _spec(
            "qwen3.8-27b", "Qwen3.8 27B", "local",
            "Qwen/Qwen3.8-27B",
        ),
        _spec(
            "gemma-4-12b", "Gemma 4 12B", "local",
            "google/gemma-4-12B-it",
        ),
        _spec(
            "gemma-4-31b", "Gemma 4 31B", "local",
            "google/gemma-4-31B-it",
        ),
        _spec(
            "gpt-6-astra", "GPT-6 Astra", "openai",
            "gpt-6-astra", env_key="OPENAI_API_KEY",
        ),
        _spec(
            "claude-fable-5-1", "Claude Fable 5.1", "anthropic",
            "claude-fable-5-1", env_key="ANTHROPIC_API_KEY",
        ),
    )
}


def analyst_menu_options() -> list[tuple[str, str]]:
    """Dropdown rows: the catalog, then Gemma trunks under ``weights/``.

    Weight rows are newest first. Their value is the directory
    :meth:`agent.analyst.FailureAnalyst.load_gemma_dir` opens. An empty
    weights directory adds no rows.
    """
    from neural_net.weight_catalog import list_analyst_models

    options = [(spec.label, spec.key) for spec in ANALYST_MODELS.values()]
    seen = {value for _, value in options}
    for text, path in list_analyst_models():
        value = str(path)
        if value in seen:
            raise RuntimeError(
                f"weight checkpoint {value!r} collides with a catalog key"
            )
        seen.add(value)
        options.append((text, value))
    return options


def analyst_model_spec(key: str) -> AnalystModelSpec:
    """Loud, exact lookup. An unknown key is a hard error."""
    try:
        return ANALYST_MODELS[key]
    except KeyError:
        raise KeyError(
            f"Unknown analyst model {key!r}. "
            f"Known keys: {list(ANALYST_MODELS)}"
        ) from None


def build_analyst_model(
    key: str, cfg: AgentConfig | None = None,
) -> "LocalAnalystModel | ApiAnalystModel":
    """An unloaded model for ``key``. The caller calls :meth:`load`."""
    spec = analyst_model_spec(key)
    if spec.kind == "local":
        return LocalAnalystModel(spec, cfg)
    if spec.kind in ("openai", "anthropic"):
        return ApiAnalystModel(spec, cfg)
    raise RuntimeError(
        f"analyst model {spec.key} has unknown kind {spec.kind!r}"
    )


class LocalAnalystModel:
    """One unfinetuned Hugging Face multimodal checkpoint.

    ``generate`` matches the call :meth:`agent.analyst.FailureAnalyst.ask`
    already makes: messages, ``max_new_tokens``, ``stop_regex``, and a
    decoded string back.
    """

    def __init__(
        self, spec: AnalystModelSpec, cfg: AgentConfig | None = None,
    ):
        if spec.kind != "local":
            raise RuntimeError(
                f"{spec.key} is {spec.kind}, not a local checkpoint"
            )
        self.spec = spec
        self.cfg = cfg or CONFIG
        self.key = spec.key
        self.label = spec.label
        self.model_id = spec.model_id
        self.hf_id = spec.model_id
        self.model: Any = None
        self.processor: Any = None
        self._loaded = False
        self._images = FamilyAdapter()

    def load(self) -> "LocalAnalystModel":
        if self._loaded:
            return self
        import transformers
        from packaging.version import Version

        installed = transformers.__version__
        if Version(installed) < Version(MIN_TRANSFORMERS):
            raise RuntimeError(
                f"{self.label} ({self.model_id}) requires transformers "
                f">= {MIN_TRANSFORMERS}, but {installed} is installed. "
                f"Run: pip install -U '{_TRANSFORMERS_SPEC}'"
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
        logger.info("Loading %s %s (dtype=%s)", self.label, self.model_id, dtype)
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                token=self.cfg.hf_token or None,
            )
            self.model = AutoModelForMultimodalLM.from_pretrained(
                self.model_id, **kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {self.label} ({self.model_id}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self.model.eval()
        self._loaded = True
        logger.info("%s loaded.", self.label)
        return self

    def unload(self) -> None:
        if self.model is not None:
            logger.info("Unloading %s.", self.label)
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
                model=f"{self.key} ({self.model_id})",
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


class ApiAnalystModel:
    """OpenAI or Anthropic. No weights, no GPU.

    The key is read from the environment at :meth:`load`. A missing key
    raises with the variable name. The value is never logged.
    """

    def __init__(
        self, spec: AnalystModelSpec, cfg: AgentConfig | None = None,
    ):
        if spec.kind not in ("openai", "anthropic"):
            raise RuntimeError(
                f"{spec.key} is {spec.kind}, not an API model"
            )
        if not spec.env_key:
            raise RuntimeError(f"{spec.key} has no env_key")
        self.spec = spec
        self.cfg = cfg or CONFIG
        self.key = spec.key
        self.label = spec.label
        self.model_id = spec.model_id
        self.hf_id = spec.model_id
        self.client: Any = None
        self._loaded = False

    def load(self) -> "ApiAnalystModel":
        if self._loaded:
            return self
        env_key = self.spec.env_key or ""
        secret = os.environ.get(env_key) or ""
        if not secret.strip():
            raise RuntimeError(
                f"{self.label} needs {env_key} in the repo-root .env"
            )
        if self.spec.kind == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=secret)
        else:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=secret)
        self._loaded = True
        logger.info("API client ready for %s.", self.label)
        return self

    def unload(self) -> None:
        if self._loaded:
            logger.info("Dropping API client for %s.", self.label)
        self.client = None
        self._loaded = False

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int | None = None,
        stop_regex: str | None = None,
    ) -> str:
        """One visible reply. Stops once ``stop_regex`` matches.

        Reasoning channels are not part of the string. An empty visible
        answer raises.
        """
        if not self._loaded:
            self.load()
        limit = max_new_tokens or self.cfg.max_new_tokens
        pattern = re.compile(stop_regex) if stop_regex else None
        reply: str | None = None
        err: str | None = None
        try:
            if self.spec.kind == "openai":
                text = self._openai(messages, limit, pattern)
            else:
                text = self._anthropic(messages, limit, pattern)
            if not text.strip():
                raise RuntimeError(f"{self.label} returned no visible text")
            reply = text.strip()
            return reply
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            run_logging.log_llm_call(
                model=f"{self.key} ({self.model_id})",
                kind="generate",
                request={"messages": messages, "rendered_prompt": None},
                params={
                    "max_new_tokens": limit,
                    "do_sample": self.cfg.do_sample,
                    "temperature": self.cfg.temperature,
                    "top_p": self.cfg.top_p,
                    "stop_regex": stop_regex,
                },
                response=None if reply is None else {"raw": reply},
                error=err,
            )

    def _sampling(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if not self.cfg.do_sample:
            params["temperature"] = 0
            return params
        if self.cfg.temperature is not None:
            params["temperature"] = self.cfg.temperature
        if self.cfg.top_p is not None:
            params["top_p"] = self.cfg.top_p
        if self.spec.kind == "anthropic" and self.cfg.top_k is not None:
            params["top_k"] = self.cfg.top_k
        return params

    def _openai(
        self,
        messages: list[dict],
        limit: int,
        pattern: re.Pattern[str] | None,
    ) -> str:
        stream = self.client.chat.completions.create(
            model=self.model_id,
            messages=_openai_messages(messages),
            max_completion_tokens=limit,
            stream=True,
            **self._sampling(),
        )
        chunks: list[str] = []
        try:
            for event in stream:
                if not event.choices:
                    continue
                delta = event.choices[0].delta
                piece = getattr(delta, "content", None) or ""
                if not isinstance(piece, str) or not piece:
                    continue
                chunks.append(piece)
                cut = _until_tool("".join(chunks), pattern)
                if cut is not None:
                    return cut
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
        return "".join(chunks)

    def _anthropic(
        self,
        messages: list[dict],
        limit: int,
        pattern: re.Pattern[str] | None,
    ) -> str:
        system, payload = _anthropic_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": limit,
            "messages": payload,
        }
        if system:
            kwargs["system"] = system
        kwargs.update(self._sampling())
        chunks: list[str] = []
        with self.client.messages.stream(**kwargs) as stream:
            for piece in stream.text_stream:
                if not isinstance(piece, str) or not piece:
                    continue
                chunks.append(piece)
                cut = _until_tool("".join(chunks), pattern)
                if cut is not None:
                    return cut
        return "".join(chunks)


def _until_tool(text: str, pattern: "re.Pattern[str] | None") -> str | None:
    """Text through the first tool token, or None while the token is open."""
    if pattern is None:
        return None
    match = pattern.search(text)
    if match is None:
        return None
    return text[:match.end()]


def _parts(content: Any) -> list[tuple[str, str]]:
    if isinstance(content, str):
        return [("text", content)]
    if not isinstance(content, list):
        raise RuntimeError(
            f"unsupported message content: {type(content).__name__}"
        )
    out: list[tuple[str, str]] = []
    for part in content:
        if isinstance(part, str):
            out.append(("text", part))
            continue
        if not isinstance(part, dict):
            raise RuntimeError(
                f"unsupported content part: {type(part).__name__}"
            )
        kind = part.get("type")
        if kind == "text":
            out.append(("text", str(part.get("text") or "")))
        elif kind == "image":
            url = part.get("url") or part.get("path")
            if not url:
                raise RuntimeError("image part has no url")
            out.append(("image", str(url)))
        else:
            raise RuntimeError(f"unsupported content part {kind!r}")
    return out


def _image_file(url: str) -> tuple[str, str]:
    src = url[len("file://"):] if url.startswith("file://") else url
    path = Path(src)
    if not path.is_file():
        raise FileNotFoundError(
            f"image url/path is not an existing local file: {url!r}"
        )
    media = _IMAGE_TYPES.get(path.suffix.lower())
    if media is None:
        raise RuntimeError(f"unsupported analyst image type: {path.suffix!r}")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return media, data


def _openai_messages(messages: list[dict]) -> list[dict]:
    converted: list[dict] = []
    for message in messages:
        role = message.get("role")
        if role not in ("system", "user", "assistant"):
            raise RuntimeError(f"unsupported message role {role!r}")
        content: list[dict] = []
        for kind, value in _parts(message.get("content")):
            if kind == "text":
                content.append({"type": "text", "text": value})
            else:
                media, data = _image_file(value)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media};base64,{data}"},
                })
        converted.append({"role": role, "content": content})
    return converted


def _anthropic_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    system_parts: list[str] = []
    payload: list[dict] = []
    for message in messages:
        role = message.get("role")
        parts = _parts(message.get("content"))
        if role == "system":
            for kind, value in parts:
                if kind != "text":
                    raise RuntimeError(
                        "Claude system messages cannot carry images"
                    )
                system_parts.append(value)
            continue
        if role not in ("user", "assistant"):
            raise RuntimeError(f"unsupported message role {role!r}")
        content: list[dict] = []
        for kind, value in parts:
            if kind == "text":
                content.append({"type": "text", "text": value})
            else:
                media, data = _image_file(value)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media,
                        "data": data,
                    },
                })
        payload.append({"role": role, "content": content})
    if not payload:
        raise RuntimeError("Claude request has no user message")
    return "\n\n".join(part for part in system_parts if part), payload
