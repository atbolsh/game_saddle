"""Multimodal model wrapper, registry, and family adaptation layer.

One :class:`VLModel` wraps whichever registry model is active, loaded via
HuggingFace ``transformers`` (``AutoModelForMultimodalLM`` + ``AutoProcessor``).
Inputs follow the HF chat format with content lists supporting text and image
parts, e.g.::

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "..."}]},
        {"role": "user", "content": [
            {"type": "image", "url": "/path/to/frame.png"},
            {"type": "text", "text": "Make the best move."},
        ]},
    ]

REGISTRY. :data:`MODEL_REGISTRY` lists every model the notebooks can switch
to, in recommendation order. After the 2026-07 bake-off (see
MODEL_CANDIDATES.md) it holds only the two Gemma 4 variants -- the thinking
models never worked well in this harness and their think-tag plumbing was
removed -- but the registry/adapter architecture stays so future prototypes
drop in as one :class:`ModelSpec` (+ adapter, if a new family).

FAMILY ADAPTERS. Per-family quirks live in ONE place each
(:data:`ADAPTERS`): currently just message normalization.

The model is loaded once per process and shared across modes; switching
models (:func:`switch_default`) unloads the old weights from the GPU first.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any

import re

import torch
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
)

from .config import AgentConfig, CONFIG
from . import run_logging

logger = logging.getLogger(__name__)


# ================================================================= registry

@dataclass(frozen=True)
class ModelSpec:
    """One switchable model: HF repo + the conventions the wrapper needs."""

    key: str           #: stable id used in dropdowns / MODEL_KEY env
    label: str         #: human-readable dropdown label
    hf_id: str         #: HuggingFace repo id (verified 2026-07-23)
    family: str        #: adapter key into ADAPTERS
    trust_remote_code: bool = False
    #: minimum transformers release whose native code knows this architecture
    #: (None = anything satisfying requirements.txt works). Checked at load
    #: time so the failure is instant and actionable instead of a cryptic
    #: "model type not recognized" after a multi-GB download.
    min_transformers: str | None = None
    #: sampling defaults for THIS model (env overrides win; None = leave the
    #: knob to the model's own generation_config).
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    notes: str = ""


#: All switchable models, in recommendation order (top = default). dict
#: preserves insertion order.
MODEL_REGISTRY: dict[str, ModelSpec] = {s.key: s for s in [
    ModelSpec(
        key="gemma-4-12b", label="Gemma 4 12B Unified",
        hf_id="google/gemma-4-12B-it", family="gemma",
        min_transformers="5.10.0",  # gemma4_unified arch added in 5.10.0
        temperature=1.0, top_p=0.95, top_k=64,
        notes="12B dense, encoder-free unified architecture, 256K context. "
              "Won the 2026-07 bake-off: best analyst, best debrief.",
    ),
    ModelSpec(
        key="gemma-4-e4b", label="Gemma 4 E4B",
        hf_id="google/gemma-4-E4B-it", family="gemma",
        temperature=1.0, top_p=0.95, top_k=64,
        notes="4.5B effective dense; the model the harness was built on.",
    ),
]}

DEFAULT_MODEL_KEY = "gemma-4-12b"


def spec_for(key: str) -> ModelSpec:
    """Loud, exact lookup -- an unknown key is a hard error, never a guess."""
    try:
        return MODEL_REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"Unknown model key {key!r}. Known keys: {list(MODEL_REGISTRY)}"
        ) from None


# ========================================================== family adapters

class FamilyAdapter:
    """Per-family conventions. Currently just message normalization; future
    model families with different quirks get their own subclass here."""

    @staticmethod
    def _resolve_image_url(url: str) -> str:
        """Allow ``url`` to be a local filesystem path; HF processors accept
        paths directly. We also tolerate a ``file://`` prefix."""
        if url.startswith("file://"):
            return url[len("file://"):]
        return url

    def prepare_messages(self, messages: list[dict]) -> list[dict]:
        """Normalize messages for this family. Default: resolve image URLs
        (paths) in content lists; the standard HF chat format needs nothing
        else. Override per family only if a processor rejects the
        ``{"type": "image", "url": ...}`` content format."""
        norm: list[dict] = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                new_content = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        part = {**part, "url": self._resolve_image_url(part["url"])}
                    new_content.append(part)
                norm.append({**m, "content": new_content})
            else:
                norm.append(m)
        return norm


#: One adapter instance per family.
ADAPTERS: dict[str, FamilyAdapter] = {
    "gemma": FamilyAdapter(),
}


# ============================================================ stop criteria

class RegexStopCriteria(StoppingCriteria):
    """Stop generation as soon as ``pattern`` matches the decoded tail of the
    generated text.

    HF's built-in ``StopStringCriteria`` handles only literal strings, which
    cannot capture parameterized tokens like ``[SHOW 42]`` (stopping on
    ``[SHOW`` would halt before the parameter is generated). This criteria
    decodes a window of the generated tokens each step and applies a regex,
    so generation halts right after the complete call.
    """

    #: How many of the most recent generated tokens to decode per check.
    #: Sized generously so a junk-padded call (e.g. ``[SHOW: step 42 ]``) or
    #: a multi-word ``[SEARCH ...]`` query still fits entirely in the window
    #: -- if the opening ``[SEARCH`` scrolled out of the decoded tail before
    #: the closing ``]`` arrived, the pattern would never match and
    #: generation would run on.
    TAIL_TOKENS = 48

    def __init__(
        self, pattern: str | re.Pattern, tokenizer: Any, prompt_len: int
    ):
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: Any, **kwargs: Any) -> bool:
        # Only consider generated tokens (not the prompt, which may legitimately
        # contain tool-call examples).
        gen = input_ids[0][self.prompt_len:]
        if len(gen) == 0:
            return False
        tail = gen[-self.TAIL_TOKENS:]
        text = self.tokenizer.decode(tail, skip_special_tokens=True)
        return self.pattern.search(text) is not None


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "auto": "auto",
}


# ================================================================== wrapper

class VLModel:
    """Thin sync wrapper around one registry model (spec-parameterized)."""

    def __init__(self, spec: ModelSpec, cfg: AgentConfig | None = None):
        self.spec = spec
        self.adapter = ADAPTERS[spec.family]
        self.cfg = cfg or CONFIG
        self.model: Any = None
        self.processor: Any = None
        self._loaded = False

    def load(self) -> "VLModel":
        if self._loaded:
            return self
        spec = self.spec
        if spec.min_transformers is not None:
            import transformers
            from packaging.version import Version

            installed = transformers.__version__
            if Version(installed) < Version(spec.min_transformers):
                raise RuntimeError(
                    f"Model {spec.key!r} ({spec.hf_id}) requires transformers "
                    f">= {spec.min_transformers} (its architecture is not in "
                    f"older releases), but {installed} is installed. Run: "
                    f"pip install -U 'transformers>={spec.min_transformers}'"
                )
        dtype = _DTYPE_MAP.get(self.cfg.model_dtype.lower(), "auto")
        logger.info(
            "Loading model %s (%s, dtype=%s, trust_remote_code=%s)",
            spec.key, spec.hf_id, dtype, spec.trust_remote_code,
        )
        kwargs: dict[str, Any] = {
            "dtype": dtype,
            "attn_implementation": "sdpa",
        }
        if self.cfg.model_device == "auto":
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = {"": self.cfg.model_device}
        if spec.trust_remote_code:
            kwargs["trust_remote_code"] = True
        if self.cfg.hf_token:
            kwargs["token"] = self.cfg.hf_token
        try:
            self.processor = AutoProcessor.from_pretrained(
                spec.hf_id,
                token=self.cfg.hf_token or None,
                trust_remote_code=spec.trust_remote_code,
            )
            self.model = AutoModelForMultimodalLM.from_pretrained(spec.hf_id, **kwargs)
        except Exception as exc:
            # No fallback loaders (no-fuzzy-fallbacks): name the model so the
            # failure is actionable, then re-raise.
            raise RuntimeError(
                f"Failed to load model {spec.key!r} ({spec.hf_id}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self.model.eval()
        self._loaded = True
        logger.info("Model %s loaded.", spec.key)
        return self

    def unload(self) -> None:
        """Free the GPU: drop model + processor and empty the CUDA cache."""
        if self.model is not None:
            logger.info("Unloading model %s.", self.spec.key)
        self.model = None
        self.processor = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------------------------------------------------- sampling
    def _sampling_kwargs(self) -> dict[str, Any]:
        """Resolve sampling knobs: env override > spec default > the model's
        own generation_config (i.e. pass nothing)."""
        out: dict[str, Any] = {"do_sample": self.cfg.do_sample}
        if not self.cfg.do_sample:
            return out
        for name, env_val, spec_val in [
            ("temperature", self.cfg.temperature, self.spec.temperature),
            ("top_p", self.cfg.top_p, self.spec.top_p),
            ("top_k", self.cfg.top_k, self.spec.top_k),
        ]:
            val = env_val if env_val is not None else spec_val
            if val is not None:
                out[name] = val
        return out

    # ---------------------------------------------------------- generate
    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int | None = None,
        stop_strings: list[str] | None = None,
        stop_regex: str | None = None,
    ) -> str:
        """Run one generation and return the decoded reply text.

        If ``stop_strings`` is given, generation halts as soon as any of
        those strings is emitted; the stop string is included at the tail of
        the returned text. If ``stop_regex`` is given, generation halts as
        soon as the pattern matches the decoded generated text (see
        :class:`RegexStopCriteria`) -- use this for parameterized tokens like
        ``[SHOW 42]`` that literal stop strings cannot capture. The model's
        native end-of-turn/eos still terminates generation on its own, so a
        reply that emits no stop token simply ends the turn."""
        if not self._loaded:
            self.load()
        norm_messages = self.adapter.prepare_messages(messages)
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
            "max_new_tokens": max_new_tokens or self.cfg.max_new_tokens,
            **self._sampling_kwargs(),
        }
        prompt_len = inputs["input_ids"].shape[-1]
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if stop_strings:
            # StopStringCriteria requires the tokenizer to be passed to generate.
            gen_kwargs["stop_strings"] = stop_strings
            gen_kwargs["tokenizer"] = tokenizer
        if stop_regex:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([
                RegexStopCriteria(stop_regex, tokenizer, prompt_len=prompt_len)
            ])

        reply: str | None = None
        err: str | None = None
        try:
            with torch.inference_mode():
                out = self.model.generate(**inputs, **gen_kwargs)
            gen = out[0][prompt_len:]
            reply = self.processor.decode(gen, skip_special_tokens=True).strip()
            return reply
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
                model=f"{self.spec.key} ({self.spec.hf_id})",
                kind="generate",
                request={"messages": messages, "rendered_prompt": rendered},
                params={
                    "max_new_tokens": gen_kwargs.get("max_new_tokens"),
                    "do_sample": gen_kwargs.get("do_sample"),
                    "temperature": gen_kwargs.get("temperature"),
                    "top_p": gen_kwargs.get("top_p"),
                    "top_k": gen_kwargs.get("top_k"),
                    "stop_strings": stop_strings,
                    "stop_regex": stop_regex,
                },
                response=None if reply is None else {"raw": reply},
                error=err,
            )


# ======================================================= process singleton

_DEFAULT: VLModel | None = None


def get_model(cfg: AgentConfig | None = None) -> VLModel:
    """The process-wide model (loads ``cfg.model_key`` on first call)."""
    global _DEFAULT
    if _DEFAULT is None:
        cfg = cfg or CONFIG
        _DEFAULT = VLModel(spec_for(cfg.model_key), cfg).load()
    return _DEFAULT


def switch_default(key: str, cfg: AgentConfig | None = None) -> VLModel:
    """Replace the process-wide model: unload the old weights from the GPU,
    then load (downloading if needed) the registry model ``key``."""
    global _DEFAULT
    if _DEFAULT is not None:
        _DEFAULT.unload()
        _DEFAULT = None
    _DEFAULT = VLModel(spec_for(key), cfg).load()
    return _DEFAULT


def switch_session_model(
    session: Any, key: str, purge_others: bool = False
) -> dict[str, Any]:
    """Shared implementation behind the sessions' ``switch_model`` methods.

    ``session`` is any object with ``model`` / ``cfg`` / ``restart()`` (both
    ``InteractiveSession`` and ``DebriefSession`` qualify). Sequence:

      1. If ``purge_others`` ("save only one set of weights at a time"):
         restart the conversation FIRST, so the old thread never mixes with
         the new model's output.
      2. Unload the current model from the GPU.
      3. If ``purge_others``: delete every OTHER registry model's cached
         weights from disk (including the one just unloaded).
      4. Load (downloading if needed) the new model and rebind it.
    """
    info: dict[str, Any] = {"key": key, "restarted": False, "purge": None}
    if purge_others:
        info["restart"] = session.restart()
        info["restarted"] = True
    if session.model is not None:
        session.model.unload()
    if purge_others:
        info["purge"] = purge_other_weights(key)
    session.model = switch_default(key, session.cfg)
    info["hf_id"] = session.model.spec.hf_id
    info["label"] = session.model.spec.label
    return info


# ============================================================= weight purge

def purge_other_weights(keep_key: str) -> dict[str, Any]:
    """Delete the HF cache entries of every REGISTRY model except
    ``keep_key`` ("save only one set of weights at a time").

    Scoped strictly to :data:`MODEL_REGISTRY` repo ids: GLiNER/spaCy/
    sentence-transformers and any other cached repos are never touched.
    Returns ``{"purged": [repo ids], "freed_bytes": int}``."""
    from huggingface_hub import scan_cache_dir

    keep_id = spec_for(keep_key).hf_id
    registry_ids = {s.hf_id for s in MODEL_REGISTRY.values()}
    cache = scan_cache_dir()
    hashes: list[str] = []
    purged: list[str] = []
    freed = 0
    for repo in cache.repos:
        if repo.repo_type != "model":
            continue
        if repo.repo_id not in registry_ids or repo.repo_id == keep_id:
            continue
        for rev in repo.revisions:
            hashes.append(rev.commit_hash)
        purged.append(repo.repo_id)
        freed += repo.size_on_disk
    if hashes:
        strategy = cache.delete_revisions(*hashes)
        strategy.execute()
        logger.info(
            "Purged %d cached model repo(s) (%.1f GB): %s",
            len(purged), freed / 1e9, ", ".join(purged),
        )
    return {"purged": purged, "freed_bytes": freed}
