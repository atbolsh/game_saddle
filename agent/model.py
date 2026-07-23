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
to, in recommendation order (see MODEL_CANDIDATES.md). Each
:class:`ModelSpec` carries the HF repo id plus the family conventions the
wrapper needs (trust_remote_code, thinking-tag protocol, sampling defaults).

FAMILY ADAPTERS. Per-family quirks live in ONE place each
(:data:`ADAPTERS`): message normalization and the think-block protocol.

THINKING MODELS. Several registry models emit reasoning inside think tags
(``<think>...</think>``, Kimi's ``◁think▷...◁/think▷``). The harness must
never parse a move token from inside a think block, so:

  * Generation-time stopping is GATED: stop patterns only match after the
    close tag (or anywhere, if the reply provably has no think block). If the
    close tag never arrives, generation runs to its natural end.
  * :meth:`VLModel.generate` returns a :class:`ModelReply` -- a ``str`` of
    the VISIBLE text only (thinking stripped), so every existing caller
    parses/persists exactly what a user would see. The stripped thinking is
    kept on the reply (``.thinking``) and in the run logs.
  * Small thinking models sometimes FORGET the close tag. Per the harness
    contract: if the close tag is missing entirely, the full raw text is the
    visible text (so an intended move token is still accepted), and
    ``.missing_think_close`` is set -- sessions surface it as a FORMAT ERROR
    (same pattern as ``bare_move``).

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
    #: whether the model emits think blocks the harness must strip/gate.
    thinking: bool = False
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


#: All switchable models, in recommendation order (top = default; see
#: MODEL_CANDIDATES.md for the reasoning). dict preserves insertion order.
MODEL_REGISTRY: dict[str, ModelSpec] = {s.key: s for s in [
    ModelSpec(
        key="gemma-4-e4b", label="Gemma 4 E4B (current baseline)",
        hf_id="google/gemma-4-E4B-it", family="gemma",
        temperature=1.0, top_p=0.95, top_k=64,
        notes="4.5B effective dense; the model the harness was built on.",
    ),
    ModelSpec(
        key="qwen3-vl-8b-thinking", label="Qwen3-VL 8B Thinking",
        hf_id="Qwen/Qwen3-VL-8B-Thinking", family="qwen", thinking=True,
        notes="~9B dense; grounding-RL lineage + reasoning RL. Top pick.",
    ),
    ModelSpec(
        key="qwen3-vl-8b-instruct", label="Qwen3-VL 8B Instruct",
        hf_id="Qwen/Qwen3-VL-8B-Instruct", family="qwen",
        notes="~9B dense; same base as the Thinking variant, no think blocks.",
    ),
    ModelSpec(
        key="gemma-4-12b", label="Gemma 4 12B Unified",
        hf_id="google/gemma-4-12B-it", family="gemma",
        min_transformers="5.10.0",  # gemma4_unified arch added in 5.10.0
        temperature=1.0, top_p=0.95, top_k=64,
        notes="12B dense, encoder-free unified architecture, 256K context.",
    ),
    ModelSpec(
        key="glm-4.1v-9b-thinking", label="GLM-4.1V 9B Thinking",
        hf_id="zai-org/GLM-4.1V-9B-Thinking", family="glm", thinking=True,
        notes="9B dense; curriculum-RL reasoner (RLCS), strong grounding.",
    ),
    ModelSpec(
        key="step3-vl-10b", label="Step3-VL 10B",
        hf_id="stepfun-ai/Step3-VL-10B", family="step",
        trust_remote_code=True, thinking=True,
        notes="10B; RLVR+RLHF. Custom code (auto_map) -- first remote load "
              "may need loader attention.",
    ),
    ModelSpec(
        key="phi-4-reasoning-vision-15b", label="Phi-4 Reasoning Vision 15B",
        hf_id="microsoft/Phi-4-reasoning-vision-15B", family="phi",
        trust_remote_code=True, thinking=True,
        notes="15B dense; optional think blocks. 16K context -- tight for "
              "long debriefs. Custom code (auto_map).",
    ),
    ModelSpec(
        key="kimi-vl-a3b-thinking-2506", label="Kimi-VL A3B Thinking (2506)",
        hf_id="moonshotai/Kimi-VL-A3B-Thinking-2506", family="kimi",
        trust_remote_code=True, thinking=True, temperature=0.8,
        notes="16B total / 2.8B active MoE; native-resolution encoder. "
              "Moonshot recommends temperature 0.8. Custom code (auto_map).",
    ),
    ModelSpec(
        key="mimo-vl-7b-rl", label="MiMo-VL 7B RL",
        hf_id="XiaomiMiMo/MiMo-VL-7B-RL", family="mimo", thinking=True,
        notes="7B dense on the Qwen2.5-VL architecture (no custom code); "
              "thinks by default.",
    ),
    ModelSpec(
        key="internvl3.5-8b", label="InternVL3.5 8B",
        hf_id="OpenGVLab/InternVL3_5-8B", family="internvl",
        trust_remote_code=True,
        notes="8B dense; Cascade-RL. Custom code (InternVLChatModel) -- "
              "first remote load may need loader attention.",
    ),
    ModelSpec(
        key="internvl3.5-14b", label="InternVL3.5 14B",
        hf_id="OpenGVLab/InternVL3_5-14B", family="internvl",
        trust_remote_code=True,
        notes="14B dense; strongest dense InternVL under 20B. Custom code.",
    ),
    ModelSpec(
        key="ovis2.5-9b", label="Ovis2.5 9B",
        hf_id="ATH-MaaS/Ovis2.5-9B", family="ovis",
        trust_remote_code=True,
        notes="9B; native-resolution ViT. Repo moved from AIDC-AI to "
              "ATH-MaaS. Custom code likely.",
    ),
    ModelSpec(
        key="qwen3-vl-30b-a3b-thinking", label="Qwen3-VL 30B-A3B Thinking (QLoRA-only FT)",
        hf_id="Qwen/Qwen3-VL-30B-A3B-Thinking", family="qwen", thinking=True,
        notes="30B total / 3B active MoE; ~60GB bf16 weights.",
    ),
    ModelSpec(
        key="gemma-4-26b-a4b", label="Gemma 4 26B-A4B (QLoRA-only FT)",
        hf_id="google/gemma-4-26B-A4B-it", family="gemma",
        temperature=1.0, top_p=0.95, top_k=64,
        notes="26B total / 4B active MoE; stays inside the Gemma harness.",
    ),
    ModelSpec(
        key="internvl3.5-30b-a3b", label="InternVL3.5 30B-A3B (QLoRA-only FT)",
        hf_id="OpenGVLab/InternVL3_5-30B-A3B", family="internvl",
        trust_remote_code=True,
        notes="30B total / 3B active MoE. Custom code.",
    ),
]}

DEFAULT_MODEL_KEY = "gemma-4-e4b"


def spec_for(key: str) -> ModelSpec:
    """Loud, exact lookup -- an unknown key is a hard error, never a guess."""
    try:
        return MODEL_REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"Unknown model key {key!r}. Known keys: {list(MODEL_REGISTRY)}"
        ) from None


# ========================================================== family adapters

class ModelReply(str):
    """The visible text of a model reply, as a plain ``str`` (so every
    existing caller keeps working), plus the think-protocol metadata:

    - ``raw``: the untouched decoded generation.
    - ``thinking``: the stripped think-block text, or None.
    - ``missing_think_close``: True when a thinking model never emitted its
      close tag (FORMAT ERROR -- the full raw text was kept visible so an
      intended move token is still accepted).
    """

    raw: str
    thinking: str | None
    missing_think_close: bool

    def __new__(
        cls, visible: str, raw: str, thinking: str | None,
        missing_think_close: bool,
    ) -> "ModelReply":
        obj = super().__new__(cls, visible)
        obj.raw = raw
        obj.thinking = thinking
        obj.missing_think_close = missing_think_close
        return obj


class FamilyAdapter:
    """Per-family conventions: message normalization + think-tag protocol.

    ``always_thinks`` marks families whose chat template auto-opens the think
    block inside the generation prompt (Qwen3 Thinking, GLM, Kimi, MiMo): the
    open tag then never appears in the OUTPUT, and every reply is expected to
    contain the close tag. Families with optional, explicitly-opened blocks
    (Phi, Step) keep ``always_thinks=False``.
    """

    def __init__(
        self,
        think_open: str = "<think>",
        think_close: str = "</think>",
        always_thinks: bool = False,
    ):
        self.think_open = think_open
        self.think_close = think_close
        self.always_thinks = always_thinks

    # ---------------------------------------------------------- messages
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

    # ------------------------------------------------------- think blocks
    def split_thinking(
        self, raw: str, thinking_model: bool
    ) -> tuple[str, str | None, bool]:
        """Split a decoded reply into ``(visible, thinking, missing_close)``.

        Contract (see module docstring): tokens inside a think block never
        count; a missing close tag keeps the FULL text visible (so an
        intended move is still accepted) and flags a FORMAT ERROR -- except
        when the reply provably never opened a block at all (optional
        thinkers answering directly), which is a plain reply."""
        if not thinking_model:
            return raw, None, False
        if self.think_close in raw:
            thinking, _, visible = raw.partition(self.think_close)
            thinking = thinking.replace(self.think_open, "", 1).strip()
            return visible.strip(), thinking, False
        if self.always_thinks or self.think_open in raw:
            # A think block was (or must have been) opened and never closed.
            return raw.strip(), None, True
        return raw, None, False  # no block at all: a plain reply

    def stop_region(self, generated_text: str) -> str | None:
        """The slice of the generated-so-far text where stop patterns may
        legitimately match, or None if stopping must wait (we are inside a
        think block whose close tag has not arrived)."""
        if self.think_close in generated_text:
            return generated_text.split(self.think_close, 1)[1]
        if self.always_thinks or self.think_open in generated_text:
            return None
        return generated_text


class GlmAdapter(FamilyAdapter):
    """GLM-4.1V wraps its final answer in <answer> tags; strip them from the
    visible text so the harness parses clean prose."""

    def split_thinking(
        self, raw: str, thinking_model: bool
    ) -> tuple[str, str | None, bool]:
        visible, thinking, missing = super().split_thinking(raw, thinking_model)
        for tag in ("<answer>", "</answer>"):
            visible = visible.replace(tag, "")
        return visible.strip(), thinking, missing


#: One adapter instance per family. always_thinks per the family's template
#: behavior (auto-opened think block -> close tag expected in every reply).
ADAPTERS: dict[str, FamilyAdapter] = {
    "gemma": FamilyAdapter(),
    "qwen": FamilyAdapter(always_thinks=True),
    "glm": GlmAdapter(always_thinks=True),
    "step": FamilyAdapter(),
    "phi": FamilyAdapter(),
    "kimi": FamilyAdapter(
        think_open="\u25c1think\u25b7", think_close="\u25c1/think\u25b7",
        always_thinks=True,
    ),
    "mimo": FamilyAdapter(always_thinks=True),
    "internvl": FamilyAdapter(),
    "ovis": FamilyAdapter(),
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

    THINK GATE. For thinking models pass ``gate`` (the family adapter): the
    FULL generation is decoded each step and the patterns may only match in
    the region the adapter allows -- after the think-close tag, or anywhere
    if the reply provably has no think block. A move token inside an
    unterminated think block therefore never stops generation; the reply runs
    to its natural end and the missing-close salvage happens at parse time.

    ``patterns`` may be one pattern or a list. Each is compiled SEPARATELY
    and generation stops when any of them matches -- never spliced into one
    alternation, because a pattern may legally begin with a global inline
    flag like ``(?i)`` (SEARCH_TOOL_PATTERN does), which Python rejects
    anywhere but position 0 of an expression.
    """

    #: How many of the most recent generated tokens to decode per check (the
    #: ungated path). Sized generously so a junk-padded call (e.g.
    #: ``[SHOW: step 42 ]``) or a multi-word ``[SEARCH ...]`` query still fits
    #: entirely in the window -- if the opening ``[SEARCH`` scrolled out of
    #: the decoded tail before the closing ``]`` arrived, the pattern would
    #: never match and generation would run on.
    TAIL_TOKENS = 48

    def __init__(
        self,
        patterns: str | re.Pattern | list,
        tokenizer: Any,
        prompt_len: int,
        gate: FamilyAdapter | None = None,
    ):
        if isinstance(patterns, (str, re.Pattern)):
            patterns = [patterns]
        self.patterns = [
            re.compile(p) if isinstance(p, str) else p for p in patterns
        ]
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.gate = gate

    def _hit(self, text: str) -> bool:
        return any(p.search(text) for p in self.patterns)

    def __call__(self, input_ids: torch.LongTensor, scores: Any, **kwargs: Any) -> bool:
        # Only consider generated tokens (not the prompt, which may legitimately
        # contain tool-call examples).
        gen = input_ids[0][self.prompt_len:]
        if len(gen) == 0:
            return False
        if self.gate is None:
            tail = gen[-self.TAIL_TOKENS:]
            text = self.tokenizer.decode(tail, skip_special_tokens=True)
            return self._hit(text)
        # Gated: the think-close tag may be arbitrarily far back, so decode
        # the whole generation (bounded by max_new_tokens; fine at our scale).
        text = self.tokenizer.decode(gen, skip_special_tokens=True)
        region = self.gate.stop_region(text)
        return region is not None and self._hit(region)


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
    ) -> ModelReply:
        """Run one generation and return a :class:`ModelReply` (a ``str`` of
        the VISIBLE text; think blocks stripped per the family protocol).

        If ``stop_strings`` is given, generation halts as soon as any of
        those strings is emitted; the stop string is included at the tail of
        the returned text. If ``stop_regex`` is given, generation halts as
        soon as the pattern matches the decoded generated text (see
        :class:`RegexStopCriteria`) -- use this for parameterized tokens like
        ``[SHOW 42]`` that literal stop strings cannot capture. For thinking
        models BOTH go through one think-gated regex criteria so a move
        token inside a think block never halts generation. The model's native
        end-of-turn/eos still terminates generation on its own, so a reply
        that emits no stop token simply ends the turn."""
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
        if self.spec.thinking:
            # Literal stop strings + the regex all go through ONE think-gated
            # criteria (as separate patterns -- see RegexStopCriteria): nothing
            # may stop generation from inside a think block.
            parts = [re.escape(s) for s in (stop_strings or [])]
            if stop_regex:
                parts.append(stop_regex)
            if parts:
                gen_kwargs["stopping_criteria"] = StoppingCriteriaList([
                    RegexStopCriteria(
                        parts, tokenizer,
                        prompt_len=prompt_len, gate=self.adapter,
                    )
                ])
        else:
            if stop_strings:
                # StopStringCriteria requires the tokenizer to be passed to generate.
                gen_kwargs["stop_strings"] = stop_strings
                gen_kwargs["tokenizer"] = tokenizer
            if stop_regex:
                gen_kwargs["stopping_criteria"] = StoppingCriteriaList([
                    RegexStopCriteria(stop_regex, tokenizer, prompt_len=prompt_len)
                ])

        reply: ModelReply | None = None
        err: str | None = None
        try:
            with torch.inference_mode():
                out = self.model.generate(**inputs, **gen_kwargs)
            gen = out[0][prompt_len:]
            raw_out = self.processor.decode(gen, skip_special_tokens=True).strip()
            visible, thinking, missing_close = self.adapter.split_thinking(
                raw_out, self.spec.thinking
            )
            if missing_close:
                logger.warning(
                    "Model %s never closed its think block (%r missing) -- "
                    "keeping the full text visible (FORMAT ERROR).",
                    self.spec.key, self.adapter.think_close,
                )
            reply = ModelReply(visible, raw_out, thinking, missing_close)
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
                response=None if reply is None else {
                    "raw": reply.raw,
                    "thinking": reply.thinking,
                    "missing_think_close": reply.missing_think_close,
                },
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
