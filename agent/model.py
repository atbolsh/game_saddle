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

CHECKPOINTS. A checkpoint is a PEFT adapter folder produced by
training/train.py, living at ``weights/<architecture-key>/<checkpoint-name>/``
(see training/TRAINING_OVERVIEW.md). Base weights always come from
HuggingFace; loading a
checkpoint stacks the adapter on top. ``checkpoint=None`` (everywhere) means
bare HF weights -- the notebooks' "[default]". Selection paths: the
MODEL_CHECKPOINT .env var, ``agent.runner --checkpoint``
(:func:`set_default_checkpoint`), and the notebooks' checkpoint dropdown.

The model is loaded once per process and shared across modes; switching
models (:func:`switch_default`) unloads the old weights from the GPU first.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
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


# ============================================================== checkpoints

def checkpoint_dir(
    architecture_key: str, checkpoint: str, cfg: AgentConfig | None = None
) -> "Path":
    """Filesystem location of one named checkpoint (no existence check)."""
    cfg = cfg or CONFIG
    return Path(cfg.weights_dir) / architecture_key / checkpoint


def list_checkpoints(
    architecture_key: str, cfg: AgentConfig | None = None
) -> list[str]:
    """Names of every saved adapter checkpoint for one architecture, newest
    first (mtime order -- the natural order for a dropdown). A checkpoint is
    any direct subfolder of ``weights/<key>/`` containing an
    ``adapter_config.json``; anything else in the tree is ignored."""
    cfg = cfg or CONFIG
    root = Path(cfg.weights_dir) / architecture_key
    if not root.is_dir():
        return []
    found = [
        d for d in root.iterdir()
        if d.is_dir() and (d / "adapter_config.json").is_file()
    ]
    found.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return [d.name for d in found]


# ========================================================== family adapters

class FamilyAdapter:
    """Per-family conventions. Currently just message normalization; future
    model families with different quirks get their own subclass here."""

    @staticmethod
    def _materialize_image_part(part: dict) -> dict:
        """Normalize one ``type=image`` content part for the HF processor.

        ``apply_chat_template`` pulls values from keys
        ``image`` / ``url`` / ``path`` / ``base64`` and passes them to
        ``load_image_as_tensor``, which accepts http(s) URLs, local paths
        (via ``os.path.isfile``), base64, or a PIL image. We materialize
        *local* paths to PIL ourselves under the ``image`` key (and drop
        ``url``/``path`` so the extractor does not also enqueue the string
        and double-count the frame). Reasons:

        * A missing/deleted path otherwise falls through to the base64
          branch and raises the misleading ``Incorrect padding`` error.
        * Passing PIL is the format HF's own processor tests use and is
          immune to cwd / temp-dir lifetime surprises between record and
          collate time.
        """
        from PIL import Image

        # Already a loaded image / tensor -- leave alone (strip string keys
        # so we don't also enqueue a path/url for the same part).
        if part.get("image") is not None and not isinstance(part["image"], str):
            return {k: v for k, v in part.items()
                    if k not in ("url", "path", "base64")}

        src = part.get("url") or part.get("path") or part.get("image")
        if src is None and part.get("base64"):
            return part  # let the processor decode
        if not isinstance(src, str):
            raise TypeError(
                f"image content part has no usable source "
                f"(keys={sorted(part)}); expected url/path/image/base64"
            )
        if src.startswith("file://"):
            src = src[len("file://"):]
        if (src.startswith("http://") or src.startswith("https://")
                or src.startswith("data:")):
            return {**part, "url": src}

        path = Path(src)
        if not path.is_file():
            raise FileNotFoundError(
                f"image url/path is not an existing local file: {src!r}"
            )
        with Image.open(path) as im:
            pil = im.convert("RGB").copy()
        # Only the PIL object -- no url/path, or the chat extractor would
        # collect BOTH and ask the processor for two images.
        return {"type": "image", "image": pil}

    def prepare_messages(self, messages: list[dict]) -> list[dict]:
        """Normalize messages for this family: materialize local image paths
        to PIL (see :meth:`_materialize_image_part`). Override per family
        only if a processor rejects the standard HF content format."""
        norm: list[dict] = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                new_content = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        part = self._materialize_image_part(part)
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

    Per-row aware: returns a bool tensor of shape ``[batch]``, so in a
    batched generate each row halts individually on its own match while the
    others continue (transformers ORs per-row criteria into its
    unfinished-sequences mask). With LEFT padding all prompts end at the
    same index, so a single ``prompt_len`` is correct for every row.
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

    def __call__(
        self, input_ids: torch.LongTensor, scores: Any, **kwargs: Any
    ) -> torch.Tensor:
        done = torch.zeros(
            input_ids.shape[0], dtype=torch.bool, device=input_ids.device
        )
        # Only consider generated tokens (not the prompt, which may
        # legitimately contain tool-call examples).
        if input_ids.shape[1] <= self.prompt_len:
            return done
        for i in range(input_ids.shape[0]):
            tail = input_ids[i][self.prompt_len:][-self.TAIL_TOKENS:]
            text = self.tokenizer.decode(tail, skip_special_tokens=True)
            done[i] = self.pattern.search(text) is not None
        return done


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
    """Thin sync wrapper around one registry model (spec-parameterized).

    ``checkpoint`` (optional) names a trained PEFT adapter under
    ``weights/<spec.key>/`` to stack on the HF base weights; None = bare HF.
    """

    def __init__(
        self,
        spec: ModelSpec,
        cfg: AgentConfig | None = None,
        checkpoint: str | None = None,
    ):
        self.spec = spec
        self.adapter = ADAPTERS[spec.family]
        self.cfg = cfg or CONFIG
        self.checkpoint = checkpoint or None
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
        if self.checkpoint:
            self._apply_checkpoint()
        self.model.eval()
        self._loaded = True
        logger.info(
            "Model %s loaded%s.", spec.key,
            f" with checkpoint {self.checkpoint!r}" if self.checkpoint else "",
        )
        return self

    def _apply_checkpoint(self) -> None:
        """Stack the named PEFT adapter on the freshly loaded base. Missing
        or malformed folders are hard errors (no-fuzzy-fallbacks): silently
        running the bare base when a checkpoint was requested would poison
        every conclusion drawn from the session."""
        path = checkpoint_dir(self.spec.key, self.checkpoint, self.cfg)
        if not (path / "adapter_config.json").is_file():
            raise RuntimeError(
                f"Checkpoint {self.checkpoint!r} for {self.spec.key!r} not "
                f"found (no adapter_config.json under {path}). Known "
                f"checkpoints: {list_checkpoints(self.spec.key, self.cfg)}"
            )
        from peft import PeftModel

        logger.info("Applying checkpoint %s", path)
        try:
            self.model = PeftModel.from_pretrained(
                self.model, str(path), is_trainable=False
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to apply checkpoint {self.checkpoint!r} "
                f"({path}) onto {self.spec.key!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

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

    # ------------------------------------------------------ batched generate
    def generate_batch(
        self,
        batch: list[dict],
        max_new_tokens: int | None = None,
        stop_strings: list[str] | None = None,
        stop_regex: str | None = None,
    ) -> list[str]:
        """One batched generation over several prompts (the parallel-datagen
        fast path: decode is memory-bandwidth-bound, so a batch of N costs
        barely more than a batch of 1).

        ``batch`` is a list of ``{"messages": [...]}`` dicts. The stopping
        knobs are BATCH-WIDE by contract -- a per-row stop set would let one
        row halt on ANOTHER row's stop string (an analyst quoting
        ``[FORWARD]`` mid-analysis would be truncated), so the dispatcher
        (agent/parallel_gen.py) groups requests by identical stop signature
        and this method refuses mixed image counts, which processors pad
        inconsistently. Prompts are LEFT-padded, so every row's generation
        starts at the same index; rows finish individually (per-row stop
        strings / regex / eos)."""
        if not batch:
            return []
        if len(batch) == 1:
            return [self.generate(
                batch[0]["messages"], max_new_tokens=max_new_tokens,
                stop_strings=stop_strings, stop_regex=stop_regex,
            )]
        if not self._loaded:
            self.load()

        def _n_images(messages: list[dict]) -> int:
            return sum(
                1
                for m in messages
                for part in (m.get("content") or [])
                if isinstance(part, dict) and part.get("type") == "image"
            )

        image_counts = {_n_images(r["messages"]) for r in batch}
        if len(image_counts) > 1:
            raise ValueError(
                "generate_batch: mixed image counts in one batch "
                f"({sorted(image_counts)}) -- group requests by image count "
                "(see agent/parallel_gen.py)"
            )

        norm = [self.adapter.prepare_messages(r["messages"]) for r in batch]
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        prev_side = getattr(tokenizer, "padding_side", "right")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        try:
            inputs = self.processor.apply_chat_template(
                norm,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
        finally:
            tokenizer.padding_side = prev_side
        target_device = next(self.model.parameters()).device
        inputs = inputs.to(target_device)
        model_dtype = next(self.model.parameters()).dtype
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                inputs[k] = v.to(model_dtype)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.cfg.max_new_tokens,
            **self._sampling_kwargs(),
        }
        prompt_len = inputs["input_ids"].shape[-1]
        if stop_strings:
            gen_kwargs["stop_strings"] = stop_strings
            gen_kwargs["tokenizer"] = tokenizer
        if stop_regex:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([
                RegexStopCriteria(stop_regex, tokenizer, prompt_len=prompt_len)
            ])

        replies: list[str] | None = None
        err: str | None = None
        try:
            with torch.inference_mode():
                out = self.model.generate(**inputs, **gen_kwargs)
            replies = [
                self.processor.decode(
                    out[i][prompt_len:], skip_special_tokens=True
                ).strip()
                for i in range(len(batch))
            ]
            return replies
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            for i, r in enumerate(batch):
                run_logging.log_llm_call(
                    model=f"{self.spec.key} ({self.spec.hf_id})",
                    kind="generate_batch",
                    request={"messages": r["messages"],
                             "batch_size": len(batch), "batch_index": i},
                    params={
                        "max_new_tokens": gen_kwargs.get("max_new_tokens"),
                        "do_sample": gen_kwargs.get("do_sample"),
                        "temperature": gen_kwargs.get("temperature"),
                        "top_p": gen_kwargs.get("top_p"),
                        "top_k": gen_kwargs.get("top_k"),
                        "stop_strings": stop_strings,
                        "stop_regex": stop_regex,
                    },
                    response=None if replies is None else {"raw": replies[i]},
                    error=err,
                )


# ======================================================= process singleton

_DEFAULT: VLModel | None = None

#: Process-wide checkpoint override (set by ``agent.runner --checkpoint``
#: BEFORE the first get_model call). Wins over cfg.model_checkpoint; the
#: sentinel distinguishes "never set" from an explicit None (= bare HF).
_CHECKPOINT_UNSET = object()
_checkpoint_override: Any = _CHECKPOINT_UNSET


def set_default_checkpoint(checkpoint: str | None) -> None:
    """Force the checkpoint the process-wide model loads with, overriding
    the MODEL_CHECKPOINT env var. Call before the first :func:`get_model`
    (an already-loaded model is NOT reloaded -- use :func:`switch_default`)."""
    global _checkpoint_override
    _checkpoint_override = checkpoint or None


def _default_checkpoint(cfg: AgentConfig) -> str | None:
    if _checkpoint_override is not _CHECKPOINT_UNSET:
        return _checkpoint_override
    return cfg.model_checkpoint


def get_model(cfg: AgentConfig | None = None) -> VLModel:
    """The process-wide model (loads ``cfg.model_key`` on first call, with
    the checkpoint from --checkpoint / MODEL_CHECKPOINT if any)."""
    global _DEFAULT
    if _DEFAULT is None:
        cfg = cfg or CONFIG
        _DEFAULT = VLModel(
            spec_for(cfg.model_key), cfg, checkpoint=_default_checkpoint(cfg)
        ).load()
    return _DEFAULT


def switch_default(
    key: str,
    cfg: AgentConfig | None = None,
    checkpoint: str | None = None,
) -> VLModel:
    """Replace the process-wide model: unload the old weights from the GPU,
    then load (downloading if needed) the registry model ``key``, stacking
    the named adapter ``checkpoint`` if given (None = bare HF)."""
    global _DEFAULT
    if _DEFAULT is not None:
        _DEFAULT.unload()
        _DEFAULT = None
    _DEFAULT = VLModel(spec_for(key), cfg, checkpoint=checkpoint).load()
    return _DEFAULT


def switch_session_model(
    session: Any,
    key: str,
    purge_others: bool = False,
    checkpoint: str | None = None,
) -> dict[str, Any]:
    """Shared implementation behind the sessions' ``switch_model`` methods.

    ``session`` is any object with ``model`` / ``cfg`` / ``restart()`` (both
    ``InteractiveSession`` and ``DebriefSession`` qualify). ``checkpoint``
    names a trained adapter under ``weights/<key>/`` (None = bare HF, the
    notebooks' "[default]"). Sequence:

      1. If ``purge_others`` ("save only one set of weights at a time"):
         restart the conversation FIRST, so the old thread never mixes with
         the new model's output.
      2. Unload the current model from the GPU.
      3. If ``purge_others``: delete every OTHER registry model's cached
         weights from disk (including the one just unloaded). Adapter
         checkpoints under weights/ are never touched.
      4. Load (downloading if needed) the new model + adapter and rebind it.
    """
    info: dict[str, Any] = {
        "key": key, "checkpoint": checkpoint or None,
        "restarted": False, "purge": None,
    }
    if purge_others:
        info["restart"] = session.restart()
        info["restarted"] = True
    if session.model is not None:
        session.model.unload()
    if purge_others:
        info["purge"] = purge_other_weights(key)
    session.model = switch_default(key, session.cfg, checkpoint=checkpoint)
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
