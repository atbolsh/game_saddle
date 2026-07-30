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


# ====================================================== generate batching

def stack_equal_length(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Stack per-row encodings that already share one sequence length.

    Gemma 4 Unified: left-padding a multimodal row corrupts its prefill at
    SPECIFIC pad lengths (measured: pad=7 on a 282-token image prompt gives
    ~40-logit deltas with the argmax flipping to ``<audio|>``; pads 1-6, 8,
    9, 15, 16, 63, 64 stay within ~0.5 bf16 wobble; batch size and
    transformers version are irrelevant -- see
    scripts/gemma4_pad_batch_repro.py, reported upstream as
    https://github.com/huggingface/transformers/issues/47651). Which
    offsets are poisonous is not predictable from the outside, so the
    only safe policy is ZERO padding:
    ``generate_batch`` only stacks equal-length rows, and mixed lengths
    here are a hard error.
    """
    if not rows:
        return {}
    row_dicts = [
        {k: v for k, v in r.items() if isinstance(v, torch.Tensor)}
        for r in rows
    ]
    seq_lens = {int(d["input_ids"].shape[1]) for d in row_dicts}
    if len(seq_lens) != 1:
        raise ValueError(
            f"stack_equal_length: mixed lengths {sorted(seq_lens)}"
        )
    if len(row_dicts) == 1:
        return dict(row_dicts[0])
    key_sets = {frozenset(d) for d in row_dicts}
    if len(key_sets) > 1:
        raise ValueError(
            f"stack_equal_length: mixed keys {sorted(key_sets)}"
        )
    batch: dict[str, Any] = {}
    for key in row_dicts[0]:
        vals = [d[key] for d in row_dicts]
        if all(v.shape[0] == 1 for v in vals):
            trailing = {tuple(v.shape[1:]) for v in vals}
            if len(trailing) > 1:
                raise ValueError(
                    f"stack_equal_length: key {key!r} trailing mismatch "
                    f"{sorted(trailing)}"
                )
            batch[key] = torch.cat(vals, dim=0)
            continue
        raise ValueError(
            f"stack_equal_length: don't know how to stack {key!r} "
            f"shape {tuple(vals[0].shape)}"
        )
    return batch


# ===========================================================================
# !!! KNOWN TRANSFORMERS BUG WORKAROUND -- READ BEFORE TOUCHING ANY OF THE !!!
# !!! PADDED-BATCH CODE BELOW (left_pad_row / left_pad_stack /            !!!
# !!! VLModel._plan_padded_batch / VLModel.generate_batch)                !!!
#
# Gemma 4 Unified has an upstream bug: a LEFT-PADDED MULTIMODAL PREFILL is
# CATASTROPHICALLY CORRUPTED in (at least) TWO measured modes:
#
#   POISON MODE 1 -- padded TOTAL width ~= 1 mod 32
#         (PAD_POISON_MOD / PAD_POISON_RESIDUE): the whole target width is
#         poisoned. Measured 6/6 across prompt lengths 282..2867 in
#         training/probe_hacky_pads.py (logs probe_hacky_pads_2026-07-30_*),
#         ~35-52-logit deltas, argmax flips to junk like '<audio|>'.
#   POISON MODE 2 -- a row whose OWN unpadded length ~= 1 mod 32 is
#         corrupted by ANY left pad, at every rule-clean target width.
#         Measured in the 2026-07-30 t6 run: an L=289 row rejected at ALL
#         of T=290..297 (~24.8-logit deltas, argmax flips) while its
#         L=282 batchmates stayed clean at the same widths. The probe
#         never hit this mode (its prompt lengths had residues 11, 30, 7,
#         15, 19 -- none ~= 1). NOTE such a row is mathematically
#         unpaddable: pad=0 needs T = L ~= 1 mod 32, which is mode 1.
#
# Every clean width/row shows only deterministic bf16 kernel wobble
# (<= ~1.75 at game-size prompts, no argmax flips -- the same class of
# noise equal-length batching has always had). Decode steps crossing the
# residue mid-generation are HARMLESS (probe test 4): the poison is
# PREFILL-ONLY, almost certainly in the padded image-feature scatter /
# chunked mask construction, which never re-runs during decode.
#
# Upstream reports (check these before "simplifying" any of this away):
#   * https://github.com/huggingface/transformers/issues/47651
#   * https://huggingface.co/google/gemma-4-12B-it/discussions/50
# Standalone repro: scripts/gemma4_pad_batch_repro.py
#
# THE SCOTCH-TAPE FIX (HACKY_ANSWER):
#   (0) POISON MODE 2 RESCUE, in VLModel._nudge_unpaddable + generate_batch:
#       a row with L ~= 1 mod 32 gets ONE harmless filler token appended to
#       its last text part, moving it off the residue so it can join the
#       padded batch. Probe test 5 (probe_hacky_pads_2026-07-30_19-46-07)
#       measured a natural-residue game prompt poisoned at 32/32 pads,
#       while the nudged copy (" ." appended, L 1569 -> 1570) padded
#       cleanly at every rule-clean pad and produced a content-identical
#       greedy reply. DELIBERATE TRADE: the nudge is serving-stack-only --
#       datagen traces record the UN-nudged messages, so on ~1/32 rows the
#       trained prompt lacks the one trailing filler token inference saw.
#       If no filler moves the length (template normalization), the row
#       falls back to a natural-width cohort (WARNING).
#   In VLModel._plan_padded_batch:
#   (1) any row still on an unpaddable length is excluded and decodes in
#       natural-width equal-length cohorts (always safe);
#   (2) the remaining rows are left-padded to the longest of THEIR lengths
#       (which is ~= 1 mod 32 by construction impossible -- mode 1 dodged);
#   (3) before any decode, the prefill-parity tripwire: each row's
#       padded-batch prefill logits must match its solo prefill logits
#       within PAD_PARITY_DLOGIT with no argmax flip. Rows that fail
#       (a THIRD poison mode we have not met yet => WARNING) are demoted
#       to cohorts individually; the survivors decode as one batch.
# Remove all of this ONLY when the upstream bug is fixed AND
# training/probe_hacky_pads.py passes on the fixed version with padding
# to total % 32 == 1.
# ===========================================================================

#: The poisoned residue class (both modes): padded TOTAL widths with
#: total % PAD_POISON_MOD == PAD_POISON_RESIDUE are poisoned targets, and
#: rows whose OWN length L % PAD_POISON_MOD == PAD_POISON_RESIDUE are
#: unpaddable (corrupted by any left pad).
PAD_POISON_MOD = 32
PAD_POISON_RESIDUE = 1

#: Prefill-parity verdict: a padded row is POISONED if its prefill argmax
#: flips or any logit moves by more than this vs its solo prefill. The
#: measured populations are far apart (wobble <= ~1.75, poison >= ~24.7);
#: 8.0 sits in the empty middle.
PAD_PARITY_DLOGIT = 8.0

#: POISON MODE 2 RESCUE fillers, tried in order until one moves the row's
#: length off the poison residue. " ." is the measured winner (probe test
#: 5); "\n" is listed last because the chat template swallowed it there.
PAD_NUDGE_FILLERS: tuple[str, ...] = (" .", " Okay.", "\n")


def left_pad_row(
    enc: dict[str, Any], pad_len: int, pad_token_id: int
) -> dict[str, Any]:
    """Left-pad one batch-1 encoding by ``pad_len`` positions.

    Every sequence-aligned integer tensor (2D, second dim == the sequence
    length) is padded on the left -- ``input_ids`` with the pad token,
    everything else (attention_mask, token_type_ids, mm_token_type_ids,
    ...) with 0. Other entries (e.g. ``pixel_values``) pass through
    untouched. Semantics lifted from training/probe_pad_divergence.py,
    where the padded tensors were verified suffix-identical to the solo
    encoding.

    WARNING: on Gemma 4 Unified a left-padded multimodal prefill is
    CORRUPTED at padded totals ~= 1 mod 32 AND for rows whose own length
    is ~= 1 mod 32 (see the KNOWN TRANSFORMERS BUG WORKAROUND banner
    above). Never decode from a padded batch except through
    :meth:`VLModel._plan_padded_batch`, which dodges both modes and runs
    the prefill-parity check.
    """
    if pad_len <= 0:
        return dict(enc)
    seq_len = int(enc["input_ids"].shape[1])
    out: dict[str, Any] = {}
    for k, v in enc.items():
        if (isinstance(v, torch.Tensor) and not v.dtype.is_floating_point
                and v.dim() == 2 and v.shape[1] == seq_len):
            fill = pad_token_id if k == "input_ids" else 0
            out[k] = torch.cat(
                [v.new_full((v.shape[0], pad_len), fill), v], dim=1
            )
        else:
            out[k] = v
    return out


def left_pad_stack(
    rows: list[dict[str, Any]],
    pad_token_id: int,
    target_len: int | None = None,
) -> tuple[dict[str, Any], list[int]]:
    """Left-pad variable-length batch-1 encodings to one common length and
    stack them. Returns ``(batch, pads)`` where ``pads[i]`` is row i's pad
    amount; ``target_len`` defaults to the longest row (zero pad there).

    This is the HACKY_ANSWER path: padding is only trustworthy when the
    target dodges the poisoned widths and the batch passes the
    prefill-parity check -- see the KNOWN TRANSFORMERS BUG WORKAROUND
    banner above and :meth:`VLModel._plan_padded_batch`.
    """
    lens = [int(r["input_ids"].shape[1]) for r in rows]
    target = max(lens) if target_len is None else target_len
    if target < max(lens):
        raise ValueError(f"target_len {target} < longest row {max(lens)}")
    pads = [target - n for n in lens]
    padded = [left_pad_row(r, p, pad_token_id) for r, p in zip(rows, pads)]
    return stack_equal_length(padded), pads


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
    unfinished-sequences mask). Batches are equal-length (no padding), so a
    single ``prompt_len`` is correct for every row.
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
            # padding_side=left is required for batched decoder-only generate
            # (HF Gemma 4 docs); training builds its own processor and
            # right-pads manually in Collator.
            self.processor = AutoProcessor.from_pretrained(
                spec.hf_id,
                token=self.cfg.hf_token or None,
                trust_remote_code=spec.trust_remote_code,
                padding_side="left",
            )
            self.model = AutoModelForMultimodalLM.from_pretrained(spec.hf_id, **kwargs)
        except Exception as exc:
            # No fallback loaders (no-fuzzy-fallbacks): name the model so the
            # failure is actionable, then re-raise.
            raise RuntimeError(
                f"Failed to load model {spec.key!r} ({spec.hf_id}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        # Belt-and-suspenders: some processor builds only honor the tokenizer
        # attribute, not the from_pretrained kwarg.
        tok = getattr(self.processor, "tokenizer", self.processor)
        if hasattr(tok, "padding_side"):
            tok.padding_side = "left"
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

    # ---------------------------------------------------- encode / device
    def encode_messages(self, messages: list[dict]) -> dict[str, Any]:
        """One prompt -> processor tensors (batch dim 1), same path as
        :meth:`generate`. Used by :meth:`generate_batch` so batched rows are
        byte-identical to solo encodings before left-pad collate."""
        if not self._loaded:
            self.load()
        norm = self.adapter.prepare_messages(messages)
        # NOTE: keep this call in lockstep with training's Collator.build
        # (training/train.py) -- the trained prompt must stay byte-identical
        # to this one. Any new template kwarg goes in BOTH places.
        return self.processor.apply_chat_template(
            norm,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

    def prefill_last_logits(self, enc: dict[str, Any]) -> torch.Tensor:
        """Next-token logits after prefilling ``enc`` (any batch size).

        Mirrors generate's prefill: position_ids derive from the attention
        mask (pad positions clamped to 0, real tokens 0..L-1). Returns
        ``[batch, vocab]`` float32 on CPU. This is the probe/parity-check
        primitive: comparing a row's padded-batch prefill logits against
        its solo prefill logits detects the Gemma 4 left-pad corruption
        (transformers#47651) exactly, before any decode happens.
        """
        if not self._loaded:
            self.load()
        inputs = self._move_inputs_to_model(enc)
        mask = inputs["attention_mask"]
        inputs["position_ids"] = (mask.long().cumsum(-1) - 1).clamp(min=0)
        with torch.inference_mode():
            # logits_to_keep=1: only the last position's logits are needed,
            # and materializing the full [batch, seq, ~262k-vocab] tensor
            # costs GIGABYTES of transient VRAM at game-size prompts (the
            # parity check would then dominate batch-3 memory). A renamed/
            # removed kwarg fails loudly with TypeError -- by design, never
            # silently fall back to full logits.
            out = self.model(**inputs, logits_to_keep=1)
        return out.logits[:, -1].float().cpu()

    def _move_inputs_to_model(self, inputs: dict[str, Any]) -> dict[str, Any]:
        target_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        out: dict[str, Any] = {}
        for k, v in inputs.items():
            if not isinstance(v, torch.Tensor):
                continue
            v = v.to(target_device)
            if v.dtype.is_floating_point:
                v = v.to(model_dtype)
            out[k] = v
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
        inputs = self._move_inputs_to_model(self.encode_messages(messages))
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
    def _generate_stacked(
        self,
        inputs: dict[str, Any],
        n_rows: int,
        *,
        max_new_tokens: int | None,
        stop_strings: list[str] | None,
        stop_regex: str | None,
    ) -> list[str]:
        """Decode one prepared (device-resident) stacked batch.

        ``prompt_len`` is the stacked width; with left-padded rows every
        generated token still lands after that index for EVERY row, so the
        decode slicing and RegexStopCriteria's single prompt_len stay
        correct whether the rows were equal-length or verified-padded."""
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
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
        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        return [
            self.processor.decode(
                out[i][prompt_len:], skip_special_tokens=True
            ).strip()
            for i in range(n_rows)
        ]

    def _generate_equal_length_batch(
        self,
        rows: list[dict[str, Any]],
        *,
        max_new_tokens: int | None,
        stop_strings: list[str] | None,
        stop_regex: str | None,
    ) -> list[str]:
        """True GPU batch over encodings that already share one seq length."""
        return self._generate_stacked(
            self._move_inputs_to_model(stack_equal_length(rows)),
            len(rows),
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            stop_regex=stop_regex,
        )

    def _nudge_unpaddable(
        self, messages: list[dict]
    ) -> tuple[list[dict], dict[str, Any]] | None:
        """KNOWN TRANSFORMERS BUG WORKAROUND, POISON MODE 2 RESCUE (module
        banner): move a naturally-unpaddable prompt (L ~= 1 mod 32) off
        the poison residue by appending one harmless filler token to its
        last text part.

        Returns ``(nudged_messages, encoding)`` with a paddable length,
        or None if no filler shifts it (no text part, or the chat
        template normalizes every filler away). Probe test 5 measured the
        nudged row padding cleanly at every rule-clean pad with a
        content-identical greedy reply.
        """
        import copy

        for filler in PAD_NUDGE_FILLERS:
            msgs = copy.deepcopy(messages)
            part = None
            for m in reversed(msgs):
                content = m.get("content")
                if isinstance(content, list):
                    for p in reversed(content):
                        if isinstance(p, dict) and p.get("type") == "text":
                            part = p
                            break
                if part is not None:
                    break
            if part is None:
                return None
            part["text"] += filler
            enc = self.encode_messages(msgs)
            if (int(enc["input_ids"].shape[1]) % PAD_POISON_MOD
                    != PAD_POISON_RESIDUE):
                return msgs, enc
        return None

    def _plan_padded_batch(
        self, rows: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], list[int], list[int]] | None:
        """KNOWN TRANSFORMERS BUG WORKAROUND (see module banner): plan ONE
        verified left-padded batch over as many rows as the bug allows.

        Returns ``(device_inputs, row_indices, pads)`` -- the rows NOT in
        ``row_indices`` must decode via equal-length cohorts -- or None if
        padding cannot beat cohorts (fewer than 2 paddable rows, or the
        paddable rows already share one length).

        Defenses, all mandatory (evidence in the module banner):

        1. Rows whose OWN length ~= PAD_POISON_RESIDUE mod PAD_POISON_MOD
           are excluded up front: poison mode 2 corrupts them under ANY
           left pad (t6 2026-07-30: L=289 rejected at every T in
           290..297), and pad=0 would need a mode-1 target width.
        2. The target width is the longest remaining row's length, which
           by construction is NOT ~= 1 mod 32 -- poison mode 1 dodged
           without any search.
        3. The prefill-parity tripwire: each padded row's prefill logits
           must match its solo prefill (no argmax flip, max delta <=
           PAD_PARITY_DLOGIT). Rows that fail are a poison mode we have
           NOT catalogued yet -- logged at WARNING and demoted to
           cohorts individually; the survivors are re-planned and must
           pass before any decode.

        Steady-state cost: one solo prefill per paddable row (what serial
        would have spent anyway) plus one batched prefill per planning
        round (exactly one when no new poison mode fires) -- small next
        to the decode loop this unlocks batching for.
        """
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            logger.warning(
                "generate_batch: tokenizer has no pad token -- cannot "
                "pad, falling back to equal-length cohorts"
            )
            return None
        lens = [int(r["input_ids"].shape[1]) for r in rows]
        cand = [i for i in range(len(rows))
                if lens[i] % PAD_POISON_MOD != PAD_POISON_RESIDUE]
        excluded = [i for i in range(len(rows)) if i not in cand]
        if excluded:
            logger.info(
                "generate_batch: row(s) %s have UNPADDABLE length(s) %s "
                "(L %% %d == %d corrupts under any left pad, poison mode "
                "2 of transformers#47651) -> natural-width cohorts",
                excluded, [lens[i] for i in excluded],
                PAD_POISON_MOD, PAD_POISON_RESIDUE,
            )
        solo: dict[int, torch.Tensor] = {}
        while True:
            if len(cand) < 2 or len({lens[i] for i in cand}) < 2:
                return None  # cohorts already handle this optimally
            for i in cand:
                if i not in solo:
                    solo[i] = self.prefill_last_logits(rows[i])[0]
            target = max(lens[i] for i in cand)
            stacked, pads = left_pad_stack(
                [rows[i] for i in cand], pad_id, target_len=target
            )
            batch_logits = self.prefill_last_logits(stacked)
            bad: list[tuple[int, float, bool]] = []
            for j, i in enumerate(cand):
                delta = float((batch_logits[j] - solo[i]).abs().max())
                flipped = (int(batch_logits[j].argmax())
                           != int(solo[i].argmax()))
                if flipped or delta > PAD_PARITY_DLOGIT:
                    bad.append((i, round(delta, 3), flipped))
            if not bad:
                logger.info(
                    "generate_batch: padded batch verified clean at T=%d "
                    "(rows %s, pads %s)", target, cand, pads,
                )
                return self._move_inputs_to_model(stacked), list(cand), pads
            logger.warning(
                "generate_batch: prefill parity check REJECTED row(s) at "
                "T=%d, (row, max|dLogit|, argmax_flip): %s -- an "
                "UNCATALOGUED poison mode of transformers#47651 (not "
                "total %% 32 == 1, not L %% 32 == 1); demoting them to "
                "natural-width cohorts", target, bad,
            )
            rejected = {b[0] for b in bad}
            cand = [i for i in cand if i not in rejected]

    def generate_batch(
        self,
        batch: list[dict],
        max_new_tokens: int | None = None,
        stop_strings: list[str] | None = None,
        stop_regex: str | None = None,
    ) -> list[str]:
        """Batched generation for parallel datagen.

        ``batch`` is a list of ``{"messages": [...]}`` dicts. Stopping knobs
        are BATCH-WIDE (dispatcher groups by identical stop signature). Mixed
        image counts are refused.

        KNOWN TRANSFORMERS BUG WORKAROUND in play here (module banner,
        transformers#47651): mixed-length rows ARE left-padded into one
        true GPU batch. Naturally-unpaddable rows (L ~= 1 mod 32, poison
        mode 2) are first NUDGED off the residue with one harmless
        filler token (:meth:`_nudge_unpaddable`; serving-stack-only, the
        caller's messages are not altered). Then :meth:`_plan_padded_batch`
        excludes any row still unpaddable, pads to a width that dodges
        poison mode 1 (total ~= 1 mod 32), and verifies every padded
        row's prefill against its solo prefill before any decode. Rows
        the plan excludes or rejects decode via one batch per distinct
        prompt length (zero pad -- always safe, but decode serializes
        across cohorts).
        """
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

        rows = [self.encode_messages(r["messages"]) for r in batch]
        lens = [int(r["input_ids"].shape[1]) for r in rows]

        # POISON MODE 2 RESCUE (module banner): a row whose OWN length is
        # ~= 1 mod 32 cannot be left-padded at all (transformers#47651),
        # which would demote it to a solo cohort. One harmless filler
        # token moves it off the residue; probe test 5 measured the
        # nudged row padding cleanly. Only mixed-length batches pad, so
        # equal-length batches are left untouched. The nudge is
        # SERVING-STACK-ONLY: traces record the un-nudged messages.
        nudged_rows: list[int] = []
        if len(set(lens)) > 1:
            for i, length in enumerate(lens):
                if length % PAD_POISON_MOD != PAD_POISON_RESIDUE:
                    continue
                nudge = self._nudge_unpaddable(batch[i]["messages"])
                if nudge is None:
                    logger.warning(
                        "generate_batch: row %d (len %d ~= %d mod %d, "
                        "UNPADDABLE poison mode 2) could not be nudged "
                        "off the residue -- it will decode in its own "
                        "cohort", i, length,
                        PAD_POISON_RESIDUE, PAD_POISON_MOD,
                    )
                    continue
                rows[i] = nudge[1]
                lens[i] = int(rows[i]["input_ids"].shape[1])
                nudged_rows.append(i)
                logger.info(
                    "generate_batch: row %d nudged %d -> %d tokens "
                    "(POISON MODE 2 RESCUE, transformers#47651: length "
                    "~= 1 mod 32 corrupts under any left pad)",
                    i, length, lens[i],
                )

        by_len: dict[int, list[int]] = {}
        for i, length in enumerate(lens):
            by_len.setdefault(length, []).append(i)

        # Always log the batch structure: this is THE datagen-throughput
        # signal (mode=cohorts with cohorts of 1 means the parallel
        # sessions decode serially).
        logger.info(
            "generate_batch: %d rows -> %d equal-length cohort(s) %s",
            len(batch), len(by_len),
            {length: len(idx) for length, idx in sorted(by_len.items())},
        )

        replies: list[str | None] = [None] * len(batch)
        err: str | None = None
        mode = "cohorts"
        try:
            plan = (self._plan_padded_batch(rows)
                    if len(by_len) > 1 else None)
            done: set[int] = set()
            if plan is not None:
                inputs, idxs, pads = plan
                mode = (f"padded(T={int(inputs['input_ids'].shape[-1])},"
                        f"rows={idxs},pads={pads})")
                sub_replies = self._generate_stacked(
                    inputs, len(idxs),
                    max_new_tokens=max_new_tokens,
                    stop_strings=stop_strings,
                    stop_regex=stop_regex,
                )
                for i, reply in zip(idxs, sub_replies):
                    replies[i] = reply
                done.update(idxs)
            leftover = {
                length: [i for i in indices if i not in done]
                for length, indices in by_len.items()
            }
            leftover = {L: idx for L, idx in leftover.items() if idx}
            if leftover and plan is not None:
                mode += "+cohorts"
            for _length, indices in leftover.items():
                sub_rows = [rows[i] for i in indices]
                sub_replies = self._generate_equal_length_batch(
                    sub_rows,
                    max_new_tokens=max_new_tokens,
                    stop_strings=stop_strings,
                    stop_regex=stop_regex,
                )
                for i, reply in zip(indices, sub_replies):
                    replies[i] = reply
            assert all(r is not None for r in replies)
            return list(replies)  # type: ignore[arg-type]
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
                        "max_new_tokens": max_new_tokens or self.cfg.max_new_tokens,
                        "do_sample": self._sampling_kwargs().get("do_sample"),
                        "stop_strings": stop_strings,
                        "stop_regex": stop_regex,
                        "seq_len": int(rows[i]["input_ids"].shape[1]),
                        # Rows whose prompt got the POISON MODE 2 RESCUE
                        # filler; the logged messages are the UN-nudged
                        # originals (module banner).
                        "nudged_rows": nudged_rows,
                        "batch_mode": mode,
                        "length_cohorts": {
                            str(k): len(v) for k, v in by_len.items()
                        },
                    },
                    response=None if replies[i] is None else {"raw": replies[i]},
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
