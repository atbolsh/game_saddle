"""Source-agnostic QLoRA training loop for the game_saddle self-training project.

Design record (the full rationale lives in TRAINING_OVERVIEW.md; the one-line
justifications are repeated here so the file is self-explanatory):

DATA MODEL. Everything trainable is a :class:`TrainingExample`: chat
``messages`` (the same HF content-list format ``agent/model.py`` uses, text +
image parts), a ``target_text`` (the assistant reply to train on), and
optional ``span_weights`` -- ``(char_start, char_end, weight)`` triples over
``target_text``. Sources (game traces, arithmetic replay, planted errors,
video/audio sets, ...) are :class:`DataSource` objects; stage 1 ships only
:class:`JsonlSource` so the loop is testable end to end.

ONE LOSS FOR BOTH LABEL TYPES. Weighted token cross-entropy: prompt tokens
weigh 0; target tokens weigh 1.0 unless a span says otherwise (negative
weights suppress tokens). Plain SFT and RL-style per-token quality vectors
are the same code path. Per-example loss is normalized by the sum of
ABSOLUTE token weights so annotated and plain examples mix at comparable
gradient scale.

TWO LOSS KINDS. ``TrainingExample.loss`` selects "ce" (above -- teach the
dataset's targets; used where we want to STRENGTHEN, e.g. arithmetic replay)
or "kd" (knowledge distillation to the base model -- soft cross-entropy
against the frozen base's distribution over the same target positions; used
where we want to PRESERVE, e.g. general reasoning/instruction/VQA replay,
see TRAINING_EXTRA_DATASETS.md). KD needs no stored teacher:
``model.disable_adapter()`` gives the base forward in the same process, and
teacher/student softmaxes are computed over the TARGET POSITIONS ONLY
(a full-sequence float softmax at a 262k vocab would be GBs). Both kinds
use the same span weights and the same normalization.

SEQUENCE = EXACTLY WHAT INFERENCE PRODUCES. Trained ids are
``prompt_ids + content_ids + [end-of-turn]`` where ``prompt_ids`` come from
``apply_chat_template(..., add_generation_prompt=True)`` -- byte-identical to
the prefix ``VLModel.generate`` builds -- and ``content_ids`` tokenize
``target_text`` with an offset mapping (which is how char spans become token
weights). No train/inference template mismatch.

IMAGES ARE FIRST-CLASS. The model's own AutoProcessor runs over the full
messages, so batches carry pixel tensors and the forward pass is the
multimodal forward. An example that declares an image but produces no pixel
tensor is a hard error (no silent text-only degradation). Vision/audio
towers stay frozen (name-excluded from LoRA); the vision embedder
(``embed_vision`` on Gemma 4) trains via PEFT ``modules_to_save`` and so
rides inside the adapter checkpoint. Micro-batch is fixed at 1 (variable-size
multimodal padding is a swamp; effective batch comes from grad accumulation).

RECIPE (flags override): 4-bit NF4 + double quant + bf16 compute (QLoRA,
Dettmers et al. 2023); LoRA r=32/alpha=64/dropout=0.05 on all language-model
linears (all-linear beats attention-only in the QLoRA ablations); paged 8-bit
AdamW; peak LR 1e-4 (LoRA wants ~10x the full-FT LR), cosine decay to 10% of
peak with 3% warmup; weight decay 0.0 on LoRA params (decay fights the
low-rank update); grad clip 1.0; NEFTune embedding noise alpha=5 on by
default (Jain et al. 2023: consistently helps small-data SFT); fixed seed.
Deliberately omitted: label smoothing (distorts move-token confidence),
sequence packing (incompatible with per-example images + span weights),
adapter EMA (overkill at this scale).

CHECKPOINTS. Periodic PEFT-adapter saves to
``weights/<architecture>/<label>_step<K>/`` plus ``train_meta.json``. A
checkpoint is an adapter, never a full model copy; the notebooks/scripts load
it on top of the HF base via the [default]/checkpoint dropdowns.

ROLLBACK. After every save, registered eval hooks run: held-out loss on a
reserved slice, reported PER SOURCE (each source gets its own guard -- for
KD sources held-out KD loss IS the drift-from-base early-warning metric),
plus any ``extra_hooks`` the run script attaches (e.g. the exact-match
probes from ``training/probes.py``). A guarded metric regressing past its
threshold logs at ERROR and -- per ``--on-regression warn|rollback|abort``
(default rollback) -- restores the best adapter state and continues, or
aborts the run. Hooks are callables ``hook(ctx: TrainContext) -> dict`` so
they can generate, not just score.

LOGGING. ``logs/train_<label>_<stamp>/``: config.json (resolved config +
seed + git rev + discovered LoRA target modules), train_log.jsonl + .txt
(per-step loss, per-source loss, LR, grad norm), events.jsonl (saves, evals,
rollbacks), and eval_log.jsonl -- ONE FLAT ROW PER EVALUATION with every
metric (each source's held-out loss, each probe accuracy) as its own key,
so plotting any metric over training is a one-liner
(``pandas.read_json(..., lines=True)``).

NOTE (remote-environment rule): this file cannot be executed on the local
editing box (no torch/transformers/GPU). The default ``--projector-module``
(``embed_vision``) matches Gemma 4 / Gemma 4 Unified in transformers >= 5.10
(``model.embed_vision``; there is no ``multi_modal_projector``). An
unresolvable name still fails loudly with candidate module names from
``model.named_modules()``.

LIBRARY, NOT A RUN. This module is the reusable loop; it knows nothing about
any specific training run. All knobs live in the :class:`TrainConfig`
dataclass and each concrete run is a SHORT separate script that builds its
``DataSource`` list + config and calls :func:`run_training` (copy
``training/run_first_iteration.py`` per run)::

    from training.train import JsonlSource, TrainConfig, run_training
    run_training([JsonlSource("batch1.jsonl")], TrainConfig(label="iter1"))

A generic CLI front-end is kept for ad-hoc runs (from the repo root;
``python training/train.py ...`` also works -- the module inserts the repo
root into sys.path when run as a plain file)::

    python -m training.train --data batch1.jsonl batch2.jsonl:0.5 \
        [--architecture gemma-4-12b] [--label episode1] \
        [--resume-checkpoint <name>] [--epochs 1] ...

``--data`` accepts ``path`` or ``path:weight`` (mixture weight, default 1.0;
weight w resamples the source to ~w x its size per epoch); every other flag
maps 1:1 onto a TrainConfig field.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import logging
import math
import random
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

# Support ``python training/train.py`` in addition to ``python -m
# training.train``: run as a plain file, sys.path[0] is training/ and the
# ``agent`` package would not import.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("train")


# ============================================================== data model

#: Loss kinds a TrainingExample may carry: "ce" = weighted token
#: cross-entropy on the target text (strengthen); "kd" = soft cross-entropy
#: against the frozen base model's distribution (preserve).
VALID_LOSSES = ("ce", "kd")


@dataclass
class TrainingExample:
    """One trainable unit. ``messages`` is the HF chat content-list format
    (see agent/model.py); ``target_text`` is the assistant reply to train on;
    ``span_weights`` are (char_start, char_end, weight) triples over
    ``target_text`` -- absent means plain SFT (every target token weighs 1);
    ``loss`` picks the loss kind (see VALID_LOSSES)."""

    messages: list[dict]
    target_text: str
    span_weights: list[tuple[int, int, float]] | None = None
    loss: str = "ce"
    source: str = "unknown"
    meta: dict = field(default_factory=dict)

    def declares_image(self) -> bool:
        for m in self.messages:
            content = m.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        return True
        return False


class DataSource(ABC):
    """A named stream of TrainingExamples with a mixture weight.

    Future sources (game traces, replay sets, planted errors, ...) subclass
    this; train.py itself never learns anything source-specific."""

    name: str = "source"
    weight: float = 1.0

    @abstractmethod
    def examples(self) -> Iterator[TrainingExample]:
        ...


class JsonlSource(DataSource):
    """Reference DataSource: one JSON object per line with keys
    ``messages`` (required), ``target_text`` (required), ``span_weights``
    (optional, list of [start, end, weight]), ``loss`` (optional "ce"|"kd",
    default = the constructor's ``default_loss``), ``meta`` (optional).

    Malformed lines are hard errors with the line number -- a training set
    that silently drops examples is worse than one that refuses to load."""

    def __init__(self, path: str | Path, weight: float = 1.0,
                 default_loss: str = "ce"):
        self.path = Path(path)
        self.name = self.path.stem
        self.weight = weight
        if default_loss not in VALID_LOSSES:
            raise ValueError(f"JsonlSource: bad default_loss {default_loss!r}")
        self.default_loss = default_loss
        if not self.path.is_file():
            raise FileNotFoundError(f"JsonlSource: no such file: {self.path}")

    def examples(self) -> Iterator[TrainingExample]:
        with open(self.path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    messages = obj["messages"]
                    target_text = obj["target_text"]
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(
                        f"{self.path}:{lineno}: bad training example "
                        f"({type(exc).__name__}: {exc})"
                    ) from exc
                spans = obj.get("span_weights")
                if spans is not None:
                    spans = [(int(s), int(e), float(w)) for s, e, w in spans]
                loss = obj.get("loss", self.default_loss)
                if loss not in VALID_LOSSES:
                    raise ValueError(
                        f"{self.path}:{lineno}: bad loss kind {loss!r} "
                        f"(expected one of {VALID_LOSSES})"
                    )
                yield TrainingExample(
                    messages=messages,
                    target_text=target_text,
                    span_weights=spans,
                    loss=loss,
                    source=self.name,
                    meta=obj.get("meta", {}),
                )


def parse_data_arg(arg: str) -> JsonlSource:
    """``path`` or ``path:weight``. The rsplit tolerates ':' in paths only if
    the trailing segment is not a float -- ambiguous names should just be
    renamed."""
    if ":" in arg:
        head, tail = arg.rsplit(":", 1)
        try:
            return JsonlSource(head, weight=float(tail))
        except ValueError:
            pass
    return JsonlSource(arg)


# ========================================================= run configuration

@dataclass
class TrainConfig:
    """Every knob of one training run (defaults = the TRAINING_OVERVIEW.md
    recipe). Data sources are NOT part of the config -- they are live objects
    passed to :func:`run_training` alongside it. Run scripts build one of
    these directly; the CLI maps its flags onto the same fields 1:1."""

    # run identity
    label: str = "episode"          #: names the log dir and the checkpoints
    architecture: str | None = None  #: MODEL_REGISTRY key; None = MODEL_KEY env
    resume_checkpoint: str | None = None  #: name under weights/<architecture>/

    # schedule
    epochs: int = 1
    max_steps: int | None = None    #: optimizer-step cap (None = epochs decide)
    lr: float = 1e-4
    scheduler: str = "cosine"       #: "cosine" (with floor) or "constant"
    warmup_ratio: float = 0.03
    lr_floor: float = 0.10          #: cosine decays to this fraction of peak
    weight_decay: float = 0.0
    #: examples per forward pass (length-bucketed padding; see epoch_batches).
    #: Effective batch = micro_batch * grad_accum = 16 at the defaults.
    micro_batch: int = 4
    grad_accum: int = 4

    # LoRA / regularization
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    neftune_alpha: float = 5.0      #: 0 disables NEFTune embedding noise
    #: Module trained fully via modules_to_save; "none" = LoRA only.
    #: Gemma 4 / Gemma 4 Unified: ``embed_vision`` (resolves to
    #: ``model.embed_vision``). There is no ``multi_modal_projector``.
    projector_module: str = "embed_vision"

    # environment / reproducibility
    seed: int = 17
    device: str = "cuda:0"

    # data hygiene
    #: drop (LOUDLY, per-source counts) any example whose total text exceeds
    #: this many chars (~4 chars/token, so 16k chars ~ 4k tokens): bounds the
    #: per-sequence activation/logit memory without silent truncation.
    max_example_chars: int = 16000

    # cadence + safety
    log_steps: int = 10
    save_steps: int = 200
    holdout_fraction: float = 0.05
    #: held-out examples PER SOURCE are capped here -- the fraction is taken
    #: of the MATERIALIZED pool (~120k examples for the default manifest),
    #: which would otherwise make every eval a ~6k-forward, hour-long affair.
    holdout_cap: int = 100
    on_regression: str = "rollback"  #: "warn" | "rollback" | "abort"
    regression_tolerance: float = 0.10  #: relative slack before regression
    #: rollbacks allowed per run before aborting: rollback restores the best
    #: adapter but keeps training on the same schedule, so a persistently
    #: regressing run would otherwise oscillate (roll back, re-regress, ...)
    #: and burn the GPU weekend silently.
    max_rollbacks: int = 3


# ================================================================= logging

class TrainLogger:
    """One run directory under logs/: config.json, train_log.{jsonl,txt},
    events.jsonl, eval_log.jsonl. Same spirit as agent.run_logging:
    machine-readable + human-readable, and logging failures must never kill
    a run that is burning GPU-hours (they degrade to a one-time console
    warning)."""

    def __init__(self, label: str, base_dir: str | Path = "logs"):
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = Path(base_dir) / f"train_{label}_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.run_dir / "train_log.jsonl"
        self.txt = self.run_dir / "train_log.txt"
        self.events = self.run_dir / "events.jsonl"
        self.eval_jsonl = self.run_dir / "eval_log.jsonl"
        self._warned = False

    def _append(self, path: Path, text: str) -> None:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception as exc:  # logging must not break training
            if not self._warned:
                self._warned = True
                print(f"[train] logging disabled after write failure: {exc}")

    def write_config(self, config: dict) -> None:
        try:
            (self.run_dir / "config.json").write_text(
                json.dumps(config, indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:
            print(f"[train] could not write config.json: {exc}")

    def step(self, record: dict) -> None:
        record = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
                  **record}
        self._append(self.jsonl, json.dumps(record, default=str) + "\n")
        parts = [f"step {record.get('step', '?'):>6}"]
        for k in ("epoch", "loss", "lr", "grad_norm"):
            if k in record:
                v = record[k]
                parts.append(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}")
        if "source_loss" in record:
            parts.append("per-source " + json.dumps(record["source_loss"]))
        self._append(self.txt, "  ".join(parts) + "\n")

    def eval_row(self, step: int, epoch: int, metrics: dict[str, float]) -> None:
        """One FLAT row per evaluation -- every metric (each source's
        held-out loss, each probe accuracy) is its own key, so any metric's
        history over training plots in one line:
        ``pandas.read_json('eval_log.jsonl', lines=True).plot(x='step', y=...)``."""
        row = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
               "step": step, "epoch": epoch, **metrics}
        self._append(self.eval_jsonl, json.dumps(row, default=str) + "\n")

    def event(self, kind: str, **fields: Any) -> None:
        record = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
                  "kind": kind, **fields}
        self._append(self.events, json.dumps(record, default=str) + "\n")
        self._append(self.txt, f"== {kind}: "
                     + json.dumps(fields, default=str) + "\n")
        logger.info("event %s: %s", kind, fields)


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).parent, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ============================================================ collation

class Collator:
    """TrainingExample -> model inputs + per-position weight vector.

    The trained sequence is ``prompt_ids + content_ids + [end_of_turn]``:
    prompt via apply_chat_template(add_generation_prompt=True) exactly as
    VLModel.generate builds it, content via a plain tokenizer call whose
    offset mapping maps char spans onto token weights. Extra processor
    outputs (pixel_values, token_type_ids, ...) pass through: id-shaped int
    tensors are extended over the appended target tokens with fill 0,
    floating tensors are cast to the compute dtype."""

    def __init__(self, processor: Any, adapter: Any, terminator_id: int,
                 compute_dtype: Any, device: Any):
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.adapter = adapter
        self.terminator_id = terminator_id
        self.compute_dtype = compute_dtype
        self.device = device

    def build(self, ex: TrainingExample) -> dict[str, Any]:
        import torch

        norm = self.adapter.prepare_messages(ex.messages)
        prompt = self.processor.apply_chat_template(
            norm, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        if ex.declares_image() and not any(
            v.dtype.is_floating_point for v in prompt.values()
            if isinstance(v, torch.Tensor)
        ):
            raise RuntimeError(
                f"example from {ex.source!r} declares an image part but the "
                "processor produced no pixel tensor -- refusing to train "
                "text-only on an image example"
            )

        enc = self.tokenizer(
            ex.target_text, add_special_tokens=False,
            return_offsets_mapping=True,
        )
        content_ids: list[int] = enc["input_ids"]
        offsets: list[tuple[int, int]] = enc["offset_mapping"]
        if not content_ids:
            raise ValueError(f"example from {ex.source!r} has empty target_text")

        # Per-token weights over the target: default 1.0, spans override any
        # token whose char range overlaps the span. Terminator weighs 1.0
        # (emitting end-of-turn at the right moment is part of the behavior).
        content_w = [1.0] * len(content_ids)
        for (s, e, w) in ex.span_weights or []:
            for i, (ts, te) in enumerate(offsets):
                if ts < e and te > s:  # overlap
                    content_w[i] = w

        prompt_ids = prompt["input_ids"]  # [1, P]
        prompt_len = prompt_ids.shape[1]
        tail = torch.tensor([content_ids + [self.terminator_id]],
                            dtype=prompt_ids.dtype)
        input_ids = torch.cat([prompt_ids, tail], dim=1)
        n_total = input_ids.shape[1]

        weights = torch.zeros(1, n_total, dtype=torch.float32)
        weights[0, prompt_len:] = torch.tensor(content_w + [1.0])

        batch: dict[str, Any] = {"input_ids": input_ids.to(self.device)}
        for key, val in prompt.items():
            if key == "input_ids" or not isinstance(val, torch.Tensor):
                continue
            if key == "attention_mask":
                continue  # rebuilt below at full length
            if (not val.dtype.is_floating_point
                    and val.dim() == 2 and val.shape[1] == prompt_len):
                # id-shaped per-token tensor (e.g. token_type_ids): extend
                # over the target tokens with 0 (= ordinary text tokens).
                pad = torch.zeros(1, n_total - prompt_len, dtype=val.dtype)
                val = torch.cat([val, pad], dim=1)
            elif val.dtype.is_floating_point:
                val = val.to(self.compute_dtype)
            batch[key] = val.to(self.device)
        batch["attention_mask"] = torch.ones(
            1, n_total, dtype=torch.long, device=self.device
        )
        return {"model_inputs": batch, "weights": weights.to(self.device)}

    def build_batch(self, exs: list[TrainingExample]) -> dict[str, Any]:
        """Micro-batch of examples -> one padded model input.

        RIGHT padding (training never generates, so right pad + zeroed
        attention mask is correct); padded positions carry weight 0, so they
        vanish from weighted_loss's numerator and denominator. Callers batch
        via :func:`epoch_batches`, whose buckets guarantee what this method
        asserts: same loss kind and same image count (pixel tensors must
        stack; a text row cannot share a batch with an image row)."""
        import torch

        builds = [self.build(ex) for ex in exs]
        if len(builds) == 1:
            return builds[0]

        key_sets = {tuple(sorted(b["model_inputs"])) for b in builds}
        if len(key_sets) > 1:
            raise ValueError(
                "build_batch: examples produced different model-input keys "
                f"({sorted(key_sets)}) -- the bucketing (epoch_batches) must "
                "keep image and text examples apart"
            )
        max_len = max(b["model_inputs"]["input_ids"].shape[1] for b in builds)
        pad_id = getattr(self.tokenizer, "pad_token_id", None) or 0

        def pad_to(t: Any, fill: Any) -> Any:  # [1, L] -> [1, max_len]
            if t.shape[1] == max_len:
                return t
            pad = torch.full((1, max_len - t.shape[1]), fill,
                             dtype=t.dtype, device=t.device)
            return torch.cat([t, pad], dim=1)

        batch: dict[str, Any] = {
            "input_ids": torch.cat(
                [pad_to(b["model_inputs"]["input_ids"], pad_id)
                 for b in builds]
            ),
            "attention_mask": torch.cat(
                [pad_to(b["model_inputs"]["attention_mask"], 0)
                 for b in builds]
            ),
        }
        for key in builds[0]["model_inputs"]:
            if key in batch:
                continue
            vals = [b["model_inputs"][key] for b in builds]
            if (not vals[0].dtype.is_floating_point and vals[0].dim() == 2):
                # id-shaped per-token tensor: pad like input_ids, fill 0.
                batch[key] = torch.cat([pad_to(v, 0) for v in vals])
            elif vals[0].dtype.is_floating_point:
                # pixel tensors: stack images in example order (the model
                # consumes them in image-token order, which is row-major
                # over the batch).
                batch[key] = torch.cat(vals, dim=0)
            else:
                raise ValueError(
                    f"build_batch: don't know how to batch key {key!r} "
                    f"(shape {tuple(vals[0].shape)}, dtype {vals[0].dtype})"
                )
        weights = torch.cat([pad_to(b["weights"], 0.0) for b in builds])
        return {"model_inputs": batch, "weights": weights}


def resolve_terminator_id(model: Any, tokenizer: Any) -> int:
    """The token that ends an assistant turn, i.e. what a successful
    generation emits last. Resolution order (documented, not fuzzy): the
    tokenizer's literal '<end_of_turn>' (the Gemma convention this harness
    stops on), else the model generation_config's (first) eos, else the
    tokenizer eos. No candidate at all is a hard error."""
    tok_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    unk = getattr(tokenizer, "unk_token_id", None)
    if isinstance(tok_id, int) and tok_id >= 0 and tok_id != unk:
        return tok_id
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(eos, (list, tuple)) and eos:
        return int(eos[0])
    if isinstance(eos, int):
        return eos
    if getattr(tokenizer, "eos_token_id", None) is not None:
        return int(tokenizer.eos_token_id)
    raise RuntimeError(
        "cannot resolve an end-of-turn token id for this model; inspect the "
        "tokenizer on the remote box and extend resolve_terminator_id()"
    )


# ======================================================== model construction

#: Submodule-name fragments excluded from LoRA targeting: everything that is
#: not the language model proper. The discovered target list is logged so a
#: bad pattern is visible, and an empty result is a hard error.
NON_LANGUAGE_PATTERNS = ("vision", "visual", "image", "audio", "tower")


def discover_lora_targets(model: Any) -> list[str]:
    """Full names of every 4-bit linear in the language model (all-linear
    placement per the QLoRA ablation), excluding vision/audio submodules and
    any output head / embedding."""
    import bitsandbytes as bnb

    targets: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, bnb.nn.Linear4bit):
            continue
        low = name.lower()
        if any(p in low for p in NON_LANGUAGE_PATTERNS):
            continue
        if "lm_head" in low or "embed" in low:
            continue
        targets.append(name)
    if not targets:
        raise RuntimeError(
            "discover_lora_targets found no language-model Linear4bit "
            "modules; the base model may not be 4-bit quantized, or the "
            "NON_LANGUAGE_PATTERNS exclusions ate everything. Inspect "
            "model.named_modules() on the remote box."
        )
    return targets


def resolve_projector_modules(model: Any, projector_name: str) -> list[str]:
    """Full names of modules matching --projector-module (trained fully via
    modules_to_save). 'none' disables; an unresolvable name is a hard error
    listing top-level children plus candidate names containing
    embed/project/vision so the right flag is one look away (never guess
    the module tree)."""
    if projector_name.lower() == "none":
        return []
    matches = [
        name for name, _ in model.named_modules()
        if name == projector_name or name.endswith("." + projector_name)
    ]
    if not matches:
        top_level = [name for name, _ in model.named_children()]
        candidates = [
            name for name, _ in model.named_modules()
            if name and any(
                k in name.lower() for k in ("embed", "project", "vision")
            )
        ][:40]
        raise RuntimeError(
            f"--projector-module {projector_name!r} matches no module in "
            f"this model. Top-level submodules: {top_level}. Candidate "
            f"names (embed/project/vision): {candidates}. For Gemma 4 use "
            "'embed_vision' (resolves to model.embed_vision), or "
            "'--projector-module none' to train LoRA only."
        )
    return matches


def build_model(spec: Any, cfg: TrainConfig, hf_token: str | None):
    """Load the 4-bit base, prepare for k-bit training, wrap in LoRA."""
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if spec.min_transformers is not None:
        import transformers
        from packaging.version import Version
        if Version(transformers.__version__) < Version(spec.min_transformers):
            raise RuntimeError(
                f"{spec.key} requires transformers>={spec.min_transformers}; "
                f"{transformers.__version__} installed."
            )

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    logger.info("Loading %s (%s) in 4-bit NF4 ...", spec.key, spec.hf_id)
    processor = AutoProcessor.from_pretrained(
        spec.hf_id, token=hf_token, trust_remote_code=spec.trust_remote_code,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        spec.hf_id,
        quantization_config=quant,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": cfg.device},
        token=hf_token,
        trust_remote_code=spec.trust_remote_code,
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    targets = discover_lora_targets(model)
    projector = resolve_projector_modules(model, cfg.projector_module)
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
        modules_to_save=projector or None,
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA over %d linears + %d projector module(s); trainable params "
        "%.1fM / %.1fB total",
        len(targets), len(projector), trainable / 1e6, total / 1e9,
    )
    return model, processor, targets, projector


def attach_neftune(model: Any, alpha: float) -> None:
    """NEFTune (Jain et al. 2023): uniform noise on input embeddings during
    training, scale alpha/sqrt(seq_len * dim). Registered as a forward hook
    so it is automatically inert under model.eval()."""
    if alpha <= 0:
        return
    import torch

    emb = model.get_input_embeddings()

    def hook(module, inputs, output):
        if module.training:
            mag = alpha / math.sqrt(output.shape[1] * output.shape[2])
            return output + torch.zeros_like(output).uniform_(-mag, mag)
        return output

    emb.register_forward_hook(hook)


# ================================================================== loss

def _base_model_logits(model: Any, model_inputs: dict) -> Any:
    """Teacher forward for KD: the frozen base model's logits on the same
    inputs, via PEFT's ``disable_adapter()`` (bypasses BOTH the LoRA deltas
    and the modules_to_save projector copy -- VERIFY the projector part on
    the remote box, see TO_TEST.md). Forced to eval mode for the duration so
    the NEFTune hook and dropout cannot perturb the teacher."""
    import torch

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), model.disable_adapter():
            out = model(**model_inputs)
    finally:
        if was_training:
            model.train()
    return out.logits


def weighted_loss(model: Any, model_inputs: dict, weights: Any,
                  loss_kind: str = "ce", return_per_example: bool = False):
    """Per-token weighted loss: each example is normalized by ITS OWN sum of
    ABSOLUTE weights (so plain-SFT and annotated/negative-weight examples
    arrive at comparable gradient scale, and long examples don't dominate),
    then the batch is the mean over examples -- a micro-batch of B is
    EXACTLY equivalent to B batch-1 passes averaged, padding positions
    carry weight 0 and vanish entirely.

    ``loss_kind="ce"``: token cross-entropy against the example's targets.
    ``loss_kind="kd"``: soft cross-entropy against the frozen BASE model's
    distribution on the same inputs (self-distillation replay -- preserve,
    don't teach).

    BOTH paths slice the weighted target positions out of the logits BEFORE
    any float32 cast: at a 262k vocab a full-sequence fp32 logit copy costs
    ~1 GB per 1k tokens, all of it spent on prompt positions whose weight is
    zero.

    With ``return_per_example=True`` returns ``(loss, per_example)`` where
    ``per_example`` is the detached [B] tensor of per-example normalized
    losses (exact per-source logging from mixed-source batches)."""
    import torch
    import torch.nn.functional as F

    if loss_kind not in VALID_LOSSES:
        raise ValueError(f"weighted_loss: bad loss_kind {loss_kind!r}")

    out = model(**model_inputs)
    logits = out.logits  # [B, L, V]
    shift_logits = logits[:, :-1, :]
    shift_labels = model_inputs["input_ids"][:, 1:]
    n_rows, n_pos = shift_labels.shape
    w = weights[:, 1:].reshape(-1)

    # Slice the weighted positions BEFORE any float32 cast.
    mask = w != 0
    row_of = mask.nonzero(as_tuple=True)[0] // n_pos  # owning example per pos
    student = shift_logits.reshape(-1, shift_logits.shape[-1])[mask].float()

    if loss_kind == "ce":
        token_loss = F.cross_entropy(
            student, shift_labels.reshape(-1)[mask], reduction="none",
        )
    else:
        teacher_logits = _base_model_logits(model, model_inputs)
        teacher = teacher_logits[:, :-1, :].reshape(
            -1, teacher_logits.shape[-1]
        )[mask].float()
        token_loss = -(F.softmax(teacher, dim=-1)
                       * F.log_softmax(student, dim=-1)).sum(dim=-1)

    zeros = torch.zeros(n_rows, dtype=token_loss.dtype,
                        device=token_loss.device)
    # out-of-place index_add keeps the autograd path to token_loss clean
    numer = zeros.index_add(0, row_of, token_loss * w[mask])
    denom = zeros.index_add(0, row_of, w[mask].abs())
    per_example = numer / denom.clamp(min=1e-8)
    loss = per_example.mean()
    if return_per_example:
        return loss, per_example.detach()
    return loss


# ============================================================== eval hooks

@dataclass
class TrainContext:
    """What an eval hook gets to work with: enough to score held-out loss
    (model + collator) AND to generate (model + processor + adapter), so
    exact-match probes (training/probes.py) plug in with the same
    ``hook(ctx) -> dict[str, float]`` signature."""

    model: Any
    processor: Any
    adapter: Any        #: agent.model FamilyAdapter for this architecture
    collator: Collator
    device: str


@dataclass
class MetricGuard:
    """Guard one eval metric: regression = worse than the best seen by more
    than ``rel_tolerance`` (relative)."""

    metric: str
    higher_is_better: bool
    rel_tolerance: float

    def is_regression(self, value: float, best: float) -> bool:
        if self.higher_is_better:
            return value < best * (1.0 - self.rel_tolerance)
        return value > best * (1.0 + self.rel_tolerance)

    def is_improvement(self, value: float, best: float | None) -> bool:
        if best is None:
            return True
        return value > best if self.higher_is_better else value < best


class HeldOutLossHook:
    """Mean per-example normalized loss on a reserved slice of the training
    data, reported overall AND per source (``heldout_loss/<source>``) so
    every dataset is tracked -- and guarded -- individually. Each example is
    scored with ITS OWN loss kind, so for KD sources the held-out KD loss is
    exactly the drift-from-base early-warning metric. Task-level probes
    (exact-match accuracy etc., training/probes.py) plug in beside it as
    more callables with the same signature."""

    name = "heldout_loss"

    def __init__(self, examples: list[TrainingExample],
                 micro_batch: int = 1):
        self.examples = examples
        self.micro_batch = max(1, micro_batch)

    def __call__(self, ctx: TrainContext) -> dict[str, float]:
        import torch

        if not self.examples:
            return {}
        model = ctx.model
        was_training = model.training
        model.eval()
        per_src_sum: dict[str, float] = {}
        per_src_n: dict[str, int] = {}
        # Same bucketed batching as training (weighted_loss normalizes each
        # row by its own |w| sum, so batching changes NO per-example value);
        # the fixed rng only orders the batches, which we don't care about.
        batches = epoch_batches(self.examples, self.micro_batch,
                                random.Random(0))
        with torch.no_grad():
            for exs in batches:
                built = ctx.collator.build_batch(exs)
                _, per_example = weighted_loss(
                    model, built["model_inputs"], built["weights"],
                    loss_kind=exs[0].loss, return_per_example=True,
                )
                for ex, lv in zip(exs, per_example.tolist()):
                    per_src_sum[ex.source] = per_src_sum.get(ex.source, 0.0) + lv
                    per_src_n[ex.source] = per_src_n.get(ex.source, 0) + 1
        if was_training:
            model.train()
        metrics = {
            self.name: sum(per_src_sum.values()) / len(self.examples),
        }
        for src in sorted(per_src_sum):
            metrics[f"{self.name}/{src}"] = per_src_sum[src] / per_src_n[src]
        return metrics


# ============================================================ checkpointing

def save_checkpoint(model: Any, ckpt_dir: Path, meta: dict,
                    tlog: TrainLogger) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)  # adapter + modules_to_save (projector)
    (ckpt_dir / "train_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    tlog.event("checkpoint_saved", path=str(ckpt_dir), **{
        k: meta.get(k) for k in ("step", "epoch")
    })
    return ckpt_dir


def load_adapter_state(model: Any, ckpt_dir: Path) -> None:
    """Load a saved adapter state (LoRA + modules_to_save) into a live PEFT
    model -- used both by --resume-checkpoint and by rollback."""
    from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file

    weights_file = ckpt_dir / "adapter_model.safetensors"
    if not weights_file.is_file():
        raise FileNotFoundError(
            f"no adapter_model.safetensors under {ckpt_dir} -- not a "
            "checkpoint produced by train.py / PEFT save_pretrained"
        )
    state = load_file(str(weights_file))
    result = set_peft_model_state_dict(model, state)
    unexpected = getattr(result, "unexpected_keys", None)
    if unexpected:
        raise RuntimeError(
            f"adapter state from {ckpt_dir} has unexpected keys "
            f"(architecture mismatch?): {sorted(unexpected)[:8]} ..."
        )


# ============================================================== the trainer

def _example_chars(ex: TrainingExample) -> int:
    """Total text size of an example (all message text parts + target):
    the cheap stand-in for token count (~4 chars/token) used by the
    overlong-example guard."""
    n = len(ex.target_text)
    for m in ex.messages:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    n += len(part.get("text") or "")
    return n


def materialize(sources: list[DataSource],
                max_example_chars: int = 0,
                ) -> dict[str, list[TrainingExample]]:
    """Load every source into memory. With ``max_example_chars > 0``,
    oversized examples (mostly long OpenThoughts KD rows) are dropped with a
    per-source WARNING count -- bounding per-sequence activation/logit
    memory without ever silently truncating anyone's text."""
    by_source: dict[str, list[TrainingExample]] = {}
    for src in sources:
        exs = list(src.examples())
        if max_example_chars > 0:
            kept = [ex for ex in exs
                    if _example_chars(ex) <= max_example_chars]
            if len(kept) < len(exs):
                logger.warning(
                    "source %s: DROPPED %d/%d example(s) over "
                    "max_example_chars=%d",
                    src.name, len(exs) - len(kept), len(exs),
                    max_example_chars,
                )
            exs = kept
        if not exs:
            raise ValueError(f"data source {src.name!r} yielded no examples")
        by_source[src.name] = exs
        logger.info("source %-24s %6d examples  (weight %.2f)",
                    src.name, len(exs), src.weight)
    return by_source


def epoch_order(by_source: dict[str, list[TrainingExample]],
                weights: dict[str, float], rng: random.Random,
                ) -> list[TrainingExample]:
    """One epoch's example order: each source contributes ~weight x its size
    (sampled with replacement when weight > 1, without when <= 1), then the
    union is shuffled so sources interleave within the epoch."""
    order: list[TrainingExample] = []
    for name, exs in by_source.items():
        w = weights.get(name, 1.0)
        n = max(1, round(w * len(exs)))
        if n <= len(exs):
            order.extend(rng.sample(exs, n))
        else:
            order.extend(rng.choices(exs, k=n))
    rng.shuffle(order)
    return order


def _batch_bucket_key(ex: TrainingExample) -> tuple:
    """Examples may share a micro-batch iff these match: loss kind (KD needs
    the extra teacher forward), image count (pixel tensors must stack), and
    a coarse length bin (padding waste stays bounded; chars ~ 4x tokens is
    plenty accurate for binning)."""
    n_images = sum(
        1
        for m in ex.messages
        for part in (m.get("content") or [])
        if isinstance(part, dict) and part.get("type") == "image"
    )
    chars = _example_chars(ex)
    length_bin, edge = 0, 512
    while chars > edge:
        edge = int(edge * 1.5)
        length_bin += 1
    return (ex.loss, n_images, length_bin)


def epoch_batches(order: list[TrainingExample], micro_batch: int,
                  rng: random.Random) -> list[list[TrainingExample]]:
    """Group an epoch's example order into micro-batches: fill buckets
    (:func:`_batch_bucket_key`) in order, emit a batch whenever one fills,
    flush remainders as short batches, then shuffle the batch list so
    sources and buckets interleave. Bucket remainders make the batch count
    slightly larger than ``ceil(len(order) / micro_batch)`` -- the scheduler
    estimate tolerates that (a few trailing steps at the LR floor)."""
    if micro_batch <= 1:
        return [[ex] for ex in order]
    buckets: dict[tuple, list[TrainingExample]] = {}
    batches: list[list[TrainingExample]] = []
    for ex in order:
        bucket = buckets.setdefault(_batch_bucket_key(ex), [])
        bucket.append(ex)
        if len(bucket) >= micro_batch:
            batches.append(bucket.copy())
            bucket.clear()
    for bucket in buckets.values():
        if bucket:
            batches.append(bucket.copy())
    rng.shuffle(batches)
    return batches


def run_training(
    sources: list[DataSource],
    cfg: TrainConfig,
    extra_hooks: list[Callable[[TrainContext], dict[str, float]]] | None = None,
    extra_guards: dict[str, MetricGuard] | None = None,
) -> int:
    """The generic loop: train ``cfg.architecture`` on ``sources`` per
    ``cfg``. This is the ONLY entry point -- the CLI and every run script
    end up here. ``extra_hooks`` run beside the built-in held-out-loss hook
    after every save (see :class:`TrainContext`); ``extra_guards`` map
    metric names to :class:`MetricGuard` for the regression machinery
    (probes from training/probes.py arrive through these two)."""
    import torch
    from agent.config import CONFIG
    from agent.model import ADAPTERS, spec_for

    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    architecture = cfg.architecture or CONFIG.model_key
    spec = spec_for(architecture)
    tlog = TrainLogger(cfg.label)
    logger.info("run dir: %s", tlog.run_dir)

    # ------------------------------------------------------------- data
    by_source = materialize(sources, max_example_chars=cfg.max_example_chars)
    src_weights = {s.name: s.weight for s in sources}

    # Held-out slice: a fixed fraction, drawn proportionally from every
    # source, removed from training entirely -- capped per source, because
    # the fraction is of the materialized pool and every held-out example is
    # a full forward pass at each save.
    holdout: list[TrainingExample] = []
    if cfg.holdout_fraction > 0:
        for name, exs in by_source.items():
            rng.shuffle(exs)
            n_hold = max(1, int(len(exs) * cfg.holdout_fraction))
            if cfg.holdout_cap > 0:
                n_hold = min(n_hold, cfg.holdout_cap)
            holdout.extend(exs[:n_hold])
            by_source[name] = exs[n_hold:]
    n_train = sum(len(v) for v in by_source.values())
    logger.info("train examples: %d   held out: %d", n_train, len(holdout))

    # ------------------------------------------------------------ model
    model, processor, lora_targets, projector = build_model(
        spec, cfg, CONFIG.hf_token
    )
    attach_neftune(model, cfg.neftune_alpha)
    tokenizer = getattr(processor, "tokenizer", processor)
    terminator = resolve_terminator_id(model, tokenizer)
    collator = Collator(
        processor, ADAPTERS[spec.family], terminator,
        compute_dtype=torch.bfloat16, device=cfg.device,
    )

    ckpt_root = Path(CONFIG.weights_dir) / spec.key
    if cfg.resume_checkpoint:
        resume_dir = ckpt_root / cfg.resume_checkpoint
        load_adapter_state(model, resume_dir)
        tlog.event("resumed_from", path=str(resume_dir))

    # -------------------------------------------------------- optimizer
    import bitsandbytes as bnb
    from transformers import get_scheduler

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(
        params, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    # Estimate: bucket remainders in epoch_batches add a few extra batches
    # per epoch beyond ceil(n / micro_batch) (at most one per bucket), so
    # the cosine schedule may end a handful of steps early -- those trailing
    # steps just run at the LR floor.
    contributions = sum(max(1, round(src_weights.get(n, 1.0) * len(v)))
                        for n, v in by_source.items())
    steps_per_epoch = math.ceil(
        math.ceil(contributions / max(1, cfg.micro_batch)) / cfg.grad_accum
    )
    total_steps = cfg.max_steps or (cfg.epochs * steps_per_epoch)
    if cfg.scheduler == "cosine":
        scheduler = get_scheduler(
            "cosine_with_min_lr", optimizer,
            num_warmup_steps=int(total_steps * cfg.warmup_ratio),
            num_training_steps=total_steps,
            scheduler_specific_kwargs={"min_lr_rate": cfg.lr_floor},
        )
    else:
        scheduler = get_scheduler(
            "constant_with_warmup", optimizer,
            num_warmup_steps=int(total_steps * cfg.warmup_ratio),
        )

    # -------------------------------------------------------- hooks/guards
    ctx = TrainContext(
        model=model, processor=processor, adapter=ADAPTERS[spec.family],
        collator=collator, device=cfg.device,
    )
    eval_hooks: list[Callable[[TrainContext], dict[str, float]]] = []
    if holdout:
        eval_hooks.append(HeldOutLossHook(holdout,
                                          micro_batch=cfg.micro_batch))
    eval_hooks.extend(extra_hooks or [])
    # Overall held-out loss + one guard PER SOURCE (for KD sources the
    # per-source held-out KD loss is the drift-from-base measure), then any
    # run-script guards (probes) on top.
    guards = {
        "heldout_loss": MetricGuard(
            "heldout_loss", higher_is_better=False,
            rel_tolerance=cfg.regression_tolerance,
        ),
    }
    for src_name in by_source:
        metric = f"heldout_loss/{src_name}"
        guards[metric] = MetricGuard(
            metric, higher_is_better=False,
            rel_tolerance=cfg.regression_tolerance,
        )
    guards.update(extra_guards or {})
    best_metrics: dict[str, float] = {}
    best_ckpt: Path | None = None

    tlog.write_config({
        **dataclasses.asdict(cfg),
        "architecture_resolved": architecture,
        "architecture_hf_id": spec.hf_id,
        "sources": {s.name: s.weight for s in sources},
        "git_rev": _git_rev(),
        "n_train": n_train,
        "n_holdout": len(holdout),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "terminator_id": terminator,
        "lora_targets": lora_targets,
        "projector_modules": projector,
    })

    n_rollbacks = 0

    def run_hooks_and_guard(step: int, epoch: int) -> None:
        """After each save: run every probe, log one flat eval row, track
        bests, react to regressions per cfg.on_regression."""
        nonlocal best_ckpt, n_rollbacks
        all_metrics: dict[str, float] = {}
        for hook in eval_hooks:
            all_metrics.update(hook(ctx))
        if all_metrics:
            tlog.event("eval", step=step, **all_metrics)
            tlog.eval_row(step, epoch, all_metrics)
        for mname, value in all_metrics.items():
            guard = guards.get(mname)
            if guard is None:
                continue
            best = best_metrics.get(mname)
            if guard.is_improvement(value, best):
                best_metrics[mname] = value
                best_ckpt = last_ckpt
            elif best is not None and guard.is_regression(value, best):
                tlog.event(
                    "REGRESSION", metric=mname, value=value, best=best,
                    action=cfg.on_regression,
                )
                logger.error(
                    "REGRESSION on %s: %.5g (best %.5g, tolerance %.0f%%)"
                    " -- action: %s",
                    mname, value, best,
                    guard.rel_tolerance * 100, cfg.on_regression,
                )
                if cfg.on_regression == "abort":
                    raise SystemExit(
                        f"aborted: {mname} regressed ({value:.5g} vs best "
                        f"{best:.5g}); best checkpoint: {best_ckpt}"
                    )
                if cfg.on_regression == "rollback":
                    if best_ckpt is None:
                        logger.error(
                            "rollback requested but no best checkpoint "
                            "exists yet; continuing WITHOUT rollback"
                        )
                    else:
                        n_rollbacks += 1
                        # Rollback restores the weights but not the data or
                        # LR position, so a persistently regressing run
                        # oscillates (roll back, re-regress, ...) forever;
                        # cap it and hand back the best checkpoint instead.
                        if (cfg.max_rollbacks > 0
                                and n_rollbacks > cfg.max_rollbacks):
                            tlog.event("rollback_limit",
                                       n_rollbacks=n_rollbacks,
                                       best_checkpoint=str(best_ckpt))
                            raise SystemExit(
                                f"aborted: {n_rollbacks} rollbacks exceed "
                                f"max_rollbacks={cfg.max_rollbacks} -- the "
                                "run is oscillating, not converging. Best "
                                f"checkpoint: {best_ckpt}"
                            )
                        load_adapter_state(model, best_ckpt)
                        tlog.event("rolled_back", to=str(best_ckpt),
                                   n_rollbacks=n_rollbacks)

    # ------------------------------------------------------------- loop
    model.train()
    step = 0
    last_ckpt: Path | None = None
    src_loss_sum: dict[str, float] = {}
    src_loss_n: dict[str, int] = {}
    done = False

    for epoch in range(cfg.epochs):
        if done:
            break
        order = epoch_order(by_source, src_weights, rng)
        batches = epoch_batches(order, cfg.micro_batch, rng)
        optimizer.zero_grad(set_to_none=True)
        for i, exs in enumerate(batches):
            built = collator.build_batch(exs)
            # weighted_loss means over the batch's per-example normalized
            # losses, so each example's gradient contribution is
            # 1/(micro_batch * grad_accum) -- identical to batch-1 training
            # at the same effective batch.
            loss, per_example = weighted_loss(
                model, built["model_inputs"], built["weights"],
                loss_kind=exs[0].loss, return_per_example=True,
            )
            (loss / cfg.grad_accum).backward()
            for ex, lv in zip(exs, per_example.tolist()):
                src_loss_sum[ex.source] = src_loss_sum.get(ex.source, 0.0) + lv
                src_loss_n[ex.source] = src_loss_n.get(ex.source, 0) + 1

            if (i + 1) % cfg.grad_accum == 0 or i == len(batches) - 1:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % cfg.log_steps == 0:
                    per_src = {
                        n: round(src_loss_sum[n] / src_loss_n[n], 4)
                        for n in src_loss_sum
                    }
                    tlog.step({
                        "step": step, "epoch": epoch,
                        "loss": sum(src_loss_sum.values())
                                / max(1, sum(src_loss_n.values())),
                        "lr": scheduler.get_last_lr()[0],
                        "grad_norm": grad_norm,
                        "source_loss": per_src,
                    })
                    src_loss_sum.clear()
                    src_loss_n.clear()

                if step % cfg.save_steps == 0:
                    last_ckpt = save_checkpoint(
                        model, ckpt_root / f"{cfg.label}_step{step}",
                        {"step": step, "epoch": epoch, "git_rev": _git_rev(),
                         "sources": {n: len(v) for n, v in by_source.items()},
                         "best_metrics": best_metrics},
                        tlog,
                    )
                    run_hooks_and_guard(step, epoch)

                if cfg.max_steps and step >= cfg.max_steps:
                    done = True
                    break

    # Final save (skip if the loop happened to save on the very last step).
    if last_ckpt is None or not str(last_ckpt).endswith(f"step{step}"):
        last_ckpt = save_checkpoint(
            model, ckpt_root / f"{cfg.label}_step{step}",
            {"step": step, "epoch": cfg.epochs, "git_rev": _git_rev(),
             "sources": {n: len(v) for n, v in by_source.items()},
             "best_metrics": best_metrics, "final": True},
            tlog,
        )
        run_hooks_and_guard(step, cfg.epochs)

    tlog.event("done", steps=step, final_checkpoint=str(last_ckpt),
               best_checkpoint=str(best_ckpt), best_metrics=best_metrics)
    print(f"done. final checkpoint: {last_ckpt}")
    print(f"best checkpoint ({best_metrics}): {best_ckpt}")
    return 0


# ==================================================================== CLI

def configure_logging() -> None:
    """Console logging for training entry points (CLI and run scripts)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    """Generic CLI front-end: --data builds JsonlSources, every other flag
    maps 1:1 onto a TrainConfig field (defaults read FROM TrainConfig, so
    the dataclass stays the single source of truth)."""
    d = TrainConfig()
    p = argparse.ArgumentParser(
        prog="python -m training.train",
        description="Source-agnostic QLoRA training "
                    "(see training/TRAINING_OVERVIEW.md)",
    )
    p.add_argument("--data", nargs="+", required=True,
                   help="jsonl file(s), each 'path' or 'path:weight'")
    p.add_argument("--architecture", default=d.architecture,
                   help="MODEL_REGISTRY key (default: MODEL_KEY from .env)")
    p.add_argument("--label", default=d.label,
                   help="run label; names the log dir and checkpoints")
    p.add_argument("--resume-checkpoint", default=d.resume_checkpoint,
                   help="checkpoint name under weights/<architecture>/ to "
                        "resume the adapter from")
    # recipe knobs (defaults per TRAINING_OVERVIEW.md, via TrainConfig)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--max-steps", type=int, default=d.max_steps)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--scheduler", choices=("cosine", "constant"),
                   default=d.scheduler)
    p.add_argument("--warmup-ratio", type=float, default=d.warmup_ratio)
    p.add_argument("--lr-floor", type=float, default=d.lr_floor,
                   help="cosine decays to this fraction of peak LR")
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--micro-batch", type=int, default=d.micro_batch,
                   help="examples per forward pass (length-bucketed); "
                        "effective batch = micro_batch * grad_accum")
    p.add_argument("--grad-accum", type=int, default=d.grad_accum)
    p.add_argument("--lora-r", type=int, default=d.lora_r)
    p.add_argument("--lora-alpha", type=int, default=d.lora_alpha)
    p.add_argument("--lora-dropout", type=float, default=d.lora_dropout)
    p.add_argument("--neftune-alpha", type=float, default=d.neftune_alpha,
                   help="0 disables NEFTune embedding noise")
    p.add_argument("--projector-module", default=d.projector_module,
                   help="module name trained fully via modules_to_save "
                        "(default embed_vision for Gemma 4); "
                        "'none' to train LoRA only")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default=d.device)
    # data hygiene
    p.add_argument("--max-example-chars", type=int, default=d.max_example_chars,
                   help="drop (loudly) examples whose total text exceeds "
                        "this many chars; 0 disables")
    # cadence + safety
    p.add_argument("--log-steps", type=int, default=d.log_steps)
    p.add_argument("--save-steps", type=int, default=d.save_steps)
    p.add_argument("--holdout-fraction", type=float,
                   default=d.holdout_fraction)
    p.add_argument("--holdout-cap", type=int, default=d.holdout_cap,
                   help="max held-out examples per source; 0 = uncapped")
    p.add_argument("--on-regression", choices=("warn", "rollback", "abort"),
                   default=d.on_regression)
    p.add_argument("--regression-tolerance", type=float,
                   default=d.regression_tolerance,
                   help="relative tolerance before a guarded metric counts "
                        "as regressed")
    p.add_argument("--max-rollbacks", type=int, default=d.max_rollbacks,
                   help="abort after this many rollbacks in one run")
    return p


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    sources: list[DataSource] = [parse_data_arg(a) for a in args.data]
    cfg = TrainConfig(**{
        f.name: getattr(args, f.name)
        for f in dataclasses.fields(TrainConfig)
    })
    return run_training(sources, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
