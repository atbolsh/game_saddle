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

SEQUENCE = EXACTLY WHAT INFERENCE PRODUCES. Trained ids are
``prompt_ids + content_ids + [end-of-turn]`` where ``prompt_ids`` come from
``apply_chat_template(..., add_generation_prompt=True)`` -- byte-identical to
the prefix ``VLModel.generate`` builds -- and ``content_ids`` tokenize
``target_text`` with an offset mapping (which is how char spans become token
weights). No train/inference template mismatch.

IMAGES ARE FIRST-CLASS. The model's own AutoProcessor runs over the full
messages, so batches carry pixel tensors and the forward pass is the
multimodal forward. An example that declares an image but produces no pixel
tensor is a hard error (no silent text-only degradation). Vision tower
frozen; the multimodal projector trains via PEFT ``modules_to_save`` and so
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

ROLLBACK. After every save, registered eval hooks run (stage 1 ships one:
held-out loss on a reserved slice). A guarded metric regressing past its
threshold logs at ERROR and -- per ``--on-regression warn|rollback|abort``
(default rollback) -- restores the best adapter state and continues, or
aborts the run.

LOGGING. ``logs/train_<label>_<stamp>/``: config.json (resolved config +
seed + git rev + discovered LoRA target modules), train_log.jsonl + .txt
(per-step loss, per-source loss, LR, grad norm), events.jsonl (saves, evals,
rollbacks).

NOTE (remote-environment rule): this file cannot be executed on the local
editing box (no torch/transformers/GPU). Anything that depends on the exact
Gemma4Unified module tree -- most importantly the projector module name
passed via ``--projector-module`` -- must be verified on the remote box
against ``model.named_modules()``; the code fails loudly with instructions
when the name does not resolve.

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

@dataclass
class TrainingExample:
    """One trainable unit. ``messages`` is the HF chat content-list format
    (see agent/model.py); ``target_text`` is the assistant reply to train on;
    ``span_weights`` are (char_start, char_end, weight) triples over
    ``target_text`` -- absent means plain SFT (every target token weighs 1)."""

    messages: list[dict]
    target_text: str
    span_weights: list[tuple[int, int, float]] | None = None
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
    (optional, list of [start, end, weight]), ``meta`` (optional).

    Malformed lines are hard errors with the line number -- a training set
    that silently drops examples is worse than one that refuses to load."""

    def __init__(self, path: str | Path, weight: float = 1.0):
        self.path = Path(path)
        self.name = self.path.stem
        self.weight = weight
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
                yield TrainingExample(
                    messages=messages,
                    target_text=target_text,
                    span_weights=spans,
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
    grad_accum: int = 16

    # LoRA / regularization
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    neftune_alpha: float = 5.0      #: 0 disables NEFTune embedding noise
    #: module trained fully via modules_to_save; "none" = LoRA only.
    #: VERIFY against model.named_modules() on the remote box.
    projector_module: str = "multi_modal_projector"

    # environment / reproducibility
    seed: int = 17
    device: str = "cuda:0"

    # cadence + safety
    log_steps: int = 10
    save_steps: int = 200
    holdout_fraction: float = 0.05
    on_regression: str = "rollback"  #: "warn" | "rollback" | "abort"
    regression_tolerance: float = 0.10  #: relative slack before regression


# ================================================================= logging

class TrainLogger:
    """One run directory under logs/: config.json, train_log.{jsonl,txt},
    events.jsonl. Same spirit as agent.run_logging: machine-readable +
    human-readable, and logging failures must never kill a run that is
    burning GPU-hours (they degrade to a one-time console warning)."""

    def __init__(self, label: str, base_dir: str | Path = "logs"):
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = Path(base_dir) / f"train_{label}_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.run_dir / "train_log.jsonl"
        self.txt = self.run_dir / "train_log.txt"
        self.events = self.run_dir / "events.jsonl"
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
    listing the model's top-level children so the right flag is one look
    away (per the remote-environment rule: never guess the module tree)."""
    if projector_name.lower() == "none":
        return []
    matches = [
        name for name, _ in model.named_modules()
        if name == projector_name or name.endswith("." + projector_name)
    ]
    if not matches:
        top_level = [name for name, _ in model.named_children()]
        raise RuntimeError(
            f"--projector-module {projector_name!r} matches no module in "
            f"this model. Top-level submodules: {top_level}. Inspect "
            "model.named_modules() on the remote box and pass the right "
            "name, or '--projector-module none' to train LoRA only."
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

def weighted_loss(model: Any, model_inputs: dict, weights: Any):
    """Weighted token cross-entropy, normalized by the sum of ABSOLUTE
    weights (so plain-SFT and annotated/negative-weight examples arrive at
    comparable gradient scale, and long examples don't dominate)."""
    import torch
    import torch.nn.functional as F

    out = model(**model_inputs)
    logits = out.logits  # [1, L, V]
    shift_logits = logits[:, :-1, :]
    shift_labels = model_inputs["input_ids"][:, 1:]
    shift_w = weights[:, 1:]
    token_ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]).float(),
        shift_labels.reshape(-1),
        reduction="none",
    )
    w = shift_w.reshape(-1)
    denom = w.abs().sum().clamp(min=1e-8)
    return (token_ce * w).sum() / denom


# ============================================================== eval hooks

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
    """Stage-1 placeholder probe: mean per-example normalized loss on a
    reserved slice of the training data. The real early-warning suite
    (fixed boards, planted-error miss rate, replay slices -- see
    TRAINING_EXTRA_DATASETS.md) plugs in beside it in stage 3 as more
    callables with the same signature."""

    name = "heldout_loss"

    def __init__(self, examples: list[TrainingExample], collator: Collator):
        self.examples = examples
        self.collator = collator

    def __call__(self, model: Any) -> dict[str, float]:
        import torch

        if not self.examples:
            return {}
        was_training = model.training
        model.eval()
        total = 0.0
        with torch.no_grad():
            for ex in self.examples:
                built = self.collator.build(ex)
                total += float(
                    weighted_loss(model, built["model_inputs"], built["weights"])
                )
        if was_training:
            model.train()
        return {self.name: total / len(self.examples)}


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

def materialize(sources: list[DataSource]) -> dict[str, list[TrainingExample]]:
    by_source: dict[str, list[TrainingExample]] = {}
    for src in sources:
        exs = list(src.examples())
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


def run_training(sources: list[DataSource], cfg: TrainConfig) -> int:
    """The generic loop: train ``cfg.architecture`` on ``sources`` per
    ``cfg``. This is the ONLY entry point -- the CLI and every run script
    end up here."""
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
    by_source = materialize(sources)
    src_weights = {s.name: s.weight for s in sources}

    # Held-out slice: a fixed fraction, drawn proportionally from every
    # source, removed from training entirely.
    holdout: list[TrainingExample] = []
    if cfg.holdout_fraction > 0:
        for name, exs in by_source.items():
            rng.shuffle(exs)
            n_hold = max(1, int(len(exs) * cfg.holdout_fraction))
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
    steps_per_epoch = math.ceil(
        sum(max(1, round(src_weights.get(n, 1.0) * len(v)))
            for n, v in by_source.items()) / cfg.grad_accum
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
    eval_hooks: list[Callable[[Any], dict[str, float]]] = []
    if holdout:
        eval_hooks.append(HeldOutLossHook(holdout, collator))
    guards = {
        "heldout_loss": MetricGuard(
            "heldout_loss", higher_is_better=False,
            rel_tolerance=cfg.regression_tolerance,
        ),
    }
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

    def run_hooks_and_guard(step: int, epoch: int) -> None:
        """After each save: run every probe, track bests, react to
        regressions per cfg.on_regression."""
        nonlocal best_ckpt
        for hook in eval_hooks:
            metrics = hook(model)
            tlog.event("eval", step=step, **metrics)
            for mname, value in metrics.items():
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
                            load_adapter_state(model, best_ckpt)
                            tlog.event("rolled_back", to=str(best_ckpt))

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
        optimizer.zero_grad(set_to_none=True)
        for i, ex in enumerate(order):
            built = collator.build(ex)
            loss = weighted_loss(model, built["model_inputs"], built["weights"])
            (loss / cfg.grad_accum).backward()
            lv = float(loss.detach())
            src_loss_sum[ex.source] = src_loss_sum.get(ex.source, 0.0) + lv
            src_loss_n[ex.source] = src_loss_n.get(ex.source, 0) + 1

            if (i + 1) % cfg.grad_accum == 0 or i == len(order) - 1:
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
    p.add_argument("--grad-accum", type=int, default=d.grad_accum)
    p.add_argument("--lora-r", type=int, default=d.lora_r)
    p.add_argument("--lora-alpha", type=int, default=d.lora_alpha)
    p.add_argument("--lora-dropout", type=float, default=d.lora_dropout)
    p.add_argument("--neftune-alpha", type=float, default=d.neftune_alpha,
                   help="0 disables NEFTune embedding noise")
    p.add_argument("--projector-module", default=d.projector_module,
                   help="module name trained fully via modules_to_save; "
                        "'none' to train LoRA only. VERIFY against "
                        "model.named_modules() on the remote box.")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default=d.device)
    # cadence + safety
    p.add_argument("--log-steps", type=int, default=d.log_steps)
    p.add_argument("--save-steps", type=int, default=d.save_steps)
    p.add_argument("--holdout-fraction", type=float,
                   default=d.holdout_fraction)
    p.add_argument("--on-regression", choices=("warn", "rollback", "abort"),
                   default=d.on_regression)
    p.add_argument("--regression-tolerance", type=float,
                   default=d.regression_tolerance,
                   help="relative tolerance before a guarded metric counts "
                        "as regressed")
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
