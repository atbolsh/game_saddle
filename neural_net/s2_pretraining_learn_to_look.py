"""Learn-to-look pretraining for the S2 coordinate head.

Sixty-four game beginnings, one generation at a time, then an 18-hour
mix of teacher-forced looks and moves with the usual replay sources.
A reply ends when generation stops on eos. This trainer does not treat
[HOLD] or [RELOAD] as controls; a beginning that emits [HOLD] is saved
and skipped.

Traces are appended and flushed under data_game/<label>/. Logs go to
logs/train_<label>_<stamp>/. A full checkpoint (the Gemma adapter and
coord_head.pt) goes to weights/s2/. Gemma-only adapters stay under
weights/<architecture>/.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import io
import json
import logging
import multiprocessing
import os
import random
import re
import signal
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from torch import nn

from agent import game_io
from agent import memory as mem
from agent.config import CONFIG
from agent.modes import (
    S2_BEGINNING_USER,
    SYSTEM_PROMPT_S2,
    SYSTEM_PROMPT_S2_ANALYST,
    _S2_LOOK_LINES,
    _S2_MOVE_LINES,
    parse_target,
)
from agent.model import ADAPTERS, VLModel, spec_for
from neural_net.s2 import _find_lm_head, _hidden_size
from training.external_data import sources_from_manifest
from training.image_noise import (
    INFERENCE_STRENGTH,
    TRAINING_STRENGTH,
    noise_image,
)
from training.run_weekend import VramMonitor
from training.train import (
    ANCHOR_ADAPTER,
    PrefixKVRuntime,
    TrainConfig,
    Collator,
    TrainingExample,
    _forward_last_hidden,
    _gpu_mem_snapshot,
    build_model,
    configure_logging,
    epoch_batches,
    epoch_order,
    load_adapter_state,
    materialize,
    resolve_terminator_id,
    save_checkpoint,
    weighted_loss,
)

logger = logging.getLogger("s2_look")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_GAME = REPO_ROOT / "data_game"
N_BEGINNINGS = 64
#: Weekend optimizer steps landed near 25 s. The cosine length uses that
#: figure when --max-steps is omitted; the clock is still --hours.
MEASURED_STEP_S = 25.0
HEAD_LR = 1e-4
HEAD_WEIGHT_DECAY = 0.01
BEGINNING_MAX_NEW_TOKENS = 256
ANALYST_MAX_NEW_TOKENS = 64
SKIP_STREAK_LIMIT = 25
TRAINER_NAME = "s2_pretraining_learn_to_look"

_HOLD_RE = re.compile(r"\[HOLD\]", re.IGNORECASE)
_PRIMITIVE_RE = re.compile(
    r"\[(" + "|".join(game_io.ACTIONS) + r")\b",
    re.IGNORECASE,
)

#: Corner and edge points. Upper right is (0.9, 0.9); y is up.
_CORNERS = (
    ("upper-right", 0.9, 0.9),
    ("upper-left", 0.1, 0.9),
    ("lower-right", 0.9, 0.1),
    ("lower-left", 0.1, 0.1),
    ("upper-middle", 0.5, 0.9),
    ("lower-middle", 0.5, 0.1),
    ("left-middle", 0.1, 0.5),
    ("right-middle", 0.9, 0.5),
)


def render_clean_board(payload: dict) -> dict:
    """One clean frame plus settings. Runs in a spawn worker: no CUDA."""
    import random as _random

    from PIL import Image

    from neural_net.render import canonical_frame

    _random.seed(int(payload["seed"]))
    game = game_io.new_multi_gold_game(
        gameSize=int(payload["game_size"]),
        n_gold=payload["n_gold"],
        opening="any",
    )
    frame = canonical_frame(game)
    settings = game_io.settings_to_dict(game.settings)
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return {
        "png": buf.getvalue(),
        "settings": settings,
        "seed": int(payload["seed"]),
    }


def _gold_phrase(x: float, y: float) -> str:
    vert = "upper" if y >= 0.5 else "lower"
    if x < 0.33:
        horiz = "left"
    elif x > 0.67:
        horiz = "right"
    else:
        horiz = "middle"
    if horiz == "middle":
        return f"the {vert}-middle gold"
    return f"the {vert}-{horiz} gold"


def _ask(kind: str, phrase: str) -> str:
    if kind == "look":
        return f"Look at {phrase}."
    if kind == "move":
        return f"Move to {phrase}."
    raise ValueError(f"bad kind {kind!r}")


def _coords(kind: str, agent: tuple[float, float],
            target: tuple[float, float]) -> list[float]:
    """s, v, v_bar. Look leaves the body; move sends S1 to the target."""
    sx, sy = agent
    vx, vy = target
    if kind == "look":
        bx, by = (2.0 * sx - vx, 2.0 * sy - vy)
    elif kind == "move":
        bx, by = (2.0 * vx - vx, 2.0 * vy - vy)
    else:
        raise ValueError(f"bad kind {kind!r}")
    return [sx, sy, vx, vy, bx, by]


def _system_message(text: str) -> dict:
    return {"role": "system", "content": [{"type": "text", "text": text}]}


def _text_message(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _save_png(img: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    with open(path, "rb") as handle:
        os.fsync(handle.fileno())


def _retarget_image(messages: list[dict], url: str) -> list[dict]:
    out = copy.deepcopy(messages)
    found = 0
    for message in out:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                part["url"] = url
                part.pop("image", None)
                found += 1
    if found != 1:
        raise RuntimeError(
            f"expected one image part when retargeting, found {found}"
        )
    return out


class BoardPump:
    """Prefetch clean boards. ``workers`` 0 renders in this process."""

    def __init__(self, workers: int, game_size: int,
                 n_gold_choices: tuple[int, ...], rng: random.Random) -> None:
        self.game_size = game_size
        self.n_gold_choices = n_gold_choices
        self.rng = rng
        self.pool = None
        self.pending: list[Any] = []
        self.depth = 0
        self.n_gold_choices = n_gold_choices
        if workers > 0:
            ctx = multiprocessing.get_context("spawn")
            self.pool = ctx.Pool(workers)
            self.depth = workers * 2

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def set_choices(self, choices: tuple[int, ...]) -> None:
        if choices == self.n_gold_choices:
            return
        stale = self.pending
        self.pending = []
        self.n_gold_choices = choices
        for item in stale:
            if self.pool is not None:
                item.get()

    def _submit(self) -> None:
        payload = {
            "seed": self.rng.randrange(1 << 30),
            "game_size": self.game_size,
            "n_gold": self.rng.choice(self.n_gold_choices),
        }
        if self.pool is None:
            self.pending.append(render_clean_board(payload))
        else:
            self.pending.append(self.pool.apply_async(
                render_clean_board, (payload,),
            ))

    def take(self) -> dict:
        while len(self.pending) < max(1, self.depth):
            self._submit()
        item = self.pending.pop(0)
        if self.pool is None:
            return item
        return item.get()


class ReplayCycle:
    def __init__(self, by_source: dict, weights: dict, micro_batch: int,
                 rng: random.Random) -> None:
        self.by_source = by_source
        self.weights = weights
        self.micro_batch = micro_batch
        self.rng = rng
        self.batches: list[list[TrainingExample]] = []
        self.index = 0

    def next_batch(self) -> list[TrainingExample]:
        if self.index >= len(self.batches):
            order = epoch_order(self.by_source, self.weights, self.rng)
            self.batches = epoch_batches(order, self.micro_batch, self.rng)
            if not self.batches:
                raise RuntimeError("replay sources produced no batches")
            self.index = 0
        batch = self.batches[self.index]
        self.index += 1
        return batch


def _align_hidden(hidden: torch.Tensor, weights: torch.Tensor,
                  input_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Match lm_head's hidden (full sequence or a tail) to ``weights``."""
    length = int(hidden.shape[1])
    if length == input_len:
        return hidden, weights
    if length > input_len:
        raise RuntimeError(
            f"lm_head hidden length {length} exceeds input length {input_len}"
        )
    start = input_len - length
    if bool((weights[:, :start] != 0).any()):
        raise RuntimeError(
            f"coordinate hidden covers the last {length} positions of "
            f"{input_len}, but a reply token sits before position {start}"
        )
    return hidden, weights[:, start:]


def _coord_l2(head: nn.Linear, hidden: torch.Tensor, weights: torch.Tensor,
              targets: torch.Tensor) -> torch.Tensor:
    pred = head(hidden.float())
    err = (pred - targets[:, None, :].to(pred.dtype)).pow(2).mean(dim=-1)
    mask = weights != 0
    counts = mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise RuntimeError("coordinate loss row has no reply tokens")
    per = (err * mask).sum(dim=1) / counts.to(err.dtype)
    return per.mean()


def _generate_text(vl: VLModel, model: Any, messages: list[dict],
                   max_new_tokens: int) -> str:
    """One reply. Generation stops on eos; eval mode skips checkpointing."""
    was_training = model.training
    prev_cache = getattr(model.config, "use_cache", False)
    model.eval()
    model.config.use_cache = True
    try:
        return vl.generate(messages, max_new_tokens=max_new_tokens)
    finally:
        model.config.use_cache = prev_cache
        if was_training:
            model.train()


def _open_png(raw: bytes):
    from PIL import Image
    return Image.open(io.BytesIO(raw)).convert("RGB")


class Run:
    def __init__(self, cfg: TrainConfig, *, hours: float, workers: int,
                 head_lr: float, gemma_base: str | None) -> None:
        self.cfg = cfg
        self.hours = hours
        self.workers = workers
        self.head_lr = head_lr
        self.gemma_base = gemma_base
        self.rng = random.Random(cfg.seed)
        self.label = cfg.label
        self.data_dir = DATA_GAME / self.label
        self.images = self.data_dir / "images"
        self.traces_path = self.data_dir / "traces.jsonl"
        self.analyst_path = self.data_dir / "analyst_traces.jsonl"
        self.look_path = self.data_dir / "look_moves.jsonl"
        self.state_path = DATA_GAME / f"{self.label}_state.json"
        self.images.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.beginnings_trained = 0
        self.phase = "beginnings"
        self.seen_beginnings = 0
        self.seen_looks = 0
        self.skip_streak = 0
        self.session_ids: list[str] = []
        self.scratchpads_cleared = False
        self.deadline = 0.0
        self.model = None
        self.processor = None
        self.head: nn.Linear | None = None
        self.vl: VLModel | None = None
        self.collator = None
        self.optimizer = None
        self.scheduler = None
        self.prefix_state = PrefixKVRuntime()
        self.tlog = None
        self.metrics_path: Path | None = None
        self.vram: VramMonitor | None = None
        self.replay: ReplayCycle | None = None
        self.pump: BoardPump | None = None
        self.spec = None
        self.gemma_root: Path | None = None
        self.s2_root: Path | None = None

    def _state(self) -> dict:
        return {
            "trainer": TRAINER_NAME,
            "opt_step": self.step,
            "beginnings_trained": self.beginnings_trained,
            "phase": self.phase,
            "seen_beginnings": self.seen_beginnings,
            "seen_looks": self.seen_looks,
        }

    def _flush_state(self) -> None:
        _write_json_atomic(self.state_path, self._state())

    def _log_metrics(self, record: dict) -> None:
        assert self.metrics_path is not None
        _append_jsonl(self.metrics_path, record)

    def load(self) -> None:
        import bitsandbytes as bnb
        from transformers import get_scheduler

        from training.token_fence import (
            count_prefix_tokens,
            measure_image_soft_tokens,
        )
        from training.train import TrainLogger, attach_neftune

        cfg = self.cfg
        architecture = cfg.architecture or CONFIG.model_key
        self.spec = spec_for(architecture)
        self.tlog = TrainLogger(cfg.label)
        self.metrics_path = self.tlog.run_dir / "metrics.jsonl"
        log_path = self.tlog.run_dir / "train.log"
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        ))
        logging.getLogger().addHandler(file_handler)
        logger.info("run dir: %s", self.tlog.run_dir)
        logger.info("data dir: %s", self.data_dir)

        self.model, self.processor, _targets, _projector = build_model(
            self.spec, cfg, CONFIG.hf_token,
        )
        attach_neftune(self.model, cfg.neftune_alpha)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        terminator = resolve_terminator_id(self.model, tokenizer)
        self.collator = Collator(
            self.processor, ADAPTERS[self.spec.family], terminator,
            compute_dtype=torch.bfloat16, device=cfg.device,
        )
        probe = [_system_message(SYSTEM_PROMPT_S2)]
        prefix_n = count_prefix_tokens(probe, tokenizer, self.processor)
        logger.info("S2 system prefix tokens=%d", prefix_n)

        from neural_net.paths import weights_root

        root = weights_root()
        self.gemma_root = root / self.spec.key
        self.s2_root = root / "s2"
        logger.info("S2 checkpoints: %s", self.s2_root)
        self._load_resume()
        self._load_anchor()

        hidden = _hidden_size(self.model)
        device = torch.device(cfg.device)
        self.head = nn.Linear(hidden, 6).to(device=device, dtype=torch.float32)
        self._load_head()

        self.vl = VLModel(self.spec)
        self.vl.model = self.model
        self.vl.processor = self.processor
        self.vl.adapter = ADAPTERS[self.spec.family]
        self.vl._loaded = True

        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        head_params = list(self.head.parameters())
        self.optimizer = bnb.optim.PagedAdamW8bit(
            [
                {"params": lora_params, "lr": cfg.lr,
                 "weight_decay": cfg.weight_decay},
                {"params": head_params, "lr": self.head_lr,
                 "weight_decay": HEAD_WEIGHT_DECAY},
            ],
            lr=cfg.lr,
        )
        total_steps = cfg.max_steps or max(
            1, int(self.hours * 3600.0 / MEASURED_STEP_S)
        )
        warmup = int(total_steps * cfg.warmup_ratio)
        if cfg.scheduler == "cosine":
            self.scheduler = get_scheduler(
                "cosine_with_min_lr", self.optimizer,
                num_warmup_steps=warmup,
                num_training_steps=total_steps,
                scheduler_specific_kwargs={"min_lr_rate": cfg.lr_floor},
            )
        else:
            self.scheduler = get_scheduler(
                "constant_with_warmup", self.optimizer,
                num_warmup_steps=warmup,
            )
        for _ in range(self.step):
            self.scheduler.step()

        with tempfile.TemporaryDirectory(prefix="token_fence_") as tmp:
            from PIL import Image
            img = Path(tmp) / "measure.png"
            Image.new("RGB", (32, 32), (18, 18, 18)).save(img)
            image_soft = measure_image_soft_tokens(self.processor, str(img))
        sources = sources_from_manifest()
        by_source = materialize(
            sources, cfg, processor=self.processor,
            image_soft_tokens=image_soft,
        )
        weights = {src.name: src.weight for src in sources}
        self.replay = ReplayCycle(by_source, weights, cfg.micro_batch, self.rng)
        self.tlog.write_config({
            "trainer": TRAINER_NAME,
            "label": cfg.label,
            "hours": self.hours,
            "workers": self.workers,
            "head_lr": self.head_lr,
            "n_beginnings": N_BEGINNINGS,
            "cosine_steps": total_steps,
            "prefix_tokens": prefix_n,
            "resume_checkpoint": cfg.resume_checkpoint,
            "gemma_base": self.gemma_base,
            "anchor_checkpoint": cfg.anchor_checkpoint,
        })

    def _load_resume(self) -> None:
        cfg = self.cfg
        if self.gemma_base:
            assert self.gemma_root is not None and self.tlog is not None
            gemma_dir = self.gemma_root / self.gemma_base
            load_adapter_state(self.model, gemma_dir)
            self._read_state(ours=False)
            logger.info(
                "loaded Gemma 4 adapter %s; new coordinate head",
                gemma_dir,
            )
            self.tlog.event("gemma_base", path=str(gemma_dir))
            return
        if not cfg.resume_checkpoint:
            self._read_state(ours=False)
            logger.info(
                "no checkpoint: base Gemma 4, fresh LoRA, fresh coordinate head"
            )
            return
        assert self.s2_root is not None
        resume_dir = self.s2_root / cfg.resume_checkpoint
        load_adapter_state(self.model, resume_dir)
        meta_path = resume_dir / "train_meta.json"
        meta: dict = {}
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ours = meta.get("trainer") == TRAINER_NAME
        if ours:
            self.step = int(meta["step"])
        self._read_state(ours=ours)
        logger.info("resumed adapter from %s (step %d)", resume_dir, self.step)
        assert self.tlog is not None
        self.tlog.event("resumed_from", path=str(resume_dir), step=self.step)

    def _read_state(self, *, ours: bool) -> None:
        if not self.state_path.is_file():
            return
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state.get("trainer") != TRAINER_NAME:
            raise RuntimeError(
                f"{self.state_path} is not an {TRAINER_NAME} state file"
            )
        state_step = int(state.get("opt_step", 0))
        if ours and state_step != self.step:
            raise RuntimeError(
                f"checkpoint step {self.step} does not match "
                f"{self.state_path} opt_step {state_step}"
            )
        if not ours and state_step != 0:
            source = self.cfg.resume_checkpoint or self.gemma_base
            raise RuntimeError(
                f"{self.state_path} already has opt_step {state_step}, but "
                f"{source!r} is not an {TRAINER_NAME} checkpoint"
            )
        self.beginnings_trained = int(state.get("beginnings_trained", 0))
        self.phase = str(state.get("phase", "beginnings"))
        self.seen_beginnings = int(state.get("seen_beginnings", 0))
        self.seen_looks = int(state.get("seen_looks", 0))

    def _adapter_dir(self, name: str) -> Path:
        """The one directory that holds this adapter name.

        Gemma-only adapters live under weights/<architecture>/. Full S2
        checkpoints live under weights/s2/. The same name in both is an
        error.
        """
        assert self.gemma_root is not None and self.s2_root is not None
        gemma = self.gemma_root / name
        s2 = self.s2_root / name
        gemma_ok = (gemma / "adapter_config.json").is_file()
        s2_ok = (s2 / "adapter_config.json").is_file()
        if gemma_ok and s2_ok:
            raise RuntimeError(
                f"adapter {name!r} is in both {gemma} and {s2}"
            )
        if s2_ok:
            return s2
        if gemma_ok:
            return gemma
        raise FileNotFoundError(
            f"no adapter {name!r} under {s2} or {gemma}"
        )

    def _load_anchor(self) -> None:
        cfg = self.cfg
        if cfg.anchor_checkpoint:
            anchor_dir = self._adapter_dir(cfg.anchor_checkpoint)
        elif cfg.resume_checkpoint:
            assert self.s2_root is not None
            anchor_dir = self.s2_root / cfg.resume_checkpoint
        elif self.gemma_base:
            assert self.gemma_root is not None
            anchor_dir = self.gemma_root / self.gemma_base
        else:
            return
        assert self.tlog is not None
        if not (anchor_dir / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"anchor checkpoint: no adapter under {anchor_dir}"
            )
        self.model.load_adapter(
            str(anchor_dir), adapter_name=ANCHOR_ADAPTER, is_trainable=False,
        )
        self.model.set_adapter("default")
        self.tlog.event("anchor_loaded", path=str(anchor_dir))
        logger.info("anchor adapter %s", anchor_dir)

    def _load_head(self) -> None:
        assert self.head is not None and self.s2_root is not None
        if self.gemma_base or not self.cfg.resume_checkpoint:
            logger.info("new coordinate head")
            return
        path = self.s2_root / self.cfg.resume_checkpoint / "coord_head.pt"
        if not path.is_file():
            logger.info("fresh coordinate head (no %s)", path)
            return
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.head.load_state_dict(state)
        device = torch.device(self.cfg.device)
        self.head.to(device=device, dtype=torch.float32)
        logger.info("loaded coordinate head %s", path)

    def save(self, reason: str) -> Path:
        assert self.s2_root is not None and self.head is not None
        assert self.tlog is not None and self.model is not None
        name = f"{self.label}_step_{self.step:06d}"
        directory = self.s2_root / name
        meta = self._state()
        meta["reason"] = reason
        save_checkpoint(self.model, directory, meta, self.tlog)
        torch.save(self.head.state_dict(), directory / "coord_head.pt")
        last = self.s2_root / f"{self.label}_last"
        save_checkpoint(self.model, last, meta, self.tlog)
        torch.save(self.head.state_dict(), last / "coord_head.pt")
        self._flush_state()
        logger.info("checkpoint %s (%s)", directory, reason)
        return directory

    def _heartbeat(self, exs: list[TrainingExample], kind: str,
                   built: dict) -> None:
        assert self.tlog is not None
        ids = built["model_inputs"]["input_ids"]
        self.tlog.heartbeat({
            "phase": self.phase,
            "kind": kind,
            "sources": [ex.source for ex in exs],
            "batch": int(ids.shape[0]),
            "n_tokens": int(ids.shape[1]),
            "gpu": _gpu_mem_snapshot(),
        })

    def _optimizer_step(self, loss_value: float, kind: str) -> None:
        assert self.optimizer is not None and self.scheduler is not None
        assert self.head is not None and self.model is not None
        assert self.tlog is not None
        params = (
            [p for p in self.model.parameters() if p.requires_grad]
            + [p for p in self.head.parameters() if p.requires_grad]
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            params, self.cfg.max_grad_norm,
        )
        if not torch.isfinite(grad_norm):
            raise RuntimeError(
                f"non-finite grad_norm {grad_norm} at step {self.step}"
            )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        self.prefix_state.opt_steps += 1
        if (self.cfg.prefix_kv_refresh_steps > 0
                and self.prefix_state.opt_steps
                % self.cfg.prefix_kv_refresh_steps == 0):
            self.prefix_state.cache = None
            self.prefix_state.key = None
        lrs = self.scheduler.get_last_lr()
        record = {
            "step": self.step,
            "phase": self.phase,
            "kind": kind,
            "loss": loss_value,
            "grad_norm": float(grad_norm),
            "lr_lora": lrs[0],
            "lr_head": lrs[-1],
            "gpu": _gpu_mem_snapshot(),
        }
        self.tlog.last_step(record)
        self._log_metrics(record)
        if self.step % self.cfg.log_steps == 0:
            logger.info(
                "step %d %s loss %.4f grad %.3f lr %.3g / %.3g",
                self.step, kind, loss_value, float(grad_norm),
                lrs[0], lrs[-1],
            )
        if self.cfg.save_steps and self.step % self.cfg.save_steps == 0:
            self.save("save_steps")
        self._flush_state()

    def _backward(self, loss: torch.Tensor, scale: int) -> float:
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {self.step}")
        (loss / scale).backward()
        return float(loss.detach())

    def _look_loss(self, exs: list[TrainingExample], *,
                   cross_entropy: bool) -> torch.Tensor:
        assert self.collator is not None and self.model is not None
        assert self.head is not None
        built = self.collator.build_batch(exs)
        self._heartbeat(exs, "look" if cross_entropy else "beginning", built)
        weights = built["weights"]
        inputs = built["model_inputs"]
        input_len = int(inputs["input_ids"].shape[1])
        targets = torch.tensor(
            [ex.meta["coords"] for ex in exs],
            dtype=torch.float32, device=weights.device,
        )
        if not cross_entropy:
            hidden = _forward_last_hidden(self.model, inputs, logits_to_keep=1)
            if int(hidden.shape[1]) != input_len:
                raise RuntimeError(
                    f"beginning hidden length {int(hidden.shape[1])} "
                    f"!= input length {input_len}"
                )
            loss = _coord_l2(self.head, hidden, weights, targets)
            if not loss.requires_grad:
                raise RuntimeError("beginning loss is not connected to the graph")
            return loss

        captured: dict[str, torch.Tensor] = {}

        def _hook(_module: nn.Module, args: tuple, _output: Any) -> None:
            captured["h"] = args[0]

        handle = _find_lm_head(self.model).register_forward_hook(_hook)
        try:
            ce, _per = weighted_loss(
                self.model, inputs, weights,
                loss_kind=exs[0].loss,
                return_per_example=True,
                example_weight=built["example_weight"],
                prefix_kv=self.prefix_state,
                prefix_len=int(exs[0].prefix_n_tokens),
                prefix_hash=exs[0].prefix_hash,
                kd_chunk=self.cfg.kd_lm_head_chunk,
            )
        finally:
            handle.remove()
        if "h" not in captured:
            raise RuntimeError("lm_head hook did not fire on a look batch")
        hidden, aligned = _align_hidden(captured["h"], weights, input_len)
        coord = _coord_l2(self.head, hidden, aligned, targets)
        if not coord.requires_grad:
            raise RuntimeError(
                "coordinate loss is not connected to the look-batch graph"
            )
        return ce + coord

    def _replay_loss(self, exs: list[TrainingExample]) -> torch.Tensor:
        assert self.collator is not None and self.model is not None
        built = self.collator.build_batch(exs)
        self._heartbeat(exs, "replay", built)
        loss, _per = weighted_loss(
            self.model, built["model_inputs"], built["weights"],
            loss_kind=exs[0].loss,
            return_per_example=True,
            example_weight=built["example_weight"],
            prefix_kv=self.prefix_state,
            prefix_len=int(exs[0].prefix_n_tokens),
            prefix_hash=exs[0].prefix_hash,
            kd_chunk=self.cfg.kd_lm_head_chunk,
        )
        return loss

    def _stopped(self) -> bool:
        if self.hours > 0 and time.perf_counter() >= self.deadline:
            return True
        return bool(self.cfg.max_steps and self.step >= self.cfg.max_steps)

    def _take_board(self, choices: tuple[int, ...]) -> dict:
        assert self.pump is not None
        self.pump.set_choices(choices)
        return self.pump.take()

    def _noise_beginning(self, raw: dict) -> tuple[Path, Path]:
        seen = self.images / f"begin_{self.seen_beginnings:06d}.png"
        trained = self.images / f"begin_{self.seen_beginnings:06d}_train.png"
        shown = noise_image(
            _open_png(raw["png"]), self.rng, INFERENCE_STRENGTH,
        )
        _save_png(shown, seen)
        again = noise_image(shown, self.rng, TRAINING_STRENGTH)
        _save_png(again, trained)
        return seen, trained

    def _target_from_analyst(self, settings: dict,
                             parsed: dict) -> tuple[float, float]:
        kind = parsed["kind"]
        index = parsed["index"]
        if kind == "gold":
            golds = settings["gold"]
            if not isinstance(index, int) or not 0 <= index < len(golds):
                raise ValueError(f"gold index {index!r} out of range")
            x, y = golds[index]
            return float(x), float(y)
        if kind == "openings":
            openings = settings["openings"]
            if not isinstance(index, int) or not 0 <= index < len(openings):
                raise ValueError(f"opening index {index!r} out of range")
            x, y = openings[index]["center"]
            return float(x), float(y)
        raise ValueError(f"unusable analyst kind {kind!r}")

    def _example(self, messages: list[dict], target_text: str,
                 coords: list[float], source: str,
                 loss: str) -> TrainingExample:
        return TrainingExample(
            messages=messages,
            target_text=target_text,
            loss=loss,
            source=source,
            meta={"coords": coords},
        )

    def _pending_beginnings(self) -> list[TrainingExample]:
        if not self.traces_path.is_file():
            if self.beginnings_trained:
                raise RuntimeError(
                    f"state has {self.beginnings_trained} trained beginnings "
                    f"but {self.traces_path} is missing"
                )
            return []
        kept: list[dict] = []
        with open(self.traces_path, encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("meta", {}).get("status") != "kept":
                    continue
                kept.append(rec)
        if self.beginnings_trained > len(kept):
            raise RuntimeError(
                f"state beginnings_trained={self.beginnings_trained} but "
                f"{self.traces_path} has {len(kept)} kept records"
            )
        out = []
        for rec in kept[self.beginnings_trained:]:
            train_image = Path(rec["meta"]["train_image"])
            if not train_image.is_file():
                raise RuntimeError(
                    f"kept beginning is missing its training frame {train_image}"
                )
            messages = _retarget_image(rec["messages"], str(train_image))
            out.append(self._example(
                messages, rec["target_text"], rec["meta"]["coords"],
                "s2_beginning", "ce",
            ))
        return out

    def _generate_beginning(self, loop: asyncio.AbstractEventLoop,
                            client: Any) -> TrainingExample | None:
        assert self.vl is not None and self.model is not None
        assert self.tlog is not None
        raw = None
        settings = None
        for _draw in range(20):
            raw = self._take_board((1, 2, 3))
            settings = raw["settings"]
            if settings["gold"] or settings["openings"]:
                break
            logger.info("beginning board had nothing to look at; redrawing")
        else:
            raise RuntimeError(
                "no beginning board with a gold or an exit after 20 draws"
            )
        assert raw is not None and settings is not None
        seen, trained = self._noise_beginning(raw)
        sid = mem.new_session_id()
        self.session_ids.append(sid)
        notes = loop.run_until_complete(mem.get_session_notes(client, sid))
        notepad = mem.format_notepad(notes)
        messages = [
            _system_message(SYSTEM_PROMPT_S2),
            {"role": "user", "content": [
                {"type": "image", "url": str(seen)},
                {"type": "text", "text": notepad + "\n\n" + S2_BEGINNING_USER},
            ]},
        ]
        reply = _generate_text(
            self.vl, self.model, messages, BEGINNING_MAX_NEW_TOKENS,
        )
        self.seen_beginnings += 1
        hits = len(_PRIMITIVE_RE.findall(reply))
        if hits:
            self.tlog.event(
                "primitive_token", n=hits, seen=self.seen_beginnings,
            )
            logger.info(
                "beginning %d emitted %d primitive token(s); not applied",
                self.seen_beginnings, hits,
            )
        for key, value in game_io.parse_remember_notes(reply):
            loop.run_until_complete(
                mem.set_session_note(client, sid, key, value, 1)
            )
        reason = None
        parsed = None
        coords = None
        if not reply.strip():
            reason = "empty"
        elif _HOLD_RE.search(reply):
            reason = "hold"
        else:
            analyst_text = _analyst_text(settings, reply)
            analyst_messages = [
                _system_message(SYSTEM_PROMPT_S2_ANALYST),
                {"role": "user", "content": [
                    {"type": "image", "url": str(seen)},
                    {"type": "text", "text": analyst_text},
                ]},
            ]
            analysis = _generate_text(
                self.vl, self.model, analyst_messages, ANALYST_MAX_NEW_TOKENS,
            )
            _append_jsonl(self.analyst_path, {
                "seen": self.seen_beginnings,
                "session_id": sid,
                "messages": analyst_messages,
                "target_text": analysis,
            })
            parsed = parse_target(analysis)
            if parsed is None or parsed["kind"] == "none":
                reason = "no_target"
            else:
                try:
                    point = self._target_from_analyst(settings, parsed)
                except ValueError as exc:
                    logger.info("beginning analyst rejected: %s", exc)
                    reason = "no_target"
                else:
                    agent = (
                        float(settings["agent_x"]),
                        float(settings["agent_y"]),
                    )
                    coords = _coords("look", agent, point)
        status = "skipped" if reason else "kept"
        record = {
            "messages": messages,
            "target_text": reply,
            "meta": {
                "status": status,
                "reason": reason,
                "coords": coords,
                "train_image": str(trained),
                "seen_image": str(seen),
                "session_id": sid,
                "analyst_target": parsed,
                "primitive_hits": hits,
            },
        }
        _append_jsonl(self.traces_path, record)
        if reason:
            self.skip_streak += 1
            logger.info(
                "beginning %d skipped (%s)", self.seen_beginnings, reason,
            )
            if self.skip_streak >= SKIP_STREAK_LIMIT:
                raise RuntimeError(
                    f"{SKIP_STREAK_LIMIT} beginnings skipped in a row"
                )
            return None
        self.skip_streak = 0
        assert coords is not None
        return self._example(
            _retarget_image(messages, str(trained)),
            reply, coords, "s2_beginning", "ce",
        )

    def beginnings(self) -> None:
        if self.phase == "synthetic":
            logger.info("beginnings already finished; skipping")
            return
        assert self.vram is not None and self.optimizer is not None
        self.vram.set_stage("train-beginnings")
        loop = asyncio.new_event_loop()
        client = None
        try:
            client = loop.run_until_complete(mem.connect())
            pending = self._pending_beginnings()
            while self.beginnings_trained < N_BEGINNINGS:
                if self._stopped():
                    logger.info("clock finished during beginnings")
                    return
                while (len(pending) < self.cfg.micro_batch
                       and self.beginnings_trained + len(pending) < N_BEGINNINGS):
                    if self._stopped():
                        return
                    made = self._generate_beginning(loop, client)
                    if made is not None:
                        pending.append(made)
                full = len(pending) >= self.cfg.micro_batch
                tail = (
                    pending
                    and self.beginnings_trained + len(pending) >= N_BEGINNINGS
                )
                if not full and not tail:
                    return
                n = self.cfg.micro_batch if full else len(pending)
                batch = pending[:n]
                pending = pending[n:]
                self.optimizer.zero_grad(set_to_none=True)
                loss = self._look_loss(batch, cross_entropy=False)
                value = self._backward(loss, 1)
                self.phase = "beginnings"
                self._optimizer_step(value, "beginning")
                self.beginnings_trained += len(batch)
                self._flush_state()
                logger.info(
                    "beginnings trained %d / %d",
                    self.beginnings_trained, N_BEGINNINGS,
                )
            self.phase = "synthetic"
            self._flush_state()
        finally:
            if client is not None:
                try:
                    self._clear_open_scratchpads(loop, client)
                finally:
                    loop.run_until_complete(client.close())
            loop.close()

    def _clear_open_scratchpads(self, loop: asyncio.AbstractEventLoop,
                                client: Any) -> None:
        sids = set(self.session_ids)
        sids.update(self._trace_session_ids())
        for sid in sids:
            loop.run_until_complete(mem.clear_session_notes(client, sid))
        self.scratchpads_cleared = True
        logger.info("NAMS scratchpads cleared (%d sessions)", len(sids))

    def _drop_scratchpads(self) -> None:
        """Resume path: the beginnings phase already finished, so the notes go."""
        loop = asyncio.new_event_loop()
        client = loop.run_until_complete(mem.connect())
        try:
            self._clear_open_scratchpads(loop, client)
        finally:
            loop.run_until_complete(client.close())
            loop.close()

    def _trace_session_ids(self) -> list[str]:
        if not self.traces_path.is_file():
            return []
        found: list[str] = []
        with open(self.traces_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                sid = json.loads(line).get("meta", {}).get("session_id")
                if sid:
                    found.append(sid)
        return found

    def _synthetic_example(self) -> TrainingExample:
        raw = self._take_board((2, 3))
        settings = raw["settings"]
        golds = settings["gold"]
        phrases = [_gold_phrase(float(g[0]), float(g[1])) for g in golds]
        unique = len(phrases) == len(set(phrases)) and bool(phrases)
        kind = "look" if self.rng.random() < 0.5 else "move"
        if unique and self.rng.random() < 0.5:
            index = self.rng.randrange(len(golds))
            phrase = phrases[index]
            point = (float(golds[index][0]), float(golds[index][1]))
        else:
            name, x, y = self.rng.choice(_CORNERS)
            phrase = f"the {name} corner"
            point = (float(x), float(y))
        lines = _S2_LOOK_LINES if kind == "look" else _S2_MOVE_LINES
        reply = self.rng.choice(lines)
        n_hist = 0 if self.rng.random() < 0.5 else self.rng.choice((1, 2))
        messages = [_system_message(SYSTEM_PROMPT_S2)]
        for _turn in range(n_hist):
            hist_kind = "look" if self.rng.random() < 0.5 else "move"
            hist_name, _hx, _hy = self.rng.choice(_CORNERS)
            hist_phrase = f"the {hist_name} corner"
            hist_lines = (
                _S2_LOOK_LINES if hist_kind == "look" else _S2_MOVE_LINES
            )
            messages.append(_text_message("user", _ask(hist_kind, hist_phrase)))
            messages.append(_text_message(
                "assistant", self.rng.choice(hist_lines),
            ))
        path = self.images / f"look_{self.seen_looks:06d}.png"
        noised = noise_image(
            _open_png(raw["png"]), self.rng, TRAINING_STRENGTH,
        )
        _save_png(noised, path)
        self.seen_looks += 1
        messages.append({"role": "user", "content": [
            {"type": "image", "url": str(path)},
            {"type": "text", "text": _ask(kind, phrase)},
        ]})
        agent = (float(settings["agent_x"]), float(settings["agent_y"]))
        coords = _coords(kind, agent, point)
        record = {
            "messages": messages,
            "target_text": reply,
            "meta": {
                "kind": kind,
                "phrase": phrase,
                "coords": coords,
                "image": str(path),
            },
        }
        _append_jsonl(self.look_path, record)
        return self._example(messages, reply, coords, "s2_look", "ce")

    def synthetic(self) -> None:
        if self.phase != "synthetic":
            return
        assert self.vram is not None and self.optimizer is not None
        assert self.replay is not None
        self.vram.set_stage("train-synthetic")
        if not self.scratchpads_cleared:
            self._drop_scratchpads()
        while not self._stopped():
            self.optimizer.zero_grad(set_to_none=True)
            losses: list[float] = []
            for i in range(self.cfg.grad_accum):
                if i % 2 == 0:
                    exs = [
                        self._synthetic_example()
                        for _ in range(self.cfg.micro_batch)
                    ]
                    loss = self._look_loss(exs, cross_entropy=True)
                else:
                    exs = self.replay.next_batch()
                    loss = self._replay_loss(exs)
                losses.append(self._backward(loss, self.cfg.grad_accum))
            self._optimizer_step(sum(losses) / len(losses), "synthetic")

    def run(self) -> int:
        self.deadline = time.perf_counter() + self.hours * 3600.0
        self.vram = VramMonitor(self.label)
        self.pump = BoardPump(
            self.workers, CONFIG.game_size, (1, 2, 3), self.rng,
        )
        try:
            self.load()
            assert self.model is not None and self.optimizer is not None
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            self.beginnings()
            self.synthetic()
            self.save("clock")
            return 0
        except KeyboardInterrupt:
            logger.info("interrupted; saving")
            if self.model is not None and self.head is not None:
                self.save("signal")
            return 0
        except Exception as exc:
            if self.tlog is not None:
                self.tlog.record_crash(exc)
            else:
                crash = DATA_GAME / f"{self.label}_crash.txt"
                crash.write_text(traceback.format_exc(), encoding="utf-8")
            if self.model is not None and self.head is not None:
                try:
                    self.save("crash")
                except Exception:
                    logger.exception("checkpoint save after crash failed")
            raise
        finally:
            if self.pump is not None:
                self.pump.close()
            if self.vram is not None:
                self.vram.finish()


def _analyst_text(settings: dict, reply: str) -> str:
    lines = ["Gold list (index, x, y):"]
    for i, gold in enumerate(settings["gold"]):
        lines.append(f"  {i}: {float(gold[0]):.4f}, {float(gold[1]):.4f}")
    lines.append("Openings list (index, center x, center y):")
    for i, opening in enumerate(settings["openings"]):
        center = opening["center"]
        lines.append(
            f"  {i}: {float(center[0]):.4f}, {float(center[1]):.4f}"
        )
    lines.append("Player reply:")
    lines.append(reply)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    d = TrainConfig()
    parser = argparse.ArgumentParser(
        prog="python -m neural_net.s2_pretraining_learn_to_look",
        description="S2 learn-to-look pretraining",
    )
    parser.add_argument("--label", default="s2_look")
    parser.add_argument("--architecture", default=d.architecture)
    parser.add_argument(
        "--resume-checkpoint", default=d.resume_checkpoint,
        help="Learn-to-look checkpoint name under weights/s2/. "
             "Loads the adapter and coord_head.pt and continues its step.",
    )
    parser.add_argument(
        "--gemma-base", default=None,
        help="Gemma 4 adapter name under weights/<architecture>/. "
             "Loads that adapter and starts a new coordinate head. "
             "For a pretrained Gemma that has no S2 head yet.",
    )
    parser.add_argument(
        "--anchor-checkpoint", default=None,
        help="KD teacher adapter. Default: --gemma-base or "
             "--resume-checkpoint, or the base model when neither is set.",
    )
    parser.add_argument("--hours", type=float, default=18.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=d.lr)
    parser.add_argument("--lr-floor", type=float, default=d.lr_floor)
    parser.add_argument("--scheduler", choices=("cosine", "constant"),
                        default=d.scheduler)
    parser.add_argument("--micro-batch", type=int, default=d.micro_batch)
    parser.add_argument("--grad-accum", type=int, default=d.grad_accum)
    parser.add_argument("--prefix-kv-refresh-steps", type=int,
                        default=d.prefix_kv_refresh_steps)
    parser.add_argument("--log-steps", type=int, default=d.log_steps)
    parser.add_argument(
        "--save-steps", type=int, default=d.save_steps,
        help="How often to write weights/s2/<label>_step_NNNNNN and "
             "<label>_last. Ctrl-C and SIGTERM also write both.",
    )
    parser.add_argument("--seed", type=int, default=d.seed)
    parser.add_argument("--device", default=d.device)
    parser.add_argument("--lora-r", type=int, default=d.lora_r)
    parser.add_argument("--lora-alpha", type=int, default=d.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=d.lora_dropout)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    if args.micro_batch < 1:
        raise SystemExit("--micro-batch must be >= 1")
    if N_BEGINNINGS % args.micro_batch != 0:
        raise SystemExit(
            f"--micro-batch {args.micro_batch} does not divide "
            f"{N_BEGINNINGS} beginnings"
        )
    if args.grad_accum < 2 or args.grad_accum % 2:
        raise SystemExit("--grad-accum must be even and at least 2")
    if args.hours <= 0 and args.max_steps <= 0:
        raise SystemExit("set --hours or --max-steps")
    if args.workers < 0:
        raise SystemExit("--workers must be >= 0")
    if args.gemma_base and args.resume_checkpoint:
        raise SystemExit(
            "pass either --gemma-base or --resume-checkpoint"
        )
    cfg = TrainConfig(
        label=args.label,
        architecture=args.architecture,
        resume_checkpoint=args.resume_checkpoint,
        anchor_checkpoint=args.anchor_checkpoint,
        max_steps=args.max_steps or None,
        lr=args.lr,
        lr_floor=args.lr_floor,
        scheduler=args.scheduler,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        prefix_kv_refresh_steps=args.prefix_kv_refresh_steps,
        log_steps=args.log_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        device=args.device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    run = Run(
        cfg, hours=args.hours, workers=args.workers, head_lr=HEAD_LR,
        gemma_base=args.gemma_base,
    )
    interrupted: dict[str, int | None] = {"signum": None}

    def _on_signal(signum: int, _frame: Any) -> None:
        interrupted["signum"] = signum
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    try:
        return run.run()
    except KeyboardInterrupt:
        logger.info("signal %s", interrupted["signum"])
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
