"""Phase-2 coord embedder: aug27 LoRA reads a frozen Fourier code.

Each optimizer step is an anchor with probability 1/2, otherwise one
board. The board is prefilled once (system prompt and picture together).
Sixteen short replies are teacher-forced on that cache, then the cache
is dropped. It is not kept for the next image: the LoRA that wrote those
keys changes at the optimizer step.

Anchors are the manifest datasets, scored by ``weighted_loss``. CE
sources train the dataset reply. KD sources match the frozen aug27
adapter with the coordinate code muted. ``--rl`` replaces the board
with the archived online sampler. The anchor coin stays.

``--hours`` requires ``--cosine-steps``. Past that horizon the
learning-rate multiplier stays at the floor. ``--bench`` loads the
model, times four steady steps, and prints the Monday launch line.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import torch

from agent import game_io
from agent.config import CONFIG
from agent.model import ADAPTERS, spec_for
from agent.modes import SYSTEM_PROMPT_S2, _build_game_messages
from neural_net.coord_embed import CoordEmbedder, resolve_magnitude
from neural_net.paths import weights_root
from neural_net.render import canonical_frame
from neural_net.s2 import _hidden_size, _wants_cache_position
from neural_net.s2_coord_oracle import (
    AXIS_MARGIN,
    KIND_HOUR,
    KIND_LEFTRIGHT,
    KIND_UPDOWN,
    REPLY_JITTER_SIGMA,
    S_RADIUS,
    S_SIGMA,
    clock_hour,
    correct_line,
    direction_label,
    gaze_vbar,
    hour_is_unique,
)
from neural_net.s2_coord_rl import rl_loss
from PIL import Image
from training.external_data import sources_from_manifest
from training.run_weekend import VramMonitor
from training.token_fence import measure_image_soft_tokens
from training.train import (
    ANCHOR_ADAPTER,
    Collator,
    TrainConfig,
    TrainLogger,
    TrainingExample,
    _gpu_mem_snapshot,
    apply_lm_head_chunked,
    build_model,
    configure_logging,
    expand_train_kv,
    load_adapter_state,
    materialize,
    resolve_terminator_id,
    save_checkpoint,
    weighted_loss,
)

logger = logging.getLogger("s2_coord")

HOLDOUT_FRACTION = 0.05
HOLDOUT_CAP = 100
COORD_LOW = -1.0
COORD_HIGH = 2.0
V_TRIES = 400
BOARD_TRIES = 40
DIRECTIONS = ("up", "down", "left", "right")
_MONDAY = ZoneInfo("America/New_York")

QUESTION_DIRECTION = (
    "Which way are you looking? Reply with exactly one of these lines "
    "and say nothing else:\n"
    "Looking: up\n"
    "Looking: down\n"
    "Looking: left\n"
    "Looking: right"
)
QUESTION_HOUR = (
    "What hour are you looking at? Twelve o'clock is straight up, three "
    "is straight right, six is straight down, and nine is straight left. "
    "Reply with exactly one line and say nothing else:\n"
    "Hour: N\n"
    "N is an integer from 1 to 12, with no leading zero."
)


def _factor(step: int, warmup: int, total: int, floor: float) -> float:
    """Warmup, then cosine down to ``floor``, then stay there."""
    if step >= total:
        return floor
    if warmup > 0 and step < warmup:
        return step / max(1, warmup)
    span = max(1, total - warmup)
    progress = (step - warmup) / span
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _holdout_count(n: int) -> int:
    return min(HOLDOUT_CAP, max(1, int(n * HOLDOUT_FRACTION)))


def _direction_kind(direction: str) -> str:
    if direction in ("up", "down"):
        return KIND_UPDOWN
    if direction in ("left", "right"):
        return KIND_LEFTRIGHT
    raise ValueError(f"not a direction {direction!r}")


def _in_box(x: float, y: float) -> bool:
    return COORD_LOW <= x <= COORD_HIGH and COORD_LOW <= y <= COORD_HIGH


def _separated(direction: str, sx: float, sy: float, vx: float, vy: float) -> bool:
    if direction == "up":
        return (vy - sy) >= AXIS_MARGIN
    if direction == "down":
        return (sy - vy) >= AXIS_MARGIN
    if direction == "right":
        return (vx - sx) >= AXIS_MARGIN
    if direction == "left":
        return (sx - vx) >= AXIS_MARGIN
    raise ValueError(f"not a direction {direction!r}")


def _lcp_len(rows: list[torch.Tensor]) -> int:
    """Shared token prefix. Same cut as the localizer's ``_lcp_len``."""
    seqs = [row.view(-1) for row in rows]
    n = min(int(seq.numel()) for seq in seqs)
    if n <= 0:
        raise RuntimeError("image batch has an empty token row")
    first = seqs[0]
    for i in range(n):
        token = int(first[i])
        for seq in seqs[1:]:
            if int(seq[i]) != token:
                return i
    return n


def _token_aligned(key: str, val: torch.Tensor, seq_len: int) -> bool:
    return (
        key not in ("input_ids", "attention_mask")
        and val.dim() == 2
        and val.shape[0] == 1
        and int(val.shape[1]) == seq_len
        and not val.dtype.is_floating_point
    )


def _last_image_index(model_inputs: dict[str, Any]) -> int:
    if "mm_token_type_ids" in model_inputs:
        mask = model_inputs["mm_token_type_ids"]
    elif "token_type_ids" in model_inputs:
        mask = model_inputs["token_type_ids"]
    else:
        raise RuntimeError(
            "image batch has neither mm_token_type_ids nor token_type_ids"
        )
    idx = (mask[0] != 0).nonzero(as_tuple=False).flatten()
    if int(idx.numel()) == 0:
        raise RuntimeError("image batch has no image tokens")
    return int(idx[-1])


def _cache_len(cache: Any) -> int:
    if not hasattr(cache, "layers"):
        raise RuntimeError(f"prefix cache has no layers: {type(cache)!r}")
    keys = cache.layers[0].keys
    if keys is None or not torch.is_tensor(keys) or keys.numel() == 0:
        raise RuntimeError("prefix cache layer 0 has no keys")
    return int(keys.shape[-2])


def _detach_cache(cache: Any) -> None:
    if cache is None:
        raise RuntimeError("prefix cache is missing")
    layers = getattr(cache, "layers", None)
    if layers is not None:
        n = 0
        for layer in layers:
            for name in ("keys", "values"):
                tensor = getattr(layer, name, None)
                if torch.is_tensor(tensor):
                    setattr(layer, name, tensor.detach())
                    n += 1
        if n == 0:
            raise RuntimeError(
                f"prefix cache {type(cache).__name__} has no keys/values"
            )
        return
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if isinstance(key_cache, list) and isinstance(value_cache, list):
        cache.key_cache = [tensor.detach() for tensor in key_cache]
        cache.value_cache = [tensor.detach() for tensor in value_cache]
        return
    raise RuntimeError(f"cannot detach cache type {type(cache).__name__}")


def _lm_head_dtype(model: Any) -> torch.dtype:
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    head = getattr(inner, "lm_head", None)
    if head is None:
        raise RuntimeError(f"{type(inner).__name__} has no lm_head")
    return next(head.parameters()).dtype


def _nvidia_smi_used_total() -> tuple[int, int] | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    used = 0
    total = 0
    for line in proc.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        parts = [part.strip() for part in text.split(",")]
        if len(parts) != 2:
            return None
        used += int(parts[0])
        total += int(parts[1])
    if total <= 0:
        return None
    return used, total


def _hours_until_monday(now: datetime | None = None) -> tuple[datetime, float] | None:
    """Hours from ``now`` until Monday 08:00 America/New_York.

    Monday after 08:00 returns None. That stop has passed; the next
    Monday is not invented.
    """
    if now is None:
        now = datetime.now(_MONDAY)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=_MONDAY)
    else:
        now = now.astimezone(_MONDAY)
    days = (0 - now.weekday()) % 7
    target = (now + timedelta(days=days)).replace(
        hour=8, minute=0, second=0, microsecond=0,
    )
    if target <= now:
        return None
    hours = (target - now).total_seconds() / 3600.0
    return target, hours


class Trainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rng = random.Random(args.seed)
        self.device = args.device
        self.spec = spec_for(CONFIG.model_key)
        self.magnitude = resolve_magnitude(args.magnitude)
        self.step = 0
        self._stop = False
        self._signal = 0
        self.label = args.label
        self.tlog: TrainLogger | None = None
        self.vram: VramMonitor | None = None
        self.model: Any = None
        self.embedder: CoordEmbedder | None = None
        self.collator: Collator | None = None
        self.processor: Any = None
        self.terminator = 0
        self.anchors: list[TrainingExample] = []
        self.tmp: Path | None = None
        self._rl_rewards: list[float] = []
        self._rl_parsed: list[bool] = []
        self.optimizer: Any = None
        self.scheduler: Any = None

    def load(self) -> None:
        if self.args.bench and self.args.cosine_steps <= 0:
            self.args.cosine_steps = 1
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        self.tlog = TrainLogger(self.label)
        self.vram = VramMonitor(self.label)
        cfg = TrainConfig(
            label=self.label,
            device=self.device,
            neftune_alpha=0.0,
            lr=self.args.lr,
            lr_floor=self.args.lr_floor,
            warmup_ratio=self.args.warmup_ratio,
            max_grad_norm=self.args.max_grad_norm,
        )
        model, processor, _, _ = build_model(self.spec, cfg, CONFIG.hf_token)
        ckpt = weights_root() / self.spec.key / self.args.gemma_base
        if not (ckpt / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"no adapter under {ckpt} (--gemma-base {self.args.gemma_base})"
            )
        load_adapter_state(model, ckpt)
        model.load_adapter(str(ckpt), adapter_name=ANCHOR_ADAPTER, is_trainable=False)
        model.set_adapter("default")
        tokenizer = getattr(processor, "tokenizer", processor)
        self.terminator = resolve_terminator_id(model, tokenizer)
        self.collator = Collator(
            processor, ADAPTERS[self.spec.family], self.terminator,
            compute_dtype=torch.bfloat16, device=self.device,
        )
        embedder = CoordEmbedder(_hidden_size(model)).to(
            device=self.device, dtype=torch.float32,
        )
        embedder.set_magnitude(self.magnitude)
        if any(param.requires_grad for param in embedder.parameters()):
            raise RuntimeError("coord embedder gates or Linears are still trainable")
        embedder.enabled = True
        embedder.install(model)
        self.model = model
        self.processor = processor
        self.embedder = embedder
        self._load_anchors(cfg)
        self._build_optimizer(cfg)
        self.tmp = Path(tempfile.mkdtemp(prefix=f"coord_{self.label}_"))
        assert self.tlog is not None
        self.tlog.write_config({
            **vars(self.args),
            "magnitude_tanh": self.magnitude,
            "gemma_checkpoint": str(ckpt),
            "n_anchors": len(self.anchors),
        })
        self.tlog.event(
            "start",
            magnitude=self.magnitude,
            n_anchors=len(self.anchors),
            checkpoint=str(ckpt),
        )
        assert self.scheduler is not None
        task = "rl" if self.args.rl else "teacher-force"
        logger.info(
            "magnitude tanh=%.4f frozen | task %s | anchors %d | "
            "cosine %d warmup %d | lr %.3g",
            self.magnitude, task, len(self.anchors),
            self.args.cosine_steps,
            int(self.args.cosine_steps * self.args.warmup_ratio),
            float(self.scheduler.get_last_lr()[0]),
        )

    def _load_anchors(self, cfg: TrainConfig) -> None:
        assert self.processor is not None
        with tempfile.TemporaryDirectory(prefix="coord_fence_") as tmp:
            image = Path(tmp) / "measure.png"
            Image.new("RGB", (32, 32), (18, 18, 18)).save(image)
            soft = measure_image_soft_tokens(self.processor, str(image))
        by_source = materialize(
            sources_from_manifest(), cfg,
            processor=self.processor, image_soft_tokens=soft,
        )
        kept: list[TrainingExample] = []
        for exs in by_source.values():
            self.rng.shuffle(exs)
            n_hold = _holdout_count(len(exs))
            if n_hold >= len(exs):
                raise RuntimeError(
                    f"holdout consumed every example of a source ({len(exs)})"
                )
            kept.extend(exs[n_hold:])
        if not kept:
            raise RuntimeError("anchor sources produced no training examples")
        self.anchors = kept

    def _build_optimizer(self, cfg: TrainConfig) -> None:
        import bitsandbytes as bnb

        assert self.model is not None
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("no trainable LoRA parameters")
        self.optimizer = bnb.optim.PagedAdamW8bit(
            params, lr=self.args.lr, weight_decay=0.0,
        )
        warmup = int(self.args.cosine_steps * self.args.warmup_ratio)
        total = self.args.cosine_steps
        floor = self.args.lr_floor

        def factor(step: int, _warmup: int = warmup, _total: int = total,
                   _floor: float = floor) -> float:
            return _factor(step, _warmup, _total, _floor)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, factor)
        self._params = params
        self._clip = cfg.max_grad_norm

    def close(self) -> None:
        if self.vram is not None:
            self.vram.finish()
            self.vram = None
        if self.embedder is not None:
            self.embedder.uninstall()
        if self.tmp is not None:
            shutil.rmtree(self.tmp, ignore_errors=True)
            self.tmp = None

    def _heartbeat(self, stage: str, source: str, packed_b: int, packed_t: int) -> None:
        assert self.tlog is not None
        self.tlog.heartbeat({
            "stage": stage,
            "source": source,
            "n_tokens": packed_t,
            "packed_B": packed_b,
            "packed_T": packed_t,
            "step": self.step,
            "gpu": _gpu_mem_snapshot(),
        })

    def _use_cache(self, on: bool) -> None:
        assert self.model is not None
        if on:
            self.model.eval()
            if hasattr(self.model, "gradient_checkpointing_disable"):
                self.model.gradient_checkpointing_disable()
            self.model.config.use_cache = True
            return
        self.model.config.use_cache = False
        self.model.train()
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def _arm(self, weights: torch.Tensor, rows: list[list[float]]) -> None:
        assert self.embedder is not None
        index = (weights[0] != 0).nonzero(as_tuple=False).flatten()
        if int(index.shape[0]) != len(rows):
            raise RuntimeError(
                f"coord rows {len(rows)} != reply tokens {int(index.shape[0])}"
            )
        coords = torch.tensor(rows, dtype=torch.float32, device=self.device)
        self.embedder.set_pending(coords.view(1, -1, 6), index.view(1, -1))

    def _jitter(self, points: tuple[tuple[float, float], ...],
                agent: tuple[float, float] | None) -> list[float]:
        for _ in range(100):
            row: list[float] = []
            for x, y in points:
                row.append(x + self.rng.gauss(0.0, REPLY_JITTER_SIGMA))
                row.append(y + self.rng.gauss(0.0, REPLY_JITTER_SIGMA))
            if agent is not None:
                if math.hypot(row[0] - agent[0], row[1] - agent[1]) > S_RADIUS:
                    continue
            if all(COORD_LOW <= value <= COORD_HIGH for value in row):
                return row
        raise RuntimeError("coordinate jitter stayed illegal")

    def _jitter_rows(
        self, points: tuple[tuple[float, float], ...],
        agent: tuple[float, float] | None, n: int,
    ) -> list[list[float]]:
        return [self._jitter(points, agent) for _ in range(n)]

    def _anchor_rows(self, n: int) -> list[list[float]]:
        s = (self.rng.random(), self.rng.random())
        v = (self.rng.random(), self.rng.random())
        v_bar = (
            self.rng.uniform(COORD_LOW, COORD_HIGH),
            self.rng.uniform(COORD_LOW, COORD_HIGH),
        )
        return self._jitter_rows((s, v, v_bar), None, n)

    def _anchor_loss(self) -> torch.Tensor:
        assert self.collator is not None and self.model is not None and self.vram is not None
        chosen = copy.copy(self.rng.choice(self.anchors))
        if chosen.loss == "kd":
            chosen.loss = "kd_anchor"
        elif chosen.loss != "ce":
            raise RuntimeError(f"anchor loss {chosen.loss!r} is not ce or kd")
        self.vram.set_stage("anchor")
        built = self.collator.build(chosen)
        ids = built["model_inputs"]["input_ids"]
        self._heartbeat("anchor", chosen.source, int(ids.shape[0]), int(ids.shape[1]))
        n = int((built["weights"][0] != 0).sum())
        self._arm(built["weights"], self._anchor_rows(n))
        return weighted_loss(
            self.model, built["model_inputs"], built["weights"],
            loss_kind=chosen.loss, example_weight=built["example_weight"],
        )

    def _sample_board(self) -> tuple[Any, tuple[float, float]]:
        game = game_io.new_multi_gold_game(
            gameSize=CONFIG.game_size,
            n_gold=self.rng.choice((0, 1, 2, 3)),
            opening="any",
        )
        settings = game.settings
        agent = (float(settings.agent_x), float(settings.agent_y))
        return game, agent

    def _sample_s(self, agent: tuple[float, float]) -> tuple[float, float]:
        ax, ay = agent
        for _ in range(200):
            sx = ax + self.rng.gauss(0.0, S_SIGMA)
            sy = ay + self.rng.gauss(0.0, S_SIGMA)
            if math.hypot(sx - ax, sy - ay) <= S_RADIUS:
                return sx, sy
        raise RuntimeError(
            f"s jitter never stayed within {S_RADIUS} of the agent at {agent}"
        )

    def _sample_v_for(
        self, game: Any, s: tuple[float, float], direction: str,
    ) -> tuple[float, float] | None:
        sx, sy = s
        kind = _direction_kind(direction)
        for _ in range(V_TRIES):
            vx = self.rng.random()
            vy = self.rng.random()
            if not game.full_wall_check(vx, vy, agent_r=1e-3):
                continue
            if not _separated(direction, sx, sy, vx, vy):
                continue
            if not hour_is_unique(sx, sy, vx, vy):
                continue
            look = gaze_vbar(sx, sy, vx, vy, looking=True)
            move = gaze_vbar(sx, sy, vx, vy, looking=False)
            if not (_in_box(*look) and _in_box(*move)):
                continue
            got = direction_label(kind, sx, sy, vx, vy)
            if got != direction:
                raise RuntimeError(
                    f"placed {direction} but the oracle says {got}"
                )
            return vx, vy
        return None

    def _place_image(self) -> dict[str, Any]:
        assert self.tmp is not None
        for _ in range(BOARD_TRIES):
            game, agent = self._sample_board()
            sx, sy = self._sample_s(agent)
            if not _in_box(sx, sy):
                continue
            placed: dict[str, tuple[float, float]] = {}
            good = True
            for direction in DIRECTIONS:
                found = self._sample_v_for(game, (sx, sy), direction)
                if found is None:
                    good = False
                    break
                placed[direction] = found
            if not good:
                continue
            specs = []
            for direction in DIRECTIONS:
                vx, vy = placed[direction]
                kind = _direction_kind(direction)
                hour = clock_hour(sx, sy, vx, vy)
                for looking in (True, False):
                    vbx, vby = gaze_vbar(sx, sy, vx, vy, looking=looking)
                    specs.append({
                        "points": ((sx, sy), (vx, vy), (vbx, vby)),
                        "agent": agent,
                        "dir_line": correct_line(kind, direction),
                        "hour_line": correct_line(KIND_HOUR, hour),
                    })
            frame = canonical_frame(game)
            path = self.tmp / f"tf_{self.step}_{time.time_ns()}.png"
            Image.fromarray(frame).save(path)
            return {"path": path, "specs": specs}
        raise RuntimeError(
            f"no board placed four directions in {BOARD_TRIES} draws"
        )

    def _collate_replies(
        self, prepared: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        assert self.collator is not None
        path = prepared["path"]
        builds: list[dict[str, Any]] = []
        metas: list[dict[str, Any]] = []
        for spec in prepared["specs"]:
            for question, line in (
                (QUESTION_DIRECTION, spec["dir_line"]),
                (QUESTION_HOUR, spec["hour_line"]),
            ):
                messages = _build_game_messages(
                    SYSTEM_PROMPT_S2, str(path), "", question,
                )
                example = TrainingExample(
                    messages=messages,
                    target_text=line,
                    loss="ce",
                    source="coord_tf",
                )
                builds.append(self.collator.build(example))
                metas.append({
                    "points": spec["points"],
                    "agent": spec["agent"],
                })
        if len(builds) != 16:
            raise RuntimeError(f"expected 16 replies, got {len(builds)}")
        return builds, metas

    def _prefill(self, model_inputs: dict[str, Any], lcp: int) -> Any:
        assert self.model is not None
        seq = int(model_inputs["input_ids"].shape[1])
        pref: dict[str, Any] = {
            "input_ids": model_inputs["input_ids"][:, :lcp],
            "attention_mask": torch.ones(
                1, lcp, dtype=torch.long, device=self.device,
            ),
            "position_ids": torch.arange(lcp, device=self.device).unsqueeze(0),
        }
        for key, val in model_inputs.items():
            if key in ("input_ids", "attention_mask") or not torch.is_tensor(val):
                continue
            if _token_aligned(key, val, seq):
                pref[key] = val[:, :lcp]
            else:
                pref[key] = val
        out = self.model(**pref, use_cache=True, logits_to_keep=1)
        if out.past_key_values is None:
            raise RuntimeError("prefix forward returned no past_key_values")
        return out.past_key_values

    def _suffix_pack(
        self,
        builds: list[dict[str, Any]],
        metas: list[dict[str, Any]],
        lcp: int,
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, list[tuple[int, int]], list[int]]:
        assert self.model is not None
        tails: list[torch.Tensor] = []
        extras_per_row: list[dict[str, torch.Tensor]] = []
        first_keys: list[str] | None = None
        reply_local: list[list[int]] = []
        predictors: list[tuple[int, int]] = []
        labels: list[int] = []
        for row, build in enumerate(builds):
            full = build["model_inputs"]
            seq = int(full["input_ids"].shape[1])
            tails.append(full["input_ids"][0, lcp:])
            sliced: dict[str, torch.Tensor] = {}
            keys: list[str] = []
            for key, val in full.items():
                if torch.is_tensor(val) and _token_aligned(key, val, seq):
                    keys.append(key)
                    sliced[key] = val[0, lcp:]
            if first_keys is None:
                first_keys = keys
            elif keys != first_keys:
                raise RuntimeError(f"suffix aux keys {keys} != {first_keys}")
            extras_per_row.append(sliced)
            weights = build["weights"][0]
            ids = full["input_ids"][0]
            index = (weights != 0).nonzero(as_tuple=False).flatten()
            if int(index.numel()) == 0:
                raise RuntimeError("a reply has no weighted tokens")
            local: list[int] = []
            real = seq - lcp
            for p in index.tolist():
                if p < lcp:
                    raise RuntimeError(
                        f"reply token at {p} is inside the cached prefix {lcp}"
                    )
                suf = p - lcp
                if suf >= real:
                    raise RuntimeError(
                        f"reply token {suf} is outside the suffix of length {real}"
                    )
                pred = suf - 1
                if pred < 0:
                    raise RuntimeError(
                        "a reply token is the first suffix token; "
                        "its predictor was in the cached prefix"
                    )
                local.append(suf)
                predictors.append((row, pred))
                labels.append(int(ids[p]))
            reply_local.append(local)
        max_real = max(int(tail.numel()) for tail in tails)
        max_reply = max(len(local) for local in reply_local)
        width = max_real + max_reply
        batch = len(builds)
        input_ids = torch.zeros(
            batch, width, dtype=tails[0].dtype, device=self.device,
        )
        attn_suf = torch.zeros(batch, width, dtype=torch.long, device=self.device)
        position_ids = torch.empty(batch, width, dtype=torch.long, device=self.device)
        coord_pos = torch.empty(
            batch, max_reply, dtype=torch.long, device=self.device,
        )
        coord_rows: list[list[list[float]]] = []
        for i, tail in enumerate(tails):
            n = int(tail.numel())
            input_ids[i, :n] = tail.to(self.device)
            attn_suf[i, :n] = 1
            position_ids[i] = lcp + torch.arange(width, device=self.device)
            extra = max_reply - len(reply_local[i])
            pads = list(range(n, width))
            if len(pads) < extra:
                raise RuntimeError(
                    f"row {i} has {len(pads)} pad slots for {extra} "
                    "extra coord positions"
                )
            chosen = reply_local[i] + pads[:extra]
            if len(set(chosen)) != len(chosen):
                raise RuntimeError(f"row {i} coord positions collide: {chosen}")
            coord_pos[i] = torch.tensor(chosen, dtype=torch.long, device=self.device)
            coord_rows.append(self._jitter_rows(
                metas[i]["points"], metas[i]["agent"], max_reply,
            ))
        past = torch.ones(batch, lcp, dtype=torch.long, device=self.device)
        inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": torch.cat([past, attn_suf], dim=1),
            "position_ids": position_ids,
        }
        for key in first_keys or []:
            sample = extras_per_row[0][key]
            stacked = torch.zeros(
                batch, width, dtype=sample.dtype, device=self.device,
            )
            for i, sliced in enumerate(extras_per_row):
                src = sliced[key]
                stacked[i, : int(src.numel())] = src.to(self.device)
            inputs[key] = stacked
        if _wants_cache_position(self.model):
            inputs["cache_position"] = torch.arange(
                lcp, lcp + width, device=self.device,
            )
        coords = torch.tensor(coord_rows, dtype=torch.float32, device=self.device)
        return inputs, coord_pos, coords, predictors, labels

    def _cached_reply_loss(
        self, builds: list[dict[str, Any]], metas: list[dict[str, Any]],
    ) -> torch.Tensor:
        assert self.model is not None and self.embedder is not None and self.vram is not None
        ids_rows = [build["model_inputs"]["input_ids"][0] for build in builds]
        lcp = _lcp_len(ids_rows)
        if lcp <= 0:
            raise RuntimeError("image batch has an empty common prefix")
        for row in ids_rows:
            if int(row.numel()) <= lcp:
                raise RuntimeError(
                    f"common prefix {lcp} consumed a row of length {int(row.numel())}"
                )
        image_at = max(_last_image_index(build["model_inputs"]) for build in builds)
        if image_at >= lcp:
            raise RuntimeError(
                f"common prefix of {lcp} tokens does not cover the image "
                f"(last image token at {image_at})"
            )
        self.vram.set_stage("image")
        with torch.no_grad():
            cache = self._prefill(builds[0]["model_inputs"], lcp)
        _detach_cache(cache)
        if _cache_len(cache) != lcp:
            raise RuntimeError(
                f"prefix cache length {_cache_len(cache)} != {lcp}"
            )
        self.model.train()
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        self.model.config.use_cache = True
        packed, positions, coords, predictors, labels = self._suffix_pack(
            builds, metas, lcp,
        )
        packed["past_key_values"] = expand_train_kv(cache, len(builds))
        width = int(packed["input_ids"].shape[1])
        self._heartbeat("image", "coord_tf", len(builds), lcp + width)
        self.embedder.set_pending(coords, positions)
        out = self.model(
            **packed, use_cache=True, output_hidden_states=True, logits_to_keep=1,
        )
        if out.past_key_values is None:
            raise RuntimeError(
                "suffix forward returned no past_key_values; "
                "the prefix cache was dropped"
            )
        hidden_states = getattr(out, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("suffix forward returned no hidden states")
        hidden = hidden_states[-1]
        if int(hidden.shape[1]) != width:
            raise RuntimeError(
                f"suffix hidden length {int(hidden.shape[1])} != {width}"
            )
        grown = _cache_len(out.past_key_values)
        if grown != lcp + width:
            raise RuntimeError(
                f"suffix cache length {grown} != prefix {lcp} + suffix {width}"
            )
        if not predictors:
            raise RuntimeError("image batch has no reply tokens")
        gathered = torch.stack(
            [hidden[row, index] for row, index in predictors], dim=0,
        )
        gathered = gathered.to(dtype=_lm_head_dtype(self.model))
        logits = apply_lm_head_chunked(self.model, gathered, 1024)
        target = torch.tensor(labels, dtype=torch.long, device=gathered.device)
        return torch.nn.functional.cross_entropy(logits.float(), target)

    def _image_loss(self) -> tuple[torch.Tensor, Path]:
        prepared = self._place_image()
        path: Path = prepared["path"]
        try:
            builds, metas = self._collate_replies(prepared)
            self._use_cache(True)
            try:
                return self._cached_reply_loss(builds, metas), path
            finally:
                self._use_cache(False)
        except Exception:
            path.unlink(missing_ok=True)
            raise

    def _finite(self, value: float, what: str) -> None:
        if math.isfinite(value):
            return
        assert self.tlog is not None
        self.tlog.event("nonfinite_" + what, step=self.step, value=value)
        raise RuntimeError(f"non-finite {what} at step {self.step}: {value}")

    def _log(self, loss_value: float, grad_norm: float) -> None:
        assert self.tlog is not None and self.scheduler is not None
        lr = float(self.scheduler.get_last_lr()[0])
        rewards = self._rl_rewards
        parsed = self._rl_parsed
        reward_mean = sum(rewards) / len(rewards) if rewards else None
        parse_rate = (
            sum(1 for item in parsed if item) / len(parsed) if parsed else None
        )
        record = {
            "step": self.step,
            "loss": loss_value,
            "lr": lr,
            "grad_norm": grad_norm,
            "reward_mean": reward_mean,
            "parse_rate": parse_rate,
            "magnitude": self.magnitude,
        }
        self.tlog.last_step(record)
        if self.step % self.args.log_steps == 0:
            self.tlog.step(record)
            reward_s = "n/a" if reward_mean is None else f"{reward_mean:.3f}"
            parse_s = "n/a" if parse_rate is None else f"{parse_rate:.3f}"
            logger.info(
                "step %d loss %.4f lr %.3g grad %.3f reward %s parse %s",
                self.step, loss_value, lr, grad_norm, reward_s, parse_s,
            )

    def save(self, reason: str) -> None:
        assert self.model is not None and self.embedder is not None and self.tlog is not None
        meta = {
            "trainer": "s2_coord",
            "gemma_base": self.args.gemma_base,
            "step": self.step,
            "magnitude": self.magnitude,
            "reason": reason,
            "signal": self._signal,
        }
        root = weights_root() / self.spec.key
        for directory in (
            root / f"{self.label}_step_{self.step:06d}",
            root / f"{self.label}_last",
        ):
            save_checkpoint(self.model, directory, meta, self.tlog)
            self.embedder.save(directory / "coord_embed.pt")
        logger.info("saved step %d (%s)", self.step, reason)

    def _example_loss(self, kind: str) -> tuple[torch.Tensor | None, Path | None]:
        if kind == "anchor":
            return self._anchor_loss(), None
        if kind == "image":
            return self._image_loss()
        if kind == "rl":
            return rl_loss(self)
        raise ValueError(f"unknown example kind {kind!r}")

    def train(self) -> None:
        assert self.optimizer is not None and self.scheduler is not None
        assert self.model is not None and self.tlog is not None
        deadline = time.perf_counter() + self.args.hours * 3600.0
        self.optimizer.zero_grad(set_to_none=True)
        accum = 0
        loss_sum = 0.0
        self.model.train()
        while time.perf_counter() < deadline and not self._stop:
            if self.rng.random() < 0.5:
                kind = "anchor"
            elif self.args.rl:
                kind = "rl"
            else:
                kind = "image"
            loss = None
            cleanup: Path | None = None
            loss_value = 0.0
            assert self.embedder is not None
            try:
                loss, cleanup = self._example_loss(kind)
                if loss is not None:
                    loss_value = float(loss.detach())
                    self._finite(loss_value, "loss")
                    loss.backward()
            finally:
                self.embedder.clear_pending()
                if cleanup is not None:
                    cleanup.unlink(missing_ok=True)
            if loss is None:
                continue
            accum += 1
            loss_sum += loss_value
            if accum < self.args.grad_accum:
                continue
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                self._params, self._clip,
            ))
            self._finite(grad_norm, "grad_norm")
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            self._log(loss_sum / accum, grad_norm)
            accum = 0
            loss_sum = 0.0
            if self.args.save_steps > 0 and self.step % self.args.save_steps == 0:
                self.save("step")
        reason = "signal" if self._signal else "hours"
        self.save(reason)

    def _bench_step(self, kind: str, samples: list[tuple[int, int]] | None) -> None:
        assert self.embedder is not None
        assert self.optimizer is not None and self.scheduler is not None
        loss: torch.Tensor | None = None
        cleanup: Path | None = None
        try:
            loss, cleanup = self._example_loss(kind)
            if loss is None:
                raise RuntimeError(f"bench {kind} produced no loss")
            self._finite(float(loss.detach()), "loss")
            if samples is not None:
                pair = _nvidia_smi_used_total()
                if pair is not None:
                    samples.append(pair)
            loss.backward()
            if samples is not None:
                pair = _nvidia_smi_used_total()
                if pair is not None:
                    samples.append(pair)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                self._params, self._clip,
            ))
            self._finite(grad_norm, "grad_norm")
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
        finally:
            self.embedder.clear_pending()
            if cleanup is not None:
                cleanup.unlink(missing_ok=True)

    def _bench_report(
        self, elapsed: float, samples: list[tuple[int, int]], torch_peak: float,
    ) -> None:
        assert self.tlog is not None
        examples = 4
        per_s = examples / elapsed
        accum = max(1, int(self.args.grad_accum))
        steps_per_s = per_s / accum
        lines = [
            "bench",
            "  warmup    1 anchor, 1 image",
            "  timed     anchor, image, anchor, image",
            f"  steps/s   {steps_per_s:.4f}",
        ]
        if self.args.rl:
            lines.insert(
                3,
                "  note      timed the teacher-force image; --rl was not measured",
            )
        if accum != 1:
            lines.append(
                f"  grad_accum {accum}  (steps/s counts optimizer steps)"
            )
        if samples:
            used = max(item[0] for item in samples)
            total = samples[-1][1]
            lines.append(
                f"  vram      nvidia-smi {used / 1024:.1f} / {total / 1024:.1f} "
                f"GiB ({100.0 * used / total:.0f}%)"
            )
        else:
            lines.append("  vram      nvidia-smi unavailable")
        lines.append(f"            torch peak {torch_peak:.1f} GiB")
        monday = _hours_until_monday()
        steps = None
        hours = None
        if monday is None:
            lines.append(
                "  monday    08:00 ET has passed; no --hours or --cosine-steps"
            )
        else:
            target, hours = monday
            steps = int(round(steps_per_s * hours * 3600.0))
            lines.append(f"  monday    {target.strftime('%Y-%m-%d %H:%M')} ET")
            lines.append(f"  hours     {hours:.2f}")
            lines.append(f"  steps     {steps}")
            if steps >= 1:
                command = [
                    "python -m neural_net.s2_coord_train \\",
                    f"  --label {self.label} \\",
                ]
                if accum != 1:
                    command.append(f"  --grad-accum {accum} \\")
                command.append(f"  --hours {hours:.2f} \\")
                command.append(f"  --cosine-steps {steps}")
                lines.append("")
                lines.extend(command)
            else:
                lines.append(
                    "  the measured rate does not finish one optimizer step "
                    "before Monday"
                )
        text = "\n".join(lines)
        print(text, flush=True)
        self.tlog.event(
            "bench",
            steps_per_s=steps_per_s,
            hours=hours,
            cosine_steps=steps,
            torch_peak_gib=torch_peak,
            nvidia_smi=samples,
        )

    def bench(self) -> None:
        """Six optimizer steps: one warmup of each path, then two timed of each."""
        assert self.optimizer is not None
        self.optimizer.zero_grad(set_to_none=True)
        self._bench_step("anchor", None)
        self._bench_step("image", None)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        samples: list[tuple[int, int]] = []
        started = time.perf_counter()
        for kind in ("anchor", "image", "anchor", "image"):
            self._bench_step(kind, samples)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if elapsed <= 0:
            raise RuntimeError("bench timer did not advance")
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        else:
            peak = 0.0
        self._bench_report(elapsed, samples, peak)

    def _on_signal(self, signum: int, _frame: Any) -> None:
        self._stop = True
        self._signal = int(signum)
        logger.warning("signal %s; will save at the next step boundary", signum)


def _write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _argv_value(argv: list[str], name: str) -> str | None:
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return None


def _dmesg_oom_lines(pid: int) -> list[str]:
    try:
        proc = subprocess.run(
            ["dmesg", "-T"], capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"dmesg failed: {exc}"]
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return [f"dmesg exit {proc.returncode}: {err[:500]}"]
    recent: list[str] = []
    matched: list[str] = []
    needle = str(pid)
    for line in proc.stdout.splitlines():
        low = line.lower()
        if "out of memory" not in low and "killed process" not in low:
            continue
        recent.append(line)
        if needle in line:
            matched.append(line)
    return (matched or recent)[-40:]


def _signal_name(rc: int) -> str | None:
    if rc < 0:
        try:
            return signal.Signals(-rc).name
        except ValueError:
            return f"SIG{-rc}"
    if rc > 128:
        try:
            return signal.Signals(rc - 128).name
        except ValueError:
            return None
    return None


def _supervise(argv: list[str]) -> int:
    label = _argv_value(argv, "--label") or "coord"
    child = list(argv)
    if "--no-supervise" not in child:
        child.append("--no-supervise")
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "neural_net.s2_coord_train", *child],
    )
    forwarded: list[int] = []

    def _forward(signum: int, _frame: Any) -> None:
        forwarded.append(int(signum))
        if proc.poll() is None:
            proc.send_signal(signum)

    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT, _forward)
    rc = proc.wait()
    name = _signal_name(rc)
    if name is None and forwarded:
        try:
            name = signal.Signals(forwarded[-1]).name
        except ValueError:
            name = f"SIG{forwarded[-1]}"
    record = {
        "label": label,
        "pid": proc.pid,
        "exit_code": rc,
        "signal": name,
        "dmesg": _dmesg_oom_lines(proc.pid),
    }
    out = Path("logs") / f"train_{label}_supervise.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(out, record)
    print(
        f"supervise: child {proc.pid} exit {rc} signal {name} -> {out}",
        flush=True,
    )
    return rc if rc >= 0 else 128 + (-rc)


def _is_cuda_oom(exc: BaseException) -> bool:
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    text = str(exc).lower()
    return "out of memory" in text and "cuda" in text


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", default="coord_tf")
    parser.add_argument("--gemma-base", default="aug27_big_step_iter1_step313")
    parser.add_argument(
        "--magnitude", default="half",
        help="barely, half, recommended, or a float in (0, 1). "
             "tanh of every gate. Frozen for the run. Default half (0.5).",
    )
    parser.add_argument(
        "--rl", action="store_true",
        help="Archived online sampler instead of the sixteen teacher-forced "
             "replies. The anchor coin stays.",
    )
    parser.add_argument(
        "--bench", action="store_true",
        help="Load, warm one anchor and one image, time two of each, "
             "print VRAM and the Monday --hours / --cosine-steps, then exit.",
    )
    parser.add_argument("--hours", type=float, default=0.0)
    parser.add_argument("--cosine-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-floor", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-supervise", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not args.no_supervise:
        raise SystemExit(_supervise(sys.argv[1:]))
    configure_logging()
    if not args.bench:
        if args.hours <= 0:
            raise SystemExit("--hours must be positive")
        if args.cosine_steps <= 0:
            raise SystemExit("--cosine-steps must be positive")
    if args.grad_accum < 1:
        raise SystemExit("--grad-accum must be >= 1")
    resolve_magnitude(args.magnitude)
    trainer = Trainer(args)
    signal.signal(signal.SIGTERM, trainer._on_signal)
    signal.signal(signal.SIGINT, trainer._on_signal)
    try:
        trainer.load()
        if args.bench:
            trainer.bench()
        else:
            trainer.train()
    except BaseException as exc:
        if trainer.tlog is not None and not isinstance(exc, KeyboardInterrupt):
            oom = _is_cuda_oom(exc)
            trainer.tlog.record_crash(
                exc, step=trainer.step, **({"tag": "cuda_oom"} if oom else {}),
            )
            if oom:
                trainer.tlog.event(
                    "cuda_oom", step=trainer.step, error=str(exc)[:500],
                )
        if (trainer.model is not None and not args.bench
                and not isinstance(exc, KeyboardInterrupt)):
            try:
                trainer.save("crash")
            except Exception:
                logger.error("crash snapshot failed\n%s", traceback.format_exc())
        if (isinstance(exc, KeyboardInterrupt) and trainer.model is not None
                and not args.bench):
            trainer._signal = int(signal.SIGINT)
            try:
                trainer.save("signal")
            except Exception:
                logger.error("signal snapshot failed\n%s", traceback.format_exc())
        raise
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
