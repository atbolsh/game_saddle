"""Phase-2 coord embedder: aug27 LoRA reads a frozen Fourier code.

Anchors are the manifest datasets. CE sources train the dataset targets.
KD sources match the frozen aug27 adapter with the coordinate code
muted. The student forward has the code on, on reply tokens only.

The other half of the examples are online replies about s, v, and
v_bar. The oracle scores a single required line. A reward of 0 is
skipped. One quarter of those slots are teacher-forced on the correct
line so the format has a gradient before the policy emits it.

``--no-rl`` drops that half. ``--hours`` requires ``--cosine-steps``.
Past that horizon the learning-rate multiplier stays at the floor.
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
from pathlib import Path
from typing import Any

import torch

from agent import game_io
from agent.config import CONFIG
from agent.memory import format_notepad
from agent.model import ADAPTERS, spec_for
from agent.modes import SYSTEM_PROMPT_S2, _build_game_messages
from neural_net.coord_embed import CoordEmbedder, resolve_magnitude
from neural_net.paths import weights_root
from neural_net.render import canonical_frame
from neural_net.s2 import _hidden_size, _wants_cache_position
from neural_net.s2_coord_oracle import (
    KIND_HOUR,
    KIND_LEFTRIGHT,
    KIND_LOOKING,
    KIND_MOVING,
    KIND_UPDOWN,
    KINDS,
    REPLY_JITTER_SIGMA,
    S_RADIUS,
    S_SIGMA,
    clock_hour,
    correct_line,
    direction_label,
    hour_is_unique,
    parse_reply,
    reply_reward,
)
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
    build_model,
    configure_logging,
    load_adapter_state,
    materialize,
    resolve_terminator_id,
    save_checkpoint,
    weighted_loss,
)

logger = logging.getLogger("s2_coord")

BOOTSTRAP = 0.25
HOLDOUT_FRACTION = 0.05
HOLDOUT_CAP = 100
MAX_NEW_TOKENS = 24
COORD_LOW = -1.0
COORD_HIGH = 2.0
V_TRIES = 400
NOTEPAD = [{"key": "target", "value": "the upper-left gold", "updated_round": 1}]


def _factor(step: int, warmup: int, total: int, floor: float) -> float:
    """Warmup, then cosine down to ``floor``, then stay there."""
    if step >= total:
        return floor
    if warmup > 0 and step < warmup:
        return step / max(1, warmup)
    span = max(1, total - warmup)
    progress = (step - warmup) / span
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _question(kind: str) -> str:
    if kind == KIND_UPDOWN:
        return (
            "Are you looking up or down? Reply with exactly one line and "
            "no other text:\nLooking: up\nor\nLooking: down"
        )
    if kind == KIND_LEFTRIGHT:
        return (
            "Are you looking left or right? Reply with exactly one line and "
            "no other text:\nLooking: left\nor\nLooking: right"
        )
    if kind == KIND_HOUR:
        return (
            "Which clock hour are you looking at? Twelve o'clock is straight "
            "up, three is straight right, six is straight down, nine is "
            "straight left. Reply with exactly one line and no other text:\n"
            "Hour: N\n"
            "N is an integer from 1 to 12. No half-hours."
        )
    if kind == KIND_LOOKING:
        return (
            "Are you looking right now? Reply with exactly one line and "
            "no other text:\nAnswer: yes\nor\nAnswer: no"
        )
    if kind == KIND_MOVING:
        return (
            "Are you moving right now? Reply with exactly one line and "
            "no other text:\nAnswer: yes\nor\nAnswer: no"
        )
    raise ValueError(f"unknown kind {kind!r}")


def _holdout_count(n: int) -> int:
    return min(HOLDOUT_CAP, max(1, int(n * HOLDOUT_FRACTION)))


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
        logger.info(
            "magnitude tanh=%.4f frozen | anchors %d | cosine %d warmup %d | lr %.3g",
            self.magnitude, len(self.anchors),
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

    def _anchor_rows(self, n: int) -> list[list[float]]:
        s = (self.rng.random(), self.rng.random())
        v = (self.rng.random(), self.rng.random())
        v_bar = (
            self.rng.uniform(COORD_LOW, COORD_HIGH),
            self.rng.uniform(COORD_LOW, COORD_HIGH),
        )
        return [self._jitter((s, v, v_bar), None) for _ in range(n)]

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

    def _sample_v(self, game: Any) -> tuple[float, float]:
        for _ in range(V_TRIES):
            vx = self.rng.random()
            vy = self.rng.random()
            if game.full_wall_check(vx, vy, agent_r=1e-3):
                return vx, vy
        raise RuntimeError(f"no v outside a wall after {V_TRIES} draws")

    def _scenario(self) -> dict[str, Any] | None:
        """One labeled question, or None when the draw was ambiguous."""
        assert self.tmp is not None
        game, agent = self._sample_board()
        kind = self.rng.choice(KINDS)
        mode = "look" if self.rng.random() < 0.5 else "move"
        sx, sy = self._sample_s(agent)
        vx, vy = self._sample_v(game)
        if kind in (KIND_UPDOWN, KIND_LEFTRIGHT):
            expected: str | int | None = direction_label(kind, sx, sy, vx, vy)
            if expected is None:
                return None
        elif kind == KIND_HOUR:
            if not hour_is_unique(sx, sy, vx, vy):
                return None
            expected = clock_hour(sx, sy, vx, vy)
        elif kind == KIND_LOOKING:
            expected = "yes" if mode == "look" else "no"
        elif kind == KIND_MOVING:
            expected = "yes" if mode == "move" else "no"
        else:
            raise RuntimeError(f"unhandled kind {kind}")
        if mode == "move":
            vbx, vby = vx, vy
        else:
            vbx, vby = 2.0 * sx - vx, 2.0 * sy - vy
        if not (COORD_LOW <= vbx <= COORD_HIGH and COORD_LOW <= vby <= COORD_HIGH):
            raise RuntimeError(
                f"v_bar {(vbx, vby)} outside [-1, 2] for s {(sx, sy)} v {(vx, vy)}"
            )
        frame = canonical_frame(game)
        path = self.tmp / f"rl_{self.step}_{time.time_ns()}.png"
        Image.fromarray(frame).save(path)
        try:
            notepad = format_notepad(NOTEPAD) if self.rng.random() < 0.5 else None
            messages = _build_game_messages(
                SYSTEM_PROMPT_S2, str(path), "", _question(kind), notepad=notepad,
            )
            line = correct_line(kind, expected)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return {
            "kind": kind,
            "expected": expected,
            "agent": agent,
            "points": ((sx, sy), (vx, vy), (vbx, vby)),
            "messages": messages,
            "path": path,
            "line": line,
        }

    def _rows_for(self, scenario: dict[str, Any], n: int) -> list[list[float]]:
        points = scenario["points"]
        agent = scenario["agent"]
        return [self._jitter(points, agent) for _ in range(n)]

    def _begin_sample(self) -> None:
        assert self.model is not None
        self.model.eval()
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        self.model.config.use_cache = True

    def _end_sample(self) -> None:
        assert self.model is not None
        self.model.config.use_cache = False
        self.model.train()
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def _generate(
        self, scenario: dict[str, Any],
    ) -> tuple[str, list[int], list[list[float]]]:
        assert self.model is not None and self.processor is not None and self.embedder is not None
        assert self.collator is not None
        tokenizer = self.collator.tokenizer
        norm = self.collator.adapter.prepare_messages(scenario["messages"])
        prompt = self.processor.apply_chat_template(
            norm, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        inputs: dict[str, Any] = {}
        for key, val in prompt.items():
            if not isinstance(val, torch.Tensor):
                continue
            if val.dtype.is_floating_point:
                val = val.to(torch.bfloat16)
            inputs[key] = val.to(self.device)
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        produced: list[int] = []
        rows: list[list[float]] = []
        prompt_len = int(inputs["input_ids"].shape[1])
        self._heartbeat("rl-generate", scenario["kind"], 1, prompt_len)
        self._begin_sample()
        try:
            with torch.inference_mode():
                out = self.model(**inputs, use_cache=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :]
                seq_len = prompt_len
                for _step in range(MAX_NEW_TOKENS):
                    probs = torch.softmax(logits.float(), dim=-1)
                    next_id = int(torch.multinomial(probs.reshape(-1), 1).item())
                    if next_id == self.terminator:
                        break
                    produced.append(next_id)
                    row = self._rows_for(scenario, 1)[0]
                    rows.append(row)
                    seq_len += 1
                    step_inputs: dict[str, Any] = {
                        "input_ids": torch.tensor([[next_id]], device=self.device),
                        "attention_mask": torch.ones(
                            1, seq_len, dtype=torch.long, device=self.device,
                        ),
                        "past_key_values": past,
                    }
                    if _wants_cache_position(self.model):
                        step_inputs["cache_position"] = torch.tensor(
                            [seq_len - 1], device=self.device,
                        )
                    coord = torch.tensor(row, dtype=torch.float32, device=self.device)
                    positions = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                    self.embedder.set_pending(coord.view(1, 1, 6), positions)
                    try:
                        out = self.model(**step_inputs, use_cache=True)
                    finally:
                        self.embedder.clear_pending()
                    past = out.past_key_values
                    logits = out.logits[:, -1, :]
        finally:
            self._end_sample()
        if not produced:
            return "", [], []
        text = tokenizer.decode(produced, skip_special_tokens=True)
        again = tokenizer(text, add_special_tokens=False)["input_ids"]
        if list(again) != produced:
            raise RuntimeError(
                "generated tokens do not re-encode; refusing a misaligned reply. "
                f"ids {produced} text {text!r} re-encoded {list(again)}"
            )
        return text, produced, rows

    def _rl_loss(self) -> tuple[torch.Tensor | None, Path]:
        assert self.collator is not None and self.model is not None and self.vram is not None
        scenario = None
        for _ in range(50):
            scenario = self._scenario()
            if scenario is not None:
                break
        if scenario is None:
            raise RuntimeError("50 board draws had no unambiguous label")
        path: Path = scenario["path"]
        try:
            self.vram.set_stage("rl")
            bootstrap = self.rng.random() < BOOTSTRAP
            stored: list[list[float]] | None
            if bootstrap:
                text = scenario["line"]
                weight = 1.0
                stored = None
            else:
                text, _ids, stored = self._generate(scenario)
                reward = reply_reward(scenario["kind"], text, scenario["expected"])
                parsed = parse_reply(scenario["kind"], text) is not None
                self._rl_rewards.append(reward)
                self._rl_parsed.append(parsed)
                self._rl_rewards = self._rl_rewards[-50:]
                self._rl_parsed = self._rl_parsed[-50:]
                weight = reward
            if weight == 0.0:
                return None, path
            example = TrainingExample(
                messages=scenario["messages"],
                target_text=text,
                loss="ce",
                source="coord_rl",
                example_weight=weight,
            )
            built = self.collator.build(example)
            ids = built["model_inputs"]["input_ids"]
            self._heartbeat(
                "rl", scenario["kind"], int(ids.shape[0]), int(ids.shape[1]),
            )
            n = int((built["weights"][0] != 0).sum())
            if stored is None:
                rows = self._rows_for(scenario, n)
            else:
                if n != len(stored) + 1:
                    raise RuntimeError(
                        f"reply tokens {n} != generated {len(stored)} plus the terminator"
                    )
                rows = stored + self._rows_for(scenario, 1)
            self._arm(built["weights"], rows)
            loss = weighted_loss(
                self.model, built["model_inputs"], built["weights"],
                loss_kind="ce", example_weight=built["example_weight"],
            )
            return loss, path
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

    def train(self) -> None:
        assert self.optimizer is not None and self.scheduler is not None
        assert self.model is not None and self.tlog is not None
        deadline = time.perf_counter() + self.args.hours * 3600.0
        self.optimizer.zero_grad(set_to_none=True)
        accum = 0
        loss_sum = 0.0
        self.model.train()
        while time.perf_counter() < deadline and not self._stop:
            assert self.embedder is not None
            loss = None
            cleanup: Path | None = None
            loss_value = 0.0
            try:
                if self.args.no_rl or self.rng.random() < 0.5:
                    loss = self._anchor_loss()
                else:
                    loss, cleanup = self._rl_loss()
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
    parser.add_argument("--label", default="coord_phase2")
    parser.add_argument("--gemma-base", default="aug27_big_step_iter1_step313")
    parser.add_argument(
        "--magnitude", default="recommended",
        help="barely, half, recommended, or a float in (0, 1). "
             "tanh of every gate. Frozen for the run.",
    )
    parser.add_argument("--no-rl", action="store_true")
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--cosine-steps", type=int, required=True)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--lr-floor", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--grad-accum", type=int, default=4)
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
        if trainer.model is not None and not isinstance(exc, KeyboardInterrupt):
            try:
                trainer.save("crash")
            except Exception:
                logger.error("crash snapshot failed\n%s", traceback.format_exc())
        if isinstance(exc, KeyboardInterrupt) and trainer.model is not None:
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
