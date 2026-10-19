"""Weekend joint S2 run: LoRA and the localizer, S1 frozen.

One example is one image (or one anchor). The loss is reply-token CE
plus the localizer's L2 and cell CE on detached tap-layer states. The
coordinate code stays the frozen code in the start checkpoint.

On a token through the ``]`` that closes ``[REMEMBER ...]``, and on
every token of a sealed reply or ``End the game.``, the injected ``v``
and the localizer's L2 target are ``s``. A ``]`` that only closes
``[MOVE]`` or ``[END_GAME]`` does not start that span. Cell CE for
``v`` is still ignored on midpoint tokens.

Launch (Monday 08:00 America/New_York, cosine from the startup bench):

    cd /workspace/game_saddle && source /venv/main/bin/activate && PYTHONUNBUFFERED=1 python -m neural_net.s2_weekend_train --start localizer_finetuned_v1 --label joint_v1 --monday
"""

from __future__ import annotations

import argparse
import inspect
import logging
import math
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

from agent.config import CONFIG
from agent.model import ADAPTERS
from neural_net.coord_embed import CoordEmbedder
from neural_net.localizer import (
    KEPT_LAYER,
    LayerTap,
    Localizer,
    cell_index,
    gather_grid,
    grid_positions,
)
from neural_net.paths import full_weights_dir
from neural_net.s2 import (
    _hidden_size,
    assert_readout_blob,
    write_gemma_meta,
    write_readout,
)
from neural_net.s2_coord_train import (
    Trainer,
    _argv_value,
    _detach_cache,
    _dmesg_oom_lines,
    _factor,
    _holdout_count,
    _hours_until_monday,
    _is_cuda_oom,
    _last_image_index,
    _lcp_len,
    _lm_head_dtype,
    _signal_name,
    _token_aligned,
    _write_json_atomic,
)
from neural_net.s2_pretraining_localizer import (
    MODE_FULL,
    MODE_MID,
    _close_content_index,
)
from neural_net.s2_weekend_eval import (
    build_train_family,
    messages_for,
    reasoning_items,
    run_eval,
)
from training.external_data import sources_from_manifest
from training.run_weekend import VramMonitor
from training.token_fence import measure_image_soft_tokens
from training.train import (
    ANCHOR_ADAPTER,
    Collator,
    TrainConfig,
    TrainLogger,
    TrainingExample,
    apply_lm_head_chunked,
    build_model,
    configure_logging,
    expand_train_kv,
    load_adapter_state,
    materialize,
    resolve_terminator_id,
)

logger = logging.getLogger("s2_weekend")

DEFAULT_MIX = {
    "A": 0.30,
    "B": 0.10,
    "C": 0.20,
    "D": 0.15,
    "E": 0.15,
    "F": 0.10,
}
FAMILY_ORDER = ("A", "B", "C", "D", "E", "F")
STAGE_NAME = {
    "A": "gaze",
    "B": "gaze_game",
    "C": "look_move",
    "D": "begin",
    "E": "retarget",
    "F": "anchor",
}


def parse_mix(text: str | None) -> dict[str, float]:
    """Override weights. A zero drops that family; the rest are renormalized."""
    mix = dict(DEFAULT_MIX)
    if text:
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            key, sep, raw = part.partition("=")
            if not sep:
                raise SystemExit(f"--mix entry {part!r} is not KEY=WEIGHT")
            key = key.strip()
            if key not in DEFAULT_MIX:
                raise SystemExit(f"--mix key {key!r} is not one of A,B,C,D,E,F")
            try:
                mix[key] = float(raw)
            except ValueError as exc:
                raise SystemExit(f"--mix weight {raw!r} is not a float") from exc
    if any(value < 0.0 for value in mix.values()):
        raise SystemExit("--mix weights must be >= 0")
    kept = {
        key: mix[key] for key in FAMILY_ORDER if mix[key] > 0.0
    }
    total = sum(kept.values())
    if total <= 0.0:
        raise SystemExit("--mix dropped every family")
    return {key: kept[key] / total for key in kept}


def supervision_span(tokenizer: Any, reply: str, all_mid: bool) -> tuple[int, int]:
    """Weighted reply tokens (content plus the terminator) and how many are mid.

    Mid is the ``[REMEMBER ...]`` span, through the token that contains
    its closing ``]``, and not the terminator. A reply with no
    ``[REMEMBER`` has no mid tokens, so a bare ``[MOVE]`` stays a full
    look at the named place. ``all_mid`` marks the whole reply, including
    the terminator: sealed rooms and ``End the game.``.
    """
    ids = tokenizer(reply, add_special_tokens=False)["input_ids"]
    n_content = len(ids)
    if n_content <= 0:
        raise RuntimeError(f"empty reply: {reply!r}")
    n = n_content + 1
    if all_mid:
        return n, n
    remember_at = reply.find("[REMEMBER")
    if remember_at < 0:
        return n, 0
    bracket_at = reply.find("]", remember_at)
    if bracket_at < 0:
        raise RuntimeError(f"[REMEMBER reply has no closing bracket: {reply!r}")
    earlier = reply.find("]")
    if earlier < remember_at:
        raise RuntimeError(f"a ] appears before [REMEMBER in {reply!r}")
    close = _close_content_index(tokenizer, reply)
    if close < 0 or close >= n_content:
        raise RuntimeError(
            f"closing bracket index {close} outside 0..{n_content - 1} in {reply!r}"
        )
    return n, close + 1


def _tokens_seen(cache: Any) -> int:
    """Tokens the cache has seen. Not the sliding layer's stored key count."""
    if not hasattr(cache, "layers") or not cache.layers:
        raise RuntimeError(f"prefix cache has no layers: {type(cache)!r}")
    layer = cache.layers[0]
    if not hasattr(layer, "get_seq_length"):
        raise RuntimeError(
            f"prefix cache layer 0 ({type(layer).__name__}) has no get_seq_length"
        )
    return int(layer.get_seq_length())


class WeekendTrainer(Trainer):
    def __init__(self, args: argparse.Namespace) -> None:
        args.magnitude = "half"
        args.rl = False
        args.gemma_base = args.start
        super().__init__(args)
        self.mix = parse_mix(args.mix)
        self.localizer: Localizer | None = None
        self.tap: LayerTap | None = None
        self.holdout: list[TrainingExample] = []
        self.start_dir = full_weights_dir() / args.start
        self.cosine_steps = 0
        self.deadline = 0.0
        self._hours = 0.0
        self._next_eval = 0.0
        self._last_eval_step = -1
        self.opt_lora: Any = None
        self.opt_loc: Any = None
        self.sched_lora: Any = None
        self.sched_loc: Any = None
        self._lora_params: list[Any] = []
        self._loc_params: list[Any] = []
        self._win_sum: dict[str, float] = {}
        self._win_n: dict[str, int] = {}
        self._grid: torch.Tensor | None = None

    def load(self) -> None:
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        self.rng = random.Random(self.args.seed)
        self.tlog = TrainLogger(self.label)
        self.vram = VramMonitor(self.label)
        self._require_start()
        # The GSM8K probe has to be on disk before the trunk loads.
        reasoning_items()
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
        gemma = self.start_dir / "gemma"
        load_adapter_state(model, gemma)
        model.load_adapter(str(gemma), adapter_name=ANCHOR_ADAPTER, is_trainable=False)
        model.set_adapter("default")
        tokenizer = getattr(processor, "tokenizer", processor)
        self.terminator = resolve_terminator_id(model, tokenizer)
        self.collator = Collator(
            processor, ADAPTERS[self.spec.family], self.terminator,
            compute_dtype=torch.bfloat16, device=self.device,
        )
        self.processor = processor
        hidden = _hidden_size(model)
        embedder = CoordEmbedder(hidden).to(device=self.device, dtype=torch.float32)
        embedder.load(self.start_dir / "coord_embed.pt")
        if not embedder.enabled:
            raise RuntimeError(
                f"{self.start_dir / 'coord_embed.pt'} has enabled=False; "
                "this run needs the trained coordinate code"
            )
        embedder.freeze_readout()
        embedder.install(model)
        self.embedder = embedder
        blob = torch.load(
            self.start_dir / "localizer.pt", map_location="cpu", weights_only=True,
        )
        localizer = Localizer(hidden).to(device=self.device, dtype=torch.float32)
        kept = assert_readout_blob(
            blob, localizer, path=str(self.start_dir / "localizer.pt"),
        )
        if kept != KEPT_LAYER:
            raise RuntimeError(f"localizer kept_layer {kept} != {KEPT_LAYER}")
        localizer.load_state_dict(blob["state_dict"])
        localizer.to(device=self.device, dtype=torch.float32)
        for param in localizer.parameters():
            param.requires_grad_(True)
        self.localizer = localizer
        self.tap = LayerTap(model, KEPT_LAYER)
        self.model = model
        self._load_anchors(cfg)
        self._build_optimizers()
        self.tmp = Path(tempfile.mkdtemp(prefix=f"weekend_{self.label}_"))
        self.model.train()
        self.localizer.train()
        logger.info(
            "loaded %s | mix %s | lora lr %.3g localizer lr %.3g | holdout %d anchors %d",
            self.start_dir,
            " ".join(f"{key}={self.mix[key]:.3f}" for key in self.mix),
            self.args.lr,
            self.args.localizer_lr,
            len(self.holdout),
            len(self.anchors),
        )

    def close(self) -> None:
        if self.tap is not None:
            self.tap.remove()
            self.tap = None
        super().close()

    def place_gaze(self, rng: random.Random, tmp: Path, step: int) -> dict[str, Any]:
        """Coord-tf placement using ``rng`` and ``tmp`` instead of the train stream."""
        old = (self.rng, self.tmp, self.step)
        self.rng, self.tmp, self.step = rng, tmp, step
        try:
            return self._place_image()
        finally:
            self.rng, self.tmp, self.step = old

    def prepare_horizon(self) -> None:
        if self.opt_lora is None or self.opt_loc is None:
            raise RuntimeError("prepare_horizon before load")
        rate: float | None
        if int(self.args.cosine_steps) > 0:
            self.cosine_steps = int(self.args.cosine_steps)
            rate = None
        else:
            rate = self._bench_rate()
            remaining = self.deadline - time.monotonic()
            if remaining <= 0.0:
                raise RuntimeError("deadline passed during the startup bench")
            self.cosine_steps = max(1, int(round(rate * remaining)))
        self._build_schedulers()
        self._write_start(rate)
        logger.info(
            "cosine %d warmup %d | steps/s %s | lr %.3g localizer %.3g",
            self.cosine_steps,
            int(self.cosine_steps * self.args.warmup_ratio),
            "given" if rate is None else f"{rate:.4f}",
            float(self.sched_lora.get_last_lr()[0]),
            float(self.sched_loc.get_last_lr()[0]),
        )

    def train(self) -> None:
        if self.sched_lora is None or self.sched_loc is None:
            raise RuntimeError("train before prepare_horizon")
        assert self.model is not None and self.localizer is not None
        assert self.tlog is not None and self.embedder is not None
        self.model.train()
        self.localizer.train()
        self.opt_lora.zero_grad(set_to_none=True)
        self.opt_loc.zero_grad(set_to_none=True)
        self._next_eval = time.monotonic() + float(self.args.eval_hours) * 3600.0
        while time.monotonic() < self.deadline and not self._stop:
            family = self._pick_family()
            path: Path | None = None
            assert self.embedder is not None
            try:
                loss, path, parts = self._example_loss(family)
                loss_value = float(loss.detach())
                self._finite(loss_value, "loss")
                loss.backward()
            finally:
                self.embedder.clear_pending()
                if path is not None:
                    path.unlink(missing_ok=True)
            self._optimizer_step(family, loss_value, parts)
            if self.args.save_steps > 0 and self.step % int(self.args.save_steps) == 0:
                self.save("step")
            if time.monotonic() >= self._next_eval:
                self.evaluate()
                self.save("eval")
                self._next_eval = time.monotonic() + float(self.args.eval_hours) * 3600.0
        self._flush_window()
        if self._last_eval_step != self.step:
            self.evaluate()
        if self._signal or self._stop:
            reason = "signal"
        elif self.args.monday:
            reason = "monday"
        else:
            reason = "hours"
        self.save(reason)

    def evaluate(self) -> None:
        assert self.model is not None and self.localizer is not None
        assert self.embedder is not None and self.tlog is not None and self.vram is not None
        self.vram.set_stage("eval")
        self.model.set_adapter("default")
        self.localizer.eval()
        self._use_cache(True)
        try:
            flat = run_eval(self)
        finally:
            self.embedder.clear_pending()
            self._use_cache(False)
            self.localizer.train()
        self.tlog.eval_row(self.step, 0, flat)
        self.tlog.event("eval", step=self.step, **flat)
        self._last_eval_step = self.step

    def save(self, reason: str) -> None:
        if self.model is None or self.localizer is None or self.embedder is None:
            raise RuntimeError("save before the model is loaded")
        if self.tlog is None:
            raise RuntimeError("save before the logger exists")
        root = full_weights_dir()
        step_dir = root / f"{self.label}_step_{self.step:06d}"
        last_dir = root / f"{self.label}_last"
        self._write_save(step_dir, reason)
        self._write_save(last_dir, reason)
        self.tlog.event(
            "save", step=self.step, reason=reason, path=str(step_dir), last=str(last_dir),
        )

    def _require_start(self) -> None:
        if self.args.label == self.args.start:
            raise RuntimeError(
                f"label {self.args.label!r} would overwrite the start checkpoint"
            )
        root = self.start_dir
        required = (
            root / "gemma" / "adapter_model.safetensors",
            root / "localizer.pt",
            root / "coord_embed.pt",
            root / "s1.pt",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "start checkpoint is missing " + ", ".join(missing)
            )

    def _load_anchors(self, cfg: TrainConfig) -> None:
        assert self.processor is not None
        from PIL import Image

        with tempfile.TemporaryDirectory(prefix="weekend_fence_") as tmp:
            image = Path(tmp) / "measure.png"
            Image.new("RGB", (32, 32), (18, 18, 18)).save(image)
            soft = measure_image_soft_tokens(self.processor, str(image))
        by_source = materialize(
            sources_from_manifest(), cfg,
            processor=self.processor, image_soft_tokens=soft,
        )
        kept: list[TrainingExample] = []
        holdout: list[TrainingExample] = []
        for exs in by_source.values():
            self.rng.shuffle(exs)
            n_hold = _holdout_count(len(exs))
            if n_hold >= len(exs):
                raise RuntimeError(
                    f"holdout consumed every example of a source ({len(exs)})"
                )
            holdout.extend(exs[:n_hold])
            kept.extend(exs[n_hold:])
        if not kept:
            raise RuntimeError("anchor sources produced no training examples")
        if not holdout:
            raise RuntimeError("anchor sources produced no holdout examples")
        self.anchors = kept
        self.holdout = holdout

    def _build_optimizers(self) -> None:
        import bitsandbytes as bnb

        assert self.model is not None and self.localizer is not None and self.embedder is not None
        lora = [param for param in self.model.parameters() if param.requires_grad]
        loc = [param for param in self.localizer.parameters() if param.requires_grad]
        if not lora:
            raise RuntimeError("no trainable LoRA parameters")
        if not loc:
            raise RuntimeError("no trainable localizer parameters")
        lora_ids = {id(param) for param in lora}
        if any(id(param) in lora_ids for param in loc):
            raise RuntimeError("localizer parameters are inside the LoRA optimizer")
        if {id(param) for param in self.embedder.parameters()} & lora_ids:
            raise RuntimeError("coord embedder is trainable")
        self.opt_lora = bnb.optim.PagedAdamW8bit(
            lora, lr=self.args.lr, weight_decay=0.0,
        )
        self.opt_loc = torch.optim.AdamW(
            loc, lr=self.args.localizer_lr, weight_decay=0.01,
        )
        self.optimizer = self.opt_lora
        self._lora_params = lora
        self._loc_params = loc

    def _build_schedulers(self) -> None:
        warmup = int(self.cosine_steps * float(self.args.warmup_ratio))
        total = int(self.cosine_steps)
        floor = float(self.args.lr_floor)

        def factor(
            step: int, _warmup: int = warmup, _total: int = total, _floor: float = floor,
        ) -> float:
            return _factor(step, _warmup, _total, _floor)

        self.sched_lora = torch.optim.lr_scheduler.LambdaLR(self.opt_lora, factor)
        self.sched_loc = torch.optim.lr_scheduler.LambdaLR(self.opt_loc, factor)
        self.scheduler = self.sched_lora

    def _pick_family(self) -> str:
        roll = self.rng.random()
        acc = 0.0
        chosen: str | None = None
        for key, weight in self.mix.items():
            chosen = key
            acc += weight
            if roll < acc:
                return key
        if chosen is None:
            raise RuntimeError("mixture is empty")
        return chosen

    def _example_loss(
        self, family: str,
    ) -> tuple[torch.Tensor, Path | None, dict[str, tuple[float, int]]]:
        assert self.embedder is not None and self.vram is not None and self.tmp is not None
        self.embedder.clear_pending()
        if family == "F":
            loss, parts = self._anchor_loss()
            self._discard_tap()
            value = float(loss.detach())
            merged = dict(parts)
            merged["F"] = (value, 1)
            merged["lm"] = (value, 1)
            return loss, None, merged
        if family not in STAGE_NAME:
            raise RuntimeError(f"unknown family {family!r}")
        batch = build_train_family(
            family, self.rng, self.tmp, self.step, self._place_image,
        )
        self.vram.set_stage(STAGE_NAME[family])
        try:
            self._use_cache(True)
            try:
                loss, parts = self._joint(batch, family)
            finally:
                self._use_cache(False)
        except Exception:
            batch.path.unlink(missing_ok=True)
            raise
        return loss, batch.path, parts

    def _joint(
        self, batch: Any, family: str,
    ) -> tuple[torch.Tensor, dict[str, tuple[float, int]]]:
        assert (
            self.model is not None and self.embedder is not None
            and self.localizer is not None and self.tap is not None
            and self.collator is not None and self.vram is not None
        )
        self._discard_tap()
        builds, metas = self._collate(batch)
        ids_rows = [build["model_inputs"]["input_ids"][0] for build in builds]
        lcp = _lcp_len(ids_rows)
        # Identical prompts (one user line, several replies) share every
        # token through the reply. The token that predicts the first reply
        # token has to sit in the suffix, so the cache stops one before it.
        earliest = min(self._first_reply(build) for build in builds)
        if earliest < 1:
            raise RuntimeError("a reply starts at the first token")
        lcp = min(lcp, earliest - 1)
        if lcp <= 0:
            raise RuntimeError("common prefix is empty after leaving room for the reply")
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
        for build, meta in zip(builds, metas):
            n_weight = int((build["weights"][0] != 0).sum())
            if n_weight != int(meta["n_tokens"]):
                raise RuntimeError(
                    f"reply tokens {n_weight} != supervision span {meta['n_tokens']}"
                )
            if int(meta["n_mid"]) > n_weight:
                raise RuntimeError(
                    f"mid span {meta['n_mid']} longer than the reply ({n_weight})"
                )
        pref = self._prefix_batch(builds[0]["model_inputs"], lcp)
        positions = grid_positions(pref)
        self._heartbeat(STAGE_NAME[family], "prefill", 1, lcp)
        with torch.no_grad():
            out = self.model(**pref, use_cache=True, logits_to_keep=1)
        if out.past_key_values is None:
            raise RuntimeError("prefix forward returned no past_key_values")
        cache = out.past_key_values
        hidden = self.tap.take()
        if int(hidden.shape[1]) != lcp:
            raise RuntimeError(
                f"prefill tap length {int(hidden.shape[1])} != prefix {lcp}"
            )
        self._grid = gather_grid(hidden, positions).detach()
        _detach_cache(cache)
        seen = _tokens_seen(cache)
        if seen != lcp:
            raise RuntimeError(f"prefix cache saw {seen} tokens, common prefix is {lcp}")
        self.model.train()
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        self.model.config.use_cache = True
        packed, positions, coords, predictors, labels = self._suffix_pack(builds, metas, lcp)
        for index, meta in enumerate(metas):
            n_mid = int(meta["n_mid"])
            if n_mid:
                coords[index, :n_mid, 2:4] = coords[index, :n_mid, 0:2]
        packed["past_key_values"] = expand_train_kv(cache, len(builds))
        width = int(packed["input_ids"].shape[1])
        self._heartbeat(STAGE_NAME[family], "suffix", len(builds), lcp + width)
        self.embedder.set_pending(coords, positions)
        out = self.model(
            **packed, use_cache=True, output_hidden_states=True, logits_to_keep=1,
        )
        if out.past_key_values is None:
            raise RuntimeError(
                "suffix forward returned no past_key_values; the prefix cache was dropped"
            )
        hidden_states = getattr(out, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("suffix forward returned no hidden states")
        lm_hidden = hidden_states[-1]
        if int(lm_hidden.shape[1]) != width:
            raise RuntimeError(
                f"suffix hidden length {int(lm_hidden.shape[1])} != {width}"
            )
        if not predictors:
            raise RuntimeError("image batch has no reply tokens")
        gathered = torch.stack(
            [lm_hidden[row, index] for row, index in predictors], dim=0,
        )
        gathered = gathered.to(dtype=_lm_head_dtype(self.model))
        logits = apply_lm_head_chunked(self.model, gathered, 1024)
        target = torch.tensor(labels, dtype=torch.long, device=gathered.device)
        token_loss = torch.nn.functional.cross_entropy(
            logits.float(), target, reduction="none",
        )
        lm = token_loss.mean()
        queries = self._reply_queries(self.tap.take(), positions, metas)
        label_rows, mode_rows = self._label_rows(metas)
        if int(queries.shape[0]) != len(label_rows):
            raise RuntimeError(
                f"localizer queries {int(queries.shape[0])} != labels {len(label_rows)}"
            )
        loc, v_rms = self._readout_loss(
            queries,
            torch.tensor(label_rows, dtype=torch.float32, device=self.device),
            torch.tensor(mode_rows, dtype=torch.long, device=self.device),
        )
        self._finite(float(loc.detach()), "localizer_loss")
        if v_rms is not None:
            self._finite(v_rms, "v_rms")
        weight = float(self.args.localizer_weight)
        if weight != 0.0:
            total = lm + weight * loc
        else:
            total = lm
        parts: dict[str, tuple[float, int]] = {
            family: (float(total.detach()), 1),
            "lm": (float(lm.detach()), 1),
            "localizer": (float(loc.detach()), 1),
        }
        if v_rms is not None:
            parts["v_rms"] = (v_rms, 1)
        return total, parts

    def _collate(self, batch: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        assert self.collator is not None
        tokenizer = self.collator.tokenizer
        builds = []
        metas = []
        if not batch.turns:
            raise RuntimeError(f"family {batch.family} produced no replies")
        for turn in batch.turns:
            example = TrainingExample(
                messages=messages_for(batch, turn),
                target_text=turn.reply,
                loss="ce",
                source="s2_weekend",
            )
            n_tokens, n_mid = supervision_span(tokenizer, turn.reply, turn.all_mid)
            builds.append(self.collator.build(example))
            metas.append({
                "points": (turn.s, turn.v),
                "agent": turn.agent,
                "n_tokens": n_tokens,
                "n_mid": n_mid,
            })
        return builds, metas

    def _first_reply(self, build: dict[str, Any]) -> int:
        index = (build["weights"][0] != 0).nonzero(as_tuple=False).flatten()
        if int(index.numel()) == 0:
            raise RuntimeError("a reply has no weighted tokens")
        return int(index[0].item())

    def _prefix_batch(self, model_inputs: dict[str, Any], lcp: int) -> dict[str, Any]:
        seq = int(model_inputs["input_ids"].shape[1])
        pref: dict[str, Any] = {
            "input_ids": model_inputs["input_ids"][:, :lcp],
            "attention_mask": torch.ones(1, lcp, dtype=torch.long, device=self.device),
            "position_ids": torch.arange(lcp, device=self.device).unsqueeze(0),
        }
        for key, val in model_inputs.items():
            if key in ("input_ids", "attention_mask") or not torch.is_tensor(val):
                continue
            if _token_aligned(key, val, seq):
                pref[key] = val[:, :lcp]
            else:
                pref[key] = val
        return pref

    def _reply_queries(
        self, hidden: torch.Tensor, positions: torch.Tensor, metas: list[dict[str, Any]],
    ) -> torch.Tensor:
        if hidden.dim() != 3:
            raise RuntimeError(
                f"tap hidden rank {hidden.dim()} shape {tuple(hidden.shape)}"
            )
        if int(hidden.shape[0]) != len(metas):
            raise RuntimeError(
                f"tap batch {int(hidden.shape[0])} != replies {len(metas)}"
            )
        chunks = []
        for index, meta in enumerate(metas):
            n = int(meta["n_tokens"])
            chunks.append(hidden[index].index_select(0, positions[index, :n]))
        return torch.cat(chunks, dim=0).detach()

    def _label_rows(
        self, metas: list[dict[str, Any]],
    ) -> tuple[list[list[float]], list[int]]:
        """Clean ``(s, v)``. Mid tokens use ``v = s`` for the L2 target."""
        rows: list[list[float]] = []
        modes: list[int] = []
        for meta in metas:
            sx, sy = meta["points"][0]
            vx, vy = meta["points"][1]
            n_mid = int(meta["n_mid"])
            for token in range(int(meta["n_tokens"])):
                if token < n_mid:
                    rows.append([float(sx), float(sy), float(sx), float(sy)])
                    modes.append(MODE_MID)
                else:
                    rows.append([float(sx), float(sy), float(vx), float(vy)])
                    modes.append(MODE_FULL)
        if not rows:
            raise RuntimeError("localizer labels are empty")
        return rows, modes

    def _readout_loss(
        self, queries: torch.Tensor, labels: torch.Tensor, modes: torch.Tensor,
    ) -> tuple[torch.Tensor, float | None]:
        assert self.localizer is not None
        grid = self._grid
        if grid is None or grid.dim() != 3 or int(grid.shape[0]) != 1:
            raise RuntimeError("joint step has no single-image grid")
        n = int(queries.shape[0])
        if n != int(labels.shape[0]) or n != int(modes.shape[0]):
            raise RuntimeError("localizer rows do not line up")
        pred, cell_logits = self.localizer(
            grid.detach().expand(n, -1, -1), queries.unsqueeze(1),
        )
        pred = pred[:, 0, :]
        logits = cell_logits[:, 0]
        err = (pred - labels).pow(2)
        full = modes == MODE_FULL
        mid = modes == MODE_MID
        if int((full | mid).sum()) != n:
            raise RuntimeError("a reply token is neither full nor mid")
        terms: list[torch.Tensor] = []
        weights: list[int] = []
        if bool(full.any()):
            terms.append(err[full].mean())
            weights.append(int(full.sum()))
        if bool(mid.any()):
            # v's label is s on these tokens, so both heads are supervised.
            terms.append(err[mid].mean())
            weights.append(int(mid.sum()))
        if not terms:
            raise RuntimeError("localizer loss has no tokens")
        coord = terms[0] * float(weights[0])
        for term, count in zip(terms[1:], weights[1:]):
            coord = coord + term * float(count)
        coord = coord / float(sum(weights))
        ces: list[torch.Tensor] = []
        for head, sl in enumerate((slice(0, 2), slice(2, 4))):
            target = cell_index(labels[:, sl]).clone()
            if head > 0:
                target[mid] = -1
            if bool((target != -1).any()):
                ces.append(torch.nn.functional.cross_entropy(
                    logits[:, head, :], target, ignore_index=-1,
                ))
            else:
                ces.append(logits.new_zeros(()))
        cell = torch.stack(ces).mean()
        loss = coord + float(self.args.cell_ce_weight) * cell
        v_rms = None
        if bool(full.any()):
            dist2 = err.detach()[full][:, 2:4].sum(dim=-1)
            v_rms = float(dist2.mean().sqrt())
        return loss, v_rms

    def _optimizer_step(
        self, family: str, loss_value: float, parts: dict[str, tuple[float, int]],
    ) -> None:
        assert self.opt_lora is not None and self.opt_loc is not None
        assert self.sched_lora is not None and self.sched_loc is not None
        saw = "localizer" in parts and float(self.args.localizer_weight) != 0.0
        gn_lora = float(torch.nn.utils.clip_grad_norm_(
            self._lora_params, float(self.args.max_grad_norm),
        ))
        self._finite(gn_lora, "grad_norm")
        gn_loc = 0.0
        if saw:
            gn_loc = float(torch.nn.utils.clip_grad_norm_(
                self._loc_params, float(self.args.localizer_clip),
            ))
            self._finite(gn_loc, "localizer_grad_norm")
        self.opt_lora.step()
        if saw:
            self.opt_loc.step()
        self.sched_lora.step()
        self.sched_loc.step()
        self.opt_lora.zero_grad(set_to_none=True)
        self.opt_loc.zero_grad(set_to_none=True)
        self.step += 1
        self._log_step(family, loss_value, gn_lora, gn_loc, parts)

    def _log_step(
        self,
        family: str,
        loss_value: float,
        gn_lora: float,
        gn_loc: float,
        parts: dict[str, tuple[float, int]],
    ) -> None:
        assert self.tlog is not None and self.sched_lora is not None and self.sched_loc is not None
        for key, (total, n) in parts.items():
            if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
                raise RuntimeError(f"task {key} has count {n!r}")
            if not math.isfinite(total):
                raise RuntimeError(f"non-finite task {key}: {total}")
            self._win_sum[key] = self._win_sum.get(key, 0.0) + float(total) * n
            self._win_n[key] = self._win_n.get(key, 0) + n
        lr = float(self.sched_lora.get_last_lr()[0])
        loc_lr = float(self.sched_loc.get_last_lr()[0])
        record = {
            "step": self.step,
            "loss": loss_value,
            "lr": lr,
            "localizer_lr": loc_lr,
            "grad_norm": gn_lora,
            "grad_norm_localizer": gn_loc,
            "family": family,
            "tasks": {key: round(total, 4) for key, (total, _n) in parts.items()},
            "task_n": {key: n for key, (_total, n) in parts.items()},
        }
        self.tlog.last_step(record)
        if self.step % int(self.args.log_steps) == 0:
            means = {
                key: self._win_sum[key] / self._win_n[key] for key in self._win_sum
            }
            window = {
                **record,
                "tasks": {key: round(value, 4) for key, value in means.items()},
                "task_n": dict(self._win_n),
                "task_scope": "window",
            }
            self.tlog.step(window)
            logger.info(
                "step %d %s loss %.4f lr %.3g loc %.3g grad %.3f",
                self.step, family, loss_value, lr, loc_lr, gn_lora,
            )
            self._win_sum.clear()
            self._win_n.clear()

    def _flush_window(self) -> None:
        if not self._win_sum or self.tlog is None:
            return
        means = {key: self._win_sum[key] / self._win_n[key] for key in self._win_sum}
        self.tlog.step({
            "step": self.step,
            "tasks": {key: round(value, 4) for key, value in means.items()},
            "task_n": dict(self._win_n),
            "task_scope": "window",
        })
        self._win_sum.clear()
        self._win_n.clear()

    def _bench_rate(self) -> float:
        families = [key for key in FAMILY_ORDER if key in self.mix]
        for family in families:
            self._bench_once(family)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        timed = 0
        for _ in range(2):
            for family in families:
                self._bench_once(family)
                timed += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if elapsed <= 0.0 or timed <= 0:
            raise RuntimeError("bench timer did not advance")
        return timed / elapsed

    def _bench_once(self, family: str) -> None:
        assert self.embedder is not None and self.opt_lora is not None and self.opt_loc is not None
        path: Path | None = None
        try:
            loss, path, _parts = self._example_loss(family)
            self._finite(float(loss.detach()), "loss")
            loss.backward()
        finally:
            self.embedder.clear_pending()
            if path is not None:
                path.unlink(missing_ok=True)
            self.opt_lora.zero_grad(set_to_none=True)
            self.opt_loc.zero_grad(set_to_none=True)

    def _discard_tap(self) -> None:
        if self.tap is not None and self.tap._captured is not None:
            self.tap.take()

    def _write_start(self, rate: float | None) -> None:
        assert self.tlog is not None
        payload = {
            **vars(self.args),
            "mixture": self.mix,
            "cosine_steps": self.cosine_steps,
            "steps_per_s": rate,
            "start_dir": str(self.start_dir),
            "hours": self._hours,
            "trainer": "s2_weekend",
        }
        self.tlog.write_config(payload)
        self.tlog.event(
            "start",
            step=self.step,
            cosine_steps=self.cosine_steps,
            steps_per_s=rate,
            mixture=self.mix,
            start=str(self.start_dir),
            hours=self._hours,
        )

    def _write_save(self, dest: Path, reason: str) -> None:
        assert self.model is not None and self.localizer is not None and self.embedder is not None
        if dest.resolve() == self.start_dir.resolve():
            raise RuntimeError(f"refusing to overwrite the start checkpoint {dest}")
        dest.mkdir(parents=True, exist_ok=True)
        gemma = dest / "gemma"
        gemma.mkdir(parents=True, exist_ok=True)
        self._save_gemma(gemma)
        write_readout(dest, self.localizer, self.embedder, KEPT_LAYER)
        shutil.copy2(self.start_dir / "s1.pt", dest / "s1.pt")
        signal_name = None
        if reason == "signal" and self._signal:
            try:
                signal_name = signal.Signals(int(self._signal)).name
            except ValueError:
                signal_name = f"SIG{self._signal}"
        _write_json_atomic(dest / "train_meta.json", {
            "trainer": "s2_weekend",
            "start": self.args.start,
            "step": self.step,
            "mixture": self.mix,
            "cosine_steps": self.cosine_steps,
            "reason": reason,
            "signal": signal_name,
            "kept_layer": KEPT_LAYER,
            "model_key": self.spec.key,
            "format": "peft",
            "lr": self.args.lr,
            "localizer_lr": self.args.localizer_lr,
        })
        logger.info("saved %s (%s) step %d", dest, reason, self.step)

    def _save_gemma(self, path: Path) -> None:
        from peft import PeftModel

        assert self.model is not None
        if not isinstance(self.model, PeftModel):
            raise RuntimeError("weekend trainer expected a PEFT Gemma")
        self.model.set_adapter("default")
        kwargs: dict[str, Any] = {}
        if "selected_adapters" in inspect.signature(self.model.save_pretrained).parameters:
            kwargs["selected_adapters"] = ["default"]
        self.model.save_pretrained(path, **kwargs)
        write_gemma_meta(path, self.spec, "peft")


def _supervise(argv: list[str]) -> int:
    label = _argv_value(argv, "--label") or "joint_v1"
    child = list(argv)
    if "--no-supervise" not in child:
        child.append("--no-supervise")
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "neural_net.s2_weekend_train", *child],
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="localizer_finetuned_v1")
    parser.add_argument("--label", default="joint_v1")
    parser.add_argument(
        "--monday", action="store_true",
        help="Stop at Monday 08:00 America/New_York. Mutually exclusive with --hours.",
    )
    parser.add_argument("--hours", type=float, default=0.0)
    parser.add_argument(
        "--cosine-steps", type=int, default=0,
        help="Optimizer-step horizon. 0 measures it from the startup bench.",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--localizer-lr", type=float, default=1e-4)
    parser.add_argument("--localizer-weight", type=float, default=1.0)
    parser.add_argument("--cell-ce-weight", type=float, default=0.5)
    parser.add_argument("--localizer-clip", type=float, default=1.0)
    parser.add_argument("--lr-floor", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--eval-hours", type=float, default=12.0)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--mix", default=None,
        help="A=0.30,B=0.10,C=0.20,D=0.15,E=0.15,F=0.10. "
             "A zero weight drops that family and the rest are renormalized.",
    )
    parser.add_argument("--no-supervise", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not args.no_supervise:
        raise SystemExit(_supervise(sys.argv[1:]))
    configure_logging()
    if args.grad_accum != 1:
        raise SystemExit("--grad-accum must be 1")
    if args.monday == (args.hours > 0.0):
        raise SystemExit("pass exactly one of --monday and --hours")
    if args.lr <= 0.0 or args.localizer_lr <= 0.0:
        raise SystemExit("learning rates must be positive")
    if args.save_steps < 1:
        raise SystemExit("--save-steps must be positive")
    if args.eval_hours <= 0.0:
        raise SystemExit("--eval-hours must be positive")
    if args.cosine_steps < 0:
        raise SystemExit("--cosine-steps must be >= 0")
    if args.warmup_ratio < 0.0 or args.lr_floor < 0.0 or args.lr_floor > 1.0:
        raise SystemExit("--warmup-ratio must be >= 0 and --lr-floor must be in [0, 1]")
    if args.monday:
        monday = _hours_until_monday()
        if monday is None:
            raise SystemExit("Monday 08:00 America/New_York has already passed")
        _target, hours = monday
        if hours <= 0.0:
            raise SystemExit("Monday 08:00 America/New_York has already passed")
    else:
        hours = float(args.hours)
    trainer = WeekendTrainer(args)
    trainer.deadline = time.monotonic() + hours * 3600.0
    trainer._hours = float(hours)
    signal.signal(signal.SIGTERM, trainer._on_signal)
    signal.signal(signal.SIGINT, trainer._on_signal)
    try:
        trainer.load()
        trainer.evaluate()
        if trainer._stop:
            trainer.save("signal")
            return
        if time.monotonic() >= trainer.deadline:
            trainer.save("monday" if args.monday else "hours")
            return
        trainer.prepare_horizon()
        if trainer._stop:
            trainer.save("signal")
            return
        trainer.save("eval")
        trainer.train()
    except BaseException as exc:
        if trainer.tlog is not None and not isinstance(exc, KeyboardInterrupt):
            oom = _is_cuda_oom(exc)
            trainer.tlog.record_crash(
                exc, step=trainer.step, **({"tag": "cuda_oom"} if oom else {}),
            )
            if oom:
                trainer.tlog.event("cuda_oom", step=trainer.step, error=str(exc)[:500])
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
        try:
            trainer._flush_window()
        except Exception:
            logger.error("task-window flush failed\n%s", traceback.format_exc())
        trainer.close()


if __name__ == "__main__":
    main()
