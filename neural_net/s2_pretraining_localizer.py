"""Learn to look: a frozen Gemma trunk and a trained Localizer.

The trunk is the aug27 adapter (``--gemma-base``). Nothing in it
receives gradient: no LoRA step, no replay, no KD, no token CE. Each
image is encoded once (stage A, through the shared question stem) and
its kept-layer cache is reused for that image's K questions (stage B).
The Localizer trains on a GPU ring of those grids and reply states.

``CoordEmbedder`` is built and installed with its gate at 0 and
``enabled`` False. It is not in the optimizer. It trains only in the
later learn-to-tell-where-you-are-looking phase, after a Localizer
hits the floors.

``--bench N`` times N rounds of the same path and exits without a
snapshot and without ``labels.jsonl``.

A three-layer mix over layers 16, 24, and 32 is not built. Change
``--kept-layer`` to try one layer. The shelved rule is: keep the
shallowest layer that reaches the text-only floor.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import signal
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from agent.config import CONFIG
from agent.modes import (
    SYSTEM_PROMPT_S2,
    _S2_LOOK_LINES,
    _S2_MOVE_LINES,
    _build_game_messages,
)
from agent.model import ADAPTERS, repeat_kv_cache, spec_for
from neural_net.coord_embed import CoordEmbedder
from neural_net.localizer import (
    COORD_ORDER,
    GRID_CELLS,
    KEPT_LAYER,
    LayerTap,
    Localizer,
    cell_index,
    gather_grid,
    grid_positions,
    truncated_forward,
)
from neural_net.paths import weights_root
from neural_net.s2 import _hidden_size
from neural_net.s2_pretraining_learn_to_look import (
    DATA_GAME,
    BoardPump,
    _append_jsonl,
    _open_png,
    _save_png,
    _system_message,
    _write_json_atomic,
    board_context,
    draw_questions,
)
from training.image_noise import TRAINING_STRENGTH, noise_image
from training.run_weekend import VramMonitor
from training.token_fence import (
    count_prefix_tokens,
    measure_image_soft_tokens,
    system_prefix_hash,
)
from training.train import (
    Collator,
    PrefixKVRuntime,
    TrainConfig,
    TrainingExample,
    TrainLogger,
    _ensure_prefix_cache,
    _gpu_mem_snapshot,
    _suffix_inputs_with_cache,
    build_model,
    configure_logging,
    load_adapter_state,
    resolve_terminator_id,
)

logger = logging.getLogger("s2_localizer")

TRAINER_NAME = "s2_pretraining_localizer"
# Placeholder until the first --bench. Steps per second of the Localizer
# optimizer, not of the trunk. Replace with the bench's "localizer s/step"
# inverted (1 / seconds) before trusting --cosine-steps 0.
LOCALIZER_STEPS_PER_S = 100.0
# RMS of the whole tensor. A wrong mask or a shifted grid is ~1.
# The first bench's grid *max* was 0.053 against a query RMS well under
# this, which is one bf16 element, not a bad gather.
_EQUIV_TOL = 0.05
# Worst single element. bf16 across a 256x3840 grid trips 0.05; a real
# disagreement is far above this.
_EQUIV_MAX_TOL = 0.25
_KIND_MOVE = {"look": 0, "move": 1}
_KIND_CAND = {"gold": 0, "exit": 1, "corner": 2}


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """``(rms_rel, max_rel)`` of ``got`` against ``ref``, both fp32."""
    diff = (got.float() - ref.float()).abs()
    ref_abs = ref.float().abs()
    rms = float(diff.square().mean().sqrt() / (ref_abs.square().mean().sqrt() + 1e-6))
    peak = float(diff.max() / (ref_abs.max() + 1e-6))
    return rms, peak


def _factor(step: int, warmup: int, total: int, floor: float) -> float:
    """Warmup, then cosine down to ``floor``, then stay there.

    At and after ``total`` the multiplier is the floor. An unclamped
    cosine walks back up toward the peak.
    """
    if step >= total:
        return floor
    if warmup > 0 and step < warmup:
        return step / max(1, warmup)
    span = max(1, total - warmup)
    progress = (step - warmup) / span
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _reply_cap(tokenizer: Any, reply_positions: int) -> int:
    lines = list(_S2_LOOK_LINES) + list(_S2_MOVE_LINES)
    longest = 0
    for line in lines:
        n = len(tokenizer(line, add_special_tokens=False)["input_ids"])
        longest = max(longest, n)
    cap = longest + 1
    if reply_positions > 0:
        cap = min(cap, reply_positions)
    if cap < 1:
        raise RuntimeError("reply-position cap is 0")
    return cap


def _lcp_len(rows: list[torch.Tensor]) -> int:
    seqs = [row.view(-1) for row in rows]
    n = min(int(seq.numel()) for seq in seqs)
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
        and val.shape[1] == seq_len
        and not val.dtype.is_floating_point
    )


class _Floors:
    """Label-only baselines. The constant floor is every label ever
    pushed. The text-only floor is scored on the current minibatch."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = torch.zeros(6, dtype=torch.float64)
        self.m2 = torch.zeros(6, dtype=torch.float64)
        self.s_sum = torch.zeros(2, dtype=torch.float64)
        self.s_count = 0
        self.phrase_sum: dict[int, torch.Tensor] = {}
        self.phrase_count: dict[int, int] = {}

    def update(self, labels: torch.Tensor, phrase_ids: list[int]) -> None:
        """``labels [n, 6]`` on CPU."""
        for row, phrase in zip(labels, phrase_ids):
            x = row.detach().to(dtype=torch.float64, device="cpu")
            self.count += 1
            delta = x - self.mean
            self.mean = self.mean + delta / self.count
            self.m2 = self.m2 + delta * (x - self.mean)
            self.s_sum = self.s_sum + x[0:2]
            self.s_count += 1
            acc = self.phrase_sum.get(phrase)
            if acc is None:
                self.phrase_sum[phrase] = x[2:4].clone()
                self.phrase_count[phrase] = 1
            else:
                self.phrase_sum[phrase] = acc + x[2:4]
                self.phrase_count[phrase] = self.phrase_count[phrase] + 1

    def constant(self) -> dict[str, float]:
        if self.count < 2:
            return {name: float("nan") for name in COORD_ORDER}
        var = self.m2 / self.count
        return {
            "s": math.sqrt(float(var[0:2].mean())),
            "v": math.sqrt(float(var[2:4].mean())),
            "v_bar": math.sqrt(float(var[4:6].mean())),
        }

    def text_only(
        self,
        labels: torch.Tensor,
        phrase_ids: list[int],
        move_kinds: list[int],
    ) -> dict[str, float]:
        if self.s_count < 1:
            return {name: float("nan") for name in COORD_ORDER}
        s_hat = (self.s_sum / self.s_count).to(dtype=torch.float32)
        pred = torch.empty_like(labels)
        for i, (phrase, kind) in enumerate(zip(phrase_ids, move_kinds)):
            count = self.phrase_count.get(phrase, 0)
            if count < 1:
                raise RuntimeError(f"text-only floor has no mean for phrase {phrase}")
            v_hat = (self.phrase_sum[phrase] / count).to(dtype=torch.float32)
            vbar = (2.0 * s_hat - v_hat) if kind == 0 else v_hat
            pred[i, 0:2] = s_hat
            pred[i, 2:4] = v_hat
            pred[i, 4:6] = vbar
        err = (labels - pred).pow(2)
        return {
            "s": math.sqrt(float(err[:, 0:2].mean())),
            "v": math.sqrt(float(err[:, 2:4].mean())),
            "v_bar": math.sqrt(float(err[:, 4:6].mean())),
        }


class _Buffer:
    def __init__(
        self,
        n: int,
        k: int,
        r_max: int,
        hidden: int,
        device: torch.device,
    ) -> None:
        self.n = n
        self.k = k
        self.r_max = r_max
        self.device = device
        self.grids = torch.empty(n, GRID_CELLS, hidden, dtype=torch.bfloat16, device=device)
        self.queries = torch.empty(
            n, k, r_max, hidden, dtype=torch.bfloat16, device=device,
        )
        self.valid_r = torch.zeros(n, k, r_max, dtype=torch.bool, device=device)
        self.labels = torch.empty(n, k, 6, dtype=torch.float32, device=device)
        self.move_kind = torch.zeros(n, k, dtype=torch.int8, device=device)
        self.cand_kind = torch.zeros(n, k, dtype=torch.int8, device=device)
        self.phrase_id = torch.zeros(n, k, dtype=torch.int32, device=device)
        self.dup = torch.zeros(n, k, dtype=torch.bool, device=device)
        self.filled = 0
        self.write = 0

    def add(self, row: dict[str, Any]) -> int:
        slot = self.write % self.n
        self.grids[slot] = row["grid"]
        self.queries[slot] = row["queries"]
        self.valid_r[slot] = row["valid_r"]
        self.labels[slot] = row["labels"]
        self.move_kind[slot] = row["move_kind"]
        self.cand_kind[slot] = row["cand_kind"]
        self.phrase_id[slot] = row["phrase_id"]
        self.dup[slot] = row["dup"]
        self.write += 1
        self.filled = min(self.n, self.filled + 1)
        return slot


class Trainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.label = args.label
        self.rng = __import__("random").Random(args.seed)
        self.data_dir = DATA_GAME / self.label
        self.images_dir = self.data_dir / "images"
        self.labels_path = self.data_dir / "labels.jsonl"
        self.state_path = DATA_GAME / f"{self.label}_state.json"
        self.s2_root = weights_root() / "s2"
        self.step = 0
        self.images_seen = 0
        self.labels_seen = 0
        self.phrase_ids: dict[str, int] = {}
        self.floors = _Floors()
        self._stop = False
        self._floor_warned = False
        self._checked = False
        self.model = None
        self.localizer: Localizer | None = None
        self.coord_embed: CoordEmbedder | None = None
        self.tap: LayerTap | None = None
        self.tlog: TrainLogger | None = None
        self.vram: VramMonitor | None = None
        self.buffer: _Buffer | None = None
        self.optimizer = None
        self.scheduler = None
        self.cosine_steps = 0
        self.prefix_kv = PrefixKVRuntime()
        self.prefix_n = 0
        self.prefix_hash = ""
        self.collator: Collator | None = None
        self.tokenizer = None
        self.pump: BoardPump | None = None
        self.tmp: Path | None = None
        self.device = torch.device(args.device)
        self.hidden = 0
        self.r_max = 0

    def load(self) -> float:
        t0 = time.perf_counter()
        cfg = TrainConfig(
            label=self.label,
            seed=self.args.seed,
            device=self.args.device,
            architecture="gemma-4-12b",
        )
        self.tlog = TrainLogger(self.label)
        log_path = self.tlog.run_dir / "train.log"
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        ))
        logging.getLogger().addHandler(handler)
        self.vram = VramMonitor(self.label)
        spec = spec_for("gemma-4-12b")
        base = weights_root() / "gemma-4-12b" / self.args.gemma_base
        if not (base / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"--gemma-base {self.args.gemma_base!r} has no "
                f"adapter_config.json under {base}"
            )
        self.model, processor, _targets, _projector = build_model(
            spec, cfg, CONFIG.hf_token,
        )
        load_adapter_state(self.model, base)
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()
        self.model.gradient_checkpointing_disable()
        self.model.config.use_cache = True
        self.hidden = _hidden_size(self.model)
        self.localizer = Localizer(self.hidden).to(
            device=self.device, dtype=torch.float32,
        )
        self.coord_embed = CoordEmbedder(self.hidden).to(
            device=self.device, dtype=torch.float32,
        )
        self.coord_embed.install(self.model)
        self.tap = LayerTap(self.model, self.args.kept_layer)
        self.tokenizer = getattr(processor, "tokenizer", processor)
        terminator = resolve_terminator_id(self.model, self.tokenizer)
        self.collator = Collator(
            processor, ADAPTERS[spec.family], terminator,
            compute_dtype=torch.bfloat16, device=self.device,
        )
        with tempfile.TemporaryDirectory(prefix="token_fence_") as tmp:
            from PIL import Image

            img = Path(tmp) / "measure.png"
            Image.new("RGB", (32, 32), (18, 18, 18)).save(img)
            soft = measure_image_soft_tokens(processor, str(img))
        system = [_system_message(SYSTEM_PROMPT_S2)]
        self.prefix_n = count_prefix_tokens(system, self.tokenizer, processor)
        self.prefix_hash = system_prefix_hash(system)
        self.r_max = _reply_cap(self.tokenizer, self.args.reply_positions)
        self._resume()
        self.buffer = _Buffer(
            self.args.buffer_images, self.args.questions_per_image,
            self.r_max, self.hidden, self.device,
        )
        n_params = sum(p.numel() for p in self.localizer.parameters())
        self.cosine_steps = self.args.cosine_steps
        if self.cosine_steps <= 0:
            self.cosine_steps = max(
                1, int(self.args.hours * 3600.0 * LOCALIZER_STEPS_PER_S),
            )
        warmup = int(self.cosine_steps * self.args.warmup_ratio)
        self.optimizer = torch.optim.AdamW(
            self.localizer.parameters(), lr=self.args.lr, weight_decay=0.01,
        )
        floor = self.args.lr_floor
        total = self.cosine_steps

        def factor(step: int, _warmup: int = warmup, _total: int = total,
                   _floor: float = floor) -> float:
            return _factor(step, _warmup, _total, _floor)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, factor)
        for _ in range(self.step):
            self.scheduler.step()
        self.tlog.write_config({
            **vars(self.args),
            "kept_layer": self.args.kept_layer,
            "localizer_params": n_params,
            "prefix_tokens": self.prefix_n,
            "image_soft_tokens": soft,
            "gemma_base": self.args.gemma_base,
            "cosine_steps": self.cosine_steps,
            "r_max": self.r_max,
            "trainer": TRAINER_NAME,
        })
        self.tlog.event(
            "gemma_base", gemma_base=self.args.gemma_base,
            localizer_params=n_params, prefix_tokens=self.prefix_n,
        )
        logger.info(
            "localizer params=%d prefix_tokens=%d image_soft_tokens=%d r_max=%d",
            n_params, self.prefix_n, soft, self.r_max,
        )
        self.pump = BoardPump(
            self.args.workers, CONFIG.game_size, (0, 1, 2, 3), self.rng,
        )
        self.tmp = Path(tempfile.mkdtemp(prefix=f"loc_{self.label}_"))
        return time.perf_counter() - t0

    def close(self) -> None:
        if self.pump is not None:
            self.pump.close()
            self.pump = None
        if self.vram is not None:
            self.vram.finish()
            self.vram = None
        if self.tmp is not None:
            shutil.rmtree(self.tmp, ignore_errors=True)
            self.tmp = None

    def _resume(self) -> None:
        name = self.args.resume_snapshot
        if not name:
            return
        path = self.s2_root / name
        meta_path = path / "train_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"--resume-snapshot missing {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("trainer") != TRAINER_NAME:
            raise RuntimeError(
                f"{path} trainer is {meta.get('trainer')!r}, "
                f"expected {TRAINER_NAME!r}"
            )
        if int(meta.get("kept_layer", -1)) != self.args.kept_layer:
            raise RuntimeError(
                f"snapshot kept_layer {meta.get('kept_layer')} != "
                f"--kept-layer {self.args.kept_layer}"
            )
        blob = torch.load(path / "localizer.pt", map_location="cpu", weights_only=True)
        if int(blob["hidden_size"]) != self.hidden:
            raise RuntimeError(
                f"snapshot hidden_size {blob['hidden_size']} != model {self.hidden}"
            )
        assert self.localizer is not None and self.coord_embed is not None
        self.localizer.load_state_dict(blob["state_dict"])
        self.coord_embed.load(path / "coord_embed.pt")
        self.coord_embed.enabled = False
        self.step = int(meta.get("step", 0))
        self.images_seen = int(meta.get("images_seen", 0))
        self.labels_seen = int(meta.get("labels_seen", 0))
        phrases = meta.get("phrase_ids") or {}
        self.phrase_ids = {str(k): int(v) for k, v in phrases.items()}
        logger.info(
            "resumed %s at step %d images %d labels %d",
            path, self.step, self.images_seen, self.labels_seen,
        )

    def _phrase(self, phrase: str) -> int:
        found = self.phrase_ids.get(phrase)
        if found is None:
            found = len(self.phrase_ids)
            self.phrase_ids[phrase] = found
        return found

    def _encode_image(self, bench: bool) -> dict[str, Any]:
        assert self.pump is not None and self.collator is not None and self.tmp is not None
        raw = self.pump.take()
        settings = raw["settings"]
        ctx = board_context(settings, self.rng)
        k = self.args.questions_per_image
        drawn = draw_questions(ctx, self.rng, k)
        if not drawn:
            raise RuntimeError("draw_questions returned nothing")
        dups = [False] * len(drawn)
        while len(drawn) < k:
            drawn.append(drawn[-1])
            dups.append(True)
        index = self.images_seen
        path = self.tmp / f"img_{index:07d}.png"
        noised = noise_image(_open_png(raw["png"]), self.rng, TRAINING_STRENGTH)
        _save_png(noised, path)
        kept: str | None = None
        every = self.args.save_image_every
        if every > 0 and index % every == 0:
            self.images_dir.mkdir(parents=True, exist_ok=True)
            dest = self.images_dir / f"img_{index:07d}.png"
            _save_png(noised, dest)
            kept = str(dest)
        builds = []
        questions = []
        for item, is_dup in zip(drawn, dups):
            messages = _build_game_messages(
                SYSTEM_PROMPT_S2, str(path), ctx.context, item.question,
                notepad=ctx.notepad,
            )
            reply = self.rng.choice(
                _S2_LOOK_LINES if item.kind == "look" else _S2_MOVE_LINES
            )
            example = TrainingExample(
                messages=messages,
                target_text=reply,
                loss="ce",
                source="s2_localizer",
                meta={"coords": item.coords},
            )
            builds.append(self.collator.build(example))
            questions.append({
                "kind": item.kind,
                "phrase": item.phrase,
                "point": list(item.point),
                "candidate_kind": item.candidate_kind,
                "question_kind": item.question_kind,
                "question": item.question,
                "coords": item.coords,
                "dup": is_dup,
            })
        path.unlink(missing_ok=True)
        if not bench:
            _append_jsonl(self.labels_path, {
                "image_index": index,
                "settings": settings,
                "notepad": ctx.notepad,
                "context": ctx.context,
                "questions": questions,
                "saved_image": kept,
            })
        return {"builds": builds, "questions": questions, "index": index}

    def _heartbeat(self, stage: str) -> None:
        assert self.tlog is not None
        self.tlog.heartbeat({
            "stage": stage,
            "images_seen": self.images_seen,
            "labels_seen": self.labels_seen,
            "step": self.step,
            "gpu": _gpu_mem_snapshot(),
        })

    def _stage_a_inputs(
        self, images: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[int], torch.Tensor]:
        """Pad the shared stem of each image. Returns inputs, splits, grid index."""
        assert self.collator is not None
        splits: list[int] = []
        stems: list[torch.Tensor] = []
        masks_src: list[dict[str, Any]] = []
        for image in images:
            builds = image["builds"]
            ids = [b["model_inputs"]["input_ids"] for b in builds]
            split = _lcp_len(ids)
            probe = {
                "input_ids": ids[0],
                "mm_token_type_ids": builds[0]["model_inputs"].get("mm_token_type_ids"),
                "token_type_ids": builds[0]["model_inputs"].get("token_type_ids"),
            }
            probe = {k: v for k, v in probe.items() if v is not None}
            positions = grid_positions(probe, self.tokenizer)
            if split <= int(positions.max()) + 1:
                raise RuntimeError(
                    f"stage-A split {split} is not past the image span "
                    f"(max image index {int(positions.max())}); "
                    "the questions did not diverge after the picture. "
                    "Use --single-stage."
                )
            if split <= self.prefix_n:
                raise RuntimeError(
                    f"stage-A split {split} does not cover the "
                    f"{self.prefix_n}-token system prefix"
                )
            weights = builds[0]["weights"]
            reply = (weights[0] != 0).nonzero(as_tuple=False).flatten()
            if reply.numel() == 0 or int(reply[0]) < split:
                raise RuntimeError(
                    "reply tokens are inside the shared stem; use --single-stage"
                )
            splits.append(split)
            stems.append(ids[0][0, :split])
            masks_src.append(builds[0]["model_inputs"])
        width = max(int(stem.numel()) for stem in stems)
        bsz = len(images)
        input_ids = torch.zeros(
            bsz, width, dtype=stems[0].dtype, device=self.device,
        )
        attention = torch.zeros(bsz, width, dtype=torch.long, device=self.device)
        for i, stem in enumerate(stems):
            n = int(stem.numel())
            input_ids[i, :n] = stem.to(self.device)
            attention[i, :n] = 1
        inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention,
        }
        sample = masks_src[0]
        seq0 = int(sample["input_ids"].shape[1])
        for key, val in sample.items():
            if key in ("input_ids", "attention_mask") or not torch.is_tensor(val):
                continue
            if _token_aligned(key, val, seq0):
                stacked = torch.zeros(
                    bsz, width, dtype=val.dtype, device=self.device,
                )
                for i, image in enumerate(images):
                    src = image["builds"][0]["model_inputs"][key]
                    n = splits[i]
                    stacked[i, :n] = src[0, :n].to(self.device)
                inputs[key] = stacked
            else:
                rows = []
                for image in images:
                    src = image["builds"][0]["model_inputs"][key]
                    if not torch.is_tensor(src) or src.shape[0] != 1:
                        raise RuntimeError(
                            f"stage A aux {key} is not a batch-1 tensor"
                        )
                    rows.append(src)
                trailing = rows[0].shape[1:]
                for row in rows[1:]:
                    if row.shape[1:] != trailing:
                        raise RuntimeError(
                            f"stage A aux {key} trailing {tuple(row.shape[1:])} "
                            f"!= {tuple(trailing)}"
                        )
                inputs[key] = torch.cat(rows, dim=0)
        full_for_grid = {
            "input_ids": input_ids,
            "mm_token_type_ids": inputs.get("mm_token_type_ids"),
            "token_type_ids": inputs.get("token_type_ids"),
        }
        full_for_grid = {k: v for k, v in full_for_grid.items() if v is not None}
        return inputs, splits, grid_positions(full_for_grid, self.tokenizer)

    def _run_stage_a(
        self, inputs: dict[str, Any],
    ) -> tuple[torch.Tensor, Any]:
        assert self.model is not None and self.tap is not None and self.tlog is not None
        self._heartbeat("A")
        if self.vram is not None:
            self.vram.set_stage("stage-a")
        _ensure_prefix_cache(
            self.model, inputs, self.prefix_kv, self.prefix_n, self.prefix_hash,
        )
        suf = _suffix_inputs_with_cache(inputs, self.prefix_kv, self.prefix_n)
        cache = suf["past_key_values"]
        hidden = truncated_forward(self.model, suf, self.tap)
        _drop_layers_above(cache, self.tap.index)
        return hidden, cache

    def _run_stage_b(
        self,
        images: list[dict[str, Any]],
        splits: list[int],
        cache_a: Any,
        a_width: int,
        a_mask: torch.Tensor,
    ) -> torch.Tensor:
        assert self.model is not None and self.tap is not None
        k = self.args.questions_per_image
        self._heartbeat("B")
        if self.vram is not None:
            self.vram.set_stage("stage-b")
        cache_b = repeat_kv_cache(cache_a, k)
        rows_ids: list[torch.Tensor] = []
        rows_extra: list[dict[str, torch.Tensor]] = []
        splits_row: list[int] = []
        for image, split in zip(images, splits):
            for build in image["builds"]:
                ids = build["model_inputs"]["input_ids"][0, split:]
                rows_ids.append(ids)
                extra = {}
                full = build["model_inputs"]
                seq = int(full["input_ids"].shape[1])
                for key, val in full.items():
                    if torch.is_tensor(val) and _token_aligned(key, val, seq):
                        extra[key] = val[0, split:]
                rows_extra.append(extra)
                splits_row.append(split)
        b_max = max(int(row.numel()) for row in rows_ids)
        bk = len(rows_ids)
        input_ids = torch.zeros(bk, b_max, dtype=rows_ids[0].dtype, device=self.device)
        b_mask = torch.zeros(bk, b_max, dtype=torch.long, device=self.device)
        position_ids = torch.empty(bk, b_max, dtype=torch.long, device=self.device)
        for i, row in enumerate(rows_ids):
            n = int(row.numel())
            input_ids[i, :n] = row.to(self.device)
            b_mask[i, :n] = 1
            position_ids[i] = splits_row[i] + torch.arange(
                b_max, device=self.device,
            )
        past_mask = a_mask.repeat_interleave(k, dim=0)
        attention = torch.cat((past_mask, b_mask), dim=1)
        inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention,
            "position_ids": position_ids,
            "past_key_values": cache_b,
            # Append index in the cache. RoPE uses the per-row
            # position_ids above; this cursor is the same for every row
            # because every stem was padded to ``a_width``.
            "cache_position": torch.arange(
                a_width, a_width + b_max, device=self.device,
            ),
        }
        keys = rows_extra[0].keys()
        for key in keys:
            stacked = torch.zeros(
                bk, b_max, dtype=rows_extra[0][key].dtype, device=self.device,
            )
            for i, extra in enumerate(rows_extra):
                src = extra[key]
                stacked[i, : int(src.numel())] = src.to(self.device)
            inputs[key] = stacked
        if attention.shape[1] != a_width + b_max:
            raise RuntimeError(
                f"stage-B mask width {attention.shape[1]} != "
                f"A_max {a_width} + B_max {b_max}"
            )
        return truncated_forward(self.model, inputs, self.tap)

    def _queries_from_hidden(
        self,
        image: dict[str, Any],
        split: int,
        hidden_rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``hidden_rows [K, T, H]`` -> queries ``[K, R, H]``, valid ``[K, R]``."""
        k = self.args.questions_per_image
        queries = torch.zeros(
            k, self.r_max, self.hidden, dtype=torch.bfloat16, device=self.device,
        )
        valid = torch.zeros(k, self.r_max, dtype=torch.bool, device=self.device)
        limit = self.args.reply_positions
        for q, build in enumerate(image["builds"]):
            weights = build["weights"][0]
            reply = (weights != 0).nonzero(as_tuple=False).flatten()
            if limit > 0:
                reply = reply[:limit]
            local = reply - split
            if local.numel() == 0:
                raise RuntimeError("question has no reply tokens")
            if int(local[0]) < 0 or int(local[-1]) >= hidden_rows.shape[1]:
                raise RuntimeError(
                    f"reply positions {local[:4].tolist()} outside stage B "
                    f"length {hidden_rows.shape[1]}"
                )
            if int(local.numel()) > self.r_max:
                raise RuntimeError(
                    f"reply length {int(local.numel())} exceeds r_max {self.r_max}"
                )
            taken = hidden_rows[q, local.to(hidden_rows.device)]
            n = int(taken.shape[0])
            queries[q, :n] = taken.to(torch.bfloat16)
            valid[q, :n] = True
        return queries, valid

    def _pack_row(
        self,
        image: dict[str, Any],
        grid: torch.Tensor,
        queries: torch.Tensor,
        valid: torch.Tensor,
        *,
        record: bool,
    ) -> dict[str, Any]:
        k = self.args.questions_per_image
        labels = torch.tensor(
            [q["coords"] for q in image["questions"]],
            dtype=torch.float32, device=self.device,
        )
        if labels.shape != (k, 6):
            raise RuntimeError(f"labels shape {tuple(labels.shape)}, expected {(k, 6)}")
        move = torch.tensor(
            [_KIND_MOVE[q["kind"]] for q in image["questions"]],
            dtype=torch.int8, device=self.device,
        )
        cand = torch.tensor(
            [_KIND_CAND[q["candidate_kind"]] for q in image["questions"]],
            dtype=torch.int8, device=self.device,
        )
        phrases = torch.tensor(
            [self._phrase(q["phrase"]) for q in image["questions"]],
            dtype=torch.int32, device=self.device,
        )
        dup = torch.tensor(
            [bool(q["dup"]) for q in image["questions"]],
            dtype=torch.bool, device=self.device,
        )
        if record:
            fresh = [q for q in image["questions"] if not q["dup"]]
            self.floors.update(
                labels[~dup].detach().cpu(),
                [self.phrase_ids[q["phrase"]] for q in fresh],
            )
            self.labels_seen += len(fresh)
            self.images_seen += 1
        return {
            "grid": grid.to(torch.bfloat16),
            "queries": queries,
            "valid_r": valid,
            "labels": labels,
            "move_kind": move,
            "cand_kind": cand,
            "phrase_id": phrases,
            "dup": dup,
        }

    def _two_stage(
        self, images: list[dict[str, Any]], *, record: bool,
    ) -> tuple[float, float, list[dict[str, Any]]]:
        """One stage-A batch and one stage-B batch for every image.

        Stems are right-padded to one width, so the trunk batch is
        ``--images-per-batch`` even when the questions diverge at
        different tokens. The cached length is that padded width (it
        already includes the system prefix). Stage B's mask is
        ``[B*K, width + B_max]``, not ``prefix + width``.
        """
        inputs, splits, positions = self._stage_a_inputs(images)
        self._assert_same_prefix(inputs["input_ids"])
        width = int(inputs["input_ids"].shape[1])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        hidden_a, cache_a = self._run_stage_a(inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stage_a = time.perf_counter() - t0
        if int(hidden_a.shape[1]) != width - self.prefix_n:
            raise RuntimeError(
                f"stage-A hidden length {int(hidden_a.shape[1])} != "
                f"stem {width} - prefix {self.prefix_n}"
            )
        local_pos = positions - self.prefix_n
        if int(local_pos.min()) < 0 or int(local_pos.max()) >= int(hidden_a.shape[1]):
            raise RuntimeError(
                f"image positions {int(local_pos.min())}..{int(local_pos.max())} "
                f"fall outside the stage-A suffix of length {int(hidden_a.shape[1])}"
            )
        grid = gather_grid(hidden_a, local_pos)
        seq_len = _layer0_len(cache_a)
        if seq_len != width:
            raise RuntimeError(
                f"stage-A cache length {seq_len} != padded stem {width}"
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        hidden_b = self._run_stage_b(
            images, splits, cache_a, width, inputs["attention_mask"],
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stage_b = time.perf_counter() - t1
        k = self.args.questions_per_image
        if hidden_b.shape[0] != len(images) * k:
            raise RuntimeError(
                f"stage-B batch {hidden_b.shape[0]} != {len(images)} * {k}"
            )
        packed = []
        for i, image in enumerate(images):
            rows = hidden_b[i * k:(i + 1) * k]
            queries, valid = self._queries_from_hidden(image, splits[i], rows)
            packed.append(self._pack_row(
                image, grid[i], queries, valid, record=record,
            ))
        return stage_a, stage_b, packed

    def _assert_same_prefix(self, input_ids: torch.Tensor) -> None:
        if self.prefix_n <= 0 or int(input_ids.shape[1]) <= self.prefix_n:
            raise RuntimeError(
                f"sequence length {int(input_ids.shape[1])} does not cover "
                f"the {self.prefix_n}-token system prefix"
            )
        pref = input_ids[:, :self.prefix_n]
        if not torch.equal(pref, pref[:1].expand_as(pref)):
            raise RuntimeError("system prefix ids differ across the batch")

    def _single_stage(
        self, images: list[dict[str, Any]], *, record: bool,
    ) -> tuple[float, list[dict[str, Any]]]:
        assert self.model is not None and self.tap is not None and self.collator is not None
        self._heartbeat("A")
        if self.vram is not None:
            self.vram.set_stage("stage-a")
        builds = [build for image in images for build in image["builds"]]
        keys = builds[0]["model_inputs"].keys()
        width = max(int(b["model_inputs"]["input_ids"].shape[1]) for b in builds)
        bsz = len(builds)
        sample_ids = builds[0]["model_inputs"]["input_ids"]
        input_ids = torch.zeros(bsz, width, dtype=sample_ids.dtype, device=self.device)
        attention = torch.zeros(bsz, width, dtype=torch.long, device=self.device)
        weights = torch.zeros(bsz, width, dtype=torch.float32, device=self.device)
        for i, build in enumerate(builds):
            src = build["model_inputs"]
            n = int(src["input_ids"].shape[1])
            input_ids[i, :n] = src["input_ids"][0].to(self.device)
            attention[i, :n] = 1
            weights[i, :n] = build["weights"][0].to(self.device)
        inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention,
        }
        seq0 = int(builds[0]["model_inputs"]["input_ids"].shape[1])
        for key in keys:
            val = builds[0]["model_inputs"][key]
            if key in ("input_ids", "attention_mask") or not torch.is_tensor(val):
                continue
            if _token_aligned(key, val, seq0):
                stacked = torch.zeros(bsz, width, dtype=val.dtype, device=self.device)
                for i, build in enumerate(builds):
                    src = build["model_inputs"][key][0]
                    stacked[i, : int(src.numel())] = src.to(self.device)
                inputs[key] = stacked
            else:
                rows = [b["model_inputs"][key] for b in builds]
                trailing = rows[0].shape[1:]
                for row in rows[1:]:
                    if row.shape[1:] != trailing:
                        raise RuntimeError(
                            f"single-stage aux {key} trailing shapes differ"
                        )
                inputs[key] = torch.cat(rows, dim=0)
        self._assert_same_prefix(inputs["input_ids"])
        _ensure_prefix_cache(
            self.model, inputs, self.prefix_kv, self.prefix_n, self.prefix_hash,
        )
        suf = _suffix_inputs_with_cache(inputs, self.prefix_kv, self.prefix_n)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        hidden = truncated_forward(self.model, suf, self.tap)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        positions = grid_positions(inputs, self.tokenizer) - self.prefix_n
        if int(positions.min()) < 0 or int(positions.max()) >= int(hidden.shape[1]):
            raise RuntimeError(
                f"image positions {int(positions.min())}..{int(positions.max())} "
                f"fall outside the suffix of length {int(hidden.shape[1])}"
            )
        grids = gather_grid(hidden, positions)
        k = self.args.questions_per_image
        packed = []
        limit = self.args.reply_positions
        for i, image in enumerate(images):
            rows = hidden[i * k:(i + 1) * k]
            wrows = weights[i * k:(i + 1) * k]
            queries = torch.zeros(
                k, self.r_max, self.hidden, dtype=torch.bfloat16, device=self.device,
            )
            valid = torch.zeros(k, self.r_max, dtype=torch.bool, device=self.device)
            for q in range(k):
                reply = (wrows[q] != 0).nonzero(as_tuple=False).flatten() - self.prefix_n
                if limit > 0:
                    reply = reply[:limit]
                if reply.numel() == 0 or int(reply[0]) < 0:
                    raise RuntimeError("single-stage reply position is outside the suffix")
                if int(reply[-1]) >= int(rows.shape[1]):
                    raise RuntimeError(
                        f"reply index {int(reply[-1])} outside hidden length {int(rows.shape[1])}"
                    )
                if int(reply.numel()) > self.r_max:
                    raise RuntimeError(
                        f"reply length {int(reply.numel())} exceeds r_max {self.r_max}"
                    )
                taken = rows[q, reply.to(rows.device)]
                n = int(taken.shape[0])
                queries[q, :n] = taken.to(torch.bfloat16)
                valid[q, :n] = True
            packed.append(self._pack_row(
                image, grids[i * k], queries, valid, record=record,
            ))
        return elapsed, packed

    def _equivalence(self, image: dict[str, Any], two: dict[str, Any]) -> None:
        """First question, two-stage against a one-row single pass.

        The single pass does not count the image again.
        """
        assert self.tlog is not None
        _elapsed, rows = self._single_stage([image], record=False)
        one = rows[0]
        q_two = two["queries"][0][two["valid_r"][0]]
        q_one = one["queries"][0][one["valid_r"][0]]
        if q_two.shape != q_one.shape:
            raise RuntimeError(
                f"equivalence query shapes {tuple(q_two.shape)} vs {tuple(q_one.shape)}"
            )
        q_rms, q_max = _rel_err(q_two, q_one)
        g_rms, g_max = _rel_err(two["grid"], one["grid"])
        logger.info(
            "equivalence query rms %.6f max %.6f | grid rms %.6f max %.6f "
            "(rms limit %.2f, max limit %.2f)",
            q_rms, q_max, g_rms, g_max, _EQUIV_TOL, _EQUIV_MAX_TOL,
        )
        self.tlog.event(
            "equivalence_check",
            query_rms=q_rms, query_max=q_max,
            grid_rms=g_rms, grid_max=g_max,
            rms_limit=_EQUIV_TOL, max_limit=_EQUIV_MAX_TOL,
        )
        if (
            q_rms >= _EQUIV_TOL or g_rms >= _EQUIV_TOL
            or q_max >= _EQUIV_MAX_TOL or g_max >= _EQUIV_MAX_TOL
        ):
            raise RuntimeError(
                f"two-stage vs single-stage query rms {q_rms:.6f} max {q_max:.6f}, "
                f"grid rms {g_rms:.6f} max {g_max:.6f} "
                f"(rms limit {_EQUIV_TOL}, max limit {_EQUIV_MAX_TOL}). "
                "Rerun with --single-stage."
            )

    def ingest(self, n_images: int, *, bench: bool, check: bool) -> None:
        assert self.buffer is not None
        if self.args.questions_per_image < 2 and not self.args.single_stage:
            raise RuntimeError(
                "--questions-per-image must be >= 2 on the two-stage path; "
                "pass --single-stage for one question per image"
            )
        batch = self.args.images_per_batch
        remaining = n_images
        while remaining > 0:
            if self._stop:
                return
            take = min(batch, remaining)
            images = [self._encode_image(bench) for _ in range(take)]
            if self.args.single_stage:
                _elapsed, packed = self._single_stage(images, record=True)
            else:
                _stage_a, _stage_b, packed = self._two_stage(images, record=True)
            if check and not self._checked and not self.args.single_stage:
                self._equivalence(images[0], packed[0])
                self._checked = True
            for row in packed:
                self.buffer.add(row)
            remaining -= take

    def bench_round(self, images: list[dict[str, Any]], check: bool) -> dict[str, float]:
        assert self.buffer is not None
        if self.args.single_stage:
            stage_a, packed = self._single_stage(images, record=True)
            stage_b = 0.0
        else:
            stage_a, stage_b, packed = self._two_stage(images, record=True)
        if check:
            self._equivalence(images[0], packed[0])
            self._checked = True
        for row in packed:
            self.buffer.add(row)
        n_labels = sum(
            1 for image in images for q in image["questions"] if not q["dup"]
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        steps, examples = self.localizer_pass()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {
            "stage_a": stage_a,
            "stage_b": stage_b,
            "localizer": time.perf_counter() - t1,
            "steps": float(steps),
            "examples": float(examples),
            "images": float(len(images)),
            "labels": float(n_labels),
        }
    def _valid_index(self) -> torch.Tensor:
        assert self.buffer is not None
        buf = self.buffer
        slots = []
        # The ring's live rows are the last ``filled`` writes.
        start = self.buffer.write - self.buffer.filled
        for offset in range(buf.filled):
            i = (start + offset) % buf.n
            for q in range(buf.k):
                if bool(buf.dup[i, q]):
                    continue
                for r in range(buf.r_max):
                    if bool(buf.valid_r[i, q, r]):
                        slots.append((i, q, r))
        if not slots:
            raise RuntimeError("localizer pass has no valid (image, question, reply) triples")
        return torch.tensor(slots, dtype=torch.long, device=self.device)

    def localizer_pass(self) -> tuple[int, int]:
        assert self.buffer is not None and self.localizer is not None
        self._heartbeat("localizer")
        if self.vram is not None:
            self.vram.set_stage("localizer")
        filled = self.buffer.filled
        k = self.args.questions_per_image
        steps = max(1, filled * k // self.args.localizer_batch)
        index = self._valid_index()
        examples = 0
        done = 0
        for _ in range(steps):
            if self._stop:
                break
            examples += self._optimizer_step(index)
            done += 1
        return done, examples

    def _optimizer_step(self, index: torch.Tensor) -> int:
        assert (
            self.buffer is not None and self.localizer is not None
            and self.optimizer is not None and self.scheduler is not None
            and self.tlog is not None and self.coord_embed is not None
        )
        n = self.args.localizer_batch
        choice = torch.randint(0, index.shape[0], (n,), device=self.device)
        picked = index[choice]
        ii, qq, rr = picked[:, 0], picked[:, 1], picked[:, 2]
        grids = self.buffer.grids[ii]
        queries = self.buffer.queries[ii, qq, rr].unsqueeze(1)
        labels = self.buffer.labels[ii, qq]
        pred, cell_logits = self.localizer(grids, queries)
        pred = pred[:, 0, :]
        logits = cell_logits[:, 0]
        err = (pred - labels).pow(2)
        l2_s = err[:, 0:2].mean()
        l2_v = err[:, 2:4].mean()
        l2_vbar = err[:, 4:6].mean()
        ces = []
        accs = []
        for head, sl in enumerate((slice(0, 2), slice(2, 4), slice(4, 6))):
            target = cell_index(labels[:, sl])
            valid = target != -1
            if bool(valid.any()):
                ces.append(F.cross_entropy(
                    logits[:, head, :], target, ignore_index=-1,
                ))
            else:
                ces.append(logits.new_zeros(()))
            if bool(valid.any()):
                hit = logits[:, head, :].argmax(dim=-1) == target
                accs.append(float(hit[valid].float().mean()))
            else:
                accs.append(float("nan"))
        if not ces:
            raise RuntimeError("cell CE has no valid targets")
        cell_ce = torch.stack(ces).mean()
        loss = (l2_s + l2_v + l2_vbar) / 3 + self.args.cell_ce_weight * cell_ce
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite localizer loss {float(loss)}")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.localizer.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite grad norm {float(grad_norm)}")
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        if self.step >= self.cosine_steps and not self._floor_warned:
            self._floor_warned = True
            self.tlog.event("cosine_floor", step=self.step, lr=self._lr())
            logger.warning("cosine floor reached at step %d; lr stays at the floor", self.step)
        record = self._metrics(
            loss, l2_s, l2_v, l2_vbar, cell_ce, accs,
            err[:, 2:4].mean(dim=-1).detach(),
            labels, ii, qq, float(grad_norm),
        )
        self.tlog.last_step(record)
        _append_jsonl(self.tlog.run_dir / "metrics.jsonl", record)
        if self.step % self.args.log_steps == 0:
            logger.info("%s", _format_log(record))
        if self.step % 100 == 0:
            self._flush_state()
        if self.step % self.args.save_steps == 0:
            self.save("save_steps")
        return n

    def _lr(self) -> float:
        assert self.optimizer is not None
        return float(self.optimizer.param_groups[0]["lr"])

    def _metrics(
        self,
        loss: torch.Tensor,
        l2_s: torch.Tensor,
        l2_v: torch.Tensor,
        l2_vbar: torch.Tensor,
        cell_ce: torch.Tensor,
        accs: list[float],
        err_v: torch.Tensor,
        labels: torch.Tensor,
        ii: torch.Tensor,
        qq: torch.Tensor,
        grad_norm: float,
    ) -> dict[str, Any]:
        assert self.buffer is not None and self.coord_embed is not None
        move = self.buffer.move_kind[ii, qq].tolist()
        cand = self.buffer.cand_kind[ii, qq].tolist()
        phrases = self.buffer.phrase_id[ii, qq].tolist()
        cpu_labels = labels.detach().float().cpu()
        const = self.floors.constant()
        text = self.floors.text_only(
            cpu_labels, [int(p) for p in phrases], [int(m) for m in move],
        )
        err_v = err_v.detach().float().cpu()
        by_kind = {}
        for name, code in _KIND_CAND.items():
            mask = [c == code for c in cand]
            if any(mask):
                sel = err_v[[i for i, flag in enumerate(mask) if flag]]
                by_kind[name] = math.sqrt(float(sel.mean()))
            else:
                by_kind[name] = None
        return {
            "step": self.step,
            "images_seen": self.images_seen,
            "labels_seen": self.labels_seen,
            "loss": float(loss.detach()),
            "l2_s": float(l2_s.detach()),
            "l2_v": float(l2_v.detach()),
            "l2_vbar": float(l2_vbar.detach()),
            "rms_s": math.sqrt(max(0.0, float(l2_s.detach()))),
            "rms_v": math.sqrt(max(0.0, float(l2_v.detach()))),
            "rms_vbar": math.sqrt(max(0.0, float(l2_vbar.detach()))),
            "rms_v_gold": by_kind["gold"],
            "rms_v_exit": by_kind["exit"],
            "rms_v_corner": by_kind["corner"],
            "cell_ce": float(cell_ce.detach()),
            "cell_acc_s": accs[0],
            "cell_acc_v": accs[1],
            "floor_const_s": const["s"],
            "floor_const_v": const["v"],
            "floor_const_vbar": const["v_bar"],
            "floor_text_s": text["s"],
            "floor_text_v": text["v"],
            "floor_text_vbar": text["v_bar"],
            "gate_v": float(self.coord_embed.gate_v.detach()),
            "gate_vbar": float(self.coord_embed.gate_vbar.detach()),
            "lr": self._lr(),
            "grad_norm": grad_norm,
            "gpu": _gpu_mem_snapshot(),
        }

    def save(self, reason: str) -> None:
        assert self.localizer is not None and self.coord_embed is not None and self.tlog is not None
        meta = {
            "trainer": TRAINER_NAME,
            "gemma_base": self.args.gemma_base,
            "kept_layer": self.args.kept_layer,
            "model_key": "gemma-4-12b",
            "step": self.step,
            "images_seen": self.images_seen,
            "labels_seen": self.labels_seen,
            "phrase_ids": self.phrase_ids,
            "reason": reason,
        }
        blob = {
            "state_dict": self.localizer.state_dict(),
            "kept_layer": self.args.kept_layer,
            "dim": self.localizer.dim,
            "refine_layers": self.localizer.refine_layers,
            "heads": self.localizer.heads,
            "mlp_dim": self.localizer.mlp_dim,
            "hidden_size": self.localizer.hidden_size,
        }
        for directory in (
            self.s2_root / f"{self.label}_step_{self.step:06d}",
            self.s2_root / f"{self.label}_last",
        ):
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(blob, directory / "localizer.pt")
            self.coord_embed.save(directory / "coord_embed.pt")
            _write_json_atomic(directory / "train_meta.json", meta)
        self._flush_state()
        self.tlog.event("snapshot_saved", step=self.step, reason=reason)
        logger.info("saved step %d (%s)", self.step, reason)

    def _flush_state(self) -> None:
        _write_json_atomic(self.state_path, {
            "trainer": TRAINER_NAME,
            "step": self.step,
            "images_seen": self.images_seen,
            "labels_seen": self.labels_seen,
            "phrase_ids": self.phrase_ids,
        })

    def _on_signal(self, signum: int, _frame: Any) -> None:
        self._stop = True
        logger.warning("signal %s; will save at the next step boundary", signum)

    def train(self) -> None:
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)
        deadline = time.perf_counter() + self.args.hours * 3600.0
        refill = max(1, math.ceil(self.args.buffer_images * self.args.refill_fraction))
        epochs = self.args.localizer_epochs_per_refill
        logger.info(
            "train hours=%.2f refill_images=%d epochs_per_refill=%d cosine_steps=%d",
            self.args.hours, refill, epochs, self.cosine_steps,
        )
        while not self._stop and time.perf_counter() < deadline:
            self.ingest(refill, bench=False, check=not self._checked)
            for _epoch in range(epochs):
                if self._stop or time.perf_counter() >= deadline:
                    break
                self.localizer_pass()
        reason = "signal" if self._stop else "hours"
        self.save(reason)

    def bench(self, rounds: int) -> None:
        assert self.tlog is not None and self.buffer is not None
        if rounds < 1:
            raise ValueError(f"--bench {rounds}")
        totals = {"stage_a": 0.0, "stage_b": 0.0, "localizer": 0.0,
                  "steps": 0.0, "examples": 0.0, "images": 0.0, "labels": 0.0}
        peak_torch = 0
        peak_smi = 0
        total_mi = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for rnd in range(rounds):
            images = [
                self._encode_image(True) for _ in range(self.args.images_per_batch)
            ]
            stats = self.bench_round(images, check=(rnd == 0 and not self.args.single_stage))
            for key in totals:
                totals[key] += stats[key]
            snap = _gpu_mem_snapshot()
            peak_smi = max(peak_smi, int(snap.get("nvidia_smi_used_mi") or 0))
            total_mi = int(snap.get("total_mi") or 0)
        if torch.cuda.is_available():
            peak_torch = int(torch.cuda.max_memory_allocated() / (1024 ** 2))
        images = totals["images"]
        labels = totals["labels"]
        a_s = totals["stage_a"] / rounds
        b_s = totals["stage_b"] / rounds
        loc_s = totals["localizer"] / max(1.0, totals["steps"])
        img_rate = images / totals["stage_a"] if totals["stage_a"] else 0.0
        if self.args.single_stage:
            lab_rate = labels / totals["stage_a"] if totals["stage_a"] else 0.0
        else:
            lab_rate = labels / totals["stage_b"] if totals["stage_b"] else 0.0
        ex_rate = totals["examples"] / totals["localizer"] if totals["localizer"] else 0.0
        hours_labels = (1_000_000 / lab_rate / 3600.0) if lab_rate else float("inf")
        hours_images = (1_000_000 / img_rate / 3600.0) if img_rate else float("inf")
        torch_gib = peak_torch / 1024.0
        smi_gib = peak_smi / 1024.0
        total_gib = total_mi / 1024.0
        lines = [
            (
                f"bench: B={self.args.images_per_batch} "
                f"K={self.args.questions_per_image} R={self.r_max} "
                f"kept_layer={self.args.kept_layer} "
                f"single_stage={self.args.single_stage}"
            ),
            f"stage A  {a_s:.3f} s/batch   {img_rate:.1f} images/s",
            f"stage B  {b_s:.3f} s/batch   {lab_rate:.1f} labels/s",
            (
                f"localizer {loc_s:.3f} s/step  {ex_rate:.0f} examples/s"
                f"   (batch {self.args.localizer_batch})"
            ),
            (
                f"peak VRAM torch {torch_gib:.1f} GiB   "
                f"nvidia-smi {smi_gib:.1f} GiB / {total_gib:.1f} GiB"
            ),
            (
                f"at this rate: 1,000,000 labels in {hours_labels:.2f} h; "
                f"1,000,000 images in {hours_images:.1f} h"
            ),
        ]
        text = "\n".join(lines)
        print(text, flush=True)
        logger.info("\n%s", text)
        payload = {
            "lines": lines,
            "images_per_s": img_rate,
            "labels_per_s": lab_rate,
            "examples_per_s": ex_rate,
            "hours_per_million_labels": hours_labels,
            "hours_per_million_images": hours_images,
            "peak_torch_mib": peak_torch,
            "peak_nvidia_smi_mib": peak_smi,
            "total_mib": total_mi,
            "single_stage": self.args.single_stage,
            "kept_layer": self.args.kept_layer,
        }
        (self.tlog.run_dir / "bench.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8",
        )


def _drop_layers_above(cache: Any, index: int) -> None:
    """Upper layers keep only the prefix after a truncated forward.

    Stage B repeats batch on the layers that actually ran. A shorter
    prefix left on the layers above ``index`` disagrees with that
    length, so those tensors are dropped. ``repeat_kv_cache`` leaves
    ``None`` alone.
    """
    if not hasattr(cache, "layers"):
        raise RuntimeError(f"cache has no layers: {type(cache)!r}")
    for i, layer in enumerate(cache.layers):
        if i <= index:
            continue
        if getattr(layer, "keys", None) is not None:
            layer.keys = None
        if getattr(layer, "values", None) is not None:
            layer.values = None


def _layer0_len(cache: Any) -> int:
    if not hasattr(cache, "layers"):
        raise RuntimeError(f"stage-A cache has no layers: {type(cache)!r}")
    keys = cache.layers[0].keys
    if keys is None or not torch.is_tensor(keys) or keys.numel() == 0:
        raise RuntimeError("stage-A cache layer 0 has no keys")
    return int(keys.shape[-2])


def _format_log(record: dict[str, Any]) -> str:
    def kind(value: Any) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "n/a"
        return f"{value:.2f}"

    def num(value: Any, spec: str) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "n/a"
        return format(value, spec)

    return (
        f"step {record['step']} loc "
        f"rms s {record['rms_s']:.3f} v {record['rms_v']:.3f} "
        f"vbar {record['rms_vbar']:.3f} | "
        f"v by kind gold {kind(record['rms_v_gold'])} "
        f"exit {kind(record['rms_v_exit'])} "
        f"corner {kind(record['rms_v_corner'])} | "
        f"cell acc s {num(record['cell_acc_s'], '.2f')} "
        f"v {num(record['cell_acc_v'], '.2f')} | "
        f"floor const s {num(record['floor_const_s'], '.3f')} "
        f"v {num(record['floor_const_v'], '.3f')} "
        f"vbar {num(record['floor_const_vbar'], '.3f')} | "
        f"floor text s {num(record['floor_text_s'], '.3f')} "
        f"v {num(record['floor_text_v'], '.3f')} "
        f"vbar {num(record['floor_text_vbar'], '.3f')} | "
        f"images {record['images_seen']} labels {record['labels_seen']} | "
        f"lr {record['lr']:.1e}"
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m neural_net.s2_pretraining_localizer",
        description="Frozen-trunk Localizer training, or --bench N.",
    )
    p.add_argument("--gemma-base", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--hours", type=float, default=18.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--kept-layer", type=int, default=KEPT_LAYER)
    p.add_argument("--images-per-batch", type=int, default=16)
    p.add_argument("--questions-per-image", type=int, default=6)
    p.add_argument("--reply-positions", type=int, default=0)
    p.add_argument("--buffer-images", type=int, default=4096)
    p.add_argument("--refill-fraction", type=float, default=0.125)
    p.add_argument("--localizer-batch", type=int, default=256)
    p.add_argument("--localizer-epochs-per-refill", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-floor", type=float, default=0.1)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--cosine-steps", type=int, default=0)
    p.add_argument("--cell-ce-weight", type=float, default=0.5)
    p.add_argument("--single-stage", action="store_true")
    p.add_argument("--save-steps", type=int, default=2000)
    p.add_argument("--log-steps", type=int, default=20)
    p.add_argument("--save-image-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resume-snapshot", default=None)
    p.add_argument("--bench", type=int, default=0)
    return p


def main() -> None:
    configure_logging()
    args = _build_parser().parse_args()
    if args.images_per_batch < 1 or args.questions_per_image < 1:
        raise SystemExit("batch sizes must be >= 1")
    if args.buffer_images < 1 or args.localizer_batch < 1:
        raise SystemExit("buffer and localizer batch must be >= 1")
    trainer = Trainer(args)
    try:
        load_s = trainer.load()
        print(f"load {load_s:.1f} s", flush=True)
        if args.bench:
            trainer.bench(args.bench)
            return
        trainer.train()
    except BaseException as exc:
        if trainer.tlog is not None and not isinstance(exc, KeyboardInterrupt):
            trainer.tlog.record_crash(exc, step=trainer.step)
        if (
            not args.bench
            and trainer.localizer is not None
            and not isinstance(exc, KeyboardInterrupt)
        ):
            try:
                trainer.save("crash")
            except Exception:
                logger.error("crash snapshot failed\n%s", traceback.format_exc())
        if isinstance(exc, KeyboardInterrupt):
            if not args.bench and trainer.localizer is not None:
                trainer.save("signal")
        raise
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
