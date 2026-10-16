"""Five-minute control, then three five-minute volume probes.

The control is aug27 with no vector embedding. Its answer-token hidden
states stay in CPU RAM. Each later pass adds the same coordinates at
one volume (barely, half, recommended) and reports CE against the
dataset targets, the control CE, and KD against the stored control.

Load time is outside the clocks. A pass that finishes its work early
stops. The volume passes also print two short generations, control and
with the embedding.
"""

from __future__ import annotations

import argparse
import logging
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from agent.config import CONFIG
from agent.model import ADAPTERS, spec_for
from neural_net.coord_embed import MAGNITUDES, CoordEmbedder
from neural_net.paths import weights_root
from neural_net.s2 import _hidden_size, _wants_cache_position
from neural_net.s2_coord_oracle import REPLY_JITTER_SIGMA
from training.external_data import sources_from_manifest
from training.token_fence import measure_image_soft_tokens
from training.train import (
    Collator,
    TrainConfig,
    TrainingExample,
    _forward_last_hidden,
    apply_lm_head_chunked,
    build_model,
    configure_logging,
    load_adapter_state,
    materialize,
    resolve_terminator_id,
)

logger = logging.getLogger("s2_coord_probe")

PASS_S = 5.0 * 60.0
CACHE_TOKEN_CAP = 2_000_000
GEN_RESERVE_S = 45.0
MAX_NEW_TOKENS = 48
HOLDOUT_FRACTION = 0.05
HOLDOUT_CAP = 100
KD_CHUNK = 1024
COORD_LOW = -1.0
COORD_HIGH = 2.0


@dataclass
class CachedExample:
    example: TrainingExample
    hidden: torch.Tensor
    labels: torch.Tensor
    control_ce: float
    coords: torch.Tensor
    base: tuple[float, float, float, float]


def _holdout(by_source: dict[str, list[TrainingExample]],
             rng: random.Random) -> list[TrainingExample]:
    """Same split as training: 5% per source, cap 100, fixed seed."""
    held: list[TrainingExample] = []
    for exs in by_source.values():
        rng.shuffle(exs)
        n_hold = max(1, int(len(exs) * HOLDOUT_FRACTION))
        n_hold = min(n_hold, HOLDOUT_CAP)
        held.extend(exs[:n_hold])
    rng.shuffle(held)
    return held


def _jitter_pair(
    rng: random.Random,
    s: tuple[float, float],
    v: tuple[float, float],
) -> list[float]:
    for _ in range(100):
        row: list[float] = []
        for x, y in (s, v):
            row.append(x + rng.gauss(0.0, REPLY_JITTER_SIGMA))
            row.append(y + rng.gauss(0.0, REPLY_JITTER_SIGMA))
        if all(COORD_LOW <= value <= COORD_HIGH for value in row):
            return row
    raise RuntimeError("coordinate noise stayed outside [-1, 2]")


def _draw_base(rng: random.Random) -> tuple[float, float, float, float]:
    return (
        rng.random(), rng.random(),
        rng.random(), rng.random(),
    )


def _coords_for(
    rng: random.Random,
    base: tuple[float, float, float, float],
    n: int,
) -> torch.Tensor:
    s = (base[0], base[1])
    v = (base[2], base[3])
    rows = [_jitter_pair(rng, s, v) for _ in range(n)]
    return torch.tensor(rows, dtype=torch.float32)


def _target_gather(input_ids: torch.Tensor, weights: torch.Tensor) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Gold ids, hidden indexes that predict them, and those token indexes.

    Batch size is 1. ``weights`` marks assistant tokens, including the
    terminator. The hidden that predicts token ``t`` sits at ``t - 1``.
    """
    if int(input_ids.shape[0]) != 1:
        raise RuntimeError(f"probe gather expects batch 1, got {tuple(input_ids.shape)}")
    seq_len = int(input_ids.shape[1])
    nz = weights[:, 1:] != 0
    if not bool(nz.any()):
        raise RuntimeError("example has no target tokens")
    first = int(nz.float().argmax(dim=1).min()) + 1
    tail = seq_len - first
    labels_tail = input_ids[:, -tail:]
    n_pos = int(labels_tail.shape[1])
    flat_w = weights[:, -tail:].reshape(-1)
    mask_idx = (flat_w != 0).nonzero(as_tuple=True)[0]
    col = mask_idx % n_pos
    labels = labels_tail.reshape(-1)[mask_idx]
    hidden_index = seq_len - (tail + 1) + col
    reply_index = seq_len - tail + col
    return labels, hidden_index, reply_index


def _token_ce(logits: torch.Tensor, labels: torch.Tensor) -> float:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    nll = -log_probs.gather(1, labels[:, None]).squeeze(1)
    return float(nll.mean())


def _soft_ce(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> float:
    teacher = F.softmax(teacher_logits.float(), dim=-1)
    student = F.log_softmax(student_logits.float(), dim=-1)
    return float(-(teacher * student).sum(dim=-1).mean())


def _report(message: str) -> None:
    logger.info(message)
    print(message, flush=True)


def _user_text(example: TrainingExample) -> str:
    parts: list[str] = []
    for message in example.messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
    return "\n".join(parts)


class Probe:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rng = random.Random(args.seed)
        self.device = args.device
        self.spec = spec_for(CONFIG.model_key)
        self.model: Any = None
        self.processor: Any = None
        self.collator: Collator | None = None
        self.embedder: CoordEmbedder | None = None
        self.terminator = 0
        self.cached: list[CachedExample] = []

    def load(self) -> None:
        cfg = TrainConfig(
            label="coord_probe", device=self.device, neftune_alpha=0.0,
        )
        model, processor, _, _ = build_model(self.spec, cfg, CONFIG.hf_token)
        ckpt = weights_root() / self.spec.key / self.args.gemma_base
        if not (ckpt / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"no aug27 adapter under {ckpt} (--gemma-base {self.args.gemma_base})"
            )
        load_adapter_state(model, ckpt)
        model.eval()
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        model.config.use_cache = False
        tokenizer = getattr(processor, "tokenizer", processor)
        self.terminator = resolve_terminator_id(model, tokenizer)
        self.collator = Collator(
            processor, ADAPTERS[self.spec.family], self.terminator,
            compute_dtype=torch.bfloat16, device=self.device,
        )
        embedder = CoordEmbedder(_hidden_size(model)).to(
            device=self.device, dtype=torch.float32,
        )
        embedder.enabled = False
        embedder.install(model)
        self.model = model
        self.processor = processor
        self.embedder = embedder
        logger.info("loaded %s hidden %d", ckpt, embedder.hidden_size)

    def examples(self) -> list[TrainingExample]:
        assert self.processor is not None
        cfg = TrainConfig(label="coord_probe", device=self.device)
        with tempfile.TemporaryDirectory(prefix="coord_probe_fence_") as tmp:
            image = Path(tmp) / "measure.png"
            Image.new("RGB", (32, 32), (18, 18, 18)).save(image)
            soft = measure_image_soft_tokens(self.processor, str(image))
        by_source = materialize(
            sources_from_manifest(), cfg,
            processor=self.processor, image_soft_tokens=soft,
        )
        held = _holdout(by_source, self.rng)
        logger.info("held-out examples %d", len(held))
        return held

    def _forward_hidden(self, model_inputs: dict[str, Any]) -> torch.Tensor:
        assert self.model is not None
        return _forward_last_hidden(self.model, model_inputs, logits_to_keep=1)

    def cache_control(self, held: list[TrainingExample]) -> None:
        """Five minutes of aug27 with no vector embedding."""
        assert self.collator is not None and self.embedder is not None
        self.embedder.enabled = False
        self.embedder.clear_pending()
        deadline = time.perf_counter() + PASS_S
        tokens = 0
        for example in held:
            if time.perf_counter() >= deadline:
                break
            built = self.collator.build(example)
            labels, hidden_index, _reply = _target_gather(
                built["model_inputs"]["input_ids"], built["weights"],
            )
            n = int(labels.shape[0])
            if tokens + n > CACHE_TOKEN_CAP:
                _report(
                    f"control cache token cap {CACHE_TOKEN_CAP} reached "
                    f"at {len(self.cached)} examples"
                )
                break
            with torch.inference_mode():
                hidden = self._forward_hidden(built["model_inputs"])
                seq_len = int(built["model_inputs"]["input_ids"].shape[1])
                if int(hidden.shape[1]) != seq_len:
                    raise RuntimeError(
                        f"control hidden length {int(hidden.shape[1])} != "
                        f"sequence {seq_len}; logits_to_keep sliced the states"
                    )
                gathered = hidden[0, hidden_index]
                logits = apply_lm_head_chunked(self.model, gathered, KD_CHUNK)
                ce = _token_ce(logits, labels)
            base = _draw_base(self.rng)
            self.cached.append(CachedExample(
                example=example,
                hidden=gathered.detach().to("cpu", dtype=torch.bfloat16),
                labels=labels.detach().to("cpu"),
                control_ce=ce,
                coords=_coords_for(self.rng, base, n),
                base=base,
            ))
            tokens += n
            del hidden, gathered, logits, built
        if not self.cached:
            raise RuntimeError("control pass stored no examples")
        _report(
            f"control cache: examples {len(self.cached)} target_tokens {tokens}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _volume_forward(self, row: CachedExample) -> tuple[float, float]:
        assert self.collator is not None and self.embedder is not None and self.model is not None
        built = self.collator.build(row.example)
        labels, hidden_index, reply_index = _target_gather(
            built["model_inputs"]["input_ids"], built["weights"],
        )
        if int(labels.shape[0]) != int(row.labels.shape[0]):
            raise RuntimeError(
                f"target length changed for {row.example.source}: "
                f"{int(labels.shape[0])} vs cached {int(row.labels.shape[0])}"
            )
        if not torch.equal(labels.cpu(), row.labels):
            raise RuntimeError(
                f"target ids changed for {row.example.source}; the control cache is stale"
            )
        coords = row.coords.to(self.device)
        positions = reply_index.view(1, -1)
        self.embedder.set_pending(coords.view(1, -1, 4), positions)
        try:
            with torch.inference_mode():
                hidden = self._forward_hidden(built["model_inputs"])
                seq_len = int(built["model_inputs"]["input_ids"].shape[1])
                if int(hidden.shape[1]) != seq_len:
                    raise RuntimeError(
                        f"volume hidden length {int(hidden.shape[1])} != "
                        f"sequence {seq_len}; logits_to_keep sliced the states"
                    )
                gathered = hidden[0, hidden_index]
                student = apply_lm_head_chunked(self.model, gathered, KD_CHUNK)
                # The cache is bf16. This head is float32, so score in the
                # live hidden's dtype. Storage stays bf16.
                teacher_h = row.hidden.to(device=self.device, dtype=gathered.dtype)
                teacher = apply_lm_head_chunked(self.model, teacher_h, KD_CHUNK)
                probe_ce = _token_ce(student, labels)
                kd = _soft_ce(teacher, student)
        finally:
            self.embedder.clear_pending()
        return probe_ce, kd

    def score_volume(self, name: str, magnitude: float) -> None:
        assert self.embedder is not None
        self.embedder.set_magnitude(magnitude)
        self.embedder.enabled = True
        deadline = time.perf_counter() + PASS_S
        scored = 0
        sum_probe = 0.0
        sum_control = 0.0
        sum_kd = 0.0
        for row in self.cached:
            remaining = deadline - time.perf_counter()
            if remaining <= GEN_RESERVE_S:
                break
            probe_ce, kd = self._volume_forward(row)
            sum_probe += probe_ce
            sum_control += row.control_ce
            sum_kd += kd
            scored += 1
        if scored == 0:
            raise RuntimeError(f"{name}: scored no examples")
        n = float(scored)
        probe_ce = sum_probe / n
        control_ce = sum_control / n
        _report(
            f"{name} tanh={magnitude:.2f} scored {scored}/{len(self.cached)} "
            f"probe_ce {probe_ce:.4f} control_ce {control_ce:.4f} "
            f"delta {probe_ce - control_ce:+.4f} kd {sum_kd / n:.4f}"
        )
        self._print_generations(name)

    def _print_generations(self, name: str) -> None:
        picks = self.cached[:2]
        for index, row in enumerate(picks):
            control = self._generate(row, use_embed=False)
            volume = self._generate(row, use_embed=True)
            question = _user_text(row.example)
            print(f"\n===== {name} example {index} source {row.example.source} =====", flush=True)
            print("--- question ---", flush=True)
            print(question, flush=True)
            print("--- control ---", flush=True)
            print(control, flush=True)
            print("--- volume ---", flush=True)
            print(volume, flush=True)
            logger.info(
                "%s example %d source %s\nquestion:\n%s\ncontrol:\n%s\nvolume:\n%s",
                name, index, row.example.source, question, control, volume,
            )

    def _generate(self, row: CachedExample, *, use_embed: bool) -> str:
        assert (
            self.model is not None and self.collator is not None
            and self.embedder is not None and self.processor is not None
        )
        tokenizer = self.collator.tokenizer
        norm = self.collator.adapter.prepare_messages(row.example.messages)
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
        self.model.config.use_cache = True
        self.embedder.enabled = use_embed
        produced: list[int] = []
        try:
            with torch.inference_mode():
                out = self.model(**inputs, use_cache=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :]
                seq_len = int(inputs["input_ids"].shape[1])
                for step in range(MAX_NEW_TOKENS):
                    next_id = int(logits.argmax(dim=-1).reshape(-1)[0].item())
                    if next_id == self.terminator:
                        break
                    produced.append(next_id)
                    step_ids = torch.tensor([[next_id]], device=self.device)
                    seq_len += 1
                    step_inputs: dict[str, Any] = {
                        "input_ids": step_ids,
                        "attention_mask": torch.ones(
                            1, seq_len, dtype=torch.long, device=self.device,
                        ),
                        "past_key_values": past,
                    }
                    if _wants_cache_position(self.model):
                        step_inputs["cache_position"] = torch.tensor(
                            [seq_len - 1], device=self.device,
                        )
                    if use_embed:
                        coord = _coords_for(self.rng, row.base, 1).to(self.device)
                        positions = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                        self.embedder.set_pending(coord.view(1, 1, 4), positions)
                    try:
                        out = self.model(**step_inputs, use_cache=True)
                    finally:
                        if use_embed:
                            self.embedder.clear_pending()
                    past = out.past_key_values
                    logits = out.logits[:, -1, :]
        finally:
            self.model.config.use_cache = False
            self.embedder.enabled = True
        if not produced:
            return ""
        return tokenizer.decode(produced, skip_special_tokens=True)

    def run(self) -> None:
        self.load()
        held = self.examples()
        logger.info("pass control (%d s)", int(PASS_S))
        self.cache_control(held)
        for name, magnitude in MAGNITUDES.items():
            logger.info("pass %s tanh=%.2f (%d s)", name, magnitude, int(PASS_S))
            self.score_volume(name, magnitude)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gemma-base", default="aug27_big_step_iter1_step313")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    configure_logging()
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    path = Path("logs") / f"coord_probe_{stamp}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
    ))
    logging.getLogger().addHandler(handler)
    logger.info("log %s", path)
    Probe(args).run()


if __name__ == "__main__":
    main()
