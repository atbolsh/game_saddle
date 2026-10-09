"""S1 and S2 together.

generate / batched_generate freeze one frame for S2 and decode until
eos, a reply that ends with ``[MOVE]``, or a reply that ends with
``[END_GAME]``. S1 stays dormant. A reply that closes ``[MOVE]`` ends
immediately; ``v`` is the Localizer output at the token that completed
``[MOVE]``, and ``run_s1_window`` steps the live game toward that
point for ``S1_WINDOW_SECONDS`` or ``S1_NOOP_STOP`` consecutive noops,
and also stops if the agent exits. ``[END_GAME]`` ends the reply and
does not wake S1.

Every token, every primitive, and the window's stop reason are
appended to a jsonl log. ``replay_frames`` rebuilds the window from
the recorded actions. The engine is deterministic, so no frames are
stored.

Equal prompt lengths share a prefill. Mixed lengths are separate
cohorts and are never left-padded (transformers#47651).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import torch
from PIL import Image

from agent.model import LONG_PREFILL_GPU_BATCH, LONG_PREFILL_TOKENS, stack_equal_length
from game.discreteEngine import discreteGame
from neural_net.oracle import ACTIONS
from neural_net.paths import weights_root
from neural_net.render import WallCache, apply_primitive, canonical_frame, pose
from neural_net.s1 import S1
from neural_net.s2 import GemmaS2

logger = logging.getLogger(__name__)

S1_WINDOW_SECONDS = 10.0
S1_NOOP_STOP = 20


@dataclass
class MoveRecord:
    """One S1 window. ``start_settings`` is the board before the first step."""

    start_settings: dict[str, Any]
    v: list[float]
    actions: list[str]
    collected: int
    exited: bool
    n_steps: int
    elapsed: float
    stopped_by: str


@dataclass
class GenerateResult:
    text: str
    n_tokens: int
    n_s1_steps: int
    log_path: str
    moved: bool
    ended: bool
    v: list[float] | None
    move: MoveRecord | None


class S1S2:
    """One actor and one decider. No forward method; step s1 and s2
    yourself if you want a custom loop. Both are public attributes.

    ``save`` writes a ``save_all`` directory plus ``s1.pt``. ``load``
    reads that directory back. Assembled checkpoints live under
    ``weights/full/``.
    """

    def __init__(
        self,
        s1: S1 | None = None,
        s2: GemmaS2 | None = None,
        *,
        model_key: str = "gemma-4-12b",
        checkpoint: str | None = None,
    ) -> None:
        self.s1 = s1 if s1 is not None else S1()
        self.s2 = s2 if s2 is not None else GemmaS2(model_key, checkpoint)

    def save(self, path: str | Path) -> Path:
        """Write this pair into ``path``: Gemma, the readout, and ``s1.pt``."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.s2.save_all(path)
        self.s1.save(path / "s1.pt")
        return path

    @classmethod
    def load(cls, path: str | Path, *, model_key: str = "gemma-4-12b") -> S1S2:
        """Load a directory written by :meth:`save`.

        The Gemma slice and the readout go through ``GemmaS2.load_all``.
        ``s1.pt`` is an ``S1`` state dict.
        """
        path = Path(path)
        s1_path = path / "s1.pt"
        if not s1_path.is_file():
            raise FileNotFoundError(f"S1S2.load: missing {s1_path}")
        s2 = GemmaS2(model_key)
        s2.load_all(path)
        s1 = S1()
        s1.load_weights(s1_path)
        return cls(s1=s1, s2=s2)

    def generate(
        self,
        game: discreteGame,
        prompt: str | list[dict],
        *,
        max_new_tokens: int | None = None,
        log_path: str | Path | None = None,
        on_token=None,
    ) -> GenerateResult:
        return self.batched_generate(
            [game],
            [prompt],
            max_new_tokens=max_new_tokens,
            log_path=log_path,
            on_token=on_token,
        )[0]

    def batched_generate(
        self,
        games: list[discreteGame],
        prompts: list[str | list[dict]],
        *,
        max_new_tokens: int | None = None,
        log_path: str | Path | None = None,
        on_token=None,
    ) -> list[GenerateResult]:
        if len(games) != len(prompts):
            raise ValueError(
                f"batched_generate: {len(games)} games vs {len(prompts)} prompts"
            )
        if not games:
            return []
        self._ensure_loaded()
        log_file = _open_log(log_path)
        try:
            encoded = [
                self.s2.vl.encode_messages(_messages(prompt, _frozen_image(game)))
                for game, prompt in zip(games, prompts)
            ]
            lens = [int(row["input_ids"].shape[1]) for row in encoded]
            results: list[GenerateResult | None] = [None] * len(games)
            for cohort in _cohorts(lens):
                sub = self._run_cohort(
                    [games[i] for i in cohort],
                    stack_equal_length([encoded[i] for i in cohort]),
                    max_new_tokens=max_new_tokens,
                    log_file=log_file,
                    row_ids=cohort,
                    on_token=on_token,
                )
                for index, result in zip(cohort, sub):
                    results[index] = result
            if any(result is None for result in results):
                raise RuntimeError("batched_generate: a cohort produced no result")
            return list(results)  # type: ignore[arg-type]
        finally:
            log_file.close()

    def _ensure_loaded(self) -> None:
        if self.s2.vl.model is None or self.s2.localizer is None:
            self.s2.load()
        device = next(self.s2.vl.model.parameters()).device
        self.s1.to(device)
        self.s1.eval()

    def _run_cohort(
        self,
        games: list[discreteGame],
        encoded: dict[str, Any],
        *,
        max_new_tokens: int | None,
        log_file,
        row_ids: list[int],
        on_token=None,
    ) -> list[GenerateResult]:
        from agent.config import CONFIG
        from agent.modes import ends_with_end_game, ends_with_move

        limit = CONFIG.max_new_tokens if max_new_tokens is None else max_new_tokens
        if limit < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {limit}")
        sampling = self.s2.vl._sampling_kwargs()
        eos = _eos_ids(self.s2.vl)
        tokenizer = getattr(self.s2.vl.processor, "tokenizer", self.s2.vl.processor)
        device = next(self.s2.vl.model.parameters()).device

        batch = len(games)
        prompt_len = int(encoded["input_ids"].shape[1])
        logger.info("S1S2 cohort rows=%s prompt_tokens=%d", row_ids, prompt_len)

        logits, prev_coords, past = self.s2.prefill(encoded)
        seq_len = prompt_len
        generated: list[list[int]] = [[] for _ in range(batch)]
        finished = [False] * batch
        moved = [False] * batch
        ended = [False] * batch
        move_v: list[list[float] | None] = [None] * batch
        records: list[MoveRecord | None] = [None] * batch
        log_path = log_file.name

        def emit(record: dict) -> None:
            line = json.dumps(record, separators=(",", ":")) + "\n"
            log_file.write(line)
            log_file.flush()
            os.fsync(log_file.fileno())

        eos_feed = min(eos)
        for token_index in range(limit):
            token_ids = _sample(logits, sampling)
            include = [not flag for flag in finished]
            for row in range(batch):
                if not include[row]:
                    token_ids[row] = eos_feed
                    continue
                generated[row].append(int(token_ids[row]))
            logits, coords, past = self.s2.decode(
                token_ids, past, seq_len, prev_coords=prev_coords,
            )
            seq_len += 1
            prev_coords = coords
            for row in range(batch):
                if not include[row]:
                    continue
                tid = generated[row][-1]
                text = tokenizer.decode(generated[row], skip_special_tokens=True)
                s_xy, v_xy = _split_coords(coords[row])
                move_close = ends_with_move(text)
                end_close = ends_with_end_game(text)
                emit({
                    "kind": "token",
                    "row": row_ids[row],
                    "token_index": token_index,
                    "token_id": tid,
                    "piece": tokenizer.decode([tid], skip_special_tokens=False),
                    "s": s_xy,
                    "v": v_xy,
                    "move_close": move_close,
                    "end_close": end_close,
                    "eos": tid in eos,
                })
                if on_token is not None:
                    on_token({
                        "row": row_ids[row],
                        "piece": tokenizer.decode([tid], skip_special_tokens=False),
                        "s": s_xy,
                        "v": v_xy,
                        "eos": tid in eos,
                        "move_close": move_close,
                        "end_close": end_close,
                    })
                if move_close:
                    finished[row] = True
                    moved[row] = True
                    move_v[row] = v_xy
                elif end_close:
                    finished[row] = True
                    ended[row] = True
                elif tid in eos:
                    finished[row] = True
            if all(finished):
                break

        for row in range(batch):
            if not moved[row] or move_v[row] is None:
                continue
            records[row] = run_s1_window(
                self.s1,
                games[row],
                move_v[row],
                device=device,
                emit=emit,
                row_id=row_ids[row],
            )

        return [
            GenerateResult(
                text=tokenizer.decode(
                    generated[row], skip_special_tokens=True
                ).strip(),
                n_tokens=len(generated[row]),
                n_s1_steps=0 if records[row] is None else records[row].n_steps,
                log_path=log_path,
                moved=moved[row],
                ended=ended[row],
                v=move_v[row],
                move=records[row],
            )
            for row in range(batch)
        ]


def run_s1_window(
    s1: S1,
    game: discreteGame,
    v: list[float] | tuple[float, float],
    *,
    device: torch.device,
    seconds: float = S1_WINDOW_SECONDS,
    noop_stop: int = S1_NOOP_STOP,
    emit=None,
    row_id: int = 0,
) -> MoveRecord:
    """Step ``game`` toward ``v`` until time, a noop streak, or exit.

    The clock is wall time. The noop counter resets on any other action.
    ``stopped_by`` is ``time``, ``noops``, or ``exit``.
    """
    from agent.game_io import game_to_settings_dict

    point = [float(v[0]), float(v[1])]
    if len(point) != 2:
        raise ValueError(f"v must be two numbers, got {len(v)}")
    if seconds < 0:
        raise ValueError(f"seconds must be >= 0, got {seconds}")
    if noop_stop < 1:
        raise ValueError(f"noop_stop must be >= 1, got {noop_stop}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    s1.to(device)
    s1.eval()
    start_settings = game_to_settings_dict(game)
    cache = WallCache()
    xy = torch.tensor([point], dtype=torch.float32, device=device)
    actions: list[str] = []
    collected_total = 0
    noop_run = 0
    exited = False
    stopped = "time"
    started = time.perf_counter()
    while True:
        if time.perf_counter() - started >= seconds:
            stopped = "time"
            break
        if noop_run >= noop_stop:
            stopped = "noops"
            break
        images = _frames_tensor([cache.render(game)], device)
        with torch.inference_mode():
            logits = s1(images, xy)
        action = ACTIONS[int(torch.argmax(logits, dim=-1).item())]
        before = pose(game)
        collected = int(apply_primitive(game, action))
        after = pose(game)
        exited = bool(game.agent_exited())
        actions.append(action)
        collected_total += collected
        if action == "noop":
            noop_run += 1
        else:
            noop_run = 0
        if emit is not None:
            emit({
                "kind": "move",
                "row": row_id,
                "step": len(actions) - 1,
                "action": action,
                "v": [round(item, 6) for item in point],
                "collected": collected,
                "exited": exited,
                "before": before,
                "after": after,
            })
        if exited:
            stopped = "exit"
            break
        if noop_run >= noop_stop:
            stopped = "noops"
            break
    elapsed = time.perf_counter() - started
    record = MoveRecord(
        start_settings=start_settings,
        v=[round(item, 6) for item in point],
        actions=actions,
        collected=collected_total,
        exited=exited,
        n_steps=len(actions),
        elapsed=elapsed,
        stopped_by=stopped,
    )
    if emit is not None:
        emit({
            "kind": "window_end",
            "row": row_id,
            "stopped_by": record.stopped_by,
            "n_steps": record.n_steps,
            "elapsed": record.elapsed,
            "collected": record.collected,
            "exited": record.exited,
        })
    return record


def replay_frames(record: MoveRecord) -> Iterator[np.ndarray]:
    """Rebuild the window. Yields the start frame, then one frame per action."""
    from agent.game_io import game_from_settings_dict

    game = game_from_settings_dict(record.start_settings)
    cache = WallCache()
    yield cache.render(game)
    for action in record.actions:
        apply_primitive(game, action)
        yield cache.render(game)


def _open_log(log_path: str | Path | None):
    if log_path is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = weights_root() / "s1s2_runs" / stamp / "moves.jsonl"
    else:
        path = Path(log_path)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "a", encoding="utf-8")


def _split_coords(row: torch.Tensor) -> tuple[list[float], list[float]]:
    values = [round(float(v), 6) for v in row.detach().float().cpu()]
    if len(values) != 4:
        raise RuntimeError(
            f"coordinate head returned {len(values)} values, expected 4 (s, v)"
        )
    return values[0:2], values[2:4]


def _frames_tensor(frames: list[np.ndarray], device: torch.device) -> torch.Tensor:
    stacked = np.ascontiguousarray(np.stack(frames, axis=0))
    tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()
    return tensor.to(device, non_blocking=device.type == "cuda")


def _cohorts(lens: list[int]) -> Iterator[list[int]]:
    """Equal token-length groups. Longs are chunked to the VRAM cap.

    No left pad, so the #47651 nudge is not applied: mixed lengths never
    share a prefill, and each cohort is stacked with ``stack_equal_length``.
    """
    by_len: dict[int, list[int]] = {}
    for index, length in enumerate(lens):
        by_len.setdefault(length, []).append(index)
    for length, indices in by_len.items():
        cap = LONG_PREFILL_GPU_BATCH if length >= LONG_PREFILL_TOKENS else len(indices)
        for start in range(0, len(indices), cap):
            yield indices[start:start + cap]


def _messages(prompt: str | list[dict], image: Image.Image) -> list[dict]:
    """Prompt text first, frozen frame last.

    A string becomes one user turn. A message list is copied and the
    image is appended to its last turn. No words are added.
    """
    image_part = {"type": "image", "image": image}
    if isinstance(prompt, str):
        return [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}, image_part],
        }]
    if not isinstance(prompt, list) or not prompt:
        raise TypeError(
            "prompt must be a string or a non-empty list of chat messages, "
            f"got {type(prompt).__name__}"
        )
    messages = copy.deepcopy(prompt)
    last = messages[-1]
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = [{"type": "text", "text": content}, image_part]
    elif isinstance(content, list):
        if not any(
            isinstance(part, dict) and part.get("type") == "image"
            for part in content
        ):
            content.append(image_part)
    else:
        raise TypeError(
            "the last message content must be a string or a list of parts, "
            f"got {type(content).__name__}"
        )
    return messages


def _frozen_image(game: discreteGame) -> Image.Image:
    return Image.fromarray(canonical_frame(game), mode="RGB")


def _eos_ids(vl) -> set[int]:
    eos = getattr(vl.model.generation_config, "eos_token_id", None)
    if eos is None:
        tokenizer = getattr(vl.processor, "tokenizer", vl.processor)
        eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        raise RuntimeError(
            "Gemma has no eos_token_id on the generation config or the tokenizer"
        )
    if isinstance(eos, int):
        return {eos}
    return {int(item) for item in eos}


def _sample(logits: torch.Tensor, sampling: dict) -> torch.Tensor:
    if not sampling.get("do_sample", True):
        return torch.argmax(logits, dim=-1)
    chosen = [
        _sample_row(logits[row], sampling) for row in range(logits.shape[0])
    ]
    return torch.tensor(chosen, dtype=torch.long, device=logits.device)


def _sample_row(logits: torch.Tensor, sampling: dict) -> int:
    scores = logits.float()
    temperature = sampling.get("temperature", None)
    if temperature is not None and float(temperature) != 1.0:
        scores = scores / float(temperature)
    top_k = sampling.get("top_k")
    if top_k:
        k = min(int(top_k), int(scores.shape[-1]))
        cutoff = torch.topk(scores, k).values[-1]
        scores = scores.masked_fill(scores < cutoff, float("-inf"))
    top_p = sampling.get("top_p")
    if top_p is not None and float(top_p) < 1.0:
        sorted_scores, sorted_idx = torch.sort(scores, descending=True)
        cumulative = sorted_scores.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
        scores = torch.full_like(scores, float("-inf"))
        scores.scatter_(0, sorted_idx, sorted_scores)
    probs = scores.softmax(dim=-1)
    if not torch.isfinite(probs).all():
        raise RuntimeError("token sampling produced non-finite probabilities")
    return int(torch.multinomial(probs, 1))

