"""S1 and S2 together.

generate / batched_generate freeze one frame for S2, prefill it at the
end of the prompt, and then race: S2 decodes the next token on one CUDA
stream while S1 steps the live game on another, until that forward
returns. s1_steps_per_token caps an interval. The default (None) is
as many primitives as finish before the next token.

After the last token there is no next forward to wait on, so S1 keeps
the last target until it emits noop, the agent exits, the cap (if set)
hits, or S1_TAIL_SAFETY steps. Every move and every interval count is
appended to a jsonl log.

Equal prompt lengths share a prefill. Mixed lengths are separate cohorts
and are never left-padded (transformers#47651).

Learn-to-look pretraining does not use this loop. That run stops a reply
on eos and does not emit [HOLD] or [RELOAD].
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
import threading
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

# Tail interval only. Overlapped intervals stop when the next token's
# forward returns (or at s1_steps_per_token). The tail has no next token,
# so this bounds a policy that never emits noop.
S1_TAIL_SAFETY = 256


@dataclass
class GenerateResult:
    text: str
    n_tokens: int
    n_s1_steps: int
    log_path: str


class S1S2:
    """One actor and one decider. No forward method; step s1 and s2
    yourself if you want a custom loop. Both are public attributes."""

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

    def generate(
        self,
        game: discreteGame,
        prompt: str | list[dict],
        *,
        max_new_tokens: int | None = None,
        s1_steps_per_token: int | None = None,
        log_path: str | Path | None = None,
    ) -> GenerateResult:
        return self.batched_generate(
            [game],
            [prompt],
            max_new_tokens=max_new_tokens,
            s1_steps_per_token=s1_steps_per_token,
            log_path=log_path,
        )[0]

    def batched_generate(
        self,
        games: list[discreteGame],
        prompts: list[str | list[dict]],
        *,
        max_new_tokens: int | None = None,
        s1_steps_per_token: int | None = None,
        log_path: str | Path | None = None,
    ) -> list[GenerateResult]:
        if len(games) != len(prompts):
            raise ValueError(
                f"batched_generate: {len(games)} games vs {len(prompts)} prompts"
            )
        if not games:
            return []
        if s1_steps_per_token is not None and s1_steps_per_token < 1:
            raise ValueError(
                "s1_steps_per_token must be >= 1 or None, "
                f"got {s1_steps_per_token}"
            )
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
                    s1_steps_per_token=s1_steps_per_token,
                    log_file=log_file,
                    row_ids=cohort,
                )
                for index, result in zip(cohort, sub):
                    results[index] = result
            if any(result is None for result in results):
                raise RuntimeError("batched_generate: a cohort produced no result")
            return list(results)  # type: ignore[arg-type]
        finally:
            log_file.close()

    def _ensure_loaded(self) -> None:
        if self.s2.vl.model is None or self.s2.coord_head is None:
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
        s1_steps_per_token: int | None,
        log_file,
        row_ids: list[int],
    ) -> list[GenerateResult]:
        from agent.config import CONFIG

        limit = CONFIG.max_new_tokens if max_new_tokens is None else max_new_tokens
        if limit < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {limit}")
        sampling = self.s2.vl._sampling_kwargs()
        eos = _eos_ids(self.s2.vl)
        tokenizer = getattr(self.s2.vl.processor, "tokenizer", self.s2.vl.processor)
        device = next(self.s2.vl.model.parameters()).device
        s1_stream, s2_stream = _streams(device)

        batch = len(games)
        prompt_len = int(encoded["input_ids"].shape[1])
        logger.info("S1S2 cohort rows=%s prompt_tokens=%d", row_ids, prompt_len)

        with _use_stream(s2_stream):
            logits, coords, past = self.s2.prefill(encoded)
        _sync(s2_stream)

        seq_len = prompt_len
        generated: list[list[int]] = [[] for _ in range(batch)]
        finished = [False] * batch
        step_counts = [0] * batch
        log_lock = threading.Lock()
        log_path = log_file.name

        def emit(record: dict) -> None:
            line = json.dumps(record, separators=(",", ":")) + "\n"
            with log_lock:
                log_file.write(line)
                log_file.flush()
                os.fsync(log_file.fileno())

        for token_index in range(limit):
            token_ids = _sample(logits, sampling)
            include = [not flag for flag in finished]
            eos_feed = min(eos)
            for row in range(batch):
                if not include[row]:
                    token_ids[row] = eos_feed
                    continue
                tid = int(token_ids[row])
                generated[row].append(tid)
                if tid in eos:
                    finished[row] = True
                s_xy, v_xy, vbar_xy, target = _split_coords(coords[row])
                emit({
                    "kind": "token",
                    "row": row_ids[row],
                    "token_index": token_index,
                    "token_id": tid,
                    "piece": tokenizer.decode([tid], skip_special_tokens=False),
                    "s": s_xy,
                    "v": v_xy,
                    "v_bar": vbar_xy,
                    "target": target,
                    "eos": tid in eos,
                })

            snapshot = coords.detach().float().cpu().clone()
            last = (token_index + 1 == limit) or all(finished)
            if last:
                n_steps = self._s1_interval(
                    games, snapshot, stop=None,
                    cap=_tail_cap(s1_steps_per_token), until_settled=True,
                    token_index=token_index, row_ids=row_ids, emit=emit,
                    device=device, stream=s1_stream, step_counts=step_counts,
                    include=include,
                )
                emit({
                    "kind": "interval_end",
                    "token_index": token_index,
                    "rows": row_ids,
                    "n_steps": n_steps,
                    "tail": True,
                })
                break

            stop = threading.Event()
            error: dict[str, BaseException] = {}
            holder = {"n": 0}

            def _worker(
                snap: torch.Tensor = snapshot,
                stop_event: threading.Event = stop,
                index: int = token_index,
                include_rows: list[bool] = include,
            ) -> None:
                try:
                    holder["n"] = self._s1_interval(
                        games, snap, stop=stop_event, cap=s1_steps_per_token,
                        until_settled=False, token_index=index, row_ids=row_ids,
                        emit=emit, device=device, stream=s1_stream,
                        step_counts=step_counts, include=include_rows,
                    )
                except BaseException as exc:  # re-raised on the main thread
                    error["exc"] = exc
                    stop_event.set()

            worker = threading.Thread(target=_worker, name="s1-interval")
            worker.start()
            decode_exc: BaseException | None = None
            try:
                with _use_stream(s2_stream):
                    logits, coords, past = self.s2.decode(token_ids, past, seq_len)
            except BaseException as exc:
                decode_exc = exc
            finally:
                _sync(s2_stream)
                stop.set()
                worker.join()
            if decode_exc is not None and "exc" in error:
                raise decode_exc from error["exc"]
            if decode_exc is not None:
                raise decode_exc
            if "exc" in error:
                raise error["exc"]
            seq_len += 1
            emit({
                "kind": "interval_end",
                "token_index": token_index,
                "rows": row_ids,
                "n_steps": holder["n"],
                "tail": False,
            })

        return [
            GenerateResult(
                text=tokenizer.decode(
                    generated[row], skip_special_tokens=True
                ).strip(),
                n_tokens=len(generated[row]),
                n_s1_steps=step_counts[row],
                log_path=log_path,
            )
            for row in range(batch)
        ]

    def _s1_interval(
        self,
        games: list[discreteGame],
        coords: torch.Tensor,
        *,
        stop: threading.Event | None,
        cap: int | None,
        until_settled: bool,
        token_index: int,
        row_ids: list[int],
        emit,
        device: torch.device,
        stream,
        step_counts: list[int],
        include: list[bool],
    ) -> int:
        """Step still-active games until stop, cap, or (tail only) every
        active row has emitted noop or exited.

        coords is [B, 6] for the token that opened this interval. Returns
        how many S1 forwards ran.
        """
        if device.type == "cuda":
            torch.cuda.set_device(device)
        active = list(include)
        caches = [WallCache() for _ in games]
        target = (coords[:, 2:4] + coords[:, 4:6]) / 2
        n_forwards = 0
        while True:
            if stop is not None and stop.is_set():
                break
            if cap is not None and n_forwards >= cap:
                break
            rows = [i for i, on in enumerate(active) if on]
            if not rows:
                break
            frames = [caches[i].render(games[i]) for i in rows]
            xy = target[rows].to(device, non_blocking=device.type == "cuda")
            images = _frames_tensor(frames, device)
            with _use_stream(stream):
                with torch.inference_mode():
                    logits = self.s1(images, xy)
            _sync(stream)
            choice = torch.argmax(logits, dim=-1).tolist()
            settled = True
            for local, row in enumerate(rows):
                action = ACTIONS[int(choice[local])]
                before = pose(games[row])
                collected = apply_primitive(games[row], action)
                after = pose(games[row])
                exited = bool(games[row].agent_exited())
                chased = [
                    round(float(v), 6) for v in target[row].tolist()
                ]
                emit({
                    "kind": "move",
                    "row": row_ids[row],
                    "token_index": token_index,
                    "step": n_forwards,
                    "action": action,
                    "target": chased,
                    "collected": collected,
                    "exited": exited,
                    "before": before,
                    "after": after,
                })
                step_counts[row] += 1
                if exited:
                    active[row] = False
                if not until_settled or action != "noop":
                    settled = False
            n_forwards += 1
            if until_settled and settled:
                break
        return n_forwards


def _tail_cap(s1_steps_per_token: int | None) -> int:
    if s1_steps_per_token is None:
        return S1_TAIL_SAFETY
    return s1_steps_per_token


def _open_log(log_path: str | Path | None):
    if log_path is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = weights_root() / "s1s2_runs" / stamp / "moves.jsonl"
    else:
        path = Path(log_path)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "a", encoding="utf-8")


def _split_coords(row: torch.Tensor):
    values = [round(float(v), 6) for v in row.detach().float().cpu()]
    if len(values) != 6:
        raise RuntimeError(
            f"coordinate head returned {len(values)} values, expected 6"
        )
    s_xy, v_xy, vbar_xy = values[0:2], values[2:4], values[4:6]
    target = [
        round((v_xy[0] + vbar_xy[0]) / 2, 6),
        round((v_xy[1] + vbar_xy[1]) / 2, 6),
    ]
    return s_xy, v_xy, vbar_xy, target


def _frames_tensor(frames: list[np.ndarray], device: torch.device) -> torch.Tensor:
    stacked = np.ascontiguousarray(np.stack(frames, axis=0))
    tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()
    return tensor.to(device, non_blocking=device.type == "cuda")


def _streams(device: torch.device):
    if device.type != "cuda":
        return None, None
    return torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)


def _use_stream(stream):
    if stream is None:
        return contextlib.nullcontext()
    return torch.cuda.stream(stream)


def _sync(stream) -> None:
    if stream is not None:
        stream.synchronize()


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

