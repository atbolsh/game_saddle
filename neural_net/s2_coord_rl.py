"""Archived online-RL task for the coordinate embedder.

Off by default. ``--rl`` on the teacher-force trainer runs this sampler
instead of the sixteen-reply cache. The anchor coin stays in the trainer.

Replies are sampled, not teacher-forced. A reward of 0 is skipped. One
quarter of the slots are teacher-forced on the correct line so the format
has a gradient before the policy emits it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from agent.memory import format_notepad
from agent.modes import SYSTEM_PROMPT_S2, _build_game_messages
from neural_net.render import canonical_frame
from neural_net.s2 import _wants_cache_position
from neural_net.s2_coord_oracle import (
    KIND_HOUR,
    KIND_LEFTRIGHT,
    KIND_LOOKING,
    KIND_MOVING,
    KIND_UPDOWN,
    KINDS,
    clock_hour,
    correct_line,
    direction_label,
    gaze_vbar,
    hour_is_unique,
    parse_reply,
    reply_reward,
)
from PIL import Image
from training.train import TrainingExample, weighted_loss

BOOTSTRAP = 0.25
MAX_NEW_TOKENS = 24
V_TRIES = 400
COORD_LOW = -1.0
COORD_HIGH = 2.0
NOTEPAD = [{"key": "target", "value": "the upper-left gold", "updated_round": 1}]


def question(kind: str) -> str:
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


def sample_v(trainer: Any, game: Any) -> tuple[float, float]:
    for _ in range(V_TRIES):
        vx = trainer.rng.random()
        vy = trainer.rng.random()
        if game.full_wall_check(vx, vy, agent_r=1e-3):
            return vx, vy
    raise RuntimeError(f"no v outside a wall after {V_TRIES} draws")


def scenario(trainer: Any) -> dict[str, Any] | None:
    """One labeled question, or None when the draw was ambiguous."""
    assert trainer.tmp is not None
    game, agent = trainer._sample_board()
    kind = trainer.rng.choice(KINDS)
    mode = "look" if trainer.rng.random() < 0.5 else "move"
    sx, sy = trainer._sample_s(agent)
    vx, vy = sample_v(trainer, game)
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
    looking = mode != "move"
    vbx, vby = gaze_vbar(sx, sy, vx, vy, looking=looking)
    if not (COORD_LOW <= vbx <= COORD_HIGH and COORD_LOW <= vby <= COORD_HIGH):
        raise RuntimeError(
            f"v_bar {(vbx, vby)} outside [-1, 2] for s {(sx, sy)} v {(vx, vy)}"
        )
    frame = canonical_frame(game)
    path = trainer.tmp / f"rl_{trainer.step}_{time.time_ns()}.png"
    Image.fromarray(frame).save(path)
    try:
        notepad = format_notepad(NOTEPAD) if trainer.rng.random() < 0.5 else None
        messages = _build_game_messages(
            SYSTEM_PROMPT_S2, str(path), "", question(kind), notepad=notepad,
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


def generate(
    trainer: Any, drawn: dict[str, Any],
) -> tuple[str, list[int], list[list[float]]]:
    assert trainer.model is not None and trainer.processor is not None
    assert trainer.embedder is not None and trainer.collator is not None
    tokenizer = trainer.collator.tokenizer
    norm = trainer.collator.adapter.prepare_messages(drawn["messages"])
    prompt = trainer.processor.apply_chat_template(
        norm, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs: dict[str, Any] = {}
    for key, val in prompt.items():
        if not isinstance(val, torch.Tensor):
            continue
        if val.dtype.is_floating_point:
            val = val.to(torch.bfloat16)
        inputs[key] = val.to(trainer.device)
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    produced: list[int] = []
    rows: list[list[float]] = []
    prompt_len = int(inputs["input_ids"].shape[1])
    trainer._heartbeat("rl-generate", drawn["kind"], 1, prompt_len)
    trainer._use_cache(True)
    try:
        with torch.inference_mode():
            out = trainer.model(**inputs, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            seq_len = prompt_len
            for _step in range(MAX_NEW_TOKENS):
                probs = torch.softmax(logits.float(), dim=-1)
                next_id = int(torch.multinomial(probs.reshape(-1), 1).item())
                if next_id == trainer.terminator:
                    break
                produced.append(next_id)
                row = trainer._jitter_rows(drawn["points"], drawn["agent"], 1)[0]
                rows.append(row)
                seq_len += 1
                step_inputs: dict[str, Any] = {
                    "input_ids": torch.tensor([[next_id]], device=trainer.device),
                    "attention_mask": torch.ones(
                        1, seq_len, dtype=torch.long, device=trainer.device,
                    ),
                    "past_key_values": past,
                }
                if _wants_cache_position(trainer.model):
                    step_inputs["cache_position"] = torch.tensor(
                        [seq_len - 1], device=trainer.device,
                    )
                coord = torch.tensor(row, dtype=torch.float32, device=trainer.device)
                positions = torch.zeros(1, 1, dtype=torch.long, device=trainer.device)
                trainer.embedder.set_pending(coord.view(1, 1, 6), positions)
                try:
                    out = trainer.model(**step_inputs, use_cache=True)
                finally:
                    trainer.embedder.clear_pending()
                past = out.past_key_values
                logits = out.logits[:, -1, :]
    finally:
        trainer._use_cache(False)
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


def rl_loss(trainer: Any) -> tuple[torch.Tensor | None, Path]:
    """One sampled reply. Reward 0 returns ``(None, path)`` and is skipped."""
    assert trainer.collator is not None and trainer.model is not None
    assert trainer.vram is not None
    drawn = None
    for _ in range(50):
        drawn = scenario(trainer)
        if drawn is not None:
            break
    if drawn is None:
        raise RuntimeError("50 board draws had no unambiguous label")
    path: Path = drawn["path"]
    try:
        trainer.vram.set_stage("rl")
        bootstrap = trainer.rng.random() < BOOTSTRAP
        stored: list[list[float]] | None
        if bootstrap:
            text = drawn["line"]
            weight = 1.0
            stored = None
        else:
            text, _ids, stored = generate(trainer, drawn)
            reward = reply_reward(drawn["kind"], text, drawn["expected"])
            parsed = parse_reply(drawn["kind"], text) is not None
            trainer._rl_rewards.append(reward)
            trainer._rl_parsed.append(parsed)
            trainer._rl_rewards = trainer._rl_rewards[-50:]
            trainer._rl_parsed = trainer._rl_parsed[-50:]
            weight = reward
        if weight == 0.0:
            return None, path
        example = TrainingExample(
            messages=drawn["messages"],
            target_text=text,
            loss="ce",
            source="coord_rl",
            example_weight=weight,
        )
        built = trainer.collator.build(example)
        ids = built["model_inputs"]["input_ids"]
        trainer._heartbeat(
            "rl", drawn["kind"], int(ids.shape[0]), int(ids.shape[1]),
        )
        n = int((built["weights"][0] != 0).sum())
        if stored is None:
            rows = trainer._jitter_rows(drawn["points"], drawn["agent"], n)
        else:
            if n != len(stored) + 1:
                raise RuntimeError(
                    f"reply tokens {n} != generated {len(stored)} plus the terminator"
                )
            rows = stored + trainer._jitter_rows(drawn["points"], drawn["agent"], 1)
        trainer._arm(built["weights"], rows)
        loss = weighted_loss(
            trainer.model, built["model_inputs"], built["weights"],
            loss_kind="ce", example_weight=built["example_weight"],
        )
        return loss, path
    except Exception:
        path.unlink(missing_ok=True)
        raise
