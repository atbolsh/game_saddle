"""Greedy eval for the weekend joint S2 run, and the boards it trains on.

The families, their shares, and the shift family's knobs are listed in
:mod:`neural_net.s2_families`; this module holds the builders and the
eval functions that list points at. Training and eval share the
builders. A generated reply is scored here, not by teacher-forced CE,
except the anchor holdout. The reasoning slice is greedy exact match on
the first 32 GSM8K test-split probe items.

:mod:`neural_net.s2_families` imports this module, so anything here that
needs the list reads it through :func:`_families` at call time.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image

from agent import game_io
from agent.config import CONFIG
from agent.memory import format_notepad, format_recent_block
from agent.modes import (
    S2_BEGINNING_MOVE_USER,
    S2_BEGINNING_USER,
    S2_END_GAME_USER,
    S2_END_REPLY,
    S2_END_TOKEN,
    S2_LOOK_TARGET_USER,
    S2_MOVE_TARGET_USER,
    S2_MOVE_VERDICT_LINES,
    S2_SEALED_NOTE,
    SYSTEM_PROMPT_S2,
    _S2_LOOK_LINES,
    _S2_MOVE_LINES,
    _build_game_messages,
    ends_with_end_game,
    ends_with_move,
    s2_move_reply,
)
from neural_net.render import canonical_frame
from neural_net.s2 import decode_step, prefill_step
from neural_net.s2_coord_oracle import (
    KIND_HOUR,
    KIND_LEFTRIGHT,
    KIND_UPDOWN,
    clock_hour,
    correct_line,
    direction_label,
    hour_is_unique,
    hour_reward,
    parse_reply,
)
from neural_net.s2_coord_train import (
    BOARD_TRIES,
    QUESTION_DIRECTION,
    QUESTION_HOUR,
)
from neural_net.s2_pretraining_learn_to_look import (
    _ask,
    _coords,
    _draw_target,
    _targets_on_board,
    beginning_messages,
    board_context,
    draw_question,
    draw_questions,
    gold_region_phrases,
)
from neural_net.s2_pretraining_localizer import _begin_reply, _begin_targets
from training.external_data import load_manifest
from training.probes import answers_match, extract_answer
from training.train import weighted_loss

logger = logging.getLogger("s2_weekend")

C_WEIGHTS = {"gold": 0.40, "exit": 0.35, "corner": 0.25}
HOLDOUT_EVAL = 32
MAX_NEW_TOKENS = 64
REASONING_DATASET = "gsm8k"
REASONING_N = 32
REASONING_MAX_NEW = 192

SHIFT_HIT = 0.025            # a shift counts when the gaze lands within this of v
SHIFT_EVAL_PER_CASE = 8      # eval boards per shift sub-case

_REMEMBER = re.compile(r"\[REMEMBER target: ([^\]]*)\]")


def _families():
    """The family list, imported at call time (it imports this module)."""
    from neural_net import s2_families

    return s2_families


class Redraw(Exception):
    def __init__(self, path: Path | None = None) -> None:
        super().__init__("redraw")
        self.path = path


@dataclass
class Turn:
    question: str
    reply: str
    s: tuple[float, float]
    v: tuple[float, float]
    agent: tuple[float, float]
    all_mid: bool
    expect_move: bool
    task: str
    extra: dict[str, Any] = field(default_factory=dict)
    # One ``v`` per reply line when the gaze moves inside the reply (the
    # shift family's "Look at X. Now, look a little to the left." answers
    # with two look lines: the first on X, the second on X + d). ``v`` is
    # then the last entry. None: every reply token carries ``v``.
    v_lines: list[tuple[float, float]] | None = None


@dataclass
class Batch:
    family: str
    path: Path
    context: str
    notepad: str | None
    settings: dict[str, Any]
    turns: list[Turn]
    inject: str


def messages_for(batch: Batch, turn: Turn) -> list[dict]:
    """The user turn the collator and the greedy loop both encode."""
    if turn.question in (S2_BEGINNING_USER, S2_BEGINNING_MOVE_USER):
        kind = "look" if turn.question == S2_BEGINNING_USER else "move"
        notepad = batch.notepad if batch.notepad is not None else format_notepad([])
        return beginning_messages(str(batch.path), notepad, kind)
    return _build_game_messages(
        SYSTEM_PROMPT_S2,
        str(batch.path),
        batch.context,
        turn.question,
        notepad=batch.notepad,
    )


def build_shift(
    rng: random.Random,
    tmp: Path,
    step: int,
    case: str | None,
) -> Batch:
    """Family G: "look a little to the left" moves the gaze from the place
    it is on. ``case`` is a key of ``s2_families.SHIFT_CASES`` or None for
    the weighted draw."""
    fam = _families()
    for _attempt in range(BOARD_TRIES):
        path: Path | None = None
        try:
            path, settings = _deal(rng, tmp, step, "c")
            which = case if case is not None else _weighted_key(rng, fam.SHIFT_CASES)
            return _shift_batch(rng, path, settings, which)
        except Redraw as exc:
            _drop(exc.path if exc.path is not None else path)
            continue
        except Exception:
            _drop(path)
            raise
    raise RuntimeError(f"shift drew no board in {BOARD_TRIES} tries")


def build_gaze(
    rng: random.Random,
    place_image: Callable[[], dict[str, Any]],
    mode: str,
) -> Batch:
    """Hour on every placement and direction on two, or one placement and both."""
    prepared = place_image()
    path = Path(prepared["path"])
    try:
        specs = list(prepared["specs"])
        if mode == "one":
            specs_used = [rng.choice(specs)]
            want_dir = set(range(len(specs_used)))
        elif mode == "train":
            specs_used = specs
            want_dir = set(rng.sample(range(len(specs_used)), 2))
        else:
            raise ValueError(f"gaze mode {mode!r}")
        turns: list[Turn] = []
        placements = []
        for index, spec in enumerate(specs_used):
            sx, sy = spec["points"][0]
            vx, vy = spec["points"][1]
            hour = clock_hour(sx, sy, vx, vy)
            agent = spec["agent"]
            turns.append(_gaze_turn(
                QUESTION_HOUR, spec["hour_line"], spec["points"], agent, "hour", hour,
            ))
            if index in want_dir:
                turns.append(_gaze_turn(
                    QUESTION_DIRECTION, spec["dir_line"], spec["points"], agent,
                    "direction", hour,
                ))
            placements.append({
                "direction": spec["direction"],
                "s": [float(sx), float(sy)],
                "v": [float(vx), float(vy)],
                "hour": hour,
            })
        if not turns:
            raise RuntimeError("gaze batch has no replies")
        return Batch(
            family="A",
            path=path,
            context="",
            notepad=None,
            settings={"agent": list(specs_used[0]["agent"]), "placements": placements},
            turns=turns,
            inject="oracle",
        )
    except Exception:
        path.unlink(missing_ok=True)
        raise


def build_gaze_game(
    rng: random.Random,
    tmp: Path,
    step: int,
    n_turns: int,
    question: str | None,
) -> Batch:
    """Notepad and fake history, then "Look at X." plus the hour (2/3) or
    direction question in one user turn. The reply is a look line and
    the answer line."""
    for _attempt in range(BOARD_TRIES):
        path: Path | None = None
        try:
            path, settings = _deal(rng, tmp, step, "c")
            return _gaze_game_batch(rng, path, settings, n_turns, question)
        except Redraw as exc:
            _drop(exc.path if exc.path is not None else path)
            continue
        except Exception:
            _drop(path)
            raise
    raise RuntimeError(f"gaze_game drew no board in {BOARD_TRIES} tries")


def build_look_move(
    rng: random.Random,
    tmp: Path,
    step: int,
    force_kind: str | None,
) -> Batch:
    for _attempt in range(BOARD_TRIES):
        path = None
        try:
            path, settings = _deal(rng, tmp, step, "c")
            return _look_move_batch(rng, path, settings, force_kind)
        except Redraw as exc:
            _drop(exc.path if exc.path is not None else path)
            continue
        except Exception:
            _drop(path)
            raise
    raise RuntimeError(f"look_move drew no board in {BOARD_TRIES} tries")


def build_begin(
    rng: random.Random,
    tmp: Path,
    step: int,
    board: str | None,
    limit_one: bool,
) -> Batch:
    for _attempt in range(BOARD_TRIES):
        path = None
        try:
            kind_board = board if board is not None else _begin_board(rng)
            path, settings = _deal(rng, tmp, step, kind_board)
            return _begin_batch(rng, path, settings, limit_one)
        except Redraw as exc:
            _drop(exc.path if exc.path is not None else path)
            continue
        except Exception:
            _drop(path)
            raise
    raise RuntimeError(f"begin drew no board in {BOARD_TRIES} tries")


def build_retarget(
    rng: random.Random,
    tmp: Path,
    step: int,
    sub: str | None,
    board: str | None,
) -> Batch:
    for _attempt in range(BOARD_TRIES):
        path = None
        try:
            which = sub if sub is not None else _retarget_sub(rng)
            kind_board = board if board is not None else _retarget_board(rng)
            path, settings = _deal(rng, tmp, step, kind_board)
            if which == "after_miss" and not (settings.get("gold") or []):
                which = "after_eat"
            return _retarget_batch(rng, path, settings, which)
        except Redraw as exc:
            _drop(exc.path if exc.path is not None else path)
            continue
        except Exception:
            _drop(path)
            raise
    raise RuntimeError(f"retarget drew no board in {BOARD_TRIES} tries")


def run_eval(trainer: Any) -> dict[str, float]:
    """Step-0 / periodic greedy eval.

    Writes summary.json, samples.jsonl, and reasoning.jsonl.
    """
    if trainer.tlog is None or trainer.collator is None:
        raise RuntimeError("eval before the trainer is loaded")
    if trainer.tap is None or trainer.localizer is None or trainer.embedder is None:
        raise RuntimeError("eval before the readout is loaded")
    rng = random.Random(int(trainer.step))
    tmp = Path(tempfile.mkdtemp(prefix="weekend_eval_"))
    out_dir = trainer.tlog.run_dir / "eval" / f"step_{int(trainer.step):06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    trainer.model.set_adapter("default")
    trainer.embedder.clear_pending()
    try:
        # Every family with a board eval, in list order, trained or not.
        for spec in _families().FAMILIES:
            if spec.eval is not None:
                samples.extend(spec.eval(trainer, rng, tmp))
        anchor = _eval_anchor(trainer)
        reasoning = _eval_reasoning(trainer)
    finally:
        trainer.embedder.clear_pending()
        shutil.rmtree(tmp, ignore_errors=True)
    summary = _summarize(int(trainer.step), samples, anchor, reasoning)
    _write_eval(out_dir, summary, samples)
    _write_jsonl(out_dir / "reasoning.jsonl", reasoning["samples"])
    flat = dict(summary["metrics"])
    logger.info("eval step %d %s", trainer.step, _format_flat(flat))
    return flat


def _eval_gaze(trainer: Any, rng: random.Random, tmp: Path) -> list[dict[str, Any]]:
    rows = []
    for _ in range(32):
        batch = build_gaze(
            rng, lambda: trainer.place_gaze(rng, tmp, trainer.step), mode="one",
        )
        try:
            rows.extend(_generate_batch(trainer, batch))
        finally:
            batch.path.unlink(missing_ok=True)
    return rows


def _eval_gaze_game(trainer: Any, rng: random.Random, tmp: Path) -> list[dict[str, Any]]:
    rows = []
    for question, n in (("hour", 32), ("direction", 16)):
        for _ in range(n):
            batch = build_gaze_game(rng, tmp, trainer.step, n_turns=1, question=question)
            try:
                rows.extend(_generate_batch(trainer, batch))
            finally:
                batch.path.unlink(missing_ok=True)
    return rows


def _eval_look(trainer: Any, rng: random.Random, tmp: Path) -> list[dict[str, Any]]:
    rows = []
    for kind, n in (("look", 24), ("move", 24)):
        for _ in range(n):
            batch = build_look_move(rng, tmp, trainer.step, force_kind=kind)
            try:
                rows.extend(_generate_batch(trainer, batch))
            finally:
                batch.path.unlink(missing_ok=True)
    return rows


def _eval_begin(trainer: Any, rng: random.Random, tmp: Path) -> list[dict[str, Any]]:
    rows = []
    for board in ("sealed", "exit", "gold"):
        for _ in range(16):
            batch = build_begin(rng, tmp, trainer.step, board=board, limit_one=True)
            try:
                rows.extend(_generate_batch(trainer, batch))
            finally:
                batch.path.unlink(missing_ok=True)
    return rows


def _eval_retarget(trainer: Any, rng: random.Random, tmp: Path) -> list[dict[str, Any]]:
    rows = []
    plan = (
        ("after_eat", None),
        ("after_miss", "gold"),
        ("empty_pad", None),
        ("end_ask", None),
    )
    for sub, board in plan:
        for _ in range(12):
            batch = build_retarget(rng, tmp, trainer.step, sub=sub, board=board)
            try:
                if len(batch.turns) > 1:
                    batch.turns = [rng.choice(batch.turns)]
                rows.extend(_generate_batch(trainer, batch))
            finally:
                batch.path.unlink(missing_ok=True)
    return rows


def _eval_shift(trainer: Any, rng: random.Random, tmp: Path) -> list[dict[str, Any]]:
    rows = []
    for case in _families().SHIFT_CASES:
        for _ in range(SHIFT_EVAL_PER_CASE):
            batch = build_shift(rng, tmp, trainer.step, case=case)
            try:
                rows.extend(_generate_batch(trainer, batch))
            finally:
                batch.path.unlink(missing_ok=True)
    return rows


def _eval_anchor(trainer: Any) -> dict[str, float]:
    held = list(trainer.holdout[:HOLDOUT_EVAL])
    if not held:
        raise RuntimeError("anchor eval has no holdout examples")
    trainer.model.set_adapter("default")
    trainer.embedder.clear_pending()
    losses = []
    for ex in held:
        built = trainer.collator.build(ex)
        with torch.inference_mode():
            loss = weighted_loss(
                trainer.model,
                built["model_inputs"],
                built["weights"],
                loss_kind="ce",
                example_weight=built["example_weight"],
            )
        value = float(loss)
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite holdout CE {value}")
        losses.append(value)
        trainer.embedder.clear_pending()
    return {"n": float(len(losses)), "mean": sum(losses) / len(losses)}


def _generate_batch(trainer: Any, batch: Batch) -> list[dict[str, Any]]:
    rows = []
    for turn in batch.turns:
        oracle = None
        if batch.inject == "oracle":
            oracle = torch.tensor(
                [[turn.s[0], turn.s[1], turn.v[0], turn.v[1]]],
                dtype=torch.float32,
                device=trainer.device,
            )
        text, where = greedy(trainer, messages_for(batch, turn), oracle)
        rows.append(_sample(trainer.step, batch, turn, text, where))
    return rows


def greedy(
    trainer: Any, messages: list[dict], oracle: torch.Tensor | None,
) -> tuple[str, list[float]]:
    """Argmax through prefill_step / decode_step. Stop on eos, move, end, or 64."""
    trainer._discard_tap()
    inputs = _encode_prompt(trainer, messages)
    logits, coords, past, grid = prefill_step(
        trainer.model, trainer.tap, trainer.localizer, trainer.embedder, inputs,
    )
    if grid is None:
        raise RuntimeError("eval prefill found no image tokens")
    seq_len = int(inputs["input_ids"].shape[1])
    new_ids: list[int] = []
    text = ""
    tokenizer = trainer.collator.tokenizer
    for _ in range(MAX_NEW_TOKENS):
        token = int(logits[0].argmax().item())
        if token == int(trainer.terminator):
            break
        new_ids.append(token)
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        prev = oracle if oracle is not None else coords
        token_ids = torch.tensor([token], dtype=torch.long, device=trainer.device)
        logits, coords, past = decode_step(
            trainer.model,
            trainer.tap,
            trainer.localizer,
            trainer.embedder,
            grid,
            token_ids,
            past,
            seq_len,
            prev,
        )
        seq_len += 1
        if ends_with_move(text) or ends_with_end_game(text) or len(new_ids) >= MAX_NEW_TOKENS:
            break
    where = [float(coords[0, 2].item()), float(coords[0, 3].item())]
    return text, where


def _encode_prompt(trainer: Any, messages: list[dict]) -> dict[str, Any]:
    norm = trainer.collator.adapter.prepare_messages(messages)
    prompt = trainer.processor.apply_chat_template(
        norm,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    moved: dict[str, Any] = {}
    for key, val in prompt.items():
        if not torch.is_tensor(val):
            continue
        if val.is_floating_point():
            val = val.to(device=trainer.device, dtype=torch.bfloat16)
        else:
            val = val.to(trainer.device)
        moved[key] = val
    if "attention_mask" not in moved:
        moved["attention_mask"] = torch.ones_like(moved["input_ids"])
    if not any(val.is_floating_point() for val in moved.values()):
        raise RuntimeError("eval prompt produced no image tensor")
    return moved


def _sample(
    step: int, batch: Batch, turn: Turn, text: str, where: list[float],
) -> dict[str, Any]:
    return {
        "step": int(step),
        "family": _families().family(batch.family).sample,
        "task": turn.task,
        "settings": batch.settings,
        "user": turn.question,
        "context": batch.context,
        "notepad": batch.notepad,
        "reply": text,
        "expected": turn.reply,
        "scores": _scores(batch, turn, text, where),
    }


def _scores(batch: Batch, turn: Turn, text: str, where: list[float]) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    answer = text.strip()
    expected = turn.reply.strip()
    if batch.family == "B":
        # A look line, then the answer line. The look line is any of
        # the trainer's look lines, not the one this turn drew.
        lines = [line.strip() for line in answer.splitlines()]
        scores["assent_ok"] = len(lines) == 2 and lines[0] in _S2_LOOK_LINES
        answer = lines[-1] if lines else ""
        expected = expected.splitlines()[-1]
    if turn.task == "hour":
        parsed = parse_reply(KIND_HOUR, answer)
        scores["hour_parse"] = parsed is not None
        scores["hour_exact"] = parsed == turn.extra["hour"]
        scores["hour_reward"] = (
            0.0 if parsed is None else hour_reward(int(turn.extra["hour"]), int(parsed))
        )
    if turn.task == "direction":
        scores["direction_exact"] = answer == expected
    if batch.family == "B":
        scores["v_dist"] = _dist(where, turn.v)
    if batch.family == "C":
        scores["format_ok"] = _format_ok(turn.task, text)
        scores["move_token_correct"] = ends_with_move(text) == turn.expect_move
        scores["spurious_remember"] = "[REMEMBER" in text
        scores["v_dist"] = _dist(where, turn.v)
        scores["v_hit_0.05"] = scores["v_dist"] < 0.05
    if batch.family in ("D", "E"):
        scores.update(_game_scores(batch, turn, text, where))
    if batch.family == "G":
        scores.update(_shift_scores(turn, text, where))
    return scores


def _shift_scores(turn: Turn, text: str, where: list[float]) -> dict[str, Any]:
    """One look line per instruction, and the final gaze on ``v``.

    ``shift_sign_ok``: the gaze moved from the anchor the way the (summed)
    instruction said, at all. ``shift_hit``: it landed within
    :data:`SHIFT_HIT` of ``v``. ``v_dist`` feeds the rms.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    n_expected = len(turn.reply.strip().splitlines())
    anchor = turn.extra["anchor"]
    total = (float(turn.v[0]) - float(anchor[0]), float(turn.v[1]) - float(anchor[1]))
    moved = (float(where[0]) - float(anchor[0]), float(where[1]) - float(anchor[1]))
    dist = _dist(where, turn.v)
    return {
        "format_ok": len(lines) == n_expected and all(line in _S2_LOOK_LINES for line in lines),
        "shift_sign_ok": moved[0] * total[0] + moved[1] * total[1] > 0.0,
        "shift_hit": dist < SHIFT_HIT,
        "v_dist": dist,
    }


def _game_scores(
    batch: Batch, turn: Turn, text: str, where: list[float],
) -> dict[str, Any]:
    extra = turn.extra
    scores: dict[str, Any] = {
        "move_token_correct": ends_with_move(text) == turn.expect_move,
    }
    value = _remember_value(text)
    rule = extra.get("target_rule")
    if rule == "absent":
        scores["target_valid"] = value is None
    elif rule == "phrase":
        scores["target_valid"] = value in extra["phrases"]
    if extra.get("verdict"):
        first = text.strip().splitlines()[0].strip() if text.strip() else ""
        scores["verdict_ok"] = first == extra["verdict"]
    if extra.get("sub") == "end_ask":
        scores["end_ask_ok"] = text.strip() == S2_END_TOKEN
    sealed = bool(extra.get("sealed"))
    if sealed and extra.get("sub") != "end_ask":
        scores["end_game_when_sealed"] = (
            ends_with_end_game(text) and S2_SEALED_NOTE in text
        )
    if "[END_GAME]" not in turn.reply:
        scores["end_game_spurious"] = "[END_GAME]" in text
    if not (batch.settings.get("gold") or []):
        scores["gold_hallucinated"] = bool(value is not None and "gold" in value)
    if batch.family == "D":
        point = extra["phrase_points"].get(value) if value is not None else None
        if point is not None:
            scores["v_dist"] = _dist(where, point)
    if batch.family == "E":
        scores["v_dist"] = _dist(where, turn.v)
    return scores


def _summarize(
    step: int,
    samples: list[dict[str, Any]],
    anchor: dict[str, float],
    reasoning: dict[str, Any],
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    metrics: dict[str, float] = {}
    rate_keys = {
        "gaze/oracle": ("hour_parse", "hour_exact", "direction_exact"),
        "gaze/game": ("assent_ok", "hour_parse", "hour_exact", "direction_exact"),
        "look_move": (
            "format_ok", "move_token_correct", "spurious_remember", "v_hit_0.05",
        ),
        "begin": (
            "end_game_when_sealed", "end_game_spurious", "target_valid",
            "gold_hallucinated", "move_token_correct",
        ),
        "retarget": (
            "verdict_ok", "target_valid", "gold_hallucinated",
            "end_game_when_sealed", "end_game_spurious", "end_ask_ok",
            "move_token_correct",
        ),
        "shift": ("format_ok", "shift_sign_ok", "shift_hit"),
    }
    mean_keys = {
        "gaze/oracle": ("hour_reward",),
        "gaze/game": ("hour_reward",),
    }
    rms_keys = {
        "gaze/game": ("v_dist",),
        "look_move": ("v_dist",),
        "begin": ("v_dist",),
        "retarget": ("v_dist",),
        "shift": ("v_dist",),
    }
    prefix = {
        "gaze/oracle": "A_",
        "gaze/game": "B_",
        "look_move": "C_",
        "begin": "D_",
        "retarget": "E_",
        "shift": "G_",
    }
    for name, keys in rate_keys.items():
        rows = [row for row in samples if row["family"] == name]
        group: dict[str, Any] = {"n": len(rows)}
        for key in keys:
            stat = _rate(rows, key)
            if stat is None:
                continue
            group[key] = stat
            metrics[prefix[name] + key] = stat["rate"]
            metrics[prefix[name] + key + "_n"] = float(stat["n"])
        for key in mean_keys.get(name, ()):
            stat = _mean(rows, key)
            if stat is None:
                continue
            group[key] = stat
            metrics[prefix[name] + key] = stat["mean"]
            metrics[prefix[name] + key + "_n"] = float(stat["n"])
        for key in rms_keys.get(name, ()):
            stat = _rms(rows, key)
            if stat is None:
                continue
            label = "v_rms" if key == "v_dist" else key
            group[label] = stat
            metrics[prefix[name] + label] = stat["rms"]
            metrics[prefix[name] + label + "_n"] = float(stat["n"])
        groups[name] = group
    look_move = groups.get("look_move")
    if isinstance(look_move, dict):
        for task in ("look", "move"):
            task_rows = [
                row for row in samples
                if row["family"] == "look_move" and row["task"] == task
            ]
            hit = _rate(task_rows, "v_hit_0.05")
            if hit is not None:
                look_move[task + "_v_hit_0.05"] = hit
                metrics["C_" + task + "_v_hit_0.05"] = hit["rate"]
                metrics["C_" + task + "_v_hit_0.05_n"] = float(hit["n"])
            dist = _rms(task_rows, "v_dist")
            if dist is not None:
                look_move[task + "_v_rms"] = dist
                metrics["C_" + task + "_v_rms"] = dist["rms"]
                metrics["C_" + task + "_v_rms_n"] = float(dist["n"])
    shift = groups.get("shift")
    if isinstance(shift, dict):
        # Per sub-case hit rate (the task field of a shift sample is its case).
        for case in sorted({row["task"] for row in samples if row["family"] == "shift"}):
            case_rows = [
                row for row in samples
                if row["family"] == "shift" and row["task"] == case
            ]
            hit = _rate(case_rows, "shift_hit")
            if hit is not None:
                shift[case + "_shift_hit"] = hit
                metrics["G_" + case + "_shift_hit"] = hit["rate"]
                metrics["G_" + case + "_shift_hit_n"] = float(hit["n"])
    groups["anchor"] = {"ce": {"n": int(anchor["n"]), "mean": anchor["mean"]}}
    metrics["F_ce"] = float(anchor["mean"])
    metrics["F_ce_n"] = float(anchor["n"])
    groups["reasoning"] = {
        "dataset": reasoning["dataset"],
        "slice": (
            f"first {REASONING_N} rows of the {REASONING_DATASET} "
            "test-split probe"
        ),
        "exact": {
            "n": float(reasoning["n"]),
            "correct": float(reasoning["correct"]),
            "rate": float(reasoning["rate"]),
        },
    }
    metrics["gsm8k_exact"] = float(reasoning["rate"])
    metrics["gsm8k_exact_n"] = float(reasoning["n"])
    return {"step": step, "groups": groups, "metrics": metrics}


def reasoning_items() -> list[dict[str, Any]]:
    """First ``REASONING_N`` rows of the GSM8K test-split probe.

    That probe is the held-out test split, not the train rows in family F.
    The file is written by ``python -m training.download_external``.
    """
    matches = [entry for entry in load_manifest() if entry.name == REASONING_DATASET]
    if len(matches) != 1:
        raise RuntimeError(
            f"manifest must contain one {REASONING_DATASET!r} entry, "
            f"found {len(matches)}"
        )
    entry = matches[0]
    if not entry.enabled:
        raise RuntimeError(f"{REASONING_DATASET} is disabled in the manifest")
    kind = (entry.probe or {}).get("kind")
    if kind != "exact_match":
        raise RuntimeError(
            f"{REASONING_DATASET} probe kind is {kind!r}, expected 'exact_match'"
        )
    path = entry.probe_path
    if not path.is_file():
        raise FileNotFoundError(
            f"reasoning probe missing: {path} -- run: "
            "python -m training.download_external --only gsm8k"
        )
    items = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(items) < REASONING_N:
        raise RuntimeError(
            f"{path} has {len(items)} items, need at least {REASONING_N}"
        )
    chosen = items[:REASONING_N]
    for item in chosen:
        if "answer" not in item or "messages" not in item:
            raise RuntimeError(f"{path} item is missing messages or answer")
    return chosen


def _user_text(messages: list[dict]) -> str:
    chunks: list[str] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            chunks.append(content)
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                chunks.append(str(part.get("text") or ""))
    return "\n".join(chunks)


def _eval_reasoning(trainer: Any) -> dict[str, Any]:
    """Greedy exact match. Text only: no coordinate payload is armed."""
    if (
        trainer.collator is None
        or trainer.model is None
        or trainer.processor is None
        or trainer.embedder is None
    ):
        raise RuntimeError("reasoning eval before the model is loaded")
    items = reasoning_items()
    tokenizer = trainer.collator.tokenizer
    adapter = trainer.collator.adapter
    trainer.model.set_adapter("default")
    trainer.embedder.clear_pending()
    trainer._discard_tap()
    correct = 0
    samples: list[dict[str, Any]] = []
    logger.info("gsm8k exact-match on %d held-out probe items", len(items))
    with torch.inference_mode():
        for index, item in enumerate(items):
            norm = adapter.prepare_messages(item["messages"])
            inputs = trainer.processor.apply_chat_template(
                norm,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(trainer.device)
            out = trainer.model.generate(
                **inputs,
                max_new_tokens=REASONING_MAX_NEW,
                do_sample=False,
            )
            prompt_len = int(inputs["input_ids"].shape[1])
            if int(out.shape[1]) < prompt_len:
                raise RuntimeError(
                    f"gsm8k generate returned {int(out.shape[1])} tokens "
                    f"for a prompt of {prompt_len}"
                )
            reply = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
            pred = extract_answer(reply)
            ok = answers_match(pred, str(item["answer"]))
            if ok:
                correct += 1
            samples.append({
                "step": int(trainer.step),
                "dataset": REASONING_DATASET,
                "index": index,
                "question": _user_text(item["messages"]),
                "reply": reply,
                "answer": str(item["answer"]),
                "pred": pred,
                "correct": ok,
            })
            trainer.embedder.clear_pending()
            trainer._discard_tap()
            done = index + 1
            if done % 8 == 0 or done == len(items):
                logger.info("gsm8k %d/%d correct %d", done, len(items), correct)
    return {
        "dataset": REASONING_DATASET,
        "n": len(items),
        "correct": correct,
        "rate": correct / len(items),
        "samples": samples,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [json.dumps(row) for row in rows]
    _write_text(path, ("\n".join(lines) + "\n") if lines else "")


def _write_eval(out_dir: Path, summary: dict[str, Any], samples: list[dict[str, Any]]) -> None:
    lines = [json.dumps(row) for row in samples]
    _write_text(out_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    _write_text(out_dir / "samples.jsonl", ("\n".join(lines) + "\n") if lines else "")


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _gaze_turn(
    question: str,
    reply: str,
    points: tuple[tuple[float, float], tuple[float, float]],
    agent: tuple[float, float],
    task: str,
    hour: int,
) -> Turn:
    return Turn(
        question=question,
        reply=reply,
        s=(float(points[0][0]), float(points[0][1])),
        v=(float(points[1][0]), float(points[1][1])),
        agent=(float(agent[0]), float(agent[1])),
        all_mid=False,
        expect_move=False,
        task=task,
        extra={"hour": int(hour)},
    )


def _gaze_game_batch(
    rng: random.Random,
    path: Path,
    settings: dict[str, Any],
    n_turns: int,
    question: str | None,
) -> Batch:
    candidates = _targets(settings)
    note = rng.choice(candidates)
    if sum(1 for item in candidates if item[1] == note[1]) != 1:
        raise Redraw(path)
    ctx = board_context(settings, rng, forced_note=note[1])
    usable = []
    for kind, phrase, point in ctx.candidates:
        got = _gaze_of(ctx.agent, point)
        if got is None:
            continue
        usable.append((kind, phrase, point, got))
    if len(usable) < n_turns:
        raise Redraw(path)
    rng.shuffle(usable)
    chosen = usable[:n_turns]
    if question is None:
        question = "hour" if rng.random() < (2.0 / 3.0) else "direction"
    if question not in ("hour", "direction"):
        raise ValueError(f"gaze question {question!r}")
    asked = QUESTION_HOUR if question == "hour" else QUESTION_DIRECTION
    turns = []
    for _kind, phrase, point, (axis, label, hour) in chosen:
        if question == "hour":
            answer = correct_line(KIND_HOUR, hour)
            task = "hour"
        else:
            answer = correct_line(axis, label)
            task = "direction"
        # The look and the question share one user turn; the reply is a
        # look line, then the answer. ``_scores`` reads the last line.
        turns.append(Turn(
            question=f"{_ask('look', phrase)} {asked}",
            reply=f"{rng.choice(_S2_LOOK_LINES)}\n{answer}",
            s=ctx.agent,
            v=(float(point[0]), float(point[1])),
            agent=ctx.agent,
            all_mid=False,
            expect_move=False,
            task=task,
            extra={"hour": int(hour)},
        ))
    return Batch(
        family="B",
        path=path,
        context=ctx.context,
        notepad=ctx.notepad,
        settings=settings,
        turns=turns,
        inject="feedback",
    )


def _look_move_batch(
    rng: random.Random,
    path: Path,
    settings: dict[str, Any],
    force_kind: str | None,
) -> Batch:
    ctx = board_context(settings, rng, weights=C_WEIGHTS)
    if force_kind is None:
        drawn = draw_questions(ctx, rng, 6, C_WEIGHTS)
    else:
        drawn = []
        for _try in range(40):
            item = draw_question(ctx, rng, C_WEIGHTS)
            if item.kind == force_kind:
                drawn = [item]
                break
    if not drawn:
        raise Redraw(path)
    turns = []
    for item in drawn:
        sx, sy, vx, vy = _coords(ctx.agent, item.point)
        if item.kind == "look":
            reply = rng.choice(_S2_LOOK_LINES)
            expect_move = False
        elif item.kind == "move":
            reply = s2_move_reply(rng.choice(_S2_MOVE_LINES))
            expect_move = True
        else:
            raise RuntimeError(f"draw kind {item.kind!r}")
        turns.append(Turn(
            question=item.question,
            reply=reply,
            s=(float(sx), float(sy)),
            v=(float(vx), float(vy)),
            agent=ctx.agent,
            all_mid=False,
            expect_move=expect_move,
            task=item.kind,
            extra={"phrase": item.phrase},
        ))
    return Batch(
        family="C",
        path=path,
        context=ctx.context,
        notepad=ctx.notepad,
        settings=settings,
        turns=turns,
        inject="feedback",
    )


def _weighted_key(rng: random.Random, table: dict[str, float]) -> str:
    total = sum(max(0.0, float(w)) for w in table.values())
    if total <= 0.0:
        raise RuntimeError(f"weights sum to {total}: {table}")
    pick = rng.random() * total
    acc = 0.0
    chosen = next(iter(table))
    for key, weight in table.items():
        if weight <= 0.0:
            continue
        chosen = key
        acc += float(weight)
        if pick < acc:
            break
    return chosen


def _shift_inside(point: tuple[float, float]) -> bool:
    inner = _families().SHIFT_INNER
    return inner <= point[0] <= 1.0 - inner and inner <= point[1] <= 1.0 - inner


def _shift_step(
    rng: random.Random,
    start: tuple[float, float],
    size: float,
    exclude: tuple[str, ...] = (),
) -> tuple[str, tuple[float, float]] | None:
    """A direction whose step of ``size`` from ``start`` stays in the room."""
    fam = _families()
    names = [name for name in fam.SHIFT_DIRECTIONS if name not in exclude]
    rng.shuffle(names)
    for name in names:
        ux, uy = fam.SHIFT_DIRECTIONS[name]
        end = (start[0] + ux * size, start[1] + uy * size)
        if _shift_inside(end):
            return name, end
    return None


def _opposite(direction: str) -> str:
    return {"left": "right", "right": "left", "up": "down", "down": "up"}[direction]


def _shift_phrase(rng: random.Random, kind: str, direction: str) -> str:
    fam = _families()
    words = fam.SHIFT_DIRECTION_WORDS[direction]
    # "look left" reads fine; "look a little left" is rarer than "a little
    # to the left", so the sized forms keep the first (long) wording.
    word = rng.choice(words) if kind == "bare" else words[0]
    return rng.choice(fam.SHIFT_TEMPLATES[kind]).format(dir=word)


def _shift_batch(
    rng: random.Random,
    path: Path,
    settings: dict[str, Any],
    case: str,
) -> Batch:
    """One board, one anchor place, one shift prompt of the given case.

    The anchor is a named place drawn with ``SHIFT_TARGETS``; the shift
    starts from it. Replies are look lines, one per instruction in the
    user text; ``v_lines`` carries the gaze of each line.
    """
    fam = _families()
    if case not in fam.SHIFT_CASES:
        raise ValueError(f"shift case {case!r}; cases are {list(fam.SHIFT_CASES)}")
    ctx = board_context(settings, rng, weights=fam.SHIFT_TARGETS)
    # The anchor may sit in a wall (an exit); only the shifted points must
    # be inside, which _shift_step checks, redrawing the board otherwise.
    kind, phrase, anchor = _draw_target(ctx.candidates, rng, fam.SHIFT_TARGETS)
    anchor = (float(anchor[0]), float(anchor[1]))
    little = float(fam.SHIFT_LITTLE)
    look = lambda: rng.choice(_S2_LOOK_LINES)  # noqa: E731
    anchor_round = (_ask("look", phrase), look())
    context = ctx.context
    v_lines: list[tuple[float, float]] | None = None
    if case == "two_turn":
        step = _shift_step(rng, anchor, little)
        if step is None:
            raise Redraw(path)
        direction, end = step
        context = _append_round(context, *anchor_round)
        question = _shift_phrase(rng, "little", direction)
        reply = look()
        directions = [direction]
    elif case == "one_turn":
        step = _shift_step(rng, anchor, little)
        if step is None:
            raise Redraw(path)
        direction, end = step
        question = f"{anchor_round[0]} {_shift_phrase(rng, 'little', direction)}"
        reply = f"{look()}\n{look()}"
        v_lines = [anchor, end]
        directions = [direction]
    elif case in ("stack_msg", "stack_turns"):
        first = _shift_step(rng, anchor, little)
        if first is None:
            raise Redraw(path)
        d1, mid = first
        # The second step is the same way or perpendicular, never straight
        # back (that would cancel the first and leave nothing to score).
        second = _shift_step(rng, mid, little, exclude=(_opposite(d1),))
        if second is None:
            raise Redraw(path)
        d2, end = second
        phrase_1 = _shift_phrase(rng, "little", d1)
        phrase_2 = _shift_phrase(rng, "little", d2)
        if case == "stack_msg":
            question = f"{anchor_round[0]} {phrase_1} {phrase_2}"
            reply = f"{look()}\n{look()}\n{look()}"
            v_lines = [anchor, mid, end]
        else:
            context = _append_round(context, *anchor_round)
            context = _append_round(context, phrase_1, look())
            question = phrase_2
            reply = look()
        directions = [d1, d2]
    elif case == "more":
        # Two steps the same way must fit; "more" repeats the prior one.
        step = None
        for _try in range(8):
            cand = _shift_step(rng, anchor, little)
            if cand is None:
                break
            direction, mid = cand
            ux, uy = fam.SHIFT_DIRECTIONS[direction]
            end = (mid[0] + ux * little, mid[1] + uy * little)
            if _shift_inside(end):
                step = (direction, end)
                break
        if step is None:
            raise Redraw(path)
        direction, end = step
        context = _append_round(context, *anchor_round)
        context = _append_round(context, _shift_phrase(rng, "little", direction), look())
        question = _shift_phrase(rng, "more", direction)
        reply = look()
        directions = [direction, direction]
    elif case == "bare":
        step = _shift_step(rng, anchor, float(fam.SHIFT_BARE))
        if step is None:
            raise Redraw(path)
        direction, end = step
        context = _append_round(context, *anchor_round)
        question = _shift_phrase(rng, "bare", direction)
        reply = look()
        directions = [direction]
    else:
        raise ValueError(f"shift case {case!r}")
    turn = Turn(
        question=question,
        reply=reply,
        s=ctx.agent,
        v=(float(end[0]), float(end[1])),
        agent=ctx.agent,
        all_mid=False,
        expect_move=False,
        task=case,
        extra={
            "case": case,
            "anchor": anchor,
            "anchor_kind": kind,
            "phrase": phrase,
            "directions": directions,
        },
        v_lines=v_lines,
    )
    return Batch(
        family="G",
        path=path,
        context=context,
        notepad=ctx.notepad,
        settings=settings,
        turns=[turn],
        inject="feedback",
    )


def _begin_batch(
    rng: random.Random, path: Path, settings: dict[str, Any], limit_one: bool,
) -> Batch:
    kind = "look" if rng.random() < 0.5 else "move"
    question = S2_BEGINNING_USER if kind == "look" else S2_BEGINNING_MOVE_USER
    targets = _begin_targets(settings)
    if not targets:
        raise Redraw(path)
    agent = _agent(settings)
    phrase_points = {phrase: point for _cand, phrase, point in targets}
    sealed = game_io.sealed_empty(settings)
    lines = _S2_LOOK_LINES if kind == "look" else _S2_MOVE_LINES
    turns = []
    for cand, phrase, point in targets:
        reply = _begin_reply(phrase, kind, cand, line=rng.choice(lines))
        turns.append(Turn(
            question=question,
            reply=reply,
            s=agent,
            v=(float(point[0]), float(point[1])),
            agent=agent,
            all_mid=cand == "none",
            expect_move=kind == "move" and cand != "none",
            task="begin",
            extra={
                "sealed": sealed,
                "phrases": set(phrase_points),
                "phrase_points": phrase_points,
                "target_rule": "phrase",
                "sub": "begin",
                "kind": kind,
            },
        ))
    if limit_one and len(turns) > 1:
        turns = [rng.choice(turns)]
    return Batch(
        family="D",
        path=path,
        context="",
        notepad=format_notepad([]),
        settings=settings,
        turns=turns,
        inject="feedback",
    )


def _retarget_batch(
    rng: random.Random, path: Path, settings: dict[str, Any], sub: str,
) -> Batch:
    agent = _agent(settings)
    sealed = game_io.sealed_empty(settings)
    targets = _begin_targets(settings)
    if not targets:
        raise Redraw(path)
    phrase_points = {phrase: point for _cand, phrase, point in targets}
    if sub == "after_eat":
        turns, notepad, context = _after_eat(
            rng, settings, agent, sealed, targets, phrase_points,
        )
    elif sub == "after_miss":
        turns, notepad, context = _after_miss(rng, settings, agent, phrase_points)
    elif sub == "empty_pad":
        turns, notepad, context = _empty_pad(
            rng, agent, sealed, targets, phrase_points,
        )
    elif sub == "end_ask":
        turns, notepad, context = _end_ask(rng, settings, agent, sealed)
    else:
        raise ValueError(f"retarget sub {sub!r}")
    if not turns:
        raise Redraw(path)
    return Batch(
        family="E",
        path=path,
        context=context,
        notepad=notepad,
        settings=settings,
        turns=turns,
        inject="feedback",
    )


def _after_eat(
    rng: random.Random,
    settings: dict[str, Any],
    agent: tuple[float, float],
    sealed: bool,
    targets: list[tuple[str, str, tuple[float, float]]],
    phrase_points: dict[str, tuple[float, float]],
) -> tuple[list[Turn], str, str]:
    note = _absent_gold_phrase(settings, rng)
    notepad = _note(note, rng)
    remaining = len(settings.get("gold") or [])
    question = game_io.compose_s2_question(
        S2_MOVE_TARGET_USER, _move_record(rng, 1, remaining),
    )
    verdict = S2_MOVE_VERDICT_LINES[0]
    turns = []
    for cand, phrase, point in targets:
        if cand == "none":
            reply = verdict + "\n" + S2_END_REPLY
            all_mid = True
            expect_move = False
        else:
            reply = verdict + "\n" + _begin_reply(
                phrase, "move", cand, line=rng.choice(_S2_MOVE_LINES),
            )
            all_mid = False
            expect_move = True
        turns.append(Turn(
            question=question,
            reply=reply,
            s=agent,
            v=(float(point[0]), float(point[1])),
            agent=agent,
            all_mid=all_mid,
            expect_move=expect_move,
            task="after_eat",
            extra=_phrase_extra("after_eat", sealed, phrase_points, verdict),
        ))
    return turns, notepad, ""


def _after_miss(
    rng: random.Random,
    settings: dict[str, Any],
    agent: tuple[float, float],
    phrase_points: dict[str, tuple[float, float]],
) -> tuple[list[Turn], str, str]:
    golds = [
        (phrase, point)
        for cand, phrase, point in _begin_targets(settings)
        if cand == "gold"
    ]
    if not golds:
        raise Redraw()
    phrase, point = rng.choice(golds)
    remaining = len(settings.get("gold") or [])
    question = game_io.compose_s2_question(
        S2_LOOK_TARGET_USER, _move_record(rng, 0, remaining),
    )
    verdict = S2_MOVE_VERDICT_LINES[1]
    reply = verdict + "\n" + rng.choice(_S2_LOOK_LINES)
    extra = _phrase_extra("after_miss", False, phrase_points, verdict)
    extra["target_rule"] = "absent"
    turn = Turn(
        question=question,
        reply=reply,
        s=agent,
        v=(float(point[0]), float(point[1])),
        agent=agent,
        all_mid=False,
        expect_move=False,
        task="after_miss",
        extra=extra,
    )
    return [turn], _note(phrase, rng), ""


def _empty_pad(
    rng: random.Random,
    agent: tuple[float, float],
    sealed: bool,
    targets: list[tuple[str, str, tuple[float, float]]],
    phrase_points: dict[str, tuple[float, float]],
) -> tuple[list[Turn], str, str]:
    kind = "look" if rng.random() < 0.5 else "move"
    question = S2_LOOK_TARGET_USER if kind == "look" else S2_MOVE_TARGET_USER
    lines = _S2_LOOK_LINES if kind == "look" else _S2_MOVE_LINES
    turns = []
    for cand, phrase, point in targets:
        reply = _begin_reply(phrase, kind, cand, line=rng.choice(lines))
        turns.append(Turn(
            question=question,
            reply=reply,
            s=agent,
            v=(float(point[0]), float(point[1])),
            agent=agent,
            all_mid=cand == "none",
            expect_move=kind == "move" and cand != "none",
            task="empty_pad",
            extra=_phrase_extra("empty_pad", sealed, phrase_points, None),
        ))
    return turns, format_notepad([]), ""


def _end_ask(
    rng: random.Random,
    settings: dict[str, Any],
    agent: tuple[float, float],
    sealed: bool,
) -> tuple[list[Turn], str, str]:
    if rng.random() < 0.5:
        notepad = format_notepad([])
    else:
        _cand, phrase, _point = rng.choice(_begin_targets(settings))
        notepad = _note(phrase, rng)
    turn = Turn(
        question=S2_END_GAME_USER,
        reply=S2_END_TOKEN,
        s=agent,
        v=agent,
        agent=agent,
        all_mid=True,
        expect_move=False,
        task="end_ask",
        extra={
            "sub": "end_ask",
            "sealed": sealed,
            "target_rule": None,
            "phrases": set(),
            "phrase_points": {},
            "verdict": None,
        },
    )
    return [turn], notepad, ""


def _phrase_extra(
    sub: str,
    sealed: bool,
    phrase_points: dict[str, tuple[float, float]],
    verdict: str | None,
) -> dict[str, Any]:
    return {
        "sub": sub,
        "sealed": sealed,
        "phrases": set(phrase_points),
        "phrase_points": phrase_points,
        "target_rule": "phrase",
        "verdict": verdict,
    }


def _deal(
    rng: random.Random, tmp: Path, step: int, board: str,
) -> tuple[Path, dict[str, Any]]:
    if board == "sealed":
        n_gold, opening = 0, "forbid"
    elif board == "exit":
        n_gold, opening = 0, "require"
    elif board == "gold":
        n_gold, opening = rng.choice((1, 2, 3)), "any"
    elif board == "c":
        n_gold, opening = _c_gold(rng), "any"
    else:
        raise ValueError(f"board kind {board!r}")
    game = game_io.new_multi_gold_game(
        gameSize=CONFIG.game_size, n_gold=n_gold, opening=opening,
    )
    settings = game_io.game_to_settings_dict(game)
    path = tmp / f"wk_{step}_{rng.randrange(1_000_000_000)}.png"
    Image.fromarray(canonical_frame(game)).save(path)
    if board == "sealed" and not game_io.sealed_empty(settings):
        path.unlink(missing_ok=True)
        raise Redraw()
    if board == "exit" and (settings.get("gold") or not settings.get("openings")):
        path.unlink(missing_ok=True)
        raise Redraw()
    if board == "gold" and not settings.get("gold"):
        path.unlink(missing_ok=True)
        raise Redraw()
    return path, settings


def _c_gold(rng: random.Random) -> int:
    roll = rng.random()
    if roll < 0.40:
        return 0
    if roll < 0.60:
        return 1
    if roll < 0.80:
        return 2
    return 3


def _begin_board(rng: random.Random) -> str:
    roll = rng.random()
    if roll < (1.0 / 3.0):
        return "sealed"
    if roll < (2.0 / 3.0):
        return "exit"
    return "gold"


def _retarget_board(rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.25:
        return "sealed"
    if roll < 0.50:
        return "exit"
    return "gold"


def _retarget_sub(rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.35:
        return "after_eat"
    if roll < 0.70:
        return "after_miss"
    if roll < 0.90:
        return "empty_pad"
    return "end_ask"


def _gaze_of(
    agent: tuple[float, float], point: tuple[float, float],
) -> tuple[str, str, int] | None:
    sx, sy = float(agent[0]), float(agent[1])
    vx, vy = float(point[0]), float(point[1])
    if not hour_is_unique(sx, sy, vx, vy):
        return None
    dx = abs(vx - sx)
    dy = abs(vy - sy)
    if dy > dx:
        axis = KIND_UPDOWN
    elif dx > dy:
        axis = KIND_LEFTRIGHT
    else:
        return None
    label = direction_label(axis, sx, sy, vx, vy)
    if label is None:
        return None
    return axis, label, clock_hour(sx, sy, vx, vy)


def _targets(settings: dict[str, Any]) -> list[tuple[str, str, tuple[float, float]]]:
    return _targets_on_board(settings)


def _agent(settings: dict[str, Any]) -> tuple[float, float]:
    return (float(settings["agent_x"]), float(settings["agent_y"]))


def _absent_gold_phrase(settings: dict[str, Any], rng: random.Random) -> str:
    present = {
        phrase
        for cand, phrase, _point in _begin_targets(settings)
        if cand == "gold"
    }
    pool = ["the gold", *gold_region_phrases()]
    absent = [phrase for phrase in pool if phrase not in present]
    if not absent:
        raise Redraw()
    return rng.choice(absent)


def _note(phrase: str, rng: random.Random) -> str:
    return format_notepad([{
        "key": "target",
        "value": phrase,
        "updated_round": rng.randrange(1, 4),
    }])


def _move_record(rng: random.Random, collected: int, remaining: int) -> dict[str, Any]:
    return {
        "collected": int(collected),
        "gold_remaining": int(remaining),
        "n_steps": rng.randint(1, 40),
        "stopped_by": rng.choice(("noops", "time")),
    }


def _append_round(context: str, user: str, assistant: str) -> str:
    if context:
        return context + "\n" + f"[user] {user}\n[assistant] {assistant}"
    return format_recent_block([("user", user), ("assistant", assistant)])


def _remember_value(text: str) -> str | None:
    found = _REMEMBER.search(text)
    if found is None:
        return None
    return found.group(1).strip()


def _format_ok(kind: str, text: str) -> bool:
    got = text.strip()
    if kind == "look":
        return got in _S2_LOOK_LINES
    if kind == "move":
        return got in {s2_move_reply(line) for line in _S2_MOVE_LINES}
    raise ValueError(f"format kind {kind!r}")


def _dist(where: list[float], point: tuple[float, float]) -> float:
    return math.hypot(float(where[0]) - float(point[0]), float(where[1]) - float(point[1]))


def _drop(path: Path | None) -> None:
    if path is not None:
        path.unlink(missing_ok=True)


def _rate(rows: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    picked = [row for row in rows if key in row["scores"]]
    if not picked:
        return None
    correct = sum(1 for row in picked if row["scores"][key])
    return {"n": float(len(picked)), "correct": float(correct), "rate": correct / len(picked)}


def _mean(rows: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    vals = [float(row["scores"][key]) for row in rows if key in row["scores"]]
    if not vals:
        return None
    return {"n": float(len(vals)), "mean": sum(vals) / len(vals)}


def _rms(rows: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    vals = [float(row["scores"][key]) for row in rows if key in row["scores"]]
    if not vals:
        return None
    mean_sq = sum(value * value for value in vals) / len(vals)
    return {"n": float(len(vals)), "rms": math.sqrt(mean_sq)}


def _format_flat(flat: dict[str, float]) -> str:
    bits = []
    for key in sorted(flat):
        if key.endswith("_n"):
            continue
        bits.append(f"{key} {flat[key]:.3f}")
    return " ".join(bits)
