"""Greedy eval for the weekend joint S2 run, and the boards it trains on.

Training and eval share :func:`build_train_family`. A generated reply is
scored here, not by teacher-forced CE, except the anchor holdout.
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
    _targets_on_board,
    beginning_messages,
    board_context,
    draw_question,
    draw_questions,
    gold_region_phrases,
)
from neural_net.s2_pretraining_localizer import _begin_reply, _begin_targets
from training.train import weighted_loss

logger = logging.getLogger("s2_weekend")

C_WEIGHTS = {"gold": 0.40, "exit": 0.35, "corner": 0.25}
HOLDOUT_EVAL = 32
MAX_NEW_TOKENS = 64

SAMPLE_FAMILY = {
    "A": "gaze/oracle",
    "B": "gaze/game",
    "C": "look_move",
    "D": "begin",
    "E": "retarget",
    "F": "anchor",
}

_REMEMBER = re.compile(r"\[REMEMBER target: ([^\]]*)\]")


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


def build_train_family(
    family: str,
    rng: random.Random,
    tmp: Path,
    step: int,
    place_image: Callable[[], dict[str, Any]],
) -> Batch:
    """One training image. ``place_image`` is the coord-tf four-direction deal."""
    if family == "A":
        return build_gaze(rng, place_image, mode="train")
    if family == "B":
        return build_gaze_game(rng, tmp, step, n_turns=3, question=None)
    if family == "C":
        return build_look_move(rng, tmp, step, force_kind=None)
    if family == "D":
        return build_begin(rng, tmp, step, board=None, limit_one=False)
    if family == "E":
        return build_retarget(rng, tmp, step, sub=None, board=None)
    raise ValueError(f"no board family {family!r}")


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
    """Notepad, one look round, then hour (2/3) or direction questions."""
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
    """Step-0 / periodic greedy eval. Writes summary.json and samples.jsonl."""
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
        samples.extend(_eval_gaze(trainer, rng, tmp))
        samples.extend(_eval_gaze_game(trainer, rng, tmp))
        samples.extend(_eval_look(trainer, rng, tmp))
        samples.extend(_eval_begin(trainer, rng, tmp))
        samples.extend(_eval_retarget(trainer, rng, tmp))
        anchor = _eval_anchor(trainer)
    finally:
        trainer.embedder.clear_pending()
        shutil.rmtree(tmp, ignore_errors=True)
    summary = _summarize(int(trainer.step), samples, anchor)
    _write_eval(out_dir, summary, samples)
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
        "family": SAMPLE_FAMILY[batch.family],
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
    if turn.task == "hour":
        parsed = parse_reply(KIND_HOUR, text)
        scores["hour_parse"] = parsed is not None
        scores["hour_exact"] = parsed == turn.extra["hour"]
        scores["hour_reward"] = (
            0.0 if parsed is None else hour_reward(int(turn.extra["hour"]), int(parsed))
        )
    if turn.task == "direction":
        scores["direction_exact"] = text.strip() == turn.reply.strip()
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
    return scores


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
    return scores


def _summarize(
    step: int, samples: list[dict[str, Any]], anchor: dict[str, float],
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    metrics: dict[str, float] = {}
    rate_keys = {
        "gaze/oracle": ("hour_parse", "hour_exact", "direction_exact"),
        "gaze/game": ("hour_parse", "hour_exact", "direction_exact"),
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
    }
    mean_keys = {
        "gaze/oracle": ("hour_reward",),
        "gaze/game": ("hour_reward",),
    }
    rms_keys = {
        "gaze/game": ("v_dist",),
        "look_move": ("v_dist",),
        "begin": ("v_dist",),
    }
    prefix = {
        "gaze/oracle": "A_",
        "gaze/game": "B_",
        "look_move": "C_",
        "begin": "D_",
        "retarget": "E_",
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
    groups["anchor"] = {"ce": {"n": int(anchor["n"]), "mean": anchor["mean"]}}
    metrics["F_ce"] = float(anchor["mean"])
    metrics["F_ce_n"] = float(anchor["n"])
    return {"step": step, "groups": groups, "metrics": metrics}


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
    hist_user = _ask("look", chosen[0][1])
    context = _append_round(ctx.context, hist_user, rng.choice(_S2_LOOK_LINES))
    turns = []
    for _kind, _phrase, point, (axis, label, hour) in chosen:
        if question == "hour":
            reply = correct_line(KIND_HOUR, hour)
            task = "hour"
        else:
            reply = correct_line(axis, label)
            task = "direction"
        turns.append(Turn(
            question=asked,
            reply=reply,
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
        context=context,
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
