"""Behavioral analyst gate: 12 planted boards, one generate_batch.

The held-out analyst KD meter missed the step332 TARGET-missing regression
(0.2123 -> 0.2131 while missing-TARGET went 2.4% -> 11.9%). This gate
measures grading behavior directly. It does NOT pin the analyst.

    python -m training.analyst_gate --checkpoint <name>

One InteractiveSelfEvalSession (NAMS scene prompts) + exactly one
12-wide ``generate_batch``. Budget: model load ~5 min + one analyst
wave ~2 min. Exit 0 pass / 1 fail. Verdict JSON under
``logs/analyst_gate_<stamp>/`` (and ``--out`` if given).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from agent import game_io  # noqa: E402
from agent.memory import format_notepad  # noqa: E402
from agent.model import get_model, set_default_checkpoint  # noqa: E402
from agent.modes import (  # noqa: E402
    DEFAULT_ANALYST_QUESTION,
    DEFAULT_PLAYER_QUESTION,
    WRONG_SPAN_RE,
    _without_analyst_line_tags,
    build_scene_analyst_messages,
    parse_rating,
    parse_target,
    parse_wrong_spans,
)
from agent.self_eval_session import InteractiveSelfEvalSession  # noqa: E402
from training.generate_game_traces import _oracle_meta  # noqa: E402
from training.infer_situations import GOLD_AHEAD, GOLD_RIGHT, _board  # noqa: E402

_ANALYST_NEW = 700
_QUESTION_FACING = (
    "Are you facing the gold? Answer in one or two sentences."
)
_REMEMBER_LINE = "[REMEMBER target: the gold ahead]"
_NOTEPAD_FILLED = format_notepad([
    {"key": "target", "value": "the gold ahead", "updated_round": 1},
])
_NOTEPAD_EMPTY = format_notepad([])

GOLD_AHEAD_E = _board([0.8, 0.5], direction=math.pi / 2)
GOLD_BEHIND = _board([0.5, 0.2])


@dataclass(frozen=True)
class Case:
    """One planted analyst prompt and the checks it must satisfy."""

    id: str
    board: dict[str, Any]
    reply: str
    question: str = DEFAULT_PLAYER_QUESTION
    notepad: str | None = None
    rating_ge: float | None = None
    rating_lt: float | None = None
    rating_max: float | None = None
    expect_target_none: bool = False
    require_remember_wrong: bool = False
    forbid_remember_wrong: bool = False
    is_sign: bool = False
    is_resave: bool = False


def _cases() -> list[Case]:
    return [
        Case(
            "rayhit_n_forward", GOLD_AHEAD,
            "The gold is straight ahead.\n[FORWARD]",
            rating_ge=0.0, is_sign=True,
        ),
        Case(
            "rayhit_e_forward", GOLD_AHEAD_E,
            "The gold is dead ahead of the eye.\n[FORWARD]",
            rating_ge=0.0, is_sign=True,
        ),
        Case(
            "rayhit_n_anticlock5", GOLD_AHEAD,
            "I will turn a little.\n[ANTICLOCK 5]",
            rating_lt=0.0, is_sign=True,
        ),
        Case(
            "rayhit_e_clock5", GOLD_AHEAD_E,
            "I will turn a little.\n[CLOCK 5]",
            rating_lt=0.0, is_sign=True,
        ),
        Case(
            "right90_clock15", GOLD_RIGHT,
            "The gold is about 90 degrees clockwise. Fifteen steps.\n"
            "[CLOCK 15]",
            rating_ge=0.0, is_sign=True,
        ),
        Case(
            "right90_anticlock15", GOLD_RIGHT,
            "I will turn the other way.\n[ANTICLOCK 15]",
            rating_lt=0.0, is_sign=True,
        ),
        Case(
            "right90_clock2", GOLD_RIGHT,
            "A small turn should be enough.\n[CLOCK 2]",
            rating_max=0.3,
        ),
        Case(
            "behind_forward", GOLD_BEHIND,
            "Walking forward.\n[FORWARD]",
            rating_lt=0.0, is_sign=True,
        ),
        Case(
            "resave_target", GOLD_AHEAD,
            "Still the same gold.\n"
            f"{_REMEMBER_LINE}\n[FORWARD]",
            notepad=_NOTEPAD_FILLED,
            rating_max=0.5,
            require_remember_wrong=True,
            is_resave=True,
        ),
        Case(
            "first_commit", GOLD_AHEAD,
            "I see gold ahead. Committing to it.\n"
            f"{_REMEMBER_LINE}\n[FORWARD]",
            notepad=_NOTEPAD_EMPTY,
            rating_ge=0.0,
            forbid_remember_wrong=True,
            is_sign=True,
        ),
        Case(
            "question_round", GOLD_AHEAD,
            "Yes, the gold is straight ahead of me.",
            question=_QUESTION_FACING,
            rating_ge=0.0,
            expect_target_none=True,
            is_sign=True,
        ),
        Case(
            "fake_forward10", GOLD_AHEAD,
            "The gold is straight ahead.\n[FORWARD 10]",
            rating_lt=0.0, is_sign=True,
        ),
    ]


def _pending_action(reply: str) -> str | None:
    parsed = game_io.parse_move(reply)
    if parsed is None:
        return None
    action, count = parsed
    return game_io.format_move_inner(action, count)


def _remember_wrong(analysis: str, reply: str) -> bool:
    """True when a WRONG line quotes a REMEMBER write from ``reply``."""
    parsed = parse_wrong_spans(analysis, reply)
    spans = list(parsed["verified"]) + list(parsed["unverified"])
    tagged = _without_analyst_line_tags(analysis)
    for m in WRONG_SPAN_RE.finditer(tagged):
        span = (m.group("quoted") or m.group("span") or "").strip()
        if span:
            spans.append(span)
    return any("REMEMBER" in s.upper() for s in spans)


def _fmt_target(target: dict[str, Any] | None) -> str:
    if target is None:
        return "—"
    if target.get("kind") == "none":
        return "none"
    return f"{target['kind']},{target.get('index')}"


def _check_oracle_geometry() -> None:
    """Fail before the GPU load if a planted board is the wrong shape."""
    ahead_n = _oracle_meta(GOLD_AHEAD, {"kind": "gold", "index": 0})
    ahead_e = _oracle_meta(GOLD_AHEAD_E, {"kind": "gold", "index": 0})
    right = _oracle_meta(GOLD_RIGHT, {"kind": "gold", "index": 0})
    behind = _oracle_meta(GOLD_BEHIND, {"kind": "gold", "index": 0})
    problems: list[str] = []
    if not ahead_n.get("oracle_ray_hit") or ahead_n.get("oracle_move") != "FORWARD":
        problems.append(f"GOLD_AHEAD not ray-hit FORWARD: {ahead_n}")
    if not ahead_e.get("oracle_ray_hit") or ahead_e.get("oracle_move") != "FORWARD":
        problems.append(f"GOLD_AHEAD_E not ray-hit FORWARD: {ahead_e}")
    if right.get("oracle_move") != "CLOCK":
        problems.append(f"GOLD_RIGHT oracle_move={right.get('oracle_move')}")
    steps = abs(float(right["oracle_rel_bearing"])) / (math.pi / 30)
    if abs(steps - 15) > 0.5:
        problems.append(f"GOLD_RIGHT steps={steps:.2f}, want ~15")
    if behind.get("oracle_ray_hit") or behind.get("oracle_move") == "FORWARD":
        problems.append(f"GOLD_BEHIND should not be FORWARD: {behind}")
    if problems:
        raise RuntimeError(
            "analyst_gate planted geometry is wrong:\n  " + "\n  ".join(problems)
        )


def _plant(session: Any, board: dict[str, Any]) -> dict[str, Any]:
    session.restart()
    session.game = game_io.game_from_settings_dict(board)
    return game_io.game_to_settings_dict(session.game)


def _build_prompts(
    session: Any, cases: list[Case], frames_dir: Path,
) -> tuple[list[list[dict]], list[dict[str, Any]]]:
    """Plant each board, copy its frame, assemble analyst messages."""
    batch: list[list[dict]] = []
    metas: list[dict[str, Any]] = []
    for case in cases:
        settings = _plant(session, board=case.board)
        src = Path(session.current_frame_path())
        dest = frames_dir / f"{case.id}.png"
        shutil.copy2(src, dest)
        oracle = _oracle_meta(settings, {"kind": "gold", "index": 0})
        messages = build_scene_analyst_messages(
            case.question,
            case.reply,
            _pending_action(case.reply),
            str(dest),
            json.dumps(settings),
            "",
            DEFAULT_ANALYST_QUESTION,
            system_prompt=session.ANALYST_SYSTEM_PROMPT,
            player_notepad=case.notepad,
        )
        batch.append(messages)
        metas.append({"settings": settings, "oracle": oracle, "frame": str(dest)})
    return batch, metas


def _evaluate(
    case: Case, analysis: str,
) -> tuple[dict[str, Any], list[str], bool]:
    """Return (row, hard_failures, numeric_miss)."""
    rating = parse_rating(analysis)
    target = parse_target(analysis)
    remember_wrong = _remember_wrong(analysis, case.reply)
    hard: list[str] = []
    numeric_miss = False

    if rating is None:
        hard.append("RATING unparseable")
    else:
        if case.rating_ge is not None and rating < case.rating_ge:
            msg = f"rating {rating} < {case.rating_ge}"
            if case.is_sign:
                numeric_miss = True
            else:
                hard.append(msg)
        if case.rating_lt is not None and not (rating < case.rating_lt):
            msg = f"rating {rating} not < {case.rating_lt}"
            if case.is_sign:
                numeric_miss = True
            else:
                hard.append(msg)
        if case.rating_max is not None and rating > case.rating_max:
            msg = f"rating {rating} > {case.rating_max}"
            if case.is_resave:
                hard.append(msg)
            else:
                numeric_miss = True

    if case.expect_target_none:
        if target is None or target.get("kind") != "none":
            hard.append(f"TARGET want none, got {_fmt_target(target)}")
    if case.require_remember_wrong and not remember_wrong:
        hard.append("re-save was not flagged with a WRONG line on REMEMBER")
    if case.forbid_remember_wrong and remember_wrong:
        hard.append("first-commit REMEMBER was marked WRONG")

    row = {
        "id": case.id,
        "rating": rating,
        "target": target,
        "remember_wrong": remember_wrong,
        "numeric_miss": numeric_miss,
        "hard_failures": hard,
    }
    return row, hard, numeric_miss


def _print_table(rows: list[dict[str, Any]]) -> None:
    print()
    print(f"{'id':24s} {'rating':8s} {'target':12s} "
          f"{'num':5s} {'remW':5s} result")
    print("-" * 72)
    for row in rows:
        rating = row["rating"]
        rtxt = f"{rating:+.2f}" if rating is not None else "—"
        ntxt = "MISS" if row["numeric_miss"] else "ok"
        wtxt = "YES" if row["remember_wrong"] else "no"
        ok = not row["hard_failures"] and not row["numeric_miss"]
        print(
            f"{row['id']:24s} {rtxt:8s} {_fmt_target(row['target']):12s} "
            f"{ntxt:5s} {wtxt:5s} {'PASS' if ok else 'FAIL'}"
        )
        for f in row["hard_failures"]:
            print(f"  - {f}")


def _parse_cli(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--checkpoint", default=None,
        help="adapter under weights/<model_key>/; default MODEL_CHECKPOINT",
    )
    p.add_argument(
        "--out", default=None,
        help="also write the compact verdict JSON to this path",
    )
    return p.parse_args(argv)


def run_gate(
    checkpoint: str | None,
    out_path: Path | None = None,
    log: Callable[[str], None] = print,
) -> int:
    _check_oracle_geometry()
    os.environ["MODEL_DO_SAMPLE"] = "0"
    set_default_checkpoint(checkpoint)

    stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = REPO_ROOT / "logs" / f"analyst_gate_{stamp}"
    replies_dir = run_dir / "replies"
    frames_dir = run_dir / "frames"
    replies_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    cases = _cases()
    if len(cases) != 12:
        raise RuntimeError(f"analyst_gate wants 12 cases, got {len(cases)}")

    session = InteractiveSelfEvalSession(
        log_label="analyst_gate", enable_logging=False, load_model=False,
    )
    try:
        session.model = get_model()
        model = session.model
        prompts, metas = _build_prompts(session, cases, frames_dir)
        batch = [{"messages": m} for m in prompts]
        log(f"analyst_gate: checkpoint={checkpoint!r}  "
            f"batch={len(batch)}  max_new_tokens={_ANALYST_NEW}")
        analyses = model.generate_batch(batch, max_new_tokens=_ANALYST_NEW)
    finally:
        session.close()

    if len(analyses) != 12:
        raise RuntimeError(
            f"generate_batch returned {len(analyses)} replies, want 12"
        )

    rows: list[dict[str, Any]] = []
    hard_all: list[str] = []
    numeric_misses: list[str] = []
    n_rating = 0
    n_target = 0
    resave_ok = False
    for case, analysis, meta in zip(cases, analyses, metas):
        text = analysis if analysis.endswith("\n") else analysis + "\n"
        (replies_dir / f"{case.id}.txt").write_text(text, encoding="utf-8")
        row, hard, nmiss = _evaluate(case, analysis)
        row["oracle"] = meta["oracle"]
        rows.append(row)
        if row["rating"] is not None:
            n_rating += 1
        if row["target"] is not None:
            n_target += 1
        for h in hard:
            hard_all.append(f"{case.id}: {h}")
        if nmiss:
            numeric_misses.append(case.id)
        if case.is_resave:
            resave_ok = (
                row["remember_wrong"]
                and row["rating"] is not None
                and row["rating"] <= 0.5
            )

    _print_table(rows)
    rating_ok = n_rating == 12
    target_ok = n_target >= 11
    sign_ok = len(numeric_misses) <= 1
    passed = rating_ok and target_ok and sign_ok and resave_ok and not hard_all

    verdict = {
        "passed": passed,
        "checkpoint": checkpoint,
        "n_rating_parseable": n_rating,
        "n_target_parseable": n_target,
        "numeric_misses": numeric_misses,
        "resave_flagged": resave_ok,
        "hard_failures": hard_all,
        "cases": [
            {
                "id": r["id"],
                "rating": r["rating"],
                "target": r["target"],
                "remember_wrong": r["remember_wrong"],
                "numeric_miss": r["numeric_miss"],
                "hard_failures": r["hard_failures"],
                "oracle_move": (r["oracle"] or {}).get("oracle_move"),
                "oracle_ray_hit": (r["oracle"] or {}).get("oracle_ray_hit"),
            }
            for r in rows
        ],
    }
    verdict_path = run_dir / "verdict.json"
    verdict_path.write_text(
        json.dumps(verdict, indent=2) + "\n", encoding="utf-8",
    )
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(verdict, indent=2) + "\n", encoding="utf-8",
        )

    log("")
    log(f"verdict: {verdict_path}")
    if out_path is not None:
        log(f"verdict copy: {out_path}")
    log(
        f"RATING {n_rating}/12  TARGET {n_target}/12  "
        f"numeric_misses={numeric_misses}  resave_flagged={resave_ok}"
    )
    if passed:
        log("ANALYST GATE PASS")
        return 0
    log("ANALYST GATE FAIL")
    if not rating_ok:
        log("  - RATING must parse on 12/12")
    if not target_ok:
        log("  - TARGET must parse on >= 11/12")
    if not sign_ok:
        log("  - at most 1 miss across the sign/bound cases")
    if not resave_ok:
        log("  - re-save case must be flagged (WRONG on REMEMBER, "
            "rating <= 0.5)")
    for h in hard_all:
        log(f"  - {h}")
    return 1


def main(argv: list[str] | None = None) -> int:
    args = _parse_cli(argv)
    out = Path(args.out) if args.out else None
    return run_gate(args.checkpoint, out_path=out)


if __name__ == "__main__":
    raise SystemExit(main())
