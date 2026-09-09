"""Planted-board player / analyst smoke for the HF path on master.

Two commands, launch separately:

    python -m training.infer_situations check
    python -m training.infer_situations speed --modes control,bc

``check`` plants a few sealed one-gold boards, runs the real
``InteractiveSelfEvalSession`` player and analyst paths, and prints the
replies. Protocol failures (no move token on a move question; missing
RATING / TARGET on analyst) exit nonzero. Oracle match is printed, not
asserted -- sampled replies are allowed to be wrong.

``speed`` times the same calls under ``GS_PREFIX_KV=0`` (control) vs
production B+C. One process, one HF load; flags flip at generate time.
This is a smoke timing, not the weekend estimate -- that is still
``python -m training.bench_speed --what infer --modes control,bc``.

Remote GPU + NAMS. Does not wipe the graph (no ``reset_memory_to_seed``).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Four full side walls, same geometry as discreteEngine.random_side_walls
# with no exit (exit_wall = -1).
_WALL_T = 50.0 / 800.0
_SEALED_WALLS = [
    [0.0, 0.0, _WALL_T, 1.0, 0.0],
    [0.0, 0.0, 1.0, _WALL_T, 0.0],
    [0.0, 1.0 - _WALL_T, 1.0, _WALL_T, 0.0],
    [1.0 - _WALL_T, 0.0, _WALL_T, 1.0, 0.0],
]


def _board(gold: list[float], direction: float = 0.0) -> dict[str, Any]:
    return {
        "gameSize": 768,
        "direction": direction,
        "agent_x": 0.5,
        "agent_y": 0.5,
        "agent_r": 0.05,
        "gold_r": 1.0 / 64,
        "gold": [list(gold)],
        "walls": [list(w) for w in _SEALED_WALLS],
    }


# Gold placements match training/selftest.py t1 oracle fixtures
# (facing theta=0 = 12 o'clock).
GOLD_AHEAD = _board([0.5, 0.8])
GOLD_RIGHT = _board([0.8, 0.5])
GOLD_LEFT = _board([0.2, 0.5])


def _parse_cli(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="what", required=True)
    check = sub.add_parser("check", help="print player/analyst replies")
    speed = sub.add_parser("speed", help="time the same calls, control vs bc")
    for sp in (check, speed):
        sp.add_argument(
            "--checkpoint", default=None,
            help="adapter folder under weights/<model_key>/; default is "
                 "MODEL_CHECKPOINT",
        )
        sp.add_argument(
            "--greedy", action="store_true",
            help="MODEL_DO_SAMPLE=0 (check: more readable; speed: less "
                 "length noise)",
        )
    speed.add_argument(
        "--modes", default="control,bc",
        help="comma-separated infer flag sets (default control,bc)",
    )
    speed.add_argument(
        "--repeat", type=int, default=1,
        help="repeat the situation suite this many times per mode",
    )
    return p.parse_args(argv)


def _apply_cli_env(args: argparse.Namespace) -> None:
    """Must run before agent.config / get_model import CONFIG."""
    if args.greedy:
        os.environ["MODEL_DO_SAMPLE"] = "0"
    if args.checkpoint:
        os.environ["MODEL_CHECKPOINT"] = args.checkpoint


def _import_runtime():
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
    from agent import game_io
    from agent.model import get_model, set_default_checkpoint
    from agent.modes import DEFAULT_ANALYST_QUESTION, DEFAULT_PLAYER_QUESTION
    from agent.self_eval_session import InteractiveSelfEvalSession
    from training.game_traces import oracle_verdict
    from training.generate_game_traces import _oracle_meta
    return {
        "game_io": game_io,
        "get_model": get_model,
        "set_default_checkpoint": set_default_checkpoint,
        "DEFAULT_ANALYST_QUESTION": DEFAULT_ANALYST_QUESTION,
        "DEFAULT_PLAYER_QUESTION": DEFAULT_PLAYER_QUESTION,
        "InteractiveSelfEvalSession": InteractiveSelfEvalSession,
        "oracle_verdict": oracle_verdict,
        "oracle_meta": _oracle_meta,
    }


def _plant(session: Any, board: dict[str, Any], rt: dict) -> dict[str, Any]:
    session.restart()
    session.game = rt["game_io"].game_from_settings_dict(board)
    return rt["game_io"].game_to_settings_dict(session.game)


def _fmt_oracle(meta: dict[str, Any]) -> str:
    return (
        f"{meta.get('oracle_move')}  "
        f"rel_bearing={meta.get('oracle_rel_bearing')}  "
        f"ray_hit={meta.get('oracle_ray_hit')}"
    )


def _print_block(title: str, body: str) -> None:
    print()
    print(f"======== {title} ========")
    print(body.rstrip() or "(empty)")
    print(f"======== end {title} ========")


def run_check(rt: dict) -> int:
    q_player = rt["DEFAULT_PLAYER_QUESTION"]
    q_analyst = rt["DEFAULT_ANALYST_QUESTION"]
    q_facing = "Are you facing the gold? Answer in one or two sentences."
    Session = rt["InteractiveSelfEvalSession"]
    session = Session(log_label="infer_situations_check")
    failures: list[str] = []
    try:
        player_cases = [
            ("player gold_ahead", GOLD_AHEAD, q_player, True),
            ("player gold_right", GOLD_RIGHT, q_player, True),
            ("player gold_left", GOLD_LEFT, q_player, True),
            ("player facing_q (gold_ahead)", GOLD_AHEAD, q_facing, False),
        ]
        for title, board, question, need_move in player_cases:
            settings = _plant(session, board, rt)
            meta = rt["oracle_meta"](settings, {"kind": "gold", "index": 0})
            print()
            print(f"######## {title} ########")
            print(
                f"board: agent ({settings['agent_x']:.2f}, "
                f"{settings['agent_y']:.2f}) facing {settings['direction']:.2f}; "
                f"gold {settings['gold'][0]}"
            )
            print(f"oracle: {_fmt_oracle(meta)}")
            print(f"question: {question}")
            out = session.ask_player(question)
            raw = out.get("raw") or ""
            action = out.get("action")
            count = out.get("turn_count")
            _print_block(f"{title} reply", raw)
            verdict = rt["oracle_verdict"](
                action, meta.get("oracle_move"),
                meta.get("oracle_rel_bearing"), meta.get("oracle_ray_hit"),
                count,
            )
            print(
                f"parse: action={action} count={count}  "
                f"oracle_verdict={verdict}"
            )
            if need_move and action is None:
                failures.append(f"{title}: no parseable move token")
            if not raw.strip():
                failures.append(f"{title}: empty reply")

        analyst_cases = [
            (
                "analyst on planted FORWARD (gold_ahead)",
                GOLD_AHEAD,
                "The gold is straight ahead.\n[FORWARD]",
            ),
            (
                "analyst on planted ANTICLOCK 30 (gold_ahead, wrong)",
                GOLD_AHEAD,
                "I will turn away from the gold.\n[ANTICLOCK 30]",
            ),
        ]
        for title, board, planted in analyst_cases:
            settings = _plant(session, board, rt)
            meta = rt["oracle_meta"](settings, {"kind": "gold", "index": 0})
            print()
            print(f"######## {title} ########")
            print(f"oracle: {_fmt_oracle(meta)}")
            print(f"planted player reply:\n{planted}")
            session.ask_player(q_player, human_reply=planted)
            out = session.ask_analyst(q_analyst)
            analysis = out.get("analysis") or ""
            rating = out.get("rating")
            target = out.get("target")
            _print_block(f"{title} analysis", analysis)
            print(f"parse: RATING={rating}  TARGET={target}")
            if rating is None:
                failures.append(f"{title}: missing RATING line")
            if target is None:
                failures.append(f"{title}: missing TARGET line")
            if not analysis.strip():
                failures.append(f"{title}: empty analysis")
    finally:
        session.close()

    print()
    if failures:
        print(f"CHECK FAIL ({len(failures)} protocol):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("CHECK PASS: every move question parsed a move; every analyst "
          "emitted RATING and TARGET. Read the replies above for sense.")
    return 0


def _set_infer_mode(mode: str) -> None:
    if mode == "control":
        os.environ["GS_PREFIX_KV"] = "0"
        os.environ.pop("GS_PAD_PARITY", None)
    elif mode == "bc":
        os.environ["GS_PREFIX_KV"] = "1"
        os.environ.pop("GS_PAD_PARITY", None)
    else:
        raise SystemExit(
            f"unknown speed mode {mode!r} -- want control or bc "
            "(sglang is the other branch)"
        )


def _clear_prefix(model: Any) -> None:
    if getattr(model, "_prefix_kv", None) is not None:
        model._prefix_kv.clear()
        model._prefix_kv_len.clear()


def _run_suite(session: Any, rt: dict) -> list[dict[str, Any]]:
    q_player = rt["DEFAULT_PLAYER_QUESTION"]
    q_analyst = rt["DEFAULT_ANALYST_QUESTION"]
    rows: list[dict[str, Any]] = []
    timed = [
        ("player", "gold_ahead", GOLD_AHEAD, "player", None),
        ("player", "gold_right", GOLD_RIGHT, "player", None),
        ("analyst", "planted_forward", GOLD_AHEAD, "analyst",
         "The gold is straight ahead.\n[FORWARD]"),
    ]
    for role, name, board, kind, planted in timed:
        _plant(session, board, rt)
        t0 = time.perf_counter()
        if kind == "player":
            session.ask_player(q_player)
        else:
            session.ask_player(q_player, human_reply=planted)
            session.ask_analyst(q_analyst)
        dt = time.perf_counter() - t0
        rows.append({"role": role, "situation": name, "s": round(dt, 2)})
    return rows


def run_speed(rt: dict, modes: list[str], repeat: int) -> int:
    from agent.model import get_model

    model = get_model()
    Session = rt["InteractiveSelfEvalSession"]
    session = Session(log_label="infer_situations_speed", load_model=False)
    session.model = model
    try:
        print("warmup (untimed player gold_ahead)...")
        _set_infer_mode(modes[0])
        _clear_prefix(model)
        _plant(session, GOLD_AHEAD, rt)
        session.ask_player(rt["DEFAULT_PLAYER_QUESTION"])

        all_rows: list[dict[str, Any]] = []
        for mode in modes:
            _set_infer_mode(mode)
            _clear_prefix(model)
            for i in range(repeat):
                print(f"timing mode={mode} pass={i + 1}/{repeat}...")
                for r in _run_suite(session, rt):
                    all_rows.append({"mode": mode, **r})

        keys = ["mode", "role", "situation", "s"]
        widths = {
            k: max(len(k), *(len(str(r.get(k, ""))) for r in all_rows))
            for k in keys
        }
        print()
        print("  ".join(k.ljust(widths[k]) for k in keys))
        print("  ".join("-" * widths[k] for k in keys))
        for r in all_rows:
            print("  ".join(str(r.get(k, "")).ljust(widths[k]) for k in keys))

        print()
        for mode in modes:
            subset = [r for r in all_rows if r["mode"] == mode]
            if not subset:
                continue
            mean = sum(r["s"] for r in subset) / len(subset)
            print(f"{mode}: n={len(subset)}  mean_s/call={mean:.2f}")
        print()
        print("Smoke timing only. Weekend-shaped estimate:")
        print("  python -m training.bench_speed --what infer --modes control,bc")
    finally:
        session.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_cli(argv)
    _apply_cli_env(args)
    os.chdir(REPO_ROOT)
    rt = _import_runtime()
    if args.checkpoint:
        rt["set_default_checkpoint"](args.checkpoint)
    if args.what == "check":
        return run_check(rt)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes:
        raise SystemExit("--modes is empty")
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    return run_speed(rt, modes, args.repeat)


if __name__ == "__main__":
    raise SystemExit(main())
