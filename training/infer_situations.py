"""Show B+C (prefix-KV + leftover left-pad) and write every reply to a file.

    python -m training.infer_situations          # check + speed  (~10-25 min)
    python -m training.infer_situations check    # replies only   (~3-8 min)
    python -m training.infer_situations speed    # leftover-pad vs prefix-KV

The first line of output is the report path. Full player/analyst text
lives there (and under ``replies/``). The terminal keeps the table.

Datagen was already mixed-length and parallel. The overhaul number is
**bc vs control**, not vs serial:

* **control** — ``generate_batch`` leftover left-pad (C). That is the
  old parallel path: dispatcher + mixed-length pad, no resident prefix.
* **bc** — the same batch with resident prefix-KV (B+C).
* **serial** — N solo ``generate`` calls. Diagnostic only (did the
  batch actually beat N solos?). Not the pre-overhaul datagen baseline.

Mixed token lengths are chosen at encode time so the pad / prefix-KV
path must engage (same tripwire as t6). Greedy decode on the timed
waves so control vs bc is the same work. Check uses production
sampling unless ``--greedy``.

Budget (from the ~3.5 s player / ~14 s analyst you already measured):
6 serial player + 3 serial analyst + two batched modes + 6 live check
generates, plus one 12B load. Target well under 30 min, hard cap 1 h.

Remote GPU + NAMS. Does not wipe the graph.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_WALL_T = 50.0 / 800.0
_SEALED_WALLS = [
    [0.0, 0.0, _WALL_T, 1.0, 0.0],
    [0.0, 0.0, 1.0, _WALL_T, 0.0],
    [0.0, 1.0 - _WALL_T, 1.0, _WALL_T, 0.0],
    [1.0 - _WALL_T, 0.0, _WALL_T, 1.0, 0.0],
]

# Widths: enough to show a batch, small enough for the 1 h cap.
# 6 × ~3.5 s serial player + 3 × ~14 s serial analyst ≈ 63 s of solo
# decode; batched modes should be a fraction of that (the point).
_PLAYER_N = 6
_ANALYST_N = 3
_PLAYER_NEW = 128
_ANALYST_NEW = 256


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


GOLD_AHEAD = _board([0.5, 0.8])
GOLD_RIGHT = _board([0.8, 0.5])
GOLD_LEFT = _board([0.2, 0.5])


class _Tee:
    def __init__(self, path: Path, real: TextIO):
        self._f = path.open("w", encoding="utf-8")
        self._real = real

    def write(self, s: str) -> int:
        self._real.write(s)
        self._f.write(s)
        self._f.flush()
        return len(s)

    def flush(self) -> None:
        self._real.flush()
        self._f.flush()

    def close(self) -> None:
        self._f.close()


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def _parse_cli(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--checkpoint", default=None,
        help="adapter under weights/<model_key>/; default MODEL_CHECKPOINT",
    )
    p.add_argument(
        "--greedy", action="store_true",
        help="MODEL_DO_SAMPLE=0 for the check session (speed is always greedy)",
    )
    sub = p.add_subparsers(dest="what")
    sub.add_parser("check", help="planted player/analyst replies")
    sub.add_parser("speed", help="leftover-pad (control) vs prefix-KV (bc)")
    sub.add_parser("all", help="check then speed (default)")
    p.set_defaults(what="all")
    return p.parse_args(argv)


def _apply_cli_env(args: argparse.Namespace) -> None:
    if args.greedy:
        os.environ["MODEL_DO_SAMPLE"] = "0"
    if args.checkpoint:
        os.environ["MODEL_CHECKPOINT"] = args.checkpoint


def _import_runtime() -> dict[str, Any]:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
    from agent import game_io
    from agent.model import (
        PAD_POISON_MOD,
        PAD_POISON_RESIDUE,
        get_model,
        set_default_checkpoint,
    )
    from agent.modes import (
        DEFAULT_ANALYST_QUESTION,
        DEFAULT_PLAYER_QUESTION,
        _build_game_messages,
        build_scene_analyst_messages,
    )
    from agent.self_eval_session import InteractiveSelfEvalSession
    from training.game_traces import oracle_verdict
    from training.generate_game_traces import _oracle_meta
    return {
        "game_io": game_io,
        "PAD_POISON_MOD": PAD_POISON_MOD,
        "PAD_POISON_RESIDUE": PAD_POISON_RESIDUE,
        "get_model": get_model,
        "set_default_checkpoint": set_default_checkpoint,
        "DEFAULT_ANALYST_QUESTION": DEFAULT_ANALYST_QUESTION,
        "DEFAULT_PLAYER_QUESTION": DEFAULT_PLAYER_QUESTION,
        "_build_game_messages": _build_game_messages,
        "build_scene_analyst_messages": build_scene_analyst_messages,
        "InteractiveSelfEvalSession": InteractiveSelfEvalSession,
        "oracle_verdict": oracle_verdict,
        "oracle_meta": _oracle_meta,
    }


def _open_report() -> tuple[Path, Path, _Tee]:
    stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = REPO_ROOT / "logs" / f"infer_situations_{stamp}"
    replies = run_dir / "replies"
    replies.mkdir(parents=True, exist_ok=True)
    report = run_dir / "report.txt"
    tee = _Tee(report, sys.stdout)
    sys.stdout = tee  # type: ignore[assignment]
    return run_dir, replies, tee


def _announce(run_dir: Path) -> None:
    report = run_dir / "report.txt"
    print(f"REPORT FILE (every reply is here):")
    print(f"  {report}")
    print(f"Individual replies:")
    print(f"  {run_dir / 'replies'}")
    print(
        "If the terminal scrolled away, open that report.txt — "
        "do not hunt the scrollback."
    )
    print()


def _write_reply(replies_dir: Path, name: str, text: str) -> None:
    path = replies_dir / f"{name}.txt"
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


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


def _make_session(rt: dict, label: str, model: Any | None) -> Any:
    Session = rt["InteractiveSelfEvalSession"]
    session = Session(
        log_label=label, enable_logging=False, load_model=model is None,
    )
    if model is not None:
        session.model = model
    return session


def run_check(rt: dict, replies_dir: Path, model: Any | None) -> int:
    q_player = rt["DEFAULT_PLAYER_QUESTION"]
    q_analyst = rt["DEFAULT_ANALYST_QUESTION"]
    q_facing = "Are you facing the gold? Answer in one or two sentences."
    session = _make_session(rt, "infer_situations_check", model)
    failures: list[str] = []
    try:
        player_cases = [
            ("player_gold_ahead", GOLD_AHEAD, q_player, True),
            ("player_gold_right", GOLD_RIGHT, q_player, True),
            ("player_gold_left", GOLD_LEFT, q_player, True),
            ("player_facing_q", GOLD_AHEAD, q_facing, False),
        ]
        for name, board, question, need_move in player_cases:
            settings = _plant(session, board, rt)
            meta = rt["oracle_meta"](settings, {"kind": "gold", "index": 0})
            print()
            print(f"######## check {name} ########")
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
            _write_reply(replies_dir, f"check_{name}", raw)
            _print_block(f"check {name} reply", raw)
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
                failures.append(f"{name}: no parseable move token")
            if not raw.strip():
                failures.append(f"{name}: empty reply")

        analyst_cases = [
            (
                "analyst_planted_forward",
                GOLD_AHEAD,
                "The gold is straight ahead.\n[FORWARD]",
            ),
            (
                "analyst_planted_wrong",
                GOLD_AHEAD,
                "I will turn away from the gold.\n[ANTICLOCK 30]",
            ),
        ]
        for name, board, planted in analyst_cases:
            settings = _plant(session, board, rt)
            meta = rt["oracle_meta"](settings, {"kind": "gold", "index": 0})
            print()
            print(f"######## check {name} ########")
            print(f"oracle: {_fmt_oracle(meta)}")
            print(f"planted player reply:\n{planted}")
            session.ask_player(q_player, human_reply=planted)
            out = session.ask_analyst(q_analyst)
            analysis = out.get("analysis") or ""
            rating = out.get("rating")
            target = out.get("target")
            _write_reply(replies_dir, f"check_{name}", analysis)
            _print_block(f"check {name} analysis", analysis)
            print(f"parse: RATING={rating}  TARGET={target}")
            if rating is None:
                failures.append(f"{name}: missing RATING line")
            if target is None:
                failures.append(f"{name}: missing TARGET line")
            if not analysis.strip():
                failures.append(f"{name}: empty analysis")
    finally:
        session.close()

    print()
    if failures:
        print(f"CHECK FAIL ({len(failures)} protocol):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("CHECK PASS: move questions parsed a move; analysts emitted "
          "RATING and TARGET.")
    return 0


def _set_infer_mode(mode: str) -> None:
    if mode == "control":
        os.environ["GS_PREFIX_KV"] = "0"
        os.environ.pop("GS_PAD_PARITY", None)
    elif mode == "bc":
        os.environ["GS_PREFIX_KV"] = "1"
        os.environ.pop("GS_PAD_PARITY", None)
    else:
        raise SystemExit(f"unknown speed mode {mode!r} -- want control or bc")


def _clear_prefix(model: Any) -> None:
    if getattr(model, "_prefix_kv", None) is not None:
        model._prefix_kv.clear()
        model._prefix_kv_len.clear()


def _force_greedy(model: Any):
    original = model._sampling_kwargs

    def _greedy() -> dict[str, Any]:
        return {"do_sample": False}

    model._sampling_kwargs = _greedy
    return original


def _path_tag(lines: list[str]) -> str:
    joined = " | ".join(lines)
    tags: list[str] = []
    if "prefix-kv batch" in joined:
        tags.append("prefix-kv")
    if "padded batch" in joined:
        tags.append("padded")
    if "POISON MODE 2 RESCUE" in joined:
        tags.append("nudge")
    if any("equal-length cohort" in ln for ln in lines) and not tags:
        tags.append("cohorts-only")
    return "+".join(tags) if tags else "no-batch-log"


def _generate_batch_logged(model: Any, batch: list[dict], **kwargs: Any):
    cap = _Capture()
    log = logging.getLogger("agent.model")
    log.addHandler(cap)
    prev = log.level
    log.setLevel(logging.INFO)
    try:
        replies = model.generate_batch(batch, **kwargs)
    finally:
        log.removeHandler(cap)
        log.setLevel(prev)
    return replies, cap.lines


def _pick_length_grid(
    model: Any,
    make_messages,
    n: int,
    rt: dict,
) -> list[list[dict]]:
    """Distinct paddable token lengths, encode-first (not char buckets)."""
    residue = rt["PAD_POISON_RESIDUE"]
    mod = rt["PAD_POISON_MOD"]
    by_len: dict[int, list[dict]] = {}
    for k in range(64):
        msgs = make_messages(k)
        length = int(model.encode_messages(msgs)["input_ids"].shape[1])
        if length % mod == residue:
            continue
        by_len.setdefault(length, msgs)
        if len(by_len) >= n:
            break
    if len(by_len) < 2:
        raise RuntimeError(
            f"could not find 2+ paddable prompt lengths "
            f"(got {sorted(by_len)}); generate_batch cannot show leftover "
            f"left-pad without mixed lengths"
        )
    chosen = [by_len[L] for L in sorted(by_len)[:n]]
    lens = [int(model.encode_messages(m)["input_ids"].shape[1]) for m in chosen]
    print(f"prompt token lengths: {lens}")
    return chosen


def _build_player_prompts(session: Any, model: Any, rt: dict) -> list[list[dict]]:
    settings = _plant(session, GOLD_AHEAD, rt)
    del settings
    frame = session.current_frame_path()
    base = rt["DEFAULT_PLAYER_QUESTION"]
    sys_p = session.PLAYER_SYSTEM_PROMPT

    def make(k: int) -> list[dict]:
        return rt["_build_game_messages"](
            sys_p, frame, "", base + " again" * k, notepad=None,
        )

    return _pick_length_grid(model, make, _PLAYER_N, rt)


def _build_analyst_prompts(session: Any, model: Any, rt: dict) -> list[list[dict]]:
    settings = _plant(session, GOLD_AHEAD, rt)
    frame = session.current_frame_path()
    planted = "The gold is straight ahead.\n[FORWARD]"
    q = rt["DEFAULT_ANALYST_QUESTION"]

    def make(k: int) -> list[dict]:
        return rt["build_scene_analyst_messages"](
            rt["DEFAULT_PLAYER_QUESTION"],
            planted + " again" * k,
            "FORWARD",
            frame,
            json.dumps(settings),
            "",
            q,
            system_prompt=session.ANALYST_SYSTEM_PROMPT,
            player_notepad=None,
        )

    return _pick_length_grid(model, make, _ANALYST_N, rt)


def _time_serial(
    model: Any,
    prompts: list[list[dict]],
    *,
    max_new_tokens: int,
    stop_regex: str | None,
) -> tuple[float, list[str]]:
    t0 = time.perf_counter()
    replies = [
        model.generate(
            p, max_new_tokens=max_new_tokens, stop_regex=stop_regex,
        )
        for p in prompts
    ]
    return time.perf_counter() - t0, replies


def _time_batch(
    model: Any,
    prompts: list[list[dict]],
    *,
    max_new_tokens: int,
    stop_regex: str | None,
) -> tuple[float, list[str], list[str]]:
    batch = [{"messages": p} for p in prompts]
    t0 = time.perf_counter()
    replies, lines = _generate_batch_logged(
        model, batch,
        max_new_tokens=max_new_tokens, stop_regex=stop_regex,
    )
    return time.perf_counter() - t0, replies, lines


def _print_table(rows: list[dict[str, Any]]) -> None:
    keys = ["wave", "mode", "path", "n", "wall_s", "s/row", "vs_control"]
    widths = {k: max(len(k), *(len(str(r.get(k, ""))) for r in rows))
              for k in keys}
    print()
    print("  ".join(k.ljust(widths[k]) for k in keys))
    print("  ".join("-" * widths[k] for k in keys))
    for r in rows:
        print("  ".join(str(r.get(k, "")).ljust(widths[k]) for k in keys))


def run_speed(rt: dict, replies_dir: Path, model: Any) -> int:
    from agent.model import get_model

    model = model or get_model()
    session = _make_session(rt, "infer_situations_speed", model)
    original = _force_greedy(model)
    failures: list[str] = []
    table: list[dict[str, Any]] = []
    try:
        print()
        print("######## SPEED: leftover-pad (control) vs prefix-KV (bc) ########")
        print(
            "control = mixed-length generate_batch, leftover left-pad (C) "
            "-- the old parallel datagen path.  "
            "bc = the same batch with resident prefix-KV (B+C).  "
            "serial = N solos, diagnostic only, not the old system."
        )
        print("Timed waves are greedy so control vs bc is the same work.")
        stop_player = rt["game_io"].PLAYER_STOP_PATTERN

        print()
        print(f"building {_PLAYER_N} player prompts (distinct token lengths)...")
        player_prompts = _build_player_prompts(session, model, rt)
        print(f"building {_ANALYST_N} analyst prompts (distinct token lengths)...")
        analyst_prompts = _build_analyst_prompts(session, model, rt)

        waves = [
            ("player", player_prompts, _PLAYER_NEW, stop_player),
            ("analyst", analyst_prompts, _ANALYST_NEW, None),
        ]
        for wave, prompts, n_new, stop in waves:
            print()
            print(f"--- {wave} wave n={len(prompts)} max_new_tokens={n_new} ---")
            _set_infer_mode("control")
            _clear_prefix(model)
            print(f"{wave} serial (untimed warmup generate on row 0)...")
            model.generate(
                prompts[0], max_new_tokens=n_new, stop_regex=stop,
            )
            print(f"{wave} serial timed...")
            serial_s, serial_replies = _time_serial(
                model, prompts, max_new_tokens=n_new, stop_regex=stop,
            )
            for i, text in enumerate(serial_replies):
                _write_reply(replies_dir, f"speed_{wave}_serial_{i}", text)
            print(f"{wave} serial: {serial_s:.2f}s wall  "
                  f"{serial_s / len(prompts):.2f}s/row  (diagnostic)")

            timed: dict[str, tuple[float, str]] = {}
            for mode in ("control", "bc"):
                _set_infer_mode(mode)
                _clear_prefix(model)
                print(f"{wave} {mode} warmup batch (untimed; fills prefix on bc)...")
                _generate_batch_logged(
                    model, [{"messages": p} for p in prompts],
                    max_new_tokens=n_new, stop_regex=stop,
                )
                print(f"{wave} {mode} timed batch...")
                wall, replies, lines = _time_batch(
                    model, prompts, max_new_tokens=n_new, stop_regex=stop,
                )
                tag = _path_tag(lines)
                for i, text in enumerate(replies):
                    _write_reply(replies_dir, f"speed_{wave}_{mode}_{i}", text)
                print(f"{wave} {mode} log: " + " | ".join(lines[-6:]))
                timed[mode] = (wall, tag)
                if mode == "control" and "padded" not in tag:
                    failures.append(
                        f"{wave} control did not take leftover left-pad "
                        f"(path={tag!r}). Mixed lengths should have "
                        f"logged 'padded batch'."
                    )
                if mode == "bc" and "prefix-kv" not in tag and "padded" not in tag:
                    failures.append(
                        f"{wave} bc took neither prefix-kv nor leftover "
                        f"pad (path={tag!r}) — silent cohort fallback."
                    )

            control_s = timed["control"][0]

            def _vs(wall: float) -> str:
                return f"{(control_s / wall):.2f}x" if wall > 0 else "?"

            table.append({
                "wave": wave, "mode": "serial", "path": "solo-generate",
                "n": len(prompts), "wall_s": round(serial_s, 2),
                "s/row": round(serial_s / len(prompts), 2),
                "vs_control": _vs(serial_s),
            })
            for mode in ("control", "bc"):
                wall, tag = timed[mode]
                table.append({
                    "wave": wave, "mode": mode, "path": tag,
                    "n": len(prompts), "wall_s": round(wall, 2),
                    "s/row": round(wall / len(prompts), 2),
                    "vs_control": _vs(wall),
                })

        _print_table(table)
        print()
        print(
            "Read vs_control: 1.00x is leftover-pad (old parallel path). "
            "bc above 1.00x is B on the same mixed-length batch. "
            "serial below 1.00x only confirms the batch beat N solos."
        )
    finally:
        model._sampling_kwargs = original
        session.close()

    if failures:
        print()
        print("SPEED FAIL (batch path did not engage):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print()
    print("SPEED PASS: leftover-pad and/or prefix-KV engaged on the timed batches.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_cli(argv)
    _apply_cli_env(args)
    os.chdir(REPO_ROOT)
    run_dir, replies_dir, tee = _open_report()
    rc = 0
    try:
        _announce(run_dir)
        print(
            "Budget: check ~3-8 min (6 live generates at your measured "
            "~3.5 s player / ~14 s analyst). speed ~4-12 min (6 serial "
            "player + 3 serial analyst + 2 batched modes, greedy, "
            "capped tokens). all ≈ 10-25 min after the 12B load. "
            "Hard cap: under 1 h."
        )
        print()
        rt = _import_runtime()
        if args.checkpoint:
            rt["set_default_checkpoint"](args.checkpoint)
        model = rt["get_model"]()
        if args.what in ("check", "all"):
            rc |= run_check(rt, replies_dir, model)
        if args.what in ("speed", "all"):
            rc |= run_speed(rt, replies_dir, model)
        print()
        _announce(run_dir)
        return rc
    finally:
        sys.stdout = tee._real
        tee.close()
        tee._real.write(f"\nReport closed: {run_dir / 'report.txt'}\n")
        tee._real.flush()


if __name__ == "__main__":
    raise SystemExit(main())
