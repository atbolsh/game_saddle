"""CLI entry point for the agent.

Usage::

    python -m agent.runner seed
    python -m agent.runner link
    python -m agent.runner game     --session <id> --question "..." [--solve]
    python -m agent.runner discuss  --session <id> --text "..."
    python -m agent.runner eval     --session <id>

If ``--session`` is omitted a fresh UUID-based session id is generated and
printed. The NAMS ``MemoryClient`` is connected for the duration of the
command and closed on exit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

from .config import CONFIG
from . import memory as mem
from . import modes


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


async def _cmd_seed(args: argparse.Namespace) -> int:
    client = await mem.connect()
    try:
        counts = await mem.build_semantic_model(client)
    finally:
        await client.close()
    print(json.dumps({"seeded": counts}, indent=2))
    return 0


async def _cmd_link(args: argparse.Namespace) -> int:
    client = await mem.connect()
    try:
        counts = await mem.add_semantic_relationships(client)
    finally:
        await client.close()
    print(json.dumps({"linked": counts}, indent=2))
    return 0


async def _cmd_game(args: argparse.Namespace) -> int:
    session_id = args.session or mem.new_session_id()
    client = await mem.connect()
    try:
        result = await modes.mode_game(
            client, session_id, args.question, solve=args.solve,
            max_steps=args.max_steps,
        )
    finally:
        await client.close()
    print(json.dumps({"session_id": session_id, **result}, default=str, indent=2))
    return 0


async def _cmd_discuss(args: argparse.Namespace) -> int:
    session_id = args.session or mem.new_session_id()
    client = await mem.connect()
    try:
        result = await modes.mode_discuss(client, session_id, args.text)
    finally:
        await client.close()
    print(json.dumps(result, default=str, indent=2))
    return 0


async def _cmd_eval(args: argparse.Namespace) -> int:
    if not args.session:
        print("error: --session is required for eval", file=sys.stderr)
        return 2
    client = await mem.connect()
    try:
        result = await modes.mode_self_eval(client, args.session)
    finally:
        await client.close()
    print(json.dumps(result, default=str, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agent.runner", description="Gemma 4 E4B game agent (NAMS-backed)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_seed = sub.add_parser("seed", help="Seed the NAMS long-term semantic model (run once).")
    p_seed.set_defaults(func=_cmd_seed)

    p_link = sub.add_parser(
        "link",
        help="Add the hardwired entity relationships to an already-seeded graph "
             "(no wipe; does not duplicate entities/preferences).",
    )
    p_link.set_defaults(func=_cmd_link)

    p_game = sub.add_parser("game", help="Mode 1: play / answer a question about a game screen.")
    p_game.add_argument("--session", default=None)
    p_game.add_argument("--question", required=True,
                        help='e.g. "make the best move", "is the gold to your left or right?", "solve the game"')
    p_game.add_argument("--solve", action="store_true",
                        help="Loop moves until the gold is eaten (ignored if the question is not a move request).")
    p_game.add_argument("--max-steps", type=int, default=CONFIG.max_solve_steps)
    p_game.set_defaults(func=_cmd_game)

    p_disc = sub.add_parser("discuss", help="Mode 2: open-ended discussion with full memory access.")
    p_disc.add_argument("--session", default=None)
    p_disc.add_argument("--text", required=True)
    p_disc.set_defaults(func=_cmd_discuss)

    p_eval = sub.add_parser("eval", help="Mode 3: self-evaluate a recorded session.")
    p_eval.add_argument("--session", required=True)
    p_eval.set_defaults(func=_cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return asyncio.run(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
