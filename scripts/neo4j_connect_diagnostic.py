#!/usr/bin/env python3
"""neo4j_connect_diagnostic.py -- confirm the game_saddle NAMS stack can talk
to the local Neo4j before opening the notebooks.

Two probes:

  1. bolt pre-check   -- direct neo4j-driver connectivity (bypasses NAMS): a
                         plain ``RETURN 1``. If this fails, Neo4j isn't up or
                         the credentials are wrong -- run
                         ``bash scripts/vast_neo4j_launch.sh`` first.
  2. NAMS round-trip  -- build + connect the agent's ``MemoryClient`` via
                         ``agent.memory.connect`` and issue one
                         ``get_context`` call. Exercises the exact path the
                         notebook / runner use.

Run from the repo root so the ``agent`` package and ``.env`` are found:
    python scripts/neo4j_connect_diagnostic.py
"""
from __future__ import annotations

import asyncio
import datetime
import sys
import time

from agent.config import CONFIG
from agent import memory as mem


def log(msg: str) -> None:
    print(f"[diag {datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def bolt_precheck() -> bool:
    log("PROBE 1: bolt pre-check (direct neo4j driver, bypasses NAMS)...")
    t0 = time.time()
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        CONFIG.neo4j_uri, auth=(CONFIG.neo4j_username, CONFIG.neo4j_password)
    )
    try:
        with driver.session(database=CONFIG.neo4j_database) as s:
            rec = s.run("RETURN 1 AS x").single()
            log(f"PROBE 1 ok in {time.time() - t0:.1f}s: RETURN 1 -> {rec['x']}")
        return True
    except Exception as exc:
        log(f"PROBE 1 FAILED in {time.time() - t0:.1f}s: {type(exc).__name__}: {exc}")
        log("  -> Neo4j is not reachable. Run: bash scripts/vast_neo4j_launch.sh")
        return False
    finally:
        driver.close()


def nams_roundtrip() -> bool:
    log("PROBE 2: NAMS MemoryClient connect + get_context...")
    t0 = time.time()

    async def _go() -> None:
        client = await mem.connect()
        try:
            ctx = await client.get_context(query="ping", session_id="diagnostic")
            preview = str(ctx)
            if len(preview) > 200:
                preview = preview[:200] + "..."
            log(f"PROBE 2 ok in {time.time() - t0:.1f}s: get_context -> {preview!r}")
        finally:
            await client.close()

    try:
        asyncio.run(_go())
        return True
    except Exception as exc:
        log(f"PROBE 2 FAILED in {time.time() - t0:.1f}s: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    log(
        f"game_saddle NAMS connect diagnostic: uri={CONFIG.neo4j_uri!r} "
        f"user={CONFIG.neo4j_username!r} db={CONFIG.neo4j_database!r}"
    )
    ok1 = bolt_precheck()
    ok2 = nams_roundtrip() if ok1 else False
    log("done." if (ok1 and ok2) else "done (with failures).")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
