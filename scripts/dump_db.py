#!/usr/bin/env python3
"""Dump the current Neo4j memory graph (all nodes + relationships) to a
``.dump`` JSON file, over the live bolt connection.

Same functionality as the "Dump DB status" button in ``notebooks/play.ipynb``:
a *logical* snapshot for offline inspection -- distinct from the native binary
``neo4j-admin`` dump that ``scripts/neo4j_db.sh save`` produces. The DB stays
up; this is safe to run mid-session.

Usage:
    python scripts/dump_db.py                    # -> <repo>/<timestamp>.dump
    python scripts/dump_db.py my_snapshot.dump   # -> any path you like
    python scripts/dump_db.py --include-embeddings out.dump
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

# Load the repo's .env explicitly so the script works from any cwd.
load_dotenv(REPO_ROOT / ".env")

from agent import memory as mem  # noqa: E402  (needs sys.path + .env first)


async def main(path: Path, include_embeddings: bool) -> None:
    client = await mem.connect()
    try:
        info = await mem.dump_database_to_file(
            client, path, include_embeddings=include_embeddings
        )
    finally:
        await client.close()
    print(f"DB dumped -> {info['path']}")
    print(f"  nodes:         {info['nodes']}")
    print(f"  relationships: {info['relationships']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dump the memory graph to a .dump JSON file (live bolt read)."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="output file (default: <timestamp>.dump in the repo root)",
    )
    parser.add_argument(
        "--include-embeddings",
        action="store_true",
        help="keep embedding vectors (huge; dropped by default)",
    )
    args = parser.parse_args()

    if args.path is not None:
        out = Path(args.path)
    else:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out = REPO_ROOT / f"{stamp}.dump"

    asyncio.run(main(out, args.include_embeddings))
