"""Materialize the external replay datasets described in training/datasets.json.

Called by scripts/setup_env.sh after the environment is up; safe to re-run
any time (an existing meta.json / probe file is skipped unless --force).
Needs network + the HF token from .env; run it on the remote box.

Usage:
    python -m training.download_external            # everything enabled
    python -m training.download_external --only gsm8k --only navigation
    python -m training.download_external --force --mode stream

Download mode: from data_external/settings.json (written by setup_env.sh
based on free disk -- "full" keeps whole datasets in the HF cache, "stream"
touches only the consumed shards); --mode overrides.

Failures are per-dataset and loud: the script continues to the next entry,
prints every failure at the end, and exits nonzero if there was any.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.external_data import (
    DATA_DIR,
    download_mode,
    load_manifest,
    materialize,
    materialize_probe,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", action="append", default=None, metavar="NAME",
                        help="materialize only these manifest entries (repeatable)")
    parser.add_argument("--force", action="store_true",
                        help="re-materialize even if meta.json exists")
    parser.add_argument("--mode", choices=("full", "stream"), default=None,
                        help="override data_external/settings.json download_mode")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    log = logging.getLogger("download_external")

    mode = args.mode or download_mode()
    entries = load_manifest()
    known = {e.name for e in entries}
    if args.only:
        unknown = set(args.only) - known
        if unknown:
            parser.error(f"unknown dataset name(s): {sorted(unknown)}; "
                         f"manifest has: {sorted(known)}")
        entries = [e for e in entries if e.name in args.only]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log.info("data root: %s   download mode: %s", DATA_DIR, mode)

    results: list[dict] = []
    failures: list[tuple[str, str]] = []
    for entry in entries:
        if not entry.enabled:
            if args.only and entry.name in args.only:
                # explicitly requested a disabled entry -- that is an error,
                # not a silent skip
                failures.append((entry.name, "entry is disabled in the manifest"))
            else:
                log.info("%s: disabled, skipping", entry.name)
            continue
        try:
            meta = materialize(entry, mode, force=args.force)
            probe = materialize_probe(entry, mode, force=args.force)
            results.append({"entry": entry, "meta": meta, "probe": probe})
        except Exception as exc:
            log.error("%s: FAILED: %s: %s", entry.name, type(exc).__name__, exc)
            failures.append((entry.name, f"{type(exc).__name__}: {exc}"))

    if results:
        print()
        print(f"{'dataset':<18} {'examples':>9} {'images':>7} {'size':>9}  probe")
        print("-" * 60)
        total_bytes = 0
        for r in results:
            meta = r["meta"]
            total_bytes += int(meta.get("bytes", 0))
            probe = r["probe"]
            probe_str = ("-" if probe is None
                         else "cached" if probe.get("skipped")
                         else f"{probe.get('items', '?')} items")
            skipped = " (cached)" if meta.get("skipped") else ""
            print(f"{meta['name']:<18} {meta['examples']:>9} "
                  f"{meta.get('images', 0):>7} "
                  f"{meta.get('bytes', 0) / 1e6:>7.1f}MB  {probe_str}{skipped}")
        print("-" * 60)
        print(f"{'total':<18} {'':>9} {'':>7} {total_bytes / 1e9:>8.2f}GB")

    if failures:
        print("\nFAILED datasets:", file=sys.stderr)
        for name, msg in failures:
            print(f"  {name}: {msg}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
