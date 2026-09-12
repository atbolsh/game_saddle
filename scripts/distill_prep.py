#!/usr/bin/env python3
"""Corpus-specific preprocessor for one context-distillation run.

Reads ``data_game/<label>/traces.jsonl`` and, if present,
``analyst_traces.jsonl``. Writes ``traces_distill.jsonl`` and
``analyst_traces_distill.jsonl`` into the SAME directory (relative image
URLs keep resolving). Per record:

  1. Replace ``messages[0]`` (system) with this checkout's composed
     prompt -- play dump for player traces, scene-analyst dump for
     analyst traces. Byte-identical to what NAMS seeds here.
  2. Inside any user text part starting ``Memory context:``, drop bullet
     lines that quote ``[core_player_*]`` / ``[core_analyst_*]`` (stale
     prompt text NAMS retrieval quoted). Leave ``tip_*`` / ``goal`` /
     entity lines.
  3. Leave ``target_text`` and ``meta`` untouched.

This script is the deletable, corpus-shaped half. The reusable trainer is
``training/context_distill.py``.

    python scripts/distill_prep.py data_game/<label>

Prints records written and tip-lines stripped. Does not claim token
counts from character math.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Category tag as NAMS retrieval quotes it: [core_player_030_scene_scope]
_CORE_CAT_RE = re.compile(r"\[core_(?:player|analyst)_[A-Za-z0-9_]+\]")
_BULLET_RE = re.compile(r"^\s*[-*]\s+")


def scrub_memory_context(text: str) -> tuple[str, int]:
    """Drop core_* tip bullets (and their indented continuations).

    Returns ``(rewritten_text, n_lines_stripped)``. Non-Memory-context
    strings are returned unchanged with count 0.
    """
    if not text.startswith("Memory context:"):
        return text, 0
    out: list[str] = []
    stripped = 0
    skipping = False
    for line in text.splitlines():
        if _CORE_CAT_RE.search(line):
            stripped += 1
            skipping = True
            continue
        if skipping:
            if _BULLET_RE.match(line):
                skipping = False
            elif line.strip() and not line[0].isspace():
                skipping = False
            elif not line.strip():
                skipping = False
                out.append(line)
                continue
            else:
                stripped += 1
                continue
        out.append(line)
    return "\n".join(out), stripped


def _set_system_text(message: dict[str, Any], text: str) -> None:
    content = message.get("content")
    if isinstance(content, str):
        message["content"] = text
        return
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = text
                return
        content.insert(0, {"type": "text", "text": text})
        return
    message["content"] = [{"type": "text", "text": text}]


def _scrub_messages(messages: list[dict[str, Any]]) -> int:
    """Scrub Memory-context parts in place. Returns tip-lines stripped."""
    n = 0
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            new, k = scrub_memory_context(content)
            if k:
                msg["content"] = new
                n += k
            continue
        if not isinstance(content, list):
            continue
        keep: list[Any] = []
        for part in content:
            if not (isinstance(part, dict) and part.get("type") == "text"):
                keep.append(part)
                continue
            text = part.get("text") or ""
            new, k = scrub_memory_context(text)
            n += k
            if k and new.strip() in ("", "Memory context:"):
                continue
            part = dict(part)
            part["text"] = new
            keep.append(part)
        msg["content"] = keep
    return n


def rewrite_record(obj: dict[str, Any], system_text: str) -> int:
    """Rewrite one trace record in place. Returns tip-lines stripped."""
    messages = obj["messages"]
    if not messages:
        raise ValueError("trace record has empty messages")
    _set_system_text(messages[0], system_text)
    return _scrub_messages(messages)


def _resolve_dir(raw: str) -> Path:
    path = Path(raw)
    if path.is_file() and path.name in (
        "traces.jsonl", "analyst_traces.jsonl",
        "traces_distill.jsonl", "analyst_traces_distill.jsonl",
    ):
        return path.parent
    if path.is_dir():
        return path
    raise SystemExit(f"expected data_game/<label> or a traces jsonl, got {raw}")


def _rewrite_file(src: Path, dest: Path, system_text: str) -> tuple[int, int]:
    n_records = 0
    n_stripped = 0
    with src.open(encoding="utf-8") as fin, dest.open(
        "w", encoding="utf-8",
    ) as fout:
        for lineno, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                n_stripped += rewrite_record(obj, system_text)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise SystemExit(
                    f"{src}:{lineno}: {type(exc).__name__}: {exc}"
                ) from exc
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_records += 1
    print(f"wrote {dest}")
    print(f"  records written: {n_records}")
    print(f"  core_* tip-lines stripped: {n_stripped}")
    return n_records, n_stripped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "corpus",
        help="data_game/<label> directory (or a traces jsonl inside it)",
    )
    args = p.parse_args(argv)

    from agent.modes import (
        CORE_ANALYST_TIPS,
        CORE_PLAYER_TIPS,
        ROLE_SCENE_ANALYST,
        ROLE_SCENE_PLAY,
        SCENE_ANALYST_EXCLUDE,
        format_core_tips_dump,
    )

    player_system = format_core_tips_dump(
        ROLE_SCENE_PLAY, dict(CORE_PLAYER_TIPS), set(),
    )
    analyst_system = format_core_tips_dump(
        ROLE_SCENE_ANALYST, dict(CORE_ANALYST_TIPS), SCENE_ANALYST_EXCLUDE,
    )
    trace_dir = _resolve_dir(args.corpus)
    src = trace_dir / "traces.jsonl"
    if not src.is_file():
        raise SystemExit(f"no traces.jsonl in {trace_dir}")
    _rewrite_file(src, trace_dir / "traces_distill.jsonl", player_system)

    analyst_src = trace_dir / "analyst_traces.jsonl"
    if analyst_src.is_file():
        _rewrite_file(
            analyst_src,
            trace_dir / "analyst_traces_distill.jsonl",
            analyst_system,
        )
    else:
        print(f"no {analyst_src} -- skipped analyst rewrite")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
