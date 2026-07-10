"""Per-run logging for game_saddle: every LLM ``generate`` call and every DB
retrieval, plus an on-demand DB dump, written into one timestamped run
directory under ``logs/``.

Design (mirrors the sister ``generative_agents`` prompt-log style the author
likes):

  * Two concerns, two log pairs. Each is written as machine-readable JSONL
    **and** a human-readable ``.txt`` sibling (banner-delimited, one block per
    record, in the same order):
      - ``llm_calls.{jsonl,txt}``    -- every ``model.generate`` (input + output)
      - ``db_retrieval.{jsonl,txt}`` -- every memory retrieval (function, args,
                                        result)
  * A run directory ``logs/<label>_<YYYY-MM-DD_HH-MM-SS>/`` holds both pairs and
    any ``.dump`` files produced during the run.
  * **On by default.** Entry points (``InteractiveSession``, the ``runner`` CLI)
    create and activate a :class:`RunLogger`. The low-level call sites
    (``model.generate`` and the ``memory`` retrievals) log only if a logger is
    *active*, so merely importing the library writes nothing.
  * Logging must never break a run: any I/O/serialization failure degrades to a
    one-time console warning and execution proceeds.
"""

from __future__ import annotations

import datetime
import json
import re
import threading
from pathlib import Path
from typing import Any, Optional

_BANNER = "=" * 78
_RULE_WIDTH = 78


def _rule(label: str) -> str:
    head = f"--- {label} "
    return head + "-" * max(0, _RULE_WIDTH - len(head))


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip()).strip("-._")
    return slug.lower() or "run"


def _now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


class RunLogger:
    """Owns one run directory and appends to its LLM + DB logs."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.llm_jsonl = self.run_dir / "llm_calls.jsonl"
        self.llm_txt = self.run_dir / "llm_calls.txt"
        self.db_jsonl = self.run_dir / "db_retrieval.jsonl"
        self.db_txt = self.run_dir / "db_retrieval.txt"
        self._lock = threading.Lock()
        self._llm_seq = 0
        self._db_seq = 0
        self._warned = False

    # ------------------------------------------------------------------ helpers
    def _write(self, jsonl_path: Path, txt_path: Path, record: dict, text: str) -> None:
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
            with self._lock:
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                with open(txt_path, "a", encoding="utf-8") as f:
                    f.write(text)
        except Exception as exc:  # pragma: no cover - logging must not break runs
            if not self._warned:
                self._warned = True
                print(f"[run_logging] disabled after write failure "
                      f"({type(exc).__name__}): {exc}")

    # ------------------------------------------------------------------- LLM log
    def log_llm_call(
        self,
        *,
        model: str,
        kind: str,
        request: Any,
        params: Optional[dict] = None,
        response: Any = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            self._llm_seq += 1
            seq = self._llm_seq
        record = {
            "seq": seq,
            "ts": _now_iso(),
            "model": model,
            "kind": kind,
            "request": request,
            "params": params,
            "response": response,
        }
        if error is not None:
            record["error"] = error
        self._write(self.llm_jsonl, self.llm_txt, record, _format_llm_text(record))

    # -------------------------------------------------------------------- DB log
    def log_db_retrieval(
        self,
        *,
        function: str,
        arguments: Optional[dict] = None,
        result: Any = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            self._db_seq += 1
            seq = self._db_seq
        record = {
            "seq": seq,
            "ts": _now_iso(),
            "function": function,
            "arguments": arguments or {},
            "result": result,
        }
        if error is not None:
            record["error"] = error
        self._write(self.db_jsonl, self.db_txt, record, _format_db_text(record))

    # ------------------------------------------------------------------ dumps
    def dump_path(self, name: Optional[str] = None) -> Path:
        stamp = datetime.datetime.now().strftime("%H-%M-%S")
        base = _slugify(name) if name else "db_snapshot"
        return self.run_dir / f"{base}_{stamp}.dump"


def _format_llm_text(record: dict) -> str:
    lines = [
        _BANNER,
        f"llm call #{record['seq']}  {record['ts']}  {record['model']}  "
        f"[{record['kind']}]",
    ]
    if record.get("params") is not None:
        lines.append("params: " + json.dumps(record["params"], ensure_ascii=False, default=str))
    lines.append("")

    request = record.get("request")
    messages = request.get("messages") if isinstance(request, dict) else None
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                lines += [_rule("message"), str(msg)]
                continue
            role = str(msg.get("role", "?"))
            content = msg.get("content", "")
            lines.append(_rule(role))
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        lines.append(f"[image] {part.get('url', '')}")
                    elif isinstance(part, dict) and part.get("type") == "text":
                        lines.append(str(part.get("text", "")))
                    else:
                        lines.append(str(part))
            else:
                lines.append(str(content))
    elif request is not None:
        lines += [_rule("request"), json.dumps(request, ensure_ascii=False, indent=2, default=str)]

    if isinstance(request, dict) and request.get("rendered_prompt"):
        lines += [_rule("rendered prompt (exact model input)"), str(request["rendered_prompt"])]

    response = record.get("response")
    if isinstance(response, dict):
        lines += [_rule("response"), str(response.get("raw", ""))]
    elif response is not None:
        lines += [_rule("response"), str(response)]

    if record.get("error") is not None:
        lines += [_rule("ERROR"), str(record["error"])]

    lines.append("")
    return "\n".join(lines) + "\n"


def _format_db_text(record: dict) -> str:
    lines = [
        _BANNER,
        f"retrieval #{record['seq']}  {record['ts']}  {record['function']}",
        "args: " + json.dumps(record.get("arguments", {}), ensure_ascii=False, default=str),
        "",
    ]
    if record.get("error") is not None:
        lines += [_rule("ERROR"), str(record["error"])]
    else:
        lines += [_rule("result"), str(record.get("result", ""))]
    lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------- active logger

_active: Optional[RunLogger] = None
_active_lock = threading.Lock()


def get_active_logger() -> Optional[RunLogger]:
    return _active


def set_active_logger(logger: Optional[RunLogger]) -> None:
    global _active
    with _active_lock:
        _active = logger


def new_run_logger(
    label: Optional[str] = None, base_dir: str | Path = "logs", activate: bool = True
) -> RunLogger:
    """Create a fresh timestamped run directory + logger. Activates it (so the
    low-level call sites start logging to it) unless ``activate=False``."""
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    slug = _slugify(label) if label else "run"
    logger = RunLogger(Path(base_dir) / f"{slug}_{stamp}")
    if activate:
        set_active_logger(logger)
    return logger


# ------------------------------------------------------- module-level passthrough

def log_llm_call(**kwargs: Any) -> None:
    logger = get_active_logger()
    if logger is not None:
        logger.log_llm_call(**kwargs)


def log_db_retrieval(**kwargs: Any) -> None:
    logger = get_active_logger()
    if logger is not None:
        logger.log_db_retrieval(**kwargs)
