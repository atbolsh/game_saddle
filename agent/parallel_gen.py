"""Cross-thread generation batching for parallel datagen.

Why: batch-1 decode of a 12B model is memory-bandwidth-bound -- every token
step re-reads all the weights, so N sequential streams cost N times as much,
while ONE batched call serves N streams for barely more than one. Running
N game sessions in parallel therefore only pays if their generations are
merged into batched ``VLModel.generate_batch`` calls; N threads politely
taking turns on ``generate`` would buy almost nothing.

The pieces:

  * :class:`GenerationDispatcher` -- owns the (process-singleton) model.
    Worker threads submit requests and block; the dispatcher waits a short
    window for peers to arrive, groups compatible requests, and runs one
    batched call. Compatibility = identical stop signature and image count.
    Stop knobs are batch-wide in ``generate_batch`` (a per-row mix would let
    an analyst quoting "[FORWARD]" be truncated by a player row's stop
    string). Prompt length is deliberately NOT part of the signature:
    ``generate_batch`` itself decides how a group decodes -- mixed lengths
    go through the VERIFIED LEFT-PAD path (the KNOWN TRANSFORMERS BUG
    WORKAROUND for transformers#47651, see agent/model.py's banner), with
    exact-token-length cohorts only as its fallback. Bucketing here by any
    length proxy would just fragment groups that the padded path can batch.
  * :class:`BatchingProxy` -- duck-types ``VLModel`` for a session: same
    ``generate(...)`` signature, everything else forwarded to the real
    model. Installed as ``session.model``, so the session code does not know
    it is sharing a GPU.

Worker threads that arrive alone are not penalized beyond the gather window
(default 50 ms, ~2 orders of magnitude below a generation's cost).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class _Request:
    __slots__ = ("messages", "max_new_tokens", "stop_strings", "stop_regex",
                 "n_images", "done", "result", "error")

    def __init__(self, messages: list[dict], max_new_tokens: int | None,
                 stop_strings: list[str] | None, stop_regex: str | None):
        self.messages = messages
        self.max_new_tokens = max_new_tokens
        self.stop_strings = stop_strings
        self.stop_regex = stop_regex
        self.n_images = sum(
            1
            for m in messages
            for part in (m.get("content") or [])
            if isinstance(part, dict) and part.get("type") == "image"
        )
        self.done = threading.Event()
        self.result: str | None = None
        self.error: BaseException | None = None

    def signature(self) -> tuple:
        """Requests may share a batch iff these match (see module docstring)."""
        return (
            tuple(self.stop_strings or ()),
            self.stop_regex or "",
            self.max_new_tokens,
            self.n_images,
        )


class GenerationDispatcher:
    """Gather concurrent generate requests into batched model calls."""

    def __init__(self, model: Any, max_batch: int = 3,
                 window_s: float = 0.05):
        self.model = model
        self.max_batch = max_batch
        self.window_s = window_s
        self._pending: list[_Request] = []
        self._cond = threading.Condition()
        self._closed = False
        self._thread = threading.Thread(
            target=self._loop, name="gen-dispatcher", daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------- workers
    def generate(self, messages: list[dict],
                 max_new_tokens: int | None = None,
                 stop_strings: list[str] | None = None,
                 stop_regex: str | None = None) -> str:
        """Blocking submit; same contract as ``VLModel.generate``."""
        req = _Request(messages, max_new_tokens, stop_strings, stop_regex)
        with self._cond:
            if self._closed:
                raise RuntimeError("GenerationDispatcher is closed")
            self._pending.append(req)
            self._cond.notify()
        req.done.wait()
        if req.error is not None:
            raise req.error
        assert req.result is not None
        return req.result

    # ---------------------------------------------------------- dispatcher
    def _loop(self) -> None:
        while True:
            with self._cond:
                while not self._pending and not self._closed:
                    self._cond.wait()
                if self._closed and not self._pending:
                    return
            # Let concurrent peers arrive before forming the batch.
            time.sleep(self.window_s)
            with self._cond:
                if not self._pending:
                    continue
                sig = self._pending[0].signature()
                group = [r for r in self._pending
                         if r.signature() == sig][: self.max_batch]
                for r in group:
                    self._pending.remove(r)
                left_behind = len(self._pending)
            logger.info(
                "dispatch: group of %d (%d incompatible request(s) left "
                "for the next window)", len(group), left_behind,
            )
            try:
                replies = self.model.generate_batch(
                    [{"messages": r.messages} for r in group],
                    max_new_tokens=group[0].max_new_tokens,
                    stop_strings=list(group[0].stop_strings or []) or None,
                    stop_regex=group[0].stop_regex or None,
                )
                for r, reply in zip(group, replies):
                    r.result = reply
            except BaseException as exc:  # propagate to EVERY waiter
                logger.exception("batched generate failed (batch of %d)",
                                 len(group))
                for r in group:
                    r.error = exc
            finally:
                for r in group:
                    r.done.set()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._thread.join(timeout=10)


class BatchingProxy:
    """Duck-types ``VLModel`` for a session: ``generate`` goes through the
    dispatcher, every other attribute (spec, processor, ...) is forwarded to
    the real model."""

    def __init__(self, dispatcher: GenerationDispatcher):
        # Avoid __setattr__/__getattr__ recursion: set via object.
        object.__setattr__(self, "_dispatcher", dispatcher)

    def generate(self, messages: list[dict],
                 max_new_tokens: int | None = None,
                 stop_strings: list[str] | None = None,
                 stop_regex: str | None = None) -> str:
        return self._dispatcher.generate(
            messages, max_new_tokens=max_new_tokens,
            stop_strings=stop_strings, stop_regex=stop_regex,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dispatcher.model, name)
