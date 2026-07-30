"""Cross-thread generation batching for parallel datagen.

Why: batch-1 decode of a 12B model is memory-bandwidth-bound -- every token
step re-reads all the weights, so N sequential streams cost N times as much,
while ONE batched call serves N streams for barely more than one. Running
N game sessions in parallel therefore only pays if their generations are
merged into batched ``VLModel.generate_batch`` calls; N threads politely
taking turns on ``generate`` would buy almost nothing.

The pieces:

  * :class:`GenerationDispatcher` -- owns the (process-singleton) model.
    Worker threads submit requests and block; a scheduler thread groups
    COMPATIBLE requests into one batched call. Compatibility = identical
    stop signature and image count. Stop knobs are batch-wide in
    ``generate_batch`` (a per-row mix would let an analyst quoting
    "[FORWARD]" be truncated by a player row's stop string). Prompt length
    is deliberately NOT part of the signature: ``generate_batch`` itself
    decides how a group decodes -- mixed lengths go through the VERIFIED
    LEFT-PAD path (the KNOWN TRANSFORMERS BUG WORKAROUND for
    transformers#47651, see agent/model.py's banner), with exact-length
    cohorts only as its fallback.

    SCHEDULING = PHASE LOCKING, not a fixed gather window. Each datagen
    round alternates a player generation and an analyst generation, and
    those two have DIFFERENT signatures, so they can never share a batch.
    With a naive short window, two sessions drift into a stable ANTI-PHASE
    (A's player call always concurrent with B's analyst call): every batch
    has size 1 and the speedup collapses -- measured x1.18 at --parallel 2
    (t9, 2026-07-30) when it should approach x2. There is no re-alignment
    force in a fixed window, so this dispatcher creates one:

      1. NEVER serve a lone request while other LIVE workers are still
         computing between generations -- their next request arrives in
         seconds, a generation costs tens of seconds. (Liveness comes from
         :meth:`worker_started`/:meth:`worker_finished`, called by the
         datagen workers around their game loop; with no registered
         workers every request is served immediately.)
      2. Once EVERY live worker has a request pending, no bigger batch can
         form by waiting. If they are all compatible, serve them as one
         full batch. Otherwise serve the SMALLEST signature group (oldest
         first on ties) and hold the larger groups: the workers just
         released will loop around and JOIN the held group, locking all
         sessions into the same phase (players batch with players,
         analysts with analysts) after at most one solo generation of
         re-sync cost.
      3. ``hold_max_s`` bounds every wait (a worker wedged in NAMS I/O
         must not starve its peers): the oldest request's group is served
         unconditionally past the deadline.

  * :class:`BatchingProxy` -- duck-types ``VLModel`` for a session: same
    ``generate(...)`` signature, everything else forwarded to the real
    model. Installed as ``session.model``, so the session code does not
    know it is sharing a GPU.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class _Request:
    __slots__ = ("messages", "max_new_tokens", "stop_strings", "stop_regex",
                 "n_images", "t_submit", "done", "result", "error")

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
        self.t_submit = time.monotonic()
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
    """Gather concurrent generate requests into batched model calls,
    phase-locking the workers (see module docstring)."""

    def __init__(self, model: Any, max_batch: int = 3,
                 hold_max_s: float = 120.0):
        self.model = model
        self.max_batch = max_batch
        self.hold_max_s = hold_max_s
        self._pending: list[_Request] = []  #: arrival order
        self._n_live = 0
        self._cond = threading.Condition()
        self._closed = False
        self._thread = threading.Thread(
            target=self._loop, name="gen-dispatcher", daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------- workers
    def worker_started(self) -> None:
        """A worker entered its generate loop; the scheduler may hold other
        requests waiting for this worker's next submission."""
        with self._cond:
            self._n_live += 1
            self._cond.notify_all()

    def worker_finished(self) -> None:
        """The worker will submit no more requests; stop waiting for it."""
        with self._cond:
            self._n_live -= 1
            self._cond.notify_all()

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
            self._cond.notify_all()
        req.done.wait()
        if req.error is not None:
            raise req.error
        assert req.result is not None
        return req.result

    # ---------------------------------------------------------- dispatcher
    def _pick_group_locked(self) -> tuple[list[_Request], str]:
        """The scheduling policy (module docstring). Returns the group to
        serve now plus the reason tag, or ``([], "")`` to keep waiting.
        Caller holds ``self._cond``."""
        by_sig: dict[tuple, list[_Request]] = {}
        for r in self._pending:  # arrival order -> group[0] is its oldest
            by_sig.setdefault(r.signature(), []).append(r)
        groups = list(by_sig.values())

        # No registered workers => no one to phase-lock with: serve
        # immediately (cap 1 makes any group "full").
        cap = min(self.max_batch, self._n_live) if self._n_live > 0 else 1
        best = max(groups, key=len)
        if len(best) >= cap:
            return best[: self.max_batch], "full-batch"

        oldest = self._pending[0]
        if time.monotonic() - oldest.t_submit >= self.hold_max_s:
            return (by_sig[oldest.signature()][: self.max_batch],
                    "hold-timeout")
        if self._closed:
            return by_sig[oldest.signature()][: self.max_batch], "drain"

        if len(self._pending) >= self._n_live:
            # Every live worker is blocked on us; waiting cannot grow any
            # group. Serve the smallest (oldest on ties) so the released
            # workers come back and join the larger held group -- this is
            # the re-alignment step that breaks the anti-phase attractor.
            small = min(groups, key=lambda g: (len(g), g[0].t_submit))
            return small[: self.max_batch], "phase-lock"
        return [], ""

    def _loop(self) -> None:
        while True:
            with self._cond:
                while True:
                    if self._closed and not self._pending:
                        return
                    if self._pending:
                        group, reason = self._pick_group_locked()
                        if group:
                            for r in group:
                                self._pending.remove(r)
                            left = len(self._pending)
                            n_live = self._n_live
                            break
                    # Timed wait: the hold-timeout deadline must fire even
                    # if no new request ever arrives to notify us.
                    self._cond.wait(timeout=0.05)
            logger.info(
                "dispatch: group of %d [%s] (%d request(s) held, %d live "
                "worker(s))", len(group), reason, left, n_live,
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
