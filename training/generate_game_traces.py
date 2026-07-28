"""Headless self-eval datagen: play games, grade every move, write traces.

The default data generator of the self-training loop (TRAINING_GAME_TRACES.md).
Drives the EXACT machinery of the interactive self-eval notebook --
:class:`agent.self_eval_session.InteractiveSelfEvalSession` with its default
player and analyst questions, same prompts, same memory screening -- with no
human in the loop:

    per round:  ask_player(DEFAULT_PLAYER_QUESTION)
                -> ask_analyst(DEFAULT_ANALYST_QUESTION)   (one exchange)
                -> end_round()                             (move propagates)

**A game is formally: the gold eaten, OR --max-moves player rounds**
(default 50 -- early wandering traces carry little signal per round),
whichever comes first. Records buffer per game and are written at
game close, each stamped with ``meta.game_won`` and ``meta.moves_from_end``
(0 = the round whose move ate the gold) -- the discounted win boost is
computed from these at TRAINING time by ``GameTraceSource``, not here; this
script stores raw annotations only (rating, verified WRONG spans, outcome),
so every reward ratio stays tunable without regenerating data.

One record per player generation, in ``data_game/<label>/traces.jsonl``:

  * ``messages``  -- the EXACT prompt of the accepted player generation
    (system prompt, screened NAMS context, search notes, frame), with the
    frame's image url rewritten to a stable copy under
    ``data_game/<label>/images/`` (byte-identical to what the player saw --
    the live ``memory_images/`` copy does not survive NAMS resets);
  * ``target_text`` -- the raw player reply (the ONLY trainable tokens);
  * ``meta``      -- rating, wrong spans, action, game/move indices, outcome.

Analyst text is stored to NAMS exactly as in the notebook and never enters
any record's ``messages``/``target_text`` -- it exists in training only as
annotation. As a TRIPWIRE against a screening regression, every analyst
analysis generated in the current session is checked against each record's
serialized player context; a hit aborts the run (a silently poisoned
dataset is the worst outcome).

Housekeeping per the committed plan: frames are noised at inference
(mild label-safe degradation, ``--no-noise`` to disable) and NAMS episodic
memory is reset to the seeded semantic model (tips are ``Preference`` nodes
and survive) every ``--reset-every`` games.

**Parallelism** (``--parallel``, default 3): N sessions play N games
concurrently, one worker thread each, sharing ONE model through
``agent.parallel_gen`` -- concurrent generations merge into batched decode
calls (batch-1 decode is bandwidth-bound, so a batch of 3 costs barely more
than a batch of 1). NAMS resets happen at BLOCK boundaries: games run in
sequential blocks of ``--reset-every``, all workers drain between blocks,
one session resets, all restart. Each concurrent session is invisible to
the others' players exactly as PAST sessions are (current-session
screening is per session; cross-session analyst memories are the intended
learning mechanism), so the leak tripwire stays session-scoped. At the end
of the run, summary plots land in ``logs/datagen_stats_<label>_<stamp>/``.

GPU + NAMS required; run on the remote box, typically once per training
iteration with the current checkpoint::

    python -m training.generate_game_traces --label iter1
    python -m training.generate_game_traces --label iter2 \
        --checkpoint iter1_step600 --games 60 --seed 11
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.image_noise import INFERENCE_STRENGTH, make_image_filter

logger = logging.getLogger("train.generate_game_traces")

#: Root of the generated game-trace tree (gitignored; a symlink onto
#: removable storage on the owner's local box).
DATA_GAME_DIR = Path(__file__).resolve().parent.parent / "data_game"

#: How much of each stored analyst analysis the leak tripwire matches on.
#: Long enough to never fire on a coincidental phrase, short enough to catch
#: a truncated leak.
_TRIPWIRE_PREFIX_LEN = 100


def _rewrite_image_urls(messages: list[dict], mapping: dict[str, str]) -> int:
    """Rewrite image part urls in place per ``mapping`` (absolute live path
    -> trace-relative path). Returns the number of rewrites; an image url
    with no mapping is a hard error -- every frame the player saw must have
    a stable copy."""
    n = 0
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                url = part["url"]
                if url not in mapping:
                    raise RuntimeError(
                        f"player context references image {url!r} with no "
                        "stable copy -- refusing to write a record whose "
                        "frame cannot be reproduced at training time"
                    )
                part["url"] = mapping[url]
                n += 1
    return n


def _record_text(record: dict) -> str:
    """All raw text of a record, concatenated: every text part of every
    message plus the target. NOT the JSON serialization -- json escapes
    newlines, and analyses contain newlines, so a needle would never match
    the escaped form."""
    parts: list[str] = []
    for m in record["messages"]:
        content = m.get("content")
        if isinstance(content, list):
            parts.extend(
                p["text"] for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
    parts.append(record["target_text"])
    return "\n".join(parts)


def _assert_no_analyst_leak(record: dict, analyses: list[str]) -> None:
    """Tripwire: no analysis generated in THIS session may appear inside the
    record (the load-side screening -- exclude_analyst + exclude_session --
    should make this impossible; if it ever regresses, abort the run rather
    than silently poison the dataset)."""
    haystack = _record_text(record)
    for analysis in analyses:
        needle = analysis[:_TRIPWIRE_PREFIX_LEN]
        if needle and needle in haystack:
            raise RuntimeError(
                "ANALYST LEAK: a current-session analyst analysis appears in "
                "a recorded player context -- the player peeked over the "
                "analyst's shoulder. Screening (exclude_analyst / "
                f"exclude_session) has regressed. Leaked prefix: {needle!r}"
            )


class _Shared:
    """Everything the worker threads touch together: the trace file, the
    counters, and the generation budget. One lock covers it all -- these are
    microsecond operations next to multi-second generations."""

    def __init__(self, out: Any, images_dir: Path, max_generations: int):
        self.lock = threading.Lock()
        self.out = out
        self.images_dir = images_dir
        self.max_generations = max_generations
        self.stats: Counter = Counter()
        self.rating_hist: Counter = Counter()
        self.records: list[dict] = []  #: in order written (for the plots)

    def take_generation(self) -> bool:
        """Reserve one generation slot; False once the budget is spent."""
        with self.lock:
            if self.stats["generations"] >= self.max_generations:
                return False
            self.stats["generations"] += 1
            return True

    def budget_spent(self) -> bool:
        with self.lock:
            return self.stats["generations"] >= self.max_generations


def _play_game(session: Any, game_idx: int, args: argparse.Namespace,
               shared: _Shared, session_analyses: list[str]) -> None:
    """One full game on ``session``: rounds until the gold is eaten, the
    move cap is hit, or the run-wide generation budget runs out. Buffers the
    game's records, stamps the outcome, writes them under the shared lock."""
    game_records: list[dict] = []
    won = False
    for move_idx in range(args.max_moves):
        if not shared.take_generation():
            break
        player = session.ask_player(session.DEFAULT_PLAYER_QUESTION)

        # Stable image copy FIRST (before anything else can touch the live
        # file): byte-identical to the (possibly noised) frame the player
        # just saw. The uuid in the snapshot filename keeps copies unique
        # across workers and across --append runs.
        before_path = player["before_path"]
        img_name = f"g{game_idx:04d}_m{move_idx:03d}_{Path(before_path).name}"
        shutil.copy2(before_path, shared.images_dir / img_name)

        analyst = session.ask_analyst(session.DEFAULT_ANALYST_QUESTION)
        session_analyses.append(analyst["analysis"])
        outcome = session.end_round()

        record = {
            "messages": player["messages"],
            "target_text": player["raw"],
            "meta": {
                "rating": analyst["rating"],
                "wrong_spans": analyst["wrong_spans"]["verified"],
                "unverified_spans": analyst["wrong_spans"]["unverified"],
                "action": player["action"],
                "bare_move": player["bare_move"],
                "searches": [
                    {"query": s["query"], "thought": s["thought"]}
                    for s in player["searches"]
                ],
                "gold_collected": outcome["gold_collected"],
                "game_index": game_idx,
                "move_index": move_idx,
                "session_id": player["session_id"],
            },
        }
        n_imgs = _rewrite_image_urls(
            record["messages"], {before_path: f"images/{img_name}"}
        )
        if n_imgs == 0:
            raise RuntimeError(
                "player context contains no image part -- the player "
                "answered blind; refusing to record"
            )
        # Session-scoped by design: a CONCURRENT session's analyses are as
        # visible-by-retrieval as a past game's, which is the intended
        # cross-game learning; only current-session screening is invariant.
        _assert_no_analyst_leak(record, session_analyses)
        game_records.append(record)

        with shared.lock:
            if analyst["rating"] is None:
                shared.stats["rating_missing"] += 1
            else:
                shared.rating_hist[f"{analyst['rating']:+.1f}"] += 1
            shared.stats["wrong_spans"] += len(analyst["wrong_spans"]["verified"])
            if player["action"]:
                shared.stats["moves"] += 1

        if outcome["gold_remaining"] == 0:
            won = True
            break

    # Stamp the outcome and flush the game (one writer at a time).
    last = len(game_records) - 1
    with shared.lock:
        for i, record in enumerate(game_records):
            record["meta"]["game_won"] = won
            record["meta"]["moves_from_end"] = (last - i) if won else None
            shared.out.write(json.dumps(record, ensure_ascii=False) + "\n")
            shared.records.append(record)
        shared.out.flush()
        shared.stats["games"] += 1
        shared.stats["games_won" if won else "games_lost"] += 1
        n_gen = shared.stats["generations"]
    logger.info(
        "game %d done: %s in %d round(s) (total generations: %d)",
        game_idx, "WON" if won else "lost", len(game_records), n_gen,
    )


def _worker(worker_id: int, session: Any, block: list[int], shared: _Shared,
            args: argparse.Namespace, session_analyses: list[str],
            fresh: list[bool], errors: list[BaseException]) -> None:
    """Pull game indices off the shared block queue until it drains, the
    budget runs out, or any worker errored (fail the whole run, never
    silently continue with fewer workers)."""
    try:
        while True:
            with shared.lock:
                if errors or not block:
                    return
                game_idx = block.pop(0)
            if shared.budget_spent():
                return
            if fresh[worker_id]:
                fresh[worker_id] = False  # restart() already dealt the board
            else:
                session.reset_game()
            _play_game(session, game_idx, args, shared, session_analyses)
    except BaseException as exc:
        logger.exception("datagen worker %d failed", worker_id)
        with shared.lock:
            errors.append(exc)


def run_generation(args: argparse.Namespace) -> dict[str, Any]:
    """The loop. Split from main() so a run script / notebook cell can call
    it with a Namespace built by hand."""
    from agent.model import get_model, set_default_checkpoint
    from agent.parallel_gen import BatchingProxy, GenerationDispatcher
    from agent.self_eval_session import InteractiveSelfEvalSession

    set_default_checkpoint(args.checkpoint)

    out_dir = DATA_GAME_DIR / args.label
    images_dir = out_dir / "images"
    traces_path = out_dir / "traces.jsonl"
    if traces_path.exists() and not args.append:
        raise SystemExit(
            f"{traces_path} already exists; pass --append to add to it or "
            "pick a fresh --label"
        )
    images_dir.mkdir(parents=True, exist_ok=True)

    n_workers = max(1, args.parallel)
    sessions = [
        InteractiveSelfEvalSession(log_label=f"datagen_{args.label}_w{i}")
        for i in range(n_workers)
    ]
    dispatcher: GenerationDispatcher | None = None
    if n_workers > 1:
        # One model, batched decode across the workers (see
        # agent/parallel_gen.py); the sessions themselves never know.
        dispatcher = GenerationDispatcher(get_model(), max_batch=n_workers)
        for s in sessions:
            s.model = BatchingProxy(dispatcher)
    if args.noise:
        for i, s in enumerate(sessions):
            s.image_filter = make_image_filter(
                args.seed + i, strength=INFERENCE_STRENGTH
            )

    try:
        with open(traces_path, "a", encoding="utf-8") as out:
            shared = _Shared(out, images_dir, args.max_generations)
            # Per-session tripwire corpora (cleared on memory reset).
            analyses: list[list[str]] = [[] for _ in range(n_workers)]
            for s in sessions:
                s.restart()
            fresh = [True] * n_workers

            # Games run in sequential BLOCKS of --reset-every: workers drain
            # between blocks, so the global memory reset never yanks another
            # worker's episodic memory mid-game.
            for block_start in range(0, args.games, args.reset_every):
                if shared.budget_spent():
                    logger.info("--max-generations reached; stopping.")
                    break
                if block_start > 0:
                    logger.info(
                        "NAMS hygiene: resetting episodic memory to the "
                        "seeded semantic model (game %d).", block_start,
                    )
                    sessions[0].reset_memory_to_seed()
                    for i, s in enumerate(sessions):
                        s.restart()
                        analyses[i].clear()
                        fresh[i] = True
                block = list(range(
                    block_start, min(block_start + args.reset_every, args.games)
                ))
                errors: list[BaseException] = []
                threads = [
                    threading.Thread(
                        target=_worker, name=f"datagen-w{i}",
                        args=(i, sessions[i], block, shared, args,
                              analyses[i], fresh, errors),
                    )
                    for i in range(n_workers)
                ]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                if errors:
                    raise errors[0]
    finally:
        if dispatcher is not None:
            dispatcher.close()
        for s in sessions:
            s.close()

    summary = {
        **{k: shared.stats[k] for k in sorted(shared.stats)},
        "rating_histogram": dict(sorted(shared.rating_hist.items())),
        "traces": str(traces_path),
        "parallel": n_workers,
    }
    (out_dir / "generation_stats.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("done: %s", json.dumps(summary, indent=2))
    if shared.stats["rating_missing"]:
        logger.warning(
            "%d record(s) have no parseable RATING and will be DROPPED by "
            "GameTraceSource at load time.", shared.stats["rating_missing"],
        )
    try:
        from training.datagen_plots import write_datagen_plots
        plot_dir = write_datagen_plots(args.label, summary, shared.records)
        summary["plots"] = str(plot_dir)
        logger.info("stats plots: %s", plot_dir)
    except Exception:
        # Plots are a convenience, never worth losing a finished run over --
        # but say so loudly.
        logger.exception("stats plotting failed (traces are safe on disk)")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--label", required=True,
                   help="iteration label; writes data_game/<label>/")
    p.add_argument("--checkpoint", default=None,
                   help="adapter under weights/<architecture>/ to generate "
                        "with (None = bare HF weights)")
    p.add_argument("--games", type=int, default=60,
                   help="games to play (default 60; volume rationale in "
                        "TRAINING_GAME_TRACES.md)")
    p.add_argument("--max-moves", type=int, default=50,
                   help="rounds per game before it counts as lost (the "
                        "formal game definition: gold eaten or this cap; "
                        "default 50 -- early wandering traces carry little "
                        "signal per round, and each round costs two "
                        "generations)")
    p.add_argument("--max-generations", type=int, default=3000,
                   help="hard cap on player generations for the whole run")
    p.add_argument("--parallel", type=int, default=3,
                   help="concurrent game sessions sharing one model via "
                        "batched decode (agent/parallel_gen.py); 1 = the "
                        "plain sequential loop")
    p.add_argument("--reset-every", type=int, default=100,
                   help="reset NAMS episodic memory (tips survive) every N "
                        "games (default 100 per TRAINING_EXTRA_DATASETS.md)")
    p.add_argument("--seed", type=int, default=17,
                   help="seed for the inference-time image noise stream")
    p.add_argument("--noise", dest="noise", action="store_true", default=True)
    p.add_argument("--no-noise", dest="noise", action="store_false",
                   help="disable inference-time frame degradation")
    p.add_argument("--append", action="store_true",
                   help="append to an existing traces.jsonl instead of failing")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run_generation(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
