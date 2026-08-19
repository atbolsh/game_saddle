"""Headless self-eval datagen: play games, grade every move, write traces.

The default data generator of the self-training loop (TRAINING_GAME_TRACES.md).
Drives the EXACT machinery of the interactive self-eval notebook --
:class:`agent.self_eval_session.InteractiveSelfEvalSession` with its default
player and analyst questions, same prompts, same memory screening -- with no
human in the loop:

    per round:  ask_player(DEFAULT_PLAYER_QUESTION,
                           or a perception question at --question-rate)
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
  * ``meta``      -- rating, wrong spans, action, game/move indices, outcome,
    the round's agent-to-gold distances (``dist_to_gold_before/after``,
    normalized units; the rating-independent quality cross-check), the raw
    engine-oracle facts for the position the move was made from
    (``oracle_move`` / ``oracle_rel_bearing`` / ``oracle_ray_hit`` --
    ``_oracle_meta``; graded train-side by ``game_traces.oracle_verdict``),
    and the exact ``question`` the round asked (perception rounds are the
    records whose question differs from the default move request).

Analyst text is stored to NAMS exactly as in the notebook and never enters
any PLAYER record's ``messages``/``target_text`` -- it exists in player
training only as annotation. As a TRIPWIRE against a screening regression,
every analyst analysis generated in the current session is checked against
each player record's serialized player context; a hit aborts the run (a
silently poisoned dataset is the worst outcome).

A SECOND file, ``data_game/<label>/analyst_traces.jsonl``, records each
analyst exchange itself -- the exact analyst prompt (privileged: settings,
unscrubbed context) plus the analysis it produced, referencing the same
stable frame copies. It feeds ``AnalystTraceSource`` (training/
game_traces.py): a KD-vs-frozen-base replay anchor that keeps analyst
behavior from drifting while the shared weights train on player data. The
strict player/analyst separation is structural: the two files are never
read by the same source.

Housekeeping per the committed plan: frames are noised at inference
(mild label-safe degradation, ``--no-noise`` to disable) and NAMS episodic
memory is reset to the seeded semantic model (tips are ``Preference`` nodes
and survive) every ``--reset-every`` games AND at run start (2026-08-06:
orchestrated 60-game epochs never reach the 100-game block boundary, and
nothing reset between epochs, so multi-day runs grew episodic memory
unboundedly). An ``--append`` resume keeps the epoch's own memories.

**Degeneracy fuse**: ``DEGENERACY_FUSE`` consecutive generations with no
parseable RATING and no parseable move mean the checkpoint is collapsed,
not unlucky -- the run drains all workers and exits ``EXIT_POISONED`` (3)
so the orchestrator stops instead of burning hours on gibberish (the
2026-08-01 collapse generated ~15h of unparseable output before anyone
noticed). Per-move gold distances (``meta.dist_to_gold_before/after``) are
recorded as a rating-independent quality cross-check.

**Parallelism** (``--parallel``, default 12): N sessions play N games
concurrently, one worker thread each, sharing ONE model through
``agent.parallel_gen`` -- concurrent generations merge into batched decode
calls (batch-1 decode is bandwidth-bound, so extra rows are cheap).
Measured 2026-07-30 (96 GB box, base weights): serial 24.1 s/gen,
``--parallel 10`` 8.5, ``--parallel 24`` 6.4 -- returns diminish but never
invert. 16 looked like VRAM headroom until 2026-08-17: 16 long KV caches
plus the NAMS MiniLM embedder on the same GPU died with
``CUBLAS_STATUS_ALLOC_FAILED``. Default 12 is that lesson. NAMS resets happen at
BLOCK boundaries: games run in
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
import math
import random
import shutil
import sys
import threading
import time
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

#: How much of each stored analyst analysis the leak tripwire matches on
#: (belt on top of the ``[ANALYST]`` tag check). Long enough to never fire
#: on a coincidental phrase, short enough to catch a truncated leak.
_TRIPWIRE_PREFIX_LEN = 100

#: DEGENERACY FUSE (2026-08-01 postmortem): a collapsed checkpoint produces
#: replies with NO parseable move AND analyses with NO parseable RATING --
#: 100% of its output is dropped at training time, so every further
#: generation is pure waste (the collapsed weekend burned ~15h generating
#: gibberish across 5 epochs). After this many CONSECUTIVE degenerate
#: generations (run-wide, across all workers) the run declares the
#: checkpoint poisoned, drains every worker, and exits with
#: EXIT_POISONED so the orchestrator can stop the whole loop. 25 is far
#: past anything a healthy model produces (a healthy run breaks a streak
#: within a handful of generations) yet costs only ~3 minutes at
#: --parallel 12 to trip.
DEGENERACY_FUSE = 25

#: Distinct exit code for "the checkpoint is poisoned" (vs 1 = ordinary
#: crash): run_weekend treats it as fatal for the whole run, not a
#: retry-and-continue failure.
EXIT_POISONED = 3

#: Perception-question rounds: with probability --question-rate a round asks
#: one of these instead of the default move request. The player must answer
#: in prose with NO move token (the scene-play prompt says so explicitly);
#: the analyst grades the answer against the exact settings it already
#: receives, and an unrequested move token is itself a graded mistake.
#:
#: DIRECTION BALANCE: the pool is a list of GROUPS of mirrored variants --
#: sampling picks a group uniformly, then a variant uniformly within it, so
#: "is the gold to your left?" and "... to your right?" (and above/below)
#: are asked with exactly the same probability, and the data never teaches a
#: directional prior. Questions drawn from the real interactive logs and the
#: scene-play prompt's own examples.
PERCEPTION_QUESTION_GROUPS: list[list[str]] = [
    ["Are you facing the gold?"],
    [
        "Is the gold to your left?",
        "Is the gold to your right?",
        "Is the gold above you?",
        "Is the gold below you?",
    ],
    [
        "Disregard the gold. Are you facing generally to the left or right "
        "of the screen?",
        "Disregard the gold. Are you facing generally toward the top or the "
        "bottom of the screen?",
    ],
    ["Which way is your eye pointing, in clock terms?"],
    [
        "Is the gold closer to the top or the bottom of the board?",
        "Is the gold closer to the left or the right edge of the board?",
    ],
    ["Are you close to any wall right now?"],
]


def _sample_perception_question(rng: random.Random,
                                question_rate: float) -> str | None:
    """One draw of the round's question: a perception question with
    probability ``question_rate`` (group-uniform, then variant-uniform --
    see PERCEPTION_QUESTION_GROUPS), else None (= the default move
    request)."""
    if rng.random() >= question_rate:
        return None
    return rng.choice(rng.choice(PERCEPTION_QUESTION_GROUPS))


def _dist_to_gold(settings: dict) -> float | None:
    """Agent-to-NEAREST-gold distance in normalized board units ([0,1]
    square, the ``game_io`` convention), from a serialized settings dict.
    None when no gold remains. Recorded per move (before/after) as a
    rating-INDEPENDENT quality signal: the analyst can be wrong or absent,
    but "did the player get closer to the gold" cannot -- it is the
    morning cross-check on the rating distribution, not (yet) a training
    input."""
    gold = settings.get("gold") or []
    if not gold:
        return None
    ax, ay = settings["agent_x"], settings["agent_y"]
    return round(min(math.hypot(gx - ax, gy - ay) for gx, gy in gold), 4)


def _oracle_meta(settings: dict) -> dict:
    """Engine-oracle facts for the CURRENT position, stamped into each
    move's meta (computed before the move executes, next to
    dist_to_gold_before). Only RAW FACTS are stored -- classification into
    correct/neutral/wrong lives train-side (game_traces.oracle_verdict,
    with its own ULTRAVIOLET crutch block), so thresholds can be retuned
    without regenerating data.

      * oracle_rel_bearing: signed bearing (radians, wrapped to [-pi, pi))
        from the agent's facing to the NEAREST gold. The game_io compass
        convention: bearing = atan2(dx, dy), theta increases clockwise,
        so POSITIVE = the gold is clockwise of the facing;
      * oracle_ray_hit: whether stepping straight ahead intersects ANY
        gold within pickup reach (agent_r + gold_r). EXACT, because bare
        levels have no internal walls to route around;
      * oracle_move: FORWARD on a ray hit, else the shorter rotation
        toward the nearest gold.

    Empty when no gold remains (game already won)."""
    gold = settings.get("gold") or []
    if not gold:
        return {}
    ax, ay = settings["agent_x"], settings["agent_y"]
    theta = settings["direction"]
    reach = settings["agent_r"] + settings["gold_r"]

    _, gx, gy = min(
        (math.hypot(gx - ax, gy - ay), gx, gy) for gx, gy in gold
    )
    rel = (math.atan2(gx - ax, gy - ay) - theta + math.pi) \
        % (2 * math.pi) - math.pi

    fx, fy = math.sin(theta), math.cos(theta)
    ray_hit = False
    for cx, cy in gold:
        dx, dy = cx - ax, cy - ay
        # positive projection along the facing ray, perpendicular
        # miss distance within pickup reach
        if dx * fx + dy * fy > 0 and abs(dx * fy - dy * fx) <= reach:
            ray_hit = True
            break

    if ray_hit:
        move = "FORWARD"
    else:
        move = "CLOCK" if rel > 0 else "ANTICLOCK"
    return {
        "oracle_move": move,
        "oracle_rel_bearing": round(rel, 4),
        "oracle_ray_hit": ray_hit,
    }


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
    """Tripwire: no analysis generated in THIS session, and no ``[ANALYST]``
    tag, may appear inside the player record (the load-side screening --
    per-line tag + exclude_analyst + exclude_session -- should make this
    impossible; if it ever regresses, abort the run rather than silently
    poison the dataset)."""
    from agent.memory import ANALYST_TAG

    haystack = _record_text(record)
    if ANALYST_TAG in haystack:
        raise RuntimeError(
            "ANALYST LEAK: the [ANALYST] tag appears in a recorded player "
            "context -- the player peeked over the analyst's shoulder. "
            "Screening (per-line tag / exclude_analyst / exclude_session) "
            "has regressed."
        )
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
    counters, the generation budget, and the degeneracy fuse. One lock
    covers it all -- these are microsecond operations next to multi-second
    generations."""

    def __init__(self, out: Any, analyst_out: Any, images_dir: Path,
                 max_generations: int, checkpoint: str | None = None):
        self.lock = threading.Lock()
        self.out = out
        self.analyst_out = analyst_out
        self.images_dir = images_dir
        self.max_generations = max_generations
        self.checkpoint = checkpoint  #: named in the fuse's ERROR line
        self.stats: Counter = Counter()
        self.rating_hist: Counter = Counter()
        self.records: list[dict] = []  #: in order written (for the plots)
        #: Degeneracy fuse state (see DEGENERACY_FUSE): consecutive
        #: rating-less AND move-less generations, run-wide.
        self.degenerate_streak = 0
        self.poisoned = False

    def take_generation(self) -> bool:
        """Reserve one generation slot; False once the budget is spent or
        the degeneracy fuse tripped (poisoned = drain everyone)."""
        with self.lock:
            if self.poisoned:
                return False
            if self.stats["generations"] >= self.max_generations:
                return False
            self.stats["generations"] += 1
            return True

    def budget_spent(self) -> bool:
        with self.lock:
            return (self.poisoned
                    or self.stats["generations"] >= self.max_generations)

    def note_degeneracy(self, degenerate: bool) -> None:
        """Feed the fuse one generation's verdict (a parseable RATING or a
        parseable move both count as signs of life). Trips at
        DEGENERACY_FUSE consecutive degenerates, after which
        :meth:`take_generation` refuses and all workers drain."""
        with self.lock:
            if not degenerate:
                self.degenerate_streak = 0
                return
            self.stats["degenerate_generations"] += 1
            self.degenerate_streak += 1
            if self.degenerate_streak >= DEGENERACY_FUSE and not self.poisoned:
                self.poisoned = True
                logger.error(
                    "DEGENERACY FUSE TRIPPED: %d consecutive generations "
                    "with no parseable RATING and no parseable move -- "
                    "checkpoint %r is POISONED. Draining workers and "
                    "exiting with code %d (run_weekend stops the whole "
                    "run on this code).",
                    self.degenerate_streak,
                    self.checkpoint or "<bare HF weights>", EXIT_POISONED,
                )


def _play_game(session: Any, game_idx: int, args: argparse.Namespace,
               shared: _Shared, session_analyses: list[str],
               qrng: random.Random) -> None:
    """One full game on ``session``: rounds until the gold is eaten, the
    move cap is hit, or the run-wide generation budget runs out. Buffers the
    game's records, stamps the outcome, writes them under the shared lock.

    Legacy eat-gold win; ``[END_GAME]`` may appear because the unified
    prompt mentions it, but it does not terminate datagen games (the
    base ``end_round`` grades it without applying a board action). The
    next player question then says the token is unavailable in this
    mode. Analyst prompts are left as-is until training switches
    (FUTURE_GOALS goal 10).

    Each round asks either the default move request or (with probability
    ``--question-rate``, drawn from ``qrng``) a perception question --
    those rounds do not advance the game (a correct answer is prose with no
    move token; a mistakenly emitted token still propagates, exactly as the
    game contract promises, and the analyst grades it as unrequested)."""
    from agent import game_io  # heavy (pygame); already loaded by the session

    game_records: list[dict] = []
    analyst_records: list[dict] = []
    won = False
    for move_idx in range(args.max_moves):
        if not shared.take_generation():
            break
        settings_before = game_io.game_to_settings_dict(session.game)
        dist_before = _dist_to_gold(settings_before)
        # Engine-oracle facts for THIS position (before the move executes);
        # raw facts only, classified train-side (_oracle_meta docstring).
        oracle = _oracle_meta(settings_before)
        question = _sample_perception_question(qrng, args.question_rate)
        is_perception = question is not None
        if question is None:
            question = session.DEFAULT_PLAYER_QUESTION
        player = session.ask_player(question)

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
        # After the pending move propagated: dist_before - dist_to_gold_after
        # > 0 means this round moved the player toward the gold.
        dist_after = _dist_to_gold(game_io.game_to_settings_dict(session.game))

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
                # Rating-independent quality signal (see _dist_to_gold);
                # after is None when the round's move ATE the gold.
                "dist_to_gold_before": dist_before,
                "dist_to_gold_after": dist_after,
                # oracle_move / oracle_rel_bearing / oracle_ray_hit
                # (raw engine-oracle facts, _oracle_meta docstring)
                **oracle,
                "game_index": game_idx,
                "move_index": move_idx,
                "session_id": player["session_id"],
                "question": question,
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

        # Analyst record for the KD anchor (AnalystTraceSource): the exact
        # analyst prompt + the analysis it produced, referencing the SAME
        # stable image copy as the player record. Kept in a SEPARATE file so
        # no loader can ever mix analyst text into player training data.
        # The leak tripwire does NOT apply here -- these records contain
        # analyses by construction. Skipped (and counted) when the accepted
        # generation was a truncated search call (budget exhausted
        # mid-search): never anchor on a cut-off tool call.
        truncated_search = analyst["replies"][-1]["search_query"] is not None
        if truncated_search:
            with shared.lock:
                shared.stats["analyst_skipped_search"] += 1
        else:
            analyst_record = {
                "messages": analyst["messages"],
                "target_text": analyst["analysis"],
                "meta": {
                    "rating": analyst["rating"],
                    "wrong_spans": analyst["wrong_spans"]["verified"],
                    "unverified_spans": analyst["wrong_spans"]["unverified"],
                    "question": analyst["question"],
                    "player_question": question,
                    "n_search_calls": analyst["n_search_calls"],
                    "game_index": game_idx,
                    "move_index": move_idx,
                    "session_id": player["session_id"],
                },
            }
            n_imgs = _rewrite_image_urls(
                analyst_record["messages"], {before_path: f"images/{img_name}"}
            )
            if n_imgs == 0:
                raise RuntimeError(
                    "analyst context contains no image part -- the analyst "
                    "reviewed blind; refusing to record"
                )
            analyst_records.append(analyst_record)

        with shared.lock:
            if analyst["rating"] is None:
                shared.stats["rating_missing"] += 1
            else:
                shared.rating_hist[f"{analyst['rating']:+.1f}"] += 1
            shared.stats["wrong_spans"] += len(analyst["wrong_spans"]["verified"])
            if player["action"]:
                shared.stats["moves"] += 1
            if is_perception:
                shared.stats["perception_rounds"] += 1

        # Degeneracy fuse: no parseable RATING anywhere in the analysis AND
        # no parseable move token in the reply (bare or bracketed) means
        # this generation is un-trainable gibberish; a long run of those is
        # a collapsed checkpoint, not bad luck (see DEGENERACY_FUSE).
        shared.note_degeneracy(
            analyst["rating"] is None
            and not player["action"] and not player["bare_move"]
        )

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
        for record in analyst_records:
            shared.analyst_out.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
        shared.analyst_out.flush()
        shared.stats["analyst_records"] += len(analyst_records)
        shared.stats["games"] += 1
        shared.stats["games_won" if won else "games_lost"] += 1
        n_gen = shared.stats["generations"]
    logger.info(
        "game %d done: %s in %d round(s) (total generations: %d)",
        game_idx, "WON" if won else "lost", len(game_records), n_gen,
    )


def _worker(worker_id: int, session: Any, block: list[int], shared: _Shared,
            args: argparse.Namespace, session_analyses: list[str],
            fresh: list[bool], errors: list[BaseException],
            qrng: random.Random, dispatcher: Any = None) -> None:
    """Pull game indices off the shared block queue until it drains, the
    budget runs out, or any worker errored (fail the whole run, never
    silently continue with fewer workers).

    The dispatcher liveness hooks bracket the whole loop: while registered,
    the phase-locking scheduler (agent/parallel_gen.py) may hold OTHER
    workers' requests waiting for this worker's next generation, so
    deregistering on every exit path is what keeps the run deadlock-free."""
    if dispatcher is not None:
        dispatcher.worker_started()
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
            _play_game(session, game_idx, args, shared, session_analyses, qrng)
    except BaseException as exc:
        logger.exception("datagen worker %d failed", worker_id)
        with shared.lock:
            errors.append(exc)
    finally:
        if dispatcher is not None:
            dispatcher.worker_finished()


def run_generation(args: argparse.Namespace) -> dict[str, Any]:
    """The loop. Split from main() so a run script / notebook cell can call
    it with a Namespace built by hand."""
    from agent import memory as mem
    from agent.model import get_model, set_default_checkpoint
    from agent.parallel_gen import BatchingProxy, GenerationDispatcher
    from agent.self_eval_session import InteractiveSelfEvalSession

    t_start = time.perf_counter()
    set_default_checkpoint(args.checkpoint)

    out_dir = DATA_GAME_DIR / args.label
    images_dir = out_dir / "images"
    traces_path = out_dir / "traces.jsonl"
    analyst_path = out_dir / "analyst_traces.jsonl"
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
        with open(traces_path, "a", encoding="utf-8") as out, \
                open(analyst_path, "a", encoding="utf-8") as analyst_out:
            shared = _Shared(out, analyst_out, images_dir,
                             args.max_generations,
                             checkpoint=args.checkpoint)
            # Per-session tripwire corpora (cleared on memory reset).
            analyses: list[list[str]] = [[] for _ in range(n_workers)]
            # Per-worker question rng, persistent across blocks so no two
            # blocks (or workers) repeat the same question sequence.
            qrngs = [random.Random(args.seed * 1000 + i)
                     for i in range(n_workers)]
            # NAMS hygiene at RUN START (2026-08-06): the block-boundary
            # reset below never fires on orchestrated epochs (60-game
            # epochs < --reset-every 100), and nothing resets BETWEEN
            # epochs, so a multi-day run grew episodic memory unboundedly
            # -- stale reasoning traces from obsolete checkpoints
            # polluting analyst retrieval, and retrieval itself slowing
            # down. A fresh run now starts from the seeded semantic model
            # (tips are Preference nodes and SURVIVE, as everywhere).
            # --append (mid-epoch crash resume) skips it: the epoch keeps
            # its own memories.
            if not args.append:
                logger.info(
                    "NAMS hygiene: run-start reset of episodic memory to "
                    "the seeded semantic model (tips survive; --append "
                    "would skip this)."
                )
                # The deletion census is the clearing audit trail: it
                # counts what the PREVIOUS stage left behind, so it
                # should stay ~one stage's worth run over run -- counts
                # growing across epochs mean a reset is being skipped.
                deleted = sessions[0].reset_memory_to_seed()
                logger.info(
                    "NAMS hygiene: run-start reset deleted %d episodic "
                    "node(s): %s", sum(deleted.values()), deleted,
                )
            # Heal core tips once before workers spawn so concurrent
            # sessions only ever read Preference rows (first-run insert
            # races would trip get_core_tips' duplicate check).
            sessions[0]._run(mem.ensure_core_tips(sessions[0].client))
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
                    deleted = sessions[0].reset_memory_to_seed()
                    logger.info(
                        "NAMS hygiene: block reset deleted %d episodic "
                        "node(s): %s", sum(deleted.values()), deleted,
                    )
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
                              analyses[i], fresh, errors, qrngs[i],
                              dispatcher),
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

    wall_s = time.perf_counter() - t_start
    n_gen = shared.stats.get("generations", 0)
    summary = {
        **{k: shared.stats[k] for k in sorted(shared.stats)},
        "rating_histogram": dict(sorted(shared.rating_hist.items())),
        "traces": str(traces_path),
        "analyst_traces": str(analyst_path),
        "parallel": n_workers,
        #: True = the degeneracy fuse tripped; main() exits EXIT_POISONED.
        "poisoned": shared.poisoned,
        # THE number to compare across --parallel settings (model load
        # included; identical either way, so it cancels in the comparison).
        "wall_seconds": round(wall_s, 1),
        "seconds_per_generation": round(wall_s / n_gen, 2) if n_gen else None,
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
    p.add_argument("--parallel", type=int, default=12,
                   help="concurrent game sessions sharing one model via "
                        "batched decode (agent/parallel_gen.py); 1 = the "
                        "plain sequential loop; default 12 leaves GPU "
                        "headroom for the NAMS MiniLM embedder (16 died "
                        "with CUBLAS_STATUS_ALLOC_FAILED on a 96 GB box, "
                        "2026-08-17; module docstring)")
    p.add_argument("--reset-every", type=int, default=100,
                   help="reset NAMS episodic memory (tips survive) every N "
                        "games (default 100 per TRAINING_EXTRA_DATASETS.md)")
    p.add_argument("--question-rate", type=float, default=0.15,
                   help="probability that a round asks a perception "
                        "question (direction-balanced pool, see "
                        "PERCEPTION_QUESTION_GROUPS) instead of the default "
                        "move request (default 0.15)")
    p.add_argument("--seed", type=int, default=17,
                   help="seed for the inference-time image noise stream and "
                        "the question sampler")
    p.add_argument("--noise", dest="noise", action="store_true", default=True)
    p.add_argument("--no-noise", dest="noise", action="store_false",
                   help="disable inference-time frame degradation")
    p.add_argument("--append", action="store_true",
                   help="append to an existing traces.jsonl instead of failing")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    summary = run_generation(build_parser().parse_args(argv))
    if summary.get("poisoned"):
        # The distinctive code is the ENTIRE point: run_weekend maps 3 to
        # "stop the whole run", vs 1 = ordinary crash = retry once.
        return EXIT_POISONED
    return 0


if __name__ == "__main__":
    sys.exit(main())
