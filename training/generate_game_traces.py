"""Headless self-eval datagen: play games, grade every move, write traces.

The default data generator of the self-training loop (TRAINING_GAME_TRACES.md).
Drives the EXACT machinery of the interactive self-eval notebook --
:class:`agent.self_eval_session.InteractiveSelfEvalSession` with its default
player and analyst questions, same prompts, same memory screening -- with no
human in the loop:

    per round:  ask_player(DEFAULT_PLAYER_QUESTION)
                -> ask_analyst(DEFAULT_ANALYST_QUESTION)   (one exchange)
                -> end_round()                             (move propagates)

**A game is formally: the gold eaten, OR --max-moves (default 200) player
rounds**, whichever comes first. Records buffer per game and are written at
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


def run_generation(args: argparse.Namespace) -> dict[str, Any]:
    """The loop. Split from main() so a run script / notebook cell can call
    it with a Namespace built by hand."""
    from agent.model import set_default_checkpoint
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

    stats: Counter = Counter()
    rating_hist: Counter = Counter()

    session = InteractiveSelfEvalSession(log_label=f"datagen_{args.label}")
    if args.noise:
        session.image_filter = make_image_filter(
            args.seed, strength=INFERENCE_STRENGTH
        )
    try:
        session.restart()
        # Analyses produced in the CURRENT session (cleared on memory reset
        # + restart) -- the tripwire corpus.
        session_analyses: list[str] = []

        with open(traces_path, "a", encoding="utf-8") as out:
            for game_idx in range(args.games):
                if stats["generations"] >= args.max_generations:
                    logger.info("--max-generations reached; stopping.")
                    break
                if game_idx > 0 and game_idx % args.reset_every == 0:
                    logger.info(
                        "NAMS hygiene: resetting episodic memory to the "
                        "seeded semantic model (game %d).", game_idx,
                    )
                    session.reset_memory_to_seed()
                    session.restart()
                    session_analyses = []
                elif game_idx > 0:
                    session.reset_game()

                game_records: list[dict] = []
                won = False
                for move_idx in range(args.max_moves):
                    player = session.ask_player(
                        session.DEFAULT_PLAYER_QUESTION
                    )
                    stats["generations"] += 1

                    # Stable image copy FIRST (before anything else can
                    # touch the live file): byte-identical to the (possibly
                    # noised) frame the player just saw.
                    before_path = player["before_path"]
                    img_name = f"g{game_idx:04d}_m{move_idx:03d}_{Path(before_path).name}"
                    shutil.copy2(before_path, images_dir / img_name)

                    analyst = session.ask_analyst(
                        session.DEFAULT_ANALYST_QUESTION
                    )
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
                            "player context contains no image part -- the "
                            "player answered blind; refusing to record"
                        )
                    _assert_no_analyst_leak(record, session_analyses)
                    game_records.append(record)

                    if analyst["rating"] is None:
                        stats["rating_missing"] += 1
                    else:
                        rating_hist[f"{analyst['rating']:+.1f}"] += 1
                    stats["wrong_spans"] += len(analyst["wrong_spans"]["verified"])
                    if player["action"]:
                        stats["moves"] += 1

                    if outcome["gold_remaining"] == 0:
                        won = True
                        break
                    if stats["generations"] >= args.max_generations:
                        break

                # Stamp the outcome and flush the game.
                last = len(game_records) - 1
                for i, record in enumerate(game_records):
                    record["meta"]["game_won"] = won
                    record["meta"]["moves_from_end"] = (last - i) if won else None
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                stats["games"] += 1
                stats["games_won" if won else "games_lost"] += 1
                logger.info(
                    "game %d done: %s in %d round(s) (total generations: %d)",
                    game_idx, "WON" if won else "lost", len(game_records),
                    stats["generations"],
                )
    finally:
        session.close()

    summary = {
        **{k: stats[k] for k in sorted(stats)},
        "rating_histogram": dict(sorted(rating_hist.items())),
        "traces": str(traces_path),
    }
    (out_dir / "generation_stats.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("done: %s", json.dumps(summary, indent=2))
    if stats["rating_missing"]:
        logger.warning(
            "%d record(s) have no parseable RATING and will be DROPPED by "
            "GameTraceSource at load time.", stats["rating_missing"],
        )
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
    p.add_argument("--max-moves", type=int, default=200,
                   help="rounds per game before it counts as lost "
                        "(the formal game definition; default 200)")
    p.add_argument("--max-generations", type=int, default=3000,
                   help="hard cap on player generations for the whole run")
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
