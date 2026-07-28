"""End-of-datagen summary plots: is the reward signal worth training on?

Called automatically by ``training/generate_game_traces.py`` after a run;
figures + a copy of the stats land in ``logs/datagen_stats_<label>_<stamp>/``.
The one to look at FIRST is the rating histogram: if the analyst's grades
are compressed into a narrow positive band and verified WRONG spans are
rare, the effective per-token reward is nearly uniform and the batch will
train like plain self-SFT -- know that BEFORE spending the GPU-days
(TRAINING_GAME_TRACES.md, "The reward scheme").
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def write_datagen_plots(label: str, summary: dict,
                        records: list[dict]) -> Path:
    """Write the 2x2 summary figure + per-panel data. ``records`` are the
    trace records in the order written (only ``meta`` is read)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = LOGS_DIR / f"datagen_stats_{label}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "generation_stats.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    metas = [r["meta"] for r in records]
    ratings = [m["rating"] for m in metas if m.get("rating") is not None]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"datagen {label}: {summary.get('generations', 0)} generations, "
                 f"{summary.get('games', 0)} games "
                 f"({summary.get('games_won', 0)} won)")

    # -- 1. rating histogram: THE reward-signal health check
    ax = axes[0][0]
    if ratings:
        ax.hist(ratings, bins=21, range=(-1.05, 1.05), edgecolor="black")
    ax.set_title(f"analyst ratings (n={len(ratings)}, "
                 f"{summary.get('rating_missing', 0)} unparseable)")
    ax.set_xlabel("RATING")
    ax.set_ylabel("count")

    # -- 2. rolling mean rating over the run: is quality drifting?
    ax = axes[0][1]
    if ratings:
        window = max(1, min(25, len(ratings) // 4 or 1))
        rolling = [
            sum(ratings[max(0, i - window + 1): i + 1])
            / len(ratings[max(0, i - window + 1): i + 1])
            for i in range(len(ratings))
        ]
        ax.plot(rolling)
        ax.axhline(0.0, linewidth=0.8)
    ax.set_title(f"rolling mean rating (window {max(1, min(25, len(ratings) // 4 or 1))})")
    ax.set_xlabel("generation (rated only)")
    ax.set_ylabel("mean RATING")

    # -- 3. game lengths, wins marked
    ax = axes[1][0]
    by_game: dict[int, dict] = {}
    for m in metas:
        g = by_game.setdefault(m["game_index"], {"rounds": 0, "won": False})
        g["rounds"] = max(g["rounds"], m["move_index"] + 1)
        g["won"] = g["won"] or bool(m.get("game_won"))
    if by_game:
        idxs = sorted(by_game)
        rounds = [by_game[i]["rounds"] for i in idxs]
        colors = ["tab:green" if by_game[i]["won"] else "tab:gray"
                  for i in idxs]
        ax.bar(idxs, rounds, color=colors, edgecolor="black")
    ax.set_title("rounds per game (green = won)")
    ax.set_xlabel("game index")
    ax.set_ylabel("rounds")

    # -- 4. WRONG-span rate: the sharpest token-level signal
    ax = axes[1][1]
    flags = [1 if m.get("wrong_spans") else 0 for m in metas]
    if flags:
        window = max(1, min(25, len(flags) // 4 or 1))
        rolling = [
            sum(flags[max(0, i - window + 1): i + 1])
            / len(flags[max(0, i - window + 1): i + 1])
            for i in range(len(flags))
        ]
        ax.plot(rolling)
        ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"share of replies with a verified WRONG span "
        f"(total spans: {summary.get('wrong_spans', 0)})"
    )
    ax.set_xlabel("generation")
    ax.set_ylabel("rolling share")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "summary.png", dpi=120)
    plt.close(fig)
    return out_dir
