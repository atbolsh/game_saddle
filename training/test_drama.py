"""TEMPORARY morning-review counters for training/current_drama.md.

Not a library. Not a selftest. Delete when the watch-list is done.
Skips pruned tombstones for anything that needs target_text; cone and
oracle_verdict still read meta on pruned rows.

    python -m training.test_drama --traces data_game/<prefix>_iter<k>/traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from training.game_traces import oracle_verdict

CONE = math.radians(20.0)
REMEMBER_RE = re.compile(r"\[REMEMBER\s+target\s*:", re.I)


def _load(path: Path) -> list[dict]:
    recs = []
    for line in path.open(encoding="utf-8"):
        if line.strip():
            recs.append(json.loads(line))
    return recs


def _full(rec: dict) -> bool:
    return not rec.get("pruned") and rec.get("target_text") is not None


def thought_rate(recs: list[dict]) -> tuple:
    n = hit = 0
    for rec in recs:
        if not _full(rec):
            continue
        t = (rec.get("target_text") or "").lstrip()
        n += 1
        if t.lower().startswith("thought") or t.lower().startswith(".thought"):
            hit += 1
    return hit, n, (hit / n if n else None)


def remember_after_eat(recs: list[dict]) -> tuple:
    games: dict[tuple, list] = defaultdict(list)
    for rec in recs:
        if not _full(rec):
            continue
        meta = rec["meta"]
        games[(meta.get("session_id"), meta.get("game_index"))].append(rec)
    missing = total = 0
    for group in games.values():
        group.sort(key=lambda r: int(r["meta"].get("move_index") or 0))
        eat_at = {int(r["meta"]["move_index"])
                  for r in group
                  if int(r["meta"].get("gold_collected") or 0) > 0
                  and r["meta"].get("move_index") is not None}
        for r in group:
            mi = r["meta"].get("move_index")
            if mi is None or (int(mi) - 1) not in eat_at:
                continue
            if r["meta"].get("perception"):
                continue
            total += 1
            if not REMEMBER_RE.search(r.get("target_text") or ""):
                missing += 1
    return missing, total, (missing / total if total else None)


def _quantile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, int(p * (len(sorted_vals) - 1)))
    return sorted_vals[i]


def cone_forgive(recs: list[dict], rating_cut: float = 0.8) -> dict:
    n_out = n_forgive = 0
    hist: list[float] = []
    for rec in recs:
        meta = rec["meta"]
        if meta.get("action") != "FORWARD":
            continue
        if meta.get("oracle_ray_hit"):
            continue
        rel = meta.get("oracle_rel_bearing")
        if rel is None or abs(float(rel)) <= CONE:
            continue
        n_out += 1
        r = meta.get("rating")
        if r is not None:
            hist.append(float(r))
        if r is not None and float(r) >= rating_cut:
            n_forgive += 1
    hist.sort()
    return {
        "forgive": n_forgive,
        "out_of_cone": n_out,
        "rate": (n_forgive / n_out if n_out else None),
        "rating_n": len(hist),
        "rating_min": (hist[0] if hist else None),
        "rating_p50": _quantile(hist, 0.5),
        "rating_p90": _quantile(hist, 0.9),
        "rating_max": (hist[-1] if hist else None),
    }


def counted_turn_extras(recs: list[dict]) -> dict:
    n_ct = sum(
        1 for r in recs
        if r["meta"].get("turn_count") and int(r["meta"]["turn_count"]) > 1
    )
    verdicts: Counter[str] = Counter()
    for rec in recs:
        m = rec["meta"]
        verdicts[oracle_verdict(
            m.get("action"), m.get("oracle_move"),
            m.get("oracle_rel_bearing"), m.get("oracle_ray_hit"),
            m.get("turn_count"),
        )] += 1

    games: dict[tuple, list] = defaultdict(list)
    for rec in recs:
        if not _full(rec):
            continue
        meta = rec["meta"]
        games[(meta.get("session_id"), meta.get("game_index"))].append(rec)
    miss = n_after = 0
    for group in games.values():
        group.sort(key=lambda r: int(r["meta"].get("move_index") or 0))
        for i, r in enumerate(group[:-1]):
            m = r["meta"]
            if m.get("action") not in ("CLOCK", "ANTICLOCK"):
                continue
            tc = m.get("turn_count")
            if not tc or int(tc) <= 1:
                continue
            nxt = group[i + 1]["meta"]
            n_after += 1
            if (nxt.get("action") in ("CLOCK", "ANTICLOCK")
                    and nxt.get("oracle_ray_hit")):
                miss += 1
    return {
        "counted_turns_gt1": n_ct,
        "n_records": len(recs),
        "oracle_verdicts": dict(verdicts),
        "missed_fwd_after_counted_turn": (
            miss, n_after, (miss / n_after if n_after else None),
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--traces", required=True,
                   help="path to traces.jsonl (player file, not analyst)")
    args = p.parse_args(argv)
    path = Path(args.traces)
    recs = _load(path)
    print(f"loaded {len(recs)} records from {path}")
    print("thought", thought_rate(recs))
    print("remember_after_eat", remember_after_eat(recs))
    print("cone", cone_forgive(recs))
    extras = counted_turn_extras(recs)
    print("counted_turns_gt1", extras["counted_turns_gt1"],
          "of", extras["n_records"])
    print("oracle_verdicts", extras["oracle_verdicts"])
    print("missed_fwd_after_counted_turn",
          extras["missed_fwd_after_counted_turn"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
