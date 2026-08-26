# Current drama

Dated watch-list for in-flight weekends. Not a design doc — morning-review
notes. Python below is a sketch; customize paths / prefixes on the box
when you ask.

---

## 2026-08-26 — counted-turn launch (this box) + `aug26_multi-gold` (other box)

### Launch (this box, `big-step`)

Wed afternoon → Tue 1 Sep 08:00 is ~136 h. At `--parallel 8` a full
epoch is roughly **12 h** (3000 gens × ~9.5 s/gen ≈ 8 h datagen; train
~2.3 h; smoke 400 gens ≈ 1.1 h; load/reset slack). 10 epochs ≈ 120 h →
Monday morning, ~16 h slack for a retry. 11 is tight; 12 overshoots.
`--parallel 8` is the post-OOM default (12 died at T~7k on 96 GB). Do
**not** `--append` the first launch: the run-start NAMS reset re-heals
the counted-turn tip rows.

Confirm the adapter directory name under `weights/<arch>/` (user-facing
name `aug16_restart_iter5`; it may have a `_stepN` suffix). Then:

```bash
python -m training.selftest t1
# tmux / nohup, repo root, NAMS up, .env loaded via agent.config
nohup python -m training.run_weekend \
  --prefix aug26_counted \
  --checkpoint aug16_restart_iter5 \
  --epochs 10 \
  --parallel 8 \
  > weekend.log 2>&1 &
```

`run_weekend` already passes `--multi-gold` to datagen. Smoke stays
sealed eat-to-win. First log line to grab: `base seed N (...)`.

Health while it runs: `grep -E 'datagen|train|smoke_eval|POISON|VRAM' weekend.log | tail`.
State: `data_game/aug26_counted_state.json`.

### Watch list (both boxes unless noted)

Traces: `data_game/<prefix>_iter<k>/traces.jsonl` (one player record per
generation). Analyst file is sibling `analyst_traces.jsonl` — not needed
for these three.

**1. `thought` / `.thought` prefix (this box first; parent if curious)**

Gemma thinking-scaffold leaking into `target_text`. Notebook: 3/19 player
gens (seq 9, 15, 20). Screening is clean — not analyst leakage. Harmless
at this rate; cloning could amplify it. Count, don't prompt-chase yet.

```python
import json
from pathlib import Path

def thought_rate(path):
    n = hit = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        t = (rec.get("target_text") or "").lstrip()
        n += 1
        if t.lower().startswith("thought") or t.lower().startswith(".thought"):
            hit += 1
    return hit, n, hit / n if n else None

print(thought_rate("data_game/aug26_counted_iter1/traces.jsonl"))
```

Morning: rate by epoch. If it climbs across iters, it's being cloned.

**2. Failure to re-remember after an eat (both boxes)**

Post-eat board update asks for a new `[REMEMBER target: ...]`. Seq 27
redescribed the next gold in OBS and never saved it. Prompt pressure
("most turns need NO REMEMBER") is the suspected cause — **do not**
stuff this into `board_update_line`; tip-block edit is tomorrow's job if
the rate is real.

Detect: the *next* move-round after `gold_collected > 0` (skip
perception). The eat record itself is the eating FORWARD.

```python
import json
from collections import defaultdict
from pathlib import Path

def remember_after_eat(path):
    games = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        meta = rec["meta"]
        key = (meta.get("session_id"), meta.get("game_index"))
        games[key].append(rec)
    missing = total = 0
    examples = []
    for recs in games.values():
        recs.sort(key=lambda r: int(r["meta"].get("move_index") or 0))
        eat_at = {int(r["meta"]["move_index"])
                  for r in recs if int(r["meta"].get("gold_collected") or 0) > 0
                  and r["meta"].get("move_index") is not None}
        for r in recs:
            mi = r["meta"].get("move_index")
            if mi is None or (int(mi) - 1) not in eat_at:
                continue
            if r["meta"].get("perception"):
                continue
            total += 1
            text = r.get("target_text") or ""
            if "[REMEMBER" not in text.upper().replace(" ", "") and \
               "[REMEMBER target:" not in text:
                # exact token, case-insensitive
                if "remember target" not in text.lower():
                    missing += 1
                    examples.append((r["meta"].get("session_id"),
                                     r["meta"].get("game_index"), mi, text[:180]))
    return missing, total, examples[:8]

print(remember_after_eat("data_game/aug26_counted_iter1/traces.jsonl")[:2])
```

(Tighten the REMEMBER match to `re.search(r'\[REMEMBER\s+target\s*:', text, re.I)`
when customizing.) Same script on the other box:
`data_game/aug26_multi-gold_iter<k>/traces.jsonl`. Compare rates — if the
parent is already high, this isn't a counted-turn regression.

**3. Out-of-cone forgiveness (both boxes)**

Analyst rates a FORWARD ≥ +0.8 while `|oracle_rel_bearing|` is outside
the 20° cone and there is no ray hit. Seq 25: computed 32°, wrote
"slightly outside", rated +0.8. Engine stamps these `wrong` at train
time (`ORACLE_WRONG_SCALE` 0.25), so the leak is in the *rating
distribution*, not the move-token span. Track the rate; if it grows,
ratings lose contrast exactly on the cone.

```python
import json, math
from pathlib import Path

CONE = math.radians(20.0)

def cone_forgive(path, rating_cut=0.8):
    n_out = n_forgive = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        meta = rec["meta"]
        if meta.get("action") != "FORWARD":
            continue
        if meta.get("oracle_ray_hit"):
            continue
        rel = meta.get("oracle_rel_bearing")
        if rel is None:
            continue
        if abs(float(rel)) <= CONE:
            continue
        n_out += 1
        r = meta.get("rating")
        if r is not None and float(r) >= rating_cut:
            n_forgive += 1
    return n_forgive, n_out, (n_forgive / n_out if n_out else None)

print(cone_forgive("data_game/aug26_counted_iter1/traces.jsonl"))
```

Other box: same, on `aug26_multi-gold` iters. Also useful: rating
histogram on those out-of-cone FORWARDs (not just the ≥0.8 cut).

**This box extra (not on the parent):** `counted_turns` in
`generation_stats.json`; `count_off` share when traces hit
`GameTraceSource` (train log line `oracle verdicts {...}`). Missed-forward
*after* a counted turn (CLOCK/ANTICLOCK with `turn_count>1`, next record
is a turn under `oracle_ray_hit`) is the leak the notebook showed.

### Other box (`aug26_multi-gold`, still running)

Do not sync `big-step` onto it. Same three rates on whatever `iter*`
dirs exist. If you stop it: orchestrator first
(`pkill -TERM -f 'python -m training.run_weekend'`), then the datagen
child. Kill order matters.
