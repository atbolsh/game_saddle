"""The formal test suite: numbered stages, one command each, PASS/FAIL lines.

Run on the REMOTE box (GPU + NAMS), in order, as soon as setup_env.sh has
finished::

    python -m training.selftest t0     # seconds   env + imports
    python -m training.selftest t1     # seconds   pure-python units
    python -m training.selftest t2     # seconds   materialized data + manifest
    python -m training.selftest t3     # minutes   4-bit load, forward/backward
    python -m training.selftest t4     # ~15-30m   batch parity + smoke train + rollback
    python -m training.selftest t5     # ~10-20m   datagen 2 games, --parallel 2
    python -m training.selftest t6     # minutes   batched-vs-solo generation A/B
    python -m training.selftest t7     # minutes   t5 traces -> real train steps
    python -m training.selftest t8     # ~15-40m   real serial datagen timing
    python -m training.selftest t9     # ~10-25m   same workload at --parallel 3
    python -m training.selftest t10    # ~20-30m   timed train batches + epoch est.

or everything in order with ``python -m training.selftest all``. Every stage
prints exactly one line ``TEST <id> PASS/FAIL: <evidence>`` (paste those
lines back for review); failures also print the traceback to stderr. Later
stages assume earlier ones passed (t7 consumes t5's output). No pytest, no
new dependencies -- plain asserts.

Stage map (rationale in the Intermission plan):

  * t0-env      imports + versions + CUDA + bitsandbytes
  * t1-pure     parse_rating, the shape/scale reward split
                (rating_advantage / example_scale / build_span_weights),
                oracle_verdict + _oracle_meta geometry, image noise,
                tripwire, _rewrite_image_urls, Game/PlayerAnchor/Analyst
                sources on fabricated dirs (example_weight, oracle
                modifiers, novelty toggle), epoch_batches bucketing,
                stack_equal_length, run_weekend --checkpoint (rejects
                --start-checkpoint / --resume-checkpoint), boundary
                openings / multi-gold / END_GAME parse
  * t2-data     manifest loads, per-source counts vs meta.json, probes exist
  * t3-model    4-bit QLoRA load, terminator, CE/KD forward+backward
                (image example included), teacher-path sanity, kd_anchor
                base-fallback parity, example_weight exact-scaling
                regression, bounded-unlikelihood loss finite/non-negative,
                kd rejects negative weights
  * t4-train    batch-4 vs batch-1 loss parity, CLI smoke train (CE + KD +
                negative span + image), forced-rollback variant (hard tier)
  * t5-datagen  2 games x 5 moves at --parallel 2: traces, images, stats,
                plots, tripwire silent
  * t6-ab       generate_batch vs generate equivalence, greedy
  * t7-e2e      Game + PlayerAnchor + Analyst sources over t5's output
                through real train steps
  * t8-timing   REAL serial datagen run (startup excluded), s/generation
                + serial epoch extrapolation; t9's baseline
  * t9-parallel t8's exact workload at --parallel 3, compared on
                seconds_per_generation (speedup reported, slowdown asserted)
  * t10-traintime  4 timed micro-batches from every loss category on the
                real overnight corpus (T10_DATAGEN_LABEL to change), peak
                VRAM per category (tripwire vs the 2026-07-31 OOM), and a
                whole-epoch train-time estimate; saves NO checkpoint
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent


# ================================================================ fixtures

def _tiny_png(path: Path, seed: int = 0, size: int = 96) -> Path:
    """A small deterministic board-like image (colored cells on a grid)."""
    import random

    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    img = Image.new("RGB", (size, size), (235, 235, 235))
    draw = ImageDraw.Draw(img)
    cell = size // 6
    for gx in range(6):
        for gy in range(6):
            if rng.random() < 0.25:
                color = rng.choice(
                    [(200, 40, 40), (40, 160, 40), (220, 180, 30), (60, 60, 200)]
                )
                draw.rectangle(
                    [gx * cell, gy * cell, (gx + 1) * cell - 1,
                     (gy + 1) * cell - 1],
                    fill=color,
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def _text_example(text_in: str, text_out: str, **extra) -> dict:
    return {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": text_in}]},
        ],
        "target_text": text_out,
        **extra,
    }


def _image_example(img_path: Path, text_in: str, text_out: str,
                   **extra) -> dict:
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "url": str(img_path)},
                {"type": "text", "text": text_in},
            ]},
        ],
        "target_text": text_out,
        **extra,
    }


def _write_smoke_jsonl(dir_: Path) -> Path:
    """The t4 smoke set: CE text, CE image, KD, negative-span -- two of each
    so micro-batching has something to bucket."""
    img_a = _tiny_png(dir_ / "img_a.png", seed=1)
    img_b = _tiny_png(dir_ / "img_b.png", seed=2)
    rows = [
        _text_example("What is 2 + 3? Reply with the number.", "5"),
        _text_example("Name the color of a clear daytime sky.",
                      "The sky is blue."),
        _image_example(img_a, "Describe this board in one short sentence.",
                       "A small grid with a few colored cells."),
        _image_example(img_b, "Is anything drawn on this board?",
                       "Yes, several colored cells on a light grid."),
        _text_example("Say the word 'hello'.", "hello", loss="kd"),
        _text_example("Count from 1 to 3.", "1, 2, 3", loss="kd"),
        _text_example(
            "Where is the gold?",
            "The gold is at 3 o'clock. WRONG: it is at 9 o'clock.",
            span_weights=[[0, 27, 0.5], [28, 58, -1.0]],
        ),
        _text_example(
            "Which way is the wall?",
            "The wall is ahead. Actually behind.",
            span_weights=[[0, 18, 0.6], [19, 35, -0.5]],
        ),
    ]
    path = dir_ / "smoke.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _run_cli(argv: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    """Run a repo CLI as a subprocess, capturing output for the evidence."""
    return subprocess.run(
        [sys.executable, "-m", *argv],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
    )


def _newest(pattern: str, base: Path = REPO_ROOT) -> Path:
    matches = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise AssertionError(f"nothing matches {base / pattern}")
    return matches[-1]


def _free_cuda(*objs) -> None:
    import gc

    import torch

    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


# ================================================================== stages

def t0_env() -> str:
    """Imports, versions, CUDA."""
    import platform

    import torch
    import transformers
    from packaging.version import Version

    assert Version(transformers.__version__) >= Version("5.10"), (
        f"transformers {transformers.__version__} < 5.10 (the registry "
        "models need it)"
    )
    assert torch.cuda.is_available(), "CUDA not visible from torch"
    import bitsandbytes  # noqa: F401  (import is the test)
    import peft
    import matplotlib
    from PIL import Image  # noqa: F401

    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30
    return (
        f"python {platform.python_version()}, torch {torch.__version__}, "
        f"transformers {transformers.__version__}, peft {peft.__version__}, "
        f"bitsandbytes {bitsandbytes.__version__}, "
        f"matplotlib {matplotlib.__version__}; {gpu} ({vram:.0f} GiB)"
    )


def t1_pure() -> str:
    """The pure-python units -- no GPU, no NAMS, no downloads."""
    import io
    import math
    import random
    from collections import Counter

    from PIL import Image

    from agent.memory import (
        ANALYST_TAG,
        is_analyst_text,
        strip_analyst_lines,
        tag_analyst_text,
    )
    from agent.modes import parse_rating
    from training.game_traces import (
        ACTION_BALANCE_CAP,
        ADV_CAP,
        ORACLE_MATCH_SPAN,
        ORACLE_WRONG_SCALE,
        ORACLE_WRONG_SPAN,
        TRANSITION_BOOST,
        WRONG_SPAN_WEIGHT,
        AnalystTraceSource,
        GameTraceSource,
        NoveltyTracker,
        PlayerAnchorSource,
        action_balance_multipliers,
        build_span_weights,
        example_scale,
        oracle_verdict,
        rating_advantage,
    )
    from training.generate_game_traces import (
        PERCEPTION_QUESTION_GROUPS,
        _assert_no_analyst_leak,
        _oracle_meta,
        _rewrite_image_urls,
        _sample_perception_question,
    )
    from training.image_noise import make_image_filter, noise_image
    from training.planted_errors import (
        scramble_clock,
        scramble_directions,
        scramble_move_token,
        scramble_player_reply,
    )
    from training.train import (
        DataSource,
        MetricGuard,
        TrainingExample,
        epoch_batches,
    )

    checks = 0

    # ---- parse_rating: plain, bold variants, last-wins, clamp, absent
    for text, expected in [
        ("...verdict.\nRATING: 0.5", 0.5),
        ("**RATING:** -0.25", -0.25),
        ("**RATING**: 1", 1.0),
        ("RATING: 0.1\nrethinking...\nRATING: -0.9", -0.9),
        ("RATING: 7", 1.0),          # clamped
        ("no verdict here", None),
    ]:
        got = parse_rating(text)
        assert got == expected, f"parse_rating({text!r}) = {got}, want {expected}"
        checks += 1

    # ---- rating_advantage / example_scale: the reply-wide SCALE half of
    # the 2026-08-05 shape/scale split (module docstring + postmortem in
    # training/game_traces.py). Exp-advantage vs the corpus mean, hard
    # floor at -0.5, cap at ADV_CAP, ADDITIVE win boost 1.0 * 0.95^d.
    assert abs(rating_advantage(0.6, baseline=0.6) - 1.0) < 1e-9
    assert abs(rating_advantage(0.8, baseline=0.6)
               - math.exp(0.2 / 0.3)) < 1e-9      # ~2x per +0.2 of rating
    assert rating_advantage(-0.5, baseline=0.6) == 0.0   # floor: exactly 0
    assert rating_advantage(-1.0, baseline=0.6) == 0.0   # NEVER negative
    assert rating_advantage(1.0, baseline=-0.4) == ADV_CAP  # capped at 3.0
    # win boost is additive on the scale: rescues even a floored reply
    assert abs(example_scale(0.6, 0.6, game_won=True, moves_from_end=0)
               - 2.0) < 1e-9                      # 1.0 base + 1.0 boost
    assert abs(example_scale(-1.0, 0.6, game_won=True, moves_from_end=8)
               - 0.95 ** 8) < 1e-9                # floored base, boost only
    assert example_scale(-0.5, 0.6, game_won=False,
                         moves_from_end=None) == 0.0     # floor, no rescue
    checks += 8

    # ---- build_span_weights: the within-reply SHAPE half -- base 1.0,
    # verified WRONG spans at WRONG_SPAN_WEIGHT (-0.5 = bounded
    # unlikelihood, train.py NEGATIVE WEIGHTS), the oracle's move-token
    # modifier LAST (ground truth outranks the analyst on overlap).
    tgt = "go left WRONG: gold is right [CLOCK]"
    spans = build_span_weights(tgt, wrong_spans=["WRONG: gold is right"])
    assert spans[0] == (0, len(tgt), 1.0), spans[0]
    assert spans[1][2] == WRONG_SPAN_WEIGHT and \
        tgt[spans[1][0]:spans[1][1]].startswith("WRONG"), spans[1]
    ok = build_span_weights(tgt, wrong_spans=[], action="CLOCK",
                            verdict="correct")
    assert ok[-1] == (tgt.rfind("[CLOCK]"), len(tgt), ORACLE_MATCH_SPAN), ok
    bad = build_span_weights(tgt, wrong_spans=[], action="CLOCK",
                             verdict="wrong")
    assert bad[-1][2] == ORACLE_WRONG_SPAN, bad
    # neutral / unknown verdicts add NO modifier; missing token is skipped
    assert len(build_span_weights(tgt, wrong_spans=[], action="CLOCK",
                                  verdict="neutral")) == 1
    assert len(build_span_weights(tgt, wrong_spans=[], action="FORWARD",
                                  verdict="correct")) == 1
    checks += 6

    # ---- oracle_verdict geometry (crutch block in game_traces.py):
    # compass convention, positive rel_bearing = gold clockwise of facing
    for args, want in [
        (("FORWARD", "FORWARD", 0.0, True), "correct"),   # exact match
        (("CLOCK", "CLOCK", 0.7, False), "correct"),
        (("ANTICLOCK", "CLOCK", 3.05, False), "correct"),  # ~behind: either
        (("FORWARD", "CLOCK", 0.25, False), "neutral"),   # inside 20deg cone
        # missed forward: ANY turn under a ray hit is wrong since the
        # 2026-08-11 tightening (was the "fine-tuning" neutral)
        (("CLOCK", "FORWARD", 0.3, True), "wrong"),
        (("ANTICLOCK", "FORWARD", 0.3, True), "wrong"),   # other direction
        (("FORWARD", "CLOCK", 0.5, False), "wrong"),      # 28.6deg: outside 20deg cone (was neutral at 45deg)
        (("FORWARD", "CLOCK", 1.2, False), "wrong"),      # well outside the cone
        (("CLOCK", "ANTICLOCK", -0.8, False), "wrong"),   # away from gold
        ((None, "FORWARD", 0.0, True), "unknown"),        # perception round
        (("CLOCK", None, None, None), "unknown"),         # pre-oracle corpus
    ]:
        got = oracle_verdict(*args)
        assert got == want, f"oracle_verdict{args} = {got}, want {want}"
        checks += 1

    # ---- _oracle_meta: raw facts from a settings dict (datagen side).
    # Agent center-board facing 12 o'clock (theta=0, facing (sin,cos)).
    base_settings = {"agent_x": 0.5, "agent_y": 0.5, "direction": 0.0,
                     "agent_r": 0.05, "gold_r": 0.03}
    north = _oracle_meta({**base_settings, "gold": [[0.5, 0.8]]})
    assert north["oracle_move"] == "FORWARD" and north["oracle_ray_hit"], north
    assert abs(north["oracle_rel_bearing"]) < 1e-9, north
    east = _oracle_meta({**base_settings, "gold": [[0.8, 0.5]]})
    assert east["oracle_move"] == "CLOCK" and not east["oracle_ray_hit"], east
    # the stamped bearing is round(rel, 4), so compare against the SAME
    # rounding -- pi/2 differs from 1.5708 by ~3.7e-6, far over any "exact"
    # tolerance (this assert failed at 1e-9 before the 2026-08-05 sweep)
    assert east["oracle_rel_bearing"] == round(math.pi / 2, 4), east
    west = _oracle_meta({**base_settings, "gold": [[0.2, 0.5]]})
    assert west["oracle_move"] == "ANTICLOCK", west
    # nearest gold sets the bearing; a second, farther gold on the ray
    # still counts for ray_hit
    two = _oracle_meta({**base_settings, "gold": [[0.5, 0.9], [0.52, 0.6]]})
    assert two["oracle_ray_hit"], two          # 0.02 perp < 0.08 reach
    assert two["oracle_rel_bearing"] == round(
        math.atan2(0.02, 0.1), 4
    ), two  # nearest = (0.52, 0.6); stamped bearing is round(rel, 4)
    assert _oracle_meta({**base_settings, "gold": []}) == {}  # game won
    checks += 8

    # ---- action balance (TEMPORARY HACK -- the screaming block above
    # action_balance_multipliers in training/game_traces.py): inverse
    # frequency mean/count, mass-preserving, capped at 4x either way.
    assert action_balance_multipliers({}) == {}
    eq = action_balance_multipliers({"A": 7, "B": 7, "C": 7})
    assert all(abs(m - 1.0) < 1e-9 for m in eq.values()), eq
    # the 2026-08-04 retest's real iter2 mix
    bal = action_balance_multipliers(
        {"ANTICLOCK": 1194, "CLOCK": 806, "FORWARD": 471}
    )
    mean = (1194 + 806 + 471) / 3
    assert abs(bal["ANTICLOCK"] - mean / 1194) < 1e-9, bal
    assert abs(bal["FORWARD"] - mean / 471) < 1e-9, bal
    assert bal["ANTICLOCK"] < 1.0 < bal["FORWARD"], bal
    # mass preservation: sum(count * mult) == total count (no cap engaged)
    assert abs(sum(n * bal[a] for a, n in
                   {"ANTICLOCK": 1194, "CLOCK": 806, "FORWARD": 471}.items())
               - 2471) < 1e-6, bal
    # degenerate mixes -> cap engages (both directions)
    capped = action_balance_multipliers({"A": 1000, "B": 5})
    assert capped["B"] == ACTION_BALANCE_CAP, capped     # not ~100x
    assert abs(capped["A"] - 502.5 / 1000) < 1e-9, capped  # under mean: no cap
    lo = action_balance_multipliers({"A": 100, "B": 1, "C": 1, "D": 1, "E": 1})
    assert lo["A"] == 1.0 / ACTION_BALANCE_CAP, lo       # dominant: floored
    assert lo["B"] == ACTION_BALANCE_CAP, lo
    checks += 6

    # ---- NoveltyTracker (WORK IN PROGRESS -- "boredom" decay, block above
    # the class in training/game_traces.py): 0.9^k on consecutive identical
    # moves, floor 0.1, reset on a different move, perception (action None)
    # skipped WITHOUT resetting, independent per (session, game) key.
    nt = NoveltyTracker()
    seq = [nt.multiplier("s", 0, "ANTICLOCK") for _ in range(3)]
    assert all(abs(m - e) < 1e-9 for m, e in zip(seq, [1.0, 0.9, 0.81])), seq
    assert nt.multiplier("s", 0, None) == 1.0            # perception: skip...
    assert abs(nt.multiplier("s", 0, "ANTICLOCK") - 0.9 ** 3) < 1e-9, (
        "perception round RESET the streak (must skip, not reset)"
    )
    assert nt.multiplier("s", 0, "FORWARD") == 1.0       # different move
    assert abs(nt.multiplier("s", 0, "ANTICLOCK") - 1.0) < 1e-9  # streak reset
    assert nt.multiplier("s", 1, "ANTICLOCK") == 1.0     # other game: fresh
    for _ in range(40):
        floor = nt.multiplier("s2", 0, "CLOCK")
    assert abs(floor - 0.1) < 1e-9, f"floor not enforced: {floor}"
    checks += 6

    # ---- MetricGuard relative=False (ceiling-only drift meters -- the
    # 2026-08-04 retest fix): the best-ever multiplier and the soft tier
    # are OFF; only the absolute ceiling fires. relative=True unchanged.
    drift = MetricGuard("m", higher_is_better=False, rel_tolerance=0.1,
                        hard_multiplier=2.0, ceiling=1.0, relative=False)
    assert not drift.is_hard_regression(0.5, best=0.08), (
        "ceiling-only guard fired below its ceiling (the exact failure "
        "that rolled back the 2026-08-04 retest)"
    )
    assert drift.is_hard_regression(1.5, best=0.08), "ceiling did not fire"
    assert not drift.is_soft_regression(0.5, best=0.08), "soft tier not off"
    rel = MetricGuard("m", higher_is_better=False, rel_tolerance=0.1,
                      hard_multiplier=2.0)
    assert rel.is_hard_regression(0.5, best=0.08), "relative guard broken"
    assert rel.is_soft_regression(0.1, best=0.08), "soft tier broken"
    # warn_only (2026-08-05 aug4 fix): DETECTION is unchanged -- the flag
    # strips authority in run_hooks_and_guard (demote to drift_warning),
    # not the predicates. Defaults must stay False so no source is
    # silently demoted.
    wo = MetricGuard("m", higher_is_better=False, rel_tolerance=0.1,
                     hard_multiplier=2.0, warn_only=True)
    assert wo.is_hard_regression(0.5, best=0.08), (
        "warn_only must not change detection -- only authority"
    )
    assert not rel.warn_only and not drift.warn_only, "warn_only default"
    assert DataSource.guard_warn_only is False, "DataSource default"
    # ceiling_breached (2026-08-05 sweep fix): the absolute bound is
    # independent of best-tracking, so the trainer can check it BEFORE the
    # improvement short-circuit -- a first eval already past the ceiling,
    # or a value improving on best while still past it, must both fire
    # (both used to sail through run_hooks_and_guard silently).
    assert drift.ceiling_breached(1.5), "first-eval-above-ceiling missed"
    assert drift.ceiling_breached(1.2), (
        "improving (1.5 -> 1.2) but still above ceiling 1.0 -- must fire"
    )
    assert not drift.ceiling_breached(0.9), "below ceiling must not fire"
    assert not rel.ceiling_breached(99.0), "no ceiling declared: never fires"
    hib = MetricGuard("m", higher_is_better=True, rel_tolerance=0.1,
                      ceiling=0.5)
    assert hib.ceiling_breached(0.4) and not hib.ceiling_breached(0.6), (
        "higher-is-better ceiling must act as a floor"
    )
    checks += 13

    # ---- image noise: deterministic per seed, identity at strength 0
    base_img = Image.new("RGB", (64, 64), (120, 40, 40))
    def png_bytes(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    a = png_bytes(noise_image(base_img, random.Random(3), 1.0))
    b = png_bytes(noise_image(base_img, random.Random(3), 1.0))
    assert a == b, "noise_image not deterministic under a fixed seed"
    assert a != png_bytes(base_img), "noise_image(strength=1) changed nothing"
    ident = png_bytes(noise_image(base_img, random.Random(3), 0.0))
    assert ident == png_bytes(base_img.convert("RGB")), "strength 0 not identity"
    # 10% clean pass-through (_SKIP_PROB, 2026-08-04): seed 31's first
    # draw lands inside the gate, so the frame must come back untouched
    # even at full strength (the network must also see uncorrupted boards)
    assert random.Random(31).random() < 0.1, "seed 31 no longer skips"
    skipped = png_bytes(noise_image(base_img, random.Random(31), 1.0))
    assert skipped == png_bytes(base_img.convert("RGB")), (
        "the _SKIP_PROB pass-through did not return a clean frame"
    )
    checks += 4

    with tempfile.TemporaryDirectory(prefix="selftest_t1_") as tmp:
        tmp_dir = Path(tmp)
        # ---- make_image_filter: in-place, successive frames differ
        f1 = _tiny_png(tmp_dir / "f1.png", seed=5)
        f2 = _tiny_png(tmp_dir / "f2.png", seed=5)
        raw = f1.read_bytes()
        filt = make_image_filter(seed=11)
        filt(str(f1))
        filt(str(f2))
        assert f1.read_bytes() != raw, "image_filter did not modify the file"
        assert f1.read_bytes() != f2.read_bytes(), (
            "identical inputs noised identically along one stream"
        )
        checks += 2

        # ---- _rewrite_image_urls
        msgs = [
            {"role": "user", "content": [
                {"type": "image", "url": "/live/frame.png"},
                {"type": "text", "text": "hi"},
            ]},
        ]
        n = _rewrite_image_urls(msgs, {"/live/frame.png": "images/g0.png"})
        assert n == 1 and msgs[0]["content"][0]["url"] == "images/g0.png"
        checks += 1

        # ---- tripwire: fires on a leak (incl. multi-line), silent when clean
        analysis = "The move was poor.\nThe agent ignored the wall at 9."
        leaky = {
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "context: " + analysis + " more"},
            ]}],
            "target_text": "[FORWARD]",
        }
        clean = {
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "an innocent player context"},
            ]}],
            "target_text": "[FORWARD]",
        }
        fired = False
        try:
            _assert_no_analyst_leak(leaky, [analysis])
        except RuntimeError:
            fired = True
        assert fired, "tripwire did not fire on an embedded analysis"
        _assert_no_analyst_leak(clean, [analysis])  # must not raise
        checks += 2

        # ---- per-line [ANALYST] tag: write every line, drop any tagged line
        raw = "The move was poor.\nThe agent ignored the wall at 9."
        tagged = tag_analyst_text(raw)
        lines = tagged.splitlines()
        assert len(lines) == 2
        assert all(ln.startswith(ANALYST_TAG + " ") for ln in lines)
        assert tag_analyst_text(tagged) == tagged  # idempotent
        assert is_analyst_text(f"**assistant**: {lines[0]}")
        assert is_analyst_text("(analyst) leftover first-line prefix")
        assert not is_analyst_text("player said [FORWARD]")
        nams_shaped = (
            f"**assistant**: {ANALYST_TAG} (analyst) The player's response\n"
            f"{ANALYST_TAG} - **Agent Position**: (5, 5)\n"
            "player said [FORWARD]\n"
        )
        stripped = strip_analyst_lines(nams_shaped)
        assert ANALYST_TAG not in stripped
        assert "Agent Position" not in stripped
        assert "player said [FORWARD]" in stripped
        leftover = (
            "**assistant**: (analyst) first line of an old untagged blob\n"
            "player said [FORWARD]\n"
        )
        leftover_stripped = strip_analyst_lines(leftover)
        assert "first line" not in leftover_stripped
        assert "player said [FORWARD]" in leftover_stripped
        checks += 1

        # tripwire fires on a tagged fragment even when analysis[:100] is absent
        tagged_fragment = {
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": (
                    f"context: {ANALYST_TAG} - **Agent Position**: (5, 5)"
                )},
            ]}],
            "target_text": "[FORWARD]",
        }
        fired = False
        try:
            _assert_no_analyst_leak(tagged_fragment, [analysis])
        except RuntimeError:
            fired = True
        assert fired, "tripwire did not fire on a tagged fragment without analysis[:100]"
        _assert_no_analyst_leak(clean, [analysis])  # still silent on clean text
        checks += 1

        # ---- GameTraceSource on a fabricated trace dir
        trace_dir = tmp_dir / "traces_fab"
        img = _tiny_png(trace_dir / "images" / "g0.png", seed=7)
        records = [
            {
                "messages": [{"role": "user", "content": [
                    {"type": "image", "url": "images/g0.png"},
                    {"type": "text", "text": "move?"},
                ]}],
                "target_text": "thinking WRONG bit [FORWARD]",
                "meta": {"rating": 0.5, "wrong_spans": ["WRONG bit"],
                         "game_won": True, "moves_from_end": 0},
            },
            {   # analyst forgot the rating -> must be dropped
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "move?"},
                ]}],
                "target_text": "[LEFT]",
                "meta": {"rating": None, "wrong_spans": []},
            },
        ]
        with open(trace_dir / "traces.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        src = GameTraceSource(trace_dir / "traces.jsonl", noise_strength=0.0)
        exs = list(src.examples())
        assert len(exs) == 1, f"expected 1 example (1 dropped), got {len(exs)}"
        ex = exs[0]
        assert ex.loss == "ce" and ex.span_weights, ex
        assert ex.messages[0]["content"][0]["url"] == str(img), (
            "image url not resolved against the trace dir"
        )
        # SHAPE: base 1.0, WRONG span at -0.5 (unlikelihood) -- the
        # reply-wide reward must NOT appear here (it would cancel under
        # per-example normalization, train.py SHAPE VS SCALE)
        assert ex.span_weights[0][2] == 1.0, ex.span_weights
        assert ex.span_weights[1][2] == WRONG_SPAN_WEIGHT, ex.span_weights
        # SCALE: single rated record, so r_bar == its own rating 0.5 ->
        # exp(0) = 1.0 base, + 1.0 win boost at moves_from_end=0
        assert abs(src.rating_baseline - 0.5) < 1e-9, src.rating_baseline
        assert abs(ex.example_weight - 2.0) < 1e-9, ex.example_weight
        checks += 6

        # ---- novelty decay through GameTraceSource (WIP, OFF by default
        # -- novelty=True to enable): three identical consecutive moves in
        # one game -> example_weight scaled by 1.0 / 0.9 / 0.81; the
        # parallel game is unaffected.
        nov_dir = tmp_dir / "traces_novelty"
        nov_dir.mkdir()
        nov_records = []
        for i in range(3):
            nov_records.append({
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "move?"},
                ]}],
                "target_text": "turning again [ANTICLOCK]",
                "meta": {"rating": 1.0, "wrong_spans": [], "game_won": False,
                         "moves_from_end": None, "session_id": "sA",
                         "game_index": 0, "move_index": i,
                         "action": "ANTICLOCK"},
            })
        nov_records.append({   # same action, DIFFERENT game: fresh streak
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "move?"},
            ]}],
            "target_text": "turning again [ANTICLOCK]",
            "meta": {"rating": 1.0, "wrong_spans": [], "game_won": False,
                     "moves_from_end": None, "session_id": "sA",
                     "game_index": 1, "move_index": 0,
                     "action": "ANTICLOCK"},
        })
        with open(nov_dir / "traces.jsonl", "w", encoding="utf-8") as f:
            for r in nov_records:
                f.write(json.dumps(r) + "\n")
        nsrc = GameTraceSource(nov_dir / "traces.jsonl", novelty=True,
                               noise_strength=0.0)
        # single-action corpus -> balance multiplier exactly 1.0 and
        # r_bar == 1.0 (exp(0) base 1.0), so this measures the decay alone
        assert nsrc.action_balance == {"ANTICLOCK": 1.0}, nsrc.action_balance
        nexs = list(nsrc.examples())
        assert len(nexs) == 4, nexs
        nscales = [x.example_weight for x in nexs]
        assert all(abs(w - e) < 1e-9
                   for w, e in zip(nscales, [1.0, 0.9, 0.81, 1.0])), (
            f"novelty decay wrong through GameTraceSource: {nscales}"
        )
        # span weights stay pure shape (uniform 1.0) -- the decay must
        # ride example_weight or it cancels in the loss
        assert all(x.span_weights[0][2] == 1.0 for x in nexs), nexs
        # OFF BY DEFAULT: the same corpus without the toggle decays nothing
        noff = [x.example_weight
                for x in GameTraceSource(nov_dir / "traces.jsonl",
                                         noise_strength=0.0).examples()]
        assert all(abs(w - 1.0) < 1e-9 for w in noff), (
            f"novelty leaked into a default (novelty=False) source: {noff}"
        )
        checks += 5

        # ---- action balance x novelty through GameTraceSource: mixed
        # actions (3 ANTICLOCK, 1 FORWARD -> mean 2 -> x2/3 and x2), last
        # ANTICLOCK also consecutive-repeated (novelty 0.9); multipliers
        # compose.
        bal_dir = tmp_dir / "traces_balance"
        bal_dir.mkdir()
        bal_actions = ["ANTICLOCK", "FORWARD", "ANTICLOCK", "ANTICLOCK"]
        with open(bal_dir / "traces.jsonl", "w", encoding="utf-8") as f:
            for i, action in enumerate(bal_actions):
                f.write(json.dumps({
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": "move?"},
                    ]}],
                    "target_text": f"[{action}]",
                    "meta": {"rating": 1.0, "wrong_spans": [],
                             "game_won": False, "moves_from_end": None,
                             "session_id": "sB", "game_index": 0,
                             "move_index": i, "action": action},
                }) + "\n")
        bsrc = GameTraceSource(bal_dir / "traces.jsonl", novelty=True,
                               noise_strength=0.0)
        assert (abs(bsrc.action_balance["ANTICLOCK"] - 2 / 3) < 1e-9
                and abs(bsrc.action_balance["FORWARD"] - 2.0) < 1e-9), (
            bsrc.action_balance
        )
        bscales = [x.example_weight for x in bsrc.examples()]
        # exp-advantage base 1.0 (uniform ratings) x balance, last record
        # additionally x0.9 novelty -- all on example_weight
        expect = [2 / 3, 2.0, 2 / 3, 0.9 * 2 / 3]
        assert all(abs(w - e) < 1e-9 for w, e in zip(bscales, expect)), (
            f"balance x novelty composition wrong: {bscales} != {expect}"
        )
        checks += 2

        # ---- oracle through GameTraceSource: stamped meta -> move-token
        # span modifier + the x0.25 example-scale penalty on "wrong" +
        # the x2 TRANSITION_BOOST on ray-hit rounds (crutch block in
        # game_traces.py); floored ratings are SKIPPED.
        orc_dir = tmp_dir / "traces_oracle"
        orc_dir.mkdir()
        orc_records = [
            # matches the oracle -> [FORWARD] span at ORACLE_MATCH_SPAN;
            # ray hit -> example_weight x TRANSITION_BOOST
            {"action": "FORWARD", "oracle_move": "FORWARD",
             "oracle_rel_bearing": 0.05, "oracle_ray_hit": True,
             "rating": 0.0},
            # contradicts it (turn away) -> [CLOCK] span at
            # ORACLE_WRONG_SPAN and example_weight x ORACLE_WRONG_SCALE
            {"action": "CLOCK", "oracle_move": "ANTICLOCK",
             "oracle_rel_bearing": -0.9, "oracle_ray_hit": False,
             "rating": 0.0},
            # missed forward (turn under ray hit): "wrong" since the
            # 2026-08-11 tightening -> ORACLE_WRONG_SPAN on [CLOCK] and
            # example_weight x ORACLE_WRONG_SCALE x TRANSITION_BOOST
            {"action": "CLOCK", "oracle_move": "FORWARD",
             "oracle_rel_bearing": 0.05, "oracle_ray_hit": True,
             "rating": 0.0},
            # floored rating, no win -> zero scale -> record SKIPPED
            {"action": "FORWARD", "oracle_move": "FORWARD",
             "oracle_rel_bearing": 0.0, "oracle_ray_hit": True,
             "rating": -1.0},
        ]
        with open(orc_dir / "traces.jsonl", "w", encoding="utf-8") as f:
            for i, m in enumerate(orc_records):
                f.write(json.dumps({
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": "move?"},
                    ]}],
                    "target_text": f"reasoning [{m['action']}]",
                    "meta": {**m, "wrong_spans": [], "game_won": False,
                             "moves_from_end": None, "session_id": "sC",
                             "game_index": 0, "move_index": i},
                }) + "\n")
        osrc = GameTraceSource(orc_dir / "traces.jsonl", noise_strength=0.0)
        oexs = list(osrc.examples())
        assert len(oexs) == 3, f"floored record not skipped: {len(oexs)}"
        # r_bar = (0 + 0 + 0 - 1)/4 (the floored record still counts in
        # the pre-pass -- it happened); survivors share rating 0.0, so
        # with the balance factored out the scales isolate the oracle
        # penalty and the transition boost
        adv = rating_advantage(0.0, osrc.rating_baseline)
        bal = osrc.action_balance
        assert abs(oexs[0].example_weight - adv * bal["FORWARD"]
                   * TRANSITION_BOOST) < 1e-9, (
            oexs[0].example_weight, adv, bal,
        )
        assert abs(oexs[1].example_weight - adv * bal["CLOCK"]
                   * ORACLE_WRONG_SCALE) < 1e-9, (
            oexs[1].example_weight, adv, bal,
        )
        assert abs(oexs[2].example_weight - adv * bal["CLOCK"]
                   * ORACLE_WRONG_SCALE * TRANSITION_BOOST) < 1e-9, (
            oexs[2].example_weight, adv, bal,
        )
        assert oexs[0].span_weights[-1][2] == ORACLE_MATCH_SPAN, oexs[0]
        assert oexs[1].span_weights[-1][2] == ORACLE_WRONG_SPAN, oexs[1]
        assert oexs[2].span_weights[-1][2] == ORACLE_WRONG_SPAN, oexs[2]
        checks += 7

        # ---- PlayerAnchorSource on the same fabricated dir: trust-region
        # semantics -- EVERY parseable record kept (rating-null included),
        # uniform weights, kd_anchor loss.
        psrc = PlayerAnchorSource(trace_dir / "traces.jsonl",
                                  noise_strength=0.0)
        pexs = list(psrc.examples())
        assert len(pexs) == 2, (
            f"expected 2 anchor examples (rating-null KEPT), got {len(pexs)}"
        )
        assert all(x.loss == "kd_anchor" and x.span_weights is None
                   for x in pexs), pexs
        assert abs(psrc.weight - 0.25) < 1e-9, psrc.weight
        # Ceiling-only drift guards (2026-08-04 retest fix): both anchors
        # opt out of the best-ever multiplier and declare absolute bounds.
        assert psrc.guard_relative is False and psrc.guard_ceiling == 1.0, (
            psrc.guard_relative, psrc.guard_ceiling,
        )
        checks += 4

        # ---- AnalystTraceSource: KD anchor semantics on a fabricated file
        analyst_fixtures = [
            {   # usable
                "messages": [{"role": "user", "content": [
                    {"type": "image", "url": "images/g0.png"},
                    {"type": "text", "text": "review this"},
                ]}],
                "target_text": "Solid move. RATING: 0.5",
                "meta": {"rating": 0.5, "wrong_spans": [],
                         "unverified_spans": []},
            },
            {   # rating missing -> KEPT (a KD anchor needs no reward)
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "review this"},
                ]}],
                "target_text": "Forgot the verdict line.",
                "meta": {"rating": None, "wrong_spans": [],
                         "unverified_spans": []},
            },
            {   # hallucinated WRONG quote -> DROPPED by default
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "review this"},
                ]}],
                "target_text": "WRONG: never said this. RATING: -0.5",
                "meta": {"rating": -0.5, "wrong_spans": [],
                         "unverified_spans": ["never said this"]},
            },
        ]
        with open(trace_dir / "analyst_traces.jsonl", "w",
                  encoding="utf-8") as f:
            for r in analyst_fixtures:
                f.write(json.dumps(r) + "\n")
        asrc = AnalystTraceSource(trace_dir / "analyst_traces.jsonl",
                                  examples_per_epoch=100, noise_strength=0.0)
        aexs = list(asrc.examples())
        assert len(aexs) == 2, (
            f"expected 2 analyst examples (1 dropped), got {len(aexs)}"
        )
        assert all(x.loss == "kd" and x.span_weights is None for x in aexs)
        assert aexs[0].messages[0]["content"][0]["url"] == str(img), (
            "analyst image url not resolved against the trace dir"
        )
        assert abs(asrc.weight - 100 / 2) < 1e-9, (
            f"quota weight wrong: {asrc.weight}"
        )
        assert asrc.guard_relative is False and asrc.guard_ceiling == 5.0, (
            asrc.guard_relative, asrc.guard_ceiling,
        )
        checks += 2

    # ---- epoch_batches: bucketing separates loss kinds / image counts,
    #      batch sizes bounded, nothing lost
    exs = (
        [TrainingExample([{"role": "user", "content": [
            {"type": "text", "text": "q" * 50}]}], "a", loss="ce")] * 5
        + [TrainingExample([{"role": "user", "content": [
            {"type": "text", "text": "q" * 50}]}], "a", loss="kd")] * 3
        + [TrainingExample([{"role": "user", "content": [
            {"type": "image", "url": "x.png"},
            {"type": "text", "text": "q" * 50}]}], "a", loss="ce")] * 2
    )
    batches = epoch_batches(exs, micro_batch=4, rng=random.Random(0))
    assert sum(len(b) for b in batches) == len(exs), "examples lost in batching"
    for b in batches:
        assert len(b) <= 4
        assert len({x.loss for x in b}) == 1, "mixed loss kinds in a batch"
        assert len({x.declares_image() for x in b}) == 1, (
            "image and text examples share a batch"
        )
    checks += 1

    # ---- epoch_batches: batch_cap (long-target KD sources, e.g.
    #      openthoughts) bounds THEIR batches without touching others
    capped = [TrainingExample([{"role": "user", "content": [
        {"type": "text", "text": "q" * 50}]}], "a", loss="kd", batch_cap=1)
        for _ in range(3)]
    batches = epoch_batches(exs + capped, micro_batch=4,
                            rng=random.Random(0))
    assert sum(len(b) for b in batches) == len(exs) + 3, "capped ex lost"
    for b in batches:
        if b[0].batch_cap is not None:
            assert len(b) <= b[0].batch_cap, "batch_cap not enforced"
            assert all(x.batch_cap == b[0].batch_cap for x in b)
        else:
            assert len(b) <= 4
    checks += 1

    # ---- planted-error scrambler: deterministic, one labeled change,
    #      span points at the new text
    reply = (
        "OBS: my eye points toward 12 o'clock; the gold is at the top "
        "right, toward 1 o'clock of me.\n"
        "REASON: the gold is slightly to the right of my facing.\n[CLOCK]"
    )
    r1 = scramble_player_reply(reply, random.Random(5))
    assert r1 == scramble_player_reply(reply, random.Random(5)), (
        "scramble_player_reply not deterministic under a fixed seed"
    )
    text1, changes1 = r1
    assert len(changes1) == 1 and text1 != reply, (text1, changes1)
    ch = changes1[0]
    assert {"kind", "span", "old", "new", "within_tolerance"} <= set(ch), ch
    assert text1[ch["span"][0]:ch["span"][1]] == ch["new"], ch
    checks += 1

    # ---- move token: all three corruption modes reachable, each well-formed
    kinds_seen = set()
    for s in range(40):
        t, chs = scramble_move_token(reply, random.Random(s))
        (c,) = chs
        kinds_seen.add(c["kind"])
        assert c["old"] == "[CLOCK]", c
        if c["kind"] == "move_token_replace":
            assert c["new"] in ("[ANTICLOCK]", "[FORWARD]"), c
        elif c["kind"] == "move_token_strip":
            assert c["new"] == "CLOCK" and t.endswith("CLOCK") and "[CLOCK]" not in t, (t, c)
        else:
            assert c["kind"] == "move_token_delete" and c["new"] == "" and "[CLOCK]" not in t, (t, c)
    assert kinds_seen == {
        "move_token_replace", "move_token_strip", "move_token_delete"
    }, kinds_seen
    checks += 1

    # ---- clock: hour always changes; tolerance label = circular shift <= 2
    for s in range(40):
        t, chs = scramble_clock("the gold sits toward 4 o'clock of me",
                                random.Random(s))
        (c,) = chs
        new_hour = int(c["new"].split()[0])
        assert 1 <= new_hour <= 12 and new_hour != 4, c
        shift = min((new_hour - 4) % 12, (4 - new_hour) % 12)
        assert c["shift_hours"] == shift, c
        assert c["within_tolerance"] == (shift <= 2), c
    checks += 1

    # ---- directions: case-preserving swap; the "right = correct" guard;
    #      unchanged-text fallthrough
    t, chs = scramble_directions("Left of me sits the wall.", random.Random(0))
    assert t == "Right of me sits the wall." and chs[0]["new"] == "Right", (t, chs)
    guarded = "That was the right move."
    assert scramble_directions(guarded, random.Random(0)) == (guarded, []), (
        "semantic 'right' was swapped"
    )
    plain = "nothing scrambleable here."
    assert scramble_player_reply(plain, random.Random(0)) == (plain, []), (
        "scramble fabricated a change on inert text"
    )
    checks += 1

    # ---- perception questions: rate honored; groups equiprobable; mirrored
    #      variants within a group near-equal (the direction-balance promise)
    qrng = random.Random(1)
    n_rate = 20_000
    n_q = sum(
        _sample_perception_question(qrng, 0.2) is not None
        for _ in range(n_rate)
    )
    assert abs(n_q / n_rate - 0.2) < 0.02, f"question rate off: {n_q / n_rate}"
    counts: Counter = Counter(
        _sample_perception_question(qrng, 1.0) for _ in range(60_000)
    )
    assert None not in counts, "rate-1.0 draw returned the default question"
    totals = []
    for group in PERCEPTION_QUESTION_GROUPS:
        assert all(isinstance(q, str) and q.strip() for q in group), group
        got = [counts[q] for q in group]
        assert min(got) > 0 and max(got) < 1.15 * min(got), (
            f"unbalanced variants: {dict(zip(group, got))}"
        )
        totals.append(sum(got))
    assert max(totals) < 1.1 * min(totals), f"unbalanced groups: {totals}"
    checks += 1

    # ---- stack_equal_length: generate_batch's equal-length-cohort
    #      collation (mixed lengths must be a hard error, not a silent pad:
    #      only the parity-checked path in _plan_padded_batch may left-pad
    #      on Gemma 4 -- see the workaround banner in agent/model.py)
    import torch

    from agent.model import stack_equal_length

    a = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "pixel_values": torch.zeros(1, 2, 2),
    }
    b = {
        "input_ids": torch.tensor([[4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "pixel_values": torch.ones(1, 2, 2),
    }
    stacked = stack_equal_length([a, b])
    assert stacked["input_ids"].tolist() == [[1, 2, 3], [4, 5, 6]]
    assert stacked["attention_mask"].shape == (2, 3)
    assert stacked["pixel_values"].shape == (2, 2, 2)
    assert torch.equal(stacked["pixel_values"][1], b["pixel_values"][0])
    longer = {
        "input_ids": torch.tensor([[7, 8, 9, 10]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        "pixel_values": torch.ones(1, 2, 2),
    }
    try:
        stack_equal_length([a, longer])
        raise AssertionError("stack_equal_length accepted mixed lengths")
    except ValueError:
        pass
    checks += 1

    # ---- run_weekend checkpoint flag: one name, the old two rejected.
    # --resume-checkpoint used to be a child-only dest the parent ignored
    # (2026-08-16 started from bare HF). --start-checkpoint was the parent
    # dest. Both collapsed to --checkpoint.
    from contextlib import redirect_stderr
    from training.run_weekend import build_parser as weekend_parser

    wp = weekend_parser()
    ns = wp.parse_args(["--checkpoint", "aug13_iter2_step212"])
    assert ns.checkpoint == "aug13_iter2_step212"
    ns = wp.parse_args([])
    assert ns.checkpoint is None
    ns = wp.parse_args(["--train-iter", "2", "--checkpoint", "parent"])
    assert ns.train_iter == 2 and ns.checkpoint == "parent"
    for argv in (
        ["--start-checkpoint", "aug13_iter2_step212"],
        ["--resume-checkpoint", "aug13_iter2_step212"],
        # the 2026-08-16 form: parent argv with the train.py name
        ["--prefix", "aug16", "--resume-checkpoint", "aug13_iter3_step177"],
        # child mode must not re-open the old dest either
        ["--train-iter", "1", "--resume-checkpoint", "aug13_iter2_step212"],
    ):
        err = io.StringIO()
        try:
            with redirect_stderr(err):
                wp.parse_args(argv)
            raise AssertionError(f"{argv} should have been rejected")
        except SystemExit as exc:
            assert exc.code == 2, (argv, exc.code)
            msg = err.getvalue()
            assert "--checkpoint" in msg, msg
            if "--resume-checkpoint" in argv:
                assert "training.train" in msg, msg
            else:
                assert "renamed" in msg, msg
    checks += 1

    # ---- boundary_openings + new_multi_gold_game (2026-08-12 multi-gold mode)
    from agent.game_io import (
        _SIDE_WALL_WIDTH,
        board_update_line,
        boundary_openings,
        new_multi_gold_game,
        parse_remember_notes,
        truncate_at_first_move_token,
    )
    from agent.memory import format_notepad
    # Notebook human-takeover: first bracketed move token ends the reply.
    assert (
        truncate_at_first_move_token("aim then [FORWARD]\njunk after")
        == "aim then [FORWARD]"
    )
    checks += 1
    assert truncate_at_first_move_token("no token here") == "no token here"
    checks += 1
    # Session scratchpad: [REMEMBER key: value] parser, last-one-wins
    # ordering, truncation interplay, board-update line, notepad render.
    assert parse_remember_notes(
        "[REMEMBER target: left gold] then [REMEMBER plan: hug the wall]"
    ) == [("target", "left gold"), ("plan", "hug the wall")]
    checks += 1
    assert parse_remember_notes(
        "[REMEMBER target: a] mid [REMEMBER target: b]"
    ) == [("target", "a"), ("target", "b")]
    checks += 1
    assert parse_remember_notes("[REMEMBER Target: x]") == [("target", "x")]
    checks += 1
    assert parse_remember_notes(
        "[REMEMBER no-colon] [REMEMBER bad key: x] [REMEMBER unclosed: y "
        "[REMEMBER nl: a\nb]"
    ) == []
    checks += 1
    assert parse_remember_notes("[REMEMBER k: keep]drop]") == [("k", "keep")]
    checks += 1
    assert parse_remember_notes("[REMEMBER k:   padded  ]") == [("k", "padded")]
    checks += 1
    assert parse_remember_notes(
        truncate_at_first_move_token(
            "[REMEMBER a: b]\n[FORWARD]\n[REMEMBER c: d]"
        )
    ) == [("a", "b")]
    checks += 1
    eaten = board_update_line("FORWARD", 1, 2)
    assert "[FORWARD]" in eaten and "2 gold(s)" in eaten
    assert "[REMEMBER target:" in eaten
    checks += 1
    uneaten = board_update_line("CLOCK", 0, 3)
    assert "[CLOCK]" in uneaten and "3 gold(s)" in uneaten
    assert "ATE" not in uneaten
    checks += 1
    empty_pad = format_notepad([])
    assert "Your notepad is empty" in empty_pad
    assert "[REMEMBER key: short note]" in empty_pad
    checks += 1
    filled_pad = format_notepad([
        {"key": "target", "value": "the left gold, near the top wall",
         "updated_round": 4},
    ])
    assert "target: the left gold, near the top wall" in filled_pad
    assert "(updated round 4)" in filled_pad
    checks += 1
    w = _SIDE_WALL_WIDTH
    full = [
        [0.0, 0.0, w, 1.0, 0.0],
        [1.0 - w, 0.0, w, 1.0, 0.0],
        [0.0, 0.0, 1.0, w, 0.0],
        [0.0, 1.0 - w, 1.0, w, 0.0],
    ]
    assert boundary_openings({"walls": full}) == [], boundary_openings({"walls": full})
    checks += 1

    gapped = [
        [0.0, 0.0, w, 1.0, 0.0],
        [0.0, 0.0, 1.0, w, 0.0],
        [0.0, 1.0 - w, 1.0, w, 0.0],
        [1.0 - w, 0.0, w, 0.4, 0.0],
        [1.0 - w, 0.6, w, 0.4, 0.0],
    ]
    ops = boundary_openings({"walls": gapped})
    assert len(ops) == 1, ops
    op = ops[0]
    assert op["side"] == "right", op
    assert abs(op["width"] - 0.2) < 1e-9, op
    assert abs(op["center"][0] - 1.0) < 1e-9, op
    assert abs(op["center"][1] - 0.5) < 1e-9, op
    checks += 1

    none = boundary_openings({"walls": []})
    assert len(none) == 4, none
    assert {o["side"] for o in none} == {"left", "right", "top", "bottom"}
    assert all(abs(o["width"] - 1.0) < 1e-9 for o in none), none
    checks += 1

    rotated = [[0.0, 0.0, w, 1.0, 0.3]]
    try:
        boundary_openings({"walls": rotated})
        raise AssertionError("rotated boundary wall did not raise")
    except ValueError as exc:
        assert "nonzero angle" in str(exc), exc
    checks += 1

    random.seed(20260812)
    g0 = new_multi_gold_game(n_gold=0, opening="forbid")
    assert len(g0.settings.gold) == 0, g0.settings.gold
    assert boundary_openings(game_io_settings := {
        "walls": [list(x) for x in g0.settings.walls]
    }) == [], game_io_settings
    checks += 1
    g2 = new_multi_gold_game(n_gold=2, opening="require")
    assert len(g2.settings.gold) == 2, g2.settings.gold
    assert boundary_openings({"walls": [list(x) for x in g2.settings.walls]})
    checks += 1
    g3 = new_multi_gold_game(n_gold=3, opening="require")
    assert len(g3.settings.gold) == 3, g3.settings.gold
    checks += 1

    from agent.multi_gold_session import MultiGoldSelfEvalSession
    from agent.self_eval_session import InteractiveSelfEvalSession
    stub = object.__new__(MultiGoldSelfEvalSession)
    assert MultiGoldSelfEvalSession._parse_player_action(
        stub, "TARGET: none\n[END_GAME]", "answer"
    ) == "END_GAME"
    assert MultiGoldSelfEvalSession._parse_player_action(
        stub, "aiming [CLOCK]", "move"
    ) == "CLOCK"
    # base class never sees END_GAME as a move
    base_stub = object.__new__(InteractiveSelfEvalSession)
    assert InteractiveSelfEvalSession._parse_player_action(
        base_stub, "TARGET: none\n[END_GAME]", "answer"
    ) is None
    checks += 1

    return (
        f"{checks} unit groups passed (ratings, rewards, noise, tripwire, "
        "source, batching, scrambler, questions, stack-eq, weekend-ckpt, "
        "openings, multi-gold, end-game parse)"
    )


def t2_data() -> str:
    """Materialized external data vs the manifest -- run AFTER
    ``python -m training.download_external`` (setup_env.sh does it)."""
    from training.external_data import load_manifest, sources_from_manifest

    entries = [e for e in load_manifest() if e.enabled]
    assert entries, "manifest has no enabled entries"
    parts = []
    for entry in entries:
        meta_path = entry.data_dir / "meta.json"
        assert meta_path.is_file(), (
            f"{entry.name}: not materialized -- run python -m "
            "training.download_external"
        )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        data_path = entry.data_dir / "data.jsonl"
        n_lines = sum(
            1 for line in open(data_path, encoding="utf-8") if line.strip()
        )
        assert n_lines == int(meta["examples"]), (
            f"{entry.name}: data.jsonl has {n_lines} rows but meta.json "
            f"says {meta['examples']}"
        )
        if entry.probe:
            assert entry.probe_path.is_file(), (
                f"{entry.name}: probe file missing: {entry.probe_path}"
            )
        parts.append(f"{entry.name}={n_lines}")
    sources = sources_from_manifest()
    assert all(s.weight > 0 for s in sources)
    return f"{len(entries)} source(s) verified: " + ", ".join(parts)


def t3_model() -> str:
    """4-bit QLoRA load + one collated forward/backward per loss kind
    (image example included) + teacher-path sanity. GPU required."""
    import torch

    from agent.config import CONFIG
    from agent.model import ADAPTERS, spec_for
    from training.train import (
        Collator,
        JsonlSource,
        TrainConfig,
        build_model,
        resolve_terminator_id,
        weighted_loss,
    )

    cfg = TrainConfig(label="selftest_t3")
    spec = spec_for(cfg.architecture or CONFIG.model_key)
    model, processor, lora_targets, projector = build_model(
        spec, cfg, CONFIG.hf_token
    )
    try:
        tokenizer = getattr(processor, "tokenizer", processor)
        terminator = resolve_terminator_id(model, tokenizer)
        collator = Collator(processor, ADAPTERS[spec.family], terminator,
                            compute_dtype=torch.bfloat16, device=cfg.device)

        # Keep the temp dir alive for the WHOLE stage: image examples hold
        # absolute paths into it, and the processor hard-fails (misleading
        # "Incorrect padding" / base64 error) if the file is already gone.
        with tempfile.TemporaryDirectory(prefix="selftest_t3_") as tmp:
            smoke = _write_smoke_jsonl(Path(tmp))
            exs = list(JsonlSource(smoke).examples())
            by_kind = {
                "ce_text": next(e for e in exs
                                if e.loss == "ce" and not e.declares_image()
                                and not e.span_weights),
                "ce_image": next(e for e in exs if e.declares_image()),
                "kd": next(e for e in exs if e.loss == "kd"),
                "neg_span": next(e for e in exs if e.span_weights),
            }
            losses = {}
            model.train()
            for kind, ex in by_kind.items():
                built = collator.build(ex)
                if kind == "ce_image":
                    assert any(
                        v.dtype.is_floating_point
                        for v in built["model_inputs"].values()
                        if isinstance(v, torch.Tensor)
                    ), "image example collated without pixel values"
                loss = weighted_loss(model, built["model_inputs"],
                                     built["weights"], loss_kind=ex.loss)
                loss.backward()
                model.zero_grad(set_to_none=True)
                lv = float(loss.detach())
                assert lv == lv and abs(lv) < 1e4, f"{kind}: bad loss {lv}"
                losses[kind] = lv

            # Teacher-path sanity: with a FRESH (identity) adapter, student ==
            # teacher, so the KD soft-CE must equal the teacher's own entropy --
            # i.e. the smallest value it can take. A mismatch means
            # disable_adapter() is not returning the base distribution.
            import torch.nn.functional as F

            from training.train import _base_model_logits

            ex = by_kind["kd"]
            built = collator.build(ex)
            kd_loss = float(weighted_loss(
                model, built["model_inputs"], built["weights"], loss_kind="kd"
            ).detach())
            with torch.no_grad():
                t_logits = _base_model_logits(model, built["model_inputs"])
                w = built["weights"][:, 1:].reshape(-1)
                mask = w != 0
                t = t_logits[:, :-1, :].reshape(
                    -1, t_logits.shape[-1]
                )[mask].float()
                entropy = float(
                    (-(F.softmax(t, -1) * F.log_softmax(t, -1)).sum(-1)
                     * w[mask]).sum()
                    / w[mask].abs().sum()
                )
            assert abs(kd_loss - entropy) < 0.05, (
                f"fresh-adapter KD loss {kd_loss:.4f} != teacher entropy "
                f"{entropy:.4f} -- disable_adapter() may not bypass the adapter"
            )

            # kd_anchor fallback sanity: with NO anchor adapter loaded the
            # anchor teacher deliberately falls back to the base (epoch-1
            # semantics, train.py THREE LOSS KINDS), so kd and kd_anchor
            # must agree on the same example (bf16 tolerance only).
            kda_loss = float(weighted_loss(
                model, built["model_inputs"], built["weights"],
                loss_kind="kd_anchor",
            ).detach())
            assert abs(kda_loss - kd_loss) < 0.02, (
                f"kd_anchor without an anchor adapter gave {kda_loss:.4f} "
                f"vs kd {kd_loss:.4f} -- the base fallback is broken"
            )

            # SHAPE VS SCALE regression (2026-08-05, weighted_loss
            # docstring): an example_weight of 0.1 must scale the loss by
            # EXACTLY 0.1. The old code carried reply-wide reward on the
            # span weights, where per-example normalization cancelled it
            # -- every reward knob was a near-no-op. eval() + no_grad like
            # the t4 parity check: the two forwards must differ ONLY in
            # example_weight, never in dropout/NEFTune noise, and neither
            # needs an autograd graph.
            model.eval()
            ex = by_kind["ce_text"]
            built = collator.build(ex)
            with torch.no_grad():
                base_loss = weighted_loss(
                    model, built["model_inputs"], built["weights"],
                    loss_kind="ce", example_weight=built["example_weight"],
                )
                scaled_loss = weighted_loss(
                    model, built["model_inputs"], built["weights"],
                    loss_kind="ce",
                    example_weight=built["example_weight"] * 0.1,
                )
            assert abs(float(scaled_loss) - 0.1 * float(base_loss)) < 1e-4, (
                f"example_weight x0.1 gave {float(scaled_loss):.5f}, want "
                f"{0.1 * float(base_loss):.5f} -- the scale is being "
                "normalized away again"
            )

            # Bounded unlikelihood (weighted_loss NEGATIVE WEIGHTS): the
            # smoke set's negative-span example must produce a FINITE,
            # NON-NEGATIVE loss (negative-weight CE -- the 2026-08-01
            # collapse objective -- goes negative immediately), and KD
            # kinds must refuse negative weights outright.
            neg_built = collator.build(by_kind["neg_span"])
            assert float((neg_built["weights"] < 0).sum()) > 0, (
                "smoke neg_span example lost its negative weights"
            )
            with torch.no_grad():
                neg_loss = float(weighted_loss(
                    model, neg_built["model_inputs"], neg_built["weights"],
                    loss_kind="ce",
                    example_weight=neg_built["example_weight"],
                ))
            assert neg_loss == neg_loss and neg_loss >= 0.0, (
                f"unlikelihood loss {neg_loss} -- negative or NaN means "
                "the bounded -log(1-p) branch regressed to negative CE"
            )
            try:
                weighted_loss(model, neg_built["model_inputs"],
                              neg_built["weights"], loss_kind="kd")
                raise AssertionError(
                    "kd accepted negative span weights -- must ValueError"
                )
            except ValueError:
                pass

            peak = torch.cuda.max_memory_allocated() / 2**30
            return (
                f"{len(lora_targets)} LoRA targets, {len(projector)} projector "
                f"module(s), terminator {terminator}; losses "
                + ", ".join(f"{k}={v:.3f}" for k, v in losses.items())
                + f"; KD==teacher-entropy ok; kd_anchor fallback ok; "
                f"example_weight x0.1 scales exactly; unlikelihood "
                f"{neg_loss:.3f} finite/non-negative; kd rejects negatives; "
                f"peak {peak:.1f} GiB"
            )
    finally:
        _free_cuda(model)


def t4_train() -> str:
    """Micro-batch parity + CLI smoke train + forced rollback and the
    consecutive-rollback early stop."""
    import torch

    from agent.config import CONFIG
    from agent.model import ADAPTERS, spec_for
    from training.train import (
        Collator,
        JsonlSource,
        TrainConfig,
        build_model,
        epoch_batches,
        resolve_terminator_id,
        weighted_loss,
    )

    smoke_dir = REPO_ROOT / "logs" / "selftest_smoke"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    smoke_dir.mkdir(parents=True)
    smoke = _write_smoke_jsonl(smoke_dir)

    # ---- (a) batch-4 vs batch-1 loss parity on the mixed smoke set
    cfg = TrainConfig(label="selftest_t4")
    spec = spec_for(cfg.architecture or CONFIG.model_key)
    model, processor, _, _ = build_model(spec, cfg, CONFIG.hf_token)
    try:
        tokenizer = getattr(processor, "tokenizer", processor)
        terminator = resolve_terminator_id(model, tokenizer)
        collator = Collator(processor, ADAPTERS[spec.family], terminator,
                            compute_dtype=torch.bfloat16, device=cfg.device)
        exs = list(JsonlSource(smoke).examples())
        model.eval()  # dropout off -- parity must be deterministic aside
                      # from bf16/SDPA shape-dependent numerics
        solo: dict[int, float] = {}
        with torch.no_grad():
            for i, ex in enumerate(exs):
                built = collator.build(ex)
                solo[id(ex)] = float(weighted_loss(
                    model, built["model_inputs"], built["weights"],
                    loss_kind=ex.loss,
                ))
            import random as _random
            worst_rel = 0.0
            worst_abs = 0.0
            worst_desc = ""
            for batch in epoch_batches(exs, 4, _random.Random(0)):
                built = collator.build_batch(batch)
                _, per_ex = weighted_loss(
                    model, built["model_inputs"], built["weights"],
                    loss_kind=batch[0].loss, return_per_example=True,
                )
                for ex, lv in zip(batch, per_ex.tolist()):
                    ref = solo[id(ex)]
                    abs_d = abs(lv - ref)
                    rel = abs_d / max(abs(ref), 1e-6)
                    if rel > worst_rel:
                        worst_rel = rel
                        worst_abs = abs_d
                        worst_desc = (
                            f"loss={ex.loss} image={ex.declares_image()} "
                            f"solo={ref:.4f} batch={lv:.4f}"
                        )
        # bf16 + SDPA: padded vs unpadded shapes are not bit-identical.
        # Gemma 4 multimodal batches are noisier still (vision aux + longer
        # soft-token stretches). A few–ten percent is numerical; a factor
        # of 2+ is a padding/masking/image-routing bug. Accept either a
        # relative OR a small absolute gap (tiny solo losses make relative
        # error explode without a real bug).
        ok = worst_rel < 0.10 or worst_abs < 0.05
        assert ok, (
            f"batch-4 per-example loss deviates {worst_rel:.1%} "
            f"(abs {worst_abs:.4f}) from batch-1; worst: {worst_desc}"
        )
        worst = worst_rel  # for the return string below
    finally:
        _free_cuda(model)

    # ---- (b) CLI smoke train: mixed sources, checkpoint + eval row land
    r = _run_cli([
        "training.train", "--data", str(smoke),
        "--label", "selftest_t4", "--epochs", "2",
        "--micro-batch", "2", "--grad-accum", "2",
        "--save-steps", "2", "--max-steps", "4",
        "--holdout-fraction", "0.2",
    ], timeout=5400)
    assert r.returncode == 0, (
        f"smoke train failed (rc {r.returncode}): ...{r.stderr[-800:]}"
    )
    ckpt = _newest(f"weights/{spec.key}/selftest_t4_step*")
    assert (ckpt / "adapter_model.safetensors").is_file(), f"no adapter in {ckpt}"
    run_dir = _newest("logs/train_selftest_t4_*")
    eval_rows = [
        json.loads(line)
        for line in (run_dir / "eval_log.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert eval_rows and any("heldout_loss" in row for row in eval_rows), (
        "no heldout_loss eval row in eval_log.jsonl"
    )

    # ---- (c) forced rollback + consecutive-rollback stop: a destructive
    # LR regresses the holdout fast. --hard-multiplier 1.0 makes ANY
    # worse-than-best eval a HARD regression, so a rollback fires at the
    # first bad eval and the SECOND consecutive one must end the run
    # early (2026-08-05 aug4 fix: rollback/re-regress loops burned both
    # epochs) -- with exit 0 and a done event standing behind
    # last_good_checkpoint, which is the orchestrator's hand-off contract.
    r2 = _run_cli([
        "training.train", "--data", str(smoke),
        "--label", "selftest_t4rb", "--epochs", "8",
        "--lr", "0.05", "--micro-batch", "2", "--grad-accum", "1",
        "--save-steps", "2", "--max-steps", "10",
        "--holdout-fraction", "0.2", "--regression-tolerance", "0",
        "--hard-multiplier", "1.0",
        "--on-regression", "rollback", "--max-rollbacks", "0",
    ], timeout=5400)
    assert r2.returncode == 0, (
        f"rollback run crashed (rc {r2.returncode}): ...{r2.stderr[-800:]}"
    )
    rb_dir = _newest("logs/train_selftest_t4rb_*")
    events = [
        json.loads(line)
        for line in (rb_dir / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    rolled = [e for e in events if e["kind"] == "rolled_back"]
    assert rolled, (
        "no rolled_back event despite lr=0.05 + tolerance 0 -- rollback "
        "path untested (rerun; if it persists, inspect events.jsonl: "
        f"{rb_dir})"
    )
    stops = [e for e in events if e["kind"] == "consecutive_rollback_stop"]
    dones = [e for e in events if e["kind"] == "done"]
    assert dones, f"no done event in {rb_dir} -- hand-off contract broken"
    assert dones[-1].get("last_good_checkpoint") not in (None, "", "None"), (
        f"done event has no last_good_checkpoint: {dones[-1]}"
    )
    assert stops, (
        "no consecutive_rollback_stop despite a destructive LR and "
        "hard-multiplier 1.0 -- either the run improved on consecutive "
        "evals (rerun; unlikely at lr=0.05) or the early-stop path is "
        f"broken (events.jsonl: {rb_dir})"
    )
    assert dones[-1].get("ended_early"), (
        f"stop fired but done event lacks ended_early: {dones[-1]}"
    )
    return (
        f"parity worst dev {worst:.2%}; smoke ckpt {ckpt.name} + "
        f"{len(eval_rows)} eval row(s); rollback fired {len(rolled)}x, "
        f"consecutive-rollback stop ended the run early"
    )


def t5_datagen() -> str:
    """Tiny parallel datagen run. GPU + NAMS required."""
    label = "selftest_t5"
    out_dir = REPO_ROOT / "data_game" / label
    if out_dir.exists():
        shutil.rmtree(out_dir)
    r = _run_cli([
        "training.generate_game_traces", "--label", label,
        "--games", "2", "--max-moves", "5", "--parallel", "2",
        "--seed", "7",
    ], timeout=7200)
    assert r.returncode == 0, (
        f"datagen failed (rc {r.returncode}) -- tripwire or crash: "
        f"...{r.stderr[-800:]}"
    )
    traces = out_dir / "traces.jsonl"
    records = [
        json.loads(line)
        for line in traces.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records, "no trace records written"
    for rec in records:
        assert rec["messages"] and rec["target_text"] is not None
        urls = [
            part["url"]
            for m in rec["messages"]
            for part in (m.get("content") or [])
            if isinstance(part, dict) and part.get("type") == "image"
        ]
        assert urls, "record without an image part"
        for u in urls:
            assert (out_dir / u).is_file(), f"referenced frame missing: {u}"
    stats = json.loads((out_dir / "generation_stats.json").read_text())
    assert stats["games"] == 2, stats
    assert stats["generations"] == len(records), (
        f'{stats["generations"]} generations vs {len(records)} records'
    )
    n_images = len(list((out_dir / "images").glob("*.png")))
    assert n_images == len(records), (
        f"{n_images} stored frames vs {len(records)} records"
    )
    rated = sum(1 for rec in records if rec["meta"].get("rating") is not None)

    # Analyst KD-anchor file: one record per round minus the (counted)
    # truncated-search skips, same stable frames, analyses nonempty.
    analyst_path = out_dir / "analyst_traces.jsonl"
    assert analyst_path.is_file(), "no analyst_traces.jsonl written"
    analyst_records = [
        json.loads(line)
        for line in analyst_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    n_expected = len(records) - stats.get("analyst_skipped_search", 0)
    assert len(analyst_records) == n_expected, (
        f"{len(analyst_records)} analyst records vs {n_expected} expected "
        f"({len(records)} rounds - "
        f"{stats.get('analyst_skipped_search', 0)} skipped)"
    )
    assert stats.get("analyst_records", 0) == len(analyst_records), stats
    for rec in analyst_records:
        assert rec["target_text"], "analyst record with empty analysis"
        urls = [
            part["url"]
            for m in rec["messages"]
            for part in (m.get("content") or [])
            if isinstance(part, dict) and part.get("type") == "image"
        ]
        assert urls, "analyst record without an image part"
        for u in urls:
            assert (out_dir / u).is_file(), f"analyst frame missing: {u}"

    plots = _newest(f"logs/datagen_stats_{label}_*")
    assert (plots / "summary.png").is_file(), f"no summary.png under {plots}"
    return (
        f"{len(records)} records / {stats['games']} games, {rated} rated, "
        f"{len(analyst_records)} analyst records, {n_images} frames, "
        f"plots at {plots.name}"
    )


def t6_ab() -> str:
    """Batched vs solo generation A/B.

      (1) identical prompts x3 -- true GPU batch (no pad); must match solo.
      (2) variable-length prompts -- the KNOWN TRANSFORMERS BUG WORKAROUND
          in agent/model.py (transformers#47651): rows decode as ONE
          verified left-padded batch. A row whose own length ~= 1 mod 32
          is unpaddable (poison mode 2, discovered by THIS test on
          2026-07-30) and gets the POISON MODE 2 RESCUE: one harmless
          filler token nudges it off the residue so it joins the batch,
          so its byte-parity reference is the NUDGED prompt's solo
          reply. Prompt lengths are chosen AT RUNTIME: two distinct
          paddable lengths (so padding genuinely engages -- asserted via
          the generate_batch log; silent cohort fallback = FAIL) plus,
          when the token grid allows, one unpaddable row to exercise the
          rescue. All rows must match their solo reference byte-for-byte.
    """
    import logging as _logging

    from agent.model import (
        PAD_POISON_MOD,
        PAD_POISON_RESIDUE,
        get_model,
        stack_equal_length,
    )

    model = get_model()
    with tempfile.TemporaryDirectory(prefix="selftest_t6_") as tmp:
        img = _tiny_png(Path(tmp) / "board.png", seed=9)
        q_same = "In one short sentence, what colors do you see?"
        same_prompts = [
            [{"role": "user", "content": [
                {"type": "image", "url": str(img)},
                {"type": "text", "text": q_same},
            ]}]
            for _ in range(3)
        ]
        same_enc = [model.encode_messages(p) for p in same_prompts]
        same_lens = {int(e["input_ids"].shape[1]) for e in same_enc}
        assert len(same_lens) == 1, same_lens
        stacked = stack_equal_length(same_enc)
        assert stacked["input_ids"].shape[0] == 3

        # Variable-length prompts, chosen AT RUNTIME around the poison
        # arithmetic (KNOWN TRANSFORMERS BUG WORKAROUND banner in
        # agent/model.py): repeated " again" nudges the token length ~1
        # per rep, giving a spread of lengths to pick from. We need two
        # DISTINCT lengths not ~= 1 mod 32 (so the padded path genuinely
        # engages) and, if the token grid yields one, a length ~= 1 mod
        # 32 (unpaddable, poison mode 2) to exercise the hybrid
        # padded+cohorts split.
        base_q = ("Answer briefly: is the grid mostly empty? Explain in "
                  "one sentence why you think so.")

        def _var_prompt(text: str) -> list[dict]:
            return [{"role": "user", "content": [
                {"type": "image", "url": str(img)},
                {"type": "text", "text": text},
            ]}]

        len_to_text: dict[int, str] = {}
        for k in range(48):
            text = base_q + " again" * k
            n = int(
                model.encode_messages(_var_prompt(text))["input_ids"].shape[1]
            )
            len_to_text.setdefault(n, text)
            paddable = sorted(
                n for n in len_to_text
                if n % PAD_POISON_MOD != PAD_POISON_RESIDUE
            )
            unpaddable = sorted(
                n for n in len_to_text
                if n % PAD_POISON_MOD == PAD_POISON_RESIDUE
            )
            if len(paddable) >= 2 and unpaddable:
                break
        assert len(paddable) >= 2, (
            f"could not build 2 distinct paddable lengths from the filler "
            f"scan: {sorted(len_to_text)}"
        )
        var_lens = [paddable[0], paddable[-1]] + unpaddable[:1]
        var_prompts = [_var_prompt(len_to_text[n]) for n in var_lens]

        original = model._sampling_kwargs
        model._sampling_kwargs = lambda: {"do_sample": False}
        try:
            solo_same = [
                model.generate(p, max_new_tokens=48) for p in same_prompts
            ]
            batched_same = model.generate_batch(
                [{"messages": p} for p in same_prompts], max_new_tokens=48,
            )
            solo_var = []
            for n, p in zip(var_lens, var_prompts):
                if n % PAD_POISON_MOD == PAD_POISON_RESIDUE:
                    # The padded batch decodes the NUDGED prompt (POISON
                    # MODE 2 RESCUE, banner in agent/model.py), so this
                    # row's byte-parity reference is the nudged prompt's
                    # solo reply, built with the same canonical helper.
                    nudge = model._nudge_unpaddable(p)
                    assert nudge is not None, (
                        f"POISON MODE 2 RESCUE could not nudge the L={n} "
                        "prompt off the residue"
                    )
                    solo_var.append(
                        model.generate(nudge[0], max_new_tokens=48)
                    )
                else:
                    solo_var.append(model.generate(p, max_new_tokens=48))

            class _Capture(_logging.Handler):
                def __init__(self) -> None:
                    super().__init__()
                    self.lines: list[str] = []

                def emit(self, record: _logging.LogRecord) -> None:
                    self.lines.append(record.getMessage())

            cap = _Capture()
            _logging.getLogger("agent.model").addHandler(cap)
            try:
                batched_var = model.generate_batch(
                    [{"messages": p} for p in var_prompts],
                    max_new_tokens=48,
                )
            finally:
                _logging.getLogger("agent.model").removeHandler(cap)
        finally:
            model._sampling_kwargs = original

    padded_engaged = any(
        "padded batch verified clean" in line for line in cap.lines
    )
    assert padded_engaged, (
        "variable-length batch did NOT take the verified-padded path "
        "(KNOWN TRANSFORMERS BUG WORKAROUND, transformers#47651) -- it "
        "fell back to length cohorts, which kills parallel-datagen "
        "throughput. generate_batch log: " + " | ".join(cap.lines)
    )
    if unpaddable:
        assert any("POISON MODE 2 RESCUE" in line for line in cap.lines), (
            f"row of length {unpaddable[0]} (~= "
            f"{PAD_POISON_RESIDUE} mod {PAD_POISON_MOD}) was NOT nudged "
            "off the poison residue -- mode 2 rescue missing? "
            "generate_batch log: " + " | ".join(cap.lines)
        )

    mism_same = [
        (i, s, b) for i, (s, b) in enumerate(zip(solo_same, batched_same))
        if s != b
    ]
    mism_var = [
        (i, s, b) for i, (s, b) in enumerate(zip(solo_var, batched_var))
        if s != b
    ]
    assert not mism_same, (
        "equal-length true batch != solo: "
        + "; ".join(
            f"[{i}] solo={s!r} batched={b!r}" for i, s, b in mism_same
        )
    )
    assert not mism_var, (
        "variable-length VERIFIED-PADDED batch != solo (the parity check "
        "passed but decode diverged -- transformers#47651 workaround "
        "assumption broken?): "
        + "; ".join(
            f"[{i}] solo={s!r} batched={b!r}" for i, s, b in mism_var
        )
    )
    hybrid = (
        f" + unpaddable {unpaddable[0]} nudged into the batch"
        if unpaddable
        else " (no unpaddable length on the filler grid this run)"
    )
    return (
        f"equal-len 3/3 identical (true batch); var-len "
        f"{len(var_prompts)}/{len(var_prompts)} identical, padded rows "
        f"{paddable[0]}+{paddable[-1]}{hybrid}; "
        f"e.g. same0={solo_same[0][:50]!r}"
    )


def t7_e2e() -> str:
    """t5's traces through real train steps: GameTraceSource (weighted CE)
    plus PlayerAnchorSource (kd_anchor; no anchor checkpoint -> base
    teacher, epoch-1 semantics) plus AnalystTraceSource (KD vs the frozen
    base), mixed buckets."""
    from training.game_traces import (
        AnalystTraceSource,
        GameTraceSource,
        PlayerAnchorSource,
    )
    from training.train import TrainConfig, run_training

    traces = REPO_ROOT / "data_game" / "selftest_t5" / "traces.jsonl"
    assert traces.is_file(), "run t5 first (its output is this stage's input)"
    src = GameTraceSource(traces, noise_seed=1)
    psrc = PlayerAnchorSource(traces, weight=1.0, noise_seed=1)
    asrc = AnalystTraceSource(
        traces.parent / "analyst_traces.jsonl",
        examples_per_epoch=4, noise_seed=1,
    )
    # No step cap: the tiny t5 corpus is a handful of batches, and draining
    # the epoch guarantees the KD (analyst) and kd_anchor (player anchor)
    # buckets actually train. Anchor weight 1.0 here (vs the production
    # 0.25) so the tiny corpus reliably yields anchor batches.
    cfg = TrainConfig(
        label="selftest_t7", epochs=1, max_steps=None, save_steps=999,
        micro_batch=2, grad_accum=1, holdout_fraction=0.0,
        log_steps=1,
    )
    rc = run_training([src, psrc, asrc], cfg)
    assert rc == 0, f"run_training returned {rc}"
    run_dir = _newest("logs/train_selftest_t7_*")
    rows = [
        json.loads(line)
        for line in (run_dir / "train_log.jsonl").read_text().splitlines()
        if line.strip()
    ]
    losses = [row["loss"] for row in rows if "loss" in row]
    assert losses and all(l == l for l in losses), (
        f"no finite losses logged in {run_dir}"
    )
    return (
        f"{len(losses)} step(s) on {src.name}+{psrc.name}+{asrc.name}, "
        "losses " + ", ".join(f"{l:.3f}" for l in losses[:4])
    )


#: One small datagen workload, shared by t8 (serial) and t9 (--parallel 3)
#: so their seconds_per_generation are directly comparable. Sized for the
#: parallel measurement: 3 games so each of t9's 3 workers plays exactly
#: one (no straggler imbalance beyond game-length variance), 4 moves so the
#: phase-locked steady state (agent/parallel_gen.py) outweighs the one-off
#: re-sync cost. The old 2x3 workload was mostly ramp and tail, which is
#: why its speedup read as noise.
_TIMING_GAMES = 3
_TIMING_MOVES = 4
_TIMING_WORKLOAD = ["--games", str(_TIMING_GAMES),
                    "--max-moves", str(_TIMING_MOVES), "--seed", "11"]


def _warmed_timed_datagen(label: str, parallel: int) -> dict:
    """Run the shared timing workload through the REAL datagen harness with
    startup excluded: the process-singleton model is loaded and warmed (one
    untimed generation) before run_generation's wall clock starts."""
    from agent.model import get_model
    from training.generate_game_traces import build_parser, run_generation

    model = get_model()
    with tempfile.TemporaryDirectory(prefix="selftest_warmup_") as tmp:
        img = _tiny_png(Path(tmp) / "warm.png", seed=3)
        model.generate([{"role": "user", "content": [
            {"type": "image", "url": str(img)},
            {"type": "text", "text": "One word: bright or dark?"},
        ]}], max_new_tokens=8)  # warmup, untimed

    out_dir = REPO_ROOT / "data_game" / label
    if out_dir.exists():
        shutil.rmtree(out_dir)
    args = build_parser().parse_args(
        ["--label", label, "--parallel", str(parallel)] + _TIMING_WORKLOAD
    )
    summary = run_generation(args)
    assert summary["generations"] > 0, summary
    assert summary["seconds_per_generation"], summary
    return summary


def t8_timing() -> str:
    """Per-generation timing of REAL serial datagen + epoch extrapolation.
    GPU + NAMS required.

    No synthetic prompts: this plays the tiny shared workload through the
    actual session harness (--parallel 1) and reads
    ``seconds_per_generation`` from generation_stats.json. Startup (model
    load, CUDA warmup) is excluded; what remains is the true per-episode
    cost -- prompt building, NAMS retrieval, image noise, the analyst
    generation -- amortized per PLAYER generation, which is exactly the
    unit ``--max-generations`` caps. t9 reuses this run as its serial
    baseline.
    """
    summary = _warmed_timed_datagen("selftest_t8_serial", parallel=1)
    s_gen = summary["seconds_per_generation"]
    epoch_h = 3000 * s_gen / 3600
    return (
        f"{s_gen:.1f}s/gen over {summary['generations']} real player gens "
        f"({summary['wall_seconds']:.0f}s wall, serial); default epoch "
        f"(--max-generations 3000) ~= {epoch_h:.1f}h serial"
    )


def t9_parallel() -> str:
    """t8's exact workload at ``--parallel 3``, compared on
    ``seconds_per_generation``. GPU + NAMS; run t8 FIRST (its
    generation_stats.json is the serial baseline).

    With the phase-locking dispatcher (agent/parallel_gen.py) plus the
    verified left-pad workaround (TO_TEST stage-6 note,
    transformers#47651), concurrent sessions should lock into the same
    round phase and decode as real batches -- expect a solid speedup, not
    the x1.18 the old fixed-window dispatcher measured. If it reads
    x~1.0-1.2, check the `dispatch: group of N [reason]` lines in the run
    log (are groups forming?) and `batch_mode` in llm_calls.jsonl (did
    padding engage?). The speedup is still REPORTED, not asserted; the
    only assertion is the slowdown tripwire (dispatcher pathology).
    """
    base = (REPO_ROOT / "data_game" / "selftest_t8_serial"
            / "generation_stats.json")
    assert base.is_file(), "run t8 first (its serial run is the baseline)"
    serial = json.loads(base.read_text(encoding="utf-8"))
    s_ser = serial["seconds_per_generation"]
    assert serial["parallel"] == 1 and s_ser, serial
    assert serial.get("games") == _TIMING_GAMES, (
        f"t8 baseline played {serial.get('games')} game(s) but the current "
        f"timing workload is {_TIMING_GAMES} -- the workload definition "
        "changed since that run; rerun t8 first"
    )

    summary = _warmed_timed_datagen("selftest_t9_par3", parallel=3)
    s_par = summary["seconds_per_generation"]
    speedup = s_ser / s_par
    # Tripwire only: --parallel 3 must not be catastrophically SLOWER
    # than serial (that would mean dispatcher pathology, e.g. phase-lock
    # holds gone wrong). Wins are evidence, not assertions.
    assert s_par <= 1.6 * s_ser, (
        f"--parallel 3 much slower than serial: {s_par:.1f} vs "
        f"{s_ser:.1f} s/gen -- dispatcher pathology?"
    )
    return (
        f"serial {s_ser:.1f}s/gen ({serial['generations']} gens, "
        f"{serial['wall_seconds']:.0f}s wall) vs --parallel 3 "
        f"{s_par:.1f}s/gen ({summary['generations']} gens, "
        f"{summary['wall_seconds']:.0f}s wall): speedup x{speedup:.2f} "
        "(low? check 'dispatch: group of N' log lines and batch_mode in "
        "llm_calls.jsonl)"
    )


#: t10 profiles the train loop on a REAL datagen corpus. Default: the
#: 2026-07-30 overnight run (3000 generations, game-length contexts -- the
#: corpus that exposed all three training OOMs; the four VRAM rules that
#: fixed them live in train.weighted_loss's docstring). Override with
#: T10_DATAGEN_LABEL=<label> to profile a different data_game/<label>/.
_T10_LABEL = os.environ.get("T10_DATAGEN_LABEL", "overnight_iter1")
_T10_BATCHES_PER_SOURCE = 4
#: Peak-VRAM tripwire: the box is ~95 GiB; the overnight OOM happened at
#: 86.85 GiB held + 18.52 GiB requested. If profiling peaks above this,
#: do NOT launch the weekend run.
_T10_VRAM_LIMIT_GIB = 88.0


def t10_traintime() -> str:
    """Timed train micro-batches from EVERY loss category, on the real
    overnight corpus, + a whole-epoch runtime estimate. GPU required; no
    NAMS. NO CHECKPOINT IS SAVED (this stage never calls save_checkpoint).

    Categories = data sources: GameTraceSource (player records, weighted
    CE -- the RL/"REINFORCE" signal), PlayerAnchorSource (kd_anchor -- the
    player trust region; teacher = base here, no anchor checkpoint),
    AnalystTraceSource (KD vs the frozen base -- the analyst anchor), and
    every enabled manifest source. For each: 4 real micro-batches
    (collation + forward + backward, KD incl.
    the teacher forward) through the same
    build_model/Collator/weighted_loss/PagedAdamW8bit stack as
    run_training, timed with CUDA sync, with per-source peak VRAM (this is
    the pre-launch check on weighted_loss's four VRAM rules; consolidated
    stage-10 note in TO_TEST.md).

    The epoch estimate uses run_training's own batch arithmetic
    (weight-scaled contributions / micro_batch, + optimizer steps every
    grad_accum batches). It slightly UNDERestimates a real epoch: bucket
    remainders add a few short batches, and save-time eval hooks (held-out
    loss + probes, ~3-5 min per save) are not included.
    """
    import math
    import random as pyrandom
    import time

    import torch

    from agent.config import CONFIG
    from agent.model import ADAPTERS, spec_for
    from training.external_data import sources_from_manifest
    from training.game_traces import (
        AnalystTraceSource,
        GameTraceSource,
        PlayerAnchorSource,
    )
    from training.train import (
        Collator,
        TrainConfig,
        attach_neftune,
        build_model,
        epoch_batches,
        materialize,
        resolve_terminator_id,
        weighted_loss,
    )

    traces_dir = REPO_ROOT / "data_game" / _T10_LABEL
    assert (traces_dir / "traces.jsonl").is_file(), (
        f"No generated data to use for the simulated training run: "
        f"{traces_dir}/traces.jsonl does not exist. t10 profiles a REAL "
        "datagen corpus -- run a datagen (training/generate_game_traces) "
        "first, or point T10_DATAGEN_LABEL at an existing "
        "data_game/<label> directory."
    )
    # Exactly the weekend run's source stack (run_weekend.train_one_epoch).
    sources = [
        GameTraceSource(traces_dir / "traces.jsonl"),
        PlayerAnchorSource(traces_dir / "traces.jsonl"),
        AnalystTraceSource(traces_dir / "analyst_traces.jsonl"),
        *sources_from_manifest(),
    ]
    cfg = TrainConfig(label="selftest_t10")  # defaults = the real recipe
    src_weights = {s.name: s.weight for s in sources}

    print(f"[t10] corpus {traces_dir.name} + {len(sources) - 3} manifest "
          "source(s); materializing (expect ~10-15 min: game-frame "
          "noising + manifest loading)...", flush=True)
    t0 = time.perf_counter()
    by_source = materialize(sources, max_example_chars=cfg.max_example_chars)
    setup_data_s = time.perf_counter() - t0
    assert by_source, "materialize produced no sources"
    print(f"[t10] materialized {sum(len(v) for v in by_source.values())} "
          f"examples from {len(by_source)} source(s) in "
          f"{setup_data_s / 60:.1f} min; loading 4-bit model + LoRA...",
          flush=True)

    t0 = time.perf_counter()
    spec = spec_for(cfg.architecture or CONFIG.model_key)
    model, processor, _targets, _proj = build_model(spec, cfg, CONFIG.hf_token)
    attach_neftune(model, cfg.neftune_alpha)
    terminator = resolve_terminator_id(
        model, getattr(processor, "tokenizer", processor)
    )
    collator = Collator(processor, ADAPTERS[spec.family], terminator,
                        compute_dtype=torch.bfloat16, device=cfg.device)
    import bitsandbytes as bnb
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(params, lr=cfg.lr,
                                         weight_decay=cfg.weight_decay)
    model.train()
    setup_model_s = time.perf_counter() - t0
    print(f"[t10] model ready in {setup_model_s / 60:.1f} min; timing "
          f"{_T10_BATCHES_PER_SOURCE} micro-batches per source (warmup "
          "first, untimed)...", flush=True)

    def _one_batch(exs: list) -> float:
        """One real training micro-batch (collation + fwd + bwd), timed."""
        torch.cuda.synchronize()
        t = time.perf_counter()
        built = collator.build_batch(exs)
        loss = weighted_loss(model, built["model_inputs"], built["weights"],
                             loss_kind=exs[0].loss)
        assert torch.isfinite(loss), f"non-finite loss on {exs[0].source}"
        (loss / cfg.grad_accum).backward()
        torch.cuda.synchronize()
        return time.perf_counter() - t

    def _opt_step() -> float:
        torch.cuda.synchronize()
        t = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return time.perf_counter() - t

    rng = pyrandom.Random(0)
    per_source: dict[str, dict] = {}
    opt_s: float | None = None
    warmed = False
    for name, exs in sorted(by_source.items()):
        batches = epoch_batches(list(exs), cfg.micro_batch, rng)
        assert batches, f"source {name} materialized but produced no batches"
        if not warmed:
            # One untimed batch: CUDA context, allocator pools, cudnn
            # autotuning all land here instead of in source #1's numbers.
            _one_batch(batches[0])
            _opt_step()
            warmed = True
        torch.cuda.reset_peak_memory_stats()
        times = [_one_batch(b)
                 for b in batches[:_T10_BATCHES_PER_SOURCE]]
        step_s = _opt_step()
        opt_s = step_s if opt_s is None else min(opt_s, step_s)
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
        # run_training's own arithmetic for this source's share of an epoch
        # (micro_batch_cap sources ride in smaller batches -> more of them)
        eff_mb = min(cfg.micro_batch, exs[0].batch_cap or cfg.micro_batch)
        n_epoch_batches = math.ceil(
            max(1, round(src_weights.get(name, 1.0) * len(exs))) / eff_mb
        )
        per_source[name] = {
            "loss_kind": batches[0][0].loss,
            "n_examples": len(exs),
            "mean_batch_s": sum(times) / len(times),
            "peak_gib": peak_gib,
            "epoch_batches": n_epoch_batches,
        }
        print(f"[t10] {name}[{batches[0][0].loss}]: "
              f"{per_source[name]['mean_batch_s']:.2f}s/batch over "
              f"{len(times)}, peak {peak_gib:.1f} GiB", flush=True)
        assert peak_gib < _T10_VRAM_LIMIT_GIB, (
            f"source {name} peaked at {peak_gib:.1f} GiB (> "
            f"{_T10_VRAM_LIMIT_GIB} tripwire) -- the weekend run would "
            "OOM; shrink micro_batch for this bucket before launching"
        )

    total_batches = sum(s["epoch_batches"] for s in per_source.values())
    train_s = sum(s["epoch_batches"] * s["mean_batch_s"]
                  for s in per_source.values())
    train_s += (total_batches / cfg.grad_accum) * (opt_s or 0.0)
    epoch_h = (setup_data_s + setup_model_s + train_s) / 3600

    lines = [
        f"{name}[{s['loss_kind']}]: {s['mean_batch_s']:.2f}s/batch, "
        f"peak {s['peak_gib']:.1f}GiB, {s['epoch_batches']} batches/epoch"
        for name, s in per_source.items()
    ]
    return (
        f"epoch estimate ~{epoch_h:.1f}h ({total_batches} micro-batches + "
        f"setup {(setup_data_s + setup_model_s) / 60:.0f}min; excludes "
        "save-time eval hooks) -- " + "; ".join(lines)
    )


# ================================================================== runner

STAGES: list[tuple[str, str, "callable"]] = [
    ("t0-env", "imports, versions, CUDA", t0_env),
    ("t1-pure", "pure-python units", t1_pure),
    ("t2-data", "materialized data vs manifest", t2_data),
    ("t3-model", "4-bit load + forward/backward", t3_model),
    ("t4-train", "batch parity + smoke train + rollback", t4_train),
    ("t5-datagen", "parallel datagen smoke", t5_datagen),
    ("t6-ab", "batched vs solo generation", t6_ab),
    ("t7-e2e", "traces -> train steps", t7_e2e),
    ("t8-timing", "real serial datagen s/gen + epoch extrapolation", t8_timing),
    ("t9-parallel", "t8's workload at --parallel 3, compared", t9_parallel),
    ("t10-traintime", "timed train batches per loss category + epoch "
                      "estimate", t10_traintime),
]


def _resolve(arg: str) -> list[tuple[str, str, "callable"]]:
    if arg == "all":
        return STAGES
    picked = [s for s in STAGES if s[0] == arg or s[0].split("-")[0] == arg]
    if not picked:
        raise SystemExit(
            f"unknown stage {arg!r}; stages: "
            + ", ".join(s[0] for s in STAGES) + ", all"
        )
    return picked


def main(argv: list[str] | None = None) -> int:
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print(__doc__)
        return 2
    os.chdir(REPO_ROOT)  # relative paths (logs/, weights/, data_*) anchor here
    failures = 0
    for stage_id, _desc, fn in _resolve(args[0]):
        try:
            evidence = fn()
            print(f"TEST {stage_id} PASS: {evidence}", flush=True)
        except BaseException as exc:
            failures += 1
            traceback.print_exc()
            first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            print(f"TEST {stage_id} FAIL: {first_line}", flush=True)
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
