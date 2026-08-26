"""HACKY_ANSWER probe: can we compute SAFE left-pads for real game prompts?

Background (TO_TEST.md stage-6 note, transformers#47651): Gemma 4 Unified
corrupts a left-padded multimodal row's prefill at SPECIFIC pad lengths --
measured pad=7 poisonous on one 282-token prompt while 1-6, 8, 9, 15, 16,
63, 64 stayed benign. That sweep used ONE base length, so "pad=7 is poison"
and "total=289 is poison" are confounded. This probe decouples them on REAL
game prompts (loaded from a datagen traces.jsonl, byte-identical to what
the player saw) and rehearses the production hack: pad to a common length,
verify each row's prefill against its solo prefill (exact check), decode
batched only if every row passes.

FINDING (2026-07-30 run, logs/probe_hacky_pads_2026-07-30_18-31-54.log):
it is the TOTAL. Corruption fired exactly when padded_total ~= 1 (mod 32),
6/6 across L = 282..2867 (pad amounts 22, 3, 26, 18, 14, 7 -- no pattern);
everything else is deterministic bf16 kernel wobble (<= ~1.75, no argmax
flip, grows with L, same class equal-length batching shows). This rerun
scores that rule and rehearses production with it.

STATUS: the workaround this probe validated NOW SHIPS in
agent/model.py's generate_batch (VLModel._plan_padded_batch; see the
KNOWN TRANSFORMERS BUG WORKAROUND banner there). Keep this probe: it is
the regression check for the mod-32 rule after any transformers upgrade,
and the evidence trail for why the hack exists.

SECOND FINDING (t6 remote run, 2026-07-30, discovered AFTER this probe's
sweeps): a row whose OWN unpadded length is ~= 1 (mod 32) is corrupted by
ANY left pad (L=289 rejected at every T in 290..297, ~24.8-logit deltas,
argmax flips). This probe's prompts never hit that residue class
(L mod 32 was 11, 30, 7, 15, 19), which is why the sweeps only exposed
the padded-TOTAL mode. Test 5 then confirmed the mode at full strength
(natural-residue game prompt poisoned at 32/32 pads, run
2026-07-30_19-46-07) AND validated the rescue: appending one harmless
token (" ."; "\n" is swallowed by the chat template) moves the row off
the residue and it pads cleanly, greedy reply content-identical.
Production now ships that rescue (VLModel._nudge_unpaddable) instead of
demoting such rows to solo cohorts.

The tests, all output marked HACKY_ANSWER:

  1. pad-vs-total: several real prompts of different lengths x a pad sweep
     (dense 1..32, coarse up to 1024). Prints the poison map and whether
     poisons align in pad space (same p poisons every length) or total
     space, plus a determinism spot-check (same (L, p) probed twice).
  2. content dependence: two DIFFERENT real prompts adjusted to the SAME
     token length; if their poison sets differ, no static table can exist
     and only the runtime parity check (outcome b) is viable.
  3. end-to-end rehearsal: mixed-length real prompts, T chosen by the
     production algorithm (start at max length, bump while any row fails
     the parity check), then padded batched GREEDY decode with datagen's
     actual stop machinery vs solo generate -- byte equality is the pass
     criterion, and it exercises RegexStopCriteria/stop_strings under
     padding.
  4. mid-decode boundary crossing: pad to T ~= 0 mod 32 so the running
     width hits the poison residue at decode steps 1 and 33; per-step
     logits + tokens vs solo decide whether the poison is prefill-only.
     (2026-07-30 result: prefill-only, crossings harmless.)
  5. naturally ~= 1 mod 32 + rescue: poison mode 2 (a row whose OWN
     length is ~= 1 mod 32 corrupts under ANY pad -- found by t6, not by
     this probe's sweeps). Reproduces it on a real prompt, then tries
     nudging the length off the residue with one harmless token (append
     or prepend) and prices the nudge against the original greedy reply.
     If a rescue works, production can pad ex-unpaddable rows instead of
     demoting them to solo cohorts.

Exit code: 1 if the parity check passed but greedy decode still diverged
(the outcome that kills the whole approach), else 0.

Run on the REMOTE box (after t8, whose traces are the default input):

    python -m training.probe_hacky_pads \
        [--traces data_game/selftest_t8_serial/traces.jsonl]
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Dense small pads (the region we have prior data for) + coarse samples of
#: the game-scale pads that real mixed-length batches would need.
PAD_SWEEP: list[int] = list(range(1, 33)) + [
    48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
]

#: The threshold and THE RULE this probe discovered (2026-07-30 runs) are
#: now PRODUCTION constants in agent/model.py -- see the KNOWN TRANSFORMERS
#: BUG WORKAROUND banner there. Alias them so the probe always scores the
#: exact values generate_batch ships with:
#:   * POISON_DLOGIT: two cleanly separated populations -- bf16 wobble up
#:     to ~1.75 with no argmax flips vs 34-52-logit corruption WITH a
#:     flip; 8.0 sits in the empty middle.
#:   * corruption fires exactly when PADDED TOTAL ~= 1 (mod 32), 6/6
#:     across L = 282..2867; pad amount alone showed no pattern.
from agent.model import (  # noqa: E402  (needs the sys.path shim above)
    PAD_PARITY_DLOGIT as POISON_DLOGIT,
    PAD_POISON_MOD as POISON_TOTAL_MOD,
    PAD_POISON_RESIDUE as POISON_TOTAL_RESIDUE,
)

#: How many prompts test 1 sweeps (spread across the length range).
N_PROMPTS = 5


# ============================================================ trace loading

def load_real_prompts(traces_path: Path, limit: int = 80) -> list[list[dict]]:
    """Real player prompts from a datagen traces.jsonl, image urls resolved
    to the stable copies next to the trace file. Exact data, no synthesis:
    these are byte-identical to what the player saw during datagen."""
    assert traces_path.is_file(), (
        f"{traces_path} not found -- run datagen (e.g. selftest t8) first, "
        "or pass --traces"
    )
    base = traces_path.parent
    prompts: list[list[dict]] = []
    with open(traces_path, encoding="utf-8") as f:
        for line in itertools.islice(f, limit):
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            for m in messages:
                content = m.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        url = Path(part["url"])
                        if not url.is_absolute():
                            url = base / url
                        assert url.is_file(), (
                            f"trace references missing frame {url}"
                        )
                        part["url"] = str(url)
            prompts.append(messages)
    assert prompts, f"no records in {traces_path}"
    return prompts


def pick_spread(model: Any, prompts: list[list[dict]],
                n: int) -> list[tuple[int, dict[str, Any]]]:
    """Encode all prompts, dedupe by token length, pick n spread across the
    length range (shortest, quartiles, longest). Returns (length, enc)."""
    by_len: dict[int, dict[str, Any]] = {}
    for msgs in prompts:
        enc = model.encode_messages(msgs)
        by_len.setdefault(int(enc["input_ids"].shape[1]), enc)
    lens = sorted(by_len)
    assert len(lens) >= 2, f"need >=2 distinct lengths, got {lens}"
    idx = sorted({round(i * (len(lens) - 1) / max(n - 1, 1))
                  for i in range(n)})
    picked = [(lens[i], by_len[lens[i]]) for i in idx]
    print(f"prompt lengths available: {lens}")
    print(f"picked for sweep: {[length for length, _ in picked]}")
    return picked


# ================================================================== helpers

def _poison(solo: Any, padded: Any) -> tuple[bool, float, bool]:
    """(is_poison, max_delta, argmax_flipped) for one padded row."""
    delta = float((padded - solo).abs().max())
    flipped = int(padded.argmax()) != int(solo.argmax())
    return (flipped or delta > POISON_DLOGIT), delta, flipped


def _sweep_one(model: Any, enc: dict[str, Any], pad_id: int,
               pads: list[int]) -> dict[int, tuple[bool, float, bool]]:
    from agent.model import left_pad_row

    solo = model.prefill_last_logits(enc)[0]
    out: dict[int, tuple[bool, float, bool]] = {}
    for p in pads:
        padded = model.prefill_last_logits(
            left_pad_row(enc, p, pad_id)
        )[0]
        out[p] = _poison(solo, padded)
    return out


# ==================================================================== tests

def test1_pad_vs_total(model: Any, picked: list[tuple[int, dict]],
                       pad_id: int) -> dict[int, set[int]]:
    """Sweep pads over several real lengths; is poison a function of the
    pad, of the total, or of neither?"""
    print("\n=== HACKY_ANSWER test 1: pad-vs-total poison map ===")
    poison_map: dict[int, set[int]] = {}
    for length, enc in picked:
        results = _sweep_one(model, enc, pad_id, PAD_SWEEP)
        poisons = {p for p, (bad, _, _) in results.items() if bad}
        poison_map[length] = poisons
        worst = max(results.values(), key=lambda r: r[1])
        print(f"L={length}: poison pads {sorted(poisons) or 'NONE'} "
              f"(worst max|dLogit|={worst[1]:.3f})")
        for p in sorted(poisons):
            bad, delta, flipped = results[p]
            print(f"    pad={p:<5d} total={length + p:<6d} "
                  f"max|dLogit|={delta:.3f} argmax_flipped={flipped}")

    # Score THE RULE: poison iff (L + p) % 32 == 1.
    print("\n--- mod-32 rule scorecard ---")
    hits = misses = false_alarms = 0
    for length, poisons in sorted(poison_map.items()):
        predicted = {
            p for p in PAD_SWEEP
            if (length + p) % POISON_TOTAL_MOD == POISON_TOTAL_RESIDUE
        }
        hits += len(predicted & poisons)
        misses += len(poisons - predicted)
        false_alarms += len(predicted - poisons)
        print(f"L={length}: predicted {sorted(predicted)} "
              f"observed {sorted(poisons)} "
              f"{'MATCH' if predicted == poisons else 'MISMATCH'}")
    print(f"rule totals: {hits} hit(s), {misses} unpredicted poison(s), "
          f"{false_alarms} false alarm(s) -- 0/0 in the last two slots "
          "means 'never pad to a total ~= 1 mod 32' is sufficient")

    # Alignment analysis over prompt pairs.
    print("\n--- alignment analysis ---")
    pairs = list(itertools.combinations(sorted(poison_map), 2))
    pad_aligned = sum(
        1 for a, b in pairs if poison_map[a] == poison_map[b]
    )
    print(f"pad-space: {pad_aligned}/{len(pairs)} prompt pairs have "
          "IDENTICAL poison-pad sets "
          "(all pairs identical => pad length alone decides)")
    for a, b in pairs:
        totals_a = {a + p for p in poison_map[a]}
        totals_b = {b + p for p in poison_map[b]}
        # Only totals reachable by both prompts' sweeps are comparable.
        reach_a = {a + p for p in PAD_SWEEP}
        reach_b = {b + p for p in PAD_SWEEP}
        overlap = reach_a & reach_b
        if overlap:
            agree = sum(
                1 for t in overlap
                if (t in totals_a) == (t in totals_b)
            )
            print(f"total-space L={a} vs L={b}: {agree}/{len(overlap)} "
                  f"comparable totals agree")
        else:
            print(f"total-space L={a} vs L={b}: sweeps share no totals "
                  "(inconclusive)")

    # Determinism spot-check: same (L, p) probed twice must match exactly.
    # One poison pad and one safe pad when both exist; a prompt where ALL
    # pads poison (or none do) still gets checked with what it has.
    length, enc = picked[0]
    poisons = sorted(poison_map[length])
    safes = [p for p in PAD_SWEEP if p not in poison_map[length]]
    probe_pads = poisons[:1] + safes[:1]
    for p in probe_pads:
        r1 = _sweep_one(model, enc, pad_id, [p])[p]
        r2 = _sweep_one(model, enc, pad_id, [p])[p]
        print(f"determinism L={length} pad={p}: "
              f"run1={r1[1]:.6f} run2={r2[1]:.6f} "
              f"{'STABLE' if r1 == r2 else 'UNSTABLE -- rethink everything'}")
    return poison_map


def _retarget_length(
    model: Any, messages: list[dict], target: int
) -> tuple[list[dict], dict[str, Any]] | None:
    """Trim words from (or append single-token filler to) the last text
    part until the encoded length is exactly ``target``. Keeps the prompt
    game-real except for the trailing words. Returns ``(messages, enc)``
    or None if it will not converge."""
    import copy

    msgs = copy.deepcopy(messages)
    text_part = None
    for m in reversed(msgs):
        content = m.get("content")
        if isinstance(content, list):
            for part in reversed(content):
                if isinstance(part, dict) and part.get("type") == "text":
                    text_part = part
                    break
        if text_part:
            break
    assert text_part is not None
    for _ in range(300):
        enc = model.encode_messages(msgs)
        n = int(enc["input_ids"].shape[1])
        if n == target:
            return msgs, enc
        if n > target + 40:
            # Coarse chop: ~4 chars/token, deliberately undershooting the
            # estimate so the fine loop below lands the exact target.
            cut = (n - target) * 3
            if cut >= len(text_part["text"]):
                return None
            text_part["text"] = text_part["text"][:-cut]
        elif n > target:
            words = text_part["text"].rsplit(" ", 1)
            if len(words) < 2:
                return None
            text_part["text"] = words[0]
        else:
            text_part["text"] += " x"
    return None


def test2_content(model: Any, prompts: list[list[dict]],
                  picked: list[tuple[int, dict]], pad_id: int) -> None:
    """Two different real prompts at the SAME token length: do they share
    one poison set? (If not, no static safe-pad table can exist.)"""
    print("\n=== HACKY_ANSWER test 2: content dependence at equal length ===")
    target = picked[0][0]
    enc_a = picked[0][1]
    # Candidates nearest the target length first: least trimming, most
    # preserved game content.
    ranked = sorted(
        prompts,
        key=lambda m: abs(
            int(model.encode_messages(m)["input_ids"].shape[1]) - target
        ),
    )
    enc_b = None
    for msgs in ranked:
        got = _retarget_length(model, msgs, target)
        if got is not None:
            enc_b = got[1]
            same = bool(
                (enc_b["input_ids"] == enc_a["input_ids"]).all()
            )
            if not same:
                break
            enc_b = None
    if enc_b is None:
        print("SKIPPED: could not adjust a second prompt to exactly "
              f"{target} tokens -- rerun with a bigger traces file")
        return
    # Include the rule-predicted poison pad for this length (plus
    # neighbors) -- the interesting comparison point.
    predicted = ((POISON_TOTAL_RESIDUE - target) % POISON_TOTAL_MOD
                 or POISON_TOTAL_MOD)
    sweep = sorted({*range(1, 17), 24, 32, 64, 256,
                    max(predicted - 1, 1), predicted, predicted + 1})
    res_a = _sweep_one(model, enc_a, pad_id, sweep)
    res_b = _sweep_one(model, enc_b, pad_id, sweep)
    set_a = {p for p, (bad, _, _) in res_a.items() if bad}
    set_b = {p for p, (bad, _, _) in res_b.items() if bad}
    print(f"L={target} prompt A poison pads: {sorted(set_a) or 'NONE'}")
    print(f"L={target} prompt B poison pads: {sorted(set_b) or 'NONE'}")
    if set_a == set_b:
        print("IDENTICAL poison sets -> poison is content-independent at "
              "this length (static table plausible, outcome a)")
    else:
        print("DIFFERENT poison sets -> content-dependent: only the "
              "runtime parity check works (outcome b)")


def test3_rehearsal(model: Any, picked: list[tuple[int, dict]],
                    prompts_by_len: dict[int, list[dict]],
                    pad_id: int) -> int:
    """Production rehearsal: choose T by the parity check, padded batched
    greedy decode with the REAL datagen stop machinery, compare to solo."""
    from agent import game_io, modes
    from agent.model import left_pad_stack

    print("\n=== HACKY_ANSWER test 3: end-to-end padded greedy decode ===")
    chosen = picked[:3]
    lens = [length for length, _ in chosen]
    rows = [enc for _, enc in chosen]
    messages = [prompts_by_len[length] for length in lens]
    solo_logits = [model.prefill_last_logits(enc)[0] for enc in rows]

    # The production T search: start at max length, SKIP totals hitting
    # the mod-32 poison rule, and verify the remainder with the parity
    # check (belt and suspenders).
    import torch

    target = max(lens)
    chosen_T = None
    for T in range(target, target + 9):
        if T % POISON_TOTAL_MOD == POISON_TOTAL_RESIDUE:
            print(f"T={T}: SKIPPED by rule (T ~= "
                  f"{POISON_TOTAL_RESIDUE} mod {POISON_TOTAL_MOD})")
            continue
        stacked, pads = left_pad_stack(rows, pad_id, target_len=T)
        batch_logits = model.prefill_last_logits(stacked)
        verdicts = [
            _poison(solo_logits[i], batch_logits[i])
            for i in range(len(rows))
        ]
        bad = [i for i, (b, _, _) in enumerate(verdicts) if b]
        detail = "  ".join(
            f"row{i}(pad={pads[i]}):d={verdicts[i][1]:.3f}"
            f"{'/FLIP' if verdicts[i][2] else ''}"
            for i in range(len(rows))
        )
        print(f"T={T}: {detail}  -> {'POISON rows ' + str(bad) if bad else 'ALL CLEAN'}")
        if not bad:
            chosen_T = T
            break
    if chosen_T is None:
        print("NO clean T within +8 of max length -- parity-checked "
              "padding cannot batch this trio (outcome c territory)")
        return 0

    # Greedy end-to-end with datagen's real stop machinery.
    original = model._sampling_kwargs
    model._sampling_kwargs = lambda: {"do_sample": False}
    try:
        solo = [
            model.generate(
                msgs, max_new_tokens=96,
                stop_strings=None,
                stop_regex=(
                    game_io.PLAYER_STOP_PATTERN + "|"
                    + modes.SEARCH_TOOL_PATTERN
                ),
            )
            for msgs in messages
        ]
        stacked, pads = left_pad_stack(rows, pad_id, target_len=chosen_T)
        tokenizer = getattr(model.processor, "tokenizer", model.processor)
        from transformers import StoppingCriteriaList

        from agent.model import RegexStopCriteria

        inputs = model._move_inputs_to_model(stacked)
        with torch.inference_mode():
            out = model.model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                tokenizer=tokenizer,
                stopping_criteria=StoppingCriteriaList([
                    RegexStopCriteria(
                        game_io.PLAYER_STOP_PATTERN + "|"
                        + modes.SEARCH_TOOL_PATTERN,
                        tokenizer,
                        prompt_len=chosen_T,
                    )
                ]),
            )
        batched = [
            model.processor.decode(
                out[i][chosen_T:], skip_special_tokens=True
            ).strip()
            for i in range(len(rows))
        ]
    finally:
        model._sampling_kwargs = original

    failures = 0
    for i, (s, b) in enumerate(zip(solo, batched)):
        ok = s == b
        if not ok:
            failures += 1
        gen = out[i][chosen_T:]
        n_gen = int((gen != pad_id).sum())
        crossed = [w for w in range(chosen_T + 1, chosen_T + n_gen + 1)
                   if w % POISON_TOTAL_MOD == POISON_TOTAL_RESIDUE]
        print(f"row{i} (L={lens[i]}, pad={chosen_T - lens[i]}): "
              f"{'IDENTICAL' if ok else 'DIVERGED'} "
              f"({n_gen} tok generated; mid-decode widths ~=1 mod 32 "
              f"crossed: {crossed or 'none'})")
        if not ok:
            print(f"    solo    : {s[:120]!r}")
            print(f"    batched : {b[:120]!r}")
    if failures:
        print("\nPARITY CHECK PASSED BUT GREEDY DECODE DIVERGED -- the "
              "prefill check does NOT guarantee clean decode; the padded "
              "approach is dead as designed (escalate to outcome c: solo "
              "prefill + KV stitching -- but read 'KV reuse: attempted, "
              "reverted' in TO_TEST.md first: transformers 5.14 cache "
              "surgery has landmines).")
    else:
        print("\nALL ROWS IDENTICAL: parity-checked padding survives the "
              "full decode INCLUDING stop_strings + stop_regex -- "
              "implementable in generate_batch.")
    return failures


def test4_mid_decode(model: Any, picked: list[tuple[int, dict]],
                     pad_id: int) -> int:
    """Does decode corrupt when the RUNNING width crosses ~=1 mod 32?

    The prefill poison fires at padded width ~= 1 mod 32. During decode
    the attended width grows by one per step, so every reply longer than
    32 tokens crosses that residue WITH pads in the mask -- if the
    crossing corrupts, padding is unusable without per-step repadding.
    Adversarial setup: T ~= 0 mod 32, so the very FIRST decode step
    attends width T+1 ~= 1 mod 32 (and step 33 crosses again). Greedy, no
    stop strings, raw per-step logits via output_logits=True; compare
    each padded row's step logits and tokens against its solo run.
    """
    import torch
    from agent.model import left_pad_stack

    print("\n=== HACKY_ANSWER test 4: mid-decode boundary crossing ===")
    chosen = picked[:2]
    lens = [length for length, _ in chosen]
    rows = [enc for _, enc in chosen]
    steps = 64
    # Smallest T >= max length with T ~= 0 mod 32.
    T = max(lens) + (-max(lens)) % POISON_TOTAL_MOD
    boundary_steps = [k for k in range(steps)
                      if (T + k) % POISON_TOTAL_MOD == POISON_TOTAL_RESIDUE]
    print(f"rows L={lens} padded to T={T} (T%{POISON_TOTAL_MOD}="
          f"{T % POISON_TOTAL_MOD}); padded width hits residue "
          f"{POISON_TOTAL_RESIDUE} at decode steps {boundary_steps}")

    def _gen_with_logits(enc: dict[str, Any]):
        inputs = model._move_inputs_to_model(enc)
        with torch.inference_mode():
            out = model.model.generate(
                **inputs, max_new_tokens=steps, do_sample=False,
                return_dict_in_generate=True, output_logits=True,
            )
        return (out.sequences.cpu(),
                [step.float().cpu() for step in out.logits])

    solo = [_gen_with_logits(enc) for enc in rows]
    stacked, pads = left_pad_stack(rows, pad_id, target_len=T)
    seqs, step_logits = _gen_with_logits(stacked)

    eos_id = getattr(
        getattr(model.processor, "tokenizer", model.processor),
        "eos_token_id", None,
    )
    failures = 0
    for i in range(len(rows)):
        length = lens[i]
        solo_seq, solo_logits = solo[i]
        gen_solo = solo_seq[0][length:]
        gen_pad = seqs[i][T:]
        n = min(len(gen_solo), len(gen_pad),
                len(solo_logits), len(step_logits))
        # Stop comparing at solo's own end-of-turn: beyond it a finished
        # row only accumulates padding.
        for k in range(n):
            if eos_id is not None and int(gen_solo[k]) == eos_id:
                n = k + 1
                break
        first_bad: int | None = None
        max_delta = 0.0
        print(f"--- row{i} (L={length}, pad={T - length}, comparing "
              f"{n} steps) ---")
        for k in range(n):
            delta = float(
                (step_logits[k][i] - solo_logits[k][0]).abs().max()
            )
            max_delta = max(max_delta, delta)
            tok_ok = int(gen_pad[k]) == int(gen_solo[k])
            if not tok_ok and first_bad is None:
                first_bad = k
            marks = []
            if (T + k) % POISON_TOTAL_MOD == POISON_TOTAL_RESIDUE:
                marks.append("padded-width boundary")
            if (length + k) % POISON_TOTAL_MOD == POISON_TOTAL_RESIDUE:
                marks.append("solo-width boundary")
            if marks or not tok_ok or delta > POISON_DLOGIT:
                print(f"  k={k:<3d} max|dLogit|={delta:8.3f} "
                      f"token_match={tok_ok}"
                      f"{('  <-- ' + ', '.join(marks)) if marks else ''}")
        if first_bad is None:
            print(f"row{i}: all {n} tokens IDENTICAL, peak per-step "
                  f"max|dLogit|={max_delta:.3f} -- boundary crossings "
                  "harmless in decode")
        else:
            failures += 1
            print(f"row{i}: FIRST TOKEN MISMATCH at step {first_bad} "
                  f"(padded width {(T + first_bad)}, residue "
                  f"{(T + first_bad) % POISON_TOTAL_MOD}); peak "
                  f"max|dLogit|={max_delta:.3f}")
    if failures:
        print("\nMID-DECODE CORRUPTION CONFIRMED: padding needs per-step "
              "repadding around the boundary (or cohorts stay).")
    else:
        print("\nNO mid-decode corruption: the poison is prefill-only; "
              "skipping T ~= 1 mod 32 at batch build time is sufficient.")
    return failures


def _nudged_copy(messages: list[dict], where: str,
                 filler: str) -> list[dict]:
    """Deep-copy `messages` with `filler` glued onto the first ('prepend')
    or last ('append') text part -- the candidate production rescue for
    naturally-unpaddable rows."""
    import copy

    msgs = copy.deepcopy(messages)
    parts = [
        part
        for m in msgs
        for part in (m.get("content") or [])
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    assert parts, "no text part to nudge"
    if where == "prepend":
        parts[0]["text"] = filler + parts[0]["text"]
    else:
        parts[-1]["text"] = parts[-1]["text"] + filler
    return msgs


def _greedy_reply(model: Any, enc: dict[str, Any], steps: int = 48) -> str:
    """Plain greedy decode of one batch-1 encoding, decoded reply text."""
    import torch

    inputs = model._move_inputs_to_model(enc)
    with torch.inference_mode():
        out = model.model.generate(
            **inputs, max_new_tokens=steps, do_sample=False,
        )
    n = int(enc["input_ids"].shape[1])
    return model.processor.decode(out[0][n:], skip_special_tokens=True).strip()


def test5_natural_residue(model: Any, prompts: list[list[dict]],
                          pad_id: int) -> int:
    """HACKY_ANSWER test 5: rows NATURALLY ~= 1 mod 32, and their rescue.

    Poison mode 2 (discovered by the t6 remote run 2026-07-30, AFTER this
    probe's sweeps): a row whose OWN unpadded length is ~= 1 mod 32 is
    corrupted by ANY left pad, even at rule-clean target widths.
    Production currently refuses to pad such rows (they decode in their
    own cohort). This test (a) reproduces the mode on a REAL game prompt
    retargeted to the residue, sweeping pads 1..32, and (b) evaluates the
    candidate rescue: nudge the row off the residue with one generally
    harmless extra token -- appended after the last text part, or
    prepended before the first -- and check that the nudged row pads
    cleanly like any ordinary row. It also prices the nudge: greedy reply
    of the nudged prompt vs the original (REPORTED, not asserted -- a
    near-tie flip from one extra whitespace token is acceptable noise in
    sampled production datagen, but we want to see it).

    Returns the number of FAILED rescues (a shifted length that still
    poisons at rule-clean pads); whitespace swallowed by the chat
    template is reported but not a failure.
    """
    print("\n=== HACKY_ANSWER test 5: naturally ~= 1 mod 32 + rescue ===")
    base = None
    for msgs in prompts:
        n = int(model.encode_messages(msgs)["input_ids"].shape[1])
        target = n + ((POISON_TOTAL_RESIDUE - n) % POISON_TOTAL_MOD)
        got = _retarget_length(model, msgs, target)
        if got is not None:
            base = got
            break
    if base is None:
        print("SKIPPED: could not retarget any real prompt to "
              f"~= {POISON_TOTAL_RESIDUE} mod {POISON_TOTAL_MOD}")
        return 0
    base_msgs, base_enc = base
    L = int(base_enc["input_ids"].shape[1])
    print(f"natural-residue row: L={L} "
          f"(L % {POISON_TOTAL_MOD} = {L % POISON_TOTAL_MOD})")

    # (a) Baseline poison map: pads 1..32 on the natural-residue row.
    # t6 evidence predicts poison at (nearly) EVERY pad; the one pad
    # hitting total ~= 1 mod 32 is poison mode 1 regardless.
    sweep = list(range(1, POISON_TOTAL_MOD + 1))
    res = _sweep_one(model, base_enc, pad_id, sweep)
    poisons = sorted(p for p, (bad, _, _) in res.items() if bad)
    mode1 = [p for p in poisons
             if (L + p) % POISON_TOTAL_MOD == POISON_TOTAL_RESIDUE]
    worst = max(res.values(), key=lambda r: r[1])
    print(f"baseline: {len(poisons)}/{len(sweep)} pads poisoned "
          f"{sorted(poisons) or 'NONE'} (mode-1 totals among them: "
          f"{mode1}; worst max|dLogit|={worst[1]:.3f})")

    solo_reply = _greedy_reply(model, base_enc)
    failures = 0
    for where, fillers in (
        ("append", ["\n", " .", " Okay."]),
        ("prepend", [" ", ". ", "Note: "]),
    ):
        rescued = None
        tried: list[tuple[str, int]] = []
        for filler in fillers:
            msgs2 = _nudged_copy(base_msgs, where, filler)
            enc2 = model.encode_messages(msgs2)
            L2 = int(enc2["input_ids"].shape[1])
            tried.append((filler, L2))
            if (L2 != L
                    and L2 % POISON_TOTAL_MOD != POISON_TOTAL_RESIDUE):
                rescued = (filler, enc2, L2)
                break
        if rescued is None:
            print(f"{where}: NO filler moved the length off the residue "
                  f"(filler -> new L: {tried}) -- chat template likely "
                  "normalizes it; strategy unusable")
            continue
        filler, enc2, L2 = rescued
        res2 = _sweep_one(model, enc2, pad_id, sweep)
        poisons2 = sorted(p for p, (bad, _, _) in res2.items() if bad)
        predicted2 = [p for p in sweep
                      if (L2 + p) % POISON_TOTAL_MOD
                      == POISON_TOTAL_RESIDUE]
        unpredicted = [p for p in poisons2 if p not in predicted2]
        print(f"{where} filler {filler!r}: L {L} -> {L2} "
              f"(residue {L2 % POISON_TOTAL_MOD}); poison pads "
              f"{poisons2 or 'NONE'}, mode-1 prediction {predicted2}")
        if unpredicted:
            failures += 1
            print(f"  RESCUE FAILED: rule-clean pads still poisoned: "
                  f"{unpredicted}")
        else:
            print("  RESCUE WORKS: nudged row pads like any ordinary row "
                  "(only the mode-1 total is poisoned, and production "
                  "never pads to it)")
        reply2 = _greedy_reply(model, enc2)
        if reply2 == solo_reply:
            print(f"  nudge cost ({where} {filler!r}): greedy reply "
                  "IDENTICAL to the un-nudged prompt's")
        else:
            print(f"  nudge cost ({where} {filler!r}): greedy reply "
                  "DIFFERS from the un-nudged prompt's:")
            print(f"    original: {solo_reply[:120]!r}")
            print(f"    nudged  : {reply2[:120]!r}")
    return failures


# ===================================================================== main

class _Tee:
    """Duplicate a text stream into the console and the probe log file."""

    def __init__(self, *streams: Any):
        self._streams = streams

    def write(self, s: str) -> int:
        for st in self._streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self) -> None:
        for st in self._streams:
            st.flush()


def main(argv: list[str] | None = None) -> int:
    import time

    log_path = (REPO_ROOT / "logs"
                / f"probe_hacky_pads_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.StreamHandler(log_file)],
    )
    print(f"(full output duplicated to {log_path})")
    parser = argparse.ArgumentParser(
        prog="python -m training.probe_hacky_pads",
        description="HACKY_ANSWER: characterize safe left-pads on real "
                    "game prompts (see module docstring)",
    )
    parser.add_argument(
        "--traces",
        default="data_game/selftest_t8_serial/traces.jsonl",
        help="datagen traces.jsonl to draw real prompts from",
    )
    parser.add_argument(
        "--only", default=None,
        help="comma-separated test numbers to run (e.g. --only 4); "
             "default: all",
    )
    args = parser.parse_args(argv)

    from agent.model import get_model

    model = get_model()
    tok = getattr(model.processor, "tokenizer", model.processor)
    pad_id = tok.pad_token_id
    assert pad_id is not None, "tokenizer has no pad token"

    prompts = load_real_prompts(REPO_ROOT / args.traces)
    # Keep messages addressable by encoded length for test 3.
    prompts_by_len: dict[int, list[dict]] = {}
    for msgs in prompts:
        enc = model.encode_messages(msgs)
        prompts_by_len.setdefault(int(enc["input_ids"].shape[1]), msgs)
    picked = pick_spread(model, prompts, N_PROMPTS)

    want = set((args.only or "1,2,3,4,5").split(","))
    failures = 0
    if "1" in want:
        test1_pad_vs_total(model, picked, pad_id)
    if "2" in want:
        test2_content(model, prompts, picked, pad_id)
    if "3" in want:
        failures += test3_rehearsal(model, picked, prompts_by_len, pad_id)
    if "4" in want:
        failures += test4_mid_decode(model, picked, pad_id)
    if "5" in want:
        failures += test5_natural_residue(model, prompts, pad_id)
    print(f"\n(full output saved to {log_path})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
