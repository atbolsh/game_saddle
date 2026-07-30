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

Three tests, all output marked HACKY_ANSWER:

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

#: A pad is POISON if the argmax flips or any logit moves by more than
#: this. The 2026-07-30 run on real game prompts (L 1547-2867) measured
#: two cleanly separated populations: deterministic bf16 kernel wobble up
#: to ~1.75 with NO argmax flips (same class equal-length batching shows,
#: and t6 passes byte-for-byte there), vs 34-52-logit corruption WITH a
#: flip. 8.0 sits in the empty middle.
POISON_DLOGIT = 8.0

#: THE RULE (hypothesis from the 2026-07-30 run, 6/6 including the old
#: 282-token repro): corruption fires exactly when the PADDED TOTAL
#: length is ~= 1 (mod 32). Pad amount alone showed no pattern
#: (22, 3, 26, 18, 14). Test 1 scores this prediction; test 3 uses it to
#: choose T.
POISON_TOTAL_MOD = 32
POISON_TOTAL_RESIDUE = 1

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


def _retarget_length(model: Any, messages: list[dict],
                     target: int) -> dict[str, Any] | None:
    """Trim words from (or append single-token filler to) the last text
    part until the encoded length is exactly ``target``. Keeps the prompt
    game-real except for the trailing words. None if it will not converge."""
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
            return enc
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
        enc_b = _retarget_length(model, msgs, target)
        if enc_b is not None:
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
                stop_strings=game_io.MOVE_STOP_STRINGS,
                stop_regex=modes.SEARCH_TOOL_PATTERN,
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
                stop_strings=list(game_io.MOVE_STOP_STRINGS),
                tokenizer=tokenizer,
                stopping_criteria=StoppingCriteriaList([
                    RegexStopCriteria(
                        modes.SEARCH_TOOL_PATTERN, tokenizer,
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
        print(f"row{i} (L={lens[i]}, pad={chosen_T - lens[i]}): "
              f"{'IDENTICAL' if ok else 'DIVERGED'}")
        if not ok:
            print(f"    solo    : {s[:120]!r}")
            print(f"    batched : {b[:120]!r}")
    if failures:
        print("\nPARITY CHECK PASSED BUT GREEDY DECODE DIVERGED -- the "
              "prefill check does NOT guarantee clean decode; the padded "
              "approach is dead as designed (escalate to outcome c: solo "
              "prefill + KV stitching).")
    else:
        print("\nALL ROWS IDENTICAL: parity-checked padding survives the "
              "full decode INCLUDING stop_strings + stop_regex -- "
              "implementable in generate_batch.")
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

    test1_pad_vs_total(model, picked, pad_id)
    test2_content(model, prompts, picked, pad_id)
    failures = test3_rehearsal(model, picked, prompts_by_len, pad_id)
    print(f"\n(full output saved to {log_path})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
