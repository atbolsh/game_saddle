"""Left-pad regression probe on real game prompts.

Historical finding (2026-07-30, transformers#47651): mem-efficient SDPA
corrupted the last position when the attended width was 1 mod 32, both as
a left-padded total and as a prompt's own length. Torch >= 2.14 fixes
that kernel (pytorch#191984). This probe now requires those widths to
stay inside bf16 wobble (max |dLogit| <= PAD_PARITY_DLOGIT, no argmax
flip). A poisoned pad is a failure, including totals and natural lengths
that are 1 mod 32. Decode crossing that width stays a separate assertion
(test 4) and must stay clean. No filler-token rescue: production no
longer appends one.

The tests:

  1. pad-vs-total: several real prompts x a pad sweep (dense 1..32, coarse
     up to 1024). Every pad must stay clean, including totals that are
     1 mod 32. A poisoned pad fails the probe.
  2. content dependence: two different real prompts adjusted to the same
     token length. Both must stay clean at the historical residue pad.
  3. end-to-end rehearsal: mixed-length real prompts padded to the
     smallest width >= max length that is 1 mod 32, then greedy decode
     with datagen's stop machinery vs solo generate.
  4. mid-decode boundary crossing: pad to T ≡ 0 mod 32 so the running
     width hits 1 mod 32 during decode. Tokens must match solo. Decode
     was already clean before the kernel fix and must stay clean.
  5. natural length ≡ 1 mod 32: a real prompt retargeted onto that
     residue, left-padded across pads 1..32. Every pad must stay clean.
     No filler is appended.

Exit code: 1 if any of those widths poisons or greedy decode diverges.

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

#: bf16 wobble vs a real last-position flip. 8.0 is the empty middle
#: between ~1.75 wobble and the old ~34-logit corruption.
from agent.model import PAD_PARITY_DLOGIT as POISON_DLOGIT  # noqa: E402

#: Historical kernel condition (transformers#47651). These widths must
#: now be clean; production does not dodge them.
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
                       pad_id: int) -> int:
    """Sweep pads over several real lengths. Every pad must stay clean,
    including totals that are 1 mod 32."""
    print("\n=== test 1: pad-vs-total, residue widths must be clean ===")
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

    print("\n--- residue widths (total % 32 == 1) must be clean ---")
    failures = 0
    for length, poisons in sorted(poison_map.items()):
        residue = sorted(
            p for p in PAD_SWEEP
            if (length + p) % POISON_TOTAL_MOD == POISON_TOTAL_RESIDUE
        )
        bad_residue = sorted(set(residue) & poisons)
        bad_other = sorted(poisons - set(residue))
        print(f"L={length}: residue pads {residue} "
              f"poisoned {bad_residue or 'NONE'}; "
              f"other poisoned {bad_other or 'NONE'}")
        failures += len(bad_residue) + len(bad_other)
    if failures:
        print(f"FAIL: {failures} pad(s) left bf16 wobble or flipped argmax")
    else:
        print("CLEAN: every swept pad, including totals ≡ 1 mod 32, "
              "stayed inside bf16 wobble")

    length, enc = picked[0]
    # Spot-check the residue pad twice: same logits, and still clean.
    residue_pad = (
        (POISON_TOTAL_RESIDUE - length) % POISON_TOTAL_MOD
    ) or POISON_TOTAL_MOD
    r1 = _sweep_one(model, enc, pad_id, [residue_pad])[residue_pad]
    r2 = _sweep_one(model, enc, pad_id, [residue_pad])[residue_pad]
    print(f"determinism L={length} pad={residue_pad} "
          f"(total {length + residue_pad}): "
          f"run1={r1[1]:.6f} run2={r2[1]:.6f} "
          f"{'STABLE' if r1 == r2 else 'UNSTABLE'}")
    if r1[0] or r2[0] or r1 != r2:
        failures += 1
    return failures


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
                  picked: list[tuple[int, dict]], pad_id: int) -> int:
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
        return 0
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
    poisoned = set_a | set_b
    if poisoned:
        print(f"FAIL: pads {sorted(poisoned)} left bf16 wobble "
              f"(residue pad for this length is {predicted})")
        return len(poisoned)
    print(f"CLEAN at every swept pad, including residue pad {predicted}")
    return 0


def test3_rehearsal(model: Any, picked: list[tuple[int, dict]],
                    prompts_by_len: dict[int, list[dict]],
                    pad_id: int) -> int:
    """Pad to a 1-mod-32 total and greedy-decode against solo."""
    from agent import game_io, modes
    from agent.model import left_pad_stack

    print("\n=== HACKY_ANSWER test 3: end-to-end padded greedy decode ===")
    chosen = picked[:3]
    lens = [length for length, _ in chosen]
    rows = [enc for _, enc in chosen]
    messages = [prompts_by_len[length] for length in lens]
    solo_logits = [model.prefill_last_logits(enc)[0] for enc in rows]

    import torch

    target = max(lens)
    # Smallest width >= the longest row whose total is 1 mod 32.
    bump = (POISON_TOTAL_RESIDUE - target) % POISON_TOTAL_MOD
    chosen_T = target if bump == 0 else target + bump
    stacked, pads = left_pad_stack(rows, pad_id, target_len=chosen_T)
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
    print(f"T={chosen_T} (T % {POISON_TOTAL_MOD} == "
          f"{chosen_T % POISON_TOTAL_MOD}): {detail}")
    if bad:
        print(f"FAIL: residue total poisoned rows {bad}")
        return len(bad)

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
        print("\nFAIL: decode diverged while the running width crossed "
              "1 mod 32.")
    else:
        print("\nCLEAN: decode matches solo across the 1-mod-32 width.")
    return failures


def test5_natural_residue(model: Any, prompts: list[list[dict]],
                          pad_id: int) -> int:
    """A prompt whose own length is 1 mod 32 must left-pad cleanly.

    Before torch 2.14 that length corrupted under every left pad. No
    filler is appended; the un-nudged encoding is the one under test.
    """
    print("\n=== test 5: natural length ≡ 1 mod 32, no filler ===")
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
              f"{POISON_TOTAL_RESIDUE} mod {POISON_TOTAL_MOD}")
        return 0
    _msgs, base_enc = base
    L = int(base_enc["input_ids"].shape[1])
    print(f"natural-residue row: L={L} "
          f"(L % {POISON_TOTAL_MOD} = {L % POISON_TOTAL_MOD})")
    sweep = list(range(1, POISON_TOTAL_MOD + 1))
    res = _sweep_one(model, base_enc, pad_id, sweep)
    poisons = sorted(p for p, (bad, _, _) in res.items() if bad)
    worst = max(res.values(), key=lambda r: r[1])
    print(f"{len(poisons)}/{len(sweep)} pads poisoned "
          f"{poisons or 'NONE'}; worst max|dLogit|={worst[1]:.3f}")
    if poisons:
        print("FAIL: a length ≡ 1 mod 32 must left-pad with no filler")
        return len(poisons)
    print("CLEAN: every pad of the natural residue length stayed "
          "inside bf16 wobble")
    return 0


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
        description="Left-pad regression: residue widths must match solo "
                    "(see module docstring)",
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
        failures += test1_pad_vs_total(model, picked, pad_id)
    if "2" in want:
        failures += test2_content(model, prompts, picked, pad_id)
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
