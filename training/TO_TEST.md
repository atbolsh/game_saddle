# The remote verification protocol

Nothing in `training/` has run on this box until you run it — the local
machine only edits code (`.cursor/rules/remote-environment.mdc`). The test
suite is `training/selftest.py`: numbered stages, one command each, each
printing exactly one `TEST <id> PASS/FAIL: <evidence>` line. Run the
commands below **in order** (later stages consume earlier stages' output)
as soon as `scripts/setup_env.sh` has finished, and paste the `TEST ...`
lines back for review. On a FAIL, also paste the traceback that precedes it.

| # | Command | Cost | What it proves |
|---|---------|------|----------------|
| 0 | `python -m training.selftest t0` | seconds | imports; transformers >= 5.10; CUDA visible; bitsandbytes loads |
| 1 | `python -m training.selftest t1` | seconds | pure units: `parse_rating` (incl. bold variants), `build_span_weights` (WRONG override, win boost, negative scale), image-noise determinism/identity, analyst-leak tripwire (incl. multi-line), `_rewrite_image_urls`, `GameTraceSource` on a fabricated trace dir, micro-batch bucketing, planted-error scrambler (seed determinism, all three move-token modes, clock-shift tolerance labeling, direction swap incl. the "right move" guard, inert-text fallthrough), perception-question sampling (rate honored, groups and mirrored variants balanced), `AnalystTraceSource` on a fabricated file (KD loss kind, rating-null kept, unverified-span record dropped, quota weight), `stack_equal_length` (equal-length rows stack, mixed lengths hard-error) |
| 2 | `python -m training.selftest t2` | seconds | every enabled manifest entry materialized; `data.jsonl` row counts match `meta.json`; probe files present |
| 3 | `python -m training.selftest t3` | minutes | 4-bit QLoRA load; LoRA target discovery + projector resolution; terminator id; one collated forward+backward per loss kind (image example included); fresh-adapter KD loss equals teacher entropy (the `disable_adapter()` teacher path) |
| 4 | `python -m training.selftest t4` | ~15–30 min | batch-4 vs batch-1 per-example loss parity (mixed CE/KD/image/negative-span buckets); CLI smoke train lands a checkpoint + `eval_log.jsonl` rows; destructive-LR variant fires the rollback path |
| 5 | `python -m training.selftest t5` | ~10–20 min | datagen 2 games x 5 moves at `--parallel 2`: traces + stored frames + stats + plots written, one record per generation, tripwire silent, ratings parsed; `analyst_traces.jsonl` has one record per round (minus counted truncated-search skips), analyses nonempty, frames shared with player records |
| 6 | `python -m training.selftest t6` | minutes | equal-length identical-prompt true GPU batch vs solo; variable-length via the VERIFIED LEFT-PAD workaround vs solo (must byte-match AND must actually take the padded path, not the cohort fallback) |
| 7 | `python -m training.selftest t7` | minutes | t5's traces through `GameTraceSource` + `AnalystTraceSource` and real train steps (RL weights + KD-vs-base analyst anchor, mixed CE/KD buckets, per-run noised frames, finite losses; the tiny epoch is drained so the KD bucket is guaranteed to train) |
| 8 | `python -m training.selftest t8` | ~15–40 min | REAL serial datagen timing: the shared workload (3 games × 4 moves, `--parallel 1`) through the actual session harness with the model pre-loaded/warmed (startup excluded); reports `seconds_per_generation` and the serial wall-clock a default epoch (3000 gens) implies. Its `generation_stats.json` is t9's baseline (t9 checks the game count and refuses a stale one) |
| 9 | `python -m training.selftest t9` | ~10–25 min | t8's EXACT workload at `--parallel 3` (run t8 first — its stats file is the serial baseline), compared on `seconds_per_generation`; asserts parallel is not ≫ slower, REPORTS the speedup (expect a real one now: phase-locking dispatcher + verified left-pad batching) |
| 10 | `python -m training.selftest t10` | ~20–30 min | 4 timed train micro-batches from EVERY loss category (player CE/RL, analyst-anchor KD incl. teacher forward, each manifest source) on the real overnight corpus (`data_game/overnight_iter1`, override `T10_DATAGEN_LABEL`); per-category peak VRAM (asserts < 88 GiB — the 2026-07-31 OOM tripwire) and a whole-epoch train-time estimate; saves NO checkpoint |

(`python -m training.selftest all` runs everything in order; the exit code
is the number of failures.)

To wipe selftest leftovers (failed / Ctrl-C mid-run) and re-start cleanly::

    bash scripts/clean_selftest.sh
    python -m training.selftest all   # or resume from the failed stage

That script only removes `selftest_*` paths under `data_game/`, `logs/`,
`weights/`, and `/tmp`; it never touches `data_external/` or Neo4j. If a
GPU stage was interrupted, also check `nvidia-smi` for a leftover python
PID holding VRAM.

Stage 6 note: greedy equality is the strict criterion. If it fails only on
a near-tie token deep into a reply (both outputs sane, divergence late),
that is bf16 batched-matmul nondeterminism — paste both replies and we
judge. KNOWN TRANSFORMERS BUG WORKAROUND in play (full banner in
`agent/model.py`): Gemma 4 Unified corrupts a left-padded multimodal
prefill in two measured modes — (1) padded TOTAL width ≡ 1 mod 32
(~35–52-logit deltas, argmax flipping to junk like `<audio|>`; 6/6
across prompt lengths 282–2867, `training/probe_hacky_pads.py`), and
(2) a row whose OWN length ≡ 1 mod 32 corrupts under ANY left pad
(found by the 2026-07-30 t6 run: L=289 rejected at every T in 290–297,
~24.8-logit deltas). Standalone repro `scripts/gemma4_pad_batch_repro.py`;
upstream https://github.com/huggingface/transformers/issues/47651 and
https://huggingface.co/google/gemma-4-12B-it/discussions/50. The poison
is prefill-only — decode steps crossing the residue are harmless (probe
test 4). Probe test 5 (2026-07-30) measured a natural-residue prompt
poisoned at 32/32 pads and validated the rescue: one harmless filler
token (" .") moves it off the residue and it pads cleanly, with a
content-identical greedy reply. So `generate_batch` NUDGES unpaddable
rows off the residue (POISON MODE 2 RESCUE — serving-stack-only, traces
keep the un-nudged prompt; ~1/32 of mixed-length rows), left-pads to the
longest length (never ≡ 1 mod 32 by construction — mode 1 dodged), and
parity-checks each padded row's prefill against its solo prefill before
decoding; rows the nudge cannot move (WARNING) or that the parity check
rejects (an UNCATALOGUED third mode — WARNING) decode via cohorts. t6
picks its prompt lengths at runtime around this arithmetic, compares the
nudged row against the NUDGED prompt's solo reply, and FAILS if the
padded path or the rescue did not engage: a silent cohort fallback would
pass equality while quietly serializing parallel datagen. Equal-length
early divergence is a true-batch bug.

Stage 8/9 note: these two stages measure, they mostly don't judge — the
decision they feed is "is serial datagen fast enough for an overnight
epoch, or do we need real batching (server / upstream fix)?". Both time
the REAL harness (no synthetic prompts), so t8's `s/gen` already includes
prompt building, NAMS retrieval, image noise, and the analyst call,
amortized per player generation — the unit `--max-generations` caps. Read
t8's epoch extrapolation first; then t9's speedup. Two mechanisms have to
cooperate for a real speedup: the verified left-pad workaround (stage 6
note) lets mixed-length rows decode as ONE padded batch, and the
PHASE-LOCKING dispatcher (agent/parallel_gen.py) gets compatible requests
to be concurrent in the first place — player and analyst generations have
different stop signatures and can never batch with each other, so with
the old fixed 50 ms window two sessions drifted into stable anti-phase
and measured x1.18; the scheduler now holds requests until every live
worker has submitted, then serves the smallest signature group first so
the sessions re-align (one solo generation of re-sync cost, then players
batch with players and analysts with analysts). Diagnosis order for a low
number: (1) `dispatch: group of N [reason]` lines in the run log — N
should reach the worker count with reason `full-batch` after the first
round or two; persistent `phase-lock`/size-1 groups mean the sessions
never aligned; (2) `batch_mode` in `llm_calls.jsonl` — `padded(...)`
entries confirm the stage-6 workaround engaged; (3) frequent
`hold-timeout` reasons mean a worker keeps stalling >120 s in non-GPU
work (NAMS?). The only assertion is that `--parallel 3` is not ≥1.6x
SLOWER per generation than serial, which would indicate dispatcher
pathology. NOTE for this rerun: `prefill_last_logits` now passes
`logits_to_keep=1` (kwarg assumed present in transformers 5.14 — a
rename fails loudly as TypeError in t6); without it the parity check
transiently materialized multi-GB full-vocab logits, which is what would
have made `--parallel 3` VRAM-tight. With it, batch 3 adds only ~0.5 GB
of KV cache per extra row at game-size contexts.

KV reuse: attempted, REVERTED — do not retry. The "obvious" optimization
of handing the parity check's KV cache to `generate` (skipping the
second prefill, ~10% of round time) fails on two hard transformers 5.14
facts, both observed on the remote box 2026-07-30: (1)
`DynamicSlidingWindowLayer.crop` raises ValueError once a sliding layer
has seen more tokens than its window, so "crop the cache by one so
generate has an uncached token" is impossible at game-size prompts; (2)
Gemma 4 Unified's `generate` passes `pixel_values` into its first
forward even when a cache is supplied, and the model hard-errors with
"Image features and image tokens do not match, tokens: 0, features:
768" when the remaining input has no image tokens. A workaround exists
(prefill T-1 tokens, deepcopy the cache for the parity step, strip
`pixel_values`, shift the mod-32 rule to T-1) but was judged not worth
the complexity for ~10%. The parity forward now runs `use_cache=False`
and the padded batch is prefilled twice, deliberately.

Both stages share t9's caveat: sampled replies make single runs noisy —
treat ±15% as measurement error, rerun before drawing conclusions from
small differences.

Stage 10 note: t10 exists because training OOM'd three times on
2026-07-31 (the overnight run, then t10 itself twice — each crash exposed
a different memory bug). `weighted_loss` is now written around four VRAM
rules, spelled out in its docstring; the short version: (1) both forwards
pass `logits_to_keep=tail+1` so full-sequence `[batch, seq, 262k-vocab]`
logit tensors are never materialized (the cut happens inside the model —
slicing in loss code is too late); (2) for KD the teacher forward runs
BEFORE the student and is reduced to probabilities first, so the two
sides' multi-GiB tensors never coexist; (3) weighted positions are
gathered by 2-D index, never `[:, :-1, :].reshape(...)` — that reshape
silently copies the whole non-contiguous tensor; (4) sources whose target
spans nearly the whole sequence (openthoughts) get `micro_batch_cap` in
`datasets.json` (→ `TrainingExample.batch_cap` → `epoch_batches`), which
per-example loss normalization makes mathematically free. Two supporting
pieces: `train.py` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
at import (variable tail sizes fragment the default allocator — one crash
had 14 GiB stranded as reserved-but-unallocated; an explicit env setting
wins over the setdefault), and `weighted_loss` logs a WARNING whenever a
batch's kept-logit tensor would exceed ~10 GiB — the early alarm for a
new long-target source or a bad char-based length bucket.

**Verified 2026-07-31 18:05** on the overnight corpus: all 12 sources
pass, worst peak 39.1 GiB (analyst KD; tripwire 88), epoch estimate ~2.0h
plus setup. Reading t10: it times 4 micro-batches per source, so it
samples rather than proves — the single worst batch of a real epoch can
run somewhat hotter than the printed peaks (bounded by
`max_example_chars` and the 1.5x length buckets). The epoch estimate
excludes save-time eval hooks (~3–5 min per save) and bucket-remainder
short batches; pad it ~10–15% when fitting the weekend window. After ANY
change to `weighted_loss` or collation, rerun t3 and t4 first (the loss
correctness tests: fresh-adapter KD equals teacher entropy; batch-4 vs
batch-1 parity), then t10. A `logits_to_keep` rename in a future
transformers fails loudly as TypeError in t3/t4/t10 — never silently
fall back to full-sequence logits.

Weekend rehearsal (before launching `training/run_weekend.py` for
real): with NAMS up and external data downloaded, run

    python -m training.run_weekend --prefix smoke --epochs 2 \
        --games 2 --max-generations 8 --train-max-steps 25

(~30–60 min). Verify: (1) it runs datagen → train → datagen → train with
each stage in its own subprocess; (2) epoch 2's datagen logs
`--checkpoint smoke_iter1_step<N>` — the checkpoint hand-off is the whole
point; (3) `data_game/smoke_state.json` records the stages, and re-running
the same command immediately exits after "already complete" skips; (4)
checkpoints exist under `weights/<arch>/smoke_iter1_step*` and
`smoke_iter2_step*`; (5) `data_game/smoke_vram.jsonl` has stage-tagged
samples from BOTH stage kinds and the log ends with a `VRAM summary`
line (peak + per-stage means) and a `stage time:` line — that's the
monitoring you'll read Monday morning. Then delete `data_game/smoke_iter*`,
`data_game/smoke_state.json`, `data_game/smoke_vram.jsonl`, and
`weights/<arch>/smoke_iter*` before the real run. Kill-resume is worth one extra check if time permits: Ctrl-C
mid-datagen, rerun, and confirm it resumes with `--append` and the
remaining budget.

Stage 4 note: the rollback variant relies on `--lr 0.05` wrecking the
adapter between saves, which is near-certain but stochastic; if no
`rolled_back` event fires, rerun once before treating it as a bug. The
batch-4 vs batch-1 loss parity check allows up to ~10% relative (or 0.05
absolute) deviation -- bf16/SDPA and Gemma 4 multimodal padding are not
bit-identical across shapes; treat order-of-magnitude gaps as a real bug.

## Manual appendix (not automatable)

These need eyes or box-level state; check them once per box / after big
changes rather than every run. For anything that means reading recorded
datagen traces (noised frames, win stamping, question-round analyses),
`notebooks/trace_viewer.ipynb` steps through any `data_game/<label>` run
game-by-game, move-by-move — no GPU or NAMS needed:

1. **Checkpoint dropdowns** — the smoke checkpoint from t4 appears in the
   notebooks' Checkpoint dropdown, loads and generates via the picker, via
   `MODEL_CHECKPOINT` in `.env`, and via
   `python -m agent.runner --checkpoint <name> game ...`; `[default]` still
   loads bare HF weights.
2. **Disk-mode settings** — `scripts/setup_env.sh` writes
   `data_external/settings.json` (`download_mode`) correctly for the box's
   free disk and preserves an existing file on re-run. On a disk-tight box:
   `python -m training.download_external --force --mode stream --only
   cauldron_vqav2` stops after `max_rows` without pulling the full 13.5 GB,
   and `meta.json` counts match full mode.
3. **Noised frames readable** — open a few `data_game/<label>/images/`
   frames from t5 (noise on by default): visibly degraded but agent, gold,
   and walls clearly distinguishable. If not, lower `INFERENCE_STRENGTH`
   in `training/image_noise.py`. Also spot one per-run training copy from
   t7's temp dir (path in the run log) at `TRAINING_STRENGTH`.
4. **Cross-session leak grep** — the tripwire covers current-session leaks
   automatically; once per big prompt/memory change, also grep
   `data_game/<label>/traces.jsonl` for distinctive phrases from analyst
   analyses in the datagen run logs (`logs/datagen_*/llm_calls.txt`) to
   confirm nothing arrives through NAMS retrieval either.
5. **NAMS reset cadence** — with `--reset-every 1 --games 3`, node counts
   between games drop back to the seeded semantic model while `Preference`
   (tip) nodes survive; datagen continues cleanly after each reset.
6. **Win stamping** — when a game is actually won (shrink the board via
   config if needed), its records carry `game_won: true` and
   `moves_from_end` counting down to 0.
7. **Probe sanity before training** — `exact_match/gsm8k` and
   `exact_match/navigation` accuracies from t4's eval rows are well above 0
   on the base model (the model emits parseable `ANSWER:` lines), and the
   probe pass stays within ~3–5 min per save.
8. **Datagen stats plots** — open `logs/datagen_stats_<label>_*/summary.png`
   after every real datagen run: a rating histogram compressed into a
   narrow positive band with rare WRONG spans means a weak reward signal —
   reconsider before spending the training hours.
9. **Question-round analyses** — from a t5 (or real) datagen run, pull a
   few records whose `meta.question` is a perception question and read the
   analyst analyses in the run log: prose answers with no move token are
   graded on correctness (not punished for the missing token), and a reply
   that DID emit a move token on a question round gets the unrequested
   token called out.
