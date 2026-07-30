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
| 6 | `python -m training.selftest t6` | minutes | equal-length identical-prompt true GPU batch vs solo; variable-length via length cohorts vs solo (Gemma 4: no left-pad decode) |
| 7 | `python -m training.selftest t7` | minutes | t5's traces through `GameTraceSource` + `AnalystTraceSource` and real train steps (RL weights + KD-vs-base analyst anchor, mixed CE/KD buckets, per-run noised frames, finite losses; the tiny epoch is drained so the KD bucket is guaranteed to train) |
| 8 | `python -m training.selftest t8` | ~5–15 min | REAL serial datagen timing: the tiny shared workload (2 games × 3 moves, `--parallel 1`) through the actual session harness with the model pre-loaded/warmed (startup excluded); reports `seconds_per_generation` and the serial wall-clock a default epoch (3000 gens) implies. Its `generation_stats.json` is t9's baseline |
| 9 | `python -m training.selftest t9` | ~5–15 min | t8's EXACT workload at `--parallel 2` (run t8 first — its stats file is the serial baseline), compared on `seconds_per_generation`; asserts parallel is not ≫ slower, REPORTS the speedup |

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
judge. Gemma 4 Unified: left-padding a multimodal row corrupts its prefill
at SPECIFIC pad lengths — `scripts/gemma4_pad_batch_repro.py` measured
pad=7 on a 282-token image prompt giving ~40-logit deltas with the argmax
flipping to `<audio|>`, while pads 1–6, 8, 9, 15, 16, 63, 64 stay within
~0.5-logit bf16 wobble (batch size and transformers 5.13/5.14 version
irrelevant; bit-identical). Upstream transformers bug
(https://github.com/huggingface/transformers/issues/47651), not collation:
every aux tensor was verified suffix-identical to solo. Since poisonous
offsets are unpredictable, `generate_batch` runs one true GPU batch per
distinct prompt length (zero pad); rows whose length is unique decode
alone. Equal-length early divergence is a true-batch bug.

Stage 8/9 note: these two stages measure, they mostly don't judge — the
decision they feed is "is serial datagen fast enough for an overnight
epoch, or do we need real batching (server / upstream fix)?". Both time
the REAL harness (no synthetic prompts), so t8's `s/gen` already includes
prompt building, NAMS retrieval, image noise, and the analyst call,
amortized per player generation — the unit `--max-generations` caps. Read
t8's epoch extrapolation first; then t9's speedup: x~1.0 means the parallel
sessions never landed equal-length prompts in the same dispatch window
(decode fully serialized — expected with divergent game histories), and
anything ≥ x1.3 means cohorts DO form on this workload. The only assertion
is that `--parallel 2` is not ≥1.6x SLOWER per generation than serial,
which would indicate dispatcher pathology rather than absent cohorts.
Both stages share t9's caveat: sampled replies make single runs noisy —
treat ±15% as measurement error, rerun before drawing conclusions from
small differences.

Weekend rehearsal (NEW, before launching `training/run_weekend.py` for
real): with NAMS up and external data downloaded, run

    python -m training.run_weekend --prefix smoke --epochs 2 \
        --games 2 --max-generations 8 --train-max-steps 25

(~30–60 min). Verify: (1) it runs datagen → train → datagen → train with
each stage in its own subprocess; (2) epoch 2's datagen logs
`--checkpoint smoke_iter1_step<N>` — the checkpoint hand-off is the whole
point; (3) `data_game/smoke_state.json` records the stages, and re-running
the same command immediately exits after "already complete" skips; (4)
checkpoints exist under `weights/<arch>/smoke_iter1_step*` and
`smoke_iter2_step*`. Then delete `data_game/smoke_iter*`,
`data_game/smoke_state.json`, and `weights/<arch>/smoke_iter*` before the
real run. Kill-resume is worth one extra check if time permits: Ctrl-C
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
changes rather than every run:

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
