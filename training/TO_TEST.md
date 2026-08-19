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
| 0 | `python -m training.selftest t0` | seconds | imports; transformers >= 5.10; CUDA visible; bitsandbytes loads; NAMS `add_preference` writes a throwaway Preference (newlines, `[FORWARD]`, `[ANALYST]` line) and reads it back byte-identical, then deletes it |
| 1 | `python -m training.selftest t1` | seconds | pure units: `parse_rating` (incl. bold variants), the 2026-08-05 SHAPE/SCALE reward split — `rating_advantage` (exp-advantage vs a baseline: 1.0 at the mean, ~2x per +0.2, hard floor 0 at r ≤ -0.5, cap `ADV_CAP` 3.0) and `example_scale` (ADDITIVE win boost `1.0 * 0.95^d`, rescues a floored reply, exact-0 floor otherwise) on the SCALE side; `build_span_weights` on the SHAPE side (base 1.0, WRONG spans at −0.5 = bounded unlikelihood, oracle move-token modifier LAST and only on correct/wrong verdicts); `oracle_verdict` geometry (12 cases: exact match, either-turn ≥170°, in-cone (20°) FORWARD neutral, 28.6° FORWARD wrong (the 2026-08-13 cone shrink), ray-hit turns WRONG in both directions (the 2026-08-11 missed-forward tightening), out-of-cone/away-turn wrong, unknown on missing meta or perception rounds) and `_oracle_meta` raw facts from fabricated settings (compass convention, nearest-gold bearing, any-gold ray hit, empty when won); `action_balance_multipliers` (TEMPORARY hack: inverse-frequency mean/count per action, mass-preserving, cap 4x both ways on degenerate mixes), `NoveltyTracker` (WIP boredom decay: 0.9^k, floor 0.1, reset on a different move, perception rounds skipped WITHOUT resetting, per-game keys) standalone AND through `GameTraceSource` with `novelty=True` (OFF by default — a default source must NOT decay); `GameTraceSource` integration on fabricated corpora: span weights stay pure shape while `example_weight` carries exp-advantage × balance × novelty × oracle penalty × ray-hit `TRANSITION_BOOST` (r_bar pre-pass asserted, oracle-wrong record gets ×0.25 + a −0.5 move span, missed-forward record gets ×0.25 × 2.0 + the −0.5 span, ray-hit FORWARD gets ×2.0, floored records SKIPPED, rating-null DROPPED), `MetricGuard` ceiling-only mode (`relative=False`) and `warn_only`, image-noise determinism/identity + the 10% clean pass-through (`_SKIP_PROB`), analyst-leak tripwire (incl. multi-line), `_rewrite_image_urls`, `PlayerAnchorSource` (kd_anchor loss, rating-null KEPT, uniform weights, 0.25 weight, ceiling-only guard at 1.0), micro-batch bucketing, planted-error scrambler, perception-question sampling, `AnalystTraceSource` (KD loss kind, rating-null kept, unverified-span record dropped, quota weight, ceiling-only guard at 5.0), `stack_equal_length`; `run_weekend --checkpoint` (rejects `--start-checkpoint` / `--resume-checkpoint`); 2026-08-12: `boundary_openings` (full walls → empty, fabricated right-wall gap, no walls → 4 sides, rotated boundary wall → ValueError), `new_multi_gold_game` (n_gold=0/2/3, opening require/forbid), `[END_GAME]` parser on the multi-gold subclass AND the base class; universal `openings` on every `settings_to_dict`; `oracle_verdict` END_GAME → unknown; `action_balance_multipliers` pins END_GAME at 1.0; unified prompt composition (`PICK ONE TARGET AND COMMIT` present, multi-move wording gone, debrief uses `RATING:` not `overall_score`); 2026-08-14: `parse_remember_notes` (order, overwrite-in-order, case-fold, malformed ignored, value-to-`]`, strip), truncation+parse composition, `board_update_line` (eaten / not-eaten), `format_notepad` (empty / populated) |
| 2 | `python -m training.selftest t2` | seconds | every enabled manifest entry materialized; `data.jsonl` row counts match `meta.json`; probe files present |
| 3 | `python -m training.selftest t3` | minutes | 4-bit QLoRA load; LoRA target discovery + projector resolution; terminator id; one collated forward+backward per loss kind (image example included); fresh-adapter KD loss equals teacher entropy (the `disable_adapter()` teacher path); kd_anchor with NO anchor adapter loaded equals kd (the epoch-1 base fallback); `example_weight` ×0.1 scales the loss by EXACTLY 0.1 (the 2026-08-05 shape/scale regression — the old code normalized reply-wide reward away); the negative-span smoke example yields a FINITE NON-NEGATIVE loss (bounded unlikelihood, not negative CE); `kd` with negative span weights raises ValueError |
| 4 | `python -m training.selftest t4` | ~15–30 min | batch-4 vs batch-1 per-example loss parity (mixed CE/KD/image/negative-span buckets); CLI smoke train lands a checkpoint + `eval_log.jsonl` rows (INCLUDING the new step-0 baseline eval); destructive-LR variant fires the rollback path via the HARD tier (`--hard-multiplier 1.0` pins any regression to hard) AND must now END EARLY on the second consecutive rollback (2026-08-05): asserts a `consecutive_rollback_stop` event, exit 0, and a `done` event carrying `ended_early` + a usable `last_good_checkpoint` (the orchestrator hand-off contract) |
| 5 | `python -m training.selftest t5` | ~10–20 min | datagen 2 games x 5 moves at `--parallel 2`: traces + stored frames + stats + plots written, one record per generation, tripwire silent, ratings parsed; `analyst_traces.jsonl` has one record per round (minus counted truncated-search skips), analyses nonempty, frames shared with player records |
| 6 | `python -m training.selftest t6` | minutes | equal-length identical-prompt true GPU batch vs solo; variable-length via the VERIFIED LEFT-PAD workaround vs solo (must byte-match AND must actually take the padded path, not the cohort fallback) |
| 7 | `python -m training.selftest t7` | minutes | t5's traces through `GameTraceSource` + `PlayerAnchorSource` + `AnalystTraceSource` and real train steps (RL weights + player trust region + KD-vs-base analyst anchor, mixed ce/kd/kd_anchor buckets, per-run noised frames, finite losses; the tiny epoch is drained so every KD bucket is guaranteed to train) |
| 8 | `python -m training.selftest t8` | ~15–40 min | REAL serial datagen timing: the shared workload (3 games × 4 moves, `--parallel 1`) through the actual session harness with the model pre-loaded/warmed (startup excluded); reports `seconds_per_generation` and the serial wall-clock a default epoch (3000 gens) implies. Its `generation_stats.json` is t9's baseline (t9 checks the game count and refuses a stale one) |
| 9 | `python -m training.selftest t9` | ~10–25 min | t8's EXACT workload at `--parallel 3` (run t8 first — its stats file is the serial baseline), compared on `seconds_per_generation`; asserts parallel is not ≫ slower, REPORTS the speedup (expect a real one now: phase-locking dispatcher + verified left-pad batching) |
| 10 | `python -m training.selftest t10` | ~20–30 min | 4 timed train micro-batches from EVERY loss category (player CE/RL, player-anchor kd_anchor, analyst-anchor KD — both incl. the teacher forward — each manifest source) on the real overnight corpus (`data_game/overnight_iter1`, override `T10_DATAGEN_LABEL`); per-category peak VRAM (asserts < 88 GiB — the 2026-07-31 OOM tripwire) and a whole-epoch train-time estimate; saves NO checkpoint |

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

Collapse-proofing note (2026-08-03, after the weekend collapse — full
postmortem in TRAINING_GAME_TRACES.md): the retest run carries five
interlocking changes, most covered by the stages above (t1 = mapping +
anchor source units, t3 = kd_anchor fallback, t4 = baseline eval + hard
rollback, t7 = kd_anchor through real steps). Two things the selftests
canNOT prove and the retest run itself must:

* **The anchor adapter on a REAL parent checkpoint.** t3/t7 only exercise
  the no-anchor base fallback. Epoch 2 of the retest is the first time
  `model.load_adapter(<parent ckpt>, adapter_name="anchor",
  is_trainable=False)` + `set_adapter("anchor")` runs with
  `modules_to_save` (the embed_vision projector copy) in play — a PEFT
  API assumption verified only from docs, not on the box (remote-
  environment rule). It fails loudly if wrong; if train2 crashes at
  startup around `anchor_loaded`, this is the suspect. A quick manual
  check on the box: watch train2's log for the `anchor_loaded` event and
  a normal first-step loss on the `player_anchor_*` source (~teacher
  entropy, low single digits — NOT ~0: the student has drifted from the
  parent by then).
* **The fuse + stop-on-poison end to end** only fires against an actually
  collapsed checkpoint; it was designed against the weekend logs. If the
  retest stays healthy the fuse should remain silent
  (`degenerate_generations` ≈ 0 in generation_stats.json).

Spin-bot follow-ups (2026-08-04, after the retest run — full story in
TRAINING_GAME_TRACES.md "The 2026-08-04 spin-bot"): four changes, three
of them covered by a t1 RERUN (pure python, seconds — the mandatory
verification for this batch):

* **Ceiling-only anchor guards** (`MetricGuard.relative=False`,
  `guard_relative` on both anchor sources; player ceiling 1.0, analyst
  5.0). The retest's train1 hard-rolled back to base TWICE because the
  player-anchor drift meter was guarded against its own step-0 floor.
  t1 units cover the guard logic and the source attributes; the next
  real train run should show NO `HARD REGRESSION on
  heldout_loss/player_anchor_*` at healthy drift (~0.1–0.3 nats/token
  above the step-0 baseline is normal).
* **Novelty decay + action balance** (t1 units, standalone and through
  `GameTraceSource`). Morning check: every datagen-source load now logs
  its action-balance multipliers at INFO (`game_<dir>: action balance
  multipliers {...}`) — a WARNING there means the cap engaged on a
  degenerate corpus mix. Watch the next run's smoke evals for the action
  mix rebalancing (the retest degraded to 6.5% FORWARD / 69% ANTICLOCK);
  the balance is a marked TEMPORARY hack (equal mass per move type is
  only defensible in this 3-move game).
* **Checkpoint hand-off honors the trainer's verdict**: `run_weekend`
  now reads `last_good_checkpoint` from the train stage's `done` event
  (newest `logs/train_<label>_*/events.jsonl`) instead of taking the
  highest-step directory — the retest promoted a checkpoint its own
  final eval had just rejected. NOT covered by any selftest stage (it
  needs a full orchestrated epoch): verify during the weekend rehearsal
  below — the log line is now `[trainN] checkpoint (last good per the
  trainer's done event): ...`, and after a run whose FINAL eval
  hard-regresses, the next epoch must resume from the earlier good save,
  not the newest directory under weights/.

Aug4 guard follow-ups (2026-08-05, after the aug4 overnight run — the
rollback loop postmortem is in TRAINING_OVERVIEW.md's rollback section):
three changes, covered by t1 + t4 RERUNS:

* **Warn-only KD replay guards** (`MetricGuard.warn_only`,
  `DataSource.guard_warn_only`; set automatically on every `loss: "kd"`
  manifest entry). The aug4 run's rollbacks were fired by
  cauldron/openthoughts relative guards — drift meters pinned to their
  step-0 entropy floor, same disease as the anchor guards fixed
  2026-08-04. Detection and reference are unchanged (drift vs. base);
  a breach now logs `DRIFT WARNING` + a `drift_warning` event and never
  rolls back. t1 units cover the flag; on the next real run grep for
  `DRIFT WARNING` — expected at moderate drift — and there must be NO
  `HARD REGRESSION on heldout_loss/cauldron_*|slimorca|openthoughts`.
* **Consecutive-rollback early stop**: two rollbacks at consecutive
  evals end the run loudly (ERROR log + `consecutive_rollback_stop`
  event + early `done` with `ended_early` and `last_good_checkpoint`,
  exit 0). t4's destructive-LR variant now asserts this path.
* **navigation ×3** (`examples_per_epoch` 300 → 900, ~14% of epoch
  gradient mass): the aug4 runs degraded navigation held-out loss
  0.054 → 0.175 nats — its guard stays STRICT by design, so the defense
  is corpus weight. No selftest coverage (manifest quotas aren't unit
  tested); on the next run watch `heldout_loss/navigation` stay flat.

Anti-self-cloning rework follow-ups (2026-08-05 evening — full design in
TRAINING_GAME_TRACES.md "The reward scheme" + the train.py SHAPE VS SCALE
docstring): the reward now splits into span SHAPE × `example_weight`
SCALE, negative weights are bounded unlikelihood, the engine oracle
grades move tokens, and the LR dropped to RL scale (3e-6, clip 0.1).
Verification order:

* **Rerun t1 and t3 first** (this batch rewrote both), then **t4** (loss
  parity — `weighted_loss` changed) and **t10** (per the standing rule:
  any `weighted_loss`/collation change reruns t3+t4 before t10). t7 is a
  cheap extra confidence pass (kd_anchor through real steps).
* **Datagen stamps oracle meta now** — after the first real datagen, spot
  a `traces.jsonl` record for `oracle_move`/`oracle_rel_bearing`/
  `oracle_ray_hit`, and check the source-load log line
  `game_<dir>: oracle verdicts {...}`: on a healthy corpus "unknown"
  should be ~the perception-round count (old corpora are all-unknown by
  design and log a one-time INFO).
* **Reward contrast greps on the next train run:** the source-load INFO
  line now logs `rating baseline (r_bar)`; the per-record scales are in
  each example's meta only, so the honest check is behavioral — the
  smoke evals' action mix and win rate, not offline loss. Expect
  `DRIFT WARNING`s to stay rare at 3e-6 (the aug4 runs drifted at 1e-4;
  if KD replay drift warnings vanish entirely AND game metrics move,
  the LR drop did its job).
* **Floored/zero-scale records** log `SKIPPED n/m record(s) at zero
  example weight` at INFO — a large fraction there means the analyst
  floored much of the corpus (read the rating histogram before burning
  the train hours).

Run-hygiene follow-ups (2026-08-06, before the 11-epoch run): two
orchestration-level changes, NOT covered by any selftest stage (both need
a real epoch boundary):

* **Run-start NAMS reset** (`generate_game_traces.run_generation`): every
  non-`--append` datagen run now resets episodic memory to the seeded
  semantic model before the first game (tips survive; the block-boundary
  reset at `--reset-every` never fired on 60-game orchestrated epochs, so
  memory grew across the whole weekend). Verify: each epoch's datagen log
  opens with `NAMS hygiene: run-start reset ...`; an `--append` crash
  resume must NOT log it. Note this also means every fresh datagen —
  including t5/t8/t9 and smoke evals — wipes episodic memory at start;
  that is intended (episodic memory is disposable by design).
* **Reset now purges EXTRACTED entities too** (2026-08-11, found by the
  post-aug6 census: 33k extraction-minted `Entity` nodes vs 5 seed ones
  had survived every reset). `_reset_memory_to_seed` keeps only the
  `_SEMANTIC_MODEL_ENTITIES` (by name) + all `Preference` nodes, and both
  datagen reset sites now log the per-label deletion census. Verify on
  the box BEFORE trusting it: `MATCH (e:Entity) WHERE e.name IN ['Agent',
  'Gold','BoundaryWall','DiscreteGame','Direction'] RETURN e.name;` must
  return all 5 (if NAMS entity resolution renamed them, the name match is
  wrong — flag it). After the next run's first reset, `MATCH (e:Entity)
  RETURN count(*)` should be ~5, and reset censuses in the datagen log
  should stay ~one stage's worth (growth across epochs = a skipped
  reset).
* **Post-train corpus pruning** (`run_weekend._prune_datagen`): after each
  SUCCESSFUL train stage, `data_game/<prefix>_iter<k>` is tombstoned down
  to one won game (if any) + one other random game; all other records in
  both jsonl files become `{"pruned": true, "meta": ...}` and their frames
  are deleted. Stats stay computable (`_summarize_traces` reads only
  `meta`); a pruned corpus fed back to `GameTraceSource` fails loudly on
  the missing `messages` key. Verify after epoch 1: `pruned` appears in
  the state file with a sane `mib_freed`, `images/` shrank to the kept
  games' frames, and `grep -c '"pruned": true' traces.jsonl` ≈ records −
  kept. Smoke dirs and failed epochs' corpora are never pruned.
* **Noised-frame temp dir cleanup** (2026-08-11, after the aug6 run
  leaked ~65 GiB into /tmp): `game_traces._make_noise_dir` registers
  every noise dir for deletion at interpreter exit, and the orchestrator
  sweeps stale `*_noise_*` dirs at each epoch boundary (hard-killed
  stages). Verify: after any train stage exits, `/tmp` has no
  `game_*_noise_*` / `player_anchor_*_noise_*` / `analyst_*_noise_*`
  dirs left; the sweep logs `swept N stale noised-frame temp dir(s)`
  when it finds crash leftovers.

Transition reward + analyst tightening (2026-08-11 evening, before the
aug11 2-epoch run — design note in TRAINING_GAME_TRACES.md's oracle
section): oracle ray-hit turns → "wrong", `TRANSITION_BOOST` ×2 on
ray-hit rounds, novelty decay ON at the `run_weekend` call site, and the
shared `_BLOCK_GRADING_TOLERANCE` analyst block gained explicit
must-be-negative rules. All source-side; `weighted_loss` and collation
are untouched, so the standing t3/t4 rule does NOT trigger:

* **Rerun t1 only** (seconds — oracle table, transition-boost
  assertions through `GameTraceSource`, novelty units). t7 is an
  optional cheap end-to-end pass.
* **On the run itself:** epoch 1's datagen plays with the old
  checkpoint but is graded under the new analyst prompt and trained
  under the new reward; epoch 2's smoke is the first behavioral read.
  Grep the smoke stats for FORWARD-on-ray-hit compliance (the aug6 run
  sat at ~50%) and check the rating histogram — the tightened prompt
  should push missed-forward ratings negative rather than the old
  42%-at-≥+0.8 coin flip. Novelty decay's fingerprint is shorter
  identical-turn runs in the traces.

Weekend-run tweaks (2026-08-13, before the aug13 9-epoch run): random
`--seed` unless specified (logged + `base_seed` in the state file);
aim cone 45°→20° (prompt blocks + `ORACLE_CONE_RAD`; 20–45° FORWARDs
reclassify from oracle-neutral to oracle-wrong at the next train);
navigation `examples_per_epoch` 900→450; smoke-only 400 gens and
`--question-rate 0.075`. `weighted_loss` / collation untouched, so
t3/t4 do NOT trigger.

* **Rerun t1 only** (seconds — oracle table now has an in-20° neutral
  and a 28.6° wrong). Prompt byte-identity is NOT required: the cone
  wording is an intentional prompt edit.
* **On the run itself:** omit `--seed`; grep `base seed` from the log
  and keep it. Smoke wall is ~65 min healthy / ~15 min if poisoned.

AV eval + multi-gold room mode (2026-08-12): two additive deliverables;
`weighted_loss` / collation / the existing scene prompts are untouched, so
the standing t3/t4 rule does NOT trigger.

* **Rerun t1 only** (seconds — `boundary_openings`, `new_multi_gold_game`,
  `[END_GAME]` parser). Byte-identity check for the critical path: the
  command below must print the same digests on this branch as on the
  pre-change commit (run it once under `git stash` / the old checkout for
  the "before" values). NOTE: sha256, not `hash()` — Python's str hash is
  randomized per process, so `hash()` can never match across two runs.
  SUPERSEDED 2026-08-14: `SYSTEM_PROMPT_SCENE_PLAY` was intentionally
  edited by the scratchpad work, so printed digests no longer match
  aug-12 values; "before" values must come from a pre-scratchpad commit.

      python -c "import hashlib; from agent import modes; print(hashlib.sha256(modes.SYSTEM_PROMPT_SCENE_PLAY.encode()).hexdigest(), hashlib.sha256(modes.SYSTEM_PROMPT_SCENE_ANALYST.encode()).hexdigest())"

* **AV eval script** (`python -m training.eval_av <checkpoint>`): on the
  remote box, first verify the NExT-QA HF id (`lmms-lab/NExTQA`; override
  with `--video-hf-id` if Hub renamed it), materialize with `--force` if
  needed, then a 5-item smoke (`--n-audio 5 --n-video 5`) on tomorrow's
  checkpoint before the full ~20-minute run (`--n-audio 40 --n-video 30`).
  A processor rejection of `{"type": "audio"}` / `{"type": "video"}` is a
  finding — do not route around it. Results land in
  `logs/av_eval_<checkpoint>_<UTC>/`.

Session scratchpad + Board update (2026-08-14): player `[REMEMBER key:
value]` notes, engine-truth Board update line after each applied move,
player recency window 8, analyst sees the notepad the player saw.
Follow-up: `_BLOCK_NOTEPAD` added to `SYSTEM_PROMPT_GAME` (mode-1
`InteractiveSession` + `mode_game` wiring); `_BLOCK_END_GAME` declaration
now uses the `[REMEMBER]` form. Prompt + interactive wiring only.
`weighted_loss` / collation untouched, so t3/t4 do NOT trigger.

* **Rerun t1 only** (seconds — `parse_remember_notes`, `board_update_line`,
  `format_notepad`, truncation interplay). Prompt byte-identity is NOT
  required: the notepad / target-commit / target-grading wording is an
  intentional prompt edit. No additional t1 cases for the mode-1 wiring
  or `_BLOCK_END_GAME` wording (no new pure functions).

Gold vs opening vocabulary (2026-08-14 follow-up): multi-gold player
prompts (`_BLOCK_HOW_TO_PLAY_MULTI` / aim / current-screen, target-commit,
no-gold-explore) and matching analyst blocks distinguish gold from
opening/exit and treat calling an opening "the gold" as a naming defect.
Prompt-only; `weighted_loss` / collation untouched, so t3/t4 do NOT
trigger. Prompt byte-identity is NOT required. No new t1 cases.


Weekend rehearsal (before launching `training/run_weekend.py` for
real): with NAMS up and external data downloaded, run

    python -m training.run_weekend --prefix smoke --epochs 2 \
        --games 2 --max-generations 8 --train-max-steps 25

(~30–60 min; NOTE the confusing name collision: `--prefix smoke` is the
rehearsal label, while a `smoke<k>` STAGE now also runs after every train
stage — the post-train smoke eval). To continue from an existing adapter,
`--checkpoint NAME` (not `--resume-checkpoint` — that is a `training.train`
flag and is rejected here). Verify: (1) it runs datagen → train →
smoke eval → datagen → train → smoke eval with each stage in its own
subprocess; (2) epoch 2's datagen logs `--checkpoint smoke_iter1_step<N>`
— the checkpoint hand-off is the whole point; (3)
`data_game/smoke_state.json` records the stages, a `smoke` dict with a
per-epoch performance summary (games/wins/ratings/degeneracy/distance
deltas — the same line greps as `smoke_eval` in the log), and re-running
the same command immediately exits after "already complete" skips; (4)
checkpoints exist under `weights/<arch>/smoke_iter1_step*` and
`smoke_iter2_step*` (including the new `_step0` baselines); (5)
`data_game/smoke_vram.jsonl` has stage-tagged samples from BOTH stage
kinds and the log ends with a `VRAM summary` line (peak + per-stage
means) and a `stage time:` line — that's the monitoring you'll read
Monday morning. Then delete `data_game/smoke_iter*`, `data_game/smoke_smoke*`,
`data_game/smoke_state.json`, `data_game/smoke_vram.jsonl`, and
`weights/<arch>/smoke_iter*` before the real run. Kill-resume is worth one extra check if time permits: Ctrl-C
mid-datagen, rerun, and confirm it resumes with `--append` and the
remaining budget.

Stage 4 note: the rollback variant relies on `--lr 0.05` wrecking the
adapter between saves, which is near-certain but stochastic; if no
`rolled_back` (or `consecutive_rollback_stop`) event fires, rerun once
before treating it as a bug. The
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

Unified play / self-eval / debrief onto the former multi-gold prompts (2026-08-19):
play is one generation per `ask`, debrief/self-eval grade on the self-eval rubric
(`RATING: -1..1`, not 0–10). Training stays sealed one-gold eat-to-win.
`weighted_loss` / collation untouched, so t3/t4 do NOT trigger.

* **Rerun t1 only** (seconds — END_GAME parse on the base class, openings on
  `settings_to_dict`, `settings_json_with_openings` backfill, oracle/action-balance
  END_GAME guards, prompt composition).
  Prompt byte-identity vs old `SCENE_PLAY` is **not** required (intentional switch
  onto today's multi-gold strings). The recorded sha256 digests in this file
  therefore change: `SYSTEM_PROMPT_SCENE_PLAY` / `SYSTEM_PROMPT_SCENE_ANALYST`
  must match the pre-change `*_MULTI` hashes, not the old unsuffixed ones.

Core tips in NAMS (2026-08-19): scene-play / scene-analyst / debrief system
prompts are assembled from numbered `core_player_*` / `core_analyst_*`
Preference rows (exact category fetch, category-sort join). Code seed remains
source of truth; assembled prompts must stay byte-identical to the
`SYSTEM_PROMPT_*` constants. `weighted_loss` / collation untouched.

* **Rerun t0** (seconds — NAMS `add_preference` byte-identity probe: newlines,
  `[FORWARD]`, an `[ANALYST]` line; leftover Preference must be gone).
  A FAIL here means NAMS mutates stored text and core tips cannot use
  `add_preference`.
* **Rerun t1** (seconds — numbered category regex, sorted-assembly byte
  identity vs the three `SYSTEM_PROMPT_*` constants, exclusion-set
  membership, `untag_analyst_text(tag_analyst_text(text))` round-trip).
* **Remote:** with a live client, `load_scene_prompts` returns strings
  hash-equal to the constants; a fresh graph (post-reset) self-heals with
  INFO logs; a manually edited core row heals with a WARNING.
  `session.reset_memory_to_seed()` (notebooks + datagen) now re-heals
  `core_player_*` / `core_analyst_*` after the episodic wipe, so a graph
  that never had them still gets the current crop.
* **Rerun t5** (datagen smoke — parallel path + tripwire silent: no analyst
  tip text in player context). Recorded prompt hashes do NOT change.
