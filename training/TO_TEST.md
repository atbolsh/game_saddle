# To test on the remote box (stage 1)

Nothing here has run yet — the local box cannot execute this code (see
`.cursor/rules/remote-environment.mdc`). Checklist before trusting train.py:

1. **`pip install bitsandbytes`** installs and imports cleanly next to
   torch/CUDA on the GPU box.
2. **4-bit load of gemma-4-12b**: `AutoModelForMultimodalLM.from_pretrained`
   accepts `quantization_config=BitsAndBytesConfig(...)` +
   `prepare_model_for_kbit_training` on the unified architecture.
3. **`--projector-module`**: the default `multi_modal_projector` must match a
   real module in `model.named_modules()`. On mismatch train.py fails loudly
   and prints the top-level submodules — pass the right name (or `none`).
4. **LoRA target discovery**: check `lora_targets` in the run's
   `config.json` — language-model linears only, no vision/audio modules,
   non-empty.
5. **Terminator id**: `resolve_terminator_id` should pick `<end_of_turn>`
   (logged in config.json as `terminator_id`).
6. **Scheduler name**: `get_scheduler("cosine_with_min_lr", ...,
   scheduler_specific_kwargs={"min_lr_rate": ...})` under transformers 5.x.
7. **End-to-end smoke run**: tiny hand-written jsonl (a few examples, at
   least one with an image part) → loss decreases, `pixel_values` present
   (the image assert must NOT fire), checkpoint lands in
   `weights/gemma-4-12b/<label>_step<K>/` with `adapter_config.json` +
   `train_meta.json`. Exercise both entry points: the CLI
   (`python -m training.train --data ...`) and a run script (point
   `run_first_iteration.py`'s SOURCES at the smoke jsonl).
8. **Rollback path**: rerun with `--regression-tolerance 0 --save-steps 1`
   to force a regression; expect the ERROR log + `rolled_back` event.
9. **Checkpoint loading**: the smoke checkpoint appears in the notebooks'
   Checkpoint dropdown, loads (and generates) via the picker, via
   `MODEL_CHECKPOINT` in `.env`, and via `python -m agent.runner
   --checkpoint <name> game ...`; `[default]` still loads bare HF weights.
10. **Span weights**: an example with `span_weights` (incl. a negative one)
    trains without error; spot-check the per-token mapping by logging one
    collated example.

# To test on the remote box (phase 2: external datasets)

The synthetic-navigation path (generator → materialize → probe →
`ExternalSource` → per-line loss) and every converter (on fabricated rows)
were smoke-tested locally; everything touching HF downloads, the GPU, or
PEFT internals still needs the remote box:

11. **Manifest ids resolve**: `python -m training.download_external` (run by
    `setup_env.sh`) materializes every enabled entry. Watch for per-row
    schema mismatches — the converters assume `question/answer` (gsm8k),
    `query/response` (metamathqa), `problem/solution` (numinamath),
    ShareGPT `conversations` (openthoughts/slimorca), `images/texts`
    (cauldron) — a wrong field name fails loudly with the row number.
12. **Streaming respects caps**: `--force --mode stream --only cauldron_vqav2`
    must stop after `max_rows` source rows without pulling the full 13.5 GB,
    and produce output identical to full mode (compare `meta.json` counts).
13. **settings.json**: `setup_env.sh` writes `download_mode` correctly for
    the box's free disk, and preserves an existing file on re-run.
14. **KD teacher path**: `model.disable_adapter()` on the PEFT-wrapped
    4-bit model bypasses BOTH the LoRA deltas and the `modules_to_save`
    projector copy (compare a teacher logit slice against the bare base
    model on one input). This is the linchpin of the KD loss.
15. **KD memory**: a long OpenThoughts example (thousands of target tokens)
    fits — the target-position slicing bounds the float32 softmax pair, but
    the two full forwards (student + teacher) still cost activation memory.
    If it OOMs, cap target length in the converter or drop
    `openthoughts` per-epoch count.
16. **Probe runtime + parsing**: `exact_match/gsm8k` and
    `exact_match/navigation` run in the promised ~3–5 min combined per save
    and the base model actually emits parseable `ANSWER:` lines (accuracy
    well above 0 before any training).
17. **eval_log.jsonl**: after a smoke run with probes attached, one flat row
    per evaluation with `heldout_loss/<source>` per dataset and both
    `exact_match/*` keys; loads with `pandas.read_json(..., lines=True)`.
18. **Replay-only smoke run**: `python -m training.run_first_iteration`
    (game line still commented out) completes a few hundred steps with
    mixed CE + KD sources, per-source losses all finite in
    `train_log.jsonl`.
19. **Self-distill utility** (optional path):
    `python -m training.generate_self_distill --dataset slimorca --limit 20`
    writes 20 non-empty regenerated targets.
