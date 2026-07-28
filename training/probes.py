"""Task-level capability probes: exact-match accuracy on fixed item sets.

These are the first REAL entries of the early-warning suite promised in
TRAINING_EXTRA_DATASETS.md: unlike held-out loss (which measures
distribution drift), a probe generates an actual answer and scores it, so
it measures the capability itself. Two probes ship by default, both built
by training/download_external.py:

  * ``exact_match/gsm8k``      -- 100 fixed GSM8K test items (arithmetic);
  * ``exact_match/navigation`` -- 100 fixed synthetic clock/compass/bearing
    items (the direct instrument for the geometry failure modes).

Probe items are ``{"messages": [...], "answer": "<gold>"}`` jsonl rows whose
prompts explicitly ask for a final ``ANSWER: <value>`` line; extraction
takes the last ANSWER: line, falling back to the last number in the reply.
Generation is greedy and short, ~3-5 min per probe pair per save on an A100
at the default ``save_steps`` -- acceptable.

Usage (run scripts)::

    from training.probes import build_probe_hooks
    hooks, guards = build_probe_hooks()
    run_training(sources, cfg, extra_hooks=hooks, extra_guards=guards)
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.external_data import load_manifest
from training.train import MetricGuard, TrainContext

logger = logging.getLogger("train.probes")

_ANSWER_RE = re.compile(r"ANSWER\s*:\s*(.+)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def extract_answer(text: str) -> str | None:
    """Last 'ANSWER: <value>' line, else the last number in the reply,
    else None (scored as wrong -- a model that stops emitting the answer
    format has ALSO regressed)."""
    matches = _ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()
    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1]
    return None


def _normalize(ans: str) -> str:
    ans = ans.strip().strip(".*` ").lower()
    # "anti-clockwise" / "anticlockwise" / "counter-clockwise" are one answer
    ans = ans.replace("-", "").replace(" ", "").replace("anticlockwise",
                                                        "counterclockwise")
    return ans


def answers_match(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    p, g = _normalize(pred), _normalize(gold)
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return False


class ExactMatchHook:
    """Eval hook (``hook(ctx) -> dict``): greedy-generate on every probe
    item, report exact-match accuracy under ``metric``."""

    def __init__(self, metric: str, probe_path: str | Path,
                 max_new_tokens: int = 256):
        self.metric = metric
        self.probe_path = Path(probe_path)
        self.max_new_tokens = max_new_tokens
        if not self.probe_path.is_file():
            raise FileNotFoundError(
                f"probe file missing: {self.probe_path} -- run: "
                "python -m training.download_external"
            )
        self.items = [
            json.loads(line)
            for line in self.probe_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.items:
            raise ValueError(f"probe file is empty: {self.probe_path}")

    def __call__(self, ctx: TrainContext) -> dict[str, float]:
        import torch

        model = ctx.model
        tokenizer = getattr(ctx.processor, "tokenizer", ctx.processor)
        was_training = model.training
        model.eval()
        # use_cache is disabled for gradient checkpointing; generation
        # without it is quadratic, so re-enable just for the probe.
        prev_cache = getattr(model.config, "use_cache", False)
        model.config.use_cache = True
        n_correct = 0
        try:
            with torch.no_grad():
                for item in self.items:
                    norm = ctx.adapter.prepare_messages(item["messages"])
                    inputs = ctx.processor.apply_chat_template(
                        norm, tokenize=True, add_generation_prompt=True,
                        return_dict=True, return_tensors="pt",
                    ).to(ctx.device)
                    out = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                    )
                    reply = tokenizer.decode(
                        out[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                    if answers_match(extract_answer(reply), item["answer"]):
                        n_correct += 1
        finally:
            model.config.use_cache = prev_cache
            if was_training:
                model.train()
        accuracy = n_correct / len(self.items)
        logger.info("%s: %d/%d = %.3f", self.metric, n_correct,
                    len(self.items), accuracy)
        return {self.metric: accuracy}


def build_probe_hooks(
    manifest_path: str | Path | None = None,
    rel_tolerance: float = 0.10,
) -> tuple[list[ExactMatchHook], dict[str, MetricGuard]]:
    """One (hook, guard) pair per enabled manifest entry that declares a
    probe. Missing probe files fail loudly at construction time (before any
    GPU-hours are spent), pointing at the downloader."""
    hooks: list[ExactMatchHook] = []
    guards: dict[str, MetricGuard] = {}
    for entry in load_manifest(manifest_path):
        if not (entry.enabled and entry.probe):
            continue
        kind = entry.probe.get("kind")
        if kind != "exact_match":
            raise ValueError(
                f"{entry.name}: unknown probe kind {kind!r} -- only "
                "'exact_match' is implemented (training/probes.py)"
            )
        metric = f"exact_match/{entry.name}"
        hooks.append(ExactMatchHook(metric, entry.probe_path))
        guards[metric] = MetricGuard(
            metric, higher_is_better=True, rel_tolerance=rel_tolerance,
        )
    return hooks, guards
