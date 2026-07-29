"""Game-trace DataSources: player RL examples and the analyst KD anchor.

``GameTraceSource`` consumes ``data_game/<label>/traces.jsonl`` written by
``training/generate_game_traces.py`` and turns each record's RAW annotations
(analyst rating, verified WRONG spans, game outcome) into the per-token
weights of train.py's weighted-CE loss -- which makes the whole scheme
single-sample offline REINFORCE with a shaped per-token advantage
(TRAINING_GAME_TRACES.md names and justifies the algorithm and every ratio).

``AnalystTraceSource`` consumes the sibling ``analyst_traces.jsonl`` (the
exact analyst prompts + the analyses they produced) as a KD replay anchor:
uniform-weight ``loss="kd"`` examples, i.e. soft cross-entropy against the
FROZEN BASE model's distribution on the analyst's own contexts. It exists
because the analyst is never trained but the shared network under it is --
this source pins analyst behavior to base Gemma while player RL moves the
weights, and its per-source held-out KD loss doubles as the analyst-drift
meter (TRAINING_EXTRA_DATASETS.md). The teacher is always the frozen base
regardless of which checkpoint generated the trace, so the anchor never
chases its own drift.

Per-token weight construction, in order:

  1. base    = ``rating * rating_scale`` over the whole player reply
     (every player token in a reply is the same "move");
  2. WRONG   = ``wrong_weight`` (default -1.0) overrides the base on every
     occurrence of every harness-verified WRONG span;
  3. boost   = ``win_boost * win_gamma ** moves_from_end`` added UNIFORMLY
     to every token of the message when the game was won (the win is the
     one ground-truth signal, so it softens even verified mistakes);
  4. clamp to [-1, 1];
  5. any still-negative weight is multiplied by ``negative_scale``
     (1.0 = symmetric REINFORCE, 0.5 = gentler unlearning, 0.0 = filtered
     behavior cloning / strictly positive).

Records whose analyst forgot the RATING line are DROPPED with a warning
count -- never trained with a guessed reward (no-fuzzy-fallbacks).

Images: message urls are stored relative to the trace folder; at load time
each referenced frame gets a per-run noised copy (label-safe degradation,
``training/image_noise.py``) in a fresh temp directory, so every training
run regularizes differently on top of the (already mildly noised) stored
frame. ``noise_strength=0`` trains on the stored bytes directly.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import tempfile
from pathlib import Path
from typing import Iterator

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.image_noise import TRAINING_STRENGTH, noise_file
from training.train import DataSource, TrainingExample

logger = logging.getLogger("train.game_traces")


def _clamp(x: float) -> float:
    return max(-1.0, min(1.0, x))


def build_span_weights(
    target_text: str,
    rating: float,
    wrong_spans: list[str],
    game_won: bool,
    moves_from_end: int | None,
    rating_scale: float = 1.0,
    wrong_weight: float = -1.0,
    win_boost: float = 0.2,
    win_gamma: float = 0.9,
    negative_scale: float = 1.0,
) -> list[tuple[int, int, float]]:
    """The reward mapping (module-level so tests and TRAINING_TRACE_EXTRAS
    tooling can reuse it verbatim). Returns Collator-ready
    ``(char_start, char_end, weight)`` spans; later spans override earlier
    ones, so the whole-reply base span comes first."""
    boost = 0.0
    if game_won and moves_from_end is not None:
        boost = win_boost * (win_gamma ** moves_from_end)

    def final(w: float) -> float:
        w = _clamp(w + boost)
        return w * negative_scale if w < 0 else w

    spans: list[tuple[int, int, float]] = [
        (0, len(target_text), final(rating * rating_scale))
    ]
    for span_text in wrong_spans:
        if not span_text:
            continue
        start = target_text.find(span_text)
        while start != -1:
            spans.append((start, start + len(span_text), final(wrong_weight)))
            start = target_text.find(span_text, start + 1)
    return spans


class _TraceFileSource(DataSource):
    """Shared plumbing for the two trace-file sources: path validation,
    record parsing, and per-run noised frame copies. Subclasses set
    ``self.name`` and implement :meth:`examples`."""

    def __init__(self, path: str | Path, noise_strength: float,
                 noise_seed: int | None):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"{type(self).__name__}: no such file: {self.path} -- run "
                "python -m training.generate_game_traces first"
            )
        self.trace_dir = self.path.parent
        self.noise_strength = noise_strength
        self.noise_seed = noise_seed

    def _noised_copy(self, rel_url: str, rng: random.Random,
                     noise_dir: Path, index: int) -> str:
        src = self.trace_dir / rel_url
        if not src.is_file():
            raise FileNotFoundError(
                f"{self.path}: referenced frame missing: {src}"
            )
        if self.noise_strength <= 0:
            return str(src)
        dest = noise_dir / f"{index:06d}_{src.name}"
        noise_file(src, rng, self.noise_strength, out_path=dest)
        return str(dest)

    def _iter_records(self) -> Iterator[tuple[int, list[dict], str, dict]]:
        """Parse-validated raw records: (lineno, messages, target_text,
        meta). Malformed lines are a hard error, never skipped."""
        with open(self.path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    messages = obj["messages"]
                    target_text = obj["target_text"]
                    meta = obj["meta"]
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(
                        f"{self.path}:{lineno}: bad trace record "
                        f"({type(exc).__name__}: {exc})"
                    ) from exc
                yield lineno, messages, target_text, meta

    def _rewrite_frames(self, messages: list[dict], rng: random.Random,
                        noise_dir: Path, lineno: int) -> None:
        """Resolve every image url against the trace dir (noised per run)."""
        for m in messages:
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    part["url"] = self._noised_copy(
                        part["url"], rng, noise_dir, lineno
                    )


class GameTraceSource(_TraceFileSource):
    """One materialized datagen run as a DataSource. ``weight`` is the usual
    epoch-mixture knob; the reward knobs are documented above and in
    TRAINING_GAME_TRACES.md."""

    def __init__(
        self,
        path: str | Path,
        weight: float = 1.0,
        rating_scale: float = 1.0,
        wrong_weight: float = -1.0,
        win_boost: float = 0.2,
        win_gamma: float = 0.9,
        negative_scale: float = 1.0,
        noise_strength: float = TRAINING_STRENGTH,
        noise_seed: int | None = None,
    ):
        super().__init__(path, noise_strength, noise_seed)
        self.name = f"game_{self.trace_dir.name}"
        self.weight = weight
        self.rating_scale = rating_scale
        self.wrong_weight = wrong_weight
        self.win_boost = win_boost
        self.win_gamma = win_gamma
        self.negative_scale = negative_scale

    def examples(self) -> Iterator[TrainingExample]:
        # Fresh noise every run: an unseeded Random gives a new degradation
        # per training run; pass noise_seed for reproducibility. The temp
        # dir lives for the run and is left to OS tmp cleanup.
        rng = random.Random(self.noise_seed)
        noise_dir = Path(tempfile.mkdtemp(prefix=f"{self.name}_noise_"))
        n_dropped = 0
        n_yielded = 0
        for lineno, messages, target_text, meta in self._iter_records():
            if meta.get("rating") is None:
                n_dropped += 1
                continue

            self._rewrite_frames(messages, rng, noise_dir, lineno)

            yield TrainingExample(
                messages=messages,
                target_text=target_text,
                span_weights=build_span_weights(
                    target_text,
                    rating=float(meta["rating"]),
                    wrong_spans=list(meta.get("wrong_spans") or []),
                    game_won=bool(meta.get("game_won")),
                    moves_from_end=meta.get("moves_from_end"),
                    rating_scale=self.rating_scale,
                    wrong_weight=self.wrong_weight,
                    win_boost=self.win_boost,
                    win_gamma=self.win_gamma,
                    negative_scale=self.negative_scale,
                ),
                loss="ce",
                source=self.name,
                meta=meta,
            )
            n_yielded += 1
        if n_dropped:
            logger.warning(
                "%s: DROPPED %d/%d record(s) with no parseable RATING "
                "(never train on a guessed reward).",
                self.name, n_dropped, n_dropped + n_yielded,
            )


class AnalystTraceSource(_TraceFileSource):
    """The analyst KD anchor (rationale in the module docstring and
    TRAINING_EXTRA_DATASETS.md).

    Every example is ``loss="kd"`` with uniform span weights: the recorded
    analysis defines the supervised positions, the frozen base supplies the
    per-token target distribution at train time -- no reward enters (this
    is an anchor, not RL), so records with ``rating: null`` are KEPT.
    Records whose analysis quoted unverified (hallucinated) WRONG spans are
    DROPPED loudly by default -- the one analyst failure the harness detects
    for free is not worth anchoring on.

    ``examples_per_epoch`` follows the manifest quota convention
    (``weight = quota / usable records``), so the per-epoch mix stays fixed
    as the trace corpus grows across datagen runs.
    """

    def __init__(
        self,
        path: str | Path,
        examples_per_epoch: int = 150,
        noise_strength: float = TRAINING_STRENGTH,
        noise_seed: int | None = None,
        drop_unverified: bool = True,
    ):
        super().__init__(path, noise_strength, noise_seed)
        self.name = f"analyst_{self.trace_dir.name}"
        self.drop_unverified = drop_unverified
        self.examples_per_epoch = examples_per_epoch

        n_usable = sum(
            1 for _, _, _, meta in self._iter_records() if self._keep(meta)
        )
        if n_usable == 0:
            raise ValueError(
                f"{self.name}: {self.path} has no usable analyst records "
                "(empty file, or every record dropped by the unverified-"
                "span filter)"
            )
        self.n_usable = n_usable
        self.weight = examples_per_epoch / n_usable

    def _keep(self, meta: dict) -> bool:
        return not (self.drop_unverified and meta.get("unverified_spans"))

    def examples(self) -> Iterator[TrainingExample]:
        rng = random.Random(self.noise_seed)
        noise_dir = Path(tempfile.mkdtemp(prefix=f"{self.name}_noise_"))
        n_dropped = 0
        n_yielded = 0
        for lineno, messages, target_text, meta in self._iter_records():
            if not self._keep(meta):
                n_dropped += 1
                continue

            self._rewrite_frames(messages, rng, noise_dir, lineno)

            yield TrainingExample(
                messages=messages,
                target_text=target_text,
                span_weights=None,
                loss="kd",
                source=self.name,
                meta=meta,
            )
            n_yielded += 1
        if n_dropped:
            logger.warning(
                "%s: DROPPED %d/%d record(s) whose analysis quoted "
                "unverified WRONG spans (drop_unverified=True).",
                self.name, n_dropped, n_dropped + n_yielded,
            )
