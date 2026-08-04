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

Per-token weight construction (ALL WEIGHTS NON-NEGATIVE -- see the collapse
postmortem above :func:`build_span_weights`), in order:

  1. base    = ``max(0, (rating + 0.5) / 1.5) * rating_scale`` over the
     whole player reply (every player token in a reply is the same "move"):
     0 at rating <= -0.5, rising continuously to 1.0 at rating +1.0;
  2. WRONG   = ``wrong_weight`` (default 0.0) overrides the base on every
     occurrence of every harness-verified WRONG span -- masked out of
     cloning, never "unlearned" with a negative weight;
  3. boost   = ``win_boost * win_gamma ** moves_from_end`` added UNIFORMLY
     to every token of the message when the game was won (the win is the
     one ground-truth signal, so it softens even verified mistakes). The
     boost is DELIBERATELY LARGE (1.0, decaying by 0.95/round) and there
     is NO upper clamp: near a win every token weighs ``base + ~1.0`` (up
     to ~2.0), so real-game success OUTWEIGHS anything the analyst can
     award elsewhere -- wins must be clamped onto and kept, not
     re-litigated by a hackable analyst. Non-negativity is the only hard
     bound (the collapse postmortem);
  4. action balance (TEMPORARY HACK, see the screaming block above
     :func:`action_balance_multipliers`): each move record is scaled by
     ``mean_count / count(its action)`` over the same corpus (capped at
     4x), so every move TYPE carries equal total cloning mass -- weights
     may exceed 1.0 (still non-negative, so the collapse-proofing is
     untouched);
  5. novelty decay (WORK IN PROGRESS, :class:`NoveltyTracker`): the whole
     record's weights are multiplied by ``max(0.1, 0.9 ** k)`` where ``k``
     counts consecutive identical moves -- repeating the same move over and
     over earns less and less cloning weight ("boredom").

Records whose analyst forgot the RATING line are DROPPED with a warning
count -- never trained with a guessed reward (no-fuzzy-fallbacks).

``PlayerAnchorSource`` is the third source over the same trace file: an
RLHF-style trust region that pins the player's distribution to the PARENT
checkpoint (``loss="kd_anchor"``, uniform weights, every parseable record
including the rating-null and rating <= -0.5 ones -- the tokens the reward
mapping zeroes out of cloning are exactly where a collapse would hide).
See the class docstring for why the anchor is the parent, never the base.

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


# =====================================================================
# WHY EVERY WEIGHT IS NON-NEGATIVE -- THE 2026-08-01 COLLAPSE POSTMORTEM
# =====================================================================
# The original mapping used the analyst rating directly (weights in
# [-1, 1]). Weighted cross-entropy with NEGATIVE weights is UNBOUNDED
# BELOW: the model is paid to make negatively-weighted tokens arbitrarily
# unlikely, and the cheapest way to do that is to destroy the language
# model itself. The 2026-08-01 weekend run did exactly that -- epoch 2's
# corpus skewed negative, training loss fell 0.11 -> -37.5 in ~300 steps,
# and the checkpoint fled into `<unused...>` token gibberish, burning the
# remaining ~15h of datagen on unparseable output.
#
# The fix is this mapping: max(0, (rating + 0.5) / 1.5).
#   * BOUNDED OBJECTIVE: no weight is ever negative, so weighted CE is an
#     ordinary (reward-weighted) cloning loss -- minimizing it can never
#     profit from wrecking the LM. This is the collapse fix.
#   * CONTINUOUS AT THE CUTOFF: weight hits 0 exactly at rating -0.5 and
#     rises linearly to 1.0 at rating +1.0 -- no jump where a tiny rating
#     difference flips a move between "clone fully" and "ignore".
#   * FLOOR, NOT PUNISHMENT: moves rated at or below -0.5 (confirmed bad)
#     teach NOTHING rather than being unlearned; gradation is preserved
#     everywhere above the floor, so the analyst's dense per-move signal
#     still shapes what gets cloned hardest.
#   * NO BATCH-RELATIVE NORMALIZATION: the analyst's zero is semantically
#     absolute ("neither helps nor hurts"), so scores are shifted by the
#     FIXED offset +0.5, not by a per-batch mean -- a batch of all-good
#     moves must not have its best moves relatively punished.
# What negative weights were meant to buy (suppressing bad behavior) is
# instead provided by the PlayerAnchorSource trust region (bounded by
# construction) -- see the class below.
# =====================================================================

def build_span_weights(
    target_text: str,
    rating: float,
    wrong_spans: list[str],
    game_won: bool,
    moves_from_end: int | None,
    rating_scale: float = 1.0,
    wrong_weight: float = 0.0,
    win_boost: float = 1.0,
    win_gamma: float = 0.95,
) -> list[tuple[int, int, float]]:
    """The reward mapping (module-level so tests and TRAINING_TRACE_EXTRAS
    tooling can reuse it verbatim). Returns Collator-ready
    ``(char_start, char_end, weight)`` spans; later spans override earlier
    ones, so the whole-reply base span comes first. All outputs are
    NON-NEGATIVE (the loud block above explains why that is the one hard
    invariant); there is deliberately NO upper clamp -- near a win,
    ``base + boost`` reaches ~2.0, so real-game success outweighs even a
    perfectly analyst-rated move in a lost game."""
    boost = 0.0
    if game_won and moves_from_end is not None:
        boost = win_boost * (win_gamma ** moves_from_end)

    def final(w: float) -> float:
        return max(0.0, w + boost)

    base = max(0.0, (float(rating) + 0.5) / 1.5) * rating_scale
    spans: list[tuple[int, int, float]] = [
        (0, len(target_text), final(base))
    ]
    for span_text in wrong_spans:
        if not span_text:
            continue
        start = target_text.find(span_text)
        while start != -1:
            spans.append((start, start + len(span_text), final(wrong_weight)))
            start = target_text.find(span_text, start + 1)
    return spans


# =====================================================================
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!  TEMPORARY HACK -- ACTION BALANCE -- REMOVE WHEN NO LONGER NEEDED !!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# The 2026-08-04 retest showed the self-training loop collapsing onto
# turning: reward-weighted regression with a near-saturated analyst
# (77% of iter2 moves rated exactly +1.0) degenerates into behavior
# cloning of the corpus ACTION MIX, and the corpus was already
# turn-heavy, so each epoch amplified turns (smoke evals went from 28%
# FORWARD / 40% ANTICLOCK to 6.5% FORWARD / 69% ANTICLOCK -- a spin-bot
# that never approaches the gold).
#
# THIS IS NOT A PRINCIPLED FIX. Every move record's weights are scaled by
# mean_count / count(its action), computed over the SAME corpus being
# loaded, so every move TYPE carries equal total cloning mass per epoch.
# It replaced an earlier flat FORWARD bonus, which was open-loop: it
# pushed FORWARD up regardless of how much FORWARD the corpus already
# had, so a future FORWARD-heavy corpus would have been pushed further --
# the spin-bot runaway pointed the other way. Balancing is closed-loop:
# an over-represented action is automatically damped below 1.0, and the
# only equilibrium is a balanced mix.
#
# WHY THE CAP: uncapped inverse frequency explodes on a near-collapsed
# corpus -- 5 FORWARD records out of ~2500 moves would each get ~165x
# weight, amplifying a handful of possibly-bad examples into a third of
# the epoch's signal. Multipliers clamp to [1/CAP, CAP]; a binding cap is
# logged at WARNING because it means the corpus itself is degenerate.
#
# REMOVE THIS once a principled per-move signal exists (analyst revamp
# with per-span ratings / an arousal axis, or distance-based rewards --
# FUTURE_GOALS.md). It CANNOT survive into richer environments where move
# types genuinely differ in importance or legitimate frequency: "equal
# mass per move type" is only defensible when every move type is roughly
# equally load-bearing, as in this 3-move game. Anyone reading this in
# 2027: if it is still here, it has overstayed its welcome.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ACTION_BALANCE_CAP = 4.0


def action_balance_multipliers(counts: dict[str, int]) -> dict[str, float]:
    """Per-action inverse-frequency multipliers (TEMPORARY HACK -- the
    screaming block above): ``mean(counts) / counts[a]``, clamped to
    ``[1/ACTION_BALANCE_CAP, ACTION_BALANCE_CAP]``. Total mass is roughly
    preserved by construction (sum of count x multiplier = n_actions x
    mean, before clamping)."""
    if not counts:
        return {}
    mean_count = sum(counts.values()) / len(counts)
    out: dict[str, float] = {}
    for action, n in counts.items():
        raw = mean_count / n
        clamped = max(1.0 / ACTION_BALANCE_CAP, min(ACTION_BALANCE_CAP, raw))
        if clamped != raw:
            logger.warning(
                "action balance CAP engaged for %r: raw x%.2f clamped to "
                "x%.2f -- the corpus action mix is degenerate (counts %s)",
                action, raw, clamped, counts,
            )
        out[action] = clamped
    return out


# =====================================================================
# WORK IN PROGRESS -- NOVELTY ("BOREDOM") DECAY
# =====================================================================
# Biological feedback loops do not reward every step equally: repeating
# the same action over and over stops being exciting ("boredom"), while
# novel or highly consequential moments are consolidated more thoroughly.
# This is the first, deliberately crude cut at that idea: the k-th
# consecutive identical move earns only NOVELTY_DECAY**k of its cloning
# weight (floored at NOVELTY_FLOOR ~= "unrewarded"), so the 2026-08-04
# spin-bot pattern (dozens of consecutive ANTICLOCKs) stops feeding on
# itself, while a *varied* move sequence keeps full weight.
#
# KNOWN GAPS (future refinements live in FUTURE_GOALS.md):
#   * only catches IDENTICAL CONSECUTIVE moves -- an alternating
#     CLOCK/ANTICLOCK oscillation or a revisited position sails through;
#     repeat detection on (position, heading, action) state hashes would
#     catch those;
#   * decay-only: there is no positive arousal side yet (upweighting the
#     moves leading into gold pickups / wins / near-misses);
#   * operates on training weights after the fact; datagen-time rejection
#     sampling or an analyst "novelty/arousal" axis would shape the data
#     itself.
# =====================================================================
NOVELTY_DECAY = 0.9   #: per-repeat multiplier on cloning weight (WIP)
NOVELTY_FLOOR = 0.1   #: never below this -- "unrewarded", not "unlearned"


class NoveltyTracker:
    """Streaming repeat-streak tracker (WORK IN PROGRESS -- rationale and
    known gaps in the block above).

    Call :meth:`multiplier` once per trace record IN FILE ORDER. Records
    are keyed by ``(session_id, game_index)`` (worker files are merged by
    concatenation, so within one game the ``move_index`` order is the file
    order). Perception records (``action is None``) return 1.0 and are
    SKIPPED, NOT RESET: a turn-loop interrupted by a perception question
    is still a loop."""

    def __init__(self, decay: float = NOVELTY_DECAY,
                 floor: float = NOVELTY_FLOOR):
        self.decay = decay
        self.floor = floor
        #: (session_id, game_index) -> (last_action, streak_length)
        self._state: dict[tuple, tuple[str, int]] = {}

    def multiplier(self, session_id, game_index, action) -> float:
        if action is None:
            return 1.0
        key = (session_id, game_index)
        last_action, streak = self._state.get(key, (None, 0))
        streak = streak + 1 if action == last_action else 0
        self._state[key] = (action, streak)
        return max(self.floor, self.decay ** streak)


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
        wrong_weight: float = 0.0,
        win_boost: float = 1.0,
        win_gamma: float = 0.95,
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
        # ACTION BALANCE (TEMPORARY HACK -- screaming block above): one
        # counting pre-pass over the records that will actually train
        # (rated move records), so each move type gets equal total mass.
        counts: dict[str, int] = {}
        for _, _, _, meta in self._iter_records():
            action = meta.get("action")
            if action is not None and meta.get("rating") is not None:
                counts[action] = counts.get(action, 0) + 1
        self.action_balance = action_balance_multipliers(counts)
        if self.action_balance:
            logger.info(
                "%s: action balance multipliers %s (counts %s)", self.name,
                {a: round(m, 3) for a, m in self.action_balance.items()},
                counts,
            )

    def examples(self) -> Iterator[TrainingExample]:
        # Fresh noise every run: an unseeded Random gives a new degradation
        # per training run; pass noise_seed for reproducibility. The temp
        # dir lives for the run and is left to OS tmp cleanup.
        rng = random.Random(self.noise_seed)
        noise_dir = Path(tempfile.mkdtemp(prefix=f"{self.name}_noise_"))
        novelty = NoveltyTracker()
        n_dropped = 0
        n_yielded = 0
        for lineno, messages, target_text, meta in self._iter_records():
            # The streak advances on EVERY move record -- even ones dropped
            # below for a missing rating -- because the move still happened
            # in the game; only the training weight is per-record.
            novelty_mult = novelty.multiplier(
                meta.get("session_id"), meta.get("game_index"),
                meta.get("action"),
            )
            if meta.get("rating") is None:
                n_dropped += 1
                continue

            self._rewrite_frames(messages, rng, noise_dir, lineno)

            spans = build_span_weights(
                target_text,
                rating=float(meta["rating"]),
                wrong_spans=list(meta.get("wrong_spans") or []),
                game_won=bool(meta.get("game_won")),
                moves_from_end=meta.get("moves_from_end"),
                rating_scale=self.rating_scale,
                wrong_weight=self.wrong_weight,
                win_boost=self.win_boost,
                win_gamma=self.win_gamma,
            )
            # Two whole-record multipliers compose here (the reply is a
            # rationalization of its move; zero weights -- WRONG spans,
            # floored ratings -- stay zero; results may exceed 1.0, which
            # is fine: NON-NEGATIVITY is the collapse-proofing property):
            #   * novelty decay (WIP, NoveltyTracker above);
            #   * action balance (TEMPORARY HACK, screaming block above)
            #     -- perception records (action None) are untouched.
            mult = novelty_mult * self.action_balance.get(
                meta.get("action"), 1.0
            )
            if mult != 1.0:
                spans = [(s, e, w * mult) for s, e, w in spans]

            yield TrainingExample(
                messages=messages,
                target_text=target_text,
                span_weights=spans,
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

    #: Absolute catastrophe bound for this source's held-out KD guard
    #: (train.py MetricGuard.ceiling): the held-out analyst KD loss is THE
    #: analyst-drift meter, healthy runs sit ~1, and the 2026-08-01
    #: collapse pushed it to 63 -- anything past 5.0 is a destroyed
    #: analyst, not noise, regardless of the best-seen value.
    guard_ceiling = 5.0
    #: Ceiling-only guard (train.py MetricGuard.relative rationale): this
    #: metric starts at its floor (student == teacher at step 0) and can
    #: only rise, so best-ever multipliers measure "any drift at all",
    #: not catastrophe. The 2026-08-04 retest soft-warned at 1.37x of the
    #: entropy floor -- pure noise.
    guard_relative = False

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


#: PlayerAnchorSource mixture weight: each epoch replays ~a quarter of the
#: player corpus as trust-region examples on top of the full reward-weighted
#: pass. Enough anchor mass to bound drift on every kind of player context
#: (the records repeat across both sources, so coverage is shared), small
#: enough to cost only ~half an hour per train stage.
PLAYER_ANCHOR_WEIGHT = 0.25

#: Absolute catastrophe bound for the player anchor's held-out KD guard.
#: The metric is soft cross-entropy vs the parent = parent entropy
#: (~0.06-0.08, the step-0 floor) + per-token KL(parent||student), so 1.0
#: allows ~0.9 nats/token of drift. Calibration: healthy RLHF fine-tuning
#: lives at 0.05-0.3 nats/token (trl's adaptive KL controller targets ~6
#: nats per whole response; Stiennon et al. 2020 saw quality decay past
#: ~20-25 nats/sequence), outputs go visibly weird past ~1-2 nats/token,
#: and the 2026-08-01 collapse measured ~6 nats/token. 1.0 sits an order
#: of magnitude above healthy drift and well below collapse. The
#: 2026-08-04 retest showed why a RELATIVE bound cannot work here: best-
#: ever pins to the step-0 entropy floor, so "2x best" rolled back a
#: healthy run at 0.15 nats/token of drift (see MetricGuard docstring).
PLAYER_ANCHOR_CEILING = 1.0


class PlayerAnchorSource(_TraceFileSource):
    """RLHF-style trust region: pin the player's distribution to the PARENT
    checkpoint (``loss="kd_anchor"`` -- soft cross-entropy against the
    adapter this epoch resumed from, train.py THREE LOSS KINDS).

    WHY THE PARENT AND NEVER THE BASE: the point of the whole loop is to
    SURPASS the base model, so a base anchor would fight the objective. The
    parent anchor only bounds how far a SINGLE epoch can move -- the anchor
    itself advances every epoch, so cumulative improvement stays unbounded
    while no one epoch can flee to gibberish. (On an epoch-1-from-base run
    there is no parent checkpoint and the anchor falls back to the base,
    which IS that epoch's parent -- same semantics.)

    WHY EVERY PARSEABLE RECORD IS KEPT -- including ``rating: null`` and
    rating <= -0.5 ones that ``GameTraceSource`` drops or zero-weights: the
    tokens the reward mapping excludes from cloning are exactly the
    unconstrained directions a collapse escapes through (the 2026-08-01
    collapse lived in tokens nothing was pulling on). Uniform weight,
    no reward: this is a leash, not a lesson.
    """

    #: Ceiling-only drift guard -- see PLAYER_ANCHOR_CEILING above and
    #: MetricGuard.relative in train.py for the full rationale.
    guard_ceiling = PLAYER_ANCHOR_CEILING
    guard_relative = False

    def __init__(
        self,
        path: str | Path,
        weight: float = PLAYER_ANCHOR_WEIGHT,
        noise_strength: float = TRAINING_STRENGTH,
        noise_seed: int | None = None,
    ):
        super().__init__(path, noise_strength, noise_seed)
        self.name = f"player_anchor_{self.trace_dir.name}"
        self.weight = weight

    def examples(self) -> Iterator[TrainingExample]:
        rng = random.Random(self.noise_seed)
        noise_dir = Path(tempfile.mkdtemp(prefix=f"{self.name}_noise_"))
        for lineno, messages, target_text, meta in self._iter_records():
            self._rewrite_frames(messages, rng, noise_dir, lineno)
            yield TrainingExample(
                messages=messages,
                target_text=target_text,
                span_weights=None,   # uniform: constrain EVERY reply token
                loss="kd_anchor",
                source=self.name,
                meta=meta,
            )
