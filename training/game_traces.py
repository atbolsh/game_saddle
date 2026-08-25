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

Reward construction (2026-08-05 SHAPE/SCALE SPLIT -- train.py
``weighted_loss`` SHAPE VS SCALE explains the bug that forced it: the
per-example loss normalization cancels any uniform factor on span weights,
so every reply-wide reward knob must ride ``example_weight`` instead).

SCALE -- ``TrainingExample.example_weight``, "how much this reply matters":

  1. base = 0 if rating <= -0.5 (hard floor: confirmed-bad replies teach
     nothing) else ``min(ADV_CAP, exp((rating - r_bar) / ADV_BETA))`` --
     EXPONENTIAL ADVANTAGE against the corpus mean rating ``r_bar``
     (:func:`rating_advantage`; a pre-pass computes r_bar per corpus).
     The analyst's *ranking* is the usable signal, not its absolute
     numbers: with a saturated analyst (aug4: 77% of iter2 moves rated
     exactly +1.0) any affine mapping puts nearly all weights in one
     band, and reward-weighted regression with near-uniform weights is
     just behavior cloning. The exp restores contrast (~2x per 0.2 of
     rating around the mean) however compressed the ratings are;
  2. + win boost ``win_boost * win_gamma ** moves_from_end`` (ADDITIVE:
     ground truth, so it rescues even an analyst-floored reply near a
     win; 1.0 decaying by 0.95/round -- sized to outweigh anything the
     analyst can award, wins must be clamped onto and kept);
  3. x action balance (TEMPORARY HACK, screaming block above
     :func:`action_balance_multipliers`): equal total cloning mass per
     move TYPE, capped at 4x;
  4. x novelty decay -- OFF BY DEFAULT behind ``GameTraceSource(...,
     novelty=True)``, ON at the run_weekend call site since 2026-08-11
     (:class:`NoveltyTracker`);
  5. x ``ORACLE_WRONG_SCALE`` (0.25) when the engine oracle says the move
     contradicted ground truth (crutch block above :func:`oracle_verdict`);
  6. x ``TRANSITION_BOOST`` (2.0) on RAY-HIT rounds (2026-08-11): the
     gold is dead ahead and FORWARD gates the win -- the decisive
     transition moves are a handful per game and this keeps them from
     drowning in continuation moves. Compounds with 5 on a missed
     forward (0.25 x 2 = 0.5, with a sharpened -0.5 token span).

  A scale of exactly 0 (floored rating, no win boost) SKIPS the record.

SHAPE -- ``span_weights``, relative emphasis WITHIN the reply
(:func:`build_span_weights`):

  * base 1.0 over the whole reply (every token of a reply is the same
    "move");
  * harness-verified WRONG spans -> ``WRONG_SPAN_WEIGHT`` (-0.5): bounded
    UNLIKELIHOOD in the loss (train.py NEGATIVE WEIGHTS) -- actively
    suppressed with a floored objective, not just masked;
  * the move token gets ``ORACLE_MATCH_SPAN`` (1.5) when it matches the
    engine oracle and ``ORACLE_WRONG_SPAN`` (-0.5) when it contradicts it
    (neutral / no oracle meta: no modifier).

Records whose analyst forgot the RATING line are DROPPED with a warning
count -- never trained with a guessed reward (no-fuzzy-fallbacks).
Move-round records stamped ``target_missing`` or ``target_invalid`` (no
parseable TARGET line / a TARGET naming a NONEXISTENT object) are
DROPPED the same way -- corrupted analysis. A TARGET naming a real
object the rules forbid (an opening while gold remains) is a PLAYER
mistake, not corruption: the record trains and the oracle facts aim at
the engine target, so the verdict punishes the move. Perception rounds
are excused at DATAGEN time (both stamps False -- no move requested, no
target required) and train as prose graded by rating alone; old corpora
lack the keys (falsy → kept).

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

import atexit
import json
import logging
import math
import random
import shutil
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
# NEGATIVE SIGNAL -- THE 2026-08-01 COLLAPSE POSTMORTEM, CORRECTED
# =====================================================================
# The original mapping used the analyst rating directly as a CE weight
# (weights in [-1, 1]). Negative-weight CROSS-ENTROPY (w * -log p) is
# UNBOUNDED BELOW with a gradient that GROWS as p -> 0: the model is paid
# ever more to make those tokens ever less likely, and the cheapest
# descent direction is to destroy the whole distribution. The 2026-08-01
# weekend run did exactly that -- epoch 2's corpus skewed negative,
# training loss fell 0.11 -> -37.5 in ~300 steps, and the checkpoint fled
# into `<unused...>` token gibberish.
#
# THE REAL DICHOTOMY (2026-08-05 correction): the problem was never
# "negative signal" per se, it was the UNBOUNDED FORM of it. Bounded
# unlikelihood, -log(1 - p), also pushes a bad token's probability down,
# but the loss floors at 0 and its gradient VANISHES as p -> 0 -- there
# is no collapse well to dive into. train.py's weighted_loss now treats
# every negative span weight as unlikelihood with |w| emphasis, which is
# why WRONG spans and oracle-contradicted move tokens below carry -0.5
# instead of a 0.0 mask: verified-bad text gets actively suppressed
# again, safely. (An earlier note here claimed the PlayerAnchorSource
# trust region "provides what negative weights bought" -- wrong: a trust
# region only bounds drift from the parent; it suppresses nothing.)
#
# The reply-wide reward itself lives on example_weight (module docstring,
# SCALE) as an EXPONENTIAL ADVANTAGE against the corpus mean rating.
# The analyst's RANKING is the signal we trust; its absolute calibration
# drifts with the checkpoint (aug4's analyst saturated at +1.0), so a
# baseline-relative mapping stays informative where any fixed affine one
# degenerates into uniform cloning. The floor stays absolute: at or below
# rating -0.5 ("confirmed bad") a reply teaches nothing at all.
# =====================================================================

ADV_BETA = 0.3    #: rating units per e-fold of example weight
ADV_CAP = 3.0     #: max example-weight base (one reply must not own a batch)
RATING_FLOOR = -0.5   #: at/below this rating the reply teaches NOTHING
WRONG_SPAN_WEIGHT = -0.5  #: unlikelihood emphasis on verified WRONG spans


def rating_advantage(rating: float, baseline: float,
                     beta: float = ADV_BETA, cap: float = ADV_CAP) -> float:
    """Exponential-advantage base for ``example_weight`` (postmortem block
    above): 0 at/below the RATING_FLOOR, else ``exp((rating - baseline) /
    beta)`` capped at ``cap``. Equal to 1.0 for a corpus-average reply;
    ~2x per +0.2 of rating above the mean, symmetric below."""
    if float(rating) <= RATING_FLOOR:
        return 0.0
    return min(cap, math.exp((float(rating) - baseline) / beta))


def example_scale(rating: float, baseline: float, game_won: bool,
                  moves_from_end: int | None,
                  win_boost: float = 1.0,
                  win_gamma: float = 0.95) -> float:
    """Reply-wide reward scale (module docstring, SCALE steps 1-2):
    exp-advantage base plus the additive win boost. Action balance,
    novelty, and the oracle penalty multiply on top in GameTraceSource
    (they need corpus / stream / oracle state a pure function can't
    carry)."""
    boost = 0.0
    if game_won and moves_from_end is not None:
        boost = win_boost * (win_gamma ** moves_from_end)
    return rating_advantage(rating, baseline) + boost


def build_span_weights(
    target_text: str,
    wrong_spans: list[str],
    action: str | None = None,
    verdict: str = "unknown",
    wrong_weight: float = WRONG_SPAN_WEIGHT,
) -> list[tuple[int, int, float]]:
    """WITHIN-reply SHAPE only (module docstring, SHAPE; the reply-wide
    reward is :func:`example_scale` -- train.py's SHAPE VS SCALE explains
    why a uniform factor here would silently cancel). Returns
    Collator-ready ``(char_start, char_end, weight)`` spans; later spans
    override earlier ones, so: whole-reply base 1.0 first, then verified
    WRONG spans at ``wrong_weight`` (negative = bounded unlikelihood),
    then the oracle's move-token modifier LAST -- ground truth outranks
    the analyst even where the spans overlap."""
    spans: list[tuple[int, int, float]] = [(0, len(target_text), 1.0)]
    for span_text in wrong_spans:
        if not span_text:
            continue
        start = target_text.find(span_text)
        while start != -1:
            spans.append((start, start + len(span_text), wrong_weight))
            start = target_text.find(span_text, start + 1)
    if action and verdict in ("correct", "wrong"):
        token = f"[{action}]"
        start = target_text.rfind(token)
        if start != -1:
            spans.append((
                start, start + len(token),
                ORACLE_MATCH_SPAN if verdict == "correct"
                else ORACLE_WRONG_SPAN,
            ))
    return spans


# =====================================================================
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!  CRUTCH -- ENGINE ORACLE -- ACTIVATED 2026-08-05                 !!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# This is crutch #1 from TRAINING_GAME_TRACES.md's disagree-and-commit
# section ("engine-derived baselines"), previously rejected and now
# DELIBERATELY activated after two failed runs: the analyst's ratings
# saturated (aug4: 77% at +1.0) and stopped carrying enough contrast to
# steer anything, so the one incorruptible judge -- the game engine
# itself -- now grades the MOVE TOKEN directly. Datagen stamps the raw
# oracle facts per move (generate_game_traces._oracle_meta: rel bearing
# to the nearest gold, whether the facing ray hits any gold, and the
# resulting correct move -- EXACT, because the arena has no internal
# walls to route around; after golds are gone the same geodesic aims at
# the nearest opening, and a sealed empty board's oracle_move is
# END_GAME); this block classifies a recorded move against
# them at train time:
#
#   "correct" -- matches the oracle move (or turns when the gold is
#                nearly behind, |bearing| >= 170 deg, where either turn
#                is a correct move; or [END_GAME] on a sealed empty
#                board): move-token span ORACLE_MATCH_SPAN;
#   "neutral" -- defensible under the 20-degree aim tolerance the player
#                is INSTRUCTED to use (FORWARD inside the cone without a
#                ray hit): no modifier, the analyst's rating stands;
#   "wrong"   -- contradicts ground truth (turn away from the shorter
#                rotation, FORWARD outside the cone, ANY turn under a
#                ray hit -- the 2026-08-11 tightening below; [END_GAME]
#                while gold or an opening remains; a board move on a
#                sealed empty board): move-token span ORACLE_WRONG_SPAN
#                (bounded unlikelihood) AND the whole reply's
#                example_weight is multiplied by ORACLE_WRONG_SCALE --
#                the rationalization of a wrong move is not worth
#                cloning at full strength either;
#   "unknown" -- no oracle meta (pre-2026-08-05 corpora) or no move:
#                neutral, counted and logged once per source.
#
# 2026-08-11 TIGHTENING (after the aug6 11-epoch run): a turn under a
# ray hit -- the gold dead ahead, FORWARD demanded -- is now "wrong",
# not "fine-tuning neutral". The aug5 leniency let the run's biggest
# leak through unpunished: 900 missed forwards whose only grader was
# the analyst's coin-flip (42% rated >= +0.8), and FORWARD-on-ray-hit
# compliance sat at ~50% for 11 epochs. Ray-hit rounds also get the
# reply-wide TRANSITION_BOOST (below): they are the handful of decisive
# transition moves per game, drowned in ~2.6k continuation moves.
# Turns inside the 20-degree cone but OFF the ray keep their previous
# grades (toward = matches oracle_move = correct; away = wrong).
#
# WHY THIS CANNOT STAY: it hard-wires "good play == greedy geodesic to
# the nearest gold", which is only true because the current game is a
# featureless arena. The moment the environment grows internal walls,
# multiple competing goals, or moves whose value is strategic rather
# than geometric, this oracle becomes actively WRONG and must be removed
# or demoted to a feature the analyst sees (FUTURE_GOALS.md, analyst
# revamp). It also further concentrates reward on the move token --
# fine for a 3-move game, wrong for free-form action spaces. Anyone
# reading this in 2027: if it is still here, it has overstayed its
# welcome.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ORACLE_MATCH_SPAN = 1.5    #: move-token span weight when oracle agrees
ORACLE_WRONG_SPAN = -0.5   #: move-token unlikelihood when oracle disagrees
ORACLE_WRONG_SCALE = 0.25  #: example_weight multiplier on oracle-wrong moves
ORACLE_CONE_RAD = math.radians(20.0)   #: the player's instructed tolerance
ORACLE_EITHER_TURN_RAD = math.radians(170.0)  #: behind you: any turn is fine
#: Reply-wide example_weight boost on RAY-HIT rounds (2026-08-11, the
#: "arousal" first cut from FUTURE_GOALS goal 5): a ray hit is the
#: transition moment that gates winning -- taking the FORWARD is cloned
#: at double weight, and missing it compounds with ORACLE_WRONG_SCALE
#: (0.25 x 2 = 0.5 net) while the boost sharpens the -0.5 unlikelihood
#: span on the turn token.
TRANSITION_BOOST = 2.0


def oracle_verdict(action: str | None, oracle_move: str | None,
                   rel_bearing: float | None,
                   ray_hit: bool | None) -> str:
    """Classify a recorded move against the stamped oracle facts (crutch
    block above): "correct" | "neutral" | "wrong" | "unknown". Pure so t1
    can enumerate the geometry; thresholds live here (train side), the
    raw facts in the trace meta -- retuning never requires regenerating
    data."""
    if action is None or oracle_move is None:
        return "unknown"
    if oracle_move == "END_GAME":
        return "correct" if action == "END_GAME" else "wrong"
    if action == "END_GAME":
        # Quit while gold or an opening remains.
        return "wrong"
    if rel_bearing is None:
        return "unknown"
    rel = float(rel_bearing)
    if action == oracle_move:
        return "correct"
    if (action in ("CLOCK", "ANTICLOCK")
            and abs(rel) >= ORACLE_EITHER_TURN_RAD):
        return "correct"   # gold ~behind: either rotation is the move
    if action == "FORWARD" and not ray_hit and abs(rel) <= ORACLE_CONE_RAD:
        return "neutral"   # inside the instructed cone; analyst decides
    # A turn under a ray hit falls through to "wrong" (2026-08-11
    # tightening in the crutch banner: the aug5 "fine-tuning neutral"
    # here let missed forwards escape both reward halves).
    return "wrong"


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
    # [END_GAME] stays pinned at 1.0 even when it is a real terminal
    # (multi-gold sealed-empty). Inverse-frequency would amplify a
    # handful of quits -- including the wrong ones -- which is exactly
    # the failure mode this pin exists to prevent. The oracle match
    # span still teaches a correct quit.
    balanceable = {a: n for a, n in counts.items() if a != "END_GAME"}
    out: dict[str, float] = {}
    if "END_GAME" in counts:
        out["END_GAME"] = 1.0
    if not balanceable:
        return out
    mean_count = sum(balanceable.values()) / len(balanceable)
    for action, n in balanceable.items():
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
# WORK IN PROGRESS -- NOVELTY ("BOREDOM") DECAY -- OFF BY DEFAULT
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
# OFF BY DEFAULT (2026-08-05, ``GameTraceSource(..., novelty=True)`` to
# enable): with the RL-scale LR, the exp-advantage contrast, and the
# oracle penalty on wrong-way turns, the spin-bot should be starved
# without it -- and now that example scaling actually reaches the
# gradient (train.py SHAPE VS SCALE; it used to be a near-no-op), an
# untuned x0.1 decay is a much sharper knife than it ever was in a live
# run. Turn it on only if a run shows repetition degeneracy despite the
# new reward shape. The tracker and its t1 units stay regardless.
#
# 2026-08-11: that condition fired -- the aug6 11-epoch run's turn runs
# were 97% self-continuing and training DEEPENED the commitment (flip
# rate 0.10 -> 0.03) -- so run_weekend.train_one_epoch now passes
# novelty=True. The class default stays False (t1 asserts a default
# source must not decay).
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


def _make_noise_dir(prefix: str) -> Path:
    """A temp dir for this run's noised frame copies, DELETED AT
    INTERPRETER EXIT (atexit). The copies must outlive ``examples()``
    -- the collator re-opens the files at every training step -- so the
    dir can only die with the process. It used to be "left to OS tmp
    cleanup", which never runs on a long-lived box: the aug6 11-epoch
    run leaked ~6 GiB per train stage (~8k frames x 0.7 MB x 3 sources)
    into /tmp, ~65 GiB total. atexit covers normal exits and unhandled
    exceptions; a hard kill (OOM, SIGKILL) still leaks, so run_weekend
    additionally sweeps stale ``*_noise_*`` dirs between stages."""
    d = Path(tempfile.mkdtemp(prefix=prefix))
    atexit.register(shutil.rmtree, d, ignore_errors=True)
    return d


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
    epoch-mixture knob; the reward knobs are documented above (SHAPE/SCALE)
    and in TRAINING_GAME_TRACES.md. ``novelty`` gates the WIP boredom decay
    (off by default -- WIP block above)."""

    def __init__(
        self,
        path: str | Path,
        weight: float = 1.0,
        wrong_weight: float = WRONG_SPAN_WEIGHT,
        win_boost: float = 1.0,
        win_gamma: float = 0.95,
        novelty: bool = False,
        noise_strength: float = TRAINING_STRENGTH,
        noise_seed: int | None = None,
    ):
        super().__init__(path, noise_strength, noise_seed)
        self.name = f"game_{self.trace_dir.name}"
        self.weight = weight
        self.wrong_weight = wrong_weight
        self.win_boost = win_boost
        self.win_gamma = win_gamma
        self.novelty = novelty
        # One pre-pass over the records that will actually train (rated
        # records) feeds two corpus statistics:
        #   * ACTION BALANCE counts (TEMPORARY HACK -- screaming block
        #     above): equal total mass per move type;
        #   * the mean rating r_bar, the baseline of the exp-advantage
        #     scale (rating_advantage) -- "advantage" is only meaningful
        #     relative to what this corpus considers average.
        counts: dict[str, int] = {}
        rating_sum = 0.0
        n_rated = 0
        for _, _, _, meta in self._iter_records():
            if meta.get("rating") is None:
                continue
            rating_sum += float(meta["rating"])
            n_rated += 1
            action = meta.get("action")
            if action is not None:
                counts[action] = counts.get(action, 0) + 1
        self.rating_baseline = rating_sum / n_rated if n_rated else 0.0
        self.action_balance = action_balance_multipliers(counts)
        if n_rated:
            logger.info(
                "%s: rating baseline (r_bar) %.3f over %d rated record(s); "
                "action balance multipliers %s (counts %s)", self.name,
                self.rating_baseline, n_rated,
                {a: round(m, 3) for a, m in self.action_balance.items()},
                counts,
            )

    def examples(self) -> Iterator[TrainingExample]:
        # Fresh noise every run: an unseeded Random gives a new degradation
        # per training run; pass noise_seed for reproducibility. The temp
        # dir lives for the run and is removed at process exit
        # (_make_noise_dir).
        rng = random.Random(self.noise_seed)
        noise_dir = _make_noise_dir(f"{self.name}_noise_")
        novelty = NoveltyTracker() if self.novelty else None
        n_dropped = 0
        n_dropped_target = 0
        n_floored = 0
        n_yielded = 0
        verdicts = {"correct": 0, "neutral": 0, "wrong": 0, "unknown": 0}
        for lineno, messages, target_text, meta in self._iter_records():
            # The streak advances on EVERY move record -- even ones dropped
            # below -- because the move still happened in the game; only
            # the training weight is per-record.
            novelty_mult = novelty.multiplier(
                meta.get("session_id"), meta.get("game_index"),
                meta.get("action"),
            ) if novelty else 1.0
            if meta.get("rating") is None:
                n_dropped += 1
                continue
            if meta.get("target_missing") or meta.get("target_invalid"):
                n_dropped_target += 1
                continue

            action = meta.get("action")
            verdict = oracle_verdict(
                action, meta.get("oracle_move"),
                meta.get("oracle_rel_bearing"), meta.get("oracle_ray_hit"),
            )
            verdicts[verdict] += 1

            # SCALE (module docstring): reply-wide reward on
            # example_weight -- per-example normalization would cancel it
            # on the span weights (train.py, SHAPE VS SCALE).
            scale = example_scale(
                float(meta["rating"]), self.rating_baseline,
                bool(meta.get("game_won")), meta.get("moves_from_end"),
                self.win_boost, self.win_gamma,
            )
            scale *= self.action_balance.get(action, 1.0)
            scale *= novelty_mult
            if verdict == "wrong":
                # The rationalization of an oracle-wrong move is suspect
                # end to end -- damp its cloning AND its suppression.
                scale *= ORACLE_WRONG_SCALE
            if meta.get("oracle_ray_hit"):
                # Transition moment (TRANSITION_BOOST comment): the
                # gold is dead ahead and FORWARD gates the win.
                scale *= TRANSITION_BOOST
            if scale == 0.0:
                # Floored rating, no win boost: teaches NOTHING -- skip
                # the forward pass entirely rather than burn it on a
                # zero-scaled loss.
                n_floored += 1
                continue

            self._rewrite_frames(messages, rng, noise_dir, lineno)

            # SHAPE (module docstring): within-reply emphasis only.
            spans = build_span_weights(
                target_text,
                wrong_spans=list(meta.get("wrong_spans") or []),
                action=action,
                verdict=verdict,
                wrong_weight=self.wrong_weight,
            )

            yield TrainingExample(
                messages=messages,
                target_text=target_text,
                span_weights=spans,
                loss="ce",
                source=self.name,
                meta=meta,
                example_weight=scale,
            )
            n_yielded += 1
        logger.info(
            "%s: oracle verdicts %s over %d rated record(s)",
            self.name, verdicts, sum(verdicts.values()),
        )
        if verdicts["unknown"] == sum(verdicts.values()) and n_yielded:
            logger.info(
                "%s: no oracle meta anywhere -- pre-2026-08-05 corpus, "
                "move tokens graded by the analyst alone", self.name,
            )
        n_seen = n_dropped + n_dropped_target + n_floored + n_yielded
        if n_floored:
            logger.info(
                "%s: SKIPPED %d/%d record(s) at zero example weight "
                "(rating at/below the %.1f floor, no win boost).",
                self.name, n_floored, n_seen, RATING_FLOOR,
            )
        if n_dropped:
            logger.warning(
                "%s: DROPPED %d/%d record(s) with no parseable RATING "
                "(never train on a guessed reward).",
                self.name, n_dropped, n_seen,
            )
        if n_dropped_target:
            logger.warning(
                "%s: DROPPED %d/%d record(s) with a corrupted TARGET "
                "line -- missing or naming a nonexistent object (same "
                "contract as a missing RATING).",
                self.name, n_dropped_target, n_seen,
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
        noise_dir = _make_noise_dir(f"{self.name}_noise_")
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
        noise_dir = _make_noise_dir(f"{self.name}_noise_")
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
