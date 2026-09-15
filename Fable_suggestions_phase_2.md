# Phase 2: memory that earns its keep

Notes from the 2026-09-15 discussion. Not a plan to execute in order —
a set of constraints and literature pointers for the memory work, now
that play quality is “good enough” (the ~7/8 sealed-room win rate) and
the point of the project is no longer “be good at this game.”

The point is a system that **plays, reflects, and then either writes a
tip that fixes the failure or writes a recipe that generates many
similar situations** for the night-time training cycle. NAMS, except
for loading the scene prompt and the session scratchpad, has not been
useful. This branch starts by turning the rest of retrieval off.

## Why this is the right object

Getting a strong policy in one room could have been done with a much
smaller net and less training. Multi-room play is where memory is
actually load-bearing. Building memory skill in one room first is the
right order **if** the room is set up so that a note is the only path
to the answer. Otherwise the model will correctly ignore memory.

## Unfair advantage: writes are verifiable here

Most memory-agent papers grade memory only indirectly (“did downstream
QA improve”). This game engine has complete ground truth. A note that
says “gold at (3,7), opening on the north wall” can be graded exactly,
the same way `oracle_verdict` grades moves and the WRONG-span harness
grades analyst claims.

Make **memory accuracy a first-class graded skill before memory
usefulness**: an oracle that diffs each written note against engine
state gives per-note rewards without the analyst-saturation problem
that plagued move ratings. That slots into the existing
offline-REINFORCE machinery.

Papers that RL-train memory ops
([Memory-R1](https://arxiv.org/abs/2508.19828) — ADD / UPDATE / DELETE
/ NOOP as trained actions, notable for working with ~150 pairs; MEM1 —
constant-size self-consolidated state) have to reward through final
task outcome. We can reward the write itself.

## Pitfall: memory theater in a fully-observable room

In one fully-observable room the current frame already shows
everything, so a note is never load-bearing. The model will learn to
ignore memory while still producing plausible-looking notes.

Mitigations:

- Design tasks where the note is the **only** path to the answer: the
  agent turns away and is later asked what it saw; frames are
  degraded (we already noise them); objects leave view. Partial
  observability is what makes memory causal, and it can be injected
  without leaving the room.
- Measure **counterfactually**: same checkpoint, same sealed seeds,
  notes ablated vs. present. If ablation does not hurt, the memory
  did nothing. This is the distill-run lesson again: holdout CE fell,
  live play did not move. Grade tips and notes in action space (win
  rate, on-ray FORWARD, question accuracy), never by how sensible
  they read.

## Tips and notes are different objects

Keep two channels with different lifetimes and different verification.
They fail differently.

- **Episodic notes** (this board’s golds / openings): per-game
  lifetime, engine-oracle-gradable, high volume.
- **Semantic tips** (cross-game policy advice): long lifetime, **not**
  oracle-gradable. The only honest test is a sealed A/B smoke, **one
  tip at a time**. The one-knob rule applies to self-written tips with
  full force; a batch of five simultaneous self-tips is
  uninterpretable.

NAMS conflates these. The memory that helps agents in the literature
is **self-written, task-structured, and loaded exactly** (Voyager’s
skill library, Agent Workflow Memory, Dynamic Cheatsheet), not
similarity-retrieved. Vector retrieval returns the nearest thing, not
the right thing — the same no-fuzzy-fallbacks rule, discovered
independently by the field. When we get to multi-room, the structured
answer is a spatial graph the agent builds as it explores (AriGraph
did this for text games), not embeddings.

Notebook conversations: notes written in a chatty human-dialogue
register may be unusable by the terse game-mode prompts later. Keep
the note format machine-parseable from day one, both so the oracle can
grade it and so the format survives the transfer back to the harness.

## Night-time cycle: failure-driven task synthesis

The “recipe for generating lots of similar situations to the failure”
is now a real literature. 2026 closed-loop versions:

- [SENTINEL](https://arxiv.org/abs/2606.12908) — Controller diagnoses
  failure patterns → Proposer synthesizes targeted tasks → Solver RLs
  on them.
- [TRACE](https://arxiv.org/abs/2604.05336) — contrastive analysis of
  success vs. failure trajectories → synthesized capability-targeted
  environments.
- CoEvolve — also tracks *forgetting* and *unstable-boundary* tasks
  (useful given our regression-guard history).

Older RL lineage: Unsupervised Environment Design (PAIRED, PLR,
ACCEL).

**Both SENTINEL and TRACE train only the solver / agent.** The
failure-analysis and task-generation roles are separate **frozen** LLM
helpers, often a stronger model than the one being trained. SENTINEL’s
Solver is Qwen3-4B-Thinking (GRPO); Controller and Proposer sit
outside the loop (they also host a much larger Qwen3-235B just for the
user simulator). TRACE trains LoRA adapters on the base agent; the
“analysis agent” and “generation agent” are separate helpers.

That is a real difference from this repo: player and analyst share one
network. A self-contained reflect-and-propose loop is harder than
these papers’ setup. Do not assume the same weights that fail at play
will write a good recipe for the failure.

Distilled pitfalls:

1. **Verify the recipe reproduces the failure before spending GPU.**
   Generate candidate boards from the recipe, run the current
   checkpoint, require an elevated failure rate. Training on
   situations the model already handles is the wasted-weekend failure
   mode, and a recipe-writing model will happily produce
   plausible-but-easy boards.
2. **Keep solvability checks.** A board generator steered toward
   failure drifts toward unsolvable boards (POET needed an explicit
   minimal criterion). The engine can verify solvability cheaply.
3. **Distribution collapse.** Training only on adversarial boards
   degrades general play. The KD-anchor / replay mixture discipline
   already covers this; do not drop it for the night cycle.

Also worth stealing: the “sleep-time compute” framing (Letta, 2025) —
the night cycle has two possible products: weight updates from
targeted datagen, and **memory consolidation** (compress many episodic
notes into few semantic tips, then A/B the tips). The second is much
cheaper and might be the right first thing to automate, because it
exercises the whole reflect-and-write loop without a training run at
risk.

## Tension: tip-writing is a prompt-growth machine

We just spent weeks trying to shrink the prompt. A tip-writing loop
grows it. Set a token budget and an eviction rule from the start: tips
pay rent with measured smoke deltas or get consolidated away.
Otherwise the memory project quietly regenerates the bloat the distill
project was trying to remove.

The reflection loop also needs a supply of failures. At 7/8 in one
room, pure-play failures are scarce. That is fine: the interesting
failures are now *memory* failures (mis-recorded gold, stale notes,
unretrieved facts), and an engine oracle makes those cheap to detect
and mine. That is a coherent research object on its own, and it is
the one that matters for multi-room.

## Starting stance on this branch

- Old prompt (core-tip dump still loaded from NAMS).
- Scratchpad (`[REMEMBER]` / `SessionNote`) stays.
- Last-K recency is on everywhere: `RECENT_MESSAGES_WINDOW=7` and
  `PLAYER_RECENT_MESSAGES_WINDOW=7`. Verbatim last 7 session messages
  in play, discuss, and the analyst. Older messages drop off the
  cliff — no digest, no attention score. See below.
- `[SEARCH]` hits stay live (the tool the standing weights already
  know).
- `get_context` semantic dump and the privileged semantic-model dump
  stay off. Flip those back with `NAMS_RETRIEVAL=1`.
- Debrief `[SHOW]` (exact cursor on recorded play) and exact
  session-trace Cypher stay — those are not similarity search.
- Next: notebook conversations with the standing weights, then prompt
  / interaction changes from what those conversations actually do.
  Custom night-time datagen only later.

## Last-K now; per-token attention later (not implemented)

The live policy is the old sliding window: keep the last K=7 messages
verbatim, drop everything older. `_summarize_actions` still exists
(`agent/modes.py`) but it is **not** in the player prompt — only the
NAMS turn-trace outcome string and the debrief trace block use it.

A better eviction rule was discussed and is **not built**: score each
in-context message by (attention it received) + recency, knock it out
when the score is too low. Closest published matches:

- **H2O** (Zhang et al., NeurIPS 2023, “Heavy-Hitter Oracle”). Keep a
  running sum of attention each token has *received*; when the KV
  budget is hit, evict the lowest scorer, always retaining a recency
  window next to the heavy hitters. Training-free. Models: OPT-6.7B /
  30B, LLaMA-1-7B / 13B, GPT-NeoX-20B. Headline: ~20% of the full KV
  cache loses almost nothing.
- **TOVA** (Oren et al., 2024, “Transformers are Multi-State RNNs”).
  Same problem, cleaner rule: at each decode step drop the token
  whose key gets the lowest attention from the *current query only*
  — no accumulation, so no seniority bias (H2O lets old tokens pile
  up score just by having been around). Models: LLaMA-2-7B, Mistral-7B,
  Yi-6B. ~1/8 of the full state matched full-context performance.

Both are inference-only bolt-ons from just before flash attention
became universal. Flash / SDPA never materialize the attention matrix,
so getting these scores on our stack means `attn_implementation="eager"`
(slower, more VRAM at ~8k) or a recompute pass. Attention is also a
biased proxy: attention sinks collect weight regardless of content;
low attention *now* is not low value *later* (the target-naming
message ignored for ten turns). That last failure is exactly what
the notepad is for.

The research branch is alive but moved *inside* the model: GQA, MLA,
hybrid local/global attention (Gemma 2/3’s 5:1), trained sparse
attention (DeepSeek NSA, Kimi MoBA, DeepSeek V3.2). Production
serving mostly chose lossless KV management (PagedAttention, prefix
cache) over lossy score eviction.

If we ever prototype the message-level version, cheaper than eager
attention: use behavioral signals already in the run logs (`[SEARCH]`
hit, reply reference, `[REMEMBER]` derived from it) as the “was
attended” score, plus recency decay. Do not implement any of that
until last-K=7 has been lived with.
