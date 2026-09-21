# Alternative game environments for a Gemma4-12B-class VLM

Scratchpad for the 2026-09-16 survey and the later ViGaL notebook
excursion. Question: which games has anyone gotten an out-of-the-box
OR finetuned ~7–12B VLM to play WELL, with visual input, simple
discrete-ish actions, the whole 2D scene in play (no gridworld
navigation), real-time preferred?

Folder is isolated: `vigal.ipynb` / `vigal_io.py` / `vigal_setup.sh`
are not wired through `agent/` / NAMS / Gemma.

## DECISIONS (2026-09-19, after the ViGaL notebook)

**Do not use ViGaL-7B directly.** We may switch to its *base*
(Qwen2.5-VL-7B-Instruct) for a next step, not the Snake/Rotation RLOO
checkpoint.

Reasons:

1. It was trained on **settings + image**, not image alone. The
   public default (paper Appendix A.2 + `train_snake.sh`) dumps live
   coordinates / last action / snake bodies into the text. Tab 7(e)
   was the opposite ablation (text-only, no image). Vision-only — what
   `vigal.ipynb` ran — is unpublished. The image probably mostly
   helped when a crash was imminent but not obvious from the text.
2. It is *ok* at interpreting the visual scene and recording *close*
   to where everything is. It is not *good* at it. Even the faulty
   pretrained game_saddle system is better at this, for our game.
3. Additional finetuning of ViGaL-7B is possible. Prefer finetuning
   something designed for *our* game (an S1 / S2 system). Reasoning
   gains can come from merely using Gemma4.

Good excursion. Nobody seems to have the small system we want and can
picture. That is a good sign. These guys started from a very similar
setup, then stopped short and published.

What we actually ran: static Q&A in `vigal.ipynb` (New board 10×10
Snake, 512² pixels only, official A.2 *instructions* minus the state
dump, oracle printed for the human only). Renderer matched their
canvas (`snake-game.js`: light grid, olive/teal, y-up) except apples
as full-cell red squares (their JS uses circles; paper/screenshots
are squares). General Ask box: no board. Atan2 probe (our bearing
recipe, x-diff first) — even with a worked example it fell back to
textbook `atan2(y, x)` and the clock hour was right only by a
cancelled arithmetic error. Consistent with (2).

## DECISIONS (2026-09-16, end of survey)

Verdicts on the candidates, in priority order. **Superseded for
ViGaL-7B on 2026-09-19** (see above): do not use that checkpoint.

1. **ViGaL was the #1 option** (survey-time), with a later expansion
   for full 2D motion instead of a gridworld. 2026-09-19: checkpoint
   rejected; base model still a possible next step.
   - Needs to be tested on its native games first (Snake via
     SnakeBench + the Rotation task; weights
     https://huggingface.co/yunfeixie/ViGaL-7B, training scripts at
     https://github.com/yunfeixie233/ViGaL, data
     https://huggingface.co/datasets/yunfeixie/vigal_data).
   - Just for fun: test it on OUR current game, maybe with simpler
     motion types (its native action vocabulary is 4-way grid moves;
     counted CLOCK/ANTICLOCK/FORWARD is out of distribution).
   - The expansion path: swap SnakeBench's coarse grid for a
     continuous-motion 2D env (ours, or a modified snake) and rerun
     their RLOO recipe — that is the "full 2D motion" upgrade.
2. **Odysseus needs a quality examination.** Weights (both
   click-through gated on HF):
   - https://huggingface.co/Odysseus-Project/Odysseus
   - https://huggingface.co/Odysseus-Project/Odysseus-Zero
   - Collection: https://huggingface.co/collections/Odysseus-Project/odysseus
   - Project page: odysseus-project.github.io (training-code repo not
     surfaced by search; check there).
   Examine: does the released checkpoint reproduce the reported level
   progress; does it still talk outside Mario; how narrow is the
   run-right-and-survive skill.
3. **The labeled-grid Doom approach (DoomVLM) is worth keeping.**
   The harness converts aiming into reading — but those networks
   *might* be trainable to use memory and master more enemy types /
   a wider world. I.e., use the grid crutch to bootstrap, then grow
   the skill set under training. https://github.com/Felliks/DoomVLM
4. **The remaining two hobbyist approaches are rejected** (Tesserack
   Pokemon Red, STS2 agent): not pixel-input (RAM reads / JSON
   state), and they do not represent anything like real motor skills
   or visual comprehension.

## Headline finding

Nobody has an out-of-the-box 7–30B VLM that is good at a continuous-2D
real-time game. Even frontier models are near-zero without heavy
scaffolds (VideoGameBench: best overall 0.48%, Doom II 0% for every
model including paused "Lite" mode; Atari-GPT: GPT-4o 23% of human,
Pong BELOW random for all models). Every success story at our model
class is a finetune, and the strongest finetuned results are on games
with discrete boards, not continuous scenes. The continuous-2D
real-time + small-VLM combination that game_saddle occupies is mostly
unclaimed territory — which is validation of the project, and also why
the shopping list below involves compromises.

Gemma 3 27B appears directly in two benchmarks:
- lmgame-bench Super Mario Bros: 786 ± 463 — BELOW the random-agent
  baseline (987 ± 415).
- StarDojo (Stardew Valley): 4.0% success, worst of the evaluated
  open models.

## The five candidates

### 1. Minecraft — JARVIS-VLA / OpenHA (CraftJarvis, PKU)

The strongest "same model class, finetuned, actually good" result.

- Model: Qwen2-VL-7B post-trained (SFT on 3.78M-row open dataset,
  ~800k trajectories + vision-language tasks). Follows instructions on
  1,000+ atomic tasks (craft/smelt/cook/mine/kill), ~40% over the
  previous best baseline (VPT-derived). Newer Sept-2025 OpenHA family
  adds Chain-of-Action unified models.
- Weights: open — `CraftJarvis/JarvisVLA-Qwen2-VL-7B`,
  `CraftJarvis/minecraft-openha-qwen2vl-7b-2509` (HF; textvla variant
  is click-through gated).
- Env install: `github.com/CraftJarvis/JarvisVLA` (conda + openjdk8 +
  pip -e; vLLM serving for rollouts). Related MineStudio toolkit.
- Fit: real-time YES, visual YES, actions = keyboard+mouse (low-level).
  MISS: 3D, not 2D. Otherwise the best evidence that a 7B VLM can be
  made genuinely competent at a visually rich real-time game.
- Papers: arXiv 2503.16365 (JARVIS-VLA), 2509.13347 (OpenHA).

### 2. VLM-Gym / G1 (2048, Shisen-Sho, Shisen-Sho-CIFAR10, Swap)

The strongest "small model BEATS frontier after RL" result.

- Model: Qwen2.5-VL-7B. G0 = pure RL (GRPO via EasyR1); G1 = perception
  cold-start + RL. G1-7B beats Claude-3.7-Sonnet-Thinking on ALL four
  games.
- Env: `github.com/chenllliang/G1` — parallel gym built specifically
  for VLM RL (unified obs/action, adjustable difficulty, parallel
  rollout for GRPO). Easiest-to-install item on this list. Training
  scripts verified at 4×80G (QLoRA on 96 GiB would need porting).
- Weights: training code + env released; trained G1 checkpoints not
  clearly on HF (check repo issues before counting on them).
- Fit: visual YES, simple actions YES, install YES. MISS: these are
  tile/board games — discrete cell positions, exactly the "artificially
  limited positions" line. Swap is match-3; Shisen-Sho-CIFAR10 at least
  forces natural-image perception per tile. Turn-based.
- Paper: arXiv 2505.13426.

### 3. Snake — ViGaL ("Play to Generalize")

Small-model-beats-frontier, on a game with an avatar.

- Model: Qwen2.5-VL-7B RL-tuned (RLOO) on Snake (SnakeBench data
  engine) + a rotation-estimation game. Beats proprietary models
  head-to-head at Snake (10-game direct matches); transfers zero-shot
  to Atari-GPT games and to math/reasoning benchmarks.
- Env: SnakeBench (`github.com/gkamradt/SnakeBench` lineage) — two-snake
  competitive, simple 4 actions.
- Weights: paper open-sources pipeline; checkpoint availability on HF
  should be verified before install (project: yunfeixie233.github.io/ViGaL).
- Fit: visual YES, avatar + whole scene YES-ish, real-time-ish
  (turn-stepped). MISS: movement is on a coarse grid — borderline
  against the gridworld line, though closer to game_saddle than 2048.
- Paper: arXiv 2506.08011.

### 4. Super Mario Bros — lmgame-bench / GamingAgent (LMSYS)

The environment that fits the constraints best; the model class that
does not (yet).

- Env: `github.com/lmgame-org/GamingAgent` — pip-installable harness
  over SMB, Tetris, Sokoban, 2048, Candy Crush, Ace Attorney. SMB is
  the real-time entry: continuous side-scroller, whole scene in play,
  simple discrete actions. Actively maintained; eval pipeline notebooks.
- Results (no finetunes offered): only frontier reasoning models beat
  random meaningfully. o3+harness 3445 vs random 987; GPT-4.1 1991;
  Gemma-3-27B 786 (below random). lmgame's own follow-up did multi-turn
  RL (PPO) on Qwen2.5-7B for Sokoban/Tetris with cross-game transfer —
  text observations, but proves the harness supports training loops.
- Fit: env is the closest match on this list (2D continuous scene,
  real-time, tiny action set, trivially installable). To be GOOD here
  at 12B you would be doing the finetuning yourself — i.e., the month
  of work the pivot is trying to avoid.
- Paper: arXiv 2505.15146.

### 5. Stardew Valley — StarDojo

The best-engineered 2D open-world env with published Gemma baselines;
nobody is good at it.

- Env: `github.com/StarDojo2025/stardojo` (MIT). 1,000 tasks (100-task
  Lite), farming/crafting/exploration/combat/social. Unified API — no
  kb/mouse emulation — parallel instances, all OSes. 2D top-down,
  whole scene, real-time (pausable).
- Results: GPT-4.1 12.7%, Gemini 2.5 Pro / Claude 3.7 >10%, all open
  models <8%, Gemma 3 27B 4.0%. Near-zero on medium/hard tasks.
- Fit: env checks every constraint box except "someone is good at it."
  Include because it is the most installable serious 2D world, and the
  gap is the research opportunity.
- Paper: arXiv 2507.07445.

## Near-misses / anti-list (why not included)

- **VideoGameBench** (Doom II, Kirby, Zelda:LA, Pokémon Crystal, DOS
  games; arXiv 2505.18134): all models ≈0%, including QwenVL-2.5-7B/32B
  at exactly 0% everywhere, EVEN in the paused Lite mode. Great harness
  (PyBoy/DOSBox), devastating results. The clearest evidence for the
  headline finding.
- **Atari-GPT** (arXiv 2408.15950): zero-shot frontier ≈8–23% of human;
  Pong below random. No open finetune fixes it at our class (ViGaL
  transfers there but is not "good").
- **Pokémon Red/Blue/Crystal** (Claude/Gemini Plays Pokémon, PokeAgent):
  completions exist but only with frontier models + custom pathfinding
  tools + weeks of wall-clock; VideoGameBench's stricter ruleset drops
  the same games to ~0%. Turn-based; heavy scaffold; not model skill.
- **Crafter / Craftax / NetHack / MiniHack (BALROG)**: explicitly
  excluded — gridworlds.
- **G1-style board games in lmgame (Tetris, Candy Crush, 2048)**:
  discrete boards; also generally weak results.
- **VPT / STEVE-1 (Minecraft)**: strong players but not VLM-class
  architectures (behavior-cloned policies), excluded by "comparable
  model."
- **Street Fighter III (LLM Colosseum), PokéLLMon**: text observations,
  fail the visual-input requirement.
- **Cradle (RDR2, Stardew via GPT-4o)**: frontier-only, screenshots +
  kb/mouse, slow and mediocre; superseded for our purpose by StarDojo.
- **Game-RL / GameQA (OpenMOSS)**: 30 games but as VQA reasoning data,
  not interactive play. Open Qwen2.5-VL-7B weights exist and transfer
  well; a useful data recipe, not a playable-agent result.

## Second pass (2026-09-16 PM): winnow by "must still competently talk"

New criterion: the finetuned model must retain language/reasoning —
narrate decisions, keep general benchmarks. Also: micro-decisions in a
visual env are the requirement; macro-decision systems are noted but
not candidates (we build task-selection ON TOP of motor control, not
instead of it).

### Winnowed ranking

1. **Odysseus (arXiv 2605.00347, COLM 2026) — best overall fit;
   WEIGHTS CONFIRMED RELEASED.**
   - Who: Princeton Language and Intelligence (Shi / Li / Liang et
     al.; senior authors Danqi Chen, Karthik Narasimhan, Chi Jin), w/
     Fudan + Tsinghua.
   - What: RL on **Super Mario Land** (PyBoy-class GB emulator) —
     continuous-2D side-scroller, 100+ turn horizons, base
     **Qwen3-VL-8B-Instruct**, FULL-model training (vision encoder +
     projector + LM backbone).
   - Interface: observation = CURRENT FRAME ONLY + fixed text prompt
     (deliberately no history scaffolding). Output per turn: describe
     the screen -> step-by-step CoT -> up to two buttons from
     a/b/up/down/left/right/noop. Frame-skip: a jump action advances
     15 emulator frames, others 5 (~real-time cadence without
     latency pressure).
   - Recipe: (1) light SFT — ~5,000 frames sampled from two YouTube
     walkthrough videos across 10 levels, with **GPT-o3 as teacher**
     writing the CoT+action labels (no human action annotation);
     SFT via the Qwen3-VL repo. (2) Multi-task RL on 5 levels —
     adapted PPO with a **lightweight turn-level critic** (not a
     second large model) + positive-advantage filtering; critic-free
     GRPO / Reinforce++ were unstable in this regime. Level sampling
     inverse-weighted by trajectory count. Modified veRL. "Tens of
     millions of interaction samples."
   - Results: ~6x base-model game progress, ~5x GPT-5.4, ~3x
     GLM-4.6V on the first five levels. Generalization: +32.2% on
     off-policy states of training levels, +41.5% on 5 HELD-OUT
     levels, +23.1% cross-game. **General benchmarks (MMMU,
     MathVision, RealWorldQA) retained vs base.** CoT examples show
     real spatial reasoning (jump timing vs enemy distance; base
     model mistakes background art for an enterable pipe).
   - Weights: HF collection `Odysseus-Project/odysseus` —
     https://huggingface.co/Odysseus-Project/Odysseus (9B) and
     https://huggingface.co/Odysseus-Project/Odysseus-Zero (9B,
     RL-from-base ablation). Both click-through gated (agree to share
     contact), updated 2026-05-25. Project page:
     odysseus-project.github.io. A dedicated training-code GitHub repo
     did NOT surface in search — check the project page directly.
   - This is micro-decision + visual + 2D + talks + retention-checked:
     every box.

2. **ViGaL — verified pass, weights CONFIRMED available.**
   - Weights: **https://huggingface.co/yunfeixie/ViGaL-7B** (BF16
     safetensors, 8B, base Qwen2.5-VL-7B-Instruct, ungated). Obscure
     (~7 downloads/mo) but public. Training data:
     https://huggingface.co/datasets/yunfeixie/vigal_data (repo README
     still says "coming soon" for the data link; the HF dataset page
     exists — rotation subfolder confirmed).
   - Training scripts OPEN: github.com/yunfeixie233/ViGaL —
     `examples/scripts/train_snake.sh`, `train_rotation.sh`,
     `train_snake_rotation.sh`. Stack: OpenRLHF / MM-Eureka codebase,
     RLOO, rule-based format+accuracy rewards, batch 128, lr 1e-6,
     rollout temp 1.0, **6x A100-80G**, 36K samples per game "to
     convergence". Wall-clock duration NOT stated anywhere found.
   - Talks: play format is CoT then move ("Reasoning: ... Action: 1",
     `<best_answer>RIGHT</best_answer>`); general-VLM suite (MMBench
     class, MME 1685, OCR) retained vs base — the one paper that did
     our kind of retention diligence.
   - Granularity CORRECTION for the record: Rotation targets are ONLY
     {CW 90, CCW 90, 180} posed as a short choice list; the 30-degree
     steps are initial-pose jitter, never the predicted quantity. No
     fine angle regression anywhere. Snake = SnakeBench 10x10 board
     (configurable w/h/apples/max_rounds; ViGaL adds image obs to
     SnakeBench's text-only states).

3. **TiG (Tencent, Honor of Kings)** — talk-retention by construction
   ("RL as language modeling"); retained text/math/QA verified; Qwen3-14B
   beats its DeepSeek-R1 teacher on action accuracy (90.9% vs 86.7%).
   NOT a candidate: JSON state input (not frames), 40 MACRO actions
   (not micro-decisions), checkpoints unreleased (hok_env repo is the
   old RL env). Kept as the design-philosophy reference for building
   reasoning on top of motor control later.

4. **G1/VLM-Gym** — thinks in text during play, but no general-benchmark
   retention eval; board games. Env still the easiest RL testbed.

5. **JARVIS-VLA — demoted.** 51 action tokens are APPENDED to the vocab
   (not a mute policy head), and their ActVLP stage post-trains on
   Minecraft VQA/grounding BEFORE action cloning — a cousin of our
   "protect the understanding first". But: final stage is pure behavior
   cloning, no KD anchor, no drift gates, no general-benchmark eval of
   the final VLA, and the published play format is instruction ->
   observation -> action-token blocks with no reasoning narration.
   "Competently talks" unverified and probably degraded.

### Community trawl (HF / ModelScope / Bilibili-adjacent)

Searched ModelScope + Chinese tech press + Bilibili tutorial ecosystem
+ HF collections. Result: NO hidden community "finetuned it to play X
and released the LoRA" gems found beyond the academic repos above.
What exists instead:

- **MagicGUI** (Fudan NLP + Honor, 2025): Qwen2-VL-7B, CPT + RFT, open
  weights on HF/ModelScope — but a phone-GUI agent (tap/scroll/drag),
  not a game player. Closest Chinese-community analog in spirit.
- **Qwen3-VL official cookbooks / computer-use demos**: zero-shot GUI
  and game-screen "observe->decide->act" demos (PyAutoGUI wrappers);
  no game finetunes, no released game LoRAs.
- **LLaMA-Factory / ms-swift Bilibili tutorial ecosystem**: abundant
  Qwen-VL finetuning tutorials; nothing game-play-specific with
  released weights found.
- **lmgame/GamingAgent** now integrates stable-retro (Genesis/SNES
  etc. via ROMs) — enlarging the env library, still no finetunes.
- **latent-bridge-games** (19PINE-AI): fast/slow frozen-VLM bridge on
  Atari (MiniCPM-o 4.5 + Qwen3-VL-8B-Thinking, 33M trained bridge) —
  architecture experiment for real-time latency, not a skill finetune.

The takeaway from the trawl: the open ecosystem's energy at our model
class went to GUI agents (screens with buttons), not games. The
game-playing successes remain the academic RL projects — and Odysseus
is the one that matches our constraints nearly exactly.

### Hobbyist "look what I did" pass (2026-09-16, third sweep)

Explicitly hunting forum/creator demos, not checkpoints. Found the
needle:

- **DoomVLM** (r/LocalLLaMA, open-sourced at
  https://github.com/Felliks/DoomVLM after the demo post blew up).
  **Qwen 3.5 0.8B — out of the box, no RL, no finetune — playing
  ViZDoom** and actually getting kills on basic scenarios. The trick
  is pure harness engineering: screenshot -> draw a NUMBERED COLUMN
  GRID on the frame -> two tools only, `shoot(column)` and
  `move(direction)`, with tool_choice=required. 11 solo scenarios,
  4 deathmatch maps, benchmark mode (identical conditions per model)
  and arena mode (simultaneous; faster inference = more turns —
  latency is part of fitness). Any OpenAI-compatible endpoint
  (LM Studio / Ollama / vLLM). ~10 s/step on an M1 Mac.
  Lessons for us: (1) the grid overlay converts spatial reasoning
  into symbol lookup — the model never estimates angles, it reads
  them; this is the "privileged geometry in the prompt" idea taken
  to its extreme, and it works for a 0.8B. (2) A commenter reports
  SFT'ing VLMs on ViZDoom since LLaVA: easy to reach competence on
  basic scenarios, "never got very good at long episode performance"
  — the same long-horizon wall Odysseus built the turn-level critic
  for.
- **Tesserack** (r/LocalLLaMA,
  https://github.com/sidmohan0/tesserack): Qwen 2.5 1.5B via WebLLM
  + a small TF.js policy net playing Pokemon Red fully client-side
  in the browser (binjgb WASM emulator, RAM-read ground truth).
  Architecture is admittedly "wonky" per the author; notable as
  infrastructure, not as skill.
- **Slay the Spire 2 agent** (r/LocalLLaMA): Qwen3.5-27B local via
  KoboldCPP against a community REST-API mod. Text-state, not visual
  — but the lessons transfer: state-based tool routing (expose 1-3
  tools relevant to the current phase, not 20) cut hallucinated
  calls dramatically; ~88% action success; beat the Act 1 boss.
- Adjacent, not an agent: a Flappy Bird DIFFUSION WORLD MODEL
  running 30 FPS local/in-browser (njkumar.com) — the community is
  also building the environments themselves now.

Pattern across all hobbyist wins: nobody makes the small model
smarter; they make the INTERFACE dumber (grid overlays, tool_choice
constraints, phase-scoped tools). Out-of-the-box competence at a
visual game appears exactly when the harness converts perception
into reading and action selection into a 2-way choice.

Follow-up on the DoomVLM SFT commenter (2026-09-16): anonymous
r/LocalLLaMA user, hobbyist scale — "on and off since llava first
came out," SFT "with simple datasets," gave up before getting
GRPO-era RL working with VLMs. Only artifact is one Google Colab
notebook (LLaVA/ViZDoom SFT):
https://colab.research.google.com/drive/1HdxbV_X2dDp93FaktqcwpilqedXBcAIa
No HF models/datasets, no GitHub repo surfaced. His negative result
(SFT -> basic scenarios easy, long episodes never) is the field's
story in miniature; the piece he was missing (working VLM RL at
long horizon) is exactly what Odysseus published.

### Odysseus: competence-vs-spam analysis (2026-09-16)

Reward is dense forward progress, r_t = x_{t+1} - x_t, and death ends
the trajectory — so "spam a plausible move" (the classic hold-right
Mario policy, the analog of our 7/8 wander-and-eat) is directly
punished: right-spam dies at the first pipe+enemy setpiece. Their
controlled ablation scenario (W1L1: tall pipe + two approaching
enemies) is chosen precisely because naive progress-chasing dies
there. Qualitative traces show the base model doing exactly the spam
thing (keeps outputting 'right' because it misjudges enemy distance;
dies) while Odysseus times the jump. Critic-free methods (GRPO,
Reinforce++, outcome AND process rewards) failed to make consistent
multi-step progress — the env genuinely resists cheap policies; only
PPO + turn-level critic + positive-advantage filtering worked.
Caveats for honesty: the learned skill is narrow ("run right and
survive" — no coins/secrets/multi-objective play), reported wins are
level-PROGRESS multiples (3x-10x base), not consistent full-game
completion, and CoT quality is evidenced anecdotally, not audited
systematically. Verdict: true motor competence at the one skill our
7/8 agent lacks (the jump-timing analog of on-ray FORWARD), obtained
with ~5k SFT frames + dense-reward PPO. Minimal harness (current
frame only, no overlays, no tools) — the anti-DoomVLM.

## Reading of the field for the pivot decision

1. The exact niche (continuous 2D, real-time, small VLM, actually
   good) is empty. game_saddle's 7/8 sealed-room result may be closer
   to the frontier of that niche than anything on this list.
2. Where small VLMs win, the recipe is always: parallel env + verifiable
   reward + GRPO/PPO-family RL, with a perception cold-start (G1) or
   large SFT corpus (JARVIS-VLA) first. Same shape as our pipeline.
3. If the goal is "install someone else's working env + weights and
   play": JARVIS-VLA is the only turnkey good player (3D caveat) —
   UNLESS Odysseus released checkpoints (second pass, above), which
   would supersede it on every axis we care about.
4. If the goal is "install an env and test/finetune our own Gemma":
   VLM-Gym (easiest), then GamingAgent/SMB (best fit), then StarDojo
   (richest world). Second pass adds: the Odysseus recipe (SFT from
   walkthrough videos -> PPO with turn-level critic) is the closest
   published blueprint to "our pipeline, on Mario."
