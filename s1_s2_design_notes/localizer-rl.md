# Training the localizer with RL (2026-10-16)

The localizer is a regression head. RL needs a policy you can take a
log-probability of. The recipe: (1) make the localizer stochastic,
(2) roll out by sampling instead of taking the mean, (3) replay the
rollout teacher-forced with the sampled coordinates injected, (4) apply
the existing localizer loss with the sampled points as labels and the
advantage as the per-token weight. Almost all of the machinery exists.

## 1. Turn the localizer into a policy

`Localizer.forward` (neural_net/localizer.py) returns `coords [B, Q, 4]`
and `cell_logits [B, Q, 2, 256]`. The cell logits are already a
categorical distribution over the 16x16 grid: a policy as-is. The coord
head is deterministic; the standard fix for a continuous action is a
Gaussian around it:

    # policy over v; s stays deterministic (its ground truth is always known)
    mu = coords[..., 2:4]
    sigma = 0.02                      # fixed, or a learned log_sigma parameter
    v_sample = mu + sigma * torch.randn_like(mu)
    log_pi = -((v_sample - mu) ** 2).sum(-1) / (2 * sigma**2)   # + const

REINFORCE on that is `loss = -(A * log_pi).mean()`, which expands to
`A * ||v_sample - mu||^2 / (2 sigma^2)`. That is the L2 term in
`_optimizer_step` (neural_net/s2_pretraining_localizer.py,
`err = (pred - labels).pow(2)`), with the sample as the label and the
advantage as a weight. The one structural change to the loss is a
per-token weight tensor: supervised rows carry weight 1 and a
ground-truth label; RL rows carry weight `A` and a sampled label. The
cell head gets the same treatment: `A * cross_entropy(cell_logits,
sampled_cell)`. One loss function serves both.

Two notes on the advantage. Negative `A` means "push away from this
point", which is unstable with L2 (it diverges). The usual answers are
clipping `A` to `[-c, c]`, a PPO ratio clip, or dropping negative rows
entirely (reward-weighted regression / filtered cloning). And `A` must
be a baseline-subtracted quantity, not the raw reward, or every sample
gets pulled toward itself.

## 2. Rollout: sample instead of taking the mean

`GemmaS2.decode` (neural_net/s2.py) takes the mean coordinate every
token and feeds it back into the next token's embedding through
`coord_embed.set_pending(prev_coords, ...)`. For RL the rollout samples
`v` from the Gaussian (or a cell) at the tokens treated as actions and
feeds the sample forward as `prev_coords`. Everything downstream (the
LM's next tokens, and S1 if `[MOVE]` closes) sees the sampled point,
which is what makes the reward attributable to it. Record the sampled
coords per token. The `turns.jsonl` record written by `S1S2.generate`
already stores per-token `s`/`v` plus the S1 outcome (`collected`,
`exited`, `stopped_by`), so play logs are already trajectory records
with rewards; offline reward-weighted regression from them needs no
new rollout code.

Which tokens are actions has a large variance consequence:

- For a move reply, only the `v` at the token that closes `[MOVE]`
  reaches S1. Sample there, take the mean everywhere else: one Gaussian
  per reply, one reward. Lowest variance.
- For a question answered through the injected code (the hour task),
  every reply token's `v` conditions the answer. Sample at all of them
  with a shared reward, or use a group baseline.

## 3. Gradient pass: teacher-forced replay with the samples injected

`GemmaS2._forward` runs under `inference_mode`, so the rollout carries
no graph. The gradient pass is a separate teacher-forced forward, which
is how the existing text RL works (`rl_loss` in
neural_net/s2_coord_rl.py: one sampled reply, scored, then weighted CE
on that reply). The localizer version is the coord_tf suffix forward
(`_cached_reply_loss` in neural_net/s2_coord_train.py) with the sampled
coordinates in `embedder.set_pending(coords, positions)` on the reply
positions instead of jittered labels, the tap-layer hidden at those
positions as the queries, and the weighted loss from section 1. The
joint weekend trainer (neural_net/s2_weekend_train.py) builds exactly
this path for SFT; an RL family is that path with labels = samples and
weights = advantages.

If the LM is also being trained, the trajectory log-prob is the sum of
the token log-probs and the coordinate log-probs, and PPO/GRPO applies
to the sum unchanged. If the LM is frozen, the localizer must stay
inside the code distribution the LM learned to read; a small `sigma`
and a KL/L2 anchor to the SFT localizer's mean keep it there.

## 4. GRPO fits the existing batching

The localizer trainer's stage B repeats one prefix cache `k` times and
runs `k` suffixes. That is a group: one board, `K` sampled `v`s, `K`
rewards, advantage `(r_k - mean) / std`. No critic. For S1-in-the-loop
rewards each sample needs its own game copy from the same start:
`game_from_settings_dict(start_settings)` (the same rebuild
`replay_frames` in neural_net/loop.py uses) plus
`run_s1_window(s1, game, v_k, ...)` gives `collected` / `exited` /
`n_steps` per sample deterministically. The cost is the 10 s wall-clock
window: `K = 8` is 80 s per board. Training rollouts pass a shorter
`seconds` (already a parameter of `run_s1_window`) and score on
`collected`, exit, or distance-to-target after the window; the eval
must use the same cap so the numbers are comparable.

## 5. Where only a score exists

`s` never needs RL (the body position is always known), and `v` has a
ground truth for every "look at X" task, so SFT wins there. The
score-only cases are the ones where S1's dynamics or the LM's
downstream behaviour define success:

- Where to aim so S1 actually eats the gold. The gold center is known,
  but the best `v` for the controller can be offset (approach angle,
  wall clearance, the noop-stop rule). Reward: `collected`, or
  `-n_steps` to contact.
- Where to aim to walk out. The opening's center is a ground truth; the
  point that gets the disc through under the window budget may not be.
  Reward: `exited`, `gold_remaining == 0` at exit.
- Multi-turn target sequencing. Which gold first when several remain is
  a choice, not a label; reward is golds eaten per turn or turns to
  clear the room. (Target text is the LM's action; the `v` that follows
  is the localizer's.)
- Answering through the code. Hour/direction correctness when the only
  path from picture to answer is the injected `v`; reward is
  `hour_reward` / exact match. Ground truth for `v` exists here, so this
  is a consistency signal more than a necessity.

Everything else in the current task list (look lines, `[END_GAME]` on a
sealed board, the verdict lines) is text with a known answer; the
weekend joint trainer handles it with teacher forcing.
