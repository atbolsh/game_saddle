# S1 / S2 — recorded instructions (2026-09-21)

Recorded as given. No design, no log reading, no code. The v / v_bar
split is intentional; do not argue against it.

## What he wants

Change the architecture. Frustrated by the repeated mistakes this
system makes.

1. System 1 / System 2 division between the **decider** and the
   **actor** (the actor running at higher frequency).

2. The S1 net is much smaller. It probably shares a vision tower with
   the actor. It will receive a set of coordinates in floating point.
   If the coordinates are inside the agent, it will not act. If the
   coordinates are outside of the agent, it will take the necessary
   steps to get there (current Oracle correct moves). It has no
   capabilities (yet) beyond CLOCK / ANTICLOCK / FORWARD / do nothing.
   Train it using SFT, and quickly.

3. The S2 net is Gemma 4 (or equivalent; see questions below), but at
   the final layer, in addition to logits, it also emits several
   floating points. Thinking **6**: 2 for agent position, call it
   **s**; 2 for "attention" position, call it **v**; and 2 more, call
   it **v_bar**. The coordinates sent to S1 are `(v + v_bar) / 2`.
   That way S2 will learn to either focus on what it is discussing, or
   quickly switch to moving, and moving in the direction that it was
   previously focusing on. Unlike s and v, v_bar coordinates are
   allowed to be outside the square sometimes (that has to happen for
   the math to work).

4. Train these coordinates probably also using SFT. Training correct
   **s** emission will be easiest, but there are good ideas about how
   to train **v** and **v_bar** too, later. This happens a little
   later.

5. The S2 system will **remain an LLM**. It must be able to talk, and
   to reason through decisions. On every token generation, it also
   emits 6 float values for the 3 coordinates. This will be trained to
   be "automatic" or "unconscious", although it will later also be
   trained to respond in particular ways to particular questions
   (e.g. move to gold vs don't move vs "look but don't move to the
   upper right-hand corner" — right now it keeps calling "upper"
   corners "bottom" corners).

The v / v_bar system is a little unusual, but it is biologically
inspired and not much harder to install than s and v alone. Do not
argue that part.

## Questions to answer later (not in this pass)

1. What designs for S1? Good target size if tokens should be ~10×
   faster than Gemma 4 on the hardware used here? What is typical for
   VLA networks in this role? Can the Gemma 4 vision tower be reused,
   or is that tower too cumbersome / slow?

2. Use the currently pre-trained checkpoints (`aug27` from the prior
   discussion, or earlier), or a fresh Gemma 4? How bad is "only
   answer in the pattern from the prompts" at this stage (e.g. always
   using `OBS` / emitting `[ANALYS]` before analyst replies)? Look at
   the logs produced by that and other checkpoints.

3. Best way to emit s / v / v_bar? Are there logits currently never
   used in the output (reserved special characters)? Can 6 parts of
   the logit vector be masked before the last softmax (so they are not
   passed to the softmax)? Or add a second layer with 6 outputs, called
   on the same input vector as what had been passed to the layer before
   the logits? Or something else?

4. Anything wrong with the current Gemma 4 architecture that would make
   it a bad S2 choice? Any other networks to recommend over it? The old
   concerns about LLM quality and reasoning quality still apply.

5. Any other suggestions, comments, concerns?
