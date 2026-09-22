# The self-configuring agent — a toolbox that turns itself on and off

The north-star architecture for the generic agent (see `ROADMAP.md`, and the
ARC-AGI-3 findings in `FINDINGS.md`). The competition eval runs on **games we
will never see**, so nothing can be hand-tuned per game. The consequence is a
design principle:

> **Every capability is either a universal default or self-configuring — never
> a knob a human sets per game.**

## Why

On Kaggle we can't look at the game and pick `gamma`, decide to enable
frame-stacking, or turn on memory. The agent must **measure the game and choose
its own inductive biases**. A capability that only helps when a human flips it on
for the right game is disqualified. So the agent is not one fixed network — it's a
**library of tools**, each of which the agent switches **on when useful and off
when useless** (off = saved compute, less noise).

## The three layers

1. **Tools** — independent, individually-toggleable capabilities. The codebase is
   already tool-shaped (each is a flag/callback):
   | tool | what it buys | when it helps |
   |---|---|---|
   | RND curiosity | exploration signal | sparse/absent reward |
   | auto-gamma | far-sighted credit assignment | long-horizon reward |
   | SPR | dynamics-aware representation | any (sample-efficiency) |
   | frame-stack | perceive motion | things move between frames |
   | recurrence (LSTM) | remember task state | non-Markovian / must recall |
   | latent encoder | compact obs, faster | high-dim compressible obs |
   | self-imitation | reuse rare successes | sparse wins, on-policy forgets |

2. **Detectors** — cheap probes that read the environment and set each tool's
   on/off (the self-configuration). Examples:
   - *motion detector*: frame-to-frame grid change → frame-stack on/off.
   - *reward-sparsity detector*: reward density over N steps → curiosity/gamma.
   - *non-Markov detector*: does memory improve next-step prediction? → recurrence.
   auto-gamma is already a detector+tool in one (γ = 1 − 1/observed-horizon).

3. **Meta-analysis** — periodically measure which *enabled* tools correlate with
   progress and drop the rest. This is the top layer and comes **last**: you can't
   auto-select among tools that don't each work on their own yet.

## Build order (deliberate)

Tools must each work in isolation **before** the auto-toggle sits on top —
otherwise the meta-controller is selecting among broken parts. So:

1. **Build the tools**, one at a time, each measured against a clean baseline
   (the ARC ls20 ablation is exactly this). *Current front: self-imitation.*
2. **Add detectors** — start with motion → frame-stack (cheap, concrete). This is
   what makes "keep frame-stack but manage it automatically" real.
3. **Meta-analysis** — once ≥3–4 tools are validated, add the layer that turns
   them on/off from measured utility.

## Measurement discipline

For every tool, log enough to *compare and analyze*: does it change the training
signal (e.g. completion frequency on a sparse game), the eval score, and the
throughput cost? A tool that helps but is expensive, or is free but useless, both
matter for the detector logic later. Findings go in `FINDINGS.md`; each tool is
one clean, single-variable experiment against the baseline.
