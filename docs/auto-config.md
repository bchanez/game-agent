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
   on/off (the self-configuration). Built in `auto_config.py` (`probe` +
   `configure`), exposed as `train_ppo_spr.py --auto`:
   - *reward-sparsity detector* ✅ reliable: reward density over a random probe →
     curiosity + self-imitation on when sparse. On every ARC game it reads ~0 and
     correctly enables them; on dense games it disables them.
   - *motion detector* ❌ unreliable: "fraction of cells changed" does NOT separate
     agent-caused change (Markovian) from ambient motion (the non-Markovian case
     frame-stack is for) — ls20 (no motion) measured *higher* than Breakout. So
     frame-stack is defaulted off for grids; a real non-Markovianity detector is a
     layer-2 TODO.
   - auto-gamma is already a detector+tool in one (γ = 1 − 1/observed-horizon).

3. **Meta-analysis / the automated loop** — periodically measure which *enabled*
   tools correlate with progress and keep the winners. This is the top layer and
   comes **last**: you can't auto-select among tools that don't each work on their
   own. Its full form is *automating the propose→test→keep loop we now run by
   hand* — see below.

## Build order (deliberate)

Tools must each work in isolation **before** the auto-toggle sits on top —
otherwise the meta-controller is selecting among broken parts. So:

1. **Build the tools** ✅ (mostly) — curiosity, auto-gamma, SPR, grid embedding,
   self-imitation, frame-stack, each measured against a clean baseline (the ARC
   ls20 ablation). SIL was the first to move ls20 off 0 (→ level 1).
2. **Add detectors** — in progress: sparsity works, motion doesn't (see above).
3. **Automate the loop** ← *next*.

## Self-improvement: automating the loop (the north star)

The end goal isn't a toolbox *we* hand-tune — it's a toolbox the system optimizes
itself. Key legality constraint: the ARC-AGI-3 Kaggle eval forbids external LLMs,
so self-improvement must happen at **dev time** (an offline loop designs/tests the
agent) and the *submitted* agent is fixed and LLM-free.

- **Level A — auto-search over existing tools** (buildable now, no LLM): a harness
  that hill-climbs the tool set — start from the best known config (auto-gamma +
  SIL), toggle/adjust one tool at a time on a short training budget, evaluate, keep
  improvements, log everything. This is exactly the ls20 ablation, run
  automatically. Compute-aware: short per-candidate budgets (triage), full run only
  on the winner; sequential (CPU-bound).
- **Level B — generated tools** (later, dev-time LLM): the LLM proposes *new* tool
  code inside the loop, tests it, keeps the winners (à la Voyager / ADAS). The
  agent is designed by AI but shipped frozen — legal at eval.

## Measurement discipline

For every tool, log enough to *compare and analyze*: does it change the training
signal (e.g. completion frequency on a sparse game), the eval score, and the
throughput cost? A tool that helps but is expensive, or is free but useless, both
matter for the loop's selection. Findings go in `FINDINGS.md`; each tool is one
clean, single-variable experiment against the baseline.
