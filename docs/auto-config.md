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
     curiosity + self-imitation on when sparse. Verified: Mario reads 0.72 (dense →
     both OFF), every ARC game reads 0.00 (sparse → both ON), Breakout 0.007 (ON).
   - *non-Markov detector* ✅ (replaces the broken "fraction of cells changed"
     motion heuristic): if the SAME (obs, action) ever leads to a DIFFERENT next
     obs, a single frame hides state. Routed by obs type — **pixels** (hidden
     *velocity*) → frame-stack; **grids** (hidden *counter/rule*) → memory/LSTM,
     not frame-stack. Verified: Mario/Breakout → frame-stack; ls20/g50t flagged
     non-Markov (ls20's depleting "life") → `recurrent=True`, which `--recurrent`
     now actions (an LSTM policy / RecurrentPPO, auto-selected). Caveat: SPR and
     self-imitation aren't wired into the recurrent buffer yet (off on that path);
     and the held-last-grid on empty transition frames can inflate the grid signal.
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
2. **Fast configurator** ✅ — the default: probe → each tool's rule → config in
   seconds (`auto_config.py`, `--auto`). Detectors: sparsity → curiosity/SIL;
   non-Markov → frame-stack (pixels) or memory (grids). Verified per-game.
3. **Slow ablation loop** — built (`auto_loop.py`) but reserved for genuinely-new
   tools, *not* the default (a cheap detector answers "activate?" far faster).

## Two separate decisions (don't conflate them)

A tool's lifecycle has two decisions, and only one is expensive:

1. **Keep-in-library** — *additive, once*: a tool enters the library the moment it
   helps on **≥1 game**, and stays forever. No re-testing. (SIL helped ls20 → kept;
   SPR helped Mario → kept; curiosity helps sparse games → kept.) The library only
   grows.
2. **Activate-per-game** — *fast, every game*: given a new game, which library tools
   to switch on and how to set their params. This is the **fast configurator**
   (`auto_config.py`, `--auto`): one short probe → each tool's rule fires → a full
   config in **seconds**.

The mistake to avoid: using a slow ablation (a full training run per tool, ~30 min)
to answer decision 2. That's what a cheap **detector** is for. Re-running "does
removing SIL hurt on ls20?" is redundant — SIL is already kept, and the sparsity
detector already decides to activate it in 2 seconds.

## The fast configurator (`auto_config.py`) — the default path

`TOOLS` is the library: each entry is `(name, rule)` where `rule(signals)` returns
the config overrides that tool wants (on/off and/or params). `configure` applies
every rule to one probe. Growing the box = appending a tool. Some tools are on/off
(curiosity, self-imitation — gated on reward sparsity); some *tune params* (auto-gamma
sets γ from the observed horizon). This runs in seconds and needs no training.

## When the slow loop *is* justified (rare)

The automated ablation loop (`auto_loop.py`, "Level A") is **not** the default — it's
for the cases a cheap detector can't cover: validating a **genuinely new** tool whose
usefulness is unknown, or tuning a hyperparameter with no cheap signal. It hill-climbs
configs on a short budget. Caveat learned the hard way: single-eval scores are noisy
(the same config scored 0.35 then 0.10 across identical runs), so only large gaps are
trustworthy without multi-seed — which multiplies compute (an argument for a GPU).

## Level B — generated tools (later, dev-time LLM)

The end of the north star: the LLM proposes *new* tool code inside the dev loop, tests
it, keeps the winners (à la Voyager / ADAS). Self-improvement happens at **dev time**
(the Kaggle eval forbids external LLMs); the shipped agent is fixed and LLM-free.

## Measurement discipline

For every tool, log enough to *compare and analyze*: does it change the training
signal (e.g. completion frequency on a sparse game), the eval score, and the
throughput cost? A tool that helps but is expensive, or is free but useless, both
matter for the loop's selection. Findings go in `FINDINGS.md`; each tool is one
clean, single-variable experiment against the baseline.
