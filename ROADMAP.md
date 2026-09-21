# Roadmap — game-agent

## The vision: one generic game-playing agent

The long-term goal is **one agent that can play many games**, ideally through a
*normal* interface (look at the screen, press keys) rather than a modified
emulator with a clean `env.step()`.

Two axes, often confused — we keep them separate:

1. **Multiple games** — breadth, with a single code path.
2. **Normal interface** — driving an *unmodified* game (screen capture + input
   injection), not a research emulator.

And two meanings of "generic":

| | Generic *framework* | Generic *brain* |
|---|---|---|
| What's shared | the code / pipeline | the trained model itself |
| Reality | retrain **per game** | one model plays games it never saw |
| Difficulty | easy — we're nearly there | research frontier |

**Architecture that unifies it all** — a clean `GameEnv` boundary (pixels in,
button out) with swappable pieces:

```
        ┌─────────────────────────────┐
        │   AGENT (the brain)          │   PPO+RND now → reasoning/VLM later
        └─────────────────────────────┘
                     ↕  obs (pixels) / action (button)
        ┌─────────────────────────────┐
        │   ADAPTER (the interface)    │
        │  NES emu · ALE/retro · grab  │   swappable per game
        └─────────────────────────────┘
```

So this is **not several projects** — it's *one* architecture, several game
adapters, and (eventually) two kinds of brain. **ARC-style abstract reasoning**
is not a detour but the bridge to the "generic brain" (a reasoning agent that
looks at any screen and acts).

**Guiding principle — no human indication.** The agent learns from **raw pixels
only**. No hand-crafted perception (template matching, sprite detection,
scripted state), no reward engineering tailored to a specific game. I hand it
the game; it figures the rest out. (This is why we deleted the OpenCV
`detection.ipynb` branch — it did the perception *for* the agent.)

---

## Status (done — details in `FINDINGS.md` and git history)

- **First study — curiosity-driven Mario.** PPO + RND (curiosity as intrinsic
  reward), reproducing Pathak et al. 2017. Results and analysis: `FINDINGS.md`.
- **Phase 6 — the `GameEnv` interface.** Agent (brain) split from game (adapter):
  `src/game_env.py` (interface + shared obs pipeline), `src/games/*` (one adapter
  per game + registry). Adding a game is a new file, not a rewrite.
- **Phase 7 — Atari (ALE).** Montezuma's Revenge + Breakout plug in with zero
  training-code changes. Confirms the sparse-reward story: PPO alone gets no
  signal on Montezuma, curiosity does. Real scores still pending compute
  (Montezuma needs millions of steps / a GPU).
- **Phase 7.5 — Breakout validation run.** A proper PPO run proved the framework
  produces a *learning* agent on a game that isn't Mario, with zero training-code
  changes — the `GameEnv` boundary holds. 500k steps / 8 envs / ~23 min CPU:
  `ep_rew_mean` 1.06 → 6.7 (max 7.32), `explained_variance` 0.85.
- **Level 2 — multi-game transfer (first probe) ✅ DONE.** Unified the action
  space to a fixed canonical controller (`CANONICAL_ACTIONS`, `Discrete(14)`, one
  map per adapter) so a single SPR policy can train on a mix (`make_multi_venv`,
  per-env reward normalization). An SPR policy pretrained on `{mario, montezuma}`
  learns held-out **Breakout ~15–25× faster** than from scratch (reaches
  `ep_rew_mean ≥ 2.5` in ~4–8k vs ~82–139k steps), same ceiling — positive
  transfer across dissimilar games, **confirmed across 3 seeds**. Details:
  `FINDINGS.md`.

---

## What's next

### Phase 8 — Adapter #3: a *normal* interface (no modified emulator)
- Generic **screen-capture + key-injection** env, pointed at the original
  `smbgames.be` web Mario (full circle!) or any browser game.
- New challenge: no clean reward signal from the game. Per the *no human
  indication* principle, the answer is **pure curiosity** (`--extrinsic-coef 0`)
  — the agent needs *no* reward at all, just novelty. We already proved this
  works on Mario (x~1834 with zero game reward). This is the purest form of the
  goal: hand it an unmodified game, it plays, no hints.
- Remaining challenges are honest engineering: capture latency and reliable
  input injection.

### Phase 8.5 — Learned world model (the missing link)
Instead of PPO's CNN re-learning to *see* from scratch on every game, learn a
compact **latent state of the world** behind the same `GameEnv`. The agent learns
the representation itself — no hand-coded object detection, so the *no human
indication* principle holds. It serves **both** roadmap axes: more efficient
(a few hundred latent numbers vs ~50k pixels) and a step toward the generic brain.

- **Stage A — offline autoencoder + frozen encoder ✅ DONE.** Learn a 64-dim
  latent per frame offline, freeze it, run PPO on the latent (`autoencoder.py`,
  `collect_frames.py`, `train_encoder.py`, `latent_env.py`,
  `train_ppo_latent.py`). Result on Breakout (details in `FINDINGS.md`): the
  compact latent *does* learn and is **~3× faster** wall-clock, but underperforms
  raw pixels (~3.0 vs ~6.7 @500k). A plain MSE loss drops the ball; a generic
  **motion-weighted loss** recovers it. The frozen encoder (trained on
  random-agent frames) is the ceiling — distribution shift as the policy improves.
- **Stage B — self-predictive representation (SPR) ✅ DONE.** Fix Stage A's two
  limits at once: learn the latent by **predicting the next latent from the
  action** (no reconstruction, so nothing to blur away), **jointly** with the
  policy (no freezing, no distribution shift), with an EMA target to prevent
  collapse (`spr.py`, `train_ppo_spr.py`). Result on Breakout (details in
  `FINDINGS.md`): **~3× more sample-efficient** than pixel PPO (reaches the
  baseline's final score in ~176k steps vs 500k) with a **higher ceiling**
  (~8.6 vs ~6.7). Cost: no per-step speedup (still a CNN on pixels).
- **Stage B2 — combine A + B.** Goal: an SPR-trained latent driving the fast
  MlpPolicy — Stage A's ~3× speed *and* Stage B's sample-efficiency together.
  - *Offline probe ✅ tried, dead end* (details in `FINDINGS.md`): training the
    encoder with SPR offline then freezing it **collapses** (no RL loss to anchor
    "useful", only "predictable"). A VICReg variance term stops the constant
    collapse but the frozen latent is still degenerate.
  - *Real path 💡 NEXT — online*: train the encoder by SPR **jointly** with PPO
    (RL loss as the anchor), policy on the compact latent, encoder updated slowly
    to keep the observation roughly stationary. **DreamerV3** remains the
    full-scale reference (one hyperparameter set across 150+ tasks) if/when a GPU
    is available.
- **SPR + curiosity ✅ validated (sparse / no-reward).** SPR composes with RND
  (`--intrinsic-coef`). On Mario *pure curiosity* (no game reward) SPR explores
  ~10% further than curiosity alone; Montezuma at 500k/CPU was inconclusive (too
  little compute). Supports carrying SPR into Phase 8, where games give no reward.

### Phase 9 — The "generic brain": a reasoning / VLM agent 🌫️ FRONTIER
- On the same `GameEnv`, swap PPO for a **reasoning agent** (look at the screen,
  think, press a key) aiming to play games it *never trained on*. This is where
  **ARC-style abstract reasoning** connects — the reasoning benchmark for a
  generic agent, not a separate project.

---

## Infra note — the GPU unlock

Several items above wait on the same thing: **a GPU learner**. Docker-on-Mac
can't provide one (no Metal passthrough — see `CLAUDE.md`), so everything today
is CPU-only and already tuned about as far as it goes (perf details in
`CLAUDE.md`). Moving to a GPU — native MPS on this Mac, or a cloud CUDA box —
unlocks *as a group*:
- **Montezuma-scale compute** (millions of steps, currently the blocker on the
  Atari sparse-reward results).
- **The online DreamerV3-style world model** (Phase 8.5, Stage B2).
- **MPS/CUDA training acceleration** for every existing script.
- **Async acting/learning** (Sample Factory / IMPALA): only worth it *with* a GPU
  learner — on CPU both phases contend for the same cores (~1.76× ceiling
  collapses to near-nothing). It's a corollary of the GPU move, not a CPU win.

So treat "get on a GPU" as one decision that opens all four, not four separate
efforts.
