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

### Phase 9 — The "generic brain": a reasoning / VLM agent 🌫️ FRONTIER
- On the same `GameEnv`, swap PPO for a **reasoning agent** (look at the screen,
  think, press a key) aiming to play games it *never trained on*. This is where
  **ARC-style abstract reasoning** connects — the reasoning benchmark for a
  generic agent, not a separate project.

---

## Suggested next step

A proper **Breakout** training run: dense reward, quick on CPU, and it proves the
framework produces a *learning* agent on a game that isn't Mario. Then decide
between pushing on the **generic brain** (Phase 9) and the **capture env**
(Phase 8).
