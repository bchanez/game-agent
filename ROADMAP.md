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

## First study: curiosity-driven Mario ✅ DONE

An RL agent that plays Super Mario **by curiosity** — exploring because new
situations are *interesting*, not only because we hand it a reward. Reference:
*Curiosity-driven Exploration by Self-supervised Prediction*, Pathak et al. 2017.

- **Env**: `gym-super-mario-bros` (NES emulator via `nes-py`), bridged to
  Gymnasium with **shimmy**. Since Phase 6 it's the first adapter behind the
  generic `GameEnv` interface (`src/games/mario.py`); the shared obs pipeline
  (`src/game_env.py`) is frame-skip 4 → resize 84×84 → grayscale → stack 4
  (final `84×84×4`). Runs CPU-only in Docker.
- **Algorithm**: PPO (`CnnPolicy`, Stable-Baselines3), 8 parallel envs.
- **Curiosity**: `src/rnd.py` — RND (Burda 2018) as a `VecEnvWrapper`; prediction
  error = intrinsic reward. `src/train_curiosity.py` trains on
  `extrinsic_coef*game_reward + intrinsic_coef*curiosity` — `--extrinsic-coef 0`
  gives **pure curiosity** (no game reward at all).
- **Result** (1M steps each, `src/eval_models.py`, x~3200 = end of 1-1). Full
  analysis in `FINDINGS.md`:

  | model | mean x | max x | flags |
  |---|--:|--:|--:|
  | PPO baseline | 2543 | 3161 | 1/10 |
  | PPO + curiosity | **2798** | 3161 | 1/10 |
  | pure curiosity | 1834 | 2814 | 0/10 |

Open follow-ups: longer runs (5–10M steps), implement **ICM** (the exact
Mario-paper method), and hit the **"noisy TV"** failure mode.

---

## Going generic (the multi-game direction)

### Phase 6 — Extract the `GameEnv` interface ✅ DONE
- Split the old `mario_env.py` into a game-agnostic boundary — **pixels in,
  discrete button out**:
  - `src/game_env.py` — the interface: a `GameSpec` dataclass + the shared obs
    pipeline + `make_venv(spec, n_envs)`. Never mentions Mario.
  - `src/games/mario.py` — adapter #1 (~20 lines): how to build the raw Mario
    env + its `progress_key`/`success_key`.
  - `src/games/__init__.py` — a registry; every script takes `--game mario`.
- `eval_models.py` is now generic too (uses the spec's progress/success keys).
  Model names unchanged (`mario_ppo_final`…) so existing models still load.
- **Adding a game is now a new file in `src/games/`, not a rewrite.**

### Phase 7 — Adapter #2: Atari suite (ALE), incl. Montezuma's Revenge ⏭️ NEXT
- ALE gives ~60 games behind one Gym interface → the **"multiple games"** goal
  almost for free, same PPO+RND code.
- **Montezuma's Revenge** is *the* canonical sparse-reward game curiosity was
  built for. It tests our own `FINDINGS.md` takeaway #4: on Mario (dense reward)
  curiosity only *helps*; on Montezuma it should be the difference between **0
  and real progress**.

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

**Phase 6**: extract the `GameEnv` interface and re-express Mario as the first
adapter behind it. Small, self-contained, and the unlock for everything after —
Atari (Phase 7) and a screen-capture env (Phase 8) then drop in as new adapters
instead of rewrites.
