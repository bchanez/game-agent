# Findings — curiosity-driven Mario

Research notes from the first study. See `ROADMAP.md` for the plan and status.

## Setup

- **Goal**: a Super Mario Bros agent driven by **curiosity** (intrinsic
  motivation), reproducing Pathak et al. 2017.
- **Env**: `gym-super-mario-bros` (NES emulator), obs = frame-skip 4 → resize
  84×84 → grayscale → stack 4 (`84×84×4`). CPU-only in Docker — the bottleneck
  is the emulator, not the net, so CPU is fine.
- **Throughput**: ~140 steps/s (1 env), ~340 steps/s (8 envs, `SubprocVecEnv`).
  1M steps ≈ 70 min on an M4 Max (16 cores).

## One algorithm (PPO), a reward dial

The algorithm is always **PPO**. What we vary is the reward:

    reward = extrinsic_coef * game_reward  +  intrinsic_coef * curiosity

| Setup | extrinsic | intrinsic | meaning |
|---|:-:|:-:|---|
| PPO baseline | 1 | 0 | guided only by the game (reference) |
| PPO + curiosity | 1 | 1 | guided **and** curious |
| pure curiosity | 0 | 1 | no game reward at all — only novelty |

One script (`train_curiosity.py`) with different coefficients, not three
programs. Curiosity is **RND**: a frozen random target net + a predictor net;
prediction error = intrinsic reward (high on novel frames, decaying as they
become familiar).

## Results (1M steps each, 10 eval episodes)

`x_pos ~3200` = end of World 1-1. Random agent reaches only ~400.

| model | mean x | max x | flags (level done) |
|---|--:|--:|:-:|
| PPO baseline | 2543 | 3161 | 1/10 |
| **PPO + curiosity** | **2798** | 3161 | 1/10 |
| pure curiosity | 1834 | 2814 | 0/10 |

Interaction with the world (coins / score, 6 episodes):

| model | coins | score | power-up? |
|---|--:|--:|:-:|
| PPO baseline | 0.3 | 283 | 0/6 |
| PPO + curiosity | 3.8 | 14058 | 0/6 |
| pure curiosity | 3.8 | 4792 | 0/6 |

## Key takeaways

1. **Curiosity helps** — best as a *complement* to a reward, not a replacement
   (2798 > 2543).
2. **Pure curiosity works** — with *zero* game reward, Mario still reaches
   x~1834: moving right = new scenery = novelty. The Pathak headline result,
   reproduced.
3. **Curiosity drives interaction** — the baseline ignores everything (0.3
   coins, score 283); curious agents break blocks / grab coins far more (score
   14058) as a *side effect* of seeking novelty. None learned to grab a power-up.
4. **The right dial depends on the game** — Mario has a **dense, well-aligned**
   reward (right = good), so guidance is strong. In **sparse-reward** games
   (Montezuma's Revenge) plain PPO gets stuck at zero and curiosity becomes
   essential. → the motivation for Phase 7.

## Practical default

**PPO + curiosity** as the go-to. Keep the baseline as a scientific control and
pure curiosity for sparse-reward experiments — all one flag away.

## Ideas to push further

- Longer training (5–10M steps) to raise the flag-completion rate.
- Implement **ICM** (the exact Mario-paper method) and add it to the table.
- Test a **sparse-reward** setting to show curiosity taking the lead.
- Overlay the TensorBoard curves (`data/tb/`) of the three runs.
