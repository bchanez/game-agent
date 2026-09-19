# Findings — Curiosity-driven Mario

Research notes from our experiments. See `ROADMAP.md` for the plan and status.

## The setup

- **Goal**: an agent that plays Super Mario Bros, and specifically one driven by
  **curiosity** (intrinsic motivation), reproducing Pathak et al. 2017.
- **Environment**: `gym-super-mario-bros` (NES emulator), obs pipeline =
  frame-skip 4 → resize 84×84 → grayscale → stack 4  (final `84×84×4`).
- **Everything runs in Docker** (nothing installed on the host). Training is
  **CPU-only** (no GPU inside Docker on Mac), but the bottleneck is the NES
  emulator, not the network, so CPU is fine.
- **Throughput**: ~140 steps/s with 1 env, ~340 steps/s with 8 parallel envs
  (`SubprocVecEnv`). 1M steps ≈ ~70 min on this machine (M4 Max, 16 cores).

## PPO vs the reward: one algorithm, a dial

The **algorithm is always PPO** (Proximal Policy Optimization — the RL method
that updates the agent's neural net from experience). What we vary is the
**reward**, via two coefficients:

    reward = extrinsic_coef * game_reward  +  intrinsic_coef * curiosity

| Setup            | extrinsic | intrinsic | meaning                              |
|------------------|:---------:|:---------:|--------------------------------------|
| PPO baseline     |     1     |     0     | guided only by the game (reference)  |
| PPO + curiosity  |     1     |     1     | guided **and** curious               |
| pure curiosity   |     0     |     1     | no game reward at all — only novelty |

It's a **single script** (`train_curiosity.py`) with different coefficients, not
three programs. The curiosity is **RND** (Random Network Distillation): a frozen
random target net + a predictor net; prediction error = intrinsic reward (high
on novel frames, decaying as they become familiar).

## Results (1M steps each, evaluated over 10 episodes)

`x_pos ~3200` = end of World 1-1. Random agent reaches only ~400.

| model            | mean x | max x | flags (level done) |
|------------------|-------:|------:|:------------------:|
| PPO baseline     |   2543 |  3161 |        1/10        |
| **PPO + curiosity** | **2798** | 3161 |     1/10        |
| pure curiosity   |   1834 |  2814 |        0/10        |

### Interaction with the world (coins / score, 6 episodes)

| model            | coins | score  | got a power-up? |
|------------------|------:|-------:|:---------------:|
| PPO baseline     |   0.3 |    283 |       0/6       |
| PPO + curiosity  |   3.8 | 14058  |       0/6       |
| pure curiosity   |   3.8 |  4792  |       0/6       |

## Key takeaways

1. **Curiosity helps.** PPO+curiosity goes furthest on average (2798 > 2543).
   Best as a **complement** to a reward, not a replacement.
2. **Pure curiosity works.** With *zero* game reward, Mario still gets to
   x~1834 — moving right = new scenery = novelty. This is the Pathak headline
   result, reproduced here.
3. **Curiosity drives interaction.** The baseline ignores everything (0.3 coins,
   score 283); the curious agents break blocks / grab coins far more (score
   14058 vs 283) — as a *side effect* of seeking novelty, not because they were
   told to. None of them learned to grab a power-up (mushroom), though.
4. **The right dial depends on the game.** Mario has a **dense, well-aligned**
   reward (right = always good), so guidance is strong here. In **sparse-reward**
   games (e.g. Montezuma's Revenge) plain PPO gets stuck at zero and curiosity
   becomes essential — pure/heavy curiosity would likely win there.

## Practical default

Use **PPO + curiosity** as the go-to. Keep the baseline as a scientific control
(to prove curiosity helped) and pure curiosity for sparse-reward experiments —
they're all one flag away.

## Ideas to push further

- Longer training (5–10M steps) to raise the flag-completion rate.
- Implement **ICM** (the exact Mario-paper method) and add it to the table.
- Test a **sparse-reward** setting to show curiosity taking the lead.
- Overlay the TensorBoard curves (`data/tb/`) of the three runs.
