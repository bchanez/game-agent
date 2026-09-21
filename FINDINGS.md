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

---

# Findings — learned latent representation (Phase 8.5, Stage A)

Does a **compact learned latent** beat raw pixels as the agent's input? Stage A
tests the simplest version: learn the representation **offline** with an
autoencoder, **freeze** it, then run PPO on the latent.

## Setup

- **Game**: Breakout (fast, and we have a pixel-PPO baseline to compare against).
- **Encoder**: conv autoencoder, single 84×84 frame → **64-dim** latent
  (`autoencoder.py`). The 4 stacked frames are each encoded and concatenated, so
  the policy input is **256 numbers** (vs 84×84×4 = 28k pixels) and motion is
  preserved in latent space.
- **Training**: 100k frames from a **random agent** (`collect_frames.py`),
  autoencoder trained offline for 15 epochs (`train_encoder.py`), then **frozen**.
  PPO uses an `MlpPolicy` on the latent (`latent_env.py`, `train_ppo_latent.py`).
- All game-agnostic and behind the same `GameEnv` boundary — no training-code or
  adapter changes.

## The small-moving-object problem (and the fix)

A plain MSE reconstruction loss **drops the ball**: it is ~2 pixels out of 7056,
so ignoring it barely moves the average error while the static background is
rendered perfectly. Low MSE, useless latent.

Fix — a **motion-weighted loss**: weight each pixel by how much it changed since
the previous frame, `w = 1 + α·|frameₜ − frameₜ₋₁|`, so moving objects become
expensive to ignore. This is a **generic** signal ("what moved"), not game
knowledge ("where the ball is") — the *no human indication* principle holds.

| α | ball in reconstruction? |
|---|---|
| 0 (plain MSE) | **gone** in every frame |
| 40 | **retained** (faint but at the right position) |

## Results (Breakout, 500k steps, 8 envs, CPU)

| | Pixel PPO (baseline) | **Latent PPO (Stage A)** |
|---|--:|--:|
| `ep_rew_mean` @500k | ~6.7 (max 7.32) | **~3.0** |
| Wall-clock | ~23 min (~360 fps) | **~7.3 min (~1150 fps)** |
| Policy input | 28k pixels | **256 numbers** |

Latent curve: 1.9 → 3.0, still gently rising at 500k.

## Key takeaways

1. **A compact latent is enough to learn** — 256 numbers drive real learning
   (1.9 → 3.0). Compression works in principle.
2. **~3× faster wall-clock** — the `MlpPolicy` is far cheaper than a CNN, more
   than paying back the encoder forward pass. A real win on a CPU budget.
3. **But it underperforms raw pixels** (3.0 vs 6.7). The latent is lossy exactly
   where it hurts: the ball is faint, so its position/velocity are imprecise.
4. **The frozen encoder is the core limitation** — it was trained on
   **random-agent** frames, so as PPO improves it visits states the encoder never
   saw; the latent degrades precisely as the agent gets good, and a frozen
   encoder cannot adapt (distribution shift).
5. **Motion weighting is essential** for games with small, critical moving
   objects — plain reconstruction is actively harmful there.

## Ideas to push further

- **Stage B (the real fix)**: a representation that is **not frozen** and is
  **dynamics-aware** — reconstruction-free (predict the next latent from the
  action), so it never wastes capacity redrawing pixels and keeps up with the
  states the policy actually visits.
- Cheaper Stage-A tweaks: bigger latent (128), collect encoder frames from a
  trained/curious agent for better state coverage, or simply train PPO longer
  (it had not plateaued).

---

# Findings — self-predictive representations (Phase 8.5, Stage B)

Stage A learned the representation **offline then froze it**, which capped
performance. Stage B keeps it **learning jointly with the policy**, and drops
pixel reconstruction for a **self-predictive** objective (SPR): predict the next
latent from the current latent and action. See `spr.py` for the mechanism.

## Setup

- **Game**: Breakout (same as Stage A, for a direct comparison).
- **Method**: standard PPO on pixels (`CnnPolicy`) **plus** an auxiliary loss on
  the policy's own CNN features — a BYOL-style next-latent prediction with an
  EMA target encoder + stop-gradient to prevent collapse (`train_ppo_spr.py`,
  `PPOSPR`). `--spr-coef 0` recovers plain PPO as a control.
- No reconstruction, no freezing, no game-specific knowledge — the *no human
  indication* principle holds.

## Results (Breakout, 500k steps, 8 envs, CPU)

| | Pixel PPO | Latent (Stage A) | **SPR (Stage B)** |
|---|--:|--:|--:|
| Final `ep_rew_mean` @500k | ~6.7 | ~3.0 | **~8.6** |
| Steps to reach ~6.7 | 500k | never | **~176k** |
| Wall-clock to reach ~6.7 | ~23 min | — | **~9 min** |
| Total wall-clock (500k) | ~23 min | ~7 min | ~25 min |

SPR curve: 0.7 → 6.5 by 176k → 8.6 at 500k.

## Key takeaways

1. **SPR is the clear winner** — ~3× more sample-efficient (reaches the pixel
   baseline's *final* score in ~176k steps) and a higher ceiling (8.6 > 7.3).
2. **Faster to a given quality in wall-clock too** — ~9 min vs ~23 min to hit
   6.7, despite slightly lower throughput; fewer steps more than pay for the SPR
   compute.
3. **The two Stage-A flaws are fixed at once** — no reconstruction (nothing to
   blur away) and no freezing (no distribution shift).
4. **The cost vs Stage A**: no per-step speedup (still a CNN on pixels). Stage A
   bought speed, Stage B bought efficiency — they are complementary.

## Generality — a second game (Mario, 300k, matched control)

Re-run on Super Mario Bros (a different engine, dense reward) with zero code
changes — SPR vs plain PPO (`--spr-coef 0`):

| Mario @300k | plain PPO | **SPR** |
|---|--:|--:|
| Final `ep_rew_mean` | ~2020 (plateaued ~50k) | **~2500 (still rising)** |

The edge is smaller than on Breakout — Mario's dense reward means plain PPO
already does well — but the pattern holds: SPR keeps improving where the control
flattens. The effect is not a Breakout artefact.

## Caveats

- **Single seed.** Strong signal, not proof. RL is high-variance; 3 seeds would
  harden the claim.

## Next

- **Combine A + B**: an SPR-trained latent that is *not* frozen, with the policy
  running on the compact latent (`MlpPolicy`) — aiming for Stage A's ~3× speed
  *and* Stage B's sample-efficiency at once.

---

# Findings — Stage B2: offline SPR latent (a dead end)

Attempt: train the compact encoder with the SPR objective **offline** (on
random-agent transitions), freeze it, run the fast MlpPolicy on the latent —
hoping for Stage A's ~3× speed *and* a good representation (`train_encoder_spr.py`).

**Result: it collapses.** Two runs:
- Plain offline SPR → *constant collapse*: the encoder maps everything to a
  near-constant, the SPR loss trivially hits 0, latent std ~0.01.
- + a VICReg variance term → the constant-collapse is gone but the scale
  explodes (std ~40) and the loss is still ~0: an *informational* collapse (a
  trivially-predictable, useless latent).

**Why:** SPR asks the latent to be *predictable*, not *useful*. In Stage B it
worked because PPO's RL loss anchored "useful"; offline there is no such anchor,
so SPR degenerates. Even stabilized, a frozen encoder trained on random-agent
states would keep Stage A's distribution-shift problem.

**Takeaway:** the frozen-offline SPR latent is a dead end. The real B2 is
**online** — the encoder trained by SPR *jointly* with PPO, the RL loss as the
anchor. Kept as a documented negative result.

---

# Findings — SPR + curiosity (sparse / no reward)

Does SPR also help when the agent has *no game reward* and explores by curiosity
alone? SPR composes with RND (`train_ppo_spr.py --intrinsic-coef`); we compare
against a curiosity-only control (`--spr-coef 0`).

## Montezuma's Revenge (500k, CPU) — inconclusive

Both SPR+curiosity and curiosity-only stayed at **0 extrinsic score throughout**;
episode length wandered ~300–570 for both, no differentiator. Montezuma needs
~100M+ frames (the RND papers); 500k on CPU is ~0.5% of that — far too little to
reach the first key. **Not that SPR fails here — the budget can't test it.**

## Mario pure-curiosity (500k, CPU) — SPR helps, modestly

`--extrinsic-coef 0`: no game reward at all, only novelty. Native Mario reward (a
progress proxy the agent is *not* optimizing) is compared:

| Mario, no game reward | curiosity only | **SPR + curiosity** |
|---|--:|--:|
| max native reward | 2040 | **2210** |
| final native reward | ~1920 (drifts down) | **~2100 (climbs)** |

SPR explores ~8–10% further and keeps climbing where the control stalls — the
same "goes further, doesn't stagnate" pattern seen with rewards, now across four
settings (Breakout, Mario-reward, Mario-no-reward). This supports carrying SPR
into Phase 8 (unmodified games, no reward).

## Caveats (read before trusting the numbers)

- **Training metric, not eval.** The Mario numbers are *training* `ep_rew_mean`
  (shaped reward), **not** eval `x_pos`. They compare SPR vs control against each
  other only — they are **not** comparable to the "x~1834" eval figure of the
  first study.
- **VecMonitor sits under the RND wrapper**, so `ep_rew_mean` always logs the
  *native* game reward (pre-curiosity), not what PPO optimizes. This is why
  Montezuma reads 0, and why Mario-no-reward is measurable at all.
- **Intrinsic reward was not logged** for these runs — we only know extrinsic=0
  on Montezuma, not what curiosity actually did. Log `intrinsic_mean` next time.
- **Single seed; timings indicative** (shared machine). Horizons differ across
  experiments (Mario-with-reward was 300k, the others 500k).
- **SPR checkpoints are ~68M** (vs 20M pixel / 3.1M latent): `PPOSPR` serializes
  the SPR head + target encoder + optimizer into the save.

---

# Findings — multi-game transfer (Level 2, first probe)

The goal of the whole boundary: **one policy that plays many games**. First test
of the payoff — does an SPR policy pretrained on several games learn a *new,
held-out* game faster than from scratch?

## Setup

- **Unified action space** (prerequisite): a fixed canonical controller
  (`CANONICAL_ACTIONS`, 14 = d-pad + A/B), so every game exposes the same
  `Discrete(14)` head and can share one policy. Per-adapter map (identity for
  NES, one shared canonical→ALE map for Atari). This is what lets a single
  `SubprocVecEnv` mix games (`make_multi_venv`).
- **Reward normalization per env** (`NormalizeReward`) in the mix, so Breakout
  (~units) and Mario (~thousands) reach comparable scale — a single value head
  needs this. Generic, no per-game constants.
- **Protocol**: pretrain SPR on `{mario, montezuma}` (300k), then learn the
  held-out **Breakout** (150k) two ways — `--init-from` the pretrained policy vs
  from scratch. Held-out = Breakout for the clearest curve per compute minute;
  Montezuma contributes to the encoder via SPR (reward-free) even at ~0 score.

## Results (Breakout `ep_rew_mean`, raw reward, matched steps)

| step (k) | 4 | 8 | 12 | 28 | 65 | 110 | 151 |
|---|--:|--:|--:|--:|--:|--:|--:|
| **transfer** | 2.38 | 2.41 | 2.60 | 2.72 | 2.38 | 2.53 | 2.54 |
| from-scratch | 1.50 | 2.03 | 2.06 | 2.23 | 2.44 | 2.43 | 2.66 |

Steps to reach `ep_rew_mean ≥ 2.5`: **transfer ~12k vs scratch ~127k (~10×)**.

## Key takeaways

1. **Positive transfer on sample-efficiency** — the pretrained encoder gives a
   warm start (2.38 vs 1.50 at 4k) and reaches a given quality ~10× sooner. The
   textbook "faster warm-up" transfer shape.
2. **Same ceiling, not a higher one** — both converge to ~2.5–2.7 by 150k
   (scratch a touch higher at the end). Transfer accelerates learning, doesn't
   raise the plateau.
3. **It works across dissimilar games** — side-scroller + sparse platformer →
   paddle. Encouraging: the shared representation carries even when dynamics
   differ. Montezuma helped only via reward-free SPR.
4. **Validates the Level-2 direction** — one SPR policy behind `GameEnv` *does*
   transfer. The boundary earns its keep.

## Caveats

- **n=1, single seed.** Strong signal, not proof. The transfer curve is oddly
  flat (starts near its own plateau) — consolidate with 2–3 seeds.
- **Short horizon** (150k), low absolute reward (early CnnPolicy Breakout). The
  transfer/scratch *ratio* is the result, not the absolute values.
- One held-out game only. Next: pretrain `{mario, breakout}` → transfer to
  Montezuma **with curiosity**, to test transfer into a sparse-reward setting.
