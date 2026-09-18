# Roadmap — Curiosity-driven Mario

Goal: an agent that learns to play Super Mario **by curiosity** — it explores
because new situations are *interesting*, not (only) because we hand it a reward.

This is **Reinforcement Learning (RL)**. Reference paper:
*Curiosity-driven Exploration by Self-supervised Prediction*, Pathak et al., 2017
(the ICM module — and their headline demo is literally Mario).

---

## Where we are today

- `src/detection.ipynb` = OpenCV **template matching**. This is *perception*
  (where things are on screen), **not** machine learning — there is no learning.
- It stays useful as a "perception playground", but the RL agent below will not
  need it at first: modern RL usually eats **raw pixels**.

---

## Important decision up front: which Mario?

The README targets the **web** game (smbgames.be, Flash/JS). For RL that's a
dead-end: no clean `reset()/step()/reward`, you'd need browser automation +
screen capture + key injection, and it's slow and fragile.

➡️ **Switch to `gym-super-mario-bros`** (a real NES emulator via `nes-py`). It's
*the* standard research environment and gives you out of the box:
`observation, reward, done, info = env.step(action)`, plus frames and controls.
Everything still runs in Docker (Python 3.9 is compatible).

---

## The big constraint: Docker on Mac = CPU only

RL training is compute-heavy. On your Mac, a Linux Docker container **cannot use
the GPU** (no CUDA, no Apple MPS inside the container). So:

- Training in Docker will be **CPU-only and slow** (fine for learning + small
  experiments, painful for "beat the level").
- Options when it gets too slow: (a) accept it for learning, use small nets +
  frame-skip; (b) run PyTorch natively on Mac with MPS (breaks "nothing local");
  (c) rent a cloud GPU later. We'll start with (a).

---

## Phases

### Phase 1 — A controllable environment ✅ DONE
- ✅ Added `gym-super-mario-bros` + `nes-py` to `docker/requirements.txt`
  (pinned combo: `gym==0.25.2`, `nes-py==8.2.1`, `gym-super-mario-bros==7.4.0`,
  `numpy<2`, `opencv<4.12` — see the comments in requirements.txt for why).
- ✅ `src/random_agent.py`: random agent with `SIMPLE_MOVEMENT`, saves a video
  to `data/random_agent.mp4`. Run it via the VSCode task
  **"Mario: Random agent (Phase 1)"** or `python src/random_agent.py`.
- ⏭️ Still TODO here (small): the standard observation wrappers
  (grayscale → resize 84×84 → frame-skip 4 → frame-stack 4). We'll add these
  at the start of Phase 2 since PPO needs them.
- Done: we can step the env and watch Mario flail around.

### Phase 2 — Baseline agent (extrinsic reward) ✅ PIPELINE DONE
- ✅ Observation wrappers in `src/mario_env.py`: frame-skip 4 → resize 84×84 →
  grayscale → stack 4  (final obs `84×84×4`). Old-gym is bridged to Gymnasium
  via **shimmy** so **Stable-Baselines3** can consume it.
- ✅ `src/train_ppo.py`: trains **PPO** (`CnnPolicy`) on the game's built-in
  reward. Checkpoints → `data/models/`, TensorBoard logs → `data/tb/`.
  Throughput ≈ 140 agent-steps/s on CPU (≈560 game fps thanks to frame-skip).
- ✅ `src/record_agent.py`: plays one episode with a trained model → mp4.
- ⏭️ TODO (just compute time): run a **long** training (500k–1M steps) to get
  an agent that clearly beats random. On CPU this is hours — candidate for a
  background run, or later a cloud GPU.
- Run via VSCode tasks **"Mario: Train PPO"** / **"Mario: Record trained agent"**.

### Phase 3 — Add curiosity 🎯
- Plug in an **intrinsic reward**: the agent is rewarded for reaching states its
  own model **fails to predict** (= surprising = new).
- Two options:
  - **ICM** (Pathak 2017): forward + inverse models, curiosity = prediction
    error in a *learned* feature space. Matches the paper exactly.
  - **RND** (Random Network Distillation, Burda 2018): simpler & more stable,
    often the better first implementation. Good fallback.
- The headline experiment: **pure curiosity, zero extrinsic reward** — how far
  does Mario get driven *only* by the desire to see new things?

### Phase 4 — Experiments & understanding
- Compare: PPO-only vs PPO+ICM vs pure-curiosity.
- Meet the classic failure: the **"noisy TV" problem** (curiosity gets addicted
  to random noise) — a great lesson in why curiosity is subtle.
- Metrics: distance travelled, % level completed, intrinsic-reward curves.

### Phase 5 — (optional) reconnect your OpenCV work
- Feed a **feature-based** state (from detection) instead of raw pixels, or a
  hybrid. Lets you revisit the notebook with a purpose.

---

## Suggested next step

Phase 1 only: wire `gym-super-mario-bros` into the container and get a random
agent running with a saved video. Small, self-contained, and it tells us fast
whether the NES env behaves well inside Docker before investing in training.
