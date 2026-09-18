"""Phase 2 baseline: train a PPO agent on Super Mario Bros (extrinsic reward).

No curiosity yet — this uses the game's built-in reward (move right, don't die)
to prove the whole learning loop works before we add intrinsic motivation.

Run (inside the container):
    python src/train_ppo.py --timesteps 100000
    python src/train_ppo.py --timesteps 25000 --n-envs 1   # quick proof

Training is CPU-only in Docker (no GPU on Mac), so it's slow. Watch the
`fps` and `ep_rew_mean` columns in the output; checkpoints land in
data/models/ and TensorBoard logs in data/tb/.
"""
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from mario_env import make_venv

MODELS_DIR = "/app/data/models"
TB_DIR = "/app/data/tb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--n-envs", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    venv = make_venv(args.n_envs)
    model = PPO(
        "CnnPolicy",
        venv,
        verbose=1,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        learning_rate=1e-4,
        gamma=0.9,
        gae_lambda=1.0,
        ent_coef=0.01,
        tensorboard_log=TB_DIR,
        device="cpu",
    )

    ckpt = CheckpointCallback(
        save_freq=max(20_000 // args.n_envs, 1),
        save_path=MODELS_DIR,
        name_prefix="mario_ppo",
    )

    print(f"Training PPO for {args.timesteps} steps on {args.n_envs} env(s)...", flush=True)
    model.learn(total_timesteps=args.timesteps, callback=ckpt)

    final = os.path.join(MODELS_DIR, "mario_ppo_final")
    model.save(final)
    print(f"Saved {final}.zip", flush=True)


if __name__ == "__main__":
    main()
