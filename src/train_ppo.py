"""Train a PPO agent on a game's built-in (extrinsic) reward.

This is the curiosity-free baseline the other setups are measured against.

Run (inside the container):
    python src/train_ppo.py --game mario --timesteps 100000
    python src/train_ppo.py --game breakout --timesteps 25000 --n-envs 1

CPU-only in Docker. Checkpoints land in data/models/, TensorBoard logs in
data/tb/.
"""
import perf  # first: sets BLAS thread limits before torch/numpy import

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from game_env import make_venv
from games import get_game

MODELS_DIR = "/app/data/models"
TB_DIR = "/app/data/tb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="mario", help="which game to train on")
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument("--resume", default=None,
                    help="path to a .zip to continue training from (else fresh)")
    ap.add_argument("--torch-threads", type=int, default=None,
                    help="learner torch threads (default: all cores)")
    args = ap.parse_args()

    perf.setup_cpu_threads(args.torch_threads)
    spec = get_game(args.game)
    os.makedirs(MODELS_DIR, exist_ok=True)
    venv = make_venv(spec, args.n_envs)

    if args.resume:
        prefix = f"{spec.name}_ppo_cont"
        print(f"Resuming from {args.resume}", flush=True)
        model = PPO.load(args.resume, env=venv, device="cpu",
                         tensorboard_log=TB_DIR)
    else:
        prefix = f"{spec.name}_ppo"
        model = PPO(
            "CnnPolicy", venv, verbose=1,
            n_steps=512, batch_size=64, n_epochs=10,
            learning_rate=1e-4, gamma=0.9, gae_lambda=1.0, ent_coef=0.01,
            tensorboard_log=TB_DIR, device="cpu",
        )

    ckpt = CheckpointCallback(
        save_freq=max(20_000 // args.n_envs, 1),
        save_path=MODELS_DIR, name_prefix=prefix,
    )

    print(f"Training PPO for {args.timesteps} steps on {args.n_envs} env(s)...", flush=True)
    model.learn(total_timesteps=args.timesteps, callback=ckpt,
                reset_num_timesteps=args.resume is None)

    final = os.path.join(MODELS_DIR, f"{prefix}_final")
    model.save(final)
    print(f"Saved {final}.zip", flush=True)


if __name__ == "__main__":
    main()
