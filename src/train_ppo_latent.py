"""Train PPO on the frozen encoder's latent instead of raw pixels.

Stage A step 3/4 (see ROADMAP.md). Same PPO recipe as train_ppo.py, but the
observation is the compact latent (LatentObs) and the policy is an MlpPolicy
(no CNN — the encoder already did the seeing). Compare its ep_rew_mean against
the raw-pixel baseline to judge whether the learned representation is enough.

Run (inside the container), after training an encoder first:
    python src/train_encoder.py --game breakout --motion-alpha 40
    python src/train_ppo_latent.py --game breakout --timesteps 500000 --n-envs 8

Smoke-test to a scratch path so a real model is never clobbered:
    python src/train_ppo_latent.py --game breakout --timesteps 25000 --out /app/data/models/_smoke
"""
import perf  # first: sets BLAS thread limits before torch/numpy import

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from game_env import make_venv
from games import get_game
from latent_env import LatentObs

MODELS_DIR = "/app/data/models"
TB_DIR = "/app/data/tb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="breakout")
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--encoder", default=None,
                    help="encoder .pt (else data/models/<game>_encoder.pt)")
    ap.add_argument("--out", default=None,
                    help="output prefix (else data/models/<game>_ppo_latent_final); "
                         "use a scratch path for smoke tests")
    ap.add_argument("--torch-threads", type=int, default=None,
                    help="learner torch threads (default: all cores)")
    ap.add_argument("--n-epochs", type=int, default=4,
                    help="PPO passes per rollout (lower = faster updates)")
    ap.add_argument("--net-arch", default="256,256",
                    help="MLP hidden sizes; smaller is faster on compact latents")
    args = ap.parse_args()

    perf.setup_cpu_threads(args.torch_threads)
    spec = get_game(args.game)
    os.makedirs(MODELS_DIR, exist_ok=True)
    encoder = args.encoder or os.path.join(MODELS_DIR, f"{spec.name}_encoder.pt")

    venv = LatentObs(make_venv(spec, args.n_envs), encoder, device="cpu")

    model = PPO(
        "MlpPolicy", venv, verbose=1,
        n_steps=512, batch_size=64, n_epochs=args.n_epochs,
        learning_rate=1e-4, gamma=0.9, gae_lambda=1.0, ent_coef=0.01,
        policy_kwargs=dict(net_arch=[int(x) for x in args.net_arch.split(",")]),
        tensorboard_log=TB_DIR, device="cpu",
    )

    prefix = f"{spec.name}_ppo_latent"
    ckpt = CheckpointCallback(
        save_freq=max(20_000 // args.n_envs, 1),
        save_path=MODELS_DIR, name_prefix=prefix,
    )

    print(f"Training PPO on latents ({encoder}) for {args.timesteps} steps "
          f"on {args.n_envs} env(s)...", flush=True)
    model.learn(total_timesteps=args.timesteps, callback=ckpt)

    final = args.out or os.path.join(MODELS_DIR, f"{prefix}_final")
    model.save(final)
    print(f"Saved {final}.zip", flush=True)


if __name__ == "__main__":
    main()
