"""Phase 3: train PPO with curiosity (RND intrinsic reward).

    # extrinsic game reward + curiosity (the usual, strongest setup)
    python src/train_curiosity.py --timesteps 100000

    # PURE curiosity, no game reward at all — how far does Mario get driven
    # only by the desire to see new things? (the headline experiment)
    python src/train_curiosity.py --timesteps 100000 --extrinsic-coef 0

`ep_rew_mean` in the logs is the reward PPO optimizes (extrinsic+intrinsic).
The `curiosity/*` lines report the two components separately, and the true
game score (extrinsic only) so you can tell real progress from novelty-seeking.
"""
import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from mario_env import make_venv
from rnd import RNDReward

MODELS_DIR = "/app/data/models"
TB_DIR = "/app/data/tb"


class CuriosityStats(BaseCallback):
    """Logs the intrinsic vs extrinsic reward components each rollout."""

    def _on_step(self):
        for info in self.locals["infos"]:
            if "intrinsic" in info:
                self._intr.append(info["intrinsic"])
                self._extr.append(info["extrinsic"])
        return True

    def _on_rollout_start(self):
        self._intr, self._extr = [], []

    def _on_rollout_end(self):
        if self._intr:
            self.logger.record("curiosity/intrinsic_mean", float(np.mean(self._intr)))
            self.logger.record("curiosity/extrinsic_mean", float(np.mean(self._extr)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument("--intrinsic-coef", type=float, default=1.0)
    ap.add_argument("--extrinsic-coef", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    tag = "pure_curiosity" if args.extrinsic_coef == 0 else "curiosity"

    venv = RNDReward(
        make_venv(args.n_envs),
        intrinsic_coef=args.intrinsic_coef,
        extrinsic_coef=args.extrinsic_coef,
        device="cpu",
    )
    model = PPO(
        "CnnPolicy", venv, verbose=1,
        n_steps=512, batch_size=64, n_epochs=10,
        learning_rate=1e-4, gamma=0.9, ent_coef=0.01,
        tensorboard_log=TB_DIR, device="cpu",
    )
    ckpt = CheckpointCallback(
        save_freq=max(20_000 // args.n_envs, 1),
        save_path=MODELS_DIR, name_prefix=f"mario_{tag}",
    )

    print(f"Training PPO+RND ({tag}) for {args.timesteps} steps "
          f"[intrinsic={args.intrinsic_coef}, extrinsic={args.extrinsic_coef}]", flush=True)
    model.learn(total_timesteps=args.timesteps, callback=[ckpt, CuriosityStats()])

    final = os.path.join(MODELS_DIR, f"mario_{tag}_final")
    model.save(final)
    print(f"Saved {final}.zip", flush=True)


if __name__ == "__main__":
    main()
