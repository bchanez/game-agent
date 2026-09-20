"""Train PPO with curiosity (RND intrinsic reward).

    # game reward + curiosity (the usual, strongest setup)
    python src/train_curiosity.py --game mario --timesteps 100000

    # pure curiosity: no game reward at all, driven only by novelty
    python src/train_curiosity.py --game montezuma --timesteps 100000 --extrinsic-coef 0

`ep_rew_mean` in the logs is the reward PPO optimizes (extrinsic+intrinsic); the
`curiosity/*` lines split the two components so you can tell real progress from
novelty-seeking.
"""
import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from game_env import make_venv
from games import get_game
from rnd import RNDReward, RNDTrainCallback

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
    ap.add_argument("--game", default="mario", help="which game to train on")
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument("--intrinsic-coef", type=float, default=1.0)
    ap.add_argument("--extrinsic-coef", type=float, default=1.0)
    ap.add_argument("--resume", default=None,
                    help="path to a .zip to continue training from (else fresh)")
    args = ap.parse_args()

    spec = get_game(args.game)
    os.makedirs(MODELS_DIR, exist_ok=True)
    base_tag = "pure_curiosity" if args.extrinsic_coef == 0 else "curiosity"
    tag = f"{base_tag}_cont" if args.resume else base_tag

    venv = RNDReward(
        make_venv(spec, args.n_envs),
        intrinsic_coef=args.intrinsic_coef,
        extrinsic_coef=args.extrinsic_coef,
        device="cpu",
    )

    if args.resume:
        print(f"Resuming from {args.resume} (+ RND state)", flush=True)
        rnd_path = (args.resume[:-4] if args.resume.endswith(".zip") else args.resume) + ".rnd"
        if os.path.exists(rnd_path):
            venv.load_rnd(rnd_path)   # restore the "second brain" too
        else:
            print(f"  warning: {rnd_path} not found — RND predictor restarts fresh", flush=True)
        model = PPO.load(args.resume, env=venv, device="cpu", tensorboard_log=TB_DIR)
    else:
        model = PPO(
            "CnnPolicy", venv, verbose=1,
            n_steps=512, batch_size=64, n_epochs=10,
            learning_rate=1e-4, gamma=0.9, ent_coef=0.01,
            tensorboard_log=TB_DIR, device="cpu",
        )
    ckpt = CheckpointCallback(
        save_freq=max(20_000 // args.n_envs, 1),
        save_path=MODELS_DIR, name_prefix=f"{spec.name}_{tag}",
    )

    print(f"Training PPO+RND ({tag}) for {args.timesteps} steps "
          f"[intrinsic={args.intrinsic_coef}, extrinsic={args.extrinsic_coef}]", flush=True)
    model.learn(total_timesteps=args.timesteps,
                callback=[ckpt, CuriosityStats(), RNDTrainCallback(venv)],
                reset_num_timesteps=args.resume is None)

    final = os.path.join(MODELS_DIR, f"{spec.name}_{tag}_final")
    model.save(final)
    venv.save_rnd(final + ".rnd")   # save both brains together
    print(f"Saved {final}.zip (+ .rnd)", flush=True)


if __name__ == "__main__":
    main()
