"""Diagnostic — why does Mario 1-1 plateau at x_pos_mean ~1400?

Not a product script: it deliberately bypasses the shared canonical-14 boundary
(which multi-game transfer needs) to A/B a *single-game* Mario on different action
sets, same PPO recipe as train_ppo_spr.py. Suspect #1 from FINDINGS: the 14-action
canonical space dilutes exploration of the one action that clears the pit after the
pipes — a running jump (right+A+B). SIMPLE_MOVEMENT (7) / RIGHT_ONLY (5) drop the
useless combos (B alone, left+A/B, up/down, ...).

Also logs *where episodes end* (death x_pos distribution) to locate the wall.

    python src/diag_mario_actions.py --actions simple --timesteps 1000000 --out /app/data/models/_diag_simple

--actions canonical|simple|right ; --spr-coef 0 = plain PPO (isolate the action space).
"""
import perf  # first: BLAS thread limits before torch/numpy

import argparse
import os

import gym_super_mario_bros
import numpy as np
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from shimmy import GymV21CompatibilityV0
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor

from game_env import CANONICAL_ACTIONS, RgbCapture, _obs_pipeline
from nets import policy_kwargs_for
from train_ppo_spr import PPOSPR, ProgressStats

MODELS_DIR = "/app/data/models"
TB_DIR = "/app/data/tb"

ACTION_SETS = {
    "canonical": [sorted(c) or ["NOOP"] for c in CANONICAL_ACTIONS],   # 14
    "simple": SIMPLE_MOVEMENT,                                          # 7
    "right": RIGHT_ONLY,                                                # 5
}


def make_env_fn(movements):
    def _fn():
        env = JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-v0"), movements)
        env = GymV21CompatibilityV0(env=env)
        env = RgbCapture(env)
        return _obs_pipeline(env)
    return _fn


def make_venv(movements, n_envs):
    venv = SubprocVecEnv([make_env_fn(movements) for _ in range(n_envs)])
    venv = VecFrameStack(venv, 4, channels_order="last")
    return VecMonitor(venv)


class DeathStats(BaseCallback):
    """Locate the wall: on every episode end, record where Mario stopped (x_pos)
    and whether he reached the flag. Logs the death-x distribution per rollout —
    if most episodes end near one x, that's the obstacle they can't pass."""

    def _on_rollout_start(self):
        self._deaths, self._flags = [], 0

    def _on_step(self):
        for info in self.locals["infos"]:
            if info.get("flag_get"):
                self._flags += 1
            ep = info.get("episode")
            if ep is not None and "x_pos" in info:
                self._deaths.append(float(info["x_pos"]))
        return True

    def _on_rollout_end(self):
        if not self._deaths:
            return
        d = np.asarray(self._deaths)
        self.logger.record("death/x_mean", float(d.mean()))
        self.logger.record("death/x_median", float(np.median(d)))
        self.logger.record("death/x_p90", float(np.percentile(d, 90)))
        self.logger.record("death/flags", float(self._flags))
        self.logger.record("death/n_episodes", float(len(d)))


def evaluate(model, movements, n_episodes=20):
    venv = make_venv(movements, 1)
    xs, flags = [], 0
    for _ in range(n_episodes):
        obs = venv.reset()
        done, last_x = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = venv.step(action)
            done = bool(dones[0])
            if "x_pos" in infos[0]:
                last_x = float(infos[0]["x_pos"])
            if infos[0].get("flag_get"):
                flags += 1
        xs.append(last_x)
    venv.close()
    return np.asarray(xs), flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", choices=list(ACTION_SETS), default="simple")
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--spr-coef", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--n-epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=None)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    perf.setup_cpu_threads(args.torch_threads)
    os.makedirs(MODELS_DIR, exist_ok=True)
    movements = ACTION_SETS[args.actions]

    venv = make_venv(movements, args.n_envs)
    model = PPOSPR(
        "CnnPolicy", venv, spr_coef=args.spr_coef,
        policy_kwargs=policy_kwargs_for(venv.observation_space),
        n_steps=512, batch_size=64, n_epochs=args.n_epochs, verbose=1,
        learning_rate=1e-4, gamma=args.gamma, gae_lambda=1.0, ent_coef=0.01,
        tensorboard_log=TB_DIR, device="cpu", seed=args.seed,
    )
    tb_name = f"mario_diag_{args.actions}"
    print(f"[diag] actions={args.actions} ({len(movements)}) spr={args.spr_coef} "
          f"gamma={args.gamma} seed={args.seed} steps={args.timesteps}", flush=True)
    model.learn(total_timesteps=args.timesteps,
                callback=[ProgressStats("x_pos"), DeathStats()],
                tb_log_name=tb_name)

    xs, flags = evaluate(model, movements, args.eval_episodes)
    print(f"[diag] EVAL actions={args.actions}: x_pos mean={xs.mean():.0f} "
          f"max={xs.max():.0f} median={np.median(xs):.0f} "
          f"flags={flags}/{args.eval_episodes}", flush=True)

    if args.out:
        model.save(args.out)
        print(f"[diag] saved {args.out}.zip", flush=True)


if __name__ == "__main__":
    main()
