"""Evaluate trained models over several episodes for a fair comparison.

Reports, per model: mean / max furthest x_pos, and how often Mario reaches the
flag (level completed). x_pos ~3200 = end of World 1-1.

    python src/eval_models.py --episodes 10
"""
import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecTransposeImage

from mario_env import make_venv

MODELS = {
    "PPO baseline": "/app/data/models/mario_ppo_final.zip",
    "PPO + curiosity": "/app/data/models/mario_curiosity_final.zip",
    "pure curiosity": "/app/data/models/mario_pure_curiosity_final.zip",
}


def evaluate(model_path, episodes, max_steps=4000):
    venv = VecTransposeImage(make_venv(1))
    model = PPO.load(model_path)
    xs, flags = [], 0
    for _ in range(episodes):
        obs = venv.reset()
        done, steps, max_x = False, 0, 0
        got_flag = False
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, dones, infos = venv.step(action)
            max_x = max(max_x, infos[0].get("x_pos", 0))
            got_flag = got_flag or bool(infos[0].get("flag_get", False))
            done = bool(dones[0])
            steps += 1
        xs.append(max_x)
        flags += int(got_flag)
    venv.close()
    return np.mean(xs), np.max(xs), flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()

    print(f"Evaluating over {args.episodes} episodes each "
          f"(x_pos ~3200 = end of 1-1)\n")
    print(f"{'model':<18} {'mean x':>8} {'max x':>7} {'flags':>7}")
    print("-" * 44)
    for name, path in MODELS.items():
        mean_x, max_x, flags = evaluate(path, args.episodes)
        print(f"{name:<18} {mean_x:>8.0f} {max_x:>7.0f} {flags:>4}/{args.episodes}", flush=True)


if __name__ == "__main__":
    main()
