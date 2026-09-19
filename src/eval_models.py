"""Evaluate the three trained variants over several episodes, for a fair comparison.

Reports, per model: mean / max progress (the game's progress_key, e.g. x_pos for
Mario) and how often the agent "wins" (the game's success_key, e.g. reaching the
flag). x_pos ~3200 = end of Mario World 1-1.

    python src/eval_models.py --episodes 10
    python src/eval_models.py --game mario --episodes 10
"""
import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecTransposeImage

from game_env import make_venv
from games import get_game

# The three standard variants, as <label>: <model-file suffix>.
VARIANTS = {
    "PPO baseline": "ppo",
    "PPO + curiosity": "curiosity",
    "pure curiosity": "pure_curiosity",
}


def evaluate(spec, model_path, episodes, max_steps=4000):
    venv = VecTransposeImage(make_venv(spec, 1))
    model = PPO.load(model_path)
    scores, wins = [], 0
    for _ in range(episodes):
        obs = venv.reset()
        done, steps = False, 0
        best_progress, ep_return, won = 0, 0.0, False
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, dones, infos = venv.step(action)
            ep_return += float(reward[0])
            if spec.progress_key:
                best_progress = max(best_progress, infos[0].get(spec.progress_key, 0))
            if spec.success_key:
                won = won or bool(infos[0].get(spec.success_key, False))
            done = bool(dones[0])
            steps += 1
        # progress_key games (Mario) score by furthest reached; others by return.
        scores.append(best_progress if spec.progress_key else ep_return)
        wins += int(won)
    venv.close()
    return np.mean(scores), np.max(scores), wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="mario", help="which game to evaluate")
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()

    spec = get_game(args.game)
    metric = spec.progress_key or "episode return"
    print(f"Evaluating {spec.name} over {args.episodes} episodes each "
          f"(score = {metric})\n")
    print(f"{'model':<18} {'mean':>8} {'max':>7} {'wins':>7}")
    print("-" * 44)
    for label, suffix in VARIANTS.items():
        path = f"/app/data/models/{spec.name}_{suffix}_final.zip"
        if not os.path.exists(path):
            print(f"{label:<18} {'(no model: ' + os.path.basename(path) + ')':>22}")
            continue
        mean_x, max_x, wins = evaluate(spec, path, args.episodes)
        print(f"{label:<18} {mean_x:>8.0f} {max_x:>7.0f} {wins:>4}/{args.episodes}", flush=True)


if __name__ == "__main__":
    main()
