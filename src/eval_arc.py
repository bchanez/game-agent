"""Evaluate a trained PPO model on an ARC game: mean/max levels completed, wins.

    python src/eval_arc.py <model.zip> <game> [episodes] [frame_stack]

Raw grid obs (no VecTransposeImage); scores by the game's own `levels_completed`
and WIN state — the same generic signal training uses.
"""
import perf  # first: BLAS thread limits

import argparse

import numpy as np
from stable_baselines3 import PPO

from game_env import make_venv
from games import get_game


def evaluate(model_path, game, episodes=20, frame_stack=1, max_steps=1000,
             recurrent=False):
    venv = make_venv(get_game(game), 1, frame_stack=frame_stack)
    if recurrent:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(model_path, device="cpu")
    else:
        model = PPO.load(model_path, device="cpu")
    levels, wins = [], 0
    for _ in range(episodes):
        obs = venv.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)     # LSTM state carried per step
        done, steps, best, won = False, 0, 0, False
        while not done and steps < max_steps:
            if recurrent:
                action, lstm_states = model.predict(
                    obs, state=lstm_states, episode_start=episode_starts,
                    deterministic=False)
            else:
                action, _ = model.predict(obs, deterministic=False)
            obs, r, dones, infos = venv.step(action)
            episode_starts = dones
            best = max(best, infos[0].get("levels_completed", 0))
            won = won or bool(infos[0].get("won", False))
            done = bool(dones[0])
            steps += 1
        levels.append(best)
        wins += int(won)
    venv.close()
    return float(np.mean(levels)), int(np.max(levels)), wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("game")
    ap.add_argument("episodes", nargs="?", type=int, default=20)
    ap.add_argument("frame_stack", nargs="?", type=int, default=1)
    ap.add_argument("--recurrent", action="store_true", help="load with RecurrentPPO")
    args = ap.parse_args()
    mean_l, max_l, wins = evaluate(args.model, args.game, args.episodes,
                                   args.frame_stack, recurrent=args.recurrent)
    print(f"{args.game}: levels_completed mean {mean_l:.2f}  max {max_l}  "
          f"wins {wins}/{args.episodes}", flush=True)


if __name__ == "__main__":
    main()
