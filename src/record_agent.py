"""Record a trained PPO agent playing one episode, as an mp4.

Run (inside the container):
    python src/record_agent.py --model data/models/mario_ppo_final.zip
"""
import argparse
import os

import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecTransposeImage

from mario_env import make_venv

OUT = "/app/data/mario_ppo.mp4"
MAX_STEPS = 3000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/app/data/models/mario_ppo_final.zip")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    venv = make_venv(1)
    # SB3 auto-transposes image obs to channels-first during training, so the
    # loaded model expects that layout — replicate it here for prediction.
    venv = VecTransposeImage(venv)
    model = PPO.load(args.model)

    obs = venv.reset()
    frames = [venv.get_attr("last_rgb")[0].copy()]
    max_x, steps, done = 0, 0, False
    while not done and steps < MAX_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = venv.step(action)
        frames.append(venv.get_attr("last_rgb")[0].copy())
        max_x = max(max_x, infos[0].get("x_pos", 0))
        done = bool(dones[0])
        steps += 1
    venv.close()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    imageio.mimsave(args.out, frames, fps=30)
    print(f"Episode: {steps} steps, furthest x_pos: {max_x}", flush=True)
    print(f"Saved video: {args.out}", flush=True)


if __name__ == "__main__":
    main()
