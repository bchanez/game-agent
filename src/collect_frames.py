"""Collect a dataset of game frames for training the offline autoencoder.

A random agent is enough here: we only need the *distribution of pixels* the
encoder must learn to compress, not good play. Frames come through the exact same
observation pipeline the policy will use (make_venv), so the encoder trains on
what the agent will actually see.

Run (inside the container):
    python src/collect_frames.py --game breakout --frames 100000 --n-envs 8

Saves the latest grayscale frame of each step to data/frames/<game>.npz (uint8,
84x84), compressed, plus the *previous* frame (the one before it in the stack) so
a motion-weighted loss can later tell what moved between two steps. ~100k frames
is a few minutes on CPU.
"""
import argparse
import os

import numpy as np

from game_env import make_venv
from games import get_game

FRAMES_DIR = "/app/data/frames"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="breakout", help="which game to sample from")
    ap.add_argument("--frames", type=int, default=100_000, help="total frames to collect")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--out", default=None, help="override output path (else data/frames/<game>.npz)")
    args = ap.parse_args()

    spec = get_game(args.game)
    os.makedirs(FRAMES_DIR, exist_ok=True)
    out = args.out or os.path.join(FRAMES_DIR, f"{spec.name}.npz")

    venv = make_venv(spec, args.n_envs)
    obs = venv.reset()

    buf = np.empty((args.frames, 84, 84), dtype=np.uint8)
    prev = np.empty((args.frames, 84, 84), dtype=np.uint8)
    n = 0
    print(f"Collecting {args.frames} frames from {spec.name} "
          f"({args.n_envs} random envs)...", flush=True)
    while n < args.frames:
        actions = [venv.action_space.sample() for _ in range(args.n_envs)]
        obs, _, _, _ = venv.step(actions)
        # keep the latest frame and its predecessor so a motion-weighted loss can
        # later tell what moved between the two
        take = min(args.n_envs, args.frames - n)
        buf[n:n + take] = obs[..., -1][:take]
        prev[n:n + take] = obs[..., -2][:take]
        n += take
        if n % 10_000 < args.n_envs:
            print(f"  {n}/{args.frames}", flush=True)

    venv.close()
    np.savez_compressed(out, frames=buf, prev=prev)
    print(f"Saved {n} frames (+ prev) to {out}", flush=True)


if __name__ == "__main__":
    main()
