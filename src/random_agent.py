"""A random agent plays a game and saves a video.

Runs any registered game's raw env with random actions — a fast check that the
env works inside Docker before investing in training.

Run (inside the container):
    python src/random_agent.py --game mario
"""
import argparse
import os

import imageio

from games import get_game

STEPS = 800
OUT = "/app/data/random_agent.mp4"
FPS = 30


def unpack_reset(ret):
    # gym <0.26 returns obs; gym >=0.26 returns (obs, info)
    return ret[0] if isinstance(ret, tuple) else ret


def unpack_step(ret):
    # gym <0.26: (obs, reward, done, info)
    # gym >=0.26: (obs, reward, terminated, truncated, info)
    if len(ret) == 5:
        obs, reward, terminated, truncated, info = ret
        return obs, reward, terminated or truncated, info
    obs, reward, done, info = ret
    return obs, reward, done, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="mario", help="which game to run")
    args = ap.parse_args()

    print("Creating env...", flush=True)
    env = get_game(args.game).make_raw_env()
    print("action_space:", env.action_space, flush=True)

    obs = unpack_reset(env.reset())
    print("first obs:", obs.shape, obs.dtype, flush=True)

    # nes-py returns a *view* into the emulator's screen buffer (reused every
    # step and freed on close()), so every frame must be copied before storing.
    frames = [obs.copy()]
    max_x = 0
    for i in range(STEPS):
        action = env.action_space.sample()
        obs, reward, done, info = unpack_step(env.step(action))
        frames.append(obs.copy())
        max_x = max(max_x, info.get("x_pos", 0))
        if done:
            obs = unpack_reset(env.reset())
            frames.append(obs.copy())
    env.close()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    imageio.mimsave(OUT, frames, fps=FPS)
    print(f"Ran {STEPS} steps, furthest x_pos reached: {max_x}", flush=True)
    print(f"Saved video: {OUT} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    main()
