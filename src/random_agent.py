"""Phase 1 sanity check: a random agent plays Super Mario Bros (NES).

Runs the gym-super-mario-bros environment with random actions and saves a
video to data/. This proves the NES environment works inside Docker before
we invest in any training.

Run (inside the container):
    python src/random_agent.py
"""
import os

import imageio
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

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
    print("Creating env...", flush=True)
    env = JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-v0"), SIMPLE_MOVEMENT)
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
