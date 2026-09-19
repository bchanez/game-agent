"""Generic, game-agnostic environment plumbing — the `GameEnv` interface.

The vision (see ROADMAP.md) is one agent that plays *many* games. The trick is
a clean boundary: **pixels in, discrete button out**. Everything game-specific
lives in a `GameSpec` (see `games/`); everything shared lives here.

The shared observation pipeline is the classic Atari/Mario recipe:
    raw RGB
      -> frame-skip 4        (act every 4 frames, max-pool over the last 2)
      -> resize 84x84        (smaller = faster to learn)
      -> grayscale (1 chan)  (colour rarely helps)
      -> stack 4 frames      (so the agent can perceive motion)
    => final observation: 84x84x4 uint8

This pipeline is game-agnostic, so it lives here *once*. Adding a new game is a
new `GameSpec`, not a new pipeline.
"""
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor,
)


class RgbCapture(gym.Wrapper):
    """Stashes the latest raw RGB frame so we can record real-looking videos
    even though the model only sees the downsized grayscale observation."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_rgb = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_rgb = obs
        return obs, reward, terminated, truncated, info


@dataclass
class GameSpec:
    """Everything game-specific, in one place. Adding a game = adding one of these.

    Attributes:
        name:         short id, e.g. "mario" (used to name models/videos).
        make_raw_env: builds a Gymnasium env with **raw RGB obs** and a
                      **discrete action space** — the emulator/adapter, before
                      the shared observation pipeline is applied.
        progress_key: info-dict key measuring how far the agent got (for eval).
        success_key:  info-dict key that flags an episode as "won" (for eval).
    """
    name: str
    make_raw_env: Callable[[], gym.Env]
    progress_key: str = "x_pos"
    success_key: str = "flag_get"


def _obs_pipeline(env):
    env = MaxAndSkipEnv(env, skip=4)
    env = ResizeObservation(env, (84, 84))          # resize before grayscale, else the
    env = GrayscaleObservation(env, keep_dim=True)  # single channel gets squeezed
    return env


def make_single_env(spec):
    env = spec.make_raw_env()
    env = RgbCapture(env)          # keep raw frame for videos
    return _obs_pipeline(env)


def make_venv(spec, n_envs=1, subproc=None):
    """Vectorized, frame-stacked env ready for Stable-Baselines3, for any game.

    With n_envs>1 the environments run in separate processes (SubprocVecEnv),
    the main way to speed up training on a multi-core CPU (~8 envs is the sweet
    spot here). Scripts using this must be guarded by `if __name__ == "__main__"`
    (SubprocVecEnv re-imports the module).
    """
    if subproc is None:
        subproc = n_envs > 1
    env_fns = [(lambda s=spec: make_single_env(s)) for _ in range(n_envs)]
    venv = SubprocVecEnv(env_fns) if subproc else DummyVecEnv(env_fns)
    venv = VecFrameStack(venv, 4, channels_order="last")
    venv = VecMonitor(venv)
    return venv
