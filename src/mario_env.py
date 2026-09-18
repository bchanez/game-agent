"""Shared Super Mario Bros environment factory for the RL agents.

The observation pipeline (the classic Atari/Mario recipe):
    raw RGB 240x256x3
      -> frame-skip 4        (act every 4 frames, max-pool over the last 2)
      -> resize 84x84        (smaller = faster to learn)
      -> grayscale (1 chan)  (colour doesn't help Mario)
      -> stack 4 frames      (so the agent can perceive motion)
    => final observation: 84x84x4 uint8

Old-gym (`gym-super-mario-bros`) is bridged to Gymnasium via shimmy so that
Stable-Baselines3 (which speaks Gymnasium) can consume it.
"""
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from shimmy import GymV21CompatibilityV0
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor


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


def make_single_env():
    env = JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-v0"), SIMPLE_MOVEMENT)
    env = GymV21CompatibilityV0(env=env)   # old gym -> gymnasium
    env = RgbCapture(env)                   # keep raw frame for videos
    env = MaxAndSkipEnv(env, skip=4)
    env = ResizeObservation(env, (84, 84))  # resize before grayscale, else the
    env = GrayscaleObservation(env, keep_dim=True)  # single channel gets squeezed
    return env


def make_venv(n_envs=1):
    """Vectorized, frame-stacked env ready for Stable-Baselines3."""
    venv = DummyVecEnv([make_single_env for _ in range(n_envs)])
    venv = VecFrameStack(venv, 4, channels_order="last")
    venv = VecMonitor(venv)
    return venv
