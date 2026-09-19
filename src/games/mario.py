"""Adapter #1 — Super Mario Bros (NES).

`gym-super-mario-bros` runs on old-gym, so we bridge it to Gymnasium with
shimmy. This is *all* that's Mario-specific: how to build the raw env, and which
info keys mean "progress" (x position) and "won" (reached the flag).
"""
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from shimmy import GymV21CompatibilityV0

from game_env import GameSpec


def make_raw_env():
    env = JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-v0"), SIMPLE_MOVEMENT)
    return GymV21CompatibilityV0(env=env)   # old gym -> gymnasium


SPEC = GameSpec(
    name="mario",
    make_raw_env=make_raw_env,
    progress_key="x_pos",
    success_key="flag_get",
)
