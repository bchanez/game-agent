"""Adapter #1 — Super Mario Bros (NES).

`gym-super-mario-bros` runs on old-gym, so we bridge it to Gymnasium with
shimmy. This is *all* that's Mario-specific: how to build the raw env, and which
info keys mean "progress" (x position) and "won" (reached the flag).
"""
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from shimmy import GymV21CompatibilityV0

from game_env import CANONICAL_ACTIONS, GameSpec

# The NES joypad accepts any button combo, so Mario natively speaks the whole
# canonical set (an unused combo like left+A just makes Mario jump left). Build
# JoypadSpace straight from it -> the local action index equals the canonical id
# (identity map). nes-py denotes "no buttons" as ['NOOP'].
_MOVEMENTS = [sorted(combo) or ["NOOP"] for combo in CANONICAL_ACTIONS]


def make_raw_env():
    env = JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-v0"), _MOVEMENTS)
    return GymV21CompatibilityV0(env=env)   # old gym -> gymnasium


SPEC = GameSpec(
    name="mario",
    make_raw_env=make_raw_env,
    progress_key="x_pos",
    success_key="flag_get",
)
