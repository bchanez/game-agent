"""Atari adapters, via the Arcade Learning Environment (ALE).

One factory builds a `GameSpec` for any Atari game, so adding another is a
single line at the bottom.

ALE v5 defaults to its own `frameskip=4`; we pass `frameskip=1` so the shared
pipeline's `MaxAndSkipEnv(skip=4)` owns the skipping (and max-pools the last 2
frames — the classic Atari recipe). Without this we'd skip 16 frames.

Atari exposes no progress/success info keys (no `x_pos`, no `flag_get`), so both
are left None — `eval_models.py` falls back to episode return.
"""
import ale_py
import gymnasium as gym

from game_env import CANONICAL_ACTIONS, GameSpec

gym.register_envs(ale_py)

# ALE's full action set (`full_action_space=True`) is a fixed 18 in this order,
# the same for every game — so one map serves all Atari games (an unsupported
# action is a no-op in the emulator anyway). The Atari joystick has a single
# button, so canonical B has no equivalent: we drop it and map the residual
# (e.g. right+B -> RIGHT, right+A+B -> RIGHTFIRE).
_ALE = {
    "NOOP": 0, "FIRE": 1, "UP": 2, "RIGHT": 3, "LEFT": 4, "DOWN": 5,
    "UPRIGHT": 6, "UPLEFT": 7, "DOWNRIGHT": 8, "DOWNLEFT": 9,
    "UPFIRE": 10, "RIGHTFIRE": 11, "LEFTFIRE": 12, "DOWNFIRE": 13,
    "UPRIGHTFIRE": 14, "UPLEFTFIRE": 15, "DOWNRIGHTFIRE": 16, "DOWNLEFTFIRE": 17,
}


def _to_ale(combo):
    direction = "".join(d.upper() for d in ("UP", "DOWN", "LEFT", "RIGHT") if d.lower() in combo)
    fire = "FIRE" if "A" in combo else ""            # A = the joystick button; B has none
    return _ALE.get(direction + fire) or _ALE.get(direction) or _ALE["NOOP"]


_ACTION_MAP = [_to_ale(combo) for combo in CANONICAL_ACTIONS]


def atari_spec(name, env_id):
    def make_raw_env():
        return gym.make(env_id, frameskip=1, full_action_space=True)
    return GameSpec(name=name, make_raw_env=make_raw_env, action_map=_ACTION_MAP)


montezuma = atari_spec("montezuma", "ALE/MontezumaRevenge-v5")
breakout = atari_spec("breakout", "ALE/Breakout-v5")
