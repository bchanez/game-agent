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

# Discover each game's action layout from the emulator itself: ALE reports its
# action names (e.g. 'RIGHTFIRE') via get_action_meanings(), so we match each
# canonical combo to a name instead of hardcoding indices. This auto-adapts to
# any ALE game — and to a game's own minimal action set (an absent combo like
# 'RIGHTFIRE' falls back to 'RIGHT', then NOOP). The joystick has one button, so
# canonical B has no equivalent and is dropped.
def _canonical_to_ale(env):
    index = {name: i for i, name in enumerate(env.unwrapped.get_action_meanings())}
    noop = index.get("NOOP", 0)

    def resolve(combo):
        direction = "".join(d for d in ("UP", "DOWN", "LEFT", "RIGHT") if d.lower() in combo)
        fire = "FIRE" if "A" in combo else ""
        for name in (direction + fire, direction, "NOOP"):
            if name in index:
                return index[name]
        return noop

    return [resolve(combo) for combo in CANONICAL_ACTIONS]


def atari_spec(name, env_id):
    def make_raw_env():
        return gym.make(env_id, frameskip=1, full_action_space=True)
    return GameSpec(name=name, make_raw_env=make_raw_env, action_map=_canonical_to_ale)


montezuma = atari_spec("montezuma", "ALE/MontezumaRevenge-v5")
breakout = atari_spec("breakout", "ALE/Breakout-v5")
