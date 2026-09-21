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
from dataclasses import dataclass, field
from typing import Callable

import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, NormalizeReward, ResizeObservation
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor,
)


# The canonical controller: NES d-pad + two buttons (A = jump/fire, B =
# run/secondary), a superset of every console we target. A single policy can
# only transfer a learned action head across games if output k means the *same*
# button combo in every game — so the index below is fixed, and each adapter
# maps it to its own raw action (a combo it can't do -> its local NOOP). This is
# what lets several games share one Discrete(N) head in a mixed vec-env.
CANONICAL_ACTIONS = [
    frozenset(),                    # 0  NOOP
    frozenset({"A"}),               # 1  jump / fire
    frozenset({"B"}),               # 2  run / secondary
    frozenset({"right"}),           # 3
    frozenset({"right", "A"}),      # 4
    frozenset({"right", "B"}),      # 5
    frozenset({"right", "A", "B"}), # 6
    frozenset({"left"}),            # 7
    frozenset({"left", "A"}),       # 8
    frozenset({"left", "B"}),       # 9
    frozenset({"up"}),              # 10
    frozenset({"down"}),            # 11
    frozenset({"up", "A"}),         # 12
    frozenset({"down", "A"}),       # 13
]
N_ACTIONS = len(CANONICAL_ACTIONS)


class ActionRemap(gym.ActionWrapper):
    """Exposes the shared `Discrete(N_ACTIONS)` head and translates each canonical
    action id to this game's local raw action, so every game speaks the same
    action language and can share one policy in a mixed vec-env."""

    def __init__(self, env, action_map):
        super().__init__(env)
        self._map = action_map
        self.action_space = gym.spaces.Discrete(N_ACTIONS)

    def action(self, action):
        return self._map[int(action)]


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
                      None if the game exposes no such key (e.g. Atari) — eval
                      then falls back to episode return.
        success_key:  info-dict key that flags an episode as "won" (for eval).
                      None if the game has no explicit win condition.
        action_map:   canonical id -> local raw-env action index (see
                      CANONICAL_ACTIONS). None means identity — the raw env
                      already exposes the canonical actions in order. May also
                      be a callable `env -> list`, to derive the map from the
                      built env (e.g. from its reported action meanings).
    """
    name: str
    make_raw_env: Callable[[], gym.Env]
    progress_key: str = None
    success_key: str = None
    action_map: object = None


def _obs_pipeline(env):
    env = MaxAndSkipEnv(env, skip=4)
    env = ResizeObservation(env, (84, 84))          # resize before grayscale, else the
    env = GrayscaleObservation(env, keep_dim=True)  # single channel gets squeezed
    return env


def make_single_env(spec, normalize_reward=False):
    env = spec.make_raw_env()
    action_map = spec.action_map
    if callable(action_map):                       # derive from the built env
        action_map = action_map(env)
    env = ActionRemap(env, action_map or list(range(N_ACTIONS)))
    env = RgbCapture(env)          # keep raw frame for videos
    env = _obs_pipeline(env)
    if normalize_reward:
        # each env normalizes its own reward by the std of its discounted
        # returns. Per-env (not global VecNormalize) so mixed games — Breakout
        # ~units, Mario ~thousands — reach comparable scale, making a single
        # shared value head learnable. Generic, no per-game constants.
        env = NormalizeReward(env)
    return env


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


def make_multi_venv(specs, n_envs=8, subproc=None, normalize_reward=True):
    """One vec-env mixing several games behind a single policy — the substrate
    for a multi-game agent. Obs (84x84x1) and actions (Discrete(N_ACTIONS)) are
    already unified per game, so a SubprocVecEnv happily runs the mix.

    n_envs is the *total* number of workers (kept at the tuned sweet spot ~8),
    distributed round-robin across `specs` — not n_envs per game, which would
    blow up RAM. VecMonitor sits above per-env NormalizeReward, so logged
    `ep_rew_mean` is normalized; use eval on raw envs for true performance.
    """
    if subproc is None:
        subproc = n_envs > 1
    chosen = [specs[i % len(specs)] for i in range(n_envs)]
    env_fns = [(lambda s=s: make_single_env(s, normalize_reward)) for s in chosen]
    venv = SubprocVecEnv(env_fns) if subproc else DummyVecEnv(env_fns)
    venv = VecFrameStack(venv, 4, channels_order="last")
    venv = VecMonitor(venv)
    return venv
