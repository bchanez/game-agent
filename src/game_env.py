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
import numpy as np
from gymnasium.wrappers import GrayscaleObservation, NormalizeReward, ResizeObservation
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecEnvWrapper, VecFrameStack, VecMonitor,
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
        raw:          True if make_raw_env already returns a model-ready env
                      (its own obs + action space), so the shared pixel pipeline,
                      the d-pad ActionRemap and frame-stacking are all skipped.
                      Used by non-pixel games (e.g. ARC-AGI-3's discrete grid).
    """
    name: str
    make_raw_env: Callable[[], gym.Env]
    progress_key: str = None
    success_key: str = None
    action_map: object = None
    raw: bool = False


def _obs_pipeline(env):
    env = MaxAndSkipEnv(env, skip=4)
    env = ResizeObservation(env, (84, 84))          # resize before grayscale, else the
    env = GrayscaleObservation(env, keep_dim=True)  # single channel gets squeezed
    return env


def make_single_env(spec, normalize_reward=False):
    if spec.raw:                                   # already model-ready (e.g. ARC grid)
        env = spec.make_raw_env()
        return NormalizeReward(env) if normalize_reward else env
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


def make_venv(spec, n_envs=1, subproc=None, frame_stack=1):
    """Vectorized, frame-stacked env ready for Stable-Baselines3, for any game.

    With n_envs>1 the environments run in separate processes (SubprocVecEnv),
    the main way to speed up training on a multi-core CPU (~8 envs is the sweet
    spot here). Scripts using this must be guarded by `if __name__ == "__main__"`
    (SubprocVecEnv re-imports the module).

    Pixel games always stack 4 frames (the classic recipe). A raw grid game stacks
    only if frame_stack>1 (channels-first), so an agent can perceive motion across
    grids when it matters — same generic knob, off by default.
    """
    if subproc is None:
        subproc = n_envs > 1
    env_fns = [(lambda s=spec: make_single_env(s)) for _ in range(n_envs)]
    venv = SubprocVecEnv(env_fns) if subproc else DummyVecEnv(env_fns)
    if not spec.raw:
        venv = VecFrameStack(venv, 4, channels_order="last")
    elif frame_stack > 1:
        venv = VecFrameStack(venv, frame_stack, channels_order="first")
    venv = VecMonitor(venv)
    return venv


class VecPrevActionReward(VecEnvWrapper):
    """RL² conditioning: fold the previous action (one-hot) and previous reward
    into each observation, turning it into a Dict{image, prev_action, prev_reward}.

    This is the ingredient that separates meta-RL from "just add an LSTM": a
    recurrent policy can only *adapt in-context* to an unknown game if it sees the
    consequence of its own actions — the reward it just earned. Sits above
    VecFrameStack so only the image is stacked, and zeroes the carried
    action/reward at each episode boundary so a fresh episode starts blank.

    The reward fed in is whatever the wrapped venv emits (with the multi-game mix
    that is the per-env NormalizeReward output — homogeneous scale across games,
    which is exactly what a single shared input wants)."""

    def __init__(self, venv):
        super().__init__(venv)
        self.observation_space = gym.spaces.Dict({
            "image": venv.observation_space,
            "prev_action": gym.spaces.Box(0.0, 1.0, (N_ACTIONS,), np.float32),
            "prev_reward": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
        })
        self._actions = None
        self._prev_action = np.zeros((venv.num_envs, N_ACTIONS), np.float32)
        self._prev_reward = np.zeros((venv.num_envs, 1), np.float32)

    def _obs(self, image):
        return {
            "image": image,
            "prev_action": self._prev_action.copy(),
            "prev_reward": self._prev_reward.copy(),
        }

    def reset(self):
        self._prev_action[:] = 0.0
        self._prev_reward[:] = 0.0
        return self._obs(self.venv.reset())

    def step_async(self, actions):
        self._actions = np.asarray(actions).astype(int)
        self.venv.step_async(actions)

    def step_wait(self):
        image, rewards, dones, infos = self.venv.step_wait()
        onehot = np.zeros((self.num_envs, N_ACTIONS), np.float32)
        onehot[np.arange(self.num_envs), self._actions] = 1.0
        prev_reward = rewards.reshape(-1, 1).astype(np.float32)
        done = dones.astype(bool)
        # auto-reset stashes the pre-reset image in infos["terminal_observation"];
        # a recurrent policy bootstraps its value on truncation, so it must carry
        # the same Dict shape as a live obs — else obs_to_tensor chokes on a bare
        # array. Use the action/reward of the terminating step (pre-blanking).
        for i in np.nonzero(done)[0]:
            term = infos[i].get("terminal_observation")
            if term is not None:
                infos[i]["terminal_observation"] = {
                    "image": term,
                    "prev_action": onehot[i],
                    "prev_reward": prev_reward[i],
                }
        self._prev_action = onehot.copy()
        self._prev_reward = prev_reward.copy()
        self._prev_action[done] = 0.0                   # next episode's first obs
        self._prev_reward[done] = 0.0                   # starts blank
        return self._obs(image), rewards, dones, infos


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
