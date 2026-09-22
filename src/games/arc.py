"""ARC-AGI-3 adapter — the interactive ARC Prize 2026 track (see docs/arc-agi-3.md).

An ARC-AGI-3 game is a `64×64` grid of color indices (`0–15`), a handful of
semantic actions whose meaning varies per game, sparse/absent reward, and a
WIN/GAME_OVER/NOT_FINISHED state. None of the Mario/Atari pixel pipeline applies,
so this adapter builds a **model-ready** Gymnasium env directly (`raw=True` in the
spec, so `game_env` skips frame-skip/resize/grayscale/stack and the d-pad remap).

v1 is deliberately minimal (see the porting plan): the **simple** actions only
(ACTION1-5, ACTION7); the spatial-click ACTION6 and legal-action masking come
next. Reward is generic — the SDK's own `levels_completed` delta, no per-game
engineering — keeping the "no human indication" principle.

The `arc_agi` SDK is imported lazily so mario/atari runs don't pay for it (and so
`import games` stays cheap): `Arcade()` fetches an anon key + game files online.
"""
import gymnasium as gym
import numpy as np

from game_env import GameSpec

N_COLORS = 16
GRID = 64


def _simple_actions():
    from arcengine import GameAction
    # fixed vocabulary so one policy speaks the same actions across games; RESET is
    # env.reset(), and the spatial ACTION6 is deferred to v2.
    return [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
            GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION7]


class ArcEnv(gym.Env):
    def __init__(self, game_id, max_steps=1000):
        import arc_agi

        self._actions = _simple_actions()
        self._env = arc_agi.Arcade().make(game_id)
        self.action_space = gym.spaces.Discrete(len(self._actions))
        # raw color-index grid (0..15), NOT one-hot: cheap to stack across frames,
        # and a learned embedding in the net beats 16 one-hot channels per frame.
        self.observation_space = gym.spaces.Box(0, N_COLORS - 1, (1, GRID, GRID), np.uint8)
        self._max_steps = max_steps
        self._steps = 0
        self._levels = 0

    def _obs(self, fd):
        return np.asarray(fd.frame)[0].astype(np.uint8)[None]   # (1,64,64), colors 0..15

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        fd = self._env.reset()
        self._steps = 0
        self._levels = fd.levels_completed
        return self._obs(fd), {}

    def step(self, action):
        from arcengine import GameState

        fd = self._env.step(self._actions[int(action)])
        self._steps += 1
        reward = float(fd.levels_completed - self._levels)  # sparse: only on progress
        self._levels = fd.levels_completed
        terminated = fd.state in (GameState.WIN, GameState.GAME_OVER)
        truncated = self._steps >= self._max_steps
        info = {"levels_completed": fd.levels_completed,
                "won": fd.state == GameState.WIN}
        return self._obs(fd), reward, terminated, truncated, info


ls20 = GameSpec(
    name="arc_ls20",
    make_raw_env=lambda: ArcEnv("ls20"),
    progress_key="levels_completed",
    success_key="won",
    raw=True,
)
