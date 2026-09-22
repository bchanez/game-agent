"""Unit tests for the toolbox plumbing — pure logic, no training, no network.

Plain-python (no pytest dependency): run with `python tests/test_tools.py`. These
lock down the pieces we built fast (auto-config rules, the SIL episode capture, the
recurrent terminal_observation fix, the grid CNN), so a refactor can't silently
break them. Convention (CLAUDE.md): test_should_<outcome>_when_<condition>, with
explicit given/when/then.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gymnasium as gym
import numpy as np
import torch

from auto_config import configure
from game_env import VecPrevActionReward, N_ACTIONS
from nets import GridCNN
from train_ppo_spr import PPOSPR, SILCollector


def test_should_keep_sil_and_only_recommend_memory_when_sparse_grid():
    # given a sparse, non-Markov grid game (like ARC ls20)
    signals = {"reward_density": 0.0, "non_markov": 0.7, "is_grid": True}
    # when
    cfg = configure(signals)
    # then curiosity + SIL stay on (SIL beats memory-without-SIL); memory is only
    # *recommended*, not auto-enabled, and no frame-stack on a grid
    assert cfg["intrinsic_coef"] > 0 and cfg["sil_coef"] > 0
    assert cfg["recurrent"] is False
    assert cfg["memory_recommended"] is True
    assert cfg["frame_stack"] == 1


def test_should_disable_curiosity_and_sil_when_reward_dense():
    # given a dense-reward game (like Mario)
    signals = {"reward_density": 0.72, "non_markov": 1.0, "is_grid": False}
    # when
    cfg = configure(signals)
    # then curiosity/SIL are off, SPR stays on, and it's not recurrent (pixel)
    assert cfg["intrinsic_coef"] == 0.0 and cfg["sil_coef"] == 0.0
    assert cfg["spr_coef"] > 0
    assert cfg["recurrent"] is False


def test_should_frame_stack_when_non_markov_pixel():
    # given a sparse non-Markov *pixel* game (like Breakout: hidden ball velocity)
    signals = {"reward_density": 0.0, "non_markov": 0.4, "is_grid": False}
    # when
    cfg = configure(signals)
    # then it stacks frames (velocity) and stays non-recurrent
    assert cfg["frame_stack"] == 4
    assert cfg["recurrent"] is False
    assert cfg["sil_coef"] > 0


def test_should_capture_winning_episode_only_when_reward_earned():
    # given a SIL collector on a 1-env model
    class Model:
        def __init__(self):
            from collections import deque
            self._sil_obs, self._sil_act, self._sil_ret = deque(), deque(), deque()
            self.gamma = 0.99
            self._last_obs = np.zeros((1, 1, 4, 4), np.uint8)
        sil_push_episode = PPOSPR.sil_push_episode

    cb = SILCollector()
    cb.model = Model()
    cb._obs, cb._act, cb._ext = [[]], [[]], [[]]

    def step(a, ext, done):
        cb.locals = {"actions": np.array([a]), "dones": np.array([done]),
                     "infos": [{"extrinsic": ext}], "rewards": np.array([ext])}
        cb._on_step()

    # when a winning episode (earns +1) ends
    step(0, 0.0, False); step(1, 0.0, False); step(2, 1.0, True)
    # then its transitions are kept, with discounted returns back from the win
    assert len(cb.model._sil_obs) == 3
    assert cb.model._sil_ret[-1] == 1.0 and cb.model._sil_ret[0] < 1.0

    # when a losing episode (no reward) ends
    step(0, 0.0, False); step(1, 0.0, True)
    # then nothing new is kept
    assert len(cb.model._sil_obs) == 3


def test_should_make_terminal_observation_a_dict_when_done():
    # given a VecPrevActionReward over a fake image vec-env
    class FakeVenv:
        num_envs = 2
        observation_space = gym.spaces.Box(0, 255, (84, 84, 1), np.uint8)
        action_space = gym.spaces.Discrete(N_ACTIONS)
        def step_async(self, actions): pass
        def step_wait(self): return self._next
        def reset(self): return np.zeros((2, 84, 84, 1), np.uint8)

    fake = FakeVenv()
    term_img = np.ones((84, 84, 1), np.uint8)
    fake._next = (np.zeros((2, 84, 84, 1), np.uint8), np.array([1.0, 0.0]),
                  np.array([True, False]), [{"terminal_observation": term_img}, {}])
    vpar = VecPrevActionReward(fake)
    # when a step ends env 0
    vpar.step_async(np.array([3, 5]))
    _, _, _, infos = vpar.step_wait()
    # then env 0's terminal_observation is a Dict shaped like a live obs
    term = infos[0]["terminal_observation"]
    assert isinstance(term, dict)
    assert term["image"].shape == (84, 84, 1)
    assert term["prev_action"].shape == (N_ACTIONS,) and term["prev_reward"].shape == (1,)


def test_should_output_features_dim_when_grid_cnn_forward():
    # given a grid CNN over a stacked-grid obs space (4 frames, 64x64)
    space = gym.spaces.Box(0, 15, (4, 64, 64), np.uint8)
    net = GridCNN(space, features_dim=256)
    # when a batch of index grids is fed
    out = net(torch.zeros(2, 4, 64, 64))
    # then it returns (batch, features_dim)
    assert out.shape == (2, 256)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
