"""Random Network Distillation (RND) — curiosity as an intrinsic reward.

Idea (Burda et al., 2018): a *target* network with frozen random weights maps
each observation to an embedding. A *predictor* network is trained to reproduce
that embedding. The prediction error is the intrinsic reward:

    novel observation  -> predictor is wrong -> high error -> high curiosity
    familiar one       -> predictor learned it -> low error -> low curiosity

So the agent is pulled toward states it hasn't mastered yet. This is a simpler,
more stable cousin of ICM (the Mario curiosity paper), and a good first step.

`RNDReward` is a VecEnvWrapper: it leaves the environment untouched but replaces
the reward with  extrinsic_coef * game_reward + intrinsic_coef * curiosity.
Set extrinsic_coef=0 for the "pure curiosity" experiment.
"""
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper


class RunningMeanStd:
    """Online mean/variance (Welford), used to normalize obs and rewards."""

    def __init__(self, shape=(), eps=1e-4):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = eps

    def update(self, x):
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.var = M2 / total
        self.count = total


class _RNDNet(nn.Module):
    """Small Nature-style CNN mapping an 84x84 single frame to an embedding."""

    def __init__(self, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class RNDReward(VecEnvWrapper):
    def __init__(self, venv, intrinsic_coef=1.0, extrinsic_coef=1.0,
                 lr=1e-4, device="cpu", train_epochs=4, train_batch=256):
        super().__init__(venv)
        self.intrinsic_coef = intrinsic_coef
        self.extrinsic_coef = extrinsic_coef
        self.device = torch.device(device)
        self.train_epochs = train_epochs
        self.train_batch = train_batch

        self.target = _RNDNet().to(self.device).eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = _RNDNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)

        # obs pixels are normalized per-pixel; intrinsic reward by its running std
        self.obs_rms = RunningMeanStd(shape=(1, 84, 84))
        self.rew_rms = RunningMeanStd(shape=())

        # frames seen since the last predictor update; drained by train_predictor
        # so the backward pass stays off the env-stepping hot path.
        self._obs_buffer = []

    def reset(self):
        return self.venv.reset()

    def _prep(self, obs):
        # obs: (n, 84, 84, 4) uint8 channels-last -> use the latest frame only
        frame = obs[..., -1:].astype(np.float32)          # (n,84,84,1)
        frame = np.transpose(frame, (0, 3, 1, 2))         # (n,1,84,84)
        self.obs_rms.update(frame)
        norm = (frame - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        norm = np.clip(norm, -5.0, 5.0)
        return torch.as_tensor(norm, dtype=torch.float32, device=self.device)

    def _curiosity(self, obs):
        x = self._prep(obs)
        # forward only: reward is cheap and stays on the acting hot path; the
        # predictor's backward pass is deferred to train_predictor (per rollout).
        with torch.no_grad():
            target_feat = self.target(x)
            pred_feat = self.predictor(x)
        error = (pred_feat - target_feat).pow(2).mean(dim=1)
        self._obs_buffer.append(x)

        intr = error.cpu().numpy()
        self.rew_rms.update(intr)
        return intr / np.sqrt(self.rew_rms.var + 1e-8)

    def train_predictor(self):
        """Fit the predictor on everything seen since the last call, in
        shuffled minibatches. Runs once per PPO rollout, not once per step."""
        if not self._obs_buffer:
            return
        data = torch.cat(self._obs_buffer, dim=0)
        self._obs_buffer = []
        n = data.shape[0]
        for _ in range(self.train_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.train_batch):
                batch = data[perm[start:start + self.train_batch]]
                with torch.no_grad():
                    target_feat = self.target(batch)
                pred_feat = self.predictor(batch)
                loss = (pred_feat - target_feat).pow(2).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        intr = self._curiosity(obs)
        total = self.extrinsic_coef * rews + self.intrinsic_coef * intr
        for i, info in enumerate(infos):
            info["intrinsic"] = float(intr[i])
            info["extrinsic"] = float(rews[i])
        return obs, total, dones, infos

    # --- save / load the "second brain" (RND nets + normalization stats) so
    # --- curiosity training can be resumed instead of restarting from scratch.
    def save_rnd(self, path):
        torch.save({
            "target": self.target.state_dict(),
            "predictor": self.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_rms": (self.obs_rms.mean, self.obs_rms.var, self.obs_rms.count),
            "rew_rms": (self.rew_rms.mean, self.rew_rms.var, self.rew_rms.count),
        }, path)

    def load_rnd(self, path):
        # our own file (contains numpy normalization stats) -> weights_only=False
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.target.load_state_dict(ck["target"])
        self.predictor.load_state_dict(ck["predictor"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.obs_rms.mean, self.obs_rms.var, self.obs_rms.count = ck["obs_rms"]
        self.rew_rms.mean, self.rew_rms.var, self.rew_rms.count = ck["rew_rms"]


class RNDTrainCallback(BaseCallback):
    """Trains the RND predictor once per rollout, after collection.

    Kept separate from reward computation so the predictor's backward pass
    never blocks env stepping — the main speed win over per-step training.
    """

    def __init__(self, rnd):
        super().__init__()
        self._rnd = rnd

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        self._rnd.train_predictor()
