"""LatentObs — swap raw pixels for the frozen encoder's latent, on the fly.

Stage A step 3 (see ROADMAP.md). This VecEnvWrapper sits on top of make_venv
(exactly like RNDReward): it leaves the game untouched but replaces each pixel
observation with its compact latent, so PPO learns on ~256 numbers instead of
~28k pixels.

The venv already frame-stacks 4 consecutive frames (channels-last). We encode
each of the 4 and concatenate, so motion is preserved *in latent space* — the
same reason raw pixels are stacked. Observation goes from (84,84,4) uint8 to
(4*latent_dim,) float32, and the policy becomes an MlpPolicy.

The encoder is frozen (eval, no grad): the representation was learned offline and
must not drift while PPO trains.
"""
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnvWrapper

from autoencoder import Encoder


class LatentObs(VecEnvWrapper):
    def __init__(self, venv, encoder_path, device="cpu"):
        self.device = torch.device(device)
        ckpt = torch.load(encoder_path, map_location=self.device, weights_only=False)
        self.latent_dim = ckpt["latent_dim"]
        self.encoder = Encoder(self.latent_dim).to(self.device).eval()
        self.encoder.load_state_dict(ckpt["encoder"])
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        self.n_stack = venv.observation_space.shape[-1]     # channels-last: last dim is the frame stack
        obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_stack * self.latent_dim,), dtype=np.float32,
        )
        super().__init__(venv, observation_space=obs_space)

    def _encode(self, obs):
        # encode each stacked frame separately then regroup per env, so the policy
        # sees motion as stacked latents (mirrors raw frame-stacking)
        n = obs.shape[0]
        frames = np.transpose(obs, (0, 3, 1, 2)).reshape(-1, 1, 84, 84)
        x = torch.as_tensor(frames, dtype=torch.float32, device=self.device).div_(255.0)
        with torch.no_grad():
            z = self.encoder(x)
        return z.view(n, -1).cpu().numpy().astype(np.float32)

    def reset(self):
        return self._encode(self.venv.reset())

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        return self._encode(obs), rews, dones, infos
