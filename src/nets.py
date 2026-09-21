"""Observation-space-driven network choices, so training stays game-agnostic.

Mario/Atari feed an 84x84 stacked-frame *image* (channels-last, 1-4 channels) →
SB3's default NatureCNN is right. ARC-AGI-3 feeds a one-hot color *grid*
(channels-first, 16 channels, 64x64) → not an image (NatureCNN's 3-4 channel /255
assumptions don't hold), so it needs its own small CNN. The dispatch keys off the
obs *shape*, never the game name — the boundary holds.
"""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def is_grid_space(observation_space):
    """A channels-first multi-channel grid (e.g. ARC one-hot 16x64x64), as opposed
    to a channels-last stacked-frame image (84x84x4)."""
    shape = observation_space.shape
    return len(shape) == 3 and shape[-1] not in (1, 2, 3, 4)


class GridCNN(BaseFeaturesExtractor):
    """CNN for a one-hot color grid (C channels, HxH). Fed 0/1 floats, so the
    policy must run with normalize_images=False (see policy_kwargs_for)."""

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        c, h, _ = observation_space.shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, obs):
        return self.linear(self.cnn(obs))


def policy_kwargs_for(observation_space):
    """policy_kwargs for a PPO policy, chosen from the obs space. Empty (SB3
    defaults) for images; the grid CNN + no image-normalization for grids."""
    if is_grid_space(observation_space):
        return dict(features_extractor_class=GridCNN,
                    features_extractor_kwargs=dict(features_dim=256),
                    normalize_images=False)
    return {}
