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


N_COLORS = 16


class GridCNN(BaseFeaturesExtractor):
    """CNN for stacked color-index grids (N frames, HxH, values 0..15). Each color
    is mapped through a learned embedding (dim d) rather than one-hot, so N frames
    cost N*d input channels (e.g. 4*4=16) instead of N*16 — cheap frame-stacking
    that still lets the net perceive motion. Fed as floats (normalize_images=False,
    see policy_kwargs_for); cast back to long indices for the embedding."""

    def __init__(self, observation_space, features_dim=256, embed_dim=4):
        super().__init__(observation_space, features_dim)
        n_frames, h, _ = observation_space.shape
        self.embed = nn.Embedding(N_COLORS, embed_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_frames * embed_dim, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *observation_space.shape, dtype=torch.long)
            n_flat = self.cnn(self._embed(dummy)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def _embed(self, obs):
        # (batch, N, H, W) color indices -> (batch, N*embed_dim, H, W)
        b, n, h, w = obs.shape
        e = self.embed(obs.long())                          # (b, N, H, W, d)
        return e.permute(0, 1, 4, 2, 3).reshape(b, -1, h, w)

    def forward(self, obs):
        return self.linear(self.cnn(self._embed(obs)))


def policy_kwargs_for(observation_space):
    """policy_kwargs for a PPO policy, chosen from the obs space. Empty (SB3
    defaults) for images; the grid CNN + no image-normalization for grids."""
    if is_grid_space(observation_space):
        return dict(features_extractor_class=GridCNN,
                    features_extractor_kwargs=dict(features_dim=256),
                    normalize_images=False)
    return {}
