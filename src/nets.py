"""Observation-space-driven network choices, so training stays game-agnostic.

Mario/Atari feed an 84x84 stacked-frame *image* (channels-last, 1-4 channels) →
SB3's default NatureCNN is right. ARC-AGI-3 feeds a one-hot color *grid*
(channels-first, 16 channels, 64x64) → not an image (NatureCNN's 3-4 channel /255
assumptions don't hold), so it needs its own small CNN. The dispatch keys off the
obs *shape*, never the game name — the boundary holds.
"""
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from objects import FEAT_DIM, object_features


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


class ObjectCentricExtractor(BaseFeaturesExtractor):
    """Perceive the grid as a SET of objects (connected components) and encode it with
    a permutation-invariant DeepSets net. Objecthood is a generic structural transform
    (no game knowledge, non-differentiable); the *encoding* is learned. The bet: a set
    of entities + relations transfers across unseen games far better than raw cells
    (ROADMAP Phase 1). Objects are read from the latest frame in the stack."""

    def __init__(self, observation_space, features_dim=256, max_objects=32, hidden=128):
        super().__init__(observation_space, features_dim)
        self.max_objects = max_objects
        self.phi = nn.Sequential(nn.Linear(FEAT_DIM, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(2 * hidden, features_dim), nn.ReLU())

    def _featurize(self, obs):
        # obs (B, N, H, W) float grid -> object features on the latest frame. The CC
        # parse is numpy/scipy (no grad) — gradients start at phi.
        arr = obs.detach().cpu().numpy().astype(np.uint8)
        b = arr.shape[0]
        feats = np.zeros((b, self.max_objects, FEAT_DIM), np.float32)
        masks = np.zeros((b, self.max_objects), np.float32)
        for i in range(b):
            f, m, _ = object_features(arr[i, -1], self.max_objects)
            feats[i], masks[i] = f, m
        return (torch.as_tensor(feats, device=obs.device),
                torch.as_tensor(masks, device=obs.device))

    def forward(self, obs):
        feats, masks = self._featurize(obs)              # (B,K,F), (B,K)
        h = self.phi(feats) * masks.unsqueeze(-1)        # zero padded slots
        summed = h.sum(dim=1)
        maxed = (h + (1.0 - masks).unsqueeze(-1) * -1e9).max(dim=1).values
        maxed = torch.nan_to_num(maxed, neginf=0.0)      # grids with 0 objects
        return self.rho(torch.cat([summed, maxed], dim=1))


class HybridGridObjectExtractor(BaseFeaturesExtractor):
    """Both worlds: the color-embedding CNN (keeps exact spatial layout, which
    navigation needs) concatenated with the object-centric DeepSets (entities +
    relations). A pure object-set can blur the precise geometry pathfinding needs;
    the hybrid adds object understanding without discarding the spatial map."""

    def __init__(self, observation_space, features_dim=256, max_objects=32):
        super().__init__(observation_space, features_dim)
        half = features_dim // 2
        self.grid = GridCNN(observation_space, features_dim=features_dim - half)
        self.objs = ObjectCentricExtractor(observation_space, features_dim=half,
                                           max_objects=max_objects)

    def forward(self, obs):
        return torch.cat([self.grid(obs), self.objs(obs)], dim=1)


_GRID_ENCODERS = {"grid": GridCNN, "object": ObjectCentricExtractor,
                  "hybrid": HybridGridObjectExtractor}


def policy_kwargs_for(observation_space, encoder="grid"):
    """policy_kwargs for a PPO policy, chosen from the obs space. Empty (SB3 defaults)
    for images; for grids, one of the grid encoders (color CNN / object DeepSets /
    hybrid), all with no image-norm."""
    if is_grid_space(observation_space):
        return dict(features_extractor_class=_GRID_ENCODERS[encoder],
                    features_extractor_kwargs=dict(features_dim=256),
                    normalize_images=False)
    return {}
