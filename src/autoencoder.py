"""Convolutional autoencoder — the *representation* half of the world model.

Stage A of Phase 8.5 (see ROADMAP.md): instead of PPO's CNN re-learning to see
from raw pixels, we learn a compact latent of a single frame offline, then train
the policy on those latents. The encoder mirrors the Nature-CNN used elsewhere
(see rnd.py) so a frame and its RND embedding share the same visual frontend.

The agent never sees hand-crafted features — the latent is learned purely from
pixels by reconstruction, so the *no human indication* principle holds.

    84x84x1 frame -> Encoder -> latent (D dims) -> Decoder -> 84x84x1 frame

The decoder exists only to train the encoder; the policy keeps the encoder alone.
"""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        return self.fc(self.conv(x))


class Decoder(nn.Module):
    """Mirrors the encoder: latent -> 84x84x1. The transposed convs invert the
    encoder's spatial reductions exactly (7 -> 9 -> 20 -> 84)."""

    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 8, stride=4), nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 64, 7, 7)
        return self.deconv(x)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
