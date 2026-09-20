"""SPR — Self-Predictive Representations as a PPO auxiliary task.

Stage B of Phase 8.5 (see ROADMAP.md / FINDINGS.md). Stage A's frozen,
reconstruction-trained encoder had two flaws: the MSE loss dropped small moving
objects (the ball), and freezing it caused distribution shift as the policy
improved. SPR fixes both by learning the representation *from prediction*, learnt
*jointly* with PPO:

    predict the next latent from (this latent, action) — never reconstruct pixels.

To predict the future you must encode what moves and matters (the ball); a static
background predicts itself, so no capacity is wasted redrawing it. And the encoder
keeps training with the policy, so it never goes stale.

Collapse guard (the BYOL/SPR trick): the prediction *target* comes from an EMA
copy of the encoder with a stop-gradient, so the trivial "map everything to one
constant" solution is unavailable — the target keeps moving on its own.

These are game-agnostic nn.Modules; the SB3 glue lives in train_ppo_spr.py.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


@torch.no_grad()
def ema_update(target, online, tau):
    """target <- (1-tau)*target + tau*online, in place."""
    for t, o in zip(target.parameters(), online.parameters()):
        t.mul_(1.0 - tau).add_(o, alpha=tau)


class SPRHead(nn.Module):
    """Predicts the next latent from (latent, action) and scores it against an
    EMA target latent. Loss is negative cosine similarity (BYOL-style)."""

    def __init__(self, feature_dim, n_actions, proj_dim=128):
        super().__init__()
        self.n_actions = n_actions
        self.transition = _mlp([feature_dim + n_actions, feature_dim, feature_dim])
        self.proj_online = _mlp([feature_dim, proj_dim, proj_dim])
        self.pred_online = _mlp([proj_dim, proj_dim])
        self.proj_target = copy.deepcopy(self.proj_online)
        for p in self.proj_target.parameters():
            p.requires_grad_(False)

    def forward(self, z_t, actions, z_tp1_target):
        a = F.one_hot(actions.long().flatten(), self.n_actions).float()
        pred_next = self.transition(torch.cat([z_t, a], dim=1))
        p = self.pred_online(self.proj_online(pred_next))
        with torch.no_grad():
            y = self.proj_target(z_tp1_target)          # target is already EMA-encoded
        p = F.normalize(p, dim=1)
        y = F.normalize(y, dim=1)
        return (2.0 - 2.0 * (p * y).sum(dim=1)).mean()

    def update_target(self, tau):
        ema_update(self.proj_target, self.proj_online, tau)
