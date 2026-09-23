"""Controllability — an inverse-dynamics auxiliary task: "what do I control?".

The step after SPR. SPR learns a *dynamics-aware* latent (predict the next latent),
but it is an unstructured blob: no notion of "me" vs "the world". This head asks a
different, complementary question — given two consecutive latents, **which action
did I just take?** Only the parts of the scene the agent *controls* (its avatar)
move as a deterministic function of the action; the rest (enemies, scrolling,
noise) does not help predict it. So minimizing this cross-entropy shapes the
encoder to locate and track the controllable region — a learned "self vs world"
split, with no labels and no game-specific perception (the *no human indication*
principle holds). This is ICM's inverse model (Pathak et al. 2017) repurposed as a
*representation* objective rather than a curiosity signal.

The payoff is twofold: a representation prior (does controllability-awareness make
PPO more sample-efficient?) and interpretability — the input-gradient of the
predicted action is a saliency map over the frame, and it should light up on the
avatar with zero supervision (see viz_controllability.py).

A game-agnostic nn.Module; the SB3 glue lives in train_ppo_spr.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InverseDynamicsHead(nn.Module):
    """Predict the action taken between two consecutive latents (z_t, z_{t+1})."""

    def __init__(self, feature_dim, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def logits(self, z_t, z_tp1):
        return self.net(torch.cat([z_t, z_tp1], dim=1))

    def forward(self, z_t, z_tp1, actions):
        """Cross-entropy of predicting `actions`, plus the accuracy (for logging —
        how controllable-aware the encoder has become)."""
        logits = self.logits(z_t, z_tp1)
        target = actions.long().flatten()
        loss = F.cross_entropy(logits, target)
        acc = (logits.argmax(dim=1) == target).float().mean()
        return loss, acc
