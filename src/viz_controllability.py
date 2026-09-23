"""Visualize what the inverse-dynamics head thinks the agent controls.

The interpretability payoff of the controllability tool (see controllability.py).
For consecutive frames (s_t, s_{t+1}) we ask the trained inverse model "which action
did I take?", then backprop the predicted action's logit to the input pixels. The
resulting |gradient| is a saliency map: the pixels that most determine the action
are the ones the agent *controls*. If the tool worked, it lights up on the avatar —
learned with zero labels, no game-specific perception.

    python src/viz_controllability.py --game mario --model /app/data/models/mario_idm --out /app/data/diag/mario_control.png

Needs the model .zip (encoder, via PPO.load) and its `<model>_idm.pt` sidecar.
"""
import perf  # first: BLAS thread limits before torch/numpy

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO

from controllability import InverseDynamicsHead
from game_env import make_venv
from games import get_game


def saliency(model, idm, obs_t, obs_tp1):
    """|d logit(predicted action) / d input| over both frames, reduced to one 84x84
    map. Grad flows through the policy's own encoder, so it reflects the learned
    representation, not a fresh one."""
    def prep(obs):
        t = model.policy.obs_to_tensor(obs)[0].float()
        t.requires_grad_(True)
        return t
    x_t, x_tp1 = prep(obs_t), prep(obs_tp1)
    z_t = model.policy.extract_features(x_t)
    z_tp1 = model.policy.extract_features(x_tp1)
    logits = idm.logits(z_t, z_tp1)
    pred = int(logits.argmax(dim=1))
    model.policy.zero_grad(set_to_none=True)
    logits[0, pred].backward()
    # channels-first (1, C, 84, 84) -> max over frames -> (84, 84)
    g = torch.maximum(x_t.grad.abs().amax(dim=1), x_tp1.grad.abs().amax(dim=1))[0]
    return g.cpu().numpy(), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="mario")
    ap.add_argument("--model", required=True, help="model prefix (expects .zip + _idm.pt)")
    ap.add_argument("--out", default="/app/data/diag/controllability.png")
    ap.add_argument("--n-panels", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = PPO.load(f"{args.model}.zip", device="cpu")
    feature_dim = model.policy.features_extractor.features_dim
    idm = InverseDynamicsHead(feature_dim, model.action_space.n)
    idm.load_state_dict(torch.load(f"{args.model}_idm.pt", map_location="cpu"))
    idm.eval()

    venv = make_venv(get_game(args.game), n_envs=1)
    obs = venv.reset()
    panels = []
    steps = 0
    while len(panels) < args.n_panels and steps < 4000:
        steps += 1
        prev = obs
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = venv.step(action)
        if bool(dones[0]):
            continue
        sal, pred = saliency(model, idm, prev, obs)
        # show the newest grayscale frame (channels-first after VecTranspose: last frame)
        frame = model.policy.obs_to_tensor(obs)[0][0, -1].cpu().numpy()
        panels.append((frame, sal, pred))
        for _ in range(15):                     # spread panels across the episode
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = venv.step(action)
            if bool(dones[0]):
                break
    venv.close()

    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i, (frame, sal, pred) in enumerate(panels):
        axes[0, i].imshow(frame, cmap="gray")
        axes[0, i].set_title(f"act {pred}", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(frame, cmap="gray")
        axes[1, i].imshow(sal, cmap="hot", alpha=0.6)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("frame")
    fig.suptitle(f"{args.game}: top=frame, bottom=controllability saliency "
                 f"(what the agent thinks it controls)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"[viz] saved {args.out} ({n} panels)", flush=True)


if __name__ == "__main__":
    main()
