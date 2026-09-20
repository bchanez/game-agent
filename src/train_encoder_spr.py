"""Train the compact encoder with the SPR objective, offline.

Stage B2 (see ROADMAP.md). Stage A trained the encoder by *reconstruction* (MSE),
which drops small moving objects; Stage B showed a *self-predictive* objective
learns a far better representation. B2 brings that objective to the compact-latent
path: train `autoencoder.Encoder` to predict the next latent from (latent, action)
instead of redrawing pixels, then freeze it and run the fast MlpPolicy on the
latent (train_ppo_latent.py, unchanged).

    python src/collect_frames.py --game breakout          # transitions with actions
    python src/train_encoder_spr.py --game breakout
    python src/train_ppo_latent.py --game breakout --encoder data/models/breakout_encoder_spr.pt

Collapse: offline, with no RL loss to anchor the encoder, the EMA target alone
does not prevent it — the encoder maps everything to a near-constant and the SPR
loss trivially hits zero. We add a VICReg variance term that keeps each latent
dimension above a minimum std, which makes collapse impossible. This is a generic
"do not collapse" signal, not game knowledge. The final latent-std print is the
sanity check — a near-zero std means it collapsed anyway.
"""
import argparse
import copy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import Encoder
from spr import SPRHead, ema_update

FRAMES_DIR = "/app/data/frames"
MODELS_DIR = "/app/data/models"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="breakout")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--tau", type=float, default=0.01, help="EMA rate for the target encoder")
    ap.add_argument("--var-coef", type=float, default=1.0,
                    help="VICReg variance-regularization weight (anti-collapse)")
    ap.add_argument("--frames-path", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-frames", type=int, default=None, help="cap dataset (smoke tests)")
    args = ap.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    device = "cpu"
    frames_path = args.frames_path or os.path.join(FRAMES_DIR, f"{args.game}.npz")
    out = args.out or os.path.join(MODELS_DIR, f"{args.game}_encoder_spr.pt")

    data = np.load(frames_path)
    if "actions" not in data:
        raise SystemExit(f"{frames_path} has no 'actions' — re-run collect_frames.py")
    s_t = torch.from_numpy(data["prev"]).float().div_(255.0).unsqueeze(1)   # (N,1,84,84)
    s_tp1 = torch.from_numpy(data["frames"]).float().div_(255.0).unsqueeze(1)
    a_t = torch.from_numpy(data["actions"]).long()
    if args.max_frames:
        s_t, s_tp1, a_t = s_t[:args.max_frames], s_tp1[:args.max_frames], a_t[:args.max_frames]
    n_actions = int(a_t.max().item()) + 1
    print(f"Loaded {len(s_t)} transitions from {frames_path} (n_actions={n_actions})", flush=True)

    encoder = Encoder(args.latent_dim).to(device)
    target_encoder = copy.deepcopy(encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad_(False)
    head = SPRHead(args.latent_dim, n_actions).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=args.lr)

    loader = DataLoader(TensorDataset(s_t, a_t, s_tp1), batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.epochs):
        total = 0.0
        for bs_t, ba, bs_tp1 in loader:
            bs_t, ba, bs_tp1 = bs_t.to(device), ba.to(device), bs_tp1.to(device)
            z_t = encoder(bs_t)
            with torch.no_grad():
                z_tp1 = target_encoder(bs_tp1)
            # VICReg variance hinge: push each latent dim's std toward >=1 so the
            # encoder cannot collapse the batch to a constant
            std = torch.sqrt(z_t.var(dim=0) + 1e-4)
            var_loss = torch.relu(1.0 - std).mean()
            loss = head(z_t, ba, z_tp1) + args.var_coef * var_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema_update(target_encoder, encoder, args.tau)
            head.update_target(args.tau)
            total += loss.item() * len(bs_t)
        print(f"epoch {epoch + 1}/{args.epochs}  spr_loss={total / len(s_t):.6f}", flush=True)

    torch.save({"encoder": encoder.state_dict(), "latent_dim": args.latent_dim}, out)
    print(f"Saved encoder to {out}", flush=True)

    with torch.no_grad():
        z = encoder(s_t[:2000].to(device))
    print(f"latent std (mean over dims): {z.std(dim=0).mean().item():.4f} "
          f"(near 0 = collapsed)", flush=True)


if __name__ == "__main__":
    main()
