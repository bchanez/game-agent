"""Train the offline autoencoder, then keep its encoder for the policy.

Stage A step 2 (see ROADMAP.md). Purely self-supervised: reconstruct each frame
from a D-dim latent. No RL loop, batched over a fixed dataset, so it's cheap on
CPU. The decoder is discarded afterwards; train_ppo_latent.py loads the encoder.

Run (inside the container):
    python src/train_encoder.py --game breakout --epochs 10

Saves data/models/<game>_encoder.pt and a reconstruction preview grid
data/recon_<game>.png. The preview is the gate: if a small-but-critical object
(the Breakout ball!) vanishes in the reconstruction, the latent dropped it and
the policy will be blind — see ROADMAP Phase 8.5.

A plain MSE loss averages over all pixels, so a 2-pixel ball is negligible and
gets dropped while the static background is rendered perfectly. The fix is a
*motion-weighted* loss: pixels that changed since the previous frame count more,
so moving objects (the ball!) become expensive to ignore. This is a generic
signal ("what moved"), not game knowledge ("where the ball is") — the *no human
indication* principle holds.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import Autoencoder

FRAMES_DIR = "/app/data/frames"
MODELS_DIR = "/app/data/models"
DATA_DIR = "/app/data"


def load_frames(path, motion_alpha):
    data = np.load(path)
    x = torch.from_numpy(data["frames"]).float().div_(255.0).unsqueeze(1)  # (N,1,84,84)
    if motion_alpha == 0 or "prev" not in data:
        return x, torch.ones_like(x)
    prev = torch.from_numpy(data["prev"]).float().div_(255.0).unsqueeze(1)
    weight = 1.0 + motion_alpha * (x - prev).abs()   # moving pixels weigh more
    return x, weight


def save_reconstructions(model, x, weight, out, n=8, device="cpu"):
    model.eval()
    with torch.no_grad():
        sample = x[:n].to(device)
        recon, _ = model(sample)
    sample = sample.cpu().numpy()[:, 0]
    recon = recon.cpu().numpy()[:, 0]
    motion = (weight[:n, 0].cpu().numpy() - 1.0)     # back to |frame diff| for display

    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))
    for i in range(n):
        axes[0, i].imshow(sample[i], cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(recon[i], cmap="gray", vmin=0, vmax=1)
        axes[2, i].imshow(motion[i], cmap="magma")
        for r in range(3):
            axes[r, i].axis("off")
    axes[0, 0].set_title("original", loc="left")
    axes[1, 0].set_title("reconstruction", loc="left")
    axes[2, 0].set_title("motion weight", loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="breakout")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--motion-alpha", type=float, default=10.0,
                    help="how much moving pixels outweigh static ones (0 = plain MSE)")
    ap.add_argument("--frames-path", default=None)
    ap.add_argument("--out", default=None, help="encoder output path (else data/models/<game>_encoder.pt)")
    ap.add_argument("--max-frames", type=int, default=None, help="cap dataset size (for smoke tests)")
    args = ap.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    device = "cpu"
    frames_path = args.frames_path or os.path.join(FRAMES_DIR, f"{args.game}.npz")
    out = args.out or os.path.join(MODELS_DIR, f"{args.game}_encoder.pt")

    x, weight = load_frames(frames_path, args.motion_alpha)
    if args.max_frames:
        x, weight = x[:args.max_frames], weight[:args.max_frames]
    print(f"Loaded {len(x)} frames from {frames_path} "
          f"(motion_alpha={args.motion_alpha})", flush=True)

    loader = DataLoader(TensorDataset(x, weight), batch_size=args.batch_size, shuffle=True)
    model = Autoencoder(args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch, w in loader:
            batch, w = batch.to(device), w.to(device)
            recon, _ = model(batch)
            loss = (w * (recon - batch) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        print(f"epoch {epoch + 1}/{args.epochs}  weighted_mse={total / len(x):.6f}", flush=True)

    torch.save({"encoder": model.encoder.state_dict(),
                "latent_dim": args.latent_dim}, out)
    print(f"Saved encoder to {out}", flush=True)

    recon_path = os.path.join(DATA_DIR, f"recon_{args.game}.png")
    save_reconstructions(model, x, weight, recon_path, device=device)
    print(f"Saved reconstruction preview to {recon_path}", flush=True)


if __name__ == "__main__":
    main()
