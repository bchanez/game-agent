"""Show what object-centric perception discovers on a game's grids (Phase 1 proof).

Grabs a few distinct grids from random play, extracts objects (no labels), and draws
each with its objects' bounding boxes + centroids. Logs per-grid object counts so the
"understanding" layer is measurable, not just pretty.

    python src/viz_objects.py --game arc_ls20 --out data/diag/ls20_objects.png
"""
import perf  # first: BLAS thread limits before numpy

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from game_env import make_single_env
from games import get_game
from objects import _PALETTE, extract_objects, object_features


def collect_grids(spec, n, seed=0):
    env = make_single_env(spec)
    obs, _ = env.reset(seed=seed)
    grids, seen = [], set()
    steps = 0
    while len(grids) < n and steps < 3000:
        steps += 1
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        g = np.asarray(obs)[0]                         # (1,H,W) -> (H,W)
        key = g.tobytes()
        if key not in seen:
            seen.add(key)
            grids.append(g)
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    return grids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="arc_ls20")
    ap.add_argument("--out", default="data/diag/objects.png")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--max-objects", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    grids = collect_grids(get_game(args.game), args.n, args.seed)
    counts = []
    fig, axes = plt.subplots(1, len(grids), figsize=(3.0 * len(grids), 3.4))
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, grids):
        objs, bg = extract_objects(g)
        _, _, trunc = object_features(g, args.max_objects)
        counts.append(len(objs))
        ax.imshow(_PALETTE[g])
        for o in objs[:args.max_objects]:
            ax.add_patch(mpatches.Rectangle(
                (o["x0"] - 0.5, o["y0"] - 0.5), o["w"], o["h"],
                fill=False, edgecolor="white", linewidth=0.8))
            ax.plot(o["cx"], o["cy"], "w+", markersize=4)
        ax.set_title(f"{len(objs)} objs (bg {bg})"
                     + (f" +{trunc} cut" if trunc else ""), fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{args.game}: object-centric perception (no labels)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"[viz-objects] {args.game}: objects/grid = "
          f"{np.mean(counts):.1f} mean, {min(counts)}-{max(counts)} range "
          f"over {len(grids)} grids; saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
