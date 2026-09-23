"""Human/agent-in-the-loop ARC driver — replay an action sequence and render the state.

The zero-cost reasoning path: instead of paying an LLM per step, a reasoner (me, via
the terminal) drives the game directly. ls20 is deterministic, so state = replay of an
action prefix from reset (like Go-Explore). Pass the actions so far, look at the rendered
grid + per-step feedback, decide the next action, and re-run with it appended.

    python src/arc_play.py --game arc_ls20 --actions "" --out data/diag/play.png
    python src/arc_play.py --game arc_ls20 --actions "ACTION3,ACTION3,ACTION4" --out data/diag/play.png
"""
import perf  # first: BLAS thread limits before numpy

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from games import get_game
from objects import _PALETTE

ACTION_NAMES = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="arc_ls20")
    ap.add_argument("--actions", default="", help="comma-separated action names or indices")
    ap.add_argument("--out", default="data/diag/play.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seq = [a.strip() for a in args.actions.split(",") if a.strip()]
    idxs = [int(a) if a.isdigit() else ACTION_NAMES.index(a) for a in seq]

    env = get_game(args.game).make_raw_env()
    obs, info = env.reset(seed=args.seed)
    grid = np.asarray(obs)[0]
    print(f"reset: levels={info.get('levels_completed', 0)}", flush=True)
    for i, a in enumerate(idxs):
        obs, reward, term, trunc, info = env.step(a)
        prev, grid = grid, np.asarray(obs)[0]
        changed = int(np.count_nonzero(prev != grid))
        print(f"[{i:2d}] {ACTION_NAMES[a]:8} reward={reward:+.0f} "
              f"levels={info.get('levels_completed',0)} changed={changed} "
              f"won={info.get('won')} term={term} trunc={trunc}", flush=True)
        if term or trunc:
            print("  (episode ended)", flush=True)
            break

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(_PALETTE[grid])
    ax.set_title(f"{args.game} after {len(idxs)} actions "
                 f"(levels={info.get('levels_completed',0)})", fontsize=10)
    ax.set_xticks(range(0, 64, 8)); ax.set_yticks(range(0, 64, 8))
    ax.grid(color="white", alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
