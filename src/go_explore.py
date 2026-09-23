"""Go-Explore (Ecoffet et al. 2021) exploration on a deterministic ARC game.

The failure on ls20 (see FINDINGS.md) was *discovery*: PPO+self-imitation reliably
reproduces the wins it stumbles on, but only ever stumbled to level 1. Go-Explore
attacks discovery head-on: keep an archive of reached *cells*, each with the best
trajectory that reaches it; repeatedly **return** to a promising cell and explore
from there, so search resumes from the frontier instead of restarting from reset.

ls20 is deterministic (verified: same actions → same grids), so "return" is just
replaying a stored action sequence from reset — no state-save API needed. This is
the tabular/exploration phase; the best trajectory it finds is dumped so a policy
can be robustified from it later (via self-imitation / behavioral cloning).

Random bursts reach level 1 but never level 2 (FINDINGS): they squander level 2's
depleting "life" budget. **Policy-guided** exploration (`--policy`) fixes the *what
you do at the frontier*: sample the explore actions from a trained SIL policy (which
already solves level 1) instead of uniform random, keeping an ε fraction random for
raw novelty. The policy is a directed prior ("move toward things") that random lacks.

    python src/go_explore.py --game arc_ls20 --steps 500000 --explore-len 30
    python src/go_explore.py --game arc_ls20 --policy data/models/arc_ls20_final.zip

Cell = (levels_completed, grid subsampled by --factor). Generic: a spatial
downsample plus the game's own level counter — no per-game perception.
"""
import perf  # first: BLAS thread limits before numpy

import argparse
import pickle
import random

import numpy as np

from games import get_game


def cell_key(grid, levels, factor):
    return (int(levels), grid[::factor, ::factor].tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="arc_ls20")
    ap.add_argument("--steps", type=int, default=500_000, help="env-step budget")
    ap.add_argument("--explore-len", type=int, default=30,
                    help="steps taken after returning to a cell")
    ap.add_argument("--factor", type=int, default=2, help="grid subsample for cells")
    ap.add_argument("--policy", default=None,
                    help="trained model.zip: sample explore actions from it (policy-guided) "
                         "instead of uniform random")
    ap.add_argument("--epsilon", type=float, default=0.3,
                    help="fraction of explore steps kept uniform-random even with --policy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/app/data/models/arc_ls20_goexplore.pkl",
                    help="dump the best trajectory found (for robustification)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # the raw ARC env (its own grid obs / GameAction step), bypassing the RL pipeline
    spec = get_game(args.game)
    env = spec.make_raw_env()
    n_actions = env.action_space.n

    # policy-guided: the raw grid obs (1,64,64) is exactly what the policy trained on
    # (ARC grids skip the pixel pipeline), so it consumes obs[None] directly.
    model = None
    if args.policy:
        from stable_baselines3 import PPO
        model = PPO.load(args.policy, device="cpu")
        print(f"Policy-guided from {args.policy} (epsilon={args.epsilon})", flush=True)

    def explore_action(obs):
        if model is None or random.random() < args.epsilon:
            return random.randrange(n_actions)
        a, _ = model.predict(np.asarray(obs)[None], deterministic=False)
        return int(a[0])

    def key_of(obs, info):
        grid = np.asarray(obs)[0]                 # (64,64) color indices
        return cell_key(grid, info.get("levels_completed", 0), args.factor)

    def return_to(traj):
        obs, info = env.reset()
        for a in traj:
            obs, r, term, trunc, info = env.step(a)
        return obs, info

    obs, info = env.reset()
    archive = {key_of(obs, info): {"traj": [], "score": 0, "visits": 0}}
    best = {"score": 0, "traj": [], "levels": 0}
    steps = 0
    iters = 0

    while steps < args.steps:
        iters += 1
        # select a cell to explore from. Bias hard to the *frontier* (the deepest
        # level reached so far): with p=0.8 pick among max-score cells, else anywhere
        # — else the thousands of level-0 cells drown out the few level-1 ones and
        # the frontier never gets explored. Within the pool, favor least-visited.
        cells = list(archive.items())
        top = max(e["score"] for _, e in cells)
        pool = [c for c in cells if c[1]["score"] == top] if random.random() < 0.8 else cells
        w = np.array([1.0 / (1 + e["visits"]) for _, e in pool])
        key, entry = pool[np.random.choice(len(pool), p=w / w.sum())]
        entry["visits"] += 1

        obs, info = return_to(entry["traj"])
        steps += len(entry["traj"])
        traj = list(entry["traj"])

        for _ in range(args.explore_len):
            a = explore_action(obs)
            obs, r, term, trunc, info = env.step(a)
            traj.append(a)
            steps += 1
            levels = info.get("levels_completed", 0)
            k = key_of(obs, info)
            if k not in archive or len(traj) < len(archive[k]["traj"]):
                archive[k] = {"traj": list(traj), "score": levels, "visits": 0}
            if levels > best["levels"]:
                best = {"score": levels, "traj": list(traj), "levels": levels}
                print(f"[{steps} steps] new best: level {levels} "
                      f"in {len(traj)} actions", flush=True)
            if term or trunc:
                break

    print(f"\nDone: {steps} steps, {iters} iterations, {len(archive)} cells archived, "
          f"deepest level reached = {best['levels']}", flush=True)
    with open(args.out, "wb") as f:
        pickle.dump(best, f)
    print(f"Saved best trajectory ({len(best['traj'])} actions) to {args.out}", flush=True)


if __name__ == "__main__":
    main()
