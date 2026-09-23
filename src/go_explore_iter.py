"""Iterated Go-Explore: explore <-> robustify, the full algorithm (Ecoffet et al. 2021).

Policy-guided explore alone (`go_explore.py`) re-solves the *known* frontier fast but
can't cross into a level the guide never saw — it is out-of-distribution there (see
FINDINGS). The fix is to **iterate**:

  1. EXPLORE  — policy-guided Go-Explore, archive cells + best trajectories.
  2. ROBUSTIFY — behavioral-clone the policy on the frontier trajectories found, so the
     guide now *knows* that far (no longer OOD at the old frontier).
  3. repeat — the improved guide pushes exploration one level deeper each round.

Cheap on a deterministic game: "return" is replaying an action sequence, and robustify
is supervised (no extra env interaction beyond replay). Generic — no per-game knowledge.

    python src/go_explore_iter.py --game arc_ls20 --rounds 6 --explore-steps 80000 \
        --init-from data/models/arc_ls20_final.zip
"""
import perf  # first: BLAS thread limits before torch/numpy

import argparse
import random

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor

from game_env import make_venv
from games import get_game
from go_explore import cell_key
from nets import policy_kwargs_for


def explore_phase(env, model, archive, best, steps, explore_len, epsilon, factor, seed_off):
    """Policy-guided Go-Explore for `steps` env-steps, updating archive + best in place."""
    n_actions = env.action_space.n

    def key_of(obs, info):
        return cell_key(np.asarray(obs)[0], info.get("levels_completed", 0), factor)

    def act(obs):
        if random.random() < epsilon:
            return random.randrange(n_actions)
        a, _ = model.predict(np.asarray(obs)[None], deterministic=False)
        return int(a[0])

    used = 0
    while used < steps:
        cells = list(archive.items())
        top = max(e["score"] for _, e in cells)
        pool = [c for c in cells if c[1]["score"] == top] if random.random() < 0.8 else cells
        w = np.array([1.0 / (1 + e["visits"]) for _, e in pool])
        key, entry = pool[np.random.choice(len(pool), p=w / w.sum())]
        entry["visits"] += 1

        obs, info = env.reset()
        for a in entry["traj"]:
            obs, r, term, trunc, info = env.step(a)
        used += len(entry["traj"])
        traj = list(entry["traj"])

        for _ in range(explore_len):
            a = act(obs)
            obs, r, term, trunc, info = env.step(a)
            traj.append(a)
            used += 1
            levels = info.get("levels_completed", 0)
            k = key_of(obs, info)
            if k not in archive or len(traj) < len(archive[k]["traj"]):
                archive[k] = {"traj": list(traj), "score": levels, "visits": 0}
            if levels > best["levels"]:
                best.update(levels=levels, traj=list(traj))
                print(f"    new best: level {levels} in {len(traj)} actions", flush=True)
            if term or trunc:
                break
    return top


def collect_frontier(env, archive, max_cells, max_pairs):
    """Replay the deepest cells' trajectories to gather (obs, action) demonstrations for
    behavioral cloning — the frontier is what we want the policy to reproduce reliably."""
    cells = sorted(archive.values(), key=lambda e: (-e["score"], len(e["traj"])))
    obs_buf, act_buf = [], []
    for entry in cells[:max_cells]:
        if not entry["traj"]:
            continue
        obs, _ = env.reset()
        for a in entry["traj"]:
            obs_buf.append(np.asarray(obs))
            act_buf.append(a)
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        if len(obs_buf) >= max_pairs:
            break
    return np.asarray(obs_buf), np.asarray(act_buf)


def robustify(model, obs_np, act_np, epochs, batch):
    """Behavioral cloning: make the policy reproduce the demonstrated actions (the only
    new learning per round — supervised, cheap)."""
    if len(obs_np) < batch:
        return 0.0
    n = len(obs_np)
    model.policy.set_training_mode(True)
    last = 0.0
    for _ in range(epochs):
        idx = np.random.permutation(n)
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            obs_t = obs_as_tensor(obs_np[b], model.device)
            act_t = torch.as_tensor(act_np[b], device=model.device).long()
            _, log_prob, _ = model.policy.evaluate_actions(obs_t, act_t)
            loss = -log_prob.mean()                         # imitate the frontier actions
            model.policy.optimizer.zero_grad()
            loss.backward()
            model.policy.optimizer.step()
            last = float(loss.item())
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="arc_ls20")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--explore-steps", type=int, default=80_000, help="env steps per round")
    ap.add_argument("--explore-len", type=int, default=30)
    ap.add_argument("--epsilon", type=float, default=0.3)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--bc-epochs", type=int, default=3)
    ap.add_argument("--bc-batch", type=int, default=64)
    ap.add_argument("--bc-max-cells", type=int, default=300)
    ap.add_argument("--bc-max-pairs", type=int, default=20_000)
    ap.add_argument("--init-from", default=None, help="warm-start the guide policy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    spec = get_game(args.game)
    venv = make_venv(spec, 1)                                # to construct a policy with the right spaces
    model = PPO("CnnPolicy", venv, policy_kwargs=policy_kwargs_for(venv.observation_space),
                device="cpu", n_steps=64, batch_size=64)
    if args.init_from:
        src = PPO.load(args.init_from, device="cpu")
        model.policy.load_state_dict(src.policy.state_dict())
        print(f"Warm-started guide from {args.init_from}", flush=True)

    env = spec.make_raw_env()
    obs, info = env.reset()
    archive = {cell_key(np.asarray(obs)[0], 0, args.factor): {"traj": [], "score": 0, "visits": 0}}
    best = {"levels": 0, "traj": []}

    for r in range(args.rounds):
        top = explore_phase(env, model, archive, best, args.explore_steps,
                            args.explore_len, args.epsilon, args.factor, r)
        obs_np, act_np = collect_frontier(env, archive, args.bc_max_cells, args.bc_max_pairs)
        bc_loss = robustify(model, obs_np, act_np, args.bc_epochs, args.bc_batch)
        print(f"[round {r+1}/{args.rounds}] cells={len(archive)} frontier_level={top} "
              f"deepest={best['levels']} bc_pairs={len(obs_np)} bc_loss={bc_loss:.3f}", flush=True)

    print(f"\nDone: deepest level = {best['levels']} ({len(best['traj'])} actions), "
          f"{len(archive)} cells", flush=True)
    venv.close()


if __name__ == "__main__":
    main()
