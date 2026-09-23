"""Action->effect probe: discover what each action *does*, by trying it.

Bastien's phase-1 idea, formalized: before optimizing reward, be curious and
characterize every action by the change it causes — "jump on the spot does nothing,
left is blocked, so run right". Pure (no labels, no game knowledge), cheap (a few k
random steps), interpretable. And it is *the* ARC-AGI-3 problem: ACTION1-4 mean
something different per game, invisible until you act.

For each action k, over random play, we measure:
  - effect magnitude: mean fraction of the observation that changes after taking k.
    We report it *relative to the NOOP baseline* (the ambient change you get from
    doing nothing — gravity, scrolling momentum, enemies), so an action whose effect
    ~ NOOP is flagged as a disguised no-op here.
  - a spatial effect map: WHERE it changes (mean abs diff per cell) — localized
    (self-motion), global (world scroll), or nothing.

Caveat (honest): NOOP is an imperfect control — momentum from a previous move leaks
into later steps, so magnitudes are gross, not causal-clean. Structure (null vs
active, local vs global) is what to trust. On a deterministic discrete game (ARC)
it is far cleaner than on Mario.

    python src/action_probe.py --game mario --out data/diag/mario_actions.png
    python src/action_probe.py --game arc_ls20
"""
import perf  # first: BLAS thread limits before numpy

import argparse

import numpy as np

from game_env import make_single_env
from games import get_game
from nets import is_grid_space


def _diff_map(cur, prev, is_grid):
    # collapse channels to a HxW map (grid is channels-first, image channels-last)
    d = np.abs(cur - prev)
    return d.mean(axis=0) if is_grid else d.mean(axis=-1)


def probe_random(spec, n_steps=8000, seed=0):
    """Rough version: average the change each action causes over random play. Cheap,
    but confounded on animated/scrolling games (ambient change swamps the action's
    own effect) — use the counterfactual mode for a clean signal."""
    env = make_single_env(spec)
    n = env.action_space.n
    is_grid = is_grid_space(env.observation_space)
    obs, _ = env.reset(seed=seed)
    prev = np.asarray(obs, dtype=np.float32)
    hw = prev.shape[1:] if is_grid else prev.shape[:2]
    sum_change, sum_map = np.zeros(n), np.zeros((n,) + hw)
    counts = np.zeros(n, dtype=int)
    fresh = True
    for _ in range(n_steps):
        a = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(a)
        cur = np.asarray(obs, dtype=np.float32)
        if not fresh and cur.shape == prev.shape:
            m = _diff_map(cur, prev, is_grid)
            sum_map[a] += m
            sum_change[a] += float(m.mean())
            counts[a] += 1
        prev, fresh = cur, False
        if term or trunc:
            obs, _ = env.reset()
            prev, fresh = np.asarray(obs, dtype=np.float32), True
    env.close()
    safe = np.maximum(counts, 1)
    return {"n_actions": n, "counts": counts, "change": sum_change / safe,
            "effect_map": sum_map / safe[:, None, None]}


def _replay(env, seed, actions):
    """Deterministic replay from reset: returns the obs after `actions` and whether
    the episode ended during replay (so branches from a terminal state are skipped)."""
    obs, _ = env.reset(seed=seed)
    done = False
    for a in actions:
        obs, _, term, trunc, _ = env.step(int(a))
        if term or trunc:
            done = True
            break
    return np.asarray(obs, dtype=np.float32), done


def probe_counterfactual(spec, n_branch=80, prefix_max=150, seed=0, compensate=False):
    """Clean version for a *deterministic* game: from the same state s, take each
    action and measure its own effect |obs_after - s| — a paired, counterfactual
    comparison ("from this exact spot, what does each button do?"). Averaged over
    many states reached by random prefixes. No NOOP assumption, no ambient confound.
    Relies on determinism (same actions from reset -> same state), true for ARC and
    Mario-v0.

    compensate=True additionally decomposes each action's effect via ego-motion: the
    mean global shift vector (locomotion) and the independent-motion residual — the
    fix for screen-locked scrolling games where raw |Δobs| is swamped by the scroll."""
    from egomotion import compensate as ego_compensate
    env = make_single_env(spec)
    n = env.action_space.n
    is_grid = is_grid_space(env.observation_space)
    probe0, _ = env.reset(seed=seed)
    hw = probe0.shape[1:] if is_grid else probe0.shape[:2]
    sum_change, sum_map = np.zeros(n), np.zeros((n,) + hw)
    sum_resid, sum_shift = np.zeros(n), np.zeros((n, 2))
    counts = np.zeros(n, dtype=int)
    rng = np.random.RandomState(seed)
    for _ in range(n_branch):
        prefix = rng.randint(0, n, size=int(rng.randint(0, prefix_max))).tolist()
        s, done = _replay(env, seed, prefix)
        if done:
            continue
        for k in range(n):
            obs_k, _ = _replay(env, seed, prefix + [k])
            if obs_k.shape != s.shape:
                continue
            m = _diff_map(obs_k, s, is_grid)
            sum_map[k] += m
            sum_change[k] += float(m.mean())
            if compensate and not is_grid:
                resid, (dy, dx) = ego_compensate(s, obs_k)
                sum_resid[k] += float(resid.mean())
                sum_shift[k] += (dx, dy)
            counts[k] += 1
    env.close()
    safe = np.maximum(counts, 1)
    out = {"n_actions": n, "counts": counts, "change": sum_change / safe,
           "effect_map": sum_map / safe[:, None, None]}
    if compensate and not is_grid:
        out["residual"] = sum_resid / safe
        out["shift"] = sum_shift / safe[:, None]     # mean (dx, dy) per action
    return out


def classify(change, null_frac=0.1):
    """Null = an action whose effect is under `null_frac` of the most impactful
    action's (no assumption about which index is NOOP). The rest are 'active', scaled
    to the strongest action so magnitudes are comparable across games."""
    top = max(float(np.max(change)), 1e-9)
    labels = []
    for c in change:
        r = c / top
        labels.append("null" if r < null_frac else f"active {r:.2f}")
    return labels


def action_names(spec, n):
    """Human-readable button combo per action if the game exposes one (Mario), else
    the bare index. For *reporting only* — the probe itself uses no such knowledge."""
    from game_env import CANONICAL_ACTIONS
    if spec.name == "mario":
        return ["+".join(sorted(c)) or "NOOP" for c in CANONICAL_ACTIONS[:n]]
    return [str(k) for k in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="mario")
    ap.add_argument("--mode", choices=["counterfactual", "random"], default="counterfactual",
                    help="counterfactual = clean paired effect per action (deterministic "
                         "games); random = rough average over random play")
    ap.add_argument("--n-steps", type=int, default=8000, help="random mode: rollout length")
    ap.add_argument("--n-branch", type=int, default=80, help="counterfactual: branch states")
    ap.add_argument("--prefix-max", type=int, default=150, help="counterfactual: max prefix len")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compensate", action="store_true",
                    help="ego-motion tool: also report per-action global shift "
                         "(locomotion) + independent-motion residual — for scrolling games")
    ap.add_argument("--out", default=None, help="save a per-action effect-map montage")
    args = ap.parse_args()

    spec = get_game(args.game)
    if args.mode == "counterfactual":
        r = probe_counterfactual(spec, args.n_branch, args.prefix_max, args.seed, args.compensate)
    else:
        r = probe_random(spec, args.n_steps, args.seed)
    names = action_names(spec, r["n_actions"])
    labels = classify(r["change"])
    has_ego = "residual" in r

    order = np.argsort(-r["change"])
    budget = (f"{args.n_branch} branches" if args.mode == "counterfactual"
              else f"{args.n_steps} random steps")
    print(f"[action-probe] {args.game} ({args.mode}): {r['n_actions']} actions, {budget}")
    head = f"{'idx':>3} {'action':16} {'change':>8}"
    if has_ego:
        head += f" {'shift_x':>8} {'shift_y':>8} {'residual':>9}"
    print(head + f" {'verdict':>10} {'n':>5}")
    for k in order:
        line = f"{k:3d} {names[k]:16} {r['change'][k]:8.4f}"
        if has_ego:
            line += f" {r['shift'][k][0]:8.2f} {r['shift'][k][1]:8.2f} {r['residual'][k]:9.4f}"
        print(line + f" {labels[k]:>10} {r['counts'][k]:5d}")
    n_null = sum(l == "null" for l in labels)
    print(f"[action-probe] {n_null}/{r['n_actions']} actions look like disguised no-ops "
          f"on {args.game} — discovered, not assumed.")

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = r["n_actions"]
        cols = min(7, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.2 * rows))
        axes = np.atleast_2d(axes)
        for k in range(rows * cols):
            ax = axes[k // cols, k % cols]
            ax.axis("off")
            if k < n:
                ax.imshow(r["effect_map"][k], cmap="hot")
                ax.set_title(f"{k}:{names[k]}\n{labels[k]}", fontsize=7)
        fig.suptitle(f"{args.game}: what each action does (spatial effect map)", fontsize=11)
        fig.tight_layout()
        fig.savefig(args.out, dpi=110)
        print(f"[action-probe] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
