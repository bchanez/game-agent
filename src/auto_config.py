"""Self-configuration — the detector layer of the toolbox (see docs/auto-config.md).

Layer 1 is the tools (curiosity, auto-gamma, SIL, frame-stack, ...). This is layer
2: cheap probes that *measure an unknown game* and switch each tool on or off, so
nothing is tuned by hand per game (the Kaggle constraint). `probe` runs a short
random rollout and reads three signals; `configure` maps them to tool flags.

    signals = probe(spec)         # {motion, reward_density, mean_ep_len}
    cfg = configure(signals)      # {intrinsic_coef, sil_coef, frame_stack, auto_gamma}

The rules reproduce, from measurement alone, the choices we reached by hand on the
ARC ls20 ablation (sparse → curiosity+SIL; no motion → no frame-stack) and on Mario
(dense reward → no curiosity; scrolling → frame-stack).
"""
import numpy as np

from game_env import make_single_env
from nets import is_grid_space


def probe(spec, n_steps=2000, seed=0):
    """Run a random policy briefly and measure what kind of game this is."""
    env = make_single_env(spec)
    obs, _ = env.reset(seed=seed)
    prev, prev_a = np.asarray(obs), None
    changes, rewards, ep_lens, cur_len = [], [], [], 0
    # non-Markov test: if the SAME (obs, action) ever leads to a DIFFERENT next obs,
    # a single frame doesn't capture the state (hidden velocity) -> frame-stack helps
    trans, repeats, collisions, fresh = {}, 0, 0, True
    for _ in range(n_steps):
        a = env.action_space.sample()
        obs, r, term, trunc, _ = env.step(a)
        cur = np.asarray(obs)
        if cur.shape == prev.shape:
            changes.append(float(np.mean(cur != prev)))
            if not fresh:                                 # skip the post-reset step
                key = (prev.tobytes(), int(prev_a))
                val = cur.tobytes()
                if key in trans:
                    repeats += 1
                    collisions += (trans[key] != val)
                else:
                    trans[key] = val
        prev, prev_a, fresh = cur, a, False
        rewards.append(float(r))
        cur_len += 1
        if term or trunc:
            ep_lens.append(cur_len)
            cur_len = 0
            obs, _ = env.reset()
            prev, fresh = np.asarray(obs), True
    is_grid = is_grid_space(env.observation_space)
    env.close()
    return {
        "motion": float(np.mean(changes)) if changes else 0.0,
        "reward_density": float(np.mean(np.asarray(rewards) != 0)),
        "mean_ep_len": float(np.mean(ep_lens)) if ep_lens else float(n_steps),
        "non_markov": collisions / repeats if repeats else 0.0,
        "is_grid": is_grid,
    }


SPARSE_BELOW = 0.02

# --- the tool library --------------------------------------------------------
# Each tool is a rule: signals -> the config overrides it wants. A tool enters the
# library once it has helped on >=1 game (kept forever); its rule decides, per game,
# whether to switch it on and how to set its params. Configuring a game is then one
# probe + apply every rule — seconds, not an ablation. Grow the box by appending a
# tool here. Keep-in-library (additive) and activate-per-game (this) are separate.


def _curiosity(signals):
    # RND exploration: needed only when the reward is sparse/absent
    return {"intrinsic_coef": 1.0 if signals["reward_density"] < SPARSE_BELOW else 0.0}


def _self_imitation(signals):
    # replay winning episodes: pays off when wins are sparse (else on-policy is fine)
    return {"sil_coef": 1.0 if signals["reward_density"] < SPARSE_BELOW else 0.0}


def _auto_gamma(signals):
    # a *param-tuning* tool, not on/off: it self-tunes gamma from the observed
    # horizon every rollout, so it's always safe to leave on
    return {"auto_gamma": True}


def _spr(signals):
    # self-predictive representation: a general sample-efficiency booster (no cheap
    # per-game signal tells you when dynamics-learning helps), so on by default —
    # like auto-gamma. The recurrent path forces it off (not wired there yet).
    return {"spr_coef": 1.0}


def _frame_stack(signals):
    # frame-stack reveals a hidden *velocity* (a ball's direction between pixel
    # frames). It cannot reveal a hidden *counter* (a grid's life/energy). So on only
    # for a non-Markov PIXEL game; grids stay at 1 (memory is their fix, below).
    on = signals.get("non_markov", 0.0) > 0.01 and not signals.get("is_grid", False)
    return {"frame_stack": 4 if on else 1}


def _memory(signals):
    # a non-Markov grid hides state a frame-stack can't show (ls20's life) that memory
    # could track — BUT memory-without-SIL underperforms SIL-without-memory (g50t: 0.05
    # vs 0.30, see FINDINGS), and SIL isn't wired into the recurrent path yet. So DON'T
    # auto-enable it (that would drop SIL, the better tool): keep SIL, just log the
    # recommendation. Flip recurrent to `recommend` here once SIL+recurrent beats SIL.
    recommend = signals.get("non_markov", 0.0) > 0.01 and signals.get("is_grid", False)
    return {"recurrent": False, "memory_recommended": recommend}


TOOLS = [
    ("curiosity", _curiosity),
    ("self-imitation", _self_imitation),
    ("auto-gamma", _auto_gamma),
    ("spr", _spr),
    ("frame-stack", _frame_stack),
    ("memory", _memory),
]
DEFAULTS = {"intrinsic_coef": 0.0, "sil_coef": 0.0, "auto_gamma": False,
            "spr_coef": 0.0, "frame_stack": 1, "recurrent": False,
            "memory_recommended": False}


def configure(signals):
    """Apply every tool's rule to the probe signals -> the full per-game config."""
    cfg = dict(DEFAULTS)
    for _, rule in TOOLS:
        cfg.update(rule(signals))
    if cfg["recurrent"]:
        # SPR and self-imitation aren't wired into the recurrent buffer yet, so a
        # recurrent game can't use them — reflect that instead of pretending
        cfg["spr_coef"] = cfg["sil_coef"] = 0.0
    return cfg


def auto_configure(spec, n_steps=2000, seed=0):
    """probe + configure, with a printed explanation of the decision."""
    signals = probe(spec, n_steps, seed)
    cfg = configure(signals)
    print(f"[auto-config] {spec.name}: reward_density={signals['reward_density']:.4f} "
          f"non_markov={signals['non_markov']:.4f} "
          f"mean_ep_len={signals['mean_ep_len']:.0f}", flush=True)
    print(f"[auto-config] -> curiosity={cfg['intrinsic_coef']>0} sil={cfg['sil_coef']>0} "
          f"spr={cfg['spr_coef']>0} frame_stack={cfg['frame_stack']} "
          f"auto_gamma={cfg['auto_gamma']} recurrent={cfg['recurrent']} "
          f"(memory_rec={cfg['memory_recommended']})", flush=True)
    return cfg, signals
