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


def probe(spec, n_steps=2000, seed=0):
    """Run a random policy briefly and measure what kind of game this is."""
    env = make_single_env(spec)
    obs, _ = env.reset(seed=seed)
    prev = np.asarray(obs)
    changes, rewards, ep_lens, cur_len = [], [], [], 0
    for _ in range(n_steps):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        cur = np.asarray(obs)
        if cur.shape == prev.shape:
            changes.append(float(np.mean(cur != prev)))   # fraction of cells changed
        prev = cur
        rewards.append(float(r))
        cur_len += 1
        if term or trunc:
            ep_lens.append(cur_len)
            cur_len = 0
            obs, _ = env.reset()
            prev = np.asarray(obs)
    env.close()
    return {
        "motion": float(np.mean(changes)) if changes else 0.0,
        "reward_density": float(np.mean(np.asarray(rewards) != 0)),
        "mean_ep_len": float(np.mean(ep_lens)) if ep_lens else float(n_steps),
    }


def configure(signals, sparse_below=0.02):
    """Pick the tool set from the probe signals — one rule for every game.

    The *sparsity* detector is reliable (ARC games all read ~0 reward density).
    The *motion* signal (fraction of cells changed) is NOT a good frame-stack
    detector — it can't tell agent-caused change (Markovian) from ambient motion
    (non-Markovian, the case frame-stack is for): ls20 (no motion) measured *higher*
    than Breakout (ball motion). So frame-stack stays off for raw grids (they're
    usually fully observed) and pixel games keep their built-in 4-stack. A proper
    non-Markovianity detector is future work (layer-2 TODO)."""
    sparse = signals["reward_density"] < sparse_below
    return {
        "intrinsic_coef": 1.0 if sparse else 0.0,   # curiosity only when reward is sparse
        "sil_coef": 1.0 if sparse else 0.0,          # reuse rare wins only when sparse
        "frame_stack": 1,                            # motion detector unreliable -> off
        "auto_gamma": True,                          # always: it self-tunes from horizon
    }


def auto_configure(spec, n_steps=2000, seed=0):
    """probe + configure, with a printed explanation of the decision."""
    signals = probe(spec, n_steps, seed)
    cfg = configure(signals)
    print(f"[auto-config] {spec.name}: motion={signals['motion']:.4f} "
          f"reward_density={signals['reward_density']:.4f} "
          f"mean_ep_len={signals['mean_ep_len']:.0f}", flush=True)
    print(f"[auto-config] -> curiosity={cfg['intrinsic_coef']>0} sil={cfg['sil_coef']>0} "
          f"frame_stack={cfg['frame_stack']} auto_gamma={cfg['auto_gamma']}", flush=True)
    return cfg, signals
