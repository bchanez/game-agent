"""Meta-RL (RL²) training: a *recurrent* policy that adapts within an episode.

Exp B showed transferring frozen weights fails when action semantics differ
across games (the index is shared, the meaning isn't). Meta-RL attacks the same
gap differently: instead of baking a behavioural prior into weights, train an
LSTM policy that — fed its own previous action and reward — *infers which game
it's in and adapts in-context*, with no weight update. Transfer becomes few-shot,
not frozen.

    python src/train_ppo_meta.py --games mario,breakout --timesteps 500000 --n-envs 8

Smoke-test to a scratch path (never clobber a real model):
    python src/train_ppo_meta.py --games mario,breakout --timesteps 8000 --out /app/data/models/_smoke

Same obs pipeline as the feedforward path, but VecPrevActionReward turns the
observation into Dict{image, prev_action, prev_reward} — the RL² conditioning —
and RecurrentPPO's MultiInputLstmPolicy runs its LSTM over the mix.
"""
import perf  # first: sets BLAS thread limits before torch/numpy import

import argparse
import os

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback

from game_env import VecPrevActionReward, make_multi_venv, make_venv
from games import get_game, get_games

MODELS_DIR = "/app/data/models"
TB_DIR = "/app/data/tb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="breakout")
    ap.add_argument("--games", default=None,
                    help="comma-separated list to train ONE recurrent policy on a "
                         "mix of games (overrides --game); e.g. 'mario,breakout'")
    ap.add_argument("--init-from", default=None,
                    help="load recurrent policy weights from this model.zip before "
                         "training (recurrent->recurrent transfer)")
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--lstm-hidden", type=int, default=256,
                    help="LSTM hidden size (the in-episode memory capacity)")
    ap.add_argument("--out", default=None,
                    help="output prefix (else data/models/<label>_ppo_meta_final); "
                         "use a scratch path for smoke tests")
    ap.add_argument("--torch-threads", type=int, default=None,
                    help="learner torch threads (default: all cores)")
    ap.add_argument("--n-epochs", type=int, default=4,
                    help="PPO passes per rollout (lower = faster updates)")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed (env, torch, action sampling) for reproducible runs")
    args = ap.parse_args()

    perf.setup_cpu_threads(args.torch_threads)

    os.makedirs(MODELS_DIR, exist_ok=True)
    if args.games:
        names = [n.strip() for n in args.games.split(",") if n.strip()]
        venv = make_multi_venv(get_games(names), args.n_envs)
        game_label = "+".join(names)
    else:
        venv = make_venv(get_game(args.game), args.n_envs)
        game_label = args.game
    venv = VecPrevActionReward(venv)

    model = RecurrentPPO(
        "MultiInputLstmPolicy", venv, verbose=1,
        n_steps=512, batch_size=64, n_epochs=args.n_epochs,
        learning_rate=1e-4, gamma=0.9, gae_lambda=1.0, ent_coef=0.01,
        tensorboard_log=TB_DIR, device="cpu", seed=args.seed,
        policy_kwargs=dict(lstm_hidden_size=args.lstm_hidden),
    )

    if args.init_from:
        src = RecurrentPPO.load(args.init_from, device="cpu")
        model.policy.load_state_dict(src.policy.state_dict())
        print(f"Initialized policy from {args.init_from}", flush=True)

    prefix = f"{game_label}_ppo_meta"
    ckpt = CheckpointCallback(
        save_freq=max(20_000 // args.n_envs, 1),
        save_path=MODELS_DIR, name_prefix=prefix,
    )

    print(f"Training RecurrentPPO (RL²) on {game_label} for {args.timesteps} "
          f"steps, {args.n_envs} env(s)...", flush=True)
    model.learn(total_timesteps=args.timesteps, callback=ckpt)

    final = args.out or os.path.join(MODELS_DIR, f"{prefix}_final")
    model.save(final)
    print(f"Saved {final}.zip", flush=True)


if __name__ == "__main__":
    main()
