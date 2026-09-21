"""Evaluate a meta-RL (RecurrentPPO) policy on a held-out game, frozen weights.

The scientific question mirrors Exp B (see FINDINGS.md): pretrain on games A,B,
then face an unseen game C. Frozen-weight transfer *regressed* there — the shared
action index carried the wrong behavioural prior. Meta-RL's bet is that a
recurrent policy, fed its own previous action and reward, *adapts in-context* to
C within the episode, no weight update. So the natural replication is:

    train RecurrentPPO on {breakout, montezuma}, evaluate frozen on mario.

    python src/eval_meta.py --model data/models/breakout+montezuma_ppo_meta_final.zip \
        --game mario --episodes 20

Two adaptation views:
  * default (--trial-episodes 1): each episode is independent, LSTM state reset at
    its start — matches training. Score = mean/max progress, wins.
  * --trial-episodes K > 1: run K consecutive episodes WITHOUT resetting the LSTM
    between them (a "trial"), and report score per episode index — the few-shot
    in-context adaptation curve (does episode k improve on episode 1?). This is
    off-distribution for a trial-simple-trained policy, so it's exploratory.
"""
import perf  # first: sets BLAS thread limits before torch/numpy import

import argparse

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecTransposeImage

from game_env import VecPrevActionReward, make_venv
from games import get_game


def _make_eval_venv(spec):
    # same wrapping order as training: prev-action/reward Dict, then the
    # automatic image transpose the recurrent policy was trained with
    return VecTransposeImage(VecPrevActionReward(make_venv(spec, 1)))


def _run_episode(venv, model, spec, lstm_states, carry, max_steps):
    """One episode. `carry` False resets the LSTM at the start (independent
    episode); True continues from `lstm_states` (few-shot trial). Returns
    (score, won, next_lstm_states)."""
    obs = venv.reset()
    episode_start = np.array([not carry])
    best_progress, ep_return, won, done, steps = 0, 0.0, False, False, 0
    while not done and steps < max_steps:
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_start, deterministic=False)
        obs, reward, dones, infos = venv.step(action)
        episode_start = np.array([False])
        ep_return += float(reward[0])
        if spec.progress_key:
            best_progress = max(best_progress, infos[0].get(spec.progress_key, 0))
        if spec.success_key:
            won = won or bool(infos[0].get(spec.success_key, False))
        done = bool(dones[0])
        steps += 1
    score = best_progress if spec.progress_key else ep_return
    return score, won, lstm_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="RecurrentPPO model.zip")
    ap.add_argument("--game", default="mario", help="held-out game to evaluate on")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--trial-episodes", type=int, default=1,
                    help=">1 carries LSTM state across consecutive episodes and "
                         "reports the per-episode adaptation curve")
    ap.add_argument("--max-steps", type=int, default=4000)
    args = ap.parse_args()

    perf.setup_cpu_threads(None)
    spec = get_game(args.game)
    venv = _make_eval_venv(spec)
    model = RecurrentPPO.load(args.model, device="cpu")

    metric = spec.progress_key or "episode return"
    print(f"Evaluating meta policy {args.model}\n  held-out on {spec.name} "
          f"(score = {metric}), {args.episodes} episode(s)\n", flush=True)

    if args.trial_episodes <= 1:
        scores, wins = [], 0
        for _ in range(args.episodes):
            score, won, _ = _run_episode(venv, model, spec, None, False, args.max_steps)
            scores.append(score)
            wins += int(won)
        print(f"{'mean':>8} {'max':>8} {'wins':>8}")
        print(f"{np.mean(scores):>8.0f} {np.max(scores):>8.0f} "
              f"{wins:>5}/{args.episodes}", flush=True)
        venv.close()
        return

    # few-shot trial: per-episode-index curve, averaged over independent trials
    k = args.trial_episodes
    per_index = [[] for _ in range(k)]
    for _ in range(args.episodes):
        lstm_states = None
        for i in range(k):
            score, _, lstm_states = _run_episode(
                venv, model, spec, lstm_states, carry=(i > 0), max_steps=args.max_steps)
            per_index[i].append(score)
    print(f"few-shot adaptation curve ({args.episodes} trials of {k} episodes)")
    print(f"{'episode':>8} {'mean score':>12}")
    for i in range(k):
        print(f"{i + 1:>8} {np.mean(per_index[i]):>12.0f}", flush=True)
    venv.close()


if __name__ == "__main__":
    main()
