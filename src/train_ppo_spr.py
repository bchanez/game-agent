"""Train PPO with an SPR self-predictive auxiliary loss on its own CNN features.

Stage B of Phase 8.5 (see spr.py / ROADMAP.md). Same PPO recipe as train_ppo.py,
but each update also trains the shared CNN encoder to predict the next latent from
the current latent and action. The bet: a dynamics-aware representation makes PPO
learn *faster* (better sample-efficiency) than raw-pixel PPO.

    python src/train_ppo_spr.py --game breakout --timesteps 500000 --n-envs 8

Smoke-test to a scratch path (never clobber a real model):
    python src/train_ppo_spr.py --game breakout --timesteps 25000 --out /app/data/models/_smoke

--spr-coef 0 falls back to plain PPO (a scientific control).
"""
import perf  # first: sets BLAS thread limits before torch/numpy import

import argparse
import copy
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.preprocessing import preprocess_obs

from game_env import make_multi_venv, make_venv
from games import get_game, get_games
from rnd import RNDReward, RNDTrainCallback
from spr import SPRHead, ema_update

MODELS_DIR = "/app/data/models"
TB_DIR = "/app/data/tb"


class PPOSPR(PPO):
    def __init__(self, *args, spr_coef=1.0, spr_lr=1e-4, spr_epochs=1,
                 spr_batch=256, spr_tau=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.spr_coef = spr_coef
        self.spr_epochs = spr_epochs
        self.spr_batch = spr_batch
        self.spr_tau = spr_tau
        if spr_coef <= 0:
            return

        feature_dim = self.policy.features_extractor.features_dim
        self.spr_head = SPRHead(feature_dim, self.action_space.n).to(self.device)
        # EMA copy of the encoder: the moving target that prevents collapse
        self.target_encoder = copy.deepcopy(self.policy.features_extractor).to(self.device)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.spr_optimizer = torch.optim.Adam(
            list(self.policy.features_extractor.parameters())
            + list(self.spr_head.parameters()),
            lr=spr_lr,
        )

    def _features(self, obs_np, encoder):
        obs_t = torch.as_tensor(obs_np, device=self.device)
        prep = preprocess_obs(obs_t, self.observation_space, normalize_images=True)
        return encoder(prep)

    def _spr_update(self):
        # must run before super().train(): rollout_buffer.get() flattens
        # observations/actions in place, but leaves episode_starts (n_steps, n_envs)
        buf = self.rollout_buffer
        obs, actions = buf.observations, buf.actions      # (n_steps, n_envs, ...)
        # a transition t->t+1 is valid only within one episode
        mask = ~buf.episode_starts[1:].astype(bool)       # (n_steps-1, n_envs)
        t_idx, e_idx = np.nonzero(mask)
        if len(t_idx) == 0:
            return
        s_t, s_tp1 = obs[t_idx, e_idx], obs[t_idx + 1, e_idx]
        a_t = actions[t_idx, e_idx]

        self.policy.set_training_mode(True)
        n = len(t_idx)
        for _ in range(self.spr_epochs):
            perm = np.random.permutation(n)
            for start in range(0, n, self.spr_batch):
                idx = perm[start:start + self.spr_batch]
                z_t = self._features(s_t[idx], self.policy.features_extractor)
                with torch.no_grad():
                    z_tp1 = self._features(s_tp1[idx], self.target_encoder)
                a = torch.as_tensor(a_t[idx], device=self.device)
                loss = self.spr_coef * self.spr_head(z_t, a, z_tp1)
                self.spr_optimizer.zero_grad()
                loss.backward()
                self.spr_optimizer.step()

        self.spr_head.update_target(self.spr_tau)
        ema_update(self.target_encoder, self.policy.features_extractor, self.spr_tau)
        self.logger.record("spr/loss", float(loss.item()))

    def train(self):
        if self.spr_coef > 0:
            self._spr_update()
        super().train()


class CuriosityStats(BaseCallback):
    """With RND, the reward PPO sees is extrinsic+intrinsic mixed. Log the raw
    extrinsic separately so we can tell real game progress from novelty-seeking."""

    def _on_rollout_start(self):
        self._extr = []

    def _on_step(self):
        for info in self.locals["infos"]:
            if "extrinsic" in info:
                self._extr.append(info["extrinsic"])
        return True

    def _on_rollout_end(self):
        if self._extr:
            self.logger.record("curiosity/extrinsic_mean", float(np.mean(self._extr)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="breakout")
    ap.add_argument("--games", default=None,
                    help="comma-separated list to train ONE policy on a mix of "
                         "games (overrides --game); e.g. 'mario,montezuma'")
    ap.add_argument("--init-from", default=None,
                    help="load policy weights from this model.zip before training "
                         "(measures transfer: pretrain on games A,B -> learn C)")
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--spr-coef", type=float, default=1.0,
                    help="weight of the self-predictive loss (0 = plain PPO control)")
    ap.add_argument("--spr-lr", type=float, default=1e-4)
    ap.add_argument("--intrinsic-coef", type=float, default=0.0,
                    help=">0 adds RND curiosity (for sparse-reward games); composes with SPR")
    ap.add_argument("--extrinsic-coef", type=float, default=1.0)
    ap.add_argument("--out", default=None,
                    help="output prefix (else data/models/<game>_ppo_spr_final); "
                         "use a scratch path for smoke tests")
    ap.add_argument("--torch-threads", type=int, default=None,
                    help="learner torch threads (default: all cores)")
    ap.add_argument("--n-epochs", type=int, default=4,
                    help="PPO passes per rollout (lower = faster updates)")
    ap.add_argument("--rnd-update-proportion", type=float, default=0.25,
                    help="fraction of each rollout used to train the RND predictor")
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
    curious = args.intrinsic_coef > 0
    if curious:
        # RND only rewrites the reward, so it stacks under PPOSPR untouched
        venv = RNDReward(venv, intrinsic_coef=args.intrinsic_coef,
                         extrinsic_coef=args.extrinsic_coef, device="cpu",
                         update_proportion=args.rnd_update_proportion)

    model = PPOSPR(
        "CnnPolicy", venv, verbose=1,
        n_steps=512, batch_size=64, n_epochs=args.n_epochs,
        learning_rate=1e-4, gamma=0.9, gae_lambda=1.0, ent_coef=0.01,
        tensorboard_log=TB_DIR, device="cpu", seed=args.seed,
        spr_coef=args.spr_coef, spr_lr=args.spr_lr,
    )

    if args.init_from:
        src = PPO.load(args.init_from, device="cpu")
        model.policy.load_state_dict(src.policy.state_dict())
        if model.spr_coef > 0:  # restart the EMA target aligned with the transferred encoder
            model.target_encoder.load_state_dict(model.policy.features_extractor.state_dict())
        print(f"Initialized policy from {args.init_from}", flush=True)

    prefix = f"{game_label}_ppo_spr" + ("_rnd" if curious else "")
    ckpt = CheckpointCallback(
        save_freq=max(20_000 // args.n_envs, 1),
        save_path=MODELS_DIR, name_prefix=prefix,
    )

    callbacks = [ckpt, CuriosityStats(), RNDTrainCallback(venv)] if curious else ckpt
    print(f"Training PPO+SPR (coef={args.spr_coef}, intrinsic={args.intrinsic_coef}) "
          f"for {args.timesteps} steps on {args.n_envs} env(s)...", flush=True)
    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    final = args.out or os.path.join(MODELS_DIR, f"{prefix}_final")
    model.save(final)
    print(f"Saved {final}.zip", flush=True)


if __name__ == "__main__":
    main()
