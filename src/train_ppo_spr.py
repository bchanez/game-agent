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
from collections import deque

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.utils import obs_as_tensor

from auto_config import auto_configure
from game_env import make_multi_venv, make_venv
from games import get_game, get_games
from nets import is_grid_space, policy_kwargs_for
from rnd import RNDReward, RNDTrainCallback
from spr import SPRHead, ema_update

MODELS_DIR = "/app/data/models"
TB_DIR = "/app/data/tb"


class PPOSPR(PPO):
    def __init__(self, *args, spr_coef=1.0, spr_lr=1e-4, spr_epochs=1,
                 spr_batch=256, spr_tau=0.01,
                 sil_coef=0.0, sil_epochs=4, sil_batch=64, sil_buffer=20000,
                 sil_value_coef=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.spr_coef = spr_coef
        self.spr_epochs = spr_epochs
        self.spr_batch = spr_batch
        self.spr_tau = spr_tau
        # SPR trains the *policy's* encoder, so it must feed it obs on the same
        # scale the policy does: /255 for images, but a one-hot grid is already 0/1.
        self._normalize_images = not is_grid_space(self.observation_space)

        # Self-imitation (Oh et al. 2018), sparse-reward variant: keep the
        # transitions of episodes that actually *earned reward* and replay them,
        # imitating actions whose Monte-Carlo return beat the value estimate — so
        # PPO stops discarding its rare successes. SILCollector fills the buffer
        # (whole episodes, across rollouts), sil_push_episode keeps the winners.
        self.sil_coef = sil_coef
        self.sil_epochs = sil_epochs
        self.sil_batch = sil_batch
        self.sil_value_coef = sil_value_coef
        self._sil_obs = deque(maxlen=sil_buffer)
        self._sil_act = deque(maxlen=sil_buffer)
        self._sil_ret = deque(maxlen=sil_buffer)

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
        prep = preprocess_obs(obs_t, self.observation_space,
                              normalize_images=self._normalize_images)
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

    def sil_push_episode(self, obs_list, act_list, ext_list):
        """A finished episode (from SILCollector). Keep it only if it earned reward;
        the imitation target is the Monte-Carlo return per step."""
        if sum(ext_list) <= 0:
            return
        g, rets = 0.0, [0.0] * len(ext_list)
        for i in reversed(range(len(ext_list))):
            g = ext_list[i] + self.gamma * g
            rets[i] = g
        for o, a, r in zip(obs_list, act_list, rets):
            self._sil_obs.append(o)
            self._sil_act.append(a)
            self._sil_ret.append(r)

    def _sil_update(self):
        n = len(self._sil_obs)
        self.logger.record("sil/buffer", float(n))       # log even before it fires
        if n < self.sil_batch:
            return
        self.policy.set_training_mode(True)
        obs_all, act_all = np.asarray(self._sil_obs), np.asarray(self._sil_act)
        ret_all = np.asarray(self._sil_ret, dtype=np.float32)
        last = 0.0
        for _ in range(self.sil_epochs):
            idx = np.random.randint(0, n, self.sil_batch)
            obs_b = obs_as_tensor(obs_all[idx], self.device)
            act_b = torch.as_tensor(act_all[idx], device=self.device).long().flatten()
            ret_b = torch.as_tensor(ret_all[idx], device=self.device)
            values, log_prob, _ = self.policy.evaluate_actions(obs_b, act_b)
            # imitate only where the return beat the critic (the "better than
            # expected" past actions) — the SIL clipped advantage
            adv = (ret_b - values.flatten()).clamp(min=0.0)
            policy_loss = -(log_prob * adv.detach()).mean()
            value_loss = 0.5 * (adv ** 2).mean()
            loss = self.sil_coef * (policy_loss + self.sil_value_coef * value_loss)
            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()
            last = float(loss.item())
        self.logger.record("sil/loss", last)

    def train(self):
        if self.spr_coef > 0:
            self._spr_update()
        super().train()
        if self.sil_coef > 0:
            self._sil_update()


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


class AutoGamma(BaseCallback):
    """Set the discount from the *observed* episode horizon, one generic rule for
    every game (no per-game tuning): PPO's effective horizon is 1/(1-gamma), so we
    track the running mean episode length L and set gamma = 1 - 1/L. A sparse,
    long-horizon game (ARC) gets a far-sighted discount automatically; a short one
    gets a myopic one — without us ever picking a value by hand.

    Rationale: a fixed gamma tuned for Mario's dense reward (0.9) is near-blind on
    a task where the payoff is 100+ steps away (0.9^100 ~ 3e-5). The horizon is the
    thing the agent can *measure*, so we derive gamma from it."""

    def __init__(self, min_gamma=0.9, max_gamma=0.999, momentum=0.99):
        super().__init__()
        self.min_gamma, self.max_gamma, self.momentum = min_gamma, max_gamma, momentum
        self._mean_len = None

    def _on_step(self):
        for info in self.locals["infos"]:
            ep = info.get("episode")
            if ep is not None:                         # VecMonitor tags episode end
                l = float(ep["l"])
                self._mean_len = l if self._mean_len is None else \
                    self.momentum * self._mean_len + (1 - self.momentum) * l
        return True

    def _on_rollout_end(self):
        if self._mean_len is None:
            return
        gamma = float(np.clip(1.0 - 1.0 / max(self._mean_len, 1.0),
                              self.min_gamma, self.max_gamma))
        self.model.gamma = gamma                       # read by the next GAE computation
        self.logger.record("train/auto_gamma", gamma)
        self.logger.record("train/mean_ep_len", self._mean_len)


class SILCollector(BaseCallback):
    """Accumulate each episode's (obs, action, extrinsic reward) across rollout
    boundaries and hand *winning* episodes to the self-imitation buffer on
    termination — robust to episodes that span rollouts (completing a level does
    not end the episode). info['extrinsic'] is the true game reward under RND;
    self.model._last_obs is still the pre-step obs when _on_step fires."""

    def _on_training_start(self):
        n = self.training_env.num_envs
        self._obs = [[] for _ in range(n)]
        self._act = [[] for _ in range(n)]
        self._ext = [[] for _ in range(n)]

    def _on_step(self):
        last_obs, actions = self.model._last_obs, self.locals["actions"]
        dones, infos, rews = (self.locals["dones"], self.locals["infos"],
                              self.locals["rewards"])
        for e in range(len(dones)):
            self._obs[e].append(np.asarray(last_obs[e]))
            self._act[e].append(np.asarray(actions[e]))
            self._ext[e].append(float(infos[e].get("extrinsic", rews[e])))
            if dones[e]:
                self.model.sil_push_episode(self._obs[e], self._act[e], self._ext[e])
                self._obs[e], self._act[e], self._ext[e] = [], [], []
        return True


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
    ap.add_argument("--sil-coef", type=float, default=0.0,
                    help="self-imitation weight (0 = off): replay winning episodes "
                         "so PPO reuses its rare sparse-reward successes")
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
    ap.add_argument("--gamma", type=float, default=0.9,
                    help="discount (0.9 suits dense reward; raise for sparse/long-horizon)")
    ap.add_argument("--auto-gamma", action="store_true",
                    help="derive gamma from the observed episode horizon (generic, "
                         "no per-game tuning) — overrides --gamma dynamically")
    ap.add_argument("--frame-stack", type=int, default=1,
                    help="stack N frames for a raw grid game (perceive motion); "
                         "pixel games already stack 4")
    ap.add_argument("--auto", action="store_true",
                    help="self-configure: probe the game and pick the tools "
                         "(curiosity/SIL/frame-stack) from measurement, not by hand")
    args = ap.parse_args()

    perf.setup_cpu_threads(args.torch_threads)

    if args.auto and not args.games:
        cfg, _ = auto_configure(get_game(args.game))
        args.intrinsic_coef = cfg["intrinsic_coef"]
        args.sil_coef = cfg["sil_coef"]
        args.frame_stack = cfg["frame_stack"]
        args.auto_gamma = cfg["auto_gamma"]

    os.makedirs(MODELS_DIR, exist_ok=True)
    if args.games:
        names = [n.strip() for n in args.games.split(",") if n.strip()]
        venv = make_multi_venv(get_games(names), args.n_envs)
        game_label = "+".join(names)
    else:
        venv = make_venv(get_game(args.game), args.n_envs, frame_stack=args.frame_stack)
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
        learning_rate=1e-4, gamma=args.gamma, gae_lambda=1.0, ent_coef=0.01,
        tensorboard_log=TB_DIR, device="cpu", seed=args.seed,
        spr_coef=args.spr_coef, spr_lr=args.spr_lr, sil_coef=args.sil_coef,
        policy_kwargs=policy_kwargs_for(venv.observation_space),
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

    callbacks = [ckpt]
    if curious:
        callbacks += [CuriosityStats(), RNDTrainCallback(venv)]
    if args.auto_gamma:
        callbacks.append(AutoGamma())
    if args.sil_coef > 0:
        callbacks.append(SILCollector())
    print(f"Training PPO+SPR (coef={args.spr_coef}, intrinsic={args.intrinsic_coef}, "
          f"gamma={'auto' if args.auto_gamma else args.gamma}) "
          f"for {args.timesteps} steps on {args.n_envs} env(s)...", flush=True)
    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    final = args.out or os.path.join(MODELS_DIR, f"{prefix}_final")
    model.save(final)
    print(f"Saved {final}.zip", flush=True)


if __name__ == "__main__":
    main()
