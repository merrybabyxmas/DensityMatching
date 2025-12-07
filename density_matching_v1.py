import os, math, random, time, json
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from sklearn.decomposition import PCA


# ============================================================================
# Config
# ============================================================================
@dataclass
class CFG:
    env_name: str = "Humanoid-v4"
    K: int = 32
    hidden: int = 256
    n_layers: int = 2

    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005

    batch_size: int = 512
    buffer_size: int = 1_000_000
    start_steps: int = 10000
    update_after: int = 10000
    update_every: int = 50
    max_steps: int = 6_000_000

    # Density matching params
    T: float = 1.0         # temperature for softmax
    M: int = 64            # number of sampled actions per state
    lambda_entropy: float = 0.01

    device: str = "cuda:2"
    seed: int = 42
    model_save_path = "./best_policy.pt"


# ============================================================================
# Replay Buffer
# ============================================================================
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2 = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts = np.zeros((size, act_dim), dtype=np.float32)
        self.rews = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max = size

    def store(self, s, a, r, s2, d):
        self.obs[self.ptr] = s
        self.acts[self.ptr] = a
        self.rews[self.ptr] = r
        self.obs2[self.ptr] = s2
        self.done[self.ptr] = d

        self.ptr = (self.ptr + 1) % self.max
        self.size = min(self.size + 1, self.max)

    def sample(self, B, device):
        idx = np.random.randint(0, self.size, B)
        return {
            "obs":  torch.tensor(self.obs[idx], device=device),
            "acts": torch.tensor(self.acts[idx], device=device),
            "rews": torch.tensor(self.rews[idx], device=device),
            "obs2": torch.tensor(self.obs2[idx], device=device),
            "done": torch.tensor(self.done[idx], device=device),
        }


# ============================================================================
# Networks
# ============================================================================
class MLP(nn.Module):
    def __init__(self, inp, out, hidden, n):
        super().__init__()
        layers = [nn.Linear(inp, hidden), nn.ReLU()]
        for _ in range(n - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GMMPolicy(nn.Module):
    def __init__(self, cfg, act_dim):
        super().__init__()
        self.cfg = cfg
        K, A = cfg.K, act_dim

        self.backbone = MLP(cfg.obs_dim, cfg.hidden, cfg.hidden, cfg.n_layers)
        self.head = nn.Linear(cfg.hidden, K * (2*A + 1))

    def forward(self, s):
        h = self.backbone(s)
        raw = self.head(h).view(s.size(0), self.cfg.K, -1)

        A = self.cfg.act_dim
        mu = raw[:, :, :A]
        log_std = torch.clamp(raw[:, :, A:2*A], -4, 2)
        w_logits = raw[:, :, 2*A]
        W = F.softmax(w_logits, dim=-1)
        return mu, log_std, W


    @torch.no_grad()
    def act_no_critic(self, s):
        mu, log_std, W = self.forward(s)
        idx = torch.argmax(W, dim=1)
        best = mu[torch.arange(s.size(0)), idx]
        return best

    def log_prob(self, a, mu, log_std, W):
        """Compute log π(a|s) for each sample"""
        B, M, A = a.shape       # a: (B, M, A)
        K = self.cfg.K

        a_exp = a.unsqueeze(2)   # (B,M,1,A)
        mu = mu.unsqueeze(1)     # (B,1,K,A)
        log_std = log_std.unsqueeze(1)  # (B,1,K,A)

        var = torch.exp(2 * log_std)

        log_gauss = -0.5 * ( ((a_exp - mu)**2)/var + 2*log_std + A*math.log(2*math.pi) )
        log_gauss = log_gauss.sum(-1)   # (B,M,K)

        log_mix = torch.log(W.unsqueeze(1) + 1e-12)        # (B,1,K)

        log_prob = torch.logsumexp(log_mix + log_gauss, dim=-1)   # (B,M)

        return log_prob
    
    @torch.no_grad()
    def act(self, s, critic):
        mu, log_std, W = self.forward(s)   # (B,K,A)
        B, K, A = mu.shape

        s_rep = s.repeat_interleave(K, dim=0)
        mu_flat = mu.reshape(B*K, A)

        q1, q2 = critic(s_rep, mu_flat)
        qvals = torch.min(q1, q2).view(B, K)

        idx = torch.argmax(qvals, dim=1)
        best = mu[torch.arange(B), idx]
        return best

class QNet(nn.Module):
    def __init__(self, sdim, adim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sdim + adim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


class Critic(nn.Module):
    def __init__(self, sdim, adim):
        super().__init__()
        self.q1 = QNet(sdim, adim)
        self.q2 = QNet(sdim, adim)

    def forward(self, s, a):
        return self.q1(s, a), self.q2(s, a)

    def soft_update(self, tgt, tau):
        with torch.no_grad():
            for p, tp in zip(self.parameters(), tgt.parameters()):
                tp.data.mul_(1-tau).add_(tau*p.data)


# ============================================================================
# KL-density matching loss
# ============================================================================
def density_matching_loss(cfg, policy, critic, states):
    """
    For each state s:
        1. sample M actions
        2. compute Q(s,a)
        3. target density p(a|s) = softmax(Q/T)
        4. log π(a|s) under GMM
        5. KL ≈ - Σ p * log π
    """

    B = states.size(0)
    A = cfg.act_dim
    M = cfg.M

    # sample actions
    a = torch.randn(B, M, A, device=states.device)

    # compute Q(s,a)
    s_exp = states.unsqueeze(1).repeat(1, M, 1)
    s_flat = s_exp.reshape(B*M, -1)
    a_flat = a.reshape(B*M, -1)

    q1, q2 = critic(s_flat, a_flat)
    Qvals = torch.min(q1, q2).reshape(B, M)

    # build target density
    p = F.softmax(Qvals / cfg.T, dim=-1)  # (B,M)

    # get policy params
    mu, log_std, W = policy(states)

    # compute log π(a|s)
    logp = policy.log_prob(a, mu, log_std, W)   # (B,M)

    # KL approx: L = - Σ p logπ
    L_kl = - torch.sum(p * logp) / B

    # entropy bonus
    ent = (W * log_std.sum(-1)).sum(-1).mean()
    L_ent = - cfg.lambda_entropy * ent

    return L_kl + L_ent, L_kl.detach(), L_ent.detach()


# ============================================================================
# Trainer
# ============================================================================
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        env = gym.make(cfg.env_name)
        obs, _ = env.reset(seed=cfg.seed)

        cfg.obs_dim = obs.shape[0]
        cfg.act_dim = env.action_space.shape[0]
        self.env = env

        self.policy = GMMPolicy(cfg, cfg.act_dim).to(cfg.device)
        self.critic = Critic(cfg.obs_dim, cfg.act_dim).to(cfg.device)
        self.target = Critic(cfg.obs_dim, cfg.act_dim).to(cfg.device)
        self.target.load_state_dict(self.critic.state_dict())

        self.opt_actor = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

        self.buffer = ReplayBuffer(cfg.obs_dim, cfg.act_dim, cfg.buffer_size)
        self.best_return = -float("inf")
        self.save_path = cfg.model_save_path

        print("[INFO] Trainer ready — Using density matching GMM.")

    # -----------------------------
    def update_critic(self, batch):
        cfg = self.cfg
        s, a, r, s2, d = batch["obs"], batch["acts"], batch["rews"], batch["obs2"], batch["done"]

        q1, q2 = self.critic(s, a)
        with torch.no_grad():
            a2 = self.policy.act(s2, self.critic)
            tq1, tq2 = self.target(s2, a2)
            y = r.unsqueeze(-1) + cfg.gamma * (1-d).unsqueeze(-1) * torch.min(tq1,tq2)

        loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.opt_critic.zero_grad()
        loss.backward()
        self.opt_critic.step()

        return loss.item()

    # -----------------------------
    def update_actor(self, states):
        cfg = self.cfg
        loss, L_kl, L_ent = density_matching_loss(cfg, self.policy, self.critic, states)

        self.opt_actor.zero_grad()
        loss.backward()
        self.opt_actor.step()

        return loss.item(), L_kl.item(), L_ent.item()

    # -----------------------------
    def train(self):
        cfg = self.cfg
        obs, _ = self.env.reset(seed=cfg.seed)
        episode_ret = 0

        critic_loss_val = None
        actor_loss_val = None

        for step in range(1, cfg.max_steps+1):
            s_time = time.time()

            # rollout
            if step < cfg.start_steps:
                act = self.env.action_space.sample()
            else:
                s_t = torch.tensor(obs, device=cfg.device).float().unsqueeze(0)
                act = self.policy.act(s_t, self.critic)[0].cpu().numpy()

            next_obs, r, done, trunc, _ = self.env.step(act)
            self.buffer.store(obs, act, r, next_obs, float(done or trunc))

            obs = next_obs
            episode_ret += r


                
                            
            if done or trunc:
                print(f"[Episode] return={episode_ret:.1f}")

                # ---- wandb log: episode return ----
                wandb.log({
                    "episode/return": episode_ret,
                    "episode/best_return": self.best_return,
                    "episode/step": step
                })

                # ---- BEST MODEL SAVE ----
                if episode_ret > self.best_return:
                    self.best_return = episode_ret
                    torch.save(self.policy.state_dict(), self.save_path)
                    print(f"💾 Saved BEST policy (return={episode_ret:.1f}) → {self.save_path}")

                obs, _ = self.env.reset()
                episode_ret = 0



            # updates
            if step >= cfg.update_after and step % cfg.update_every == 0:
                for _ in range(cfg.update_every):
                    batch = self.buffer.sample(cfg.batch_size, cfg.device)
                    critic_loss_val = self.update_critic(batch)
                    actor_loss_val, kl_val, ent_val = self.update_actor(batch["obs"])
                    wandb.log({
                        "update/critic_loss": critic_loss_val,
                        "update/actor_loss": actor_loss_val,
                        "update/KL_loss": kl_val,
                        "update/entropy_loss": ent_val,
                        "update/step": step
                    })

                self.critic.soft_update(self.target, cfg.tau)

            if step % 1000 == 0:
                print(f"[{step}] critic={critic_loss_val} actor={actor_loss_val}")

                wandb.log({
                    "step/critic_loss": critic_loss_val,
                    "step/actor_loss": actor_loss_val,
                    "step": step
                })


#!/usr/bin/env python3
import argparse
import torch
import random
import numpy as np
import wandb

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="Humanoid-v4",
                        help="Environment name")

    parser.add_argument("--device", type=str, default="cuda:2",
                        help="Device for training")

    parser.add_argument("--K", type=int, default=32,
                        help="Number of GMM components")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional wandb run name")

    parser.add_argument("--project", type=str, default="RL_humanoid",
                        help="wandb project name")

    return parser.parse_args()



def main():
    args = parse_args()

    # -------------------------------------------------------------
    # Setup config
    # -------------------------------------------------------------
    cfg = CFG()
    cfg.env_name = args.env
    cfg.device = args.device
    cfg.K = args.K
    cfg.seed = args.seed

    # -------------------------------------------------------------
    # Seeding
    # -------------------------------------------------------------
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # -------------------------------------------------------------
    # wandb init
    # -------------------------------------------------------------
    wandb.init(
        project=args.project,
        name=args.run_name if args.run_name is not None
                           else f"{cfg.env_name}_K{cfg.K}_seed{cfg.seed}",
        config=cfg.__dict__
    )

    # -------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------
    trainer = Trainer(cfg)

    # -------------------------------------------------------------
    # Train
    # -------------------------------------------------------------
    trainer.train()



if __name__ == "__main__":
    main()
