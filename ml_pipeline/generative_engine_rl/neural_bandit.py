"""
neural_bandit.py
================
Neural Contextual Bandit for adversarial stress scenario generation.

Addresses the root failure identified in v5 forensics: PPO degenerates to
a context-free bandit with a broken value-function baseline in the 1-step MDP.
This replaces PPO with the correct algorithm class — a proper contextual
bandit that learns f(obs, action) → reward from beam oracle demonstrations.

Architecture
------------
    BanditRewardNet(obs, action) → scalar reward prediction

    obs_encoder:  Linear(obs_dim → 128) → LayerNorm → GELU →
                  Dropout → Linear(128 → 128) → LayerNorm → GELU
    action_embed: Embedding(n_targets, 16) ⊕ Embedding(n_families, 8)
                  ⊕ Embedding(n_mags, 8)
    fusion head:  Linear(160 → 64) → GELU → Dropout → Linear(64 → 1)

Inference
---------
    For a given obs, score all N actions and return argmax.
    Optional: MC-Dropout UCB — score = mean + beta * std over K dropout passes.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class BanditRewardNet(nn.Module):
    """Maps (observation, action) → expected adversarial reward."""

    def __init__(
        self,
        obs_dim: int,
        n_targets: int = 25,
        n_families: int = 8,
        n_mags: int = 21,
        hidden: int = 128,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_targets = n_targets
        self.n_families = n_families
        self.n_mags = n_mags
        self.hidden = hidden
        self.dropout_p = dropout

        self.obs_net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        emb_t, emb_f, emb_m = 16, 8, 8
        self.target_emb = nn.Embedding(n_targets, emb_t)
        self.family_emb = nn.Embedding(n_families, emb_f)
        self.mag_emb = nn.Embedding(n_mags, emb_m)

        fuse_in = hidden + emb_t + emb_f + emb_m
        self.head = nn.Sequential(
            nn.Linear(fuse_in, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,           # (B, obs_dim)
        target_idx: torch.Tensor,    # (B,) int64
        family_idx: torch.Tensor,    # (B,) int64
        mag_idx: torch.Tensor,       # (B,) int64
    ) -> torch.Tensor:               # (B,) float32
        z_obs = self.obs_net(obs)
        z_a = torch.cat(
            [
                self.target_emb(target_idx),
                self.family_emb(family_idx),
                self.mag_emb(mag_idx),
            ],
            dim=-1,
        )
        return self.head(torch.cat([z_obs, z_a], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def score_all(
        self,
        obs: torch.Tensor,      # (obs_dim,) single obs
        catalog: torch.Tensor,  # (n_actions, 3) int64
    ) -> torch.Tensor:          # (n_actions,) predicted rewards
        self.eval()
        n = catalog.shape[0]
        obs_exp = obs.unsqueeze(0).expand(n, -1)
        return self.forward(obs_exp, catalog[:, 0], catalog[:, 1], catalog[:, 2])

    def score_all_mc(
        self,
        obs: torch.Tensor,
        catalog: torch.Tensor,
        n_mc: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MC-Dropout: returns (mean, std) of predicted reward for UCB exploration."""
        self.train()  # enable dropout
        n = catalog.shape[0]
        obs_exp = obs.unsqueeze(0).expand(n, -1)
        preds = []
        with torch.no_grad():
            for _ in range(n_mc):
                preds.append(
                    self.forward(obs_exp, catalog[:, 0], catalog[:, 1], catalog[:, 2])
                )
        self.eval()
        stacked = torch.stack(preds, dim=0)  # (n_mc, n_actions)
        return stacked.mean(0), stacked.std(0)

    def save(self, path: str, meta: Optional[dict] = None) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "meta": {
                    "obs_dim": self.obs_dim,
                    "n_targets": self.n_targets,
                    "n_families": self.n_families,
                    "n_mags": self.n_mags,
                    "hidden": self.hidden,
                    "dropout": self.dropout_p,
                    **(meta or {}),
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "BanditRewardNet":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        m = ckpt["meta"]
        net = cls(
            obs_dim=m["obs_dim"],
            n_targets=m.get("n_targets", 25),
            n_families=m.get("n_families", 8),
            n_mags=m.get("n_mags", 21),
            hidden=m.get("hidden", 128),
            dropout=m.get("dropout", 0.15),
        )
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        return net


def build_catalog_tensor(env) -> torch.Tensor:
    """Build (n_actions, 3) int64 tensor of all [target, family, mag] tuples."""
    nvec = env.action_space.nvec
    rows = [
        [t, f, m]
        for t in range(int(nvec[0]))
        for f in range(int(nvec[1]))
        for m in range(int(nvec[2]))
    ]
    return torch.tensor(rows, dtype=torch.long)


def bandit_sequence(
    net: BanditRewardNet,
    catalog: torch.Tensor,
    env,
    seed: int,
    ucb_beta: float = 0.0,
    ucb_n_mc: int = 20,
) -> Dict:
    """Run one evaluation episode using the neural bandit policy.

    Returns the same dict schema as rl_sequence() / run_sequence() so it
    slots directly into the held-out benchmark pipeline.
    """
    obs_np, reset_info = env.reset(seed=seed)
    obs_t = torch.tensor(obs_np, dtype=torch.float32)

    decoded_actions = []
    total_reward = 0.0
    final_info: Dict = {}
    done = False

    while not done:
        if ucb_beta > 0.0:
            mean_s, std_s = net.score_all_mc(obs_t, catalog, n_mc=ucb_n_mc)
            scores = mean_s + ucb_beta * std_s
        else:
            scores = net.score_all(obs_t, catalog)

        best_idx = int(scores.argmax().item())
        action_np = catalog[best_idx].numpy().copy()
        decoded_actions.append(dict(env.decode_action(action_np)))

        obs_np, reward, terminated, truncated, info = env.step(action_np)
        obs_t = torch.tensor(obs_np, dtype=torch.float32)
        total_reward += float(reward)
        done = bool(terminated or truncated)
        final_info = info

    br = final_info.get("reward_breakdown")
    return {
        "seed": seed,
        "sequence": decoded_actions,
        "reward": float(total_reward),
        "portfolio_loss": float(br.portfolio_loss) if br else 0.0,
        "dfast_breach": float(br.dfast_breach) if br else 0.0,
        "causal_fidelity": float(br.causal_fidelity) if br else 0.0,
        "diversity": float(br.diversity) if br else 0.0,
        "sampled_state": reset_info.get("sampled_state"),
    }
