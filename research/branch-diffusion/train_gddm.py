#!/usr/bin/env python3
"""
train_gddm.py — Train a feature-denoising diffusion model for GDDM-style purification.

This is a PRAGMATIC simplification of GDDM (arXiv 2501.11568, 2025):
  Paper:  Binomial DDPM over adjacency A; clean-only training via VLB.
  Ours:   Gaussian DDPM over node features X; clean-only training via L2 noise prediction.

Why simplify?  See GDDM_NOTES.md. In short: GDDM's GSDR mask (A' ⊙ A^t) cannot
remove an *injected node*, only its incident edges, so on KATZE's node-injection
regime an X-purifier is more meaningful than an A-purifier. We keep the *paradigm*
(clean-only DDPM + partial-noise reverse at test time, DiffPure style) but on the
channel that actually matters for our attack family.

Architecture (small): 3-layer GCN denoiser with sinusoidal timestep embedding +
linear projection. Trained per-dataset on the clean training graphs.

Schedule: linear β_t, T=200, β_1=1e-4, β_T=0.02 (standard).

Usage:
    micromamba run -n graph_adversarial python train_gddm.py PROTEINS
    micromamba run -n graph_adversarial python train_gddm.py ENZYMES
"""
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import TUDataset

from prepare import MODELS_DIR, DATA_DIR, DEVICE
from prepare_cont import load_normalized_graphs, MODELS_CONT_DIR

CKPT_DIR = Path(__file__).parent / "gddm_checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Diffusion schedule (kept as module constants so defenses.py can import)
T_TRAIN = 200
BETA_1, BETA_T = 1e-4, 0.02
HIDDEN = 64
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3


def make_schedule(T=T_TRAIN, b1=BETA_1, bT=BETA_T, device="cpu"):
    """Linear β schedule + precomputed α / ᾱ for closed-form q(x_t | x_0)."""
    betas = torch.linspace(b1, bT, T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {"betas": betas, "alphas": alphas, "alpha_bar": alpha_bar}


def sinusoid_t_emb(t, dim):
    """Standard sinusoidal positional embedding for diffusion timestep."""
    device = t.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / max(half - 1, 1))
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class GCNDenoiser(nn.Module):
    """3-layer GCN that predicts the Gaussian noise added to node features at step t.

    Inputs:  x_t (n, d), edge_index (2, e), t in [0, T) per-graph -> broadcast to nodes.
    Outputs: ε_pred (n, d), same shape as x_t.
    """
    def __init__(self, in_dim, hidden=HIDDEN, t_emb_dim=64):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_emb_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.in_proj = nn.Linear(in_dim, hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.out = nn.Linear(hidden, in_dim)

    def forward(self, x_t, edge_index, t_per_node):
        # t_per_node: (n,) long timestep for each node (broadcast from per-graph t via batch)
        te = sinusoid_t_emb(t_per_node, self.t_emb_dim)        # (n, t_emb_dim)
        te = self.t_mlp(te)                                    # (n, hidden)
        h = self.in_proj(x_t) + te
        h = F.relu(self.conv1(h, edge_index)) + te
        h = F.relu(self.conv2(h, edge_index)) + te
        h = F.relu(self.conv3(h, edge_index))
        return self.out(h)


def q_sample(x0, t, sched, noise=None):
    """Forward: x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε. t is per-node long tensor."""
    if noise is None:
        noise = torch.randn_like(x0)
    ab = sched["alpha_bar"][t].unsqueeze(-1)                    # (n, 1)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise, noise


def train_one(graphs_train, num_features, save_path, epochs=EPOCHS, device=DEVICE):
    sched = make_schedule(device=device)
    model = GCNDenoiser(num_features).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    # Each Data object is a single graph; DataLoader batches them into a disjoint
    # super-graph with .batch indicating per-node graph id.
    loader = DataLoader(graphs_train, batch_size=BATCH_SIZE, shuffle=True)
    losses_per_epoch = []
    model.train()
    t0 = time.time()
    for ep in range(epochs):
        ep_losses = []
        for batch in loader:
            batch = batch.to(device)
            x0 = batch.x.float()
            # Per-graph timestep, broadcast to per-node via batch index
            n_graphs = int(batch.batch.max().item()) + 1
            t_g = torch.randint(0, T_TRAIN, (n_graphs,), device=device)
            t_n = t_g[batch.batch]                              # (n,)
            x_t, eps = q_sample(x0, t_n, sched)
            eps_pred = model(x_t, batch.edge_index, t_n)
            loss = F.mse_loss(eps_pred, eps)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_losses.append(loss.item())
        ep_loss = float(sum(ep_losses) / max(len(ep_losses), 1))
        losses_per_epoch.append(ep_loss)
        if ep == 0 or (ep + 1) % 10 == 0:
            print(f"  ep {ep+1:3d}/{epochs}  loss={ep_loss:.4f}  [{time.time()-t0:.1f}s]")
    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": num_features,
        "hidden": HIDDEN,
        "T": T_TRAIN, "beta_1": BETA_1, "beta_T": BETA_T,
        "losses": losses_per_epoch,
        "schedule": "linear-Gaussian-DDPM",
        "noise_target": "node_features_X",
    }, save_path)
    print(f"  saved {save_path}  final_loss={losses_per_epoch[-1]:.4f}")
    return losses_per_epoch


def _train_idx_union_from_meta(meta):
    """Each fold has its own test_idx; the union of all val/train idxs across folds
    is approximately the dataset minus a small held-out — for denoiser training we
    just use the whole dataset (clean graphs only; no labels needed)."""
    return None  # signal: use all graphs


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "PROTEINS"
    print(f"=== train_gddm {name}  T={T_TRAIN}  epochs={EPOCHS}  device={DEVICE} ===")

    if name == "PROTEINS":
        ds = TUDataset(root=str(DATA_DIR / "PROTEINS"), name="PROTEINS")
        graphs = [Data(x=ds[i].x.float(), edge_index=ds[i].edge_index, y=ds[i].y)
                  for i in range(len(ds))]
        num_features = ds.num_features
    elif name == "ENZYMES":
        # Use the SAME z-score normalization as the victim training
        meta = json.loads((MODELS_CONT_DIR / "meta_cont_ENZYMES.json").read_text())
        mean = torch.tensor(meta["feat_mean"]).view(1, -1)
        std = torch.tensor(meta["feat_std"]).view(1, -1)
        graphs, _, _, _, _ = load_normalized_graphs(name, mean, std)
        num_features = meta["num_features"]
    else:
        raise ValueError(f"unknown dataset {name}")

    print(f"  n_graphs={len(graphs)}  num_features={num_features}")
    save_path = CKPT_DIR / f"{name}.pt"
    train_one(graphs, num_features, save_path)


if __name__ == "__main__":
    main()
