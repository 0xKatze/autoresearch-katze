#!/usr/bin/env python3
"""
train_gddm_a.py — Train the A-channel (adjacency-side) GDDM denoiser.

This is the paper-faithful GDDM channel (arXiv 2501.11568):
  - Forward process: Binomial diffusion on the *adjacency* matrix A.
    Each edge is independently flipped with probability beta_t at step t.
  - Denoiser: 3-layer GCN on (X, A_t) producing node embeddings; a per-edge
    MLP head consumes (h_u, h_v, t_emb) and outputs a Bernoulli logit
    "is this edge present in the clean graph A_0?".
  - Loss: binary cross-entropy over the full n*n pair set per graph.
    For graphs with n > MAX_N_FULL the negative pairs are subsampled to keep
    the per-step work O(n).

Schedule: cosine beta_t from BETA_1=1e-4 to BETA_T=0.02, T=200 (paper-style).
Companion of train_gddm.py (the X-channel DiffPure variant — kept untouched).
See GDDM_A_NOTES.md.

Usage:
    micromamba run -n graph_adversarial python train_gddm_a.py PROTEINS
    micromamba run -n graph_adversarial python train_gddm_a.py ENZYMES
"""
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset

from prepare import DATA_DIR, DEVICE
from prepare_cont import load_normalized_graphs, MODELS_CONT_DIR

CKPT_DIR = Path(__file__).parent / "gddm_a_checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Diffusion schedule
T_TRAIN = 200
BETA_1, BETA_T = 1e-4, 0.02
HIDDEN = 64
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-5

# Per-edge classification cap: above this n, subsample negative pairs to keep cost O(n).
# PROTEINS max n ~481 -> with full n^2 we'd see 230k pairs/graph; we sub-sample to ~4*pos.
MAX_N_FULL = 80
NEG_POS_RATIO = 4.0  # for large graphs: sample ~4x as many non-edges as edges


def make_schedule(T=T_TRAIN, b1=BETA_1, bT=BETA_T, device="cpu"):
    """Linear beta schedule (paper alludes to cosine; linear is fine for our small T).

    Returns:
        betas:     (T,) per-step flip probability
        cum_keep:  (T,) prob that an edge survives WITHOUT a single flip across [1..t]
                   (we don't actually use this — see binomial_q_sample for the
                   closed-form aggregate flip prob)
        cum_flip:  (T,) closed-form q(A_t | A_0): each edge is flipped with prob
                   p_t = 0.5*(1 - prod(1-2*beta_s, s<=t))  (parity argument).
    """
    betas = torch.linspace(b1, bT, T, device=device)
    # Parity trick: flipping twice = no flip. P[parity = 1 after t indep flips]
    #   = 0.5 * (1 - prod_s (1 - 2*beta_s)).
    one_minus_two_beta = 1.0 - 2.0 * betas
    cum = torch.cumprod(one_minus_two_beta, dim=0)
    cum_flip = 0.5 * (1.0 - cum)            # P[bit flipped at step t]
    cum_keep = 1.0 - cum_flip
    return {"betas": betas, "cum_keep": cum_keep, "cum_flip": cum_flip}


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


def binomial_q_sample(A0, t_idx, sched):
    """Closed-form q(A_t | A_0): each entry (i,j) is independently flipped with
    probability cum_flip[t]. A0 is an n*n binary tensor; t_idx is a scalar long.

    Returns A_t (binary, symmetric, self-loops zeroed).
    """
    n = A0.shape[0]
    p_flip = sched["cum_flip"][t_idx].item()
    # Sample flips on upper triangle, mirror to keep symmetry
    triu = torch.triu(torch.ones(n, n, device=A0.device, dtype=torch.bool), diagonal=1)
    flips = (torch.rand(n, n, device=A0.device) < p_flip) & triu
    flips = flips | flips.t()
    A_t = (A0.bool() ^ flips).float()
    A_t.fill_diagonal_(0.0)
    return A_t


class EdgeDenoiser(nn.Module):
    """3-layer GCN on (X, A_t) + per-edge MLP head -> Bernoulli logit.

    The GCN is implemented manually via the symmetrically-normalized A_hat so
    we can feed dense A_t directly without converting to edge_index each step.
    For our small graphs (n <= ~500) dense is fine and gradient-friendly.
    """
    def __init__(self, in_dim, hidden=HIDDEN, t_emb_dim=64):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_emb_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden))
        self.in_proj = nn.Linear(in_dim, hidden)
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.lin3 = nn.Linear(hidden, hidden)
        # Edge head: [h_u || h_v || h_u*h_v || t_emb] -> logit
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * hidden + hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def _norm_adj(A):
        n = A.shape[0]
        A_hat = A + torch.eye(n, device=A.device)
        deg = A_hat.sum(dim=1).clamp_min(1e-6)
        d_inv_sqrt = deg.pow(-0.5)
        D = torch.diag(d_inv_sqrt)
        return D @ A_hat @ D

    def _gcn_layer(self, lin, h, A_hat_norm):
        return F.relu(A_hat_norm @ lin(h))

    def node_embeddings(self, x, A_t, t_scalar):
        """Run the 3-layer GCN on dense A_t. Returns (n, hidden)."""
        A_hat_norm = self._norm_adj(A_t)
        n = x.shape[0]
        t_vec = torch.full((n,), t_scalar, dtype=torch.long, device=x.device)
        te = self.t_mlp(sinusoid_t_emb(t_vec, self.t_emb_dim))  # (n, hidden)
        h = self.in_proj(x.float()) + te
        h = self._gcn_layer(self.lin1, h, A_hat_norm) + te
        h = self._gcn_layer(self.lin2, h, A_hat_norm) + te
        h = self._gcn_layer(self.lin3, h, A_hat_norm)
        return h, te

    def edge_logits(self, h, te, pairs):
        """pairs: (P, 2) long tensor of (u, v) indices. Returns (P,) logits."""
        u, v = pairs[:, 0], pairs[:, 1]
        hu, hv = h[u], h[v]
        teu = te[u]
        feat = torch.cat([hu, hv, hu * hv, teu], dim=-1)
        return self.edge_mlp(feat).squeeze(-1)

    def forward(self, x, A_t, t_scalar, pairs):
        h, te = self.node_embeddings(x, A_t, t_scalar)
        return self.edge_logits(h, te, pairs)


def _sample_pairs(n, A0, neg_ratio=NEG_POS_RATIO, device="cpu"):
    """For large graphs, subsample pairs: all upper-tri edges + neg_ratio*n_pos non-edges.
    For small graphs (n <= MAX_N_FULL), return ALL upper-triangle pairs.
    Returns (pairs (P,2), labels (P,) in {0,1})."""
    triu = torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)
    if n <= MAX_N_FULL:
        pairs = triu.nonzero(as_tuple=False)
        labels = A0[pairs[:, 0], pairs[:, 1]]
        return pairs, labels.float()
    # Large: take all positives + subsample negatives
    pos_mask = triu & (A0 > 0.5)
    neg_mask = triu & (A0 < 0.5)
    pos_pairs = pos_mask.nonzero(as_tuple=False)
    n_pos = pos_pairs.shape[0]
    n_neg_take = int(max(n_pos * neg_ratio, n_pos + 1))
    neg_all = neg_mask.nonzero(as_tuple=False)
    if neg_all.shape[0] > n_neg_take:
        sel = torch.randperm(neg_all.shape[0], device=device)[:n_neg_take]
        neg_pairs = neg_all[sel]
    else:
        neg_pairs = neg_all
    pairs = torch.cat([pos_pairs, neg_pairs], dim=0)
    labels = torch.cat([torch.ones(pos_pairs.shape[0], device=device),
                        torch.zeros(neg_pairs.shape[0], device=device)], dim=0)
    return pairs, labels


def graph_to_dense(g, device):
    """Build dense A0 (n,n) from edge_index; return (x, A0, n)."""
    n = g.num_nodes
    ei = g.edge_index.to(device)
    A0 = torch.zeros(n, n, device=device)
    if ei.numel() > 0:
        A0[ei[0], ei[1]] = 1.0
        A0 = ((A0 + A0.t()) > 0).float()
    A0.fill_diagonal_(0.0)
    return g.x.to(device).float(), A0, n


def train_one(graphs, num_features, save_path, epochs=EPOCHS, device=DEVICE):
    sched = make_schedule(device=device)
    model = EdgeDenoiser(num_features).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Precompute dense (x, A0) for each graph (eats some RAM but graphs are small)
    dense = [graph_to_dense(g, device) for g in graphs]

    losses_per_epoch = []
    t0 = time.time()
    model.train()
    n_graphs = len(dense)
    for ep in range(epochs):
        ep_losses = []
        perm = torch.randperm(n_graphs)
        for gi in perm.tolist():
            x, A0, n = dense[gi]
            if n < 2:
                continue
            # Per-graph timestep (1..T-1; t=0 is identity, not useful)
            t_scalar = int(torch.randint(1, T_TRAIN, (1,)).item())
            A_t = binomial_q_sample(A0, t_scalar, sched)
            pairs, labels = _sample_pairs(n, A0, device=device)
            logits = model(x, A_t, t_scalar, pairs)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
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
        "schedule": "linear-Binomial-DDPM-on-A",
        "noise_target": "adjacency_A",
        "edge_head": "MLP([h_u, h_v, h_u*h_v, t_emb]) -> logit",
    }, save_path)
    print(f"  saved {save_path}  final_loss={losses_per_epoch[-1]:.4f}")
    return losses_per_epoch


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "PROTEINS"
    print(f"=== train_gddm_a {name}  T={T_TRAIN}  epochs={EPOCHS}  device={DEVICE} ===")

    if name == "PROTEINS":
        ds = TUDataset(root=str(DATA_DIR / "PROTEINS"), name="PROTEINS")
        graphs = [Data(x=ds[i].x.float(), edge_index=ds[i].edge_index, y=ds[i].y)
                  for i in range(len(ds))]
        num_features = ds.num_features
    elif name == "ENZYMES":
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
