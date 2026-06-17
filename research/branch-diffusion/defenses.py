#!/usr/bin/env python3
"""
defenses.py — Lightweight graph defense wrappers for evaluating attack robustness.

Three defenses implemented (no external dependencies beyond torch / torch_geometric):

1. GNN-Jaccard (Wu et al., IJCAI 2019):
   Prune edges where endpoint feature similarity < threshold.
   Defends against feature-dissimilarity injection attacks.

2. GCN-SVD (Entezari et al., WSDM 2020):
   Replace adjacency with rank-k SVD approximation.
   Defends by low-frequency filtering — high-frequency attack perturbations are smoothed out.

3. NormFilter (folklore / KATZE-specific):
   Remove nodes whose feature L2 norm exceeds threshold (e.g. mean + k*std of training data).
   Direct counter to KATZE's high-feat_scale injection.

Usage:
    from defenses import DefendedModel, jaccard_defense, svd_defense, norm_filter_defense

    defended = DefendedModel(victim_model, jaccard_defense, threshold=0.01)
    asr = attack.run_attack(defended, test_graphs, device)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


# ============================================================
# Defense functions: data -> filtered_data
# ============================================================

def jaccard_defense(data, threshold=0.01, similarity="cosine"):
    """
    Prune edges where endpoint feature similarity < threshold.

    For binary features (PROTEINS), use Jaccard; for continuous, cosine.
    Both reduce to similarity-based edge filtering.
    """
    x, ei = data.x, data.edge_index
    src, dst = ei[0], ei[1]
    if similarity == "cosine":
        # Cosine similarity, robust for continuous + binary features
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-8)
        sim = (x_norm[src] * x_norm[dst]).sum(dim=1)
    else:  # jaccard
        # |A ∩ B| / |A ∪ B| for binary features
        inter = (x[src] * x[dst]).sum(dim=1)
        union = ((x[src] + x[dst]) > 0).float().sum(dim=1).clamp(min=1)
        sim = inter / union

    keep = sim >= threshold
    new_ei = ei[:, keep]
    return Data(x=data.x, edge_index=new_ei,
                y=getattr(data, "y", None),
                batch=getattr(data, "batch", None))


def svd_defense(data, rank=10):
    """
    Replace edge_index with rank-k SVD reconstruction of adjacency.

    Caution: O(n^3) eigendecomposition — only practical for small graphs (<300 nodes).
    """
    n = data.num_nodes
    ei = data.edge_index
    device = data.x.device

    # Dense adjacency
    A = torch.zeros(n, n, device=device)
    A[ei[0], ei[1]] = 1.0
    A = (A + A.t()).clamp(max=1.0)  # symmetric

    # SVD (use eigendecomp for symmetric)
    try:
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        k = min(rank, S.shape[0])
        A_recon = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
    except Exception:
        # If SVD fails (rare), return original
        return data

    # Re-binarize: keep edges with reconstruction value > 0.5
    mask = (A_recon > 0.5).nonzero(as_tuple=False)
    new_ei = mask.t()  # (2, num_edges)

    return Data(x=data.x, edge_index=new_ei,
                y=getattr(data, "y", None),
                batch=getattr(data, "batch", None))


def norm_filter_defense(data, max_norm=None, percentile=99):
    """
    Remove nodes whose feature L2 norm exceeds threshold.

    If max_norm is None, threshold = percentile-th of current graph's norms.
    This is a within-graph anomaly detector — robust if attack uses high feat_scale.
    """
    x = data.x
    norms = x.norm(p=2, dim=1)

    if max_norm is None:
        # Within-graph percentile threshold
        max_norm = torch.quantile(norms, percentile / 100.0).item()

    keep = norms <= max_norm
    if keep.sum() == 0:
        # All nodes flagged — fall back to keeping all (defense saturated)
        return data

    new_x = x[keep]
    # Remap node indices in edge_index
    n_orig = x.shape[0]
    idx_map = torch.full((n_orig,), -1, dtype=torch.long, device=x.device)
    idx_map[keep] = torch.arange(keep.sum(), device=x.device)

    ei = data.edge_index
    edge_keep = (idx_map[ei[0]] >= 0) & (idx_map[ei[1]] >= 0)
    new_ei = torch.stack([idx_map[ei[0][edge_keep]],
                          idx_map[ei[1][edge_keep]]], dim=0)

    batch = getattr(data, "batch", None)
    new_batch = batch[keep] if batch is not None else None

    return Data(x=new_x, edge_index=new_ei,
                y=getattr(data, "y", None),
                batch=new_batch)


def featclip_defense(data, mean=None, std=None, k=2.0):
    """Clip each node feature per-dim to [mean - k*std, mean + k*std].

    Distribution-aware (uses train per-dim mean/std passed as kwargs; falls back to
    within-batch stats if absent). Does NOT remove nodes — it pulls off-distribution
    feature *values* back to the data range. Counters injections that rely on extreme
    per-dim magnitudes; a norm-capped node with extreme *direction* can still be clipped.
    """
    x = data.x
    if mean is None or std is None:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    else:
        mean = mean.to(x.device).view(1, -1)
        std = std.to(x.device).clamp_min(1e-6).view(1, -1)
    lo, hi = mean - k * std, mean + k * std
    new_x = torch.maximum(torch.minimum(x, hi), lo)
    return Data(x=new_x, edge_index=data.edge_index,
                y=getattr(data, "y", None), batch=getattr(data, "batch", None))


def node_degrees(data):
    """Per-node degree = number of incident edge-endpoints in edge_index.

    Counts appearances across BOTH rows of edge_index, so the result is robust to
    whether the (possibly attacker-added) edges are stored single- or bi-directional.
    For a standard symmetric undirected PyG graph this equals 2*deg, but the defense
    and its threshold are computed with the SAME convention, so the factor cancels.
    """
    n = data.num_nodes
    deg = torch.zeros(n, device=data.x.device)
    ei = data.edge_index
    if ei.numel() > 0:
        idx = ei.reshape(-1)
        deg.scatter_add_(0, idx, torch.ones_like(idx, dtype=deg.dtype))
    return deg


def degree_filter_defense(data, max_degree=None, mean=None, std=None, k=3.0):
    """Remove nodes whose degree exceeds a clean-distribution threshold.

    STRUCTURE-aware analogue of norm_filter_defense / mahalanobis_defense: it targets
    the *connectivity* outlier, not the *feature* outlier. A full-edge single-node
    injection attaches the injected node to ~every existing node (degree ~= N), a gross
    outlier vs the clean degree distribution (PROTEINS clean: mean~3.7, p99.5~7, hub max
    ~25 counting one direction). Feature filters (NormFilter/Mahalanobis) never engage on
    such an attack because the injected feature can sit in-distribution while the flip
    rides the dense attachment — this defense binds against exactly that.

    Threshold precedence (mirrors mahalanobis_defense's "stats injected at wrap time"):
      1. explicit `max_degree`               -> use as-is
      2. clean `mean` + `std` (+ k)          -> threshold = mean + k*std
      3. neither -> within-graph mean + k*std fallback (standalone use)

    Note: with (1)/(2) the threshold is a FIXED clean-distribution value, so the injected
    node cannot evade it by inflating in-graph statistics. Node removal + edge/batch remap
    mirror norm_filter_defense exactly.
    """
    deg = node_degrees(data)
    if max_degree is not None:
        thr = float(max_degree)
    elif mean is not None and std is not None:
        thr = float(mean) + k * float(std)
    else:
        thr = deg.mean().item() + k * deg.std().item()

    keep = deg <= thr
    if keep.sum() == 0:
        # Defense saturated — keep all nodes rather than empty the graph.
        return data

    x = data.x
    new_x = x[keep]
    n_orig = x.shape[0]
    idx_map = torch.full((n_orig,), -1, dtype=torch.long, device=x.device)
    idx_map[keep] = torch.arange(int(keep.sum()), device=x.device)
    ei = data.edge_index
    edge_keep = (idx_map[ei[0]] >= 0) & (idx_map[ei[1]] >= 0)
    new_ei = torch.stack([idx_map[ei[0][edge_keep]],
                          idx_map[ei[1][edge_keep]]], dim=0)
    batch = getattr(data, "batch", None)
    new_batch = batch[keep] if batch is not None else None
    return Data(x=new_x, edge_index=new_ei,
                y=getattr(data, "y", None), batch=new_batch)


def _gnnsafe_node_logits(victim_model, data):
    """Per-node "would-be" class logits from the victim GCN, by replaying its
    conv stack and then applying the classifier head per-node (skipping global_mean_pool).

    Mirrors prepare.GCN_GraphClassification.forward but stops before the pool:
      x = conv_i -> bn_i -> relu -> dropout, for each layer; then
      node_logits = classifier(x).

    This is the only way to get C-dim logits per node from a graph-classification GCN.
    """
    x, edge_index = data.x, data.edge_index
    for conv, bn in zip(victim_model.conv_layers, victim_model.batch_norms):
        x = conv(x, edge_index)
        x = bn(x)
        x = F.relu(x)
        # Eval-mode dropout is identity; no need to call F.dropout here.
    return victim_model.classifier(x)  # (N, C)


def _energy_propagation(e, edge_index, num_nodes, K=2, alpha=0.5):
    """GNNSafe Eq. 8 energy belief propagation (verified vs. official repo
    `GNNSafe/gnnsafe.py::propagation`). e: (N,) energy per node.

    Builds D^{-1} A as in the reference code (`d_norm = 1/deg(col)`,
    SparseTensor(row=col, col=row, value=d_norm)) and iterates
      e <- alpha * e + (1-alpha) * (D^{-1} A) e   for K steps.
    """
    if num_nodes == 0 or edge_index.numel() == 0 or K <= 0:
        return e
    row, col = edge_index[0], edge_index[1]
    deg = torch.zeros(num_nodes, device=e.device, dtype=e.dtype)
    deg.scatter_add_(0, col, torch.ones_like(col, dtype=e.dtype))
    deg = deg.clamp_min(1.0)
    inv_deg = 1.0 / deg  # (N,)
    out = e.clone()
    for _ in range(K):
        # message_j = (1/deg(j)) * out[j]  flowing along edge j->i (paper uses A from i to j on column-norm)
        # Per repo: SparseTensor(row=col, col=row, value=d_norm) @ e  ==  for each i: sum_{j: (j,i) in E} (1/deg(i)) * e[j]
        # We replicate via scatter_add by destination=col with source=inv_deg[col]*out[row].
        msg = inv_deg[col] * out[row]
        agg = torch.zeros_like(out)
        agg.scatter_add_(0, col, msg)
        out = alpha * out + (1.0 - alpha) * agg
    return out


def gnnsafe_defense(data, victim_model=None, threshold=None,
                    T=1.0, K=2, alpha=0.5):
    """GNNSafe (ICLR 2023) energy-based OOD node filter.

    Post-hoc: uses the victim model's own logits — no separate OOD head, no OOD
    training samples needed (cross-checked against official GraphOOD-GNNSafe repo
    + vault ref `GNNSafe-ICLR2023.md`).

    Pipeline (paper Eq. 3 + Eq. 8, verified):
      1. node_logits = victim conv stack + classifier (per-node, pre-pool)
      2. neg_energy = T * logsumexp(node_logits / T, dim=-1)            # = -E
      3. propagate neg_energy K steps with mixing alpha along D^-1 A    # Eq. 8
      4. flag node OOD iff neg_energy < threshold  (equiv. E > -threshold)
      5. drop flagged nodes + their incident edges; remap node ids; remap batch

    `threshold` is calibrated upstream as the 5th-percentile of clean-graph
    propagated neg_energies (so 95% of clean nodes pass — symmetric to the 95th-
    percentile cut on positive energy E). See calibrate_gnnsafe.py.

    Falls back to identity if `victim_model` or `threshold` is None (mirrors
    mahalanobis_defense convention).
    """
    if victim_model is None or threshold is None:
        return data
    x = data.x
    n = x.shape[0]
    if n == 0:
        return data

    with torch.no_grad():
        logits = _gnnsafe_node_logits(victim_model, data)        # (N, C)
        # neg_energy in code = T * logsumexp(logits/T, dim=-1)   (== -E in paper)
        neg_energy = T * torch.logsumexp(logits / T, dim=-1)     # (N,)
        if K > 0:
            neg_energy = _energy_propagation(neg_energy, data.edge_index, n,
                                             K=K, alpha=alpha)

    # OOD iff neg_energy < threshold  (i.e. E > -threshold)
    keep = neg_energy >= threshold
    if keep.sum() == 0:
        # Defense saturated — keep all nodes rather than empty the graph.
        return data

    new_x = x[keep]
    idx_map = torch.full((n,), -1, dtype=torch.long, device=x.device)
    idx_map[keep] = torch.arange(int(keep.sum()), device=x.device)
    ei = data.edge_index
    ek = (idx_map[ei[0]] >= 0) & (idx_map[ei[1]] >= 0)
    new_ei = torch.stack([idx_map[ei[0][ek]], idx_map[ei[1][ek]]], dim=0)
    batch = getattr(data, "batch", None)
    new_batch = batch[keep] if batch is not None else None
    return Data(x=new_x, edge_index=new_ei,
                y=getattr(data, "y", None), batch=new_batch)


def mahalanobis_defense(data, mean=None, cov_inv=None, threshold=None):
    """Remove nodes whose Mahalanobis distance to the train feature distribution exceeds
    `threshold` (e.g. a high train-set percentile of Mahalanobis distances).

    This is the DISTRIBUTION-aware analogue of NormFilter: it catches nodes that are
    off-manifold in *direction / inter-dim correlation*, even when their L2 norm is in
    range. A magnitude-only norm-cap injection (N4) keeps ||f|| small but can sit in a
    low-density direction -> caught here. A distribution-matched (diffusion) injection
    sits inside the data ellipsoid -> passes. Node removal + edge/batch remap mirror
    norm_filter_defense.
    """
    x = data.x
    if mean is None or cov_inv is None or threshold is None:
        # No stats -> identity (cannot define the data ellipsoid from a single batch reliably)
        return data
    mean = mean.to(x.device).view(1, -1)
    cov_inv = cov_inv.to(x.device)
    diff = x - mean                                  # (n, d)
    m = (diff @ cov_inv * diff).sum(dim=1)           # (n,) squared Mahalanobis distance
    keep = m <= threshold
    if keep.sum() == 0:
        return data
    new_x = x[keep]
    idx_map = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
    idx_map[keep] = torch.arange(int(keep.sum()), device=x.device)
    ei = data.edge_index
    ek = (idx_map[ei[0]] >= 0) & (idx_map[ei[1]] >= 0)
    new_ei = torch.stack([idx_map[ei[0][ek]], idx_map[ei[1][ek]]], dim=0)
    batch = getattr(data, "batch", None)
    new_batch = batch[keep] if batch is not None else None
    return Data(x=new_x, edge_index=new_ei,
                y=getattr(data, "y", None), batch=new_batch)


# ============================================================
# GDDM-style purification (arXiv 2501.11568) — see GDDM_NOTES.md
# ============================================================
#
# Pragmatic DiffPure-style instance of the GDDM paradigm:
#   - Train a GCN denoiser on clean graphs (see train_gddm.py).
#   - At test time: noise X' to a small t* (DiffPure), then reverse-denoise to t=0.
#   - A is NOT diffused (KATZE attacks node-injection; X is the channel that matters).
#
# Simplifications vs the paper (intentional, documented in GDDM_NOTES.md):
#   - Gaussian DDPM on X instead of Binomial diffusion on A
#   - No GSDR (adjacency intersection) — A is left as-is
#   - No NFCR (degree/feature-smoothness pruning) — clean-feature prior alone
#   - L2 noise prediction instead of VLB
#   - Single forward+reverse pass from partial t* (DiffPure), not full T-step reverse

import math
import os as _os


_GDDM_CACHE = {}  # key=ckpt_path -> (denoiser_module, schedule_dict)


def _gddm_sinusoid_t_emb(t, dim):
    device = t.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / max(half - 1, 1))
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class _GDDMDenoiser(nn.Module):
    """Mirror of train_gddm.GCNDenoiser. Re-declared here so defenses.py has no
    train-script import dependency at eval time."""
    def __init__(self, in_dim, hidden=64, t_emb_dim=64):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.t_emb_dim = t_emb_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_emb_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.in_proj = nn.Linear(in_dim, hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.out = nn.Linear(hidden, in_dim)

    def forward(self, x_t, edge_index, t_per_node):
        te = self.t_mlp(_gddm_sinusoid_t_emb(t_per_node, self.t_emb_dim))
        h = self.in_proj(x_t) + te
        h = F.relu(self.conv1(h, edge_index)) + te
        h = F.relu(self.conv2(h, edge_index)) + te
        h = F.relu(self.conv3(h, edge_index))
        return self.out(h)


def _gddm_load(ckpt_path, device):
    cache_key = (ckpt_path, str(device))
    if cache_key in _GDDM_CACHE:
        return _GDDM_CACHE[cache_key]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = _GDDMDenoiser(ckpt["in_dim"], hidden=ckpt.get("hidden", 64)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    T = ckpt["T"]
    betas = torch.linspace(ckpt["beta_1"], ckpt["beta_T"], T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    sched = {"T": T, "betas": betas, "alphas": alphas, "alpha_bar": alpha_bar}
    _GDDM_CACHE[cache_key] = (model, sched)
    return model, sched


@torch.no_grad()
def gddm_defense(data, ckpt_path=None, t_star=30):
    """DiffPure-style purification: noise X to t_star then reverse-denoise to 0.

    Args:
        data:       a torch_geometric.data.Data object.
        ckpt_path:  trained denoiser .pt path (from train_gddm.py). If None
                    or missing on disk, defense is a no-op (returns data).
        t_star:     partial-noise level in [1, T-1]. Lower = lighter purification.

    Returns:
        Data with purified .x (same shape); .edge_index, .y, .batch untouched.
    """
    if ckpt_path is None or not _os.path.exists(ckpt_path):
        return data
    device = data.x.device
    model, sched = _gddm_load(ckpt_path, device)
    T = sched["T"]
    t_star = int(max(1, min(T - 1, t_star)))

    x = data.x.float()
    if x.shape[0] == 0:
        return data
    t_vec = torch.full((x.shape[0],), t_star, dtype=torch.long, device=device)
    # Forward: q(x_{t*} | x_0)
    ab = sched["alpha_bar"][t_vec].unsqueeze(-1)
    eps0 = torch.randn_like(x)
    x_t = ab.sqrt() * x + (1 - ab).sqrt() * eps0

    # Reverse DDPM from t_star -> 0 (ε-prediction ancestral sampler)
    for t in range(t_star, 0, -1):
        t_now = torch.full((x.shape[0],), t, dtype=torch.long, device=device)
        eps_pred = model(x_t, data.edge_index, t_now)
        beta_t = sched["betas"][t]
        alpha_t = sched["alphas"][t]
        ab_t = sched["alpha_bar"][t]
        coef = (1 - alpha_t) / (1 - ab_t).clamp_min(1e-8).sqrt()
        mean = (x_t - coef * eps_pred) / alpha_t.sqrt()
        if t > 1:
            noise = torch.randn_like(x_t)
            x_t = mean + beta_t.sqrt() * noise
        else:
            x_t = mean

    return Data(x=x_t, edge_index=data.edge_index,
                y=getattr(data, "y", None),
                batch=getattr(data, "batch", None))


# ============================================================
# GDDM-A: paper-faithful A-channel (adjacency) variant
# ============================================================
#
# Binomial diffusion on the adjacency matrix A (forward) + per-edge Bernoulli
# denoiser conditioned on (x, A_t, t). At test time we follow a *noise-then-
# reverse* DiffPure-style protocol: forward-diffuse the (possibly perturbed)
# adjacency A' to a small t_star, then reverse-denoise to t=0. A GSDR-lite
# guard is applied at each reverse step: predicted_A <- predicted_A * A_t.
# This means the defense can only KEEP or DROP edges that exist in the noised
# graph, never SPAWN new ones — which is the spirit of the paper's GSDR mask
# (A' ⊙ A^t). For an injected node, its edges live in A' so they survive the
# initial noising, but each reverse step gets a chance to prune them.

_GDDM_A_CACHE = {}  # key=(ckpt_path, device) -> (model, sched, in_dim)


class _GDDMAEdgeDenoiser(nn.Module):
    """Mirror of train_gddm_a.EdgeDenoiser — duplicated so defenses.py has no
    train-script import dependency at eval time. Architecture must match
    bit-for-bit so checkpoints load."""
    def __init__(self, in_dim, hidden=64, t_emb_dim=64):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_emb_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden))
        self.in_proj = nn.Linear(in_dim, hidden)
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.lin3 = nn.Linear(hidden, hidden)
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
        A_hat_norm = self._norm_adj(A_t)
        n = x.shape[0]
        t_vec = torch.full((n,), t_scalar, dtype=torch.long, device=x.device)
        te = self.t_mlp(_gddm_sinusoid_t_emb(t_vec, self.t_emb_dim))
        h = self.in_proj(x.float()) + te
        h = self._gcn_layer(self.lin1, h, A_hat_norm) + te
        h = self._gcn_layer(self.lin2, h, A_hat_norm) + te
        h = self._gcn_layer(self.lin3, h, A_hat_norm)
        return h, te

    def edge_logits(self, h, te, pairs):
        u, v = pairs[:, 0], pairs[:, 1]
        hu, hv = h[u], h[v]
        teu = te[u]
        feat = torch.cat([hu, hv, hu * hv, teu], dim=-1)
        return self.edge_mlp(feat).squeeze(-1)


def _gddm_a_make_sched(T, b1, bT, device):
    betas = torch.linspace(b1, bT, T, device=device)
    one_minus_two_beta = 1.0 - 2.0 * betas
    cum = torch.cumprod(one_minus_two_beta, dim=0)
    cum_flip = 0.5 * (1.0 - cum)
    return {"T": T, "betas": betas, "cum_flip": cum_flip}


def _gddm_a_load(ckpt_path, device):
    cache_key = (ckpt_path, str(device))
    if cache_key in _GDDM_A_CACHE:
        return _GDDM_A_CACHE[cache_key]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = _GDDMAEdgeDenoiser(ckpt["in_dim"], hidden=ckpt.get("hidden", 64)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    sched = _gddm_a_make_sched(ckpt["T"], ckpt["beta_1"], ckpt["beta_T"], device)
    _GDDM_A_CACHE[cache_key] = (model, sched, ckpt["in_dim"])
    return model, sched, ckpt["in_dim"]


def _data_to_dense_adj(data):
    """Build symmetric binary dense A from edge_index. Returns (n, n) on data.x.device."""
    n = data.num_nodes
    device = data.x.device
    A = torch.zeros(n, n, device=device)
    ei = data.edge_index
    if ei.numel() > 0:
        A[ei[0], ei[1]] = 1.0
        A = ((A + A.t()) > 0).float()
    A.fill_diagonal_(0.0)
    return A


def _binomial_q_sample_dense(A0, t_idx, sched):
    """Forward diffuse a dense binary A0 to step t_idx (symmetric flips)."""
    n = A0.shape[0]
    p_flip = sched["cum_flip"][t_idx].item()
    triu = torch.triu(torch.ones(n, n, device=A0.device, dtype=torch.bool), diagonal=1)
    flips = (torch.rand(n, n, device=A0.device) < p_flip) & triu
    flips = flips | flips.t()
    A_t = (A0.bool() ^ flips).float()
    A_t.fill_diagonal_(0.0)
    return A_t


def _dense_to_edge_index(A):
    """Symmetric binary (n,n) -> edge_index (2, E) including both directions."""
    idx = (A > 0.5).nonzero(as_tuple=False)
    return idx.t().contiguous().long()


@torch.no_grad()
def gddm_a_defense(data, ckpt_path=None, t_star=30, edge_threshold=0.5,
                   chunk_pairs=20000):
    """A-channel GDDM purification: noise-then-reverse on the adjacency.

    Protocol:
      1. Build dense A' from data.edge_index.
      2. Forward-diffuse A' -> A_{t*} (each edge flipped indep. w.p. cum_flip[t*]).
      3. For t = t*, t*-1, ..., 1:
           a. Predict per-pair edge probability under the clean prior: p_hat.
           b. GSDR-lite guard: A_pred = (p_hat > thr) AND A_t   (intersect with noised A).
           c. Step down: A_{t-1} <- A_pred  (deterministic; the per-edge head IS the
              posterior estimate, so we don't re-noise).
      4. Return Data with .edge_index built from the final A; .x untouched.

    Args:
        data:           torch_geometric Data.
        ckpt_path:      .pt path from train_gddm_a.py. None or missing -> no-op.
        t_star:         partial-noise level in [1, T-1]. Default 30 (~15% noise w/ T=200).
        edge_threshold: probability cutoff for the per-edge Bernoulli head.
        chunk_pairs:    cap on pairs evaluated per chunk (memory safety on n=481 graphs).

    Returns:
        Data with purified .edge_index. .x, .y, .batch passed through unchanged.
    """
    if ckpt_path is None or not _os.path.exists(ckpt_path):
        return data
    device = data.x.device
    n = data.num_nodes
    if n < 2:
        return data

    model, sched, in_dim = _gddm_a_load(ckpt_path, device)
    if data.x.shape[1] != in_dim:
        # Feature-dim mismatch — denoiser trained on a different dataset; skip.
        return data
    T = sched["T"]
    t_star = int(max(1, min(T - 1, t_star)))

    A_prime = _data_to_dense_adj(data)        # the (possibly perturbed) input adjacency
    A_t = _binomial_q_sample_dense(A_prime, t_star, sched)

    # Pre-build the upper-triangle pair index once (we always evaluate the same pairs).
    triu_mask = torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)
    all_pairs = triu_mask.nonzero(as_tuple=False)        # (P, 2)

    def _predict_clean_adj(A_t_cur, t_scalar):
        """Run denoiser, GSDR-lite mask, return binary dense A_pred (symmetric)."""
        x = data.x.float()
        h, te = model.node_embeddings(x, A_t_cur, t_scalar)
        # Chunk over pairs to bound memory (per-edge MLP head is the hot path)
        P = all_pairs.shape[0]
        probs = torch.empty(P, device=device)
        for s in range(0, P, chunk_pairs):
            pl = all_pairs[s:s + chunk_pairs]
            logits = model.edge_logits(h, te, pl)
            probs[s:s + chunk_pairs] = torch.sigmoid(logits)
        A_pred = torch.zeros(n, n, device=device)
        keep = probs > edge_threshold
        kp = all_pairs[keep]
        A_pred[kp[:, 0], kp[:, 1]] = 1.0
        A_pred = A_pred + A_pred.t()
        # GSDR-lite: intersect with current noised graph — can drop, never add.
        A_pred = A_pred * A_t_cur
        return A_pred

    # Reverse loop. Use a small number of steps for speed; do not re-noise between
    # steps (deterministic descent toward clean). The number of "descent" steps is
    # bounded by t_star; for efficiency we walk in unit increments since each step
    # already produces a candidate clean A.
    for t in range(t_star, 0, -1):
        A_t = _predict_clean_adj(A_t, t)

    new_ei = _dense_to_edge_index(A_t)
    return Data(x=data.x, edge_index=new_ei,
                y=getattr(data, "y", None),
                batch=getattr(data, "batch", None))


def gddm_a_ckpt_path(dataset_name):
    """Path to the trained GDDM-A denoiser for a given dataset."""
    from pathlib import Path as _P
    return str(_P(__file__).parent / "gddm_a_checkpoints" / f"{dataset_name}.pt")


# ============================================================
# DefendedModel wrapper — drop-in replacement for victim
# ============================================================

class DefendedModel(nn.Module):
    """
    Wraps a victim model with a defense preprocessor.

    The attack's run_attack() sees this as a normal model — it queries via
    .forward(data), gets logits back. The defense is transparent.
    """
    def __init__(self, base_model, defense_fn, defense_name="custom", **defense_kwargs):
        super().__init__()
        self.base_model = base_model
        self.defense_fn = defense_fn
        self.defense_kwargs = defense_kwargs
        self.defense_name = defense_name
        # Inherit base_model's parameters so attack ZO loop works unchanged
        self.eval()

    def forward(self, data):
        defended = self.defense_fn(data, **self.defense_kwargs)
        # Defense may have changed num_nodes; ensure batch tensor is consistent
        if not hasattr(defended, "batch") or defended.batch is None:
            defended.batch = torch.zeros(defended.num_nodes,
                                         dtype=torch.long,
                                         device=defended.x.device)
        return self.base_model(defended)

    def train(self, mode=True):
        # Keep base model in eval mode; defense is non-parametric
        self.base_model.eval()
        return self


# ============================================================
# Defense registry — for easy iteration in eval scripts
# ============================================================

DEFENSE_REGISTRY = {
    "none":       (None, {}),
    "jaccard":    (jaccard_defense,     {"threshold": 0.01, "similarity": "cosine"}),
    "jaccard_strong": (jaccard_defense, {"threshold": 0.05, "similarity": "cosine"}),
    "svd":        (svd_defense,         {"rank": 10}),
    "svd_strong": (svd_defense,         {"rank": 5}),
    "normfilter": (norm_filter_defense, {"percentile": 95}),
    "normfilter_strong": (norm_filter_defense, {"percentile": 90}),
    # degfilter: STRUCTURE/degree-aware. Binds against full-edge single-node
    # injection (injected degree ~= N) that feature filters miss. Clean degree
    # mean/std are injected at wrap time (like mahalanobis stats); registry holds
    # the default k only. Standalone (no stats) -> within-graph mean + k*std.
    "degfilter": (degree_filter_defense, {"k": 3.0}),
    "degfilter_strong": (degree_filter_defense, {"k": 2.0}),
    # gnnsafe needs a `victim_model` + calibrated `threshold` injected at wrap
    # time (analogous to mahalanobis needing mean/cov_inv); registry holds the
    # default propagation hyperparams only.
    "gnnsafe":    (gnnsafe_defense,     {"T": 1.0, "K": 2, "alpha": 0.5}),
    # gddm needs a `ckpt_path` injected at wrap time (per-dataset trained denoiser);
    # registry holds the default partial-noise level only.
    "gddm":       (gddm_defense,        {"t_star": 30}),
    # gddm_a: A-channel (adjacency) variant. Also needs a `ckpt_path` injected at
    # wrap time. Default t_star=30 is ~15% edge noise with T=200.
    "gddm_a":     (gddm_a_defense,      {"t_star": 30}),
    # ============================================================
    # Renamed aliases (2026-05-27) — disambiguate from GDDM paper.
    # GDDM paper (arXiv 2501.11568) is adjacency-only diffusion against
    # edge-perturbation attacks (Nettack, GraD) on node classification.
    #
    #   featurepure        = NOT GDDM. DiffPure on features (X).
    #                        gddm_defense was a misnomer; keep alias for
    #                        backward compatibility, prefer new name in
    #                        new code / paper tables.
    #   gddm_adapted       = GDDM A-channel + DiffPure-style partial noise
    #                        + GSDR-lite. Paper-faithful on architecture,
    #                        adapted to graph-classification + node-injection
    #                        threat model. Use this name in paper §B.
    # ============================================================
    "featurepure":     (gddm_defense,    {"t_star": 20}),
    "gddm_adapted":    (gddm_a_defense,  {"t_star": 30}),
}


# --- Aliases for the renamed defenses (function-level) ---
# These are the canonical names going forward. The original gddm_defense /
# gddm_a_defense are kept for backward compatibility with running experiments.
featurepure_defense = gddm_defense          # NOT GDDM (paper diffuses A, not X)
gddm_adapted_defense = gddm_a_defense       # GDDM-style A-channel, adapted to injection setting


def gddm_ckpt_path(dataset_name):
    """Path to the trained GDDM denoiser for a given dataset (PROTEINS / ENZYMES)."""
    return str(Path(__file__).parent / "gddm_checkpoints" / f"{dataset_name}.pt")


# Aliases for ckpt path helpers (matching renamed defenses)
def featurepure_ckpt_path(dataset_name):
    """Path to the trained featurepure (X-channel DiffPure) denoiser."""
    return gddm_ckpt_path(dataset_name)


def gddm_adapted_ckpt_path(dataset_name):
    """Path to the trained gddm_adapted (A-channel) denoiser."""
    return gddm_a_ckpt_path(dataset_name)


# Re-import Path here for the helper above (kept local so we don't perturb other imports)
from pathlib import Path  # noqa: E402


def wrap_with_defense(model, defense_key):
    """Convenience: get a DefendedModel for a registry key."""
    if defense_key == "none":
        return model
    fn, kwargs = DEFENSE_REGISTRY[defense_key]
    return DefendedModel(model, fn, defense_name=defense_key, **kwargs)


# ============================================================
# Sanity check
# ============================================================

if __name__ == "__main__":
    import torch
    from torch_geometric.data import Data

    # Synthetic small graph
    x = torch.randn(10, 3)
    ei = torch.tensor([[0, 1, 1, 2, 3, 4, 5], [1, 0, 2, 1, 4, 3, 6]])
    data = Data(x=x, edge_index=ei, y=torch.tensor([0]))

    print("Original edges:", data.edge_index.shape[1])

    j = jaccard_defense(data, threshold=0.5)
    print(f"After Jaccard (thr=0.5): {j.edge_index.shape[1]} edges")

    s = svd_defense(data, rank=3)
    print(f"After SVD (rank=3): {s.edge_index.shape[1]} edges")

    # Add a high-norm injected node
    x_inj = torch.cat([x, torch.ones(1, 3) * 100], dim=0)
    data_inj = Data(x=x_inj, edge_index=ei, y=torch.tensor([0]))
    n = norm_filter_defense(data_inj, percentile=90)
    print(f"After NormFilter (pct=90): {n.num_nodes}/{data_inj.num_nodes} nodes kept")

    print("Defense registry keys:", list(DEFENSE_REGISTRY.keys()))
