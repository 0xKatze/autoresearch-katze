#!/usr/bin/env python3
"""
flow_net.py — N28 variant B core: trained conditional flow-matching velocity field.

This module factors the velocity network + the flow-matching training loss out of
attack_flow_b.py so that:
  * train_flow.py can fit v_theta OFFLINE on benign node features and persist it, and
  * attack_flow_b.py can LOAD that trained field (or, for convenience, train in-process)
    without duplicating the network / loss definitions.

Everything here is BLACK-BOX SAFE by construction: the velocity field is a function of
(clean node feature, flow time) only. It never sees the victim model, labels, or any
attack-time query. The adversarial signal (ZO-CW drift) is injected at attack time in
attack_flow_b.py, NOT here.

Formulation — conditional / rectified flow matching (OT-CFM linear interpolant):
    base    x0 ~ N(0, I)
    data    x1 ~ p_clean        (benign node features)
    path    x_t = (1 - t) * x0 + t * x1 ,   t ~ U[0, 1]
    target  u   = x1 - x0                    (constant conditional velocity)
    loss    L   = E_{x0, x1, t} || v_theta(x_t, t) - u ||^2

For a Gaussian p_clean this regresses toward the analytic straight-line velocity
(variant A); for a genuinely non-Gaussian / multimodal manifold (e.g. continuous-attr
ENZYMES, 21-dim) v_theta can learn structure the analytic Gaussian prior cannot — that
is the whole point of variant B.
"""

import math
import os

import torch
import torch.nn as nn


# ============================================================
# Time conditioning + velocity network
# ============================================================

class FourierTime(nn.Module):
    """Random-frequency Fourier features for the scalar flow time t in [0, 1].

    Frequencies are fixed (non-trained) — standard, robust time conditioning for a
    low-dim regression field. Seeded so a saved net reloads bit-identically.
    """

    def __init__(self, dim, seed=0):
        super().__init__()
        assert dim % 2 == 0, "time_dim must be even"
        g = torch.Generator().manual_seed(seed)
        self.register_buffer(
            "freqs", torch.randn(dim // 2, generator=g) * 2.0 * math.pi)

    def forward(self, t):                      # t: (B, 1)
        proj = t * self.freqs.view(1, -1)      # (B, dim/2)
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


class VelocityField(nn.Module):
    """MLP velocity field v_theta(x, t) -> R^d, operating in raw feature space."""

    def __init__(self, in_dim, hidden=128, n_layers=3, time_dim=32, seed=0):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.n_layers = n_layers
        self.time_dim = time_dim
        self.seed = seed
        self.time = FourierTime(time_dim, seed=seed)
        dims = [in_dim + time_dim] + [hidden] * n_layers
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.SiLU()]
        layers += [nn.Linear(hidden, in_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        if t.dim() == 1:
            t = t.view(-1, 1)
        if t.shape[0] == 1 and x.shape[0] > 1:
            t = t.expand(x.shape[0], 1)
        te = self.time(t)
        return self.net(torch.cat([x, te], dim=-1))

    def arch(self):
        """Architecture hyperparameters needed to rebuild this net from a checkpoint."""
        return {
            "in_dim": self.in_dim, "hidden": self.hidden,
            "n_layers": self.n_layers, "time_dim": self.time_dim, "seed": self.seed,
        }


# ============================================================
# Flow-matching conditional loss (rectified / OT-CFM linear interpolant)
# ============================================================

def fm_conditional_loss(net, x1, generator=None, device=None):
    """Single-minibatch conditional flow-matching loss on a batch of data rows x1.

    x1       : (B, d) benign feature rows (the conditioning data sample).
    generator: optional torch.Generator (CPU) for reproducible (x0, t) draws.
    Returns a scalar loss tensor (with grad) — caller does opt.zero_grad/step.
    """
    device = device or x1.device
    bs, d = x1.shape
    x0 = torch.randn(bs, d, generator=generator).to(device)
    t = torch.rand(bs, 1, generator=generator).to(device)
    x_t = (1.0 - t) * x0 + t * x1
    u = x1 - x0                                # constant conditional velocity
    v = net(x_t, t)
    return ((v - u) ** 2).mean()


@torch.no_grad()
def fm_eval_loss(net, X, n=2000, seed=777, device=None):
    """Held-out-style FM loss estimate over n random rows of X (no grad).

    Used by trainers / smoke tests to report a single comparable loss number.
    """
    device = device or X.device
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, X.size(0), (n,), generator=gen)
    x1 = X[idx].to(device)
    x0 = torch.randn(n, X.size(1), generator=gen).to(device)
    t = torch.rand(n, 1, generator=gen).to(device)
    v = net((1.0 - t) * x0 + t * x1, t)
    return ((v - (x1 - x0)) ** 2).mean().item()


# ============================================================
# Training loop on a feature matrix (shared by attack_flow_b + train_flow)
# ============================================================

def train_field_on_features(X, device, hidden=128, n_layers=3, time_dim=32,
                            train_steps=400, batch=256, lr=1e-3, seed=0,
                            log_every=0):
    """Fit a VelocityField on the feature matrix X via the FM loss. BLACK-BOX SAFE.

    X          : (N, d) benign node features (already normalized as appropriate).
    log_every  : if > 0, append (step, loss) to the returned history every log_every
                 steps (also at step 0 and the final step).
    Returns (net, history) where history is a list of (step, loss) tuples (possibly
    empty if log_every == 0). The returned net is in eval mode with grads disabled.
    """
    X = X.to(device).float()
    in_dim = X.size(1)
    n_rows = X.size(0)
    net = VelocityField(in_dim, hidden, n_layers, time_dim, seed=seed).to(device)
    gen = torch.Generator(device="cpu").manual_seed(seed + 12345)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    bs = min(batch, n_rows)
    history = []
    net.train()
    for step in range(train_steps):
        idx = torch.randint(0, n_rows, (bs,), generator=gen)
        loss = fm_conditional_loss(net, X[idx], generator=gen, device=device)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and (step % log_every == 0 or step == train_steps - 1):
            history.append((step, float(loss.item())))
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, history


# ============================================================
# Checkpoint I/O — self-describing (arch stored so the loader rebuilds the net)
# ============================================================

def save_checkpoint(net, path, extra=None):
    """Persist a trained VelocityField with its architecture + optional metadata."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "format": "flow_net.v1",
        "arch": net.arch(),
        "in_dim": net.in_dim,                  # kept top-level for back-compat checks
        "state_dict": net.state_dict(),
        "meta": dict(extra or {}),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path, device="cpu", expect_in_dim=None):
    """Rebuild a VelocityField from a checkpoint. Returns (net, meta) or raises.

    expect_in_dim: if given, raises ValueError on a feature-dim mismatch (the most
    common footgun — a field trained on one dataset loaded for another).
    """
    state = torch.load(path, map_location=device)
    arch = state.get("arch")
    if arch is None:                           # tolerate the older flat format
        arch = {"in_dim": state["in_dim"], "hidden": 128, "n_layers": 3,
                "time_dim": 32, "seed": 0}
    if expect_in_dim is not None and arch["in_dim"] != expect_in_dim:
        raise ValueError(
            f"checkpoint in_dim={arch['in_dim']} != expected {expect_in_dim} "
            f"({path}) — wrong dataset/feature space for this field")
    net = VelocityField(**arch).to(device)
    net.load_state_dict(state["state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, state.get("meta", {})


if __name__ == "__main__":
    # tiny self-test: train on synthetic data, loss should drop, ckpt round-trips
    torch.manual_seed(0)
    X = torch.randn(500, 5) @ torch.randn(5, 5) + 3.0     # non-isotropic cloud
    l0 = fm_eval_loss(VelocityField(5), X)
    net, hist = train_field_on_features(X, "cpu", train_steps=300, log_every=50)
    l1 = fm_eval_loss(net, X)
    print(f"FM loss: init={l0:.4f} -> fit={l1:.4f} ({'DROP' if l1 < l0 else 'NO DROP'})")
    print(f"history: {[(s, round(v, 4)) for s, v in hist]}")
    p = save_checkpoint(net, "/tmp/_flownet_selftest.pt", extra={"k": "v"})
    net2, meta = load_checkpoint(p, expect_in_dim=5)
    same = torch.allclose(net2(X[:4], torch.rand(4, 1)), net(X[:4], torch.rand(4, 1)))
    # (different t draws -> not equal; just check it runs + meta survives)
    print(f"reload OK; meta={meta}; out shape {tuple(net2(X[:4], torch.rand(4,1)).shape)}")
