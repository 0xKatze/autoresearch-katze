#!/usr/bin/env python3
"""
train_flow.py — OFFLINE trainer for the N28 variant B conditional flow-matching
velocity field v_theta (used by attack_flow_b.py).

THREAT MODEL (must hold): this trainer fits v_theta on BENIGN node features only.
It never loads, queries, or even imports the victim model — data-distribution-known,
NO victim access. The adversarial (ZO-CW) signal enters only at attack time, inside
attack_flow_b.py. The output is a self-describing checkpoint (flow_net.save_checkpoint)
that attack_flow_b.py loads via CONFIG["ckpt_path"].

Why continuous features (default ENZYMES, use_node_attr=True, z-scored, 21-dim):
PROTEINS is 3-dim binary one-hot, so a learned velocity collapses toward the analytic
Gaussian prior (variant A) — there is no non-Gaussian manifold to learn (design-spec
risk #1). The continuous-attr testbed (prepare_cont.py / models_cont/) gives a feature
cloud with genuine spread + inter-dim correlation where a TRAINED field can express
structure the Gaussian prior cannot. Use --dataset PROTEINS only to demonstrate the
collapse caveat explicitly.

Usage:
  micromamba run -n graph_adversarial python train_flow.py \
      --dataset ENZYMES --continuous \
      --train_steps 2000 --hidden 128 --layers 3 \
      --out models_cont/flow_ENZYMES.pt

  # binary PROTEINS (collapse-caveat demo):
  micromamba run -n graph_adversarial python train_flow.py \
      --dataset PROTEINS --train_steps 1000 --out flow_ckpts/flow_PROTEINS.pt
"""

import argparse
import os
import sys

# Force CPU by default before importing prepare (its module-global DEVICE binds at
# import); this trainer is light and sidesteps the env's CUDA segfault on this GPU.
if os.environ.get("FLOW_TRAIN_GPU", "0") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from flow_net import train_field_on_features, fm_eval_loss, save_checkpoint


def _load_benign_features(dataset, continuous, device):
    """Return (X, meta) where X is (N, d) BENIGN node features. No victim, no labels.

    continuous=True -> prepare_cont.load_normalized_graphs (use_node_attr, z-scored).
    continuous=False -> raw TUDataset node features (e.g. PROTEINS one-hot).
    """
    if continuous:
        from prepare_cont import load_normalized_graphs
        graphs, _labels, mean, std, n_classes = load_normalized_graphs(dataset)
        X = torch.cat([g.x for g in graphs], dim=0).to(device).float()
        meta = {
            "dataset": dataset, "continuous": True, "zscore": True,
            "use_node_attr": True, "num_classes": int(n_classes),
            "feat_mean": mean.squeeze(0).tolist(), "feat_std": std.squeeze(0).tolist(),
            "n_graphs": len(graphs), "n_rows": int(X.size(0)),
        }
        return X, meta
    # binary / one-hot path
    from torch_geometric.datasets import TUDataset
    from prepare import DATA_DIR
    ds = TUDataset(root=str(DATA_DIR / dataset), name=dataset)
    X = torch.cat([ds[i].x for i in range(len(ds))], dim=0).to(device).float()
    meta = {
        "dataset": dataset, "continuous": False, "zscore": False,
        "use_node_attr": False, "n_graphs": len(ds), "n_rows": int(X.size(0)),
    }
    return X, meta


def main():
    ap = argparse.ArgumentParser(description="Offline FM velocity trainer (variant B)")
    ap.add_argument("--dataset", default="ENZYMES")
    ap.add_argument("--continuous", action="store_true",
                    help="load use_node_attr z-scored features (recommended)")
    ap.add_argument("--train_steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--time_dim", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=0,
                    help="0 -> auto (~12 log points)")
    ap.add_argument("--out", default=None, help="checkpoint path (auto if unset)")
    args = ap.parse_args()

    device = "cuda" if (os.environ.get("FLOW_TRAIN_GPU") == "1"
                        and torch.cuda.is_available()) else "cpu"

    X, meta = _load_benign_features(args.dataset, args.continuous, device)
    in_dim = X.size(1)
    log_every = args.log_every or max(1, args.train_steps // 12)
    out = args.out or os.path.join(
        _HERE, "flow_ckpts",
        f"flow_{args.dataset}{'_cont' if args.continuous else ''}.pt")

    print(f"=== train_flow {args.dataset} | continuous={args.continuous} "
          f"feat_dim={in_dim} rows={X.size(0)} device={device} ===")
    print(f"net: {args.layers}x{args.hidden} (time_dim={args.time_dim}) | "
          f"steps={args.train_steps} batch={args.batch} lr={args.lr} seed={args.seed}")
    print("THREAT MODEL: benign features only, NO victim access.")

    l_init = fm_eval_loss(VelocityField_init(in_dim, args, device), X, device=device)

    net, history = train_field_on_features(
        X, device, hidden=args.hidden, n_layers=args.layers, time_dim=args.time_dim,
        train_steps=args.train_steps, batch=args.batch, lr=args.lr, seed=args.seed,
        log_every=log_every)

    l_fit = fm_eval_loss(net, X, device=device)
    print("--- FM training loss curve (step: loss) ---")
    for step, loss in history:
        print(f"  step {step:5d}: {loss:.5f}")
    print(f"eval FM loss: init={l_init:.5f} -> fit={l_fit:.5f} "
          f"({'DROPPED' if l_fit < l_init else 'NO DROP'})")

    meta.update({
        "arch": net.arch(), "train_steps": args.train_steps, "lr": args.lr,
        "seed": args.seed, "fm_loss_init": l_init, "fm_loss_fit": l_fit,
    })
    path = save_checkpoint(net, out, extra=meta)
    print(f"Saved trained velocity field -> {path}")
    print("Load in attack_flow_b via: CONFIG['ckpt_path'] = '%s'" % path)


def VelocityField_init(in_dim, args, device):
    """Fresh (untrained) net for the init-loss baseline."""
    from flow_net import VelocityField
    return VelocityField(in_dim, args.hidden, args.layers, args.time_dim,
                         seed=args.seed).to(device)


if __name__ == "__main__":
    main()
