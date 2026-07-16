#!/usr/bin/env python3
"""plot_degrade_curve.py — N34 headline figure: certified-accuracy collapse under the adaptive
label-preserving certificate-degradation attack. Clean vs adaptive(k=1,30e) vs adaptive(k=3,90e),
matched 30-graph set (cap10 x 3 folds, seed42, AGNNCert-E T=30)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BD = Path("/workspace/autoresearch-katze-n23/research/branch-diffusion")
RES = BD / "results"
k1 = json.loads((RES / "agnncert_degrade_fast_seed42_nb10_k1.json").read_text())
k3 = json.loads((RES / "agnncert_degrade_fast_seed42_nb10_k3.json").read_text())

# clean_M is per-graph and identical across runs (same graph set); use k3's
clean = [r["clean_M"] for r in k3["rows"]]
deg_k1 = [r["degraded_M"] for r in k1["rows"]]
deg_k3 = [r["degraded_M"] for r in k3["rows"]]
n = len(clean)
assert len(deg_k1) == n == len(deg_k3), f"size mismatch {len(deg_k1)},{n},{len(deg_k3)}"

ms = list(range(0, 16))
def curve(Ms):  # certified accuracy among the (correct) attackable set
    return [sum(1 for M in Ms if M >= m) / n for m in ms]
c_clean, c_k1, c_k3 = curve(clean), curve(deg_k1), curve(deg_k3)

fig, ax = plt.subplots(figsize=(7.2, 4.8))
ax.plot(ms, c_clean, "-o", color="#1f77b4", lw=2.2, ms=4, label="clean (AGNNCert published)")
ax.plot(ms, c_k1, "-s", color="#ff7f0e", lw=2.0, ms=4, label="adaptive degrade, budget 30 edges (k=1)")
ax.plot(ms, c_k3, "-^", color="#d62728", lw=2.2, ms=4, label="adaptive degrade, budget 90 edges (k=3)")
ax.axvline(15, color="gray", ls=":", lw=1.2)
ax.text(15, 0.5, " certified cap ⌊T/2⌋=15", rotation=90, va="center", ha="right", fontsize=8, color="gray")
ax.set_xlabel("certified radius m  (perturbations the certificate guarantees)")
ax.set_ylabel("certified accuracy  (correct & M ≥ m)")
ax.set_title("AGNNCert-E certified accuracy collapses under an adaptive\nlabel-preserving certificate-degradation attack (PROTEINS, seed42, n=%d)" % n)
ax.set_xlim(0, 15.5); ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
out = RES / "charts" / "n34_cert_degrade_curve.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)

print(f"n={n} graphs  budgets: k1={k1['budget_edges']}e  k3={k3['budget_edges']}e")
for label, c in [("clean", c_clean), ("k1(30e)", c_k1), ("k3(90e)", c_k3)]:
    print(f"  {label:9s} cert-acc  m=1:{c[1]:.2f}  m=3:{c[3]:.2f}  m=5:{c[5]:.2f}  m=10:{c[10]:.2f}  m=15:{c[15]:.2f}")
print(f"  clean M mean={sum(clean)/n:.2f}  ->  k1 {sum(deg_k1)/n:.2f}  k3 {sum(deg_k3)/n:.2f}")
print(f"  driven to M=0:  k1 {sum(1 for M in deg_k1 if M==0)}/{n}   k3 {sum(1 for M in deg_k3 if M==0)}/{n}")
print(f"-> {out}")
