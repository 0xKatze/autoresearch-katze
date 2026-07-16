#!/usr/bin/env python3
"""Render the Diffusion-KATZE attack pipeline as a standalone flow diagram.
Output: diffusion_katze_pipeline.png in the current working directory."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patheffects import withSimplePatchShadow

NAVY="#21295C"; TEAL="#1C7293"; BLUE="#065A82"; CRIMSON="#E63946"
GREEN="#2A9D8F"; GOLD="#E9C46A"; LIGHT="#F4F6FB"; INK="#1A1F3D"; GREY="#8D99AE"; WHITE="#FFFFFF"

fig, ax = plt.subplots(figsize=(13, 9), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
fig.patch.set_facecolor(LIGHT); ax.set_facecolor(LIGHT)

def box(cx, cy, w, h, title, sub, fill, fg=WHITE, subfg=None):
    x, y = cx - w/2, cy - h/2
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=1.4",
                       fc=fill, ec="none", zorder=3)
    p.set_path_effects([withSimplePatchShadow(offset=(2.2, -2.2), alpha=0.22)])
    ax.add_patch(p)
    ax.text(cx, cy + (h*0.16 if sub else 0), title, ha="center", va="center",
            color=fg, fontsize=12.5, fontweight="bold", zorder=4)
    if sub:
        ax.text(cx, cy - h*0.24, sub, ha="center", va="center",
                color=subfg or fg, fontsize=9.3, zorder=4)
    return dict(cx=cx, cy=cy, top=(cx, cy+h/2), bot=(cx, cy-h/2),
                left=(cx-w/2, cy), right=(cx+w/2, cy))

def arrow(p1, p2, color=GREY, lw=2.3, rad=0.0, label=None, lx=0, ly=0, italic=True):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=17,
                 color=color, lw=lw, connectionstyle=f"arc3,rad={rad}", zorder=2))
    if label:
        ax.text((p1[0]+p2[0])/2 + lx, (p1[1]+p2[1])/2 + ly, label, color=color,
                fontsize=9, ha="center", va="center", zorder=5,
                fontstyle="italic" if italic else "normal", fontweight="bold")

# Title
ax.text(2, 97.5, "Diffusion-KATZE — Attack Pipeline", fontsize=20, fontweight="bold",
        color=INK, ha="left", va="top")
ax.text(2, 93.2, "a cascade of fallbacks: each tier fires only when the previous one fails",
        fontsize=11.5, fontstyle="italic", color="#5A5F75", ha="left", va="top")

SX = 28  # spine x
# --- setup lane ---
b1 = box(SX, 86, 30, 7.4, "Input graph G", "correctly classified", BLUE)
b2 = box(SX, 75.5, 30, 7.4, "Select 1 node + edge targets", "topk-degree  /  full", TEAL)
b3 = box(SX, 65, 30, 7.4, "Generator Gθ  +  ZO-CGE gradient", "query-only ∇ · feat_scale = 2√N", NAVY)
# --- fallback ladder ---
c1 = box(SX, 52, 33, 8.2, "①  5-restart Adam baseline", "the cheap solver", NAVY)
c2 = box(SX, 39, 33, 8.2, "②  Diffusion restart  (DDIM on x)", "feature retry from best x₀", BLUE)
c3 = box(SX, 26, 33, 8.2, "③  Joint X+A diffusion  +  k_sweep", "edges enter the diffusion state", CRIMSON)
c4 = box(SX, 13, 33, 8.2, "④  Per-node escalation  (m = 2)", "independent edge masks", GOLD, fg=INK, subfg="#6B5A22")
# --- success sink ---
g  = box(80, 32.5, 28, 12, "Perturbed graph G′", "label flipped ✓", GREEN)

# setup arrows (grey)
arrow(b1["bot"], b2["top"]); arrow(b2["bot"], b3["top"])
arrow(b3["bot"], c1["top"])
# fallback "fail" arrows (crimson, down)
for a, b in [(c1, c2), (c2, c3), (c3, c4)]:
    arrow(a["bot"], b["top"], color=CRIMSON, label="fail", lx=4.0, ly=0)
# success arrows (green, right -> G')
arrow(c1["right"], (g["cx"]-14, g["cy"]+4.0), color=GREEN, rad=-0.18, label="✓ ~95%", lx=2, ly=2.4, italic=False)
arrow(c2["right"], (g["cx"]-14, g["cy"]+1.5), color=GREEN, rad=-0.10, label="✓", lx=1.5, ly=1.3, italic=False)
arrow(c3["right"], (g["cx"]-14, g["cy"]-1.5), color=GREEN, rad=0.10, label="✓", lx=1.5, ly=-1.3, italic=False)
arrow(c4["right"], (g["cx"]-14, g["cy"]-4.0), color=GREEN, rad=0.18, label="✓", lx=1.5, ly=-2.4, italic=False)

# lane brackets / labels
ax.text(7.5, 75.5, "SETUP", rotation=90, ha="center", va="center", fontsize=10.5,
        fontweight="bold", color=GREY, alpha=0.9)
ax.text(7.5, 32.5, "FALLBACK  LADDER", rotation=90, ha="center", va="center", fontsize=10.5,
        fontweight="bold", color=CRIMSON, alpha=0.85)
ax.plot([9.2, 9.2], [61.8, 89.2], color=GREY, lw=1.4, alpha=0.5)
ax.plot([9.2, 9.2], [8.9, 56.1], color=CRIMSON, lw=1.4, alpha=0.45)

# footnote
ax.text(2, 3.2, "Black-box · query-only · injection-only · node_budget = 1.  "
        "Core idea: edges (A) are part of the diffusion state, not just features (X).",
        fontsize=10, fontstyle="italic", color="#5A5F75", ha="left", va="bottom")

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
out = "diffusion_katze_pipeline.png"
plt.savefig(out, facecolor=LIGHT, bbox_inches="tight", pad_inches=0.15)
print("WROTE", out)
