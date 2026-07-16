# N23 (DiSGIA-NoParam) — Session Handoff

**Date**: 2026-05-19
**Branch**: `n23-disgia-noparam` (git worktree at `/workspace/autoresearch-katze-n23/`)
**Master location**: `/workspace/autoresearch-katze/` (branch `master`)

---

## 0. Three-Line Summary

We're prototyping **N23 DiSGIA-NoParam** — an AdvAD-style non-parametric diffusion graph injection attack — as a drop-in for the autoresearch-katze pipeline. New file: `attack_diffusion.py` (Path B, graph-classification, query-only, injection-only enforced). Smoke test (`quick_test_n23.py`) was almost passing; fixed schema mismatch but **not yet executed end-to-end**.

---

## 1. What N23 Is (research context)

| | |
|---|---|
| **Source idea** | AdvAD (Li et al., NeurIPS 2024, [arXiv:2503.09124](https://arxiv.org/abs/2503.09124)) — non-parametric diffusion attack on image classifiers |
| **N23 contribution** | Port to **graph classification** + **query-only black-box** + **injection-only (GIA)** threat model |
| **Why empty quadrant** | Image-domain AdvAD is white-box; graph-domain diffusion literature is defense (GDDM, DiffSP) or backdoor on diffusion model itself — no diffusion-based attack generator for GNN classifier |
| **Core trick** | Cast attack as discrete-time non-parametric diffusion: forward = noise injected features; reverse = ZO-guided DDIM denoising. **No diffusion network needed** (unlike DiGress route) |
| **Injection-only guarantee** | `construct_perturbed_graph()` in baseline `attack.py` is pure — `original_data` never mutated, only appended. N23 inherits this. Proposition 3.1 in [[directions-log N23]] formalizes |

**Full context**: `/workspace/obsidian-vault/projects/gzoo-katze/`:
- `directions-log.md` — N23 formal entry (lines ~28-110)
- `synthesis/path-b-optimization-survey.md` — 7-axis decomposition, 31 algorithm candidates
- `synthesis/diffusion-graph-adversarial-literature.md` — 50+ paper bibliography

---

## 2. Code Status

### Files in this worktree

| File | Status | Purpose |
|---|---|---|
| `attack.py` | unchanged from master | Baseline KATZE (generator + Adam + ZO) — for A/B comparison |
| `attack_diffusion.py` | **NEW** (written this session) | N23-NoParam implementation — AdvAD-style DDIM + ZO |
| `quick_test_n23.py` | **NEW** (smoke test) | One-fold 5-graph sanity check |
| `data/`, `models_saved/` | **symlinked** to master | Avoid re-running prepare.py |
| `program.md` | M (modified, uncommitted from master) | Carried over; may need update for N23 framing |

### attack_diffusion.py architecture

```
make_alpha_schedule()       — cosine/linear noise schedule, T+1 length
predict_x0_from_xt()        — DDIM x0 prediction
ddim_step()                 — deterministic reverse step
zo_score_cge()              — CGE on x0 (mirrors baseline estimate_gradient_cge)
zo_score_rgf()              — RGF alternative
_attack_single_diffusion()  — main reverse loop (Eq. 8 AdvAD adapted)
_estimate_score()           — ZO dispatcher
run_attack()                — entry point, same signature as attack.run_attack()
```

**Key equation (AdvAD AMG, adapted for CW minimization)**:
```
eps_hat_t = eps_0 - guidance_scale * sqrt(1 - alpha_bar_t) * ZO_grad(CW, x0)
```

**Reuses from attack.py**: `construct_perturbed_graph`, `batch_loss`, `get_prediction`, `select_targets_topk`, `select_targets_spectral` (via `from attack import ...`).

### CONFIG block

```python
CONFIG = {
    # Diffusion
    "T_steps": 50,                # reverse steps (baseline uses attack_epochs=50)
    "schedule": "cosine",
    "alpha_min": 1e-3, "alpha_max": 0.999,

    # ZO
    "sigma": 5e-3, "grad_method": "cge", "rgf_samples": 20,

    # AdvAD-specific
    "guidance_scale": 1.0,
    "use_AdvAD_X": False,         # AdvAD-X variant (DGI skip)
    "dgi_skip_threshold": 0.01,

    # Inherited
    "feat_scale": "auto_x2", "loss_type": "cw", "kappa": -0.1,
    "edge_strategy": "full", "node_budget": 1, "spectral_top_k_eig": 10,
    "n_restarts": 3,
}
```

---

## 3. What's Done

- [x] git worktree at `/workspace/autoresearch-katze-n23/` on branch `n23-disgia-noparam`
- [x] `attack_diffusion.py` written with full docstrings + injection-only audit comments
- [x] Module import + alpha schedule sanity test PASSED (`python attack_diffusion.py`)
- [x] `quick_test_n23.py` written
- [x] `data/`, `models_saved/` symlinked to master (avoids re-running prepare.py)
- [x] Smoke test schema bug fixed (`fold_info["model_path"]`, `meta["num_features"]`)
- [x] N23 entry in vault `directions-log.md` with Proposition 3.1 + H23.1–H23.4 + R1–R4 risk register

---

## 4. What's Pending (resume here)

### Immediate (next 30 min)

1. **Run smoke test**:
   ```bash
   cd /workspace/autoresearch-katze-n23
   micromamba run -n graph_adversarial python quick_test_n23.py
   ```
   Expected: ASR on 5 graphs in <60s. If crashes, debug.

2. **If smoke passes — run full evaluation**:
   ```bash
   # Option A (clean): swap files temporarily
   cp attack.py attack_baseline.py.bak
   cp attack_diffusion.py attack.py
   micromamba run -n graph_adversarial python run.py --quick
   # restore
   cp attack_baseline.py.bak attack.py && rm attack_baseline.py.bak
   ```

3. **Compare to baseline**: baseline ASR is ~94.85% on PROTEINS GCN (5-fold). H23.1 says N23 should be **≥3pp better** or it's killed.

### Phase 1 hypotheses to test

| ID | Hypothesis | How to test |
|---|---|---|
| H23.1 | N23 ≥3pp uplift over KATZE-clique on PROTEINS GCN | A/B run via file swap above |
| H23.2 | Annealed noise > constant noise (vs fixed-lr CGE) | Ablation: set `schedule="linear"` with flat alpha vs cosine |
| H23.3 | Injection-only constraint costs <2pp ASR | Already enforced; measure vs hypothetical naive (not needed for Phase 1 — current code IS injection-only) |
| H23.4 | DiGress prior (Phase 2) gives further uplift | Out of Phase 1 scope |

### Known risks

- **R1**: ZO variance — current N23 uses CGE same as baseline; if ASR is low, try RGF or increase guidance_scale
- **R2**: Annealed schedule might collapse to fixed-lr — verify by inspecting x0_pred trajectory across t
- **R3**: T_steps=50 matches baseline attack_epochs=50 — but baseline has multi-restart (5 configs) so effective query budget is 5× higher. May need to bump T_steps or n_restarts.

### Decision gate at Phase 1 end

- **If ASR ≥ baseline + 3pp**: write Proposition 3.1 proof in `theory/` folder, draft §3 of paper
- **If 0 ≤ ASR < baseline + 3pp**: investigate which mechanism failed (annealed schedule? guidance_scale?); try AdvAD-X variant
- **If ASR << baseline**: kill N23-NoParam, pivot to N23-Guided (DiGress prior; needs ~6h GPU to train DiGress on PROTEINS first)

---

## 5. Quick Reference

### Filesystem map

```
/workspace/autoresearch-katze/              ← master, KATZE baseline (don't touch)
  └─ uncommitted: M program.md (carried over to n23 worktree)

/workspace/autoresearch-katze-n23/          ← N23 worktree (work here)
  ├─ attack.py                              ← unchanged baseline (reference)
  ├─ attack_diffusion.py                    ← N23 implementation
  ├─ quick_test_n23.py                      ← smoke test (5 graphs, 1 fold)
  ├─ data/, models_saved/                   ← symlinks to master
  └─ HANDOFF_N23.md                         ← this file

/workspace/obsidian-vault/projects/gzoo-katze/
  ├─ directions-log.md                      ← N23 formal entry
  └─ synthesis/
     ├─ path-b-optimization-survey.md       ← N23–N31 ranked
     └─ diffusion-graph-adversarial-literature.md   ← 50+ paper bibliography
```

### Key references to re-read first

1. AdvAD paper: arXiv:2503.09124, code: github.com/XianguiKang/AdvAD (NeurIPS 2024)
   - Eq. 4 (forward), Eq. 8 (AMG reverse) — the core math we ported
2. `attack_diffusion.py` lines 145–215 — `_attack_single_diffusion()` the main loop
3. Vault `directions-log.md` N23 entry — full hypothesis table + Proposition 3.1

### Environment

```bash
# Always use micromamba env:
micromamba run -n graph_adversarial python <script>
```

### Git status (at handoff time)

```
Branch:    n23-disgia-noparam
Ahead of:  master by 0 commits (nothing committed yet on this branch)
Untracked: attack_diffusion.py, quick_test_n23.py, HANDOFF_N23.md
Modified:  program.md (inherited from master uncommitted state)
```

**Suggested first commit when smoke passes**:
```bash
git add attack_diffusion.py quick_test_n23.py HANDOFF_N23.md
git commit -m "feat(n23): AdvAD-style non-parametric diffusion attack"
```

---

## 6. Open Questions for Next Session

1. **Should `program.md` be updated** to describe N23 as the research goal, so `pipeline.py --max-iterations` works on the diffusion variant rather than baseline?
2. **What's the right baseline number to beat**? The latest `best.json` shows 94.85% with multi-restart. With T_steps=50 + n_restarts=3, N23 has 30% fewer "epochs" than baseline. Apples-to-apples = match query budget.
3. **A_sv / A_ss extension**: current N23 only diffuses F_s (edges are fixed `full` strategy, A_ss N/A since node_budget=1). Phase 2 should add inter-injected diffusion if node_budget > 1.
4. **Should we commit before testing**? Argument for: provenance. Argument against: if smoke fails we'd want to amend. → My call: write smoke result first, then commit working state.

---

End of handoff. Resume from §4 step 1.
