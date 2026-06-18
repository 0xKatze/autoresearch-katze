# Critic punchlist round-2: H3 (GNNSafe in battery) + H4 (per-attack queries) + M3 (200-epoch victims)

Date: 2026-06-18 · branch `fix/critic-h3h4m3` (child of `fix/critic-punchlist`)
Scope: defenses-wiring / driver / victim-gen ONLY. OM2C / FM attack code untouched.

## H3 — GNNSafe added to the floor battery as a 5th (EFFECTIVE) defense

`eval_pg_floor_battery5_gpu.py`:
- `DEFENSES = F.DEFENSES + ["gnnsafe"]`, `EFFECTIVE = F.EFFECTIVE + ["gnnsafe"]`.
- New `calibrate_gnnsafe_threshold(model, ref_graphs)`: 5th-percentile of the
  **propagated clean-node neg_energy** over the held-out **TRAINING fold** graphs
  (T=1, K=2, α=0.5 — verbatim with `calibrate_gnnsafe.py` / official repo). Threshold
  never sees the attacked test graphs → same C3 train-fold principle as the
  normfilter/mahalanobis/degfilter fixes. Calibrated once per fold.
- New `build_defended_ext(model, dn, ref_graphs, gnnsafe_thr)`: handles `gnnsafe`
  (wraps with `DefendedModel(... gnnsafe_defense, victim_model=model, threshold=thr ...)`),
  delegates the original 4 defenses to the unmodified `F.build_defended`. The two
  `F.build_defended(...)` call sites in the loop now route through it.
- GNNSafe's internal per-node energy forward passes use the raw base model, so they
  are NOT counted as attacker queries (defense-side, correct).

### Smoke (PROTEINS GCN, 200-epoch victims, seed 42, 3 folds × 10 graphs = 30)
`results/floor_battery5_gpu_PROTEINS_full_20260618_025525.json`

| attack | ASR none | ASR gnnsafe | floor |
|---|---|---|---|
| OM2C-nb10 | 66.7 | 80.0 | 63.3 |
| FM-nb10-eb1-lownorm | 73.3 | 80.0 | 16.7 |
| SGDI-nb10-full | 100.0 | 100.0 | 0.0 |
| FM-nb10-full | 100.0 | 100.0 | 0.0 |
| FM-nb10-eb1-highnorm | 100.0 | 100.0 | 0.0 |

- **Engages / not a no-op**: clean acc 0.7024 → 0.6935 (−0.89pp, preserved), i.e. it
  actively drops the bottom-5% clean-energy nodes. Standalone `calibrate_gnnsafe.py`
  independently flags 20–47% of high-magnitude random OOD nodes at <1pp clean cost.
  Driver ran error-free (errored_cells=0).
- **Honest negative**: GNNSafe does **NOT** reduce ASR for any of the 5 battery attacks
  (slightly raises OM2C / FM-lownorm by node-dropping noise). Mechanism (direct probe,
  fold 0): threshold neg_energy = 0.661, drop iff below.
  - high-norm injected nodes (fs≈2√N): neg_energy ≈ **17.5** → ultra-confident → 0 dropped.
  - on-manifold injected nodes (mean feat, fs=1.0): neg_energy ≈ **0.95 > 0.661** → 0 dropped.
  Energy-OOD catches LOW-confidence OOD; injection attacks here are either on-manifold or
  high-magnitude — both give HIGH neg_energy and evade the low-energy cut. GNNSafe therefore
  never sets the floor, so the worst-case-defense floor is unchanged by adding it. This is a
  legitimate floor-study finding (another defense axis that fails against the candidate).

## H4 — per-attack total query count reported alongside floor

`eval_pg_floor_battery5_gpu.py`:
- `seed_floor_summary[a]` now carries `total_queries` (+ `queries_by_defense`).
- Seed-level floor print table gained a `totQ` column.
  (The pre-existing `total_queries_by_attack` / `queries_by_attack_defense` JSON fields
  remain.) Example from the smoke: OM2C-nb10=596358, FM-nb10-eb1-lownorm=103901,
  SGDI-nb10-full=823080, FM-nb10-full=110070, FM-nb10-eb1-highnorm=155870.

## M3 — victims retrained at 200 epochs (project standard, was 100)

`gen_victims_pg.py`:
- `--train-epochs` (sets `prepare.TRAIN_EPOCHS`) and `--force` (backs up + removes stale
  checkpoints to `pg_floor_seed{S}/_bak_epoch100/` so `train_victim`'s exists-guard does
  not silently reuse the old 100-epoch weights).
- Records population-level `clean_test_acc` per fold + `clean_test_acc_mean` in each
  `meta_pg_seed{S}.json`.

Regenerated seeds 42, 7, 1027 (10 folds each) at 200 epochs:

| seed | mean clean test acc (200ep) |
|---|---|
| 42 | 0.7233 |
| 7 | 0.7268 |
| 1027 | 0.7205 |
| **overall** | **0.7235** (old 100ep: 0.7224) |

Clean acc held ≥ the old ~0.72 (marginally higher), so the 100→200 epoch correction does
not degrade the victims. New checkpoints overwrite the meta paths the floor driver loads.
