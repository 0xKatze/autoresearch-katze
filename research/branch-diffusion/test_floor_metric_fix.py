#!/usr/bin/env python3
"""test_floor_metric_fix.py — FIX1 unit check for the floor-harness ASR metric.

Confirms the baseline-margin gate: a graph the DEFENDED model already misclassifies
pre-attack (defended-margin < 0) is EXCLUDED from BOTH the ASR numerator and denominator
(no free flips), while the 'none'-defense path (defended-correct == base-correct) is
unchanged. Pure CPU, no GPU, no victim training — uses a tiny fake defended model whose
forward is fully controllable, and a stubbed greedy flipper that records which graphs it
is asked to attack.

Run: micromamba run -n graph_adversarial python test_floor_metric_fix.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from torch_geometric.data import Data

import eval_paper_grade_floor as F   # imports the FIXED harness (CPU)


def toy_graph(gid, label, n=4):
    x = torch.ones(n, 3)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    d = Data(x=x, edge_index=ei, y=torch.tensor([label]))
    d.batch = torch.zeros(n, dtype=torch.long)
    d.gid = torch.tensor([gid])
    return d


class FakeVictim(torch.nn.Module):
    """Forward returns 2-class logits; misclassifies clean graphs whose gid is in
    `wrong_ids` (defended-margin < 0 at baseline), classifies the rest correctly."""
    def __init__(self, wrong_ids):
        super().__init__()
        self.wrong_ids = set(int(i) for i in wrong_ids)

    def forward(self, data):
        gid = int(data.gid.view(-1)[0].item())
        y = int(data.y.view(-1)[0].item())
        logits = torch.full((1, 2), -5.0)
        pick = (1 - y) if gid in self.wrong_ids else y   # wrong vs correct class
        logits[0, pick] = 5.0
        return logits


def main():
    fails = []

    # 3 clean graphs, all base-correct (collect_test_graphs would have kept all 3).
    graphs = [toy_graph(0, 1), toy_graph(1, 0), toy_graph(2, 1)]

    # --- Test A: eligible_graphs excludes the defended-misclassified graph -------------
    victim_wrong1 = FakeVictim(wrong_ids=[1])      # defended model gets gid=1 WRONG
    assert F.defended_correct(victim_wrong1, graphs[0], "cpu") is True
    assert F.defended_correct(victim_wrong1, graphs[1], "cpu") is False   # margin<0
    assert F.defended_correct(victim_wrong1, graphs[2], "cpu") is True
    elig = F.eligible_graphs(victim_wrong1, graphs, "cpu")
    elig_ids = sorted(int(g.gid.item()) for g in elig)
    if elig_ids != [0, 2]:
        fails.append(f"A: eligible ids {elig_ids} != [0,2] (gid=1 not excluded)")
    else:
        print(f"[A] PASS  eligible={elig_ids}  (defended-margin<0 graph gid=1 excluded)")

    # --- Test B: run_attack_cell (om2c) — pre-misclassified graph counts in NEITHER
    #            numerator nor denominator. Stub flips EVERY graph it is given. -----------
    attacked = []
    orig = F.om2c_greedy_flip
    def stub_greedy(model, data, prior, cfg, device):
        attacked.append(int(data.gid.view(-1)[0].item()))
        return (True, 1, 1)   # always flips, 1 edge, max inj degree 1
    F.om2c_greedy_flip = stub_greedy
    try:
        r = F.run_attack_cell("om2c", {"node_budget": 1, "feat_scale": 1.0},
                              victim_wrong1, graphs, om2c_prior_cache=None)
    finally:
        F.om2c_greedy_flip = orig

    if sorted(attacked) != [0, 2]:
        fails.append(f"B-numerator: attack ran on {sorted(attacked)} (gid=1 should be skipped)")
    else:
        print(f"[B] PASS  attack ran on {sorted(attacked)}  (gid=1 never entered numerator)")
    if r.get("n_eligible") != 2 or r.get("n_total") != 3:
        fails.append(f"B-denominator: n_eligible={r.get('n_eligible')} n_total={r.get('n_total')} "
                     f"(expected 2 and 3 -> gid=1 excluded from denominator)")
    else:
        print(f"[B] PASS  denominator n_eligible=2 (of n_total=3)  (gid=1 excluded)")
    # all 2 eligible flipped by the stub -> ASR over defended-correct denom = 1.0
    if abs(r["asr"] - 1.0) > 1e-9:
        fails.append(f"B-asr: asr={r['asr']} != 1.0 (2 flips / 2 eligible)")
    else:
        print(f"[B] PASS  ASR=1.0 = 2 flips / 2 eligible (NOT 3 base-correct)")

    # --- Test C: BUGGY behaviour would have given asr=3/3 by counting the pre-
    #            misclassified gid=1 as a flip. Show the fix changed the denominator. -----
    buggy_asr = 3 / 3   # what flips/len(graphs) would have produced
    if abs(r["asr"] - buggy_asr) < 1e-9 and r.get("n_eligible") == r.get("n_total"):
        fails.append("C: fixed asr equals buggy base-correct asr (gate not applied)")
    else:
        print(f"[C] PASS  fixed denom {r['n_eligible']} < base-correct {r['n_total']} "
              f"(bug would have inflated by counting gid=1)")

    # --- Test D: 'none'-equivalence — a victim that gets ALL right => eligible==all -----
    victim_all_ok = FakeVictim(wrong_ids=[])
    elig_all = F.eligible_graphs(victim_all_ok, graphs, "cpu")
    attacked2 = []
    def stub2(model, data, prior, cfg, device):
        attacked2.append(int(data.gid.view(-1)[0].item())); return (True, 1, 1)
    F.om2c_greedy_flip = stub2
    try:
        r2 = F.run_attack_cell("om2c", {"node_budget": 1, "feat_scale": 1.0},
                               victim_all_ok, graphs, om2c_prior_cache=None)
    finally:
        F.om2c_greedy_flip = orig
    if len(elig_all) == 3 and r2.get("n_eligible") == 3 and sorted(attacked2) == [0, 1, 2] \
            and abs(r2["asr"] - 1.0) < 1e-9:
        print(f"[D] PASS  none-equivalence: all 3 eligible, denom unchanged (ASR=1.0)")
    else:
        fails.append(f"D: none-equivalence broken: elig={len(elig_all)} "
                     f"n_eligible={r2.get('n_eligible')} attacked={sorted(attacked2)} asr={r2['asr']}")

    print("=" * 64)
    if fails:
        print("UNIT CHECK FAILED:")
        for f in fails:
            print("  - " + f)
        raise SystemExit(1)
    print("UNIT CHECK PASSED: defended-margin<0 graph excluded from numerator AND "
          "denominator; 'none' path unchanged.")


if __name__ == "__main__":
    main()
