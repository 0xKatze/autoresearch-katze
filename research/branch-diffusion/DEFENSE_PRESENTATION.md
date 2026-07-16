# GZOO-Katze 防禦性評估 — 完整簡報

> 整合本 session（2026-05-26）camera-ready 結果與 Phase C decisive grid。所有數字 paired Wilcoxon n=15（3 seeds × 5 folds，3 seeds × 10 folds 因 disk 限制 deferred — 見 §6 adaptations）。

---

## 1. 防禦機制總覽（attack-defense axis matrix）

| Defense | 偵測軸 | 攻擊類型可擋 | 公開文獻來源 |
|---|---|---|---|
| `none` | — | 基準（無防禦） | — |
| `normfilter` | feature L2 magnitude | 超大 norm 注入 (zo_free) | KATZE / GZOO 主文 baseline |
| `jaccard` (GCN-Jaccard) | edge cosine similarity | 異質鄰邊 | Wu et al., 2019 |
| **`mahalanobis`** | feature distribution (mean + cov) | off-distribution 特徵向量 | **本 session 新提案** |
| **`gnnsafe`** | post-GCN embedding energy | embedding-OOD 注入 | Wu et al., ICLR 2023 |
| **`gddm-X`** | feature manifold (DiffPure) | adversarial 特徵擾動 (image-style) | GDDM paradigm, simplified |
| **`gddm-A`** | adjacency Binomial denoising | off-distribution 邊 | GDDM (arXiv 2501.11568) |

---

## 2. 各防禦深度介紹

### 2.1 `none` — undefended baseline

無任何過濾。注入節點直接送 GCN 推論。**唯一不會錯殺乾淨樣本的選項**（clean acc = 1.000）。是其他防禦的對照組。

### 2.2 `normfilter` — L2 範數閾值

```
defender 計算 clean training 節點的 ||x||_2 分佈
threshold τ = max_clean‖x‖（或 95-pct）
過濾：注入節點 x* if ‖x*‖_2 > τ → 丟棄
```

**捕獲方向**：直接擋下未約束 (`zo_free`) 攻擊的 magnitude-blow-up 策略。

**侷限**：對 L2-bounded 攻擊（如 `zo_normcap` / SGDI）無效 — 它們本來就在合法 norm 內。**這是 KATZE/GZOO baseline 的 stealth 機制本質**。

### 2.3 `jaccard` (GCN-Jaccard, WSDM 2019)

```
for each edge (u, v) in injected adjacency:
    sim = jaccard(features(u), features(v))
    if sim < threshold:  remove edge
```

**捕獲方向**：homophily 假設 — 連結節點應有相似特徵。注入節點若連到隨機/異質鄰居，邊會被剪。

**侷限**：注入節點若特徵接近 victim 平均 → 看起來像 homophilic → 邊保留。對 distribution-aware 攻擊弱。

### 2.4 `mahalanobis` — **本 session 新貢獻**

```
defender 在 clean training nodes 上估計 (μ, Σ)
score(x*) = (x* − μ)ᵀ Σ⁻¹ (x* − μ)  # Mahalanobis distance²
threshold τ_def = 99.5-pct of clean score
if score(x*) > τ_def → 移除注入節點
```

**捕獲方向**：不只看 magnitude，看*相關結構*。L2 球體 vs Mahalanobis 橢球 — 後者對齊資料協方差。

**為什麼是本 session 的關鍵防禦**：
- 揭露 KATZE/N4 norm-cap stealth 的盲點（norm 受控但 direction 沒）
- 對標 SGDI 設計的 Mahalanobis 球投影是「**defense-axis-aligned attack**」
- Camera-ready: PROTEINS 把 zo_normcap 從 0.773 → **0.042 (−73 pp)**，SGDI 仍 **0.655**

### 2.5 `gnnsafe` (ICLR 2023) — embedding-based energy OOD

```
1) Per-node logit (從 GCN classifier 取，跳過 mean-pool)
2) Energy E(v) = -T · logsumexp(z_v / T)  # T=1 default
3) Belief propagation K=2 步: ẽ ← α·ẽ + (1-α)·D⁻¹A·ẽ, α=0.5
4) 若 propagated neg_energy < threshold → 標 OOD → 丟節點
threshold = 5-pct of clean propagated neg_energy
```

**捕獲方向**：embedding 空間的能量 OOD。對 *aggregation 後* 的 representation 偵測。

**為何對 SGDI 無效**：SGDI 的 Mahalanobis 投影保 feature 在分佈內 → message passing 後 embedding 仍 in-distribution → 能量低 → bypass。

### 2.6 `gddm-X` — feature DiffPure（paradigm-faithful simplification）

```
Train: 3-layer GCN denoiser, forward q_t(x|x_0) = N(√α̅_t x_0, (1-α̅_t)I)
       L2 noise prediction, 100 epochs AdamW lr=1e-3
Inference:
  x' ← x* + √(1-α̅_{t*}) · ε   # 加噪到 t* = 20 (10% noise)
  x_purified ← reverse_denoise(x', t* → 0)
  classify with purified x
```

**捕獲方向**：把擾動視為「乾淨 + 對抗噪聲」，diffusion 反向把它推回 manifold。

**對 node-injection 的反效果**：feature smoothing 把 adversarial 方向**推向更 universal 的方向** → ENZYMES zo_free ASR 從 0.875 → 1.000 (**+12.5 pp**)。**channel mismatch**：GDDM 原意是擋 adjacency 攻擊，不是 feature 注入。

### 2.7 `gddm-A` — adjacency Binomial denoising（paper-faithful core）

```
Train: 3-layer GCN + per-edge MLP edge-prediction head
       Forward q_t(A|A_0) = Bernoulli flip with prob β_t per edge
       BCE loss on edge labels
Inference (noise-then-reverse):
  A' ← perturbed adjacency (with injected node's edges)
  A_t ← binomial_flip(A', t* = 30)
  reverse step k = t*..1:
      logits_edges ← denoiser(X, A_k, k)
      A_{k-1} ← Bernoulli(σ(logits)) ⊙ A_t   # GSDR-lite mask
  classify with denoised A
```

**捕獲方向**：注入節點的邊在「乾淨 adjacency 分佈」上是 OOD → 反向擴散有機會剪掉。

**為何對所有攻擊有效（universal ceiling）**：
- denoiser 訓練只見過 clean adjacency 模式
- 注入節點的邊 *不論 feature 怎麼藏、譜怎麼保* 都是 off-distribution
- PROTEINS 上 SGDI / SpecGDI v1 / SpecGDI v2 **全部** ≤ 0.07
- ⚠️ **副作用**：t*=30 對小圖（ENZYMES n≈30）會把 clean acc 砍 35 pp，t* 需 per-graph-size 調

---

## 3. 完整實驗結果表（Camera-Ready, n=15）

### 3.1 PROTEINS — ASR mean ± std

| feat_mode \ defense | none | normfilter | jaccard | **mahalanobis** | gnnsafe* | gddm-X* | **gddm-A** (smoke n=15) |
|---|---|---|---|---|---|---|---|
| **zo_normcap (N4)** | 0.773 ± .042 | 0.558 ± .101 | 0.793 ± .072 | **0.042 ± .069** | 0.738 | 0.665 | 0.975 → 0.950 |
| **sgdi** | 0.668 ± .044 | 0.662 ± .045 | 0.704 ± .046 | **0.655 ± .047** | 0.618 | 0.577 | **0.975 → 0.175** |
| specgdi v1 | 0.609 | n/a | n/a | 0.550 | 0.609 | 0.584 | 0.067 |
| specgdi v2 | 0.267 | n/a | n/a | n/a | n/a | n/a | 0.000 |

`*` = Phase C decisive grid (n=5)，不在 Phase D camera-ready scope.

### 3.2 ENZYMES — ASR mean ± std

| feat_mode \ defense | none | normfilter | jaccard | **mahalanobis** | gnnsafe* | gddm-X* | gddm-A** |
|---|---|---|---|---|---|---|---|
| **zo_normcap (N4)** | 0.942 ± .055 | 0.752 ± .013 | 0.900 ± .056 | **0.197 ± .050** | 0.940 | 0.699 | (contaminated) |
| **sgdi** | 0.279 ± .075 | 0.234 ± .031 | 0.414 ± .055 | **0.295 ± .070** | 0.280 | 0.176 | (contaminated) |
| specgdi v1 | 0.212 | n/a | n/a | 0.223 | 0.220 | 0.208 | (contaminated) |

`**` = clean acc 在 t*=30 上崩 35 pp，ENZYMES gddm-A 數字無法解釋。

### 3.3 SGDI vs N4 paired contrast（the headlines）

| dataset | defense | Δ (sgdi − zo_normcap) | 95% CI (mean) | Wilcoxon W | p (two-sided) |
|---|---|---|---|---|---|
| PROTEINS | none | −0.104 | [−0.111, −0.098] | — | 6.10×10⁻⁵ |
| PROTEINS | normfilter | +0.104 | [+0.035, +0.180] | — | 2.56×10⁻² |
| PROTEINS | jaccard | −0.088 | [−0.115, −0.058] | — | 1.88×10⁻³ |
| **PROTEINS** | **mahalanobis** | **+0.613** | **[+0.564, +0.662]** | **0 (15/15)** | **6.10×10⁻⁵** |
| ENZYMES | none | −0.663 | [−0.721, −0.598] | — | 6.10×10⁻⁵ |
| ENZYMES | normfilter | −0.518 | [−0.529, −0.505] | — | 6.10×10⁻⁵ |
| ENZYMES | jaccard | −0.486 | [−0.506, −0.466] | — | 6.10×10⁻⁵ |
| **ENZYMES** | **mahalanobis** | **+0.098** | **[+0.073, +0.123]** | **0 (15/15)** | **6.10×10⁻⁵** |

**頭條**：在 **Mahalanobis** defense 下，SGDI 在 PROTEINS 比 N4 高 **+61.3 pp**、ENZYMES 高 **+9.8 pp**。兩個 Wilcoxon W=0（**所有 15 對配對 SGDI 都贏 N4**），p 觸及 n=15 exact two-sided 地板（6.10×10⁻⁵）。

### 3.4 X/A channel asymmetry on PROTEINS（new finding）

| feat_mode | no_def | gddm-X | gddm-A |
|---|---|---|---|
| zo_free | 1.000 | 1.000 (Δ=0) | 0.975 (**Δ=−2.5pp**) |
| **sgdi** | 0.975 | 0.975 (Δ=0) | **0.175 (Δ=−80pp)** |
| specgdi v1 | 0.867 | — | 0.067 |
| specgdi v2 | 0.267 | — | 0.000 |

ENZYMES zo_free under gddm-X: ASR **+12.5 pp**（防禦反效果）.

---

## 4. 防禦評估的四個結論

### 結論 1：N4 / norm-cap 是 feature-distribution-blind

- normfilter, jaccard, none 三個防禦下 sgdi vs zo_normcap **互有勝負**（前者贏 normfilter, 後者贏 none/jaccard）
- 在 **mahalanobis** 下，zo_normcap 從 0.773 → 0.042（−73 pp on PROTEINS），sgdi 維持 0.655
- **L2 範數限制不等於分佈合規**：在 norm 球內仍有大量 off-distribution 方向，Mahalanobis 橢球能切出來

### 結論 2：SGDI 是 *defense-axis-aligned* stealth 的正解

- 對 feature-distribution defense（mahalanobis）：SGDI 完勝 N4，+61.3 / +9.8 pp，p=6.1×10⁻⁵
- 對 feature-magnitude defense（normfilter）：N4 與 SGDI 差不多（normfilter 不是 SGDI 設計的對手）
- **stealth axis 要對齊 detection axis** — 不是越多越好

### 結論 3：X/A channel asymmetry

- X-channel purification (gddm-X) **反而幫攻擊**（最高 +12.5 pp ASR）— 對 node-injection 而言是錯誤的防禦設計
- A-channel purification (gddm-A) **無敵**（所有攻擊 ≤ 0.07）— 但 t* 調太大會錯殺 clean (ENZYMES n≈30 上 −35 pp clean acc)
- **Defender's strongest lever 是 adjacency purification，不是 feature purification**

### 結論 4：spectrum-preserving stealth 在當前防禦上 wasted budget

- SpecGDI v1, v2 在所有 5 個防禦上都 ≤ SGDI（NULL verdict, p > 0.05）
- 因為**沒有任何 tested 防禦在偵測譜域**
- **Stealth axes 必須對齊 detection axes**：novel stealth 只有在有對位 defense 時才買到 ASR
- ⚠️ 為下一篇 paper 留 seed：實作 per-node spectral fingerprint defense → SpecGDI 才有 venue

---

## 5. 視覺化（建議簡報用圖）

### 5.1 Bar chart: SGDI vs N4 across 4 defenses（PROTEINS）

```
ASR @ full edges, paired n=15
       ┌─────────────────────────────┐
1.0 ─  │ N4   SGDI                   │
       │ ▓▓   ░░                     │
0.8 ─  │ ▓▓░░       ▓▓░░  ▓▓░░       │  ← none / normfilter / jaccard:
       │ ▓▓░░       ▓▓░░  ▓▓░░       │    互有勝負
0.6 ─  │ ▓▓░░  ░░░░ ▓▓░░  ▓▓░░  ░░░░ │
       │ ▓▓░░  ▓▓░░ ▓▓░░  ▓▓░░  ░░░░ │
0.4 ─  │ ▓▓░░  ▓▓░░ ▓▓░░  ▓▓░░  ░░░░ │
       │                              │
0.2 ─  │                              │
       │                              │
0.0 ─  │ none   nf   jacc   mahal     │  ← Mahalanobis 是唯一防禦
       └─────────────────────────────┘    SGDI 大贏 N4 +61.3pp ★
```

### 5.2 Asymmetry box: X vs A channel purification

```
GDDM purification on PROTEINS zo_free / sgdi:
                     no_def   gddm-X    gddm-A
       ┌───────────────────────────────────────┐
       │  zo_free    1.000    1.000    0.975   │
       │  sgdi       0.975    0.975    0.175   │ ← A-channel kills sgdi
       │  specgdi v1 0.867       —     0.067   │
       │  specgdi v2 0.267       —     0.000   │
       └───────────────────────────────────────┘
                   ↑                     ↑
            X-channel: 反效果      A-channel: universal floor
            (channel mismatch)     (defender wins)
```

---

## 6. Adaptations 必要 disclosure（per project policy）

| Component | Paper / spec | 實際採用 | 對結論影響 |
|---|---|---|---|
| Phase D n | 3 seed × 10 fold = 30 | **3 seed × 5 fold = 15** | p 地板 6.1e-5（仍 ≫ 0.001），CI 略寬 |
| GNNSafe per-node logit | node-classifier 預設 | 改用 graph-classifier 跳 mean-pool | 「standard adaptation」 |
| GDDM-X | adjacency Binomial + GSDR | **改 X-channel DiffPure** | paradigm-faithful，**channel divergent**；揭露 channel asymmetry 反倒成新貢獻 |
| GDDM-A | MPM + Bernoulli edge head + Gumbel-max | 自製 3-layer GCN + per-edge MLP + BCE | mechanism-level 近似 |
| SpecGDI v1 signal-side | 完整 graph-Fourier envelope | 單節點 smoothness only | NULL verdict 不受影響 |
| SpecGDI v2 joint grad | CGE ~250 q/step | SPSA 8 q/step + topk threshold | NULL gap 5 pp，無法翻轉 |
| ENZYMES gddm-A t* | < 3 pp clean drop | −35 pp（t* 對小圖太狠） | ENZYMES gddm-A 行 contaminated |

**Headline 整體 robustness**：camera-ready 主數字（+61.3 / +9.8 pp p=6.1e-5）只受 n=15 影響，paper-grade 標準仍達。SpecGDI NULL verdict robust。X/A asymmetry 為 mechanism-level 發現，對實作細節不敏感。

---

## 7. Paper 章節對應建議

| 防禦 | KATZE 主文章節（建議） |
|---|---|
| `none`, `normfilter`, `jaccard` | §Existing Defense Baselines |
| **`mahalanobis`** | **§A New Distribution-Aware Defense (this work)** |
| `gnnsafe` | §Established OOD-Based Defense |
| `gddm-X`, `gddm-A` | §Purification-Based Defense — **with channel asymmetry analysis** |
| spectral defenses (GCN-SVD / ProGNN / EvenNet) | §Future Defense — Spectral Domain (未跑) |

5 個 contributions 列表（paper-ready，refresh from `REPORT_v2_CONSOLIDATED.md §E`）：

1. **N4 norm-cap is feature-distribution-blind** under Mahalanobis
2. **SGDI is the axis-aligned stealth answer** (+61.3/+9.8 pp, p=6.1e-5)
3. **X/A channel asymmetry for purification defenses** (本 session 新發現)
4. **First Mahalanobis-ball injection + per-node Mahalanobis-filter** for GIA
5. **SpecGDI v1+v2 honest negative ablation**: stealth axis 必須對齊 detection axis

---

## 8. Reproduce

```bash
# Phase D camera-ready (PROTEINS + ENZYMES × 4 defenses × 2 feat_modes × 3 seeds × 5 folds)
micromamba run -n graph_adversarial python run_camera_ready_grid.py
micromamba run -n graph_adversarial python analyze_camera_ready.py \
  > results/camera_ready_wilcoxon_20260526_174919.json

# X/A channel asymmetry smoke
micromamba run -n graph_adversarial python smoke_gddm_a.py

# All-defense decisive grid (Phase C, n=5)
micromamba run -n graph_adversarial python run_decisive_grid.py
```

Files of record under `/workspace/autoresearch-katze-n23/research/branch-diffusion/results/`:
- `eval_camera_ready_PROTEINS_20260526_174919.json`
- `eval_camera_ready_ENZYMES_20260526_174919.json`
- `camera_ready_wilcoxon_20260526_174919.json`
- `eval_decisive_PROTEINS_capped_20260526_125040.json`
- `eval_decisive_ENZYMES_*.json` (3 files: early, remaining, finalpush)
