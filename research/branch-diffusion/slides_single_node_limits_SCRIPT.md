# Verbatim 逐字稿 — *Why Single-Node Injection Is Not Enough*

> 配合 `slides_single_node_limits.{html,pptx}` 12 張投影片。中文口播版（繁體）。每張投影片預計 45–90 秒。總時長約 12–15 分鐘。

---

## Slide 1 — Title

> *"今天這場討論的核心問題只有一個：當我們在做圖分類的對抗注入攻擊時，**只用一個被注入節點到底夠不夠？**"*
>
> *過去十年的主流工作 — GZOO、KATZE、我們最近做的 SGDI — 預設都是 K = 1，也就是一個被注入的節點。看起來簡單、優雅、可分析。但我會用三個獨立的角度來說明：這個預設其實已經到頂了。理論上有限制，實驗上有上限，防禦上更是不堪一擊。"*
>
> *結束時各位會帶著一個明確的判斷：**多節點注入不是「再進一步」，而是「不得不」**。*

---

## Slide 2 — Threat Model

> *先把威脅模型對齊。我們在說的是 graph classification — 整張圖一個 label，例如分子是不是有毒、蛋白質是不是某個 fold。攻擊者拿到一張乾淨圖，加上 K 個「被注入節點」，控制：*
>
> 1. *這些節點的 **特徵向量** $X^*$；*
> 2. *它們連到原圖上哪些節點 — 也就是 **邊集合** $A^*$。*
>
> *目標：把整張圖的分類翻掉，但 stealth — 不能讓防禦端的異常偵測器抓到。*
>
> *KATZE、GZOO、我們的 SGDI，全部預設 **K = 1**。一個節點、一組特徵、一束邊。這套設定就是我們今天要挑戰的。*

---

## Slide 3 — The Research Question

> *問題其實很簡單：**K = 1 有沒有天花板？***
>
> *如果有，那後面所有圍著 K = 1 設計的 stealth 技巧 — 包括我自己花了一兩週做的 SGDI — 從一開始就是次優的起點。*
>
> *我用三個獨立的角度逼近這個問題：*
>
> - **理論**：GCN 的 normalization 本身會不會壓制單一節點？
> - **實驗**：在「邊預算固定」的公平條件下，多節點是不是真的贏？
> - **防禦**：當防禦端真的部署了，單節點扛得住嗎？
>
> *三個角度互相獨立。如果它們得出同一個結論，就不是偶然。*

---

## Slide 4 — Theory: GCN Self-Dilution

> *先看數學。GCN 的核心是這個 normalization：*
>
> $$\tilde{A} = D^{-1/2} A D^{-1/2}$$
>
> *對稱歸一，每條邊 $(v^*, u)$ 在訊息傳遞時的權重就是 $\frac{1}{\sqrt{\deg(v^*)\,\deg(u)}}$。*
>
> *關鍵：當我們把注入節點 $v^*$ 連到非常多個 victim，它的 degree 接近 $N$。每條邊的權重就掉到 $\frac{1}{\sqrt{N}}$ — 而且 **每多接一條邊，所有現有邊都跟著變弱**。*
>
> *這不是工程取捨，是純數學。**為自己的入侵買越多管道，每個管道的水流就越細**。*
>
> *這個現象有個名字：**self-dilution**。是 GCN 譜歸一化的副作用，沒辦法繞過。*

---

## Slide 5 — Single vs Multi (Comparison)

> *把單節點和多節點對照來看。*
>
> ***K = 1 的好處** — 模型乾淨：一個特徵向量、一個邊 mask、ZO 優化目標清晰。傳統 GIA 文獻幾乎都這樣設計。*
>
> ***K = 1 的代價** — 你剛才看到的 self-dilution；而且一個 degree = N 的節點，在乾淨圖上根本不存在 — 中位數 degree 才 4 左右。**這個 spike 在防禦端是 9 到 15 σ 的離群值**，純粹從 degree 就抓到。*
>
> ***K = 3 multi-node** — 把邊預算分散到 3 個節點。每個節點的 degree ≈ clean 中位數。加上注入節點之間的 inter-edge，整個 trojan 看起來像是「圖裡多了一個小社群」 — 而 clean 圖本來就有社群。*
>
> ***K = 3 的代價** — query 預算多 1.5×，需要 partition 策略，每個節點的特徵要獨立優化。但這些都是 *工程* 問題，不是 *結構* 問題。*

---

## Slide 6 — Experiment 1: ASR Ceiling

> *理論講完，看數據。PROTEINS + GCN + mean-pool，5-fold，**邊預算 τ 固定**（兩個方案總邊數一樣）：*
>
> - **K = 1，τ = 0.5（半預算）**：ASR **93.3%**
> - **K = 1，τ = 1.0（全預算，意思是這個節點全連 victim）**：ASR **88.6%**
>
> ***注意了：邊加倍，ASR 反而 *掉* 了 4.7 個百分點。***
>
> *這就是 self-dilution 的 *活體驗證*。把同樣的邊預算給一個節點，多到一個程度後反而傷自己。理論預測 → 實驗印證，完全一致。*
>
> *現在換 K = 3 partition（3 個節點，邊預算平均分散）：*
>
> - **K = 3，τ = 0.5**：ASR **98.1%**
>
> *paired 觀察：K = 3 vs K = 1 在 τ = 0.5 上 **+4.77 pp**；在 τ = 1.0 上 **+8.35 pp**。多節點 *穩贏*，預算越大贏越多。*

---

## Slide 7 — Experiment 1: Full Grid

> *攤開完整 6-config 對照表。重點看 K = 3 內部的三種邊分配：*
>
> - **duplicate**（3 個節點 → 同樣的 victim）：ASR 97.2%
> - **partition**（3 個節點 → 不重複的 victim）：ASR **98.1%** ← *最高*
> - **interconnect**（partition + 3 節點互連）：ASR 97.0%
>
> ***partition 顯著大於 duplicate**：核心是「**獨立邊 mask + 獨立特徵優化**」的協同 — 把同樣預算給三個獨立攻擊向量，比給三個複製品強。*
>
> *q/g（query per graph）成本 188 — 比 K = 1 的 98 多 92 個 query，但換來 **+8.35 pp ASR**，邊際效益清晰。*
>
> *資料來源 `multinode_20260524_061442.json`，PROTEINS 5-fold zo_free τ=1.0 mean。*

---

## Slide 8 — Experiment 2: Defense Vulnerability

> *上面講的還是無防禦下的 ceiling。**有防禦了，single-node 直接崩塌。***
>
> *我們在 PROTEINS 上 deploy **GDDM-A** — adjacency-side Binomial denoiser。看每個攻擊家族變什麼樣：*
>
> | 攻擊 | no_def | **gddm_a** |
> |---|---|---|
> | zo_free | 1.000 | 0.975（−2.5 pp）|
> | **SGDI**（單節點，特徵 Mahalanobis 投影）| 0.975 | **0.175（−80 pp）** |
> | **SpecGDI v1**（單節點 + 譜域保持）| 0.867 | **0.067** |
>
> ***SGDI 從 97.5% 掉到 17.5%，−80 pp 災難性崩塌**。SpecGDI 在 stealth 上更精緻 — 連譜域都保持 — *還是* 掉到 6.7%。*
>
> *結論：**所有單節點注入，不論特徵多隱形、譜域多平滑，在 A-channel denoiser 下都 ≤ 0.175**。這是個普世天花板。*

---

## Slide 9 — Defense-Side Mechanism

> *為什麼會這樣？因為 **denoiser 跟 OOD detector 都是 per-node 評分** — 每個節點獨立看，不看整體 gestalt。*
>
> *算個快速 z-score：clean PROTEINS 圖的 degree 中位數 ≈ 4，標準差 ≈ 5。注入節點 degree = N（比方 30 到 60）。我們注入的這個節點在 **degree 這個維度** 上就 9 到 15 σ outlier。*
>
> ***注意：這計算完全不看特徵、不看譜域、不看任何 stealth。** 純看 degree 統計。*
>
> *這就是為什麼 SGDI 的特徵 Mahalanobis 投影擋不住 — 防禦端根本沒在看 feature；SpecGDI 的譜保持也擋不住 — 防禦端根本沒在看譜。**它只看 per-node degree，而 K = 1 在這一維度上已經暴露**。*

---

## Slide 10 — Mitigation: K-Node Trojan

> *自然的解法：把單一 degree-N spike 拆成 **K = 3 到 5** 個 degree-中位數節點。*
>
> *每個注入節點的 degree 落在 clean 分布內 → 個別不被抓。再加上注入節點之間的 inter-node edges $A^*_{VV}$，整個 trojan 看起來像 clean 圖本來就有的 small community。Denoiser 訓練時就見過社群，會傾向「保留」。*
>
> ***誠實揭露 caveat**：我們做了 SpecGDI v2（K = 3 joint trojan）。Smoke 顯示在 GDDM-A 下 *也* 沒救回 ASR — denoiser 對 *每條邊* 打分數，trojan 的 inter-edge 一樣是 out-of-distribution。*
>
> ***但這不是論點失敗** — 是 *機制更精細*：光把 K 拆開不夠，內部社群結構要對齊 clean training 的 community signature。這條 future work 留給 *community-structure stealth*。*

---

## Slide 11 — Literature Consensus

> *別只信我們一家。看 *近期* GIA stealth 文獻的共識：*
>
> - **HAO**（ICLR'22）：K ≈ 5 / homophily-aware multi-node
> - **TDGIA**（KDD'21）：K ≈ 30 / topology-defective multi-node injection
> - **GANI**（TCSS'24）：K ≈ 5–10 / generative multi-node
> - **JANUS**（2509.13266, 2025）：K ≈ 5 / RL + manifold-aware
> - **QUGIA**（2025）：query-budgeted multi-node
>
> ***5 out of 5** 近期 GIA stealth 論文都已棄單節點。圖上紅色 short bar 就是 KATZE / SGDI 的 K = 1 — 在整個 *當代* 文獻地景裡是 outlier。*
>
> *這不是我們在挑事。**整個 field 已經往前走了**。我們今天要做的是說明清楚 *為什麼* — 而不只是「別人都這樣，所以我們也跟風」。理論、實驗、防禦三條獨立證據才是說服 reviewer 的東西。*

---

## Slide 12 — Conclusion + Future Work

> *把三個獨立論證收束起來：*
>
> 1. ***理論** — GCN self-dilution，per-edge influence O(1/√N)，有天然上限；*
> 2. ***實驗** — 在 matched edge budget 下，K = 3 partition 比 K = 1 高 **+4.8 到 +8.4 pp ASR**，paired 顯著；*
> 3. ***防禦** — 在 GDDM-A adjacency denoiser 下，single-node *任何 stealth 變體* 都 ≤ 0.175，K = 1 是 *普世* 弱點。*
>
> ***結論：多節點注入不是 incremental upgrade，是 *必要條件* — 對 ASR ceiling 必要，對 defense survival 也必要。***
>
> ***Future work**：community-structure stealth — 注入 trojan 子圖匹配 clean 社群的 *內部結構統計* 跟 *邊密度*；以及 per-node spectral fingerprint adversarial training，把譜域 stealth 真正放上戰場。*
>
> *問題的最終答案 — 「single-node 夠不夠？」 — *不夠*。理論、實驗、防禦三角同步否決。**多節點注入是 GIA 攻擊的新預設值。***
>
> *謝謝。*

---

## 講者註記

### 配速指引
- 1-3, 11-12 slide：節奏快（每張 30-45 秒，總攬 / 引子 / 收束）
- 4-5 slide：節奏中等（每張 60 秒，理論 / 設計取捨）
- 6-8 slide：節奏慢（每張 60-90 秒，**數據**，給聽眾消化時間）
- 9-10 slide：節奏快（45 秒，機制解釋）

### 視覺輔助動作
- Slide 4：講到 $1/\sqrt{N \cdot \deg(u)}$ 公式時，停 2 秒讓觀眾讀
- Slide 6：講 **93.3% → 88.6%** 時刻意放慢、語氣加重（這是 self-dilution 的「死亡證據」）
- Slide 8：講 **−80 pp 災難性崩塌** 時可以用手勢或滑鼠 highlight
- Slide 11：講「5 out of 5」時點頭強調
- Slide 12：講「不夠」兩字時看著觀眾，停 2 秒

### Q&A 預備問題
1. **「K = 3 為什麼不是 K = 5 或 K = 10？」** → smoke 規模限制；handoff 文件預留 K-sweep ablation
2. **「K = 3 多 query 成本可不可接受？」** → +92 q/g for +8.4 pp ASR，邊際效益明顯
3. **「為什麼 GDDM-A 對 K = 3 還是有效？」** → denoiser per-edge scoring；future work 列在 §10
4. **「跟 KATZE 主文怎麼接？」** → KATZE 應加 §Limitations of single-node 並引此論證
5. **「自家 SpecGDI 也單節點 NULL，是不是打臉自己？」** → 對，這正是我們 *為什麼* 做這場討論的原因 — 我們從失敗裡學到單節點才是真正的限制因素

### 投影片連結
- HTML 互動：`slides_single_node_limits.html`
- PPTX 匯出：`slides_single_node_limits.pptx`
- 數據來源：`results/multinode_20260524_061442.json`, `smoke_gddm_a.log`
- 整合報告：`REPORT_v2_CONSOLIDATED.md`
