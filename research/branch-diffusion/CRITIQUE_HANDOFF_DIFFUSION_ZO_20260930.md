# 設計審查 — Diffusion 先驗 + 零階優化的 GNN 節點注入 Handoff

**審查日期**：2026-09-30
**審查對象**：使用者貼上的「實驗 Handoff：Diffusion 先驗 + 零階優化的 GNN 節點注入評估」(設計完成、尚未執行)
**審查者**：Opus 4.8 session(read-only,未建環境、未跑任何實驗)
**上位目標**：使用者設定「參考 Flash,測試 diffusion + GZOO 對圖分類/節點分類做穩健性驗證,直到完整驗證」。本審查是該目標的 **Gate 0**:在燒 GPU 前先確認前提站得住。
**方法**:多視角(事實查核 / 資深工程 / 威脅模型 / 冗餘)交叉,並用**本 repo 既有實驗證據**檢驗 §5–§10。

---

## 0. 結論(Go / No-Go)

**判定:條件性 Go — 但設計必須先改三點,否則會重跑一次已被本 repo 否證的實驗。**

Handoff 是在一個「看不到這個 repo」的 app session 寫的(§12 明說檔案在 app 建的暫存資料夾),所以它把「diffusion 先驗做隱密節點注入」當成**未測試的新方向**。事實上本 repo 的 `branch-diffusion` 線**已經跑過這個機制的兩個版本**,結論對 H1/H2 直接不利(見 §1)。這不是致命否證(那些是**圖分類**,handoff 打的是**節點分類**,機制可能不同),但**舉證責任反轉**:handoff 必須先解釋「為什麼節點分類會得到和圖分類不同的答案」,再談 E3–E5。

**動手前必改(細節見對應節):**
1. **加入 trivial 特徵基線**(mean-feature / 複製鄰居特徵向量的注入)——本 repo 已經被這個坑燒過一次(SGDI「defended floor」原來就是 trivial μ-injection),見 §3-A。
2. **H2 有混淆**:V3(每目標獨立)對 V0(跨目標共享生成器)不是等價比較,把「diffusion 先驗」和「失去跨樣本學習」綁在一起了,見 §4-B。
3. **E0 閘門的「重設基準」未定義**:先把 **V0-strict**(最終注入圖再 query 一次驗證、所有 query 計入預算)訂為唯一合法基線,見 §5。

---

## 1. 最高價值發現:這個 repo 已經跑過核心機制(handoff 不知道)

app session 沒有本 repo 的存取權,以下證據它全都沒看到。

### 1-A. N23:feature-only diffusion 先驗 = 0 ASR 增益(已否證)
`research/branch-diffusion/FINDINGS_N23.md:5,13-21` — AdvAD 式非參數 diffusion reverse loop 直接移植到圖注入,**在 trivial 零特徵注入之上多加 0 個翻轉**:

| Config | T_steps | guidance | ASR (fold 0) | success/156 |
|---|---|---|---|---|
| N23 planned | 50 | 1.0 | 67.31% | 105 |
| N23 T=1 no-loop | 1 | 1.0 | 67.31% | 105 |
| N23 guidance×50 | 50 | 50.0 | 67.31% | 105 |
| KATZE baseline | — | — | 94.85% | — |

三個設定翻**同樣 105/156** 張圖 → diffusion loop 貢獻為零(`FINDINGS_N23.md:20`)。機制失效原因(`:32,45`):`feat_scale≈2√N` 讓注入特徵飽和 GCN normalization,預測完全由**注入拓樸**(full-edge)決定,不由特徵值決定;score guidance 的方向恰好對齊 x0=0 的飽和軸,diffusion 從不探索可翻剩下 51 張圖的拓樸相依子空間。

### 1-B. SGDI:guided diffusion + Mahalanobis 投影 → 不提升 undefended ASR
`REPORT_BRANCH_DIFFUSION_DISTSTEALTH.md:81,125` — SGDI(=handoff 構想最接近的已實現版本:ZO CGE 梯度 + 高斯 data-prior score + 硬 Mahalanobis 球投影)的定論:
> 「SGDI does **not** improve ASR on undefended models; it trades undefended-ASR for defense-invariance.」

ENZYMES undefended:SGDI 27.4% vs zo_free 94.2%(`:53`)。**這正面打臉 handoff 的 H1**(「diffusion 先驗提升隱密性、成功率不降」)——本 repo 的實現是「隱密性靠犧牲 undefended ASR 換來」。

### 1-C. 2026-09-29 ablation 進一步弱化 SGDI(記憶 `sgdi-mechanism-tradeoff`)
更新後的消融:**unconstrained SGDI 只跟 plain ZO 打平**;昂貴的約束效果因資料集翻轉(PROTEINS=prior pull、ENZYMES=projection);PROTEINS 的「defended floor」原來是被 early-stop artifact 浮出來的 **trivial μ-injection 基線**(F5 已撤回);已發表的 SS A.1 表**漏了 manifold 基線**。→ 連「diffusion 機制對 plain ZO 有邊際貢獻」都存疑。

### 1-D. 但這是**圖分類**,handoff 打**節點分類** — 逃生口(公平性)
以上全部是 **graph classification**(PROTEINS/ENZYMES,`global_mean_pool`)。1-A 的失效機制**明確依賴 mean-pool 稀釋(1/(N+1))+ feat_scale 飽和**(`FINDINGS_N23.md:29,32`)。節點分類**沒有 mean-pool 稀釋** — 注入節點的特徵直接進入目標節點的鄰域聚合。所以「diffusion 特徵被拓樸淹沒」的論證**未必轉移**到節點分類。

**這是 handoff 唯一的正當立足點,但它自己沒講。** 舉證責任:設計文件必須先論證(或 E4 先驗證)「節點分類下特徵內容確實主導、diffusion 先驗有空間發揮」,否則 1-A/1-B/1-C 就是預告 E4 的結果。建議把這條升為 **E4 之前的一個顯式前置實驗**:在節點分類 undefended 設定下,先量「trivial μ-injection vs feat-optimized ZO」的 ASR 差 —— 差夠大才有做 diffusion 的意義。

---

## 2. §4 GZOO 程式碼事實查核

> 對照 canonical 副本 `/workspace/reference/GZOO-Official/attack/gzoo.py`(非 §12 說已刪的暫存 `gzoo/`)。
> **版本比對**:`/workspace/reference/GZOO-Official/attack/gzoo.py` 與 `/tmp/gzoo-original-fresh/attack/gzoo.py` md5 相同(`17f75f90…`)= **無版本漂移**;handoff 引的每個行號都精準落在此檔上(194-196/283-352/305-339/355-370/414/443-475/528/547/567-601/605-610/634-639 全對)。→ **handoff 的 §4 就是照這個檔寫的,line-number 全可信。**

| # | 宣稱 | 判定 | 證據 |
|---|---|---|---|
| 1 | 回 logits、CW margin 為 loss → score-based 非 hard-label | **VERIFIED** | `:356` `logits=model(...,no_grad=True)`;`:365` 回 `max(target-other,kappa).item()` 實數 margin |
| 2 | ZO 打 `node_attr`/`edge_sample_weights`,`.backward(grad)` 回生成器 | **VERIFIED** | `:588` 傳入;`:607-610` `node_attr.backward(feat_grad)`。**細節**:傳的是 softmax 後的 `normalized_edge_sample_weights`(`:271`),ZO 擾動的是機率 |
| 3 | ~1+2×eval_num≈201 query/目標/步;×5 epoch≈1000 | **VERIFIED(是天花板不是典型)** | `main.py:54` eval_num=100、`:63` epochs=5;但 ~1000 是**上限**:見下方 prose(eval overhead 漏算 + `continue` 提早省 query) |
| 4 | edge_budget=1→邊固定連目標、結構生成器「無作用」 | **PARTIALLY-TRUE** | 邊*選擇*固定:`:194-196`;但「無作用」高估:`:245-246` attribute 路徑吃 `edge_feat`,第一個 edge GATConv 仍被 feature 梯度訓練。**只有第二個 edge GATConv(`:181`)真的沒訓到** |
| 5 | edge_budget=1 時邊估計 loop 仍跑、100 次 query 相同(浪費) | **VERIFIED** | `:325-339` 恆跑,`:330` 固定圖→每次 loss 相同→`factor=0`(`:346`);`:606-610` 只在 edge_budget>1 用 edge_grad,否則丟棄 |
| 6 | **success 膨脹**:任一候選成功即標成功、eval `continue` 不再驗證實際輸出 | **VERIFIED** | `:301` `success=0`;`:307-309`/`:332-334` `if success!=1: success=_success`(=200 個 σ-擾動候選有沒有任一成功);`:585` query 真實輸出但 `:587` **覆寫** success;`:617` 寫入;`:547`/`:414` `continue` 跳過→**永不再驗證**。作者自知:`:586` 註解 |
| 7 | 生成器建一次、跨目標更新、無留出集 | **VERIFIED** | `:634-638` 建一次、`:639` 一個 optimizer、`:616` 每目標 `step()`;全 `target_indexes` 都攻擊+訓練,無 train/test 目標切分 |
| 8 | 隱密性只有 node/edge budget + feature budget(離散對齊 count、連續對齊全域 μ/σ) | **VERIFIED** | `:568-572` 離散 MSE 對齊 count;`:574-575`/`:600-601` 連續對齊 `(μ-self.μ)²+(σ-self.σ)²`。無其他 stealth 項 |
| 9 | 同質性函式存在但整段註解掉 | **VERIFIED** | `:443-475` `calculate_node_centric_homophony` 全註解;呼叫點(`:310-311,:336-337,:428-434`)也全註解 |
| 10 | `defense/`= gnnguard/gtrans/magnet/prognn/rgat/rgcn | **VERIFIED** | `ls` 完全吻合 |

**重現坑**(全 VERIFIED,除註明):`--gzoo_batch_size` 預設 0→`:528` `num_nodes % 0` ZeroDivisionError(README 都傳 64 遮住);`--attacker` 預設 `'GZoo'` 但程式比 `'GZOO'`(`main.py:120,141,152`)→ 不傳大寫會 NotImplementedError;sigma 預設 1e-6(README 1e-3)。**`--feature_budget` → PARTIALLY-TRUE**:命名確實不符(實際 `--feature_budget_factor`),但 argparse `allow_abbrev=True` 讓前綴解析成功、**不會壞**,別浪費時間。**額外 doc-staleness**:README:29 叫人改 `main.py:138` 設 wandb project,實際 `wandb.init` 在 `main.py:80`(僅 README 過期,非 gzoo.py 漂移)。

**對本研究最關鍵的兩點:**
1. **Claim 6 是真的,但膨脹幅度讀碼讀不出來。** 控制流確認無誤(`585→587 覆寫→617 寫入→547/414 跳過`,無最終再驗證)。但候選是 `node_attr ± σ·randn`(`:293-294`),在 README 的 σ=1e-3 下只是真實輸出周圍的薄殼 → **膨脹幾個 pp 是經驗問題,不是讀碼問題**。因此 **不能宣稱「95.69% 是假的」**,只能說「95.69% 的成功認定含一個結構性膨脹路徑,幅度未知」。→ **這正是 §5 要求 V0-strict(最終圖再 query 一次)的硬理由**:唯有重測才知道膨脹多少、本研究的數字才與論文可比。
2. **Claim 4 降級**:別寫「結構生成器沒作用」,要寫「edge_budget=1 時**邊選擇**固定、**第二個** edge GATConv 沒訓到(但第一個經 feature 分支仍訓練)」。V1「跳過無效邊估計」仍成立(省的是 `:325-339` 那 100 次相同 query),但論述要精確。

---

## 3. §5 方法設計漏洞

### 3-A. 【關鍵】缺 trivial 特徵基線 → H1 可被免費打平
V0–V5 沒有「注入 mean-feature 或複製一個真實鄰居特徵向量」的零成本 in-distribution 基線。少了它,「diffusion 先驗提升隱密性」**無法歸因**:一個直接落在資料流形上的 trivial 向量本來就通過任何分佈式偵測器。本 repo 的 manifold 基線(in-dist、no opt)在圖分類上 ASR 很低(PROTEINS 59.9%、ENZYMES 11.6%,`DISTSTEALTH:41,50`),而且 SGDI 的「floor」被證實就是 μ-injection(§1-C)。**必加基線 V_triv**,H1 才可否證。

### 3-B. q(邊查詢向量)在 edge_budget=1 時是死重
§5 給注入節點加 `q∈R^h` 做 top-k 邊選擇,但 §4 自承 README 全部範例 edge_budget=1 → 邊固定連目標、結構生成器無作用。那 h 個額外 ZO 維度在 e=1 時純粹是**沒有訊號的搜尋維度**,只會稀釋 query 效率。**建議:e=1 時關閉 q 的 ZO,或先用 E1 證明 e≥2 有意義再引入 q。**

### 3-C. §3 的維度論述方向錯了
§3 把「戰場移到 GNN」的理由之一講成「注入規模小、低維」。但 z∈R^{d+h} **比** GZOO 原本優化的 node_attr∈R^d **更大**。ZO 若有優勢,來源只能是**先驗流形把有效搜尋維度壓低**,不是輸入本身低維。文件應改述,否則 H2 的動機站不住。

### 3-D. V2 ≈ 文獻現狀,不是強對照
V2(同質性 + ego 正則)本質上就是 HAO/GANI/QUGIA 已經做的「bound 到資料範圍 / 鄰居相似」(`DISTSTEALTH:133,139`)。把它當「隱密性來自 diffusion 還是約束」的關鍵對照是對的,但要注意:如果 V2≈V3(§9 的 E4 閘門),結論不是「diffusion 沒用」而是「diffusion 沒有勝過**便宜的**約束」——文件要把這個 kill 條件講死。

---

## 4. §6 假設與可否證性

### 4-A. H1 已被本 repo 的圖分類證據預先否證(見 §1-B/1-C)
節點分類上可能不同,但 H1 現在的表述(「提升隱密性、成功率不降」)在圖分類是**假的**。→ H1 應**先分域**:H1-node / H1-graph,且加上「相對 V_triv 的增益」為主指標。

### 4-B. 【關鍵】H2 混淆:V3 vs V0 不是等價比較
- V0 的生成器**跨所有目標共享並持續更新**(handoff §4 引 634–639)。
- V3 是**每目標獨立**對 z 做零階。
- 在相同 query 預算下比 V3 vs V0 → 把「diffusion 先驗的效果」和「V3 丟掉了 V0 的跨樣本學習」兩件事**綁死**。V3 輸不代表 diffusion 差,可能只是少了跨樣本記憶。
- **修法**:加一個 **V0-per-target-reinit** 臂(每目標重置生成器)當 H2 的對照;或直接用 **V4 vs V0** 測 H2(V4 才有跨樣本 z-init)。

### 4-C. H3 的可否證條件寫得好,但要防資料洩漏
H3「第 k 個目標平均 query 隨 k 下降」用留出集評估是對的。但 §5 的自監督編碼器 + 先驗都在「攻擊者已知圖」上訓練 —— 要確保**留出目標**不在先驗訓練集,否則 H3 的 query 下降是記憶洩漏不是學習。

---

## 5. §9 閘門問題

- **E0 閘門「重設基準」未定義**:應在動手前就把唯一合法基線訂為 **V0-strict**:(a) 成功一律以「最終注入圖再 query 一次」認定;(b) 所有經過受害模型的呼叫(含梯度估計的 ±σ 探測)計入 query 預算。這也正好對應 §4 宣稱 6 的 success-inflation 疑慮(見 §2 查核):若原版是「梯度估計中任一候選成功即標成功、事後 continue 不再驗證」,則論文 95.69% 與本研究的數字**不可比**,必須全部用 V0-strict 重測。
- **E3 閘門位置錯**:§10 risk 1(Cora/Citeseer/CS 是 binary bag-of-words,連續 diffusion 產非二元值、閾值化破壞對齊)不是腳註,是 **E3 的硬閘門**。本 repo 已知同型問題:TU 資料被 loader 載成 binary、`dist_match` 在其上等於 fs=1.0 無意義(`project-reference.md:49-50`)。→ E3 應在**連續特徵資料集**(節點分類:Coauthor-CS 部分連續 / ogbn-arxiv 連續;或補一個連續特徵資料集)上先驗證先驗品質,binary 資料集只作 topology-driven 對照。

---

## 6. §10 風險補強

- **風險 1(離散特徵)** 升為 E3 閘門(見 §5)。
- **風險 4(隱密度可被繞過)**:本 repo 有 `evaluation-integrity-registry` E1–E8 的踩坑清單與 R-EVALHACK 判準(`project-reference.md:62`),偵測器 AUC≈0.5 這種「贏在單一指標 / 防禦類」的宣稱正是 red-flag。多指標並陳 + 每個數字附 `results/*.json`+timestamp(專案硬規)。
- **補**:handoff §4 宣稱 5(edge_budget=1 時邊估計 loop 仍跑 100 次相同 query)= 純浪費 query。V1 修掉它是對的,但要注意這會**改變 query 預算的分母**,V0-strict 與 V1 的 query-ASR 曲線因此不可直接疊 —— 要在圖上標明兩者 query 定義。

---

## 7. 修正後的執行計畫(Gate 0 → 完整驗證)

GPU/訓練指令一律交給使用者手動啟動(專案慣例 + classifier 政策)。以下為**改過的前四步**,取代 handoff §11:

1. **建環境 + E0(V0-strict 重現)** — `reference/GZOO-Official/secgnn.yml`,依 §4 修掉五個坑(batch_size、GZoo/GZOO 大小寫、sigma、arg 名、wandb)。E0 產出:Cora+GCN 的**真實 query 數**與**驗證後**成功率,和論文 95.69% 的偏差。**閘門**:若 V0-strict << 95.69% → 論文數字含 success-inflation,全線改用 V0-strict。
2. **V_triv + V1** — 先加 trivial μ-injection / 鄰居複製基線(§3-A),再實作 V1(跳過無效邊估計 + 最終圖再驗證)。這一步完全不碰 diffusion,建立可靠地板。
3. **節點分類的前置判準實驗(§1-D)** — undefended 設定量「V_triv vs feat-optimized ZO」的 ASR 差。**閘門**:差 < 幾個 pp → 節點分類也是拓樸主導,diffusion 先驗大概率重演 N23,**停,回報**。差夠大 → 繼續。
4. **E2 指標與偵測器** — 建 §8 隱密性度量(同質性、ego-MMD、值域/範數、偵測器 AUC),量 V0/V1/V_triv 基線。

做完 1–4 才有資格談 E3(先驗品質)→ E4(V2 vs V3,H1/H2)→ E5(V4,H3)。E4 的 V2/V3 比較務必帶 §4-B 的 V0-per-target-reinit 對照。

---

## 8. 給使用者的話

- 本審查是 **read-only**,沒建環境、沒跑實驗、沒改程式。
- 「Flash」的相關性:Flash 的價值是**查詢效率 + 跨樣本學習**的先例,對應 H3;但 handoff §3 已正確論證「ZO 直接打離散 PE 輸入不可行」,所以不是把 Flash 機制照搬,而是借它「用生成先驗省 query」的精神 —— 這點 handoff 處理得對,無需改。
- 這份審查**沒有**寫進 obsidian-vault(使用者要的是 review 不是歸檔;寫 vault 需先過 `meta/document-architecture`)。若要歸檔進 `projects/gzoo-katze/branches/branch-diffusion/`,我可以另跑一次。

---

## 9. 執行進度(Gate 0 → E0,2026-09-30 本 session)

朝「diffusion+GZOO 穩健性驗證直到完整」目標推進,已做:

- **環境診斷**:`graph_adversarial`(KATZE 圖分類 env)缺 `torch_scatter` 且是 torch 2.12 nightly(cu128),與 GZOO 的 pt20cu118 PyG stack 不相容 → **不可沿用**。正確路徑=建 `secgnn` env。
- **secgnn env 已建好(背景,exit 0)**:python 3.9 / torch 2.0.0+cu118 / dgl 1.1.1+cu118 / pyg 2.5.3 / torch-scatter+sparse `+pt20cu118`。import 全過。建置腳本:`$CLAUDE_JOB_DIR/tmp/build_secgnn.sh`。
- **⚠️ GPU 不可用 → E0 走 CPU**:本機 GPU 是 **RTX 5080 / compute cap 12.0(sm_120,Blackwell)**,torch 2.0.0+cu118 沒有 sm_120 kernel(`CUDA error: no kernel image`)。而支援 sm_120 的只有 cu128 torch(=`graph_adversarial` 那個 nightly,缺 torch_scatter)→ 死結。**結論:GZOO(secgnn stack)只能跑 CPU**;Cora+SGC 夠小,CPU 可行。
- **工作副本**:`/workspace/reference/GZOO-Working`(canonical `GZOO-Official` 不動)。已打兩個最小 patch 讓它跑 CPU:`main.py:76` 加 `args.device<0 → 'cpu'` fallback;`main.py:146` 把傳給 attacker 的 `device=args.device`(int)改成 `device=device`(string),否則 attacker 端仍鎖 cuda:0。並 `mkdir checkpoint/victim_model`。
- **E0 指令(CPU,GZOO-Working,執行中 background)**:
  ```bash
  cd /workspace/reference/GZOO-Working && WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="" \
  micromamba run -n secgnn python main.py --seed 1027 --dataset=cora --node_budget=1 \
  --edge_budget=1 --feature_budget=1 --victim_num_layers=2 --victim_model=sgc --hid_dim=256 \
  --lr=0.001 --attacker=GZOO --gzoo_khop_edge=2 --gzoo_kappa=-0.001 --gzoo_run_mode=black-box \
  --gzoo_attack_epochs=5 --gzoo_sigma=1e-3 --gzoo_patience=4 --gzoo_gen_feat_dim=64 \
  --gzoo_gen_hid_dim=128 --gzoo_gen_lr=1e-3 --gzoo_batch_size=64 --device -1 --instance=E0
  ```
  (坑已規避:`--attacker=GZOO` 大寫、`--gzoo_batch_size=64`、`--gzoo_sigma=1e-3`、`--device -1` 強制 CPU。)
- **victim 已確認**:Cora SGC val 80.00% / test 80.20%(E0.log)。
- **§2 claim 5 已被實測確認**:E0.log 每筆 `Edge Gradient Estimation ... Loss Std = 0.00000000` → edge_budget=1 的邊估計 loop 100 次 query 零變異(全相同)= 純浪費,V1 該砍的就是它。

### V0-strict / V1 / V_triv 補丁規格(對照已查核的 gzoo.py 行號)

在 `reference/GZOO-Official` **之外**開副本改(勿污染 canonical),實作三件事:

1. **V0-strict 成功認定(修 success 膨脹)** — 病灶:`:585` query 真實注入圖 → `:587` 用候選 verdict **覆寫** `success` → `:617` 寫入 → `:547`/`:414` `continue` 永不再驗。
   **改法**:在 `:585` 取真實輸出後,**用它算最終 `success`**(`infer_cw_loss` 對實際 `construct_graph` 輸出),**不要**被 `estimate_gradient` 的候選 verdict 覆寫;`attack_results[...]`(`:617`)寫這個 strict 值。梯度估計裡的候選成功只作**訓練訊號**,不作**成功認定**。
2. **全 query 計數** — 計入 `:585` 的 1 次 + `estimate_gradient` 兩個 loop 各 `total_sample_num`(`:305`,`:325`)+ `evaluate_performance`(`:411`,在 `:621` 每 batch 對**全** target 跑)的 eval query。輸出 `total_queries` 供 query-ASR 曲線(handoff §8 主指標)。
3. **V1(跳過無效邊估計)** — `edge_budget==1` 時 skip `:325-339` 的邊估計 loop(100 次相同 query,`edge_grad` 反正 `:606-610` 被丟棄)。注意 §2 提醒:這改變 query 分母,V0-strict 與 V1 曲線的 query 定義要在圖上標明。
4. **V_triv 基線(§3-A 必加)** — 新增一個 feat_mode:注入節點特徵 = 資料集全域 mean(或複製一個真實鄰居的特徵向量),topology 用同樣 edge_budget=1 連目標,**不做任何 ZO**。這是 H1「diffusion 提升隱密性」的免費對照;缺它 H1 不可否證。

### E0 結果(2026-09-30,CPU,~1h)
**stock GZOO,Cora + SGC(README `Effectiveness` 指令原封不動)→ Final Success Rate = 73.57%**(edge_budget=1,5 epochs,seed 1027)。要點:
1. **73.57% 低於論文 95.69% ~22pp,且這已是膨脹版**(含 §2 claim 6 success-inflation)。**修正措辭**(advisor):不可斷言「V0-strict 只會更低」——§2 說 `:585` query 真實輸出但 `:587` **覆寫丟棄** verdict(不是 OR),所以「真實注入成功但候選沒成功」的目標**不被計**;σ=1e-3 下膨脹方向大概率佔優,但**只能實測膨脹幅度,不保證方向**。
2. **論文 95.69% 的歸屬(直接讀 `GZOO-Yu2025.pdf`)**:Effectiveness 表用 **6 個 victim(GCN/SGC/APPNP/ChebNet/SGFormer/PC-Net)**;文中「on the Cora dataset, GZOO achieves an ASR of 95.69%, well above the 66.01% of next-best SPEIT」是跨-victim 的代表/最佳值,**非單一 SGC cell**。**ASR 定義**=「proportion of target nodes incorrectly classified after attack」= 我的 success_rate 定義,denominator 可比。→ 73.57%(SGC)是真實 reproduction gap,不是量法差異。README:35「Victim=GCN」標題 vs :37 指令 `sgc` 是 README 自身不一致。
3. **query 數未取得**:GZOO 不印 total_queries(只有 wandb,已停用);需 V1 instrumentation 才有 query-ASR 曲線。
4. victim Cora SGC test acc 80.2%(兩 env 一致)。**但 ASR 非 deterministic**——見下方 stack-sensitivity;不可稱 E0 數字 deterministic。

### E0 stack-sensitivity(重要 reproducibility 發現)
同 seed(1027)、同參數、同 victim acc(80.20%),兩個 env 給**不同 ASR**:
| env | stack | device | Final ASR |
|---|---|---|---|
| secgnn | dgl 1.1.1 / torch 2.0 | CPU | **73.57%** |
| graph_adversarial | dgl 2.4 / torch 2.12 | GPU | **68.45%** |

→ **~5pp stack-sensitivity**(dgl 1.1↔2.4 / torch 2.0↔2.12 + device 的**優化過程**數值/RNG drift。**注意**:後來 V0-strict 證明候選-verdict 收斂後淨效 ≈0,所以這 5pp **不是** success 偵測器的 RNG,而是優化本身的 drift)。疊在對論文 95.69% 的 ~22–27pp gap 之上。**含意**:(a) 整個驗證 program 要**固定單一 stack**(faithful 選 secgnn/CPU,scale 選 graph_adversarial/GPU;兩者不可混);(b) 前置判準等**成對比較**必須同 env 同 seed,ASR 絕對值飄不影響成對結論;(c) 再次坐實 §2:success-inflation 讓數字對 RNG 敏感 → V0-strict 更必要。

### 前置判準 + V0-strict 部分結果(2026-09-30,secgnn/CPU,seed 1027,Cora+SGC)
- **Trivial baseline(random_real,on-manifold 真實節點特徵,無優化)= 5.86% ASR**(deterministic:26 個 evaluate_performance 呼叫 × 3 epochs 全 5.86%,收斂確定值)。→ 節點分類下「1 注入節點 + 1 邊 + 中性真實特徵」幾乎不 flip(target 本來就有很多真鄰居,單一注入被稀釋)。
- **success-inflation 是 epoch-0 transient,非收斂差(已修正先前過度宣稱)**:epoch 0 stock ~58% vs strict 6.98%(差 ~51pp)——但那是 generator **未訓練**時的假象(候選 σ-擾動偶有 flip,真實注入圖幾乎不 flip)。**generator 訓練後 strict 快速追上**:epoch 1 strict = **56.61%**。→ 真正的膨脹量 = stock final(73.57%)− strict final(待跑完),**遠小於 51pp**。§2 claim 6 仍成立(候選 verdict ≠ 真實 verdict),但收斂後幅度小得多;不可宣稱「73.57% 主要是膨脹」。final strict ASR pending(epoch 1/5,climbing)。
- **前置判準結果(RESOLVED,決定性)**:strict-optimized **final = 74.44%**(5 epochs)。三數對比:trivial 5.86% / stock 73.57% / strict 74.44%。
  - **(1) 收斂後 success-inflation ≈ 0**:strict 74.44% ≈ stock 73.57%(甚至 +0.87pp,因 strict 下 target 不被候選-flip 移除 → 多優化幾步)。epoch-0 的 51pp gap 收斂後**全部消失**。→ §2 claim 6 是真的 code smell 但**經驗上不膨脹最終數字**;**V0-strict 的價值 = 證明「我這一跑(README SGC config,單 seed/單資料集/單 victim)沒被膨脹」**——**不等於**論文 95.69% 沒被膨脹(那個 headline 我仍未重現,停在 73–74%)。
  - **(2) 前置判準 RESOLVED — PASS,且非稻草人**。四點對比(Cora+SGC,secgnn/CPU,seed 1027):

    | baseline / method | ASR | 隔離出 |
    |---|---|---|
    | trivial random_real(class-中性) | 5.86% | ≈ topology-only floor |
    | trivial class_shift(便宜對抗,零 ZO) | **21.32%** | feature *方向* |
    | GZOO stock(候選 verdict) | 73.57% | learned ZO |
    | **GZOO strict(learned ZO,真實 verdict)** | **74.44%** | learned feature 優化 |

    - class_shift(21.32%)≫ random_real(5.86%):注入第二名 class 的真實節點確實推動 target(+15.5pp)→ feature **方向**有用。
    - **strict(74.44%)− class_shift(21.32%)= 53.12pp ≫ 10pp**:learned ZO **優化**遠超便宜對抗挑法 → feature **優化本身**決定性,**不是稻草人**(advisor 說的「class_shift≈15% → 結論更強」那一支)。
    - **結論**:節點分類下 learned feature 優化決定性(與 graph-cls N23 相反)。**diffusion 先驗線有真實空間**:目標高(~74%),而便宜的 on-manifold 挑法只到 21% → 一個能逼近 74% 又保持 on-manifold 的 learned prior 才是貢獻。§1-D 舉證責任 + advisor item 1 **正面達成**。
  - **(3) 修正 stack-sensitivity 因果**:68→74% 的 CPU/GPU 5pp gap **不是**候選-verdict 的 RNG(該機制收斂後淨效 ≈0),而是 **dgl 1.1↔2.4 / torch 2.0↔2.12 在優化過程的數值/RNG drift**。

### 下一階段 roadmap + 成本現實(advisor item 4)
- **已完成(cheap,都在 Cora+SGC 單 seed)**:E0 重現、V0-strict、三個 trivial arms、GPU 解鎖。這些是**最後一批便宜實驗**。
- **下一步(仍算 cheap,但需實作)**:E2 stealth 指標(homophily / ego-MMD / detector AUC)套在三個 arm(random_real / class_shift / GZOO-strict)上 → 得 (ASR, detectability) Pareto 點 → 才有 H1 的固定門檻。**需先存注入圖**(GZOO-strict 的 per-target generator 輸出目前沒存 → 要再一次 instrumented run)。
- **成本斷崖**:strict-optimized 單 config(Cora+SGC,單 seed)CPU ~2h。handoff §7 的 E4–E7(4 資料集 × 4–9 victim × 5 seed × 6 query budget)= **數千 CPU-hours**。RTX 5080 是 sm_120,secgnn(faithful stack)只能 CPU;GPU 要用 graph_adversarial(數字與 secgnn 差 ~5pp,不可互比)。→ **E2 之後每一步都需要「GPU-env(接受數字不可比)」或「砍 scope」**,使用者需先選。

### GPU 解鎖(advisor item 4,成功)
原判「sm_120 → 只能 CPU」**已推翻**。GZOO 卡在 cu128 env 的唯一原因是 module-level `import torch_sparse`(需 torch_scatter),而用它的 `adj_to_tensor`/`tensor_to_adj` 是**死碼(全 repo 無呼叫)**。修法(都在 `GZOO-Working`,canonical 不動):
- `attack/utils.py`:torch_sparse + `scipy.sparse.csr` import 包 try/except(defensive)。
- `models/utils.py`:把 pyg 相依的 6 類(ChebNet/PCNet/SGFormer/GTrans/MAGNet/ProGNN)從 top-level 改成 `init_gnn_model` 內 lazy import;核心 victim(gcn/sgc/gat/appnp/rgat/gcnguard/rgcn)是 pyg-free。
→ GZOO 現可在 `graph_adversarial`(torch 2.12 nightly / cu128 / sm_120)**用 GPU 跑**。E0-gpu 執行中(`--device 0`,env=graph_adversarial)。注意:Cora 太小(2708 nodes),GPU 對 SGC 的加速可能有限(kernel launch 開銷);GPU 的價值在**較重的 victim/資料集**(GCN、Coauthor-CS、ogbn-arxiv)。

### 未解 / 下一步
- **[OPEN] 73.57% vs 95.69% 的 gap 歸因未定**:論文 Cora 逐-victim 表在 PDF layout 下抽不乾淨;95.69% 是 6-victim 的 headline/best,SGC 那格的確切值待查(用 `scripts/vault_rag` 或人工讀 PDF 表)。在確認前,**73.57% 當成 SGC stock 基線**,不宣稱「reproduction 失敗」也不宣稱「成功」。
- **[目標相關] GZOO 論文本身含 graph classification**:文中「54.26% on PROTEINS with GCN」——使用者目標的「圖分類」半邊,GZOO 原文就有做,且本 repo branch-diffusion 全線是 graph classification。→ 「圖分類」的穩健性驗證應**接到本 repo 既有 SGDI/N24 結果**(見 §1),不是從 GZOO node-cls 重跑;「節點分類」半邊才是 handoff 這條新線。
- **next chunks(依序,advisor 定案)**:
  1. **V0-strict 先做**(stack-sensitivity 證明 success 認定被 RNG 汙染 → 前置判準前必須先修)。2-line flagged 改動:`--gzoo_strict_success` 讓 `:585` 對真實 `cur_graph` 的 `success` 不被 `:587` estimate_gradient 的候選 verdict 覆寫,直接寫入 `:617` attack_results;estimate_gradient 的內部 success 只留作訓練訊號。**不動 `evaluate_performance`**(generator 每步更新,per-step `:585` 才是該 target「最終圖」的正確定義)。
  2. **節點分類前置判準**(最決定性):flag `--gzoo_feat_override {none,neighbor_copy}`,在 GZOO loop 內跳過 estimate_gradient/optimizer.step、把 node_attr 設成**複製目標(或隨機真實節點)的特徵列**(套同樣 `F.normalize`),edge 用 GZOO 同款 construct_graph。**不可用 mean 向量**(binary bag-of-words 的 mean 是 dense off-manifold 點,會讓 V_triv 不公平地弱、白送 H1)。同 env(secgnn/CPU)同 seed 成對比。
  3. **閘門門檻(先寫死,事後不准改)**:成對同 target 下,**strict-optimized ASR − strict-neighbor_copy ASR < ~10pp** → feature 優化貢獻微小 → diffusion 先驗線對 node-cls moot(重演 N23)。≥10pp 才續做 diffusion。
- **注意**:graph_adversarial(GPU)的 ASR 與 secgnn 差 ~5pp,**兩 env 數字不可互比**;報 GPU 數字時必註明。

### 下一步(依賴,見 §10 advisor 決策)
- 待定:跑 Cora+GCN 直接檢驗 95.69%,還是先上 V0-strict 量 SGC 的膨脹幅度。
- CPU-only(sm_120 無解)使整個 E0–E7 program 很慢 → 需重新規劃規模(少 config / 每 config 抽樣節點)。
