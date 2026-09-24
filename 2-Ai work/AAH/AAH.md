---
title: "AAH: Learning Attribute–Affordance Hierarchies in Hyperbolic Space for Open-Vocabulary 3D Object Affordance Grounding"
aliases: ["AAH", "Attribute-Affordance Hierarchies", "Hyperbolic Affordance", "双曲空间属性-可供性层级"]
authors: [Yuxuan Wang, Tong Li, Yihang Zhu, Guangtao Lyu, Yukuan Min, Chenghao Xu, Jiexi Yan, Xu Yang, Cheng Deng]
venue: "ICML 2026 (PMLR 306, Seoul)"
affiliation: ["Xidian University", "Hohai University"]
tags: [paper, cs.CV, 3D-affordance, open-vocabulary, hyperbolic-space, hypergraph, MLLM-prompt, counterfactual, related-work]
created: 2026-09-23
year: 2026
pdf: "paper/1-inbox/AAH.pdf"
bib: |
  @inproceedings{wang2026aah,
    title={Learning Attribute--Affordance Hierarchies in Hyperbolic Space for Open-Vocabulary 3D Object Affordance Grounding},
    author={Wang, Yuxuan and Li, Tong and Zhu, Yihang and Lyu, Guangtao and Min, Yukuan and Xu, Chenghao and Yan, Jiexi and Yang, Xu and Deng, Cheng},
    booktitle={Proceedings of the 43rd International Conference on Machine Learning (ICML)},
    year={2026}
  }
---

# AAH（Attribute–Affordance Hierarchies）

> [!abstract] 摘要
> **AAH**（Wang et al., ICML 2026，西电 Deng Cheng 组）针对 **open-vocabulary 3D object affordance grounding（OVAG）**。
> 核心批评：现有方法（[[GREAT]]、LASO、IAGNet 等）把交互图像/文本当作**外部知识源**去和 3D 表征对齐，却忽略了**物体局部属性（attribute）与可供性（affordance）之间的内在层级关系** —— 杯把手能抓，不是因为「见过人抓杯」，而是因为它**呈半圆、粗细合适**；affordance 由**属性组合**涌现，而非由全局外观决定。
> 解法三件套：① **超图（hypergraph）** 建模局部区域间关系 → ② **双曲空间（Lorentz 模型）** 编码「属性 ⊃ 可供性 ⊃ 3D 区域」的层级偏序 → ③ **反事实属性 prompt**（"如果这个物体没有 [属性]，还能 [affordance] 吗？"）削弱虚假相关。

> [!info] 一句话定位
> 是 [[GREAT]] 这条「MLLM prompt → 文本/图像/点云融合」路线的**结构性升级**：不改骨架，而是把融合后的表征**从欧氏搬到双曲**，并显式施加**蕴含锥（entailment cone）**的层级约束与**反事实排除**约束。backbone 仍是 DINO-ViT-S + PointNet++，文本用 RoBERTa。

---

## 1. 动机与问题设定

- **任务（OVAG）**：给定 3D 点云 `P ∈ R^{N×3}` + 人-物交互图像 `I` + 文本 prompt，预测逐点 affordance 热图 `Y_g`；训练监督只有 affordance 标注 `P_label ∈ R^{N×1}`。
- **评测三 split**：Seen / Unseen Object / Unseen Affordance（见 [[PIAD]]）。

> [!question] 现有范式的两个病
> 1. **视觉-only 对齐**（IAGNet 等）：只在「见过的 affordance 类型」里打转，closed-set。
> 2. **文本/几何直接对齐**（LASO、OpenAD）：强依赖训练分布。**遇到未见的 affordance（如 pour）时，因为没有建模属性级信息，模型会退化成靠粗粒度视觉相似，把未见 affordance 塌缩到高频 affordance 上**。
>
> **AAH 的处方**：显式建「属性 → 可供性 → 3D 区域」的层级，让泛化来自**结构**而不是**记忆**。

![[images/fig1-hierarchy.png]]
*论文 Fig.1：杯把手能"被抓"源于「半圆形 + 合适厚度」等局部属性的组合——属性节点 → 可供性节点 → 3D 点云实例的层级，在双曲空间中离原点越远语义越具体。*

---

## 2. 方法

![[images/fig2-pipeline.png]]
*论文 Fig.2：AAH 总体框架。左：图像分支（DINO-ViT → 超图卷积）与反事实 prompt 分支（MLLM → RoBERTa → Fusion）；中：图像/点云特征经指数映射抬入双曲空间；右：层级目标函数 `L_cont / L_ord / L_cf` 在蕴含锥中的几何示意（❄ 冻结、🔥 可训练）。*

> [!info] 心智模型（对应论文 Fig.2）
> ```
> I ─DINO-ViT─▶ F ─超图构建/卷积─▶ F̃ ─┬─▶ 指数映射 exp_o^c ─▶ 双曲空间 H_c^d
> MLLM prompts ─RoBERTa─▶ T_o / T_a / T_cf ─┘        │
> P ─PointNet++─▶ F_p ────────────────────────────────┘
>                                                     │
>              ┌─ L_cont（跨模态正样本对比）          │
>              ├─ L_ord^{o→a} + L_ord^{a→P}（蕴含锥排序）◀┘
>              └─ L_cf（反事实排除）
>                       ↓
>        Decoder（To 扩展 ⊕ F_p → 上采样 ⊕ F_a）→ sigmoid → affordance heatmap φ
>        L_point = L_focal + L_dice
> ```

### 2.1 超图构建（Sec 4.1）
- DINO-ViT 提图像特征 `F ∈ R^{C×H×W}`，**每个空间特征向量作为一个顶点** `V_exo = {v_1..v_HW}`。
- **超边**由距离阈值 `ε` 定义 α-space：`α-space(v_i, ε) = { v_j | ||v_i − v_j||_d < ε }`，该集合构成一条超边（一条超边可连多于两个顶点）。
- 关联矩阵 `A(v,e) ∈ {0,1}`；用带残差的**超图卷积**（Feng et al., AAAI19）：
  `F̃ = F + D_v^{-1} A D_e^{-1} A^T F Θ`
- 作用：让网络聚焦 **in-context 的局部关系**，为后续 affordance 推理提供 relation-aware 的几何基础。
- **ε 消融**：1→3 递增，4/5 反而下降（超边过密 → 引入背景噪声）；**ε=3 最优**（Fig.3）。

![[images/fig3-epsilon.png]]
*论文 Fig.3：距离阈值 ε 的消融。ε 过小 → 超边稀疏、局部关系捕不全；ε 过大 → 过度平滑、背景特征混入。三个 split 都在 ε=3 达峰。*

### 2.2 反事实 Prompt 生成（Sec 4.2）
- 沿 [[GREAT]] 的 prompt 1–4，**新增 prompt 5（反事实）**：
  > "Explain why this part would fail to interact if its geometric structure of the {object} no longer satisfied the requirements for interaction."
  > （图 2 中的通俗版："What if the [object] did not have the [attribute]—would it still [affordance]? Explain why?"）
- 动机：属性/可供性 prompt 虽带来强语义先验，但也带来**虚假相关** —— 频繁共现的属性被误当成 affordance 成立的必要条件，损害泛化。
- MLLM 用世界知识回答「属性被扰动后 affordance 是否仍成立」，逼模型学**因果链**而非**共现统计**。
- RoBERTa 编码 → 物体几何属性特征 `T_o ∈ R^{N_o×C}`、可供性意图特征 `T_a ∈ R^{N_a×C}`、反事实特征 `T_cf ∈ R^{N_cf×C}`；再 reshape + 两层 Conv1d 与图像特征融合得 `F_o, F_a, F_cf`。

> [!question] 「属性」到底是什么？一个还是多个？
> - **是什么**：物体上**可交互部件的几何结构属性** —— 形状、尺寸、空间、支撑、稳定性等物理几何特征，由 MLLM 看图后用**一句话自由描述**，再交给 RoBERTa 编码。**不是**类别标签，也**不是**颜色/纹理这类外观属性。
>   - Fig.1 例：杯把手 `semi-circular, proper space, thickness support` → grasp。
>   - Fig.2 例：`Due to their flat and wide surfaces allowing the person to sit` → sit。
> - **几个**：实现上**每个样本恰好 1 个属性节点** —— `N_o = 1`（Sec 4.2 定义 `N_o` = number of interaction objects），`T_o ∈ R^{1×512}`。
>   - 但这一条描述是**组合式**的，一句话里可同时含多个几何特征（半圆 + 空间 + 厚度）。Intro 里说的 **attribute compositions** 指描述内部的多特征组合，**不是**多个独立属性节点。
> - **对照**：affordance 侧是 **3 个**（`N_a` = number of interaction ways = 3，来自 P3 当前交互 + P4 的另两种常见交互），`T_a ∈ R^{3×512}`；反事实 `T_cf ∈ R^{1×512}`。
> - **所以层级是扇形而非链**：1 个属性 → 3 个可供性 → `N_p` 个 3D 点。
>
> [!warning] 语义粒度与实现口径不一致
> 论文反复强调属性是 **object-region / 局部部件级**（P1 让 MLLM 指出「与人交互的部件」，若与主要功能区不同则**再指出一个** —— 即可能指出 2 个部件），但 `N_o` 的官方定义却是「交互**物体**数」而非部件数，且最终仍拼成**单条** `T_o`。
> 复现口径：按「拼接后单条 `T_o`（1×512）」实现；若要真正建模多部件，需把 `N_o` 改成部件数并相应改 `L_ord^{o→a}` 的配对方式 —— 这是一个现成的改进切入点。

> [!info] Prompt 1–5 与分组（附录 A）
> 沿用 [[GREAT]] 的 prompt 1–4，AAH 只**新增 prompt 5**（反事实）：
> - **P1**「指出图中与人交互的部件；若它与主要功能区不同，指出主要功能区」+ **P2**「从几何结构解释为何该部件能交互（一句话）」→ 拼接为**物体几何知识** `T_o`
> - **P3**「描述人与物的交互（含交互类型…）」+ **P4**「列出该物体另外两种常见交互…」→ 拼接为**可供性意图知识** `T_a`
> - **P5**「如果该物体的几何结构不再满足交互要求，该部件为何会失效？（一句话）」→ **反事实属性描述** `T_cf`
>
> 设计要点：P5 与 P1/P2 描述的是**同一个属性**，只是把它「扰动掉」——构成成对的 factual / counterfactual，这样 `L_cf` 才有明确的对立面。

### 2.3 双曲层级学习（Sec 4.3）

> [!tip] 定位：这是「训练期的几何约束头」，不是特征提取器
> - **输入**：欧氏特征 —— 图像侧 `F_o / F_a / F_cf ∈ R^{512×196}`（Sec 4.2 融合输出）、点云侧 `F_p ∈ R^{512×64}`；外加可学习曲率 `c`（优化 `log(c)`，初始化 1.0）。
> - **输出**：**只有三个标量损失** `L_cont / L_ord / L_cf`（+ 学到的 `c`），梯度回传去塑形 encoder 的欧氏特征。
> - **注意**：Sec 4.4 的 decoder 吃的是 `F_p + T_o + F_a`（欧氏），**不含双曲坐标** → 双曲分支只在训练期把表征"掰"成有层级结构的样子；推理时理论上可整块摘掉（论文 Tab 7 报的 2.1s 是含它的）。
> - 数据流：`欧氏特征 → 池化出 anchor → exp_o^c 抬升（512→513 维）→ 量 θ 与 φ → 三个损失`

- **几何**：Lorentz（双曲面）模型，常负曲率 `−c`；Lorentz 内积 `⟨u,v⟩_L = ⟨ũ,ṽ⟩_E − u_{d+1}v_{d+1}`；距离 `d_H(u,v) = (1/√c)·arccosh(−c⟨u,v⟩_L)`。
- **指数映射**：以原点 `o = (0,...,0,1/√c)` 为中心把欧氏特征抬到流形上。附录 Eq.18 闭式解：
  `ũ = sinh(√c‖v‖_E)/(√c‖v‖_E) · v`，`u_{d+1} = cosh(√c‖v‖_E)/√c`
  （一般式 Eq.17：`exp_w^c(v) = cosh(√c‖v‖_L)·w + sinh(√c‖v‖_L)/(√c‖v‖_L)·v`）
- **外角 exterior angle（Eq.19，排序损失要用）**：
  `θ(h_t,h_p) = arccos( (h_{p,d+1} + c·h_{t,d+1}·⟨h_t,h_p⟩_L) / (‖h̃_t‖_E · √((c⟨h_t,h_p⟩_L)² − 1)) )`
- **层级直觉**：**原点 = 最抽象/最一般的属性概念，离原点越远 = 语义越具体**（attribute → affordance → 具体 3D 实例）。双曲体积随半径指数增长，天然容纳树状层级，缓解欧氏嵌入的「拥挤」。
- **聚合**：`z̄_I = (1/HW) Σ_k F_I^k`、`z̄_P = (1/HW) Σ_i F_P^i` → `h_I = exp_o^c(z̄_I)`, `h_P = exp_o^c(z̄_P)`。

> [!warning] 论文措辞与公式不一致
> 正文写的是 "a geometry-aware weighting scheme **based on the correlation between local tokens and global semantics**… prevents over-smoothing"，但**给出的公式就是朴素均值池化** `1/HW Σ`，恰好是最容易 over-smooth 的做法。
> 复现时应按公式来（或自己换成相关性加权）

> [!info] 概念速查：欧氏 vs 双曲、θ 与 φ
> - **欧氏特征**：曲率为 0 的平坦空间 `R^d`，距离 `d_E(a,b)=‖a−b‖`，球体积 `∝ r^d`（多项式）。encoder 输出 / decoder 输入都在这里。
> - **双曲流形**：常负曲率 `−c` 的曲面 `H^d`，距离沿测地线 `d_H = arccosh(−c⟨u,v⟩_L)/√c`，球体积 `∝ e^{(d−1)√c·r}`（指数）→ 节点数随深度指数增长的树能被低失真装下。
> - **指数映射** `exp_o^c`：把切空间向量贴到曲面上，`ũ = sinh(√c‖v‖)/(√c‖v‖)·v`，`u_{d+1} = cosh(√c‖v‖)/√c`；512 维进、513 维出。`‖v‖→0` 退化为 `v`，`c→0` 退化回欧氏。
> - **θ(h_t,h_p)**：Lorentz 外角 = 子节点偏离父节点的角度；**φ(h)**：蕴含锥半孔径 = `arcsin(2K/(√c‖h̃‖))`，**越靠近原点锥越宽**（抽象概念能罩住更多具体子概念）。
> - 三个损失都是 `max(0, ·)` 型 hinge：**满足约束就零梯度，只在违规时推一把**。

三个损失：

| 损失                                  | 形式                                                                                                              | 作用                                           |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| **Positive Contrastive** `L_cont`   | `s(i,j) = −d_H(h_i^I, h_j^P)/τ`，对称 InfoNCE                                                                      | 跨模态实例级对齐,跟标准 InfoNCE 唯一的区别：**相似度换成了负的双曲距离**。 |
| **Ordering** `L_ord`                | 蕴含锥半孔径 `φ(h) = arcsin(2K/(√c·‖h̃‖_E))`，`K=0.1`；惩罚 `max(0, θ(h_o,h_a) − φ(h_o))` 与 `max(0, θ(h_a,h_P) − φ(h_a))` | 非对称偏序：属性**蕴含**可供性，可供性**蕴含** 3D 区域            |
| **Counterfactual Exclusion** `L_cf` | `max(0, φ(h_cf) − θ(h_cf, h_a))`，要求真实 affordance **落在反事实锥之外**                                                   | 削掉虚假的「属性-可供性共现」                              |

- 关键设计：`L_ord^{a→P}` 把抽象 affordance 语义**落地**到具体 3D 点特征上 —— 这是「层级」真正服务于 localization 的那一环。

### 2.4 Decoder 与总损失（Sec 4.4）
- `F̄_p = f([F_p, f_Θ(T_o)])`（把 `T_o` 扩展到 `N_p` 后与点特征拼接卷积）→ Feature Propagation 上采样到 `R^{C×N}` → 与 reshape 后的 `F_a` 融合 → `F_α`。
- head + sigmoid → `φ ∈ R^{N×1}`；`L_point = L_focal + L_dice`。
- **总损失**：`L_total = L_point + L_cont + λ(L_ord^{o→a} + L_ord^{a→P}) + μ L_cf`，`λ = μ = 1`。

---

## 3. 实验

### 3.1 设置
- **Backbone**：PointNet++（3 个 SA，FPS 512/128/64，2048 点）+ **DINO-ViT-S**（224×224，`F ∈ R^{512×14×14}` → flatten `512×196`）。文本 RoBERTa：`T_o 1×512`、`T_a 3×512`、`T_cf 1×512`。
- **训练**：65 epochs，batch 16，Adam lr 1e-4；曲率 `c` 学 `log(c)`，初始化 1.0；对比温度 `τ=0.07`。
- **数据**：[[PIAD]] v2（15,213 图 / 38,889 点云 / 43 物体类 / 24 affordance 类），指标 AUC↑ / aIOU↑ / SIM↑ / MAE↓。
  - 点云来源：3DIR + 3D-AffordanceNet + Objaverse；交互图像来源：AGD20K + OpenImages + 公开授权网络图。图像只标 affordance 类别，**不要求与 3D 实例严格一一对应**。
  - 逐点 affordance 热图稠密标注；图像侧弱标注（这也是「跨模态弱对齐」设定下 `L_cont` 能起作用的背景）。

> [!info] 张量维度（附录 Tab 6）
> | 张量 | 维度 | 含义 |
> |---|---|---|
> | `F` | 512×14×14 → flatten 512×196 | DINO-ViT-S 图像特征 |
> | `F_p` | **512×64** | PointNet++ 最后一层输出（注意：是 FPS 后的 64，不是 2048） |
> | `T_o` / `T_a` / `T_cf` | 1×512 / 3×512 / 1×512 | 物体几何 / 可供性意图 / 反事实 |
> | `F_α` | 512×2048 | 融合后的 affordance 表征（上采样回全点数） |
> | `φ` | 2048×1 | 输出热图 |

### 3.2 主结果（PIADv2, Tab 1）— 三 split 全项 SOTA

| 方法                | Seen AUC / aIOU / SIM / MAE       | Unseen Obj                        | Unseen Aff                        |
| ----------------- | --------------------------------- | --------------------------------- | --------------------------------- |
| Baseline          | 87.04 / 34.18 / 0.594 / 0.079     | 72.74 / 16.34 / 0.336 / 0.156     | 58.09 / 7.88 / 0.208 / 0.160      |
| FRCNN             | 87.05 / 33.55 / 0.600 / 0.082     | 72.20 / 18.08 / 0.362 / 0.152     | 59.08 / 7.96 / 0.210 / 0.156      |
| XMF               | 87.39 / 33.91 / 0.604 / 0.078     | 74.61 / 17.40 / 0.361 / 0.126     | 60.99 / 8.11 / 0.225 / 0.152      |
| IAG               | 89.03 / 34.29 / 0.623 / 0.076     | 73.03 / 16.78 / 0.351 / 0.123     | 62.29 / 8.99 / 0.251 / 0.141      |
| LASO              | 90.34 / 34.88 / 0.627 / 0.077     | 73.32 / 16.05 / 0.354 / 0.123     | 64.07 / 8.37 / 0.228 / 0.140      |
| GREAT             | 91.99 / 38.03 / 0.676 / 0.067     | 79.57 / 20.16 / 0.402 / 0.109     | 69.81 / 12.05 / 0.290 / 0.127     |
| GREAT(DINO-ViT-S) | 91.95 / 38.01 / 0.671 / 0.066     | 79.59 / 20.19 / 0.403 / 0.109     | 69.82 / 12.06 / 0.291 / 0.126     |
| **Ours**          | **93.29 / 38.98 / 0.688 / 0.060** | **81.01 / 21.18 / 0.430 / 0.090** | **70.90 / 13.26 / 0.320 / 0.113** |

![[images/tab1-piad2.png]]
*论文 Tab.1 原表：PIADv2 主结果，下划线行为换 DINO-ViT-S 的 GREAT。*

> [!note] 表中下划线行的含义
> 论文 Tab 1 里带下划线的 GREAT 行 = **把 ResNet 换成 DINO-ViT-S、其余不变**的重跑结果（91.95/38.01/0.671/0.066）。
> 即：**换更强的 2D backbone 几乎没给 GREAT 带来收益**（91.99→91.95），AAH 的 +1.3 AUC 不是靠 backbone 堆出来的。这个对照很关键。

- 相对 [[GREAT]]：Unseen Object MAE **0.109→0.090**（−17%）、Unseen Affordance SIM **0.290→0.320**（+10%）—— 增益集中在**泛化 split**，与「层级结构带来泛化」的主张一致。
### 3.3 PIAD v1（Tab 8，附录 B）— 增益不大且有回退

| 方法       | Seen AUC / aIOU / SIM / MAE               | Unseen Obj                                   | Unseen Aff                                   |
| -------- | ----------------------------------------- | -------------------------------------------- | -------------------------------------------- |
| IAG      | 82.88 / **18.88** / 0.544 / 0.098         | 68.49 / 7.22 / 0.344 / 0.139                 | 55.36 / 6.50 / 0.203 / 0.170                 |
| GREAT    | 85.22 / **19.61** / 0.569 / 0.093         | 69.41 / 7.49 / 0.352 / 0.127                 | 62.59 / 6.56 / 0.264 / 0.143                 |
| **Ours** | **86.97** / 17.24 / **0.575** / **0.091** | **70.22** / **7.56** / **0.363** / **0.122** | **64.16** / **6.76** / **0.283** / **0.135** |
|          |                                           |                                              |                                              |

- AUC/SIM/MAE 全胜，**aIOU 在 Seen 上反而低于 IAG 和 GREAT（17.24 vs 18.88 / 19.61）**。
- 相对 IAG 的提升幅度很大（Unseen Aff SIM +39.4%、MAE −20.6%），相对 GREAT 只 +2.5%~7.2% —— **越强的基线，AAH 的边际收益越小**。
- 解读：层级约束优化的是**热图的排序/分布质量**（AUC、SIM、MAE），不一定改善**阈值化后的重叠面积**（aIOU）。这在 PIADv2 上被 GREAT 的弱 aIOU 掩盖了，PIAD v1 才暴露出来。

### 3.4 消融（Tab 2 / 3 / 4 / 5）
- **组件（Tab 2）**：去掉 `L_cont` 掉最多（UnseenObj MAE 0.090→0.109）；去掉 `L_ord` 在 Unseen 上掉最明显（概念被压平成 flat similarity space）；去掉 `L_cf` 小幅下降；去掉超图（仅 backbone）也掉。

![[images/tab2-ablation.png]]
*论文 Tab.2 原表：逐组件消融（✗ 表示移除对应模块）。*
- **双曲 vs 欧氏（Tab 3）**：EU+MP（欧氏+均值池化）92.20 < EU+CL（欧氏+拼接线性）92.75 < **Ours 93.29**（Seen AUC）；Unseen Affordance 70.00 / 70.25 / **70.90**。递进验证双曲几何的必要性。
- **几何 vs 监督解耦（Tab 5）**：EU+Hierarch. 92.75 < Hyperbolic+Standard 92.50 < **Ours 93.29** → **增益不是只来自「更强的 loss」或「更好的融合」，几何本身有贡献**。
- **把 `L_cont` 加到 GREAT（Tab 4）**：提升很微弱（Seen AUC 91.99→92.34）→ 对比目标**只有在配上双曲结构表征时才真正生效**。
### 3.5 换 3D backbone（Tab 9，附录 B）

| 方法                     | Seen AUC / aIOU / SIM / MAE                   | Unseen Obj                                    | Unseen Aff                                    |
| ---------------------- | --------------------------------------------- | --------------------------------------------- | --------------------------------------------- |
| GREAT (DGCNN)          | 92.32 / 38.22 / 0.679 / 0.065                 | 79.86 / 20.42 / 0.409 / 0.108                 | 69.93 / 12.07 / 0.296 / 0.125                 |
| **Ours (DGCNN)**       | **93.70** / **39.00** / **0.689** / **0.060** | **81.29** / **21.39** / **0.431** / **0.090** | **70.96** / **13.30** / **0.326** / **0.112** |
| GREAT (Transformer)    | 92.91 / 38.64 / 0.682 / 0.063                 | 80.15 / 20.66 / 0.412 / 0.107                 | 70.24 / 12.10 / 0.299 / 0.123                 |
| **Ours (Transformer)** | **94.20** / **39.23** / **0.690** / **0.057** | **81.67** / **21.65** / **0.433** / **0.087** | **80.11** / **13.46** / **0.328** / **0.111** |

> [!warning] Tab 9 里 Ours(Transformer) 的 Unseen Affordance AUC = 80.11 大概率是笔误
> 同行的 aIOU 13.46 / SIM 0.328 / MAE 0.111 与 DGCNN 版（70.96 / 13.30 / 0.326 / 0.112）几乎一致 —— AUC 不可能在 aIOU、SIM 都只微增的情况下暴涨 **+10**。合理值应为 **70.11 左右**（把 8 看成 7）。**引用这组数时要小心**，别把它当成「换 Transformer 带来的巨大增益」。

- 排除该异常后：DGCNN / Point Transformer 下 AAH 在**同 backbone** 上一致优于 GREAT（Unseen Aff AUC +1.0 左右）→ 框架不绑特定 encoder，增益来自属性-可供性结构建模本身。
- **Prompt 噪声鲁棒性（Tab 10）**：注入 12.5% / 25% 噪声，Seen AUC 仅 93.29→93.02→92.63，性能几乎不掉 → 归因于 `L_cf` 教会模型**解耦 affordance 与辅助信息**。

### 3.6 开销（Tab 7）
| Model    | 推理时间     | 参数量        | 可训练参数    |     |
| -------- | -------- | ---------- | -------- | --- |
| IAG      | 1.426s   | 24.7M      | 24.7M    |     |
| LASO     | 1.336s   | 130.5M     | 130.5M   |     |
| GREAT    | 1.272s   | 256.7M     | 23.1M    |     |
| **Ours** | **2.1s** | **415.2M** | **150M** |     |

- 开销主要来自超图模块 + 双曲嵌入/损失计算；作者称因作用在紧凑特征上，相对 backbone 增幅有限。但**推理时间是 GREAT 的 1.65 倍**，是实打实的代价。

### 3.7 定性可视化（Fig 4 / 5 / 6）

![[images/fig4-visualization.png]]
*论文 Fig.4：Baseline / IAG / LASO / GREAT / Ours / GT 在三个 split 下的逐点热图对比。Seen 各方法差距不大；Unseen Object（钟/滑板/水桶）与 Unseen Affordance（lay/carry/pour）上，直接对齐类方法明显塌缩，Ours 的热图分布与 GT 最接近。*

![[images/fig5-vs-great.png]]
*论文 Fig.5：2D 侧 attention 热图对比（GREAT vs Ours）。GREAT 的注意力常散到人手或无关区域，Ours 更集中在真正承担交互的物体部件上。*

![[images/fig6-more-visual.png]]
*论文 Fig.6：更多定性结果（前两行 Seen，中间两行 Unseen Object，后四行 Unseen Affordance）。值得注意 Unseen Affordance 的 pour（水壶嘴）、cut（刀刃）等结果——热图落在的是**几何上合理**的交互区域，而非训练高频区域，与"属性层级带来泛化"的主张一致。*

---

## 4. 评价

> [!tip] 亮点
> - **切入点新且站得住**：把 affordance 定位从「跨模态对齐」重新表述为「**层级蕴含**」问题，并用双曲几何的原生性质（指数体积增长）来承载它 —— 这在 3D affordance 领域是少见的表征视角。
> - **反事实 prompt 是低成本高收益的增量**：只加一条 prompt 5 + 一个 `L_cf`，换来噪声鲁棒性（Tab 10）和 Unseen 上的稳定增益，可迁移性强（任何用 MLLM prompt 的 affordance 方法都能套）。
> - **消融做得扎实**：几何 vs 监督解耦（Tab 5）、把 `L_cont` 移植到 GREAT（Tab 4）这两组对照实验， effectively 挡住了「是不是只是 loss 更强」的质疑。
> - 层级设计真正闭环到 3D：`L_ord^{a→P}` 把 affordance 语义压进点云特征的蕴含锥内，不是只做语义侧的自嗨。

> [!warning] 局限（论文 Conclusion 自述 + 我的观察）
> - **依赖文本 prompt 质量**：模糊/不完整的属性描述会削弱所建层级（开放词表下更甚）。
> - **双曲蕴含约束带来额外几何计算**，扩展到超大规模点云时的可扩展性存疑。
> - **反事实 prompt 是人工设计的**；自动生成多样且可靠的反事实条件仍是开放问题。
> - **跨论文对比需谨慎**：PIADv2 aIOU 上 AAH（38.98 / 21.18 / 13.26）**略低于 [[HAMMER]]（40.06 / 24.28 / 13.28）**，两者都自称 SOTA，比较时务必对齐 split 与 backbone 设置。PIAD Seen aIOU 甚至低于 GREAT（17.24 vs 19.61）—— 说明增益主要在 AUC/SIM 这类**排序/分布**指标，而非阈值化的重叠指标。
> - 未做 corrupted / 旋转鲁棒性评测（[[HAMMER]] 做了），泛化结论只建立在 PIAD 系列 split 上。
> - **数值可信度**：Tab 9 的 Ours(Transformer) Unseen Aff AUC 80.11 疑为笔误（详见上文）；全文未报告方差 / 多次运行；PIAD v1 上 aIOU 的回退在正文被略过，只在附录 Tab 8 出现。
> - **层级是人为先验**：「属性 ⊃ 可供性 ⊃ 3D 区域」这个偏序是作者定义的，不是从数据中学出来的；如果属性 prompt 里 MLLM 给出的描述与该偏序冲突，蕴含锥约束反而可能**压错**表征。
>
> [!question] 属性必须走 LLM 吗？—— 论文没回答的问题
> - **缺关键基线**：论文**没有**「去掉 `T_o` / 去掉属性 prompt」的消融（Tab 2–5 只消融损失项、超图、双曲 vs 欧氏）→ **属性分支到底值多少分，作者自己没量化过**。这是评估「能否去掉 LLM」的第一件事该补的。
> - **属性是冻结的**：附录 A 明确 P1–P5 答案经 RoBERTa 编码成 `T_o`，梯度**不回传**到 LLM → 属性表征是**给定的**而非学出来的；且每张交互图都要跑一次 MLLM（PIADv2 有 15,213 张图），数据准备成本不小。
> - **属性文本高度冗余**：同类物体（如所有 mug 的把手）的几何描述几乎一致，per-image 调 MLLM 的**边际收益可能很有限** → 缓存/蒸馏的性价比很高。
> - **属性可拆成两部分**：①**几何可见信息**（形状、尺寸、空间）→ 图像/点云里直接可提；②**功能语义**（为何能支撑、能抓）→ 这部分才真正需要世界知识。去掉 LLM 丢的主要是 ②。
> - **去掉 LLM 后 `L_cf` 怎么办**（最关键的障碍）：反事实不必靠文本造，可改成 ——
>   - **区域 mask**：把属性对应的部件区域在图像/点云上置零，重提属性 → 天然的反事实对，完全不需要 P5；
>   - **跨样本 swap**：用同类/异类物体的属性向量替换当前属性（hard negative），要求 affordance 落在其锥外；
>   - **特征扰动**：沿属性向量的反方向加噪或做 dropout。
> - **推荐先做蒸馏（风险最低）**：训练期用文本属性当 teacher，训一个轻量视觉属性头（可学习 attribute query + cross-attn）回归/对齐 `T_o`；推理期丢掉 MLLM + RoBERTa。这样能直接量出「去掉 LLM 掉多少点」。
> - 另一条更激进的路线：**点云侧直接算几何算子**（曲率、平面度、局部厚度、环形拓扑/把手性、对称性、尺寸比）当属性 —— 与论文宣称的「几何结构属性」最贴，但泛化受限于手工算子设计。

---

## 5. 对我们的启发（可借鉴点）

> [!question] 能拿来做什么？
> 1. **反事实 prompt 机制**：我们的 [[3D-native HOI Affordance 换输入质疑与防御]] 方案若用 MLLM 提属性描述，可照搬「prompt 5 + `L_cf`」来抑制虚假相关，成本很低。
> 2. **双曲蕴含锥做层级约束**：`φ(h) = arcsin(2K/(√c‖h̃‖))` + exterior angle（附录 Eq.19）是现成模块，可用于任何「抽象语义 → 具体 3D 区域」的偏序建模（不止 affordance）。
> 3. **超图做局部关系建模**：α-space + 距离阈值 `ε` 构图非常轻，可作为点云/图像 token 的 relation-aware 前置模块替代 GNN / attention 的一种选择。
> 4. **值得质疑的点**：属性层级是**人为定义**的（属性 → affordance），不是数据驱动学出来的；双曲的优势在所有 split 上约 +0.5~1.5 AUC，**幅度不算大**，需要判断这个复杂度（2.1s 推理）是否值得。
> 5. **把属性从 LLM 里解放出来（对组内方向最直接）**：我们现在若要走 3D-native 路线，不该默认「属性 = MLLM 文本」。可行顺序是 —— 先补 `w/o T_o` 基线性化属性贡献 → 再用**区域 mask / 跨样本 swap** 造非文本反事实 → 最后用**蒸馏**把文本属性压进视觉属性头，推理期零 LLM。详见 §4 的「属性必须走 LLM 吗」。
> 6. **几何算子即属性**：点云侧可直接算曲率、平面度、局部厚度、环形拓扑（把手性）、对称性 —— 这些是**真几何属性**，比从 RGB 猜更可靠，也不需要任何语言模型。适合作为 3D-native 方案的属性来源。

## 6. 相关
- 领域地图：[[3D Affordance Grounding]]
- 数据集：[[PIAD]]
- 直接基线 / 同路线：[[GREAT]]（prompt 1–4 的来源，主对比对象）、[[HAMMER]]（aIOU 上更强）、[[InteractVLM]]、LASO、OpenAD、IAGNet
- 技术组件：DINO-ViT（见 [[DINOv2]]）、PointNet++、RoBERTa、Lorentz/Hyperbolic NN（Nickel & Kiela 2017; Ganea et al. 2018）、Hypergraph NN（Feng et al., AAAI19）
