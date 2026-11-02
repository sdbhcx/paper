---
title: "AffordAny: Open-World 3D Affordance Grounding from Monocular RGB Images via Vision-Language-Guided Geometric Reasoning"
aliases:
  - AffordAny
  - AffordAny 3D affordance
  - Affordance Anything
authors:
  - Junqi Wu
  - Kaihua Tang
  - Xuanwen Chen
  - Hongzhi Li
  - Jianqiang Huang
  - Xian-Sheng Hua
venue: arXiv preprint arXiv:2608.20720v1 [cs.CV] (2026-08-21)
affiliation:
  - Tongji University
  - Chinese Academy of Sciences
year: 2026
tags:
  - paper
  - 3D-affordance
  - open-world
  - open-vocabulary
  - language-driven
  - monocular-reconstruction
  - SAM3D
  - 3DGS
  - cosmos
  - VLM
  - pseudo-label
  - benchmark
  - related-work
created: 2026-09-21
code: https://github.com/lzlfwow/AffordAny
dataset: https://modelscope.cn/datasets/lzlfwow/AffordAny
pdf: paper/1-inbox/AffordAny.pdf
bib: |
  @article{wu2026affordany,
    title={AffordAny: Open-World 3D Affordance Grounding from Monocular RGB Images via Vision-Language-Guided Geometric Reasoning},
    author={Wu, Junqi and Tang, Kaihua and Chen, Xuanwen and Li, Hongzhi and Huang, Jianqiang and Hua, Xian-Sheng},
    journal={arXiv preprint arXiv:2608.20720},
    year={2026}
  }
---

# AffordAny

> [!abstract] 摘要
> **AffordAny**（*Affordance Anything*）做的是 **open-world 3D affordance grounding**：给定**一张单目 RGB 图** + **一句自由文本指令**，在重建出的物体 3D 几何上定位「该指令指向的功能部件」。
>
> 针对现有方法**三条硬伤**——(1) 训练/推理都假定已有 3D 几何、(2) 缺少有真实图像来源且带 3D 部件级监督的可扩展数据、(3) CLIP/哈希文本表征对**指令改写**脆弱——论文给出对应的三件套：
> 1. **全自动数据构建流水线**（LVIS → SAM 3D 重建 → 多视角渲染 → LLM 提部件/指令 → 2D 分割 → 多视角 3D 标签提升），产出 **473 类 / 5,334 物体 / 10,633 部件样本 / 678 种部件名 / 31,899 条指令** 的基准，类别多样性比先前工作高**一个数量级**；
> 2. **VLM 引导的跨模态 3D affordance 解码器**：冻结 **Cosmos-2B**，通过 **投影注入 → 指令条件语义压缩 → 双向几何-语义交互（GPBlock）** 把图像-语言证据融进点云，**不微调 VLM**；
> 3. **伪标签自训练**（最小扰动训练 + 事后 Platt 校准），无需人工标注即可新增物体、提升类别级泛化。
>
> 结果：unseen object **IoU 0.428**、unseen category **IoU 0.315**（自训练后），unseen-category **mIoU 相对提升 6.3%**（p<0.01），**指令敏感度差仅 0.105**（OpenAD 0.166，LASO/LMAffordance3D >0.34）。

---

## 0. 一句话速览

| 维度 | 内容 |
|------|------|
| **任务** | 单目 RGB 图 + 物体 mask + 自由文本指令 → 3D 点/高斯逐点 affordance 概率 |
| **关键转变** | 从「已有 3D 资产 + 闭集本体」→「**原始 RGB 观测 → 自动建 3D 监督 → 开集语言查询**」 |
| **两大支柱** | ① 自动化 benchmark（数据）② 冻结 VLM + 轻量解码器（方法）③ 伪标签自训练（泛化） |
| **可训练参数** | 解码器仅 **10.54M**（VLM 冻结），BF16 单次前向中位 **16.5 ms** @A100 |
| **最亮眼的点** | 指令改写鲁棒性（ΔIoU 仅 0.105）+ 尾类泛化提升（≤50 样本类 +0.012） |
| **代码/数据** | [GitHub](https://github.com/lzlfwow/AffordAny) / [ModelScope](https://modelscope.cn/datasets/lzlfwow/AffordAny) 均开源 |

---

## 1. 研究背景与动机

### 1.1 任务定位

3D affordance grounding 要回答「**在哪里、以什么方式**与物体交互」——如马克杯的**把手**、盒子的**盖子**、椅子的**坐面**。这是具身智能体做 **manipulation / navigation / task planning** 的先决条件。

现实部署天然是 **open-world** 的：
- 遇到**长尾物体**；
- **部件分类体系多样**（不同物体怎么切分部件没有统一定义）；
- **自然语言指令不受约束**；
- 观测通常只是**原始 RGB 图像**，而非现成的 3D 资产。

### 1.2 现有方法为什么不够（三条硬伤）

> [!question] 论文明确列出的三条 gap
> **① 几何依赖**：LASO、LMAffordance3D、OpenAD 等方法**在训练与推理时都假定已有关注物体的 3D 几何**。这把真实部署问题中最难的部分（从 RGB 拿到 3D）直接绕开了。
> **② 数据稀缺**：具备**真实图像来源 + 3D 部件级监督**的可扩展数据集仍然稀缺；当监督语义用**自由文本指令**（而非固定闭集本体）表达时更是如此。
> **③ 文本编码脆弱**：基于 CLIP 或哈希 embedding 的文本编码对**指令改写（paraphrasing）脆弱**——「grab the handle」vs「reach for the grip」就掉点。这是语言条件交互的致命伤。

> [!info] 相关方法谱系（论文视角）
> - **语言驱动 3D affordance**：LASO（文本 query 双向融合做部件分割）、LMAffordance3D / AGPIL（affordance triple + CLIP 空间特征）、OpenAD（文本 embedding 与 3D 点特征余弦对齐，**开词表**）
> - **3D affordance 数据集**：3D AffordanceNet（点级概率标签）、AGD20K（真实交互图）、AffordPose（部件 + 手部姿态）、GAPartNet、SceneFun3D、AffordBridge / AffordMatcher（真实 3D 场景）
> - **VLM → 3D**：ULIP / OpenShape / Uni3D（图文先验迁移到点云）、PartSLIP（低样本 3D 部件分割）、3D-LLM / PointLLM / LEO（3D + LLM）、LISA（把特殊 token 解码成 2D mask，**本文 `<AFF>` token 的灵感来源**）
> - **开词表 grounding**：CLIP、Grounding DINO、ODISE、VLPart、SAM、PixelLM
> - **开词表 3D**：PLA、OpenScene、OV3D、OpenMask3D、Feature 3DGS、LangSplat、OpenGaussian

### 1.3 本文的定位

> [!info] 一句话：**不做「3D 资产的 affordance」，做「从一张照片长出 3D 并给出 affordance」**
> 用 2D 基础模型（SAM 3D / SAM3 / LLM）**只用来挖掘监督信号**，通过多视角融合把标签「提升」到 3D；最终解码器**直接在点几何上预测**，条件是**冻结 VLM 特征**。

---

## 2. 四条核心贡献

1. **首个自动化流水线**：从**真实图像**构建大规模、文本条件化的 **3D 部件级 affordance 监督**，产出 473 类 / 5,334 物体 / 10,633 部件样本的基准。
2. **VLM 引导解码器**：把 **Cosmos-2B** 特征与 3D 点云几何有效融合，做开词表 affordance grounding，**VLM 全程冻结、无梯度**。
3. **伪标签自训练**：**minimal perturbation training + Platt calibration**，在不增加人工标注的前提下显著提升类别级泛化。
4. **系统性泛化评测协议**：把 **unseen objects / unseen categories / unseen instruction paraphrases** 三条轴**拆开单独度量**，避免三种「新」混在一个分数里。

---

## 3. AffordAny Benchmark：数据构建流水线

### 3.1 五步流水线

```
LVIS 训练集 (1,270,141 instances / 1,203 类)
      │  ① 类别过滤（LLM 判断该类是否通常含「可操作/可交互区域」）→ 677 类
      │  ② 图像级过滤 + √n 类别配额（抑制头类主导）→ 7,407 候选 / 473 类
      ▼
候选物体 7,407 → 完成物体级处理 7,398
      │  ③ 单图 3D 重建：SAM-3D-Objects → object-centric 3D Gaussian + 相机参数
      │  ④ 6 个规范视角渲染 512×512（front/right/back/left/top/bottom）
      │  ⑤ LLM 部件发现（≤5 候选，规范化为 "the {part} of the {object}"）
      │  ⑥ 文本条件 2D 部件分割（SAM3）
      │  ⑦ 多视角 3D 标签提升（多视角投票 + 分组 + 补洞）
      ▼
5,334 物体 / 10,633 已验证部件 / 678 规范部件名
      │  ⑧ 每条部件生成 3 条同义英文指令（≤14 词）
      ▼
31,899 条 instruction-level 样本
```

### 3.2 关键工程细节（可复现性）

> [!tip] 细节里藏着不少「能直接抄」的工程参数
> **类别过滤**：结构化语言模型判断（"该类别是否通常包含有意义的人机交互/可操作区域"）。
> **实例过滤**：以 masked object 为主证据、原图为弱上下文；要求可辨识、不模糊、未被严重裁切、可交互。模型 `gemini-3.1-flash-lite-preview`，JSON 约束输出，60s 超时，最多 5 次重试。
> **确定性几何前置检查**：mask 面积占比 ≥ 0.03、长宽比 ≤ 4.0、与最多 1 个图像边界接触（0.02 图像尺寸 margin，下限 2 px）。
> **类别配额**：目标配额 ∝ `n_c^0.5`（n_c 为该类合格 LVIS 图数），每类至少 1 个；余量按**最大余数法**分配（floor 后按小数余数排名）。
> **渲染相机**：4 个水平视角 yaw = 0°/90°/180°/−90°，top/bottom pitch = 89°/−89°；radius 2.0、FOV 40°；渲染前固定预旋转把源 +z 对齐到归一化 (1,1,1) 对角方向。
> **2D 分割**：SAM3，processor 置信阈 0.5，保留 mask 分数下限 0.1，每部件最多留 3 个 mask；**任何视角都没成功 mask 的部件被丢弃**。
> **3D 标签提升**：高斯中心按保存的相机投影，**深度归属**决定每个像素可见的高斯；footprint 半径 = 1.75 × 投影高斯尺度（上限 12 px），深度容差 = max(0.75 × 尺度, 1e-3)，footprint 至少覆盖 0.02 的 mask 面积才投正票。**每个高斯的分数 = 正票视角数 / 该高斯可见的视角数**。
> **融合**：分数阈值 0.3、歧义 margin 0.05；当两部件在较小 mask 上的重叠 >0.5 且正点数相对差 ≤0.3 时判重复并剪枝。
> **补洞**：K = 32 邻居，≥2 个支持邻居、支持率 0.3，≤32 次迭代，传播距离上限 = 7 × 邻居间距中位数。**仍然不可见/低于阈值的点保持 unknown 并从监督中排除**。

### 3.3 与已有数据集对比（Table 1）

| 数据集 | 年份 | 来源 / 输入 | 标注 | 规模 | Obj. | Part | Open Part | Obj. | Cat. |
|--------|------|-------------|------|------|------|------|-----------|------|------|
| 3D AffordanceNet | 2021 | CAD / 点云 | 3D heatmap | 23K shapes | 23 | 18 | ✗ | ✓ | ✗ |
| GAPartNet | 2023 | 铰接 CAD / 点云+mesh | 3D part mask + pose | 8,489 parts | 27 | 9 | ✗ | ✓ | ✓ |
| AffordPose | 2023 | CAD / mesh + affordance | 3D part + hand pose | 26.7K inter. | 13 | 8 | ✗ | ✗ | ✗ |
| LASO | 2024 | CAD / 点云 + 语言 | 3D part mask | 19,751 pairs | 23 | 17 | ✗ | ✓ | ✓ |
| AGPIL / LMAffordance3D | 2025 | CAD+图像 / 图像+点云+语言 | 3D heatmap | 30,972 trip. | 23 | 17 | ✗ | ✓ | ✓ |
| 3DAffordSplat | 2025 | 3DGS / 点云+3DGS | 3D heatmap + 3DGS | 6,631 labels | 21 | 18 | ✗ | ✓ | ✓ |
| SceneFun3D | 2024 | 真实扫描 / 场景 + 语言 | 3D part + motion | 14.8K ann. | – | 9 | ✗ | ✓ | ✗ |
| AffordBridge (AffordMatcher) | 2026 | 真实扫描 / 场景 + 图像 | 3D interaction labels | 291,637 ann. | 157 | – | ✗ | – | – |
| **AffordAny (Ours)** | **2026** | **真实图像 / 图像** | **3D heatmap + 3DGS + text** | **31,899 pairs** | **473** | **678** | **✓** | **✓** | **✓** |

> [!success] 这张表想传达什么
> 前作的物体类别数都在 **13–27** 量级（AffordBridge 例外，157 但部件维度为空）；AffordAny 直接把类别拉到 **473**、部件类型拉到 **678**，并且是**唯一三个 Open 都打勾**的（开放部件词表 + 未见实例 + 未见类别）。

### 3.4 划分语义（Table 7）

| 划分 | 用途 | Inst. rows | Objects | Cats. | Unique parts |
|------|------|-----------|---------|-------|--------------|
| Training | 优化 | 12,134 | 3,049 | 331 | 511 |
| Unseen instruction | 改写鲁棒性 | 6,067 | 3,049 | 331 | 511 |
| Validation / Unseen object | 验证 | 710 | 351 | 159 | 233 |
| Validation / Unseen category | 验证 | 1,515 | 764 | 71 | 138 |
| Test / Unseen object | 测试 | 789 | 391 | 209 | 275 |
| Test / Unseen category | 测试 | 1,552 | 779 | 71 | 140 |
| **合计** | | **22,767** | **5,334** | **473** | **678** |

> [!note] 划分逻辑
> **类别**：70% 见（训练），15%/15% 留作 unseen-category 验证/测试。
> **物体**：在见到的类别内，实例按 80%/10%/10% 分 train/val/test 做 unseen-object 评测。
> **指令**：每个物体-部件对 3 条同义指令，**index 1/2 用于优化，index 0 专门留给 matched 的 unseen-instruction 评测**——这样「换措辞」的敏感度与「换几何」的泛化**不会被混在一起**。

---

## 4. 方法：VLM-Guided 3D Affordance Decoder

### 4.1 问题设定

给定 RGB 图 `I`、物体 mask `M`、自由文本指令 `l`：
1. 先用 SAM 3D 重建 **object-centric 点云** `P = {p_i}_{i=1}^N = SAM3D(I, M)`
2. 预测**逐点分数** `ŝ ∈ [0,1]^N`，标示 `l` 指定的功能区域

```
             ┌───────────── 冻结 (offline cache) ─────────────┐
  I(masked) ─┤ Cosmos-2B VLM ├─▶ h_aff ∈ R^2048   (指令条件全局语义)
  l + <AFF> ─┘               └─▶ F_v  ∈ R^{144×2048}  (12×12 图像 token 网格)
                                          │
  P (4096 点, F∈R^{N×13}) ─▶ Point Encoder ─▶ P ∈ R^{N×256}
                                          │
     ①投影注入: P ← P + MLP( BilinearSample(F_v, u, v) )   ← 用重建相机把 3D 点投到 2D
                                          │
     ②语义压缩: 16 个可学习 prototype ──FiLM(h_aff)──▶ TransDec(Q=Z̃, KV=[h_aff;F_v]) ─▶ Z ∈ R^{16×256}
                                          │
     ③GPBlock 双向交互: Group(Z←P) → Mix → Ungroup(P←Z) ─▶ Z′, P′
                                          │
     ④指令引导解码: H = TransDec(Q=P′, KV=[h_aff; Z′]) ─▶ ŝ = σ(Head(H))
```

> [!info] 核心原则（论文原话）
> **图像-语言证据负责「是哪个功能部件」，几何负责「它的空间范围」。**
> Image–language evidence identifies the functional part, whereas geometry constrains its spatial extent.

### 4.2 Offline VLM Feature Extraction（离线冻结特征）

- 冻结编码器 = `nvidia/Cosmos-Reason2-2B`，隐藏宽度 `D = 2048`
- 输入：**mask 外置白**的源图 + 「指令 + `<AFF>` 标记」
- `<AFF>` token **只加一次**到 tokenizer，初始化为其**已有 token embedding 的均值**；标记 embedding 与 VLM 权重**都不更新**
- 取标记位置的最后层隐状态缓存为 `h_aff ∈ R^2048`（**指令相关的语义摘要**）
- 源图 token 池化为 **12×12 网格** → `F_v ∈ R^{144×2048}`（**保留空间布局**）
- 投影坐标与可见性 flag 与特征一起缓存，保证每次解码器运行**看到同一份点-图对应关系**

> [!tip] 为什么要冻结？
> (1) 昂贵的跨模态编码器**可以复用于多个解码器变体**；
> (2) **避免把 backbone 适配到 seen-category 分布**——这正是 open-world 泛化的关键；
> (3) 论文强调「**没有 VLM 参数收到梯度**」，所有报告的性能提升都来自解码器与其对缓存特征的融合方式。

> [!warning] 「instruction-conditioned」≠「可训练」
> 缓存语义里，**"instruction-conditioned" 指冻结 VLM 产出的上下文状态**；"learnable" 指**解码器侧的 prototype 与投影**。两者不要混。

### 4.3 Point Encoder

- 点特征 `F ∈ R^{N×13}` + 坐标 `X` → 投影到 `d = 256` → 相加 → **3 个残差 MLP block（width 256）** → 点 token `P`
- 145 个 VLM token `[h_aff; F_v]` 投影到同一维度
- **分开编码**：几何线索一直保留到融合阶段才与语义相遇

### 4.4 Spatial Correspondence via Projection Injection（投影注入，Eq. 1）

用重建相机把每个 3D 点投到 `(u, v)`，从 reshape 成 12×12 的 `F_v` 里**双线性采样**，把局部特征加到点 token 上：

```
P ← P + MLP( BilinearSample(F_v, u, v) )
```

- 让**细长/部分可见区域**能直接拿到其投影位置处的图像证据
- 对比**全局池化**：注入**保留了「单个 3D 点 ↔ 源图区域」由重建推导出的对应关系**，而不是糊成一个全局向量

### 4.5 Instruction-Conditioned Semantic Compression（语义压缩，Eq. 2–3）

把 145 个 VLM token 压成 `K = 16` 个**指令条件化 prototype**：

- 可学习 embedding `{z_k}` 被 `h_aff` 用 **FiLM** 调制：
  `z̃_k = z_k ⊙ (1 + γ) + β`，其中 `γ = W_γ h_aff`，`β = W_β h_aff`，`W_γ, W_β ∈ R^{d×D}`
- 条件化 prototype 用 **2 层 Transformer decoder** 去 attend VLM 序列：
  `Z = TransDec(Q = Z̃, KV = [h_aff; F_v]) ∈ R^{K×d}`

> [!tip] 为什么这个瓶颈有用
> 紧凑表示让后续的 **点-prototype 注意力复杂度与 VLM 序列长度解耦**（O(N·K) 而不是 O(N·145)）。
> 论文强调：这是**任务感知的语义瓶颈**，不是 VLM token 的通用全局池化；**FiLM 在它读完整序列之前就让这个固定大小的瓶颈变成指令相关的**。

### 4.6 GPBlock：双向几何-语义交互（Group–Mix–Ungroup）

| 阶段 | 操作 | 作用 |
|------|------|------|
| **Group** | prototype 做 Query、point 做 KV（`Q=Z, KV=P`） | 让语义**吸收 3D 上下文** |
| **Mix** | token-based 与 FFN-based 通道混合 | 协调 prototype 之间 |
| **Ungroup** | point 做 Query、精炼后的 prototype 做 KV（`Q=P, KV=Z`）+ FFN | 把**已落地的上下文回传给每个点** |

产出：几何感知 prototype `Z′` 与增强点 `P′`。

> [!info] 双向性的意义
> **Group 把语义锚在形状上，Ungroup 把锚定的上下文送回每个点。**
> 这种双向性支撑了「**同一条指令在不同类别上指代不同几何实现**」时的迁移（例如「handle」在马克杯 vs 抽屉上是完全不同的形状）。

### 4.7 Instruction-Guided Decoding（Eq. 4–5）

- 增强点 token 用 **8 头、4 层点→语义 Transformer cross-decoder** 去检索 affordance token 与精炼 prototype：
  `H = TransDec(Q = P′, KV = [h_aff; Z′])`
- 轻量 head 把 decoded token 映射为逐点 logit，推理时过 sigmoid：
  `ŝ_i = σ(Head(H_i))`
- Head 结构：`LayerNorm → Linear → GELU → Dropout → Linear(1)`，dropout 0.1
- 最终 cross-attention 让**每个点既取到全局指令线索、又取到其几何精炼后的语义 prototype**

### 4.8 Instruction Dropout

- 训练时**随机把 `h_aff` 和 `F_v` 置零**，逼迫模型不依赖语义捷径、强化几何先验
- 副作用是**测试指令与训练 query 词汇不同时的鲁棒性提升**（直接对应 Table 3 的 ΔIoU 0.105）
- AffordAny 的 instruction dropout = **0.2**，评测时关闭

### 4.9 训练目标与超参

```
L = λ_bce · L_bce + λ_focal · L_focal + λ_dice · L_dice
```

| 损失 | 权重 | 负责任么 |
|------|------|----------|
| weighted BCE | 0.2 | **逐点概率校准**（positive weight = 8.0） |
| focal | 1.0 | **困难少数点**（α = 0.75, γ = 2.0） |
| Dice | 0.5 | **稀疏区域的重叠** |

| 超参 | 值 |
|------|-----|
| 优化器 | AdamW，8 GPU，per-GPU batch 16，AMP |
| 梯度裁剪 / 权重衰减 | 1.0 / 5e-4 |
| LR 调度 | 5 warmup epochs，cosine `2e-4 → 1e-5` |
| 训练轮数 / seed | **300 epochs** / 42 |
| checkpoint 选择 | 每 10 epoch 评一次，按 **unseen-object 验证 IoU**；入选 **epoch 140** |
| 样本配置 | 每样本 **N = 4,096** 采样高斯点，每点 **13** 维缓存几何/外观特征 |

---

## 5. Pseudo-Label Self-Training（伪标签自训练）

> [!info] 设计哲学：**把「分割变化」与「概率校准」分开处理**
> 用训练好的解码器当 teacher，对其预测做事后校准；自训练只负责**加几何多样性**，不负责改语言词表。

### 5.1 伪标签生成

- 处理 **5,325 个与监督基准不重叠的 LVIS 物体**
- teacher 产出 **32,718** 条 object–part–instruction 伪记录（10,906 条 index-0 + 21,812 条 index-1/2）
- 概率落在 **[0.15, 0.65]** 区间的点**忽略**，其余二值化
- seed 42 固定 **0.37 子采样** → 保留 **12,105** 条，与 **12,134** 条真实记录形成近似 **1:1** 混合
- **忽略带**只保留有把握的前景/背景；**子采样**防止噪声伪标签主导真值监督

### 5.2 Minimal Perturbation Training（最小扰动训练）

- 从**监督模型**出发，**只跑一轮**：伪标签权重 `w̄ = 0.1`，BCE/Focal 再乘 0.5 → **实际权重约 5%**
- 自训练阶段配置：5 GPU、**100 epochs**、batch 16/GPU、LR `5e-5`、10 warmup、**不开混合精度**；loss 权重 `(λ_bce, λ_focal, λ_dice) = (0.2, 1.0, 0.7)`
- 评测 epoch 1 与之后每 10 epoch，按 unseen-object 验证 IoU 选 checkpoint

> [!warning] 为什么是「最小扰动」
> 论文的预实验发现：**更激进的混合虽然能提升阈值化重叠，但会扭曲预测概率分布**。所以选择「一轮、低权重」——既让解码器见到更多**类内几何**，又限制对已监督表征与概率校准的漂移。

### 5.3 Post-hoc Platt Calibration（事后 Platt 校准，Eq. 7/9）

```
ŝ_i^cal = σ(a · z_i + b)        # 验证集选 (a, b) = (1.10, −0.75)
```

- `z_i` 是自训练后的**原始 logit**；`a` 调**锐度**，`b` 修**teacher 的正样本率偏置**
- 效果：**恢复 MAE**，但**部分抵消 mIoU 增益**；**AUC 不变**（单调变换保序）
- 论文**同时报告 raw 与 calibrated** 结果，把「重叠-校准权衡」与「调阈值带来的收益」分开

---

## 6. 实验

### 6.1 评测协议（三轴 / 六划分）

| 泛化轴 | 含义 | 规模 |
|--------|------|------|
| **Unseen object** | 同类别、新实例 | test 789 / val 710 |
| **Unseen category** | 全新物体类型 | test 1,552 / val 1,515 |
| **Unseen instruction** | 同物体同部件、换措辞 | 6,067 条（每部件 index 0） |

> [!success] 这个协议的价值
> 它**阻止了「实例新」「类别新」「措辞变」被混进同一个分数**。指令鲁棒性评测刻意**匹配物体与目标部件**，保证配对评估之间**只有措辞变化**。

### 6.2 指标

| 指标 | 定义 | 视角 |
|------|------|------|
| **IoU** | 阈值 0.5 的 pooled 逐点 IoU | 主要定位指标（阈值相关） |
| **mIoU** | **先按类别**池化交并、再对类别取平均（各类分母 clamp ≥1） | 头尾类**等权** |
| **AUC** | 全局逐点 ROC AUC（rank-sum） | **无阈值**的排序能力 |
| **MAE** | 逐点平均绝对误差（用**二值标签**，不用软热图） | 概率误差 |
| **SIM** | 行内单位质量归一化后的**直方图交集**，再对行平均 | 分布一致性（保留软目标） |

> [!note] 为什么要一起报
> IoU/mIoU **依赖阈值**；AUC/SIM 提供**排序与分布**的互补视角。
> pooled IoU 反映**聚合的逐点表现**，macro mIoU 给每类**相同权重**——两个一起报，就能**暴露出被高频类别主导的「虚高」**。

### 6.3 主结果（Table 2，300 epochs）

| Model | Best Val | UI: IoU | UI: mIoU | UI: AUC | UI: MAE↓ | UI: SIM | UO: IoU | UO: AUC | UC: IoU | UC: mIoU | UC: AUC |
|-------|----------|---------|----------|---------|----------|---------|---------|---------|---------|----------|---------|
| OpenAD | .413 | .484 | .390 | .878 | .255 | .360 | .362 | .783 | .266 | .234 | .683 |
| LASO | .453 | .451 | .320 | .853 | .205 | .374 | .394 | .820 | .266 | .227 | **.723** |
| LMAffordance3D | **.454** | .469 | .330 | .852 | .188 | .387 | .418 | .820 | .244 | .211 | .669 |
| **AffordAny (Ours)** | **.473** | **.680** | **.630** | **.956** | **.112** | **.604** | **.428** | **.829** | **.305** | **.243** | .716 |

> [!success] 关键读数
> - **Unseen instruction IoU 0.680**（远超次优 0.484）——指令改写场景下的巨大优势
> - **Unseen object IoU 0.428 vs 0.418**，**unseen category IoU 0.305 vs 0.266**
> - 同时在 **UO AUC / UO SIM / UC SIM** 上最佳
> - **LASO 保住 unseen-category 的 AUC 与 MAE**（.723 / .264）——说明**重叠、集中度**与**排序、校准**是两回事

> [!warning] 论文的自我克制（值得学习）
> 作者明确声明：**不宣称全面碾压**。失败案例包括 LMAffordance3D 在 unseen object 上有竞争力、但掉到 **unseen category IoU 0.244**。所以本文的**主要收益是「类别漂移下更好的阈值化定位与空间集中度」**。

### 6.4 指令鲁棒性（Table 3，unseen − seen，绝对值越小越好）

| Model | ΔIoU | ΔmIoU | ΔAUC | ΔMAE | ΔSIM |
|-------|------|-------|------|------|------|
| OpenAD | −0.166 | −0.166 | −0.087 | 0.075 | **−0.065** |
| LASO | −0.342 | −0.429 | −0.135 | 0.133 | −0.323 |
| LMAffordance3D | −0.360 | −0.466 | −0.140 | 0.129 | −0.351 |
| **AffordAny** | **−0.105** | **−0.108** | **−0.031** | **0.037** | −0.078 |

> [!success] 结论
> AffordAny 在 **IoU / mIoU / AUC / MAE** 四个 gap 上都最小（|ΔIoU| = 0.105，OpenAD 0.166，LASO 与 LMAffordance3D > 0.34）。OpenAD 的 SIM gap 略小（0.065 vs 0.078）。
> 因为**重叠、排序、误差指标上同时占优**，这个优势**不能归因于某一个讨巧的决策阈值**。

### 6.5 消融（Table 4，200 epochs）

| 配置 | UO IoU | UO AUC | UO MAE | UC IoU | UC AUC | UC MAE |
|------|--------|--------|--------|--------|--------|--------|
| **Full model** | **.428** | **.829** | **.212** | **.305** | **.716** | **.266** |
| w/o Proj. Inj. | .416 | .803 | .211 | .275 | .691 | .277 |
| w/o Sem. Compr. | .398 | .793 | .219 | **.245** | .653 | .276 |
| w/o GPBlock | .403 | .808 | .218 | .270 | .686 | .273 |
| w/o Instr. dropout | .402 | .807 | .217 | .288 | .694 | .273 |

> [!info] 消融结论（按 unseen-category IoU 贡献排序）
> **语义压缩 −0.060 ≫ GPBlock −0.035 > 投影注入 −0.030 > instruction dropout −0.017**
> - **类别迁移需要「指令专属瓶颈」+「与局部 3D 证据的交互」**——两者缺一不可
> - 去掉**双向融合** → 削弱**形状锚定**
> - 去掉**投影注入** → 主要影响**未见实例**（因为此时**局部外观是决定性的**）
> - dropout 增益虽小但稳定，说明它的角色是**正则化器**而非语义知识的主要来源

### 6.6 自训练结果（Table 5）

| Split | 配置 | IoU | mIoU | AUC | MAE |
|-------|------|-----|------|-----|-----|
| Unseen **Object** | Baseline | .428 | .306 | .829 | .212 |
| | + Self-Training | .428 | **.315‡** | **.832‡** | .223 |
| | + Platt cal. | .428 | .306 | **.832‡** | **.211** |
| Unseen **Category** | Baseline | .305 | .243 | .716 | .266 |
| | + Self-Training | **.315‡** | **.259‡** | **.720‡** | .275 |
| | + Platt cal. | .309 | .246 | **.720‡** | **.265*** |

‡ p<0.01，* p<0.05

> [!success] 关键数字
> - 校准前：unseen-category **mIoU +0.016（相对 +6.3%）**、**IoU +0.010**、**AUC +0.004**，均 **p<0.01**
> - **尾类增益更大**：≤50 训练样本的类别 **+0.012**（64/135 提升），vs 头类 +0.003；**49/71** 个未见类别提升
> - Paired bootstrap（B = 10,000）差异（Table 12）：UC IoU `+0.0098 [0.0068, 0.0131]`，UC mIoU `+0.0152 [0.0104, 0.0183]`，UO mIoU `+0.0089 [0.0050, 0.0118]`，UO IoU `+0.0002 [−0.0039, 0.0042]`（**p = 0.461，即维持不降**）

> [!info] 因果边界（作者自己划的）
> 伪标签**不扩展语言词表**，所以这些增益**只隔离了「几何覆盖度」的贡献**，而不是新的语言监督。原始模型对**稀有正样本过度不自信**，Platt 校准改善了它的 reliability curve。

### 6.7 人工验证子集（Table 11，共 95 行：41 unseen-cat + 54 unseen-obj）

| Split | Model | N | IoU↑ | mIoU↑ | AUC↑ | MAE↓ |
|-------|-------|---|------|-------|------|------|
| Unseen category | **AffordAny** | 41 | **.3387** | **.2537** | **.7095** | **.3030** |
| | LASO | 41 | .2628 | .1953 | .6879 | .3193 |
| | LMAffordance3D | 41 | .2596 | .1988 | .6452 | .3151 |
| | OpenAD | 41 | .2897 | .2437 | .6747 | .3715 |
| Unseen object | **AffordAny** | 54 | **.4371** | **.3570** | **.8556** | .1901 |
| | LASO | 54 | .3656 | .2766 | .8299 | .2165 |
| | LMAffordance3D | 54 | .4133 | .3299 | .8409 | **.1787** |
| | OpenAD | 54 | .3360 | .2653 | .7902 | .2846 |

- 人工验证标准：重建可用、目标部件存在且无歧义、部件定位准确、指令自然且**视觉上可定位**
- `ignore` 标签**保守映射为背景**；四个冻结 checkpoint 在 0.5 阈值上统一评测

### 6.8 计算开销（Table 9，batch = 1，4,096 点）

| 项 | 值 |
|----|-----|
| 可训练参数 | **10.54 M** |
| BF16 参数占用 | 20.1 MiB |
| Profiled 矩阵运算 | **121.63 GFLOPs** |
| — 注意力 batched matmul | 69.14 GFLOPs (56.8%) |
| — 线性 matmul | 52.46 GFLOPs (43.1%) |
| BF16 延迟 中位 / P95 | **16.5 / 18.6 ms** |
| 峰值额外显存 | 45.3 MiB |

- 测量环境：NVIDIA **A100-SXM4-80GB**，BF16 autocast，20 次 warm-up 后 100 次计时前向
- **主导项是 4 层点解码器内部的自注意力**：约 `4LN²d = 68.7 GFLOPs`（L=4, N=4,096, d=256），几乎构成全部注意力开销；点–prototype 交互按 `NKd`（K=16）缩放
- 16.5 ms 中位 ≈ **约 60 次缓存解码器评估/秒**

> [!tip] 推理接口（很重要，别误读）
> **训练**：SAM-3D 几何、相机投影元数据、冻结 Cosmos 状态**全部离线算好并缓存**，只优化解码器。
> **推理**：RGB–mask 对**先过 SAM-3D-Objects** 拿到 Gaussians 与源相机对应 → **同一张源 RGB**（mask 外置白）+ 指令**在线**过冻结 Cosmos → 训练好的解码器输出逐点概率。
> ⇒ **推理不需要真值 3D 几何，也不需要预计算的 VLM 缓存。**

### 6.9 基线实现对照（Table 10）

| Method | 冻结特征输入 | 适配的解码器 | 可训练参数 | 入选 epoch |
|--------|--------------|--------------|-----------|-----------|
| OpenAD | CLIP pooled text (512-D) | point–text cosine alignment | 6.31 M | 130 |
| LASO | 77 CLIP text tokens | 2× AFM/GPBlock + 2 层 RPD | 7.01 M | 120 |
| LMAffordance3D | 77 CLIP text tokens | 2 融合层 + 2 解码层 | 6.38 M | 140 |
| **AffordAny** | **Cosmos state + 144 visual tokens** | **prototypes + GPBlock + 4 层解码器** | **10.54 M** | 140 |

> [!warning] 公平性的准确边界
> 所有方法共享**同一套点云、mask、标签、划分、损失、300-epoch 预算、seed、per-GPU batch、以及 unseen-object 验证选型标准**；但**各自使用其设计规定的冻结特征与解码器容量，不做参数量/特征提取成本的对齐**。
> ⇒ 这个比较**控制的是「数据与监督」，而不是「参数规模或特征成本」**。读结果时要记住这一点。

---

## 7. 结论与局限

### 7.1 结论

- AffordAny 从**单张单目 RGB 图**出发做 open-world 3D affordance grounding，端到端
- 自动化流水线从真实图像构建 473 类 / 5,334 物体的开词表文本条件部件监督
- 冻结 Cosmos-2B + 投影注入 / 语义压缩 / 双向融合，取得 strong unseen-object / unseen-category 泛化与**极低的指令敏感度**
- 伪标签自训练把 unseen-category mIoU 相对提升 **6.3%**（p<0.01），**尾类收益最大**
- 未来方向：VLM 端到端微调、把伪标签流水线扩展到更大图像源、部署到下游机器人操作

### 7.2 局限

> [!warning] 作者列出的四条（外加附录的发布约束）
> 1. **2,064 / 7,398 个完成物体没有任何有效部件**——这**混淆了「物体本身不可分解」与「重建/提示失败」**两种情况。
> 2. **长尾分布**：类别与部件分布长尾，可能**偏向常见部件**，弱化稀有部件的估计。
> 3. **标签无人工校验**：自动多视角融合标签**不可避免地含投影与分割噪声**。
> 4. **冻结特征限制任务适配**：端到端微调可能更好，但**算力需求大幅上升**。
>
> **附录 H 补充**：
> - 评测**只覆盖感知与 grounding**；抓取规划、避碰运动、物理执行**留作未来工作**
> - **单视角重建在未观测表面更不可靠**；标签质量依赖上游重建与分割
> - 在**薄、反光、透明、严重遮挡**的部件上重建质量会受影响；**规范视角覆盖不足**时若有效部件不在任何渲染视图中就会失败；2D→3D 提升对**深度归属**与**紧邻边界**最敏感；指令 grounding 对**功能相近的部件**或**空间上很宽的目标区域**最难
> - 发布约束：需保留 LVIS 标识与出处，遵守 LVIS 与源图条款；生成标注受**上游图像再分发权**约束；流水线**只用公开基准图像，不含私有用户交互数据**

### 7.3 附录里两个值得单独记的点

> [!note] F.1 排序-重叠背离（Ranking–Overlap Discrepancy）
> 论文筛出满足 `AUC > 0.93 且 MAE < 0.015 且 IoU ≤ 0.02` 的 unseen-category 样本：
> - toaster oven：IoU **0.000** / AUC 0.934 / MAE 0.014
> - drawer ×2：IoU 0.000 与 0.020 / AUC 0.934 与 0.944
>
> 机理：**目标点能排在背景点之上（高 AUC），但很少有点分数超过 0.5 阈值（低 IoU）；MAE 低是因为绝大多数点是背景。**
> ⇒ **只看排序类与平均误差类指标，无法刻画「可操作的定位」。** 部署时应**在验证集选阈值并固定在测试集上**使用。

> [!tip] C.5 复杂度的「可迁移」写法
> 论文明确说明 `121.63 GFLOPs` **只统计执行图里的矩阵运算**；softmax / normalization / GELU / indexing / 双线性采样等**体现在实测延迟与显存上，而不计入 FLOPs**。这种**界定口径**的写法值得在自己的论文里照抄。

---

## 8. 与我的研究的关系 / 可借鉴点

> [!info] 定位对照
> 本文属于 **language-driven / open-vocabulary 3D affordance** 一支，与 [[HAMMER]]、[[GREAT]]、[[InteractVLM]] 的 **intention-driven（图像驱动）** 路线是**互补**的：
> - 驱动信号：**自由文本指令**（vs 交互图像）
> - 3D 来源：**单目单图重建**（SAM 3D → 3DGS）（vs 已有点云）
> - 3D 模态：**3D 高斯 / 采样点云**（本文用 3DGS 做多视角标签提升的载体，与 [[Splat and Distill]]、[[3D感知视觉基础模型蒸馏 MOC]] 的 3DGS+VFM 蒸馏方向同源）
>
> 相关链接：[[3D Affordance Grounding]]（MOC）、[[PIAD]]（基准对照）、[[DINOv2]]

### 可直接借鉴的做法

1. **「冻结 VLM 离线缓存 + 轻量解码器」的接口设计**——把 `h_aff`（指令条件全局态）+ `F_v`（12×12 空间 token 网格）缓存下来，投影坐标与可见性一并缓存。这让**同一份特征可复用到多个解码器变体**，是**低成本做大量消融**的关键。
2. **`<AFF>` 特殊 token 的思路**（源自 LISA 的 `<SEG>`）：tokenizer 只加一次，初始化成已有 token embedding 的均值，取该位置的 hidden 作为 query-specific 摘要。
3. **指令条件语义瓶颈 = FiLM + 可学习 prototype + Transformer decoder**——用 `K=16` 固定大小的 bottleneck 把点-语义注意力的复杂度与 VLM 序列长解耦，且**先 FiLM 后读序列**让瓶颈在读取前就变成指令相关的。这是消融里贡献最大的一块（UC IoU −0.060）。
4. **Instruction dropout（0.2）**：训练时随机置零 `h_aff` 与 `F_v`，逼模型学几何先验，**直接换来指令改写鲁棒性**。这是一个极低成本、高回报的正则化。
5. **多轴泛化协议**：把 unseen object / unseen category / unseen paraphrase **拆成三个互不混淆的分数**，且**配对评测只改措辞**（匹配物体与部件）。这是「声称鲁棒」最省事也最难反驳的证据形态。
6. **「最小扰动」自训练 + 事后校准分离**：伪标签只跑 1 轮、权重 ~5%，并**把 Platt 校准当成独立一步**报告 raw 与 calibrated 两栏。既有增益，又不掩盖概率退化。
7. **消融表格的写法**：把每个组件的贡献**换算成目标指标的绝对下降**（0.060 / 0.035 / 0.030 / 0.017），并**解释每个组件在哪个泛化轴上起作用**（投影注入主要利于 unseen instance，语义压缩主要利于 unseen category）。

### 需要警惕 / 可质疑的点

- **数据流水线依赖若干闭源 LLM API**（`gemini-3.1-flash-lite-preview` 做类别/实例过滤与部件发现、指令生成），**可复现性受供应商行为影响**；不同 API 版本会带来分布漂移。
- **标签质量**：自动融合标签无人工校验；论文自认有投影与分割噪声，且 2,064/7,398 物体零有效部件——**监督信号的「覆盖率」值得追问**。
- **`3D heatmap` 与「真值」的关系**：IoU 用的是 `y = 1[t > 0]` 的**二值化软热图**，而监督本身来自多视角投票分数——**评估的 ground truth 与训练监督同源**，存在**自证循环**的隐患（人工验证子集 95 行是唯一独立锚点，但规模小）。
- **基线未对齐参数量**：AffordAny 10.54M vs 其余 6.3–7.0M，且特征来源不同（Cosmos-2B vs CLIP）。论文也承认这点，因此「提升」中**有多少来自解码器设计、多少来自更强的 VLM 特征**，并未完全解耦。
- **对照 [[HAMMER]] 的接口差异**：HAMMER 直接把 MLLM 隐状态（`f_c`）注入点特征；本文把 VLM 输出**压缩成 16 个 prototype** 再双向交互。**「压成 K 个 prototype」是否损失了细粒度 instruct 信息**——消融只测了「不加压缩」（−0.060），没测 `K` 的敏感度。

---

## 术语表

| 术语 | 定义 |
|------|------|
| **Affordance Grounding** | 在物体上定位「可交互区域」的任务；给定语言或图像线索预测功能区域 |
| **Open-world / Open-vocabulary** | 物体类别与语义（部件名、指令措辞）在测试时**不限于训练闭集** |
| **Object-centric point cloud** | 以单个物体为中心、由单图重建得到的点云（此处由 SAM 3D 从 RGB+mask 重建） |
| **SAM 3D / SAM-3D-Objects** | 从单张图像「3D 化任意物体」的模型，输出 object-centric 3D Gaussian 表示 + 相机参数 |
| **3D Gaussian Splatting (3DGS)** | 用可微高斯原语表示 3D 场景/物体，支持实时渲染；此处作为重建与多视角渲染的载体 |
| **h_aff** | 冻结 VLM 在指令末尾 `<AFF>` 标记处的上下文隐状态（∈R²⁰⁴⁸），**图像-指令语义摘要** |
| **F_v** | 源图 token 池化成的 12×12 网格特征（∈R¹⁴⁴×²⁰⁴⁸），**保留空间布局** |
| **Projection Injection** | 用重建相机把 3D 点投到 2D，双线性采样 `F_v` 并把局部特征加到点 token |
| **Semantic Compression** | 用 `K=16` 个 FiLM 条件化的可学习 prototype 压缩 145 个 VLM token，形成**任务感知语义瓶颈** |
| **GPBlock** | Group–Mix–Ungroup：prototype↔point 的**双向**交互块（Group 让语义吸几何，Ungroup 把上下文回传点） |
| **Instruction Dropout** | 训练时随机置零 `h_aff` 与 `F_v`，抑制语义捷径、强化几何先验、提升措辞鲁棒性（p=0.2） |
| **FiLM** | Feature-wise Linear Modulation：`z ⊙ (1+γ) + β`，用条件向量对特征做逐通道仿射调制 |
| **Pseudo-label Self-training** | 用已训练模型当 teacher 为新数据生成标签，再以低权重回训 student |
| **Minimal Perturbation Training** | 自训练只跑 1 轮、伪标签权重 ~5%，避免破坏原有表征与概率校准 |
| **Platt Calibration** | 事后单调校准 `σ(a·z + b)`，修正概率尺度；**保序 ⇒ AUC 不变** |
| **Pooled IoU / macro mIoU** | 前者全局池化交并；后者**先按类别**池化再对类平均（头尾类等权） |
| **SIM** | 单位质量归一化后的逐行直方图交集再平均，衡量分布一致性 |
| **Depth ownership** | 在 2D→3D 提升时，判定某个投影像素「归谁所有」（哪个高斯可见）的机制 |
| **LVIS** | Large Vocabulary Instance Segmentation 数据集；本文的**唯一图像来源** |

---

## 延伸阅读

- [ ] **LASO** — `Language-guided affordance segmentation on 3D object`（Li et al., CVPR 2024）。本文最强的 unseen-category AUC/MAE 竞争者，且 GPBlock 思路被本文借鉴（关联度：**高**）
- [ ] **LMAffordance3D / AGPIL** — `Grounding 3D object affordance with language instructions, visual observations and interactions`（Zhu et al., CVPR 2025）。语言条件 3D affordance triple，闭词表（关联度：**高**）
- [ ] **OpenAD** — `Open-vocabulary affordance detection in 3D point clouds`（Nguyen et al., IROS 2023）。CLIP 文本-点特征余弦对齐，open-vocabulary 先驱（关联度：**高**）
- [ ] **SAM 3D** — `3DFy Anything in Images`（Chen et al. 2025）。本文重建流水线的核心，也是「单图→3D 监督」类工作的基础设施（关联度：**高**）
- [ ] **Cosmos** — `Cosmos world foundation model platform for physical AI`（Agarwal et al. 2025）。本文的冻结 VLM 骨干（`nvidia/Cosmos-Reason2-2B`）（关联度：**高**）
- [ ] **LISA** — `Reasoning segmentation via large language model`（Lai et al., CVPR 2024）。`<AFF>` token 的直接灵感来源（关联度：**高**）
- [ ] **3D AffordanceNet**（Deng et al., CVPR 2021）/ **AGD20K**（Luo et al., CVPR 2022）/ **AffordPose**（Jian et al., ICCV 2023）。3D affordance 数据集的奠基与对照（关联度：中）
- [ ] **AffordMatcher / AffordBridge** — `Affordance Learning in 3D Scenes from Visual Signifiers`（Vu et al., CVPR 2026）。场景级 3D interaction 标注，Table 1 中标注规模最大（关联度：中）
- [ ] **3DAffordSplat** — `Efficient affordance reasoning with 3D Gaussians`（Wei et al., ACM MM 2025）。3DGS 与 affordance 结合，与本文的 3DGS 使用方式可对照（关联度：中）
- [ ] **PartSLIP**（Liu et al., CVPR 2023）/ **OpenMask3D**（Takmaz et al. 2023）/ **Feature 3DGS**（Zhou et al., CVPR 2024）。图文先验迁移到 3D 部件/实例/高斯场的代表工作（关联度：中）
- [ ] **Platt (1999)** — `Probabilistic outputs for support vector machines...`。校准方法的原始出处（关联度：低，但方法上有用）

---

## 个人批注

> [!note] 思考
> - 「冻结 VLM 特征 + 轻量解码器」这个设计把「特征成本」与「解码器设计」拆开了——但如果换一个更强的 VLM（比如更大的 Cosmos 或 Qwen2.5-VL），收益曲线会怎样？论文没做 VLM 规模/种类的消融。
> - `K=16` 个 prototype 作为语义瓶颈是本文最核心的模块（消融第一大）。但 16 这个数字是怎么选的？**「瓶颈容量」与「指令细粒度」之间的取舍**可能是一个可做的方向：例如按指令复杂度自适应 K。
> - 本文的 3D 几何来自**单图重建**，与「VGGT 前馈重建」是同一条思路。区别在于本文重建后**只做多视角标签提升**，最终预测仍在点上；而我们方案里是把重建结果融进 MLLM。两条路线值得并排比较。

> [!warning] 疑问
> - `[需核实:附录 A.2]` 类别过滤用 LLM 判断「该类别是否通常含可操作区域」——**误判率是多少？** 473 类里有多少是被误留/误删的？附录只给了流程与计数，没给校验。
> - `[需核实:Table 2]` Unseen Instruction 一列的数值（AffordAny 0.680）数量级明显高于 Unseen Object（0.428）。这是否因为 **unseen instruction 是在训练物体上评的**（同物体同部件、只换措辞），因而比跨实例/跨类别的任务「更容易」？各方法在该列上的**相对差距反而更大**（0.680 vs 0.484 的 40% 相对优势），这个现象本身值得深挖——它是措辞鲁棒性，还是**IoU 阈值在偏正预测上被放大**了？
> - `[需核实:C.4]` 伪标签的 `[0.15, 0.65]` 忽略带与 `0.37` 子采样率、Platt 的 `(1.10, −0.75)` 都固定在测试前——但**这些量是在 validation 上选的还是事先约定的？** 若是前者，多轮调参后的选择偏差未被讨论。
> - 人工验证子集只有 **95 行**（41 + 54）。用它做「独立锚点」时，**置信区间会很宽**，但论文没给这 95 行的 CI。是否足以支撑「自动标签可信」的结论？

> [!todo] 待办
> - [ ] 拉取 GitHub 代码，确认 `K=16` / `d=256` / 4 层 decoder 之外，**point encoder 的 13 维点特征具体是什么**（正文只说「13 cached geometric/appearance features」，需看附录/代码）
> - [ ] 在 [[3D Affordance Grounding]] MOC 中补充本条，并标注它与 intention-driven 路线的关系
> - [ ] 对照 [[HAMMER]] 的 `f_c` 注入方式，做一次「**瓶颈 vs 直注**」的机制级对比笔记
