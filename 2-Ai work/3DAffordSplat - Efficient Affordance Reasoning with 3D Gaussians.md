---
title: "3DAffordSplat：基于 3D Gaussian 的高效 Affordance Reasoning（数据集 + AffordSplatNet）"
aliases:
  - 3DAffordSplat
  - AffordSplat
  - AffordSplatNet
  - 3DAffordSplatNet
  - Efficient Affordance Reasoning with 3D Gaussians
authors:
  - Zeming Wei
  - Junyi Lin
  - Yang Liu
  - Weixing Chen
  - Jingzhou Luo
  - Guanbin Li
  - Liang Lin
year: 2025
venue: arXiv 2504.11218v2 (CV 方向，2025-04-16)
arxiv: 2504.11218
code: https://github.com/HCPLab-SYSU/3DAffordSplat
affiliation: 中山大学 / 鹏城实验室 / 广东省大数据分析与处理重点实验室
paper_status: 已通读全文
note_status: 已完成首轮深度分析
research_area:
  - affordance grounding
  - 3D Gaussian Splatting
  - point cloud affordance
  - cross-modal alignment
  - embodied AI
tags:
  - paper/affordance-grounding
  - paper/3dgs
  - method/pointnet++
  - method/cross-modal-alignment
  - method/attention
  - dataset/3dgs-affordance
  - application/robot-manipulation
created: 2026-08-12
source_pdf: "[[../1-inbox/2026-7-13DAffordSplat.pdf]]"
---

# 3DAffordSplat：基于 3D Gaussian 的高效 Affordance Reasoning

> [!summary] 一句话总结
> 3DAffordSplat 是**第一个**面向 3D Gaussian Splatting（3DGS）的大规模、多模态 affordance 数据集；在此之上提出的 **AffordSplatNet** 通过「跨模态结构对齐（CMSA）」把海量点云 affordance 监督迁移到连续、稠密的 3DGS 表征上，显著优于仅用点云的基线，并在 seen / unseen 设定下均取得更好泛化。

> [!important] 核心判断
> 这篇论文最有价值的两点不是某个新模块，而是：**(1) 把 3DGS 作为 affordance 的可行表征正式立桩（填补 3DGS affordance 数据集空白）**；**(2) 指出「传统给每个 Gaussian 静态挂一个语义特征」的单语义范式不适合多属性 affordance，提出由查询动态生成任务相关描述符的解法方向。** 模型本身相对标准（PointNet++ + 注意力 + 动态卷积），其增益主要来自数据质量与跨模态对齐。

## 0. 快速索引

- **任务**：给定 3DGS 表征 + 自然语言指令，输出「文本回答 A」与「点级（逐 Gaussian）affordance mask M」。
- **核心交付物**：数据集 `3DAffordSplat` + 模型 `AffordSplatNet`。
- **方法简称混乱**：正文称 `AffordSplat` / `AffordSplatNet`，Figure 4 图注又称 `3DAffordSplatNet`；笔记保留全部别名以便检索。
- **最重要结果**：AffordSplatNet（RoBERTa 配置）Seen mIoU `33.03`、UnSeen `18.91`；对比基线 PointRefer（Seen `18.40` / UnSeen `15.90`）、IAGNet（Seen `14.63` / UnSeen `4.70`）。
- **最大问题**：3DGS affordance 仍严重依赖人工标注的「18 个 Gaussian / 每 object-affordance 组合」做微调；unseen 泛化偏弱；小词汇量 LM 导致文本生成质量差；对多不连续 affordance 区域处理不佳。
- **适合复用的思想**：结构一致性先验做跨模态对齐、逐粒度门控融合、用 Chamfer Distance 给跨模态样本加权、动态 kernel 卷积做 mask 生成、⟨Aff⟩ 特殊 token 注入。
- **关键数字（待核对一致性）**：GS 实例数正文写 `23,677`、Table 5 写 `23,672`；点云实例数正文写 `8,354`、Table 5 写 `8,231`、Table 1 写 `8.4k`。多处口径不一致，见 [[#14. 原文定位]] 与 [[#15. 待办与复现检查清单]]。

## 1. 研究问题

### 1.1 任务定义

给定 3D Gaussian Splatting 表征
$$G=\{m,s,r,o,c\}$$
其中：

- $m\in\mathbb{R}^3$：Gaussian 中心位置；
- $s\in\mathbb{R}^3$：尺度参数；
- $r\in\mathbb{R}^4$：旋转参数（统称 **structural features**）；
- $o\in\mathbb{R}$：不透明度；
- $c$：基于球谐（SH）的颜色特征（与 $o$ 统称 **appearance features**）。

论文假设 affordance 主要由**局部结构特征**决定，因此模型只处理
$$G_{struct}=\{m,s,r\}\in\mathbb{R}^{10}$$

对于文本查询 $Q$，模型输出：

- 文本回答 $A$；
- 对应的逐 Gaussian affordance mask $M\in\{0,1\}^N$（$N$ 为 Gaussian 数量）。

### 1.2 为什么困难

- **点云表征的固有限制**：离散采样，稀疏、几何分辨率低，难以表达连续表面与细粒度 affordance 结构（Figure 2 直观对比）。
- **3DGS 缺乏 affordance 标注**：3DGS 虽高保真、可实时渲染，但没有大规模带 affordance 标注的数据集，模型难以训练与评测。
- **跨模态对齐困难**：点云稀疏有噪声，3DGS 稠密连续，二者形状/结构不匹配，需精心设计保证几何与语义一致。
- **单语义范式失效**：传统给每个 Gaussian 静态挂一个语义特征的方法，无法表达「一个 Gaussian 同时参与多个功能上下文」的多属性 affordance 场景。

## 2. 研究背景与现有方法局限

### 2.1 2D / 视频 / 点云 affordance 的局限

| 表征 | 优点 | 局限（论文立场） |
|---|---|---|
| Image | 易获取、有纹理 | 缺深度，抓不住完整 3D 结构 |
| Video | 有动态线索 | 无直接 3D 空间信息，标注难，难表达细微交互变化 |
| Point cloud | 直接 3D 几何 | 离散、稀疏、几何分辨率低，难表达连续 surface |

### 2.2 3DGS 作为 affordance 表征的优势（论文主张）

1. 更高几何精度、保留表面细节 → 解决点云离散/不完整；
2. 集成丰富颜色信息 → 弥补图像法缺 3D 空间信息；
3. 高效实时渲染、低算力 → 1080p 下 30+ fps，优于视频法。

### 2.3 现有 3D affordance 数据集对比（Table 1）

所有已有 3D affordance 数据集（3DAffordanceNet、PIAD、LASO、SeqAfford、AGPIL 等）均基于**点云**；现有 3DGS 数据集（CLIP-GS、ShapeSplat）**都没有 affordance 标注**。3DAffordSplat 是首个同时整合 3DGS + 点云 + 语言、且带 affordance 标注的 benchmark。

| Benchmark | 3DGS | Point Cloud | Text | Afford. Anno | Task |
|---|---:|---:|---:|---:|---|
| 3DAffordanceNet | ✗ | 56k | ✗ | none | No limit |
| PIAD | ✗ | 7k | ✓ | none | Grounding |
| LASO | ✗ | 8.4k | ✓ | none | Reasoning |
| SeqAfford | ✗ | 1.8k | ✓ | none | Reasoning |
| AGPIL | ✗ | 41k | ✓ | none | Reasoning |
| **3DAffordSplat** | **23k** | **8.4k** | **✓** | **6,631** | **Reasoning** |

> [!warning] 数字口径不一致
> 上表「23k / 8.4k」取自正文摘要与 Table 1。但 Table 5 实际统计为 GS `23,672`（其中 `6,631` 为标注数）、点云 `8,231`。正文第 3 节又写「23,677 Gaussian instances，8,354 point clouds」。三处不一致，应以 Table 5 的逐项统计为准，但摘要级数字以作者写法保留。

## 3. 数据集：3DAffordSplat

### 3.1 数据来源

- **3D Gaussians**：来自 ShapeSplat [41]（v1 子集），其来源为 ModelNet（→ ModelSplat，含 door/vase）与 ShapeNet（→ ShapeSplat，覆盖多数对象）。
- **Point clouds & Instructions**：来自 LASO [30]。
- **剔除类别**：ShapeSplat 缺 LASO 的 `scissors`、`refrigerator` 的 Gaussian 对象，故排除这两类。
- **合并后**：21 个物体类别、18 个 affordance 类型，每实例含点云 + 3DGS + 文本三模态。

### 3.2 标注与指令格式

- 按 3DAffordanceNet [10] 标准人工标注 Gaussian affordance。
- 每 object-affordance 组合配 15 个问题 + 3 个答案。
- **创新点**：在指令句中 affordance 词后插入特殊 token `⟨Aff⟩`，增强模型对 affordance 语义的识别与 grounding。

### 3.3 统计与设定（Table 5）

| Object | Affordance | NumGS | NumPC |
|---|---|---:|---:|
| Bag | grasp, lift, contain, open | 83 | 100 |
| Bed | lay, sit, support | 233 | 145 |
| Bottle | contain, open, wrap_grasp, grasp, pour | 498 | 328 |
| Bowl | contain, wrap_grasp, pour | 186 | 150 |
| Chair | sit, support, move | 6,731 | 1,886 |
| Clock | display | 651 | 353 |
| Dishwasher | open, contain | 93 | 132 |
| Display | display | 1,091 | 488 |
| Door | open, push, pull | 129 | 175 |
| Earphone | grasp, listen | 73 | 178 |
| Faucet | grasp, open | 744 | 359 |
| Hat | grasp, wear | 218 | 177 |
| Keyboard | press | 65 | 125 |
| Knife | grasp, cut, stab | 423 | 255 |
| Laptop | display, press | 460 | 337 |
| Microwave | open, contain, support | 152 | 148 |
| Mug | contain, pour, wrap_grasp, grasp | 214 | 151 |
| Storage Furniture | contain, open | 2,321 | 690 |
| Table | support, move | 8,390 | 1,420 |
| Trash Can | contain, pour, open | 342 | 251 |
| Vase | contain, pour, wrap_grasp | 575 | 383 |
| **Total** | — | **23,672 (6,631 标注)** | **8,231** |

> [!note] 6,631 的口径
> 正文写「每 object-affordance 组合人工标注 18 个 Gaussian 用于验证/测试，共计 6,631 个 Gaussian affordance 标注」。即 6,631 是**带 affordance 标注的 Gaussian 实例数**，而非 affordance 类别数（类别数为 18）。

### 3.4 Seen / UnSeen 设定

- **Seen**：训练/测试的物体类别与 affordance 类型分布一致。
- **UnSeen**：测试集的 object-affordance 组合与训练集**完全不同**；且专门设计使 `display` 这类 object 类型、`lift` 这类 affordance 类型、`mug-grasp` 这类组合**只出现在测试/验证集**。

### 3.5 预训练与评测协议

- **Pretrain**：每个 Gaussian 实例随机配同类的多个点云 + 从 15 个模板问题采样的问题 + 固定的 object-affordance 文本标签。
- **Evaluation**：用带标注的 Gaussian 数据评测，固定多问题测试泛化。

## 4. AffordSplatNet 整体架构

```text
文本查询 Q
   │
   ▼
预训练 LM (RoBERTa + LoRA) ──提取 ⟨Aff⟩ token 最后一层 embedding──► H_Aff
   │                                                          │
   ▼                                                          │
3D Gaussian G_struct = {m,s,r}                               │
   │                                                          │
   ▼                                                          │
PointNet++ 分层 3D Encoder ──{F^i_g} 三层多粒度特征───────────┤
   │                                                          │
   ▼ (Cross-Attention + Channel-Attention) ◄─────────────────┘
逐粒度融合特征 F^i_spatial → F^i_channel
   │
   ▼ IDW 上采样到统一分辨率 N
可学习门控 W_gate (Softmax) ── Granularity-Adaptive Selection ──► F_fused
   │
   ▼ Validity Mask 过滤 + Transformer Decoder 生成动态 kernel
Dynamic Kernel 卷积 ──► Affordance Mask M (逐 Gaussian)
   │
   ▼
文本 Decode ──► 回答 A
```

训练两阶段：**Pretrain**（用 CMSA 无监督对齐点云↔GS 结构关系）→ **Finetune**（用 3DAffordSplat 标注监督精修）。

### 4.1 Gaussian-Text 特征融合

- **文本侧**：预训练 LM $\Psi_{LM}$ 提取 `⟨Aff⟩` token 最后一层 embedding $h_{Aff}$，经 MLP 投影到 $H_{Aff}=MLP(h_{Aff})\in\mathbb{R}^{B\times1\times d_{text}}$；LM 同时生成文本回答 $\tilde{y}_{text}$。
- **几何侧**：分层 3D 编码器 $\Phi_{3D}$（PointNet++ 为 backbone）提取多粒度几何特征 $\{F^i_g\}_{i=1}^3\in\mathbb{R}^{B\times N_i\times d}$；最后解码阶段 point-level 特征作为 backbone 输出，并在 3D 编码器后加 transformer encoder 增强。
- **多模态融合（Eq.1–2）**：
  $$F^i_{spatial}=CrossAtt(H_{Aff},F^i_g,F^i_g)+PosEmb(N_i)$$
  $$F^i_{channel}=ChannelAtt([F^i_{spatial},F^i_g])+F^i_g$$
  其中 $H_{Aff}$ 作 query，$\{F^i_g\}$ 作 key/value；channel-attention 通过残差保留原几何保真度。

### 4.2 粒度自适应选择与解码器

- **IDW 上采样**（Eq.3）：把所有粒度特征上采样到统一分辨率 $N$。
- **门控选择**（Eq.4–5）：
  $$W=Softmax(W_{gate}\odot[F_1\Vert F_2\Vert F_3])$$
  $$F_{fused}=\sum_{i=1}^3 w_i\odot F_i$$
  可学习门控加权，竞争式分配各粒度重要性。
- **解码器**（Eq.6–9）：
  - $F_{up}=IDW(F_{fused})$ 上采样回原 Gaussian 密度；
  - 用 validity mask $M_{valid}$（依据 $G_{struct}$ 中 padding 位置）过滤无效点；
  - Transformer 解码器基于文本 embedding 生成位置感知**动态 kernel**：$K_{dynamic}=TransformerDecoder(F_{valid},H_{Aff})$；
  - 最终 mask：$M_{gs}=\sigma(F_{valid}*K_{dynamic})\odot M_{valid}$（* 为卷积，σ 为 Sigmoid）。

### 4.3 跨模态结构对齐（CMSA，Pretrain 阶段）

**核心先验**：同一物体类别，显式 3D 表征不同，但 affordance 区域与整体结构的**相对空间关系保持不变**。

- 用模态专属编码器把「点云 affordance 区域」「Gaussian affordance 区域」及其各自完整模型编码到共享 $d_{consis}$ 维空间（Eq.10–11）：
  $$F^{Aff}_{gs}=\Phi_{gs}(M_{gs}\odot G_{struct}),\;F_{gs}=\Phi_{gs}(G_{struct})$$
  $$F^{Aff}_{pc}=\Phi_{pc}(M_{pc}\odot P),\;F_{pc}=\Phi_{pc}(P)$$
  其中 $M_{gs}=STE(M_{gs})$，STE 为 Straight-Through Estimator。
- 共享多头 cross-attention 计算结构亲和矩阵（Eq.12），再经共享 FFN 投影到隐空间得相对结构特征 $Z_{gs},Z_{pc}$。
- 用 Chamfer Distance 给跨模态样本加权（Eq.13）：
  $$w^i_{consis}=Softmax(-D_{Chamfer}(G_{struct},P^k)/\tau)$$
  解决 Gaussian 对象与点云对象形状/结构差异导致的对齐权重问题。

### 4.4 训练目标

- **Pretrain（Eq.14–15）**：
  $$L_{pretrain}=L_{consis}=w_{consis}\odot L_{cosine}$$
  $L_{cosine}$ 为对齐 Gaussian 与点云间 affordance 相对结构关系的余弦损失（无监督）。
- **Finetune（Eq.16）**：
  $$L_{finetune}=L_{BCE}+L_{Dice}+L_{text}$$
  $L_{BCE}$、$L_{Dice}$ 解决类别不平衡、提升分割精度；$L_{text}$ 为文本生成的交叉熵损失。

### 4.5 实现细节

- RoBERTa 预训练 + LoRA 微调；特征维度 $d=512$。
- **Pretrain**：未标注 Gaussian + 带标注点云做跨模态对齐；每 Gaussian 随机配 4 个点云实例，生成 **94,708** 个 GS-PC 样本对；训练 1 epoch，lr $1e-5$。
- **Finetune**：除语言模块外全量微调；lr $1e-4$，训练 60 epoch。
- 两阶段均用 AdamW；4× RTX 4090。

## 5. 实验与结果

### 5.1 评测指标

沿用 3D affordance grounding 文献 [30,37,51,63]：mIoU↑、AUC↑、SIM↑、MAE↓（部分表另有 KLD↓）。详见 [[#7.2]] 附录定义。

### 5.2 数据集迁移与有效性（Table 2 摘要，Sec.5.2）

- **高质量数据**：纯点云训练次优（IAGNet-Seen mIoU `21.22`、PointRefer-Seen `19.20`，因 LASO/PIAD 标注噪声）；3DAffordSplat 细粒度人工标注微调后大幅跃升（IAGNet `30.77`、PointRefer `49.40`）；训练/测试都用 3DAffordSplat 最佳（IAGNet `31.52`、PointRefer `51.80`）。
- **pc→gs 迁移强**：LASO 预训练 mIoU 从 `5.10` → 微调后 `49.40`；反向 gs→pc 仅 `3.80`→`18.50`，说明 3DGS 适配性更强。
- **gs→pc 可比**：预训练于 3DGS、测于点云，性能与纯点云模型相当（`18.20`/`18.50`），降低点云依赖。
- **UnSeen 泛化**：3DAffordSplat 支持更好（同测试集 PointRefer `7.37` vs LASO `4.19`）。

> [!warning] Table 2 数字口径
> Table 2 行数多、设定杂（PIADv1/LASO × 3DAffordSplat × Seen/UnSeen × 是否 FT）。本文笔记只抽取与结论直接相关的关键行，完整数值以 PDF 表格为准。

### 5.3 与基线模型对比（Table 3，主结果）

| Setting | Method | mIoU↑ | AUC↑ | SIM↑ | MAE↓ |
|---|---|---:|---:|---:|---:|
| Seen | IAGNet | 14.63 | 56.67 | 0.35 | 0.41 |
| Seen | PointRefer | 18.40 | 78.50 | 0.43 | 0.20 |
| **Seen** | **AffordSplatNet (Ours)** | **30.25** | **83.85** | **0.44** | **0.21** |
| UnSeen | IAGNet | 4.70 | 40.77 | 0.24 | 0.43 |
| UnSeen | PointRefer | 15.90 | 67.00 | 0.31 | 0.29 |
| **UnSeen** | **AffordSplatNet (Ours)** | **17.31** | **67.18** | **0.32** | **0.31** |

> [!note] 与 Table 9（消融表）的差异
> Table 3 的 Ours 主结果为 Seen `30.25` / UnSeen `17.31`；而 Table 9（CMSA 消融，RoBERTa 配置）报告 Ours Pretrain-finetune 为 Seen `33.03` / UnSeen `18.91`。差异来自是否使用 RoBERTa 全配置与是否含 CMSA 预训练。笔记把 `33.03`/`18.91` 视为「完整 AffordSplatNet（RoBERTa）」的更完整数字，把 `30.25`/`17.31` 视为 Table 3 主对比数字。作者未明确说明两个表的配置差异，记待核对。

**结论**：AffordSplatNet 在 seen/unseen 均优于两基线；IAGNet 因强调 image-point cloud 对齐、缺语言引导，且难处理高维 Gaussian，表现最差。加 scale/rotate 通道反而略降基线性能，说明「为点云设计的模型」不足以直接学 3DGS。

### 5.4 定性结果

- Figure 4：AffordSplatNet 能同时分割细粒度区域（Door-Open）与大连续区域（Clock-Display）；PointRefer/IAGNet 有漏区域、噪声预测、边界模糊。
- Figure 5：用 3DGS 从真实图像重建的 Mug/Bag，模型能适配真实物体。

### 5.5 Per-affordance / Per-object 统计（Table 6–7 摘要）

- **Affordance**：结构清晰者（cut `SIM 0.6888`、wear `0.8322`、stab `MAE 0.1159`、pull `MAE 0.0083`）表现好；模糊/多变者（press、listen、push、display）较低。
- **Object**：几何典型者（hat `IoU 0.5358`、door `MAE 0.0263`、knife、chair、vase）强；少样本/高形变差者（clock `IoU 0.0957`）弱。

## 6. 消融实验

### 6.1 语言编码器（Table 8）

| Language Encoder | mIoU↑ | AUC↑ | SIM↑ | MAE↓ |
|---|---:|---:|---:|---:|
| BART（论文标 decoder-only） | 20.61 | 73.52 | 0.35 | 0.27 |
| GPT2（论文标 encoder-decoder） | 32.96 | 81.34 | 0.44 | 0.22 |
| RoBERTa（论文标 encoder-only） | **33.03** | **84.67** | **0.46** | **0.21** |

RoBERTa 最佳（双向上下文 + 易接 MLLM）；GPT2 次之（生成能力强但易偏离推理）；BART 最差且最慢。

> [!warning] 语言模型类型标注疑似颠倒
> 论文把 BART 标为 decoder-only、把 GPT2 标为 encoder-decoder，这与公认事实相反（BART 是 encoder-decoder，GPT2 是 decoder-only）。笔记保留原文标签但提醒读者：此处标注很可能写反，不影响数值结论，但影响对「为何 BART 最差」的解释。记待核对。

### 6.2 对齐模块 CMSA（Table 9）

| Setting | Variant | mIoU↑ | AUC↑ | SIM↑ | MAE↓ |
|---|---|---:|---:|---:|---:|
| Seen | Ours (Pretrain-finetune) | 33.03 | 84.67 | 0.46 | 0.21 |
| Seen | w/o CMSA (Finetune) | 37.18 | 81.34 | 0.48 | 0.20 |
| UnSeen | Ours (Pretrain-finetune) | 18.91 | 66.71 | 0.32 | 0.31 |
| UnSeen | w/o CMSA (Finetune) | 17.93 | 62.39 | 0.27 | 0.31 |

**关键发现（反直觉）**：

- **UnSeen**：去掉 CMSA 全面下降（mIoU `18.91→17.93`、AUC `66.71→62.39`、SIM `0.32→0.27`），证明 CMSA 对未见物体泛化至关重要——它把点云预训练的 affordance 先验迁移到 3DGS，并充当知识瓶颈过滤任务无关几何变化。
- **Seen**：去掉 CMSA 反而 mIoU 升（`33.03→37.18`）。论文给出两点解释：
  1. **任务特定过对齐（Task-Specific Overalignment）**：预训练对齐可能强加过 rigid 的对应关系，与微调数据冲突；
  2. **数据充分性缓解**：seen 物体微调数据充足，掩盖了预训练收益。

> [!note] CMSA 价值的辩证结论
> CMSA 是「未见泛化」的必要组件，但在「已见且数据充足」时可能轻微拖累定位指标。这与「预训练对齐未必对所有设定都正向」的普遍观察一致。

### 6.3 输入参数选择（Table 4，附录）

在 PointRefer 框架下对 Gaussian 参数组合实验（下采样到 2048 点）：

| 参数组合 | mIoU↑ | AUC↑ |
|---|---:|---:|
| xyz+rotate+scale | **51.20** | 94.0 |
| xyz+opacity+rgb | 48.40 | 93.8 |
| xyz+rotate+scale+opacity+rgb | 50.60 | 94.1 |

最终选 **xyz+rotate+scale**（= structural features，对应 $G_{struct}\in\mathbb{R}^{10}$），在 mIoU 最优与算力间平衡。

## 7. 评测指标定义（附录 Sec.7.2）

- **mIoU**：各类 IoU 均值，$IoU=TP/(TP+FP+FN)$，越高越好。
- **AUC**：saliency map 当作二分类器的 ROC 曲线下面积，越高越好（区分正负区域能力）。
- **SIM**：预测图与 GT 图归一化后逐元素最小值之和，越高越一致。
- **MAE**：预测与 GT 绝对误差均值，越低越好。
- 理想模型：高 mIoU + 高 AUC + 高 SIM + 低 MAE。

## 8. 失败案例分析（Figure 9，Sec.9.5）

- **错误文本回答**：复杂指令下 LM（RoBERTa/GPT2/BART 参数量小、词表覆盖有限）生成质量差。
- **复杂物体结构处理不足**：多层 storage furniture 等多不连续 affordance 区域能力弱。

## 9. 论文贡献总结

1. **3DAffordSplat 数据集**：首个大规模、多模态 3DGS affordance reasoning 数据集（GS + 点云 + 文本），含 21 类、18 类 affordance、6,631 标注。
2. **首个 3DGS affordance 评测框架**：复用 mIoU/AUC/SIM/MAE，与点云 benchmark 向后兼容。
3. **AffordSplatNet**：首个可泛化的 3DGS affordance 架构，用 CMSA 建立点云↔GS 跨模态结构对应，实现知识迁移。
4. **实验证明**：3DAffordSplat 显著提升已有点云方法在 3DGS 上的表现；AffordSplatNet 在 seen/unseen 均优于现有方法。

## 10. 论文局限性

### 10.1 作者明确承认

- UnSeen 设定所有指标低于 Seen，泛化到未见过数据仍困难。
- 现有 LM（RoBERTa/GPT2/BART）参数小、词表有限，文本生成不精确。
- 对具多不连续 affordance 区域的复杂物体（如多层 storage furniture）处理不足。

### 10.2 批判性分析（非作者原话）

#### A. 3DGS affordance 标注仍极稀疏

仅每 object-affordance 组合标注 18 个 Gaussian（共 6,631），且微调高度依赖这些标注。与 94,708 的 GS-PC 预训练对相比，监督信号规模悬殊，unseen 偏弱可能源于标注覆盖不足而非方法本身。

#### B. 单语义 → 动态描述符的承诺未充分兑现

论文在 Related Work 强调「传统静态单语义特征不适合多属性 affordance，应动态生成任务相关描述符」，但 AffordSplatNet 实际是用查询驱动的 ⟨Aff⟩ embedding 通过注意力融合，**并未显式实现「同一 Gaussian 同时输出多语义」的表征**。该主张更多是动机陈述，方法上仍输出单 mask。

#### C. CMSA 在 seen 上的负向效应未被充分解释

去掉 CMSA 后 seen mIoU 反而升 ~4 点，论文归因于过对齐与数据充分性。但这是否意味着预训练阶段对齐目标（余弦结构相似）与下游 BCE/Dice 目标存在系统性张力，值得进一步量化（如可视化对齐前后特征分布）。

#### D. 基线对比的公平性存疑

- Table 3 中 IAGNet/PointRefer 数字（`14.63`/`18.40`）明显低于其在 Table 2 自身最佳（`31.52`/`51.80`），因输入/设定不同。基线在 3DGS 上的输入处理（如 scale/rotate 通道反而降性能）说明「为点云设计的模型」直接套 3DGS 不公平，但这也削弱「AffordSplatNet 大幅领先」的相对说服力。
- 未见与同时期 3DGS affordance/分割方法（如 Segment Any 3D Gaussians、Click-Gaussian、LangSplat、Feature3DGS）的直接对比。

#### E. 真实场景泛化证据弱

Figure 5 仅 2 个真实重建对象（Mug/Bag）做定性展示，无定量指标、无机器人闭环成功率/碰撞率/延迟统计。

#### F. 数字口径多处不一致

见 [[#0. 快速索引]] 与 [[#14. 原文定位]]：GS 实例 `23,677` vs `23,672`、点云 `8,354` vs `8,231` vs `8.4k`；Table 3 与 Table 9 主结果不一致；语言模型类型标注颠倒。

#### G. 实时性主张未量化

正文说 3DGS 可达 30+ fps@1080p，但这是 3DGS 渲染属性，并非 AffordSplatNet 推理吞吐；论文未报告模型本身 FPS/延迟。

## 11. 未来发展方向

### 11.1 作者提出

- 将 affordance 推理框架集成到 embodied robot，在动态环境中与物体物理交互。

### 11.2 可进一步推演

#### 方向 1：多属性 / 组合式 affordance 表征

真正让每个 Gaussian 输出多语义概率（多通道 mask 或 query-conditioned 多 head），而非单 mask，兑现论文 Related Work 的动态描述符主张。

#### 方向 2：更强语言模型

用更大词表、更强推理的 LM/MLLM（如 LLM 骨干）替换 RoBERTa，改善复杂指令理解与文本生成，缓解 Figure 9 失败。

#### 方向 3：更密标注 + 自动标注

用 2D affordance / foundation model 蒸馏半自动扩展 3DGS affordance 标注，缓解 6,631 标注稀疏导致的 unseen 瓶颈。

#### 方向 4：与 3DGS 分割/语义方法对齐

与 Segment Any 3D Gaussians、LangSplat、Feature3DGS、GraspSplats 等建立统一对比与结合，验证 CMSA 是否对开放词汇/抓取任务同样有效。

#### 方向 5：embodiment-conditioned affordance

显式输入 agent 形态（夹爪/吸盘/人手）与安全约束，而非把 affordance 当作物体固有标签。

#### 方向 6：真实世界定量 benchmark

建立含成功率、碰撞率、延迟、遮挡/噪声鲁棒性的机器人 affordance benchmark。

## 12. 对 affordance grounding 研究的启示

### 12.1 可直接借鉴

- 用「结构一致性先验」做点云↔3DGS 跨模态对齐，比单靠 2D 蒸馏更几何可解释。
- 逐粒度门控融合 + IDW 上采样适合处理变密度 3D 表征（点云/GS 混用）。
- ⟨Aff⟩ 特殊 token 是把语言查询锚定到分割区域的轻量技巧。
- 用 Chamfer Distance 给跨模态配对样本加权，缓解模态形状差异。

### 12.2 需要谨慎借鉴

- seen 上 CMSA 负向效应提示：预训练对齐目标需与下游目标解耦或课程化。
- 仅用 mIoU 选输入参数可能牺牲 AUC/MAE 以外的维度（Table 4 加 rgb/opacity 仅 AUC +0.1 但其他指标降）。
- 3DGS affordance「实时」需单独报告模型推理延迟，不能引用渲染 fps。

### 12.3 可形成的研究假设

> **假设 H1**：3DGS affordance 的 unseen 瓶颈主要来自标注稀疏（6,631）而非表征本身，扩标注后 CMSA 收益会进一步放大。

> **假设 H2**：论文主张的「动态多语义描述符」若不落实到多通道输出，其相对单语义 embedding 的优势有限。

> **假设 H3**：CMSA 的余弦结构对齐与下游 BCE/Dice 存在目标张力，seen 上的负向效应可通过对齐损失加权或仅用于初始化缓解。

> **假设 H4**：更强 LM 骨干（替代 RoBERTa）会显著提升复杂指令下的文本生成与细粒度区域定位。

## 13. 复习卡片（Active Recall）

> [!question]- Q1：3DAffordSplat 相比已有 3D affordance 数据集的最大区别是什么？
> 它是首个同时整合「3D Gaussian + 点云 + 语言」三模态并带 affordance 标注的大规模数据集；已有数据集（3DAffordanceNet/PIAD/LASO 等）都只有点云，已有 3DGS 数据集（CLIP-GS/ShapeSplat）没有 affordance 标注。

> [!question]- Q2：AffordSplatNet 处理 Gaussian 时只用哪些参数？为什么？
> 只用 structural features $G_{struct}=\{m,s,r\}\in\mathbb{R}^{10}$（中心、尺度、旋转）。论文假设 affordance 主要来自局部结构特征，故丢弃 appearance（opacity、SH 颜色）。

> [!question]- Q3：CMSA 模块的核心先验是什么？
> 同一物体类别下，虽然显式 3D 表征不同，但 affordance 区域与整体结构的相对空间关系保持不变；据此用跨模态结构对齐把点云 affordance 先验迁移到 3DGS。

> [!question]- Q4：模型如何给跨模态配对样本加权？
> 用 Chamfer Distance：$w^i_{consis}=Softmax(-D_{Chamfer}(G_{struct},P^k)/\tau)$，形状/结构越接近的样本权重越高。

> [!question]- Q5：完整 AffordSplatNet（RoBERTa）的主结果是多少？
> Seen mIoU 33.03 / AUC 84.67 / SIM 0.46 / MAE 0.21；UnSeen mIoU 18.91 / AUC 66.71 / SIM 0.32 / MAE 0.31（Table 9 口径）。

> [!question]- Q6：去掉 CMSA 在 seen 和 unseen 上分别如何变化？
> UnSeen 全面下降（mIoU 18.91→17.93 等）；Seen 反而 mIoU 升（33.03→37.18），因任务特定过对齐 + 微调数据充分性。

> [!question]- Q7：哪个语言编码器最好？论文对模型类型的标注有什么问题？
> RoBERTa 最好（33.03）。论文把 BART 标为 decoder-only、GPT2 标为 encoder-decoder，与事实相反（应为 BART encoder-decoder、GPT2 decoder-only），标注疑似颠倒。

> [!question]- Q8：⟨Aff⟩ 特殊 token 的作用是什么？
> 插在指令句中 affordance 词之后，让 LM 提取该 token 的最后一层 embedding 作为「中间分割表示」，锚定查询的 affordance 语义。

> [!question]- Q9：论文明确承认哪些局限？
> UnSeen 泛化仍弱；小词汇量 LM 文本生成差；对多不连续 affordance 区域的复杂物体处理不足。

> [!question]- Q10：3DAffordSplat 数据量口径有几处不一致？
> GS 实例数正文 23,677 / Table 5 写 23,672；点云实例数正文 8,354 / Table 5 写 8,231 / Table 1 写 8.4k；Table 3 与 Table 9 主结果数字不一致。

## 14. 原文定位

| 内容 | 原文章节/图表 | 备注 |
|---|---|---|
| 背景、挑战、贡献 | Abstract / Sec.1 | 第 1–2 页 |
| 相关工作与数据集对比 | Sec.2 / Table 1 | 第 3–4 页 |
| 数据集构建、统计、设定 | Sec.3 / Table 5, Fig.1,6,7 | 第 4–5 页、12–13 页 |
| 任务定义与符号 | Sec.4 开头 | 第 5 页 |
| 整体架构 | Fig.3 / Sec.4 | 第 6 页 |
| Gaussian-Text 融合 | Sec.4.1, Eq.1–2 | 第 6–7 页 |
| 粒度选择 + 解码器 | Sec.4.2, Eq.3–9 | 第 7 页 |
| CMSA | Sec.4.3, Eq.10–13 | 第 7 页 |
| 训练目标 | Sec.4.4, Eq.14–16 | 第 7–8 页 |
| 实验设置与迁移 | Sec.5.1–5.2 / Table 2 | 第 8 页 |
| 主对比 | Sec.5.3 / Table 3, Fig.4–5 | 第 8–10 页 |
| 定性 + 真实案例 | Sec.5.4–5.5 / Fig.4–5 | 第 9–10 页 |
| 输入参数选择 | 附录 Sec.7.1 / Table 4 | 第 11 页 |
| 指标定义 | 附录 Sec.7.2 | 第 11–12 页 |
| 消融（语言 / CMSA） | 附录 Sec.9.4 / Table 8–9 | 第 14–15 页 |
| 失败分析 | 附录 Sec.9.5 / Fig.9 | 第 15 页 |
| 潜在应用 | Sec.10 | 第 16–18 页 |

> [!note] 原文定位页码说明
> 本论文 PDF 无印刷页码字段，第 1–21 页为连续 PDF 页。上表「第 N 页」指 PDF 页序（封面/摘要为 1–2，正文续接）。具体图表可在 PDF 内按名称检索。

## 15. 待办与复现检查清单

- [ ] 以清晰 PDF 页核对 GS 实例数（23,677 vs 23,672）与点云数（8,354 vs 8,231）的最终口径。
- [ ] 核对 Table 3（主对比 30.25/17.31）与 Table 9（RoBERTa 33.03/18.91）的配置差异，确认是否为同一模型不同训练设定。
- [ ] 核对语言模型类型标注（BART/GPT2 的 decoder/encoder-decoder 标签疑似颠倒）。
- [ ] 检查 GitHub 是否公开完整训练/评估代码、split、指令模板与 ⟨Aff⟩ 注入脚本。
- [ ] 对照 ShapeSplat [41] 与 LASO [30] 原始论文确认数据合并细节（scissors/refrigerator 剔除、18 Gaussian/组合标注来源）。
- [ ] 设计实验验证 H1–H4（扩标注、强 LM、对齐损失解耦、多通道多语义输出）。
- [ ] 与 Segment Any 3D Gaussians / LangSplat / GraspSplats 等方法建立对比笔记。

## 16. 相关链接

- arXiv：https://arxiv.org/abs/2504.11218 （v2, 2025-04-16）
- 代码仓库：https://github.com/HCPLab-SYSU/3DAffordSplat
- 原始 PDF：[[../1-inbox/2026-7-13DAffordSplat.pdf]]
- 关联数据集/方法：ShapeSplat [41]、LASO [30]、3DAffordanceNet [10]、IAGNet [63]、PointRefer [30]、Segment Any 3D Gaussians [3]、LangSplat [45]、Feature3DGS [70]、GraspSplats [19]

---

> [!abstract] 最终评价
> 3DAffordSplat 是一篇「立桩型」工作：它首次把 3DGS 与 affordance reasoning 正式结合起来，提供了稀缺的大规模标注数据集与可复用的跨模态对齐思路（CMSA + 结构一致性先验 + Chamfer 加权），并诚实暴露了 seen 上 CMSA 负向、unseen 仍弱、LM 生成差等真实问题。其方法本身偏标准（PointNet++ + 注意力 + 动态卷积），核心增量在数据与对齐；论文 Related Work 提出的「动态多语义描述符」主张尚未在方法中充分兑现，标注口径也有多处不一致。对后续研究，最值得沿用的是跨模态结构对齐范式与 ⟨Aff⟩ 锚定技巧，最值得改进的是多属性表征、强 LM 骨干、标注扩展与真实世界定量评测。
