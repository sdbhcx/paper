---
title: "对比：2D 语义迁移到 3D 的三种做法 — Unlocking(CMAT) vs GEAL vs PointGS"
aliases:
  - 2D-to-3D semantic transfer comparison
  - CMAT vs GEAL vs PointGS
  - 2D 语义迁移 3D 对比
type: cross-paper-comparison
research_area:
  - 2D-3D cross-modal semantic transfer
  - 3D affordance segmentation
  - unsupervised 3D point cloud segmentation
  - 3D Gaussian Splatting
papers_compared:
  - Unlocking 3D Affordance Segmentation with 2D Semantic Knowledge (Huang et al., CVPR 2026)
  - GEAL: Generalizable 3D Affordance Learning with Cross-Modal Consistency (Lu et al., CVPR 2025)
  - PointGS: Semantic-Consistent Unsupervised 3D Point Cloud Segmentation with 3D Gaussian Splatting (Song et al., CVPR 2026)
status: analyzed
created: 2026-08-12
updated: 2026-08-12
tags:
  - comparison
  - 2D-to-3D
  - affordance
  - 3DGS
  - cross-modal
related_notes:
  - "2-Ai work/Unlocking_3D_Affordance_Segmentation_with_2D_Semantic_Knowledge/Unlocking_3D_Affordance_Segmentation_with_2D_Semantic_Knowledge.md"
  - "2-Ai work/GEAL - Generalizable 3D Affordance Learning with Cross-Modal Consistency/"
  - "2-Ai work/PointGS - Semantic-Consistent Unsupervised 3D Point Cloud Segmentation with 3D Gaussian.md"
---

# 对比：2D 语义迁移到 3D 的三种做法

> 核心问题：三者都认同「稀疏点云缺乏语义、2D 基础模型语义丰富」，都需要一个 2D↔3D 对应关系机制；但 **2D 语义被用在了完全不同的环节**。

## 一句话定位

- **Unlocking（CMAT, CVPR2026）** → 当作 **预训练监督信号**，把 2D 的「功能部件结构」蒸馏进 3D 编码器的表征里。
- **GEAL（CVPR2025）** → 当作 **双分支里的教师**，用一致性对齐让 3D 分支模仿 2D 分支。
- **PointGS（CVPR2026）** → 当作 **无监督的伪标签源**，SAM 的掩码经对比学习灌注到高斯，再传回点。

---

## 各自做法

### ① Unlocking 3D Affordance Segmentation with 2D Semantic Knowledge（Huang et al.）
- **不直接用 3DGS**。Stage 0：对 Objaverse/Behavior-1K 的 1 万+ 模型渲染多视角 RGB，用冻结的 **DINOv3** 取密集特征图，反投影+插值到 3D 点，得到逐点 2D 语义描述符 F²ᴰ。
- **CMAT（Cross-Modal Affinity Transfer）**：预训练 3D 骨干时，**对齐「3D 块亲和矩阵」与「2D 块亲和矩阵」**——强迫 3D 编码器学到 2D 所隐含的「哪些点属于同一功能部件」的关系结构，而非逐点特征硬匹配；辅以几何重建、特征多样性两个辅助损失。
- 2D 模型只离线产目标，推理只用 3D 分支 + 轻量 segmentor（文本/视觉 prompt 经 cross-attention 注入）。

### ② GEAL（Lu et al.）
- **双分支 + 3DGS 渲染器**。3D 分支用 PointNet++ 处理点云；2D 分支把点云经 **3D Gaussian Splatting** 渲染成逼真多视角图像（深度图+colormap），喂给冻结的 **DINOv2**（文本用 RoBERTa）。
- 迁移机制是 **2D-3D 一致性对齐（CAM）+ 粒度自适应融合（GAFM）**：2D 分支先在带标签的渲染空间学好，再冻结，训练 3D 分支与 2D 分支在共享嵌入空间保持一致。两阶段训练，推理只用 3D 分支。
- 3DGS 在这里的作用是**把稀疏点云变成 DINOv2 能吃的高质量 2D 输入**。

### ③ PointGS（Song et al.）
- 任务是无监督 3D 点云语义分割。**3DGS 作为统一中间表征**弥合「离散点 ↔ 连续图像」的域差。
- 流程：稀疏 RGB 点云 → 多视角投影 → 3D-GS 重建**稠密高斯场**（填补空洞、编码遮挡，消除投影重叠导致的语义混淆）→ 渲染多视角图像 → **SAM** 出掩码 → **尺度感知对比学习（SAGA 式）**把语义蒸馏进高斯基元的亲和特征，保证跨视角一致 → 两步 ICP 对齐 + 最近邻把标签传回原始点。
- 2D 语义 = SAM 的掩码提案（无类别名、无需标注），靠对比学习变成高斯上的伪标签。

---

## 对比表

| 维度 | Unlocking (CMAT) | GEAL | PointGS |
|---|---|---|---|
| 下游任务 | 提示驱动的 3D affordance 分割（有标签微调） | 通用/鲁棒 3D affordance 学习（有标签） | 无监督 3D 点云语义分割 |
| 2D 知识源 | DINOv3 密集特征 | DINOv2 特征 + RoBERTa 文本 | SAM 掩码 |
| 是否用 3DGS | **否**（直接特征提升/lifting） | **是**，作为 3D→2D 渲染器 | **是**，作为统一中间表征（双向桥） |
| 迁移机制 | **亲和度对齐**（3D 块亲和 ↔ 2D 块亲和）+ 重建/多样性辅助损失 | **双分支一致性对齐**（2D 教师→3D 学生）+ 粒度自适应融合 | **尺度感知对比蒸馏**（SAM 掩码→高斯特征）+ ICP+NN 标签回传 |
| 2D 语义的角色 | 离线预训练目标（结构化表征） | 并行教师分支（一致性监督） | 伪标签源（无监督标注） |
| 对齐粒度 | 部件级关系结构 | 多尺度特征一致性 | 多粒度（scale gate）跨视角一致 |
| 是否需 affordance/语义标注 | 预训练阶段不需要（仅 2D 特征） | 需要（渲染 affordance 掩码监督） | 完全不需要 |

---

## 核心差异提炼（四个角度）

1. **2D 语义「用来干什么」**：CMAT 是**表征预训练**（让 3D 骨干自带功能部件结构）；GEAL 是**教师蒸馏**（2D 分支在线教 3D 分支）；PointGS 是**标签传播**（SAM 掩码当伪标签）。
2. **3DGS 的角色**：Unlocking 干脆不用；GEAL 只做**渲染器**（3D→2D）；PointGS 做**稠密中间场**（既重建又回传，双向）。
3. **对齐目标函数**：亲和矩阵对齐（关系结构）vs 一致性损失（特征空间一致）vs 对比损失（跨视角掩码一致）。
4. **2D 模型类型**：DINO 系「识别型 VFM」（语义表征）vs SAM「分割型 VFM」（掩码提案）——这决定了迁移的是「语义特征」还是「分割提案」。

---

## 共性 insight

三者都靠「先把 2D 知识落到 3D 的某个几何对应上（点 / 渲染像素 / 高斯基元），再让 3D 侧去匹配/对齐」来突破点云语义匮乏；区别只在于落点选在哪、匹配用什么损失、匹配结果服务于预训练还是最终分割。

---

## 原文待核对点
- PointGS 的 scale gate 公式与 scale sM 计算（原文公式细节需回看）。
- CMAT 亲和损失的精确形式（亲和矩阵如何构造、相似度度量）。
- GEAL 的 3DGS 渲染中高斯协方差/不透明度是否参与训练（papernotes 称固定以保留几何）。

（以上细节以各论文原文为准；详见 related_notes 中的单篇笔记。）
