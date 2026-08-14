---
title: "对比：把语义/结构知识迁移到 3D 表征的七种做法 — Unlocking / GEAL / PointGS / IAAO / 3DAffordSplat / Fun3DU / LISA-3D"
aliases:
  - 2D-to-3D semantic transfer comparison
  - CMAT vs GEAL vs PointGS vs IAAO vs 3DAffordSplat vs Fun3DU vs LISA-3D
  - 2D 语义迁移 3D 对比
  - 3DGS affordance 方法谱系
  - VLM 2D 分割到 3D 对比
type: cross-paper-comparison
research_area:
  - 2D-3D cross-modal semantic transfer
  - 3D affordance segmentation
  - unsupervised 3D point cloud segmentation
  - articulated object affordance + motion
  - 3D Gaussian Splatting as representation
  - VLM-driven 2D segmentation to 3D
papers_compared:
  - Unlocking 3D Affordance Segmentation with 2D Semantic Knowledge (Huang et al., CVPR 2026)
  - GEAL: Generalizable 3D Affordance Learning with Cross-Modal Consistency (Lu et al., CVPR 2025)
  - PointGS: Semantic-Consistent Unsupervised 3D Point Cloud Segmentation with 3D Gaussian Splatting (Song et al., CVPR 2026)
  - IAAO: Interactive Affordance Learning for Articulated Objects in 3D Environments (CVPR 2025)
  - 3DAffordSplat: Efficient Affordance Reasoning with 3D Gaussians (arXiv:2504.11218, 2025)
  - Fun3DU: Functionality Understanding and Segmentation in 3D Scenes (CVPR 2025, arXiv:2411.16310)
  - LISA-3D: Lifting Language-Image Segmentation to 3D via Multi-View Consistency (arXiv 2025)
status: analyzed
created: 2026-08-12
updated: 2026-08-14
tags:
  - comparison
  - 2D-to-3D
  - affordance
  - 3DGS
  - cross-modal
  - articulated-object
  - VLM
  - zero-shot
related_notes:
  - "2-Ai work/Unlocking_3D_Affordance_Segmentation_with_2D_Semantic_Knowledge/Unlocking_3D_Affordance_Segmentation_with_2D_Semantic_Knowledge.md"
  - "2-Ai work/GEAL - Generalizable 3D Affordance Learning with Cross-Modal Consistency/"
  - "2-Ai work/PointGS - Semantic-Consistent Unsupervised 3D Point Cloud Segmentation with 3D Gaussian.md"
  - "2-Ai work/IAAO_Interactive_Affordance_Learning_for_Articulated_Objects_in_3D_Environments/IAAO_Interactive_Affordance_Learning_for_Articulated_Objects_in_3D_Environments.md"
  - "2-Ai work/3DAffordSplat - Efficient Affordance Reasoning with 3D Gaussians.md"
  - "2-Ai work/Fun3DU - Functionality Understanding and Segmentation in 3D Scenes.md"
  - "2-Ai work/LISA-3D - Lifting Language-Image Segmentation to 3D via Multi-View Consistency.md"
---

# 对比：把语义/结构知识迁移到 3D 表征的七种做法

> 核心问题：点云语义匮乏，2D 基础模型（VFM）或语言驱动的多模态大模型（VLM）或已有 3D 监督如何被「搬」进 3D 表征。
> 七篇可视为一条**谱系**：从「不用 3DGS、直接对齐点云」（A），到「把 3DGS 当一体化语义载体」（D），再到「连 2D 都不用、只做 3D→3D 结构对齐」的对照边界（E），最后分出一支**全新的「VLM 指令驱动」分支（F）**——它的 2D 端不是 VFM，而是能吃自然语言指令、输出分割掩码的多模态大模型，靠多视角一致性 / 投票聚合把掩码提升到 3D。

---

## 谱系速览（按 2D 知识的「类型 / 落点」与「3DGS 角色」）

```
流派 A  不用 3DGS，2D 特征 lifting 后对齐点云（VFM 特征）
        └─ Unlocking (CMAT)        : 2D 块亲和 → 3D 块亲和对齐（预训练目标）

流派 B  3DGS 当渲染器（3D→2D）→ 2D 模型 → 一致性回流点云（VFM 特征）
        └─ GEAL                    : 3DGS 渲染喂 DINOv2，双分支一致性对齐（教师）

流派 C  3DGS 当稠密中间桥（重建→渲染→蒸馏→回传点）（VFM 掩码）
        └─ PointGS                 : 3DGS 场接 SAM 掩码，对比蒸馏 + ICP/NN 回传点（伪标签）

流派 D  3DGS 当一体化载体，把多类 2D VFM 蒸馏进场内（VFM 组合）
        └─ IAAO                    : CLIP+SAM+DINOv2 蒸馏进高斯潜空间，场即语义/铰接接口

流派 E（对照，非 2D→3D）
        3DGS 当表征 + 目标，做 3D→3D 结构对齐
        └─ 3DAffordSplat (CMSA)    : 点云↔3DGS 结构亲和对齐，不蒸馏 2D VFM

流派 F（NEW，VLM 指令驱动 2D 分割 → 提升/聚合到 3D）
        ├─ Fun3DU   : LLM(CoT) 解析任务 → VLM+SAM 多视角分割功能物体 → 2D-3D 投票聚合到点云（零训练）
        └─ LISA-3D  : LISA(推理分割 VLM) + LoRA 多视角一致性 → 冻结 SAM-3D 提升为 3D 高斯/网格
```

> **关键分水岭**：A–E 的 2D 知识源都是 **VFM（DINO / SAM / CLIP，纯视觉或视觉-文本对齐）**；F 的 2D 知识源是 **VLM（LISA / VLM+SAM+LLM，接收自然语言指令做推理分割）**。前者给「特征 / 掩码」，后者给「带语言推理的分割提案」。

---

## 各自做法（精简）

### ① Unlocking 3D Affordance Segmentation with 2D Semantic Knowledge（Huang et al., CVPR2026）
- **不用 3DGS**。Stage 0：Objaverse/Behavior-1K 1 万+ 模型渲染多视角 RGB，冻结 **DINOv3** 取密集特征，反投影+插值到 3D 点 → 逐点 2D 语义描述符 F²ᴰ。
- **CMAT（Cross-Modal Affinity Transfer）**：预训练 3D 骨干时对齐「3D 块亲和矩阵 ↔ 2D 块亲和矩阵」，迫使 3D 编码器学到 2D 隐含的「哪些点属同一功能部件」关系结构；辅几何重建、特征多样性损失。
- 2D 模型仅离线产目标，推理只用 3D 分支 + 轻量 segmentor（文本/视觉 prompt 经 cross-attention）。

### ② GEAL（Lu et al., CVPR2025）
- **双分支 + 3DGS 渲染器**。3D 分支 PointNet++ 处理点云；2D 分支把点云经 **3DGS** 渲染成逼真多视角图像（深度+colormap），喂冻结 **DINOv2**（文本 RoBERTa）。
- 迁移机制：**2D-3D 一致性对齐（CAM）+ 粒度自适应融合（GAFM）**。先训好 2D 分支，再冻结，训 3D 分支与之在共享空间一致。两阶段，推理只用 3D 分支。
- 3DGS 作用：把稀疏点云变成 DINOv2 能吃的高质量 2D 输入。

### ③ PointGS（Song et al., CVPR2026）
- 无监督 3D 点云语义分割。**3DGS 当统一中间表征**弥合「离散点 ↔ 连续图像」域差。
- 流程：稀疏 RGB 点云 → 多视角投影 → 3D-GS 重建**稠密高斯场**（填洞、编码遮挡，消除投影重叠的语义混淆）→ 渲染多视角图 → **SAM** 出掩码 → **尺度感知对比学习（SAGA 式）**把语义蒸馏进高斯基元亲和特征（跨视角一致）→ 两步 ICP 对齐 + 最近邻把标签传回原始点。
- 2D 语义 = SAM 掩码提案（无类别名、无需标注），靠对比学习变伪标签。

### ④ IAAO — Interactive Affordance Learning for Articulated Objects in 3D Environments（CVPR2025）
- 任务：铰接物体的「可供性 + 运动学」联合恢复（两状态多视角图像，无需相机位姿/静态对齐假设）。
- **3DGS 是核心场景表征（几何+语义+标签一体化载体）**。每状态先用 SfM 稀疏点云初始化 3DGS；再把 **CLIP、SAM、DINOv2 三类 2D 基础模型输出蒸馏进高斯基元的低维潜空间**，由三分支小解码器 MLP 在反向传播时还原为 2D 特征。
  - **SAM** → 跨视角掩码图聚类 → 标签场（交叉熵）把「哪些高斯属同一物体/部件」烘焙进 3D；
  - **CLIP** → 实例/部件级文本特征，供语言查询可供性（文本相似度命中掩码）；
  - **DINOv2** → 像素特征，驱动 2D-3D 对应（把 3D 高斯投到目标状态视角、在部件掩码内做加权像素匹配、过滤碰撞对）以恢复局部铰接。
- 2D 语义的角色：**作为场内置的可查询语义/几何信号**，既是可供性接口（CLIP 文本），又是铰接恢复命脉（DINOv2 匹配，消融显示去掉 L_match 部件运动飙 78×）。

### ⑤ 3DAffordSplat — Efficient Affordance Reasoning with 3D Gaussians（2025）
- 交付物：首个 3DGS+点云+语言的 affordance 数据集 + **AffordSplatNet**。
- **关键点：它不做 2D VFM→3D 的语义迁移**。其「跨模态」是 **点云 ↔ 3DGS 的 3D→3D 结构对齐（CMSA，Pretrain 阶段）**：核心先验是「同物体类别，affordance 区域相对整体结构的空间关系不变」；用共享 cross-attention 分别算点云与 3DGS 的结构亲和矩阵，投影到隐空间对齐，并用 **Chamfer Distance 给跨模态样本加权**解决二者形状/结构差异。
- 3DGS 既是**表征**（连续稠密、可实时渲染、补点云稀疏缺陷）又是**监督目标**（人工标每 Gaussian 的 affordance，Finetune 精修）。文本（RoBERTa）仅作 prompt，不蒸馏语义。
- 它是本对比网的**对照边界**：揭示「不依赖 2D 语义、仅靠 3D 内部模态对齐 + 标注」也能立起 3DGS affordance 表征。

### ⑥ Fun3DU — Functionality Understanding and Segmentation in 3D Scenes（CVPR2025, arXiv:2411.16310）【流派 F】
- 任务：3D 场景**功能性理解/分割**——给定自然语言任务描述（如"打开天花板灯"），在真实 3D 场景中**定位并分割功能性交互元素**（开关/把手/旋钮/按钮）。这类元素常未在描述中显式提及，需世界知识推理 + 精细空间感知。
- 动机：开放词汇 3D 分割（OpenMask3D / LERF / OpenIns3D）严重偏向大物体，对小型功能元素失效；3D 数据太少无法训功能理解模型，但 2D 预训练 **VLM/LLM 拥有丰富世界知识**与精细视觉感知。
- 流程（**零训练**）：(1) **LLM 链式思维**解析任务描述，推理出需交互的功能物体名称；(2) 在精选视角上用 **VLM 定位 + SAM 分割**该功能物体；(3) 多视角掩码经投影映射回 3D 点云，**2D-3D 投票聚合**得最终 3D 功能分割。
- 2D 语义角色：**推理 + 分割提案源**（功能物体定位），靠**投票聚合**（而非特征对齐/蒸馏）落到 3D。
- 结果：SceneFun3D 上 mIoU +13.2 超开放词汇基线。**最贴合 GEAL 的 affordance 线**，且零训练、VLM 指令驱动。

### ⑦ LISA-3D — Lifting Language-Image Segmentation to 3D via Multi-View Consistency（arXiv 2025）【流派 F】
- 任务：**文本驱动 3D 重建/分割（language-to-3D）**——从自然语言指令（如"靠窗的蓝椅子"）得到干净 3D 高斯/网格。
- 动机：2D VLM 擅长按复杂语言指令分割，但搬 3D 时跨视角掩码不一致会产生 ghost/浮片；且缺大规模 3D-text 数据。
- 两阶段：(1) **几何感知微调**：取 **LISA（2D 指令跟随推理分割 VLM）**，在 attention 注入 **LoRA（仅 ~1.1% 参数 ≈11.6M）**，用 **多视角一致性损失 L_geo** 自监督——已知相机位姿+深度，把视角 A 掩码 warp 到视角 B，惩罚与直接预测的偏差，迫使模型学 3D 结构、跨视角掩码稳定；(2) **掩码引导提升**：微调后 LISA 出多视角一致掩码，拼接 RGB 成 RGBA prompt 喂**冻结 SAM-3D**，零样本聚合为 3D 实体。
- 2D 语义角色：**指令驱动的 2D 分割**（经几何一致性微调的 VLM）；3DGS 作为最终输出表征。
- 结果：ScanRefer / Nr3D 上语言→3D 精度 +15.6，2D mIoU 10.2→25.4，重建 <40s，零样本。

---

## 主对比表（7 列）

| 维度 | Unlocking (CMAT) | GEAL | PointGS | IAAO | 3DAffordSplat | Fun3DU | LISA-3D |
|---|---|---|---|---|---|---|---|
| 下游任务 | 提示驱动 affordance 分割 | 通用/鲁棒 affordance 学习 | 无监督 3D 语义分割 | 铰接物体 affordance + 运动 | 3DGS affordance 推理 | 3D 场景功能性理解/分割（affordance） | 文本驱动 3D 重建/分割（language-to-3D） |
| 2D 知识源 | DINOv3 特征 | DINOv2 + RoBERTa | SAM 掩码 | CLIP+SAM+DINOv2 三合一 | 无 2D VFM（RoBERTa 仅 prompt） | **LLM(CoT)+VLM+SAM**（VLM 指令驱动） | **LISA**（推理分割 VLM） |
| 是否用 3DGS | 否 | 是（渲染器） | 是（双向桥） | 是（一体化载体） | 是（表征+目标） | 否（点云+RGBD 投票聚合） | 是（输出为高斯/网格，SAM-3D 重建） |
| 迁移机制 | 块亲和对齐（2D↑→3D） | 双分支一致性（2D 教师→3D） | 对比蒸馏（SAM→高斯）→ICP+NN | 三解码器蒸馏 2D 特征进高斯潜空间 + 掩码聚类标签场 | CMSA：点云↔3DGS **结构亲和对齐（3D→3D）** | **2D-3D 投票聚合**（多视角掩码投影回头，多数投票），零训练 | **LoRA 多视角一致性**（L_geo 自监督 warp）训 LISA→一致掩码喂冻结 SAM-3D |
| 2D 语义角色 | 离线预训练目标 | 教师分支 | 无监督伪标签 | 场内置语义/几何接口（查询+匹配） | 不涉及 2D 语义 | 推理+分割提案（功能物体定位），投票聚合到 3D | 指令驱动 2D 分割（几何感知微调的 VLM） |
| 3DGS 方向 | — | 3D→2D（外送 2D 模型） | 3D⇄3D（重建+回传点） | 2D→3DGS（进场内） | 点云→3DGS（进场内） | 不涉及 | 2D→3DGS（重建） |
| 对齐粒度 | 部件级关系 | 多尺度一致 | 多粒度跨视角 | 分层（实例/部件/像素） | 结构/类别级 | 功能物体级（多视角投票） | 掩码级跨视角一致 |
| 是否需标注 | 预训练不需 | 需（affordance） | 不需 | 不需（自监督，两状态） | 需（每 Gaussian affordance 标注） | 不需（零样本） | 不需 3D 标注（需位姿+深度自监督） |
| **VLM 指令驱动？** | 否（VFM） | 否（VFM） | 否（VFM） | 否（VFM 组合） | 否 | **是** | **是** |

---

## 关键差异提炼（五个角度）

1. **2D 语义「用来干什么」**：CMAT=**预训练目标**；GEAL=**教师**；PointGS=**伪标签**；IAAO=**场内置接口**（查询+匹配）；3DAffordSplat=**不用 2D 语义**（纯 3D→3D）；Fun3DU=**推理+分割提案 + 投票聚合**；LISA-3D=**指令分割 + 一致性提升**。
2. **2D 模型类型（VFM vs VLM 分水岭）**：A–E 全是 **VFM**（DINO 识别型 / SAM 分割型 / CLIP 对齐型 / 组合拳）；F 是 **VLM（LISA / VLM+SAM+LLM）**，接收自然语言指令做推理分割。这是 ⑥⑦ 与前面五篇最本质的区别。
3. **3DGS 的角色**：无（Unlocking）/ 渲染器（GEAL）/ 双向桥（PointGS）/ 一体化载体（IAAO）/ 表征+目标（3DAffordSplat）/ 不涉及（Fun3DU）/ 最终输出表征（LISA-3D）。GEAL 与 IAAO 都用 3DGS 但方向相反——前者「3D→2D 外送」，后者「2D→3DGS 内收」。
4. **对齐目标函数**：亲和矩阵对齐 / 一致性损失 / 对比损失 / 解码器重构+标签场 / 结构亲和对齐（Chamfer 加权）/ **投票聚合（Fun3DU）** / **多视角一致性损失 L_geo（LISA-3D）**。
5. **提升机制（F 支特有）**：F 不靠「特征空间对齐」，而靠**跨视角一致性（LISA-3D 的 LoRA+L_geo）**或**多视角投票聚合（Fun3DU）**把 2D 掩码稳定/聚合到 3D——novelty 落在「如何让 2D VLM 的分割在 3D 上一致」，而非 VLM 本身。

---

## 流派 F 对你的 GEAL 路线的启示
- **不要用 VLM 当"换编码器"的补丁**。把 GEAL 的 DINOv2 换成某个 VLM 仍是组件替换（你定的方法论红线：换编码器不构成新颖性）。
- **真正可借鉴的架构级增量**是 F 支的**提升/聚合机制**：
  - **Fun3DU 的 2D-3D 投票聚合** → 可作为「用语言指令驱动的多视角功能分割提案，经投票聚合到 3DGS 高斯」的接口，与你路线①（MLLM 意图教师）天然互补；
  - **LISA-3D 的多视角一致性微调（LoRA+L_geo）** → 可作为「让 2D VLM 分割在 3D 上自洽」的训练范式，与你路线②（生成式完整几何）的补全结果互相引导。
- **危险对照**：若只把 Fun3DU/LISA-3D 的 VLM 直接塞进 GEAL 双分支当 2D 教师，审稿一句话即可驳回（=换编码器）。必须回答"这是范式级改动还是 encoder swap"。

---

## 3DAffordSplat 作为对照边界的启示
- 它证明「把知识搬进 3D」不只有 2D→3D 一条路：**点云 affordance 监督（已是 3D 模态）也可通过对齐落到 3DGS 上**，且 3DGS 本身能成为比点云更强的表征。
- 与 IAAO 对比尤其有趣：IAAO 把 **2D VFM** 蒸馏进 3DGS 场，3DAffordSplat 把 **3D 点云标注**对齐进 3DGS 场——二者都「把某种外部知识灌进 3DGS 高斯」，区别仅在知识来源是 2D 还是 3D。

---

## 共性 insight
- 七篇都认同「稀疏/离散 3D 表征语义匮乏，需外部知识补强」，都需要一个**几何对应关系**机制（点 / 渲染像素 / 高斯基元 / 结构亲和 / 多视角投票）。
- 区别只在于：知识来源（2D VFM vs 2D VLM vs 3D 标注）、落点（点云 vs 渲染图 vs 高斯场 vs 结构矩阵 vs 掩码投票）、对齐损失、以及匹配结果服务于预训练还是最终分割/重建。
- 3DGS 在 B/C/D/E/⑦ 六篇里都是枢纽，但角色从「工具（渲染/桥）」逐渐升级为「本体（语义/表征载体/输出）」。
- **新增的 F 支揭示趋势**：当 2D 端升级为语言驱动的 VLM，迁移瓶颈从「特征对齐」转为「跨视角一致性 / 投票聚合」，且往往能做到零训练、开放词汇。

---

## 原文待核对点
- PointGS 的 scale gate 公式与 scale sM 计算；CMAT 亲和损失构造；GEAL 的 3DGS 渲染是否参与训练。
- IAAO：L_match / L_mask / L_label 的精确权重与平衡（见论文补充）；DINO 相似度 softmax 温度、碰撞过滤阈值。
- 3DAffordSplat：GS 实例数 23,677 vs 23,672、点云 8,354 vs 8,231 vs 8.4k 多处口径不一致（以 Table 5 为准）。
- **Fun3DU**：VLM 具体型号（笔记据 WebSearch 摘要，待回看原文）；投票聚合的逐视角权重与投影映射实现；是否真零训练（有无轻量 adaptor finetune）。
- **LISA-3D**：LISA 具体版本；「SAM-3D」指哪一现成模块（SAM3D / 自研）；L_geo 的精确 warp 与损失形式；LoRA 注入的具体 attention 层。

（细节以各论文原文 / related_notes 单篇笔记为准。）
