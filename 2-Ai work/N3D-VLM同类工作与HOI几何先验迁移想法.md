---
title: "类 N3D-VLM 的 3D 空间理解工作 & HOI→物体几何先验迁移想法调研"
authors: "AI 调研 (基于 web 检索, 未逐篇精读)"
date: "2026-08-31"
tags: [3D-VLM, 3D空间推理, N3D-VLM, VGGT, HOI点云, affordance, 知识蒸馏, 几何先验迁移]
status: "草稿-待精读核对"
---

# 类 N3D-VLM 的 3D 空间理解工作 & "HOI→物体几何先验迁移"想法调研

> 事实来源：web 检索到的论文摘要/主页/综述（N3D-VLM、VGGT、VAGNet/VideoAfford、SRKD/HVDistill、RoboPoint/PAVLM 等）。
> **未逐篇精读原文**，引用具体数值与模块前请回查原论文。文中"推演"标记为分析者观点。

---

## 1. N3D-VLM 核心机制（确认）

- **出处**：Wang et al., *N3D-VLM: Native 3D Grounding Enables Accurate Spatial Reasoning in Vision-Language Models*, arXiv 2512.16561 (2025-12), Tencent AI Lab + HKUST。CVPR 级别投稿。
- **核心主张**：当前 VLM 只做 2D 感知、缺乏原生 3D 物体感知，导致空间关系/深度推理弱。解决思路 = **把"3D 空间理解"拆成两个能力：3D 物体定位(grounding) → 3D 空间推理(reasoning)**。
- **架构要点**：
  1. 输入 RGB-D；用 **depth-aware positional encoding + 结构化 3D bbox 表示**（相机坐标系下的尺度/位置），让 token 直接携带度量 3D 坐标。
  2. **Native 3D grounding head** 直接预测 3D bounding box（pixel space / camera space）。
  3. **级联推理**：先 3D grounding（定位物体），再做 **CoT 式 3D 空间推理**（物体间关系、距离、属性）。
- **数据飞轮（关键贡献）**：用深度估计把大规模 2D 检测标注（COCO/OpenImages/Objects365）**lift 到 3D**，生成 3D 检测/grounding/空间 QA 数据，规模比最大单图 3D 检测集大 6×；并用模板 CoT 问题 + LLM 改写生成 3D 空间推理 QA。
- **局限（作者自述）**：镜面反射误检、密集场景漏检；依赖 RGB 深度估计，低质量深度下受限。

**一句话**：N3D-VLM 的本质是"原生 3D 感知（显式 3D bbox/位置编码）作为空间推理的可解释基础"，而非端到端从 2D 直接猜答案。

---

## 2. 类 N3D-VLM 的 3D 空间理解 VLM 谱系

按"如何把 3D 几何喂给 LLM/VLM"可分成四类（含与 N3D-VLM 的关系）：

### A. 点云—语言对齐（point cloud 直接 tokenize 进 LLM）
| 工作 | 做法 | 与 N3D-VLM 关系 |
|---|---|---|
| **PointLLM** (Guo et al. 2023) | 彩色点云→point encoder→投影到 LLM 嵌入；PEFT 对齐 | 单物体描述/QA，无显式 grounding |
| **3D-LLM** (Hong et al. 2023) | 多视图渲染特征+3D点嵌入，位置 token | 场景级 QA/字幕，非原生度量 3D |
| **LEO** (2024) | 物体/场景/全局三级描述数据联合对齐；embodied 规划 | 更偏 agent，grounding 弱 |
| **GPT4Point** (Qi et al. 2024) | BLIP2 式两阶段：PTC/PTM/PTG 对齐 | 点-文本对比/匹配/生成 |
| **Grounded 3D-LLM** (Chen et al. 2024) | 两阶段：先点-文对齐预训练，再 LLM 多任务微调 | 显式 grounding + 推理 |
| **PAVLM** (2024) | **点云 affordance** + 几何引导传播 + LLM 隐藏嵌入 | 最贴近你的 affordance 主线（见 §3-D） |

### B. 3D 位置编码注入（"native 3D token"，与 N3D-VLM 最像）
| 工作 | 做法 |
|---|---|
| **LLaVA-3D** (Zhu et al. 2024) | 2D CLIP patch + 来自 depth/相机参数的 **3D positional embedding** → 3D patch，无需外置 3D segmentor |
| **SR-3D** (Cheng et al. 2025) | 单/多视图统一：2D 特征 + depth 派生的 3D PE；支持 region prompting 跨视图传播 |
| **SpatialLM** (Mao et al. 2025) | 从单目视频/RGB-D/LiDAR 点云抽密集空间特征→LLM 输出语义+有向 bbox+墙门窗结构 |
| **N3D-VLM** | 如上，depth-aware PE + 原生 3D grounding + CoT |
> 这一支的共同思想：**几何结构直接注入 token 表示 / 对齐层**，比"先 2D 后投影"更高效、可解释。N3D-VLM 的增量在于"原生 3D 定位→推理的级联 + 数据飞轮"。

### C. 空间推理数据合成（弥补 3D 标注稀缺）
- **SpatialVLM** (Chen et al. 2024)：自动合成 **20亿 QA** 对（1000万真实图），注入定性+定量 3D 关系（含度量距离），训练出 CoT 与机器人可用空间能力。
- **SAT** (Ray et al. 2025)：在 22K ProcTHOR 场景合成 175K QA，提升静态/动态空间能力。
- **N3D-VLM 的 lift 管线**：2D→3D 标注提升，与上述同思路，但聚焦 grounding+推理联合。

### D. 3D Affordance VLM（**与你的研究主线最直接相关**）
| 工作 | 输入 → 输出 | 要点 |
|---|---|---|
| **RoboPoint** (Yuan et al. 2024) | RGB 图 + 语言 → **2D affordance 关键点**（深度投影成 3D） | 合成数据训练，无需真实演示；空间关系/自由空间 affordance；下游操作/导航/AR |
| **PAVLM** (2024) | 点云 + 语言 → affordance | 几何引导传播模块 + LLM 隐藏嵌入；3D-AffordanceNet 上 SOTA，支持开世界 |
| **AffordanceLLM** | 用 MLLM 世界知识做 affordance 推理 | 开放词汇 |
| **O³Afford** (2025) | 源/目标**物体对点云** → 交互 affordance | object-to-object affordance，LLM 生成约束函数 |
| **VAGNet** (2026) | **HOI 视频 + 3D 物体点云** → affordance 区域 | "视频驱动的 3D affordance"，PVAD 数据集（4K 视频/37K 点云/38类/22 affordance）|
| **VideoAfford** (2026) | **HOI 视频 + MLLM** → 3D affordance | VIDA 数据集（38K 视频/16 affordance/22K 点云）；"把 HOI 交互先验迁移进 3D affordance grounding" |

> **重要**：VAGNet 与 VideoAfford 已经把"人类-物体交互(HOI)作为 affordance 的功能监督信号"做出来了——这正是你想法的一个近邻。区别在于它们用**视频帧**作为交互先验，并且是**联合训练**（视频+点云一起进模型），而你的想法是用 **VGGT 重建的 3D HOI 点云**作为几何先验，并通过**蒸馏**让"只有物体点云"的模型学到它。

---

## 3. VGGT → 3D HOI 点云：可行性 & 已有基础

### 3.1 VGGT 能力（确认，CVPR 2025 Best Paper）
- **VGGT: Visual Geometry Grounded Transformer** (Wang et al., CVPR 2025)：前馈网络，从 **1张/几张/上百张视图** 直接推断相机参数、**point maps、depth maps、3D point tracks**。
- 特点：<1 秒重建；DINO patchify + 帧内/全局自注意力交替；可作下游 backbone（如非刚性跟踪、前馈新视角合成）。代码 `facebookresearch/vggt` 已开源。
- **对你想法的意义**：VGGT 是"2D HOI 图像 / 多视图 → 3D 点云"的现成引擎，且能给出逐点 3D 坐标与置信度。

### 3.2 已有 3D HOI 重建数据集（提供真值/验证）
- **BEHAVE** (Bhatnagar et al. 2022)：多人-物交互多视图，20 个物体，带 SMPL+物体 mesh 真值。
- **InterCap** (Huang et al. 2022)：10 物体，IMU+多视图。
- **CHORE** (Xie et al. 2022)：需已知物体模板的交互重建。
- **ProciGen / HDM** (2023)：合成 110万图训练、template-free 交互重建，F-score@0.01m 超越 CHORE/PC²。
- **Open3DHOI / Open4DHOI / IMHD2 / PICO / 4DHOISolver** (2024-2025)：更大规模、更多物体类别、含动作标注（Open4DHOI 135 类、133 动作）。

### 3.3 工程注意点（推演）
- **尺度歧义**：单图 VGGT 存在度量尺度模糊；HOI 建议用**多视图/短视频多帧**输入，尺度更稳。
- **human/object 分离**：VGGT 给的是全局 point map，不带语义。需用 **SAM2 做 2D 人体/物体 mask → 抬升到 3D**（借 VGGT 的 point map/depth），或结合 SMPL 人体先验。
- **对齐**：HOI 场景里的物体点云 vs 你"干净物体点云（canonical）"需配准/对应——可复用 §先前调研的 **functional maps / 配准（GeoTransformer/Predator）**（见 `点云几何相似度迁移学习相关工作.md`）。
- **噪声/遮挡**：重建点云有孔洞，物体自遮挡区域 affordance 难学（O³Afford 也指出这是失败主因之一）。

---

## 4. 你的想法解构 & 定位

> **原话**：「2D HOI 图像通过 VGGT 为 3D HOI 点云，让物体点云学习 HOI 点云的几何先验」

### 4.1 解构为一句话
用一个**teacher**（HOI 点云，包含"人如何使用物体"的完整几何上下文：接触区、人体相对位姿、抓握分布）去监督/蒸馏一个**student**（只有物体点云），使其在**测试时仅见物体点云**也能预测出"若与人交互会发生的几何先验"（接触区/交互热图/相对位姿分布）。

这正好落在你 workspace 主线 **affordance grounding** 上：affordance 本质是"物体被使用时的几何上下文"，HOI 点云是它最自然的监督源。

### 4.2 与已有工作的关系（定位 novelty）
| 已有工作 | 它做了什么 | 你的想法 vs 它 |
|---|---|---|
| VAGNet / VideoAfford | HOI **视频帧** 作交互先验，与 3D 物体点云**联合训练**出 affordance | 你用 **VGGT 重建的 3D HOI 点云**（真 3D 接触几何，非投影帧）；且用**蒸馏**让 student 推理时**无需视频/人** |
| SRKD / HVDistill | **点云间**几何/结构知识蒸馏（teacher→student 同构点云、同任务）| 你是**跨组合点云(HOI)→部件点云(物体)**的几何先验迁移（异质组合→单体），更贴近 affordance 迁移 |
| 3D-AffordanceNet / PAVLM | 点云 affordance，但多用静态/语言线索 | 你的监督来自**重建的 3D 交互几何**，天然含动态接触信息 |
| 先前调研的"两点云几何相似度迁移" | source↔target 形状对应/域适应 | 此处 source=HOI 点云（富几何），target=物体点云（缺交互上下文），是同一族思想的**affordance 实例化** |

**结论（推演）**："用 VGGT 把 HOI 图像变成 3D 点云、再以蒸馏方式把交互几何先验迁移到物体点云"这一**具体组合**——据现有检索——**尚未被直接做过**。最近邻是 VAGNet/VideoAfford（但用视频而非重建 3D 点云、用联合训练而非蒸馏）和 SRKD（但同构点云）。因此这个想法**有清晰的新颖空间**，且工程上所需模块（VGGT、SAM2、点云蒸馏）都已成熟可用。

### 4.3 一个可行的具体管线（推演草案）
1. **数据生成**：收集 2D HOI 图像/短视频 → VGGT 得 3D point map + 相机 → SAM2 得人体/物体 2D mask → 抬升得到 **HOI 点云（人体+物体）** 与 **物体点云（从 HOI 场景抠出）**；同时保留一个**干净 canonical 物体点云**（来自 ShapeNet/Objaverse/BEHAVE 模板）。
2. **Teacher 训练**：在 HOI 点云上训练一个"交互几何预测头"（接触热图 / 人体相对位姿 / affordance 区域），监督来自 VAGNet/VideoAfford 式 HOI 标注或 BEHAVE 的接触/部位标签。
3. **蒸馏到 Student**：Student 仅吃**物体点云**，用 SRKD 式 **affinity-matrix 关系对齐 + 跨样本几何结构对齐**，让 student 特征逼近 teacher 在"同一物体 + 交互上下文"下的几何表征；预测目标（接触区/affordance heatmap）做 feature/response 蒸馏。
4. **推理**：只给一个物体点云 → Student 输出"它若被使用时的几何先验（affordance）"，无需人和视频。

### 4.4 关键研究问题与风险（推演）
- **HOI 点云与物体点云的配准/对应**：不同实例、不同姿态下如何对齐 Teacher/Student 特征空间（可借 functional maps / GeoTransformer，参见先前文档）。
- **蒸馏目标设计**：是蒸馏最终 affordance 热图（response KD），还是中间几何结构（feature/relation KD，如 SRKD）？两者结合更稳。
- **尺度与噪声**：VGGT 单图尺度模糊会污染"几何先验"的度量意义；建议多用视图 + 归一化到 canonical。
- **泛化**：HOI 数据集中在少数物体/动作，student 能否泛化到未见物体类别？（可结合 Open4DHOI 的 135 类/133 动作做规模验证）。
- **与 VAGNet/VideoAfford 的差异化卖点**：必须讲清"为何重建 3D HOI 点云 + 蒸馏"比"直接 HOI 视频联合训练"更好——论点应是：① 测试时免人/免视频，更轻量；② 真 3D 接触几何比投影帧更准；③ 可用 VGGT 从海量 in-the-wild 2D HOI 规模化生成 3D 监督。

---

## 5. 建议优先精读的论文（待写标准精读笔记）
1. **N3D-VLM** (2512.16561) — native 3D grounding + CoT 范式。
2. **VGGT** (2503.11651) — 你的 3D 重建引擎。
3. **VAGNet** (2602.20608) / **VideoAfford** (2602.09638) — HOI→3D affordance 的最近邻，务必对比。
4. **SRKD** (2506.17290) — 点云结构/关系感知蒸馏，可直接借鉴。
5. **PAVLM** (2410.11564) — 点云 affordance + VLM。
6. **RoboPoint** (2406.10721) — 空间 affordance 点预测 VLM。
7. **BEHAVE / Open4DHOI** — 3D HOI 真值数据来源。

---

## 6. 参考文献（检索来源）
- Wang et al. N3D-VLM, arXiv:2512.16561, 2025. https://arxiv.org/abs/2512.16561 / https://n3d-vlm.github.io/
- Wang et al. VGGT: Visual Geometry Grounded Transformer, CVPR 2025. arXiv:2503.11651. https://vgg-t.github.io/
- Mao et al. VAGNet: Grounding 3D Affordance from Human-Object Interactions in Videos, arXiv:2602.20608, 2026.
- Wang et al. VideoAfford: Grounding 3D Affordance from HOI Videos via MLLM, arXiv:2602.09638, 2026.
- Li et al. SRKD: Structure- and Relation-aware Knowledge Distillation for 3D Point Cloud Segmentation, arXiv:2506.17290, 2025.
- Zhang et al. HVDistill: Transferring Knowledge from Images to Point Clouds via Unsupervised Hybrid-View Distillation, 2024.
- Ning et al. Hyperbolic Distillation: Geometry-Guided Cross-Modal Transfer for 3D Object Detection, arXiv:2605.09899, 2026.
- Yuan et al. RoboPoint: A VLM for Spatial Affordance Prediction for Robotics, arXiv:2406.10721, 2024. https://robo-point.github.io/
- PAVLM: Advancing Point Cloud based Affordance Understanding Via VLM, arXiv:2410.11564, 2024.
- O³Afford: One-Shot 3D Object-to-Object Affordance Grounding, arXiv:2509.06233, 2025.
- Bhatnagar et al. BEHAVE; Huang et al. InterCap; Xie et al. CHORE / ProciGen-HDM; Open4DHOI / 4DHOISolver (2024-2025).
- Survey: Spatial Reasoning in MLLMs (arXiv:2511.15722, 2025) — 提供 3D 空间推理 VLM 的完整分类。

---

## 7. 重要澄清：N3D-VLM 与 affordance grounding 的关系（2026-08-31 用户追问）

> 用户问："N3D-VLM 怎么应用的 affordance grounding？"
> **结论（已原文核验 arXiv:2512.16561）：N3D-VLM 没有做、也没有提到 affordance grounding。** 正文及附录全文检索不到 "affordance" 一词。

### 7.1 "grounding" 一词的两种含义（关键澄清）
- **N3D-VLM 的 grounding = 3D 物体定位（3D object grounding）**：用文本描述在 3D 空间中定位**物体的实例级边界框**——回答"物体在哪、长什么样（尺寸/朝向）"。
- **affordance grounding = 功能可供性定位**：预测物体上**可交互的区域**（接触区/抓取区/功能部位）——回答"物体哪里/如何能被使用"。通常是逐点/区域预测，强调功能语义而非位置。

二者是**不同层级**的任务，但可级联衔接：先 N3D-VLM 式地把物体定位到 3D（实例级），再在其上预测 affordance 区域（部件级）。

### 7.2 N3D-VLM 实际做了什么（核验）
- **任务**：① 3D Detection（自建 278 万样本库）；② 3D Grounding（RefCOCO/RefCOCO+/RefCOCOg、Objects365，指标投影 IoU/中心偏移/3D IoU）；③ 3D Spatial Reasoning QA（N3D-Bench 2000 题、SpatialRGPT-Bench 1406 题、CV-Bench-3D 1200 多选题）。
- **输出格式**：结构化 3D bbox `bbox(id, class, u, v, z, sx, sy, sz)`（相机坐标系，可转世界坐标）；**不是点/区域掩码**，故与 affordance 的逐点预测本质不同。

### 7.3 对你研究的真正可借鉴点（不是 affordance）
N3D-VLM 与你主线最相关的不是 affordance，而是两项可复用技术：
1. **2D→3D lift 数据飞轮**：用深度估计把大规模 2D 标注提升为 3D——**概念上等价于你的"2D HOI 图 → VGGT 重建 3D 点云"**（只是它用深度估计、你用 VGGT，且目标是 affordance 而非物体框）。
2. **depth-aware positional encoding + 结构化 3D 表示**：如何把度量 3D 几何直接注入 VLM token，使模型具备原生 3D 感知——这是你若要做"affordance VLM"时的表示层参考。

### 7.4 该去看哪类工作（避免方向错配）
若目标是 **affordance grounding**，应重点检索 §2-D 而非 N3D-VLM：
- **RoboPoint**：RGB + 语言 → 2D affordance 关键点（深度投影成 3D）。
- **PAVLM**：点云 + 语言 → affordance（几何引导传播）。
- **VAGNet / VideoAfford**：HOI 视频/MLLM → 3D affordance（与你的 VGGT+HOI 想法最近邻）。
- **O³Afford**：源/目标物体对点云 → object-to-object affordance。

> 一句话：**N3D-VLM 证明的是"原生 3D 定位能显著提升空间推理"，它不是 affordance 方法；但如果你想做"先 3D 物体定位、再预测 affordance"的 unified 模型，它的级联范式（grounding → reasoning）值得作为模板，把第二阶段换成 affordance 预测头。**

---

## 8. 设计蓝图：把 N3D-VLM 范式迁移到 affordance grounding（2026-08-31 用户追问）

> 用户问："N3D-VLM 怎么**应用到** affordance grounding？"
> 思路：它不能"直接"用（输出是 bbox），但它的三项资产——① 原生 3D 感知（depth-aware PE + 结构化 3D 表示）、② 级联推理（grounding → reasoning）、③ 2D→3D lift 数据飞轮——可被"改造复用"到 affordance。

### 8.1 三种复用模式

**模式一：级联式（最自然，最小改动）**
- Stage 1 = N3D-VLM 的 native 3D grounding：输入 RGB-D/多视图 → 输出**物体 3D bbox**（解决"哪个物体、在哪"）。
- Stage 2 = affordance head：在定位到的物体 3D 区域（crop 出点云/特征）上预测 **affordance 热图/区域**（解决"哪里可抓/可坐/可放"）。
- CoT 推理：先 grounding 再 affordance reasoning，例如"先把杯子定位到 3D，再判断其把手在左侧、可抓握"。
- 复用依据：N3D-VLM 已证明"先准确定位、再推理"显著提升可解释性与准确性；affordance 同样受益于先定位再预测（避免全局模糊预测）。

**模式二：表示层复用（depth-aware PE + 结构化 3D 表示）**
- 把 N3D-VLM 的 **depth-aware positional encoding** 与结构化 3D 坐标表示，作为 affordance 模型的几何注入方式，替换"只吃 2D 图 / 纯点云无尺度"的输入。
- 让 affordance 模型具备**度量 3D 感知**（尺度/朝向/深度），而非投影 2D。这是你若做 "affordance VLM" 时的表示层参考。

**模式三：数据飞轮复用（与你的 VGGT 想法最契合）**
- N3D-VLM 用深度估计把 **2D 检测标注 lift 成 3D** 训练数据（规模 ×6）。
- 类比：用 **VGGT（或深度）把 2D HOI 图像/视频里的 affordance 标注**（接触区/抓取区）**lift 成 3D affordance 点云标注** → 大规模 3D affordance 训练集。
- 这正是你"2D HOI → VGGT → 3D HOI 点云"想法的**数据侧实现**；区别在于 N3D-VLM lift 的是物体框，你 lift 的是 affordance 区域。

### 8.2 必须正视的改造 gap
1. **输出表示**：从 3D bbox（9 维：`id,class,u,v,z,sx,sy,sz`）改为 **affordance 热图/区域**（逐点 / 逐体素）——bbox 表示不能直接给 affordance。
2. **模态适配**：N3D-VLM 吃 RGB-D 图像；你的 affordance（尤其 HOI 点云）吃**点云**。需把 depth-aware PE 适配到点云 backbone（PointNet++ / Point Transformer），或直接用 VGGT 的 point map 作点云输入。
3. **监督信号**：affordance 需接触/抓取真值；来源 = 你的 HOI 重建（VAGNet/VideoAfford 式标注、BEHAVE 接触标签、Open4DHOI 的 135 类动作接触）。
4. **联合训练**：N3D-VLM 联合训 localization+reasoning；你联合训 grounding+affordance+（可选）蒸馏（参考 §4.3 管线）。

### 8.3 与你已有想法的对接（合成方案）
把三项复用串起来，得到你的方法雏形：
- **数据侧（模式三）**：2D HOI 图 → VGGT 重建 3D HOI 点云 → 用 HOI affordance 标注 lift 出 3D affordance 真值（teacher 监督源）。
- **teacher（模式一）**：在 HOI 点云上先 native 3D grounding 定位物体，再预测交互几何（接触热图/人体相对位姿）。
- **student（模式二）**：仅吃**物体点云** + depth-aware/结构化 3D 表示；用 SRKD 式关系/结构蒸馏（§4）逼近 teacher 的几何表征 → 推理时免人、免视频。
- **一句话总结**：N3D-VLM 给你的是**方法论脚手架**——原生 3D 感知 + 级联 + 数据飞轮；把第二阶段输出从 bbox 换成 affordance 区域、把输入从 RGB-D 换成 VGGT 点云，就得到你的方法骨架。

---

## 9. 几何保真设计：把 N3D-VLM 的"几何不丢失"哲学注入你的 pipeline（2026-08-31 用户原始想法）

> 用户原始想法（复述）：text + **3D HOI 点云**（2D HOI 图经前馈单目重建如 VGGT 生成）→ 3D MLLM → 生成**接触意图嵌入（contact intent embedding，以 hammer 等为示例）** → 与**物体点云**解码 → affordance/接触预测。
> 用户痛点（关键洞见）：HOI 点云压进 3D MLLM 时，经 point encoder/聚合器压成少量 token/embedding，**细粒度 3D 几何（接触区、相对位姿）丢失**；最终 intent embedding 变成语义/文本对齐向量，HOI 点云的几何没被真正利用 → "只是换了输入，没用到几何"。

### 9.1 N3D-VLM 恰好回答这个问题
N3D-VLM 的核心哲学 = **不让几何只活在压缩 embedding 里，而让 3D 几何成为显式、结构化的表示**，手段正是：
- **depth-aware positional encoding**：每个 token 携带度量 3D 坐标，几何不被池化抹平；
- **结构化 3D 输出（bbox）**：几何以显式、可解释形式存在，而非埋进语义向量；
- **先 grounding 再 reasoning 的级联**：显式 3D 定位先于抽象推理。

### 9.2 结合方案（4 个具体改造点）

**① 几何感知 tokenizer（治本）**
- 不用会把几何压没的 vanilla point encoder。采用 N3D-VLM 式 **depth-aware PE / 结构化 3D 表示**作为 HOI 点云 tokenizer——每个 token/region 携带度量 3D 坐标（类比 LLaVA-3D / N3D-VLM 把 3D 坐标作 positional embedding 注入）。
- 效果：MLLM 所见、intent embedding 所残留的都是真 3D 结构，而非被压平的语义。

**② 在 intent embedding 之前加"3D 接触 grounding"阶段（级联）**
- 仿 N3D-VLM 的 grounding→reasoning：先对 HOI 点云做**显式 3D 接触定位**（定位人手接触区、物体接触面，输出 3D 接触框/接触点集），再基于显式接触几何生成 intent embedding。
- 即"**grounding → intent**"而非"compress → intent"；几何在中间以结构化形式显式存在，不被压缩吞掉。

**③ 双流条件送入解码器（让几何真正被用上）**
- 解码物体点云时，**不要只用压缩的 intent embedding**，同时把**显式 3D 接触几何描述子**（已配准到物体 canonical 帧）一并送入解码器。
- decoder 输入 = `{contact intent embedding（语义）, explicit 3D contact geometry（几何）}` → 几何被真正用于解码，而非"只换输入"。

**④ 把"显式 3D 接触几何"作为蒸馏目标（呼应 §4 蒸馏想法）**
- teacher(HOI 点云) 输出**显式 3D 接触几何**；student(仅物体点云) 学预测它。
- 这把 §4 的"几何先验蒸馏"落到**显式几何目标**上——蒸馏的是结构化 3D 接触，而非模糊语义 embedding。N3D-VLM 的"显式几何输出"哲学在此成为蒸馏目标的设计原则。

### 9.3 架构草图（文字版）
```
2D HOI 图 --VGGT--> HOI 点云
                  |
                  v
   [几何感知 tokenizer: depth-aware PE]  --> per-point/region 3D tokens
                  |
                  v
   [3D MLLM + text] --级联-->  (a) 显式 3D 接触 grounding (接触框/点集)
                  |            (b) contact intent embedding (语义)
                  v
   decoder(物体点云) <<-- (a)+(b) 都送入 --> affordance/接触预测
   （teacher 的 (a) 显式几何 经 SRKD 式蒸馏 → student 仅物体点云也能出 (a)）
```

### 9.4 一句话总结
你担心的是"几何在 MLLM 压缩里丢了"；N3D-VLM 给的解法是"**让几何显式化、结构化、可定位**"——把它的 depth-aware PE + 先 grounding 后推理 + 结构化 3D 输出搬进 pipeline，HOI 点云的几何就既**保留（不被压没）**又**被解码器真正用上**。这同时修掉了"只换输入、没用几何"的隐患，并把 §4 的蒸馏想法锚定到显式几何目标。

---

## 10. 可行性量化：depth-aware PE tokenizer 的 token 量 & 3D MLLM 承接能力（2026-09-01 用户追问）

> 用户追问：把 depth-aware PE 作为 HOI 点云 tokenizer，token 量是多少？3D MLLM 能否承接住？

### 10.1 先澄清一个误解：depth-aware PE ≠ 一 token 一点
depth-aware PE 是**加在已聚合 token 上的位置编码**，不是"每个点一个 token"。原始 HOI 点云（VGGT point map 是逐像素的，单视图即 ~20 万–50 万点）**绝不可能**逐点喂给 MLLM；token 量由**聚合/抽象策略**决定。所以"token 量"是你可以设计的可调参数，不是点云规模。

### 10.2 现有 3D MLLM 的实际 token 预算（实证，来自论文）
| 方法 | 表示方式 | 视觉 token 数 | 说明 |
|---|---|---|---|
| **3D-LLM** | Q-Former Query | **32** | 极度压缩，几何损失最大（印证用户担忧） |
| **PQ3D** | Query | 80 | object-centric |
| **LEO / 3D-VisTA** | Object | 60 / 80 | 离线实例分割特征 |
| **3D-LLaVA** | Query | 100 | |
| **Chat-Scene / Inst3D-LMM** | Object | 200 | object-centric 上限附近 |
| **PointLLM** | Point-BERT 编码后投影 | **513** | 8192 输入点 → 编码器输出 513 特征 → 投影为 token（dim 5120），单物体点云实测 |
| **LLaVA-3D** | Voxel（多视图） | **3096** | 整场景体素，0.2m 分辨率，推理 0.2s |
| **Video-3D LLM / GPT4Scene** | Video | 6720 / 8262 | 视频方法，最重 |

**规律**：object-centric（单物体/HOI 物体，正是你的场景）普遍落在 **32–200 token**；单物体富表示如 PointLLM 用 **~513**；整场景体素/视频才会到 3000–8000。

### 10.3 你的 HOI 点云该用多少 token？
- **推荐预算：~200–512 token**（对齐 PointLLM 513 / Chat-Scene 200 这一档）。这是 object-centric 的"富表示但可控"区间。
- 实现路径：FPS 采样 8192 点 → PointNet++/Point Transformer set abstraction 到 ~256–512 seed token → 每个 seed 附 **depth-aware PE**（来自 seed 的 3D 坐标 / 局部帧）→ 投影进 LLM。即 "§9.① 几何感知 tokenizer" 的落地形态。

### 10.4 3D MLLM 能否承接住？——能，但分两层回答
- **算力/上下文层：完全能。** LLM 主干支持 2k–128k context；视觉 token 在数百～低千级是标准区间。PointLLM 513、LLaVA-3D 3096、甚至 GPT4Scene 8262 都实际跑通；自注意力 O(n²) 在 ~500 token 时开销极小，3000 也仅 0.2s 级推理。
- **几何保真层：低 token 预算下仍会丢几何（这正是你原担忧）。** 实证佐证：3D-LLM 压到 32 token 几何损失最大；LLaVA-3D 论文也指出"体素池化引入信息损失"。**depth-aware PE 缓解但不消除**低 token 下的几何丢失——它保证每个 token 带坐标，但 token 太少时接触区细节仍被平均掉。

### 10.5 因此：用 §9 的几何策略在低 token 预算下"省着用几何"
在 ~200–512 token 预算内保住接触几何，靠的不是更多 token，而是**token 怎么放、几何还在不在 token 之外**：
1. **接触感知 token 放置**：FPS 时对接触区/人手-物体邻近点**过采样/加权**，让有限 token 集中在 task-critical 几何上（而非均匀下采样把接触区冲淡）。
2. **depth-aware PE 必加**：LLaVA-3D 实证——3D PE 在"需要真 3D 空间理解"的任务上增益巨大（MMScan QA +13.3、Scan2Cap +54.4），对你的 affordance（几何关键）是必需而非可选。
3. **几何在 token 之外仍显式存在**：保留 §9.② 的"3D 接触 grounding 显式输出" + §9.③ 解码器双流条件——即使 token 压到 200，接触几何也以结构化形式（接触框/点集）直接进解码器，不依赖被压扁的 embedding。
4. **可选层次化**：少量粗 token（全局）+ 少量细 token（接触区邻域，类 PointAttN 注意力分区），用有限预算换局部保真。

### 10.6 结论
- token 量：**可设计为 ~200–512（object-centric 推荐档）**，由聚合策略控制，非点云规模。
- MLLM 承接：**算力上完全没问题**（数百～低千级是标准区间）。
- 真正要解的仍是"低 token 下几何保真"——靠 §9 的接触感知放置 + depth-aware PE + 显式 grounding 输出 + 蒸馏，而非堆 token 数。与 N3D-VLM "几何显式化"哲学一致。

---

## 11. QueryMe vs SRKD：区别 & QueryMe 与你想法的关键对比（2026-09-01 重要近邻更新）

> 用户问："SRKD 进行点云间几何蒸馏和 QueryMe 任务有什么区别？"
> **⚠️ 重要前提**：此前的调研（§2-D）**漏掉了 QueryMe**——它是 CVPR 2026 的工作，且与你想法高度重合，现补入并重新定位。

### 11.1 QueryMe 是什么（原文核验，CVPR Open Access PDF 前 3 页）
- **出处**：Zhao, Li, Liu, Zhao, Liu, Zhang. *QueryMe: Query-Driven Open-Vocabulary 3D Object Affordances Grounding from Multimodal Evidence*, **CVPR 2026**. 哈尔滨工业大学（HIT）。
- **任务**：**开放词汇 3D 物体可供性接地**——给定任意语义描述，在 3D 物体点云上定位**功能区域**。
- **输入/输出**：3D 物体点云 $P \in \mathbb{R}^{N\times3}$ + **单张 HOI RGB 图** $I \in \mathbb{R}^{3\times H\times W}$ + 自然语言指令 → 点云上的 affordance 区域（**AUC** 评测）。
- **方法（原文显式）**：
  1. 用 **feed-forward 3D reconstruction pipeline**（前馈单目重建）把 HOI 图映射到 3D → **3D HOI space**；
  2. **Adaptive Spatial Attention Module**：选择性抑制无关像素，缓解**重建误差与背景杂波**敏感性；
  3. 在重建点云上做 **random spatial sampling**，构造一组**可学习的 affordance query vectors**（数量为 compact set，K 未在已获取页给出）；
  4. **Multimodal Guided Query Learning Module**：在 **3D HOI space → text space → 3D object space** 三域**顺序检索** affordance 线索，检索"**几何一致的功能部件**"，融合视觉/语言/几何线索；
  5. 轻量 **query decoder** 定位 affordance 区域。
- **卖点**：用**几何相似度做类比推理**（analogy reasoning through geometric similarity）提升 unseen scene/category 泛化；**AUC 较先前工作 +4.19%**。
- **原文未提供（待补）**：3.2–3.3 公式细节、几何相似度的具体度量、实验数据集全名、QueryMe 自身 Limitations（PDF 第 4–10 页未获取）。

### 11.2 SRKD vs QueryMe：层次不同，严格说不可直接比较
| 维度 | **SRKD**（点云间几何蒸馏） | **QueryMe**（开放词汇 3D affordance grounding） |
|---|---|---|
| **本质** | **训练技术 / 优化手段**（how to train），非任务定义 | **任务 + 框架**（what to do + 一种做法） |
| **目标** | 让轻量 student 逼近强 teacher（模型压缩/提效） | 开放词汇下在物体点云上定位功能区域，泛化到未见类别/场景 |
| **迁移什么** | **网络→网络**；teacher/student 看**同构点云、同任务（3D 分割）** | **模态/证据→任务**；2D HOI 图 + 文本 → 3D 点云 affordance |
| **几何的角色** | 几何 = **被蒸馏的对象**（结构关系 affinity matrix 对齐、跨样本几何结构对齐） | 几何 = **检索的键 / 类比的依据**（用几何相似度检索"几何一致"的功能部件） |
| **机制** | 显式 teacher–student 对 + 蒸馏损失对齐特征/关系 | **无 teacher/student**；learnable query vectors + 注意力三域顺序检索（DETR 式查询解码） |
| **开放词汇** | 不涉及（封闭集分割上的压缩） | **核心卖点**（unseen scene / unseen category，AUC +4.19%） |
| **输入→输出** | 点云 → 逐点分割标签（student 逼近 teacher） | 物体点云 + HOI 图 + 文本 → affordance 区域（AUC） |

**一句话**：SRKD 是"**怎么把大模型能力压进小模型**"（训练范式）；QueryMe 是"**怎么用 HOI+文本在 3D 物体上找功能区域**"（任务与方法）。在你的想法里，SRKD 是被借鉴的**手段**，QueryMe 是需要正面对比的**同类工作**。

### 11.3 ⚠️ 关键提醒：QueryMe 是你目前最强近邻（比 VAGNet/VideoAfford 更近）
与你想法（text + 3D HOI 点云 → 3D MLLM → contact intent embedding → 物体点云解码）重合度极高：
- ✅ **都用前馈单目重建把 2D HOI 抬到 3D**（你说的 VGGT 正是此类 feed-forward 3D reconstruction pipeline）；
- ✅ **都做多模态融合**（HOI + 文本 + 3D 物体几何）；
- ✅ **都用查询/嵌入在物体点云上检索 affordance**（你的 intent embedding ↔ 它的 affordance query vectors）；
- ✅ **都强调几何一致性 / 几何相似度**。

**你的想法与 QueryMe 的差异（= 潜在 novelty，推演）**：
1. **推理时的输入依赖（最强差异点）**：QueryMe **推理时仍需输入 HOI 图像**（HOI RGB 图是它的显式输入）；而你的想法若走 §4 / §9.④ 的 **teacher(HOI 点云) → student(仅物体点云) 蒸馏**，则**推理时只需物体点云，免 HOI 图、免人、免视频**。
2. **推理骨干不同**：QueryMe 用 **learnable query vectors + 注意力解码器**（DETR 式）；你要用 **3D MLLM 生成 contact intent embedding**（LLM 式语义推理 + 开放世界知识）。
3. **显式 3D 接触几何输出**：QueryMe 只输出 affordance 区域；你的 §9.② 借鉴 N3D-VLM 会额外输出**结构化 3D 接触 grounding**，并以 §9.③ 双流条件直接进解码器。
4. **几何蒸馏范式**：QueryMe 无 teacher–student 蒸馏；你的 §4/§9.④ 是**跨组合点云（HOI）→ 单体点云（物体）**的几何先验蒸馏。
5. **几何保真设计**：你针对"HOI 点云压进 MLLM 会丢几何"专门做了 §9.① 几何感知 tokenizer + §10 的 contact-aware token 放置；QueryMe 侧重点是"抑制重建误差与背景杂波"（Adaptive Spatial Attention），未处理 MLLM 压缩的几何损失。

### 11.4 建议（推演）
- **必须精读 QueryMe 全文**（尤其缺失的 3.2–3.3 公式与实验章节），并把它作为**首要 baseline**。
- 差异化叙事应主打四点组合：**"推理时免 HOI 图像（蒸馏后只需物体点云）+ 3D MLLM 语义推理 + 显式 3D 接触几何 + 几何保真 tokenizer"**；而**不要**把"用 HOI 做 3D affordance"本身当卖点（这一点 QueryMe 已做）。
- 更新 §2-D 的最近邻列表：加入 **QueryMe（CVPR 2026）**，且它比 VAGNet/VideoAfford 更近。

---

## 12. SRKD 的点云间几何蒸馏能否借鉴？（2026-09-01 精读原文后分析）

> 用户问："SRKD 的点云间几何蒸馏可以借鉴吗"
> 依据：SRKD 原文 PDF（`D:/article/2026-9-1SRKD.pdf`，13 页，arXiv:2506.17290v1）+ QueryMe 本地精读笔记。
> **结论：可以借鉴，而且是目前最适合你 §9.④ 蒸馏环节的现成工具箱；但必须做一次"同输入 → 异输入"的范式改造。**

### 12.0 先补正 §11：QueryMe 本地笔记已补全此前缺失信息
| 项 | §11.1（CVPR PDF 前 3 页） | 本地精读笔记（已确认） |
|---|---|---|
| 3D 重建器 | 未命名，"feed-forward 3D reconstruction" | **明确用 VGGT**（与你计划一致） |
| 数据集 | 未获取 | **PIADv2**（3DIR + 3D-AffordanceNet + Objaverse），43 物体类 / 24 affordance 类 |
| 骨干 / 点数 | 未获取 | **PointNet++**，2048 点；2× RoBERTa 编码 Interaction/Geometric Attributes |
| 查询位置编码 | 未获取 | **MLP 位置编码**（作者实测 3D RoPE 增益有限） |
| 损失 | 未获取 | **Focal + Dice** |
| 结果 | AUC +4.19% | Seen 92.34 / Unseen-Obj **83.03**（GREAT 79.57）/ Unseen-Aff **74.00** |
| 最强消融 | 未获取 | **✗3DHOI（换成 2D ResNet18 特征）→ Unseen-Aff AUC 60.50（−13.5）**，证明 3D 化 HOI 是关键 |

### 12.1 SRKD 精确机制（原文核验）
- **出处**：Li, Dong, Dong, Yang, An, Xu. *SRKD: Towards Efficient 3D Point Cloud Segmentation via Structure- and Relation-aware Knowledge Distillation*, arXiv:2506.17290v1 (2025-06)。中科院计算所 + 南洋理工。代码 `github.com/itsnotacie/SRKD`。
- **任务**：3D 点云**语义分割**（室内 ScanNet、室外 SemanticKITTI）。
- **teacher / student**：teacher = **CDSegNet 101.4M**（PTv3 骨干 + DDPM 去噪分支），**冻结**；student = 通道减半的 PTv3，**11.6M**（<1/8）。
- **⚠️ 关键前提：teacher 与 student 看同一个点云输入**（同输入、同任务）→ 本质是**模型压缩**，不是跨输入迁移。
- **四类损失**：
  1. `L_task`：与 GT 的 CE；
  2. `L_kd`：逐点软标签 KL（T=2）；
  3. **CSMBGD（跨样本 mini-batch 几何蒸馏）**：对 mini-batch 内第 i、j 个点云，L2 归一化点特征后算跨样本相似矩阵 $M_{ij}=F_iF_j^\top$；用**行级 softmax(T) + KL** 对齐 student 的 $M^s_{ij}$ 与 teacher 的 $M^t_{ij}$；
  4. **AMBRA（亲和矩阵关系对齐）**：单样本内算点间亲和 $D(i,j,w_i)=w_i\|F_i-F_j\|_2$、体素间亲和，student 用 **MSE 逼近** teacher 的亲和矩阵；另有**通道级 KL**。
     - 类别感知 supervoxel 采样：$w_i=\tau_{class}/N_v\cdot D_i/R$，$\tau_{class}=1-C_{current}/C_{total}$（稀有类权重更高）；
     - 体素划分 $R_v\times A_v\times H_v$ 选 K 个 super-voxel；点数/体素数按阈值**截断或补零固定**以应对变长。
- `L_total = L_task + λ_kd L_kd + λ_p L_p_amra + λ_v L_v_amra + λ_c L_c_amra + λ_batch-GD L_batch-GD`；权重 λ_kd=0.3、λ_p=λ_v=0.001、λ_c=1000、λ_batch-GD=0.1，T=2。
- **结果**：ScanNet Baseline 76.7 → **Ours 77.9 mIoU @11.6M**，追平 teacher CDSegNet 77.9@101.4M（1/8 参数）。
- **消融（Tab 5）**：Baseline 70.8 → +L_kd 72.3 → **+CSMBGD 74.3（+2.0，单组件最大增益）** → +AMBRA 75.0（+0.7）。
- **batch size 消融**：batch 2→12，mIoU 77.0→78.0（跨样本对比依赖大 batch）。

### 12.2 与你的场景的本质差异（最容易踩的坑）
| | **SRKD** | **你的想法** |
|---|---|---|
| teacher/student 输入 | **同一个点云**（同输入） | teacher 看 **HOI 点云**，student 只看**物体点云**（**异输入**） |
| 目的 | 模型压缩（小模型追大模型） | **跨输入的几何先验迁移**（让无 HOI 的模型学会 HOI 里的交互几何） |
| teacher 是否信息更多 | 否（只是更大，输入相同） | **是**（teacher 严格多出人手 / 接触上下文） |
| 任务形式 | 封闭集分割（逐点 C 类 softmax + CE） | 开放词汇 affordance（逐点 0/1 概率 + Focal/Dice） |

→ **不能照搬 SRKD 的设定，但可以借鉴它的损失构造思想。** 这是本节最重要的判断。

### 12.3 可借鉴的四个组件（按可迁移价值排序）

**① CSMBGD 跨样本几何相似矩阵 —— 最有价值，且最贴合你的"类比推理"目标**
- SRKD 核心洞察：单样本内蒸馏抓不到"跨样本的通用几何结构"；mini-batch 内做**样本间**相似矩阵对齐能让 student 学到更稳定的几何语义。消融证明它是**单组件增益最大**（+2.0 mIoU）。
- **对你的改造**：把"object_i ↔ object_j"换成 **"HOI 样本 ↔ 物体样本"** 的跨输入相似矩阵对齐——把 teacher 在 HOI 上下文中看到的几何关系结构，蒸馏给只看到物体的 student。这比 SRKD 的同类对比更强，因为它本身就是"几何先验迁移"。
- **另一用法**：蒸馏**跨实例的 affordance 类比结构**——"在 teacher 的 HOI 空间里几何相似的物体，在 student 的物体空间里也应几何相似"。这与 QueryMe"用几何相似度做类比推理"的卖点直接呼应，而你把它变成**可蒸馏的监督信号**（QueryMe 只是推理时用，没有蒸馏）。
- 注意：SRKD 的 $M_{ij}$ 要求两侧特征逐点可比，你需先用共享锚点/规范化解决（见 12.4）。

**② AMBRA 亲和矩阵关系对齐 —— 直击你"几何在压缩中丢失"的痛点**
- SRKD 用点间/体素间的**成对特征距离矩阵**作蒸馏目标，而非只对齐逐点输出，迫使 student 复现 teacher 的**结构关系**。
- 这正对症：你担心"HOI 点云压进 MLLM 后几何被压没"，那蒸馏目标就不该只有 affordance 热图（response KD），必须加**关系/结构级对齐**（relation KD）。
- **改造**：把 $w_i$ 的"类别感知权重"换成**"接触感知权重"**——对接触区 / 人手邻近区域给更高采样权重，让关系对齐集中在 task-critical 几何上。这与 §10.5 的"接触感知 token 放置"是同一思想在蒸馏侧的落地。

**③ L_kd 软标签 + GT 混合监督 —— 直接可用，风险最低**
- teacher 的 affordance 热图作软标签 → KL 到 student；同时用 GT 的 Focal+Dice（QueryMe 用法）监督。SRKD 已证明"soft + hard 协同"有效，可直接迁移。

**④ 重要性感知采样 + 点数固定技巧 —— 工程细节可直接抄**
- SRKD 用 $w_i$ 做稀有类上采样、用截断/补零固定点数与体素数以应对变长点云。HOI 与物体点云点数天然不同，这个"固定到阈值"的技巧正是你需要的。

### 12.4 必须解决的三个适配难题（SRKD 没替你解决）
1. **对应关系缺失（最大难题）**：SRKD 两侧特征在同一批点上、逐点可比；你的 HOI 与物体点云是**不同点集**，无法直接算逐点亲和矩阵差。可选解：
   - (a) 两者都**归一化到物体 canonical 帧**，再在物体上取**共享锚点**（FPS 采 K 个），在锚点邻域上算亲和 → AMBRA 可直接用；
   - (b) 复用已调研的**配准 / 对应**（GeoTransformer、functional maps，见 `点云几何相似度迁移学习相关工作.md`）；
   - (c) 退到**粗粒度**（supervoxel / 部件级 token）再算亲和，降低对应精度要求。
2. **信息不对称**：SRKD 的 student 架构上有能力复现 teacher；你的 teacher 掌握 student 看不到的信息（人手/接触），student 可能**无法完全匹配**。建议只在**物体可见区域 / 共享锚点**上对齐，并把接触几何作为**显式预测目标**（§9.④）而非隐式特征。
3. **损失形式**：affordance 是逐点 0/1 概率 + 开放词汇，不能套 CE over C classes；用 **Focal+Dice**（QueryMe）作 `L_task`，`L_kd` 用二值 KL。

### 12.5 落地草案：把 SRKD 嵌进你的 teacher→student 管线
```
Teacher 分支（仅训练时）: HOI 点云 H + 文本 T ──> affordance 热图 ω^T + 中间几何特征 F^T
Student 分支（部署时）  : 仅物体点云 P（+ 文本 T）──> ω^S, F^S

L_task  = Focal(ω^S, y) + Dice(ω^S, y)                 # GT 监督（QueryMe 式）
L_kd    = KL( σ(ω^S/T) || σ(ω^T/T) )                   # 输出级软标签（SRKD 式）
L_amra  = Σ_anchor || D^S(i,j,w) − D^T(i,j,w) ||²      # 关系/亲和对齐（SRKD AMBRA，接触感知加权）
L_csgd  = KL_row( σ(M^S_{H,P}/T) || σ(M^T_{H,P}/T) )   # 跨输入几何相似矩阵（SRKD CSMBGD 的异输入版）
L_total = L_task + λ_kd L_kd + λ_amra L_amra + λ_csgd L_csgd
```
- **锚点**：HOI 中的物体点 ↔ 物体点云，统一到 canonical 帧后 FPS 取 K 个共享锚点。
- **权重初值**可参照 SRKD（λ_kd≈0.3、λ_amra≈0.001 量级、λ_csgd≈0.1），但需按你的损失量级重新平衡。
- **batch**：跨输入对比依赖大 batch（SRKD 消融 batch 2→12 增益明显），训练资源需预留。

### 12.6 结论
- **可以借鉴**，而且它是目前最适合你 §9.④ 蒸馏环节的现成工具箱；它给的不是"能不能做"的答案，而是"**怎么设计蒸馏损失**"。
- **但必须做范式改造**：SRKD = 同输入压缩；你 = 异输入（HOI→物体）几何先验迁移。改造关键是**共享锚点/规范化**（让亲和矩阵可比）+ **接触感知加权**（让有限监督集中在接触区）。
- **最值得抄的是 CSMBGD**（消融单组件增益最大 +2.0），因为"跨样本的几何结构"正是你要迁移的东西；**AMBRA 直接对症"几何在压缩中丢失"**。
- **与 QueryMe / GEAL 的组合**：QueryMe 提供 teacher 侧强架构（VGGT + 三域查询，PIADv2 Unseen-Aff AUC 74.00）但**没有蒸馏**；你把它（或 GEAL）当 teacher，用 SRKD 的损失把 HOI 几何先验压进"只看物体点云"的 student，就得到"**推理免 HOI 图**"这一最强差异点。

---

## 13. 集成设计草案：SRKD 如何进入你的 pipeline（2026-09-01）

> 目标：text + 3D HOI 点云（VGGT 前馈单目重建）→ 3D MLLM → **contact intent embedding** → 与物体点云解码 → affordance；
> 且必须解决"**HOI 点云压进 MLLM 后几何丢失 → 只换输入、没用几何**"。
> 本节把 §9（几何保真）+ §10（token 预算）+ §12（SRKD）收敛为一个可执行的 teacher→student 方案。

### 13.1 设计目标（三个必须同时解决的问题）
1. **G1 免 HOI 推理**：部署时只给物体点云 + 文本，不要 HOI 图、不要 VGGT、不要 MLLM（相对 QueryMe 的核心差异）。
2. **G2 几何不丢失**：HOI 的接触区/相对位姿必须在压缩后仍被保留，并被解码器**显式利用**（而非只换输入）。
3. **G3 跨实例泛化**：能对未见物体/未见 affordance 做类比推理（PIADv2 的 Unseen 划分）。

### 13.2 整体架构
```
【训练时】
2D HOI 图 I ──VGGT──> HOI 点云 H ──(SAM2/SMPL 分离)──> 物体子云 H_obj
                                                          │
                          文本 T ──┐                       │
                                   ▼                       ▼
              ┌──── 几何感知 tokenizer: depth-aware PE + 接触感知 token 放置 (§9.①/§10.5) ────┐
              │                        → 256–512 个 3D token                                  │
              ▼                                                                               │
        3D MLLM ──> ① 接触意图嵌入 z^T                                                        │
                    ② 显式 3D 接触 grounding G^T（接触框/点集，N3D-VLM 式，§9.②）              │
                    ③ 中间几何特征 F^T（共享锚点上）                                           │
                                   │                                                          │
                                   ▼                                                          │
        Decoder(物体点云 P | z^T, G^T) ──> affordance 热图 ω^T   ← 双流条件（§9.③）            │
                                                                                              │
【Student（部署时的全部）】                                                                    │
物体点云 P + 文本 T ──> 轻量 3D 编码器(PointNet++/PTv3-lite) ──> z^S, G^S, F^S ──> ω^S        │
                                                                                              │
                    ▲────────── SRKD 式蒸馏损失把 T 的知识压进 S ───────────┘
【推理时】仅跑 Student：P + T → z^S, G^S → ω^S（免 HOI 图 / 免 VGGT / 免 MLLM）
```

### 13.3 SRKD 的三处注入点 —— 对应"几何丢失"的三个堵点
这是本草案的核心：**把"保住几何"从一种期望，变成可优化的训练目标**。

| 堵点 | 症状 | 对策（§9/§10） | **SRKD 注入** |
|---|---|---|---|
| **① tokenizer 压缩时丢** | 稠密 HOI 点云 → 少量 token，接触细节被平均掉 | 几何感知 tokenizer（depth-aware PE）+ **接触感知 token 放置** | **`L_amra`**：强制 student 的特征**亲和矩阵**匹配 teacher 的 HOI-informed 亲和矩阵 → 想匹配成对关系就必须留住几何 |
| **② intent embedding 是瓶颈** | z 只是语义向量，几何进不去也出不来 | **显式 3D 接触 grounding 输出**（几何在 embedding 之外显式存在，§9.②） | **`L_ground`**：teacher 的显式接触几何 G^T 作蒸馏目标（§9.④）；**`L_intent`**：z^S → z^T 对齐 |
| **③ 跨实例不会类比** | 未见物体/affordance 上崩 | 几何相似度类比（QueryMe 卖点） | **`L_csgd`**：跨样本/跨输入几何相似矩阵对齐（CSMBGD 的异输入版），把"类比推理"变成**可蒸馏的监督** |

> **一句话**：SRKD 在这里的角色不是"压缩模型"，而是**"给几何保真提供可微的训练目标"**——AMBRA 让 student 被迫复现 teacher 的几何关系结构，CSMBGD 让跨实例的几何类比结构可迁移。

### 13.4 共享锚点机制（`L_amra` 的关键实现）
HOI 与物体点云是**不同点集**，无法逐点比亲和矩阵。解法：
1. HOI 中的物体子云 `H_obj` 与干净物体点云 `P` 都**归一化到物体 canonical 帧**（去尺度、去位姿）；
2. 在 `P` 上 **FPS 采样 K 个锚点**（K=64–256），并把 `H_obj` 的对应点映射到同一坐标系；
3. 在锚点上取 teacher/student 的点级特征，算亲和 $D(i,j)=w_{ij}\|F_i-F_j\|_2$；
4. **接触感知权重 $w_{ij}$**：对靠近接触区/人手邻近的锚点给更高权重（可用 teacher 预测的接触分数，或 HOI 中人手距离）→ 把有限监督集中在 task-critical 几何上（与 §10.5 同源思想）。

### 13.5 完整损失函数
```
L_task   = Focal(ω^S, y) + Dice(ω^S, y)                    # GT affordance（QueryMe 式）
L_kd     = KL( σ(ω^S/T) ‖ σ(ω^T/T) )                       # 输出级软标签（SRKD L_kd）
L_intent = 1 − cos(z^S, z^T)   [或 MSE / InfoNCE]           # 接触意图嵌入对齐
L_ground = SmoothL1(G^S, G^T) + Chamfer(接触点集)            # 显式 3D 接触几何蒸馏（§9.④）
L_amra   = Σ_{i,j∈anchors} w_ij · ‖D^S(i,j) − D^T(i,j)‖²   # 关系/亲和对齐（SRKD AMBRA，接触感知加权）
L_csgd   = KL_row( σ(M^S_{H,P}/T) ‖ σ(M^T_{H,P}/T) )       # 跨输入几何相似矩阵（SRKD CSMBGD 异输入版）

L_total  = L_task + λ_kd L_kd + λ_intent L_intent + λ_ground L_ground + λ_amra L_amra + λ_csgd L_csgd
```
- **hard + soft 协同**（SRKD 已证明有效）：`L_task` 用 GT，`L_kd` 用 teacher 软标签。
- 权重初值见 §13.6；按 SRKD 经验"先归一化到同一量级再调"。

### 13.6 超参与形状参考表
| 项 | 参考值 | 依据 |
|---|---|---|
| 物体点云点数 N_P | 2048（或 8192） | QueryMe 用 2048 |
| HOI 点云点数 N_H | 8192–16384 | VGGT 输出后下采样 |
| MLLM 视觉 token 数 | **256–512** | §10（object-centric 推荐档） |
| 共享锚点 K | 64–256 | 设计选择，先取 128 |
| intent embedding 维数 d | 256 | 设计选择 |
| 文本编码 | RoBERTa×2（Interaction/Geometric） | QueryMe 验证有效 |
| KL 温度 T | 2 | SRKD |
| λ_kd | 0.3 | SRKD |
| λ_amra | 0.001 量级 | SRKD（λ_p=λ_v=0.001） |
| λ_csgd | 0.1 | SRKD（λ_batch-GD） |
| λ_intent / λ_ground | 从 1.0 / 1.0 起调 | 无先例，需按损失量级标定 |
| batch size | **≥8，建议 12–16** | SRKD 消融：batch 2→12 时 mIoU 77.0→78.0 |
| 优化器 | AdamW，cosine（OneCycleLR） | SRKD |

### 13.7 分阶段训练 & 推理流程
- **Stage 0（可选）**：用 Point-MAE/Point-BERT 式自监督预训练几何 tokenizer（depth-aware PE）。
- **Stage 1：训 Teacher**（HOI + text → z^T, G^T, ω^T）。监督 = GT affordance（Focal+Dice）+ GT 接触几何（BEHAVE / PIADv2 接触标注）。
- **Stage 2：冻结 Teacher，训 Student**（仅 P + text）。损失 = §13.5 的 `L_total`。
- **Stage 3（可选）**：解冻 decoder 微调。
- **推理**：**只跑 Student** —— P + text → z^S, G^S → ω^S。**免 HOI 图、免 VGGT、免 MLLM**。

### 13.8 消融验证计划（逐条证明每个组件有效）
| 实验 | 去掉/改变 | 验证什么 |
|---|---|---|
| A1 | w/o `L_amra` | **核心**：关系对齐是否真的保住几何（直接回应你的担忧） |
| A2 | w/o `L_csgd` | 跨输入几何结构对齐的贡献（SRKD 中增益最大的组件） |
| A3 | w/o `L_ground`（只留 intent embedding） | §9.④ "以显式几何为蒸馏目标"是否必要 |
| A4 | w/o 显式 grounding 输出（§9.②） | "几何在 embedding 之外显式存在"是否必要 |
| A5 | w/o depth-aware PE / w/o 接触感知 token 放置 | §9.① + §10.5 的贡献 |
| A6 | token 数 256 / 512 / 1024 扫描 | 验证 §10 的预算结论 |
| A7 | 与 QueryMe 对照 | **推理免 HOI 图 vs QueryMe 需 HOI 图**（最强差异点的量化） |
> 评测用 **PIADv2 三划分**（Seen / Unseen Object / Unseen Affordance），指标 AUC / aIoU / SIM / MAE，与 QueryMe 数字对齐可比。

### 13.9 风险与备选方案
| 风险 | 备选 |
|---|---|
| 锚点对应不准（H_obj ↔ P） | 用配准（GeoTransformer / functional maps，见 `点云几何相似度迁移学习相关工作.md`）；或退到粗粒度 supervoxel |
| Student 容量不足，匹配不了 teacher | 只在**共享锚点/可见区**对齐；放宽为"关系分布匹配"而非逐点严格一致 |
| 3D MLLM 训练成本过高 | 直接用**冻结的现成 3D MLLM**（PointLLM / LLaVA-3D）作 teacher 起点 |
| 接触真值稀缺 | VGGT + 人手检测自动生成伪标签；或用 BEHAVE / PIADv2 的接触标注 |
| 六个损失权重难平衡 | 先各自归一化到同一量级（SRKD 经验），再网格搜索 λ |
| VGGT 尺度模糊污染几何 | 多视图/短视频多帧输入 + canonical 归一化（§4.4 已列） |

### 13.10 Novelty Statement（可直接改写成 Introduction 的贡献段）
> 现有 HOI→3D affordance 方法（如 QueryMe）虽已把交互图像前馈重建到 3D 并做多模态查询，但**推理时仍依赖 HOI 图像**，且未处理"3D 几何经 MLLM 压缩后丢失"的问题。本文提出：以 **3D MLLM + HOI 点云**为 teacher，用 **SRKD 式结构-关系感知蒸馏**将其交互几何先验压入**仅看物体点云**的 student，并引入两点针对性设计——(i) **几何感知 tokenizer**（depth-aware 位置编码 + 接触感知 token 放置）以减缓源头压缩损失；(ii) 以 **显式 3D 接触 grounding** 作为蒸馏目标与解码器的第二路条件，使接触几何在压缩 embedding 之外仍显式存在并被解码利用。由此，模型在**推理时免 HOI 图像、免 MLLM**，同时保住并利用几何信息。

### 13.11 与既有工作的定位（一句话）
| 工作 | 它做了 | 你加了什么 |
|---|---|---|
| QueryMe (CVPR26) | VGGT 3D 化 HOI + 三域查询 affordance | **蒸馏**（免 HOI 推理）+ **显式接触几何** + **几何保真 tokenizer** |
| SRKD (arXiv 2506.17290) | 同输入点云分割的模型压缩（CSMBGD + AMBRA） | **异输入（HOI→物体）几何先验迁移** + 接触感知加权 + 显式几何目标 |
| N3D-VLM | 3D 物体 grounding + 空间推理（非 affordance） | 借其**几何显式化哲学**（depth-aware PE / grounding→intent / 结构化 3D 输出） |
| GEAL (CVPR25, 你的 baseline) | 3D 高斯桥 + DINOv2 一致性蒸馏 | 加 **MLLM 意图教师** + **生成式完整几何** + **结构-关系级蒸馏** |
