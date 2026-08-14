---
tags:
  - papers/3d-affordance
  - papers/3d-vision
  - topic/open-vocabulary-3d
aliases:
  - "T-FunS3D"
date: 2026-06-04
doi: 10.48550/arxiv.2606.05975
---

# T-FunS3D: Task-Driven Hierarchical Open-Vocabulary 3D Functionality Segmentation

## 核心信息

- 标题: T-FunS3D: Task-Driven Hierarchical Open-Vocabulary 3D Functionality Segmentation
- 标题翻译: 任务驱动的层级式开放词汇 3D 功能分割
- 作者: Jingkun Feng, Reza Sabzevari
- 机构: P4MARS Lab, Faculty of Aerospace Engineering, Delft University of Technology
- 发表时间: 2026-06-04（ICRA 2026 收录，arXiv 预印）
- 发表渠道: IEEE ICRA 2026 / arXiv
- DOI: 10.48550/arxiv.2606.05975
- arXiv: 2606.05975
- 论文链接: https://arxiv.org/abs/2606.05975
- 代码 / 项目: [EdwardjkFeng/T-FunS3D: Official code release for T-FunS3D](https://github.com/EdwardjkFeng/T-FunS3D)
- 数据 / 资源: SceneFun3D
- 论文类型: AI_method（training-free 的开放词汇 3D 功能/affordance 分割 pipeline）

## 原文摘要翻译

开放词汇 3D 功能分割使机器人能够在 3D 场景中定位物体的功能性部件。这是一项需要空间理解与任务解读的难题。当前开放词汇 3D 分割方法主要聚焦于物体级识别，而全场景部件分割方法试图穷举式地分割整个场景，导致极高的资源消耗与时间开销。如何在粒度、精度与速度之间平衡分割性能，仍是一个挑战。作为缓解该问题的一步，我们提出 `T-FunS3D`——一种任务驱动的层级式开放词汇 3D 功能分割方法，为机器人应用提供可操作的感知能力。我们的方法以室内场景的 3D 点云与带位姿的 `RGB-D` 图像为输入。我们通过提取场景中的实例及其视觉嵌入，构建一张开放词汇场景图。给定任务描述后，`T-FunS3D` 在场景图中识别最相关的实例，并借助视觉语言模型定位其功能部件。在 `SceneFun3D` 数据集上的实验表明，`T-FunS3D` 在开放词汇 3D 功能分割上达到与当前最优方法相当的水平，同时实现了更快的推理速度与更低的内存占用。

## 创新点

- **免训练的（training-free）任务驱动层级分割**：不微调任何模型，仅靠自由文本任务描述即可从 3D 场景中定位可交互功能部件；把"实例级识别"与"部件级功能分割"拆成两阶段，仅对任务相关对象做细粒度分割，避免整场景穷举过分割。
- **开放词汇场景图以视觉嵌入编码节点与边**：节点存 `FG-CLIP` 视觉嵌入而非类别标签；边（物体间空间关系）在**查询时**针对候选上下文/参照对象对临时计算，而非像 `ConceptGraphs` 那样预存全部关系——既保留开放世界参照定位能力，又省去预计算与存储全关系图的开销。
- **用 LLM 联合抽取任务本体并显式抽取空间关系**：`Qwen3-14B` 一次性以 `JSON` 输出任务本体；其中包含上下文对象、功能部件、参照对象与空间关系，解决"只抽上下文对象会歧义、只抽功能部件不知作用于门还是窗"的问题。
- **两个关键工程取舍被消融证实有效**：① 用 `masked crop`（遮住无关像素）缓解前景-背景偏差；② `SAM` 取**最小** mask 而非最高置信 mask，避免把整物当部件。两者共同带来明显精度/IoU 提升。
- **效率**：场景图只建一次并缓存视图/嵌入，新查询无需像 `Fun3DU` 那样在所有 2D 视图重跑对象检测，功能分割每查询 78s（A40）且内存更低。

## 一句话总结

用两阶段、`training-free` 的 pipeline 完成开放词汇 3D 功能分割：第一阶段离线建图（开放词汇场景图），第二阶段每次查询轻量检索与分割。

它把多个预训练模型拼接为任务驱动的层级式框架：`Mask3D` + `FG-CLIP` + `Qwen3` + `Molmo` + `SAM`。

在 `SceneFun3D` 上以更低时延与内存达到甚至超过 `Fun3DU`。

## 研究问题

开放词汇 3D **功能分割**（即 affordance 分割）：给定带位姿的 `RGB-D` 序列与室内点云，以及一句自由文本操作任务查询，目标是得到定位"可交互功能部件"的 3D mask。

论文把查询形式化拆解为：

$$Q = \{C,\, F,\, (S_1,R_1),\, (S_2,R_2),\, \dots\}$$

- $C$：上下文对象，即机器人要操作的对象，如"用暖气片温控器调温"中的暖气片。
- $F$：功能部件，如温控器、抽屉、门把手。
- $R$：参照对象，用作空间坐标参考，如"打开床右侧床头柜抽屉"中的床。
- $S=(C,R)$ 是二者的二元空间关系；$R$ 与 $S$ 均可选、且可有多个。

现有方法短板（论文立论基点）：

- 物体级开放词汇方法（`OpenMask3D`/`OpenIns3D`/`LeRF`）只有实例、没有部件，且彼此无关系推理。
- 全场景部件方法（`Search3D`）穷举过分割整场景，算力/存储不可承受。
- `Fun3DU` 对每个新查询都要在所有 2D 视图重跑对象检测，效率低。
- 既有功能场景图（`OpenFunGraph`/`FunGraph`）用 `LLM` 生成文本描述建边，重且慢；且大多只到实例级，缺部件级。

论文假设环境布局**静态**（实例间空间关系保持，允许局部小位移，如茶几可局部移动但仍"在沙发右、暖气片左"）。

> [!figure] Fig. 1 方法概念总览（teaser）
> 建议位置：研究问题 / 方法主线
> 放置原因：teaser 图直观展示"建图→查询→分割"两阶段，但当前提取结果被判为含大量文本块、无法稳定恢复成可独立解释的完整原图。
> 当前状态：保留占位；当前提取结果只拿到局部子图，未找到高置信度整图。

## 数据与任务定义

- 数据集：`SceneFun3D` 验证集，30 个场景、445 条任务描述。
- 约四分之三的描述带参照式表达（referring expressions，见图 Fig. 3）；部分描述含多个空间关系。
- 输入：带位姿的 `RGB-D` 帧序列 + 室内 3D 点云（提供真实相机位姿）。
- 输出：与任务相关的功能部件 3D mask。
- 评测指标（遵循 `SceneFun3D` 与 `Fun3DU`）：
- 精度类：AP25、AP50、mAP、mIoU。
- 召回类：AR25、AR50、mAR。
- 基线：主基线 `Fun3DU`；因细粒度功能分割开源方法稀缺，另对比开放词汇 3D 实例分割 `OpenMask3D`/`OpenIns3D`/`LeRF`；并复现 `Fun3DU`（同任务解析输入）记为 `Fun3DU†`。

> [!figure] Fig. 3 验证集任务描述中的空间关系分布
> 建议位置：数据与任务定义
> 放置原因：这张图说明测试查询中参照式描述的比例与类型分布，是理解"为何要显式建模空间关系"的关键背景。
> 当前状态：保留占位；当前提取结果被判为含大量文本块（large_text_block_suspected），不足以稳定替换。

## 方法主线

整体 pipeline（Fig. 2）：输入 posed `RGB-D` + 点云，分两阶段、四模块。

![[images/page_003_fig_fig_2.png]]
*论文原图编号：Fig. 2。T-FunS3D 总览：先用 `FG-CLIP` 关联 `Mask3D` 建立开放词汇实例分割与场景图。*
*任务到达后，用 `Qwen3` 解析任务本体并定位上下文对象，再用 `Molmo` 与 `SAM` 聚合 2D mask 得到功能部件 3D 分割。*

### 机制流程

1. 离线建图（Stage I，每场景一次）：生成实例提议并编码视觉嵌入。
`Mask3D` 生成类别无关实例提议；对每实例取 top-$k$ 高可见度视图。
在**全尺寸 RGB、实例 crop、masked crop** 多尺度上提 `FG-CLIP` 视觉嵌入并跨视图/尺度平均，得到节点嵌入。
据此建场景图 $G=(V,E)$，节点存嵌入，边在查询时算。
2. 任务解析（Stage II-C）：
`Qwen3-14B` 关闭思维链、用多轮对话加速，把自由文本解析为 `JSON` 任务本体。
解析时显式抽取空间关系，输出字段如下。
$$\{C,F,R,S\}$$
字段集合为上下文、功能部件、参照对象与空间关系的联合表达。
3. 场景图检索（Stage II-D）：
用 `FG-CLIP` 文本编码器 $g(\cdot)$ 把 $t_C,t_R,t_S$ 编码，按视觉-文本余弦相似度检索。
锚定上下文与参照对象实例。
4. 功能部件分割（Stage II-E）：
把上下文对象选中视图送 `Molmo-7B-D`，以解析本体作提示输出功能部件像素坐标。
作为 `SAM` 提示得 2D mask，取最小 mask，多视图反投影聚合得 3D mask。

### A. 开放词汇 3D 实例分割

沿用 `OpenMask3D` 思路：`Mask3D` 出类别无关掩码提议。
对每实例 $P_n$（$n$ 属于 $\{1,\dots,N\}$）取高可见度视图。
每视图在 $L$ 个尺度上做 crop，取 top-$k$ 共 $k$ 个视图。
实例 crop 记为 $I_{n,l}^{k}$。
遮去无关像素的 masked crop 记为 $M_{n,l}^{k}$。
用 `FG-CLIP` 视觉编码器从**全尺寸 RGB、crop、masked crop** 三类来源提嵌入，最终实例嵌入为跨视图与尺度的平均：

$$f(P_n) = \mathrm{mean}_{k\in\{1..K\},\,l\in\{1..L\}}\big[\,f(I_n^{k}) + f(I_{n,l}^{k}) + f(M_{n,l}^{k})\,\big]$$

> 注：上式为"跨视图/尺度对三种来源嵌入取平均"的聚合表述；论文原文写作在视图与尺度上平均，未给出标量加权细节。与 `OpenMask3D` 仅用对象 crop 不同，这里额外用全尺寸与 masked crop，以保留上下文并抑制前景-背景偏差（如窗口框被误认为窗外植物）。实现上只保留高置信类别无关 mask（一半以上提议无意义或重复），并每 5 帧采样 1 帧、取 $k=5$ 视图。

### B. 开放词汇场景图

$$G = (V, E),\quad V=\{v_i\},\quad E=\{e_{ij}=(v_i,v_j)\}$$

- 节点 $v_i$ 存**视觉嵌入**（非类别标签，借鉴 `HOV-SG`），支持开放词汇查询。
- 边编码两节点对应全尺寸图像视觉嵌入的平均，从而把实例周围的语义上下文与跨视图空间关系一并编码。
- 关键：边在查询时针对候选 $(C,R)$ 对临时计算，而非像 `ConceptGraphs` 那样预存全关系图。

### C. 任务理解（LLM）

用 `Qwen3-14B` 把自由文本解析为 `JSON` 的任务本体。
其中 $Q$ 联合输出 $\{C,F,R,S\}$，即上下文对象、功能部件、参照对象与空间关系。
单独抽 $C$ 会得到歧义标签（多门场景只给 door），单独抽 $F$ 会得到 handle 却不知是门还是窗的把手。
故**联合输出**所有类别，并显式抽取空间关系（区别于 `Fun3DU`）。

### D. 上下文对象定位

视觉嵌入 $f(\cdot)$ 与文本嵌入 $g(\cdot)$ 分别由 `FG-CLIP` 的图像/文本编码器给出。
$\mathrm{sim}(\cdot,\cdot)$ 为余弦相似度。上下文对象候选：

$$C_i = \arg\max_{v_i\in V}\ \mathrm{sim}\big(f(v_i),\, g(t_C)\big)$$

若存在参照对象，类似地得到参照对象候选 $R_j$：

$$R_j = \arg\max_{v_j\in V}\ \mathrm{sim}\big(f(v_j),\, g(t_R)\big)$$

再在所有候选对上，检验与查询空间关系 $t_S$ 的相似度：

$$S_{ij}=\{(C_i,R_j)\},\quad S = \arg\max_{s\in S_{ij}}\ \mathrm{sim}\big(f(s),\, g(t_S)\big)$$

保留具有有效 $S$ 的候选对作为上下文/参照对象；无空间关系时直接取最佳上下文节点。

### E. 功能部件分割

参考 `Fun3DU`：用 `Molmo-7B-D` 在选中视图上依解析本体提示检测功能部件像素坐标，再作为 `SAM` 提示生成 2D mask。
关键设计：`SAM` 每提示点出多个 mask 并给出置信（估计 IoU）。
最高分 mask 往往分割**整物**而非部件，尤其低分辨率 crop 时。
故改用**最小 mask**。
该选择在 `TABLE IV` 的消融实验中被证实更优。
2D mask 反投影到 3D，多视图聚合得最终 3D 功能分割。

## 关键结果

### TABLE I 功能部件分割（SceneFun3D val）

主对比聚焦三个有意义系统（论文核心 claim 锚定在 `Fun3DU` 与 `Fun3DU†`）。

| method | mAP | AP50 | AP25 | mAR | AR50 | AR25 | mIoU |
|---|---|---|---|---|---|---|---|
| `Fun3DU` [5] | 7.6 | 16.9 | 33.3 | 27.4 | 38.2 | 46.7 | 15.2 |
| `Fun3DU`†（同解析复现） | 4.4 | 10.3 | 23.4 | 30.9 | 42.3 | 49.7 | 12.0 |
| **T-FunS3D（ours）** | **8.1** | **17.8** | **34.5** | 23.8 | 35.8 | 46.9 | **15.7** |

- 对比原 `Fun3DU`：AP25 +1.2、mIoU +0.5（小幅领先，精度/IoU 最高）。
- 对比同输入复现 `Fun3DU†`：AP25 +11.1、mIoU +3.7（大幅领先）。
- `OpenMask3D`/`OpenIns3D`/`LeRF` 等纯实例分割方法在功能分割上精度近 0（倾向分割整物而非细粒度部件），论文正文已定性说明，数值从略。

### TABLE II 带空间参照描述的子集

| method | mAP | AP50 | AP25 | mAR | AR50 | AR25 | mIoU |
|---|---|---|---|---|---|---|---|
| `Fun3DU`† | 3.83 | 9.26 | 22.53 | 30.77 | 41.36 | 48.77 | 11.73 |
| **T-FunS3D（ours）** | **8.11** | **19.20** | **34.98** | 25.17 | 37.46 | 47.06 | **16.24** |

在含参照式表达的任务子集上，T-FunS3D 比 `Fun3DU†` 高 +12.45 AP25 / +4.51 mIoU——空间关系抽取 + 场景图边使参照定位显著更准。

### TABLE III 运行时间（NVIDIA A40；Fun3DU 功能分割在 A100 上报 118.4s）

| method | ① 实例分割（每场景） | ② 功能分割（每查询） |
|---|---|---|
| `OpenMask3D` [11] | a 30s + b 720s | — |
| `Fun3DU` [5] | 1920s | 167s |
| **T-FunS3D（ours）** | **a 12s + b 580s** | **78s** |

T-FunS3D 实例阶段更快（仅保留高置信提议），功能阶段因复用缓存视图而远快于 `Fun3DU`；且每个新查询无需重跑整条 pipeline。

### TABLE IV 消融（验证集前 10 visits）

| config | mAP | AP50 | AP25 | mAR | AR50 | AR25 | mIoU |
|---|---|---|---|---|---|---|---|
| `Fun3DU`† | 4.41 | 10.49 | 22.38 | 30.28 | 41.26 | 49.65 | 11.07 |
| Ours w/o FI（去全尺寸图） | 5.73 | 13.99 | 26.57 | 22.80 | 33.57 | 41.26 | 13.40 |
| Ours w/o MC（去 masked crop） | 5.53 | 17.02 | 29.08 | 21.84 | 28.37 | 36.17 | 13.37 |
| Ours w/o SM（用最高分 mask） | 4.46 | 10.36 | 23.42 | 30.95 | 42.34 | 49.77 | 12.03 |
| **Ours standard** | **6.41** | **15.49** | **31.69** | 23.03 | 32.39 | 40.14 | **14.98** |

- 富像素信息（FI+MC）贡献约 +0.8 mAP / +1.5 mIoU。
- **最小 mask（SM）**贡献约 +2.0 mAP / +3.0 mIoU。
- 去掉最小 mask 时召回更高。
- 具体为 mAR 30.95 vs 23.03。
- 说明最高分 mask 更偏"整物"，最小 mask 把预测聚焦到功能实体。
- 代价是更多漏检（false negative）。

## 深度分析

![[images/page_007_fig_fig_4.png]]
*论文原图编号：Fig. 4。定性结果：红=预测、蓝=真值、绿=重叠。*
*可见大斜视视角与无物理依附部件（如顶灯开关）会失败。*

### 为什么结果成立

- 最小 mask 与 masked crop 直接提升细粒度定位。
- `TABLE IV` 显示去掉任一都掉点。
- 最小 mask 单独贡献最大，约 +2.0 mAP 与 +3.0 mIoU。
- 机理是抑制"整物级"预测，把 mask 收敛到功能部件。
- 边在查询时计算，加 LLM 抽空间关系，使参照定位更准。
- `TABLE II` 上带参照描述的子集提升 +12.45 AP25。
- 机制来自 (C) 的 $S$ 抽取与 (D) 的边匹配。
- 缓存视图、只建一次图，使时延下降；`TABLE III` 功能阶段 78s vs `Fun3DU` 167s（A40），核心是第一阶段只跑一次、第二阶段只处理选中对象而非全图。

### 哪些地方容易被误读

- "training-free" 指本方法本身不训练，边界需注意。
- 但所用模块均为已预训练模型。
- 包括 `Mask3D`（ScanNet200 训练）。
- 视觉模型用 `FG-CLIP`。
- 语言与分割用 `Qwen3`、`Molmo`、`SAM`。
- 因此 `training-free` 不等于完全无训练依赖。
- +1.2 vs +11.1 的差距：与**原** `Fun3DU`（其自带任务解析）仅 +1.2 AP25；与**同输入复现**的 `Fun3DU†` 才 +11.1。性能增益有一部分来自"公平同输入"的复现基线较弱，不宜外推为全面碾压。
- 运行时间不可直接横比：`T-FunS3D` 报 A40、`Fun3DU` 功能阶段报 A100（118.4s），GPU 与实现不同，绝对数值不完全等价。
- 召回反而略低：最小 mask 带来精度收益是以召回下降为代价的（TABLE I/II mAR 低于 `Fun3DU`），即存在漏检。

## 局限

- 大斜视（oblique）视角导致分割明显不准（Fig. 4 第 2、3 列精度/召回低）。
- 功能部件与上下文对象**无物理依附**时失败：顶灯开关示例 $IoU=0$（`Fig. 4` 最右列）。
- 召回略低于 `Fun3DU`（最小 mask 取舍）。
- 仅 `SceneFun3D` 验证集评测（30 场景/445 任务），缺跨数据集验证；与同期功能场景图（`OpenFunGraph`/`FunGraph`）无头对头对比。
- 假设环境**静态**，布局变化则需重建场景图；模块间无容错机制（作者列为未来工作）。

## 我的笔记

- 与本知识库的关系：本工作属于开放词汇 3D `affordance` / 功能分割线。
- 与库内 `GEAL`、`3DAffordSplat`、`Aff3DFunc`、`Fun3DU` 同域。
- 区别在于 T-FunS3D 是纯 `training-free` 的 2D 基础模型拼接 pipeline，不做跨模态蒸馏、也不学 3DGS 表征。
- 而 `GEAL` 走 2D 教师→3D 分支蒸馏以换推理效率。二者在是否训练轻量 3D 分支上根本对立。
- 可复用技巧：
- `SAM` 取最小 mask 做部件分割。
- `masked crop` 去前景-背景偏差。
- 场景图边在查询时计算，而非预存。
- 任务用 `LLM` 联合 `JSON` 输出 $\{C,F,R,S\}$ 并显式抽空间关系。
- 复现注意：`Mask3D` 在 ScanNet200（无顶、点更稀）训练，需预处理 `SceneFun3D` 点云以适配；只保留高置信类别无关 mask 省算力；每 5 帧采 1 帧、取 top-$k=5$ 视图。

## 引用

- [[Fun3DU - Functionality Understanding and Segmentation in 3D Scenes|Fun3DU (Corsetti et al., CVPR 2025)]] — 主基线，同样用 `Molmo`+`SAM` 做部件，但每查询重跑对象检测。
- Delitzas et al. (2024). `SceneFun3D`: Fine-grained functionality and affordance understanding in 3D scenes. CVPR.
- Takmaz et al. (2024). `Search3D`: Hierarchical Open-Vocabulary 3D Segmentation. RA-L.
- Zhang et al. (2025). `OpenFunGraph`: Open-vocabulary functional 3D scene graphs. CVPR.
- Rotondi et al. (2025). `FunGraph`: Functionality aware 3D scene graphs. IROS.
- Takmaz et al. (2023). `OpenMask3D`: open-vocabulary 3D instance segmentation. NeurIPS.
- Huang et al. (2024). `OpenIns3D`. ECCV.
- Kerr et al. (2023). `LeRF`: Language embedded radiance fields. ICCV.
- Schult et al. (2023). `Mask3D`. ICRA.
- Xie et al. (2025). `FG-CLIP`. ICML.
- Yang et al. (2025). `Qwen3` technical report.
- Deitke et al. (2025). `Molmo` and `PixMo`. CVPR.
- Kirillov et al. (2023). `SAM`: Segment anything. ICCV.
- Werby et al. (2024). `HOV-SG`: Hierarchical open-vocabulary 3D scene graphs. ICRA Workshop.
- Gu et al. (2024). `ConceptGraphs`. ICRA.
