---
title: "GEAL Unseen 指标低与泛化性改进：问题诊断及相关论文"
aliases:
  - GEAL unseen 泛化
  - 3D affordance grounding 泛化综述
created: 2026-08-07
status: research-brief
base_paper: "GEAL: Generalizable 3D Affordance Learning with Cross-Modal Consistency"
---

# GEAL Unseen 指标低与泛化性改进：问题诊断及相关论文

## 1. 先给结论

GEAL 的 Unseen 低，并不单纯是 PointNet++ "不够强"，更可能是以下因素共同造成：

1. **语义组合泛化不足**：LASO Unseen 主要是训练中未出现的 object-affordance 组合；PIAD Unseen 更接近未见物体类别迁移。模型可能学到的是“类别—部件共现”，而不是可迁移的 affordance 原理。
2. **affordance 语义表达过粗**：`grasp`、`pour` 等裸标签无法表达动作、功能、形态、环境和安全约束。
3. **局部功能部件太小**：把手、按钮、刀尖等只占少量点；多尺度抽象或 dominant-label 监督容易将小区域淹没。
4. **跨模态迁移仍有域差距**：GEAL 的 2D 教师由点云渲染得到，DINOv2 的通用图像语义可以迁移，但合成深度图与真实视觉、点云几何之间仍存在差距。
5. **二维先验与三维边界职责混在一起**：2D 特征擅长语义候选，3D 几何才更适合确定局部边界；直接依赖 2D 预测可能产生漏检或边界漂移。
6. **训练数据和评测协议仍有限**：GEAL 的 PIAD-C/LASO-C 主要是 Seen 数据上的单一合成 corruption，尚未充分覆盖 unseen + corruption、组合扰动、真实传感器域偏移。

因此最推荐的方向不是直接把 GEAL 换成更大的 backbone，而是：

> **结构化功能文本 + 跨类别局部几何原型 + visibility-aware 的局部 2D–3D 对齐 + 软先验/局部残差细化 + 更严格的双重分布外评测。**

---

## 2. 论文证据分层

### A. 直接针对 3D affordance unseen/open-vocabulary 的论文

| 论文 | 主要解决的泛化瓶颈 | 方法启示 | 证据强度 |
|---|---|---|---|
| **LASO: Language-guided Affordance Segmentation on 3D Object**, CVPR 2024 | 传统 3D affordance 方法缺少语言条件，难以泛化到未见 object-affordance 组合 | PointRefer 的自适应多尺度融合 + 语言条件动态核；明确设置 Seen/Unseen 组合划分 | 直接；但其 Unseen 仍主要是组合泛化 |
| **Grounding 3D Object Affordance from 2D Interactions in Images**, ICCV 2023 | 只从静态几何学习，缺少真实交互先验 | IAGNet 对齐 2D 交互区域与 3D 几何，利用人—物交互图像迁移 affordance | 直接；PIAD 上验证，外部图像仍有域差距 |
| **Open-Vocabulary Affordance Detection in 3D Point Clouds**, IROS 2023 | 固定 affordance 标签集合，无法识别训练外 affordance | OpenAD 联合学习文本和点特征，支持 zero-shot affordance detection | 直接；但主要依赖文本—几何相似性，细粒度边界有限 |
| **GEAL: Generalizable 3D Affordance Learning with Cross-Modal Consistency**, CVPR 2025 | 标注少、3D 主干语义弱、点云 corruption 鲁棒性不足 | 3DGS 建立可追踪 2D–3D 对应；DINOv2 教师 + GAFM + CAM；推理只保留 3D 分支 | 直接；PIAD/LASO 的 unseen 与 corruption 验证，但 PIAD Unseen aIoU 仍仅 8.7 |
| **GREAT: Geometry-Intention Collaborative Inference for Open-Vocabulary 3D Object Affordance Grounding**, CVPR 2025 | 固定语义空间、未利用跨物体不变几何、缺少类比/意图推理 | 用 MLLM 的多头 affordance CoT 提取几何属性和潜在交互意图，再通过跨模态自适应融合注入点云和图像；构建 PIADv2 | 直接；重点是 open-vocabulary 和 unseen 泛化，但计算与推理链较重 |
| **Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions**, CVPR 2025 | 真实机器人只能获得部分视角、旋转视角或遮挡观察 | AGPIL 数据集 + LMAffordance3D，联合语言、视觉观察和交互信息；在 unseen 设置下评估 | 直接；更贴近部分观测/交互场景，不等同于 LASO 的 object split |
| **Open-Vocabulary 3D Affordance Understanding via Functional Text Enhancement and Multilevel Representation Alignment**, ACM MM 2025 | 裸 affordance label 语义过粗；文本—几何表征判别性不足 | FTE 从 actions/functions/appearance/environment 构造功能文本；多层语义—几何对齐 + supervised contrastive learning | 直接；zero-shot label/question 查询，Label full-view mIoU 0.2942，但 Question zero-shot mIoU 仅 0.1315 |
| **Affogato: Open-Vocabulary Affordance Grounding with Automated Data Generation at Scale**, arXiv:2506.12009v2, 2026 | 训练数据规模、类别多样性和跨域覆盖不足 | 自动生成 Affogato-750K：约 150,104 个 3D instances、750,520 个 affordance annotations、>450 类物体、>350 类 affordance，并提供 5K 人工核验测试集；提出 Espresso-3D | 直接；是目前最直接的数据规模化解决路线 |
| **QueryMe: Query-Driven Open-Vocabulary 3D Object Affordances Grounding from Multimodal Evidence**, CVPR 2026 | 固定类别、几何先验不足、2D→3D 域差、缺乏几何不变性和类比推理 | 将人—物交互图像投影到 3D；Adaptive Spatial Attention 聚焦交互区域；多模态 query 检索几何一致的功能部件，并利用几何相似性做类比 | 直接；论文报告 unseen affordance grounding AUC 相比前作提升 4.19% |

### B. 与 GEAL 改造高度相关的邻近论文

| 论文 | 可迁移思想 | 与 GEAL 的对应关系 |
|---|---|---|
| **Task-Aware 3D Affordance Segmentation via 2D Guidance and Geometric Refinement**, AAAI 2026 | 2D 语义负责任务相关候选视角，3D Point Transformer 负责局部几何细化；粗到细 | 直接支持 GEAL 的“2D 高召回、3D 定边界”非对称分工；适合引入软二维先验和局部残差细化 |
| **SceneFun3D: Fine-Grained Functionality and Affordance Understanding in 3D Scenes**, CVPR 2024 | 从单物体扩展到真实室内场景、功能部件、任务驱动 affordance 和运动参数 | 提醒 GEAL 不能只在单物体静态点云上声称真实场景泛化；应增加 scene-level、遮挡和动作条件评测 |
| **Zero-Shot Point Cloud Segmentation by Semantic-Visual Aware Synthesis**, ICCV 2023 | 用语义—视觉特征合成未见类别特征，并加一致性正则 | 可将 affordance 原型/局部几何 token 做 feature synthesis，缓解 unseen object-affordance 组合样本不足 |
| **See More and Know More: Zero-shot Point Cloud Segmentation via Multi-modal Visual Data**, ICCV 2023 | 利用图像外观补足无纹理点云，增强视觉—语义对齐 | 支持 GEAL 的跨模态路线，但也说明真实图像、合成渲染与点云之间的域差必须单独控制 |
| **Generalized Few-Shot Point Cloud Segmentation Via Geometric Words**, ICCV 2023 | 学习跨 base/novel 类共享的 geometric words 和 geometric prototypes | 适合构建 affordance-aware local geometric code/token；比直接做大规模 VQ 更稳妥 |

---

## 3. 最值得优先复现的直接证据

### 3.1 Affogato 对“增加数据多样性”的证据

Affogato-750K 的实验定义是：先在 Affogato-750K train 上预训练，再在 LASO train 上微调。

LASO test split 结果如下，指标顺序为 `aIoU / AUC / SIM / MAE`：

| 方法 | Seen | Unseen |
|---|---|---|
| OpenAD | 14.2 / 85.1 / 53.3 / 0.103 | 14.6 / 80.7 / 51.8 / 0.109 |
| OpenAD + Affogato-750K | **16.1 / 86.8 / 53.9 / 0.100** | **15.5 / 81.8 / 53.4 / 0.103** |
| PointRefer | 20.8 / 87.3 / 62.9 / 0.093 | 14.6 / 80.2 / 50.7 / 0.119 |
| PointRefer + Affogato-750K | 20.2 / 86.0 / 60.0 / 0.098 | **18.6 / 81.4 / 56.1 / 0.103** |
| Espresso-3D | 20.4 / 86.0 / 63.3 / 0.102 | 18.7 / 80.0 / 60.0 / 0.101 |
| Espresso-3D + Affogato-750K | **21.9 / 85.9 / 63.7 / 0.116** | **20.8 / 82.9 / 61.4 / 0.122** |

关键解读：

- OpenAD：Seen aIoU +1.9，Unseen aIoU +0.9，AUC 和 SIM 提升，MAE 下降；说明规模化预训练对较弱 baseline 也有效。
- PointRefer：Seen 轻微退化，但 Unseen aIoU +4.0，是表中 Unseen aIoU 增益最大的已有方法；说明大规模数据预训练可能牺牲少量 seen 适配换取更强组合泛化。
- Espresso-3D：Unseen aIoU +2.1、AUC +2.9、SIM +1.4，但 MAE 从 0.101 变为 0.122；不能说所有指标都改善。
- Affogato 的证据支持“数据规模与类别/功能多样性是 unseen 泛化瓶颈”，但自动生成标注仍可能带来噪声，且 Affogato 预训练并不能保证每个指标、每个域都提升。

Affogato-750K 自身的跨域实验把对象域拆为 `Daily-used` 与 `Furnitures`：

- `All → All`：所有对象域内训练和测试；
- `Daily-used → Furnitures`：日用品训练、家具测试；
- `Furnitures → Daily-used`：家具训练、日用品测试。

Espresso-3D 的 aIoU 为 `13.6 / 18.2 / 4.6`，显示明显的方向不对称：日用品域覆盖更广，迁移到家具比家具迁移到日用品容易。

### 3.2 Aff3DFunc 对“改进文本语义”的证据

Aff3DFunc 的 full-view Label-as-Query mIoU 为 `0.2942`，KD-TPC 为 `0.2233`；Question-as-Query zero-shot mIoU 为 `0.1315`。组件消融中：

- baseline → +FTE：Label mIoU 从 `0.1531` 到 `0.2504`；
- +SC+FTE+ML+CA：Label mIoU 达到 `0.2942`；
- FTE 采用 actions/functions/appearance/environment 四种描述视角；
- 多层对齐和监督对比学习分别增强不同粒度的语义—几何对应与类别判别性。

对 GEAL 的启示：在 GAFM 前加入结构化 functional text 或 query decomposition，可能比单纯扩大文本编码器更直接；但必须避免 dominant-label 多层监督压制小 affordance。

### 3.3 GREAT / QueryMe 对“几何不变性与类比推理”的证据

GREAT 明确把问题归因于：有限语义空间、没有挖掘跨物体不变几何、缺少潜在交互意图推理。QueryMe 进一步使用 query-driven 的多模态证据检索几何一致的功能部件，并报告 unseen affordance AUC 相比前作提升 4.19%。

对 GEAL 的启示不是直接加入完整 MLLM/CoT，而是先实现轻量版本：

```text
affordance query
→ action / goal / part / geometry attribute 分解
→ 检索跨类别局部几何 prototype
→ 与 PointNet++ / local token 做 soft matching
→ 点级 affordance decoder
```

---

## 4. 给 GEAL 的解决方案优先级

### P0：先把 unseen 诊断拆开

不要只报告一个 Unseen 均值，至少拆为：

1. `seen object + unseen affordance`；
2. `unseen object + seen affordance`；
3. `unseen object + unseen affordance`；
4. 新 query 同义改写；
5. 小区域 / 中区域 / 大区域；
6. clean / corruption / unseen+corruption。

同时报告：aIoU、AUC、SIM、MAE、小区域召回、Boundary F-score、假阳性点比例，以及三随机种子均值±标准差。

### P1：结构化 affordance 文本

借鉴 Aff3DFunc，但不必一开始引入在线 LLM：

```text
query
→ action: 做什么
→ goal: 达成什么结果
→ part: 可能是哪类部件
→ geometry: 具有什么几何属性
→ context: 在什么场景/关系下成立
→ safety: 哪些区域不能碰
```

保留完整 query 分支，结构化文本只作为辅助分支；用 gated fusion，避免错误解析导致语义覆盖。

### P2：二维软先验 + 三维局部几何细化

借鉴 GEAL 的 3DGS 和 TASA：

- 2D 教师输出 affordance heatmap，不做硬裁剪；
- 用 3DGS alpha/visibility 将多视角热图反投影为点级软先验；
- PointNet++ 保留全局分支；
- 对高置信、边界和高不确定区域做 KNN/Point Transformer 局部残差细化；
- 最终预测为 `global 3D logit + gated local residual`。

该路线比完全替换 PointNet++ 或直接依赖 2D lift 更安全。

### P3：visibility-aware token-level CAM

在 GEAL 当前 feature-level MSE 之上增加：

1. 3D local token；
2. 2D ROI token；
3. 由 3DGS 可见性建立 soft matching；
4. 对比损失只在高可见、低歧义 token 上计算；
5. 遮挡区域和一对多投影使用 mask/soft assignment，不强制一一对应。

### P4：跨类别几何 prototype / geometric words

先使用连续 prototype，不建议第一版直接上大 VQ codebook：

- handle-like；
- support-plane；
- graspable-edge；
- pushable-surface；
- container-opening；
- spout-like。

prototype 只作为 query-conditioned 辅助检索，不替代逐点连续特征，以避免边界精度下降。

### P5：规模化预训练与鲁棒一致性

如果能获得 Affogato-750K 或构建相似数据，先做：

```text
大规模 affordance 预训练 → GEAL / 改进模型微调
```

再加入 clean-corrupt consistency，但必须分别报告：

- clean-trained → clean；
- clean-trained → corrupt；
- corruption-augmented → corrupt；
- unseen + corrupt。

局部删点时只约束仍存在的对应点，不能要求模型恢复完全不存在的几何证据。

---

## 5. 推荐的 GEAL 主线

### 最小可行版本

```text
GEAL baseline
+ P0 unseen 细分评测
+ P1 结构化 affordance text
+ P2 2D soft prior + local geometric residual
```

### 更适合论文主模型的版本

```text
GEAL
+ structured functional query
+ visibility-aware soft 2D prior
+ local 3D token/refiner
+ token-level 2D–3D consistency
```

### 暂不建议第一版加入

- 完整 MLLM 在线 CoT；
- GRPO 或开放式文本奖励；
- 大规模 VQ codebook；
- 只依赖二维候选的硬裁剪；
- 完全替换 PointNet++；
- 同时引入模型结构、数据增强和新评测协议，导致无法做因果归因。

---

## 6. 可以形成的论文研究问题

1. **RQ1：** 结构化功能文本是否能比裸 affordance label 更有效地提升 GEAL 的 unseen object-affordance 组合泛化？
2. **RQ2：** visibility-aware 的局部 token 对齐是否比 GEAL 的连续 feature-level CAM 更能改善小区域与边界定位？
3. **RQ3：** 跨类别局部几何 prototype 能否在不损害 Seen 性能的情况下提升 `unseen object + seen affordance`？
4. **RQ4：** 大规模自动生成 affordance 数据预训练的收益，是否能迁移到 GEAL 的 2D 教师—3D 学生架构？
5. **RQ5：** clean–corrupt consistency 是否能同时改善 unseen 与 corruption，还是只改善已见类别上的鲁棒性？

---

## 7. 重要限制与证据边界

- GREAT、QueryMe、Affogato 的摘要/公开页面支持其方法方向和总体结论；若写论文中的完整逐类数字，应再核对 PDF/补充材料原始表格。
- Aff3DFunc 的结果来自本地 PDF 笔记与论文资料，适合做方法与数量级参考；Question-as-Query 的绝对 mIoU 仍低，不能把它表述为已解决开放世界 grounding。
- Affogato 的自动标注规模非常大，但自动生成并不等于无噪声；收益可能来自数据规模、类别覆盖、预训练目标和模型初始化的共同作用。
- GEAL PIAD Unseen aIoU 低，说明“优于基线”不能等同于“新类别定位可靠”。后续工作必须报告绝对值、方差和 failure cases，而不能只报告相对提升。

---

## 8. 论文链接

- GEAL：<https://openaccess.thecvf.com/content/CVPR2025/html/Lu_GEAL_Generalizable_3D_Affordance_Learning_with_Cross-Modal_Consistency_CVPR_2025_paper.html>
- LASO：<https://openaccess.thecvf.com/content/CVPR2024/html/Li_LASO_Language-guided_Affordance_Segmentation_on_3D_Object_CVPR_2024_paper.html>
- IAGNet：<https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Grounding_3D_Object_Affordance_from_2D_Interactions_in_Images_ICCV_2023_paper.html>
- OpenAD：<https://openad2023.github.io/>；<https://doi.org/10.48550/arXiv.2303.02401>
- GREAT：<https://openaccess.thecvf.com/content/CVPR2025/html/Shao_GREAT_Geometry-Intention_Collaborative_Inference_for_Open-Vocabulary_3D_Object_Affordance_Grounding_CVPR_2025_paper.html>
- LMAffordance3D：<https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_Grounding_3D_Object_Affordance_with_Language_Instructions_Visual_Observations_and_CVPR_2025_paper.html>
- Aff3DFunc：<https://doi.org/10.1145/3746027.3755239>；<https://wulin97.github.io/aff3dfunc/>
- Affogato：<https://arxiv.org/html/2506.12009>
- QueryMe：<https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_QueryMe_Query-Driven_Open-Vocabulary_3D_Object_Affordances_Grounding_from_Multimodal_Evidence_CVPR_2026_paper.html>
- TASA：<https://doi.org/10.1609/aaai.v40i6.42466>；<https://arxiv.org/abs/2511.11702>
- SceneFun3D：<https://openaccess.thecvf.com/content/CVPR2024/html/Delitzas_SceneFun3D_Fine-Grained_Functionality_and_Affordance_Understanding_in_3D_Scenes_CVPR_2024_paper.html>
- Zero-Shot Point Cloud Segmentation by Semantic-Visual Aware Synthesis：<https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Zero-Shot_Point_Cloud_Segmentation_by_Semantic-Visual_Aware_Synthesis_ICCV_2023_paper.html>
- See More and Know More：<https://openaccess.thecvf.com/content/ICCV2023/html/Lu_See_More_and_Know_More_Zero-shot_Point_Cloud_Segmentation_via_ICCV_2023_paper.html>
- Generalized Few-Shot Point Cloud Segmentation Via Geometric Words：<https://doi.org/10.1109/ICCV51070.2023.01966>
