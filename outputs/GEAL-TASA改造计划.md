---
title: "GEAL 基于 TASA 启发的分阶段改造计划"
aliases:
  - GEAL 改造计划
  - GEAL-TASA Roadmap
tags:
  - affordance-grounding
  - 3d-vision
  - cross-modal-learning
  - research-plan
status: proposed
created: 2026-08-06
base_model: "GEAL: Generalizable 3D Affordance Learning with Cross-Modal Consistency"
inspiration: "Task-Aware 3D Affordance Segmentation via 2D Guidance and Geometric Refinement"
---

# GEAL 基于 TASA 启发的分阶段改造计划

## 1. 计划摘要

本计划不直接复制 TASA 的完整级联，而是将其最有价值的设计原则嵌入 GEAL：

> **二维分支负责高召回地回答“可能在哪里、是什么功能部件”；三维分支负责利用局部几何回答“精确边界到底在哪里”。**

GEAL 原有的 3DGS、DINOv2、GAFM、CAM 和推理期纯三维分支全部保留。改造重点包括：

1. 任务感知的二维视角与教师语义；
2. 基于 3DGS 可见性的二维软三维先验；
3. PointNet++ 并联局部几何残差细化器；
4. visibility-aware token-level 2D–3D consistency；
5. 主线稳定后的 clean–corrupt consistency 训练。

推荐主线：

```text
V0 基线诊断
  → V1 任务感知视角加权
  → V2 二维软先验反投影
  → V3 局部几何残差细化
  → V4 Token-level CAM
  → V5 可选鲁棒一致性训练
```

最低可行改造版本为 `V0 + V2 + V3`；推荐论文主模型为 `V0–V4`。

---

## 2. 基线与改造边界

### 2.1 GEAL 基线中保留的部分

- 3DGS：建立点云与多视角二维表示的空间对应；
- DINOv2：作为训练期二维视觉教师；
- PointNet++：保留全局及层级三维几何编码能力；
- RoBERTa：继续提供语言 query；
- GAFM：保留粒度自适应视觉—文本融合；
- CAM：保留连续特征级二维—三维一致性；
- 两阶段训练：先训练二维分支，再蒸馏到三维分支；
- 推理阶段：只运行三维分支。

### 2.2 不建议第一版加入的内容

- 完整外部 LLM/VLM 级联；
- 二维候选区域外的硬裁剪；
- RGB 直接输入三维细化器；
- 大规模 VQ codebook；
- 完全替换 PointNet++；
- 同时引入结构改动与 corruption training；
- 直接照搬 SAGE 的 GRPO 或文本奖励。

原因是这些改动会同时增加计算、训练不稳定性和实验归因难度。

---

## 3. 总体架构

```text
语言 query
  → 动作 / 对象 / 功能部件 / 空间关系拆解
  → 任务感知视角加权
  → 3DGS + DINOv2 二维教师
  → 二维 affordance 概率图与可见性权重
  → 软二维先验反投影到三维
  → PointNet++ 全局三维分支
  → 局部 KNN / Point Transformer 几何细化
  → token-level 2D–3D consistency
  → token-to-point decoder
  → 逐点 affordance map
```

核心约束：

1. 二维先验只作为软信息，不作为不可逆的硬筛选；
2. 全局 PointNet++ 分支始终保留，以抵抗二维教师漏检；
3. 局部几何模块采用残差修正，不直接替换 GEAL 预测；
4. 二维分支主要在训练期使用，保持 GEAL 的部署优势；
5. 每个阶段必须有独立对照和晋级门槛。

---

# 4. V0：基线复现与诊断体系

## 4.1 目标

确定 GEAL 的主要瓶颈来自：

- 二维教师没有关注功能部件；
- CAM 的空间对齐过于粗糙；
- PointNet++ 的局部边界能力不足；
- 渲染分辨率导致小区域信息消失；
- 文本融合没有充分影响三维预测。

## 4.2 实施内容

完整复现以下组件：

```text
PointNet++
DINOv2
RoBERTa
GAFM
CAM
3DGS 多视角渲染
二维预训练 + 三维蒸馏
```

初始采用 GEAL 的折中设置：

| 配置项 | 初始值 |
|---|---:|
| 渲染分辨率 | 112×112 |
| 视角数 | 12 |
| 视角条件 prompt | 启用 |
| 优化器 | Adam |
| 初始学习率 | 1×10⁻⁴ |
| 训练轮数 | 50 epoch |
| Batch size | 12 |

## 4.3 必做消融

完整执行 GAFM 与 CAM 的 2×2 因子消融：

| 实验 | GAFM | CAM |
|---|---:|---:|
| B0 | × | × |
| B1 | ✓ | × |
| B2 | × | ✓ |
| B3 | ✓ | ✓ |

同时加入等参数对照：

- 原始 GEAL；
- 参数量匹配的普通 MLP 增强；
- 具有明确机制的改造模块。

## 4.4 评测指标

除 GEAL 原有 AUC、aIoU、SIM、MAE 外，增加：

- 小区域 aIoU；
- 小区域召回率；
- Boundary F-score；
- 假阳性面积；
- affordance 面积分桶结果；
- 不同点云密度下的性能；
- corruption severity 曲线；
- 参数量、FLOPs、显存和推理时间。

## 4.5 晋级门槛

只有满足以下条件，才进入 V1：

- 基线结果基本复现；
- 三个随机种子方向一致；
- GAFM 与 CAM 的独立贡献明确；
- 至少发现二维教师或局部边界中的一个主要瓶颈。

---

# 5. V1：任务感知的视角加权

## 5.1 目标

让二维教师优先关注真正的功能部件，而不只是显著对象。

例如：

```text
打开抽屉
→ 动作：打开
→ 对象：抽屉
→ 功能部件：把手
→ 关系：抽屉前侧的接触区域
```

## 5.2 文本拆解

第一版不必引入大型视觉语言模型，可采用：

- affordance–part 静态映射表；
- 模板或规则抽取；
- RoBERTa 辅助预测功能部件；
- 保留完整任务句作为全局语义分支。

## 5.3 视角打分

对第 `i` 个视角计算：

$$
S_i = \alpha S_{task,i} + (1-\alpha)S_{part,i}
$$

其中：

- `S_task`：完整任务语句与视角的相似度；
- `S_part`：功能部件词与视角的相似度；
- `α`：任务语义与部件语义的平衡系数。

推荐采用软权重聚合：

$$
F^{2D}=\sum_i \operatorname{softmax}(S_i/\tau)F_i^{2D}
$$

不要第一版就完全使用硬 Top-K，以免视角误选导致信息不可恢复。

## 5.4 对照实验

1. 固定均匀视角；
2. 仅完整任务句加权；
3. 仅功能部件加权；
4. 完整任务句与功能部件联合加权；
5. 联合加权 + 硬 Top-K；
6. 联合加权 + 软权重聚合。

## 5.5 风险控制

- 保留完整任务句分支；
- 设置最小均匀视角权重；
- 防止权重集中于单一视角；
- 记录视角权重熵；
- 分析功能部件提取错误对性能的影响。

## 5.6 晋级门槛

- Unseen 指标稳定提高；
- 小区域召回提高；
- Seen 性能下降不超过随机波动；
- 二维计算量没有明显增加。

---

# 6. V2：二维软先验反投影到三维

## 6.1 目标

让二维教师不仅参与 CAM 特征蒸馏，还为三维分支提供任务相关的软候选先验。

## 6.2 先验构造

二维分支输出每个视角的 affordance 概率图。对三维点 `p_j`，使用 3DGS 可见性和视角权重进行反向聚合：

$$
q_j=
\frac{\sum_v a_{jv}c_v\hat y^{2D}_{jv}}
{\sum_v a_{jv}c_v+\epsilon}
$$

其中：

- `a_jv`：点 `p_j` 在视角 `v` 中的可见性或 alpha 权重；
- `c_v`：V1 得到的任务感知视角权重；
- `ŷ²ᴰ_jv`：二维预测在该点投影位置的概率。

同时构造跨视角不确定性：

$$
 u_j=\operatorname{Var}_v(\hat y^{2D}_{jv})
$$

输入三维分支：

$$
F'_j=\operatorname{MLP}([F^{3D}_j,q_j,u_j])
$$

## 6.3 关键原则

**不根据二维结果硬裁剪点云。**

始终保留：

```text
全局三维特征
+ 二维软先验
+ 局部候选特征
```

这样即使二维教师漏检，三维主干仍有机会恢复目标。

## 6.4 必做消融

| 版本 | 二维先验 | 可见性加权 | 不确定性 |
|---|---:|---:|---:|
| P0 | × | × | × |
| P1 | ✓ | × | × |
| P2 | ✓ | ✓ | × |
| P3 | ✓ | ✓ | ✓ |

## 6.5 晋级门槛

- 严格区域指标得到改善；
- 二维漏检样本上的三维召回不出现灾难性下降；
- 可见性加权优于简单均值；
- 不确定性与预测错误存在可解释相关性。

---

# 7. V3：局部几何残差细化器

## 7.1 目标

借鉴 TASA 的三维几何细化思想，利用局部相对几何收紧 affordance 边界，同时保留 GEAL 的全局泛化能力。

## 7.2 推荐结构

```text
PointNet++ 全局特征
        ├────────────→ 基础预测
        │
二维软先验 → 候选中心采样
        │
        → KNN 多尺度邻域
        → Point Transformer / local attention
        → 几何残差
        │
        └────────────→ 最终预测
```

局部模块并联在 PointNet++ 旁边，不替换 PointNet++。

## 7.3 候选采样

建议第一版候选点比例为：

| 候选来源 | 比例 |
|---|---:|
| 高置信二维候选 | 50% |
| 候选边界 | 25% |
| 二维不确定区域 | 15% |
| 全局探索点 | 10% |

这样既利用二维先验，也避免二维前端错误后完全丢失目标。

## 7.4 多尺度局部编码

建议比较：

- `K=16`：细边界；
- `K=32`：局部功能部件；
- `K=64`：较大结构。

相对位置编码：

$$
\gamma_{ij}=\operatorname{MLP}_{pos}(p_j-p_i)
$$

第一版只输入：

- 坐标与相对坐标；
- PointNet++ 特征；
- 二维软先验；
- 二维不确定性。

暂时不加入 RGB 和原始文本，避免三维细化器受到低质量模态干扰。

## 7.5 残差预测

让细化器预测 logit 残差：

$$
 l_j^{final}=l_j^{base}+g_j\Delta l_j
$$

其中 `g_j` 为可学习门控，并初始化为较小值，使模型从 GEAL 基线稳定开始训练。

## 7.6 损失函数

$$
\mathcal L_{seg}
=
\mathcal L_{BCE}
+\mathcal L_{Dice}
+\lambda_f\mathcal L_{Focal}
+\lambda_b\mathcal L_{boundary}
$$

可从以下初始权重开始搜索：

```text
BCE      0.3
Dice     0.3
Focal    0.2
Boundary 0.2
```

这些权重是实验起点，不是已经验证的最终配置。

## 7.7 必做对照

- 原始 GEAL；
- 等参数普通 MLP；
- 单尺度 KNN；
- 多尺度 KNN；
- Point Transformer；
- 直接替换预测；
- 残差修正；
- 硬候选裁剪；
- 软候选采样。

## 7.8 晋级门槛

- Boundary F-score 与严格 IoU 提升；
- 小区域召回不下降；
- AP50/mAP 改善；
- 宽松覆盖指标没有灾难性下降；
- 收益不能由参数量增加完全解释。

---

# 8. V4：Visibility-aware Token-level CAM

## 8.1 目标

将 GEAL 的连续特征级 CAM 升级为局部结构级二维—三维对齐。

## 8.2 三维局部 token

利用 V3 的局部邻域生成：

$$
 z_i^{3D}=\operatorname{Pool}(\phi(p_j-p_i,F_j))
$$

每个 token 必须保留：

- token 中心点；
- 邻居点索引；
- token-to-point 映射；
- 各视角可见性。

## 8.3 二维 token

根据 3DGS 投影，在 DINOv2 特征图中提取对应区域：

$$
 z_{iv}^{2D}=\operatorname{ROIAlign}(F_v^{2D},\Pi_v(\mathcal N_i))
$$

再利用可见性对多视角二维 token 做加权聚合。

## 8.4 对齐损失

保留 GEAL 原有特征一致性：

$$
\mathcal L_{feature}
=
\|F^{3D\rightarrow2D}-F^{2D}\|_2^2
$$

新增 token 对比一致性：

$$
\mathcal L_{token}
=-\log
\frac{\exp(\operatorname{sim}(z_i^{3D},z_i^{2D})/\tau)}
{\sum_j\exp(\operatorname{sim}(z_i^{3D},z_j^{2D})/\tau)}
$$

总损失：

$$
\mathcal L
=
\mathcal L_{seg}
+\lambda_c\mathcal L_{feature}
+\lambda_t\mathcal L_{token}
$$

第一版使用 batch 内负样本即可，暂不引入复杂 optimal transport。

## 8.5 Hard negative

优先选择：

- 同一物体上的相似非功能部件；
- 把手与支撑杆；
- 座面与桌面；
- 容器边缘与容器内部；
- corruption 新增的局部噪声簇。

## 8.6 必做消融

| 实验 | Feature CAM | Token CAM | 可见性 |
|---|---:|---:|---:|
| C0 | × | × | × |
| C1 | ✓ | × | ✓ |
| C2 | × | ✓ | ✓ |
| C3 | ✓ | ✓ | × |
| C4 | ✓ | ✓ | ✓ |

## 8.7 成功判断

V4 应重点改善：

- Unseen；
- 小型功能部件；
- Add Local；
- 相似局部结构误检；
- 跨视角预测一致性。

若只提升 Seen 而损害 Unseen，应判断 token 对齐是否过拟合训练类别。

---

# 9. V5：可选的 Clean–Corrupt 一致性训练

## 9.1 使用条件

V5 只有在 V0–V4 主线成立后才加入，不应与 V3、V4 同时首次上线。

## 9.2 一致性目标

对同一样本生成：

$$
P_{clean},P_{corrupt}
$$

在存在对应关系的点上约束：

$$
\mathcal L_{rob}
=D(\hat y_{clean},T^{-1}(\hat y_{corrupt}))
$$

## 9.3 按 corruption 类型处理

| Corruption | 一致性策略 |
|---|---|
| Jitter | 相同点或最近邻点匹配 |
| Rotate | 将预测旋回原坐标系 |
| Drop Global | 只约束仍保留点 |
| Drop Local | 只约束可见点，不要求恢复被删除点 |
| Add Local | 新增点作为 hard negative |
| Add Global | 约束新增背景点低响应 |

## 9.4 实验协议

必须分开报告：

1. clean-trained → clean；
2. clean-trained → corrupt；
3. corruption-augmented training → corrupt；
4. unseen + corrupt。

否则无法区分天然鲁棒性和增强训练收益。

---

# 10. 实验总矩阵

| 维度 | 设置 |
|---|---|
| 模型 | GEAL / V1 / V2 / V3 / V4 / V5 |
| 数据 | PIAD / LASO |
| 泛化 | Seen / Unseen |
| 区域尺度 | 小 / 中 / 大 |
| 输入质量 | Clean / 单一 corruption / 组合 corruption |
| 几何细化 | 无 / 单尺度 / 多尺度 |
| 二维先验 | 无 / 均值 / 可见性 / 可见性+不确定性 |
| 对齐 | MSE / Token contrast / 联合 |
| 稳定性 | 至少 3 个随机种子 |
| 效率 | 参数量 / FLOPs / 显存 / 推理时间 |

每个新版本都必须与前一版本和原始 GEAL 同时比较。

---

# 11. 统一晋级与停止规则

一个版本只有同时满足以下条件，才进入下一阶段：

1. Seen 性能没有显著下降；
2. 至少一套数据的 Unseen 稳定提升；
3. 小区域或边界指标有明确改善；
4. 三个随机种子方向一致；
5. 等参数对照不能解释全部收益；
6. 新模块改善了其设计目标对应的指标；
7. 推理开销仍在可接受范围。

如果阶段失败，采用以下回退策略：

| 失败阶段 | 回退方案 |
|---|---|
| V1 | 降低功能部件权重，保留完整任务语义 |
| V2 | 仅将二维先验用于辅助损失，不输入三维网络 |
| V3 | 改用轻量 edge-aware MLP |
| V4 | 只在高置信、可见 token 上计算对比损失 |
| V5 | 仅用于测试，不参与训练 |

---

# 12. 最终建议

## 最低可行版本

```text
V0 + V2 + V3
```

即：

- 完整基线诊断；
- 3DGS 反投影二维软先验；
- PointNet++ 并联局部几何残差细化。

该版本最容易验证 TASA 的核心思想是否适合 GEAL。

## 推荐论文主模型

```text
V0 + V1 + V2 + V3 + V4
```

建议论文叙事为：

> 通过任务感知视角加权构建可靠的二维软先验，再利用局部几何残差细化和 visibility-aware token consistency，将二维教师的功能部件语义精确迁移到三维点级预测中。

## 预期贡献

1. 任务感知的二维—三维不对称蒸馏；
2. 基于 3DGS 可见性的二维软三维先验；
3. 保留全局分支的局部几何残差细化；
4. visibility-aware token-level 2D–3D consistency；
5. 面向小区域、Unseen 和局部 corruption 的细粒度评测。

---

## 13. 第一轮实现清单

### 代码模块

- [ ] `task_parser`：动作、对象、功能部件抽取；
- [ ] `view_scorer`：完整任务与部件加权；
- [ ] `soft_prior_backprojector`：3DGS 二维先验反投影；
- [ ] `visibility_estimator`：跨视角可见性与不确定性；
- [ ] `local_refiner`：KNN / Point Transformer 局部残差模块；
- [ ] `token_projector`：二维、三维局部 token 构造与匹配；
- [ ] `boundary_loss`：边界监督；
- [ ] `region_metrics`：小区域和边界指标；
- [ ] `corruption_consistency`：V5 鲁棒一致性训练。

### 实验模块

- [ ] GEAL 完整基线复现；
- [ ] GAFM/CAM 2×2 因子消融；
- [ ] 任务感知视角消融；
- [ ] 软先验与可见性消融；
- [ ] 局部细化器结构消融；
- [ ] Token CAM 对齐消融；
- [ ] clean/corrupt 分离评测；
- [ ] 三随机种子统计；
- [ ] 参数量、FLOPs、显存、推理耗时统计。

---

## 14. 研究结论

TASA 对 GEAL 的最核心启发是：

> **不要让二维模型直接负责最终三维边界；让二维模型负责高召回语义候选，让三维模型负责局部几何确认。**

因此最稳妥的 GEAL 改造不是添加更多模态，而是建立更加清晰的非对称分工：

```text
二维教师：找什么、可能在哪里
三维主干：保持全局泛化
局部细化器：精确边界和空间连贯性
3DGS：提供可追踪的跨模态对应
```

所有公式、权重和模块配置均属于本研究计划中的待验证设计，不应表述为 TASA 或 GEAL 已经证明的结论。

---

# 15. V2 代码级改造方案（基于 geal_test / TASA-main 实测）

> 本节把第 6 节的 V2（二维软先验反投影）落到当前代码库的真实接口上。
> 所有改动点都标注了 `geal_test/` 与 `TASA-main/` 中的现有文件、类与函数，
> 以便直接进入实现。结论仍是**待验证设计**，不等于已验证结果。

## 15.1 现有代码就绪度盘点

| 计划要素 | 代码现状（证据） | 结论 |
|---|---|---|
| 二维逐视角 affordance 概率图 `ŷ²ᴰ` | `Branch2D.forward` 仅在 **stage1** 路径产出 `attn_map`（`branch_2d.py:181-194`，shape `[B*n_view,1,H,W]`）；**stage2** 路径只返回 `fused_features, render_feats`（`branch_2d.py:196-200`），**不含概率图** | V2 必须先让 2D 分支在 stage2 也吐出 `attn_map` |
| 3DGS 可见性 `a_jv` | `Gaussian_Renderer.__call__` 本就返回 `render_idx [12,H,W]` 与 `rendered_contrib [12,H,W]`（`gaussian_render.py:136-140`）；但 `Branch2D._render_views` 把它们**丢弃了**，只回传 `render_tensor, mask_tensor, feat_tensor`（`branch_2d.py:248-257`） | 可见性信号已存在，只是没接到上层 |
| 三维分支接收二维先验 | `Branch3D.forward(text, xyz)` 签名**不含任何先验输入**；局部 tokenizer（V3/V4）已通过 `use_local_tokenizer` 接好（`branch_3d.py:53-80,193-202`） | V2 需要**新增**一个注入点，与 tokenizer 并联 |
| 训练主循环接线 | `train_stage2.py:208-209` 调用 `model_3d(question, point)` 与 `model_2d(question, point, feat_3d)`，3D 分支从不接收 2D 先验 | 主循环需插入 backprojector |
| 任务感知视角权重 `c_v`（来自 V1） | 全仓库**无任何** `view_scorer` / `task_parser`（grep 为空） | V2 先用**均匀** `c_v`，V1 落地后再替换，二者解耦 |
| 局部几何细化（V3/V4） | `Local3DTokenizer` + `TokenToPointInterpolator` + `TokenFusion(gated_residual, init=0.0)` 已实现（`local_3d_tokenizer.py`），由 `train_stage2_v1_tokenizer.yaml` 的 `local_tokenizer.enabled` 开关控制 | **V3/V4 已就绪**，V2 的 `q_j` 应作为与其并联的全局软候选 |
| TASA 的 2D→3D lift 参考 | `TASA-main/pipeline/step7_lift_3d/molmo_lift_2d_to_3d.py` 用 `PointCloudToImageMapper.compute_multi_masked_mapping(pose, pcd, mask, depth, intrinsic)` 把 2D mask 抬到 3D；`TASA-main/models/scene_models/pointtransformer.py` 的 `PointTransformerLayer` 提供 KNN+相对位置编码的局部几何 | GEAL 用 3DGS 的 `render_idx` 做等价反投影；局部细化可直接复用已就位的 tokenizer |

**核心结论**：V2 的“原材料”（`attn_map`、`render_idx`、`rendered_contrib`）在代码里**都已存在**，缺的是三处接线——① stage2 路径输出 `attn_map`；② 把 `render_idx/rendered_contrib` 透传出来；③ 在 `Branch3D` 加一个先验注入点。

## 15.2 改动原因

1. **当前 2D→3D 通道太“弱”且只在特征层**：stage2 唯一用到的 2D→3D 监督是
   `loss_kld = MSELoss(render_feats, feat_2d)`（`train_stage2.py:211,185`）——
   它对齐的是**特征嵌入**而非**空间概率**，且权重 `kl_loss_weight` 很小（yaml 默认 0.1）。
   2D 教师最有价值的高召回 affordance *位置*信息从未进过 3D 分支。

2. **小区域 / 边界是已诊断的瓶颈**：V0 诊断（§4.1）指出 PointNet++ 局部边界能力不足、
   渲染分辨率导致小区域信息消失。把 2D 软先验直接喂给 3D，等于给“去哪看”一个高召回提示，
   正好针对该瓶颈。

3. **可见性信号是“免费”的**：3DGS 渲染天然产出每个像素由哪个点贡献
   （`rendered_contrib`）和该点索引（`render_idx`），与 CAM 用的是同一套 splatting。
   复用它做反投影，零额外渲染成本，且天然规避了“被遮挡点的伪先验”。

4. **与 V3/V4 形成互补分工**：tokenizer（已就位）负责“局部几何确认”，V2 负责“全局软候选”，
   二者并联，符合 §3 的非对称分工原则。

## 15.3 改动原理

对每个三维点 `p_j`，在第 `v` 视角：

- 用 `render_idx` 找到 `p_j` 投影到的像素，取出该处 2D 概率 `ŷ²ᴰ_jv`；
- 用 `rendered_contrib` 作可见性权重 `a_jv`（被遮挡点 `a_jv≈0`，自然被剔除）；
- 用视角权重 `c_v`（V2 阶段均匀 `=1/V`，V1 后替换为任务感知权重）；
- 跨视角聚合得软先验 `q_j` 与不确定性 `u_j`：

$$
q_j=\frac{\sum_v a_{jv}c_v\hat y^{2D}_{jv}}{\sum_v a_{jv}c_v+\epsilon},
\qquad
u_j=\operatorname{Var}_v(\hat y^{2D}_{jv})
$$

再注入三维特征（与 tokenizer 并联，残差加和）：

$$
F'_j=\operatorname{MLP}([F^{3D}_j,q_j,u_j])
$$

**为什么用 `render_idx` 反投影而不是重新投影**：它与 CAM 共用同一渲染管线，
保证 2D→3D 对应关系和特征蒸馏严格一致；且 `render_idx` 的 winner-take-all 语义
天然实现了“遮挡点不可见”，无需另写深度测试。

## 15.4 改造方案（文件 / 接口 / 伪代码）

### 改动 1 — `Branch2D` 暴露 `attn_map`（stage2）

`branch_2d.py` 的 stage2 分支新增 `return_affordance_map` 开关：

```python
# branch_2d.py, forward(), stage2 分支
if self.stage1:
    ...
    return attn_map
else:
    dense_feat_map = cross_modal_feat.transpose(1,2).reshape(Bn,-1,H//14,W//14)
    fused_features = self.feature_upsampler(dense_feat_map)
    # —— 新增：与 stage1 共用同一套 decoder+upsample 产出概率图 ——
    attn_map = None
    if return_affordance_map:
        text_feat = self.decoder(text_embeds, cross_modal_feat,
                                 tgt_key_padding_mask=text_mask, query_pos=self.pos1d)
        text_feat *= text_mask.unsqueeze(-1).float()
        attn = torch.einsum('blc,bcn->bln', text_feat, fused_feat.transpose(1,2))
        attn = attn.sum(1)/text_mask.float().sum(1).unsqueeze(-1)
        attn_map = attn.reshape(Bn,-1,H//14,W//14)
        attn_map = self.learnable_upsample(torch.cat([attn_map, cls_token], dim=1))
        attn_map = torch.sigmoid(attn_map)
    return fused_features, render_feats, attn_map
```

> 注意：`attn_map` 走的是**冻结** 2D 教师权重，所以作为先验是稳定的，不会和 3D 训练互相污染。

### 改动 2 — `Branch2D._render_views` 透传 `render_idx / rendered_contrib`

```python
# branch_2d.py, _render_views()
render_tensor = torch.stack(rendered_images)       # [B, V, 3, H, W]
mask_tensor   = torch.stack(masks)
feat_tensor   = torch.stack(feats)
idx_tensor    = torch.stack(render_idx_list)        # [B, V, H, W]  ← 新增
contrib_tensor= torch.stack(rendered_contrib_list)  # [B, V, H, W]  ← 新增
...
return render_tensor, mask_tensor, feat_tensor, idx_tensor, contrib_tensor
```

并让 `forward` 在 stage2 把这两张量一并返回（供 backprojector 使用）。

### 改动 3 — 新增 `geal_test/model/soft_prior_backprojector.py`

```python
class SoftPriorBackprojector(nn.Module):
    """
    输入: attn_map [B*V,1,H,W], render_idx [B,V,H,W](long), contrib [B,V,H,W]
    输出: q [B,N], u [B,N]   （N 为点数，按点序对齐 Branch3D 输入 xyz）
    用 gather 而非 dense scatter，避免 O(V*H*W*N) 显存。
    """
    def forward(self, attn_map, render_idx, contrib, c_v=None):
        B, V, H, W = render_idx.shape
        attn = attn_map.view(B, V, H*W)                 # [B,V,P]
        idx  = render_idx.view(B, V, H*W)               # [B,V,P]
        a    = contrib.view(B, V, H*W).clamp_min(0)     # 可见性 a_jv
        if c_v is None:
            c_v = attn.new_ones(V) / V                  # V2 阶段：均匀视角权重
        # 把每个像素的 2D 概率按 render_idx 放到点序上
        gathered = attn.gather(2, idx.clamp_min(0))     # [B,V,P] 伪；实际用 index_select
        # 更稳妥：逐样本 index_select + 加权求和
        q = torch.zeros(B, N, device=attn.device)
        wsum = torch.zeros(B, N, device=attn.device)
        for v in range(V):
            y = torch.index_select(gathered[:,v], 1, idx[:,v].view(-1))  # 示意
            q += (a[:,v]*c_v[v]) * y_per_point
            wsum += a[:,v]*c_v[v]
        q = q / (wsum + 1e-6)
        u = ((gathered - q.unsqueeze(1))**2 * a.unsqueeze(-1)).sum(1) / (wsum+1e-6)
        return q, u
```

> 实现细节：用 `torch.gather` / `index_select` 按 `render_idx` 把像素值映射到点序，
> 不要用 `scatter_add` 在 `[B,N,H,W]` 上 densify，否则显存爆炸。
> 点数 `N` 取 `xyz.shape[1]`（与 `Branch3D` 输入严格一致）。

### 改动 4 — `Branch3D` 新增先验注入点

```python
# branch_3d.py, __init__
sp_cfg = cfg.get("soft_prior", {})
self.use_soft_prior = sp_cfg.get("enabled", False)
if self.use_soft_prior:
    self.soft_prior_mlp = nn.Sequential(
        nn.Conv1d(self.emb_dim + 2, self.emb_dim, 1),
        nn.BatchNorm1d(self.emb_dim), nn.GELU(),
        nn.Conv1d(self.emb_dim, self.emb_dim, 1),
    )
    self.soft_prior_gate = nn.Parameter(torch.tensor(0.0))  # 近零初始化，新增先验不破坏基线

# branch_3d.py, forward 签名：def forward(self, text, xyz, soft_prior=None)
# 在 Step 5b（tokenizer 融合之后）插入：
if self.use_soft_prior and soft_prior is not None:
    q, u = soft_prior[:,0:1], soft_prior[:,1:2]            # [B,1,N]
    prior_in = torch.cat([fused_feat, q, u], dim=1)        # [B, emb_dim+2, N]
    fused_feat = fused_feat + self.soft_prior_gate * self.soft_prior_mlp(prior_in)
```

> **关键设计**：门控 `soft_prior_gate` 以 0 初始化（与 `TokenFusion` 的 `residual_scale=0.0` 一致）。
> 这样训练初期 V2 几乎不改动基线输出，先验作为“渐进修正”融入，归因更干净，
> 也满足 §3 的“软信息、不硬筛选”约束。

### 改动 5 — `train_stage2.py` 主循环接线

```python
# train_one_epoch() 内，baseline 路径：
pred_3d, feat_3d = model_3d(question, point)
feat_2d, render_feats, attn_map, idx, contrib = model_2d(question, point, feat_3d,
                                                          return_affordance_map=True)
if use_soft_prior:
    q, u = soft_prior_backprojector(attn_map, idx, contrib)   # 2D 教师冻结，先验稳定
    pred_3d, feat_3d = model_3d(question, point, soft_prior=torch.cat([q,u],1))
# 原有 loss_hm + kl_loss_weight*loss_kld 保持不变
```

### 改动 6 — 配置与消融

新增 `geal_test/config/train_stage2_v2_softprior.yaml`，沿用 `train_stage2_v1_tokenizer.yaml`
的开关风格，新增：

```yaml
model_3d:
  soft_prior:
    enabled: true
  # 消融用：prior_mode 控制 q 是否含可见性/不确定性
  # P0 关闭 soft_prior；P1 仅均值；P2 均值+可见性；P3 可见性+不确定性
```

消融矩阵（对应 §6.4）：

| 版本 | `soft_prior.enabled` | 可见性 `a_jv` | 不确定性 `u_j` |
|---|---:|---:|---:|
| P0 | × | × | × |
| P1 | ✓（均匀 `c_v`） | × | × |
| P2 | ✓ | ✓ | × |
| P3 | ✓ | ✓ | ✓ |

### 改动 7 — 与 §9 轻量管线区分（避免重复计功）

`utils/affordance_loss.py` 的 `loss_3d2img` / `loss_contrastive` 也做 2D→3D 对齐，
但它是**基于真实交互图的不变性损失**（需要 `k_images`、RGB 图），且只作 loss。
V2 是**把渲染得到的 2D 概率作为特征先验注入 3D 主干**，二者机制不同、可叠加，
但论文里必须分开报告，不能把 `loss_3d2img` 的提升算到 V2 头上。

## 15.5 风险点

1. **训练/推理分布偏移（最高风险）**：V2 训练时 3D 分支吃了 2D 先验，但 GEAL 的部署优势是
   “推理只跑 3D 分支”（§2.1）。推理时没有 2D 分支 → 拿不到 `q_j`。
   **缓解**：门控近零初始化 + 推理时 `soft_prior=None` 走回退路径；若实证显示回退掉点严重，
   再考虑推理期跑一次轻量 2D 渲染（代价是失去纯 3D 部署优势，属 V2 回退策略）。

2. **`render_idx` 的 winner-take-all 语义**：每像素只记“贡献最大的点”，
   被遮挡点在所有视角 `a_jv≈0`，`q_j` 未定义（回退为 0）。
   这恰好逼 3D 全局分支自己找回遮挡目标——符合设计约束，但需在消融里显式看
   “遮挡点 / 背面点”子集的召回，确认未塌缩。

3. **点序一致性**：`render_idx` 的索引基于喂给渲染器的 `xyz` 顺序；
   `Branch3D` 内部对 `xyz` 做了 `/0.5`（`branch_3d.py:157`）但**不改顺序**。
   必须保证 `model_2d` 与 `model_3d` 收到的是**同一份 `point`**（主循环已满足），
   且 backprojector 输出的 `q` 维度 `[B,N]` 中 `N == xyz.shape[1]`。

4. **2D 教师天花板**：`attn_map` 来自冻结的 stage1 权重，先验质量上限 = stage1 小区域能力。
   若 V0 诊断确认 stage1 在 small bin 很弱，则 `q_j` 噪声大——此时应先修 V1（任务感知视角）
   或先上 §9 不变性损失，而不是只堆 V2。

5. **显存/实现坑**：`attn_map` 为 `[B*12,1,112,112]`，逐样本 `index_select` 即可，
   **禁止**构造 `[B,N,H,W]` dense 张量；`render_idx` 含 `-1`/填充值时需 `clamp_min(0)` 后再 gather。

6. **与 V3 tokenizer 的耦合**：tokenizer 已就位且同样注入 `fused_feat`。
   V2 与 V3 都改 `fused_feat` 时，建议 V2 在 tokenizer **之前**注入（先给全局软候选，
   再让局部几何细化），避免两条残差路径互相掩盖；消融里单独给 `V2 only / V3 only / V2+V3`。

## 15.6 与 V1/V3 的衔接与验收

- **V1（视角权重 `c_v`）**：V2 代码已预留 `c_v` 入参，默认均匀。V1 落地后只需把
  `view_scorer` 输出的 `c_v` 传进 `SoftPriorBackprojector`，**无需改 V2 主干**。
- **V3/V4（tokenizer）**：已实现，V2 与之并联、互补。最小可行版 `V0+V2+V3` 中，
  V2 用均匀 `c_v` 即可启动，不阻塞。
- **晋级门槛**（沿用 §6.5，并补两条）：
  - 严格区域（aIoU small bin）改善；
  - 2D 漏检样本上 3D 召回不灾难性下降；
  - **遮挡/背面点子集**召回不塌缩（对应风险 2）；
  - 推理期 `soft_prior=None` 回退时性能跌幅在可接受范围（对应风险 1）。

---

## 16. 第一轮实现清单（V2 增量）

- [ ] `Branch2D.forward` stage2 增加 `return_affordance_map` 输出 `attn_map`
- [ ] `Branch2D._render_views` 透传 `render_idx` / `rendered_contrib`
- [ ] `model/soft_prior_backprojector.py`：`SoftPriorBackprojector`（gather 实现）
- [ ] `Branch3D`：`soft_prior` 注入点 + 近零门控
- [ ] `train_stage2.py`：主循环接入 backprojector（冻结 2D 教师）
- [ ] `config/train_stage2_v2_softprior.yaml` + P0–P3 消融开关
- [ ] 评测脚本增加“遮挡/背面点子集”召回维度
- [ ] 与 §9 轻量损失分开报告，避免重复计功
