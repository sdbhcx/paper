---
title: "GEAL + VGGT 前馈单目重建融合方案（查询机制延后）"
aliases:
  - 前馈重建融合方案
  - VGGT Privileged-Hint Integration
tags:
  - affordance-grounding
  - feed-forward-reconstruction
  - privileged-hint
  - knowledge-distillation
  - research-plan
status: proposed
created: 2026-08-14
revised: 2026-08-15
base_model: "GEAL: Generalizable 3D Affordance Learning with Cross-Modal Consistency (CVPR 2025)"
integration_source: "QueryMe: Query-Driven Open-Vocabulary 3D Object Affordances Grounding from Multimodal Evidence (CVPR 2026)"
reconstruction_backbone: "VGGT: Visual Geometry Grounded Transformer (CVPR 2025 Best Paper)"
scope_note: "v2 修订：按用户决策暂不串入 QueryMe 查询机制（FPS 查询 + T→H→P 解码器），改为「VGGT 特权提示 + 轻量融合进 GEAL 原有解码器 + 推理期丢弃 HOI」。查询机制保留为 §15 延后附录。"
---

# GEAL + VGGT 前馈单目重建融合方案（查询机制延后）

## 0. 修订说明（v1 → v2）

| 项 | v1（2026-08-14） | v2（2026-08-15） |
|---|---|---|
| 解码器 | **替换**为 QueryMe 查询机制（FPS + T→H\*→P 三层交叉注意力） | **保留** GEAL 原有解码器（text-as-query 动态核） |
| HOI 先验接入 | 作为查询机制的 Step-2 输入 | 作为一层轻量 `HOIFusion` 交叉注意力，注入物体点云特征 P~ |
| 推理期 HOI | 原型库检索 φ_A 替代 H\*（零图像但需原型） | **直接丢弃 H\***（置空 token），纯 3D 推理，零原型库 |
| Unseen Affordance 风险 R1 | 高（无 φ_A） | **消除**（推理不依赖任何 HOI/原型） |
| 改动规模 | 高（核心改动解码器） | 低（新增一个融合模块 + 蒸馏训练策略） |
| 潜在收益 | 高（HOI 推理期仍被消费） | 中-高（HOI 仅训练期正则化，推理退回纯 3D） |

**v2 的核心论点**：把 QueryMe 的 VGGT 重建不看作「推理期多模态输入」，而看作**训练期特权提示（privileged hint）**——冻结 VGGT 提供 HOI 交互几何，仅在训练时作为额外监督/特征提示注入 GEAL 标准解码器；推理时丢弃该提示，模型以纯 GEAL 形式部署。这兼得「训练期吃 HOI 先验」和「推理期纯 3D 部署」，且因推理零 HOI 而从根本规避 unseen affordance 的原型缺失风险。

### 0.1 二次修订（数据集事实修正, 2026-08-15）

用户确认 **PIAD 中 HOI 图像与点云是同物体数据**，且 HOI 图按「物体-affordance」文件夹分类、点云仅按「物体」分类。据此修正原 v2 草稿中「HOI 图与点云不配对、坐标系不可对齐」的假设（原 §3.3.2 / §5.4）：

- 同物体 ⇒ 存在语义对应 ⇒ 曾计划「注册对齐 → position-aware + content 双路融合」（§3.3.2、§4.2 原始设计）；**后于 §0.2 修订为：主线纯内容检索、位置感知+注册降级为备选**（因强行对齐可能损坏 3D HOI 几何，且增益未经证实，QueryMe 纯内容检索已能达 74 AUC）。
- H\* 计算由「离线 hint_pool 预计算」改为「训练时在线生成」（AnchorEncoder 可学习），VGGT 重建仅离线缓存全量 H_raw（§3.5、§5.4）；
- 数据集名统一为 **PIAD**（非 PIADv2）。
- **实证人证**（读 `Point_Train_Bag_158.txt`，脚本 `outputs/analyze_piad.py`）：点云单文件=单实例 2048 点、含逐点多通道 affordance 真值（Bag 实测 3 通道）、坐标逐实例归一化且朝向类别规范。据此把 §3.3.2 注册改为「H* 与 P 均归一到物体类别规范帧 O_can」，并新增 §3.3.3 数据格式实证。
- **HOI 图像侧实证**（读 `Img_Train_Bag_contain_205/206.json`，脚本 `outputs/analyze_hoi_json.py`）：HOI 图是真实人-物照片（LabelMe 标注，含 `object`/`subject` 两个 bbox）。据此新增 §3.2 输入裁剪策略（以 object bbox 外扩 1.2~1.5× 喂 VGGT）、新增 §3.3.4 数据格式实证、并把 bbox crop 接入 §3.5 预计算脚本。
- **M0 验证通过（2026-08-16 真机实测）**：独立文档 `VGGT_HOI_raw_pipeline.md` §5 记录。VGGT 单图前向跑通（<0.1s），H_raw 落盘 `.npz` 三字段完整无 NaN/Inf；**`crop_scale=1.4` 确认最优**（scale=1.0 丢失手部，Z 跨度仅 0.190；1.4 扩到 0.855、4.5×，多出 0.66 深度即手部/接触区）；conf 可分层（物体主体 ≥1.5 / 交互区 1.0~1.5 / 纯噪声 <1.0 占 0%）→ 该信号已写入 §3.2 与 §3.3 第 2 步作交互锚点初筛先验。raw 管线解锁，可进入 §3.3 自适应空间注意力（纯内容检索）。

推理期仍零 HOI、零原型库，Unseen Affordance 风险 R1 仍消除。

### 0.2 三次修订（主线定纯内容检索，位置感知降级备选, 2026-08-16）

用户决策：位置感知融合 QueryMe 并未采用（其在 HOI↔物体 一步也是纯内容检索），且 3D HOI 点云与物体点云不在同一坐标系，强行注册/对齐可能**损坏 3D HOI 的交互几何**（跨实例形状强制、Sim3 尺度恢复失败、前置分割误差）。因此：

- **主线改为纯内容检索**：H* ∈ R^{k×d} 仅作内容检索的 key/value，HOIFusion 为纯内容交叉注意力（§3.3.1、§4.2），不读任何 xyz、不做位置偏置、不引入注册；
- **位置感知 + 注册降级为〔备选〕**（§3.3.2）：仅当 §9.4 消融证明「位置感知+注册 显著优于 纯内容」时才启用，且启用时以「低置信退化为纯内容」兜底；
- **纯内容检索的 GEAL 完整做法**：H_raw（离线缓存）→ 在线经自适应空间注意力 + conf 初筛 → top-k 锚点 → AnchorEncoder（可学习）→ H\*（k×d）；训练期 P~ 对 H* 做内容交叉注意力得 P~_aug 喂 GEAL 解码器，同时跑无提示路径并以 Hint KL 对齐；推理期 H\*=null 退回纯 3D 部署（见 §3.3 / §4.2 / §5.3）。

---

## 1. 方案定位与核心矛盾

### 1.1 目标（v2 收敛）

在「不改 GEAL 解码器、推理期保持纯 3D」的约束下，把 QueryMe（CVPR 2026）的前馈单目重建（VGGT）作为 HOI 交互先验引入 GEAL 训练，提升 unseen affordance 与开放词表泛化。

### 1.2 核心矛盾（与 v1 同，解法不同）

| 维度 | GEAL | QueryMe | 本方案解法（v2） |
|---|---|---|---|
| 推理模态 | 纯 3D 分支（零图像） | 需 HOI 图 + VGGT 重建 | **保持纯 3D**，VGGT 仅训练期用 |
| 2D→3D 通道 | 3DGS 渲染 + DINOv2 蒸馏 | VGGT 直接重建 3D HOI 点云 | VGGT 另立 HOI 通道，仅训练期 |
| 解码器 | text-as-query + 动态核 | FPS 查询 + T→H→P 注意力 | **保留 GEAL 解码器**，HOI 经融合模块注入 |
| 数据集 | PIAD / LASO（无 HOI 图） | PIAD（HOI 图与点云同物体：HOI 按 物体-affordance，点云按 物体） | 训练期按 (物体,affordance) 同物体配对采样 H\*，推理期零 HOI |

### 1.3 解决思路（v2）

**VGGT 作为特权提示（privileged information / hint training）**：

- 训练期：冻结 VGGT 重建 HOI 图 → 3D HOI 点云 H_raw → 自适应空间注意力 → HOI 特征 H\*；一层 `HOIFusion`（物体点云特征 P~ 对 H\* 做交叉注意力）把交互先验注入 P~，再送 GEAL 原有解码器 → ω_full。
- 推理期：不提供 H\*（用可学习 null token 或置零），`HOIFusion` 退化为恒等/弱投影，解码器以纯 P~ + T 运行 → ω_3D，**零图像、零 VGGT、零原型库**。

等价视角：VGGT 是「训练期教师给的几何提示」，通过 hint + 蒸馏（KL 逼 ω_3D 复现 ω_full）把收益内化进 GEAL 解码器权重，推理时自然消失。

### 1.4 与既有可行性分析的关系

本方案是对 `Feasibility_QueryMe` §7.1「弃用 H'，改用 G'」的**保留修正 + 延后**：

- §7.3b 指出弃用 H' 后 Unseen Affordance 空间交互先验无法完全内化（经验问题）；
- v1 用「保留 H' 训练 + 原型库推理」解决，但带来 R1 原型缺失风险；
- **v2 进一步：推理期彻底不用 H'，HOI 仅在训练期作为特权提示**。Unseen Affordance 不会比 GEAL 基线更差（推理路径就是 GEAL），且训练期 HOI 提示仍能正则化特征，预期在 seen/unseen-obj/unseen-aff 上均有增益；
- 若后续需要「推理期也吃 HOI」的高性能档，再启用 §15 的查询机制（策略 B）。

---

## 2. 总体架构

### 2.1 训练期架构（VGGT 特权提示 + 轻量融合）

```
输入: 文本 Q + 目标物体点云 P + HOI 交互图像 I (训练期专用)
                    │
     ┌──────────────┼──────────────────┐
     ▼              ▼                   ▼
 [GEAL 原有路径]   [新增: VGGT 路径]    文本编码
  P→3DGS(冻结)     I→VGGT(冻结)        Q→RoBERTa
  →多视角渲染       →深度图+相机参数     →T (双向交叉注意力)
  →DINOv2(冻结)    →反投影              →T~
  →GAFM            →H_raw(M×3)
  →F^2D            →自适应空间注意力
     │              →FPS + PointNet++   │
     │              →H* (k×d)           │
     ▼              │                   │
  P→PointNet++      │                   │
  →GAFM            │                   │
  →P~ (N×d) ───────┘ (cross-attn P~→H*) │
     │                                    │
     ▼                                    │
  [新增: HOIFusion]                        │
  P~_aug = P~ + CrossAttn(P~, H*, H*)      │
     │                                    │
     ▼                                    │
  [GEAL 解码器 不变]                       │
  text-as-query 动态核 × P~_aug → sigmoid  │
     → ω_full                              │
     ┌──────────────┴──────────────┐      │
     ▼                             ▼       │
 [GEAL CAM 不变]            [新增: Hint 蒸馏]│
 3D→Gaussian→渲染2D          L_hint = KL(ω_3D ‖ ω_full)
 MSE(F^2D, F^3D→2D)         ω_3D = 解码器(P~, T, H*=null)
 L_consis
```

### 2.2 推理期架构（纯 3D，零图像）

```
输入: 文本 Q + 物体点云 P (零图像, 零VGGT)
              │
     ┌────────┴────────┐
     ▼                 ▼
  P→PointNet++     Q→RoBERTa
  →GAFM            →T~
  →P~ (N×d)             │
     │                  │
     ▼                  │
  [HOIFusion, H*=null]  │
  P~_aug = P~ + CrossAttn(P~, null_token, null_token)
        ≈ P~ (退化)      │
     │                  │
     ▼                  │
  [GEAL 解码器 不变] ────┘
  → ω_3D
```

### 2.3 与 v1 三种策略的关系

| 策略 | v1 含义 | v2 状态 |
|---|---|---|
| A: 纯 3D 蒸馏 | 原型库检索 φ_A 替代 H\* | **→ v2 主线**：直接丢弃 H\*（null），无需原型库 |
| B: 全路径双模态 | 推理期也跑 VGGT | **延后（§15）**：仅在需要「with-HOI」高性能档时启用 |
| C: 混合可选 | 推理可选 HOI | 延后：等于 A + B 开关 |

v2 默认只有 A（纯 3D），B/C 作为可选项 future work，因此**部署优势完整保留、Unseen Affordance 风险归零**。

---

## 3. 前馈重建模块（VGGT）接入设计

> 本节与 v1 §3 完全一致，VGGT 接入方式不受查询机制取舍影响。

### 3.1 VGGT 概述

VGGT（CVPR 2025 Best Paper，Oxford VGG + Meta AI）是 1.2B 参数的前馈前馈 Transformer，单次前向即可从 1~数百张图像预测相机参数、深度图、点云图和 3D 点轨迹。关键特性：

- **单图零样本重建**：虽训练于多视角数据，单图输入仍产出连贯深度与 3D 结构；
- **输出格式**：深度图 D + 相机参数 g → 反投影得 3D 点云（比直接点云头更精确）；
- **特征骨干**：中间特征可作下游骨干；
- **推理速度**：单图 < 0.1s（H100）；
- **开源**：`github.com/facebookresearch/vggt`，权重 `facebook/VGGT-1B`。

### 3.2 VGGT 在本方案中的角色

```
HOI 图像 I
  → 读 JSON: object bbox B_obj, subject bbox B_subj
  → crop(I, B_obj 外扩 1.2~1.5×)      # 保留物体+手部/接触区，抑制背景
  → VGGT (冻结, fp16)
  → 深度图 D + 相机参数 g
  → unproject_depth_map_to_point_map(D, g)
  → 3D HOI 点云 H_raw (M×3, 含人+物)
```

VGGT **仅训练期使用、始终冻结、不参与梯度反传**，作用为提供 3D HOI 交互几何先验。

**输入裁剪策略**：PIAD 的 HOI 图是带人-物交互的真实照片（见 §3.3.4 实证）。JSON 中 `object` bbox 仅框出物体，`subject` bbox 框出整个人。若直接喂全图给 VGGT，背景与人体会引入大量无关点云；若只 crop `object` bbox，又会切掉手部/接触区，丢失 affordance 最强线索。因此推荐：
- 以 `object` bbox 为中心，外扩 **1.2~1.5 倍**（或取 object 与 subject 的合理交集 padding），得到 crop 区域；
- 该 crop 既保留物体主体，又大概率包含手/臂等交互部位，同时显著抑制远距离背景；
- 裁剪后按 VGGT 要求 resize 到 518×518。
- **M0 实测结论（2026-08-16）**：`crop_scale=1.4` 确认为最优——scale=1.0 时所有点挤在 Z=0.958~1.148 薄层（仅物体平面，手部全部丢失，Z 跨度仅 0.190），scale=1.4 时 Z 跨度扩到 0.855（4.5×）、多出 0.66 深度范围即手部/接触区，且 conf<1.0 纯噪声占 0%（见独立文档 `VGGT_HOI_raw_pipeline.md` §5）。**默认 1.4，区间 1.2~1.5 内无需再扫。**

> 消融项：可在 §9.4 增加「VGGT 输入：全图 vs object-crop vs crop+padding」。

### 3.3 HOI 重建输出处理（自适应空间注意力）

从 H_raw 提取交互锚点特征 H\*（供 `HOIFusion` 做 key/value）：

1. **全局采样**：FPS 得 P_s = {p_j}（N_s = ρ·M，ρ=0.3~0.5）；
2. **重要性预测**：MLP 编码坐标 → 1D 卷积建模空间连续性 → 分数 s_j。**M0 实测可用 VGGT 的 `conf` 作初筛先验**（见 `VGGT_HOI_raw_pipeline.md` §5）：conf≥1.5 多为物体主体、conf∈[1.0,1.5) 富集手部/接触区（交互线索最强），conf<1.0 纯噪声占 0%。故可把「低 conf 偏向交互区」的先验作为重要性初始化，引导锚点向交互区偏移，**无需额外手部检测模型**——这与 step 1–4 用坐标在 HOI 场景内部定位「手/接触区」的设计自洽。
3. **距离插值**：w_ij = 1/(‖p_i - p_j‖ + ε) → 全点云分数 ŝ_i；
4. **锚点选取**：取 top-k（k=128~256）高重要性采样点；
5. **特征编码（在线，可学习）**：共享权重的 `AnchorEncoder`（PointNet 式 MLP + max-pool）编码 top-k 锚点局部 patch → **H\* = F\* ∈ R^{k×d}**（纯特征 token，**不带 xyz**）。**`AnchorEncoder` 从零初始化、参与训练梯度回传**——affordance loss + Hint KL 经 `HOIFusion` → `AnchorEncoder` 反向传播，驱动编码从"通用几何描述子"进化为"任务相关交互特征"（类似 branch_3d 的 PointNet++ 末层可学习的道理，见第 5 节讨论）。主方案（纯内容检索）不引入任何坐标对齐/注册。

> **本步详细实现流程（含伪代码/超参/边界/接口）见独立文档 `outputs/GEAL_HOI_feature_pipeline.md`**——即 §3.5.1 的 `compute_hstar(H_raw, cfg)`：①conf 过滤 → ②FPS 全局 → ③conf 加权重要性 → ④top-k 锚点 → ⑤局部 patch(xyz+conf) → ⑥PointNet/PP++ 共享权重 → H\*∈R^{k×d}。

### 3.3.1 管线输出规格（关键，先读这条）

**主输出 `H*` 是一组交互特征 token，不带任何坐标**：

```
H* ∈ R^{k × d}            # k 个交互锚点，每点 d 维语义/几何特征（如 k=128, d=256）
φ_A ∈ R^{d}  (可选)       # 按 affordance 类别对 H* 做 mean-pool 得到的全局交互描述子
```

- **为什么是 token 集而非单向量**：单向量丢掉了「交互发生在物体的哪些相对部位」这一空间结构，而这是 affordance 定位最该利用的先验。k 个 token 让 `HOIFusion` 的 cross-attention 能按内容把「物体点 ↔ 交互锚点」一一关联。
- **为什么主方案不带 xyz（核心决策）**：3D HOI（H_raw，VGGT 世界系、up-to-scale）与目标点云 P（独立 3D 扫描/规范帧）**不在同一坐标系**；要把 H* 的 xyz 注册到 P 帧，需要 Sim3 尺度恢复 + 跨实例刚体配准，这**会引入几何失真并可能损坏 3D HOI 的相对交互结构**（详见 §3.3.2 备选论证）。因此主方案（纯内容检索）**只用 H* 的特征 token，完全不碰坐标**——cross-attention 按内容把物体点与 HOI 锚点关联，与 QueryMe 的 HOI↔物体步做法一致（见 §3.3.2 末对比）。坐标仅在步骤 1–4 的 *HOI 场景内部* 用于「按 conf 选交互锚点」，步骤 5 之后只保留特征。
- **缓存的就是 H_raw**：离线预计算（§3.5）仅落盘 H_raw（M×3 + conf）；训练时 dataset `__getitem__` 读取 H_raw，在线经自适应空间注意力 + `AnchorEncoder` 生成 H\*（k×d 纯特征），零 VGGT 前向。`AnchorEncoder` 是模型可学习组件，H\* 随训练动态更新（见 §3.5.1）。

**`HOIFusion` 实际消费的（主方案）**：`key=value=H*`（k×d 纯特征），`query=P~`（N×d）。仅有内容检索，无位置偏置。带位置感知的变体见 §3.3.2 备选。

### 3.3.2 〔备选〕坐标注册与位置感知融合（仅当消融证明有效时启用）

> **状态：备选方案（非主线）。主线 §3.3.1 已定为「纯内容检索、H\* 不带坐标」。本节保留技术细节，但默认不启用。**

**数据集事实（用户确认）**：PIAD 中 HOI 图像与点云是**同物体（同物体类别 O）**数据——HOI 图按「物体-affordance」文件夹分类（如 `mug-grasp`、`chair-sit`），点云按「物体」文件夹分类（如 `mug`、`chair`）。两者针对同一物体类别 O，存在语义对应关系（与「不同实例、完全不配对」不同）。

**为什么仍降级为备选——强行对齐可能损坏 3D HOI**：
即便同物体类别，把 H\* 的 xyz 注册到 P 帧仍是**跨采集、跨实例**的刚性变换（VGGT 无度量尺度，需 Sim3 尺度恢复；HOI 是实例 A、P 是实例 B），会从以下环节**扭曲甚至损坏 3D HOI 本该保留的交互几何**：
- **尺度失真**：VGGT 单图 up-to-scale，尺度由 FPFH/ICP 对应点估计；对应点若有噪声（VGGT 点本就带重建误差），尺度因子估计偏差会把整段 HOI 几何按比例拉伸/压缩，手-物相对距离失真。
- **跨实例形状强制**：注册是把「实例 A 的 HOI」刚性贴合到「实例 B 的点云」形态——实例 A 的真实物体形状被强行改形以匹配 B，而 affordance 最关心的「手如何接触物体」的相对结构随这个错误变换一起被带歪。
- **ICP 局部最优**：HOI 场景里「物体部分」占比小、点稀疏，粗配初始差即陷局部最优，后续精配只是把错误固化。
- **前置分割误差**：注册前需先分割出 HOI 里的物体点（去人/背景），VGGT + 人体分割一旦出错，喂给 ICP 的是垃圾对应 → 变换即垃圾。
- **⇒ 结论**：注册是一个「为换取位置偏置信号、却拿 3D HOI 几何当赌注」的操作。而位置偏置的增量收益**未经证实**（QueryMe 纯内容检索已能在 Unseen-Affordance 上达 74 AUC）。因此**默认不注册、不引入位置偏置**，只在 §9.4 消融证明「注册(位置感知) 显著优于 纯内容」时才启用——届时用「低置信退化为纯内容」作为兜底。

**〔备选〕注册方案（规范物体帧，decouple 预计算）**：
```
1. H_raw = VGGT(I)                      # 冻结前向 → 含人+物场景点云
2. seg    = person_segmenter(H_raw)     # 去人/背景，保留物体点 H_obj
3. O_can  = load_canonical(O)           # 物体类别 O 的规范帧模型（PIAD 同类点云已对齐）
4. T      = ICP(H_obj, O_can)           # 粗配 RANSAC/FPFH + 精配 ICP（含 Sim3 尺度恢复）
5. H_aligned = T · H_raw               # 锚点 xyz 落到 O_can / P 帧
6. H* = [PointNet++(H_aligned_topk); H_aligned_topk.xyz_norm]   # k×(d+3)
```
- 注册到「物体类别规范帧 O_can」（朝向规范、尺度/平移统一）而非具体某个 P；训练时 P 也需先对齐到 O_can（见 §3.3.3 实测：逐实例归一化、朝向规范）。
- 若注册置信度低（重叠过少 / FPFH 对应少），退化为仅用特征 token（k×d）做内容检索。

**与 QueryMe 原文的对比（已修正，此前此处有误）**：QueryMe 的 P 也是「给定的三维对象点云」——来自 PIADv2 的 **3DIR / 3D-AffordanceNet / Objaverse**（均为**物体级**扫描/模型，非场景级），而 I 是**另一来源**的 HOI 照片，经 VGGT 单独重建得到 H。即 QueryMe 的 H 与 P **同样来自不同采集、不同坐标系**，**并未**「VGGT 同帧共重建人+物到同一坐标」。QueryMe 处理这种不对齐的方式是**纯内容检索**：在 P 上初始化查询、用 P 自身的 MLP 位置编码建模物体几何，再以 cross-attention 让查询**按内容**检索 HOI 特征 token（H\* 的 xyz 不与 P 对齐，也不参与位置偏置）。
⇒ 本方案的**主线（纯内容检索）与 QueryMe 的 HOI↔物体步做法一致**——H\* 都是不带坐标的特征 token、靠内容检索关联。我们相对 QueryMe 的增量不在「位置感知」（那是备选、且风险如上），而在「HOI 仅作训练期特权提示 + 蒸馏后推理期零 HOI」（见 §4、§5、§11）。差异总结：(a) QueryMe 内容检索、我们内容检索（同思路）；(b) 我们 P 是独立高质量 3D 扫描（与 QueryMe 的 3DIR/Objaverse 同性质）；(c) 我们推理为纯 3D（零 HOI），QueryMe 推理仍需 HOI 图。

### 3.3.3 数据格式实证（`Point_Train_Bag_158.txt`）

用户提供的真实 PIAD 点云文件，逐行解析后的结构（脚本见 `outputs/analyze_piad.py`）：

```
文件名约定:  Point_Train_{Category}_{idx}.txt   （如 Bag_158）
每行 = 1 个点:
  [obj_id(32位hex)] [category] [x y z] [a1 a2 a3 ...] [0...0]
  数值列固定 20 列
本样本实测:
  - 总行数 = 2048 点（单物体实例）
  - 不同 obj_id = 1  → feeae9bd7f1336693a75e6d000e08290
  - 不同 category = 1 → Bag
  - col3-5 : xyz，已归一化（x∈[-0.355,0.353], y∈[-0.405,0.930], z∈[-0.203,0.188]）
  - col6+  : affordance 通道；Bag 样本 3 个通道非零
            idx3 全 2048 点非零（基础/主 affordance）
            idx4 非零 1701 点（≈83%，次 affordance）
            idx5 非零  347 点（≈17%，局部 affordance）
            其余通道全 0（padding / 未用）
```

**对设计的三个确认 / 约束**：

1. **点云内含逐点 affordance 真值（多通道 heatmap）** → 确认 §5.4「点云 affordance 由标注提供」成立；训练监督 y 直接可取，无需从 HOI 反推。这与用户「点云只有物体分类」（指**文件夹/组织层级**仅物体级）不矛盾：文件夹是 `Bag/`，但文件内部带逐点 affordance 标签。
2. **坐标逐实例归一化、朝向类别规范** → 仅〔备选〕位置感知融合路径（§3.3.2）需要时，才把 P 也归一到 O_can；主方案（纯内容检索）**不读 P 的 xyz 做对齐**，故该归一化细节对主线无影响。
3. **单实例单文件、类别在 col2、obj_id 在 col1** → 可用作 (物体 O) 索引与同物体配对键；HOI 图像侧按 `O-A` 文件夹组织，训练时以 O 为键配对。

> 注：本文件是**点云侧**样本；HOI 图像侧（按 物体-affordance 文件夹）未在此文件中，其组织方式以用户描述为准。

### 3.3.4 HOI 图像侧格式实证（`Img_Train_Bag_contain_205/206.json`）

用户提供的两个 PIAD HOI 图像及其 JSON 标注（脚本见 `outputs/analyze_hoi_json.py`），格式为 **LabelMe 矩形标注**：

```
文件名约定:  Img_Train_{Object}_{Affordance}_{idx}.json
              └──── 如 Bag / contain / 205
每个 JSON:
  - imageHeight, imageWidth: 图像尺寸（实测 612×408 / 612×393）
  - shapes: 两个矩形
      label="object"  : 物体 bbox（Bag 区域）
      label="subject" : 人 bbox（整个人区域）
  - imagePath: 指向对应 JPG（本例指向 Test 集，文件名体现 O-A 层级）

实测 205:
  object  bbox: (212.4, 227.6) -> (360.9, 384.2),  w=148.4, h=156.6
  subject bbox: (220.6,  92.3) -> (436.4, 405.0),  w=215.8, h=312.7
  image size: 612 × 408

实测 206:
  object  bbox: (178.2, 195.6) -> (383.0, 385.2),  w=204.8, h=189.5
  subject bbox: (189.5,  39.3) -> (428.8, 391.3),  w=239.3, h=352.0
  image size: 612 × 393
```

**对设计的三个确认 / 约束**：

1. **HOI 图像是真实人-物交互照片**（非合成渲染），含完整场景背景与人体。VGGT 直接重建会同时得到人、物体和部分背景点云，因此**必须用 bbox 做输入裁剪 / 后过滤**。
2. **object bbox 仅框出物体，subject bbox 框出整个人**。
   - 只 crop `object`：会切掉手部/接触区，丢失 affordance 关键线索；
   - 只 crop `subject`：背景与人体上半身引入大量噪声；
   - **推荐**（§3.2）：以 `object` 为中心外扩 1.2~1.5×，兼顾物体主体与手部/接触区。
3. **JSON 是训练管线可直接消费的元数据**：预计算脚本（§3.5）读取 `object`/`subject` bbox，自动 crop 后喂给 VGGT；无需手工标注或额外预处理。

### 3.4 VGGT 冻结与工程约束

| 约束 | 说明 |
|---|---|
| 冻结权重 | VGGT 1.2B 参数全部冻结，仅 fp16 前向 |
| 显存预算 | 单图 VGGT 前向 ~4-6GB（fp16）；与 GEAL 3DGS + DINOv2 叠加，总训练显存 ~20-24GB |
| 预计算缓存 | 仅 H_raw 离线缓存；H* 训练时在线计算（`AnchorEncoder` 可学习） |
| 分辨率 | VGGT 输入 518×518；输出深度图同 DPT head |
| 噪声容忍 | 单图重建无度量尺度、可能不完整；QueryMe 表 3 已证 3D HOI 特征仍优于 2D HOI |

### 3.5 VGGT 重建预计算管线（仅 H_raw 缓存，全量存储）

```python
# 离线预计算（训练前一次性运行）—— 仅 VGGT 前向，产出不可变基底 H_raw
# H* 的生成移到训练时在线完成（可学习 AnchorEncoder），此处不碰
for object_category O in PIAD:
    for affordance A in affordances_of(O):
        for HOI_image I in folder(O, A):
            json_path = I.with_suffix(".json")
            bboxes = read_labelme(json_path)
            crop = expand_crop(bboxes["object"], scale=1.4, img=I.size)
            I_c = I.crop(crop).resize((518, 518))
            H_raw = VGGT(I_c)                   # 冻结前向，xyz(M,3)+conf(M)
            save(f"hraw/{O}/{A}/{I.id}.npz", H_raw)   # 全量存储，不降采样
```

**H_raw 全量落盘，不在离线阶段降采样**——降采样是处理参数（`M_ds`），现在设 40k、未来可能改 80k，若离线先降采样再存则改 `M_ds` 就得重跑 VGGT，正好踩 §3.5.1 要避免的"处理策略焊死"的坑。268k 点约 2MB/文件，几万张图几十 GB，可接受。

训练时按 (物体 O, affordance A) 从 `hraw/` 读取全量 H_raw，在线完成降采样 + 自适应注意力 + `AnchorEncoder` 生成 H*（k×d）。
点云的 affordance 真值 y 由 PIAD 点云标注提供（文件夹仅物体级别，标注内含 affordance）。
〔备选〕位置感知路径若启用，才在 `compute_hstar` 内额外接 §3.3.2 注册产出对齐坐标，主线无需。

### 3.5.1 缓存策略：H_raw 不可变，H* 在线生成

**规范：H_raw 作不可变基底（全量存储、仅 VGGT 产一次），H* 作在线派生产物（AnchorEncoder 可学习，不落盘）。**

```
hraw/{O}/{A}/{img_id}.npz      # 不可变基底：全量 xyz(M,3) + conf(M)，仅 VGGT 产一次，不删
  │
  │ ← 训练时在线（dataset __getitem__ → H_rawHandler → AnchorEncoder）
  │   ① 降采样 M_raw → M_ds（cfg.M_ds 可调）
  │   ② FPS 全局候选 + conf 加权重要性 → top-k 锚点（numpy）
  │   ③ AnchorEncoder（可学习 MLP + max-pool）→ H* ∈ R^{k×d}
  ▼
H*  在线计算，不落盘，每 batch 由可学习 AnchorEncoder 实时生成
  │
  │  锚点参数（N_fps / k / conf_floor / w_hand）作为 cfg 字段，
  │  改参数只需改 config 重跑训练，H_raw 不变
  ▼
HOIFusion(H*, P~)  →  解码器  →  ω_full
```

- **H_raw 只算一次**：VGGT 前向是最贵步骤，全量落盘后永不再跑；
- **H* 永远在线**：AnchorEncoder 是可学习模型组件，H* 随梯度更新，训练时实时生成，不落盘；
- **处理参数全走 cfg**：`M_ds / N_fps / k / conf_floor / w_hand / w_struct / patch_r_ratio` 作为 config 字段，改参数不碰 H_raw；
- **M_ds 不在离线阶段应用**：268k 全量存储 2MB/文件，降采样放在在线 __getitem__ 里按 cfg 实时做，改 M_ds 无需重跑 VGGT。

---

## 4. 轻量融合模块（HOIFusion）设计 —— 取代查询机制

> **本模块是 v2 相对 v1 的唯一核心新增。** 它用一层交叉注意力替代 v1 的整个查询解码器，把 HOI 先验「提示」式地注入 GEAL 原有解码器的输入特征。

### 4.1 设计动机

- v1 的查询机制改动大、训练不稳、推理需原型的 R1 风险高；
- 更稳的做法：**不动解码器，只动输入特征**。用 `HOIFusion` 让物体点云特征 P~ 对 HOI 特征 H\* 做一次交叉注意力，使「哪些点的几何更可能对应人机接触/交互」的线索被编码进 P~_aug；
- 推理期 H\* 置空 → 交叉注意力退化为对 null token 的注意力（输出≈0 残差），P~_aug ≈ P~，解码器行为退回 GEAL 基线。

### 4.2 模块结构（纯内容检索，无位置偏置）

> 主方案 `HOIFusion` 是**纯内容交叉注意力**：`query=P~`（物体点特征），`key=value=H*`（HOI 锚点特征，**仅特征、无坐标**）。不读任何 xyz，不做位置偏置。〔备选〕位置感知变体见 §3.3.2。

```python
class HOIFusion(nn.Module):
    def __init__(self, d_model=256, n_heads=4):
        self.cross = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # 推理期 H*=None 时使用的可学习空 token（仅特征）
        self.null_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, obj_feat, hoi_feat=None):
        # obj_feat: [B, N, d]  物体点云特征 P~（query）
        # hoi_feat: [B, k, d]  HOI 锚点特征 H*（key/value，训练期）；推理期为 None
        if hoi_feat is None:
            hoi_feat = self.null_token.expand(obj_feat.size(0), -1, -1)
        # 纯内容交叉注意力：物体点按内容检索 HOI 交互特征，无需任何坐标
        attn_out, _ = self.cross(obj_feat, hoi_feat, hoi_feat)
        fused = self.norm1(obj_feat + attn_out)
        fused = self.norm2(fused + self.ffn(fused))
        return fused  # P~_aug
```

参数量：仅一个 cross-attn + FFN + LayerNorm + 1 个 null token，约 1~2M 参数（远低于 v1 查询解码器）。

> **注意：本模块的内容检索机制本身不是创新点，且明确与 QueryMe 的 HOI↔物体步一致（务必讲清，规避 R6「加个 cross-attn」质疑）。**
> - **QueryMe 的「HOI↔物体」跨模态融合是内容检索、而非几何对齐**：QueryMe 把 HOI 图经 VGGT 重建为带坐标的 3D 点 H，但 H 与给定物体点云 P（来自 3DIR/3D-AffordanceNet/Objaverse，独立采集）**不在同一坐标系**，QueryMe 并未将两者配准。其查询在 P 上以 FPS 位置初始化并注入 P **自身**的位置编码（用于物体几何自注意力），cross-attention 让查询**按内容**检索 HOI 特征 token——H\* 的坐标不参与对 P 的位置偏置。即 QueryMe 在「HOI↔物体」这一步是**纯内容**的。**本方案主线与之做法相同**（H\* 同为不带坐标的特征 token、靠内容检索关联），因此 HOIFusion 的内容检索算子是对 QueryMe 该步的沿用/桥接，而非首创。
> - **位置感知交叉注意力是既有技术**（FLAT 2025、LEAPARD CVPR 2022、Rotation-Invariant Transformer、Instance-free Text-to-PointCloud 等），且本方案已将其**降级为备选**（§3.3.2），因为强行把 H\* 对齐到 P 帧**可能损坏 3D HOI 几何**，而增益未经证实。
> - **本方案的真正增量不在「跨模态检索算子」，而在范式**：把 HOI 作为**训练期特权提示（privileged hint）**，通过 Hint KL 蒸馏把交互先验内化进 GEAL 解码器权重，使**推理期完全丢弃 HOI、零 VGGT、纯 3D 部署**——这是 QueryMe（推理仍需 HOI 图）没有的。新颖性应落在「特权提示蒸馏范式 + GEAL 解码器的轻量桥接 + 纯 3D 部署零代价」，而非「内容检索 cross-attn」或「位置感知 cross-attn」。

### 4.3 与 GEAL 解码器的衔接（关键：解码器不变）

GEAL 解码器消费「文本特征 T（as query）+ 视觉特征（as key/value）」。v2 中视觉特征由 P~ 换成 P~_aug，**其余完全不变**：

```
ω_full = GEAL_Decoder(query=T~, key_value=P~_aug)   # 训练期（有 H*）
ω_3D   = GEAL_Decoder(query=T~, key_value=P~)       # 推理期 / 纯路径（H*=null）
```

- Phase 1-2 训出的解码器权重在 Phase 3 可低 LR 微调；
- `HOIFusion` 是新模块，从零训练；
- CAM、GAFM、3DGS、DINOv2、PointNet++ 全部不变。

### 4.4 与 GEAL CAM 的关系

CAM 仍负责「特征层 2D→3D 一致性蒸馏」（3D 特征渲染到 2D 与 DINOv2 对齐）。`HOIFusion` 在特征层之后、解码器之前插入，与 CAM 正交、互不冲突。无需修改 CAM。

---

## 5. 训练管线设计（三阶段，v2 收敛）

### 5.1 阶段总览

Phase 1: GEAL Stage 1 (2D 分支)          ── 不变
    ↓
Phase 2: GEAL Stage 2 (3D 分支 + CAM + HOI 并行路径)
    ── P~ 与 H* 同一 loss 下共同演化
    ↓
Phase 3: Hint 蒸馏强化（p_drop 渐增，ω_3D 内化）
```

> **关键设计决策：AnchorEncoder + HOIFusion 与 PointNet++ 在 Phase 2 中并行训练，而非留到 Phase 3 从零启动。**
>
> 原因：HOIFusion 的 cross-attn 以 P~ 为 query、H* 为 key/value，其有效性依赖于两者处于同一语义空间。若 Phase 3 才启动 AnchorEncoder（此时 P~ 特征空间已定型），H* 必须单方面适配已冻结的 P~，对齐质量差且无梯度驱动。Phase 2 并行训练下，同一 loss 信号同时驱动 PointNet++（学习"什么物体几何对应 affordance"）和 AnchorEncoder（学习"什么 HOI 交互模式对应 affordance"），HOIFusion 天然处于两者之间，特征空间对齐是损失优化的副产品而非额外任务。

### 5.2 Phase 1 / 2（Phase 2 加入 HOI 并行路径）

Phase 1 完全沿用 GEAL 原始 Stage 1（2D 分支）。Phase 2 在 GEAL Stage 2（3D 分支 + CAM）基础上**加入 HOI 并行训练流**，改动为：

```
Phase 2 forward:
1. P → 3DGS 渲染 → DINOv2 → GAFM → F^2D          (冻结, CAM 用)
2. P → PointNet++ → GAFM → P~  (N×d)               ← 可训练（末层）
3. H_raw = load(f"hraw/{O}/{A}/{img_id}.npz")       # 离线缓存，全量
4. H* = AnchorEncoder(compute_hstar(H_raw, cfg))     # 在线计算，可学习
5. Q → RoBERTa → T~
6a. P~_aug = HOIFusion(P~, H*)              → 解码器 → ω_full   (有提示)
6b. P~_aug = HOIFusion(P~, None)            → 解码器 → ω_3D    (无提示)
7. 6a 额外算 CAM: 3D→渲染→2D vs F^2D → L_consis
```

**冻结**：DINOv2、VGGT、3DGS 几何参数、Phase 1 的 2D 分支权重。

**可训练**（Phase 2 即启动，从零权重训练 AnchorEncoder + HOIFusion，从 Phase 1 权重加载其余）：
- `AnchorEncoder` 全部参数（可学习 HOI 局部几何编码器）；
- `HOIFusion` 全部参数（含 null_token）；
- PointNet++ 末层（低 LR）；
- 解码器末层（低 LR）；
- RoBERTa（低 LR，1e-5）。

**损失**：
```
L_phase2 = L_BCE(ω_full, y) + L_Dice(ω_full, y)
         + λ_3D · [L_BCE(ω_3D, y) + L_Dice(ω_3D, y)]
         + λ_hint · KL(softmax(ω_3D/τ) ‖ softmax(ω_full.detach()/τ))
         + λ_consis · L_CAM
```

AnchorEncoder 的梯度链：`L_phase2 → ω_full/ω_3D → 解码器 → HOIFusion → CrossAttn → H*(AnchorEncoder)`，与 P~ 梯度链共享 `L_BCE + L_Dice` 头，两路特征在同一目标下共同学习。

**超参**：
| 参数 | 值 |
|---|---|
| 优化器 | Adam |
| LR (AnchorEncoder / HOIFusion) | 1e-4 |
| LR (解码器末层 / PointNet++末层) | 1e-5 |
| LR (RoBERTa) | 1e-5 |
| Batch | 8 |
| Epochs | 20~30 |
| H\* 维度 d | 256（与 P~ 对齐） |
| H\* 锚点数 k | 128（初始），256（搜索） |
| λ_3D 初始 → 目标 | 0.5 → 1.0 |
| λ_hint | 1.0 |
| λ_consis | 0.1 |
| τ | 2.0 |
| p_drop | 0.2（固定，Phase 3 再渐增） |

### 5.3 Phase 3：Hint 蒸馏强化（核心，仅调度参数变化）

Phase 2 已产出完整权重（含 AnchorEncoder + HOIFusion），Phase 3 **不新增可训练组件**，仅通过损失权重调度进一步逼 ω_3D 内化 HOI 先验：

**冻结**：DINOv2、VGGT、3DGS 几何参数、2D 分支权重。

**可训练**：Phase 2 所有组件（不变）。

**与 Phase 2 的差异**：
- p_drop 从 0.2 逐渐线性增至 0.8~1.0（cosine），训练后期 ω_3D 路径占主导；
- λ_hint 可适度提升（如 1.0 → 2.0），强化蒸馏信号；
- 其余超参不变。

**前向流程**：与 Phase 2 完全相同（见 §5.2），仅损失权重不同。

**损失**：
```
L_phase3 = L_BCE(ω_full, y) + L_Dice(ω_full, y)
         + λ_3D · [L_BCE(ω_3D, y) + L_Dice(ω_3D, y)]
         + λ_hint · KL(softmax(ω_3D/τ) ‖ softmax(ω_full.detach()/τ))
         + λ_consis · L_CAM
```
其中 λ_3D=1.0、λ_hint=2.0、λ_consis=0.1、p_drop: 0.2→0.8 (cosine)。

**为什么 Phase 3 不新增组件**：Phase 2 已让 P~ 和 H* 在同一 loss 下学成对齐特征空间。Phase 3 的任务是调度——让模型在 ω_3D（无 HOI 路径）上独立准确，使推理期零 HOI 性能不劣于训练期有 HOI 性能。这只需调整权重，无需新模块。

### 5.4 训练数据协议迁移（同 v1 §5.6）

| 数据集 | 当前 | 迁移目标 |
|---|---|---|
| 训练 | PIAD / LASO | PIAD（HOI 图按 物体-affordance，点云按 物体；同物体配对） |
| 评测 | PIAD/LASO seen/unseen | PIAD seen/unseen-obj/unseen-aff + PIAD/LASO 兼容对照 |

**同物体配对采样**（PIAD：HOI 图与点云同物体，HOI 按 物体-affordance、点云按 物体 分类）：
- 训练时给定点云 P（物体 O）+ 查询 affordance A，读取 H_raw（hraw/O/A/），**在线**生成 H\*（同物体，k×d 纯特征 token）；
- 点云 affordance 真值 y 由点云标注提供（文件夹仅物体级，标注内含 affordance）；
- 推理期零 HOI，无配对问题、无原型依赖（与 v1 相比根本性地规避了 R1）。

---

## 6. 损失函数设计

### 6.1 损失总览

| 阶段 | 损失项 | 权重 |
|---|---|---|
| Phase 1-2 | BCE + Dice | 1.0 + 1.0 |
| Phase 2 | CAM 一致性 | 0.1 |
| Phase 3 | BCE(ω_full) + Dice(ω_full) | 1.0 + 1.0 |
| Phase 3 | BCE(ω_3D) + Dice(ω_3D) | λ_3D（0.5→1.0） |
| Phase 3 | Hint KL(ω_3D ‖ ω_full) | 1.0 |
| Phase 3 | CAM 一致性 | 0.1 |

### 6.2 Hint KL 的作用

逼「无 HOI 路径」ω_3D 的分布逼近「有 HOI 提示路径」ω_full，使 HOI 交互先验被内化进解码器权重。最终部署用 ω_3D，故 ω_3D 必须既对标签准、又与 ω_full 接近。

### 6.3 相对 v1 的简化

v1 的 `L_query_contrast`（查询-文本对比）为查询机制专属，**v2 删除**；Unseen Affordance 的类间可分由 GEAL 原有文本对齐 + CAM 承担，无需额外对比损失。

---

## 7. 推理期策略

### 7.1 默认部署：纯 3D（v2 主线）

```
输入: Q + P  (零图像, 零VGGT, 零原型库)
→ RoBERTa → T~
→ P→PointNet++ → GAFM → P~
→ HOIFusion(P~, H*=null) → P~_aug ≈ P~
→ GEAL 解码器 → ω_3D
```

完全继承 GEAL 的纯 3D 部署优势；HOIFusion 在 null 下近乎恒等，开销可忽略（或部署时直接去掉该模块）。

### 7.2 可选高性能档（延后，对应 v1 策略 B）

若评估/演示需要「提供 HOI 图时的上限」，启用 §15 查询机制或简单的「推理期也跑 VGGT → H\* → HOIFusion」：
```
输入: Q + P + I
→ VGGT(I) → H*
→ HOIFusion(P~, H*) → 解码器 → ω_with_HOI
```
作为消融上界报「zero-image vs with-HOI」两档。

---

## 8. 与 GEAL 现有组件的融合关系

### 8.1 组件级对照（v2）

| GEAL 组件 | v2 中的角色 | 改动 |
|---|---|---|
| 3DGS（几何桥） | 不变，为 CAM 提供空间对应 | 无 |
| DINOv2（2D 教师） | 不变 | 无 |
| GAFM（粒度融合） | 不变 | 无 |
| CAM（一致性对齐） | 不变，继续正则化 | 无 |
| PointNet++（3D 骨干） | 不变，末层可低 LR | 低 |
| RoBERTa（文本编码） | 不变，低 LR | 低 |
| **GEAL 解码器** | **不变**（仅输入特征由 P~ 换 P~_aug） | **无（v2 关键简化）** |
| **AnchorEncoder** | **新增 HOI 可学习编码器（MLP + max-pool）** | **Phase 2 起训练** |
| **HOIFusion** | **新增轻量交叉注意力（含 null_token）** | **Phase 2 起训练** |
| H_raw 缓存 | 离线 VGGT 全量 H_raw，训练时在线生成 H* | 低（仅 hraw/ 存储） |

> 与 v1 的本质区别：v1 把「GEAL 解码器」标记为「高（核心改动）」，v2 改为「无」。这是用户决策的直接结果。

### 8.2 与既有 V0-V7 / V0-V5 路线的关系（同 v1 §9.2）

本方案是**第三条改进轴**，与前两条正交：
```
路线 1 (SAGE): 改进 3D 特征表示          ── 局部 token + VQ
路线 2 (TASA): 改进 2D→3D 先验传递      ── 软先验反投影
路线 3 (本方案): 新增 HOI 交互先验(特权提示) ── VGGT 重建 + HOIFusion + hint 蒸馏
```
可组合：路线 1 的局部 token 可作 P~ 喂入 HOIFusion；路线 2 的软先验 q_j 可作辅助条件。推荐先落地路线 3（本方案）验证 HOI 增益，再叠加。

---

## 9. 实现路线图

### 9.1 里程碑

| 里程碑 | 内容 | 依赖 | 产出 |
|---|---|---|---|
| M0 | 环境 + VGGT 集成测试 | VGGT 仓库, PIAD | 单图 HOI → 3D 点云 |
| M1 | VGGT 重建预计算（仅 H_raw 缓存，全量存储） | M0 | hraw/ 全量 H_raw |
| M2 | AnchorEncoder + HOIFusion 实现 | GEAL 基线 (Phase1-2) | 可学习 HOI 编码 + 融合模块 |
| M3 | Phase 3 训练（提示+蒸馏） | M1, M2 | ω_full / ω_3D 模型 |
| M4 | 评测 + 消融 | M3 | PIAD 主表 + 消融 |

比 v1 少一个 M2（查询机制）和 M4（原型蒸馏），实现量约减半。

### 9.2 M0: VGGT 集成测试（同 v1 §10.2）

```python
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load import load_and_preprocess_images
from vggt.utils.geometry import unproject_depth_map_to_point_map

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
images = load_and_preprocess_images(["hoi_image.jpg"]).to(device)
with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
    preds = model(images)
    pc = unproject_depth_map_to_point_map(preds["depth"], preds["extrinsic"], preds["intrinsic"])
```

### 9.3 M2: HOIFusion（伪代码，见 §4.2）

（代码已在 §4.2 给出，单交叉注意力 + FFN + null_token。）

### 9.4 M4: 消融矩阵

| 实验 | VGGT HOI | 融合方式 | 推理 HOI | 目标 |
|---|---|---|---|---|
| Baseline | × | — | × | GEAL 基线 |
| +VGGT-2D | 2D HOI feat | 拼接到 P~ | × | 2D vs 3D HOI |
| +VGGT-3D-fuse | 3D H\* | HOIFusion | × | **v2 主线** |
| +Hint-Distill | 3D H\* | HOIFusion | × | 蒸馏是否必要 |
| +HOI-Dropout | 3D H\* | HOIFusion | × | p_drop 调度 |
| (延后) +With-HOI | 3D H\* | HOIFusion | ✓ | 上界（策略 B） |
| H\* 锚点数 k | 3D H\* | HOIFusion | × | k=64/128/256 |
| 融合位置 | 3D H\* | 解码器前 / 解码器中 | × | 注入点消融 |
| 〔备选〕位置感知 | 3D H\* 注册 | 位置感知+注册 / 纯内容 | × | 位置偏置是否优于纯内容（需 ≥ 显著才启用 §3.3.2） |
| VGGT 输入 | 全图 / object-crop / crop+padding | HOIFusion | × | 裁剪策略是否必要（§3.2） |

---

## 10. 风险分析与缓解

### 10.1 风险矩阵（v2）

| 风险 | 等级 | 影响 | 缓解 |
|---|---|---|---|
| **R1: Unseen Affordance 无原型** | **已消除** | v1 最高风险 | v2 推理零 HOI，不依赖原型 |
| **R2: HOI 增益有限** | 中 | 仅训练期提示，推理退回纯 3D，增益可能小于 v1 | 增大 H_raw 数据池（同 affordance 多图）；提高 AnchorEncoder 容量/多样性；必要时启用 §15 查询机制 |
| **R3: 训练/推理偏移（null token）** | 低 | 推理 H*=null，训练多 H\* | p_drop 渐增 + Hint KL 对齐；监控 ω_3D vs ω_full aIoU 差 |
| **R4: 显存** | 中 | VGGT + DINOv2 + 3DGS 叠加 | H\* 预计算缓存；fp16；batch 4~8 |
| **R5: 配对采样协议** | 低 | HOI 按 (O,A)、点云按 O 配对采样逻辑 | 同物体配对、纯内容检索无需注册；评测仍兼容 PIAD/LASO |
| **R6: 新颖性质疑** | 中 | 被读成「加个 cross-attn」 | 见 §11 新颖性自检 |
| **R8: 〔仅备选〕注册质量** | 中（仅备选路径） | 若启用 §3.3.2 位置感知，VGGT 尺度/帧/人体分割/部分可见影响 H* 对齐 | 仅备选路径风险；主线纯内容检索完全规避。启用时：注册到规范帧 + 低置信退化为纯内容；监控重叠率 |
| **R7: 与 v1 路线冲突** | 低 | 两版并存 | v2 为当前主线，v1 查询机制降为 §15 附录 |

### 10.2 R2 详细分析（v2 核心风险）

v2 的最大不确定是「训练期提示能否有效内化」。若 ω_3D 相对 GEAL 基线增益很小（如 < 1 aIoU），说明 HOI 提示未被解码器吸收。排查：
1. 检查 Hint KL 是否过早归零（ω_3D 直接学成 ω_full 的复制但丢弃了几何）；
2. 增大 H_raw 数据池（同 affordance 更多 HOI 图），让 AnchorEncoder 看到更多样的交互模式；
3. 在 PointNet++ 末层 + 解码器给更高 LR，让特征真正吸收 HOI 线索；
4. **兜底**：启用 §15 查询机制（增益更高但需原型/推理 HOI）。

---

## 11. 新颖性自检

### 11.1 危险点

- 被读成「在 GEAL 解码器前加一个 cross-attention」=  trivial 模块堆叠 → 需上升到范式层防守。

### 11.2 防守点（范式级）

1. **特权提示蒸馏范式（privileged hint training）**：首次把 QueryMe 的 VGGT 重建**从「推理期多模态输入」重构为「训练期特权提示 + 推理期蒸馏丢弃」**。这不是换模块，而是把「推理期 HOI 依赖」整个消解，同时保留 GEAL 纯 3D 部署——对 affordance grounding 领域是新的训练范式。

2. **双通道 2D→3D 桥**：GEAL 用 3DGS/DINOv2 建立物体语义 3D 场；VGGT 另立 HOI 几何 3D 通道。两通道在 `HOIFusion` 处对齐：物体点云特征对 HOI 锚点做交叉注意力，显式建模「人-物交互几何 → 物体功能区」的映射，而 GEAL 原版完全无此交互先验。

3. **与 CAM 协同**：`HOIFusion` 在特征层之后、解码器之前插入，与 CAM 的「特征层 2D→3D 一致性」正交互补，形成「特征层一致性（CAM）+ 推理层 HOI 提示（HOIFusion）」双层先验，且 GEAL 解码器零改动。

4. **部署无代价**：相比 QueryMe（推理需 HOI 图 + VGGT）和 v1（推理需原型库），v2 推理与 GEAL 完全一致（零图像、零 VGGT、零原型），Unseen Affordance 风险归零。这是相对两条前驱工作的实质性工程/范式优势。

### 11.3 相对 QueryMe / v1 的增量

| 方面 | QueryMe | v1（查询机制） | v2（本方案，主线） |
|---|---|---|---|
| 解码器 | FPS 查询 | 替换 GEAL 解码器 | **保留 GEAL 解码器** |
| HOI 角色 | 推理期输入 | 训练+推理(原型) | **仅训练期提示** |
| 推理模态 | HOI+VGGT+点云 | 纯3D/原型(零图) | **纯3D(零图,零原型)** |
| Unseen Aff 风险 | 推理吃HOI图 | 原型缺失风险 | **无（零HOI依赖）** |
| 改动规模 | — | 高 | **低** |

---

## 12. 第一轮实现清单

### 代码模块

- [ ] `model/vggt_wrapper.py`：VGGT 冻结前向 + 深度反投影 → 3D HOI 点云
- [ ] `model/hraw_handler.py`：训练时在线处理 H_raw（降采样 + 自适应空间注意力 → 锚点 + patch 构建）
- [ ] `model/anchor_encoder.py`：**可学习 AnchorEncoder**（PointNet 式 MLP + max-pool，H* 编码器）
- [ ] `model/hoi_fusion.py`：**`HOIFusion` 单交叉注意力 + null_token（v2 核心）**
- [ ] `scripts/precompute_hraw.py`：离线 VGGT 重建全量 H_raw 落盘 hraw/
- [ ] `scripts/train_phase3.py`：Phase 3 提示训练 + Hint 蒸馏 + p_drop 调度（dataset __getitem__ 在线生成 H*）
- [ ] `config/train_phase3_hint.yaml`：Phase 3 配置（含 M_ds / N_fps / k / conf_floor 等处理参数）
- [ ] （延后）`model/query_learning_decoder.py`：§15 查询机制

### 实验模块

- [ ] VGGT 单图重建质量验证（HOI 图 → 3D 点云可视化）
- [ ] Phase 3 ω_full vs ω_3D（Hint KL + aIoU 差监控）
- [ ] v2 主线 vs GEAL 基线（PIAD 主表 + PIAD/LASO 兼容）
- [ ] 消融：2D HOI vs 3D HOI、hint 蒸馏有无、p_drop 调度、H\* 锚点数 k、融合注入位置
- [ ] 三随机种子统计
- [ ] 参数量 / 显存 / 推理延迟（应≈GEAL 基线）
- [ ] （延后）with-HOI 上界（策略 B）

---

## 13. 论文叙事建议（v2）

### 推荐主叙事

> 现有 3D affordance 方法要么在推理期依赖外部 HOI 图像（QueryMe），要么缺乏交互先验（GEAL）。我们提出把前馈单目重建（VGGT）作为**训练期特权提示**引入 GEAL：冻结 VGGT 重建 HOI 图为 3D 交互点云 H_raw（离线缓存、全量存储），训练时在线经自适应空间注意力与置信度初筛提取交互锚点，由**可学习**的 AnchorEncoder 编码为 H\*（纯内容特征 token，不引入坐标对齐/注册），通过一个轻量融合模块让物体点云特征对 HOI 锚点做**内容交叉注意力**，将人机交互几何先验注入 GEAL 的标准解码器。训练时同时优化「有提示」与「无提示」两条路径并以 KL 对齐，使交互先验内化进解码器权重；推理时直接丢弃 HOI 提示，模型以纯 3D 形式部署，零图像、零重建、零原型依赖。AnchorEncoder 由 affordance loss 经 HOIFusion 反向传播驱动，从通用几何描述子进化为任务相关交互特征。在 GEAL 的 3DGS + CAM 语义场上，该方案以对解码器零改动的代价获得 HOI 交互先验，且不依赖推理期 HOI，因而从根本上规避了 unseen affordance 的原型缺失风险。

### 核心贡献

1. 将 VGGT 前馈重建从「推理期多模态输入」重构为「训练期特权提示 + 推理期蒸馏丢弃」的新范式；
2. 轻量 `HOIFusion` 单交叉注意力模块，对 GEAL 解码器零改动地注入 HOI 交互先验；
3. 双通道 2D→3D 桥（3DGS/DINOv2 物体语义 + VGGT HOI 几何）在特征层与 CAM 协同；
4. 纯 3D 推理部署，Unseen Affordance 无原型依赖风险。

---

## 14. 与 v1 的衔接说明

- v1（查询机制版）的完整设计（FPS 查询、T→H\*→P 三层注意力、原型库、KL 蒸馏到纯 3D）**未丢弃**，仅**延后**。
- 当 v2 主线在 PIAD 上验证 HOI 提示增益有限（R2），或需要「推理期也吃 HOI」的高性能档时，启用下一节查询机制。
- 两版共享 §3（VGGT 接入）、§5.4/§7（数据迁移）、§9（路线关系）。

---

## 15. 附录：延后模块 —— QueryMe 查询机制（未来可选）

> 仅在 v2 主线增益不足或需 with-HOI 上界时启用。以下为 v1 完整设计摘要，细节见 v1 存档。

### 15.1 何时启用

- v2 的 ω_3D 相对 GEAL 基线增益 < 1 aIoU（R2 触发）；
- 需要报「with-HOI」高性能档（策略 B）。

### 15.2 查询机制结构

- **查询初始化**：FPS(P) 取 M 个位置 → MLP 位置编码 + 可学习内容 → queries；
- **每层 L 步更新**：
  ```
  Step1: Q_t = CrossAttn(Q, T~, T~)
  Step2: Q_h = CrossAttn(Q_t, H*, H*)   # 训练 H* / 推理 φ_A
  Step3: Q_p = CrossAttn(Q_h, P~, P~)
  Step4: Q_{l+1} = SelfAttn(Q_p + pos)
  ```
- **解码**：最终查询 × P~ 注意力 → sigmoid → ω；
- **替代 GEAL 解码器**（v2 是保留解码器 + HOIFusion）。

### 15.3 推理期差异

- v1 推理需原型库 φ_A（零图像但需原型）→ **Unseen Affordance 无 φ_A 风险（R1）**；
- 或启用策略 B（推理期也跑 VGGT）→ 破坏纯 3D 部署。

### 15.4 与 v2 的切换成本

- 共享 VGGT 接入（§3）、H_raw 缓存（§3.5）、AnchorEncoder（§3.3）；
- 额外新增：`QueryLearningDecoder`、`QueryLayer`、`prototype_bank`、Phase 4 原型蒸馏；
- 解码器从「保留」改为「替换」，改动规模升回 v1 的「高」。

---

*本方案所有模块、损失、超参与训练策略均为待验证设计，不等于已验证结果。落地前须回看 VGGT 原文确认 API、回看 QueryMe 原文确认自适应注意力细节、回看 GEAL 代码确认 HOIFusion 接入点。v2 主线编码量约为 v1 的一半，优先实现。*
