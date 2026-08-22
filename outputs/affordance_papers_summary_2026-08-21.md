# 7 篇 Affordance 论文：模型与创新点总结

> 整理日期：2026-08-21
> 用途：竞品/底盘态势梳理（与你 GEAL-based affordance grounding 研究相关）
> 标注：`【事实】`=论文原文内容；`【分析】`=结合你研究计划的推演，非原文

---

## 1. Probing and Bridging Geometry–Interaction Cues for Affordance Reasoning in Vision Foundation Models
- **出处**：CVPR 2026，arXiv:2602.20501（ANU，Qing Zhang 等）
- **任务**：2D affordance 估计（图像 + (agent, object, verb) 三元组 → affordance mask）

### 模型
【事实】**完全训练无关（training-free, zero-shot）** 的探测式框架，分三步：
1. **Geometry Probing**：用 DINOv3 提取 part-level 几何原型（PCA 得到几何基）；
2. **Interaction Probing**：用 Flux Kontext 提取 verb-conditioned 空间注意力图（隐式交互先验）；
3. **Geometry–Interaction Fusion**：物体注意力裁剪 ROI → PCA 选与 verb 最对齐的几何基 → 几何基 × verb 注意力融合成 mask。

### 创新点
【事实】**机制性研究（mechanistic account）** 而非新方法堆叠：
- 提出假设并验证：几何感知 + 交互感知 是 VFM 中 affordance 理解的两块**可组合（composable）基本构件**；
- 发现 DINO 类模型天然编码几何，生成模型（Flux）天然编码 verb 条件交互先验；
- **零参数、零训练**融合即与弱监督方法 competitive；
- 顺带结论：往 CLIP/SigLIP 灌 Metric3Dv2 深度/法线可涨点，但 DINOv2 几乎不涨——说明其几何先验已内化在权重里。

【分析】这篇是你的"几何 vs 交互"双维论证的理论支撑。注意它证明 **DINO 几何先验已饱和**，与你"换 DINO→CLIP 对齐"路线3 的作废判断一致：VLM 类文本对齐并不比 DINO 几何更"本质"。

---

## 2. CompassAD: Intent-Driven 3D Affordance Grounding in Functionally Competing Objects
- **出处**：arXiv:2604.02060（NTU MARS Lab，Jianfei Yang 组）
- **任务**：多物体点云 + 隐式自然语言意图 → 正确物体上的逐点 affordance mask（"confusing pairs" 选择）

### 模型
【事实】**CompassNet**：
- 主干：Uni3D（点编码器）+ RoBERTa（文本编码器）逐点/文本特征；
- **ICI（Instance-bounded Cross Injection）**：实例边界内做 region–language cross-attention + 可学习 background token + gated 回传，从构造上阻止跨物体语义泄漏；
- **BCR（Bi-level Contrastive Refinement）**：训练期双损失——TG-Softmax（组内选最匹配意图的 region）、TP-HardNeg（抑制混淆表面的高分流负样本）；**推理零新增参数/计算**。

### 创新点
【事实】
- 新设定 **Intent-Driven Confusable Affordance Grounding**（多物体、隐式意图、query-dependent）；
- **CompassAD 基准**：30 个混淆对、16 类 affordance、6,422 组合、88K+ QA；
- 结果：seen 18.20 IoU（超 GLANCE 4.02），unseen 15.36 IoU；真实机械臂抓取验证。

【分析】这是**非 POSTECH 的 intent 槽竞争者**（你记忆里的 CompassAD）。它有 intent 但**无 completion、无蒸馏到轻量 3D 分支**——与你 G1（意图⊕补全⊕轻量蒸馏三者耦合）正交，不构成直接冲突，但占掉了"意图单点"叙事，故你的 moat 必须落在**补全×蒸馏交叉**而非意图单点。

---

## 3. GROW²: Grounding Which and Where for Robot Tool Use
- **出处**：arXiv:2606.30632（David Hsu 组，Yuhong Deng、Yuyao Liu）
- **任务**：开放世界工具使用——"选哪个物体当工具 + 工具上何处交互"（open-category）

### 模型
【事实】**GROW²（GROunding Which and Where）** 分层两阶段，以**物体部件（object parts）为中间抽象**：
- **语义层（Which）**：VLM 解析自然语言指令 → 列物体 → SAM 裁剪 → VLM 用 in-context/CoT 选工具 o_A、目标 o_B、抓取部件 p_G、任务相关部件；
- **几何层（Where）**：单视角 RGB-D 重建并注册 mesh → 多视角渲染 → 在各视角分割所选部件 → 融合成 3D affordance 区域；
- 把 affordance 定义为**两个实体间的非对称二元关系**（机器人–工具抓取；工具–目标功能）。

### 创新点
【事实】
- 把开放世界 affordance grounding 拆成 **语义选择 + 几何落地** 两级，绕开 data-heavy 端到端训练；
- **零样本泛化**到开放类别；GROW2Bench 基准；
- 仿真 + 真实机器人双验证，10 类任务真实抓取成功率领先。

【分析】典型的"VLM 常识 + 几何精修"两段式，**推理仍跑重型 VLM**（延迟 ~16.6s），无轻量蒸馏、无遮挡补全。与你路线②的"MLLM 意图仅训练时用、推理零重模型"形成**直接效率对比**——你的护城河在"推理不跑 MLLM"。

---

## 4. A-Harness: Affordance Agent Harness — Verification-Gated Skill Orchestration
- **出处**：arXiv:2605.00663（Haojian Huang、Yingcong Chen 等）
- **任务**：开放世界 affordance grounding 的**系统/运行时**层（非新感知模型）

### 模型
【事实】**A-Harness 闭环运行时**，统一异构技能（检测、分割、zoom-in、web search、interaction imagination）：
- **Evidence Store**：带溯源（provenance）的异质技能输出累积；
- **Two-Tier Memory**：常识库（频繁物体稳定先验）+ 测试时 Episode Bank（验证通过的轨迹在线累积）；
- **Budget-Aware Router**：按"单位成本信息增益"选下一技能与参数；
- **Verifier**：用**相对信号**（cross-tool 一致性、cross-scale 稳定性、evidence 充分性）门控承诺、触发定向重试；
- **Final Judge**：融合证据+轨迹+记忆出预测。

### 创新点
【事实】
- 用**每样本自适应路由**替代固定 pipeline；
- **相对可操作验证信号**门控承诺 + 定向重试（无需测试时 GT）；
- Episode Memory 摊销反复物体的成功工具链；
- 在多个基准上取得**更优 accuracy–cost Pareto 前沿**，平均技能调用与延迟更低。

【分析】这是**正交维度**（test-time 系统编排），可与任何感知模型叠加。对你而言是可借鉴的"推理期验证/重试"机制，非底盘竞争者；但其"不靠更强模型、靠系统级证据获取"的论点，呼应你"效率护城河缩水、须靠架构耦合而非单纯效率"的判断。

---

## 5. VoxAfford: Multi-Scale Voxel-Token Fusion for Open-Vocabulary 3D Affordance Detection
- **出处**：arXiv:2605.01365（Xuguang Lan 组，西北工大）
- **任务**：开放词汇 3D affordance 检测（点云 + 新 affordance 描述 → 逐点 mask）

### 模型
【事实】**VoxAfford**：
- 主干：LoRA 微调 LLM 自回归生成 affordance 输出 token；
- **分层 Voxel-Token 空间融合**：冻结 3D VQVAE 编码器提取三尺度体素特征（16³/32³/64³）→ 每个输出 token 以 affordance 语义为 query，cross-attention 从配对体素尺度检索几何模式 → **learned compatibility gate** 控制注入强度；
- **Affordance-Aware Spatial Propagation**：增强 token 经语义条件注意力聚成空间感知 prompt，与逐点特征双路传播 → affordance decoder。

### 创新点
【事实】
- 指出 MLLM 自回归 **"语义富、空间贫"瓶颈**（建模序列依赖而非空间邻域）；
- **生成后注入**预学分层几何（不改 LLM 架构）绕过该瓶颈；
- 每个 token 配对专属体素分辨率 → 多尺度专精；
- SOTA：OpenAD full-view **34.48 mIoUc / 39.12 mIoUi**（超前 best +4.05/+8.84）；Franka 真实机器人零样本 5 新物体 66% 成功率。

【分析】这是**MLLM + 3D affordance 的强基线**，且同样意识到"MLLM token 空间贫乏"。它与你的 ① MLLM 教师路线**部分重合**：你若用 MLLM 作教师蒸馏到轻量 3D 分支，需明确与 VoxAfford 的区别——VoxAfford **推理仍跑 LLM**（无轻量蒸馏、无遮挡补全），你的 novelty 在"训练用重教师、推理零 MLLM + 隐式补全"。

---

## 6. Intermediate Connectors and Geometric Priors for Language-Guided Affordance Segmentation on Unseen Object Categories（GLANCE）
- **出处**：ICCV 2025，DOI:10.1109/ICCV51701.2025.02120（Angela Yao 组）
- **任务**：LASO（Language-guided Affordance Segmentation on 3D Object）的 unseen 类别泛化

### 模型
【事实】**GLANCE**：
- **CMC（Cross-Modal Connector）**：轻量 MLP 桥接**冻结的 3D 与文本主干的中间层**（而非仅末层），共享投影层联合优化，保留可泛化几何/语言模式（专为高分辨率、低数据 3D 设计，避免注意力融合的计算/过拟合）；
- **GAQG（Geometric-Aware Query Generator）**：用 VLM 多视角 2D 分割 → 按 intra-view 可靠性 + cross-view 一致性提取 3D 关键点，提供稀疏结构先验。

### 创新点
【事实】
- 诊断 unseen 泛化差的两大根因：中间层可泛化模式未被利用 + mask decoder 缺结构知识；
- CMC 把跨模态对齐下推到**中间层**；GAQG 用多视角 VLM 分割提供几何 query 先验；
- 两基准 SoTA，unseen 类别 IoU gap 显著缩小。

【分析】GLANCE 正是 CompassAD 对比并超越的 3D LASO 基线（CompassAD seen 18.20 vs GLANCE 14.18）。它是**单物体、闭集/弱开放**路线的代表，无意图、无补全、无蒸馏——属于你 G1 空白地图中"已被占的 LASO 线"，你的工作应站在其肩膀上而非与之对标。

---

## 7. Three-dimensional affordance segmentation for object point cloud driven by language instructions（IDAS）
- **出处**：Engineering ITEE 2026, 27(4):260044（山东大学，控制学院）
- **任务**：指令驱动的 3D 物体 affordance 分割（服务机器人 6-DoF 抓取先验）

### 模型
【事实】**IDAS 网络**：
- 点云：PointNet++ 三层 set abstraction 下采样；语言：RoBERTa-base（投影到 512 维）；
- **VN 机制**：spaCy 词性标注取 verb/noun 位置掩码 → 两层 LSTM 编码（强调动词/名词）；
- **QAM（Query-conditioned Affordance Modulation）**：用语言特征经 MLP 学 α,β，对逐层点云特征做仿射变换（FiLM 式条件调制），实现"逐层"语言-几何融合；
- cross-attention mask decoder 出 affordance 概率。

### 创新点
【事实】
- 新任务 **instruction-driven 3D affordance segmentation** + **IAD 数据集**（7,190 实例、20 类、624 指令、seen/unseen）；
- VN 机制突出动词/名词（指令语义核心）；QAM 把语言条件**逐层注入**点云特征；
- seen/unseen 均超基线。

【分析】偏"工程/服务机器人"取向，方法相对传统（PointNet++ + RoBERTa + FiLM），属**闭集/弱开放**早期路线，无 MLLM、无补全、无蒸馏。作为 3D LASO 家族的底层基线参考即可。

---

## 与你研究计划的关联速览（【分析】）
| 论文 | 占的槽位 | 与你的 G1（意图⊕补全⊕轻量蒸馏）关系 |
|---|---|---|
| Probing/Bridging | 几何+交互机制理论 | 支撑"几何先验已饱和"，印证路线3 作废 |
| CompassAD | intent 单点（多物体混淆） | 占 intent 槽但无补全/无蒸馏 → 你 moat 落补全×蒸馏 |
| GROW² | VLM 语义+几何两段式 | 推理跑重 VLM，无轻量蒸馏/补全 → 你的"推理零 MLLM"对比项 |
| A-Harness | test-time 系统编排 | 正交，可叠加的验证/重试机制 |
| VoxAfford | MLLM+3D affordance 强基线 | 推理仍跑 LLM、无补全 → 你 ① 路线需明确区分 |
| GLANCE | 3D LASO unseen 泛化 | 单物体闭集基线，CompassAD 已超越，非对标对象 |
| IDAS | 传统 3D LASO 工程基线 | 底层参考，闭集/弱开放 |

**结论**：7 篇中无人同时做"意图⊕生成式补全⊕轻量蒸馏三者耦合"（G1 仍空）；你的 novelty 真正落点仍是 ① MLLM 意图教师 + ② 生成式完整几何 + GEAL 一致性蒸馏进轻量 3D 分支（推理零重模型、隐式补全）。VoxAfford/CompassAD 分别占住 MLLM 与 intent 单点，倒逼你用**架构级双向耦合 + 不可分解消融**辩护。
