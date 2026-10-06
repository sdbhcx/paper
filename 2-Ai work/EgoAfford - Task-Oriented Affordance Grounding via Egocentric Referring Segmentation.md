---
title: "EgoAfford: Task-Oriented Affordance Grounding via Egocentric Referring Segmentation"
authors:
  - Xinyuan Guan
  - Feifan Chen
  - Xinyu Zhan
  - Fu-Cheng Zhang
  - Cewu Lu
  - Lixin Yang
year: 2026
venue: arXiv:2608.04533v1 [cs.CV]
doi: arXiv:2608.04533
tags:
  - 文献笔记
  - affordance
  - egocentric
  - referring segmentation
  - MLLM
  - task planning
  - tabletop
  - EgoAfford
type: paper-note
source: D:/article/2026-8-4EgoAfford.pdf
created: 2026-08-25
code: https://github.com/Pandenotium/EgoLens
---

# EgoAfford: Task-Oriented Affordance Grounding via Egocentric Referring Segmentation（arXiv 2026-08）

> [!summary] 一句话总结
> EgoAfford 把"部件级 affordance grounding"从**单目标、单步**推进到**多步任务导向**：给定一张第三人称/第一人称桌面观测和高级任务，模型要同时**生成剩余计划**并**分割下一步动作的三个语义角色（直接物体 Mo / 工具 Ms / 目的地 Md）的部件级功能区域**。配套提出 3B 多模态大模型 **EgoLens**（三路角色专属 mask 解码器 + 显式剩余计划生成），并在 ~15.5k 生成图像 + 102 张真实图像（EgoAfford-Real）上建立了联合"感知—规划"的基准。

---

## 1. 研究背景与动机

> [!info] 原文事实（背景）
> - Affordance 描述环境提供给智能体的"行动可能性"[Gibson 1979]，桥接感知与交互。
> - 物体操作中，affordance 常定位于"功能性部件"——如刀刃用于切、壶嘴用于倒。
> - 近期方法把 grounding 泛化到跨物体、跨类别、跨指令 [11,20,26]。

**既有范式两个核心局限**（作者原文指出，P1）：
1. ** predominantly single-target（单目标）**：每个样本只 grounding 一个"动作条件功能区域"，不明确分配多个参与物体的角色 [26,27,29]。
2. ** predominantly single-step（单步）**：即使最近的双手/场景级方法，也只为"给定动作/指令"预测 affordance，而不是基于任务进度推断后续流程 [7,8]。

> [!question] 方法缺口（作者定义的问题空间）
> 要把 affordance grounding 从"元动作"扩展到"完整任务"，需要把三件事连起来：
> - 参与物体的**语义角色**（谁是被操作物、谁是工具、谁是目的地）
> - 与**任务状态对齐**的视觉观测
> - **多步规划**

---

## 2. 任务定义（Problem Formulation，原文事实）

给定第一人称桌面图像 `I` 和任务的自然语言描述 `T`，模型须：
1. **(i) 生成计划**：完成 `T` 的动作步骤序列 `{T1, …, Tn}`；
2. **(ii) 分割** `I` 为下一步动作对应的三个细粒度 mask `{Mo, Ms, Md}`：

| 符号 | 名称 | 定义 |
|------|------|------|
| `Mo` | Direct Object（直接物体） | 被操作的物体（徒手或经工具） |
| `Ms` | Instrument（工具） | 握在手中作用于直接物体的物体；徒手时缺失 |
| `Md` | Destination（目的地） | 转移动作的目标位置/物体；无物质转移时缺失 |

**部件级（part-level）而非物体级**：mask 勾勒功能性部件（刀刃、壶嘴），而非物体整体。**把手（handle）被排除**，除非握持外还有功能——因为把手的抓取 affordance 可由类别级先验给出，而功能性部件承载动作特定信息。缺失成分对应全零 mask。

> [!note] 三条补充规则（原文 P3）
> 1. 被操作材料在容器中（如壶中水）→ 容器标注为直接物体；
> 2. 承载材料的"装载工具"（如一勺糖浆）→ 标注为工具并含其携带材料；
> 3. 多个同类语义实例 → 需分割所有实例（动作可在任一上执行）。

> [!warning] 分析者推演
> 这套形式化把"缺失角色"显式建模（全零 mask + 计划推断），是相对单步 referring segmentation 的关键扩展；但它要求模型同时具备**可行动作推断 + 任务分解 + 部件级分割**三类能力，难度显著叠加。

---

## 3. 整体架构：EgoAfford（数据集与基准）

### 3.1 生成式数据管线（Task Metadata Generation，原文事实）

- **任务种子**：用多个商用 LLM（Gemini 3 Flash Preview、GPT 5.5、DeepSeek V4 Pro、Claude Opus 4.8）各自"想象"不同场景中的多样物体作为锚点物体，促进多样性。
- **任务与步骤分解**：LLM 为每个锚点物体+场景生成若干复杂任务及其步骤分解。
- **语义对齐图像**：LLM 枚举任务所需全部物体，并为每一步生成"执行前场景状态"的文本描述（位置+状态）。
- **图像生成**：多数用 **GPT-Image-2**（质量），早期验证保留少量 **FLUX-2[dev]** 图像。
- **人工筛选**：状态不一致的图像用人工修正的场景描述重新生成，或标记为无效。

> [!warning] 原文待核对
> 论文称"to our knowledge, no off-the-shelf dataset provides such imagery"，且明确说明**图像为生成式**，存在域差距（见 Limitations）。

### 3.2 标注管线（Annotation Pipeline，原文事实）

- **初始提案**：Grounded-SAM [24] 用 bbox 定位每个列出物体 → 标注 LLM（GPT-5.5）把动作成分连到 bbox 并在功能性部件上打点 prompt → SAM2 产出部件级 mask。
- **人工验证修正**：因 LLM 幻觉与空间 grounding 有限，自动标注仅作初始提案；标注员核验角色分配并重新 prompt SAM2 修正每个 mask。
- **提案质量**：修正前 GPT-5.5–SAM2 管线相对最终人工修正 mask 仅达 **0.696 gIoU / 0.523 cIoU**——说明即便有分解、bbox 等辅助信息，开箱即用的 grounding 仍不足以产出可靠细粒度标注。

### 3.3 数据集规模与基准协议（原文事实）

- **EgoAfford**：**15,537** 张人工核验图像，来自 **2,000** 个生成场景；含 **4,123** 个唯一物体名、391 个唯一动词，聚类（spaCy）后为 **1,188** 物体类型、357 动作类型。
- **EgoAfford-Real**：**102** 张人工实拍桌面图像，覆盖 **26** 个任务，与 2000 生成场景不相交，用于评测对真实图像的泛化。
- **划分**：场景级划分，1,900 场景训练 / 100 场景（488 图）测试，同场景所有步骤归入同划分。
- **admissible candidates（可采纳候选）**：一个任务状态可有多条可执行下一步；两位标注员独立标注候选描述与角色 mask，取并集作为 annotated admissible candidates。训练观测只保留一条参考路径。
- **规划评测**：每个任务额外关联参考步骤上的成对**优先级约束**（precedence constraints），自动提案经人工审核后用于打分。

### 3.4 评测指标（原文事实）

**分割**：主指标 **gIoU**（逐 mask IoU 在测试集上平均）与 **cIoU**（全局累计交并比）。缺失成分约定：GT 为空时预测也为空则 per-mask IoU=1，否则 0；cIoU 中空–空对不贡献，假阳性只进 union。gIoU 强调"成分是否正确识别（含识别缺失）"，cIoU 反映整体像素级精度。多候选时选使 gIoU 最高的 GT。补充 **NE-mIoU**（仅对 GT 非空样本平均 IoU）。

**规划**（用现成 cross-encoder [23] 算预测步骤与参考剩余计划的成对语义相似度，Hungarian 一对一最大权重匹配，阈值 τ=0.6）：
- **Semantic F1** = 2·P·R/(P+R)，P=Semantic Precision（按预测步平均），R=Semantic Recall（按参考步平均）；冗余降精度，遗漏降召回，重复预测不能复用同一参考步。
- **First Step Similarity (F.S. Sim)**：首预测步与所选 GT 候选的下一步描述相似度。
- **Constraint Satisfaction Ratio (CSR)**：阈值匹配下，优先级约束两端都被匹配且顺序正确才算满足，占所有有效约束比例；无剩余约束的状态记 1。
- **Coverage Score**：即 Semantic Recall（参考计划步骤是否被覆盖）。

---

## 4. EgoLens：参考模型（原文事实）

### 4.1 架构

- 基于 **LENS [38]** 架构（query-based 解码支持单前向多 mask，且保留规划所需的语言建模能力）。
- 基座：**3B Qwen2.5-VL**，消费视觉-语言 token + 可学习 queries；query 隐状态过 **4 层 transformer + decoder** 产出 **SAM2 prompt embedding**，由 SAM2 预测最终 mask。
- **关键改造**：把 decoder 复制为**三路并行分支**，每路对应一个动作角色（Mo/Ms/Md）。这样"mask 与角色的语义绑定"是**架构级**的，而非靠文本推断（Fig.4）。
- 为何不用 `<SEG>`-token 范式（LISA、Sa2VA）：该类方法依赖正确文本生成来决定 mask 数量，不适合"三成分"形式化。

### 4.2 训练

- 统一 prompt 模板（任务描述 + 输出格式约束）。模型以 `<think>` 标签包裹的 **CoT 风格规划 trace** + 结构化答案（剩余步骤 + 下一步三成分，每成分含文本名与 bbox）作答。
- **Teacher forcing**：训练时规划 trace 用参考剩余步骤实例化并作为语义前缀，其 token **不计入 LM loss**，但对因果 transformer 与后续 mask query 可见。推理时不给参考计划，模型自回归生成规划与答案，再据生成序列预测 mask。
- **损失**：
  - `Llm`：mask 掉被 teacher-force 的规划 trace token，只监督 `<answer>` 内 token。
  - `Lseg`：非空 mask 用 BCE+Dice 之和监督，空 mask 仅用 BCE（Dice 对空目标无定义）。
    `Lseg = (1/3)[ Σ_{i∈P}(LBCE(i)+LDice(i)) + Σ_{i∈E} LBCE(i) ]`
  - 总损失 `L = λlm·Llm + λseg·Lseg`，实验中 `λlm=λseg=1.0`。
- **实现**：从 Qwen2.5-VL-3B + SAM2 初始化，AdamW，lr=3e-5，训练 40 epoch，8×NVIDIA H200，单卡 batch 16，4 步梯度累积，约 12 小时，随机种子 42，图像 1024×768。

---

## 5. 实验与结果

### 5.1 设置（原文事实）

- **基线**：4 个近期 MLLM referring segmentation 模型——OMG-LLaVA、Sa2VA、UniPixel、LENS（官方 checkpoint）；2 个诊断变体：①商用 VLM 做规划+分解+视觉查询，SAM2 出 mask；②**EgoLens-Seg**（仅分割变体，从给定动作步预测三成分）。
- **三种评测设置**：Full task（图+任务→计划+三 mask）/ Reference-step segmentation（+ 给参考下一步）/ Task-only segmentation（不给参考步，自推动作相关成分）。
- 两个分割-only 设置下，开源 MLLM 基线对每个成分分别查询（每样本 3 次前向）。

### 5.2 表 1：Full Task 主结果（原文事实）

| Method | gIoU | cIoU | F.S.Sim | Sem.F1 | CSR | Cover. |
|--------|------|------|---------|--------|-----|--------|
| Sa2VA* | — | 0.152 | 0.466 | 0.277 | 0.386 | 0.261 |
| UniPixel* | — | 0.218 | — | — | — | — |
| LENS* | — | 0.245 | 0.225 | 0.151 | 0.241 | 0.179 |
| OMG-LLaVA* | — | 0.140 | 0.001 | 0.055 | 0.231 | 0.057 |
| Claude 4.8 (+SAM2) | 0.360 | 0.249 | 0.491 | 0.372 | 0.514 | 0.402 |
| GPT 5.5 (+SAM2) | 0.476 | 0.339 | 0.621 | 0.444 | 0.500 | 0.424 |
| Gemini 3 F.P. (+SAM2) | 0.553 | 0.219 | 0.680 | 0.537 | 0.609 | 0.556 |
| **EgoLens** | **0.700** | **0.486** | **0.666** | **0.500** | **0.624** | **0.566** |

> [!note] 原文解读
> - 开源基线在 full-task 协议下**无法稳定单次返回全部三角色 mask**（* 标记），故省略其 gIoU（缺失预测被计为刻意空输出会虚高）。
> - EgoLens 最强端到端：gIoU **0.700** / cIoU **0.486**；相对最强商用 VLM 管线，**gIoU +14.7** 而仅用 3B 紧凑基座。
> - 规划上 EgoLens 拿下最高 CSR 与 Coverage、第二的 F.S.Sim 与 Sem.F1（Gemini 在后两者第一）。
> - 非空目标 NE-mIoU：Mo 0.668 / Ms 0.614 / Md 0.568，说明聚合性能不靠"空 mask 识别"刷分。

### 5.3 表 3：解耦"下一步推断"与"grounding"（原文事实）

| Method | w/ Ref.Step gIoU/cIoU | w/o Ref.Step gIoU/cIoU |
|--------|----------------------|------------------------|
| Sa2VA | 0.394 / 0.348 | 0.320 / 0.244 |
| UniPixel | 0.487 / 0.216 | 0.330 / 0.123 |
| LENS | 0.246 / 0.239 | 0.197 / 0.208 |
| OMG-LLaVA | 0.204 / 0.142 | 0.050 / 0.333 |
| Claude 4.8+SAM2 | 0.463 / 0.332 | 0.292 / 0.233 |
| GPT 5.5+SAM2 | 0.637 / 0.485 | 0.478 / 0.339 |
| **EgoLens-Seg** | **0.839 / 0.668** | —（不适用） |

> [!note] 原文解读
> 去掉参考步后所有基线 gIoU 均下降，说明**下一步推断仍是主要误差来源**。EgoLens-Seg 在给定参考步下达 0.839 gIoU / 0.668 cIoU（最佳 grounding）。

### 5.4 表 2：EgoAfford-Real 零样本（原文事实）

| Method | gIoU | cIoU | F.S.Sim | Sem.F1 | CSR | Cover. |
|--------|------|------|---------|--------|-----|--------|
| LENS* | — | 0.275 | 0.245 | 0.123 | 0.294 | 0.159 |
| Claude 4.8+SAM2 | 0.377 | 0.214 | 0.401 | 0.377 | 0.608 | 0.473 |
| GPT 5.5+SAM2 | 0.569 | 0.450 | 0.594 | 0.451 | 0.535 | 0.439 |
| **EgoLens** | **0.666** | **0.455** | **0.631** | 0.426 | 0.609 | 0.481 |
| EgoLens-Seg+GPT5.5† | 0.606 | 0.329 | — | — | — | — |
| EgoLens-Seg+GT† | 0.724 | 0.472 | — | — | — | — |

> [!note] 原文解读
> - 未用真实图训练，EgoLens 仍取最佳 gIoU/cIoU 0.666/0.455，**gIoU +9.7** 超 GPT5.5+SAM2；cIoU 接近（0.455 vs 0.450）说明 GPT5.5+SAM2 在聚合像素重叠上仍有竞争力，但 EgoLens 样本级预测更一致。
> - 相对生成测试集（0.700/0.486）仅小幅掉到 0.666/0.455，迁移差距 modest。
> - EgoLens-Seg 把 GPT5.5 预测步换成 GT 步，gIoU/cIoU 从 0.606/0.329 → 0.724/0.472，再次印证下一步推断在真实观测上仍是重要误差源。

---

## 6. 结论与贡献（原文事实）

三项贡献：
1. 把多步任务导向 affordance grounding 形式化为**联合"剩余计划生成 + 直接物体/工具/目的地部件级 grounding"**，并在 EgoAfford 基准中实例化。
2. 提出 **EgoLens**——单次前向同时生成剩余计划并预测三路角色专属成分 mask 的参考 MLLM。
3. 在生成与真实观测上做广泛评测，刻画 EgoAfford 提出的规划与 grounding 双重挑战，建立强参考结果。

> [!summary] 价值定位
> EgoAfford + EgoLens 为"多步桌面任务中联合研究感知与规划"提供了基础。

---

## 7. 作者局限（Limitations，原文事实）

- **数据侧**：EgoAfford 主要为**生成式**，相较自然环境含更少无关杂物，存在视觉保真度与场景复杂度的**域差距**；静态图像评测无法捕捉运动可行性、接触动力学或闭环成功。
- **模型侧**：EgoLens 用**单一有效路径 + teacher-forced 规划 trace** 训练，对"替代解法"监督有限；自生成计划的误差可能在推理中累积。

---

## 8. 批判性分析 / 分析者推演

> [!warning] 推演（非原文结论，待商榷）
> - **生成数据的真实性风险**：核心结论建立在 GPT-Image-2/FLUX 合成图 + GPT-5.5 自动标注上，尽管有人工核验，但"任务状态一致性""部件 mask 质量"在合成域上未必迁移到真实杂乱桌面；真实集仅 102 图，统计置信度有限。
> - **指标对"缺失成分"敏感**：gIoU 对空 mask 识别敏感，而商用 VLM 管线被计为"能回三 mask"可能受益于其接口设计；开源基线因接口限制被豁免 gIoU，跨组比较存在口径不一致。
> - **3B 基座的"强参考"是否公平**：EgoLens 为任务专门训练（域内），而商用 VLM+SAM2 是零样本管线，二者可比性需谨慎；EgoLens-Seg + GT 步的 0.724 gIoU 才更接近纯 grounding 上限。
> - **未做闭环/执行验证**：只评视觉 grounding 与计划文本相似度，未验证机器人实际能否据此完成任务（作者未来工作提及）。

---

## 9. 未来方向（Future Work，原文事实）

- 将**角色结构化 grounding 与具身执行**连接：识别并定位 Mo/Ms/Md 后，它们可作为可复用操作技能的空间接地参数，支持**agentic 替代端到端 World Action Models / VLA 策略**：高层 agent 调用 `pour(source, destination)`、`cut(target, tool)`，低层控制器处理具身运动并返回观测用于重规划。

---

## 10. 术语表

| 术语 | 定义 |
|------|------|
| Affordance | 环境提供给智能体的行动可能性（功能性部件上的可操作区域） |
| Egocentric Referring Segmentation | 以第一人称观测为依据、按语言/角色指代分割像素区域 |
| Direct Object (Mo) | 被操作的直接物体（徒手或经工具） |
| Instrument (Ms) | 握持中作用于直接物体的工具；徒手时缺失 |
| Destination (Md) | 转移动作的目标位置/物体；无物质转移时缺失 |
| Part-level mask | 仅勾勒功能性部件（刀刃/壶嘴），排除把手等纯握持结构 |
| Admissible candidate | 当前状态下可推进任务的下一步动作（含粗动作的细化前提） |
| CSR (Constraint Satisfaction Ratio) | 优先级约束满足比例，衡量计划步骤顺序正确性 |
| Semantic F1 | 基于 cross-encoder 相似度的计划步骤匹配精确率/召回率调和均 |
| NE-mIoU | 仅对 GT 非空样本平均的 mIoU，隔离"缺失成分识别"的影响 |
| EgoAfford-Real | 102 张人工实拍、26 任务的零样本泛化测试集 |
| Teacher forcing (planning trace) | 训练时用参考计划作前缀，其 token 不计入 LM loss 但可见 |

---

## 11. 与知识库其他成员的关联

- **相对单步/单目标 referring segmentation（LISA、Sa2VA、LENS、OMG-LLaVA、UniPixel）**：EgoAfford 把"给定表达式→mask"扩展为"给定任务+观测→剩余计划+三角色部件 mask"，直接以这些模型为基线并指出其不足（见 [[LISA-3D]] 的 `<SEG>`-token 范式讨论，本文明确其不适合三成分形式化）。
- **相对多物体 affordance 数据集（WorldAfford [1]、SeqAfford [32]、AGD20K [16]、InstructPart [26]、AffordanceLLM [20]）**：这些扩大目标/时间范围但未**联合**"语义对齐第一人称观测 + 剩余计划推断 + 角色专属部件 grounding"，EgoAfford 补足这一组合。
- **相对 3D affordance（3DAffordSplat、Aff3DFunc、PointGS 等本库笔记）**：EgoAfford 是**纯 2D 第一人称 + 语言计划**路线，未涉及 3D 点云/3DGS，可作为 2D 端对比锚点。

> [!note] 流派定位（待用户确认归类）
> 本文属"**2D 第一人称任务导向 affordance + MLLM 联合规划/分割**"一支；与本库中 LISA-3D（3D 提升）、3DAffordSplat（3DGS）等构成 2D vs 3D 对照。

---

## 12. 延伸阅读（关键引用，原文事实）

- [1] Chen et al. 2024. **WorldAfford**: Affordance grounding from natural language instructions. ICTAI. （多物体 affordance，未联合规划）
- [16] **AGD20K**: large-scale affordance grounding from exocentric human–object interactions.
- [20] **AffordanceLLM**: 用 VLM 世界知识做 affordance grounding.
- [26] **InstructPart**: 带指令推理的任务导向部件分割.
- [29] **RAGNet**: 推理式 affordance 分割扩展到大规模语料.
- [32] **SeqAfford**: 把复杂指令分解为 3D 点云上的 affordance mask 序列.
- [12] **LISA**: reasoning segmentation 开山，`<SEG>` token 连接 MLLM 与 mask decoder.
- [38] **LENS**: CoT 作为推理强线索的 referring segmentation（EgoLens 基座）.
- [22] **SAM2** / [24] **Grounded-SAM**: 本文 mask 解码与初始标注基础.
- [5] Gibson 1979. *The Ecological Approach to Visual Perception*（affordance 起源）.

---

## 13. 个人批注

> [!note] 思考
> （留白，待你补充：与自己的研究问题如何对接？是否准备用 EgoAfford 数据/协议做对比？）

> [!warning] 疑问
> - 真实集 102 图是否足以支撑"零样本泛化"结论的统计显著性？
> - 三路 decoder 是否会造成角色间耦合/冲突（如 Mo 与 Ms 在同一物体时）？
> - teacher-forced 单路径训练是否会抑制多解规划？

---

## 14. Active Recall

- **Q：EgoAfford 把 affordance grounding 从单步单目标扩展成了什么形式化？**
  A：联合"剩余计划生成 + 下一步三角色（直接物体 Mo / 工具 Ms / 目的地 Md）的部件级 grounding"。

- **Q：Mo / Ms / Md 分别是什么？何时为全零 mask？**
  A：Mo=被操作的直接物体；Ms=握持中作用于 Mo 的工具（徒手时缺失）；Md=转移动作的目标（无物质转移时缺失）；缺失即对应全零 mask。

- **Q：为何 EgoLens 用三路并行 decoder 而非 `<SEG>`-token 范式？**
  A：`<SEG>` 范式依赖正确文本生成决定 mask 数量，不适合固定的三成分形式化；三路 decoder 让"mask–角色绑定"成为架构级约束。

- **Q：表 1 中 EgoLens 相对最强商用 VLM+SAM2 管线提升多少 gIoU？基座多大？**
  A：gIoU 0.700，相对 GPT5.5+SAM2（0.476）提升约 +14.7 点；基座仅 3B Qwen2.5-VL。

- **Q：为什么表 3 去掉参考步后所有基线 gIoU 都下降？说明什么？**
  A：说明"下一步动作推断"是主要误差来源，grounding 性能严重依赖正确的下一步识别。

- **Q：EgoAfford-Real 是什么、规模多大、是否参与训练？**
  A：102 张人工实拍、26 任务的测试集，与 2000 生成场景不相交，零样本评测、未训练。

- **Q：论文作者自陈的两类局限是什么？**
  A：数据侧——生成式图像有域差距、静态图不捕捉运动/接触/闭环；模型侧——单路径+teacher forcing 训练对替代解法监督有限、误差会累积。

- **Q：未来工作想把 EgoLens 的 grounding 接到什么上？**
  A：具身执行——把 Mo/Ms/Md 作为可复用技能（如 `pour(source,destination)`）的空间接地参数，形成 agentic 替代端到端 VLA 的方案。

---

## 15. 原文定位（页码，基于提取文本）

- P1：标题、作者、摘要、Figure 1（任务导向 affordance 形式化图示）。
- P2：Figure 2（EgoAfford & EgoLens 总览）、贡献三点、Related Work 开头。
- P3：Problem Formulation（3.1）、Task Metadata Generation（3.2）、Annotation Pipeline（3.3，含 0.696/0.523 提案质量）。
- P4：Benchmark Protocol（3.4，划分、admissible candidates、EgoAfford-Real、metrics）。
- P5：EgoLens 架构（4.1）开头、Figure 3。
- P6：EgoLens 训练（4.2）、Experiments 设置、Table 1。
- P7：Table 3（解耦推断与 grounding）、Table 2（EgoAfford-Real）、Figure 5。
- P8：Discussion（Limitations / Future Work）、References 开始。

> [!warning] 提示
> 本笔记由 AI 据 PDF 文本提取整理，**数据/引用以原文为准**；带 [待核对]/[原文事实]/[推演] 标注处请回看原文确认。图表数据（Fig.5 等）需人工目视核对。

（项目页：egoafford.github.io）
