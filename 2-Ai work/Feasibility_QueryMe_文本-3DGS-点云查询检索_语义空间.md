---
title: "QueryMe 能否接入当前 3DGS 框架：文本-3DGS-点云查询检索 + 语义空间构建"
type: feasibility
topic: queryme-integration
created: 2026-08-14
source: WebSearch（CVPR2026 openaccess + papernotes）；本地 Methodology_Affordance_Semantic_Space_Construction.md；本对话既有 Feasibility_稀疏PC-3DGS-CAM-SAGE架构.md
confidence: medium-high（QueryMe 方法经官方摘要+papernotes 双源核对；与 Methodology 的对接为分析推演）
related_notes:
  - "2-Ai work/Feasibility_稀疏PC-3DGS-CAM-SAGE架构.md"
  - "2-Ai work/Methodology_Affordance_Semantic_Space_Construction.md"
  - "2-Ai work/SAGE - Point Cloud as a Foreign Language for Multi-modal Large Language Model.md"
tags: [queryme, open-vocabulary, affordance, query-retrieval, 3dgs, semantic-space, 路线集成]
---

# QueryMe 接入当前 3DGS 框架的可行性

> 用户问题：QueryMe 把 2D HOI 经单目前馈重建映射到 3D，再用「文本–HOI重建–点云」组合查询把功能区域当「查询命中」而非「分类」来定位；这套范式能否构成我们框架的 **文本-3DGS-点云组合查询检索**？同时能否按 Methodology_Affordance_Semantic_Space_Construction 构建语义空间？

---

## 0. QueryMe 实测方法（CVPR 2026，赵伟宇等，哈工大）

| 环节 | 实际做法 | 备注 |
|---|---|---|
| 2D→3D HOI 重建 | **VGGT 前馈**（图像→相机/深度/点云/轨迹） | 产出**点云**，非 3DGS；单目、可能无度量/有噪声 |
| 目标物体表示 | **PointNet++** backbone 吃目标**点云**（PIADv2 干净点云） | 与 HOI 重建是两套点云 |
| 查询结构 | 多模态记忆 `M={T'(文本), H'(3D HOI), P'(目标点云)}` | query 与 FPS 点一一对应 |
| 注意力顺序 | `T'→H'→P'` cross-attention（先文本先验→3D HOI 线索→点云中间特征），coarse-to-fine | 末端自注意抽象物体几何 |
| 输出 | per-point heatmap `ω=σ(...)` | **focal + Dice，无显式类别监督** → 「查询命中」非分类 |
| 评测 | PIADv2：Seen / Unseen Object / Unseen Affordance；AUC/aIoU/SIM/MAE | Unseen Affordance AUC +4.19% |

**关键定性**（papernotes）：QueryMe = 「把 affordance 重述为多模态查询检索 + 用前馈重建做 2D→3D HOI 投影」，组合新颖、单项是成熟模块拼装；相对 3D-AffordanceLLM/DAG 的差异是**不依赖生成式渲染、直接在重建 3D HOI 空间做轻量查询**，结构更紧凑。

---

## 1. 问题一：能否构成「文本-3DGS-点云组合查询检索」？ → ✅ 可行，且是自然的 ④ 检索头

### 1.1 表示对齐（3DGS 即富特征点云）
我们框架的 3D 表示是**语义 3DGS**（μ=位置，{opacity,scale,rot,SH,CAM语义} = 特征）。按既有结论（Feasibility 文档 §4.1），一个高斯 = `(μ, attrs)` 的富特征点，可直接进 PointNet++ / cross-attention 查询机。故把 QueryMe 的 `P'` 换成我们的**语义 3DGS** 即可：
- `P'` ← 语义 3DGS（μ 当点、语义属性当通道），或 3DGS + 3DAffordSplat 原始点云两者并喂。
- `T'` ← 冻结 CLIP 功能文本 / Methodology 的 T_A 概念池。
- `H'` ← 3D HOI 证据：QueryMe 原用 **VGGT** 重建；我们路线①有 **MLLM 意图嵌入**（可投影到 3D）作互补来源。

### 1.2 范式对齐（「查询命中」正是我们要的）
QueryMe 输出 heatmap + focal/Dice、无类别头 → 天然满足 Methodology **坑1（别用闭集 K 头）** 与 **B1（开放表征）**。这恰好替代我们原计划里「token-to-point 密集解码器」作为 ④ 的**检索/定位头**，且比纯解码器多一层「跨模态查询」机制（文本+HOI+几何联合检索），更贴合 OV 设定。

### 1.3 结论
**QueryMe 的查询范式可直接作为我们框架的 ④ 检索头**，组合成「文本(T')–语义3DGS(P')–HOI/意图(H')」三模态查询检索。这是表示升级（连续语义场 vs 裸点采样），不是简单换点云。

---

## 2. 问题二：能否按 Methodology 构建语义空间？ → ⚠️ QueryMe 本身不建空间，但二者可**组合**

### 2.1 关键澄清：QueryMe ≠ 语义空间构造器
Methodology 的语义空间 = **(affordance-type 轴) ⊕ (object/context 轴)**，靠 Step 1–5 训练出来：
- Step 1 功能文本池 T_A（多样）
- Step 2 多视角一致 2D 语义提升（DINO/CLIP/VLM）到 3D
- Step 3 对齐冻结功能文本（可分性来源）
- Step 4 对比损失：类内多样（拉向 T_A 集合）+ 类间可分（推离 T_B + 原型分离 margin）
- Step 5 关系/层级建模解重叠

QueryMe 只做**前向查询检索**，不维护、不训练这样一个结构化、可探测的语义空间。它的 heatmap 监督**无类别标签**，也就**没有 Step 4b 的原型/margin 分离**——即 QueryMe 头本身不强制类间可分。

### 2.2 正确组合方式（推荐）
把两者**串成两段**，而非等同：
- **Stage A（按 Methodology 建空间）**：用 SAGE tokenizer（离散语言对齐词汇表）+ CAM 蒸馏 DINOv2 + 多视角一致性 L_geo，按 Step 1–5 把 3DGS 高斯基元特征训练成「开放、多样、可分」的语义空间（保留类内流形宽度 + 类间原型分离）。
- **Stage B（QueryMe 查询头做检索接口）**：在 A 训好的语义 3DGS 上挂 QueryMe 式多模态查询 cross-attention，推理时给文本/HOI 查询即出 per-point heatmap。

→ 即：**Methodology 训练「空间」，QueryMe 提供「查询接口」**。二者互补，不冲突。

### 2.3 必须解决的一个张力
QueryMe 头**无类别监督** → 若直接拿来当 ④ 且丢掉 Stage A 的对比训练，会**丧失 Methodology 的「类间可分」**。补救（任选）：
- 保留 Stage A 的对比/原型分离损失作为**特征预训练正则器**，再接 QueryMe 头微调；
- 或在 QueryMe 训练里**补一个跨 affordance 对比项**（把 T_A/T_B 相似度差拉大），把 Step 4b 塞回去。

---

## 3. 与路线①+②（GEAL-based）及既有框架的关系

| 组件 | 来源 | 在本组合中的角色 |
|---|---|---|
| 语义 3DGS（μ+语义属性） | 我们框架（3DAffordSplat + CAM + SAGE tokenizer） | 被查询的「空间」载体 |
| T_A 功能文本池 | Methodology Step 1 / 路线① MLLM | 查询的文本先验 + 可分轴 |
| HOI/意图证据 H' | QueryMe(VGGT) 或 路线①(MLLM intent) | 查询的几何/意图线索 |
| 查询 cross-attention 头 | QueryMe（移植） | ④ 检索/定位头（替代 token-to-point 解码） |
| 空间训练损失 | Methodology Step 4 | 保证多样+可分，防 QueryMe 头丢类间可分 |

---

## 4. 新颖性自检（对照用户硬规则「换编码器/换表示不构成新颖性」）

- **危险点**：若被读成「把 QueryMe 的 P' 从点云换成 3DGS」= 表示替换，一句话驳回（类似 LangSplat/Feature3DGS 领地）。
- **防守点（范式级）**：
  1. 被查询的是**按 Methodology 训练出的、语言对齐+离散词汇+结构可分**的语义 3DGS 场，而非裸点云；
  2. 查询接口同时消费**冻结功能文本(T_A) + 3D HOI/MLLM 意图(H') + 语义 3DGS(P')** 三模态，且 coarse-to-fine 顺序可学习；
  3. 用 **Methodology 的对比/原型分离**补足 QueryMe 头缺失的类间可分，构成「可探测语义空间 + 查询检索」统一范式。
- 相对 QueryMe 的增量：QueryMe 查**裸点云**、无结构化语义空间、HOI 只靠 VGGT 单目前馈；我们查**语义蒸馏+语言对齐的 3DGS 场**并显式建空间 → 能力加法非替换。

---

## 5. 必须正视的实现/评测差异

1. **评测协议不对齐**：QueryMe 在 **PIADv2**，我们框架在 **3DAffordSplat seen/unseen**。要声明「兼容 QueryMe 范式」须**加 PIADv2 协议**做主表对照（尤其 Unseen Affordance AUC），否则无法公平比较。
2. **HOI 证据来源**：QueryMe 用 VGGT 单目重建 HOI（可能度量缺失/噪声）；我们若用路线① MLLM 意图，来源不同，需在消融里区分「VGGT-HOI vs MLLM-intent」贡献。
3. **3DGS 体素性**：μ 非表面 + 可能含 floater → 进 PointNet++/查询机前宜用物体 mask 只取物体高斯（3DAffordSplat 已带 mask）。
4. **表示选择**：P' 用 3DGS(μ+attrs) 直接（优雅）还是 3DGS+原始点云并喂（更字面「3DGS+点云」）——建议前者，3DGS 已包含点云信息。

---

## 6. 一句话结论

- **问题一（文本-3DGS-点云查询检索）**：✅ 可行。QueryMe 查询范式天然可作我们框架的 ④ 检索头——其 `P'`(目标点云) 保留，`H'`(VGGT 单目 HOI 重建) 由我们的**语义 3DGS(G')** 取代，`T'` 承载文本，且「查询命中」满足开放表征要求。详见 §7 用户精炼方案。
- **问题二（按 Methodology 建语义空间）**：⚠️ QueryMe 本身**不建**空间；正确做法是 **Methodology 训空间 + QueryMe 做查询接口** 两段串联，并补回 QueryMe 头缺失的类间可分（对比/原型分离正则）。
- **总定位**：组合 = 「Methodology 训练的可探测语义 3DGS 场」+「QueryMe 式三模态查询检索头」，新颖性落在**语义蒸馏+语言对齐的 3DGS 场被结构化查询**，而非任一单模块替换。

---

## 7. 用户精炼方案（2026-08-14 15:46）：M={T'(文本), G'(3D GS), P'(点云)}，弃用单目前馈重建

### 7.1 方案定义
- 查询记忆改为 **M={T'(文本), G'(3D GS), P'(点云)}**：丢弃 QueryMe 的 `H'`(VGGT 单目 HOI 重建)，改由我们的**语义 3DGS(G')** 进查询记忆。
- 3DGS 在 **Methodology 语义空间**中训出语义特征（Stage A：SAGE tokenizer 离散词表 + CAM 蒸馏 DINOv2 + 多视角一致性 L_geo + Step 4 对比(类内多样/类间可分)）；推理时 QueryMe 式 cross-attention 在 {T', G', P'} 上检索，出 per-point heatmap 定位功能区域。
- 推理期**零额外图像**（不需 HOI 图、不需 VGGT），与 GEAL「零图像高效」卖点一致。

### 7.2 可行性判定：✅ 可行，且比保留 H' 更自洽
1. **G' 可直接进查询记忆**：高斯=(μ,attrs) 即富特征点（既有结论），G' 作为「语义记忆库」{(μ_i, f_i^semantic)} 被 cross-attention 消费，与 P'(几何记忆库) 并列。
2. **P' 与 G' 分工（必须讲清，防「为何两者都要」）**：
   - `P'` = 原始目标点云（PointNet++ 几何/结构线索，同 QueryMe 的 P' 角色）；
   - `G'` = 同一物体的**语义 3DGS**（每高斯带 Methodology 训出的功能特征，原始点云没有）；
   - 即「几何 + 语义」双记忆，严格富於 QueryMe 的「HOI + 目标」（后者无显式语义空间）。
   - 定位：query 点=FPS 采样的 P' 点（heatmap 输出坐标帧），语义查表走 G'（按 μ 与 P' 点对齐）。
3. **注意力顺序**：QueryMe 原 T'→H'→P'；现建议 **T'→G'→P'**（文本→语义先验→几何精修，coarse-to-fine），须论证。

### 7.3 关键权衡：丢了 QueryMe 的「类比推理」，用「语义空间检索」补偿
- QueryMe 的 `H'` 提供**交互几何类比**（看人怎么坐→迁移到目标椅），是其 Unseen 设定 +4.19% 的来源之一。
- 弃用 H' 后，泛化改靠：**(a)** Methodology 的多样 T_A 池 + 冻结 CLIP 文本几何（OV 泛化）；**(b)** CAM 蒸馏 DINOv2 的功能区域线索。即「外部类比」→「内部语义场检索」。
- **经验性风险**：Unseen Affordance 是否仍能追平 QueryMe 的 +4.19%，取决于语义场质量——须在 **PIADv2** 上实测验证（这是组合成立的关键实验）。

### 7.3b 直接回答 Q1：M={T',G',P'} 是否缺失了 2D HOI 图像的交互语义？
**结论（诚实版）：在「推理期」确实丢失了 HOI 的几何交互先验；但可通过「训练期把交互语义灌进 G'」来补偿——补偿是否充分是经验问题，不是自动成立。**

- QueryMe 的 `H'` 是一个**3D 重建的几何 HOI**（人-物接触配置、肢体/手部接触几何），它是**空间级**交互先验：「手抓的位置」「身体压上去的姿态」。这正是它 Unseen Affordance +4.19% 的来源之一。
- 我们的 `M={T',G',P'}`：
  - `T'` 携带交互语义是**类别级/符号级**（"人坐在椅子上"）——告诉你"做什么 affordance"，但不告诉你"相对身体接触几何在哪"。
  - `P'` 是纯几何。
  - `G'`（语义 3DGS）是否含交互语义，**完全取决于它训练时喂了什么**：
    - 若 G' 只在 3DAffordSplat 的**静态 affordance mask** 上训 → 它学到"座位区域在哪"，但**没学到"人坐时的身体接触几何"** → 此时替换 H' 确实**静默丢失了空间交互先验**。
    - 若 G' 训练时显式注入 HOI/contact 语义 → 交互先验被**内化进 3D 场**，推理期无需 H' 也能检索。
- **所以真正的修复不是"用 G' 替 H' 就完事"，而是**：
  1. **训练期**按 Methodology Step 2/5 把 HOI 交互语义灌进 G'（见 §7.6 的 L_interact）；
  2. 路线① **MLLM 意图教师**承载 `(part, action)` 关系型交互文本（Step 5 的解重叠/关系消歧处），作为 T' 的互补来源。
- 这意味着对 **2D HOI 图像的使用从"推理期"挪到"训练期"**——与 GEAL「训练用重、推理零图像」的一致，但必须明文写清：推理零图像 ≠ 训练不用 HOI；否则审稿人会指出"你其实也需要 HOI 监督来训 G'"。
- **残留风险**：内部化(进 G') vs 外部化(作 H') 哪个对 **Unseen Affordance** 更优，是经验问题 → 必须在 PIADv2 Unseen Affordance 上实测（同 §7.3 关键实验）。若追不平 +4.19%，说明空间交互先验难以纯靠语义场内化替代，需回退"保留轻量 H'（仅训练期，或推理期用 3DAffordSplat 自带接触/姿态先验）"。

### 7.4 新颖性再自检（对照用户硬规则）
- 危险：被读成「把 QueryMe 的 H' 从 HOI 重建换成 3DGS」= 查询证据源替换， borderline 驳回。
- 防守（范式级）：(1) G' 是 **Methodology 训出的(多样+可分)语义场**，非原始重建；(2) 查询在「结构化语义场 + 几何」双记忆上做 OV 检索，**无需 HOI 样本**；(3) 概念上从 QueryMe 的「外部类比定位」重构为「内部语义场查询」——这是范式 reframing，非单模块替换。
- 相对 QueryMe 增量：查的是**语义蒸馏+语言对齐的 3DGS 场**而非裸点云/HOI，且显式建 Methodology 空间（QueryMe 无此空间）→ 能力加法。

### 7.5 待决设计点
- P' 是否真必需？若 G' 已含 μ（点）且特征够强，可只用 G'（P' 退化为查询坐标帧）；但保留 P' 作几何 backbone 更稳、更像 QueryMe。
- 注意力顺序 T'→G'→P' vs T'→P'→G' 需消融。
- 补回类间可分的对比项（Step 4b）仍须加，否则 QueryMe 头丢可分性。

### 7.6 直接回答 Q2：语义 3DGS（G'）怎么实现（具体管线）

> 语义 3DGS = 标准 3DGS（每高斯 μ, Σ/scale, 四元数 rot, opacity, SH 外观）**+ 新增可学习逐高斯语义属性 f_i ∈ R^d**（d≈256/512，语言对齐轴用 SAGE 时须对齐 codebook 维）。G' 即 {(μ_i, f_i)}。

**训练管线（按 Methodology Step 1–5 落到 3DGS，CAM/IAAO 路线）：**

1. **初始化 3DGS**：来自 3DAffordSplat（已重建）或 PointGS stage0（稀疏 PC → 多视角渲染 → 3DGS 拟合）。几何+外观按 GEAL 约定**冻结**（只训语义属性）。
2. **加语义属性**：每高斯挂可学习 f_i（随机或 backbone 初始化）。
3. **可微语义 splatting（CAM 核心）**：把 f_i 当"额外特征通道"，与 SH→RGB 共用同一套 α-合成光栅化器 → 渲染出多视角语义特征图 F_2D^sem。全程可微（PointGS/IAAO 路线，非原样复用 GEAL 的 CAM——GEAL 原 CAM 把 PointNet++ 特征当属性，这里改成高斯原生语义属性）。
4. **2D 教师特征提取（Step 2 提升）**：从**同一组多视角图像**抽：
   - DINOv2 → 部件/结构特征；
   - CLIP 图像 → 外观/情境；
   - VLM（LISA/Fun3DU 式）功能掩码 → 带语言锚点的功能区域；
   - **交互语义来源（关键，回应 Q1）**：**训练期**额外喂 **2D HOI 图像 / contact map**，抽 HOI/接触特征 → 让 G' 内化空间交互先验（推理期不用）。
5. **监督损失**：
   - `L_distill`（CAM）：逐像素 ||F_2D^sem − 教师特征||（cosine/L2），配 **L_geo 多视角一致性**（LISA-3D）防语义抹糊；
   - `L_text_align`（Step 3/4）：3D 区域特征 f_r（由 μ + T_A attention 池化）对**冻结 CLIP** 做对比——拉向 **T_A 集合**（类内多样，非质心）、推离 T_B、加原型 margin（Step 4b 类间可分）；
   - `L_interact`（**新增，补 Q1 缺口**）：HOI/contact 一致性蒸馏 + (part,action) 关系三元组（来自路线① MLLM 教师），确保 f_i 携带交互语义而非纯几何；
   - 可选 MI 目标（Methodology Step 4c）：逼 f_r 携带 affordance 信息而非纯几何。
6. **反传**：只更新 f_i（及语义 splat 权重）；几何冻结。
7. **产出**：逐高斯 f_i = 语义 3DGS(G')，满足 Methodology「多视角一致 + 类内多样 + 类间可分」。

**两套实现变体（澄清）**：
- **连续版（推荐先做）**：G' = CAM 蒸馏 DINOv2 + T_A 连续对比对齐（Methodology 原生，不需 SAGE）。
- **离散版**：G' 接 SAGE tokenizer(VQ codebook) → 离散语言对齐几何-词汇表（我们之前 Stage A 规划）。两者不冲突，可并行/级联；离散版 novelty 更强但 codebook 域偏移需重训（见 Feasibility 文档 §4.1）。
- **诚实声明**：Methodology 文档把精确损失形式标为"需结合实现回看原文核对"（Step 4 形式、L_geo 的 warp/深度来源），故上述是**设计稿**而非逐字复刻某文损失；落地前须回看 IAAO/CAMAT/LISA-3D 原文定损失。

### 7.7 小结：两个问题的合并答案
- **Q1（缺交互语义吗）**：推理期丢了 HOI 几何先验；靠"训练期把 HOI/contact 灌进 G'（L_interact）+ MLLM 关系文本"内化补偿，补偿充分性须在 PIADv2 Unseen Affordance 实测验证。
- **Q2（G' 怎么实现）**：标准 3DGS + 逐高斯可学习语义属性 f_i + 可微 splatting 渲染语义图 + CAM 蒸馏 DINOv2/CLIP/VLM + T_A 对比对齐 + （新增）HOI/contact 交互蒸馏，几何冻结、只训 f_i。

### 7.8 如何补「空间交互先验」（L_interact 精确设计，回应 16:05 提问）

> 目标：把 QueryMe 推理期的 3D HOI 几何先验，替换为**训练期内化进 G'** 的「3D 接触先验 + 配置感知原型」。

**0. 数据现实核验（PIADv2，已 WebSearch 双源核对）**
- PIADv2 = **15,213 交互图**（AGD20k/OpenImage，按 affordance 类别标注）+ **38,889 3D 点云**（3DIR/3D-AffordanceNet/Objaverse，每点带 affordance 热图，标注形态 `(2048, 4)=坐标+热图`）。
- **关键约束**：论文原话 "images and point clouds do not require a fixed one-to-one pairing, sampled from different instances" → **HOI 图与点云不逐实例配对**，不能做"按位姿反投影某 HOI 图到某 3DGS"。
- **修正推论**：空间交互先验必须拆成**两路互补信号**，而非单路逐实例投影：
  - **「在哪」(空间定位)** ← 3D 点云 affordance 热图（逐实例，几何侧标签；3DAffordSplat 更给**逐高斯**热图，配对更强）；
  - **「怎么交互」(配置)** ← HOI 交互图（类别级聚合，图像侧信号，不配对）。

**1. 注入 G' 的三项可加损失（L_interact = λ1·L_spatial + λ2·L_cfg + λ3·L_rel）**

- **(a) 空间定位损失 L_spatial（补「在哪」）**：
  - 把 3D 点云 affordance 热图（或 3DAffordSplat 逐高斯 mask）通过**物体 mask + 最近 μ 归属**搬到 G' 高斯上 → 得每高斯热图 `h_i^A`（affordance A）。
  - 监督：在 `h_i^A` 高的 μ 上逼 `f_i` 与该 affordance 的 T_A 文本相似度高、与其它 T_B 低（= 局部化 Step4b）。即把"交互发生在这里"钉进 G'。
  - 这是**逐实例**空间先验，直接给 G' 接触位置几何。

- **(b) 交互配置蒸馏 L_cfg（补「怎么交互」）**：
  - 对每个 affordance A，汇集 PIADv2 中**标注为 A 的全部 HOI 图** → 抽交互特征：接触热图(contact predictor / VLM "where hand touches")、人体姿态(ViTPose/4D-Humans→SMPL 关节)、功能 grounding(Fun3DU/LISA 式)。
  - **类别级聚合**（因不配对）：把该 affordance 所有 HOI 图的交互特征池化为**类别级交互表示 φ_A**（均值/注意力池）。
  - 蒸馏：逼 G' 在 affordance-A 区域的 `f_i` 向 φ_A 对齐（与 T_A 文本互补，T_A 给"语义"、φ_A 给"交互配置"）。→ 让 G' 特征携带"人如何做 A"。

- **(c) 关系/部件-动作损失 L_rel（Methodology Step5）**：
  - (part, action) 三元组由路线① MLLM 意图教师给（如 (seat-surface, support-bodyweight)），把"哪部件+哪动作"绑进 `f_i`，解 sit/lean/support 重叠。

**2. 与 QueryMe H' 的精确对照（gain / lose）**
- 得：交互先验**内化**进每高斯特征（含"此处是否被交互、如何交互"），推理零图、零 H'、零 VGGT。
- 失：H' 的**在线类比**——QueryMe 推理时给一张新 HOI 图可即时适配姿态；我们靠训练期覆盖的 T_A 多样 + 配置原型近似。

**3. ⚠️ 结构性诚实限制：Unseen Affordance 上 HOI 先验不可内化**
- Unseen Affordance 拆分的**定义**就是"测试 affordance 类型训练未出现" → 该类型的 HOI 图**不在训练集** → L_cfg 对该类型**无信号**。
- 故 Unseen Affordance 上 G' 只能 fallback 到 **T_A 语言描述 + 几何相似**（T_A 可用 LLM 描述未见交互，这是唯一能泛化的桥）。
- 这正是方案**最大风险点**：QueryMe 的 H' 推理时可吃未见 affordance 的 HOI 图，我们不行。→ **PIADv2 Unseen Affordance AUC 是生死实验**。
-  mitigation（任选）：① 更丰富的 T_A 描述未见交互；② **混合推理**（若查询时附 HOI 图，把它作额外模态喂查询头——代价是破"纯零图推理"，但可在 ablations 里报告"零图 vs +HOI 图"两档）；③ 组合式（未见 affordance = 已见部件-动作组合，靠 L_rel 重组）。

**4. 训练配方（5 阶段细化）**
- Stage 0: 3DGS 拟合（3DAffordSplat / PointGS stage0）
- Stage 1: CAM 蒸馏 DINOv2/CLIP/VLM（纯语义，无交互）
- Stage 2: **+ L_spatial + L_cfg 灌空间交互先验（本步核心新增）**
- Stage 3: T_A 连续对比对齐（类内多样/类间可分）
- Stage 4: QueryMe 查询头 M={T',G',P'} 微调，补回类间可分
- **消融**：w/o Stage2（无空间交互先验）vs w/ → 直接隔离"补空间交互先验"对 Unseen Affordance 的增益；再加"零图 vs +HOI 图查询"两档。

**5. 实现待核对**
- PIADv2 HOI 图是否真含人体/接触（AGD20k 是动作检测集，应含）→ 确认接触提取器选型。
- 3DAffordSplat 逐高斯 mask 与 PIADv2 点云热图的坐标帧对齐（两数据集合并训 G' 需配准）。
- L_spatial 锚定方式：硬归属（最近 μ）vs 软热图加权高斯 splat。
- L_cfg 聚合：均值池 vs 注意力池（建议注意力池，突出典型接触）。

---

## 待回看原文核对 / 开放问题
- QueryMe 的 VGGT 是否也用于目标物体（还是仅 HOI）？影响 P' 是否需额外点云。（精炼方案已弃 H'，此问弱化）
- QueryMe `H'` 的 3D HOI 特征维度 / 与 PointNet++ 特征空间是否对齐（精炼方案已不用 H'，此问弱化）。
- 补回类间可分的对比项具体形式（是否直接复用 Methodology Step 4b 的 T_A/T_B margin）。
- **精炼方案关键实验**：在 PIADv2 上验证「M={T',G',P'} 弃 H'」的 Unseen Affordance AUC 是否追平/超越 QueryMe 原 {T',H',P'}（+4.19%）。
- **Q1 补全项**：G' 训练必须显式用 2D HOI/contact 图（训练期，非推理期）+ 路线① MLLM 意图教师承载 (part,action) 关系文本，否则丢 HOI 空间交互先验；须声明「推理零图像 ≠ 训练不用 HOI」。
- **L_interact 具体形式**：见 §7.8 已给出三项可加损失（L_spatial 空间定位 + L_cfg 交互配置蒸馏 + L_rel 关系）；精确超参/聚合方式见 §7.8.5 待核对。
- P' 与 G' 分工表述（几何 vs 语义双记忆）须在方法段显式论证，防审稿「为何两者都要」。
- 注意力顺序 T'→G'→P' 的消融。
- **G' 实现损失回看**：IAAO 的 3DGS 语义蒸馏、CMAT/LISA-3D 的 L_geo 多视角一致、Methodology Step 4 对比形式的精确原文核对（文档已自标 medium-high、损失形式待核对）。
