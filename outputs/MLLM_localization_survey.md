# 使用 MLLM（以 Qwen2.5-VL-3B-Instruct 为代表）辅助“定位”的研究综述

> 调研日期：2026-08-25
> 目标：搜集用 MLLM 帮助“定位 / grounding / 检测”的论文，并归纳 MLLM 被使用的范式。
> 说明：文中「论文事实」与「分析者推演」分开标注；凡涉及公式/指标需二次核对的，均已注明。

---

## 1. 为什么用 MLLM 做定位（动机）

MLLM 原生擅长 2D 图像理解，但存在公认的“空间缺口”：
- GPT-4o 在 OCR-free 文本定位任务 IoU 仅 5.26%，但给它 OCR 候选框后飙到 85% —— 瓶颈在**坐标生成**而非推理（TRIG 论文，arXiv 2025）。
- MLLM 缺乏显式空间参考系，视觉编码器偏向全局语义、弱于细粒度空间（VPP-LLaVA）。
- 3D 任务（3DVG / affordance）需要 3D 结构，而 MLLM 只吃 2D —— 因此出现大量“3D→2D 投影 + MLLM 推理”的方案。

关键工程事实（Qwen2.5-VL 技术报告 arXiv 2502.13923）：Qwen2.5-VL 系列**使用输入图像真实尺寸的“绝对像素坐标”表示 bbox/point（非 0–1 归一化）**，原生输出 JSON 格式的 box/point，是做定位任务的天然底座；并提供 3B / 7B / 72B 三档。

---

## 2. MLLM 用于定位的 6 类范式（核心归纳）

### 范式 A：直接端到端 Grounding（MLLM 直接吐坐标）
微调/提示 MLLM，让其在文本或 JSON 中直接输出 bbox / point / 3D box。
- **VPP-LLaVA**（arXiv 2503.15426）：加强空间提示后微调 LLaVA，RefCOCO/+/g SOTA。
- **TRIG**（arXiv 2025）：两种方式——指令式（直接输出坐标 token）vs 嵌入式（图块与文本 token 相似度→热力图）。
- **ECVGPO**（见范式 G，RL 微调小模型精准吐坐标）。
- **SeeGround**（CVPR 2025, arXiv 2412.04383）：零样本开放词汇 3D 定位，**靠 2D VLM 输出 3D box**（经投影对齐）。
- *分析*：Qwen2.5-VL-3B 原生就支持 JSON bbox/point，范式 A 对它几乎开箱即用。

### 范式 B：空间提示 / 坐标参考注入（给模型一个“标尺”）
- **VPP-LLaVA**：global VPP = 可学习的轴状 tensor 叠加到图像（空间坐标系）；local VPP = DETR object queries 提示潜在物体位置。
- **AffordBot** 的 adaptive labeling：把 3D 候选投影成带 ID 的 2D 标注框，避免遮挡堆叠。
- *分析*：范式 B 多为“轻量/不微调”增强，对 3B 小模型尤其划算——不改架构，只改输入渲染/提示。

### 范式 C：3D→2D 投影 + 多步 CoT 推理（弥合模态鸿沟）
MLLM 只吃 2D，于是把 3D 渲染成多视图/全景图，把候选/锚点投影成标注，再用 CoT 分步推理。
- **AffordBot**（NeurIPS 2025, arXiv 2511.10017）：360° 全景图 + 投影 affordance 候选（bbox+ID）→ CoT 三步：主动选视角 → 定位目标 ID → 推断运动类型/轴。**直接用 Qwen2.5-VL-72B 不微调**，SceneFun3D AP25 23.3%（GPT-o1 达 33.4%）。
- **SeeGround**：动态选视角（PAM，按 anchor object 算最佳观察角）+ 视觉提示对齐（FAM，把 3D 坐标投影回 2D 标注）。
- **S²-MLLM**（arXiv 2512.01223）：用**前馈 3D 重建（VGGT 类）**做结构引导，配合结构增强模块（视图内/间注意力 + 多级位置编码），避免低效点云渲染，ScanRefer/Nr3D/Sr3D SOTA。
- *分析*：范式 C 与你的 VGGT 单目重建思路高度同源——都是“3D 结构先建好，再喂给 MLLM”。

### 范式 D：MLLM 作 Proposal / 先验，喂给下游专精模块
MLLM 只做“大致在哪”，最终精度交给几何/SAM/检测器。
- **医疗零样本分割**（MDPI 2025）：Gemini Pro 2.5 出 bbox 先验（JSON）→ SAM 精分割。结论：“框不算精确，但能稳定定位 ROI，有效引导 SAM”。
- **Propose and Rectify**（arXiv 2508.17976）：MLLM 生成篡改初提案 → 取证证据修正 + SAM。
- **城市车辆检测**（Inżynieria Mineralna 2025）：Gemini 2.0 flash 在图像出 2D 框 → KITTI-360 标定投影到 LiDAR → 3D 框 + 时序合并。
- *分析*：范式 D 最贴合“小模型当 cheap prior”的工程现实——3B 不必追求最终精度。

### 范式 E：解耦 Locator + Reasoner（可靠性 / 拒识）
- **ROD-MLLM**：现成 OVD（如 OWLv2）出候选框（low-level locator），Vicuna-7B 做语义校验/选择/输出“None”（high-level reasoner）。OmniLabel 25.3 AP（+9.7），D3 29.7 AP（+13.7）；有效抑制“不存在物体”的幻觉。
- *分析*：若你的 HOI 场景里“手/物体不存在或遮挡”，范式 E 的拒识机制值得借鉴。

### 范式 F：高层规划 + 专精定位器（pipeline 角色）
- **GLOVER++**（arXiv 2505.11865）：用 **Qwen-2.5-VL 作高层 VLM planner**，把“把罐子放进上层抽屉”拆成子目标，需要 affordance 时调用 GLOVER++ 专精模型。
- **EIVA**（arXiv 2508.17922，第一人称 affordance）：用 Qwen2.5-VL + GPT-4o 做“Actor-Verifier 自迭代”——同一 MLLM 既生成 affordance（接触区+运动方向）又当验证器，逐步自精炼。
- *分析*：范式 F 把 MLLM 当“大脑”，与 GEAL baseline 的 planner/grounder 分层思路一致。

### 范式 G：RL / GRPO 把小 MLLM 训成精准坐标生成器
- **ECVGPO / “The Role of Entropy in Visual Grounding”**（arXiv 2512.06726）：发现 grounding 任务的熵“持续偏高”（与推理任务熵坍塌相反），提出 entropy-controlled GRPO。在 **Qwen2.5-VL-3B** 上 overall 83.38（vs GRPO 83.13 / SFT 78.95），LISA-Grounding、RefGTA 等泛化更稳，训练开销几乎为零。另有 LISA grounding test 上 Qwen2.5-VL-3B base 56.82 → ECVGPO/GRPO 65.9。
- *分析*：范式 G 直接证明 **3B 小模型经 RL 微调可达实用 grounding 精度**，是“自己训一个 Qwen2.5-VL-3B 定位器”的最强参照。

---

## 3. 论文清单（速查表）

| 论文 | 任务 | 定位方式（MLLM 角色） | 是否用 Qwen2.5-VL-3B | 代码 |
|---|---|---|---|---|
| VPP-LLaVA (2503.15426) | 2D REC grounding | 空间提示(坐标网格)+微调 | 否（LLaVA/Vicuna） | GitHub: WayneTomas/VPP-LLaVA |
| TRIG (2025) | 文本密集图 grounding | 指令式/嵌入式相似度 | 否（LLaVA-v1.5） | — |
| SeeGround (CVPR25, 2412.04383) | 零样本开放词汇 3D VG | 2D VLM 出 3D box + 动态视角/对齐 | 否（GPT-4V 类） | github.com/iris0329/SeeGround |
| AffordBot (NeurIPS25, 2511.10017) | 3D 细粒度 affordance 推理 | 3D→2D 投影+CoT，**不微调** | **是（72B）** | 待发布 |
| S²-MLLM (2512.01223) | 3D VG 空间推理 | 前馈 3D 重建结构引导+结构增强 | 未明确（结构无关） | 待发布 |
| ROD-MLLM | 可靠目标检测 | OVD 出候选 + LLM 校验/拒识 | 否（Vicuna-7B） | — |
| Propose and Rectify (2508.17976) | 篡改定位 | MLLM 提案 + 取证修正 + SAM | 否 | — |
| 医疗零样本分割 (MDPI25) | 医学分割 | Gemini 出 bbox 先验 → SAM | 否（Gemini 2.5） | — |
| 城市车辆检测 (2025) | 3D 车辆 | Gemini 2D 框 → LiDAR 投影 | 否（Gemini 2.0） | — |
| ECVGPO (2512.06726) | 2D grounding RL | 熵控 GRPO 微调 | **是（3B/InternVL）** | — |
| EIVA (2508.17922) | 第一人称 affordance | Actor-Verifier 自迭代 | **是（Qwen2.5-VL）** | — |
| GLOVER++ (2505.11865) | affordance + 操作 | Qwen-2.5-VL 作 planner | **是（Qwen-2.5-VL）** | — |
| 3D Spatial Understanding/COLD (2412.06613) | 3D 消歧定位 | 增强 MLLM 在干扰物下定位 | 未明确 | github 项目页 |

---

## 4. Qwen2.5-VL-3B 的能力边界（工程须知）

**强项（实测/报告）：**
- RefCOCO 类 grounding 强；原生绝对像素坐标 + JSON bbox/point，利于精确。
- 经 GRPO 微调（ECVGPO）grounding overall 83.38，训练高效。
- 长视频秒级事件定位、动态分辨率、边缘部署友好。

**弱项（需警惕）：**
- 开词汇目标检测（Roboflow100-VL）零/少样本 mAP < 8%，OOD 检测明显落后专精模型。
- 高分辨率专业 GUI（ScreenSpot-Pro）overall 仅 16.1%。
- 对细小/遮挡/强 OOD 目标，直接吐框精度有限。

**结论（分析者推演）：** 3B 适合当「定位先验 / 规划器 / 候选排序器」，若要最终高精度细框，建议 SFT/GRPO 微调或接 OVD/SAM 作双重校验——这与 GEAL 用 CAM/GAFM 对齐几何与语义的思路同源。

---

## 5. 对你项目（QueryMe + VGGT HOI affordance）的可迁移建议

你的管线：HOI 图像 → VGGT 单目重建 → H_raw → 自适应空间注意力（FPS+conf）→ top-k anchors → PointNet++ → H*
MLLM（Qwen2.5-VL-3B）可嵌入的位置：

1. **HOI 图像端粗定位（范式 A/D）**：先让 3B 在 HOI 图上输出 hand / object / contact-region 的 bbox 或 point（JSON 绝对坐标），作为 VGGT 输入裁剪/聚焦区，或作为 H_raw 的空间先验掩码。避开 3B 的 OOD 短板——HOI 场景物体有限、类别可控。
2. **Anchor 语义排序（范式 C/E）**：把 top-k anchors 投影回 2D，给 3B 做“哪个 anchor 对应目标接触点”的语义选择/拒识，替代或增强纯几何的 FPS+conf 加权。
3. **高层 planner（范式 F）**：参考 GLOVER++，用 3B 把“打开抽屉”拆成 affordance 子目标，调用 GEAL baseline 落地，与现有 CAM/GAFM 接口对齐。
4. **自训定位器（范式 G）**：若追求端到端，直接参照 ECVGPO，用 GRPO 把 Qwen2.5-VL-3B 训成 HOI 专用定位器（输出接触区/物体框），成本可控。

**推荐组合（分析者推演，需实验验证）：** 3B 只做「粗定位先验 + anchor 语义筛选」，最终精度交给 VGGT/PointNet++/GEAL 几何与专精模块精修——既发挥 3B 的便宜与开箱即用，又规避其精度短板。

---

## 6. 残留风险 / 开放问题

- 3B 在「手-物遮挡、细小接触区、未见物体」上精度未经验证；建议先在 PIAD/LASO 子集上跑 baseline 量化。
- MLLM 输出坐标的分布可能与 VGGT/PointNet++ 期望的 3D 锚点格式不对齐，需要投影/归一化桥接层（参考 SeeGround 的 FAM、AffordBot 的 adaptive labeling）。
- 若用 RL 微调（ECVGPO），需设计可验证的奖励（IoU/命中接触区），且 3B 的 grounding 熵动态与推理任务不同，需沿用其 entropy-control 设置而非套用推理 RL。

---

*本综述为信息聚合+归纳，具体指标与公式请以原始论文为准；带「分析者推演」标记处为用户项目落地的建议性判断，非论文原文结论。*
