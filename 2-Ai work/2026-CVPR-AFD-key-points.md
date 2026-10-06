# Affordance-First Decomposition for Continual Learning in Video–Language Understanding

> CVPR 2026 (Open Access). 论文要点提取，非逐字翻译。

## 1. 基本信息
- **作者**：Mengzhu Xu（悉尼大学，共同一作）、Hanzhi Liu（UCSB，共同一作）、Ningkang Peng（南京师范大学）、Qianyu Chen（NTU）、Canran Xiao（中山大学深圳校区，通讯）
- **问题域**：视频–语言理解中的**持续学习（Continual Learning）**，模型需面对非平稳的数据、域、查询风格流。

## 2. 研究动机与现有缺陷
现实部署中视频–语言推理面临数据/域/查询风格随时间演化，但现有方法有两个核心缺陷：
1. **稳定性 vs 适应性目标界定不清**：要么用 prompt/adapter 做专门化，要么用蒸馏/拓扑约束保护几何结构，但很少说明"哪些结构应保持不变、哪些应随流适应"——稳定性是偶然的、难以诊断的。
2. **可塑性（plasticity）按启发式分配**：容量与路由通常固定或按任务索引；干扰靠事后合并或全局正则缓解，几乎没有方法用**在线信号**决定"何时/何处"改变。

→ 本文核心问题：**能否把持续视频–语言学习锚定在一个缓慢变化、以交互为中心的基底（substrate）上，从而显式分离稳定性与适应性？**

## 3. 核心思想：AFD（Affordance-First Decomposition）
- **Affordance（可供性）** = 物体–动作规律，跨域/跨任务**变化缓慢**。稳定 affordance 空间可降低下游推理的梯度冲突。
- **两部件分工**：
  - **共享 affordance head（h_ψ）**：把视频映射为时间对齐、可复用的 affordance tokens → 形成缓慢变化的共享基底。稳定性只作用于此。
  - **轻量 LLM-backbone 调度器（g^LLM_φ）**：消费 query 与 affordance tokens，做事件级推理。可塑性与任务专门化全部吸收在此；仅在冲突出现时增长容量。
- **隐私/内存友好**：采用 **question-only replay**（只存历史问题，不存视频）做蒸馏。

## 4. 方法架构
### 4.1 共享 affordance head
- 帧特征 X_t → affordance 分布 P_t(a)=softmax(⟨w_a, z_t⟩/τ)
- 取 **Top-L 稀疏重归一化**分布 q_t，用嵌入表 E_A 构建连续 token：A_t = Σ q_t(a)·E_A[a]
- 投影到 LLM 隐藏空间：(K_t, V_t) = (W_K A_t, W_V A_t)

### 4.2 LLM-backbone 调度器（查询路由 + 冲突感知）
- **逐层路由（per-layer router）**：在适配器层 ℓ，路由器对 query 池化状态 u 计算 LoRA expert 混合权重 α^(ℓ)=softmax(W_r^(ℓ) u)
- **LoRA 混合注入**：W̃^(ℓ)=W^(ℓ)+Σ_j α_j^(ℓ)·B_j^(ℓ)A_j^(ℓ)/s_j^(ℓ)
- **冲突度量与容量增长**：用截断负余弦相似度度量冲突 c_j；当冲突超阈值 τ_c 时**离散增长 LoRA rank**（封顶 r_max）：Δr = min(r_max−r, ⌊γ(c−τ_c)_+⌋)
- **Affordance 交叉注意力**：在适配器层 Q=U W_Q，对 (K,V) 做注意力，融合语言证据与 affordance 证据
- **统一监督**：支持三类查询格式
  - 生成式（open answer）：负对数似然
  - 时序跨度（span）：起止帧预测 + tIoU 损失
  - 步骤序列（step）：序列负对数似然

### 4.3 两个紧凑记忆
- **MQ**：存储多样化历史问题，用于 replay 蒸馏
- **MA**：存储 affordance 原型，用于诊断

## 5. 训练目标（三部分解耦）
- **L_aff（只更新 ψ）**：弱对齐（基于 ASR 动词候选的弱监督）+ 教师一致性（当前分布对冻结上一任务教师分布的 KL），β 平衡
- **L_replay（只更新 φ）**：question-only replay 蒸馏，温度 T_kd，置信度掩码 ρ（仅保留教师最大概率 > ρ 的样本，抑制噪声监督）
- **全目标**：L = L_task + λ_aff L_aff + λ_rep L_replay

## 6. 实验结果（SOTA）
| 基准 | 协议 | 关键指标 | AFD | 对比最佳基线 |
|---|---|---|---|---|
| Domain-Incremental VideoQA | 6 数据集序列 | Avg Acc / Forgetting | **51.6% / −1.8%** | 超 DAM +1.4，遗忘最低 |
| ViLCo-Bench (Ego4D) | query-incremental | MQ R@1@0.5 / NLQ / VQ stAP@0.25 | **29.6% / 20.7% / 18.4%** | 全面最佳（+/+2.5/+2.5/+1.9） |
| Time-Incremental iVQA | 4 时间片 | Avg Acc / Forgetting | **39.5% / −1.6%** | 超 DAM +1.4、Bisecle +1.9 |
| 复杂推理 | CVQA / 11-VideoQA | EM | **62.8 / 67.4** | 略超 VQAGuider、LTR |
| 长视频压力测试 | VideoMME / MLVU | — | **61.7 / 57.9** | 同骨架下优于专用长视频系统，且架构轻量 |

- 评测覆盖：ViLCo-Bench、domain-incremental、time-incremental、复杂多步推理、长视频鲁棒性，且**对任务顺序鲁棒、计算高效**。

## 7. 消融与分析
- **去除 affordance tokens 影响最大**（Avg −2.9，forgetting +1.5）→ 稳定 affordance 空间是核心。
- **去除 router / 固定 LoRA rank** 均有损害 → 实例级路由 + 冲突触发容量增长都重要。
- question-only replay、ASR 弱对齐、教师一致性、Top-L 稀疏、小内存预算，均有正向贡献。
- **稳定性验证**：跨任务 affordance 原型漂移极小（cosine distance 中位数约 0.065–0.078），相邻任务 CKA 高 → 证实"缓慢变化共享空间"假设；verb/action 覆盖随 Top-L 单调上升、L≈8 平台收敛 → 软稀疏混合在不增加调度器容量下编码共现 affordance。
- **案例**：AFD 正确预判交互（Bisecle 关注偶然线索），如预判"卡车将翻倒"而非"人会摔倒"。

## 8. 结论与未来
- 显式分离"稳定的交互中心基底"与"定向适应"，在 ViLCo 与 domain/time-incremental VideoQA 上取得 SOTA 且遗忘显著降低。
- 未来方向：在线 affordance 发现、多传感器扩展。

## 批判性提示（待核对）
- 论文提供 PDF 文本提取，公式/图例以原文为准；上述架构细节源自正文描述，具体超参与初始化（如截断 SVD 初始化新 rank 列）见 Supplementary，未在正文中给出完整数值。
- 实验对比基线多为 2024–2025 年方法，未与同期最新 MoE/router 类方法做更细的容量-精度权衡对比。
