---
title: IntentHead 设计卡片
subtitle: 把 MLLM 意图离线蒸馏进轻量 3D 分支的意图嵌入头
type: method-design-card
based_on: HAMMER (arXiv 2603.02329) + GEAL (CVPR 2025) GAFM
route: 路线① (MLLM 意图教师) + 路线③ (GAFM 式意图条件)
date: 2026-08-25
status: design-draft
tags: [affordance-grounding, knowledge-distillation, mllm, intent-embedding, lightweight-3d]
---

# IntentHead 设计卡片

> 一句话：**IntentHead 是坐在 PointNet++ 上方的一个新头，吃"多尺度点特征 + 一句文本"，吐"单向量意图嵌入 f̂_c"；训练时用 MSE 逼它像 MLLM 教师产出的 f_c，训完把 MLLM 拔掉、让 f̂_c 顶替 f_c 进解码器。** 它是路线①（离线意图蒸馏）的载体，也是路线③（GAFM 式意图条件）的落地，更是 ①② 双向耦合的胶水。

## 0. 在流水线里的位置

```
PointNet++ ──{多尺度 F₀…Fₘ}──► IntentHead ──f̂_c──► 解码器(共用) ──► affordance 热图
      ▲                              ▲
  不换编码器                      t = 冻结小文本编码器(替 MLLM)
```

- **① PointNet++ 编码器**：保留 HAMMER 原主干，**不换**，不踩红线。
- **② 冻结小文本编码器**：查询句 T → `t`（RoBERTa / 小 CLIP 文本塔，几 M 参数），推理期替掉 3B MLLM。
- **③ IntentHead（新增）**：本卡片主角，产 `f̂_c`。
- **④ 解码器**：与 HAMMER 共用，只把意图来源从 `f_c` 换成 `f̂_c`，一字不改——这是"推理零 MLLM"的关键。

## 1. 内部三阶段

### A. 文本条件噪声门控（GAFM 式，路线③落地）
- 每个尺度 `Fᵢ` 先 1×1 卷积对齐到工作维 `d_p`，全局池化成尺度描述符 `pᵢ`。
- 用文本 `t` 算逐尺度门控：`gᵢ = ⟨Wq·pᵢ, Wk·t⟩ + σε`（ε 为带噪门控正则噪声）。
- `w = Softmax(g)`，加权上采样聚合：`F_agg = Σ wᵢ·Up(Fᵢ)`。
- **作用**：让"按压"挑细粒度层、"坐"挑粗粒度层——意图驱动尺度选择。

### B. 文本→点跨注意力（意图注入）
- `Q = F_agg`（点问），`K = V = t`（文本答）；`F_att = CA(F_agg, t)`。
- 把查询意图写进每个点特征，等价 HAMMER 后期"点到期意图注意力"的轻量版，但不依赖 MLLM 隐态。

### C. 全局池化 + MLP（产出单向量）
- `f̂_c = MLP( [MaxPool(F_att); AvgPool(F_att)] )` → 单向量 ℝᵈ。
- 单向量才能和 MLLM 的 `f_c`（也是单向量）做 MSE。
- 加一条 `t` 残差短路 `f̂_c += W_t·t`，让头初期不至于忽略文本；**必须进消融**。

## 2. 维度账本（示例）

| 张量 | 形状 | 说明 |
|---|---|---|
| 输入点 `F₀…Fₘ` | `Nᵢ × Cᵢ`（如 2048×128 → 256×512） | PointNet++ 多尺度 |
| 工作维 `d_p` | 256 | 头内部统一通道 |
| 文本 `t` | 768（RoBERTa）/ 512（CLIP 文本） | 投影到 `d_p` |
| **输出 `f̂_c`** | **d = dim(f_c)** | 须与教师对齐，见 §3 |

## 3. 关键决策：蒸馏目标维度怎么对齐

`f̂_c` 必须和缓存的 `f_c` **同维**，否则 MSE 不成立。两条路：

- **方案 a（直白）**：头最后一层线性直接输出 `d_mllm`（Qwen2.5-VL-3B 隐维约 2048）。简单但头偏重、被教师空间绑死。
- **方案 b（推荐）**：Stage 1 给教师加投影头 `P_θ`（线性，输出小维如 256），缓存 `z_c = P_θ(f_c)`；学生 IntentHead 只输出 256 维匹配 `z_c`。
  - 好处：① 头更轻；② 共享维可自由选（CLIP 式对比）；③ `P_θ` 训完即冻结，缓存一次、Stage 2 不再动教师。

> 推荐 b：`L_kd = MSE(f̂_c, z_c)`，其中 `z_c` 是离线缓存的已投影教师意图。

## 4. PyTorch 伪代码

```python
class IntentHead(nn.Module):
    def __init__(self, scales_c, d_p=256, d_txt=768, d_out=256, heads=4, noise=0.1):
        self.align   = nn.ModuleList([nn.Conv1d(c, d_p, 1) for c in scales_c])
        self.phi     = nn.Linear(d_p, d_p)          # 尺度描述符
        self.Wq      = nn.Linear(d_p, d_p)
        self.Wk      = nn.Linear(d_txt, d_p)
        self.txt_proj= nn.Linear(d_txt, d_p)        # t → d_p 供 K/V
        self.ca      = nn.MultiheadAttention(d_p, heads, batch_first=True)
        self.pool_mlp= nn.Sequential(nn.Linear(2*d_p, d_p), nn.ReLU(),
                                    nn.Linear(d_p, d_out))
        self.txt_skip= nn.Linear(d_txt, d_out)      # 残差(可消融)
        self.noise   = noise

    def forward(self, feats, t):
        # feats: list [B, N_i, C_i];  t: [B, d_txt]
        up, gates = [], []
        for i, F in enumerate(feats):
            x   = self.align[i](F.transpose(1,2)).transpose(1,2)   # [B,N_i,d_p]
            xup = interpolate_to_finest(x)                         # 对齐到 N_0
            up.append(xup)
            p_i = x.mean(1)                                        # [B,d_p]
            gates.append((self.Wq(self.phi(p_i)) * self.Wk(t)).sum(-1))
        w   = F.softmax(torch.stack(gates,1) + self.noise*torch.randn_like(gates), 1)
        Fagg= sum(w[:,i:i+1,None,None] * up[i] for i in range(len(up)))   # [B,N0,d_p]
        tkv = self.txt_proj(t).unsqueeze(1)                        # [B,1,d_p]
        Fatt,_ = self.ca(Fagg, tkv, tkv)                           # [B,N0,d_p]
        pooled = torch.cat([Fatt.max(1).values, Fatt.mean(1).values], -1)
        return self.pool_mlp(pooled) + self.txt_skip(t)           # [B,d_out]
```

训练：`loss = λ_aff·L_aff^GT + λ_kd·MSE(f̂_c, z_c)`。
推理：只跑 `PointNet++ + t(小编码器) + IntentHead + 解码器`，MLLM 不加载。

## 5. 两阶段训练配方（GEAL 式）

1. **Stage 1（教师）**：按 HAMMER 原样训练 MLLM+3D 分支（LoRA 微调 Qwen2.5-VL）；同时训投影头 `P_θ`。对全部训练样本离线跑一遍，缓存 `z_c = P_θ(f_c)`（可选再缓存逐点 `f^{3D}`）。
2. **Stage 2（学生）**：只加载 PointNet++ + 小文本编码器 + IntentHead + 解码器，用缓存 `z_c` 当目标训练。MLLM 权重**完全不进 Stage 2 前向**。
3. **推理**：只跑学生。

> 显存：Stage 2 不加载 MLLM，显存只剩 PointNet++ + 小文本塔 + 头，单张 4090 宽松；Stage 1 跑 Qwen2.5-VL-3B 用 4×4090 一次性离线生成目标后即弃。正好绕开"4×4090 训不动 TRELLIS 级补全"的算力约束——蒸馏①与补全②要分开算显存。

### 5.1 Stage 1 离线缓存 `z_c` 实现（工程细节）

**目标**：用冻结的"微调 MLLM + `ψ_c` + `P_θ`"对全训练集跑一遍无梯度前向，把每个样本的 256 维意图嵌入 `z_c = P_θ(f_c)` 按 `sample_id` 存盘；Stage 2 学生按同 id 查表拿目标做 MSE，教师权重全程不进 Stage 2 前向。

**缓存链路**：
```
(I, T) ──MLLM(HAMMER,微调后)──> h ──取 [CONT] 位──> h_[CONT] ──ψ_c──> f_c ──P_θ(冻结)──> z_c ──存盘
```

**(a) 前置（Stage 1 结束时手里有什么）**
- 微调后的 MLLM（Qwen2.5-VL + LoRA，merged 或 load_adapter）
- `[CONT]` 提取逻辑 `ψ_c`（HAMMER 原有）
- **投影头 `P_θ`**（新增，线性 `2048→256`，与教师同训后冻结）——缓存的是已投影 `z_c`，Stage 2 不碰 MLLM 与 `P_θ`

**(b) 单样本前向（复刻 HAMMER 预处理）**
```python
mlm.eval(); psi_c.eval(); P_theta.eval()
with torch.no_grad():
    prompt = build_hammer_prompt(T)          # 复刻 HAMMER 模板，[CONT] 插到正确位置
    h      = qwen2vl(image=I, text=prompt).last_hidden_state   # [B, L, 2048]
    h_cont = h[:, cont_token_pos, :]          # 取 [CONT] 位
    f_c    = psi_c(h_cont)                    # [B, 2048]
    z_c    = P_theta(f_c)                     # [B, 256]  ← 要缓存的目标
```
注意 `z_c` 是**单向量**（非逐点），故学生 IntentHead 输出也是单向量，MSE 直接成立。

**(c) 批处理整份训练集 + 存盘**
```python
cache, loader = {}, DataLoader(train_set, batch_size=8, shuffle=False, num_workers=4)
with torch.no_grad():
    for batch in loader:
        ids, I, T = batch["id"], batch["image"], batch["text"]
        z_c = teacher_intent(I, T)            # 上面的前向, [B,256]
        for i, sid in enumerate(ids):
            cache[sid] = z_c[i].cpu().to(torch.float32)
torch.save(cache, "cache/zc/teacher_zc.pt")   # 单文件字典(最简单)
torch.save({"dim":256, "teacher":"HAMMER-Qwen2.5-VL-LoRA",
            "proj":"P_theta-2048to256", "n":len(cache)}, "cache/zc/manifest.pt")
```
存盘格式：样本数 < 10万、dim 256 用单 `.pt` 字典即可（10万×256×4B ≈ 100MB）；极多样本用 `np.memmap([N,256]` + `id→row` 索引按需 `mmap[idx]`。

**(d) 四个必踩的坑（决定缓存能不能用）**
1. **预处理必须与 Stage 1 训练完全一致**：图像分辨率/归一化/tokenizer/`[CONT]` 位置——任何偏差让 `z_c` 偏移到另一语义空间。直接从 HAMMER 代码 import `build_hammer_prompt` 与 `cont_token_pos`，勿重写。
2. **确定性**：`model.eval()` + 关 dropout + `cudnn.deterministic=True`。Qwen2.5-VL 若开 flash-attn 可能非确定，缓存只跑一次、跑完锁版本，记 `manifest` 教师 hash 以便复现。
3. **id 必须稳定可对齐**：缓存按 `sample_id` 索引，Stage 2 dataloader 用同一 id 体系取 `(P, T)`。推荐 id 用 `(object_uid, affordance, interaction_idx)` 三元组字符串，避免与 shuffle 错位。
4. **只缓存训练集，不缓存测试集**：测试时学生本不该见 `z_c`（否则泄漏教师）；推理只用 `(P, T)` 走学生，不查缓存。

**(e) Stage 2 消费缓存**
```python
teacher_cache = torch.load("cache/zc/teacher_zc.pt")   # 仅训练时加载
def __getitem__(k):
    P, T, aff, sid = load_sample(k)
    return P, T, aff, teacher_cache[sid]               # 查表拿教师目标
for P, T, aff, z_c in student_loader:
    f_hat = intent_head(pointnetpp(P), text_enc(T))    # [B,256]
    loss  = lam_aff * L_aff(afford_decoder(f_hat, ...), aff) \
          + lam_kd  * F.mse_loss(f_hat, z_c)
```
推理时 `teacher_cache` **不加载**，只有 `(P, T) → intent_head → decoder`。

**(f) 显存与耗时（4×4090 现实）**：本步即一次性推理，4×4090 跑 Qwen2.5-VL-3B 批 8，整份训练集（即便几万样本）通常几十分钟到一两小时跑完即弃；之后 Stage 2 所有迭代不加载 MLLM，显存只剩 PointNet++ + 小文本塔 + 头，单张 4090 宽松。

## 6. 与 GAFM / 路线① / 路线② 的关系

- **IntentHead = 改进点③ 的具体实现**：阶段 A 的带噪门控直接搬自 GEAL 的 GAFM，只把条件从 RoBERTa 换成 MLLM 意图嵌入 `t`。
- **它是 ① 的载体**：蒸馏把 MLLM 意图搬进这个头，推理时头自产 `f̂_c`。
- **它是 ①② 的耦合胶水**：若做路线②（生成式补全），把补全先验作为**额外一路输入**进阶段 A（或进解码器），即可实现"意图引导补全 / 补全反哺意图"双向耦合。没有 IntentHead 这个汇聚点，①② 易退化成红线明文驳回的"pipeline stacking"。

## 7. 红线自检

| 审查点 | 判定 |
|---|---|
| 换 3D 编码器了吗？ | 否，PointNet++ 原封不动 |
| 只是"把 MLLM 换成冻结 CLIP"？ | 否——新增的是头 + 文本条件路径 + 训练/推理图改写 + 离线蒸馏范式 |
| 单独成篇有 novelty 吗？ | **无**（memory 已裁定：① 单独被驳回）。必须和 ② 双向耦合 + 不可分解消融才成立 |

## 8. 变体与必做消融

| 变体 | 说明 |
|---|---|
| V1（推荐） | 阶段 A+B+C 全上，条件=t |
| V2（基线） | 去掉 A、B，直接 `MLP(concat[Pool(F₀), t])` → f̂_c（证明 A/B 贡献）|
| V3（逐点） | 阶段 C 不做全局池化，输出 `f̂^{3D}∈ℝ^{N×d}` 模仿 HAMMER 逐点意图（需缓存逐点教师目标，更贵）|

**必做消融**：
1. w/o 阶段 A（门控）→ 验证尺度选择贡献；
2. w/o 阶段 B（跨注意力）→ 验证意图注入贡献；
3. w/o `t` 残差 → 验证头没偷懒只靠文本；
4. w/o 点特征（只喂 t）→ 应塌方，证明几何没被忽略；
5. 若同时引入 ②，做 **桥 × 门控 的 2×2 正交消融**（GEAL 自己没做，审稿人会问"是不是只靠其中一个"）。

## 9. 风险与对策

- **容量落差**（学生 << MLLM）：教师取微调后 HAMMER（非冻结，见 HAMMER 消融：冻结未见 aIOU 7.55 vs 微调 13.71）；报告 drop 并展示被 ② 补回；强调效率换可接受精度。
- **推理仍需文本条件**：诚实表述——丢的是 3B MLLM + 图像管线，保留"一句话 + 点云"经小文本编码器。仍是意图驱动，比 HAMMER 轻两个数量级，且与 GEAL（语言驱动）一致。
- **不可分解消融必做**：去掉意图引导补全 / 去掉补全反哺意图 → 性能应显著塌方，否则退回组件堆叠。

## 10. 待办 / 开放问题

- [ ] 确认 Qwen2.5-VL-3B 实际隐维 `d_mllm`，定 `P_θ` 输出维 `d_out`（建议 256）。
- [ ] 与路线② 补全先验的接入点（阶段 A 加一路 vs 进解码器）待原型验证。
- [ ] 是否引入 V3 逐点目标（更贵但更贴合 HAMMER 的 `f^{3D}`），先以 V1 起步。
