# GEAL × VGGT — HOI Fusion 前馈重建集成方案

## 改造原因

GEAL 基线仅依赖物体点云几何 + 文本查询做 3D affordance 预测，缺少"人手如何与物体交互"这一强先验信号。PIAD 数据集中每张 HOI 图像都包含人手-物体交互场景，经 VGGT 重建后可获得 3D 交互点云 `H_raw`（含置信度 `conf`）。如何将该交互几何注入 GEAL 解码器，同时保证推理期零代价（不依赖 VGGT），是本改造的核心动机。

## 改造原理

采用 **训练期特权提示蒸馏（Privileged Hint Distillation）** 范式：

1. **H_raw → H\***：对每张 HOI 图的重建点云，在线执行 conf 过滤 → FPS 全局采样 → conf 加权重要性 → top-k 锚点选取 → 局部 patch 编码，得到 `k` 个交互特征 token `H* ∈ R^{k×d}`（纯特征，无坐标）。
2. **HOIFusion**：以物体点特征 `P~` 为 query、`H*` 为 key/value 做纯内容交叉注意力，将交互先验注入 `P~` 得到 `P~_aug`。推理期 `H*` 缺失时使用可学习 `null_token` 退化。
3. **Hint KL 蒸馏**：训练时同时跑有 HOI（`ω_full`）和无 HOI（`ω_3D`）两条路径，用 KL 散度约束 `ω_3D` 逼近 `ω_full`，将交互先验内化到解码器权重中。

## 论文出处

- **QueryMe**（CVPR 2024）：跨模态 affordance 学习的 HOI↔物体内容检索机制。本方案的 HOIFusion 内容检索算子与之做法一致（`H*` 不带坐标，纯内容 cross-attention）。
- **QueryMe 的 HOI↔物体步**明确是内容检索而非几何对齐（`H` 与物体点云 `P` 不在同一坐标系），本方案与之对齐，规避"加个 cross-attention"的质疑。
- **本方案的新颖性**在于：(1) 特权提示蒸馏范式——推理期完全丢弃 HOI；(2) 可学习 AnchorEncoder 将通用几何描述子进化为任务相关交互特征；(3) 与 GEAL 解码器的轻量桥接。

## 改造方案

### 数据流

```
PiadDataset.__getitem__
  → 加载 H_raw .npz (xyz[M,3] + conf[M])
  → 返回 (point_input, class_id, binary_mask, questions, affordance_id, gt_mask, h_raw_xyz, h_raw_conf)
    ↓ batch collate (变长 padding)
train_one_epoch
    ↓
model_3d.hoi_handler(h_raw_xyz, h_raw_conf)
    → numpy: conf_filter → downsample → FPS → importance → top-k → patches [B, k, m, 4]
    → AnchorEncoder (可学习, GPU): patches → H* [B, k, d_model]
    ↓
Branch3D.forward(question, point, hoi_feat=H*)
    → Step 1-5c: 标准 PointNet++ + GAFM + 多尺度融合 → P~ [B, d, N]
    → Step 5d: HOIFusion(P~, H*) → P~_aug
    → Step 6: Transformer Decoder → ω_full
    ↓ (并行)
Branch3D.forward(question, point, hoi_feat=None)
    → Step 5d: HOIFusion(P~, null_token) → P~ (近恒等)
    → Step 6: Transformer Decoder → ω_3D
    ↓
L = L_BCE+Dice(ω_full, y)
  + λ_3D · L_BCE+Dice(ω_3D, y)
  + λ_hint · KL(softmax(ω_3D/τ) ‖ softmax(ω_full.detach()/τ))
  + λ_consis · L_CAM(MSE between render_feats and feat_2d)
```

### 关键超参

| 参数 | 值 | 说明 |
|---|---|---|
| 优化器 | Adam | |
| LR (AnchorEncoder / HOIFusion) | 1e-4 | 从零初始化 |
| LR (解码器末层 / PointNet++末层) | 1e-5 | 微调 |
| LR (RoBERTa) | 1e-5 | 微调 |
| Batch | 8 | |
| Epochs | 20-30 | |
| H* 维度 d | 256 | AnchorEncoder 输出 |
| H* 锚点数 k | 128 | |
| d_model | 512 | = emb_dim，cross-attn 维度 |
| λ_3D | 0.5 | |
| λ_hint | 1.0 | |
| λ_consis | 0.1 | |
| τ | 2.0 | KL 温度 |

### 模块文件

| 文件 | 说明 |
|---|---|
| `model/hoi_handler.py` | `HOIHandler`: numpy 锚点选择 + 可学习 `AnchorEncoder` |
| `model/hoi_fusion.py` | `HOIFusion`: cross-attn + FFN + LayerNorm + null_token |
| `model/branch_3d.py` | Step 5d 插入 HOIFusion；forward 增加 `hoi_feat` 参数 |
| `dataset/piad.py` | 新增 `use_hoi` 参数，加载 H_raw npz |
| `config/train_stage2_hoi_fusion.yaml` | HOIFusion 专用训练配置 |
| `scripts/train_stage2.py` | 新增 HOI 训练路径（with/without 双路 + Hint KL） |

## 风险点

1. **AnchorEncoder 从零初始化可能导致训练初期 H* 噪声过大**：缓解方案——`HOIFusion` 的 residual 连接（`obj_feat + attn_out`）保证初始退化；`null_token` 初始为零向量，without-HOI 路径初始等价于基线。
2. **变长 H_raw 批量处理**：不同样本的 `M_raw` 不同（VGGT 重建点数不一致），需 per-batch padding。缓解——HOIHandler 逐样本处理 numpy 部分，仅在 AnchorEncoder 处 stack 到 batch。
3. **H_raw 缺失**：部分 (class, affordance) 对可能没有对应的 HOI npz。缓解——dataset 中若找不到 npz，返回 `None`，train loop 中跳过该样本的 HOI 路径（退化到 without-HOI only）。
4. **VGGT 重建质量参差**：conf 分布可能异常。缓解——`conf_floor` 过滤 + `M' < 2k` 时退化到 FPS 均匀采样。