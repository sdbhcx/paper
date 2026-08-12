# GEAL 改进方案：从局部 3D Token 到鲁棒 Affordance Grounding

下面的方案以 **GEAL 为基线**，借鉴 SAGE 的局部 3D tokenizer、可变 token 数和离散 vocabulary 思想，但不直接迁移 SAGE 的完整 LLM 与 GRPO。

先区分两类内容：

- **论文事实**：GEAL 使用 3DGS 建立二维—三维对应，采用 DINOv2、PointNet++、RoBERTa、GAFM 和 CAM；SAGE 使用 FPS、KNN、相对位置编码、局部池化、VQ 和三阶段训练。
- **下面的改进方案**：是基于两篇论文组合得到的研究设计，需要通过实验验证，不能当作已有论文结论。

---

## 一、总体迭代路线

|版本|核心改动|主要目标|风险等级|
|---|---|---|---|
|V0|基线复现与诊断|找出 GEAL 的真实瓶颈|低|
|V1|连续局部 3D Tokenizer|改善局部结构表达和跨类别复用|低—中|
|V2|语言条件 Token 交互|让 query 更早选择相关局部区域|中|
|V3|粗到细的可变 Token 数|提升小区域定位与分辨率适应性|中—高|
|V4|Token-level 2D–3D Consistency|将 CAM 从特征对齐提升为局部 token 对齐|中|
|V5|Affordance-aware VQ|构建跨物体复用的局部功能词表|高|
|V6|Clean–Corrupt 鲁棒一致性训练|改善组合腐化和真实噪声下的稳定性|中—高|
|V7|区域级偏好排序|探索 SAGE preference learning 的有限迁移|高，不建议作为第一主线|

**推荐主线：**

V0→V1→V2→V4V0→V1→V2→V4​

如果 V3 的自适应分辨率效果明显，再加入 V3；V5 和 V6 更适合作为后续扩展。

---

# 二、V0：基线复现与问题诊断

## 1. 改动点

不修改 GEAL 网络，先建立严格的诊断基线：

1. 复现 GEAL 的 PointNet++、DINOv2、GAFM、CAM 和两阶段训练。
2. 对 GAFM 与 CAM 做完整的 2×22×2 因子消融：

|配置|GAFM|CAM|
|---|---|---|
|A|×|×|
|B|✓|×|
|C|×|✓|
|D|✓|✓|

1. 分别统计：
    
    - Seen / Unseen；
    - PIAD / LASO；
    - clean / corruption；
    - 小面积 affordance / 大面积 affordance；
    - 不同点云密度；
    - 不同随机种子。

## 2. 改动原因

GEAL 原始消融能够证明完整模型有效，但不能完全分离 GAFM 和 CAM 的独立贡献。若直接加入新模块，后续无法判断性能提升来自：

- 新 tokenizer；
- CAM；
- GAFM；
- 训练策略；
- 参数量增加。

因此 V0 的目的不是提高指标，而是建立一个可以解释的实验地基。

## 3. 原理

控制变量原则：

一次只改变一个主要因素一次只改变一个主要因素

同时建议增加区域级指标：

- affordance 区域召回率；
- 小区域 aIoU；
- 边界 F-score；
- 点数减少时的性能曲线；
- corruption severity—性能曲线。

GEAL 当前的整体平均指标可能掩盖小区域定位失败，因此需要把“整体性能”和“局部区域能力”分开。

## 4. 合理性

这是所有后续版本的必要前提。GEAL 的 PIAD Unseen aIoU 仍然较低，说明超过基线并不等于新类别定位已经可靠。V0 可以确认问题究竟主要来自：

- 局部几何特征不足；
- 语言条件没有充分作用；
- 二维—三维对齐不够精确；
- 低分辨率下小区域丢失。

## 5. 可行性

可行性最高，不需要新增核心网络，只需要：

- 补充训练配置；
- 增加完整消融；
- 统一随机种子；
- 编写区域级评测脚本。

**进入 V1 的条件：**

- 基线结果基本复现；
- GAFM/CAM 的独立贡献明确；
- 确认 PointNet++ 或局部特征是主要瓶颈之一。

---

# 三、V1：连续局部 3D Tokenizer

## 1. 改动点

在 GEAL 的 PointNet++ 三维分支旁边增加一个 SAGE 风格的局部 tokenizer：

**text**

```text
点云
→ FPS 选择局部中心
→ KNN 构造局部邻域
→ 相对位置编码
→ 局部特征聚合
→ 连续 3D token
```

对于第 ii 个局部中心：

Ni=KNN⁡(ci,P)Ni​=KNN(ci​,P)zi=Pool⁡(ϕ(xj−xci,xj)∣xj∈Ni)zi​=Pool(ϕ(xj​−xci​​,xj​)∣xj​∈Ni​)

其中：

- cici​：FPS 选出的局部中心；
- xj−xcixj​−xci​​：相对位置；
- zizi​：局部连续 token。

第一版不建议替换 PointNet++，而是并联：

F3D=FPointNet+++γFtokenF3D​=FPointNet++​+γFtoken​

其中 γγ 是可学习门控系数。

## 2. 为什么要改

GEAL 的 PointNet++ 擅长层级局部几何，但其输出仍然是传统点级或层级特征。SAGE 的启发是：

> 将局部结构显式组织成一组可交互、可匹配、可复用的 3D token。

这样可以更直接地实现：

- 局部部件建模；
- 跨类别局部结构复用；
- 2D token 与 3D token 的对应；
- 后续语言条件注意力；
- 后续可变分辨率和 VQ。

## 3. 关键原理

SAGE 中 FPS 强调空间覆盖，KNN 保留局部邻接。相对位置编码让 token 不只记录“有什么点”，还记录“点之间如何排列”。

相比直接使用绝对坐标，相对坐标更有利于：

- 平移不变性；
- 局部形状迁移；
- 未见类别泛化；
- 不同物体位置变化下的复用。

## 4. 必须保留 token-to-point 映射

这是 V1 最重要的约束。

每个 token 必须保存：

**text**

```text
token_id
center_point_id
neighbor_point_ids
local coordinate frame
```

解码时将 token 特征传播回原始点：

f^j=∑i∈N(j)wijzif^​j​=i∈N(j)∑​wij​zi​

再通过逐点解码器得到：

y^j=MLP⁡([f^j,FPointNet++(j),t])y^​j​=MLP([f^​j​,FPointNet++​(j),t])

如果只生成 token 而丢失点映射，模型可能提高对象级语义，却无法完成精确 affordance segmentation。

## 5. 合理性

V1 是最稳妥的 SAGE 迁移方式：

- 不改变 GEAL 的二维教师分支；
- 不改变 CAM 的基本训练目标；
- 不引入离散量化；
- 不引入 LLM；
- 保留 PointNet++ 的空间定位能力。

它可以直接检验一个核心假设：

> 局部 token 是否比单纯 PointNet++ 层级特征更适合未见类别 affordance grounding？

## 6. 可行性

可行性较高。所需组件都是标准操作：

- FPS；
- KNN；
- MLP；
- 相对位置编码；
- max-pooling 或 attention pooling；
- 插值传播。

建议初始配置不要完全照搬 SAGE，而是做小规模搜索：

- token 数：128 / 256 / 512；
- 邻域大小：K=32/64/81K=32/64/81；
- 聚合：max-pooling / mean-pooling / attention-pooling；
- 融合：加法 / 拼接后 MLP / gated fusion。

**V1 进入 V2 的判断标准：**

- PIAD Unseen 和 LASO Unseen 有稳定提升；
- 小 affordance 区域召回率提升；
- Seen 性能不明显下降；
- 三个随机种子趋势一致；
- 参数量增加能够解释性能提升，而不是单纯扩大模型容量。

---

# 四、V2：语言条件的中期 Token 交互

## 1. 改动点

GEAL 当前已经进行文本条件视觉融合，但可以将语言 query 更明确地注入局部 token 层。

设：

- Z={zi}i=1MZ={zi​}i=1M​：局部 3D token；
- TT：RoBERTa 文本特征。

增加：

Zq=CrossAttention⁡(Q=Z,K=T,V=T)Zq=CrossAttention(Q=Z,K=T,V=T)

或者使用 token-to-text 交互：

ziq=zi+αi(t)⋅Wttziq​=zi​+αi​(t)⋅Wt​t

其中 αi(t)αi​(t) 表示第 ii 个 token 与当前 affordance query 的相关性。

## 2. 为什么要改

同一个局部形状，在不同语言任务下功能可能不同：

- “可以握持的位置”；
- “可以放置的位置”；
- “可以打开的位置”。

如果语言只在较晚阶段注入，三维分支可能先形成与任务无关的表示，后续只能进行有限的筛选。中期交互允许模型根据 query 调整局部 token 表示。

## 3. 为什么不建议最早注入

不建议一开始就让文本影响 FPS 或 KNN 分组，原因是：

1. FPS/KNN 应首先建立稳定几何邻域；
2. 过早语言条件化可能让模型依赖训练类别中的词语；
3. 同一几何区域在不同 query 下可能被错误切分；
4. 语言噪声会影响底层几何表示。

因此推荐：

**text**

```text
纯几何 tokenizer
→ 中期文本交互
→ token consistency
→ token-to-point decoder
```

## 4. 损失设计

V2 可以先不增加复杂损失，只保留 GEAL 原有目标：

L=LBCE+LDice+λconsisLCAML=LBCE​+LDice​+λconsis​LCAM​

另外可以添加一个轻量 token-query 对比损失：

Lq=−log⁡exp⁡(sim⁡(zi,t+)/τ)∑jexp⁡(sim⁡(zi,tj)/τ)Lq​=−log∑j​exp(sim(zi​,tj​)/τ)exp(sim(zi​,t+)/τ)​

其中正样本是对应 affordance 的 query，负样本是其他功能描述。

## 5. 合理性

V2 直接利用了 GEAL 本身已经存在的 RoBERTa 和文本条件融合机制，因此不是完全重新设计，而是改变融合位置和粒度。

它可以验证：

> 语言是否应该在局部 token 层参与，而不是仅在多尺度特征融合层参与？

## 6. 可行性

可行性中等。主要风险在于：

- 文本特征过强导致几何分支退化；
- 训练集语言模板与测试 query 分布不一致；
- token attention 增加显存。

建议采用：

- 冻结或低学习率训练 RoBERTa；
- 只在中间一层加入 cross-attention；
- 增加纯几何残差通道；
- 使用门控融合而不是完全替换。

---

# 五、V3：粗到细的可变 Token 数

## 1. 改动点

固定 token 数对所有点云和所有 affordance 使用同样预算，可能造成：

- 简单区域计算浪费；
- 复杂区域 token 不足；
- 小型 affordance 区域被统一下采样淹没。

V3 采用粗到细策略：

**text**

```text
低分辨率全局 token
→ query 预测局部重要性
→ 选择候选区域
→ 对候选区域增加细粒度 token
→ token-to-point 解码
```

先得到粗粒度 token zizi​，再计算：

si=MLP⁡([zi,t])si​=MLP([zi​,t])

选择高分区域进行局部细化：

Ifine=TopK⁡(si)Ifine​=TopK(si​)

最终 token 集合为：

Z=Zcoarse∪ZfineZ=Zcoarse​∪Zfine​

## 2. 为什么要改

GEAL 的 GAFM 已经说明不同 affordance 需要不同粒度：

- 按钮、把手：细粒度；
- 座面、支撑面：大范围。

V3 将这种“粒度自适应”从特征融合层进一步推进到 token 数量层。

## 3. 原理

给每个局部区域分配不同计算预算：

Mi=Mmin+B⋅softmax⁡(si)Mi​=Mmin​+B⋅softmax(si​)

其中：

- MiMi​：区域 ii 的 token 预算；
- BB：额外 token 总预算；
- sisi​：query-conditioned importance score。

这种机制可以让模型把更多资源分配给：

- 语言相关区域；
- 几何复杂区域；
- affordance 边界；
- 点云不完整但仍有证据的区域。

## 4. 必须避免的失败模式

### 失败模式一：只关注高分区域

如果模型早期预测错误，Top-K 选择会把真正的 affordance 区域丢掉。

解决方法：

- 保留全局粗 token；
- 使用固定比例的随机探索区域；
- 增加空间覆盖损失；
- 训练前期使用均匀预算，后期再启用自适应选择。

### 失败模式二：训练和推理不一致

如果训练中使用 soft attention，推理时使用硬 Top-K，可能出现性能落差。

解决方法：

- 采用逐阶段训练；
- 先训练连续权重；
- 再切换到硬选择；
- 比较 soft routing、hard routing 和 straight-through routing。

## 5. 合理性

这是对 GEAL 和 SAGE 两者优势的结合：

- GEAL 提供任务相关粒度；
- SAGE 提供分辨率自适应 token 思路。

尤其适合解决：

- 小区域 affordance；
- 点数从 2K 到 8K 的变化；
- 局部缺失；
- 不同物体复杂度差异。

## 6. 可行性

中等偏高，但训练复杂度增加。建议先做一个简化版：

**text**

```text
固定全局 token + 固定数量局部 refinement token
```

例如：

- 全局 token：128；
- 细化 token：64 或 128；
- 只对 query 相关度最高的若干候选区域细化。

不要第一版就使用完全动态的任意 token 数。

---

# 六、V4：Token-level 2D–3D Consistency

这是我最推荐作为主论文核心的版本。

## 1. 改动点

GEAL 的 CAM 目前主要对齐：

LMSE=MSE⁡(fcam3D→2D,fcam2D)LMSE​=MSE(fcam3D→2D​,fcam2D​)

V4 将其扩展为三个层次：

Ltoken=λfLfeature+λcLcontrast+λsLspatialLtoken​=λf​Lfeature​+λc​Lcontrast​+λs​Lspatial​

### 特征一致性

保留原 CAM：

Lfeature=∥F~3D→2D−F2D∥22Lfeature​=​F~3D→2D−F2D​22​

### Token 对比一致性

将对应的二维和三维局部 token 设为正样本：

Lcontrast=−log⁡exp⁡(sim⁡(zi3D,zi2D)/τ)∑jexp⁡(sim⁡(zi3D,zj2D)/τ)Lcontrast​=−log∑j​exp(sim(zi3D​,zj2D​)/τ)exp(sim(zi3D​,zi2D​)/τ)​

### 空间一致性

利用 3DGS 的可见性和 alpha 权重，对投影关系进行加权：

Lspatial=∑i,jaij∥zi3D−zj2D∥22Lspatial​=i,j∑​aij​​zi3D​−zj2D​​22​

其中 aijaij​ 表示三维 token ii 与二维区域 jj 的可见性或投影关联程度。

## 2. 为什么要改

GEAL 的 CAM 已经利用高斯基元实现二维—三维对应，但逐点特征 MSE 仍然比较“连续”和“整体”。

Token-level 对齐可以进一步保证：

- 哪些三维局部对应哪些二维区域；
- 对应局部是否具有相似语义；
- 非对应区域是否被区分；
- 二维教师的信息是否正确传到局部几何 token。

## 3. 合理性

V4 与 GEAL 的核心思想高度一致，不是另起炉灶：

- GEAL 已经有 3DGS；
- GEAL 已经有二维教师；
- GEAL 已经有 CAM；
- 新增的只是对齐粒度和损失形式。

相比直接引入 VQ 或 LLM，V4 更容易形成明确的机制贡献：

> 从连续特征一致性扩展到具有空间对应关系的局部 token 一致性。

## 4. 关键风险

二维和三维 token 并不总是一一对应：

- 遮挡；
- 多视角投影；
- 一个三维区域对应多个二维区域；
- 一个二维区域包含多个三维区域。

因此不应简单使用一一对应 MSE。建议采用：

- visibility-aware matching；
- soft assignment；
- optimal transport；
- top-k positive matching；
- 遮挡区域 mask。

## 5. 可行性

可行性较高，是最值得优先实现的创新版本。

实现步骤：

1. 使用 3DGS 将三维 token 附着到 Gaussian primitive；
2. 渲染到二维；
3. 从 DINOv2 特征图中提取对应二维局部 token；
4. 根据投影和可见性建立软匹配；
5. 训练 MSE+对比损失；
6. 保留三维 token-to-point decoder。

---

# 七、V5：Affordance-aware VQ

## 1. 改动点

SAGE 使用 VQ 将连续几何特征映射到 codebook：

qi=arg⁡min⁡k∥zi−ek∥22qi​=argkmin​∥zi​−ek​∥22​

GEAL 可以引入 VQ，但不建议直接把最终逐点预测完全离散化。更稳妥的形式是残差量化：

zi=eqi+rizi​=eqi​​+ri​

其中：

- eqieqi​​：离散几何原型；
- riri​：连续残差；
- riri​ 保留细粒度 affordance 信息。

## 2. 为什么不能直接照搬纯几何 VQ

形状相似不代表 affordance 相同。例如：

- 柱状结构可能是椅腿，也可能是工具手柄；
- 平面结构可能是桌面，也可能是不可交互的墙面；
- 中空区域可能是容器内部，也可能是孔洞。

因此建议使用双 codebook：

zi→{qigeo几何原型qiaff功能原型zi​→{qigeo​qiaff​​几何原型功能原型​

或者使用 query-conditioned code assignment：

qiaff=arg⁡min⁡k∥g(zi,t)−ekaff∥22qiaff​=argkmin​∥g(zi​,t)−ekaff​∥22​

## 3. 研究价值

如果训练成功，codebook 可能学习到跨类别复用的局部 primitive：

- handle-like；
- support-plane；
- container-interior；
- pushable-surface；
- graspable-edge。

这比类别级特征更符合 affordance 的本质：功能通常由局部几何和上下文共同决定，而不是由物体类别决定。

## 4. 合理性

VQ 有两个潜在作用：

1. **跨类别结构复用**  
    不同物体中的相似局部结构可以共享 code。
    
2. **跨模态离散对齐**  
    二维和三维分支可以对齐到相同或兼容的局部 code。
    

但必须强调：SAGE 的 VQ 主要服务于 3D—语言生成，GEAL 需要额外验证它是否真的有利于点级定位。

## 5. 可行性

可行性中等偏低，主要问题包括：

- codebook collapse；
- code 使用不均衡；
- 几何 code 与功能 code 混淆；
- 小数据集无法充分学习大型 codebook；
- 离散化损失可能损害边界精度。

不建议直接使用 SAGE 的 8192 作为默认配置。GEAL 数据规模和任务不同，应搜索：

∣C∣∈{256,1024,4096,8192}∣C∣∈{256,1024,4096,8192}

建议先从 256 或 1024 开始。

必须加入：

- commitment loss；
- code usage entropy；
- dead code 检查；
- 连续 token 对照；
- codebook 可视化；
- 跨类别检索实验。

**V5 适合作为高风险创新，不建议作为第一版主模型。**

---

# 八、V6：Clean–Corrupt 鲁棒一致性训练

## 1. 改动点

GEAL 的 PIAD-C 和 LASO-C 主要测试 clean-trained 模型在腐化数据上的退化。V6 将腐化样本用于训练，但必须和原始 clean-trained 结果分开报告。

对同一个样本生成：

Pclean,PcorruptPclean,Pcorrupt

要求二者在对应区域上的预测保持一致：

Lrobust=D(Y^clean,T(Y^corrupt))Lrobust​=D(Y^clean,T(Y^corrupt))

其中 TT 用于处理旋转、缩放等几何变换。

完整损失可以写成：

L=LBCE+LDice+λCAMLtoken+λrobLrobustL=LBCE​+LDice​+λCAM​Ltoken​+λrob​Lrobust​

## 2. 不同腐化需要不同处理

|腐化类型|一致性方式|
|---|---|
|Jitter|对相同点或最近邻点做预测一致性|
|Scale|先进行尺度归一化或坐标变换|
|Rotate|将预测旋转回原坐标系|
|Drop Global|对剩余点做一致性|
|Drop Local|对未丢失点做一致性，不能要求恢复不存在的点|
|Add Global|约束新增背景点低响应|
|Add Local|使用局部 hard negative 约束|

对于局部删点，不能要求模型恢复完全不存在的几何证据。应该区分：

- 仍然可见的点；
- 被删除的点；
- 预测不确定区域。

## 3. 为什么要改

GEAL 当前腐化实验揭示了局部删点是难点：关键交互区域一旦被移除，单纯的语义先验无法恢复缺失几何。

V6 的目标不是让模型“凭空恢复”缺失区域，而是：

- 对轻微噪声保持稳定；
- 对新增背景降低误检；
- 对局部缺失利用上下文；
- 对多种腐化组合减少灾难性退化。

## 4. 合理性

GEAL 已经提供了系统化 corruption benchmark，因此 V6 的实验基础很好。它可以进一步扩展当前评测的边界：

- 单一腐化；
- 组合腐化；
- corruption severity；
- Unseen+corruption 双重分布外。

## 5. 可行性

可行性中等。合成腐化容易实现，但需要严格区分三种实验协议：

1. **Clean-trained → clean/corrupt test**  
    测试自然泛化能力。
    
2. **Corruption-augmented training → corrupt test**  
    测试鲁棒训练收益。
    
3. **Unseen + corruption test**  
    测试双重分布外泛化。
    

否则会把“使用了腐化增强训练”误写成“模型天然对腐化鲁棒”。

---

# 九、V7：区域级偏好优化，不直接使用 GRPO

## 1. 不建议的方案

不建议直接使用 SAGE 的：

- Sentence-BERT 文本语义奖励；
- 文本长度奖励；
- GRPO 开放式生成优化。

原因是 GEAL 的输出是点级概率图，不是文本序列。文本语义相似度不能评价：

- 区域边界；
- 点级定位；
- 小区域召回；
- 假阳性分布。

## 2. 可迁移的思想

可以把 preference learning 改造成区域级排序：

Lrank=max⁡(0,m−s(r+)+s(r−))Lrank​=max(0,m−s(r+)+s(r−))

其中：

- r+r+：真实 affordance 区域；
- r−r−：形状相似但功能不匹配的 hard negative；
- s(r)s(r)：区域得分。

例如让模型满足：

s(真实把手)>s(相似支撑杆)s(真实把手)>s(相似支撑杆)

也可以构造 corruption preference：

s(clean-aligned prediction)>s(corruption-induced false region)s(clean-aligned prediction)>s(corruption-induced false region)

## 3. 合理性

这保留了 SAGE “通过偏好信号优化高层语义”的思想，同时适配 GEAL 的 dense prediction 结构。

但它需要：

- 高质量 hard negative；
- 区域级标注；
- 合理的 reward 设计；
- 避免和 BCE/Dice 重复。

因此 V7 只能作为后续探索，不应成为第一主线。

---

# 十、推荐的最终模型

## 推荐主模型：V4 为核心，V1 和 V2 为基础

**text**

```text
点云 P
  ├─ PointNet++ 三维分支
  └─ FPS/KNN 局部连续 tokenizer
          ↓
      局部 3D tokens
          ↓
      中期语言条件交互
          ↓
      token-to-point 对齐
          ↓
      token-level 2D–3D consistency
          ↓
      token-to-point decoder
          ↓
      逐点 affordance map
```

二维训练教师仍然保留：

**text**

```text
3DGS
  → 多视角深度/着色图
  → DINOv2
  → GAFM
  → 2D local tokens
  → 与 3D tokens 做 visibility-aware matching
```

推理时仍然只使用三维分支，保持 GEAL 的部署优势。

---

# 十一、每个版本的优先级

## 第一优先级：V1 + V4

最适合作为论文主线：

- 改动相对清晰；
- 保留 GEAL 核心结构；
- 有明确的空间机制；
- 不依赖大规模语言模型；
- 可以自然解释 unseen 和 corruption 改善。

核心论文叙事可以是：

> 传统连续特征级 CAM 难以保证局部结构对应。本文引入保留点映射的局部 3D token，并通过 3DGS 建立 visibility-aware token-level cross-modal consistency，从而提升跨类别 affordance grounding。

## 第二优先级：V2

作为语言条件增强：

- 适合开放词汇 affordance；
- 能解释不同 query 对粒度的影响；
- 与 GEAL 现有 RoBERTa/GAFM 兼容。

## 第三优先级：V3

如果发现：

- 小区域性能确实是主要瓶颈；
- 固定 token 数在不同分辨率下退化明显；
- 计算量成为问题；

再引入粗到细的自适应 token 预算。

## 第四优先级：V5

VQ 有较强新颖性，但风险也最大。适合作为：

- 辅助分支；
- codebook 分析；
- 跨类别功能 primitive 实验；

不建议一开始让 VQ 完全替代连续特征。

## 第五优先级：V6

如果目标偏向鲁棒性或机器人部署，可以加入 clean–corrupt consistency；否则它更适合作为扩展实验。

---

# 十二、建议的实验判定标准

每个版本都不要只看总平均 aIoU。建议设置以下进入下一版本的门槛：

1. **Seen 不明显下降**；
2. **Unseen 至少在一个数据集上稳定提升**；
3. **小面积 affordance 召回率提升**；
4. **局部删点或局部加点下退化减缓**；
5. **三种随机种子趋势一致**；
6. **性能提升能被对应机制解释**；
7. **参数量和推理开销单独报告**。

最终实验矩阵建议包括：

|维度|设置|
|---|---|
|表示|PointNet++ / 连续 token / VQ token|
|融合|后期融合 / 中期 token cross-attention|
|对齐|MSE / MSE+对比 / MSE+对比+空间匹配|
|分辨率|固定 token / 可变 token|
|泛化|Seen / Unseen|
|鲁棒性|clean / 单一 corruption / 组合 corruption|
|区域尺度|小区域 / 中等区域 / 大区域|
|效率|参数量 / 显存 / FPS / token 数|
|稳定性|三个随机种子|

---

## 最终建议

如果只允许选择一条最稳妥、最有研究价值的路线，我建议：

PointNet++并联局部Tokenizer+中期语言交互+Visibility-aware Token CAMPointNet++并联局部Tokenizer+中期语言交互+Visibility-aware Token CAM​

不要第一版就加入：

- 完整 LLM；
- GRPO；
- 大规模 VQ codebook；
- 完全动态 token 数；
- 彻底替换 PointNet++。

这样做的优势是：**改动有层次、因果关系清楚、实现成本可控，并且始终保持 GEAL 的核心能力——逐点 affordance 定位。**