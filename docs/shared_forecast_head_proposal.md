# 共享主干（SharedUNet/Transformer）embedding头 + 预测头 设计方案：GDN学出的注意力关系替换"预测值 vs 实测值"残差，物理先验/FTA只负责attention的prior bias

**2026-08-17 更新**：新增第4节"数据划分"，明确预测头必须复用现有 fit/calib/test_normal
划分，通过跨窗口配对（同机器人、非重叠尾段）构造预测目标，而不是窗口内部切分前后两段
（窗口内切分版本见 `memory/forecast-head-signal-k.md` 的 v1 记录，已被 v2 取代——v1 的
预测目标与输入在原始时间上有 47% 重叠，实测证实这个重叠让任务失真）。

## 背景与动机

当前 `SharedEncoder`（`joint_prototype_model.py`）是一层 `Linear(window_size, embed_dim)`，只在窗口重建预训练（`x_hat = decoder(z)`，仅在 `training_mode=True` 时计算，只作训练损失）
下游任何最终结论——B/C/H/I/J 等所有已验证信号都只依赖 `d_node`、`resid_struct`、`d_mahal`，
没有一个用到 `x_hat`。这次建议的方向是把这层 Linear encoder 升级为一个更强的共享主干
（UNet 或 Transformer），主干产出的 `z` **继续同样支持** prototype 检索和 B/C/H 等已验证信号
（接口/下游消费方式完全不变），同时主干**新增一个预测头**，用类似原始 GDN 的跨节点注意力
（每个节点看哪些邻居来预测自己的未来值），预测未来一段时间的信号数值——预测值不再依赖
`kin_residual`/`ExplicitPhysicsScorer` 那样的手写物理公式，而是让学出来的注意力关系，物理先验
（或 FTA，如果故障树分析或类似结构化先验知识）只负责给这个预测注意力的 logits 加上一个先验偏置
机制不复用 `TrendGraphAttentionHead` 已经验证过的"先验只做注意力偏置、不锁死边"
（`prior_bias_strength * prior_mask`）那一套。

**接收新 memory 更新（`memory/scoring-signals-B-C-E-H.md`，2026-08-17，两处修正）：**

1. **robo3er 当前故障类型只有 2 种**——`data/robo3er/metadata.json` 实测确认标签只有
   `cable trapped`、`stuck`。此前 `memory/physics-informed-gdn.md`、
   `memory/feature-purification-audit.md`、`docs/PROJECT_OVERVIEW.md` 里出现的
   `Broken Pipe`、`Low battery` 是旧版数据快照残留的历史记录（**不是当前的有效评测目标**），
   本方案后面的所有提及都只针对 `cable trapped`/`stuck` 两种故障核算。
2. 现有信号当前已收敛为 **B（node_max）/ C（struct_max）/ E（phys_max）/ H（cov_mahal）**，
   以及组合信号 F=max(C,E)、I=max(B,H)、J=max(B,C,E,H)（`scoring-signals-B-C-E-H.md` 是当前
   最权威参考；`memory/robo3er-explicit-physics-scoring.md` 里出现的另一套 D/G/K 字母表是已删除的
   `ExplicitPhysicsScorer` 收敛脚本的临时命名，和这里的字母表**不是同一套**）。本文档建议的新
   信号沿用当前字母表接续（当前为 **`K_forecast_max`**，因为 B/C/E/F/H/I/J 都未被占用）。

**目标等价性**：用"GDN学出的注意力关系"替代"物理公式"来产生预测值，看"预测 vs 实测"的残差作为
新信号——如果这套注意力关系确实学到了接近真实物理关系的东西，理论上应该逼近已关闭方案
`ExplicitPhysicsScorer` 在 robo3er 上得到的结果（组合分数 0.810，纯物理信号 0.637，均为旧
4故障口径下的历史数字，仅作机制参照，不是当前2故障口径下可比的基准），但不同要求是不对节点
都有人工写好的解析公式,可以要看物理先验没有明确要注的边。

## 需要先说清楚的两条独立风险轴（不要混为一谈）

项目过去的实验把两件事绑在了一起的图,但它们其实是两个独立的赌注,这个方案必须分开验证:

**轴1（表征空间：原始信号 vs. 缩放后的 embedding/deviation）。**
`kin_residual`（原始信号空间替换特征）和 `ExplicitPhysicsScorer`（原始信号空间线性公式）都在
robo3er 上生效，而 `TypedRelationAnomalyHead`/`TrendGraphAttentionHead` 的对应机制在缩放后的
偏差空间 `d=z-p*` 里工作——已确认的结论是,`stuck` 故障破坏的那条运动学关系,在以重建/编码信诺
为目标训练出的缩放表征里被抑制了,所以缩放空间里的残差不如原始空间敏感。

**轴2（任务类型：瞬时一致性检查 vs. 时间维度的未来预测）。**
`memory/physics-informed-gdn.md` 记录的"纯数据GDN基线"就是标准的跨特征单步预测,在旧版4故障
口径下（当时还包含现在已弃用的 `Broken Pipe`、`Low battery`）,一个突发类型故障上表现得很好
（Broken Pipe 0.914、cable trapped 0.950、Low battery 0.975）,但在 `stuck` 上只有 0.528
（接近随机）。**折算到当前 robo3er 2故障口径（`cable trapped`、`stuck`），仍然直接相关的只有
`cable trapped` 这一条突发类型故障的历史结果（0.950）**，`Broken Pipe`/`Low battery` 的数字已
不对应当前有效评测目标（仅保留作为"预测类方法对突发故障普遍有效"这一结论的旁证，不能作为
本方案的时效来源之一去核算。原因是时缩偏差（shrinkage bias），预测误差衡量的是局部的。以当前
趋势为条件的最大程度,而 `stuck` 的电流恰好足够平滑,局部是可预测的,预测误差因此不明显。这是
预测类方法本身的结构性瓶颈,与预测是否使用物理公式,比如在原始空间还是缩放空间无关——
`kin_residual` 之所以能修复 `stuck`,根本是因为它是瞬时一致性检查,不是时间预测,不继承这个
时缩偏差。

**这次建议的方案同时能在"轴1可以反对（如果预测目标是原始信号）"和"轴2的已知风险区（因为是时间预测）"间。**
必须把这两条分开设计验证,而不是假设"用GDN学出的注意力替代物理公式"就能同时规避长时缩偏差——
它规避的只是"依赖人工公式"这一点,不规避"预测类任务对渐进性故障不敏感"这一点。

## 建议架构

### 1. 单一共享主干（SharedUNet/Transformer）分头输出,而非另起一个独立分支

我把你说清楚：embedding 继续单独支持 prototype 检索和 B/C/H 等现有信号),这不是"额外接一个
子 encoder 无关的独立预测分支",而是把 `SharedEncoder`（目前只是一层 `Linear(window_size,
embed_dim)`）本身升级为一个更强的共享主干（UNet 式多尺度卷积或 Transformer 编码器都可以),
这个主干**同时输出两样东西**,两者共享底层特征提取,但各自接不同的头：

```
x [B,T,N]
   |
   v
SharedBackbone (UNet/Transformer, 替换旧 Linear encoder)
   |
   +--> z [B,N,D]  (embedding头,接口/形状与现有 SharedEncoder 输出完全一致)
   |        |
   |        +--> JointPrototypeMemory -> p* -> TrendGraphAttentionHead -> resid_struct (信号C), d_node (信号B)
   |        +--> DeviationCovarianceHead -> d_mahal (信号H)
   |        +--> decoder(z) -> x_hat  (训练期窗口重建,沿用现状,仍用于训练损失)
   |
   +--> ForecastHead(z 或主干中间层特征) -> x_hat_future [B,H,N]  (未来H步预测,原始特征尺度)
            |
            +--> 信号K = zscore(未来实测值 - x_hat_future) 逐节点或 max
```

**关键约束（对应你的要求）**：`z` 的接口形状 `[B,N,D]` 和下游消费方式（`JointPrototypeMemory`/
`TrendGraphAttentionHead`/`DeviationCovarianceHead`）保持不变,B/C/H 三个已验证信号的计算逻辑
**不需要改一行**,只是产生 `z` 的主干从单层 Linear 换成了更强的 UNet/Transformer。这样即使新增
的 `ForecastHead` 效果不理想,也不会破坏已验证的三个信号——风险被限制在"主干换成 UNet/Transformer
后 `z` 的分布/质量是否退化"这一点上,需要在验证计划中单独部分"仅替换主干,不加预测头"的时机
（见验证计划步骤0）,把"主干升级"和"新增预测头"这两个变量分开检验。

**预测头 `ForecastHead` 的具体机制**（GDN式注意力,而非公式)：在 `z`（或主干重构造的、时间
信息保留更完整的中间特征)上再上一层类似 `gdn_model.GDN` 的图注意力：node embedding 余弦相似度
选 top-k 邻居 → 拼接自身/邻居特征算 attention logits → softmax 汇聚 → MLP 输出未来值预测。
**关键改动只有一处**：在 attention logits 上加物理先验偏置,和法与 `TrendGraphAttentionHead`
完全一致：

```python
# gdn_model.GDN.forward 中截取:
logits = self.attn_a(torch.cat([wf_self, wf_neighbors], dim=-1)).squeeze(-1)
alpha = F.softmax(F.leaky_relu(logits), dim=-1)

# 改为(新增两行,机制与 TrendGraphAttentionHead.prior_bias_strength 完全一致):
prior_bias = self.prior_mask[torch.arange(N, device=x.device).unsqueeze(1), neighbor_idx]
logits = logits + self.prior_bias_strength * prior_bias.unsqueeze(0)
alpha = F.softmax(F.leaky_relu(logits), dim=-1)
```

`prior_mask`/`prior_bias_strength` 复用 `EDGES_NAMED` 声明的同一批边（也预留补FTA/故障树
分析给出的结构化先验边,如果 FTA 真有这类知识来源),不需要为预测头重新设计先验注入机制——
声明的物理边只是让预测头"更快/更稳地"学到已知关系,未声明的关系仍然可以通过学习到的 top-k
注意力自由建立,这与 `TrendGraphAttentionHead` 的设计初衷"结构不锁死,只是加偏"完全一致。

**训练目标是多任务联合损失**,不是单独训练：

```python
loss = l_pred + BETA*l_me + l_mc + LAMBDA_EDGE*l_edge + LAMBDA_TYPED*l_typed + LAMBDA_FORECAST*l_forecast
# l_forecast = MSE(x_hat_future, 未来H步的实际值),超参 LAMBDA_FORECAST 需要单独调,
# 避免预测任务的梯度主导,把 z 学成"更适合预测"而非"更适合重建+结构匹配"的方向
# ——这正是验证计划步骤0要检查的退化风险。
```

### 2. 预测目标必须是原始信号空间,不是 embedding/deviation 空间

`ForecastGraphAttentionHead` 的输入输出都应该是原始特征值（和 `gdn_model.GDN` 一样直接在
`x [B,T,N]` 上操作),不要建在 `z` 或 `d=z-p*` 之上。这是为了让轴1（表征空间）站在已验证有效的
一侧——如果预测目标也放到缩放后的 embedding 空间,等于同时引入两个未验证的变量,一旦效果不好
无法判断是时缩偏差的问题还是表征空间的问题。

### 3. 新信号命名与迭代方式

按 `scoring-signals-B-C-E-H.md` 当前有效的命名体系（B_node_max / C_struct_max / E_phys_max /
F_v3_max=max(C,E) / H_cov_mahal / I_node_cov_max=max(B,H) / J_v3_cov_max=max(B,C,E,H)），
`EDGES_NAMED`/`TOP_K`/`NUM_PROTOTYPES` 等超参沿用现有配置),新信号命名为 `K_forecast_max`：

```python
resid_forecast = (x_future_actual - x_hat_future).pow(2).sum(axis=-1)  # 逐节点单步或多步预测偏差
z_forecast = zscore(resid_forecast_normal, resid_forecast_calib)
scores["K_forecast_max"] = z_forecast.max(axis=1)
scores["L_full_plus_forecast"] = np.max(np.stack([b, z_struct.max(axis=1), z_phys.max(axis=1), z_mahal, z_forecast], axis=1), axis=1)
# L = max(B, C, E, H, K) ——J 的基础上再合并入预测信号, 与 J=max(B,C,E,H) 的构造方式完全一致
```

作为**独立可选信号**接入现有的 zscore+max 汇总框架和联邦评估协议（复用 `align_and_split`),
与 B/C/E/F/H/I/J 一起横向比较,不直接替换任何现有信号或改动现有 encoder 的训练目标。

### 4. 数据划分：安全复用现有 fit/calib/test_normal 划分，无需新建划分逻辑

**结论先行：预测头的训练/校准/测试三段划分与 B/C/E/H 完全一样，不需要另外设计。** 唯一新增的
是在这三段划分内部，把"单个窗口"重新组织成"(输入窗口, 未来目标)"配对，配对本身要遵守三条
规则（细节见下），配对本身不改变哪个窗口被分到 fit/calib/test_normal 还是某个 fault 类别的
归属。

**现有划分是什么？`src/robo3er/dataset.py::split_normal`（`FIT_FRACTION=0.70`/
`CALIB_FRACTION=0.15`）**：划分只在所有 `normal`（label=0）窗口的（按窗口编号，等价于时间顺序，见
下）**排序后按序**：前 70% 作为 `fit`（只用于训练），接下来 15% 作为 `calib`（只用于校准
zscore 的 median/IQR），剩下 15% 作为 `test_normal`（所有 fault 窗口整体另跑测试，不参与
训练或校准）。实测当前 robo3er 数据：`fit`=3940 个正常窗口（索引 0～4227），`calib`=844 个
（索引 4228～5071），`test_normal`=845 个（索引 5072～5967），此外 `cable trapped`=206、
`stuck`=149 个窗口只用于测试评分。联邦版本（`fl_dataset.py`）用同一套 70/15/15 比例，只是
按 5 个机器人各自的窗口范围分别切（而不是全局切一次），这样每个 client 的 fit/calib/test
仍各自保持时间顺序，符合"share weights, calibrate locally"的约定。**新增的预测头训练/
校准/测试完全沿用哪一套划分（全局 vs 按 client）取决于跑的是 `run_robo3er_v3_1.py`（集中式）
还是 `run_robo3er_v3_1_federated.py`（联邦式），和现有 B/C/E/H 的算法保持字面一致，不新增
参数。**

**"用旧窗预测新窗新段"完全可行，正是本方案的核心机制，但有两个必须处理的细节：**

**(a) 配对只能在同一台机器人的连续窗口之间进行，不能跨越 fault↔normal 边界的机器人边界。**
`data/robo3er/metadata.json` 里 `stride=32 < window_size=60`，窗口编号本身就是按每个机器人
CSV 文件按时间顺序切出来的（`robot_map`：0～352 属于 robot00、353～706 属于 robot01、
707～884 属于 robot02、885～1217 属于 robot03、1218～5983 属于 robot04），所以"窗口编号 i 和
i+1 是否原始时间相邻"不能想当然认为一定成立——两种情况会导致 i 和 i+1 并不是训练意义上有效的
"旧窗→新窗"对（跨机器人边界，例如 i=352 属于 robot00，i+1=353 属于 robot01，
两者毫无时间关系；跨故障边界，例如 robot01 从某个索引开始进入连续 206 个 `cable trapped`
窗口，这段之前最后一个 `normal` 窗口的"下一个窗口"其实是故障窗口，不能当作训练 pair 的
target，因为 fit 划分只用 normal 窗口）。实测在当前数据上，`fit`/`calib`/`test_normal` 三段
里分别有 3929/843/839 个索引 `i` 满足"`i+1` 也在同一段、同一机器人、同一normal标签下"
（即 3940/844/845 个窗口集合中只有少数处在跨机器人或跨故障边界处没有配对的下一个窗口，直接丢弃这些
"孤儿"窗口，不强行配对）。具体做法：先用 `robot_map` 给每个窗口标注归属机器人，再检查
`i+1` 与 `i` 是否同机器人、`targets[i+1]==0`（配对目标必须是 normal，训练阶段的预测头只
学习"正常如何演化"），以及 `i+1` 与 `i` 同属 fit（或同属 calib、同属 test_normal，视用途而定）。

**(b) 相邻窗口编号在原始时间步上有 47% 重叠，直接用"下一个窗口"做预测目标会让任务被大幅
拉平（几乎白给）。必须把预测目标限定为不重叠的未来段。** `window_size=60`、`stride=32` 意味着
窗口 i 覆盖原始时间步 `[32i, 32i+60)`，窗口 i+1 覆盖 `[32i+32, 32i+92)`——两者重叠 28 个时间步
（60 步里 47%）。如果直接把"窗口 i+1 的全部 60 步"当作预测目标，模型只需要靠输入窗口尾部 28
步就能让这一段的预测误差趋近于零，这就不是真正的"预测未来"，也会让预测残差的异常敏感度集中在一大段
"照抄就能对"的时间步上，与本方案"预测误差反映动力学关系是否被破坏"的设计意图相悖。**正确
做法是把预测目标限定为窗口 i+1 中并未在窗口 i 出现过的尾部 `stride=32` 步**（即原始时间步
`[32i+60, 32i+92)`），预测头的输出形状从"预测一整个窗口"改为"预测一个 stride 长度的
新增段"（`x_hat_future` 形状是 `[B, stride, N]` 而不是 `[B, window_size, N]`，如果之后想要
更长的预测视野（验证计划第2步要对比的"多步预测"），也应该以 stride 为单位累积多个不重叠的
未来段，而不是简单叠多个重叠的下一个窗口。

**校准和评分的配对方式与训练一致**：`calib` 段同样按"同机器人、同normal、非重叠尾部"配对，
用来计算 `resid_forecast` 的 median/IQR；`test_normal` 和每个 fault 类别在评分时，用其
**实际的同一个窗口**（不要求是 normal，可能本身已经处于故障中），这是故意为之——评分阶段要
检验"用最近观测预测的未来是否符合实际发生的偏离"，输入窗口是否异常正是需要被residual放大的
部分作为输入，同样只用重叠尾部作比较（孤儿窗口，无法配对出下个窗口，例如某个 fault 类别的
第一个窗口，直接跳过评分，不强行填充）。

## 验证计划（必须分,不顺序不能颠倒）

0. **先做"仅换主干、不加预测头"的隔离测试。** 把 `SharedEncoder` 换成 UNet/Transformer,
   `ForecastHead`/`l_forecast` 暂时关闭,看 B/C/H 三个信号在新主干上是否至少持平现有 Linear
   encoder 的结果。这一步不通过,后面所有步骤都不必做——说明主干升级本身就已经在损害
   B/C/H 依赖的 embedding 质量,需要先解决这个问题（换更小的 UNet、调整容量、加正则化),
   而不是急着加预测头。
1. **主干确认不退化后,再打开预测头和 `l_forecast`,单独在 `stuck`（渐进性故障)上验证时缩
   偏差是否复现。** 这是最关键的对照——如果 `K_forecast_max` 在 `stuck` 上和纯数据GDN
   基线 0.528 接近而失败,说明"用学出的注意力替代物理公式"并没有终止时缩偏差,只是换了一种
   预测值的来源,轴2的风险同样保留;同时要检查 `LAMBDA_FORECAST` 权重是否让 B/C/H 在 `stuck`
   上相对步骤0出现退化（多任务负迁移)。
2. **对比"1步预测"（单个 stride=32 段)和"多步预测"（累积多个不重叠 stride 段,视野更长)
   对时缩偏差的影响。** 沿用第4节定义的非重叠尾部配对方式做累积,而不是简单叠多个重叠的
   下一个窗口。项目此前没有测试过预测步长对渐进性故障敏感度的影响,这是一个开放问题,不能假设
   更长的预测窗口就能规避时缩偏差,需要实测。
3. **确认物理先验偏置确实帮助了预测头在声明边上的收敛速度/精度**（例如对比有先验偏置 vs
   `prior_edges=None` 两种初始化下,声明边对应节点的验证集预测 MSE),而不只是看最终异常检测
   AUROC——这样即使最终 AUROC 提升不明显,也能确认先验偏置本身起作用,便于把问题定位在预测机制
   还是先验注入环节。
4. **只有通过步骤0和步骤1（主干不退化、`stuck` 上不明显劣于现有最佳信号)之后,才继续做
   跨消融和跨数据集验证**,避免在一个已知会失败的任务类型或已经退化的主干上多花更多篇幅。

## 与本项目其他相关结论的关系

- 与"路径1"（用 BPFO/BPFI 解析公式初始化 `prior_bias_strength`)是**同一种先验注入机制**在
  不同场景下的复用——路径1 加在 `TrendGraphAttentionHead`（瞬时一致性,偏差空间),这次建议加在
  新的 `ForecastHead`（时间预测,原始信号空间),两者互不冲突,可以并存。
- 与已关闭的 `ExplicitPhysicsScorer` 方向不同：这次预测值来自学习到的注意力而非人工公式,
  不重回"手写数值解被关闭方向"的老路,是一个结构上更接近 `gdn_model.GDN` 而非
  `explicit_physics_score.py` 的数学基。
- 与 `memory/physics-informed-gdn.md` 的时缩偏差结论不冲突,而是直接沿其约束——按当前 robo3er
  2故障口径,本方案预期在 `cable trapped`（突发性)上可能有增量价值,但在 `stuck` 这类渐进性
  故障上大概率不会比现有 `kin_residual`/`resid_struct`（信号C)更好,验证计划第1步就是为了尽早
  确认这一点,避免过度投入。

## 待确认事项

- "FTA" 在你的语境是还没有在项目文档/代码中找到明确定义,本文暂以"故障树分析（Fault Tree Analysis）
  这类结构化先验知识源"来理解,和物理公式一样可以转换成 `prior_mask` 的声明边输入（如果 FTA
  另有含义的机制,请指出,我会修正这部分的具体接入方式。
- 本文档已按你本轮明确要求核对 `data/robo3er/metadata.json`（确认 robo3er 当前实际故障标签只有
  `cable trapped`、`stuck` 两种；若其他数据集（Paderborn/Sielaff/voraus-AD）还有 memory 也有
  类似的故障种类和调整未同步到,请指出对应的 memory 文件,我会一并核实修正。

## 实现记录（2026-08-17，两轮）

- **v1**（`benchmark/run_robo3er_forecast_v1.py`，已废弃）：用第4节明确否定的"窗口内前缀/
  后缀切分"实现，预测目标与输入在原始时间上重叠 47%，K 在 `cable trapped` 上只有 0.721（远
  低于其他信号的 ~1.0），任务被证实失真。
- **v2**（`benchmark/run_robo3er_forecast_v2.py`，当前版本）：严格按第4节实现——复用现有
  fit/calib/test_normal 划分，跨窗口配对（同机器人、`targets[i+1]==0`、非重叠 stride=32 尾段），
  `ForecastHead` 输入改为完整 60 步窗口（与其他头一致），输出 `[B, 32, N]`。结果：K 升至
  cable trapped=0.916、stuck=0.690（均值 0.803，仅次于 H=0.868，超过 B/C/E/F/J），证实 v1 的
  失真确实是任务设计问题而非"学出的注意力关系天生不如物理公式"。详见
  `memory/forecast-head-signal-k.md`。
- **改进1（声明边先验消融，`--no-forecast-prior`）**：结果是否定的——去掉先验后 K 均值反而
  略升（0.803→0.808），声明边目标节点自身的校准集预测 MSE 也更低（0.549→0.518）。说明
  K 的价值几乎全部来自无偏学习注意力，先验注入的重点应留在 `TrendGraphAttentionHead`
  （信号C），不需要在预测头里重复。
- **改进2（多步预测视野，`--horizon-mult M`，M=1..8）**：本轮最重要的发现——`stuck` 上的
  K 随视野拉长几乎单调上升（0.690→0.819），从 `horizon_mult=4`（预测未来128步）开始
  **K 在 stuck 上超过 H**（H 全程稳定在约0.74，因为它不依赖预测头），horizon_mult=8 时
  K_stuck=0.819 比 H_stuck=0.739 高8个点。代价是 cable_trapped 随视野拉长小幅下降
  （0.916→0.878，有噪声），K 均值在 horizon_mult=6~8 附近趋于平台（约0.849-0.850），仍
  略低于 H 均值（约0.87）。max 组合在长视野下依然无效，甚至更明显地把 cable_trapped 上
  原本接近满分的排序拖低——K 不适合塞进 elementwise-max 组合，但作为独立信号，在长视野下
  已经是这个项目里第一个能在 `stuck` 上超过 H 的信号。详见
  `memory/forecast-head-signal-k.md` 的改进1/改进2章节。
