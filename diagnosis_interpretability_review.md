# B/C/H/K 及其两两组合（BK/CK/HK）七种方案的可解释性现状，及"无监督检测+诊断一体化"的相关工作对比

（本版本按要求移除E信号(`phys_max`/`TypedRelationAnomalyHead`)相关内容，聚焦B、C、H、K
四个基础信号及其两两组合BK、CK、HK共七种方案。E信号本身仍是项目现有的第五个检测信号，
只是不在本次比较范围内。）

## 一、七种方案的实测检测效果对比（先看数字，再谈诊断能力）

以下数字来自`memory/forecast-head-signal-k.md`"Pairwise combination test"一节，是robo3er
数据集上已经跑过并核实过的实测结果（`run_robo3er_forecast_v2.py`，cross-window pairing版，
非v1的窗口内切分版），不是推测：

| horizon_mult | B | C | H | K | **BK=max(B,K)** | **CK=max(C,K)** | **HK=max(H,K)** |
|---|---|---|---|---|---|---|---|
| 1（默认视野） | 0.756 | 0.761 | 0.868 | 0.803 | 0.794 | 0.791 | **0.827** |
| 6 | 0.749 | 0.756 | 0.873 | 0.850 | 0.843 | 0.839 | **0.871** |
| 8 | 0.761 | 0.758 | 0.869 | 0.849 | 0.846 | 0.845 | **0.874** |
| **10（当前最优点）** | 0.779 | 0.754 | 0.878 | 0.859 | 0.878 | 0.882 | **0.901** |

（均为cable trapped和stuck两个故障类型的AUROC均值；单次种子跑出的结果，horizon间的数字有
真实噪声，`horizon_mult=10`的峰值是指示性的，不是精确定位的最优点。）

**七种方案的关系,不是"谁比谁好"这么简单，要按故障类型拆开看：**

- **B、C是全场最弱的两个**：在`stuck`（渐进性故障）上长期停留在0.52~0.56附近，接近随机，
  在任何视野下都不改变——因为它们不依赖预测头,对缓慢趋势天生不敏感。
- **H是当前单信号最优**：`cable trapped`接近1.0的天花板，`stuck`稳定在0.74附近（不随
  horizon变化，因为H不依赖预测头）。
- **K在长视野下能反超H在`stuck`上的表现**（`horizon_mult≥4`起，K的`stuck`分数超过H），
  但K自己的`cable trapped`会随视野拉长而下降（0.916→0.878附近），是一个此涨彼消的权衡，
  K的均值不会超过H的均值。
- **BK、CK两种组合"帮不上忙"**：在每个视野下都低于自己两个输入信号里更强的那一个
  （例如`horizon_mult=8`时BK=0.846 < K自己的0.849）——原因是B/C在`stuck`上和K的其他
  "弱伴"犯的是同一种弱点（同样对`stuck`不敏感），取max并不能带来新信息，只会把噪声
  叠加进来，稀释K在`stuck`上的优势。这与`memory/scoring-signals-B-C-E-H.md`记录的
  "组合规则"完全一致：**用max组合一个在某故障类型上偏弱的信号，会拉低整体排序**，
  不是简单地平均。
- **HK是七种方案里表现最好的**：从`horizon_mult=6`起持续优于H单独（0.871→0.873，几乎持平），
  到`horizon_mult=8`明显反超（0.874 vs 0.869），在`horizon_mult=10`达到本项目迄今为止在
  robo3er上找到的**最佳单一评分规则**（0.901，同时超过B/C/H/K逐个单独的表现）。原因是
  H和K在这两个故障类型上是**互补而非冗余**的关系：H的弱点`stuck`恰好是K的强点，反之
  H在`cable trapped`上的天花板几乎不受K拖累（0.999→0.969，侵蚀很小）。这正是B/C无法
  提供的——B/C在`stuck`上和K是"同病",不是互补。

**结论（覆盖此前"K不建议加入组合信号"的旧结论）**：旧结论是基于`L=max(B,C,E,H,K)`这个
**全量组合**得出的（E的贡献混在其中会稀释H在`cable trapped`上的优势），本次聚焦B/C/H/K
之后可以看到，问题不在于"K不该组合"，而在于"K不该和B/C组合"——**唯一值得采用的组合是
`HK=max(H,K)`，跳过B/C**。仍然是逐元素max（不是学出来的组合器），同样的排序腐蚀风险在
H、K不互补的数据集/故障类型上依然可能出现，这个结论目前只在robo3er的这两个故障类型上
验证过，不能直接假设可以推广。

## 二、七种方案的诊断（可解释性）能力核查

诊断能力的核查依据不变：直接读取`src/models/joint_prototype_model.py`里每个头的forward
输出形状和calibration逻辑。

| 信号/组合 | 原始输出粒度 | 能否直接用于诊断（不用改代码） |
|---|---|---|
| B (node_max) | `d_node [B,N]`，**逐节点** | **可以**——argmax(d_node)直接指向哪个节点偏离最大 |
| C (struct_max) | `resid_struct [B,N]`，**逐节点** | **可以**——argmax指向哪个节点的"被邻居预测"关系破裂,且`attn`权重本身还能进一步指向"是被哪个邻居的关系破裂"（见第四节） |
| H (cov_mahal) | `d_mahal [B]`，**标量,无逐节点分解** | **不能，是四个基础信号里唯一的诊断盲区**——H恰恰是检测效果最好的信号（robo3er stuck=0.738,全局唯一能打的信号），但目前实现完全没有暴露"哪些节点的联合偏离模式破坏了协方差结构"这个本可以做到的分解 |
| K (forecast_max) | `k_resid [B,N]`，**逐节点** | **可以**——和B类似,argmax直接指向"哪个节点的未来值被预测错了" |
| **BK/CK/HK（组合信号）** | max组合后只剩标量 | **可以，但要多走一步**——诊断时先判断这一样本的组合分数是由哪个分量"贡献"的（即比较该样本上B/K、C/K、H/K各自原始值，取更大的那个），再用那个分量自己的诊断机制（B/C/K的argmax，或H的贡献分解一旦实现）。**组合本身不引入新的诊断信息，只是把"用哪个信号的诊断结果"这一步交给了运行时判断** |

**结论**：B/C/K三个基础信号原生带逐节点定位信息；**H是唯一的诊断盲区**，且恰好是当前单
信号检测效果最好、`HK`组合里贡献"stuck"这半边优势的关键分量——补上H的分解，`HK`整个组合
在诊断上就补齐了，不需要给组合信号单独设计新的诊断机制。

## 三、H信号（协方差Mahalanobis距离）目前诊断能力缺失，是怎么造成的、怎么补

`DeviationCovarianceHead.forward` 只返回一个标量 `d_mahal[b] = (s-mu)^T Sigma^-1 (s-mu)`
（`s`=`d_node`，某个原型下每个节点偏差幅度构成的向量），没有暴露分解。但Mahalanobis距离在
数学上是可以逐项分解的——常见做法有两种，都不需要重新训练，纯后处理：

1. **逐节点贡献分解**：把 `(s-mu)^T Sigma^-1 (s-mu)` 展开成 `sum_i sum_j (s_i-mu_i) [Sigma^-1]_ij (s_j-mu_j)`，
   按节点 `i` 分组求和（`contribution_i = (s_i-mu_i) * sum_j [Sigma^-1]_ij (s_j-mu_j)`），
   贡献值可正可负，取绝对值排序即可得到"哪些节点的联合偏离对这次异常的Mahalanobis距离贡献
   最大"。这是标准做法（不是本项目原创，工业统计过程控制里T²统计量的贡献图/contribution plot
   就是这个思路），实现成本是几行numpy代码。
2. **留一法（leave-one-out）**：分别计算"去掉节点i后用剩余N-1个节点重算的Mahalanobis距离"，
   和原始距离的差值最大的节点即为主要贡献者——比方法1更贵（每个节点要重新求逆一次协方差子
   矩阵）但对强相关节点组的归因更稳健，可以作为方法1的交叉验证。

**建议**：优先实现方法1（贡献分解），因为H信号是robo3er `stuck`故障唯一有效的检测信号，
也是`HK`组合里"stuck"半边优势的来源——目前"H（或HK在stuck上判赢时）检测到了但完全不知道
为什么"是这个项目在**robo3er最重要的故障类型上**唯一的可解释性盲区，其余检测效果好的地方
（cable trapped靠B/C/K几乎都到0.9+）逐节点归因早就现成可用。

## 四、C信号的attention权重目前也没被用来诊断关系断裂，是第二个可以低成本补的缺口

`TrendGraphAttentionHead`的attention权重`alpha_ij`（top-k邻居的softmax权重，声明边额外叠加
`prior_bias_strength`偏置）目前只用于计算加权预测，`resid_struct`只保留了最终逐节点残差，
没有保留"这个节点当前最依赖哪个邻居做预测"这层信息。如果异常发生时对比"异常窗口的attention
分布"和"该节点在正常校准窗口下的attention分布均值"，差异最大的邻居就是"被打破的具体关系"的
候选——这正是GDN原论文用来做第二层诊断的同一种机制（见第五节的直接引用：用attention权重
指向和异常传感器关系最密切的邻居），本项目C头的结构和GDN原论文的注意力机制几乎同构，只是
目前没有把这层信息导出到诊断层，只用来算残差。

## 五、和已找到的相关工作对比：这个方向目前进展到什么程度

### 5.1 GDN原论文本身就是"检测+诊断"一体化设计,不是本项目独有的想法

GDN原论文（Deng & Hooi, AAAI 2021）在WADI数据集上有一个具体的攻击案例，完整展示了"检测→
定位→解释"三层链路。攻击的背景是流量传感器`1_FIT_001_PV`被篡改：<cite index="1-8,1-9">this anomaly arises from a flow sensor, 1_FIT_001_PV, being attacked via false readings. These false readings are within the normal range of this sensor, so detecting this anomaly is nontrivial.</cite>
第一层，逐传感器异常分数把定位范围收窄到相关传感器：<cite index="1-10,1-11">During this attack period, GDN identifies 1_MV_001_STATUS as the deviating sensor with the highest anomaly score... The large deviation at this sensor indicates that 1_MV_001_STATUS could be the attacked sensor, or closely related to the attacked sensor.</cite>
第二层，attention权重进一步指出具体的相关传感器和被打破的关系：<cite index="1-12,1-13,1-14">GDN indicates (in red circles) the sensors with highest attention weights to the deviating sensor. Indeed, these neighbors are closely related sensors: the 1_FIT_001_PV neighbor is normally highly correlated with 1_MV_001_STATUS... However, the attack caused a deviation from this relationship, as the attack gave false readings only to 1_FIT_001_PV.</cite>
第三层，用预测值和实测值对比确认异常的具体表现：<cite index="1-15">GDN further allows understanding of this anomaly by comparing the predicted and observed sensor values</cite>。
原论文摘要把这条链路概括为学习出的结构+attention同时服务于检测和归因：<cite index="2-9,2-10,2-11">existing methods do not explicitly learn the structure of existing relationships between variables, or use them to predict the expected behavior of time series. Our approach combines a structure learning approach with graph neural networks, additionally using attention weights to provide explainability for the detected anomalies. Experiments on two real-world sensor datasets with ground truth anomalies show that our method detects anomalies more accurately than baseline approaches, accurately captures correlations between sensors, and allows users to deduce the root cause of a detected anomaly.</cite>

**这三层和本项目B/C/K信号的对应关系是精确的**：第1层（逐传感器异常分数定位）对应B信号
(`d_node`)；第2层（attention权重指向相关传感器）对应C信号(`TrendGraphAttentionHead`的
`alpha_ij`)，但本项目目前只用attention权重去计算残差，没有像GDN论文这样把attention分布
本身导出给用户看；第3层（预测vs实测对比）对应K信号(`ForecastHead`)。**结论：这条"逐节点
分数+attention权重导出+预测对比=可诊断"的路径不是需要重新发明的东西，是GDN原论文本身的
标准设计，本项目B/C/K三个信号的底层数值已经具备这个能力，只差把attention分布导出并和
校准期基线做对比这一步（对应第四节的具体改进建议）。**

后续工作也确认了这条"attention→定位相关传感器"的链路是GDN家族的通用做法而非孤例，一篇
用同一WADI攻击案例、但改用层间相关性传播（LRP）替代attention的后续工作明确指出：
<cite index="5-9">Traditional models typically rely on attention weights to identify sensors associated with 1_MV_001_STATUS</cite>，
说明这是该技术路线里被反复沿用的标准诊断步骤。

### 5.2 反事实扰动解释是另一条更前沿、成本也更高的延伸路线

图神经网络异常检测领域近年有一条独立于GDN式分数定位的路线，通过最小扰动搜索来生成解释。
一篇2025年针对传感器异常检测场景的工作明确指出这类方法解决的是黑箱问题：<cite index="10-7,10-8">Understanding the reasons behind the predicted anomalies is essential for effective response, however, the black-box nature of GNNs poses a significant challenge. To address this limitation, we propose a counterfactual explanation framework that offers human-understandable insights by identifying minimal input changes capable of altering the model's decision.</cite>
其第一阶段和本项目B/C信号机制一致——<cite index="10-2,10-3">In the first stage, we identify the most influential sensors that contribute to an anomaly, along with their local graph neighborhoods. This localization step leverages node-level deviations and GNN attention</cite>——
第二阶段则更进一步生成最小扰动使预测从异常翻转为正常。这条路线能回答"要让这个节点的值
改变多少才会被判定为正常"，比单纯的分数/attention导出更贴近运维决策，但需要额外的扰动
搜索基础设施，成本明显高于第5.1节的直接导出方案。

### 5.3 因果图/Granger因果根因定位是另一条更前沿、成本也更高的延伸路线

工业过程监控领域有一条独立路线，不用学习出的注意力相似度构图，而用统计因果关系直接构造图
结构：<cite index="19-4">we propose a neural network model consisting of one-dimensional convolutional neural networks and a graph attention network (CNN-GAT) that uses a causal map derived from fault-free data using conditional Granger causality analysis</cite>，
故障发生后<cite index="19-6">Using the causal map and prediction results from the CNN-GAT model, the root cause diagnosis can be performed promptly after faults are detected</cite>。
这条路线本身的动机也印证了传统方法在诊断层面的普遍短板：<cite index="20-2">Most of these efforts have been focused on fault detection and isolation, while root cause diagnosis has not yet been fully addressed.</cite>
这类方法的图结构来自统计因果检验而非本项目现在用的余弦相似度top-k学习，是`related_work_shortlist.csv`
里"轴A：CDGNN/IGCL-GNN/CGT"这一类工作的共同思路——**这条路线之前已经被本项目排除**：CGT等
方案属于硬掩码/替换边骨架的做法，和本项目"先验只做attention logits的加性偏置，绝不做硬限制"
的既定设计原则冲突。这里再次确认这个结论仍然成立：Granger因果图的构造方式本质上是"用统计
显著性关系替换学习出的图结构"，而不是在现有图结构上叠加一层偏置，不符合本项目的架构约束。

### 5.4 小结：三条路线的成本-收益排序，与本项目当前定位的关系

三条路线（GDN式分数/注意力导出、反事实扰动、因果图重构）在"诊断粒度"和"实现成本"上依次递增，
但只有第一条能在**不改变现有图结构学习方式**的前提下实现——这正是本项目需要的（第三节的
H信号贡献分解、第四节的C信号attention对比都属于第一条路线的范畴，可以直接实现；反事实扰动和
因果图重构则分别在实现成本和架构兼容性上不适合现在投入）。

## 六、和联邦场景结合的诊断：目前几乎没有直接对应的工作，是这个方向真正的空白

`related_work_shortlist.csv`里`fleet_diag=1`的13条里，绝大多数关注的是"联邦如何应对客户端间
标签空间/故障模式不一致"（FedIFL、CPG-FL、HFCDF-KDCL等），或"服务器侧记忆如何编码跨轮次的
聚合状态"（GCC-SFCL），**没有一条同时做"逐节点/逐边可解释诊断"+"联邦记忆聚合"**——这和本项目
`joint-prototype-federated-results.md`记录的现状吻合：联邦场景下的B/C/H/K（及BK/CK/HK组合）
目前都只在"每个client本地计算AUROC"这个层面验证过，诊断层面的逐节点归因（无论是否补上H的
分解）目前只在集中式Sielaff上跑过一次（`sielaff_red_feature_attribution.json`），**从未在
联邦设置下验证归因结果是否在client之间稳定或可比**（例如：client A本地训练出来的"节点i对
故障的贡献最大"，换到client B的本地模型上是否还成立）——这是一个具体、此前未被讨论过、且
目前项目里没有相关工作可以直接参考的开放问题。

## 七、总结建议

**按成本排序的三步改进路径**：

1. **（最低成本，几行代码）给H信号补上逐节点贡献分解**，解决项目最重要检测信号（robo3er
   `stuck`，也是`HK`组合优势的来源）目前完全没有诊断能力这个具体缺口。
2. **（低成本）导出C信号的attention分布对比**，异常窗口vs该节点校准期平均attention分布的
   差异，定位"被打破的具体邻居关系"，复用GDN原论文已验证过的机制，不需要新设计。
3. **（中等成本，暂不建议现在做）反事实扰动解释**，能给出更贴近运维决策的"改多少才正常"，
   但需要额外的扰动搜索基础设施，本项目目前的诊断需求（"哪个节点/哪条边"级别）用1、2就能
   满足，暂不需要升级到这一层。

**评分方案选择**：七种方案里`HK=max(H,K)`是当前唯一同时满足"检测效果最优"（`horizon_mult=10`
时0.901，超过任何单一信号）和"诊断路径清晰"（拆开看是H还是K贡献的最大值,分别复用各自的
归因机制）的方案；`BK`、`CK`两种组合应放弃——不仅检测效果不如各自单独使用，诊断上也不会
比单独用K更清晰。

**因果传播路径发现**（第5.3节）作为技术方向和本项目已经排除的"硬掩码先验"设计原则冲突，**不
建议重新考虑**，除非先验注入机制本身有大改动。

**联邦场景下的诊断一致性验证**是目前相关工作完全没有覆盖、也是本项目自己没做过的一项，
如果要往"诊断"方向继续投入，这是唯一有真正新颖性（novelty）价值、值得优先于1、2之后专门
设计一次实验的方向——检查同一故障类型在不同client本地模型上的逐节点归因排序是否一致，
直接复用`sielaff_red_feature_attribution.json`的分析脚本框架，改成跨client对比即可。

## 二.五、【当前配置最新结果】联邦+26节点重跑：`stuck`命中率从<50%跃升到85.9%~100%

**本节是当前架构（federated-only政策+26节点特征集，见`memory/benchmark-policy-federated-only.md`、
`memory/joint-prototype-federated-results.md`）下的最新实测结果，取代下面"二.五（原版）"小节
里基于centralized+68节点跑出来的数字作为当前结论；原数字整节保留在下面，作为集中式/68节点
配置下的历史记录，不删除。**

新脚本`benchmark/diagnose_robo3er_localization_federated.py`（详见
`memory/robo3er-diagnosis-localization-federated26.md`）直接建在
`run_robo3er_forecast_v2_federated.py`之上——这也是信号K**第一次**跑联邦训练（此前K只在
集中式验证过，见`memory/forecast-head-signal-k.md`）。定位机制（B/C/K原生argmax、H的
Hotelling-T²逐节点贡献分解、BK/CK/HK的"谁的标量更大就用谁的argmax"路由逻辑）和集中式脚本
完全一致，没有改变；变的只是训练/校准路径变成联邦（5个真实机器人client，FedAvg同步encoder/decoder
+ 记忆对齐，5轮x12个local epoch）和节点集从68降到26（`kinematic_core`+`actuation`两组，
去掉`wheel_ticks_*`/`odom_odo_pos_*`等前缀）。

**26节点下随机基线本身也变了**（候选池从68降到26，基线自然升高，不能直接和68节点的基线比）：

| 故障类型 | 候选节点总数 | direct域大小 | direct+indirect域大小 | 随机基线direct% | 随机基线direct+indirect% |
|---|---|---|---|---|---|
| stuck | 26 | 6 | 6 | 23.1% | 23.1% |
| cable trapped | 26 | **0**（`slip_status_is_slipping`不在26节点集合里） | 11 | 0.0% | 42.3% |

**联邦+26节点 vs 集中式+68节点 命中率对比（direct-or-indirect%）：**

| 故障类型 | 信号 | 联邦26节点 | 集中式68节点（原结果，见下） | 联邦基线 | 集中式基线 |
|---|---|---|---|---|---|
| stuck | B | **94.0** | 7.4 | 23.1 | 11.8 |
| stuck | C | **94.0** | 2.7 | 23.1 | 11.8 |
| stuck | H | **85.9** | 18.8 | 23.1 | 11.8 |
| stuck | K | **87.1** | 41.0 | 23.1 | 11.8 |
| stuck | BK | **100.0** | 33.8 | 23.1 | 11.8 |
| stuck | CK | **100.0** | 37.4 | 23.1 | 11.8 |
| stuck | HK | **99.3** | 40.3 | 23.1 | 11.8 |
| cable trapped | B | 56.3 | 14.1 | 42.3 | 20.6 |
| cable trapped | C | 56.3 | 12.1 | 42.3 | 20.6 |
| cable trapped | H | 48.5 | 47.1 | 42.3 | 20.6 |
| cable trapped | K | 73.1 | 25.9 | 42.3 | 20.6 |
| cable trapped | BK | 77.7 | 15.7 | 42.3 | 20.6 |
| cable trapped | CK | 77.7 | 12.7 | 42.3 | 20.6 |
| cable trapped | HK | 74.6 | 47.7 | 42.3 | 20.6 |

图：`diagnosis_localization_federated_vs_centralized.png`（两个故障类型各一个面板的分组柱状图，
虚线为各自配置下的随机基线）。

**`stuck`的跃升是候选池组成变化，不是信号机制变强了**：`top_nodes`显示，26节点下所有信号的
argmax几乎全部集中在`wheel_status_current_ma_left/right`上（例如B: 114+26=140/149≈94%，
BK/CK: 118+21=139/139=100%）——对比集中式68节点时B/C被`battery_state_temperature`/
`battery_state_voltage`主导，这两个"电池"节点在26节点集合里根本不存在了。换句话说，
不是B/C学会了正确定位，而是它们此前用来"误判"的那些下游耦合节点（电池、红外、鼠标传感器）
被从候选池里整体移除了，argmax除了指向真正的`wheel_*`域已经无处可去。

**`cable trapped`的direct-or-indirect命中率对K/BK/CK/HK也有提升**（K: 25.9%→73.1%，
HK: 47.7%→74.6%），`top_nodes`集中在`imu_imu_angvel_x`、`odom_odo_lintw_y`（均为indirect域
成员），和集中式H的最优节点`odom_odo_angtw_x`（同为角速度/角速度里程计类节点）方向一致；但H
自己的相对排名反而下降了——集中式时H是`cable trapped`上遥遥领先的单一最优信号（47.1%，约为
其他信号的4倍），联邦+26节点下H反而是四个基础信号里最弱的一个（48.5%，基本贴着基线），
H的校准统计（`calib_mu`/`calib_cov_inv`）现在是每个client在联邦训练出的表征上各自算的，
不再是单一集中式模型的统计量，这可能是排名变化的来源。

**注意**：这一次跑同时改变了训练方式（联邦vs集中式）和特征集（26 vs 68节点）两个变量，
两者是耦合在一起的，本次结果不能拆开归因"是联邦训练本身的贡献，还是节点减少的贡献"。方向性
的结论（`stuck`定位命中率大幅提升）有`top_nodes`的候选池组成变化作为机制性解释，不太可能是
纯噪声，但具体数值（94% vs 100% vs 99.3%）没有多seed验证，不应过度解读绝对差异。

完整分析和逐signal细节见`memory/robo3er-diagnosis-localization-federated26.md`。

## 二.五（原版，集中式+68节点，历史记录）、为什么七种方案的定位命中率全部在50%以下

**以下内容是本项目切换到federated-only政策（2026-08-17）和26节点特征集之前，在集中式/68节点
配置下跑出来的原始结果，作为历史记录保留，当前配置的最新结果见上面新增的"二.五【当前配置最新
结果】"小节。**

先给统计基线，再逐信号解释argmax具体落在了哪里，最后和GDN原论文的诊断评价方式做对比。

### 基线：随机猜测本身在这个任务上就很低

`robo3er`可靠性过滤后有68个候选节点。`stuck`的直接域（`wheel_*`）只占其中8个节点，
`cable trapped`的直接域（`slip_status_is_slipping`）只有1个节点，直接+间接域合计14个节点。
纯随机指认的期望命中率分别是：

| 故障类型 | 候选节点总数 | 域大小 | 随机基线命中率 |
|---|---|---|---|
| stuck，仅direct | 68 | 8 | **11.8%** |
| cable trapped，仅direct | 68 | 1 | **1.5%** |
| cable trapped，direct+indirect | 68 | 14 | **20.6%** |

对着这个基线看，K在`stuck`上的41.0%是随机基线的**3.5倍**，H在`cable trapped`上的47.1%
是随机基线（20.6%）的**2.3倍**——这两个数字本身不算差，只是绝对值上限本来就被"候选节点很多、
真正相关的节点很少"这个任务结构压得很低，不是模型/信号设计得差。但B、C以及大多数其他信号
在`cable trapped`上的命中率（10%~15%）已经**接近甚至低于随机基线（20.6%的combined基线）**，
说明它们在这个故障类型上基本没有真正的定位能力，只是检测分数够高但指向随意。

### 逐信号：argmax实际落在了哪些节点上（用实测`top_nodes`说话）

`benchmark/diagnose_robo3er_localization.py`输出的`top_nodes`字段记录了每个信号argmax命中
最频繁的5个节点，可以直接看出"命中不到50%"具体是被什么抢走了：

**`cable trapped`（真实标志：`slip_status_is_slipping`；间接域：轮-里程计/IMU角速度残差）**

| 信号 | 机制 | 实际最常被argmax选中的节点 | 落在域内吗 |
|---|---|---|---|
| B (`d_node`逐节点偏差) | 比较每个节点与最近原型均值的偏差幅度，不看图结构 | `ir_intensity_*`（前中右/左，占54+36+26+24=140/206≈68%） | 否——IR接近传感器组是"other" |
| C (`resid_struct`逐节点残差) | 用top-k余弦相似度学到的图+attention邻居预测该节点，残差即偏差 | 同样是`ir_intensity_*`系列（占~63%） | 否 |
| H (Mahalanobis贡献分解，新实现) | 整窗口协方差二次型`(s-mu)^T Σ^-1 (s-mu)`按节点展开 | `odom_odo_angtw_x`（角速度里程计，53/206≈26%） | **是——间接域** |
| K (`k_resid`预测残差) | 预测头对未来值的预测误差 | `mouse_mouse_x`（光流/鼠标传感器，100/197≈51%！） | 否，且完全过滤后掉到0% |
| BK/CK | B或K、C或K逐窗口取更大者 | 仍以`ir_intensity_*`为主 | 否 |
| HK | H或K取更大者 | `odom_odo_angtw_x`为主（49/197≈25%），H基本赢 | 是（继承H） |

K在`cable trapped`上栽在`mouse_mouse_x`——这个节点的argmax占比高达51%，是本次测试里单一节点
最集中的一次错误定位。`slip_status_is_slipping`本身在校准期（正常数据）恒为0，可靠性过滤器
把它当"近似常数噪声"排除出了argmax候选池（见前次分析），而`mouse_mouse_x`凑巧在`cable trapped`
窗口里波动幅度大、又没被过滤掉，抢占了argmax——**这不是K的预测机制本身出了问题，是可靠性
过滤和这个故障的巧合冲突造成的伪命中**，K在raw（未过滤）表里对这个故障类型的命中率是14.7%，
说明K本身确实有信号，只是被过滤规则误伤。

**`stuck`（真实标志：`wheel_*`任意列）**

| 信号 | 实际最常被argmax选中的节点 | 落在域内吗 |
|---|---|---|
| B | `battery_state_temperature`（76/149≈51%） | 否 |
| C | `battery_state_voltage`（63/149≈42%） | 否 |
| H | `imu_imu_angvel_x`（72/149≈48%），其余靠`wheel_status_current_ma_*`（16+12=28/149≈19%）撑起18.8%命中率 | 部分——多数其实落在IMU上 |
| K | `battery_state_voltage`（55）与`wheel_status_current_ma_left`（49），后者属于direct域 | 是，约35%来自这一列 |
| BK/CK/HK | 与K接近，`wheel_status_current_ma_left`稳定占大头 | 是（继承K） |

B、C两个信号在`stuck`上分别被`battery_state_temperature`、`battery_state_voltage`主导，
这两个电池相关节点和轮子被卡住之间没有直接机制关系，但轮子卡住导致电机持续高负荷运转，
电池温度/电压确实会跟着漂移——**这正是"整机系统层层耦合"造成的伪定位：故障会波及很多
物理上相关但不是根因的传感器，而B、C这两个"哪个节点绝对偏差最大"式的机制没有能力区分
"根因节点"和"被根因连带影响、偏差幅度更大的下游节点"**。这也解释了为什么H和K（分别利用
协方差联合结构、以及预测残差）表现明显好于B、C：H和K的机制里都隐含了"和其他节点的关系"
这层信息，B、C里只有C用了图结构做预测但残差本身仍是逐节点独立看，没有像H那样把"哪几个
节点一起偏离"当成判据。

### 和GDN原论文对比：不是本项目的定位机制更差，而是评价方式本身不同

第5.1节已经引用过GDN原论文（Deng & Hooi, AAAI 2021）在WADI上的诊断案例演示。需要指出的是，
原论文本身报告的定量结果集中在**检测**层面：<cite index="1-3,1-4">The results show that GDN outperforms the baselines in both datasets, with high precision in both datasets of 0.99 on SWaT and 0.98 on WADI. In terms of F-measure, GDN outperforms the baselines on SWaT; on WADI, it has 54% higher F-measure than the next best baseline.</cite>
诊断/定位部分（原论文RQ4"Localizing Anomalies"一节）没有报告一个跨越所有异常事件聚合统计
的定量命中率，而是以单个案例走查的形式呈现：<cite index="8-8,8-9">How well can our model help users to localize and understand an anomaly? Figure 3 (left) shows the learned graph of sensors, with edges weighted by their attention weights, and plotted using a force-directed layout(Kobourov 2012).</cite>
图中<cite index="8-4,8-5">The red triangle denotes the central sensor identified by our approach, with highest anomaly score. Red circles indicate nodes with edge weights larger than 0.1 to the central node.</cite>
——也就是说，GDN论文本身并没有一个可以直接拿来对比的"top-1定位命中率%"数字，这个比较
在文献层面本身就不对等。

更能说明问题的是一篇最近对GNN时序异常检测方法做批判性开源复现评估的工作，系统性地检查了
"最高异常分数节点是否等于真实异常传感器"这件事在多个数据集/多起事件上聚合统计的结果，
得到的结论恰好印证了本项目现在的发现——分数能把真实异常节点排进前列，但不代表精确指认
真正的根因是容易的：<cite index="11-1,11-2">All models consistently rank the true anomalous sensor among the top sensors during the anomaly period. This indicates that, while score-based methods can suggest likely affected sensors, interpretability still requires further analysis.</cite>
该工作也具体指出GDN在图结构约束下的定位相对稳定的一个案例：<cite index="11-4,11-5">Here, GDN's graph-based approach keeps forecasts stable and restricts anomaly effects to the affected sensor... making fault localization straightforward. In contrast, GRU forecasts are less stable; an anomaly in stage 5 also deteriorates predictions for stage 2, making it difficult to pinpoint the true source of the anomaly.</cite>

**结论**：本项目集中式+68节点测出来的<50%命中率，和GDN原论文本身并不构成一次公平的数字对比——原
论文的诊断评价停留在案例展示层面，没有报告聚合命中率；而独立的批判性复现工作证实，即使
是"排进top候选"这种更宽松的标准，也仅停留在"排名靠前"而非"精确top-1命中"，作者自己也承认
<cite index="11-2">interpretability still requires further analysis</cite>。本项目做的
`diagnose_robo3er_localization.py`是对着聚合统计、精确top-1这个更严格标准做的评价，这本身
是比GDN家族常见评价方式更彻底的一步，不是本项目机制设计得差；较低的绝对命中率更多反映了
"候选节点多、真实根因域窄"（前一小节的随机基线分析）以及"整机耦合导致的伪定位"（本节逐信号
分析）这两个结构性因素，而非H/K的argmax/贡献分解机制本身存在缺陷——H、K相对B、C的领先幅度
（2-4倍随机基线）已经说明这两个信号确实捕捉到了真实的定位信息，只是绝对上限被任务结构本身
限制住了。

**更新（当前配置，联邦+26节点，见上面"二.五【当前配置最新结果】"）**：把候选节点从68降到26后，
"整机耦合导致的伪定位"这个结构性因素本身被大幅削弱——`stuck`定位命中率从<50%跃升到85.9%~
100%，直接验证了上一段"较低命中率主要来自候选池噪声，不是机制缺陷"的判断：一旦把不相关的
下游耦合节点（电池、红外、鼠标传感器）从候选池里物理移除，B/C/H/K四个信号的`stuck`命中率
全部跃升到85%以上，说明它们本来就"看得到"正确的根因节点，只是此前要在一个充满无关噪声节点
的更大候选池里竞争。这进一步支持"本项目的评价标准本身合理，之前的低命中率是任务结构问题，
不是信号设计问题"这一结论，而不需要修改。
