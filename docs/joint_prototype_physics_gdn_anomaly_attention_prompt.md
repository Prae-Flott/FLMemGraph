# 设计提示词：Joint Prototype Memory + Physics-Relation GDN + 三层异常检测

> 用途：作为下一步模型设计、算法推导、代码生成、实验规划与论文方法章节的统一设计提示词。
> 核心目标：在不要求精确物理参数的情况下，将**物理变量之间可靠的关系结构**、**Joint Prototype 正常工况记忆**与 **GDN 的数据驱动关系学习**结合起来，实现可定位、可解释、可扩展到联邦学习的无监督异常检测。

---

## 0. 核心思想

模型不以"精确掌握理论物理参数"为核心目标。

现实设备中的理论参数往往因制造公差、装配差异、老化、摩擦系数变化、温度影响以及传感器标定误差以及设备异质性的影响。因此不应强制：

\[
y=f_{\text{physics}}(x;\theta_{\text{theory}})
\]

并直接使用：

\[
|y-f_{\text{physics}}(x;\theta_{\text{theory}})|
\]

作为主要异常残差。

更合理的是：

\[
\boxed{\text{Physics defines who should relate and how they should relate}}
\]

\[
\boxed{\text{Data learns how this relationship is realized on the real device}}
\]

即：

\[
\boxed{
G_{\text{physics}}+R_{\text{type}}+P_m+\phi_\theta
}
\]

其中：

- \(G_{\text{physics}}\)：哪些 feature 之间存在可靠物理关系；
- \(R_{\text{type}}\)：关系属于比例、积分、微分、动力学、耦合、非线性等哪一类；
- \(P_m\)：当前属于哪一种正常 operating condition；
- \(\phi_\theta\)：该关系在真实设备中的 data-driven realization。

最终异常定义为：

\[
\boxed{
\text{Anomaly}=\text{deviation from normal joint state}+\text{deviation from normal physical relationships}
}
\]

---

## 1. 输入与共享时序编码器

假设设备包含 \(N\) 个 feature：

\[
X_t=\{x_1^{t-W:t},x_2^{t-W:t},\dots,x_N^{t-W:t}\}
\]

所有 feature 使用同一个 channel-independent temporal encoder：

\[
z_i=f_{\theta_E}(x_i^{t-W:t})
\]

其中：

\[
z_i\in\mathbb R^D
\]

表示 feature \(i\) 当前窗口的 latent representation。

整个设备当前状态：

\[
\boxed{Z_t=[z_1,z_2,\dots,z_N]}
\]

每个 feature 同时具有可学习 node identity embedding：

\[
v_i\in\mathbb R^{d_v}
\]

用于：

- 表示 feature identity；
- graph topology learning；
- attention conditioning；
- shared predictor / message function conditioning。

Memory 与 GDN 必须共用同一个 \(z_i\)，避免重复时序编码。

---

## 2. Joint Prototype Memory

### 2.1 Prototype 必须保留 feature 间联合状态

不要使用完全独立的 per-feature prototype dictionary。

定义第 \(m\) 个 Joint Prototype：

\[
\boxed{P_m=[p_{m,1},p_{m,2},\dots,p_{m,N}]}
\]

其中：

\[
p_{m,i}\in\mathbb R^D
\]

表示：

> 在正常 operating condition \(m\) 下，feature \(i\) 应该具有的典型 latent state。

因此：

\[
P_m\in\mathbb R^{N\times D}
\]

Joint Prototype 表示的是一个完整的正常多变量 operating state，而不是单独某个 sensor pattern。

---

## 3. Joint Prototype Retrieval

当前状态：

\[
Z=[z_1,\dots,z_N]
\]

与第 \(m\) 个 prototype 的距离：

\[
D(Z,P_m)=\sum_i w_iD_i(z_i,p_{m,i})
\]

找到：

\[
\boxed{m^*=\arg\min_mD(Z,P_m)}
\]

得到：

\[
P^*=P_{m^*}
\]

表示当前设备最接近的已知正常 operating condition。

整体 novelty：

\[
\boxed{d_G=D(Z,P^*)}
\]

但是 \(d_G\gg0\) 既可能是 new normal，也可能是 fault，所以 memory distance 不能单独完成 anomaly decision。

---

## 4. 第一层异常：Node Anomaly

对于：

\[
P^*=[p_1^*,\dots,p_N^*]
\]

定义：

\[
\boxed{S_i^{node}=D_i(z_i,p_i^*)}
\]

其含义为：

> 在当前最接近的正常 operating condition 下，feature \(i\) 是否符合该状态？

Node anomaly 用于：

- sensor-level localization；
- component fault indication；
- feature contribution analysis。

---

## 5. Physics Graph

建立：

\[
G=(V,E)
\]

其中每个节点对应一个 feature。

Physics knowledge 主要提供 topology，即哪些变量应该存在直接关系，例如：

\[
Current\rightarrow Torque
\]

\[
Torque\rightarrow Speed
\]

\[
Speed\rightarrow Vibration
\]

\[
Force\rightarrow Vibration
\]

Graph candidate edges 推荐：

\[
\boxed{E=E_{\text{physics}}\cup E_{\text{learned}}}
\]

其中：

- physics edge 保证已知关系不会被 GDN 忽略；
- learned edge 用于补充未知规律。

---

## 6. GDN 应该学习哪些参数

GDN 不主要负责 identification 精确机械参数，而应该学习真实设备中的正常关系 realization。

### 6.1 Temporal Encoder

\[
\theta_E
\]

学习：

\[
x_i^{t-W:t}\rightarrow z_i
\]

### 6.2 Node Embedding

\[
\boxed{v_i}
\]

学习：

- feature identity；
- feature compatibility；
- optional learned topology。

### 6.3 Normal Relation Attention

定义：

\[
\boxed{\alpha_{ij}^{rel}}
\]

表示：

> 在当前正常 operating condition 下，节点 \(j\) 对节点 \(i\) 的正常状态有多重要。

例如：

\[
e_{ij}^{rel}=a_\theta(z_i,z_j,v_i,v_j,q_{r_{ij}},c_m)
\]

\[
\alpha_{ij}^{rel}=softmax_j(e_{ij}^{rel})
\]

其中：

- \(q_{r_{ij}}\)：relation type embedding；
- \(c_m\)：prototype / operating-condition context。

### 6.4 Relation-Specific Message Function

定义：

\[
\boxed{m_{ij}=\alpha_{ij}^{rel}\phi_{r_{ij}}(z_j,c_m)}
\]

其中 \(\phi_{r_{ij}}\) 由关系类型决定。

### 6.5 Graph Aggregation / Conditional Predictor

节点 \(i\) 的正常 latent prediction：

\[
\hat z_i=g_\theta\left(\sum_{j\in N(i)}m_{ij};v_i,c_m\right)
\]

因此 GDN 学习：

\[
\boxed{p(z_i\mid z_{N(i)},\text{condition})}
\]

即正常设备中，根据相关 feature 和 operating condition，feature \(i\) 应该处于什么样的状态。

---

## 7. 已知 Physics Edge 如何体现

Edge 不应该只是：

\[
A_{ij}=1
\]

而应该包含 relation type：

\[
\boxed{e_{ij}=(i,j,r_{ij})}
\]

关系类型例如：

\[
r_{ij}\in\{\text{proportional},\text{monotonic},\text{integral},\text{derivative},\text{dynamic},\text{thermal},\text{frequency},\text{nonlinear},\text{unknown}\}
\]

---

## 8. 三个 Physics Knowledge Level

### Level 1 — Topology Knowledge

只知道：

\[
x_i\leftrightarrow x_j
\]

此时只固定 graph connectivity。

### Level 2 — Relation-Type Knowledge

知道：

\[
y\propto x
\]

或者：

\[
y\sim\int xdt
\]

但不知道可靠参数。

此时使用 relation type embedding 和 relation-specific message function，参数由正常数据学习。

### Level 3 — Reliable Parameter Knowledge

只有当理论参数的正确性确认时，才加入显式 physics constraint：

\[
y=kx
\]

例如：

\[
L_{phys}=\|y-kx\|^2
\]

因此：

\[
\boxed{\text{Level 3 if reliable; otherwise Level 2; otherwise Level 1}}
\]

---

## 9. Relation-Specific Message Functions

### Proportional-like Relation

例如：

\[
Current\rightarrow Torque
\]

可以：

\[
\phi_{prop}(z)=W_{prop}z+b_{prop}
\]

不要求：

\[
W_{prop}=k_t^{theory}
\]

而是让正常数据学习真实设备中的关系。

### Integral-like Relation

例如：

\[
Acceleration\rightarrow Velocity
\]

需要 temporal history：

\[
\phi_{int}=TemporalAggregator(z_a^{t-L:t})
\]

其 inductive bias 是该关系具有 accumulation / memory 特征。

### Derivative-like Relation

例如：

\[
Velocity\rightarrow Acceleration
\]

使用：

\[
\phi_{diff}=TemporalDifference(z_v^t,z_v^{t-1})
\]

### Dynamic Relation

例如：

\[
Torque\rightarrow Speed
\]

使用带状态的轻量 temporal mapping：

\[
\phi_{dyn}(z_\tau^{t-L:t},c_m)
\]

### Nonlinear Relation

例如：

\[
Force,Speed\rightarrow Vibration
\]

允许：

\[
\phi_{nonlinear}=MLP/TemporalMLP
\]

但 topology 与 relation semantics 由 physics knowledge 提供。

---

## 10. Prototype 作为 Edge Context

同一条 physical relation 在不同 operating condition 中不一定完全相同。

因此：

\[
\boxed{\phi_{ij}=\phi_{ij}(z_j,c_m)}
\]

其中 \(c_m\) 来自当前最近 Joint Prototype。

例如 \(Speed\rightarrow Vibration\) 在 900 rpm 与 1500 rpm 工况下应该具有不同正常 realization。

因此模型学习：

\[
\boxed{p(z_i\mid z_j,\text{operating condition})}
\]

---

## 11. 第二层异常：Edge Anomaly

对于每条：

\[
j\rightarrow i
\]

单独产生该边 prediction：

\[
\hat z_i^{(j)}=\phi_{ij}(z_j,c_m)
\]

定义 edge residual：

\[
\boxed{r_{ij}=D(z_i,\hat z_i^{(j)})}
\]

它表示当前这条 feature relation 与正常关系相比偏离多少。

Edge anomaly 是 relationship anomaly，而不是简单 sensor anomaly。

---

## 12. Edge Residual 标准化

不同 physics edge 的自然 variability 不同，因此不能直接比较 \(r_{ij}\)。

基于学习正常状态下：

\[
\mu_{ij}^{(m)}
\]

以及：

\[
\sigma_{ij}^{(m)}
\]

标准化：

\[
\boxed{\tilde r_{ij}=\frac{r_{ij}-\mu_{ij}^{(m)}}{\sigma_{ij}^{(m)}+\epsilon}}
\]

于是 \(\tilde r_{ij}\) 表示当前 edge deviation 相对于该工况正常分布有多异常。

---

## 13. Normal Relation Attention 与 Anomaly Attention 必须分开

\[
\boxed{\alpha_{ij}^{rel}\neq\beta_{ij}^{anom}}
\]

二者回答的是完全不同的问题。

---

## 14. Normal Relation Attention

\[
\alpha_{ij}^{rel}
\]

回答：

> 正常情况下，哪个邻居对节点 \(i\) 更重要？

用于：

\[
\hat z_i=\sum_j\alpha_{ij}^{rel}\phi_{ij}(z_j,c_m)
\]

训练目标可以使用：

\[
L_{rel}=\sum_i\|z_i-\hat z_i\|^2
\]

或 SmoothL1。

---

## 15. Anomaly Attention

定义第二套 attention：

\[
\boxed{\beta_{ij}^{anom}}
\]

回答：

> 当前异常中，哪条关系被破坏得最值得关注？

可以定义：

\[
u_{ij}^{anom}=MLP_{anom}(\tilde r_{ij},z_i,z_j,q_{r_{ij}},c_m)
\]

然后：

\[
\boxed{\beta_{ij}^{anom}=softmax_j(u_{ij}^{anom})}
\]

最终节点级 edge anomaly：

\[
\boxed{S_i^{edge}=\sum_{j\in N(i)}\beta_{ij}^{anom}\tilde r_{ij}}
\]

---

## 16. 第一版 Anomaly Attention：无需额外训练

纯无监督 baseline 可以使用：

\[
\boxed{\beta_{ij}^{anom}=\frac{\exp(\gamma\tilde r_{ij})}{\sum_k\exp(\gamma\tilde r_{ik})}}
\]

其中 \(\gamma\) 控制 attention sharpness。

这样越远离自身正常 edge distribution 的关系，获得越高 anomaly attention。

---

## 17. 第二版：Self-Supervised Anomaly Attention

通过 relation-breaking augmentation 人工制造 edge anomaly。

重点不是简单添加随机噪声，而是破坏 feature relationship。

### Temporal Mismatch

\[
(x_i(t),x_j(t))\rightarrow(x_i(t),x_j(t+\Delta))
\]

### Cross-Condition Swap

将：

\[
x_i^{condition A}
\]

与：

\[
x_j^{condition B}
\]

组合，制造 individually normal but relationally inconsistent 样本。

### Scaling Perturbation

对于 proportional-like edge：

\[
x_j\rightarrow\lambda x_j
\]

### Trend Reversal

针对 monotonic relation 破坏正常趋势。

### Frequency Perturbation

针对：

\[
Speed\rightarrow Vibration
\]

修改 vibration spectral structure，使其与当前 speed 不匹配。

这些 synthetic relation anomalies 可以产生：

\[
y_{ij}^{edge}
\]

用于训练 anomaly attention。

---

## 18. 三层异常结果

### Level 1 — Node Anomaly

\[
\boxed{S_i^{node}=D(z_i,p_i^*)}
\]

含义：当前 feature 是否偏离当前 normal operating prototype？

### Level 2 — Edge Anomaly

保留每条：

\[
\boxed{S_{ij}^{edge}=\tilde r_{ij}}
\]

同时节点级 edge anomaly：

\[
\boxed{S_i^{edge}=\sum_j\beta_{ij}^{anom}\tilde r_{ij}}
\]

含义：哪些 feature relationships 被破坏？

### Level 3 — Subgraph / Device Anomaly

节点最终 anomaly representation：

\[
\boxed{s_i=[S_i^{node},S_i^{edge}]}
\]

全图：

\[
\boxed{S_G=AGG_i(s_i)}
\]

可以使用：

- max；
- weighted sum；
- learned graph pooling；
- lightweight GDN/GAT aggregation。

---

## 19. 三层异常判定逻辑

### Known Normal

\[
d_G\ low,\quad S_{node}\ low,\quad S_{edge}\ low
\]

### Local Node Anomaly

\[
d_G\ low/moderate
\]

但某些：

\[
S_i^{node}\ high
\]

进一步检查相关 edge。

### Novel but Physically Consistent

\[
d_G\ high,\quad S_{node}\ high,\quad S_{edge}\ low
\]

解释为：

\[
\boxed{\text{new normal operating condition candidate}}
\]

### Fault / Structural Anomaly

\[
d_G\ high,\quad S_{edge}\ high
\]

通常伴随多个：

\[
S_i^{node}\ high
\]

判断：

\[
\boxed{\text{high-confidence fault}}
\]

---

## 20. Long-Term Relationship Drift

除瞬时 edge residual 外，可以进一步监控 relation drift。

正常 baseline：

\[
R_{ij}^{baseline}
\]

当前长窗口：

\[
R_{ij}^{current}
\]

定义：

\[
\boxed{S_{ij}^{drift}=D(R_{ij}^{current},R_{ij}^{baseline})}
\]

如果使用显式 learned parameter：

\[
S_{ij}^{drift}=D(\theta_{ij}^{current},\theta_{ij}^{baseline})
\]

如果不使用显式参数，则比较：

- edge embedding；
- relation response distribution；
- normalized residual distribution；
- attention statistics。

用于区分 sudden fault 与 gradual degradation。

---

## 21. 整体计算流程

```text
                 Multivariate time series
                          |
                          v
                 Shared temporal encoder
                          |
                 z1 z2 ... zN
                          |
          +---------------+----------------+
          |                                |
          v                                v
 Joint Prototype Memory              Physics Graph
          |                                |
          v                                v
 nearest normal P*               typed physics edges
          |                                |
   +------+-------+              relation-specific phi_ij
   |              |                       |
   v              v                       v
global dG      node refs pi*         edge prediction
                  |                       |
                  v                       v
             Node anomaly              r_ij
               Si_node                  |
                                        v
                               prototype-conditioned
                                  normalization
                                        |
                                        v
                                  r_tilde_ij
                                        |
                         +--------------+-------------+
                         |                            |
                         v                            v
                relation attention alpha      anomaly attention beta
                "who matters normally"       "who is broken now"
                                                      |
                                                      v
                                               Edge anomaly
                                                      |
                         +----------------------------+
                         v
              [Node anomaly, Edge anomaly]
                         |
                         v
                   Graph aggregation
                         |
                         v
              Subgraph / Device anomaly
```

---

## 22. 推荐训练阶段

### Stage A — Normal Representation Learning

训练：

- temporal encoder；
- joint prototypes；
- node embedding；
- normal relation attention；
- relation-specific message functions；
- graph predictor。

目标：

\[
L_A=L_{memory}+\lambda_{rel}L_{relation}
\]

### Stage B — Edge Statistics Estimation

对每个 \(P_m\) 统计：

\[
\mu_{ij}^{(m)},\sigma_{ij}^{(m)}
\]

用于 standardized edge anomaly。

### Stage C — Optional Self-Supervised Anomaly Attention

通过 relation-breaking augmentation 训练：

\[
Attention_{anom}
\]

让模型识别哪一条 physics relationship 被人为破坏。

---

## 23. 为未来 FL 保留的分层

未来联邦层建议区分三类知识。

### Global Knowledge

共享：

\[
\boxed{G_{physics}}
\]

和：

\[
\boxed{R_{type}}
\]

因为同类设备共享变量关系结构。

### Personalized Relation Realization

本地保留轻度个性化：

\[
\boxed{\phi_{ij}^{device}}
\]

因为不同设备的实际参数和 response 可以不同。

### Federated Memory

跨设备共享：

\[
\boxed{P_m}
\]

表示 Fleet collectively observed normal operating conditions。

因此未来 FL 的基本思想不是：

\[
\text{average all physical parameters}
\]

而是：

\[
\boxed{\text{shared physics semantics}+\text{personalized relation realization}+\text{federated normal experience}}
\]

---

## 24. 必做消融实验

至少比较：

1. GDN learned topology only；
2. Physics topology + standard GDN；
3. Physics topology + relation type；
4. Physics topology + relation-specific message；
5. Joint Prototype only；
6. Joint Prototype + Node anomaly；
7. Edge anomaly only；
8. Joint Prototype + Node + Edge；
9. Normal relation attention only；
10. Residual-based anomaly attention；
11. Learnable self-supervised anomaly attention；
12. Theory-parameter residual vs relationship-aware learned residual。

重点验证：

- 是否减少 false positive；
- 是否改善 new-normal vs fault separation；
- 是否提高 fault localization；
- 是否能定位异常 edge；
- 是否对理论参数误差更 robust；
- 是否对不同 operating condition 更 robust。

---

## 25. 最终方法定义

整个方法可以定义为：

\[
\boxed{\textbf{Prototype-conditioned Physics-Relation Graph Anomaly Detection}}
\]

其基本逻辑：

1. **Joint Prototype** 判断当前系统最接近哪种正常 operating condition；
2. **Node anomaly** 判断各 feature 是否符合该正常工况；
3. **Physics-relation GDN** 学习真实设备中 feature relationship 的正常 realization；
4. **Edge residual** 判断当前 relation 是否偏离正常；
5. **Anomaly attention** 判断哪些 broken relationships 对当前异常最关键；
6. **Graph aggregation** 综合 node 与 edge evidence，形成 subsystem/device anomaly。

核心思想：

\[
\boxed{\text{Memory tells us what normal state we expect}}
\]

\[
\boxed{\text{Physics tells us which relationships should exist}}
\]

\[
\boxed{\text{GDN learns how these relationships look on the real device}}
\]

\[
\boxed{\text{Anomaly attention tells us which relationships are breaking now}}
\]

最终不要求：

\[
\theta_{real}=\theta_{theoretical}
\]

而要求：

\[
\boxed{R_{current}\approx R_{normal}}
\]

正常；如果：

\[
\boxed{R_{current}\not\approx R_{normal}}
\]

则产生 relationship-level anomaly，并与 Joint Prototype 的 node-level deviation 一起决定最终结果。
