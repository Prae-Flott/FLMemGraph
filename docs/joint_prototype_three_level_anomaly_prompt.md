# 设计提示词：Joint Prototype Memory + 三层异常检测

> 用途：作为下一步模型设计、代码实现、算法推导和收敛实验的统一提示词。
> 当前范围：只聚焦 **Joint Prototype Memory** 与 **Node / Edge / Subgraph 三层异常检测**。
> 暂不讨论 feature space 很大、原型压缩、subgraph factorization、通信量优化等扩展问题。

---

## 0. 核心目标

搭建一个面向多变量时序设备的无监督异常检测框架，使模型同时回答三个不同层级的问题：

1. **当前整体状态是否属于某个已知正常工况？**
2. **在该正常工况下，具体哪个 feature 偏离了正常状态？**
3. **feature 之间的物理/结构关系是否被破坏？**

整体设计遵循：

\[
\boxed{
\text{Joint Prototype Memory}
+
\text{Node Anomaly}
+
\text{Edge Anomaly}
+
\text{Subgraph / Device Anomaly}
}
\]

其中：

- Joint Prototype 负责表示一个正常工况下多个 feature 的**联合状态**；
- Node anomaly 负责定位单个 feature 是否偏离该联合工况；
- Edge anomaly 负责判断 feature 之间已知的物理或结构关系是否失效；
- Subgraph / device anomaly 将 node 与 edge 两类异常证据聚合成整机判断。

---

# 1. 输入与共享时序编码器

假设设备包含 \(N\) 个 feature：

\[
X_t=
\{
x_1^{t-W:t},
x_2^{t-W:t},
\ldots,
x_N^{t-W:t}
\}
\]

其中每个 \(x_i^{t-W:t}\) 是 feature \(i\) 在长度为 \(W\) 的时间窗口内的时序数据。

所有 feature 共享一个跨通道的时序编码器：

\[
z_i=f_\theta(x_i^{t-W:t})
\]

其中：

\[
z_i\in\mathbb R^D
\]

表示 feature \(i\) 当前时间窗口的 latent representation。

整个设备当前的 latent state 定义为：

\[
\boxed{
Z_t=
[z_1,z_2,\ldots,z_N]
}
\]

注意：

- encoder 在不同 feature 之间共享参数；
- 每个 feature 仍保留自己的节点身份 embedding \(v_i\)；
- 后续 memory 与 graph 两个分支都使用同一组 \(z_i\)，避免重复编码。

---

# 2. Joint Prototype Memory

## 2.1 不使用 per-feature prototype

不要为每个 feature 单独维护独立 prototype dictionary：

\[
P_1,P_2,\ldots,P_N
\]

因为这样会把不同 feature 的正常模式分开建模，无法直接保留：

\[
x_i \leftrightarrow x_j
\]

之间的联合工况信息。

本设计使用：

\[
\boxed{
\text{Joint Prototype}
}
\]

即一个 prototype 表示多个 feature 在某一种正常 operating condition 下的联合 latent state。

---

## 2.2 Joint Prototype 定义

第 \(m\) 个 joint prototype 定义为：

\[
\boxed{
P_m=
[
p_{m,1},
p_{m,2},
\ldots,
p_{m,N}
]
}
\]

其中：

\[
p_{m,i}\in\mathbb R^D
\]

表示在第 \(m\) 个正常工况下，feature \(i\) 应该具有的典型 latent state。

因此：

\[
P_m\in\mathbb R^{N\times D}
\]

一个 prototype 不再表示"某个 feature 的典型模式"，而是表示：

\[
\boxed{
\text{一个完整的正常多变量设备状态}
}
\]

例如：

\[
P_1
=
\{
\text{Current},
\text{Torque},
\text{Speed},
\text{Force},
\text{Vibration},
\text{Temperature}
\}_{\text{Operating Condition 1}}
\]

\[
P_2
=
\{
\text{Current},
\text{Torque},
\text{Speed},
\text{Force},
\text{Vibration},
\text{Temperature}
\}_{\text{Operating Condition 2}}
\]

因此不同 feature 之间的联合统计状态被保留在同一个 prototype 内。

---

# 3. Joint Prototype Retrieval

对当前设备 latent state：

\[
Z_t=[z_1,\ldots,z_N]
\]

计算其与每个 joint prototype 的距离：

\[
D(Z_t,P_m)
\]

例如可以使用：

\[
D(Z_t,P_m)
=
\sum_{i=1}^{N}
w_i
D_i(z_i,p_{m,i})
\]

其中：

- \(w_i\) 为 feature 权重；
- \(D_i\) 可使用 Euclidean distance、cosine distance 或 normalized latent distance。

选择最近 prototype：

\[
\boxed{
m^*
=
\arg\min_m
D(Z_t,P_m)
}
\]

并得到当前匹配的正常工况：

\[
\boxed{
P^*=P_{m^*}
}
\]

这一过程回答：

> 当前设备整体上最接近哪一个已知正常 operating condition？

---

# 4. Global / Joint Memory Distance

首先定义整体 memory novelty：

\[
\boxed{
d_G
=
D(Z_t,P^*)
}
\]

它表示：

> 当前整体状态离最近已知正常 joint prototype 有多远。

解释：

### 已知正常工况

\[
d_G\approx0
\]

### 未见过的新工况

\[
d_G\gg0
\]

### 故障

也可能：

\[
d_G\gg0
\]

因此：

\[
\boxed{
d_G
\text{ 只能表示 novelty，不能单独区分 new normal 与 fault}
}
\]

这正是后续 edge anomaly / physics consistency 必须存在的原因。

---

# 5. 第一层异常：Node Anomaly

虽然 prototype 是 joint 的，但 anomaly score 可以拆分到每个 feature。

对于当前选中的 prototype：

\[
P^*
=
[
p_1^*,p_2^*,\ldots,p_N^*
]
\]

定义节点 \(i\) 的 anomaly score：

\[
\boxed{
S_i^{node}
=
D_i(z_i,p_i^*)
}
\]

也可以记作：

\[
d_i
=
D_i(z_i,p_i^*)
\]

其含义不是：

> feature \(i\) 是否偏离历史上的整体正常数据。

而是：

\[
\boxed{
\text{在当前匹配的整体正常工况 }P^*
\text{ 中，feature }i\text{ 是否符合该场景？}
}
\]

这使 node anomaly 天然带有 operating-condition context。

---

## 5.1 Node anomaly 示例

假设当前 joint prototype 对应：

\[
P^*
=
\text{1500 rpm + 1000 N + normal bearing}
\]

其中 vibration 的 prototype component 为：

\[
p_a^*
\]

当前 vibration latent 为：

\[
z_a
\]

则：

\[
S_a^{node}
=
\|z_a-p_a^*\|
\]

如果：

\[
S_a^{node}\gg0
\]

表示：

> vibration 在"当前整体工况"下异常。

而不是简单：

> vibration 绝对幅值很大。

---

# 6. 第二层异常：Edge Anomaly

Node anomaly 只能表示：

\[
z_i
\]

是否偏离当前 prototype。

但真正的物理系统还包含 feature 之间的关系：

\[
x_i\rightarrow x_j
\]

例如：

\[
I\rightarrow Torque
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

这些关系必须通过 graph edge 明式表达。

---

## 6.1 Typed Physics Edge

每条 edge 定义为：

\[
\boxed{
e_{ij}
=
(i,j,r_{ij},\theta_{ij})
}
\]

其中：

- \(i,j\)：源节点和目标节点；
- \(r_{ij}\)：relation type；
- \(\theta_{ij}\)：物理关系参数。

relation type 可以包括：

- proportional；
- inverse proportional；
- integral；
- derivative；
- linear dynamics；
- nonlinear dynamics；
- thermal dynamics；
- frequency scaling；
- learned soft relation。

每种 relation 对应一个 edge operator：

\[
\phi_{ij}
\]

---

## 6.2 Edge Physics Prediction

对于 edge：

\[
i\rightarrow j
\]

使用：

\[
\hat z_j^{(i)}
=
\phi_{ij}
(
z_i,
\text{history},
\theta_{ij}
)
\]

或在 raw physical space：

\[
\hat x_j^{(i)}
=
\phi_{ij}
(
x_i,
\text{history},
\theta_{ij}
)
\]

例如：

### 比例关系

\[
Torque=k_t I
\]

则：

\[
\hat \tau=k_t I
\]

### 积分关系

\[
v_t
=
v_{t-1}
+
a_t\Delta t
\]

### 微分关系

\[
a_t
=
\frac{v_t-v_{t-1}}{\Delta t}
\]

---

## 6.3 Edge anomaly

定义：

\[
\boxed{
S_{ij}^{edge}
=
D
\left(
z_j,
\phi_{ij}(z_i)
\right)
}
\]

或在物理空间：

\[
\boxed{
S_{ij}^{edge}
=
\left|
x_j-\phi_{ij}(x_i)
\right|
}
\]

其含义是：

\[
\boxed{
\text{当前节点之间的物理/结构关系是否被破坏}
}
\]

因此 edge anomaly 与 node anomaly 是两个不同概念。

---

# 7. Node Anomaly 与 Edge Anomaly 的区别

## Node anomaly

\[
S_i^{node}
=
D(z_i,p_i^*)
\]

回答：

> 当前 feature 与最近 normal operating prototype 中该 feature 的状态是否一致？

---

## Edge anomaly

\[
S_{ij}^{edge}
=
D(z_j,\phi_{ij}(z_i))
\]

回答：

> 当前 feature \(i\) 与 feature \(j\) 是否仍遵守既定物理/结构关系？

---

二者必须同时存在，因为：

### 新正常工况

可能：

\[
S_i^{node}\uparrow
\]

但：

\[
S_{ij}^{edge}\approx0
\]

即：

> 当前 feature 值以前没见过，但彼此之间仍物理自洽。

### 真故障

通常：

\[
S_i^{node}\uparrow
\]

同时：

\[
S_{ij}^{edge}\uparrow
\]

即：

> 状态新，而且物理关系也被破坏。

---

# 8. 第三层异常：Subgraph / Device Anomaly

第三层不再直接分析单个 feature，而是综合：

\[
\{S_i^{node}\}
\]

和：

\[
\{S_{ij}^{edge}\}
\]

得到一个局部子图或整机设备级异常分数。

---

## 8.1 Subgraph anomaly

对于子图：

\[
G_s=(V_s,E_s)
\]

定义：

\[
\boxed{
S_s^{subgraph}
=
AGG
\left(
\{S_i^{node}\}_{i\in V_s},
\{S_{ij}^{edge}\}_{(i,j)\in E_s}
\right)
}
\]

最简单可使用：

\[
S_s
=
\lambda_n
\sum_{i\in V_s}
w_iS_i^{node}
+
\lambda_e
\sum_{(i,j)\in E_s}
w_{ij}S_{ij}^{edge}
\]

也可以使用轻量 GDN/GAT 对 anomaly representations 进行融合。

---

## 8.2 Device anomaly

整个设备：

\[
G=(V,E)
\]

最终：

\[
\boxed{
S_{device}
=
AGG
\left(
\{S_i^{node}\},
\{S_{ij}^{edge}\}
\right)
}
\]

但训练与推理时必须保留：

\[
S_i^{node}
\]

与：

\[
S_{ij}^{edge}
\]

不能只输出单一 scalar，因为 node/edge score 是异常定位与解释的来源。

---

# 9. 三层异常的语义

最终模型提供三个层次的输出。

---

## Level 1 — Node anomaly

\[
\boxed{
S_i^{node}
}
\]

含义：

> 哪个 sensor / feature 偏离了当前正常工况？

用途：

- sensor-level localization；
- component fault indication；
- feature contribution analysis。

---

## Level 2 — Edge anomaly

\[
\boxed{
S_{ij}^{edge}
}
\]

含义：

> 哪条 feature relationship / physics relation 被破坏？

用途：

- physics inconsistency detection；
- fault propagation localization；
- causal / structural interpretation。

---

## Level 3 — Subgraph / Device anomaly

\[
\boxed{
S_G
}
\]

含义：

> 整个 subsystem 或 device 是否异常？

用途：

- alarm decision；
- fleet-level device ranking；
- maintenance prioritization。

---

# 10. Joint Prototype 与三层异常的整体流程

```text
Multivariate time-series
        ↓
        ▼
Shared temporal encoder
        ↓
        ▼
 z1  z2  z3 ... zN
        ↓
        ├───────────────────────────────┐
        ↓                               ↓
Joint Prototype Retrieval          Physics Graph
        ↓                               ↓
        ▼                               ▼
nearest prototype P*              typed edge operators
        ↓                               ↓
        ▼                               ▼
prototype components             physics predictions
p1* p2* ... pN*                       ↓
        ↓                               ↓
        ▼                               ▼
Node anomaly                      Edge anomaly
Si_node = D(zi,pi*)          Sij_edge = D(zj,phiij(zi))
        ↓                               ↓
        └──────────────┬────────────────┘
                       ▼
              Graph-level aggregation
                       ↓
                       ▼
          Subgraph / Device anomaly
```

---

# 11. 三层判定逻辑

定义：

\[
d_G
=
D(Z,P^*)
\]

表示整体 prototype novelty。

定义：

\[
S_{node}
\]

表示节点偏离。

定义：

\[
S_{edge}
\]

表示 physics inconsistency。

推荐将异常 reasoning 保持为多维，而不是立即坍缩成一个 scalar。

---

## 情况 A：已知正常

\[
d_G\ low
\]

\[
S_{node}\ low
\]

\[
S_{edge}\ low
\]

判断：

\[
\boxed{\text{Known Normal}}
\]

---

## 情况 B：局部 feature 偏离

\[
d_G\ low/moderate
\]

但某些：

\[
S_i^{node}\ high
\]

判断：

\[
\boxed{
\text{Localized Node Anomaly}
}
\]

进一步检查相关 physics edges。

---

## 情况 C：新正常工况候选

\[
d_G\ high
\]

\[
S_{node}\ high
\]

但：

\[
S_{edge}\ low
\]

判断：

\[
\boxed{
\text{Novel but Physically Consistent State}
}
\]

即：

> 可能是未见过的新正常 operating condition。

---

## 情况 D：高置信故障

\[
d_G\ high
\]

\[
S_{node}\ high
\]

\[
S_{edge}\ high
\]

判断：

\[
\boxed{
\text{Fault / Structural Anomaly}
}
\]

---

# 12. 与 GDN 的结合方式

GDN 不再只负责：

\[
x_t\rightarrow\hat x_{t+1}
\]

而更适合被重新定义为：

\[
\boxed{
\text{Graph consistency / anomaly aggregation module}
}
\]

输入为：

\[
z_i
\]

以及：

\[
S_i^{node},
S_{ij}^{edge}
\]

图结构由：

1. physics skeleton；
2. optional learned attention；

共同构成。

GDN 的任务包括：

- 聚合节点 latent state；
- 建模未完全显式定义的 feature coupling；
- 对 edge residual 进行上下文修正；
- 将 node + edge anomaly 融合成 subgraph/device anomaly；
- 保留可解释的 node 与边级 anomaly score。

---

# 13. Prototype 与 GDN 的职责分离

必须保持以下职责边界。

## Joint Prototype Memory

负责：

\[
\boxed{
\text{Operating-condition memory}
}
\]

回答：

> 当前多变量联合状态是否匹配某个过去见过的 normal mode？

---

## Node anomaly

负责：

\[
\boxed{
\text{Conditional feature deviation}
}
\]

回答：

> 在当前 normal mode 下，具体哪个 feature 不正常？

---

## Edge anomaly / Physics Graph

负责：

\[
\boxed{
\text{Relationship consistency}
}
\]

回答：

> feature 之间的物理/结构关系是否仍成立？

---

## GDN aggregation

负责：

\[
\boxed{
\text{Contextual graph reasoning}
}
\]

回答：

> 多个局部异常组合起来是否足以说明 subsystem / device 出现故障？

---

# 14. 为下一步 FL 保留的接口

本阶段暂不展开 FL 算法，但 joint prototype 必须设计成未来可被联邦共享。

未来每个设备拥有本地 prototype set：

\[
\mathcal P_n
=
\{
P_{n,1},\ldots,P_{n,M}
\}
\]

联邦层可以：

1. 上传 prototype，而不是原始数据；
2. 在服务器侧做 prototype semantic alignment；
3. 融合归类正常 operating condition；
4. 下发共享 prototype；
5. 保留本地 personalized prototype。

因此本阶段需确保：

\[
\boxed{
\text{prototype 是可查询、可比较、可融合的正常工况表示}
}
\]

而不是 encoder 参数中的隐式 normal distribution。

---

# 15. 当前实现建议

第一版实现建议至少包含以下内容：

1. 共享 temporal encoder：
   \[
   x_i\rightarrow z_i
   \]

2. Joint prototype memory：
   \[
   Z\rightarrow P^*
   \]

3. Global prototype distance：
   \[
   d_G=D(Z,P^*)
   \]

4. Node anomaly：
   \[
   S_i^{node}=D(z_i,p_i^*)
   \]

5. Physics graph：
   为已知 feature relations 定义 edge。

6. Edge residual：
   \[
   S_{ij}^{edge}
   =
   D(z_j,\phi_{ij}(z_i))
   \]

7. GDN / graph aggregation：
   \[
   \{S_i^{node},S_{ij}^{edge}\}
   \rightarrow
   S_{device}
   \]

8. 输出必须同时保留：
   - nearest prototype ID；
   - global memory distance；
   - per-node anomaly；
   - per-edge anomaly；
   - final device anomaly。

---

# 16. 必做消融

至少包含：

### A. Joint prototype only

只使用：

\[
d_G
\]

测试 memory 本身能力。

### B. Joint prototype + node anomaly

使用：

\[
d_G+S_i^{node}
\]

测试 prototype localization 能力。

### C. Physics edge only

只使用：

\[
S_{ij}^{edge}
\]

测试结构一致性能力。

### D. Joint prototype + node + edge

完整模型：

\[
\boxed{
d_G
+
S_i^{node}
+
S_{ij}^{edge}
}
\]

测试三层结构是否提高：

- fault detection；
- fault localization；
- new-normal vs fault discrimination。

---

# 17. 当前核心研究假设

整个设计建立在以下假设上：

\[
\boxed{
\text{Normal operating conditions form discrete or locally clustered joint states}
}
\]

因此可以由 joint prototypes 表示。

同时：

\[
\boxed{
\text{True faults tend to break either node-level conditional normality,
edge-level physical consistency,
or both}
}
\]

最终：

\[
\boxed{
\text{Memory determines whether the joint state has been seen;}
}
\]

\[
\boxed{
\text{Node anomaly determines which feature deviates from that state;}
}
\]

\[
\boxed{
\text{Edge anomaly determines which relationship is broken;}
}
\]

\[
\boxed{
\text{Graph aggregation determines whether the subsystem/device is faulty.}
}
\]

---

# 18. 一句话模型定义

> 使用共享时序编码器将多变量时序映射到节点 latent state，通过 Joint Prototype Memory 检索最接近的正常联合工况，并由 prototype 对齐 feature component 计算 node-level deviation，同时利用带物理算子的图边计算 feature 间 relationship residual，最后由 GDN 融合 node anomaly 与 edge anomaly，形成可定位、可解释、未来可联邦共享的三层异常检测框架。
