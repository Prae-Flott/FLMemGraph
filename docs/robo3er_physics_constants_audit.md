# robo3er 物理关系与常数溯源审计

本表逐条列出 `kinematics.py` 及 v3 `EDGES_NAMED`(`benchmark/run_robo3er_v3_1.py`)已声明、
或本次讨论中考虑新增的物理关系,为每个常数标注来源类别,并标注该关系是否能写成
"参数少、系数可解释"的显式闭式公式(能否进入后续"显式公式残差评分头"设计)。

来源类别定义:
- **[A] 可查官方规格**:厂商文档/CAD/开源驱动参数文件里有精确数值,可直接引用。
- **[B] 需用户实测**:物理上是确定关系,但官方未公开精确值,需要在实机上测量。
- **[C] 必须训练拟合**:没有物理捷径可查或可测(电气/机械耦合、非线性、多因素),
  只能用回归/训练从数据里估计,且系数没有唯一"真值"可对照验证。

---

## 1. 差速驱动几何关系(轮速 -> 底盘线/角速度)

```
v_lin = (r/2) * (v_l + v_r) = k_v * (v_l + v_r)
v_ang = (r/L) * (v_r - v_l) = k_w * (v_r - v_l)
```

| 常数 | 物理含义 | 类别 | 备注 |
|---|---|---|---|
| `r`(有效轮半径) | 轮子接地滚动半径 | **[B]**(官方未公开精确值,见步骤2检索结果) | 磨损、胎压/材质形变会使"有效"半径偏离标称值 |
| `L`(有效轮距) | 两轮接地点间距 | **[B]**(同上) | 悬挂/安装公差同样使"有效"值偏离标称值 |
| `k_v = r/2`, `k_w = r/L` | 当前 `kinematics.py` 实际拟合的量 | **[C]**(现状) | 因为原始 topic 的单位不确定(m/s?编码器原始速率?),无法直接代入查到的 r、L 算出 k_v/k_w,必须用 OLS 拟合标量系数——即使查到/测到 r、L 的物理值,仍需一个"单位换算常数"配合,详见下方"是否可显式化"结论 |

**是否可显式化**:关系形式(`sum`/`diff`)本身是**确定的闭式几何公式**,系数虽仍要靠拟合确定数值,
但拟合出的是**可解释的物理标量**(2 个自由度:`k_v`,`k_w`,外加截距吸收标定偏置),不是黑箱网络——
**可以**直接作为显式公式残差评分头的一条边,且已验证按 regime(同向/异向)分开拟合能显著改善拟合优度。

---

## 2. 跨传感器一致性关系(底盘角速度:odom vs IMU)

```
imu_angvel_z ≈ odom_angtw_z   (理论上应为 1:1,同一物理量的两次独立测量)
```

用真实 fit-split 数据实测线性拟合:

```
imu_angvel_z = 0.9095 * odom_angtw_z + 0.00252     R² = 0.8325
```

| 常数 | 物理含义 | 类别 |
|---|---|---|
| 斜率 0.9095 | 理论应为 1(同一物理量),实测偏离约 9% | **[C]**(反映的是标定/坐标系/滤波延迟等系统性偏差,不是可查的物理常数,只能拟合刻画,但拟合出的"该等于1"这件事本身是可验证的物理期望) |
| 截距 0.00252 | 理论应为 0 | **[C]**,同上,数值很小,基本符合"无常量零偏"的预期 |

**是否可显式化**:**可以**,且是本项目已识别的、比"轮速→里程计"更严格的滑移检测通道
(两侧都是独立传感器,不像"单轮→里程计"两者共享同一批编码器数据源)。R²=0.83 且不受
regime(同向/异向)影响(之前已验证两态下都稳定),是显式公式残差评分头里最干净的一条边。

---

## 3. 电机电流 <-> 轮速/PWM(执行器-运动学耦合)

用真实 fit-split 数据实测的简单线性拟合(仅作诊断,非声称电流是线性关系):

```
current_left  = 0.245 * wheel_v_left  - 0.100    R² = 0.079
current_right = 0.289 * wheel_v_right - 0.039    R² = 0.099
current_left  = 0.271 * pwm_left      - 0.095    R² = 0.092
current_right = 0.405 * pwm_right     - 0.015    R² = 0.189
```

单变量线性拟合的 R² 全部低于 0.2——说明电流不能用"轮速"或"PWM"单独一个变量的线性关系
良好预测,这与 `EDGE_TYPES` 里把这两条边标记为 `"nonlinear"`(而非 `"proportional"`)一致。

| 常数/关系 | 物理含义 | 类别 |
|---|---|---|
| 反电动势常数 `Ke` | 电机转速->反电动势电压 | **[C]**,iRobot 未公开电机型号规格书,无法查表 |
| 转矩常数 `Kt` | 电流->输出扭矩 | **[C]**,同上 |
| 电枢电阻 `R_motor` | 电机绕组直流电阻 | **[B]**,理论上可用万用表直接测量电机两端直流电阻(需拆机接触电机端子,若不可拆机则仍为 [C]) |
| 齿轮箱减速比/效率 | 电机转速->轮子转速 | **[B]或不可测**,若无法拆解齿轮箱,只能视为拟合内隐含的一部分 |
| PWM 占空比 -> 施加电压 | `V_applied ≈ duty% * battery_state_voltage` | **[A]**,纯电路欧姆定律关系,`battery_state_voltage` 已是数据集的一列,系数是确定性的(占空比定义本身),不需要拟合——但这只是"电压"这一中间量,不直接等于电流或轮速,不能单独构成一条完整的物理预测边 |

**是否可显式化**:**不能**用简单闭式公式覆盖电流的主要变异——数据本身显示单变量线性关系解释力不足 20%,
需要电机方程 `V = I*R + Ke*ω`(至少 2 个变量:PWM 施加电压 + 轮速)且 `R`、`Ke` 仍缺乏可查值。
建议:保留为 **[C] 训练拟合**,但限定用**多元线性回归**(`current ~ pwm + wheel_v`,2-3 个可解释系数),
不用深度网络;若拟合优度仍不足,应如实记录"此关系当前无法可靠公式化",按用户"宁可少建边"的
标准,可以选择降级为不参与显式评分头、只保留在 `d_node`/`resid_struct` 等通用通道里。

---

## 4. 轮编码器计数 <-> 轮速(尚未声明为边,potential 新增)

`feature_groups.py` 把 `wheel_ticks_ticks_left/right` 归入运动学核心组,理论上:

```
wheel_velocity ≈ (Δticks / ticks_per_revolution) * 2π * r / Δt
```

| 常数 | 物理含义 | 类别 |
|---|---|---|
| 编码器每转脉冲数(ticks/rev) | 编码器分辨率 | **[B]**(官方检索未找到精确值,需实测或查阅底层驱动固件源码/datasheet) |
| 采样间隔 `Δt` | 数据记录频率 | **[B]**(项目文档未记录采样率,需从原始 rosbag 时间戳或采集脚本配置确认) |

**是否可显式化**:形式上是确定性的位置->速度求导关系,**一旦 `ticks_per_rev` 和 `Δt` 确认,是最容易验证
的一条边**(几乎是纯粹的数值微分,不涉及任何未知物理耦合)。但目前两个关键常数均缺失,暂不建议
现在实现,列为可选的"tier-2"新增边,取决于步骤2/3 检索和用户测量结果。

---

## 结论汇总表

| 关系 | 类别 | 能否显式化(闭式,可解释系数) | 建议 |
|---|---|---|---|
| 轮速(sum)-> 底盘线速度 | B(r,L) / C(k_v 拟合) | ✅ 可以,已验证 regime 相关 | 保留,分 regime 重新拟合 |
| 轮速(diff)-> 底盘角速度 | B(r,L) / C(k_w 拟合) | ✅ 可以,已验证 regime 相关 | 保留,分 regime 重新拟合 |
| 单轮 -> 底盘线/角速度 | 同上(局部近似) | ⚠️ 仅 TRANSLATING regime 下可用 | 按 regime 掩码,或直接放弃改用 sum/diff 超边 |
| 底盘角速度 odom vs IMU | C(标定偏差) | ✅ 可以,最干净的一条边 | 保留,regime 无关 |
| 电流 <-> 轮速/PWM | C(电机方程) | ❌ 单变量不可靠,多变量线性也待验证 | 暂不纳入显式评分头,或用多元线性回归尝试后再评估 |
| 编码器计数 -> 轮速 | B(ticks/rev, Δt 均缺失) | ✅(若常数确认) 潜力最大但当前信息不足 | 列为 tier-2,待步骤2/3结果 |

---

## 附:iRobot Create 3 官方几何规格检索结果(步骤2)

检索了 iRobot 官方文档站(`iroboteducation.github.io/create3_docs`)、iRobot 客服尺寸页、教育版产品页,均只给出**整机外部尺寸**(机身直径约 342mm、高约 90mm 等)和**无关尺寸**(12mm 网格安装孔间距、最大有效载重 9kg),**没有**直接给出轮径或轮距的精确毫米数。

官方机械系统文档([iroboteducation.github.io/create3_docs/hw/mechanical/](https://iroboteducation.github.io/create3_docs/hw/mechanical/))原文:"The Create® 3 is a differential drive robot... The drive wheels feature independent suspensions... a regular 12 mm grid of 3 mm diameter mounting holes. The maximum recommended payload weight (without changing acceleration limits) is 9 kg..."——即确认了差速驱动、独立悬挂、12mm/3mm 安装孔网格、9kg 建议载荷上限,但这份官方文档本身**不含**轮径/轮距数值。

**更有价值的发现**:iRobot 官方 GitHub 组织 `iRobotEducation` 维护的 `create3_sim` 仿真仓库里有一份**官方 URDF 机械参数文件**(`irobot_create_common/irobot_create_description/urdf/create3.urdf.xacro`),这是 iRobot 自己发布的仿真模型定义,包含真实机械参数,比任何第三方转述都权威。目前抓取到的片段确认了:

```
body_mass   = 2.300 kg
body_radius = 16.4 cm   (= 0.164 m, 机身圆柱半径,非轮子)
body_length = 6 cm
```

文件中还定义了 `distance_between_wheels`(轮距)这个变量名——**这正是 `kinematics.py` 里 `L` 的官方对应量**——但受限于本次搜索片段截断,具体数值未能读取到;同样,轮子专属的 `wheel_radius` 参数应该在同一文件的 wheel 相关 xacro 片段中,也未在本次片段中完整读到。

**结论与下一步**:
- 官方最终来源已锁定:`https://github.com/iRobotEducation/create3_sim/blob/main/irobot_create_common/irobot_create_description/urdf/create3.urdf.xacro`(以及同目录下被 include 的 `wheel_with_wheeldrop.urdf.xacro`)。
- 这两个文件的完整内容需要下一轮直接读取 GitHub raw 文件(而非仅搜索片段)才能拿到 `wheel_radius`、`distance_between_wheels` 的精确数值——若本会话后续被允许访问 `raw.githubusercontent.com`/`github.com`,应直接拉取这两个文件确认。
- 在此之前,**标记为"来源已定位、数值待补全"**,不得用本文档以外的记忆数字替代;实测方案(步骤3)作为并行的独立验证路径保留,即使查到官方 URDF 数值,仍建议用户用卡尺/卷尺实测做交叉验证——因为 URDF 里的数值是仿真用的标称设计值,不一定等于实际生产轮胎在特定地面上的"有效"半径(`kinematics.py` 文档里反复强调的磨损/材质形变问题)。

### 更新:数值已拉取(直接读取官方 GitHub raw 文件确认)

直接拉取 `create3.urdf.xacro`、`wheel_with_wheeldrop.urdf.xacro`、`wheel.urdf.xacro`(均来自 `iRobotEducation/create3_sim` 官方仓库 main 分支)确认:

| 参数 | 官方 URDF 数值 | 来源 |
|---|---|---|
| 轮子半径 `radius` | **35.75 mm** (`3.575*cm2m`) | `wheel.urdf.xacro` |
| 轮子宽度 `width` | 15.0 mm | `wheel.urdf.xacro` |
| 轮距 `distance_between_wheels` | **233.0 mm** (`23.3*cm2m`) | `create3.urdf.xacro` |
| 机身半径 `body_radius` | 164 mm | `create3.urdf.xacro` |

这两个数值(轮径35.75mm、轮距233mm)是**iRobot 官方仿真模型的标称设计值**,类别升级为 **[A] 已查到官方规格**(取代此前审计表里的 [B])。

### 关键发现:用官方几何值预测的 k_v/k_w 与真实拟合值不匹配——原因不是单位换算,而是 `/odom` 本身是多传感器融合输出

用官方值代入理论公式 `k_v_theory = r/2`、`k_w_theory = r/L`(`r`=0.03575m,`L`=0.233m):

```
k_v_theory = 0.01788      vs 真实拟合 k_v = 0.4729    比值 = 26.5倍
k_w_theory = 0.15343      vs 真实拟合 k_w = 1.1623    比值 = 7.6倍
```

两个比值**不相等**(26.5 vs 7.6),说明这不是简单的"整体乘一个单位换算系数"就能解释的偏差——如果只是单位不统一(比如 v_l/v_r 实际是原始编码器计数而非国际单位),两个比值应该相同或成简单倍数关系。真实原因在官方文档里找到了直接证据:

直接抓取并核对了官方文档全文([iroboteducation.github.io/create3_docs/api/odometry/](https://iroboteducation.github.io/create3_docs/api/odometry/)),确认:该机配备 IMU、光学鼠标(optical mouse)、轮编码器三类传感器,原文:"The Create® 3 robot fuses the reading from its various sensors in order to produce a dead reckoning estimate of its pose on the odom topic." 该 topic 发布频率为 20 Hz。

**这意味着**:`odom_odo_lintw_x`/`odom_odo_angtw_z` 并不是"用轮速通过差速驱动几何公式积分得到的纯轮系运动学量",而是 Create 3 主板内部一个融合了轮编码器 + IMU + 光学鼠标(optical flow)传感器的**状态估计输出**(很可能是卡尔曼滤波器一类的融合算法)。`kinematics.py` 拟合出的 `k_v`、`k_w` 因此**不是**、也不应该被期望等于 `r/2`、`r/L` 这两个纯几何量——它们额外吸收了这个融合算法的増益/权重、坐标系转换、以及可能的编码器原始单位到 rad/s 的转换系数。

同时官方消息定义([irobot_create_msgs/msg/WheelVels.msg](https://github.com/iRobotEducation/irobot_create_msgs/blob/rolling/msg/WheelVels.msg))确认了 `wheel_vels` 话题的单位,原文注释:"Velocity measure for left wheel in rad/sec."——即数据集里的 `wheel_vels_velocity_left/right` 已经是标准单位(弧度/秒),不存在"原始编码器计数"这层歧义;单位不是导致上述比值不匹配的原因,融合算法才是。

**对本项目的直接影响**:
1. `kinematics.py` 现有设计(不硬编码 r、L,而是从数据里拟合 k_v/k_w)在这个新证据下**更加正确**,而不是权宜之计——因为真实的 wheel→odom 映射本来就不是纯几何公式,拟合是唯一可行路径,硬编码官方 r/L 数值反而会引入系统性错误。
2. 这也重新解释了"跨传感器边"(`odom_ang -> imu_angvel_z`,R²=0.83)为什么表现最干净:因为 `/odom` 本身就是融合了 IMU 的输出,两者天然高度相关,该边测的其实是"融合后估计 vs 融合前的原始 IMU 读数"之间的**残余误差/滤波延迟**,而不是两个完全独立传感器路径的交叉验证(比原先的理解更弱一层,但依然是本项目当前最干净的可解释关系)。
3. `kin_residual`("融合估计 - 拟合的轮系线性预测")里,除了轮子打滑,理论上也会包含"IMU/光学鼠标对融合结果的修正量"——这是此前讨论里没有覆盖到的一个新的残差来源,值得在后续故障分析里留意区分。

### 附带确认:轮编码器分辨率与采样率(解决了审计条目4的两个缺失常数)

官方文档([iroboteducation.github.io/create3_docs/api/odometry/](https://iroboteducation.github.io/create3_docs/api/odometry/))原文:"The wheel_ticks topic produces messages of type irobot_create_msgs/msg/WheelTicks. This topic publishes at 62.5 Hz. There are 508.8 ticks per wheel rotation." `odom` 话题本身如前所述发布频率为 20 Hz(融合后的输出频率,低于 62.5Hz 的原始轮编码器采样率)。

这两个数值(508.8 ticks/rev,62.5Hz 原始采样)可以把审计条目4("轮编码器计数->轮速")的类别从 [B] 需实测 **上调为 [A] 官方已知**——但采集脚本落到 68 列表格时实际用的帧间隔 `Δt` 仍需与项目自己的数据采集/重采样代码核对(原始话题发布率不等于最终落盘的帧率),这一步留给步骤5重写 `kinematics.py` 时一并验证。
