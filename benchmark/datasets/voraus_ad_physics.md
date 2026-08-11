# voraus-AD: physical prior relationships & formulas

Full source: `docs/voraus_ad_paper.pdf` (Brockmann, Rudolph, Rosenhahn,
Wandt, "The voraus-AD Dataset for Anomaly Detection in Robot
Applications", IEEE T-RO 2023, arXiv:2311.04765 -- Sections III-B, III-D
and the ablation in V-D are the primary sources for everything below).
Current implementation: `benchmark/datasets/voraus_ad_adapter.py`
(18-node graph, only a subset of the relations documented here --
see "What's declared vs. not" at the end). See `memory/voraus-ad-dataset.md`
for the raw data-structure verification and `memory/voraus-ad-joint-prototype-v3.md`
for the first model run this doc's findings help explain.

## Platform

Yu-Cobot, a 6-axis collaborative robot arm performing a pick-and-place
task (grip a randomly-placed can off a conveyor, move it to a fixed
target, place it). Each of the 6 axes has its own closed-loop position/
velocity/torque controller, an absolute encoder (RLS AksIM-2, position),
and an independent torque sensor (Sensordrive GMS).

## 1. The target -> motor -> joint tracking chain (per axis, per quantity)

For position and velocity, the paper documents a straightforward
closed-loop control relationship: the controller commands a
`target_position`/`target_velocity`, the motor is driven to track it
(`motor_position`/`motor_velocity`), and the actual joint-side motion
follows through the gearbox (`joint_position`/`joint_velocity`). Under
normal operation these three should track closely (near-identity up to
sensor placement / gearbox compliance); under a fault (collision,
friction, miscommutation) the controller measurably deviates from target
to correct the disturbance -- the paper's own example (Fig. 6a, collision
with foam): *"After the robotic arm has deviated from its joint angle
target due to the collision the closed-loop control counteracts to
correct the path which influences the joint angle phi_5 and its
velocity phi_5_dot."*

```
target_position_i  ~=  motor_position_i  ~=  joint_position_i     (near-identity, "proportional")
target_velocity_i  ~=  motor_velocity_i  ~=  joint_velocity_i     (near-identity, "proportional")
```

This is the most direct within-axis analogue of `kinematics.py`'s
wheel-vs-odometry idea in this project's other datasets, but with THREE
stages instead of two, and it's native to the dataset rather than
something to derive.

## 2. Motor current -> motor torque (the paper's own worked example of exactly this project's "proportional" relation type)

The paper explicitly separates the motor's current into two physically
distinct components: `motor_iq` (torque-forming current) and `motor_id`
(magnetizing current) -- *"Measured signals related to each axis include
... torque-forming current I_q and magnetizing current I_d."* Only `I_q`
drives torque; `I_d` sets the field and should NOT correlate with torque
under normal operation. This is a stronger, more literal match to
`docs/joint_prototype_physics_gdn_anomaly_attention_prompt.md` Sec 9's
"proportional relation" textbook example (current ~ torque via the motor
torque constant Kt) than any other dataset in this project has offered --
here it's stated directly by the paper's own instrumentation choice, not
inferred.

```
motor_torque_i  ~=  Kt_i * motor_iq_i          ("proportional")
```

**The miscommutation fault is defined as exactly a break in this
relation, explicitly**: *"The motor control would adjust the motor
current and thus the motor torque to maintain the specified path and,
through this, the velocity. The wear of electrical and magnetic
components would affect the efficiency and motor current of the robot
axis instead of the velocity, without leading to increased motor
torque."* In other words: under `MOTOR_COMMUTATION`, current rises but
does NOT produce the expected proportional torque increase -- current and
torque decouple while velocity/position stay nominal. This is a clean,
paper-confirmed prediction for exactly the kind of "declared edge whose
residual should spike under one specific fault type" signal
`TypedRelationAnomalyHead` is built to catch.

## 3. Motor torque -> independent torque sensor (redundant cross-check, per axis)

Each axis has TWO separate ways to know its torque: the motor-side
estimate (`motor_torque`, derived from current/commutation) and an
independent physical sensor (`torque_sensor_a`, a second `torque_sensor_b`
also exists). This is the same "two independent estimators of the same
physical quantity" pattern as robo3er's odometry-vs-IMU yaw rate idea
(`robo3er_physics.md`), but built into the hardware here (Sensordrive
GMS torque sensor) rather than something to newly wire up. Under normal
operation these should agree closely; under a fault that affects the
mechanical path AFTER the motor (friction, gearbox wear, external
contact) but not the motor's own current/commutation, this is the
relation that should break instead of #2's current-torque relation --
i.e. #2 and #3 are two structurally different "torque anomaly" checks
that should light up for DIFFERENT fault categories, a concrete testable
prediction:

```
motor_torque_i  ~=  torque_sensor_a_i  ~=  torque_sensor_b_i   ("proportional"/"nonlinear",
                                                                   see friction note below)
```

`benchmark/datasets/voraus_ad_adapter.py` currently declares this edge as
"nonlinear" rather than "proportional" -- deliberately, because of the
friction relationship below.

## 4. Friction: velocity/target-torque -> motor torque, NOT current -> torque

`AXIS_FRICTION` is explicitly defined by the paper as a DIFFERENT
mechanism from miscommutation: *"Due to the friction, a higher torque of
the motor is needed for the same movement."* I.e. friction breaks the
relation between the COMMANDED motion (target velocity/torque, or the
kinematic target-vs-measured chain in #1) and the torque actually
required -- not the current-to-torque relation in #2, which friction
leaves largely intact (the motor still converts current to torque
normally; it just needs MORE current/torque to achieve the same
velocity). This means:

```
target_torque_i  (or target_velocity_i)  ->  motor_torque_i      ("nonlinear" -- friction
                                                                     is a velocity/load-dependent
                                                                     extra torque term, not a
                                                                     fixed offset or scale factor)
```

**This relation is NOT currently declared anywhere in
`voraus_ad_adapter.py`'s 18-node graph** -- the graph only includes
`motor_iq`, `motor_torque`, `torque_sensor_a` per joint, with no
target/velocity/position nodes at all. This is very likely why
`memory/voraus-ad-joint-prototype-v3.md`'s first run scored worst on
exactly `axis_friction` (0.540-0.566) among the larger-sample categories --
the one relation the paper says friction actually breaks isn't in the
graph.

## 5. Energy conservation chain: electrical power -> mechanical power -> load power

Three power quantities per axis form a physical dissipation chain:
`power_motor_el` (electrical power into the motor) -> `power_motor_mech`
(mechanical power the motor outputs, after motor efficiency losses) ->
`power_load_mech` (mechanical power delivered to the load/joint, after
gearbox/transmission losses). Each stage's efficiency should be
consistent under normal operation (electrical-to-mechanical conversion
efficiency, then transmission efficiency); a fault that changes friction
or mechanical loading (axis wear, collision) would show up as an
efficiency shift at the relevant stage. Additionally, `power_motor_mech`
is itself derivable from `motor_torque_i * motor_velocity_i` (mechanical
power = torque x angular velocity) -- a genuinely MULTIPLICATIVE
relation, not linear or a simple MLP-approximable nonlinearity in the
same sense as #3/#4; not one of the two relation types currently
implemented in `TypedRelationAnomalyHead` (only "proportional"/
"nonlinear" exist there today).

```
power_motor_el_i   ~eta_motor~>   power_motor_mech_i   ~eta_gearbox~>   power_load_mech_i
power_motor_mech_i  =  motor_torque_i * motor_velocity_i     (exact, multiplicative)
```

Not currently declared in `voraus_ad_adapter.py` -- would need either a
new "multiplicative" relation type or a feature-engineering step (compute
the power ratio explicitly as a derived feature) before it could be used.

## 6. Empirical finding from the paper: mechanical signals matter far more than electrical for detection

The paper's own ablation (Fig. 13a, MVT-Flow on the full 130-signal
dataset) reports mean AUROC by signal subset: **electrical alone ~65%,
mechanical alone ~92%, measured-only ~85%, computed-only ~86%, all
signals ~93%** -- *"Mechanical signals provide clearly more importance
for AD compared to electrical signals improving the performance by 25%."*

This is a striking, directly relevant number: `voraus_ad_adapter.py`'s
current 18-node graph is built almost entirely from what the paper calls
electrical (`motor_iq`) plus one mechanical-but-narrow signal
(`torque_sensor_a`) -- it excludes every position/velocity signal, i.e.
the exact category the paper's own ablation says matters most. The
first V3 run's weak overall AUROC (0.66-0.68, see
`memory/voraus-ad-joint-prototype-v3.md`) landing almost exactly at the
paper's own "electrical alone" figure (~65%) is a strong, now
paper-corroborated explanation, not a coincidence -- adding
`joint_velocity_i`/`joint_position_i` (or the target-vs-measured tracking
chain from #1) as nodes is the most evidence-backed next step for
improving this dataset's graph, ahead of tuning hyperparameters or
epochs.

## What's declared in `voraus_ad_adapter.py` vs. what this doc covers

| relation | in this doc | declared as an edge in the adapter |
|---|---|---|
| target/motor/joint position+velocity tracking chain (#1) | yes | **no** -- no position/velocity nodes at all |
| motor_iq -> motor_torque (#2) | yes | **yes** ("proportional") |
| motor_torque -> torque_sensor_a (#3) | yes | **yes** ("nonlinear") |
| target_torque/velocity -> motor_torque, friction (#4) | yes | **no** |
| power_motor_el -> power_motor_mech -> power_load_mech (#5) | yes | **no** (no multiplicative relation type exists yet) |
| cross-joint kinematic coupling | not covered here -- needs DH parameters or an empirical check, see `memory/voraus-ad-dataset.md` | no |

Only #2 and #3 are implemented today. #1 and #4 are the most
evidence-backed additions (directly explains the friction/miscommutation
category split and the paper's own mechanical-vs-electrical finding);
#5 would require extending `TypedRelationAnomalyHead` with a new
relation type before it's usable.
