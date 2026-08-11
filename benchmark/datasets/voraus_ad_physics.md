# voraus-AD: physical prior relationships & formulas

Full source: `docs/voraus_ad_paper.pdf` (Brockmann, Rudolph, Rosenhahn,
Wandt, "The voraus-AD Dataset for Anomaly Detection in Robot
Applications", IEEE T-RO 2023, arXiv:2311.04765 -- Sections III-B, III-D
and the ablation in V-D are the primary sources for everything below).
Current implementation: `benchmark/datasets/voraus_ad_adapter.py`
(66-node graph -- covers relations #1-#3 below, see "What's declared vs.
not" at the end for what's still missing). See `memory/voraus-ad-dataset.md`
for the raw data-structure verification and `memory/joint-prototype-scheme-v3.md`
for the model results this doc's findings helped explain and improve.

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

**This relation IS now declared** (`joint_velocity_i -> motor_torque_i`,
"nonlinear") in the current 66-node graph, after the original 18-node
version (`motor_iq`/`motor_torque`/`torque_sensor_a` per joint only, no
target/velocity/position nodes at all) scored worst on exactly
`axis_friction` (0.540-0.566) among the larger-sample categories --
adding this edge (plus fixing a max-aggregation artifact that initially
masked its benefit) raised `axis_friction`'s AUROC to 0.789, see
`memory/joint-prototype-scheme-v3.md`.

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

This finding directly explained the ORIGINAL 18-node graph's weak
overall AUROC (0.66-0.68, landing almost exactly at the paper's own
"electrical alone" figure) -- that graph was built almost entirely from
`motor_iq` (electrical) plus one mechanical-but-narrow signal
(`torque_sensor_a`), excluding every position/velocity signal. The
current 66-node graph adds the full target/motor/joint tracking chain
(#1) precisely to close this gap -- see `memory/joint-prototype-scheme-v3.md`
for the resulting improvement.

## What's declared in `voraus_ad_adapter.py` (66-node graph) vs. what this doc covers

| relation | in this doc | declared as an edge in the adapter |
|---|---|---|
| target/motor/joint position+velocity tracking chain (#1) | yes | **yes** ("proportional", 4 edges/joint) |
| motor_iq -> motor_torque (#2) | yes | **yes** ("proportional") |
| motor_torque -> torque_sensor_a/b (#3) | yes | **yes** ("nonlinear", both sensors + a proportional a-vs-b cross-check) |
| joint_velocity -> motor_torque, friction (#4) | yes | **yes** ("nonlinear") |
| power_motor_el -> power_motor_mech -> power_load_mech (#5) | yes | **no** (no multiplicative relation type exists yet) |
| cross-joint kinematic coupling | not covered here -- needs DH parameters or an empirical check, see `memory/voraus-ad-dataset.md` | no |

Relations #1-#4 are all implemented (66 nodes, 54 within-joint edges).
#5 (the power-conservation chain) is the only documented relation left
out, since it's a multiplicative rather than proportional/nonlinear
relationship -- would require extending `TypedRelationAnomalyHead` with a
new relation type before it's usable. Cross-joint edges also remain
undeclared, needing the arm's geometry.
