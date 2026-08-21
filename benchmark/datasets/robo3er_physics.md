# robo3er: physical prior relationships & formulas

Full implementation: `src/robo3er/kinematics.py`. This is the reference
implementation every other dataset's physics-prior work in this project
is modeled after ("expert-knowledge-informed structure, data-fit
parameters").

## Platform

robo3er is a 5-robot fleet of iRobot Create3-class **differential-drive**
mobile robots. Each robot has two independently-driven wheels; steering
is achieved by commanding different left/right wheel velocities, not a
separate steering mechanism.

## The physical relation

Textbook forward kinematics for a differential-drive robot relates the
two wheel velocities `(v_l, v_r)` to the chassis's linear and angular
velocity:

```
v_lin = k_v * (v_l + v_r)     where k_v = r/2
v_ang = k_w * (v_r - v_l)     where k_w = r/L
```

`r` = effective wheel radius, `L` = effective track width (distance
between the two wheels' contact points) -- both physical constants of
the specific robot.

## Why the constants are FIT, not hard-coded

`r` and `L` are physically real, but their exact values in the units the
logged topics use are unknown (could be wheel surface speed in m/s, raw
encoder rate, affected by calibration drift or gearbox slack). Rather
than assume a spec-sheet number, `k_v` and `k_w` are fit by ordinary
least squares from each robot's own held-out NORMAL (fit-split) data --
keeping the physically-motivated two-parameter structure of the equation,
not a free 4-coefficient linear regression. An intercept term (`b_v`,
`b_w`) is fit alongside each slope to absorb constant sensor
offset/calibration bias -- physically that term shouldn't exist for an
ideal robot at rest, so a fitted intercept far from 0 is itself a
diagnostic flag worth reporting, not silently discarded.

In the federated system (`src/robo3er/fl_dataset.py`), this fit is done
**per client** (per robot) -- each robot may have slightly different
wheel radius/track width (e.g. from wear), so a per-robot fitted constant
captures that rather than forcing one global fit across robots with
different physical calibration.

## The residual (what actually gets fed to the model)

```
residual_lin = measured(odom_odo_lintw_x) - predicted(v_l, v_r; k_v, b_v)
residual_ang = measured(odom_odo_angtw_z) - predicted(v_l, v_r; k_w, b_w)
```

These two residuals REPLACE the raw `odom_odo_lintw_x` /
`odom_odo_angtw_z` columns before GDN (or any downstream model) sees the
data -- every other feature in the active feature group is untouched.
Whatever variance the kinematic equation predicts is treated as
"explained by rigid-body kinematics" and stripped out; what's left is
fed forward.

## Why this residual is diagnostic

A wheel that is commanded/measured to spin but whose motion does NOT show
up in the chassis's actual odometry twist -- i.e. a large kinematic
residual -- is the textbook definition of wheel slip. This residual
carries direct signal for robo3er's `cable trapped` (slip) fault type,
on top of whatever raw-feature signal GDN could already find on its own.

## Measured impact (see `memory/physics-informed-gdn.md` for the full
## writeup)

This residual, combined with GDN's own forecasting mechanism
(`src/training/train_gdn_physics.py`), produced this project's best-ever `stuck`
fault result (AUROC 0.53 -> 0.72-0.74) -- the single largest improvement
found anywhere in this project's history, via a mechanism orthogonal to
every other approach tried on that fault type (memory, feature selection,
architecture changes).

## What's NOT modeled

- Only the linear/angular VELOCITY relation is used. The chain continues
  further (velocity -> displacement via integration, and there's a
  second independent estimator of chassis yaw rate via the IMU,
  `imu_imu_angvel_z`, that `src/robo3er/feature_groups.py`'s docstring flags as
  "arguably a MORE rigorous slip detector than wheel-vs-odom, since both
  sides of THAT comparison are wheel-derived" -- not yet implemented).
- Wheel radius/track width are assumed constant per robot per run (no
  time-varying wear model).

---

## Fault ground truth: physical mechanisms

The four fault types in robo3er each have a distinct physical cause and
a distinct sensor signature. This section documents the verified physical
scenario for each, to ground anomaly detection claims in mechanistic
causation rather than correlation alone.

### `stuck` — cable tangled in wheel axle

**Mechanism**: A loose cable gradually wraps around one of the robot's
wheel axles during operation. Wrapping is progressive: initial turns
increase friction on that axle → the motor draws more current to maintain
the commanded velocity → as more cable wraps, the load increases further
→ eventually the axle locks completely and the motor stalls.

**Expected sensor signatures (temporal order):**

1. `wheel_status_current_ma_left` OR `_right` (one side only) shows a
   **gradual monotonic increase** while the opposite side stays at the
   normal load level. The asymmetry between left and right drive current
   is the earliest detectable signal of this fault.
2. `wheel_vels_velocity_*`: the affected wheel's measured velocity begins
   to lag behind the commanded velocity as friction increases, and drops
   to zero at full lock.
3. `odom_odo_lintw_x` / `odom_odo_angtw_z`: robot veers and eventually
   cannot translate, as one wheel provides thrust while the other is
   locked.
4. At full stall: ALL motion-related features simultaneously plateau near
   zero (both `wheel_vels_velocity_*`, both `odom_odo_*`), while the
   locked wheel's current may be at a HIGH plateau (stall current).

**Why the covariance signal (H) uniquely detects this:**
Under normal operation all motion nodes co-deviate coherently: both wheel
velocities rise and fall together (straight/turning drive), current tracks
velocity on both sides. `stuck` creates a strongly **asymmetric** joint
deviation pattern -- one current channel high while the other and all
velocity channels are zero -- that the per-node `d_node` signal misses
(individual magnitudes may not be extreme) but the covariance signal
catches as a completely atypical joint structure for any known operating
regime. This explains why H=0.747 >> B=0.510 on this specific fault
(see benchmark results in `memory/joint-prototype-federated-results.md`).

**Signal limitation**: `battery_state_*` (removed from feature set for
stationarity reasons) would also show a side effect (locked motor draws
peak battery current), but the direct diagnostic path -- left/right
current ASYMMETRY -- is fully observable from the kept `wheel_status_*`
channels. A dedicated left-vs-right current ratio feature could make this
fault much easier to detect; not yet implemented.

---

### `Broken Pipe` — ROS2 clock-sync failure with robot base

**Mechanism**: A clock synchronization issue between the robot's onboard
compute and its base station causes the ROS2 DDS layer to reject topic
publications (timestamps too far from wall clock). All topics that the
base publishes -- including odometry, wheel status, and IMU -- **stop
arriving completely** from the robot's perspective. The gap lasts until
the clock re-synchronizes or the relevant ROS2 node restarts, after which
topic flow resumes normally.

**Expected sensor signatures:**

1. **Abrupt, simultaneous freeze** across ALL base-published topics
   (`odom_*`, `wheel_vels_*`, `wheel_status_*`, `imu_*`): values hold
   at the last received reading for the full gap duration. From the
   recording side this appears as sudden near-zero variance on those
   channels (the value doesn't change, so derivatives/deviations are zero
   -- a different kind of abnormal than "value is wrong").
2. Topics that originate from OTHER nodes (battery, navigation layer, if
   any) may remain unaffected.
3. **Recovery**: all topics resume abruptly at the reconnection point;
   there may be a transient at reconnect as accumulated commands flush.

**Why this is well-detected globally (B=0.997):** The joint prototype
memory detects it as an operating state that has no close match in any
stored prototype -- normal driving sees non-trivial variance across all
channels; the frozen-topic state is a degenerate corner of the state
space the prototype codebook has never seen. `d_proto` alone is 0.511
(below chance), but `d_node` catches the per-channel freezing (channels
that should have non-trivial variance now show near-zero deviation across
the window, which is abnormal relative to any prototype's expected
per-channel spread).

---

### `cable_trapped` — cable caught under chassis

**Mechanism**: One or more cables get pinched under the robot chassis
(typically between the bumper and the floor surface). The cable applies
a lateral or torsional external force on the chassis, causing the robot
to **continuously spin in place or oscillate left/right** (the drive
system fights the constraint, producing high-frequency steering
corrections). Unlike `stuck`, both wheels remain free to spin; the
abnormality is in the commanded trajectories and their kinematic
consistency, not in a mechanical blockage.

**Expected sensor signatures:**

1. `odom_odo_angtw_z` (yaw rate): large, erratic, high-variance --
   robot is spinning or rocking constantly.
2. `wheel_vels_velocity_left` vs `_right`: large and frequently
   sign-reversed (left/right alternation at high frequency as the
   robot corrects).
3. Kinematic residual (`residual_ang` from `kinematics.py`): may be
   elevated if the cable also introduces a slip component (chassis
   position does not match what the differential-drive model predicts
   from wheel velocities).
4. `imu_imu_angvel_z`: corroborates `odom_odo_angtw_z` for angular
   motion; discrepancy between the two (one rises, other doesn't) is
   a slip indicator.

**Why easily detected by all signals (B=0.996, E=1.000):** The
oscillatory pattern is completely unlike any normal operating regime
stored in the prototype codebook, AND the current-to-velocity and
wheel-to-odom typed relations are systematically violated under the
forced spinning. This is the fault most aligned with the physics prior
-- all declared edges (wheel→odom, odom_ang→imu_angvel) see a genuine
physical break in their normal coupling.

---

### `Low battery` — battery level below 30%

**Mechanism**: Battery state-of-charge drops below 30%, crossing the
threshold where motor drive performance begins to degrade visibly.
Below this threshold: (1) available peak current from the battery
decreases, (2) the drive controller may apply software-level speed
limiting or increased current limiting, (3) motor efficiency drops
because the bus voltage is lower than the rated operating voltage.

**Expected sensor signatures:**

1. `battery_state_percentage` / `battery_state_charge`: directly below
   the threshold -- the most obvious signal, but **currently excluded
   from the feature set** (cumulative/monotonically decreasing, causing
   IQR-floor issues in z-score normalization, see
   `run_robo3er_joint_prototype_v3_federated.py`'s EXCLUDE comment).
2. `battery_state_voltage`: drops below nominal (~14.4V for a 4S LiPo
   at ~30% SOC). Also excluded for same reason.
3. `wheel_status_current_ma_*` (indirect): at low battery voltage, the
   same commanded velocity requires higher current from the motor (power
   = voltage × current, torque demand is unchanged, voltage is lower →
   current is higher). This is a **within-window detectable** signal IF
   the robot is actively driving. At rest, current is near zero regardless
   of battery state and this signal carries nothing.
4. Drive speed ceiling: if the controller applies soft speed limiting,
   `wheel_vels_velocity_*` values are bounded at a reduced maximum --
   only detectable if the robot is actively commanded beyond the limit.

**Why detection is hard (B=0.999 centralized, but 0.415 for robot00
specifically in FL):** The centralized result pools all 5 robots whose
data jointly calibrates the normal baseline -- a specific battery-level
signature can be detected if at least some robot demonstrates it
consistently. In the federated setting, robot00 (which has the Low
battery fault windows) calibrates only from its own 22-window calib
split, which may or may not include windows at comparable drive-activity
levels. The fundamental issue: whether `current_ma` is elevated at low
battery depends on HOW HARD the robot is driving in that window -- a
parked robot at 25% battery is indistinguishable from a parked robot at
80% battery by any kept feature.

**Recommended improvement**: Replace `battery_state_percentage` (raw
cumulative level, bad for IQR) with `battery_state_percentage_delta`
(rate of change per window, stationary and detects rapid drain) as a
new node. This would make the fault directly observable without the
IQR blowup problem, and represents the diagnostic logic a human
technician would apply ("battery draining faster than expected" is the
real fault signal, not "battery is low").
