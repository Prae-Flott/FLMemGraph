"""
Maps robo3er's 68 kept features (71 raw minus the 3 confirmed-dead
columns, see `dataset.py`) onto which physical quantity each one measures,
so a physics-informed model can be built up tier by tier instead of all-
or-nothing 71 features at once.

## The kinematic chain

Rigid-body kinematics is a chain of time-derivatives:

    displacement (position + orientation)
        --d/dt-->  velocity (linear + angular)
        --d/dt-->  acceleration

For a wheeled robot this chain exists at TWO levels, connected by the
robot's geometry (wheel radius, track width -- `kinematics.py`'s
differential-drive equations):

    wheel-level:   wheel position (ticks) -> wheel velocity
    chassis-level: chassis position/orientation -> chassis lin/ang velocity -> lin acceleration

And each chassis-level quantity may be measured through more than one
independent estimator:
  - **odometry** (`odom_odo_*`): computed onboard FROM the wheel encoders
    -- so `odom_odo_lintw_x`/`odom_odo_angtw_z` are not an independent
    check on the wheels, they're a re-expression of the same wheel data at
    chassis scale (this is exactly what `kinematics.py`'s residual
    strips out).
  - **IMU** (`imu_imu_*`): an independent inertial sensor (gyroscope +
    accelerometer), NOT derived from wheel encoders. `imu_imu_angvel_z`
    vs `odom_odo_angtw_z` is therefore a genuinely independent
    cross-check of the same physical quantity (chassis yaw rate) --
    arguably a MORE rigorous slip detector than wheel-vs-odom, since both
    sides of THAT comparison are wheel-derived. Not yet implemented in
    `kinematics.py`; a natural next tier.
  - **tf** (`tf_link_base_link_*`, `tf_footprint_base_footprint_*`): a
    software-broadcast re-expression of the SAME odometry estimate in a
    different frame, not an independent sensor -- useful as a "should
    match odom almost exactly" pipeline sanity check, not a physical
    fault signal.

## Groups

`KINEMATIC_CORE` (38 features): every raw topic that IS one of
displacement/velocity/acceleration, at either level, from any estimator.
This is the "minimal validation" set -- restrict a model to only this
group to test whether kinematics-derived signal alone (no actuation
current/PWM, no proximity/battery/docking sensors) is sufficient before
adding anything else.

`ACTUATION` (4): motor command/current -- these DRIVE the kinematics
(via a force/torque -> acceleration dynamics relationship, e.g. current
draw under load) but are not themselves a kinematic quantity. Natural
tier-2 addition once the pure-kinematics minimal model is validated.

`STATUS_FLAGS` (6): binary/near-constant status bits. `slip_status_is_slipping`
in particular is worth noting specially: it's the ROBOT'S OWN FIRMWARE
flag for the exact same physical event (wheel motion not matching chassis
motion) this project's kinematic residual is trying to detect from raw
signals -- i.e. it's a candidate "silver label" to validate the residual
against later, not a kinematics *input* feature.

`ENVIRONMENT` (20): proximity/IR/battery/docking sensors -- physically
unrelated to the robot's own equations of motion, excluded from the
kinematics-only minimal model entirely (not just deferred to a later tier
of the SAME model, since there's no equation-of-motion reason to ever
feed these into a kinematics-residual computation, only into the
full/final detector).

These 4 groups are mutually exclusive and exhaustive over the 68 kept
features (38+4+6+20=68).
"""

KINEMATIC_CORE = [
    # wheel-level: velocity and its position integral (ticks)
    "wheel_vels_velocity_left", "wheel_vels_velocity_right",
    "wheel_ticks_ticks_left", "wheel_ticks_ticks_right",
    # chassis-level position/orientation, velocity/angular-velocity -- odometry estimator
    "odom_odo_pos_x", "odom_odo_pos_y", "odom_odo_pos_z",
    "odom_odo_orient_x", "odom_odo_orient_y", "odom_odo_orient_z", "odom_odo_orient_w",
    "odom_odo_lintw_x", "odom_odo_lintw_y", "odom_odo_lintw_z",
    "odom_odo_angtw_x", "odom_odo_angtw_y", "odom_odo_angtw_z",
    # chassis-level orientation/angular-velocity/acceleration -- IMU estimator (independent)
    "imu_imu_orient_x", "imu_imu_orient_y", "imu_imu_orient_z", "imu_imu_orient_w",
    "imu_imu_angvel_x", "imu_imu_angvel_y", "imu_imu_angvel_z",
    "imu_imu_acc_x", "imu_imu_acc_y", "imu_imu_acc_z",
    # chassis-level position/orientation, re-broadcast frames (same source as odom)
    "tf_link_base_link_trans_x", "tf_link_base_link_trans_y", "tf_link_base_link_trans_z",
    "tf_link_base_link_rot_x", "tf_link_base_link_rot_y", "tf_link_base_link_rot_z", "tf_link_base_link_rot_w",
    "tf_footprint_base_footprint_trans_x", "tf_footprint_base_footprint_trans_y",
    "tf_footprint_base_footprint_rot_z", "tf_footprint_base_footprint_rot_w",
]

ACTUATION = [
    "wheel_status_current_ma_left", "wheel_status_current_ma_right",
    "wheel_status_pwm_left", "wheel_status_pwm_right",
]

STATUS_FLAGS = [
    "slip_status_is_slipping",
    "dock_status_is_docked", "dock_status_dock_visible",
    "stop_status_is_stopped", "kidnap_status_is_kidnapped",
    "ir_opcode_sensor",
]

ENVIRONMENT = [
    "cliff_intensity_cliff_side_left", "cliff_intensity_cliff_front_left",
    "cliff_intensity_cliff_front_right", "cliff_intensity_cliff_side_right",
    "ir_opcode_opcode",
    "battery_state_voltage", "battery_state_current", "battery_state_charge",
    "battery_state_capacity", "battery_state_temperature", "battery_state_percentage",
    "ir_intensity_ir_intensity_side_left", "ir_intensity_ir_intensity_left",
    "ir_intensity_ir_intensity_front_left", "ir_intensity_ir_intensity_front_center_left",
    "ir_intensity_ir_intensity_front_center_right", "ir_intensity_ir_intensity_front_right",
    "ir_intensity_ir_intensity_right",
    "mouse_mouse_x", "mouse_mouse_y",
]

GROUPS = {
    "kinematic_core": KINEMATIC_CORE,
    "actuation": ACTUATION,
    "status_flags": STATUS_FLAGS,
    "environment": ENVIRONMENT,
}


def active_feature_names(active_groups):
    """active_groups: list of group names, e.g. ["kinematic_core"] for the
    minimal model, ["kinematic_core", "actuation"] for tier 2, etc."""
    names = []
    for g in active_groups:
        names.extend(GROUPS[g])
    return names


def select_columns(data, cols, active_groups):
    """Filter `data` [N,T,F] and `cols` down to only the named groups,
    preserving `cols`'s original relative order."""
    active = set(active_feature_names(active_groups))
    keep_idx = [i for i, c in enumerate(cols) if c in active]
    missing = active - set(cols)
    if missing:
        raise ValueError(f"feature_groups names not found in cols: {missing}")
    return data[:, :, keep_idx], [cols[i] for i in keep_idx]
