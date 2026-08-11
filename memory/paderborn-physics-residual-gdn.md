# Paderborn: physics-residual GDN, same method as robo3er (2026-08-07)

## What was built

Following the exact pattern of robo3er's `src/train_gdn_physics.py`
(fit a physical relation from healthy data only, replace a raw signal
with its residual, train plain GDN forecasting, no memory yet):

- `src/paderborn_physics.py`: bearing 6203 defect-frequency formulas
  (reference, geometry verified against the paper) + the actual residual
  used, `fit_current_model()`/`apply_current_residual()` -- basic motor
  physics, current envelope ~= a*torque + b*speed + c.
- `benchmark/datasets/paderborn_adapter.py` extended:
  `load_bearing_with_physics()` also loads `force`/`speed`/`torque`
  (native ~4kHz, resampled onto the same decimated time axis as the
  electrical channels via a new `_decimate_blocks()` helper that uses
  `np.array_split` instead of an exact reshape, robust to non-integer
  decimation ratios). Current channels now use RMS-per-block decimation,
  not mean -- mean-decimating an oscillating AC signal is the wrong
  envelope representation for regressing against torque (a DC-ish
  quantity); the earlier `run_paderborn_fl_model.py` run used mean
  decimation for current too, now understood as a real (if not
  necessarily fatal) methodological wrinkle in that earlier run.
- `benchmark/run_paderborn_physics_gdn.py`: trains plain `src/gdn_model.GDN`
  (forecasting, dense pairs -- matching `run_bearing_gdn.py`'s mechanism,
  NOT `fl_model.FLGDNMemory`'s reconstruction task) twice, same seed/
  architecture, only the current channels differ (A: raw RMS-decimated
  current: B: physics-residual current) -- a clean single-variable
  ablation.

## Result: the residual made things WORSE -- an honest negative, the
## opposite of robo3er's result

| category | raw current AUROC | physics-residual AUROC | delta |
|---|---|---|---|
| outer_ring | 0.697 | 0.598 | -0.100 |
| inner_ring | 0.819 | 0.751 | -0.068 |
| combined | 0.998 | 0.970 | -0.029 |

23 of 26 damaged bearings got worse with the residual (full per-bearing
table in `benchmark/datasets/paderborn_physics.md`). The fitted
torque/speed -> current relation has R^2 only ~0.25 on healthy fit data --
torque and speed together explain barely a quarter of the current
envelope's normal-operation variance. `kinematics.py`'s equivalent
relation (wheel velocity -> odometry) is near-deterministic under normal
conditions by contrast, which is why THAT residual isolates anomalies
cleanly while this one mostly just subtracts a noisy, weak prediction and
leaves GDN with a harder-to-model remainder than the raw signal was.

**This is a genuine, useful negative finding, not a bug** -- verified the
regression/residual computation has no leakage (fit params come only
from the FIT split, applied identically to calib/test_normal/fault).
The takeaway generalizes beyond this one dataset: `kinematics.py`-style
residualization is only worth doing when the underlying physical relation
is STRONG (high R^2 / near-deterministic) under normal operation --
weaker relations (like this one) can make things worse by injecting
regression noise rather than removing genuine "explained" variance.

## Secondary finding (not the main comparison, but notable)

This run's "raw current" arm (plain GDN forecasting + RMS-decimated
current) scored HIGHER than `run_paderborn_fl_model.py`'s earlier result
(mean-decimated current + `FLGDNMemory` reconstruction) on both
overlapping categories: outer_ring 0.697 vs 0.641, inner_ring 0.819 vs
0.750. Plausibly the RMS-vs-mean current-decimation fix, though task
type (forecasting vs. reconstruction) also differs between the two runs
-- confounded, not isolated as a controlled comparison.

## What's NOT done

- The bearing-geometry defect-frequency formulas
  (`characteristic_frequencies()`) are implemented but never exercised as
  an actual feature (no envelope-spectrum band-energy extraction was
  built for this dataset, unlike the deleted IMS exploration that tried
  this) -- flagged in `paderborn_physics.md` as the more promising
  unused avenue if this dataset gets revisited, since it doesn't depend
  on a weak linear torque/speed fit.
- `force` is loaded by `load_bearing_with_physics()` but not used in the
  current residual model (only torque + speed) -- untried whether adding
  it as a third regressor improves the R^2 enough to flip this result.
- No memory head added to this physics-residual arm yet (matches
  robo3er's own historical order: `train_gdn_physics.py` came before
  `train_gdn_memory.py`) -- natural next step once/if a working physics
  signal is found for this dataset.

Full report: `checkpoints/paderborn/paderborn_physics_gdn_report.json`.
