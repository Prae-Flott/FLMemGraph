# Feature purification audit for test_gdn_physi/ (re-run on current, regenerated data)

Re-audited robo3er's 71 raw features on the CURRENT on-disk `data/robo3er/`
(regenerated 2026-07-23, 6043 windows / 5 classes with `docked` not yet in
this cut) rather than trusting the pre-regeneration stats in
`robo3er-anomaly-detection-approaches.md`/`deployment-architecture.md`.
Conclusion landed in the same place as before, but re-verified, not
assumed. See `test_gdn_physi/dataset.py`'s module docstring for the
in-code version of this reasoning.

## Verdict: drop 3, keep 68

**Drop (confirmed exactly zero-variance, 1 unique value, across ALL 6043
windows -- normal AND every one of the 4 fault types, checked directly,
not inferred)**:
- `tf_footprint_base_footprint_trans_z`
- `tf_footprint_base_footprint_rot_x`
- `tf_footprint_base_footprint_rot_y`

**Keep everything else (68 features)**, for two different reasons:

1. **32 pairs with |corr|>0.95 on the current fit-split data** (cliff
   sensors + `ir_opcode_opcode` collapse to ~1 signal; pose triplicated
   across `odom_odo_pos_*`/`tf_link_base_link_*`/
   `tf_footprint_base_footprint_*`; battery voltage/temperature/current;
   wheel_ticks left/right) -- same redundant clusters the pre-regeneration
   audit found. Not dropped: `auto_encoder/train_reduced_feature_ae.py`
   already tested removing 19 such features (71->52) and found `stuck`
   got WORSE (0.796->0.717) from losing conditioning context, even though
   3/4 other fault types improved slightly. GDN's graph-attention
   architecture is if anything MORE sensitive to this than the flat
   autoencoder that finding came from, since a neighbor's raw feature is
   used directly (not just via a shared encoder) to predict another
   node's value -- removing a node removes it from every neighbor pool it
   was ever a candidate for, not just from its own input.
2. **6 near-constant/binary status flags** (`slip_status_is_slipping`,
   `dock_status_is_docked`, `dock_status_dock_visible`,
   `stop_status_is_stopped`, `kidnap_status_is_kidnapped`,
   `ir_opcode_sensor`) are simultaneously (a) the root cause of GDN's
   documented per-feature z-score blowup (near-zero calib IQR -> any tiny
   error explodes the robust z-score) and (b) robo3er's clearest fault
   SIGNATURES (`deployment-architecture.md`). Dropping them would remove
   real detection signal for 3 of 4 fault types. Handled at the scoring
   layer instead (see below), not by removal.

## The scoring-layer fix has a real trade-off, measured -- not applied by default

`dataset.gdn_score(..., iqr_floor_frac=0.1)` floors each feature's IQR at
a fraction of the median IQR across informative features, instead of the
original `max(iqr, 1e-8)` (a floor too small to stop the blowup).
Measured on the current data (physics-residual GDN, dense history=30):

| fault | epsilon floor (matches baseline) | iqr_floor_frac=0.1 | delta |
|---|---|---|---|
| Broken Pipe | 0.956 | 0.922 | -0.034 |
| cable trapped | 0.905 | 0.909 | +0.004 |
| Low battery | 0.957 | 0.947 | -0.010 |
| stuck | 0.742 | 0.630 | -0.112 |

The floor fix reduced AUROC on 3/4 fault types here (numbers above are
from two back-to-back same-config runs, so part of this is also GDN's
documented run-to-run instability, not purely the floor's effect -- see
`robo3er-anomaly-detection-approaches.md`'s SHAP-vs-GDN section). Net:
**this is a real detection-AUROC vs. attribution-stability trade-off, not
a strict win** -- the near-constant flags' "blown up" z-scores happen to
already point at the right feature for the fault types they signature
(e.g. `slip_status_is_slipping` for cable trapped), so damping that
blowup for interpretability's sake costs some detection sensitivity for
exactly those fault types. Kept OFF by default in `dataset.gdn_score`;
enable explicitly only if the deployment use case values stable
attribution over peak AUROC on those fault types.

## dataloader

`test_gdn_physi/dataset.py` centralizes: loading + the 3-column drop,
fit/calib/test_normal chronological split, scaler fit/transform, and
`gdn_score`. `train_gdn_physics.py` now imports from it instead of
duplicating the split logic (previously copy-pasted from
`test_gdn/train_gdn.py`). Any future script in `test_gdn_physi/` should
import from here rather than re-deriving the split.
