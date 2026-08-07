# Sielaff: physical prior relationships & formulas

**Status: no hard physical prior has been derived or verified for this
dataset.** This file exists to make that explicit rather than silently
absent, per the same documentation standard as `robo3er_physics.md` and
`paderborn_physics.md`. Everything currently protecting/detecting
anomalies for Sielaff (`benchmark/run_sielaff_gdn.py`) relies entirely on
GDN's own learned graph-attention structure (the SOFT prior -- see the
general design discussion in `docs/mem_phys_prompt_zh.md` Sec 2A/2B) --
there is no `kinematics.py`- or `paderborn_physics.py`-style residual
feature for this dataset.

## Why no hard prior exists yet (not for lack of trying to find one)

robo3er's hard prior (differential-drive kinematics) exists because the
platform is a simple, well-understood rigid-body mechanism with a
textbook governing equation. Paderborn's hard prior (bearing defect
frequencies, motor-current/torque relation) exists because rolling-
element bearing physics and basic motor physics are both well-established,
literature-verified relations. Sielaff (a reverse-vending machine: bottle/
can intake, sorting, compacting) is a much more complex, multi-subsystem
electromechanical machine -- per `data/sielaff_data/sielaff_ground_truth.md`,
its hardware includes Flexisort (sorting), Bandabweiser (belt diverter),
Kompaktor (compactor), Kameras (cameras, incl. the RingCamera features
already flagged as unreliable/sparsely-reported in
`run_sielaff_gdn.py`'s `RELIABILITY_RATIO` handling), and Kastenwaage
(crate scale) -- no single governing equation covers a system like this
the way rigid-body kinematics or bearing geometry does.

## Candidate relations that COULD be investigated (not done)

These are plausible, not verified -- listed as concrete next steps rather
than left unstated:

1. **Kastenwaage (crate scale) vs. journal/receipt counts**: a returned
   container's registered weight should be consistent with its
   registered category (bottle/can/crate) -- a mismatch could indicate a
   sensor fault or an attempted fraud pattern, a genuinely relational
   (cross-sensor) signal in the same spirit as robo3er's wheel-vs-odometry
   check.
2. **Brightness/cleaning-pixel counts vs. camera read success rate**: if
   `Helligkeit_Flaschenerkennung` (bottle-recognition brightness) drops
   and `cleaning_begin_pixels`/`cleaning_end_pixels` change accordingly,
   that's a plausible optical-fouling relation (dirty lens -> reduced
   brightness -> degraded recognition) with a testable direction, but no
   formula has been derived.
3. **Belt diverter (Bandabweiser) actuation vs. downstream sort outcome**:
   a commanded diverter state should predict which downstream path (e.g.
   which `Weiche` gate) a container takes -- analogous to robo3er's
   "commanded wheel motion should predict measured odometry," but the
   exact mechanical linkage/timing needed to build this relation is not
   documented anywhere available to this project.

## Current state

Only GDN's learned graph-attention (soft prior) is in use for this
dataset -- see `run_sielaff_gdn.py`'s module docstring for the scoring
convention (per-feature median/IQR normalize from calib, then max, with
`RELIABILITY_RATIO` masking for sparsely-reported sensors). No residual
feature engineering has been attempted. If this dataset gets revisited,
the Kastenwaage-vs-journal relation (candidate 1 above) is the most
promising starting point -- it has the cleanest "two independently
measured signals that should agree" structure, the same shape as
robo3er's and Paderborn's hard priors.
