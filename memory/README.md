# FLMemGraph project memory

Development history for this project, carried over from `~/Projects/FL-bench`
where this work started (see each file's own note on provenance).

- [`voraus-ad-dataset.md`](voraus-ad-dataset.md) — downloaded and
  structure-verified `voraus-AD` (vorausrobotik 6-DOF pick-and-place arm,
  2122 samples, 12 named fault categories, 112 machine-data signals: a
  full per-joint kinematic+electromechanical chain, six joints x 18
  signals each). Richest physics-graph structure of any dataset in this
  project so far — a strong next JointPrototypeGDNv3 target.
- [`voraus-ad-joint-prototype-v3.md`](voraus-ad-joint-prototype-v3.md) —
  first V3 run on voraus-AD (18 nodes: 3 signals x 6 joints,
  within-joint-only typed edges). Result is WEAK across the board (mean
  AUROC 0.66-0.68, most categories 0.55-0.72, `entangled` near/below
  chance) — but **unlike Paderborn/robo3er, no fair GDN/AE baseline
  exists yet on this dataset**, so these numbers can't be read as
  good/bad/better-than-v2 until that comparison is built. Several
  untested hypotheses noted (no cross-joint edges, zero-padding dilution
  from variable-length samples, untuned hyperparameters for the much
  larger 18-node/1164-timestep scale).
- [`fl-memory-physics-system.md`](fl-memory-physics-system.md) — the
  federated memory+structure system itself: encoder/memory-head/
  structure-head design, cross-robot codebook alignment, two real bugs
  found and fixed (independent-init cosine similarity being meaningless,
  tiny-client IQR blowup), current results, and the known unresolved
  robot00-small-sample limitation. Read this first.
- [`physics-informed-gdn.md`](physics-informed-gdn.md) — the
  differential-drive kinematic residual that produced this project's best
  `stuck`-fault result, and why it sidesteps the shrinkage bias that beat
  every prior forecasting-based approach on that fault type.
- [`paderborn-bearing-dataset.md`](paderborn-bearing-dataset.md) — a
  real dataset: Paderborn University's KAt Bearing DataCenter (32 type-6203
  bearings — 6 healthy, 12 artificial-damage, 14 real accelerated-lifetime
  damage — motor current + vibration + mechanical channels, official
  direct-download mirror, no registration needed). Has DEFINITE per-bearing
  damage-class ground truth (verified damage-code taxonomy: KA=outer-ring,
  KI=inner-ring, KB=combined), close to robo3er/Sielaff's
  fit-on-normal/per-class-AUROC convention.
  64 per-bearing PDF fact sheets/measuring logs extracted from inside the
  archives into `docs/paderborn_bearing_facts/` per explicit request.
- [`paderborn-fl-model-run.md`](paderborn-fl-model-run.md) — the full
  single-machine (no federation) memory+structure pipeline run on
  Paderborn, reusing `src/fl_model.FLGDNMemory` unchanged. Two real data
  bugs found+fixed (non-constant per-file sample counts, one unparseable
  `.mat` file). Category-mean AUROC looks strong (combined 1.000,
  inner_ring 0.750, outer_ring 0.641) but hides a real bimodal split: 15
  of 26 damaged bearings near-perfect, 10 BELOW 0.5 AUROC (score
  direction inverted, mostly artificial-damage bearings) — leading
  hypothesis is that crude decimation washes out the high-frequency
  impulsive signature artificial single-point defects rely on, untested.
- [`paderborn-physics-residual-gdn.md`](paderborn-physics-residual-gdn.md)
  — same method as robo3er's `train_gdn_physics.py`: fit motor
  current-envelope ~= a·torque + b·speed + c from healthy data, replace
  current with the residual, train plain GDN. Unlike robo3er, this made
  things WORSE for 23/26 damaged bearings — the fitted relation's R²
  (~0.25) is far weaker than kinematics.py's near-deterministic one,
  so residualizing against it injects noise instead of removing
  explained variance. A genuine negative finding, not a bug — see the
  file for why `kinematics.py`-style residuals need a strong underlying
  relation to help.
- [`paderborn-torque-residual-gdn.md`](paderborn-torque-residual-gdn.md)
  — follow-up per user design direction (unify speed/torque/friction/load
  into ONE torque-balance equation: predicted_torque = a·force + b·speed
  + c). ALSO made detection worse (10/26 bearings). Root cause found: the
  fitted force coefficient is NEGATIVE (physically implausible) because
  Paderborn's 4 operating conditions don't vary force/speed/torque
  independently — force only takes 2 values, confounded with the other
  two — so a 2-predictor OLS fit across 4 discrete points can't reliably
  separate the physical effects. Explains BOTH physics-residual failures
  on this dataset as one methodological root cause, not two unrelated
  weak-signal findings.
- [`paderborn-fixed-mu-torque-residual.md`](paderborn-fixed-mu-torque-residual.md)
  — re-fit with the force coefficient FIXED at a published catalog
  friction coefficient instead of freely fit. Confirms the collinearity
  diagnosis (every bearing that degraded, degraded LESS with the fixed
  coefficient) but still net negative overall — R² dropped further to
  0.089, revealing bearing friction is genuinely a small fraction of this
  rig's total shaft torque, not just a collinearity artifact.
- [`paderborn-joint-prototype.md`](paderborn-joint-prototype.md) — after
  3 consecutive negative parameter-level-residual results, pivoted to
  **trend-based** relations per `docs/joint_prototype_three_level_anomaly_prompt.md`
  (new design doc, added to `docs/`): `src/joint_prototype_model.py`'s
  `JointPrototypeMemory` (prototypes are full multi-node joint-state
  snapshots) + `TrendEdgeHead` (predicts a node's DEVIATION from its own
  matched-prototype baseline from another node's deviation — relative,
  not absolute magnitude). **First net-positive physics-informed result
  on Paderborn** — beats the prior best baseline on outer_ring
  (0.697→0.736) and inner_ring (0.819→0.841), though not uniformly (7/26
  bearings improved, 6/26 worse). Confirms the redesign's core bet: the
  problem wasn't that torque/force/current don't relate physically, it
  was that exact-magnitude regression couldn't separate the effects
  across this dataset's 4 confounded operating conditions — relative
  deviation-tracking sidesteps that entirely.
- [`paderborn-6ch-fair-comparison.md`](paderborn-6ch-fair-comparison.md)
  — the honest correction: the joint-prototype result above was compared
  against a 3-channel GDN baseline, not apples-to-apples. Reran plain GDN
  and AE on the SAME 6 channels — **plain GDN actually beats Joint
  Prototype on every category** (mean AUROC 0.819 vs. 0.802) once
  compared fairly. The added architectural complexity doesn't clearly
  pay for itself in raw detection AUROC on this dataset — its real value
  is node/edge-level localization/interpretability, not accuracy. Also
  found AE flipped from strongest (3-channel runs) to weakest (6-channel,
  0.679) — open question why.
- [`paderborn-joint-prototype-v2-attention.md`](paderborn-joint-prototype-v2-attention.md)
  — design correction (per user feedback): edge/structural anomaly should
  come from GDN-style LEARNED ATTENTION over neighbors, not a fixed
  hand-declared edge list — physics-known edges bias attention logits
  (one learned scalar) but don't restrict which relationships can be
  learned, so undeclared relationships stay fully learnable. Still
  operates on deviations from the matched joint prototype, keeping v1's
  sound response to the collinearity problem. **Result: the first clear,
  substantial win in this dataset's whole physics-prior exploration** —
  beats the fair 6-channel GDN baseline on every category (mean AUROC
  0.873 vs. 0.819), 11/26 bearings improved vs. only 2 with tiny
  regressions. Per-dataset
  physical-prior reference docs
  (including ones with NO hard prior found, like Sielaff) are in
  `benchmark/datasets/*_physics.md`.
- [`robo3er-joint-prototype-v3.md`](robo3er-joint-prototype-v3.md) — first
  run of `JointPrototypeGDNv3` outside Paderborn, on robo3er's kinematic
  chain (7 nodes: wheels/odom/IMU/current). **Opposite pattern from
  Paderborn**: node/prototype signal alone (0.942 mean AUROC) beats the
  full v3 combination (0.861) — the new typed-edge signal is the weakest
  in the table and net-HURTS via max-aggregation, especially on `stuck`
  (this project's historically hardest fault). Confirms edge-vs-node
  signal strength is dataset-dependent (Paderborn's faults break
  cross-channel relationships; robo3er's faults look like single-node
  value anomalies), sharpening the case for a learned (not fixed-max)
  signal combination.
- [`paderborn-joint-prototype-v3-typed-attention.md`](paderborn-joint-prototype-v3-typed-attention.md)
  — per `docs/joint_prototype_physics_gdn_anomaly_attention_prompt.md`:
  adds relation-specific message functions (linear for current edges,
  MLP for vibration/torque edges), prototype-conditioned edge-residual
  standardization, and a separate unsupervised anomaly attention,
  alongside (not replacing) v2's generic attention. **Full v3 beats v2's
  original result (0.898 vs. 0.873 mean AUROC)** and the GDN baseline
  (+0.079), but with an important honest caveat: the new typed-edge
  signal is WEAKER standalone (0.816) than v2's generic attention alone
  in the same run (0.904) — most of the headline gain traces to v2's own
  mechanism scoring higher when retrained jointly with the new auxiliary
  loss (likely a shared-encoder regularization effect), not to the new
  mechanism itself out-performing generic attention. Only 5/16
  prototypes had enough calib data for genuine prototype-conditioned
  stats, so that idea is under-exercised here too.
- [`fl-bench-migrations.md`](fl-bench-migrations.md) — four pieces
  migrated from `~/Projects/FL-bench` as real, standalone, in-repo runnable
  code (not framework pointers): a real FedAvg baseline (the registry
  previously falsely claimed one was already implemented), the tuned-GDN
  training procedure, an IFCAAE (unsupervised clustered FL) baseline, and
  a second real dataset (Sielaff, 10 reverse-vending machines) with its
  own tuned-GDN run. Read this for why each was reimplemented standalone
  rather than ported directly, and the first-run numbers for each.
- [`feature-purification-audit.md`](feature-purification-audit.md) — which
  of robo3er's 71 raw features are genuinely droppable (3, confirmed
  zero-variance) vs. kept on purpose (redundant-but-conditioning-useful
  continuous features; near-constant status flags handled at the scoring
  layer instead of by removal), plus a measured detection-AUROC vs.
  attribution-stability trade-off in the GDN scoring rule's IQR floor.
