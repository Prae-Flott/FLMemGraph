# FLMemGraph project memory

Development history for this project, carried over from `~/Projects/FL-bench`
where this work started (see each file's own note on provenance).

## Active architecture

**Single recommended model: `JointPrototypeV31` federated+sync** (`src/models/joint_prototype_model.py`).
All non-competitive baselines (GDN, FedAvg, IFCAAE, Paderborn FL Model, V2, V3)
and corresponding checkpoints deleted 2026-08-16.

**Standing policy (2026-08-17): [`benchmark-policy-federated-only.md`](benchmark-policy-federated-only.md)
-- all benchmark runs must use the federated (`*_federated.py`) script variant going forward,
never centralized. Read this before running or writing any new benchmark script.**

**2026-08-17 memory cleanup**: removed `fl-bench-migrations.md`,
`paderborn-fl-model-run.md`, `fl-memory-physics-system.md` (all
historical, describing deleted architectures/scripts), `joint-prototype-scheme-v2.md`
and `joint-prototype-v2-1-sielaff.md` (both superseded by, and partly
contradicting, `scoring-signals-B-C-E-H.md`'s current federated numbers).
`joint-prototype-scheme-v3.md`'s stale results table was also trimmed in
favor of `scoring-signals-B-C-E-H.md`, which now carries the full
cross-dataset B/C/E/F/H/I/J matrix. Physics-explicit exploration memories
(`physics-informed-gdn.md`, `robo3er-explicit-physics-scoring.md`) were
kept in full per explicit request, even though the direction itself is
closed.

- [`calib-in-prototype-ab.md`](calib-in-prototype-ab.md) — per-prototype vs.
  EMA calibration for B/C/K (`--calib-mode`): helps robo3er, hurts Sielaff.
- [`sielaff-num-prototypes-sweep.md`](sielaff-num-prototypes-sweep.md) — `NUM_PROTOTYPES` sweep: partially, not fully, explains Sielaff's regression.
- [`three-dataset-bck-comparison.md`](three-dataset-bck-comparison.md) —
  **cross-dataset B/C/K/BK/CK comparison, detection AUROC vs. diagnosis
  accuracy, all 3 datasets.** Headline finding (robo3er + Paderborn, on
  solid footing): the two tasks rank signals differently, sometimes
  OPPOSITELY, on the same dataset — Paderborn's best detector (C, 0.976
  AUROC) is nearly its worst localizer (50.9%), while its worst-but-one
  detector (B, 0.913) is by far its best localizer (93.8%); robo3er is
  the one dataset where the tasks roughly agree (K/BK/CK lead both).
  Per-dataset diagnosis detail in
  [`robo3er-diagnosis-localization.md`](robo3er-diagnosis-localization.md),
  [`paderborn-diagnosis-localization.md`](paderborn-diagnosis-localization.md),
  [`sielaff-diagnosis-localization.md`](sielaff-diagnosis-localization.md).
  **Sielaff's diagnosis doc was retracted and redone (2026-08-19)** against
  the correct fault-type ground truth (the 47 company/domain-expert
  RED-severity error IDs, not `data/sielaff/`'s coarse 5-group scheme —
  see `sielaff-red-fault-federated.md`), but the redone numbers are
  still flagged NOT FULLY TRUSTWORTHY: every signal's aggregate sits
  BELOW a domain-size-weighted random-guess baseline (~48.8%), traced to
  a second-order "activity proxy" confound
  (`receipt_count`/`reject_count`/`cleaning_duration`) that survives the
  already-applied `transaction_throughput` exclusion. Two follow-up
  fixes tested: dropping 22 hardware-heterogeneous RingCamera/gate
  columns made K/BK/CK WORSE (ruled out, also ruling out per-client node
  masking as worth building); activity-level residualization (regress
  B/C/K's node deviations on a window-activity covariate, score the
  residual) genuinely improved C (13.4→23.6%, a real domain-relevant
  argmax shift, verified) but only partly cleared B's confound and left
  K unmoved — still below baseline overall, leave-one-out activity proxy
  flagged as the next refinement. Also documents the diagnosis-scoring
  methodology itself (hand-built domain maps, not learned labels), the
  general max()-vs-argmax mechanism for why detection AUROC and
  diagnosis accuracy diverge, and a 5-factor structural explanation for
  why Sielaff's diagnosis ceiling is far below robo3er's/Paderborn's.
- [`scoring-signals-B-C-E-H.md`](scoring-signals-B-C-E-H.md) — **THE reference
  for the four anomaly signals** B (node amplitude), C (structural attention
  residual), E (typed physics edge residual), H (joint covariance Mahalanobis).
  Principle, formula, and per-dataset performance for each. Recommended
  combination per dataset: Paderborn→max(C,E,H); Robo3er→max(B,C,H) or H alone;
  Sielaff→max(B,C). **Read this first when choosing a scoring strategy.**
- [`joint-prototype-scheme-v3.md`](joint-prototype-scheme-v3.md) — **V3/V3.1**
  architecture reference: typed relation edges + prototype-conditioned
  standardization, the Lineage section (why the design evolved through
  fixed-edge → attention-only → typed-relation), and the max-aggregation
  change (2026-08-16). Its old cross-dataset results table was removed
  2026-08-17 as stale/conflicting with current numbers — for current
  per-dataset performance see `scoring-signals-B-C-E-H.md` instead.
- [`forecast-head-signal-k.md`](forecast-head-signal-k.md) — signal K
  (`ForecastHead`, GDN-attention forecast in raw-signal space, per
  `docs/shared_forecast_head_proposal.md`): robo3er centralized. v1
  (in-window split) badly understated K (0.679 mean) due to a data-leak
  artifact; v2 (cross-window pairing) recovers K to 0.803 mean. Two
  follow-up experiments: (1) declared-edge prior ablation shows K does
  NOT depend on the physics-edge bias (0.808 without it, even slightly
  higher) -- keep prior-injection effort on signal C, not K; (2) a
  multi-step forecast-horizon sweep is the standout finding -- at
  `horizon_mult>=4`, K actually EXCEEDS H specifically on `stuck` (0.819
  vs H's 0.739 at horizon_mult=8), the first signal in this project to
  beat H there. Full max(B,C,E,H,K) still corrupts ranking, but the
  PAIRWISE `HK = max(H, K)` combination (skipping B/C/E) is a genuine
  win -- 0.901 mean at horizon_mult=10, the best single scoring rule
  found across this whole investigation, because H and K are
  complementary (each covers the other's weak fault type) while B/C/E
  are weak on `stuck` the same way K's other pairings already are.
- [`robo3er-diagnosis-localization.md`](robo3er-diagnosis-localization.md)
  — node-level root-cause LOCALIZATION accuracy (does the argmax point at
  the physically correct sensor), not detection AUROC, for
  B/C/H/K/BK/CK/HK. Federated (per `benchmark-policy-federated-only.md`;
  also signal K's first-ever federated run), 26-node feature set
  (`kinematic_core`+`actuation`, matching
  `joint-prototype-federated-results.md`'s current architecture). Adds
  H's per-node Mahalanobis contribution decomposition (previously
  missing). Key results: `stuck` localization hits 85.9-100%
  (`BK`/`CK`=100%), driven by a clean 26-node candidate pool (only
  `wheel_*` and a few IMU/odom nodes are even reachable). `cable
  trapped`'s direct domain is structurally the EMPTY SET at 26 nodes
  (`slip_status_is_slipping` lives outside the active feature groups) --
  only direct-or-indirect is meaningful there; K/BK/CK/HK lead (73-78%),
  H is merely average (48.5%). Includes a follow-up mechanistic analysis:
  B and C are near-identical for these two faults (argmax agreement
  74.9%, Pearson r=0.994) because C's attention-corrected residual can't
  cancel a fault that also elevates the attended neighbor; K is the
  genuinely distinct signal (argmax agreement with B/C only 42-44%); and
  BK vs CK are empirically indistinguishable sample-for-sample (BK
  recommended as the more mechanistically diverse, cheaper pairing).
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
- [`joint-prototype-federated-results.md`](joint-prototype-federated-results.md) —
  V2/V3 run FEDERATED (not centralized) for the first time, across all 4
  datasets. Sielaff (10 real machines) and Paderborn (6 synthetic
  per-bearing clients) hold up or beat centralized; robo3er (5 real
  robots, tiny per-client samples) and voraus-AD (5 fully synthetic
  time-chunk clients, single arm) collapse toward chance -- tracks almost
  exactly with whether `align_and_split` found any cross-client shared
  prototype clusters at all (0 rounds for voraus-AD). Also documents the
  node-aware rewrite of `federated_memory.align_and_split` (keeps a
  per-node similarity dimension for `[M, N, D]` joint-prototype
  codebooks instead of flattening) and a real `REPO_ROOT` path bug fixed
  in `src/robo3er/dataset.py`/`fl_dataset.py`.
- [`sielaff-red-fault-federated.md`](sielaff-red-fault-federated.md) —
  Sielaff V2 federated restricted to domain-expert RED-severity faults
  only (47 IDs from `sielaff_ground_truth.md`), on a freshly-built
  windowed dataset (`data/sielaff_red/`, `build_sielaff_red.py`) since
  severity isn't recoverable from the existing group_6-labeled arrays.
  4h/1h-stride window definition, red AUROC 0.905-0.941 federated, no
  collapsed machine. Follow-up: per-fault feature attribution (which
  sensor deviates most per specific red error ID) -- correctly recovers
  cleaning/printer mechanisms but honestly degrades to a generic
  activity-proxy for compactor/crate/safety faults the raw logs never
  instrumented a sensor for.
- [`feature-purification-audit.md`](feature-purification-audit.md) — which
  of robo3er's 71 raw features are genuinely droppable (3, confirmed
  zero-variance) vs. kept on purpose (redundant-but-conditioning-useful
  continuous features; near-constant status flags handled at the scoring
  layer instead of by removal), plus a measured detection-AUROC vs.
  attribution-stability trade-off in the GDN scoring rule's IQR floor.
- [`robo3er-explicit-physics-scoring.md`](robo3er-explicit-physics-scoring.md)
  — **CLOSED DIRECTION.** Explicit `y=k·x+b` scorers tested on robo3er and
  Paderborn, all deleted. Robo3er apparent win was a representation-level
  difference (raw kinematics vs. embedding space), NOT an aggregation fix —
  confirmed by experiment: changing TypedRelationAnomalyHead from softmax→max
  left all robo3er metrics unchanged (E stays 0.760, J stays 0.725). The
  actual best robo3er signal is H_cov_mahal (0.948 federated). Paderborn
  explicit scorer below random (faults are spectral, not mean-value).
