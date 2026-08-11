# FLMemGraph project memory

Development history for this project, carried over from `~/Projects/FL-bench`
where this work started (see each file's own note on provenance).

- [`joint-prototype-scheme-b.md`](joint-prototype-scheme-b.md) — **Scheme B**
  (Joint Prototype Memory, prototype + per-node deviation, NO edges),
  consolidated final conclusions across all 4 datasets tested. Best or
  near-best signal on 3 of 4 (Sielaff 0.978 vs. a 0.887 GDN baseline,
  robo3er 0.942, voraus-AD 0.756 winning/tying 11/12 fault categories) —
  needs no physics prior, mechanism-agnostic, the recommended default
  starting point for any new dataset.
- [`joint-prototype-scheme-v3.md`](joint-prototype-scheme-v3.md) — **Scheme
  V3** (`JointPrototypeGDNv3`: Scheme B + typed relation-specific edges +
  prototype-conditioned standardization + anomaly attention),
  consolidated final conclusions plus a condensed lineage (physics-
  residual precursor -> v1 fixed-edge-list -> v2 learned attention -> v3
  typed relations, intermediate code removed). Wins clearly only on
  Paderborn (0.904 vs. a 0.819 GDN baseline) — the one dataset with
  literature-verified physical relations a real fault mechanism actually
  breaks. Documents the max-aggregation noise-floor problem and its
  two-stage top-k-mean fix, found while expanding voraus-AD's graph.
- [`voraus-ad-dataset.md`](voraus-ad-dataset.md) — downloaded and
  structure-verified `voraus-AD` (vorausrobotik 6-DOF pick-and-place arm,
  2122 samples, 12 named fault categories, 130 machine-data signals: a
  full per-joint kinematic+electromechanical chain, six joints x 21
  signals each). Richest physics-graph structure of any dataset in this
  project so far.
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
  Paderborn, reusing `src/fl_model.FLGDNMemory` unchanged (a different
  architecture line from the Joint Prototype scheme above). Two real data
  bugs found+fixed (non-constant per-file sample counts, one unparseable
  `.mat` file). Category-mean AUROC looks strong (combined 1.000,
  inner_ring 0.750, outer_ring 0.641) but hides a real bimodal split: 15
  of 26 damaged bearings near-perfect, 10 BELOW 0.5 AUROC (score
  direction inverted, mostly artificial-damage bearings) — leading
  hypothesis is that crude decimation washes out the high-frequency
  impulsive signature artificial single-point defects rely on, untested.
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
