# FLMemGraph project memory

Development history for this project, carried over from `~/Projects/FL-bench`
where this work started (see each file's own note on provenance).

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
- [`feature-purification-audit.md`](feature-purification-audit.md) — which
  of robo3er's 71 raw features are genuinely droppable (3, confirmed
  zero-variance) vs. kept on purpose (redundant-but-conditioning-useful
  continuous features; near-constant status flags handled at the scoring
  layer instead of by removal), plus a measured detection-AUROC vs.
  attribution-stability trade-off in the GDN scoring rule's IQR floor.
