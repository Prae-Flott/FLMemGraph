# v2: GDN-style attention edge head -- the first CLEAR, well-earned win
# on Paderborn (2026-08-07)

## The design correction that produced this

After `memory/paderborn-6ch-fair-comparison.md` showed v1's fixed-edge-
list `TrendEdgeHead` did NOT beat a fair (same 6-channel) plain-GDN
baseline, and a follow-up analysis identified several compounding
weaknesses (fixed 8-edge skeleton never validated edge-by-edge; a single
linear map per edge has far less capacity than GDN's learned attention;
no mechanism for edges the physics prior didn't declare), the user
corrected the design directly: edge/structural anomaly should be
produced by **GDN-style learned attention over neighbors** (same
mechanism as `gdn_model.GDN`/`fl_model.StructureHead`), with declared
physics relations acting as an **additive bias on attention logits**, not
a hard restriction -- "对于物理先验没有表示的边，GDN也可以学习他们之间的关系."

## What was built

`src/joint_prototype_model.py`:
- `TrendGraphAttentionHead` -- replaces v1's `TrendEdgeHead`. Same
  "operate on deviations from the matched joint prototype, not raw z"
  idea kept from v1 (still the mechanistically sound response to
  Paderborn's confounded-operating-condition collinearity problem). But
  now: TopK learned-embedding-similarity attention over ALL other nodes
  (not just the 8 declared pairs), no self-loop (same reasoning as
  `StructureHead` -- same-instant consistency check, not forecasting).
  Declared physics edges (the same 8 as v1) contribute a single LEARNED
  scalar bias added to attention logits for that (source, target) pair --
  informed, not restrictive.
- `JointPrototypeGDNv2` -- same encoder + `JointPrototypeMemory` as v1,
  `TrendGraphAttentionHead` as the edge/structural component. Per-node
  `s_edge` (structural anomaly per node, generalizing v1's fixed
  per-edge scores).
- `benchmark/run_paderborn_joint_prototype_v2.py` -- same node set
  (6 channels), same 8 physics edges (now bias not skeleton), same
  fit/calib/split, same 4 ablations as v1, `top_k=5` (~full attention at
  N=6, kept as an explicit knob for larger node sets later).

## Result: a real, well-earned win

| method | mean AUROC (26) | outer_ring | inner_ring | combined |
|---|---|---|---|---|
| AE (6ch, fair baseline) | 0.679 | 0.569 | 0.712 | 0.996 |
| v1 D (fixed-edge, full) | 0.802 | 0.736 | 0.823 | 0.991 |
| GDN (6ch, fair baseline) | 0.819 | 0.757 | 0.838 | 0.994 |
| v2 C: edge-only (attention) | 0.861 | 0.815 | 0.873 | 0.997 |
| **v2 D: full (attention)** | **0.873** | **0.829** | **0.888** | **0.997** |

**v2's full model beats the fair GDN baseline on every category** --
+0.054 mean AUROC, +0.072 on outer_ring, +0.050 on inner_ring. This is
the first time in this dataset's entire physics-prior exploration
(current residual, torque residual free-fit, torque residual fixed-mu,
v1 joint-prototype+fixed-edges) that ANY approach has clearly,
substantially beaten the strongest available baseline -- not a narrow or
ambiguous win.

Per-bearing (v2 D vs. fair GDN baseline): **11 improved, 2 got worse
(both tiny: KA09 -0.031, KI01 -0.021), 13 unchanged** (all already at the
1.000 ceiling in both). The biggest gains are dramatic and land exactly
on the bearings that were hardest for every prior method:

| bearing | GDN baseline | v2 full | delta |
|---|---|---|---|
| KI04 | 0.364 | 0.638 | **+0.274** |
| KA08 | 0.516 | 0.756 | **+0.240** |
| KA07 | 0.734 | 0.924 | **+0.190** |
| KI08 | 0.667 | 0.849 | **+0.182** |
| KA30 | 0.489 | 0.624 | +0.136 |
| KA01 | 0.556 | 0.700 | +0.144 |

Notably, `C: edge-only` (0.861) already beats the GDN baseline (0.819) by
itself, before even adding the node/memory signals -- the attention-based
structural consistency check is doing most of the work, exactly the
signal type the design correction was aiming to strengthen.

## Why this worked where v1 didn't

1. **Capacity match**: `TrendGraphAttentionHead`'s learned attention
   aggregation is the same expressive mechanism GDN itself uses, so the
   "physics prior" version is no longer handicapped relative to the
   plain baseline the way v1's single-linear-map-per-fixed-edge design
   was.
2. **Edges not restricted to the declared 8**: attention is computed over
   ALL other nodes (top_k=5 at N=6 is effectively full attention), so any
   useful relationship GDN's own baseline could discover is still
   available here -- the physics bias only ADDS information (a nudge
   toward known-relevant pairs), it never subtracts capacity.
3. **Still operates on deviations, not raw magnitudes** -- the one
   genuinely load-bearing idea from v1 survived the redesign: because
   attention aggregates `d_j = z_j - p_j*` (deviation from the matched
   joint prototype), the "which of the 4 operating conditions is this"
   confound is still sidestepped the same way it was in v1, but now
   with a mechanism capable of actually exploiting that clean signal.

`prior_bias_strength` grew from its 1.0 init to a peak of 1.247 around
epoch 6, then settled back to 1.113 by epoch 12 -- the model engaged with
the prior (didn't collapse it to ~0) but didn't lock onto it rigidly
either, consistent with "informed nudge, not a hard constraint" working
as intended.

## What's NOT done / open questions

- `top_k=5` was not tuned and is nearly meaningless at N=6 (only 5 other
  nodes exist) -- this design's real test of "does physics-biased
  attention scale" needs a larger node set (e.g. adding the 1Hz
  temperature channel, or the bearing-geometry defect-frequency features
  from `paderborn_physics.characteristic_frequencies()`, still unused)
  where top_k actually constrains something.
- No per-edge attention-weight inspection was done -- would directly show
  whether the model is actually using the declared physics edges more
  than average, or spreading attention roughly uniformly regardless of
  the bias (the aggregate `prior_bias_strength` growing is suggestive but
  not conclusive on its own).
- KA09 and KI01's small regressions (-0.031, -0.021) aren't explained --
  worth checking whether they share anything in common with each other.
- The typed-relation-strength extension flagged in v1's memory file
  (per-relation-type bias instead of one shared scalar, matching the
  ARCHIVED IMS-era `GDNWithTypedPrior` finding that typed bias gave the
  most consistent detection improvement of anything tried there) hasn't
  been tried here yet -- a natural next step given how well the
  single-scalar version already worked.

Full report: `checkpoints/paderborn_joint_prototype_v2_report.json`,
model weights: `checkpoints/paderborn_joint_prototype_v2.pth`.
