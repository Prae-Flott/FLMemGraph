---
name: robo3er-diagnosis-localization
description: Node-level root-cause localization accuracy (not detection AUROC) for B/C/H/K/BK/CK/HK on robo3er, federated (per benchmark-policy-federated-only.md), 26-node current feature set. Adds H's per-node Mahalanobis contribution decomposition and a B/C/K mechanistic comparison.
metadata:
  type: project
---

# robo3er diagnosis localization — federated, 26-node (current config)

Follow-up to `diagnosis_interpretability_review.md`'s methodology table
(section 2) and its section-6 recommendation #1. All prior numbers in
[[forecast-head-signal-k]] and [[scoring-signals-B-C-E-H]] are DETECTION
AUROC (does the score separate normal from fault windows). **This is a
different question: when a fault window IS flagged, does the signal's
own argmax mechanism point at the physically correct node?** Run
federated per `benchmark-policy-federated-only.md` — 5 per-robot
clients, `JointPrototypeV31Forecast`, FedAvg'd encoder/decoder + aligned
prototype memory, 5 rounds x 12 local epochs, `horizon_mult=10`. This is
also **signal K's first-ever federated training run** (previously
validated centralized-only, per [[forecast-head-signal-k]]'s caveat).
Script: `benchmark/diagnose_robo3er_localization_federated.py`, built on
`run_robo3er_forecast_v2_federated.py` (imported as a module). Each
client evaluated on its own fault windows with its own calibration
stats; results pooled across all 5 clients into one hit-rate table per
signal per fault type.

## New code: H's per-node contribution decomposition

`DeviationCovarianceHead` has no native per-node output (only the scalar
Mahalanobis distance `H`). Implemented the standard Hotelling-T²
contribution-plot expansion, read-only against the head's existing
calibration buffers, no model change:

```
diff_i         = d_node_i - mu*(idx)_i
contribution_i = diff_i * sum_j cov_inv*(idx)_ij * diff_j
argmax_i contribution_i  =  H's diagnosed node
```

`sum_i contribution_i` equals the scalar Mahalanobis distance exactly.

## Ground-truth domain at 26 nodes

Node set is `feature_groups.KINEMATIC_CORE + ACTUATION` minus
`EXCLUDE_NODES_PREFIXES` (drops `wheel_ticks_*`, `odom_odo_pos_*`,
footprint/tf columns), matching
[[joint-prototype-federated-results]]'s current architecture.

- **`cable trapped`'s direct domain is the EMPTY SET.** Its only
  direct-domain member, `slip_status_is_slipping`, lives in
  `feature_groups.STATUS_FLAGS`, not one of this run's two active
  groups. Every signal's `direct_pct` for `cable trapped` is therefore
  structurally 0% here — a candidate-pool exhaustion, not a mechanism
  failure. Only `direct_or_indirect_pct` is meaningful for this fault
  type.
- **`stuck`'s direct domain (`wheel_*`) is intact**: all 6 of its 26-node
  `wheel_*` members (`wheel_vels_velocity_left/right`,
  `wheel_status_current_ma_left/right`, `wheel_status_pwm_left/right`)
  survive the reduction.

## Random-guess baseline at 26 nodes (`baseline_26node.csv`)

| fault | direct domain size | direct+indirect size | baseline direct% | baseline direct-or-indirect% |
|---|---|---|---|---|
| stuck | 6/26 | 6/26 | 23.1% | 23.1% |
| cable trapped | 0/26 | 11/26 | 0.0% | 42.3% |

Compare every result below against these baselines, not against 0%.

## Results: direct-or-indirect hit rate (%), federated/26-node

| fault | signal | hit rate | baseline |
|---|---|---|---|
| stuck | B | 94.0 | 23.1 |
| stuck | C | 94.0 | 23.1 |
| stuck | H | 85.9 | 23.1 |
| stuck | K | 87.1 | 23.1 |
| stuck | BK | **100.0** | 23.1 |
| stuck | CK | **100.0** | 23.1 |
| stuck | HK | 99.3 | 23.1 |
| cable trapped | B | 56.3 | 42.3 |
| cable trapped | C | 56.3 | 42.3 |
| cable trapped | H | 48.5 | 42.3 |
| cable trapped | K | 73.1 | 42.3 |
| cable trapped | BK | 77.7 | 42.3 |
| cable trapped | CK | 77.7 | 42.3 |
| cable trapped | HK | 74.6 | 42.3 |

Figure: `diagnosis_localization_federated_vs_centralized.png` (title
kept from when this ran alongside a centralized comparison; only the
federated panels are current).

## Why the numbers land where they do

**`stuck` localization is high (85.9-100%) mainly because the candidate
pool at 26 nodes is small and clean.** `top_nodes` shows every signal's
argmax for `stuck` concentrates overwhelmingly on
`wheel_status_current_ma_left/right` (e.g. B: 140/149=94%, BK/CK:
139/139=100%) — the battery/IR/mouse "environment"/other-group sensors
that a wider feature set exposes (and that a weaker signal can mis-fire
on) simply aren't in this 26-node pool. This is a **candidate-pool-
composition effect**: B/C's underlying "pick the single node with the
largest deviation, ignore graph structure" mechanism didn't get more
accurate, the set of things it could get wrong got smaller.

**`cable trapped` direct-or-indirect hit rate is strongest for K/BK/CK/HK**
(K=73.1%, HK=74.6%). `top_nodes` for `cable trapped` concentrates on
`imu_imu_angvel_x` and `odom_odo_lintw_y` (both indirect-domain members)
across every signal — angular-twist/linear-twist odometry, consistent
with `kinematics.py`'s stated mechanism (a trapped/slipping cable breaks
the wheel↔odometry kinematic relationship). K's strong direct-or-indirect
rate here (73.1%) despite 0% direct (structurally impossible at 26
nodes) suggests the forecast head, trained federated, does pick up the
wheel-velocity/angular-twist coupling that indirectly signals a trapped
cable.

**H is only average here (48.5%), not a standout.** Its own top node is
`odom_odo_orient_x` (orientation, not angular-twist) — its per-client
calibration statistics (`cov_head`'s `calib_mu`/`calib_cov_inv`) are now
computed per-client on federated-trained representations, and the target
node its contribution decomposition locks onto shifted under
federation.

**Caveat**: one seed, one run — training regime (federated FedAvg +
memory-alignment) and feature set (26 vs a wider set) both differ from
any earlier exploration at once, so this table is not decomposable into
"how much is federation vs how much is feature-set" without a controlled
re-run. Treat exact magnitudes (94% vs 100% vs 99.3% for B/BK/HK on
`stuck`) as indicative of one run, not separable from each other without
a multi-seed re-run.

## Open follow-up (not done in this run)

Per-client consistency of argmax attribution (does client A's local
model point at the same node as client B's, for the same fault type) was
flagged as an open question in `diagnosis_interpretability_review.md` §6
and remains untested here — this run pools all 5 clients' argmax votes
into one aggregate table; it does not report per-client agreement.

## Why B and C agree so strongly, and what C's attention structure actually buys (2026-08-19)

Follow-up mechanistic analysis (`benchmark/analyze_bck_mechanism.py`,
`analyze_bck_mechanism2.py`, `analyze_bck_scatter.py`, run on the same
federated 26-node pipeline, same seed) into the `stuck` / `cable trapped`
result above:

**B and C are almost the same quantity for these two fault types.** B is
`d_node[i] = ||z_i - p_i*||^2`, the raw squared deviation of node *i*
from its matched joint-prototype. C (`resid_struct[i] = ||d_i - d_hat_i||^2`)
subtracts off `d_hat_i`, the deviation that the learned top-k cross-node
attention predicts for node *i* from its neighbors' deviations. Over all
355 fault-window samples (both fault types, all 5 clients), raw B and
raw C scores correlate at Pearson r=0.994 (r=0.9999 for `cable trapped`,
r=0.989 for `stuck`), and their argmax (winning node) agrees on 74.9% of
samples — far above B/K (42.3%) or C/K (44.0%) agreement with the
forecast-residual signal K. The reason is structural, not coincidental:
the attention module's predicted deviation `d_hat` has norm only ~59% of
the raw deviation `||d||` on average (range 49.9% for `cable trapped`,
71.7% for `stuck`, per-client), so it is a partial correction, not a
reconstruction that can drive `resid_struct` toward zero. When a fault
inflates one node's deviation, its top-k attention neighbors — trained
only on non-fault dynamics — predict a *normal-regime* deviation
pattern, which cannot cancel a fault-specific spike; the residual after
subtraction stays large in essentially the same direction as the raw
deviation, so B and C track each other almost 1:1.

**C's learned cross-node structure is real but doesn't help localize
*these* particular faults.** Inspecting the `stuck`-client's learned
top-k attention graph directly: `wheel_status_current_ma_left` and
`wheel_status_current_ma_right` are each other's rank-1 attended
neighbor (mutual top-1, `wheel_pair_mutual_neighbor=True`) even though
this specific pair is not one of the hand-declared physics edges
(`prior_mask` is 0 for this pair in both directions) — the model learned
a sensible drivetrain-load coupling on its own, with the shared
prior-edge scalar (`prior_bias_strength≈1.32`) presumably reinforcing
*other* declared edges. But `stuck` and `cable trapped` both manifest as
a **correlated, shared** current-draw elevation across both wheel nodes
together (a systemic drivetrain load increase), not a single relation
breaking between an otherwise-normal pair. Because the anomaly appears
in the neighbor too, the neighbor's contribution to `d_hat` is *also*
elevated in the fault direction, so attention ends up partially
reconstructing the anomaly itself rather than isolating it — exactly the
failure mode the module docstring already anticipates ("V2/V21 wins on
faults that manifest as single-node value drift... [TypedRelation/C]
pays off specifically when a fault's failure mechanism matches a
declared relation breaking"). `stuck`/`cable trapped` are
single-node-value-drift-like (shared across a coupled pair, but not a
snapped relation), so C is structurally redundant with B here, not
complementary.

**K is the mechanistically distinct signal.** K (multi-step-ahead
forecast residual) correlates weakly with B (mean Pearson 0.193 pooled;
0.100 for `cable trapped`, 0.286 for `stuck`) and shares far fewer
argmax winners with either B or C (42-44% vs 74.9% for B-C). This is
expected: B/C measure instantaneous deviation-from-prototype magnitude,
while K measures whether a node's short-horizon trajectory is well-
predicted going forward — a node can have modest instantaneous deviation
but a hard-to-forecast (erratic/non-stationary) trajectory during a
fault, or vice versa. K's `BK`/`CK`/`HK` combination signals in the
table above outperform bare K precisely because they fall back to B/C's
more reliable node choice whenever K's own scalar confidence is not the
largest.

**Practical implication for signal selection**: for wheel-drivetrain
faults that manifest as shared/systemic deviation across mechanically
coupled nodes, B alone already captures nearly all of what C's
structural-residual computation offers — the extra attention machinery
earns its cost on faults that specifically break a single declared or
learnable relation while leaving the connected node normal, which
robo3er's `stuck`/`cable trapped` faults are not. This does not
generalize to Paderborn/voraus-AD relation-breaking faults, where C/H
are documented to be the stronger signals.

## BK vs CK: which combo is more principled for this fault set (2026-08-19)

Follow-up to the B/C/K mechanism section above, in response to a direct
question: since aggregate `direct_or_indirect_pct` for BK and CK were
literally tied (100.0%/100.0% on `stuck`, 77.7%/77.7% on `cable
trapped`), does that tie hide a difference at the sample level, and
which combination rule is more defensible given what each signal
actually measures? (`benchmark/analyze_bk_ck_overlap.py`, same run/seed
as above.)

**Sample-level: BK and CK are not just tied in aggregate, they pick the
same winning node on nearly every sample.** Comparing the two combo
rules per-sample (`combo = B/C's node if its z-score >= K's z-score,
else K's node`): `stuck` — 139/139 paired samples have
`argmax_BK == argmax_CK` (100%), both hit or both miss on every single
sample (0 samples where one succeeds and the other fails). `cable
trapped` — 178/197 (90.4%) have the same argmax; of the 19 where they
differ, BK gets exactly 1 uniquely right and CK gets exactly 1 uniquely
right (statistically a wash, not a real edge for either), with 43
samples both miss and 152 both hit. There is no sample population where
switching from BK to CK (or vice versa) meaningfully changes the outcome
for these two fault types.

**Why: because B and C are the redundant signal (see mechanism section
above), not because K dominates the pairing.** In fact K's own scalar
confidence wins the max-selection inside the combo rule only a
**minority** of the time — B beats K's scalar on 36.0% of `stuck`
samples and 13.2% of `cable trapped` samples; C beats K's scalar on
32.4% / 9.1% respectively (nearly identical fractions to B's, again
reflecting B≈C). So B/C do contribute the winning node on a real
fraction of samples, but whichever of B or C is in the combo contributes
*the same* node on those samples — because, per the mechanism section,
C's attention-corrected residual rarely disagrees with B's raw deviation
for these two current-draw-coupled faults.

**Answering "which combo is more reasonable."** B is a pure
self-comparison: it measures how far node *i*'s current encoding sits
from *node i's own* matched prototype, with zero cross-node information.
C and K are both literally GDN-style: `TrendGraphAttentionHead` (C) and
`ForecastHead` (K) share the identical top-k-by-learned-embedding-
similarity attention mechanism (declared physics edges enter as an
additive learned-strength bias in both), differing only in *what* they
attend over — C attends over other nodes' current deviations
`d_j = z_j - p_j*` to predict node *i*'s own deviation (instantaneous,
same-timestep consistency check); K attends over other nodes' current
raw-window encodings to forecast node *i*'s own future raw values
(temporal, cross-window). So: **C and K are both neighbor-attention-
based; B is not.**

That makes **BK the more mechanistically diverse pairing**: it combines
one signal with zero cross-node structure and one signal built entirely
from learned cross-node attention, so a shared blind spot in the
*attention topology itself* (e.g. an edge the model never learned to
attend to, or a neighbor pair whose learned coupling happens to be
wrong) can only hurt K's half of the combo, not both halves
simultaneously. CK instead pairs two signals that both depend on the
SAME class of learned attention mechanism (different weights, but the
same top-k-neighbor architecture and the same additive-physics-bias
term) — if that mechanism systematically mis-attends for a given node
(e.g., because the true causal neighbor isn't in its top-k set), both C
and K could be degraded by the same root cause at once, which is a less
robust ensembling property even where — as measured here — it does not
yet show up as an aggregate accuracy difference, because for
`stuck`/`cable trapped` (systemic/coupled-node faults where attention's
correction step barely moves the residual, per the mechanism section)
neither C nor K's attention pathway is being meaningfully exercised as a
discriminator anyway.

**Practical recommendation for this fault set**: prefer **BK** over CK —
it achieves the identical accuracy at lower cost (one fewer head's worth
of learned attention parameters duplicating information B already
carries) and is the mechanistically more diverse combination in
principle, even though the two are empirically indistinguishable on
`stuck`/`cable trapped` specifically. This should **not** be read as "C
is unnecessary in general" — the mechanism section's caveat still
applies: C/H are documented to be the stronger signals on relation-
breaking faults (Paderborn bearing, voraus-AD motor miscommutation)
where the neighbor relationship itself is the thing that breaks, a case
not represented in robo3er's fault set and not re-tested here.

See also `diagnosis_interpretability_review.md` (the planning doc this
test was requested from) and [[scoring-signals-B-C-E-H]] /
[[forecast-head-signal-k]] (detection-AUROC reference this extends).
