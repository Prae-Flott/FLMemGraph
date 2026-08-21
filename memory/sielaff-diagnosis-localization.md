---
name: sielaff-diagnosis-localization
description: Node-level root-cause localization accuracy (not detection AUROC) for B/C/H/K/BK/CK/HK on Sielaff, federated, against the CORRECT fault-type ground truth -- the 47 company/domain-expert RED-severity error IDs. Two confounds found (transaction_throughput, then receipt_count/reject_count/cleaning_duration); domain-size-weighted random baseline is ~48.8%, above every signal's number. Dropping 22 hardware-heterogeneous columns made K/BK/CK worse (ruled out). Activity-level residualization (regressing d_node/resid_struct/k_resid on a window-activity covariate before z-scoring) tested next: helped C substantially (13.4->23.6% confound-excluded) via a genuine domain-relevant feature shift, left K/H unchanged, only partly helped B (residual confound just moved from receipt_count to reject_count) -- real progress, still below the random baseline in aggregate, not yet a full fix.
metadata:
  type: project
---

# Sielaff diagnosis localization — federated, RED-severity error IDs (corrected 2026-08-19)

Second dataset in the requested B/C/K/BK/CK classification-vs-diagnosis
comparison (robo3er: [[robo3er-diagnosis-localization]]; Paderborn:
[[paderborn-diagnosis-localization]]). **Supersedes a retracted earlier
version of this doc** that used `data/sielaff/`'s coarse
`dominant_error_group_in_window` 5-class scheme as if those were the
real fault types. Per [[sielaff-red-fault-federated]] (established
2026-08-13) and explicit user correction: **every one of the 47
company/domain-expert RED-severity error IDs in
`data/sielaff_data/sielaff_ground_truth.md` is a fault type** — they cut
across the 5 coarse groups, not reducible to them. This run targets
`data/sielaff_red/` (built by `benchmark/datasets/build_sielaff_red.py`)
and its per-window `window_red_ids.json`/`red_id_names` labels instead.

Script: `benchmark/diagnose_sielaff_red_localization_federated.py`.
`JointPrototypeV21Forecast` (no declared physics edges), 10 real
machines as 10 federated clients, memory-only exchange, `horizon_mult=10`,
same 8-bucket/4h window as the red AUROC script. **Multi-label windows**:
a window can contain up to 23 concurrent red IDs; each window's single
argmax vote is attributed to EVERY red ID it contains, per
`analyze_sielaff_red_feature_attribution.py`'s established convention.
**Pooled calibration reference**: per-client calib+test_normal is only
~6-9 windows (too thin for a stable per-node IQR), so the z-scoring
median/IQR is pooled across all 10 clients (140 windows total) while
each red window is still scored by its OWN client's own trained model —
same deliberate, documented departure from per-client-calibrates-locally
that the B-only attribution script already established.

## Ground truth

Reuses `feature_domain()` / `FAULT_EXPECTED_DOMAIN` /
`NO_SENSOR_KEYWORDS` from `analyze_sielaff_red_feature_attribution.py`
verbatim. Of the 47 red IDs, **34 (72%) have NO directly-instrumented
sensor** in this 39-column set (compactor door/flap, crate handling,
safety circuit, misc restart/reboot — matches that script's own
established finding) and are excluded from scoring; **13 are `mapped`**
(have an expected sensor-domain rule) and are what the tables below
score.

## A dominant confound required a methodology fix mid-run

Before adding confound handling, RAW argmax was found to be dominated by
`total_weight` (a `transaction_throughput` feature) on **every single
mapped red ID**, regardless of that ID's true expected domain — e.g.
`status_bottle_collision` (expects `ring_camera_read`/
`bottle_recognition_sensor`) had `total_weight` win 599/724 votes for B.
This is the exact "activity/exposure confound"
`analyze_sielaff_red_feature_attribution.py` already documented (a
genuinely `no_error` window is disproportionately an idle window with
zero throughput, so ANY window with real transaction activity looks
anomalous on throughput features regardless of which fault occurred).
Added a confound-excluded variant (masks `transaction_throughput` out of
the argmax candidate pool, on top of the existing reliability mask) —
both are reported below, not one discarded in favor of the other, since
excluding the confound also removes GENUINE signal for the 2 red IDs
whose true expected domain legitimately includes throughput
(`status_crate_bottle_wrong`, `status_bottle_compacted`).

## Results: aggregate direct% across the 13 mapped red IDs, pooled votes

| signal | raw (confound-included) | confound-excluded |
|---|---|---|
| B | 10.7 | 15.7 |
| C | 11.2 | 13.4 |
| H | 11.2 | 15.0 |
| **K** | 15.6 | **26.3** |
| BK | 15.0 | 26.0 |
| CK | 15.1 | 25.9 |
| HK | 11.0 | 18.3 |

(n=3174 votes raw / for B,C,H; n=2916 for K-derived signals — fewer
because K needs a valid next-window pair, a handful of orphan red
windows at split/machine boundaries are dropped, matching every other
K-signal script's convention.)

**K has the highest raw aggregate number** (15.6% raw, 26.3%
confound-excluded — nominally the highest of the 7 signals), and BK/CK
track just behind it (never above it, consistent with the combo-routing
finding already established for robo3er/Paderborn). **But do not read
26.3% as "K localizes well" without the random-baseline correction
below — the vote-weighted random-guess baseline across these 13 IDs is
~48.8%, well above every signal's aggregate number.** B/C/H cluster
together, confound-excluded around 13-16%, essentially indistinguishable
from each other and also below that baseline.

## Per-red-ID detail: the confound exclusion changes which faults look "diagnosable" in a specific, mechanistically legible way

| red ID | name | n | B raw/excl | K raw/excl | note |
|---|---|---|---|---|---|
| 308 | status_crate_bottle_wrong | 239 | 93.3/56.9 | 82.1/28.8 | throughput IS its true domain — excluding it removes real signal, raw is more appropriate here |
| 235 | status_bottle_compacted | 85 | 96.5/0.0 | 88.0/18.7 | same as above, more extreme |
| 250 | action_cleaning_start | 118 | 6.8/86.4 | 0.0/3.6 | throughput was PURE confound here — B jumps to 86.4% once excluded, landing correctly on `cleaning_duration`/`cleaning_begin_pixels` |
| 252 | action_cleaning_daily | 106 | 3.8/84.9 | 0.0/4.0 | same pattern |
| 801 | status_printer_ok | 389 | 4.1/33.4 | 8.7/45.8 | confound exclusion roughly triples/quintuples the hit rate |
| 806 | status_receipt_retracted | 26 | 0.0/84.6 | 0.0/50.0 | B goes from 0% to 84.6% |
| 403 | status_bottle_collision | 724 | 0.6/0.7 | 12.0/33.5 | B stays near-zero either way; K is the only signal that finds real signal here |
| 203 | status_bottle_out_of_range | 513 | 0.0/0.0 | 12.3/38.4 | same pattern as above |
| 202 | status_bottle_direction_wrong | 248 | 0.0/0.0 | 14.4/36.9 | same pattern |

Full per-red-ID table (all 13 mapped + 34 no-direct-sensor IDs, both
variants, top-3 nodes per signal) in
`checkpoints/sielaff/sielaff_red_diagnosis_localization_federated_h10.json`.

**Finding**: the confound exclusion is not a uniform boost — it reveals
that Sielaff's red faults split into two groups. The `ring_camera_read`/
`bottle_recognition_sensor`-domain faults (`status_bottle_collision`,
`status_bottle_out_of_range`, `status_bottle_direction_wrong`,
`status_backward_label_movement`, `status_slow_label_movement`,
`status_last_ls_passed`) are **only ever found by K** (12-14% raw, up to
38% excluded) — B/C/H score ~0% on them both raw and excluded. The
`cleaning_cycle`/`receipt_printing`-domain faults (`action_cleaning_*`,
`status_printer_ok`, `status_receipt_retracted`) are the OPPOSITE — B/C/H
jump to 84-100% once the confound is removed, while K stays weak
(0-50%) there.

**Correction (added after this doc's first pass): the "K finds real
signal" framing above needs a random-baseline caveat, not just a
before/after comparison.** Each fault's expected domain has a different
SIZE (out of the 34-node confound-excluded pool): the `bottle_*` group's
combined domain (`ring_camera_read`+`bottle_recognition_sensor`+
`sorting_gate_switch`) is 27/34 nodes = a **79.4% random-guess
baseline**; `label_movement`'s domain (`ring_camera_read` alone) is
12/34 = 35.3%; `receipt_printing` is 1/34 = 2.9%; `cleaning_cycle` is
4/34 = 11.8%; `reject_stream` (crate_bottle_wrong's domain once
throughput is excluded) is 2/34 = 5.9%; `sorting_gate_switch` alone
(bottle_compacted) is 14/34 = 41.2%. Vote-count-weighted across all 13
mapped IDs, the random-guess baseline for the WHOLE aggregate metric is
**~48.8%** — well ABOVE the reported 26.3% aggregate for K. So in
aggregate, K's argmax is actually landing on the correct domain LESS
often than chance would, once domain size is accounted for — the
"K finds real signal on bottle_* faults" framing is not correctly
supported by these numbers (33-38% against a 79.4% baseline is a
below-chance result, not a finding); **only the small-domain IDs
(receipt_printing, cleaning_cycle, reject_stream) show genuinely
above-chance hit rates**, and per the top-nodes evidence below, that's
substantially because their expected domain happens to overlap with
`receipt_count`/`reject_count`/`cleaning_duration` — three near-generic
"was the machine active" proxy features that dominate argmax almost
everywhere once `transaction_throughput` is excluded (same activity
confound, one level down — verified via `top_nodes_confound_excluded`,
e.g. `status_bottle_collision`'s B argmax is `receipt_count`(358)/
`reject_count`(284) in >88% of votes, neither being that fault's actual
domain). **Net assessment: this experiment has NOT demonstrated
trustworthy node-level fault localization for Sielaff-red** — most of
what looks like signal is a second-order activity-proxy confound that
the first exclusion pass didn't catch, not real per-fault mechanism
discrimination.

## Follow-up (2026-08-19): does dropping the 22 hardware-heterogeneous columns help? Tested — it makes things WORSE

Per-machine zero-fraction check (not assumed) found `read SM/barcode_RingCamera *`
(12 cols) and `schließen/öffnen_Weiche 3-7` (10 cols) are permanently 0.0
for most of the 10 real machines because those machines simply lack that
hardware (only 1/10 has any RingCamera reading; only 3-4/10 have gates
beyond the first two) — not event sparsity, genuine hardware
heterogeneity across federated clients, structurally worse than
robo3er's near-constant status flags (which at least varied on every
robot). Proposed fix, tested via `--drop-heterogeneous-hardware`
(17-node set, keeping only near-universal columns):

| signal | 39-node (all columns) | 17-node (22 heterogeneous cols dropped) |
|---|---|---|
| B | 15.7 | 15.0 |
| C | 13.4 | 12.4 |
| H | 15.0 | 14.5 |
| **K** | **26.3** | 13.7 |
| BK | 26.0 | 14.0 |
| CK | 25.9 | 13.7 |
| HK | 18.3 | 16.7 |

**Result: dropping these columns roughly HALVES K/BK/CK, barely moves
B/C/H.** Cause (verified via `top_nodes_confound_excluded`): the removed
columns (`öffnen_Weiche 2`'s neighbors, ring-camera reads) were carrying
REAL signal for the few equipped machines on exactly the
`ring_camera_read`/`sorting_gate_switch`-domain fault types — with them
gone, argmax on those fault types falls through completely to the same
`receipt_count`/`reject_count`/`cleaning_duration` confound documented
above, since nothing else is left to compete with it. **A blanket
column drop is the wrong fix here: it throws away a minority of
clients' real signal without addressing the actual dominant problem
(the activity-proxy confound persists with or without these columns).**

**Answering the "is per-client node masking worth building" question
this test was meant to inform: no, not on this evidence.** The
hypothesis was that these columns hurt by injecting cross-client
calibration noise (pooling a few real readings with a majority of
permanent zeros); the blanket-drop test shows the opposite — removing
them hurts, because their problem was never really "noise from dead
channels," it's that a DIFFERENT, still-unaddressed confound
(receipt_count/reject_count/cleaning_duration) wins in their absence
regardless. Building the more invasive per-client masking machinery is
not justified before that confound is dealt with — likely candidate next
step is proper per-window ACTIVITY-LEVEL normalization, not further
feature-set surgery (see below — implemented and tested as a follow-up).
Script default reverted to keep all 39 columns
(`--drop-heterogeneous-hardware` kept as an opt-in flag for reproducing
this negative finding).

## Follow-up (2026-08-19): activity-level residualization — real but partial improvement

Implemented the normalization proposed above: for each node, fit
`signal ~ a*activity_level + b` by OLS on pooled calib data (`activity_level`
= sum over the window of `journal_count + reject_count + cleaning_count +
receipt_count`, i.e. total logged-event count), then score the RESIDUAL
instead of the raw `d_node`/`resid_struct`/`k_resid` — the same
"explained-by-formula, score what's left" idea `kinematics.py` uses for
robo3er's wheel/odometry residual, applied post-hoc (no retraining
needed) rather than pre-hoc on raw features. **Scoped to B/C/K only** —
H's Mahalanobis mechanism has its own, different, covariance-based
defense against correlated confounds (inverse-covariance reweighting
already suppresses shared-variance directions) and was left on raw
`d_node` as a comparison point. Default in
`benchmark/diagnose_sielaff_red_localization_federated.py`
(`--no-activity-normalize` to disable).

| signal | before (13.4-26.3 baseline) | after activity-normalization |
|---|---|---|
| B | 15.7 | 17.5 |
| **C** | 13.4 | **23.6** |
| H (unchanged by design) | 15.0 | 15.0 |
| K | 26.3 | 26.3 |
| BK | 26.0 | 26.0 |
| CK | 25.9 | 26.2 |
| HK | 18.3 | 18.3 |

**C improved substantially and for a mechanistically real reason, not
just numerically.** `top_nodes_confound_excluded` on the `bottle_*`
faults shows C's argmax shifting from pure activity-count features
(`journal_count`/`total_weight`) to `schließen_Weiche 6` appearing as a
top pick (182/724 votes on `status_bottle_collision`, up from ~0) —
`schließen_Weiche 6` (gate 6) is genuinely in the `sorting_gate_switch`
domain these faults expect, so this is real signal recovery, not noise.
(Gate 6 is one of the "hardware-heterogeneous" columns from the section
above, present on only 3-4/10 machines — this is independent evidence
those columns carry real signal worth keeping, consistent with that
section's finding.)

**B only partially improved, and K/H are unaffected.** B's dominant
argmax feature shifted from `receipt_count` to `reject_count` (still not
in these faults' expected domain) rather than clearing — the linear
correction removed some but not all of the shared-activity component.
K's numbers are essentially IDENTICAL before/after, meaning K's own
argmax pattern wasn't primarily driven by this particular activity
proxy to begin with (consistent with K already using a different top
feature, e.g. `öffnen_Weiche 2`, per the confound section above).

**Net assessment: real, mechanistically-verified progress (especially
for C), but not a complete fix.** C's 23.6% is still below the
domain-size-weighted random baseline (~48.8%) computed earlier in this
doc, and B's residual `reject_count` dominance is unexplained. A likely
reason: `reject_count` is itself one of the 4 columns summed into
`activity_level`, so regressing a feature against a proxy that includes
itself removes mostly its OWN linear trend, not necessarily whatever
non-linear or genuinely-independent large-magnitude variance makes it
keep winning argmax — a **leave-one-out activity proxy** (exclude the
target node's own column from its own regressor) is the natural next
refinement, not implemented here. Also untested: whether a log/rank
transform of the (right-skewed, mostly-zero) count features before OLS
would fit the shared-activity trend better than raw linear regression.

## Caveats

- Single seed, no repeats.
- 34/47 red IDs (72%) have no expected-domain rule at all — this dataset
  has the LOWEST fraction of directly-diagnosable fault types of the 3
  datasets tested (robo3er: 2/2 fault types have a domain, at least
  indirectly; Paderborn: 3/3; Sielaff-red: 13/47).
- The confound-exclusion domain (`transaction_throughput`) was defined
  post-hoc after observing it dominate the raw results — this is the
  same domain `analyze_sielaff_red_feature_attribution.py` already
  flagged, not a new discovery, but the specific exclusion was applied
  here for the first time in a B/C/H/K-level (not just B) analysis.
- Multi-label vote attribution (one window's argmax counted toward every
  co-occurring red ID) means highly-co-occurring IDs' numbers are not
  independent samples — a real property of this fault data (4h windows
  routinely contain many simultaneous event types), not a scoring bug.

See also `diagnosis_interpretability_review.md`,
[[sielaff-red-fault-federated]] (why red-severity IDs are the correct
fault-type ground truth), and `analyze_sielaff_red_feature_attribution.py`
(the domain mapping and confound this reuses/extends).
