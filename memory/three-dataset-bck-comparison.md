---
name: three-dataset-bck-comparison
description: Cross-dataset comparison of B/C/K/BK/CK -- DETECTION AUROC and DIAGNOSIS/localization accuracy, federated, all 3 datasets (Sielaff's diagnosis is against the CORRECTED red-severity-error-ID fault-type ground truth). Headline finding: detection and diagnosis rank signals differently, sometimes OPPOSITELY, on the same dataset (Paderborn), because max-aggregation detection only needs SOME node elevated while argmax diagnosis needs THE CORRECT node elevated. Includes the diagnosis-correctness scoring methodology (hand-built domain maps, not learned labels) and a 5-factor structural explanation for why Sielaff's diagnosis ceiling is far lower than robo3er's/Paderborn's (event-log vs. continuous sensor data, fault-domain coverage, confound separability, client hardware heterogeneity, multi-label ambiguity).
metadata:
  type: project
---

# B / C / K / BK / CK across 3 datasets: detection AUROC vs. diagnosis accuracy

Requested comparison: run the same 5-signal set (B=node_max, C=struct_max,
K=forecast_max, BK=max(B,K), CK=max(C,K)) on 3 datasets, federated per
`benchmark-policy-federated-only.md`, for BOTH classification (detection
AUROC, does the score separate normal from fault) and diagnosis (does
argmax point at the correct node).

## Methodology: how a diagnosis is scored "correct" (all 3 datasets)

No dataset has a literal per-sample "which sensor was the true root
cause" label — diagnosis correctness is judged against a HAND-BUILT
domain map, not a learned or dataset-native ground truth:

1. Each signal already produces a per-node score array (`d_node` for B,
   `resid_struct` for C, `k_resid` for K; H's Hotelling-T² contribution
   decomposition, new this investigation). Take `argmax` over the
   reliable (and, where applicable, confound-excluded) candidate nodes
   → one feature NAME per scored window. Combo signals (BK/CK/HK) don't
   get their own mechanism — per-window, whichever component's own
   z-scored scalar is larger supplies its own argmax node (pure
   routing, no new information).
2. Look up that feature name in a per-dataset, per-fault-type domain map
   built from physical/mechanical domain knowledge (robo3er:
   `kinematics.py`'s differential-drive physics — `stuck`→any `wheel_*`
   column, `cable trapped`→`slip_status_is_slipping` direct +
   wheel/odometry kinematic-residual columns indirect; Paderborn:
   bearing-fault literature — `vibration_1` direct, `phase_current_*`
   indirect, identical for all 3 damage categories since the 6-channel
   set can't resolve damage location; Sielaff: `feature_domain()` groups
   the 39 raw columns into physical sensor families,
   `FAULT_EXPECTED_DOMAIN` maps each red-ID's name keywords to an
   expected family, reusing `analyze_sielaff_red_feature_attribution.py`'s
   already-established mapping).
3. Classify the argmax node as `direct` (in the fault's expected direct
   domain), `indirect` (a plausible secondary domain), or `other` (miss).
   Aggregate hit rate = direct (or direct+indirect) count / total scored
   samples — for Sielaff specifically, one window's single argmax vote
   is counted toward EVERY red ID that window contains (windows can
   list up to 23 concurrent IDs), not deduplicated.

This is a domain-knowledge PROXY for correctness, not verified
ground truth — and domain sizes vary hugely across fault types (as
little as 1/34 candidate nodes, as much as 27/34 on Sielaff), so a raw
hit-rate number is not comparable across fault types or datasets without
a domain-size-weighted random-guess baseline computed the same way (see
the Sielaff row's ~48.8% baseline below).

## Part 1 — Detection AUROC (federated, existing checkpoint reports)

| dataset | B | C | K | BK | CK | source |
|---|---|---|---|---|---|---|
| robo3er (26-node, encdec-synced, h10) | 0.948 | 0.958 | **0.9996** | 0.995 | **0.9999** | `robo3er_forecast_v2_federated_h10_encdec_synced_report.json` |
| Paderborn (encdec-synced, h10) | 0.913 | **0.976** | 0.883 | 0.910 | 0.960 | `paderborn_forecast_v2_federated_h10_encdec_synced_report.json` |
| Sielaff (h10) | 0.974 | **0.975** | 0.910 | **0.976** | 0.974 | `sielaff_forecast_v2_federated_h10_report.json` |

**Caveat on Sielaff's detection-AUROC fault-type labels**: this
particular AUROC breaks down by `data/sielaff/`'s coarse
`dominant_error_group_in_window` scheme, not the red-severity IDs used
for diagnosis below (Part 2) — it has not been re-run against the
red-severity split, so it isn't necessarily comparable label-for-label
to Part 2's Sielaff row. Not withdrawn, just flagged.

**Detection-AUROC takeaway**: C is consistently strong (0.958-0.976,
never the worst). K is the standout on robo3er specifically (0.9996,
essentially perfect — 26-node federated is a different regime from the
0.80-0.90 centralized K numbers in [[forecast-head-signal-k]], see that
file's own caveat about the confound between federation and the 26-node
feature reduction) but the weakest single signal on Paderborn (0.883,
below even B). BK/CK never clearly beat max(B,C) alone on any dataset —
consistent with [[forecast-head-signal-k]]'s original combination-rules
finding that pairing K with B/C rarely helps detection ranking, now
confirmed to hold across all 3 datasets, not just robo3er.

## Part 2 — Diagnosis: node-level localization hit rate

| dataset | B | C | K | BK | CK | fault types averaged | ground truth |
|---|---|---|---|---|---|---|---|
| robo3er (26-node) | 75.2 | 75.2 | 80.1 | **88.9** | **88.9** | stuck, cable trapped | direct-or-indirect%, real 2-class dataset labels |
| Paderborn (6-node) | **93.8** | 50.9 | 50.6 | 63.0 | 48.2 | outer_ring, inner_ring, combined | direct-or-indirect%, real damage-code labels |
| Sielaff (39-node, RED error IDs) | 17.5 | 23.6 | **26.3** | 26.0 | 26.2 | 13 of 47 red-severity error IDs with a defined sensor domain (34/47 have none) | direct%, confound-excluded + activity-normalized, company/domain-expert red-severity IDs |

Full per-fault-type breakdown, ground-truth domain definitions, and
`top_nodes` evidence for each cell: [[robo3er-diagnosis-localization]],
[[paderborn-diagnosis-localization]], [[sielaff-diagnosis-localization]].

**Sielaff's diagnosis number here is the CORRECTED red-severity-ID run**
(2026-08-19), reusing `analyze_sielaff_red_feature_attribution.py`'s
already-established fault-type ground truth (the 47 company/
domain-expert RED-severity error IDs), not the earlier retracted version
that used `data/sielaff/`'s coarse 5-group scheme.

**Do not read Sielaff's numbers as comparable to robo3er's/Paderborn's
without a caveat**: [[sielaff-diagnosis-localization]] found the
vote-weighted RANDOM-GUESS baseline across Sielaff's 13 diagnosable red
IDs is ~48.8% (their expected domains vary hugely in size, from 1 to 27
of 34 candidate nodes) — every signal's number in this row is still
BELOW that baseline, even after two fixes. robo3er's and Paderborn's
numbers don't have this problem (their domains are close to uniform
size across fault types), so this row is not on equal footing with the
other two and should be read as "improving but still likely partly
confounded," not "Sielaff's weakest diagnosis dataset." Two follow-ups
tested: (1) dropping 22 hardware-heterogeneous columns made K/BK/CK
WORSE, ruling out hardware heterogeneity as the fix; (2) activity-level
residualization (regressing B/C/K's node-level deviations on a
window-activity covariate before z-scoring, H left unchanged) genuinely
helped C (13.4→23.6, verified via a real domain-relevant argmax shift
onto `schließen_Weiche 6`) but only partly helped B (residual confound
moved from `receipt_count` to `reject_count`, not cleared) and didn't
move K at all — see that doc's full writeup for the mechanism and the
proposed next refinement (leave-one-out activity proxy).

## The headline cross-dataset finding: detection and diagnosis rank signals DIFFERENTLY, sometimes OPPOSITELY, on the SAME dataset

- **Paderborn inverts outright.** Detection AUROC ranks
  C(0.976) > B(0.913) > BK(0.910) > K(0.883); diagnosis ranks
  B(93.8) ≫ BK(63.0) > C(50.9) ≈ K(50.6) > CK(48.2). The BEST detector
  (C) is the WORST-but-one localizer, and the worst-but-one detector (B)
  is by far the best localizer. Mechanism: C's learned cross-node
  attention correction is exactly what makes it a strong detector for
  Paderborn's relation-sensitive bearing faults, but that same
  correction step dilutes WHICH node looks most anomalous, scattering
  its argmax onto `force`/`speed` instead of the true `vibration_1`
  channel — see [[paderborn-diagnosis-localization]] for the mechanism.
- **Sielaff's diagnosis numbers are improving but still not trustworthy
  enough to draw a detection-vs-diagnosis comparison from** (see the
  baseline caveat above) — after activity-normalization, C actually
  moved OFF the bottom (23.6, no longer the weakest) and K nominally
  leads (26.3) despite merely mid-pack detection (0.910) — but since
  every signal's number here still sits below the ~48.8% random
  baseline, none of these differences should be read as demonstrated
  skill yet. Held here as an open question, not a finding, pending
  further refinement (leave-one-out activity proxy is the next candidate
  — see [[sielaff-diagnosis-localization]]).
- **robo3er is the one dataset where the two tasks roughly agree**: K is
  the best single detector AND ties for best diagnosis contributor
  (via BK/CK, 88.9), because [[robo3er-diagnosis-localization]]'s
  mechanism section already established H and K are the genuinely
  distinct signals there, not B/C.

**Practical implication**: the two tasks must be validated and
potentially SELECTED separately per dataset. `scoring-signals-B-C-E-H.md`'s
per-dataset detection recommendations (Paderborn→max(C,E,H);
robo3er→H or max(B,C,H); Sielaff→max(B,C)) should NOT be assumed to also
be the right signal to read off for root-cause localization — on
Paderborn specifically, the detection recommendation (C-heavy) is
actively the wrong thing to read for diagnosis (**use B's own argmax for
Paderborn localization**). Sielaff's equivalent recommendation is
withheld until the activity-proxy confound is resolved (see above) —
current numbers don't support recommending any signal's argmax there yet.

## Why detection AUROC and diagnosis accuracy diverge at all: max() aggregation vs. argmax specificity

General mechanism behind the headline finding above, not specific to any
one dataset:

- **Detection only needs SOME node to be elevated; diagnosis needs THE
  CORRECT node to be elevated.** `B = max_i zscore(d_node_i)` is
  logically an OR across all N nodes — any single node correlating with
  the fault label is enough to separate fault from normal windows.
  Diagnosis (`argmax_i`) is a strictly harder AND: the winning node must
  be BOTH elevated AND mechanistically correct for that specific fault.
- **A confound that's generic across fault types is a detection ASSET
  and a diagnosis LIABILITY, simultaneously, for the same underlying
  reason.** Sielaff's activity-proxy features (`receipt_count`/
  `reject_count`/`total_weight`/etc.) spike during essentially every red
  fault, because faults occur during active operation while calibration
  windows are disproportionately idle — that correlation with the
  binary fault/normal label is real and helps `max()`-based detection
  (fault windows really are more "active" than calib windows, on
  average). The SAME correlation is fault-non-specific by construction
  (it fires for every fault alike), so it dominates `argmax` and drowns
  out whatever node-specific signal exists — helping the exact metric
  (detection) that doesn't care WHICH node fired, hurting the exact
  metric (diagnosis) that does.
- **Paderborn shows the identical mechanism from a different angle,
  without any activity confound at all.** C's learned cross-node
  attention correction is what makes it Paderborn's best detector
  (0.976 AUROC) — it captures the relational structure bearing faults
  break. But that same correction step, by construction, blends
  neighbor information INTO each node's residual, diluting which single
  node looks most anomalous and scattering `argmax` onto `force`/
  `speed` instead of the true `vibration_1` channel. A mechanism that
  makes detection more robust (using more information) makes diagnosis
  less precise (mixing information across nodes) — the same trade-off
  as the activity confound, arrived at through model architecture
  instead of data structure.
- **Practical rule of thumb this project should carry forward**: a
  strong detection AUROC for a signal is NOT evidence that its `argmax`
  is trustworthy for root-cause localization, and the mechanisms that
  make a signal a good detector (broad correlation with the fault
  label, cross-node information pooling) are often exactly what make it
  a poor localizer. The two properties must be measured and selected
  independently per dataset per signal, never inferred from each other.

## BK/CK never clearly beat the better of their two inputs, on either task, on any of the 3 datasets

Confirms [[robo3er-diagnosis-localization]]'s combo-routing finding
generalizes on detection AUROC (BK/CK sit between B/C's and K's own,
never a clear win over max(B,C) alone, on all 3 datasets) and on
Paderborn's diagnosis specifically (BK=63.0 between B=93.8 and K=50.6).
Sielaff's diagnosis numbers (BK=26.0, CK=25.9, just under K's 26.3) show
the same routing PATTERN but are subject to the same below-baseline
caveat as the rest of that row — not treated as a confirmed instance
here. No dataset in this comparison shows BK/CK clearly falling below
BOTH of their inputs (the sharper Sielaff counter-example reported in an
earlier retracted version of this comparison does not reproduce against
the corrected red-ID ground truth, though that version had its own
unresolved confound issues too).

## Why Sielaff's diagnosis task is structurally harder than robo3er's, not just weaker signals

robo3er's diagnosis hit rate (75.2-88.9%) vs. Sielaff's (17.5-26.3%,
BELOW the ~48.8% random baseline) is not primarily a signal-quality gap
— five structural differences between the two datasets stack up:

1. **Data nature**: robo3er's nodes are genuinely continuous physical
   telemetry (wheel velocity, IMU, odometry) — every node carries
   continuous, physically meaningful variation every window. Sielaff's
   39 columns are discrete event logs (journal/receipt/reject/cleaning/
   statistic_values tables) artificially bucketized into pseudo-windows
   — most buckets are idle (many count features are 0 in 43-99% of
   buckets), so real activity makes many features jump from 0 to
   nonzero TOGETHER, creating the activity-proxy confound that has no
   robo3er analogue.
2. **Fault-type domain coverage**: robo3er's fault taxonomy is 2/2
   diagnosable (both `stuck` and `cable trapped` have a defined direct
   or indirect domain); Paderborn 3/3. Sielaff is 13/47 (28%) — 34 of 47
   red IDs (compactor doors/flaps, crate handling, safety circuit) have
   NO instrumented sensor in this 39-column set at all, a coverage gap
   robo3er never has.
3. **Whether the confound can be cleanly excised**: robo3er's 68→26-node
   reduction ([[robo3er-diagnosis-localization]]) is the direct
   precedent this was tested against — dropping battery/IR/mouse nodes
   worked because they were genuinely irrelevant to the fault mechanism,
   and diagnosis jumped from 2.7-41% (68-node) to 85.9-100% (26-node).
   The analogous move on Sielaff (dropping the 22 hardware-heterogeneous
   columns) was TESTED and made K/BK/CK WORSE — because those columns
   carried real signal for the few equipped machines, and because 2 of
   Sielaff's 13 diagnosable red IDs (`status_crate_bottle_wrong`,
   `status_bottle_compacted`) have a true expected domain that legitimately
   OVERLAPS the confound (`transaction_throughput`/`reject_stream`).
   robo3er's confound was cleanly separable noise; Sielaff's is entangled
   with real signal for a meaningful fraction of fault types.
4. **Federated client hardware homogeneity**: robo3er's 5 clients are the
   same iRobot Create3-class platform, identical 68-channel sensor set.
   Sielaff's 10 real machines have DIFFERENT hardware configurations
   (only 1/10 has any RingCamera reading, only 3-4/10 have more than 2
   sorting gates, verified via per-machine zero-fraction) — a genuine
   federated heterogeneity problem robo3er never faces, on top of the
   activity confound.
5. **Label granularity/ambiguity**: robo3er windows are single-labeled
   (exactly one of `stuck`/`cable trapped`/normal). Sielaff's 4h windows
   can contain UP TO 23 concurrent red IDs — even a correct `argmax`
   can't be unambiguously credited to one specific co-occurring fault,
   an irreducible scoring ambiguity independent of model quality.

None of these five are fixable by a better signal or a smarter combination
rule — they are properties of how the Sielaff dataset itself is
constructed. This reframes the earlier "K/C underperform on Sielaff"
framing: the ceiling itself is lower here, for reasons upstream of any
B/C/H/K/BK/CK choice.

## What's still open

- **Sielaff diagnosis still needs further work before its numbers are
  trustworthy** — activity-level residualization (implemented) helped C
  substantially but every signal's aggregate still sits below the
  domain-size-weighted random baseline (~48.8%); B's residual confound
  moved from `receipt_count` to `reject_count` rather than clearing. A
  leave-one-out activity proxy (excluding each node's own column from
  its own regressor) is the flagged next refinement. See
  [[sielaff-diagnosis-localization]].
- H and HK were computed in every underlying run (needed for the combo
  routing logic) but are outside this comparison's requested 5-signal
  scope — see each per-dataset doc for the full B/C/H/K/BK/CK/HK table.
- Single seed per dataset, no repeats — same caveat every underlying doc
  already carries.
- Paderborn's ground-truth domain can't distinguish outer- vs
  inner-ring damage location (6-node set only has one vibration
  channel) — its 3 "fault types" all share literally the same domain
  definition, unlike robo3er's/Sielaff's genuinely distinct fault
  mechanisms.
- Sielaff's detection AUROC (Part 1) has not been re-run against the
  red-severity-ID split used for diagnosis (Part 2) — the two Sielaff
  rows use different label schemes, flagged above, not yet reconciled.
- 34/47 Sielaff red IDs (72%) have no expected sensor domain at all —
  the highest fraction of structurally-undiagnosable fault types of the
  3 datasets.

See also `diagnosis_interpretability_review.md` (the original
methodology this whole line of investigation implements) and
[[scoring-signals-B-C-E-H]] (this project's primary detection-AUROC
reference).
