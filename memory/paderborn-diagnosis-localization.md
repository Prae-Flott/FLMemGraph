---
name: paderborn-diagnosis-localization
description: Node-level root-cause localization accuracy (not detection AUROC) for B/C/H/K/BK/CK/HK on Paderborn, federated, 6-node channel set. B is the clear localization winner (vibration_1), unlike detection AUROC where C/E lead.
metadata:
  type: project
---

# Paderborn diagnosis localization — federated, 6-node

Second dataset in the requested B/C/K/BK/CK classification-vs-diagnosis
comparison (robo3er: [[robo3er-diagnosis-localization]]; Sielaff's
diagnosis run is retracted — wrong fault-type label scheme, see
[[three-dataset-bck-comparison]]'s retraction note — and still pending a
redo against the red-severity error IDs). Federated per
`benchmark-policy-federated-only.md`: K001-K006 healthy bearings as 6
clients, `JointPrototypeV31Forecast` with the 8 declared torque/speed/
force↔vibration/current physics edges, FedAvg'd encoder/decoder,
`horizon_mult=10`. Every damaged bearing code scored against every
client's personalized model (same cross-client convention as
`run_paderborn_forecast_v2_federated.py`'s detection-AUROC report), at
WINDOW level (not that script's per-file averaging, which would blur the
node-level array diagnosis needs). Script:
`benchmark/diagnose_paderborn_localization_federated.py`.

Only 6 nodes exist here (`vibration_1`, `phase_current_1`,
`phase_current_2`, `force`, `speed`, `torque`) — a structurally easier
localization task than robo3er/Sielaff (random-guess baseline
1/6=16.7% direct).

## Ground truth

All three damage categories (outer_ring/inner_ring/combined) share the
SAME domain — this 6-channel set cannot distinguish damage *location*,
only that vibration is the affected *channel* (per
[[robo3er-explicit-physics-scoring]]'s Paderborn section: "bearing
faults manifest as impulsive vibration... does NOT change the mean-value
relationships between torque, speed, and current"):

- direct = `{vibration_1}`
- indirect = `{phase_current_1, phase_current_2}` (motor current
  signature analysis literature documents bearing-fault energy leaking
  into the current spectrum via load fluctuation — real but indirect)
- other = `{force, speed, torque}` (drive-side operating-point
  parameters, not bearing-damage sensors)

## Results: direct / direct-or-indirect hit rate (%)

| category | signal | n | direct% | direct-or-indirect% |
|---|---|---|---|---|
| outer_ring | **B** | 339486 | **68.2** | **88.4** |
| outer_ring | C | 339486 | 36.2 | 37.0 |
| outer_ring | H | 339486 | 64.1 | 73.5 |
| outer_ring | K | 281946 | 34.8 | 43.5 |
| outer_ring | BK | 281946 | 47.5 | 55.2 |
| outer_ring | CK | 281946 | 34.0 | 35.0 |
| outer_ring | HK | 281946 | 59.2 | 66.5 |
| inner_ring | **B** | 311520 | **81.5** | **93.0** |
| inner_ring | C | 311520 | 50.0 | 50.7 |
| inner_ring | H | 311520 | 74.8 | 80.9 |
| inner_ring | K | 258720 | 38.5 | 44.2 |
| inner_ring | BK | 258720 | 56.0 | 60.7 |
| inner_ring | CK | 258720 | 44.1 | 45.1 |
| inner_ring | HK | 258720 | 71.9 | 76.3 |
| combined | **B** | 84960 | **95.2** | **99.9** |
| combined | C | 84960 | 65.0 | 65.0 |
| combined | H | 84960 | 87.5 | 91.8 |
| combined | K | 70560 | 64.0 | 64.0 |
| combined | BK | 70560 | 73.0 | 73.0 |
| combined | CK | 70560 | 64.4 | 64.4 |
| combined | HK | 70560 | 85.6 | 88.6 |

## The headline finding: B is the clear localization winner, the OPPOSITE of detection-AUROC ranking

`scoring-signals-B-C-E-H.md` documents Paderborn's detection AUROC
ranking as C(0.970)/E(0.977)/H(0.961) all clearly above B(0.888) — C and
the physics-edge signal E are the recommended detection combination. For
**localization**, the ranking inverts: B wins every category by a wide
margin (68-95% direct vs C's 36-65%), and even beats H (64-88%). `B`'s
top node is `vibration_1` in every category by a large plurality
(e.g. outer_ring: 231559/339486 = 68%); C's argmax scatters onto `force`
and `speed` — plausible operating-point channels a fault-sensitive
learned-attention residual can mis-fire on, but not the actual damage
channel. **Why**: B is the simplest possible mechanism (raw per-node
deviation magnitude, no cross-node structure at all) — exactly the
mechanism that wins when a fault's true signature IS one node's
amplitude spiking (bearing vibration during impact events), and C's
attention-based correction step, exactly as documented for robo3er's
`stuck`/`cable trapped` in [[robo3er-diagnosis-localization]]'s
mechanism section, doesn't help isolate a single-node amplitude fault —
it can actively dilute it by blending in neighbor predictions.

**K is the weakest single signal here** (34.8-64.0% direct-or-indirect,
below even C), and `BK`/`CK` sit between B and K/C respectively —
consistent with the combo-routing finding from the other two datasets
(BK/CK never clearly beat the better of their two inputs; `BK` tracks
closer to B, `CK` tracks closer to C, because whichever component wins
the per-window scalar comparison supplies its own, unchanged, argmax).

**Practical implication**: detection and diagnosis want DIFFERENT
signals on Paderborn specifically — recommend `max(C, E, H)` for
detection (per `scoring-signals-B-C-E-H.md`) but read off **B's own
argmax**, not C's or the combo's, when the deployment question is "which
channel is the damaged bearing showing up in." This is the sharpest
concrete illustration yet of `diagnosis_interpretability_review.md`'s
core thesis that detection AUROC and diagnosis accuracy are genuinely
different axes, not proxies for each other.

## Caveats

- One `.mat` file (`N15_M01_F10_KA08_2.mat`) failed to parse and was
  skipped (`Expecting matrix here`) — a pre-existing data-quality issue
  in the raw Paderborn archive, not specific to this run; affects one of
  80 files for one damaged code, negligible at this sample size.
- Single seed, no repeats.
- `outer_ring`/`inner_ring`/`combined`'s domain is identical by
  construction (6-node set can't separate damage location) — this
  dataset cannot test whether B/C/H/K distinguish outer- from
  inner-ring damage, only whether they find "the vibration channel" at
  all.

See also `diagnosis_interpretability_review.md` and
[[scoring-signals-B-C-E-H]] (Paderborn's detection-AUROC reference this
contrasts with).
