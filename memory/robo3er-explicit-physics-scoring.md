---
name: robo3er-explicit-physics-scoring
description: Explicit-formula physics scoring tested on robo3er and Paderborn; direction closed. Key finding: the apparent win over TypedRelationAnomalyHead is explained by max- vs. softmax-aggregation, not by explicit formulas being better. Do not revisit.
metadata:
  type: project
---

# Explicit physics scoring: tried, findings recorded, direction closed

**Status: closed. Do not add explicit-formula physics scorers to future benchmarks.**

## What was tried

Replaced `TypedRelationAnomalyHead` with a non-learned `ExplicitPhysicsScorer` that
fits `y = k·x + b` per declared edge on raw healthy data, standardizes residuals
per-edge via calib IQR, and combines edges by MAX (not softmax). Tested on:

- **robo3er** (`run_robo3er_v2_1_explicit_physics.py`, now deleted): differential-drive
  kinematics edges (wheel → odom/imu), regime-gated, 9 edges.
- **Paderborn** (`run_paderborn_v2_1_explicit_physics.py`, deleted): 8 electromechanical
  edges (torque/speed → current/vibration), first global then per-operating-point.

All benchmark scripts, checkpoints, and scorer source files are deleted:
`src/models/explicit_physics_score.py`, `src/models/paderborn_explicit_physics_scorer.py`,
`benchmark/run_robo3er_v2_1_explicit_physics.py`,
`benchmark/run_paderborn_v2_1_explicit_physics{,_per_op}.py`.

## What the numbers actually showed

### robo3er

| metric (mean AUROC) | V3.1 typed+softmax | explicit+max |
|---|---|---|
| phys-only (E vs D) | 0.759 | 0.637 |
| combined no-cov (F vs G) | 0.715 | 0.769 |
| combined +cov (J vs K) | 0.725 | **0.810** |

Explicit+max won the combined score (+0.085 on K vs J), most sharply on `stuck`
(J=0.450 below chance → K=0.619). But the phys-only signal is weaker (0.637 vs 0.759).

### Paderborn

| metric (mean AUROC) | V3.1 typed+softmax | global explicit | per-op explicit |
|---|---|---|---|
| phys-only | 0.816 | **0.471** (below random) | 0.574 |
| combined +cov | **0.894** | 0.891 | 0.872 |

Global explicit completely failed (D=0.471). Per-operating-point regression recovered
to 0.574 but still well below the typed head's 0.816. Even with regime gating the
explicit approach adds nothing to the combined score.

## Why the apparent robo3er win does NOT justify the explicit approach

The robo3er win (K=0.810 vs J=0.725) was hypothesised to come from max-aggregation
replacing softmax, but **that hypothesis was refuted by experiment**.

After changing `TypedRelationAnomalyHead` aggregation from softmax to max (2026-08-16),
results on robo3er are **identical** to the softmax version (E_phys_max 0.760 vs 0.759,
J_v3_cov_max 0.725 vs 0.725, stuck E=0.519 unchanged). The reason: of robo3er's 7
declared edges, the `stuck`-critical ones (`current → wheel_vel`) have exactly 1
incoming edge per target, so softmax and max produce the same value. Max vs. softmax
only differs when a node has ≥2 incoming edges.

The **actual** reason the explicit scorer did better is that it operates on **raw
kinematic signals** where `y = k·x + b` directly captures differential-drive physics
(`v_lin = k_v·(v_l + v_r)`). `TypedRelationAnomalyHead` operates in embedding
deviation space `d = z − p*`, which compresses 60-frame windows through a trained
encoder; the encoder is trained on reconstruction and prototype commitment, not on
preserving the precise linear kinematic relationship that `stuck` breaks. The raw
signal residual is more sensitive to the specific physics violation that `stuck` causes.

**Corrected conclusion:** the explicit scorer's robo3er win reflects a genuine
representation-level difference (raw kinematics vs. compressed embedding), not an
aggregation defect. The aggregation change alone does not help.

## Why the Paderborn failure is structural, not fixable by better formulas

Paderborn bearing faults (outer/inner ring wear) manifest as **impulsive vibration at
defect characteristic frequencies** (ball-pass frequency and its harmonics). This is a
time-frequency signature embedded in the vibration channel's temporal pattern. It does
NOT change the mean-value relationships between torque, speed, and current.

The explicit scorer asks "is y's value consistent with k·x + b?" Bearing faults do not
break that relationship — the torque-current proportionality is intact even on a severely
damaged bearing. No formulation of explicit-formula physics scoring can detect what is
essentially a spectral anomaly from a mean-value regression residual.

`TypedRelationAnomalyHead` works on Paderborn (E=0.816) because the SharedEncoder
compresses 64-frame windows into embeddings that encode temporal structure (including
frequency content), and the typed head operates on **deviation embeddings** `d = z − p*`
in that compressed space, not on raw signal values. Operating-point effects are absorbed
by prototype selection — different operating points map to different prototypes, so `d`
is already regime-normalised without explicit gating.

## Architectural lesson recorded here

The explicit scorer and `TypedRelationAnomalyHead` are asking structurally the same
question ("does src's deviation predict dst's deviation?") at different representational
levels:

- **ExplicitPhysicsScorer**: raw signal space, formula `y = k·x + b`, needs regime gating
- **TypedRelationAnomalyHead**: embedding deviation space `d = z − p*`, typed Linear/MLP,
  regime-normalised implicitly via prototype matching

The embedding-deviation approach subsumes the explicit approach on every dataset where
the encoder captures the relevant physics (which it does, because it is trained to
reconstruct the windows and committed to the prototypes). The only legitimate complaint
against `TypedRelationAnomalyHead` is the softmax aggregation, not the use of learned
functions.

**Why:** the explicit approach was abandoned because it adds implementation complexity
(regime-gating logic, per-dataset scorer classes, raw-signal pipeline alongside the
scaled-signal model pipeline) with no net benefit over fixing the softmax aggregation in
the existing model. On Paderborn it actively fails due to structural mismatch between the
fault signal and the scoring mechanism.

**How to apply:** if physics-informed scoring seems attractive for a new dataset, first
check whether the fault mechanism actually breaks a mean-value relationship (not just a
spectral one). If yes, the fix is to use max-aggregation in `TypedRelationAnomalyHead`,
not to write an explicit scorer. See [[joint-prototype-scheme-v3]] for the architecture.
