# v3: typed relation-specific edges + prototype-conditioned standardization
# + anomaly attention -- a small, honest gain, with a caveat (2026-08-07)

## What this implements

Per `docs/joint_prototype_physics_gdn_anomaly_attention_prompt.md`, added
`TypedRelationAnomalyHead` (`src/joint_prototype_model.py`) alongside v2's
`TrendGraphAttentionHead`, forming `JointPrototypeGDNv3`
(`benchmark/run_paderborn_joint_prototype_v3.py`):

- **Relation-specific message functions** (Sec 9) on the same 8 declared
  physics edges as v1/v2: the 4 current-related edges
  (`speed/torque -> phase_current_1/2`) get a linear map ("proportional",
  the design doc's own textbook example -- motor current ~ torque via
  Kt); the 4 vibration/torque-related edges
  (`force/speed -> vibration_1/torque`) get a small MLP ("nonlinear" --
  contact mechanics, induction-motor torque-speed curve).
- **Prototype-conditioned residual standardization** (Sec 12): a Stage-B
  calibration pass after training computes per-(prototype, edge) mean/std
  of the raw residual from the calib split, falling back to global stats
  for any prototype with < 5 calib windows.
- **Unsupervised anomaly attention** (Sec 16, NOT the self-supervised
  Sec 17 version -- see scoping notes below): softmax over each node's
  incoming declared edges' standardized residuals, producing
  `s_node_typed` -- structurally separate from v2's `s_edge`
  ("who matters normally" vs. "who is broken now", Sec 13).

Same 6 nodes, same fit/calib/split, same core hyperparameters as v1/v2.
Ablations A-D reproduce v2's signals (retrained jointly with the new
typed-edge loss term); E and F are new (typed-edge-only, full v3).

## Result

| method | mean AUROC (26) | outer_ring | inner_ring | combined |
|---|---|---|---|---|
| AE (6ch, fair baseline) | 0.679 | 0.569 | 0.712 | 0.996 |
| v1 D (fixed-edge) | 0.802 | 0.736 | 0.823 | 0.991 |
| GDN (6ch, fair baseline) | 0.819 | 0.757 | 0.838 | 0.994 |
| v2 D (learned attention, original run) | 0.873 | 0.829 | 0.888 | 0.997 |
| v3 E: typed-edge only (new) | 0.816 | 0.756 | 0.831 | 1.000 |
| v3 D: v2 mechanism, retrained jointly with v3 | 0.889 | 0.860 | 0.892 | 0.995 |
| **v3 F: full (dG + node + v2 edge + typed edge)** | **0.898** | 0.869 | **0.903** | 1.000 |
| v3 C: v2 edge-attention alone, retrained jointly | **0.904** | 0.886 | 0.899 | 0.998 |

`v3 F_full` beats v2's original full model by **+0.025 mean AUROC**
(0.898 vs 0.873), and both v3 F and v3's own C (v2's exact mechanism,
just retrained alongside the new typed head) clearly beat the GDN
baseline (+0.079 / +0.085) and v1 (+0.096 / +0.102).

## The important caveat: where did the gain actually come from?

**The new typed-edge signal is NOT, by itself, stronger than v2's
generic learned attention.** `E_typed_edge_only` (0.816) is weaker than
`C_edge_only` (0.904) in the same run, and barely matches the *original*
v2 report's full-model score (0.873). Most of v3's headline improvement
over v2 traces to `C_edge_only` -- literally the same
`TrendGraphAttentionHead` mechanism v2 used -- scoring higher in this run
(0.904) than it did in v2's original run (0.861), most plausibly because
training it jointly with the new typed-edge auxiliary loss
(`LAMBDA_TYPED * l_typed` added to the training objective) acted as a
helpful regularizer on the *shared encoder* both heads read from, not
because the typed/anomaly-attention mechanism itself is doing strong
independent work. This is a legitimate, useful finding (multi-task
auxiliary losses helping shared representations is a well known effect)
but it is a different claim than "typed relations + anomaly attention
beat generic attention," which this run does NOT support on its own.

Confirming this isn't purely noise would need a controlled ablation
(same run, encoder trained identically, with/without the typed loss
term) which was not done here -- flagged as the natural next check.

Where `F` (the combined signal) DOES clearly help over `D` (v2's
mechanism alone, same run) is a specific, meaningful subset of bearings:
KA06 (0.554->0.670, +0.116), KA30 (0.545->0.606, +0.061), KI04
(0.558->0.666, +0.108), KI01 (0.929->0.996, +0.067), KB23
(0.985->1.000, +0.015) -- notably, several of these (KA06, KA30, KI04)
were among the hardest bearings for every prior method in this dataset's
whole exploration. On other bearings (KA07, KA08, KA09, KI07, KI08) `F`
is flat or slightly worse than `D`, consistent with the known
max-aggregation noise-floor issue (`memory/paderborn-joint-prototype.md`'s
root-cause analysis) -- adding one more max-aggregated dimension can
raise false-positive rates on the healthy test split even when it adds
genuine positive signal on some faults.

## Why the "prototype-conditioned" part is under-exercised here

Only **5 of 16 prototypes** had >= 5 calib windows; the other 11 fell back
to global (unconditioned) edge statistics. With only 360 calib windows
total (6 healthy bearings x 60 windows) spread across a 16-slot codebook
that saturates to 100% utilization almost immediately (same
non-convergence flag raised in `memory/paderborn-joint-prototype.md`),
most declared-edge residual statistics are NOT actually
prototype-conditioned in this run -- Sec 12's core idea (different
operating conditions should have different normal edge-residual scales)
is implemented but largely untested by this dataset's calib-split size.
A larger calib split or fewer prototypes would be needed to properly
exercise this mechanism.

## What's NOT done (deliberately scoped down, per this project's convention)

- Only 2 relation-type classes (linear/proportional vs. nonlinear), not
  the full 9-type taxonomy (Sec 7: integral/derivative/thermal/frequency/
  dynamic/monotonic/unknown) -- finer types need dedicated temporal
  structure (difference/aggregation operators) not built here.
- Anomaly attention is the unsupervised Sec 16 version (softmax over
  standardized residual magnitude with a learned temperature), not the
  self-supervised relation-breaking-augmentation trained version from
  Sec 17 (temporal mismatch, cross-condition swap, scaling/trend/
  frequency perturbation) -- a substantial separate undertaking.
- No 3-stage training curriculum (Sec 22) -- Stage A (representation) and
  the typed-edge loss are trained jointly in one pass, Stage B (edge
  statistics) runs once after training completes; no Stage C.
- Per-edge attention-weight inspection wasn't done for the typed head
  either (same open item v2 left for its own attention).

Full report: `checkpoints/paderborn/paderborn_joint_prototype_v3_report.json`,
model weights: `checkpoints/paderborn/paderborn_joint_prototype_v3.pth`.
