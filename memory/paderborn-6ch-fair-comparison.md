# Fair 6-channel GDN vs. AE vs. Joint Prototype comparison -- a more
# honest, humbling result than the earlier framing (2026-08-07)

## Why this run was needed

`memory/paderborn-joint-prototype.md` compared the new
`JointPrototypeGDN` (6 channels: vibration_1, phase_current_1/2, force,
speed, torque) against `run_paderborn_torque_residual_gdn.py`'s plain-GDN
baseline -- which only used 3 channels (vibration_1 + phase_current_1/2).
That was NOT an apples-to-apples comparison: JointPrototype had 3 extra
input channels (force/speed/torque) plain GDN never saw. This run
(`benchmark/run_paderborn_6ch_comparison.py`) fixes that -- plain
`src.gdn_model.GDN` and `src.conv_autoencoder.ConvAutoEncoder` trained on
the EXACT SAME 6-channel data, same fit/calib/split, same seed.

## Result: plain GDN is actually the BEST of the three on fair footing

| method | mean AUROC (26) | outer_ring | inner_ring | combined |
|---|---|---|---|---|
| **A: GDN (6ch, no memory, no physics prior)** | **0.819** | **0.757** | **0.838** | 0.994 |
| C: Joint Prototype (full, D) | 0.802 | 0.736 | 0.823 | 0.991 |
| B: AE (6ch, no graph, no memory) | 0.679 | 0.569 | 0.712 | 0.996 |

**Plain GDN beats Joint Prototype on every category once both see the
same 6 channels.** The earlier "first net-positive physics-informed
result" framing in `memory/paderborn-joint-prototype.md` was true
relative to the WEAKER 3-channel baseline it was compared against, but
not the strongest available baseline -- once GDN is given the same
information (force/speed/torque added), its own learned graph attention
already captures most or all of whatever the joint-prototype/trend-edge
machinery was adding. This is being reported plainly, not reframed as a
win -- the added architectural complexity (joint prototypes + trend
edges) did not clearly pay for itself once the comparison is done fairly.

Per-bearing, GDN's advantage over JointProto is concentrated in exactly
the bearings JointProto had regressed on relative to the OLD 3-channel
baseline (`memory/paderborn-joint-prototype.md`'s per-bearing table):
KA07 (GDN 0.734 vs. JointProto 0.526), KI08 (0.667 vs. 0.525), KA09
(0.807 vs. 0.537) -- suggesting plain GDN's forecasting mechanism, given
the full 6-channel input directly, was already extracting the
cross-channel signal the joint-prototype design was trying to add more
explicitly, at least for these specific bearings.

## Secondary finding: AE got much worse with 6 channels, the opposite of
## its earlier standing

In the OLD 3-channel comparisons (`benchmark/datasets/paderborn_physics.md`),
AE was consistently the STRONGEST of the methods tried (e.g. category
means as high as outer_ring 0.697/inner_ring 0.819 territory in some
runs). Here, with 6 channels, AE is clearly the WEAKEST (0.679 mean,
outer_ring 0.569). Plausible reason (not verified): AE's whole-window
reconstruction task may be more sensitive to being given raw, differently
-scaled/behaved channels (force/speed/torque, DC-ish and nearly constant
within a file, mixed in with oscillating vibration/current) without any
relational structure to help it weight or contextualize them -- whereas
GDN's graph attention and the joint prototype's per-node deviation
scoring both have SOME mechanism for treating channels differently by
node identity. Not confirmed, flagged as an open question.

## What this means for the joint-prototype redesign

Not a refutation of the trend-based-edge idea itself (`TrendEdgeHead`'s
mechanism -- predicting deviations, not magnitudes -- is still a
methodologically sound response to the collinearity problem diagnosed in
`memory/paderborn-torque-residual-gdn.md`), but a clear signal that on
THIS dataset, at this scale/tuning, it doesn't yet outperform the
simpler GDN baseline once the comparison is fair. Worth remembering
before recommending the joint-prototype architecture as a strict
improvement -- its real, distinguishing value proposition is the
node/edge-level LOCALIZATION and INTERPRETABILITY (which specific
feature and which specific relationship broke, not just a single AUROC
number), which plain GDN's forecast residual doesn't provide at all --
not raw detection AUROC, where this comparison shows GDN still ahead.

Full report: `checkpoints/paderborn/paderborn_6ch_comparison_report.json`.
