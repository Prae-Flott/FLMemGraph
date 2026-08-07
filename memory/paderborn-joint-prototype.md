# Joint Prototype Memory + trend-based edge anomaly: the first net-positive
# physics-informed result on Paderborn (2026-08-07)

## What was built

Following a full design conversation (`docs/joint_prototype_three_level_anomaly_prompt.md`,
added to `docs/` this session) proposing a pivot from parameter-level
(exact magnitude) physical relations to TREND relations, after three
consecutive negative results trying the former
(`memory/paderborn-physics-residual-gdn.md`,
`memory/paderborn-torque-residual-gdn.md`,
`memory/paderborn-fixed-mu-torque-residual.md`):

- `src/joint_prototype_model.py`: `SharedEncoder` (unchanged pattern from
  `fl_model.py`) + `JointPrototypeMemory` (M prototypes, each a FULL
  `[N,D]` snapshot of all nodes together -- "has the whole device been in
  roughly this joint operating state before," not per-feature novelty
  the way `gdn_memory_model.DiscretePrototypicalMemory`/`fl_model`'s
  memory head works) + `TrendEdgeHead` (the key redesign: predicts node
  j's DEVIATION from its own matched-prototype baseline from node i's
  deviation from ITS baseline, via a learned linear map -- operates on
  RELATIVE deviations, not raw physical magnitudes, sidestepping the
  confounded-4-operating-condition collinearity problem that sank every
  prior `paderborn_physics.py` residual).
- `src/decision_logic.py`: added `ThreeLevelDecision`/`three_level_decision()`,
  the stateless version of the design doc's Sec 11 four-quadrant table
  (Known Normal / Localized Node Anomaly / Novel-but-Consistent / Fault).
  Not yet wired into a stateful monitor (matches the still-unwired status
  of `StreamMonitor` itself, a pre-existing gap noted earlier this
  project).
- `benchmark/run_paderborn_joint_prototype.py`: single-machine (no
  federation), ALL 6 real-time-varying channels as nodes (`vibration_1`,
  `phase_current_1/2`, `force`, `speed`, `torque` -- more than any prior
  Paderborn script, which used 3-4), 8 physics-skeleton edges (force/speed
  as independently-controlled inputs, torque/current/vibration
  downstream). Runs all 4 mandatory ablations from the design doc's
  Sec 16 (A: prototype-only, B: +node, C: edge-only, D: full).

## Result: net positive, the first one on this dataset

Mean AUROC across all 26 damaged bearings:

| method | mean AUROC |
|---|---|
| A: prototype only (d_G) | 0.776 |
| B: +node | 0.806 |
| C: edge only (trend-based) | 0.799 |
| D: full (all 3 signals) | 0.802 |

Category means, compared against the best prior baseline
(`run_paderborn_torque_residual_gdn.py`'s method A -- plain GDN
forecasting, RMS-decimated current, no physics residual):

| category | prior best baseline | A: proto only | B: +node | C: edge only | D: full |
|---|---|---|---|---|---|
| outer_ring | 0.697 | 0.699 | **0.729** | **0.729** | **0.736** |
| inner_ring | 0.819 | 0.818 | **0.841** | 0.822 | 0.823 |
| combined | 0.998 | 0.931 | 0.988 | 0.992 | 0.991 |

**Adding node and/or edge signals to the joint prototype beats the prior
best baseline on outer_ring and inner_ring** -- the two categories where
every previous physics-informed addition (current residual, torque
residual free-fit, torque residual fixed-mu) made things WORSE. This is
the first time in this dataset's exploration that a physics-informed
signal beyond plain forecasting has net-helped rather than net-hurt.

Per-bearing (D vs. prior baseline): **7 improved, 6 got worse, 13
unchanged** (mostly already at the AUROC=1.000 ceiling in both). Not a
clean sweep -- worth reporting the real spread, not just the mean:

| bearing | baseline | D (full) | delta |
|---|---|---|---|
| KA05 | 0.392 | 0.630 | **+0.238** |
| KA01 | 0.544 | 0.744 | **+0.200** |
| KA07 | 0.340 | 0.526 | **+0.186** |
| KA08 | 0.356 | 0.491 | **+0.135** |
| KI07 | 0.509 | 0.676 | **+0.167** |
| KA09 | 0.822 | 0.537 | **-0.285** |
| KI01 | 0.974 | 0.901 | -0.073 |
| KI04 | 0.477 | 0.409 | -0.068 |

The improvements land mostly on bearings that were ALREADY the weakest
under the old approach (KA01/05/07/08 were among the worst-performing
outer-ring bearings before) -- the new signal helps most exactly where
help was most needed. The regressions are concentrated in a few bearings
that were already doing well (KA09, KI01) or already weak in a different
way (KI04) -- not random noise, but not yet explained either (open
follow-up).

## Why this worked where parameter-level residuals didn't

`TrendEdgeHead` predicts `d_j = z_j - p_j*` (node j's deviation from ITS
OWN locally-matched-prototype baseline) from `d_i = z_i - p_i*`, not
`x_j` from `x_i` in raw physical units. Because the joint prototype
retrieval already picks out which of the (up to 16 learned, roughly
tracking the 4 known + within-condition variation) operating regimes the
current window is in, the edge relation only has to explain CO-DEVIATION
around that regime's own baseline -- a question that doesn't require
separating force's effect from speed's effect from "which of the 4
conditions this is" the way a single global OLS regression across pooled
raw magnitudes did. This directly targets the root cause diagnosed in
`memory/paderborn-torque-residual-gdn.md` (Paderborn's 4 discrete
conditions confound the predictors) without needing to fix the
regression itself -- it changes what's being predicted (relative
trend/co-deviation, locally referenced) rather than how it's fit.

## What's NOT done / open questions

- `NUM_PROTOTYPES=16` was not tuned -- codebook utilization hit 100% by
  epoch 1 and stayed there, same "reached ceiling immediately" pattern
  seen in earlier memory experiments on this dataset; unclear if more
  prototypes would help or just add capacity nothing uses.
- The 8-edge physics skeleton (force/speed -> vibration/torque/current)
  is domain-knowledge-declared, not learned or ablated edge-by-edge --
  unknown which specific edges are pulling the most weight in `C`'s
  positive result vs. which might be inert or even net-negative
  individually.
- `ThreeLevelDecision`/`three_level_decision()` is implemented but not
  yet run against this experiment's actual (d_G, s_node, s_edge) values
  to see how many windows land in each of the 4 quadrants -- would be a
  natural, cheap follow-up given the scores are already computed.
- KA09/KI01's regressions aren't explained -- worth checking whether
  they're cases where the OLD single-signal (forecast residual) approach
  happened to be unusually well-suited, or whether the new architecture
  has a specific weakness there.
- No comparison yet against `run_paderborn_fl_model.py`'s FLGDNMemory
  (the other joint-encoder design, using per-node independent memory
  rather than joint prototypes) on the SAME 6-channel node set -- the
  comparison table above uses the 3-channel GDN-raw baseline since that
  was the strongest prior result, but an apples-to-apples 6-channel
  FLGDNMemory run doesn't exist yet.

Full report: `checkpoints/paderborn_joint_prototype_report.json`, model
weights: `checkpoints/paderborn_joint_prototype.pth`.
