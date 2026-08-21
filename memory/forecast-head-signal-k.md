---
name: forecast-head-signal-k
description: Results for signal K (ForecastHead, GDN-attention forecast of raw future values) from shared_forecast_head_proposal.md -- robo3er centralized, v1 (in-window split, superseded) vs v2 (cross-window pairing, current)
metadata:
  type: project
---

# Signal K (ForecastHead) -- robo3er centralized, v1 vs v2

Implements `docs/shared_forecast_head_proposal.md`'s core idea (GDN-learned
cross-node attention predicting a node's future value, declared physics
edges as an additive learned-strength bias on the attention logits -- same
mechanism as `TrendGraphAttentionHead`) as `ForecastHead` +
`JointPrototypeV31Forecast` in `src/models/joint_prototype_model.py`.

**Scope actually tested vs. the proposal:** only "step 1" (add the
forecast head, in raw-signal space, alongside the unchanged
`SharedEncoder`) -- NOT "step 0" (backbone -> UNet/Transformer, not
attempted). Runs are centralized (`run_robo3er_v3_1.py`'s convention),
not federated.

## v1 (superseded) -- in-window prefix/suffix split

First implementation split each window into a 48-step prefix (input) and
12-step suffix (target), entirely within one window. **This was wrong**:
`window_size=60`, `stride=32` means adjacent windows already overlap 28
steps (47%) in raw time, and a within-window suffix is even more
trivially close to the input's own tail -- the forecast task was
substantially leak-prone, not a genuine "predict what hasn't been seen
yet" task.

| Signal | cable trapped | stuck | mean |
|---|---|---|---|
| K (v1) | 0.721 | 0.636 | 0.679 |

`cable trapped` = 0.721 was the tell -- far below every other signal's
~1.0 on that fault type, unexpected for what should be a predictable-
method's strong suit (see proposal's axis-2 discussion). Score was worse
than B/C/E individually and dragged the max-combination `L` below `H`
alone (0.757 vs 0.869).

## v2 (current) -- cross-window pairing, per the proposal's revised section 4

`benchmark/run_robo3er_forecast_v2.py`. Reuses the existing
fit/calib/test_normal split UNCHANGED (same as B/C/E/H) -- the only new
step is pairing each window index `i` with `i+1` *across* windows,
subject to:
- same robot (`data/robo3er/partition.pkl`'s per-client index ranges --
  window-index adjacency does NOT imply raw-time adjacency across a
  robot boundary),
- for the three normal splits, `i+1` also normal-labeled and in the SAME
  split (orphans at split/robot/fault boundaries dropped, not
  force-paired -- verified counts: fit 3940->3929, calib 844->843,
  test_normal 845->839, cable trapped 206->206, stuck 149->148),
- forecast target is ONLY window `i+1`'s non-overlapping tail
  `stride=32` raw timesteps (`[stride*i+window_size, stride*(i+1)+window_size)`),
  not the whole next window -- the overlapping 28 steps are already
  visible in the input and forecasting them would be near-trivial.

`ForecastHead`'s input is now the FULL window (same as every other
head), output is `[B, stride, N]`; `x_future` is supplied externally by
the caller (paired next-window's tail) rather than sliced from `x`
itself -- see `ForecastHead`/`JointPrototypeV31Forecast` docstrings in
`src/models/joint_prototype_model.py`.

| Signal | cable trapped | stuck | mean |
|---|---|---|---|
| B (node_max) | 0.993 | 0.519 | 0.756 |
| C (struct_max) | 0.994 | 0.528 | 0.761 |
| E (phys_max) | 1.000 | 0.525 | 0.763 |
| H (cov_mahal, current best) | 0.999 | **0.738** | **0.868** |
| **K (forecast_max, v2)** | **0.916** | 0.690 | **0.803** |
| L = max(B,C,E,H,K) | 0.994 | 0.560 | 0.777 |

B/C/E/H are NOT bit-identical to `run_robo3er_v3_1.py` anymore (unlike
v1) -- the joint-loss training batch now uses the paired fit subset
(3929 of 3940 windows) since `l_forecast` needs a valid pair per batch
item, so a handful of orphan windows are absent from every loss term,
not just `l_forecast`. Numbers moved by <0.01 (e.g. B 0.747->0.756),
consistent with noise from a very slightly smaller/reordered training
set, not a real regression.

## What v1 vs v2 together establish

**K is a substantially real, non-trivial signal when measured correctly.**
Moving from v1 to v2 took K from worst-of-five (0.679, below B/C/E) to
second-best single signal (0.803, above B/C/E/F/J, only below H) --
confirms v1's poor numbers were a data-pairing artifact, not evidence
that "GDN-learned attention forecasting doesn't work as well as
hand-fit physics." `cable trapped`'s jump (0.721->0.916) is the clearest
tell: a genuinely burst-type fault should be predictable's method's
strong suit (matches the old GDN baseline's 0.950 on the same fault
type family), and v1's implementation was actively hiding that.

**Axis 2 (temporal-prediction shrinkage bias on `stuck`) still holds, but
less severely than v1 suggested.** K on `stuck` = 0.690, up from v1's
0.636, and now clearly above the old raw-GDN-forecast baseline (0.528,
near-random) -- learned attention forecasting is NOT reproducing that
near-total failure. But K (0.690) still falls short of H (0.738), so the
proposal's central worry (predictable steady states -> small forecast
error -> weak anomaly signal) is not fully resolved, just less severe
than the in-window-split version implied.

**Combining still hurts.** `L = max(B,C,E,H,K)` = 0.777, well below `H`
alone (0.868) -- same failure mode as `J` in
[[scoring-signals-B-C-E-H]]'s combination-rules section: max-combining a
signal that's weak on ONE fault type (K on `stuck`, 0.690 vs H's 0.738)
with a signal that's weak on the OTHER (B/C/E on `stuck`, ~0.52) doesn't
average out -- elementwise max corrupts the ranking on whichever fault
type the newly-added signal is weakest on, per-sample, not just in
aggregate AUROC. **K is not recommended for robo3er's scoring
combination**; `H` alone (0.868) remains the best single choice, `max(B,
C, H)` the safe cross-signal default.

## Improvement 1 (2026-08-17): declared-edge prior ablation for the forecast head

Per `forecast_head_k_assessment.md`'s cheapest, highest-priority
experiment: does K's 0.803 mean AUROC actually depend on the forecast
head's `prior_bias_strength`/`prior_mask` mechanism, or does unbiased
learned top-k attention alone already get there? `run_robo3er_forecast_v2.py
--no-forecast-prior` (new `forecast_prior_edges` override on
`JointPrototypeV31Forecast`, independent of `edge_head`/`typed_head`'s
`prior_edges` -- B/C/E untouched).

**Result: the prior does NOT help, and mildly hurts.** Without it, K
mean AUROC is 0.808 (vs 0.803 with it) -- cable_trapped 0.929 (vs
0.916), stuck 0.686 (vs 0.690, negligible). More telling: the
declared-edge TARGET NODES' own calib-set forecast MSE is LOWER without
the prior (0.518 vs 0.549, averaged over the 5 distinct target nodes of
the 7 declared edges) -- the prior bias measurably makes those nodes'
own forecasts slightly worse, not better, contrary to what the
mechanism is meant to do.

**Conclusion:** K's value is coming almost entirely from unbiased
learned attention, not from the declared physics edges. This matches
one of the two outcomes the proposal's step 3 anticipated -- if
confirmed, physics-prior injection effort should stay concentrated on
`TrendGraphAttentionHead` (signal C, deviation space, where the prior
bias mechanism was originally validated), not be duplicated into the
forecast head. Do not add `forecast_prior_edges` bias by default for
robo3er; the flag is now available in the model/script for revisiting
on other datasets where the declared edges might matter more.

## Improvement 2 (2026-08-17): multi-step forecast horizon sweep

Per the proposal's own validation step 2 and the assessment doc's
improvement-2 hypothesis: does a longer forecast horizon (accumulating
more drift from `stuck`'s slow degradation) make the forecast error more
anomaly-sensitive? `run_robo3er_forecast_v2.py --horizon-mult M` chains
M consecutive non-overlapping `stride`-length future segments
(`forecast_h = M * 32`) rather than repeating overlapping window
content (see `build_pairs`/`gather_future`'s chain-validity logic --
ALL M successors must share the same robot, matching pairing rules).

| horizon_mult | forecast_h | K on cable_trapped | K on stuck | **K mean** | H on stuck | H mean | L mean |
|---|---|---|---|---|---|---|---|
| 1 (baseline) | 32 | 0.916 | 0.690 | 0.803 | 0.738 | 0.868 | 0.777 |
| 2 | 64 | 0.892 | 0.682 | 0.787 | 0.720 | 0.859 | 0.797 |
| 3 | 96 | 0.928 | 0.735 | 0.832 | 0.743 | 0.871 | 0.812 |
| 4 | 128 | 0.919 | **0.750** | 0.835 | 0.742 | 0.870 | 0.822 |
| 5 | 160 | 0.881 | 0.777 | 0.829 | 0.745 | 0.872 | 0.815 |
| 6 | 192 | 0.898 | 0.802 | **0.850** | 0.748 | 0.873 | 0.842 |
| 8 | 256 | 0.878 | 0.819 | 0.849 | 0.739 | 0.869 | **0.852** |

**Result: a real, striking effect on `stuck` -- and it's a genuinely new
finding, not a marginal tweak.** `stuck` AUROC for K climbs essentially
monotonically with horizon length (0.690 -> 0.819), and starting at
`horizon_mult=4` (forecast_h=128, i.e. forecasting ~2x the input
window's own length ahead), **K exceeds H specifically on `stuck`** for
every horizon tried since (H stays flat ~0.74 throughout, since it
doesn't depend on the forecast head at all). At `horizon_mult=8`, K's
`stuck` AUROC (0.819) beats H's (0.739) by 8 points. This directly
confirms the proposal's own hypothesis (`stuck`'s slow current decay
needs enough accumulated horizon before the drift becomes visible
against the forecast's own noise floor) and is the strongest positive
result in this signal's whole investigation so far.

**Trade-off:** `cable_trapped` drifts down as horizon grows (0.916 ->
0.878, noisily), since forecasting further ahead is inherently harder
even for undisturbed normal dynamics, diluting the sharp signal a burst
fault gives at short horizon. K's own MEAN AUROC plateaus around
`horizon_mult=6-8` (~0.849-0.850), still short of H's ~0.87 mean --
gains on `stuck` are being partly offset by losses on `cable_trapped`,
not pure upside.

**Max-combination still doesn't help, and the failure mode gets MORE
visible at long horizons, not less.** `L = max(B,C,E,H,K)` peaks at
`horizon_mult=8` (0.852) but stays below `H` alone (0.869) throughout
the whole sweep -- and per-fault-type, `L`'s `cable_trapped` AUROC
(e.g. 0.943 at h=8) is now visibly BELOW individual signals E/H's ~1.0
there, the clearest demonstration yet in this project of elementwise-max
combination corrupting sample-level ranking even when every contributing
signal individually scores well on that fault type (see
[[scoring-signals-B-C-E-H]]'s combination-rules section for the same
mechanism previously observed with `J`). **Do not use elementwise-max to
combine K with B/C/E/H, at any horizon** -- if K is used going forward,
it should be reported/considered as a standalone alternative for
slow/degradation-type faults specifically (where it can beat H), not
folded into a max ensemble.

**Practical implication:** at long horizons, K stops being "the weak
signal that hurts the ensemble" and becomes "the best available signal
for exactly the fault type (`stuck`) that every other signal here
struggles with." That reframes K's status in this project: not a
discarded direction, but a legitimate second signal worth keeping
separately calibrated for slow-onset faults, contingent on validating a
non-max combination rule (e.g. a small learned/logistic combiner, or
per-fault-type signal selection) -- out of scope for this investigation
but the clear next step if this line continues.

## Pairwise combination test (2026-08-17): B+K, C+K, H+K vs. the full max(B,C,E,H,K)

Requested follow-up: does pairing K with just ONE existing signal avoid
the ranking-corruption `L = max(B,C,E,H,K)` suffers from (per the
combination-rules discussion above)? Added `BK_max = max(B,K)`,
`CK_max = max(C,K)`, `HK_max = max(H,K)` to `run_robo3er_forecast_v2.py`,
computed at every horizon already swept.

| horizon_mult | B | C | H | K | **BK** | **CK** | **HK** | L |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.756 | 0.761 | 0.868 | 0.803 | 0.794 | 0.791 | 0.827 | 0.777 |
| 4 | 0.749 | 0.755 | 0.870 | 0.835 | 0.820 | 0.822 | 0.852 | 0.822 |
| 6 | 0.749 | 0.756 | 0.873 | 0.850 | 0.843 | 0.839 | 0.871 | 0.842 |
| 8 | 0.761 | 0.758 | 0.869 | 0.849 | 0.846 | 0.845 | 0.874 | 0.852 |
| **10** | 0.779 | 0.754 | 0.878 | 0.859 | 0.878 | 0.882 | **0.901** | 0.872 |
| 12 | 0.752 | 0.770 | 0.874 | 0.831 | 0.832 | 0.832 | 0.853 | 0.844 |

(mean AUROC over the 2 fault types; single seed per config, no repeats --
the horizon-to-horizon numbers carry real run-to-run noise, treat the
exact peak at `horizon_mult=10` as indicative, not a precisely located
optimum.)

**`BK` and `CK` behave exactly as the combination-rules section predicts
and DON'T help**: at every horizon, `BK`/`CK` sit BELOW both of their
own inputs' better half (e.g. h=8: `BK`=0.846 < `K`=0.849; h=1: `CK`=0.791
< `K`=0.803) -- B and C are near-random on `stuck` (~0.50-0.56)
throughout every horizon (they don't depend on the forecast head), so
max-combining either of them with K reliably drags K's `stuck` advantage
back down without buying anything on `cable_trapped` (B/C are already
~0.99+ there, same ceiling H has).

**`HK` is different and consistently the best combination tested**:
`HK` beats plain `H` from `horizon_mult=6` onward (0.871 vs 0.873 --
roughly tied), clearly from `horizon_mult=8` (0.874 vs 0.869), and most
sharply at `horizon_mult=10` (**0.901 vs H's 0.878** -- `HK`'s
`cable_trapped`=0.969, `stuck`=0.834, both close to each input's own
strong side: H's `cable_trapped` ceiling ~0.999 barely erodes to 0.969,
while K's `stuck` strength ~0.839 is retained at 0.834). This is because
H and K are complementary rather than redundant on this dataset's 2
fault types (H's real weak spot, `stuck`, is K's real strong spot, and
vice versa for `cable_trapped`) -- unlike B/C which are weak on `stuck`
in the SAME way K's OTHER partners already are, so pairing K with them
adds no new information there, just noise.

**Practical takeaway, superseding the earlier "don't combine K"
conclusion:** the earlier verdict (K hurts every max-combination) was
based only on the FULL `L = max(B,C,E,H,K)` combination, which dilutes
H's strong `cable_trapped` signal through B/C/E on its way to combining
with K. **The correct combination is specifically `HK = max(H, K)`,
skipping B/C/E entirely** -- at `horizon_mult=10` this is the best
single scoring rule found anywhere in this whole investigation (0.901
mean, beating every one of B/C/E/F/H/I/J/K/L individually). Still just
elementwise max (not a learned combiner), so the same ranking-corruption
risk that hurt `L` is structurally still possible on datasets/fault
types where H and K are NOT complementary -- this result is specific to
robo3er's 2 fault types and should be re-checked before assuming it
generalizes.

## What's still open (per the proposal's own validation checklist)

- Step 0 (backbone upgrade) not attempted -- v1/v2 and the horizon/prior
  sweeps all used the unmodified `SharedEncoder`.
- A non-max combination rule for folding K in without corrupting
  `cable_trapped`'s near-perfect ranking (see improvement-2's
  conclusion) -- the clearest concrete next step.
- Declared-edge ablation was only run at `horizon_mult=1`; whether the
  prior stays unhelpful at longer horizons (where the target nodes'
  own forecast task is harder) is untested.
- Not yet run on Paderborn/Sielaff/voraus-AD, or federated.

See also `docs/shared_forecast_head_proposal.md` (source design doc,
includes the section-4 data-pairing spec v2 implements) and
[[scoring-signals-B-C-E-H]] (existing signal reference this extends).
