---
name: sielaff-num-prototypes-sweep
description: NUM_PROTOTYPES grid search {2,4,8,16,24,32} x calib_mode {global,per_prototype} on Sielaff federated -- tests whether prototype count explains per_prototype's B/C regression from calib-in-prototype-ab.md.
metadata:
  type: project
---

# Sielaff `NUM_PROTOTYPES` grid search: does prototype count explain the `per_prototype` calibration regression?

## Why this exists

[[calib-in-prototype-ab]] found that `--calib-mode per_prototype` (Path B,
offline per-prototype median/IQR for B/C/K) helps robo3er but clearly hurts
Sielaff (B: 0.974→0.886, C: 0.976→0.942 at Sielaff's default
`NUM_PROTOTYPES=16`). That doc's stated but *unconfirmed* hypothesis: Sielaff
splits small per-machine calib sets (10 real machines, `CALIB_FRACTION=0.15`)
across 16 prototypes, so per-prototype median/IQR estimates are too noisy
(diluted calib samples) to help and instead hurt. Unlike robo3er
(`NUM_PROTOTYPES=2`, already swept over `{1,2,4,8}` historically), Sielaff's
prototype count had never been swept. This doc runs that sweep.

## Method

Added `--num-prototypes` CLI override to `benchmark/run_sielaff_v2_1_federated.py`
and `benchmark/run_sielaff_forecast_v2_federated.py` (default unchanged =
16, so existing baselines stay reproducible). Grid: `NUM_PROTOTYPES ∈
{2, 4, 8, 16, 24, 32}` (extended down to 2 beyond the requested `{4,8,16,24,32}`
since the curve was still moving at the low end) × `calib_mode ∈ {global,
per_prototype}`, federated, single seed (42), on `run_sielaff_v2_1_federated.py`
(B/C/H, no K). All 12 cells run; each takes ~15-20s on this machine (5 rounds
x 30 local epochs x 10 clients), so the full grid was cheap. `ema` mode was
out of scope per the task (separate known-buggy path).

As a nice-to-have, `run_sielaff_forecast_v2_federated.py` (adds K) was also
run at `NUM_PROTOTYPES ∈ {2, 16}` × both modes to check whether the pattern
holds with K in the mix -- not the full grid, prioritized lower per the task.

## Full grid: `run_sielaff_v2_1_federated.py` (B/C/H, single seed)

| M (NUM_PROTOTYPES) | calib_mode | B | C | H | I | valid frac (node, sum/total) |
|---:|---|---:|---:|---:|---:|---|
| 2 | global | 0.974 | 0.971 | 0.964 | 0.974 | -- |
| 4 | global | 0.975 | 0.978 | 0.966 | 0.975 | -- |
| 8 | global | 0.975 | 0.977 | 0.966 | 0.975 | -- |
| **16 (default)** | **global (baseline)** | **0.974** | **0.976** | 0.964 | 0.973 | -- |
| 24 | global | 0.973 | 0.976 | 0.968 | 0.974 | -- |
| 32 | global | 0.975 | 0.978 | 0.968 | 0.974 | -- |
| 2 | per_prototype | 0.955 | 0.953 | 0.964 | 0.955 | 0.600 (12/20) |
| 4 | per_prototype | 0.944 | 0.934 | 0.966 | 0.945 | 0.375 (15/40) |
| 8 | per_prototype | 0.905 | 0.930 | 0.966 | 0.917 | 0.188 (15/80) |
| 16 | per_prototype | 0.886 | 0.942 | 0.964 | 0.889 | 0.113 (18/160) |
| 24 | per_prototype | 0.901 | 0.945 | 0.968 | 0.904 | 0.058 (14/240) |
| 32 | per_prototype | 0.888 | 0.942 | 0.968 | 0.892 | 0.053 (17/320) |

"valid frac (node)" = sum across all 10 clients of `n_valid_score_node_prototypes`
(prototypes with ≥`MIN_PROTO_SAMPLES=5` calib windows, from the new
`ScoreCalibrationHead.calib_valid` diagnostic added this task) divided by
`10 x M` (total prototype slots across all clients). `n_valid_score_struct_prototypes`
is identical to node's in every cell here (B/C share the same `idx_calib`
routing, so the same windows-per-prototype counts apply to both). Per-client
raw counts are in each report JSON's `summary_n_valid_score_node_prototypes`.

Checkpoint reports: `checkpoints/sielaff/sielaff_v2_1_federated{,_m2,_m4,_m8,_m24,_m32}_report.json`
(global) and `checkpoints/sielaff/sielaff_v2_1_federated_calibmode_per_prototype{,_m2,_m4,_m8,_m24,_m32}_report.json`
(per_prototype). The M=16 filename (no `_m` suffix, matching the default)
reruns and overwrites the pre-existing baseline reports from
[[calib-in-prototype-ab]] -- numbers match to 3 decimals (0.974/0.976/0.964
global; 0.886/0.942/0.964 per_prototype), confirming this task's code change
(the new diagnostic fields, `--num-prototypes` plumbing) didn't alter existing
behavior.

## Nice-to-have: `run_sielaff_forecast_v2_federated.py` (adds K), M ∈ {2, 16}

| M | calib_mode | B | C | H | I | K | BK | CK | HK |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | global (baseline) | 0.974 | 0.975 | 0.955 | 0.974 | 0.910 | 0.976 | 0.974 | 0.953 |
| 16 | per_prototype | 0.924 | 0.975 | 0.955 | -- | 0.902 | 0.930 | 0.963 | -- |
| 2 | global | 0.968 | 0.978 | 0.953 | 0.974 | 0.930 | 0.963 | 0.968 | 0.952 |
| 2 | per_prototype | 0.966 | 0.959 | 0.953 | 0.973 | 0.930 | 0.963 | 0.968 | 0.952 |

(M=16 per_prototype's H/I/HK cells are the pre-existing values from
[[calib-in-prototype-ab]]'s checkpoint, not rerun this task -- that file
predates the diagnostic fields added here, so it has no
`n_valid_score_*_prototypes` entry; not rerun since forecast_v2 was explicitly
lower priority and the M=16 headline numbers were already established.)

At M=2, per_prototype's damage to B nearly disappears (0.968→0.966, vs.
0.974→0.924 at M=16) but C still drops meaningfully (0.978→0.959). K is
*bit-identical* between global and per_prototype at M=2 (0.930/0.930,
BK/CK/HK all identical too) -- `n_valid_score_k_prototypes` for this run
shows only 1-2 of 10 clients ever cross `MIN_PROTO_SAMPLES` for K's own
calib split (`[None, 0, 1, 1, 1, 1, 1, 1, 1, 1]`, `None`/`0` = client had no
valid forecast-pairing calib windows at all), so K's per_prototype path
falls back to global almost everywhere at M=2 by construction -- this is
consistent with, not contradictory to, the main B/C/H grid's story.

## Headline finding: the "too few calib samples" hypothesis is CONFIRMED IN PART, but does not fully explain the regression

**Part 1 -- confirmed: `global` mode is essentially flat across the whole
`NUM_PROTOTYPES` range (0.973-0.978 for B/C, no trend with M), while
`per_prototype` mode gets monotonically worse as M increases and the
fraction of prototypes with enough calib samples (`valid_frac`) shrinks.**
B drops from 0.955 (M=2, 60% of prototype slots valid) to ~0.886-0.905 (M≥16,
5-11% valid); the valid fraction and the AUROC drop track each other closely
across the grid (Spearman-obvious from the table, not formally tested given
single-seed data). This is exactly the mechanism [[calib-in-prototype-ab]]
hypothesized: more prototypes to split Sielaff's small per-machine calib sets
across means more prototypes fall back to global anyway, and the few that
don't are calibrated on noisier, smaller samples.

**Part 2 -- refuted as the *complete* explanation: even at M=2, where 60% of
prototype slots are valid and most clients have only 1-2 prototypes total
(so per-prototype calibration is nearly equivalent to per-client-global
calibration), `per_prototype` still underperforms `global` by a real margin**
(B: 0.955 vs. 0.974, -0.019; C: 0.953 vs. 0.971, -0.018). If sparse-calib-data
dilution were the *only* mechanism, M=2 should have closed most or all of the
gap to `global` -- it closes roughly three-quarters of the B gap (from -0.088
at M=16 to -0.019 at M=2) and about a fifth of the C gap (-0.034 → -0.018),
but does not eliminate either. Something beyond "average calib-samples-per-
prototype" is also in play -- plausibly that even a *valid* (≥5-sample)
per-prototype IQR/median estimated from a subset of an already-small calib
split (10 machines, `CALIB_FRACTION=0.15`) is intrinsically noisier than one
estimated from the FULL pooled calib set, independent of the nominal
"valid" threshold; `MIN_PROTO_SAMPLES=5` is a floor, not a guarantee of a
stable estimate at Sielaff's sample sizes. Not directly tested here (would
need e.g. bootstrapped IQR variance at n=5-30 vs. n~100+ to confirm) --
flagged as the most likely residual mechanism, not proven.

**So: does prototype count explain the per_prototype regression on Sielaff?
Partially.** It's a real, monotonic contributor (confirmed by the
`valid_frac` correlation) but not sufficient on its own -- a second,
un-diagnosed noise-floor effect from Sielaff's small calib splits persists
even when nearly all prototypes are individually "valid" by the
`min_samples` criterion.

## Recommendation

**No change to Sielaff's default `NUM_PROTOTYPES=16` recommended, and
`calib_mode` should stay `global` regardless of `NUM_PROTOTYPES`.** No cell
in this grid -- at any M from 2 to 32 -- beats `global`'s B/C AUROC at any M.
`global` itself is essentially insensitive to `NUM_PROTOTYPES` in this range
(0.973-0.978 throughout), so there is no detection-AUROC reason to move off
16 either. If per-prototype calibration is revisited for Sielaff in the
future, M=2 is the least-bad `per_prototype` setting found here (smallest
gap to `global`), but it is still a net loss vs. simply using `global` at
any M -- this is a negative result, not a tuning recommendation.

## Caveats

- **Single seed (42) per cell**, matching this project's standing
  convention -- no variance estimate. The `global`-mode deltas across M
  (0.973-0.978) are well within plausible single-seed noise and should NOT
  be read as "M helps global mode" or "M=4/32 beats M=16" -- they're flat.
  The `per_prototype`-mode trend across M, by contrast, is large (0.955 down
  to 0.886, a 7pt spread) and consistently monotonic-ish (M=24 ticks up
  slightly from M=16, plausibly single-seed noise on top of the real trend)
  -- unlikely to be pure noise, but exact per-cell magnitudes are still
  single-run estimates.
- The residual (Part 2) mechanism above is a hypothesis, not confirmed by
  direct measurement of per-prototype IQR estimation variance.
- robo3er/Paderborn are explicitly out of scope for this sweep (Sielaff-only
  per the task). robo3er's own prototype count (`NUM_PROTOTYPES=2`) already
  showed the OPPOSITE result under `per_prototype` (a small win, see
  [[calib-in-prototype-ab]]) -- the fact that Sielaff at the SAME M=2 still
  shows a regression is itself evidence that prototype count alone isn't the
  full story; something dataset-specific (calib split size, per-machine data
  volume) also differs between the two datasets.
- Forecast-script (K) grid is a nice-to-have, not the full grid -- only M ∈
  {2, 16} tested there, not the complete {2,4,8,16,24,32}.

See also [[calib-in-prototype-ab]] (the Path A/B calibration architecture and
the original unconfirmed hypothesis this sweep tests) and
[[scoring-signals-B-C-E-H]] (B/C/H signal definitions and the `global`-mode
baseline table this grid's `global` rows reproduce).
