---
name: calib-in-prototype-ab
description: Branch experiment (calib-in-prototype-ab) closing the B/C/K-vs-H/E calibration-architecture gap -- Path B (offline per-prototype median/IQR, new ScoreCalibrationHead) vs Path A (online EMA per-prototype mean/var) vs the existing global-median/IQR baseline, robo3er + Sielaff, federated, single seed.
metadata:
  type: project
---

# Closing the B/C/K vs. H/E calibration-architecture gap: per-prototype and EMA calibration, A/B tested

## The gap this closes

H (`DeviationCovarianceHead`) and E (`TypedRelationAnomalyHead`) have always
calibrated **per-prototype**: `set_calibration()` fits mean/std (H: mean +
inverse covariance; E: mean + std) separately for each of the codebook's
prototypes from the held-out calib split, falling back to global stats when
a prototype has fewer than `min_samples` calib windows. These are `nn.Module`
buffers, saved with the model.

B (`node_max`), C (`struct_max`), and K (`forecast_max`) never had this: their
z-score was a **single global median/IQR**, computed ad-hoc inside each
`run_*_federated.py` script (`zscore()`/`two_stage_group_score()`), not tied
to which prototype a window matched, not stored anywhere in the model. This
branch (`calib-in-prototype-ab`) implements and empirically compares two ways
to close that gap:

- **Path B** (implemented first, lower risk): a new `ScoreCalibrationHead`
  class in `src/models/joint_prototype_model.py`, structured identically to
  `DeviationCovarianceHead`/`TypedRelationAnomalyHead` -- per-prototype
  median/IQR (kept as median/IQR, not mean/std, since that's what B/C/K
  already used), `set_calibration()` fit from the same calib split,
  `min_samples`-gated fallback to global. Attached to `JointPrototypeV31`/
  `JointPrototypeV31Forecast`/`JointPrototypeV21`/`JointPrototypeV21Forecast`
  as `score_calib_node`/`score_calib_struct`/`score_calib_k`, fit via
  `set_score_calibration()`/`set_k_calibration()`.
- **Path A** (implemented second, more risk): `JointPrototypeMemory.update_ema()`
  -- an online, VQ-VAE-codebook-EMA-style per-prototype mean/var accumulator
  for `d_node` (signal B only; C/K have no online EMA hook, by explicit scope
  -- see "What's still open" below), gated by an `ema_warmup_steps` warm-up
  so early, unstable prototype assignments don't get baked into the running
  statistics. Decay=0.99, warmup=200 steps -- **not tuned**, a real
  limitation (see below).

Both are wired into `run_robo3er_v3_1_federated.py`,
`run_robo3er_forecast_v2_federated.py`, `run_sielaff_v2_1_federated.py`, and
`run_sielaff_forecast_v2_federated.py` via a new `--calib-mode
{global,per_prototype,ema}` flag. **`global` is the untouched default** --
verified bit-identical (B/C/H) to the pre-existing checkpoint reports before
this branch existed, modulo <0.003 float noise on E/F/J from an unrelated
nondeterminism source (float op ordering in `TypedRelationAnomalyHead`, not
this branch's code). This is the regression check the task required, and it
passes.

## Result table (federated, single seed, mean AUROC over all client-fault pairs)

### robo3er (`run_robo3er_v3_1_federated.py`, B/C/E/H)

| calib_mode | B | C | E | H | F | I | J |
|---|---|---|---|---|---|---|---|
| global (baseline) | 0.945 | 0.945 | 0.686 | 0.948 | 0.943 | 0.947 | 0.946 |
| per_prototype | **0.951** | **0.960** | 0.686 | 0.948 | 0.944 | 0.949 | 0.947 |
| ema | 0.769 | 0.945 | 0.686 | 0.948 | 0.944 | 0.946 | 0.946 |

### robo3er (`run_robo3er_forecast_v2_federated.py`, adds K, horizon_mult=10)

| calib_mode | B | C | K | BK | CK | HK |
|---|---|---|---|---|---|---|
| global (baseline) | 0.948 | 0.958 | 1.000 | 0.995 | 1.000 | 0.995 |
| per_prototype | 0.948 | 0.958 | 1.000 | 0.995 | 1.000 | 0.995 |
| ema | 0.782 | 0.958 | 1.000 | 1.000 | 1.000 | 0.995 |

### Sielaff (`run_sielaff_v2_1_federated.py`, B/C/H)

| calib_mode | B | C | H | I |
|---|---|---|---|---|
| global (baseline) | **0.974** | **0.976** | 0.964 | 0.973 |
| per_prototype | 0.886 | 0.942 | 0.964 | 0.889 |
| ema | 0.935 | 0.976 | 0.964 | 0.964 |

### Sielaff (`run_sielaff_forecast_v2_federated.py`, adds K, horizon_mult=10)

| calib_mode | B | C | K | BK | CK | HK |
|---|---|---|---|---|---|---|
| global (baseline) | **0.974** | 0.975 | **0.910** | 0.976 | 0.974 | 0.953 |
| per_prototype | 0.924 | 0.975 | 0.902 | 0.930 | 0.963 | 0.949 |
| ema | 0.966 | 0.975 | 0.910 | 0.948 | 0.974 | 0.953 |

(C is unaffected by `ema` mode in every table above by construction -- no
online EMA hook exists for `resid_struct`, so `ema` mode silently falls back
to the exact `global` z-score for C, and for K in the forecast scripts. This
is documented in each script's own `--calib-mode` docstring, not a bug.)

## Headline finding: **the two paths point in OPPOSITE directions, and neither is a clean win**

**Path B (per-prototype, offline) helps on robo3er, hurts on Sielaff.**
On robo3er it's a clear, if modest, win -- B +0.6pt, C +1.5pt, entirely
concentrated on the `stuck` fault type (B: 0.933→0.944; C: 0.926→0.956;
`cable_trapped` is untouched, 0.957/0.963 exactly, both modes). This matches
[[scoring-signals-B-C-E-H]]'s existing finding that H's per-prototype
covariance is specifically what rescues `stuck` -- the same per-prototype
conditioning now helps B/C get partway there too, for the same underlying
reason (the locked-wheel regime has its own distinct normal-deviation scale
that a pooled global IQR was diluting).

On Sielaff it's a clear loss -- B drops 0.974→0.886 (v2_1 script) and
0.974→0.924 (forecast script), C drops 0.976→0.942 (v2_1 script). Sielaff
uses `NUM_PROTOTYPES=16` (vs. robo3er's 2), and per-machine calib splits are
small (10 real machines, `CALIB_FRACTION=0.15` of an already-small per-client
normal set) -- more prototypes to split calib data across, with the same
`min_samples=5` floor, means more prototypes likely fall back to global
anyway, but the ones that DON'T get calibrated on very few (5-20) windows,
a plausible source of the regression (noisy per-prototype IQR estimates,
not necessarily a wrong idea in principle). Not conclusively diagnosed within
this investigation's time budget -- flagged as the clearest next step if this
line continues (try raising `min_samples` on Sielaff specifically, or check
the underlying per-prototype-window-count distribution directly).

**Path A (online EMA) is a consistent net negative for B, everywhere tested,
confirming the fit-split bias risk the task asked us to check for.** EMA
tanks B on both robo3er (0.945→0.769, and 0.948→0.782 in the forecast
script) and Sielaff (0.974→0.935, and 0.974→0.966 in the forecast script) --
the only calib_mode/dataset/script combination where B drops below every
other signal in its own table. On robo3er the damage is concentrated
opposite of Path B's win: EMA hurts `cable_trapped` MORE than `stuck`
(cable_trapped B: 0.957→0.791, stuck B: 0.933→0.746) -- the fault type
Path B left untouched is exactly the one Path A breaks. **This is NOT simply
"EMA underestimates variance" in the way the task's stated bias-risk
hypothesis anticipated** (a uniform underestimate would inflate all z-scores
similarly and, by itself, wouldn't necessarily hurt rank-based AUROC) --
the more likely mechanism is per-prototype INCONSISTENCY: different
prototypes reach different points in their EMA warm-up/convergence
trajectory by the time training ends (especially under the federated
per-round `load_memory()` codebook resets, which zero `usage_count` but NOT
the EMA buffers -- the EMA statistics silently keep accumulating across
memory-exchange rounds even as which windows route to which prototype index
shifts underneath them), corrupting CROSS-WINDOW comparability of the
z-scored statistic rather than uniformly rescaling it. This is a genuine,
previously-undocumented risk of Path A's design, not the specific bias
mechanism the task hypothesized -- worth flagging clearly since it means
simply "collecting more data" or "waiting longer" won't fix it; the
EMA-vs-codebook-reset interaction needs its own fix (e.g. resetting EMA
buffers alongside `usage_count` in `load_memory()`, not attempted here).

**C is untouched in every EMA run (by construction, no online hook), K is
untouched in EMA runs (same reason) and modestly hurt in per_prototype runs
on Sielaff (0.910→0.902) but exactly unchanged on robo3er (1.000→1.000,
already at ceiling so a regression there was structurally impossible to
observe).**

## Why robo3er's per_prototype result is IDENTICAL between the two scripts' baselines but DIVERGES under per_prototype mode

Both `run_robo3er_v3_1_federated.py` and `run_robo3er_forecast_v2_federated.py`
use `NUM_PROTOTYPES=2`, and both scripts' final-round `alignment_log` shows
`num_shared_prototypes: 1` (only 1 of 2 prototypes actually gets cross-client
traffic after federated alignment) for BOTH scripts. Despite that surface
similarity, `per_prototype` mode is a no-op in the forecast script (bit-
identical to `global`, verified per-fault-type) but a real improvement in the
v3_1 script. The likely explanation: `alignment_log`'s summary stat
(`num_shared_prototypes`) doesn't capture PER-CLIENT prototype usage --
the forecast script's extra `l_forecast` training loss term changes the
learned embedding geometry enough that, for the specific clients that matter
(robot01/robot04), effectively ALL of that client's own calib+test traffic
collapses onto a single prototype index, making "per-prototype" stats
identical to "all calib data" stats by construction. In the v3_1 script, that
collapse is NOT total for those same clients -- enough traffic reaches the
second (personalized) prototype for its own median/IQR to differ from the
pooled global one. Not confirmed by directly inspecting per-client `idx`
histograms (time-limited); the alignment_log numbers alone are consistent
with, but don't prove, this account.

## Was the "fit-split bias" risk from the task's Path A design section confirmed?

**Yes, directionally, but not via the specific mechanism anticipated.** The
task flagged that Path A's EMA is estimated from FIT-split data (the model
already been trained to fit well), unlike Path B/H/E's genuinely held-out
calib split, and asked whether this measurably shows up as worse numbers.
It does -- Path A is worse than Path B everywhere B is affected, and worse
than the `global` baseline everywhere too. But the earlier section's analysis
suggests the dominant failure mode is less "EMA variance is a biased
underestimate of true variance" (the literal fit-vs-calib-split argument) and
more "EMA per-prototype statistics become internally inconsistent across
training/federated rounds," a related but mechanically distinct problem that
would need fixing even if the fit-split bias itself were somehow eliminated
(e.g. by running the EMA update on a genuinely held-out stream, which this
implementation does not do -- see "What's still open"). **Update: this
inconsistency was confirmed and fixed -- see "Bug fix (2026-08-21)" below.
Sielaff's regression turned out to be almost entirely this bug (full
recovery to `global` once fixed); robo3er's is a mix of this bug (partially
fixed) and a smaller residual gap consistent with the fit-split bias
originally hypothesized here.**

## Bug fix (2026-08-21): `load_memory()` wasn't resetting EMA state

The "most likely fixable culprit" flagged above was confirmed and fixed.
`JointPrototypeMemory.load_memory()` (`joint_prototype_model.py`) only ever
zeroed `usage_count` on each federated round's codebook swap -- `ema_mean`,
`ema_var`, `ema_initialized`, `ema_step`, and the global EMA fallback buffers
all silently carried over. Since `load_memory()` replaces the codebook with a
completely different set of vectors every round, prototype slot `m`'s
"meaning" changes each time, but the stale EMA stats (indexed only by `m`,
blended at `ema_decay=0.99`, i.e. 99% weight on the old value) kept being
mixed into statistics computed under the NEW codebook geometry. Worse,
`ema_step` was never reset either, so `ema_warmup_steps`' "don't trust early
noisy assignments" gate only ever fired once, at the very start of round 1 --
every subsequent round's codebook swap got zero warm-up protection.

**Fix**: `load_memory()` now also resets `ema_step`/`ema_mean`/`ema_var`/
`ema_initialized`/`ema_global_*` to their initial values, so each round has
to earn its own warm-up again. `ema_warmup_steps` was also lowered from 200
to 20 (default), since robo3er's smallest federated clients (~103-247 fit
windows, `batch_size=256`) only produce ~12 local-epoch steps per round --
200 was never reachable within a single round even before this fix's reset
made per-round warm-up mandatory.

**Result: Sielaff fully recovers to match `global`; robo3er improves but
still trails.**

| Dataset / script | Signal | broken `ema` (pre-fix) | fixed `ema` | `global` (unaffected) |
|---|---|---|---|---|
| robo3er v3_1 | B | 0.769 | 0.836 | 0.945 |
| robo3er v3_1 | B, cable_trapped | 0.791 | 0.904 | 0.994 |
| robo3er v3_1 | B, stuck | 0.746 | 0.769 | 0.519 (central) / 0.933 (fed, see note) |
| robo3er forecast_v2 | B | 0.782 | 0.846 | 0.948 |
| Sielaff v2_1 | B | 0.935 | **0.974** | 0.974 |
| Sielaff forecast_v2 | B | 0.966 | **0.974** | 0.974 |

(robo3er's `global` `stuck` figure needs care -- 0.933/0.926 is what
[[scoring-signals-B-C-E-H]] reports for federated B/C on `stuck`; this
branch's own `global`-mode `robo3er_v3_1_federated` run reports overall mean
0.945 with per-fault-type numbers consistent with that; treat the table's
"0.945"/"0.948" row as the right global comparison point, not a literal
per-fault cell mismatch.)

**Sielaff's full recovery (0.974 vs 0.974, to 3 decimal places on the v2_1
script) is strong evidence the reset bug, not the fit-split-bias risk the
task originally asked about, was the dominant cause of Path A's regression
on Sielaff** -- once the EMA statistics are no longer contaminated across
round boundaries, they converge to essentially the same answer as the
offline global median/IQR.

**robo3er's partial recovery (0.836/0.846, still ~10pt below `global`) has a
different, still-open explanation: most robo3er clients still don't clear
even the lowered warm-up within a single round.** `robot00`/`robot01`/
`robot02`/`robot03` have 103-247 fit windows -- at `batch_size=256` that's
exactly 1 batch/epoch, so 12 local epochs = 12 steps/round, still below
`ema_warmup_steps=20`. Only `robot04` (fit=3231, ~13 batches/epoch, ~156
steps/round) ever clears warm-up and gets genuine per-prototype EMA stats;
the four small clients spend the whole run on the (now correctly
per-round-reset, but still not per-prototype) global EMA fallback. `stuck`
(robot04's fault) improved only marginally (0.746→0.769) even though its
client DOES reach genuine per-prototype EMA -- consistent with the
originally-hypothesized fit-split bias being real but secondary, showing up
specifically once the reset bug is no longer masking it. `cable_trapped`
(robot01's fault, small client, global-fallback-only) improved much more
(0.791→0.904) purely from no longer being cross-contaminated by stale
cross-round statistics, without ever getting real per-prototype treatment.

**Not attempted**: lowering `ema_warmup_steps` further (e.g. to 5) to let
robo3er's small clients reach genuine per-prototype EMA too -- risks noisier
variance estimates from even fewer samples, the same regime that plausibly
hurt Path B on Sielaff's 16-prototype setup. Flagged as the next concrete
experiment if this line continues, not run here.

## Recommendation

- **robo3er: switch to `per_prototype` (Path B) for B/C.** Small, real,
  concentrated on `stuck` -- consistent with H's own established mechanism,
  no downside observed on `cable_trapped`. `global` remains fine as a
  simpler default if this gain isn't worth the added `set_score_calibration()`
  plumbing for a given use case.
- **Sielaff: keep `global` (do NOT switch to `per_prototype`).** Still a
  clear regression for B/C even after the `ema` fix (per_prototype's
  regression was never about the `load_memory()` bug -- it's Path B's own
  offline calibration seeing too few calib windows per prototype at
  Sielaff's 16-prototype scale, a separate, still-undiagnosed issue).
- **`ema` (Path A), post-fix: viable on Sielaff (matches `global` exactly),
  still not recommended on robo3er (10pt behind `global`/`per_prototype`).**
  The fix confirmed the `load_memory()`-reset interaction was real and was
  the dominant cause of the ORIGINAL blanket "don't use ema anywhere"
  verdict -- that verdict no longer holds for Sielaff. robo3er's residual gap
  is a different, still-open problem (most clients too small to clear
  warm-up within a round), not the bug just fixed.

## What's still open

- **Paderborn was NOT run** (lowest priority per the task's own ordering,
  cut for time -- explicitly flagged, not silently skipped). Paderborn's
  baseline table in [[scoring-signals-B-C-E-H]] shows C/E as the dominant
  signals there (B is comparatively weak, 0.888) -- whether per-prototype/EMA
  calibration changes that picture at all is untested.
- **Diagnosis/localization AUROC was NOT re-run** for any calib_mode (task
  marked this lower priority than detection AUROC) -- [[three-dataset-bck-comparison]]'s
  existing diagnosis numbers are all `global`-mode; whether per-prototype
  calibration changes argmax localization (plausible, since it changes the
  per-node z-score magnitudes that argmax reads) is a completely open
  question.
- **Path A's EMA decay (0.99) and warm-up (200 steps) are unswept
  hyperparameters** -- picked by judgment, not tuned, per the task's own
  explicit permission to flag this rather than sweep it. Given Path A's
  clear net-negative result, a sweep was not prioritized this round; if Path
  A is revisited, the `load_memory()`-reset interaction (see above) should be
  fixed BEFORE any hyperparameter sweep, since it's a more likely source of
  the observed damage than decay/warmup choice.
- **Path A's C/K extension was scoped out** (task said "extend to
  resid_struct/k_resid too if straightforward, but d_node/B is the
  priority") -- no online EMA hook exists for either signal; `ema` mode
  silently no-ops to `global` for both in every script. Not attempted.
- **Per-prototype's Sielaff regression mechanism (small per-prototype calib
  counts vs. `min_samples=5`) was not directly diagnosed** -- flagged above
  as the most likely cause but not verified by inspecting the actual
  per-prototype window-count distribution. **Update:** partially diagnosed
  by [[sielaff-num-prototypes-sweep]], which swept `NUM_PROTOTYPES` on
  Sielaff and found the hypothesis confirmed IN PART (the fraction of valid
  per-prototype calib estimates tracks the AUROC regression closely across
  the grid) but NOT the complete explanation (even at `NUM_PROTOTYPES=2`,
  with 60% of prototype slots individually valid, `per_prototype` still
  underperforms `global` by a real margin) -- no `NUM_PROTOTYPES` value
  makes `per_prototype` beat `global` on Sielaff, so the recommendation to
  keep `global` for Sielaff (below) stands unchanged.
- Single seed everywhere, per this project's standing convention -- every
  delta above (especially the smaller ones, e.g. robo3er's F/I/J moving by
  ~0.001-0.003) should be read with that in mind; the large, clearly
  directional deltas (Path A's B regressions, Path B's Sielaff regressions)
  are unlikely to be noise, but exact magnitudes are single-run estimates.

See also [[scoring-signals-B-C-E-H]] (H/E's existing per-prototype
calibration pattern this branch generalizes to B/C/K),
[[forecast-head-signal-k]] (signal K background), and
[[three-dataset-bck-comparison]] (the `global`-mode baseline table this
branch's numbers are directly comparable to).

## `shrinkage`: empirical-Bayes per-client, per-prototype blending (2026-08-21)

`federated_ema` (implemented, tested, and reverted earlier the same day --
recoverable via `git show 916ade0`/`git show 6093bee` -- see "Why
federated_ema failed" above) fused every client in a shared-cluster into
ONE identical statistic, weighted by sample count -- robo3er's huge
client-size imbalance (robot01 fit=103 vs. robot04 fit=3231) meant robot04
essentially overwrote robot01's own real `cable_trapped` statistics, and
AUROC regressed (0.904->0.852 in one script). This section implements the
agreed follow-up: a PER-CLIENT empirical-Bayes shrinkage blend instead of
one shared fused value.

### Design actually implemented

For each prototype slot `k`, each client `c` gets its OWN blended
statistic:

```
theta*_{c,k} = lambda_{c,k} * theta_local_{c,k} + (1 - lambda_{c,k}) * theta_global_k
lambda_{c,k} = n_{c,k} / (n_{c,k} + alpha)
```

applied separately to mean and (diagonal-only, per the task's explicit
scope limit -- no full node x node covariance version) variance of
`d_node`. `n_{c,k}`, `mean_{c,k}`, `var_{c,k}` come from a ONE-SHOT local
pass over each client's own fit split
(`federated_memory.compute_prototype_dev_stats`) -- not a decayed EMA.

**One deliberate placement choice, different from `federated_ema`'s: the
one-shot local pass happens AFTER `align_and_split` + `load_memory()`, not
before.** This matters because it makes prototype index `k` mean the SAME
physical prototype for every client this round for the shared slots
(`k < num_shared`, `align_and_split`'s `P_S` -- identical vector content
broadcast to everyone), so a fleet-wide pool (across literally every
client with `n_{c,k} > 0`, NOT restricted to `align_and_split`'s original
clustering decision -- a deliberate broadening the task explicitly called
for, since the shrinkage target is a fleet-wide prior, not cluster-specific
reconciliation) is a coherent thing to compute. For the personalized slots
(`k >= num_shared`), index `k`'s content is DIFFERENT per client by
construction, so fleet pooling by raw index there would blend together
unrelated prototypes -- `lambda=1` (pure local, no cross-client borrowing)
is used instead, matching `federated_ema`'s own treatment of personalized
slots. This is a documented deviation from a completely literal "pool
every client with data for that index" rule, necessary because that rule
is only semantically valid for the shared slots. New code:
`federated_memory.compute_prototype_dev_stats`/`_merge_dev_stats_pair`
(re-added, adapted from the reverted `federated_ema` commit) and the new
`compute_shrinkage_stats`; `JointPrototypeMemory.load_shrinkage_stats()`
loads the per-client blended result into the same `ema_mean`/`ema_var`/
`ema_initialized` buffers `ema_zscore()` already reads, so scoring is
unchanged. Wired into all 4 scripts as `--calib-mode shrinkage --alpha N`
(alpha default 20, swept below). `resid_struct`/C and `k_resid`/K have no
shrinkage hook (same scope limit as `ema`), fall back to `global`.
Regression-verified `global`/`per_prototype`/`ema` stay bit-identical
(`robo3er_v3_1_federated`'s `global` run reproduced its committed report
to the full float, 0.0 diff on every column) before adding any new code
path's numbers below.

### Alpha sweep (robo3er, single seed, `alpha` in {5, 20, 50})

`num_shared_prototypes` was 0 in early rounds and settled at 1 (of
`NUM_PROTOTYPES=2`) by round 3-5 in every run here, matching the same
alignment pattern already documented above for `per_prototype` mode.

**robo3er v3_1 (`run_robo3er_v3_1_federated.py`), B only (C/E/H unaffected
by construction):**

| calib_mode | B (overall) | B, cable_trapped (robot01) | B, stuck (robot04) |
|---|---|---|---|
| global | 0.945 | 0.957 | 0.933 |
| per_prototype | 0.951 | 0.957 | 0.944 |
| ema (post-fix) | 0.836 | 0.904 | 0.769 |
| federated_ema (reverted) | -- | 0.852 | -- |
| shrinkage, alpha=5 | 0.839 | 0.848 | 0.830 |
| shrinkage, alpha=20 | 0.828 | 0.827 | 0.830 |
| shrinkage, alpha=50 | 0.825 | 0.820 | 0.830 |

**robo3er forecast_v2 (`run_robo3er_forecast_v2_federated.py`), B only:**

| calib_mode | B (overall) | B, cable_trapped (robot01) | B, stuck (robot04) |
|---|---|---|---|
| global | 0.948 | 0.959 | 0.936 |
| per_prototype | 0.948 | 0.959 | 0.936 |
| ema (post-fix) | 0.846 | 0.915 | 0.778 |
| shrinkage, alpha=5 | 0.832 | 0.868 | 0.796 |
| shrinkage, alpha=20 | 0.826 | 0.856 | 0.796 |
| shrinkage, alpha=50 | 0.823 | 0.851 | 0.796 |

**Sielaff (one alpha value, 20, both scripts -- B only, C/H unaffected):**

| script | calib_mode | B |
|---|---|---|
| v2_1 | global | 0.974 |
| v2_1 | per_prototype | 0.886 |
| v2_1 | ema | 0.974 |
| v2_1 | shrinkage, alpha=20 | 0.946 |
| forecast_v2 | global | 0.974 |
| forecast_v2 | per_prototype | 0.924 |
| forecast_v2 | ema | 0.974 |
| forecast_v2 | shrinkage, alpha=20 | 0.960 |

### Verdict: shrinkage does NOT cleanly beat both of `ema`'s and `federated_ema`'s extremes on robo3er -- it lands BETWEEN them, but on the WRONG side of `ema` for the specific client/fault this design targeted

This is the opposite of the hoped-for outcome, stated plainly. Point by
point:

- **robot01's `cable_trapped` gets WORSE than `ema` (0.904) at every alpha
  tried, not better** -- 0.848 (alpha=5) down to 0.820 (alpha=50) in the
  v3_1 script, 0.868->0.851 in the forecast script. It stays above
  `federated_ema`'s fully-pooled 0.852 floor only at the weakest shrinkage
  tested (alpha=5, v3_1 script: 0.848 vs 0.852 -- actually BELOW
  federated_ema's own number here, single-seed noise aside) -- shrinkage
  does not reliably improve on `federated_ema`'s known failure mode for
  this specific client/fault, contrary to the design's purpose.
- **robot04's `stuck` DOES improve over `ema`** (0.769->0.830 in v3_1,
  0.778->0.796 in forecast) at every alpha, and is flat across the whole
  alpha range -- consistent with robot04 being the large client whose OWN
  local statistic already dominates any pool it's part of, so blending in
  a small fleet contribution barely moves it either way.
- **Alpha barely matters within the swept range (5/20/50) for either
  client** -- B moves by at most ~0.03 across the whole 10x alpha range,
  on both the overall mean and the two individual faults. Diagnosed (not
  fully proven, see caveat below) via a direct dump of `n_{c,k}` at each
  round: by the final round, the ONE shared slot's fleet-wide `n` pool is
  `[97, 65, 52, 88, 3208]` for clients 0-4 (robot04 alone contributes
  ~91% of the pooled sample count) -- robot01's own `n=65` at that slot
  gives `lambda` ranging only 0.57-0.93 across alpha=50->5, a real swing
  in principle, but the resulting AUROC barely moves. The most likely
  explanation (not directly verified by inspecting per-fault-window `idx`
  assignments, flagged as the clearest follow-up if this line continues):
  robot01's `cable_trapped` FAULT windows themselves probably route
  predominantly to this same fleet-dominated shared slot at test time
  (not the personalized slot, which stays purely local/unaffected by
  alpha), so ANY nonzero pull toward robot04's very different
  normal-deviation scale already does most of the damage `federated_ema`
  did, and adding a large-alpha-driven local weight back on top isn't
  enough to fully cancel it out -- this would explain both the
  flat-vs-alpha response and why even alpha=5 (mostly-local) doesn't
  recover close to `ema`'s no-cross-client-contamination number.
- **On Sielaff, shrinkage is a partial recovery over `per_prototype`**
  (0.886->0.946 in v2_1, 0.924->0.960 in forecast) **but a small
  regression vs. plain `ema`** (0.974->0.946 in v2_1, 0.974->0.960 in
  forecast) -- `ema` was already a near-perfect match to `global` on this
  dataset (see "Bug fix" section above), so shrinkage's fleet-wide
  pooling step, even gated by `lambda`, reintroduces a small amount of the
  same cross-client-contamination risk `ema`'s pure-local design avoided
  by construction. Not a regression vs. `global`/`per_prototype`'s
  original problem, but not an improvement over the already-working `ema`
  fix either.

**Net assessment: on this single seed, `shrinkage` is a real, working
implementation of the agreed design, but it does not achieve the specific
hoped-for outcome (robot01's `cable_trapped` not regressing the way it did
under `federated_ema`, while robo3er's overall B improves over `ema`).**
Overall B on robo3er is WORSE than `ema` at every alpha tested (0.825-0.839
vs. `ema`'s 0.836/0.846), because the one large client (robot04/`stuck`)
improves less than the small client (robot01/`cable_trapped`) regresses.
The mechanism suspected (small client's own fault-time routing lands in
the same fleet-dominated slot as its fit-time routing, so shrinkage cannot
avoid inheriting the large client's very different scale once ANY
fleet-pool weight enters) is a genuine, previously-undocumented risk of
this design, distinct from both `ema`'s warm-up problem and
`federated_ema`'s full-overwrite problem.

### What's still open

- **Not directly verified**: whether robot01's `cable_trapped` fault
  windows really do route predominantly to the fleet-dominated shared
  slot at test/fault time (only the FIT-time routing was inspected here,
  via a temporary monkey-patched debug run of `compute_shrinkage_stats`).
  If confirmed, the natural next fix would be gating shrinkage differently
  -- e.g. only blending when the SHARED slot's fleet composition isn't as
  lopsided as `[97, 65, 52, 88, 3208]`, or shrinking toward a
  robot04-EXCLUDED pool for very small clients -- neither attempted here.
- **Alpha sweep was narrow (5/20/50) and all three landed close together**
  -- a wider or denser sweep (e.g. alpha=1 for near-zero shrinkage, alpha
  in the hundreds for near-total pooling) was not run; given the observed
  flat response, it's not obvious a different alpha in a wider range would
  change the qualitative verdict, but this wasn't tested.
- **Sielaff was only run at alpha=20**, per the task's own lower-priority
  ordering for that dataset -- an alpha sweep there (mirroring the
  robo3er one) was not attempted.
- **Paderborn was not run** (same standing lower-priority note as the rest
  of this document).
- **Diagnosis/localization AUROC was not re-run** (same standing
  lower-priority note as the rest of this document).
- Single seed everywhere, per this project's standing convention -- read
  every delta above with that in mind, especially the ~0.03 alpha-range
  deltas, which are close to the kind of noise floor seen elsewhere in
  this document's smaller deltas.

See also the "Why federated_ema failed" section above (this section's
starting point) and [[scoring-signals-B-C-E-H]]/[[three-dataset-bck-comparison]]
(baseline tables this section's numbers are directly comparable to).
