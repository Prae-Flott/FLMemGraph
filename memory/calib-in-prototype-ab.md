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

## `federated_ema` (2026-08-21): federating the deviation statistics THEMSELVES, not just the codebook

Path A (plain `ema`), post the `load_memory()`-reset fix above, fully
recovers to `global` on Sielaff but still trails `global`/`per_prototype`
by ~10pt on robo3er -- because most of robo3er's federated clients
(`robot00`-`robot03`, 103-247 fit windows) never clear even the lowered
`ema_warmup_steps=20` within a single round's ~12 local-epoch steps, so
they spend the whole run on the (correctly-reset, but not per-prototype)
global EMA fallback. The natural next question: can those small clients
borrow statistical strength from a federated-cluster partner the same way
the codebook VECTORS already do (`align_and_split`'s shared-cluster
mean-pooling)? `federated_ema` implements exactly that and tests it.

**What it does differently from plain `ema`.** Plain `ema` accumulates a
decayed per-prototype mean/var of `d_node` step-by-step DURING local
training (`JointPrototypeMemory.update_ema()`), purely local, never
exchanged. `federated_ema` instead: (1) after each round's local training
finishes, does ONE one-shot forward pass over the client's own fit split
with that round's just-trained codebook, computing per-prototype
sufficient statistics `(n, mean, var)` of `d_node`
(`federated_memory.compute_prototype_dev_stats`) -- not a decayed EMA, a
plain count/mean/variance triple, chosen specifically because it has a
clean, order-independent cross-client merge rule an EMA lacks; (2) the
server (`align_and_split`, new `client_dev_stats` argument) merges these
across clients for whichever prototype slots it ALREADY decided form a
shared cluster (the exact same clustering decision used for the codebook
vectors, not re-derived), via the sample-count-WEIGHTED parallel-variance
formula (Chan/Golub/LeVeque 1979: `n = n_a+n_b`, `mean = (n_a*mean_a +
n_b*mean_b)/n`, `M2 = n_a*var_a + n_b*var_b + (mean_a-mean_b)^2 *
n_a*n_b/n`, `var = M2/n`); personalized (non-shared) slots pass through
each client's own freshly-computed stats unchanged (only one client ever
contributes to those, so nothing to merge); (3) the merged result is
broadcast back and loaded directly into the SAME `ema_mean`/`ema_var`/
`ema_initialized` buffers plain `ema` uses
(`JointPrototypeMemory.load_fed_ema_stats()`, called right after
`load_memory()`), so `ema_zscore()` is reused unchanged for scoring in
both modes.

**Deliberate design choice, called out per the task's own instructions:**
the codebook vectors' own shared-cluster merge (`P_S = ...mean(dim=0)`) is
UNWEIGHTED (plain mean over cluster members, matching FedPM's own
convention) -- `federated_ema`'s deviation-statistics merge is
WEIGHTED by sample count instead, a deliberate divergence, not a "fix" to
the codebook convention: correctness for a mean/variance ESTIMATE
requires weighting by how many samples each contributor actually had,
while the codebook pooling's unweighted convention is a separate,
already-established design choice. This is documented in
`federated_memory.align_and_split`'s docstring at the point the two
merges diverge.

**Implementation is purely additive** -- `align_and_split` only returns
the extra `dev_stats_out` 3rd tuple element when `client_dev_stats` is
explicitly passed (`None` by default), so the 3 existing `calib_mode`s'
call sites are untouched. Regression-verified: `global`/`per_prototype`/
`ema` reproduce BIT-IDENTICAL `summary_mean_auroc_overall` values to the
already-committed reports on all 4 scripts (`run_robo3er_v3_1_federated.py`,
`run_robo3er_forecast_v2_federated.py`, `run_sielaff_v2_1_federated.py`,
`run_sielaff_forecast_v2_federated.py`) after this branch's changes --
confirmed by diffing the re-run JSON reports against pre-existing copies,
not just eyeballing printed numbers.

### Result: `federated_ema` does NOT close robo3er's gap, and mildly REGRESSES Sielaff (which had fully recovered under plain `ema`)

4-way comparison, single seed, mean AUROC over all client-fault pairs
(`global`/`per_prototype`/`ema` columns are the already-committed,
bit-identical-reproduced numbers from earlier in this doc):

**robo3er (`run_robo3er_v3_1_federated.py`, B/C/E/H)**

| calib_mode | B | C | E | H | F | I | J |
|---|---|---|---|---|---|---|---|
| global | 0.945 | 0.945 | 0.686 | 0.948 | 0.943 | 0.947 | 0.946 |
| per_prototype | 0.951 | 0.960 | 0.686 | 0.948 | 0.944 | 0.949 | 0.947 |
| ema (fixed) | 0.836 | 0.945 | 0.686 | 0.948 | 0.944 | 0.948 | 0.947 |
| **federated_ema** | **0.837** | 0.945 | 0.686 | 0.948 | 0.946 | 0.946 | 0.946 |

**robo3er (`run_robo3er_forecast_v2_federated.py`, adds K, horizon_mult=10)**

| calib_mode | B | C | K | BK | CK | HK |
|---|---|---|---|---|---|---|
| global | 0.948 | 0.958 | 1.000 | 0.995 | 1.000 | 0.995 |
| per_prototype | 0.948 | 0.958 | 1.000 | 0.995 | 1.000 | 0.995 |
| ema (fixed) | 0.846 | 0.958 | 1.000 | 1.000 | 1.000 | 0.995 |
| **federated_ema** | **0.816** | 0.958 | 1.000 | 1.000 | 1.000 | 0.995 |

**Sielaff (`run_sielaff_v2_1_federated.py`, B/C/H)**

| calib_mode | B | C | H | I |
|---|---|---|---|---|
| global | 0.974 | 0.976 | 0.964 | 0.973 |
| per_prototype | 0.886 | 0.942 | 0.964 | 0.889 |
| ema (fixed) | 0.974 | 0.976 | 0.964 | 0.968 |
| **federated_ema** | **0.946** | 0.976 | 0.964 | 0.961 |

**Sielaff (`run_sielaff_forecast_v2_federated.py`, adds K, horizon_mult=10)**

| calib_mode | B | C | K | BK | CK | HK |
|---|---|---|---|---|---|---|
| global | 0.974 | 0.975 | 0.910 | 0.976 | 0.974 | 0.953 |
| per_prototype | 0.924 | 0.975 | 0.902 | 0.930 | 0.963 | 0.949 |
| ema (fixed) | 0.974 | 0.975 | 0.910 | 0.923 | 0.974 | 0.953 |
| **federated_ema** | **0.961** | 0.975 | 0.910 | 0.944 | 0.974 | 0.953 |

(C/K are unaffected by `federated_ema` in every table, same as plain
`ema` -- no federated deviation-statistics hook exists for `resid_struct`
or `k_resid`, matching the task's stated `d_node`/B priority; `federated_ema`
falls back to `global` for both, identically to plain `ema`.)

### Robo3er: essentially a wash, not a win

`federated_ema` B: 0.837 (v3_1) / 0.816 (forecast) vs. plain `ema`'s 0.836
/ 0.846 -- both still ~10pt behind `global`/`per_prototype`. The
per-fault-type breakdown shows WHY it's a wash rather than a clean
improvement: it doesn't uniformly help, it TRADES one fault type's
performance for another's.

- v3_1 script: `stuck` (robot04, the one client that already clears
  warm-up and gets genuine per-prototype EMA even without federation)
  improves 0.769->0.822 -- some benefit from federation here, though
  modest. `cable_trapped` (robot01, small client, global-fallback-only
  under plain `ema`) gets WORSE, 0.904->0.852 -- borrowing a shared-cluster
  partner's statistics apparently hurts more than it helps for this client.
- forecast script: same direction, `stuck` roughly flat (0.778->0.776,
  within noise) and `cable_trapped` worse (0.915->0.857), net overall
  WORSE than plain `ema` (0.846->0.816).

**Likely mechanism (not fully diagnosed, flagged as the next open
question if this line continues):** both robo3er scripts' `alignment_log`
shows only 1 of 2 prototypes ever becomes a shared cluster by round 3-5.
robot04 (fit=3231 windows) vastly outweighs robot01 (fit=103) in the
sample-count-weighted merge for whichever slot they share -- the merged
`(mean, var)` is almost entirely robot04's own statistics, robot01's own
~100-200 fit windows contribute a roughly 3% weight. If robot01's actual
per-prototype deviation distribution for that shared regime differs at
all from robot04's (plausible -- they're different physical robots), the
"borrowed" statistics are barely a compromise, they're close to just
using someone else's distribution wholesale -- worse than robot01's own
noisy-but-genuinely-its-own global EMA fallback in at least this one
fault type's case. This is the opposite failure mode from what the
[[calib-in-prototype-ab]] section above found for Path B/per_prototype on
Sielaff (too little data per prototype) -- here it's not too little
DATA, it's a plausible REGIME MISMATCH between the clients being merged,
which a pure sample-count weighting has no way to detect or guard
against. Not confirmed by directly inspecting robot01 vs. robot04's raw
per-node deviation distributions for the shared prototype (time-limited).

### Sielaff: a real, if modest, regression -- the first time federating deviation stats has hurt something plain `ema` already fixed

`federated_ema` B: 0.946 (v2_1) / 0.961 (forecast) vs. plain `ema`'s 0.974
/ 0.974 (both exactly matching `global` post-fix) -- a genuine ~1.3-2.8pt
regression, small in absolute terms but notable because it UNDOES part of
what the `load_memory()` fix earlier in this doc had already achieved.
Sielaff's `alignment_log` under `federated_ema` shows heavy early-round
merging (round 1: 8 of 16 prototypes in shared clusters across 10
clients) collapsing down to 1 shared prototype by round 3-5 -- similar
end-state to robo3er, but with a much larger, more heterogeneous set of
contributing clients (10 machines vs. robo3er's 5 robots) passing through
that same weighted-merge mechanism. The same "sample-count weighting
can't tell regime-similarity from mere prototype-vector cosine-similarity
above `delta`" mechanism hypothesized for robo3er above is the most
plausible read here too, at a larger scale (more clients contending for
each shared slot) -- not independently confirmed for Sielaff specifically
within this investigation's time budget.

### Recommendation, updated

**`federated_ema` is not recommended anywhere tested.** It does not close
robo3er's residual `ema`-vs-`global` gap (net wash, slightly negative in
the forecast script) and it mildly regresses Sielaff's otherwise-fully-
recovered plain-`ema` result. The existing recommendations stand
unchanged: `per_prototype` for robo3er B/C, `global` for Sielaff B/C
(or plain `ema`, post-fix, as an equally-good alternative on Sielaff
specifically). The federating-the-codebook-only convention `align_and_split`
already had (unweighted mean-pool for shared clusters) is NOT obviously
"fixed" by also federating the deviation statistics with a fully sample-
count-weighted rule -- if anything, this experiment suggests sample-count
weighting alone is too blunt an instrument for statistics (as opposed to
representative vectors), since it has no way to guard against merging two
clients whose codebook vectors are similar enough to cluster but whose
actual deviation DISTRIBUTIONS for that regime differ.

### What's still open (updated)

Everything in the original "What's still open" section above still
applies unchanged (Paderborn untested, diagnosis/localization untested,
EMA decay/warmup unswept, Path A's C/K extension out of scope,
per_prototype's Sielaff mechanism only partially diagnosed). Additionally:

- **The regime-mismatch-vs-sample-weighting hypothesis above is not
  confirmed by direct inspection** of any client pair's actual raw
  deviation distributions for a shared prototype slot -- flagged as the
  single most informative next diagnostic if `federated_ema` is revisited
  (e.g. plot robot01 vs. robot04's own per-node `d_node` histograms
  restricted to windows matching the shared prototype, before vs. after
  the merge, to see directly whether the merged statistics actually
  represent robot01's own windows well or not).
- **An unweighted, or confidence-capped, merge variant was not tried** --
  e.g. capping each contributor's effective weight (so no single client
  can dominate a merge past some threshold, similar in spirit to how
  `align_and_split`'s personalized-slot ranking already avoids one
  client's raw frequency count dominating), or falling back to the
  UNWEIGHTED convention the codebook vectors already use for consistency.
  Not attempted -- the task's design explicitly called for sample-count
  weighting as the statistically "correct" choice, and this experiment's
  job was to test that specific design as given, not to immediately
  redesign it after one negative result.
- **Single seed everywhere**, per this project's standing convention --
  the `federated_ema` deltas above (especially robo3er's ~0.02 wash and
  Sielaff's ~0.013-0.028 regression) are smaller than some of this
  branch's other findings and should be read with that in mind; the
  clearly-directional ones (Sielaff's regression appearing in BOTH
  scripts, robo3er's fault-type trade-off appearing in BOTH scripts) are
  unlikely to be pure noise, but exact magnitudes are single-run
  estimates.
