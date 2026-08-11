# Paderborn bearing: single-machine full pipeline run (2026-08-06)

## What was built

`benchmark/datasets/paderborn_adapter.py` (loader) +
`benchmark/run_paderborn_fl_model.py` (driver) -- this project's full
"记忆检索 + 物理验证" pipeline minus federation, on the Paderborn KAt
dataset, reusing `src/fl_model.FLGDNMemory` UNCHANGED (the Sec-2A/2B
architecture already validated on robo3er and on the deleted IMS bearing
exploration). Unlike IMS, this dataset has DEFINITE per-bearing damage
ground truth, so the evaluation is this project's NORMAL convention:
fit + calibrate on healthy bearings only (K001-K006, pooled, 70/15/15
split per bearing then pooled -- matches Sielaff's per-source-then-pooled
pattern), evaluate AUROC per damaged bearing code against the pooled
healthy test_normal split.

**Two data-loading bugs found and fixed while building the adapter**
(both real, not assumptions -- verified against the actual files):
1. Per-file sample counts are NOT constant (`vibration_1` etc. range
   249,940 to 256,744+ across the 2560 files; most but not all are
   exactly 256,001). The initial `RAW_LEN=256000` assumption crashed on
   the first file below that. Fixed: `RAW_LEN=240000`, safely below the
   observed minimum, divisible by the decimation factor.
2. One file, `KA08/N15_M01_F10_KA08_2.mat`, fails to parse with
   `scipy.io.loadmat` (`TypeError: Expecting matrix here`) -- not a
   download/extraction truncation (file size is normal), the internal
   MAT5 struct layout itself is malformed. `load_bearing()` now skips
   unreadable files with a printed warning instead of crashing (1 of
   2560 files project-wide, KA08 has 79 not 80 files as a result).

## What's fed to the model (a stated simplification, not the full 7 channels)

Only the three ~64kHz channels (`vibration_1`, `phase_current_1`,
`phase_current_2`) -- see the adapter's docstring for why the 4kHz
mechanical channels (force/speed/torque) and 1Hz temperature aren't
loaded yet. Each 4-second file is box-average decimated 240,000 -> 1,920
samples (factor 125, ~512Hz effective rate) -- the SAME crude
box-average technique (not a designed anti-aliasing filter) already
flagged as a real limitation in the deleted IMS bearing work, and, per
the results below, plausibly responsible for a genuinely interesting
finding here too.

`FLGDNMemory` reconstructs overlapping 64-sample sub-windows (stride 32)
of that decimated sequence -- 19,824 training windows from 336 fit files
(6 healthy bearings x 56 files each). `M=32` prototypes, `embed_dim=64`,
`beta=0.25`, `lambda=0.5` (all matching the values already established
for the bearing architecture-gap fix). 10 epochs, codebook utilization
reached 1.00 (all 32 prototypes used) by epoch 1 and stayed there --
worth noting given M=32 was carried over from a much smaller fit set
(IMS's ~300 files) without retuning for this dataset's larger one.

## Results: category means look strong, but hide a real bimodal split

| damage category | n bearings | mean AUROC(d) | mean AUROC(r) | mean AUROC(d+r) |
|---|---|---|---|---|
| combined (IR+OR) | 3 | 1.000 | 1.000 | 1.000 |
| inner_ring | 11 | 0.750 | 0.803 | 0.742 |
| outer_ring | 12 | 0.641 | 0.669 | 0.617 |

These category averages look like a clean success. But the per-bearing
table is NOT a tight cluster around these means -- it's sharply bimodal:
**15 of 26 damaged bearings hit AUROC(d+r) >= 0.999 (essentially
perfect)**, while **10 of 26 are BELOW 0.5** (worse than random under the
score's stated "higher = more anomalous" direction):

| bearing | category | origin | AUROC(d+r) | flipped (1-AUROC) |
|---|---|---|---|---|
| KA01 | outer_ring | artificial | 0.455 | 0.545 |
| KA05 | outer_ring | artificial | 0.245 | 0.755 |
| KA06 | outer_ring | artificial | 0.324 | 0.676 |
| KA07 | outer_ring | artificial | 0.260 | 0.740 |
| KA08 | outer_ring | artificial | 0.262 | 0.738 |
| KA30 | outer_ring | real | 0.286 | 0.714 |
| KI05 | inner_ring | artificial | 0.254 | 0.746 |
| KI07 | inner_ring | artificial | 0.256 | 0.744 |
| KI08 | inner_ring | artificial | 0.292 | 0.708 |
| KI04 | inner_ring | real | 0.360 | 0.640 |

**The genuinely interesting part**: this isn't noise clustered around
0.5 (which is what "no signal" looks like) -- it's confidently WRONG in
one consistent direction, and when flipped, most of these recover to a
respectable 0.64-0.76 AUROC. **8 of these 10 inverted-score bearings are
ARTIFICIAL damage** (only KA30 and KI04 are real); conversely, of the 15
near-perfect bearings, most are either real-damage or two specific
artificial ones (KA03, KI01, KI03 -- notably, KA03 was made by electric
engraver like several of the failing ones, so damage METHOD alone
doesn't cleanly explain it either).

## Leading hypothesis (stated as hypothesis, not verified this session)

Artificial damage (EDM/electric engraver/drilling) creates a sharp,
localized, single-point defect -- its vibration signature is dominated by
brief high-frequency IMPACT/resonance content each time a rolling element
crosses it (the same physics discussed for the deleted IMS work's
BPFO/BPFI analysis). Real fatigue damage from the accelerated-lifetime
test tends to be rougher/more distributed pitting, whose effect on
vibration is plausibly more broadband/lower-frequency. The 125x
box-average decimation used here is a crude low-pass filter -- it would
disproportionately wash out exactly the sharp high-frequency impulsive
content artificial single-point defects rely on, while leaving real
damage's more broadband signature comparatively intact. If the decimated,
smoothed signal from an artificially-damaged bearing ends up LESS
variable / more "regular" than a healthy bearing's natural mechanical
noise floor at this coarse resolution, the model would score it as MORE
normal, not less -- exactly the inverted pattern observed. This is a
coherent, physically-grounded explanation, consistent with the
decimation-as-limitation concern already flagged for IMS, but it is a
hypothesis inferred from the pattern, not confirmed by a targeted
follow-up experiment (e.g. re-running a subset of the inverted bearings
at a much finer decimation factor or with actual envelope-spectrum
features instead of raw box-averaged amplitude).

## What's NOT done (open follow-ups)

- The decimation-washes-out-impulsive-content hypothesis above is
  untested -- rerun a few of the inverted bearings (KA05/06/07/08,
  KI05/07/08) with a much finer decimation factor, or with the
  envelope-spectrum feature approach already built (and later deleted)
  for the IMS exploration, to see if the score direction corrects itself.
- Damage severity level (Table 2/4/5's 1-5 scale) isn't transcribed into
  `paderborn_adapter.py` yet -- checking whether the inverted-score
  bearings cluster at severity level 1 (the mildest, single-point,
  possibly genuinely closer to "still looks pretty normal at this coarse
  resolution") is a natural, cheap next check before assuming decimation
  is the whole story.
- 4kHz mechanical channels (force/speed/torque) and 1Hz temperature still
  not loaded -- unclear whether adding them would help or is irrelevant
  to this specific inversion pattern.
- `M=32` prototypes reached 100% codebook utilization from epoch 1 on a
  fit set ~66x larger than the one this value was tuned against (IMS) --
  worth trying a larger M to see if headroom helps either signal.
- No d-only/r-only/d+r comparison analysis was done beyond reporting all
  three side by side -- category means show `r` (structure) consistently
  edges out `d` (memory) across all three categories here, unlike some of
  the deleted IMS findings where the relationship flipped depending on
  architecture -- worth a closer look if this dataset gets revisited.

Full report: `checkpoints/paderborn/paderborn_fl_model_report.json`, model weights:
`checkpoints/paderborn/paderborn_fl_model.pth`.
