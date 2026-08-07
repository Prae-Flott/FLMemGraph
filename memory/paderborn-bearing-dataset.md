# Paderborn University KAt bearing dataset added (2026-08-06)

## What was added

- `data/paderborn_bearing_data/{K001-K006,KA01...,KB23...,KI01...}/` -- 32
  bearing folders, 80 `.mat` files each (2560 total), ~21GB, gitignored
  like all of `data/`. Downloaded from the official source
  (`https://groups.uni-paderborn.de/kat/BearingDataCenter/`, 32 `.rar`
  archives, ~5GB compressed, no registration needed) -- NOT the Kaggle
  mirror this session's earlier bearing dataset request needed credentials
  for; this one has a direct public download.
- `docs/paderborn_bearing_KAt2016.pdf` -- the reference paper (Lessmeier,
  Kimotho, Zimmer, Sextro, PHME 2016) describing the full experimental
  design, damage categorization scheme, and bearing geometry.
- `docs/paderborn_bearing_facts/` -- 64 PDFs (32 per-bearing fact sheets +
  32 measuring logs), extracted from INSIDE each bearing's `.rar` archive
  and copied out per the user's explicit request ("把其中的PDF文档也下载
  下来" -- these bundled fact sheets, not just the external paper, turned
  out to be what was inside "其中"). Original copies inside
  `data/paderborn_bearing_data/<code>/` were removed after copying to
  avoid duplication -- `docs/` is the single source for all PDF
  documentation, `data/` is data-only.

## What's actually in each `.mat` file (verified directly, not assumed
## from the paper's summary alone -- the paper's own text said "26 kHz"
## in one place, which does NOT match the actual data)

Loaded with `scipy.io.loadmat`. Each file is a MATLAB struct with fields
`Info`, `X`, `Y`, `Description`; `Y` holds 7 channels, verified directly
from `N15_M07_F10_K001_1.mat`:

| channel | raster/rate | samples (4s recording) | actual rate |
|---|---|---|---|
| `force` | Mech_4kHz | 16,001 | ~4 kHz |
| `speed` | Mech_4kHz | 16,001 | ~4 kHz |
| `torque` | Mech_4kHz | 16,001 | ~4 kHz |
| `phase_current_1` | HostService | 256,001 | ~64 kHz |
| `phase_current_2` | HostService | 256,001 | ~64 kHz |
| `vibration_1` | HostService | 256,001 | ~64 kHz |
| `temp_2_bearing_module` | Temp_1Hz | 5 | ~1 Hz |

64kHz for current+vibration matches the paper's own claimed headline
number ("high sampling rate for the two main signals (64 kHz)") in a
different section -- the "26 kHz" figure that showed up in an early web
search summary of this dataset was wrong, not carried into this file.

## Bearing damage taxonomy (verified from the paper's Table 4/5/6/7, not
## inferred from filenames alone)

**Naming convention**: `<Speed>_<Torque>_<Force>_<BearingCode>_<Rep>.mat`,
e.g. `N15_M07_F10_KA01_1.mat`. 4 fixed operating conditions (Table 6),
20 repetitions each = 80 files/bearing:

| condition | rotational speed | load torque | radial force |
|---|---|---|---|
| N15_M07_F10 | 1500 rpm | 0.7 Nm | 1000 N |
| N09_M07_F10 | 900 rpm | 0.7 Nm | 1000 N |
| N15_M01_F10 | 1500 rpm | 0.1 Nm | 1000 N |
| N15_M07_F04 | 1500 rpm | 0.7 Nm | 400 N |

**Bearing code prefix = which ring is damaged**, NOT artificial-vs-real
(that's a separate, orthogonal axis -- a common misread of this dataset):
- `K001`-`K006`: healthy reference bearings (different run-in periods,
  Table 7) -- the "normal" class.
- `KA*`: **outer ring (OR)** damage. Both artificial (KA01/03/05/06/07/
  08/09, via EDM/electric engraver/drilling) and real accelerated-
  lifetime damage (KA04/15/16/22/30, fatigue pitting or plastic-
  deformation indentations) use the KA prefix.
- `KI*`: **inner ring (IR)** damage. Artificial (KI01/03/05/07/08) and
  real (KI04/14/16/17/18/21, all fatigue pitting) both use KI.
- `KB*`: **combined/multiple ring** damage (IR+OR together), only appears
  among the real-damage bearings (KB23/24/27).

32 bearings total = 6 healthy + 12 artificial-damage + 14 real-damage
(accelerated lifetime test), matching the paper's own count exactly.
Each damaged bearing also has a severity level (1-5, by % of raceway
circumference damaged, bearing-6203-specific mm thresholds in the paper's
Table 2) and a damage-extent characteristic (single point / repetitive /
distributed) -- all recorded per-bearing in the individual fact-sheet
PDFs now in `docs/paderborn_bearing_facts/`, not yet transcribed into a
machine-readable table in this repo.

## Why this dataset is a good fit for THIS project's design

This dataset is a **discrete fault-classification** dataset, structurally
close to robo3er/Sielaff: every bearing has a definite, verified damage
class (healthy / OR-damaged / IR-damaged / combined, each with known
severity and generation method), measured under controlled, repeated,
varied operating conditions -- a natural fit for this project's existing
GDN/memory training convention (fit on healthy-only K001-K006, evaluate
AUROC against each damage class). It also includes **motor current
signals** (phase_current_1/2) alongside vibration -- a genuinely
different sensing modality from every other dataset in this project so
far (all vibration/kinematics-based), and the whole point of the
reference paper is comparing current-based vs. vibration-based diagnosis.

## What's NOT done yet (next steps if this dataset gets used for training)

- No adapter (`benchmark/datasets/paderborn_adapter.py`) exists yet --
  this session only downloaded, extracted, and documented the raw data.
- No windowing/train-test-split convention decided (each `.mat` file is
  one 4-second continuous recording per channel per condition per
  bearing -- needs chunking into fixed-length windows, similar in spirit
  to how `generate_sielaff_data.py` did upstream for Sielaff).
- The per-bearing damage severity/characteristic details (Table 4/5's
  columns) are only in the PDF fact sheets so far, not transcribed into
  machine-readable Python metadata yet.
- Given both current AND vibration channels exist, a natural first
  experiment once an adapter exists: does GDN's graph-attention structure
  signal do better when built over BOTH modalities together (motor
  current channels + vibration channel as separate graph nodes) vs. either
  modality alone -- directly extends this session's "does cross-channel
  relational structure help" question with a genuinely new kind of
  relationship (electrical vs. mechanical) to test.
