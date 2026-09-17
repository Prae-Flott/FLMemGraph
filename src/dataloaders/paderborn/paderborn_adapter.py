"""
Paderborn University KAt Bearing DataCenter -- adapter for the raw
`.mat` files in `data/paderborn_bearing_data/` (downloaded from
`https://groups.uni-paderborn.de/kat/BearingDataCenter/`, official mirror,
no registration; reference paper: Lessmeier, Kimotho, Zimmer, Sextro,
PHME 2016, `docs/paderborn_bearing_KAt2016.pdf`; per-bearing fact sheets
in `docs/paderborn_bearing_facts/`). See `memory/paderborn-bearing-dataset.md`
for the full verified damage-code taxonomy this module transcribes.

Unlike `data/bearing_data/` (NASA IMS, deleted -- a run-to-failure
dataset with no per-window ground truth), this dataset has DEFINITE
ground truth: every bearing has a verified damage class, so it can use
this project's normal fit-on-healthy/evaluate-AUROC-per-fault-class
convention (like robo3er/Sielaff), not IMS's forced trend-ranking
workaround.

## Bearing code -> damage category (verified against the paper's Table 4/5)

`K***`: healthy reference bearings (6: K001-K006).
`KA**`: OUTER RING damage (7 artificial: KA01/03/05/06/07/08/09; 5 real
        accelerated-lifetime: KA04/15/16/22/30).
`KI**`: INNER RING damage (5 artificial: KI01/03/05/07/08; 6 real:
        KI04/14/16/17/18/21).
`KB**`: COMBINED inner+outer ring damage (3, all real: KB23/24/27).

## What's actually loaded here

Each `.mat` file has 7 channels at 3 different sample rates (force/speed/
torque @~4kHz, phase_current_1/2 + vibration_1 @~64kHz, bearing temp
@~1Hz -- verified directly, see memory file). `load_bearing()` (the
original, still used by `run_paderborn_fl_model.py`) only loads the three
~64kHz channels -- no resampling needed since they share a rate.
`load_bearing_with_physics()` (added for `run_paderborn_physics_gdn.py`,
the `kinematics.py`-style physics-residual pipeline) additionally loads
`force`/`speed`/`torque` at their native ~4kHz rate and resamples them
onto the SAME decimated time axis as the electrical channels, needed to
fit `src/paderborn_physics.py`'s torque/speed -> current-envelope
regression. The 1Hz temperature channel is still not loaded (5 samples
per 4s file -- too coarse to align meaningfully to either time axis).

**Current channels use RMS-per-block decimation, not mean-per-block.**
`vibration_1` uses mean decimation (a crude box-average low-pass, kept
for continuity with `load_bearing()`/the earlier bearing work). Phase
current is fundamentally an AC signal oscillating around zero at line
frequency (~50Hz) -- mean-decimating it doesn't drive it to exactly zero
(each decimation block here is much shorter than one full AC cycle,
~10%), but it's still the wrong representation for what this module
actually wants: a smooth current MAGNITUDE/ENVELOPE signal that can be
meaningfully regressed against torque (a DC-ish quantity) in
`fit_current_model()`. RMS-per-block (`sqrt(mean(x**2))`) extracts that
envelope directly -- this is the standard current-envelope convention in
motor-current-signature-analysis (MCSA) literature, not an ad hoc choice.
"""
import re
from pathlib import Path

import numpy as np
import scipy.io as sio

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "paderborn_bearing_data"

CHANNELS = ["vibration_1", "phase_current_1", "phase_current_2"]
# Per-file sample counts are NOT exactly constant (verified across all 2560 files: 249,940 -
# 256,744, most but not all at 256,001) -- RAW_LEN is set safely below the observed minimum
# (249,940), not the common-case ~256,001, and divisible by every DECIMATE factor this module
# is called with (125 by default: 240000/125=1920).
RAW_LEN = 240000

HEALTHY_CODES = ["K001", "K002", "K003", "K004", "K005", "K006"]
OUTER_RING_CODES = ["KA01", "KA03", "KA05", "KA06", "KA07", "KA08", "KA09",
                      "KA04", "KA15", "KA16", "KA22", "KA30"]
INNER_RING_CODES = ["KI01", "KI03", "KI05", "KI07", "KI08",
                      "KI04", "KI14", "KI16", "KI17", "KI18", "KI21"]
COMBINED_CODES = ["KB23", "KB24", "KB27"]
ARTIFICIAL_CODES = {"KA01", "KA03", "KA05", "KA06", "KA07", "KA08", "KA09",
                      "KI01", "KI03", "KI05", "KI07", "KI08"}

ALL_DAMAGED_CODES = OUTER_RING_CODES + INNER_RING_CODES + COMBINED_CODES
ALL_CODES = HEALTHY_CODES + ALL_DAMAGED_CODES


def category_of(code: str) -> str:
    if code in HEALTHY_CODES:
        return "healthy"
    if code in OUTER_RING_CODES:
        return "outer_ring"
    if code in INNER_RING_CODES:
        return "inner_ring"
    if code in COMBINED_CODES:
        return "combined"
    raise ValueError(f"unknown bearing code {code!r}")


def damage_origin_of(code: str) -> str:
    if code in HEALTHY_CODES:
        return "n/a"
    return "artificial" if code in ARTIFICIAL_CODES else "real"


_FNAME_RE = re.compile(r"^(N\d+)_(M\d+)_(F\d+)_(\w+)_(\d+)\.mat$")


def list_files(code: str):
    """80 files for one bearing code, sorted (condition, repetition)."""
    bearing_dir = DATA_DIR / code
    files = sorted(bearing_dir.glob(f"*_{code}_*.mat"),
                    key=lambda p: (p.name.split("_")[0], int(p.stem.rsplit("_", 1)[1])))
    if not files:
        raise FileNotFoundError(f"No .mat files found under {bearing_dir}")
    return files


def _load_raw_file(path: Path) -> np.ndarray:
    """[RAW_LEN, 3] float32, columns = CHANNELS, in that order."""
    d = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    top = d[path.stem]
    by_name = {item.Name: item.Data for item in top.Y}
    out = np.stack([np.asarray(by_name[ch][:RAW_LEN], dtype=np.float32) for ch in CHANNELS], axis=1)
    return out  # [RAW_LEN, 3]


def load_bearing(code: str, decimate: int = 125):
    """Load all measurement files for one bearing code (normally 80;
    fewer if any are unreadable -- see below), box-average decimated
    (same technique as the deleted IMS adapter used -- crude but real
    anti-aliasing, not naive subsampling).

    One file in this dataset (`KA08/N15_M01_F10_KA08_2.mat`) fails to
    parse with scipy.io.loadmat (verified: not a download truncation --
    file size is normal, the internal MAT5 struct layout itself is
    malformed) -- skipped with a printed warning rather than crashing the
    whole load, since it's 1 of 2560 files project-wide.

    Returns (windows [n_ok, RAW_LEN//decimate, 3] float32, conditions
    [n_ok] str -- the 'N../M../F..' operating-condition code parsed from
    each filename, channels list)."""
    files = list_files(code)
    decimated_len = RAW_LEN // decimate
    windows_list, conditions = [], []
    for path in files:
        try:
            raw = _load_raw_file(path)  # [RAW_LEN, 3]
        except Exception as e:
            print(f"  WARNING: skipping unreadable file {path.name}: {e}")
            continue
        windows_list.append(raw.reshape(decimated_len, decimate, len(CHANNELS)).mean(axis=1))
        m = _FNAME_RE.match(path.name)
        conditions.append(f"{m.group(1)}_{m.group(2)}_{m.group(3)}" if m else "unknown")
    windows = np.stack(windows_list, axis=0).astype(np.float32)
    return windows, np.array(conditions), list(CHANNELS)


MECH_CHANNELS = ["force", "speed", "torque"]
# Mechanical channels run at ~4kHz (~1/16 of the electrical channels' ~64kHz) -- verified
# minimum length across a 200-file sample was 16001 (speed), so RAW_LEN_MECH is set safely
# below that, same margin-below-observed-minimum convention as RAW_LEN above.
RAW_LEN_MECH = 15000


def _load_raw_mech_file(path: Path) -> np.ndarray:
    """[RAW_LEN_MECH, 3] float32, columns = MECH_CHANNELS, in that order."""
    d = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    top = d[path.stem]
    by_name = {item.Name: item.Data for item in top.Y}
    out = np.stack([np.asarray(by_name[ch][:RAW_LEN_MECH], dtype=np.float32) for ch in MECH_CHANNELS], axis=1)
    return out


def _decimate_blocks(arr: np.ndarray, target_len: int, mode: str = "mean") -> np.ndarray:
    """arr: [T, F] -> [target_len, F], via np.array_split into target_len
    nearly-equal blocks along axis 0 (robust to T not being an exact
    multiple of target_len, unlike a plain reshape). mode='rms' for AC
    signals (current), mode='mean' for everything else."""
    blocks = np.array_split(arr, target_len, axis=0)
    if mode == "rms":
        return np.stack([np.sqrt(np.mean(b.astype(np.float64) ** 2, axis=0)) for b in blocks]).astype(np.float32)
    return np.stack([b.mean(axis=0) for b in blocks]).astype(np.float32)


def load_bearing_with_physics(code: str, decimate: int = 125):
    """Like `load_bearing()`, but also loads force/speed/torque (resampled
    onto the same decimated time axis) and uses RMS decimation for the
    two current channels instead of mean -- see module docstring.

    Returns dict with keys 'vibration_1', 'phase_current_1',
    'phase_current_2' (RMS-decimated), 'force', 'speed', 'torque'
    (mean-decimated, resampled), each [n_ok, decimated_len] float32, plus
    'conditions' [n_ok] str."""
    files = list_files(code)
    decimated_len = RAW_LEN // decimate
    out = {ch: [] for ch in CHANNELS + MECH_CHANNELS}
    conditions = []
    for path in files:
        try:
            raw_elec = _load_raw_file(path)      # [RAW_LEN, 3]
            raw_mech = _load_raw_mech_file(path)  # [RAW_LEN_MECH, 3]
        except Exception as e:
            print(f"  WARNING: skipping unreadable file {path.name}: {e}")
            continue
        for i, ch in enumerate(CHANNELS):
            mode = "rms" if ch.startswith("phase_current") else "mean"
            out[ch].append(_decimate_blocks(raw_elec[:, i : i + 1], decimated_len, mode)[:, 0])
        for i, ch in enumerate(MECH_CHANNELS):
            out[ch].append(_decimate_blocks(raw_mech[:, i : i + 1], decimated_len, "mean")[:, 0])
        m = _FNAME_RE.match(path.name)
        conditions.append(f"{m.group(1)}_{m.group(2)}_{m.group(3)}" if m else "unknown")
    result = {ch: np.stack(v, axis=0).astype(np.float32) for ch, v in out.items()}
    result["conditions"] = np.array(conditions)
    return result


if __name__ == "__main__":
    for code in ALL_CODES:
        n = len(list_files(code))
        print(f"{code:<6} category={category_of(code):<11} origin={damage_origin_of(code):<10} n_files={n}")
