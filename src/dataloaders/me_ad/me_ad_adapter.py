"""
ME-AD (Mitsubishi Electric Research Laboratories, "Progressive Robotics
Anomaly Detection Dataset," Zenodo DOI 10.5281/zenodo.20817530,
CC-BY-SA-4.0) -- adapter for the raw archive at
`data/me_ad_raw/ME-AD.zip` (10.79GB, downloaded directly from Zenodo, not
committed to the repo). Records a single 6-DoF Mitsubishi RV-7FM-D1-S15
manipulator executing repeated pick-and-place cycles while a REAL,
intentionally-introduced defect (lubricant escape in joint 3's actuator)
causes progressive mechanical wear over cycles -- unlike Paderborn's
swapped-in damaged bearings or robo_fleet's induced faults, this is one
continuously-progressing fault, and degradation is a function of CYCLE
INDEX (time), not of which pick-and-place operation is running.

## Client = operation code (16 total), matching Paderborn's convention

The archive's `ME-AD/README.md` documents 16 four-digit "FMMV" operation
codes across 3 motion families (F=1: two-transfer pick-and-place, varies
by speed/layout, 8 codes; F=2: one transfer factored into 6 stop-and-go
sub-paths; F=3: same geometry as F=2 but continuous/blended motion, no
stops). Each code is a REAL, independently-recorded motion program (same
distinction Paderborn draws between bearing codes) -- treated here as one
federated client, exactly like Paderborn's 6 healthy bearing codes.
Because the fault is on the SAME physical joint 3 across every code (not
independent per-client fault occurrence like robo_fleet/Paderborn), this
dataset tests "does federated alignment help when a shared root-cause
fault manifests differently across operation modes" -- a different (still
valid) federated question than the other three datasets ask; state this
plainly wherever ME-AD results are reported.

## Healthy/faulty split: the paper's OWN convention, RE-VERIFIED empirically

`README.md`'s "Suggested AD Tasks" table describes, per operation code:
"the first 70 cycles are considered nominal" / "in the last 50 cycles,
the anomaly is considered evident." A first reading of this as literal
cycles [0:70]/[70:120] of the ~1600-2000 available produces near-chance
AUROC end to end (verified: both `ours` and the FedAvg baseline land at
~0.50 on every client when sliced that way) -- and a direct check of
joint-3 filtered torque (`tau_filt_3`, the actuator with the injected
defect) confirms why: comparing per-cycle mean `tau_filt_3` between
cycles [0:70] and cycles [70:120] of operation code 1050 shows NO shift
(-153.29 vs -153.33, std of per-cycle means 0.93 vs 0.50) -- cycle 120 is
still far too early in this code's ~2063-cycle run for the gradual wear
to show. Comparing cycles [0:70] against the code's ACTUAL LAST 50 cycles
(index [-50:]) shows a clear, real shift instead (-153.29 -> -151.46,
std 0.93 -> 2.50 -- both a mean shift and a large variance increase,
consistent with progressive mechanical wear). So "the last 50 cycles"
means the last 50 of the FULL run, not cycles 71-120 of a literal
120-cycle window -- `load_operation_code` reflects this: `num_healthy`
cycles from the START, `num_faulty` cycles from the END, of however many
cycles actually exist for that code (1595-2063, varies by code).

## Channels: filtered only (24 nodes)

Each cycle's `.pkl` has 54 columns: raw `q/dq/ddq` (18) + FILTERED
`q_filt/dq_filt/ddq_filt` (18) + raw `tau` + `tau_MAT` + FILTERED
`tau_filt` (18, three torque variants). We use only the 24 filtered
channels (`q_filt_1..6, dq_filt_1..6, ddq_filt_1..6, tau_filt_1..6`),
matching this project's existing preference for cleaned/denoised signals
(Paderborn/robo_fleet both use their datasets' cleaned channels).
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
ZIP_PATH = REPO_ROOT / "data" / "me_ad_raw" / "ME-AD.zip"

OPERATION_CODES = ["1050", "1051", "1052", "1053", "1100", "1101", "1102", "1103",
                    "2000", "2100", "2200", "2300", "2400", "2500", "3000", "3100"]


def motion_family(code):
    return "F1" if code.startswith("1") else ("F2" if code.startswith("2") else "F3")


NODE_NAMES = ([f"q_filt_{i}" for i in range(1, 7)] + [f"dq_filt_{i}" for i in range(1, 7)]
              + [f"ddq_filt_{i}" for i in range(1, 7)] + [f"tau_filt_{i}" for i in range(1, 7)])

# Adjacent-joint dynamic coupling in a serial 6-DoF manipulator (textbook
# robot dynamics: link i's motion couples into link i+1's required
# torque) -- declared only among the torque channels, "nonlinear" like
# Paderborn's torque/current-coupling edges.
EDGES_NAMED = [(f"tau_filt_{i}", f"tau_filt_{i+1}") for i in range(1, 6)]
EDGE_TYPES = ["nonlinear"] * len(EDGES_NAMED)

NUM_HEALTHY_CYCLES = 70
NUM_FAULTY_CYCLES = 50


def _list_cycle_names(zf, code):
    prefix = f"ME-AD/Pandas/{code}/cleaned_dataset_"
    names = [n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".pkl")]

    def idx_of(n):
        return int(n[len(prefix):-len(".pkl")])

    names.sort(key=idx_of)
    return names


def _load_pkls(zf, names):
    cycles = []
    for n in names:
        with zf.open(n) as f:
            df = pd.read_pickle(f)
        cycles.append(df[NODE_NAMES].to_numpy(dtype=np.float32))
    return cycles


def load_operation_code(code, num_healthy=NUM_HEALTHY_CYCLES, num_faulty=NUM_FAULTY_CYCLES):
    """Returns `(healthy_cycles, faulty_cycles)`: `num_healthy` cycles
    from the START and `num_faulty` cycles from the END of this operation
    code's full run (sorted cycle-index order -- the archive's
    `cleaned_dataset_<idx>.pkl` numeric suffixes aren't perfectly
    sequential, some indices are missing where `Import_data.py`'s
    corruption-removal step dropped bad cycles), NOT a contiguous
    first-N block -- see module docstring for why (the fault only
    becomes visible at the true end of the run, verified empirically)."""
    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = _list_cycle_names(zf, code)
        healthy_cycles = _load_pkls(zf, names[:num_healthy])
        faulty_cycles = _load_pkls(zf, names[-num_faulty:])
    return healthy_cycles, faulty_cycles


def count_cycles(code):
    with zipfile.ZipFile(ZIP_PATH) as zf:
        return len(_list_cycle_names(zf, code))
