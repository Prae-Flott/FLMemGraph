---
name: sielaff-red-fault-federated
description: Sielaff V2 federated run restricted to domain-expert RED-severity faults only, on a freshly-built windowed dataset (data/sielaff_red/) since severity isn't recoverable from the existing group_6-labeled arrays
metadata:
  type: project
---

# Sielaff V2 federated, RED-severity faults only (2026-08-13)

Follow-up to [[joint-prototype-federated-results]] (Sielaff V2 federated,
`run_sielaff_joint_prototype_v2_federated.py`), scoped to the request: run
V2 on Sielaff using only the faults the domain-expert severity table in
`data/sielaff_data/sielaff_ground_truth.md` marks **red** (47 IDs, 5,593
events — "Important").

## Why a whole new dataset build was needed

`data/sielaff/` (the existing arrays used by every prior Sielaff run) labels
windows by `dominant_error_group_in_window` over 6 coarse GROUPS (bottle_
recognition, mechanical_jam, compactor_status, crate_recognition, other) —
see its `metadata.json`. Severity (red/orange/green) is a per-error-ID
attribute that cuts across those groups (e.g. both a red and a heavily-
represented orange error can dominate the same "bottle_recognition"
window), so it cannot be recovered from the group-labeled arrays alone —
the raw per-event error IDs are required. Wrote
`benchmark/datasets/build_sielaff_red.py`, which re-derives BOTH features
and labels together, directly from `data/sielaff_data/data_sielaff.zip`'s
raw tables, so they stay aligned. Sanity check: filtering `error_logs.csv`
to the 47 hardcoded red IDs reproduces exactly 5,593 events, matching the
ground-truth doc.

## Degradation-window definition (the actual design question)

Sielaff logs are discrete point events, not a continuously-drifting sensor
signal — "degradation" has no native meaning until a window is imposed.
Resolved (user-confirmed) to reuse the SAME convention every other
Joint-Prototype run already uses: a fixed-length window that CONTAINS the
anomalous event(s), scored against windows containing no logged error at
all — not a separate before-the-fault precursor concept. Window = 8 x
30min buckets = 4h, stride 2 buckets (1h), matching the "Sliding Window
Level (4h windows, N=18757)" `severity_4` scheme already benchmarked in
`sielaff_ground_truth.md`.

3-class label scheme per window (red is the sole target of interest):
- `0 = no_error` — zero logged errors of ANY severity in the window
- `1 = red` — >=1 of the 47 red-severity error IDs present
- `2 = non_red_error` — only orange/green errors, no red — kept OUT of the
  "normal" fit/calib reference set (would otherwise contaminate it), and
  reported as a contrast class, not the target

## Result: known-frequent-fault effect on the data itself

Because red-severity events are frequent (5,593 events over just 335
buckets x 10 machines), a 4h/1h-stride window is red-positive **71.6%** of
the time (1,174/1,640 windows) — leaving only ~24-64 strictly-clean
(`no_error`) windows PER MACHINE for fit+calib+test_normal combined (fit
16-44, calib 3-9, test_normal 5-11 per client — thin, but workable). This
is an inherent property of red-severity Sielaff faults at this window
length, not a bug — a shorter window would yield more clean windows at the
cost of noisier per-window aggregation.

## Federated AUROC (5 rounds, `run_sielaff_red_joint_prototype_v2_federated.py`)

| method | red (target, n=1174) | non_red_error (contrast, n=13 — too few to trust) |
|---|---|---|
| B_node_max | **0.941** | 0.995 |

Per-machine red AUROC (B_node_max): 0.89-0.98, no collapsed machine — consistent
with [[joint-prototype-federated-results]]'s finding that Sielaff's
federation "~matches centralized" thanks to real per-client sample volume,
even though here each client's OWN normal set is much smaller than the
group_6 dataset's (that one had ~1,400 no_error windows/client vs. ~30-60
here). `non_red_error`'s 0.99-1.0 AUROC is not a meaningful number (1-4
samples per machine) — reported for completeness only, not a real result.

## Follow-up: per-fault feature attribution (2026-08-13)

Added `benchmark/analyze_sielaff_red_feature_attribution.py`: for every red
window, take the trained model's per-node z-scores (`d_node`), find the
argmax (most-deviated) feature, and aggregate by the SPECIFIC red error ID
present in that window (needs `build_sielaff_red.py`'s new
`data/sielaff_red/window_red_ids.json`, saved alongside the labels so
feature and fault-ID stay aligned — the existing 3-class `targets.npy`
alone only says "red" vs not, not WHICH red fault).

**Two real problems found and fixed along the way, both about calibration
sample size, not the attribution logic itself:**

1. Per-client calib is only 6-9 windows here (see thin-normal-set note
   above) — too few for `zscore()`'s per-node IQR to be anything but its
   numerical floor (1e-8) for nearly every node, verified directly
   (`node_iqr` was `1e-8` for ~all 39 features on a test client). Fixed by
   pooling calib+test_normal SCORES (not weights — each window still
   scored by its own client's own model) across all 10 clients before
   z-scoring (63/77 pooled samples) — a deliberate, explicit departure
   from the federated AUROC run's per-client-calibrates-locally
   convention, justified because this script answers a different question
   (which node, not is-it-anomalous) needing more resolution.
2. Even with real (non-floored) IQR, `total_weight` still won top-1 for
   nearly every one of the 47 red fault types (0.5-1.0 vote share) — NOT
   an artifact this time, but a genuine confound: a strictly `no_error`
   window is disproportionately an IDLE window (no transactions -> no
   chance of ANY error, including red), so any window with real
   throughput looks anomalous relative to that idle baseline regardless
   of which specific fault occurred. Added a confound-excluded ranking
   (masks `total_weight`/`total_quantity`/`total_deposit`/`journal_count`/
   `unique_categories` — the `transaction_throughput` feature domain —
   out of the argmax) as the actually-informative signal; vote shares
   dropped to a more plausible 0.24-0.86 range and, critically, started
   DIFFERENTIATING by fault type instead of converging on one column.

## Result after both fixes (`checkpoints/sielaff/sielaff_red_feature_attribution.json`)

Confound-excluded top feature by fault, with an automated plausibility tag
(hand-built feature-domain vs. fault-name-keyword mapping, not learned —
see `feature_domain`/`FAULT_EXPECTED_DOMAIN`/`NO_SENSOR_KEYWORDS` in the
script):

- **Matches physical expectation** (`plausible`): `action_cleaning_start`/
  `action_cleaning_daily`/`action_cleaning_week` -> `cleaning_duration` /
  `cleaning_begin_pixels`; `status_printer_ok` / `status_receipt_retracted`
  -> `receipt_count`; `status_crate_bottle_wrong` / `status_bottle_
  compacted` -> `reject_count`/`total_weight`-family. These are the
  fault types that DO have a directly-instrumented feature in the
  39-column set, and the model found it.
- **Indirect** (`indirect`): most bottle-recognition/mechanical faults
  (`status_bottle_collision`, `status_bottle_out_of_range`,
  `status_backward_label_movement`, etc.) top out on `reject_count` rather
  than the RingCamera read-time features a bottle-recognition problem
  would mechanistically point to — plausible as a downstream correlate
  (a bottle that fails recognition is more likely to also get rejected)
  but not a direct sensor match.
- **No direct sensor at all** (`no_direct_sensor`, the largest group): ALL
  compactor door/flap faults (461-476, 481-493 series), ALL crate-handling
  faults (603, 618, 620, 303, 307, 309), `status_safety_circuit_off`,
  `status_dooropen`, `status_misc_sw_restart`/`status_misc_pc_reboot`.
  This 39-feature set (built from `journal`/`receipts`/`reject`/
  `cleaning`/`statistic_values` only, per `build_sielaff_red.py`) never
  logged a compactor, crate, or safety-circuit sensor at all — the model
  necessarily falls back to `receipt_count`/`reject_count` as an indirect
  proxy (machine activity correlates with everything), which is real
  detection signal (AUROC above confirms it) but should NOT be read as
  "the compactor door and receipt_count are physically linked."

**Practical implication**: the binary red/normal AUROC (0.905-0.941,
above) is trustworthy — it doesn't need per-feature attribution to hold.
The per-fault feature attribution is only as good as the underlying
feature set's actual sensor coverage: it correctly recovers the
mechanism for cleaning/printer faults (which ARE instrumented) and
honestly degrades to "generic activity proxy" for compactor/crate/safety
faults (which AREN'T) — this is the analysis surfacing a real limitation
of the raw log tables available, not a modeling failure.

## What's NOT done

- No repeat/multi-seed runs (same caveat as every other run in
  [[joint-prototype-federated-results]]).
- `non_red_error` contrast class is too data-starved to say anything about
  whether V2 actually distinguishes red from other-severity faults, only
  that it distinguishes red from strictly-clean.
- Centralized (non-federated) red-only run not done — no baseline to
  compare the federated number against on this specific label scheme.
