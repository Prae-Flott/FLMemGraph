#!/usr/bin/env python3
"""
Compare each FL baseline's own single anomaly score against our BHK
mainline's AUROC -- see `benchmark/baselines/registry.py` for what each
baseline is (each has its own minimal architecture and own score; only
`ours` produces the B/H/K/BK/BHK breakdown).

Usage:
    python3 compare_baselines.py --dataset {robo_fleet,paderborn,alfa,me_ad}
"""
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

OURS_SUFFIX = {
    "robo_fleet": "forecast_v2_federated_h10_encdec_synced",
    "paderborn": "forecast_v2_federated_h10_encdec_synced",
    "alfa": "bck_federated_h1",
    "me_ad": "forecast_v2_federated_h4_encdec_synced",
}
BASELINE_SCORE_NAME = {"fedavg": "recon_score", "ifcaae": "recon_score", "fedexdnn": "exemplar_score",
                        "fedpro": "fedpro_score", "fedcpg": "fedcpg_score", "ufedhy": "recon_score"}
BASELINES = ["fedavg", "ifcaae", "fedexdnn", "fedpro"]
# FedPRO runs on all 4 original datasets (see registry.py's FedPRO entry for the per-dataset class
# definitions -- ME-AD uses binary healthy-vs-joint3_degradation, since it has no fault subtypes to
# enumerate), but it's a supervised classifier, not fit-on-healthy, so its AUROC is not apples-to-apples
# with the other three; caveat printed alongside its row below.
# fedcpg/ufedhy (2026-09-10) all use a plain "{baseline}" out_suffix (not the OURS_SUFFIX-based
# naming robo_fleet/paderborn/alfa/me_ad's original four baselines use), handled as a load_report special
# case below.
DATASET_BASELINES = {
    "robo_fleet": BASELINES + ["fedcpg", "ufedhy"],
    "paderborn": BASELINES + ["fedcpg", "ufedhy"],
    "alfa": BASELINES + ["fedcpg", "ufedhy"],
    "me_ad": BASELINES + ["fedcpg", "ufedhy"],
}


def load_report(dataset, baseline):
    if baseline == "ours":
        suffix = OURS_SUFFIX[dataset]
    elif baseline in ("fedcpg", "ufedhy"):
        suffix = baseline
    else:
        suffix = OURS_SUFFIX[dataset] + f"_{baseline}" if dataset != "alfa" else f"bck_federated_h1_{baseline}"
    path = REPO_ROOT / "checkpoints" / dataset / f"{dataset}_{suffix}_report.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main(dataset):
    print(f"\n=== {dataset}: baseline AUROC vs. our BHK mainline ===")
    print(f"  {'method':<16}{'score':<16}{'AUROC':>10}{'AUPRC':>10}{'Precision':>12}{'F1':>10}")

    ours = load_report(dataset, "ours")
    if ours is None:
        print("  [skip] ours: no report found")
    else:
        summary = ours.get("summary_mean_auroc_overall", {})
        for key in ("BHK_max", "BHK_lw"):
            m = summary.get(key)
            if m is None:
                continue
            print(f"  {'ours':<16}{key:<16}{m['auroc']:>10.3f}{m['auprc']:>10.3f}"
                  f"{m['precision']:>12.3f}{m['f1']:>10.3f}")

    for baseline in DATASET_BASELINES[dataset]:
        report = load_report(dataset, baseline)
        if report is None:
            print(f"  [skip] {baseline}: no report found")
            continue
        score_name = BASELINE_SCORE_NAME[baseline]
        summary = report.get("summary_mean_auroc_overall", {})
        m = summary.get(score_name)
        if m is None:
            print(f"  [skip] {baseline}: no {score_name} in summary_mean_auroc_overall")
            continue
        print(f"  {baseline:<16}{score_name:<16}{m['auroc']:>10.3f}{m['auprc']:>10.3f}"
              f"{m['precision']:>12.3f}{m['f1']:>10.3f}")
        if baseline in ("fedpro", "fedcpg"):
            print(f"    NOTE: {baseline} is supervised on ALL labels (accuracy={report.get('mean_accuracy'):.3f}), "
                  f"not fit-on-healthy -- its AUROC is not apples-to-apples with the other baselines above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                         choices=["robo_fleet", "paderborn", "alfa", "me_ad"])
    args = parser.parse_args()
    main(args.dataset)
