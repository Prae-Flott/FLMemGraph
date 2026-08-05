"""
Dataset registry, transcribed from `mem_phys_prompt_zh.md` Sec 8.1's
benchmark plan. Four categories, matching the four things this project's
"记忆检索 + 物理验证" design (src/) needs to be evaluated
against: (A) multivariate time-series anomaly detection with physical/
sensor structure -- the PRIMARY target, closest to robo3er itself; (B)
federated image anomaly detection -- tests the memory+federation idea in
a completely different modality; (C) dynamic-graph anomaly detection --
tests the "structure/relation" half of the design in a domain where the
graph itself evolves; (D) federated time-series forecasting -- FEDPM's
own original benchmark, the most direct comparison for the memory module
alone.

`status` values:
  - "local"       : already on disk in this repo, ready to use (robo3er)
  - "downloaded"   : fetched into benchmark/datasets/raw/ this session
  - "downloadable" : freely available, not yet fetched (small/no-registration)
  - "registration_required" : needs a manual data request (SWaT/WADI via
    iTrust) -- cannot be automated from this environment
  - "large_download" : freely available but multi-GB (MVTec-AD etc.) --
    not fetched by default, left as a manual step
"""

DATASETS = {
    # ---------------------------------------------------------- category A --
    "robo3er": {
        "category": "A",
        "description": "This project's own 5-robot iRobot Create3 fleet, 71 raw sensor channels, 60-step windows.",
        "num_features": 71,
        "used_in_papers": ["this project"],
        "why_relevant": "The actual multi-source deployment target -- every other dataset here is a generalization check, this one is the real target.",
        "status": "local",
        "path": "data/robo3er/",
    },
    "SWaT": {
        "category": "A",
        "description": "Secure Water Treatment testbed, SCADA, 51 sensors/actuators, includes physical attacks with documented ground-truth attack vectors.",
        "num_features": 51,
        "used_in_papers": ["GDN (Deng & Hooi, AAAI 2021, arXiv:2106.06947)"],
        "why_relevant": "Real actuator/sensor relational structure + physical attack ground truth -- directly tests whether the structure-residual signal catches a relation-breaking attack the way it's meant to.",
        "status": "registration_required",
        "obtain": "https://itrust.sutd.edu.sg/itrust-labs_datasets/ -- iTrust dataset request form, manual approval, cannot be automated.",
    },
    "WADI": {
        "category": "A",
        "description": "Water Distribution testbed, 123 sensors/actuators, same iTrust family as SWaT.",
        "num_features": 123,
        "used_in_papers": ["GDN"],
        "status": "registration_required",
        "obtain": "https://itrust.sutd.edu.sg/itrust-labs_datasets/",
    },
    "SMD": {
        "category": "A",
        "description": "Server Machine Dataset, 28 machines x 38 metrics, unsupervised server monitoring.",
        "num_features": 38,
        "num_clients_available": 28,
        "used_in_papers": ["FedKO (arXiv:2503.11255)", "FedKAD (arXiv:2607.08978)"],
        "why_relevant": "28 naturally separate machines = a real non-IID federated client set (bigger than robo3er's 5), good stress test for the federated memory alignment at higher client count.",
        "status": "downloadable",
        "obtain": "https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset (train/test/interpretation_label per machine, plain text, no registration).",
    },
    "SMAP": {
        "category": "A",
        "description": "NASA Soil Moisture Active Passive satellite telemetry, 55 channels.",
        "num_features": 55,
        "used_in_papers": ["GDformer (arXiv:2501.18196)", "FedKO", "FedKAD"],
        "status": "downloadable",
        "obtain": "https://github.com/khundman/telemanom (NASA data + labeled_anomalies.csv).",
    },
    "MSL": {
        "category": "A",
        "description": "NASA Mars Science Laboratory (Curiosity rover) telemetry, 27 channels.",
        "num_features": 27,
        "used_in_papers": ["GDformer", "FedKO", "FedKAD"],
        "why_relevant": "Actual ROBOT telemetry (a rover) with documented anomaly episodes -- closest public analogue to robo3er outside this project's own data.",
        "status": "downloadable",
        "obtain": "https://github.com/khundman/telemanom",
    },
    "PSM": {
        "category": "A",
        "description": "Pooled Server Metrics (eBay), 25-dim server monitoring.",
        "num_features": 25,
        "used_in_papers": ["GDformer", "FedKO", "FedKAD"],
        "status": "downloadable",
        "obtain": "https://github.com/eBay/RANSynCoders (data/ directory).",
    },
    "GECCO": {
        "category": "A",
        "description": "Drinking water quality monitoring, physical sensor stream.",
        "used_in_papers": ["GDformer"],
        "status": "downloadable",
        "obtain": "https://www.spotseven.de/gecco/gecco-challenge/gecco-challenge-2018/",
    },
    # ---------------------------------------------------------- category B --
    "MVTec-AD": {
        "category": "B",
        "description": "Industrial visual inspection, 15 object/texture classes, pixel-level anomaly masks.",
        "used_in_papers": ["FedDyMem (arXiv:2502.21012)"],
        "why_relevant": "Tests the memory-bank idea in a completely different modality (images, not time series) -- if the discrete-prototype mechanism only works for time series this won't transfer, which is itself informative.",
        "status": "large_download",
        "obtain": "https://www.mvtec.com/company/research/datasets/mvtec-ad (~5GB, direct download, no registration but large).",
    },
    "MPDD": {"category": "B", "used_in_papers": ["FedDyMem"], "status": "large_download",
              "obtain": "https://github.com/stepanje/MPDD"},
    "VisA": {"category": "B", "used_in_papers": ["FedDyMem"], "status": "large_download",
              "obtain": "https://github.com/amazon-science/spot-diff"},
    # ---------------------------------------------------------- category C --
    "Bitcoin-Alpha": {
        "category": "C",
        "description": "Dynamic trust-weighted graph, bitcoin trading platform, labeled fraud.",
        "used_in_papers": ["DP-DGAD (arXiv:2508.00664)"],
        "status": "downloadable",
        "obtain": "http://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html",
    },
    "Wikipedia": {"category": "C", "used_in_papers": ["DP-DGAD"], "status": "downloadable",
                   "obtain": "http://snap.stanford.edu/jodie/wikipedia.csv"},
    "MOOC": {"category": "C", "used_in_papers": ["DP-DGAD"], "status": "downloadable",
              "obtain": "http://snap.stanford.edu/jodie/mooc.csv"},
    # ---------------------------------------------------------- category D --
    "ETTh1": {
        "category": "D",
        "description": "Electricity Transformer Temperature, hourly, 7 features.",
        "used_in_papers": ["FEDPM (arXiv:2604.04475) -- this project's own core reference"],
        "why_relevant": "FEDPM's own benchmark -- the most direct apples-to-apples comparison for the memory module in isolation, before this project's GDN/physics additions.",
        "status": "downloadable",
        "obtain": "https://github.com/zhouhaoyi/ETDataset",
    },
    "Electricity": {"category": "D", "used_in_papers": ["FEDPM"], "status": "downloadable",
                      "obtain": "https://github.com/zhouhaoyi/ETDataset (UCI Electricity subset commonly bundled with Informer/Autoformer repos)."},
    "Weather": {"category": "D", "used_in_papers": ["FEDPM"], "status": "downloadable",
                 "obtain": "https://github.com/thuml/Autoformer (dataset release)."},
    "Exchange": {"category": "D", "used_in_papers": ["FEDPM"], "status": "downloadable",
                  "obtain": "https://github.com/laiguokun/multivariate-time-series-data"},
}


def by_category(cat):
    return {k: v for k, v in DATASETS.items() if v["category"] == cat}


def by_status(status):
    return {k: v for k, v in DATASETS.items() if v["status"] == status}


if __name__ == "__main__":
    for cat in "ABCD":
        print(f"\n=== category {cat} ===")
        for name, info in by_category(cat).items():
            print(f"  {name:<15} status={info['status']:<24} {info.get('description', '')}")
