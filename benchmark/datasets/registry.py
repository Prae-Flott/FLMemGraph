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
    "Paderborn-KAt-bearing": {
        "category": "A",
        "description": "Paderborn University KAt Bearing DataCenter: 32 type-6203 ball bearings "
                         "(6 healthy, 12 artificially damaged, 14 real accelerated-lifetime-test "
                         "damage), 4 controlled operating conditions x 20 reps x 4s each = 80 "
                         ".mat files/bearing, 7 channels (motor current x2 @64kHz, vibration @64kHz, "
                         "force/speed/torque @4kHz, bearing temp @1Hz).",
        "num_features": 7,
        "used_in_papers": ["Lessmeier, Kimotho, Zimmer, Sextro, PHME 2016 (original reference paper, "
                             "motor-current-signal-based bearing diagnosis benchmark)"],
        "why_relevant": "Not from mem_phys_prompt_zh.md Sec 8.1 -- added on request as a real "
                          "dataset alongside robo3er/Sielaff. A DISCRETE fault-classification dataset "
                          "(definite damage class per bearing: healthy / outer-ring / inner-ring / "
                          "combined, each with known severity and generation method) -- close in "
                          "structure to robo3er/Sielaff's fit-on-normal/evaluate-per-class-AUROC "
                          "convention. Also includes MOTOR CURRENT channels alongside "
                          "vibration -- a genuinely different sensing modality from every other "
                          "dataset in this project, testing whether GDN's cross-channel structure "
                          "signal transfers across modalities (electrical vs. mechanical), not just "
                          "across datasets.",
        "status": "implemented",
        "path": "data/paderborn_bearing_data/{K001..,KA..,KB..,KI..}/ (~21GB uncompressed), docs "
                 "(reference paper + 64 per-bearing fact-sheet/measuring-log PDFs) in "
                 "docs/paderborn_bearing_KAt2016.pdf and docs/paderborn_bearing_facts/",
        "obtain": "https://groups.uni-paderborn.de/kat/BearingDataCenter/ (official KAt DataCenter "
                   "mirror, 32 direct-download .rar files, no registration).",
        "note_extra": "benchmark/datasets/paderborn_adapter.py (loader, 3 of 7 channels used -- "
                        "vibration_1 + phase_current_1/2, see its docstring) + "
                        "benchmark/run_paderborn_fl_model.py (single-machine, no federation, reuses "
                        "src.models.fl_model.FLGDNMemory unchanged) implement this project's full memory+"
                        "structure pipeline here. Result: strong category-mean AUROC (combined 1.000, "
                        "inner_ring 0.750, outer_ring 0.641) but a real bimodal split underneath -- "
                        "15/26 damaged bearings near-perfect, 10/26 BELOW 0.5 AUROC (score direction "
                        "inverted, mostly artificial-damage bearings), leading hypothesis is that "
                        "crude 125x box-average decimation washes out the high-frequency impulsive "
                        "signature artificial single-point defects rely on -- see "
                        "memory/paderborn-fl-model-run.md for the full per-bearing table and "
                        "untested follow-ups. Separately, this is also the dataset Joint Prototype "
                        "Memory's typed-relation Scheme V3 was originally developed on and wins most "
                        "clearly on (0.904 vs. a 0.819 fair GDN baseline) -- see "
                        "memory/joint-prototype-scheme-v3.md and the "
                        "joint-prototype-scheme-v2/joint-prototype-scheme-v3 entries in "
                        "benchmark/baselines/registry.py.",
    },
    "voraus-AD": {
        "category": "A",
        "description": "vorausrobotik 6-DOF pick-and-place robot arm, 100Hz, 2122 pick-place-cycle "
                         "samples (2.32M rows), 12 anomaly categories (axis friction, axis weight, "
                         "3 collision types, missed/lost/heavy can, entangled cable, invalid "
                         "position, motor commutation fault, wobbling station) + normal operation. "
                         "Rich per-joint telemetry: 6 joints x 21 signals each (target/motor/joint "
                         "position/velocity/acceleration/torque, computed inertia/torque, dual "
                         "torque sensors A/B, motor Iq/Id current, electrical/mechanical power, "
                         "motor/supply/brake voltage) + 4 robot-level electrical signals "
                         "(robot voltage/current, IO current, system current). 948 samples are a "
                         "dedicated pure-normal training variant (PRE_A); the rest mix normal and "
                         "the 12 fault categories across 77 named settings/variants.",
        "num_features": "130 machine-data signals (4 robot-level + 6 joints x 21) + 7 meta columns, "
                          "verified directly against the parquet's columns (see "
                          "benchmark/datasets/voraus_ad_physics.md's feature table).",
        "used_in_papers": ["Brockmann, Rudolph, Rosenhahn, Wandt, IEEE T-RO 2023, arXiv:2311.04765 "
                             "(original reference paper, introduces MVT-Flow baseline)"],
        "why_relevant": "A 6-DOF articulated arm with an explicit KINEMATIC CHAIN (joint 1..6, each "
                          "with target vs. motor vs. joint-side position/velocity/torque and dual "
                          "redundant torque sensors A/B) -- the richest physics-graph structure of "
                          "any dataset in this project so far, well beyond robo3er's 2-wheel "
                          "differential-drive relation or Paderborn's single-bearing channel set. "
                          "Each joint is a natural graph node with well-defined intra-joint edges "
                          "(target->motor->joint position/velocity/torque chain, computed vs. "
                          "measured torque, electrical vs. mechanical power) AND inter-joint edges "
                          "(kinematic coupling along the arm) to declare as physics priors for "
                          "JointPrototypeGDNv3. 12 distinct, named fault categories with real "
                          "physical mechanisms (friction, added weight, 3 collision types, gripper/"
                          "part handling faults, motor commutation) give much finer-grained "
                          "comparison than Paderborn's 3-category damage taxonomy.",
        "status": "implemented",
        "path": "data/voraus_ad/voraus-ad-dataset-100hz.parquet (~1.1GB, the 100Hz variant used by "
                 "the reference repo; a 500Hz/~5.3GB variant also exists but was not fetched), "
                 "reference paper in docs/voraus_ad_paper.pdf.",
        "obtain": "https://media.vorausrobotik.com/voraus-ad-dataset-100hz.parquet (direct download, "
                   "no registration; code/loader reference at "
                   "https://github.com/vorausrobotik/voraus-ad-dataset). Dataset itself is CC "
                   "BY-NC-SA 4.0 (non-commercial), repo code is MIT.",
        "note_extra": "benchmark/datasets/voraus_ad_physics.md (physical relations transcribed from "
                        "the reference paper: target/motor/joint tracking chain, current->torque "
                        "(the paper's own worked 'proportional' example, with miscommutation defined "
                        "as exactly this relation breaking), redundant torque-sensor cross-check, "
                        "friction as a target/velocity->torque relation, the 3-stage power-conservation "
                        "chain, and the paper's own ablation finding that mechanical signals matter far "
                        "more than electrical ones) + benchmark/datasets/voraus_ad_adapter.py (66 nodes: "
                        "full target/motor/joint tracking chain + both torque sensors + Iq/Id current x "
                        "6 joints, 54 within-joint-only declared edges) + "
                        "benchmark/run_voraus_ad_joint_prototype_v3.py implement Scheme V3 here. FINAL "
                        "result: V2 (prototype+node, no edges) is the top path at 0.756 mean "
                        "AUROC, winning/tying on 11/12 fault categories -- the sole exception, "
                        "motor_commutation, is the one category whose fault is a textbook edge-relation "
                        "break. UNVALIDATED -- no fair GDN/AE baseline exists on this dataset yet. See "
                        "memory/joint-prototype-scheme-v3.md and memory/joint-prototype-scheme-v2.md for "
                        "the full cross-dataset conclusions (this dataset's iteration history is "
                        "condensed there, not kept separately).",
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
