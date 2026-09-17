#!/usr/bin/env python3
"""
Node-level DIAGNOSIS (root-cause localization) test for B/C/H/K/BK/BCK
(the mainline signal set -- I/CK/HK deliberately omitted, per user
direction) on ALFA, federated, 6 fault-type clients
(`run_alfa_bck_federated.py`'s architecture -- `JointPrototypeV21Forecast`,
no declared physics edges, memory-only exchange). Mirrors
`diagnose_robo3er_localization_federated.py`'s single-label per-fault-type
direct/indirect-domain table (ALFA windows are single-label, unlike
Sielaff's multi-label red windows), minus everything that script has for
V31's `typed_head`/E signal, which V21 doesn't have (same omission
`run_alfa_bck_federated.py`/`run_sielaff_red_bck_federated.py` already
make).

Same per-flight-continuity fix as `run_alfa_bck_federated.py`: forecast
pairing chains require the SAME `window_flight` id, not just same client,
since a client here pools multiple discontinuous flights.

## Ground truth: flight-dynamics axis domains, NOT RC-actuator channels

Each ALFA fault type maps to a specific control axis by textbook fixed-
wing aerodynamics (ailerons control roll, elevator controls pitch, rudder
controls yaw) -- unambiguous, unlike trying to identify WHICH of the 8
anonymous `rc_out_*` PWM channels is the aileron/elevator/rudder servo.
That channel mapping WAS attempted (correlating each `rc_out_i` against
roll/pitch/yaw/throttle on a `no_failure` flight): `rc_out_2` correlates
at r=1.00 with `throttle` (confirmed, used nowhere here since engine
failure's direct signal is the vehicle's flight-performance RESPONSE, not
the still-nominally-commanded throttle output -- see below), but
`rc_out_1`/`rc_out_{3,4,5}` gave ambiguous/degenerate correlations
(identical values across 3 channels, i.e. mixed/redundant outputs) --
inconclusive, so RC channels are deliberately NOT used as ground truth
here, only the unambiguous attitude-axis quantities are.

- `engine_failure`: direct = flight-PERFORMANCE response to lost thrust
  (`airspeed`, `groundspeed`, `altitude`, `climb` -- these decay/collapse
  when thrust is lost, regardless of what the throttle channel still
  commands). indirect = compensatory/consequential channels (`throttle`
  commanded value, `battery_current/voltage`, `aspd_error`, `alt_error`,
  `pitch` glide compensation).
- `aileron_failure`: direct = `roll`, `ang_vel_x` (the roll axis ailerons
  control). indirect = lateral consequences (`heading`, `xtrack_error`,
  `pos_y`, `vel_ang_x`, `path_dev_y`).
- `rudder_failure`: direct = `yaw`, `ang_vel_z`. indirect = `heading`,
  `xtrack_error`, `vel_ang_z`, `path_dev_y`.
- `elevator_failure`: direct = `pitch`, `ang_vel_y`. indirect = `altitude`,
  `climb`, `alt_error`, `vel_lin_z`, `path_dev_z`.
- `aileron_rudder_combo_failure`: direct = union of aileron's and
  rudder's direct sets; indirect = union of both indirect sets.

Usage:
    python3 src/dataloaders/alfa/build_alfa.py   # once, to build data/alfa/
    python3 diagnose_alfa_localization_federated.py [--horizon-mult M] [--rounds R] [--local-epochs E] [--out-suffix NAME]
"""
import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT.parent / "checkpoints" / "alfa"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT.parent / "src" / "models"))
sys.path.insert(0, str(REPO_ROOT.parent / "src" / "federated"))

import importlib.util
spec = importlib.util.spec_from_file_location("alfa_bck_fed", REPO_ROOT / "run_alfa_bck_federated.py")
v2f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2f)

RELIABILITY_RATIO = 0.05

# ---------------------------------------------------------------- ground truth --
_AILERON_DIRECT = {"roll", "ang_vel_x"}
_AILERON_INDIRECT = {"heading", "xtrack_error", "pos_y", "vel_ang_x", "path_dev_y"}
_RUDDER_DIRECT = {"yaw", "ang_vel_z"}
_RUDDER_INDIRECT = {"heading", "xtrack_error", "vel_ang_z", "path_dev_y"}

DIRECT_DOMAIN = {
    "engine_failure": {"airspeed", "groundspeed", "altitude", "climb"},
    "aileron_failure": _AILERON_DIRECT,
    "rudder_failure": _RUDDER_DIRECT,
    "elevator_failure": {"pitch", "ang_vel_y"},
    "aileron_rudder_combo_failure": _AILERON_DIRECT | _RUDDER_DIRECT,
}
INDIRECT_DOMAIN = {
    "engine_failure": {"throttle", "battery_current", "battery_voltage", "aspd_error", "alt_error", "pitch"},
    "aileron_failure": _AILERON_INDIRECT,
    "rudder_failure": _RUDDER_INDIRECT,
    "elevator_failure": {"altitude", "climb", "alt_error", "vel_lin_z", "path_dev_z"},
    "aileron_rudder_combo_failure": _AILERON_INDIRECT | _RUDDER_INDIRECT,
}


def classify(name, fault):
    if name in DIRECT_DOMAIN.get(fault, ()):
        return "direct"
    if name in INDIRECT_DOMAIN.get(fault, ()):
        return "indirect"
    return "other"


# --------------------------------------------------------------------- data --
class Client:
    def __init__(self, client_id, client_name, all_idx, fit_idx, calib_idx, test_normal_idx, fault_idx_by_label):
        self.client_id = client_id
        self.client_name = client_name
        self.all_idx = all_idx
        self.fit_idx = fit_idx
        self.calib_idx = calib_idx
        self.test_normal_idx = test_normal_idx
        self.fault_idx_by_label = fault_idx_by_label


def build_clients():
    data, targets, cols, label_map, client_names, base_clients, window_flight = v2f.build_clients()
    clients = []
    for bc in base_clients:
        fault_idx_by_label = {}
        for label_id_str in label_map:
            label_id = int(label_id_str)
            if label_id == 0:
                continue
            fault_idx = np.intersect1d(np.where(targets == label_id)[0], bc.all_idx)
            if len(fault_idx):
                fault_idx_by_label[label_id] = fault_idx
        clients.append(Client(bc.client_id, bc.client_name, bc.all_idx, bc.fit_idx,
                               bc.calib_idx, bc.test_normal_idx, fault_idx_by_label))
    return data, targets, cols, label_map, client_names, clients, window_flight


def h_contributions(model, d_node, idx):
    cov_head = model.cov_head
    valid = cov_head.calib_valid.cpu().numpy()[idx]
    mu = np.where(valid[:, None], cov_head.calib_mu.cpu().numpy()[idx],
                  cov_head.global_mu.cpu().numpy()[None, :])
    cov_inv = np.where(valid[:, None, None], cov_head.calib_cov_inv.cpu().numpy()[idx],
                        cov_head.global_cov_inv.cpu().numpy()[None, :, :])
    diff = d_node - mu
    weighted = np.einsum("bij,bj->bi", cov_inv, diff)
    return diff * weighted


def zn(x, med, iqr):
    return (x - med) / iqr


def masked_argmax(z, reliable):
    z_masked = np.where(reliable[None, :], z, -np.inf)
    return np.argmax(z_masked, axis=1)


def combo_conf(scalars, per_node_arrays, reliable_masks, T=1.0):
    if len(scalars[0]) == 0:
        return np.zeros(0, dtype=int)
    stacked_scalar = np.stack(scalars, axis=1)
    w = np.exp((stacked_scalar - stacked_scalar.max(axis=1, keepdims=True)) / T)
    w /= w.sum(axis=1, keepdims=True)
    stacked_pernode = np.stack(per_node_arrays, axis=0)
    combined = np.einsum("nk,knd->nd", w, stacked_pernode)
    reliable = np.all(np.stack(reliable_masks, axis=0), axis=0)
    return masked_argmax(combined, reliable)


def main(horizon_mult=1, rounds=None, local_epochs=None, out_suffix=None,
         num_prototypes=v2f.NUM_PROTOTYPES, linkage="single"):
    torch.manual_seed(v2f.SEED)
    np.random.seed(v2f.SEED)
    forecast_h = v2f.STRIDE * horizon_mult
    rounds = rounds if rounds is not None else v2f.ROUNDS
    local_epochs = local_epochs if local_epochs is not None else v2f.LOCAL_EPOCHS

    print(f"loading ALFA, 6 fault-type federated clients, horizon_mult={horizon_mult}, "
          f"rounds={rounds}, local_epochs={local_epochs} ...")
    data, targets, cols, label_map, client_names, clients, window_flight = build_clients()
    num_nodes = len(cols)
    print(f"num_nodes={num_nodes}")
    for c in clients:
        print(f"  client {c.client_id} ({c.client_name}): fit={len(c.fit_idx)} calib={len(c.calib_idx)} "
              f"test_normal={len(c.test_normal_idx)} "
              f"faults={ {label_map[str(k)]: len(v) for k, v in c.fault_idx_by_label.items()} }")

    scalers = [v2f.fit_client_scaler(data, c.fit_idx) for c in clients]
    models = [v2f.JointPrototypeV21Forecast(num_nodes=num_nodes, window_size=v2f.WINDOW_LEN,
                                             embed_dim=v2f.EMBED_DIM, num_prototypes=num_prototypes,
                                             forecast_h=forecast_h, prior_edges=None, top_k=v2f.TOP_K).to(v2f.DEVICE)
              for _ in clients]

    shared_init = {k: v.clone() for k, v in models[0].state_dict().items() if not k.startswith("memory.")}
    for model in models[1:]:
        model.load_state_dict(shared_init, strict=False)

    fit_pairs = []
    for c in clients:
        vi, chains, _ = v2f.build_pairs(c.fit_idx, targets, window_flight, horizon_mult, True, True)
        fit_pairs.append((vi, data[vi], v2f.gather_future(data, chains, v2f.STRIDE)))

    def local_train_step(c, model):
        idx = c.client_id
        scaler = scalers[idx]
        vi, x_in_raw, x_future_raw = fit_pairs[idx]
        n, t, f = x_in_raw.shape
        x_in = scaler.transform(x_in_raw.reshape(-1, f)).reshape(n, t, f).astype(np.float32)
        n2, t2, f2 = x_future_raw.shape
        x_future = (scaler.transform(x_future_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
                    if n2 > 0 else x_future_raw)
        v2f.train_local(model, x_in, x_future, local_epochs)
        return len(x_in)

    print(f"\nfederated training: {rounds} rounds x {local_epochs} local epochs, memory-only exchange ...")
    from federated_train_eval import run_federated_rounds  # noqa: E402
    run_federated_rounds(clients, models, local_train_step, rounds, v2f.GAMMA, v2f.DELTA,
                          sync_encoder_decoder=False, linkage=linkage)

    print("\ncalibrating cov_head per client ...")
    calib_stats = {}
    cov_min = max(5, num_nodes + 1)
    for c, model, scaler in zip(clients, models, scalers):
        calib_w = v2f.scale_client(data, scaler, c.calib_idx)
        _, d_node_calib_b, _, calib_idx_b, _ = v2f.per_sample_scores(model, calib_w)
        model.cov_head.set_calibration(d_node_calib_b, calib_idx_b, min_samples=cov_min)

        _, d_node_calib, resid_struct_calib, _, d_mahal_calib = v2f.per_sample_scores(model, calib_w)

        vi, chains, _ = v2f.build_pairs(c.calib_idx, targets, window_flight, horizon_mult, True, True)
        if len(vi):
            calib_in = v2f.scale_client(data, scaler, vi)
            calib_future_raw = v2f.gather_future(data, chains, v2f.STRIDE)
            n2, t2, f2 = calib_future_raw.shape
            calib_future = scaler.transform(calib_future_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
            k_resid_calib = v2f.forecast_scores(model, calib_in, calib_future)
        else:
            k_resid_calib = np.zeros((0, num_nodes), dtype=np.float32)

        node_median = np.median(d_node_calib, axis=0)
        node_q75, node_q25 = np.percentile(d_node_calib, [75, 25], axis=0)
        node_iqr = np.maximum(node_q75 - node_q25, 1e-8)
        struct_median = np.median(resid_struct_calib, axis=0)
        struct_q75, struct_q25 = np.percentile(resid_struct_calib, [75, 25], axis=0)
        struct_iqr = np.maximum(struct_q75 - struct_q25, 1e-8)
        if len(k_resid_calib):
            k_median = np.median(k_resid_calib, axis=0)
            k_q75, k_q25 = np.percentile(k_resid_calib, [75, 25], axis=0)
            k_iqr = np.maximum(k_q75 - k_q25, 1e-8)
        else:
            k_median, k_iqr = np.zeros(num_nodes), np.ones(num_nodes)

        node_reliable = node_iqr >= RELIABILITY_RATIO * np.median(node_iqr)
        struct_reliable = struct_iqr >= RELIABILITY_RATIO * np.median(struct_iqr)
        k_reliable = k_iqr >= RELIABILITY_RATIO * np.median(k_iqr)
        d_mahal_med = np.median(d_mahal_calib)
        d_mahal_iqr = max(np.percentile(d_mahal_calib, 75) - np.percentile(d_mahal_calib, 25), 1e-8)

        calib_stats[c.client_id] = dict(
            node_median=node_median, node_iqr=node_iqr, node_reliable=node_reliable,
            struct_median=struct_median, struct_iqr=struct_iqr, struct_reliable=struct_reliable,
            k_median=k_median, k_iqr=k_iqr, k_reliable=k_reliable,
            d_mahal_med=d_mahal_med, d_mahal_iqr=d_mahal_iqr,
        )
        print(f"  {c.client_name}: reliable nodes B/H={node_reliable.sum()}/{num_nodes}  "
              f"C={struct_reliable.sum()}/{num_nodes}  K={k_reliable.sum()}/{num_nodes}")

    print(f"\n{'fault':<28}{'signal':<8}{'n':<6}{'direct%':<10}{'direct_or_indirect%':<20}")
    pooled = {}
    for c, model, scaler in zip(clients, models, scalers):
        if not c.fault_idx_by_label:
            continue
        st = calib_stats[c.client_id]
        for label_id, fault_idx in c.fault_idx_by_label.items():
            fault_name = label_map[str(label_id)]
            fault_w = v2f.scale_client(data, scaler, fault_idx)
            _, d_node_f, resid_struct_f, idx_f, d_mahal_f = v2f.per_sample_scores(model, fault_w)

            z_node_f = zn(d_node_f, st["node_median"], st["node_iqr"])
            z_struct_f = zn(resid_struct_f, st["struct_median"], st["struct_iqr"])
            contrib_h_f = h_contributions(model, d_node_f, idx_f)
            z_mahal_f = zn(d_mahal_f, st["d_mahal_med"], st["d_mahal_iqr"])

            argmax_b = masked_argmax(z_node_f, st["node_reliable"])
            argmax_c = masked_argmax(z_struct_f, st["struct_reliable"])
            argmax_h = masked_argmax(contrib_h_f, st["node_reliable"])

            vi_f, chains_f, mask_f = v2f.build_pairs(fault_idx, targets, window_flight, horizon_mult, False, False)
            if len(vi_f):
                fault_in = v2f.scale_client(data, scaler, vi_f)
                future_f_raw = v2f.gather_future(data, chains_f, v2f.STRIDE)
                n2, t2, f2 = future_f_raw.shape
                fault_future = scaler.transform(future_f_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
                k_resid_f = v2f.forecast_scores(model, fault_in, fault_future)
                z_k_f = zn(k_resid_f, st["k_median"], st["k_iqr"])
                argmax_k_paired = masked_argmax(z_k_f, st["k_reliable"])
            else:
                z_k_f = np.zeros((0, num_nodes))
                argmax_k_paired = np.zeros(0, dtype=int)

            def add(fault_name, key, argmax_idx):
                pooled.setdefault(fault_name, {}).setdefault(key, []).extend(cols[i] for i in argmax_idx)

            add(fault_name, "B", argmax_b)
            add(fault_name, "C", argmax_c)
            add(fault_name, "H", argmax_h)
            add(fault_name, "K", argmax_k_paired)

            b_paired = z_node_f.max(axis=1)[mask_f]
            c_paired = z_struct_f.max(axis=1)[mask_f]
            k_scalar_paired = z_k_f.max(axis=1) if len(z_k_f) else np.zeros(0)
            b_pernode_paired = z_node_f[mask_f]
            c_pernode_paired = z_struct_f[mask_f]

            add(fault_name, "BK", combo_conf([b_paired, k_scalar_paired], [b_pernode_paired, z_k_f],
                                              [st["node_reliable"], st["k_reliable"]]))
            add(fault_name, "BCK", combo_conf([b_paired, c_paired, k_scalar_paired],
                                               [b_pernode_paired, c_pernode_paired, z_k_f],
                                               [st["node_reliable"], st["struct_reliable"], st["k_reliable"]]))

    results = {}
    for fault_name, sigs in pooled.items():
        for key, names in sigs.items():
            n = len(names)
            classes = [classify(name, fault_name) for name in names]
            counts = Counter(classes)
            direct_pct = 100.0 * counts["direct"] / n if n else 0.0
            combined_pct = 100.0 * (counts["direct"] + counts["indirect"]) / n if n else 0.0
            print(f"{fault_name:<28}{key:<8}{n:<6}{direct_pct:<10.1f}{combined_pct:<20.1f}")
            results.setdefault(fault_name, {})[key] = {
                "n": n, "direct_pct": round(direct_pct, 1), "direct_or_indirect_pct": round(combined_pct, 1),
                "top_nodes": Counter(names).most_common(5),
            }

    # Random-guess baseline per fault, for comparison against the table above.
    baseline = {}
    for fault_name in DIRECT_DOMAIN:
        direct_size = len(DIRECT_DOMAIN[fault_name])
        combined_size = len(DIRECT_DOMAIN[fault_name] | INDIRECT_DOMAIN.get(fault_name, set()))
        baseline[fault_name] = {
            "direct_pct": round(100.0 * direct_size / num_nodes, 1),
            "direct_or_indirect_pct": round(100.0 * combined_size / num_nodes, 1),
        }
    print(f"\nrandom-guess baseline (direct%/direct-or-indirect%, {num_nodes} nodes):")
    for fault_name, b in baseline.items():
        print(f"  {fault_name:<28}{b['direct_pct']:<10}{b['direct_or_indirect_pct']}")

    results["_baseline"] = baseline
    results["_config"] = {
        "num_nodes": num_nodes, "cols": cols, "horizon_mult": horizon_mult,
        "rounds": rounds, "local_epochs": local_epochs, "reliability_ratio": RELIABILITY_RATIO,
        "linkage": linkage, "num_prototypes": num_prototypes,
    }
    suffix = out_suffix if out_suffix is not None else f"h{horizon_mult}"
    out_path = OUT_DIR / f"alfa_diagnosis_localization_federated_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-mult", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--num-prototypes", type=int, default=v2f.NUM_PROTOTYPES)
    parser.add_argument("--out-suffix", type=str, default=None)
    parser.add_argument("--linkage", type=str, default="single", choices=["single", "complete"])
    args = parser.parse_args()
    main(horizon_mult=args.horizon_mult, rounds=args.rounds, local_epochs=args.local_epochs,
         out_suffix=args.out_suffix, num_prototypes=args.num_prototypes, linkage=args.linkage)
