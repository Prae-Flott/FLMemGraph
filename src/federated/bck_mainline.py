"""
Shared federated training+eval "mainline" for the B/C/K/BK/CK/BCK family
of benchmark scripts (`benchmark/run_{robo3er,paderborn}_forecast_v2_federated.py`
share ALL of this module byte-for-byte; `run_sielaff_bck_federated.py`
uses only `run_federated_rounds` -- see that function's docstring for why).

Everything here was previously duplicated verbatim across the three
per-dataset scripts. Two things stay OUT of this module on purpose,
because they are NOT actually identical across datasets (confirmed by
direct diff, not assumed):

- Sielaff's per-node z-score + reliability-masking + plain-max scoring
  is a genuinely different aggregation algorithm from `two_stage_group_score`
  (top-k-mean + re-zscore), not the same function with different config --
  it stays local to `run_sielaff_bck_federated.py`.
- Sielaff's model is `JointPrototypeV21Forecast` (no typed_head, no
  resid_phys/r_edge outputs), so `per_sample_scores`/`train_local`'s loss
  composition here (which assumes V31Forecast's 7-tuple output and
  includes an `l_typed` term) does not apply to it -- it keeps its own
  copies of those two functions as well.

robo3er and paderborn are both `JointPrototypeV31Forecast` with byte-
identical `train_local`/`per_sample_scores`/`forecast_scores`/`zscore`/
`topk_mean`/`two_stage_group_score`/`base_scores`/`add_forecast_scores`
bodies (confirmed by direct diff) -- those move here unmodified.
"""
import numpy as np
import torch

from federated_memory import (align_and_split, fedavg_state_dict,  # noqa: E402
                               compute_prototype_dev_stats, compute_shrinkage_stats)


# ---------------------------------------------------------------------------
# Generic scoring primitives (V31Forecast-shaped: robo3er + paderborn)
# ---------------------------------------------------------------------------

def zscore(x, calib_x):
    median = np.median(calib_x, axis=0)
    q75, q25 = np.percentile(calib_x, [75, 25], axis=0)
    iqr = np.maximum(q75 - q25, 1e-8)
    return (x - median) / iqr


def topk_mean(x, k):
    k = min(k, x.shape[1])
    return np.sort(x, axis=1)[:, -k:].mean(axis=1)


def two_stage_group_score(raw_calib, raw_x, k, idx_calib=None, idx_x=None, dev_memory=None):
    if dev_memory is not None:
        z_calib = dev_memory.dev_zscore(raw_calib, idx_calib)
        z_x = dev_memory.dev_zscore(raw_x, idx_x)
    else:
        z_calib = zscore(raw_calib, raw_calib)
        z_x = zscore(raw_x, raw_calib)
    stat_calib = topk_mean(z_calib, k)
    stat_x = topk_mean(z_x, k)
    median = np.median(stat_calib)
    iqr = max(np.percentile(stat_calib, 75) - np.percentile(stat_calib, 25), 1e-8)
    return (stat_x - median) / iqr


def base_scores(node_score, edge_score, typed_score, cov_score):
    b = node_score
    return {
        "B_node_max": b,
        "C_struct_max": edge_score,
        "E_phys_max": typed_score,
        "F_v3_max": np.max(np.stack([node_score, edge_score, typed_score], axis=1), axis=1),
        "H_cov_mahal": cov_score,
        "I_node_cov_max": np.maximum(b, cov_score),
        "J_v3_cov_max": np.max(np.stack([node_score, edge_score, typed_score, cov_score], axis=1), axis=1),
    }


def add_forecast_scores(base, forecast_score, mask):
    """B/C/E/F/H/I/J stay FULL-length (unmasked) in the output -- only
    K/BK/CK/HK/BCK use the forecast-pairing-masked subset. `base` here is
    the FULL, unmasked score dict; `mask` (over the same index order) is
    applied to LOCAL copies only, never written back onto the full-length
    B/C/E/F/H/I/J entries themselves."""
    b, c, h = base["B_node_max"][mask], base["C_struct_max"][mask], base["H_cov_mahal"][mask]
    out = dict(base)
    out["K_forecast_max"] = forecast_score
    out["BK_max"] = np.maximum(b, forecast_score)
    out["CK_max"] = np.maximum(c, forecast_score)
    out["HK_max"] = np.maximum(h, forecast_score)
    out["BCK_max"] = np.max(np.stack([b, c, forecast_score], axis=1), axis=1)
    return out


@torch.no_grad()
def per_sample_scores(model, windows, device, batch_size=256):
    model.eval()
    d_proto_all, d_node_all, resid_struct_all, resid_phys_all, idx_all, r_edge_all, d_mahal_all = [], [], [], [], [], [], []
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i : i + batch_size]).to(device)
        out = model(batch, training_mode=False)
        d_proto_all.append(out["d_proto"].cpu().numpy())
        d_node_all.append(out["d_node"].cpu().numpy())
        resid_struct_all.append(out["resid_struct"].cpu().numpy())
        resid_phys_all.append(out["resid_phys"].cpu().numpy())
        idx_all.append(out["idx"].cpu().numpy())
        r_edge_all.append(out["r_edge"].cpu().numpy())
        d_mahal_all.append(out["d_mahal"].cpu().numpy())
    return (np.concatenate(d_proto_all), np.concatenate(d_node_all), np.concatenate(resid_struct_all),
            np.concatenate(resid_phys_all), np.concatenate(idx_all), np.concatenate(r_edge_all),
            np.concatenate(d_mahal_all))


@torch.no_grad()
def forecast_scores(model, x_in, x_future, device, batch_size=256):
    model.eval()
    k_resid_all = []
    for i in range(0, len(x_in), batch_size):
        xb = torch.from_numpy(x_in[i : i + batch_size]).to(device)
        fb = torch.from_numpy(x_future[i : i + batch_size]).to(device)
        out = model(xb, training_mode=False, x_future=fb)
        k_resid_all.append(out["k_resid"].cpu().numpy())
    return np.concatenate(k_resid_all)


def train_local(model, fit_in, fit_future, epochs, device, lr=1e-3, beta=0.25,
                 lambda_edge=0.5, lambda_typed=0.5, lambda_forecast=0.5, batch_size=256):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(fit_in), torch.from_numpy(fit_future)),
        batch_size=min(batch_size, len(fit_in)), shuffle=True,
    )
    model.train()
    for _ in range(epochs):
        for batch, future in loader:
            batch, future = batch.to(device), future.to(device)
            optimizer.zero_grad()
            out = model(batch, training_mode=True, x_future=future)
            l_pred = torch.nn.functional.mse_loss(out["x_hat"], batch)
            l_me, l_mc = model.memory.commitment_losses(out["z"], out["p_star"])
            l_edge = out["resid_struct"].mean()
            l_typed = out["r_edge"].mean()
            l_forecast = out["k_resid"].mean()
            loss = (l_pred + beta * l_me + l_mc + lambda_edge * l_edge
                    + lambda_typed * l_typed + lambda_forecast * l_forecast)
            loss.backward()
            optimizer.step()
    return model


# ---------------------------------------------------------------------------
# Federated round loop (shared by ALL THREE datasets, including sielaff --
# this part genuinely is identical: local-train-per-client, then exchange
# codebook/usage_count via `align_and_split`, `load_memory`, optionally
# collect shrinkage stats, optionally FedAvg the encoder/decoder).
# ---------------------------------------------------------------------------

def run_federated_rounds(clients, models, local_train_step, rounds, gamma, delta,
                          calib_mode="global", alpha=20.0, dev_stats_step=None,
                          sync_encoder_decoder=False, fedavg_prefixes=("encoder.", "decoder.")):
    """`local_train_step(client, model) -> fit_count:int` runs one client's
    local epochs (dataset-specific: which windows, which loss). Returns
    `fit_count` (used to weight the optional FedAvg step).

    `dev_stats_step(client, model) -> (n, mean, var)` (per
    `compute_prototype_dev_stats`'s return shape) is REQUIRED if
    `calib_mode == "shrinkage"` -- runs a one-shot local pass over the
    client's own fit data with the just-received aligned codebook (must
    happen after `load_memory()`, hence the callback rather than doing it
    up front).

    Returns `diagnostics_log`: list of per-round alignment diagnostics
    dicts (same shape `align_and_split` returns, plus `"round"` and,
    under shrinkage, `"shrinkage_alpha"`)."""
    diagnostics_log = []
    for rnd in range(1, rounds + 1):
        fit_counts = [local_train_step(c, m) for c, m in zip(clients, models)]

        codebooks = [m.memory.codebook.detach().clone() for m in models]
        usage_counts = [m.memory.usage_count.detach().clone() for m in models]
        P_G, diag = align_and_split(codebooks, usage_counts, gamma=gamma, delta=delta)
        diag = {k: v for k, v in diag.items() if k != "per_cluster_per_node_agreement"}
        diag["round"] = rnd
        diagnostics_log.append(diag)
        print(f"  round {rnd}: {diag}")

        for model, p_g in zip(models, P_G):
            model.memory.load_memory(p_g)

        if calib_mode == "shrinkage":
            client_dev_stats = [dev_stats_step(c, m) for c, m in zip(clients, models)]
            num_shared = diag["num_shared_prototypes"]
            shrink_stats, alpha_used = compute_shrinkage_stats(client_dev_stats, num_shared, alpha)
            diag["shrinkage_alpha"] = alpha_used.tolist()
            for model, (mean_s, var_s, valid_s) in zip(models, shrink_stats):
                model.memory.load_shrinkage_stats(mean_s, var_s, valid_s)

        if sync_encoder_decoder:
            avg = fedavg_state_dict([m.state_dict() for m in models], fit_counts,
                                     prefixes=fedavg_prefixes)
            for model in models:
                model.load_state_dict(avg, strict=False)

    return diagnostics_log
