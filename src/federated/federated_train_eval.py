"""
PROJECT MAINLINE (finalized 2026-08-30, FD+PD-only trim 2026-09-02 for public
release): the shipped combined anomaly score is **FD+PD = smooth_max(
FD_max, PD_max)** with **single-linkage BFS** connected
components (`federated_memory.align_and_split`'s `linkage="single"`, the
original/default behavior) -- this is now the ONLY detection score this
module computes. The earlier C/E/F/H/I/J signals and CK/HK/BCK combinations
(see `memory/scoring-signals-B-C-E-H.md`, `memory/forecast-head-signal-k.md`
for the ablation results that justified picking FD+PD) are retired from the
active pipeline; a verbatim snapshot of the code that produced them lives in
`archive/src/federated/legacy_scores.py` for anyone who wants to reproduce
that comparison. `linkage="complete"` remains available as an opt-in
alternative to the default single-linkage BFS. See `src/federated/pipeline.py`
for the single documented entry point that wires dataset selection together.

Node-level fault LOCALIZATION (a separate task from detection AUROC) still
uses C (`resid_struct`/`edge_head`) and H (`d_mahal`/`cov_head`) directly, via
`per_sample_scores` and the model's own heads, in
`benchmark/diagnose_*_localization_federated.py` -- those heads and this
module's `per_sample_scores` were NOT removed, only the detection-report
aggregation of C/E/F/H/I/J into extra columns was.

As of the 2026-08-30 finalization, `ForecastHead`'s parameters are ALSO
FedAvg'd whenever `sync_encoder_decoder=True` (robo3er, paderborn -- see
`run_federated_rounds`'s `fedavg_prefixes` default below): K is no longer
purely client-local. This was previously false (`forecast_head` was
explicitly excluded from the `("encoder.", "decoder.")` FedAvg prefix
filter) -- any memory file or docstring still saying "K/forecast_head is
never federated" predates this change and is stale. Sielaff is unaffected
(it never sets `sync_encoder_decoder=True`, so no prefix list is consulted
for it either way -- it keeps the original memory-only exchange design).

Shared federated training+eval "mainline" for `benchmark/run_{robo3er,
paderborn}_bck_federated.py` (share ALL of this module byte-for-byte);
`run_sielaff_red_bck_federated.py`/`run_alfa_bck_federated.py` use only
`run_federated_rounds` -- see that function's docstring for why.

Everything here was previously duplicated verbatim across the per-dataset
scripts. Two things stay OUT of this module on purpose, because they are
NOT actually identical across datasets (confirmed by direct diff, not
assumed):

- Sielaff/ALFA's per-node z-score + reliability-masking + plain-max scoring
  is a genuinely different aggregation algorithm from `two_stage_group_score`
  (top-k-mean + re-zscore), not the same function with different config --
  it stays local to their own `run_*_bck_federated.py`.
- Sielaff/ALFA's model is `JointPrototypeV21Forecast` (no typed_head, no
  resid_phys/r_edge outputs), so `per_sample_scores`/`train_local`'s loss
  composition here (which assumes V31Forecast's 7-tuple output and
  includes an `l_typed` term) does not apply to it -- they keep their own
  copies of those two functions as well.

robo3er and paderborn are both `JointPrototypeV31Forecast` with byte-
identical `train_local`/`per_sample_scores`/`forecast_scores`/`zscore`/
`topk_mean`/`two_stage_group_score`/`base_scores`/`add_forecast_scores`
bodies (confirmed by direct diff) -- those move here unmodified.
"""
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
from sklearn import metrics as sk_metrics

from federated_memory import (align_and_split, align_and_split_fedcc, fedavg_state_dict,  # noqa: E402
                               fedavg_state_dict_full, confidence_weights_from_losses)
from comm_cost import tensor_bytes, state_dict_bytes, round_comm_record  # noqa: E402


# ---------------------------------------------------------------------------
# Generic scoring primitives (V31Forecast-shaped: robo3er + paderborn)
# ---------------------------------------------------------------------------

def detection_metrics(scores_normal, scores_fault):
    """Threshold-free (AUROC/AUPRC) + threshold-based (precision/F1 at the
    F1-optimal operating point on the pooled normal+fault score
    distribution, via `sklearn.metrics.precision_recall_curve`) detection
    metrics for one signal. No single deployment alarm threshold is fixed
    by the paper's design (scores are compared as continuous distributions,
    e.g. via AUROC), so the F1-optimal point is reported as the best
    achievable operating point for that signal -- standard practice for
    comparing anomaly-scoring signals without assuming a shared alarm rate.
    Returns None if either array is empty (nothing to score)."""
    scores_normal = np.asarray(scores_normal)
    scores_fault = np.asarray(scores_fault)
    if len(scores_normal) == 0 or len(scores_fault) == 0:
        return None
    y = np.concatenate([np.zeros(len(scores_normal)), np.ones(len(scores_fault))])
    s = np.concatenate([scores_normal, scores_fault])
    auroc = float(sk_metrics.roc_auc_score(y, s))
    auprc = float(sk_metrics.average_precision_score(y, s))
    precision, recall, _ = sk_metrics.precision_recall_curve(y, s)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0)
    best = int(np.argmax(f1))
    return {"auroc": auroc, "auprc": auprc, "precision": float(precision[best]),
            "recall": float(recall[best]), "f1": float(f1[best])}


def zscore(x, calib_x):
    median = np.median(calib_x, axis=0)
    q75, q25 = np.percentile(calib_x, [75, 25], axis=0)
    iqr = np.maximum(q75 - q25, 1e-8)
    return (x - median) / iqr


def gdn_score(raw_calib, raw_x):
    """Vanilla-GDN-style deviation score (Deng & Hooi, AAAI 2021): per-node
    robust z-score against a SINGLE global reference computed once over
    `raw_calib` (median/IQR, via `zscore` above -- no per-prototype/regime
    conditioning), then max-aggregated across nodes. Used for K (`k_resid`)
    instead of `two_stage_group_score`'s two-stage top-k-mean + re-zscore:
    `forecast_head` is never FedAvg'd or memory-shared (see this module's
    docstring), so K deliberately keeps the simpler, ungrouped, single-stage
    GDN reference instead."""
    z_x = zscore(raw_x, raw_calib)
    return z_x.max(axis=1)


def topk_mean(x, k):
    k = min(k, x.shape[1])
    return np.sort(x, axis=1)[:, -k:].mean(axis=1)


def smooth_max(scores, T=1.0):
    """Softmax-weighted average over axis=1 -- a confidence-weighted
    drop-in replacement for `np.max`/`np.maximum` when combining DIFFERENT
    signals (e.g. B+C+K into BCK), as opposed to `topk_mean`/`gdn_score`'s
    per-node aggregation WITHIN one signal (left untouched, out of scope
    here). Each stacked signal is weighted by `softmax(score/T)`, so a
    signal that's only slightly behind the largest still contributes
    instead of being discarded outright the way hard max does. `T -> 0`
    recovers hard max exactly; `T -> inf` recovers a plain unweighted
    mean. `scores`: [N, k] (k signals stacked along axis=1)."""
    scores = np.asarray(scores)
    w = np.exp((scores - scores.max(axis=1, keepdims=True)) / T)
    w /= w.sum(axis=1, keepdims=True)
    return (w * scores).sum(axis=1)


def two_stage_group_score(raw_calib, raw_x, k):
    z_calib = zscore(raw_calib, raw_calib)
    z_x = zscore(raw_x, raw_calib)
    stat_calib = topk_mean(z_calib, k)
    stat_x = topk_mean(z_x, k)
    median = np.median(stat_calib)
    iqr = max(np.percentile(stat_calib, 75) - np.percentile(stat_calib, 25), 1e-8)
    return (stat_x - median) / iqr


# ---------------------------------------------------------------------------
# Per-prototype ("by operating regime") z-score calibration for B/K -- the
# same offline median/IQR-per-prototype-with-global-fallback pattern H's
# `DeviationCovarianceHead`/E's `TypedRelationAnomalyHead` have always used
# (see those classes' `set_calibration`), generalized here as free functions
# so B/K can opt into it without a new nn.Module/checkpoint format. This is
# a REVIVAL of the `ScoreCalibrationHead` design tried and removed on
# 2026-08-24 (`memory/calib-in-prototype-ab.md`) -- that experiment found a
# real but dataset-dependent effect (Paderborn/robo3er won, Sielaff lost,
# attributed to Sielaff's 16 prototypes x 10 thin per-machine calib splits
# starving individual slots of calib samples). Reimplemented as plain
# functions (not a class) since B/K's calibration was already
# ad-hoc/script-local, not a persisted model buffer -- no need to round-trip
# through a checkpoint the way H/E's DOES.
def fit_prototype_zscore_stats(raw_calib, idx_calib, num_prototypes, min_samples=5):
    """raw_calib: [Ncalib, F] (F=num_nodes for B's d_node, or num_nodes for
    K's k_resid -- same shape either signal already uses with plain
    `zscore`). idx_calib: [Ncalib] int, the prototype each calib window
    matched (`JointPrototypeMemory.forward`'s `idx` output). Returns a dict
    of per-prototype (median, IQR) plus the global fallback and a
    `valid` mask (prototype had >= min_samples calib windows), mirroring
    `DeviationCovarianceHead.set_calibration`'s fallback convention exactly."""
    med_g = np.median(raw_calib, axis=0)
    q75_g, q25_g = np.percentile(raw_calib, [75, 25], axis=0)
    iqr_g = np.maximum(q75_g - q25_g, 1e-8)

    F = raw_calib.shape[1]
    med_p = np.tile(med_g, (num_prototypes, 1))
    iqr_p = np.tile(iqr_g, (num_prototypes, 1))
    valid = np.zeros(num_prototypes, dtype=bool)
    for m in range(num_prototypes):
        mask = idx_calib == m
        n_m = int(mask.sum())
        if n_m >= min_samples:
            d_m = raw_calib[mask]
            med_p[m] = np.median(d_m, axis=0)
            q75_m, q25_m = np.percentile(d_m, [75, 25], axis=0)
            iqr_p[m] = np.maximum(q75_m - q25_m, 1e-8)
            valid[m] = True
    return {"med_p": med_p, "iqr_p": iqr_p, "valid": valid, "med_g": med_g, "iqr_g": iqr_g}


def zscore_prototype(x, idx_x, stats):
    """x: [N, F], idx_x: [N] int (each row's matched prototype). Z-scores
    each row against ITS OWN matched prototype's (median, IQR) from `stats`
    (see `fit_prototype_zscore_stats`), falling back row-by-row to the
    global stats for any prototype that didn't clear `min_samples` at fit
    time -- same per-row fallback selection `DeviationCovarianceHead.forward`
    does via `torch.where`, done here with `np.where` since B/K's calibration
    has always been plain numpy, never a model buffer."""
    valid_x = stats["valid"][idx_x]  # [N]
    med = np.where(valid_x[:, None], stats["med_p"][idx_x], stats["med_g"][None, :])
    iqr = np.where(valid_x[:, None], stats["iqr_p"][idx_x], stats["iqr_g"][None, :])
    return (x - med) / iqr


def two_stage_group_score_proto(raw_calib, idx_calib, raw_x, idx_x, k, num_prototypes, min_samples=5):
    """Prototype-conditioned counterpart of `two_stage_group_score`: ONLY the
    first-stage per-node z-score is made prototype-conditioned (matching
    2026-08-24's `calib-in-prototype-ab` design) -- the second stage
    (re-normalizing the aggregated top-k-mean statistic) stays a single
    global median/IQR over ALL calib windows regardless of prototype, since
    that stage's whole point is comparing windows to the SAME pooled
    reference after the first stage has already removed regime-conditional
    scale differences; conditioning it again would double-apply the
    correction."""
    stats = fit_prototype_zscore_stats(raw_calib, idx_calib, num_prototypes, min_samples)
    z_calib = zscore_prototype(raw_calib, idx_calib, stats)
    z_x = zscore_prototype(raw_x, idx_x, stats)
    stat_calib = topk_mean(z_calib, k)
    stat_x = topk_mean(z_x, k)
    median = np.median(stat_calib)
    iqr = max(np.percentile(stat_calib, 75) - np.percentile(stat_calib, 25), 1e-8)
    return (stat_x - median) / iqr


def gdn_score_proto(raw_calib, idx_calib, raw_x, idx_x, num_prototypes, min_samples=5):
    """Prototype-conditioned counterpart of `gdn_score` (used for B's
    `node_agg="max"` mode and for K): single-stage per-node z-score against
    the window's OWN matched prototype's calib stats, then max over nodes."""
    stats = fit_prototype_zscore_stats(raw_calib, idx_calib, num_prototypes, min_samples)
    z_x = zscore_prototype(raw_x, idx_x, stats)
    return z_x.max(axis=1)


def loss_weighted_combo(zscores: dict, weights: dict):
    """Fixed PER-CLIENT weighted sum of already-z-scored signals (e.g.
    B/H/K), using weights from `federated_memory.confidence_weights_
    from_losses` -- a fundamentally different fusion mechanism from
    `smooth_max`: `smooth_max` re-decides, PER SAMPLE, which signal to
    trust based on that sample's raw z-score magnitude (a "confidence"
    proxy that silently breaks down when a signal is simply noisy/
    poorly-calibrated for a given client, since noise inflates magnitude
    without correlating with true anomaly -- see the ALFA `elevator`
    case in `memory/robo-fleet-num-prototypes-sweep.md`'s discussion,
    where JD's small-sample-covariance noise got amplified by `smooth_max`
    on NORMAL windows and dragged FD+JD+PD below FD alone). This function
    instead uses ONE FIXED weight per signal per CLIENT, set by how well
    that signal was learned on that client's OWN calib split RELATIVE TO
    OTHER CLIENTS (`confidence_weights_from_losses`'s cross-client
    z-scoring) -- a client whose JD calibration is unreliable (e.g. too
    few calib windows per prototype for a stable covariance estimate)
    gets a uniformly LOW JD weight for every window it scores, rather than
    JD's per-window noise being free to hijack the softmax weight on
    whichever normal window happens to spike. All arrays in `zscores`
    must already be the same length (the caller is responsible for
    slicing every signal to a common mask, e.g. PD's forecast-pairing
    subset) and `weights` should sum to ~1 across its keys (guaranteed by
    `confidence_weights_from_losses`)."""
    total = None
    for k, w in weights.items():
        term = w * zscores[k]
        total = term if total is None else total + term
    return total


def base_scores(node_score):
    """Wraps FD into the dict-shaped interface `add_h_score`/`add_forecast_scores`
    expect. The full B/C/E/F/I/J version (and CK/HK/BCK) is archived verbatim
    in `archive/src/federated/legacy_scores.py` for anyone reproducing the
    retired ablation comparison; JD (`add_h_score` below) was promoted BACK
    out of that archive into the active mainline (FD+JD+PD, see module docstring)."""
    return {"FD_max": node_score}


def add_h_score(base, h_score, T=1.0):
    """Adds JD (`JD_mahal`, `cov_head`'s prototype-conditioned Mahalanobis
    deviation signal, already median/IQR-normalized by the caller -- see
    `DeviationCovarianceHead`) and `FD_JD_max = smooth_max(FD, JD)` to a
    `base_scores`-shaped dict. `h_score` must be FULL-length (same
    unmasked convention as `FD_max`) -- call this BEFORE
    `add_forecast_scores`, which then also emits `FD_JD_PD_max` automatically
    once it sees `JD_mahal` already present in `base`."""
    out = dict(base)
    out["JD_mahal"] = h_score
    out["FD_JD_max"] = smooth_max(np.stack([base["FD_max"], h_score], axis=1), T)
    return out


def add_forecast_scores(base, forecast_score, mask, T=1.0):
    """FD (and JD, if present) stay FULL-length (unmasked) in the output --
    only PD/FD+PD/FD+JD+PD use the forecast-pairing-masked subset. `base` here is
    the FULL, unmasked score dict; `mask` (over the same index order) is
    applied to a LOCAL copy only, never written back onto the full-length
    FD/JD entries themselves. `FD_JD_PD_max` is only added if `add_h_score` was
    already called on `base` (i.e. `JD_mahal` is present) -- callers
    that never calibrate `cov_head` simply keep getting FD/PD/FD+PD, unchanged."""
    b = base["FD_max"][mask]
    out = dict(base)
    out["PD_max"] = forecast_score
    out["FD_PD_max"] = smooth_max(np.stack([b, forecast_score], axis=1), T)
    if "JD_mahal" in base:
        h = base["JD_mahal"][mask]
        out["FD_JD_PD_max"] = smooth_max(np.stack([b, h, forecast_score], axis=1), T)
    return out


@torch.no_grad()
def per_sample_scores(model, windows, device, batch_size=256):
    """Version-agnostic (2026-09-12: the project mainline dropped V31's
    typed-edge head -- see `joint_prototype_model.py`'s module docstring
    and `memory/v21-mainline-switch.md`): `resid_phys`/`r_edge` (V31-only,
    the declared-physics-edge residual) are returned as `None` when the
    model has no `typed_head` (`JointPrototypeV21Forecast`, now used by
    EVERY dataset) instead of raising `KeyError` -- callers that only ever
    discard those two positions (every current caller) are unaffected."""
    model.eval()
    d_proto_all, d_node_all, resid_struct_all, resid_phys_all, idx_all, r_edge_all, d_mahal_all = [], [], [], [], [], [], []
    has_typed = None
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i : i + batch_size]).to(device)
        out = model(batch, training_mode=False)
        d_proto_all.append(out["d_proto"].cpu().numpy())
        d_node_all.append(out["d_node"].cpu().numpy())
        resid_struct_all.append(out["resid_struct"].cpu().numpy())
        if has_typed is None:
            has_typed = "r_edge" in out
        if has_typed:
            resid_phys_all.append(out["resid_phys"].cpu().numpy())
            r_edge_all.append(out["r_edge"].cpu().numpy())
        idx_all.append(out["idx"].cpu().numpy())
        d_mahal_all.append(out["d_mahal"].cpu().numpy())
    return (np.concatenate(d_proto_all), np.concatenate(d_node_all), np.concatenate(resid_struct_all),
            np.concatenate(resid_phys_all) if has_typed else None, np.concatenate(idx_all),
            np.concatenate(r_edge_all) if has_typed else None,
            np.concatenate(d_mahal_all))


@torch.no_grad()
def recon_eval_loss(model, x_in, device, batch_size=256):
    """Mean whole-window reconstruction MSE, version-agnostic (V21Forecast
    and V31Forecast both expose `x_hat` whenever `training_mode=True` --
    see `joint_prototype_model.py`), used ONLY as the IFCAAE baseline's
    cluster-assignment criterion (`run_federated_rounds(mode="ifcaae")`):
    hard-assign each client to whichever cluster's current global model
    reconstructs the client's OWN fit windows best, matching the original
    IFCA/IFCAAE design's `mean_recon_error` (pure reconstruction error, not
    this project's full composite training loss)."""
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(x_in), batch_size):
        batch = torch.from_numpy(x_in[i : i + batch_size]).to(device)
        out = model(batch, training_mode=True)
        err = (out["x_hat"] - batch).pow(2).mean(dim=(1, 2))
        total += err.sum().item()
        count += err.numel()
    return total / count if count else float("inf")


@torch.no_grad()
def per_node_recon_error(model, windows, device, batch_size=256):
    """[N_windows, num_nodes] per-node squared reconstruction error, mean
    over time -- feeds `gdn_score` for the FedAvg/IFCAAE `ReconOnlyModel`
    baseline (`src/models/baseline_models.py`)."""
    model.eval()
    errs = []
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i : i + batch_size]).to(device)
        out = model(batch, training_mode=True)
        err = (out["x_hat"] - batch).pow(2).mean(dim=1)  # mean over T -> [B, N]
        errs.append(err.cpu().numpy())
    return np.concatenate(errs, axis=0)


@torch.no_grad()
def per_node_exemplar_deviation(model, windows, device, batch_size=256):
    """[N_windows, num_nodes] per-node deviation from the matched exemplar
    (`d_node`) -- feeds `gdn_score` for the Fed-ExDNN `ExemplarOnlyModel`
    baseline (`src/models/baseline_models.py`)."""
    model.eval()
    d_node_all = []
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i : i + batch_size]).to(device)
        out = model(batch, training_mode=False)
        d_node_all.append(out["d_node"].cpu().numpy())
    return np.concatenate(d_node_all, axis=0)


def train_local_recon(model, fit_windows, epochs, device, lr=1e-3, batch_size=256):
    """Pure reconstruction MSE training loop for `ReconOnlyModel`
    (FedAvg/IFCAAE baselines) -- no VQ/edge/typed/forecast terms, matching
    the deleted `run_ifcaae_baseline.py`'s `train_local` faithfully."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(fit_windows)),
        batch_size=min(batch_size, len(fit_windows)), shuffle=True,
    )
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch, training_mode=True)
            loss = torch.nn.functional.mse_loss(out["x_hat"], batch)
            loss.backward()
            optimizer.step()
    return model


def train_local_exemplar(model, fit_windows, epochs, device, lr=1e-3, beta=0.25, batch_size=256):
    """Reconstruction + VQ commitment losses only, for `ExemplarOnlyModel`
    (Fed-ExDNN baseline) -- no edge/typed/forecast terms."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(fit_windows)),
        batch_size=min(batch_size, len(fit_windows)), shuffle=True,
    )
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch, training_mode=True)
            l_pred = torch.nn.functional.mse_loss(out["x_hat"], batch)
            l_me, l_mc = model.memory.commitment_losses(out["z"], out["p_star"])
            loss = l_pred + beta * l_me + l_mc
            loss.backward()
            optimizer.step()
    return model


def _perturb_state_dict(state_dict, std, generator=None):
    """Gaussian-perturb every floating-point tensor, scaled by that
    tensor's own std -- ported from the deleted `run_ifcaae_baseline.py`
    (`git show 9cc70fe`); integer buffers are copied unperturbed."""
    out = OrderedDict()
    for k, v in state_dict.items():
        if torch.is_floating_point(v):
            noise = torch.randn(v.shape, generator=generator).to(v.device) if generator is not None \
                else torch.randn_like(v)
            out[k] = v + noise * std * v.float().std(unbiased=False)
        else:
            out[k] = v.clone()
    return out


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
    """Version-agnostic (2026-09-12, see `per_sample_scores`'s docstring
    above): `lambda_typed * l_typed` (the V31-only declared-physics-edge
    residual loss) is only added when the model actually has a
    `typed_head` (`out` contains `"r_edge"`) -- `JointPrototypeV21Forecast`,
    now the project mainline for every dataset, has no such term, and
    `lambda_typed` is simply ignored (not an error) when passed to it."""
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
            l_typed = out["r_edge"].mean() if "r_edge" in out else None
            l_forecast = out["k_resid"].mean()
            loss = l_pred + beta * l_me + l_mc + lambda_edge * l_edge + lambda_forecast * l_forecast
            if l_typed is not None:
                loss = loss + lambda_typed * l_typed
            loss.backward()
            optimizer.step()
    return model


# ---------------------------------------------------------------------------
# Federated round loop (shared by ALL THREE datasets, including sielaff --
# this part genuinely is identical: local-train-per-client, then exchange
# codebook/usage_count via `align_and_split`, `load_memory`, optionally
# FedAvg the encoder/decoder).
# ---------------------------------------------------------------------------

def run_federated_rounds(clients, models, local_train_step, rounds, gamma, delta,
                          sync_encoder_decoder=False,
                          fedavg_prefixes=("encoder.", "decoder.", "forecast_head."),
                          linkage="single", mode="ours", num_clusters=2,
                          get_fit_windows=None, device=None,
                          ifcaae_warmup_frac=0.34, ifcaae_respawn_patience=1,
                          ifcaae_perturb_std=0.05,
                          metric="cosine", edge_quantile=None):
    """`local_train_step(client, model) -> fit_count:int` runs one client's
    local epochs (dataset-specific: which windows, which loss). Returns
    `fit_count` (used to weight the optional FedAvg step).

    `fedavg_prefixes` now includes `"forecast_head."` by default (project
    mainline finalization, see module docstring) -- K's forecast head is
    FedAvg'd alongside the shared encoder/decoder whenever
    `sync_encoder_decoder=True`. Only takes effect when
    `sync_encoder_decoder=True` is actually passed (robo3er, paderborn);
    Sielaff never sets that flag, so this default is inert for it and it
    keeps the original memory-only exchange.

    `mode` selects the server-aggregation strategy (all four share the
    identical per-round `local_train_step` call, differing only in what
    happens afterward):
      - "ours" (default): `align_and_split` codebook alignment (BFS over a
        pairwise-agreement graph, `metric`-selectable -- "cosine" default,
        byte-identical to before `metric`/`edge_quantile` existed), plus the
        optional prefix-FedAvg above. `edge_quantile`, if set, overrides
        `delta` with a per-round quantile-calibrated threshold (see
        `align_and_split`'s docstring) -- only meaningful for `mode="ours"`,
        ignored otherwise.
      - "fedavg": generic-FL baseline -- no codebook alignment at all;
        `fedavg_state_dict_full` averages EVERY parameter (including the
        memory codebook) into one single global model shared by all
        clients every round.
      - "fedexdnn": same loop as "ours" but calls `align_and_split_fedcc`
        (FedCC-style constrained-clustering alignment, a documented
        approximation -- see that function's docstring) instead of
        `align_and_split`.
      - "ifcaae": clustering-based baseline (Ghosh et al. IFCA, adapted
        unsupervised per `docs/baseline_selection_for_benchmark.md`) --
        maintains `num_clusters` full global model replicas, hard-assigns
        each client per round to whichever replica reconstructs the
        client's own fit windows best (`recon_eval_loss`), then
        `fedavg_state_dict_full`'s within each assigned cluster. Requires
        `get_fit_windows(client) -> np.ndarray` and `device` to be passed.
        Carries the warmup + respawn-on-idle collapse-prevention fixes
        ported from the deleted `run_ifcaae_baseline.py` (`git show
        9cc70fe`).

    Returns `diagnostics_log`: list of per-round diagnostics dicts (for
    "ours"/"fedexdnn", same shape `align_and_split` returns plus "round";
    for "fedavg"/"ifcaae", a smaller dict with "round" and, for "ifcaae",
    "cluster_assignment")."""
    if mode in ("ours", "fedexdnn"):
        align_fn = align_and_split if mode == "ours" else align_and_split_fedcc
        align_kwargs = dict(gamma=gamma, delta=delta, linkage=linkage)
        if mode == "ours":
            align_kwargs.update(metric=metric, edge_quantile=edge_quantile)
        diagnostics_log = []
        for rnd in range(1, rounds + 1):
            fit_counts = [local_train_step(c, m) for c, m in zip(clients, models)]

            codebooks = [m.memory.codebook.detach().clone() for m in models]
            usage_counts = [m.memory.usage_count.detach().clone() for m in models]
            # comm cost: every client uploads its own [M,N,D] codebook + [M]
            # usage counts, then downloads back its own aligned P_G,n (same
            # shape as its upload -- align_and_split never changes shape).
            bytes_up = sum(tensor_bytes(cb) + tensor_bytes(uc)
                            for cb, uc in zip(codebooks, usage_counts))
            P_G, diag = align_fn(codebooks, usage_counts, **align_kwargs)
            bytes_down = sum(tensor_bytes(p_g) for p_g in P_G)
            diag = {k: v for k, v in diag.items() if k != "per_cluster_per_node_agreement"}
            diag["round"] = rnd

            for model, p_g in zip(models, P_G):
                model.memory.load_memory(p_g)

            if sync_encoder_decoder:
                sds = [m.state_dict() for m in models]
                # every client uploads its matched-prefix subset; server
                # broadcasts back the SAME averaged dict to every client.
                bytes_up += sum(state_dict_bytes(sd, prefixes=fedavg_prefixes) for sd in sds)
                avg = fedavg_state_dict(sds, fit_counts, prefixes=fedavg_prefixes)
                bytes_down += len(models) * state_dict_bytes(avg, prefixes=fedavg_prefixes)
                for model in models:
                    model.load_state_dict(avg, strict=False)

            diag.update(round_comm_record(bytes_up, bytes_down))
            diagnostics_log.append(diag)
            print(f"  round {rnd}: {diag}")

        return diagnostics_log

    if mode == "fedavg":
        diagnostics_log = []
        for rnd in range(1, rounds + 1):
            fit_counts = [local_train_step(c, m) for c, m in zip(clients, models)]
            sds = [m.state_dict() for m in models]
            # every client uploads its FULL state_dict; server broadcasts
            # back the same averaged full model to every client.
            bytes_up = sum(state_dict_bytes(sd) for sd in sds)
            avg = fedavg_state_dict_full(sds, fit_counts)
            bytes_down = len(models) * state_dict_bytes(avg)
            for model in models:
                model.load_state_dict(avg)
            diag = {"round": rnd, "mode": "fedavg", **round_comm_record(bytes_up, bytes_down)}
            diagnostics_log.append(diag)
            print(f"  round {rnd}: {diag}")
        return diagnostics_log

    if mode == "ifcaae":
        if get_fit_windows is None or device is None:
            raise ValueError("mode='ifcaae' requires get_fit_windows and device")
        warmup_rounds = max(1, round(ifcaae_warmup_frac * rounds))
        cluster_params = None
        rounds_unused = [0] * num_clusters
        gen = torch.Generator().manual_seed(0)
        diagnostics_log = []
        assign_ids = [0] * len(clients)
        for rnd in range(1, rounds + 1):
            in_warmup = rnd <= warmup_rounds
            if cluster_params is None:
                cluster_params = [deepcopy(models[0].state_dict())]

            # comm cost (download): warmup broadcasts the single model to
            # every client; post-warmup broadcasts ALL num_clusters
            # replicas to every client so it can locally evaluate which
            # cluster fits best (standard IFCA protocol), not just the one
            # it ends up assigned to.
            if in_warmup:
                bytes_down = len(models) * state_dict_bytes(cluster_params[0])
            else:
                bytes_down = len(models) * sum(state_dict_bytes(p) for p in cluster_params)
            bytes_up = 0

            packages_by_cluster = {}
            for ci, (c, m) in enumerate(zip(clients, models)):
                if in_warmup:
                    cid = 0
                else:
                    losses = []
                    for params in cluster_params:
                        m.load_state_dict(params)
                        losses.append(recon_eval_loss(m, get_fit_windows(c), device))
                    cid = int(np.argmin(losses))
                assign_ids[ci] = cid
                m.load_state_dict(cluster_params[cid])
                fit_count = local_train_step(c, m)
                sd = m.state_dict()
                bytes_up += state_dict_bytes(sd)  # upload of this client's post-training model
                packages_by_cluster.setdefault(cid, []).append((sd, fit_count))

            for cid, packages in packages_by_cluster.items():
                state_dicts = [p[0] for p in packages]
                weights = [p[1] for p in packages]
                cluster_params[cid] = fedavg_state_dict_full(state_dicts, weights)

            diag = {"round": rnd, "mode": "ifcaae", "warmup": in_warmup,
                     "cluster_assignment": list(assign_ids),
                     **round_comm_record(bytes_up, bytes_down)}
            diagnostics_log.append(diag)
            print(f"  round {rnd}: {diag}")

            if rnd == warmup_rounds:
                base = cluster_params[0]
                cluster_params = [base] + [_perturb_state_dict(base, ifcaae_perturb_std, gen)
                                            for _ in range(num_clusters - 1)]
                rounds_unused = [0] * num_clusters
            elif rnd > warmup_rounds:
                picked = set(packages_by_cluster.keys())
                busiest = max(packages_by_cluster, key=lambda cid: len(packages_by_cluster[cid]))
                for cid in range(num_clusters):
                    if cid in picked:
                        rounds_unused[cid] = 0
                        continue
                    rounds_unused[cid] += 1
                    if rounds_unused[cid] >= ifcaae_respawn_patience:
                        cluster_params[cid] = _perturb_state_dict(cluster_params[busiest],
                                                                   ifcaae_perturb_std, gen)
                        rounds_unused[cid] = 0

        for ci, m in enumerate(models):
            m.load_state_dict(cluster_params[assign_ids[ci]])
        return diagnostics_log

    raise ValueError(f"unknown mode {mode!r}, expected 'ours', 'fedavg', 'ifcaae', or 'fedexdnn'")
