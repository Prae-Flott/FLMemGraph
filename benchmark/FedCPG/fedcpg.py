"""
FedCPG (Li et al., *Computers in Industry* 164 (2025) 104180, "A class
prototype guided personalized lightweight federated learning framework
for cross-factory fault detection" -- PDF at
`docs/1-s2.0-S0166361524001088-main.pdf`). Supervised multi-class fault
classification, model-decoupled: `src/models/baseline_models.py::FedCPGModel`
splits into a FedAvg'd backbone (`encoder`+`proj`) and a personalized,
never-aggregated `head`. See `benchmark/run_robo_fleet_bck_federated.py`'s
`main_fedcpg_baseline` for the full pipeline this module's functions are
called from -- robo_fleet-only, same fit-on-ALL-labels exception as
FedPRO (`fedpro.py`'s module docstring), since class-prototype contrast
requires multiple known classes at training time.

Two prototype losses (paper Sec 3.4.2, Eq. (12)-(14)), both operating in
`r(x)` = `FedCPGModel.represent(x)`'s projection-network output space:
  - `l_g` (Eq. 12): supervised contrastive loss pulling each sample's
    `r(x)` toward the GLOBAL class prototype of its own label (server-
    aggregated across all clients, Eq. (3)-(4)) and away from other
    classes' global prototypes -- the mechanism that lets a client learn
    from other factories' data without seeing it directly.
  - `l_p` (Eq. 13): the same contrastive form but against that CLIENT'S
    OWN LOCAL class prototypes (recomputed once per local epoch from the
    client's current encoder, frozen for that epoch's SGD steps) -- keeps
    the personalized head from drifting entirely to the global model's
    class geometry.
Total local objective (Eq. 14): `l_ce + alpha*l_g + beta*l_p`.
"""
import numpy as np
import torch
import torch.nn.functional as F


def compute_class_prototypes(r, y, num_classes):
    """Per-class mean of `r` (the representation-space tensor, paper's
    `r(x)`), paper Eq. (3). Returns `(protos [num_classes, proj_dim] or
    None for empty rows, counts [num_classes])` -- classes absent from
    `y` get an all-zero prototype and 0 count (excluded from aggregation
    weighting by the caller, same as the paper's `|D_i,j|`-weighted
    average naturally zeroing out absent classes)."""
    proj_dim = r.shape[1]
    protos = torch.zeros(num_classes, proj_dim, device=r.device)
    counts = torch.zeros(num_classes, device=r.device)
    for c in range(num_classes):
        mask = y == c
        if mask.any():
            protos[c] = r[mask].mean(dim=0)
            counts[c] = mask.sum()
    return protos, counts


def aggregate_global_prototypes(client_protos, client_counts):
    """Server-side weighted average across clients, paper Eq. (4):
    `P_bar^(j) = sum_i (|D_i,j| / N_j) * P_i^(j)`. `client_protos`/
    `client_counts`: lists of this round's per-client
    `compute_class_prototypes` outputs."""
    stacked_protos = torch.stack(client_protos)  # [K, C, D]
    stacked_counts = torch.stack(client_counts)  # [K, C]
    totals = stacked_counts.sum(dim=0, keepdim=True).clamp_min(1e-8)  # [1, C]
    weights = (stacked_counts / totals).unsqueeze(-1)  # [K, C, 1]
    return (stacked_protos * weights).sum(dim=0)  # [C, D]


def prototype_contrastive_loss(r, y, protos, valid_classes, tau=0.1):
    """Shared form of `l_g`/`l_p` (Eq. (12)/(13)): softmax-temperature
    contrast of each sample's representation against ALL classes'
    prototypes, cross-entropy against its own label -- i.e. `r @ protos.T
    / tau` used as logits. `valid_classes`: bool mask of classes with a
    real (non-empty) prototype this round; samples whose own label lacks
    one are skipped (can't be positive against a prototype that doesn't
    exist)."""
    sample_valid = valid_classes[y]
    if not sample_valid.any():
        return r.new_zeros(())
    r, y = r[sample_valid], y[sample_valid]
    logits = (r @ protos.T) / tau
    logits = logits.masked_fill(~valid_classes.unsqueeze(0), float("-inf"))
    return F.cross_entropy(logits, y)


def train_local_fedcpg(model, x_in, y_in, epochs, device, global_protos, num_classes,
                        lr=1e-3, batch_size=256, alpha=0.5, beta=0.2, tau=0.1):
    """One client's local FedCPG training round (paper Eq. (14)). Local
    class prototypes are recomputed once per epoch (frozen for that
    epoch's mini-batches, matching the paper's per-round prototype-then-
    train structure) from the CURRENT encoder -- cheap relative to a full
    forward pass per batch. Returns this round's own
    `(local_protos, local_counts)` for the caller to fold into next
    round's global aggregation (Eq. (4))."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x_t = torch.from_numpy(x_in).to(device)
    y_t = torch.from_numpy(y_in).to(device)
    global_protos = global_protos.to(device)
    global_valid = global_protos.abs().sum(dim=1) > 0

    model.train()
    local_protos, local_counts = None, None
    for _ in range(epochs):
        with torch.no_grad():
            r_all, _ = model.represent(x_t)
            local_protos, local_counts = compute_class_prototypes(r_all, y_t, num_classes)
        local_valid = local_counts > 0

        perm = torch.randperm(len(x_t))
        for start in range(0, len(x_t), batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = x_t[idx], y_t[idx]
            r, z = model.represent(xb)
            logits = model.head(z)
            l_ce = F.cross_entropy(logits, yb)
            l_g = prototype_contrastive_loss(r, yb, global_protos, global_valid, tau)
            l_p = prototype_contrastive_loss(r, yb, local_protos, local_valid, tau)
            loss = l_ce + alpha * l_g + beta * l_p

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return local_protos.detach().cpu(), local_counts.detach().cpu()


def predict_anomaly_score(model, x_in, device, batch_size=256):
    """Bridged detection score for FedCPG's own classifier, matching
    FedPRO's convention (`1 - P(normal)`) -- FedCPG's own paper only
    reports classification Acc/AUC/F1 (multi-class, not fit-on-healthy
    detection), so this is OUR bridge to this project's AUROC convention,
    not part of FedCPG itself."""
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(x_in), batch_size):
            xb = torch.from_numpy(x_in[start:start + batch_size]).to(device)
            probs = F.softmax(model(xb), dim=1)
            scores.append((1.0 - probs[:, 0]).cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros(0)
