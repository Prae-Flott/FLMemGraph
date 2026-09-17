"""
FedPRO (Zhou et al., IEEE TC 2026, "Prototype Retrieval-Augmented
Federated Learning System for Robust Intrusion Detection" -- PDF at
`docs/Prototype_Retrieval-Augmented_Federated_Learning_System_for_Robust_Intrusion_Detection.pdf`,
official code `github.com/zza234s/FedPRO`): a test-time, plug-and-play
wrapper around an ALREADY-TRAINED off-the-shelf FL classifier
(`src/models/baseline_models.py::FedPROModel`, trained via plain FedAvg --
see that class's docstring), not a standalone trainable FL algorithm.

Unlike this project's other three baselines (FedAvg/IFCAAE/Fed-ExDNN, all
unsupervised fit-on-healthy anomaly scoring), FedPRO is inherently
SUPERVISED multi-class classification: its margin-based prototype
optimization (Eq. 5) requires multiple KNOWN classes to discriminate
between at training time, which collapses to nothing under fit-on-healthy
(only ever "normal" during training). Per the user's explicit direction,
this is scoped to robo_fleet ONLY, trained on ALL of robo_fleet's labeled
data (Normal + its 4 induced fault types as 5 classes) rather than the
project's usual fit-on-healthy split -- a deliberate, documented exception
to the federated-only benchmark policy, because FedPRO has no other
faithful realization. See `benchmark/run_robo_fleet_bck_federated.py`'s
`main_fedpro_baseline` for the full pipeline this module's functions are
called from.

Three stages (paper Sec. III, Algorithm 1):
  1. `build_initial_prototypes_kmeans` -- per-client, per-class clustering
     of the (frozen, post-FedAvg-training) encoder's embeddings, Eq. 3-4.
     The paper's own clustering algorithm is FINCH (parameter-free); we
     substitute a small fixed K-means instead, since FINCH has no
     maintained Python package matching this repo's dependency footprint.
     This substitution is paper-validated, not a fabrication: the paper's
     own ablation (Table V, Sec. IV-H) finds K-means "performs
     competitively" with FINCH (89.57 vs 86.66 global-test accuracy, 96.73
     vs 97.31 personalized-test).
  2. `refine_prototypes_margin` -- per-client margin-based hinge-loss
     refinement of ONLY the prototype tensor (encoder stays frozen), Eq. 5.
  3. `retrieve_and_vote` + `confidence_ensemble` -- test-time retrieval
     from the GLOBAL prototype bank (all clients' refined prototypes
     concatenated by the caller, NO cross-client alignment -- a deliberate
     contrast with this project's own `align_and_split`/Fed-ExDNN's FedCC,
     see the paper's Fig. 3(a): the server just concatenates), weighted
     voting (Eq. 8-9), and confidence-weighted ensemble with the trained
     classifier's own prediction (Eq. 10).
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


def train_local_classifier(model, x_in, y_in, epochs, device, lr=1e-3, batch_size=256):
    """Cross-entropy training loop for `FedPROModel` -- the "off-the-shelf
    FL classifier" stage (paper Sec. III-A/Algorithm 1's prerequisite),
    driven by `run_federated_rounds(mode="fedavg")` exactly like the
    unsupervised baselines' `train_local_recon`/`train_local_exemplar`."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x_in), torch.from_numpy(y_in)),
        batch_size=min(batch_size, len(x_in)), shuffle=True,
    )
    model.train()
    for _ in range(epochs):
        for batch, labels in loader:
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def embed_all(model, x_in, device, batch_size=256):
    """[N, embed_dim] embeddings via the (frozen) encoder half of
    `FedPROModel` -- shared by prototype building and retrieval."""
    model.eval()
    zs = []
    for i in range(0, len(x_in), batch_size):
        batch = torch.from_numpy(x_in[i : i + batch_size]).to(device)
        zs.append(model.embed(batch).cpu().numpy())
    return np.concatenate(zs, axis=0)


def build_initial_prototypes_kmeans(embeddings, labels, num_classes, k_per_class=3, min_per_cluster=10):
    """Eq. 3-4: per-class clustering of this client's own training
    embeddings into initial prototypes. K-means substitute for FINCH (see
    module docstring) -- `k` for class `s` is `min(k_per_class,
    max(1, n_s // min_per_cluster))`, so a thinly-populated class (or one
    this client doesn't have at all) still gets at least one prototype,
    never more clusters than it has samples to support.

    Returns `(prototypes [P, D], proto_labels [P])` -- P is the client's
    total prototype count across all classes it has samples for."""
    protos, proto_labels = [], []
    for s in range(num_classes):
        mask = labels == s
        n_s = int(mask.sum())
        if n_s == 0:
            continue
        k = max(1, min(k_per_class, n_s // min_per_cluster if n_s >= min_per_cluster else 1))
        k = min(k, n_s)
        if k == 1:
            protos.append(embeddings[mask].mean(axis=0, keepdims=True))
        else:
            km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(embeddings[mask])
            protos.append(km.cluster_centers_)
        proto_labels.append(np.full(protos[-1].shape[0], s, dtype=np.int64))
    return np.concatenate(protos, axis=0).astype(np.float32), np.concatenate(proto_labels, axis=0)


def refine_prototypes_margin(prototypes, proto_labels, embeddings, labels, epochs=50, lr=1e-3, tau=0.1):
    """Eq. 5: margin-based hinge-loss refinement of the prototype tensor
    ONLY (embeddings/encoder are fixed inputs, not optimized). For each
    training sample, `s_y` = max cosine similarity to its OWN class's
    prototypes, `s_notY` = max cosine similarity to any OTHER class's
    prototypes (both restricted to THIS client's own prototype set, per
    Algorithm 1 line 3 -- refinement happens locally, before upload);
    loss = `max(0, tau - (s_y - s_notY))`, summed over samples with both a
    same-class and a different-class prototype available (a client with
    only one class present has nothing to push away from -- those samples
    contribute zero loss, matching the paper's Eq. 5 definition exactly,
    not a special-cased skip).

    Returns the refined `prototypes` array (same shape as input)."""
    proto_t = torch.tensor(prototypes, dtype=torch.float32, requires_grad=True)
    labels_t = torch.tensor(proto_labels, dtype=torch.int64)
    z = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.int64)

    optimizer = torch.optim.Adam([proto_t], lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        proto_n = F.normalize(proto_t, dim=1)
        z_n = F.normalize(z, dim=1)
        sim = z_n @ proto_n.t()  # [N, P]
        same_class = labels_t.unsqueeze(0) == y.unsqueeze(1)  # [N, P]
        has_same = same_class.any(dim=1)
        has_diff = (~same_class).any(dim=1)
        active = has_same & has_diff
        if not active.any():
            break
        s_y = sim.masked_fill(~same_class, float("-inf")).max(dim=1).values
        s_noty = sim.masked_fill(same_class, float("-inf")).max(dim=1).values
        loss = F.relu(tau - (s_y - s_noty))[active].sum()
        loss.backward()
        optimizer.step()
    return proto_t.detach().numpy()


def retrieve_and_vote(z, bank_protos, bank_labels, num_classes, top_k=3):
    """Eq. 8-9: retrieve the `top_k` most cosine-similar prototypes from
    the GLOBAL bank for each embedding in `z` [N, D], each casting a
    similarity-weighted vote for its class. Returns `(y_p [N, num_classes],
    alpha [N])` -- `alpha` is the top-1 retrieved similarity per sample,
    the confidence score `confidence_ensemble` uses for Eq. 10."""
    z_n = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
    bank_n = bank_protos / np.maximum(np.linalg.norm(bank_protos, axis=1, keepdims=True), 1e-8)
    sim = z_n @ bank_n.T  # [N, P]
    k = min(top_k, sim.shape[1])
    top_idx = np.argsort(-sim, axis=1)[:, :k]  # [N, k]
    top_sim = np.take_along_axis(sim, top_idx, axis=1)  # [N, k]
    top_labels = bank_labels[top_idx]  # [N, k]

    y_p = np.zeros((z.shape[0], num_classes), dtype=np.float32)
    for j in range(k):
        np.add.at(y_p, (np.arange(z.shape[0]), top_labels[:, j]), top_sim[:, j])
    alpha = top_sim[:, 0]
    return y_p, alpha


def confidence_ensemble(y_p, y_wi, alpha):
    """Eq. 10: `y_en = alpha * y_p + (1 - alpha) * y_wi`. `y_wi` should
    already be a probability distribution (softmax of the trained
    classifier's logits); `y_p` need not be normalized (Eq. 9 doesn't
    normalize it either -- it's a raw similarity-weighted vote sum)."""
    return alpha[:, None] * y_p + (1 - alpha[:, None]) * y_wi
