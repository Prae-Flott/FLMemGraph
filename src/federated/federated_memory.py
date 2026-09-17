"""
Server-side discrete-memory alignment across clients (robots), adapted
from FedPM's cross-domain memory alignment
(`mem_phys_prompt_zh.md` Sec 3/4; arXiv:2604.04475). `align_and_split`
(the codebook alignment below) is the ORIGINAL design's only exchange
between clients, matching the source doc's "只传字典" communication-
efficiency commitment -- structure heads (`edge_head`/`typed_head`) still
stay fully local per client. `SharedEncoder`/decoder are the one
exception: `fedavg_state_dict` (below) now also FedAvg's those every
round, and is the shipped default (`SYNC_ENCODER_DECODER = True`) for
robo3er and Paderborn -- see that function's docstring. Only Sielaff
still matches the original codebook-only design end to end.

Algorithm, per round:
  1. Every client uploads its current codebook P_n [M, D] and per-
     prototype usage counts (Freq) -- NOT gradients, NOT raw data.
  2. Build a cosine-similarity graph over the pooled C*M prototypes,
     edges only between DIFFERENT clients' prototypes above threshold
     delta (same-client pairs are never linked -- aligning a client's
     dictionary against itself is meaningless).
  3. Connected components (BFS) with >1 member are candidate "shared
     semantic clusters" -- a component of size 1 (no other client agreed)
     cannot be shared by definition. `linkage="single"` (default) accepts
     any connected component (transitive chaining allowed); `linkage=
     "complete"` additionally requires every pair within the component to
     be above `delta`, rejecting chained-but-not-mutually-similar groups.
  4. Top-K clusters by member count (K = min(#multi-member clusters,
     floor(gamma*M))) become P_S: one representative per cluster,
     mean-pooled over its member prototype vectors.
  5. Each client's remaining (unclustered) prototypes are ranked by
     utility-diversity V(e) = Freq(e)/max_client_Freq -
     max_similarity_to_other_clients'_unclustered_prototypes -- favors
     prototypes this client actually uses a lot AND that no other client
     already has something similar for (domain-specific knowledge, not
     redundant with what's about to be shared).
  6. Broadcast P_G,n = [P_S ; P_p,n] back to client n -- shared slots
     first (same across all clients), personalized slots fill the rest.

This is O((C*M)^2) in similarity computation (vectorized via torch, not
a python double loop) -- fine at FedPM's own scale (M=256, our C=5
clients -> 1280x1280).
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F


def confidence_weights_from_losses(loss_per_client: dict, temperature: float = 1.0):
    """Cross-component, per-client confidence weighting for combining V3's
    ablation signals (A/prototype, C/edge, E/typed-edge), replacing a hard
    `max()` across them with a weighted sum -- see
    `memory/joint-prototype-federated-results.md`'s "max-aggregation noise
    floor" follow-up: folding in a poorly-learned component via max() can
    pull the combined score BELOW dropping that component entirely.

    `loss_per_client`: {component_name: [loss_client_0, loss_client_1, ...]},
    each component's OWN raw training-objective quantity evaluated on that
    client's calib split (e.g. `d_G.mean()` for the memory/prototype
    component, `s_edge.mean()` for edge attention, `r.mean()` for typed
    edges -- exactly what each component's training loss term already
    minimizes, just measured on held-out calib data instead of fit data).
    All clients must report every component (same keys, same length lists).

    Each component's loss is z-scored ACROSS CLIENTS (not across
    components) first, since raw loss magnitudes are not comparable
    between components (different natural scales/dimensionality) but ARE
    comparable for the same component across clients -- "did this client
    learn edge attention worse than its peers" is a well-posed question,
    "is this client's edge loss bigger than its own typed-edge loss" isn't,
    without further normalization this function doesn't attempt.

    Returns list of C dicts {component_name: weight}, weights summing to 1
    per client -- lower relative loss (vs. peers) means higher weight."""
    components = list(loss_per_client.keys())
    arr = np.array([loss_per_client[k] for k in components], dtype=np.float64)  # [K, C]
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True) + 1e-8
    z = (arr - mean) / std  # [K, C] -- higher z = worse (higher loss) than peers for that component
    logits = -temperature * z
    logits = logits - logits.max(axis=0, keepdims=True)  # numerical stability, doesn't change softmax
    exp = np.exp(logits)
    weights = exp / exp.sum(axis=0, keepdims=True)  # softmax over components, per client
    return [{components[k]: float(weights[k, c]) for k in range(len(components))} for c in range(arr.shape[1])]


def fedavg_state_dict(state_dicts, client_weights, prefixes):
    """Sample-size-weighted FedAvg over a SUBSET of each client's state_dict
    (only keys starting with one of `prefixes`, e.g. `("encoder.",
    "decoder.")`) -- a departure from this module's original "only exchange
    the memory dictionary" design (see module docstring), first added to
    test whether client drift in the LOCALLY-trained `SharedEncoder` (never
    federated before this existed, only `JointPrototypeMemory` was) was a
    real cause of the federated-vs-centralized gap documented in
    `memory/joint-prototype-federated-results.md`. It confirmed the
    diagnosis on robo3er and has since been promoted to the default
    (`SYNC_ENCODER_DECODER = True`) in `run_robo3er_bck_federated.py` and
    `run_paderborn_bck_federated.py` -- it is no longer an experimental
    opt-in for those two. `run_sielaff_bck_federated.py` still passes
    `sync_encoder_decoder=False` and remains on the original codebook-only
    design.

    `state_dicts`: list of C full `model.state_dict()` outputs.
    `client_weights`: list of C non-negative floats (need not already sum
    to 1 -- typically each client's local fit-sample count, i.e. standard
    FedAvg weighting so a 3000-window client isn't diluted to the same
    influence as a 100-window client).

    Returns ONE averaged dict (same averaged weights for every client, by
    construction) containing only the matched keys -- load into each
    client's model with `model.load_state_dict(avg, strict=False)` to
    synchronize just that submodule while leaving everything else
    (memory codebook, edge/typed heads) untouched."""
    total = float(sum(client_weights))
    norm_weights = [w / total for w in client_weights]
    keys = [k for k in state_dicts[0] if k.startswith(tuple(prefixes))]
    avg = {}
    for k in keys:
        acc = norm_weights[0] * state_dicts[0][k].float()
        for w, sd in zip(norm_weights[1:], state_dicts[1:]):
            acc = acc + w * sd[k].float()
        avg[k] = acc
    return avg


def fedavg_state_dict_full(state_dicts, client_weights):
    """Standard FedAvg (McMahan et al., AISTATS 2017): sample-size-weighted
    average over EVERY key of the state_dict (encoder + decoder + memory
    codebook + structure/typed/cov heads + forecast_head), unlike
    `fedavg_state_dict` above which only averages a caller-chosen prefix
    subset. Ported from `archive/benchmark/baselines/fedavg_baseline.py`'s
    `federated_average` -- used as the "FedAvg" baseline (single global
    model, no personalization at all) and as the within-cluster aggregation
    step of the "IFCAAE" baseline below.

    `state_dicts`: list of C full `model.state_dict()` outputs, all
    identical keys/shapes (same architecture across clients -- no
    personalization layers, unlike this project's own memory-only design).
    `client_weights`: list of C non-negative floats (need not sum to 1)."""
    total = float(sum(client_weights))
    norm_weights = [w / total for w in client_weights]
    avg = OrderedDict()
    for key in state_dicts[0].keys():
        stacked = torch.stack(
            [sd[key].float() * w for sd, w in zip(state_dicts, norm_weights)], dim=0
        )
        avg[key] = stacked.sum(dim=0).to(state_dicts[0][key].dtype)
    return avg


def align_and_split_fedcc(client_codebooks, client_usage_counts, gamma: float = 0.5, delta: float = 0.8,
                           linkage: str = "single"):
    """Fed-ExDNN baseline's server-side step: local "exemplar" learning is
    already identical to this project's own VQ-VAE codebook (no public
    Fed-ExDNN code exists -- see `docs/baseline_selection_for_benchmark.md`
    Sec 3.1 -- so the local side is reused unchanged), but the cross-client
    alignment rule is FedCC-style CONSTRAINED CLUSTERING instead of
    `align_and_split`'s cosine-threshold + BFS single-linkage graph. This is
    a documented approximation of "FedCC" (no public reference
    implementation to match against): k-means over the pooled C*M
    prototypes with k = floor(gamma*M) clusters, and a hard per-cluster size
    cap (the "constrained" part -- prevents one big degenerate cluster from
    swallowing every client's prototypes, the discrete-clustering analogue
    of `align_and_split`'s same-client-pair exclusion). A cluster is kept as
    "shared" only if, after the cap, it still contains prototypes from >= 2
    distinct clients -- mirroring `align_and_split`'s multi-client
    requirement exactly. Same [C*M -> P_S + per-client P_p,n] output
    contract as `align_and_split`, so it's a drop-in replacement wherever
    that function is called.

    `linkage` is accepted but unused (kept for call-signature parity with
    `align_and_split` -- FedCC's clustering has no single/complete-linkage
    distinction)."""
    del linkage  # unused, kept for signature parity with align_and_split
    C = len(client_codebooks)
    was_2d = client_codebooks[0].dim() == 2
    codebooks = [cb.unsqueeze(1) if was_2d else cb for cb in client_codebooks]  # [M, N, D]
    M, N, D = codebooks[0].shape
    device = codebooks[0].device

    all_protos = torch.cat(codebooks, dim=0)  # [C*M, N, D]
    client_of = torch.arange(C, device=device).repeat_interleave(M)  # [C*M]
    n = all_protos.shape[0]
    flat = all_protos.reshape(n, N * D)

    k = max(1, min(int(gamma * M), n))
    cap = max(1, (n + k - 1) // k)  # per-cluster size cap ("constrained" clustering)

    # Simple constrained k-means: standard Lloyd iteration, but at each
    # assignment step, prototypes are greedily assigned to their nearest
    # centroid IN DISTANCE ORDER, skipping any centroid already at `cap`
    # members -- keeps clusters balanced instead of letting a few centroids
    # absorb everything (the failure mode this baseline is meant to guard
    # against, matching FedCC's stated "constrained" clustering intent).
    gen = torch.Generator(device="cpu").manual_seed(0)
    init_idx = torch.randperm(n, generator=gen)[:k]
    centroids = flat[init_idx].clone()
    assign = torch.zeros(n, dtype=torch.long, device=device)
    for _ in range(10):
        dist = torch.cdist(flat, centroids)  # [n, k]
        order = torch.argsort(dist.reshape(-1))
        counts = [0] * k
        assign = torch.full((n,), -1, dtype=torch.long, device=device)
        remaining = n
        for flat_idx in order.tolist():
            if remaining == 0:
                break
            i, c = flat_idx // k, flat_idx % k
            if assign[i] != -1 or counts[c] >= cap:
                continue
            assign[i] = c
            counts[c] += 1
            remaining -= 1
        for c in range(k):
            members = flat[assign == c]
            if len(members) > 0:
                centroids[c] = members.mean(dim=0)

    shared_clusters, cluster_node_agreement = [], []
    for c in range(k):
        members = torch.nonzero(assign == c, as_tuple=True)[0]
        if len(members) < 2:
            continue
        n_distinct_clients = len(set(client_of[members].tolist()))
        if n_distinct_clients < 2:
            continue  # FedCC's cluster only agrees within one client -- not "shared"
        shared_clusters.append(members.tolist())

    shared_clusters.sort(key=len, reverse=True)
    K_shared = min(len(shared_clusters), int(gamma * M))
    shared_clusters = shared_clusters[:K_shared]

    if shared_clusters:
        P_S = torch.stack([all_protos[idx].mean(dim=0) for idx in shared_clusters])  # [K, N, D]
        normed = F.normalize(all_protos, dim=2)
        sim_per_node = torch.einsum("and,bnd->abn", normed, normed)
        for idx in shared_clusters:
            idx_t = torch.tensor(idx, device=device)
            pair_sim = sim_per_node[idx_t][:, idx_t]
            off_diag = ~torch.eye(len(idx), dtype=torch.bool, device=device)
            cluster_node_agreement.append(pair_sim[off_diag].mean(dim=0).tolist() if off_diag.any() else [0.0] * N)
    else:
        P_S = torch.empty(0, N, D, device=device)

    clustered_global = set(i for cl in shared_clusters for i in cl)
    P_G = []
    n_personal_target = M - P_S.shape[0]
    for c in range(C):
        offset = c * M
        unclustered_local = [i for i in range(M) if (offset + i) not in clustered_global]
        freq = client_usage_counts[c].to(device)
        max_freq = freq.max().clamp(min=1.0)
        scored = sorted(unclustered_local, key=lambda i: -float(freq[i] / max_freq))
        chosen = scored[:n_personal_target]
        if len(chosen) < n_personal_target:
            remaining = n_personal_target - len(chosen)
            fallback = [i for i in range(M) if i not in chosen]
            fallback.sort(key=lambda i: -float(freq[i]))
            chosen += fallback[:remaining]
        P_p_n = all_protos[[offset + i for i in chosen]]
        P_G.append(torch.cat([P_S, P_p_n], dim=0))

    if was_2d:
        P_G = [p.squeeze(1) for p in P_G]
        cluster_node_agreement = [row[0] for row in cluster_node_agreement]

    diagnostics = {
        "num_components": k,
        "num_multi_client_clusters": len(shared_clusters),
        "num_shared_prototypes": int(P_S.shape[0]),
        "num_personalized_prototypes_per_client": n_personal_target,
        "per_cluster_per_node_agreement": cluster_node_agreement,
    }
    return P_G, diagnostics


def _pooled_precision(all_protos: torch.Tensor, reg: float = 1e-2) -> torch.Tensor:
    """Global precision matrix (inverse covariance) for Mahalanobis alignment,
    estimated by pooling EVERY (prototype, node) row of `all_protos` [n, N, D]
    into one [n*N, D] sample matrix -- consistent with `SharedEncoder` being
    one shared per-node projection (channel-independent), so every node's
    D-dim vector is a draw from the same distribution rather than needing its
    own per-node covariance. Ridge-regularized for invertibility when the
    pooled sample count is small relative to D."""
    D = all_protos.shape[-1]
    flat = all_protos.reshape(-1, D)
    centered = flat - flat.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / max(flat.shape[0] - 1, 1)
    cov = cov + reg * torch.eye(D, device=all_protos.device, dtype=all_protos.dtype)
    return torch.linalg.inv(cov)


def _poincare_exp_map(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Exponential map at the Poincare ball's origin (curvature c=1): lifts a
    flat embedding into hyperbolic space via tanh(||x||) * x/||x||, so
    large-norm vectors saturate toward the ball's boundary instead of leaving
    it -- the standard way to compare an existing Euclidean-trained embedding
    under a hyperbolic metric without retraining the encoder itself."""
    norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
    return torch.tanh(norm) * x / norm * (1 - eps)


def _pairwise_agreement(all_protos: torch.Tensor, metric: str) -> torch.Tensor:
    """all_protos: [n, N, D]. Returns sim_per_node [n, n, N] where HIGHER
    always means "more similar" regardless of metric -- cosine is a genuine
    similarity, the other three are negated distances -- so every caller can
    threshold with the same `sim > delta` convention no matter which metric
    is selected.

    - "cosine": per-node cosine similarity (original behavior).
    - "euclidean": negated per-node squared L2 distance.
    - "hyperbolic": negated Poincare-ball geodesic distance (`_poincare_exp_map`
      then the standard ball distance formula), curvature c=1.
    - "mahalanobis": negated Mahalanobis distance under a global pooled
      precision matrix (`_pooled_precision`) -- NOT a per-prototype
      covariance (that's `DeviationCovarianceHead`'s job on `d_node`, a
      different, downstream quantity)."""
    if metric == "cosine":
        normed = F.normalize(all_protos, dim=2)
        return torch.einsum("and,bnd->abn", normed, normed)
    if metric == "euclidean":
        diff = all_protos.unsqueeze(1) - all_protos.unsqueeze(0)  # [n, n, N, D]
        return -diff.pow(2).sum(-1)
    if metric == "hyperbolic":
        y = _poincare_exp_map(all_protos)  # [n, N, D]
        norm_sq = y.pow(2).sum(-1)  # [n, N]
        diff_sq = (y.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(-1)  # [n, n, N]
        denom = ((1 - norm_sq.unsqueeze(1)) * (1 - norm_sq.unsqueeze(0))).clamp(min=1e-6)
        arg = (1 + 2 * diff_sq / denom).clamp(min=1 + 1e-6)
        return -torch.acosh(arg)
    if metric == "mahalanobis":
        cov_inv = _pooled_precision(all_protos)
        diff = all_protos.unsqueeze(1) - all_protos.unsqueeze(0)  # [n, n, N, D]
        return -torch.einsum("abnd,de,abne->abn", diff, cov_inv, diff)
    raise ValueError(f"unknown metric {metric!r}, expected 'cosine', 'euclidean', "
                      "'hyperbolic', or 'mahalanobis'")


def _bfs_components(adj, n):
    visited = [False] * n
    components = []
    for s in range(n):
        if visited[s]:
            continue
        comp = []
        stack = [s]
        visited[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        components.append(comp)
    return components


def _complete_linkage_filter(components, sim, delta):
    """Keep only components where EVERY pair agrees (sim > delta), not just
    connected transitively -- avoids single-linkage's chaining artifact
    (A-B and B-C above delta forcing A/B/C into one cluster even if A-C is
    below delta). Same-client pairs are already forced to -1 in `sim`
    (see `align_and_split`), so this also naturally rejects any component
    that (via chaining) pulled in >1 prototype from the same client."""
    filtered = []
    for comp in components:
        if len(comp) <= 1:
            continue
        idx = torch.tensor(comp, device=sim.device)
        sub = sim[idx][:, idx]
        off_diag = ~torch.eye(len(comp), dtype=torch.bool, device=sim.device)
        if bool((sub[off_diag] > delta).all()):
            filtered.append(comp)
    return filtered


def align_and_split(client_codebooks, client_usage_counts, gamma: float = 0.5, delta: float = 0.8,
                     linkage: str = "single", metric: str = "cosine", edge_quantile: float = None):
    """client_codebooks: list of C tensors, each either [M, D] (FLGDNMemory's
    per-feature-independent codebook) or [M, N, D] (JointPrototypeMemory's
    joint multi-node snapshot -- see `joint_prototype_model.JointPrototypeMemory`).
    client_usage_counts: list of C tensors, each [M] (Freq, non-negative).
    Returns (list of C tensors, same per-client shape as the input -- P_G,n per
    client, diagnostics dict).

    For [M, N, D] input, agreement is computed PER NODE (not by flattening
    the [N, D] snapshot into one vector) and only then averaged across nodes
    to decide edges/clusters -- this keeps a node axis in `sim_per_node`
    throughout so a later diagnosis pass can ask "which node(s) actually
    agreed" for any given cross-client match, instead of that information
    being collapsed away by flattening before any similarity is computed.
    [M, D] input is treated as the N=1 special case of the same code path
    (unsqueezed to [M, 1, D]), so `sim_per_node.mean(dim=-1) == sim` exactly
    reproduces the original 2D behavior -- no change for FLGDNMemory callers.

    `metric`: which pairwise agreement `_pairwise_agreement` computes --
    "cosine" (default, original behavior), "euclidean", "hyperbolic"
    (Poincare-ball geodesic), or "mahalanobis" (global pooled precision).
    All four share the same `sim > delta` edge rule since `_pairwise_agreement`
    always returns higher-is-closer values.

    `edge_quantile`: if given (0 < q < 1), `delta` is IGNORED and instead
    recomputed per call as the (1-q)-quantile of the actual cross-client
    `sim` distribution -- e.g. q=0.05 links the closest 5% of cross-client
    prototype pairs. Needed for a fair cross-metric comparison: a fixed
    `delta=0.8` means something totally different for cosine (bounded
    [-1, 1]) than for negated Euclidean/Mahalanobis distance (unbounded,
    dataset- and scale-dependent), so comparing metrics at the same raw
    `delta` would really be comparing them at wildly different edge
    densities. Quantile calibration instead holds the EDGE DENSITY constant
    across metrics, which is the fair, metric-agnostic knob."""
    C = len(client_codebooks)
    was_2d = client_codebooks[0].dim() == 2
    codebooks = [cb.unsqueeze(1) if was_2d else cb for cb in client_codebooks]  # [M, N, D]
    M, N, D = codebooks[0].shape
    device = codebooks[0].device

    all_protos = torch.cat(codebooks, dim=0)  # [C*M, N, D]
    client_of = torch.arange(C, device=device).repeat_interleave(M)  # [C*M]
    n = all_protos.shape[0]

    sim_per_node = _pairwise_agreement(all_protos, metric)  # [C*M, C*M, N]
    sim = sim_per_node.mean(dim=2)  # [C*M, C*M] -- aggregate used for edge/cluster decisions
    same_client = client_of.unsqueeze(0) == client_of.unsqueeze(1)
    if edge_quantile is not None:
        cross_client_vals = sim[~same_client]
        delta = torch.quantile(cross_client_vals, 1.0 - edge_quantile).item()
    sim = sim.masked_fill(same_client, float("-inf"))  # never link same-client prototypes
    edge_i, edge_j = torch.nonzero(sim > delta, as_tuple=True)

    adj = [[] for _ in range(n)]
    for a, b in zip(edge_i.tolist(), edge_j.tolist()):
        adj[a].append(b)

    components = _bfs_components(adj, n)
    if linkage == "complete":
        multi = _complete_linkage_filter(components, sim, delta)
    elif linkage == "single":
        multi = [c for c in components if len(c) > 1]
    else:
        raise ValueError(f"unknown linkage {linkage!r}, expected 'single' or 'complete'")
    multi.sort(key=len, reverse=True)
    K = min(len(multi), int(gamma * M))
    shared_clusters = multi[:K]

    if shared_clusters:
        P_S = torch.stack([all_protos[idx].mean(dim=0) for idx in shared_clusters])  # [K, N, D]
        # per-node agreement within each shared cluster (mean pairwise sim_per_node
        # over the cluster's member pairs) -- kept per-node rather than collapsed to
        # one number, so a later pass can see e.g. "this cluster agrees on node 2
        # (vibration) but not node 4 (torque)" instead of just an overall score.
        cluster_node_agreement = []
        for idx in shared_clusters:
            idx_t = torch.tensor(idx, device=device)
            pair_sim = sim_per_node[idx_t][:, idx_t]  # [len(idx), len(idx), N]
            off_diag = ~torch.eye(len(idx), dtype=torch.bool, device=device)
            cluster_node_agreement.append(pair_sim[off_diag].mean(dim=0).tolist())  # [N]
    else:
        P_S = torch.empty(0, N, D, device=device)
        cluster_node_agreement = []

    clustered_global = set(i for cl in shared_clusters for i in cl)

    P_G = []
    n_personal_target = M - P_S.shape[0]
    for c in range(C):
        offset = c * M
        unclustered_local = [i for i in range(M) if (offset + i) not in clustered_global]
        freq = client_usage_counts[c].to(device)
        max_freq = freq.max().clamp(min=1.0)

        other_unclustered_global = [
            c2 * M + j for c2 in range(C) if c2 != c
            for j in range(M) if (c2 * M + j) not in clustered_global
        ]

        scored = []
        for i in unclustered_local:
            g_idx = offset + i
            f_score = (freq[i] / max_freq).item()
            if other_unclustered_global:
                # reuse the already-computed, node-aggregated `sim` matrix rather than
                # re-deriving it from `normed` (which is now [C*M, N, D], not [C*M, D])
                max_sim = float(sim[g_idx, other_unclustered_global].max())
            else:
                max_sim = 0.0
            scored.append((f_score - max_sim, i))
        scored.sort(key=lambda t: -t[0])

        chosen = [i for _, i in scored[:n_personal_target]]
        if len(chosen) < n_personal_target:  # not enough unclustered protos, pad by raw frequency
            remaining = n_personal_target - len(chosen)
            fallback = [i for i in range(M) if i not in chosen]
            fallback.sort(key=lambda i: -float(freq[i]))
            chosen += fallback[:remaining]

        P_p_n = all_protos[[offset + i for i in chosen]]
        P_G.append(torch.cat([P_S, P_p_n], dim=0))  # [M, N, D]

    if was_2d:
        P_G = [p.squeeze(1) for p in P_G]
        cluster_node_agreement = [row[0] for row in cluster_node_agreement]  # N=1 -> scalar

    diagnostics = {
        "metric": metric,
        "delta_used": float(delta),
        "num_components": len(components),
        "num_multi_client_clusters": len(multi),
        "num_shared_prototypes": int(P_S.shape[0]),
        "num_personalized_prototypes_per_client": n_personal_target,
        # per-node mean agreement (cosine sim) within each shared cluster, in cluster
        # order (largest cluster first) -- a [N]-length list per cluster for the [M, N, D]
        # case, a scalar per cluster for the [M, D] (N=1) case. Lets a later diagnosis
        # pass identify which physical signal(s) actually drove any given cross-client
        # match, instead of only a single flattened similarity number.
        "per_cluster_per_node_agreement": cluster_node_agreement,
    }
    return P_G, diagnostics
