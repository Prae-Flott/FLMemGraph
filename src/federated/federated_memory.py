"""
Server-side discrete-memory alignment across clients (robots), adapted
from FedPM's cross-domain memory alignment
(`mem_phys_prompt_zh.md` Sec 3/4; arXiv:2604.04475). This is the ONLY
thing exchanged between clients in this design -- encoders and structure
heads stay fully local per client (see `fl_model.py`'s module docstring)
-- matching the source doc's "只传字典" communication-efficiency
commitment.

Algorithm, per round:
  1. Every client uploads its current codebook P_n [M, D] and per-
     prototype usage counts (Freq) -- NOT gradients, NOT raw data.
  2. Build a cosine-similarity graph over the pooled C*M prototypes,
     edges only between DIFFERENT clients' prototypes above threshold
     delta (same-client pairs are never linked -- aligning a client's
     dictionary against itself is meaningless).
  3. Connected components (BFS) with >1 member are candidate "shared
     semantic clusters" -- a component of size 1 (no other client agreed)
     cannot be shared by definition.
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
    "decoder.")`) -- an explicit, opt-in departure from this module's
    original "only exchange the memory dictionary" design (see module
    docstring), added to test whether client drift in the LOCALLY-trained
    `SharedEncoder` (never previously federated, only `JointPrototypeMemory`
    was) is a real cause of the federated-vs-centralized gap documented in
    `memory/joint-prototype-federated-results.md`.

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


def compute_prototype_dev_stats(d_node: np.ndarray, idx: np.ndarray, num_prototypes: int):
    """Sufficient statistics (n, mean, var) of `d_node` per matched prototype,
    computed in ONE one-shot local pass over a client's own data (using
    whatever codebook that pass's `idx` was matched against) -- NOT a decayed
    EMA. Used by both the `federated_ema` calib mode (recoverable via `git
    show 916ade0`, reverted from the tree -- see `memory/calib-in-prototype-ab.md`)
    and the `shrinkage` calib mode below: a plain per-prototype sample-count/
    mean/variance triple is used specifically because it has a clean,
    associative, order-independent cross-client merge rule (the
    parallel-variance / Chan-et-al. formula, `_merge_dev_stats_pair` below);
    an EMA has no equivalent closed-form merge.

    `d_node`: [W, num_nodes] numpy. `idx`: [W] numpy matched-prototype
    indices, same forward pass. Returns `(n [M] int64, mean [M, num_nodes]
    float64, var [M, num_nodes] float64)` -- unpopulated prototypes (no
    window routed there) get `n=0`/`mean=0`/`var=0`; callers must gate on
    `n == 0` before trusting `mean`/`var` for that slot."""
    num_nodes = d_node.shape[1]
    n = np.zeros(num_prototypes, dtype=np.int64)
    mean = np.zeros((num_prototypes, num_nodes), dtype=np.float64)
    var = np.zeros((num_prototypes, num_nodes), dtype=np.float64)
    for m in range(num_prototypes):
        mask = idx == m
        n_m = int(mask.sum())
        n[m] = n_m
        if n_m > 0:
            d_m = d_node[mask].astype(np.float64)
            mean[m] = d_m.mean(axis=0)
            var[m] = d_m.var(axis=0) if n_m > 1 else np.zeros(num_nodes)
    return n, mean, var


def _merge_dev_stats_pair(n_a, mean_a, var_a, n_b, mean_b, var_b):
    """Parallel-variance (Chan, Golub, LeVeque 1979) pairwise merge of two
    (n, mean, var) sufficient-statistics triples into one, exact (not an
    approximation) for combining two disjoint samples' mean/variance from
    only their own summary statistics. Associative/commutative, so folding
    more than two members into one pool can be done in any order."""
    n = n_a + n_b
    if n <= 0:
        return 0.0, mean_a, var_a  # both empty; values are unused (n==0 downstream)
    mean = (n_a * mean_a + n_b * mean_b) / n
    m2 = n_a * var_a + n_b * var_b + (mean_a - mean_b) ** 2 * (n_a * n_b) / n
    return n, mean, m2 / n


def compute_shrinkage_stats(client_dev_stats, num_shared: int, alpha: float):
    """Empirical-Bayes per-client, per-prototype shrinkage, "shrinkage" calib
    mode of `memory/calib-in-prototype-ab.md` -- the design that replaces
    both plain `ema` (pure local stats, small clients never clear warm-up)
    and the reverted `federated_ema` (one fused stat shared IDENTICALLY by
    every client in a cluster, which let large clients overwrite small ones'
    real statistics -- see that doc's "Why federated_ema failed" section).

    `client_dev_stats`: list of C `(n, mean, var)` triples from
    `compute_prototype_dev_stats`, one per client, each computed via a
    one-shot local pass over that client's OWN fit split, using THIS ROUND'S
    FINAL (already server-aligned) codebook -- i.e. called AFTER
    `align_and_split` + `load_memory()`, not before (unlike `federated_ema`'s
    placement). This matters: because every client's `idx` this round is
    drawn under the identical aligned codebook, prototype index `k` means
    the SAME physical prototype for every client when `k < num_shared` (the
    broadcast-shared slots, `align_and_split`'s `P_S` -- identical vector
    content for every client by construction) but a DIFFERENT,
    client-specific prototype when `k >= num_shared` (`P_p_n`, personalized
    slots -- each client's own content there).

    Fleet-wide pooling (across every client with `n_{c,k} > 0` this round --
    deliberately NOT restricted to `align_and_split`'s original clustering
    decision, a broadening explicitly called for by the design doc, since the
    shrinkage target is a fleet-wide prior, not cluster-specific
    reconciliation) is therefore applied ONLY to the shared slots
    `k < num_shared`: those are the only indices where "the same prototype"
    is a coherent cross-client concept this round. Personalized slots
    (`k >= num_shared`) pass through with `lambda=1` (no cross-client
    borrowing) -- pooling raw index `k` there would blend together unrelated
    per-client prototype content, since each client's personalized-slot
    content differs by construction (this is a deliberate, documented
    DEVIATION from a literal "pool every client with data for that index"
    rule, necessary because that rule is only semantically valid for the
    shared slots).

    For each shared slot `k`, blends client `c`'s own local `(mean, var)`
    with the fleet-pooled `(mean, var)` via `lambda_{c,k} = n_{c,k} /
    (n_{c,k} + alpha)` (client's own local `n` this round; `alpha` is the
    shrinkage-strength hyperparameter, swept by callers, not tuned here) --
    `lambda` never reaches exactly 0 unless `n=0` (full fallback to the
    fleet-pooled prior, expected/fine) and never reaches exactly 1 unless
    `alpha=0`.

    Returns a list of C `(mean_star [M, num_nodes], var_star [M, num_nodes],
    valid [M] bool)` triples ready for
    `JointPrototypeMemory.load_shrinkage_stats()`. `valid[k]` is True for
    shared slots whenever the FLEET pool has any data (`global_n[k] > 0`,
    regardless of this client's own `n_{c,k}`) and for personalized slots
    whenever this client's OWN `n_{c,k} > 0` (matching plain `ema`'s
    validity convention there, since no fleet borrowing applies)."""
    C = len(client_dev_stats)
    M, num_nodes = client_dev_stats[0][1].shape
    n_list = [ds[0].astype(np.float64) for ds in client_dev_stats]
    mean_list = [ds[1].astype(np.float64) for ds in client_dev_stats]
    var_list = [ds[2].astype(np.float64) for ds in client_dev_stats]

    global_n = np.zeros(num_shared, dtype=np.float64)
    global_mean = np.zeros((num_shared, num_nodes), dtype=np.float64)
    global_var = np.zeros((num_shared, num_nodes), dtype=np.float64)
    for k in range(num_shared):
        n_acc, mean_acc, var_acc = 0.0, None, None
        for c in range(C):
            n_c = n_list[c][k]
            if n_c <= 0:
                continue
            if mean_acc is None:
                n_acc, mean_acc, var_acc = n_c, mean_list[c][k], var_list[c][k]
            else:
                n_acc, mean_acc, var_acc = _merge_dev_stats_pair(
                    n_acc, mean_acc, var_acc, n_c, mean_list[c][k], var_list[c][k])
        global_n[k] = n_acc
        if mean_acc is not None:
            global_mean[k] = mean_acc
            global_var[k] = var_acc
    shared_valid = global_n > 0  # [num_shared]

    out = []
    for c in range(C):
        mean_star = mean_list[c].copy()
        var_star = var_list[c].copy()
        valid = n_list[c] > 0  # personalized-slot validity: this client's own coverage only
        if num_shared > 0:
            lam = (n_list[c][:num_shared] / (n_list[c][:num_shared] + alpha))[:, None]
            mean_star[:num_shared] = lam * mean_list[c][:num_shared] + (1 - lam) * global_mean
            var_star[:num_shared] = lam * var_list[c][:num_shared] + (1 - lam) * global_var
            valid[:num_shared] = shared_valid  # even this client's own n=0 is valid (lambda->0)
        out.append((mean_star, var_star, valid))
    return out


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


def align_and_split(client_codebooks, client_usage_counts, gamma: float = 0.5, delta: float = 0.8):
    """client_codebooks: list of C tensors, each either [M, D] (FLGDNMemory's
    per-feature-independent codebook) or [M, N, D] (JointPrototypeMemory's
    joint multi-node snapshot -- see `joint_prototype_model.JointPrototypeMemory`).
    client_usage_counts: list of C tensors, each [M] (Freq, non-negative).
    Returns (list of C tensors, same per-client shape as the input -- P_G,n per
    client, diagnostics dict).

    For [M, N, D] input, similarity is computed PER NODE (not by flattening
    the [N, D] snapshot into one vector) and only then averaged across nodes
    to decide edges/clusters -- this keeps a node axis in `sim_per_node`
    throughout so a later diagnosis pass can ask "which node(s) actually
    agreed" for any given cross-client match, instead of that information
    being collapsed away by flattening before any similarity is computed.
    [M, D] input is treated as the N=1 special case of the same code path
    (unsqueezed to [M, 1, D]), so `sim_per_node.mean(dim=-1) == sim` exactly
    reproduces the original 2D behavior -- no change for FLGDNMemory callers."""
    C = len(client_codebooks)
    was_2d = client_codebooks[0].dim() == 2
    codebooks = [cb.unsqueeze(1) if was_2d else cb for cb in client_codebooks]  # [M, N, D]
    M, N, D = codebooks[0].shape
    device = codebooks[0].device

    all_protos = torch.cat(codebooks, dim=0)  # [C*M, N, D]
    client_of = torch.arange(C, device=device).repeat_interleave(M)  # [C*M]
    n = all_protos.shape[0]

    normed = F.normalize(all_protos, dim=2)  # per-node unit vectors, [C*M, N, D]
    sim_per_node = torch.einsum("and,bnd->abn", normed, normed)  # [C*M, C*M, N]
    sim = sim_per_node.mean(dim=2)  # [C*M, C*M] -- aggregate used for edge/cluster decisions
    same_client = client_of.unsqueeze(0) == client_of.unsqueeze(1)
    sim = sim.masked_fill(same_client, -1.0)  # never link same-client prototypes
    edge_i, edge_j = torch.nonzero(sim > delta, as_tuple=True)

    adj = [[] for _ in range(n)]
    for a, b in zip(edge_i.tolist(), edge_j.tolist()):
        adj[a].append(b)

    components = _bfs_components(adj, n)
    multi = [c for c in components if len(c) > 1]
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
