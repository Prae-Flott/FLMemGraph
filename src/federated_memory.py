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
import torch
import torch.nn.functional as F


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
    """client_codebooks: list of C tensors, each [M, D].
    client_usage_counts: list of C tensors, each [M] (Freq, non-negative).
    Returns (list of C tensors [M, D] -- P_G,n per client, diagnostics dict)."""
    C = len(client_codebooks)
    M, D = client_codebooks[0].shape
    device = client_codebooks[0].device

    all_protos = torch.cat(client_codebooks, dim=0)  # [C*M, D]
    client_of = torch.arange(C, device=device).repeat_interleave(M)  # [C*M]
    n = all_protos.shape[0]

    normed = F.normalize(all_protos, dim=1)
    sim = normed @ normed.T  # [C*M, C*M]
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
        P_S = torch.stack([all_protos[idx].mean(dim=0) for idx in shared_clusters])
    else:
        P_S = torch.empty(0, D, device=device)

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
                max_sim = normed[g_idx] @ normed[other_unclustered_global].T
                max_sim = float(max_sim.max())
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
        P_G.append(torch.cat([P_S, P_p_n], dim=0))

    diagnostics = {
        "num_components": len(components),
        "num_multi_client_clusters": len(multi),
        "num_shared_prototypes": int(P_S.shape[0]),
        "num_personalized_prototypes_per_client": n_personal_target,
    }
    return P_G, diagnostics
