"""
Joint Prototype Memory + three-level (node/edge/device) anomaly
detection, per `docs/joint_prototype_three_level_anomaly_prompt.md` and
`docs/joint_prototype_physics_gdn_anomaly_attention_prompt.md`.

This module holds three versions this project's Joint Prototype Memory
line has produced (see `memory/joint-prototype-scheme-v2.md` and
`memory/joint-prototype-scheme-v3.md` for the full cross-dataset results
and rationale; earlier intermediate iterations -- a fixed-edge-list
version and a physics-residual precursor -- were superseded and removed,
their conclusions preserved in those memory files):

- **V2** (`SharedEncoder` + `JointPrototypeMemory`, no edge head at
  all): "has the WHOLE device been in roughly this joint state before"
  (`d_G`, device-level) + "has this ONE signal's value drifted from what
  it normally looks like in this operating regime" (`s_node`, per-node) --
  mechanism-agnostic, needs no declared physics relations. `JointPrototypeMemory`
  differs from `fl_model.FLGDNMemory`'s per-feature-independent codebook:
  prototypes here are a full `[N, D]` snapshot of ALL nodes together, so
  node anomaly is inherently conditioned on which joint operating regime
  the memory thinks the device is in.
- **V2.1** (`JointPrototypeGDNv21`): V2 + `TrendGraphAttentionHead` only
  (GDN-style learned attention over neighbors, no typed relation head) --
  treats an ABNORMAL CHANGE IN CROSS-FEATURE ATTENTION as one
  manifestation of a fault, without needing any declared physics edges
  (`prior_edges` is optional). Runs on any dataset, including ones with
  no verified physical relation.
- **Scheme V3** (`JointPrototypeGDNv3`): V2.1 + `TypedRelationAnomalyHead`
  (relation-specific message functions + prototype-conditioned residual
  standardization + anomaly attention over ONLY the declared edges).
  Needs domain knowledge of which signal pairs relate and how
  (proportional/nonlinear) -- pays off specifically when a fault's
  failure mechanism matches a declared relation breaking (e.g.
  Paderborn's vibration/torque/current coupling, voraus-AD's
  current-vs-torque miscommutation fault); V2 otherwise wins on
  faults that manifest as a single value drifting rather than a relation
  breaking (robo3er, most of voraus-AD, all of Sielaff).

Both operate on DEVIATIONS from the locally-matched joint prototype
(`d_i = z_i - p_i*`), not raw magnitudes -- "does feature j's deviation
from ITS OWN normal-for-this-regime baseline track feature i's deviation
from ITS OWN baseline the way their physical relation says it should," a
trend/co-movement question answerable consistently across different
operating regimes, unlike a raw-magnitude regression which can't
separate a physical effect from which regime a sample happens to be in
(the specific failure mode of this project's earlier, now-removed,
physics-residual precursor on Paderborn).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedEncoder(nn.Module):
    """One Linear(window_size -> D), shared across every node (channel-
    independent, PatchTST-style) -- same as fl_model.SharedEncoder."""

    def __init__(self, window_size: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(window_size, embed_dim)

    def forward(self, x):  # x: [B, T, N] -> z: [B, N, D]
        return self.proj(x.transpose(1, 2))


class JointPrototypeMemory(nn.Module):
    """M joint prototypes, each [N, D] -- a full multi-node snapshot, not
    N independent per-feature codebooks (see module docstring). Hard
    nearest-neighbor retrieval by weighted sum of per-node squared
    distances, VQ-VAE-style straight-through/commitment-loss training
    (no EMA), matching gdn_memory_model.DiscretePrototypicalMemory's
    convention but jointly over all N nodes instead of per-node."""

    def __init__(self, num_prototypes: int, num_nodes: int, embed_dim: int):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.num_nodes = num_nodes
        self.codebook = nn.Parameter(torch.randn(num_prototypes, num_nodes, embed_dim) * 0.02)
        self.register_buffer("usage_count", torch.zeros(num_prototypes))

    def forward(self, z, node_weights=None):
        """z: [B, N, D]. Returns dict with:
        idx [B] (matched prototype index), p_star [B, N, D] (matched
        prototype, straight-through for gradient), d_G [B] (global joint
        distance to the matched prototype), s_node [B, N] (per-node
        squared distance to the matched prototype's component)."""
        B, N, D = z.shape
        if node_weights is None:
            node_weights = torch.ones(N, device=z.device) / N

        diff = z.unsqueeze(1) - self.codebook.unsqueeze(0)  # [B, M, N, D]
        node_dist = diff.pow(2).sum(-1)  # [B, M, N]
        joint_dist = (node_dist * node_weights).sum(-1)  # [B, M]

        idx = joint_dist.argmin(dim=1)  # [B]
        p_star = self.codebook[idx]  # [B, N, D]
        d_G = joint_dist.gather(1, idx.unsqueeze(1)).squeeze(1)  # [B]
        s_node = node_dist.gather(1, idx.view(B, 1, 1).expand(-1, 1, N)).squeeze(1)  # [B, N]

        if self.training:
            with torch.no_grad():
                self.usage_count += torch.bincount(idx, minlength=self.num_prototypes).float()

        p_star_st = z + (p_star - z).detach()  # straight-through, matches DiscretePrototypicalMemory
        return {"idx": idx, "p_star": p_star, "p_star_st": p_star_st, "d_G": d_G, "s_node": s_node}

    def commitment_losses(self, z, p_star):
        l_me = F.mse_loss(z, p_star.detach())
        l_mc = F.mse_loss(z.detach(), p_star)
        return l_me, l_mc

    def codebook_utilization(self):
        return float((self.usage_count > 0).float().mean())


class TrendGraphAttentionHead(nn.Module):
    """Scheme V3's edge/structural head: edge anomaly is computed by
    GDN-style learned ATTENTION over each node's neighbors (TopK by
    learned embedding similarity, same mechanism as `gdn_model.GDN` /
    `fl_model.StructureHead`), not a fixed hand-declared edge list with
    one linear map per edge (an earlier, removed design) --
    "对于物理先验没有表示的边，GDN也可以学习他们之间的关系" (for edges the
    physics prior doesn't cover, GDN can still learn them).

    Operates on deviations from the matched joint prototype
    (`d = z - p_star`, not raw `z` -- see module docstring for why this
    sidesteps Paderborn's confounded-operating-condition collinearity
    problem): attention aggregates
    NEIGHBORS' deviations to predict THIS node's deviation, exactly the
    way `StructureHead` predicts z_i from other nodes' z_j, but now in
    deviation space.

    Known physics relations (if supplied) are NOT a hard edge restriction
    -- attention is computed over ALL other nodes (or a TopK subset by
    learned similarity, same as GDN), and declared physics edges only
    contribute an ADDITIVE, LEARNED-STRENGTH bias to the attention logits
    for that (source, target) pair -- "结构不锁死，只是加权" (informed
    prior, not a fixed skeleton). Undeclared relationships remain fully
    learnable, exactly matching the design correction. No self-loop, same
    reasoning as `StructureHead`: this is a same-instant consistency
    check, not a forecasting task with a temporal offset, so allowing a
    node to attend to itself would let it trivially "explain" its own
    deviation by copying it."""

    def __init__(self, num_nodes: int, embed_dim: int, top_k: int = None,
                 prior_edges=None, prior_bias_init: float = 1.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.top_k = min(top_k, num_nodes - 1) if top_k is not None else num_nodes - 1
        self.embeddings = nn.Embedding(num_nodes, embed_dim)
        nn.init.uniform_(self.embeddings.weight, -1.0, 1.0)
        self.attn_w = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_a = nn.Linear(2 * embed_dim, 1, bias=False)
        self.g = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # prior_mask[target, source] = 1 if (source -> target) is a declared physics edge.
        # A single LEARNED scalar strength scales the whole mask -- see
        # memory/joint-prototype-scheme-v3.md for why a per-relation-TYPE strength
        # (TypedRelationAnomalyHead below) was added on top of this single shared scalar.
        prior_mask = torch.zeros(num_nodes, num_nodes)
        if prior_edges:
            for src, dst in prior_edges:
                prior_mask[dst, src] = 1.0
        self.register_buffer("prior_mask", prior_mask)
        self.prior_bias_strength = nn.Parameter(torch.tensor(float(prior_bias_init)))

    def _topk_neighbors(self):
        v = F.normalize(self.embeddings.weight, dim=1)
        sim = v @ v.T
        sim.fill_diagonal_(-float("inf"))  # no self-loop, see class docstring
        _, topk_idx = sim.topk(self.top_k, dim=1)
        return topk_idx  # [N, top_k]

    def forward(self, z, p_star):
        """z, p_star: [B, N, D]. Returns d_hat [B, N, D] (predicted
        deviation per node, from its attended neighbors) and s_node_edge
        [B, N] (per-node structural/edge anomaly -- generalizes v1's
        per-fixed-edge scores into a per-node score covering ALL learned
        relationships, declared or not)."""
        B, N, D = z.shape
        d = z - p_star  # [B, N, D]
        wd = self.attn_w(d)

        neighbor_idx = self._topk_neighbors()  # [N, top_k], never self
        wd_neighbors = wd[:, neighbor_idx, :]  # [B, N, top_k, D]
        wd_self = wd.unsqueeze(2).expand(-1, -1, neighbor_idx.shape[1], -1)

        logits = self.attn_a(torch.cat([wd_self, wd_neighbors], dim=-1)).squeeze(-1)  # [B, N, top_k]
        prior_bias = self.prior_mask[torch.arange(N, device=z.device).unsqueeze(1), neighbor_idx]  # [N, top_k]
        logits = logits + self.prior_bias_strength * prior_bias.unsqueeze(0)

        alpha = F.softmax(F.leaky_relu(logits), dim=-1)
        agg = torch.einsum("bnk,bnkd->bnd", alpha, wd_neighbors)
        agg = F.relu(agg)

        gated = agg * self.embeddings.weight.unsqueeze(0)
        d_hat = self.g(gated)  # [B, N, D]

        s_node_edge = (d - d_hat).pow(2).sum(dim=-1)  # [B, N]
        return d_hat, s_node_edge


class JointPrototypeGDNv21(nn.Module):
    """V2.1: V2 (`SharedEncoder` + `JointPrototypeMemory`, see module
    docstring) + `TrendGraphAttentionHead` -- GDN-style learned attention
    over neighbors, treating an ABNORMAL CHANGE IN CROSS-FEATURE
    ATTENTION as one manifestation of a fault, on top of V2's node-level
    "did this one value drift" signal. Unlike V3, there is no typed
    relation head (`TypedRelationAnomalyHead`) -- `prior_edges` is
    optional and, when omitted, the attention is FULLY GENERIC (no
    physics bias at all), so this runs on any dataset with zero domain
    knowledge, including ones with no verified physical relation (e.g.
    Sielaff, `benchmark/datasets/sielaff_physics.md`).

    Named "V2.1" rather than reviving the deleted historical
    `JointPrototypeGDNv2` class under its old name, to avoid the naming
    collision documented in `memory/joint-prototype-scheme-v3.md` -- this
    project's current "V2" means something else (no edge signal at all).
    Structurally this class IS what that old class used to be, just
    trained independently rather than jointly with a typed-relation
    auxiliary loss -- see `memory/joint-prototype-scheme-v3.md`'s lineage
    section and its Paderborn caveat about joint-training confounds
    before comparing V2.1's numbers to V3's own C/D ablation columns."""

    def __init__(self, num_nodes: int, window_size: int, embed_dim: int,
                 num_prototypes: int, prior_edges=None, top_k: int = None, node_weights=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.encoder = SharedEncoder(window_size, embed_dim)
        self.memory = JointPrototypeMemory(num_prototypes, num_nodes, embed_dim)
        self.edge_head = TrendGraphAttentionHead(num_nodes, embed_dim, top_k=top_k, prior_edges=prior_edges)
        self.decoder = nn.Linear(embed_dim, window_size)  # training-only
        self.register_buffer(
            "node_weights",
            node_weights if node_weights is not None else torch.ones(num_nodes) / num_nodes,
        )

    def forward(self, x, training_mode=False):
        """x: [B, T, N]. Returns dict with z, memory outputs (idx, p_star,
        d_G, s_node), s_edge [B, N] (per-node structural/attention
        anomaly). `training_mode=True` also runs the decoder."""
        B, T, N = x.shape
        assert N == self.num_nodes

        z = self.encoder(x)
        mem_out = self.memory(z, self.node_weights)
        d_hat, s_edge = self.edge_head(z, mem_out["p_star"])

        out = {"z": z, "d_hat": d_hat, "s_edge": s_edge, **mem_out}
        if training_mode:
            out["x_hat"] = self.decoder(z).transpose(1, 2)
        return out

    def device_score(self, s_node, s_edge, lambda_n: float = 1.0, lambda_e: float = 1.0):
        return lambda_n * s_node.mean(axis=-1) + lambda_e * s_edge.mean(axis=-1)


class TypedRelationAnomalyHead(nn.Module):
    """v3: relation-specific per-edge message functions (Sec 9 of
    `docs/joint_prototype_physics_gdn_anomaly_attention_prompt.md`) +
    prototype-conditioned edge-residual standardization (Sec 12) +
    an anomaly attention over each node's incoming declared edges (Sec
    15-16), layered ALONGSIDE v2's `TrendGraphAttentionHead` rather than
    replacing it -- the two mechanisms answer different questions and the
    design doc is explicit that they must stay separate (Sec 13:
    "alpha_ij^rel != beta_ij^anom"). `TrendGraphAttentionHead` covers ALL
    node pairs generically and remains the primary structural signal for
    nodes with no declared incoming edge; this head only covers the
    DECLARED physics edges, but with type-appropriate functional forms,
    prototype-conditioned normalization, and a genuinely separate
    "which relationship is broken right now" attention.

    Simplifications vs. the full design doc (v3, deliberately scoped
    down, matching this project's repeated "implement the simplest sound
    version first" convention -- see `TrendEdgeHead`/`TrendGraphAttentionHead`
    docstrings for the same pattern at v1/v2):
    - Only 2 relation-type classes are distinguished (linear/proportional
      vs. nonlinear/dynamic), assigned by domain knowledge per declared
      edge (see `run_paderborn_joint_prototype_v3.py`'s EDGE_TYPES), not
      the full 9-type taxonomy in Sec 7 -- assigning finer types
      (integral/derivative/thermal/frequency) would need dedicated
      temporal-difference/aggregation structure this project has no
      validated need for yet on Paderborn's 6 channels.
    - Anomaly attention beta_ij is the UNSUPERVISED Sec 16 version
      (softmax over standardized residual magnitude with a learned
      temperature), not the self-supervised relation-breaking-augmentation
      trained version from Sec 17 -- that augmentation pipeline (temporal
      mismatch, cross-condition swap, scaling/trend/frequency perturbation)
      is a substantial separate undertaking, left as a documented future
      step once this simpler version's value is established.
    - Prototype-conditioned standardization falls back to a GLOBAL mean/std
      for any prototype with fewer than `min_samples` calibration windows,
      since a 16-prototype codebook can have sparsely populated entries.
    """

    def __init__(self, num_nodes: int, edges, edge_types, embed_dim: int, temperature_init: float = 1.0):
        super().__init__()
        assert len(edges) == len(edge_types)
        self.num_nodes = num_nodes
        self.edges = edges
        self.edge_types = edge_types

        self.ops = nn.ModuleList()
        for t in edge_types:
            if t == "proportional":
                self.ops.append(nn.Linear(embed_dim, embed_dim, bias=False))
            else:  # "nonlinear" / "dynamic" -- anything not confidently proportional
                self.ops.append(nn.Sequential(
                    nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim),
                ))

        self.by_target = {}
        for e_idx, (_src, dst) in enumerate(edges):
            self.by_target.setdefault(dst, []).append(e_idx)

        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature_init))))

        num_edges = len(edges)
        self.register_buffer("calib_mu_global", torch.zeros(num_edges))
        self.register_buffer("calib_sigma_global", torch.ones(num_edges))
        self.calib_mu_proto = None
        self.calib_sigma_proto = None
        self.calib_valid_proto = None

    def raw_residuals(self, z, p_star):
        """z, p_star: [B, N, D]. Returns r [B, num_edges], the raw squared
        residual per declared edge using each edge's typed message function."""
        d = z - p_star
        rs = []
        for (src, dst), op in zip(self.edges, self.ops):
            pred = op(d[:, src])
            actual = d[:, dst]
            rs.append((actual - pred).pow(2).sum(-1))
        return torch.stack(rs, dim=1)

    def set_calibration(self, r_calib_np, idx_calib_np, num_prototypes, min_samples: int = 5):
        """r_calib_np: [Ncalib, num_edges] raw per-window residuals; idx_calib_np:
        [Ncalib] matched prototype index -- both from a calibration pass run AFTER
        training (Stage B, Sec 22). Computes global stats and, per prototype, either
        its own stats (if enough calib windows matched it) or a fallback to global."""
        mu_g = r_calib_np.mean(axis=0)
        sigma_g = r_calib_np.std(axis=0) + 1e-8
        self.calib_mu_global = torch.tensor(mu_g, dtype=torch.float32)
        self.calib_sigma_global = torch.tensor(sigma_g, dtype=torch.float32)

        mu_p = np.tile(mu_g, (num_prototypes, 1))
        sigma_p = np.tile(sigma_g, (num_prototypes, 1))
        valid = np.zeros(num_prototypes, dtype=bool)
        for m in range(num_prototypes):
            mask = idx_calib_np == m
            if int(mask.sum()) >= min_samples:
                mu_p[m] = r_calib_np[mask].mean(axis=0)
                sigma_p[m] = r_calib_np[mask].std(axis=0) + 1e-8
                valid[m] = True
        self.calib_mu_proto = torch.tensor(mu_p, dtype=torch.float32)
        self.calib_sigma_proto = torch.tensor(sigma_p, dtype=torch.float32)
        self.calib_valid_proto = torch.tensor(valid)

    def standardize(self, r, idx):
        device = r.device
        mu_global = self.calib_mu_global.to(device).unsqueeze(0).expand_as(r)
        sigma_global = self.calib_sigma_global.to(device).unsqueeze(0).expand_as(r)
        if self.calib_mu_proto is None:
            return (r - mu_global) / sigma_global
        mu_proto = self.calib_mu_proto.to(device)[idx]
        sigma_proto = self.calib_sigma_proto.to(device)[idx]
        valid = self.calib_valid_proto.to(device)[idx].unsqueeze(1)
        mu = torch.where(valid, mu_proto, mu_global)
        sigma = torch.where(valid, sigma_proto, sigma_global)
        return (r - mu) / sigma

    def forward(self, z, p_star, idx):
        """Returns dict with r [B, num_edges] (raw), r_tilde [B, num_edges]
        (prototype-conditioned standardized, falls back to global stats before
        `set_calibration()` is called -- fine during Stage A training), and
        s_node_typed [B, N] (anomaly-attention-weighted standardized residual per
        node; 0 for nodes with no declared incoming edge, e.g. force/speed here)."""
        r = self.raw_residuals(z, p_star)
        r_tilde = self.standardize(r, idx)

        B = r.shape[0]
        s_node_typed = torch.zeros(B, self.num_nodes, device=r.device)
        temperature = self.log_temperature.exp()
        for target, e_idxs in self.by_target.items():
            r_t_group = r_tilde[:, e_idxs]
            beta = F.softmax(temperature * r_t_group, dim=-1)
            s_node_typed[:, target] = (beta * r_t_group.clamp(min=0)).sum(dim=-1)
        return {"r": r, "r_tilde": r_tilde, "s_node_typed": s_node_typed}


class JointPrototypeGDNv3(nn.Module):
    """v3: adds `TypedRelationAnomalyHead` (relation-specific message
    functions + prototype-conditioned standardization + anomaly attention
    over declared physics edges) alongside v2's `TrendGraphAttentionHead`
    (kept unchanged -- generic learned attention over ALL node pairs,
    still the primary structural signal for nodes with no declared
    incoming edge). See `TypedRelationAnomalyHead` docstring for the
    design-doc section mapping and explicit scoping-down notes."""

    def __init__(self, num_nodes: int, window_size: int, embed_dim: int,
                 num_prototypes: int, prior_edges, edge_types, top_k: int = None, node_weights=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.encoder = SharedEncoder(window_size, embed_dim)
        self.memory = JointPrototypeMemory(num_prototypes, num_nodes, embed_dim)
        self.edge_head = TrendGraphAttentionHead(num_nodes, embed_dim, top_k=top_k, prior_edges=prior_edges)
        self.typed_head = TypedRelationAnomalyHead(num_nodes, prior_edges, edge_types, embed_dim)
        self.decoder = nn.Linear(embed_dim, window_size)  # training-only
        self.register_buffer(
            "node_weights",
            node_weights if node_weights is not None else torch.ones(num_nodes) / num_nodes,
        )

    def forward(self, x, training_mode=False):
        """x: [B, T, N]. Returns dict with z, memory outputs (idx, p_star,
        d_G, s_node), s_edge [B, N] (v2 generic attention), r/r_tilde/
        s_node_typed (v3 typed-edge signals). `training_mode=True` also
        runs the decoder."""
        B, T, N = x.shape
        assert N == self.num_nodes

        z = self.encoder(x)
        mem_out = self.memory(z, self.node_weights)
        d_hat, s_edge = self.edge_head(z, mem_out["p_star"])
        typed_out = self.typed_head(z, mem_out["p_star"], mem_out["idx"])

        out = {"z": z, "d_hat": d_hat, "s_edge": s_edge, **mem_out, **typed_out}
        if training_mode:
            out["x_hat"] = self.decoder(z).transpose(1, 2)
        return out

    def device_score(self, s_node, s_edge, s_node_typed, lambda_n: float = 1.0,
                      lambda_e: float = 1.0, lambda_t: float = 1.0):
        return (lambda_n * s_node.mean(axis=-1) + lambda_e * s_edge.mean(axis=-1)
                + lambda_t * s_node_typed.mean(axis=-1))
