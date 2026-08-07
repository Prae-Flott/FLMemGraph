"""
Joint Prototype Memory + three-level (node/edge/device) anomaly
detection, per `docs/joint_prototype_three_level_anomaly_prompt.md`.

## How this differs from `fl_model.FLGDNMemory` (the previous design)

`fl_model.py`'s memory head quantizes each node INDEPENDENTLY against its
own shared codebook (per-feature normality: "has THIS feature's value
been seen before, in isolation"). This module's `JointPrototypeMemory`
instead maintains prototypes that are a full `[N, D]` snapshot of ALL
nodes together -- a prototype answers "has the WHOLE device been in
roughly this joint state before" (e.g. "1500rpm + 1000N + normal
bearing"), not "has vibration alone looked like this." Retrieval picks
ONE best-matching joint prototype for the whole window, then node anomaly
is measured against THAT prototype's per-node component -- so node
anomaly is inherently conditioned on which operating regime the memory
thinks we're in, not compared against a pooled-across-all-regimes normal
range.

## Why the edge head predicts TREND (relative deviation), not absolute
## value -- the key design change from this project's earlier physics
## residual work

Every `kinematics.py`-style exact-magnitude residual tried on Paderborn
(`src/paderborn_physics.py`, `memory/paderborn-*-residual*.md`) made
detection WORSE, and the diagnosis was specific: this dataset's 4
discrete operating conditions confound the predictors (force/speed/
torque aren't varied independently), so a regression fit across POOLED
RAW MAGNITUDES can't reliably separate physical effects from
which-condition-this-is.

`TrendEdgeHead` sidesteps this by operating on DEVIATIONS from the
locally-matched joint prototype (`d_i = z_i - p_i*`), not raw magnitudes.
Physically: "does feature j's deviation from ITS OWN normal-for-this-
regime baseline track feature i's deviation from ITS OWN baseline, the
way their physical relation says it should" -- a trend/co-movement
question, answerable consistently across different operating regimes,
rather than "what absolute current value does this torque value predict"
-- a magnitude question that Paderborn's confounded 4-point design can't
support a reliable answer to. This is the literal implementation of
"关注趋势关系，不追求参数级别" from the design conversation that produced
`docs/joint_prototype_three_level_anomaly_prompt.md`.

## Simplifications vs. the full design doc (v1, deliberately scoped down)

- Only ONE generic learned edge operator type (a linear map
  `d_j_hat = W_ij @ d_i`, trained end-to-end with the encoder on healthy
  data) is implemented, not the full typed-relation taxonomy (proportional/
  integral/derivative/thermal/frequency-scaling/...) Sec 6.1 describes --
  matches this project's repeated choice (`kinematics.py`, `gdn_model.py`)
  to implement the simplest physically-motivated structure first, not a
  claim the richer taxonomy isn't worth adding later.
- The physics-edge SKELETON (which node pairs get an edge at all) is
  still supplied by the caller (domain knowledge), not learned -- matches
  Sec 6.1's "each edge needs a declared relation," just without yet
  discriminating relation TYPES from each other.
"""
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


class TrendEdgeHead(nn.Module):
    """Typed physics-skeleton edges, TREND-based (see module docstring
    for why): predicts node j's DEVIATION from its own matched-prototype
    baseline from node i's deviation from ITS baseline, via one learned
    linear map per edge -- not an absolute-value prediction."""

    def __init__(self, edges, embed_dim: int):
        """edges: list of (src_node_idx, dst_node_idx) int pairs, indices
        into the shared node ordering used everywhere else in the model."""
        super().__init__()
        self.edges = edges
        self.ops = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=False) for _ in edges])

    def forward(self, z, p_star):
        """z, p_star: [B, N, D]. Returns edge_scores [B, num_edges]
        (squared residual per edge) and preds (list of [B, D] predicted
        deviations, for diagnostics)."""
        d = z - p_star  # [B, N, D], deviation from matched prototype per node
        edge_scores, preds = [], []
        for (i, j), op in zip(self.edges, self.ops):
            pred_dj = op(d[:, i])  # [B, D]
            actual_dj = d[:, j]
            s = (actual_dj - pred_dj).pow(2).sum(-1)  # [B]
            edge_scores.append(s)
            preds.append(pred_dj)
        return torch.stack(edge_scores, dim=1), preds


class TrendGraphAttentionHead(nn.Module):
    """v2 edge/structural head, per the design correction: edge anomaly
    should be computed by GDN-style learned ATTENTION over each node's
    neighbors (TopK by learned embedding similarity, same mechanism as
    `gdn_model.GDN` / `fl_model.StructureHead`), not a fixed hand-declared
    edge list with one linear map per edge (`TrendEdgeHead` above, v1) --
    "对于物理先验没有表示的边，GDN也可以学习他们之间的关系" (for edges the
    physics prior doesn't cover, GDN can still learn them).

    Keeps the ONE genuinely new idea from v1 (operate on deviations from
    the matched joint prototype, `d = z - p_star`, not raw `z` -- see
    module docstring for why this sidesteps Paderborn's confounded-
    operating-condition collinearity problem): attention aggregates
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
        # A single LEARNED scalar strength scales the whole mask (matches this project's
        # earlier archived FTA-prior experiment's scalar-bias variant -- see
        # memory/paderborn-joint-prototype.md's "what's not done" list; the typed
        # per-relation-strength version found there is a natural future extension, not
        # implemented here).
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


class JointPrototypeGDNv2(nn.Module):
    """v2: same encoder + joint prototype memory as JointPrototypeGDN,
    but the edge/structural head is TrendGraphAttentionHead (learned
    attention over neighbors, physics-informed but not physics-
    restricted) instead of TrendEdgeHead (fixed edge list, one linear map
    each). See TrendGraphAttentionHead's docstring for the design
    correction this implements."""

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
        d_G, s_node), s_edge [B, N] (per-node structural anomaly from
        attention over deviations). `training_mode=True` also runs the
        decoder (dropped otherwise)."""
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


class JointPrototypeGDN(nn.Module):
    """Full model: shared encoder + joint prototype memory (device-level
    novelty + node anomaly) + trend edge head (edge anomaly) +
    training-only decoder (grounds z against representation collapse,
    dropped at inference -- same "解码器可裁" convention as
    fl_model.FLGDNMemory)."""

    def __init__(self, num_nodes: int, window_size: int, embed_dim: int,
                 num_prototypes: int, edges, node_weights=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.encoder = SharedEncoder(window_size, embed_dim)
        self.memory = JointPrototypeMemory(num_prototypes, num_nodes, embed_dim)
        self.edge_head = TrendEdgeHead(edges, embed_dim)
        self.decoder = nn.Linear(embed_dim, window_size)  # training-only
        self.register_buffer(
            "node_weights",
            node_weights if node_weights is not None else torch.ones(num_nodes) / num_nodes,
        )

    def forward(self, x, training_mode=False):
        """x: [B, T, N]. Returns dict with z, memory outputs (idx, p_star,
        d_G, s_node), edge_scores [B, num_edges]. `training_mode=True`
        also runs the decoder (dropped otherwise)."""
        B, T, N = x.shape
        assert N == self.num_nodes

        z = self.encoder(x)  # [B, N, D]
        mem_out = self.memory(z, self.node_weights)
        edge_scores, _ = self.edge_head(z, mem_out["p_star"])

        out = {"z": z, "edge_scores": edge_scores, **mem_out}
        if training_mode:
            out["x_hat"] = self.decoder(z).transpose(1, 2)  # [B, N, T] -> [B, T, N]
        return out

    def device_score(self, s_node, edge_scores, lambda_n: float = 1.0, lambda_e: float = 1.0):
        """S_device = lambda_n * mean(node scores) + lambda_e * mean(edge
        scores), Sec 8.1's simplest aggregation. s_node: [*, N],
        edge_scores: [*, num_edges] (numpy or torch, same leading dims)."""
        return lambda_n * s_node.mean(axis=-1) + lambda_e * edge_scores.mean(axis=-1)
