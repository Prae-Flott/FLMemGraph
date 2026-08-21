"""
Joint Prototype Memory + three-level (node/edge/device) anomaly
detection, per `docs/joint_prototype_three_level_anomaly_prompt.md` and
`docs/joint_prototype_physics_gdn_anomaly_attention_prompt.md`.

Four model classes, organized into two families by physics-prior requirement
(see `memory/scoring-signals-B-C-E-H.md` for current cross-dataset results,
and `memory/joint-prototype-scheme-v3.md` for architecture rationale):

No-prior family (runs on any dataset, no domain knowledge needed):
- **`JointPrototypeV2`**: SharedEncoder + JointPrototypeMemory +
  TrendGraphAttentionHead. Signals: d_proto (global prototype distance),
  d_node [N] (per-node deviation), resid_struct [N] (attention residual).
- **`JointPrototypeV21`**: JointPrototypeV2 + DeviationCovarianceHead.
  Adds d_mahal [B] (Mahalanobis distance of joint d_node pattern from
  prototype-conditioned normal). Stage-B calibration required.

Physics-prior family (requires declared edges + relation types):
- **`JointPrototypeV3`**: JointPrototypeV2 + TypedRelationAnomalyHead
  (typed message functions per edge + prototype-conditioned standardization
  + anomaly attention). Adds r_edge [E], r_tilde [E], resid_phys [N].
- **`JointPrototypeV31`**: JointPrototypeV3 + DeviationCovarianceHead.
  Adds d_mahal [B]. Stage-B calibration required.

Pays off specifically when a fault's failure mechanism matches a declared
relation breaking (Paderborn bearing, voraus-AD motor miscommutation);
V2/V21 wins on faults that manifest as single-node value drift (robo3er,
most of voraus-AD, all of Sielaff).

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

    def __init__(self, num_prototypes: int, num_nodes: int, embed_dim: int,
                 ema_decay: float = 0.99, ema_warmup_steps: int = 20):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.num_nodes = num_nodes
        self.codebook = nn.Parameter(torch.randn(num_prototypes, num_nodes, embed_dim) * 0.02)
        self.register_buffer("usage_count", torch.zeros(num_prototypes))

        # Path A of memory/calib-in-prototype-ab.md: online, VQ-VAE-codebook-EMA-style
        # per-prototype mean/var of d_node, as an ALTERNATIVE to Path B's offline
        # calib-split pass (ScoreCalibrationHead above). Not touched unless
        # `update_ema()` is called explicitly (calib_mode="ema" only) -- default
        # behavior for every other calib_mode is unaffected by these buffers existing.
        self.ema_decay = ema_decay
        self.ema_warmup_steps = ema_warmup_steps
        self.register_buffer("ema_step", torch.zeros(1))
        self.register_buffer("ema_mean", torch.zeros(num_prototypes, num_nodes))
        self.register_buffer("ema_var", torch.ones(num_prototypes, num_nodes))
        self.register_buffer("ema_initialized", torch.zeros(num_prototypes, dtype=torch.bool))
        self.register_buffer("ema_global_mean", torch.zeros(num_nodes))
        self.register_buffer("ema_global_var", torch.ones(num_nodes))
        self.register_buffer("ema_global_initialized", torch.zeros(1, dtype=torch.bool))

    def forward(self, z, node_weights=None):
        """z: [B, N, D]. Returns dict with:
        idx [B] (matched prototype index), p_star [B, N, D] (matched
        prototype, straight-through for gradient), d_proto [B] (global joint
        distance to the matched prototype = weighted mean of d_node),
        d_node [B, N] (per-node squared distance to the matched prototype's
        component; no attention, purely per-node amplitude deviation)."""
        B, N, D = z.shape
        if node_weights is None:
            node_weights = torch.ones(N, device=z.device) / N

        diff = z.unsqueeze(1) - self.codebook.unsqueeze(0)  # [B, M, N, D]
        sq_dist = diff.pow(2).sum(-1)  # [B, M, N]
        joint_dist = (sq_dist * node_weights).sum(-1)  # [B, M]

        idx = joint_dist.argmin(dim=1)  # [B]
        p_star = self.codebook[idx]  # [B, N, D]
        d_proto = joint_dist.gather(1, idx.unsqueeze(1)).squeeze(1)  # [B]
        d_node = sq_dist.gather(1, idx.view(B, 1, 1).expand(-1, 1, N)).squeeze(1)  # [B, N]

        if self.training:
            with torch.no_grad():
                self.usage_count += torch.bincount(idx, minlength=self.num_prototypes).float()

        p_star_st = z + (p_star - z).detach()  # straight-through, matches DiscretePrototypicalMemory
        return {"idx": idx, "p_star": p_star, "p_star_st": p_star_st, "d_proto": d_proto, "d_node": d_node}

    def commitment_losses(self, z, p_star):
        l_me = F.mse_loss(z, p_star.detach())
        l_mc = F.mse_loss(z.detach(), p_star)
        return l_me, l_mc

    def codebook_utilization(self):
        return float((self.usage_count > 0).float().mean())

    @torch.no_grad()
    def update_ema(self, d_node: torch.Tensor, idx: torch.Tensor):
        """Path A: online per-prototype mean/var EMA update of `d_node`,
        analogous to a VQ-VAE codebook's EMA update -- call once per
        training step (AFTER `forward()`, with `d_node`/`idx` detached)
        when `calib_mode="ema"`. Gated by `ema_warmup_steps`: prototype
        assignments are unstable in early training (codebook still
        settling), so the first `ema_warmup_steps` calls are counted but
        do not update the running statistics -- avoids baking in noisy
        early-training deviation scales. Mirrors `DeviationCovarianceHead`
        mean/var (NOT median/IQR -- an EMA has no closed-form running
        median, this is Path A's documented divergence from Path B/the
        current global-median/IQR script-level behavior). A global
        (non-prototype-conditioned) EMA is also tracked unconditionally
        as an initialization/fallback source for prototypes never seen
        before warm-up ends."""
        self.ema_step += 1
        # unconditional global EMA (available even before/without warm-up, as fallback)
        batch_mean_g = d_node.mean(dim=0)
        batch_var_g = d_node.var(dim=0, unbiased=False) if d_node.shape[0] > 1 else torch.zeros_like(batch_mean_g)
        if not bool(self.ema_global_initialized.item()):
            self.ema_global_mean = batch_mean_g
            self.ema_global_var = batch_var_g.clamp(min=1e-6)
            self.ema_global_initialized[0] = True
        else:
            d = self.ema_decay
            self.ema_global_mean = d * self.ema_global_mean + (1 - d) * batch_mean_g
            self.ema_global_var = d * self.ema_global_var + (1 - d) * batch_var_g.clamp(min=1e-6)

        if self.ema_step.item() < self.ema_warmup_steps:
            return
        for m in torch.unique(idx):
            mask = idx == m
            n_m = int(mask.sum().item())
            if n_m == 0:
                continue
            batch_mean = d_node[mask].mean(dim=0)
            batch_var = d_node[mask].var(dim=0, unbiased=False) if n_m > 1 else torch.zeros_like(batch_mean)
            mi = int(m.item())
            if not bool(self.ema_initialized[mi].item()):
                self.ema_mean[mi] = batch_mean
                self.ema_var[mi] = batch_var.clamp(min=1e-6)
                self.ema_initialized[mi] = True
            else:
                d = self.ema_decay
                self.ema_mean[mi] = d * self.ema_mean[mi] + (1 - d) * batch_mean
                self.ema_var[mi] = d * self.ema_var[mi] + (1 - d) * batch_var.clamp(min=1e-6)

    def ema_zscore(self, raw: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """raw: [B, num_nodes] numpy, idx: [B] numpy matched-prototype
        indices. Returns EMA-mean/std z-scored [B, num_nodes], numpy --
        Path A's drop-in replacement for `zscore(x, calib_x)`'s first
        stage, using the online-accumulated statistics instead of an
        offline calib pass. Falls back to the global EMA for prototypes
        never initialized (e.g. unused this federated round, or warm-up
        never reached)."""
        valid = self.ema_initialized.cpu().numpy()[idx]
        cmean = self.ema_mean.cpu().numpy()[idx]
        cstd = np.sqrt(self.ema_var.cpu().numpy()[idx])
        gmean = self.ema_global_mean.cpu().numpy()[None, :]
        gstd = np.sqrt(self.ema_global_var.cpu().numpy())[None, :]
        mean = np.where(valid[:, None], cmean, gmean)
        std = np.where(valid[:, None], cstd, gstd)
        return (raw - mean) / np.maximum(std, 1e-8)

    def load_memory(self, prototypes: torch.Tensor):
        """Overwrite this client's codebook with a server-broadcast [M, N, D]
        tensor (`federated_memory.align_and_split`'s P_G,n) and reset usage
        counts for the new round -- same convention as `fl_model.FLGDNMemory.load_memory`.

        Also resets ALL EMA calibration state (Path A, `calib_mode="ema"`).
        Bug fix (2026-08-21): this used to only zero `usage_count`, leaving
        `ema_mean`/`ema_var`/`ema_initialized`/`ema_step`/the global EMA
        buffers untouched across rounds. Since the codebook is a completely
        different set of vectors after each `load_memory()` call, slot index
        `m`'s "meaning" changes every round, but the old EMA stats (indexed
        purely by `m`) kept being blended in at `ema_decay` (0.99, i.e. 99%
        weight on the stale value) via `update_ema()` -- and since `ema_step`
        was never reset either, `ema_warmup_steps`'s "don't trust early
        updates" gate only ever fired once, at the very start of round 1, not
        again after every subsequent codebook swap. Net effect: from round 2
        onward, every EMA update was dominated by statistics computed under a
        codebook geometry that no longer existed. This was the dominant cause
        of Path A's measured AUROC regression, not the (also real, but
        smaller) fit-split bias risk it was originally flagged for -- see
        [[calib-in-prototype-ab]]. Resetting here means each round must earn
        its own warm-up again; `ema_warmup_steps` was lowered from 200 to 20
        to fit within a single round's local-epoch step budget (robo3er's
        smallest federated client has ~135 fit windows / batch_size=256, i.e.
        as few as ~12 steps per local epoch)."""
        with torch.no_grad():
            self.codebook.copy_(prototypes.to(self.codebook.device))
        self.usage_count.zero_()
        self.ema_step.zero_()
        self.ema_mean.zero_()
        self.ema_var.fill_(1.0)
        self.ema_initialized.zero_()
        self.ema_global_mean.zero_()
        self.ema_global_var.fill_(1.0)
        self.ema_global_initialized.zero_()

    @torch.no_grad()
    def load_fed_ema_stats(self, n: np.ndarray, mean: np.ndarray, var: np.ndarray):
        """"federated_ema" of `memory/calib-in-prototype-ab.md`: load
        server-merged per-prototype deviation statistics directly into the
        SAME `ema_mean`/`ema_var`/`ema_initialized` buffers `update_ema()`/
        `ema_zscore()` already use (so `ema_zscore()` is reused unchanged for
        scoring) -- but populated by a one-shot per-round sufficient-
        statistics merge (`federated_memory.align_and_split`'s
        `client_dev_stats` path) instead of `update_ema()`'s decayed online
        accumulation. Call once per round, right after `load_memory()` (which
        resets these buffers to scratch) -- this then overwrites them with
        the real per-round values instead of leaving them at scratch.

        `n` [M] (per-prototype sample count from this round's merge, 0 for
        slots nobody routed traffic to), `mean`/`var` [M, num_nodes] (merged
        for shared-cluster slots; this client's own freshly-computed values,
        untouched, for personalized slots -- see `align_and_split`'s
        docstring for why personalized slots don't need a cross-client
        merge). Slots with `n <= 0` are marked NOT initialized (so
        `ema_zscore()` falls back to the global estimate below for them,
        same convention as the plain `ema` mode), not left at `load_memory()`'s
        raw 0/1 scratch defaults.

        The (client-local, NOT federated) global fallback estimate is
        recomputed here as the sample-count-weighted pool of every
        populated prototype slot THIS client has this round -- the design
        only specifies federating the per-prototype merge; the global
        fallback is the same "hardly used once per-prototype coverage
        improves" role it already plays in `ema_zscore()`, so it is not
        additionally federated across clients."""
        device = self.ema_mean.device
        n_t = torch.tensor(n, dtype=torch.float32, device=device)
        valid = n_t > 0
        mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
        var_t = torch.tensor(var, dtype=torch.float32, device=device).clamp(min=1e-6)

        self.ema_mean = torch.where(valid.unsqueeze(1), mean_t, self.ema_mean)
        self.ema_var = torch.where(valid.unsqueeze(1), var_t, self.ema_var)
        self.ema_initialized = valid

        if bool(valid.any()):
            n_v = n_t[valid]
            mean_v = mean_t[valid]
            var_v = var_t[valid]
            total_n = n_v.sum()
            g_mean = (n_v.unsqueeze(1) * mean_v).sum(0) / total_n
            g_m2 = (n_v.unsqueeze(1) * var_v + n_v.unsqueeze(1) * (mean_v - g_mean.unsqueeze(0)) ** 2).sum(0)
            self.ema_global_mean = g_mean
            self.ema_global_var = (g_m2 / total_n).clamp(min=1e-6)
            self.ema_global_initialized[0] = True


class DeviationCovarianceHead(nn.Module):
    """No-prior anomaly signal: per-prototype covariance of per-node deviation
    norms (s_node [N]) estimated at calibration time, scored at inference via
    Mahalanobis distance from the matched prototype's normal joint-deviation
    structure.

    Captures regime-conditioned co-deviation patterns that s_node misses:
    s_node asks "is node i's value abnormal?"; this asks "is the JOINT pattern
    of which nodes deviate together abnormal for this operating regime?" -- the
    two questions are orthogonal (a fault can shift one node alone, or it can
    change which nodes co-deviate without changing their individual magnitudes).

    No training parameters -- pure post-training calibration statistics, same
    Stage B convention as TypedRelationAnomalyHead. Ridge regularization
    (reg * I) ensures invertibility even when calib_windows < num_nodes.
    Falls back to global (across-prototype) covariance for prototypes with
    fewer than min_samples calib windows."""

    def __init__(self, num_prototypes: int, num_nodes: int, reg: float = 1e-2):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.num_nodes = num_nodes
        self.reg = reg
        self.register_buffer("calib_mu", torch.zeros(num_prototypes, num_nodes))
        self.register_buffer("calib_cov_inv",
                             torch.eye(num_nodes).unsqueeze(0).repeat(num_prototypes, 1, 1))
        self.register_buffer("calib_valid", torch.zeros(num_prototypes, dtype=torch.bool))
        self.register_buffer("global_mu", torch.zeros(num_nodes))
        self.register_buffer("global_cov_inv", torch.eye(num_nodes))

    def set_calibration(self, d_node_calib: np.ndarray, idx_calib: np.ndarray,
                        min_samples: int = 10):
        """d_node_calib: [Ncalib, N] numpy (per-node squared deviation norms from
        JointPrototypeMemory.forward's d_node output); idx_calib: [Ncalib] numpy
        (matched prototype indices). Computes global and per-prototype mean and
        inverse covariance of the joint deviation pattern."""
        N, reg = self.num_nodes, self.reg
        g_mu = d_node_calib.mean(axis=0)
        g_cov = np.cov(d_node_calib.T) + reg * np.eye(N) if len(d_node_calib) >= 2 else reg * np.eye(N)
        g_cov_inv = np.linalg.inv(g_cov)
        self.global_mu = torch.tensor(g_mu, dtype=torch.float32)
        self.global_cov_inv = torch.tensor(g_cov_inv, dtype=torch.float32)

        mu_all = np.tile(g_mu, (self.num_prototypes, 1))
        cov_inv_all = np.tile(g_cov_inv, (self.num_prototypes, 1, 1))
        valid = np.zeros(self.num_prototypes, dtype=bool)
        for m in range(self.num_prototypes):
            mask = idx_calib == m
            n_m = int(mask.sum())
            if n_m >= min_samples and n_m >= 2:
                d_m = d_node_calib[mask]
                mu_all[m] = d_m.mean(axis=0)
                cov_inv_all[m] = np.linalg.inv(np.cov(d_m.T) + reg * np.eye(N))
                valid[m] = True
        self.calib_mu = torch.tensor(mu_all, dtype=torch.float32)
        self.calib_cov_inv = torch.tensor(cov_inv_all, dtype=torch.float32)
        self.calib_valid = torch.tensor(valid)

    @torch.no_grad()
    def forward(self, d_node: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """d_node: [B, N], idx: [B]. Returns d_mahal [B] (Mahalanobis distance
        of the current window's per-node deviation pattern from the matched
        prototype's normal joint-deviation distribution)."""
        device = d_node.device
        valid = self.calib_valid.to(device)[idx]  # [B]
        mu = torch.where(
            valid.unsqueeze(1),
            self.calib_mu.to(device)[idx],
            self.global_mu.to(device).unsqueeze(0).expand(len(idx), -1),
        )
        cov_inv = torch.where(
            valid.unsqueeze(1).unsqueeze(2),
            self.calib_cov_inv.to(device)[idx],
            self.global_cov_inv.to(device).unsqueeze(0).expand(len(idx), -1, -1),
        )
        diff = d_node - mu  # [B, N]
        return torch.einsum("bi,bij,bj->b", diff, cov_inv, diff)  # [B]


class ScoreCalibrationHead(nn.Module):
    """Per-prototype (with global fallback) median/IQR calibration for a
    raw [B, N] score array (`d_node`/`resid_struct`/`k_resid` -- signals
    B/C/K), replacing the GLOBAL median/IQR computed ad-hoc at benchmark-
    script level (`zscore()`/`two_stage_group_score()` in
    `run_*_federated.py`). Structured to match `DeviationCovarianceHead`/
    `TypedRelationAnomalyHead`'s existing pattern exactly: no trainable
    parameters, buffers populated by a one-time offline `set_calibration()`
    call after training from the held-out calib split, `min_samples`
    fallback to global stats for sparsely-populated prototypes. Kept as a
    separate small head (not folded into `JointPrototypeMemory`) since it's
    reused identically for 3 different raw score arrays of possibly
    different dimensionality (`num_dims` = num_nodes for B/C, num_nodes for
    K too, but this class doesn't assume any particular meaning for the
    dimension).

    This is Path B of `memory/calib-in-prototype-ab.md`'s branch experiment
    -- median/IQR (not mean/std) is kept deliberately, matching what B/C/K
    already used at script level, unlike H/E which already used mean/std.
    Per-prototype IS a real behavior change vs. the current global
    behavior (not just a refactor); `calib_mode="global"` bypasses this
    class entirely and reproduces the exact prior numbers (regression
    check)."""

    def __init__(self, num_prototypes: int, num_dims: int, min_samples: int = 10):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.num_dims = num_dims
        self.min_samples = min_samples
        self.register_buffer("calib_median", torch.zeros(num_prototypes, num_dims))
        self.register_buffer("calib_iqr", torch.ones(num_prototypes, num_dims))
        self.register_buffer("calib_valid", torch.zeros(num_prototypes, dtype=torch.bool))
        self.register_buffer("global_median", torch.zeros(num_dims))
        self.register_buffer("global_iqr", torch.ones(num_dims))

    def set_calibration(self, raw_calib: np.ndarray, idx_calib: np.ndarray, min_samples: int = None):
        """raw_calib: [Ncalib, num_dims] numpy (e.g. d_node from the calib
        split); idx_calib: [Ncalib] numpy matched-prototype indices. Same
        calib split / disjoint-from-fit-set discipline as
        `DeviationCovarianceHead.set_calibration`."""
        min_samples = self.min_samples if min_samples is None else min_samples
        g_med = np.median(raw_calib, axis=0)
        q75, q25 = np.percentile(raw_calib, [75, 25], axis=0)
        g_iqr = np.maximum(q75 - q25, 1e-8)
        self.global_median = torch.tensor(g_med, dtype=torch.float32)
        self.global_iqr = torch.tensor(g_iqr, dtype=torch.float32)

        med_all = np.tile(g_med, (self.num_prototypes, 1))
        iqr_all = np.tile(g_iqr, (self.num_prototypes, 1))
        valid = np.zeros(self.num_prototypes, dtype=bool)
        for m in range(self.num_prototypes):
            mask = idx_calib == m
            n_m = int(mask.sum())
            if n_m >= min_samples:
                d_m = raw_calib[mask]
                med_all[m] = np.median(d_m, axis=0)
                q75m, q25m = np.percentile(d_m, [75, 25], axis=0)
                iqr_all[m] = np.maximum(q75m - q25m, 1e-8)
                valid[m] = True
        self.calib_median = torch.tensor(med_all, dtype=torch.float32)
        self.calib_iqr = torch.tensor(iqr_all, dtype=torch.float32)
        self.calib_valid = torch.tensor(valid)

    def zscore(self, raw: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """raw: [B, num_dims] numpy, idx: [B] numpy matched-prototype
        indices (for calib-split rows, pass the SAME idx used to fit this
        head). Returns per-prototype (global-fallback) median/IQR z-scored
        [B, num_dims], numpy -- drop-in replacement for the script-level
        `zscore(x, calib_x)`'s first stage."""
        valid = self.calib_valid.cpu().numpy()[idx]
        cmed = self.calib_median.cpu().numpy()[idx]
        ciqr = self.calib_iqr.cpu().numpy()[idx]
        gmed = self.global_median.cpu().numpy()[None, :]
        giqr = self.global_iqr.cpu().numpy()[None, :]
        med = np.where(valid[:, None], cmed, gmed)
        iqr = np.where(valid[:, None], ciqr, giqr)
        return (raw - med) / iqr


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
        deviation per node, from its attended neighbors) and resid_struct
        [B, N] (per-node structural residual -- squared error between each
        node's actual deviation and what the learned cross-node attention
        predicted; covers ALL node pairs, declared or not)."""
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

        resid_struct = (d - d_hat).pow(2).sum(dim=-1)  # [B, N]
        return d_hat, resid_struct


class JointPrototypeV2(nn.Module):
    """No-prior model: SharedEncoder + JointPrototypeMemory +
    TrendGraphAttentionHead. Runs on any dataset with no domain
    knowledge -- attention is fully generic (no physics-edge bias) when
    `prior_edges=None`. Detection signals: d_proto (global prototype
    distance), d_node [B,N] (per-node amplitude deviation), resid_struct
    [B,N] (cross-node attention prediction error).

    `JointPrototypeV21` extends this with `DeviationCovarianceHead`
    (covariance-based joint-deviation signal, Stage-B calibration)."""

    def __init__(self, num_nodes: int, window_size: int, embed_dim: int,
                 num_prototypes: int, prior_edges=None, top_k: int = None,
                 node_weights=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.encoder = SharedEncoder(window_size, embed_dim)
        self.memory = JointPrototypeMemory(num_prototypes, num_nodes, embed_dim)
        self.edge_head = TrendGraphAttentionHead(num_nodes, embed_dim, top_k=top_k,
                                                 prior_edges=prior_edges)
        self.decoder = nn.Linear(embed_dim, window_size)  # training-only
        self.register_buffer(
            "node_weights",
            node_weights if node_weights is not None else torch.ones(num_nodes) / num_nodes,
        )

    def forward(self, x, training_mode=False):
        """x: [B, T, N]. Returns dict: z, idx, p_star, p_star_st,
        d_proto [B], d_node [B,N], d_hat [B,N,D], resid_struct [B,N].
        `training_mode=True` also computes x_hat via decoder."""
        B, T, N = x.shape
        assert N == self.num_nodes
        z = self.encoder(x)
        mem_out = self.memory(z, self.node_weights)
        d_hat, resid_struct = self.edge_head(z, mem_out["p_star"])
        out = {"z": z, "d_hat": d_hat, "resid_struct": resid_struct, **mem_out}
        if training_mode:
            out["x_hat"] = self.decoder(z).transpose(1, 2)
        return out

    def device_score(self, d_node, resid_struct, lambda_n: float = 1.0,
                     lambda_e: float = 1.0):
        return lambda_n * d_node.mean(axis=-1) + lambda_e * resid_struct.mean(axis=-1)


class JointPrototypeV21(JointPrototypeV2):
    """V2 + DeviationCovarianceHead: adds a per-prototype covariance model
    over d_node, scored at inference via Mahalanobis distance (d_mahal [B]).
    Stage-B calibration (`cov_head.set_calibration`) must be called after
    training before d_mahal carries signal. All other outputs identical to
    JointPrototypeV2."""

    def __init__(self, num_nodes: int, window_size: int, embed_dim: int,
                 num_prototypes: int, prior_edges=None, top_k: int = None,
                 node_weights=None):
        super().__init__(num_nodes, window_size, embed_dim, num_prototypes,
                         prior_edges=prior_edges, top_k=top_k,
                         node_weights=node_weights)
        self.cov_head = DeviationCovarianceHead(num_prototypes, num_nodes)
        # Path B (memory/calib-in-prototype-ab.md): see JointPrototypeV31's identical
        # score_calib_node/struct additions -- same purpose, mirrored here for V21's
        # no-declared-edge datasets (Sielaff).
        self.score_calib_node = ScoreCalibrationHead(num_prototypes, num_nodes)
        self.score_calib_struct = ScoreCalibrationHead(num_prototypes, num_nodes)

    def forward(self, x, training_mode=False):
        out = super().forward(x, training_mode=training_mode)
        out["d_mahal"] = self.cov_head(out["d_node"], out["idx"])
        return out

    def set_score_calibration(self, d_node_calib: np.ndarray, resid_struct_calib: np.ndarray,
                              idx_calib: np.ndarray, min_samples: int = 10):
        self.score_calib_node.set_calibration(d_node_calib, idx_calib, min_samples=min_samples)
        self.score_calib_struct.set_calibration(resid_struct_calib, idx_calib, min_samples=min_samples)


class JointPrototypeV21Forecast(JointPrototypeV21):
    """V21 + `ForecastHead`, for datasets with no declared physics edges
    (e.g. Sielaff -- see `run_sielaff_v2_1.py`'s `prior_edges=None`).
    Same signal K (`k_resid`) and cross-window pairing convention as
    `JointPrototypeV31Forecast` (see that class's and `ForecastHead`'s
    docstrings) -- just layered on V21 instead of V31, since V21 has no
    `typed_head`/declared-edge machinery to begin with. `prior_edges`
    (and therefore `forecast_prior_edges`, absent an explicit override)
    is `None` by construction on these datasets, so the forecast head's
    attention is fully unbiased top-k learned attention, same as
    `edge_head`."""

    _NO_OVERRIDE = object()

    def __init__(self, num_nodes: int, window_size: int, embed_dim: int,
                 num_prototypes: int, forecast_h: int, prior_edges=None,
                 top_k: int = None, node_weights=None, forecast_prior_edges=_NO_OVERRIDE):
        super().__init__(num_nodes, window_size, embed_dim, num_prototypes,
                         prior_edges=prior_edges, top_k=top_k, node_weights=node_weights)
        self.forecast_h = forecast_h
        fe = prior_edges if forecast_prior_edges is self._NO_OVERRIDE else forecast_prior_edges
        self.forecast_head = ForecastHead(num_nodes, window_size, forecast_h, embed_dim,
                                          top_k=top_k, prior_edges=fe)
        self.score_calib_k = ScoreCalibrationHead(num_prototypes, num_nodes)

    def forward(self, x, training_mode=False, x_future=None):
        out = super().forward(x, training_mode=training_mode)
        x_hat_future = self.forecast_head(x)
        out["x_hat_future"] = x_hat_future
        if x_future is not None:
            out["k_resid"] = (x_future - x_hat_future).pow(2).sum(dim=1)
        return out

    def set_k_calibration(self, k_resid_calib: np.ndarray, idx_calib: np.ndarray, min_samples: int = 10):
        self.score_calib_k.set_calibration(k_resid_calib, idx_calib, min_samples=min_samples)


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
        """Returns dict with r_edge [B, num_edges] (raw per-declared-edge
        residuals), r_tilde [B, num_edges] (prototype-conditioned standardized,
        falls back to global stats before `set_calibration()` is called -- fine
        during Stage A training), and resid_phys [B, N] (anomaly-attention-weighted
        standardized physics residual per node; 0 for nodes with no declared
        incoming edge)."""
        r_edge = self.raw_residuals(z, p_star)
        r_tilde = self.standardize(r_edge, idx)

        B = r_edge.shape[0]
        resid_phys = torch.zeros(B, self.num_nodes, device=r_edge.device)
        for target, e_idxs in self.by_target.items():
            r_t_group = r_tilde[:, e_idxs]
            resid_phys[:, target] = r_t_group.clamp(min=0).max(dim=-1).values
        return {"r_edge": r_edge, "r_tilde": r_tilde, "resid_phys": resid_phys}


class JointPrototypeV3(nn.Module):
    """Physics-prior model: V2 + TypedRelationAnomalyHead (declared physics
    edges with typed message functions + prototype-conditioned standardization
    + anomaly attention over declared edges). Requires `prior_edges` and
    `edge_types` -- one per declared (src, dst) pair.

    Detection signals on top of V2's d_proto/d_node/resid_struct:
    r_edge [B, E] (raw typed-edge residuals), r_tilde (standardized),
    resid_phys [B, N] (anomaly-attention-weighted physics residual per node;
    0 for nodes with no declared incoming edge).

    `JointPrototypeV31` extends this with `DeviationCovarianceHead`."""

    def __init__(self, num_nodes: int, window_size: int, embed_dim: int,
                 num_prototypes: int, prior_edges, edge_types,
                 top_k: int = None, node_weights=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.encoder = SharedEncoder(window_size, embed_dim)
        self.memory = JointPrototypeMemory(num_prototypes, num_nodes, embed_dim)
        self.edge_head = TrendGraphAttentionHead(num_nodes, embed_dim, top_k=top_k,
                                                 prior_edges=prior_edges)
        self.typed_head = TypedRelationAnomalyHead(num_nodes, prior_edges, edge_types,
                                                   embed_dim)
        self.decoder = nn.Linear(embed_dim, window_size)  # training-only
        self.register_buffer(
            "node_weights",
            node_weights if node_weights is not None else torch.ones(num_nodes) / num_nodes,
        )

    def forward(self, x, training_mode=False):
        """x: [B, T, N]. Returns dict: z, idx, p_star, p_star_st, d_proto,
        d_node, d_hat, resid_struct, r_edge, r_tilde, resid_phys.
        `training_mode=True` also computes x_hat."""
        B, T, N = x.shape
        assert N == self.num_nodes
        z = self.encoder(x)
        mem_out = self.memory(z, self.node_weights)
        d_hat, resid_struct = self.edge_head(z, mem_out["p_star"])
        typed_out = self.typed_head(z, mem_out["p_star"], mem_out["idx"])
        out = {"z": z, "d_hat": d_hat, "resid_struct": resid_struct, **mem_out, **typed_out}
        if training_mode:
            out["x_hat"] = self.decoder(z).transpose(1, 2)
        return out

    def device_score(self, d_node, resid_struct, resid_phys, lambda_n: float = 1.0,
                     lambda_e: float = 1.0, lambda_t: float = 1.0):
        return (lambda_n * d_node.mean(axis=-1) + lambda_e * resid_struct.mean(axis=-1)
                + lambda_t * resid_phys.mean(axis=-1))


class JointPrototypeV31(JointPrototypeV3):
    """V3 + DeviationCovarianceHead: adds d_mahal [B] (Mahalanobis distance
    of the joint d_node pattern from prototype-conditioned normal).
    Stage-B calibration (`cov_head.set_calibration`) required after training.
    All other outputs identical to JointPrototypeV3."""

    def __init__(self, num_nodes: int, window_size: int, embed_dim: int,
                 num_prototypes: int, prior_edges, edge_types,
                 top_k: int = None, node_weights=None):
        super().__init__(num_nodes, window_size, embed_dim, num_prototypes,
                         prior_edges, edge_types, top_k=top_k,
                         node_weights=node_weights)
        self.cov_head = DeviationCovarianceHead(num_prototypes, num_nodes)
        # Path B (memory/calib-in-prototype-ab.md): per-prototype median/IQR
        # calibration for signals B (d_node) and C (resid_struct), same
        # set_calibration()/min_samples convention as cov_head/typed_head
        # above. Unused unless the calling script's calib_mode="per_prototype"
        # explicitly calls set_score_calibration() and reads these heads'
        # .zscore() instead of the script-level global zscore() -- default
        # (calib_mode="global") behavior is completely unaffected.
        self.score_calib_node = ScoreCalibrationHead(num_prototypes, num_nodes)
        self.score_calib_struct = ScoreCalibrationHead(num_prototypes, num_nodes)

    def forward(self, x, training_mode=False):
        out = super().forward(x, training_mode=training_mode)
        out["d_mahal"] = self.cov_head(out["d_node"], out["idx"])
        return out

    def set_score_calibration(self, d_node_calib: np.ndarray, resid_struct_calib: np.ndarray,
                              idx_calib: np.ndarray, min_samples: int = 10):
        """Stage B, Path B: fit `score_calib_node`/`score_calib_struct` from
        the same held-out calib split (and the same `idx_calib` passed to
        `cov_head`/`typed_head.set_calibration`) -- one call covers both
        signals B and C."""
        self.score_calib_node.set_calibration(d_node_calib, idx_calib, min_samples=min_samples)
        self.score_calib_struct.set_calibration(resid_struct_calib, idx_calib, min_samples=min_samples)


class ForecastHead(nn.Module):
    """New signal K (`shared_forecast_head_proposal.md`): predicts each
    node's OWN future raw values from a GDN-style learned cross-node
    attention over the OTHER nodes' current raw-window encodings -- same
    attention mechanism as `TrendGraphAttentionHead` (top-k by learned
    embedding similarity, declared physics edges add an ADDITIVE learned-
    strength bias to the attention logits, no hard edge restriction), but
    operating in RAW feature space on a fresh small encoder, not on `z`/
    `d = z - p*`. This is deliberate: the anomaly signal here is "how
    wrong was the forecast vs. what actually happened," a temporal
    prediction error, not a same-instant consistency check against a
    prototype -- conflating it with the deviation space the other heads
    use would confound two different axes (see
    `shared_forecast_head_proposal.md`'s "two independent risk axes").

    Forecast target is a genuinely separate FUTURE window's
    non-overlapping tail segment, not an in-window prefix/suffix split:
    the caller is responsible for pairing each input window `x_in`
    (a full window, same as every other head's input) with the raw
    `stride`-length segment of raw time that immediately follows it in
    the SAME source sequence (robo3er: same robot, adjacent window index,
    per `run_robo3er_forecast_v2.py`'s pairing logic) -- see that script's
    docstring for why a hand-picked prefix/suffix split of one window
    (the v1 approach, since replaced) would leak most of the target
    through the window's own overlap with its stride-shifted successor
    and is not a genuine forecast.

    No self-loop (matches `TrendGraphAttentionHead`/`gdn_model.GDN`): a
    node forecasting itself by attending to its own current value would
    be a near-trivial persistence predictor and wouldn't exercise the
    cross-node relationship this signal is meant to test."""

    def __init__(self, num_nodes: int, in_window: int, out_window: int, embed_dim: int,
                 top_k: int = None, prior_edges=None, prior_bias_init: float = 1.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.out_window = out_window
        self.top_k = min(top_k, num_nodes - 1) if top_k is not None else num_nodes - 1
        self.encoder = nn.Linear(in_window, embed_dim)  # raw-space, separate from SharedEncoder
        self.embeddings = nn.Embedding(num_nodes, embed_dim)
        nn.init.uniform_(self.embeddings.weight, -1.0, 1.0)
        self.attn_w = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_a = nn.Linear(2 * embed_dim, 1, bias=False)
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, out_window),
        )

        # Same declared-physics-edge additive-bias mechanism as
        # TrendGraphAttentionHead.prior_mask -- see that class's docstring.
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

    def forward(self, x_in):
        """x_in: [B, in_window, N] raw feature window (prefix). Returns
        x_hat_future [B, out_window, N] (per-node forecast of the
        out_window raw values immediately following x_in)."""
        B, T_in, N = x_in.shape
        w = self.encoder(x_in.transpose(1, 2))  # [B, N, D]
        wf = self.attn_w(w)

        neighbor_idx = self._topk_neighbors()  # [N, top_k], never self
        wf_neighbors = wf[:, neighbor_idx, :]  # [B, N, top_k, D]
        wf_self = wf.unsqueeze(2).expand(-1, -1, neighbor_idx.shape[1], -1)

        logits = self.attn_a(torch.cat([wf_self, wf_neighbors], dim=-1)).squeeze(-1)  # [B, N, top_k]
        prior_bias = self.prior_mask[torch.arange(N, device=x_in.device).unsqueeze(1), neighbor_idx]
        logits = logits + self.prior_bias_strength * prior_bias.unsqueeze(0)

        alpha = F.softmax(F.leaky_relu(logits), dim=-1)
        agg = torch.einsum("bnk,bnkd->bnd", alpha, wf_neighbors)
        agg = F.relu(agg)

        gated = agg * self.embeddings.weight.unsqueeze(0)
        x_hat_future = self.out_proj(gated)  # [B, N, out_window]
        return x_hat_future.transpose(1, 2)  # [B, out_window, N]


class JointPrototypeV31Forecast(JointPrototypeV31):
    """V31 + `ForecastHead`: adds signal K (`k_resid` [B,N], the per-node
    squared error between the forecast head's prediction and the actual
    future). Purely additive on top of V31 -- B/C/E/H are computed
    exactly as in V31, from the unmodified full-window SharedEncoder
    path; the forecast head reads the FULL input window `x` (same input
    every other head sees), never touching `z`/`d`/the memory/edge/typed
    heads. See `ForecastHead`'s docstring and
    `shared_forecast_head_proposal.md` for why this is scoped as an
    independent additive signal rather than a backbone replacement.

    Unlike v1 of this class (in-window prefix/suffix split, since
    replaced), `x_future` is NOT derived from `x` itself -- the caller
    must supply it (the paired next-window's non-overlapping tail
    segment, per `run_robo3er_forecast_v2.py`'s pairing logic). Passing
    `x_future=None` (e.g. for orphan windows with no valid pair, or when
    only `x_hat_future` is needed) skips `k_resid`.

    `forecast_prior_edges` defaults to the SAME edges as `prior_edges`
    (what `edge_head`/`typed_head` use), but can be overridden
    independently -- e.g. `forecast_prior_edges=None` to ablate the
    forecast head's prior bias specifically while leaving B/C/E
    untouched, per `memory/forecast-head-signal-k.md`'s improvement-1
    experiment (does the declared-edge bias actually help the forecast
    head, or is `K`'s 0.803 mean AUROC coming entirely from unbiased
    learned top-k attention?)."""

    _NO_OVERRIDE = object()

    def __init__(self, num_nodes: int, window_size: int, embed_dim: int,
                 num_prototypes: int, prior_edges, edge_types,
                 forecast_h: int, top_k: int = None, node_weights=None,
                 forecast_prior_edges=_NO_OVERRIDE):
        super().__init__(num_nodes, window_size, embed_dim, num_prototypes,
                         prior_edges, edge_types, top_k=top_k, node_weights=node_weights)
        self.forecast_h = forecast_h
        fe = prior_edges if forecast_prior_edges is self._NO_OVERRIDE else forecast_prior_edges
        self.forecast_head = ForecastHead(num_nodes, window_size, forecast_h, embed_dim,
                                          top_k=top_k, prior_edges=fe)
        # Path B: per-prototype median/IQR calibration for signal K (k_resid),
        # same convention as score_calib_node/score_calib_struct on JointPrototypeV31.
        self.score_calib_k = ScoreCalibrationHead(num_prototypes, num_nodes)

    def forward(self, x, training_mode=False, x_future=None):
        """x: [B, T, N] (full input window). x_future: [B, forecast_h, N]
        or None -- the paired next-window's tail segment (raw feature
        space), supplied by the caller (see class docstring). Returns
        everything JointPrototypeV31 returns, plus x_hat_future
        [B, forecast_h, N] and (if x_future is not None) k_resid [B, N]
        (per-node squared forecast error, raw feature space)."""
        out = super().forward(x, training_mode=training_mode)
        x_hat_future = self.forecast_head(x)
        out["x_hat_future"] = x_hat_future
        if x_future is not None:
            out["k_resid"] = (x_future - x_hat_future).pow(2).sum(dim=1)  # [B, N]
        return out

    def set_k_calibration(self, k_resid_calib: np.ndarray, idx_calib: np.ndarray, min_samples: int = 10):
        """Stage B, Path B, for signal K specifically (separate from
        `set_score_calibration` since K needs its own paired calib pass --
        see `run_robo3er_forecast_v2_federated.py`'s `build_pairs`)."""
        self.score_calib_k.set_calibration(k_resid_calib, idx_calib, min_samples=min_samples)
