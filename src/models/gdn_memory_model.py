"""
Discrete prototypical memory added to GDN, adapted from FedPM (Deng,
Liu, Niu, Chen, Sun, Wu, Long, Liang, "Discrete Prototypical Memories for
Federated Time Series Foundation Models", arXiv:2604.04475, 2026,
https://arxiv.org/abs/2604.04475).

FedPM's core idea: time-series semantics recur as a small number of
discrete regimes rather than a smooth continuous latent space, so it
quantizes each patch's latent representation against a learned codebook
of M prototype vectors (VQ-VAE-style hard nearest-neighbor lookup, no
EMA -- the codebook is updated purely by a gradient commitment loss)
instead of letting the model represent it as an arbitrary continuous
vector. Adapted here for two purposes beyond the paper's original
federated-forecasting motivation:

1. If the codebook is trained only on normal data (matching this
   project's AE/GDN convention throughout, see
   memory/robo3er-anomaly-detection-approaches.md), it becomes a
   dictionary of "normal recurring dynamical regimes." An anomalous
   window whose representation doesn't resemble any learned normal
   regime sits far from every prototype -- this quantization distance is
   a NEW anomaly signal, independent of and additional to GDN's existing
   prediction-error score.
2. FedPM's federated cross-domain memory alignment (cosine-similarity
   clustering into shared prototypes + utility-diversity-scored
   personalized ones) is NOT implemented here -- this file only adds the
   memory module to the centralized (non-FL) GDN pipeline already in
   this folder, to answer "does memory help detection at all" first. The
   federated aggregation is flagged as a natural next step once this one
   is validated (see the multi-robot design discussion earlier in this
   conversation), not attempted in this pass.

## Adaptation to GDN

FedPM quantizes patch-level encoder latents between an encoder and
decoder. GDN has no explicit patch encoder, but its graph-attention
aggregation output `z_i` (one d-dim vector per sensor node -- see
`test_gdn/gdn_model.py`'s `forward`, right after `z = F.relu(z)` and
before the elementwise gate with the node embedding) plays the same role:
it's the representation the rest of the network turns into a prediction.
This module quantizes THAT vector against a single codebook shared across
every sensor node and every batch element (matching FedPM's per-domain
global memory shared across all patches of one domain -- this project's
centralized model is one domain). `embed_dim=64` already matches FedPM's
own D=64, so no shape changes ripple through the rest of GDN.

## Losses (matches the paper's three-term objective, no EMA)

    L = L_Pred + beta * L_ME + L_MC

  - L_Pred: the existing forecasting loss (unchanged)
  - L_ME = ||z - sg(z_hat)||^2   (encoder commitment: pulls z toward its
    nearest prototype; codebook frozen for this term)
  - L_MC = ||sg(z) - z_hat||^2   (codebook commitment: pulls the chosen
    prototype toward z; this is the ONLY term that updates the codebook,
    no EMA, matching the paper)

A straight-through estimator (`z + (z_hat - z).detach()`) is used in the
forward pass so L_Pred's gradient still reaches the encoder through the
otherwise non-differentiable argmin lookup -- standard VQ-VAE mechanics,
necessary for L_Pred to train the encoder at all through a hard
quantization step, though not spelled out explicitly in the paper text
available to this project.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscretePrototypicalMemory(nn.Module):
    """Shared VQ-VAE-style codebook: M prototypes of dim D, hard
    nearest-neighbor quantization with a straight-through estimator."""

    def __init__(self, num_prototypes: int = 256, dim: int = 64):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.codebook = nn.Parameter(torch.randn(num_prototypes, dim) * 0.02)
        self.register_buffer("usage_count", torch.zeros(num_prototypes))

    def forward(self, z):
        """z: [..., D]. Returns (z_straight_through, z_quantized,
        prototype_idx, quant_dist) -- quant_dist and prototype_idx have
        z's shape with the last (D) dim reduced."""
        shape = z.shape
        flat = z.reshape(-1, shape[-1])  # [B*N, D]

        dist = torch.cdist(flat, self.codebook)  # [B*N, M]
        idx = dist.argmin(dim=1)  # [B*N]
        z_q = self.codebook[idx]  # [B*N, D]

        if self.training:
            with torch.no_grad():
                self.usage_count += torch.bincount(idx, minlength=self.num_prototypes).float()

        quant_dist = (flat - z_q).pow(2).sum(dim=1)  # [B*N] squared L2 dist to chosen prototype

        z_q = z_q.reshape(shape)
        quant_dist = quant_dist.reshape(shape[:-1])
        idx = idx.reshape(shape[:-1])

        z_st = z + (z_q - z).detach()  # straight-through estimator
        return z_st, z_q, idx, quant_dist

    def commitment_losses(self, z, z_q):
        l_me = F.mse_loss(z, z_q.detach())
        l_mc = F.mse_loss(z.detach(), z_q)
        return l_me, l_mc

    def codebook_utilization(self):
        """Fraction of prototypes used at least once since the last reset
        -- a cheap collapse check (VQ-VAE codebooks can collapse to using
        only a handful of codes)."""
        return float((self.usage_count > 0).float().mean())


class GDNMemory(nn.Module):
    """GDN (see test_gdn/gdn_model.py) with a DiscretePrototypicalMemory
    inserted right after the graph-attention aggregation, before the
    elementwise gate with the node embedding. Architecturally identical
    to GDN otherwise -- same embeddings, feature extractor, attention,
    output MLP -- so any AUROC delta isolates the memory module's effect."""

    def __init__(self, num_nodes: int, window_size: int, embed_dim: int = 64,
                 top_k: int = 15, num_prototypes: int = 256):
        super().__init__()
        self.num_nodes = num_nodes
        self.top_k = min(top_k, num_nodes - 1)
        self.embed_dim = embed_dim

        self.embeddings = nn.Embedding(num_nodes, embed_dim)
        nn.init.uniform_(self.embeddings.weight, -1.0, 1.0)

        self.feature_extractor = nn.Linear(window_size, embed_dim)
        self.attn_w = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_a = nn.Linear(2 * embed_dim, 1, bias=False)

        self.memory = DiscretePrototypicalMemory(num_prototypes, embed_dim)

        self.out_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def _topk_neighbors(self):
        v = self.embeddings.weight
        v_norm = F.normalize(v, dim=1)
        sim = v_norm @ v_norm.T
        sim.fill_diagonal_(-float("inf"))
        _, topk_idx = sim.topk(self.top_k, dim=1)
        return topk_idx

    def forward(self, x, return_memory_diagnostics=False):
        """x: [B, window_size, N]. Returns predicted next-step values
        [B, N] by default; with return_memory_diagnostics=True also
        returns (z, z_q, quant_dist) for computing commitment losses and
        the quantization-distance anomaly signal."""
        B, T, N = x.shape
        assert N == self.num_nodes

        f = self.feature_extractor(x.transpose(1, 2))  # [B, N, d]
        wf = self.attn_w(f)

        topk_idx = self._topk_neighbors()
        self_idx = torch.arange(N, device=x.device).unsqueeze(1)
        neighbor_idx = torch.cat([self_idx, topk_idx], dim=1)

        wf_neighbors = wf[:, neighbor_idx, :]
        wf_self = wf.unsqueeze(2).expand(-1, -1, neighbor_idx.shape[1], -1)

        logits = self.attn_a(torch.cat([wf_self, wf_neighbors], dim=-1)).squeeze(-1)
        alpha = F.softmax(F.leaky_relu(logits), dim=-1)

        z = torch.einsum("bnk,bnkd->bnd", alpha, wf_neighbors)  # [B, N, d]
        z = F.relu(z)

        z_st, z_q, _, quant_dist = self.memory(z)  # [B, N, d], [B, N, d], -, [B, N]

        gated = z_st * self.embeddings.weight.unsqueeze(0)
        out = self.out_mlp(gated).squeeze(-1)  # [B, N]

        if return_memory_diagnostics:
            return out, z, z_q, quant_dist
        return out

    def graph_summary(self, node_names, focus_names, k=5):
        idx_of = {name: i for i, name in enumerate(node_names)}
        with torch.no_grad():
            v = F.normalize(self.embeddings.weight, dim=1)
            sim = v @ v.T
            sim.fill_diagonal_(-float("inf"))
        lines = []
        for name in focus_names:
            i = idx_of[name]
            top_sim, top_idx = sim[i].topk(k)
            neighbors = ", ".join(
                f"{node_names[j]}({s:.2f})" for j, s in zip(top_idx.tolist(), top_sim.tolist())
            )
            lines.append(f"{name} -> {neighbors}")
        return "\n".join(lines)
