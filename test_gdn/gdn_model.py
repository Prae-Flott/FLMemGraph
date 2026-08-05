"""
GDN (Deng & Hooi, "Graph Neural Network-Based Anomaly Detection in
Multivariate Time Series", AAAI 2021, https://arxiv.org/abs/2106.06947).

Unlike our ConvAutoEncoder (reconstructs a whole window at once), GDN is a
FORECASTING model: given the past `window_size` steps of every sensor, it
predicts the NEXT step's value for every sensor, using a learned sensor
graph so each sensor's prediction is conditioned on its most-similar other
sensors (found via cosine similarity of learned embeddings, not a fixed
prior graph) instead of either "all 71 features" (our pooled ConvAE) or
"only its own hand-picked subsystem" (our MoE experiment) -- the graph is
learned from data, per sensor, rather than us drawing the boundary by hand.

Architecture (single graph-attention hop, matching the paper's core
mechanism):
  1. Node embedding v_i in R^d, one per sensor -- purely learned, not tied
     to sensor semantics.
  2. Graph structure: for each node i, its neighbors are the top-k other
     nodes by cosine similarity of (v_i, v_j). Recomputed every forward
     pass from the current embeddings, so the graph itself evolves during
     training as embeddings update.
  3. Feature extractor: each node's own past window (length w) is mapped
     to a d-dim feature f_i via a shared Linear(w -> d).
  4. Graph attention: node i attends over {i} union its top-k neighbors,
     attention logits from concatenated (W f_i, W f_j), softmax-normalized,
     producing an aggregated d-dim representation z_i.
  5. Output: predicted next value s_hat_i = MLP(v_i (*) z_i) (elementwise
     product of the node's own embedding with its aggregated
     representation, then a small shared 2-layer MLP to a scalar) -- this
     lets the same MLP behave differently per node depending on that
     node's own learned embedding, without needing per-node parameters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GDN(nn.Module):
    def __init__(self, num_nodes: int, window_size: int, embed_dim: int = 64, top_k: int = 15):
        super().__init__()
        self.num_nodes = num_nodes
        self.top_k = min(top_k, num_nodes - 1)
        self.embed_dim = embed_dim

        self.embeddings = nn.Embedding(num_nodes, embed_dim)
        nn.init.uniform_(self.embeddings.weight, -1.0, 1.0)

        self.feature_extractor = nn.Linear(window_size, embed_dim)
        self.attn_w = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_a = nn.Linear(2 * embed_dim, 1, bias=False)

        self.out_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def _topk_neighbors(self):
        v = self.embeddings.weight  # [N, d]
        v_norm = F.normalize(v, dim=1)
        sim = v_norm @ v_norm.T  # [N, N] cosine similarity
        sim.fill_diagonal_(-float("inf"))  # exclude self, self-loop is added explicitly below
        _, topk_idx = sim.topk(self.top_k, dim=1)  # [N, top_k]
        return topk_idx

    def forward(self, x):
        """x: [B, window_size, N] past values of every node. Returns
        predicted next-step values, shape [B, N]."""
        B, T, N = x.shape
        assert N == self.num_nodes

        f = self.feature_extractor(x.transpose(1, 2))  # [B, N, d]
        wf = self.attn_w(f)  # [B, N, d]

        topk_idx = self._topk_neighbors()  # [N, top_k]
        self_idx = torch.arange(N, device=x.device).unsqueeze(1)  # [N, 1]
        neighbor_idx = torch.cat([self_idx, topk_idx], dim=1)  # [N, 1+top_k], self-loop included

        # gather neighbor features for every node: [B, N, 1+top_k, d]
        wf_neighbors = wf[:, neighbor_idx, :]
        wf_self = wf.unsqueeze(2).expand(-1, -1, neighbor_idx.shape[1], -1)

        logits = self.attn_a(torch.cat([wf_self, wf_neighbors], dim=-1)).squeeze(-1)  # [B, N, 1+top_k]
        alpha = F.softmax(F.leaky_relu(logits), dim=-1)

        z = torch.einsum("bnk,bnkd->bnd", alpha, wf_neighbors)  # [B, N, d]
        z = F.relu(z)

        gated = z * self.embeddings.weight.unsqueeze(0)  # [B, N, d], v_i (*) z_i
        out = self.out_mlp(gated).squeeze(-1)  # [B, N]
        return out

    def graph_summary(self, node_names, focus_names, k=5):
        """Human-readable top-k learned neighbors for a few named nodes."""
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
