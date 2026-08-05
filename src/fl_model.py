"""
Unified per-client model for the "记忆检索 + 物理验证" design
(`mem_phys_prompt_zh.md`, Sec 2A/2B/3): ONE shared per-node encoder feeds
TWO heads that operate entirely in latent space, plus a training-only
decode head that grounds the latent representation in the real signal.

    x_i (raw window, per sensor node)
      -> f_theta (shared Linear window_size -> D, PatchTST-style channel
         independence: same weights across all N nodes)
      -> z_i in R^D
           |
           +--> MEMORY HEAD (DiscretePrototypicalMemory, see
           |    gdn_memory_model.py -- reused as-is): d_i = distance to
           |    nearest of M learned prototypes. Federated across clients
           |    (federated_memory.py). Zero extra network, pure lookup.
           |
           +--> STRUCTURE HEAD (this file): TopK graph attention over
           |    OTHER nodes' z_j (self excluded -- see warning below),
           |    -> conditional MLP g(.;v_i) -> z_hat_i, r_i = ||z_i - z_hat_i||.
           |    Stays local per client (kinematics/graph structure is
           |    robot-specific in general, though see kinematics.py for
           |    the one relation this project already knows is shared:
           |    differential-drive geometry).
           |
           +--> [TRAINING ONLY] DECODE HEAD: Linear(D -> window_size)
                reconstructs x_i. Grounds z_i in the real signal so the
                encoder can't collapse to a trivial constant (which would
                make BOTH d_i and r_i uselessly small for everything,
                normal or not). Dropped entirely at inference -- scoring
                only ever touches z_i, d_i, r_i (Sec 2A.3's "解码器可裁").

## Why NOT reuse GDN's self-loop convention for the structure head

The original GDN (`test_gdn/gdn_model.py`) includes a self-loop in its
neighbor set because it's a FORECASTING task (predict step t+1 from steps
1..t) -- attending to "myself" means attending to my own PAST, not a
trivial copy of the prediction target. Here there is no time offset: the
structure head predicts z_i from other nodes' CURRENT z_j, in the same
forward pass. If self were included in that neighbor set, attention could
put all its weight on the self-loop and reconstruct z_i by copying it,
making r_i approx 0 for every window regardless of whether the sensor
relationships actually hold -- a trivial, useless "always agrees with
itself" shortcut. The structure head's neighbor set here is therefore
strictly the TopK OTHER nodes, no self-loop -- r_i genuinely measures
"can my current state be predicted from everyone else's," which is the
actual physical-consistency question this design is trying to answer.

## Training losses

    L = L_Pred + beta * L_ME + L_MC + lam * L_struct

  - L_Pred = MSE(decode(z), x)            grounds z (encoder + decode head)
  - L_ME   = MSE(z, sg(z_hat_memory))     encoder commitment (FedPM)
  - L_MC   = MSE(sg(z), z_hat_memory)     codebook commitment (FedPM, only
             term that updates the memory codebook, no EMA)
  - L_struct = MSE(sg(z), z_hat_struct)   structure head chases z one-way
             (stop-gradient on z) -- a design choice made here because the
             source doc (Sec 7) explicitly leaves "structure head training
             direction" as an undecided team variable. One-way (like
             L_MC) keeps the encoder's shaping controlled only by L_Pred
             and L_ME, so the structure head can't fight the memory
             commitment loss for control of z's geometry -- it's a pure
             downstream consumer, not a co-shaper, of the representation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gdn_memory_model import DiscretePrototypicalMemory  # noqa: E402


class SharedEncoder(nn.Module):
    """One Linear(window_size -> D), shared across every node/channel."""

    def __init__(self, window_size: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(window_size, embed_dim)

    def forward(self, x):  # x: [B, T, N] -> z: [B, N, D]
        return self.proj(x.transpose(1, 2))


class StructureHead(nn.Module):
    """Latent-space graph-attention relation head. Learns node embeddings
    v_i (for TopK neighbor selection, same mechanism as GDN), attends over
    the TopK OTHER nodes' z_j (no self-loop, see module docstring), then a
    v_i-conditioned MLP predicts z_hat_i. r_i = ||z_i - z_hat_i||."""

    def __init__(self, num_nodes: int, embed_dim: int, top_k: int = 15):
        super().__init__()
        self.num_nodes = num_nodes
        self.top_k = min(top_k, num_nodes - 1)
        self.embeddings = nn.Embedding(num_nodes, embed_dim)
        nn.init.uniform_(self.embeddings.weight, -1.0, 1.0)
        self.attn_w = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_a = nn.Linear(2 * embed_dim, 1, bias=False)
        self.g = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def _topk_neighbors(self):
        v = F.normalize(self.embeddings.weight, dim=1)
        sim = v @ v.T
        sim.fill_diagonal_(-float("inf"))  # excluded permanently, no self-loop added back
        _, topk_idx = sim.topk(self.top_k, dim=1)
        return topk_idx  # [N, top_k], never includes i itself

    def forward(self, z):
        """z: [B, N, D]. Returns z_hat [B, N, D] and r [B, N]."""
        B, N, D = z.shape
        wz = self.attn_w(z)  # [B, N, D]

        neighbor_idx = self._topk_neighbors()  # [N, top_k], no self
        wz_neighbors = wz[:, neighbor_idx, :]  # [B, N, top_k, D]
        wz_self = wz.unsqueeze(2).expand(-1, -1, neighbor_idx.shape[1], -1)

        logits = self.attn_a(torch.cat([wz_self, wz_neighbors], dim=-1)).squeeze(-1)  # [B, N, top_k]
        alpha = F.softmax(F.leaky_relu(logits), dim=-1)

        agg = torch.einsum("bnk,bnkd->bnd", alpha, wz_neighbors)  # [B, N, D]
        agg = F.relu(agg)

        gated = agg * self.embeddings.weight.unsqueeze(0)
        z_hat = self.g(gated)  # [B, N, D]

        r = (z - z_hat).pow(2).sum(dim=-1)  # [B, N]
        return z_hat, r


class FLGDNMemory(nn.Module):
    """Encoder + memory head + structure head + (training-only) decode
    head, all in one module for convenience. See module docstring for the
    data flow and loss terms."""

    def __init__(self, num_nodes: int, window_size: int, embed_dim: int = 64,
                 top_k: int = 15, num_prototypes: int = 256):
        super().__init__()
        self.num_nodes = num_nodes
        self.encoder = SharedEncoder(window_size, embed_dim)
        self.memory = DiscretePrototypicalMemory(num_prototypes, embed_dim)
        self.structure = StructureHead(num_nodes, embed_dim, top_k)
        self.decoder = nn.Linear(embed_dim, window_size)  # training-only

    def forward(self, x, training_mode=False):
        """x: [B, T, N]. Returns a dict of everything scoring/training needs.
        `training_mode=True` also runs the decode head (dropped otherwise,
        matching the "decoder prunable at inference" design)."""
        B, T, N = x.shape
        assert N == self.num_nodes

        z = self.encoder(x)  # [B, N, D]

        z_st, z_q, _, d = self.memory(z)  # straight-through, quantized, idx, quant_dist
        z_hat, r = self.structure(z)

        out = {"z": z, "z_q": z_q, "d": d, "z_hat": z_hat, "r": r}
        if training_mode:
            x_hat = self.decoder(z).transpose(1, 2)  # [B, N, T] -> [B, T, N]
            out["x_hat"] = x_hat
        return out

    def load_memory(self, prototypes: torch.Tensor):
        """Overwrite this client's codebook with a server-broadcast
        P_G,n (federated_memory.py). Shape must match [M, D]."""
        with torch.no_grad():
            self.memory.codebook.copy_(prototypes.to(self.memory.codebook.device))
        self.memory.usage_count.zero_()
