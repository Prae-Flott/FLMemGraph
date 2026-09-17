"""
Minimal, own-architecture models for the three FL baselines (FedAvg,
IFCAAE, Fed-ExDNN) -- see `docs/baseline_selection_for_benchmark.md` and
`benchmark/baselines/registry.py`. Each baseline gets ONLY the machinery
its own algorithm actually specifies, deliberately excluding this
project's relational (GDN edge_head/typed_head) and forecast (K)
machinery, so a baseline's score can't accidentally inherit a signal its
own algorithm was never designed to have:

- **FedAvg** and **IFCAAE** (`ReconOnlyModel`, wrapping `ConvAutoEncoder`):
  plain reconstruction, no memory, no relational signal, no forecast.
  FedAvg = one global model (`run_federated_rounds(mode="fedavg")`);
  IFCAAE = `num_clusters` clustered global models
  (`mode="ifcaae"`) -- identical model/training/scoring for both, isolating
  "does clustering help over plain FedAvg" as the only variable between
  them.
- **Fed-ExDNN** (`ExemplarOnlyModel`): local exemplar memory only (reuses
  `JointPrototypeMemory` unchanged, this project's own discrete-memory
  submodule -- Fed-ExDNN's local exemplar IS what that submodule already
  is), no relational signal, no forecast. Server-side alignment is
  `federated_memory.align_and_split_fedcc` (`mode="fedexdnn")`) instead of
  our cosine+BFS `align_and_split`.

Both model classes expose `forward(x, training_mode=False) -> dict with
"x_hat"` whenever `training_mode=True` -- the interface
`federated_train_eval.recon_eval_loss`/`run_federated_rounds(mode="ifcaae")`
already expect, so `run_federated_rounds` itself needs no changes for any
of the three baselines (confirmed model-agnostic).
"""
import torch
import torch.nn as nn

from joint_prototype_model import SharedEncoder, JointPrototypeMemory  # noqa: E402


class ConvAutoEncoder(nn.Module):
    """1D-conv autoencoder over windowed multivariate time series, ported
    verbatim from `archive/src/models/conv_autoencoder.py` (itself
    migrated from `~/Projects/FL-bench`'s `src/utils/models.ConvAutoEncoder`).
    Pure reconstruction -- no VQ memory, no relational/forecast anything."""
    conv_channels = [64, 128, 256]
    kernel_sizes = [3, 3, 3]

    def __init__(self, num_nodes: int, window_size: int):
        super().__init__()
        encoder_layers = []
        in_channels = num_nodes
        for i, (out_channels, kernel_size) in enumerate(zip(self.conv_channels, self.kernel_sizes)):
            encoder_layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2) if i < len(self.conv_channels) - 1 else nn.AdaptiveMaxPool1d(8),
            ]
            in_channels = out_channels
        self.encoder = nn.Sequential(*encoder_layers)

        reversed_channels = self.conv_channels[::-1]
        decode_out_channels = reversed_channels[1:] + [num_nodes]
        decoder_layers = []
        for i, (out_channels, kernel_size) in enumerate(zip(decode_out_channels, self.kernel_sizes[::-1])):
            if i < len(reversed_channels) - 1:
                decoder_layers += [
                    nn.ConvTranspose1d(reversed_channels[i], out_channels, kernel_size,
                                        stride=2, padding=kernel_size // 2, output_padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                ]
            else:
                decoder_layers.append(
                    nn.ConvTranspose1d(reversed_channels[i], out_channels, kernel_size, padding=kernel_size // 2)
                )
        self.decoder = nn.Sequential(*decoder_layers)
        self.size_adjuster = nn.AdaptiveAvgPool1d(window_size)

    def forward(self, x):  # x: [B, T, N] -> recon [B, T, N]
        sequence = x.transpose(1, 2)  # [B, N, T]
        encoded = self.encoder(sequence)
        decoded = self.decoder(encoded)
        decoded = self.size_adjuster(decoded)
        return decoded.transpose(1, 2)  # [B, T, N]


class ReconOnlyModel(nn.Module):
    """Thin adapter giving `ConvAutoEncoder` the
    `forward(x, training_mode=...) -> {"x_hat": ...}` dict interface
    `federated_train_eval.recon_eval_loss`/`train_local_recon` expect, so it's a
    drop-in for `run_federated_rounds(mode="fedavg"|"ifcaae")` unchanged.
    Used identically by BOTH FedAvg and IFCAAE (see module docstring)."""

    def __init__(self, num_nodes: int, window_size: int):
        super().__init__()
        self.ae = ConvAutoEncoder(num_nodes, window_size)

    def forward(self, x, training_mode=False):
        return {"x_hat": self.ae(x)}


class FedPROModel(nn.Module):
    """The 'off-the-shelf FL classifier' `w_i = (h_i, f_i)` FedPRO wraps
    (Zhou et al., IEEE TC 2026, "Prototype Retrieval-Augmented Federated
    Learning System for Robust Intrusion Detection" -- PDF at
    `docs/Prototype_Retrieval-Augmented_Federated_Learning_System_for_Robust_Intrusion_Detection.pdf`,
    official code `github.com/zza234s/FedPRO`). Encoder `h_i`: flattens
    the `[T, N]` window into a single vector and maps it to one `embed_dim`
    embedding (unlike this project's own per-node `SharedEncoder` --
    FedPRO's embedding is one vector per SAMPLE, not per node, since its
    prototypes are per-CLASS not per-operating-regime). Classifier `f_i`:
    a plain linear head to `num_classes` logits, trained end-to-end via
    plain FedAvg (`run_federated_rounds(mode="fedavg")`, cross-entropy --
    see `benchmark/FedPRO/fedpro.py::train_local_classifier`). Robo_fleet
    only: FedPRO needs multiple KNOWN classes (Normal + each fault type)
    to discriminate between at both train and test time, which the
    project's usual fit-on-healthy convention (only ever sees "normal"
    during training) cannot supply -- see `fedpro.py`'s module docstring
    for the full rationale.

    After FedAvg training, `self.encoder` is frozen and reused unchanged
    by the prototype memory bank building / retrieval stages in
    `fedpro.py` -- `embed(x)` exposes just the encoder half for that."""

    def __init__(self, num_nodes: int, window_size: int, embed_dim: int, num_classes: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 2
        in_dim = num_nodes * window_size
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def embed(self, x):  # x: [B, T, N] -> z: [B, embed_dim]
        return self.encoder(x.reshape(x.shape[0], -1))

    def forward(self, x):  # x: [B, T, N] -> logits [B, num_classes]
        return self.classifier(self.embed(x))


class ExemplarOnlyModel(nn.Module):
    """Fed-ExDNN's local model: `SharedEncoder` + `JointPrototypeMemory`
    (this project's own discrete-memory submodule, reused unchanged -- a
    local VQ-VAE-style exemplar dictionary IS what Fed-ExDNN's local
    exemplar learning already is, per
    `docs/baseline_selection_for_benchmark.md` Sec 3.1) + a plain
    `nn.Linear` decoder for the reconstruction training signal. Exactly
    `JointPrototypeV2` (`joint_prototype_model.py`) MINUS its `edge_head`
    (`TrendGraphAttentionHead`) -- Fed-ExDNN has no relational/graph
    signal, only local exemplar memory. Detection signal: `d_node`
    (per-node deviation from the matched exemplar), scored the same
    GDN-style way as `ReconOnlyModel`'s reconstruction error (see
    `federated_train_eval.gdn_score`) -- deliberately NOT this project's own tuned
    `two_stage_group_score`/prototype-conditioned calibration, since those
    are OUR design choices, not Fed-ExDNN's."""

    def __init__(self, num_nodes: int, window_size: int, embed_dim: int, num_prototypes: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.encoder = SharedEncoder(window_size, embed_dim)
        self.memory = JointPrototypeMemory(num_prototypes, num_nodes, embed_dim)
        self.decoder = nn.Linear(embed_dim, window_size)

    def forward(self, x, training_mode=False):
        B, T, N = x.shape
        assert N == self.num_nodes
        z = self.encoder(x)
        mem_out = self.memory(z)
        out = {"z": z, **mem_out}
        if training_mode:
            out["x_hat"] = self.decoder(z).transpose(1, 2)
        return out


class FedCPGModel(nn.Module):
    """FedCPG (Li et al., *Computers in Industry* 164 (2025) 104180,
    "A class prototype guided personalized lightweight federated learning
    framework for cross-factory fault detection" -- PDF at
    `docs/1-s2.0-S0166361524001088-main.pdf`). Model-decoupled personalized
    FL: a backbone `encoder` (FedAvg'd across clients every round -- the
    paper's Eq. (2)) mapped through a small projection `proj` to get the
    representation `r(x)` the paper's class-prototype losses (Eq. (12),
    (13)) operate on, plus a personalized `head` classifier that is NEVER
    aggregated (stays local to each factory/client, paper Sec. 3.2 Step 3).
    Own-minimal-architecture, like every other baseline here: the paper's
    own backbone is a wide-kernel-CNN + lightweight-MLP feature extractor
    (`light-WMLP`, Fig. 4) purpose-built for raw 1D vibration signals; we
    substitute this project's own flatten-then-MLP encoder (same one
    `FedPROModel` uses) since our windows are already multivariate
    feature-group tensors, not raw single-channel signal -- the paper's
    OWN mechanism (backbone/head split + dual class-prototype contrastive
    losses, `benchmark/FedCPG/fedcpg.py`) is reproduced faithfully; only the
    backbone's internal layers are swapped for one consistent with this
    project's other baselines' encoders."""

    def __init__(self, num_nodes: int, window_size: int, embed_dim: int, proj_dim: int, num_classes: int,
                 hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 2
        in_dim = num_nodes * window_size
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embed_dim),
        )
        self.proj = nn.Linear(embed_dim, proj_dim)  # paper's f_r projection network -> r(x)
        self.head = nn.Linear(embed_dim, num_classes)  # personalized, never FedAvg'd

    def represent(self, x):  # x: [B, T, N] -> r(x): [B, proj_dim], paper Eq. (12)/(13)'s embedding space
        z = self.encoder(x.reshape(x.shape[0], -1))
        return self.proj(z), z

    def forward(self, x):  # x: [B, T, N] -> logits [B, num_classes]
        _, z = self.represent(x)
        return self.head(z)


class SCNorTransformerModel(nn.Module):
    """SC Nor-Transformer (Hao et al., *Information Processing & Management*
    62 (2025) 104107, "Effectively detecting and diagnosing distributed
    multivariate time series anomalies via Unsupervised Federated
    Hypernetwork" (uFedHy-DisMTSADD) -- PDF at
    `docs/1-s2.0-S0306457325000494-main.pdf`, code at
    github.com/Hjfyoyo/uFedHy-DisMTSADD). This is the CLIENT-SIDE target
    network the paper's federated hypernetwork (`Hypernetwork` below)
    generates weights for every round (paper Fig. 3): per-window series
    NORMALIZATION (Eq. (7)-(9), z-score over the time axis, undone at the
    output by Eq. (10)'s de-normalization) + series CONVERSION embedding
    (Eq. (11), Fig. 4 -- unlike a vanilla Transformer's per-timestep
    tokens, each of the `num_nodes` channels' whole window becomes ONE
    token, i.e. variate-as-token) + an encoder-only Transformer block
    (Eq. (12)-(15): multi-head self-attention over the channel tokens,
    LayerNorm, feed-forward) + a feature-projection layer back to a
    per-channel reconstruction. Own-minimal-architecture simplification:
    the paper defaults to 3 stacked encoder layers; this uses 1, since the
    number of encoder layers multiplies the hypernetwork's own per-tensor
    output-head count and this project keeps the hypernetwork small enough
    to train in the same compute budget as the other baselines -- a
    documented fidelity gap, not an oversight.

    `forward(x, training_mode=...) -> {"x_hat": ...}`, the same dict
    interface `per_node_recon_error`/`gdn_score` (and `ReconOnlyModel`)
    already expect, so scoring is unchanged from FedAvg/IFCAAE; only WHERE
    a client's weights come from each round (hypernetwork generation,
    `benchmark/uFedHy-DisMTSADD/ufedhy.py`, instead of FedAvg) differs."""

    def __init__(self, num_nodes: int, window_size: int, embed_dim: int, n_heads: int = 2):
        super().__init__()
        self.num_nodes, self.window_size = num_nodes, window_size
        self.token_embed = nn.Linear(window_size, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.ReLU(),
                                  nn.Linear(embed_dim * 4, embed_dim))
        self.ln2 = nn.LayerNorm(embed_dim)
        self.proj_out = nn.Linear(embed_dim, window_size)

    def forward(self, x, training_mode=False):  # x: [B, T, N]
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True).clamp_min(1e-6)
        x_norm = (x - mu) / sigma
        tokens_in = x_norm.transpose(1, 2)  # [B, N, T] -- one token per channel
        tokens = self.token_embed(tokens_in)  # [B, N, D]
        attn_out, _ = self.attn(tokens, tokens, tokens)
        z = self.ln1(tokens + attn_out)
        z = self.ln2(z + self.ffn(z))
        recon_norm = self.proj_out(z).transpose(1, 2)  # [B, T, N]
        x_hat = recon_norm * sigma + mu  # series de-normalization, Eq. (10)
        return {"x_hat": x_hat}


class Hypernetwork(nn.Module):
    """Federated hypernetwork (pFedHN, Shamsian et al., ICML 2021 --
    uFedHy-DisMTSADD's own cited foundation, Sec 2.3/4.3 and Eq. (5)-(6)):
    a per-client learnable embedding `e_i` fed through a shared trunk MLP
    and one linear output head per target-model parameter tensor, each
    sized to that tensor's own `numel()` and reshaped back to its shape.
    `generate(client_id) -> state_dict` for `SCNorTransformerModel`.

    Training follows the paper's own simplified update rule (Eq. (6):
    `L̃_i(mu) = 0.5 * ||theta_tilde_i - H(mu)||^2`, exactly Algorithm 1's
    `Delta_mu = -lr * (grad_mu theta_i)^T Delta_theta_i` since that is the
    gradient of this MSE distillation loss) -- NOT full second-order
    unrolled backprop through the local SGD steps: generate theta_i =
    H(mu) once (detached) to seed a round of ordinary local SGD, then
    after local training treat the resulting theta_tilde_i as a fixed
    target and take ONE more hypernetwork forward pass (this time WITH
    gradient) to minimize its distance to that target -- see
    `benchmark/uFedHy-DisMTSADD/ufedhy.py::hypernet_round_update`."""

    def __init__(self, num_clients: int, target_shapes: dict, embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.target_shapes = target_shapes
        self.client_embed = nn.Embedding(num_clients, embed_dim)
        self.trunk = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU())
        self.heads = nn.ModuleDict({
            name.replace(".", "__"): nn.Linear(hidden_dim, shape.numel())
            for name, shape in target_shapes.items()
        })

    def generate(self, client_id: int):
        h = self.trunk(self.client_embed(torch.tensor([client_id], device=self.client_embed.weight.device)))
        return {
            name: self.heads[name.replace(".", "__")](h).reshape(shape)
            for name, shape in self.target_shapes.items()
        }
