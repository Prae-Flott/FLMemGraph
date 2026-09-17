"""
uFedHy-DisMTSADD (Hao et al., *Information Processing & Management* 62
(2025) 104107 -- PDF at `docs/1-s2.0-S0306457325000494-main.pdf`, code at
github.com/Hjfyoyo/uFedHy-DisMTSADD). Federated-hypernetwork half of the
method (the client-side SC Nor-Transformer target network lives in
`src/models/baseline_models.py::SCNorTransformerModel`/`Hypernetwork`).
Unsupervised, fit-on-healthy -- unlike FedCPG/FedPRO, this baseline DOES
fit this project's usual detection-AUROC convention, so it runs on every
dataset (not robo_fleet-only).

Implements the paper's own simplified hypernetwork update (Eq. (5)-(6),
Algorithm 1): each round, generate a client's weights from the shared
hypernetwork, run ordinary local SGD starting from those weights, then
update ONLY the hypernetwork via one MSE ("distillation") step toward the
post-local-training weights -- not full second-order unrolled backprop
through the local SGD steps. See `Hypernetwork`'s own docstring for why
this is equivalent to the paper's stated gradient rule.

The paper's own anomaly-diagnosis stage (PC-algorithm causal graph +
PageRank root-cause ranking, Sec 4.5) is NOT reproduced here -- localization
instead uses this project's own per-node argmax-of-reconstruction-error
convention, same as every other baseline's localization comparison. This
is a documented scope cut (causal discovery is a materially different,
non-differentiable module), not an oversight.
"""
import numpy as np
import torch
import torch.nn.functional as F

from comm_cost import state_dict_bytes


def hypernet_round_update(hypernet, hyper_optimizer, client_id, target_model, x_in, epochs, device,
                           lr=1e-3, batch_size=256):
    """One client's full round: generate -> local SGD -> hypernetwork
    distillation update (paper Eq. (6)). `target_model`
    (`SCNorTransformerModel`) is trained in place and left holding
    `theta_tilde_i` (the post-local-training weights) on return -- the
    caller scores directly off this `target_model`, exactly like
    `ReconOnlyModel`'s FedAvg/IFCAAE baselines score off theirs.

    Returns `(hyper_loss, n, bytes_down, bytes_up)` -- `bytes_down` is the
    generated `theta_i` (server -> this one client, the hypernetwork never
    leaves the server), `bytes_up` is `theta_tilde_i` sent back for the
    distillation step. Both are ONE client's bytes for this round; the
    caller sums over clients for the round total (same convention as
    `federated_train_eval.run_federated_rounds`)."""
    with torch.no_grad():
        generated = hypernet.generate(client_id)
        theta_i = {name: t.clone() for name, t in generated.items()}
    bytes_down = state_dict_bytes(theta_i)
    target_model.load_state_dict(theta_i, strict=True)

    optimizer = torch.optim.Adam(target_model.parameters(), lr=lr)
    x_t = torch.from_numpy(x_in).to(device)
    target_model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(x_t))
        for start in range(0, len(x_t), batch_size):
            xb = x_t[perm[start:start + batch_size]]
            x_hat = target_model(xb, training_mode=True)["x_hat"]
            loss = F.mse_loss(x_hat, xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    theta_tilde_i = {name: t.detach().clone() for name, t in target_model.state_dict().items()}
    bytes_up = state_dict_bytes(theta_tilde_i)

    generated_grad = hypernet.generate(client_id)
    hyper_loss = sum(
        0.5 * F.mse_loss(generated_grad[name], theta_tilde_i[name], reduction="sum")
        for name in generated_grad
    )
    hyper_optimizer.zero_grad()
    hyper_loss.backward()
    hyper_optimizer.step()
    return float(hyper_loss.item()), len(x_t), bytes_down, bytes_up
