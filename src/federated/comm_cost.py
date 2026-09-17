"""
Per-round communication-cost accounting, shared across every federated
mode in this project ("ours"/"fedexdnn" codebook alignment,
"fedavg"/"ifcaae" full-model averaging, and the standalone fedpro/fedcpg/
ufedhy baselines). This project simulates federation in a single process
(no real network transport), so there is nothing to literally measure on
the wire -- these helpers instead compute the THEORETICAL payload size
(`numel() * element_size()`, no serialization/compression/quantization
assumed) of whatever tensors a method's own code actually uploads/
downloads each round. That is a proxy, not a measured byte count, but it
is the one basis every method can be compared on fairly, since none of
them do real network I/O here.

Every call site that exchanges something appends a
`{"comm_bytes_up": int, "comm_bytes_down": int}` pair (from
`round_comm_record`) into that round's existing diagnostics dict --
"up" = client(s) -> server, "down" = server -> client(s), summed across
ALL clients for that round (not per-client), so a run's total
communication cost is just `sum(d["comm_bytes_up"] + d["comm_bytes_down"]
for d in diagnostics_log)`.
"""
import torch


def tensor_bytes(t):
    """Bytes `t` would occupy on the wire at its current dtype, no
    compression/quantization assumed. `t` may be a torch.Tensor or
    anything `torch.as_tensor` can wrap (e.g. a Python list/np.ndarray of
    counts)."""
    t = t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
    return t.numel() * t.element_size()


def state_dict_bytes(state_dict, prefixes=None):
    """Sum of `tensor_bytes` over a state_dict-like mapping (str -> Tensor).
    `prefixes`, if given, restricts the sum to keys starting with one of
    them -- pass the SAME `prefixes` tuple a caller uses with
    `federated_memory.fedavg_state_dict` to get the exact subset actually
    exchanged, not the whole model. `prefixes=None` sums every key (the
    "fedavg"/"ifcaae"/full-model-sync case)."""
    total = 0
    for k, v in state_dict.items():
        if prefixes is not None and not k.startswith(tuple(prefixes)):
            continue
        total += tensor_bytes(v)
    return total


def round_comm_record(bytes_up, bytes_down):
    """One round's total communication in bytes, both directions, summed
    across every client -- the two keys every `diagnostics_log` entry in
    this project carries for its communication cost."""
    return {"comm_bytes_up": int(bytes_up), "comm_bytes_down": int(bytes_down)}
