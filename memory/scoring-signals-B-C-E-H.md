---
name: scoring-signals-B-C-E-H
description: Principles and cross-dataset performance of the four anomaly scoring signals B/C/E/H produced by JointPrototypeV31. The definitive reference for which signal to use on which dataset and why.
metadata:
  type: project
---

# Anomaly scoring signals B / C / E / H

All four signals are derived from the same forward pass of `JointPrototypeV31`
(`src/models/joint_prototype_model.py`). The shared backbone:

```
x [B,T,N]  →  SharedEncoder  →  z [B,N,D]
                                    ↓
                          JointPrototypeMemory
                          ├─ idx  [B]        matched prototype index
                          ├─ p*   [B,N,D]    matched prototype embedding
                          └─ d_node [B,N]    ||z_i − p*_i||²  per node
```

Each signal head then reads from this shared output in a different way.

---

## B — node_max

**What it measures:** amplitude deviation of individual nodes.

```
B = max_i  zscore(d_node_i)
```

`d_node_i = ||z_i − p*_i||²` — how far node i's embedding lands from its slot
in the matched prototype. Taking the max over all N nodes picks the single most
deviant node. No edge information is used at all.

**What it can detect:** any fault that causes at least one node's embedding to
shift significantly away from the matched prototype's slot — the "something
changed" baseline.

**What it misses:** faults that change the *relationship between* nodes without
making any single node individually abnormal (e.g. a correlation structure
breaking while all individual magnitudes stay normal).

**Performance (federated+sync, mean AUROC):**

| Dataset | Score | Notes |
|---|---|---|
| Robo3er | 0.945 | tied-best on cable_trapped; stuck = 0.933 (fed), 0.499 (central) |
| Paderborn | 0.888 | weakest of B/C/E/H here; bearing faults need relational signals |
| Sielaff | **0.974** | best or tied-best; fault mechanism is single-node amplitude drift |

---

## C — struct_max

**What it measures:** whether a node's deviation is predictable from its
neighbors' deviations via data-driven graph attention.

```
resid_struct_i = ||d_i − Σ_j α_ij · f(d_j)||²
C = max_i  zscore(resid_struct_i)
```

`TrendGraphAttentionHead` learns attention weights `α_ij` over each node's
top-k nearest embedding neighbors during training. Declared physics edges bias
the attention logits (one learned scalar per declared edge) but do NOT restrict
which neighbors can be attended to — the full graph is reachable. Operating on
`d = z − p*` means the attention is already in deviation space and implicitly
regime-normalised.

**What it can detect:** faults that break *learned structural co-variation*
between nodes — if node i normally follows its neighborhood and stops doing so,
`resid_struct_i` spikes.

**What it misses:** faults where the learned graph simply hasn't encoded the
relevant dependency (e.g. the encoder compressed away the exact signal needed),
or datasets where no inter-node structural coupling exists.

**Performance (federated+sync, mean AUROC):**

| Dataset | Score | Notes |
|---|---|---|
| Robo3er | 0.945 | tied-best; stuck = 0.926 (fed, vs 0.522 central — federation critical) |
| Paderborn | **0.970** | dominant single signal in federated setting |
| Sielaff | **0.976** | best federated signal; Sielaff machines do have structural coupling |

---

## E — phys_max

**What it measures:** whether *declared physics edges* are violated in deviation
space.

```
r_edge(src→dst) = ||d_dst − f_type(d_src)||²
E = max_{target nodes}  zscore(r_edge per incoming edge, max-aggregated)
```

`TypedRelationAnomalyHead` fits a type-specific message function per declared
edge: `Linear` for "proportional" relations, small `MLP` for "nonlinear". It
operates on `d = z − p*`. Per-prototype calibration (μ/σ per prototype per
edge). Aggregation over multiple incoming edges per target is `max` (changed
from softmax 2026-08-16; see [[joint-prototype-scheme-v3]]).

**What it can detect:** faults that specifically break a *declared and
physically verified* cross-node relationship — the targeted physics violation
signal.

**What it misses:** anything not covered by a declared edge. Produces zero for
nodes with no declared incoming edge. Useless (near-random) for fault
mechanisms that don't break mean-value relationships in deviation space (e.g.
robo3er `stuck` AUROC 0.767 federated, 0.519 central).

**When to enable:** only when domain-verified physics edges exist for the
dataset. Set `prior_edges=[]` to disable (E=0 everywhere, equivalent to V2.1).

**Performance (federated+sync, mean AUROC):**

| Dataset | Score | Notes |
|---|---|---|
| Robo3er | 0.686 | weak overall; stuck is the drag (0.767); cable_trapped fine (0.605) |
| Paderborn | **0.977** | high because bearing faults affect vibration/current coupling |
| Sielaff | — | no declared edges; E disabled |

---

## H — cov_mahal

**What it measures:** whether the *joint pattern* of all-node deviations
matches the prototype's normal covariance structure.

```
diff   = d_node − μ*(idx)          [B, N]   per-prototype normal mean
H      = diff @ Σ*(idx)⁻¹ @ diffᵀ  [B]      Mahalanobis distance
```

`DeviationCovarianceHead` has **no trainable parameters**. In Stage B it fits
per-prototype mean `μ*[m]` and inverse covariance `Σ*[m]⁻¹` from the
calibration split, using only calibration windows whose `idx == m`. Falls back
to global statistics for prototypes with fewer than `min_samples` calib windows.
Ridge regularisation `reg·I` ensures invertibility.

H uses `d_node` (same as B) but asks a fundamentally different question: not
"is any single node's magnitude large?" but "does the vector of all N nodes'
deviations together fall inside the normal covariance ellipsoid for this
prototype?"

**What it can detect:** faults that change the *correlation structure* of
deviations without necessarily making individual nodes large — a node that
normally co-varies with others but suddenly doesn't, even if its own `d_node`
is not extreme.

**What it misses:** faults visible only in inter-node *graph-structured*
relationships (C's domain) or in specific declared physics edges (E's domain).
Also sensitive to prototype coverage: if a prototype has few calib windows,
it falls back to global statistics and loses regime-specificity.

**Performance (federated+sync, mean AUROC):**

| Dataset | Score | Notes |
|---|---|---|
| Robo3er | **0.948** | best single signal; stuck = 0.946 — the ONLY signal that rescues stuck |
| Paderborn | 0.961 | strong but below C (0.970) and E (0.977) |
| Sielaff | 0.964 | competitive but below C; small per-prototype calib may limit it |

---

## Full cross-dataset matrix (federated+sync, mean AUROC, 2026-08-17)

All seven derived columns from a single JointPrototypeV31 run per dataset
(F = max(C,E); I = max(B,H); J = max(C,E,H) -- the "everything" combo).
Sielaff has no declared physics edges, so E/F/J are undefined there.

| Signal | Paderborn | Robo3er | Sielaff |
|---|---|---|---|
| B (node_max) | 0.888 | 0.945 | **0.974** |
| C (struct_max) | 0.970 | 0.945 | **0.976** |
| E (phys_max) | 0.977 | 0.686 ⚠️ | — (no edges) |
| F (max C,E) | 0.989 | 0.944 | — |
| H (cov_mahal) | 0.961 | **0.948** | 0.964 |
| I (max B,H) | 0.944 | 0.947 | **0.973** |
| J (max C,E,H) | **0.991** | 0.946 | — |
| **Recommended** | J (0.991) | H (0.948) | C or I (0.973-0.976) |

Robo3er's E column (0.686) is the one clearly weak cell in the whole
matrix -- confirms `current → wheel_vel` physics edges hurt more than
help there (see the E section above and
[[robo3er-explicit-physics-scoring]]). There is no dataset-agnostic best
signal: which column wins is a property of each dataset's fault
mechanisms, not of the model itself (echoed in
[[joint-prototype-scheme-v3]]'s Lineage section).

## Combination rules

J = `max(B, C, E, H)` is the full combination. It is only safe to use when
**all** contributing signals have AUROC > 0.5 for every fault type in the
dataset. Adding a near-random signal via max can invert rankings and reduce
AUROC (robo3er centralized: J=0.725 < H=0.869 because E and B near-random
for `stuck` corrupt the score ordering for that fault).

**Recommended per-dataset scoring:**

| Dataset | Enable E? | Best combination | Federated AUROC |
|---|---|---|---|
| Paderborn | Yes (8 edges verified) | `max(C, E, H)` ≈ J | 0.991 |
| Robo3er | No (E weak for stuck) | `max(B, C, H)` or H alone | 0.948 |
| Sielaff | No (no edges) | `max(B, C)` or C alone | 0.976 |

The combination `max(B, C, H)` — node amplitude, structural coupling,
covariance pattern — is the safe cross-dataset default when physics edge
validity is uncertain.

---

## Why H uniquely rescues robo3er `stuck`

`stuck` means one wheel is locked. The locked wheel's `d_node` stays near zero
because the prototype already models the near-zero-velocity state. B and C
see no deviation amplitude (B≈0.5, C≈0.52 centralized). E is near-random
because `current → wheel_vel` edge has only 1 incoming target, so max=softmax,
and the learned message function in embedding space doesn't capture the raw
kinematic constraint.

H fires because normal motion has correlated deviations across wheel, odom,
and IMU nodes — the arm normally moves all three together. When one wheel is
stuck, the correlation pattern breaks: wheel nodes deviate low while imu/odom
deviate in an uncorrelated direction. The covariance matrix encodes this
expected joint structure; Mahalanobis distance to its inverse catches the break
even when no individual node is extreme.

**Why:** See also [[robo3er-explicit-physics-scoring]] for why explicit OLS
kinematics (which DID catch stuck) was abandoned: the H signal achieves nearly
the same result (0.946 vs 0.619 from explicit) without raw-signal pipelines,
and is already available inside the embedding-space architecture.
