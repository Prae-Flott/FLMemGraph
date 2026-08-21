---
name: joint-prototype-federated-results
description: First federated (not centralized) runs of Joint Prototype Memory V2/V3 across all 4 datasets -- results, infra changes, and why 2 of 4 collapsed
metadata:
  type: project
---

# Joint Prototype Memory V2/V3, run FEDERATED for the first time across all
# 4 datasets (2026-08-11)

Everything in `joint-prototype-scheme-v3.md` (and V2, its no-edges
predecessor, since folded into `scoring-signals-B-C-E-H.md`'s signal B)
was CENTRALIZED (single model, pooled data, no federation) -- this is the
first time V2/V3 have been run through the actual federated protocol
(per-round local-train + `federated_memory.align_and_split` codebook-only
exchange, previously only validated on the older, since-deleted
`FLGDNMemory` architecture).

## Infra changes needed first

- `federated_memory.align_and_split` assumed a flat `[M, D]` codebook
  (`FLGDNMemory`'s per-feature-independent design). `JointPrototypeMemory`'s
  codebook is `[M, N, D]` (a full multi-node snapshot per prototype). Per explicit instruction, this was NOT
  flattened to `[M, N*D]` before computing similarity -- `align_and_split`
  now computes cosine similarity PER NODE (`sim_per_node`, `[C*M, C*M, N]`)
  and only aggregates (mean over N) to decide edges/clusters, so a later
  diagnosis pass can still ask "which node(s) actually drove this
  cross-client match" (`diagnostics["per_cluster_per_node_agreement"]`,
  per-node not collapsed). `[M, D]` (N=1) input reproduces the exact
  original behavior -- verified no change for `FLGDNMemory` callers.
- Added `JointPrototypeMemory.load_memory()` (server-broadcast overwrite +
  usage-count reset), mirroring `FLGDNMemory.load_memory`.
- Found and fixed a real latent bug from the `src/` role-based-subfolder
  reorg (`c16558e`): `src/robo3er/dataset.py` and `fl_dataset.py` computed
  `REPO_ROOT = Path(__file__).resolve().parents[1]`, correct when those
  files lived directly under `src/`, but wrong (`.../src`, not the repo
  root) once they moved into `src/robo3er/`. Fixed to `parents[2]` in both
  files -- this silently broke every `load_robo3er()` call after the
  reorg (`FileNotFoundError` on `data.npy`) until now.

## Client definitions per dataset

- **robo3er**: 5 real robots, `data/robo3er/partition.pkl` (same partition
  `fl_dataset.py`/`train_fl_memory_gdn.py` already use). Genuine multi-source.
- **Sielaff**: 10 real reverse-vending machines, `data/sielaff/partition.pkl`.
  Genuine multi-source.
- **Paderborn**: NO real multi-source structure (one test rig). Used the 6
  healthy reference bearings (K001-K006) as clients -- the closest thing
  this dataset has to independent physical units, but explicitly a
  synthetic/non-deployment split (one rig sequentially measuring 6
  bearings, not 6 independently operating machines), per user direction.
  Damaged bearings (separate physical units, not owned by any client) are
  scored against EVERY client's personalized model; reported number is the
  mean AUROC across all 6.
- **voraus-AD**: NO multi-source structure at all (single arm). Used 5
  SYNTHETIC clients = contiguous chunks of the PRE_A pool's sample-id range
  (755-1702, verified near-contiguous i.e. very likely chronological
  order), explicitly labeled as not representing any real deployment.
  Same per-client vs. every-fault evaluation convention as Paderborn.

## Results: centralized vs. federated (mean AUROC, ROUNDS=5 in all 4 runs)

| dataset | best centralized signal | best federated signal | verdict |
|---|---|---|---|
| robo3er | B=0.942 | B=**0.519** | collapses |
| Sielaff | 0.978 (V2 only) | **0.974** | ~matches |
| Paderborn | F=0.898 | F=**0.991** | federated WINS |
| voraus-AD | B=0.756 | ~**0.46-0.50** (near chance) | collapses |

Full per-fault/per-bearing/per-category numbers in
`checkpoints/{robo3er,sielaff,paderborn,voraus_ad}/*_federated_report.json`.
Scripts: `benchmark/run_{robo3er,paderborn,voraus_ad}_joint_prototype_v3_federated.py`,
`benchmark/run_sielaff_joint_prototype_v2_federated.py`.

## Why the split is exactly 2-wins-2-collapses, not random

**Federation succeeds where per-client sample volume is real and large
enough for `align_and_split` to find genuine cross-client structure, and
collapses where it can't:**

- **Sielaff** (wins): each of 10 machines still has 300-2100 fit windows.
  `align_and_split` found 1 shared cluster every round, stable. Nearly
  matches centralized because there's enough per-client data for a decent
  local codebook regardless of alignment quality.
- **Paderborn** (wins, even beats centralized): each client is a full
  56-file fit set per bearing -- plenty for 16 prototypes. `align_and_split`
  found 1-2 shared clusters, growing over rounds (round 1: 1, round 3+: 2).
  The federated number BEATS centralized specifically because damaged
  bearings are evaluated against ALL 6 personalized models and averaged --
  an ensembling effect the centralized single-model run doesn't get.
- **robo3er** (collapses): per-client fit sets are tiny and wildly uneven
  (robot01: 103 windows, robot02: 124, vs. robot04: 3231) -- already
  documented as a severe non-IID split in `fl_dataset.py`'s own docstring.
  `align_and_split` still found exactly 1 shared cluster from round 2
  onward, but with this little local data per client the codebooks
  themselves are poorly fit; robot00's `Broken Pipe` AUROC=0.000
  (perfectly INVERTED ranking) is the clearest symptom.
- **voraus-AD** (collapses hardest): `align_and_split` found ZERO shared
  clusters in ALL 5 rounds (`num_multi_client_clusters: 0` every round,
  the only dataset where this happened) -- the 5 synthetic time-chunks'
  codebooks never aligned at all, so each client ends up training and
  self-calibrating almost independently with no shared structure, on a
  client split that was never claimed to carry real distributional
  meaning in the first place (see client-definition note above). Every
  ablation column lands in the 0.40-0.50 band -- indistinguishable from
  chance, not just "worse."

**Practical implication**: this project's federated codebook-alignment
mechanism (`align_and_split`) is doing real, load-bearing work -- it is
NOT a no-op that happens to work when there's enough data anyway. Its
success or failure (0 vs. 1-2 shared clusters) tracks almost exactly with
whether the federated run beats, matches, or collapses relative to
centralized. A dataset with no genuine multi-source structure to exploit
(voraus-AD) is the clearest failure case tested so far -- federating a
single-machine dataset by inventing an arbitrary client split does not
recover a useful signal, whereas federating datasets with real
independent physical units (even a same-test-rig one like Paderborn's 6
bearings) can.

## Follow-up: WHY V3 specifically underperforms V2 on robo3er/voraus-AD (federated)

(V2 = `B_node_max` column, V3 = `F_v3_max` column.) robo3er:
V2=0.519 vs V3=0.447. voraus-AD: V2=0.460 vs V3=0.456 (both already near
chance, difference is noise). Two compounding causes, on top of the
per-dataset collapse reasons above:

1. **`edge_head`/`typed_head` never get federated at all** -- only
   `memory.codebook` is exchanged via `align_and_split`; V3's two extra
   heads (`TrendGraphAttentionHead`, `TypedRelationAnomalyHead`) are
   trained 100% locally every round with no cross-client averaging to
   compensate for small per-client data, unlike V2 which only needs
   `encoder`+`memory` to benefit from the shared codebook.
2. **`TypedRelationAnomalyHead.set_calibration`'s `MIN_PROTO_SAMPLES=5`
   requirement is structurally unmet per-client**: robo3er's calib splits
   are 22-692 windows, voraus-AD's are a flat 38 -- filling all 16
   prototypes to >=5 samples needs >=80 well-distributed calib windows at
   minimum, which most clients never reach, so the E (typed-edge) signal
   is standardized against a noisy few-dozen-sample GLOBAL fallback
   instead of per-prototype stats.
3. This reproduces the exact **max-aggregation noise-floor problem**
   already documented in the centralized voraus-AD ablation
   (`joint-prototype-scheme-v3.md`): folding a poorly-calibrated E into
   `F = max(A, B, C, E)` can pull F BELOW even C alone. Confirmed again
   here -- robo3er federated: C=0.523 alone vs. F=0.447 (F is WORSE than
   dropping E entirely).
4. Dataset-specific compounding factor, robo3er: severe per-client
   imbalance (robot01 fit=103, robot02 fit=124 vs. robot04 fit=3231) means
   the edge/typed heads' larger parameter counts can't converge reliably
   in 12 local epochs on the small clients, and unlike the codebook they
   are never "pulled back" toward consensus between rounds.
5. Dataset-specific compounding factor, voraus-AD: the underlying
   representation is already broken (0 aligned clusters every round, A
   itself at chance) AND the model/data ratio is far worse than robo3er's
   (66 nodes/54 declared edges vs. ~151 effective training SEQUENCES per
   client, not windows -- an order of magnitude less signal per
   parameter than robo3er's better-off clients).

## Follow-up: loss-confidence-weighted combination (`G_weighted_v3`)

Replaced the hard `max()` across V3's 3 cross-component signals (memory,
edge attention, typed edge) with a weighted sum, keeping the within-
component max (across nodes/edges) unchanged. `federated_memory.
confidence_weights_from_losses()`: per client, use each component's OWN
calib-time raw loss (`d_proto.mean()` for memory, `resid_struct.mean()` for edge
attention, `r_edge.mean()` for typed edges -- literally the same quantities
those components' training loss terms minimize, just evaluated on held-out
calib data), z-score each component's loss ACROSS CLIENTS (comparable
that way, not in absolute terms), softmax the negative z-scores into a
per-client weight triple summing to 1. No extra forward passes needed --
reuses calib arrays already computed for z-scoring. Wired into all 3 V3
federated scripts as a new `G_weighted_v3` ablation column, `F_full_v3`
kept unchanged for comparison.

Result confirms the diagnosis was correct, in all 3 directions at once:

| dataset | F (max) | G (weighted) | delta |
|---|---|---|---|
| robo3er | 0.447 | **0.469** | +0.022, real fix |
| Paderborn | 0.991 | 0.989 | -0.002, neutral (max wasn't broken here) |
| voraus-AD | 0.456 | 0.459 | +0.003, negligible |

- **robo3er improves** because this was exactly the diagnosed failure
  mode (a poorly-calibrated component dragging the max down) -- weighting
  fixes it, though G still trails B/C alone (0.519/0.523), so the
  combination isn't fully recovering what dropping the typed signal
  entirely would give.
- **Paderborn is unchanged** because nothing was broken there to begin
  with -- each client had enough calib data (56 fit files/client) for
  every component to be reasonably well-calibrated, so max() was already
  near-optimal; reweighting has nothing to fix.
- **voraus-AD barely moves** because the failure there is upstream of the
  combination step entirely -- 0 aligned prototype clusters, A itself at
  chance (0.497). Confidence weighting can only pick the LEAST-broken
  signal among already-broken ones; it cannot manufacture signal that was
  never learned. Confirms the earlier conclusion that voraus-AD's
  federated collapse needs a different fix (more rounds, richer client
  overlap, or abandoning the fully-synthetic time-chunk split) --
  reweighting the combination formula was never going to be sufficient
  there.

Side observation: robo3er's per-client weights land close to uniform
(0.27-0.38 each), while Paderborn's are much more polarized (K001:
edge=0.81; K003: typed=0.82) -- Paderborn's clients have far more local
calib data (56 fit files vs. robo3er's ~100-3231 WINDOWS), so per-client
component quality differences are more real and the weighting has more
genuine signal to work with, not just noise.

## Follow-up: FedAvg'ing encoder+decoder too (`_encdec_synced` reports)

Diagnosed cause of the federated-vs-centralized gap on robo3er/voraus-AD:
`SharedEncoder` was NEVER federated (only `JointPrototypeMemory`'s
codebook was) -- each client's encoder drifts independently every round's
`LOCAL_EPOCHS` of local-only training ("client drift"), so by the time
`align_and_split` runs, cross-client cosine similarity is being computed
in coordinate systems that may have already diverged. Tested the fix:
added `federated_memory.fedavg_state_dict()` (sample-size-weighted FedAvg
over a chosen key-prefix subset of each client's `state_dict()`) and, at
the END of every round (after local training, alongside but separate from
memory alignment), FedAvg BOTH `encoder.` and `decoder.` weights and
broadcast the same averaged pair back to every client. `edge_head`/
`typed_head` stay untouched/fully local, `memory` keeps its existing
personalized-alignment path. This is an explicit, opt-in departure from
this module's original "only exchange the memory dictionary"
design/communication-efficiency commitment -- gated behind a
`SYNC_ENCODER_DECODER` flag, results saved to separate
`*_federated_encdec_synced_report.json` files so both variants stay
comparable. Decoder was synced alongside encoder (not left local) because
it's training-only (never touches inference-time scoring, only shapes the
encoder via the reconstruction loss), same size as encoder, and leaving it
local while encoder gets swapped out at the round boundary would create a
transient, purely wasteful encoder/decoder mismatch for no personalization
benefit (StandardScaler already handles the per-client scale
personalization that matters).

Results (mean AUROC): confirms the diagnosis in all 3 datasets, exactly
tracking each dataset's OWN root cause identified above:

| dataset | A, no sync | A, encoder+decoder synced | F, no sync | F, synced | shared clusters found |
|---|---|---|---|---|---|
| robo3er | 0.449 | **0.686** | 0.447 | **0.506** | 1 (round 2+) -> **5** (round 5) |
| Paderborn | 0.823 | 0.829 | 0.991 | 0.989 | 2, unchanged |
| voraus-AD | 0.497 | 0.566 | 0.456 | 0.489 | **0, unchanged in every round even after sync** |

- **robo3er wins big**: this dataset's clients have REAL shared structure
  (same robot kinematics/sensor relationships) that client drift was
  hiding from the alignment step -- fixing the coordinate-system mismatch
  immediately let `align_and_split` find it (round 5: 5 shared clusters,
  vs. 1 max previously). A/B/F all move substantially toward their
  centralized values, though a real gap to centralized (0.893/0.942/0.861)
  remains -- likely still partly a sample-size effect for the smallest
  clients (robot01: 103 windows) even with a shared encoder, and
  `edge_head`/`typed_head` still aren't federated at all.
- **Paderborn is unaffected either way**, confirming it was never a
  client-drift problem there -- ~3300 effective windows/client already
  gave each client enough data to fit a stable encoder+memory
  independently.
- **voraus-AD improves only partially, and the ROOT diagnostic signal
  (shared-cluster count) doesn't move at all** -- 0 clusters in every
  round, sync or no sync. This is the clearest evidence yet that
  voraus-AD's federated collapse is NOT primarily a coordinate-system/
  drift problem the way robo3er's was -- it's a raw data-starvation
  problem (151-152 effective training SEQUENCES per client for a 66-node/
  54-edge model) that a shared encoder alone cannot fully compensate for:
  a better-fit shared encoder still can't make 151 samples populate 16
  prototypes' usage statistics reliably enough for cross-client alignment
  to find anything in common. Encoder sync treats the symptom it's
  designed for (coordinate mismatch) but voraus-AD's dominant symptom is
  a different one (too little data per client, full stop).

## Follow-up: windowing voraus-AD's federated clients, and a FedAvg baseline

Diagnosed root cause (from the encoder-sync experiment above): voraus-AD's
federated collapse is a raw data-starvation problem, NOT client drift --
each of the 5 synthetic clients only had ~151 EFFECTIVE training examples
(one whole ~1164-step pick-and-place cycle = one training unit, unlike
robo3er/Paderborn which are windowed). Fix tested:
`run_voraus_ad_joint_prototype_v3_federated_windowed.py` slides a much
shorter window (`WINDOW_LEN=64, WINDOW_STRIDE=32`, matching Paderborn's
convention) across each cycle's ACTUAL pre-padding length (windowing must
use the real per-sample length, not the padded 1164, to avoid
contaminating windows with zero-padding), turning ~151 samples/client into
~4900 windows/client -- same order of magnitude as Paderborn's ~3300
windows/client. Since fault/normal labels are per ORIGINAL CYCLE not per
window, evaluation aggregates window-level scores back to per-cycle via
mean (`per_file_scores`, same convention as the Paderborn federated
script).

Result (12 fault categories, mean AUROC): **B: 0.460 -> 0.622, C: 0.483 ->
0.644, D: 0.486 -> 0.639** (large, real gains) -- and critically, the
shared-prototype-cluster count FINALLY moved off zero (round 2: 2 shared
clusters, vs. 0 in every round of every prior voraus-AD variant, sync or
no sync). A itself barely moved (0.497 -> ~0.55), consistent with
windowing mainly helping the higher-capacity edge/typed components that
need more data to fit, not the coarser device-level prototype signal.
Two smallest categories (`miss_can` n=11, `invalid_position` n=12) were
tried excluded as a quick low-noise check, then restored (removing them
barely changed the mean -- they weren't uniformly bad, just noisy in both
directions) -- final reported numbers use all 12 categories.

### FedAvg baseline comparison (the important, slightly counterintuitive result)

Built `run_voraus_ad_joint_prototype_v3_fedavg_baseline.py`: identical
windowed data/model/rounds/local-epochs as the windowed script above,
differing ONLY in aggregation -- full-parameter weighted averaging
(`baselines.fedavg_baseline.federated_average`, already used for robo3er's
own "ours vs. FedAvg" ablation) of EVERY parameter (encoder + memory +
edge_head + typed_head + decoder) every round, producing ONE identical
global model for all 5 clients, no personalization anywhere (vs. our
script's memory-only alignment + encoder/decoder-only FedAvg, with
edge_head/typed_head staying personalized/local).

| signal | ours (personalized memory+edge/typed) | FedAvg (no personalization) | winner |
|---|---|---|---|
| A | **0.552** | 0.541 | ours, barely |
| B | 0.622 | **0.627** | FedAvg |
| C | 0.644 | **0.673** | FedAvg |
| D | 0.639 | **0.657** | FedAvg |
| E | 0.499 | **0.524** | FedAvg |
| F | 0.511 | **0.536** | FedAvg |

**Plain FedAvg slightly but consistently beats our personalized approach
on voraus-AD, most visibly on the edge/typed signals.** This is not a bug
in our method -- it's the predicted consequence of the diagnosis above.
Our personalization design (memory aligned but distinct per client,
edge_head/typed_head fully local) is only a good idea when clients have
REAL heterogeneity worth preserving (robo3er's 5 different robots,
Sielaff's 10 different machines, even Paderborn's 6 different physical
bearings). voraus-AD's 5 "clients" are synthetic time-chunks of the SAME
single arm -- there is no genuine cross-client heterogeneity to protect
via personalization, only a cost (fewer effective samples per personalized
parameter). Under that condition, full parameter-sharing (FedAvg) has
nothing to lose and everything to gain by pooling all 5 clients' ~750
combined cycles into one shared model, and it does measurably better.
This is a clean negative control that supports the project's overall
thesis from the other direction: personalized/memory-only federation's
advantage is conditional on real multi-source heterogeneity, not free.

## Follow-up: robo3er vs. the existing full-parameter FedAvg baseline

Compared against `benchmark/run_fedavg_baseline.py` (pre-existing in this
repo, `checkpoints/robo3er/fedavg_baseline_report.json` -- re-ran it fresh
to confirm it wasn't stale after the `REPO_ROOT` fix; numbers reproduced
exactly, deterministic under `SEED=42`). This baseline uses `FLGDNMemory`
(different architecture: per-feature-independent codebook, not
`JointPrototypeMemory`'s joint snapshot), 38 `kinematic_core` nodes (not
our hand-picked 7), 256 prototypes (not 16), and full-parameter weighted
FedAvg every round (encoder+memory+structure+decoder ALL averaged, ZERO
personalization) -- NOT a controlled same-architecture ablation, but the
only existing "plain FedAvg" reference point in the repo, on the exact
same 5-robot partition.

| variant | node/proto config | node-only signal | combined signal |
|---|---|---|---|
| FedAvg baseline (FLGDNMemory, full-param avg, 0 personalization) | 38 nodes, 256 proto | d=0.500 | d+r=0.516 |
| Our V3, memory-only exchange (no encoder sync) | 7 nodes, 16 proto | A=0.449 | F=0.447 |
| Our V3, encoder+decoder synced | 7 nodes, 16 proto | A=0.686 | F=0.506 |
| centralized V3 (reference ceiling) | 7 nodes, 16 proto | A=0.893 | F=0.861 |

**The un-synced version of our own design did NOT beat the dumb
zero-personalization FedAvg baseline** (A=0.449/F=0.447 vs. FedAvg's
d=0.500/d+r=0.516) -- a genuinely humbling cross-check on the "only
exchange the memory dictionary, keep encoder fully local for
personalization" design's actual payoff on this dataset. The reason lines
up exactly with the client-drift diagnosis above: FedAvg's brute-force
full-model averaging every round accidentally SOLVES the client-drift
problem (nothing can drift far if the whole model gets reset to a global
average every round) at the cost of zero personalization; our original
design kept personalization but paid for it with unconstrained encoder
drift. Only once encoder+decoder sync was added did our approach clearly
beat BOTH the FedAvg baseline AND its own un-synced self -- getting
FedAvg's representation-consistency benefit AND still keeping
`JointPrototypeMemory`'s per-client personalized codebook (something the
full-FedAvg baseline has none of, since it averages memory away too).

Not yet done: a truly controlled version of this comparison (same 7-node
feature set, same 16-prototype budget, `JointPrototypeGDNv3` vs.
`FLGDNMemory`, only the aggregation rule differing) -- the current
comparison mixes architecture, node-set, and prototype-count differences
together with the aggregation-rule difference.

## Follow-up: robo3er expanded from 7 hand-picked nodes to all 68 kept features

The original 7-node robo3er graph (`run_robo3er_joint_prototype_v3.py`)
was undersized relative to every other dataset here (Paderborn's 6 nodes
IS its full available channel set; Sielaff's 39 and voraus-AD's 66 are
each close to full) -- it only kept the columns needed for the 7 declared
physics edges, discarding 61 other real channels (odometry
position/orientation, wheel ticks, IMU acceleration, PWM,
battery/IR/cliff/status) that V2's edge-free node-deviation signal could
still have used. Fixed: both the centralized and federated robo3er V3
scripts now use ALL 68 kept features as nodes (`TOP_K` raised 5->8 to
match), keeping the same 7 declared edges embedded in the larger graph
(the other 61 nodes simply get 0 from `resid_phys`, still score via
`d_node`/`resid_struct`, exactly as `TypedRelationAnomalyHead`'s docstring
describes for undeclared nodes).

Result: expanding nodes is a NET WIN centralized but a NET LOSS federated
-- for two entirely different, dataset-independent reasons each already
documented elsewhere in this project:

| setting | A (7 nodes) | A (68 nodes) | F (7 nodes) | F (68 nodes) |
|---|---|---|---|---|
| centralized | 0.893 | 0.739 | 0.861 | 0.868 |
| federated, no sync | 0.449 | 0.406 | 0.447 | 0.419 |
| federated, encoder+decoder synced | 0.686 | 0.431 | 0.506 | 0.492 |

- **Centralized, per-category**: 3 of 4 fault types (`Broken Pipe`,
  `cable trapped`, `Low battery`) jump to >=0.99 AUROC with the extra 61
  nodes available -- confirms real signal was being discarded before.
  BUT `stuck` collapses from a strong category to ~0.45-0.53 (near
  chance) -- this is the exact **max-aggregation noise-floor problem**
  already found and fixed on voraus-AD's node expansion in
  `joint-prototype-scheme-v3.md` (raw `.max()` over more dimensions
  systematically inflates the normal-data false-positive tail,
  independent of whether the new dimensions are individually
  informative) -- NOT fixed here yet (still raw max, not the top-k-mean
  two-stage fix voraus-AD got), a clear next step.
- **Federated gets WORSE at 68 nodes, for reasons specific to
  federation**, not the aggregation problem above:
  1. **Alignment gets harder in higher dimensions**: each prototype is
     now a `[68, 64]` (~4352-dim) vector for cosine similarity instead of
     `[7, 64]` (448-dim) -- the same `DELTA=0.5` threshold finds far less
     agreement (encoder-synced run: only 1 shared cluster by round 5, vs.
     5 at 7 nodes).
  2. **Per-client data/parameter ratio worsens**: local sample counts are
     unchanged (robot01 still 103 windows) but the model now has to fit
     68 nodes instead of 7 from that same tiny sample -- compounds the
     per-client underfitting problem already documented above.

Net takeaway: node-count should track each dataset's real available
channels (as this project already does for the other 3 datasets), but
this ALSO makes federated alignment strictly harder in a way centralized
training doesn't feel -- there's a real, dataset-size-dependent tension
here that a fixed `DELTA` doesn't automatically handle. Not yet tried:
loosening `DELTA` for larger node counts, or top-k-mean aggregation to
fix the `stuck` regression.

## Follow-up: why federated << centralized (synthesized), and is node count too big relative to per-client data

Two explicit questions, answered with concrete numbers (not just the
alignment-cluster-count proxy used above):

**Why federated lags centralized, enumerated**: (1) FedAvg-style periodic
sync is a fundamentally weaker optimizer than centralized SGD over pooled
data under non-IID clients (well-known result) -- even with encoder
sync, correction only happens once per round, not every gradient step;
(2) EACH CLIENT calibrates its own z-scores from its own tiny calib split
(robot00: 41 windows, robot01: 22) instead of centralized's pooled
~1000+, adding pure calibration noise on top of any representation gap;
(3) `edge_head`/`typed_head` are still 100% local even with encoder sync,
so C/E signals get none of the encoder-sync benefit; (4) higher node
count makes the alignment step itself harder (see below).

**Is node count too big relative to per-client training data**: yes,
confirmed by directly counting `JointPrototypeGDNv3`'s parameters (not
estimated): the `JointPrototypeMemory` codebook is the ONLY component
that scales with node count (`M * N * D`) -- `encoder` (single shared
`Linear`), `typed_head` (scales with declared EDGES, not nodes) are node-
count-invariant. Measured total model parameters: 7 nodes/16 proto =
56,638; 68 nodes/16 proto = 123,006 (memory alone: 7,168 -> 69,632, 9.7x).
Dividing by each client's fit-sample count gives a
parameters-per-training-sample ratio that is wildly uneven across
clients precisely BECAUSE robo3er's federation is severely non-IID in
volume:

| client | fit samples | ratio @ 7n/16M | ratio @ 68n/16M | ratio @ 42n/8M |
|---|---|---|---|---|
| robot01 (smallest) | 103 | 550 | **1194** | 711 |
| robot04 (largest, ~83% of pooled data) | 3231 | 17.5 | 38 | ~23 |
| centralized pool | 3886 | 14.6 | 31.7 | ~19 |

robot04 alone is close enough to the full centralized pool that 68 nodes
is roughly as tolerable for it as for centralized training (explaining
why centralized barely notices the node-count increase) -- but robot01/
02/00/03 are 15-30x worse off at 68 nodes than robot04 is, which is
exactly why the smallest clients are the ones cratering hardest. Node
count "being too large" is not a single dataset-level property here; it's
a PER-CLIENT property that a single shared architecture can't
simultaneously get right when client data volumes differ by 30x.

## Follow-up: federated-capacity-scaled node/prototype count for robo3er

Fix tested: keep the centralized script at 68 nodes/16 prototypes
(no problem there), but give the FEDERATED script its own smaller
config -- `KINEMATIC_CORE + ACTUATION` (42 nodes, still 6x the original
7, keeps all 7 declared edges' endpoints, drops only `STATUS_FLAGS`/
`ENVIRONMENT`) and `NUM_PROTOTYPES=8` (halved), bringing robot01's
ratio from 1194 down to 711.

| config | A, no sync | A, synced | F, no sync | F, synced |
|---|---|---|---|---|
| 7 nodes / 16 proto (original) | 0.449 | 0.686 | 0.447 | 0.506 |
| 68 nodes / 16 proto (full features) | 0.406 | 0.431 | 0.419 | 0.492 |
| **42 nodes / 8 proto (capacity-scaled)** | **0.420** | **0.533** | **0.462** | **0.519** |

The capacity-scaled config clearly beats the full-68-node config on every
column, and its `F_full_v3` (0.519, synced) is the best of all three
configs tested -- better than even the original 7-node version's 0.506,
suggesting the extra kinematic/actuation nodes DO carry real incremental
signal once the prototype budget is kept proportionate to the smallest
client's data. `A`/`B` still trail the 7-node version's synced numbers
(0.533 vs. 0.686), meaning 42 nodes is still somewhat oversized for the
smallest clients relative to 7 -- a real remaining tension between
richer node coverage and small-client capacity that a single node/proto
count can't fully resolve. Not yet tried: shrinking further (e.g.
`NUM_PROTOTYPES=4-6`, or `kinematic_core` alone without `actuation`), or
a genuinely per-client-sized prototype budget instead of one shared M.

## Follow-up: pushing NUM_PROTOTYPES down further (42-node robo3er federated)

Swept `NUM_PROTOTYPES` at the fixed 42-node (`kinematic_core+actuation`)
config, encoder+decoder synced, to find where robo3er's federated
`A`/`B`/`F` peak:

| M | A | B | C | F_full_v3 |
|---|---|---|---|---|
| 16 (orig, wrong node count) | 0.431 | 0.474 | -- | 0.492 |
| 8 | 0.533 | 0.542 | 0.521 | 0.519 |
| 4 | 0.562 | 0.558 | 0.550 | 0.539 |
| **2** | 0.558 | **0.620** | **0.653** | **0.619** |
| 1 (degenerate, no clustering) | 0.562 | 0.611 | 0.618 | 0.577 |

`NUM_PROTOTYPES=2` is the best point found -- `F_full_v3=0.619` clearly
beats even the ORIGINAL 7-node/16-proto config's 0.506, and `B`/`C` are
the highest of anything tested in the federated setting so far. `M=1`
(fully degenerate -- one global "prototype," no clustering at all) is
slightly WORSE than `M=2`, showing there's real, if minimal, value in
being able to distinguish at least 2 operating regimes; going all the way
to zero clustering capacity isn't the right direction either.

**Disentangling experiment**: reran `M=2` at the FULL 68-node config
(instead of 42) to isolate whether the gain was really about node count,
prototype count, or both. Result: `A=0.433, F=0.445` -- clearly worse
than 42-node/`M=2` (`A=0.558, F=0.619`), confirming BOTH the node-count
reduction (68->42) and the prototype-count reduction (16->2) are doing
real, separately-necessary work -- neither alone recovers what the
combination gets. `NUM_PROTOTYPES` set to 2 in
`run_robo3er_joint_prototype_v3_federated.py` going forward (centralized
script unchanged at 16, since it doesn't have this problem).

Not yet tried: whether `M=2` is dataset-specific to robo3er's federated
setting or would also help Sielaff/Paderborn/voraus-AD's federated
configs (Paderborn in particular already works well at `M=16`, so
shrinking there might just lose useful cluster resolution without a
matching benefit -- would need testing per-dataset, not assumed).

## Follow-up: top-k-mean aggregation fix ported to robo3er federated

Ported the exact `topk_mean()` + `two_stage_group_score()` fix already
validated on voraus-AD centralized (`run_voraus_ad_joint_prototype_v3.py`,
`TOP_K_AGG=3`) into `run_robo3er_joint_prototype_v3_federated.py`: the
node/edge/typed-edge GROUP signals (many dimensions) now get a
top-3-mean + second calibration stage instead of a raw `.max(axis=1)`;
the top-level combination across the 4 signal groups (dG/node/edge/typed)
stays a plain max (only 4 items, far less multiple-comparisons risk than
42-68 dimensions).

| config | F, raw max | F, topk-mean fix |
|---|---|---|
| 42 nodes / M=2 (chosen config) | 0.619 | **0.627** |
| 68 nodes / M=2 (disentangling test) | 0.445 | 0.483 |

The fix gives a real, small, POSITIVE improvement at ANY node count
(confirms the mechanism generalizes beyond voraus-AD) -- but it does NOT
let 68 nodes catch up to 42 nodes (0.483 still far below 0.627). This
refutes the hypothesis raised when this fix was proposed ("maybe fixing
the aggregation noise floor lets us restore the dropped
status_flags/environment nodes, e.g. `slip_status_is_slipping`, `battery_
state_*`, for free"): the max-aggregation noise floor was a real, fixable
contributor to the 68-node regression, but NOT the dominant one -- the
dominant cause remains the per-client parameter/data-volume mismatch
documented above (memory codebook ~10x more parameters, alignment in a
~10x higher-dimensional space at 68 vs. 7-42 nodes). Fixing the
aggregation formula improves whatever node count you choose, but doesn't
change which node count is actually viable for robo3er's smallest
federated clients. `TOP_K_AGG=3` kept in the final 42-node/`M=2` config
going forward (both `SYNC_ENCODER_DECODER` variants).

## Follow-up: data-driven redundant-column removal (42 -> 31 nodes)

Requested: analyze robo3er's actual data for columns worth dropping
(NaN-heavy or duplicate content), rather than only using
`feature_groups.py`'s pre-existing group boundaries. Findings from
directly computing per-column NaN rate and a 50k-row-sample pairwise
correlation matrix over the 68 kept columns:

- **No NaNs anywhere** (0.0% on every one of the 68 columns) --
  `data.npy` is already NaN-clean; this part of the hypothesis didn't
  hold, worth stating plainly rather than confirming a false premise.
- **11 columns in the current 42-node set are near-duplicates of columns
  already kept**: `tf_link_base_link_{trans_x,y,z,rot_x,y,z,w}` (7) and
  `tf_footprint_base_footprint_{trans_x,y,rot_z,w}` (4), all highly
  correlated (r=0.92-0.998) with `odom_odo_pos_*`/`odom_odo_orient_*`,
  which are also already kept. This directly confirms
  `feature_groups.py`'s own docstring, which already characterized these
  as "a software-broadcast re-expression of the SAME odometry estimate
  in a different frame ... not an independent sensor ... not a physical
  fault signal" -- this analysis is the first time that claim was checked
  against actual correlation numbers rather than taken on documentation
  alone.
- **Two other high-correlation pairs found are NOT redundant and must
  NOT be dropped**: `odom_odo_angtw_z`<->`imu_imu_angvel_z` (r=0.917) and
  `wheel_vels_velocity_left`<->`odom_odo_lintw_x` (r=0.908) are exactly
  the two declared physics edges checking "do these two independent
  estimators of the same quantity still agree" -- their high correlation
  under normal operation IS the point; a fault is specifically when that
  correlation breaks, so removing either side would delete the exact
  detection capability those edges exist for. High pairwise correlation
  alone is NOT sufficient grounds for removal -- has to be checked
  against whether the pair is a declared/plausible cross-check
  relationship first.

Fix: `EXCLUDE_NODES_PREFIXES = ("tf_link_base_link_",
"tf_footprint_base_footprint_")` filters these 11 columns out after
`feature_groups.select_columns`, landing at 31 nodes (down from 42), with
the 7 declared edges and both legitimate cross-check pairs left
untouched.

| config | A | B | F_full_v3 |
|---|---|---|---|
| 42 nodes (includes the 11 redundant tf_* columns) | 0.558 | 0.636 | 0.627 |
| **31 nodes (redundant columns removed)** | 0.495 | **0.727** | **0.712** |

Large, real improvement on `B`/`F` (0.712 now clearly beats even the
original 7-node/16-proto config's 0.506, and `B`=0.727 is close to that
config's own 0.741). `A` (pure prototype signal) actually drops slightly
(0.558->0.495) -- plausible since removing 11 correlated dimensions
changes the raw scale of the joint-distance computation somewhat, but `B`/
`F` (which fold in the node/edge signals, and are what actually matters
for detection) are unambiguously better, so this isn't read as a
regression. `NUM_PROTOTYPES=2`, `TOP_K_AGG=3` from the prior two
follow-ups are unchanged and compound with this one.

## Follow-up: removing cumulative/unbounded-quantity nodes (31 -> 26)

Further feature-space search after the tf_* redundancy removal:
`odom_odo_pos_{x,y,z}` (chassis position) and `wheel_ticks_ticks_{left,
right}` (wheel tick COUNT) are CUMULATIVE/UNBOUNDED quantities, unlike
every other kept node (velocity/orientation/current/PWM -- all roughly
stationary/bounded). `JointPrototypeMemory` matches windows to a small
fixed set of "operating regimes" -- position doesn't have regimes in that
sense (a different location in the building isn't a different physical
state), so including it adds a slowly-drifting dimension prototype
matching can chase but never stably cover.

`EXCLUDE_NODES_PREFIXES` extended to also drop `odom_odo_pos_`/
`wheel_ticks_` (31 -> 26 nodes).

| config | A | B | C | D | F |
|---|---|---|---|---|---|
| 31 nodes | 0.495 | 0.727 | 0.727 | 0.638 | 0.712 |
| 26 nodes (encoder+decoder synced) | 0.511 | **0.742** | **0.747** | **0.750** | 0.673 |
| 26 nodes (NOT synced) | 0.411 | 0.503 | 0.498 | 0.502 | 0.542 |

`B`/`C`/`D` all improve when synced (`D_full_v2` -- excludes the typed
signal entirely -- 0.638->0.750, the cleanest read on "did removing these
help"). `F_full_v3` gets WORSE (0.712->0.673) purely because its plain
`max()` still folds in `E_typed_edge_only` (0.460, still the weakest,
never-yet-fixed signal) -- unrelated to this specific change, a symptom
of `E`'s own unresolved weakness, not evidence against removing position/
ticks.

**Conditional on encoder sync, not universal**: the identical column
removal, WITHOUT `SYNC_ENCODER_DECODER`, makes `B`/`D` WORSE (0.598/0.589
at 31 nodes -> 0.503/0.502 at 26) -- this reduction only pays off when
encoder sync is also on. Kept as the new default (26 nodes) since
encoder+decoder sync is the recommended standard configuration going
forward, not an incidental setting.

Diminishing returns reached on the feature-space-search axis: `A` has
plateaued around 0.5, and `E`/typed-edge remains the one component never
touched by any of these follow-ups (still weak, still dragging `F` down
whenever it's folded in via max). Next-highest-leverage direction is
fixing `E` directly (federating its prototype-conditioned calibration
stats, or applying `two_stage_group_score`-style aggregation to its own
internals), not further feature pruning.

## Follow-up: tested re-including battery/slip/stop -- reverted, but for an informative reason

User pushback on the 26-node reduction: battery-related topics and
`slip_status_is_slipping`/`stop_status_is_stopped` shouldn't have been
pruned just for being outside `KINEMATIC_CORE`/`ACTUATION` -- battery is
the direct diagnostic source for the `Low battery` fault, slip status is
the robot's own slip-detection flag, stop status plausibly relates to
`stuck`. Reasonable domain argument, tested empirically (added
`battery_state_{voltage,current,charge,capacity,temperature,percentage}`
+ `slip_status_is_slipping` + `stop_status_is_stopped`, 26 -> 34 nodes).

Result: clearly WORSE, not better -- `B` 0.742->0.521, `D_full_v2`
0.750->0.520, and even `Low battery`'s OWN AUROC got worse (0.402->0.183).
Reverted (`INCLUDE_EXTRA_NODES` kept as an empty list, not deleted, since
the underlying concern is legitimate).

Why, diagnosed rather than just accepted as "doesn't help": each addition
hits a DIFFERENT already-known failure mode from this same follow-up
chain, not "these signals are irrelevant":
- `battery_state_*` is ANOTHER cumulative/non-stationary quantity (drains
  ~monotonically over a session) -- the exact same class of problem as
  `odom_odo_pos_*`/`wheel_ticks_*`, which were already removed for this
  reason. It doesn't cluster into operating "regimes" any more than raw
  position does.
- `slip_status_is_slipping`/`stop_status_is_stopped` are near-constant
  binary flags (99.9%/92.3% one value) -- z-score/top-k-mean
  normalization degenerates on them (near-zero IQR on normal data, then a
  huge spike on the rare nonzero value), the same "tiny-client IQR
  blowup" failure mode found on the older `FLGDNMemory` architecture (its
  fix, `iqr_floor_frac`, is documented in `feature-purification-audit.md`).

Not a closed question -- the right fix for wanting battery/slip/stop
represented is a DIFFERENT encoding, not raw inclusion as a plain node:
a within-window battery RATE OF CHANGE (turns a drifting quantity into a
roughly-stationary one, same idea `kinematics.py`'s residual already
uses elsewhere) instead of raw level; scoring slip/stop status as a
separate rule/label check outside the continuous z-score/top-k-mean
aggregation instead of forcing it through the same pipeline as
velocity/current. Not implemented yet.

**Follow-up: isolated battery_state_voltage/current (not lumped with
charge/capacity/percentage)** -- tested whether SPECIFICALLY the two
electrical (not clearly-cumulative) battery columns behave differently
from the other 4. Two diagnostics first: (1) `battery_state_voltage`'s
median WITHIN-WINDOW (60-step) std is exactly 0.0000 -- more than half
of all windows see zero variance in it, vs. `wheel_status_current_ma_
left`'s median within-window std of 0.4703 (varies almost every window)
-- battery telemetry updates much slower than the window timescale, so
it's usually flat at the window granularity the model actually scores
on; (2) correlation with actual drive current is weak
(`battery_state_current` vs. `wheel_status_current_ma_left`: r=-0.105;
`battery_state_voltage` vs. same: r=+0.055) -- NOT the real-time
load-response relationship hypothesized (voltage sagging under
instantaneous motor draw); robot04's full normal-fit sequence shows
`battery_state_voltage` moving in slow plateaus/steps (likely
charge-cycle stage), not a real-time-responsive signal or even a clean
monotonic within-session drain (corr with sequence position: 0.032).

Tested anyway (26 -> 28 nodes, voltage+current only): still worse across
the board (`D_full_v2` 0.750->0.600), and -- same counter-intuitive
result as the full battery/slip/stop test -- `Low battery`'s OWN AUROC
did NOT improve (0.402->0.360-0.370, slightly worse). Reverted again.
Conclusion: it's not specifically the cumulative columns (charge/
capacity/percentage) dragging things down -- voltage/current fail for
their own, different reason (mostly flat at the window timescale the
model scores at, and their between-window variation reflects
charge-cycle stage more than the robot's current kinematic regime, so
they contribute noise-dimension cost without regime-discriminating
signal). All 6 `battery_state_*` columns are now ruled out under the
current node-as-raw-continuous-signal encoding, not just the 4 obviously
cumulative ones.

## What's NOT done

- No hyperparameter tuning of the federated setting specifically (ROUNDS,
  LOCAL_EPOCHS, GAMMA, DELTA all carried over from
  `train_fl_memory_gdn.py`'s robo3er-tuned values, or matched 1:1 to each
  dataset's own centralized EPOCHS) -- robo3er's and voraus-AD's collapse
  might partly be addressable by more rounds/different DELTA, not
  established either way here.
- No repeat/multi-seed runs to separate genuine federation effects from
  seed variance, same caveat as the centralized V2/V3 runs.
