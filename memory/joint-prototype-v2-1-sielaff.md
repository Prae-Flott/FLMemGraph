---
name: joint-prototype-v2-1-sielaff
description: V2.1 (V2 + generic GDN attention, no declared edges) on Sielaff -- attention adds nothing, slightly hurts
---

# V2.1 on Sielaff: generic GDN attention adds nothing, slightly hurts
# (2026-08-11)

## What was tested

`JointPrototypeGDNv21` (`src/models/joint_prototype_model.py`): V2
(`SharedEncoder` + `JointPrototypeMemory`) + `TrendGraphAttentionHead`
(GDN-style learned attention over TopK-similar neighbors), with NO typed
relation head and `prior_edges=None` -- fully generic attention, no
physics knowledge needed at all. Treats an abnormal change in
cross-feature attention (`s_edge`) as one manifestation of a fault, on
top of V2's node-level signal. First edge/attention-based signal ever
tried on Sielaff (`benchmark/run_sielaff_joint_prototype_v2_1.py`,
39 nodes, top_k=10).

## Result: attention doesn't help here, and joint training slightly hurt the other signals too

| method | V2 (no edges) | V2.1 (+ generic attention) |
|---|---|---|
| A: prototype only | 0.971 | 0.954 |
| B: prototype + node | **0.978** | 0.965 |
| C: edge (attention) only | n/a | 0.960 |
| D: full | n/a | 0.960 |

V2.1's best combination (0.965, B) is still below plain V2 (0.978).
`A` and `B` themselves dropped too (0.971->0.954, 0.978->0.965) despite
being computed the same way in both runs -- the same joint-training
effect documented on Paderborn (`memory/joint-prototype-scheme-v3.md`,
adding an edge-related auxiliary loss changes what the shared encoder
learns even for signals that don't directly use the edge head), except
here the effect is NEGATIVE rather than positive. `prior_bias_strength`
stayed at its 1.0 init throughout training -- expected, since no
declared edges exist for it to act on.

## Interpretation

This sharpens the earlier finding ("Sielaff has no verified physical
prior," `benchmark/datasets/sielaff_physics.md`) into a stronger claim:
**even attention that needs zero domain knowledge (pure data-driven,
GDN-style) finds nothing useful to exploit on this dataset.** This isn't
a knowledge gap (we don't know which edges to declare) -- it's evidence
that Sielaff's fault mechanisms (bottle recognition failures, mechanical
jams, compactor status errors) don't manifest as cross-feature
relationships breaking, consistent with this project's broader,
repeatedly-confirmed finding
(`memory/joint-prototype-scheme-v2.md`/`-v3.md`) that edge/attention
signals only pay off on datasets whose fault mechanisms are genuinely
relational (Paderborn's bearing damage, voraus-AD's `motor_commutation`).
Sielaff joins robo3er and most of voraus-AD as a dataset where V2 (no
cross-node mechanism at all) is the right default, now confirmed even
against a domain-knowledge-free attention alternative, not just against
the (unavailable) typed-relation alternative.

## What's NOT done

- No controlled ablation isolating whether the small A/B regression is
  caused by the edge loss term specifically (`LAMBDA_EDGE`) vs. general
  training variance -- would need a same-seed rerun with `LAMBDA_EDGE=0`
  to confirm.
- Not tested on Paderborn/robo3er/voraus-AD as an independently-trained
  "clean" V2.1 (their existing C/D ablation columns come from V3 runs
  jointly trained with the typed-relation head, a different confound --
  see `memory/joint-prototype-scheme-v3.md`'s Paderborn caveat). A true
  V2.1-only run on those three would let "does generic attention alone
  help" be separated from "does typed relation help," which the current
  V3 numbers can't cleanly do.

Full report: `checkpoints/sielaff/sielaff_joint_prototype_v2_1_report.json`,
model weights: `checkpoints/sielaff/sielaff_joint_prototype_v2_1.pth`.
