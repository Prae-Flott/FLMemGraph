---
name: joint-prototype-scheme-b
description: Final conclusions for Joint Prototype Memory "Scheme B" (prototype + per-node deviation, no edges) across all 4 datasets tested
---

# Scheme B: Joint Prototype Memory, no edges -- final cross-dataset
# conclusions (consolidated 2026-08-11)

## What it is

`src/joint_prototype_model.py`'s `SharedEncoder` + `JointPrototypeMemory`
only -- no edge/relation head at all. Produces two signals:
- **A** (`d_G`): device-level novelty -- "has the WHOLE multi-sensor
  joint state been seen before" (distance to the nearest of M learned
  joint prototypes, each a full `[N, D]` snapshot across all N nodes).
- **B** (`A` combined with `s_node`'s per-node max): "has any ONE
  specific signal's value, within this matched operating regime, drifted
  from what it normally looks like."

Mechanism-agnostic by design: needs no declared physics relations, no
domain knowledge of how the system works, just normal data to fit
prototypes against. This makes it runnable on ANY dataset immediately,
unlike Scheme V3 (`memory/joint-prototype-scheme-v3.md`), which needs at
least some verified physical relation to declare as an edge.

## Cross-dataset results (mean AUROC, B vs. the next-best available signal)

| dataset | B (mean AUROC) | strongest edge-based signal | winner |
|---|---|---|---|
| Sielaff (10 reverse-vending machines, 39 sensors, 5 fault types) | **0.978** | n/a -- no physics prior exists for this dataset | **B**, by a wide margin over the existing plain-GDN baseline (0.887, +0.091) |
| robo3er (5-robot fleet, 7 kinematic-chain nodes, 4 fault types) | **0.942** | 0.826 (generic attention) | **B**, clearly |
| voraus-AD (6-DOF arm, 66 nodes, 12 fault categories) | **0.756** | 0.732 (generic attention) | **B**, wins/ties 11 of 12 categories |
| Paderborn (bearing, 6 nodes, 26 damaged bearings) | 0.750 | 0.904 (generic attention) | edge signal, clearly |

**B is the best or near-best available signal on 3 of 4 datasets**, and on
the 4th (Paderborn) it's still a reasonable second place, not a
collapse. The one dataset where it loses clearly (Paderborn) has
textbook, literature-verified physical relations (bearing defect
frequencies, motor current/torque coupling) that a real fault
mechanism actually breaks -- see `memory/joint-prototype-scheme-v3.md`.

## Why B wins where it wins

Per-category analysis (robo3er, voraus-AD) shows a consistent pattern:
**B wins on faults that manifest as a single signal's value drifting out
of its normal range** (robo3er's `stuck`/`broken pipe`, voraus-AD's
collisions/can-weight/can-loss/axis-friction), and **loses specifically
on faults whose failure mechanism IS a documented relationship between
two signals breaking** (voraus-AD's `motor_commutation` -- current stops
predicting torque the way it normally does; Paderborn's bearing damage --
vibration/force/torque/current lose their normal coupling). This is a
property of each dataset's fault MECHANISMS, not of the model
architecture -- confirmed independently on 4 different datasets with 4
different physical domains (robot arm collisions, can-handling faults,
bearing damage, vending-machine faults).

## Practical implication

**Run Scheme B first on any new dataset**, before investing in physics-
relation analysis for Scheme V3 -- it needs no domain knowledge, is cheap
to run, and is a strong default across every dataset mechanism type
tested so far except the one with the clearest textbook physics
(Paderborn). Sielaff is the cleanest demonstration: zero verified
physical relations exist for it, yet B alone beats the dataset's
existing tuned GDN baseline by +0.091 mean AUROC.

## Where results/code live

- Sielaff: `benchmark/run_sielaff_joint_prototype_b.py`,
  `checkpoints/sielaff/sielaff_joint_prototype_b_report.json` (+ `.pth`).
- robo3er / voraus-AD / Paderborn: B is one of the ablation columns
  (`A_prototype_only`, `B_prototype_plus_node`) inside each dataset's
  Scheme V3 run script and report -- see `memory/joint-prototype-scheme-v3.md`
  for those files' locations. No separate edge-free script was built for
  these three; B's numbers come from disabling V3's edge/typed-edge
  signals in the same run, not a distinct model.
