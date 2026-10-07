#!/usr/bin/env python3
"""M-grid AUROC (FD_JD_PD_max) + shared-prototype-count dual-axis line plots,
one PNG per dataset, sized/fonted for a later 2x2 composite in a paper."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ORANGE = "#E8721C"
DARKBLUE = "#0F2E5C"

DATA = {
    "robo_fleet": {
        "M":      [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
        "auroc":  [0.944,0.949,0.944,0.942,0.922,0.914,0.930,0.917,0.909,
                   0.910,0.916,0.901,0.894,0.909,0.910,0.897,0.916,0.902,0.900],
        "shared": [1,1,2,2,3,2,3,4,3,3,6,3,5,4,4,5,6,4,3],
    },
    "paderborn": {
        "M":      [2,4,8,16,24,32],
        "auroc":  [0.984,0.995,0.993,0.994,0.999,0.996],
        "shared": [1,2,4,3,4,4],
    },
    "alfa": {
        "M":      [2,4,8,16,24,32],
        "auroc":  [0.869,0.886,0.884,0.853,0.869,0.895],
        "shared": [1,2,2,4,5,4],
    },
    "me_ad": {
        "M":      [2,4,8,16,24,32,48],
        "auroc":  [0.757,0.753,0.796,0.804,0.809,0.831,0.826],
        "shared": [1,2,4,8,11,11,7],
    },
}

TITLES = {
    "robo_fleet": "robo_fleet",
    "paderborn": "Paderborn",
    "alfa": "ALFA",
    "me_ad": "ME-AD",
}

FONT = 20
TICK = 17
LEGEND = 16

for name, d in DATA.items():
    fig, ax1 = plt.subplots(figsize=(6.4, 5.0))
    x = list(range(len(d["M"])))
    # thin tick labels for the dense robo_fleet grid (19 points) so they don't collide
    show = set(d["M"]) if len(d["M"]) <= 10 else {2,4,6,8,10,12,14,16,18,20}

    ax1.set_xlabel("Number of prototypes (M)", fontsize=FONT)
    l1, = ax1.plot(x, d["auroc"], color=ORANGE, linewidth=3,
                    marker="o", markersize=8, label="AUROC")
    ax1.tick_params(axis="y", labelcolor=ORANGE, labelsize=TICK)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(m) if m in show else "" for m in d["M"]], fontsize=TICK)
    ax1.spines["top"].set_visible(False)

    ax2 = ax1.twinx()
    l2, = ax2.plot(x, d["shared"], color=DARKBLUE, linewidth=3,
                    marker="s", markersize=8, label="Shared prototypes")
    ax2.tick_params(axis="y", labelcolor=DARKBLUE, labelsize=TICK)
    ax2.spines["top"].set_visible(False)
    ax2.set_ylim(0, max(d["shared"]) * 1.35)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax1.set_title(TITLES[name], fontsize=FONT + 2, fontweight="bold", pad=10)

    lines = [l1, l2]
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower right",
               fontsize=LEGEND, frameon=False)

    fig.tight_layout()
    out = f"/home/roboserver/Projects/FLMemGraph/report/figures/protogrid_{name}.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print("saved", out)
