"""Figure 1: the optimal allocation policy on one canonical stream (seed 2000).

Single row of the first 15 of 60 slots; annotations mark the defer-then-confirm
optimum: pass every first sight, build at a class's second occurrence, collect
later occurrences of built classes for free. Style is intentionally its own
(minimal, no axes) rather than the shared plotter style.

  python scripts/plotters/plot_fig1.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
OUT = ROOT / "figs"

GREEN = "#2E8B57"
GRAY = "#747B83"
LIGHT = "#D9DEE3"
INK = "#22252A"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def main() -> None:
    configure_style()
    stream = json.loads((RUNS / "urn_haiku_n-announced" / "seed_2000" / "stream.json").read_text())
    shown = stream[:15]
    xs = np.arange(1, len(shown) + 1)

    class_ids = sorted({item["class_id"] for item in shown})
    palette = plt.get_cmap("tab10")
    class_color = {cid: palette(i % 10) for i, cid in enumerate(class_ids)}

    fig, ax = plt.subplots(figsize=(11.2, 2.4))
    y = 1.0
    for x, item in zip(xs, shown):
        cid, pos = item["class_id"], item["class_position"]
        build = pos == 2                      # defer-then-confirm: build on 2nd occurrence
        reuse = pos > 2                       # later occurrences of built classes: free
        if build:
            ax.scatter(x, y, s=560, facecolor="none", edgecolor=INK, linewidth=1.6, zorder=4)
        ax.scatter(x, y, s=310, color=class_color[cid], edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(x, y, str(cid), ha="center", va="center", color="white", fontsize=8, weight="bold")
        if build:
            # black ring has s=560 pts^2 -> radius ~sqrt(560/pi) ~= 13.3 pts;
            # shrinkB stops the line at the ring edge, out of the colored circle.
            ax.annotate("BUILD", (x, y), xytext=(x, y + 0.62), ha="center", fontsize=9,
                        weight="bold", color=INK,
                        arrowprops={"arrowstyle": "-", "color": INK, "linewidth": 1.0,
                                    "shrinkA": 0, "shrinkB": 13.5})
        elif reuse:
            ax.text(x, y + 0.45, "✓", ha="center", fontsize=11, color=GREEN, weight="bold")
        else:
            ax.text(x, y + 0.45, "pass", ha="center", fontsize=8, color=GRAY)

    ax.set_xticks([])
    ax.set_xlim(0.3, len(shown) + 0.7)
    ax.set_ylim(0.75, 1.85)
    ax.set_yticks([])
    ax.margins(0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig1_optimal_policy.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "fig1_optimal_policy.pdf", bbox_inches="tight")
    print("wrote figs/fig1_optimal_policy.png and .pdf")


if __name__ == "__main__":
    main()
