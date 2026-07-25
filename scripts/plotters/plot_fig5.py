"""Figure 5: first-sight hazard for R0, R2, and the exact hindsight optimum over
visible build charges, one panel per build budget.

  python scripts/plotters/plot_fig5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    RUNS, BLUE, ORANGE, GRAY,
    configure_style, load_json, save,
)

import matplotlib.pyplot as plt  # noqa: E402


def figure_economic_elasticity() -> None:
    """First-sight hazard for R0, R2, and exact hindsight optimum over visible charges."""
    summary = load_json(RUNS / "economic_surface_haiku" / "analysis_n24.json")
    charges = [0, 10, 24]
    budgets = [1, 3, 5]
    fig, axes = plt.subplots(1, 3, figsize=(11.1, 4.2), sharey=True, constrained_layout=True)

    for ax, budget in zip(axes, budgets):
        r0 = [summary[f"R0:B{budget}:K{k}"]["model_fs_hazard"] * 100 for k in charges]
        r2 = [summary[f"R2:B{budget}:K{k}"]["model_fs_hazard"] * 100 for k in charges]
        opt = [summary[f"R0:B{budget}:K{k}"]["reference_fs_hazard"] * 100 for k in charges]
        ax.plot(charges, opt, marker="o", color=GRAY, linewidth=1.8, linestyle="--", label="Hindsight opt*")
        ax.plot(charges, r0, marker="o", color=BLUE, linewidth=2.2, label="Abstract R0")
        ax.plot(charges, r2, marker="o", color=ORANGE, linewidth=2.2, label="Code-required R2")
        ax.set_title(f"Build budget B={budget}", weight="bold")
        ax.set_xticks(charges)
        ax.set_xticklabels(["0\nbuild", "10\nselective", "24\nnever"])
        ax.set_xlabel("Visible build charge K")
        ax.set_ylim(-4, 106)
        ax.grid(axis="y", color="#E8EBEE", linewidth=0.8)
        if budget == 1:
            ax.set_ylabel("First-sight commitment hazard (%)")

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False)
    fig.suptitle(
        "Abstract allocation responds to price; code-required commitment does not",
        x=0.02,
        ha="left",
        weight="bold",
        fontsize=12,
    )
    fig.text(
        0.5,
        -0.035,
        "Haiku · 24 paired streams per cell · opt* is the exact hindsight net optimum, not an online policy",
        ha="center",
        fontsize=8.5,
        color=GRAY,
    )
    save(fig, "fig5_economic_elasticity")


def main() -> None:
    configure_style()
    figure_economic_elasticity()


if __name__ == "__main__":
    main()
