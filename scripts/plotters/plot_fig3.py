"""Figure 3: abstract (R0) versus code-required (R2) Score, per model, with 95%
seed-bootstrap intervals shown as shaded rectangles (R0 CI on x, R2 CI on y).

  python scripts/plotters/plot_fig3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    RUNS, SEEDS, BLUE, PURPLE, GREEN, ORANGE,
    configure_style, load_json, save, boot_ratio_ci, pooled_ratio,
)

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402


# Per-model R0/R2 Score sources (competitive ratio vs the exact hindsight net optimum, B=3, K=0).
# Verified to reproduce Table~\ref{tab:score} exactly via the canonical economic_surface scorer.
SCORE_SOURCES = {
    "Haiku 4.5":    {"R0": "urn_haiku_n-announced/seed_{seed}",
                     "R2": "claim_solver_code_haiku_n-announced/seed_{seed}",
                     "color": BLUE},
    "Opus 4.8":     {"R0": "urn_opus_n-announced/seed_{seed}",
                     "R2": "economic_surface_opus/R2/B_3/K_0/seed_{seed}",
                     "color": PURPLE},
    "GPT-5.4-mini": {"R0": "economic_surface_gpt-5.4-mini-2026-03-17/R0/B_3/K_0/seed_{seed}",
                     "R2": "economic_surface_gpt-5.4-mini-2026-03-17/R2/B_3/K_0/seed_{seed}",
                     "color": GREEN},
    "GPT-5.6 Sol":  {"R0": "economic_surface_gpt-5.6-sol/R0/B_3/K_0/seed_{seed}",
                     "R2": "economic_surface_gpt-5.6-sol/R2/B_3/K_0/seed_{seed}",
                     "color": ORANGE},
}


def score_pairs(dir_tmpl: str, K: int = 0, B: int = 3) -> list[tuple[float, float]]:
    """Per-seed (model net utility, hindsight net optimum) at (B, K), the numerator/denominator of
    the competitive-ratio Score. The optimum reads the shared canonical magnitude-100 stream (paired
    across arms); the model's net utility reads the arm's own session."""
    from scripts.economic.economic_surface import (
        net_score, reference_net_score, hindsight_net_optimal_builds,
    )
    pairs = []
    for seed in SEEDS:
        row = load_json(RUNS / dir_tmpl.format(seed=seed) / "session.json")
        slots = load_json(RUNS / "urn_haiku_n-announced" / f"seed_{seed}" / "stream.json")
        util = net_score(row, K)
        opt = reference_net_score(B, K, seed, slots,
                                  builds=hindsight_net_optimal_builds(slots, B, K))
        pairs.append((util, opt))
    return pairs


def figure_capability_map() -> None:
    """Abstract (R0) versus code-required (R2) Score, per model, with 95% seed-bootstrap
    intervals shown as shaded rectangles (R0 CI on x, R2 CI on y)."""
    lo, hi = 45, 100
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    # Shade the region where the code-required Score falls below the abstract Score.
    ax.fill_between([lo, hi], [lo, hi], lo, color=ORANGE, alpha=0.06)
    ax.plot([lo, hi], [lo, hi], color="#9CA3AA", linestyle="--", linewidth=1.2)

    offsets = {
        "Haiku 4.5": (-7, -7),
        "Opus 4.8": (10, 4),
        "GPT-5.4-mini": (10, -4),
        "GPT-5.6 Sol": (10, 5),
    }
    aligns = {"Haiku 4.5": ("right", "top")}
    for name, src in SCORE_SOURCES.items():
        color = src["color"]
        r0 = score_pairs(src["R0"])
        r2 = score_pairs(src["R2"])
        x, y = pooled_ratio(r0), pooled_ratio(r2)
        x_lo, x_hi = boot_ratio_ci(r0)
        y_lo, y_hi = boot_ratio_ci(r2)
        # 95% x 95% rectangle: marginal CIs on each axis (the two arms are on the same seeds, so
        # the true joint region is correlated; the rectangle shows the per-axis intervals).
        ax.add_patch(mpatches.Rectangle(
            (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
            facecolor=color, edgecolor="none", alpha=0.15, zorder=2))
        ax.scatter(x, y, s=210, marker="o", facecolor=color, edgecolor=color,
                   linewidth=1.8, zorder=3)
        dx, dy = offsets[name]
        ha, va = aligns.get(name, ("left", "baseline"))
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=13, weight="bold", ha=ha, va=va)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Abstract (R0) Score (%)", fontsize=17)
    ax.set_ylabel("Code-required (R2) Score (%)", fontsize=17)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(color="#ECEFF1", linewidth=0.8)
    save(fig, "fig3_capability_map")


def main() -> None:
    configure_style()
    figure_capability_map()


if __name__ == "__main__":
    main()
