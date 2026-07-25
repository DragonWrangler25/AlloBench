"""Figure 6: training curves for the Qwen post-training case study.

Left: mean training reward per outer step for every run executed under this
project's reward-training protocol -- the four flat episode-scalar GRPO pilots
(v2-v5) and the successful per-decision PPO run (run 2, N-disclosed). Right:
run 2's per-step first-sight commitment rate, with run 1's archived endpoint
summaries overlaid (its per-step history was not preserved).

Data sources: runs/rl_urn_pilot_a2/MORNING_REPORT.md (run 2 per-step reward and
behavior histories), the PPO credit-assignment spec (v2-v5 reward
histories and run 1 endpoint summaries, quoted from the training logs).

  python scripts/plotters/plot_fig6.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figs"

NAVY = "#28536B"
GREEN = "#3E8E7E"
GRAY = "#747B83"
LIGHT = "#D9DEE3"
INK = "#22252A"

# Episode-scalar GRPO pilots (superseded recipe): mean training reward (balls) per outer step.
PILOTS = {
    "v2": [36.42, 37.64, 36.61, 35.63, 36.51, 35.95, 34.62],
    "v3": [37.58, 37.63, 36.05, 35.75, 38.39],
    "v4": [37.43, 36.47, 36.53, 35.18, 36.54],
    "v5": [37.89, 35.62, 36.76, 36.52, 36.46, 37.69, 35.50, 37.36],
}

# Run 2 (per-decision PPO + privileged critic, N-disclosed rollouts), 20 outer steps.
RUN2_REWARD = [37.75, 36.95, 35.96, 37.44, 38.28, 36.70, 37.34, 38.89, 37.14, 36.38,
               36.47, 35.93, 38.17, 38.64, 40.44, 40.03, 38.27, 39.54, 40.63, 39.79]
RUN2_FIRST_SIGHT = [69, 71, 71, 68, 70, 68, 69, 67, 63, 61, 58, 56, 46, 49, 41, 26, 25, 22, 15, 14]

# Run 1 (same recipe, no N disclosure): only endpoint summaries survive in the archived
# manifest/spec quotes -- first-sight 77% -> 42% over 20 steps.
RUN1_ENDPOINTS = [(0, 77), (19, 42)]

# Mean heuristic-baseline rewards over run 2's training batches (printed per step in the log).
EAGER_MEAN = 34.9
WAIT2_MEAN = 43.3


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": "#6B7075",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def main() -> None:
    configure_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4))

    label_offsets = {"v2": (4, -8), "v3": (4, 2), "v4": (4, -4), "v5": (4, 6)}
    for name, hist in PILOTS.items():
        ax1.plot(range(len(hist)), hist, color=GRAY, linewidth=1.4, alpha=0.55, zorder=2)
        ax1.annotate(name, (len(hist) - 1, hist[-1]), textcoords="offset points",
                     xytext=label_offsets[name], fontsize=8, color=GRAY)
    ax1.plot(range(len(RUN2_REWARD)), RUN2_REWARD, color=NAVY, linewidth=2.0,
             marker="o", markersize=3.5, zorder=3, label="run 2 (per-decision PPO)")
    ax1.axhline(EAGER_MEAN, color=GRAY, linestyle=":", linewidth=1.1)
    ax1.axhline(WAIT2_MEAN, color=GRAY, linestyle="--", linewidth=1.1)
    ax1.annotate("eager heuristic", (19.4, EAGER_MEAN), fontsize=8, color=GRAY,
                 va="bottom", ha="right", xytext=(0, 2), textcoords="offset points")
    ax1.annotate("wait-2 heuristic", (19.4, WAIT2_MEAN), fontsize=8, color=GRAY,
                 va="bottom", ha="right", xytext=(0, 2), textcoords="offset points")
    ax1.set_xlabel("outer step")
    ax1.set_ylabel("mean training reward (items)")
    ax1.set_title("All runs: episode-scalar pilots stay flat")
    ax1.set_xlim(-0.5, 19.9)
    ax1.set_ylim(33.5, 44.5)
    ax1.set_xticks(range(0, 20, 5))
    ax1.legend(frameon=False, fontsize=8, loc="upper left")

    ax2.plot(range(len(RUN2_FIRST_SIGHT)), RUN2_FIRST_SIGHT, color=NAVY, linewidth=2.0,
             marker="o", markersize=3.5, zorder=3, label="run 2 (N disclosed)")
    xs, ys = zip(*RUN1_ENDPOINTS)
    ax2.scatter(xs, ys, color=GREEN, marker="D", s=32, zorder=4,
                label="run 1 endpoints (no N)")
    ax2.set_xlabel("outer step")
    ax2.set_ylabel("first-sight commitment (%)")
    ax2.set_title("Reserve acquisition during training")
    ax2.set_xlim(-0.5, 19.9)
    ax2.set_ylim(0, 85)
    ax2.set_xticks(range(0, 20, 5))
    ax2.legend(frameon=False, fontsize=8, loc="lower left")

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", {"dpi": 240}), ("pdf", {})):
        path = OUT / f"fig6_rl_training_curves.{ext}"
        fig.savefig(path, bbox_inches="tight", **kw)
        print(f"wrote {path.relative_to(ROOT)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
