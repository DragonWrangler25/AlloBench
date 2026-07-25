"""Figure 2: four-panel same-information R0->R3 first-sight dissociation, one panel
per frontier model (\\ref{fig:core}).

The Haiku/Opus panels read the preserved per-seed sessions; the GPT panels read
their locked analyses.

  python scripts/plotters/plot_fig2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    RUNS, SEEDS, BLUE, ORANGE, INK,
    configure_style, load_json, save, boot_ratio_ci,
)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def r3_positions(base: str, seed: int) -> list[int]:
    """Class positions at which scripts are authored in the R3 arm ``base`` for ``seed``."""
    path = RUNS / base / f"seed_{seed}" / "sessions.jsonl"
    if not path.exists():
        return []
    line = next(line for line in path.read_text().splitlines() if line.strip())
    session = json.loads(line)
    stream = load_json(RUNS / base / f"seed_{seed}" / "stream.json")
    positions: list[int] = []
    for record in session["records"]:
        if record.get("scripts_authored"):
            positions.extend([stream[record["item_idx"]]["class_position"]] * len(record["scripts_authored"]))
    return positions


def first_sight_r3(seed: int, base: str = "arm_a1_announce_n-announced") -> float:
    positions = r3_positions(base, seed)
    if not positions:
        return np.nan
    return float(np.mean(np.asarray(positions) == 1))


def r0_positions(path: Path) -> list[int]:
    """Occurrence indices at which classes are committed in an R0 session (the ``kept`` map)."""
    if not path.exists():
        return []
    return list(load_json(path)["kept"].values())


# Per-frontier-model R0 (abstract urn) session paths and R3 (full script) arm directories.
# R0 first-sight matches Table 1; R3 first-sight matches Table 2 (tab:models).
FRONTIER_MODELS = [
    {
        "name": "Haiku 4.5",
        "r0": "urn_haiku_n-announced/seed_{seed}/session.json",
        "r3": "arm_a1_announce_n-announced",
    },
    {
        "name": "Opus 4.8",
        "r0": "urn_opus_n-announced/seed_{seed}/session.json",
        "r3": "arm_a1_announce_opus_n-announced_canonical-structure_mag100_class-bound-v1",
    },
    {
        "name": "GPT-5.4-mini",
        "r0": "economic_surface_gpt-5.4-mini-2026-03-17/R0/B_3/K_0/seed_{seed}/session.json",
        "r3": "arm_a1_announce_gpt-5.4-mini-2026-03-17_n-announced_write-budget-v2_class-bound-v1",
    },
    {
        "name": "GPT-5.6 Sol",
        "r0": "economic_surface_gpt-5.6-sol/R0/B_3/K_0/seed_{seed}/session.json",
        "r3": "arm_a1_announce_gpt-5.6-sol_n-announced_write-budget-v2_class-bound-v1",
    },
]


def model_paired_values(model: dict[str, str]) -> dict[str, np.ndarray]:
    """Per-seed first-sight percentages (R0, R3) and pooled first-sight percentages for one model."""
    r0_seed = np.asarray(
        [
            (
                float(np.mean(np.asarray(pos) == 1)) * 100
                if (pos := r0_positions(RUNS / model["r0"].format(seed=seed)))
                else np.nan
            )
            for seed in SEEDS
        ]
    )
    r3_seed = np.asarray([first_sight_r3(seed, model["r3"]) * 100 for seed in SEEDS])
    r0_all = [p for seed in SEEDS for p in r0_positions(RUNS / model["r0"].format(seed=seed))]
    r3_all = [p for seed in SEEDS for p in r3_positions(model["r3"], seed)]
    r0_pooled = 100 * np.mean(np.asarray(r0_all) == 1) if r0_all else np.nan
    r3_pooled = 100 * np.mean(np.asarray(r3_all) == 1) if r3_all else np.nan
    # Per-seed (first-sight commitments, total commitments) for the pooled-ratio seed bootstrap.
    r0_counts = [
        (sum(1 for p in pos if p == 1), len(pos))
        for seed in SEEDS
        for pos in [r0_positions(RUNS / model["r0"].format(seed=seed))]
    ]
    r3_counts = [
        (sum(1 for p in pos if p == 1), len(pos))
        for seed in SEEDS
        for pos in [r3_positions(model["r3"], seed)]
    ]
    return {
        "r0": r0_seed,
        "r3": r3_seed,
        "pooled": np.asarray([r0_pooled, r3_pooled]),
        "r0_counts": r0_counts,
        "r3_counts": r3_counts,
    }


def _dissociation_panel(ax: plt.Axes, model: dict[str, str], show_ylabel: bool) -> None:
    """Draw one model's paired R0->R3 first-sight dissociation on ``ax``."""
    values = model_paired_values(model)
    urn, tool = values["r0"], values["r3"]
    pooled = values["pooled"]

    jitter = np.linspace(-0.055, 0.055, len(SEEDS))
    for i, (u, t) in enumerate(zip(urn, tool)):
        paired = np.isfinite(u) and np.isfinite(t)
        if paired:
            ax.plot([0 + jitter[i], 1 + jitter[i]], [u, t], color="#AEB6BF",
                    linewidth=1.0, alpha=0.85, zorder=1)
        if np.isfinite(u):
            # Filled: seed commits in both frames (has a connector). Open: commits in R0
            # but realizes no R3 build (unpaired) -- the GPT under-commitment seeds.
            if paired:
                ax.scatter(0 + jitter[i], u, s=24, color=BLUE, edgecolor="white", linewidth=0.5, zorder=2)
            else:
                ax.scatter(0 + jitter[i], u, s=26, facecolor="white", edgecolor=BLUE, linewidth=1.1, zorder=2)
        if np.isfinite(t):
            ax.scatter(1 + jitter[i], t, s=24, color=ORANGE, edgecolor="white", linewidth=0.5, zorder=2)
    ax.plot([0, 1], pooled, color=INK, linewidth=2.1, zorder=3)
    # 95% seed-bootstrap CI on each pooled first-sight rate (per-arm; the paired R0->R3 difference
    # and its CI are reported in the appendix, being a difference the level plot cannot show).
    r0_ci = boot_ratio_ci(values["r0_counts"])
    r3_ci = boot_ratio_ci(values["r3_counts"])
    yerr = np.array([
        [pooled[0] - r0_ci[0], pooled[1] - r3_ci[0]],
        [r0_ci[1] - pooled[0], r3_ci[1] - pooled[1]],
    ])
    yerr = np.clip(np.nan_to_num(yerr, nan=0.0), 0.0, None)
    ax.errorbar([0, 1], pooled, yerr=yerr, fmt="none", ecolor=INK,
                elinewidth=1.4, capsize=5, capthick=1.4, zorder=3.5)
    ax.scatter([0, 1], pooled, s=95, color=[BLUE, ORANGE], edgecolor=INK, linewidth=1.2, zorder=4)
    ax.text(-0.16, pooled[0], f"{pooled[0]:.0f}%", ha="right", va="center", weight="bold", color=BLUE)
    ax.text(1.16, pooled[1], f"{pooled[1]:.0f}%", ha="left", va="center", weight="bold", color=ORANGE)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Abstract\nR0", "Script\nR3"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-5, 112)
    if show_ylabel:
        ax.set_ylabel("First-sight commitments (%)")
    ax.set_title(model["name"], loc="left", weight="bold")
    ax.grid(axis="y", color="#E8EBEE", linewidth=0.8)


def figure_core_dissociation() -> None:
    """Four-panel same-information R0->R3 dissociation, one panel per frontier model."""
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2), sharey=True, constrained_layout=True)
    for idx, (ax, model) in enumerate(zip(axes.flat, FRONTIER_MODELS)):
        _dissociation_panel(ax, model, show_ylabel=(idx % 2 == 0))
    save(fig, "fig2_core_dissociation")


def main() -> None:
    configure_style()
    figure_core_dissociation()


if __name__ == "__main__":
    main()
