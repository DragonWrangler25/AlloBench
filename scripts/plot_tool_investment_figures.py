"""Generate the main figures for the online tool-investment paper.

Outputs PNG and PDF files under ``figs/``.
The Haiku panels read the preserved per-seed sessions; economic and GPT panels
read their locked analyses. The RL transfer panel uses the publication values
reported in ``docs/rl-phase1-results.md``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "figs"
SEEDS = list(range(2000, 2024))

NAVY = "#28536B"
BLUE = "#3C78A8"
ORANGE = "#E07A3F"
GREEN = "#3E8E7E"
PURPLE = "#7566A8"
GRAY = "#747B83"
LIGHT = "#D9DEE3"
INK = "#22252A"


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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png.relative_to(ROOT)} and {pdf.relative_to(ROOT)}")


def first_sight_from_mapping(path: Path, field: str) -> float:
    session = load_json(path)
    positions = list(session[field].values())
    if not positions:
        return np.nan
    return float(np.mean(np.asarray(positions) == 1))


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
    return {"r0": r0_seed, "r3": r3_seed, "pooled": np.asarray([r0_pooled, r3_pooled])}


def figure_task() -> None:
    """Task-isomorphism schematic: one latent class stream, two decision frames.

    Lives in the Benchmark construction section (\\ref{fig:frames})."""
    stream = load_json(RUNS / "urn_haiku_n-announced" / "seed_2000" / "stream.json")

    fig, ax_task = plt.subplots(figsize=(7.6, 3.2), constrained_layout=True)

    shown = stream[:12]
    class_ids = sorted({item["class_id"] for item in shown})
    palette = plt.get_cmap("tab10")
    class_color = {cid: palette(i % 10) for i, cid in enumerate(class_ids)}
    xs = np.arange(1, len(shown) + 1)

    for y, label in [(1.0, "Abstract keep/pass"), (0.0, "Reusable script")]:
        for x, item in zip(xs, shown):
            face = class_color[item["class_id"]]
            if y == 1.0:
                ax_task.scatter(x, y, s=310, color=face, edgecolor="white", linewidth=1.2, zorder=3)
                ax_task.text(x, y, str(item["class_id"]), ha="center", va="center",
                             color="white", fontsize=8, weight="bold")
            else:
                ax_task.scatter(x, y, s=310, marker="s", color=face, edgecolor="white",
                                linewidth=1.2, zorder=3)
                ax_task.text(x, y, f"P{item['class_id']}", ha="center", va="center",
                             color="white", fontsize=7.5, weight="bold")
        ax_task.text(0.1, y, label, ha="right", va="center", fontsize=10, weight="bold")

    for x in xs:
        ax_task.plot([x, x], [0.18, 0.82], color=LIGHT, linewidth=0.8, zorder=0)
    ax_task.text(
        6.5,
        0.50,
        "same latent class stream",
        ha="center",
        va="center",
        fontsize=9,
        color=GRAY,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
    )
    ax_task.set_xlim(0.2, 12.8)
    ax_task.set_ylim(-0.55, 1.55)
    ax_task.set_xticks(xs)
    ax_task.set_xlabel("Stream position (first 12 of 60)")
    ax_task.set_yticks([])
    ax_task.set_title("One allocation problem, two decision frames", loc="left", weight="bold")
    for spine in ax_task.spines.values():
        spine.set_visible(False)
    save(fig, "fig_task_frames")


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


def figure_code() -> None:
    """Four-panel same-information R0->R3 dissociation, one panel per frontier model.

    Placed where Figure 1 was, in Failures at full construction (\\ref{fig:core})."""
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2), sharey=True, constrained_layout=True)
    for idx, (ax, model) in enumerate(zip(axes.flat, FRONTIER_MODELS)):
        _dissociation_panel(ax, model, show_ylabel=(idx % 2 == 0))
    fig.suptitle(
        "No frontier model preserves abstract allocation at full construction",
        x=0.01,
        ha="left",
        weight="bold",
        fontsize=12,
    )
    save(fig, "fig_core_dissociation")


def figure_ladder() -> None:
    """Figure 2: framing ladder across the four frontier models.

    Pooled first-sight commitment (%) among realized commitments at each of the
    four rungs (R0, R1, R2, R3), one line per model. Values are the audited
    panel numbers of Table~\\ref{tab:ladder} (Section~\\ref{sec:ladder-results}):
    the R1 (declarative-claim) column is from runs/claim_solver_<model>_n-announced/;
    R0, R2, and R3 from the gate, economic (R2c), and construction arms."""
    labels = ["R0", "R1", "R2", "R3"]
    descriptions = [
        "abstract\nkeep/pass",
        "declarative\nclaim",
        "code-required\nclaim",
        "full script\nconstruction",
    ]
    panel = {
        "Haiku 4.5":    [28.0, 33.0, 100.0, 100.0],
        "Opus 4.8":     [4.0, 0.0, 91.7, 98.6],
        "GPT-5.4-mini": [15.3, 29.0, 85.0, 88.0],
        "GPT-5.6 Sol":  [23.6, 0.0, 38.9, 100.0],
    }
    cmap = plt.get_cmap("tab10")
    model_color = {name: cmap(i) for i, name in enumerate(panel)}
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9.4, 5.4), constrained_layout=True)
    # Shade the required-code transition (R1 -> R2): the trigger for Haiku,
    # Opus, and GPT-5.4-mini.
    ax.axvspan(1.5, 2.5, color=ORANGE, alpha=0.07, zorder=0)
    for name, ys in panel.items():
        ax.plot(x, ys, color=model_color[name], linewidth=2.2, marker="o",
                markersize=8, markeredgecolor="white", markeredgewidth=1.0,
                zorder=3, label=name)
    ax.annotate(
        "required code emission\n(R1 → R2)",
        xy=(2, 92), xytext=(1.30, 60),
        arrowprops={"arrowstyle": "->", "color": ORANGE, "linewidth": 1.3},
        color=ORANGE, fontsize=9, weight="bold", ha="center",
    )
    ax.annotate(
        "Sol collapses only at\nfull construction",
        xy=(3, 100), xytext=(2.5, 55),
        arrowprops={"arrowstyle": "->", "color": GRAY, "linewidth": 1.1},
        color=GRAY, fontsize=8.5, ha="center",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{label}\n{desc}" for label, desc in zip(labels, descriptions)])
    ax.set_xlim(-0.35, 3.35)
    ax.set_ylim(-4, 112)
    ax.set_ylabel("First-sight commitments (%)")
    ax.set_title("Required code emission triggers eager commitment across the panel",
                 loc="left", weight="bold")
    ax.grid(axis="y", color="#E8EBEE", linewidth=0.8)
    ax.legend(loc="center left", frameon=False, fontsize=9)
    ax.text(
        0.0,
        -0.16,
        "Pooled first-sight percentage among realized commitments, 24 paired A2 "
        "streams. GPT-5.4-mini R1 is among its few claims (0.71/seed); GPT R3 "
        "conditional on commitment.",
        transform=ax.transAxes,
        fontsize=8.0,
        color=GRAY,
    )
    save(fig, "fig2_framing_ladder")


def figure_economic_elasticity() -> None:
    """Figure 3: R0, R2c, and exact hindsight optimum over visible charges."""
    summary = load_json(RUNS / "economic_surface_haiku" / "analysis_n24.json")
    charges = [0, 10, 24]
    budgets = [1, 3, 5]
    fig, axes = plt.subplots(1, 3, figsize=(11.1, 4.2), sharey=True, constrained_layout=True)

    for ax, budget in zip(axes, budgets):
        r0 = [summary[f"R0:B{budget}:K{k}"]["model_fs_hazard"] * 100 for k in charges]
        r2c = [summary[f"R2c:B{budget}:K{k}"]["model_fs_hazard"] * 100 for k in charges]
        opt = [summary[f"R0:B{budget}:K{k}"]["reference_fs_hazard"] * 100 for k in charges]
        ax.plot(charges, opt, marker="o", color=GRAY, linewidth=1.8, linestyle="--", label="Hindsight opt*")
        ax.plot(charges, r0, marker="o", color=BLUE, linewidth=2.2, label="Abstract R0")
        ax.plot(charges, r2c, marker="o", color=ORANGE, linewidth=2.2, label="Code-required R2c")
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
    save(fig, "fig3_economic_elasticity")


def figure_capability_map() -> None:
    """Figure 4: abstract competence versus preservation during construction."""
    points = [
        ("Haiku 4.5", 28.0, 100.0, ORANGE, "o"),
        ("Opus 4.8", 4.0, 98.6, ORANGE, "o"),
        ("GPT-5.4-mini", 15.3, 88.0, PURPLE, "o"),
        ("GPT-5.6 Sol", 23.6, 100.0, PURPLE, "o"),
        ("Qwen-14B base", 73.6, 90.9, GRAY, "s"),
        ("Qwen-14B RL", 34.7, 94.3, GRAY, "s"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    ax.fill_between([0, 100], [0, 100], [100, 100], color=ORANGE, alpha=0.06)
    ax.plot([0, 100], [0, 100], color="#9CA3AA", linestyle="--", linewidth=1.2)
    ax.plot([0, 100], [20, 100], color=GREEN, alpha=0)

    offsets = {
        "Haiku 4.5": (10, 6),
        "Opus 4.8": (6, -8),
        "GPT-5.4-mini": (7, -3),
        "GPT-5.6 Sol": (8, -14),
        "Qwen-14B base": (-46, 12),
        "Qwen-14B RL": (8, -20),
    }
    for name, x, y, color, marker in points:
        hollow = "†" in name
        ax.scatter(
            x,
            y,
            s=105,
            marker=marker,
            facecolor="white" if hollow else color,
            edgecolor=color,
            linewidth=1.8,
            zorder=3,
        )
        dx, dy = offsets[name]
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=9, weight="bold")

    ax.text(65, 52, "cross-frame preservation\n(no tested model)", color="#9CA3AA", ha="center",
            fontsize=9.5, style="italic")
    ax.text(46, 73, "constructive\nsuppression", color=ORANGE, ha="center", fontsize=10, weight="bold")
    ax.text(83, 82, "policy eager\nin both frames", color=GRAY, ha="center", fontsize=9)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Abstract (R0) first-sight commitments (%)")
    ax.set_ylabel("Full-construction (R3) first-sight commitments (%)")
    ax.set_title("No tested model preserves allocation at full construction", loc="left", weight="bold")
    ax.grid(color="#ECEFF1", linewidth=0.8)
    ax.text(
        0.0,
        -0.22,
        "Every constructive endpoint is R3 (magnitude-100, same problem instances). GPT entries are conditional\n"
        "on commitment: GPT-5.4-mini realizes 34.7% of write-budget slots, Sol 73.6% (front-loaded into each\n"
        "session's opening problems). Qwen points: one matched A2 panel (24 held-out streams, q8_0); base is\n"
        "eager in both frames, the RL-trained policy reserves abstractly but not during construction.",
        transform=ax.transAxes,
        fontsize=8.2,
        color=GRAY,
    )
    save(fig, "fig4_capability_map")


def figure_rl_transfer() -> None:
    """Figure 5: learned reserve across lexical, modality, and construction shifts."""
    categories = ["Abstract urn", "Vocabulary\nreskins", "Keep/pass\ntool calls", "Reusable\nscripts"]
    base = np.asarray([73.6, 65.3, 87.2, 90.9], dtype=float)
    rl = np.asarray([34.7, 29.2, 50.0, 94.3], dtype=float)
    x = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for xi, b, r in zip(x, base, rl):
        ax.plot([xi, xi], [r, b], color=LIGHT, linewidth=5, solid_capstyle="round", zorder=1)
    ax.plot(x, base, color=GRAY, marker="o", linewidth=1.8, markersize=7, label="Base Qwen-14B")
    ax.plot(x, rl, color=GREEN, marker="o", linewidth=2.3, markersize=8, label="RL-final")
    for xi, value in zip(x, base):
        offset = -5 if value >= 95 else 4
        va = "top" if value >= 95 else "bottom"
        ax.text(xi, value + offset, f"{value:.0f}%", ha="center", va=va,
                color=GRAY, fontsize=9, weight="bold")
    for xi, value in zip(x, rl):
        offset = -6
        ax.text(xi, value + offset, f"{value:.0f}%", ha="center", va="top", color=GREEN, fontsize=9, weight="bold")

    ax.axvspan(-0.35, 1.35, color=GREEN, alpha=0.06)
    ax.axvspan(2.65, 3.35, color=ORANGE, alpha=0.06)
    ax.text(0.5, 8, "reserve acquired and lexically generalized", ha="center", color=GREEN, fontsize=8.5)
    ax.text(3, 8, "no activation", ha="center", color=ORANGE, fontsize=8.5, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 108)
    ax.set_ylabel("First-sight commitments (%)")
    ax.set_title("Reward-learned reserve remains context-bound", loc="left", weight="bold")
    ax.grid(axis="y", color="#E8EBEE", linewidth=0.8)
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 0.98))
    ax.text(
        0.0,
        -0.20,
        "All panels: the same 24 held-out A2 streams (seeds 2000-2023), matched q8_0 quantization; "
        "the script panel is class-bound-r3-v1 with EFR4.",
        transform=ax.transAxes,
        fontsize=8.3,
        color=GRAY,
    )
    save(fig, "fig5_rl_transfer_boundary")


def keep_rate_by_occurrence(tag: str, max_position: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Pooled P(KEEP) by class_position (1st/2nd/... sighting of a color), across all 24 seeds'
    session.json transcripts for the given served Ollama tag. Positions beyond max_position are
    folded into the last bucket."""
    keeps = np.zeros(max_position, dtype=int)
    totals = np.zeros(max_position, dtype=int)
    for seed in SEEDS:
        path = RUNS / f"urn_{tag}_latest_n-announced" / f"seed_{seed}" / "session.json"
        if not path.exists():
            continue
        session = load_json(path)
        for rec in session["transcript"]:
            pos = min(rec["class_position"], max_position) - 1
            totals[pos] += 1
            if rec["decision"] == "KEEP":
                keeps[pos] += 1
    return keeps, totals


def figure_reserve_policy() -> None:
    """Figure 6: the learned reserve policy's actual decision rule -- P(keep) by occurrence count,
    base vs RL-final, pooled over the paired 24-seed held-out urn eval."""
    max_position = 5
    positions = np.arange(1, max_position + 1)
    base_k, base_n = keep_rate_by_occurrence("qwen-rl-base-q8", max_position)
    rl_k, rl_n = keep_rate_by_occurrence("qwen-rl-urn-final", max_position)
    base_rate = 100 * base_k / base_n
    rl_rate = 100 * rl_k / rl_n

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    fig.subplots_adjust(top=0.90, bottom=0.24)
    ax.plot(positions, base_rate, color=GRAY, marker="o", linewidth=1.8, markersize=7,
             label="Base Qwen-14B")
    ax.plot(positions, rl_rate, color=GREEN, marker="o", linewidth=2.3, markersize=8,
             label="RL-final")
    # only callout the two positions that carry the story; 3+ have small, noisy per-bucket n
    ax.annotate(f"{base_rate[0]:.0f}%  ({base_k[0]}/{base_n[0]})", (positions[0], base_rate[0]),
                xytext=(8, 14), textcoords="offset points", ha="left", fontsize=9, color=GRAY,
                weight="bold")
    ax.annotate(f"{rl_rate[0]:.0f}%  ({rl_k[0]}/{rl_n[0]})", (positions[0], rl_rate[0]),
                xytext=(8, -18), textcoords="offset points", ha="left", fontsize=9, color=GREEN,
                weight="bold")
    ax.annotate(f"{base_rate[1]:.0f}%  ({base_k[1]}/{base_n[1]})", (positions[1], base_rate[1]),
                xytext=(0, 14), textcoords="offset points", ha="center", fontsize=9, color=GRAY,
                weight="bold")
    ax.annotate(f"{rl_rate[1]:.0f}%  ({rl_k[1]}/{rl_n[1]})", (positions[1], rl_rate[1]),
                xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9, color=GREEN,
                weight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels([str(p) for p in positions[:-1]] + [f"{max_position}+"])
    ax.set_xlabel("Occurrence count of this color within the stream", labelpad=10)
    ax.set_ylabel("P(KEEP) (%)")
    ax.set_xlim(0.5, max_position + 0.5)
    ax.set_ylim(0, 100)
    ax.set_title("The learned policy defers, then confirms", loc="left", weight="bold")
    ax.grid(axis="y", color="#ECEFF1", linewidth=0.8)
    ax.legend(frameon=False, loc="upper center")
    fig.text(
        0.02, 0.01,
        "Pooled decisions across the paired 24-seed held-out urn eval (seeds 2000-2023, A2,\n"
        "q8_0). Base keeps eagerly from the first sighting onward; RL-final passes on the first sighting\n"
        "and keeps once recurrence is confirmed on the second -- the reserve policy's decision rule made\n"
        "visible, not just its aggregate first-sight rate. Positions 3+ have small per-bucket n (2-8\n"
        "decisions) and are a noisy tail, not part of the claim.",
        fontsize=8.2, color=GRAY, va="bottom",
    )
    save(fig, "fig6_reserve_policy_shape")


def main() -> None:
    configure_style()
    figure_task()
    figure_code()
    figure_ladder()
    figure_economic_elasticity()
    figure_capability_map()
    figure_rl_transfer()
    figure_reserve_policy()


if __name__ == "__main__":
    main()
