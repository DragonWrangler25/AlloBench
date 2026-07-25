"""Figure 4: learned reserve across lexical, modality, and construction shifts, with
95% seed-bootstrap CI bars on each arm. Points/CIs are computed from the per-seed
run artifacts; the paired base-vs-RL difference CIs are reported in
Table~\\ref{tab:rl-transfer-full}.

  python scripts/plotters/plot_fig4.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    RUNS, SEEDS, BLUE, ORANGE, GREEN, GRAY,
    configure_style, load_json, save, boot_ratio_ci, pooled_ratio,
)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

VOCAB_RESKINS = ["treasure_chest", "quiver", "cauldron"]


def _rl_rung_counts(arm_root: str, rung: str) -> list[tuple[int, int]]:
    """Per-seed (first-sight commitments, total commitments) for one Qwen arm at one ladder rung.
    ``arm_root`` is 'qwen-rl-base-q8' or 'qwen-rl-a2-final'. R0/reskins/tool-call/R1/R2 read the
    kept/claimed maps; R3 uses the class-bound scorer (a build is attributed to the class it can
    execute on), matching analyze_qwen_class_bound_r3 and Table~\\ref{tab:rl-transfer-full}."""
    if rung == "R3":
        from scripts.session.skirental_scorer import (
            actions_from_session, model_builds_from_actions,
        )
        arm = f"arm_a1_announce_{arm_root}_latest_n-announced_class-bound-v1_efr4"
        out = []
        for seed in SEEDS:
            sd = RUNS / arm / f"seed_{seed}"
            session = json.loads((sd / "sessions.jsonl").read_text().splitlines()[0])
            slots = load_json(sd / "stream.json")
            builds = model_builds_from_actions(actions_from_session(session, slots))
            commits = [pos for pos in builds.values() if pos is not None]
            out.append((sum(1 for p in commits if p == 1), len(commits)))
        return out

    dirs = {
        "R0": [f"urn_{arm_root}_latest_n-announced"],
        "Reskins": [f"urn_{arm_root}_latest_n-announced_vocab-{v}" for v in VOCAB_RESKINS],
        "ToolCall": ([f"urn_tool_{arm_root}_latest_n-announced"]
                     + [f"urn_tool_{arm_root}_latest_n-announced_vocab-{v}" for v in VOCAB_RESKINS]),
        "R1": [f"claim_solver_{arm_root}_latest_n-announced_efr4"],
        "R2": [f"claim_solver_code_{arm_root}_latest_n-announced_efr4"],
    }[rung]
    field = "claimed" if rung in ("R1", "R2") else "kept"
    out = []
    for seed in SEEDS:
        pos = []
        for base in dirs:
            path = RUNS / base / f"seed_{seed}" / "session.json"
            if path.exists():
                pos.extend(load_json(path)[field].values())
        out.append((sum(1 for p in pos if p == 1), len(pos)))
    return out


def figure_rl_transfer() -> None:
    """Learned reserve across lexical, modality, and construction shifts, with 95%
    seed-bootstrap CI bars on each arm."""
    categories = ["Abstract\nurn (R0)", "Vocabulary\nreskins", "Keep/pass\ntool calls",
                  "Declarative\nclaim (R1)", "Code\nclaim (R2)", "Reusable\nscripts (R3)"]
    rungs = ["R0", "Reskins", "ToolCall", "R1", "R2", "R3"]
    base_counts = [_rl_rung_counts("qwen-rl-base-q8", r) for r in rungs]
    rl_counts = [_rl_rung_counts("qwen-rl-a2-final", r) for r in rungs]
    base = np.asarray([pooled_ratio(c) for c in base_counts], dtype=float)
    rl = np.asarray([pooled_ratio(c) for c in rl_counts], dtype=float)
    base_ci = np.asarray([boot_ratio_ci(c) for c in base_counts], dtype=float)
    rl_ci = np.asarray([boot_ratio_ci(c) for c in rl_counts], dtype=float)
    x = np.arange(len(categories))

    def yerr(vals, cis):
        return np.array([vals - cis[:, 0], cis[:, 1] - vals])

    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    ax.errorbar(x, base, yerr=yerr(base, base_ci), fmt="none", ecolor=GRAY,
                elinewidth=1.3, capsize=4, capthick=1.3, zorder=2)
    ax.errorbar(x, rl, yerr=yerr(rl, rl_ci), fmt="none", ecolor=GREEN,
                elinewidth=1.3, capsize=4, capthick=1.3, zorder=2)
    ax.plot(x, base, color=GRAY, marker="o", linewidth=1.8, markersize=7, label="Base Qwen-14B")
    ax.plot(x, rl, color=GREEN, marker="o", linewidth=2.3, markersize=8, label="RL (Run 2, A2)")
    # higher series labeled above its point, lower series below -- avoids collisions where the lines converge
    for xi, b, r in zip(x, base, rl):
        hi_val, hi_col = (b, GRAY) if b >= r else (r, GREEN)
        lo_val, lo_col = (r, GREEN) if b >= r else (b, GRAY)
        ax.text(xi, hi_val + 3, f"{hi_val:.0f}%", ha="center", va="bottom", color=hi_col, fontsize=9, weight="bold")
        lo_y = lo_val + 4 if lo_val < 12 else lo_val - 3          # keep tiny values off the axis floor
        lo_va = "bottom" if lo_val < 12 else "top"
        ax.text(xi, lo_y, f"{lo_val:.0f}%", ha="center", va=lo_va, color=lo_col, fontsize=9, weight="bold")

    ax.axvspan(-0.35, 2.35, color=BLUE, alpha=0.06)
    ax.axvspan(2.65, 5.35, color=ORANGE, alpha=0.06)
    ax.axvline(2.5, color=GRAY, linewidth=0.8, linestyle=(0, (4, 3)), alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 108)
    ax.set_ylabel("First-sight commitments (%)")
    ax.grid(axis="y", color="#E8EBEE", linewidth=0.8)
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 0.98))
    save(fig, "fig4_rl_transfer_boundary")


def main() -> None:
    configure_style()
    figure_rl_transfer()


if __name__ == "__main__":
    main()
