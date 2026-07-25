"""Shared style, paths, and generic data helpers for the per-figure plotters.

Each ``plot_figN.py`` in this directory imports what it needs from here and holds
only its own figure-specific code. Importing this module sets the Matplotlib
backend to ``Agg`` and puts the repo root on ``sys.path`` so the
``scripts.<theme>.*`` scorers (e.g. ``scripts.session.skirental_scorer``,
``scripts.economic.economic_surface``) import when a plotter is run as a file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
OUT = ROOT / "figs"
SEEDS = list(range(2000, 2024))
sys.path.insert(0, str(ROOT))  # so scripts.<theme>.* import when run as a file

# 95% seed-bootstrap: resample the 24 paired seeds with replacement, re-pool the ratio.
# Matches the paper's estimand (Appendix, "difference the re-pooled ratios", RNG seed 0).
N_BOOT = 10000
BOOT_SEED = 0

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


def boot_ratio_ci(pairs: list[tuple[float, float]],
                  n_boot: int = N_BOOT, rng_seed: int = BOOT_SEED) -> tuple[float, float]:
    """95% percentile CI for a pooled ratio ``100 * sum(num)/sum(den)``, resampling the per-seed
    ``(num, den)`` pairs with replacement. Seeds contributing ``den == 0`` (e.g. zero-commit R3
    seeds) enter the resample but add nothing to numerator or denominator, matching the pooled
    point estimate. Returns ``(nan, nan)`` for an all-empty arm."""
    num = np.asarray([p[0] for p in pairs], dtype=float)
    den = np.asarray([p[1] for p in pairs], dtype=float)
    n = len(pairs)
    if n == 0 or den.sum() == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = 100.0 * num[idx].sum(axis=1) / den[idx].sum(axis=1)
    ratios = ratios[np.isfinite(ratios)]
    return float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def pooled_ratio(pairs: list[tuple[float, float]]) -> float:
    num = sum(p[0] for p in pairs)
    den = sum(p[1] for p in pairs)
    return 100.0 * num / den if den else float("nan")
