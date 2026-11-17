#!/usr/bin/env python3
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ANALYSIS_ROOT / "results/window_overlap"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_window_overlap as overlap  # noqa: E402


TITLES = {
    "x_complex": "Whole-protein x displacement",
    "theta": "Theta",
    "idr_x": "IDR/B2 x displacement",
    "chainA_x": "Chain A x displacement",
}
XLABELS = {
    "x_complex": "Displacement from initial position (nm)",
    "theta": "Angle from initial radial vector (deg)",
    "idr_x": "Displacement from initial position (nm)",
    "chainA_x": "Displacement from initial position (nm)",
}
COLOR = "#4c9be8"


def gaussian_smooth(values, sigma=1.45):
    radius = max(2, int(np.ceil(4 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(values, kernel, mode="same")


def main():
    OUTPUT_ROOT.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.linewidth": 0.7,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    for coordinate in overlap.COORDINATES:
        branch_data = {
            branch: overlap.branch_samples(coordinate, branch)[1]
            for branch in overlap.BRANCHES
        }
        combined = np.concatenate([
            values
            for branch in overlap.BRANCHES
            for values in branch_data[branch]
        ])
        low, high = np.quantile(combined, [0.001, 0.999])
        padding = max((high - low) * 0.045, 1e-6)
        edges = np.linspace(low - padding, high + padding, 151)
        centers = 0.5 * (edges[:-1] + edges[1:])

        curves = {}
        ymax = 0.0
        for branch in overlap.BRANCHES:
            curves[branch] = []
            for values in branch_data[branch]:
                counts, _ = np.histogram(values, bins=edges)
                curve = gaussian_smooth(counts.astype(float))
                curves[branch].append(curve)
                ymax = max(ymax, float(np.max(curve)))

        fig, axes = plt.subplots(
            1, 2, figsize=(9.0, 4.0), sharex=True, sharey=True,
            gridspec_kw={
                "left": 0.075, "right": 0.985, "bottom": 0.17,
                "top": 0.78, "wspace": 0.16,
            },
        )
        for ax, branch in zip(axes, overlap.BRANCHES):
            for curve in curves[branch]:
                ax.fill_between(
                    centers, 0.0, curve,
                    color=COLOR, alpha=0.115, linewidth=0,
                )
                ax.plot(
                    centers, curve, color=COLOR,
                    alpha=0.60, linewidth=0.70,
                )
            ax.set_xlim(edges[0], edges[-1])
            ax.set_ylim(0.0, ymax * 1.08)
            ax.set_title(
                "Relaxed DNA" if branch == "relaxed" else "Tensioned DNA",
                fontsize=10, fontweight="normal",
            )
            ax.set_xlabel(XLABELS[coordinate], fontsize=8)
            ax.grid(color="#d1d5db", linewidth=0.55, alpha=0.75)
            ax.set_axisbelow(True)
        axes[0].set_ylabel("Count", fontsize=8)

        fig.suptitle(
            f"Umbrella-window sampling: {TITLES[coordinate]}",
            x=0.075, y=0.95, ha="left", fontsize=15, fontweight="normal",
        )
        fig.savefig(
            OUTPUT_ROOT / f"re_orig_{coordinate}_window_overlap_filled.png",
            dpi=300,
        )
        plt.close(fig)

    print(f"Created 4 filled-style overlap plots in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
