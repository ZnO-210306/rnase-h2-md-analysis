#!/usr/bin/env python3
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ANALYSIS_ROOT
INPUT_ROOT = ANALYSIS_ROOT / "inputs/pmf"
OUTPUT_ROOT = ANALYSIS_ROOT / "results/window_overlap_density"
sys.path.insert(0, str(WORKSPACE / "src"))
import pmf_projection_common as pmf  # noqa: E402


COORDINATES = ("x_complex", "theta", "idr_x", "chainA_x")
BRANCHES = ("relaxed", "tensioned")
LABELS = {
    "x_complex": "Whole-protein x displacement (nm)",
    "theta": "Theta from initial radial vector (deg)",
    "idr_x": "IDR/B2 x displacement (nm)",
    "chainA_x": "Chain A x displacement (nm)",
}


def common_args(coordinate):
    return SimpleNamespace(
        root=INPUT_ROOT,
        coordinate=coordinate,
        pullx_name="production_pullx.xvg",
        tpr_name="production.tpr",
        trajectory_name="production.xtc",
        com_cache_name="pmf_projection_com.xvg",
        refresh_com_cache=False,
        gmx="gmx",
        axis_origin="initial",
        dna_reference_group=None,
        protein_complex_group=None,
        protein_b2_group=None,
        protein_a_group=None,
    )


def smooth(values, sigma=1.2):
    radius = max(2, int(np.ceil(4 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(values, kernel, mode="same")


def branch_samples(coordinate, branch):
    args = common_args(coordinate)
    windows = pmf.read_metadata(
        INPUT_ROOT, branch, coordinate, args.pullx_name, False
    )
    samples = pmf.read_samples(
        args, INPUT_ROOT, windows, branch, coordinate, 0.0, None, 1
    )
    grouped = []
    for window in windows:
        grouped.append(np.asarray([
            sample.coordinate for sample in samples
            if sample.window_index == window.index
        ], dtype=float))
    return windows, grouped


def overlap_coefficient(left, right, edges):
    left_hist, _ = np.histogram(left, bins=edges, density=True)
    right_hist, _ = np.histogram(right, bins=edges, density=True)
    widths = np.diff(edges)
    return float(np.sum(np.minimum(left_hist, right_hist) * widths))


def main():
    OUTPUT_ROOT.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=1, vmax=16)
    summary_rows = []

    for coordinate in COORDINATES:
        branch_data = {
            branch: branch_samples(coordinate, branch) for branch in BRANCHES
        }
        all_values = np.concatenate([
            values
            for _, grouped in branch_data.values()
            for values in grouped
        ])
        low, high = np.quantile(all_values, [0.001, 0.999])
        padding = max((high - low) * 0.035, 1e-6)
        edges = np.linspace(low - padding, high + padding, 141)
        centers = 0.5 * (edges[:-1] + edges[1:])

        fig, axes = plt.subplots(
            2, 1, figsize=(7.4, 6.3), sharex=True, constrained_layout=True
        )
        for ax, branch in zip(axes, BRANCHES):
            windows, grouped = branch_data[branch]
            for index, values in enumerate(grouped, start=1):
                density, _ = np.histogram(values, bins=edges, density=True)
                density = smooth(density)
                ax.plot(
                    centers, density, color=cmap(norm(index)),
                    linewidth=1.05, alpha=0.92,
                )
            for index in range(len(grouped) - 1):
                summary_rows.append({
                    "coordinate": coordinate,
                    "branch": branch,
                    "window_left": windows[index].window_id,
                    "window_right": windows[index + 1].window_id,
                    "overlap_coefficient": overlap_coefficient(
                        grouped[index], grouped[index + 1], edges
                    ),
                })
            ax.set_ylabel("Probability density")
            ax.set_title(
                f"{'Relaxed' if branch == 'relaxed' else 'Tensioned'} DNA",
                loc="left", fontweight="normal",
            )
            ax.grid(axis="y", color="#dddddd", linewidth=0.55, alpha=0.75)

        axes[-1].set_xlabel(LABELS[coordinate])
        scalar = ScalarMappable(norm=norm, cmap=cmap)
        scalar.set_array([])
        colorbar = fig.colorbar(scalar, ax=axes, pad=0.015, fraction=0.035)
        colorbar.set_label("Window")
        colorbar.set_ticks([1, 4, 8, 12, 16])
        fig.savefig(
            OUTPUT_ROOT / f"re_orig_{coordinate}_window_overlap.png", dpi=300
        )
        plt.close(fig)

    with (OUTPUT_ROOT / "adjacent_window_overlap.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "coordinate", "branch", "window_left", "window_right",
                "overlap_coefficient",
            ),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in summary_rows:
            row = dict(row)
            row["overlap_coefficient"] = f"{row['overlap_coefficient']:.6f}"
            writer.writerow(row)

    print(f"Created {len(COORDINATES)} overlap plots in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
