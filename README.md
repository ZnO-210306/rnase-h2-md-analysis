# RNase H2–DNA molecular dynamics analysis

Python programs for analysing RNase H2–DNA molecular dynamics data under relaxed-DNA and tensioned-DNA conditions. The package calculates replica statistics, IDR short-range interaction-energy fractions, time-window averages, and potential-of-mean-force (PMF) profiles. It includes the weighted histogram analysis method (WHAM) solver, umbrella-window coordinate inputs, tabulated results, and plotting scripts.

## Authors

1. Xinyang Chen — University of Science and Technology of China; ORCID: https://orcid.org/0009-0009-6898-8549
2. Yihan Wu — University of Science and Technology of China

## Installation

Use Python 3.10 or later. Run the following commands from the repository root:

```sh
python -m venv .venv
```

Activate the environment with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Linux/macOS, then install the dependencies:

```sh
python -m pip install -r requirements.txt
```

## Run the analysis

```sh
python scripts/analyze_observables.py
python scripts/plot_observables.py
python scripts/plot_timecourses.py
python scripts/plot_window_overlap_filled_style.py
python scripts/verify_pmf.py
```

Tables, figures and verification reports are written to `results/`. The bundled coordinate inputs allow these commands to run directly with Python, NumPy and Matplotlib.

To solve WHAM afresh and compare the calculated profiles with the reference values:

```sh
python scripts/verify_pmf.py --solve
```

The solver can also be run for an individual projection:

```sh
python src/pmf_projection_common.py both --coordinate theta --root inputs/pmf --output-dir results/theta_wham --temperature 310 --force-constants 500,500,1000,500 --bins 100
```

Available projection names are `x_complex`, `theta`, `idr_x`, `chainA_x` and `idr_r`. The supplied reference profiles cover the first four coordinates. The solver's `--help` output describes time selection, sampling stride, bin count and coordinate options.

## Data and programs

| Directory | Contents |
| --- | --- |
| `src/` | WHAM solver, coordinate extraction and PMF projection functions |
| `scripts/` | Statistical analysis, plotting and numerical verification |
| `data/MainText_Fig6/` | Processed contact, hydrogen-bond and short-range energy series with retained-frame flags and replica summaries |
| `data/Fig_R11/` | Four configurations of IDR and non-IDR short-range energy series |
| `data/Fig_R12/` | WT contact and hydrogen-bond time series and 5 ns block summaries |
| `data/PMF_numeric_backup/` | PMF profiles and per-window coordinate tables |
| `inputs/pmf/` | Pull coordinates, centre-of-mass coordinates, initial coordinate references and umbrella-window metadata |
| `reference/pmf/` | Reference PMF profiles, WHAM offsets and solver settings |

Dataset names follow the analysis figure identifiers: Fig. 6b–d for replica statistics; R11 for energy fractions; R12 for contact and hydrogen-bond time courses; Fig. 6f for whole-protein axial PMF; R13 for rotation; R14 for IDR axial displacement; R15 for chain-A axial displacement.

## Statistical definitions

Contact counts represent IDR–DNA atom pairs within 0.6 nm. Hydrogen-bond analysis uses a donor–acceptor distance of 0.35 nm and an H–D–A angle of 30 degrees. Short-range interaction energy is the sum of the Coulomb-SR and Lennard-Jones-SR components, in kJ/mol.

Replica statistics use rows marked `retained_after_initial20percent=1`. The initial `floor(N/5)` rows of each processed series are removed. The retained physical-time intervals are listed in `Fig6_selection_intervals.csv`. Group means weight each replica mean equally; error bars use the sample standard deviation across replica means (`ddof=1`).

The R11 energy fraction is `abs(mean(E_IDR)) / (abs(mean(E_IDR)) + abs(mean(E_nonIDR)))`, using 0–10 ns for each configuration. Each short-range energy combines the Coulomb and Lennard-Jones terms before averaging.

The R12 analysis uses three WT relaxed-DNA trajectories spanning 0–50 ns at 0.01 ns intervals. Five-nanosecond blocks are left-closed and right-open; the final block includes 50 ns. The summed trace is the pointwise sum of the three trajectories. The horizontal band reports the mean and sample standard deviation of the ten summed block means. The tensioned-DNA moving average uses 50 frames with edge padding of 25 and 24 frames.

## PMF and window distributions

Each DNA condition contains 16 windows with 3,001 frames per window (0–30 ns, 0.01 ns intervals). The temperature is 310 K. Harmonic restraint constants for A, B1, B2/IDR and C are 500, 500, 1000 and 500 kJ/mol/nm². WHAM removes these four restraint contributions and supplies weights for the selected coordinate. Each 100-bin PMF is shifted to its own minimum.

Axial displacements are measured from their initial values. Theta is the signed angle in the yz plane relative to the initial protein radial vector and initial DNA reference position. The quick verification command uses the recorded WHAM offsets; `--solve` calculates offsets from the bundled window coordinates.

Window-distribution plots pool both conditions to calculate the 0.001 and 0.999 coordinate quantiles. Each end is extended by 4.5% of their difference, with a minimum padding of 1e-6 coordinate units. The resulting 150-bin counts are convolved with a normalized Gaussian kernel of standard deviation 1.45 bins and radius 6 bins, using zero padding. Plotted values are smoothed counts. The coordinate tables retain every sampled frame.

## Citation and license

Chen, Xinyang, and Wu, Yihan (2026). RNase H2–DNA molecular dynamics analysis code (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.24318628

The GitHub repository is https://github.com/ZnO-210306/rnase-h2-md-analysis. Machine-readable citation metadata are provided in `CITATION.cff`. The code is distributed under the MIT License.

## Release validation

Version 1.0.0 was tested with Python 3.12.14, NumPy 2.3.5 and Matplotlib 3.11.0. Recomputed replica statistics, energy fractions, time-window means and moving averages agree with the supplied reference tables. Fresh WHAM solutions reached an offset-update tolerance of 1e-7 in 4,045 iterations for relaxed DNA and 3,789 iterations for tensioned DNA. All eight recomputed PMF profiles matched the supplied numerical profiles point by point. Numerical check results are stored in `validation/`.
