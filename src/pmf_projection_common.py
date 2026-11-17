#!/usr/bin/env python3
"""Shared WHAM engine for PMFs projected onto several reaction coordinates.

The umbrella bias is always removed from the four restrained protein parts
(A, B1, B2/IDR, C).  The plotted coordinate can be the existing whole-complex
observer x coordinate, one protein-part x coordinate, or a coordinate computed
from COM trajectories.
"""

import argparse
import csv
import math
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - fallback keeps the script usable without numpy.
    np = None


K_B_KJ_MOL_K = 0.00831446261815324

BRANCH_DIRS = {
    "relaxed": Path("work/relaxed_dna"),
    "tensioned": Path("work/tensioned_dna"),
}

DEFAULT_BIASED_COLUMNS = {
    "relaxed": (1, 2, 3, 4),
    "tensioned": (50, 51, 52, 53),
}

DEFAULT_X_COLUMNS = {
    "relaxed": {"a": 1, "b1": 2, "b2": 3, "c": 4, "complex": 5},
    "tensioned": {"a": 50, "b1": 51, "b2": 52, "c": 53, "complex": 54},
}

TARGET_COLUMNS = (
    "target_rc_a_nm",
    "target_rc_b1_nm",
    "target_rc_b2_nm",
    "target_rc_c_nm",
)

COM_GROUP_KEYS = ("dna", "complex", "b2", "a")

COORD_INFO = {
    "x_complex": {
        "label": "protein_complex x displacement from initial position",
        "unit": "nm",
        "source": "pullx",
        "x_key": "complex",
        "origin_column": "start_rc_nm",
    },
    "idr_x": {
        "label": "IDR/B2 x displacement from initial position",
        "unit": "nm",
        "source": "pullx",
        "x_key": "b2",
        "origin_column": "start_rc_b2_nm",
    },
    "chainA_x": {
        "label": "chain A x displacement from initial position",
        "unit": "nm",
        "source": "pullx",
        "x_key": "a",
        "origin_column": "start_rc_a_nm",
    },
    "theta": {
        "label": "theta from initial radial vector",
        "unit": "deg",
        "source": "com",
    },
    "idr_r": {
        "label": "IDR/B2 radial distance to DNA axis",
        "unit": "nm",
        "source": "com",
    },
}


class Window(object):
    def __init__(self, index, window_id, centers, pullx_path, window_dir, origin_nm):
        self.index = index
        self.window_id = window_id
        self.centers = centers
        self.pullx_path = pullx_path
        self.window_dir = window_dir
        self.origin_nm = origin_nm
        self.count = 0


class Sample(object):
    def __init__(self, window_index, time_ps, coordinate, subgroup_coords):
        self.window_index = window_index
        self.time_ps = time_ps
        self.coordinate = coordinate
        self.subgroup_coords = subgroup_coords


class ComRow(object):
    def __init__(self, time_ps, dna, complex_coords, b2, a):
        self.time_ps = time_ps
        self.dna = dna
        self.complex = complex_coords
        self.b2 = b2
        self.a = a

    def group(self, key):
        if key == "dna":
            return self.dna
        if key == "complex":
            return self.complex
        if key == "b2":
            return self.b2
        if key == "a":
            return self.a
        raise KeyError(key)


def config_path(root: Path) -> Optional[Path]:
    config_dir = root / "config"
    if not config_dir.exists():
        return None
    files = sorted(config_dir.glob("*.sh"))
    return files[0] if files else None


def parse_float_config(config: Optional[Path], key: str, default: float) -> float:
    if config is None or not config.exists():
        return default
    pattern = re.compile(rf"^\s*{re.escape(key)}=(.+?)\s*(?:#.*)?$")
    for line in config.read_text(errors="ignore").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        value = match.group(1).strip().strip('"').strip("'")
        try:
            return float(value)
        except ValueError:
            return default
    return default


def parse_string_config(config: Optional[Path], key: str, default: str) -> str:
    if config is None or not config.exists():
        return default
    pattern = re.compile(rf"^\s*{re.escape(key)}=(.+?)\s*(?:#.*)?$")
    for line in config.read_text(errors="ignore").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        value = match.group(1).strip().strip('"').strip("'")
        default_match = re.fullmatch(r"\$\{[^:}]+:-(.+)\}", value)
        if default_match:
            value = default_match.group(1)
        return value or default
    return default


def parse_force_constants(args: argparse.Namespace, root: Path) -> Tuple[float, float, float, float]:
    if args.force_constants:
        values = tuple(float(part) for part in args.force_constants.split(","))
        if len(values) != 4:
            raise SystemExit("--force-constants must contain four comma-separated values.")
        return values  # type: ignore[return-value]
    cfg = config_path(root)
    return (
        parse_float_config(cfg, "PMF_FORCE_A_KJ_MOL_NM2", 500.0),
        parse_float_config(cfg, "PMF_FORCE_B1_KJ_MOL_NM2", 500.0),
        parse_float_config(cfg, "PMF_FORCE_B2_KJ_MOL_NM2", 1000.0),
        parse_float_config(cfg, "PMF_FORCE_C_KJ_MOL_NM2", 500.0),
    )


def parse_args(default_coordinate: Optional[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project A/B1/B2/C-debiased WHAM onto one PMF coordinate."
    )
    if default_coordinate is None:
        parser.add_argument("--coordinate", required=True, choices=sorted(COORD_INFO))
    else:
        parser.set_defaults(coordinate=default_coordinate)
    parser.add_argument("branch", choices=("relaxed", "tensioned", "both"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pullx-name", default="production_pullx.xvg")
    parser.add_argument("--tpr-name", default="production.tpr")
    parser.add_argument("--trajectory-name", default="production.xtc")
    parser.add_argument("--com-cache-name", default="pmf_projection_com.xvg")
    parser.add_argument("--refresh-com-cache", action="store_true")
    parser.add_argument("--gmx", default="gmx")
    parser.add_argument("--begin-ps", type=float, default=0.0)
    parser.add_argument("--begin-ns", type=float, default=None)
    parser.add_argument("--end-ps", type=float, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--force-constants", default=None)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--axis-origin",
        choices=("initial", "frame_dna_reference"),
        default="initial",
        help="Origin used for theta and idr_r. Default: initial DNA axis.",
    )
    parser.add_argument("--dna-reference-group", default=None)
    parser.add_argument("--protein-complex-group", default=None)
    parser.add_argument("--protein-b2-group", default=None)
    parser.add_argument("--protein-a-group", default=None)
    return parser.parse_args()


def group_names(args: argparse.Namespace, root: Path) -> Dict[str, str]:
    cfg = config_path(root)
    return {
        "dna": args.dna_reference_group
        or parse_string_config(cfg, "DNA_REFERENCE_GROUP_SELECTION", "dna_reference"),
        "complex": args.protein_complex_group
        or parse_string_config(cfg, "PROTEIN_COMPLEX_GROUP_SELECTION", "protein_complex"),
        "b2": args.protein_b2_group
        or parse_string_config(cfg, "PROTEIN_PULL_GROUP_B2_SELECTION", "protein_b2"),
        "a": args.protein_a_group
        or parse_string_config(cfg, "PROTEIN_PULL_GROUP_A_SELECTION", "protein_a"),
    }


def read_metadata(
    root: Path,
    branch: str,
    coordinate: str,
    pullx_name: str,
    allow_missing: bool,
) -> List[Window]:
    branch_dir = root / BRANCH_DIRS[branch]
    metadata_path = branch_dir / "window_metadata.tsv"
    if not metadata_path.exists():
        raise SystemExit(f"Missing metadata: {metadata_path}")

    origin_column = COORD_INFO[coordinate].get("origin_column", "start_rc_nm")
    windows = []
    with metadata_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            window_dir = branch_dir / row["window_id"]
            pullx = window_dir / pullx_name
            if not pullx.exists():
                if allow_missing:
                    continue
                raise SystemExit(f"Missing pullx file: {pullx}")
            if origin_column not in row:
                raise SystemExit(f"{metadata_path} is missing {origin_column}.")
            centers = tuple(float(row[name]) for name in TARGET_COLUMNS)
            windows.append(
                Window(
                    len(windows),
                    row["window_id"],
                    centers,
                    pullx,
                    window_dir,
                    float(row[origin_column]),
                )
            )
    if not windows:
        raise SystemExit(f"No usable windows in {metadata_path}")
    return windows


def iter_numeric_rows(path: Path) -> Iterable[List[float]]:
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped[0] in "#@":
                continue
            yield [float(part) for part in stripped.split()]


def read_com_rows(path: Path) -> Dict[float, ComRow]:
    rows = {}
    for cols in iter_numeric_rows(path):
        if len(cols) < 13:
            raise SystemExit(f"{path} must contain time plus 4 COM xyz triplets.")
        triplets = [tuple(cols[1 + 3 * i : 1 + 3 * (i + 1)]) for i in range(4)]
        rows[round(cols[0], 6)] = ComRow(cols[0], triplets[0], triplets[1], triplets[2], triplets[3])
    if not rows:
        raise SystemExit(f"No COM rows found in {path}")
    return rows


def index_path(root: Path) -> Path:
    candidates = [
        root / "work/system/index.ndx",
        root / "system/index.ndx",
        root / "index.ndx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit("Could not find index.ndx; expected work/system/index.ndx.")


def run_gmx_traj_com(
    args: argparse.Namespace,
    root: Path,
    tpr: Path,
    trajectory: Path,
    output: Path,
    cwd: Path,
) -> None:
    if not tpr.exists():
        raise SystemExit(f"Missing tpr file: {tpr}")
    if not trajectory.exists():
        raise SystemExit(f"Missing trajectory/structure file: {trajectory}")
    idx = index_path(root)
    groups = group_names(args, root)
    selection = "\n".join(groups[key] for key in COM_GROUP_KEYS) + "\n"
    command = [
        args.gmx,
        "traj",
        "-s",
        str(tpr),
        "-f",
        str(trajectory),
        "-n",
        str(idx),
        "-ox",
        str(output),
        "-com",
        "-ng",
        str(len(COM_GROUP_KEYS)),
        "-xvg",
        "none",
    ]
    log_path = output.with_suffix(output.suffix + ".log")
    with log_path.open("w") as log:
        result = subprocess.run(
            command,
            input=selection,
            text=True,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise SystemExit(f"gmx traj failed for {output}; see {log_path}")


def ensure_window_com_cache(args: argparse.Namespace, root: Path, window: Window) -> Path:
    cache = window.window_dir / args.com_cache_name
    if cache.exists() and not args.refresh_com_cache:
        return cache
    run_gmx_traj_com(
        args,
        root,
        window.window_dir / args.tpr_name,
        window.window_dir / args.trajectory_name,
        cache,
        window.window_dir,
    )
    return cache


def initial_structure_paths(root: Path, branch: str) -> Tuple[Path, Path]:
    if branch == "relaxed":
        candidates = [
            (root / "work/system/npt.tpr", root / "work/system/npt.gro"),
        ]
    else:
        candidates = [
            (
                root / "work/tensioned_dna/stretch/dna_stretch.tpr",
                root / "work/tensioned_dna/stretch/dna_stretch.gro",
            )
        ]
    for tpr, structure in candidates:
        if tpr.exists() and structure.exists():
            return tpr, structure
    raise SystemExit(f"Could not find initial COM reference files for {branch}.")


def initial_com_reference(args: argparse.Namespace, root: Path, branch: str) -> ComRow:
    branch_dir = root / BRANCH_DIRS[branch]
    cache = branch_dir / "pmf_projection_initial_com.xvg"
    if not cache.exists() or args.refresh_com_cache:
        branch_dir.mkdir(parents=True, exist_ok=True)
        tpr, structure = initial_structure_paths(root, branch)
        run_gmx_traj_com(args, root, tpr, structure, cache, branch_dir)
    rows = read_com_rows(cache)
    return rows[sorted(rows)[-1]]


def logsumexp(values: Iterable[float]) -> float:
    vals = list(values)
    high = max(vals)
    if math.isinf(high):
        return high
    return high + math.log(sum(math.exp(v - high) for v in vals))


def bias_energy(
    coords: Tuple[float, float, float, float],
    centers: Tuple[float, float, float, float],
    k_values: Tuple[float, float, float, float],
) -> float:
    return 0.5 * sum(k * (x - c) ** 2 for k, x, c in zip(k_values, coords, centers))


def theta_deg(
    initial: ComRow,
    frame: ComRow,
    axis_origin: str,
) -> float:
    if axis_origin == "frame_dna_reference":
        origin_y, origin_z = frame.dna[1], frame.dna[2]
    else:
        origin_y, origin_z = initial.dna[1], initial.dna[2]

    v0_y = initial.complex[1] - initial.dna[1]
    v0_z = initial.complex[2] - initial.dna[2]
    v_y = frame.complex[1] - origin_y
    v_z = frame.complex[2] - origin_z

    n0 = math.hypot(v0_y, v0_z)
    n = math.hypot(v_y, v_z)
    if n0 == 0 or n == 0:
        raise SystemExit("theta is undefined because a radial vector has zero length.")
    cross_x = v0_y * v_z - v0_z * v_y
    dot = v0_y * v_y + v0_z * v_z
    return math.degrees(math.atan2(cross_x, dot))


def idr_radius_nm(initial: ComRow, frame: ComRow, axis_origin: str) -> float:
    if axis_origin == "frame_dna_reference":
        origin_y, origin_z = frame.dna[1], frame.dna[2]
    else:
        origin_y, origin_z = initial.dna[1], initial.dna[2]
    return math.hypot(frame.b2[1] - origin_y, frame.b2[2] - origin_z)


def read_pullx_projected_samples(
    windows: List[Window],
    branch: str,
    coordinate: str,
    begin_ps: float,
    end_ps: Optional[float],
    stride: int,
) -> List[Sample]:
    subgroup_indices = tuple(col - 1 for col in DEFAULT_BIASED_COLUMNS[branch])
    x_key = str(COORD_INFO[coordinate]["x_key"])
    projection_index = DEFAULT_X_COLUMNS[branch][x_key] - 1
    needed = max((*subgroup_indices, projection_index))
    samples = []

    for window in windows:
        accepted = 0
        kept = 0
        for cols in iter_numeric_rows(window.pullx_path):
            time_ps = cols[0]
            if time_ps < begin_ps:
                continue
            if end_ps is not None and time_ps > end_ps:
                continue
            accepted += 1
            if (accepted - 1) % stride != 0:
                continue
            pull_values = cols[1:]
            if len(pull_values) <= needed:
                raise SystemExit(f"{window.pullx_path} does not contain requested pull columns.")
            samples.append(
                Sample(
                    window.index,
                    time_ps,
                    pull_values[projection_index] - window.origin_nm,
                    tuple(pull_values[i] for i in subgroup_indices),
                )
            )
            kept += 1
        if kept == 0:
            raise SystemExit(f"No samples retained from {window.pullx_path}")
        window.count = kept
    return samples


def read_com_projected_samples(
    args: argparse.Namespace,
    root: Path,
    windows: List[Window],
    branch: str,
    coordinate: str,
    begin_ps: float,
    end_ps: Optional[float],
    stride: int,
) -> List[Sample]:
    subgroup_indices = tuple(col - 1 for col in DEFAULT_BIASED_COLUMNS[branch])
    needed = max(subgroup_indices)
    initial = initial_com_reference(args, root, branch)
    samples = []

    for window in windows:
        com_rows = read_com_rows(ensure_window_com_cache(args, root, window))
        accepted = 0
        kept = 0
        missing_times = []
        for cols in iter_numeric_rows(window.pullx_path):
            time_ps = cols[0]
            if time_ps < begin_ps:
                continue
            if end_ps is not None and time_ps > end_ps:
                continue
            accepted += 1
            if (accepted - 1) % stride != 0:
                continue
            pull_values = cols[1:]
            if len(pull_values) <= needed:
                raise SystemExit(f"{window.pullx_path} does not contain requested pull columns.")
            frame = com_rows.get(round(time_ps, 6))
            if frame is None:
                missing_times.append(time_ps)
                continue
            if coordinate == "theta":
                projected = theta_deg(initial, frame, args.axis_origin)
            elif coordinate == "idr_r":
                projected = idr_radius_nm(initial, frame, args.axis_origin)
            else:
                raise SystemExit(f"{coordinate} is not a COM-projected coordinate.")
            samples.append(
                Sample(
                    window.index,
                    time_ps,
                    projected,
                    tuple(pull_values[i] for i in subgroup_indices),
                )
            )
            kept += 1
        if missing_times:
            preview = ", ".join(f"{value:g}" for value in missing_times[:5])
            raise SystemExit(f"{window.window_id}: COM cache misses pullx times: {preview}")
        if kept == 0:
            raise SystemExit(f"No samples retained from {window.pullx_path}")
        window.count = kept
    return samples


def read_samples(
    args: argparse.Namespace,
    root: Path,
    windows: List[Window],
    branch: str,
    coordinate: str,
    begin_ps: float,
    end_ps: Optional[float],
    stride: int,
) -> List[Sample]:
    source = COORD_INFO[coordinate]["source"]
    if source == "pullx":
        return read_pullx_projected_samples(windows, branch, coordinate, begin_ps, end_ps, stride)
    return read_com_projected_samples(args, root, windows, branch, coordinate, begin_ps, end_ps, stride)


def solve_wham(
    samples: List[Sample],
    windows: List[Window],
    beta: float,
    k_values: Tuple[float, float, float, float],
    max_iter: int,
    tol: float,
) -> Tuple[List[float], List[float], int, float]:
    if np is not None:
        return solve_wham_numpy(samples, windows, beta, k_values, max_iter, tol)

    counts = [window.count for window in windows]
    log_counts = [math.log(count) for count in counts]
    reduced_biases = [
        [beta * bias_energy(sample.subgroup_coords, window.centers, k_values) for window in windows]
        for sample in samples
    ]
    offsets = [0.0] * len(windows)
    delta = float("inf")

    for iteration in range(1, max_iter + 1):
        denominators = [
            logsumexp(log_counts[i] + offsets[i] - sample_bias[i] for i in range(len(windows)))
            for sample_bias in reduced_biases
        ]
        updated = [
            -logsumexp(-sample_bias[i] - denom for sample_bias, denom in zip(reduced_biases, denominators))
            for i in range(len(windows))
        ]
        zero = updated[0]
        updated = [value - zero for value in updated]
        delta = max(abs(a - b) for a, b in zip(updated, offsets))
        offsets = updated
        if delta < tol:
            break

    final_denominators = [
        logsumexp(log_counts[i] + offsets[i] - sample_bias[i] for i in range(len(windows)))
        for sample_bias in reduced_biases
    ]
    log_weights = [-value for value in final_denominators]
    norm = logsumexp(log_weights)
    weights = [math.exp(value - norm) for value in log_weights]
    return offsets, weights, iteration, delta


def numpy_logsumexp(values, axis):
    high = np.max(values, axis=axis, keepdims=True)
    summed = np.sum(np.exp(values - high), axis=axis, keepdims=True)
    return np.squeeze(high + np.log(summed), axis=axis)


def solve_wham_numpy(
    samples: List[Sample],
    windows: List[Window],
    beta: float,
    k_values: Tuple[float, float, float, float],
    max_iter: int,
    tol: float,
) -> Tuple[List[float], List[float], int, float]:
    counts = np.asarray([window.count for window in windows], dtype=float)
    log_counts = np.log(counts)
    coords = np.asarray([sample.subgroup_coords for sample in samples], dtype=float)
    centers = np.asarray([window.centers for window in windows], dtype=float)
    k_array = np.asarray(k_values, dtype=float)
    reduced_biases = beta * 0.5 * np.sum(
        k_array[None, None, :] * (coords[:, None, :] - centers[None, :, :]) ** 2,
        axis=2,
    )
    offsets = np.zeros(len(windows), dtype=float)
    delta = float("inf")

    for iteration in range(1, max_iter + 1):
        denominators = numpy_logsumexp(
            log_counts[None, :] + offsets[None, :] - reduced_biases,
            axis=1,
        )
        updated = -numpy_logsumexp(-reduced_biases - denominators[:, None], axis=0)
        updated -= updated[0]
        delta = float(np.max(np.abs(updated - offsets)))
        offsets = updated
        if delta < tol:
            break

    final_denominators = numpy_logsumexp(
        log_counts[None, :] + offsets[None, :] - reduced_biases,
        axis=1,
    )
    log_weights = -final_denominators
    norm = float(numpy_logsumexp(log_weights, axis=0))
    weights = np.exp(log_weights - norm)
    return offsets.tolist(), weights.tolist(), iteration, delta


def histogram_pmf(
    samples: List[Sample],
    weights: List[float],
    beta: float,
    bins: int,
) -> List[Tuple[float, float, float, int]]:
    values = [sample.coordinate for sample in samples]
    xmin, xmax = min(values), max(values)
    if xmin == xmax:
        xmin -= 0.5
        xmax += 0.5
    width = (xmax - xmin) / bins
    mass = [0.0] * bins
    counts = [0] * bins

    for sample, weight in zip(samples, weights):
        index = int((sample.coordinate - xmin) / width)
        if index == bins:
            index -= 1
        mass[index] += weight
        counts[index] += 1

    pmf = [(-math.log(value) / beta if value > 0 else float("nan")) for value in mass]
    finite = [value for value in pmf if math.isfinite(value)]
    if not finite:
        raise SystemExit("No populated coordinate bins.")
    zero = min(finite)
    return [
        (xmin + (i + 0.5) * width, value - zero if math.isfinite(value) else float("nan"), mass[i], counts[i])
        for i, value in enumerate(pmf)
    ]


def write_branch_outputs(
    output_dir: Path,
    branch: str,
    coordinate: str,
    windows: List[Window],
    samples: List[Sample],
    rows: List[Tuple[float, float, float, int]],
    offsets: List[float],
    iterations: int,
    delta: float,
    temperature: float,
    beta: float,
    k_values: Tuple[float, float, float, float],
    args: argparse.Namespace,
) -> None:
    info = COORD_INFO[coordinate]
    unit = str(info["unit"])
    label = str(info["label"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{coordinate}_pmf.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow((f"bin_center_{unit}", "pmf_shifted_kj_mol", "probability_mass", "raw_count"))
        writer.writerows(rows)
    with (output_dir / f"{coordinate}_pmf.xvg").open("w") as handle:
        handle.write(f'@    title "{coordinate} WHAM PMF: {branch}"\n')
        handle.write(f'@    xaxis label "{label} ({unit})"\n')
        handle.write('@    yaxis label "PMF - min(PMF) (kJ/mol)"\n')
        handle.write("@TYPE xy\n")
        for x, y, _, _ in rows:
            if math.isfinite(y):
                handle.write(f"{x:.8f}\t{y:.8f}\n")
    with (output_dir / "wham_window_offsets.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("window_id", "offset_dimensionless", "offset_kj_mol", "sample_count"))
        for window, offset in zip(windows, offsets):
            writer.writerow((window.window_id, offset, offset / beta, window.count))
    with (output_dir / "run_summary.txt").open("w") as handle:
        handle.write(f"branch: {branch}\n")
        handle.write(f"coordinate: {coordinate}\n")
        handle.write(f"coordinate_label: {label}\n")
        handle.write(f"coordinate_unit: {unit}\n")
        handle.write(f"coordinate_source: {info['source']}\n")
        handle.write(f"axis_origin: {args.axis_origin}\n")
        handle.write(f"windows: {len(windows)}\n")
        handle.write(f"samples: {len(samples)}\n")
        handle.write(f"temperature_K: {temperature:.6f}\n")
        handle.write(f"kBT_kJ_mol: {1.0 / beta:.8f}\n")
        handle.write(f"force_constants_A_B1_B2_C: {k_values}\n")
        handle.write(f"iterations: {iterations}\n")
        handle.write(f"final_delta: {delta:.8e}\n")
        handle.write("bias_removed_from: protein_a, protein_b1, protein_b2/IDR, protein_c\n")
        if coordinate == "x_complex":
            handle.write("x_definition: protein_complex observer coordinate minus initial start_rc_nm\n")
        elif coordinate == "idr_x":
            handle.write("x_definition: protein_b2/IDR coordinate minus initial start_rc_b2_nm\n")
        elif coordinate == "chainA_x":
            handle.write("x_definition: protein_a coordinate minus initial start_rc_a_nm\n")
        elif coordinate == "theta":
            handle.write("theta_definition: signed angle in the yz plane from the initial protein_complex radial vector to the frame radial vector\n")
        elif coordinate == "idr_r":
            handle.write("r_definition: radial distance of protein_b2/IDR COM to the DNA x-axis\n")


def read_pmf_rows(path: Path, coordinate: str) -> List[Tuple[float, float]]:
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        x_key = f"bin_center_{COORD_INFO[coordinate]['unit']}"
        for row in reader:
            x = float(row[x_key])
            y = float(row["pmf_shifted_kj_mol"])
            if math.isfinite(x) and math.isfinite(y):
                rows.append((x, y))
    return rows


def ticks(vmin: float, vmax: float, count: int = 6) -> List[float]:
    if vmax <= vmin:
        return [vmin]
    raw = (vmax - vmin) / max(count - 1, 1)
    power = 10 ** math.floor(math.log10(raw))
    step = min((1, 2, 2.5, 5, 10), key=lambda v: abs(v * power - raw)) * power
    start = math.floor(vmin / step) * step
    stop = math.ceil(vmax / step) * step
    values = []
    value = start
    while value <= stop + 0.5 * step:
        values.append(round(value, 10))
        value += step
    return values


def write_overlay_svg(output_dir: Path, coordinate: str) -> None:
    series = []
    for branch, label, color in (
        ("relaxed", "normal DNA", "#2563eb"),
        ("tensioned", "tensioned DNA", "#dc2626"),
    ):
        path = output_dir / branch / f"{coordinate}_pmf.tsv"
        if path.exists():
            rows = read_pmf_rows(path, coordinate)
            if rows:
                series.append((label, color, rows))
    if not series:
        return

    width, height = 920, 560
    left, right, top, bottom = 90, 36, 58, 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [x for _, _, rows in series for x, _ in rows]
    ys = [y for _, _, rows in series for _, y in rows]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = 0.0, max(ys)
    if xmin == xmax:
        xmin -= 0.1
        xmax += 0.1
    if ymax <= ymin:
        ymax = 1.0
    ymax *= 1.08
    xt = ticks(xmin, xmax)
    yt = ticks(ymin, ymax)
    xmin, xmax = min(xmin, xt[0]), max(xmax, xt[-1])
    ymin, ymax = min(ymin, yt[0]), max(ymax, yt[-1])

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - (y - ymin) / (ymax - ymin) * plot_h

    def poly(rows: List[Tuple[float, float]]) -> str:
        return " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in rows)

    axis = f"{COORD_INFO[coordinate]['label']} ({COORD_INFO[coordinate]['unit']})"
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111827}.title{font-size:21px;font-weight:700}.small{font-size:13px;fill:#4b5563}.axis{font-size:15px}.tick{font-size:12px;fill:#4b5563}</style>',
        f'<text class="title" x="{left}" y="32">{coordinate} WHAM PMF</text>',
        f'<text class="small" x="{left}" y="52">Bias from A/B1/B2/C restraints is removed; plotted coordinate: {axis}.</text>',
    ]
    for value in xt:
        x = sx(value)
        out.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#e5e7eb"/>')
        out.append(f'<text class="tick" x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle">{value:g}</text>')
    for value in yt:
        y = sy(value)
        out.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        out.append(f'<text class="tick" x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">{value:g}</text>')
    out.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#111827"/>')
    out.append(f'<text class="axis" x="{left + plot_w / 2:.2f}" y="{height - 24}" text-anchor="middle">{axis}</text>')
    out.append(f'<text class="axis" transform="translate(24 {top + plot_h / 2:.2f}) rotate(-90)" text-anchor="middle">PMF - min(PMF) (kJ/mol)</text>')
    legend_x = left + plot_w - 190
    legend_y = top + 20
    for i, (label, color, rows) in enumerate(series):
        y = legend_y + i * 24
        out.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{poly(rows)}"/>')
        out.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 32}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        out.append(f'<text class="small" x="{legend_x + 42}" y="{y + 4}">{label}</text>')
    out.append("</svg>")
    (output_dir / f"{coordinate}_pmf_preview.svg").write_text("\n".join(out) + "\n", encoding="utf-8")


def run_branch(args: argparse.Namespace, branch: str, output_base: Path) -> None:
    root = args.root.resolve()
    coordinate = args.coordinate
    cfg = config_path(root)
    temperature = args.temperature or parse_float_config(cfg, "TEMPERATURE_K", 310.0)
    beta = 1.0 / (K_B_KJ_MOL_K * temperature)
    k_values = parse_force_constants(args, root)
    begin_ps = args.begin_ns * 1000.0 if args.begin_ns is not None else args.begin_ps
    windows = read_metadata(root, branch, coordinate, args.pullx_name, args.allow_missing)
    samples = read_samples(args, root, windows, branch, coordinate, begin_ps, args.end_ps, args.stride)
    offsets, weights, iterations, delta = solve_wham(samples, windows, beta, k_values, args.max_iter, args.tol)
    rows = histogram_pmf(samples, weights, beta, args.bins)
    branch_out = output_base / branch if args.branch == "both" else output_base
    write_branch_outputs(
        branch_out,
        branch,
        coordinate,
        windows,
        samples,
        rows,
        offsets,
        iterations,
        delta,
        temperature,
        beta,
        k_values,
        args,
    )
    print(f"{branch}: wrote {branch_out} ({len(windows)} windows, {len(samples)} samples, delta {delta:.3e})")


def main(default_coordinate: Optional[str] = None) -> None:
    args = parse_args(default_coordinate)
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1.")
    if args.bins < 2:
        raise SystemExit("--bins must be >= 2.")
    output_base = args.output_dir or (args.root / f"{args.coordinate}_pmf")
    output_base = output_base.resolve()
    branches = ("relaxed", "tensioned") if args.branch == "both" else (args.branch,)
    for branch in branches:
        run_branch(args, branch, output_base)
    if args.branch == "both":
        write_overlay_svg(output_base, args.coordinate)
        overlay_path = output_base / "{}_pmf_preview.svg".format(args.coordinate)
        print(f"Done. Overlay: {overlay_path}")
    else:
        print(f"Done. Output: {output_base}")


if __name__ == "__main__":
    main()
