"""Signal propagation utilities for the labeled M_ij Schur modes.

The input CSV convention is source cell types in rows and receiver cell types
in columns.  The discrete-time state matrix is therefore A = M_ij.T:

    x[t + 1] = A @ x[t]

For a complex Schur decomposition A = Q T Q*, z = Q* x and:

    z[t + 1] = T @ z[t].
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg

from schur_core_script import (
    format_cell_label,
    load_mij_matrix,
    normalize_state_matrix,
    region_from_label,
    spectral_radius,
)

MAX_CELL_TYPES = 5
MAX_TIMESTEPS = 10


@dataclass(frozen=True)
class SchurPropagationModel:
    """Prepared state matrix and its complex Schur decomposition."""

    A: np.ndarray
    T: np.ndarray
    Q: np.ndarray
    labels: tuple[str, ...]
    normalization: str
    scale_factor: float
    original_spectral_radius: float

    @property
    def eigenvalues(self) -> np.ndarray:
        return np.diag(self.T)

    @property
    def n_modes(self) -> int:
        return self.T.shape[0]


def prepare_schur_model(
    csv_path: str | Path = "mij_matrix.csv",
    normalization: str = "spectral_radius",
    target_spectral_radius: float = 0.95,
    include_self: bool = True,
) -> SchurPropagationModel:
    """Load M_ij, orient it for state propagation, normalize, and decompose.

    ``normalization`` may be ``"spectral_radius"``, ``"column_l1"``, or
    ``"none"``.  Spectral-radius normalization is the safe default for
    discrete-time propagation.
    """
    frame = load_mij_matrix(csv_path)
    if not include_self:
        frame = frame.copy()
        np.fill_diagonal(frame.values, 0.0)
    A_raw = frame.to_numpy(dtype=float).T
    rho = spectral_radius(A_raw)
    method = normalization.lower().strip()

    if method == "spectral_radius":
        if not 0 < target_spectral_radius:
            raise ValueError("target_spectral_radius must be positive.")
        A, norm = normalize_state_matrix(A_raw, method=method, target=target_spectral_radius)
        scale = float(norm["scale"])
    elif method in {"column_l1", "col_l1"}:
        A, _ = normalize_state_matrix(A_raw, method="column_l1", target=target_spectral_radius)
        scale = float("nan")
        method = "column_l1"
    elif method == "none":
        A, _ = normalize_state_matrix(A_raw, method=method, target=target_spectral_radius)
        scale = 1.0
    else:
        raise ValueError("normalization must be 'spectral_radius', 'column_l1', or 'none'.")

    T, Q = linalg.schur(A, output="complex")
    return SchurPropagationModel(
        A=A,
        T=T,
        Q=Q,
        labels=tuple(str(x) for x in frame.columns),
        normalization=method,
        scale_factor=scale,
        original_spectral_radius=rho,
    )


def make_initial_signal(
    model: SchurPropagationModel,
    cell_types: Sequence[str],
    amplitudes: Sequence[float] | Mapping[str, float] | None = None,
) -> np.ndarray:
    """Create x[0] from one to five named cell types."""
    names = list(cell_types)
    if not 1 <= len(names) <= MAX_CELL_TYPES:
        raise ValueError(f"Choose between 1 and {MAX_CELL_TYPES} cell types.")
    if len(set(names)) != len(names):
        raise ValueError("cell_types cannot contain duplicates.")
    missing = [name for name in names if name not in model.labels]
    if missing:
        raise KeyError(f"Unknown cell type(s): {missing}")

    if amplitudes is None:
        values = np.ones(len(names), dtype=float)
    elif isinstance(amplitudes, Mapping):
        values = np.asarray([amplitudes[name] for name in names], dtype=float)
    else:
        values = np.asarray(amplitudes, dtype=float)
        if values.shape != (len(names),):
            raise ValueError("amplitudes must have one value per cell type.")

    x0 = np.zeros(len(model.labels), dtype=complex)
    for name, value in zip(names, values):
        x0[model.labels.index(name)] = value
    return x0


def _check_timesteps(timesteps: int) -> int:
    if isinstance(timesteps, bool) or not isinstance(timesteps, (int, np.integer)):
        raise TypeError("timesteps must be an integer.")
    if not 0 <= timesteps <= MAX_TIMESTEPS:
        raise ValueError(f"timesteps must be between 0 and {MAX_TIMESTEPS}.")
    return int(timesteps)


def mode_cell_type_table(
    model: SchurPropagationModel, top_n: int = MAX_CELL_TYPES
) -> pd.DataFrame:
    """Return the largest cell-type loadings for every Schur vector."""
    if not 1 <= top_n <= MAX_CELL_TYPES:
        raise ValueError(f"top_n must be between 1 and {MAX_CELL_TYPES}.")
    rows: list[dict[str, object]] = []
    for mode in range(model.n_modes):
        order = np.argsort(np.abs(model.Q[:, mode]))[::-1][:top_n]
        for rank, cell in enumerate(order, start=1):
            weight = model.Q[cell, mode]
            rows.append(
                {
                    "mode": mode,
                    "eigenvalue": model.eigenvalues[mode],
                    "rank": rank,
                    "cell_type": format_cell_label(model.labels[cell]),
                    "loading": weight,
                    "energy_fraction": float(abs(weight) ** 2),
                }
            )
    return pd.DataFrame(rows)


def schur_mode_loadings(
    model: SchurPropagationModel,
    ei_by_cell: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Return every cell loading for every mode in tidy, ranked form."""
    rows = []
    for mode in range(model.n_modes):
        eigenvalue = model.eigenvalues[mode]
        magnitudes = np.abs(model.Q[:, mode])
        ranks = np.empty(len(magnitudes), dtype=int)
        ranks[np.argsort(magnitudes)[::-1]] = np.arange(1, len(magnitudes) + 1)
        for cell_index, cell_type in enumerate(model.labels):
            loading = model.Q[cell_index, mode]
            rows.append(
                {
                    "mode": mode,
                    "eigenvalue_real": eigenvalue.real,
                    "eigenvalue_imag": eigenvalue.imag,
                    "eigenvalue_magnitude": abs(eigenvalue),
                    "cell_index": cell_index,
                    "cell_type": format_cell_label(cell_type),
                    "region": region_from_label(cell_type),
                    "ei": None if ei_by_cell is None else ei_by_cell.get(cell_type),
                    "loading_rank": ranks[cell_index],
                    "loading_real": loading.real,
                    "loading_imag": loading.imag,
                    "loading_magnitude": abs(loading),
                    "energy_fraction": abs(loading) ** 2,
                }
            )
    result = pd.DataFrame(rows).sort_values(["mode", "loading_rank"]).reset_index(drop=True)
    if not np.allclose(result.groupby("mode")["energy_fraction"].sum(), 1.0):
        raise ValueError("Schur-mode energy fractions do not sum to one.")
    return result


def summarize_schur_modes(
    loadings: pd.DataFrame,
    high_loading_threshold: float = 0.05,
) -> pd.DataFrame:
    """Return one row per mode, including dominant region and E/I counts."""
    rows = []
    for mode, group in loadings.groupby("mode", sort=True):
        ordered = group.sort_values("loading_rank")
        dominant = ordered.iloc[0]
        high = ordered[ordered["loading_magnitude"] > high_loading_threshold]
        high_i = high[high["ei"] == "i"]
        rows.append(
            {
                "mode": int(mode),
                "dominant_region": dominant["region"],
                "dominant_cell_type": dominant["cell_type"],
                "dominant_loading": dominant["loading_magnitude"],
                "eigenvalue_real": dominant["eigenvalue_real"],
                "eigenvalue_imag": dominant["eigenvalue_imag"],
                "eigenvalue_magnitude": dominant["eigenvalue_magnitude"],
                "n_high_loading_cells": len(high),
                "n_high_loading_inhibitory": len(high_i),
                "high_loading_inhibitory_cells": ", ".join(high_i["cell_type"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["dominant_region", "eigenvalue_magnitude"],
        ascending=[True, False],
    )


def summarize_schur_regions(mode_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the mode summary by dominant anatomical region."""
    return (
        mode_summary.groupby("dominant_region")
        .agg(
            n_modes=("mode", "size"),
            mean_high_loading_inhibitory=("n_high_loading_inhibitory", "mean"),
            max_high_loading_inhibitory=("n_high_loading_inhibitory", "max"),
            mean_dominant_loading=("dominant_loading", "mean"),
        )
        .sort_values("n_modes", ascending=False)
    )


def save_schur_mode_outputs(
    model: SchurPropagationModel,
    loadings: pd.DataFrame,
    mode_summary: pd.DataFrame,
    region_summary: pd.DataFrame,
    output_dir: str | Path,
    *,
    matrix_path: str | Path,
    target_spectral_radius: float,
    high_loading_threshold: float,
) -> Path:
    """Save exact Schur arrays, tidy tables, and analysis metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "schur_modes.npz",
        A=model.A,
        T=model.T,
        Q=model.Q,
        eigenvalues=model.eigenvalues,
        labels=np.asarray(model.labels),
    )
    loadings.to_csv(output_dir / "schur_mode_loadings.csv", index=False)
    mode_summary.to_csv(output_dir / "schur_mode_summary.csv", index=False)
    region_summary.to_csv(output_dir / "schur_region_summary.csv")
    metadata = {
        "matrix_path": str(matrix_path),
        "normalization": model.normalization,
        "target_spectral_radius": target_spectral_radius,
        "scale_factor": model.scale_factor,
        "original_spectral_radius": model.original_spectral_radius,
        "high_loading_threshold": high_loading_threshold,
        "mode_indexing": "zero-based Schur order",
    }
    with (output_dir / "schur_modes_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return output_dir.resolve()


def propagate_signal(
    model: SchurPropagationModel,
    cell_types: Sequence[str],
    amplitudes: Sequence[float] | Mapping[str, float] | None = None,
    timesteps: int = MAX_TIMESTEPS,
) -> dict[str, object]:
    """Propagate the combined input through the complete Schur system."""
    steps = _check_timesteps(timesteps)
    x0 = make_initial_signal(model, cell_types, amplitudes)
    z = model.Q.conj().T @ x0
    states = np.empty((steps + 1, model.n_modes), dtype=complex)
    coordinates = np.empty_like(states)
    for t in range(steps + 1):
        coordinates[t] = z
        states[t] = model.Q @ z
        z = model.T @ z
    return {
        "time": np.arange(steps + 1),
        "cell_states": states,
        "schur_coordinates": coordinates,
        "initial_state": x0,
        "labels": model.labels,
    }


def propagate_each_schur_mode(
    model: SchurPropagationModel,
    cell_types: Sequence[str],
    amplitudes: Sequence[float] | Mapping[str, float] | None = None,
    timesteps: int = MAX_TIMESTEPS,
    modes: Iterable[int] | None = None,
    coupled: bool = True,
) -> dict[int, np.ndarray]:
    """Propagate the portion of an input initially assigned to each Schur mode.

    Returns ``{mode_index: X}``, where each ``X`` has shape
    ``(timesteps + 1, n_cell_types)``.  With ``coupled=True``, a mode's initial
    coordinate may transfer through off-diagonal entries of T.  With
    ``coupled=False``, only its diagonal multiplier is retained.
    """
    steps = _check_timesteps(timesteps)
    x0 = make_initial_signal(model, cell_types, amplitudes)
    z0 = model.Q.conj().T @ x0
    selected = list(range(model.n_modes)) if modes is None else list(modes)
    if len(set(selected)) != len(selected):
        raise ValueError("modes cannot contain duplicates.")
    invalid = [k for k in selected if not isinstance(k, (int, np.integer)) or not 0 <= k < model.n_modes]
    if invalid:
        raise IndexError(f"Invalid Schur mode indexes: {invalid}")

    trajectories: dict[int, np.ndarray] = {}
    for mode in selected:
        z = np.zeros(model.n_modes, dtype=complex)
        z[mode] = z0[mode]
        X = np.empty((steps + 1, model.n_modes), dtype=complex)
        for t in range(steps + 1):
            X[t] = model.Q @ z
            if coupled:
                z = model.T @ z
            else:
                z[mode] *= model.T[mode, mode]
        trajectories[int(mode)] = np.real_if_close(X, tol=1000)
    return trajectories


def propagate_mode_cell_types(
    model: SchurPropagationModel,
    mode: int,
    timesteps: int = MAX_TIMESTEPS,
    top_n: int = MAX_CELL_TYPES,
    amplitude: complex = 1.0,
    coupled: bool = True,
) -> pd.DataFrame:
    """Propagate one unit Schur mode and report its principal cell types.

    The initial condition is ``z[mode] = amplitude`` and all other Schur
    coordinates are zero.  The reported cells are the ``top_n`` largest
    entries of ``Q[:, mode]``.  With ``coupled=True``, ``T`` transfers signal
    from this seed mode into downstream cascade modes; with ``False``, the
    trajectory is the isolated response ``Q[:, mode] * lambda[mode]**t``.
    """
    steps = _check_timesteps(timesteps)
    if not isinstance(mode, (int, np.integer)) or not 0 <= mode < model.n_modes:
        raise IndexError(f"mode must be between 0 and {model.n_modes - 1}.")
    if not 1 <= top_n <= MAX_CELL_TYPES:
        raise ValueError(f"top_n must be between 1 and {MAX_CELL_TYPES}.")

    cell_indexes = np.argsort(np.abs(model.Q[:, mode]))[::-1][:top_n]
    z = np.zeros(model.n_modes, dtype=complex)
    z[mode] = amplitude
    rows: list[dict[str, object]] = []
    for t in range(steps + 1):
        x = model.Q @ z
        for rank, cell in enumerate(cell_indexes, start=1):
            value = x[cell]
            loading = model.Q[cell, mode]
            rows.append(
                {
                    "seed_mode": int(mode),
                    "eigenvalue": model.eigenvalues[mode],
                    "timestep": t,
                    "loading_rank": rank,
                    "cell_type": format_cell_label(model.labels[cell]),
                    "initial_mode_loading": loading,
                    "signal_real": float(np.real(value)),
                    "signal_imag": float(np.imag(value)),
                    "signal_magnitude": float(np.abs(value)),
                }
            )
        if coupled:
            z = model.T @ z
        else:
            z[mode] *= model.T[mode, mode]
    return pd.DataFrame(rows)


def propagate_all_mode_cell_types(
    model: SchurPropagationModel,
    timesteps: int = MAX_TIMESTEPS,
    top_n: int = MAX_CELL_TYPES,
    amplitude: complex = 1.0,
    coupled: bool = True,
    modes: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Return principal-cell propagation for every requested seed mode."""
    selected = list(range(model.n_modes)) if modes is None else list(modes)
    if not selected:
        raise ValueError("At least one mode is required.")
    frames = [
        propagate_mode_cell_types(
            model,
            mode=int(mode),
            timesteps=timesteps,
            top_n=top_n,
            amplitude=amplitude,
            coupled=coupled,
        )
        for mode in selected
    ]
    return pd.concat(frames, ignore_index=True)


def trajectories_to_frame(
    trajectories: Mapping[int, np.ndarray],
    labels: Sequence[str],
    cell_types: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Convert per-mode trajectory arrays to tidy plotting/export form."""
    shown = list(labels) if cell_types is None else list(cell_types)
    missing = [name for name in shown if name not in labels]
    if missing:
        raise KeyError(f"Unknown readout cell type(s): {missing}")
    columns = [list(labels).index(name) for name in shown]
    rows: list[dict[str, object]] = []
    for mode, values in trajectories.items():
        for t in range(values.shape[0]):
            for name, col in zip(shown, columns):
                value = values[t, col]
                rows.append(
                    {
                        "mode": mode,
                        "timestep": t,
                        "cell_type": format_cell_label(name),
                        "signal_real": float(np.real(value)),
                        "signal_imag": float(np.imag(value)),
                        "signal_magnitude": float(np.abs(value)),
                    }
                )
    return pd.DataFrame(rows)


def mode_activation_summary(
    model: SchurPropagationModel,
    cell_types: Sequence[str],
    amplitudes: Sequence[float] | Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Rank Schur modes by their initial activation from a cell-type signal."""
    x0 = make_initial_signal(model, cell_types, amplitudes)
    z0 = model.Q.conj().T @ x0
    frame = pd.DataFrame(
        {
            "mode": np.arange(model.n_modes),
            "eigenvalue": model.eigenvalues,
            "eigenvalue_magnitude": np.abs(model.eigenvalues),
            "initial_coordinate": z0,
            "initial_magnitude": np.abs(z0),
        }
    )
    return frame.sort_values("initial_magnitude", ascending=False, ignore_index=True)


def plot_mode_trajectories(
    frame: pd.DataFrame,
    modes: Sequence[int],
    value: str = "signal_real",
    figsize: tuple[float, float] = (11, 5),
):
    """Plot selected mode/cell-type trajectories from a tidy frame."""
    if value not in {"signal_real", "signal_imag", "signal_magnitude"}:
        raise ValueError("value must be signal_real, signal_imag, or signal_magnitude.")
    subset = frame[frame["mode"].isin(modes)]
    if subset.empty:
        raise ValueError("No requested modes are present in the frame.")
    fig, ax = plt.subplots(figsize=figsize)
    for (mode, cell), group in subset.groupby(["mode", "cell_type"], sort=True):
        ax.plot(group["timestep"], group[value], marker="o", label=f"mode {mode}: {cell}")
    ax.set(xlabel="Timestep", ylabel=value.replace("_", " ").title())
    ax.grid(alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig, ax


def plot_seed_mode_cell_types(
    frame: pd.DataFrame,
    mode: int,
    value: str = "signal_real",
    figsize: tuple[float, float] = (9, 5),
):
    """Plot the individual principal-cell signals for one seed mode."""
    if value not in {"signal_real", "signal_imag", "signal_magnitude"}:
        raise ValueError("value must be signal_real, signal_imag, or signal_magnitude.")
    subset = frame[frame["seed_mode"] == mode]
    if subset.empty:
        raise ValueError(f"Seed mode {mode} is not present in the frame.")
    fig, ax = plt.subplots(figsize=figsize)
    for cell, group in subset.groupby("cell_type", sort=False):
        ax.plot(group["timestep"], group[value], marker="o", label=cell)
    eigenvalue = subset["eigenvalue"].iloc[0]
    ax.set(
        title=f"Seed Schur mode {mode} (lambda={eigenvalue:.4g})",
        xlabel="Timestep",
        ylabel=value.replace("_", " ").title(),
    )
    ax.grid(alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig, ax


def compare_self_propagation(
    csv_path: str | Path,
    *,
    normalization: str = "spectral_radius",
    target_spectral_radius: float = 0.95,
    timesteps: int = MAX_TIMESTEPS,
    top_n: int = MAX_CELL_TYPES,
) -> dict[str, object]:
    """Compare all-mode propagation after separately normalizing self variants."""
    models = {
        "with_self": prepare_schur_model(
            csv_path, normalization, target_spectral_radius, include_self=True
        ),
        "no_self": prepare_schur_model(
            csv_path, normalization, target_spectral_radius, include_self=False
        ),
    }
    frames, summaries = [], []
    for variant, model in models.items():
        frame = propagate_all_mode_cell_types(
            model, timesteps=timesteps, top_n=top_n, coupled=True
        ).copy()
        frame.insert(0, "variant", variant)
        frames.append(frame)
        summaries.append(
            {
                "variant": variant,
                "normalization": model.normalization,
                "original_spectral_radius": model.original_spectral_radius,
                "achieved_spectral_radius": float(np.max(np.abs(model.eigenvalues))),
                "scale_factor": model.scale_factor,
                "matrix_fro_norm": float(np.linalg.norm(model.A, "fro")),
                "non_normality": float(
                    np.linalg.norm(model.A @ model.A.T - model.A.T @ model.A, "fro")
                ),
            }
        )
    propagation = pd.concat(frames, ignore_index=True)
    timestep_summary = (
        propagation.groupby(["variant", "timestep"])["signal_magnitude"]
        .agg(["mean", "median", "max", "sum"])
        .reset_index()
    )
    return {
        "models": models,
        "propagation": propagation,
        "model_summary": pd.DataFrame(summaries),
        "timestep_summary": timestep_summary,
    }
