"""Utilities for hippocampal Schur-mode notebook analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg


REGIME_THRESHOLD = 0.15
ACTIVE_THRESHOLD = 0.20


_EXCITATORY_LABEL_MARKERS = (
    "pyramidal",
    "principal",
    "granule",
    "semilunar",
    "stellate",
    "back projection",
    "mossy",
    "total molecular layer",
)

_INHIBITORY_LABEL_MARKERS = (
    "interneuron",
    "axo axonic",
    "basket",
    "bistratified",
    "ivy",
    "neurogliaform",
    "o lm",
    "o-lm",
    "lm-r",
    "lmr",
    "oriens",
    "radiatum",
    "trilaminar",
    "quadrilaminar",
    "perforant path associated",
    "apical targeting",
    "mossy fiber associated",
    "hipp",
    "hicap",
    "hiprom",
    "mopp",
    "molax",
    "mpr",
)


def infer_cell_ei(label) -> str:
    """Infer E/I type from the cell-type name for compact printed labels."""
    text = str(label).strip()
    lowered = text.lower()
    if lowered.endswith("(e)"):
        return "E"
    if lowered.endswith("(i)"):
        return "I"
    if any(marker in lowered for marker in _INHIBITORY_LABEL_MARKERS):
        return "I"
    if any(marker in lowered for marker in _EXCITATORY_LABEL_MARKERS):
        return "E"
    return "I"


def format_cell_label(label) -> str:
    """Append an E/I suffix to a cell-type label, without duplicating it."""
    text = str(label)
    stripped = text.strip()
    if stripped.lower().endswith("(e)") or stripped.lower().endswith("(i)"):
        return text
    return f"{text} ({infer_cell_ei(text)})"


def format_cell_labels(labels) -> list[str]:
    """Return all labels with E/I suffixes for printed tables and plots."""
    return [format_cell_label(label) for label in labels]


def load_connectivity_matrix(csv_path: str) -> pd.DataFrame:
    """Load a labeled connectivity matrix from CSV."""
    df = pd.read_csv(csv_path, index_col=0)
    print(f"Loaded connectivity matrix of shape: {df.shape}")
    return df


def normalize_matrix(matrix, method: str = "spectral_radius", target_spectral_radius: float = 1.0) -> dict:
    """Normalize a square connectivity matrix and return matrix plus diagnostics."""
    matrix = np.asarray(matrix, dtype=float)

    if method == "none":
        M = matrix.copy()
        eigvals_raw = np.linalg.eigvals(M)
        rho = float(np.max(np.abs(eigvals_raw)))
        return {
            "matrix": M,
            "norm_label": "No normalization",
            "method": method,
            "rho_original": rho,
            "rho_achieved": rho,
            "scale_factor": 1.0,
            "row_abs_sums": None,
            "non_normality": _non_normality(M),
        }

    if method == "row_l1":
        row_abs_sums = np.sum(np.abs(matrix), axis=1)
        row_abs_sums[row_abs_sums == 0] = 1
        M = matrix / row_abs_sums[:, np.newaxis]
        return {
            "matrix": M,
            "norm_label": "Row L1 (Markov) Normalization",
            "method": method,
            "rho_original": float(np.max(np.abs(np.linalg.eigvals(matrix)))),
            "rho_achieved": float(np.max(np.abs(np.linalg.eigvals(M)))),
            "scale_factor": None,
            "row_abs_sums": row_abs_sums,
            "non_normality": _non_normality(M),
        }

    if method == "spectral_radius":
        eigvals_raw = np.linalg.eigvals(matrix)
        rho = float(np.max(np.abs(eigvals_raw)))
        if rho == 0:
            raise ValueError("Cannot spectral-radius normalize a matrix with zero spectral radius.")
        scale_factor = target_spectral_radius / rho
        M = matrix * scale_factor
        return {
            "matrix": M,
            "norm_label": f"Spectral Radius Normalization (target rho = {target_spectral_radius})",
            "method": method,
            "rho_original": rho,
            "rho_achieved": float(np.max(np.abs(np.linalg.eigvals(M)))),
            "scale_factor": scale_factor,
            "row_abs_sums": None,
            "non_normality": _non_normality(M),
        }

    raise ValueError("method must be one of: 'none', 'row_l1', or 'spectral_radius'")


def print_normalization_summary(norm: dict, target_spectral_radius: float | None = None) -> None:
    """Print compact normalization diagnostics."""
    method = norm["method"]
    if method == "none":
        print("Method: None")
        print(f"Spectral radius: {norm['rho_achieved']:.6f}")
    elif method == "row_l1":
        print("Method: Row L1 normalization")
        print("Row |L1| sums (first 10):", np.sum(np.abs(norm["matrix"]), axis=1)[:10].round(6))
    elif method == "spectral_radius":
        print("Method: Spectral radius normalization")
        print(f"  Original spectral radius: {norm['rho_original']:.6f}")
        if target_spectral_radius is not None:
            print(f"  Target spectral radius:   {target_spectral_radius}")
        print(f"  Achieved spectral radius: {norm['rho_achieved']:.6f}")
        print(f"  Scale factor applied:     {norm['scale_factor']:.6e}")
        print(f"  Non-normality ||MM^T - M^TM||_F: {norm['non_normality']:.6f}")
    print(f"\nMatrix shape: {norm['matrix'].shape}")
    print(f"Normalization: {norm['norm_label']}")


def _non_normality(matrix) -> float:
    matrix = np.asarray(matrix, dtype=float)
    comm = matrix @ matrix.T - matrix.T @ matrix
    return float(np.linalg.norm(comm, "fro"))


def schur_decomp(input_matrix, labels=None, transpose: bool = True, output: str = "complex", normalize_signs: bool = True) -> dict:
    """Run Schur decomposition on a matrix copy and return reusable results."""
    source_matrix = np.asarray(input_matrix, dtype=float).copy()
    work_matrix = source_matrix.T.copy() if transpose else source_matrix.copy()
    T, Q = linalg.schur(work_matrix, output=output)

    if normalize_signs and np.iscomplexobj(Q):
        signs = np.sign(Q[np.argmax(np.abs(Q), axis=0), np.arange(Q.shape[1])])
        signs[signs == 0] = 1
        Q = Q * signs[np.newaxis, :]

    return {
        "source_matrix": source_matrix,
        "work_matrix": work_matrix,
        "T": T,
        "Q": Q,
        "eigenvalues": np.diag(T),
        "labels": labels,
        "transpose": transpose,
        "output": output,
    }


def prepare_schur_variants(matrix, labels=None) -> dict:
    """Create with-self and no-self Schur decompositions in complex and real form."""
    with_self = np.asarray(matrix, dtype=float).copy()
    no_self = with_self.copy()
    np.fill_diagonal(no_self, 0)
    variants = {
        "with_self": with_self,
        "no_self": no_self,
        "schur_with_self": schur_decomp(with_self, labels=labels, output="complex"),
        "schur_no_self": schur_decomp(no_self, labels=labels, output="complex"),
        "schur_with_self_real": schur_decomp(with_self, labels=labels, output="real", normalize_signs=False),
        "schur_no_self_real": schur_decomp(no_self, labels=labels, output="real", normalize_signs=False),
    }
    return variants


def print_schur_variant_summary(variants: dict) -> None:
    """Print a small comparison of with-self and no-self decompositions."""
    print("Reusable Schur decompositions ready.")
    for matrix_name, result_name in [("with_self", "schur_with_self"), ("no_self", "schur_no_self")]:
        result = variants[result_name]
        rho = np.max(np.abs(result["eigenvalues"]))
        diag_sum = np.trace(variants[matrix_name])
        print(f"  {matrix_name:<9} | spectral radius = {rho:.6f} | source diagonal sum = {diag_sum:.6g}")


def activate_schur(result: dict):
    """Return notebook-friendly core variables from a Schur result."""
    return result["work_matrix"], result["T"], result["Q"], result["eigenvalues"]


def unique_schur_modes(eigenvalues, Q, tol: float = 1e-5) -> list[tuple[str, complex, np.ndarray, int]]:
    """Group real modes and complex conjugate pairs by descending eigenvalue magnitude."""
    seen = set()
    idx_sorted = np.argsort(np.abs(eigenvalues))[::-1]
    unique_modes = []

    for i in idx_sorted:
        if i in seen:
            continue
        val = eigenvalues[i]
        if np.abs(np.imag(val)) < tol:
            unique_modes.append(("Real", np.real(val), Q[:, i], i))
            seen.add(i)
            continue

        found_conj = False
        for j in idx_sorted:
            if j == i or j in seen:
                continue
            val2 = eigenvalues[j]
            if np.abs(np.real(val) - np.real(val2)) < tol and np.abs(np.imag(val) + np.imag(val2)) < tol:
                unique_modes.append(("Complex Pair", val, Q[:, i], i))
                seen.add(i)
                seen.add(j)
                found_conj = True
                break

        if not found_conj:
            unique_modes.append(("Real", val, Q[:, i], i))
            seen.add(i)

    return unique_modes


def format_coupling(coupling) -> str:
    """Format real or complex Schur coupling values."""
    return f"{coupling.real:.4f} + {coupling.imag:.4f}i" if np.iscomplexobj(coupling) else f"{coupling:.4f}"


def describe_coupling(coupling) -> str:
    """Describe the qualitative behavior of a Schur coupling."""
    if np.abs(np.imag(coupling)) > 1e-9:
        return "phase-shifted oscillatory transfer"
    if np.real(coupling) > 0:
        return "reinforcing transfer"
    if np.real(coupling) < 0:
        return "opposing / inverting transfer"
    return "no resolved transfer"


def mode_dynamics(mode_type: str, eigenvalue) -> tuple[str, str]:
    """Return printable eigenvalue and dynamic descriptions for a mode."""
    if mode_type == "Real":
        val_str = f"{np.real(eigenvalue):.4f}"
        dyn_desc = "Non-oscillatory. " + (
            "Amplification / feedforward growth." if np.real(eigenvalue) > 0 else "Local damping / feedback stabilization."
        )
    else:
        val_str = f"{np.real(eigenvalue):.4f} +/- {np.abs(np.imag(eigenvalue)):.4f}i"
        dyn_desc = f"Oscillation ({np.abs(np.imag(eigenvalue)):.4f} rad/s) with "
        dyn_desc += "growing" if np.real(eigenvalue) > 0 else "damping"
        dyn_desc += " amplitude."
    return val_str, dyn_desc


def print_unique_schur_modes(unique_modes, T_schur, Q, labels, top_n: int = 5) -> None:
    """Print mode dynamics, strongest Schur coupling, and top participating cells."""
    print(f"Identified {len(unique_modes)} unique Schur modes.")
    print("(Eigenvalues from diag(T); spatial structure from Q columns)")
    for idx, (mode_type, val, vec, schur_idx) in enumerate(unique_modes):
        print("=" * 80)
        val_str, dyn_desc = mode_dynamics(mode_type, val)
        print(f"Mode {idx + 1}: Type = {mode_type} | Eigenvalue = {val_str}")
        print(f"Dynamics: {dyn_desc}")

        forward_couplings = T_schur[schur_idx, (schur_idx + 1):]
        if len(forward_couplings) > 0 and np.max(np.abs(forward_couplings)) > 1e-9:
            strongest_next_offset = np.argmax(np.abs(forward_couplings)) + 1
            target_mode_idx = schur_idx + strongest_next_offset
            coupling = T_schur[schur_idx, target_mode_idx]
            target_cell_name = format_cell_label(labels[np.argmax(np.abs(Q[:, target_mode_idx]))])
            print(
                f"Largest Schur coupling: Mode {target_mode_idx + 1} | "
                f"coupling = {format_coupling(coupling)} | "
                f"behavior: {describe_coupling(coupling)} toward {target_cell_name}"
            )
        else:
            print("Largest Schur coupling: terminal / no resolved downstream coupling")

        print("Principal Contributing Cell Types (Schur vector):")
        top_indices = np.argsort(np.abs(vec))[::-1][:top_n]
        for tidx in top_indices:
            cell_name = format_cell_label(labels[tidx])
            w = vec[tidx]
            energy_pct = (np.abs(w) ** 2) * 100
            weight_str = f"weight = {np.real(w):.4f}" if mode_type == "Real" else f"weight = {np.real(w):.4f} + {np.imag(w):.4f}i"
            print(f"  - {cell_name:<45} | {weight_str} ({energy_pct:5.1f}%)")


def compute_ei_routing_ratios(weight_matrix, active_ids, node_labels=None) -> dict:
    """Compute excitatory and inhibitory loop/leak routing for a node subset."""
    weights = np.asarray(weight_matrix, dtype=float)
    active_ids = np.asarray(active_ids, dtype=int)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("weight_matrix must be a square 2D array.")
    if active_ids.ndim != 1 or active_ids.size == 0:
        raise ValueError("active_ids must be a non-empty 1D array of node indices.")
    if np.any((active_ids < 0) | (active_ids >= weights.shape[0])):
        raise IndexError("active_ids contains an index outside the matrix dimensions.")
    if node_labels is not None and len(node_labels) != weights.shape[0]:
        raise ValueError("node_labels must have the same length as weight_matrix dimensions.")

    active_rows = weights[active_ids, :]
    active_submatrix = weights[np.ix_(active_ids, active_ids)]
    e_total_out = np.clip(active_rows, 0, None).sum()
    e_loop_out = np.clip(active_submatrix, 0, None).sum()
    i_total_out = np.abs(np.clip(active_rows, None, 0)).sum()
    i_loop_out = np.abs(np.clip(active_submatrix, None, 0)).sum()
    e_loop = e_loop_out / e_total_out if e_total_out > 0 else np.nan
    i_loop = i_loop_out / i_total_out if i_total_out > 0 else np.nan
    return {
        "e_loop": e_loop,
        "i_loop": i_loop,
        "e_leak": 1.0 - e_loop if np.isfinite(e_loop) else np.nan,
        "i_leak": 1.0 - i_loop if np.isfinite(i_loop) else np.nan,
        "e_total_out": e_total_out,
        "i_total_out": i_total_out,
        "active_ids": active_ids.copy(),
        "n_active": int(active_ids.size),
    }


def classify_circuit_regime(e_loop, i_loop, threshold: float = REGIME_THRESHOLD) -> str:
    """Assign a circuit-regime label from E/I loop ratios."""
    if not np.isfinite(e_loop) or not np.isfinite(i_loop):
        return "Undefined Routing"
    if e_loop >= threshold and i_loop >= threshold:
        return "Balanced Recurrent (ISN Candidate)"
    if e_loop < threshold and i_loop >= threshold:
        return "Feedforward E / Local I (Relay+Gate)"
    if e_loop < threshold and i_loop < threshold:
        return "Pure Relay / Broadcast"
    return "Recurrent E Amplifier"


def active_edge_counts(weight_matrix, active_ids) -> dict:
    """Count nonzero positive/negative edges inside the active submatrix."""
    active_submatrix = np.asarray(weight_matrix, dtype=float)[np.ix_(active_ids, active_ids)]
    return {
        "n_pos_active": int(np.count_nonzero(active_submatrix > 0)),
        "n_neg_active": int(np.count_nonzero(active_submatrix < 0)),
    }


def build_mode_routing_table(unique_modes, weight_matrix, labels, active_threshold: float = ACTIVE_THRESHOLD, regime_threshold: float = REGIME_THRESHOLD):
    """Build a DataFrame and detail records for E/I routing across Schur modes."""
    records = []
    details = []
    for mode_idx, (mode_type, eigval, vec, schur_col) in enumerate(unique_modes, start=1):
        active_ids = np.where(np.abs(vec) > active_threshold)[0]
        if active_ids.size == 0:
            active_ids = np.array([np.argmax(np.abs(vec))])
        routing = compute_ei_routing_ratios(weight_matrix, active_ids, node_labels=labels)
        edge_counts = active_edge_counts(weight_matrix, routing["active_ids"])
        dominant_idx = int(np.argmax(np.abs(vec)))
        circuit_regime = classify_circuit_regime(routing["e_loop"], routing["i_loop"], threshold=regime_threshold)
        records.append({
            "mode_idx": mode_idx,
            "eigenvalue_real": float(np.real(eigval)),
            "eigenvalue_imag": float(np.imag(eigval)),
            "mode_type": mode_type,
            "e_loop": routing["e_loop"],
            "i_loop": routing["i_loop"],
            "e_leak": routing["e_leak"],
            "i_leak": routing["i_leak"],
            "n_active": routing["n_active"],
            "dominant_cell": format_cell_label(labels[dominant_idx]),
        })
        details.append({
            "mode_idx": mode_idx,
            "schur_col": schur_col,
            "vector": vec,
            "active_ids": routing["active_ids"],
            "active_cells": format_cell_labels(
                labels[routing["active_ids"]].tolist()
                if hasattr(labels, "__getitem__")
                else [labels[i] for i in routing["active_ids"]]
            ),
            "e_total_out": routing["e_total_out"],
            "i_total_out": routing["i_total_out"],
            "circuit_regime": circuit_regime,
            **edge_counts,
        })
    return pd.DataFrame(records), details


def plot_ei_routing(mode_routing_df):
    """Plot E/I loop ratios across Schur modes."""
    plot_df = mode_routing_df.copy()
    plot_df["eigenvalue_abs"] = np.hypot(plot_df["eigenvalue_real"], plot_df["eigenvalue_imag"])
    plot_df["undefined_ratio"] = plot_df[["e_loop", "i_loop"]].isna().any(axis=1)
    plot_df["e_loop_plot"] = plot_df["e_loop"].fillna(0.0)
    plot_df["i_loop_plot"] = plot_df["i_loop"].fillna(0.0)

    real_extent = np.nanmax(np.abs(plot_df["eigenvalue_real"])) if len(plot_df) else 1.0
    real_extent = real_extent if real_extent > 0 else 1.0
    size_extent = np.nanmax(plot_df["eigenvalue_abs"]) if len(plot_df) else 1.0
    size_extent = size_extent if size_extent > 0 else 1.0
    norm = plt.Normalize(vmin=-real_extent, vmax=real_extent)
    point_sizes = 80 + 420 * (plot_df["eigenvalue_abs"] / size_extent)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    finite_mask = ~plot_df["undefined_ratio"]
    undefined_mask = plot_df["undefined_ratio"]
    scatter = ax.scatter(
        plot_df.loc[finite_mask, "e_loop_plot"],
        plot_df.loc[finite_mask, "i_loop_plot"],
        c=plot_df.loc[finite_mask, "eigenvalue_real"],
        s=point_sizes.loc[finite_mask],
        cmap="RdBu_r",
        norm=norm,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.88,
        zorder=3,
        label="Defined E/I ratios",
    )
    if undefined_mask.any():
        ax.scatter(
            plot_df.loc[undefined_mask, "e_loop_plot"],
            plot_df.loc[undefined_mask, "i_loop_plot"],
            c=plot_df.loc[undefined_mask, "eigenvalue_real"],
            s=point_sizes.loc[undefined_mask],
            cmap="RdBu_r",
            norm=norm,
            marker="s",
            edgecolor="black",
            linewidth=0.9,
            alpha=0.72,
            zorder=3,
            label="Undefined ratio plotted at 0 boundary",
        )
    for _, row in plot_df.iterrows():
        ax.annotate(int(row["mode_idx"]), (row["e_loop_plot"], row["i_loop_plot"]), xytext=(4, 4), textcoords="offset points", fontsize=7, color="black", zorder=4)
    ax.axvline(0.5, color="0.25", linestyle="--", linewidth=1.0, alpha=0.8, zorder=2)
    ax.axhline(0.5, color="0.25", linestyle="--", linewidth=1.0, alpha=0.8, zorder=2)
    quadrant_labels = [
        (0.04, 0.94, "Feedforward E / Local I\n(Relay+Gate)", "left", "top"),
        (0.96, 0.94, "Balanced Recurrent\n(ISN Candidate)", "right", "top"),
        (0.04, 0.06, "Pure Relay / Broadcast", "left", "bottom"),
        (0.96, 0.06, "Recurrent E Amplifier", "right", "bottom"),
    ]
    for x, y, label, ha, va in quadrant_labels:
        ax.text(x, y, label, transform=ax.transAxes, ha=ha, va=va, fontsize=10, color="0.15", bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.78})
    legend_levels = np.linspace(plot_df["eigenvalue_abs"].min(), plot_df["eigenvalue_abs"].max(), 3) if len(plot_df) else []
    size_handles = []
    for level in legend_levels:
        marker_size = 80 + 420 * (level / size_extent)
        size_handles.append(ax.scatter([], [], s=marker_size, facecolor="white", edgecolor="black", linewidth=0.6, label=f"{level:.2f}"))
    if size_handles:
        size_legend = ax.legend(handles=size_handles, title="|eigenvalue|", loc="lower right", frameon=True, framealpha=0.9, fontsize=8, title_fontsize=9)
        ax.add_artist(size_legend)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=False, fontsize=8)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Eigenvalue real part")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("E Loop Ratio")
    ax.set_ylabel("I Loop Ratio")
    ax.set_title("E/I Routing Ratios Across Schur Modes")
    ax.grid(True, color="0.85", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig, ax


def format_percent(value) -> str:
    """Format a finite value as a percent, otherwise nan."""
    return "nan" if not np.isfinite(value) else f"{value:.2%}"


def print_mode_routing_details(mode_routing_df, mode_routing_details) -> None:
    """Print detailed E/I routing lines for each mode."""
    for _, row in mode_routing_df.iterrows():
        detail = mode_routing_details[int(row["mode_idx"]) - 1]
        print("=" * 80)
        print(
            f"Mode {int(row['mode_idx'])}: "
            f"Type = {row['mode_type']} | "
            f"Eigenvalue = {row['eigenvalue_real']:.4f} + {row['eigenvalue_imag']:.4f}i | "
            f"Dominant cell = {row['dominant_cell']}"
        )
        print(
            "E Routing:  "
            f"Loop = {format_percent(row['e_loop'])} | "
            f"Leak = {format_percent(row['e_leak'])}  "
            f"[n_pos_active={detail['n_pos_active']}, total_E_out={detail['e_total_out']:.6g}]"
        )
        print(
            "I Routing:  "
            f"Loop = {format_percent(row['i_loop'])} | "
            f"Leak = {format_percent(row['i_leak'])}  "
            f"[n_neg_active={detail['n_neg_active']}, total_I_out={detail['i_total_out']:.6g}]"
        )
        print(f"Circuit Regime: {detail['circuit_regime']}")


def print_motif_routing(weight_matrix, motif_ids, labels) -> dict:
    """Print routing details for an arbitrary motif index set."""
    motif_ids = np.asarray(motif_ids, dtype=int)
    ratios = compute_ei_routing_ratios(weight_matrix, motif_ids, node_labels=labels)
    print("Example trimer node indices:", motif_ids.tolist())
    print("Example trimer cell types:")
    for node_id in motif_ids:
        print(f"  {node_id:2d}: {format_cell_label(labels[node_id])}")
    print("\nExample trimer E/I routing ratios:")
    for key in ["e_loop", "e_leak", "i_loop", "i_leak", "e_total_out", "i_total_out", "n_active"]:
        value = ratios[key]
        if isinstance(value, float):
            print(f"  {key}: {value:.6g}")
        else:
            print(f"  {key}: {value}")
    return ratios


def plot_eigenvalue_spectrum(eigenvalues, annotate_top: int = 10):
    """Plot Schur eigenvalues in the complex plane."""
    plt.figure(figsize=(12, 10))
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.scatter(eigenvalues.real, eigenvalues.imag, color="forestgreen", alpha=0.7, edgecolors="k", s=60)
    idx_sorted = np.argsort(np.abs(eigenvalues))[::-1]
    annotated_count = 0
    for rank, idx in enumerate(idx_sorted, start=1):
        if annotated_count >= annotate_top:
            break
        ev = eigenvalues[idx]
        plt.text(ev.real, ev.imag, str(rank), fontsize=9, ha="left", va="bottom", color="darkred")
        annotated_count += 1
    theta = np.linspace(0, 2 * np.pi, 400)
    plt.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.4, label="Unit circle")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.title("Eigenvalue Spectrum from Schur T Diagonal")
    plt.legend()
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_dominant_mode(eigenvalues, Q, labels):
    """Plot eigenvalue spectrum plus top dominant Schur-vector components."""
    dominant_index = np.argmax(np.abs(eigenvalues))
    dominant_eigenvalue = eigenvalues[dominant_index]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    ax1.scatter(eigenvalues.real, eigenvalues.imag, c="blue", alpha=0.6, label="Eigenvalues")
    ax1.scatter(dominant_eigenvalue.real, dominant_eigenvalue.imag, c="red", s=100, label="Dominant Eigenvalue")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Real Part")
    ax1.set_ylabel("Imaginary Part")
    ax1.set_title("Eigenvalue Spectrum")
    ax1.legend()
    eigenvector_magnitudes = np.abs(Q[:, dominant_index])
    eigenvector_df = pd.DataFrame({"Neuron": format_cell_labels(labels), "Magnitude": eigenvector_magnitudes}).sort_values("Magnitude", ascending=False).head(20)
    ax2.barh(eigenvector_df["Neuron"], eigenvector_df["Magnitude"], color="purple")
    ax2.set_xlabel("Magnitude")
    ax2.set_title("Top 20 Neuron Contributions to the Dominant Network Mode")
    ax2.invert_yaxis()
    plt.tight_layout()
    plt.show()
    print(f"The dominant eigenvalue is: {dominant_eigenvalue}")
    print(f"The magnitude (spectral radius) of the dominant eigenvalue is: {np.abs(dominant_eigenvalue)}")
    return eigenvector_df


def plot_schur_t_heatmap(T_schur):
    """Visualize Re(T), where columns are sources and rows are receivers."""
    import matplotlib.colors as mcolors
    plt.figure(figsize=(12, 10))
    T_real = np.real(T_schur)
    mask = np.tril(np.ones_like(T_real, dtype=bool), k=-1)
    max_abs = np.max(np.abs(T_real)) if T_real.size else 1.0
    sns.heatmap(T_real, mask=mask, cmap="coolwarm", center=0, norm=mcolors.SymLogNorm(linthresh=1e-3, vmin=-max_abs, vmax=max_abs), cbar_kws={"label": "Schur T value"})
    plt.title("Schur T Matrix: Upper-Triangular Mode Couplings")
    plt.xlabel("Source Schur mode index j")
    plt.ylabel("Receiver Schur mode index i")
    plt.tight_layout()
    plt.show()


def first_schur_station_df(Q, labels, n: int = 5) -> pd.DataFrame:
    """Return and print top contributors in the first Schur vector."""
    station_df = pd.DataFrame({"Neuron": format_cell_labels(labels), "Contribution": np.abs(Q[:, 0])}).sort_values("Contribution", ascending=False)
    print("Top 5 neurons in first Schur vector (Q[:,0]):")
    print(station_df.head(n))
    return station_df


def print_complete_schur_pipeline(complete_schur, labels, coupling_tol: float = 1e-4, top_n: int = 5) -> None:
    """Enumerate the real Schur structural pipeline.

    For ``y_next = T @ y``, columns are sending modes and rows are receiving
    modes, so ``T[i, j]`` is the coupling from mode ``j`` into mode ``i``.
    """
    T = complete_schur["T"]
    U = complete_schur["Q"]
    num_modes = T.shape[0]
    print(f"Executing Complete Pipeline Analysis: Enumerate {num_modes} Structural Schur Modes.")
    print("=" * 100)
    idx = 0
    while idx < num_modes:
        vec = U[:, idx]
        val = T[idx, idx]
        is_oscillatory = False
        if idx < num_modes - 1 and T[idx + 1, idx] != 0:
            is_oscillatory = True
            real_part = T[idx, idx]
            imag_part = np.sqrt(np.abs(T[idx, idx + 1] * T[idx + 1, idx]))
            dyn_desc = f"Localized Oscillation Loop ({real_part:.4f} +/- {imag_part:.4f}i rad/s)"
            self_conn_context = "Strong local self-connections / recurrent feedback are forcing a rhythmic resonance."
        elif idx > 0 and T[idx, idx - 1] != 0:
            is_oscillatory = True
            real_part = T[idx, idx]
            imag_part = np.sqrt(np.abs(T[idx, idx - 1] * T[idx - 1, idx]))
            dyn_desc = f"Localized Oscillation Loop ({real_part:.4f} +/- {imag_part:.4f}i rad/s) [Conjugate Counterpart]"
            self_conn_context = "Acts as the paired quadrature component sustaining the local population rhythm."
        else:
            growth_status = "growing/amplifying" if val > 0 else "decaying/absorbing"
            dyn_desc = f"Pure Directional/Feedforward Vector (Real Magnitude: {val:.4f}) [{growth_status}]"
            self_conn_context = "Minimal local self-connections; the population acts primarily as a direct spatial pipeline or transient relay."

        coupling_context = "No resolved incoming coupling."
        incoming_couplings = np.abs(T[idx, :]).copy()
        incoming_couplings[idx] = 0
        source_mode_idx = int(np.argmax(incoming_couplings))
        if incoming_couplings[source_mode_idx] > coupling_tol:
            coupling_strength = T[idx, source_mode_idx]
            coupling_behavior = "reinforcing transfer" if coupling_strength > 0 else "opposing / inverting transfer"
            source_vec = U[:, source_mode_idx]
            source_cell_name = format_cell_label(labels[np.argmax(np.abs(source_vec))])
            coupling_context = (
                f"Largest incoming coupling is from sending Mode {source_mode_idx + 1} "
                f"to receiving Mode {idx + 1} (Coupling: {coupling_strength:.4f}). "
                f"Behavior is {coupling_behavior} from the population led by {source_cell_name}."
            )

        print(f"MODE {idx + 1}:")
        print(f"  Mathematical Dynamics: {dyn_desc}")
        print(f"  Self-Connection Context: {self_conn_context}")
        print(f"  Largest Incoming Coupling: {coupling_context}")
        print("  Principal Population Makeup:")
        top_indices = np.argsort(np.abs(vec))[::-1][:top_n]
        for tidx in top_indices:
            cell_name = format_cell_label(labels[tidx])
            weight = vec[tidx]
            energy_pct = (np.abs(weight) ** 2) * 100
            print(f"    - {cell_name:<45} | Weight: {weight: .4f} ({energy_pct:5.1f}%)")
        print("=" * 100)
        idx += 1


def find_autocentric_neurons(df_numeric: pd.DataFrame) -> pd.DataFrame:
    """Find rows whose diagonal self-connection exceeds every non-self outgoing entry."""
    self_connections = np.diag(df_numeric)
    arr = df_numeric.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(arr, -np.inf)
    max_outgoing = arr.max(axis=1)
    records = []
    for i, neuron in enumerate(df_numeric.index):
        if self_connections[i] > max_outgoing[i]:
            records.append({"Neuron": format_cell_label(neuron), "Self-Connection": self_connections[i], "Max Outgoing": max_outgoing[i]})
    return pd.DataFrame(records)


def plot_autocentric_neurons(autocentric_df: pd.DataFrame) -> None:
    """Plot autocentric neuron self-connections against strongest outgoing connections."""
    if autocentric_df.empty:
        print("No neuron types were found that favor their own cell type over all others.")
        print("This indicates that all neuron types in this dataset have at least one outgoing connection to another type that is stronger than their own recurrent connection.")
        return
    autocentric_df = autocentric_df.sort_values("Self-Connection", ascending=True)
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(autocentric_df))
    plt.barh(y_pos, autocentric_df["Self-Connection"], height=0.4, align="center", color="gold", label="Self-Connection (Recurrent)")
    plt.barh(y_pos - 0.4, autocentric_df["Max Outgoing"], height=0.4, align="center", color="gray", label="Strongest Outgoing Connection")
    plt.yticks(y_pos, autocentric_df["Neuron"])
    plt.xlabel("Connection Strength")
    plt.title('"Autocentric" Neuron Types Favoring Self-Connection')
    plt.legend()
    plt.grid(True, axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
    print("--- Autocentric Neuron Types ---")
    print(autocentric_df)

def dominant_mode_summary(schur_result: dict, labels, top_n: int = 10) -> dict:
    """Summarize the dominant Schur mode by spectral radius and top vector contributors."""
    eigenvalues = schur_result["eigenvalues"]
    Q = schur_result["Q"]
    dominant_idx = int(np.argmax(np.abs(eigenvalues)))
    dominant_eigenvalue = eigenvalues[dominant_idx]
    vec = Q[:, dominant_idx]
    top_indices = np.argsort(np.abs(vec))[::-1][:top_n]
    top = pd.DataFrame({
        "rank": np.arange(1, len(top_indices) + 1),
        "cell": [format_cell_label(labels[i]) for i in top_indices],
        "weight_real": np.real(vec[top_indices]),
        "weight_imag": np.imag(vec[top_indices]),
        "magnitude": np.abs(vec[top_indices]),
        "energy_pct": (np.abs(vec[top_indices]) ** 2) * 100,
    })
    return {
        "dominant_idx": dominant_idx,
        "dominant_eigenvalue": dominant_eigenvalue,
        "spectral_radius": float(np.abs(dominant_eigenvalue)),
        "top": top,
        "vector": vec,
    }


def compare_dominant_modes(schur_with_self: dict, schur_no_self: dict, labels, top_n: int = 10):
    """Compare dominant Schur modes with and without diagonal self-connections."""
    with_summary = dominant_mode_summary(schur_with_self, labels, top_n=top_n)
    no_summary = dominant_mode_summary(schur_no_self, labels, top_n=top_n)

    with_vec = with_summary["vector"]
    no_vec = no_summary["vector"]
    overlap = abs(np.vdot(with_vec, no_vec)) / (np.linalg.norm(with_vec) * np.linalg.norm(no_vec))

    comparison = pd.DataFrame([
        {
            "case": "with_self",
            "dominant_mode_index": with_summary["dominant_idx"] + 1,
            "eigenvalue_real": np.real(with_summary["dominant_eigenvalue"]),
            "eigenvalue_imag": np.imag(with_summary["dominant_eigenvalue"]),
            "spectral_radius": with_summary["spectral_radius"],
            "top_cell": with_summary["top"].iloc[0]["cell"],
        },
        {
            "case": "no_self",
            "dominant_mode_index": no_summary["dominant_idx"] + 1,
            "eigenvalue_real": np.real(no_summary["dominant_eigenvalue"]),
            "eigenvalue_imag": np.imag(no_summary["dominant_eigenvalue"]),
            "spectral_radius": no_summary["spectral_radius"],
            "top_cell": no_summary["top"].iloc[0]["cell"],
        },
    ])

    print("Dominant Schur Mode Comparison")
    print("=" * 80)
    print(comparison.to_string(index=False))
    print("\nChange after zeroing the diagonal")
    print(f"  Spectral radius shift: {no_summary['spectral_radius'] - with_summary['spectral_radius']:+.6f}")
    print(f"  Dominant-vector overlap: {overlap:.6f}  (1.0 = same direction)")

    print("\nTop contributors WITH self-connections:")
    print(with_summary["top"].to_string(index=False))
    print("\nTop contributors WITHOUT self-connections:")
    print(no_summary["top"].to_string(index=False))

    return comparison, with_summary["top"], no_summary["top"], overlap

def sorted_eigenmode_table(schur_result: dict, labels, sort_by: str = "magnitude", ascending: bool = False, top_n: int | None = None) -> pd.DataFrame:
    """Return eigenvalues sorted for reporting without reordering the Schur basis."""
    eigenvalues = schur_result["eigenvalues"]
    Q = schur_result["Q"]
    rows = []
    for schur_index, eigenvalue in enumerate(eigenvalues):
        vec = Q[:, schur_index]
        dominant_idx = int(np.argmax(np.abs(vec)))
        rows.append({
            "schur_index": schur_index,
            "mode_number": schur_index + 1,
            "eigenvalue_real": float(np.real(eigenvalue)),
            "eigenvalue_imag": float(np.imag(eigenvalue)),
            "magnitude": float(np.abs(eigenvalue)),
            "angle_rad": float(np.angle(eigenvalue)),
            "dominant_cell": format_cell_label(labels[dominant_idx]),
            "dominant_weight_real": float(np.real(vec[dominant_idx])),
            "dominant_weight_imag": float(np.imag(vec[dominant_idx])),
            "dominant_weight_mag": float(np.abs(vec[dominant_idx])),
        })
    table = pd.DataFrame(rows)
    sort_columns = {
        "magnitude": "magnitude",
        "real": "eigenvalue_real",
        "imag": "eigenvalue_imag",
        "angle": "angle_rad",
        "schur": "schur_index",
    }
    if sort_by not in sort_columns:
        raise ValueError(f"sort_by must be one of {sorted(sort_columns)}")
    table = table.sort_values(sort_columns[sort_by], ascending=ascending).reset_index(drop=True)
    table.insert(0, "sorted_rank", np.arange(1, len(table) + 1))
    if top_n is not None:
        return table.head(top_n)
    return table


def print_sorted_eigenmode_table(schur_result: dict, labels, sort_by: str = "magnitude", ascending: bool = False, top_n: int = 20) -> pd.DataFrame:
    """Print and return a sorted eigenmode table."""
    table = sorted_eigenmode_table(schur_result, labels, sort_by=sort_by, ascending=ascending, top_n=top_n)
    direction = "ascending" if ascending else "descending"
    print(f"Eigenmodes sorted by {sort_by} ({direction}); Schur indices are preserved for T/Q lookup.")
    print(table.to_string(index=False))
    return table


def prepare_left_action_schur(
    df_source_receiver: pd.DataFrame,
    normalization: str = "none",
    target_spectral_radius: float = 1.0,
) -> dict:
    """Prepare the corrected state matrix and its Schur/eigen decompositions.

    The input convention is rows=sources and columns=receivers.  The returned
    state matrix therefore uses ``M = input.T`` so that ``M[receiver, source]``
    acts on column state vectors.
    """
    source_matrix = df_source_receiver.to_numpy(dtype=float)
    norm = normalize_matrix(
        source_matrix,
        method=normalization,
        target_spectral_radius=target_spectral_radius,
    )
    schur = schur_decomp(
        norm["matrix"],
        labels=list(df_source_receiver.columns),
        transpose=True,
        output="complex",
        normalize_signs=False,
    )
    M = schur["work_matrix"]
    eigvals, eigvecs = linalg.eig(M)
    eigvals_left, eigvecs_left = linalg.eig(M, left=True, right=False)
    schur.update(
        {
            "normalization": norm,
            "eigvals": eigvals,
            "eigvecs": eigvecs,
            "eigvals_left": eigvals_left,
            "eigvecs_left": eigvecs_left,
        }
    )
    return schur


def schur_validation_summary(schur_result: dict, *, print_summary: bool = True) -> dict:
    """Calculate numerical checks for a complex Schur decomposition."""
    M = schur_result["work_matrix"]
    T = schur_result["T"]
    Q = schur_result["Q"]
    eigvecs = schur_result.get("eigvecs")
    eps = np.finfo(float).eps
    summary = {
        "reconstruction_relative_error": float(
            np.linalg.norm(M - Q @ T @ Q.conj().T, ord="fro")
            / max(np.linalg.norm(M, ord="fro"), eps)
        ),
        "unitarity_error": float(
            np.linalg.norm(Q.conj().T @ Q - np.eye(Q.shape[0]), ord="fro")
        ),
        "lower_triangle_norm": float(np.linalg.norm(np.tril(T, k=-1), ord="fro")),
        "condition_Q": float(np.linalg.cond(Q)),
        "condition_eigenvectors": (
            float(np.linalg.cond(eigvecs)) if eigvecs is not None else np.nan
        ),
        "spectral_radius": float(np.max(np.abs(np.diag(T)))),
        "non_normality": float(
            np.linalg.norm(M @ M.conj().T - M.conj().T @ M, ord="fro")
        ),
    }
    if print_summary:
        print("Schur decomposition complete.")
        print(f"  reconstruction relative error: {summary['reconstruction_relative_error']:.3e}")
        print(f"  unitarity error ||QᴴQ-I||_F: {summary['unitarity_error']:.3e}")
        print(f"  lower-triangle norm: {summary['lower_triangle_norm']:.3e}")
        print(f"  cond(Q): {summary['condition_Q']:.6f}")
        print(f"  spectral radius: {summary['spectral_radius']:.6f}")
        print(f"  non-normality: {summary['non_normality']:.4e}")
        if eigvecs is not None:
            ratio = summary["condition_eigenvectors"] / summary["condition_Q"]
            print(f"  cond(eigenvectors): {summary['condition_eigenvectors']:.4e}")
            print(f"  eigenvector/Q condition ratio: {ratio:.2e}x")
    return summary


def format_complex(z, tol: float = 1e-8) -> str:
    """Format a real or complex scalar compactly."""
    z = complex(z)
    if abs(z.imag) < tol:
        return f"{z.real:.4f}"
    sign = "+" if z.imag >= 0 else "-"
    return f"{z.real:.4f} {sign} {abs(z.imag):.4f}i"


def format_schur_coupling(coupling) -> str:
    """Format a Schur-basis coupling term with its magnitude."""
    return f"coupling T = {format_complex(coupling)} | |coupling| = {abs(coupling):.4f}"


def _relative_phase(w, dominant_phase: float, is_oscillatory: bool) -> str:
    if not is_oscillatory:
        return "N/A"
    difference = np.angle(w) - dominant_phase
    degrees = np.degrees(np.arctan2(np.sin(difference), np.cos(difference)))
    return "0.0° (Ref)" if abs(degrees) < 1e-2 else f"{degrees:+.1f}°"


def build_upstream_schur_modes(schur_result: dict, tol: float = 1e-5):
    """Group conjugate modes while retaining reverse (upstream-first) Schur order."""
    eigenvalues = schur_result["eigenvalues"]
    Q = schur_result["Q"]
    eigvals_left = schur_result["eigvals_left"]
    eigvecs_left = schur_result["eigvecs_left"]
    order = np.arange(len(eigenvalues) - 1, -1, -1)
    seen, modes, schur_to_step = set(), [], {}
    for i in order:
        if i in seen:
            continue
        value = eigenvalues[i]
        left_index = int(np.argmin(np.abs(eigvals_left - value)))
        members = [i]
        mode_type = "Real"
        if abs(value.imag) >= tol:
            matches = [
                j for j in order
                if j != i and j not in seen and abs(eigenvalues[j] - value.conjugate()) < tol
            ]
            if matches:
                members.append(int(matches[0]))
                mode_type = "Complex Pair"
            else:
                mode_type = "Complex"
        mode = {
            "type": mode_type,
            "eigenvalue": value,
            "schur_vector": Q[:, i],
            "left_eigenvalue": eigvals_left[left_index],
            "left_eigenvector": eigvecs_left[:, left_index],
            "schur_index": int(i),
            "member_indices": members,
        }
        modes.append(mode)
        for member in members:
            seen.add(member)
            schur_to_step[member] = len(modes)
    return modes, schur_to_step


def print_upstream_schur_modes(
    schur_result: dict,
    labels,
    *,
    coupling_tol: float = 1e-5,
    top_n: int = 5,
    max_couplings: int = 3,
):
    """Print corrected upper-triangular mode flow and cell-type loadings."""
    T = schur_result["T"]
    modes, schur_to_step = build_upstream_schur_modes(schur_result, tol=coupling_tol)
    print(f"Identified {len(modes)} unique Schur modes.")
    print("Printed order: upstream drivers first; T[row receiver, column source].")
    for step, mode in enumerate(modes, start=1):
        i = mode["schur_index"]
        value = mode["eigenvalue"]
        print("=" * 80)
        print(
            f"Cascade Step {step} [Schur index {i + 1}; "
            f"paired indexes {[m + 1 for m in mode['member_indices']]}] "
            f"| Type = {mode['type']} | λ = {format_complex(value)} | |λ| = {abs(value):.4f}"
        )
        for label, orientation, candidates in (
            (
                "Top incoming source(s)",
                f"T[receiver Schur {i + 1}, source Schur {{other}}]",
                ((T[i, j], j) for j in range(T.shape[1]) if j != i),
            ),
            (
                "Top downstream target(s)",
                f"T[receiver Schur {{other}}, source Schur {i + 1}]",
                ((T[k, i], k) for k in range(T.shape[0]) if k != i),
            ),
        ):
            ranked = sorted(
                ((abs(c), c, schur_to_step.get(index), index) for c, index in candidates
                 if abs(c) > coupling_tol and schur_to_step.get(index) != step),
                key=lambda item: item[0],
                reverse=True,
            )
            unique, used_steps = [], set()
            for _, coupling, other_step, other_index in ranked:
                if other_step is not None and other_step not in used_steps:
                    unique.append((coupling, other_step, other_index))
                    used_steps.add(other_step)
                if len(unique) == max_couplings:
                    break
            text = " | ".join(
                f"Cascade Step {other_step} / Schur {other_index + 1} "
                f"({orientation.format(other=other_index + 1)}; {format_schur_coupling(coupling)})"
                for coupling, other_step, other_index in unique
            )
            print(f"{label}: {text or 'None'}")

        for heading, vector in (
            ("Schur state vector", mode["schur_vector"]),
            ("Left eigenvector action/readout", mode["left_eigenvector"]),
        ):
            print(f"Principal cell types ({heading}):")
            dominant_phase = np.angle(vector[int(np.argmax(np.abs(vector)))])
            oscillatory = mode["type"] != "Real"
            for index in np.argsort(np.abs(vector))[::-1][:top_n]:
                weight = vector[index]
                print(
                    f"  - {format_cell_label(labels[index]):<45} | weight = {format_complex(weight):>18} "
                    f"| energy = {abs(weight) ** 2 * 100:5.1f}% "
                    f"| phase = {_relative_phase(weight, dominant_phase, oscillatory)}"
                )
    return modes, schur_to_step


def upstream_mode_summary(schur_result: dict, labels, coupling_tol: float = 1e-5) -> pd.DataFrame:
    """Return one compact row per unique upstream-first Schur mode."""
    T = schur_result["T"]
    modes, _ = build_upstream_schur_modes(schur_result, tol=coupling_tol)
    rows = []
    for step, mode in enumerate(modes, start=1):
        i = mode["schur_index"]
        rows.append(
            {
                "Cascade step": step,
                "Schur index": i + 1,
                "Type": mode["type"],
                "lambda": format_complex(mode["eigenvalue"]),
                "|lambda|": abs(mode["eigenvalue"]),
                "Top left-vector/action cell": format_cell_label(labels[int(np.argmax(np.abs(mode["left_eigenvector"])))]),
                "Top Schur-state cell": format_cell_label(labels[int(np.argmax(np.abs(mode["schur_vector"])))]),
                "Incoming count": int(np.count_nonzero(np.abs(T[i, :]) > coupling_tol) - (abs(T[i, i]) > coupling_tol)),
                "Outgoing count": int(np.count_nonzero(np.abs(T[:, i]) > coupling_tol) - (abs(T[i, i]) > coupling_tol)),
            }
        )
    return pd.DataFrame(rows)


def plot_true_dominant_eigenmode(schur_result: dict, labels, top_n: int = 20) -> dict:
    """Plot the true dominant eigenvector alongside the eigenvalue spectrum."""
    eigvals, eigvecs = schur_result["eigvals"], schur_result["eigvecs"]
    index = int(np.argmax(np.abs(eigvals)))
    value, vector = eigvals[index], eigvecs[:, index]
    eigenvalues = schur_result["eigenvalues"]
    nearest_schur = int(np.argmin(np.abs(eigenvalues - value)))
    top = pd.DataFrame(
        {"Cell type": format_cell_labels(labels), "Magnitude": np.abs(vector)}
    ).sort_values("Magnitude", ascending=False).head(top_n)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    ax1.scatter(eigenvalues.real, eigenvalues.imag, c="blue", alpha=0.6)
    ax1.scatter(value.real, value.imag, c="red", s=100, edgecolors="black")
    ax1.add_artist(plt.Circle((0, 0), 1, color="gray", fill=False, linestyle="--"))
    ax1.set(xlabel="Real Part", ylabel="Imaginary Part", title="Eigenvalue Spectrum")
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax2.barh(top["Cell type"], top["Magnitude"])
    ax2.set(xlabel="Magnitude", title=f"Top {top_n} True Dominant-Eigenvector Contributions")
    ax2.invert_yaxis()
    plt.tight_layout()
    plt.show()
    print(f"Dominant eigenvalue: {format_complex(value)}")
    print(f"Spectral radius ρ(M): {abs(value):.6f}")
    print(f"Nearest Schur diagonal index: {nearest_schur + 1}")
    return {"index": index, "eigenvalue": value, "eigenvector": vector, "top": top}


def analyze_discrete_transient_dynamics(M, k_max: int = 50, grid_size: int = 50) -> dict:
    """Compute and plot finite-horizon gain and a discrete Kreiss-style proxy."""
    M = np.asarray(M, dtype=complex)
    eigvals = np.linalg.eigvals(M)
    rho = float(np.max(np.abs(eigvals)))
    powers = np.arange(k_max + 1)
    gains = np.empty(k_max + 1)
    matrix_power = np.eye(M.shape[0], dtype=complex)
    for k in powers:
        gains[k] = np.linalg.norm(matrix_power, 2)
        matrix_power = M @ matrix_power
    peak_k = int(np.argmax(gains))
    rmax = max(1.25, rho * 1.25)
    axis = np.linspace(-rmax, rmax, grid_size)
    identity = np.eye(M.shape[0], dtype=complex)
    kreiss_bound, critical_z = 0.0, None
    for real in axis:
        for imag in axis:
            z = real + 1j * imag
            if abs(z) <= 1:
                continue
            sigma_min = np.min(np.linalg.svd(z * identity - M, compute_uv=False))
            if sigma_min <= np.finfo(float).eps:
                continue
            proxy = (abs(z) - 1) / sigma_min
            if proxy > kreiss_bound:
                kreiss_bound, critical_z = float(proxy), z
    print(f"Spectral radius ρ(M): {rho:.6f} ({'stable' if rho < 1 else 'not stable'})")
    print(f"Peak ||M^k||₂: {gains[peak_k]:.6f} at k={peak_k}")
    print(f"Discrete Kreiss-style grid proxy: {kreiss_bound:.6f}")
    if critical_z is not None:
        print(f"Critical grid point: {format_complex(critical_z)}")
    plt.figure(figsize=(10, 6))
    plt.plot(powers, gains, marker="o", markersize=3)
    plt.xlabel("Discrete time step k")
    plt.ylabel(r"$||M^k||_2$")
    plt.title("Discrete-Time Transient Amplification")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    return {
        "powers": powers,
        "gains": gains,
        "peak_gain": float(gains[peak_k]),
        "peak_k": peak_k,
        "spectral_radius": rho,
        "kreiss_lower_bound": kreiss_bound,
        "critical_z": critical_z,
    }
