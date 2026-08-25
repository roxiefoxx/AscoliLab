"""Shared core utilities for the Schur decomposition analysis notebooks.

This module holds the cross-cutting pieces that should stay identical across
the eigenmode, propagation, inhibitory, greedy-tree, and linear-systems
analyses: loading the Mij data, orienting it as receiver-by-source, label
formatting, region parsing, spectral diagnostics, normalization, and
with-self/no-self matrix variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


EXCITATORY_LABEL_MARKERS = (
    "pyramidal",
    "principal",
    "granule",
    "semilunar",
    "stellate",
    "back projection",
    "mossy",
    "total molecular layer",
)

INHIBITORY_LABEL_MARKERS = (
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


@dataclass
class MijData:
    """Prepared signed connectivity data in state-update convention.

    ``M`` is oriented as ``M[receiver, source]``. ``df_source_receiver`` keeps
    the original CSV orientation, with source rows and receiver columns.
    """

    M: np.ndarray
    labels: list[str]
    ei: pd.Series
    df_source_receiver: pd.DataFrame
    netlist: pd.DataFrame | None
    matrix_path: Path
    netlist_path: Path | None

    @property
    def excitatory(self) -> list[str]:
        return self.ei[self.ei == "e"].index.tolist()

    @property
    def inhibitory(self) -> list[str]:
        return self.ei[self.ei == "i"].index.tolist()


def infer_cell_ei(label) -> str:
    """Infer E/I type from the cell-type name for compact display labels."""
    text = str(label).strip()
    lowered = text.lower()
    if lowered.endswith("(e)"):
        return "E"
    if lowered.endswith("(i)"):
        return "I"
    if any(marker in lowered for marker in INHIBITORY_LABEL_MARKERS):
        return "I"
    if any(marker in lowered for marker in EXCITATORY_LABEL_MARKERS):
        return "E"
    return "I"


def format_cell_label(label) -> str:
    """Append an E/I suffix to a cell-type label, without duplicating it."""
    text = str(label)
    stripped = text.strip()
    if stripped.lower().endswith("(e)") or stripped.lower().endswith("(i)"):
        return text
    return f"{text} ({infer_cell_ei(text)})"


def format_cell_labels(labels: Sequence[object]) -> list[str]:
    """Return all labels with E/I suffixes for printed tables and plots."""
    return [format_cell_label(label) for label in labels]


def region_from_label(label: str) -> str:
    """Extract the anatomical prefix, folding CA3c into CA3."""
    prefix = str(label).split()[0]
    return "CA3" if prefix == "CA3c" else prefix


def load_mij_matrix(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate a square, consistently labeled Mij CSV."""
    frame = pd.read_csv(csv_path, index_col=0)
    if frame.shape[0] != frame.shape[1]:
        raise ValueError(f"Mij must be square; got {frame.shape}.")
    if list(frame.index) != list(frame.columns):
        raise ValueError("Mij row and column labels must match in the same order.")
    if frame.index.has_duplicates:
        raise ValueError("Cell-type labels must be unique.")
    return frame.astype(float)


def load_source_receiver_matrix(csv_path: str | Path) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Load source-row/receiver-column CSV and return state matrix convention."""
    frame = load_mij_matrix(csv_path)
    return frame.to_numpy(dtype=float).T.copy(), list(frame.columns), frame


def infer_ei_from_signed_rows(df_source_receiver: pd.DataFrame) -> pd.Series:
    """Infer source E/I labels from the sign of outgoing source rows."""
    out: dict[str, str] = {}
    for label, row in df_source_receiver.iterrows():
        values = row.to_numpy(dtype=float)
        pos = np.sum(values > 0)
        neg = np.sum(values < 0)
        out[str(label)] = "i" if neg > pos else "e"
    return pd.Series(out)


def load_mij_data(
    matrix_path: str | Path = "matrices/mij_matrix.csv",
    netlist_path: str | Path | None = "matrices/mij_netlist.csv",
    infer_missing_ei: bool = True,
) -> MijData:
    """Load Mij matrix and optional netlist metadata.

    The returned ``M`` is transposed from the raw source-row/receiver-column
    table into ``M[receiver, source]``.
    """
    matrix_path = Path(matrix_path)
    df = load_mij_matrix(matrix_path)
    labels = list(df.columns)

    netlist = None
    netlist_path_obj = Path(netlist_path) if netlist_path is not None else None
    ei = pd.Series(index=labels, dtype="object")
    if netlist_path_obj is not None and netlist_path_obj.exists():
        netlist = pd.read_csv(netlist_path_obj)
        pre = netlist[["pre_neuron", "pre_ei"]].dropna().drop_duplicates()
        post = netlist[["post_neuron", "post_ei"]].dropna().drop_duplicates()
        mapping: dict[str, str] = {}
        for row in pre.itertuples(index=False):
            mapping[str(row.pre_neuron)] = str(row.pre_ei).lower()
        for row in post.itertuples(index=False):
            value = str(row.post_ei).lower()
            old = mapping.get(str(row.post_neuron))
            if old is not None and old != value:
                raise ValueError(f"Conflicting E/I metadata for {row.post_neuron}: {old} vs {value}")
            mapping[str(row.post_neuron)] = value
        ei = pd.Series({label: mapping.get(label, np.nan) for label in labels}, dtype="object")

    if infer_missing_ei and ei.isna().any():
        inferred = infer_ei_from_signed_rows(df)
        ei = ei.combine_first(inferred)

    if ei.isna().any():
        missing = ei[ei.isna()].index.tolist()
        raise ValueError(f"Missing E/I metadata for {missing[:10]}")

    return MijData(
        M=df.to_numpy(dtype=float).T,
        labels=labels,
        ei=ei.astype(str).str.lower(),
        df_source_receiver=df,
        netlist=netlist,
        matrix_path=matrix_path,
        netlist_path=netlist_path_obj,
    )


def spectral_radius(M: np.ndarray) -> float:
    """Return max(abs(eigvals(M)))."""
    eigvals = np.linalg.eigvals(M)
    return float(np.max(np.abs(eigvals)))


def spectral_abscissa(M: np.ndarray) -> float:
    """Return max(real(eigvals(M)))."""
    eigvals = np.linalg.eigvals(M)
    return float(np.max(np.real(eigvals)))


def spectral_summary(M: np.ndarray) -> dict[str, float | complex]:
    """Return compact eigenvalue stability diagnostics."""
    eigvals = np.linalg.eigvals(M)
    dominant_idx = int(np.argmax(np.real(eigvals)))
    return {
        "spectral_radius": float(np.max(np.abs(eigvals))),
        "spectral_abscissa": float(np.max(np.real(eigvals))),
        "dominant_eigenvalue": complex(eigvals[dominant_idx]),
        "dominant_real": float(np.real(eigvals[dominant_idx])),
        "dominant_imag": float(np.imag(eigvals[dominant_idx])),
    }


def normalize_state_matrix(
    M: np.ndarray,
    method: str = "spectral_radius",
    target: float = 0.95,
) -> tuple[np.ndarray, dict[str, float | str | complex]]:
    """Normalize a receiver-by-source state matrix and return diagnostics."""
    M = np.asarray(M, dtype=float)
    method = method.lower().strip()
    if method == "none":
        return M.copy(), {"method": "none", "scale": 1.0, **spectral_summary(M)}
    if method in {"spectral_radius", "rho"}:
        rho = spectral_radius(M)
        if rho <= np.finfo(float).eps:
            raise ValueError("Cannot spectral-radius normalize a zero matrix.")
        scale = target / rho
        out = M * scale
        return out, {"method": "spectral_radius", "target": target, "scale": scale, **spectral_summary(out)}
    if method in {"source_l1", "column_l1", "col_l1"}:
        scale = np.sum(np.abs(M), axis=0)
        scale[scale == 0] = 1.0
        out = M / scale[np.newaxis, :]
        return out, {"method": "source_l1", "scale": np.nan, **spectral_summary(out)}
    if method == "row_l1":
        scale = np.sum(np.abs(M), axis=1)
        scale[scale == 0] = 1.0
        out = M / scale[:, np.newaxis]
        return out, {"method": "row_l1", "scale": np.nan, **spectral_summary(out)}
    raise ValueError("method must be 'none', 'spectral_radius', 'source_l1', 'column_l1', or 'row_l1'.")


def matrix_variants(M: np.ndarray) -> dict[str, np.ndarray]:
    """Return with-self and no-self versions of a state matrix."""
    with_self = np.asarray(M, dtype=float).copy()
    no_self = with_self.copy()
    np.fill_diagonal(no_self, 0.0)
    return {"with_self": with_self, "no_self": no_self}


def normalized_matrix_variants(
    M: np.ndarray,
    method: str = "spectral_radius",
    target: float = 0.85,
) -> dict[str, dict[str, object]]:
    """Normalize with-self and no-self variants independently."""
    out: dict[str, dict[str, object]] = {}
    for variant, raw in matrix_variants(M).items():
        normalized, norm = normalize_state_matrix(raw, method=method, target=target)
        norm["variant"] = variant
        out[variant] = {"M": normalized, "normalization": norm}
    return out
