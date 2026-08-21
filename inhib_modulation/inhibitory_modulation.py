"""Utilities for inhibitory modulation analyses.

The functions in this module are intentionally notebook-friendly: each one
returns plain pandas or numpy objects that are easy to inspect, plot, or save.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


INHIBITORY_NAME_HINTS = (
    "basket",
    "axo axonic",
    "bistratified",
    "ivy",
    "neurogliaform",
    "o lm",
    "olm",
    "hipp",
    "hicap",
    "hiprom",
    "mopp",
    "mossy",
    "trilaminar",
    "quadrilaminar",
    "interneuron",
)


TRIAD_CENSUS_TO_REES_SUPERPATTERN = {
    "102": "-A",
    "012": "-B",
    "003": "-C",
    "021U": "A",
    "021C": "B",
    "021D": "C",
    "111D": "D",
    "030T": "E",
    "111U": "F",
    "030C": "G",
    "120D": "H",
    "201": "I",
    "120C": "J",
    "120U": "K",
    "210": "L",
    "300": "M",
}


REES_SUPERPATTERN_NAMES = {
    "-A": "disconnected mutual dyad",
    "-B": "disconnected single edge",
    "-C": "empty disconnected triad",
    "A": "simple regulation",
    "B": "three-node chain",
    "C": "single input module",
    "D": "single downlink to a mutual dyad",
    "E": "feedforward loop",
    "F": "single uplinked mutual dyad",
    "G": "feedback loop",
    "H": "double downlink to a mutual dyad",
    "I": "chain of two mutual dyads",
    "J": "single point feedforward and feedback loops",
    "K": "double uplinked mutual dyad",
    "L": "feedback with two mutual dyads",
    "M": "fully connected triad",
}


@dataclass(frozen=True)
class ConnectivityData:
    """Loaded connectivity matrix and neuron class metadata."""

    matrix: pd.DataFrame
    classes: pd.Series
    source: Path


@dataclass(frozen=True)
class BlockMatrices:
    """E/I block decomposition of a connectivity matrix."""

    ee: pd.DataFrame
    ei: pd.DataFrame
    ie: pd.DataFrame
    ii: pd.DataFrame
    excitatory: list[str]
    inhibitory: list[str]


@dataclass(frozen=True)
class TriadSchurDecomposition:
    """Schur-complement reduction of the aggregate two-edge triad matrix."""

    aggregate_matrix: pd.DataFrame
    blocks: BlockMatrices
    feedback: pd.DataFrame
    effective_excitation: pd.DataFrame
    aggregate_stability: pd.Series
    effective_stability: pd.Series


def load_connectivity(
    path: str | Path,
    *,
    weight_col: str | None = None,
    class_map: Mapping[str, str] | None = None,
    matrix_orientation: str = "pre_by_post",
    metadata_netlist_path: str | Path | None = None,
) -> ConnectivityData:
    """Load a square connectivity matrix or a pre/post edge list.

    Square matrix files should have neuron names as the index column and as
    headers. Set ``matrix_orientation`` to ``"pre_by_post"`` when rows are
    presynaptic sources and columns are postsynaptic targets, or
    ``"post_by_pre"`` when rows are postsynaptic targets and columns are
    presynaptic sources. All returned matrices use rows=post and columns=pre.
    Edge lists should include pre/post neuron columns and a weight column such
    as ``m_ij``, ``weight``, or ``w_ij``.
    """

    source = Path(path)
    metadata_classes = _classes_from_netlist(metadata_netlist_path) if metadata_netlist_path else {}
    merged_class_map = {**metadata_classes, **(class_map or {})}
    raw = pd.read_csv(source)
    lower_cols = {col.lower(): col for col in raw.columns}

    if {"pre_neuron", "post_neuron"}.issubset(lower_cols):
        matrix, classes = _load_netlist(raw, source, weight_col, merged_class_map)
    else:
        matrix = pd.read_csv(source, index_col=0)
        matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        matrix.index = matrix.index.astype(str)
        matrix.columns = matrix.columns.astype(str)
        matrix = matrix.reindex(index=matrix.index, columns=matrix.index, fill_value=0.0)
        if matrix_orientation == "pre_by_post":
            matrix = matrix.T
        elif matrix_orientation != "post_by_pre":
            raise ValueError("matrix_orientation must be 'pre_by_post' or 'post_by_pre'.")
        classes = infer_neuron_classes(matrix, class_map=merged_class_map)

    return ConnectivityData(matrix=matrix, classes=classes, source=source)


def _classes_from_netlist(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    netlist_path = Path(path)
    if not netlist_path.exists():
        return {}
    raw = pd.read_csv(netlist_path)
    lower_cols = {col.lower(): col for col in raw.columns}
    mapping: dict[str, str] = {}
    for neuron_col, ei_col in (("pre_neuron", "pre_ei"), ("post_neuron", "post_ei")):
        if neuron_col not in lower_cols or ei_col not in lower_cols:
            continue
        pairs = raw[[lower_cols[neuron_col], lower_cols[ei_col]]].dropna().drop_duplicates()
        for neuron, label in pairs.itertuples(index=False):
            normalized = _normalize_class(str(label))
            old = mapping.get(str(neuron))
            if old is not None and old != normalized:
                raise ValueError(f"Conflicting E/I metadata for {neuron}: {old} vs {normalized}")
            mapping[str(neuron)] = normalized
    return mapping


def _load_netlist(
    raw: pd.DataFrame,
    source: Path,
    weight_col: str | None,
    class_map: Mapping[str, str] | None,
) -> tuple[pd.DataFrame, pd.Series]:
    col_lookup = {col.lower(): col for col in raw.columns}
    pre_col = col_lookup["pre_neuron"]
    post_col = col_lookup["post_neuron"]
    chosen_weight = weight_col or _first_existing(raw, ("m_ij", "weight", "w_ij", "value"))
    if chosen_weight is None:
        raise ValueError(f"No usable weight column found in {source}.")

    names = pd.Index(sorted(set(raw[pre_col].astype(str)) | set(raw[post_col].astype(str))))
    matrix = (
        raw.assign(
            _pre=raw[pre_col].astype(str),
            _post=raw[post_col].astype(str),
            _weight=pd.to_numeric(raw[chosen_weight], errors="coerce").fillna(0.0),
        )
        .pivot_table(index="_post", columns="_pre", values="_weight", aggfunc="sum", fill_value=0.0)
        .reindex(index=names, columns=names, fill_value=0.0)
    )

    classes = pd.Series(index=names, dtype="object")
    if "pre_ei" in col_lookup:
        pre_classes = raw.drop_duplicates(pre_col).set_index(pre_col)[col_lookup["pre_ei"]]
        classes.loc[pre_classes.index.astype(str)] = pre_classes.astype(str).str.lower().str[0].values
    if "post_ei" in col_lookup:
        post_classes = raw.drop_duplicates(post_col).set_index(post_col)[col_lookup["post_ei"]]
        classes.loc[post_classes.index.astype(str)] = classes.loc[post_classes.index.astype(str)].fillna(
            post_classes.astype(str).str.lower().str[0]
        )

    inferred = infer_neuron_classes(matrix, class_map=class_map)
    classes = classes.fillna(inferred).replace({"excitatory": "e", "inhibitory": "i"})
    return matrix, classes


def _first_existing(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lookup = {col.lower(): col for col in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def infer_neuron_classes(
    matrix: pd.DataFrame,
    *,
    class_map: Mapping[str, str] | None = None,
) -> pd.Series:
    """Infer E/I classes from a user map, neuron-name hints, and column sign."""

    classes = pd.Series("e", index=matrix.index.astype(str), dtype="object")
    if class_map:
        for name, label in class_map.items():
            if name in classes.index:
                classes.loc[name] = _normalize_class(label)

    unmapped = classes.index.difference(class_map.keys() if class_map else [])
    for name in unmapped:
        lowered = name.lower()
        if any(hint in lowered for hint in INHIBITORY_NAME_HINTS):
            classes.loc[name] = "i"

    for name in unmapped:
        col = pd.to_numeric(matrix[name], errors="coerce").fillna(0.0)
        nonzero = col[col != 0]
        if len(nonzero) and (nonzero < 0).mean() >= 0.8:
            classes.loc[name] = "i"

    return classes


def _normalize_class(label: str) -> str:
    first = str(label).strip().lower()[0]
    if first not in {"e", "i"}:
        raise ValueError(f"Class labels must start with 'e' or 'i', got {label!r}.")
    return first


def make_ei_blocks(matrix: pd.DataFrame, classes: pd.Series) -> BlockMatrices:
    """Partition a connectivity matrix into EE, EI, IE, and II blocks.

    Rows are postsynaptic targets and columns are presynaptic sources.
    """

    aligned = classes.reindex(matrix.index).map(_normalize_class)
    excitatory = aligned[aligned == "e"].index.tolist()
    inhibitory = aligned[aligned == "i"].index.tolist()
    return BlockMatrices(
        ee=matrix.loc[excitatory, excitatory],
        ei=matrix.loc[excitatory, inhibitory],
        ie=matrix.loc[inhibitory, excitatory],
        ii=matrix.loc[inhibitory, inhibitory],
        excitatory=excitatory,
        inhibitory=inhibitory,
    )


def summarize_matrix(matrix: pd.DataFrame, name: str = "matrix") -> pd.Series:
    """Return compact descriptive statistics for a matrix."""

    values = matrix.to_numpy(dtype=float).ravel()
    nonzero = values[values != 0]
    abs_nonzero = np.abs(nonzero)
    return pd.Series(
        {
            "name": name,
            "rows": matrix.shape[0],
            "cols": matrix.shape[1],
            "entries": values.size,
            "nonzero_count": int(nonzero.size),
            "density": float(nonzero.size / values.size) if values.size else np.nan,
            "median": float(np.median(nonzero)) if nonzero.size else 0.0,
            "mean": float(np.mean(nonzero)) if nonzero.size else 0.0,
            "abs_median": float(np.median(abs_nonzero)) if nonzero.size else 0.0,
            "abs_mean": float(np.mean(abs_nonzero)) if nonzero.size else 0.0,
            "min": float(np.min(nonzero)) if nonzero.size else 0.0,
            "max": float(np.max(nonzero)) if nonzero.size else 0.0,
        }
    )


def summarize_blocks(blocks: BlockMatrices) -> pd.DataFrame:
    """Summarize all four E/I blocks."""

    return pd.DataFrame(
        [
            summarize_matrix(blocks.ee, "EE"),
            summarize_matrix(blocks.ei, "EI"),
            summarize_matrix(blocks.ie, "IE"),
            summarize_matrix(blocks.ii, "II"),
        ]
    ).set_index("name")


def eigenspectrum(matrix: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Compute eigenvalues and stability-oriented summary columns."""

    arr = np.asarray(matrix)
    if arr.size == 0:
        return pd.DataFrame(columns=["real", "imag", "abs"])
    vals = np.linalg.eigvals(arr)
    return pd.DataFrame({"real": vals.real, "imag": vals.imag, "abs": np.abs(vals)})


def stability_summary(matrix: pd.DataFrame | np.ndarray, name: str = "matrix") -> pd.Series:
    """Summarize spectrum using max real part and spectral radius."""

    spectrum = eigenspectrum(matrix)
    if spectrum.empty:
        return pd.Series({"name": name, "max_real": np.nan, "spectral_radius": np.nan, "stable_continuous": None, "stable_discrete": None})
    max_real = float(spectrum["real"].max())
    radius = float(spectrum["abs"].max())
    return pd.Series(
        {
            "name": name,
            "max_real": max_real,
            "spectral_radius": radius,
            "stable_continuous": bool(max_real < 0),
            "stable_discrete": bool(radius < 1),
        }
    )


def schur_complement(
    blocks: BlockMatrices,
    *,
    regularization: float = 1e-6,
) -> pd.DataFrame:
    """Return the E-subspace Schur complement, EE - EI * inv(II) * IE."""

    feedback = effective_inhibitory_feedback(blocks, regularization=regularization, form="static")
    return blocks.ee - feedback


def effective_inhibitory_feedback(
    blocks: BlockMatrices,
    *,
    regularization: float = 1e-6,
    form: str = "static",
    gain: float = 1.0,
) -> pd.DataFrame:
    """Compute an effective inhibitory feedback operator on the E subspace.

    ``form="static"`` computes EI * inv(II + lambda I) * IE, matching the
    Schur complement term. ``form="resolvent"`` computes
    EI * inv(I - gain * II) * IE, matching the Neumann-series expansion.
    """

    if blocks.ii.empty:
        return pd.DataFrame(0.0, index=blocks.excitatory, columns=blocks.excitatory)

    ei = blocks.ei.to_numpy(dtype=float)
    ie = blocks.ie.to_numpy(dtype=float)
    ii = blocks.ii.to_numpy(dtype=float)
    eye = np.eye(ii.shape[0])

    if form == "static":
        middle = np.linalg.pinv(ii + regularization * eye)
    elif form == "resolvent":
        middle = np.linalg.pinv(eye - gain * ii + regularization * eye)
    else:
        raise ValueError("form must be 'static' or 'resolvent'.")

    result = ei @ middle @ ie
    return pd.DataFrame(result, index=blocks.excitatory, columns=blocks.excitatory)


def ablation_analysis(
    blocks: BlockMatrices,
    *,
    block_names: Iterable[str] = ("EE", "EI", "IE", "II"),
) -> pd.DataFrame:
    """Zero each selected block and compare eigenvalue stability summaries."""

    base = assemble_blocks(blocks)
    rows = [stability_summary(base, "baseline")]
    for block_name in block_names:
        ablated = assemble_blocks(scale_block(blocks, block_name, 0.0))
        rows.append(stability_summary(ablated, f"ablate_{block_name}"))
    return pd.DataFrame(rows).set_index("name")


def perturbation_analysis(
    blocks: BlockMatrices,
    *,
    block_name: str,
    scales: Iterable[float] = np.linspace(0.0, 2.0, 21),
) -> pd.DataFrame:
    """Scale one block over a grid and summarize eigenspectrum stability."""

    rows = []
    for scale in scales:
        perturbed = assemble_blocks(scale_block(blocks, block_name, float(scale)))
        row = stability_summary(perturbed, f"{block_name}_x_{scale:.3g}")
        row["block"] = block_name
        row["scale"] = float(scale)
        rows.append(row)
    return pd.DataFrame(rows).set_index("name")


def scale_block(blocks: BlockMatrices, block_name: str, scale: float) -> BlockMatrices:
    """Return a copy of blocks with one block scaled."""

    key = block_name.lower()
    kwargs = {
        "ee": blocks.ee.copy(),
        "ei": blocks.ei.copy(),
        "ie": blocks.ie.copy(),
        "ii": blocks.ii.copy(),
        "excitatory": blocks.excitatory,
        "inhibitory": blocks.inhibitory,
    }
    if key not in {"ee", "ei", "ie", "ii"}:
        raise ValueError("block_name must be one of EE, EI, IE, or II.")
    kwargs[key] = kwargs[key] * scale
    return BlockMatrices(**kwargs)


def assemble_blocks(blocks: BlockMatrices) -> pd.DataFrame:
    """Reassemble blocks into one matrix ordered as E followed by I."""

    top = pd.concat([blocks.ee, blocks.ei], axis=1)
    bottom = pd.concat([blocks.ie, blocks.ii], axis=1)
    return pd.concat([top, bottom], axis=0)


def chain_contributions(
    blocks: BlockMatrices,
    *,
    max_depth: int = 10,
    gain: float = 1.0,
    norm: str = "fro",
) -> pd.DataFrame:
    """Measure E-I-(I repeated k)-E chain contribution sizes."""

    ei = blocks.ei.to_numpy(dtype=float)
    ie = blocks.ie.to_numpy(dtype=float)
    ii = blocks.ii.to_numpy(dtype=float)
    rows = []
    total_energy = 0.0
    running_energy = 0.0
    terms = []

    for depth in range(1, max_depth + 1):
        if depth == 1:
            term = gain * (ei @ ie)
        else:
            term = (gain ** depth) * (ei @ np.linalg.matrix_power(ii, depth - 1) @ ie)
        contribution = float(np.linalg.norm(term, ord=norm))
        terms.append((depth, contribution))
        total_energy += contribution**2

    for depth, contribution in terms:
        running_energy += contribution**2
        rows.append(
            {
                "depth": depth,
                "chain": "E-I-" + "I-" * (depth - 1) + "E",
                "contribution_norm": contribution,
                "energy": contribution**2,
                "cumulative_energy_fraction": running_energy / total_energy if total_energy else np.nan,
                "decay_ratio_vs_previous": np.nan,
            }
        )

    result = pd.DataFrame(rows)
    result["decay_ratio_vs_previous"] = result["contribution_norm"] / result["contribution_norm"].shift(1)
    return result


def select_chain_depth(
    contributions: pd.DataFrame,
    *,
    decay_threshold: float = 0.05,
    cumulative_energy_threshold: float = 0.95,
    null_threshold_depth: int | None = None,
) -> pd.Series:
    """Combine model-selection rules into a recommended chain depth."""

    decay_candidates = contributions.loc[
        contributions["decay_ratio_vs_previous"].fillna(np.inf) < decay_threshold, "depth"
    ]
    energy_candidates = contributions.loc[
        contributions["cumulative_energy_fraction"] >= cumulative_energy_threshold, "depth"
    ]
    depths = [
        int(decay_candidates.iloc[0]) if len(decay_candidates) else int(contributions["depth"].max()),
        int(energy_candidates.iloc[0]) if len(energy_candidates) else int(contributions["depth"].max()),
    ]
    if null_threshold_depth is not None:
        depths.append(int(null_threshold_depth))
    return pd.Series(
        {
            "decay_rule_depth": depths[0],
            "cumulative_energy_rule_depth": depths[1],
            "null_model_rule_depth": null_threshold_depth,
            "recommended_depth": max(depths),
        }
    )


def null_model_chain_significance(
    blocks: BlockMatrices,
    *,
    max_depth: int = 10,
    n_null: int = 250,
    random_state: int = 0,
) -> pd.DataFrame:
    """Compare chain contributions with degree-blind shuffled null weights."""

    rng = np.random.default_rng(random_state)
    observed = chain_contributions(blocks, max_depth=max_depth)
    null_values = np.zeros((n_null, max_depth), dtype=float)
    ei_values = blocks.ei.to_numpy(dtype=float).ravel()
    ie_values = blocks.ie.to_numpy(dtype=float).ravel()
    ii_values = blocks.ii.to_numpy(dtype=float).ravel()

    for draw in range(n_null):
        shuffled = BlockMatrices(
            ee=blocks.ee,
            ei=pd.DataFrame(rng.permutation(ei_values).reshape(blocks.ei.shape), index=blocks.ei.index, columns=blocks.ei.columns),
            ie=pd.DataFrame(rng.permutation(ie_values).reshape(blocks.ie.shape), index=blocks.ie.index, columns=blocks.ie.columns),
            ii=pd.DataFrame(rng.permutation(ii_values).reshape(blocks.ii.shape), index=blocks.ii.index, columns=blocks.ii.columns),
            excitatory=blocks.excitatory,
            inhibitory=blocks.inhibitory,
        )
        null_values[draw, :] = chain_contributions(shuffled, max_depth=max_depth)["contribution_norm"].to_numpy()

    observed = observed.copy()
    observed["null_mean"] = null_values.mean(axis=0)
    observed["null_p95"] = np.quantile(null_values, 0.95, axis=0)
    observed["significant_vs_null"] = observed["contribution_norm"] > observed["null_p95"]
    return observed


def neumann_series_feedback(
    blocks: BlockMatrices,
    *,
    max_order: int = 10,
    gain: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Approximate EI * inv(I - gain * II) * IE with a Neumann series."""

    ei = blocks.ei.to_numpy(dtype=float)
    ie = blocks.ie.to_numpy(dtype=float)
    ii = blocks.ii.to_numpy(dtype=float)
    accum = np.zeros((len(blocks.excitatory), len(blocks.excitatory)), dtype=float)
    rows = []

    for order in range(max_order + 1):
        term = ei @ np.linalg.matrix_power(gain * ii, order) @ ie
        accum += term
        rows.append(
            {
                "order": order,
                "term": f"EI*(gain*II)^{order}*IE",
                "term_norm": float(np.linalg.norm(term, ord="fro")),
                "partial_sum_norm": float(np.linalg.norm(accum, ord="fro")),
            }
        )

    feedback = pd.DataFrame(accum, index=blocks.excitatory, columns=blocks.excitatory)
    return feedback, pd.DataFrame(rows)


def spectral_radius(matrix: pd.DataFrame | np.ndarray) -> float:
    """Return the spectral radius of a square matrix."""

    arr = np.asarray(matrix)
    if arr.size == 0:
        return np.nan
    return float(np.max(np.abs(np.linalg.eigvals(arr))))


def spectral_summary(matrix: pd.DataFrame | np.ndarray) -> dict[str, float | complex]:
    """Return compact eigenvalue stability diagnostics."""

    arr = np.asarray(matrix)
    if arr.size == 0:
        return {
            "spectral_radius": np.nan,
            "spectral_abscissa": np.nan,
            "dominant_eigenvalue": np.nan,
            "dominant_real": np.nan,
            "dominant_imag": np.nan,
        }
    eigvals = np.linalg.eigvals(arr)
    dominant_idx = int(np.argmax(np.real(eigvals)))
    dominant = complex(eigvals[dominant_idx])
    return {
        "spectral_radius": float(np.max(np.abs(eigvals))),
        "spectral_abscissa": float(np.max(np.real(eigvals))),
        "dominant_eigenvalue": dominant,
        "dominant_real": float(np.real(dominant)),
        "dominant_imag": float(np.imag(dominant)),
    }


def normalize_connectivity(
    matrix: pd.DataFrame,
    *,
    method: str = "spectral_radius",
    target: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, float | str | complex]]:
    """Normalize a connectivity matrix and return normalization metadata."""

    method = method.lower().strip()
    arr = matrix.to_numpy(dtype=float)
    if method == "none":
        return matrix.copy(), {"method": "none", **spectral_summary(arr)}
    if method in {"spectral_radius", "rho"}:
        rho = spectral_summary(arr)["spectral_radius"]
        if not np.isfinite(rho) or float(rho) <= np.finfo(float).eps:
            raise ValueError("Cannot spectral-radius normalize a zero matrix.")
        scale = target / float(rho)
        normalized = matrix * scale
        return normalized, {"method": "spectral_radius", "target": target, "scale": scale, **spectral_summary(normalized)}
    if method in {"source_l1", "column_l1", "col_l1"}:
        scale = np.sum(np.abs(arr), axis=0)
        scale[scale == 0] = 1.0
        normalized = pd.DataFrame(arr / scale[np.newaxis, :], index=matrix.index, columns=matrix.columns)
        return normalized, {"method": "source_l1", **spectral_summary(normalized)}
    raise ValueError("method must be 'none', 'spectral_radius', or 'source_l1'.")


def matrix_variants(matrix: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return with-self and no-self variants of a matrix."""

    with_self = matrix.copy()
    no_self_array = matrix.to_numpy(dtype=float, copy=True)
    diagonal = np.arange(min(matrix.shape))
    no_self_array[diagonal, diagonal] = 0.0
    no_self = pd.DataFrame(no_self_array, index=matrix.index, columns=matrix.columns)
    return {"with_self": with_self, "no_self": no_self}


def normalized_matrix_variants(
    matrix: pd.DataFrame,
    *,
    method: str = "spectral_radius",
    target: float = 1.0,
) -> dict[str, dict[str, object]]:
    """Normalize with-self and no-self variants independently."""

    variants: dict[str, dict[str, object]] = {}
    for variant, raw in matrix_variants(matrix).items():
        normalized, normalization = normalize_connectivity(raw, method=method, target=target)
        normalization["variant"] = variant
        variants[variant] = {"matrix": normalized, "normalization": normalization}
    return variants


def _class_indices(classes: pd.Series) -> dict[str, np.ndarray]:
    aligned = classes.map(_normalize_class).to_numpy(dtype=str)
    return {"E": np.flatnonzero(aligned == "e"), "I": np.flatnonzero(aligned == "i")}


def source_receiver_block_view(matrix: pd.DataFrame, classes: pd.Series) -> dict[str, pd.DataFrame]:
    """Return blocks named as source class -> receiver class.

    The matrix convention remains rows=receiver and columns=source, so EI is
    an E-source to I-receiver block.
    """

    aligned = classes.reindex(matrix.index).map(_normalize_class)
    excitatory = aligned[aligned == "e"].index.tolist()
    inhibitory = aligned[aligned == "i"].index.tolist()
    return {
        "EE": matrix.loc[excitatory, excitatory],
        "EI": matrix.loc[inhibitory, excitatory],
        "IE": matrix.loc[excitatory, inhibitory],
        "II": matrix.loc[inhibitory, inhibitory],
    }


def zero_source_receiver_block(matrix: pd.DataFrame, classes: pd.Series, block: str) -> pd.DataFrame:
    """Return a copy with one source->receiver block zeroed."""

    block = block.upper()
    if block not in {"EE", "EI", "IE", "II"}:
        raise ValueError("block must be one of EE, EI, IE, or II.")
    aligned = classes.reindex(matrix.index).map(_normalize_class)
    source_nodes = aligned[aligned == block[0].lower()].index
    receiver_nodes = aligned[aligned == block[1].lower()].index
    out = matrix.copy()
    out.loc[receiver_nodes, source_nodes] = 0.0
    return out


def dominant_eigenpair(matrix: pd.DataFrame | np.ndarray) -> tuple[complex, np.ndarray, np.ndarray]:
    """Return the dominant-by-real-part eigenvalue plus right and left eigenvectors."""

    arr = np.asarray(matrix, dtype=float)
    eigvals, right_vectors = np.linalg.eig(arr)
    left_vals, left_vectors = np.linalg.eig(arr.T.conj())
    idx = int(np.argmax(np.real(eigvals)))
    left_idx = int(np.argmin(np.abs(left_vals - eigvals[idx].conjugate())))
    return eigvals[idx], right_vectors[:, idx], left_vectors[:, left_idx]


def block_meaning(block: str) -> str:
    """Human-readable source->receiver block meaning."""

    labels = {
        "EE": "E source -> E receiver",
        "EI": "E source -> I receiver",
        "IE": "I source -> E receiver",
        "II": "I source -> I receiver",
    }
    return labels[block.upper()]


def block_perturbation_table(matrix: pd.DataFrame, classes: pd.Series) -> pd.DataFrame:
    """Measure dominant eigenvalue effects of removing EE, EI, IE, or II blocks.

    This mirrors the newer perturbation analysis by reporting exact block
    removal effects and the first-order eigenvalue perturbation estimate.
    """

    lam0, right, left = dominant_eigenpair(matrix)
    denom = np.vdot(left, right)
    base = spectral_summary(matrix)
    rows = []
    for block in ("EE", "EI", "IE", "II"):
        removed = zero_source_receiver_block(matrix, classes, block)
        delta = removed.to_numpy(dtype=float) - matrix.to_numpy(dtype=float)
        first_order = np.vdot(left, delta @ right) / denom if abs(denom) > 1e-14 else np.nan
        summary = spectral_summary(removed)
        rows.append(
            {
                "removed_block": block,
                "block_meaning": block_meaning(block),
                "dominant_eigenvalue_before": complex(lam0),
                "dominant_real_before": base["dominant_real"],
                "dominant_imag_before": base["dominant_imag"],
                "dominant_real_after": summary["dominant_real"],
                "dominant_imag_after": summary["dominant_imag"],
                "delta_dominant_real": summary["dominant_real"] - base["dominant_real"],
                "spectral_radius_before": base["spectral_radius"],
                "spectral_radius_after": summary["spectral_radius"],
                "delta_spectral_radius": summary["spectral_radius"] - base["spectral_radius"],
                "first_order_delta_real": float(np.real(first_order)),
                "first_order_delta_imag": float(np.imag(first_order)),
            }
        )
    return pd.DataFrame(rows).sort_values("delta_dominant_real").reset_index(drop=True)


def compare_block_perturbation(
    variants: Mapping[str, Mapping[str, object]],
    classes: pd.Series,
) -> pd.DataFrame:
    """Run block-removal perturbation tables for all variants."""

    rows = []
    for variant, payload in variants.items():
        table = block_perturbation_table(payload["matrix"], classes).copy()
        table.insert(0, "variant", variant)
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def schur_effective_e(
    matrix: pd.DataFrame,
    classes: pd.Series,
    *,
    z: complex | None = None,
) -> dict[str, object]:
    """Eliminate inhibitory nodes with M_EE + M_EI inv(zI - M_II) M_IE."""

    aligned = classes.reindex(matrix.index).map(_normalize_class)
    excitatory = aligned[aligned == "e"].index.tolist()
    inhibitory = aligned[aligned == "i"].index.tolist()
    m_ee = matrix.loc[excitatory, excitatory].to_numpy(dtype=float)
    m_ei = matrix.loc[excitatory, inhibitory].to_numpy(dtype=float)
    m_ie = matrix.loc[inhibitory, excitatory].to_numpy(dtype=float)
    m_ii = matrix.loc[inhibitory, inhibitory].to_numpy(dtype=float)
    if z is None:
        z = spectral_summary(matrix)["dominant_eigenvalue"]
    resolvent = np.linalg.solve(z * np.eye(len(inhibitory), dtype=complex) - m_ii, np.eye(len(inhibitory)))
    feedback = m_ei @ resolvent @ m_ie
    effective = m_ee + feedback
    return {
        "z": z,
        "M_EE": pd.DataFrame(m_ee, index=excitatory, columns=excitatory),
        "M_EI": pd.DataFrame(m_ei, index=excitatory, columns=inhibitory),
        "M_IE": pd.DataFrame(m_ie, index=inhibitory, columns=excitatory),
        "M_II": pd.DataFrame(m_ii, index=inhibitory, columns=inhibitory),
        "resolvent_II": resolvent,
        "inhibitory_feedback": pd.DataFrame(feedback, index=excitatory, columns=excitatory),
        "M_eff_E": pd.DataFrame(effective, index=excitatory, columns=excitatory),
        "rho_MII_over_z": float(spectral_summary(m_ii / z)["spectral_radius"]) if abs(z) > 0 else np.inf,
    }


def schur_summary_table(
    variants: Mapping[str, Mapping[str, object]],
    classes: pd.Series,
) -> pd.DataFrame:
    """Summarize direct E, I-mediated feedback, and effective E dynamics."""

    rows = []
    for variant, payload in variants.items():
        schur = schur_effective_e(payload["matrix"], classes)
        for component in ("M_EE", "inhibitory_feedback", "M_eff_E"):
            summary = spectral_summary(np.real(schur[component]))
            rows.append(
                {
                    "variant": variant,
                    "component": component,
                    "z_real": float(np.real(schur["z"])),
                    "z_imag": float(np.imag(schur["z"])),
                    "rho_MII_over_z": schur["rho_MII_over_z"],
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def excitation_response(
    matrix: pd.DataFrame,
    classes: pd.Series,
    source_population: str,
    *,
    alpha: float = 0.85,
) -> pd.DataFrame:
    """Response to exciting all E or all I nodes using inv(I - alpha M) u."""

    aligned = classes.reindex(matrix.index).map(_normalize_class)
    source_population = source_population.lower()[0]
    selected = aligned[aligned == source_population].index
    u = pd.Series(0.0, index=matrix.index)
    u.loc[selected] = 1.0
    response = np.linalg.solve(np.eye(matrix.shape[0]) - alpha * matrix.to_numpy(dtype=float), u.to_numpy(dtype=float))
    return pd.DataFrame(
        {
            "cell": matrix.index,
            "ei": aligned.to_numpy(),
            "response": np.real(response),
            "abs_response": np.abs(response),
            "source_population": source_population.upper(),
        }
    ).sort_values("abs_response", ascending=False).reset_index(drop=True)


def compare_excitation_responses(
    variants: Mapping[str, Mapping[str, object]],
    classes: pd.Series,
    source_population: str,
    *,
    alpha: float = 0.85,
    top_n: int = 15,
) -> pd.DataFrame:
    """Top excitation responses for each matrix variant."""

    rows = []
    for variant, payload in variants.items():
        table = excitation_response(payload["matrix"], classes, source_population, alpha=alpha).head(top_n).copy()
        table.insert(0, "variant", variant)
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def motif_counts(matrix: pd.DataFrame, classes: pd.Series) -> pd.DataFrame:
    """Count broad directed motifs including E-I-E and E-I-I-E routes."""

    arr = matrix.to_numpy(dtype=float)
    adjacency = arr != 0
    np.fill_diagonal(adjacency, False)
    positive = arr > 0
    negative = arr < 0
    idx = _class_indices(classes.reindex(matrix.index))
    E, I = idx["E"], idx["I"]
    n = arr.shape[0]

    reciprocal = int(np.triu(adjacency & adjacency.T, 1).sum())
    convergent = int(sum(np.sum(adjacency[r, :]) * (np.sum(adjacency[r, :]) - 1) // 2 for r in range(n)))
    divergent = int(sum(np.sum(adjacency[:, c]) * (np.sum(adjacency[:, c]) - 1) // 2 for c in range(n)))
    chains = int(np.sum((adjacency.astype(int) @ adjacency.astype(int)) * (~np.eye(n, dtype=bool))))
    neg_ie = negative[np.ix_(E, I)].astype(int)
    neg_ii = negative[np.ix_(I, I)].astype(int)
    pos_ei = positive[np.ix_(I, E)].astype(int)
    disinhibitory_iie = int(np.sum(neg_ie @ neg_ii))
    e_to_i_to_i_to_e = int(np.sum(neg_ie @ neg_ii @ pos_ei))
    eee = adjacency[np.ix_(E, E)].astype(int)
    iii = adjacency[np.ix_(I, I)].astype(int)
    return pd.DataFrame(
        [
            {"motif": "all_excitatory_edges_E_to_E", "count": int(np.count_nonzero(adjacency[np.ix_(E, E)]))},
            {"motif": "all_inhibitory_edges_I_to_I", "count": int(np.count_nonzero(adjacency[np.ix_(I, I)]))},
            {"motif": "all_excitatory_E_E_E_chains", "count": int(np.sum(eee @ eee))},
            {"motif": "all_inhibitory_I_I_I_chains", "count": int(np.sum(iii @ iii))},
            {"motif": "reciprocal_pair", "count": reciprocal},
            {"motif": "convergent_two_sources_one_receiver", "count": convergent},
            {"motif": "divergent_one_source_two_receivers", "count": divergent},
            {"motif": "directed_two_step_chain", "count": chains},
            {"motif": "disinhibitory_I_to_I_to_E", "count": disinhibitory_iie},
            {"motif": "E_to_I_to_I_to_E_loop", "count": e_to_i_to_i_to_e},
        ]
    )


def motif_enrichment(
    matrix: pd.DataFrame,
    classes: pd.Series,
    *,
    n_null: int = 250,
    random_state: int = 7,
) -> pd.DataFrame:
    """Compare broad motif counts with a shuffled-weight null model."""

    rng = np.random.default_rng(random_state)
    observed = motif_counts(matrix, classes)
    arr = matrix.to_numpy(dtype=float)
    nonzero = arr != 0
    weights = arr[nonzero].copy()
    null_rows = []
    for _ in range(n_null):
        shuffled_arr = np.zeros_like(arr)
        shuffled_arr[nonzero] = rng.permutation(weights)
        shuffled = pd.DataFrame(shuffled_arr, index=matrix.index, columns=matrix.columns)
        null_rows.append(motif_counts(shuffled, classes).set_index("motif")["count"])
    null = pd.concat(null_rows, axis=1).T
    out = observed.copy()
    out["null_mean"] = out["motif"].map(null.mean(axis=0))
    out["null_sd"] = out["motif"].map(null.std(axis=0, ddof=1)).replace(0, np.nan)
    out["z_score"] = (out["count"] - out["null_mean"]) / out["null_sd"]
    out["empirical_p_ge"] = [
        (1 + np.sum(null[row.motif].to_numpy() >= row.count)) / (n_null + 1)
        for row in out.itertuples()
    ]
    return out.sort_values(["z_score", "count"], ascending=False).reset_index(drop=True)


def motif_edge_masks(matrix: pd.DataFrame, classes: pd.Series) -> list[tuple[str, np.ndarray, int, str]]:
    """Build edge masks for broad motif-removal stability experiments."""

    arr = matrix.to_numpy(dtype=float)
    adjacency = arr != 0
    np.fill_diagonal(adjacency, False)
    positive = arr > 0
    negative = arr < 0
    idx = _class_indices(classes.reindex(matrix.index))
    E, I = idx["E"], idx["I"]
    n = arr.shape[0]
    masks: list[tuple[str, np.ndarray, int, str]] = []

    ee_mask = np.zeros((n, n), dtype=bool)
    ee_mask[np.ix_(E, E)] = adjacency[np.ix_(E, E)]
    masks.append(("all_excitatory_edges_E_to_E", ee_mask, int(np.count_nonzero(ee_mask)), "All E-source to E-receiver edges."))

    ii_mask = np.zeros((n, n), dtype=bool)
    ii_mask[np.ix_(I, I)] = adjacency[np.ix_(I, I)]
    masks.append(("all_inhibitory_edges_I_to_I", ii_mask, int(np.count_nonzero(ii_mask)), "All I-source to I-receiver edges."))

    eee_mask = np.zeros((n, n), dtype=bool)
    eee_count = 0
    for e_source in E:
        for e_middle in E:
            if not adjacency[e_middle, e_source]:
                continue
            for e_receiver in E:
                if adjacency[e_receiver, e_middle]:
                    eee_count += 1
                    eee_mask[e_middle, e_source] = True
                    eee_mask[e_receiver, e_middle] = True
    masks.append(("all_excitatory_E_E_E_chains", eee_mask, eee_count, "Two-step chains containing only E nodes."))

    iii_mask = np.zeros((n, n), dtype=bool)
    iii_count = 0
    for i_source in I:
        for i_middle in I:
            if not adjacency[i_middle, i_source]:
                continue
            for i_receiver in I:
                if adjacency[i_receiver, i_middle]:
                    iii_count += 1
                    iii_mask[i_middle, i_source] = True
                    iii_mask[i_receiver, i_middle] = True
    masks.append(("all_inhibitory_I_I_I_chains", iii_mask, iii_count, "Two-step chains containing only I nodes."))

    eie_mask = np.zeros((n, n), dtype=bool)
    eie_count = 0
    for e_source in E:
        for i_middle in I:
            if not positive[i_middle, e_source]:
                continue
            for e_receiver in E:
                if negative[e_receiver, i_middle]:
                    eie_count += 1
                    eie_mask[i_middle, e_source] = True
                    eie_mask[e_receiver, i_middle] = True
    masks.append(("E_I_E_inhibitory_feedback", eie_mask, eie_count, "Expected-sign E->I->E inhibitory feedback paths."))

    eiie_mask = np.zeros((n, n), dtype=bool)
    eiie_count = 0
    for e_source in E:
        for i_first in I:
            if not positive[i_first, e_source]:
                continue
            for i_second in I:
                if not negative[i_second, i_first]:
                    continue
                for e_receiver in E:
                    if negative[e_receiver, i_second]:
                        eiie_count += 1
                        eiie_mask[i_first, e_source] = True
                        eiie_mask[i_second, i_first] = True
                        eiie_mask[e_receiver, i_second] = True
    masks.append(("E_I_I_E_disinhibitory_feedback", eiie_mask, eiie_count, "Expected-sign E->I->I->E disinhibitory paths."))
    return masks


def affected_excitatory_cells_for_mask(
    matrix: pd.DataFrame,
    classes: pd.Series,
    mask: np.ndarray,
) -> dict[str, object]:
    """Return excitatory populations touched by a motif edge mask."""

    aligned = classes.reindex(matrix.index).map(_normalize_class)
    excitatory = set(aligned[aligned == "e"].index)
    active_rows, active_cols = np.where(mask)
    receiver_cells = sorted(
        str(matrix.index[row])
        for row in active_rows
        if str(matrix.index[row]) in excitatory
    )
    source_cells = sorted(
        str(matrix.columns[col])
        for col in active_cols
        if str(matrix.columns[col]) in excitatory
    )
    receiver_cells = sorted(set(receiver_cells))
    source_cells = sorted(set(source_cells))
    all_cells = sorted(set(receiver_cells) | set(source_cells))
    return {
        "affected_excitatory_count": len(all_cells),
        "affected_excitatory_cells": " | ".join(all_cells),
        "affected_excitatory_source_count": len(source_cells),
        "affected_excitatory_sources": " | ".join(source_cells),
        "affected_excitatory_receiver_count": len(receiver_cells),
        "affected_excitatory_receivers": " | ".join(receiver_cells),
    }


def motif_stability_table(matrix: pd.DataFrame, classes: pd.Series) -> pd.DataFrame:
    """Measure stability effects from removing edges in broad motif classes."""

    arr = matrix.to_numpy(dtype=float)
    base = spectral_summary(arr)
    rows = []
    for motif_name, mask, count, description in motif_edge_masks(matrix, classes):
        removed = arr.copy()
        removed[mask] = 0.0
        after = spectral_summary(removed)
        affected_excitatory = affected_excitatory_cells_for_mask(matrix, classes, mask & (arr != 0))
        rows.append(
            {
                "motif": motif_name,
                "description": description,
                "motif_count": int(count),
                "edges_removed": int(np.count_nonzero(mask & (arr != 0))),
                **affected_excitatory,
                "dominant_real_before": base["dominant_real"],
                "dominant_real_after": after["dominant_real"],
                "delta_dominant_real": after["dominant_real"] - base["dominant_real"],
                "spectral_radius_before": base["spectral_radius"],
                "spectral_radius_after": after["spectral_radius"],
                "delta_spectral_radius": after["spectral_radius"] - base["spectral_radius"],
            }
        )
    return pd.DataFrame(rows).sort_values("delta_dominant_real").reset_index(drop=True)


def compare_motif_enrichment(
    variants: Mapping[str, Mapping[str, object]],
    classes: pd.Series,
    *,
    n_null: int = 250,
    random_state: int = 7,
) -> pd.DataFrame:
    """Motif enrichment for each matrix variant."""

    rows = []
    for offset, (variant, payload) in enumerate(variants.items()):
        table = motif_enrichment(payload["matrix"], classes, n_null=n_null, random_state=random_state + offset).copy()
        table.insert(0, "variant", variant)
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def compare_motif_stability(
    variants: Mapping[str, Mapping[str, object]],
    classes: pd.Series,
) -> pd.DataFrame:
    """Motif-removal stability experiments by matrix variant."""

    rows = []
    for variant, payload in variants.items():
        table = motif_stability_table(payload["matrix"], classes).copy()
        table.insert(0, "variant", variant)
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def neumann_ii_model_selection(
    matrix: pd.DataFrame,
    classes: pd.Series,
    *,
    alpha: float = 0.85,
    max_order: int = 12,
    contribution_decay_tol: float = 0.02,
    cumulative_energy: float = 0.95,
    spectral_tail_tol: float = 0.02,
    n_null: int = 250,
    random_state: int = 11,
) -> dict[str, object]:
    """Select how many chained I-I links to retain in inhibitory feedback."""

    rng = np.random.default_rng(random_state)
    aligned = classes.reindex(matrix.index).map(_normalize_class)
    excitatory = aligned[aligned == "e"].index.tolist()
    inhibitory = aligned[aligned == "i"].index.tolist()
    m_ei = matrix.loc[excitatory, inhibitory].to_numpy(dtype=float)
    m_ie = matrix.loc[inhibitory, excitatory].to_numpy(dtype=float)
    m_ii = matrix.loc[inhibitory, inhibitory].to_numpy(dtype=float)
    rho = spectral_summary(alpha * m_ii)["spectral_radius"]

    rows = []
    power = np.eye(len(inhibitory))
    for order in range(max_order + 1):
        if order > 0:
            power = (alpha * m_ii) @ power
        term = m_ei @ power @ m_ie
        real_term = np.real(term)
        rows.append(
            {
                "ii_chain_order": order,
                "motif_length": f"E->I{'->I' * order}->E",
                "fro_norm": float(np.linalg.norm(term, "fro")),
                "signed_sum": float(real_term.sum()),
                "positive_sum": float(np.clip(real_term, 0, None).sum()),
                "negative_sum": float(np.clip(real_term, None, 0).sum()),
                "max_abs": float(np.max(np.abs(term))) if term.size else 0.0,
                "spectral_bound": float(rho**order),
            }
        )

    table = pd.DataFrame(rows)
    total = table["fro_norm"].sum()
    table["relative_contribution"] = table["fro_norm"] / total if total > 0 else 0.0
    table["cumulative_energy"] = table["relative_contribution"].cumsum()
    table["passes_decay_rule"] = table["relative_contribution"] >= contribution_decay_tol
    table["passes_cumulative_energy_rule"] = table["cumulative_energy"] <= cumulative_energy
    table["passes_spectral_radius_rule"] = table["spectral_bound"] >= spectral_tail_tol

    null_norms = []
    nonzero = m_ii != 0
    weights = m_ii[nonzero].copy()
    for _ in range(n_null):
        null_ii = np.zeros_like(m_ii)
        if len(weights):
            null_ii[nonzero] = rng.permutation(weights)
        null_power = np.eye(len(inhibitory))
        one_draw = []
        for order in range(max_order + 1):
            if order > 0:
                null_power = (alpha * null_ii) @ null_power
            one_draw.append(float(np.linalg.norm(m_ei @ null_power @ m_ie, "fro")))
        null_norms.append(one_draw)
    null_arr = np.asarray(null_norms)
    table["null_mean_fro"] = null_arr.mean(axis=0)
    table["null_sd_fro"] = null_arr.std(axis=0, ddof=1)
    table["null_z"] = (table["fro_norm"] - table["null_mean_fro"]) / np.where(table["null_sd_fro"] == 0, np.nan, table["null_sd_fro"])
    table["passes_null_significance_rule"] = table["null_z"] >= 2.0

    selected_by_rule = {
        "contribution_decay": _last_true_order(table, "passes_decay_rule"),
        "cumulative_energy": int(table.loc[table["cumulative_energy"] >= cumulative_energy, "ii_chain_order"].min())
        if (table["cumulative_energy"] >= cumulative_energy).any()
        else max_order,
        "spectral_radius": _last_true_order(table, "passes_spectral_radius_rule"),
        "null_significance": _last_true_order(table, "passes_null_significance_rule"),
    }
    selected_order = int(np.ceil(np.nanmedian(list(selected_by_rule.values()))))
    return {
        "selection_table": table,
        "selected_order": selected_order,
        "alpha_rho_MII": float(rho),
        "selected_by_rule": pd.DataFrame(
            [{"rule": key, "selected_order": value} for key, value in selected_by_rule.items()]
        ),
    }


def _last_true_order(table: pd.DataFrame, column: str) -> int:
    passing = table.loc[table[column], "ii_chain_order"]
    return int(passing.max()) if len(passing) else 0


def compare_neumann_model_selection(
    variants: Mapping[str, Mapping[str, object]],
    classes: pd.Series,
    *,
    alpha: float = 0.85,
    max_order: int = 12,
    contribution_decay_tol: float = 0.02,
    cumulative_energy: float = 0.95,
    spectral_tail_tol: float = 0.02,
    n_null: int = 250,
    random_state: int = 11,
) -> dict[str, object]:
    """Run Neumann model selection for each matrix variant."""

    selection_tables = []
    selected_by_rule = []
    selected_orders = []
    by_variant = {}
    for offset, (variant, payload) in enumerate(variants.items()):
        result = neumann_ii_model_selection(
            payload["matrix"],
            classes,
            alpha=alpha,
            max_order=max_order,
            contribution_decay_tol=contribution_decay_tol,
            cumulative_energy=cumulative_energy,
            spectral_tail_tol=spectral_tail_tol,
            n_null=n_null,
            random_state=random_state + offset,
        )
        by_variant[variant] = result
        table = result["selection_table"].copy()
        table.insert(0, "variant", variant)
        selection_tables.append(table)
        rules = result["selected_by_rule"].copy()
        rules.insert(0, "variant", variant)
        selected_by_rule.append(rules)
        selected_orders.append(
            {
                "variant": variant,
                "selected_order": int(result["selected_order"]),
                "alpha_rho_MII": float(result["alpha_rho_MII"]),
            }
        )
    return {
        "by_variant": by_variant,
        "selection_table": pd.concat(selection_tables, ignore_index=True),
        "selected_by_rule": pd.concat(selected_by_rule, ignore_index=True),
        "selected_orders": pd.DataFrame(selected_orders),
    }


def disinhibition_sources(
    matrix: pd.DataFrame,
    classes: pd.Series,
    *,
    order: int,
    top_n: int = 20,
    alpha: float = 0.85,
) -> pd.DataFrame:
    """Rank inhibitory cells by their contribution to chained disinhibition."""

    aligned = classes.reindex(matrix.index).map(_normalize_class)
    excitatory = aligned[aligned == "e"].index.tolist()
    inhibitory = aligned[aligned == "i"].index.tolist()
    m_ei = matrix.loc[excitatory, inhibitory].to_numpy(dtype=float)
    m_ie = matrix.loc[inhibitory, excitatory].to_numpy(dtype=float)
    m_ii = matrix.loc[inhibitory, inhibitory].to_numpy(dtype=float)
    power = np.linalg.matrix_power(alpha * m_ii, order) if order > 0 else np.eye(len(inhibitory))
    outgoing_to_e = np.sum(np.abs(m_ei), axis=0)
    incoming_from_e = np.sum(np.abs(m_ie), axis=1)
    participation = np.sum(np.abs(power), axis=0) + np.sum(np.abs(power), axis=1)
    positive_drive = np.sum(np.clip(np.real(m_ei @ power), 0, None), axis=0)
    score = outgoing_to_e * (1 + participation) * (1 + incoming_from_e)
    return (
        pd.DataFrame(
            {
                "inhibitory_cell": inhibitory,
                "ii_chain_order": order,
                "score": score,
                "I_to_E_abs": outgoing_to_e,
                "E_to_I_abs": incoming_from_e,
                "II_chain_participation": participation,
                "positive_disinhibitory_drive_to_E": positive_drive,
            }
        )
        .sort_values("score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def compare_disinhibition_sources(
    variants: Mapping[str, Mapping[str, object]],
    classes: pd.Series,
    selected_orders: pd.DataFrame,
    *,
    top_n: int = 20,
    alpha: float = 0.85,
) -> pd.DataFrame:
    """Rank disinhibitory inhibitory cells side-by-side by variant."""

    rows = []
    order_lookup = dict(zip(selected_orders["variant"], selected_orders["selected_order"]))
    for variant, payload in variants.items():
        table = disinhibition_sources(
            payload["matrix"],
            classes,
            order=int(order_lookup[variant]),
            top_n=top_n,
            alpha=alpha,
        ).copy()
        table.insert(0, "variant", variant)
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def enumerate_eie_configurations(
    matrix: pd.DataFrame,
    classes: pd.Series,
    *,
    top_n: int = 50,
    require_expected_signs: bool = True,
) -> pd.DataFrame:
    """Enumerate strongest E->I->E inhibitory routes."""

    aligned = classes.reindex(matrix.index).map(_normalize_class)
    excitatory = aligned[aligned == "e"].index.tolist()
    inhibitory = aligned[aligned == "i"].index.tolist()
    rows = []
    for e_source in excitatory:
        for i_middle in inhibitory:
            w_ei = float(matrix.loc[i_middle, e_source])
            if w_ei == 0 or (require_expected_signs and w_ei <= 0):
                continue
            for e_receiver in excitatory:
                w_ie = float(matrix.loc[e_receiver, i_middle])
                if w_ie == 0 or (require_expected_signs and w_ie >= 0):
                    continue
                contribution = w_ie * w_ei
                rows.append(
                    {
                        "motif": "E-I-E",
                        "E_source": e_source,
                        "I_middle": i_middle,
                        "E_receiver": e_receiver,
                        "E_to_I": w_ei,
                        "I_to_E": w_ie,
                        "signed_contribution": contribution,
                        "abs_contribution": abs(contribution),
                    }
                )
    out = pd.DataFrame(rows)
    return out.sort_values("abs_contribution", ascending=False).head(top_n).reset_index(drop=True) if not out.empty else out


def enumerate_eiie_configurations(
    matrix: pd.DataFrame,
    classes: pd.Series,
    *,
    top_n: int = 50,
    require_expected_signs: bool = True,
) -> pd.DataFrame:
    """Enumerate strongest E->I->I->E disinhibitory routes."""

    aligned = classes.reindex(matrix.index).map(_normalize_class)
    excitatory = aligned[aligned == "e"].index.tolist()
    inhibitory = aligned[aligned == "i"].index.tolist()
    rows = []
    for e_source in excitatory:
        for i_first in inhibitory:
            w_ei = float(matrix.loc[i_first, e_source])
            if w_ei == 0 or (require_expected_signs and w_ei <= 0):
                continue
            for i_second in inhibitory:
                w_ii = float(matrix.loc[i_second, i_first])
                if w_ii == 0 or (require_expected_signs and w_ii >= 0):
                    continue
                for e_receiver in excitatory:
                    w_ie = float(matrix.loc[e_receiver, i_second])
                    if w_ie == 0 or (require_expected_signs and w_ie >= 0):
                        continue
                    contribution = w_ie * w_ii * w_ei
                    rows.append(
                        {
                            "motif": "E-I-I-E",
                            "E_source": e_source,
                            "I_first": i_first,
                            "I_second": i_second,
                            "E_receiver": e_receiver,
                            "E_to_I": w_ei,
                            "I_to_I": w_ii,
                            "I_to_E": w_ie,
                            "signed_contribution": contribution,
                            "abs_contribution": abs(contribution),
                        }
                    )
    out = pd.DataFrame(rows)
    return out.sort_values(["signed_contribution", "abs_contribution"], ascending=False).head(top_n).reset_index(drop=True) if not out.empty else out


def compare_path_configurations(
    variants: Mapping[str, Mapping[str, object]],
    classes: pd.Series,
    *,
    top_n: int = 50,
) -> dict[str, pd.DataFrame]:
    """Enumerate E-I-E and E-I-I-E configurations by matrix variant."""

    eie_rows = []
    eiie_rows = []
    for variant, payload in variants.items():
        eie = enumerate_eie_configurations(payload["matrix"], classes, top_n=top_n).copy()
        if not eie.empty:
            eie.insert(0, "variant", variant)
        eie_rows.append(eie)
        eiie = enumerate_eiie_configurations(payload["matrix"], classes, top_n=top_n).copy()
        if not eiie.empty:
            eiie.insert(0, "variant", variant)
        eiie_rows.append(eiie)
    return {
        "EIE": pd.concat(eie_rows, ignore_index=True) if eie_rows else pd.DataFrame(),
        "EIIE": pd.concat(eiie_rows, ignore_index=True) if eiie_rows else pd.DataFrame(),
    }


def normalization_table(variants: Mapping[str, Mapping[str, object]]) -> pd.DataFrame:
    """Return normalization metadata as a DataFrame."""

    return pd.DataFrame([payload["normalization"] for payload in variants.values()])


def run_side_by_side_analysis(
    data: ConnectivityData,
    *,
    normalization: str = "spectral_radius",
    target: float = 1.0,
    alpha: float = 0.85,
    n_null: int = 250,
) -> dict[str, object]:
    """Run the fuller with-self/no-self inhibitory modulation workflow."""

    variants = normalized_matrix_variants(data.matrix, method=normalization, target=target)
    perturbation = compare_block_perturbation(variants, data.classes)
    schur_summary = schur_summary_table(variants, data.classes)
    response_i = compare_excitation_responses(variants, data.classes, "I", alpha=alpha)
    response_e = compare_excitation_responses(variants, data.classes, "E", alpha=alpha)
    motif = compare_motif_enrichment(variants, data.classes, n_null=n_null)
    motif_stability = compare_motif_stability(variants, data.classes)
    neumann = compare_neumann_model_selection(variants, data.classes, alpha=alpha, n_null=n_null)
    disinhibitors = compare_disinhibition_sources(variants, data.classes, neumann["selected_orders"], alpha=alpha)
    paths = compare_path_configurations(variants, data.classes)
    return {
        "variants": variants,
        "normalization": normalization_table(variants),
        "perturbation": perturbation,
        "schur_summary": schur_summary,
        "response_I": response_i,
        "response_E": response_e,
        "motif": motif,
        "motif_stability": motif_stability,
        "neumann": neumann,
        "disinhibitors": disinhibitors,
        "paths": paths,
    }


def save_side_by_side_outputs(output_dir: str | Path, results: Mapping[str, object]) -> None:
    """Save side-by-side analysis tables."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results["normalization"].to_csv(out / "normalization_by_variant.csv", index=False)
    results["perturbation"].to_csv(out / "block_perturbation_by_variant.csv", index=False)
    results["schur_summary"].to_csv(out / "schur_summary_by_variant.csv", index=False)
    results["response_I"].to_csv(out / "excite_all_I_by_variant.csv", index=False)
    results["response_E"].to_csv(out / "excite_all_E_by_variant.csv", index=False)
    results["motif"].to_csv(out / "motif_enrichment_by_variant.csv", index=False)
    results["motif_stability"].to_csv(out / "motif_stability_by_variant.csv", index=False)
    results["neumann"]["selection_table"].to_csv(out / "neumann_ii_selection_by_variant.csv", index=False)
    results["neumann"]["selected_by_rule"].to_csv(out / "neumann_selected_by_rule_by_variant.csv", index=False)
    results["neumann"]["selected_orders"].to_csv(out / "neumann_selected_orders_by_variant.csv", index=False)
    results["disinhibitors"].to_csv(out / "top_disinhibitory_sources_by_variant.csv", index=False)
    results["paths"]["EIE"].to_csv(out / "top_EIE_configurations_by_variant.csv", index=False)
    results["paths"]["EIIE"].to_csv(out / "top_EIIE_configurations_by_variant.csv", index=False)


def infer_region(name: str) -> str:
    """Infer a coarse region label from a population name."""

    return str(name).split()[0]


def node_metadata(classes: pd.Series) -> pd.DataFrame:
    """Return class and region metadata for each population."""

    labels = classes.map(_normalize_class)
    return pd.DataFrame(
        {
            "class": labels,
            "class_label": labels.map({"e": "excitatory", "i": "inhibitory"}),
            "region": [infer_region(name) for name in labels.index],
        },
        index=labels.index,
    )


def classify_two_edge_triad(edges: list[tuple[str, str]], nodes: Iterable[str]) -> str:
    """Classify a two-edge directed triad using standard triad-census names.

    Edges are represented as ``(pre, post)`` directed pairs. Connected two-edge
    triads are classified as 021D, 021U, or 021C. A reciprocal dyad plus one
    isolated node is classified as 102.
    """

    if len(edges) != 2:
        raise ValueError("Exactly two directed edges are required.")

    first, second = edges
    if first == (second[1], second[0]):
        return "102"

    out_degree = {node: 0 for node in nodes}
    in_degree = {node: 0 for node in nodes}
    for pre, post in edges:
        out_degree[pre] += 1
        in_degree[post] += 1

    if 2 in out_degree.values():
        return "021D"
    if 2 in in_degree.values():
        return "021U"
    return "021C"


def rees_superpattern_for_triad(triad_census_label: str) -> str:
    """Map a directed triad-census label to Rees et al. superpattern notation."""

    try:
        return TRIAD_CENSUS_TO_REES_SUPERPATTERN[str(triad_census_label)]
    except KeyError as exc:
        raise ValueError(f"No Rees superpattern mapping for {triad_census_label!r}.") from exc


def _edge_records_for_nodes(
    matrix: pd.DataFrame,
    classes: pd.Series,
    nodes: tuple[str, str, str],
) -> list[dict[str, object]]:
    records = []
    for pre, post in ((pre, post) for pre in nodes for post in nodes if pre != post):
        weight = float(matrix.loc[post, pre])
        if weight == 0:
            continue
        records.append(
            {
                "pre": pre,
                "post": post,
                "pre_class": _normalize_class(classes.loc[pre]),
                "post_class": _normalize_class(classes.loc[post]),
                "pre_region": infer_region(pre),
                "post_region": infer_region(post),
                "weight": weight,
                "abs_weight": abs(weight),
            }
        )
    return records


def enumerate_two_edge_triads(
    matrix: pd.DataFrame,
    classes: pd.Series,
    *,
    connected_only: bool = True,
    include_self_connections: bool = False,
) -> pd.DataFrame:
    """Enumerate induced three-node triads with exactly two directed edges.

    Rows are postsynaptic targets and columns are presynaptic sources. The
    two-edge selection always ignores self loops. If ``include_self_connections``
    is true, nonzero diagonal entries on the selected three nodes are appended
    afterward as extra within-triad edges. By default reciprocal dyads with an
    isolated third node are excluded, leaving the connected two-edge triad
    motifs 021D, 021U, and 021C.
    """

    aligned_classes = classes.reindex(matrix.index).map(_normalize_class)
    nodes = tuple(matrix.index.astype(str))
    class_by_index = aligned_classes.loc[list(nodes)].to_numpy(dtype=object)
    regions_by_index = np.array([infer_region(node) for node in nodes], dtype=object)
    arr = matrix.loc[list(nodes), list(nodes)].to_numpy(dtype=float, copy=True)
    rows = []

    for triad_index, triad_indices in enumerate(combinations(range(len(nodes)), 3), start=1):
        edge_records = []
        for pre_index, post_index in (
            (pre_index, post_index)
            for pre_index in triad_indices
            for post_index in triad_indices
            if pre_index != post_index
        ):
            weight = float(arr[post_index, pre_index])
            if weight == 0:
                continue
            edge_records.append(
                {
                    "pre": nodes[pre_index],
                    "post": nodes[post_index],
                    "pre_class": class_by_index[pre_index],
                    "post_class": class_by_index[post_index],
                    "pre_region": regions_by_index[pre_index],
                    "post_region": regions_by_index[post_index],
                    "weight": weight,
                    "abs_weight": abs(weight),
                }
            )
        if len(edge_records) != 2:
            continue

        incident_nodes = {
            node
            for edge in edge_records
            for node in (str(edge["pre"]), str(edge["post"]))
        }
        if connected_only and len(incident_nodes) < 3:
            continue

        triad_nodes = tuple(nodes[index] for index in triad_indices)
        directed_edges = [(str(edge["pre"]), str(edge["post"])) for edge in edge_records]
        motif = classify_two_edge_triad(directed_edges, triad_nodes)
        rees_superpattern = rees_superpattern_for_triad(motif)
        triad_classes = class_by_index[list(triad_indices)]
        e_count = int(np.sum(triad_classes == "e"))
        i_count = int(np.sum(triad_classes == "i"))
        regions = regions_by_index[list(triad_indices)].tolist()
        region_signature = "-".join(sorted(set(regions)))
        edge_signature = " + ".join(
            f"{edge['pre_class'].upper()}->{edge['post_class'].upper()}" for edge in edge_records
        )
        nonself_total_weight = sum(float(edge["weight"]) for edge in edge_records)
        nonself_total_abs_weight = sum(float(edge["abs_weight"]) for edge in edge_records)

        if include_self_connections:
            for node_index in triad_indices:
                weight = float(arr[node_index, node_index])
                if weight == 0:
                    continue
                edge_records.append(
                    {
                        "pre": nodes[node_index],
                        "post": nodes[node_index],
                        "pre_class": class_by_index[node_index],
                        "post_class": class_by_index[node_index],
                        "pre_region": regions_by_index[node_index],
                        "post_region": regions_by_index[node_index],
                        "weight": weight,
                        "abs_weight": abs(weight),
                        "edge_type": "self",
                    }
                )
        for edge in edge_records:
            edge.setdefault("edge_type", "between_nodes")

        self_edges = [edge for edge in edge_records if edge["edge_type"] == "self"]
        if self_edges and motif != "102":
            self_weights = np.array([float(edge["weight"]) for edge in self_edges])
            spectral_radius_value = float(np.max(np.abs(self_weights)))
            max_real_value = float(np.max(self_weights))
        elif motif == "102":
            local_array = arr[np.ix_(triad_indices, triad_indices)].copy()
            if not include_self_connections:
                np.fill_diagonal(local_array, 0.0)
            spectrum = stability_summary(local_array, name="triad")
            spectral_radius_value = spectrum["spectral_radius"]
            max_real_value = spectrum["max_real"]
        else:
            spectral_radius_value = 0.0
            max_real_value = 0.0

        row = {
            "triad_id": f"T{triad_index:06d}",
            "motif": motif,
            "rees_superpattern": rees_superpattern,
            "rees_superpattern_name": REES_SUPERPATTERN_NAMES[rees_superpattern],
            "node_a": triad_nodes[0],
            "node_b": triad_nodes[1],
            "node_c": triad_nodes[2],
            "node_ei_signature": f"{e_count}E-{i_count}I",
            "edge_ei_signature": edge_signature,
            "region_signature": region_signature,
            "region_count": len(set(regions)),
            "within_region": len(set(regions)) == 1,
            "nonself_edge_count": 2,
            "self_edge_count": len(self_edges),
            "edge_count": len(edge_records),
            "nonself_total_weight": nonself_total_weight,
            "nonself_total_abs_weight": nonself_total_abs_weight,
            "self_total_weight": sum(float(edge["weight"]) for edge in self_edges),
            "self_total_abs_weight": sum(float(edge["abs_weight"]) for edge in self_edges),
            "total_weight": sum(float(edge["weight"]) for edge in edge_records),
            "total_abs_weight": sum(float(edge["abs_weight"]) for edge in edge_records),
            "mean_abs_weight": np.mean([float(edge["abs_weight"]) for edge in edge_records]),
            "spectral_radius": spectral_radius_value,
            "max_real": max_real_value,
        }
        for number, edge in enumerate(edge_records, start=1):
            for key, value in edge.items():
                row[f"edge_{number}_{key}"] = value
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_triad_categories(
    triads: pd.DataFrame,
    by: str | list[str],
) -> pd.DataFrame:
    """Summarize counts, weights, and local spectra by triad category."""

    if triads.empty:
        return pd.DataFrame()

    grouped = triads.groupby(by, dropna=False)
    return (
        grouped.agg(
            triad_count=("triad_id", "count"),
            total_abs_weight=("total_abs_weight", "sum"),
            median_abs_weight=("total_abs_weight", "median"),
            mean_spectral_radius=("spectral_radius", "mean"),
            max_spectral_radius=("spectral_radius", "max"),
            within_region_rate=("within_region", "mean"),
        )
        .sort_values(["triad_count", "total_abs_weight"], ascending=False)
    )


def _edge_numbers_from_triad(triad: pd.Series | object) -> range:
    if isinstance(triad, pd.Series):
        edge_count = triad.get("edge_count", 2)
    else:
        edge_count = getattr(triad, "edge_count", 2)
    return range(1, int(edge_count) + 1)


def triad_edge_participation(matrix: pd.DataFrame, triads: pd.DataFrame) -> pd.DataFrame:
    """Count how often each directed edge participates in enumerated triads."""

    if triads.empty:
        return pd.DataFrame(
            columns=["pre", "post", "weight", "triad_participation", "triad_weight_mass"]
        )

    records = []
    for _, triad in triads.iterrows():
        for number in _edge_numbers_from_triad(triad):
            records.append(
                {
                    "pre": triad[f"edge_{number}_pre"],
                    "post": triad[f"edge_{number}_post"],
                    "weight": triad[f"edge_{number}_weight"],
                    "abs_weight": triad[f"edge_{number}_abs_weight"],
                    "edge_type": triad.get(f"edge_{number}_edge_type", "between_nodes"),
                }
            )

    edges = pd.DataFrame(records)
    participation = (
        edges.groupby(["pre", "post", "edge_type"], as_index=False)
        .agg(
            weight=("weight", "first"),
            abs_weight=("abs_weight", "first"),
            triad_participation=("weight", "size"),
            triad_weight_mass=("abs_weight", "sum"),
        )
        .sort_values(["triad_participation", "triad_weight_mass"], ascending=False)
    )
    participation["matrix_weight"] = [
        float(matrix.loc[row.post, row.pre]) for row in participation.itertuples()
    ]
    return participation


def aggregate_triad_matrix(
    matrix: pd.DataFrame,
    triads: pd.DataFrame,
    *,
    normalize: str | None = None,
) -> pd.DataFrame:
    """Embed enumerated triad edges back into an aggregate population matrix.

    ``normalize=None`` sums every triad edge occurrence. ``normalize="triad_count"``
    divides the aggregate by the number of enumerated triads.
    """

    aggregate = pd.DataFrame(0.0, index=matrix.index, columns=matrix.columns)
    if triads.empty:
        return aggregate

    divisor = 1.0
    if normalize == "triad_count":
        divisor = float(len(triads))
    elif normalize is not None:
        raise ValueError("normalize must be None or 'triad_count'.")

    for triad in triads.itertuples(index=False):
        for number in _edge_numbers_from_triad(triad):
            pre = getattr(triad, f"edge_{number}_pre")
            post = getattr(triad, f"edge_{number}_post")
            weight = getattr(triad, f"edge_{number}_weight")
            aggregate.loc[post, pre] += float(weight) / divisor

    return aggregate


def triad_schur_decomposition(
    matrix: pd.DataFrame,
    classes: pd.Series,
    triads: pd.DataFrame,
    *,
    regularization: float = 1e-6,
    normalize: str | None = None,
) -> TriadSchurDecomposition:
    """Apply the E/I Schur-complement reduction to the triad aggregate."""

    aggregate = aggregate_triad_matrix(matrix, triads, normalize=normalize)
    triad_blocks = make_ei_blocks(aggregate, classes)
    feedback = effective_inhibitory_feedback(
        triad_blocks,
        form="static",
        regularization=regularization,
    )
    effective = triad_blocks.ee - feedback
    return TriadSchurDecomposition(
        aggregate_matrix=aggregate,
        blocks=triad_blocks,
        feedback=feedback,
        effective_excitation=effective,
        aggregate_stability=stability_summary(aggregate, "triad_aggregate"),
        effective_stability=stability_summary(effective, "triad_schur_effective_EE"),
    )


def randomize_directed_edges(
    matrix: pd.DataFrame,
    classes: pd.Series,
    *,
    rng: np.random.Generator,
    preserve_pre_class_weights: bool = True,
) -> pd.DataFrame:
    """Sample a random directed graph with the same edge count and weights."""

    nodes = list(matrix.index)
    possible_edges = [(pre, post) for pre in nodes for post in nodes if pre != post]
    observed_edges = [
        (pre, post)
        for pre in nodes
        for post in nodes
        if pre != post and float(matrix.loc[post, pre]) != 0
    ]
    edge_count = len(observed_edges)
    sampled_indices = rng.choice(len(possible_edges), size=edge_count, replace=False)
    sampled_edges = [possible_edges[index] for index in sampled_indices]
    observed_weights = np.array([float(matrix.loc[post, pre]) for pre, post in observed_edges])
    randomized = pd.DataFrame(0.0, index=matrix.index, columns=matrix.columns)
    class_labels = classes.reindex(matrix.index).map(_normalize_class)

    if not preserve_pre_class_weights:
        sampled_weights = rng.choice(observed_weights, size=edge_count, replace=True)
        for (pre, post), weight in zip(sampled_edges, sampled_weights):
            randomized.loc[post, pre] = float(weight)
        return randomized

    weights_by_pre_class = {
        label: np.array(
            [
                float(matrix.loc[post, pre])
                for pre, post in observed_edges
                if class_labels.loc[pre] == label
            ]
        )
        for label in ("e", "i")
    }
    fallback_weights = observed_weights
    for pre, post in sampled_edges:
        candidates = weights_by_pre_class.get(class_labels.loc[pre])
        if candidates is None or len(candidates) == 0:
            candidates = fallback_weights
        randomized.loc[post, pre] = float(rng.choice(candidates))
    return randomized


def triad_enrichment_significance(
    matrix: pd.DataFrame,
    classes: pd.Series,
    *,
    by: str,
    n_null: int = 100,
    random_state: int = 0,
    connected_only: bool = True,
    include_self_connections: bool = False,
) -> pd.DataFrame:
    """Compare observed triad category counts with randomized edge nulls."""

    rng = np.random.default_rng(random_state)
    observed_triads = enumerate_two_edge_triads(
        matrix,
        classes,
        connected_only=connected_only,
        include_self_connections=include_self_connections,
    )
    observed_counts = observed_triads[by].value_counts() if not observed_triads.empty else pd.Series(dtype=float)
    null_counts: dict[object, list[int]] = {}

    for _ in range(n_null):
        randomized = randomize_directed_edges(matrix, classes, rng=rng)
        null_triads = enumerate_two_edge_triads(
            randomized,
            classes,
            connected_only=connected_only,
            include_self_connections=include_self_connections,
        )
        counts = null_triads[by].value_counts() if not null_triads.empty else pd.Series(dtype=float)
        for category in set(observed_counts.index).union(counts.index):
            null_counts.setdefault(category, []).append(int(counts.get(category, 0)))
        for category in set(null_counts).difference(counts.index):
            if len(null_counts[category]) < _ + 1:
                null_counts[category].append(0)

    rows = []
    for category in sorted(null_counts, key=lambda value: str(value)):
        values = np.array(null_counts[category], dtype=float)
        observed = int(observed_counts.get(category, 0))
        null_sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(
            {
                by: category,
                "observed_count": observed,
                "null_mean": float(values.mean()) if len(values) else np.nan,
                "null_sd": null_sd,
                "null_p95": float(np.quantile(values, 0.95)) if len(values) else np.nan,
                "enrichment_ratio": observed / values.mean() if len(values) and values.mean() else np.inf,
                "z_score": (observed - values.mean()) / null_sd if null_sd else np.nan,
                "empirical_p_ge": float((np.sum(values >= observed) + 1) / (len(values) + 1)) if len(values) else np.nan,
                "significant_enriched": bool(len(values) and observed > np.quantile(values, 0.95)),
            }
        )

    return pd.DataFrame(rows).sort_values(["observed_count", "enrichment_ratio"], ascending=False)


def _unique_edges_from_triads(triads: pd.DataFrame) -> list[tuple[str, str]]:
    edges = set()
    for triad in triads.itertuples(index=False):
        for number in _edge_numbers_from_triad(triad):
            edges.add((getattr(triad, f"edge_{number}_pre"), getattr(triad, f"edge_{number}_post")))
    return sorted(edges)


def ablate_edges(matrix: pd.DataFrame, edges: Iterable[tuple[str, str]]) -> pd.DataFrame:
    """Return a copy of a matrix with selected directed edges zeroed."""

    ablated = matrix.copy()
    for pre, post in edges:
        ablated.loc[post, pre] = 0.0
    return ablated


def triad_ablation_analysis(
    matrix: pd.DataFrame,
    triads: pd.DataFrame,
    *,
    category_col: str | None = None,
    categories: Iterable[object] | None = None,
) -> pd.DataFrame:
    """Ablate all triad-participating edges, optionally one category at a time."""

    base = stability_summary(matrix, "baseline")
    rows = [base]

    if category_col is None:
        selected = {"all_two_edge_triads": triads}
    else:
        category_values = list(categories) if categories is not None else triads[category_col].drop_duplicates().tolist()
        selected = {f"{category_col}={value}": triads.loc[triads[category_col] == value] for value in category_values}

    for label, subset in selected.items():
        edges = _unique_edges_from_triads(subset)
        row = stability_summary(ablate_edges(matrix, edges), f"ablate_{label}")
        row["ablated_unique_edges"] = len(edges)
        row["delta_max_real"] = row["max_real"] - base["max_real"]
        row["delta_spectral_radius"] = row["spectral_radius"] - base["spectral_radius"]
        rows.append(row)

    return pd.DataFrame(rows).set_index("name")


def triad_ablation_significance(
    matrix: pd.DataFrame,
    triads: pd.DataFrame,
    *,
    category_col: str | None = None,
    categories: Iterable[object] | None = None,
    n_null: int = 100,
    random_state: int = 0,
) -> pd.DataFrame:
    """Compare triad-edge ablation effects with random edge-set ablations."""

    rng = np.random.default_rng(random_state)
    base = stability_summary(matrix, "baseline")
    include_self_edges = any(pre == post for pre, post in _unique_edges_from_triads(triads))
    observed_edges = [
        (pre, post)
        for pre in matrix.columns
        for post in matrix.index
        if (include_self_edges or pre != post) and float(matrix.loc[post, pre]) != 0
    ]

    if category_col is None:
        selected = {"all_two_edge_triads": triads}
    else:
        category_values = list(categories) if categories is not None else triads[category_col].drop_duplicates().tolist()
        selected = {str(value): triads.loc[triads[category_col] == value] for value in category_values}

    rows = []
    for label, subset in selected.items():
        edges = _unique_edges_from_triads(subset)
        edge_count = len(edges)
        observed_summary = stability_summary(ablate_edges(matrix, edges), f"observed_{label}")
        observed_delta = float(observed_summary["spectral_radius"] - base["spectral_radius"])
        null_deltas = []
        for _ in range(n_null):
            sampled_indices = rng.choice(len(observed_edges), size=edge_count, replace=False)
            sampled_edges = [observed_edges[index] for index in sampled_indices]
            null_summary = stability_summary(ablate_edges(matrix, sampled_edges), "null")
            null_deltas.append(float(null_summary["spectral_radius"] - base["spectral_radius"]))
        null_values = np.array(null_deltas)
        rows.append(
            {
                "category": label,
                "ablated_unique_edges": edge_count,
                "observed_delta_spectral_radius": observed_delta,
                "null_mean_delta": float(null_values.mean()) if len(null_values) else np.nan,
                "null_p05_delta": float(np.quantile(null_values, 0.05)) if len(null_values) else np.nan,
                "null_p95_delta": float(np.quantile(null_values, 0.95)) if len(null_values) else np.nan,
                "empirical_p_more_stabilizing": float((np.sum(null_values <= observed_delta) + 1) / (len(null_values) + 1)) if len(null_values) else np.nan,
                "empirical_p_more_destabilizing": float((np.sum(null_values >= observed_delta) + 1) / (len(null_values) + 1)) if len(null_values) else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("observed_delta_spectral_radius")
