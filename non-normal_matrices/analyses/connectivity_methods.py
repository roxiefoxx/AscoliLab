"""Analysis helpers for applying Shao et al. (2025) methods to mij_matrix.csv.

The input CSV is documented as sender-by-receiver. The paper writes J[i, j]
as the weight from sender j to receiver i, so load_mij_matrix returns J as
receiver-by-sender after transposing the CSV values.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConnectivityData:
    """Container for the empirical connectivity matrix and labels."""

    sender_by_receiver: pd.DataFrame
    J: np.ndarray
    labels: list[str]
    cell_class: pd.Series
    label_source: str
    spectral_radius_raw: float
    spectral_radius_normalized: float


def load_mij_matrix(
    csv_path: str | Path,
    netlist_path: str | Path | None = None,
    spectral_radius_target: float = 1.0,
) -> ConnectivityData:
    """Load the empirical matrix and normalize paper-oriented J.

    Parameters
    ----------
    csv_path:
        Path to mij_matrix.csv. Rows are senders, columns are receivers.
    spectral_radius_target:
        Target spectral radius for J after normalization.
    """

    sender_by_receiver = pd.read_csv(csv_path, index_col=0)
    if list(sender_by_receiver.index) != list(sender_by_receiver.columns):
        raise ValueError("mij_matrix.csv must have matching row and column labels.")

    raw = sender_by_receiver.to_numpy(dtype=float)
    # Paper convention: J[receiver, sender] is the synaptic weight from sender to receiver.
    J_paper = raw.T
    eigvals = np.linalg.eigvals(J_paper)
    spectral_radius_raw = float(np.max(np.abs(eigvals)))
    if not np.isfinite(spectral_radius_raw) or spectral_radius_raw <= 0:
        raise ValueError("Cannot normalize a matrix with nonpositive spectral radius.")

    J = J_paper * (spectral_radius_target / spectral_radius_raw)
    normalized_radius = float(np.max(np.abs(np.linalg.eigvals(J))))

    inferred_classes = classify_sender_cell_class(sender_by_receiver)
    if netlist_path is not None and Path(netlist_path).exists():
        cell_class = load_ei_labels_from_netlist(netlist_path, sender_by_receiver.index, inferred_classes)
        label_source = str(netlist_path)
    else:
        cell_class = inferred_classes
        label_source = "outgoing weight signs"

    return ConnectivityData(
        sender_by_receiver=sender_by_receiver,
        J=J,
        labels=list(sender_by_receiver.index),
        cell_class=cell_class,
        label_source=label_source,
        spectral_radius_raw=spectral_radius_raw,
        spectral_radius_normalized=normalized_radius,
    )


def classify_sender_cell_class(sender_by_receiver: pd.DataFrame) -> pd.Series:
    """Infer E/I class from the signs of outgoing weights in each sender row."""

    classes: dict[str, str] = {}
    for label, row in sender_by_receiver.iterrows():
        values = row.to_numpy(dtype=float)
        nonzero = values[values != 0]
        if nonzero.size == 0:
            classes[label] = "silent"
        elif np.all(nonzero > 0):
            classes[label] = "E"
        elif np.all(nonzero < 0):
            classes[label] = "I"
        else:
            classes[label] = "mixed"
    return pd.Series(classes, name="cell_class")


def load_ei_labels_from_netlist(
    netlist_path: str | Path,
    labels: Iterable[str],
    fallback: pd.Series,
) -> pd.Series:
    """Load E/I labels from mij_netlist.csv, falling back to sign inference."""

    netlist = pd.read_csv(netlist_path)
    required = {"pre_neuron", "pre_ei"}
    missing = required - set(netlist.columns)
    if missing:
        raise ValueError(f"Netlist is missing required columns: {sorted(missing)}")

    label_map: dict[str, str] = {}
    for neuron, values in netlist.groupby("pre_neuron")["pre_ei"]:
        unique_values = sorted(set(str(value).strip().upper() for value in values.dropna()))
        if len(unique_values) == 1 and unique_values[0] in {"E", "I"}:
            label_map[str(neuron)] = unique_values[0]

    classes = {}
    for label in labels:
        classes[str(label)] = label_map.get(str(label), str(fallback.loc[label]))
    return pd.Series(classes, name="cell_class")


def spectral_summary(J: np.ndarray, top_k: int = 12) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return leading eigenvalues and matrix-level spectral diagnostics."""

    eigvals = np.linalg.eigvals(J)
    order = np.argsort(np.abs(eigvals))[::-1]
    top = eigvals[order[:top_k]]
    eigen_table = pd.DataFrame(
        {
            "rank_by_abs": np.arange(1, len(top) + 1),
            "real": top.real,
            "imag": top.imag,
            "abs": np.abs(top),
        }
    )
    frob_sq = float(np.linalg.norm(J, ord="fro") ** 2)
    eig_sq = float(np.sum(np.abs(eigvals) ** 2))
    henrici = float(np.sqrt(max(frob_sq - eig_sq, 0.0)))
    numerical_abscissa = float(np.max(np.linalg.eigvalsh((J + J.T) / 2.0)))
    max_real = float(np.max(eigvals.real))
    metrics = {
        "spectral_radius": float(np.max(np.abs(eigvals))),
        "max_real_eigenvalue": max_real,
        "henrici_departure": henrici,
        "numerical_abscissa": numerical_abscissa,
        "condition_number_I_minus_J": safe_condition_number(np.eye(J.shape[0]) - J),
    }
    return eigen_table, metrics


def stability_isn_metrics(J: np.ndarray, labels: list[str], groups: pd.Series) -> dict[str, float | bool]:
    """Summarize full-network stability and E-subnetwork ISN diagnostics."""

    group_values = groups.loc[labels].to_numpy()
    eigenvalues = np.linalg.eigvals(J)
    max_real = float(np.max(eigenvalues.real))
    spectral_radius = float(np.max(np.abs(eigenvalues)))

    e_mask = group_values == "E"
    if np.any(e_mask):
        J_ee = J[np.ix_(e_mask, e_mask)]
        e_eigenvalues = np.linalg.eigvals(J_ee)
        e_max_real = float(np.max(e_eigenvalues.real))
        e_spectral_radius = float(np.max(np.abs(e_eigenvalues)))
    else:
        e_max_real = np.nan
        e_spectral_radius = np.nan

    stability_margin = 1.0 - max_real
    e_subnetwork_stability_margin = 1.0 - e_max_real if np.isfinite(e_max_real) else np.nan
    return {
        "spectral_radius": spectral_radius,
        "max_real_eigenvalue": max_real,
        "stability_margin_1_minus_max_real": stability_margin,
        "stable_by_max_real_lt_1": bool(max_real < 1.0),
        "near_marginal_by_max_real": bool(abs(stability_margin) <= 1e-9),
        "e_subnetwork_spectral_radius": e_spectral_radius,
        "e_subnetwork_max_real_eigenvalue": e_max_real,
        "e_subnetwork_stability_margin_1_minus_max_real": e_subnetwork_stability_margin,
        "e_subnetwork_unstable_by_max_real_gt_1": bool(e_max_real > 1.0) if np.isfinite(e_max_real) else False,
    }


def safe_condition_number(matrix: np.ndarray) -> float:
    """Compute a condition number, returning inf for singular matrices."""

    try:
        return float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        return float("inf")


def block_mean_matrix(J: np.ndarray, labels: list[str], groups: pd.Series) -> np.ndarray:
    """Build a full-size matrix whose entries are E/I block means of J."""

    group_values = groups.loc[labels].to_numpy()
    J0 = np.zeros_like(J, dtype=float)
    for receiver_group in sorted(set(group_values)):
        receiver_mask = group_values == receiver_group
        for sender_group in sorted(set(group_values)):
            sender_mask = group_values == sender_group
            block = J[np.ix_(receiver_mask, sender_mask)]
            J0[np.ix_(receiver_mask, sender_mask)] = float(np.mean(block))
    return J0


def block_summary(J: np.ndarray, labels: list[str], groups: pd.Series) -> pd.DataFrame:
    """Summarize mean, variance, and density by receiver and sender group."""

    group_values = groups.loc[labels].to_numpy()
    rows = []
    for receiver_group in sorted(set(group_values)):
        receiver_mask = group_values == receiver_group
        for sender_group in sorted(set(group_values)):
            sender_mask = group_values == sender_group
            block = J[np.ix_(receiver_mask, sender_mask)]
            rows.append(
                {
                    "receiver_group": receiver_group,
                    "sender_group": sender_group,
                    "n_receiver": int(receiver_mask.sum()),
                    "n_sender": int(sender_mask.sum()),
                    "mean_weight": float(np.mean(block)),
                    "variance": float(np.var(block)),
                    "density_nonzero": float(np.mean(block != 0)),
                    "positive_fraction": float(np.mean(block > 0)),
                    "negative_fraction": float(np.mean(block < 0)),
                }
            )
    return pd.DataFrame(rows)


def _source_target_matrix(J: np.ndarray) -> np.ndarray:
    """Return S[source, target] from paper convention J[target, source]."""

    return J.T


def _composition(indices: Iterable[int], group_values: np.ndarray) -> str:
    """Collapse motif node classes into composition labels such as EEI."""

    return "".join(sorted(str(group_values[idx]) for idx in indices))


def motif_inventory(J: np.ndarray, labels: list[str], groups: pd.Series) -> pd.DataFrame:
    """Enumerate observed binary/weighted motifs using J[target, source].

    This is a descriptive inventory inspired by the reference workflow. The
    Shao-style quantities remain the block-centered motif correlations below.
    """

    S = _source_target_matrix(J)
    group_values = groups.loc[labels].to_numpy()
    n = len(labels)
    rows = []

    for source in range(n):
        for middle in range(n):
            if source == middle or S[source, middle] == 0:
                continue
            for target in range(n):
                if target in {source, middle} or S[middle, target] == 0:
                    continue
                score = float(S[source, middle] * S[middle, target])
                rows.append(
                    {
                        "motif": "chain",
                        "node_1": labels[source],
                        "node_2": labels[middle],
                        "node_3": labels[target],
                        "motif_block": f"{group_values[source]}->{group_values[middle]}->{group_values[target]}",
                        "composition": _composition((source, middle, target), group_values),
                        "weighted_score": score,
                        "abs_weighted_score": abs(score),
                        "edge_1": f"{labels[source]} -> {labels[middle]}",
                        "edge_2": f"{labels[middle]} -> {labels[target]}",
                    }
                )

    for source in range(n):
        targets = np.flatnonzero(S[source, :] != 0)
        targets = targets[targets != source]
        for pos, target_a in enumerate(targets):
            for target_b in targets[pos + 1 :]:
                if target_a == target_b:
                    continue
                score = float(S[source, target_a] * S[source, target_b])
                rows.append(
                    {
                        "motif": "divergent",
                        "node_1": labels[source],
                        "node_2": labels[target_a],
                        "node_3": labels[target_b],
                        "motif_block": f"{group_values[source]}->{group_values[target_a]},{group_values[target_b]}",
                        "composition": _composition((source, target_a, target_b), group_values),
                        "weighted_score": score,
                        "abs_weighted_score": abs(score),
                        "edge_1": f"{labels[source]} -> {labels[target_a]}",
                        "edge_2": f"{labels[source]} -> {labels[target_b]}",
                    }
                )

    for target in range(n):
        sources = np.flatnonzero(S[:, target] != 0)
        sources = sources[sources != target]
        for pos, source_a in enumerate(sources):
            for source_b in sources[pos + 1 :]:
                if source_a == source_b:
                    continue
                score = float(S[source_a, target] * S[source_b, target])
                rows.append(
                    {
                        "motif": "convergent",
                        "node_1": labels[source_a],
                        "node_2": labels[source_b],
                        "node_3": labels[target],
                        "motif_block": f"{group_values[source_a]},{group_values[source_b]}->{group_values[target]}",
                        "composition": _composition((source_a, source_b, target), group_values),
                        "weighted_score": score,
                        "abs_weighted_score": abs(score),
                        "edge_1": f"{labels[source_a]} -> {labels[target]}",
                        "edge_2": f"{labels[source_b]} -> {labels[target]}",
                    }
                )

    for source in range(n):
        for target in range(source + 1, n):
            if S[source, target] == 0 or S[target, source] == 0:
                continue
            score = float(S[source, target] * S[target, source])
            rows.append(
                {
                    "motif": "reciprocal",
                    "node_1": labels[source],
                    "node_2": labels[target],
                    "node_3": None,
                    "motif_block": f"{group_values[source]}<->{group_values[target]}",
                    "composition": _composition((source, target), group_values),
                    "weighted_score": score,
                    "abs_weighted_score": abs(score),
                    "edge_1": f"{labels[source]} -> {labels[target]}",
                    "edge_2": f"{labels[target]} -> {labels[source]}",
                }
            )

    return pd.DataFrame(rows)


def motif_inventory_summary(motif_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize motif inventory overall, by composition, and by directed block."""

    if motif_table.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    overall = (
        motif_table.groupby("motif")
        .agg(
            count=("motif", "size"),
            signed_score_sum=("weighted_score", "sum"),
            abs_score_sum=("abs_weighted_score", "sum"),
            mean_score=("weighted_score", "mean"),
            mean_abs_score=("abs_weighted_score", "mean"),
            max_abs_score=("abs_weighted_score", "max"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )
    by_composition = (
        motif_table.groupby(["motif", "composition"])
        .agg(
            count=("motif", "size"),
            signed_score_sum=("weighted_score", "sum"),
            abs_score_sum=("abs_weighted_score", "sum"),
            mean_score=("weighted_score", "mean"),
            mean_abs_score=("abs_weighted_score", "mean"),
            max_abs_score=("abs_weighted_score", "max"),
        )
        .reset_index()
        .sort_values(["motif", "count"], ascending=[True, False])
    )
    by_block = (
        motif_table.groupby(["motif", "motif_block"])
        .agg(
            count=("motif", "size"),
            signed_score_sum=("weighted_score", "sum"),
            abs_score_sum=("abs_weighted_score", "sum"),
            mean_score=("weighted_score", "mean"),
            mean_abs_score=("abs_weighted_score", "mean"),
            max_abs_score=("abs_weighted_score", "max"),
        )
        .reset_index()
        .sort_values(["motif", "count"], ascending=[True, False])
    )
    return overall, by_composition, by_block


def centered_by_blocks(J: np.ndarray, labels: list[str], groups: pd.Series) -> np.ndarray:
    """Subtract E/I block means from J, yielding the empirical residual Z."""

    return J - block_mean_matrix(J, labels, groups)


def _corr_pair(x: list[float], y: list[float]) -> float:
    """Correlation helper that tolerates degenerate motif samples."""

    if len(x) < 2:
        return np.nan
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if np.std(xa) == 0 or np.std(ya) == 0:
        return np.nan
    return float(np.corrcoef(xa, ya)[0, 1])


def motif_correlations(J: np.ndarray, labels: list[str], groups: pd.Series) -> pd.DataFrame:
    """Estimate weighted second-order motif correlations from one empirical J.

    Definitions follow the paper's receiver-by-sender convention:
    chain:      corr(Z[i, j], Z[j, k]), with i != k
    reciprocal: corr(Z[i, j], Z[j, i])
    divergent: corr(Z[i, j], Z[k, j]), with i != k
    convergent: corr(Z[i, j], Z[i, k]), with j != k
    """

    Z = centered_by_blocks(J, labels, groups)
    group_values = groups.loc[labels].to_numpy()
    unique_groups = sorted(set(group_values))
    rows = []

    for receiver_group in unique_groups:
        receivers = np.flatnonzero(group_values == receiver_group)
        for intermediate_group in unique_groups:
            intermediates = np.flatnonzero(group_values == intermediate_group)
            for origin_group in unique_groups:
                origins = np.flatnonzero(group_values == origin_group)

                chain_x: list[float] = []
                chain_y: list[float] = []
                for i in receivers:
                    for j in intermediates:
                        for k in origins:
                            if i == k:
                                continue
                            chain_x.append(Z[i, j])
                            chain_y.append(Z[j, k])
                rows.append(
                    {
                        "motif": "chain",
                        "receiver_group": receiver_group,
                        "middle_or_sender_group": intermediate_group,
                        "origin_or_partner_group": origin_group,
                        "n_pairs": len(chain_x),
                        "correlation": _corr_pair(chain_x, chain_y),
                    }
                )

    for receiver_group in unique_groups:
        receivers = np.flatnonzero(group_values == receiver_group)
        for sender_group in unique_groups:
            senders = np.flatnonzero(group_values == sender_group)

            reciprocal_x: list[float] = []
            reciprocal_y: list[float] = []
            divergent_x: list[float] = []
            divergent_y: list[float] = []
            convergent_x: list[float] = []
            convergent_y: list[float] = []

            for i in receivers:
                for j in senders:
                    if i != j:
                        reciprocal_x.append(Z[i, j])
                        reciprocal_y.append(Z[j, i])
                    for k in receivers:
                        if i != k:
                            divergent_x.append(Z[i, j])
                            divergent_y.append(Z[k, j])
                    for k in senders:
                        if j != k:
                            convergent_x.append(Z[i, j])
                            convergent_y.append(Z[i, k])

            rows.extend(
                [
                    {
                        "motif": "reciprocal",
                        "receiver_group": receiver_group,
                        "middle_or_sender_group": sender_group,
                        "origin_or_partner_group": receiver_group,
                        "n_pairs": len(reciprocal_x),
                        "correlation": _corr_pair(reciprocal_x, reciprocal_y),
                    },
                    {
                        "motif": "divergent",
                        "receiver_group": receiver_group,
                        "middle_or_sender_group": sender_group,
                        "origin_or_partner_group": receiver_group,
                        "n_pairs": len(divergent_x),
                        "correlation": _corr_pair(divergent_x, divergent_y),
                    },
                    {
                        "motif": "convergent",
                        "receiver_group": receiver_group,
                        "middle_or_sender_group": sender_group,
                        "origin_or_partner_group": sender_group,
                        "n_pairs": len(convergent_x),
                        "correlation": _corr_pair(convergent_x, convergent_y),
                    },
                ]
            )

    return pd.DataFrame(rows)


def response_matrix(
    J: np.ndarray,
    response_scale: float = 1.0,
    *,
    fallback_to_pinv: bool = True,
) -> tuple[np.ndarray, str, float]:
    """Compute Gamma = (I - response_scale * J)^-1.

    The paper's response is undefined when I - J is singular. For empirical
    radius-one matrices, numerical inversion may be ill-conditioned; the solver
    and condition number are returned so downstream outputs can report this.
    """

    system_matrix = np.eye(J.shape[0]) - response_scale * J
    condition_number = safe_condition_number(system_matrix)
    try:
        return np.linalg.inv(system_matrix), "inverse", condition_number
    except np.linalg.LinAlgError:
        if not fallback_to_pinv:
            raise
        return np.linalg.pinv(system_matrix), "pseudoinverse", condition_number


def population_response(
    gamma: np.ndarray,
    labels: list[str],
    groups: pd.Series,
) -> pd.DataFrame:
    """Average response of receiver population p to uniform input to sender q."""

    group_values = groups.loc[labels].to_numpy()
    rows = []
    for receiver_group in sorted(set(group_values)):
        receiver_mask = group_values == receiver_group
        for input_group in sorted(set(group_values)):
            input_mask = group_values == input_group
            block = gamma[np.ix_(receiver_mask, input_mask)]
            # Uniform input to a population sums over input coordinates.
            per_receiver_response = block.sum(axis=1)
            rows.append(
                {
                    "receiver_group": receiver_group,
                    "input_group": input_group,
                    "mean_response": float(np.mean(per_receiver_response)),
                    "std_response": float(np.std(per_receiver_response)),
                    "paradoxical": bool(receiver_group == input_group and np.mean(per_receiver_response) < 0),
                }
            )
    return pd.DataFrame(rows)


def low_rank_response(
    J: np.ndarray,
    ranks: Iterable[int] = (1, 2, 4, 8, 12),
    response_scale: float = 1.0,
) -> dict[int, np.ndarray]:
    """Approximate Gamma using leading left/right eigenmodes of response_scale * J."""

    A = response_scale * J
    eigvals, right = np.linalg.eig(A)
    left_eigvals, left = np.linalg.eig(A.T.conj())
    order = np.argsort(np.abs(eigvals))[::-1]

    approximations: dict[int, np.ndarray] = {}
    for rank in ranks:
        gamma = np.eye(J.shape[0], dtype=complex)
        for idx in order[:rank]:
            value = eigvals[idx]
            left_idx = int(np.argmin(np.abs(left_eigvals.conj() - value)))
            rvec = right[:, idx]
            lvec = left[:, left_idx]
            overlap = np.vdot(lvec, rvec)
            if abs(overlap) < 1e-12 or abs(1.0 - value) < 1e-18:
                continue
            lvec = lvec / overlap.conj()
            gamma += (value / (1.0 - value)) * np.outer(rvec, lvec.conj())
        approximations[int(rank)] = np.real_if_close(gamma, tol=1000)
    return approximations


def low_rank_matrix_diagnostics(J: np.ndarray, ranks: Iterable[int] = (1, 2, 3, 5, 10)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return SVD reconstruction diagnostics as a descriptive check.

    This is not the paper's response approximation; it is included because the
    reference workflow uses SVD to ask how compressible the empirical matrix is.
    """

    U, singular_values, Vt = np.linalg.svd(J, full_matrices=False)
    variance_fraction = singular_values**2 / np.sum(singular_values**2)
    singular_summary = pd.DataFrame(
        {
            "rank": np.arange(1, len(singular_values) + 1),
            "singular_value": singular_values,
            "variance_fraction": variance_fraction,
            "cumulative_variance_fraction": np.cumsum(variance_fraction),
        }
    )

    fro_full = float(np.linalg.norm(J, ord="fro"))
    rows = []
    for rank in ranks:
        k = min(int(rank), len(singular_values))
        J_k = U[:, :k] @ np.diag(singular_values[:k]) @ Vt[:k, :]
        fro_error = float(np.linalg.norm(J - J_k, ord="fro"))
        relative_error = fro_error / fro_full if fro_full > 0 else np.nan
        eigenvalues = np.linalg.eigvals(J_k)
        rows.append(
            {
                "rank": k,
                "fro_error": fro_error,
                "relative_fro_error": relative_error,
                "variance_explained": float(1.0 - relative_error**2) if np.isfinite(relative_error) else np.nan,
                "spectral_radius": float(np.max(np.abs(eigenvalues))),
                "max_real_eigenvalue": float(np.max(eigenvalues.real)),
            }
        )
    return singular_summary, pd.DataFrame(rows)


def effective_connectivity(J: np.ndarray, labels: list[str], groups: pd.Series) -> np.ndarray:
    """Empirical analogue of J_eff = J0 + [Z^2].

    With one observed matrix rather than an ensemble, Z @ Z is used as the
    sample analogue of the expected residual square, then block-averaged to
    retain the deterministic population-level structure used in the paper.
    """

    J0 = block_mean_matrix(J, labels, groups)
    Z = J - J0
    return block_mean_matrix(J0 + Z @ Z, labels, groups)


def effective_connectivity_components(
    J: np.ndarray,
    labels: list[str],
    groups: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the empirical effective connectivity into mean, diagonal, and chain parts."""

    J0 = block_mean_matrix(J, labels, groups)
    Z2 = centered_by_blocks(J, labels, groups) @ centered_by_blocks(J, labels, groups)
    diagonal_component = np.diag(np.diag(Z2))
    chain_component = block_mean_matrix(Z2 - diagonal_component, labels, groups)
    return J0, diagonal_component, chain_component


def binary_motif_enrichment(J: np.ndarray, labels: list[str], groups: pd.Series) -> pd.DataFrame:
    """Count binary motif enrichment against block-density independence."""

    adjacency = J != 0
    group_values = groups.loc[labels].to_numpy()
    unique_groups = sorted(set(group_values))
    group_indices = {group: np.flatnonzero(group_values == group) for group in unique_groups}

    density: dict[tuple[str, str], float] = {}
    for receiver_group in unique_groups:
        receivers = group_indices[receiver_group]
        for sender_group in unique_groups:
            senders = group_indices[sender_group]
            density[(receiver_group, sender_group)] = float(np.mean(adjacency[np.ix_(receivers, senders)]))

    rows = []

    def append_row(
        motif: str,
        receiver_group: str,
        middle_or_sender_group: str,
        origin_or_partner_group: str,
        observed_count: int,
        possible_count: int,
        expected_probability: float,
    ) -> None:
        observed_probability = observed_count / possible_count if possible_count else np.nan
        expected_count = possible_count * expected_probability if possible_count else np.nan
        rows.append(
            {
                "motif": motif,
                "receiver_group": receiver_group,
                "middle_or_sender_group": middle_or_sender_group,
                "origin_or_partner_group": origin_or_partner_group,
                "possible_count": int(possible_count),
                "observed_count": int(observed_count),
                "expected_count_independent": float(expected_count),
                "observed_probability": float(observed_probability),
                "expected_probability_independent": float(expected_probability),
                "excess_probability": float(observed_probability - expected_probability),
                "enrichment_ratio": float(observed_probability / expected_probability)
                if expected_probability > 0
                else np.nan,
            }
        )

    for receiver_group in unique_groups:
        receivers = group_indices[receiver_group]
        for middle_group in unique_groups:
            middles = group_indices[middle_group]
            for origin_group in unique_groups:
                origins = group_indices[origin_group]
                possible = 0
                observed = 0
                for i in receivers:
                    for j in middles:
                        for k in origins:
                            if i == k:
                                continue
                            possible += 1
                            observed += int(adjacency[i, j] and adjacency[j, k])
                append_row(
                    "chain",
                    receiver_group,
                    middle_group,
                    origin_group,
                    observed,
                    possible,
                    density[(receiver_group, middle_group)] * density[(middle_group, origin_group)],
                )

    for receiver_group in unique_groups:
        receivers = group_indices[receiver_group]
        for sender_group in unique_groups:
            senders = group_indices[sender_group]
            possible = 0
            observed = 0
            for i in receivers:
                for j in senders:
                    if i == j:
                        continue
                    possible += 1
                    observed += int(adjacency[i, j] and adjacency[j, i])
            append_row(
                "reciprocal",
                receiver_group,
                sender_group,
                receiver_group,
                observed,
                possible,
                density[(receiver_group, sender_group)] * density[(sender_group, receiver_group)],
            )

    for receiver_group in unique_groups:
        receivers = group_indices[receiver_group]
        for sender_group in unique_groups:
            senders = group_indices[sender_group]
            for partner_receiver_group in unique_groups:
                partner_receivers = group_indices[partner_receiver_group]
                possible = 0
                observed = 0
                for i in receivers:
                    for j in senders:
                        for k in partner_receivers:
                            if i == k:
                                continue
                            possible += 1
                            observed += int(adjacency[i, j] and adjacency[k, j])
                append_row(
                    "divergent",
                    receiver_group,
                    sender_group,
                    partner_receiver_group,
                    observed,
                    possible,
                    density[(receiver_group, sender_group)] * density[(partner_receiver_group, sender_group)],
                )

    for receiver_group in unique_groups:
        receivers = group_indices[receiver_group]
        for sender_group in unique_groups:
            senders = group_indices[sender_group]
            for partner_sender_group in unique_groups:
                partner_senders = group_indices[partner_sender_group]
                possible = 0
                observed = 0
                for i in receivers:
                    for j in senders:
                        for k in partner_senders:
                            if j == k:
                                continue
                            possible += 1
                            observed += int(adjacency[i, j] and adjacency[i, k])
                append_row(
                    "convergent",
                    receiver_group,
                    sender_group,
                    partner_sender_group,
                    observed,
                    possible,
                    density[(receiver_group, sender_group)] * density[(receiver_group, partner_sender_group)],
                )

    return pd.DataFrame(rows)


def motif_totals(J: np.ndarray) -> dict[str, float]:
    """Count binary motifs and signed/absolute weighted motif totals."""

    S = _source_target_matrix(J)
    n = S.shape[0]
    totals: dict[str, float] = {
        "chain_count": 0,
        "divergent_count": 0,
        "convergent_count": 0,
        "reciprocal_count": 0,
        "chain_score_sum": 0.0,
        "chain_abs_score_sum": 0.0,
        "divergent_score_sum": 0.0,
        "divergent_abs_score_sum": 0.0,
        "convergent_score_sum": 0.0,
        "convergent_abs_score_sum": 0.0,
        "reciprocal_score_sum": 0.0,
        "reciprocal_abs_score_sum": 0.0,
    }

    for source in range(n):
        for middle in range(n):
            if source == middle or S[source, middle] == 0:
                continue
            for target in range(n):
                if target in {source, middle} or S[middle, target] == 0:
                    continue
                score = float(S[source, middle] * S[middle, target])
                totals["chain_count"] += 1
                totals["chain_score_sum"] += score
                totals["chain_abs_score_sum"] += abs(score)

    for source in range(n):
        targets = np.flatnonzero(S[source, :] != 0)
        targets = targets[targets != source]
        count = len(targets)
        totals["divergent_count"] += count * (count - 1) // 2
        for pos, target_a in enumerate(targets):
            for target_b in targets[pos + 1 :]:
                score = float(S[source, target_a] * S[source, target_b])
                totals["divergent_score_sum"] += score
                totals["divergent_abs_score_sum"] += abs(score)

    for target in range(n):
        sources = np.flatnonzero(S[:, target] != 0)
        sources = sources[sources != target]
        count = len(sources)
        totals["convergent_count"] += count * (count - 1) // 2
        for pos, source_a in enumerate(sources):
            for source_b in sources[pos + 1 :]:
                score = float(S[source_a, target] * S[source_b, target])
                totals["convergent_score_sum"] += score
                totals["convergent_abs_score_sum"] += abs(score)

    for source in range(n):
        for target in range(source + 1, n):
            if S[source, target] == 0 or S[target, source] == 0:
                continue
            score = float(S[source, target] * S[target, source])
            totals["reciprocal_count"] += 1
            totals["reciprocal_score_sum"] += score
            totals["reciprocal_abs_score_sum"] += abs(score)

    return totals


def block_preserving_weight_shuffle(J: np.ndarray, labels: list[str], groups: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Shuffle weights within E/I receiver-sender blocks as a null control."""

    group_values = groups.loc[labels].to_numpy()
    shuffled = np.zeros_like(J, dtype=float)
    for receiver_group in sorted(set(group_values)):
        receiver_idx = np.flatnonzero(group_values == receiver_group)
        for sender_group in sorted(set(group_values)):
            sender_idx = np.flatnonzero(group_values == sender_group)
            block = J[np.ix_(receiver_idx, sender_idx)].copy()
            valid_mask = np.ones(block.shape, dtype=bool)
            if receiver_group == sender_group and block.shape[0] == block.shape[1]:
                np.fill_diagonal(valid_mask, False)
            values = block[valid_mask].copy()
            rng.shuffle(values)
            new_block = block.copy()
            new_block[valid_mask] = values
            shuffled[np.ix_(receiver_idx, sender_idx)] = new_block
    return shuffled


def null_motif_controls(
    J: np.ndarray,
    labels: list[str],
    groups: pd.Series,
    n_null: int = 200,
    seed: int = 751,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare observed motifs with block-preserving shuffled controls."""

    rng = np.random.default_rng(seed)
    observed = {
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(J)))),
        "max_real_eigenvalue": float(np.max(np.linalg.eigvals(J).real)),
        **motif_totals(J),
    }

    null_rows = []
    for null_id in range(int(n_null)):
        J_null = block_preserving_weight_shuffle(J, labels, groups, rng)
        eigvals = np.linalg.eigvals(J_null)
        null_rows.append(
            {
                "null_id": null_id,
                "spectral_radius": float(np.max(np.abs(eigvals))),
                "max_real_eigenvalue": float(np.max(eigvals.real)),
                **motif_totals(J_null),
            }
        )
    null_table = pd.DataFrame(null_rows)

    comparison_rows = []
    for metric, observed_value in observed.items():
        null_values = null_table[metric].to_numpy(dtype=float)
        null_mean = float(np.mean(null_values))
        null_sd = float(np.std(null_values, ddof=1))
        z_score = (observed_value - null_mean) / null_sd if null_sd > 0 else np.nan
        p_greater = float((np.sum(null_values >= observed_value) + 1) / (len(null_values) + 1))
        p_less = float((np.sum(null_values <= observed_value) + 1) / (len(null_values) + 1))
        p_two_sided = float(min(1.0, 2.0 * min(p_greater, p_less)))
        comparison_rows.append(
            {
                "metric": metric,
                "observed": observed_value,
                "null_mean": null_mean,
                "null_sd": null_sd,
                "z_score": z_score,
                "empirical_p_greater": p_greater,
                "empirical_p_less": p_less,
                "empirical_p_two_sided": p_two_sided,
                "n_null": int(n_null),
                "null_model": "seeded block-preserving weight shuffle",
            }
        )
    return null_table, pd.DataFrame(comparison_rows).sort_values("metric").reset_index(drop=True)


def _parse_motif_edge(edge: str) -> tuple[str, str]:
    """Parse a stored source -> target edge label."""

    source, target = edge.split(" -> ")
    return source.strip(), target.strip()


def _motif_edges(motif_table: pd.DataFrame, motif: str, top_n: int, label_to_idx: dict[str, int]) -> set[tuple[int, int]]:
    """Return unique (source, target) indices from top weighted motifs."""

    if motif_table.empty:
        return set()
    sub = motif_table[motif_table["motif"] == motif].sort_values("abs_weighted_score", ascending=False).head(top_n)
    edges: set[tuple[int, int]] = set()
    for _, row in sub.iterrows():
        for column in ("edge_1", "edge_2"):
            if pd.isna(row[column]):
                continue
            source, target = _parse_motif_edge(str(row[column]))
            if source in label_to_idx and target in label_to_idx:
                edges.add((label_to_idx[source], label_to_idx[target]))
    return edges


def _remove_edges(J: np.ndarray, edges: set[tuple[int, int]]) -> np.ndarray:
    """Set source -> target edges to zero under J[target, source]."""

    perturbed = J.copy()
    for source_idx, target_idx in edges:
        perturbed[target_idx, source_idx] = 0.0
    return perturbed


def _weaken_edges(J: np.ndarray, edges: set[tuple[int, int]], factor: float) -> np.ndarray:
    """Scale source -> target edges under J[target, source]."""

    perturbed = J.copy()
    for source_idx, target_idx in edges:
        perturbed[target_idx, source_idx] *= factor
    return perturbed


def summarize_condition(
    matrix: np.ndarray,
    labels: list[str],
    groups: pd.Series,
    condition: str,
    response_scale: float = 1.0,
    extra_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    """Summarize whole-network stability and population response for one matrix."""

    eigvals = np.linalg.eigvals(matrix)
    stability = stability_isn_metrics(matrix, labels, groups)
    gamma, solver, condition_number = response_matrix(matrix, response_scale=response_scale)
    response = population_response(gamma, labels, groups)
    response_lookup = {(row.receiver_group, row.input_group): row.mean_response for row in response.itertuples(index=False)}
    row = {
        "condition": condition,
        **stability,
        "dominant_eigenvalue_real": float(eigvals[np.argmax(np.abs(eigvals))].real),
        "dominant_eigenvalue_imag": float(eigvals[np.argmax(np.abs(eigvals))].imag),
        "response_solver": solver,
        "condition_number_I_minus_scaled_J": condition_number,
        "mean_E_to_E_input": float(response_lookup.get(("E", "E"), np.nan)),
        "mean_I_to_E_input": float(response_lookup.get(("I", "E"), np.nan)),
        "mean_E_to_I_input": float(response_lookup.get(("E", "I"), np.nan)),
        "mean_I_to_I_input": float(response_lookup.get(("I", "I"), np.nan)),
        "paradoxical_I_response": bool(response_lookup.get(("I", "I"), np.nan) < 0),
        "isn_plausible_stable_full_unstable_E_negative_I_response": bool(
            stability["stable_by_max_real_lt_1"]
            and stability["e_subnetwork_unstable_by_max_real_gt_1"]
            and response_lookup.get(("I", "I"), np.nan) < 0
        ),
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def add_delta_columns(summary: pd.DataFrame, reference_condition: str = "original") -> pd.DataFrame:
    """Add deltas relative to a named baseline condition."""

    summary = summary.copy()
    baseline = summary.loc[summary["condition"] == reference_condition].iloc[0]
    delta_columns = [
        "spectral_radius",
        "max_real_eigenvalue",
        "stability_margin_1_minus_max_real",
        "e_subnetwork_max_real_eigenvalue",
        "e_subnetwork_stability_margin_1_minus_max_real",
        "chain_count",
        "reciprocal_count",
        "convergent_count",
        "divergent_count",
        "mean_I_to_I_input",
    ]
    for column in delta_columns:
        if column in summary.columns:
            summary[f"delta_{column}"] = summary[column] - baseline[column]
    return summary


def motif_stress_tests(
    J: np.ndarray,
    labels: list[str],
    groups: pd.Series,
    motif_table: pd.DataFrame,
    response_scale: float = 1.0,
    top_n_motifs: int = 250,
    seed: int = 751,
) -> pd.DataFrame:
    """Stress-test motif-associated edges and block-scrambling controls."""

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    rng = np.random.default_rng(seed)
    conditions: dict[str, np.ndarray] = {"original": J}
    for motif in ("chain", "reciprocal", "convergent", "divergent"):
        edges = _motif_edges(motif_table, motif, top_n_motifs, label_to_idx)
        conditions[f"remove_top_{motif}_edges"] = _remove_edges(J, edges)
        if motif == "chain":
            conditions["weaken_top_chain_edges_50pct"] = _weaken_edges(J, edges, 0.5)
    conditions["block_shuffle_all"] = block_preserving_weight_shuffle(J, labels, groups, rng)

    group_values = groups.loc[labels].to_numpy()
    rows = []
    for condition, matrix in conditions.items():
        totals = motif_totals(matrix)
        rows.append(
            summarize_condition(
                matrix,
                labels,
                groups,
                condition,
                response_scale=response_scale,
                extra_fields={
                    "n_excitatory": int(np.sum(group_values == "E")),
                    "n_inhibitory": int(np.sum(group_values == "I")),
                    **totals,
                },
            )
        )

    return add_delta_columns(pd.DataFrame(rows))


def block_removal_stability(
    J: np.ndarray,
    labels: list[str],
    groups: pd.Series,
    response_scale: float = 1.0,
    matrix_name: str = "J",
) -> pd.DataFrame:
    """Remove one E/I receiver-sender block at a time and recompute whole-network eigenvalues."""

    group_values = groups.loc[labels].to_numpy()
    rows = [
        summarize_condition(
            J,
            labels,
            groups,
            "original",
            response_scale=response_scale,
            extra_fields={
                "matrix": matrix_name,
                "removed_block": "none",
                "receiver_group_removed": "none",
                "sender_group_removed": "none",
            },
        )
    ]
    for receiver_group in sorted(set(group_values)):
        receiver_mask = group_values == receiver_group
        for sender_group in sorted(set(group_values)):
            sender_mask = group_values == sender_group
            perturbed = J.copy()
            perturbed[np.ix_(receiver_mask, sender_mask)] = 0.0
            block_label = f"{receiver_group}<-{sender_group}"
            rows.append(
                summarize_condition(
                    perturbed,
                    labels,
                    groups,
                    f"remove_{block_label}",
                    response_scale=response_scale,
                    extra_fields={
                        "matrix": matrix_name,
                        "removed_block": block_label,
                        "receiver_group_removed": receiver_group,
                        "sender_group_removed": sender_group,
                        "n_removed_entries": int(receiver_mask.sum() * sender_mask.sum()),
                        "n_removed_nonzero_edges": int(np.count_nonzero(J[np.ix_(receiver_mask, sender_mask)])),
                        "removed_weight_sum": float(np.sum(J[np.ix_(receiver_mask, sender_mask)])),
                        "removed_abs_weight_sum": float(np.sum(np.abs(J[np.ix_(receiver_mask, sender_mask)]))),
                    },
                )
            )
    return add_delta_columns(pd.DataFrame(rows))


def dominant_eigenvalue_sensitivity(J: np.ndarray, labels: list[str], groups: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First-order sensitivity of the dominant eigenvalue to each entry of J."""

    eigenvalues, right = np.linalg.eig(J)
    dominant_idx = int(np.argmax(np.abs(eigenvalues)))
    dominant_value = eigenvalues[dominant_idx]
    left_values, left = np.linalg.eig(J.T.conj())
    left_idx = int(np.argmin(np.abs(left_values.conj() - dominant_value)))
    rvec = right[:, dominant_idx]
    lvec = left[:, left_idx]
    overlap = np.vdot(lvec, rvec)
    if abs(overlap) < 1e-12:
        sensitivity = np.full_like(J, np.nan, dtype=float)
    else:
        lvec = lvec / overlap.conj()
        # d lambda / d J[i,j] = conjugate(left[i]) * right[j].
        sensitivity_complex = np.outer(lvec.conj(), rvec)
        sensitivity = np.real(sensitivity_complex)

    group_values = groups.loc[labels].to_numpy()
    edge_rows = []
    for receiver_idx, receiver_label in enumerate(labels):
        for sender_idx, sender_label in enumerate(labels):
            if J[receiver_idx, sender_idx] == 0:
                continue
            edge_rows.append(
                {
                    "receiver": receiver_label,
                    "sender": sender_label,
                    "receiver_group": group_values[receiver_idx],
                    "sender_group": group_values[sender_idx],
                    "weight": float(J[receiver_idx, sender_idx]),
                    "dominant_lambda_sensitivity": float(sensitivity[receiver_idx, sender_idx]),
                    "weighted_sensitivity": float(J[receiver_idx, sender_idx] * sensitivity[receiver_idx, sender_idx]),
                }
            )
    edge_table = pd.DataFrame(edge_rows).sort_values(
        "weighted_sensitivity", key=lambda values: values.abs(), ascending=False
    )

    block_rows = []
    for receiver_group in sorted(set(group_values)):
        receiver_mask = group_values == receiver_group
        for sender_group in sorted(set(group_values)):
            sender_mask = group_values == sender_group
            block_sensitivity = sensitivity[np.ix_(receiver_mask, sender_mask)]
            block_weights = J[np.ix_(receiver_mask, sender_mask)]
            nonzero_mask = block_weights != 0
            block_rows.append(
                {
                    "receiver_group": receiver_group,
                    "sender_group": sender_group,
                    "n_nonzero": int(np.count_nonzero(nonzero_mask)),
                    "mean_entry_sensitivity": float(np.mean(block_sensitivity)),
                    "sum_weighted_sensitivity": float(np.sum(block_weights * block_sensitivity)),
                    "mean_nonzero_weighted_sensitivity": float(np.mean((block_weights * block_sensitivity)[nonzero_mask]))
                    if np.any(nonzero_mask)
                    else np.nan,
                }
            )
    block_table = pd.DataFrame(block_rows)
    return edge_table, block_table


def chain_enrichment_sweep(
    J: np.ndarray,
    labels: list[str],
    groups: pd.Series,
    response_scale: float = 1.0,
    enrichment_values: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5),
) -> pd.DataFrame:
    """Sweep deterministic enrichment/depletion of the empirical chain component."""

    J0, diagonal_component, chain_component = effective_connectivity_components(J, labels, groups)
    rows = []
    for enrichment in enrichment_values:
        J_perturbed = J0 + diagonal_component + float(enrichment) * chain_component
        eigenvalues = np.linalg.eigvals(J_perturbed)
        stability = stability_isn_metrics(J_perturbed, labels, groups)
        gamma, solver, condition_number = response_matrix(J_perturbed, response_scale=response_scale)
        responses = population_response(gamma, labels, groups)
        for _, response_row in responses.iterrows():
            mean_response = float(response_row["mean_response"])
            is_ii_response = response_row["receiver_group"] == "I" and response_row["input_group"] == "I"
            rows.append(
                {
                    "chain_enrichment_multiplier": float(enrichment),
                    **stability,
                    "response_solver": solver,
                    "condition_number_I_minus_scaled_J": condition_number,
                    "isn_plausible_stable_full_unstable_E_negative_I_response": bool(
                        is_ii_response
                        and stability["stable_by_max_real_lt_1"]
                        and stability["e_subnetwork_unstable_by_max_real_gt_1"]
                        and mean_response < 0
                    ),
                    **response_row.to_dict(),
                }
            )
    return pd.DataFrame(rows)


def isn_stability_summary(
    motif_stress: pd.DataFrame,
    chain_sweep: pd.DataFrame,
    block_removal: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine motif removal/enhancement results into an ISN-focused table."""

    rows = []
    stress_baseline = motif_stress.loc[motif_stress["condition"] == "original"].iloc[0]
    for row in motif_stress.itertuples(index=False):
        rows.append(
            {
                "analysis": "motif_removal_or_scrambling",
                "condition": row.condition,
                "motif_level": np.nan,
                "spectral_radius": row.spectral_radius,
                "max_real_eigenvalue": row.max_real_eigenvalue,
                "stability_margin_1_minus_max_real": row.stability_margin_1_minus_max_real,
                "delta_stability_margin_vs_reference": row.stability_margin_1_minus_max_real
                - stress_baseline.stability_margin_1_minus_max_real,
                "e_subnetwork_max_real_eigenvalue": row.e_subnetwork_max_real_eigenvalue,
                "e_subnetwork_stability_margin_1_minus_max_real": row.e_subnetwork_stability_margin_1_minus_max_real,
                "mean_I_to_I_input": row.mean_I_to_I_input,
                "paradoxical_I_response": row.paradoxical_I_response,
                "isn_plausible_stable_full_unstable_E_negative_I_response": row.isn_plausible_stable_full_unstable_E_negative_I_response,
                "reference": "original full empirical J",
            }
        )

    chain_ii = chain_sweep[(chain_sweep["receiver_group"] == "I") & (chain_sweep["input_group"] == "I")].copy()
    chain_reference = chain_ii.loc[
        (chain_ii["chain_enrichment_multiplier"] - 1.0).abs().idxmin()
    ]
    for row in chain_ii.itertuples(index=False):
        rows.append(
            {
                "analysis": "chain_enrichment_sweep_on_Jeff",
                "condition": f"chain_multiplier_{row.chain_enrichment_multiplier:g}",
                "motif_level": row.chain_enrichment_multiplier,
                "spectral_radius": row.spectral_radius,
                "max_real_eigenvalue": row.max_real_eigenvalue,
                "stability_margin_1_minus_max_real": row.stability_margin_1_minus_max_real,
                "delta_stability_margin_vs_reference": row.stability_margin_1_minus_max_real
                - chain_reference["stability_margin_1_minus_max_real"],
                "e_subnetwork_max_real_eigenvalue": row.e_subnetwork_max_real_eigenvalue,
                "e_subnetwork_stability_margin_1_minus_max_real": row.e_subnetwork_stability_margin_1_minus_max_real,
                "mean_I_to_I_input": row.mean_response,
                "paradoxical_I_response": row.paradoxical,
                "isn_plausible_stable_full_unstable_E_negative_I_response": row.isn_plausible_stable_full_unstable_E_negative_I_response,
                "reference": "chain_multiplier_1 on empirical Jeff",
            }
        )

    if block_removal is not None:
        for row in block_removal.itertuples(index=False):
            rows.append(
                {
                    "analysis": f"block_removal_on_{row.matrix}",
                    "condition": row.condition,
                    "motif_level": np.nan,
                    "spectral_radius": row.spectral_radius,
                    "max_real_eigenvalue": row.max_real_eigenvalue,
                    "stability_margin_1_minus_max_real": row.stability_margin_1_minus_max_real,
                    "delta_stability_margin_vs_reference": row.delta_stability_margin_1_minus_max_real,
                    "e_subnetwork_max_real_eigenvalue": row.e_subnetwork_max_real_eigenvalue,
                    "e_subnetwork_stability_margin_1_minus_max_real": row.e_subnetwork_stability_margin_1_minus_max_real,
                    "mean_I_to_I_input": row.mean_I_to_I_input,
                    "paradoxical_I_response": row.paradoxical_I_response,
                    "isn_plausible_stable_full_unstable_E_negative_I_response": row.isn_plausible_stable_full_unstable_E_negative_I_response,
                    "reference": f"original {row.matrix}",
                    "removed_block": row.removed_block,
                    "receiver_group_removed": row.receiver_group_removed,
                    "sender_group_removed": row.sender_group_removed,
                }
            )

    return pd.DataFrame(rows)


def publishable_stability_table(isn_summary: pd.DataFrame) -> pd.DataFrame:
    """Format dominant-eigenvalue stability effects for manuscript-style reporting."""

    rows = []
    for row in isn_summary.itertuples(index=False):
        delta_lambda = -float(row.delta_stability_margin_vs_reference)
        if row.isn_plausible_stable_full_unstable_E_negative_I_response:
            interpretation = "ISN-like: stable full E/I, unstable E-only, paradoxical I response"
        elif row.stability_margin_1_minus_max_real < 0:
            interpretation = "Full network unstable; paradoxical response not interpretable as stable ISN"
        elif row.e_subnetwork_stability_margin_1_minus_max_real >= 0:
            interpretation = "Stable but not ISN-like because E-only subnetwork is stable"
        elif not row.paradoxical_I_response:
            interpretation = "E-only unstable, but I-to-I response is not paradoxical"
        else:
            interpretation = "Near-threshold case; inspect condition number and response scale"

        if row.analysis == "motif_removal_or_scrambling":
            study = "Ablation / scrambling"
        elif row.analysis == "chain_enrichment_sweep_on_Jeff":
            study = "Chain enrichment on Jeff"
        elif str(row.analysis).startswith("block_removal_on_"):
            study = str(row.analysis).replace("block_removal_on_", "Block removal on ")
        else:
            study = str(row.analysis)
        rows.append(
            {
                "Study": study,
                "Manipulation": row.condition,
                "Motif level": row.motif_level,
                "Reference": row.reference,
                "Stability-leading eigenvalue Re(lambda)": float(row.max_real_eigenvalue),
                "Delta Re(lambda) vs reference": delta_lambda,
                "Stability margin 1 - Re(lambda)": float(row.stability_margin_1_minus_max_real),
                "Delta stability margin vs reference": float(row.delta_stability_margin_vs_reference),
                "E-only Re(lambda)": float(row.e_subnetwork_max_real_eigenvalue),
                "E-only margin 1 - Re(lambda)": float(row.e_subnetwork_stability_margin_1_minus_max_real),
                "Mean I response to I input": float(row.mean_I_to_I_input),
                "Paradoxical I response": bool(row.paradoxical_I_response),
                "ISN-plausible": bool(row.isn_plausible_stable_full_unstable_E_negative_I_response),
                "Interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_markdown_table(table: pd.DataFrame, float_digits: int = 4) -> str:
    """Write a small GitHub-style markdown table without optional dependencies."""

    def format_value(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return ""
            return f"{float(value):.{float_digits}g}"
        return str(value)

    columns = list(table.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        values = [format_value(row[column]).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _linear_scale(values: Iterable[float], output_min: float, output_max: float, pad_fraction: float = 0.08):
    """Create a simple linear scaling function for SVG coordinates."""

    clean = [float(value) for value in values if np.isfinite(float(value))]
    low = min(clean) if clean else 0.0
    high = max(clean) if clean else 1.0
    if low == high:
        low -= 0.5
        high += 0.5
    pad = (high - low) * pad_fraction
    low -= pad
    high += pad

    def scale(value: float) -> float:
        return output_min + (float(value) - low) * (output_max - output_min) / (high - low)

    return scale, low, high


def write_svg_bar_chart(
    path: Path,
    labels: list[str],
    values: list[float],
    title: str,
    x_label: str,
) -> None:
    """Write a horizontal bar chart as standalone SVG."""

    width = 980
    row_height = 44
    margin_left = 290
    margin_right = 60
    margin_top = 72
    margin_bottom = 58
    height = margin_top + margin_bottom + row_height * len(labels)
    x_scale, x_low, x_high = _linear_scale([*values, 0.0], margin_left, width - margin_right)
    zero_x = x_scale(0.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{escape(title)}</text>',
        f'<line x1="{zero_x:.2f}" y1="{margin_top - 18}" x2="{zero_x:.2f}" y2="{height - margin_bottom + 12}" stroke="#333" stroke-width="1.2"/>',
    ]
    for idx, (label, value) in enumerate(zip(labels, values)):
        y = margin_top + idx * row_height
        bar_x = min(zero_x, x_scale(value))
        bar_width = abs(x_scale(value) - zero_x)
        fill = "#4575b4" if value <= 0 else "#d73027"
        parts.extend(
            [
                f'<text x="{margin_left - 12}" y="{y + 18}" text-anchor="end" font-family="Arial" font-size="13">{escape(label)}</text>',
                f'<rect x="{bar_x:.2f}" y="{y}" width="{bar_width:.2f}" height="24" fill="{fill}" opacity="0.88"/>',
                f'<text x="{x_scale(value) + (6 if value >= 0 else -6):.2f}" y="{y + 18}" text-anchor="{"start" if value >= 0 else "end"}" font-family="Arial" font-size="12">{value:+.3f}</text>',
            ]
        )
    axis_y = height - margin_bottom + 16
    parts.extend(
        [
            f'<line x1="{margin_left}" y1="{axis_y}" x2="{width - margin_right}" y2="{axis_y}" stroke="#333" stroke-width="1"/>',
            f'<text x="{margin_left}" y="{axis_y + 28}" text-anchor="start" font-family="Arial" font-size="12">{x_low:.2f}</text>',
            f'<text x="{width - margin_right}" y="{axis_y + 28}" text-anchor="end" font-family="Arial" font-size="12">{x_high:.2f}</text>',
            f'<text x="{(margin_left + width - margin_right) / 2}" y="{height - 12}" text-anchor="middle" font-family="Arial" font-size="14">{escape(x_label)}</text>',
            '<text x="28" y="54" font-family="Arial" font-size="12" fill="#4575b4">Blue: stabilizing</text>',
            '<text x="28" y="72" font-family="Arial" font-size="12" fill="#d73027">Red: destabilizing</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def write_svg_line_chart(
    path: Path,
    x_values: list[float],
    y_values: list[float],
    title: str,
    x_label: str,
    y_label: str,
    threshold: float | None = None,
    point_flags: list[bool] | None = None,
) -> None:
    """Write a simple line chart as standalone SVG."""

    width = 860
    height = 520
    margin_left = 86
    margin_right = 42
    margin_top = 72
    margin_bottom = 72
    extra_y_values = list(y_values)
    if threshold is not None:
        extra_y_values.append(threshold)
    x_scale, x_low, x_high = _linear_scale(x_values, margin_left, width - margin_right)
    y_scale_raw, y_low, y_high = _linear_scale(extra_y_values, height - margin_bottom, margin_top)

    def y_scale(value: float) -> float:
        return y_scale_raw(value)

    points = [(x_scale(x), y_scale(y)) for x, y in zip(x_values, y_values)]
    point_string = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1"/>',
        f'<polyline points="{point_string}" fill="none" stroke="#2c7fb8" stroke-width="3"/>',
    ]
    if threshold is not None:
        y_threshold = y_scale(threshold)
        parts.extend(
            [
                f'<line x1="{margin_left}" y1="{y_threshold:.2f}" x2="{width - margin_right}" y2="{y_threshold:.2f}" stroke="#444" stroke-dasharray="6,5" stroke-width="1.3"/>',
                f'<text x="{width - margin_right - 4}" y="{y_threshold - 6:.2f}" text-anchor="end" font-family="Arial" font-size="12">threshold {threshold:g}</text>',
            ]
        )
    for idx, ((x, y), value) in enumerate(zip(points, y_values)):
        flag = bool(point_flags[idx]) if point_flags is not None else False
        fill = "#1a9850" if flag else "#d73027"
        parts.extend(
            [
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.5" fill="{fill}" stroke="white" stroke-width="1.2"/>',
                f'<text x="{x:.2f}" y="{y - 10:.2f}" text-anchor="middle" font-family="Arial" font-size="11">{value:.3g}</text>',
            ]
        )
    parts.extend(
        [
            f'<text x="{margin_left}" y="{height - margin_bottom + 30}" text-anchor="start" font-family="Arial" font-size="12">{x_low:.2f}</text>',
            f'<text x="{width - margin_right}" y="{height - margin_bottom + 30}" text-anchor="end" font-family="Arial" font-size="12">{x_high:.2f}</text>',
            f'<text x="{margin_left - 10}" y="{height - margin_bottom}" text-anchor="end" font-family="Arial" font-size="12">{y_low:.2f}</text>',
            f'<text x="{margin_left - 10}" y="{margin_top + 4}" text-anchor="end" font-family="Arial" font-size="12">{y_high:.2f}</text>',
            f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="14">{escape(x_label)}</text>',
            f'<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle" font-family="Arial" font-size="14">{escape(y_label)}</text>',
            '<text x="620" y="58" font-family="Arial" font-size="12" fill="#1a9850">Green: ISN/plausible or paradoxical</text>',
            '<text x="620" y="76" font-family="Arial" font-size="12" fill="#d73027">Red: not in that regime</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def write_stability_figures(
    output_dir: Path,
    motif_stress: pd.DataFrame,
    chain_sweep: pd.DataFrame,
    block_removal: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Write SVG figures for ablation/enrichment stability results."""

    figure_paths = {
        "figure_ablation_delta_lambda": output_dir / "figure_ablation_delta_dominant_eigenvalue.svg",
        "figure_block_removal_delta_lambda": output_dir / "figure_block_removal_delta_dominant_eigenvalue.svg",
        "figure_chain_stability": output_dir / "figure_chain_enrichment_stability.svg",
        "figure_chain_i_response": output_dir / "figure_chain_enrichment_i_response.svg",
    }
    ablation = motif_stress[motif_stress["condition"] != "original"].copy()
    write_svg_bar_chart(
        figure_paths["figure_ablation_delta_lambda"],
        labels=ablation["condition"].tolist(),
        values=(ablation["max_real_eigenvalue"] - float(motif_stress.loc[motif_stress["condition"] == "original", "max_real_eigenvalue"].iloc[0])).tolist(),
        title="Motif Ablation/Scrambling Effect on Stability-Leading Eigenvalue",
        x_label="Delta Re(lambda_stability) vs original J",
    )

    chain_ii = chain_sweep[(chain_sweep["receiver_group"] == "I") & (chain_sweep["input_group"] == "I")].copy()
    chain_ii = chain_ii.sort_values("chain_enrichment_multiplier")
    write_svg_line_chart(
        figure_paths["figure_chain_stability"],
        x_values=chain_ii["chain_enrichment_multiplier"].tolist(),
        y_values=chain_ii["max_real_eigenvalue"].tolist(),
        title="Chain Enrichment Moves Jeff Across the Stability Boundary",
        x_label="Chain enrichment multiplier",
        y_label="Re(lambda_stability)",
        threshold=1.0,
        point_flags=chain_ii["isn_plausible_stable_full_unstable_E_negative_I_response"].tolist(),
    )
    write_svg_line_chart(
        figure_paths["figure_chain_i_response"],
        x_values=chain_ii["chain_enrichment_multiplier"].tolist(),
        y_values=chain_ii["mean_response"].tolist(),
        title="Chain Enrichment Flips the Inhibitory Paradoxical Response",
        x_label="Chain enrichment multiplier",
        y_label="Mean I response to I input",
        threshold=0.0,
        point_flags=chain_ii["paradoxical"].tolist(),
    )
    if block_removal is not None:
        block_rows = block_removal[block_removal["condition"] != "original"].copy()
        block_rows["label"] = block_rows["matrix"] + " " + block_rows["removed_block"]
        write_svg_bar_chart(
            figure_paths["figure_block_removal_delta_lambda"],
            labels=block_rows["label"].tolist(),
            values=block_rows["delta_max_real_eigenvalue"].tolist(),
            title="Whole-Network Stability Effect of Removing E/I Blocks",
            x_label="Delta Re(lambda_stability) vs original matrix",
        )
    return figure_paths


def save_outputs(
    data: ConnectivityData,
    output_dir: str | Path,
    response_scale: float = 1.0,
) -> dict[str, Path]:
    """Run the empirical analysis and write reusable CSV/JSON outputs."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = data.labels
    groups = data.cell_class
    J = data.J

    eigen_table, spectral_metrics = spectral_summary(J)
    block_stats = block_summary(J, labels, groups)
    Jeff = effective_connectivity(J, labels, groups)
    jeff_eigen_table, jeff_spectral_metrics = spectral_summary(Jeff)
    motif_table = motif_inventory(J, labels, groups)
    motif_inventory_overall, motif_inventory_by_composition, motif_inventory_by_block = motif_inventory_summary(motif_table)
    motif_stats = motif_correlations(J, labels, groups)
    motif_enrichment = binary_motif_enrichment(J, labels, groups)
    null_motifs, null_comparison = null_motif_controls(J, labels, groups)
    svd_singular_values, low_rank_matrix_errors = low_rank_matrix_diagnostics(J)
    motif_stress = motif_stress_tests(J, labels, groups, motif_table, response_scale=response_scale)
    sensitivity_edges, sensitivity_blocks = dominant_eigenvalue_sensitivity(J, labels, groups)
    chain_sweep = chain_enrichment_sweep(J, labels, groups, response_scale=response_scale)
    block_removal_J = block_removal_stability(J, labels, groups, response_scale=response_scale, matrix_name="J")
    block_removal_Jeff = block_removal_stability(Jeff, labels, groups, response_scale=response_scale, matrix_name="Jeff")
    block_removal = pd.concat([block_removal_J, block_removal_Jeff], ignore_index=True)
    isn_summary = isn_stability_summary(motif_stress, chain_sweep, block_removal=block_removal)
    stability_table = publishable_stability_table(isn_summary)
    stability_figure_paths = write_stability_figures(output_dir, motif_stress, chain_sweep, block_removal=block_removal)

    gamma, solver, response_condition_number = response_matrix(J, response_scale=response_scale)
    response_stats = population_response(gamma, labels, groups)
    response_stats["method"] = "full_matrix_inverse"
    response_stats["response_solver"] = solver
    response_stats["condition_number_I_minus_scaled_J"] = response_condition_number

    response_frames = [response_stats]
    if response_condition_number > 1e12:
        gamma_pinv = np.linalg.pinv(np.eye(J.shape[0]) - response_scale * J)
        response_pinv = population_response(gamma_pinv, labels, groups)
        response_pinv["method"] = "full_matrix_pseudoinverse"
        response_pinv["response_solver"] = "pseudoinverse_ill_conditioned"
        response_pinv["condition_number_I_minus_scaled_J"] = response_condition_number
        response_frames.append(response_pinv)

    gamma_eff, solver_eff, response_eff_condition_number = response_matrix(Jeff, response_scale=response_scale)
    response_eff = population_response(gamma_eff, labels, groups)
    response_eff["method"] = "effective_connectivity"
    response_eff["response_solver"] = solver_eff
    response_eff["condition_number_I_minus_scaled_J"] = response_eff_condition_number
    response_frames.append(response_eff)
    response_compare = pd.concat(response_frames, ignore_index=True)

    low_rank_rows = []
    for rank, gamma_lr in low_rank_response(J, response_scale=response_scale).items():
        lr_stats = population_response(np.real(np.asarray(gamma_lr)), labels, groups)
        lr_stats["rank"] = rank
        low_rank_rows.append(lr_stats)
    low_rank_stats = pd.concat(low_rank_rows, ignore_index=True)

    metadata = pd.DataFrame(
        [
            {
                "input_orientation": "sender rows, receiver columns",
                "paper_orientation": "J[receiver, sender]; input matrix transposed before analysis",
                "label_source": data.label_source,
                "raw_spectral_radius": data.spectral_radius_raw,
                "normalized_spectral_radius": data.spectral_radius_normalized,
                "response_scale": response_scale,
                "n_nodes": len(labels),
                "n_excitatory": int((groups == "E").sum()),
                "n_inhibitory": int((groups == "I").sum()),
                "n_mixed": int((groups == "mixed").sum()),
                "n_silent": int((groups == "silent").sum()),
                **spectral_metrics,
                "jeff_spectral_radius": jeff_spectral_metrics["spectral_radius"],
                "jeff_max_real_eigenvalue": jeff_spectral_metrics["max_real_eigenvalue"],
                "jeff_henrici_departure": jeff_spectral_metrics["henrici_departure"],
                "jeff_numerical_abscissa": jeff_spectral_metrics["numerical_abscissa"],
            }
        ]
    )

    paths = {
        "metadata": output_dir / "metadata.csv",
        "normalized_receiver_by_sender": output_dir / "normalized_J_receiver_by_sender.csv",
        "normalized_sender_by_receiver": output_dir / "normalized_mij_sender_by_receiver.csv",
        "eigenvalues": output_dir / "leading_eigenvalues.csv",
        "jeff_eigenvalues": output_dir / "jeff_leading_eigenvalues.csv",
        "blocks": output_dir / "ei_block_summary.csv",
        "motif_inventory": output_dir / "motif_inventory.csv",
        "motif_inventory_overall": output_dir / "motif_inventory_overall.csv",
        "motif_inventory_by_composition": output_dir / "motif_inventory_by_composition.csv",
        "motif_inventory_by_block": output_dir / "motif_inventory_by_block.csv",
        "motifs": output_dir / "motif_correlations.csv",
        "motif_enrichment": output_dir / "binary_motif_enrichment.csv",
        "null_motifs": output_dir / "block_shuffle_null_motifs.csv",
        "null_comparison": output_dir / "block_shuffle_null_comparison.csv",
        "svd_singular_values": output_dir / "svd_singular_values.csv",
        "low_rank_matrix_errors": output_dir / "low_rank_matrix_errors.csv",
        "motif_stress": output_dir / "motif_stress_tests.csv",
        "block_removal": output_dir / "block_removal_stability.csv",
        "block_removal_J": output_dir / "block_removal_stability_J.csv",
        "block_removal_Jeff": output_dir / "block_removal_stability_Jeff.csv",
        "isn_summary": output_dir / "isn_stability_summary.csv",
        "stability_table": output_dir / "dominant_eigenvalue_stability_table.csv",
        "stability_table_md": output_dir / "dominant_eigenvalue_stability_table.md",
        **stability_figure_paths,
        "sensitivity_edges": output_dir / "dominant_eigenvalue_edge_sensitivity.csv",
        "sensitivity_blocks": output_dir / "dominant_eigenvalue_block_sensitivity.csv",
        "chain_sweep": output_dir / "chain_enrichment_sweep.csv",
        "responses": output_dir / "population_responses.csv",
        "low_rank": output_dir / "low_rank_population_responses.csv",
    }
    metadata.to_csv(paths["metadata"], index=False)
    pd.DataFrame(J, index=labels, columns=labels).to_csv(paths["normalized_receiver_by_sender"])
    pd.DataFrame(J.T, index=labels, columns=labels).to_csv(paths["normalized_sender_by_receiver"])
    eigen_table.to_csv(paths["eigenvalues"], index=False)
    jeff_eigen_table.to_csv(paths["jeff_eigenvalues"], index=False)
    block_stats.to_csv(paths["blocks"], index=False)
    motif_table.to_csv(paths["motif_inventory"], index=False)
    motif_inventory_overall.to_csv(paths["motif_inventory_overall"], index=False)
    motif_inventory_by_composition.to_csv(paths["motif_inventory_by_composition"], index=False)
    motif_inventory_by_block.to_csv(paths["motif_inventory_by_block"], index=False)
    motif_stats.to_csv(paths["motifs"], index=False)
    motif_enrichment.to_csv(paths["motif_enrichment"], index=False)
    null_motifs.to_csv(paths["null_motifs"], index=False)
    null_comparison.to_csv(paths["null_comparison"], index=False)
    svd_singular_values.to_csv(paths["svd_singular_values"], index=False)
    low_rank_matrix_errors.to_csv(paths["low_rank_matrix_errors"], index=False)
    motif_stress.to_csv(paths["motif_stress"], index=False)
    block_removal.to_csv(paths["block_removal"], index=False)
    block_removal_J.to_csv(paths["block_removal_J"], index=False)
    block_removal_Jeff.to_csv(paths["block_removal_Jeff"], index=False)
    isn_summary.to_csv(paths["isn_summary"], index=False)
    stability_table.to_csv(paths["stability_table"], index=False)
    paths["stability_table_md"].write_text(dataframe_to_markdown_table(stability_table), encoding="utf-8")
    sensitivity_edges.to_csv(paths["sensitivity_edges"], index=False)
    sensitivity_blocks.to_csv(paths["sensitivity_blocks"], index=False)
    chain_sweep.to_csv(paths["chain_sweep"], index=False)
    response_compare.to_csv(paths["responses"], index=False)
    low_rank_stats.to_csv(paths["low_rank"], index=False)
    return paths
