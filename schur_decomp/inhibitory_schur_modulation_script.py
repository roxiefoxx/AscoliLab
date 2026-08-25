"""Inhibitory modulation analysis for signed, directed hippocampal matrices.

Matrix convention
-----------------
Input CSV files use rows as sources and columns as receivers. This module
converts them to the state convention used for linear analyses:

    M[receiver, source]
    x[t + 1] = M @ x[t]

Under this convention, columns are outgoing source channels and rows are
incoming receiver channels. Blocks are named by source then receiver in the
same language as the raw netlist: EE means excitatory source to excitatory
receiver, EI means excitatory source to inhibitory receiver, IE means
inhibitory source to excitatory receiver, and II means inhibitory source to
inhibitory receiver.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from schur_core_script import (
    MijData,
    infer_ei_from_signed_rows,
    load_mij_data,
    matrix_variants,
    normalize_state_matrix as normalize_matrix,
    normalized_matrix_variants,
    spectral_summary,
)


def ei_indices(ei: pd.Series) -> dict[str, np.ndarray]:
    """Return integer indices for E and I populations."""
    values = ei.to_numpy(dtype=str)
    return {"E": np.flatnonzero(values == "e"), "I": np.flatnonzero(values == "i")}


def block_view(M: np.ndarray, ei: pd.Series) -> dict[str, np.ndarray]:
    """Return E/I blocks in state convention with names source->receiver."""
    idx = ei_indices(ei)
    E, I = idx["E"], idx["I"]
    return {
        "EE": M[np.ix_(E, E)],
        "EI": M[np.ix_(I, E)],
        "IE": M[np.ix_(E, I)],
        "II": M[np.ix_(I, I)],
    }


def zero_block(M: np.ndarray, ei: pd.Series, block: str) -> np.ndarray:
    """Return a copy with one source->receiver block zeroed."""
    idx = ei_indices(ei)
    source_type, receiver_type = block[0].upper(), block[1].upper()
    cols = idx[source_type]
    rows = idx[receiver_type]
    out = np.asarray(M, dtype=float).copy()
    out[np.ix_(rows, cols)] = 0.0
    return out


def dominant_eigenpair(M: np.ndarray) -> tuple[complex, np.ndarray, np.ndarray]:
    """Return dominant-by-real-part eigenvalue plus right and left eigenvectors."""
    eigvals, vr = np.linalg.eig(M)
    left_vals, vl = np.linalg.eig(M.T.conj())
    idx = int(np.argmax(np.real(eigvals)))
    left_idx = int(np.argmin(np.abs(left_vals - eigvals[idx].conjugate())))
    return eigvals[idx], vr[:, idx], vl[:, left_idx]


def block_perturbation_table(M: np.ndarray, ei: pd.Series) -> pd.DataFrame:
    """Measure eigenvalue effects of removing EE, EI, IE, or II blocks."""
    lam0, v, w = dominant_eigenpair(M)
    denom = np.vdot(w, v)
    base = spectral_summary(M)
    rows = []
    for block in ["EE", "EI", "IE", "II"]:
        M_removed = zero_block(M, ei, block)
        delta = M_removed - M
        first_order = np.vdot(w, delta @ v) / denom if abs(denom) > 1e-14 else np.nan
        summary = spectral_summary(M_removed)
        rows.append({
            "removed_block": block,
            "block_meaning": block_meaning(block),
            "dominant_real_before": base["dominant_real"],
            "dominant_real_after": summary["dominant_real"],
            "delta_dominant_real": summary["dominant_real"] - base["dominant_real"],
            "spectral_radius_before": base["spectral_radius"],
            "spectral_radius_after": summary["spectral_radius"],
            "delta_spectral_radius": summary["spectral_radius"] - base["spectral_radius"],
            "first_order_delta_real": float(np.real(first_order)),
            "first_order_delta_imag": float(np.imag(first_order)),
        })
    return pd.DataFrame(rows).sort_values("delta_dominant_real").reset_index(drop=True)


def compare_block_perturbation(variants: Mapping[str, Mapping[str, object]], ei: pd.Series) -> pd.DataFrame:
    """Block-removal perturbation table for each matrix variant."""
    rows = []
    for variant, payload in variants.items():
        table = block_perturbation_table(payload["M"], ei)
        table.insert(0, "variant", variant)
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def block_meaning(block: str) -> str:
    labels = {
        "EE": "E source -> E receiver",
        "EI": "E source -> I receiver",
        "IE": "I source -> E receiver",
        "II": "I source -> I receiver",
    }
    return labels[block.upper()]


def schur_effective_E(M: np.ndarray, ei: pd.Series, z: complex | None = None) -> dict[str, np.ndarray | complex | float]:
    """Eliminate inhibitory nodes and return the eigenvalue-dependent E Schur complement.

    Effective E dynamics:
        M_EE + M_EI @ inv(z I - M_II) @ M_IE

    where M_EI is I-source -> E-receiver and M_IE is E-source -> I-receiver.
    """
    idx = ei_indices(ei)
    E, I = idx["E"], idx["I"]
    M_EE = M[np.ix_(E, E)]
    M_EI = M[np.ix_(E, I)]
    M_IE = M[np.ix_(I, E)]
    M_II = M[np.ix_(I, I)]
    if z is None:
        z = spectral_summary(M)["dominant_eigenvalue"]
    resolvent = np.linalg.solve(z * np.eye(len(I), dtype=complex) - M_II, np.eye(len(I)))
    feedback = M_EI @ resolvent @ M_IE
    effective = M_EE + feedback
    return {
        "z": z,
        "M_EE": M_EE,
        "M_EI": M_EI,
        "M_IE": M_IE,
        "M_II": M_II,
        "resolvent_II": resolvent,
        "inhibitory_feedback": feedback,
        "M_eff_E": effective,
        "rho_MII_over_z": float(spectral_summary(M_II / z)["spectral_radius"]) if abs(z) > 0 else np.inf,
    }


def excitation_response(
    M: np.ndarray,
    ei: pd.Series,
    source_population: str,
    alpha: float = 0.85,
) -> pd.DataFrame:
    """Response to exciting all E or all I nodes using ``(I - alpha M)^-1 u``."""
    idx = ei_indices(ei)
    source_population = source_population.upper()
    selected = idx[source_population]
    u = np.zeros(M.shape[0])
    u[selected] = 1.0
    response = np.linalg.solve(np.eye(M.shape[0]) - alpha * M, u)
    labels = ei.index.to_list()
    return pd.DataFrame({
        "cell": labels,
        "ei": ei.to_numpy(),
        "response": np.real(response),
        "abs_response": np.abs(response),
        "source_population": source_population,
    }).sort_values("abs_response", ascending=False).reset_index(drop=True)


def compare_excitation_responses(
    variants: Mapping[str, Mapping[str, object]],
    ei: pd.Series,
    source_population: str,
    alpha: float = 0.85,
    top_n: int = 15,
) -> pd.DataFrame:
    """Top excitation responses for each matrix variant."""
    rows = []
    for variant, payload in variants.items():
        table = excitation_response(payload["M"], ei, source_population, alpha=alpha).head(top_n).copy()
        table.insert(0, "variant", variant)
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def motif_enrichment(
    M: np.ndarray,
    ei: pd.Series,
    n_null: int = 250,
    seed: int = 7,
) -> pd.DataFrame:
    """Count simple directed motifs and compare with degree-preserving-ish weight shuffles.

    The null keeps the binary adjacency positions fixed and permutes nonzero
    weights globally, preserving density and weight distribution while breaking
    cell-specific sign/weight placement.
    """
    rng = np.random.default_rng(seed)
    observed = motif_counts(M, ei)
    nonzero = M != 0
    weights = M[nonzero].copy()
    null_rows = []
    for _ in range(n_null):
        shuffled = np.zeros_like(M)
        shuffled[nonzero] = rng.permutation(weights)
        null_rows.append(motif_counts(shuffled, ei).set_index("motif")["count"])
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


def motif_stability_table(M: np.ndarray, ei: pd.Series) -> pd.DataFrame:
    """Quantify stability effect of removing broad E/I motif edge sets.

    This complements count enrichment by asking whether edges participating in
    each motif class push the dominant eigenvalue up or hold it down.
    """
    M = np.asarray(M, dtype=float)
    base = spectral_summary(M)
    rows = []
    for motif_name, mask, count, description in motif_edge_masks(M, ei):
        removed = M.copy()
        removed[mask] = 0.0
        after = spectral_summary(removed)
        rows.append({
            "motif": motif_name,
            "description": description,
            "motif_count": int(count),
            "edges_removed": int(np.count_nonzero(mask & (M != 0))),
            "dominant_eigenvalue_before": base["dominant_eigenvalue"],
            "dominant_real_before": base["dominant_real"],
            "dominant_imag_before": base["dominant_imag"],
            "spectral_radius_before": base["spectral_radius"],
            "dominant_eigenvalue_after": after["dominant_eigenvalue"],
            "dominant_real_after": after["dominant_real"],
            "dominant_imag_after": after["dominant_imag"],
            "spectral_radius_after": after["spectral_radius"],
            "delta_dominant_real": after["dominant_real"] - base["dominant_real"],
            "delta_spectral_radius": after["spectral_radius"] - base["spectral_radius"],
        })
    return pd.DataFrame(rows).sort_values("delta_dominant_real").reset_index(drop=True)


def motif_edge_masks(M: np.ndarray, ei: pd.Series) -> list[tuple[str, np.ndarray, int, str]]:
    """Build edge masks for broad motif-removal stability experiments."""
    A = np.asarray(M) != 0
    np.fill_diagonal(A, False)
    pos = np.asarray(M) > 0
    neg = np.asarray(M) < 0
    idx = ei_indices(ei)
    E, I = idx["E"], idx["I"]
    n = M.shape[0]
    masks: list[tuple[str, np.ndarray, int, str]] = []

    ee_mask = np.zeros((n, n), dtype=bool)
    ee_mask[np.ix_(E, E)] = A[np.ix_(E, E)]
    ee_count = int(np.count_nonzero(ee_mask))
    masks.append((
        "all_excitatory_edges_E_to_E",
        ee_mask,
        ee_count,
        "All edges with excitatory source and excitatory receiver.",
    ))

    ii_mask = np.zeros((n, n), dtype=bool)
    ii_mask[np.ix_(I, I)] = A[np.ix_(I, I)]
    ii_count = int(np.count_nonzero(ii_mask))
    masks.append((
        "all_inhibitory_edges_I_to_I",
        ii_mask,
        ii_count,
        "All edges with inhibitory source and inhibitory receiver.",
    ))

    eee_mask = np.zeros((n, n), dtype=bool)
    eee_count = 0
    for e_source in E:
        for e_middle in E:
            if not A[e_middle, e_source]:
                continue
            for e_receiver in E:
                if A[e_receiver, e_middle]:
                    eee_count += 1
                    eee_mask[e_middle, e_source] = True
                    eee_mask[e_receiver, e_middle] = True
    masks.append((
        "all_excitatory_E_E_E_chains",
        eee_mask,
        eee_count,
        "Two-step chains containing only excitatory nodes.",
    ))

    iii_mask = np.zeros((n, n), dtype=bool)
    iii_count = 0
    for i_source in I:
        for i_middle in I:
            if not A[i_middle, i_source]:
                continue
            for i_receiver in I:
                if A[i_receiver, i_middle]:
                    iii_count += 1
                    iii_mask[i_middle, i_source] = True
                    iii_mask[i_receiver, i_middle] = True
    masks.append((
        "all_inhibitory_I_I_I_chains",
        iii_mask,
        iii_count,
        "Two-step chains containing only inhibitory nodes.",
    ))

    eie_mask = np.zeros((n, n), dtype=bool)
    eie_count = 0
    for e_source in E:
        for i_middle in I:
            if not pos[i_middle, e_source]:
                continue
            for e_receiver in E:
                if neg[e_receiver, i_middle]:
                    eie_count += 1
                    eie_mask[i_middle, e_source] = True
                    eie_mask[e_receiver, i_middle] = True
    masks.append((
        "E_I_E_inhibitory_feedback",
        eie_mask,
        eie_count,
        "Expected-sign E->I->E inhibitory feedback paths.",
    ))

    eiie_mask = np.zeros((n, n), dtype=bool)
    eiie_count = 0
    for e_source in E:
        for i_first in I:
            if not pos[i_first, e_source]:
                continue
            for i_second in I:
                if not neg[i_second, i_first]:
                    continue
                for e_receiver in E:
                    if neg[e_receiver, i_second]:
                        eiie_count += 1
                        eiie_mask[i_first, e_source] = True
                        eiie_mask[i_second, i_first] = True
                        eiie_mask[e_receiver, i_second] = True
    masks.append((
        "E_I_I_E_disinhibitory_feedback",
        eiie_mask,
        eiie_count,
        "Expected-sign E->I->I->E disinhibitory feedback paths.",
    ))
    return masks


def compare_motif_enrichment(
    variants: Mapping[str, Mapping[str, object]],
    ei: pd.Series,
    n_null: int = 250,
    seed: int = 7,
) -> pd.DataFrame:
    """Motif enrichment for each matrix variant."""
    rows = []
    for offset, (variant, payload) in enumerate(variants.items()):
        table = motif_enrichment(payload["M"], ei, n_null=n_null, seed=seed + offset).copy()
        table.insert(0, "variant", variant)
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def compare_motif_stability(
    variants: Mapping[str, Mapping[str, object]],
    ei: pd.Series,
) -> pd.DataFrame:
    """Dominant-eigenvalue motif-removal experiments by matrix variant."""
    rows = []
    for variant, payload in variants.items():
        table = motif_stability_table(payload["M"], ei).copy()
        table.insert(0, "variant", variant)
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def motif_counts(M: np.ndarray, ei: pd.Series) -> pd.DataFrame:
    """Count convergent, divergent, chain, reciprocal, and disinhibitory motifs."""
    A = np.asarray(M) != 0
    np.fill_diagonal(A, False)
    pos = np.asarray(M) > 0
    neg = np.asarray(M) < 0
    E = ei_indices(ei)["E"]
    I = ei_indices(ei)["I"]
    n = M.shape[0]

    reciprocal = int(np.triu(A & A.T, 1).sum())
    convergent = int(sum(np.sum(A[r, :]) * (np.sum(A[r, :]) - 1) // 2 for r in range(n)))
    divergent = int(sum(np.sum(A[:, c]) * (np.sum(A[:, c]) - 1) // 2 for c in range(n)))
    chains = int(np.sum((A.astype(int) @ A.astype(int)) * (~np.eye(n, dtype=bool))))
    neg_EI = neg[np.ix_(E, I)].astype(int)
    neg_II = neg[np.ix_(I, I)].astype(int)
    pos_IE = pos[np.ix_(I, E)].astype(int)
    disinh_iie = int(np.sum(neg_EI @ neg_II))
    e_to_i_to_i_to_e = int(np.sum(neg_EI @ neg_II @ pos_IE))
    EEE = A[np.ix_(E, E)].astype(int)
    III = A[np.ix_(I, I)].astype(int)
    eee_chains = int(np.sum(EEE @ EEE))
    iii_chains = int(np.sum(III @ III))
    all_excitatory_edges = int(np.count_nonzero(A[np.ix_(E, E)]))
    all_inhibitory_edges = int(np.count_nonzero(A[np.ix_(I, I)]))

    return pd.DataFrame([
        {"motif": "all_excitatory_edges_E_to_E", "count": all_excitatory_edges},
        {"motif": "all_inhibitory_edges_I_to_I", "count": all_inhibitory_edges},
        {"motif": "all_excitatory_E_E_E_chains", "count": eee_chains},
        {"motif": "all_inhibitory_I_I_I_chains", "count": iii_chains},
        {"motif": "reciprocal_pair", "count": reciprocal},
        {"motif": "convergent_two_sources_one_receiver", "count": convergent},
        {"motif": "divergent_one_source_two_receivers", "count": divergent},
        {"motif": "directed_two_step_chain", "count": chains},
        {"motif": "disinhibitory_I_to_I_to_E", "count": disinh_iie},
        {"motif": "E_to_I_to_I_to_E_loop", "count": e_to_i_to_i_to_e},
    ])


def neumann_ii_model_selection(
    M: np.ndarray,
    ei: pd.Series,
    alpha: float = 0.85,
    max_order: int = 12,
    contribution_decay_tol: float = 0.02,
    cumulative_energy: float = 0.95,
    spectral_tail_tol: float = 0.02,
    n_null: int = 250,
    seed: int = 11,
) -> dict[str, pd.DataFrame | int | float]:
    """Select an appropriate number of chained I-I connections.

    Decomposes the inhibitory feedback term
        M_EI (alpha M_II)^k M_IE
    so order ``k`` is the number of I-I links inside the eliminated inhibitory
    block. ``k=0`` is E->I->E with no I-I chain; ``k=3`` is E->I-I-I-I->E.
    """
    rng = np.random.default_rng(seed)
    idx = ei_indices(ei)
    E, I = idx["E"], idx["I"]
    M_EI = M[np.ix_(E, I)]
    M_IE = M[np.ix_(I, E)]
    M_II = M[np.ix_(I, I)]
    rho = spectral_summary(alpha * M_II)["spectral_radius"]

    rows = []
    power = np.eye(len(I))
    terms: list[np.ndarray] = []
    for k in range(max_order + 1):
        if k > 0:
            power = (alpha * M_II) @ power
        term = M_EI @ power @ M_IE
        terms.append(term)
        rows.append({
            "ii_chain_order": k,
            "motif_length": f"E->I{'->I' * k}->E",
            "fro_norm": float(np.linalg.norm(term, "fro")),
            "signed_sum": float(np.real(term).sum()),
            "positive_sum": float(np.clip(np.real(term), 0, None).sum()),
            "negative_sum": float(np.clip(np.real(term), None, 0).sum()),
            "max_abs": float(np.max(np.abs(term))) if term.size else 0.0,
            "spectral_bound": float(rho ** k),
        })

    table = pd.DataFrame(rows)
    total = table["fro_norm"].sum()
    table["relative_contribution"] = table["fro_norm"] / total if total > 0 else 0.0
    table["cumulative_energy"] = table["relative_contribution"].cumsum()
    table["passes_decay_rule"] = table["relative_contribution"] >= contribution_decay_tol
    table["passes_cumulative_energy_rule"] = table["cumulative_energy"] <= cumulative_energy
    table["passes_spectral_radius_rule"] = table["spectral_bound"] >= spectral_tail_tol

    null_norms = []
    nz = M_II != 0
    weights = M_II[nz].copy()
    for _ in range(n_null):
        M_II_null = np.zeros_like(M_II)
        if len(weights):
            M_II_null[nz] = rng.permutation(weights)
        null_power = np.eye(len(I))
        one = []
        for k in range(max_order + 1):
            if k > 0:
                null_power = (alpha * M_II_null) @ null_power
            one.append(float(np.linalg.norm(M_EI @ null_power @ M_IE, "fro")))
        null_norms.append(one)
    null_norms_arr = np.asarray(null_norms)
    null_mean = null_norms_arr.mean(axis=0)
    null_sd = null_norms_arr.std(axis=0, ddof=1)
    table["null_mean_fro"] = null_mean
    table["null_sd_fro"] = null_sd
    table["null_z"] = (table["fro_norm"] - table["null_mean_fro"]) / np.where(null_sd == 0, np.nan, null_sd)
    table["passes_null_significance_rule"] = table["null_z"] >= 2.0

    selected_by_rule = {
        "contribution_decay": _last_true_order(table, "passes_decay_rule"),
        "cumulative_energy": int(table.loc[table["cumulative_energy"] >= cumulative_energy, "ii_chain_order"].min())
        if (table["cumulative_energy"] >= cumulative_energy).any() else max_order,
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


def compare_neumann_model_selection(
    variants: Mapping[str, Mapping[str, object]],
    ei: pd.Series,
    alpha: float = 0.85,
    max_order: int = 12,
    contribution_decay_tol: float = 0.02,
    cumulative_energy: float = 0.95,
    spectral_tail_tol: float = 0.02,
    n_null: int = 250,
    seed: int = 11,
) -> dict[str, object]:
    """Run Neumann model selection for each matrix variant."""
    selection_tables = []
    selected_by_rule = []
    selected_orders = []
    by_variant = {}
    for offset, (variant, payload) in enumerate(variants.items()):
        result = neumann_ii_model_selection(
            payload["M"],
            ei,
            alpha=alpha,
            max_order=max_order,
            contribution_decay_tol=contribution_decay_tol,
            cumulative_energy=cumulative_energy,
            spectral_tail_tol=spectral_tail_tol,
            n_null=n_null,
            seed=seed + offset,
        )
        by_variant[variant] = result
        table = result["selection_table"].copy()
        table.insert(0, "variant", variant)
        selection_tables.append(table)
        rules = result["selected_by_rule"].copy()
        rules.insert(0, "variant", variant)
        selected_by_rule.append(rules)
        selected_orders.append({
            "variant": variant,
            "selected_order": int(result["selected_order"]),
            "alpha_rho_MII": float(result["alpha_rho_MII"]),
        })
    return {
        "by_variant": by_variant,
        "selection_table": pd.concat(selection_tables, ignore_index=True),
        "selected_by_rule": pd.concat(selected_by_rule, ignore_index=True),
        "selected_orders": pd.DataFrame(selected_orders),
    }


def _last_true_order(table: pd.DataFrame, column: str) -> int:
    passing = table.loc[table[column], "ii_chain_order"]
    return int(passing.max()) if len(passing) else 0


def disinhibition_sources(
    M: np.ndarray,
    ei: pd.Series,
    order: int,
    top_n: int = 20,
    alpha: float = 0.85,
) -> pd.DataFrame:
    """Rank inhibitory cells by stabilizing disinhibitory feedback at a chain order."""
    idx = ei_indices(ei)
    E, I = idx["E"], idx["I"]
    labels = np.asarray(ei.index.to_list())
    M_EI = M[np.ix_(E, I)]
    M_IE = M[np.ix_(I, E)]
    M_II = M[np.ix_(I, I)]
    P = np.linalg.matrix_power(alpha * M_II, order) if order > 0 else np.eye(len(I))
    outgoing_to_E = np.sum(np.abs(M_EI), axis=0)
    participation = np.sum(np.abs(P), axis=0) + np.sum(np.abs(P), axis=1)
    incoming_from_E = np.sum(np.abs(M_IE), axis=1)
    disinhibitory_drive_to_E = np.sum(np.clip(np.real(M_EI @ P), 0, None), axis=0)
    score = outgoing_to_E * (1 + participation) * (1 + incoming_from_E)
    return pd.DataFrame({
        "inhibitory_cell": labels[I],
        "ii_chain_order": order,
        "score": score,
        "I_to_E_abs": outgoing_to_E,
        "E_to_I_abs": incoming_from_E,
        "II_chain_participation": participation,
        "positive_disinhibitory_drive_to_E": disinhibitory_drive_to_E,
    }).sort_values("score", ascending=False).head(top_n).reset_index(drop=True)


def enumerate_eie_configurations(
    M: np.ndarray,
    ei: pd.Series,
    top_n: int = 50,
    require_expected_signs: bool = True,
) -> pd.DataFrame:
    """Enumerate top E->I->E two-hop inhibitory configurations.

    Contribution is the product:
        M[E_receiver, I_middle] * M[I_middle, E_source]
    """
    idx = ei_indices(ei)
    E, I = idx["E"], idx["I"]
    labels = np.asarray(ei.index.to_list())
    rows = []
    for e_source in E:
        for i_middle in I:
            w_ei = M[i_middle, e_source]
            if w_ei == 0:
                continue
            if require_expected_signs and w_ei <= 0:
                continue
            for e_receiver in E:
                w_ie = M[e_receiver, i_middle]
                if w_ie == 0:
                    continue
                if require_expected_signs and w_ie >= 0:
                    continue
                contribution = w_ie * w_ei
                rows.append({
                    "motif": "E-I-E",
                    "E_source": labels[e_source],
                    "I_middle": labels[i_middle],
                    "E_receiver": labels[e_receiver],
                    "E_to_I": float(w_ei),
                    "I_to_E": float(w_ie),
                    "signed_contribution": float(contribution),
                    "abs_contribution": float(abs(contribution)),
                })
    return (
        pd.DataFrame(rows)
        .sort_values("abs_contribution", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def enumerate_eiie_configurations(
    M: np.ndarray,
    ei: pd.Series,
    top_n: int = 50,
    require_expected_signs: bool = True,
) -> pd.DataFrame:
    """Enumerate top E->I->I->E disinhibitory configurations.

    Contribution is the product:
        M[E_receiver, I_second] * M[I_second, I_first] * M[I_first, E_source]

    With expected signs, E->I is positive and both inhibitory edges are
    negative, so the total contribution is positive disinhibition.
    """
    idx = ei_indices(ei)
    E, I = idx["E"], idx["I"]
    labels = np.asarray(ei.index.to_list())
    rows = []
    for e_source in E:
        for i_first in I:
            w_ei = M[i_first, e_source]
            if w_ei == 0:
                continue
            if require_expected_signs and w_ei <= 0:
                continue
            for i_second in I:
                w_ii = M[i_second, i_first]
                if w_ii == 0:
                    continue
                if require_expected_signs and w_ii >= 0:
                    continue
                for e_receiver in E:
                    w_ie = M[e_receiver, i_second]
                    if w_ie == 0:
                        continue
                    if require_expected_signs and w_ie >= 0:
                        continue
                    contribution = w_ie * w_ii * w_ei
                    rows.append({
                        "motif": "E-I-I-E",
                        "E_source": labels[e_source],
                        "I_first": labels[i_first],
                        "I_second": labels[i_second],
                        "E_receiver": labels[e_receiver],
                        "E_to_I": float(w_ei),
                        "I_to_I": float(w_ii),
                        "I_to_E": float(w_ie),
                        "signed_contribution": float(contribution),
                        "abs_contribution": float(abs(contribution)),
                    })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (
        out.sort_values(["signed_contribution", "abs_contribution"], ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def compare_path_configurations(
    variants: Mapping[str, Mapping[str, object]],
    ei: pd.Series,
    top_n: int = 50,
) -> dict[str, pd.DataFrame]:
    """Enumerate E-I-E and E-I-I-E configurations by matrix variant."""
    eie_rows = []
    eiie_rows = []
    for variant, payload in variants.items():
        eie = enumerate_eie_configurations(payload["M"], ei, top_n=top_n).copy()
        eie.insert(0, "variant", variant)
        eie_rows.append(eie)
        eiie = enumerate_eiie_configurations(payload["M"], ei, top_n=top_n).copy()
        eiie.insert(0, "variant", variant)
        eiie_rows.append(eiie)
    return {
        "EIE": pd.concat(eie_rows, ignore_index=True),
        "EIIE": pd.concat(eiie_rows, ignore_index=True),
    }


def compare_disinhibition_sources(
    variants: Mapping[str, Mapping[str, object]],
    ei: pd.Series,
    selected_orders: pd.DataFrame,
    top_n: int = 20,
    alpha: float = 0.85,
) -> pd.DataFrame:
    """Rank disinhibitory cells side-by-side for each matrix variant."""
    rows = []
    order_lookup = dict(zip(selected_orders["variant"], selected_orders["selected_order"]))
    for variant, payload in variants.items():
        order = int(order_lookup[variant])
        table = disinhibition_sources(payload["M"], ei, order, top_n=top_n, alpha=alpha).copy()
        table.insert(0, "variant", variant)
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def schur_summary_table(variants: Mapping[str, Mapping[str, object]], ei: pd.Series) -> pd.DataFrame:
    """Summarize direct E, I-mediated feedback, and effective E Schur dynamics by variant."""
    rows = []
    for variant, payload in variants.items():
        schur = schur_effective_E(payload["M"], ei)
        for component, matrix in [
            ("M_EE", schur["M_EE"]),
            ("inhibitory_feedback", np.real(schur["inhibitory_feedback"])),
            ("M_eff_E", np.real(schur["M_eff_E"])),
        ]:
            summary = spectral_summary(matrix)
            rows.append({
                "variant": variant,
                "component": component,
                "z_real": float(np.real(schur["z"])),
                "z_imag": float(np.imag(schur["z"])),
                "rho_MII_over_z": schur["rho_MII_over_z"],
                **summary,
            })
    return pd.DataFrame(rows)


def normalization_table(variants: Mapping[str, Mapping[str, object]]) -> pd.DataFrame:
    """Return normalization metadata as a DataFrame."""
    return pd.DataFrame([payload["normalization"] for payload in variants.values()])


def run_side_by_side_analysis(
    data: MijData,
    normalization: str = "spectral_radius",
    target: float = 0.85,
    alpha: float = 0.85,
    n_null: int = 250,
) -> dict[str, object]:
    """Run the full analysis for with-self and no-self matrix variants."""
    variants = normalized_matrix_variants(data.M, method=normalization, target=target)
    perturbation = compare_block_perturbation(variants, data.ei)
    schur_summary = schur_summary_table(variants, data.ei)
    response_i = compare_excitation_responses(variants, data.ei, "I", alpha=alpha)
    response_e = compare_excitation_responses(variants, data.ei, "E", alpha=alpha)
    motif = compare_motif_enrichment(variants, data.ei, n_null=n_null)
    motif_stability = compare_motif_stability(variants, data.ei)
    neumann = compare_neumann_model_selection(variants, data.ei, alpha=alpha, n_null=n_null)
    disinhibitors = compare_disinhibition_sources(variants, data.ei, neumann["selected_orders"], alpha=alpha)
    paths = compare_path_configurations(variants, data.ei)
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


def save_core_outputs(
    output_dir: str | Path,
    perturbation: pd.DataFrame,
    motif: pd.DataFrame,
    neumann: Mapping[str, object],
    disinhibitors: pd.DataFrame,
) -> None:
    """Save the main tables produced by the notebook/script."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    perturbation.to_csv(out / "block_perturbation.csv", index=False)
    motif.to_csv(out / "motif_enrichment.csv", index=False)
    neumann["selection_table"].to_csv(out / "neumann_ii_selection.csv", index=False)
    neumann["selected_by_rule"].to_csv(out / "neumann_selected_by_rule.csv", index=False)
    disinhibitors.to_csv(out / "top_disinhibitory_sources.csv", index=False)


def save_side_by_side_outputs(output_dir: str | Path, results: Mapping[str, object]) -> None:
    """Save side-by-side with-self/no-self analysis tables."""
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


def main() -> None:
    """Run the default inhibitory Schur modulation analysis."""
    data = load_mij_data()
    results = run_side_by_side_analysis(data, normalization="spectral_radius", target=0.85, alpha=0.85, n_null=100)
    print("Loaded", data.matrix_path)
    print("Cells:", len(data.labels), "| E:", len(data.excitatory), "| I:", len(data.inhibitory))
    print("\nNormalization by variant")
    print(results["normalization"])
    print("\nBlock-removal perturbation by variant")
    print(results["perturbation"])
    print("\nSchur effective E summary by variant")
    print(results["schur_summary"])
    print("\nMotif enrichment by variant")
    print(results["motif"])
    print("\nMotif stability by variant")
    print(results["motif_stability"])
    print("\nNeumann I-I chain model selection by variant")
    print(results["neumann"]["selected_orders"])
    print(results["neumann"]["selected_by_rule"])
    print("\nTop disinhibitory inhibitory cells by variant")
    print(results["disinhibitors"])
    print("\nTop E-I-E configurations by variant")
    print(results["paths"]["EIE"])
    print("\nTop E-I-I-E disinhibitory configurations by variant")
    print(results["paths"]["EIIE"])
    save_side_by_side_outputs("outputs/inhibitory_schur_modulation_script", results)


if __name__ == "__main__":
    main()
