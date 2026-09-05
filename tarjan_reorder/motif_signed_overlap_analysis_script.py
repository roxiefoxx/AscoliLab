import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh

from motif_schur_decomposition_analysis_script import (
    add_acronym_columns,
    acronym_table,
    make_acronym_map,
)


def load_blocks_and_metadata(csv_path, npz_path):
    """Load motif blocks exactly as stored; callers choose any orientation transpose."""
    metadata = pd.read_csv(csv_path)
    arrays = np.load(npz_path, allow_pickle=True)
    blocks = np.asarray(arrays["analysis_blocks"], dtype=float)
    if blocks.shape != (len(metadata), 3, 3):
        raise ValueError(f"Expected {(len(metadata), 3, 3)}, got {blocks.shape}.")
    return metadata, blocks


def _orient_eigenvector(vector):
    vector = np.asarray(vector, dtype=complex)
    anchor = int(np.argmax(np.abs(vector)))
    if np.abs(vector[anchor]) == 0:
        return np.real(vector)
    phase = np.exp(-1j * np.angle(vector[anchor]))
    oriented = vector * phase
    if oriented[anchor].real < 0:
        oriented = -oriented
    signed = np.real(oriented)
    norm = np.linalg.norm(signed)
    return signed / norm if norm else signed


def signed_dominant_mode_table(metadata, blocks, scenario, acronym_map=None):
    records = []
    for row_position, block in enumerate(blocks):
        values, vectors = np.linalg.eig(block)
        dominant = int(np.argmax(np.abs(values)))
        value = values[dominant]
        signed = _orient_eigenvector(vectors[:, dominant])
        row = metadata.iloc[row_position]
        node_names = [row.node_1, row.node_2, row.node_3]
        record = {
            "scenario": scenario,
            "motif_index": int(row.motif_index),
            "nodes": row.nodes,
            "node_1": row.node_1,
            "node_2": row.node_2,
            "node_3": row.node_3,
            "superpattern": row.superpattern,
            "superpattern_letter": getattr(row, "superpattern_letter", row.superpattern),
            "canonical_motif_id": row.canonical_motif_id,
            "dominant_eigenvalue_real": float(value.real),
            "dominant_eigenvalue_imag": float(value.imag),
            "dominant_eigenvalue_abs": float(abs(value)),
            "signed_loading_1": float(signed[0]),
            "signed_loading_2": float(signed[1]),
            "signed_loading_3": float(signed[2]),
            "dominant_signed_node": node_names[int(np.argmax(np.abs(signed)))],
            "dominant_signed_loading": float(signed[int(np.argmax(np.abs(signed)))]),
        }
        records.append(record)
    result = pd.DataFrame(records)
    if acronym_map is not None:
        result = add_acronym_columns(result, acronym_map)
        result["dominant_signed_node_acronym"] = (
            result["dominant_signed_node"].map(acronym_map).fillna(result["dominant_signed_node"])
        )
    return result


def build_signed_feature_matrix(mode_table, cell_types, eigenvalue_power=0.5):
    cell_lookup = {cell_type: index for index, cell_type in enumerate(cell_types)}
    rows = []
    cols = []
    values = []
    binary_values = []
    for row_position, row in enumerate(mode_table.itertuples(index=False)):
        scale = float(getattr(row, "dominant_eigenvalue_abs")) ** eigenvalue_power
        for node_slot in [1, 2, 3]:
            cell_type = getattr(row, f"node_{node_slot}")
            signed_loading = getattr(row, f"signed_loading_{node_slot}")
            rows.append(row_position)
            cols.append(cell_lookup[cell_type])
            values.append(scale * signed_loading)
            binary_values.append(1.0)
    shape = (len(mode_table), len(cell_types))
    signed_features = sparse.csr_matrix((values, (rows, cols)), shape=shape)
    membership = sparse.csr_matrix((binary_values, (rows, cols)), shape=shape)
    return signed_features, membership


def build_signed_jaccard_operator(mode_table, eigenvalue_power=0.5):
    cell_types = pd.Index(
        pd.concat(
            [mode_table["node_1"], mode_table["node_2"], mode_table["node_3"]],
            ignore_index=True,
        ).unique()
    ).sort_values().to_list()
    signed_features, membership = build_signed_feature_matrix(
        mode_table, cell_types, eigenvalue_power=eigenvalue_power
    )
    signed_dot = (signed_features @ signed_features.T).tocsr()
    overlap = (membership @ membership.T).tocsr()
    jaccard = overlap.copy()
    jaccard.data = jaccard.data / (6.0 - jaccard.data)
    operator = signed_dot.multiply(jaccard).tocsr()
    operator = (operator + operator.T) * 0.5
    return operator, signed_features, membership, cell_types


def decompose_signed_operator(operator, mode_table, n_modes=12):
    n_modes = min(n_modes, operator.shape[0] - 1)
    values, vectors = eigsh(operator, k=n_modes, which="LM")
    order = np.argsort(np.abs(values))[::-1]
    values = values[order]
    vectors = vectors[:, order]
    mode_records = []
    motif_records = []
    metadata = mode_table.reset_index(drop=True)
    for mode_position, eigenvalue in enumerate(values):
        mode = mode_position + 1
        vector = vectors[:, mode_position]
        top_position = int(np.argmax(np.abs(vector)))
        top = metadata.iloc[top_position]
        mode_records.append(
            {
                "mode": mode,
                "operator_eigenvalue": float(eigenvalue),
                "operator_eigenvalue_abs": float(abs(eigenvalue)),
                "positive_motif_count": int(np.sum(vector > 0)),
                "negative_motif_count": int(np.sum(vector < 0)),
                "top_motif_index": int(top.motif_index),
                "top_motif_acronym": getattr(top, "motif_acronym", top.nodes),
                "top_superpattern": top.superpattern,
                "top_loading": float(vector[top_position]),
            }
        )
        for rank, row_position in enumerate(
            np.argsort(np.abs(vector))[::-1][:50], start=1
        ):
            motif = metadata.iloc[int(row_position)]
            motif_records.append(
                {
                    "mode": mode,
                    "rank": rank,
                    "motif_index": int(motif.motif_index),
                    "motif_acronym": getattr(motif, "motif_acronym", motif.nodes),
                    "nodes": motif.nodes,
                    "superpattern": motif.superpattern,
                    "dominant_eigenvalue_abs": float(motif.dominant_eigenvalue_abs),
                    "mode_loading": float(vector[row_position]),
                    "abs_mode_loading": float(abs(vector[row_position])),
                }
            )
    return {
        "eigenvalues": values,
        "eigenvectors": vectors,
        "modes": pd.DataFrame(mode_records),
        "motif_participation": pd.DataFrame(motif_records),
    }


def summarize_mode_cells(motif_participation, mode_table, top_n=50):
    rows = []
    indexed = mode_table.set_index("motif_index")
    for mode, frame in motif_participation.groupby("mode"):
        top = frame.head(top_n)
        for row in top.itertuples(index=False):
            motif = indexed.loc[row.motif_index]
            for node_slot in [1, 2, 3]:
                rows.append(
                    {
                        "mode": mode,
                        "cell_type": motif[f"node_{node_slot}"],
                        "cell_type_acronym": motif.get(
                            f"node_{node_slot}_acronym", motif[f"node_{node_slot}"]
                        ),
                        "signed_contribution": row.mode_loading
                        * motif[f"signed_loading_{node_slot}"],
                        "abs_mode_loading": row.abs_mode_loading,
                    }
                )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return (
        summary.groupby(["mode", "cell_type", "cell_type_acronym"], as_index=False)
        .agg(
            total_signed_contribution=("signed_contribution", "sum"),
            total_abs_contribution=("signed_contribution", lambda s: float(np.abs(s).sum())),
            motif_mentions=("cell_type", "size"),
        )
        .sort_values(["mode", "total_abs_contribution"], ascending=[True, False])
        .reset_index(drop=True)
    )


def _normalized_adjacency_from_operator(operator, use_absolute=False):
    adjacency = abs(operator) if use_absolute else operator
    degree_values = np.asarray(abs(adjacency).sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degree_values, dtype=float)
    nonzero = degree_values > 0
    inv_sqrt[nonzero] = 1.0 / np.sqrt(degree_values[nonzero])
    normalizer = sparse.diags(inv_sqrt, format="csr")
    normalized_adjacency = normalizer @ adjacency @ normalizer
    return normalized_adjacency.tocsr(), degree_values


def build_laplacian(operator, kind="signed_normalized"):
    kind = kind.lower()
    if kind == "absolute_normalized":
        normalized_adjacency, degree_values = _normalized_adjacency_from_operator(
            operator, use_absolute=True
        )
        laplacian = sparse.eye(operator.shape[0], format="csr") - normalized_adjacency
    elif kind == "signed_normalized":
        normalized_adjacency, degree_values = _normalized_adjacency_from_operator(
            operator, use_absolute=False
        )
        laplacian = sparse.eye(operator.shape[0], format="csr") - normalized_adjacency
    elif kind == "signed_combinatorial":
        degree_values = np.asarray(abs(operator).sum(axis=1)).ravel()
        laplacian = sparse.diags(degree_values, format="csr") - operator
    else:
        raise ValueError(
            "kind must be absolute_normalized, signed_normalized, or signed_combinatorial"
        )
    return laplacian.tocsr(), degree_values


def decompose_laplacian(operator, mode_table, kind="signed_normalized", n_modes=12):
    laplacian, degree_values = build_laplacian(operator, kind=kind)
    n_modes = min(n_modes, laplacian.shape[0] - 1)
    values, vectors = eigsh(laplacian, k=n_modes, which="SA")
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    metadata = mode_table.reset_index(drop=True)
    mode_records = []
    motif_records = []

    for mode_position, eigenvalue in enumerate(values):
        mode = mode_position + 1
        vector = vectors[:, mode_position]
        top_position = int(np.argmax(np.abs(vector)))
        top = metadata.iloc[top_position]
        mode_records.append(
            {
                "laplacian_kind": kind,
                "mode": mode,
                "laplacian_eigenvalue": float(eigenvalue),
                "top_motif_index": int(top.motif_index),
                "top_motif_acronym": getattr(top, "motif_acronym", top.nodes),
                "top_superpattern": top.superpattern,
                "top_loading": float(vector[top_position]),
                "positive_motif_count": int(np.sum(vector > 0)),
                "negative_motif_count": int(np.sum(vector < 0)),
            }
        )
        for rank, row_position in enumerate(
            np.argsort(np.abs(vector))[::-1][:50], start=1
        ):
            motif = metadata.iloc[int(row_position)]
            motif_records.append(
                {
                    "laplacian_kind": kind,
                    "mode": mode,
                    "rank": rank,
                    "motif_index": int(motif.motif_index),
                    "motif_acronym": getattr(motif, "motif_acronym", motif.nodes),
                    "nodes": motif.nodes,
                    "superpattern": motif.superpattern,
                    "dominant_eigenvalue_abs": float(motif.dominant_eigenvalue_abs),
                    "laplacian_loading": float(vector[row_position]),
                    "abs_laplacian_loading": float(abs(vector[row_position])),
                    "operator_abs_degree": float(degree_values[row_position]),
                }
            )
    return {
        "laplacian": laplacian,
        "degree_values": degree_values,
        "eigenvalues": values,
        "eigenvectors": vectors,
        "modes": pd.DataFrame(mode_records),
        "motif_participation": pd.DataFrame(motif_records),
    }


def export_laplacian_results(output_dir, prefix, kind, result, cell_summary):
    output_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(output_dir / f"{prefix}_{kind}_laplacian.npz", result["laplacian"])
    pd.DataFrame({"operator_abs_degree": result["degree_values"]}).to_csv(
        output_dir / f"{prefix}_{kind}_laplacian_degrees.csv", index_label="row_position"
    )
    result["modes"].to_csv(output_dir / f"{prefix}_{kind}_laplacian_modes.csv", index=False)
    result["motif_participation"].to_csv(
        output_dir / f"{prefix}_{kind}_laplacian_motif_participation.csv", index=False
    )
    cell_summary.to_csv(
        output_dir / f"{prefix}_{kind}_laplacian_cell_summary.csv", index=False
    )


def export_signed_overlap_results(output_dir, prefix, mode_table, operator, decomposition, cell_summary):
    output_dir.mkdir(parents=True, exist_ok=True)
    mode_table.to_csv(output_dir / f"{prefix}_signed_dominant_mode_table.csv", index=False)
    sparse.save_npz(output_dir / f"{prefix}_signed_jaccard_operator.npz", operator)
    decomposition["modes"].to_csv(output_dir / f"{prefix}_signed_overlap_modes.csv", index=False)
    decomposition["motif_participation"].to_csv(
        output_dir / f"{prefix}_signed_overlap_motif_participation.csv", index=False
    )
    cell_summary.to_csv(output_dir / f"{prefix}_signed_overlap_cell_summary.csv", index=False)
