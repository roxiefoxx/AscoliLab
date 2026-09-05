import numpy as np
import pandas as pd
from scipy.linalg import schur


MATRIX_COLUMNS = [f"w_{i}{j}" for i in range(1, 4) for j in range(1, 4)]
LAYER_TOKENS = {"LI", "LII", "LIII", "LIV", "LV", "LVI", "I", "II", "III", "IV", "V", "VI"}
DESCRIPTOR_SHORT_NAMES = {
    "Axo": "Ax",
    "AIPRIM": "AIPRIM",
    "Back": "Back",
    "Basket": "Bas",
    "Bistratified": "Bi",
    "CA1": "CA1",
    "CA2": "CA2",
    "CA3": "CA3",
    "CA3c": "CA3c",
    "Cajal": "Caj",
    "Complex": "Cx",
    "Conjunctive": "Conj",
    "EC": "EC",
    "Giant": "G",
    "Granule": "Gr",
    "HICAP": "HICAP",
    "HIPROM": "HIPROM",
    "Interneuron": "IN",
    "Ivy": "Ivy",
    "LEC": "LEC",
    "MEC": "MEC",
    "MOLAX": "MOLAX",
    "Mossy": "Mossy",
    "Multipolar": "M",
    "Neurogliaform": "NGF",
    "OLM": "OLM",
    "Pyramidal": "P",
    "Principal": "Pr",
    "Projection": "Proj",
    "Radiatum": "Rad",
    "Schaffer": "Sch",
    "Semilunar": "Semi",
    "Stellate": "St",
    "Trilaminar": "TriLam",
    "Tripolar": "T",
}


def load_schur_blocks(csv_path, npz_path):
    """Load motif blocks exactly as stored; callers choose any orientation transpose."""
    metadata = pd.read_csv(csv_path)
    arrays = np.load(npz_path, allow_pickle=True)
    blocks = np.asarray(arrays["analysis_blocks"], dtype=float)
    if blocks.shape != (len(metadata), 3, 3):
        raise ValueError(
            f"Expected {(len(metadata), 3, 3)} analysis blocks, got {blocks.shape}."
        )
    return metadata, blocks


def _base_acronym(cell_type):
    text = str(cell_type)
    layer_replacements = {
        "LI II": "LI/II",
        "LII III": "LII/III",
        "LIV VI": "LIV/VI",
        "LV VI": "LV/VI",
    }
    for old, new in layer_replacements.items():
        text = text.replace(old, new)
    tokens = text.split()
    region = []
    layers = []
    descriptors = []
    for token in tokens:
        if token in {"DG", "EC", "MEC", "LEC"} or token.startswith("CA"):
            region.append(token)
        elif token in LAYER_TOKENS or "/" in token:
            layers.append(token)
        else:
            descriptors.append(DESCRIPTOR_SHORT_NAMES.get(token, token[:3]))
    pieces = []
    if region:
        pieces.append("-".join(region))
    if layers:
        pieces.append("/".join(layers))
    if descriptors:
        pieces.append("".join(descriptors[:3]))
    return "-".join(pieces) if pieces else str(cell_type)[:8]


def make_acronym_map(cell_types):
    unique_names = pd.Index(pd.Series(cell_types, dtype=str).dropna().unique()).sort_values()
    base_counts = {}
    acronym_map = {}
    for name in unique_names:
        base = _base_acronym(name)
        base_counts[base] = base_counts.get(base, 0) + 1
        acronym = base if base_counts[base] == 1 else f"{base}.{base_counts[base]}"
        acronym_map[name] = acronym
    return acronym_map


def acronym_table(acronym_map):
    return (
        pd.DataFrame(
            {"cell_type": list(acronym_map.keys()), "cell_type_acronym": list(acronym_map.values())}
        )
        .sort_values("cell_type_acronym")
        .reset_index(drop=True)
    )


def add_acronym_columns(frame, acronym_map):
    result = frame.copy()
    for column in ["node_1", "node_2", "node_3"]:
        if column in result:
            result[f"{column}_acronym"] = result[column].map(acronym_map).fillna(result[column])
    if {"node_1_acronym", "node_2_acronym", "node_3_acronym"}.issubset(result.columns):
        result["motif_acronym"] = (
            result["node_1_acronym"]
            + " | "
            + result["node_2_acronym"]
            + " | "
            + result["node_3_acronym"]
        )
    if "dominant_schur_node" in result:
        result["dominant_schur_node_acronym"] = (
            result["dominant_schur_node"].map(acronym_map).fillna(result["dominant_schur_node"])
        )
    return result


def _dominant_schur_participation(vectors, dominant_index):
    weights = np.abs(vectors[:, dominant_index]) ** 2
    total = weights.sum()
    return weights / total if total else weights


def decompose_blocks(metadata, blocks, scenario, acronym_map=None):
    records = []
    for index, block in enumerate(blocks):
        T, Z = schur(block, output="complex")
        eigenvalues = np.diag(T)
        magnitudes = np.abs(eigenvalues)
        dominant_index = int(np.argmax(magnitudes))
        dominant_value = eigenvalues[dominant_index]
        offdiag = T.copy()
        np.fill_diagonal(offdiag, 0.0)
        participation = _dominant_schur_participation(Z, dominant_index)
        row = metadata.iloc[index]
        sorted_magnitudes = np.sort(magnitudes)
        eigengap = (
            float(sorted_magnitudes[-1] - sorted_magnitudes[-2])
            if len(sorted_magnitudes) > 1
            else float(sorted_magnitudes[-1])
        )
        records.append(
            {
                "scenario": scenario,
                "motif_index": int(row.motif_index),
                "nodes": row.nodes,
                "node_1": row.node_1,
                "node_2": row.node_2,
                "node_3": row.node_3,
                "superpattern": row.superpattern,
                "superpattern_letter": getattr(row, "superpattern_letter", row.superpattern),
                "canonical_motif_id": row.canonical_motif_id,
                "observed_motif_code": row.observed_motif_code,
                "spectral_radius": float(magnitudes[dominant_index]),
                "dominant_eigenvalue_real": float(dominant_value.real),
                "dominant_eigenvalue_imag": float(dominant_value.imag),
                "eigengap": eigengap,
                "schur_offdiag_norm": float(np.linalg.norm(offdiag, "fro")),
                "block_frobenius_norm": float(np.linalg.norm(block, "fro")),
                "schur_departure_ratio": float(
                    np.linalg.norm(offdiag, "fro")
                    / max(np.linalg.norm(block, "fro"), np.finfo(float).eps)
                ),
                "dominant_schur_node": row[
                    f"node_{int(np.argmax(participation)) + 1}"
                ],
                "dominant_schur_participation": float(participation.max()),
                "node_1_schur_participation": float(participation[0]),
                "node_2_schur_participation": float(participation[1]),
                "node_3_schur_participation": float(participation[2]),
                "schur_eigenvalues": ", ".join(
                    f"{value.real:.4g}{value.imag:+.4g}j" for value in eigenvalues
                ),
                "schur_upper_t12_abs": float(abs(T[0, 1])),
                "schur_upper_t13_abs": float(abs(T[0, 2])),
                "schur_upper_t23_abs": float(abs(T[1, 2])),
            }
        )
    result = pd.DataFrame(records).sort_values(
        ["spectral_radius", "schur_departure_ratio"], ascending=False
    )
    if acronym_map is not None:
        result = add_acronym_columns(result, acronym_map)
    return result


def eigenspectrum_points(results):
    rows = []
    for row in results.itertuples(index=False):
        for text in str(row.schur_eigenvalues).split(", "):
            value = complex(text)
            real = float(value.real)
            imag = float(value.imag)
            rows.append(
                {
                    "scenario": row.scenario,
                    "motif_index": row.motif_index,
                    "motif_acronym": getattr(row, "motif_acronym", row.nodes),
                    "superpattern": row.superpattern,
                    "eigenvalue_real": real,
                    "eigenvalue_imag": imag,
                    "eigenvalue_abs": float(np.hypot(real, imag)),
                }
            )
    return pd.DataFrame(rows)


def enumerate_schur_modes(metadata, blocks, motif_index, acronym_map=None):
    matching = metadata.index[metadata["motif_index"] == motif_index]
    if len(matching) == 0:
        raise KeyError(f"motif_index {motif_index} was not found.")
    row_position = int(matching[0])
    row = metadata.iloc[row_position]
    block = blocks[row_position]
    T, Z = schur(block, output="complex")
    eigenvalues = np.diag(T)
    node_names = [row.node_1, row.node_2, row.node_3]
    node_labels = [
        acronym_map.get(name, name) if acronym_map is not None else name for name in node_names
    ]
    records = []
    for mode_index, eigenvalue in enumerate(eigenvalues):
        participation = np.abs(Z[:, mode_index]) ** 2
        if participation.sum():
            participation = participation / participation.sum()
        dominant_index = int(np.argmax(participation))
        records.append(
            {
                "mode": mode_index + 1,
                "eigenvalue_real": float(eigenvalue.real),
                "eigenvalue_imag": float(eigenvalue.imag),
                "eigenvalue_abs": float(abs(eigenvalue)),
                "dominant_node": node_names[dominant_index],
                "dominant_node_acronym": node_labels[dominant_index],
                f"{node_labels[0]}_participation": float(participation[0]),
                f"{node_labels[1]}_participation": float(participation[1]),
                f"{node_labels[2]}_participation": float(participation[2]),
                "upper_triangular_coupling_out_abs": float(
                    np.abs(T[mode_index, mode_index + 1 :]).sum()
                ),
            }
        )
    triangular = pd.DataFrame(
        T,
        index=[f"mode_{i + 1}" for i in range(3)],
        columns=[f"mode_{i + 1}" for i in range(3)],
    )
    block_frame = pd.DataFrame(block, index=node_labels, columns=node_labels)
    return pd.Series(row), block_frame, pd.DataFrame(records), triangular


def rank_all_schur_modes(metadata, blocks, scenario, acronym_map=None):
    records = []
    for row_position, block in enumerate(blocks):
        T, Z = schur(block, output="complex")
        eigenvalues = np.diag(T)
        row = metadata.iloc[row_position]
        node_names = [row.node_1, row.node_2, row.node_3]
        node_labels = [
            acronym_map.get(name, name) if acronym_map is not None else name
            for name in node_names
        ]
        motif_acronym = " | ".join(node_labels)
        for mode_index, eigenvalue in enumerate(eigenvalues):
            participation = np.abs(Z[:, mode_index]) ** 2
            if participation.sum():
                participation = participation / participation.sum()
            dominant_index = int(np.argmax(participation))
            records.append(
                {
                    "scenario": scenario,
                    "motif_index": int(row.motif_index),
                    "motif_acronym": motif_acronym,
                    "nodes": row.nodes,
                    "superpattern": row.superpattern,
                    "superpattern_letter": getattr(
                        row, "superpattern_letter", row.superpattern
                    ),
                    "schur_mode": mode_index + 1,
                    "eigenvalue_real": float(eigenvalue.real),
                    "eigenvalue_imag": float(eigenvalue.imag),
                    "eigenvalue_abs": float(abs(eigenvalue)),
                    "dominant_node": node_names[dominant_index],
                    "dominant_node_acronym": node_labels[dominant_index],
                    "dominant_node_participation": float(participation[dominant_index]),
                    "node_1_participation": float(participation[0]),
                    "node_2_participation": float(participation[1]),
                    "node_3_participation": float(participation[2]),
                    "schur_upper_coupling_out_abs": float(
                        np.abs(T[mode_index, mode_index + 1 :]).sum()
                    ),
                }
            )
    result = pd.DataFrame(records).sort_values(
        ["eigenvalue_abs", "dominant_node_participation"],
        ascending=False,
    )
    result.insert(0, "global_schur_mode_rank", np.arange(1, len(result) + 1))
    return result.reset_index(drop=True)


def _local_behavior_metrics(block):
    eigenvalues = np.linalg.eigvals(block)
    magnitudes = np.abs(eigenvalues)
    dominant_index = int(np.argmax(magnitudes))
    dominant = eigenvalues[dominant_index]
    offdiag = block.copy()
    np.fill_diagonal(offdiag, 0.0)
    identity = np.eye(block.shape[0])
    try:
        resolvent_gain = float(np.linalg.norm(np.linalg.inv(identity - block), 2))
        resolvent_condition = float(np.linalg.cond(identity - block))
    except np.linalg.LinAlgError:
        resolvent_gain = np.inf
        resolvent_condition = np.inf
    return {
        "spectral_radius": float(magnitudes[dominant_index]),
        "dominant_eigenvalue_real": float(dominant.real),
        "dominant_eigenvalue_imag": float(dominant.imag),
        "block_frobenius_norm": float(np.linalg.norm(block, "fro")),
        "offdiag_frobenius_norm": float(np.linalg.norm(offdiag, "fro")),
        "resolvent_gain_z1": resolvent_gain,
        "resolvent_condition_z1": resolvent_condition,
    }


def enhancement_ablation_test(metadata, blocks, scenario, activation_gain=0.5, acronym_map=None):
    records = []
    for row_position, baseline in enumerate(blocks):
        row = metadata.iloc[row_position]
        diagonal = np.diag(np.diag(baseline))
        offdiag = baseline - diagonal
        enhanced = diagonal + (1.0 + activation_gain) * offdiag
        ablated = diagonal.copy()
        baseline_metrics = _local_behavior_metrics(baseline)
        enhanced_metrics = _local_behavior_metrics(enhanced)
        ablated_metrics = _local_behavior_metrics(ablated)
        node_names = [row.node_1, row.node_2, row.node_3]
        node_labels = [
            acronym_map.get(name, name) if acronym_map is not None else name
            for name in node_names
        ]
        record = {
            "scenario": scenario,
            "motif_index": int(row.motif_index),
            "nodes": row.nodes,
            "motif_acronym": " | ".join(node_labels),
            "node_1": row.node_1,
            "node_2": row.node_2,
            "node_3": row.node_3,
            "superpattern": row.superpattern,
            "superpattern_letter": getattr(row, "superpattern_letter", row.superpattern),
            "activation_gain": float(activation_gain),
            "self_weights_preserved_in_ablation": bool(np.any(np.diag(baseline) != 0)),
        }
        for prefix, metrics in [
            ("baseline", baseline_metrics),
            ("enhanced", enhanced_metrics),
            ("ablated", ablated_metrics),
        ]:
            for key, value in metrics.items():
                record[f"{prefix}_{key}"] = value
        record["enhancement_delta_spectral_radius"] = (
            enhanced_metrics["spectral_radius"] - baseline_metrics["spectral_radius"]
        )
        record["ablation_delta_spectral_radius"] = (
            ablated_metrics["spectral_radius"] - baseline_metrics["spectral_radius"]
        )
        record["enhancement_delta_resolvent_gain_z1"] = (
            enhanced_metrics["resolvent_gain_z1"] - baseline_metrics["resolvent_gain_z1"]
        )
        record["ablation_delta_resolvent_gain_z1"] = (
            ablated_metrics["resolvent_gain_z1"] - baseline_metrics["resolvent_gain_z1"]
        )
        record["activation_sensitivity_score"] = abs(
            record["enhancement_delta_spectral_radius"]
        ) + abs(record["enhancement_delta_resolvent_gain_z1"])
        record["removal_sensitivity_score"] = abs(
            record["ablation_delta_spectral_radius"]
        ) + abs(record["ablation_delta_resolvent_gain_z1"])
        records.append(record)
    return (
        pd.DataFrame(records)
        .sort_values(
            ["activation_sensitivity_score", "removal_sensitivity_score"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def summarize_by_superpattern(results):
    return (
        results.groupby(
            ["scenario", "superpattern", "superpattern_letter", "canonical_motif_id"],
            as_index=False,
        )
        .agg(
            occurrence_count=("motif_index", "size"),
            mean_spectral_radius=("spectral_radius", "mean"),
            max_spectral_radius=("spectral_radius", "max"),
            mean_schur_offdiag_norm=("schur_offdiag_norm", "mean"),
            max_schur_offdiag_norm=("schur_offdiag_norm", "max"),
            mean_departure_ratio=("schur_departure_ratio", "mean"),
            max_departure_ratio=("schur_departure_ratio", "max"),
            mean_dominant_participation=("dominant_schur_participation", "mean"),
        )
        .sort_values(["scenario", "mean_spectral_radius"], ascending=[True, False])
        .reset_index(drop=True)
    )


def compare_self_scenarios(no_self, with_self):
    keys = [
        "motif_index",
        "nodes",
        "superpattern",
        "superpattern_letter",
        "canonical_motif_id",
    ]
    comparison = no_self.merge(with_self, on=keys, suffixes=("_without_self", "_with_self"))
    comparison["spectral_radius_change_from_self"] = (
        comparison.spectral_radius_with_self - comparison.spectral_radius_without_self
    )
    comparison["schur_offdiag_change_from_self"] = (
        comparison.schur_offdiag_norm_with_self
        - comparison.schur_offdiag_norm_without_self
    )
    comparison["departure_ratio_change_from_self"] = (
        comparison.schur_departure_ratio_with_self
        - comparison.schur_departure_ratio_without_self
    )
    return comparison.sort_values(
        "spectral_radius_change_from_self", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)


def summarize_cells(results, top_n=500):
    top = results.nlargest(top_n, "spectral_radius")
    counts = pd.concat(
        [top["node_1"], top["node_2"], top["node_3"]], ignore_index=True
    ).value_counts()
    dominant = top["dominant_schur_node"].value_counts()
    return (
        pd.DataFrame({"top_motif_mentions": counts, "dominant_schur_mentions": dominant})
        .fillna(0)
        .astype(int)
        .sort_values(["dominant_schur_mentions", "top_motif_mentions"], ascending=False)
        .rename_axis("cell_type")
        .reset_index()
    )


def build_motif_index_matrix(results, weight_column="spectral_radius", normalize_rows=False):
    cell_types = pd.Index(
        pd.concat(
            [results["node_1"], results["node_2"], results["node_3"]],
            ignore_index=True,
        ).unique()
    ).sort_values()
    cell_lookup = {cell_type: index for index, cell_type in enumerate(cell_types)}
    matrix = np.zeros((len(results), len(cell_types)), dtype=float)
    weights = results[weight_column].to_numpy(dtype=float)
    for row_index, row in enumerate(results.itertuples(index=False)):
        for cell_type in [row.node_1, row.node_2, row.node_3]:
            matrix[row_index, cell_lookup[cell_type]] = weights[row_index]
    if normalize_rows:
        norms = np.linalg.norm(matrix, axis=1)
        nonzero = norms > 0
        matrix[nonzero] = matrix[nonzero] / norms[nonzero, None]
    motif_index = results["motif_index"].to_numpy(dtype=int)
    return matrix, motif_index, cell_types.to_list()


def decompose_motif_index_matrix(
    results,
    acronym_map=None,
    weight_column="spectral_radius",
    normalize_rows=False,
    n_modes=12,
):
    matrix, motif_index, cell_types = build_motif_index_matrix(
        results, weight_column=weight_column, normalize_rows=normalize_rows
    )
    U, singular_values, Vt = np.linalg.svd(matrix, full_matrices=False)
    n_modes = min(n_modes, len(singular_values))
    total_energy = float(np.sum(singular_values**2))
    mode_records = []
    motif_records = []
    cell_records = []
    metadata = results.reset_index(drop=True)

    for mode_position in range(n_modes):
        mode = mode_position + 1
        singular_value = singular_values[mode_position]
        explained = (
            float((singular_value**2) / total_energy) if total_energy else np.nan
        )
        motif_scores = U[:, mode_position] * singular_value
        cell_loadings = Vt[mode_position, :]
        top_motif_position = int(np.argmax(np.abs(motif_scores)))
        top_cell_position = int(np.argmax(np.abs(cell_loadings)))
        top_motif = metadata.iloc[top_motif_position]
        top_cell = cell_types[top_cell_position]
        mode_records.append(
            {
                "mode": mode,
                "singular_value": float(singular_value),
                "explained_energy_fraction": explained,
                "top_motif_index": int(motif_index[top_motif_position]),
                "top_motif_acronym": getattr(top_motif, "motif_acronym", top_motif.nodes),
                "top_superpattern": top_motif.superpattern,
                "top_cell_type": top_cell,
                "top_cell_acronym": (
                    acronym_map.get(top_cell, top_cell) if acronym_map is not None else top_cell
                ),
                "top_motif_score": float(motif_scores[top_motif_position]),
                "top_cell_loading": float(cell_loadings[top_cell_position]),
            }
        )
        for rank, row_position in enumerate(
            np.argsort(np.abs(motif_scores))[::-1][:25], start=1
        ):
            motif = metadata.iloc[int(row_position)]
            motif_records.append(
                {
                    "mode": mode,
                    "rank": rank,
                    "motif_index": int(motif_index[row_position]),
                    "motif_acronym": getattr(motif, "motif_acronym", motif.nodes),
                    "nodes": motif.nodes,
                    "superpattern": motif.superpattern,
                    "motif_score": float(motif_scores[row_position]),
                    "abs_motif_score": float(abs(motif_scores[row_position])),
                    weight_column: float(getattr(motif, weight_column)),
                }
            )
        for rank, cell_position in enumerate(
            np.argsort(np.abs(cell_loadings))[::-1][:25], start=1
        ):
            cell_type = cell_types[int(cell_position)]
            cell_records.append(
                {
                    "mode": mode,
                    "rank": rank,
                    "cell_type": cell_type,
                    "cell_type_acronym": (
                        acronym_map.get(cell_type, cell_type)
                        if acronym_map is not None
                        else cell_type
                    ),
                    "cell_loading": float(cell_loadings[cell_position]),
                    "abs_cell_loading": float(abs(cell_loadings[cell_position])),
                }
            )

    matrix_frame = pd.DataFrame(
        matrix,
        index=pd.Index(motif_index, name="motif_index"),
        columns=cell_types,
    )
    return {
        "matrix": matrix_frame,
        "modes": pd.DataFrame(mode_records),
        "motif_participation": pd.DataFrame(motif_records),
        "cell_loadings": pd.DataFrame(cell_records),
        "singular_values": singular_values,
    }


def export_results(output_dir, no_self, with_self, summary, comparison, cells_no, cells_self):
    output_dir.mkdir(parents=True, exist_ok=True)
    no_self.to_csv(output_dir / "motif_schur_modes_without_self.csv", index=False)
    with_self.to_csv(output_dir / "motif_schur_modes_with_self.csv", index=False)
    summary.to_csv(output_dir / "motif_schur_superpattern_summary.csv", index=False)
    comparison.to_csv(output_dir / "motif_schur_self_weight_comparison.csv", index=False)
    cells_no.to_csv(output_dir / "motif_schur_top_cells_without_self.csv", index=False)
    cells_self.to_csv(output_dir / "motif_schur_top_cells_with_self.csv", index=False)
    return sorted(output_dir.iterdir())
