"""Shared helpers for the consolidated motif-analysis notebooks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from weighted_motif_null import TRIAD_CLASSES, CODE2CLASS, TriadIndex


@dataclass(frozen=True)
class ConnectomePaths:
    matrix: Path = Path("mij_matrix.csv")
    netlist: Path = Path("mij_netlist.csv")
    output_root: Path = Path("outputs")

    @property
    def census_dir(self) -> Path:
        return self.output_root / "consolidated_01_qc_census"

    @property
    def enrichment_dir(self) -> Path:
        return self.output_root / "consolidated_02_enrichment_chains"

    @property
    def dynamics_dir(self) -> Path:
        return self.output_root / "consolidated_03_dynamics"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_connectome(paths: ConnectomePaths | None = None) -> dict:
    paths = paths or ConnectomePaths()
    matrix = pd.read_csv(paths.matrix, index_col=0).astype(float)
    labels = list(matrix.index)
    if list(matrix.columns) != labels:
        raise ValueError("Matrix row and column labels must match in the same order.")

    netlist = pd.read_csv(paths.netlist)
    required = {"pre_neuron", "post_neuron", "pre_ei", "post_ei", "m_ij"}
    missing = sorted(required - set(netlist.columns))
    if missing:
        raise ValueError(f"Netlist is missing required columns: {missing}")

    index = {label: i for i, label in enumerate(labels)}
    missing_from_matrix = sorted((set(netlist.pre_neuron) | set(netlist.post_neuron)) - set(labels))
    if missing_from_matrix:
        raise ValueError(f"Netlist labels missing from matrix: {missing_from_matrix[:10]}")

    values = matrix.to_numpy()
    mismatches = []
    for row in netlist.itertuples(index=False):
        i, j = index[row.pre_neuron], index[row.post_neuron]
        if abs(values[i, j] - row.m_ij) > 1e-6 * max(1.0, abs(row.m_ij)):
            mismatches.append((row.pre_neuron, row.post_neuron, values[i, j], row.m_ij))
    if mismatches:
        raise ValueError(f"Matrix does not match netlist orientation [pre, post]; first mismatch: {mismatches[0]}")

    ei = {}
    for neuron_col, ei_col in (("pre_neuron", "pre_ei"), ("post_neuron", "post_ei")):
        for neuron, label in netlist[[neuron_col, ei_col]].dropna().drop_duplicates().itertuples(index=False):
            value = str(label).strip().upper()[0]
            if value not in {"E", "I"}:
                raise ValueError(f"Unexpected E/I label for {neuron}: {label!r}")
            if neuron in ei and ei[neuron] != value:
                raise ValueError(f"Conflicting E/I labels for {neuron}: {ei[neuron]} vs {value}")
            ei[neuron] = value

    missing_ei = [label for label in labels if label not in ei]
    if missing_ei:
        raise ValueError(f"Missing E/I labels for {missing_ei[:10]}")

    return {
        "paths": paths,
        "matrix_frame": matrix,
        "matrix": values,
        "weights": np.abs(values),
        "labels": labels,
        "netlist": netlist,
        "ei": np.array([ei[label] for label in labels]),
    }


def qc_summary(data: dict) -> pd.DataFrame:
    matrix = data["matrix"]
    offdiag = matrix.copy()
    np.fill_diagonal(offdiag, 0.0)
    netlist = data["netlist"]
    labels = data["labels"]
    ei = data["ei"]
    rows = [
        ("matrix nodes", len(labels)),
        ("netlist rows", len(netlist)),
        ("nonzero matrix entries", int(np.count_nonzero(matrix))),
        ("off-diagonal nonzero entries", int(np.count_nonzero(offdiag))),
        ("self-loop entries", int(np.count_nonzero(np.diag(matrix)))),
        ("excitatory labels", int((ei == "E").sum())),
        ("inhibitory labels", int((ei == "I").sum())),
        ("positive off-diagonal entries", int((offdiag > 0).sum())),
        ("negative off-diagonal entries", int((offdiag < 0).sum())),
        ("regions", len({str(label).split()[0] for label in labels})),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def _triad_bits(binary: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (
        binary[a, b].astype(np.int64)
        | (binary[b, a].astype(np.int64) << 1)
        | (binary[a, c].astype(np.int64) << 2)
        | (binary[c, a].astype(np.int64) << 3)
        | (binary[b, c].astype(np.int64) << 4)
        | (binary[c, b].astype(np.int64) << 5)
    )


def triad_census_tables(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = np.abs(data["matrix"]).copy()
    np.fill_diagonal(weights, 0.0)
    binary = weights > 0
    labels = data["labels"]
    ei = data["ei"]

    idx = TriadIndex(len(labels))
    classes = np.array(TRIAD_CLASSES)[CODE2CLASS[_triad_bits(binary, idx.a, idx.b, idx.c)]]
    census = (
        pd.Series(classes, name="triad_class")
        .value_counts()
        .reindex(TRIAD_CLASSES, fill_value=0)
        .rename_axis("triad_class")
        .reset_index(name="count")
    )
    census["share"] = census["count"] / census["count"].sum()

    participation = pd.DataFrame(0, index=labels, columns=TRIAD_CLASSES, dtype=int)
    for node in (idx.a, idx.b, idx.c):
        for class_name in TRIAD_CLASSES:
            np.add.at(
                participation[class_name].values,
                node[classes == class_name],
                1,
            )
    participation.insert(0, "ei", ei)
    participation.index.name = "cell_type"
    return census, participation.reset_index()


def run_qc_and_census(paths: ConnectomePaths | None = None) -> dict:
    data = load_connectome(paths)
    out = ensure_dir(data["paths"].census_dir)
    qc = qc_summary(data)
    census, participation = triad_census_tables(data)
    qc.to_csv(out / "qc_summary.csv", index=False)
    census.to_csv(out / "triad_census.csv", index=False)
    participation.to_csv(out / "node_triad_participation.csv", index=False)
    return {
        **data,
        "output_dir": out,
        "qc_summary": qc,
        "triad_census": census,
        "node_triad_participation": participation,
    }


def _candidate_paths(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def motif_pathway_input_dir() -> Path | None:
    return _candidate_paths(
        Path("outputs") / "motif_pathway_analysis",
        Path("..") / "tarjan_reorder" / "outputs" / "motif_pathway_analysis",
    )


def run_schur_mode_analysis(
    input_dir: Path | None = None,
    output_dir: Path | str = Path("outputs") / "consolidated_03_dynamics" / "schur",
    activation_gain: float = 0.5,
    transpose_source_target_blocks: bool = True,
) -> dict:
    try:
        import motif_schur_decomposition_analysis_script as analysis
    except ModuleNotFoundError as exc:
        if exc.name != "scipy":
            raise
        return _run_schur_mode_analysis_numpy(
            input_dir=input_dir,
            output_dir=output_dir,
            activation_gain=activation_gain,
            transpose_source_target_blocks=transpose_source_target_blocks,
        )

    input_dir = input_dir or motif_pathway_input_dir()
    if input_dir is None:
        raise FileNotFoundError("Could not find outputs/motif_pathway_analysis or ../tarjan_reorder/outputs/motif_pathway_analysis")

    output_dir = ensure_dir(Path(output_dir))
    no_self_csv = input_dir / "schur_ready_3x3_subnetworks_without_self.csv"
    no_self_npz = input_dir / "schur_ready_3x3_subnetworks_without_self.npz"
    with_self_csv = input_dir / "schur_ready_3x3_subnetworks_with_self.csv"
    with_self_npz = input_dir / "schur_ready_3x3_subnetworks_with_self.npz"

    meta_no, blocks_no = analysis.load_schur_blocks(no_self_csv, no_self_npz)
    meta_self, blocks_self = analysis.load_schur_blocks(with_self_csv, with_self_npz)
    if transpose_source_target_blocks:
        blocks_no = blocks_no.transpose(0, 2, 1)
        blocks_self = blocks_self.transpose(0, 2, 1)

    cell_types = pd.unique(pd.concat([meta_no[["node_1", "node_2", "node_3"]].stack(), meta_self[["node_1", "node_2", "node_3"]].stack()]))
    acronym_map = analysis.make_acronym_map(cell_types)
    acronyms = analysis.acronym_table(acronym_map)

    schur_no = analysis.decompose_blocks(meta_no, blocks_no, "without_self", acronym_map=acronym_map)
    schur_self = analysis.decompose_blocks(meta_self, blocks_self, "with_self", acronym_map=acronym_map)
    summary = pd.concat([analysis.summarize_by_superpattern(schur_no), analysis.summarize_by_superpattern(schur_self)], ignore_index=True)
    comparison = analysis.compare_self_scenarios(schur_no, schur_self)
    cells_no = analysis.summarize_cells(schur_no, top_n=500)
    cells_self = analysis.summarize_cells(schur_self, top_n=500)
    ranked_modes_no = analysis.rank_all_schur_modes(meta_no, blocks_no, "without_self", acronym_map=acronym_map)
    ranked_modes_self = analysis.rank_all_schur_modes(meta_self, blocks_self, "with_self", acronym_map=acronym_map)
    perturb_no = analysis.enhancement_ablation_test(meta_no, blocks_no, "without_self", activation_gain=activation_gain, acronym_map=acronym_map)
    perturb_self = analysis.enhancement_ablation_test(meta_self, blocks_self, "with_self", activation_gain=activation_gain, acronym_map=acronym_map)

    exported = analysis.export_results(output_dir, schur_no, schur_self, summary, comparison, cells_no, cells_self)
    extra_tables = {
        "cell_type_acronym_map": acronyms,
        "ranked_schur_modes_without_self": ranked_modes_no,
        "ranked_schur_modes_with_self": ranked_modes_self,
        "motif_enhancement_ablation_without_self": perturb_no,
        "motif_enhancement_ablation_with_self": perturb_self,
    }
    for name, table in extra_tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "schur_without_self": schur_no,
        "schur_with_self": schur_self,
        "superpattern_summary": summary,
        "self_weight_comparison": comparison,
        "cells_without_self": cells_no,
        "cells_with_self": cells_self,
        "ranked_modes_without_self": ranked_modes_no,
        "ranked_modes_with_self": ranked_modes_self,
        "enhancement_ablation_without_self": perturb_no,
        "enhancement_ablation_with_self": perturb_self,
        "exported": exported,
    }


def _load_motif_blocks(input_dir: Path, transpose_source_target_blocks: bool):
    no_self_csv = input_dir / "schur_ready_3x3_subnetworks_without_self.csv"
    no_self_npz = input_dir / "schur_ready_3x3_subnetworks_without_self.npz"
    with_self_csv = input_dir / "schur_ready_3x3_subnetworks_with_self.csv"
    with_self_npz = input_dir / "schur_ready_3x3_subnetworks_with_self.npz"
    meta_no = pd.read_csv(no_self_csv)
    meta_self = pd.read_csv(with_self_csv)
    blocks_no = np.asarray(np.load(no_self_npz, allow_pickle=True)["analysis_blocks"], dtype=float)
    blocks_self = np.asarray(np.load(with_self_npz, allow_pickle=True)["analysis_blocks"], dtype=float)
    if transpose_source_target_blocks:
        blocks_no = blocks_no.transpose(0, 2, 1)
        blocks_self = blocks_self.transpose(0, 2, 1)
    return meta_no, blocks_no, meta_self, blocks_self


def _henrici_departure(block: np.ndarray, values: np.ndarray) -> float:
    fro_sq = float(np.linalg.norm(block, "fro") ** 2)
    eig_sq = float(np.sum(np.abs(values) ** 2))
    return float(np.sqrt(max(fro_sq - eig_sq, 0.0)))


def _simple_acronym(cell_type: str) -> str:
    tokens = str(cell_type).replace("+", "pos").split()
    return "-".join(tokens[:3]) if tokens else "Unknown"


def _add_simple_acronyms(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for col in ["node_1", "node_2", "node_3"]:
        if col in frame:
            frame[f"{col}_acronym"] = frame[col].map(_simple_acronym)
    if {"node_1_acronym", "node_2_acronym", "node_3_acronym"} <= set(frame.columns):
        frame["motif_acronym"] = frame[["node_1_acronym", "node_2_acronym", "node_3_acronym"]].agg(" | ".join, axis=1)
    return frame


def _numpy_motif_modes(metadata: pd.DataFrame, blocks: np.ndarray, scenario: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    motif_rows = []
    mode_rows = []
    for pos, block in enumerate(blocks):
        row = metadata.iloc[pos]
        values = np.linalg.eigvals(block)
        order = sorted(range(len(values)), key=lambda i: (-abs(values[i]), -values[i].real, -values[i].imag))
        spectral_radius = float(np.max(np.abs(values)))
        departure = _henrici_departure(block, values)
        motif_rows.append(
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
                "spectral_radius": spectral_radius,
                "dominant_eigenvalue_real": float(values[order[0]].real),
                "dominant_eigenvalue_imag": float(values[order[0]].imag),
                "henrici_departure": departure,
                "departure_ratio": departure / spectral_radius if spectral_radius else np.nan,
            }
        )
        for rank, eig_idx in enumerate(order, start=1):
            value = values[eig_idx]
            mode_rows.append(
                {
                    "scenario": scenario,
                    "global_schur_mode_rank": None,
                    "motif_index": int(row.motif_index),
                    "nodes": row.nodes,
                    "node_1": row.node_1,
                    "node_2": row.node_2,
                    "node_3": row.node_3,
                    "superpattern": row.superpattern,
                    "schur_mode": rank,
                    "eigenvalue_real": float(value.real),
                    "eigenvalue_imag": float(value.imag),
                    "eigenvalue_abs": float(abs(value)),
                }
            )
    motifs = _add_simple_acronyms(pd.DataFrame(motif_rows))
    modes = _add_simple_acronyms(pd.DataFrame(mode_rows)).sort_values("eigenvalue_abs", ascending=False).reset_index(drop=True)
    modes["global_schur_mode_rank"] = np.arange(1, len(modes) + 1)
    return motifs, modes


def _numpy_enhancement_ablation(metadata: pd.DataFrame, blocks: np.ndarray, scenario: str, activation_gain: float) -> pd.DataFrame:
    rows = []
    for pos, block in enumerate(blocks):
        row = metadata.iloc[pos]
        offdiag = ~np.eye(block.shape[0], dtype=bool)
        activated = block.copy()
        activated[offdiag] *= 1 + activation_gain
        ablated = block.copy()
        ablated[offdiag] = 0.0
        baseline = float(np.max(np.abs(np.linalg.eigvals(block))))
        activation = float(np.max(np.abs(np.linalg.eigvals(activated))))
        ablation = float(np.max(np.abs(np.linalg.eigvals(ablated))))
        rows.append(
            {
                "scenario": scenario,
                "motif_index": int(row.motif_index),
                "nodes": row.nodes,
                "node_1": row.node_1,
                "node_2": row.node_2,
                "node_3": row.node_3,
                "superpattern": row.superpattern,
                "baseline_spectral_radius": baseline,
                "activation_spectral_radius": activation,
                "ablation_spectral_radius": ablation,
                "activation_delta": activation - baseline,
                "ablation_delta": ablation - baseline,
            }
        )
    return _add_simple_acronyms(pd.DataFrame(rows)).sort_values("activation_delta", ascending=False)


def _run_schur_mode_analysis_numpy(
    input_dir: Path | None,
    output_dir: Path | str,
    activation_gain: float,
    transpose_source_target_blocks: bool,
) -> dict:
    input_dir = input_dir or motif_pathway_input_dir()
    if input_dir is None:
        raise FileNotFoundError("Could not find motif-pathway analysis inputs.")
    output_dir = ensure_dir(Path(output_dir))
    meta_no, blocks_no, meta_self, blocks_self = _load_motif_blocks(input_dir, transpose_source_target_blocks)

    schur_no, ranked_modes_no = _numpy_motif_modes(meta_no, blocks_no, "without_self")
    schur_self, ranked_modes_self = _numpy_motif_modes(meta_self, blocks_self, "with_self")
    summary = (
        pd.concat([schur_no, schur_self], ignore_index=True)
        .groupby(["scenario", "superpattern"], as_index=False)
        .agg(
            motif_count=("motif_index", "size"),
            mean_spectral_radius=("spectral_radius", "mean"),
            max_spectral_radius=("spectral_radius", "max"),
            mean_henrici_departure=("henrici_departure", "mean"),
            mean_departure_ratio=("departure_ratio", "mean"),
        )
        .sort_values(["scenario", "mean_spectral_radius"], ascending=[True, False])
    )
    comparison = schur_no.merge(
        schur_self,
        on="motif_index",
        suffixes=("_without_self", "_with_self"),
    )
    comparison["spectral_radius_change_from_self"] = comparison["spectral_radius_with_self"] - comparison["spectral_radius_without_self"]
    cells_no = _cell_summary_from_motifs(schur_no)
    cells_self = _cell_summary_from_motifs(schur_self)
    perturb_no = _numpy_enhancement_ablation(meta_no, blocks_no, "without_self", activation_gain)
    perturb_self = _numpy_enhancement_ablation(meta_self, blocks_self, "with_self", activation_gain)

    tables = {
        "motif_schur_modes_without_self": schur_no,
        "motif_schur_modes_with_self": schur_self,
        "ranked_schur_modes_without_self": ranked_modes_no,
        "ranked_schur_modes_with_self": ranked_modes_self,
        "motif_schur_superpattern_summary": summary,
        "motif_schur_self_weight_comparison": comparison,
        "motif_schur_top_cells_without_self": cells_no,
        "motif_schur_top_cells_with_self": cells_self,
        "motif_enhancement_ablation_without_self": perturb_no,
        "motif_enhancement_ablation_with_self": perturb_self,
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "schur_without_self": schur_no,
        "schur_with_self": schur_self,
        "superpattern_summary": summary,
        "self_weight_comparison": comparison,
        "cells_without_self": cells_no,
        "cells_with_self": cells_self,
        "ranked_modes_without_self": ranked_modes_no,
        "ranked_modes_with_self": ranked_modes_self,
        "enhancement_ablation_without_self": perturb_no,
        "enhancement_ablation_with_self": perturb_self,
        "exported": [output_dir / f"{name}.csv" for name in tables],
        "method_note": "NumPy eigenvalue fallback used because scipy.linalg.schur is unavailable.",
    }


def _cell_summary_from_motifs(motifs: pd.DataFrame, top_n: int = 500) -> pd.DataFrame:
    frames = []
    for col in ["node_1", "node_2", "node_3"]:
        frames.append(motifs[["motif_index", col, "spectral_radius", "henrici_departure"]].rename(columns={col: "cell_type"}))
    long = pd.concat(frames, ignore_index=True)
    return (
        long.groupby("cell_type", as_index=False)
        .agg(
            motif_appearances=("motif_index", "size"),
            mean_spectral_radius=("spectral_radius", "mean"),
            max_spectral_radius=("spectral_radius", "max"),
            mean_henrici_departure=("henrici_departure", "mean"),
        )
        .sort_values(["max_spectral_radius", "motif_appearances"], ascending=False)
        .head(top_n)
    )


def run_signed_overlap_analysis(
    input_dir: Path | None = None,
    output_dir: Path | str = Path("outputs") / "consolidated_03_dynamics" / "signed_overlap",
    eigenvalue_power: float = 0.5,
    n_modes: int = 12,
    n_laplacian_modes: int = 12,
    transpose_source_target_blocks: bool = True,
) -> dict:
    import motif_schur_decomposition_analysis_script as schur_analysis
    import motif_signed_overlap_analysis_script as analysis

    input_dir = input_dir or motif_pathway_input_dir()
    if input_dir is None:
        raise FileNotFoundError("Could not find outputs/motif_pathway_analysis or ../tarjan_reorder/outputs/motif_pathway_analysis")

    output_dir = ensure_dir(Path(output_dir))
    no_self_csv = input_dir / "schur_ready_3x3_subnetworks_without_self.csv"
    no_self_npz = input_dir / "schur_ready_3x3_subnetworks_without_self.npz"
    with_self_csv = input_dir / "schur_ready_3x3_subnetworks_with_self.csv"
    with_self_npz = input_dir / "schur_ready_3x3_subnetworks_with_self.npz"
    meta_no, blocks_no = analysis.load_blocks_and_metadata(no_self_csv, no_self_npz)
    meta_self, blocks_self = analysis.load_blocks_and_metadata(with_self_csv, with_self_npz)
    if transpose_source_target_blocks:
        blocks_no = blocks_no.transpose(0, 2, 1)
        blocks_self = blocks_self.transpose(0, 2, 1)

    cell_types = pd.unique(pd.concat([meta_no[["node_1", "node_2", "node_3"]].stack(), meta_self[["node_1", "node_2", "node_3"]].stack()]))
    acronym_map = schur_analysis.make_acronym_map(cell_types)

    mode_table_no = analysis.signed_dominant_mode_table(meta_no, blocks_no, "without_self", acronym_map=acronym_map)
    mode_table_self = analysis.signed_dominant_mode_table(meta_self, blocks_self, "with_self", acronym_map=acronym_map)
    operator_no, _features_no, _membership_no, _cells_no = analysis.build_signed_jaccard_operator(mode_table_no, eigenvalue_power=eigenvalue_power)
    operator_self, _features_self, _membership_self, _cells_self = analysis.build_signed_jaccard_operator(mode_table_self, eigenvalue_power=eigenvalue_power)
    decomp_no = analysis.decompose_signed_operator(operator_no, mode_table_no, n_modes=n_modes)
    decomp_self = analysis.decompose_signed_operator(operator_self, mode_table_self, n_modes=n_modes)
    cell_summary_no = analysis.summarize_mode_cells(decomp_no["motif_participation"], mode_table_no)
    cell_summary_self = analysis.summarize_mode_cells(decomp_self["motif_participation"], mode_table_self)
    lap_no = analysis.decompose_laplacian(operator_no, mode_table_no, kind="signed_normalized", n_modes=n_laplacian_modes)
    lap_self = analysis.decompose_laplacian(operator_self, mode_table_self, kind="signed_normalized", n_modes=n_laplacian_modes)

    analysis.export_signed_overlap_results(output_dir, "without_self", mode_table_no, operator_no, decomp_no, cell_summary_no)
    analysis.export_signed_overlap_results(output_dir, "with_self", mode_table_self, operator_self, decomp_self, cell_summary_self)
    analysis.export_laplacian_results(output_dir, "without_self", "signed_normalized", lap_no, analysis.summarize_mode_cells(lap_no["motif_participation"], mode_table_no))
    analysis.export_laplacian_results(output_dir, "with_self", "signed_normalized", lap_self, analysis.summarize_mode_cells(lap_self["motif_participation"], mode_table_self))

    operator_summary = pd.DataFrame(
        [
            {"scenario": "without_self", "shape": operator_no.shape, "nonzero_entries": operator_no.nnz, "density": operator_no.nnz / np.prod(operator_no.shape), "min_value": operator_no.data.min(), "max_value": operator_no.data.max()},
            {"scenario": "with_self", "shape": operator_self.shape, "nonzero_entries": operator_self.nnz, "density": operator_self.nnz / np.prod(operator_self.shape), "min_value": operator_self.data.min(), "max_value": operator_self.data.max()},
        ]
    )
    operator_summary.to_csv(output_dir / "signed_jaccard_operator_summary.csv", index=False)

    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "operator_summary": operator_summary,
        "modes_without_self": decomp_no["modes"],
        "modes_with_self": decomp_self["modes"],
        "laplacian_modes_without_self": lap_no["modes"],
        "laplacian_modes_with_self": lap_self["modes"],
        "cell_summary_without_self": cell_summary_no,
        "cell_summary_with_self": cell_summary_self,
    }


def existing_dynamics_outputs() -> dict[str, Path | None]:
    return {
        "motif_pathway_inputs": motif_pathway_input_dir(),
        "schur_outputs": _candidate_paths(
            Path("outputs") / "motif_schur_decomposition_analysis",
            Path("..") / "tarjan_reorder" / "outputs" / "motif_schur_decomposition_analysis",
        ),
        "signed_overlap_outputs": _candidate_paths(
            Path("outputs") / "motif_signed_overlap_operator_analysis",
            Path("..") / "tarjan_reorder" / "outputs" / "motif_signed_overlap_operator_analysis",
        ),
    }


def add_parent_modules_to_path() -> None:
    parent = str((Path("..") / "tarjan_reorder").resolve())
    if parent not in sys.path:
        sys.path.insert(0, parent)
