"""Greedy information-flow trees for Schur-mode contributor subnetworks.

This script takes the top contributors to each Schur mode, extracts the small
directed M_ij subnetwork among those contributors, and prints a compact greedy
tree for each mode.

Matrix convention used here:
    mij_matrix.csv rows are sources and columns are receivers, so an entry
    M.loc[source, receiver] is interpreted as source -> receiver.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from schur_core_script import load_mij_data, load_mij_matrix, normalize_state_matrix, region_from_label


DEFAULT_MATRIX_PATHS = (Path("matrices/mij_matrix.csv"), Path("mij_matrix.csv"))
DEFAULT_NETLIST_PATHS = (Path("matrices/mij_netlist.csv"),)
DEFAULT_SCHUR_ARCHIVE = Path("outputs/schur_modes/schur_modes.npz")


@dataclass(frozen=True)
class SimpleSchurModel:
    A: np.ndarray
    T: np.ndarray
    Q: np.ndarray
    labels: tuple[str, ...]

    @property
    def eigenvalues(self) -> np.ndarray:
        return np.diag(self.T)

    @property
    def n_modes(self) -> int:
        return self.T.shape[0]


@dataclass
class ModeGreedyTree:
    mode: int
    eigenvalue: complex
    contributors: pd.DataFrame
    submatrix: pd.DataFrame
    netlist_edges: pd.DataFrame
    tree_edges: pd.DataFrame
    roots: list[str]


def first_existing(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return next(iter(paths))


def prepare_local_schur_model(
    matrix_path: str | Path,
    normalization: str = "spectral_radius",
    target_spectral_radius: float = 0.95,
    include_self: bool = True,
) -> SimpleSchurModel:
    """Load M_ij, orient it as receiver-by-source, normalize, and decompose."""
    from scipy import linalg

    frame = load_mij_matrix(matrix_path)
    if not include_self:
        frame = frame.copy()
        np.fill_diagonal(frame.values, 0.0)

    A_raw = frame.to_numpy(dtype=float).T
    method = normalization.lower().strip()
    if method == "spectral_radius":
        A, _ = normalize_state_matrix(A_raw, method=method, target=target_spectral_radius)
    elif method in {"column_l1", "col_l1"}:
        A, _ = normalize_state_matrix(A_raw, method="column_l1", target=target_spectral_radius)
    elif method == "none":
        A, _ = normalize_state_matrix(A_raw, method=method, target=target_spectral_radius)
    else:
        raise ValueError("normalization must be 'spectral_radius', 'column_l1', or 'none'.")

    T, Q = linalg.schur(A, output="complex")
    return SimpleSchurModel(A=A, T=T, Q=Q, labels=tuple(str(x) for x in frame.columns))


def load_schur_archive(archive_path: str | Path = DEFAULT_SCHUR_ARCHIVE) -> SimpleSchurModel:
    archive = Path(archive_path)
    with np.load(archive, allow_pickle=True) as data:
        A = data["A"]
        T = data["T"]
        Q = data["Q"]
        labels = tuple(str(x) for x in data["labels"])
    return SimpleSchurModel(A=A, T=T, Q=Q, labels=labels)


def format_eigenvalue(value: complex) -> str:
    sign = "+" if value.imag >= 0 else "-"
    return f"{value.real:.4g} {sign} {abs(value.imag):.4g}i |lambda|={abs(value):.4g}"


def contributor_table(model, mode: int, ei_by_cell: pd.Series, top_n: int) -> pd.DataFrame:
    loadings = model.Q[:, mode]
    order = np.argsort(np.abs(loadings))[::-1][:top_n]
    rows = []
    for rank, cell_index in enumerate(order, start=1):
        label = model.labels[cell_index]
        loading = loadings[cell_index]
        rows.append(
            {
                "rank": rank,
                "cell_type": label,
                "region": region_from_label(label),
                "ei": str(ei_by_cell.get(label, "")).upper(),
                "loading_real": float(loading.real),
                "loading_imag": float(loading.imag),
                "loading_magnitude": float(abs(loading)),
                "energy_fraction": float(abs(loading) ** 2),
            }
        )
    return pd.DataFrame(rows)


def subnetwork_netlist(netlist: pd.DataFrame | None, contributors: list[str]) -> pd.DataFrame:
    if netlist is None:
        return pd.DataFrame()
    keep = netlist[
        netlist["pre_neuron"].isin(contributors)
        & netlist["post_neuron"].isin(contributors)
        & (netlist["pre_neuron"] != netlist["post_neuron"])
    ].copy()
    if keep.empty:
        return keep
    sort_col = "m_ij" if "m_ij" in keep.columns else "w_ij"
    keep["_abs_weight"] = keep[sort_col].abs()
    keep = keep.sort_values("_abs_weight", ascending=False).drop(columns="_abs_weight")
    return keep.reset_index(drop=True)


def build_greedy_tree(submatrix: pd.DataFrame, contributors: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build a greedy directed tree over the contributor subnetwork.

    The root is the highest-loading contributor. At each step, choose the
    strongest absolute source->receiver edge from any reached node to any
    unreached node. If no outgoing edge can reach the remaining nodes, start a
    new component at the next-highest-loading unreached contributor.
    """
    ranked_cells = contributors["cell_type"].tolist()
    rank_by_cell = dict(zip(contributors["cell_type"], contributors["rank"]))
    loading_by_cell = dict(zip(contributors["cell_type"], contributors["loading_magnitude"]))
    reached: set[str] = set()
    unreached: set[str] = set(ranked_cells)
    roots: list[str] = []
    edges: list[dict[str, object]] = []
    component = 0

    while unreached:
        component += 1
        root = next(cell for cell in ranked_cells if cell in unreached)
        roots.append(root)
        reached.add(root)
        unreached.remove(root)

        while unreached:
            best: tuple[float, str, str, float] | None = None
            for source in reached:
                for receiver in unreached:
                    weight = float(submatrix.loc[source, receiver])
                    if weight == 0:
                        continue
                    score = abs(weight)
                    if best is None or score > best[0]:
                        best = (score, source, receiver, weight)

            if best is None:
                break

            score, source, receiver, weight = best
            edges.append(
                {
                    "component": component,
                    "source": source,
                    "receiver": receiver,
                    "m_ij": weight,
                    "abs_m_ij": score,
                    "source_rank": rank_by_cell[source],
                    "receiver_rank": rank_by_cell[receiver],
                    "source_loading_magnitude": loading_by_cell[source],
                    "receiver_loading_magnitude": loading_by_cell[receiver],
                }
            )
            reached.add(receiver)
            unreached.remove(receiver)

    tree = pd.DataFrame(edges)
    if tree.empty:
        return tree, roots

    depths: dict[str, int] = {root: 0 for root in roots}
    for row in tree.itertuples(index=False):
        depths[row.receiver] = depths.get(row.source, 0) + 1
    tree["depth"] = tree["receiver"].map(depths).fillna(1).astype(int)
    return tree, roots


def build_mode_greedy_tree(
    model,
    data,
    mode: int,
    top_n: int = 7,
) -> ModeGreedyTree:
    contributors = contributor_table(model, mode=mode, ei_by_cell=data.ei, top_n=top_n)
    cells = contributors["cell_type"].tolist()
    submatrix = data.df_source_receiver.loc[cells, cells].copy()
    net_edges = subnetwork_netlist(data.netlist, cells)
    tree_edges, roots = build_greedy_tree(submatrix, contributors)
    return ModeGreedyTree(
        mode=mode,
        eigenvalue=complex(model.eigenvalues[mode]),
        contributors=contributors,
        submatrix=submatrix,
        netlist_edges=net_edges,
        tree_edges=tree_edges,
        roots=roots,
    )


def build_all_mode_greedy_trees(
    matrix_path: str | Path | None = None,
    netlist_path: str | Path | None = None,
    top_n: int = 7,
    normalization: str = "spectral_radius",
    target_spectral_radius: float = 0.95,
    include_self: bool = True,
    max_modes: int | None = None,
    schur_archive_path: str | Path | None = DEFAULT_SCHUR_ARCHIVE,
    prefer_schur_archive: bool = True,
) -> list[ModeGreedyTree]:
    matrix = Path(matrix_path) if matrix_path is not None else first_existing(DEFAULT_MATRIX_PATHS)
    netlist = Path(netlist_path) if netlist_path is not None else first_existing(DEFAULT_NETLIST_PATHS)
    if not netlist.exists():
        netlist = None

    archive = Path(schur_archive_path) if schur_archive_path is not None else None
    if prefer_schur_archive and archive is not None and archive.exists() and include_self:
        model = load_schur_archive(archive)
    else:
        model = prepare_local_schur_model(
            matrix,
            normalization=normalization,
            target_spectral_radius=target_spectral_radius,
            include_self=include_self,
        )
    data = load_mij_data(matrix_path=matrix, netlist_path=netlist)
    mode_count = model.n_modes if max_modes is None else min(max_modes, model.n_modes)
    return [build_mode_greedy_tree(model, data, mode=mode, top_n=top_n) for mode in range(mode_count)]


def _children_by_source(tree_edges: pd.DataFrame) -> dict[str, list[pd.Series]]:
    children: dict[str, list[pd.Series]] = {}
    if tree_edges.empty:
        return children
    for _, row in tree_edges.sort_values(["component", "depth", "abs_m_ij"], ascending=[True, True, False]).iterrows():
        children.setdefault(row["source"], []).append(row)
    return children


def _print_branch(source: str, children: dict[str, list[pd.Series]], indent: int = 0) -> None:
    for edge in children.get(source, []):
        prefix = "  " * indent
        sign = "+" if edge["m_ij"] >= 0 else "-"
        print(
            f"{prefix}{edge['source']} -> {edge['receiver']}  "
            f"m_ij={edge['m_ij']:.4g} ({sign}, |m_ij|={edge['abs_m_ij']:.4g})"
        )
        _print_branch(edge["receiver"], children, indent + 1)


def print_greedy_tree(result: ModeGreedyTree) -> None:
    print("=" * 96)
    print(f"Mode {result.mode}    eigenvalue {format_eigenvalue(result.eigenvalue)}")
    print("Top contributors:")
    for row in result.contributors.itertuples(index=False):
        print(
            f"  {row.rank:>2}. {row.cell_type} "
            f"[{row.region}, {row.ei}] loading={row.loading_magnitude:.4f}"
        )

    print("Greedy tree flow:")
    if result.tree_edges.empty:
        for root in result.roots:
            print(f"  {root}  (no nonzero outgoing edge among the top contributors)")
        return

    children = _children_by_source(result.tree_edges)
    receivers = set(result.tree_edges["receiver"])
    for root in result.roots:
        if root not in receivers:
            print(f"  root: {root}")
            _print_branch(root, children, indent=1)


def print_all_greedy_trees(results: list[ModeGreedyTree]) -> None:
    for result in results:
        print_greedy_tree(result)


def save_greedy_tree_outputs(results: list[ModeGreedyTree], output_dir: str | Path) -> pd.DataFrame:
    output = Path(output_dir)
    submatrix_dir = output / "submatrices"
    netlist_dir = output / "netlist_edges"
    tree_dir = output / "greedy_trees"
    for folder in (submatrix_dir, netlist_dir, tree_dir):
        folder.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for result in results:
        mode_label = f"mode_{result.mode:03d}"
        result.contributors.to_csv(output / f"{mode_label}_top_contributors.csv", index=False)
        result.submatrix.to_csv(submatrix_dir / f"{mode_label}_top7_source_receiver_submatrix.csv")
        result.netlist_edges.to_csv(netlist_dir / f"{mode_label}_top7_netlist_edges.csv", index=False)
        result.tree_edges.to_csv(tree_dir / f"{mode_label}_greedy_tree_edges.csv", index=False)
        dominant = result.contributors.iloc[0]
        summary_rows.append(
            {
                "mode": result.mode,
                "eigenvalue_real": result.eigenvalue.real,
                "eigenvalue_imag": result.eigenvalue.imag,
                "eigenvalue_magnitude": abs(result.eigenvalue),
                "dominant_cell_type": dominant["cell_type"],
                "dominant_region": dominant["region"],
                "dominant_ei": dominant["ei"],
                "dominant_loading_magnitude": dominant["loading_magnitude"],
                "n_tree_edges": len(result.tree_edges),
                "n_components": len(result.roots),
                "roots": "; ".join(result.roots),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "schur_mode_greedy_tree_summary.csv", index=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-path", default=None, help="Path to mij_matrix.csv.")
    parser.add_argument("--netlist-path", default=None, help="Path to mij_netlist.csv.")
    parser.add_argument("--top-n", type=int, default=7, help="Number of Schur-mode contributors to use.")
    parser.add_argument("--max-modes", type=int, default=None, help="Optional limit for a quick preview.")
    parser.add_argument("--normalization", default="spectral_radius", choices=["spectral_radius", "column_l1", "none"])
    parser.add_argument("--target-spectral-radius", type=float, default=0.95)
    parser.add_argument("--no-self", action="store_true", help="Zero the diagonal before Schur decomposition.")
    parser.add_argument("--schur-archive-path", default=str(DEFAULT_SCHUR_ARCHIVE))
    parser.add_argument("--recompute-schur", action="store_true", help="Ignore saved outputs/schur_modes/schur_modes.npz.")
    parser.add_argument("--output-dir", default="outputs/schur_mode_greedy_trees")
    parser.add_argument("--no-save", action="store_true", help="Print only; do not save CSV outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = build_all_mode_greedy_trees(
        matrix_path=args.matrix_path,
        netlist_path=args.netlist_path,
        top_n=args.top_n,
        normalization=args.normalization,
        target_spectral_radius=args.target_spectral_radius,
        include_self=not args.no_self,
        max_modes=args.max_modes,
        schur_archive_path=args.schur_archive_path,
        prefer_schur_archive=not args.recompute_schur,
    )
    if not args.no_save:
        summary = save_greedy_tree_outputs(results, args.output_dir)
        print(f"Saved greedy-tree outputs for {len(summary)} modes to {Path(args.output_dir).resolve()}")
    print_all_greedy_trees(results)


if __name__ == "__main__":
    main()
