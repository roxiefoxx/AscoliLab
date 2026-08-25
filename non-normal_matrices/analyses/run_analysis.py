"""Run the empirical Shao et al. style analysis for data/mij_matrix.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from connectivity_methods import load_mij_matrix, save_outputs


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository/workspace root from either root or analyses/."""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "data" / "mij_matrix.csv").exists():
            return candidate
    raise FileNotFoundError("Could not find data/mij_matrix.csv in this directory or its parents.")


def resolve_project_path(path_value: str, project_root: Path) -> Path:
    """Resolve CLI paths relative to the project root when needed."""

    path = Path(path_value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        default="data/mij_matrix.csv",
        help="CSV with senders in rows and receivers in columns.",
    )
    parser.add_argument(
        "--netlist",
        default="data/mij_netlist.csv",
        help="Optional netlist CSV containing pre_neuron and pre_ei labels.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/mij_paper_replication",
        help="Directory for analysis CSV outputs.",
    )
    parser.add_argument(
        "--spectral-radius",
        type=float,
        default=1.0,
        help="Target spectral radius for the normalized matrix.",
    )
    parser.add_argument(
        "--response-scale",
        type=float,
        default=1.0,
        help=(
            "Scale used in Gamma = (I - scale * J)^-1. "
            "At 1.0, a spectral-radius-one matrix is marginal and may be ill-conditioned."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = find_project_root(Path(__file__).resolve().parent)
    matrix_path = resolve_project_path(args.matrix, project_root)
    netlist_path = resolve_project_path(args.netlist, project_root)
    output_dir = resolve_project_path(args.output_dir, project_root)

    data = load_mij_matrix(matrix_path, netlist_path=netlist_path, spectral_radius_target=args.spectral_radius)
    paths = save_outputs(data, output_dir, response_scale=args.response_scale)

    metadata = pd.read_csv(paths["metadata"])
    responses = pd.read_csv(paths["responses"])
    low_rank = pd.read_csv(paths["low_rank"])
    motif_enrichment = pd.read_csv(paths["motif_enrichment"])
    chain_sweep = pd.read_csv(paths["chain_sweep"])
    sensitivity_blocks = pd.read_csv(paths["sensitivity_blocks"])

    print("Analysis complete.")
    print(f"Output directory: {output_dir}")
    print(metadata.T.to_string(header=False))
    print("\nPopulation responses:")
    print(responses.to_string(index=False))
    print("\nLow-rank response approximation:")
    print(low_rank.to_string(index=False))
    print("\nMotif enrichment summary:")
    print(
        motif_enrichment.groupby("motif")["enrichment_ratio"]
        .agg(["mean", "min", "max"])
        .reset_index()
        .to_string(index=False)
    )
    print("\nDominant eigenvalue block sensitivity:")
    print(sensitivity_blocks.to_string(index=False))
    print("\nChain enrichment sweep:")
    print(chain_sweep.to_string(index=False))


if __name__ == "__main__":
    main()
