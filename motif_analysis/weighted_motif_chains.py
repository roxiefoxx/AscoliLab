#!/usr/bin/env python3
"""Quick text report for weighted E/I chains.

The reusable table and figure workflow lives in ``ei_chain_signature.py``.  This
script keeps the older console-oriented report, but removes the hardcoded paths.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from ei_chain_signature import build_chains, load_edges, run_chain_signature_analysis, ChainSignatureConfig


DG_CELLS = ["DG Granule", "DG AIPRIM", "DG Axo Axonic", "DG HICAP", "DG MOLAX", "DG Mossy"]


def load_matrix(matrix_path: Path) -> tuple[list[str], np.ndarray]:
    matrix = pd.read_csv(matrix_path, index_col=0).astype(float)
    names = list(matrix.index)
    if list(matrix.columns) != names:
        raise ValueError("Row and column labels must match in the same order.")
    weights = np.abs(matrix.to_numpy())
    np.fill_diagonal(weights, 0.0)
    return names, weights


def ei_by_name(edges: pd.DataFrame) -> dict[str, str]:
    source = edges[["source", "source_ei"]].rename(columns={"source": "cell", "source_ei": "ei"})
    target = edges[["target", "target_ei"]].rename(columns={"target": "cell", "target_ei": "ei"})
    labels = pd.concat([source, target], ignore_index=True).drop_duplicates()
    conflicts = labels.groupby("cell")["ei"].nunique()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        raise ValueError(f"Conflicting E/I labels for {conflicts.index.tolist()[:10]}")
    return dict(labels.itertuples(index=False, name=None))


def weighted_triad_participation(names: list[str], weights: np.ndarray, ei: dict[str, str]) -> pd.DataFrame:
    n = len(names)
    sym = weights + weights.T
    total = sym.sum() / 2.0
    participation = (n - 3) * sym.sum(axis=1) + total
    out = pd.DataFrame(
        {
            "cell": names,
            "ei": [ei[name] for name in names],
            "wpart": participation,
        }
    ).set_index("cell")
    out["rank"] = out["wpart"].rank(ascending=False, method="min").astype(int)
    return out


def directed_3_cycles(names: list[str], weights: np.ndarray) -> list[tuple[str, str, str]]:
    binary = (weights > 0).astype(int)
    cycles = []
    for a, b, c in itertools.combinations(range(len(names)), 3):
        for x, y, z in [(a, b, c), (a, c, b)]:
            if binary[x, y] and binary[y, z] and binary[z, x] and not binary[y, x] and not binary[z, y] and not binary[x, z]:
                cycles.append((names[x], names[y], names[z]))
    return cycles


def print_weighted_participation(df: pd.DataFrame) -> None:
    print("=== Top 12 by WEIGHTED triad participation ===")
    for cell, row in df.sort_values("wpart", ascending=False).head(12).iterrows():
        print(f"  {row['rank']:3d}. {cell:36s} {row['ei']}  {row['wpart']:14,.0f}")
    print("  ---- DG ----")
    for cell in DG_CELLS:
        if cell in df.index:
            row = df.loc[cell]
            print(f"  {row['rank']:3d}. {cell:36s} {row['ei']}  {row['wpart']:14,.0f}")


def print_chain_summary(chains: pd.DataFrame, source_cell: str) -> None:
    chains = chains.copy()
    chains["two_edge_abs_sum"] = chains["edge1_weight"].abs() + chains["edge2_weight"].abs()
    summary = (
        chains.groupby("signature")
        .agg(chain_count=("signature", "size"), weight=("two_edge_abs_sum", "sum"))
    )
    total_weight = summary["weight"].sum()

    print("\n=== Directed 3-chains X->Y->Z by E/I signature ===")
    for signature, row in summary.sort_values("weight", ascending=False).iterrows():
        share = 100 * row["weight"] / total_weight if total_weight else 0.0
        print(f"  {signature}  count={int(row['chain_count']):6d}  weight={row['weight']:16,.0f}  ({share:5.1f}% of chain weight)")

    if source_cell not in set(chains["source"]):
        return

    local = chains[chains["source"] == source_cell].copy()
    by_signature = local.groupby("signature")["two_edge_abs_sum"].sum().sort_values(ascending=False)
    print(f"\n=== {source_cell} (source) 3-chain weight by downstream signature ===")
    for signature, value in by_signature.items():
        share = 100 * value / by_signature.sum() if by_signature.sum() else 0.0
        print(f"  {signature}  {value:16,.0f}  ({share:5.1f}%)")

    eii = local[local["signature"] == "EII"].copy()
    if eii.empty:
        return
    eii["min_abs_edge_weight"] = np.minimum(eii["edge1_weight"].abs(), eii["edge2_weight"].abs())
    eii = eii.sort_values("min_abs_edge_weight", ascending=False)
    print(f"\n=== {source_cell} -> I -> I  top disinhibitory paths (by min edge weight) ===")
    for row in eii.head(10).itertuples(index=False):
        print(
            f"   {source_cell} -> {row.middle:26s} -> {row.target:26s}  "
            f"({row.edge1_weight:12,.0f} , {row.edge2_weight:12,.0f})"
        )
    print(f"   ...{len(eii)} such paths total")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="mij_matrix.csv")
    parser.add_argument("--netlist", default="mij_netlist.csv")
    parser.add_argument("--outdir", default="outputs/ei_chain_signature_eda")
    parser.add_argument("--source-cell", default="DG Granule")
    parser.add_argument("--write-tables", action="store_true")
    args = parser.parse_args()

    _netlist, edges = load_edges(args.netlist)
    names, weights = load_matrix(Path(args.matrix))
    ei = ei_by_name(edges)

    print_weighted_participation(weighted_triad_participation(names, weights, ei))

    cycles = directed_3_cycles(names, weights)
    print(f"\n=== 030C 3-cycles: {len(cycles)} total ===")
    for cycle in cycles:
        print("   ", " -> ".join(cycle))

    print_chain_summary(build_chains(edges), args.source_cell)

    if args.write_tables:
        config = ChainSignatureConfig(
            data_path=Path(args.netlist),
            output_dir=Path(args.outdir),
            selected_signature="EII",
        )
        run_chain_signature_analysis(config, make_figures=True, show_figures=False)


if __name__ == "__main__":
    main()
