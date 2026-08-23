#!/usr/bin/env python3
"""
Dynamic-programming E->E backbone builder for a connectomics matrix.

Interpretation
--------------
Rows are senders and columns are receivers. For each excitatory node used as an
initial injection seed, this script finds a maximum-weight simple directed path:

    seed -> v1 -> v2 -> ... -> vk

subject to:
    * nodes are not revisited;
    * only excitatory-to-excitatory candidate edges are considered;
    * by default, only positive edges are allowed;
    * the score is the sum of edge weights.

This is the dynamic-programming analogue of a greedy "take the largest outgoing
edge without revisiting nodes" backbone. Unlike the greedy method, it optimizes
the total path weight globally for each seed.

Important computational note
----------------------------
The exact DP for maximum-weight simple directed paths is exponential in the
number of excitatory nodes: O(n^2 2^n). This is Held-Karp-style bitmask DP.
For n=32, exact full-state DP is generally too large. Therefore this script
uses a configurable `--max-exact-n` guard. If the excitatory set is larger than
that limit, it automatically switches to a beam-limited DP approximation unless
`--force-exact` is specified.

Example
-------
python dynamic_programming_ee_backbone.py \
    --matrix mij_matrix.csv \
    --output-prefix ee_dp_backbone \
    --positive-only \
    --max-exact-n 22 \
    --beam-width 20000

Outputs
-------
<output-prefix>_edges.csv
<output-prefix>_paths.csv
<output-prefix>_summary.csv
<output-prefix>_adjacency_long.csv
<output-prefix>_nodes_used.csv
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_INHIBITORY_PATTERNS = [
    r"\bvip\b", r"\bpvalb\b", r"\bpv\b", r"\bsst\b", r"\bsom\b",
    r"\bchandelier\b", r"\binterneuron\b", r"\binhib\b", r"\bgaba\b",
    r"\bngf\b", r"\blamp5\b", r"\brel ntn\b", r"\bmeis2\b",
]


@dataclass
class DPResult:
    seed: str
    method: str
    score: float
    path: List[str]
    edge_weights: List[float]
    states_evaluated: int
    exact: bool


def infer_excitatory_nodes(
    labels: Sequence[str],
    inhibitory_patterns: Sequence[str] = DEFAULT_INHIBITORY_PATTERNS,
) -> List[str]:
    """
    Infer excitatory nodes by excluding labels that look inhibitory.

    For published analyses, prefer passing an explicit --exc-nodes-file.
    """
    compiled = [re.compile(pat, re.IGNORECASE) for pat in inhibitory_patterns]
    exc = []
    for lab in labels:
        s = str(lab)
        if not any(p.search(s) for p in compiled):
            exc.append(s)
    return exc


def load_exc_nodes(args: argparse.Namespace, labels: Sequence[str]) -> List[str]:
    labels_set = set(map(str, labels))
    if args.exc_nodes_file:
        node_df = pd.read_csv(args.exc_nodes_file, header=None)
        nodes = [str(x) for x in node_df.iloc[:, 0].dropna().tolist()]
        missing = [x for x in nodes if x not in labels_set]
        if missing:
            raise ValueError(
                f"{len(missing)} excitatory nodes from --exc-nodes-file are absent "
                f"from matrix labels. First missing examples: {missing[:5]}"
            )
        return nodes

    return infer_excitatory_nodes(labels)


def prepare_weight_matrix(
    df: pd.DataFrame,
    exc_nodes: Sequence[str],
    positive_only: bool = True,
    allow_self_edges: bool = False,
) -> np.ndarray:
    W = df.loc[list(exc_nodes), list(exc_nodes)].astype(float).to_numpy()
    W = np.where(np.isfinite(W), W, -np.inf)

    if positive_only:
        W = np.where(W > 0, W, -np.inf)

    if not allow_self_edges:
        np.fill_diagonal(W, -np.inf)

    return W


def reconstruct_path(
    end_state: Tuple[int, int],
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
    names: Sequence[str],
    W: np.ndarray,
) -> Tuple[List[str], List[float]]:
    state = end_state
    rev_idx = []
    while state is not None:
        mask, last = state
        rev_idx.append(last)
        state = parent[state]
    idx_path = list(reversed(rev_idx))
    node_path = [names[i] for i in idx_path]
    weights = [float(W[a, b]) for a, b in zip(idx_path[:-1], idx_path[1:])]
    return node_path, weights


def exact_bitmask_dp_for_seed(
    seed_idx: int,
    names: Sequence[str],
    W: np.ndarray,
) -> DPResult:
    """
    Exact dynamic programming over subsets.

    State:
        dp[(mask, last)] = best path score beginning at seed, visiting exactly
        nodes in mask, and ending at last.
    Transition:
        dp[mask | (1 << nxt), nxt] =
            max(dp[mask, last] + W[last, nxt])
    """
    seed_name = names[seed_idx]
    n = len(names)

    start_mask = 1 << seed_idx
    start_state = (start_mask, seed_idx)

    dp: Dict[Tuple[int, int], float] = {start_state: 0.0}
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start_state: None}

    best_state = start_state
    best_score = 0.0
    states_evaluated = 0

    # Iterate by path size to ensure all predecessor states are available.
    current_layer = {start_state: 0.0}

    for _size in range(1, n):
        next_layer: Dict[Tuple[int, int], float] = {}

        for (mask, last), score in current_layer.items():
            states_evaluated += 1
            candidates = np.where(np.isfinite(W[last]))[0]
            for nxt in candidates:
                bit = 1 << int(nxt)
                if mask & bit:
                    continue

                new_score = score + float(W[last, nxt])
                new_state = (mask | bit, int(nxt))

                if new_score > dp.get(new_state, -math.inf):
                    dp[new_state] = new_score
                    parent[new_state] = (mask, last)
                    next_layer[new_state] = new_score

                    if new_score > best_score:
                        best_score = new_score
                        best_state = new_state

        if not next_layer:
            break

        current_layer = next_layer

    path, edge_weights = reconstruct_path(best_state, parent, names, W)
    return DPResult(
        seed=seed_name,
        method="exact_bitmask_dp",
        score=float(best_score),
        path=path,
        edge_weights=edge_weights,
        states_evaluated=states_evaluated,
        exact=True,
    )


def beam_dp_for_seed(
    seed_idx: int,
    names: Sequence[str],
    W: np.ndarray,
    beam_width: int,
) -> DPResult:
    """
    Beam-limited dynamic programming approximation.

    This preserves the DP state definition but keeps only the top `beam_width`
    states after each path length. It is not guaranteed exact, but is often much
    stronger than a local greedy path while remaining tractable for n > ~22.
    """
    seed_name = names[seed_idx]
    n = len(names)

    start_state = (1 << seed_idx, seed_idx)
    current_layer: Dict[Tuple[int, int], float] = {start_state: 0.0}
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start_state: None}
    seen_best: Dict[Tuple[int, int], float] = {start_state: 0.0}

    best_state = start_state
    best_score = 0.0
    states_evaluated = 0

    for _size in range(1, n):
        proposed: Dict[Tuple[int, int], float] = {}

        for (mask, last), score in current_layer.items():
            states_evaluated += 1
            candidates = np.where(np.isfinite(W[last]))[0]
            for nxt in candidates:
                bit = 1 << int(nxt)
                if mask & bit:
                    continue

                new_state = (mask | bit, int(nxt))
                new_score = score + float(W[last, nxt])

                if new_score > proposed.get(new_state, -math.inf):
                    proposed[new_state] = new_score

                if new_score > seen_best.get(new_state, -math.inf):
                    seen_best[new_state] = new_score
                    parent[new_state] = (mask, last)

                if new_score > best_score:
                    best_score = new_score
                    best_state = new_state

        if not proposed:
            break

        if len(proposed) > beam_width:
            items = sorted(proposed.items(), key=lambda kv: kv[1], reverse=True)[:beam_width]
            current_layer = dict(items)
        else:
            current_layer = proposed

    path, edge_weights = reconstruct_path(best_state, parent, names, W)
    return DPResult(
        seed=seed_name,
        method=f"beam_dp_width_{beam_width}",
        score=float(best_score),
        path=path,
        edge_weights=edge_weights,
        states_evaluated=states_evaluated,
        exact=False,
    )


def result_to_edges(result: DPResult) -> List[dict]:
    rows = []
    for step, (src, dst, weight) in enumerate(
        zip(result.path[:-1], result.path[1:], result.edge_weights), start=1
    ):
        rows.append(
            {
                "seed": result.seed,
                "step": step,
                "sender": src,
                "receiver": dst,
                "weight": weight,
                "path_score": result.score,
                "path_length_edges": len(result.edge_weights),
                "method": result.method,
                "exact": result.exact,
                "states_evaluated": result.states_evaluated,
            }
        )
    return rows


def result_to_path_row(result: DPResult) -> dict:
    return {
        "seed": result.seed,
        "method": result.method,
        "exact": result.exact,
        "path_score": result.score,
        "path_length_edges": len(result.edge_weights),
        "path_length_nodes": len(result.path),
        "states_evaluated": result.states_evaluated,
        "path": " -> ".join(result.path),
        "edge_weights": ";".join(f"{x:.10g}" for x in result.edge_weights),
    }


def result_to_summary_row(result: DPResult) -> dict:
    return {
        "seed": result.seed,
        "method": result.method,
        "exact": result.exact,
        "path_score": result.score,
        "n_edges": len(result.edge_weights),
        "n_nodes": len(result.path),
        "terminal_node": result.path[-1] if result.path else result.seed,
        "states_evaluated": result.states_evaluated,
    }


def build_adjacency_long(results: Sequence[DPResult], exc_nodes: Sequence[str]) -> pd.DataFrame:
    rows = []
    for res in results:
        edge_lookup = {
            (src, dst): wt
            for src, dst, wt in zip(res.path[:-1], res.path[1:], res.edge_weights)
        }
        for src in exc_nodes:
            for dst in exc_nodes:
                if src == dst:
                    continue
                rows.append(
                    {
                        "seed": res.seed,
                        "sender": src,
                        "receiver": dst,
                        "in_backbone": int((src, dst) in edge_lookup),
                        "backbone_weight": edge_lookup.get((src, dst), 0.0),
                        "method": res.method,
                        "exact": res.exact,
                    }
                )
    return pd.DataFrame(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build per-injection E->E backbones using dynamic programming."
    )
    p.add_argument("--matrix", default="mij_matrix.csv", help="CSV matrix; senders are rows, receivers are columns.")
    p.add_argument("--output-prefix", default="ee_dp_backbone", help="Prefix for output CSV files.")
    p.add_argument("--exc-nodes-file", default=None, help="Optional one-column CSV listing excitatory nodes.")
    p.add_argument("--positive-only", action="store_true", help="Keep only positive E->E edges.")
    p.add_argument("--allow-self-edges", action="store_true", help="Allow self edges; usually not recommended.")
    p.add_argument("--max-exact-n", type=int, default=22, help="Maximum n for exact bitmask DP unless --force-exact.")
    p.add_argument("--force-exact", action="store_true", help="Force exact DP even when n exceeds --max-exact-n.")
    p.add_argument("--beam-width", type=int, default=20000, help="Beam width for approximate DP fallback.")
    p.add_argument(
        "--seeds-file",
        default=None,
        help="Optional one-column CSV listing injection seeds. Defaults to all excitatory nodes.",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    matrix_path = Path(args.matrix)
    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

    df = pd.read_csv(matrix_path, index_col=0)
    df.index = df.index.map(str)
    df.columns = df.columns.map(str)

    if set(df.index) != set(df.columns):
        raise ValueError("Matrix row and column labels must contain the same node set.")

    # Align columns to row order.
    df = df.loc[df.index, df.index]

    exc_nodes = load_exc_nodes(args, df.index.tolist())
    if len(exc_nodes) == 0:
        raise ValueError("No excitatory nodes detected. Provide --exc-nodes-file.")

    W = prepare_weight_matrix(
        df,
        exc_nodes,
        positive_only=args.positive_only,
        allow_self_edges=args.allow_self_edges,
    )

    if args.seeds_file:
        seed_df = pd.read_csv(args.seeds_file, header=None)
        seeds = [str(x) for x in seed_df.iloc[:, 0].dropna().tolist()]
        missing = [s for s in seeds if s not in set(exc_nodes)]
        if missing:
            raise ValueError(f"Seeds absent from excitatory node set: {missing[:5]}")
    else:
        seeds = list(exc_nodes)

    name_to_idx = {name: i for i, name in enumerate(exc_nodes)}
    n = len(exc_nodes)

    use_exact = args.force_exact or n <= args.max_exact_n
    if not use_exact and args.beam_width <= 0:
        raise ValueError("--beam-width must be positive for approximate DP.")

    results: List[DPResult] = []

    for seed in seeds:
        seed_idx = name_to_idx[seed]
        if use_exact:
            res = exact_bitmask_dp_for_seed(seed_idx, exc_nodes, W)
        else:
            res = beam_dp_for_seed(seed_idx, exc_nodes, W, beam_width=args.beam_width)
        results.append(res)
        print(
            f"{seed}: score={res.score:.6g}, edges={len(res.edge_weights)}, "
            f"states={res.states_evaluated}, method={res.method}",
            file=sys.stderr,
        )

    prefix = Path(args.output_prefix)

    edges = []
    for res in results:
        edges.extend(result_to_edges(res))

    pd.DataFrame(edges).to_csv(f"{prefix}_edges.csv", index=False)
    pd.DataFrame([result_to_path_row(r) for r in results]).to_csv(f"{prefix}_paths.csv", index=False)
    pd.DataFrame([result_to_summary_row(r) for r in results]).to_csv(f"{prefix}_summary.csv", index=False)
    build_adjacency_long(results, exc_nodes).to_csv(f"{prefix}_adjacency_long.csv", index=False)
    pd.DataFrame({"node": exc_nodes}).to_csv(f"{prefix}_nodes_used.csv", index=False)

    print("\nWrote:", file=sys.stderr)
    print(f"  {prefix}_edges.csv", file=sys.stderr)
    print(f"  {prefix}_paths.csv", file=sys.stderr)
    print(f"  {prefix}_summary.csv", file=sys.stderr)
    print(f"  {prefix}_adjacency_long.csv", file=sys.stderr)
    print(f"  {prefix}_nodes_used.csv", file=sys.stderr)

    if not use_exact:
        print(
            "\nNote: used beam-limited DP approximation because n="
            f"{n} exceeds --max-exact-n={args.max_exact_n}. "
            "Increase --max-exact-n or pass --force-exact only if resources permit.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
