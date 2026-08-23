#!/usr/bin/env python3
"""
Branch-and-bound E→E backbone construction from a connectomics matrix.

Input convention
----------------
- CSV rows are senders, columns are receivers.
- Edge i -> j has weight M.loc[i, j].
- Only positive E→E weights are considered by default.
- For each excitatory node, the algorithm treats that node as the initial
  injection point and searches for the maximum-weight simple directed path
  starting at that seed, without revisiting nodes.

Why branch-and-bound?
---------------------
The greedy backbone follows the strongest available outgoing edge. This script
instead explores alternatives and prunes branches whose best possible remaining
score cannot beat the current incumbent.

Objective
---------
For each seed s, maximize:

    sum_{(u,v) in path} weight(u, v)

over all simple directed paths that start at s.

A path may stop at any node. Therefore the empty path with score 0 is allowed,
which is useful if a seed has no positive E→E outgoing edges.

Outputs
-------
1. <prefix>_bb_edges.csv
   One row per selected path edge per seed.

2. <prefix>_bb_paths.csv
   One row per seed with the ordered node path.

3. <prefix>_bb_summary.csv
   Search diagnostics and optimality status per seed.

4. <prefix>_bb_adjacency_long.csv
   Long-form seed-specific adjacency table.

Example
-------
python branch_bound_ee_backbone.py \
  --matrix mij_matrix.csv \
  --out-prefix ee_backbone \
  --positive-only \
  --time-limit 30 \
  --node-limit 1000000

Notes on excitatory-node detection
----------------------------------
By default, labels that look like inhibitory interneurons are excluded using
case-insensitive keyword matching. You can override this by passing a text file
with one excitatory node label per line:

  --e-nodes-file my_excitatory_nodes.txt
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_INHIBITORY_KEYWORDS = (
    "sst", "pvalb", "pv", "vip", "lamp5", "sncg", "gad", "gaba",
    "chandelier", "basket", "interneuron", "inh", "ngf", "neurogliaform",
)


@dataclass
class SearchResult:
    seed: str
    best_path: List[int]
    best_score: float
    expanded_nodes: int
    pruned_branches: int
    complete: bool
    stop_reason: str
    elapsed_seconds: float


def load_matrix(path: str | Path) -> pd.DataFrame:
    """Load a square sender-by-receiver matrix."""
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)

    if df.shape[0] != df.shape[1]:
        raise ValueError(f"Matrix must be square; got {df.shape}.")

    if set(df.index) != set(df.columns):
        raise ValueError("Row and column labels differ. Senders and receivers must match.")

    # Align columns to row order.
    df = df.loc[df.index, df.index]
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def infer_excitatory_nodes(
    labels: Sequence[str],
    inhibitory_keywords: Sequence[str] = DEFAULT_INHIBITORY_KEYWORDS,
) -> List[str]:
    """
    Infer putative excitatory nodes by excluding common inhibitory labels.

    This is intentionally conservative and label-dependent. For a curated
    E-node set, use --e-nodes-file.
    """
    e_nodes = []
    for label in labels:
        lower = label.lower()
        if not any(k.lower() in lower for k in inhibitory_keywords):
            e_nodes.append(label)
    return e_nodes


def read_node_list(path: str | Path) -> List[str]:
    """Read one node label per line, ignoring blanks and comments."""
    nodes = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            item = line.strip()
            if item and not item.startswith("#"):
                nodes.append(item)
    return nodes


def build_weight_matrix(
    df: pd.DataFrame,
    nodes: Sequence[str],
    positive_only: bool = True,
    allow_self_edges: bool = False,
) -> np.ndarray:
    """Extract E→E weight matrix."""
    sub = df.loc[nodes, nodes].to_numpy(dtype=float)

    if positive_only:
        sub = np.where(sub > 0, sub, 0.0)

    if not allow_self_edges:
        np.fill_diagonal(sub, 0.0)

    return sub


def suffix_positive_bounds(weights: np.ndarray) -> np.ndarray:
    """
    Precompute an optimistic remaining-score bound.

    For a current node u and a set of unvisited nodes R, a valid but optimistic
    upper bound is the sum of the largest possible positive outgoing edge from
    each remaining node plus the current node. Because a simple path can use at
    most one outgoing edge from each visited/current node, this overestimates
    the achievable continuation and is safe for pruning.

    Returns:
        max_out[i] = max_j weight(i, j), clipped at 0.
    """
    return np.maximum(weights.max(axis=1), 0.0)


def greedy_lower_bound(seed_idx: int, weights: np.ndarray) -> Tuple[List[int], float]:
    """Fast feasible path used as initial incumbent."""
    n = weights.shape[0]
    visited = {seed_idx}
    path = [seed_idx]
    score = 0.0
    cur = seed_idx

    while len(visited) < n:
        candidates = [
            (weights[cur, j], j)
            for j in range(n)
            if j not in visited and weights[cur, j] > 0
        ]
        if not candidates:
            break
        w, nxt = max(candidates)
        score += float(w)
        path.append(nxt)
        visited.add(nxt)
        cur = nxt

    return path, score


def branch_and_bound_path(
    seed_idx: int,
    weights: np.ndarray,
    node_labels: Sequence[str],
    time_limit: float | None = None,
    node_limit: int | None = None,
) -> SearchResult:
    """
    Find the maximum-weight simple directed path from one seed.

    Depth-first branch-and-bound.
    Branching order explores high-weight outgoing edges first.
    """
    n = weights.shape[0]
    max_out = suffix_positive_bounds(weights)

    best_path, best_score = greedy_lower_bound(seed_idx, weights)
    expanded = 0
    pruned = 0
    start = time.perf_counter()
    complete = True
    stop_reason = "exhaustive"

    all_nodes_mask = (1 << n) - 1

    def elapsed() -> float:
        return time.perf_counter() - start

    def optimistic_bound(current: int, visited_mask: int, score: float) -> float:
        # Current node may still contribute one outgoing edge. Unvisited nodes
        # may also contribute at most one outgoing edge each. This is deliberately
        # loose but fast and safe.
        remaining_mask = all_nodes_mask ^ visited_mask
        bound = score + max_out[current]
        m = remaining_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            bound += max_out[idx]
            m ^= lsb
        return float(bound)

    def dfs(current: int, visited_mask: int, path: List[int], score: float) -> None:
        nonlocal best_path, best_score, expanded, pruned, complete, stop_reason

        if time_limit is not None and elapsed() >= time_limit:
            complete = False
            stop_reason = "time_limit"
            return

        if node_limit is not None and expanded >= node_limit:
            complete = False
            stop_reason = "node_limit"
            return

        expanded += 1

        # Since stopping is allowed, every partial path is feasible.
        if score > best_score:
            best_score = float(score)
            best_path = path.copy()

        if optimistic_bound(current, visited_mask, score) <= best_score + 1e-12:
            pruned += 1
            return

        candidates = []
        for nxt in range(n):
            if (visited_mask >> nxt) & 1:
                continue
            w = weights[current, nxt]
            if w > 0:
                candidates.append((float(w), nxt))

        # Explore best immediate continuations first to improve incumbent early.
        candidates.sort(reverse=True, key=lambda x: x[0])

        for w, nxt in candidates:
            if not complete:
                return
            dfs(
                current=nxt,
                visited_mask=visited_mask | (1 << nxt),
                path=path + [nxt],
                score=score + w,
            )

    dfs(seed_idx, 1 << seed_idx, [seed_idx], 0.0)

    return SearchResult(
        seed=str(node_labels[seed_idx]),
        best_path=best_path,
        best_score=float(best_score),
        expanded_nodes=expanded,
        pruned_branches=pruned,
        complete=complete,
        stop_reason=stop_reason,
        elapsed_seconds=elapsed(),
    )


def result_to_rows(result: SearchResult, weights: np.ndarray, labels: Sequence[str]) -> List[Dict[str, object]]:
    """Convert one SearchResult to edge rows."""
    rows = []
    for step, (u, v) in enumerate(zip(result.best_path[:-1], result.best_path[1:]), start=1):
        rows.append(
            {
                "seed": result.seed,
                "step": step,
                "sender": labels[u],
                "receiver": labels[v],
                "weight": float(weights[u, v]),
                "cumulative_weight": float(
                    sum(weights[a, b] for a, b in zip(result.best_path[:-1][:step], result.best_path[1:][:step]))
                ),
                "path_length_edges": max(0, len(result.best_path) - 1),
                "total_path_weight": result.best_score,
                "complete": result.complete,
                "stop_reason": result.stop_reason,
            }
        )
    return rows


def make_adjacency_long(edge_df: pd.DataFrame, seeds: Sequence[str], nodes: Sequence[str]) -> pd.DataFrame:
    """Create seed-specific long adjacency rows for selected backbone edges."""
    if edge_df.empty:
        return pd.DataFrame(columns=["seed", "sender", "receiver", "selected", "weight"])

    edge_lookup = {
        (row.seed, row.sender, row.receiver): row.weight
        for row in edge_df.itertuples(index=False)
    }

    rows = []
    for seed in seeds:
        for sender in nodes:
            for receiver in nodes:
                if sender == receiver:
                    continue
                key = (seed, sender, receiver)
                if key in edge_lookup:
                    rows.append(
                        {
                            "seed": seed,
                            "sender": sender,
                            "receiver": receiver,
                            "selected": 1,
                            "weight": float(edge_lookup[key]),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one branch-and-bound maximum-weight simple E→E path per injection seed."
    )
    parser.add_argument("--matrix", default="mij_matrix.csv", help="Sender-by-receiver CSV matrix.")
    parser.add_argument("--out-prefix", default="ee_backbone", help="Output file prefix.")
    parser.add_argument("--e-nodes-file", default=None, help="Optional file with one excitatory node label per line.")
    parser.add_argument("--positive-only", action="store_true", help="Use only positive weights.")
    parser.add_argument("--allow-self-edges", action="store_true", help="Allow self edges. Usually not recommended.")
    parser.add_argument("--time-limit", type=float, default=30.0, help="Seconds per seed. Use <=0 for no limit.")
    parser.add_argument("--node-limit", type=int, default=1_000_000, help="DFS nodes per seed. Use <=0 for no limit.")
    args = parser.parse_args()

    df = load_matrix(args.matrix)

    if args.e_nodes_file:
        e_nodes = read_node_list(args.e_nodes_file)
        missing = sorted(set(e_nodes) - set(df.index))
        if missing:
            raise ValueError(f"E-node labels not found in matrix: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    else:
        e_nodes = infer_excitatory_nodes(list(df.index))

    if len(e_nodes) < 2:
        raise ValueError("Need at least two excitatory nodes to build E→E paths.")

    weights = build_weight_matrix(
        df,
        e_nodes,
        positive_only=args.positive_only,
        allow_self_edges=args.allow_self_edges,
    )

    time_limit = None if args.time_limit <= 0 else args.time_limit
    node_limit = None if args.node_limit <= 0 else args.node_limit

    results: List[SearchResult] = []
    edge_rows: List[Dict[str, object]] = []
    path_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for seed_idx, seed in enumerate(e_nodes):
        res = branch_and_bound_path(
            seed_idx=seed_idx,
            weights=weights,
            node_labels=e_nodes,
            time_limit=time_limit,
            node_limit=node_limit,
        )
        results.append(res)

        edge_rows.extend(result_to_rows(res, weights, e_nodes))

        path_labels = [e_nodes[i] for i in res.best_path]
        path_rows.append(
            {
                "seed": seed,
                "path": " -> ".join(path_labels),
                "path_nodes": len(path_labels),
                "path_edges": max(0, len(path_labels) - 1),
                "total_path_weight": res.best_score,
            }
        )

        summary_rows.append(
            {
                "seed": seed,
                "path_edges": max(0, len(res.best_path) - 1),
                "total_path_weight": res.best_score,
                "expanded_nodes": res.expanded_nodes,
                "pruned_branches": res.pruned_branches,
                "complete": res.complete,
                "stop_reason": res.stop_reason,
                "elapsed_seconds": res.elapsed_seconds,
            }
        )

        print(
            f"{seed}: score={res.best_score:.6g}, edges={max(0, len(res.best_path)-1)}, "
            f"expanded={res.expanded_nodes}, pruned={res.pruned_branches}, "
            f"complete={res.complete}, reason={res.stop_reason}"
        )

    edge_df = pd.DataFrame(edge_rows)
    path_df = pd.DataFrame(path_rows)
    summary_df = pd.DataFrame(summary_rows)
    adjacency_long_df = make_adjacency_long(edge_df, e_nodes, e_nodes)

    prefix = Path(args.out_prefix)
    edge_df.to_csv(f"{prefix}_bb_edges.csv", index=False)
    path_df.to_csv(f"{prefix}_bb_paths.csv", index=False)
    summary_df.to_csv(f"{prefix}_bb_summary.csv", index=False)
    adjacency_long_df.to_csv(f"{prefix}_bb_adjacency_long.csv", index=False)

    pd.DataFrame({"node": e_nodes}).to_csv(f"{prefix}_bb_e_nodes.csv", index=False)

    print("\nWrote:")
    print(f"  {prefix}_bb_edges.csv")
    print(f"  {prefix}_bb_paths.csv")
    print(f"  {prefix}_bb_summary.csv")
    print(f"  {prefix}_bb_adjacency_long.csv")
    print(f"  {prefix}_bb_e_nodes.csv")


if __name__ == "__main__":
    main()
