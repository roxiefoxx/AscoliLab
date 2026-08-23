from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


EPS = 1e-12
EXCITATORY_POSITIVE_FRACTION_THRESHOLD = 0.5


@dataclass
class BackboneData:
    full_matrix: pd.DataFrame
    labels: List[str]
    excitatory_nodes: List[str]
    positive_matrix: pd.DataFrame
    transition_weights: np.ndarray
    self_weights: np.ndarray
    include_self_connections: bool
    adjacency: Dict[int, List[Tuple[int, float]]]
    node_sign_summary: pd.DataFrame
    self_table: pd.DataFrame


class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def load_matrix(path: Path) -> pd.DataFrame:
    matrix = pd.read_csv(path, index_col=0)
    if list(matrix.index) != list(matrix.columns):
        raise ValueError("Row and column labels differ. Align the matrix before running.")
    return matrix.astype(float)


def outgoing_positive_fraction(row: pd.Series, eps: float = EPS) -> float:
    nonzero = row[row.abs() > eps]
    if len(nonzero) == 0:
        return 0.0
    return float((nonzero > 0).mean())


def is_excitatory_sender(
    node: str,
    matrix: pd.DataFrame,
    threshold: float = EXCITATORY_POSITIVE_FRACTION_THRESHOLD,
) -> bool:
    return outgoing_positive_fraction(matrix.loc[node]) >= threshold


def prepare_backbone_data(
    input_csv: Path,
    threshold: float = EXCITATORY_POSITIVE_FRACTION_THRESHOLD,
    eps: float = EPS,
    include_self_connections: bool = False,
) -> BackboneData:
    matrix = load_matrix(input_csv)
    labels = list(matrix.index)

    node_sign_summary = pd.DataFrame(
        {
            "node": labels,
            "positive_fraction_nonzero_outgoing": [
                outgoing_positive_fraction(matrix.loc[x], eps=eps) for x in labels
            ],
            "n_positive_outgoing": [(matrix.loc[x] > eps).sum() for x in labels],
            "n_negative_outgoing": [(matrix.loc[x] < -eps).sum() for x in labels],
        }
    )
    if include_self_connections:
        node_sign_summary["self_weight_raw"] = [float(matrix.loc[x, x]) for x in labels]
    node_sign_summary["is_excitatory"] = (
        node_sign_summary["positive_fraction_nonzero_outgoing"] >= threshold
    )

    excitatory_nodes = node_sign_summary.loc[
        node_sign_summary["is_excitatory"], "node"
    ].tolist()

    positive_matrix = matrix.loc[excitatory_nodes, excitatory_nodes].clip(lower=0.0)
    if include_self_connections:
        self_weights = np.diag(positive_matrix.to_numpy(dtype=float)).copy()
    else:
        self_weights = np.zeros(len(excitatory_nodes), dtype=float)
    transition_weights = positive_matrix.to_numpy(dtype=float).copy()
    np.fill_diagonal(transition_weights, 0.0)

    adjacency = build_adjacency(transition_weights, self_weights, eps=eps)
    self_table = pd.DataFrame()
    if include_self_connections:
        self_table = pd.DataFrame(
            {
                "node": excitatory_nodes,
                "self_weight": self_weights,
                "has_positive_self_connection": self_weights > eps,
            }
        )

    return BackboneData(
        full_matrix=matrix,
        labels=labels,
        excitatory_nodes=excitatory_nodes,
        positive_matrix=positive_matrix,
        transition_weights=transition_weights,
        self_weights=self_weights,
        include_self_connections=include_self_connections,
        adjacency=adjacency,
        node_sign_summary=node_sign_summary,
        self_table=self_table,
    )


def prepare_excitatory_matrix_data(
    input_csv: Path,
    eps: float = EPS,
    include_self_connections: bool = False,
) -> BackboneData:
    matrix = load_matrix(input_csv)
    labels = list(matrix.index)

    node_sign_summary = pd.DataFrame(
        {
            "node": labels,
            "positive_fraction_nonzero_outgoing": [
                outgoing_positive_fraction(matrix.loc[x], eps=eps) for x in labels
            ],
            "n_positive_outgoing": [(matrix.loc[x] > eps).sum() for x in labels],
            "n_negative_outgoing": [(matrix.loc[x] < -eps).sum() for x in labels],
            "is_excitatory": True,
        }
    )
    if include_self_connections:
        node_sign_summary["self_weight_raw"] = [float(matrix.loc[x, x]) for x in labels]

    positive_matrix = matrix.loc[labels, labels].clip(lower=0.0)
    if include_self_connections:
        self_weights = np.diag(positive_matrix.to_numpy(dtype=float)).copy()
    else:
        self_weights = np.zeros(len(labels), dtype=float)
    transition_weights = positive_matrix.to_numpy(dtype=float).copy()
    np.fill_diagonal(transition_weights, 0.0)

    adjacency = build_adjacency(transition_weights, self_weights, eps=eps)
    self_table = pd.DataFrame()
    if include_self_connections:
        self_table = pd.DataFrame(
            {
                "node": labels,
                "self_weight": self_weights,
                "has_positive_self_connection": self_weights > eps,
            }
        )

    return BackboneData(
        full_matrix=matrix,
        labels=labels,
        excitatory_nodes=labels,
        positive_matrix=positive_matrix,
        transition_weights=transition_weights,
        self_weights=self_weights,
        include_self_connections=include_self_connections,
        adjacency=adjacency,
        node_sign_summary=node_sign_summary,
        self_table=self_table,
    )


def build_adjacency(
    transition_weights: np.ndarray,
    self_weights: np.ndarray,
    eps: float = EPS,
) -> Dict[int, List[Tuple[int, float]]]:
    n = len(self_weights)
    return {
        i: sorted(
            [
                (j, float(transition_weights[i, j]))
                for j in range(n)
                if j != i and transition_weights[i, j] > eps
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        for i in range(n)
    }


def transition_score(path: List[int], transition_weights: np.ndarray) -> float:
    return float(sum(transition_weights[u, v] for u, v in zip(path[:-1], path[1:])))


def self_score(path: List[int], self_weights: np.ndarray) -> float:
    return float(sum(self_weights[i] for i in path))


def total_score(
    path: List[int],
    transition_weights: np.ndarray,
    self_weights: np.ndarray,
) -> float:
    return transition_score(path, transition_weights) + self_score(path, self_weights)


def path_to_records(
    seed_i: int,
    method: str,
    path: List[int],
    data: BackboneData,
    extra: Optional[dict] = None,
) -> dict:
    extra = extra or {}
    record = {
        "method": method,
        "seed": data.excitatory_nodes[seed_i],
        "length_edges": max(0, len(path) - 1),
        "n_nodes_visited": len(path),
        "transition_weight_sum": transition_score(path, data.transition_weights),
        "summed_weight": total_score(path, data.transition_weights, data.self_weights),
        "terminus": data.excitatory_nodes[path[-1]] if path else data.excitatory_nodes[seed_i],
        "path": " -> ".join(data.excitatory_nodes[i] for i in path),
        **extra,
    }
    if data.include_self_connections:
        record["self_weight_sum"] = self_score(path, data.self_weights)
        record["n_self_connections_included"] = int(
            sum(data.self_weights[i] > EPS for i in path)
        )
    return record


def edges_from_path(seed_i: int, method: str, path: List[int], data: BackboneData) -> List[dict]:
    rows = []
    if data.include_self_connections:
        for node_pos, u in enumerate(path, start=1):
            if data.self_weights[u] <= EPS:
                continue
            rows.append(
                {
                    "method": method,
                    "seed": data.excitatory_nodes[seed_i],
                    "step": node_pos,
                    "edge_type": "self_connection",
                    "sender": data.excitatory_nodes[u],
                    "receiver": data.excitatory_nodes[u],
                    "weight": float(data.self_weights[u]),
                }
            )

    for step, (u, v) in enumerate(zip(path[:-1], path[1:]), start=1):
        rows.append(
            {
                "method": method,
                "seed": data.excitatory_nodes[seed_i],
                "step": step,
                "edge_type": "transition",
                "sender": data.excitatory_nodes[u],
                "receiver": data.excitatory_nodes[v],
                "weight": float(data.transition_weights[u, v]),
            }
        )
    return rows


def greedy_path(seed: int, data: BackboneData) -> List[int]:
    visited = {seed}
    path = [seed]
    cur = seed

    while True:
        candidates = [
            (j, w)
            for j, w in data.adjacency[cur]
            if j not in visited
        ]
        if not candidates:
            break
        candidates.sort(key=lambda x: x[1], reverse=True)
        nxt = candidates[0][0]
        visited.add(nxt)
        path.append(nxt)
        cur = nxt
    return path


def maximum_spanning_tree_edges(
    transition_weights: np.ndarray,
    eps: float = EPS,
) -> List[Tuple[int, int, float]]:
    n = transition_weights.shape[0]
    sym_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            wij = max(float(transition_weights[i, j]), float(transition_weights[j, i]))
            if wij > eps:
                sym_edges.append((wij, i, j))
    sym_edges.sort(reverse=True)

    dsu = DSU(n)
    mst = []
    for w, i, j in sym_edges:
        if dsu.union(i, j):
            mst.append((i, j, w))
            if len(mst) == n - 1:
                break
    return mst


def build_tree_adjacency(
    n: int,
    mst_edges: List[Tuple[int, int, float]],
) -> Dict[int, List[Tuple[int, float]]]:
    tree_adj = {i: [] for i in range(n)}
    for u, v, w in mst_edges:
        tree_adj[u].append((v, w))
        tree_adj[v].append((u, w))
    return tree_adj


def best_mst_path_from_seed(
    seed: int,
    tree_adj: Dict[int, List[Tuple[int, float]]],
    data: BackboneData,
) -> List[int]:
    best_path = [seed]
    best_score = total_score(best_path, data.transition_weights, data.self_weights)

    stack = [(seed, -1, [seed])]
    while stack:
        u, parent, path = stack.pop()
        children = [(v, w) for v, w in tree_adj[u] if v != parent]
        if not children:
            score = total_score(path, data.transition_weights, data.self_weights)
            if score > best_score or (
                abs(score - best_score) <= EPS and len(path) > len(best_path)
            ):
                best_path, best_score = path, score
        for v, _ in children:
            stack.append((v, u, path + [v]))
    return best_path


def branch_bound_path(
    seed: int,
    data: BackboneData,
    time_limit: float = 0.25,
    node_limit: int = 50_000,
) -> Tuple[List[int], float, int, str]:
    start = time.time()
    n = len(data.excitatory_nodes)
    max_extension = np.array(
        [
            max([w + data.self_weights[j] for j, w in data.adjacency[i]], default=0.0)
            for i in range(n)
        ]
    )

    best_path = [seed]
    best_score = total_score(best_path, data.transition_weights, data.self_weights)
    states_seen = 0
    stopped_by = "complete"

    def upper_bound(score: float, visited: Set[int]) -> float:
        unvisited = [i for i in range(n) if i not in visited]
        return score + float(max_extension[unvisited].sum()) if unvisited else score

    def dfs(u: int, visited: Set[int], path: List[int], score: float) -> None:
        nonlocal best_score, best_path, states_seen, stopped_by
        states_seen += 1
        if states_seen >= node_limit:
            stopped_by = "node_limit"
            return
        if time.time() - start >= time_limit:
            stopped_by = "time_limit"
            return
        if score > best_score or (
            abs(score - best_score) <= EPS and len(path) > len(best_path)
        ):
            best_score = score
            best_path = path.copy()
        if upper_bound(score, visited) <= best_score + EPS:
            return
        for v, w in data.adjacency[u]:
            if v in visited:
                continue
            visited.add(v)
            path.append(v)
            dfs(v, visited, path, score + w + data.self_weights[v])
            path.pop()
            visited.remove(v)
            if stopped_by != "complete":
                return

    dfs(seed, {seed}, [seed], float(data.self_weights[seed]))
    return best_path, best_score, states_seen, stopped_by


def exact_bitmask_dp_path(seed: int, data: BackboneData) -> Tuple[List[int], float, str]:
    n = len(data.excitatory_nodes)
    seed_mask = 1 << seed
    dp = {(seed_mask, seed): float(data.self_weights[seed])}
    parent = {}
    best_key = (seed_mask, seed)
    best_score = float(data.self_weights[seed])

    for size in range(1, n + 1):
        current_items = [(k, v) for k, v in dp.items() if k[0].bit_count() == size]
        for (mask, last), score in current_items:
            if score > best_score or (
                abs(score - best_score) <= EPS
                and mask.bit_count() > best_key[0].bit_count()
            ):
                best_score = score
                best_key = (mask, last)
            for nxt, w in data.adjacency[last]:
                if mask & (1 << nxt):
                    continue
                new_mask = mask | (1 << nxt)
                new_key = (new_mask, nxt)
                new_score = score + w + data.self_weights[nxt]
                if new_score > dp.get(new_key, -math.inf):
                    dp[new_key] = new_score
                    parent[new_key] = (mask, last)

    key = best_key
    rev = [key[1]]
    while key in parent:
        key = parent[key]
        rev.append(key[1])
    return list(reversed(rev)), best_score, "exact"


def beam_dp_path(
    seed: int,
    data: BackboneData,
    beam_width: int = 5_000,
) -> Tuple[List[int], float, str]:
    n = len(data.excitatory_nodes)
    start_state = (1 << seed, seed)
    states = {start_state: (float(data.self_weights[seed]), [seed])}
    best_path = [seed]
    best_score = float(data.self_weights[seed])

    for _ in range(n - 1):
        new_states = dict(states)
        for (mask, last), (score, path) in states.items():
            if score > best_score or (
                abs(score - best_score) <= EPS and len(path) > len(best_path)
            ):
                best_score, best_path = score, path
            for nxt, w in data.adjacency[last]:
                if mask & (1 << nxt):
                    continue
                new_mask = mask | (1 << nxt)
                new_key = (new_mask, nxt)
                new_score = score + w + data.self_weights[nxt]
                old = new_states.get(new_key)
                if old is None or new_score > old[0]:
                    new_states[new_key] = (new_score, path + [nxt])
        if len(new_states) > beam_width:
            ranked = sorted(
                new_states.items(),
                key=lambda kv: (kv[1][0], len(kv[1][1])),
                reverse=True,
            )[:beam_width]
            new_states = dict(ranked)
        if len(new_states) == len(states):
            break
        states = new_states

    for (_, _), (score, path) in states.items():
        if score > best_score or (
            abs(score - best_score) <= EPS and len(path) > len(best_path)
        ):
            best_score, best_path = score, path
    return best_path, best_score, f"beam_width_{beam_width}"


def milp_subtour_elimination_path(
    seed: int,
    data: BackboneData,
    time_limit: Optional[float] = 30.0,
    msg: bool = False,
) -> Tuple[List[int], float, str, float]:
    try:
        import pulp
    except ImportError as exc:
        raise ImportError(
            "PuLP is required for MILP backbone search. Install it with "
            "`python -m pip install pulp`."
        ) from exc

    n = len(data.excitatory_nodes)
    edges = [
        (i, j, float(data.transition_weights[i, j]))
        for i in range(n)
        for j in range(n)
        if i != j and data.transition_weights[i, j] > EPS
    ]
    incoming = {i: [] for i in range(n)}
    outgoing = {i: [] for i in range(n)}
    for i, j, _ in edges:
        outgoing[i].append((i, j))
        incoming[j].append((i, j))

    problem = pulp.LpProblem("max_weight_simple_directed_path", pulp.LpMaximize)
    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1, cat="Binary")
        for i, j, _ in edges
    }
    y = {
        i: pulp.LpVariable(f"y_{i}", lowBound=0, upBound=1, cat="Binary")
        for i in range(n)
    }
    order = {
        i: pulp.LpVariable(f"u_{i}", lowBound=0, upBound=n - 1, cat="Continuous")
        for i in range(n)
    }

    problem += (
        pulp.lpSum(w * x[(i, j)] for i, j, w in edges)
        + pulp.lpSum(float(data.self_weights[i]) * y[i] for i in range(n))
    )

    problem += y[seed] == 1
    problem += pulp.lpSum(x[e] for e in incoming[seed]) == 0
    problem += order[seed] == 0

    for i in range(n):
        problem += pulp.lpSum(x[e] for e in outgoing[i]) <= y[i]
        if i == seed:
            continue
        problem += pulp.lpSum(x[e] for e in incoming[i]) == y[i]
        problem += order[i] >= y[i]
        problem += order[i] <= (n - 1) * y[i]

    big_m = n
    for i, j, _ in edges:
        problem += order[j] >= order[i] + 1 - big_m * (1 - x[(i, j)])

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
    problem.solve(solver)

    status = pulp.LpStatus.get(problem.status, str(problem.status))
    selected_edges = {
        i: j
        for (i, j), var in x.items()
        if var.value() is not None and var.value() > 0.5
    }

    path = [seed]
    visited = {seed}
    while path[-1] in selected_edges:
        nxt = selected_edges[path[-1]]
        if nxt in visited:
            raise ValueError("MILP returned a cycle despite subtour constraints.")
        path.append(nxt)
        visited.add(nxt)

    objective = float(pulp.value(problem.objective) or 0.0)
    return path, total_score(path, data.transition_weights, data.self_weights), status, objective


def maximum_weight_asymmetric_hamiltonian_path(
    seed: int,
    data: BackboneData,
    time_limit: Optional[float] = 30.0,
    msg: bool = False,
) -> Tuple[List[int], float, str, float]:
    try:
        import pulp
    except ImportError as exc:
        raise ImportError(
            "PuLP is required for Hamiltonian path search. Install it with "
            "`python -m pip install pulp`."
        ) from exc

    n = len(data.excitatory_nodes)
    edges = [
        (i, j, float(data.transition_weights[i, j]))
        for i in range(n)
        for j in range(n)
        if i != j and data.transition_weights[i, j] > EPS
    ]
    incoming = {i: [] for i in range(n)}
    outgoing = {i: [] for i in range(n)}
    for i, j, _ in edges:
        outgoing[i].append((i, j))
        incoming[j].append((i, j))

    problem = pulp.LpProblem("max_weight_asymmetric_hamiltonian_path", pulp.LpMaximize)
    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1, cat="Binary")
        for i, j, _ in edges
    }
    order = {
        i: pulp.LpVariable(f"u_{i}", lowBound=0, upBound=n - 1, cat="Continuous")
        for i in range(n)
    }

    problem += pulp.lpSum(w * x[(i, j)] for i, j, w in edges)
    problem += pulp.lpSum(x[e] for e in incoming[seed]) == 0
    if n > 1:
        problem += pulp.lpSum(x[e] for e in outgoing[seed]) == 1
    problem += order[seed] == 0
    problem += pulp.lpSum(x[(i, j)] for i, j, _ in edges) == n - 1

    for i in range(n):
        problem += pulp.lpSum(x[e] for e in outgoing[i]) <= 1
        if i == seed:
            continue
        problem += pulp.lpSum(x[e] for e in incoming[i]) == 1
        problem += order[i] >= 1

    big_m = n
    for i, j, _ in edges:
        problem += order[j] >= order[i] + 1 - big_m * (1 - x[(i, j)])

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
    problem.solve(solver)

    status = pulp.LpStatus.get(problem.status, str(problem.status))
    selected_edges = {
        i: j
        for (i, j), var in x.items()
        if var.value() is not None and var.value() > 0.5
    }

    path = [seed]
    visited = {seed}
    while path[-1] in selected_edges:
        nxt = selected_edges[path[-1]]
        if nxt in visited:
            raise ValueError(
                "Hamiltonian MILP returned a cycle despite subtour constraints."
            )
        path.append(nxt)
        visited.add(nxt)

    objective = float(pulp.value(problem.objective) or 0.0)
    return path, total_score(path, data.transition_weights, data.self_weights), status, objective


def run_all_methods(
    data: BackboneData,
    bb_time_limit_per_seed: float = 0.25,
    bb_node_limit_per_seed: int = 50_000,
    exact_dp_max_e_nodes: int = 22,
    beam_width: int = 5_000,
    milp_time_limit_per_seed: Optional[float] = 30.0,
    hamiltonian_time_limit_per_seed: Optional[float] = 30.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(data.excitatory_nodes)
    results = []
    edges = []

    for seed in range(n):
        path = greedy_path(seed, data)
        method = "greedy_tree_path"
        results.append(path_to_records(seed, method, path, data))
        edges.extend(edges_from_path(seed, method, path, data))

    mst_edges = maximum_spanning_tree_edges(data.transition_weights)
    tree_adj = build_tree_adjacency(n, mst_edges)
    for seed in range(n):
        path = best_mst_path_from_seed(seed, tree_adj, data)
        results.append(path_to_records(seed, "maximum_spanning_tree", path, data))
        edges.extend(edges_from_path(seed, "maximum_spanning_tree", path, data))

    for seed in range(n):
        path, _, states, status = branch_bound_path(
            seed,
            data,
            time_limit=bb_time_limit_per_seed,
            node_limit=bb_node_limit_per_seed,
        )
        results.append(
            path_to_records(
                seed,
                "branch_and_bound",
                path,
                data,
                {"bb_states_seen": states, "bb_status": status},
            )
        )
        edges.extend(edges_from_path(seed, "branch_and_bound", path, data))

    for seed in range(n):
        if n <= exact_dp_max_e_nodes:
            path, _, mode = exact_bitmask_dp_path(seed, data)
        else:
            path, _, mode = beam_dp_path(seed, data, beam_width=beam_width)
        results.append(
            path_to_records(seed, "dynamic_programming", path, data, {"dp_mode": mode})
        )
        edges.extend(edges_from_path(seed, "dynamic_programming", path, data))

    for seed in range(n):
        path, _, status, objective = milp_subtour_elimination_path(
            seed,
            data,
            time_limit=milp_time_limit_per_seed,
        )
        results.append(
            path_to_records(
                seed,
                "milp_subtour_elimination",
                path,
                data,
                {"milp_status": status, "milp_objective": objective},
            )
        )
        edges.extend(edges_from_path(seed, "milp_subtour_elimination", path, data))

    for seed in range(n):
        path, _, status, objective = maximum_weight_asymmetric_hamiltonian_path(
            seed,
            data,
            time_limit=hamiltonian_time_limit_per_seed,
        )
        results.append(
            path_to_records(
                seed,
                "maximum_weight_asymmetric_hamiltonian_path",
                path,
                data,
                {
                    "hamiltonian_status": status,
                    "hamiltonian_objective": objective,
                },
            )
        )
        edges.extend(
            edges_from_path(
                seed,
                "maximum_weight_asymmetric_hamiltonian_path",
                path,
                data,
            )
        )

    return pd.DataFrame(results), pd.DataFrame(edges)


def build_wide_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    values = [
        "length_edges",
        "n_nodes_visited",
        "transition_weight_sum",
        "summed_weight",
        "terminus",
    ]
    for optional in ["self_weight_sum", "n_self_connections_included"]:
        if optional in comparison.columns:
            values.append(optional)
    wide = comparison.pivot_table(
        index="seed",
        columns="method",
        values=values,
        aggfunc="first",
    )
    wide.columns = [f"{metric}__{method}" for metric, method in wide.columns]
    return wide.reset_index()


def summarize_methods(comparison: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "n_seeds": ("seed", "nunique"),
        "mean_length": ("length_edges", "mean"),
        "median_length": ("length_edges", "median"),
        "mean_transition_weight": ("transition_weight_sum", "mean"),
        "mean_summed_weight": ("summed_weight", "mean"),
        "median_summed_weight": ("summed_weight", "median"),
        "max_summed_weight": ("summed_weight", "max"),
        "n_unique_termini": ("terminus", "nunique"),
    }
    if "self_weight_sum" in comparison.columns:
        aggregations["mean_self_weight"] = ("self_weight_sum", "mean")
    if "n_self_connections_included" in comparison.columns:
        aggregations["mean_self_connections_included"] = (
            "n_self_connections_included",
            "mean",
        )
    return (
        comparison.groupby("method", dropna=False)
        .agg(**aggregations)
        .reset_index()
    )


def build_disagreement_table(comparison: pd.DataFrame) -> pd.DataFrame:
    bb = comparison[comparison["method"] == "branch_and_bound"].set_index("seed")
    dp = comparison[comparison["method"] == "dynamic_programming"].set_index("seed")
    disagreement = pd.DataFrame(
        {
            "bb_weight": bb["summed_weight"],
            "dp_weight": dp["summed_weight"],
            "bb_length": bb["length_edges"],
            "dp_length": dp["length_edges"],
            "bb_terminus": bb["terminus"],
            "dp_terminus": dp["terminus"],
        }
    )
    disagreement["weight_difference_dp_minus_bb"] = (
        disagreement["dp_weight"] - disagreement["bb_weight"]
    )
    return disagreement[
        disagreement["weight_difference_dp_minus_bb"].abs() > 1e-9
    ]


def build_milp_benchmark_table(comparison: pd.DataFrame) -> pd.DataFrame:
    if "milp_subtour_elimination" not in set(comparison["method"]):
        return pd.DataFrame()

    milp = comparison[comparison["method"] == "milp_subtour_elimination"].set_index("seed")
    rows = []
    for _, row in comparison.iterrows():
        if row["method"] == "milp_subtour_elimination":
            continue
        seed = row["seed"]
        if seed not in milp.index:
            continue
        milp_row = milp.loc[seed]
        rows.append(
            {
                "seed": seed,
                "method": row["method"],
                "method_summed_weight": row["summed_weight"],
                "milp_summed_weight": milp_row["summed_weight"],
                "weight_gap_to_milp": milp_row["summed_weight"] - row["summed_weight"],
                "method_length_edges": row["length_edges"],
                "milp_length_edges": milp_row["length_edges"],
                "method_terminus": row["terminus"],
                "milp_terminus": milp_row["terminus"],
                "milp_status": milp_row.get("milp_status", ""),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["seed", "method"]).reset_index(drop=True)


def write_setup_outputs(data: BackboneData, outdir: Path) -> None:
    outdir.mkdir(exist_ok=True)
    data.node_sign_summary.to_csv(outdir / "node_sign_summary.csv", index=False)
    pd.Series(data.excitatory_nodes, name="E_node").to_csv(
        outdir / "excitatory_nodes_used.csv", index=False
    )
    if data.include_self_connections:
        data.self_table.to_csv(outdir / "self_connections_used.csv", index=False)


def write_comparison_outputs(
    comparison: pd.DataFrame,
    edge_table: pd.DataFrame,
    outdir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    outdir.mkdir(exist_ok=True)
    wide = build_wide_comparison(comparison)
    method_summary = summarize_methods(comparison)
    disagreement = build_disagreement_table(comparison)
    milp_benchmark = build_milp_benchmark_table(comparison)

    comparison.to_csv(outdir / "method_comparison_long.csv", index=False)
    wide.to_csv(outdir / "method_comparison_wide.csv", index=False)
    edge_table.to_csv(outdir / "all_methods_edges.csv", index=False)
    method_summary.to_csv(outdir / "method_summary.csv", index=False)
    disagreement.to_csv(outdir / "bb_dp_disagreements.csv")
    if not milp_benchmark.empty:
        milp_benchmark.to_csv(outdir / "milp_benchmark_gaps.csv", index=False)

    return wide, method_summary
