"""Feedback-arc-minimizing backbone ordering for directed weighted connectomes.

This module implements the ordering approach of Vahidi (2025), "Feedforward Ordering
in Neural Connectomes via Feedback Arc Minimization" (arXiv:2506.13799), which was
developed and evaluated on the FlyWire connectome. Rather than selecting a single
path, it searches for a vertex ordering that maximizes the total weight of
forward-pointing edges, i.e. minimizes the weight of the feedback arc set.

Why this differs from the path methods in ``ee_backbone_analysis``
------------------------------------------------------------------
A maximum-weight Hamiltonian path over 32 nodes scores the 31 edges it selects and
ignores the remaining positive edges. A feedback-arc ordering scores *every* edge
against the ordering, so it answers "what is the dominant direction of excitatory
flow through these neuron types" rather than "what is the single heaviest walk".

Two structural facts make this tractable and are used directly below.

1. The problem decomposes exactly over strongly connected components. Edges between
   distinct SCCs can all be made forward simultaneously by ordering the SCCs
   topologically in the condensation, so no optimal ordering ever pays for them.
   Feedback weight is therefore incurred only *within* SCCs, and each SCC can be
   solved independently. This is exact, not a heuristic.

2. Once an ordering is fixed, the subgraph of forward edges is acyclic. The
   maximum-weight simple path from any seed within that DAG is then solvable exactly
   in O(V + E) by dynamic programming over the ordering, because a forward walk can
   never revisit a node. The NP-hardness of the unrestricted simple-path problem is
   an artifact of allowing backward steps.

Ordering search follows the three ingredients Vahidi combines: a greedy
construction (a weighted variant of the Eades-Lin-Smyth GR heuristic), gain-aware
local refinement (best-position reinsertion), and SCC-based global decomposition.

References
----------
Vahidi, S. (2025). Feedforward Ordering in Neural Connectomes via Feedback Arc
    Minimization. arXiv:2506.13799. https://doi.org/10.48550/arXiv.2506.13799
Eades, P., Lin, X., & Smyth, W. F. (1993). A fast and effective heuristic for the
    feedback arc set problem. Information Processing Letters, 47(6), 319-323.
    https://doi.org/10.1016/0020-0190(93)90079-O
Borst, A. (2024). Connectivity Matrix Seriation via Relaxation. PLOS Computational
    Biology, 20(2), e1011904. https://doi.org/10.1371/journal.pcbi.1011904
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ee_backbone_analysis import BackboneData, EPS, total_score


@dataclass
class OrderingResult:
    """A vertex ordering plus the diagnostics needed to interpret it."""

    order: List[int]
    labels: List[str]
    metrics: Dict[str, float]
    scc_sizes: List[int] = field(default_factory=list)
    refinement_passes: int = 0
    n_restarts: int = 0

    @property
    def ordered_labels(self) -> List[str]:
        return [self.labels[i] for i in self.order]

    def position_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "position": np.arange(len(self.order)),
                "node": self.ordered_labels,
                "node_index": self.order,
            }
        )


# ---------------------------------------------------------------------------
# Strongly connected components and condensation
# ---------------------------------------------------------------------------


def strongly_connected_components(
    weights: np.ndarray,
    eps: float = EPS,
) -> List[List[int]]:
    """Tarjan's algorithm, iterative, returned in reverse topological order.

    Reverse topological order is what Tarjan produces naturally: every component is
    emitted only after all components it can reach. Reversing the returned list
    therefore gives a valid topological order of the condensation.
    """
    n = weights.shape[0]
    successors = [np.flatnonzero(weights[i] > eps).tolist() for i in range(n)]

    index_of: Dict[int, int] = {}
    low_of: Dict[int, int] = {}
    on_stack = [False] * n
    stack: List[int] = []
    components: List[List[int]] = []
    counter = 0

    for root in range(n):
        if root in index_of:
            continue
        # (node, iterator position into successors[node])
        work: List[Tuple[int, int]] = [(root, 0)]
        index_of[root] = low_of[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True

        while work:
            node, next_child = work[-1]
            if next_child < len(successors[node]):
                work[-1] = (node, next_child + 1)
                child = successors[node][next_child]
                if child not in index_of:
                    index_of[child] = low_of[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack[child] = True
                    work.append((child, 0))
                elif on_stack[child]:
                    low_of[node] = min(low_of[node], index_of[child])
            else:
                work.pop()
                if work:
                    parent = work[-1][0]
                    low_of[parent] = min(low_of[parent], low_of[node])
                if low_of[node] == index_of[node]:
                    component = []
                    while True:
                        member = stack.pop()
                        on_stack[member] = False
                        component.append(member)
                        if member == node:
                            break
                    components.append(component)

    return components


def condensation_topological_order(
    components: Sequence[Sequence[int]],
    weights: np.ndarray,
    eps: float = EPS,
) -> List[List[int]]:
    """Order SCCs so that every inter-component edge points forward.

    Tarjan emits components in reverse topological order, so reversing is already
    valid. The Kahn pass below is kept as an explicit, checkable construction and
    breaks ties by net outward weight, which keeps heavier sources earlier.
    """
    n_components = len(components)
    membership = np.empty(weights.shape[0], dtype=int)
    for c_index, component in enumerate(components):
        for node in component:
            membership[node] = c_index

    out_edges: Dict[int, set] = {c: set() for c in range(n_components)}
    in_degree = [0] * n_components
    senders, receivers = np.nonzero(weights > eps)
    for sender, receiver in zip(senders.tolist(), receivers.tolist()):
        a, b = membership[sender], membership[receiver]
        if a != b and b not in out_edges[a]:
            out_edges[a].add(b)
            in_degree[b] += 1

    net_out = np.zeros(n_components)
    for sender, receiver in zip(senders.tolist(), receivers.tolist()):
        a, b = membership[sender], membership[receiver]
        if a != b:
            net_out[a] += weights[sender, receiver]
            net_out[b] -= weights[sender, receiver]

    ready = [c for c in range(n_components) if in_degree[c] == 0]
    ordered: List[int] = []
    while ready:
        ready.sort(key=lambda c: net_out[c], reverse=True)
        current = ready.pop(0)
        ordered.append(current)
        for target in out_edges[current]:
            in_degree[target] -= 1
            if in_degree[target] == 0:
                ready.append(target)

    if len(ordered) != n_components:  # pragma: no cover - condensation is acyclic
        raise RuntimeError("Condensation contained a cycle; SCC computation is wrong.")

    return [list(components[c]) for c in ordered]


# ---------------------------------------------------------------------------
# Greedy construction and local refinement
# ---------------------------------------------------------------------------


def eades_lin_smyth_ordering(
    weights: np.ndarray,
    nodes: Sequence[int],
    eps: float = EPS,
) -> List[int]:
    """Weighted GR heuristic for the feedback arc set, restricted to ``nodes``.

    Repeatedly strip sinks to the right sequence and sources to the left sequence;
    when neither exists, move the vertex maximizing (outgoing weight - incoming
    weight) to the left sequence. The weighted variant uses summed edge weight in
    place of degree, which is what the maximum-weight-forward objective requires.
    """
    remaining = set(int(v) for v in nodes)
    left: List[int] = []
    right: List[int] = []

    sub = {v: {u: float(weights[v, u]) for u in remaining if u != v and weights[v, u] > eps}
           for v in remaining}
    rev = {v: {u: float(weights[u, v]) for u in remaining if u != v and weights[u, v] > eps}
           for v in remaining}

    def drop(node: int) -> None:
        for target in list(sub[node]):
            rev[target].pop(node, None)
        for source in list(rev[node]):
            sub[source].pop(node, None)
        sub.pop(node, None)
        rev.pop(node, None)
        remaining.discard(node)

    while remaining:
        moved = True
        while moved and remaining:
            moved = False
            for node in list(remaining):
                if not sub.get(node):  # sink
                    right.append(node)
                    drop(node)
                    moved = True
            for node in list(remaining):
                if not rev.get(node):  # source
                    left.append(node)
                    drop(node)
                    moved = True
        if not remaining:
            break
        best_node = max(
            remaining,
            key=lambda v: sum(sub[v].values()) - sum(rev[v].values()),
        )
        left.append(best_node)
        drop(best_node)

    return left + right[::-1]


def gain_aware_refinement(
    order: List[int],
    weights: np.ndarray,
    max_passes: int = 50,
) -> Tuple[List[int], int]:
    """Best-position reinsertion until no single move improves forward weight.

    Moving a node left past a node ``u`` flips the pair from (u before v) to
    (v before u), changing forward weight by ``W[v,u] - W[u,v]``. Scanning the
    cumulative sum of those deltas in both directions gives the best reinsertion
    position for one node in O(n); a full pass is O(n^2).
    """
    order = list(order)
    n = len(order)
    if n < 3:
        return order, 0

    passes = 0
    for _ in range(max_passes):
        passes += 1
        improved = False
        for position in range(n):
            node = order[position]

            # Gains from moving the node left across order[j] for j = position-1 .. 0
            running = 0.0
            best_gain = 0.0
            best_target = position
            for j in range(position - 1, -1, -1):
                other = order[j]
                running += float(weights[node, other]) - float(weights[other, node])
                if running > best_gain + EPS:
                    best_gain = running
                    best_target = j

            # Gains from moving the node right across order[j] for j = position+1 .. n-1
            running = 0.0
            for j in range(position + 1, n):
                other = order[j]
                running += float(weights[other, node]) - float(weights[node, other])
                if running > best_gain + EPS:
                    best_gain = running
                    best_target = j

            if best_target != position:
                order.pop(position)
                order.insert(best_target, node)
                improved = True
        if not improved:
            break

    return order, passes


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def ordering_metrics(
    order: Sequence[int],
    weights: np.ndarray,
    eps: float = EPS,
) -> Dict[str, float]:
    """Forward/backward edge weight and count for a given ordering."""
    position = np.empty(len(order), dtype=int)
    for rank, node in enumerate(order):
        position[node] = rank

    senders, receivers = np.nonzero(weights > eps)
    values = weights[senders, receivers]
    forward = position[senders] < position[receivers]

    forward_weight = float(values[forward].sum())
    backward_weight = float(values[~forward].sum())
    total = forward_weight + backward_weight

    return {
        "n_edges": int(len(values)),
        "n_forward_edges": int(forward.sum()),
        "n_feedback_edges": int((~forward).sum()),
        "forward_weight": forward_weight,
        "feedback_weight": backward_weight,
        "total_weight": total,
        "forward_weight_fraction": forward_weight / total if total > 0 else float("nan"),
        "feedback_weight_fraction": backward_weight / total if total > 0 else float("nan"),
    }


def permutation_null_forward_fraction(
    weights: np.ndarray,
    observed_fraction: float,
    n_null: int = 1_000,
    random_seed: int = 2026,
    eps: float = EPS,
) -> Dict[str, float]:
    """Compare the ordering's forward-weight fraction against random orderings.

    Uses the empirical p-value ``(1 + #{null >= observed}) / (1 + n_null)`` so the
    p-value is never exactly zero, matching the convention already prescribed in
    ``ee_backbone_deterministic_stochastic_comparison.ipynb``.
    """
    rng = np.random.default_rng(random_seed)
    n = weights.shape[0]
    fractions = np.empty(n_null)
    for draw in range(n_null):
        fractions[draw] = ordering_metrics(
            rng.permutation(n).tolist(), weights, eps=eps
        )["forward_weight_fraction"]

    n_at_least = int(np.sum(fractions >= observed_fraction))
    return {
        "null_n": n_null,
        "null_mean_forward_fraction": float(fractions.mean()),
        "null_sd_forward_fraction": float(fractions.std(ddof=1)),
        "null_max_forward_fraction": float(fractions.max()),
        "observed_percentile_in_null": float(100.0 * np.mean(fractions < observed_fraction)),
        "empirical_p_forward_fraction": (1 + n_at_least) / (1 + n_null),
    }


# ---------------------------------------------------------------------------
# Top-level ordering search
# ---------------------------------------------------------------------------


def feedback_arc_ordering(
    data: BackboneData,
    max_refinement_passes: int = 50,
    use_scc_decomposition: bool = True,
    n_restarts: int = 20,
    random_seed: int = 2026,
    eps: float = EPS,
) -> OrderingResult:
    """Search for a vertex ordering maximizing total forward-pointing edge weight.

    Combines SCC decomposition (exact), a weighted Eades-Lin-Smyth greedy
    construction, and gain-aware local refinement, following Vahidi (2025).

    Refinement converges to a local optimum of the reinsertion neighbourhood, so the
    greedy construction is supplemented with ``n_restarts`` randomized starts and the
    best ordering is kept. At n = 32 this costs a fraction of a second; set
    ``n_restarts=0`` to use the greedy construction alone.
    """
    weights = data.transition_weights
    n = weights.shape[0]
    rng = np.random.default_rng(random_seed)

    if use_scc_decomposition:
        components = strongly_connected_components(weights, eps=eps)
        blocks = condensation_topological_order(components, weights, eps=eps)
    else:
        blocks = [list(range(n))]

    order: List[int] = []
    total_passes = 0
    for block in blocks:
        if len(block) == 1:
            order.extend(block)
            continue

        candidates: List[List[int]] = [eades_lin_smyth_ordering(weights, block, eps=eps)]
        for _ in range(max(0, n_restarts)):
            candidates.append(rng.permutation(np.asarray(block)).tolist())

        best_block: Optional[List[int]] = None
        best_forward = -np.inf
        for candidate in candidates:
            refined, passes = _refine_block(
                candidate, weights, max_passes=max_refinement_passes
            )
            total_passes = max(total_passes, passes)
            forward = _block_forward_weight(refined, weights)
            if forward > best_forward + eps:
                best_forward = forward
                best_block = refined

        order.extend(best_block if best_block is not None else block)

    metrics = ordering_metrics(order, weights, eps=eps)
    return OrderingResult(
        order=order,
        labels=list(data.excitatory_nodes),
        metrics=metrics,
        scc_sizes=[len(block) for block in blocks],
        refinement_passes=total_passes,
        n_restarts=n_restarts,
    )


def _block_forward_weight(block_order: Sequence[int], weights: np.ndarray) -> float:
    """Forward edge weight among the members of one block, in the given order."""
    total = 0.0
    for rank, node in enumerate(block_order):
        for other in block_order[rank + 1 :]:
            total += float(weights[node, other])
    return total


def _refine_block(
    block_order: List[int],
    weights: np.ndarray,
    max_passes: int,
) -> Tuple[List[int], int]:
    """Refine within a block, keeping the block's members contiguous."""
    members = list(block_order)
    index_of = {node: k for k, node in enumerate(members)}
    sub_weights = np.zeros((len(members), len(members)))
    for a in members:
        for b in members:
            if a != b:
                sub_weights[index_of[a], index_of[b]] = weights[a, b]

    local_order = [index_of[node] for node in members]
    local_order, passes = gain_aware_refinement(
        local_order, sub_weights, max_passes=max_passes
    )
    return [members[k] for k in local_order], passes


# ---------------------------------------------------------------------------
# Exact maximum-weight path within a fixed ordering
# ---------------------------------------------------------------------------


def longest_forward_path_from_seed(
    seed: int,
    order: Sequence[int],
    data: BackboneData,
    eps: float = EPS,
) -> Tuple[List[int], float]:
    """Exact maximum-weight simple path from ``seed`` using only forward edges.

    Forward edges under a fixed ordering form a DAG, so a single sweep in ordering
    order solves this exactly. Unlike the branch-and-bound and beam-DP methods this
    carries a genuine optimality guarantee, conditional on the ordering.
    """
    weights = data.transition_weights
    n = weights.shape[0]
    rank = np.empty(n, dtype=int)
    for position, node in enumerate(order):
        rank[node] = position

    best = np.full(n, -math.inf)
    predecessor = np.full(n, -1, dtype=int)
    best[seed] = float(data.self_weights[seed])

    for node in order:
        if best[node] == -math.inf:
            continue
        for target, weight in data.adjacency[node]:
            if rank[target] <= rank[node]:
                continue
            candidate = best[node] + weight + float(data.self_weights[target])
            if candidate > best[target] + eps:
                best[target] = candidate
                predecessor[target] = node

    terminal = int(np.argmax(np.where(np.isfinite(best), best, -math.inf)))
    path = [terminal]
    while predecessor[path[-1]] != -1:
        path.append(int(predecessor[path[-1]]))
    path.reverse()
    return path, float(best[terminal])


def ordering_edge_table(
    result: OrderingResult,
    data: BackboneData,
    eps: float = EPS,
) -> pd.DataFrame:
    """Every positive edge labelled forward or feedback under the ordering."""
    weights = data.transition_weights
    rank = np.empty(len(result.order), dtype=int)
    for position, node in enumerate(result.order):
        rank[node] = position

    senders, receivers = np.nonzero(weights > eps)
    labels = result.labels
    return pd.DataFrame(
        {
            "sender": [labels[i] for i in senders],
            "receiver": [labels[j] for j in receivers],
            "sender_position": rank[senders],
            "receiver_position": rank[receivers],
            "displacement": rank[receivers] - rank[senders],
            "weight": weights[senders, receivers],
            "direction": np.where(
                rank[senders] < rank[receivers], "forward", "feedback"
            ),
        }
    ).sort_values("weight", ascending=False).reset_index(drop=True)


def ordering_path_records(
    result: OrderingResult,
    data: BackboneData,
    method: str = "feedback_arc_ordering",
) -> Tuple[List[dict], List[dict]]:
    """Per-seed records shaped like the other methods' output, for comparison."""
    from ee_backbone_analysis import edges_from_path, path_to_records

    records: List[dict] = []
    edge_rows: List[dict] = []
    for seed in range(len(data.excitatory_nodes)):
        path, score = longest_forward_path_from_seed(seed, result.order, data)
        records.append(
            path_to_records(
                seed,
                method,
                path,
                data,
                {
                    "ordering_dag_score": score,
                    "ordering_exact_within_ordering": True,
                    "ordering_forward_weight_fraction": result.metrics[
                        "forward_weight_fraction"
                    ],
                    "seed_position": int(result.order.index(seed)),
                },
            )
        )
        edge_rows.extend(edges_from_path(seed, method, path, data))
    return records, edge_rows
