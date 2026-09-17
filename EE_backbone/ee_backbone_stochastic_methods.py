from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ee_backbone_analysis import BackboneData, EPS, total_score


def _transition_probabilities(
    candidates: List[Tuple[int, float]],
    rng_mode: str,
    temperature: float = 0.25,
    weight_power: float = 1.0,
) -> np.ndarray:
    weights = np.asarray([max(weight, EPS) for _, weight in candidates], dtype=float)

    if rng_mode == "linear":
        scaled = weights ** weight_power
        return scaled / scaled.sum()

    if rng_mode == "boltzmann":
        if len(weights) == 1:
            return np.ones(1)
        spread = weights.max() - weights.min()
        if spread <= EPS:
            return np.full(len(weights), 1.0 / len(weights))
        normalized = (weights - weights.min()) / spread
        logits = normalized / max(temperature, EPS)
        logits = logits - logits.max()
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum()

    raise ValueError(f"Unknown rng_mode: {rng_mode}")


def _sample_simple_path(
    seed: int,
    data: BackboneData,
    rng: np.random.Generator,
    rng_mode: str,
    initial_path: List[int] | None = None,
    max_steps: int | None = None,
    stop_probability: float = 0.0,
    temperature: float = 0.25,
    weight_power: float = 1.0,
) -> List[int]:
    n = len(data.excitatory_nodes)
    limit = max_steps if max_steps is not None else n - 1
    path = initial_path.copy() if initial_path is not None else [seed]
    visited = {seed}
    visited.update(path)

    while len(path) - 1 < limit:
        candidates = [(node, weight) for node, weight in data.adjacency[path[-1]] if node not in visited]
        if not candidates:
            break
        if len(path) > 1 and rng.random() < stop_probability:
            break
        probabilities = _transition_probabilities(
            candidates,
            rng_mode=rng_mode,
            temperature=temperature,
            weight_power=weight_power,
        )
        next_index = int(rng.choice(len(candidates), p=probabilities))
        next_node = candidates[next_index][0]
        path.append(next_node)
        visited.add(next_node)

    return path


def _path_diagnostics(
    seed: int,
    method: str,
    paths: List[List[int]],
    scores: np.ndarray,
    data: BackboneData,
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    best_idx = int(np.argmax(scores))
    best_path = paths[best_idx]
    edge_counts: Counter[Tuple[int, int]] = Counter()
    terminal_counts: Counter[int] = Counter()

    for path in paths:
        terminal_counts[path[-1]] += 1
        edge_counts.update(zip(path[:-1], path[1:]))

    n_samples = len(paths)
    terminal_probabilities = np.asarray(list(terminal_counts.values()), dtype=float) / n_samples
    terminal_entropy = float(-(terminal_probabilities * np.log2(terminal_probabilities)).sum())

    edge_rows = [
        {
            "method": method,
            "seed": data.excitatory_nodes[seed],
            "sender": data.excitatory_nodes[sender],
            "receiver": data.excitatory_nodes[receiver],
            "visit_count": count,
            "visit_probability": count / n_samples,
            "weight": float(data.transition_weights[sender, receiver]),
        }
        for (sender, receiver), count in edge_counts.items()
    ]
    terminal_rows = [
        {
            "method": method,
            "seed": data.excitatory_nodes[seed],
            "terminal": data.excitatory_nodes[terminal],
            "count": count,
            "probability": count / n_samples,
        }
        for terminal, count in terminal_counts.items()
    ]

    diagnostics = {
        "best_sample_score": float(scores[best_idx]),
        "mean_sample_score": float(scores.mean()),
        "median_sample_score": float(np.median(scores)),
        "sample_score_std": float(scores.std(ddof=1)) if n_samples > 1 else 0.0,
        "terminal_entropy_bits": terminal_entropy,
        "top_edge_visit_probability": max((row["visit_probability"] for row in edge_rows), default=0.0),
        "n_unique_terminals_sampled": len(terminal_counts),
        "n_unique_edges_sampled": len(edge_counts),
    }
    return diagnostics, pd.DataFrame(edge_rows), pd.DataFrame(terminal_rows)


def weighted_random_walk_path_ensemble(
    seed: int,
    data: BackboneData,
    n_walks: int = 2_000,
    weight_power: float = 1.0,
    stop_probability: float = 0.05,
    random_seed: int = 2026,
) -> Tuple[List[int], dict, pd.DataFrame, pd.DataFrame]:
    """Sample simple paths using locally weight-proportional transition probabilities."""
    rng = np.random.default_rng(random_seed + seed)
    paths = [
        _sample_simple_path(
            seed,
            data,
            rng,
            rng_mode="linear",
            stop_probability=stop_probability,
            weight_power=weight_power,
        )
        for _ in range(n_walks)
    ]
    scores = np.asarray(
        [total_score(path, data.transition_weights, data.self_weights) for path in paths],
        dtype=float,
    )
    diagnostics, edge_frequency, terminal_distribution = _path_diagnostics(
        seed,
        "weighted_random_walk_ensemble",
        paths,
        scores,
        data,
    )
    diagnostics.update(
        {
            "n_walks": n_walks,
            "weight_power": weight_power,
            "stop_probability": stop_probability,
        }
    )
    return paths[int(np.argmax(scores))], diagnostics, edge_frequency, terminal_distribution


def boltzmann_path_sampling(
    seed: int,
    data: BackboneData,
    n_samples: int = 2_000,
    temperature: float = 0.25,
    stop_probability: float = 0.02,
    random_seed: int = 404,
) -> Tuple[List[int], dict, pd.DataFrame, pd.DataFrame]:
    """Sample simple paths using a softmax over outgoing edge weights."""
    rng = np.random.default_rng(random_seed + seed)
    paths = [
        _sample_simple_path(
            seed,
            data,
            rng,
            rng_mode="boltzmann",
            stop_probability=stop_probability,
            temperature=temperature,
        )
        for _ in range(n_samples)
    ]
    scores = np.asarray(
        [total_score(path, data.transition_weights, data.self_weights) for path in paths],
        dtype=float,
    )
    diagnostics, edge_frequency, terminal_distribution = _path_diagnostics(
        seed,
        "boltzmann_path_sampling",
        paths,
        scores,
        data,
    )
    diagnostics.update(
        {
            "n_samples": n_samples,
            "temperature": temperature,
            "stop_probability": stop_probability,
        }
    )
    return paths[int(np.argmax(scores))], diagnostics, edge_frequency, terminal_distribution


def monte_carlo_tree_search_path(
    seed: int,
    data: BackboneData,
    iterations: int = 1_500,
    exploration_weight: float = 1.4,
    rollout_temperature: float = 0.35,
    random_seed: int = 777,
) -> Tuple[List[int], dict]:
    """Use UCT-style Monte Carlo tree search over valid simple directed paths."""
    rng = np.random.default_rng(random_seed + seed)
    visits: Dict[Tuple[int, ...], int] = defaultdict(int)
    total_rewards: Dict[Tuple[int, ...], float] = defaultdict(float)
    expanded_children: Dict[Tuple[int, ...], set[int]] = defaultdict(set)

    root = (seed,)
    best_path = [seed]
    best_score = total_score(best_path, data.transition_weights, data.self_weights)
    reward_scale = max(float(data.transition_weights.max()), EPS)

    def legal_children(state: Tuple[int, ...]) -> List[int]:
        visited = set(state)
        return [node for node, _ in data.adjacency[state[-1]] if node not in visited]

    for _ in range(iterations):
        state = root
        search_trace = [state]

        while True:
            children = legal_children(state)
            if not children:
                break

            unexpanded = [child for child in children if child not in expanded_children[state]]
            if unexpanded:
                weights = np.asarray(
                    [data.transition_weights[state[-1], child] for child in unexpanded],
                    dtype=float,
                )
                probabilities = weights / weights.sum()
                child = int(rng.choice(unexpanded, p=probabilities))
                expanded_children[state].add(child)
                state = state + (child,)
                search_trace.append(state)
                break

            parent_visits = max(1, visits[state])
            scored_children = []
            for child in children:
                child_state = state + (child,)
                child_visits = visits[child_state]
                if child_visits == 0:
                    uct = math.inf
                else:
                    mean_reward = total_rewards[child_state] / child_visits
                    uct = mean_reward + exploration_weight * math.sqrt(
                        math.log(parent_visits + 1) / child_visits
                    )
                scored_children.append((uct, child))
            max_uct = max(score for score, _ in scored_children)
            tied = [child for score, child in scored_children if score == max_uct]
            child = int(rng.choice(tied))
            state = state + (child,)
            search_trace.append(state)

        rollout = _sample_simple_path(
            state[-1],
            data,
            rng,
            rng_mode="boltzmann",
            initial_path=list(state),
            stop_probability=0.0,
            temperature=rollout_temperature,
        )
        completed_path = rollout
        reward = total_score(completed_path, data.transition_weights, data.self_weights)
        search_reward = reward / reward_scale

        if reward > best_score or (abs(reward - best_score) <= EPS and len(completed_path) > len(best_path)):
            best_score = reward
            best_path = completed_path

        for traced_state in search_trace:
            visits[traced_state] += 1
            total_rewards[traced_state] += search_reward

    diagnostics = {
        "mcts_iterations": iterations,
        "mcts_exploration_weight": exploration_weight,
        "mcts_rollout_temperature": rollout_temperature,
        "mcts_states_visited": len(visits),
        "mcts_best_score": best_score,
    }
    return best_path, diagnostics


def combine_frequency_tables(
    tables: List[pd.DataFrame],
    table_type: str,
) -> pd.DataFrame:
    if not tables:
        return pd.DataFrame()
    combined = pd.concat(tables, ignore_index=True)
    combined.insert(0, "table_type", table_type)
    return combined
