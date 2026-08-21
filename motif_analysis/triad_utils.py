# triad_utils.py

import numpy as np
import pandas as pd
import math
from itertools import combinations, permutations, product

try:
    from scipy.linalg import schur
    from scipy.stats import norm, zscore
except ImportError:
    schur = None

    def zscore(values, ddof=0):
        values = np.asarray(values, dtype=float)
        std = values.std(ddof=ddof)
        if std == 0 or np.isnan(std):
            return np.zeros_like(values, dtype=float)
        return (values - values.mean()) / std

    class _NormalDistribution:
        @staticmethod
        def cdf(values):
            values = np.asarray(values, dtype=float)
            erf_values = np.vectorize(math.erf)(values / np.sqrt(2.0))
            return 0.5 * (1.0 + erf_values)

    norm = _NormalDistribution()


def _node_flags(flags):
    """Return node-type flags as a positional boolean array."""
    return np.asarray(flags, dtype=bool)


def _triad_motif_id(matrix, is_excitatory, i, j, k):
    edges = np.asarray([
        matrix[i, j], matrix[j, i],
        matrix[i, k], matrix[k, i],
        matrix[j, k], matrix[k, j],
    ])
    edge_signs = tuple(int(value) for value in np.sign(edges).astype(int))
    return (
        bool(is_excitatory[i]),
        bool(is_excitatory[j]),
        bool(is_excitatory[k]),
        edge_signs,
    )


def _two_edge_triad_occurrences(matrix, is_excitatory):
    matrix = np.asarray(matrix)
    is_excitatory = _node_flags(is_excitatory)
    occurrences = {}

    for nodes in combinations(range(len(matrix)), 3):
        a, b, c = nodes
        edges = [
            matrix[a, b], matrix[b, a],
            matrix[a, c], matrix[c, a],
            matrix[b, c], matrix[c, b],
        ]
        if np.count_nonzero(edges) != 2:
            continue

        for i, j, k in permutations(nodes, 3):
            motif_id = _triad_motif_id(matrix, is_excitatory, i, j, k)
            occurrences.setdefault(motif_id, []).append((i, j, k))

    return occurrences


def count_triad_motifs(matrix, is_excitatory, is_inhibitory):
    """
    Count occurrences of each E/I-labeled triad motif in the network.
    Returns a DataFrame with columns: motif_id, num_edges, edge_signs, count
    """
    matrix = np.asarray(matrix)
    is_excitatory = _node_flags(is_excitatory)
    occurrences = _two_edge_triad_occurrences(matrix, is_excitatory)
    motif_ids = [
        motif_id
        for motif_id, motif_occurrences in occurrences.items()
        for _ in motif_occurrences
    ]
    
    if not motif_ids:
        return pd.DataFrame(columns=['motif_id', 'count', 'num_edges', 'edge_signs'])

    motif_counts = pd.Series(motif_ids).value_counts().reset_index()
    motif_counts.columns = ['motif_id', 'count']
    motif_counts['num_edges'] = 2
    motif_counts['edge_signs'] = motif_counts['motif_id'].apply(lambda x: x[3])
    
    return motif_counts

def motif_counts(variants, is_excitatory, is_inhibitory=None):
    """
    Count two-edge E/I-labeled triad motifs for one matrix or a dict of matrices.

    The notebook passes the `variants` dictionary and then expects a dictionary
    of DataFrames keyed by variant name.
    """
    if isinstance(variants, dict):
        return {
            name: count_triad_motifs(matrix, is_excitatory, is_inhibitory)
            for name, matrix in variants.items()
        }

    return count_triad_motifs(variants, is_excitatory, is_inhibitory)


def _degree_preserving_directed_null(matrix, rng, n_swaps=None, max_attempts=None):
    """
    Randomize off-diagonal directed edges with double-edge swaps.

    This preserves each node's off-diagonal in-degree and out-degree and leaves
    diagonal/self entries untouched. Edge weights move with their source edge.
    """
    shuffled_matrix = np.asarray(matrix).copy()
    edges = [
        (i, j)
        for i, j in zip(*np.nonzero(shuffled_matrix))
        if i != j
    ]
    if len(edges) < 2:
        return shuffled_matrix

    edge_set = set(edges)
    n_swaps = 10 * len(edges) if n_swaps is None else int(n_swaps)
    max_attempts = 20 * n_swaps if max_attempts is None else int(max_attempts)

    swaps_completed = 0
    attempts = 0
    while swaps_completed < n_swaps and attempts < max_attempts:
        attempts += 1
        edge_a, edge_b = rng.choice(len(edges), size=2, replace=False)
        source_a, target_a = edges[edge_a]
        source_b, target_b = edges[edge_b]

        if source_a == source_b or target_a == target_b:
            continue

        new_edge_a = (source_a, target_b)
        new_edge_b = (source_b, target_a)
        if (
            new_edge_a[0] == new_edge_a[1]
            or new_edge_b[0] == new_edge_b[1]
            or new_edge_a in edge_set
            or new_edge_b in edge_set
        ):
            continue

        value_a = shuffled_matrix[source_a, target_a]
        value_b = shuffled_matrix[source_b, target_b]
        shuffled_matrix[source_a, target_a] = 0
        shuffled_matrix[source_b, target_b] = 0
        shuffled_matrix[new_edge_a] = value_a
        shuffled_matrix[new_edge_b] = value_b

        edge_set.remove((source_a, target_a))
        edge_set.remove((source_b, target_b))
        edge_set.add(new_edge_a)
        edge_set.add(new_edge_b)
        edges[edge_a] = new_edge_a
        edges[edge_b] = new_edge_b
        swaps_completed += 1

    return shuffled_matrix


def assess_triad_enrichment(
    matrix,
    is_excitatory,
    is_inhibitory=None,
    observed_motif_counts=None,
    num_shuffles=1000,
    random_seed=42,
    n_swaps=None,
):
    """
    Assess enrichment against a degree-preserving directed null model.

    Returns a DataFrame with columns: motif_id, observed, expected, std, z_score, p_value
    """
    if observed_motif_counts is None:
        observed_motif_counts = count_triad_motifs(matrix, is_excitatory, is_inhibitory)

    observed_counts = observed_motif_counts.set_index('motif_id')['count']

    rng = np.random.default_rng(random_seed)
    shuffled_counts = []
    for _ in range(num_shuffles):
        shuffled_matrix = _degree_preserving_directed_null(matrix, rng, n_swaps=n_swaps)
        shuffled_motif_counts = count_triad_motifs(
            shuffled_matrix,
            is_excitatory,
            is_inhibitory
        )
        shuffled_counts.append(
            shuffled_motif_counts.set_index('motif_id')['count']
        )

    shuffled_counts = pd.concat(shuffled_counts, axis=1, sort=True).fillna(0)
    motif_index = observed_counts.index.union(shuffled_counts.index)
    observed_counts = observed_counts.reindex(motif_index, fill_value=0)
    shuffled_counts = shuffled_counts.reindex(motif_index, fill_value=0)
    expected_counts = shuffled_counts.mean(axis=1)
    std_counts = shuffled_counts.std(axis=1, ddof=1)
    
    enrichment_stats = pd.DataFrame({
        'observed': observed_counts,
        'expected': expected_counts,
        'std': std_counts
    })
    enrichment_stats.index.name = 'motif_id'
    nonzero_std = enrichment_stats['std'] > 0
    enrichment_stats['z_score'] = np.nan
    enrichment_stats.loc[nonzero_std, 'z_score'] = (
        (
            enrichment_stats.loc[nonzero_std, 'observed']
            - enrichment_stats.loc[nonzero_std, 'expected']
        )
        / enrichment_stats.loc[nonzero_std, 'std']
    )
    observed_deviation = (observed_counts - expected_counts).abs()
    null_deviation = shuffled_counts.sub(expected_counts, axis=0).abs()
    enrichment_stats['p_value'] = (
        1 + null_deviation.ge(observed_deviation, axis=0).sum(axis=1)
    ) / (num_shuffles + 1)
    
    return enrichment_stats.reset_index()

def measure_triad_perturbations(matrix, motif_counts, is_excitatory, is_inhibitory):
    """
    Measure change in dominant eigenvalue after removing each triad class.
    Returns a DataFrame with columns: motif_id, eigenvalue_shift
    """
    matrix = np.asarray(matrix)
    is_excitatory = _node_flags(is_excitatory)
    base_eigenvalue = np.max(np.linalg.eigvals(matrix).real)
    occurrences = _two_edge_triad_occurrences(matrix, is_excitatory)
    
    perturbation_results = []
    for _, motif in motif_counts.iterrows():
        motif_id = motif['motif_id']
        perturbed_matrix = matrix.copy()
        for i, j, k in occurrences.get(motif_id, []):
            perturbed_matrix[i,j] = perturbed_matrix[j,i] = 0
            perturbed_matrix[i,k] = perturbed_matrix[k,i] = 0
            perturbed_matrix[j,k] = perturbed_matrix[k,j] = 0
        
        perturbed_eigenvalue = np.max(np.linalg.eigvals(perturbed_matrix).real)
        eigenvalue_shift = perturbed_eigenvalue - base_eigenvalue
        perturbation_results.append({'motif_id': motif_id, 'eigenvalue_shift': eigenvalue_shift})
    
    perturbation_results = pd.DataFrame(perturbation_results)
    
    return perturbation_results

def assign_nodes_to_regions(labels):
    """
    Assign each node to its corresponding anatomical region based on label parsing.
    Returns a dict mapping node index to region name.
    """
    region_map = {}
    for i, label in enumerate(labels):
        region = label.split('_')[0]  # assuming region is first part of label
        region_map[i] = region
    
    return region_map

def count_triads_by_region(motif_counts, region_map, matrix=None, is_excitatory=None):
    """
    Count occurrences of each triad motif within and between anatomical regions.
    Returns a DataFrame with columns: motif_id, source_region, target_region, count
    """
    if matrix is not None and is_excitatory is not None:
        occurrences = _two_edge_triad_occurrences(matrix, is_excitatory)
        regional_counts = []

        for motif_id in motif_counts['motif_id']:
            for i, j, k in occurrences.get(motif_id, []):
                regional_counts.append({
                    'motif_id': motif_id,
                    'source_regions': tuple(sorted([region_map[i], region_map[j]])),
                    'target_region': region_map[k],
                    'count': 1
                })

        if not regional_counts:
            return pd.DataFrame(columns=['motif_id', 'source_regions', 'target_region', 'count'])

        return (
            pd.DataFrame(regional_counts)
            .groupby(['motif_id', 'source_regions', 'target_region'], as_index=False)
            .sum()
        )

    regional_counts = []
    for _, motif in motif_counts.iterrows():
        motif_id = motif['motif_id']
        for i, j, k in product(range(len(region_map)), repeat=3):
            if len({i, j, k}) == 3:
                source_regions = tuple(sorted([region_map[i], region_map[j]]))
                target_region = region_map[k]
                regional_counts.append({
                    'motif_id': motif_id,
                    'source_regions': source_regions,
                    'target_region': target_region,
                    'count': motif['count']
                })
    
    regional_counts = pd.DataFrame(regional_counts)
    regional_counts = regional_counts.groupby(['motif_id', 'source_regions', 'target_region']).sum().reset_index()
    
    return regional_counts

def analyze_triad_schur_complements(matrix, motif_counts, is_excitatory, is_inhibitory):
    """
    Compute Schur complement for each triad and analyze effective 2-node dynamics.
    Returns a dict mapping each motif_id to its Schur complement spectrum and stats.
    """
    matrix = np.asarray(matrix)
    is_excitatory = _node_flags(is_excitatory)
    occurrences = _two_edge_triad_occurrences(matrix, is_excitatory)
    schur_stats = {}
    for _, motif in motif_counts.iterrows():
        motif_id = motif['motif_id']
        schur_spectra = []
        for i, j, k in occurrences.get(motif_id, []):
            sub_matrix = matrix[np.ix_([i, j, k], [i, j, k])]
            if schur is None:
                schur_spectra.append(np.linalg.eigvals(sub_matrix)[:2])
            else:
                schur_matrix = schur(sub_matrix, sort='lhp')[0]
                schur_spectra.append(np.diag(schur_matrix)[:2])
        
        if len(schur_spectra) > 0:
            schur_spectra = np.array(schur_spectra)
            schur_stats[motif_id] = {
                'spectra': schur_spectra,
                'mean_spectrum': np.mean(schur_spectra, axis=0),
                'max_real_part': np.max(schur_spectra.real),
                'num_unstable': np.sum(schur_spectra.real > 0)
            }
    
    return schur_stats

def nominate_stabilizing_triads(enrichment_stats, perturbation_results, schur_stats):
    """
    Nominate candidate stabilizing triad motifs based on integrated analysis.
    Returns a DataFrame with top nominated triads and their key statistics.
    """
    integrated_stats = enrichment_stats.merge(perturbation_results, on='motif_id')
    integrated_stats['max_real_part'] = integrated_stats['motif_id'].map(lambda x: schur_stats.get(x, {}).get('max_real_part', np.nan))
    integrated_stats['num_unstable'] = integrated_stats['motif_id'].map(lambda x: schur_stats.get(x, {}).get('num_unstable', np.nan))
    
    # Criteria for nomination: significantly enriched, strong stabilizing shift, stable Schur spectra
    nominated_triads = integrated_stats[
        (integrated_stats['z_score'] > 2) & 
        (integrated_stats['eigenvalue_shift'] < -0.1) &
        (integrated_stats['max_real_part'] < 0)
    ].sort_values('eigenvalue_shift').head(10)
    
    return nominated_triads
