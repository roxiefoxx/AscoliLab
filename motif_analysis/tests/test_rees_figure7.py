"""Independent checks of topology, regional denominators and group ratios."""
import unittest
from itertools import combinations

import networkx as nx
import numpy as np

from rees_figure7 import Figure7Config, REES_TO_DHL, calculate_figure7_tables
from weighted_motif_null import PAIR_ORDER


def dataset(matrix, labels=None, ei=None):
    n = len(matrix)
    return {"matrix": np.array(matrix, dtype=float),
            "labels": labels if labels is not None else [f"DG Type {i}" for i in range(n)],
            "ei": np.array(ei if ei is not None else ["I"] * n)}


class Figure7Tests(unittest.TestCase):
    def test_all_64_directed_graphs_against_networkx(self):
        for bits in range(64):
            matrix = np.zeros((3, 3))
            for k, (a, b) in enumerate(PAIR_ORDER):
                matrix[a, b] = -2 if bits >> k & 1 else 0
            np.fill_diagonal(matrix, 100)  # loops and signed magnitudes have no effect
            census = calculate_figure7_tables(dataset(matrix))["census"]
            graph = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
            graph.remove_edges_from(nx.selfloop_edges(graph))
            self.assertEqual(census.loc[census["count"] == 1, "triad_class"].item(), nx.triad_type(graph))

    def test_rees_letters_from_reference_arrow_patterns(self):
        # Arrows independently transcribed from Figure 7A (upper-left=0,
        # lower-left=1, right=2). In particular H and J are distinct classes.
        edges = {
            "A": [(1, 0), (2, 0)], "B": [(1, 0), (2, 1)],
            "C": [(1, 0), (1, 2)], "D": [(0, 1), (1, 0), (2, 0)],
            "E": [(1, 0), (2, 0), (2, 1)], "F": [(0, 1), (1, 0), (0, 2)],
            "G": [(1, 0), (0, 2), (2, 1)],
            "H": [(0, 1), (1, 0), (2, 0), (2, 1)],
            "I": [(0, 1), (1, 0), (0, 2), (2, 0)],
            "J": [(0, 1), (1, 0), (1, 2), (2, 0)],
            "K": [(1, 2), (2, 1), (1, 0), (2, 0)],
        }
        for letter, links in edges.items():
            graph = nx.DiGraph()
            graph.add_nodes_from(range(3))
            graph.add_edges_from(links)
            self.assertEqual(REES_TO_DHL[letter], nx.triad_type(graph), letter)

    def test_census_and_regions_reconcile_with_independent_enumeration(self):
        matrix = np.array([[0, 1, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0], [1, 0, 1, 0]])
        data = dataset(matrix, ["CA1 A", "CA1 B", "CA1 C", "DG D"])
        tables = calculate_figure7_tables(data)
        graph = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
        expected = nx.triadic_census(graph)
        self.assertEqual(tables["census"].set_index("triad_class")["count"].to_dict(), expected)
        connected_intra = 0
        for triple in combinations(range(4), 3):
            if nx.is_weakly_connected(graph.subgraph(triple)) and all(i < 3 for i in triple):
                connected_intra += 1
        self.assertEqual(tables["regional_counts"]["CA1"].sum(), connected_intra)
        self.assertEqual(tables["node_participation"][list(REES_TO_DHL)].to_numpy().sum(), 12)

    def test_group_denominators_scope_and_missing_ratios(self):
        data = dataset(np.ones((5, 5)),
                       ["CA1 Pyramidal", "CA1 Basket", "CA1 Ivy", "CA1 O LM", "DG A"],
                       ["E", "I", "I", "I", "I"])
        ratios = calculate_figure7_tables(data)["ca1_ratios"].set_index("superpattern")
        self.assertEqual(ratios.loc["M", "pyramidal_participation"], 3)
        self.assertEqual(ratios.loc["M", "pyr_int_mean_per_type_ratio"], 1)
        self.assertAlmostEqual(ratios.loc["M", "pyr_int_participation_ratio"], 1 / 3)
        self.assertEqual(ratios.loc["M", "pst_other_mean_per_type_ratio"], 1)
        self.assertTrue(np.isnan(ratios.loc["A", "pyr_int_mean_per_type_ratio"]))
        all_ratios = calculate_figure7_tables(data, Figure7Config(ratio_scope="all_triads"))["ca1_ratios"]
        self.assertEqual(all_ratios.iloc[-1].pyramidal_participation, 6)

    def test_no_triples_and_no_ca1(self):
        for n in (0, 1, 2, 4):
            tables = calculate_figure7_tables(dataset(np.zeros((n, n))))
            self.assertEqual(tables["census"].query("connected")["count"].sum(), 0)
            self.assertTrue(tables["regional_shares"].drop(columns="superpattern").isna().all().all())
            self.assertTrue(tables["ca1_ratios"].filter(like="_ratio").isna().all().all())


if __name__ == "__main__":
    unittest.main()
