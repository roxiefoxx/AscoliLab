# EE Backbone Analyses

This directory contains two active notebook analyses for connectomic backbone work:

- `ee_backbone_method_comparison.ipynb`: compares methods for building a directed excitatory-to-excitatory (E-to-E) backbone.
- `ee_backbone_deterministic_stochastic_comparison.ipynb`: branch notebook comparing a lean deterministic method set with stochastic path-sampling methods.
- `ei_backbone_analysis.ipynb`: explores how inhibitory connectivity modulates dynamics along a backbone.

The reusable E-to-E implementation lives in `ee_backbone_analysis.py`. Stochastic E-to-E methods live in `ee_backbone_stochastic_methods.py`. Feedback-arc backbone ordering lives in `ee_backbone_ordering.py`. The reusable E/I implementation lives in `ei_backbone_analysis_helpers.py`.

> **Results below need regeneration.** Two solver defects were corrected (see `docs/method_citations_and_verification.md`): the branch-and-bound optimistic bound was not admissible and could prune the optimum, and MILP optimality was read from `pulp.LpStatus`, which reports `Optimal` even when CBC stops on the time limit. Both are fixed, but the summary tables in this README predate the fix. Rerun the notebooks before quoting any path-weight figure. The feedback-arc ordering results are current.

## Active Files

| File | Role |
| --- | --- |
| `ee_backbone_method_comparison.ipynb` | Canonical E-to-E method comparison notebook. |
| `ee_backbone_deterministic_stochastic_comparison.ipynb` | Separate branch notebook adding weighted random-walk, Monte Carlo tree search, and Boltzmann path sampling without changing the canonical comparison. |
| `ee_backbone_analysis.py` | Main implementation for E-to-E graph preparation, path methods, MILP benchmarks, summaries, and CSV export. |
| `ee_backbone_stochastic_methods.py` | Stochastic E-to-E path methods and ensemble diagnostics. |
| `ee_backbone_ordering.py` | Feedback-arc-minimizing backbone ordering (Vahidi 2025), SCC decomposition, and exact within-ordering path extraction. |
| `ei_backbone_analysis.ipynb` | Separate exploratory analysis of inhibition and linear dynamics around a backbone. |
| `ei_backbone_analysis_helpers.py` | Reusable functions for the E/I notebook: matrix loading, normalization, impulse response, sweeps, ablations, motif tables, and null models. |
| `update_advisor_notebook_brief.py` | Optional utility for creating the advisor-facing Word brief in `docs/`. It is not part of the core analysis pipeline. |

## Redundant Or Superseded Scripts

These files are already in `archive/` and are redundant for current work:

| Archived file | Why it is redundant |
| --- | --- |
| `archive/build_ee_backbone.py` | Early greedy builder; superseded by the canonical notebook and helper module. |
| `archive/build_ee_backbone_per_injection.py` | Early per-seed greedy output script; superseded by `greedy_tree_path` in the canonical workflow. |
| `archive/branch_bound_ee_backbone.py` | Earlier standalone branch-and-bound implementation; superseded by `branch_bound_path` in `ee_backbone_analysis.py`. |
| `archive/dynamic_programming_ee_backbone.py` | Earlier standalone DP implementation; superseded by `exact_bitmask_dp_path` and `beam_dp_path`. |
| `archive/ee_backbone_compare_self_modes_visuals.ipynb` | Earlier notebook variant; superseded by `ee_backbone_method_comparison.ipynb`. |
| `archive/ee_backbone_analysis_script.py` | Former E-to-E implementation file; folded into `ee_backbone_analysis.py` so the notebook imports one canonical helper module. |

Generated cache files in `__pycache__/` are also redundant, but they are not analysis scripts.

## Unique Analyses

### 1. E-to-E Backbone Method Comparison

Notebook: `ee_backbone_method_comparison.ipynb`

Input matrix: `matrices/mij_EE_matrix.csv`

This analysis treats the E-to-E matrix as a directed weighted graph over 32 excitatory nodes. Diagonal/self entries are ignored. Positive off-diagonal weights are candidate transitions, giving 355 positive directed E-to-E transitions.

The notebook compares seven backbone-building methods. The first six select a path; the seventh selects a global ordering:

| Method | Rationale |
| --- | --- |
| `greedy_tree_path` | A local heuristic. From each seed, repeatedly take the strongest positive outgoing transition to an unvisited excitatory node. Useful as a simple baseline. |
| `maximum_spanning_tree` | Builds an undirected high-weight skeleton by taking the stronger direction for each pair, then extracts a seed-specific path through that tree. Useful for a compact global connectivity scaffold. |
| `branch_and_bound` | Searches directed simple paths while pruning branches whose optimistic upper bound cannot beat the current best. More global than greedy, but currently limited by node-count guardrails. |
| `dynamic_programming` | Uses exact bitmask DP when the graph is small enough, otherwise beam-limited DP. Here it ran in beam mode, so it is an approximate state-space search. |
| `milp_subtour_elimination` | Mixed-integer benchmark for maximum-weight simple directed paths, using subtour-elimination constraints. In the current outputs, all 32 seed solves reached `Optimal`. |
| `maximum_weight_asymmetric_hamiltonian_path` | MILP variant requiring every excitatory node to be visited exactly once. This answers a stricter question than the simple-path methods. |
| `feedback_arc_ordering` | Not a path method. Finds the vertex ordering maximizing total forward-pointing edge weight, scoring every positive edge rather than only one selected walk. Supplies the backbone coordinate the E/I notebook needs. |

Current output folder: `ee_backbone_comparison_outputs/`

### E-to-E Results Summary

From `ee_backbone_comparison_outputs/method_summary.csv`:

| Method | Mean path weight | Median path weight | Mean length | Max path weight |
| --- | ---: | ---: | ---: | ---: |
| `milp_subtour_elimination` | 227,744.98 | 241,065.04 | 28.47 | 241,901.42 |
| `maximum_weight_asymmetric_hamiltonian_path` | 204,189.12 | 222,249.22 | 31.00 | 223,058.62 |
| `branch_and_bound` | 177,627.44 | 171,725.20 | 25.72 | 232,244.06 |
| `greedy_tree_path` | 125,314.24 | 121,527.33 | 16.69 | 195,127.34 |
| `dynamic_programming` | 109,772.18 | 105,454.36 | 4.47 | 150,349.91 |
| `maximum_spanning_tree` | 95,745.17 | 109,051.22 | 5.47 | 114,610.22 |

Key takeaways:

- The MILP subtour-elimination method finds the highest average path weight. Optimality is now certified per seed by `milp_proven_optimal`, read from `problem.sol_status`. The earlier claim that "all 32 solves are marked `Optimal`" came from `pulp.LpStatus`, which reports `Optimal` both when CBC proves optimality and when it stops on the time limit holding an incumbent; it was not a certificate.
- The Hamiltonian MILP visits all 32 excitatory nodes by construction, so it has the longest paths. It is not directly comparable to methods that may stop early.
- Branch-and-bound scores improved after the bound fix, because the previous bound was pruning branches containing better paths. It remains exact only when `bb_status == "complete"`.
- Dynamic programming ran as `beam_width_5000` for all seeds, so it is approximate here. Its short average path length explains much of the lower score, partly because expanded states compete for beam slots against their own children.
- The maximum spanning tree method produces compact paths and the lowest average path score, which is expected because the tree skeleton sacrifices directed path optimality for a global undirected scaffold.
- The old benchmark table contained one negative gap, for seed `SUB CA1 Projecting Pyramidal`. That was **not** evidence that the two MILP formulations answer different questions: a Hamiltonian path from a seed is a simple path from that seed, so it is feasible for the unconstrained MILP and cannot beat a true optimum. It was an unproven incumbent. The gap column is now `weight_gap_to_milp_incumbent`, carries a `gap_is_certified` flag, prints a note listing uncertified negative gaps, and raises if any method ever beats a *proven* optimum.

Best seed-level result by method:

| Method | Best seed | Best weight | Length | Terminus |
| --- | --- | ---: | ---: | --- |
| `milp_subtour_elimination` | EC LI II Pyramidal Fan | 241,901.42 | 28 | DG Hilar Ectopic Granule |
| `branch_and_bound` | EC LII III Pyramidal Tripolar | 232,244.06 | 24 | CA1 Radiatum Giant |
| `maximum_weight_asymmetric_hamiltonian_path` | EC LI II Pyramidal Fan | 223,058.62 | 31 | CA1 Radiatum Giant |
| `greedy_tree_path` | MEC LII Oblique Pyramidal | 195,127.34 | 22 | CA1 Radiatum Giant |
| `dynamic_programming` | DG Semilunar Granule | 150,349.91 | 6 | DG Mossy |
| `maximum_spanning_tree` | EC LII III Pyramidal Tripolar | 114,610.22 | 6 | MEC LV VI Pyramidal Polymorphic |


### E-to-E Backbone Ordering Results

From `ee_backbone_comparison_outputs/ordering_metrics.csv` and `ordering_permutation_null.csv`:

| Quantity | Value |
| --- | ---: |
| Strongly connected components | 1 (all 32 types) |
| Positive edges | 355 |
| Forward edges under the ordering | 208 |
| Feedback edges under the ordering | 147 |
| Forward edge weight | 748,421 |
| Feedback edge weight | 39,719 |
| **Forward-weight fraction** | **0.9496** |
| Permutation null mean (n = 2,000) | 0.486 (sd 0.124) |
| Best of 2,000 random orderings | 0.815 |
| Empirical p | < 0.005 |

Roughly 95% of positive excitatory-to-excitatory edge weight points forward under the recovered ordering, against a chance expectation near 50% and a best-of-2,000-random-orderings value of 0.81. The excitatory subnetwork is close to feedforward at the neuron-type level despite being a single strongly connected component, i.e. its recurrence is carried by a small fraction of the total weight.

The recovered ordering opens with entorhinal superficial layers, passes through CA3 and CA2 into CA1, and ends in subiculum and deep entorhinal layers, which is the expected direction of flow through the hippocampal formation. That was not imposed; it fell out of maximizing forward edge weight. The heaviest individual feedback transitions are worth inspecting directly in `ordering_edge_directions.csv`.

`backbone_ordering.csv` is written for reuse as `BACKBONE_LABELS` in `ei_backbone_analysis.ipynb`, which replaces the CSV-label-order placeholder that notebook previously warned about.

### 2. E/I Inhibitory Modulation Analysis

Notebook: `ei_backbone_analysis.ipynb`

Input matrix: `matrices/mij_matrix.csv`

This notebook studies how inhibitory edges shape propagation and stability around a backbone using a linear dynamical system. It is a separate analysis from the E-to-E method comparison.

Unique analyses performed:

| Section | Rationale |
| --- | --- |
| Matrix validation | Checks the 85-node full matrix, sign structure, density, and orientation assumptions. |
| Backbone definition | Loads the feedback-arc-minimizing ordering from `ee_backbone_comparison_outputs/backbone_ordering.csv`. The 32 excitatory types come first in that order; the 53 inhibitory types are appended after them, so motif displacement is measured against excitatory backbone position. Falls back to CSV label order with a warning if the ordering file is absent. |
| Excitation/inhibition split | Separates positive and negative weights and normalizes scale so the linear system is numerically tractable. |
| Impulse propagation | Stimulates the first backbone node and measures peak response, area under response, peak timing, and inhibition index along the backbone. |
| Inhibitory strength sweep | Sweeps inhibitory gain `g` to see how inhibition changes stability and terminal propagation. |
| Global transient amplification | Computes non-normal transient gain to distinguish short-term amplification from asymptotic stability. |
| Inhibitory node ablation | Removes each inhibitory source's outgoing inhibition and measures change in terminal peak response. |
| Inhibitory edge ablation | Removes individual inhibitory edges and measures effect on terminal peak response. |
| Motif analysis | Classifies inhibitory edges as feedforward or feedback relative to the backbone coordinate. |
| Null model | Randomizes inhibitory targets while preserving each source's inhibitory out-degree and weight multiset. Inference is by empirical p-value at `N_NULL = 500`, not by z-score. |

### E/I Results Summary

The current executed notebook outputs, using the feedback-arc backbone ordering, show:

- Full matrix: 85 nodes.
- Positive edges: 707.
- Negative edges: 863.
- Off-diagonal nonzero density: 0.2199.
- Dale-style sender classification: 32 mostly excitatory and 53 mostly inhibitory nodes. The 0.9 threshold used here and the 0.5 threshold in `ee_backbone_analysis.is_excitatory_sender` return the identical 32-node set on this matrix, matching `mij_EE_matrix.csv` exactly; there are no mixed-sign senders in between.
- Weight normalization scale: approximately 23,940, based on the 95th percentile absolute weight.
- **Stability and non-normality:** the spectral abscissa runs from -0.4191 at `g=0` to -0.6085 at `g=2`, so the system is asymptotically stable at every gain tested and the observed amplification is genuinely transient rather than truncated exponential growth. The numerical abscissa is +27.20, giving a spectral/numerical gap of about 27.6. That gap is the quantitative signature of non-normality and is a stronger statement than `G_max` alone. `G_max` itself declines slightly with inhibition, from 24.63 at `g=0` to 23.04 at `g=2`, peaking at `t = 1.2` throughout.
- Inhibitory motif counts relative to the feedback-arc backbone: 285 feedforward edges and 578 feedback edges.
- Feedforward inhibitory edges are stronger per edge and much more local than feedback edges: mean magnitude 0.00748 vs 0.00444, mean absolute backbone displacement 10.5 vs 21.8, median displacement +7 vs -13. Strong local feedforward inhibition alongside weak long-range feedback inhibition is consistent with the hippocampal-entorhinal microcircuit literature and is checkable against it.
- Null model terminal peak: observed 0.10749 against a null (n = 500) with mean 0.10952, median 0.11148, sd 0.00615, skew -2.54. The observed value sits at the **22.4th percentile**, empirical p = 0.776 for observed >= null and p = 0.226 for observed <= null. No null draw was unstable. The observed inhibitory target placement is therefore **not distinguishable from chance** on this statistic under the real ordering.

Two notes on that null. First, inference is by empirical p-value `(1 + #{null >= observed}) / (1 + n_null)`, not by z-score: this null is strongly skewed, and the direction of the skew depends on which node the backbone terminates at, so a z-score is not interpretable. At the previous default of `N_NULL = 25` it was also not reproducible — resampling 25 draws from a larger null repeatedly gives z values spanning roughly -0.2 to +34. Second, the null preserves each source's inhibitory out-degree and weight multiset but not target in-degree, so it remains a first-order placement null; a strength-preserving randomization (Milisav et al., 2024, *Nature Computational Science*) would be the stronger comparison.

## Reproducibility Notes

- Run `ee_backbone_method_comparison.ipynb` first to regenerate the E-to-E outputs in `ee_backbone_comparison_outputs/`. It writes `backbone_ordering.csv`, which the E/I notebook consumes.
- Run `ee_backbone_deterministic_stochastic_comparison.ipynb` to regenerate the deterministic/stochastic branch outputs in `ee_backbone_stochastic_comparison_outputs/`.
- `ee_backbone_comparison_outputs_self_connections/` appears to be a stale output folder from an older self-connection variant and is not used by the current canonical notebook.
- Run `ei_backbone_analysis.ipynb` last to regenerate the E/I outputs in `connectome_inhibition_results/`; it loads the backbone ordering written by the first notebook.
- `docs/method_citations_and_verification.md` maps every notebook section to supporting literature and records the verification pass behind the corrections noted above.
- Requires NumPy >= 2.0 (`np.trapezoid`), SciPy, pandas, and PuLP for the MILP arms.
