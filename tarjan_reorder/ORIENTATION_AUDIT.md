# Matrix Orientation Audit

## Bottom Line

Your original `mij_matrix.csv` is described in the notebooks as rows = presynaptic/source and columns = postsynaptic/target.

For these analyses, use the paper/dynamics convention:

- rows = postsynaptic/target cells
- columns = presynaptic/source cells
- `M[post, pre]` is the weight for `pre -> post`
- feedforward edges should appear below the diagonal after a successful directional ordering

You do not need to manually transpose `mij_matrix.csv` when running notebooks `01` or `02`. Set `MATRIX_ORIENTATION` in the setup cell:

- `pre_rows_post_columns` means the CSV rows are presynaptic/source cells and columns are postsynaptic/target cells. The CSV is already source-to-target as a table, but the loader transposes it once so analysis uses `M[post, pre]`.
- `post_rows_pre_columns` means the CSV rows are postsynaptic/target cells and columns are presynaptic/source cells. The loader uses it as stored.
- `auto_from_netlist` means the loader compares both orientations against the optional netlist and chooses the lower-residual `M[post, pre]` version.

If no netlist is available, use either `pre_rows_post_columns` or `post_rows_pre_columns`; `auto_from_netlist` requires a netlist.

For any new notebook or standalone script, either call `load_inputs(...)` from `matrix_reordering_paper_analysis_script.py` or start from the exported canonical matrix under:

- `outputs/paper_matrix_reordering/canonical_inputs/mij_matrix_paper_convention_raw.csv`
- `outputs/paper_matrix_reordering/canonical_inputs/mij_matrix_paper_convention_rho1.csv`

## Notebook Tracking

| Notebook | Input orientation at file boundary | Orientation used for analysis | Manual transpose needed? |
| --- | --- | --- | --- |
| `01_matrix_reordering_paper_analysis.ipynb` | Controlled by `INPUT_MATRIX_PATH` and `MATRIX_ORIENTATION` | `M[post, pre]`, selected by `load_inputs(...)` | No |
| `02_modified_tarjan_pathway_reconstruction.ipynb` | Controlled by `INPUT_MATRIX_PATH` and `MATRIX_ORIENTATION` | `M[post, pre]`, inherited from `load_inputs(...)` | No |
| `03_motif_schur_decomposition_analysis.ipynb` | Motif exports are source rows / target columns | Blocks are transposed once after loading, then analyzed as `block[target, source]` | No, the notebook now does it |
| `04_motif_signed_overlap_operator_analysis.ipynb` | Motif exports are source rows / target columns | Blocks are transposed once after loading, then analyzed as `block[target, source]` | No, the notebook now does it |

## Script Tracking

| Script | Orientation behavior |
| --- | --- |
| `matrix_reordering_paper_analysis_script.py` | `load_inputs(...)` accepts optional `netlist_path` and explicit `matrix_orientation`. With a netlist, it can audit or auto-select orientation by reconstructing `pre_neuron -> post_neuron` rows into `M[post, pre]`. Without a netlist, it uses the explicit orientation. `graph_from_matrix(...)` then calls NetworkX as `add_edge(pre, post)`. `dfs_finish_order(...)` (added 2026-09-05) traverses that same graph, so it inherits the same orientation; it returns the reverse DFS finishing order, sources first, which places feedforward edges below the diagonal exactly as `tarjan_order(...)` does. |
| `modified_tarjan_pathway_analysis_script.py` | Reuses `load_inputs(...)`, `graph_from_matrix(...)`, and `tarjan_order(...)`, so it uses the same `M[post, pre]` convention. Direct path checks use entries such as `M[dg, ec]`, meaning `EC -> DG`. It also accepts optional `netlist_path`, explicit `matrix_orientation`, and `normalization`. |
| `motif_schur_decomposition_analysis_script.py` | Loads motif blocks exactly as stored and passes them directly to NumPy/SciPy linear algebra. It does not infer or change orientation by itself. The notebook now transposes source-to-target motif exports before calling these functions. |
| `motif_signed_overlap_analysis_script.py` | Loads motif blocks exactly as stored and passes them directly to NumPy/SciPy eigen routines. It does not infer or change orientation by itself. The notebook now transposes source-to-target motif exports before calling these functions. |

## Library Function Tracking

| Library call | What the library assumes | How this repo feeds it |
| --- | --- | --- |
| `networkx.DiGraph.add_edge(u, v)` | `u -> v` | The repo passes `labels[pre] -> labels[post]` after reading `M[post, pre]`. |
| `networkx.strongly_connected_components`, `condensation`, `topological_sort` | Directed graph edges already encode direction | Correct if `graph_from_matrix(...)` receives `M[post, pre]`. |
| `networkx.pagerank` | Directed graph edges already encode direction | Correct if built through `graph_from_matrix(...)`; PageRank ranks targets receiving directed weight. |
| `scipy.linalg.schur`, `numpy.linalg.eig`, `numpy.linalg.eigvals` | Pure matrix algebra; no biological source/target meaning | For dynamics `dV/dt = A V`, rows must be target/post and columns source/pre before calling these functions. |
| `scipy.sparse.csgraph.reverse_cuthill_mckee` | Sparse matrix row/column structure only | The repo symmetrizes with `abs(M) > 0` and `B = max(B, B.T)`, so direction is intentionally discarded. |
| Laplacian/eigendecomposition helpers | Undirected or symmetrized graph geometry unless explicitly signed | The global Laplacian uses `abs(M) + abs(M.T)`, so source/target direction is intentionally discarded after the canonical orientation boundary. |
| Motif signed-overlap operator | Rows are motif occurrences, columns are cell types | This is not a pre/post matrix. It is a feature matrix, so no source/target transpose applies. |

## Practical Rule

If a function will interpret `M[i, j]` as the influence of cell `j` on cell `i`, use `M[post, pre]`.

If your only file is the original source-row/target-column CSV, set `MATRIX_ORIENTATION = "pre_rows_post_columns"` in notebooks `01` or `02`; do not transpose it yourself. If you are writing a direct NumPy/SciPy script without `load_inputs(...)`, transpose that CSV once before dynamics or graph construction.

Set `NORMALIZATION = "spectral_radius"` to scale the matrix to `SPECTRAL_RADIUS_TARGET`, or `NORMALIZATION = "none"` to analyze raw weights.
