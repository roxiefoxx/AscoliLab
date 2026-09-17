# Orientation inventory: which functions require `M[post, pre]`

Convention: `M[post, pre]` = row is the postsynaptic target/receiver, column is
the presynaptic source/sender, so `x[t+1] = M @ x[t]`. The opposite is
`M[pre, post]`.

Scope: every `.py` outside `archive/` and `_pre_sync_backup_*/`, plus the
library calls reached from the active notebooks. Custom code is tagged
**[CUSTOM]**. Line numbers and the notebook columns were regenerated on
2026-09-17 from an AST call graph (`notebook → imported function → transitively
called functions`), not by hand.

### Notebook key

| Short | Folder |
|---|---|
| `schur/NN` | `schur_decomp/` |
| `jac/NN` | `jacobian_analysis/` |
| `tarjan/NN` | `tarjan_reorder/` |
| `inhib/NN` | `inhib_modulation/notebooks/` |
| `motif/NN` | `motif_analysis/` |
| `nn/replication` | `non-normal_matrices/analyses/mij_paper_replication.ipynb` |
| `EE/*` | `EE_backbone/` |

### What changed since the previous pass

- `jacobian_analysis/` is new (6 notebooks + `jacobian_core.py`). **It already
  implements the orientation oracle** — see §0.
- `motif_analysis/` now holds copies of `motif_schur_decomposition_analysis_script.py`
  and `motif_signed_overlap_analysis_script.py`, and notebooks `03`/`04` exist in
  **both** `motif_analysis/` and `tarjan_reorder/`. Four notebooks, two script
  copies, one convention to keep in sync.
- `EE_backbone/` split into `ee_backbone_analysis.py`, `ee_backbone_ordering.py`,
  `ee_backbone_stochastic_methods.py`, `ei_backbone_analysis_helpers.py`.
- `connectivity_methods.load_mij_matrix` was renamed `load_connectivity_data`.

---

## 0. Baseline: what the raw data actually is

Every `mij_matrix.csv` in the repo is **`[pre, post]`** — senders in rows,
receivers in columns. Modules that need dynamics transpose it on load.

| Loader | Purpose | Orientation behavior | Notebooks |
|---|---|---|---|
| `load_jacobian_data` **[CUSTOM]** `jacobian_analysis/jacobian_core.py:169` | Load the CSV, transpose to `[post, pre]`, and verify the result before returning it. | Transposes **and asserts** via `assert_post_pre`; records the measured purities on `JacobianData.orientation_check`. | `jac/01`–`jac/06` |
| `load_inputs` **[CUSTOM]** `tarjan_reorder/matrix_reordering_paper_analysis_script.py:78` | Load matrix + netlist and return `M` in paper/dynamics form. | Auto-detects orientation by reconstructing the netlist into `[post, pre]` and comparing residuals; raises if the chosen orientation is off. | `tarjan/01`, `tarjan/02` |
| `load_mij_data` **[CUSTOM]** `schur_decomp/schur_core_script.py:147` | Load Mij plus optional netlist E/I metadata. | `M = df.to_numpy().T` → `[post, pre]`. Assumed, not checked. | `schur/03`, `schur/05`, `schur/07`, `schur/08` |
| `load_connectivity_data` **[CUSTOM]** `non-normal_matrices/analyses/connectivity_methods.py:32` | Load the empirical matrix and return a normalized, paper-oriented `J` bundle. | `J_paper = raw.T`, then spectral-radius normalized. Assumed. | `nn/replication` |
| `load_source_receiver_csv` **[CUSTOM]** `schur_decomp/network_linear_systems_tools_script.py:60` | Load a source→receiver CSV into state-update convention. | `M_raw = raw.T` → `[post, pre]`. Assumed. | `schur/09` |
| `load_connectivity` **[CUSTOM]** `inhib_modulation/notebooks/inhibitory_modulation.py:110` | Load a square matrix or a pre/post edge list into `[post, pre]`. | `matrix_orientation="pre_by_post"` default, transposes. Declared by argument, not checked. | `inhib/01`–`inhib/03` |
| `prepare_local_schur_model` **[CUSTOM]** `schur_decomp/schur_mode_greedy_trees_script.py:64` | Load, orient receiver-by-source, normalize, decompose. | `A_raw = frame.to_numpy().T`. Assumed. | `schur/05` |
| `prepare_schur_model` **[CUSTOM]** `schur_decomp/schur_signal_propagation_script.py:58` | Load and orient for state propagation, then Schur-decompose. | `A_raw = frame.to_numpy().T`. Assumed. | `schur/03` |
| `load_mij_matrix` **[CUSTOM]** `schur_decomp/schur_core_script.py:118` | Validated CSV reader — square, labels aligned, no duplicates. | **No transpose.** Returns the raw `[pre, post]` frame. | `schur/03`, `04`, `05`, `07`, `08`, `09` |
| `load_matrix` **[CUSTOM]** `EE_backbone/ee_backbone_analysis.py:54` | Read the EE matrix for backbone path search. | **No transpose** — stays `[pre, post]` (see §4). | `EE/method_comparison`, `EE/det_stochastic_comparison` |
| `load_signed_matrix` **[CUSTOM]** `EE_backbone/ei_backbone_analysis_helpers.py:13` | Read the EI matrix for the continuous-time backbone. | `csv_orientation="source_rows"` → transposes to `[post, pre]`. Declared by argument. | `EE/ei_backbone_analysis` |

### Your question: how to reuse `load_inputs`-style validation elsewhere

You no longer need to port `load_inputs`. A better, cheaper check already
exists in this repo: **`jacobian_core.assert_post_pre`**
(`jacobian_analysis/jacobian_core.py:115`).

The two validators differ in what they need:

| | `load_inputs` residual test | `assert_post_pre` sign test |
|---|---|---|
| Needs | matrix **and** netlist | matrix **and** E/I labels only |
| Method | rebuild `M[post,pre]` from `pre→post` rows, compare Frobenius residual for as-is vs transposed | E→I block must be ~100% positive, I→E ~100% negative; the two swap completely under transposition |
| Tolerance | a threshold you must pick | none — separation is 100/0 on this dataset |
| Works on | the full labeled matrix | full matrix, **submatrices, motif blocks, any labeled slice** |

The sign test is the reusable one, because it does not need the netlist at the
call site and it survives slicing. Reuse it in three ways, cheapest first:

1. **Call it from a notebook** — no code changes anywhere. Add `jacobian_analysis`
   to `sys.path` and call `assert_post_pre(M, ei, name="...")` right after any
   loader. This is the only step that costs nothing.
2. **Call it inside the assuming loaders** (`load_mij_data`,
   `load_connectivity_data`, `load_source_receiver_csv`, `load_connectivity`,
   `prepare_schur_model`, `prepare_local_schur_model`). Each already has the
   E/I series in hand at the point of return, so it is a two-line addition per
   loader. Behavior is unchanged on correct data; it only adds a raise on wrong
   data.
3. **Only if you want one shared home:** there is no package here — no
   `pyproject.toml`, no `setup.py`, no `__init__.py`, and every notebook appends
   its own folder to `sys.path`. A genuinely shared import needs a new mechanism
   in all six folders. Copying the ~25-line function into a second module is the
   smaller change; note the copy in both docstrings.

Do **not** retire the `load_inputs` residual test — it is the only check that
validates against the netlist itself rather than against Dale's law, so it
catches a different failure (a matrix that disagrees with the edge list).

---

## 1. Library functions that genuinely require `M[post, pre]`

Feeding these `M[pre, post]` gives numerically valid but biologically transposed
results — the direction of causation flips.

| Function | Purpose in this repo | Why orientation matters | Notebooks |
|---|---|---|---|
| `scipy.linalg.expm` | Continuous-time flow map `expm(At)` for the EI backbone impulse response. | `expm(At)` solves `dx/dt = Ax`; needs `A[post, pre]`. | `EE/ei_backbone_analysis` |
| `scipy.sparse.linalg.expm_multiply` | Same flow map applied directly to `x0`. | Same. | `EE/ei_backbone_analysis` |
| `np.linalg.matrix_power` | `M^k` for Neumann-series inhibitory feedback and chained disinhibition. | `M^k[i,j]` is the signed k-step influence of pre `j` on post `i`. | `inhib/02`, `schur/07`, `schur/08` |
| `np.linalg.solve` | Resolvents `(I − αM)^{-1}u` and `(zI − M_II)^{-1}` — steady-state response to drive. | The resolvent maps an input on sources to a response on targets. | `inhib/02`, `schur/07`, `schur/08` |
| `np.linalg.inv` / `np.linalg.pinv` | Same resolvent role; also motif-block gain `inv(I − block)`. | Same. | `inhib/01`, `inhib/03`, `jac/01`, `motif/03`, `nn/replication`, `tarjan/03` |
| `np.linalg.eig` | Dominant eigenpair, left/right vectors, first-order eigenvalue perturbation. | Eigen*values* are transpose-invariant; **eigenvectors are not**. Right = response/receiving pattern, left = driving pattern. Every site here uses the vectors. | `inhib/01`, `inhib/02`, `jac/01`, `motif/04`, `nn/replication`, `schur/07`, `schur/08`, `tarjan/01`, `tarjan/02`, `tarjan/04` |
| `scipy.linalg.eig(..., left=True)` | Explicit left/right eigenvector split for the corrected left-action analysis. | The whole point of the call is the left/right distinction. | `schur/01`, `schur/070826_*` |
| `scipy.linalg.schur` | The core decomposition: ordered orthonormal mode basis `Q` plus upper-triangular coupling `T`. | `Q` columns are the ordered modes; upper-triangular `T` encodes *downstream* coupling. Transposing turns "upstream drivers first" into "downstream receivers first". | `schur/01`, `03`, `04`, `05`, `070826_*`, `motif/03`, `motif/triad_modulation`, `tarjan/01`, `02`, `03` |
| `scipy.linalg.solve_continuous_lyapunov` | Infinite-horizon controllability Gramian `AW + WA^H + BB^H = 0`. | Gramians are defined for the state matrix. | `schur/09` |
| `scipy.linalg.solve` (inside `frequency_response`) | Transfer function `G(iω) = C(iωI − A)^{-1}B`. | Input→output map. | `schur/09`, `schur/10` |
| `scipy.linalg.subspace_angles` | Angle between the slow Schur subspace and a Laplacian/flow subspace. | Inherits the requirement from `schur`. | `tarjan/01`, `tarjan/02` |
| `np.sum(..., axis=0/1)` on `M` | In-strength / out-strength, disinhibitory drive, degree normalization. | In `[post, pre]`: `axis=0` is a source's out-strength, `axis=1` a target's in-strength. Transposing swaps them silently. | `inhib/02`, `schur/07`, `schur/08`, `motif/04`, `tarjan/04` |
| `networkx.DiGraph.add_edge` | Build the directed graph the whole Tarjan/ordering analysis runs on. | Edge direction is the entire content. | `tarjan/01`, `tarjan/02` |
| `nx.pagerank`, `strongly_connected_components`, `condensation`, `topological_sort`, `is_directed_acyclic_graph`, `relabel_nodes`, `draw_networkx` | SCC decomposition, condensation DAG, flow ordering, plotting. | They consume the `DiGraph`, not the matrix — they inherit orientation from `graph_from_matrix`. | `tarjan/01`, `tarjan/02` |

### Your question: what "converting matrix orientation to graph orientation" means

The two data structures index direction in opposite ways, and `graph_from_matrix`
**[CUSTOM]** (`tarjan_reorder/matrix_reordering_paper_analysis_script.py:182`) is
the single place the repo translates between them.

- **Matrix convention here:** `M[post, pre]` — the *row* is the target, the
  *column* is the source. `M[i, j] ≠ 0` means `j → i`.
- **NetworkX convention:** `G.add_edge(u, v)` always means `u → v`. The first
  argument is the source.

So they are mirror images: in the matrix the source is the *second* index, in the
graph the source is the *first* argument. The conversion has to flip the pair:

```python
for post, pre in zip(*np.nonzero(np.abs(M) > theta)):   # (row, col) = (post, pre)
    G.add_edge(labels[pre], labels[post], ...)          # (source, target) = (pre, post)
```

Read it as: *"`np.nonzero` hands me (row, col); row is the target and col is the
source; NetworkX wants (source, target); so I pass them in the other order."*

That one swap is what makes `nx.topological_sort` return upstream cell types
first rather than last, and `nx.pagerank` rank cells by incoming influence rather
than outgoing. If you ever fed `graph_from_matrix` a `[pre, post]` matrix without
changing the function, every SCC would be identical (SCCs are symmetric under
edge reversal) but every *ordering* would come out reversed — which is the
failure mode that would look most plausible and be hardest to notice.

---

## 2. Library functions that are orientation-invariant

They return identical values for `M` and `M.T`. **Invariance is a property of
the number, not of the sentence you write about it.** The last column is the
important one.

| Function | Purpose | Invariant because | Standard interpretation of the result | How to orient your input | Notebooks |
|---|---|---|---|---|---|
| `np.linalg.eigvals` | Spectral radius ρ, spectral abscissa, dominant λ. | `spec(M) = spec(Mᵀ)`. | Asymptotic growth: ρ > 1 (discrete) or Re λ > 0 (continuous) means a mode grows. Names **no cell** — it is a property of the whole network. | `[post, pre]`. The number is safe either way, but the eigen*vector* you pull from the same decomposition is not, and you will pull one. | 29 notebooks — effectively all |
| `np.linalg.eigvalsh((M + Mᵀ)/2)` | Numerical abscissa. | The symmetric part is transpose-invariant. | Maximum *instantaneous* growth rate at t = 0. If it exceeds the spectral abscissa the system amplifies transiently — the non-normality signature. | `[post, pre]`. It is normally read next to `expm`, which is **not** invariant. | `EE/ei_backbone`, `inhib/01`, `nn/replication`, `schur/09`, `schur/10` |
| `np.linalg.norm` (fro, 2) | Total weight, one-step worst-case gain, residuals. | Invariant. | Frobenius = total connection energy; 2-norm = worst-case single-step amplification over all unit inputs. | Either. Keep `[post, pre]` so the norm and the vectors beside it describe the same object. | 25 notebooks |
| `np.linalg.svd` / `scipy.linalg.svdvals` | Low-rank structure; optimal transient input/output directions. | **Singular values only.** | σ₁ = best-case one-step gain. `U` columns = *response* directions in post-space; `Vᵀ` rows = *drive* directions in pre-space. | **Must be `[post, pre]`** whenever you use `U` or `Vt`. Transposing swaps them, so "which cells respond" silently becomes "which cells drive". | `jac/01`, `motif/03`, `nn/replication`, `schur/01`, `09`, `10`, `070826_*`, `tarjan/03` |
| `np.linalg.cond` | Conditioning of `Q`, eigenvector matrix, `I − block`. | Invariant. | High condition number on the eigenvector matrix = near-defective, strongly non-normal. Names no cell. | Either. | `inhib/01`, `jac/01`, `motif/03`, `nn/replication`, `schur/01`, `070826_*`, `tarjan/03` |
| `np.linalg.matrix_rank` | Gramian rank / effective controllable dimension. | Invariant. | How many independent directions the input actually reaches. | `[post, pre]` — the Gramian it is computed from is orientation-dependent even though its rank is not. | `schur/09`, `schur/10` |
| `scipy.sparse.csgraph.reverse_cuthill_mckee` | Bandwidth-reducing baseline ordering. | Called on `B = max(B, Bᵀ)` — direction is deliberately discarded first. | A permutation that clusters edges near the diagonal. It is a *sparsity* ordering, **not** a flow ordering; do not read it as upstream→downstream. | Either. | `tarjan/01` |
| `scipy.sparse.linalg.eigsh` | Modes of the motif×motif signed Jaccard operator. | Operates on a symmetric motif-similarity operator, not on connectivity. | Groups of motifs that co-occur with consistent sign structure. | n/a — orientation never reaches this operator. The *motif ids* feeding it do carry orientation (§4). | `motif/04`, `tarjan/04` |
| `scipy.stats.kendalltau`, `zscore`, `norm` | Compare orderings; standardize enrichment scores. | Operate on rank/score vectors. | Rank agreement between two orderings; z-score vs a null. | n/a directly — but the *orderings being compared* are orientation-dependent, so both must come from the same convention. | `tarjan/01`, `motif/triad_modulation` |
| `pulp` MILP solve | Exact maximum-weight backbone path. | Operates on the adjacency dict. | The heaviest directed walk through the E subnetwork. | Orientation enters via `build_adjacency` **[CUSTOM]**, which is `[pre, post]` (§4). | `EE/method_comparison` |

### The nuance you are worried about, stated plainly

Transposing the matrix changes **none** of the scalars above and **all** of the
following:

| Invariant scalar | The sentence you actually write about it — not invariant |
|---|---|
| ρ = 0.94 | *"the network is stable"* ✅ safe |
| dominant λ | *"the slowest mode is carried by CA3 Pyramidal"* ❌ that is the eigen**vector** |
| numerical abscissa > abscissa | *"the network amplifies transiently"* ✅ safe |
| σ₁ | *"the optimal input pattern is …"* ❌ that is `Vᵀ`; *"the response pattern is …"* ❌ that is `U` |
| Kendall τ between two orders | *"PageRank and Tarjan agree on flow direction"* ❌ both orders reverse together, τ stays put, and the shared direction may be backwards |
| triad counts | *"motif 7 is enriched"* ❌ the motif **id** depends on convention (§4) |

Rule of thumb: **if the claim names a cell type, a direction, or an ordering, it
is not protected by invariance.** The safe move is to feed `[post, pre]`
everywhere regardless, because it costs nothing and the invariant functions are
never used alone — they sit next to a vector, a label, or an ordering that is
orientation-dependent.

---

## 3. Custom functions that require `M[post, pre]`

| Function(s) | Purpose | File:line | Notebooks |
|---|---|---|---|
| `block_view`, `zero_block`, `block_meaning` | Slice / ablate / label the four E-I blocks by source→receiver name. | `schur_decomp/inhibitory_schur_modulation_script.py:44,56,112` | `schur/07`, `schur/08` |
| `schur_effective_E` | Eliminate inhibitory nodes: `M_EE + M_EI (zI − M_II)^{-1} M_IE`. | `…inhibitory_schur_modulation_script.py:122` | `schur/07`, `schur/08` |
| `excitation_response` | Steady-state response to driving all E or all I nodes. | `…:154` | `schur/07`, `schur/08` |
| `neumann_ii_model_selection` | Choose how many chained I-I links to retain in feedback. | `…:435` | `schur/07`, `schur/08` |
| `disinhibition_sources` | Rank inhibitory cells by stabilizing disinhibitory feedback. | `…:581` | `schur/07`, `schur/08` |
| `enumerate_eie_configurations`, `enumerate_eiie_configurations` | Enumerate strongest E→I→E and E→I→I→E routes; index as `M[receiver, source]`. | `…:612,659` | `schur/07`, `schur/08` |
| `motif_edge_masks`, `motif_counts`, `motif_stability_table` | Build motif edge masks and measure stability on removal. | `…:258,395,226` | `schur/07`, `schur/08` |
| `dominant_eigenpair` | Dominant eigenvalue with **both** left and right eigenvectors. | `…:67` | `schur/07`, `schur/08` |
| `schur_decomp` | Run `scipy.linalg.schur` and package `T`, `Q`, eigenvalues. **Has `transpose=True` default** — it expects `[pre, post]` and flips for you. | `schur_decomp/schur_analysis_utils_script.py:107` | `schur/01`, `schur/070826_*` |
| `prepare_left_action_schur` | Build the corrected state matrix and its Schur/eigen decompositions. | `…schur_analysis_utils_script.py:770` | `schur/01`, `schur/070826_*` |
| `build_upstream_schur_modes`, `upstream_mode_summary`, `plot_schur_t_heatmap` | Group conjugate modes in upstream-first order; visualize `Re(T)` as receivers×sources. | `…:871,983,537` | `schur/01`, `schur/070826_*` |
| `first_schur_station_df`, `compute_ei_routing_ratios`, `build_mode_routing_table`, `print_motif_routing`, `active_edge_counts` | Top contributors in the first Schur vector; E/I loop-vs-leak routing per mode. | `…:552,260,315,468,306` | *(defined, not currently reached by any notebook)* |
| `analyze_discrete_transient_dynamics` | Finite-horizon gain curve and a discrete Kreiss-style proxy. | `…:1034` | `schur/01`, `schur/070826_*` |
| `matrix_power_series`, `enumerate_power_top_paths`, `node_path_scores` | `M^k` path weights and per-node in/out path scores. | `schur_decomp/network_linear_systems_tools_script.py:146,196,209` | `schur/09`, `schur/10` |
| `make_B`, `make_C`, `controllability_gramian_discrete`, `controllability_gramian_continuous`, `controllability_scores`, `frequency_response`, `make_stable_continuous_A` | Input/output selection matrices, Gramians, transfer function. | `…network_linear_systems_tools_script.py:229,255,281,324,352,397,369` | `schur/09`, `schur/10` |
| `schur_mode_loadings`, `propagate_signal`, `propagate_each_schur_mode`, `propagate_mode_cell_types` | Per-cell mode loadings; discrete-time propagation through the Schur system. | `schur_decomp/schur_signal_propagation_script.py:167,287,312,352` | `schur/03` (loadings); propagation fns not currently reached |
| `block_mean_matrix`, `centered_by_blocks`, `binary_motif_enrichment` | E/I block means, residual `Z`, motif enrichment vs block-density null. All index `np.ix_(receiver_mask, sender_mask)`. | `non-normal_matrices/analyses/connectivity_methods.py:210,415,679` | `nn/replication` |
| `low_rank_matrix_diagnostics`, `low_rank_response` | SVD reconstruction error and low-rank population response — **uses `U`/`Vt`, not just singular values**. | `…connectivity_methods.py:613,584` | `nn/replication` |
| `schur_complement`, `neumann_series_feedback`, `excitation_response`, `motif_counts`, `neumann_ii_model_selection` | The `inhib_modulation` twins of the above; all index `matrix.loc[post, pre]`. | `inhib_modulation/notebooks/inhibitory_modulation.py:341,553,843,887,1135` | `inhib/02` |
| `graph_from_matrix`, `apply_order`, `pagerank_order`, `dfs_finish_order`, `tarjan_order` | Matrix→DiGraph adapter and the four orderings; block-triangularity metrics. | `tarjan_reorder/matrix_reordering_paper_analysis_script.py:182,190,199,206,289` | `tarjan/01`, `tarjan/02` |

### Your question: were these written after the transposition issue was noticed?

Partly — and the repo carries direct evidence. `schur_decomp/archive/` contains
`eigenmode_analysis_mij_schur_corrected_left_action.ipynb` alongside an
uncorrected `eigenmode_analysis_mij_schur.ipynb`. So the left-action /
orientation fix was a **retrofit** on the `schur_decomp` side, and
`prepare_left_action_schur` ("the *corrected* state matrix") is the function that
retrofit produced.

The fingerprint of the retrofit is `schur_decomp(transpose=True)`: rather than
change the loaders, the fix put the transpose inside the decomposition step. That
is why `load_connectivity_matrix` still returns `[pre, post]` while everything
downstream of it is `[post, pre]` — the seam is the `transpose=True` default, not
the loader. `jacobian_core` is the opposite case: written after the convention
was settled, it transposes and verifies at load and never flips again.

Git history cannot confirm the ordering more precisely — the repo has 4 bulk
commits, so per-function dates are not recoverable.

---

## 4. Custom functions that currently use `M[pre, post]`

| Function(s) | Purpose | File:line | Notebooks | Convert to `[post, pre]`? |
|---|---|---|---|---|
| `infer_ei_from_signed_rows` | Infer E/I from the sign of each source's **outgoing row**. | `schur_decomp/schur_core_script.py:136` | `schur/03`, `05`, `07`, `08` | **No — must stay.** It reads rows as outgoing. In `[post, pre]` rows are incoming, so converting inverts every E/I label. Its `[pre, post]` input is a correctness requirement, not an oversight. |
| `classify_sender_cell_class` | Same, for the non-normal pipeline. | `non-normal_matrices/analyses/connectivity_methods.py:96` | `nn/replication` | **No — same reason.** |
| `build_greedy_tree` | Greedy directed tree over a mode's top contributors; reads `submatrix.loc[source, receiver]`. | `schur_decomp/schur_mode_greedy_trees_script.py:146` | `schur/05` | **Optional.** Deliberately fed `data.df_source_receiver` at line 220 and self-consistent. Converting means flipping both the indexing and the caller together — a coordinated two-line change, low value. Leave it, but note it in the docstring: it is the only consumer of the raw frame inside `schur_decomp/`. |
| `load_matrix`, `outgoing_positive_fraction`, `is_excitatory_sender`, `prepare_backbone_data`, `build_adjacency`, `transition_score`, path/MILP search | EE backbone path search over the excitatory subnetwork. | `EE_backbone/ee_backbone_analysis.py:54,61,68,76,193,213` | `EE/method_comparison`, `EE/det_stochastic_comparison` | **Not now** (your call — deferred for the stochastic revamp). It is fully self-consistent in `[pre, post]`. The risk is only at the boundary: any comparison of EE-backbone paths against Schur or Tarjan orderings crosses a convention line. |
| `_triad_motif_id`, `_two_edge_triad_occurrences`, `count_triad_motifs`, `_degree_preserving_directed_null` | Triad census and degree-preserving null. Index `matrix[source, target]`. | `motif_analysis/triad_utils.py:36,51,73,113` | `motif/triad_modulation` | **Yes, eventually.** Counts are invariant, but `motif_id` is an ordered edge-sign tuple `(i→j, j→i, i→k, k→i, j→k, k→j)` — the same physical motif gets a **different id** under the two conventions. This is the one place in §4 where the mismatch produces silently non-comparable *labels*. Converting is a one-line change at the call site in `triad_modulation_analysis.ipynb` (pass `M.values.T`) plus re-running; but every stored motif id then changes, so do it once, deliberately, and regenerate downstream tables. |
| `measure_triad_perturbations`, `analyze_triad_schur_complements` | Eigenvalue shift on triad-class removal; 3×3 Schur spectra. | `motif_analysis/triad_utils.py:234,319` | `motif/triad_modulation` | **No change needed.** Both use eigenvalues only, which are invariant. They are the "invariant functions fed by early functions" you spotted — safe. |
| `load_schur_blocks`, `load_blocks_and_metadata` | Load exported 3×3 motif blocks as stored. | `motif_analysis/motif_schur_decomposition_analysis_script.py:47`; `motif_analysis/motif_signed_overlap_analysis_script.py:13` | `motif/03`, `motif/04`, `tarjan/03`, `tarjan/04` | **Yes — highest value in this table.** Correct today only because notebooks set `TRANSPOSE_SOURCE_TARGET_BLOCKS=True` afterwards. Now duplicated across two folders and four notebooks, so there are four places for that flag to drift. Move the transpose into the loader behind a required `stored_orientation=` argument. |

**Your read is right:** most of §4 is early code that happens to feed invariant
consumers, which is why none of it is producing wrong numbers. The two worth
acting on are the ones whose output is a *label* rather than a number — triad
`motif_id`, and the motif-block loaders — because a wrong label is not caught by
invariance.

---

## 5. The `EI` / `IE` collision — and which reading is correct

**Short answer to your assumption: it depends on which function you are reading,
and for `M_EI` in the Schur complement your assumption is reversed.**

Two naming systems are in play:

| System | Reading | `EI` means | Used by |
|---|---|---|---|
| **Arrow** (source-then-receiver) | first letter = source | **E source → I receiver** | `block_view`, `zero_block`, `block_meaning`, `block_perturbation_table` |
| **Partition** (row-block, column-block) | in `M[post,pre]`, row = receiver, col = source, so **second letter = source** | **I source → E receiver** | `schur_effective_E`, the Neumann functions, `disinhibition_sources` |

Your instinct (`M_EI` = E source → I receiver) is the **arrow** reading. It is
correct for `block_view`. It is backwards for `M_EI` inside `schur_effective_E`.

### Verified on your data

Slicing `M[post, pre]` and reading the sign — Dale's law makes this decisive,
since an inhibitory source produces a wholly negative block:

| Slice | Shape | % positive | Therefore the source is | Direction |
|---|---|---|---|---|
| `M[np.ix_(E,E)]` | 32×32 | 100.0 | excitatory | E → E |
| `M[np.ix_(E,I)]` | 32×53 | **0.0** | **inhibitory** | **I → E** |
| `M[np.ix_(I,E)]` | 53×32 | **100.0** | **excitatory** | **E → I** |
| `M[np.ix_(I,I)]` | 53×53 | 0.0 | inhibitory | I → I |

So `M[np.ix_(E,I)]` — which `schur_effective_E` calls `M_EI` — is the
**inhibitory feedback onto E**, not E's drive onto I.

### Does this affect how the other blocks are read for Schur analysis?

Yes, and the shapes enforce it. The eigenvalue-dependent Schur complement is

```
M_eff(z) = M_EE + M_EI (zI − M_II)^-1 M_IE
             (32×32)   (32×53)  (53×53)   (53×32)
```

The dimensions only conform one way: `M_EI` **must** be 32×53 (rows E, cols I =
I→E) and `M_IE` **must** be 53×32 (rows I, cols E = E→I). Running it with the
blocks swapped is not a subtle error — it raises a shape mismatch immediately.

Checked numerically: with `z` = the dominant eigenvalue of the full `M`, the
defining property is that `z` must also be an eigenvalue of `M_eff(z)`.

| Assignment | Distance from `z` to nearest eigenvalue of `M_eff` | Verdict |
|---|---|---|
| As coded: `M_EI = M[E,I]`, `M_IE = M[I,E]` | `2.19e-11` | **recovers z — correct** |
| Swapped | shape error, 53 vs 32 | cannot even be evaluated |

**So the code is right and the mathematics is unambiguous.** Reading the formula
left to right in biological terms: *E drives I (`M_IE`), the I population
resolves that drive through its own recurrent loops (`(zI − M_II)^{-1}`), and
that resolved inhibition lands back on E (`M_EI`).* The blocks appear in the
formula in **reverse causal order**, which is exactly why the notation reads
backwards to intuition.

The fix is not to change either function — both are correct — but to stop using
the same four letters for two things. Arrow names (`E_to_I`) for anything a human
reads; bracket names (`M[E,I]`) for the algebra; never a bare `EI`.

---

## 6. Status of the previously suggested work

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Pin down the orientation of `analysis_blocks` in the motif `.npz` | ❌ **Not done, and now wider** | `load_schur_blocks` docstring still reads *"callers choose any orientation transpose"*. The script is now duplicated into `motif_analysis/`, so the notebook flag `TRANSPOSE_SOURCE_TARGET_BLOCKS` exists in four notebooks instead of two. |
| 2 | Rename the `EI`/`IE` collision | ❌ **Not done** | `block_view` still returns `"EI": M[np.ix_(I, E)]` at `inhibitory_schur_modulation_script.py:50`, while `schur_effective_E:133` still uses the opposite `M_EI`. |
| 3 | Decide EE-backbone convention | ⚠️ **Half done — and the half that is done was done well** | `ei_backbone_analysis_helpers.py:13` (`load_signed_matrix`) now takes an explicit `csv_orientation="source_rows"` (line 15) and transposes at line 40. The **EE** side (`ee_backbone_analysis.py`) is still `[pre, post]` with no note — deferred by your decision, pending the stochastic revamp. |
| 4 | Give `schur_decomp(transpose=True)` an explicit orientation parameter | ❌ **Not done** | `schur_analysis_utils_script.py:107` still has the bare `transpose: bool = True` default; `prepare_left_action_schur:790` still passes `transpose=True` positionally. |
| 5 | Adopt `load_inputs`-style validation across loaders | ✅ **Superseded — better than proposed** | `jacobian_core.assert_post_pre:115` implements the Dale's-law sign oracle, and `load_jacobian_data:169` calls it under `verify_orientation=True`, storing the result on `JacobianData.orientation_check`. Currently used by `jac/01`–`jac/06` only; the other eight loaders still assume. See §0 for how to extend it. |

### Revised priority

1. **Motif block loaders** (item 1) — now four notebooks deep, and it is the only
   remaining case where the contract lives outside the code.
2. **Triad `motif_id` convention** (§4) — the other label-not-number risk.
3. **`EI`/`IE` rename** (item 2) — no wrong results today, but it is the trap most
   likely to catch a reader.
4. **Extend `assert_post_pre` to the eight assuming loaders** (item 5, step 2 in
   §0) — two lines each, no behavior change on correct data.
5. **`schur_decomp(transpose=)`** (item 4) — lowest urgency; correct today, but it
   is the seam the original retrofit left behind.
