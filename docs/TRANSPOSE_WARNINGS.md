# ⚠ TRANSPOSE WARNINGS

Read this before running any analysis or reusing any matrix file.
No code changes are implied by this document — it records where transposes
already happen, where they must happen, and where they are invisible.

**The rule:** if a function interprets `M[i, j]` as the influence of cell `j`
on cell `i`, it needs `M[post, pre]` — rows are postsynaptic targets, columns
are presynaptic sources.

**Every `mij_matrix.csv` in this repo is `M[pre, post]` and must be transposed
before dynamics, graph construction, or mode interpretation.** This was
verified, not assumed — see §F.

---

## A. Files on disk — verified orientations

Checked by E/I sign purity against `mij_netlist.csv` (85 cells, 32 E, 53 I, 0
unknown). Separation is 100%/0%, so these verdicts are unambiguous.

| File | Orientation | Transpose before analysis? |
|---|---|---|
| `EE_backbone/matrices/mij_matrix.csv` | `M[pre, post]` | ⚠ **YES** |
| `inhib_modulation/matrices/mij_matrix.csv` | `M[pre, post]` | ⚠ **YES** |
| `motif_analysis/mij_matrix.csv` | `M[pre, post]` | ⚠ **YES** |
| `non-normal_matrices/data/mij_matrix.csv` | `M[pre, post]` | ⚠ **YES** |
| `non-normal_matrices/data/wij_matrix.csv` | `M[pre, post]` | ⚠ **YES** |
| `schur_decomp/mij_matrix.csv` | `M[pre, post]` | ⚠ **YES** |
| `schur_decomp/matrices/mij_matrix.csv` | `M[pre, post]` | ⚠ **YES** |
| `tarjan_reorder/mij_matrix.csv` | `M[pre, post]` | ⚠ **YES** |
| `schur_decomp/outputs/moved_csv/matrices/wij_matrix.csv` | `M[pre, post]` | ⚠ **YES** |
| `EE_backbone/matrices/mij_EE_matrix.csv` | 32×32 E-only; no I cells, so the sign test cannot decide | ⚠ **UNVERIFIED — treat as `[pre, post]` to match its siblings, and confirm before reuse** |
| `**/outputs/mij_paper_replication*/normalized_J_receiver_by_sender.csv` | `M[post, pre]` | ⚠ **NO — ALREADY TRANSPOSED** |
| `**/outputs/mij_paper_replication*/normalized_mij_sender_by_receiver.csv` | `M[pre, post]` | ⚠ **YES** |

### ⚠ Highest-risk pair in the repo

These two sit in the same directory, differ by one word in the filename, and
are transposes of each other:

```
non-normal_matrices/outputs/mij_paper_replication/
    normalized_J_receiver_by_sender.csv     <- M[post, pre]   DO NOT transpose
    normalized_mij_sender_by_receiver.csv   <- M[pre, post]   MUST transpose
```

The same pair is duplicated in five `mij_paper_replication_*check*` output
directories. Any of the ten is one tab-completion away from the wrong one, and
loading the wrong one produces a plausible-looking result with reversed
causation. Read the filename to the end.

---

## B. Where the transpose happens — by entry point

"Visible" means you can see the transpose in the code you are calling.
"⚠ Invisible" means it happens inside a default argument or several frames down.

| You call | Input it expects | Transposes? | Analysis sees | Note |
|---|---|---|---|---|
| `load_mij_data(...)` `schur_core_script.py:147` | `[pre, post]` CSV | Yes, `.T` at line 188 | `M[post, pre]` | Visible in docstring |
| `load_source_receiver_matrix(...)` `schur_core_script.py:130` | `[pre, post]` CSV | Yes | `M[post, pre]` | Name states it |
| `load_mij_matrix(...)` `schur_core_script.py:118` | `[pre, post]` CSV | **No** | `[pre, post]` frame | ⚠ Returns the raw frame. Same name as the transposing loader in `connectivity_methods.py` — see §D |
| `load_mij_matrix(...)` `connectivity_methods.py:32` | `[pre, post]` CSV | Yes, `J_paper = raw.T` | `J[post, pre]`, ρ-normalized | Visible |
| `load_inputs(...)` `matrix_reordering_paper_analysis_script.py:78` | either | Yes, and **verifies** against netlist | `M[post, pre]` | Only loader that validates. Use as the model |
| `load_source_receiver_csv(...)` `network_linear_systems_tools_script.py:60` | `[pre, post]` CSV | Yes, line 87 | `M[post, pre]` | Visible |
| `load_connectivity(...)` `inhibitory_modulation.py:110` | either | Yes, per `matrix_orientation=` (default `"pre_by_post"`) | `M[post, pre]` | Visible via argument |
| `prepare_schur_model(...)` `schur_signal_propagation_script.py:58` | `[pre, post]` CSV | Yes, line 74 | `M[post, pre]` | Visible |
| `prepare_local_schur_model(...)` `schur_mode_greedy_trees_script.py:64` | `[pre, post]` CSV | Yes, line 78 | `M[post, pre]` | Visible |
| `load_connectivity_matrix(...)` `schur_analysis_utils_script.py:18` | `[pre, post]` CSV | **No** | `[pre, post]` frame | ⚠ Used by notebooks `01`, `02`, `070826_*`. The transpose comes later — next row |
| `prepare_left_action_schur(df)` `schur_analysis_utils_script.py:770` | `[pre, post]` frame | Yes — but via `schur_decomp(..., transpose=True)` at line 790 | `M[post, pre]` | ⚠ **Invisible.** Correct today, but the transpose lives in a default argument two frames down |
| `schur_decomp(m)` `schur_analysis_utils_script.py:107` | `[pre, post]` array | Yes — **`transpose: bool = True` default** | `M[post, pre]` | ⚠ **Invisible and dangerous.** Handing it an already-transposed matrix silently double-flips. See §D |
| `load_schur_blocks(csv, npz)` `motif_schur_decomposition_analysis_script.py:47` | 3×3 blocks as stored | **No** | as stored | ⚠ Notebooks `03`/`04` transpose afterwards via `TRANSPOSE_SOURCE_TARGET_BLOCKS=True`. Correct today, but the contract is a notebook boolean whose right value depends on a producer notebook outside this repo |
| `load_blocks_and_metadata(csv, npz)` `motif_signed_overlap_analysis_script.py:13` | 3×3 blocks as stored | **No** | as stored | ⚠ Duplicate of the row above; same warning |
| `load_matrix(...)` `ee_backbone_analysis_script.py:54` | `[pre, post]` CSV | **No — and never does** | `[pre, post]` | ⚠ EE-backbone runs entirely in `[pre, post]`. Self-consistent. Deliberately deferred pending the stochastic revamp — see §E |

---

## C. ⚠ Functions that take a bare matrix — nothing protects you

These accept an array or frame with no orientation argument and no loader in
front of them. Whatever you hand them is what they believe.

| Function | File | Assumes |
|---|---|---|
| `block_view`, `zero_block`, `schur_effective_E`, `excitation_response`, `neumann_ii_model_selection`, `disinhibition_sources`, `enumerate_eie_configurations`, `enumerate_eiie_configurations`, `motif_edge_masks`, `motif_counts`, `motif_stability_table`, `dominant_eigenpair`, `block_perturbation_table` | `inhibitory_schur_modulation_script.py` | `M[post, pre]` |
| `matrix_power_series`, `enumerate_power_top_paths`, `node_path_scores`, `make_B`, `make_C`, `controllability_gramian_discrete`, `controllability_gramian_continuous`, `controllability_scores`, `frequency_response`, `make_stable_continuous_A` | `network_linear_systems_tools_script.py` | `M[post, pre]` |
| `compute_ei_routing_ratios`, `build_mode_routing_table`, `print_motif_routing`, `active_edge_counts`, `analyze_discrete_transient_dynamics`, `build_upstream_schur_modes`, `upstream_mode_summary`, `first_schur_station_df`, `plot_schur_t_heatmap` | `schur_analysis_utils_script.py` | `M[post, pre]` |
| `propagate_signal`, `propagate_each_schur_mode`, `propagate_mode_cell_types`, `schur_mode_loadings` | `schur_signal_propagation_script.py` | `M[post, pre]` |
| `block_mean_matrix`, `centered_by_blocks`, `binary_motif_enrichment`, `low_rank_matrix_diagnostics`, `low_rank_response`, `effective_connectivity`, `response_matrix`, `population_response` | `connectivity_methods.py` | `J[post, pre]` |
| `apply_order`, `dfs_finish_order`, `tarjan_order`, `pagerank_order`, `graph_from_matrix` | `matrix_reordering_paper_analysis_script.py` | `M[post, pre]` |
| `count_triad_motifs`, `_two_edge_triad_occurrences`, `_degree_preserving_directed_null`, `assess_triad_enrichment`, `analyze_triad_schur_complements`, `count_triads_by_region` | `triad_utils.py` | ⚠ **`M[pre, post]`** — indexes `matrix[source, target]`. See §D |
| `build_greedy_tree` | `schur_mode_greedy_trees_script.py:146` | ⚠ **`M[pre, post]`** — indexes `submatrix.loc[source, receiver]`, correctly fed `data.df_source_receiver` at line 220 |

### ⚠ SciPy/NumPy calls with no wrapper at all

`EE_backbone/ei_backbone_analysis.ipynb` calls `scipy.linalg.expm`,
`expm_multiply`, `eigvals` and `svdvals` directly. It does transpose — cell 5,
`W_state_raw = W_csv.T` guarded by `CSV_ORIENTATION = "source_rows"`. That flag
is the only thing standing between this notebook and reversed impulse
responses. `schur_decomp/09_linear_systems.ipynb` imports `scipy.linalg`
directly as well.

---

## D. ⚠ Named double-transpose and mislabelling traps

**1. Two different `load_mij_matrix` functions.**
`schur_core_script.py:118` returns the raw `[pre, post]` frame.
`connectivity_methods.py:32` returns a transposed, ρ-normalized `J[post, pre]`.
Same name, opposite behavior, both imported by name in notebooks. Check which
module you imported from.

**2. `schur_decomp(transpose=True)` is a silent flipper.**
`schur_analysis_utils_script.py:107`. It expects `[pre, post]` and transposes
for you. Passing it output from `load_mij_data` or any `.T`-ed array returns you
to `[pre, post]` with no error and no warning. Every mode, loading and routing
statement downstream is then reversed.

**3. `motif_analysis/triad_modulation_analysis.ipynb` feeds `[pre, post]`.**
Cell 0 reads `mij_matrix.csv` and passes `M.values` straight into `triad_utils`
with no transpose. The *numbers* survive: triad counts are transpose-invariant,
and `measure_triad_perturbations` / `analyze_triad_schur_complements` use only
eigenvalues, which are transpose-invariant too. What does not survive is the
**`motif_id` vocabulary** — the edge-sign tuple in `_triad_motif_id` is ordered
`(i→j, j→i, i→k, k→i, j→k, k→j)`, so the same physical motif gets a different
id under the two conventions. ⚠ Do not join `motif_analysis` motif ids against
`tarjan_reorder` or `inhib_modulation` motif tables without remapping.

**4. `EI` means two opposite things.**
Arrow sense — `EI` = E source → I receiver — in `block_view`, `zero_block`,
`block_meaning`, `block_perturbation_table` (`inhibitory_schur_modulation_script.py`)
and in `source_receiver_block_view`, `zero_source_receiver_block`,
`block_meaning` (`inhibitory_modulation.py`).
Partition sense — `M_EI` = `M[E rows, I cols]` = I source → E receiver — in
`schur_effective_E`, the Neumann functions, `disinhibition_sources`, and in
`BlockMatrices.ei` / `make_ei_blocks` / `summarize_blocks` / `scale_block` /
`ablation_analysis`.
Both are internally consistent and no current output is wrong, but
`inhibitory_modulation.py` can emit both in one notebook run: cell 10 of
notebook `02` prints a `block` column in the arrow sense while
`summarize_blocks` prints a `name` column in the partition sense. ⚠ When
reading any table with an `EE/EI/IE/II` column, check which function produced it.

**5. Canonical exports referenced but not present.**
`tarjan_reorder/ORIENTATION_AUDIT.md` points new scripts at
`outputs/paper_matrix_reordering/canonical_inputs/mij_matrix_paper_convention_*.csv`.
`tarjan_reorder/outputs/` does not exist yet, so those files are not on disk until
notebook `01` is re-run. ⚠ Do not assume a pre-transposed canonical input is
on disk.

---

## E. Orientation-sensitive settings that are not transposes

These change meaning under transposition even though no `.T` is involved.

- **`column_l1` / `source_l1` vs `row_l1` normalization**
  (`normalize_state_matrix`, `schur_core_script.py:223`). In `M[post, pre]`,
  columns are outgoing source channels, so `column_l1` normalizes out-strength
  and `row_l1` normalizes in-strength. Under `[pre, post]` the two swap. ⚠
  `spectral_radius` normalization is transpose-invariant and is the safe default.
  Note that `prepare_left_action_schur` normalizes the matrix *before*
  transposing it (lines 782-790) — harmless for `spectral_radius`, meaning-changing
  for `column_l1` or `row_l1`.
- **`np.sum(..., axis=0)` vs `axis=1` on a connectivity matrix.** In
  `M[post, pre]`: `axis=0` gives a source's total outgoing strength, `axis=1`
  gives a target's total incoming strength. Transposing swaps in-degree and
  out-degree silently. Live at `inhibitory_schur_modulation_script.py:596,598,599`,
  `inhibitory_modulation.py:901-902,1302`, `motif_signed_overlap_analysis_script.py:204`.
- **E/I inference from weight signs.** `infer_ei_from_signed_rows`
  (`schur_core_script.py:136`) and `classify_sender_cell_class`
  (`connectivity_methods.py`) both read *rows of the untransposed frame* as
  outgoing. ⚠ They take `df_source_receiver`, not `M`. Handing either one a
  `M[post, pre]` matrix inverts every E/I label.
- **EE-backbone is `[pre, post]` throughout** — `matrix.loc[node]` is treated as
  outgoing, `build_adjacency` reads `transition_weights[i, j]` as `i → j`.
  Self-consistent, deliberately left as-is pending the stochastic revamp. ⚠ Any
  comparison between EE-backbone paths and Schur/Tarjan output crosses a
  convention boundary.

---

## F. Paste-in orientation check (read-only, changes nothing)

E/I identity here comes from outgoing weight signs, so the block sign pattern
*is* the orientation. On this dataset the separation is total, which makes this
a decisive test rather than a heuristic:

| Block | `M[post, pre]` | `M[pre, post]` |
|---|---|---|
| E → E | 100% positive (n=382) | 100% positive (n=382) |
| E → I | 100% positive (n=352) | 0% positive (n=171) |
| I → E | 0% positive (n=171) | 100% positive (n=352) |
| I → I | 0% positive (n=736) | 0% positive (n=736) |

The two off-diagonal blocks swap completely. Paste this into any notebook cell
to confirm what you are holding:

```python
import numpy as np, pandas as pd

def check_orientation(M, labels, netlist_path):
    """Print whether M is [post,pre] or [pre,post]. Read-only."""
    net = pd.read_csv(netlist_path)
    ei = {}
    for cn, ce in [("pre_neuron", "pre_ei"), ("post_neuron", "post_ei")]:
        for n, e in net[[cn, ce]].dropna().drop_duplicates().itertuples(index=False):
            ei[str(n)] = str(e).lower()
    cls = np.array([ei.get(str(l), "?") for l in labels])
    E, I = np.flatnonzero(cls == "e"), np.flatnonzero(cls == "i")
    A = np.asarray(M, dtype=float)

    def frac_pos(rows, cols):
        nz = A[np.ix_(rows, cols)]
        nz = nz[nz != 0]
        return (nz > 0).mean() if nz.size else np.nan

    e_to_i = frac_pos(I, E)   # rows=I receivers, cols=E sources
    i_to_e = frac_pos(E, I)   # rows=E receivers, cols=I sources
    if e_to_i > 0.9 and i_to_e < 0.1:
        print(f"M[post, pre]  (E->I {e_to_i:.1%} positive, I->E {i_to_e:.1%} positive)")
    elif e_to_i < 0.1 and i_to_e > 0.9:
        print(f"M[pre, post]  -- TRANSPOSE BEFORE ANALYSIS "
              f"(E->I {e_to_i:.1%} positive, I->E {i_to_e:.1%} positive)")
    else:
        print(f"AMBIGUOUS (E->I {e_to_i:.1%}, I->E {i_to_e:.1%}) -- inspect manually")

# df = pd.read_csv("mij_matrix.csv", index_col=0)
# check_orientation(df.to_numpy(), df.index, "mij_netlist.csv")
```

---

## Related documents

- `docs/matrix_orientation_inventory.md` — which library and custom functions require `M[post, pre]`, and which are transpose-invariant
- `docs/orientation_convention_proposal.md` — proposed (not applied) changes for the motif loader and the `EI` collision
- `tarjan_reorder/ORIENTATION_AUDIT.md` — the existing per-notebook audit for that folder
