# Review: notebook methodology vs. Borst & Leibold (2023), plus full orientation audit

Scope: `01`–`04` and their four `_script.py` modules, against Borst & Leibold 2023,
*Connecting Connectomes to Physiology*, J Neurosci 43(20):3599–3610 (Figs. 5–7, Eqs. 3–10).

First pass (2026-09-04) was written before `mij_matrix.csv` and `mij_netlist.csv` were in the
repo, from code reading plus synthetic reproductions built to isolate each behavior. Those
files have since been added and the full pipeline has been run against them (85 cells, 1641
netlist rows, 1570 edges, spectral-radius normalized); findings confirmed or changed by that
run are marked inline and collected in Part D.

**Status legend:** `[FIXED]` implemented and verified · `[OPEN]` not yet addressed ·
`[CONFIRMED]` re-verified against the real data.

---

## Bottom line

The orientation convention is correct everywhere it is applied. `M[post, pre]` is used
consistently, feedforward lands in the lower triangle, and every pre/post index pair in
all four scripts checks out. `ORIENTATION_AUDIT.md` is accurate as a description of what
the code does.

The earlier orientation-guard defect has now been fixed: with a netlist present, the selected
explicit orientation must agree with the netlist reconstruction in `M[post, pre]` convention,
or the loader stops before the analysis runs. No-netlist runs still rely on the explicit
`MATRIX_ORIENTATION` setting.

Two smaller correctness issues: the Schur off-diagonal coupling direction is labeled backwards
in `03`, and the `04` signed Laplacian is dominated by a self-loop that should not be there.

**As of 2026-09-05**, the orientation question is settled empirically for the current dataset
(Part D1), the guard is fixed (Part A2), the Figure 7 recreation is implemented (Part D2),
and that work turned up a method-level finding that changes `01`'s results table (Part D3).
Everything else below remains open.

---

## Part A — Orientation audit

### A1. Verified correct

Traced by hand and confirmed against a synthetic ground-truth network
(`EC → DG → CA3 → CA1`, pure feedforward, stored source-rows/target-columns with a matching netlist):

| Location | Check | Verdict |
|---|---|---|
| `load_inputs` netlist reconstruction | `recon[pos[post], pos[pre]] += m_ij` | correct |
| `graph_from_matrix` | `np.nonzero` yields `(post, pre)`; `add_edge(labels[pre], labels[post])` | correct |
| `tarjan_order` | `condensation` → `topological_sort` puts source SCCs first; `apply_order` then places feedforward strictly below the diagonal | correct — verified: 0 upper-triangle entries on the synthetic feedforward net |
| `ordering_metrics` | upper `i<j` = backward; `np.diag(B,-1)` = `pre i → post i+1` chain | correct |
| `_local_feedback_cost` (modified Tarjan) | minimizes upper-triangle count then weighted index length | correct |
| `_degree_preserving_rewire` | edges as `(post, pre)`; swap `(p1,r1),(p2,r2) → (p1,r2),(p2,r1)` | correct — preserves both in- and out-degree |
| `trisynaptic_paths` | `M[dg,ec]`, `M[ca3,dg]`, `M[ca1,ca3]` | correct |
| `respects_directional_order` | `pos[EC] < pos[DG] < …` under a lower-triangular convention | correct |
| `condensation_model` (`02`) | `i=index(post); j=index(pre); w=M[i,j]`, aggregated over `u→v` | correct |
| `canonical_cell_paths` (`02`) | same three index pairs | correct |
| `modal_followup` lesion (`02`) | `Ml[post, pre] = 0` | correct |
| `normalized_laplacian`, `rcm_order` | `abs(M)+abs(M.T)` / `max(B,B.T)` — direction deliberately discarded | correct and correctly documented |
| `A = M - I` (`01`, `02`) | matches Eq. 5 with `T = 1` | correct |
| `03`/`04` `blocks.transpose(0,2,1)` | converts source-rows exports to `[post, pre]` | correct *given* the stated export orientation — see A3 |

**Nothing needs a manual transpose.** The two things a reader might expect to be wrong are not:
`(I − M)⁻¹` in `03` is the steady-state gain of Eq. 4, and its `[i,j]` entry is the response of
cell `i` to input at cell `j`, which is right for `M[post, pre]`. The right eigenvectors used
in `01`, `02`, and `04` are activity patterns, which is also right.

### A2. `[FIXED 2026-09-05]` Orientation guard now checks the selected orientation

Earlier, `01` and `02` hard-set `MATRIX_ORIENTATION = "pre_rows_post_columns"` while a netlist
was supplied, and `01` asserted only:

```python
assert min(residuals) < 1e-10
```

That asserted *one of the two* orientations matched the netlist. It did not assert that the
selected orientation was the matching one. Reproduced before the fix:

```
=== WRONG explicit orientation, netlist present ===
notebook cell-9 assertion min(residuals)<1e-10 PASSES: True
chosen matrix equals ground truth: False  ->  it is the TRANSPOSE: True
reordered order:  ['CA1 d', 'CA3 c', 'DG b', 'EC a']       # reversed
upper-triangle nonzeros: 0    lower: 3                     # still looks perfect
trisynaptic paths found under flipped orientation: 0
trisynaptic paths found under correct orientation: 1
```

A transposed run is invisible to every structural diagnostic in the notebook. Recurrency,
order index, feedforward fraction, and bandwidth are all identical, because the reversed
permutation is exactly as good. The only symptoms are a reversed cell order and the silent
disappearance of the trisynaptic paths — both of which read as a scientific result rather
than as a bug.

Implemented fix: `load_inputs(...)` now records `selected_orientation_residual_vs_netlist_post_pre`
and raises before analysis if a supplied netlist disagrees with the selected orientation beyond
`netlist_orientation_tolerance`. `01`, `02`, and `build_matrix_reordering_notebook.py` pass
`NETLIST_ORIENTATION_TOLERANCE = 1e-10`. `01`'s guard cell also checks the selected residual:

```python
if audit["netlist_path"] is not None:
    selected_residual = audit["selected_orientation_residual_vs_netlist_post_pre"]
    assert selected_residual < NETLIST_ORIENTATION_TOLERANCE
```

Remaining optional improvement for the no-netlist case: add a biological directionality assay.
In the hippocampal formation the perforant path and mossy fibers are unidirectional onto granule
and CA3 pyramidal cells, so in the correct orientation

```python
# should be strongly asymmetric in the correct orientation
fwd  = abs(M[dg_granule_rows][:, ec_cols]).sum()   # EC -> DG
back = abs(M[ec_rows][:, dg_granule_cols]).sum()   # DG -> EC
assert fwd > back
```

Note that nilpotency (`M**a == 0`) cannot serve this purpose — it is transpose-invariant.

### A3. `[OPEN]` Motif blocks (`03`, `04`): asserted, not verified

`TRANSPOSE_SOURCE_TARGET_BLOCKS = True` is a hard-coded belief about how
`motif_pathway_analysis.ipynb` wrote its exports. The metadata CSV carries `w_11…w_33`
columns (`MATRIX_COLUMNS` is defined in `motif_schur_decomposition_analysis_script.py`
and then never used), so the belief is checkable:

```python
w = meta_no[[f"w_{i}{j}" for i in range(1,4) for j in range(1,4)]].to_numpy().reshape(-1,3,3)
assert np.allclose(w, blocks_raw)          # confirms npz matches the flat columns
# then transpose once, and record it
```

Worth knowing how much rides on this: I checked which `03` metrics actually change under
transposition.

| Metric | Transpose-sensitive? |
|---|---|
| `spectral_radius`, all eigenvalues | **no** (invariant) |
| `schur_offdiag_norm`, `schur_departure_ratio` | **no** (‖A‖_F and the spectrum are both invariant) |
| `resolvent_gain_z1`, `resolvent_condition_z1` | **no** (2-norm and condition number are invariant) |
| `node_*_schur_participation`, `dominant_schur_node` | **yes** |
| `signed_loading_*` in `04` | **yes** — right vs. left eigenvector |

So the headline rankings in `03` (superpattern spectral radius, departure ratio, enhancement/
ablation sensitivity) are unaffected either way. Everything cell-attributed — which cell type
dominates a motif mode, and the entire `04` operator, which is built from signed loadings — is
fully dependent on getting the transpose right. That is the part to verify rather than assume.

---

## Part B — Comparison with Borst & Leibold

### B1. Faithfully reproduced

- **Shared row/column permutation** (`M' = PᵀMP`). The paper is explicit that only same-permutation
  algorithms are meaningful for connectomics; `apply_order` enforces this everywhere.
- **Method panel** (Fig. 6A): RCMK, PageRank, Tarjan, modified Tarjan, scrambled baseline — all present.
  PageRank is sorted ascending so the most important node gets the last index, matching the paper's
  description.
- **Modified Tarjan** — the paper describes only "a custom-modified version minimizing the upper
  triangular elements in loops." `_improve_scc_order`'s adjacent-swap descent on
  `(N_upper, Σ|M'_ij|(j−i))` inside each SCC is a reasonable reading of that.
- **Fig. 7 benchmark design** — binary matrix, 0.50 lower / 0.25 upper occupancy, no self-connections,
  random shared permutation, 15 repeats, median with 90% interval. Matches the caption exactly.
- **Order index** — "fraction of filled entries in the first subdiagonal." Matches.
- **Objectives kept separate** — the paper's point that "optimizing for one does not necessarily
  optimize the other" is respected; the notebook never collapses the ranks. Good.
- **Thresholding** (Fig. 6B/C/D) — the paper's "even a moderately low threshold gives rise to a mostly
  feedforward structure" is directly tested by the sweep, and the SCC-fragmentation framing is right.

### B2. `[PARTLY FIXED]` Definitional mismatch: "excess ratio"

The paper: *"the first measure ('recurrency') is defined by the **increase** of entries in the upper
triangle of the matrix ordered by the respective algorithm **relative to** the original matrix."*
Fig. 7's y-axis spans roughly −0.25 to 1.0 and BFA sits near 0 — so the paper's quantity is

$$r^{\text{paper}} = \frac{N_{\text{upper}}^{\text{alg}} - N_{\text{upper}}^{\text{ref}}}{N_{\text{upper}}^{\text{ref}}}$$

The notebook computes `upper_count / baseline_upper`, i.e. **`r_paper + 1`**. The field name
`recurrency_excess_ratio` and the markdown ("1 means the method recovers the reference number")
are internally consistent, but the numbers are not on the paper's scale. `01` cell 18's
"PageRank had the lowest median recurrency excess (1.146)" is 0.146 in the paper's units, and the
Fig. 7 annotations (`upper: 0.48 / 0.39 / 0.1`) are not comparable to it as printed.

Either subtract 1, or rename the column to `upper_triangle_ratio` and say once in the markdown
that the paper reports this minus one. The second is less disruptive to the existing prose.

**Update 2026-09-05 — partly fixed.** The benchmark markdown now states the offset explicitly,
and the new Fig. 7 recreation reports the paper's excess (`upper_triangle_excess_vs_reference`)
alongside the raw ratio, so the two scales are shown side by side. The
`recurrency_excess_ratio` column itself is still the raw ratio under a name that says excess —
renaming it remains open, and touches `compare_methods`, `ordering_metrics`, `_binary_metrics`
and every table that consumes them.

### B3. `[OPEN]` Schur triangular direction — labeled backwards in `03`

The paper (Eq. 10) wants `R = U*AU` **lower**-triangular, and notes that numerical packages return
upper-triangular. `01` acknowledges this; `03` inherits the upper-triangular form and then reads the
coupling in the wrong direction.

In `motif_schur_decomposition_analysis_script.py`, `schur_upper_coupling_out_abs` and
`upper_triangular_coupling_out_abs` are `|T[m, m+1:]|.sum()`, described in `03` cell 13 as
*"the amount of upper-triangular coupling from that mode into later Schur modes."*

In Schur coordinates `p = Z*V`, `ṗ = T p`, so `ṗ_m = Σ_n T[m,n] p_n`. A nonzero `T[m,n]` with
`n > m` means **mode n drives mode m**. The cascade runs high index → low index. Demonstrated on a
pure feedforward block `1 → 2 → 3`:

```
block M[post,pre] (strictly lower = feedforward):     scipy Schur T:
 [[0. 0. 0.]                                          [[0. 3. 0.]
  [2. 0. 0.]                                           [0. 0. 2.]
  [0. 3. 0.]]                                          [0. 0. 0.]]
```

Mode 3 feeds mode 2 feeds mode 1 — the reverse of the label. Two options:

- rename to `schur_upper_coupling_in_abs` and fix the prose, or
- convert to the paper's convention, which is a basis reversal (not a transpose — transposing
  changes the operator):

```python
J = np.eye(n)[::-1]
R_lower = J @ T @ J.T
Z_lower = Z @ J.T          # A == Z_lower @ R_lower @ Z_lower.conj().T  (verified)
```

I'd do the second in `03` and add a one-line note in `01`'s Schur markdown, since the whole point of
the paper's convention is that the lower triangle reads as forward flow, and `01` currently plots an
upper-triangular `R` under a "feedforward is lower" convention lock without reconciling them.

### B4. Present in the paper, absent from the notebooks `[MIXED]`

**Nilpotency test.** The paper's stated criterion for strict feedforwardness is: *"Whenever there
exists a finite integer power a such that Mᵃ = 0, the connectome is strictly feedforward."*
`is_dag` is equivalent on the binarized graph, but the matrix version is one line, is what the paper
names, and reports the path length at which activity dies:

```python
def nilpotency_index(M, tol=0):
    B = (np.abs(M) > tol).astype(float); P = B.copy()
    for a in range(1, len(M)+1):
        if not P.any(): return a
        P = P @ B
    return None            # not nilpotent -> recurrent
```

Worth adding to the threshold sweep: the threshold at which the graph first becomes nilpotent is a
cleaner statement of "the weight scale at which the circuit is feedforward" than `is_dag`.

**Eq. 9 input-gain decomposition.** This is the largest scientific gap, and it is the one that bears
directly on non-normality. The paper's steady-state expansion is

$$V = -\sum_n b_n \frac{b_n^* T^{-1} I}{\lambda_n}$$

where **`b*_n` are the rows of `S⁻¹` — the left eigenvectors**. `01` and `02` compute participation
from right eigenvectors only (`|v_ik|²`, normalized). For a normal matrix that is the whole story;
for a connectome it is not. The overlap `b*_n T⁻¹ I` is what decides whether a slow mode is actually
excited by a given input, and the gain factor `1/(−λ_n)` is what makes a long-τ mode an integrator.
A mode can have large right-eigenvector participation in CA1 and still contribute nothing, if the
left eigenvector is orthogonal to the input.

Concretely, add to `schur_and_modes`:

```python
evals, R_vec = np.linalg.eig(A)
L_vec = np.linalg.inv(R_vec).conj().T      # columns = left eigenvectors, b*_n = rows of S^-1
cond  = 1.0 / np.abs(np.sum(L_vec.conj() * R_vec, axis=0))   # eigenvalue condition numbers
gain  = 1.0 / np.abs(evals.real)                              # Eq. 9 gain, for stable modes
```

The per-mode condition number is the standard scalar measure of non-normality and would slot
straight into the existing `modes` table next to `laplacian_energy`. Given the dissertation topic,
I'd treat this as the highest-value addition in the whole review.

**Heterogeneous time constants.** Eq. 3 is `TV̇ = −V + MV + I`, so `A = T⁻¹(M − 1)`. Both notebooks
assume `T = I` and say so. If you ever add per-cell-type τ, note the orientation consequence:
`T⁻¹` multiplies on the **left**, i.e. `A[i,j] = (M[i,j] − δ_ij)/τ_i` where `i` is the
**postsynaptic** index. Row scaling, not column. Worth stating in the convention lock now, because
it is exactly the kind of thing that gets applied to the wrong axis later.

**Brute-force approach (BFA).** Correctly declared out of scope in `01` cell 16 — the authors did not
publish the code. Nothing to do.

**N² ln N scaling reference.** `[FIXED 2026-09-05]` Fig. 7's runtime panel carries a dashed
theoretical scaling line. `plot_outputs` now draws it, anchored at the smallest benchmark size,
and all three benchmark panels use a log x-axis to match the published figure.

### B5. Legitimate extensions beyond the paper

Laplacian/Fiedler ordering, the anatomical ordering baseline, the degree-preserving null tests, and
the Tarjan–Laplacian–Schur bridge are all yours, not the paper's, and are labeled as such. The null
test in particular is a real improvement — the paper reports descriptive comparisons and never asks
whether the observed ordering beats a degree-matched random graph. The framing in `01` cell 22
("supports non-random edge placement and serial topology, but not preferential alignment of the
strongest Mij magnitudes") is exactly the right level of claim.

---

## Part C — Other defects found

**C1. `04` signed Laplacian is dominated by a self-loop.** `build_signed_jaccard_operator` leaves
`A[k,k] = ‖f_k‖² · Jaccard(k,k) = |λ_k|` on the diagonal, and `build_laplacian` then normalizes by
`|A|.sum(axis=1)` including that term. Measured on a synthetic 8-motif operator, the self-loop
supplies **40–73% of each row's absolute degree**. The signed normalized Laplacian
(Kunegis et al.) is defined on a zero-diagonal adjacency; as written, the leading modes are largely
reporting per-motif eigenvalue magnitude rather than motif-neighborhood structure. Fix:

```python
operator = (operator + operator.T) * 0.5
operator.setdiag(0); operator.eliminate_zeros()      # before build_laplacian
```

Keep the diagonal for the `decompose_signed_operator` eigsh if you want it there; strip it for the
Laplacian.

**C2. `04` duplicate-cell motifs are silently corrupted.** `sparse.csr_matrix((values,(rows,cols)))`
sums duplicate coordinates. If any motif contains the same cell type twice, its membership entry
becomes `2` instead of two `1`s, and `jaccard.data / (6.0 - jaccard.data)` then produces
self-similarity of 5.0 instead of 1.0 (reproduced). Probably the enumeration never emits such a
motif, but it costs one line to be sure:

```python
assert (mode_table[["node_1","node_2","node_3"]].nunique(axis=1) == 3).all()
```

**C3. `[RESOLVED 2026-09-05]` `build_matrix_reordering_notebook.py` overwrites `01`.** This was
not hypothetical — the two had already diverged. The builder carried sentence-case headers and
dataset-agnostic wording for the Schur, trisynaptic, and summary "Observed result" cells, while
the shipped `.ipynb` carried title-case headers and hard-coded numbers from an *older* version
of the matrix. Resolved by decision: the builder is the single source of truth, and `01` was
regenerated from it. The A2 orientation guard fix has also been mirrored in the builder.

Related and still worth a pass: several remaining "Observed result" cells quote numbers that no
longer match the current CSV. See Part D4.

**C4. Dead parameter.** `laplacian_bridge_analysis(M, labels, threshold_details, U, paths)` never uses
`U`; it recomputes its own Schur basis internally. Harmless, but it reads as though the bridge is
tied to the `01` Schur factorization when it is not.

**C5. Minor.** `01` displays `result["modes"]` twice (cells 32 and 40).
`run_analysis` returns `"M_paper_rho1": M`, which is misnamed when `NORMALIZATION="none"`.
`04` cell 8's markdown hard-codes "Its shape is `39141 x 39141`" in a notebook that is otherwise
dataset-agnostic. `enumerate_schur_modes` returns a `block_frame` labeled identically on both axes
with no axis names, so a reader cannot tell rows from columns — set
`index.name="post"`, `columns.name="pre"`.

---

## Part D — Second pass, after the real data landed (2026-09-05)

### D1. `[CONFIRMED]` The CSV orientation is settled empirically for this dataset

With both files present, the residual test in `load_inputs` is decisive:

```
netlist rows 1641, unknown rows 0
residual, CSV as stored   vs netlist M[post,pre] : 1.414
residual, CSV transposed  vs netlist M[post,pre] : 3.0e-18
```

So `mij_matrix.csv` really is presynaptic-rows / postsynaptic-columns, and the hard-coded
`MATRIX_ORIENTATION = "pre_rows_post_columns"` in `01` and `02` is **correct for this file**.
The transpose the loader performs is the right one, and no result in the repo is running on a
flipped matrix.

This retired the *risk* for the current dataset. A2 has since been fixed so a supplied netlist
also guards future regenerated, re-exported, or replaced matrices.

### D2. `[FIXED]` Figure 7 top row recreated

Previously only Fig. 7's bottom row existed (`paper_synthetic_benchmark.png`). The top row —
the example matrices with their `upper: / order: / time:` annotations — is now recreated.

No new scrambling step was needed. `paper_synthetic_benchmark` already built the reference
matrix, scrambled it under a random shared permutation, and applied every method; it just
discarded the matrices. It now takes `capture=(60, 0)` and retains that one realization, so the
top row shows exactly the draw the summary curves measure rather than a separately seeded one.

Added to `matrix_reordering_paper_analysis_script.py`:

| Object | Purpose |
|---|---|
| `dfs_finish_order(M, labels)` | reverse DFS finishing order — see D3 |
| `paper_synthetic_benchmark(..., capture=(60,0))` | returns a third value, the captured panels |
| `plot_fig7_top_row` / `FIG7_PANELS` | the six-panel figure, `paper_fig7_top_row.png` |
| `fig7_panel_metrics` | per-panel table, reporting the paper's excess and the raw ratio |
| `PAPER_FIG7_REFERENCE`, `compare_to_published_fig7` | this run lined up against the published values |

`plot_outputs` also now draws the dashed N² ln N reference and uses a log x-axis on all three
bottom-row panels. `01` gained a "Recreating the Figure 7 top row" section and later a final
method-order comparison cell, bringing it to 55 cells.

### D3. `[NEW — changes results]` "Tarjan" in the paper is a DFS order, not a condensation

This is the substantive finding from the Fig. 7 work, and it changes `01`'s method table.

The benchmark's density (0.50 lower / 0.25 upper) makes every realization a single SCC — one
60-node component at N=60. `tarjan_order(..., modified=False)` condenses and topologically
sorts, so with one component the sort contributes nothing and the code falls back to sorting
members by label. The order is arbitrary and `Tarjan SCC` lands on the scramble:

| N=60, median over 15 repeats | recurrency ratio | order index |
|---|---|---|
| Scrambled (no reordering) | 1.490 | 0.390 |
| `Tarjan SCC` | 1.503 | 0.373 |
| **`Tarjan DFS order`** | 1.501 | **0.983** |
| `Modified Tarjan` | 1.326 | 0.746 |
| PageRank | 1.146 | 0.424 |

The paper reports `order: 1.0` for its Tarjan panel. The gap is two different algorithms sharing
a name: Tarjan (1972) is a depth-first search, and a DFS assigns a position to every node
including nodes inside a cycle, whereas condensation deliberately refuses to order within an
SCC. Reverse DFS finishing order recovers the published behavior (0.983 against their 1.0).

**On the real connectome this matters more, not less.** The 85-cell graph is also a single SCC,
and the labels are already alphabetical — so plain Tarjan returns the input order *untouched*,
which is why `Tarjan SCC` and `Original` report byte-identical metrics:

| method | recurrency ratio | order index |
|---|---|---|
| PageRank | 0.697 | 0.262 |
| Modified Tarjan | 0.790 | 0.583 |
| Laplacian Fiedler | 0.807 | 0.464 |
| Anatomical | 0.830 | 0.714 |
| Reverse Cuthill-McKee | 0.887 | 0.643 |
| **`Tarjan DFS order`** | 0.920 | **0.940** |
| Original | 1.000 | 0.702 |
| `Tarjan SCC` | 1.000 | 0.702 |

`Tarjan DFS order` finds a near-complete serial chain through the connectome that no other
method comes close to (0.940 against 0.583 for the next inferred method). Two cautions before
that becomes a claim: a DFS order depends on root and successor iteration order, so it is one
valid traversal rather than a canonical one; and it buys the chain at the cost of recurrency,
leaving *more* upper-triangle entries than PageRank does. Recurrency and order index are close
to opposite objectives here, which is the paper's own point, made sharper.

Worth doing before relying on it: check the order-index stability of `dfs_finish_order` under
label shuffling, and run it through the existing degree-preserving null (`ordering_null_test`)
the way modified Tarjan already is. Neither is done.

### D4. `[OPEN]` The "Observed result" prose is stale

Several narrative cells in `01` quote numbers that predate the current CSV. Examples from the
old cell 14: PageRank recurrency 0.780 and Fiedler 0.903, against 0.697 and 0.807 on the current
data; original-order index 0.738 against 0.702. The method-ranking and benchmark cells were
rewritten to describe the pattern and point at the computed tables instead of quoting digits.
The null-test, threshold, bridge, eigenmode and trisynaptic "Observed result" cells were not,
and should be given the same treatment or re-checked against a fresh run.

### D5. `[NEW — changes conclusions]` Every implemented ordering is far from optimal, and this reverses both null-test results

Prompted by "what is the optimal order for the rows." Minimizing backward edges under a shared
permutation is the linear ordering / minimum feedback arc set problem — NP-hard, which is why the
paper needed a brute-force stage it never published. A cheap stand-in (best-insertion local search
with iterated perturbation, multi-start from the existing orders, seconds of compute on 85 nodes)
beats every method in the repo by a wide margin:

| objective | best implemented method | local-search optimum |
|---|---|---|
| backward edges (of 1570 total) | 653 — PageRank, ratio 0.697 | **445, ratio 0.475** |
| backward \|weight\| | weighted feedforward 0.533 — modified Tarjan | **weighted feedforward 0.9952** |

**Consequence for the null tests in `01`.** The current null section runs modified Tarjan on the
observed graph and on each rewire, so it compares *the heuristic's output*, not the graph's
potential. Re-running the comparison with the optimizer above (12 degree-preserving nulls,
indicative only) reverses both headline conclusions:

| statistic | notebook's claim (modified Tarjan) | with a real optimizer |
|---|---|---|
| weighted feedforward | 0.533 vs null 0.553, z = −0.20, "not unusual" | 0.9952 vs null 0.9679 ± 0.0087, **z = +3.15** |
| backward edge ratio | 0.884 vs null 0.961, z = −4.02, "fewer backward edges" | 0.475 vs null 0.456 ± 0.010, **z = +1.81 the wrong way** |

So the notebook currently reports that edge *placement* is non-random while weight alignment is
not; with an optimizer that is inverted. Nearly all of this connectome's synaptic weight (99.5%)
can be placed below the diagonal, which degree-matched rewires do not match — while the observed
graph is, if anything, slightly *harder* to make feedforward by edge count than its own nulls.

Caveats before this goes anywhere: 12 nulls, not 100; the search is heuristic, so both observed and
null values are upper bounds on the true minimum; and the optimizer could in principle behave
differently on structured versus random graphs, which would bias the comparison. Redo with the full
null count before quoting it.

**Also: the order index is saturated on this graph and should not be used as a discriminator.**
Directed density is 1570/7140 = 0.220, and every node has both in- and out-edges. A plain greedy
walk finds a simple directed path through 81 of 85 cells, so an order index of at least 0.952 is
available with no optimization at all. `Tarjan DFS order`'s 0.940 (D3) is therefore not evidence of
a discovered serial backbone — it is close to what the density alone supplies. This weakens D3
considerably and should be stated wherever that 0.940 appears.

**What the optimized order looks like.** `optimal_orders.csv` holds the min-backward-edges and
min-backward-weight permutations alongside the three heuristics. The optimized order opens with
twelve consecutive EC populations and closes with deep EC layers (LV/VI) — the entorhinal output
stage — with mean positions EC 24.0, CA3 42.9, DG 47.1, CA2 49.8, CA1 52.8. That is roughly the
canonical direction, recovered without anatomical input. By contrast modified Tarjan runs
anatomically *backwards* (EC 66.3, CA1 18.0), which is worth knowing given that it is the method
`01` currently features.

---

## Recommended order of work

Revised 2026-09-05. Done: Fig. 7 top row (D2), N² ln N reference, builder/notebook divergence
(C3), excess-ratio documentation (B2, partly).

1. **D5** — replace modified Tarjan with a real optimizer in `ordering_null_test`, and re-run with
   the full 100 nulls. As it stands `01` publishes two null-test conclusions that a better
   optimizer reverses. This is now the highest-value scientific fix in the repo.
2. **D3 follow-up** — `Tarjan DFS order`'s 0.940 order index is largely explained by density
   (D5: ≥0.952 comes free). Either drop the order index as a headline or report it against a
   density-matched baseline. Also test DFS stability under label shuffling.
3. **B4, Eq. 9 / left eigenvectors** — the substantive scientific gap, and the one closest to the
   dissertation question.
4. **B3** — Schur coupling direction in `03`; fix the label or the basis.
5. **C1** — zero the `04` operator diagonal before the Laplacian.
6. **A3** — verify the motif block orientation against the `w_ij` columns instead of assuming it.
7. **D4** — re-check or genericize the remaining "Observed result" prose.
8. **B2 remainder** — rename `recurrency_excess_ratio`, or subtract one.
9. **B4, nilpotency** — add `M**a == 0` to the threshold sweep.
10. **C2, C4, C5** — cleanup.

Items 1–5 change results. Items 6–10 change confidence and readability.

---

## Sources

- Borst A, Leibold C (2023). Connecting Connectomes to Physiology. *J Neurosci* 43(20):3599–3610.
  https://doi.org/10.1523/JNEUROSCI.2208-22.2023 (Eqs. 3–10; Figs. 5–7)
- Tarjan RE (1972). Depth-first search and linear graph algorithms. *SIAM J Comput* 1:146–160.
- Cuthill E, McKee J (1969). Reducing the bandwidth of sparse symmetric matrices. *ACM '69*:157–172.
- Page L, Brin S, Motwani R, Winograd T (1999). The PageRank citation ranking. Stanford InfoLab.
- Goldman MS (2009). Memory without feedback in a neural network. *Neuron* 61:621–634.
  (the Schur/feedforward-in-transformed-coordinates result the paper builds Eq. 10 on)
