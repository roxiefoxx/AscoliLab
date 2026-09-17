# Corrections Pass — Planned Operations

Status: plan of record. Created after the certified-optimal ordering work invalidated
two claims and changed the inputs to several notebooks.

Three classes of work, in dependency order. **B blocks A.** Do not edit the documents
until the numbers they will quote exist.

---

## 0. Root dependency: the solver and the null model

Everything downstream inherits these two. Fix them first, once.

| Item | Old | New | Where it lives |
|---|---|---|---|
| Ordering solver | greedy / Tarjan heuristic | exact LOP via `scipy.optimize.milp` (HiGHS), triangle constraints, `mip_rel_gap=0.0` | `tarjan_reorder/optimal_ordering_lop.py` |
| Null model | plain directed edge swap | in-degree, out-degree **and** dyad-census preserving rewire | `rewire_degree_and_dyad_preserving()`, same file |

Why the null model mattered: a plain edge swap destroys ~60% of the 422 mutual dyads.
Mutual dyads are exactly the structure a feedforward-ordering statistic is measuring,
so the old surrogate was not a null for the quantity under test. Census after the new
rewire: 422 mutual / 726 asymmetric, preserved exactly.

---

## 1. Class B — analyses that must be re-executed

Ordered by dependency. Each row's inputs changed; the outputs currently in the repo
are stale, not merely imprecise.

### B1. `tarjan_reorder/01_*` — `ordering_null_test`
- Replace the heuristic with the exact solver on **both** sides (observed and surrogate).
  A heuristic on the observed side and a heuristic on the null side do not cancel; the
  gap between heuristic and optimum is not the same for the two ensembles.
- 100 surrogates, dyad-preserving rewire.
- **Expected consequence:** the weighted-feedforward z-score falls from +3.15 to ≈ +1.4
  (p ≈ 0.29). The weighted-ordering result is *not* significant against a degree- and
  dyad-preserving null. The binary result should be re-reported alongside it.
- This is the single most consequential re-run. Do it first; several document edits
  are waiting on its number.

### B2. 85-node optimal ordering (all cell types)
- Currently only the 32-node E→E ordering is certified optimal (0.949604) and the
  85-node weighted ordering (0.995185) / binary ordering (443 backward of 1570,
  ratio 0.4728).
- Confirm the 85-node certificate and export `tarjan_reorder/optimal_order_85.csv`
  as the canonical ordering artifact. Every other notebook loads it rather than
  re-deriving an ordering.

### B3. `EE_backbone/ei_backbone_analysis.ipynb`
- **Current bug:** the notebook builds its ordering by concatenating 32 excitatory
  types then 53 inhibitory types. That construction forces 0 of 171 I→E edges to count
  as forward — the feedforward/feedback split it reports is an artifact of the
  concatenation, not a property of the connectome.
- Fix: load `optimal_order_85.csv` (B2) and re-run.
- **Expected consequence:** FF/FB reverses from 285/578 to 497/366.
- Everything downstream of the FF/FB split in that notebook changes with it.

### B4. `EE_backbone/ee_backbone_method_comparison.ipynb`
- Outputs predate two solver fixes. Re-run against the current solver. No conceptual
  change expected; the numbers in the comparison table will move.

### B5. `tarjan_reorder/03_*` — Schur
- Correct the Schur coupling **direction label** (`schur_upper_coupling_out_abs`): the
  quantity as computed is out-coupling; the label reads as in-coupling in one place.
- **Pin the eigenvalue order along T's diagonal.** This is free under the decomposition
  and it changes per-mode cascade statistics: `|T[m, m+1:]|.sum()` maximum moves
  242.27 → 296.64 depending on sort. `dominant_schur_node` and the greedy-tree
  attribution both depend on it. Choose one convention (descending Re(λ)), state it in
  the notebook header, and re-run.
- Note for the writeup: cell-type permutation is an orthogonal similarity, so T itself
  is invariant under reordering (verified to 5.8e-14). Reordering does **not**
  invalidate the Schur, Jacobian, motif, or GSN results. Only the diagonal ordering
  convention above is at stake.

### B6. `tarjan_reorder/04_*` — signed Laplacian
- Zero the diagonal of the signed Laplacian (item C1 from the audit). Re-run.

### B7. Motif block orientation (audit item A3)
- Verify that the motif census is built under `M[post, pre]` consistently with the rest
  of the pipeline. Confirm or fix, then re-run the census.

### B8. Housekeeping
- Run `archive_wij.py` — moves the three standalone `w_ij` files to
  `archive/wij_structural/`. Not yet executed. All reported results are `m_ij`.

---

## 2. Class C — new runs (no existing output to invalidate)

### C1. Stationary-covariance null
`trace(P) = 5217.7` observed vs 52.2 for a normal surrogate — a 100× non-normal excess
in stationary variance under white-noise drive (Lyapunov `JP + PJᵀ + I = 0`). This
currently has one surrogate. Run it against the dyad-preserving ensemble (n = 100) so
it carries the same standard of evidence as B1.

### C2. Ordering-objective sensitivity sweep
Report the certified-optimal ordering under three objectives side by side:
`|m_ij|`, binary adjacency, and `log1p|m_ij|`. **See the open question in §4 before
running the third.** The sweep is the honest way to present the DG result regardless of
which objective is adopted as primary.

### C4. Weighted motif census with a proper null
The binary triad census does not distinguish DG from any other mid-degree node (see F6). The
weighted version does, but the node-level weighted measure currently in hand inherits single
heavy edges and needs a strength- **and** dyad-preserving null before it can be reported.
The chain-signature table (E/I signature of X→Y→Z, weighted) does not have that problem and is
reportable now; it gives E→E→E 3.4% of chain weight against E→I→I 36.3%.
Deliverable: per-node weighted triad participation with z-scores against n = 100 surrogates,
plus the chain-signature table with the same null.

### C3. Weight-concentration table
The DG diagnosis rests on numbers that are not yet in any notebook:
- Top 12 edges carry 48.9% of all `|m_ij|`; all 12 are principal → interneuron.
- DG Granule → DG AIPRIM = 224,249 vs DG Granule → CA3 Pyramidal = 234.9 (~955×).
- Median 71.4% of excitatory out-weight lands on interneurons; 91.6% for DG Granule.
- E→E Frobenius norm 4.91 vs E→I 77.02 (~6% of matrix norm).
Put these in a single reproducible cell. They are load-bearing for the corrected claim.

---

## 3. Class A — document corrections (do last, after B1 and B2 land)

Two claims are wrong and appear in six places each.

**Claim 1 (over-claim).** "The optimized ordering recovers the trisynaptic direction
with no anatomical input." **False.** In the certified-optimal 32-node E→E ordering,
DG occupies positions 27–30 of 32. Region means: CA2 (7) → CA1 (8) → EC (10) → CA3 (14)
→ MEC (16) → LEC (19) → SUB (25) → DG (28).

Replacement text: *the optimal ordering of the E→E subgraph does not recover the
trisynaptic direction; DG sorts to the end. This is diagnostic rather than a failure —
it localizes the trisynaptic signal outside the E→E block.*

**Claim 2 (over-reliance).** The weighted-feedforward null at z = +3.15. With an exact
solver on both sides, z ≈ +1.4, p ≈ 0.29. Replace every instance.

### Locations

| # | File | What changes |
|---|---|---|
| A1 | `EE_backbone/README.md` | both claims |
| A2 | `docs/connectome_methodology_overview.pptx` | Stage 00 reordering slide |
| A3 | `docs/analytical_methods_brief_for_advisor.docx` | §4 Feedforward/Feedback |
| A4 | `docs/analytical_methods_slides_for_advisor.pptx` | slide 7 |
| A5 | `claude/methodology_arc.md` (project) | both claims |
| A6 | `PROJECT_CONTEXT.md` + `claude/project_context_for_llm.md` | both claims; add to the "known wrong" list |

Also add to `docs/FINDINGS.md`: an entry recording the correction itself, with the
date and the numbers that replaced the originals. The corrections are part of the
record, not an erasure of it.

---

## 4. Open question that blocks C2 — raise with Dr. Ascoli

The proposed fix for the DG ordering problem is a `log1p|m_ij|` objective. Under that
transform the mossy fibre relay sorts forward and the trisynaptic direction appears;
under raw `|m_ij|` it does not, because DG Granule → DG AIPRIM outweighs DG Granule →
CA3 Pyramidal by ~955×.

This sits against a recorded project principle: **"No log-normalization of weights —
explicitly rejected by Dr. Ascoli."**

The same record draws a distinction that may resolve it: **"Dijkstra path costs use
`log(w_max/w)` (always ≥ 0), which is not log-normalization of weights."** That is a log
transform inside a *cost function*, applied to an objective, with the weight matrix
itself untouched — which is exactly the status of a `log1p` ordering objective. The
matrix is unchanged; only the quantity being minimized over permutations is
transformed.

**Ask him directly:** is a log-transformed *ordering objective* admissible on the same
grounds the Dijkstra cost is, or does the earlier rejection extend to any log transform
anywhere in the pipeline?

Until answered, report the raw-`|m_ij|` ordering as primary and the log variant as a
sensitivity check, clearly labeled. Do not present the log ordering as the headline.

---

## 5. Suggested execution order

1. B1 (solver + null, 100 surrogates) — produces the number six documents need.
2. B2 (85-node certificate, export CSV).
3. B3 (`ei_backbone_analysis` against the 85-node ordering).
4. B4, B5, B6, B7 — independent of each other, any order.
5. C1, C3, C4 — independent, can run alongside 4.
6. §4 question to Ascoli; C2 after the answer.
7. A1–A6 document edits, once B1 and B2 numbers are final.
8. B8 (`archive_wij.py`) — anytime.

---

## 6. Concrete re-run list (verified against the repo, 2026-09-16)

Repo inventory: 56 notebooks in `AscoliLab` (23 live, rest in `archive/` or
`_pre_sync_backup_20260905/`) and 4 in `graph_subnetworks`. Of the live set, **7 need
re-running** and 16 do not.

### Tier 1 — fix the two code modules first (before any notebook)

The ordering heuristic and the null model are defined in library files, not in the notebooks.
Fixing these once propagates to everything in Tier 2.

| File | Change |
|---|---|
| `tarjan_reorder/matrix_reordering_paper_analysis_script.py` | replace the greedy/Tarjan ordering with the exact LOP solver; replace the edge-swap null with `rewire_degree_and_dyad_preserving` |
| `EE_backbone/ee_backbone_ordering.py` | same solver replacement |
| `tarjan_reorder/optimal_ordering_lop.py` | already correct — import from here |

### Tier 2 — notebooks that must re-run (5)

| # | Notebook | Why |
|---|---|---|
| 1 | `tarjan_reorder/01_matrix_reordering_paper_analysis.ipynb` | imports the reordering script; `ordering_null_test` lives here. Run 100 dyad-preserving surrogates. Produces the z ≈ +1.4 figure that six documents are waiting on. |
| 2 | `tarjan_reorder/02_modified_tarjan_pathway_reconstruction.ipynb` | its script imports `matrix_reordering_paper_analysis_script` |
| 3 | `EE_backbone/ee_backbone_method_comparison.ipynb` | imports `ee_backbone_ordering`; outputs predate both solver fixes |
| 4 | `EE_backbone/ee_backbone_deterministic_stochastic_comparison.ipynb` | also imports `ee_backbone_ordering` and carries its own `ORDERING_NULL` block |
| 5 | `EE_backbone/ei_backbone_analysis.ipynb` | the E-then-I concatenation bug; load `optimal_order_85.csv` instead. FF/FB 285/578 → 497/366 |

### Tier 3 — one pair, after the ordering is settled (2)

`tarjan_reorder/03_motif_schur_decomposition_analysis.ipynb` and
`04_motif_signed_overlap_operator_analysis.ipynb`.

**Run one copy, not two.** `tarjan_reorder/motif_schur_decomposition_analysis_script.py` and
`motif_analysis/motif_schur_decomposition_analysis_script.py` are byte-identical, as are the
two `motif_signed_overlap_analysis_script.py` files. Pick one location and delete or symlink
the other; running both duplicates the output for no gain.

**What in 03 is and is not ordering-dependent.** The Schur there is taken on 3×3 motif blocks,
not on the full matrix. Consequently:

- *Invariant* (no re-run needed for these columns): `schur_offdiag_norm` (= ‖upper T‖_F, the
  Henrici departure, basis-independent), `schur_departure_ratio`, `schur_eigenvalues`,
  `spectral_radius`.
- *Ordering-dependent*: `schur_upper_t12_abs`, `schur_upper_t13_abs`, `schur_upper_t23_abs`,
  and the `node_1/2/3_schur_participation` labels — all follow the node order within the block.
  These are the columns the re-run changes.

**04 needs a decision before it runs.** `build_laplacian` computes
`I − D^{−1/2} A D^{−1/2}` with A retaining its self-loop diagonal, so the Laplacian's diagonal
is `1 − a_ii/d_i` rather than 1 (and `signed_combinatorial` gives `d_i − a_ii`). This is a
genuine departure from the standard normalized Laplacian and it changes the eigenvalue
interpretation. Decide explicitly whether self-loops enter the Laplacian, state it in the
notebook header, then run.

### Do NOT re-run (16 live notebooks)

Cell-type permutation is an orthogonal similarity, so T is unchanged (verified to 5.8e-14).
No Schur, Jacobian, motif-census or GSN result is invalidated by the reordering work.

- `jacobian_analysis/01`–`06` (6)
- `schur_decomp/01`–`10` (10)
- `inhib_modulation/notebooks/01`–`03` (3)
- `motif_analysis/triad_modulation_analysis.ipynb`
- `non-normal_matrices/analyses/mij_paper_replication.ipynb`
- `graph_subnetworks/`: `GSN_connectome_workflow`, `GSN_paper_replication`,
  `motif_characterization`, `triad_ablation_stability` (4)

One text check only: `schur_decomp/01_schur_eigenmodes.ipynb` uses the word "feedforward" once
in prose. Confirm it is not asserting the retracted ordering claim; no re-run either way.
