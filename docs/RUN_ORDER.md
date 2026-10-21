# Run Order — all active notebooks

Status: derived from the read/write edges in the notebooks themselves, not from their
numbering. Created 17 September 2026. Companion to `CORRECTIONS_PASS.md`, which decides
*what* to re-run; this file decides *in what order*.

**The headline.** There are 37 active notebooks and only **four** hard dependency edges
among them. Nearly every notebook reloads `mij_matrix.csv` and its own helper module
independently — `inhib_modulation/README.md` states this as a design choice ("each notebook
reloads the source matrix instead of requiring outputs from the previous one"). So
execution order is almost entirely a *correctness* question governed by `CORRECTIONS_PASS.md`,
not a dependency question. Sequencing by dependency alone would re-run ~30 notebooks that
then need re-running again once §B1/§B2/§B5 land.

---

## Scope

| Count | Set |
|---:|---|
| 37 | active notebooks (excludes `archive/`, `_pre_sync_backup_20260905/`, `.ipynb_checkpoints`) |
| 54 | total `.ipynb` including archived |
| 9 | ordered steps below — 3 prerequisites (Tier 0) plus 6 notebooks (Tiers 1–3) |
| 21 | independent — any order, parallelizable, and mostly not needed at all (see Tier 4) |
| 4 | **cannot run** — missing producer (§Blocked) |
| 6 | retired — do not run (§Retired) |

Two further notebooks (GSN triad census, triad ablation stability) live in
`graph_subnetworks`, outside this repo, and are not covered here.

---

## The dependency graph

Four edges, total. Everything else is a root.

| # | Producer | Artifact | Consumer |
|---|---|---|---|
| 1 | `EE_backbone/ee_backbone_method_comparison.ipynb` | `ee_backbone_comparison_outputs/backbone_ordering.csv` | `EE_backbone/ei_backbone_analysis.ipynb` |
| 2 | `schur_decomp/03_schur_signal.ipynb` | `outputs/schur_modes/schur_modes.npz` | `schur_decomp/05_schur_greedy_trees.ipynb` |
| 3 | `schur_decomp/03_schur_signal.ipynb` | `outputs/schur_modes/schur_mode_summary.csv` | `schur_decomp/06_schur_mode_case_studies.ipynb` |
| 4 | `schur_decomp/05_schur_greedy_trees.ipynb` | `outputs/schur_mode_greedy_trees/schur_mode_greedy_tree_summary.csv` | `schur_decomp/06_schur_mode_case_studies.ipynb` |

Longest serial chain: **three deep** (`schur_decomp/03 → 05 → 06`).

Edge 1 is superseded in practice — see step 9.

---

## Tier 0 — prerequisites (scripts, not notebooks)

Do these before any notebook. Each one changes inputs that notebooks read.

**0.1 — `tarjan_reorder/optimal_ordering_lop.py`**
Confirm the 85-node LOP certificate and re-export `optimal_order_85.csv` (§B2). Exact
optimum via HiGHS, zero gap, ~2s. Every notebook that needs an ordering should load this
artifact rather than deriving its own.

**0.2 — `archive_wij.py`**
§B8 housekeeping. Not yet executed. Moves the three standalone `w_ij` files to
`archive/wij_structural/`. Run it before anything resolves a matrix path, so no notebook
can pick up a `w_ij` file by accident — the failure mode that motivated the archive is
silent, since `w_ij` and `m_ij` share shape, labels, sign pattern and block structure.

**0.3 — pin the Schur diagonal ordering convention**
Set `schur(..., sort=...)` to descending `Re(λ)` in `schur_analysis_utils_script.py` and
state the criterion in the notebook headers. `CORRECTIONS_PASS.md` files this as §B5, but
it belongs here: notebooks `03 → 05 → 06` all inherit the convention, so fixing it after 03
runs invalidates 05 and 06. Total coupling mass `‖striu T‖_F` is invariant across orders,
but the per-mode profile `|T[m, m+1:]|.sum()` moves 242.27 → 296.64 depending on the sort,
and `schur_upper_coupling_out_abs`, `dominant_schur_node` and the greedy-tree attribution
all depend on it.

---

## Tier 1 — the ordering root

**1 — `tarjan_reorder/01_matrix_reordering_paper_analysis.ipynb`** (§B1)

The single most consequential re-run in the project. Replace the heuristic with the exact
solver on **both** sides (observed and surrogate — the heuristic-to-optimum gap is not the
same for the two ensembles, so it does not cancel), and use the degree- and dyad-preserving
rewire, 100 surrogates.

Expected: weighted-feedforward z falls from +3.15 to ≈ +1.4 (p ≈ 0.29). Compare the
certified observed optimum against the null **dual bounds**, never against time-limited
incumbents. Several document edits in Class A are waiting on this number, and it decides
whether the weighted feedforward claim survives at all.

---

## Tier 2 — produces artifacts other notebooks read

**2 — `schur_decomp/03_schur_signal.ipynb`**
Writes `outputs/schur_modes/{schur_modes.npz, schur_mode_summary.csv, schur_mode_loadings.csv,
schur_mode_cell_type_propagation.csv, schur_region_summary.csv}`. Blocks steps 7 and 8.
Requires 0.3 to be done first.

**3 — `EE_backbone/ee_backbone_method_comparison.ipynb`** (§B4)
Outputs predate two solver fixes (the inadmissible branch-and-bound bound, and MILP
optimality read from `pulp.LpStatus` instead of `problem.sol_status`). No conceptual change
expected; the comparison table numbers move.

**Commit the ordering artifacts this time.** `backbone_ordering.csv`,
`ordering_metrics.csv`, `ordering_edge_directions.csv` and `ordering_permutation_null.csv`
are quoted in `EE_backbone/README.md` but exist nowhere in the repo — no `.gitignore`,
nothing untracked. The numbers reproduce exactly on re-run (208 forward / 147 feedback,
748,421 / 39,719, fraction 0.949604), so the run happened and the files were not kept.
The stochastic branch folder is the odder case: cell 22 writes `method_comparison_long.csv`
and `backbone_ordering.csv` in the same cell and only the first is on disk.

---

## Tier 3 — consumers of Tier 2

**4 — `schur_decomp/05_schur_greedy_trees.ipynb`** — needs step 2's `.npz`.

**5 — `schur_decomp/06_schur_mode_case_studies.ipynb`** — needs step 2's summary *and*
step 4's summary. Both are read as explicit paths (`SCHUR_SUMMARY_PATH`,
`GREEDY_SUMMARY_PATH`), so it fails loudly rather than silently if either is absent.

**6 — `EE_backbone/ei_backbone_analysis.ipynb`** — **rebuild before running** (§B3).

Running it as committed reproduces the concatenation artifact. The notebook builds its
backbone by putting the 32 excitatory types in ordering position and appending the 53
inhibitory types after them, which forces **0 of 171 I→E edges** to count as forward — a
property of the append, not of the connectome. Load `optimal_order_85.csv` instead.

| Quantity | As committed | Under `optimal_order_85.csv` |
|---|---:|---:|
| Feedforward / feedback inhibitory edges | 285 / 578 | 497 / 366 |
| I→E edges counted forward | 0 of 171 | 28 of 171 |
| Mean abs. displacement, ffwd vs fback | 10.5 / 21.8 | 19.4 / 24.2 |

The "feedforward inhibition is more local" claim largely disappears under the corrected
ordering. The per-edge strength claim survives and strengthens.

Two further defects to fix in the same pass:

- `terminal_i = backbone_idx[-1]`. Under the concatenation that resolves to
  `MEC LIII Superficial Trilayered Interneuron` — the last inhibitory type in CSV row
  order, unrelated to the backbone. The notebook injects at the head of the excitatory
  ordering and measures at an arbitrary interneuron; the backbone's actual endpoint
  (`CA3 Granule`, position 31) is never read. Every terminal-peak number, including the
  null-model p = 0.776, is measured there.
- The CSV-label-order fallback is not a degraded version of the same statistic. Under it
  the injection node becomes `CA1 Axo Axonic` and `terminal_i` becomes
  `SUB CA1 Projecting Pyramidal` — an excitatory pyramidal cell, not an interneuron.
  Placeholder motifs come out 387 / 476. Prefer raising over warning.

---

## Tier 4 — independent, any order, parallelizable

21 notebooks with no inbound or outbound file edges. Each reloads the source matrix and its
own helper module.

- `EE_backbone/ee_backbone_deterministic_stochastic_comparison.ipynb`
- `tarjan_reorder/02_modified_tarjan_pathway_reconstruction.ipynb`
- `schur_decomp/01, 02, 04, 07, 08, 09, 10`
- `jacobian_analysis/01–06`
- `inhib_modulation/notebooks/01, 02, 03`
- `motif_analysis/triad_modulation_analysis.ipynb`, `triad_ablation_stability.ipynb`
- `non-normal_matrices/analyses/mij_paper_replication.ipynb`

**Most of these do not need re-running.** `KICKOFF_full_ordering.md` establishes that a
cell-type permutation is an orthogonal similarity, so `PᵀJP` leaves the Schur form `T`
unchanged (verified to 5.8e-14). All of `schur_decomp/01–10`, all of `jacobian_analysis`,
and `inhib_modulation/01–03` are provably unaffected by reordering; triad and orbit counts
are isomorphism-invariant. Only matrix heatmaps change appearance — no number moves. Re-run
these for fresh figures, not for correctness.

`EE_backbone/ee_backbone_deterministic_stochastic_comparison.ipynb` is independent but not
irrelevant: it writes its own `backbone_ordering.csv` to
`ee_backbone_stochastic_comparison_outputs/`, which is the second half of the determinism
diff described under Known inconsistencies. Run it alongside step 3 if you want that check.

`non-normal_matrices/analyses/mij_paper_replication.ipynb` appears to have 23 upstream
inputs, but it calls `save_outputs()` in-notebook and the `expected_paths` dict is a
fallback merged under the real return value. It is self-contained.

---

## Blocked — cannot run

`motif_analysis/03`, `motif_analysis/04`, `tarjan_reorder/03`, `tarjan_reorder/04`.

All four read from `outputs/motif_pathway_analysis/`:

```
schur_ready_3x3_subnetworks_with_self.csv
schur_ready_3x3_subnetworks_with_self.npz
schur_ready_3x3_subnetworks_without_self.csv
schur_ready_3x3_subnetworks_without_self.npz
```

**Nothing in this repo writes them.** A grep across every file, archives included, returns
those four notebooks as the only mentions anywhere, all as reads. No `.py`, no archived
script, no notebook produces the artifacts, and `outputs/motif_pathway_analysis/` does not
exist in either folder. Same defect class as the missing `backbone_ordering.csv`, but with
no producer to run. The producer most likely lives in `graph_subnetworks` or was never
committed — resolve that before scheduling these.

They are also byte-identical duplicates, pairwise, along with their helper scripts:

| md5 | Files |
|---|---|
| `b664bd25927af7351ee9199bf8dbcd1b` | `motif_analysis/03_*.ipynb`, `tarjan_reorder/03_*.ipynb` |
| `0477c3c9c2c28037fd517a0134878fd5` | `motif_analysis/04_*.ipynb`, `tarjan_reorder/04_*.ipynb` |
| `1afd2a03081ff284c57ace473f507c68` | `motif_schur_decomposition_analysis_script.py` (both folders) |
| `211702d70b179f530984aa01bf56f1ad` | `motif_signed_overlap_analysis_script.py` (both folders) |

Four notebooks, two analyses. Pick one home before re-running either.

---

## Retired — do not run

Superseded per `inhib_modulation/README.md`; consolidated into the numbered 01/02/03:

- `inhib_modulation/notebooks/inhibitory_modulation_analysis.ipynb`
- `inhib_modulation/notebooks/inhibitory_two_edge_triad_analysis.ipynb`
- `inhib_modulation/notebooks/inhibitory_two_edge_triad_analysis_with_self_connections.ipynb`

One-offs kept for record:

- `schur_decomp/individual_analyses/070826_mij_schur_noself.ipynb`
- `schur_decomp/individual_analyses/070826_mij_schur_wself.ipynb`
- `schur_decomp/individual_analyses/BINF751_finalproject_2.ipynb`

---

## Known inconsistencies to fix while you are in here

**Two matrix-path conventions inside one folder.** `schur_decomp/01` and `09` load
`Path("mij_matrix.csv")` from the notebook directory; `02, 03, 04, 07, 08, 10` load
`matrices/mij_matrix.csv`. Both copies exist and are currently byte-identical
(`d0782dc2dd720f7f37031b53666a083e`), so nothing breaks today — but a stale edit to one
copy would be silent. Same md5 holds for the copies in `EE_backbone/matrices/`,
`motif_analysis/`, `tarjan_reorder/`, `inhib_modulation/matrices/` and
`schur_decomp/matrices/`; `mij_netlist.csv` is likewise identical across four folders
(`6f2387dd3ed8287307993197087788fe`).

**The 32-node ordering optimum is degenerate.** `feedback_arc_ordering` returns the same
objective (0.949604, which is also the LOP-certified optimum) at every random seed and at
every restart count ≥ 5, but **23 of 32 nodes occupy different positions** depending on the
seed — the optimum is tied and the RNG breaks it. `DG Granule` spans positions 17–28,
`CA3 Granule` 20–31. Consequences:

- Diffing the two `backbone_ordering.csv` files written by
  `ee_backbone_method_comparison.ipynb` and
  `ee_backbone_deterministic_stochastic_comparison.ipynb` tests *input identity*, not
  search stability. Both call the same pure function with `n_restarts=50, random_seed=2026`
  on the same matrix, so the files are bit-identical or the inputs differ. Worth asserting,
  but restart count can never be the explanation.
- The narrative in `EE_backbone/README.md` — ordering "ends in subiculum and deep
  entorhinal layers" — describes a different draw than the numbers it reports. Six of eight
  seeds tested end that way; seed 2026, the committed one whose displacement figure matches
  the README's, ends `DG Mossy < DG Hilar Ectopic Granule < CA3 Granule`. Loading
  `optimal_order_85.csv` removes the tie-break from the pipeline and makes this moot.

**Provenance gap on `mij_EE_matrix.csv`.** No active script or notebook produces it. It is
a byte-exact submatrix of `mij_matrix.csv` on 32 labels (diagonal retained, 27 non-zero,
355 positive off-diagonal, no negative entries). The README's "three-way agreement" between
the 0.9 Dale threshold, the 0.5 threshold in `is_excitatory_sender`, and this file is partly
vacuous: every sender in `mij_matrix.csv` has an outgoing-sign fraction of exactly 0.0 or
1.0, so 0.9 and 0.5 *cannot* disagree — that is a restatement of Dale's law, not an
independent check. Meanwhile the archived label-substring `EXCITATORY_RULE` in
`archive/build_ee_backbone.py`, the only selection code in the repo, returns a **different**
32: it includes `CA1 Back Projection` and excludes `CA1 Radiatum Giant`. A `00` notebook
deriving the E-to-E matrix from `mij_matrix.csv` should assert non-degeneracy and reconcile
against the label rule, not merely report agreement.

**Naming.** `ei_backbone_analysis.ipynb` writes to `connectome_inhibition_results/` while
its siblings write to `*_comparison_outputs/`. If it is rebuilt on the 85-node ordering it
is no longer downstream of `ee_backbone_method_comparison.ipynb` at all, which changes any
renumbering you had in mind.

---

## Quick reference

```
Tier 0   optimal_ordering_lop.py  ·  archive_wij.py  ·  pin Schur sort
Tier 1   tarjan_reorder/01
Tier 2   schur_decomp/03          ·  EE_backbone/ee_backbone_method_comparison
Tier 3   schur_decomp/05 → 06     ·  EE_backbone/ei_backbone_analysis (rebuild first)
Tier 4   21 independent notebooks — figures only, no number moves
Blocked  motif_analysis/03,04  ·  tarjan_reorder/03,04   (no producer for schur_ready_3x3_*)
Retired  inhib_modulation × 3  ·  schur_decomp/individual_analyses × 3
```
