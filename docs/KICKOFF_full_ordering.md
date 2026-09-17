# Kickoff brief — optimal ordering over all 85 neuron types

Paste this into a new chat. Attach or point at `AscoliLab/PROJECT_CONTEXT.md` first.

## Status: the core computation is already done

Do not redo it. `AscoliLab/tarjan_reorder/optimal_ordering_lop.py` solves the Linear Ordering
Problem exactly with HiGHS via `scipy.optimize.milp`: 3,570 binary variables, 98,770 triangle
constraints built once (they do not depend on the weights) and reused. It returns **certified
optima with zero gap in about two seconds**.

| Objective | certified optimum | prior heuristic |
|---|---|---|
| weighted, 85 nodes, `\|m_ij\|` | **0.995185** | 0.9952 |
| binary, 85 nodes | **443 backward of 1,570** (ratio 0.4728) | 445, ratio 0.475 |
| weighted, 32-node E→E | **0.949604** | 0.9496 |

The optimal 85-node ordering is in `tarjan_reorder/optimal_order_85.csv`.

## What actually remains

**1. Fix the anatomy problem.** The raw-weight optimum places DG Granule at position 31 and
CA3 Pyramidal at 5, so the mossy fibre runs backward. Twelve edges carry 48.9% of all `|m_ij|`
and dictate the optimum by themselves; the mossy fibre is 0.002% of total weight. Under
`log1p|m_ij|` the principal cells reorder correctly — DG Granule to position 8, CA3 Pyramidal to 9,
mossy fibre forward. Binary does **not** fix it (DG Granule 46, CA3 Pyramidal 26). Add `log1p` as
a standard variant and report all three with a statement of which question each answers.

**2. Finish the null test properly.** This is the highest-value open item in the project. With an
exact solver on both sides and a degree- *and dyad*-preserving null (census verified at 422 mutual
/ 726 asymmetric), 6 draws gave observed 0.995185 against null 0.99282 ± 0.00173, z ≈ +1.4,
p ≈ 0.29. Run 100. Note that random rewires are much harder LOP instances than the structured
connectome, so use a per-solve time limit and compare the certified observed optimum against the
null **dual bounds**, never against time-limited incumbents — that would overstate significance.
The script already reports both.

**3. Rebuild `EE_backbone/ei_backbone_analysis.ipynb` on the full ordering.** Its current scheme
puts 32 excitatory types in backbone order and appends 53 inhibitory types after them, which
forces **0 of 171 I→E edges** to be forward. The feedforward / feedback inhibitory split reverses
under a proper 85-node ordering (285 / 578 becomes 497 / 366), and the "feedforward inhibition is
more local" claim largely disappears (displacement 10.5 / 21.6 becomes 19.4 / 24.2). The per-edge
strength claim survives and strengthens.

**4. Update the benchmark tables** in `tarjan_reorder/01` and `ee_backbone_method_comparison.ipynb`
to cite the certified optimum as a reference row, and replace the heuristic inside
`ordering_null_test`.

## What is provably unaffected — do not re-run these

A permutation matrix is orthogonal, so `PᵀJP` is a similarity transform and the Schur form `T` is
unchanged (verified: `(P'Q) T (P'Q)*` reproduces `P'JP` to 5.8e-14). Eigenvalues match to 3.0e-13
under optimal matching — a naive `np.sort_complex` comparison reports 2.8e-2, which is a pairing
artifact on near-degenerate conjugate pairs, not an instability.

Unaffected: all of `schur_decomp/01–10`, all of `jacobian_analysis`, `inhib_modulation/01–03`
(blocks are defined by E/I membership, not index order), and `motif_analysis` /
`graph_subnetworks` (triad and orbit counts are isomorphism-invariant). Only matrix heatmaps
change appearance; no number moves.

## One thing to be careful about

Eigenvalue ordering along the Schur diagonal is a *different* ordering and it is not invariant.
`scipy.linalg.schur` picks it by LAPACK's deflation order, with no dynamical criterion. Total
coupling mass `‖striu T‖_F` is invariant across orders, but the per-mode cascade profile
`|T[m, m+1:]|.sum()` is not — it moves from max 242.27 at mode 3 to 296.64 at mode 1 depending on
the sort. Anything per-mode (`schur_upper_coupling_out_abs`, `dominant_schur_node`, greedy-tree
attribution) inherits that arbitrary choice. If the Schur form is to read as a cascade, set the
order deliberately with `schur(..., sort=...)` and report the criterion.
