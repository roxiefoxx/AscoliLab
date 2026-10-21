# Corrections Pass — Results

Run 2026-09-17 against `docs/CORRECTIONS_PASS.md`. This file records what was executed and
what came out, including the two places where the outcome contradicts the plan of record.
It does not edit any claim; the document corrections in Class A are held pending a decision
on §1 below.

Every number here is reproducible from the artifacts named beside it.

---

## 1. The two contradictions, first

### 1.1 Claim 2 does not need retracting on the 85-type graph — it survives

The plan expected the weighted-feedforward z to fall from +3.15 to about +1.4 (p ≈ 0.29)
once the exact solver was used on both sides and the null preserved the dyad census. It
does not. With an exact LOP solver on the observed graph and on all 100 surrogates, and a
rewire that preserves in-degree, out-degree and the 422/726 dyad census exactly:

| quantity | observed | null mean ± sd | z | p |
|---|---|---|---|---|
| weighted forward fraction | 0.995185 (certified) | 0.991969 ± 0.001021 | **+3.15** | 0.0099 |
| vs. null MILP dual bounds (conservative) | 0.995185 | 0.991978 ± 0.001012 | +3.17 | 0.0099 |
| binary forward edge fraction | 0.661146 | 0.642427 ± 0.009448 | +1.98 | 0.0495 |

96 of the 100 surrogates solved to certified optimality and all 100 preserved the dyad census exactly. The 4 that hit the 60 s limit have a maximum relative gap of 6.7e-4, and the decisive check is gap-proof: the certified observed optimum 0.995185 exceeds *every* surrogate's MILP dual bound, so no amount of extra solver time on the null side can close the difference.
`tarjan_reorder/outputs/corrections_pass/{null_tests.csv, null_ensemble.csv, null_meta.json}`.

The z is not an artifact of an under-mixed rewire. Raising the swap multiplier from 8 to 30
leaves the null mean flat (0.991542 → 0.991692) and moves z from +2.59 to +3.49 on an
independent 30-draw sample with a different seed. Under-mixing biases a surrogate *toward*
the observed graph, which shrinks z; the effect here runs the other way.

### 1.2 The result that does collapse is the 32-node E→E one

The same test on the excitatory subgraph, where the ordering is also certified optimal
(forward weight fraction 0.949604):

| null | observed | null mean ± sd | z | p |
|---|---|---|---|---|
| random permutation (as previously reported) | 0.949604 | 0.4983 ± 0.1219 | — | 0.0005 |
| degree- and dyad-preserving, exact LOP both sides | 0.949604 | 0.944124 ± 0.005649 | **+0.97** | **0.168** |

The permutation null asks only whether the ordering beats a random ordering, which is true
of any graph with any feedforward structure at all. Against a null that holds degree and
dyad structure fixed, the E→E ordering is not distinguishable.
`EE_backbone/ee_backbone_method_comparison.ipynb` cell 20, and the same block in
`ee_backbone_deterministic_stochastic_comparison.ipynb`.

**So the correction is not "the weighted result is not significant." It is that the two
graphs behave differently, and the earlier writeup conflated them.** The 85-type ordering
survives a proper null; the E→E ordering does not.

### 1.3 C1 also inverts

`trace(P) = 5,217.7` against a same-spectrum normal surrogate at 52.2 is a 100× excess and
that comparison is unchanged. Against the degree- and dyad-preserving ensemble it is not
exceptional at all:

| quantity | observed | null mean ± sd | z | p |
|---|---|---|---|---|
| trace(P) | 5,217.7 | 15,592.6 ± 26,596.3 | **−0.39** | 0.554 |

The surrogates are, on average, three times more non-normal than the connectome. The null
is strongly right-skewed (sd is 1.7× the mean), so read the empirical p, not the z. The
honest statement is that the circuit holds two orders of magnitude more stationary variance
than its own eigenvalues predict, and that this is *typical of graphs with its degree and
dyad structure* rather than special to it. F5 in `docs/FINDINGS.md` should say so.
Jacobian convention: self-weights retained, ρ(W) scaled to 0.95, `J = −I + W`,
`JP + PJᵀ + I = 0`.

---

## 2. What was executed

### Tier 1 — library modules

| File | Change |
|---|---|
| `tarjan_reorder/optimal_ordering_lop.py` | already correct; it is the single source of truth |
| `tarjan_reorder/matrix_reordering_paper_analysis_script.py` | `exact_lop_order` and `dyad_preserving_rewire` added; `ordering_null_test` rewritten to use the exact solver on both sides and the dyad-preserving rewire, and to carry each surrogate's dual bound |
| `EE_backbone/ee_backbone_ordering.py` | `exact_lop_order`, `dyad_preserving_null_forward_fraction`; `feedback_arc_ordering(method="exact")` is now the default, with `method="heuristic"` kept for the method-comparison table |

The heuristic is retained only where comparing heuristics is the point. It is not used on
either side of a null test.

### Class B

| Item | Status | Outcome |
|---|---|---|
| B1 ordering null, 100 dyad-preserving surrogates | done | §1.1 |
| B2 85-node certificate and `optimal_order_85.csv` | done | weighted 0.995185 and binary 0.717834 both certified; the exported ordering is byte-identical to the file already in the repo, which is now re-derivable rather than trusted |
| B3 `ei_backbone_analysis.ipynb` | done, re-run | FF/FB reverses 285/578 → **497/366** exactly as predicted; I→E forward goes 0/171 → 28/171. Feedforward inhibitory edges also carry 3.6× the mean magnitude of feedback ones (0.00786 vs 0.00217) |
| B4 `ee_backbone_method_comparison`, `ee_backbone_deterministic_stochastic_comparison` | done, re-run | §1.2; ordering certified, `lop_gap = 0` |
| B5 Schur 03 | done, re-run | direction label corrected (the plan has it backwards — see §3); diagonal order pinned to descending Re(λ). The per-motif cascade maximum differs between orders on **1,679 of 2,000** motif blocks, and the pin holds on 1,997 of 2,000 (3 defective blocks fall back rather than fake an order) |
| B6 signed Laplacian 04 | done, both variants run | retaining self-loops shifts the low signed-Laplacian eigenvalues by up to **11.3%**, always downward. `self_loops="zero"` is the default and the one to report; `laplacian_self_loop_comparison.csv` carries both |
| B7 motif block orientation (audit A3) | done | orientation is **correct as written** — `A[pre, post]` reproduces all 1,641 netlist rows, the transposed reading fails 1,570 of them. Assertion added so it is pinned rather than commented. Census re-run |
| B8 `archive_wij.py` | done | 3 files moved to `archive/wij_structural/`; the two rows in `docs/TRANSPOSE_WARNINGS.md` repointed |

### Class C

| Item | Status | Outcome |
|---|---|---|
| C1 stationary-covariance null | done | §1.3 |
| C2 ordering-objective sweep | **not run** — blocked on the §4 question to Dr. Ascoli |
| C3 weight-concentration table | done | every quoted number reproduces exactly; see §4 |
| C4 weighted motif census with a proper null | done, **with a negative result** | see §5 |

### Class A — document corrections

**Done**, on the decision to split the claim by graph rather than retract it outright.

Claim 1 went in as the plan specifies: in the certified-optimal 32-node E→E ordering DG
occupies positions 27–30 of 32, region means CA2 7.0 → CA1 8.5 → EC 9.9 → CA3 14.0 →
MEC 16.2 → LEC 18.7 → SUB 25.0 → DG 28.5. Verified against
`EE_backbone/ee_backbone_comparison_outputs/backbone_ordering.csv`.

Claim 2 was **not** replaced with the plan's "z ≈ +1.4, p ≈ 0.29", because that is not what
the re-run produces. Every location now says: the weighted ordering survives a degree- and
dyad-preserving null on the 85 cell types (z = +3.15, p = 0.0099) and does not on the E→E
subgraph (z = +0.97, p = 0.168), and the earlier writeups conflated the two graphs.

| # | File | Done |
|---|---|---|
| A1 | `EE_backbone/README.md` | both claims, plus the 497/366 split |
| A2 | `docs/connectome_methodology_overview.pptx` | slides 11, 14, 15, 35, 36 |
| A3 | `docs/analytical_methods_brief_for_advisor.docx` | §4 ¶78, 81, 82, 84, a new ¶86 for the trace(P) caveat, sources |
| A4 | `docs/analytical_methods_slides_for_advisor.pptx` | slide 7 (Technique 4) |
| A5 | `claude/methodology_arc.md` (project) | both claims, plus Stages 03/05/06 and the open items |
| A6 | `PROJECT_CONTEXT.md` + `claude/project_context_for_llm.md` | both claims; four new entries on the "do not assume" list |

`docs/FINDINGS.md` gained section **C1**, the record of the correction itself with the date
and the numbers that replaced the originals, and F5 gained the trace(P) caveat. One text
check was made without a re-run: `schur_decomp/01_schur_eigenmodes.ipynb` uses "feedforward"
once in prose about *dominant eigenvector loadings*, not about an ordering. It is unaffected,
and now carries a note saying so, so it is not read as surviving support for the retracted
claim.

---

## 3. B5 and B6 — the missing producer, and what was done about it

Notebooks 03 and 04 read `outputs/motif_pathway_analysis/schur_ready_3x3_subnetworks_*.{csv,npz}`.
**Those files were not in the repository, and neither was `motif_pathway_analysis.ipynb`,
the notebook that writes them** — `docs/RUN_ORDER.md` already recorded this as "Blocked —
cannot run", with those four notebooks the only mentions anywhere, all reads.

A replacement producer was written: **`tarjan_reorder/motif_pathway_export.py`**. Both
notebooks now run (13.5 s and 9.8 s). The file *format* is recovered exactly — it is pinned
by what the consuming scripts read, and the `w_11..w_33` columns are asserted against the
`.npz` blocks on write, which closes audit item A3's check by construction.

**The selection is not recovered, and that is a new decision rather than a restoration.**
Which triads the original export contained is unknowable from what survives, and it matters:
the connectome has 68,686 connected triads and notebook 04's motif-by-motif overlap operator
would need tens of gigabytes for all of them. The default here is the 8 *closed* DHL classes
(030T 030C 201 120D 120U 120C 210 300 — 8,431 triads, whose class counts reproduce the
independent census exactly), capped at 2,000 ranked by total |m_ij| *within* each class so the
cap never drops a class. Both are `--classes` / `--max-motifs` parameters and the run writes
`export_provenance.json`. Results are comparable across runs only at the same selection, and
the absolute cascade magnitudes below are on this selection, not the plan's.

The code corrections, now exercised:

- **Direction label.** `schur_upper_coupling_out_abs` → `schur_upper_coupling_in_abs`
  (and `upper_triangular_coupling_out_abs` likewise). **The plan has this backwards.** It
  says the quantity computed is out-coupling and the label reads as in-coupling. It is the
  other way round: blocks are analysed as `[post, pre]` (the notebook transposes them at
  load, cell 4), so `dy/dt = T y` gives `(T y)_m = Σ_k T[m, k] y_k` — row `m` is drive
  *into* mode `m`. `|T[m, m+1:]|.sum()` is therefore the coupling a mode **receives** from
  later modes. The old name and the cell-13 prose both asserted the transpose.
- **Pinned diagonal order.** `schur()` in the module is now a wrapper that sorts the
  eigenvalues descending in Re(λ), ties by |λ| then Im(λ), via QR of the sorted
  eigenvectors, with a verification step that falls back to SciPy's unpinned output rather
  than faking a pinned order on a defective block. Stated in the notebook header.
  **Measured:** the pin holds on 1,997 of 2,000 blocks (3 fall back), and the per-motif
  cascade maximum `max_m |T[m, m+1:]|.sum()` differs between the pinned and unpinned orders
  on **1,679 of 2,000** blocks. The ordering convention is not cosmetic.
- **Self-loops in the Laplacian.** `build_laplacian(..., self_loops="zero"|"keep")`, default
  `"zero"`. Notebook 04 runs both and prints the eigenvalues side by side with the relative
  difference, per the decision to compare rather than pick blind. 71 of 1,641 stored entries
  are self-loops, so this is not a rounding detail. **Measured:** retaining them shifts the
  eight low signed-Laplacian eigenvalues by up to **11.3%**, always downward, in both the
  with-self and without-self motif scenarios.
- **Duplicate scripts.** `motif_analysis/motif_schur_decomposition_analysis_script.py` and
  `motif_analysis/motif_signed_overlap_analysis_script.py` were byte-identical to their
  `tarjan_reorder/` counterparts. They are now re-export shims, so there is exactly one
  implementation. The duplicate notebook copies carry a "do not run this copy" header.

---

## 4. C3 — the weight-concentration table reproduces exactly

`tarjan_reorder/weight_concentration.py`, outputs in
`tarjan_reorder/outputs/corrections_pass/`.

```
top 12 edges                      48.9% of all |m_ij|; all 12 principal -> interneuron
DG Granule -> DG AIPRIM           224,249.5
DG Granule -> CA3 Pyramidal           234.9      ratio 955x
excitatory out-weight on interneurons  median 71.4%,  DG Granule 91.6%
E->E Frobenius 4.91  vs  E->I 77.02       E->E = 6.3% of the matrix norm
```

One note on provenance: 4.91 and 77.02 are on the **normalized** operator — self-weights
retained, ρ scaled to 0.95, the `jacobian_analysis`/`schur_decomp`/`inhib_modulation`
convention. On raw off-diagonal weights the same blocks are 134,164.75 and 2,125,319.50.
The ratio is scale-free either way; the absolute numbers move by a factor of ~1.8 depending
on whether the diagonal is kept, so the convention is now stated in the module.

---

## 5. C4 — the node-level weighted triad measure is degenerate

`motif_analysis/weighted_motif_null.py`, outputs in `motif_analysis/outputs/corrections_pass/`.

The measure in use — for each connected triad sum the weight on its edges, credit all three
members — is an **exact affine function of node strength**. With `S = |M| + |M|ᵀ`,
`s_a = Σ_b S[a,b]` and `T = S.sum()/2`:

```
participation(a) = (n − 3) · s_a + T
```

Each of `a`'s incident weights is counted once per choice of third node (`n − 2` times), and
every triad not containing `a` contributes `T − s_a`. Triads with no edges contribute zero
either way, so the "connected" filter does nothing. Verified numerically: max relative error
2.7e-15, correlation with strength 1.000000.

It carries no triad information. A z-score on it is a z-score on node strength, and it cannot
distinguish DG Granule — or anything else — on motif grounds. It should not be reported.

**Replacement.** Weighted participation resolved by triad class (DHL 16), as a share of the
node's total triad weight. Shares are strength-normalized by construction, so the degenerate
total divides out. Each share is z-scored against 100 surrogates from the same ensemble.
DG Granule's profile is strongly non-random: 102 share 0.423 vs null 0.212 (z = +5.8),
120D 0.071 vs 0.020 (z = +6.3), 300 0.036 vs 0.010 (z = +6.4), and 201 depleted at 0.0016 vs
0.083 (z = −6.1). That is a motif statement, not a strength statement.

**Null caveat, measured rather than waved at.** The rewire preserves degrees, the dyad census
and the global weight multiset exactly, but not node strength: median relative out-strength
deviation 0.35, in-strength 0.73. Because the class measure is a share, a shift that rescales
a node's whole profile cancels; a shift that redistributes it does not.

**Chain-signature table**, reportable and now with a null:

| X→Y→Z | observed share | null mean ± sd | z | p |
|---|---|---|---|---|
| E→I→I | 0.363 | 0.234 ± 0.025 | +5.07 | 0.0099 |
| I→E→I | 0.309 | 0.202 ± 0.028 | +3.88 | 0.0099 |
| E→E→I | 0.197 | 0.270 ± 0.027 | −2.71 | 1.0 |
| E→I→E | 0.076 | 0.110 ± 0.012 | −2.75 | 1.0 |
| E→E→E | 0.034 | 0.089 ± 0.026 | −2.14 | 1.0 |

E→E→E is not merely small, it is depleted relative to a degree-matched graph, and E→I→I is
enriched. That is the same fact as §4 stated at chain level.

---

## 6. Verification

`tarjan_reorder/verify_corrections_pass.py` re-derives every headline number from the artifacts
rather than from this document: **35 checks, all passing**, over B1, C1, B2, B3, B7, C3 and C4.
Run it from the repository root; it exits non-zero on any mismatch. It caught one overstatement
in an earlier draft of this file — "all 100 surrogates certified" was really 96 — which is why
§1.1 now carries the dual-bound argument, the check that does not depend on solver time.

Notebook state: `ei_backbone_analysis` 18/18 cells, `ee_backbone_method_comparison` 15/15,
`ee_backbone_deterministic_stochastic_comparison` 14/14, `tarjan_reorder/03` 13/13,
`tarjan_reorder/04` 10/10 — zero error outputs in any of them. All corrections-pass modules
import cleanly and `feedback_arc_ordering` defaults to `method="exact"`.

A repo-wide sweep confirms no retracted claim survives outside a correction context: every
remaining "285 / 578" and every remaining "recovers the trisynaptic direction" sits inside
text that is correcting it, and the only two remaining `coupling_out_abs` strings are in the
note explaining the rename.

---

## 7. Still open

1. **C2** — blocked on the log-objective question to Dr. Ascoli (plan §4, unchanged). Until
   answered, raw `|m_ij|` is primary and the log variant is a clearly labelled sensitivity
   check.
2. **The motif export selection** (§3) is a new decision and should be reviewed. If the
   original criterion turns up, re-run 03 and 04 against it before comparing to anything
   produced before September 2026.
3. **A fully strength-matched null.** The rewire preserves degrees, the dyad census and the
   global weight multiset exactly, but node strength only on the asymmetric part (median
   relative deviation 0.35 out, 0.73 in). Milisav et al. (2024) remains the stronger
   comparison, and it is the one open methodological gap the pass did not close.
4. **`docs/CORRECTIONS_PASS.md` itself is now partly superseded** — its §1 B1 expectation and
   its §B5 direction statement are both wrong. Read it as the plan of record with this file
   as the outcome, not as a current description of the numbers.
