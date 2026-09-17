# Dissertation gap analysis and strengthening to-do

Roxanne Aniceto · Ascoli Lab · drafted 10 September 2026

Companion to `docs/connectome_methodology_overview.pptx` and `claude/methodology_arc.md`.
This document is about what is *missing*, not what exists.

---

## 1. Where the project actually stands

Six analysis stages, all methodologically sound, none of them making a claim that could be
false. Every stage answers "what does this matrix look like under method X." A dissertation
needs one sentence a committee could attack, plus the evidence that survives the attack.

The second problem is structural: the six stages run in parallel rather than constraining one
another. Nothing in the Jacobian result depends on the backbone result. Nothing in the motif
census would change if the Schur analysis had come out differently. That is a portfolio, not
an argument.

The good news is that the argument is already present in the data and has been measured five
separate times without being named.

---

## 2. The thesis

> **The hippocampal–entorhinal circuit is functionally feedforward inside a structurally
> recurrent graph, and that geometry — not its spectrum — governs its transient dynamics.**

This is falsifiable, it is a statement about the hippocampus rather than about linear algebra,
and it makes the six stages serial instead of parallel.

Theoretical anchor: Goldman (2009), functionally feedforward networks; Murphy & Miller (2009)
and Hennequin et al. (2012, 2014) for non-normal amplification in E/I circuits. The
contribution is not that non-normality exists — it is expected in any Dale-respecting E/I
network — but that a **cell-type-resolved, empirically reconstructed** connectome has this
geometry, that it can be quantified against matched nulls, and that specific named cell types
can be identified as carrying it.

### Evidence already in hand

Four scales, one claim, plus a control:

| Scale | Measurement | Source |
|---|---|---|
| Edge | 95% of E→E weight forward-pointing; 99.5% of all \|weight\| placeable below the diagonal, in a single SCC | Stage 01 / 00 |
| Triad | 030T enriched 1.5×, 120D 2.9×; 030C nearly absent (16 triads), 201 suppressed to 0.11× | Stage 03 |
| Block | E→I 77.02 vs I→E 0.084 in Frobenius norm | Stage 06 |
| Operator | Henrici departure 99.3% of ‖J‖_F; eigenvector condition 9.9e4; abscissa gap 29.63 | Stage 06 |
| **Control** | normal surrogate with identical spectrum amplifies exactly 1.0× | Stage 06 |

Transitive, cycle-averse, weight-forward, one dominant stage with negligible return, and almost
entirely non-normal. These are five ways of saying the same thing.

### The sharpest available version

The D5 finding in `tarjan_reorder/NOTEBOOK_REVIEW_vs_Borst_Leibold_2023.md` is more interesting
than it currently looks. Under a degree-preserving null the observed graph was, if anything,
*harder* to make feedforward by edge count — while 99.5% of its weight could be placed below
the diagonal, far beyond what the nulls achieve. **The feedforward geometry lives in the
weights, not in the topology.** That is a stronger, stranger and more testable claim than the
one currently being made, and it should probably be the thesis sentence rather than a footnote.

It also reframes the archived `w_ij` question: if the geometry is in the weights, then decomposing
which factor of `m_ij = q_i · c_ij · g_ij · (V_ref − V_rev) · τ_ij · κ_j` supplies it is not a
side quest — it is the mechanism.

---

## 3. Gap analysis

Ranked by how badly each one threatens the thesis.

### G1 — There is no null that could kill it  ·  *critical*

Null machinery exists in three places (motif enrichment, ordering permutation, backbone
placement) and has never been pointed at the non-normality measures themselves.

Needed: rewire the whole 85×85 matrix preserving in-degree, out-degree, dyad census and Dale,
renormalise each draw to matched ρ, and build null distributions for Henrici departure, the
abscissa gap, peak amplification, and the optimised forward-weight fraction. If the observed
connectome sits inside those distributions, the thesis is dead and needs restating. If it sits
outside, this is the central figure of the dissertation.

Do this before writing anything else. It is cheap and it determines the shape of every chapter.

### G2 — No input–output statement  ·  *critical*

Non-normal amplification only matters if something can excite the amplifying direction. Right
now every quantity is a property of the operator, not a prediction about the circuit.

Needed: optimal perturbation from the SVD of exp(Jt); then the reachability question — is that
direction accessible from a physiologically plausible input (perforant path onto DG granule and
CA3 pyramidal), and is the amplified response observable at a plausible readout (deep EC,
subiculum)? Formally this is a B matrix and a C matrix rather than the identity.

If the worst-case direction requires all 85 types perturbed in a precise pattern, the 27× is a
mathematical curiosity. If it lies close to the perforant-path input direction, it is a claim
about hippocampal function. This single test converts the project from linear algebra into
neuroscience, and it is the same gap as the Eq. 9 / left-eigenvector item already on the list.

### G3 — Nothing has been simulated  ·  *critical*

Every number is analytic. Non-normal transient amplification is a **linear** phenomenon and
saturation can destroy it. A committee will ask whether the 27× survives a nonlinear model, and
"it follows from the Jacobian" is not an answer.

A four-rung validation ladder, in order:

1. **Linear ODE integration of J.** Confirms the analytics reproduce numerically. Trivial, but
   it is the baseline every later rung is compared against.
2. **Nonlinear rate model**, same W, saturating transfer function (e.g. threshold-linear or
   sigmoidal), driven at the optimal perturbation and at a physiological input. *Does the
   transient survive, and at what input amplitude does it stop surviving?* This is the rung
   that matters. If amplification vanishes under saturation, the claim must be restated as a
   statement about the linearised regime, not about the circuit.
3. **Spiking simulation.** Hippocampome carries Izhikevich parameters and κ_j is already
   derived from an Izhikevich linear transfer slope, so there is a natural path from the same
   source data. Does a population-level transient appear?
4. **Comparison against measurable transfer functions.** `schur_decomp/09` already computes
   frequency response (peak gain 69.39 with self / 193.43 without, both peaking at f = 0.01).
   That exists as a *description*; reframe it as a *prediction* and it becomes a validation
   target — for slice electrophysiology, or for the MPS route, without new machinery.

Rung 2 is the minimum for a defence. Rungs 3 and 4 are what make it a paper.

### G4 — No prediction that diverges from a simpler analysis  ·  *high*

If the cell types nominated by non-normal contribution are the same ones nominated by degree or
eigenvector centrality, a referee will say a hard analysis recovered an easy answer.

Needed: rank all 85 types three ways — non-normal contribution (ablation Δ in peak
amplification), out-degree/strength, and eigenvector centrality — and report the disagreement.
The divergence *is* the contribution. Then cross-check the top divergent types against the
optogenetic and anatomical literature. If CA1 Perforant Path Associated, MEC LIII Superficial
Multipolar or DG AIPRIM are known disinhibitory cells, that is validation; if they are not,
that is a prediction. Either outcome is publishable. Not checking is not.

### G5 — Robustness is unquantified  ·  *high*

Modelling choices currently carried without a sensitivity analysis: τ = 1 and uniform; κ_j from
an Izhikevich linear slope; TPM short-term plasticity constants (U, τ_r, τ_f) excluded entirely;
population-level rather than single-cell; α = 0.9 in the neural-mass target declared as a
modelling choice, not a calibration.

κ sensitivity is now first among these. The archived `w_ij` comparison showed that ~10.6× of the
912× E→I / I→E asymmetry comes from κ alone (interneuron-target κ median 0.0935 vs
principal-target 0.0345 Hz/pA, ~9× at the tails). If the abscissa gap is fragile to κ, that is a
chapter, not a footnote.

`jacobian_analysis/06` is scaffolded for exactly this.

### G6 — Generalisation is framed as repetition  ·  *medium*

"It also works on C. elegans" is not a result. The real risk is that non-normality is generic to
any sparse signed directed matrix — which it partly is.

Bracket the claim with four comparisons: the real connectome, a degree-matched null, a
Dale-respecting random matrix with matched E/I proportions, and **one** other real connectome. A
negative result on a non-laminar circuit (C. elegans) would be more informative than a positive
one, because it would make hidden-feedforwardness a property of memory circuits rather than of
matrices. One additional system is enough for a dissertation.

### G7 — Reproducibility debt  ·  *low severity, high examiner-findability*

Stale numbers, a reversed label, and a null test whose conclusion flips under a better
optimiser. None is fatal; all are findable. Clear these before writing, listed in §4.

---

## 4. To-do, by existing analysis

What strengthens each thing that already exists. Ordered within each folder by value.

### `EE_backbone`

- [ ] **Regenerate the path-weight tables.** They predate the branch-and-bound admissibility fix
      and the `problem.sol_status` optimality certification. The README says so; the numbers
      should not outlive the warning.
- [ ] **Re-score the 0.9496 forward-weight fraction under the local-search optimiser**, not the
      feedback-arc heuristic, so the headline number is a defensible upper bound.
- [ ] **Replace the placement null** with a strength-preserving randomisation (Milisav et al.
      2024). The current null preserves each source's out-degree and weight multiset but not
      target in-degree, and the result it produces is negative — worth knowing whether that
      survives a better null.
- [ ] **Present the 32-node E→E result and the 85-cell result as one measurement at two scopes.**
      They currently read as unrelated analyses.
- [ ] Decide the fate of the Hamiltonian MILP arm. It answers a stricter question and is not
      comparable to the simple-path methods; either drop it or state the incomparability once
      and stop ranking it alongside them.

### `inhib_modulation` + `schur_decomp/07–08`

- [ ] **Give the disinhibition result a proper null.** E→I→I→E at z ≈ 52 is the strongest
      cell-type-level claim in the whole project and it currently rests on a shuffled-weight
      null. Any Dale-respecting network with 53 interneurons has many I→I→E paths; the enrichment
      must survive a degree- and Dale-preserving null to mean anything.
- [ ] **Re-rank disinhibitory sources by non-normal contribution**, not only by the composite
      disinhibition score. That is the ranking that can diverge from degree (see G4).
- [ ] **Document the Neumann chain-depth selection rule** and report the selected order as a
      result with a sensitivity band, rather than as a preprocessing detail.
- [ ] **Literature cross-check the top three sources** (CA1 Perforant Path Associated, MEC LIII
      Superficial Multipolar Interneuron, DG AIPRIM).
- [ ] Fold the "paradox" — huge E→I weight, weak inhibitory feedback eigenvalues — into the main
      thesis explicitly. It is not a curiosity; it is the feedforward geometry seen from the
      spectral side.

### `motif_analysis` + `graph_subnetworks`

- [ ] **Unify the motif vocabulary.** `motif_analysis/triad_utils.py` builds 120 two-edge classes
      from a `[pre, post]` matrix; `graph_subnetworks/gsn_connectome.py` builds the 13 DHL classes
      from a validated lookup table. Retire the former as the primary vocabulary, keep DHL
      (validated against Rees et al. 2016), and express the 120-class scheme as a mapping if it
      is still needed. Until this is done, motif tables from the two folders cannot be joined.
- [ ] **Run a weighted motif census.** Every descriptive count is currently on binary adjacency.
      If the thesis is that the geometry lives in the weights, a binary census cannot support it.
      This moves from "nice to have" to central.
- [ ] Fix the Schur coupling-direction label in notebook 03 (B3) and verify motif-block
      orientation against the `w_ij` metadata columns rather than asserting it (A3).
- [ ] Zero the notebook 04 operator diagonal before the signed Laplacian (C1).

### `graph_subnetworks` (GSN) — reframed

The stated purpose of GSN is to establish that the 3-node motif is *important* and *usable*, not
to win a benchmark. Three changes make it do that:

- [ ] **The constant-`edge_attr` run is now the load-bearing experiment.** With `m_ij` stripped
      from the edge attribute, do orbit counts alone still predict the observable? That is the
      only test that separates "3-node structure carries dynamical information" from "the weights
      carry it and the orbits ride along." Everything else about GSN's place in the dissertation
      depends on this result.
- [ ] **Change the target to a transient observable.** The current target — steady state under DG
      Granule drive — is a *spectral* quantity, and the thesis is about transients. Predicting
      peak amplification, or gain along the amplifying direction, would make GSN speak to the
      thesis instead of running alongside it. This is the single highest-value change to the GSN
      arm.
- [ ] **Add a motif-class ablation.** Retrain with the orbit features of one triad class removed
      at a time, and report which classes carry the predictive signal. This ties GSN directly back
      to the enrichment results (300, 120D, 030T enriched; 201, 111D suppressed) and answers "which
      motifs matter" with a second, independent method.
- [ ] Keep reporting R² *and* Spearman, and keep saying why.

### `schur_decomp/01–06, 09–10`

- [ ] **Demote the with-self / no-self comparison.** It is a robustness check occupying headline
      space. The finding is what the modes *are* and which cells load onto them.
- [ ] **Add left eigenvectors / left Schur vectors** alongside the right ones, so modes get both a
      participation (who is in it) and an observability (who can read it) attribution. This is the
      same machinery G2 needs.
- [ ] Fix the coupling direction (B3) — either rename to `..._in_abs` or do the basis reversal
      `J = I[::-1]`, `R = J T Jᵀ`.
- [ ] Reframe the frequency-response output (09) as a prediction, per G3 rung 4.

### `jacobian_analysis`

- [ ] **Finish 02–06.** This is where the dissertation's core chapters live.
- [ ] **03 is the input/output chapter.** Optimal perturbation from the SVD of exp(Jt), plus
      physiological B and C matrices. Prioritise it over 02.
- [ ] **05**: ISN test comparing J_EE against full J, and the inhibitory gain sweep to the α(J) = 0
      crossing. Both are direct tests of the Stage 02 inhibition story in continuous time.
- [ ] **06**: τ heterogeneity and κ sensitivity — elevated by the κ decomposition.
- [ ] Add the nonlinear rate model here rather than in a new project (G3 rung 2). It shares the
      loader, the orientation guard and the ablation primitives.

### `tarjan_reorder`

- [ ] **Replace modified Tarjan with the feedback-arc optimiser inside `ordering_null_test`,
      re-run at 100 nulls.** Two published conclusions currently reverse under a better optimiser;
      until this is done, that section cannot be quoted.
- [ ] Drop the order index as a headline discriminator, or report it against a density-matched
      baseline (≥0.952 is free at this density).
- [ ] Add the nilpotency index to the threshold sweep — it is the paper's own stated criterion for
      strict feedforwardness and it reports the path length at which activity dies.

### New work with no current home

- [ ] **The whole-matrix null ensemble** (G1). Probably belongs in `jacobian_analysis`, since that
      is where the non-normality measures live.
- [ ] **The divergent-ranking analysis** (G4).
- [ ] **The nonlinear and spiking simulations** (G3 rungs 2–3).
- [ ] **One generalisation system plus random controls** (G6).

---

## 5. Sequence

Do them in this order. Each one determines whether the next is worth doing.

1. **G1, the whole-matrix null.** A week. Determines whether the thesis stands.
2. **GSN constant-`edge_attr` + transient target.** Determines whether motifs are a chapter or an
   appendix, and therefore what the chapter list is.
3. **G2, optimal perturbation and reachability.** Turns the operator result into a circuit claim.
4. **G3 rung 2, the nonlinear rate model.** Determines whether the headline number survives.
5. **G4, the divergent ranking**, with literature cross-check.
6. **G5, κ and τ sensitivity.**
7. **G7 hygiene pass**, then start writing.
8. **G6 generalisation**, as the scope chapter.

Everything else — the 100-null ordering redo, the notebook 03 label, the MPS work, the
structural-vs-functional `w_ij` comparison — is chapter-strengthening or a separate paper.

---

## 6. Anticipated objections

Worth having answers drafted before the defence.

| Objection | Where the answer has to come from |
|---|---|
| "Any E/I network with Dale's law and strong E→I is non-normal. What is new?" | G1 (matched nulls) and the cell-type resolution. Have the null figure ready. |
| "27× of what? Amplification of a direction nothing can excite is not amplification." | G2. |
| "Amplification is a linear artefact; saturation kills it." | G3 rung 2. Do not go in without this. |
| "Your matrix is population-level and census-weighted. These are not neurons." | Stated scope limit plus G5. Own it early rather than defending it late. |
| "κ_j is a linearised Izhikevich slope. How much of your geometry is that choice?" | G5, with the 10.6× decomposition already measured. Answer it before it is asked. |
| "You normalised ρ, so your magnitudes are arbitrary." | True. Commit to ordinal and comparative claims; state once, clearly, that magnitudes are conditional on the normalisation. |
| "Short-term plasticity is excluded and this circuit is famous for it." | Scope limit. Name it in the methods chapter. |
| "The trisynaptic direction fell out of your ordering — is that not circular?" | It is not: no anatomical information enters the optimisation. Make that explicit with the anatomical order as a scored comparison row. |
