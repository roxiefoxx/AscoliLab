# Plan — is the connectome an inhibition-stabilized network, and can it respond paradoxically?

High-level plan, written 2026-11-02. Builds on `jacobian_analysis/` (continuous-time
framing, `tau dx/dt = -x + g W x`, `W = M[post, pre]`), and on the conventions in
`PROJECT_CONTEXT.md` §3 (orientation oracle, normalise `W` never `J`, with_self /
no_self, both `EI` senses).

---

## 0. What exactly is being tested

| Property | Operational definition in this model | Notes |
|---|---|---|
| **ISN** | Full `J` stable **and** `J_EE` (inhibitory rates clamped) unstable: `max Re λ(g W_EE) > 1 > max Re λ(g W)` | A property of the *operating point* (gain `g`), not of the wiring alone. |
| **Self-paradox of type k** | Steady-state response `Γ_kk < 0`, with `Γ = (I − gW)⁻¹` | Extra drive to type k lowers type k's own rate. |
| **Class paradox** | `Σ_{k∈S} (Γ b_S)_k < 0` for a drive `b_S` on a set S of I types (e.g. all basket cells, all MEC interneurons, all 53 I types) | The optogenetic-experiment version. Depends on the basis (see §3). |

In a 2-population E/I model, ISN ⇔ paradoxical response. **With 53 inhibitory types
that equivalence breaks**, and pulling the two apart is the main scientific content
here (Litwin-Kumar et al. 2016; Sadeh & Clopath 2021; Mahrach et al. 2020; Palmigiano
et al. 2023). Short-term plasticity can also separate the two (Wu & Zenke 2021). It is
excluded from `m_ij`, so this is a stated limitation (§4).

### Key analytical handle (verified numerically, 424/424 cases)

By Cramer's rule, `Γ_kk = det(I − gW_{−k}) / det(I − gW)`, where `W_{−k}` is `W` with type k
removed (clamped). When the full network is stable, the denominator is positive. So:

> **Type k responds paradoxically ⇔ the network with type k clamped is unstable**
> (an odd number of real eigenvalues of `gW_{−k}` exceed 1).

This is the multi-type generalisation of "ISN ⇔ paradox". ISN asks whether inhibition
*as a whole* stabilises the E subnetwork. Self-paradox asks whether *one specific type*
is indispensable for stability. The set version is `det Γ_SS = det(I − gW_{−S}) / det(I − gW)`.

---

## 1. Preliminary numbers (quick check — to be reproduced in a notebook)

`W` scaled to unit spectral radius, then multiplied by global gain `g`.

| Variant | `α(W_EE)` | E-subnet unstable for | Full net unstable for | **ISN window** | Canonical ρ = 0.95 is… |
|---|---:|---|---|---|---|
| with_self | 1.042 | g > 0.960 | g > 1.000 | **(0.960, 1.000)** | just outside — *not* ISN |
| no_self | 1.172 | g > 0.854 | g > 1.000 | **(0.854, 1.000)** | inside — ISN |

- **The window exists but is narrow, and the project's normalisation constants (0.85, 0.95)
  sit right on its edge.** At ρ = 0.85 neither variant is an ISN. At ρ = 0.95 only no_self is.
  So "is it an ISN?" cannot be answered by picking a normalisation. It needs a principled
  gain (Phase 1).
- **Self-paradox is rare and specific.** Inside the window, only **MEC LIII Superficial
  Multipolar Interneuron** has `Γ_kk < 0` (with_self at g = 0.98; no_self at g = 0.95). The
  next types to go paradoxical as g → 1 are MEC LIII Superficial Trilayered, then
  (no_self only) CA3 Basket, CA1 Back Projection, CA2 Wide Arbor Basket and DG AIPRIM. The
  MEC LIII Superficial Multipolar type is already one of the three top disinhibitory sources
  in Stage 02, which links this question to existing results.
- **The diagonal decides which E assembly is unstable.** The leading unstable `W_EE` mode
  loads on superficial EC/MEC principal cells in with_self, and on DG Mossy, CA1 Pyramidal
  and deep MEC/LEC in no_self.
- **Class-level "paradox" without ISN:** no_self at g = 0.85 (outside the window) still gives
  a *negative* mean I response to uniform all-I drive (24/53 types negative).
  *Corrected 2026-11-03:* the path decomposition (Phase 3c) shows this response **does**
  need the E loop. The inhibitory-only network responds positively, so it is not I→I
  disinhibition. It is a class-level, E-loop-mediated paradox occurring just outside the
  strict ISN window. That is possible because class sums, unlike `Γ_kk`, are not tied to
  E-subnetwork instability by the determinant identity.
- **Near-critical paradox is trivial.** As g → g_full, `Γ ≈ v uᵀ / (1 − g/g_full)`, so
  every type with `u_k v_k < 0` goes paradoxical in a thin sliver below g_full (52 of 53
  types for with_self). Only onsets clearly below g_full are meaningful. The notebook uses
  a 1% margin (`robust_paradox`).
- The existing ISN outputs in `non-normal_matrices/outputs/*/population_responses.csv` are
  **not usable**. They were computed at ρ = 1, where `I − W` is singular (cond ≈ 2.4e18), so
  their "paradoxical" flags are numerical artefacts.

---

## 2. Phases

### Phase 1 — Pin down the operating point

ISN status flips inside the range of normalisation constants already in use, so this phase decides everything downstream.

1. Make the gain sweep the primary object. For g ∈ [0.5, 1), report `α(gW)`, `α(gW_EE)`,
   `α` of the inhibitory Schur complement `W_eff = W_EE + W_EI (I − W_II)⁻¹ W_IE`, the
   window `(1/α(W_EE), 1/α(W))` and its relative width. Do this for with_self and no_self.
2. Find a defensible absolute gain without relying on spectral-radius normalisation. Options,
   in order of preference:
   - Restore physical units in `m_ij` (Hz/Hz loop gain from `q·c·g·ΔV·τ·κ`) and ask what g
     the raw operator implies. Raw ρ = 26,213 means the unnormalised system is far
     supercritical, so understand why: census-extensive units, missing divisive
     normalisation per postsynaptic population size?
   - Calibrate g to an observable: in-vivo baseline rates, or the E-only gain implied by
     slice experiments with GABA_A blocked (CA3 goes epileptiform → E subnetwork unstable).
   - Report the ISN verdict as a function of g, and state the minimum g at which it becomes
     an ISN.
3. Deliverable: one figure with g on the x-axis, both abscissas, the ISN window shaded, and
   the canonical normalisations marked.

### Phase 2 — ISN test proper (fills `jacobian_analysis/05` §3–4)

1. Eigenvalues of `J_EE` vs `J`, both abscissas, with_self and no_self, over the gain sweep.
2. **Which E assembly is unstable.** Left and right eigenvectors of the leading `W_EE` mode,
   mapped to region and type.
3. **Regional ISN.** Repeat for region-restricted E subnetworks (CA3, DG, CA1, EC/MEC/LEC
   superficial vs deep). The CA3 recurrent collaterals are the classic candidate. Which
   regions are locally ISN at the operating g?
4. **Which inhibition stabilises.** Clamp each I type, and each anatomical I class, and
   record whether the network loses stability. This identifies the *stabilising set*. By §0,
   this set is exactly the set of paradoxical types, so Phase 3a comes almost for free.
5. Inhibitory Schur complement: show `α(W_eff) < 1 < α(W_EE)` explicitly and decompose the
   stabilising term by I type, i.e. the contribution of each k to the shift of the leading
   eigenvalue (first-order perturbation, `uᵀ ΔW v / uᵀ v`).

### Phase 3 — Paradoxical response conditions

a. **Per-type self-response** `Γ_kk` for all 53 I types along the gain sweep. Give the gain
   `g_k*` at which each type turns paradoxical, which ranks interneurons by how indispensable
   they are.

b. **Class-level responses** for anatomically meaningful drive vectors, the analogue of
   optogenetic targeting: basket (PV-like), axo-axonic, O-LM/SST-like, CCK, ivy/NGF, per
   region, and all-I. Use the Hippocampome marker/class annotations.

c. **Paradox vs disinhibition decomposition.** Recompute responses with the E-loop cut
   (`W_IE = 0` or `W_EI = 0`) and with I→I cut. A negative response that survives with the
   E loop cut is disinhibition, not ISN paradox. Report each class's response as direct +
   I→I path + E-loop path.

d. **Analytical conditions.** Write each paradoxical type's condition as a loop-gain
   inequality. In 2-pop terms the paradox needs `w_EE > 1` *and* stability needs
   `w_EI·w_IE > (w_EE − 1)(1 + w_II)`. With I→E about 900× weaker than E→I, check whether
   the *product* (the E→I→E loop gain, which is what matters) is large enough, and which
   specific E→I_k→E loops carry it. This connects to the E→I→I→E motif enrichment (z ≈ 52)
   from Stage 02.

e. **Transient vs steady state.** Simulate a step of drive to each paradoxical type in the
   linear model. Look for the textbook ISN signature: an initial rise, then a fall below
   baseline. Measure how the non-normal E→I amplification (27× peak) shapes the size and
   timing of that transient. This links the ISN question to the thesis rather than leaving
   it as a side result.

### Phase 4 — Robustness: what carries the result?

1. **with_self vs no_self.** Already known to change both the window width and the unstable
   assembly, so report every result for both.
2. **Weights-vs-topology surrogates** (`docs/WEIGHTS_VS_TOPOLOGY.md`): M, M/κ, M/q,
   M/(κq), binary sign. Is the ISN window and the paradoxical set produced by wiring,
   census or gain?
3. **κ and q sensitivity.** Lognormal jitter and operating-point shifts in κ, census
   uncertainty in q. Report the probability that the network is an ISN and that each type is
   paradoxical.
4. **Nulls.** Degree + dyad + Dale-preserving weighted nulls (the same family already used
   in `tarjan_reorder`). Is the window width, or the presence of a single indispensable
   interneuron, unusual for a network with this degree/weight structure?
5. **Time constants.** Steady-state `Γ` and the real-eigenvalue crossing are τ-independent,
   but Hopf (oscillatory) instabilities are not. Sweep `τ_I/τ_E` and use per-type `τ_m` from
   `inhib_modulation/hippocampal_population_model.py`. Slow inhibition can turn an ISN
   unstable, which makes the paradoxical steady state unreachable.
6. **Basis.** `Γ_kk`, and the ISN criterion, are invariant under diagonal similarity (q, √q,
   per-cell), so self-paradox survives the basis issue in `PROJECT_CONTEXT.md` §8. Class
   sums are *not* invariant: state the basis (extensive population vs per-cell rate) for
   every class-level claim.
7. **Short-term plasticity** (stated limitation, optional extension). Linearise STP at
   assumed operating rates, using Hippocampome U, τ_r and τ_f, into effective gains. Check
   whether E→E depression or E→I facilitation removes the ISN regime or decouples it from the
   paradox (Wu & Zenke 2021).

### Phase 5 — Nonlinear check (first simulation in the project)

ISN status depends on the operating state when the transfer function is supralinear
(Kraynyukova & Tchumatchenko 2018; Rubin, Van Hooser & Miller 2015). Implement a
threshold-linear or power-law rate model on the same `W`. κ_j is the linear slope of
the Izhikevich f–I curve, so take the slope at the chosen baseline rates. Then:

- Sweep background drive and find where the network enters and leaves the ISN regime.
- Repeat the "optogenetic" steps from Phase 3b. Check that rectification, where I types
  silence each other, does not abolish the paradox found in the linear model.

### Phase 6 — Predictions and literature comparison

- For each predicted paradoxical type or class and each locally ISN region, state the
  checkable experiment: optogenetic activation of class X in region Y, with a recording of
  class X. The prediction is a decrease at steady state after an initial transient increase.
- Search the literature for optogenetic interneuron-activation studies in CA1, CA3, DG and
  MEC that recorded the stimulated population (Sanzeni et al. 2020 is the cortical
  template). Compare against them.
- Interpretation table:

  | Outcome | Reading |
  |---|---|
  | ISN window contains the principled g; paradoxical set non-empty and robust | Connectome supports an ISN operating regime; names the indispensable interneurons |
  | ISN only for no_self / only near g → 1 | ISN-ness hinges on within-type E coupling and gain, so report it as conditional |
  | Never ISN at plausible g | Stability is not inhibition-dependent. That is consistent with the "functionally feedforward" thesis, where E→I is a feedforward stage rather than a stabilising loop |
  | Class-level negative responses but no `Γ_kk < 0` | Disinhibition, not paradox; the multi-type decoupling result |

---

## 3. Implementation (Python / Jupyter) — drafted 2026-11-03

- `jacobian_analysis/isn_paradox.py` holds all the code, one section per phase. It builds
  on `jacobian_core` (orientation oracle at load) and reuses the degree + dyad rewire from
  `motif_analysis/weighted_motif_null.py`.
- `jacobian_analysis/07_isn_paradox.ipynb`, generated by `build_notebooks.py --only 07`,
  covers all six phases. Cells are short calls into the module; tables and figures go to
  `outputs/07_isn_paradox/`.
- Phase 3.1 asserts `Γ_kk < 0 ⇔ clamped-k instability` (§0) at several gains and raises
  if it fails.
- Still open: a principled g (Phase 1.3), per-type τ_m (`tau_m_85_updated.csv` is not in
  the repo), STP (Phase 4.5), and the literature comparison (Phase 6).

## 4. Known limitations to state up front

Linear, population-level, type-resolved model; no STP; κ fixed at one operating point; all
magnitudes are conditional on g. Verdicts are therefore reported as functions of g, plus
ordinal statements: which types, which regions, which variant. None is a single yes/no.
