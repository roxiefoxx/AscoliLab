# Project context — hippocampal–entorhinal connectome dynamics

**Purpose of this file.** A self-contained briefing so that a language model with no prior
exposure to this project can reason about it usefully — brainstorm hypotheses, propose
dissertation titles, critique the argument, or design new analyses — without reading the
repositories. Everything needed is here. Numbers are as of 10 September 2026.

**Who.** Roxanne Aniceto, PhD student in Bioinformatics, George Mason University. Advisor:
Dr. Giorgio Ascoli, Krasnow Institute.

**How to use this file.** Read §1–§4 before proposing anything. §7 lists what is already known
to be wrong or unresolved — do not propose those as discoveries. §8 lists assumptions that look
safe and are not.

---

## 1. One paragraph

The project treats the Hippocampome.org potential connectome as a linear operator and asks what
its *geometry* — as opposed to its spectrum — implies about circuit dynamics. The central tools
are non-normal matrix theory: Schur decomposition, spectral vs numerical vs pseudospectral
abscissas, ε-pseudospectra, the Kreiss constant, and the Henrici departure from normality. The
motivating observation is that this network is asymptotically stable yet amplifies a worst-case
perturbation roughly 27-fold before decaying, and that a normal matrix with an identical
spectrum would not amplify at all. The work spans six analysis stages plus a preprocessing layer
that turns out to be load-bearing.

---

## 2. The data

### Construction

The operator is a signed, directed, weighted 85×85 matrix over hippocampal and entorhinal
**neuron types** (not individual neurons).

```
m_ij = q_i · c_ij · g_ij · (V_ref − V_rev) · τ_ij · κ_j
```

- `q_i` — census count of the presynaptic type
- `c_ij` — axonal–dendritic connection probability
- `g_ij`, `τ_ij` — conductance and decay from the Hippocampome transfer-parameter matrix
- `κ_j` — postsynaptic gain: the linear slope of the Izhikevich transfer function, one value per
  **postsynaptic** type, units Hz/pA
- Short-term plasticity constants (U, τ_r, τ_f) are **not** included

Working form used everywhere: `m_ij = w_ij · κ_j`, with `w_ij` collapsing the earlier factors.

### Dimensions and composition

| Property | Value |
|---|---|
| Neuron types | 85 (32 excitatory, 53 inhibitory) |
| Regions | 8 — DG, CA3, CA2, CA1, SUB, EC, LEC, MEC |
| Netlist rows | 1,641 |
| Directed edges | 1,570 |
| Off-diagonal density | 0.220 |
| Reciprocal (mutual) dyads | 422 |
| Types with non-zero diagonal | 71 of 85 |
| Raw spectral radius ρ(M) | 26,213 |

The wider Hippocampome v1.0 circuit is 122 types / 3,236 edges; the 85-type matrix is the subset
with the parameters needed for `m_ij`.

### Three properties asserted at load

The pipeline fails loudly rather than quietly if any breaks:

1. `m_ij = w_ij · κ_j` exactly, one κ_j per postsynaptic type.
2. The sign of `m_ij` follows the presynaptic type's E/I class for every edge — **Dale's law
   holds with zero exceptions**. This is what makes sign-based E/I inference and the orientation
   oracle (§3) possible.
3. No duplicate (pre, post) pairs.

### The diagonal

**Not autaptic.** It is coupling between distinct cells of the *same* type — a
stochastic-block-model diagonal. This matters twice: triad analyses must unroll it into twin
nodes (§5, Stage 03), and in continuous time it is a genuine synaptic term, never the membrane
leak (§3).

### Structural vs functional (`w_ij` vs `m_ij`) — decided

**All analyses use `m_ij`.** The standalone `w_ij` matrices were archived in September 2026 to
`archive/wij_structural/` because `w_ij` and `m_ij` share shape, labels, sign pattern and E/I
block structure, differing only in magnitude — so loading the wrong one produces a
plausible-looking wrong result with no diagnostic to catch it.

The decomposition that motivated the split, and which remains an open scientific question:

| Operator | E→I / I→E block-norm ratio |
|---|---:|
| `m_ij = w_ij · κ_j` | 912× |
| `w_ij` alone | 86× |

Because κ_j is indexed by the postsynaptic type, the entire E→I block carries interneuron κ and
the entire I→E block carries principal-cell κ (medians 0.0935 vs 0.0345 Hz/pA; ~9× at the
tails). So roughly one order of magnitude of the feedforward asymmetry is **excitability**, and
the remaining ~86× is **wiring**. Framing this as structural-vs-functional is a deferred
comparison, not an abandoned one. Caveat: the archived `wij_matrix.csv` was never verified to be
exactly the netlist's `w_ij` column — reconstruct from `mij_netlist.csv` before quoting 86×.

---

## 3. Hard conventions

Violating any of these silently produces wrong results that look right.

### Orientation

- **On disk**, every `mij_matrix.csv` is `M[pre, post]` — rows are sources, columns targets.
- **In analysis**, anything reading `M[i, j]` as "influence of j on i" needs `M[post, pre]`.
- Verified empirically: netlist reconstruction residual is **1.414 as stored, 3.0 × 10⁻¹⁸
  transposed**.
- **The orientation oracle.** Because E/I identity comes from outgoing weight signs, the block
  sign pattern *is* an orientation test, and the separation is total:

  | Block | `M[post, pre]` | `M[pre, post]` |
  |---|---|---|
  | E→E | 100% positive (n=382) | 100% positive (n=382) |
  | E→I | 100% positive (n=352) | 0% positive (n=171) |
  | I→E | 0% positive (n=171) | 100% positive (n=352) |
  | I→I | 0% positive (n=736) | 0% positive (n=736) |

  `assert_post_pre()` runs this at loader boundaries. A transposed matrix raises.
- **Why it matters so much:** a flipped run is invisible to every structural diagnostic —
  recurrency, order index, feedforward fraction and bandwidth are all identical under the
  reversed permutation. The only symptoms are a reversed cell order and the silent disappearance
  of the trisynaptic paths, both of which read as findings rather than bugs.

### Normalisation

| Context | Rule |
|---|---|
| Schur modes, propagation, linear systems | ρ(M) = 0.95 |
| Inhibitory stable-response analyses | ρ(M) = 0.85 |
| Jacobian | ρ(W) = 0.95, applied to **W and never to J** — scaling J rescales the leak and changes τ |
| Neural-mass target (GSN) | `S = α·Mᵀ / ρ(Mᵀ)`, α = 0.9, to make the map contractive |
| Orbit-count features | log1p then z-score per feature (monotone, so injectivity preserved) |
| Coupling as a feature | `sign(x) · log1p(|x|)` — keeps Dale's sign across 8 orders of magnitude |

Only spectral-radius normalisation is transpose-invariant. `row_l1` and `column_l1` swap meaning
under transposition. **Consequence for claims: all magnitudes are conditional on an arbitrary
normalisation constant. Ordinal and comparative claims are defensible; absolute ones are not.**

### Naming

Two vocabularies exist and both are needed:
- **Arrow sense**: `E_to_I` means E source → I receiver (rows I, columns E of a `[post, pre]`
  matrix).
- **Partition sense**: `M[E, I]` means `M[E rows, I columns]` = I source → E receiver.

A bare `EI` is ambiguous and one module can emit both senses in one run. Always disambiguate.

### Continuous vs discrete

Both readings are used and they are exactly equivalent:

```
discrete:    x[t+1] = M x[t]              stable iff ρ(M) < 1
continuous:  τ dx/dt = −x + W x           J = (−I + W)/τ,  stable iff max Re λ < 0
```

`(I + J) − W = 0` to machine precision, so M is the forward-Euler propagator of J at dt = τ.
Largest stable Euler step is 1.53τ, so dt = τ has ~50% margin; ρ(exp(Jτ)) = 0.9512 vs 0.95 for
Euler. **Self-connections are autapses in W, never the −I leak in J**; a `no_self` variant never
removes the leak.

---

## 4. Working thesis

> The hippocampal–entorhinal circuit is functionally feedforward inside a structurally recurrent
> graph, and that geometry — not its spectrum — governs its transient dynamics.

Sharper variant, better supported and stranger: **the feedforward geometry lives in the weights,
not in the topology.** Under degree-preserving nulls the observed graph is if anything *harder*
to make feedforward by edge count, while 99.5% of its total |weight| can be placed below the
diagonal — far beyond what the nulls achieve.

Anchors: Goldman (2009) functionally feedforward networks; Murphy & Miller (2009) balanced
amplification; Hennequin et al. (2012, 2014) stability-optimised circuits; Trefethen & Embree
(2005) for the non-normal machinery.

**Where the geometry comes from (measured, Sept 2026).** Because `q_i` depends only on the
presynaptic type and `κ_j` only on the postsynaptic type, the operator is a two-sided diagonal
scaling of a base synaptic matrix: `M[post,pre] = diag(κ) · Bᵀ · diag(q)` with
`B = c ⊙ g ⊙ ΔV ⊙ τ`. Surrogates, each normalised to ρ = 0.95:

| Operator | Henrici | peak amplification | E→I / I→E | fwd weight fraction |
|---|---:|---:|---:|---:|
| M (real) | 99.3% | 27.13× | 912 | 0.995 |
| binary, sign only | 24.4% | 1.06× | 1.44 | 0.716 |
| M / κ_j | 80.5% | 7.74× | 85.8 | 0.982 |
| M / q_i | 45.1% | 1.45× | 11.9 | 0.936 |
| M / (κ_j q_i) | 35.9% | 1.13× | 0.96 | 0.916 |

Binary topology amplifies 1.06×; the real operator amplifies 27×. The base synaptic matrix is
E/I-**balanced** (0.96). The ~900× feedforward asymmetry is manufactured by the census (`q`,
median 5,615 excitatory vs 333 inhibitory types) and postsynaptic gain (`κ`, ~2.7× median,
~9× at the tails). Ranking: q > κ > wiring. See `docs/WEIGHTS_VS_TOPOLOGY.md`.

Note a real distinction from balanced amplification: that mechanism requires strong E→I *and*
strong I→E. Here I→E is ~900× weaker than E→I in the normalised operator, which is a pure
feedforward stage rather than a balanced one.

---

## 5. What has been done, with numbers

### Stage 00 — preprocessing (orientation, scale, ordering)

Nine ordering methods implemented: reverse Cuthill–McKee, Laplacian Fiedler, PageRank, Tarjan
SCC condensation, Tarjan DFS finishing order, modified Tarjan, anatomical, feedback-arc local
search, scrambled. Only shared permutations `M' = PᵀMP` are used. On the 85-cell matrix, by
recurrency ratio / order index: PageRank 0.697 / 0.262, modified Tarjan 0.790 / 0.583, Fiedler
0.807 / 0.464, anatomical 0.830 / 0.714, RCM 0.887 / 0.643, Tarjan DFS 0.920 / 0.940,
Original = Tarjan SCC 1.000 / 0.702.

Key findings: "Tarjan" in Borst & Leibold (2023) is a DFS, not an SCC condensation — the graph
is a single SCC with alphabetical labels, so condensation is a no-op. A feedback-arc local search
reaches a 0.475 backward-edge ratio and places 99.52% of |weight| below the diagonal, beating
every implemented method. The order index is saturated at this density (a greedy walk finds a
path through 81 of 85 cells, so ≥0.952 is free) and should not be used as a discriminator.

### Stage 01 — excitatory backbone

32-node E→E graph, 355 positive transitions. Seven path-construction methods compared. Headline
result is the ordering, not the paths: **0.9496 forward-weight fraction** against a permutation
null of 0.486 (sd 0.124, n = 2,000), best-of-2,000 = 0.815, empirical p < 0.005. One SCC; 208
forward vs 147 feedback edges; 748,421 vs 39,719 in weight.

The recovered order runs superficial EC → CA3/CA2 → CA1 → subiculum and deep EC — the canonical
direction — with **no anatomical information supplied**.

E/I dynamics along that coordinate: spectral abscissa −0.42 → −0.61 as inhibitory gain g goes
0 → 2; numerical abscissa +27.20; gap ≈ 27.6. 285 feedforward vs 578 feedback inhibitory edges;
feedforward stronger per edge (0.00748 vs 0.00444) and more local (mean displacement 10.5 vs
21.8). The inhibitory *placement* null is **negative** — observed terminal peak at the 22.4th
percentile of a 500-draw null, p = 0.226 / 0.776.

### Stage 02 — inhibition and disinhibition

Block fractures at ρ = 0.85 (dominant real eigenvalue after removing one block):

| Removed | with_self | no_self |
|---|---|---|
| E→E | 0.85 → 0.063 | 0.85 → 0.099 |
| I→I | 0.85 → 0.844 | 0.85 → 0.838 |
| cross E/I | 0.85 → 0.886 | 0.85 → 0.996 |

E→E carries the growth mode; I→I is nearly inert; cross-E/I coupling is *stabilising*.

Inhibitory Schur complement `M_eff_E = M[E,E] + M[E,I](I − M[I,I])⁻¹M[I,E]`: ρ(M[E,E]) of 0.886
(with self) and 0.996 (no self) both map to ρ(M_eff_E) = 0.850. Inhibitory feedback sets the
operating point from either starting point.

Motif enrichment: E→I→I→E loop z ≈ 52; disinhibitory I→I→E z ≈ 28, stable across variants. Top
disinhibitory sources, stable across variants: **CA1 Perforant Path Associated, MEC LIII
Superficial Multipolar Interneuron, DG AIPRIM**.

The paradox worth noting: the E→I block carries almost all the matrix norm, yet inhibitory
feedback appears *weak* in the spectrum. Large E→I with negligible I→E return produces small
inhibitory feedback eigenvalues — the influence is transient and directional, not spectral.

### Stage 03 — motifs

Vocabulary: the 13 Davis–Holland–Leinhardt directed triad classes, validated letter-for-letter
against Rees et al. (2016) superpatterns A–M (A=021U, B=021C, C=021D, D=111D, E=030T, F=111U,
G=030C, H=120D, I=201, J=120C, K=120U, L=210, M=300), including their data-independent
combinatorics (86 connected + 18 disconnected E/I patterns).

Twin-node unrolling: each of the 71 self-connected types is split into two exchangeable instances
joined by a reciprocal dyad, each inheriting full external connectivity → 156 instances, 5,562
edges, no self-loops. Orbit counting by a 64-pattern lookup table over node triples (exact, no
|Aut| correction, 0.4 s vs 30 s for VF2), validated against `nx.triadic_census` at 119,342 triads.

Deduplication to cell-type multisets: 119,342 instance triads → **20,513 distinct** = 18,572
cross-type (which equals the 85-node type-graph census exactly, class by class) + 1,941
within-type triads that a type-graph census cannot see at all.

Significance under a degree- **and dyad**-preserving null:

| Class | Observed | Expected | Fold | z |
|---|---:|---:|---:|---:|
| 300 (fully reciprocal triangle) | 978 | 127 | 7.7× | +62 |
| 120D | 1,666 | 568 | 2.9× | +49 |
| 030T (feed-forward) | 2,249 | 1,516 | 1.5× | +23 |
| 201 | 405 | 3,617 | 0.11× | −62 |
| 111D | 1,022 | 3,857 | 0.27× | −58 |

Transitive and cycle-averse; reciprocity clusters into closed triangles rather than open chains.
`networkx.directed_edge_swap` is **unusable** as a null here — it destroys ~60% of the 422 mutual
dyads and would report spurious enrichment for every reciprocal class.

Geography: only 24.2% of triads are single-region; class 300 is 66% single-region while open
two-edge classes (021C, 111U, 021D) span 2–3 regions >94% of the time. Long-range structure is
made of two-edge motifs, not triangles.

E/I: composition is 1E2I 35.6%, 3I 32.0%, 2E1I 21.3%, 3E 11.1% — close to what the 32E/53I
inventory alone predicts. *Placement* is not: under E/I label permutation on the fixed graph,
all-excitatory triads are over-represented in the reciprocal classes (201 z=+13, 300 z=+9,
210 z=+7, 111D z=+6). 85 of the 86 Dale-consistent coloured patterns occur; the only absentee is
the purely inhibitory 3-cycle I1→I2→I3→I1.

### Stage 04 — graph substructure networks (GSN)

**Stated purpose: to establish that the 3-node motif is dynamically important and usable as an
analysis primitive** — not to win a benchmark. Bouritsas et al. (2022): append subgraph-orbit
counts so the network exceeds 1-WL expressivity. 30 vertex-orbit + 30 edge-orbit features. Target
is the steady state of a linear neural-mass model under unit DG Granule drive. Splits are drawn
over cell types, never node instances (twins are exchangeable and would leak).

| Model | Test R² (10 splits) | Spearman |
|---|---|---|
| MLP, first-order only | 0.21 ± 0.53 | 0.71 |
| MLP + orbit counts | 0.44 ± 0.42 | 0.74 |
| MPNN | 0.22 ± 0.94 | 0.73 |
| GSN-v | 0.44 ± 0.29 | 0.72 |
| GSN-e | 0.57 ± 0.23 | 0.73 |

Orbit counts help with and without message passing; GSN-e ≥ GSN-v as Theorem 3.3 predicts.
Spearman is flat across all models — **the gain is calibration, not ranking**.

Vertex disambiguation δ = 0.9529 (81 of 85 signatures distinct). The four ties — CA1 Basket /
CA1 Basket CCK+, CA1 Bistratified / CA1 Ivy, CA3 Basket / CA3 Basket CCK+, MEC LIII Superficial
Multipolar / Trilayered — are types Hippocampome separates on molecular markers, not connectivity.

Known weakness: `edge_attr` already carries `m_ij`, which is most of what determines the
dynamics, so the headroom for purely structural features is small. **A constant-`edge_attr` run is
the decisive experiment and has not been done.**

### Stage 05 — eigen / Schur

`A = Q T Q*`. Schur over eigendecomposition because the eigenvector basis has condition number
9.9 × 10⁴, so modes are not separately excitable. Direction: `T[m,n]` with n > m means mode n
drives mode m — the cascade runs high index to low.

| Variant | Raw ρ | ‖M‖_F | Non-normality | Signal at t=10 |
|---|---:|---:|---:|---:|
| with_self | 26,213.13 | 77.18 | 5,869.90 | 17.35 |
| no_self | 14,837.87 | 136.35 | 18,319.83 | 39.48 |

Dominant vector overlap between variants 0.901. CA1 dominates 26 of 85 modes; CA3 and DG 15 each;
MEC 12. High-loading cells are predominantly inhibitory. Greedy contributor trees average 4.26
edges / 2.74 components. Linear systems (no_self vs with_self): path ‖·‖_F 323.87 vs 131.96,
Gramian trace 56,246 vs 9,547, peak gain 193.43 vs 69.39, both peaking at frequency 0.01.

**Caution:** variants are normalised independently, so all with/without statements are about
matched normalised dynamics, not raw diagonal mass.

### Stage 06 — Jacobian, continuous time

At ρ(W) = 0.95, τ = 1, leak = 1:

| Quantity | Value |
|---|---:|
| Spectral abscissa α(J) | −0.0500 |
| Numerical abscissa ω(J) | 29.58 |
| Abscissa gap | 29.63 |
| Peak amplification | 27.11× at t = 1.2τ |
| Kreiss lower bound (ε = 0.05) | 18.47 |
| Henrici departure | 77.17 = 99.3% of ‖J‖_F |
| Eigenvector condition number | 9.9 × 10⁴ |

**The control:** a normal surrogate carrying identical eigenvalues has peak amplification exactly
1.0×. The amplification is structure, not spectrum.

Block Frobenius norms: E→I 77.02, E→E 4.91, I→I 0.40, I→E 0.084.

`ρ(J) = 1.31` is **not** instability — it is a speed statistic. λ_J = λ_W − 1, so the spectrum is
a radius-0.95 cloud translated to centre −1, and max|λ_J + 1| = 0.95 exactly.

Stiffness: Re(μ) spans [−0.310, 0.950], so per-mode effective time constants run 0.76τ to 20τ,
with 44 of 85 modes faster than τ. Read as spectrum, not as measurable relaxation times.

---

## 6. Repository map

**`AscoliLab`** — `docs/` (orientation and transpose audits, the methodology deck),
`EE_backbone/`, `inhib_modulation/`, `motif_analysis/`, `tarjan_reorder/`, `schur_decomp/`
(notebooks 01–10), `jacobian_analysis/` (01 written, 02–06 scaffolded), `non-normal_matrices/`,
`archive/wij_structural/`.

**`graph_subnetworks`** — `gsn_connectome.py`, `GSN_connectome_workflow.ipynb`,
`motif_characterization.ipynb`, `figures/` (14 PNGs), `motif_tables/` (13 CSVs).

Canonical inputs everywhere: `mij_netlist.csv` (1,641 rows) and `mij_matrix.csv` (85×85, stored
`[pre, post]`).

---

## 7. Known-wrong and unresolved

Do not present any of these as new findings; they are already identified.

1. Both null-test conclusions in `tarjan_reorder/01` **reverse** under a real feedback-arc
   optimiser (12 nulls so far, indicative only). Needs 100 nulls.
2. EE-backbone path-weight tables predate two solver fixes (an inadmissible branch-and-bound
   bound; `pulp.LpStatus` misread as an optimality certificate) and need regeneration.
3. The Schur coupling direction is labelled backwards in `tarjan_reorder/03`
   (`schur_upper_coupling_out_abs` is actually coupling *in*). Affects labels and prose, not the
   invariant rankings.
4. Motif-block orientation in notebooks 03/04 is asserted via a hard-coded boolean, not verified
   against the `w_ij` metadata columns.
5. `motif_analysis` motif ids are built from a `[pre, post]` matrix and use a 120-class two-edge
   vocabulary; they do **not** join against `tarjan_reorder` or `inhib_modulation` motif tables
   without remapping.
6. The E/I backbone placement null is first-order (out-degree and weight multiset preserved,
   target in-degree free) and returns a negative result.
7. The notebook 04 signed Laplacian is dominated by a self-loop that should be zeroed.
8. `wij_matrix.csv` was never verified to be exactly the netlist's `w_ij` column.

**Not yet attempted at all:** any null for the non-normality measures themselves; any
input/output (B, C matrix) formulation; any simulation, linear or nonlinear; any sensitivity
analysis on τ or κ; any second connectome.

---

## 8. Do not assume

- **Do not assume magnitudes are meaningful.** Everything is spectral-radius normalised. "27×"
  is conditional on ρ = 0.95. Ordinal comparisons are safe; absolute ones are not.
- **Do not quote peak amplification without naming the norm.** It is basis-dependent. The same
  operator, under a similarity transform that leaves the spectrum identical to 4 × 10⁻¹⁵, gives
  peak amplification 27.1× in extensive population activity (the poster's variable), 575× under
  an A/√q split, and 47,695× in per-cell firing rate. α, stability and the Euler-equivalence
  argument are invariant; ω(J), the Kreiss constant and peak amplification are not. Henrici
  departure stays 99.3–100% in every basis, so "essentially fully non-normal" is the robust
  qualitative claim.
- **Do not assume the diagonal is autaptic.** It is within-type coupling between distinct cells.
- **Do not assume ρ(J) < 1 is the stability criterion in continuous time.** It is not; the
  criterion is max Re λ < 0.
- **Do not assume non-normality is surprising.** Any Dale-respecting E/I network with strong E→I
  is non-normal. The contribution has to be cell-type resolution, or a matched-null comparison,
  or methodology — not the bare observation.
- **Do not assume this is a spiking or single-neuron model.** It is population-level, census-
  weighted, over neuron *types*, and linear.
- **Do not assume short-term plasticity is included.** U, τ_r and τ_f are excluded.
- **Do not assume the trisynaptic recovery is circular.** No anatomical information enters the
  ordering optimisation; the anatomical order is a separately scored comparison row.
- **Do not confuse the two `EI` senses** (§3, Naming).
- **Do not propose `w_ij`-based analyses as current work.** `m_ij` is the operator; `w_ij` is
  archived pending an explicit structural-vs-functional study.

---

## 9. Literature anchors

- Borst A, Leibold C (2023). Connecting Connectomes to Physiology. *J Neurosci* 43(20):3599–3610.
  (Matrix reordering, Eqs. 3–10, Figs. 5–7.)
- Bouritsas G, Frasca F, Zafeiriou S, Bronstein MM (2022). Improving graph neural network
  expressivity via subgraph isomorphism counting. *IEEE TPAMI* 45(1):657–668.
- Rees CL, Wheeler DW, Hamilton DJ, White CM, Komendantov AO, Ascoli GA (2016). Graph theoretic
  and motif analyses of the hippocampal neuron type potential connectome. *eNeuro*
  3(6):ENEURO.0205-16.2016.
- Davis JA, Leinhardt S (1972); Holland PW, Leinhardt S (1970). Triad classes; the U|MAN null.
- Trefethen LN, Embree M (2005). *Spectra and Pseudospectra.* Princeton.
- Goldman MS (2009). Memory without feedback in a neural network. *Neuron* 61:621–634.
- Murphy BK, Miller KD (2009). Balanced amplification. *Neuron* 61:635–648.
- Hennequin G, Vogels TP, Gerstner W (2012, 2014). Non-normal amplification; stability-optimised
  circuits.
- Milisav F et al. (2024). Strength-preserving network randomisation. *Nat Comput Sci.*
- Dayan P, Abbott LF. *Theoretical Neuroscience*, ch. 7.
- Santa Cruz Yabarrena J-M, Ascoli GA. CN3 poster — the `m_ij` construction.

---

## 10. What this file is good for

Reasonable prompts to pair with it:

- Propose dissertation titles that commit to the functionally-feedforward claim rather than
  describing methods.
- Generate hypotheses that connect the motif-level results (§5 Stage 03) to the operator-level
  results (§5 Stage 06) — the two currently do not constrain each other.
- Design the null ensemble that could falsify the thesis, and say what result would kill it.
- Propose input (B) and output (C) matrices that are anatomically defensible for this circuit.
- Critique the argument as a sceptical committee member from systems neuroscience, not from
  applied mathematics.
- Identify which of the 85 cell types the thesis makes checkable predictions about, and what
  published experiment would test each.

Weak prompts, given the material: anything requiring absolute magnitudes; anything assuming
single-neuron resolution; anything treating the non-normality observation itself as the novelty.
