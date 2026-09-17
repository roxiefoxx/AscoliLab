# Kickoff brief — Rees-style excitability analysis

Paste this into a new chat. Attach or point at `AscoliLab/PROJECT_CONTEXT.md` first; it carries
the data conventions, the numbers, and the "do not assume" list. This brief only adds what is
specific to this task.

## Task

Build a Rees-style excitability analysis in its own subdirectory, `AscoliLab/rees_excitability/`.
Do not modify anything outside that directory.

## Why this is being done

The 86 Dale-consistent coloured triad patterns are already enumerated and validated against
Rees et al. (2016) — their superpatterns A–M map letter-for-letter onto the 13 DHL classes, and
their data-independent combinatorics reproduce exactly. The excitability score (ES) was
deliberately removed from `graph_subnetworks/GSN_connectome_workflow.ipynb` because it should not
*replace* the raw E/I enumeration as a feature vocabulary. It is being reintroduced here as a
separate, downstream comparison computed on top of the completed enumeration.

## The key constraint, which shapes the whole design

ES is a function of (triad class, E/I colouring of its three nodes). Under Dale's law — verified
on all 1,570 edges with zero exceptions — a binary directed adjacency plus E/I node labels
reconstructs `sign(M)` exactly. So **ES lives at the `sign(M)` rung of the representation
hierarchy: it can see direction and sign, and can never see magnitude.**

Measured consequence (with-self, ρ = 0.95):

| representation | E→I / I→E | Henrici | peak amplification |
|---|---:|---:|---:|
| full weighted M | 911.81 | 99.3% | 27.13× |
| `sign(M)` | 1.43 | 24.4% | 1.06× |

**Do not build an ES matrix.** ES is defined on triads, not dyads; marginalising to an 85×85
object discards the three-node structure it is about, and the result would have no units and no
interpretable spectrum. Treating it as a second operator alongside M invites comparing eigenvalues
of objects that are not the same kind of thing.

## Two deliverables

**1. Composition test.** Compute the connectome's ES distribution over the existing 86-pattern
census — Σ count × ES, and mean ES per triad. Test against the two null ensembles already built:
the degree- and dyad-preserving rewiring, and E/I label permutation on the fixed graph. Question:
is this connectome's motif composition more or less excitable than chance? This is Rees's
statistic on Rees's vocabulary, extending their 122-type result to the 85-type parameterised
subset.

**2. Per-cell predictor — the one that matters.** Sum ES over the motifs each cell type
participates in, weighted by orbit position, giving an 85-vector of motif-derived excitability.
Correlate against the project's own dynamical rankings:

- non-normal contribution (ablation Δ in peak amplification)
- the disinhibition score (CA1 Perforant Path Associated, MEC LIII Superficial Multipolar,
  DG AIPRIM top the current ranking)
- Schur mode participation
- out-strength and eigenvector centrality, as the trivial baselines

**State the prediction before running it.** ES sits at the `sign(M)` rung, which amplifies 1.06×
against the real 27×. Expect weak correlation with the non-normal measures and better correlation
with the disinhibition ranking, which is itself largely a motif-count quantity. A stated
prediction that survives is worth far more than a correlation found by fishing. Either outcome is
publishable: agreement means a cheap combinatorial score recovers what an expensive dynamical
analysis finds; disagreement localises exactly where the weights matter.

## Inputs

- `graph_subnetworks/mij_netlist.csv` — 1,641 rows, the authoritative source
- `graph_subnetworks/motif_tables/coloured_pattern_vocabulary.csv` — the 86 patterns
- `graph_subnetworks/gsn_connectome.py` — the 64-pattern orbit-count lookup, exact, ~0.4 s
- Rees CL et al. (2016) *eNeuro* 3(6):ENEURO.0205-16.2016 — Fig. 8A carries the eight ES values,
  already reproduced exactly in earlier work; use that reproduction rather than re-deriving

## Pitfalls already paid for

- `networkx.directed_edge_swap` destroys ~60% of the 422 mutual dyads. Use the dyad-preserving
  rewiring in `tarjan_reorder/optimal_ordering_lop.py`.
- The diagonal is within-type coupling between distinct cells, not autapses; 71 of 85 types carry
  one. Triads need three distinct nodes, so use the twin-node unrolling.
- `motif_analysis/triad_utils.py` builds motif ids from a `[pre, post]` matrix under a 120-class
  two-edge vocabulary. Those ids do **not** join against the DHL tables without remapping. Use the
  DHL vocabulary.
