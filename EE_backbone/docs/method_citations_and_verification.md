# Method citations and verification for the EE/EI backbone notebooks

Scope: every section, method, and analysis in
`ee_backbone_method_comparison.ipynb`, `ee_backbone_deterministic_stochastic_comparison.ipynb`,
and `ei_backbone_analysis.ipynb`, mapped to literature support, plus a verification pass
over the mathematics in `ee_backbone_analysis.py`, `ee_backbone_stochastic_methods.py`,
and `ei_backbone_analysis_helpers.py`.

Citation policy used here, in the order the project instructions specify:
connectomics first; then network neuroscience, information theory, or graph theory;
then other disciplines with the analogy stated explicitly.

---

## Part 1 — Verification findings (read this first)

Four findings change how current results should be reported. Two are defects, one is a
statistical error whose correction flips a stated conclusion, and one is a confirmation.

### 1.1 CONFIRMED DEFECT — the MILP "Optimal" status is not a certificate, and the benchmark gaps are not gaps to an optimum

`milp_benchmark_gaps.csv` contains one negative gap: for seed `SUB CA1 Projecting Pyramidal`,
the Hamiltonian MILP returns a 31-edge path scoring **150,476.13** while the unconstrained
simple-path MILP returns a 30-edge path scoring **141,695.86**, with `milp_status == "Optimal"`.

This is a contradiction, not a modeling nuance. A Hamiltonian path from a seed *is* a simple
directed path from that seed, so it is feasible for the unconstrained MILP. If the
unconstrained solution were optimal, no feasible solution could beat it. The README currently
reads this negative gap as evidence that "the two MILP formulations answer different
questions." That reading is not available: the two formulations differ by a restriction of
the feasible set, and a restriction cannot beat the relaxation.

I re-solved that seed's unconstrained MILP with the notebook's own formulation and
`timeLimit=30`:

```
timeLimit=30s -> pulp status = Optimal, objective = 153968.717
```

Same formulation, same seed, same nominal time limit, but a different objective from the
recorded run (141,695.86) — and both are labeled `Optimal`. PuLP maps CBC's
"stopped on time / solution limit" outcome onto `LpStatusOptimal`, so `milp_status` reports
"a feasible incumbent exists", not "optimality proven". The true optimum for this seed is at
least 153,968.72.

Consequences:

- The README claim "all 32 seed solves reached `Optimal`" does not establish optimality for any seed.
- The notebook's statement that "when the solver status is `Optimal`, the reported path is the
  exact maximum-weight simple path" is false as written.
- `weight_gap_to_milp` is a gap to a possibly-suboptimal incumbent, so it can be negative and
  cannot be read as "path weight left on the table."
- `milp_subtour_elimination`'s first place in `method_summary.csv` is a lower bound on its own
  performance, not a certified ceiling for the other methods.

Fix: read CBC's own status rather than PuLP's mapping, and record the solver's MIP gap.
`problem.solverModel` is not populated by `PULP_CBC_CMD`; the robust route is to pass
`PULP_CBC_CMD(msg=True, timeLimit=...)`, parse `Result - Optimal solution found` vs
`Result - Stopped on time limit`, or switch to `pulp.HiGHS_CMD` / `pulp.GUROBI_CMD`, which
expose a gap. Then re-run with a limit large enough that the gap closes, and only then call
the result a benchmark. Until then, rename the column `weight_gap_to_milp_incumbent` and
report the time limit alongside it.

The MILP *formulation* itself is correct. I checked it independently: with `y[seed]=1`,
zero in-degree at the seed, in-degree `== y[i]` and out-degree `<= y[i]` elsewhere, and MTZ
constraints `u[j] >= u[i] + 1 - n(1-x_ij)` over `u in [0, n-1]`, any cycle disjoint from the
seed would require strictly increasing order around the cycle, which is infeasible. The
selected arcs therefore form a single simple path rooted at the seed. This is the standard
MTZ construction [17], and the big-M value `n` is tight enough to be valid and loose enough
to be non-binding when `x_ij = 0`.

### 1.2 CONFIRMED DEFECT — the branch-and-bound upper bound is not admissible, so `bb_status == "complete"` does not mean exact

In `branch_bound_path`:

```python
max_extension[i] = max(w + self_weights[j] for j, w in adjacency[i])   # best edge LEAVING i
def upper_bound(score, visited):
    unvisited = [i for i in range(n) if i not in visited]
    return score + max_extension[unvisited].sum()
```

The bound sums the best outgoing edge of every *unvisited* node. But a continuation from the
current node `u` that visits `v1..vk` uses edges leaving `u, v1, ..., v(k-1)` — it uses
`max_extension[u]` and does **not** use `max_extension[vk]`. The bound omits the term it needs
(`max_extension[u]`) and includes one it does not (`max_extension[vk]`). When the current
node's best outgoing edge is large and the remaining nodes are near-sinks, the bound falls
below the true achievable score and the optimum is pruned.

Minimal counterexample, run against the exact pruning rule from the module:

```
3 nodes, edges 0 -> 1 (w = 10) and 1 -> 2 (w = 100); node 2 has no outgoing edges.
max_extension = [10, 100, 0]
branch_and_bound returns: [0, 1] with score 10.0
true optimum:             [0, 1, 2] with score 110.0
prune event: path [0,1], score 10.0, bound 10.0, incumbent 10.0  -> branch discarded
```

The bound equals the incumbent exactly, so the `<= best_score + EPS` test fires and the
100-weight edge is never explored. This is not a guardrail artifact — it happens with no time
or node limit in force.

Consequence: the notebook's note "branch-and-bound is exact only when it finishes with
`bb_status == 'complete'`" understates the problem. Even a `complete` run is not exact. In the
current outputs every seed stopped at `node_limit`, so nothing published is affected yet, but
the claim in the markdown should be corrected before anyone raises the limits and trusts a
`complete` status.

Fix (one line, still cheap and now admissible):

```python
def upper_bound(score, current, visited):
    unvisited = [i for i in range(n) if i not in visited]
    return score + float(max_extension[current]) + float(max_extension[unvisited].sum())
```

This is looser than the tight bound but valid. The tight version takes the `k` largest values
of `{max_extension[u]} u {max_extension[v] : v unvisited}` where `k = |unvisited|`, which costs
a partial sort per node and is worth it if you intend branch-and-bound to certify anything.

### 1.3 STATISTICAL ERROR — the E/I null-model conclusion is reversed once the test matches the null distribution

`ei_backbone_analysis.ipynb` reports observed terminal peak `8.69137e-06`, null mean
`8.16921e-06`, null SD `2.21152e-05`, `z = 0.024`, and the README concludes "the observed
placement is close to the null expectation."

Two problems. First, `N_NULL = 25` cannot support a z-score: I resampled 25 draws from a
larger null 40 times and the resulting z ranged from **-0.18 to 33.70** (sd 6.80). The
statistic is not reproducible at that sample size. Second, and more fundamental, the null
distribution is severely right-skewed, so a z-score is the wrong summary regardless of `n`.

Re-running the notebook's own `randomize_inhibitory_targets` null with n = 500:

```
observed terminal peak = 8.69137e-06
null: mean = 2.15762e-06   sd = 1.07881e-05   median = 2.87196e-08
      skew = 6.80   excess kurtosis = 45.91   max/median = 2784
z-score (as coded)          =  0.606
observed percentile in null =  96.4%
empirical p (obs >= null)   =  0.0379
null spectral abscissa: [-0.7261, -0.2643], 0 of 500 unstable
```

The mean sits far above the median because a handful of null draws produce very large
terminal peaks; the SD is five times the mean. Against that distribution the observed value
is at the **96.4th percentile**, empirical p = 0.038 one-sided — the opposite of "close to the
null expectation." The observed inhibitory target placement transmits *more* to the terminal
node than randomized placement typically does.

The `ee_backbone_deterministic_stochastic_comparison.ipynb` notebook already prescribes the
correct estimator in its "Suggested significance tests" section:
`p = (1 + #{null >= observed}) / (1 + n_null)`. Apply it here too, raise `N_NULL` to at least
500 (each draw is one `expm_multiply`; 500 runs in well under a minute), and report the
percentile and empirical p instead of z. A z-score can be retained only alongside a
normality check it will not pass.

Important scope limit: `BACKBONE_LABELS` is still `None`, so "terminal node" means "last row
of the CSV." The corrected statistic is a statement about the *inference procedure*, not yet a
biological result. Once a real backbone ordering is supplied the same corrected test becomes
interpretable, and — given that the current placeholder already lands at p = 0.038 — it is
worth re-running as a primary analysis rather than a diagnostic.

One further caveat on that null: it preserves each source's inhibitory out-degree and weight
multiset but not target in-degree, which the notebook states correctly. That makes it a
first-order placement null. A strength-preserving randomization [37] would be the stronger
comparison, and the simulated-annealing procedure there handles signed and directed networks
specifically.

### 1.4 CONFIRMED — the E/I non-normality framing is sound, and stronger than the notebook claims

I re-derived the system matrix `A(g) = -alpha*I + beta*(W_E - g*W_I)` with the notebook's
`alpha = 1.0`, `beta = 0.8`, and 95th-percentile normalization (scale = 23,940.5):

```
  g   spectral abscissa   numerical abscissa   G_max (41 pts)   G_max (801 pts)
 0.0        -0.41907            27.19923           24.6321          24.6368
 0.5        -0.46065            27.18926           24.2068          24.2069
 1.0        -0.50417            27.17932           23.7997          23.8083
 1.5        -0.55206            27.16942           23.4101          23.4369
 2.0        -0.60853            27.15955           23.0374          23.0899
```

Three things follow.

- The system is **asymptotically stable at every g tested** (max Re(lambda) from -0.42 to -0.61).
  The precondition for the notebook's transient-amplification framing holds, so `G_max ~ 24`
  is genuine non-normal amplification and not truncated exponential growth. This was worth
  checking: had the spectral abscissa been positive, `G_max` over a finite window would have
  been meaningless.
- The **numerical abscissa is +27.2 while the spectral abscissa is -0.42**. That gap is the
  quantitative signature of strong non-normality, and it is a much sharper result than the
  README's "G_max is about 24.63 at g=0 and about 23.04 at g=2." Report it: the matrix is
  stable in every eigendirection yet its symmetric part has a large positive eigenvalue, so
  perturbations along the right directions grow by more than an order of magnitude before
  decaying. `transient_amplification` already computes `numerical_abscissa` and the notebook
  never mentions it.
- The 41-point time grid is fine — it recovers `G_max` to within 0.02% of an 801-point grid.
  No action needed.

Also checked, and clean: `dale_sender_summary` classifies senders at a 0.9 fraction threshold
while `ee_backbone_analysis.is_excitatory_sender` uses 0.5. On this dataset both thresholds
return the identical set of 32 excitatory senders, and that set matches
`mij_EE_matrix.csv` exactly — there are no mixed-sign senders in between. The two constants
should still be reconciled into one module-level parameter, because the agreement is a
property of this matrix and not of the code.

`load_signed_matrix` orientation also checks out: with `csv_orientation="source_rows"` it
transposes to give `W_state[target, source]`, which is what `xdot_i = sum_j A_ij x_j` requires,
and `dale_sender_summary`, `inhibitory_node_ablation` (zeroes column j = j's outgoing
inhibition), and `inhibitory_backbone_motifs` all index consistently with that convention.

### 1.5 Smaller items

- **Hamiltonian MILP objective omits the self-weight term** that `total_score` includes. With
  `include_self_connections=False` both are zero, so nothing is wrong today, but the
  self-connection variant would report a score that does not match the objective it optimized.
- **`np.trapezoid`** requires NumPy >= 2.0; pin it in the environment notes.
- **`exact_bitmask_dp_path`** is correct — the popcount-ordered sweep, the snapshot of
  `current_items` before mutation, and the parent-pointer reconstruction all hold. It is
  O(n * |dp|) because it rescans the whole table per size; a `defaultdict(list)` keyed on
  popcount would remove that factor if you ever raise `NOTEBOOK_EXACT_DP_MAX_E_NODES`.
- **`beam_dp_path`** carries `new_states = dict(states)` forward, so states already expanded
  compete for beam slots against their own children. That is a defensible design (it keeps
  short high-weight paths alive) but it means the beam width is not "5000 states per depth",
  and the mean DP path length of 4.47 edges is partly an artifact of it. Worth a sentence in
  the notebook.

---

## Status: all four findings resolved

| # | Finding | Status | Where fixed |
| --- | --- | --- | --- |
| 1.1 | MILP `Optimal` is not a certificate | **Fixed** | `_solver_status()` reads `problem.sol_status`; both solvers return `proven_optimal`; gap column renamed `weight_gap_to_milp_incumbent` with a `gap_is_certified` flag; `build_milp_benchmark_table` now raises if any method beats a *proven* optimum |
| 1.2 | Branch-and-bound bound not admissible | **Fixed** | `upper_bound()` now charges `max_extension` for the current node as well as all unvisited nodes. Verified: the 3-node counterexample now returns 110.0, and branch-and-bound matches exact bitmask DP on 0 of ~1,800 completed seed-instance pairs disagreeing across 300 random graphs |
| 1.3 | E/I null-model conclusion reversed | **Fixed** | `inhibitory_target_null_model` returns an empirical p-value, percentile, skew, kurtosis, and per-draw spectral abscissa; `N_NULL` default raised 25 → 500; z-score retained only as `z_score_do_not_use` |
| 1.4 | Non-normality understated | **Fixed** | The E/I notebook now prints the spectral/numerical abscissa pair and states explicitly whether `G_max` is genuine transient amplification |

Additionally, the maximum-weight Hamiltonian MILP objective now includes the constant
self-weight term so that its reported objective matches `total_score`, and
`summarize_methods` carries `objective` and `comparable_on_summed_weight` columns so
methods optimizing different objectives are not silently ranked against each other.

### Vahidi's feedback-arc ordering is implemented

`ee_backbone_ordering.py` implements the method recommended in Part 3, item 1, following
Vahidi (2025) [20]: weighted Eades-Lin-Smyth greedy construction, gain-aware
best-position refinement, exact decomposition over strongly connected components, and
randomized restarts. It is wired into both E-to-E notebooks as a seventh method and
writes `backbone_ordering.csv`, which `ei_backbone_analysis.ipynb` now loads in place of
the CSV-label-order placeholder.

Results on `mij_EE_matrix.csv`:

```
strongly connected components : 1  (all 32 types)
positive edges                : 355   (208 forward, 147 feedback)
forward edge weight           : 748,421
feedback edge weight          :  39,719
forward-weight fraction       : 0.9496
permutation null (n = 2,000)  : mean 0.486, sd 0.124, best-of-2000 0.815
empirical p                   : < 0.005
```

About 95% of positive E-to-E edge weight points forward, against a chance expectation
near 50%. The subnetwork is a single strongly connected component yet is close to
feedforward in weight terms, meaning its recurrence is carried by a small fraction of the
total weight. The recovered ordering runs from entorhinal superficial layers through CA3
and CA2 into CA1 and ends in subiculum and deep entorhinal layers — the expected
direction of flow, and not imposed.

Two consequences worth noting.

- **The per-seed paths under this ordering are exact.** Forward edges under a fixed
  ordering form a DAG, so the maximum-weight simple path from any seed is solvable in
  O(V + E) by one sweep. Verified against exact bitmask DP on 1,209 cases with zero
  mismatches. None of the heuristic path methods carry a comparable guarantee.
- **Those paths are short and light, and that is expected.** A seed placed late in the
  ordering has few forward options, while the MILP may travel backward through the
  ordering freely. The ordering is excluded from the path-weight boxplot and flagged
  `comparable_on_summed_weight = False` in the method summary, because ranking it on
  `summed_weight` would misrepresent what it optimizes.

### What changed in the E/I results once a real ordering replaced the placeholder

The corrections and the new ordering interact, so the E/I numbers moved:

| Quantity | Placeholder (CSV order) | Feedback-arc ordering |
| --- | ---: | ---: |
| Feedforward inhibitory edges | 387 | 285 |
| Feedback inhibitory edges | 476 | 578 |
| Feedforward mean magnitude | 0.006217 | 0.007479 |
| Feedback mean magnitude | 0.004817 | 0.004442 |
| Mean abs displacement, feedforward | — | 10.5 |
| Mean abs displacement, feedback | — | 21.8 |
| Null-model verdict | z = 0.024, "close to null" | 22.4th percentile, p = 0.776 — not distinguishable from chance |

Under the real ordering, feedforward inhibition is stronger per edge and much more local
than feedback inhibition (mean displacement +7 vs -13), which is a substantive and
independently checkable claim against the microcircuit literature cited in section 2.C.

The null-model result also illustrates why finding 1.3 mattered as a *procedural* point
rather than a biological one. Under the placeholder backbone the corrected test gave
p = 0.038; under the real ordering it gives p = 0.776. The null even flips from strongly
right-skewed (skew +6.8) to strongly left-skewed (skew -2.5), because the skew depends on
which node the backbone terminates at. A z-score would have been the wrong summary in both
cases, in opposite directions.

---

## Part 2 — Citations by notebook section

### 2.A `ee_backbone_method_comparison.ipynb`

#### Framing: treating a signed connectome as a positively weighted directed graph

The notebook restricts `mij_matrix.csv` to excitatory senders and clips to positive
off-diagonal weights. This is a defensible but consequential preprocessing choice and should
be cited as such rather than left implicit.

- Directly in connectomics: signed, directed, weighted connectome representations and the
  argument that edge-weighting schemes are a modeling choice rather than ground truth [76].
- The case for analyzing weighted rather than binarized brain graphs, and the information lost
  by thresholding [75].
- Thresholding and sparsification are not neutral preprocessing steps — different rules produce
  measurably different topologies [79], with a directed-network comparison of data-driven
  thresholding methods in [13].
- Methodological caution transferable from ecology: in weighted interaction networks,
  path-based measures are frequently computed on weights that were never transformed
  appropriately for the algorithm, and a majority of surveyed studies contained such errors
  [77]. The analogy is exact — an interaction-strength matrix used as a path cost — and the
  five-point reporting checklist there is worth adopting verbatim.
- Relevant caveat on interpretation: global graph metrics on weighted connectomes are often
  near-collinear with mean edge weight, so any claim that a *topological* property drives a
  result needs a mean-weight control [78].

#### Domain grounding for this specific matrix

The node set is Hippocampome.org neuron types, which carries its own citation obligations:

- The knowledge base and the neuron-type classification [62], its v2.0 simulation-oriented
  release [61], and the subicular extension [74].
- The potential-connectivity estimates that produce a matrix of this shape, from
  axonal-dendritic overlap [63], and the Peters'-rule caveats that govern how "potential"
  should be read [65].
- Synaptic amplitudes underlying the weights [66], and neuron-type counts [67].
- Prior graph-theoretic and motif analysis of exactly this potential connectome — the closest
  methodological precedent in the literature for this project, including its rich-club and
  hub findings [64].

#### Greedy tree/path

A locally-maximal-edge walk. Framed correctly in the notebook as a baseline.

- In connectomics, greedy local routing is one of the canonical communication policies, sitting
  at the "local information only" end of the routing spectrum [4], and it is formally
  expressible as a biased random walk within a unified multi-policy framework [5].
- Reviews placing greedy/navigation policies alongside shortest-path and diffusion [1].
- The empirical point that greedy navigation performs surprisingly well on connectomes, and
  can even recover directionality from undirected topology, is in [3] and [Seguin 2019, in 2.B].

#### Maximum spanning tree

- The MST as a connectome backbone is well established, including its use as an empirical null
  and reference skeleton for the human connectome [9], with a systematic review of network-size
  effects and pathology sensitivity [11] and a structural-backbone application after stroke [12].
- The orthogonal-MST refinement addresses the main weakness of a single MST — it discards
  most of the graph — by optimizing global efficiency minus wiring cost [10]. If the current
  MST paths look too short (mean 5.47 edges, lowest mean score), OMST is the natural upgrade
  and stays inside the connectomics literature.
- Alternative principled backbones worth benchmarking against: the distance backbone, which is
  the provably minimal subgraph preserving all shortest paths and has been computed on the
  human connectome [8]; nonparametric MDL-based backboning that selects edge count
  automatically [14]; and the statistical edge-filter family [13], available as software [15].

Note the caveat the notebook already states — the tree is built on symmetrized weights while
scores use directed weights. Cite [8] or [14] when arguing that this is a structural summary
rather than a directed-path optimizer.

#### Branch-and-bound

Almost no connectomics literature applies branch-and-bound to path extraction directly, so this
needs graph-theory and operations-research support plus a life-sciences analogue.

- Branch-and-bound as the standard exact paradigm for this class of problem, with a modern
  survey of how the bound and branching rules govern performance [Scavuzzo 2024, MILP survey — see 2.B].
- Life-sciences analogue: branch-and-cut and branch-and-bound for the quadratic TSP arising in
  a bioinformatics application, where instances up to 100 nodes were solved to proven
  optimality [26]. This is the closest precedent to what this notebook attempts at n = 32, and
  it is the right citation for arguing that exactness is achievable at this scale with a
  correct bound (see finding 1.2).

#### Dynamic programming (exact bitmask / beam)

- The subset-and-terminal state representation is Held-Karp [16], correctly cited in the notebook.
- Life-sciences analogue for the exponential state space and the resort to heuristics:
  read-overlap genome assembly, where the true sequence corresponds to a Hamiltonian path and
  the problem is NP-hard under most formulations [22]. That paper is also a good model for the
  framing move this project could make — set complexity aside and ask what the data can
  identify — and [23] is the necessary corrective on how that dichotomy is usually taught.
- For the beam variant specifically, cite it as a heuristic and do not present it as a
  benchmark; the notebook already does this correctly.

#### MILP with subtour elimination

- MTZ formulation [17], correctly cited.
- Connectomics precedent for integer programming over a connectivity matrix: matrix seriation
  by relaxing an integer program, applied to volume-EM cell-to-cell matrices specifically to
  segregate feedforward from feedback connections [18], with the companion methods review
  linking linear dynamical systems analysis and matrix reordering as the two mathematical
  entry points into connectomic data [19].
- On solver status and gaps — directly relevant to finding 1.1 — the branch-and-bound machinery
  underlying MILP solving and what its termination criteria actually certify [Scavuzzo 2024, 2.B].

#### Maximum-weight asymmetric Hamiltonian path

This is the method most in need of a citation, and the search turned up something that should
change the analysis.

- **Recommended reframing.** For a connectome, the "backbone ordering" question is more
  naturally a *feedback arc set / linear ordering* problem than a Hamiltonian path problem:
  find the vertex ordering maximizing total forward-pointing edge weight. This has been done on
  the FlyWire connectome at scale, with greedy, gain-aware local-refinement, and
  strongly-connected-component-based algorithms [20]. The difference matters: a Hamiltonian
  path scores only the 31 consecutive edges it selects, discarding the other 324 positive
  edges, whereas feedback-arc minimization scores *every* edge against the ordering. If the
  scientific question is "what is the dominant direction of excitatory flow through these 32
  types", the ordering objective answers it and the Hamiltonian path answers a much narrower
  one. Recommend adding it as a seventh method — it is cheap, it is directly precedented in
  connectomics, and it is robust in a way a single path is not.
- The seriation literature makes the same connection from the other direction: exact matrix
  seriation via mathematical optimization includes a Hamiltonian-path-based reformulation for
  structured neighborhoods [21], and the connectomics-specific seriation method is [18].
- Life-sciences analogue for maximum-weight Hamiltonian path as an inference objective:
  genome assembly and genetic linkage mapping. Ordering contigs by long-range linking
  measurements reduces to a TSP whose maximum-likelihood solution is a hidden Hamiltonian
  cycle, recoverable by LP relaxation [24]; linkage-map construction casts marker ordering
  directly as a Hamiltonian path problem solved with TSP solvers [25]. Both are strong
  analogies — an unknown linear order inferred from noisy pairwise interaction strengths —
  and both make the same point this project should absorb: the value of the Hamiltonian
  formulation is the *ordering*, not the path weight.

#### Method comparison, disagreement tables, and the boxplot

- The general practice of comparing multiple communication/routing models on the same
  connectome under one shared scoring rule, and adjudicating between them, is established
  [3, 5, 6]; [3] in particular ranks fifteen communication models on a common substrate and is
  the closest template for `method_summary.csv`.
- On statistical power and what a 32-seed comparison can and cannot support [42].

### 2.B `ee_backbone_deterministic_stochastic_comparison.ipynb`

#### Weighted random-walk / diffusion path ensemble

This is the best-supported method in the whole project.

- Random walks and diffusion are one of the two canonical connectome communication models,
  reviewed in [1], compared head-to-head against shortest-path routing with diffusion
  explaining substantially more functional-connectivity variance [6], and benchmarked among
  fifteen models [3].
- The biased-random-walk formulation the notebook implements — transition probability
  proportional to edge weight, tunable between local and global information — is exactly the
  routing-spectrum model [4], and the multi-policy generalization showing that shortest paths
  and greedy navigation are both special cases of biased random walks [5].
- Recent evidence that connectomes may be organized for within-module diffusion and
  between-module routing, across six organisms [7] — directly relevant if you later partition
  these 32 types.
- Formal grounding: random walks and diffusion on networks [27]; random-walk communication
  under an efficient-coding/compression account, which is where the entropy diagnostics
  connect [30].
- Diffusion on weighted *and directed* connectomes with reaction terms, including lesioned
  variants [51] — the closest match to this project's directed, weighted, sign-split setting.

#### Monte Carlo tree search

No connectomics precedent found. Cite the method plus a life-sciences application.

- Canonical survey [31] and the modern update covering non-game domains [32].
- **Life-sciences analogue, strong:** MCTS for biochemical circuit topology design, where
  enumeration becomes intractable past four components and circuit assembly is recast as a
  sequence of decisions [33]. This is the right citation — the analogy is a combinatorial
  search over biological network structure with a scoring function, which is what this
  notebook does.
- Network-domain analogue: MCTS for goal-directed graph construction on spatial and
  infrastructure networks, with UCT improvements for the single-agent, action-space-linear
  setting [34]; and an adapted MCTS for a combinatorial optimization problem with pruning,
  heuristic simulation policy, and beam width [35] — that combination of enhancements maps
  onto what `monte_carlo_tree_search_path` implements.

#### Boltzmann path sampling

- **Best fit, and it is in network science rather than connectomics:** the Randomized Shortest
  Paths framework defines a Boltzmann distribution over network paths with an inverse
  temperature interpolating between shortest paths and pure random walks [29]. That is
  precisely this method's mathematics, it comes with derived betweenness measures, and it
  supersedes the ad-hoc framing currently in the notebook. Cite it as the formal basis rather
  than as a comparison.
- The large-deviations treatment of random walks on networks, which recovers generalized
  optimal paths through a Doob transform and covers exactly the "which paths dominate under a
  tilted ensemble" question [28].
- For the temperature parameter as an exploration control, MCTS with Boltzmann exploration and
  the maximum-entropy tree-search family [36] — useful because it also documents the failure
  mode: optimal actions under a maximum-entropy objective need not be optimal under the
  original objective. Worth a caveat sentence given that `BOLTZMANN_TEMPERATURE = 0.25` is
  fixed and never swept, despite the notebook describing the method as a sensitivity analysis.

Recommendation: the notebook says Boltzmann sampling "makes it a useful sensitivity analysis:
if the same edges dominate across temperatures," but only one temperature is run. Either sweep
it (0.1, 0.25, 0.5, 1.0, 2.0) or reword the claim.

#### Ensemble diagnostics: terminal entropy, edge visit probability, unique termini

- Entropy over a path ensemble and edge-visitation statistics as the readout of a stochastic
  communication model [1, 30]; the path-ensemble framing with k-shortest-path ensembles,
  disjointness, and the efficiency/resilience tradeoff [2] is the closest precedent for using
  an ensemble rather than a single optimum.
- Randomized-shortest-path betweenness gives a principled edge-visitation measure with the
  same temperature parameter [29].

#### Suggested null models and significance testing

The table in cell 25 is sound and mostly needs citations attached to each row.

- Weight-shuffled and strength-preserving randomization: **[37] is the citation this table
  most needs** — a simulated-annealing procedure for randomizing weighted networks that
  preserves strength sequence, generalizes to directed and signed networks, outperforms
  other rewiring algorithms, and is shown to change inferences about brain-network
  organization. It covers the "weight_shuffled_fixed_topology" and
  "degree_strength_preserving_rewire" rows simultaneously.
- Degree-preserving configuration models and the labeling subtleties that determine which
  configuration model is appropriate [38].
- Randomization of *biological* networks preserving each node's functional characterization,
  including signed and directed cases [39] — relevant to the "rowwise_target_shuffle" row.
- The distinction between generative models and null models, and what each can license [40].
- Null-model choice as a determinant of false-positive rate, with a comparative benchmark of
  ten frameworks [41]; and statistical power in network neuroscience [42], which bears on
  whether 32 seeds and 25 nulls can support the claims being made.
- The `(1 + #{null >= obs}) / (1 + n_null)` estimator the table specifies is standard and
  correct — and, per finding 1.3, should be back-ported into the E/I notebook.

#### MILP and branch-and-bound in this notebook

Same citations as 2.A. The MILP-solving survey covering branch-and-bound, primal heuristics,
node selection, and what solver termination certifies is Scavuzzo et al. 2024 [see reference
list], which is the appropriate support for the solver-status discussion finding 1.1 requires.

### 2.C `ei_backbone_analysis.ipynb`

#### Linear dynamical model `xdot = A(g)x + u`, `A = -alpha*I + beta*(W_E - g*W_I)`

- Linear systems theory applied to brain networks — impulse response, controlled response,
  state transitions — as a tractable framework whose parameters remain biophysically
  interpretable [48]; the network-control-theory protocol built on that foundation, including
  its explicit guidance on comparing outputs against null network models [49].
- The methods review arguing that linear dynamical systems analysis is one of two principal
  mathematical routes into connectomic data, and that connectivity alone generates new time
  constants far longer than single-neuron membrane time constants [19]. This is the single
  best framing citation for this notebook.
- Connectome-constrained dynamics as an established practice: reservoir-computing on empirical
  connectomes with arbitrary imposed dynamics [50]; reaction-diffusion on weighted, directed
  connectomes including lesioned variants [51].
- **Closest methodological analogue found, and it is very close:** a "frozen rate operator"
  built from the complete larval Drosophila connectome, run with no fitted single-neuron
  parameters, and compared against a degree-and-weight-matched rewiring ensemble; it reports
  the operator as non-normal and near-linear and separates what the exact wiring fixes from
  what degree and weight statistics already fix [47]. That decomposition is exactly what this
  notebook's null model is groping toward, and the paper is a template for how to report it.

#### Excitation/inhibition split and the Dale diagnostic

- Dale's law as a structural constraint and its consequences for network dynamics [58];
  eigenspectral consequences of Dale's law under sparsity, which govern stability and the
  location of outlier eigenvalues [59]; and the stability-complexity relationship under Dale's
  law via random matrix theory [60].
- The notebook's caveat 2 ("a negative entry is an inhibitory effective interaction unless cell
  identity supports an inhibitory-neuron interpretation") is correct and is exactly the concern
  raised in [58]. Since these are Hippocampome neuron types with known neurotransmitter
  assignments [62], the identity information exists and the Dale diagnostic can be validated
  against it rather than inferred from sign fractions — worth doing.

#### Weight normalization by the 95th percentile

Robust scaling is standard practice but is rarely cited; the honest framing is that this is a
choice about the dynamic regime, not a neutral rescaling.

- The general argument that connectome edge weights are a modeling choice rather than a
  measured quantity [76], and that weighted analyses retain biologically relevant information
  binarized ones discard [75].
- Thresholding/normalization as a non-neutral methodological decision with measurable
  topological consequences [79, 13].
- Recommend reporting the spectral abscissa alongside the scale (verification 1.4 gives
  -0.42 at g=0), since the scale's only job is to place the system in a stable regime and that
  is now verifiable rather than assumed.

#### Impulse propagation, peak amplitude, AUC, time-to-peak

- Impulse response as the canonical linear-systems readout on a connectome [48, 19].
- Signal propagation in complex networks generally [Ji 2023, reference list], and network
  communication measures derived from propagation dynamics [1, 3].

#### Inhibitory strength sweep and spectral abscissa

- Eigenspectrum and stability under E/I structure and Dale's law [59, 60]; excitation-inhibition
  balance and its relationship to circuit dynamics [Liang 2024, reference list].
- Global inhibition strength as a direct control parameter on amplification, with the explicit
  finding that weakening inhibitory weights decreases amplification [45] — this is the paper
  to cite for interpreting the g-sweep, and it predicts the direction observed here.

#### Global transient amplification `G(t) = ||exp(At)||_2`

The strongest-supported analysis in the E/I notebook.

- Non-normal amplification in balanced E/I networks, using a Schur decomposition to separate
  non-normal amplification from dynamical slowing [43]; optimal control of transient dynamics
  in balanced networks [44].
- Regimes and mechanisms of transient amplification in abstract and biological networks,
  including the Dale's-law-constrained case and the amplification/dimensionality tradeoff [45].
- Non-normal dynamics on non-reciprocal networks — reactivity and effective dimensionality in
  neural circuits [46]. "Reactivity" there is the numerical abscissa this notebook already
  computes and never reports; see verification 1.4.
- Non-normality of a real connectome operator, measured directly [47].

Recommendation: report the numerical abscissa (+27.2) next to the spectral abscissa (-0.42).
The pair is the standard non-normality signature in [43, 45, 46] and is a far stronger
statement than `G_max` alone.

#### Inhibitory node and edge ablation

- In-silico lesioning of connectome models is standard [55], including virtual lesions in
  personalized dynamic models [Wei 2022, reference list] and flow-based single- and
  double-neuron ablation on the C. elegans connectome [54] — the last is the closest match,
  being a directed neuronal connectome with exhaustive systematic ablation.
- Systematic connectome manipulation as a framework, with tooling [56].
- **Necessary caveat, and it applies directly here:** exhaustive single-element lesioning of a
  small network produces *biased* contribution estimates, because function arises from
  coalitions of interacting elements; multi-site lesioning captures effects single-site
  analysis misses [52]. The notebook ablates one node or one edge at a time. The
  game-theoretic multi-perturbation framework in [53] is the principled alternative and is
  applied to large-scale brain models. At minimum this belongs in the caveats list; at 85
  nodes, a Shapley-style multi-site analysis on the top inhibitory sources is tractable.
- Empirical validation that node-degree predicts lesion impact in a real brain-wide network
  [57] — useful as a sanity check on `inhibitory_out_degree` vs `delta_terminal_peak`.

#### Feedforward/feedback inhibitory motif classification by backbone displacement

- The displacement-along-an-ordering definition is precisely the matrix-seriation logic:
  order cells along the flow of information and feedforward/feedback connections segregate
  above and below the diagonal [18, 19]. Cite these — they justify the whole construction and
  also supply a principled way to *obtain* the ordering rather than using CSV row order.
  [20] gives the scalable weighted-directed version.
- Cortical hierarchy and the feedforward/feedback distinction as an organizing principle
  [Vezoli 2020, Markov 2013 — reference list], and hierarchy assignment from connectivity
  [Harris 2019, reference list].
- **Domain-specific grounding for the hippocampal-entorhinal case, which this section needs
  most:** feedforward vs feedback inhibition is not a geometric label in this circuit, it is an
  identified microcircuit distinction. PV+ basket cells mediate feedback inhibition and
  dendrite-targeting interneurons feedforward inhibition, with lateral inhibition roughly
  ten times more abundant than recurrent inhibition in dentate gyrus [68]; PV-mediated
  feedforward inhibition gates perirhinal-entorhinal signal output [69]; neurogliaform cells
  mediate feedback inhibition in MEC layer I driven by layer II pyramidal cells [70]; PV
  connectivity onto principal cells varies systematically along the MEC dorsoventral axis [71];
  and specific inhibitory synapses shift the CA1 balance between feedforward and feedback
  inhibition [72]. Hippocampus-to-entorhinal feedback is itself organized into two functionally
  distinct pathways with layer-specific excitation vs feedforward inhibition [73].
- This matters for interpretation: the notebook currently reports 387 feedforward vs 476
  feedback edges with feedforward edges of higher mean magnitude, under a *placeholder*
  ordering. Because [68] establishes that the feedforward/feedback ratio is a real, measured,
  cell-type-specific property of this circuit, the motif table becomes checkable against
  ground truth once a real ordering is supplied. That is a genuine validation opportunity.

#### Null model for inhibitory placement

See finding 1.3 for the statistical correction. Citations: [37] for strength-preserving
signed/directed randomization, [38] for configuration-model choice, [39] for biological-network
rewiring preserving node function, [41] for null-model comparison and false-positive rates,
[47] for the degree-and-weight-matched ensemble applied to a connectome operator specifically.

#### Modeling caveats (cell 30)

All five caveats are accurate. Supporting citations: linear approximation and its limits [48];
Dale-consistent sign structure [58, 59]; Wilson-Cowan and firing-rate follow-ups
[Liang 2024, reference list]; delay-differential and distributed-delay formulations
[Mitjans 2023, reference list]. Caveat 2 in particular is well founded — see the Dale note above.

---

## Part 3 — Method suggestions ranked by value

1. ~~**Add feedback-arc-set / linear-ordering as a method** [20].~~ **DONE** — `ee_backbone_ordering.py`. It answers the backbone-ordering
   question using all 355 positive edges instead of the 31 a Hamiltonian path selects, it is
   directly precedented on a connectome at far larger scale, and it is cheap. This is the
   highest-value single addition.
2. ~~**Fix the two solver-correctness defects** (1.1, 1.2).~~ **DONE** — see the status table above.
3. ~~**Replace the E/I z-score with an empirical p-value at n >= 500** (1.3).~~ **DONE**.
4. ~~**Report the numerical abscissa alongside the spectral abscissa** (1.4).~~ **DONE**.
5. ~~**Supply a real `BACKBONE_LABELS` ordering.**~~ **DONE** — the E/I notebook loads
   `backbone_ordering.csv`. Everything downstream in the E/I notebook is conditional on it. The seriation methods [18] and feedback-arc ordering [20] are two
   principled ways to produce one from the data; the E-to-E MILP or Hamiltonian result is a
   third, which would connect the two notebooks.
6. **Move from single-site to multi-site ablation** [52, 53] for the inhibitory-influence
   analysis, or at minimum state the single-site bias as a caveat.
7. **Adopt strength-preserving randomization** [37] as the primary null in both notebooks, in
   place of the current first-order target shuffle.
8. **Upgrade MST to OMST** [10] if the MST arm is to remain a serious comparison rather than a
   floor; consider the distance backbone [8] or MDL backboning [14] as additional structural
   baselines.
9. **Sweep the Boltzmann temperature** or reword the sensitivity-analysis claim.
10. **Validate the Dale diagnostic against Hippocampome neurotransmitter assignments** [62]
    rather than inferring sign class from out-edge fractions.

---

## Reference list

[1] [Communication dynamics in complex brain networks](https://consensus.app/papers/details/35226ede27b157a6a5b07ac21de7fa7b/?utm_source=claude_desktop) (Avena-Koenigsberger et al., 2017, Nature Reviews Neuroscience, 841 citations, DOI: 10.1038/nrn.2017.149)

[2] [Path ensembles and a tradeoff between communication efficiency and resilience in the human connectome](https://consensus.app/papers/details/20cf0252ad7050fb96ff4cf32573fe0c/?utm_source=claude_desktop) (Avena-Koenigsberger et al., 2016, Brain Structure and Function, 85 citations, DOI: 10.1007/s00429-016-1238-5)

[3] [Network communication models improve the behavioral and functional predictive utility of the human structural connectome](https://consensus.app/papers/details/c8687ec9ed215270807d91eef55c1db0/?utm_source=claude_desktop) (Seguin et al., 2020, Network Neuroscience, 114 citations, DOI: 10.1101/2020.04.21.053702)

[4] [A spectrum of routing strategies for brain networks](https://consensus.app/papers/details/e8e6bee6971e545db772ef9ab12f1bfb/?utm_source=claude_desktop) (Avena-Koenigsberger et al., 2018, PLoS Computational Biology, 119 citations, DOI: 10.1371/journal.pcbi.1006833)

[5] [Multi-policy models of interregional communication in the human connectome](https://consensus.app/papers/details/6dcdd2bab0435835bedea401e5112863/?utm_source=claude_desktop) (Betzel et al., 2022, bioRxiv, 23 citations, DOI: 10.1101/2022.05.08.490752)

[6] [Comparing models of information transfer in the structural brain network and their relationship to functional connectivity: diffusion versus shortest path routing](https://consensus.app/papers/details/a602e423c6ef58fda3824a4636529a9b/?utm_source=claude_desktop) (Neudorf et al., 2022, Brain Structure & Function, 11 citations, DOI: 10.1007/s00429-023-02613-2)

[7] [Connectome architecture favours within-module diffusion and between-module routing](https://consensus.app/papers/details/6d903c5a209658539a0424ca578e0bee/?utm_source=claude_desktop) (Seguin et al., 2025, bioRxiv, 2 citations, DOI: 10.1101/2025.02.10.637586)

[8] [The distance backbone of complex networks](https://consensus.app/papers/details/bc4c1aa78e735802b636893c3b3cf0d1/?utm_source=claude_desktop) (Simas et al., 2021, Journal of Complex Networks, 40 citations, DOI: 10.1093/comnet/cnab021)

[9] [Minimum spanning tree analysis of the human connectome](https://consensus.app/papers/details/71f76f8bf7f959c88be2a69ffe1340dc/?utm_source=claude_desktop) (van Dellen et al., 2018, Human Brain Mapping, 75 citations, DOI: 10.1002/hbm.24014)

[10] [Topological Filtering of Dynamic Functional Brain Networks Unfolds Informative Chronnectomics: A Novel Data-Driven Thresholding Scheme Based on Orthogonal Minimal Spanning Trees (OMSTs)](https://consensus.app/papers/details/a750986137ee54b39068040a39b98c75/?utm_source=claude_desktop) (Dimitriadis et al., 2017, Frontiers in Neuroinformatics, 128 citations, DOI: 10.3389/fninf.2017.00028)

[11] [Minimum spanning tree analysis of brain networks: A systematic review of network size effects, sensitivity for neuropsychiatric pathology, and disorder specificity](https://consensus.app/papers/details/25704a29bd8c51258d1255df6864654b/?utm_source=claude_desktop) (Blomsma et al., 2022, Network Neuroscience, 41 citations, DOI: 10.1162/netn_a_00245)

[12] [Modified structural network backbone in the contralesional hemisphere chronically after stroke in rat brain](https://consensus.app/papers/details/bac45904bee05dbb93904c3e1ec5d38e/?utm_source=claude_desktop) (Sinke et al., 2017, Journal of Cerebral Blood Flow & Metabolism, 22 citations, DOI: 10.1177/0271678x17713901)

[13] [Comparison of data-driven thresholding methods using directed functional brain networks](https://consensus.app/papers/details/2e3cf3a740085b98b68e19c1baf2ce50/?utm_source=claude_desktop) (Manickam et al., 2024, Reviews in the Neurosciences, 8 citations, DOI: 10.1515/revneuro-2024-0020)

[14] [Fast nonparametric inference of network backbones for weighted graph sparsification](https://consensus.app/papers/details/226a0d77195257acbe2a1ce30eec8f1a/?utm_source=claude_desktop) (Kirkley, 2024, Physical Review X, 5 citations, DOI: 10.1103/4pg6-mtmt)

[15] [backbone: An R package to extract network backbones](https://consensus.app/papers/details/d321c25f1237572c910a82b4abaabef6/?utm_source=claude_desktop) (Neal, 2022, PLoS ONE, 63 citations, DOI: 10.1371/journal.pone.0269137)

[16] Held, M., & Karp, R. M. (1962). A dynamic programming approach to sequencing problems. *Journal of the Society for Industrial and Applied Mathematics*, 10(1), 196-210. DOI: 10.1137/0110015 — already cited in the notebook.

[17] Miller, C. E., Tucker, A. W., & Zemlin, R. A. (1960). Integer programming formulation of traveling salesman problems. *Journal of the ACM*, 7(4), 326-329. DOI: 10.1145/321043.321046 — already cited in the notebook.

[18] [Connectivity Matrix Seriation via Relaxation](https://consensus.app/papers/details/4368603793135d02b454e68eea058543/?utm_source=claude_desktop) (Borst, 2024, PLOS Computational Biology, 3 citations, DOI: 10.1371/journal.pcbi.1011904)

[19] [Connecting Connectomes to Physiology](https://consensus.app/papers/details/fd2af4adc7bf5ad88c2e65a7eecba35f/?utm_source=claude_desktop) (Borst et al., 2023, The Journal of Neuroscience, 7 citations, DOI: 10.1523/jneurosci.2208-22.2023)

[20] [Feedforward Ordering in Neural Connectomes via Feedback Arc Minimization](https://consensus.app/papers/details/fcc73c793885559aa4dd05458575b9a1/?utm_source=claude_desktop) (Vahidi, 2025, ArXiv, 0 citations, DOI: 10.48550/arxiv.2506.13799)

[21] [Exact Matrix Seriation through Mathematical Optimization: Stress and Effectiveness-Based Models](https://consensus.app/papers/details/c8651cc982d955189dc9ce831e67e722/?utm_source=claude_desktop) (Blanco et al., 2025, DOI: 10.48550/arxiv.2506.19821)

[22] [Information-optimal genome assembly via sparse read-overlap graphs](https://consensus.app/papers/details/177e7080b68c5b699ee1d988fd803651/?utm_source=claude_desktop) (Shomorony et al., 2016, Bioinformatics, 37 citations, DOI: 10.1093/bioinformatics/btw450)

[23] [What do Eulerian and Hamiltonian cycles have to do with genome assembly?](https://consensus.app/papers/details/bced27b567c7545089e9a8bae7d000c5/?utm_source=claude_desktop) (Medvedev et al., 2021, PLoS Computational Biology, 12 citations, DOI: 10.1371/journal.pcbi.1008928)

[24] [Hidden Hamiltonian Cycle Recovery via Linear Programming](https://consensus.app/papers/details/b33149e0805b5ce38b6601b866fd20c8/?utm_source=claude_desktop) (Bagaria et al., 2018, Operations Research, 26 citations, DOI: 10.1287/opre.2019.1886)

[25] [Use of traveling salesman problem solvers in the construction of genetic linkage maps](https://consensus.app/papers/details/ed13e0df344c5ebd91008251c25e6d8c/?utm_source=claude_desktop) (Allen, 2015, DOI: 10.25675/3.09650)

[26] [Exact algorithms and heuristics for the Quadratic Traveling Salesman Problem with an application in bioinformatics](https://consensus.app/papers/details/36e97682fcfe584cb7c5df3112dc5ddc/?utm_source=claude_desktop) (Fischer et al., 2014, Discrete Applied Mathematics, 44 citations, DOI: 10.1016/j.dam.2013.09.011)

[27] [Random walks and diffusion on networks](https://consensus.app/papers/details/fb21d88739f453968d85e2e5c221183f/?utm_source=claude_desktop) (Masuda et al., 2016, Physics Reports, 701 citations, DOI: 10.1016/j.physrep.2017.07.007)

[28] [Generalized optimal paths and weight distributions revealed through the large deviations of random walks on networks](https://consensus.app/papers/details/206488a5597f5624b1e3a6af79080f61/?utm_source=claude_desktop) (Gutierrez et al., 2020, Physical Review E, 16 citations, DOI: 10.1103/physreve.103.022319)

[29] [Two betweenness centrality measures based on Randomized Shortest Paths](https://consensus.app/papers/details/f818fa8402c15fa4869e64afbc6957c6/?utm_source=claude_desktop) (Kivimaki et al., 2015, Scientific Reports, 72 citations, DOI: 10.1038/srep19668)

[30] [Efficient coding in the economics of human brain connectomics](https://consensus.app/papers/details/c690a9a3b9e05e27b83caffb7f636a93/?utm_source=claude_desktop) (Zhou et al., 2020, Network Neuroscience, 34 citations, DOI: 10.1162/netn_a_00223)

[31] [A Survey of Monte Carlo Tree Search Methods](https://consensus.app/papers/details/1c66ac3acf6f5ede8f454e94194b2dc8/?utm_source=claude_desktop) (Browne et al., 2012, IEEE Transactions on Computational Intelligence and AI in Games, 3496 citations, DOI: 10.1109/tciaig.2012.2186810)

[32] [Monte Carlo Tree Search: a review of recent modifications and applications](https://consensus.app/papers/details/cad0f4354bc953198a66a9768a653e01/?utm_source=claude_desktop) (Swiechowski et al., 2021, Artificial Intelligence Review, 455 citations, DOI: 10.1007/s10462-022-10228-y)

[33] [Designing biochemical circuits with tree search](https://consensus.app/papers/details/94e956bce7f65d9c808cf0c47944b0e2/?utm_source=claude_desktop) (Bhamidipati et al., 2026, bioRxiv, 1 citation, DOI: 10.1101/2025.01.27.635147)

[34] [Planning spatial networks with Monte Carlo tree search](https://consensus.app/papers/details/c3521dcad7515b5c968fa3a3fb8c5629/?utm_source=claude_desktop) (Darvariu et al., 2021, Proceedings of the Royal Society A, 12 citations, DOI: 10.1098/rspa.2022.0383)

[35] [Exploring search space trees using an adapted version of Monte Carlo tree search for a combinatorial optimization problem](https://consensus.app/papers/details/913006bc44c65eb78ae077ef6c570fae/?utm_source=claude_desktop) (Jooken et al., 2020, Computers & Operations Research, 22 citations, DOI: 10.1016/j.cor.2022.106070)

[36] [Monte Carlo Tree Search with Boltzmann Exploration](https://consensus.app/papers/details/648e7e93b83e5a948f0b0d71e4ca5049/?utm_source=claude_desktop) (Painter et al., 2024, ArXiv, 17 citations, DOI: 10.48550/arxiv.2404.07732)

[37] [A simulated annealing algorithm for randomizing weighted networks](https://consensus.app/papers/details/6652ed2a554253fd9b1e1b30b0c4e5cb/?utm_source=claude_desktop) (Milisav et al., 2024, Nature Computational Science, 20 citations, DOI: 10.1038/s43588-024-00735-z)

[38] [Configuring Random Graph Models with Fixed Degree Sequences](https://consensus.app/papers/details/622ebcde482b5ed6a90591707e0dd0fb/?utm_source=claude_desktop) (Fosdick et al., 2016, SIAM Review, 248 citations, DOI: 10.1137/16m1087175)

[39] [Efficient randomization of biological networks while preserving functional characterization of individual nodes](https://consensus.app/papers/details/e531127e25c45072952adb03b42fa947/?utm_source=claude_desktop) (Iorio et al., 2016, BMC Bioinformatics, 45 citations, DOI: 10.1186/s12859-016-1402-1)

[40] [Generative models for network neuroscience: prospects and promise](https://consensus.app/papers/details/7a86b4fb0bcb5185a43aae74565eddcf/?utm_source=claude_desktop) (Betzel et al., 2017, Journal of the Royal Society Interface, 123 citations, DOI: 10.1098/rsif.2017.0623)

[41] [Comparing spatial null models for brain maps](https://consensus.app/papers/details/c7020725b28358e789ab658909d64092/?utm_source=claude_desktop) (Markello et al., 2021, NeuroImage, 325 citations, DOI: 10.1016/j.neuroimage.2021.118052)

[42] [Statistical power in network neuroscience](https://consensus.app/papers/details/41a5585607c05b3c87e91b75c86bf854/?utm_source=claude_desktop) (Helwegen et al., 2023, Trends in Cognitive Sciences, 48 citations, DOI: 10.1016/j.tics.2022.12.011)

[43] [Non-normal amplification in random balanced neuronal networks](https://consensus.app/papers/details/98efff1dbcc756ff8d50ca279236feab/?utm_source=claude_desktop) (Hennequin et al., 2012, Physical Review E, 93 citations, DOI: 10.1103/physreve.86.011909)

[44] [Optimal control of transient dynamics in balanced networks supports generation of complex movements](https://consensus.app/papers/details/a970f7d0b933506ea306a12703bd9b66/?utm_source=claude_desktop) (Hennequin et al., 2014, Neuron, 333 citations, DOI: 10.1016/j.neuron.2014.04.045)

[45] [Regimes and mechanisms of transient amplification in abstract and biological neural networks](https://consensus.app/papers/details/2c28aebc024f50d5a0141c95890a5425/?utm_source=claude_desktop) (Christodoulou et al., 2022, PLoS Computational Biology, 15 citations, DOI: 10.1371/journal.pcbi.1010365)

[46] [Non-normal dynamics on non-reciprocal networks: Reactivity and effective dimensionality in neural circuits](https://consensus.app/papers/details/f1863be5a57654098d51443cfd0ed886/?utm_source=claude_desktop) (Poggialini et al., 2025, Physical Review E, 0 citations, DOI: 10.1103/jv6l-3s5z)

[47] [A frozen rate operator from the complete larval connectome: degree and weight govern the gross response, exact wiring governs input routing and mushroom-body modes](https://consensus.app/papers/details/c9caa8cc1cda539585703f5a2e4c69bb/?utm_source=claude_desktop) (Therianos, 2026, 0 citations, DOI: 10.48550/arxiv.2606.17745)

[48] [Linear Dynamics & Control of Brain Networks](https://consensus.app/papers/details/8e3bef91fc2b503ca9545e47e8d0497b/?utm_source=claude_desktop) (Kim & Bassett, 2019, 10 citations, DOI: 10.48550/arxiv.1902.03309)

[49] [A network control theory pipeline for studying the dynamics of the structural connectome](https://consensus.app/papers/details/3ec385e7c3bf543c8e50ff8058e2154a/?utm_source=claude_desktop) (Parkes et al., 2024, Nature Protocols, 48 citations, DOI: 10.1038/s41596-024-01023-w)

[50] [Connectome-based reservoir computing with the conn2res toolbox](https://consensus.app/papers/details/4d3dabb7f4da502a8cac7c254bfd95bd/?utm_source=claude_desktop) (Suarez et al., 2024, Nature Communications, 55 citations, DOI: 10.1038/s41467-024-44900-4)

[51] [Reaction-diffusion models in weighted and directed connectomes](https://consensus.app/papers/details/f78b02938ea751ada2b121c955d940d7/?utm_source=claude_desktop) (Schmitt et al., 2022, PLOS Computational Biology, 9 citations, DOI: 10.1371/journal.pcbi.1010507)

[52] [Systematic perturbation of an artificial neural network: A step towards quantifying causal contributions in the brain](https://consensus.app/papers/details/a375aacc94e354fca415c6cd06ae55ad/?utm_source=claude_desktop) (Fakhar et al., 2021, PLoS Computational Biology, 21 citations, DOI: 10.1371/journal.pcbi.1010250)

[53] [A general framework for characterizing optimal communication in brain networks](https://consensus.app/papers/details/ed11866788af58828da207381c2c5346/?utm_source=claude_desktop) (Fakhar et al., 2025, eLife, 9 citations, DOI: 10.1101/2024.06.12.598676)

[54] [Flow-Based Network Analysis of the Caenorhabditis elegans Connectome](https://consensus.app/papers/details/65d4649de6fd54519f491eb5c10ce621/?utm_source=claude_desktop) (Bacik et al., 2015, PLoS Computational Biology, 67 citations, DOI: 10.1371/journal.pcbi.1005055)

[55] [Brain networks under attack: robustness properties and the impact of lesions](https://consensus.app/papers/details/d0a0bb7b5d8352e387212ca65e63b073/?utm_source=claude_desktop) (Aerts et al., 2016, Brain, 290 citations, DOI: 10.1093/brain/aww194)

[56] [A connectome manipulation framework for the systematic and reproducible study of structure-function relationships through simulations](https://consensus.app/papers/details/7eff10a2bf5c5558bac60aa1d333610d/?utm_source=claude_desktop) (Pokorny et al., 2024, Network Neuroscience, 5 citations, DOI: 10.1101/2024.05.24.593860)

[57] [Chemogenetic Interrogation of a Brain-wide Fear Memory Network in Mice](https://consensus.app/papers/details/dd0e6344e1e85052bd3c503ad9328263/?utm_source=claude_desktop) (Vetere et al., 2017, Neuron, 254 citations, DOI: 10.1016/j.neuron.2017.03.037)

[58] [Functional Implications of Dale's Law in Balanced Neuronal Network Dynamics and Decision Making](https://consensus.app/papers/details/3fc9235413dc530da7eb2ded5209e636/?utm_source=claude_desktop) (Barranca et al., 2022, Frontiers in Neuroscience, 17 citations, DOI: 10.3389/fnins.2022.801847)

[59] [Effect of sparsity on network stability in random neural networks obeying Dale's law](https://consensus.app/papers/details/3f8c6936bd475b578e9e473f57e0587f/?utm_source=claude_desktop) (Harris et al., 2023, Physical Review Research, 11 citations, DOI: 10.1103/physrevresearch.5.043132)

[60] [Consequences of Dale's law on the stability-complexity relationship of random neural networks](https://consensus.app/papers/details/f75c38765a9b533bb629dc78eee639b3/?utm_source=claude_desktop) (Ipsen et al., 2019, Physical Review E, 15 citations, DOI: 10.1103/physreve.101.052412)

[61] [Hippocampome.org 2.0 is a knowledge base enabling data-driven spiking neural network simulations of rodent hippocampal circuits](https://consensus.app/papers/details/8116bd7db13d573b83c7cf82081f840d/?utm_source=claude_desktop) (Wheeler et al., 2024, eLife, 17 citations, DOI: 10.7554/elife.90597)

[62] [Hippocampome.org: a knowledge base of neuron types in the rodent hippocampus](https://consensus.app/papers/details/ff151c72039f506688bae4dddfd1ec52/?utm_source=claude_desktop) (Wheeler et al., 2015, eLife, 145 citations, DOI: 10.7554/elife.09960)

[63] [Comprehensive Estimates of Potential Synaptic Connections in Local Circuits of the Rodent Hippocampal Formation by Axonal-Dendritic Overlap](https://consensus.app/papers/details/788c6e41f7725bd2a13a3b25c28aa42b/?utm_source=claude_desktop) (Tecuatl et al., 2020, The Journal of Neuroscience, 29 citations, DOI: 10.1523/jneurosci.1193-20.2020)

[64] [Graph Theoretic and Motif Analyses of the Hippocampal Neuron Type Potential Connectome](https://consensus.app/papers/details/e5fce60d44475042bd873f084fe17780/?utm_source=claude_desktop) (Rees et al., 2016, eNeuro, 25 citations, DOI: 10.1523/eneuro.0205-16.2016)

[65] [Weighing the Evidence in Peters' Rule: Does Neuronal Morphology Predict Connectivity?](https://consensus.app/papers/details/6ebee22b07d25cdba9f9cbfcfc774109/?utm_source=claude_desktop) (Rees et al., 2016, Trends in Neurosciences, 119 citations, DOI: 10.1016/j.tins.2016.11.007)

[66] [A comprehensive knowledge base of synaptic electrophysiology in the rodent hippocampal formation](https://consensus.app/papers/details/705aee39fd9257a6814a982c9408a6ff/?utm_source=claude_desktop) (Moradi et al., 2019, Hippocampus, 23 citations, DOI: 10.1002/hipo.23148)

[67] [Quantification of neuron types in the rodent hippocampal formation by data mining and numerical optimization](https://consensus.app/papers/details/9291367020a951b197b049b5c6160ed5/?utm_source=claude_desktop) (Attili et al., 2021, European Journal of Neuroscience, 20 citations, DOI: 10.1111/ejn.15639)

[68] [Parvalbumin+ interneurons obey unique connectivity rules and establish a powerful lateral-inhibition microcircuit in dentate gyrus](https://consensus.app/papers/details/a988f7bd795f58929d68ddb5b75a386b/?utm_source=claude_desktop) (Espinoza et al., 2018, Nature Communications, 149 citations, DOI: 10.1038/s41467-018-06899-3)

[69] [Parvalbumin interneuron mediated feedforward inhibition controls signal output in the deep layers of the perirhinal-entorhinal cortex](https://consensus.app/papers/details/fdcfc4a273035998b6ee5e36838250ee/?utm_source=claude_desktop) (Willems et al., 2018, Hippocampus, 25 citations, DOI: 10.1002/hipo.22830)

[70] [Neurogliaform cells mediate feedback inhibition in the medial entorhinal cortex](https://consensus.app/papers/details/d4ccd751a11d515087a96d31304d7333/?utm_source=claude_desktop) (Szocs et al., 2022, Frontiers in Neuroanatomy, 1 citation, DOI: 10.3389/fnana.2022.779390)

[71] [Parvalbumin Interneurons Are Differentially Connected to Principal Cells in Inhibitory Feedback Microcircuits along the Dorsoventral Axis of the Medial Entorhinal Cortex](https://consensus.app/papers/details/44f3d17034ce52968d274faf4979b511/?utm_source=claude_desktop) (Grosser et al., 2021, eNeuro, 21 citations, DOI: 10.1523/eneuro.0354-20.2020)

[72] [Specific inhibitory synapses shift the balance from feedforward to feedback inhibition of hippocampal CA1 pyramidal cells](https://consensus.app/papers/details/cef5a072588f589c838cdb6feac2249c/?utm_source=claude_desktop) (Elfant et al., 2007, European Journal of Neuroscience, 77 citations, DOI: 10.1111/j.1460-9568.2007.06001.x)

[73] [Hippocampus shapes entorhinal cortical output through a direct feedback circuit](https://consensus.app/papers/details/c4750a26ca2a5ae39d742ea0110593e5/?utm_source=claude_desktop) (Butola et al., 2025, Nature Neuroscience, 12 citations, DOI: 10.1038/s41593-025-01883-9)

[74] [Hippocampome.org, a resource for subicular neuron types and beyond](https://consensus.app/papers/details/943c6d51adec5ab18800c4e9a0b2397d/?utm_source=claude_desktop) (Tecuatl et al., 2025, bioRxiv, 1 citation, DOI: 10.64898/2025.12.30.697062)

[75] [Small-World Brain Networks Revisited](https://consensus.app/papers/details/b88540b39cb255edb1f8d6fb2c2e9307/?utm_source=claude_desktop) (Bassett et al., 2016, The Neuroscientist, 809 citations, DOI: 10.1177/1073858416667720)

[76] [Redefining the connectome: A multi-modal, asymmetric, weighted, and signed description of anatomical connectivity](https://consensus.app/papers/details/83d3cc280699549698b310883f1cca5e/?utm_source=claude_desktop) (Tanner et al., 2022, bioRxiv, 4 citations, DOI: 10.1101/2022.12.19.519033)

[77] [Ecological networks: Pursuing the shortest path, however narrow and crooked](https://consensus.app/papers/details/0c775f74c6175c0ab11d451abc2fa2d0/?utm_source=claude_desktop) (Costa et al., 2018, Scientific Reports, 21 citations, DOI: 10.1038/s41598-019-54206-x)

[78] [Strong intercorrelations among global graph-theoretic indices of structural connectivity in the human brain](https://consensus.app/papers/details/8940d6dd1a2958c4b0c77c24c1be4174/?utm_source=claude_desktop) (Madole et al., 2023, NeuroImage, 18 citations, DOI: 10.1016/j.neuroimage.2023.120160)

[79] [Investigation of Brain Network Sparsification Techniques and Their Impact on Network Topology](https://consensus.app/papers/details/9b6f9bf149ae5651946294dbdf68682f/?utm_source=claude_desktop) (Rohilla et al., 2025, IEEE TEMSMET, 0 citations, DOI: 10.1109/temsmet65536.2025.11467176)

### Additional references cited by name in the text

- [Machine learning augmented branch and bound for mixed integer linear programming](https://consensus.app/papers/details/658d14621026588e81077b1964cb1079/?utm_source=claude_desktop) (Scavuzzo et al., 2024, Mathematical Programming, 71 citations, DOI: 10.1007/s10107-024-02130-y)
- [Signal propagation in complex networks](https://consensus.app/papers/details/ef3eedb5080a54b4824ec7cb159dd29f/?utm_source=claude_desktop) (Ji et al., 2023, Physics Reports, 337 citations, DOI: 10.1016/j.physrep.2023.03.005)
- [Excitation-Inhibition Balance, Neural Criticality, and Activities in Neuronal Circuits](https://consensus.app/papers/details/ddef50f67c625285842dc04c785d08eb/?utm_source=claude_desktop) (Liang et al., 2024, The Neuroscientist, 40 citations, DOI: 10.1177/10738584231221766)
- [Inferring neural signalling directionality from undirected structural connectomes](https://consensus.app/papers/details/e53df6a1cf7d5aefb3321fb45431a34b/?utm_source=claude_desktop) (Seguin et al., 2019, Nature Communications, 95 citations, DOI: 10.1038/s41467-019-12201-w)
- [Effects of virtual lesions on temporal dynamics in cortical networks based on personalized dynamic models](https://consensus.app/papers/details/0baf87f0125a556eb569eca150fbe7d8/?utm_source=claude_desktop) (Wei et al., 2022, NeuroImage, 13 citations, DOI: 10.1016/j.neuroimage.2022.119087)
- [Cortical Hierarchy, Dual Counterstream Architecture and The Importance of Top-Down Generative Networks](https://consensus.app/papers/details/4ce23fdeaa8a57b5b72448807bf67a82/?utm_source=claude_desktop) (Vezoli et al., 2020, NeuroImage, 98 citations, DOI: 10.1101/2020.04.08.032706)
- [The importance of being hierarchical](https://consensus.app/papers/details/d9c873900c9750c8acb781aa147e1007/?utm_source=claude_desktop) (Markov et al., 2013, Current Opinion in Neurobiology, 180 citations, DOI: 10.1016/j.conb.2012.12.008)
- [Hierarchical organization of cortical and thalamic connectivity](https://consensus.app/papers/details/186375ffef815f288426dc05abe4b098/?utm_source=claude_desktop) (Harris et al., 2019, Nature, 643 citations, DOI: 10.1038/s41586-019-1716-z)
- [Accurate and Efficient Simulation of Very High-Dimensional Neural Mass Models with Distributed-Delay Connectome Tensors](https://consensus.app/papers/details/093773e635b35b25a39b3d64ce61798e/?utm_source=claude_desktop) (Mitjans et al., 2020, NeuroImage, 7 citations, DOI: 10.1016/j.neuroimage.2023.120137)

---

*Verification code for findings 1.1-1.4 is reproducible from the matrices in `matrices/`;
each numerical claim above was recomputed rather than read from existing outputs.*
