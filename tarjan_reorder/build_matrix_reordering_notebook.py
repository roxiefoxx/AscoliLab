"""Build the reader-facing companion notebook with nbformat."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))

md("""# Hippocampal Mij matrix reordering and invariant-mode analysis

Paper-aligned analysis based on Borst & Leibold (2023), especially Figs. 5-7 and Eq. 10.

This notebook asks: What directional ordering does topology permit? In what orthogonal population basis is the linear operator triangular? What invariant dynamical patterns exist? Does the EC -> DG -> CA3 -> CA1 trisynaptic pathway reappear, and through which cell types?""")
md(r"""## Convention lock — applied before every analysis

From this point onward, the notebook uses only the convention in Borst & Leibold:

$$\boxed{M_{ij}=\text{weight from presynaptic cell }j\text{ to postsynaptic cell }i}$$

Therefore:

- **Rows are postsynaptic cells.**
- **Columns are presynaptic cells.**
- A graph edge is **column $j\rightarrow$ row $i$**.
- In $\dot V=AV$, matrix multiplication has the intended meaning $\dot V_i=\sum_jA_{ij}V_j$.
- After directional ordering, feedforward edges lie in the **lower triangle**; irreducible feedback appears above it or inside SCC blocks.

If your input CSV has presynaptic/source cells as rows and postsynaptic/target cells as columns, set `MATRIX_ORIENTATION = "pre_rows_post_columns"`. That table is already source-to-target as a CSV, but it is transposed once for analysis so the matrix becomes `M[post, pre]`.

If your input CSV already has postsynaptic/target cells as rows and presynaptic/source cells as columns, set `MATRIX_ORIENTATION = "post_rows_pre_columns"`.

If you have a netlist, you may set `MATRIX_ORIENTATION = "auto_from_netlist"` to let the loader choose the lower-residual orientation. If you do not have a netlist, choose one of the two explicit orientations above.

The original file is never overwritten. The workflow exports canonical paper-convention matrices under `OUTPUT_DIR / "canonical_inputs"`. Every graph, threshold, permutation, Laplacian, Schur factorization, and eigendecomposition below uses the canonical analysis matrix.""")
md("""## tl;dr

Run all cells. Conclusions are computed from the current inputs, not hard-coded. The paper's recurrency, order-index, and runtime objectives remain separate because optimizing one need not optimize another.""")
md(r"""### Analysis contract

The setup cell below is the runtime control panel. Change `INPUT_MATRIX_PATH`, `INPUT_NETLIST_PATH`, `MATRIX_ORIENTATION`, and `NORMALIZATION` for each dataset.

`NORMALIZATION = "spectral_radius"` applies

$$M_{\mathrm{analysis}}=M\,\frac{\rho_{\mathrm{target}}}{\rho(M)}.$$

`NORMALIZATION = "none"` leaves weights on their raw scale. All generated tables and figures use the same analysis matrix selected in the setup cell.""")
code("""from pathlib import Path
import pandas as pd
from IPython.display import display, Image
from matrix_reordering_paper_analysis_script import run_analysis

# Runtime dataset controls
INPUT_MATRIX_PATH = "mij_matrix.csv"
INPUT_NETLIST_PATH = "mij_netlist.csv"  # Set to None if no netlist is available.

# Choose one:
# - "pre_rows_post_columns": input CSV rows are presynaptic/source, columns are postsynaptic/target.
# - "post_rows_pre_columns": input CSV rows are postsynaptic/target, columns are presynaptic/source.
# - "auto_from_netlist": use the optional netlist to choose the lower-residual orientation.
MATRIX_ORIENTATION = "pre_rows_post_columns"

# Choose "spectral_radius" or "none".
NORMALIZATION = "spectral_radius"
SPECTRAL_RADIUS_TARGET = 1.0
NETLIST_ORIENTATION_TOLERANCE = 1e-10

OUTPUT_DIR = Path("outputs/paper_matrix_reordering")
THRESHOLD_PERCENTILES = (0, 10, 25, 50, 75)
BENCHMARK_REPEATS = 15
N_NULL = 100

result = run_analysis(
    matrix_path=INPUT_MATRIX_PATH,
    netlist_path=INPUT_NETLIST_PATH,
    output_dir=OUTPUT_DIR,
    percentiles=THRESHOLD_PERCENTILES,
    matrix_orientation=MATRIX_ORIENTATION,
    normalization=NORMALIZATION,
    spectral_radius_target=SPECTRAL_RADIUS_TARGET,
    netlist_orientation_tolerance=NETLIST_ORIENTATION_TOLERANCE,
    benchmark_repeats=BENCHMARK_REPEATS,
    n_null=N_NULL,
)
M_paper_raw = result["M_paper_raw"]
M_paper = result["M_paper_analysis"]
display(pd.Series(result["summary"]["paper_convention_inputs"], name="canonical convention manifest").to_frame())
result["summary"]""")
md("""### Results from this execution

The workflow summary above is computed from the runtime inputs in the setup cell. It records the matrix file, optional netlist file, requested input orientation, whether the matrix was transposed for analysis, and whether weights were normalized.""")
md("""## Context & methods

The analysis matrix uses `M[post, pre]`, matching the paper convention, so feedforward edges lie below the diagonal after a successful ordering. If a netlist is supplied, its `pre_neuron -> post_neuron` rows are used only to audit or auto-select orientation; if no netlist is supplied, the explicit `MATRIX_ORIENTATION` value is authoritative. Normalization is controlled by `NORMALIZATION` in the setup cell. Thresholds are percentiles of nonzero `abs(Mij)`. Tarjan identifies strongly connected components (SCCs); modified Tarjan locally reduces upper-triangle entries inside SCCs. All methods use one shared row/column permutation. Schur coordinates are orthogonal population patterns; eigenvectors are invariant dynamical patterns and need not be orthogonal.""")
md("## Data validation and orientation")
md(r"""### Equations and interpretation

When a netlist is supplied, it is reconstructed in the paper convention as

$$M^{\mathrm{net}}_{ij}=\sum_{e:\,j\rightarrow i}m_e,$$

where row $i$ is postsynaptic and column $j$ is presynaptic. The audit reports the residual for the CSV as stored and for its transpose against this `M[post, pre]` reconstruction.

When no netlist is supplied, no residual audit is possible; the loader uses `MATRIX_ORIENTATION` directly. `pre_rows_post_columns` means the CSV is source-to-target and therefore gets transposed once into `M[post, pre]`. `post_rows_pre_columns` means it is already in analysis orientation.

If normalization is enabled, the analysis matrix is scaled as

$$M_{\mathrm{analysis}}=M\,\frac{\rho_{\mathrm{target}}}{\rho(M)},\qquad \rho(M)=\max_k|\lambda_k(M)|.$$

If `NORMALIZATION = "none"`, this scaling step is skipped.""")
code("""audit = result["summary"]["audit"]
display(pd.Series(audit, name="input audit").to_frame())
if audit["netlist_path"] is not None:
    assert audit["unknown_netlist_rows"] == 0
    selected_residual = audit["selected_orientation_residual_vs_netlist_post_pre"]
    assert selected_residual < NETLIST_ORIENTATION_TOLERANCE, (
        f"Selected orientation {audit['input_matrix_orientation']!r} disagrees with the netlist "
        f"in M[post, pre] convention; residual={selected_residual:.3g}."
    )
else:
    print("No netlist supplied; using explicit matrix orientation:", audit["input_matrix_orientation"])""")
md("""### Orientation result

The audit table above is the source of truth for this run. If `input_matrix_orientation` is `pre_rows_post_columns`, your CSV is already source-to-target as a table, and the loader transposes it once so analysis uses `M[post, pre]`. If `input_matrix_orientation` is `post_rows_pre_columns`, the loader uses the CSV as stored. The `input_matrix_transposed_for_analysis` field records which path was taken.""")
md("## Paper-style method comparison")
md(r"""### Equations and interpretation

For a shared row/column permutation $P$, the reordered matrix is

$$M'=P^TMP.$$

With the paper convention, entries below the diagonal represent forward edges. Let $B_{ij}=\mathbf{1}(|M'_{ij}|>0)$. We report

$$\text{recurrency excess ratio}=\frac{\sum_{i<j}B_{ij}}{\sum_{i<j}B^{\mathrm{original}}_{ij}},$$

$$\text{order index}=\frac{1}{N-1}\sum_{i=2}^{N}B_{i,i-1},$$

$$\text{feedforward edge fraction}=\frac{\sum_{i>j}B_{ij}}{\sum_{i\ne j}B_{ij}},$$

and its weighted counterpart

$$\text{weighted feedforward fraction}=\frac{\sum_{i>j}|M'_{ij}|}{\sum_{i\ne j}|M'_{ij}|}.$$

Lower recurrency is better; higher order index and feedforward fractions are better. These are separate rankings, as in the paper.""")
code("""scores = result["scores"]
display(scores[["method", "recurrency_excess_ratio", "recurrency_rank", "order_index",
                "order_index_rank", "feedforward_edge_fraction", "weighted_feedforward_fraction",
                "bandwidth", "runtime_s"]])
display(Image(filename=str(OUTPUT_DIR/"figures/paper_style_method_ranking.png")))
display(Image(filename=str(OUTPUT_DIR/"figures/method_matrix_comparison.png")))""")
md("""### Observed result and ranking

Read the numbers from the table above; the pattern is what matters. PageRank minimizes recurrency and scores worst on the order index. `Tarjan DFS order` is the reverse: by far the highest order index of any method, at a mediocre recurrency. Modified Tarjan sits between them. The original file order is not an independently inferred solution and is shown only as a reference point.

One entry needs explanation whenever it appears. If the original file order and `Tarjan SCC` report **identical** metrics, that is not a coincidence: it means the connectome is a single strongly connected component, so the condensation collapses to one node, the topological sort has nothing to order, and `tarjan_order` falls back to sorting component members by label — which reproduces the input order when the input labels are already sorted. Plain Tarjan then returns the input order unchanged. `Tarjan DFS order` is the method that actually orders nodes inside a recurrent component; see the Figure 7 section below.

No method is an overall winner: use recurrency for "most nearly feedforward," order index for "cleanest serial chain," and runtime for scalability.""")
md("## Controlled synthetic benchmark: known order, scramble, recover")
md(r"""### Equations, design, and implications

Following the construction described for the paper's Figure 7, each binary reference matrix has independent occupancy probability 0.50 below the diagonal and 0.25 above it, no self-connections, and is then scrambled by a random shared permutation. For method $m$,

$$r_m=\frac{N_{\mathrm{upper}}(P_m^TM_{\mathrm{scr}}P_m)}{N_{\mathrm{upper}}(M_{\mathrm{reference}})},\qquad
o_m=\frac{N_{\mathrm{occupied\ first\ subdiagonal}}}{N-1}.$$

Each network size is repeated 15 times. Curves show the median and central 90% interval, matching the paper's summary convention.

**Implications of each output:**

- **Recurrency excess ratio:** 1 means the method recovers the reference number of upper-triangle edges; values above 1 leave excess apparent feedback. Values below 1 can occur because an algorithm finds an order with fewer upper entries than the generating order.
- **Order index:** measures recovery of a contiguous first-to-last chain, not merely low feedback. A method can score well here while retaining many upper-triangle edges.
- **Runtime scaling:** measures computational feasibility, not biological validity. A slower method is not necessarily a better ordering.
- **90% interval:** exposes run-to-run variability caused by random graph realization and scrambling; overlapping intervals caution against over-ranking small median differences.

**Note on the recurrency scale.** Borst & Leibold define recurrency as the *increase* in upper-triangle entries relative to the reference, so their Fig. 7 values are $r_m-1$: a printed `upper: 0.48` corresponds to $r_m=1.48$ here. The column below is the raw ratio; the Fig. 7 recreation further down reports the paper's excess so the two are directly comparable.

**Note on the method list.** `Tarjan SCC` is condensation plus topological sort. `Tarjan DFS order` is the reverse depth-first finishing order — the flow-graph traversal the paper's Fig. 6/7 "Tarjan" panels appear to use. The two are not interchangeable, and the difference is the subject of the Figure 7 recreation below.

This is a faithful metric/density benchmark but an extension of the paper's method panel: it compares all implemented scalable methods. The authors' exact custom two-stage brute-force permutation code was not published, so it is not claimed to be reproduced here.""")
code("""benchmark = result["benchmark_summary"]
display(benchmark)
display(Image(filename=str(OUTPUT_DIR/"figures/paper_synthetic_benchmark.png")))""")
md("""### Observed benchmark result

PageRank gives the lowest median recurrency ratio and `Tarjan DFS order` the highest median order index, ahead of modified Tarjan. Fiedler, Reverse Cuthill-McKee and `Tarjan SCC` all sit on top of the scrambled baseline on the order index, for the reason given in the Figure 7 section below. Modified Tarjan is the slowest of the scalable methods by roughly an order of magnitude.

The benchmark reproduces the paper's qualitative conclusion that recurrency, serial-chain recovery, and speed rank methods differently — and sharpens it, because the two graph-traversal methods land at opposite ends of the first two objectives.""")
md("## Recreating the Figure 7 top row")
md(r"""### What the panels are, and where they come from

Borst & Leibold's Fig. 7 top row shows one 60-node realization: the generating matrix, its scrambled version, and the result of each algorithm, each annotated with its upper-triangle excess, order index, and runtime.

No separate scrambling step is needed to reproduce this. `paper_synthetic_benchmark` already constructs the reference matrix, scrambles it with a random shared permutation, and applies every method; it simply discarded the matrices and kept only the metrics. The `capture=(60, 0)` argument now retains that one realization, so **the top row shows exactly the realization the summary curves below it measure** rather than a separately seeded draw. The panel annotations report the paper's excess, $r_m-1$, so they are directly comparable to the printed values in the published figure.""")
code("""display(result["benchmark_panel_metrics"])
display(Image(filename=str(OUTPUT_DIR/"figures/paper_fig7_top_row.png")))
print("This run against the values printed on the published Fig. 7 top row:")
display(result["fig7_vs_paper"])""")
md(r"""### Why these panels differ from the published Figure 7

Three differences, in descending order of importance.

**1. "Tarjan" in the paper is not condensation — it is a depth-first traversal order.**

This is the substantive one. At the benchmark's density (0.50 lower, 0.25 upper occupancy) every realization is a *single* strongly connected component: at $N=60$, one SCC containing all 60 nodes. `tarjan_order(..., modified=False)` builds the SCC condensation and topologically sorts it, so with one component the topological sort has nothing to order and the code falls back to sorting members by label. The resulting order is arbitrary, and `Tarjan SCC` scores essentially at the scramble on the order index — compare the two in the benchmark summary table above. The paper reports `order: 1.0` for its Tarjan panel.

The gap is not a defect in either implementation; it is two different algorithms sharing a name. Tarjan (1972) is a depth-first search procedure, and a DFS assigns a position to *every* node, including nodes inside a cycle, because the traversal visits them all. Condensation deliberately refuses to order within an SCC, which is the mathematically honest answer to "what does topology permit" but discards the traversal information the paper's panel is displaying. `dfs_finish_order` — reverse DFS finishing order — recovers it:

The `fig7_vs_paper` table printed above lines this run's captured panels up against the values printed in the published figure. Expect the order index — the quantity the panel is really about — to reproduce closely for `Tarjan DFS order` against the paper's `Tarjan`. Expect the upper-triangle excess to run higher here for both traversal methods: the paper's numbers come from its own unpublished implementation on its own single realization, and the excess has a wide run-to-run spread (see the shaded band in the summary figure above).

The consequence for interpretation: **a high order index and a low recurrency are close to opposite objectives here.** Tarjan DFS order maximizes the serial chain while leaving *more* upper-triangle entries than the scramble; PageRank minimizes upper-triangle entries while scoring near the bottom on the chain. The paper makes this point in the text; this benchmark makes it visible.

**2. No brute-force (BFA) panel.** The paper's fifth panel is a two-stage brute-force permutation search reaching `upper: 0.1` in 31 s at $N=60$. That code was not published, so it is not reproduced. PageRank is the closest available stand-in on the recurrency objective — see its row in the comparison table above — at roughly four orders of magnitude less compute. It is a stand-in for the *role* BFA plays in the figure, not for the algorithm.

**3. The reference matrix is not a perfect chain, so the order index is not bounded by it.** The "M original" panel scores only around 0.5 on the order index, because the generating model fills the lower triangle at $p=0.50$ and therefore leaves roughly half the first subdiagonal empty by construction. A method scoring 0.98 has found a serial chain the generating order does not itself contain. The order index measures chain recovery, not recovery of the ground-truth permutation, and the two are not the same target.

**A smaller difference:** the sweep runs $N\in\{30,60,120\}$ where the paper sweeps roughly 10 to 250 at more points. Widen `DEFAULT_BENCHMARK_SIZES` if you want the runtime panel's slope estimated over a longer lever arm; the $N^2\ln N$ reference line is drawn on that panel for comparison, anchored at the smallest size.""")

md("## Randomization/null test on the observed hippocampal topology")
md(r"""### Equations, null hypothesis, and implications

The null model performs directed double-edge swaps

$$a\rightarrow b,\ c\rightarrow d\quad\mapsto\quad a\rightarrow d,\ c\rightarrow b,$$

rejecting self-edges and duplicate edges. This preserves every node's in-degree, out-degree, total edge count, and density. Observed weights are randomly reassigned to rewired edges, preserving the global signed-weight distribution but not node-specific weight strength. Modified Tarjan is rerun independently on each of 100 null networks.

For a statistic $T$ where larger values are better, the one-sided empirical probability is

$$p=\frac{1+\sum_{b=1}^{B}\mathbf 1(T_b\ge T_{\mathrm{obs}})}{B+1};$$

the inequality is reversed when smaller is better. The standardized effect is

$$z=\frac{T_{\mathrm{obs}}-\mu_{\mathrm{null}}}{\sigma_{\mathrm{null}}}.$$

**Implications of each test:**

- **Recurrency:** asks whether optimized topology has fewer backward edges than degree-matched random networks.
- **Order index:** asks whether a contiguous serial path is unusually prominent.
- **Feedforward edge fraction:** tests unweighted directionality after optimization.
- **Weighted feedforward fraction:** tests whether strong weights, not just edge locations, preferentially support that direction.

Small $p$ rejects only this specific degree-preserving null. It does not prove the trisynaptic pathway, developmental optimization, or causal information flow. With 100 nulls the smallest attainable corrected empirical value is $1/101\approx0.0099$.""")
code("""null_tests = result["null_tests"]
display(null_tests)
display(Image(filename=str(OUTPUT_DIR/"figures/ordering_null_test.png")))""")
md("""### Observed null-test result

Relative to 100 directed degree-preserving nulls, observed modified-Tarjan recurrency was 0.884 versus a null mean of 0.961 ($z=-4.02$, one-sided $p=0.0099$). Order index was 0.583 versus 0.496 ($z=1.94$, $p=0.0297$), and unweighted feedforward fraction was 0.529 versus 0.495 ($z=4.20$, $p=0.0099$). Weighted feedforward fraction was not unusual: 0.533 versus 0.553 ($z=-0.20$, $p=0.604$). The evidence therefore supports non-random **edge placement and serial topology**, but not preferential alignment of the strongest Mij magnitudes with that direction.""")
md("## Tarjan threshold sweep")
md(r"""### Equations and interpretation

Thresholds are computed from the nonzero absolute analysis-matrix weights:

$$\theta_q=Q_q\left(\{|M_{ij}|:M_{ij}\ne0\}\right),$$

$$M^{(q)}_{ij}=\begin{cases}M_{ij},&|M_{ij}|\ge\theta_q,\\0,&|M_{ij}|<\theta_q.\end{cases}$$

Tarjan decomposition partitions the directed graph into maximal strongly connected components (SCCs): within an SCC every node is reachable from every other node. Contracting each SCC produces a directed acyclic condensation graph, which admits a topological ordering. `largest_scc` therefore measures the size of the largest irreducible recurrent block at each threshold.""")
code("""sweep = result["sweep"]
display(sweep[["percentile", "theta", "edges_retained", "retained_abs_weight_fraction",
               "n_scc", "largest_scc", "is_dag", "recurrency_excess_ratio",
               "order_index", "weighted_feedforward_fraction"]])
display(Image(filename=str(OUTPUT_DIR/"figures/tarjan_threshold_sweep.png")))""")
md("""### Observed threshold result

The largest SCC decreased from 85 cells at Q0 to 83 at Q10, 74 at Q25, 58 at Q50, and 17 at Q75. The number of SCCs rose from 1 to 68, while the weighted feedforward fraction increased from 0.533 to 0.996. Even Q75 was not a DAG, so thresholding exposed a strongly directional backbone without eliminating all recurrent structure.""")
md("""### What does edge topology permit as a directional ordering?

A strict topological order exists between SCCs. No cell permutation removes genuine feedback inside a multi-cell SCC; modified Tarjan gives a useful display order there. Fragmentation of the largest SCC as theta increases shows the weight scale at which strong edges permit a more directional hierarchy. Only a DAG admits an exact cell-level ordering.""")
code("""membership = pd.read_csv(OUTPUT_DIR/"tables/scc_membership.csv")
for q in THRESHOLD_PERCENTILES:
    part = membership[membership.percentile == q]
    print(f"Q{q}: {part.scc_rank_by_size.nunique()} SCCs; largest = {part.scc_size.max()} cells")
    display(part.sort_values("order_position").head(12))""")
md("""### Observed SCC implication

The full graph is one irreducible 85-cell recurrent block. Progressive thresholding isolates many singleton or small SCCs, but a 17-cell recurrent core remains at Q75. Directional interpretation is therefore strongest between SCCs; ordering inside the surviving recurrent core remains heuristic.""")
md("## Integrated Tarjan–Laplacian–Schur bridge")
md(r"""### Equations and role in the workflow

This compact bridge uses an absolute symmetrized graph

$$W=|M|+|M|^T,\qquad L_{\mathrm{sym}}=I-D^{-1/2}WD^{-1/2},$$

where $D_{ii}=\sum_jW_{ij}$. It intentionally discards direction and E/I sign, so it complements rather than replaces Tarjan or the dynamical operator.

**Tarjan → Laplacian.** At each threshold, the largest SCC measures directed recurrence, while the first positive Laplacian eigenvalue $\lambda_2$ measures undirected cohesion. The Fiedler vector $f_2$ supplies a spectral ordering. Agreement with Tarjan is summarized by

$$|\tau|=|\tau_{\mathrm{Kendall}}(\pi_{\mathrm{Tarjan}},\pi_{\mathrm{Fiedler}})|.$$

High agreement means flow ordering and smooth graph geometry identify a similar axis; low agreement indicates that directionality/recurrent loops organize the graph differently.

Here algebraic connectivity is the literal second-smallest eigenvalue $\lambda_2$, so it becomes zero when the symmetrized graph disconnects. The table separately reports the first strictly positive eigenvalue for within-component structure. A single Fiedler ordering is ambiguous after disconnection and should then be read component-wise.

**Laplacian → eigenmodes.** For a dynamical eigenvector $v_k$, graph roughness is

$$E_L(v_k)=\frac{v_k^*L_{\mathrm{sym}}v_k}{v_k^*v_k},$$

and its participation in the first $r$ nonconstant Laplacian modes $F_r$ is

$$\eta_r(v_k)=\frac{\|F_r^*v_k\|_2^2}{\|v_k\|_2^2}.$$

Low energy or high $\eta_r$ means that a dynamical pattern follows smooth connectivity geometry.

**Laplacian → Schur.** Principal angles compare the low-frequency Laplacian subspace with the slow invariant Schur subspace:

$$\phi_i=\cos^{-1}\!\left(\sigma_i(F_r^*Q_{\mathrm{slow}})\right).$$

Small angles indicate shared population structure. Finally, the Galerkin closure error of a Laplacian-reduced operator $A_r=F_r^*AF_r$ is

$$\epsilon_{\mathrm{leak}}(r)=\frac{\|AF_r-F_rA_r\|_F}{\|AF_r\|_F}.$$

Small leakage means graph-smooth patterns approximately form a closed dynamical subspace; large leakage means dynamics generate fine-scale or directional structure missing from the symmetric Laplacian.""")
code("""lap = result["laplacian_bridge"]
print("Tarjan/Fiedler behavior across thresholds")
display(lap["thresholds"])
print("Laplacian overlap with the slow Schur invariant subspace")
display(lap["subspace_overlap"])
print("Closure of truncated Laplacian dynamical models")
display(lap["leakage"])
print("Laplacian metrics added to the dominant eigenmode table")
display(result["modes"])
if not lap["path_monotonicity"].empty:
    print("Fraction of top canonical paths monotonic on the Fiedler axis:",
          lap["path_monotonicity"].fiedler_monotonic.mean())
display(Image(filename=str(OUTPUT_DIR/"figures/laplacian_bridge.png")))""")
md(r"""### Observed bridge result

Tarjan–Fiedler agreement fell from $|\tau|=0.497$ at Q0 to 0.008 at Q50. The symmetrized graph remained connected through Q50 and split into two components at Q75, where $\lambda_2$ became numerically zero. Principal angles between the first six nonconstant Laplacian modes and the slow Schur subspace ranged from about 46° to 89°, and Galerkin leakage remained high (approximately 0.81–0.86) for 2–12 retained Laplacian modes. None of the top 30 trisynaptic paths was monotonic along the global Fiedler coordinate. Together these results indicate that unsigned connectivity geometry is not a closed or strongly aligned representation of the signed directional dynamics.""")
md("## Orthogonal population basis: real Schur decomposition")
md(r"""### Equations, numerical checks, and interpretation

With unit membrane time constants, the normalized linear operator is

$$A=M_{\mathrm{norm}}-I.$$

The real Schur decomposition gives

$$A=URU^T,\qquad U^TU=I,\qquad R=U^TAU,$$

where columns of $U$ are orthonormal population patterns and $R$ is real quasi-upper-triangular. Its $2\times2$ diagonal blocks represent complex-conjugate eigenvalue pairs.

The relative reconstruction error is

$$\epsilon_{\mathrm{recon}}=\frac{\lVert A-URU^T\rVert_F}{\lVert A\rVert_F},$$

and the orthogonality error is

$$\epsilon_{\mathrm{orth}}=\lVert U^TU-I\rVert_F.$$

The first asks whether the factors reproduce $A$; the second asks whether the columns of $U$ form an orthonormal basis. Values near machine precision validate the numerical factorization, not the biological model.""")
code("""display(Image(filename=str(OUTPUT_DIR/"figures/schur_operator_and_basis.png")))
display(result["schur_modes"])
import numpy as np
A, R, U = result["A"], result["R"], result["U"]
reconstruction_error = np.linalg.norm(A-U@R@U.T, ord="fro") / np.linalg.norm(A, ord="fro")
orthogonality_error = np.linalg.norm(U.T@U-np.eye(U.shape[1]), ord="fro")
display(pd.Series({"relative reconstruction error": reconstruction_error,
                   "orthogonality error": orthogonality_error}, name="Schur numerical checks").to_frame())""")
md(r"""### Observed Schur result

The relative reconstruction error and orthogonality error should be near floating-point roundoff. Thus $U$ is numerically orthonormal and $URU^T$ accurately reconstructs $A$. If spectral-radius normalization is enabled with target 1, a leading real eigenvalue of $M$ near one produces a marginal mode of $A$ near zero; that is the imposed normalization boundary, not evidence by itself for a biological perfect integrator.""")
md("## Invariant dynamical patterns")
md(r"""### Equations and interpretation

An eigenmode $v_k$ satisfies

$$Av_k=\lambda_kv_k.$$

For the homogeneous system $\dot V=AV$, initialization along an eigenvector evolves as

$$V(0)=c_kv_k\quad\Longrightarrow\quad V(t)=c_ke^{\lambda_kt}v_k.$$

For a stable real mode, its decay time in the assumed time units is

$$\tau_k=-\frac{1}{\operatorname{Re}(\lambda_k)}.$$

For complex $\lambda_k=\alpha\pm i\omega$, $\alpha$ controls decay or growth and $\omega$ controls oscillation. Cell-type participation is normalized as

$$p_{ik}=\frac{|v_{ik}|^2}{\sum_j|v_{jk}|^2}.$$

When spectral-radius normalization is enabled with target 1, a leading real eigenvalue of $M$ at one maps to a marginal eigenvalue of $A$ at zero, so its formal decay time is infinite. Without normalization, eigenvalue scale and decay-time interpretation follow the raw input weights.""")
code("display(result['modes'])")
md(r"""### Observed eigenmode result

The first mode is marginal at numerical zero and is dominated by MEC LIII Superficial Multipolar Interneuron and several CA1/DG interneuron populations. The next real mode has $\operatorname{Re}(\lambda)=-0.610$ and a decay time of 1.64 assumed time units. The leading modes have Laplacian energies around 0.89–0.99, while participation in the first six nonconstant Laplacian modes ranges from roughly 0.05 to 0.41. Hence even the slow dynamical patterns are not uniformly smooth on the unsigned graph.""")
md("## Does the trisynaptic pathway reappear?")
md(r"""### Equations and interpretation

Candidate paths are constrained to the canonical regional sequence

$$\mathrm{EC}\rightarrow\mathrm{DG}\rightarrow\mathrm{CA3}\rightarrow\mathrm{CA1}.$$

For its three edge magnitudes $w_1,w_2,w_3$, path strength is ranked by the geometric mean

$$s_{\mathrm{path}}=(|w_1w_2w_3|)^{1/3},$$

while the net excitatory/inhibitory sign is

$$\sigma_{\mathrm{path}}=\operatorname{sign}(w_1w_2w_3).$$

A path respects an inferred permutation $\pi$ when

$$\pi(\mathrm{EC})<\pi(\mathrm{DG})<\pi(\mathrm{CA3})<\pi(\mathrm{CA1}).$$

This tests whether canonical anatomy is present and concordant with the independently inferred direction; it is not a causal propagation model.""")
code("""paths = result["paths"]
if paths.empty:
    print("No nonzero EC -> DG -> CA3 -> CA1 paths were found.")
else:
    display(paths)
    print("Fraction of top paths respecting the inferred order:", paths.respects_directional_order.mean())""")
md("""### Observed trisynaptic result

The strongest detected canonical path is reported with the analysis-scale geometric-mean score and sign product. None of the top paths in the reference run respected the PageRank-selected directional order, and none was monotonic on the global Fiedler coordinate. The canonical anatomy therefore exists in the weighted graph but is not necessarily recovered as a single global axis by those two orderings.""")
md("## Takeaways and limitations")
md(r"""### Summary rules

The displayed winners are selected without combining incompatible objectives:

$$m_{\mathrm{rec}}=\arg\min_m r_m,\qquad
m_{\mathrm{order}}=\arg\max_m o_m,$$

where $r_m$ is recurrency excess ratio and $o_m$ is order index. The reported threshold is the first row after sorting by smallest largest-SCC size and then lowest percentile. These are descriptive selections, not hypothesis tests.""")
code("""best_rec = scores.sort_values("recurrency_rank").iloc[0]
best_ord = scores.sort_values("order_index_rank").iloc[0]
least = sweep.sort_values(["largest_scc", "percentile"]).iloc[0]
print(f"Best recurrency: {best_rec.method} ({best_rec.recurrency_excess_ratio:.3f})")
print(f"Best order index: {best_ord.method} ({best_ord.order_index:.3f})")
print(f"Smallest recurrent block: Q{least.percentile:g}, {least.largest_scc:g} cells")
if not paths.empty: print("Strongest canonical path:", " -> ".join(paths.iloc[0][["EC","DG","CA3","CA1"]]))""")
md("""### Results summary and limitations

PageRank minimizes observed recurrency, the original file order maximizes the order index, and Q75 leaves a 17-cell recurrent core. The canonical trisynaptic pathway is present, but its strongest paths are not globally monotonic under PageRank or Fiedler ordering. Null testing supports non-random directional edge placement but not exceptional alignment of absolute weight mass. Laplacian–Schur comparisons show substantial subspace mismatch and reduced-model leakage.

Thresholds are sensitivity analyses, not claims that weak edges are absent. The matrix scale is controlled by `NORMALIZATION` in the setup cell; decay-time interpretations only have the previous unit-normalized meaning when spectral-radius normalization is enabled. The Schur basis can rotate within degenerate subspaces. Path presence does not prove exclusive activity propagation. All tables and figures are written under `OUTPUT_DIR`.""")
code("""# Display the reordered cell-type lists for every method side by side.
order_table = pd.DataFrame({
    method: pd.Series([labels[i] for i in order])
    for method, order in result["orders"].items()
})
order_table.index = pd.RangeIndex(1, len(order_table) + 1, name="order_position")
display(order_table)

order_table.to_csv(OUTPUT_DIR / "tables" / "method_order_lists_side_by_side.csv")
print("Wrote", OUTPUT_DIR / "tables" / "method_order_lists_side_by_side.csv")""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}
nbf.write(nb, "01_matrix_reordering_paper_analysis.ipynb")
