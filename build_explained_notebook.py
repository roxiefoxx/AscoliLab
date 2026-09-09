#!/usr/bin/env python3
"""Build an explained, reader-facing notebook from the validated analysis outputs."""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"


def first(frame: pd.DataFrame, column: str):
    return frame.iloc[0][column]


def build() -> Path:
    s = json.loads((OUT / "summary.json").read_text())
    desc = pd.read_csv(TABLES / "matrix_descriptive_statistics.csv").set_index("metric").value
    metrics = pd.read_csv(TABLES / "non_normality_metrics.csv")
    rawm = metrics.loc[metrics.matrix == "raw"].iloc[0]
    threshold = pd.read_csv(TABLES / "stability_reactivity_thresholds.csv")
    th2 = threshold.loc[threshold.normalization == "spectral_norm"].iloc[0]
    gains = pd.read_csv(TABLES / "selected_coupling_gains.csv")
    strength_nodes = pd.read_csv(TABLES / "weight_concentration_nodes.csv")
    strength_edges = pd.read_csv(TABLES / "weight_concentration_edges.csv")
    initiators = pd.read_csv(TABLES / "optimal_perturbation_nodes.csv")
    receivers = pd.read_csv(TABLES / "optimal_response_nodes.csv")
    single = pd.read_csv(TABLES / "single_node_perturbability.csv")
    signed = pd.read_csv(TABLES / "signed_component_comparison.csv")
    qcuts = pd.read_csv(TABLES / "directionality_cutoffs.csv")
    nullstats = pd.read_csv(TABLES / "empirical_vs_null_statistics.csv")
    lesions = pd.read_csv(TABLES / "node_lesion_effects.csv")
    edges = pd.read_csv(TABLES / "edge_sensitivity.csv")
    robust = pd.read_csv(TABLES / "robustness_analysis.csv")

    def md(text: str):
        return nbf.v4.new_markdown_cell(text.strip() + "\n")

    def code(text: str):
        return nbf.v4.new_code_cell(text.strip() + "\n")

    cells = []
    cells += [
        md(r"""
# Non-normal dynamics of the AscoliLab connectomic matrix

This notebook is a guided technical analysis of a weighted, signed, directed interaction matrix. Every computational section has four parts:

1. **Question** — what the step is testing.
2. **Terms and equations** — definitions needed to read the result.
3. **Computation** — a bounded table or figure produced by the reproducible pipeline.
4. **What the results mean** — an interpretation cell immediately after the computation.

The notebook distinguishes **asymmetry**, **non-normality**, **reactivity**, **instability**, and **finite-time amplification**. These are related but not interchangeable.
"""),
        md(fr"""
## tl;dr

- The matrix contains 85 labeled nodes and {s['nonzero_count']:,} nonzero directed effects ({s['density']:.1%} density).
- Its scale-invariant non-normality is very large: commutator index $\eta_C={rawm.commutator_index:.3f}$ and Henrici index $\eta_H={rawm.henrici_index:.4f}$.
- Under spectral-norm normalization and local decay $\gamma=1$, reactivity begins at $g={s['g_react']:.3f}$, while instability begins at $g={s['g_stab']:.3f}$.
- At the midpoint of that stable-but-reactive interval, the maximum energy gain is $G_\max={s['Gmax_operating']:.1f}$ at $t_*={s['t_star_operating']:.3f}$.
- The dominant optimal initiator is **{s['top_initiator']}**; the dominant response receiver is **{s['top_receiver']}**.
- Large effects depend strongly on a small extreme-weight tail. These are mathematical sensitivity results, not demonstrations of biological causation.
"""),
        md(r"""
## Context & Methods

### Key assumptions

The working convention is

$$A_{ij}=\text{effect of source node }j\text{ on target node }i.$$

Thus columns describe outgoing effects from a source and rows describe incoming effects to a target. The baseline linear model is

$$\dot x=J(g)x,\qquad J(g)=-\gamma I+g\widetilde A,$$

where $x$ is the perturbation state, $I$ is the identity matrix, $g$ is coupling strength, $\gamma=1$ is local decay, and $\widetilde A$ is a normalized form of the empirical matrix. The raw weights have no asserted physical time unit, so raw dynamics are not treated as biologically calibrated.

All calculations use double precision. Random controls use seed `20260901`.
"""),
        code("""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown

ROOT = Path.cwd()
OUT = ROOT / "analysis_outputs"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
summary = json.loads((OUT / "summary.json").read_text())

# Set True to regenerate every table and figure from the source CSV.
RUN_FULL_ANALYSIS = False
if RUN_FULL_ANALYSIS:
    from non_normal_analysis import main
    main("mij_matrix.csv", "analysis_outputs")

print("Source:", ROOT / "mij_matrix.csv")
print("Saved analysis:", OUT)
print("Full regeneration enabled:", RUN_FULL_ANALYSIS)
"""),
        md(r"""
### What the setup output means

The notebook is an executed reader for a deterministic analysis run. With `RUN_FULL_ANALYSIS=False`, it loads the reviewed CSV outputs quickly. Setting the flag to `True` invokes the complete pipeline, including the slower null, lesion, edge, and pseudospectral calculations. No hidden notebook state is required.
"""),
    ]

    # 1
    cells += [
        md(r"""
## 1. Import and validate the matrix

### Analysis step

This step verifies that the CSV can be treated as a square linear operator and summarizes its entries before any spectral calculation.

### Terms

- **Dimension**: number of rows and columns; a square $n\times n$ matrix maps an $n$-node state back into the same state space.
- **Nonzero entry**: an observed directed interaction. **Density** is the fraction of all $n^2$ entries that are nonzero.
- **Positive/negative entry**: an excitatory-like or inhibitory-like signed effect in the supplied matrix; the sign alone does not establish biological mechanism.
- **Diagonal entry**: $A_{ii}$, a self-effect.
- **NaN/infinity**: invalid non-finite numerical values.
- **All-zero row/column**: a target with no incoming effects or a source with no outgoing effects.
- **Quantile**: a distribution cutoff; for example, the 0.99 quantile exceeds 99% of the observed nonzero weights.
"""),
        code("""
description = pd.read_csv(TABLES / "matrix_descriptive_statistics.csv")
weight_quantiles = pd.read_csv(TABLES / "nonzero_weight_quantiles.csv")
display(description)
display(weight_quantiles)
display(Image(filename=str(FIGURES / "figure01_matrix_and_weights.png")))
"""),
        md(fr"""
### What the validation results mean

The input is a valid 85×85 operator with {int(float(desc['nonzero_count'])):,} nonzero entries and {float(desc['nonzero_fraction']):.1%} density. It contains {int(float(desc['positive_count']))} positive and {int(float(desc['negative_count']))} negative entries. All values are finite, and there are {int(float(desc['all_zero_rows']))} all-zero rows and {int(float(desc['all_zero_columns']))} all-zero columns, so no node is completely disconnected in the incoming or outgoing direction.

Weights span {float(desc['minimum_weight']):,.1f} to {float(desc['maximum_weight']):,.1f}. The heat map and signed-log histogram show a highly concentrated, heavy-tailed scale. This motivates normalization and explicit extreme-weight robustness checks; it also means unscaled magnitudes should not be read as biological rates.
"""),
    ]

    # 2
    top_edge_share = strength_edges.head(5).weight_share.sum()
    top_node_share = strength_nodes.head(5).out_weight_share.sum()
    cells += [
        md(r"""
## 2. Node strengths and weight concentration

### Analysis step

This step asks which nodes send or receive the largest total interaction weight and whether a small set of nodes or edges dominates the network.

### Terms under $A_{ij}:j\to i$

- **Absolute in-strength** of node $i$: $s_i^{in}=\sum_j|A_{ij}|$; total magnitude arriving at target $i$.
- **Absolute out-strength** of node $j$: $s_j^{out}=\sum_i|A_{ij}|$; total magnitude sent by source $j$.
- **Signed strength**: the algebraic sum retaining cancellation between positive and negative weights.
- **Positive/negative strength**: sums computed separately from positive or negative entries.
- **Degree**: number of nonzero incoming or outgoing edges, ignoring their magnitudes.
- **Weight share**: a node or edge's absolute weight divided by the total absolute weight.
- **Cumulative share**: total share accounted for by the largest entries up to a given rank.
"""),
        code("""
node_strengths = pd.read_csv(TABLES / "node_strengths.csv")
dominant_nodes = pd.read_csv(TABLES / "weight_concentration_nodes.csv")
dominant_edges = pd.read_csv(TABLES / "weight_concentration_edges.csv")
display(dominant_nodes[["node", "absolute_out_strength", "out_weight_share", "absolute_in_strength", "in_weight_share"]].head(12))
display(dominant_edges[["source", "target", "weight", "weight_share", "cumulative_weight_share"]].head(12))
"""),
        md(fr"""
### What the strength results mean

Outgoing magnitude is concentrated: the five strongest source nodes account for {top_node_share:.1%} of total absolute weight. The five largest individual edges account for {top_edge_share:.1%}; the single largest edge, **{strength_edges.iloc[0].source} → {strength_edges.iloc[0].target}**, carries {strength_edges.iloc[0].weight_share:.1%} by itself.

This concentration helps explain why clipping or removing a small extreme-weight tail changes later gain estimates sharply. A high out-strength identifies a large sender by weight, but it is not automatically the optimal perturbation initiator because optimal amplification depends on coordinated paths and singular vectors, not one-node totals alone.
"""),
    ]

    # 3
    cells += [
        md(r"""
## 3. Raw and normalized matrices

### Analysis step

This step separates structural shape from arbitrary global scale by multiplying the matrix by scalar normalization factors.

### Terms

- **Eigenvalue** $\lambda$: a scalar satisfying $Av=\lambda v$ for a nonzero eigenvector $v$.
- **Spectral radius** $\rho(A)=\max_i|\lambda_i|$: largest eigenvalue magnitude.
- **Spectral norm** $\|A\|_2=\sigma_{max}(A)$: largest singular value; the largest one-step Euclidean amplification by $A$.
- **Maximum absolute row sum** $\|A\|_\infty=\max_i\sum_j|A_{ij}|$: largest total absolute input to any target row.
- **Normalization**: division by one of these scales. Scalar normalization changes numerical coupling thresholds but does not change scale-invariant asymmetry or non-normality indices.
"""),
        code("""
normalization_metrics = pd.read_csv(TABLES / "non_normality_metrics.csv")
normalization_thresholds = pd.read_csv(TABLES / "stability_reactivity_thresholds.csv")
display(normalization_metrics[["matrix", "spectral_norm", "spectral_radius", "spectral_abscissa", "numerical_abscissa", "commutator_index", "henrici_index"]])
display(normalization_thresholds)
"""),
        md(fr"""
### What the normalization results mean

The raw, spectral-radius, spectral-norm, and row-sum matrices have identical asymmetry, commutator, and normalized Henrici indices because each differs only by a scalar. Their reported coupling thresholds change inversely with that scalar. For example, spectral-norm normalization gives $g_{{react}}={th2.g_react:.3f}$ and $g_{{stab}}={th2.g_stab:.3f}$, whereas spectral-radius normalization gives thresholds on a different numerical scale.

Therefore the defensible conclusions are about the *existence and relative width* of the stable-reactive interval and gain at a stated fraction of a stability threshold—not the raw numerical value of $g$ unless biological units are supplied.
"""),
    ]

    # 4
    cells += [
        md(r"""
## 4. Asymmetry and non-normality

### Analysis step

This step determines whether directionality merely makes $A\ne A^T$ or also makes its eigenmodes non-orthogonal enough to support strong transient interactions.

### Terms and equations

$$S=\frac{A+A^T}{2},\qquad K=\frac{A-A^T}{2},\qquad A=S+K.$$

- **Symmetric component** $S$: reciprocal component; $S=S^T$.
- **Skew-symmetric component** $K$: purely directional component; $K=-K^T$.
- **Asymmetry index** $\|A-A^T\|_F/\|A\|_F$: directional mismatch measured with the Frobenius norm $\|A\|_F=\sqrt{\sum_{ij}A_{ij}^2}$.
- **Normal matrix**: $AA^T=A^TA$. A matrix can be asymmetric yet normal.
- **Commutator** $C=AA^T-A^TA$: zero exactly for a real normal matrix.
- **Normalized commutator index** $\eta_C=\|C\|_F/\|A\|_F^2$: scale-invariant commutator magnitude.
- **Henrici departure** $d_H=\sqrt{\|A\|_F^2-\sum_i|\lambda_i|^2}$; $\eta_H=d_H/\|A\|_F$ is its normalized form.
- **Eigenvector condition number** $\kappa(V)=\|V\|\|V^{-1}\|$: sensitivity of an eigenvector decomposition $A=V\Lambda V^{-1}$. Large values indicate non-orthogonal or numerically sensitive eigenvectors; an infinite value indicates defectiveness.
"""),
        code("""
non_normality = pd.read_csv(TABLES / "non_normality_metrics.csv")
display(non_normality)
"""),
        md(fr"""
### What the non-normality results mean

The empirical matrix has asymmetry index {rawm.asymmetry_index:.3f}, but the stronger conclusion comes from $\eta_C={rawm.commutator_index:.3f}$ and $\eta_H={rawm.henrici_index:.4f}$: it is far from normal, not merely directed. The eigenvector condition number is approximately {rawm.eigenvector_condition:,.0f}, showing substantial non-orthogonality and numerical sensitivity, though the decomposition is not flagged as effectively defective by the pipeline's conservative threshold.

High non-normality makes transient amplification possible, but does not by itself prove reactivity or instability. Those require the dynamical matrix $J(g)$ and are tested next.
"""),
    ]

    # 5
    cells += [
        md(r"""
## 5. Spectral and numerical abscissae

### Analysis step

This step contrasts long-time eigenvalue behavior with the largest possible instantaneous change in perturbation energy.

### Terms

- **Spectral abscissa** $\alpha(A)=\max_i\operatorname{Re}\lambda_i(A)$: largest real part among eigenvalues. For $\dot x=Jx$, $\alpha(J)<0$ means asymptotic linear stability.
- **Numerical abscissa** $\omega(A)=\lambda_{max}((A+A^T)/2)$: maximum instantaneous growth rate of the Euclidean norm.
- **Reactivity gap** $\Delta_{react}=\omega-\alpha$: how much instantaneous growth exceeds the eigenvalue-based rate.
- **Non-normal reactivity ratio** $\omega/\alpha$: a scale-free comparison when $\alpha\ne0$; it is not meaningful when the denominator is zero or changes sign.
- **Stable** means $\alpha(J)<0$; **reactive** means $\omega(J)>0$. A system can be both stable and reactive.
"""),
        code("""
abscissa_table = pd.read_csv(TABLES / "non_normality_metrics.csv")
display(abscissa_table[["matrix", "spectral_abscissa", "numerical_abscissa", "reactivity_gap", "non_normal_reactivity_ratio"]])
display(Image(filename=str(FIGURES / "figure02_eigenvalue_spectrum.png")))
"""),
        md(fr"""
### What the abscissa results mean

For every scalar normalization, $\omega/\alpha={rawm.non_normal_reactivity_ratio:.1f}$. In spectral-norm units, $\alpha(A)={th2.alpha_A:.5f}$ while $\omega(A)={th2.omega_A:.3f}$. The numerical abscissa is therefore much larger than the spectral abscissa.

This gap predicts a regime where every eigenmode of $J$ decays eventually but some carefully oriented perturbations initially grow. It would be incorrect to call that regime unstable; instability begins only when $\alpha(J)$ crosses zero.
"""),
    ]

    # 6
    cells += [
        md(r"""
## 6. Coupling phase diagram and thresholds

### Analysis step

This step embeds the normalized matrix in $J(g)=-I+g\widetilde A$ and varies coupling $g$.

### Terms

- **Local decay** $-I$: in the absence of coupling, every state component decays with nondimensional time constant 1.
- **Coupling** $g$: scalar multiplier controlling the strength of network interaction relative to local decay.
- **Reactivity threshold** $g_{react}=1/\omega(\widetilde A)$ when $\omega>0$: where $\omega(J)$ first becomes zero.
- **Stability threshold** $g_{stab}=1/\alpha(\widetilde A)$ when $\alpha>0$: where $\alpha(J)$ first becomes zero.
- **Stable/non-reactive region**: $\alpha(J)<0$ and $\omega(J)\le0$.
- **Stable/reactive region**: $\alpha(J)<0$ and $\omega(J)>0$.
- **Unstable region**: $\alpha(J)>0$.
"""),
        code("""
phase = pd.read_csv(TABLES / "coupling_phase.csv")
thresholds = pd.read_csv(TABLES / "stability_reactivity_thresholds.csv")
display(thresholds)
display(Image(filename=str(FIGURES / "figure03_coupling_phase.png")))
"""),
        md(fr"""
### What the phase diagram means

Under spectral-norm normalization, $g_{{react}}={s['g_react']:.3f}$ and $g_{{stab}}={s['g_stab']:.3f}$. Reactivity begins at only {s['g_react']/s['g_stab']:.1%} of the stability threshold, leaving a stable-reactive width of {th2.stable_reactive_width:.3f} coupling units.

The shaded middle region is the core non-normal phenomenon: perturbations can grow temporarily even though $\alpha(J)<0$ guarantees eventual decay. The broad interval is preserved under scalar normalization even though its numerical endpoints change.
"""),
    ]

    # 7
    near = gains.loc[gains.coupling_label == "near_stability"].iloc[0]
    cells += [
        md(r"""
## 7. Finite-time transient amplification

### Analysis step

This step computes how much a perturbation can grow over a finite time, rather than inferring behavior only from eigenvalues.

### Terms

- **Propagator** $P(t)=e^{Jt}$: matrix exponential mapping the initial state $x(0)$ to $x(t)$.
- **Singular value** $\sigma_{max}(P)$: maximum norm amplification over all unit-length initial conditions.
- **Energy gain** $G(t;g)=\|e^{Jt}\|_2^2=\sigma_{max}^2(e^{Jt})$: ratio of squared Euclidean response norm to squared initial norm.
- **Maximum gain** $G_{max}(g)=\max_{t\ge0}G(t;g)$.
- **Time to peak** $t_*(g)=\arg\max_tG(t;g)$.
- **Adaptive horizon**: the time window is chosen from the inverse decay rate so slow stable modes are not prematurely truncated.
- **Coarse scan plus refinement**: a time grid locates a candidate maximum, then bounded scalar optimization refines $t_*$.
"""),
        code("""
selected_gains = pd.read_csv(TABLES / "selected_coupling_gains.csv")
gain_by_coupling = pd.read_csv(TABLES / "maximum_gain_vs_coupling.csv")
display(selected_gains)
display(Image(filename=str(FIGURES / "figure04_transient_gain_curves.png")))
display(Image(filename=str(FIGURES / "figure05_max_gain_vs_coupling.png")))
"""),
        md(fr"""
### What the transient-gain results mean

Below and immediately before the reactivity threshold, the maximum is $G_{{max}}=1$ at $t=0$: every later state is smaller. At the midpoint of the stable-reactive interval, $G_{{max}}={s['Gmax_operating']:.1f}$ at $t_*={s['t_star_operating']:.3f}$. Near 95% of the stability threshold, gain rises to {near.Gmax:.1f}.

These are squared-norm or “energy” gains. A gain of {s['Gmax_operating']:.1f} corresponds to an amplitude gain of $\sqrt{{G_{{max}}}}\approx{np.sqrt(s['Gmax_operating']):.1f}$. The gain is transient: the system remains asymptotically stable at the selected couplings and eventually decays.
"""),
    ]

    # 8
    cells += [
        md(r"""
## 8. Optimal perturbation and optimal response

### Analysis step

At the peak time, this step uses the singular value decomposition

$$e^{Jt_*}=U\Sigma V^T$$

to identify the coordinated initial pattern that produces maximum amplification and the corresponding response pattern.

### Terms

- **SVD**: factorization into orthonormal left singular vectors $U$, nonnegative singular values $\Sigma$, and orthonormal right singular vectors $V$.
- **Leading right singular vector** $v_1$: unit initial perturbation producing the maximum gain.
- **Leading left singular vector** $u_1$: direction of the maximally amplified response at $t_*$.
- **Initiator score** $|v_{1,i}|$: magnitude of node $i$ in the optimal initial pattern.
- **Receiver score** $|u_{1,i}|$: magnitude of node $i$ in the optimal response pattern.
- Signs of singular vectors are globally arbitrary; rankings therefore use absolute values.
"""),
        code("""
initiators = pd.read_csv(TABLES / "optimal_perturbation_nodes.csv")
receivers = pd.read_csv(TABLES / "optimal_response_nodes.csv")
display(initiators.head(15))
display(receivers.head(15))
display(Image(filename=str(FIGURES / "figure06_optimal_vectors.png")))
"""),
        md(fr"""
### What the optimal-vector results mean

**{initiators.iloc[0].node}** has the largest initiator magnitude ($|v_1|={initiators.iloc[0].abs_v1:.3f}$), closely followed by **{initiators.iloc[1].node}**. The amplified response is dominated by **{receivers.iloc[0].node}** ($|u_1|={receivers.iloc[0].abs_u1:.3f}$) and **{receivers.iloc[1].node}**.

Initiator and receiver rankings differ sharply. This is expected in a directed non-normal system: the best input pattern and the output pattern it produces occupy different node combinations. These vectors describe a coordinated multi-node perturbation, not the effect of activating one node alone.
"""),
    ]

    # 9
    cells += [
        md(r"""
## 9. Single-node perturbability, susceptibility, and integrated response

### Analysis step

This step constrains the initial condition to one node at a time, $x(0)=e_i$, and separately asks which nodes receive large downstream responses.

### Terms

- **Basis vector** $e_i$: a vector with 1 at node $i$ and 0 elsewhere.
- **Single-node peak gain** $G_i=\max_t\|e^{Jt}e_i\|_2^2$: largest total response energy caused by a unit perturbation at node $i$.
- **Time to peak**: time at which $G_i$ is largest on the adaptive grid.
- **Integrated energy** $R_i=\int_0^\infty\|e^{Jt}e_i\|_2^2dt$: total response accumulated over all time.
- **Observability Gramian** $Q$: solution of $J^TQ+QJ=-I$; $R_i=Q_{ii}$ for stable $J$.
- **Perturbability**: how strongly a node can initiate a network-wide response.
- **Susceptibility**: how strongly a node receives responses across perturbation sources.
- Peak and integrated rankings need not agree because one favors height and the other favors duration.
"""),
        code("""
single_node = pd.read_csv(TABLES / "single_node_perturbability.csv")
susceptibility = pd.read_csv(TABLES / "node_susceptibility.csv")
display(single_node.head(20))
display(susceptibility.head(20))
display(Image(filename=str(FIGURES / "figure07_single_node_ranking.png")))
"""),
        md(fr"""
### What the single-node results mean

**{single.iloc[0].node}** is the strongest constrained single-node initiator, with peak energy gain {single.iloc[0].single_node_peak_gain:.1f} and integrated energy {single.iloc[0].integrated_energy:.1f}. This differs from the leading component of the unconstrained optimal vector, confirming that a single-node input cannot reproduce the fully coordinated optimal perturbation.

The susceptibility table answers the reverse question: which targets accumulate the largest maximum received amplitudes across sources. A node may be highly susceptible without being a strong initiator, again reflecting directional flow.
"""),
    ]

    # 10
    sp = signed.set_index("matrix")
    cells += [
        md(r"""
## 10. Signed structure and inhibitory interpolation

### Analysis step

This step separates positive and negative weights and gradually restores the negative component.

$$A^+_{ij}=\max(A_{ij},0),\qquad A^-_{ij}=\min(A_{ij},0),$$
$$A(\beta)=A^++\beta A^-,\qquad 0\le\beta\le1.$$

### Terms

- **Positive component** $A^+$: all negative entries set to zero.
- **Negative component** $A^-$: all positive entries set to zero.
- **Inhibitory-strength parameter** $\beta$: 0 is positive-only; 1 restores the empirical signed matrix.
- Each component is spectral-norm normalized for the component comparison, and gain is evaluated at 90% of that component's own stability threshold. This compares shape at matched relative stability, not identical raw coupling.
"""),
        code("""
signed_components = pd.read_csv(TABLES / "signed_component_comparison.csv")
beta_scan = pd.read_csv(TABLES / "signed_inhibition_interpolation.csv")
display(signed_components)
display(beta_scan)
"""),
        md(fr"""
### What the signed-structure results mean

The positive-only matrix already has very high non-normality and gain: $G_{{max}}={sp.loc['positive','Gmax_at_90pct_stability']:.1f}$. Restoring the empirical negative component increases the matched-relative-coupling gain only modestly to {sp.loc['signed','Gmax_at_90pct_stability']:.1f}. The negative-only component yields much smaller gain ({sp.loc['negative','Gmax_at_90pct_stability']:.1f}).

Thus the signed inhibitory structure modulates the effect but does not create the dominant non-normal amplification. This result is conditional on the chosen normalization and relative-coupling comparison; it is not a statement that inhibition is biologically unimportant.
"""),
    ]

    # 11
    cells += [
        md(r"""
## 11. Directionality interpolation and empirical cutoffs

### Analysis step

This step removes directionality continuously while preserving the symmetric component:

$$A(q)=S+qK,\qquad 0\le q\le1.$$

### Terms

- $q=0$: fully symmetric matrix $S$, which is normal.
- $q=1$: the empirical directed matrix $S+K=A$.
- **Directionality cutoff** $q_c$: smallest scanned $q$ for which $G_{max}\ge1+\delta$.
- **Amplification tolerance** $\delta$: selected excess above unit gain. Multiple values are reported because no single cutoff is biologically canonical.
- Gain is evaluated at 90% of each interpolated matrix's own stability threshold, isolating shape at matched relative stability.
"""),
        code("""
directionality = pd.read_csv(TABLES / "directionality_interpolation.csv")
direction_cutoffs = pd.read_csv(TABLES / "directionality_cutoffs.csv")
display(directionality)
display(direction_cutoffs)
display(Image(filename=str(FIGURES / "figure08_directionality_interpolation.png")))
"""),
        md(fr"""
### What the directionality results mean

At $q=0$, both non-normality indices are zero and $G_{{max}}=1$, as required for the stable symmetric normal baseline. As empirical directionality is restored, the commutator grows smoothly but large gain emerges late and sharply.

The grid cutoff is $q_c={qcuts.loc[qcuts.delta==0.01,'q_cutoff_grid'].iloc[0]:.2f}$ for a 1% gain excess and $q_c={qcuts.loc[qcuts.delta==0.25,'q_cutoff_grid'].iloc[0]:.2f}$ for a 25% excess. A substantial fraction of the empirical directional component is therefore required before amplification becomes appreciable under this criterion.
"""),
    ]

    # 12
    cells += [
        md(r"""
## 12. Null models and empirical significance

### Analysis step

This step asks whether the empirical metrics are unusual relative to randomized matrices that preserve selected features.

### Terms

- **Null model**: randomized control representing what would be expected if a specified structural feature were irrelevant.
- **Symmetric control**: $(A+A^T)/2$, removing directed skew.
- **Weight-shuffled null**: preserves the locations of nonzero directed edges but permutes observed signed weights across them.
- **Direction-randomized null**: randomly swaps pairwise directions while preserving the paired weights and signs as closely as possible.
- **Empirical percentile**: fraction of null values at or below the empirical value.
- **Upper-tail empirical p-value**: $(1+\#\{T_{null}\ge T_{emp}\})/(n+1)$; the +1 correction prevents zero p-values.
- **Standardized effect**: $(T_{emp}-\bar T_{null})/s_{null}$, measured in null standard deviations.
- Fifty nulls per family are used for spectral metrics; 20 per family receive the costlier gain computation.
"""),
        code("""
null_results = pd.read_csv(TABLES / "null_model_results.csv")
null_statistics = pd.read_csv(TABLES / "empirical_vs_null_statistics.csv")
display(null_statistics)
display(Image(filename=str(FIGURES / "figure09_null_distributions.png")))
"""),
        md(r"""
### What the null-model results mean

The empirical commutator index is above all 50 weight-shuffled and all 50 direction-randomized values; the corrected upper-tail p-value is 0.0196 for both families. The empirical Henrici index is also above all sampled nulls.

Gain is above the direction-randomized controls (p=0.0476 across 20 gain nulls), but the weight-shuffled gain comparison is less decisive (p=0.0952). The evidence therefore supports unusually strong empirical non-normality and direction-dependent gain, while the exact tail probability for gain remains uncertain because only 20 costly gain nulls were evaluated. These p-values are exploratory, not publication-final.
"""),
    ]

    # 13
    cells += [
        md(r"""
## 13. Node lesion analysis

### Analysis step

For each node, this step removes its row and column and recomputes spectral, reactivity, and gain-related metrics.

### Terms

- **Node lesion**: removal of both the node's incoming row and outgoing column.
- **Lesion effect** $\Delta G_{max,i}=G_{max}^{(-i)}-G_{max}$: negative values mean removal reduces amplification.
- **Gain screen**: lesion gain is evaluated at five times around the empirical $t_*$ rather than fully re-optimized for all 85 lesions.
- **Exact lesion thresholds**: lesion $\alpha$, $\omega$, $g_{react}$, and $g_{stab}$ are recomputed directly and are not screened approximations.
- A node can have a strong lesion effect without being the strongest input component because removal changes the entire propagation operator.
"""),
        code("""
lesions = pd.read_csv(TABLES / "node_lesion_effects.csv")
display(lesions.head(25))
display(Image(filename=str(FIGURES / "figure10_node_lesions.png")))
"""),
        md(fr"""
### What the lesion results mean

Removing **{lesions.iloc[0].node}** produces the largest screened reduction in gain ($\Delta G\approx{lesions.iloc[0].delta_Gmax_screen:.1f}$), followed by **{lesions.iloc[1].node}**. The leading lesion is the dominant response receiver, not the leading single-node initiator.

This identifies nodes structurally important to the amplification pathway, but the gain values are screening estimates. The top candidates should be fully re-optimized in time before making inferential or experimental-prioritization claims.
"""),
    ]

    # 14
    cells += [
        md(r"""
## 14. Edge perturbation and local sensitivity

### Analysis step

This step estimates how small changes in each nonzero edge affect stability, reactivity, and peak gain, prioritizing the most consequential edges.

### Terms

- **Edge perturbation**: $A_{ij}\to(1+\epsilon)A_{ij}$ or deletion $A_{ij}\to0$.
- **Derivative** $d\alpha/dA_{ij}$ or $d\omega/dA_{ij}$: first-order local change per unit edge-weight change.
- **Elasticity** $A_{ij}\,dT/dA_{ij}$: first-order change in metric $T$ for a fractional edge change.
- **Frechet derivative of the matrix exponential**: linear response of $e^{Jt}$ to a matrix perturbation; used for gain sensitivity.
- **Local sensitivity** does not guarantee the same effect for a large perturbation or full edge deletion.
- Gain Frechet derivatives are evaluated for the top 40 edges screened by spectral/reactivity sensitivity.
"""),
        code("""
edge_sensitivity = pd.read_csv(TABLES / "edge_sensitivity.csv")
display(edge_sensitivity[["source", "target", "weight", "elasticity_alpha", "elasticity_omega", "dG_dAij", "elasticity_G"]].head(25))
"""),
        md(fr"""
### What the edge-sensitivity results mean

The leading locally sensitive edge is **{edges.iloc[0].source} → {edges.iloc[0].target}**. Its gain elasticity is {edges.iloc[0].elasticity_G:.1f}, and it also has the largest reactivity elasticity among the prioritized edges. Several other top-ranked edges converge on the same CA3c Pyramidal target.

The ranking identifies high-value candidates for explicit deletion or bidirectional perturbation tests. Because these are derivatives around the empirical weight, they should not be interpreted as exact effects of removing an edge entirely.
"""),
    ]

    # 15
    cells += [
        md(r"""
## 15. Pseudospectral analysis

### Analysis step

This step evaluates how close the stable operator is to having eigenvalues elsewhere in the complex plane under small perturbations.

### Terms

- **Resolvent** $(zI-J)^{-1}$: frequency-domain response at complex point $z$.
- **Resolvent norm** $\|(zI-J)^{-1}\|_2$: large values mean $zI-J$ is nearly singular.
- **$\epsilon$-pseudospectrum** $\Lambda_\epsilon(J)=\{z:\|(zI-J)^{-1}\|_2>1/\epsilon\}$: points that can behave like eigenvalues after perturbations of approximate size $\epsilon$.
- **Pseudospectral bulging**: contours extending far from the eigenvalues, characteristic of non-normal sensitivity.
- **Right half-plane** $\operatorname{Re}z>0$: the unstable side of the continuous-time complex plane.
- The displayed result is a finite grid approximation, not a certified contour boundary.
"""),
        code("""
pseudo = np.load(OUT / "pseudospectrum_grid.npz")
right_mask = pseudo["real"] > 0
max_right_resolvent = 10 ** pseudo["log10_resolvent"][:, right_mask].max()
epsilon_grid = 1 / max_right_resolvent
display(pd.DataFrame({"quantity": ["max resolvent sampled in Re(z)>0", "smallest epsilon implied by sampled grid"],
                      "value": [max_right_resolvent, epsilon_grid]}))
display(Image(filename=str(FIGURES / "figure11_pseudospectrum.png")))
"""),
        md(fr"""
### What the pseudospectrum means

Although all eigenvalues of the selected $J$ lie in the stable half-plane, the sampled pseudospectrum reaches $\operatorname{{Re}}z>0$ at approximately $\epsilon={s['pseudospectral_right_half_min_epsilon_grid']:.3f}$. This indicates sensitivity to structured operator perturbations and is consistent with the large transient gain.

The value depends on the finite grid extent and resolution. It should be treated as a diagnostic scale, not an exact robust-stability radius.
"""),
    ]

    # 16
    cells += [
        md(r"""
## 16. Stochastic perturbations and covariance

### Analysis step

This step replaces a single deterministic impulse with continuous white-noise forcing:

$$dx=Jx\,dt+B\,dW_t,$$

using $B=I$ and a coupling stable for both empirical and symmetrized networks.

### Terms

- **Wiener process** $W_t$: idealized continuous-time white-noise source.
- **Noise matrix** $B$: maps independent noise sources into nodes; $B=I$ gives equal independent forcing.
- **Stationary covariance** $C$: long-run covariance of $x$ when $J$ is stable.
- **Lyapunov equation** $JC+CJ^T+BB^T=0$: equation solved for $C$.
- **Total stationary variance** $\operatorname{tr}(C)$: sum of node variances.
- **Nodewise variance** $C_{ii}$: stationary variance at node $i$.
- **Covariance mode**: eigenvector of $C$; its eigenvalue is variance carried by that collective pattern.
- Peak deterministic gain and stationary variance need not rank networks or nodes identically.
"""),
        code("""
node_variance = pd.read_csv(TABLES / "stochastic_node_variance.csv")
covariance_modes = pd.read_csv(TABLES / "stochastic_covariance_modes.csv")
display(pd.DataFrame({
    "network": ["empirical", "symmetrized"],
    "common coupling": [summary["stochastic_comparison_coupling"]] * 2,
    "total stationary variance": [summary["total_stationary_variance"], summary["symmetrized_total_stationary_variance"]],
}))
display(node_variance.head(15))
display(covariance_modes)
"""),
        md(fr"""
### What the stochastic results mean

At the largest tested coupling stable for both systems, $g={s['stochastic_comparison_coupling']:.3f}$, total stationary variance is {s['total_stationary_variance']:.2f} for the empirical network and {s['symmetrized_total_stationary_variance']:.2f} for the symmetrized network. The empirical value is therefore slightly smaller at this low common coupling.

This does not contradict the deterministic gain result: the empirical network's striking amplification occurs much deeper in its unusually broad stable-reactive interval, at couplings where the symmetrized network is already unstable and has no stationary covariance. Comparisons require a domain where both Lyapunov solutions are physically meaningful.
"""),
    ]

    # 17
    raw_gain = robust.loc[robust.variant == "raw", "Gmax_at_90pct_stability"].iloc[0]
    rm1 = robust.loc[robust.variant == "remove_top_1pct_edges", "Gmax_at_90pct_stability"].iloc[0]
    rm5 = robust.loc[robust.variant == "remove_top_5pct_edges", "Gmax_at_90pct_stability"].iloc[0]
    cells += [
        md(r"""
## 17. Robustness to preprocessing and extreme weights

### Analysis step

This step repeats key metrics after controlled modifications to determine which conclusions are structural and which depend on a few entries.

### Terms

- **Remove diagonal**: set self-connections $A_{ii}$ to zero.
- **Remove top $p\%$ edges**: set the largest $p\%$ of nonzero absolute weights to zero.
- **Winsorization/clipping**: cap weights at a selected percentile while preserving their signs and edge locations.
- **Random weight noise**: multiply nonzero weights by small random factors.
- **Robust conclusion**: remains qualitatively similar across plausible perturbations.
- Every variant is spectral-norm normalized, and gain is measured at 90% of that variant's own stability threshold.
"""),
        code("""
robustness = pd.read_csv(TABLES / "robustness_analysis.csv")
display(robustness)
"""),
        md(fr"""
### What the robustness results mean

Small 1% and 5% random weight noise leaves the qualitative high-gain result intact. Removing the diagonal also leaves extreme non-normality and can increase matched-relative-coupling gain.

The conclusion is not robust to removing the extreme edge tail: gain falls from {raw_gain:.1f} in the primary matrix to {rm1:.1f} after removing the largest 1% of edges and {rm5:.1f} after removing the largest 5%. Winsorization produces the same qualitative warning. The network's strong amplification is therefore structurally real in the supplied matrix but heavily concentrated in its largest weights; validating those weights is biologically important.
"""),
    ]

    # 18
    cells += [
        md(r"""
## 18. Nonlinear extension

### Analysis step

The linear calculations extend locally to a justified nonlinear model, for example

$$\tau_i\dot x_i=-x_i+\phi\!\left(g\sum_jA_{ij}x_j+I_i\right).$$

At an equilibrium $x^*$, compute the **Jacobian**

$$J=DF(x^*),$$

the matrix of partial derivatives of the nonlinear vector field $F$ evaluated at that equilibrium.

### Terms

- **Activation function** $\phi$: nonlinear input-output function.
- **Time constant** $\tau_i$: intrinsic response timescale of node $i$.
- **Baseline input** $I_i$: external drive.
- **Equilibrium** $x^*$: state satisfying $F(x^*)=0$.
- **Jacobian** $DF(x^*)$: best linear approximation near $x^*$.
- **State-dependent reactivity**: $\omega(J)$ and transient gain change with the equilibrium because derivatives of $\phi$ depend on state.
- A nonlinear simulation is not justified until $\phi$, $\tau_i$, $I_i$, parameter values, and the relevant equilibrium are specified.
"""),
        code("""
nonlinear_extension = pd.DataFrame({
    "linear diagnostic": ["spectral abscissa", "numerical abscissa", "finite-time gain", "optimal perturbation"],
    "nonlinear local calculation": ["alpha(DF(x*))", "omega(DF(x*))", "||exp(DF(x*) t)||_2^2", "leading right singular vector of exp(DF(x*) t*)"],
    "interpretation": ["local asymptotic stability", "local instantaneous reactivity", "local finite-time amplification", "most amplified small perturbation near x*"],
})
display(nonlinear_extension)
"""),
        md(r"""
### What the nonlinear extension means

The entire linear toolkit applies locally after replacing the structural matrix by the equilibrium Jacobian. Structural non-normality can then be amplified, suppressed, or redirected by state-dependent gains $\phi'$. Consequently, a network may be reactive around one equilibrium and non-reactive around another.

The present analysis does not select a nonlinear model because the required activation functions, time constants, inputs, and operating state were not supplied. Inventing them would add assumptions that could dominate the result.
"""),
    ]

    # Synthesis
    cells += [
        md(r"""
## 19. Compact scientific synthesis

### Analysis step

This final computation gathers the central numerical findings in one table. Each row answers a distinct scientific question; no row should be substituted for another.
"""),
        code("""
synthesis = pd.DataFrame([
    ("How non-normal?", f"eta_C={non_normality.loc[non_normality.matrix == 'raw', 'commutator_index'].iloc[0]:.3f}; omega/alpha={summary['omega_A2']/summary['alpha_A2']:.1f}"),
    ("Stable-reactive interval", f"g_react={summary['g_react']:.3f}; g_stab={summary['g_stab']:.3f}"),
    ("Mid-interval amplification", f"Gmax={summary['Gmax_operating']:.1f} at t*={summary['t_star_operating']:.3f}"),
    ("Leading optimal initiator", summary["top_initiator"]),
    ("Leading receiver", summary["top_receiver"]),
    ("Leading single-node initiator", summary["top_single_node"]),
    ("Strongest screened lesion", summary["strongest_gain_reducing_lesion_screen"]),
    ("Leading local edge sensitivity", summary["top_edge_by_local_spectral_or_reactivity_sensitivity"]),
], columns=["question", "result"])
display(synthesis)
"""),
        md(r"""
### What the complete analysis means

The supplied connectomic matrix is not merely asymmetric: it is strongly non-normal and supports a broad range of stable couplings with substantial transient amplification. Directionality is essential, while the negative component modestly modifies rather than creates the dominant effect. Initiators, receivers, susceptible nodes, and lesion-control nodes are distinct roles.

The empirical commutator is more extreme than the sampled null controls, and the pseudospectrum supports structural sensitivity. However, the largest weights dominate the magnitude, gain-null replication is limited, lesion gain is screened, and the raw weights lack calibrated dynamical units. The strongest defensible conclusion is therefore structural: **the supplied directed weight pattern has exceptional capacity for stable transient amplification under explicit normalized linear dynamics**. Biological meaning requires independent validation of the extreme weights and a justified operating coupling.
"""),
        md(r"""
## Takeaways and next checks

1. Validate the largest-weight pathways, because they control much of the amplification.
2. Increase gain-null replication to at least 100–500 per null family for publication-grade tail estimates.
3. Fully re-optimize gain for the leading lesion candidates and explicitly delete the top edge candidates.
4. Add a signed degree- or strength-preserving rewiring null if a well-mixed sampling procedure can be demonstrated.
5. Specify physiological time constants, inputs, activation functions, and an equilibrium before nonlinear simulation.

### Interpretation guardrail

Mathematical sensitivity identifies where a model is most responsive. It does **not** by itself demonstrate biological causation, experimental controllability, or functional importance.
"""),
    ]

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata.language_info = {"name": "python", "version": "3.12"}
    path = ROOT / "non_normal_dynamics_analysis.ipynb"
    nbf.write(nb, path)
    print(f"Wrote {path} with {len(cells)} cells")
    return path


if __name__ == "__main__":
    build()
