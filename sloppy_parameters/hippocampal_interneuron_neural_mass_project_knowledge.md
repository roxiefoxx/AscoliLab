# Hippocampal Interneuron Neural-Mass / Netlist Modeling Project Knowledge

## Purpose

This document archives and consolidates the modeling decisions developed in the associated ChatGPT conversation. It is intended to be imported into another project as background knowledge.

The project goal is to build a biologically grounded, cell-type-resolved neural-mass model of hippocampal and entorhinal circuitry using an existing directed netlist. The model should support:

- region-specific and layer-specific dynamics,
- inhibitory interneuron subtype diversity,
- explicit dendritic and axonal morphology parameters,
- neuromodulatory receptor effects,
- netlist-derived coupling,
- uncertainty-aware parameter priors,
- bifurcation / continuation analysis,
- provenance-aware biological assumptions.

The preferred architecture is:

\[
\boxed{
\text{Literature / curated databases}
\rightarrow
\text{cell-type knowledge base}
\rightarrow
\text{parameter priors}
\rightarrow
\text{netlist neural-mass ODE model}
\rightarrow
\text{bifurcation / inference}
}
\]

---

# 1. Core modeling principles

## 1.1 Inhibitory neurons should not be collapsed into one generic inhibitory mass

Hippocampal interneurons differ in:

- intrinsic electrophysiology,
- molecular phenotype,
- developmental origin,
- soma location,
- dendritic extent,
- axonal termination pattern,
- target cell class,
- postsynaptic compartment,
- synaptic kinetics,
- neuromodulatory receptor expression,
- oscillatory phase preference.

Examples include:

- PV basket cells,
- CCK basket cells,
- axo-axonic/chandelier cells,
- bistratified cells,
- O-LM cells,
- HIPP cells,
- neurogliaform cells,
- Ivy cells,
- CCK dendrite-targeting interneurons,
- interneuron-selective/VIP-related interneurons.

The model should preserve biologically meaningful cell-type distinctions whenever the netlist provides them.

---

## 1.2 Cell names are identifiers, not substitutes for parameters

A label such as

`CA1 O LM`

should not directly mean “use an O-LM equation.”

Instead, the label should map to a parameter bundle such as:

\[
\Theta_i =
\{
\tau_m,
\Delta,
\eta,
b,
D_i^{tot},
d_{i\ell},
A_i^{tot},
a_{i\ell c},
U_i,
\tau_i^{syn},
\beta_{i,m},
\ldots
\}.
\]

The label is a lookup key for biological priors.

---

## 1.3 Stable cell identity must be separated from dynamic neuromodulation

This distinction is essential.

For example:

`Basket CCK+`

is a distinct molecular/morphological subtype.

It is **not** a generic basket cell temporarily modulated by CCK.

The correct hierarchy is:

\[
\text{Basket family}
\begin{cases}
\text{Basket, molecular subtype unresolved}\\
\text{Basket CCK+}\\
\text{PV basket, if explicitly identified}
\end{cases}
\]

A PV basket cell does not become a CCK-positive basket cell when exposed to CCK.

Stable identity parameters determine which receptor/modulatory pathways are plausible. Dynamic states determine how strongly those pathways are engaged.

Example:

\[
M_{\mathrm{CCK},i}=1
\]

is a fixed cell-identity marker, whereas

\[
O_{\mathrm{CB1},i}(t)
\]

is a dynamic receptor-occupancy state.

---

# 2. Netlist scope

The netlist under discussion contains approximately 85 populations spanning the following coarse regions:

\[
\mathrm{DG,\ CA3/CA3c,\ CA2,\ CA1,\ SUB,\ EC,\ LEC,\ MEC}.
\]

A previously observed coarse census after treating CA3c as a CA3 subdivision was approximately:

| Region | Excitatory | Inhibitory | Total |
|---|---:|---:|---:|
| DG | 4 | 10 | 14 |
| CA3 | 3 | 11 | 14 |
| CA2 | 1 | 4 | 5 |
| CA1 | 2 | 26 | 28 |
| SUB | 1 | 0 | 1 |
| EC | 9 | 0 | 9 |
| LEC | 3 | 0 | 3 |
| MEC | 9 | 2 | 11 |

The model should use the actual loaded netlist rather than hard-coding this census.

---

# 3. Parsing region, layer, projection target, and cell type

Several cell-type names encode cortical layer information in the label.

Examples:

`EC LI II Multipolar Pyramidal`

should parse as:

\[
\text{region}=\mathrm{EC},
\qquad
\text{layer}=\mathrm{LI-II},
\qquad
\text{cell type}=\text{Multipolar Pyramidal}.
\]

Similarly:

`MEC LII III Pyramidal Multiform`

\[
\rightarrow
(\mathrm{MEC},\mathrm{LII-III},\mathrm{Pyramidal\ Multiform}).
\]

`EC LIII V Bipolar Pyramidal`

\[
\rightarrow
(\mathrm{EC},\mathrm{LIII-V},\mathrm{Bipolar\ Pyramidal}).
\]

`SUB CA1 Projecting Pyramidal`

should be parsed differently:

\[
\text{region}=\mathrm{SUB},
\qquad
\text{projection target}=\mathrm{CA1},
\qquad
\text{cell type}=\mathrm{Projecting\ Pyramidal}.
\]

The parser should distinguish:

- region,
- region subdivision,
- explicit cortical layer,
- projection target,
- cell-type name.

---

## 3.1 Do not infer hippocampal soma layer from every anatomical word in a cell name

Names such as:

- O-LM,
- LMR,
- Schaffer Collateral Associated,
- Perforant Path Associated,
- Mossy Fiber Associated,
- Radiatum Giant,

may encode dendritic location, axonal termination, pathway association, or multiple anatomical properties.

Therefore:

- explicit EC/MEC/LEC Roman-numeral layer tokens can be parsed automatically,
- hippocampal soma stratum should preferably come from a curated anatomical knowledge source.

A future database field should include:

\[
\texttt{soma\_stratum}
\]

separately from:

\[
\texttt{axon\_target\_stratum},
\qquad
\texttt{dendritic\_extent}.
\]

---

# 4. Neural-mass equations

The recommended base model is the Montbrió-Pazó-Roxin / quadratic integrate-and-fire mean-field system.

For population \(i\):

\[
\tau_i \dot r_i
=
\frac{\Delta_i}{\pi\tau_i}
+
2r_i v_i,
\]

\[
\tau_i \dot v_i
=
v_i^2
+
\eta_i^{eff}
+
I_i^{ext}
+
I_i^{rec}
-
(\pi\tau_i r_i)^2
-
a_i,
\]

\[
\tau_i^a\dot a_i
=
-a_i+b_i r_i.
\]

Synaptic dynamics:

\[
\tau_i^A\dot s_i^A
=
-s_i^A+U_i^A r_i,
\]

\[
\tau_i^B\dot s_i^B
=
-s_i^B+U_i^B r_i.
\]

The full state for \(N\) populations is:

\[
x=
(r_1,\ldots,r_N,
v_1,\ldots,v_N,
a_1,\ldots,a_N,
s_1^A,\ldots,s_N^A,
s_1^B,\ldots,s_N^B).
\]

For \(N=85\), the state dimension is:

\[
5N=425.
\]

---

# 5. Netlist coupling

Let \(M_{\rm raw}\) denote the signed directed connectivity derived from the netlist.

Convention:

\[
M_{ij}
=
\text{connection from source }j
\text{ to target }i.
\]

Raw netlist coefficients should not automatically be interpreted as MPR/QIF voltage currents.

A conditioning step used in the prototype was:

\[
W_0
=
\frac{M_{\rm raw}}
{\rho(|M_{\rm raw}|)},
\]

where \(\rho\) is the spectral radius.

Then:

\[
I_i^{rec}
=
g
\sum_j
W_{0,ij}
(s_j^A+s_j^B).
\]

This preserves:

- sign,
- direction,
- sparsity,
- relative coupling magnitude,

while keeping the global dynamical coupling scale \(g\) explicit.

The absolute mapping from netlist \(m_{ij}\) to physical/effective neural-mass coupling must eventually be calibrated.

---

# 6. Morphology must be explicit model structure

Dendritic extent, axonal extent, and postsynaptic compartment targeting should be model parameters, not merely name annotations.

---

## 6.1 Dendritic distribution

For population \(i\), define a laminar dendritic distribution:

\[
\mathbf d_i
=
(d_{i1},d_{i2},\ldots,d_{iL}),
\]

with:

\[
\sum_\ell d_{i\ell}=1.
\]

Separate the normalized distribution from total extent:

\[
D_i^{tot}>0.
\]

Thus:

\[
D_i^{tot}d_{i\ell}
\]

represents effective dendritic availability in layer \(\ell\).

Possible CA1 layers:

\[
\{s.o.,s.p.,s.r.,s.l.m.\}.
\]

Possible entorhinal layers:

\[
\{\mathrm{LI,LII,LIII,LIV,LV,LVI}\}.
\]

---

## 6.2 Afferent pathway overlap

If pathway \(k\) has laminar termination profile:

\[
\mathbf p_k
=
(p_{k1},\ldots,p_{kL}),
\]

define:

\[
\Omega_{ki}^{in}
=
D_i^{tot}
\sum_\ell p_{k\ell}d_{i\ell}.
\]

Then:

\[
I_i^{aff}
=
\sum_k
G_{ki}^{aff}
\Omega_{ki}^{in}
s_k.
\]

Thus dendritic anatomy affects recruitment by specific afferent pathways.

---

## 6.3 Axonal termination tensor

For source population \(j\), define:

\[
a_{j\ell c},
\]

where:

- \(\ell\) = target layer,
- \(c\) = postsynaptic compartment.

Possible compartments:

\[
c\in
\{
AIS,
soma,
proximal,
basal,
apical,
distal,
interneuron
\}.
\]

Normalize:

\[
\sum_{\ell,c}
a_{j\ell c}=1.
\]

Also define total axonal/output extent:

\[
A_j^{tot}.
\]

Then:

\[
A_j^{tot}a_{j\ell c}
\]

is the effective axonal output density to that layer/compartment.

---

## 6.4 Compartment-dependent effectiveness

A distal dendritic synapse should not be assumed dynamically equivalent to an AIS or somatic synapse.

Define:

\[
\Gamma_{ic}
\]

as the transfer effectiveness of inhibition/excitation at compartment \(c\) for target population \(i\).

A morphology-aware effective coupling can be written:

\[
J_{ij}^{eff}
=
J_{ij}^{0}
A_j^{tot}
\sum_{\ell,c}
a_{j\ell c}
d_{i\ell c}
\Gamma_{ic}.
\]

More generally:

\[
J_{ij}^{eff}
=
J_{ij}^{0}
\Omega_{ij}^{morph}
\Gamma_{ij}
M_{ij}^{mod}.
\]

---

# 7. Pathway-specific dendritic gating

For principal cells, dendritic inhibition should preferably gate the relevant excitatory pathway rather than simply subtract from a single somatic current.

Example for CA1 pyramidal cells:

\[
I_E^{prox}
=
G_{CA3}s_{CA3}
-
J_{BiC}s_{BiC}
-
J_{SCA}s_{SCA},
\]

\[
I_E^{dist}
=
G_{EC}s_{EC}
-
J_{OLM}s_{OLM}
-
J_{PPA}s_{PPA}
-
J_{NGFC}s_{NGFC}.
\]

Then:

\[
I_E
=
\Phi_{dend}
(I_E^{prox},I_E^{dist})
-
J_{PVBC}s_{PVBC}
-
J_{AAC}s_{AAC}.
\]

This preserves the fact that different inhibitory interneurons gate different dendritic input zones.

---

# 8. Neurogliaform / volume-transmission term

Neurogliaform cells may require more than point-to-point synaptic coupling because of dense axonal arborization and GABA spillover.

Introduce an effective spatial spread parameter:

\[
\sigma_i^{GABA}.
\]

A possible spatial kernel is:

\[
K_{ij}
=
\exp
\left[
-\frac{d_{ij}^2}
{2(\sigma_i^{GABA})^2}
\right].
\]

For ordinary basket cells:

\[
\sigma^{GABA}
\]

would be relatively small.

For neurogliaform cells it should be larger.

---

# 9. Delays

Axonal path length should be explicit:

\[
L_{ij}^{axon}.
\]

Define:

\[
\delta_{ij}
=
\delta_{syn}
+
\frac{L_{ij}^{axon}}
{v_j^{cond}}.
\]

Then coupling becomes delayed:

\[
s_j(t-\delta_{ij}).
\]

This may matter substantially in multi-region loops such as:

\[
EC/MEC/LEC
\leftrightarrow
DG
\leftrightarrow
CA3
\rightarrow
CA1
\rightarrow
SUB.
\]

---

# 10. Recommended anatomical parameter set

Each population should eventually have priors for:

| Parameter | Meaning |
|---|---|
| \(L_i^{soma}\) | soma layer / stratum |
| \(D_i^{tot}\) | total dendritic extent |
| \(d_{i\ell}\) | dendritic laminar distribution |
| \(A_i^{tot}\) | axonal extent / bouton abundance |
| \(a_{i\ell c}\) | axonal terminal layer + compartment distribution |
| \(q_{ij}\) | postsynaptic target-cell preference |
| \(\Gamma_{ic}\) | compartment effectiveness |
| \(L_{ij}^{axon}\) | axonal path length |
| \(v_i^{cond}\) | conduction velocity |
| \(\sigma_i^{GABA}\) | transmitter-spread radius |
| \(U_i\) | release scale / probability |
| \(\tau_i^{syn}\) | synaptic kinetics |
| receptor parameters | neuromodulatory sensitivity |

---

# 11. Basket CCK+ versus generic Basket

`Basket CCK+` should be treated as a distinct subtype.

The correct conceptual decomposition is:

\[
\text{cell identity}
=
\text{morphology}
+
\text{target domain}
+
\text{molecular phenotype}
+
\text{region}
+\cdots
\]

For example:

\[
\text{CCK+ Basket}
\neq
\text{CCK+ PPA}
\neq
\text{CCK+ SCA}.
\]

CCK expression does not define basket morphology by itself.

---

## 11.1 Tri-state molecular markers

Markers should be stored as:

\[
M_k
\in
\{
positive,
negative,
unknown
\}.
\]

A bare `Basket` should not be interpreted as CCK-negative merely because `CCK+` is absent from its name.

Thus:

`CA1 Basket`

should initially use:

\[
CCK=unknown,
\qquad
PV=unknown.
\]

unless the original database explicitly defines it otherwise.

---

## 11.2 Dynamic CB1 modulation

For a CCK-positive basket population:

\[
U_i^A(t)
=
U_{i,0}^A
\exp
\left[
-\beta_i^{CB1}
O_{CB1}(t)
\right].
\]

Here:

\[
\text{CCK+ identity}
\]

supports an informative prior on \(\beta_i^{CB1}\), while:

\[
O_{CB1}(t)
\]

is a dynamic neuromodulatory state.

---

# 12. Neuromodulation architecture

Neuromodulators should act on specific mechanisms rather than simply add a generic current.

For receptor/modulator \(m\):

\[
O_{im}
=
\frac{M_m^{n_m}}
{K_{im}^{n_m}+M_m^{n_m}}.
\]

Intrinsic excitability:

\[
\eta_i^{eff}
=
\eta_i^0
+
\sum_m
\alpha_{im}^{\eta}
O_{im}.
\]

Presynaptic release:

\[
U_i^{eff}
=
U_i^0
\exp
\left[
\sum_m
\alpha_{im}^{rel}
O_{im}
\right].
\]

Possible modeled pathways include:

- CB1/endocannabinoid,
- muscarinic M1/M4,
- nicotinic acetylcholine,
- serotonin 5-HT3A / 5-HT2A / 5-HT1-like effects,
- mu-opioid,
- dopamine receptor effects,
- beta-adrenergic modulation,
- O-LM \(I_h\) modulation.

Uncertain receptor-cell-type associations should retain broad priors rather than be hard-coded.

---

# 13. O-LM / HIPP example

An O-LM-like population can be represented by:

- soma/dendrites concentrated in oriens,
- axon concentrated in lacunosum-moleculare,
- distal pyramidal dendrite targeting,
- slower membrane dynamics,
- SST/mGluR-related identity priors where supported,
- optional \(I_h\) state.

Optional HCN state:

\[
\tau_h(V)\dot h
=
h_\infty(V)-h,
\]

\[
I_h
=
g_h^{eff}h(E_h-V).
\]

Possible neuromodulation:

\[
g_h^{eff}
=
g_h^0
\exp
(
\alpha_{NE}O_{\beta AR}
-
\alpha_{\mu}O_\mu
).
\]

---

# 14. Bifurcation formulation

Equilibria satisfy:

\[
F(x,\lambda)=0.
\]

Local stability is determined by:

\[
J_x
=
\frac{\partial F}{\partial x}.
\]

Let:

\[
\alpha
=
\max_k
\Re(\lambda_k(J_x)).
\]

Then:

- \(\alpha<0\): locally stable equilibrium,
- real eigenvalue crossing zero: zero-eigenvalue / fold-like candidate,
- complex conjugate pair crossing imaginary axis: Hopf-like candidate.

---

## 14.1 Pseudo-arclength continuation

Ordinary parameter stepping fails at folds.

Use:

\[
z=(x,\lambda)
\]

and solve the augmented system:

\[
\begin{cases}
F(x,\lambda)=0,\\
t^\top(z-z_{pred})=0.
\end{cases}
\]

Augmented Jacobian:

\[
\begin{bmatrix}
J_x & \partial_\lambda F\\
t^\top
\end{bmatrix}.
\]

This allows continuation through folds.

---

# 15. Useful continuation parameters

Possible \(\lambda\) choices:

## Global coupling

\[
\lambda=g.
\]

## Regional/layer-specific drive

Examples:

- superficial EC/MEC/LEC excitatory drive,
- MEC layer-II drive,
- CA1 pyramidal drive.

## Pathway gain

Examples:

\[
g_{CA3\rightarrow CA1},
\]

\[
g_{DG\rightarrow CA3},
\]

\[
g_{EC\rightarrow DG}.
\]

## Neuromodulator occupancy

Examples:

\[
O_{CB1},
\quad
O_{M1},
\quad
O_\mu,
\quad
O_{5HT3A}.
\]

## Morphological continuation

Potential future analyses:

- O-LM distal axonal fraction,
- PV perisomatic targeting strength,
- NGFC spatial spread \(\sigma^{GABA}\),
- dendritic overlap between EC inputs and target populations,
- axonal delay.

---

# 16. Critical eigenmode analysis

At a bifurcation:

\[
J_x q
=
\mu q.
\]

The vector \(q\) indicates which populations dominate the instability.

For rate components:

\[
|q_{r_i}|
\]

can be ranked by:

- region,
- layer,
- cell type,
- E/I class,
- molecular subtype.

This makes the bifurcation anatomically interpretable.

The goal is not only to report:

“Hopf at \(\lambda=\lambda_H\)”

but to identify which cell populations and regions participate in the critical mode.

---

# 17. Knowledge-base architecture

The recommended database is relational (SQLite or PostgreSQL initially).

The database should be a **cell-type knowledge base plus parameter-prior engine**, not a table of final fixed numbers.

Recommended entities:

| Table/entity | Purpose |
|---|---|
| `cell_type` | exact netlist name and unique ID |
| `taxonomy` / parent relation | family/subtype hierarchy |
| `region` | DG, CA3, CA3c, CA2, CA1, EC, MEC, etc. |
| `layer` | explicit or weighted layer membership |
| `morphology` | Basket, AAC, O-LM, HIPP, NGFC, etc. |
| `axon_target` | AIS, soma, proximal/distal dendrite, interneuron |
| `pathway` | PP, MF, recurrent CA3, SC, TA, etc. |
| `molecular_marker` | PV, SST, CCK, CB1, NPY, reelin, 5-HT3A, etc. |
| `intrinsic_property` | \(\tau_m\), \(R_{in}\), adaptation, AP width, firing rate |
| `synaptic_property` | GABA-A, GABA-B, STP, release probability, delay |
| `neuromodulation` | ACh, 5-HT, dopamine, NE, opioid, CB1 effects |
| `model_parameter` | model parameter definitions |
| `parameter_prior` | uncertainty distributions for model parameters |
| `evidence` | DOI/PMID/source, species, age, method, context |
| `property_claim` | property-value-context-source records |
| `model_estimate` | fitted values kept distinct from experimental claims |

---

# 18. Claims rather than final values

The database should store claims of the form:

\[
(
cell,
property,
value,
source,
confidence,
context
).
\]

Example:

```text
cell: CA1 Basket CCK+
property: axon_target
value: perisomatic
source: Pelkey et al.
confidence: high
species: rodent
context: hippocampus
```

Another claim:

```text
cell: CA1 Basket CCK+
property: CB1_effect
value: decreases_GABA_release
mechanism: presynaptic_CB1
confidence: high
```

This allows:

- multiple studies,
- disagreements,
- species dependence,
- preparation dependence,
- uncertainty.

---

# 19. Inheritance

The database should support hierarchical inheritance.

Example:

\[
\text{Basket}
\Rightarrow
\text{primarily perisomatic targeting}
\]

can be inherited by:

- CA1 Basket,
- CA1 Basket CCK+,
- CA3 Basket,
- CA3 Basket CCK+,

with some confidence.

But:

\[
\text{CCK+}
\Rightarrow
\text{strong CB1 prior}
\]

should only be applied when CCK positivity is known or independently supported.

A generic CA3 Basket should not inherit CCK-specific modulation.

---

# 20. Prior generation

For model parameter \(\theta_j\):

\[
\theta_j
\sim
p(
\theta
\mid
R_j,
F_j,
L_j,
M_j,
A_j
),
\]

where:

- \(R_j\): region,
- \(F_j\): morphological family,
- \(L_j\): layer,
- \(M_j\): molecular identity,
- \(A_j\): axonal target.

If only region and family are known:

\[
p(\theta_j)
=
p(\theta\mid R_j,F_j).
\]

If molecular identity is also known:

\[
p(\theta_j)
=
p(\theta\mid R_j,F_j,M_j).
\]

General rule:

\[
\boxed{
\text{less annotation}
\Rightarrow
\text{broader prior}
}
\]

rather than:

\[
\text{less annotation}
\Rightarrow
\text{invented subtype}.
\]

---

# 21. Confidence-dependent uncertainty

If \(c_j\) is confidence in an inherited claim, a prior variance can be widened automatically:

\[
\sigma_{\theta,j}^2
=
\sigma_{\min}^2
+
(1-c_j)
\sigma_{\rm uncertainty}^2.
\]

This allows direct integration with Bayesian fitting.

---

# 22. Compound layer representation

Compound labels such as `LI-II` should ideally be represented via an association table rather than a single categorical string.

Example:

\[
P(L=LI)=q_1,
\qquad
P(L=LII)=q_2.
\]

If no more information is available:

\[
q_1=q_2=0.5.
\]

Anatomical data can later update these weights without renaming the cell type.

---

# 23. Region hierarchy

CA3 versus CA3c should be hierarchical.

Example:

Generic CA3 population:

```text
region_parent = CA3
region_subdivision = NULL
```

CA3c-specific population:

```text
region_parent = CA3
region_subdivision = CA3c
```

The former inherits CA3-wide priors but not CA3c-specific priors.

---

# 24. Existing implementation artifacts

The following files were created during the conversation.

## Neural-mass / bifurcation notebooks

### `ca1_interneuron_bifurcation.ipynb`

Initial CA1-focused prototype containing:

- MPR/QIF mass equations,
- adaptation,
- GABA-A/GABA-B states,
- cell-type couplings,
- simple neuromodulation,
- numerical equilibrium solving,
- analytic/numerical Jacobian logic,
- pseudo-arclength continuation,
- fold/Hopf candidate detection.

### `hippocampal_netlist_region_layer_bifurcation.ipynb`

Netlist-resolved extension containing:

- one mass per netlist population,
- region parsing,
- cortical-layer parsing,
- pathway-gain continuation,
- region/layer selectors,
- analytic high-dimensional Jacobian,
- critical eigenmode analysis.

### `hippocampal_netlist_region_layer_bifurcation_v2_identity_modulation.ipynb`

Revision separating:

- stable identity,
- tri-state markers,
- CCK-positive basket subtype,
- unresolved basket subtype,
- dynamic receptor states.

---

## Knowledge-base artifacts

### `hippocampal_celltype_kb.sqlite`

Prototype SQLite knowledge base.

Seeded with representative examples such as:

- CA1 Basket,
- CA1 Basket CCK+,
- CA3 Basket,
- CA3 Basket CCK+.

### `hippocampal_celltype_kb_schema.sql`

Relational schema defining:

- region,
- layer,
- cell type,
- layer membership,
- morphology,
- axon target,
- pathway,
- molecular marker,
- evidence,
- property definitions,
- property claims,
- neuromodulation,
- model parameters,
- parameter priors,
- model estimates.

### `celltype_kb_prior_engine.py`

Prototype prior-resolution engine.

Conceptually supports:

```python
kb.resolve_prior("CA1 Basket CCK+", "beta_CB1")
```

and returns:

- distribution,
- distribution parameters,
- confidence,
- basis,
- specificity rank.

---

# 25. Example prior behavior

Illustrative test priors were used to verify inheritance logic.

For example:

\[
\text{CA1 Basket CCK+}:
\qquad
\beta_{CB1}
\sim
\mathcal N(1.8,0.35^2)
\]

with relatively high confidence.

For an unresolved bare basket cell:

\[
\text{CA1 Basket}:
\qquad
\beta_{CB1}
\sim
\mathcal N(0,1.5^2)
\]

with low confidence.

These numbers were **illustrative software tests**, not final biological estimates.

The conceptual point is:

- explicit CCK positivity → narrower/informative CB1 prior,
- unresolved molecular identity → broad prior.

---

# 26. Model fitting hierarchy

Recommended fitting order:

1. fit intrinsic cellular parameters,
2. fit local synaptic kinetics/strengths,
3. constrain morphology / anatomical overlap,
4. fit neuromodulatory perturbation effects,
5. fit network-level oscillatory observables,
6. propagate parameter uncertainty into bifurcation locations.

Potential observables:

- theta frequency,
- gamma frequency/power,
- theta-gamma coupling,
- ripple recruitment,
- interneuron phase preference,
- perturbation responses,
- region-to-region phase relationships.

---

# 27. Key modeling cautions

## Do not double-count anatomy

If netlist \(m_{ij}\) already includes:

- connection probability,
- bouton count,
- anatomical overlap,
- target-domain factors,

then those factors should not be multiplied into \(J_{ij}\) again.

Before adding morphology terms, determine how \(m_{ij}\) was constructed.

---

## Do not infer molecular phenotype from morphology alone

Examples:

- Basket does not imply PV-positive.
- Lack of `CCK+` does not imply CCK-negative.
- CCK positivity does not imply basket morphology.
- SST positivity does not uniquely imply O-LM.
- O-LM morphology can strongly inform axonal target while receptor profile may remain uncertain.

---

## Do not overinterpret demonstration bifurcations

Prototype bifurcation locations are useful for testing numerical machinery, but they are not biological predictions until:

- coupling scales are calibrated,
- intrinsic priors are constrained,
- synaptic kinetics are validated,
- receptor coefficients are evidence-based,
- uncertainty is propagated.

---

# 28. Key biological source

Primary review repeatedly used in the discussion:

Pelkey KA, Chittajallu R, Craig MT, Tricoire L, Wester JC, McBain CJ.  
**Hippocampal GABAergic Inhibitory Interneurons.**  
*Physiological Reviews*. 2017;97:1619-1747.  
doi:10.1152/physrev.00007.2017

Important topics in that review include:

- hippocampal interneuron anatomy,
- developmental origins,
- intrinsic electrophysiology,
- glutamate receptors,
- GABA receptors,
- endocannabinoids,
- acetylcholine receptors,
- dopamine receptors,
- serotonin receptors,
- opioid receptors,
- network oscillations,
- disease relevance.

Additional modeling references discussed:

- Montbrió, Pazó, Roxin: exact QIF mean-field / next-generation neural mass framework.
- Bezaire / Soltesz-related CA1 modeling work: detailed hippocampal cell-type and connectivity modeling.

---

# 29. Recommended next steps

## A. Populate every exact netlist node into the knowledge base

Every exact netlist name should become a stable database key.

Example:

```text
CA1 Basket CCK+
CA1 Basket
CA1 O LM
CA3 Basket
CA3 Basket CCK+
...
```

---

## B. Trace original definitions of ambiguous bare cell types

Especially:

- `Basket`,
- `Basket CCK+`.

Determine whether bare `Basket` means:

- PV basket,
- untyped basket,
- a database-specific morphological category.

Do not assume.

---

## C. Add curated soma-stratum and axonal-target data

For hippocampal interneurons add:

- soma stratum,
- dendritic layer fractions,
- axonal layer fractions,
- postsynaptic compartment fractions,
- target cell class,
- total axonal/dendritic extent where available.

---

## D. Connect the ODE notebook to the database

The ODE should stop containing hard-coded biological prior dictionaries.

Instead:

\[
\Theta_i
=
\mathrm{KB.resolve}(cell_i).
\]

The returned object should include:

- resolved value/prior,
- uncertainty,
- inheritance level,
- source,
- confidence,
- context.

---

## E. Carry uncertainty into bifurcation analysis

Rather than one deterministic branch, evaluate:

\[
p(\lambda_H),
\qquad
p(\lambda_F),
\]

for Hopf/fold locations across parameter-prior samples.

This yields uncertainty-aware stability diagrams.

---

## F. Build two-parameter stability maps

Examples:

\[
(I_{EC},O_{CB1}),
\]

\[
(g_{CA3\rightarrow CA1},O_{CB1}),
\]

\[
(g_{DG\rightarrow CA3},g_{CA3\rightarrow CA1}),
\]

\[
(A_{OLM}^{distal},I_{EC}),
\]

\[
(\sigma_{NGFC}^{GABA},g_{EC\rightarrow CA1}).
\]

---

# 30. Long-term architecture

The intended final architecture is:

\[
\boxed{
\begin{array}{c}
\text{Literature / Hippocampome / measurements}\\
\downarrow\\
\text{evidence-backed cell-type knowledge base}\\
\downarrow\\
\text{hierarchical parameter-prior resolver}\\
\downarrow\\
\text{85-population netlist neural-mass model}\\
\downarrow\\
\text{simulation / bifurcation / sensitivity / Bayesian fitting}\\
\downarrow\\
\text{posterior model estimates}\\
\downarrow\\
\text{stored back as separate model-estimate evidence}
\end{array}
}
\]

Measured experimental facts, inherited biological generalizations, and fitted model estimates should remain distinct evidence classes.

---

# 31. Concise design rules for future work

1. Preserve exact netlist populations.
2. Parse region/layer/projection metadata separately from cell-type name.
3. Treat molecular markers as tri-state: positive, negative, unknown.
4. Treat `Basket CCK+` as a distinct subtype.
5. Never assume bare `Basket` is PV unless the source explicitly says so.
6. Make dendritic and axonal structure explicit quantitative parameters.
7. Use morphology to modify pathway overlap and compartment targeting.
8. Use receptor identity to define priors; use receptor occupancy as a dynamic state.
9. Avoid double-counting anatomy already embedded in the netlist.
10. Store claims with evidence/context/confidence rather than only final numbers.
11. Let less annotation produce broader priors.
12. Keep fitted values separate from experimental measurements.
13. Use pseudo-arclength continuation for folds.
14. Analyze critical eigenvectors to identify which region/layer/cell types drive instabilities.
15. Report uncertainty in bifurcation boundaries.

---

# 32. Project status at end of archived conversation

Completed conceptually:

- cell-type-resolved neural-mass framework,
- region/layer parsing strategy,
- full-netlist bifurcation architecture,
- pseudo-arclength continuation workflow,
- identity versus neuromodulation distinction,
- explicit morphology proposal,
- database / evidence / inheritance architecture,
- prototype SQLite schema,
- prototype prior resolver.

Still to do:

- ingest all exact netlist nodes into the database,
- ingest the actual netlist edge table,
- trace ambiguous source taxonomy,
- populate evidence-backed morphology and receptor properties,
- connect notebook parameter retrieval directly to the database,
- calibrate netlist-to-neural-mass coupling scale,
- perform uncertainty-aware continuation,
- validate against hippocampal oscillation data.

---

# End of knowledge archive
