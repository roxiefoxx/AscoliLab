# schur_decomp

`schur_decomp` is a reproducible analysis suite for studying directed hippocampal-entorhinal cell-type connectivity with Schur decomposition, modal signal propagation, inhibitory modulation, and linear-systems tools.

The project is organized around a single matrix convention:

- The raw `Mij` matrix is stored as source cell types in rows and receiver cell types in columns.
- Analyses convert this to a state-update matrix `M[receiver, source]`, so that `x[t + 1] = M @ x[t]`.
- This orientation matters because reversing source and receiver roles changes the biological interpretation of modes, paths, transfer channels, and perturbations.

Canonical inputs:

- `matrices/mij_matrix.csv`
- `matrices/mij_netlist.csv`

Generated outputs are under `outputs/`.

## Recommended notebook order

The active notebooks are numbered in the intended reading/rerun order.

| Notebook | Purpose |
|---|---|
| `01_schur_eigenmodes.ipynb` | Foundational corrected Schur/eigenmode analysis: orientation, normalization, eigenvalue spectrum, dominant eigenmode, Schur coupling map, and transient dynamics. |
| `02_schur_eigenmodes_compare.ipynb` | Matched normalized comparison of eigenmodes with self-connections versus no self-connections. |
| `03_schur_signal.ipynb` | Main Schur signal-propagation analysis across modes, cell types, and anatomical regions. |
| `04_schur_signal_compare.ipynb` | Matched normalized with-self versus no-self comparison for signal propagation. |
| `05_schur_greedy_trees.ipynb` | Top-contributor subnetworks and greedy information-flow trees for each Schur mode. |
| `06_schur_mode_case_studies.ipynb` | Synthesis bridge: selects interpretable Schur modes by combining propagation summaries and greedy-tree structure. |
| `07_inhibitory_modulation.ipynb` | Inhibitory modulation analysis using E/I blocks, Schur complements, perturbations, motifs, and Neumann-chain summaries. |
| `08_inhibitory_compare.ipynb` | Matched normalized with-self versus no-self comparison for inhibitory modulation. |
| `09_linear_systems.ipynb` | Linear-systems companion analysis: matrix-power paths, controllability, and frequency response. |
| `10_linear_systems_compare.ipynb` | Matched normalized with-self versus no-self comparison for linear-systems results. |

## Method rationale

### Corrected Schur decomposition

The core method is a complex Schur decomposition of the oriented state matrix:

```text
A = Q T Q*
```

The diagonal of `T` contains eigenvalues, while the upper-triangular entries preserve mode-to-mode coupling. This is useful because the Mij network is directed and strongly non-normal. In non-normal systems, eigenvalues alone do not fully describe transient amplification or directional coupling. Schur coordinates are numerically stable and keep the cascade structure visible.

### Spectral-radius normalization

Several analyses use spectral-radius normalization so that the largest eigenvalue magnitude is set to a target value:

- `0.95` for eigenmode comparison, Schur signal propagation, greedy mode decomposition, and linear-systems comparisons.
- `0.85` for inhibitory stable-response analyses.

This makes with-self and no-self variants dynamically comparable while keeping each variant stable enough for propagation, resolvent, Neumann-series, and Gramian calculations.

### With-self versus no-self comparisons

The comparison notebooks zero the diagonal before normalization to test whether conclusions depend on self-connections. The with-self and no-self matrices are normalized independently. This isolates the effect of removing diagonal connections, but results should be interpreted as matched normalized dynamics rather than raw-weight comparisons.

### Schur signal propagation

Schur signal propagation seeds individual Schur modes and propagates them through the triangular Schur cascade. This turns abstract modal structure into interpretable cell-type and region-level summaries.

### Inhibitory Schur complement

The inhibitory analysis partitions the matrix into excitatory and inhibitory blocks and computes effective excitatory dynamics after accounting for inhibitory feedback. This is used to ask how inhibitory populations, E-I/E-I-I-E motifs, and disinhibitory chains alter stability and excitatory propagation.

### Linear-systems companion analysis

The linear-systems notebooks use matrix powers, controllability Gramians, and frequency response. These provide complementary views of reachability, state diffusion, and input-output gain that do not require staying exclusively in Schur coordinates.

## Unique analyses and headline results

### 1. Schur/eigenmode foundation

The canonical matrix contains 85 cell types. Based on the netlist/inferred metadata, the active matrix has 32 excitatory and 53 inhibitory cell types.

The raw oriented matrix has a dominant spectral radius of approximately `26213.13`. After spectral-radius normalization to `0.95`, the corrected Schur decomposition reconstructs the matrix with near machine-precision error.

The with-self versus no-self eigenmode comparison found:

- Both variants were normalized to spectral radius `0.95`.
- Dominant vector overlap was approximately `0.901`, meaning the dominant modal direction is broadly preserved after removing self-connections.
- Removing self-connections increased non-normality from approximately `5869.90` to `18319.83`, suggesting a stronger potential for transient amplification in the no-self network.

### 2. Schur signal propagation

The propagation analysis generated all-mode cell-type signal tables, mode-loading summaries, and region summaries.

Key results:

- CA1 dominated the most Schur modes: 26 of 85 modes.
- CA3 and DG each dominated 15 modes.
- MEC dominated 12 modes.
- High-loading cells were often inhibitory; CA1-dominant modes averaged about `25.46` high-loading inhibitory cells, CA3 modes averaged `27.67`, and DG modes averaged `26.20`.

With-self versus no-self propagation comparison:

| Variant | Raw spectral radius | Achieved spectral radius | Matrix Frobenius norm | Non-normality |
|---|---:|---:|---:|---:|
| with_self | 26213.13 | 0.95 | 77.18 | 5869.90 |
| no_self | 14837.87 | 0.95 | 136.35 | 18319.83 |

At timestep 10, total propagated signal was higher in the no-self variant (`39.48`) than the with-self variant (`17.35`). This suggests that after matched normalization, removing self-connections leaves a more strongly propagating off-diagonal network.

### 3. Schur-mode greedy trees

The greedy-tree analysis builds top-7 contributor subnetworks for each Schur mode and extracts compact information-flow trees.

Key results:

- The dominant normalized mode has eigenvalue magnitude `0.95`.
- Its strongest contributor is `MEC LIII Superficial Multipolar Interneuron`.
- Across 85 modes, greedy trees averaged about `4.26` selected tree edges.
- The trees averaged about `2.74` connected components, showing that many modal contributor sets are not explained by a single connected source tree.
- Leading modes were dominated by inhibitory contributors such as MEC LIII superficial multipolar interneuron, DG AIPRIM, CA1 trilaminar, and DG HICAP.

### 4. Schur-mode case studies

The case-study notebook combines the mode summaries from `03_schur_signal.ipynb` with the greedy-tree summaries from `05_schur_greedy_trees.ipynb`.

This notebook is designed as a bridge rather than a new decomposition method. It selects a small, auditable set of modes for closer discussion:

- the largest-eigenvalue mode;
- the strongest dominant-loading examples in major regions such as CA1, CA3, DG, and MEC;
- a compact integrated mode with a single greedy-tree component;
- an inhibitory-dense mode with many high-loading inhibitory cells.

The resulting table is saved to `outputs/schur_mode_case_studies/case_study_modes.csv` and can be used for figures, prose summaries, or advisor-facing interpretation.

### 5. Inhibitory modulation

The inhibitory analysis uses spectral-radius normalization to `0.85`.

Key results from the self-comparison:

- Both with-self and no-self variants achieved dominant real eigenvalue `0.85`.
- Removing the E-to-E block strongly reduced the dominant real eigenvalue:
  - with-self: `0.85 -> 0.063`
  - no-self: `0.85 -> 0.099`
- Removing I-to-I had a much smaller effect:
  - with-self: `0.85 -> 0.844`
  - no-self: `0.85 -> 0.838`
- Removing cross E/I blocks increased the dominant real eigenvalue:
  - with-self: `0.85 -> 0.886`
  - no-self: `0.85 -> 0.996`

Interpretation: E-to-E structure carries the main excitatory growth mode, while cross E/I pathways appear stabilizing. When E/I cross-coupling is removed, the excitatory block becomes more unstable, especially in the no-self variant.

The Schur-complement analysis showed:

- with-self `M_EE` spectral radius: `0.886`
- with-self effective excitatory matrix `M_eff_E` spectral radius: `0.850`
- no-self `M_EE` spectral radius: `0.996`
- no-self effective excitatory matrix `M_eff_E` spectral radius: `0.850`

This indicates that inhibitory feedback brings the effective excitatory dynamics back to the target stability level.

Motif enrichment results were strong for disinhibitory motifs:

- `E_to_I_to_I_to_E_loop` had z-scores of about `51.96` with self-connections and `51.60` without self-connections.
- `disinhibitory_I_to_I_to_E` had z-scores of about `27.53` with self-connections and `28.03` without self-connections.

Top disinhibitory sources were stable across variants:

- `CA1 Perforant Path Associated`
- `MEC LIII Superficial Multipolar Interneuron`
- `DG AIPRIM`

### 6. Linear-systems analysis

The linear-systems comparison analyzes structural paths, controllability, and frequency response after normalizing both variants to spectral radius `0.95`.

Key results:

| Variant | Cumulative path Frobenius norm | Gramian trace | Gramian condition | Peak frequency | Peak gain |
|---|---:|---:|---:|---:|---:|
| with_self | 131.96 | 9546.60 | 5767.20 | 0.01 | 69.39 |
| no_self | 323.87 | 56245.57 | 42740.43 | 0.01 | 193.43 |

The no-self matrix shows stronger path accumulation, larger controllability energy, worse conditioning, and higher low-frequency gain after matched normalization.

Top outgoing path-score cells:

- with-self: `CA3c Pyramidal`, `EC LI II Multipolar Pyramidal`, `CA3 Pyramidal`
- no-self: `EC LI II Multipolar Pyramidal`, `CA3c Pyramidal`, `CA3 Pyramidal`

Top controllability-energy cells:

- with-self: `CA1 Trilaminar`, `CA1 Perforant Path Associated QuadD`, `MEC LIII Superficial Multipolar Interneuron`
- no-self: `CA1 Trilaminar`, `CA1 Perforant Path Associated QuadD`, `CA1 Perforant Path Associated`

## Code organization

The project follows a clean notebook convention: notebooks hold the readable analysis, while long functions live in external scripts.

| Script | Status | Role |
|---|---|---|
| `schur_core_script.py` | Keep | Shared project core: Mij loading, orientation, E/I inference, label formatting, region parsing, spectral summaries, normalization, and with-self/no-self matrix variants. |
| `schur_analysis_utils_script.py` | Keep | Schur/eigenmode-specific utilities: corrected decomposition, validation, mode ordering, dominant-mode summaries, coupling tables, plots, and transient dynamics. |
| `schur_signal_propagation_script.py` | Keep | Schur signal-propagation model preparation, seeding, propagation, mode loadings, region summaries, and exports. |
| `schur_mode_greedy_trees_script.py` | Keep | Top-contributor subnetworks and greedy flow-tree construction. |
| `inhibitory_schur_modulation_script.py` | Keep | E/I block analysis, Schur complement, perturbations, motif enrichment, Neumann chains, and inhibitory comparison outputs. |
| `network_linear_systems_tools_script.py` | Keep | Matrix-power path analysis, controllability, frequency response, and linear-systems comparison utilities. |
| `archive/individual_analyses/schur_analysis_utils_script.py` | Redundant / archived | Older helper copy for the superseded individual with-self/no-self notebooks. Keep archived for provenance only; do not use for active analysis. |

## Redundant scripts and cleanup notes

There are no redundant active root-level Python scripts at this point. Each active `*_script.py` file supports a distinct analysis layer.

Redundant or generated items to ignore:

- `archive/individual_analyses/schur_analysis_utils_script.py` is redundant because `schur_analysis_utils_script.py` and `schur_core_script.py` now cover the active workflow.
- `__pycache__/` folders are generated Python bytecode caches and can be deleted/recreated safely.
- `archive/outputs/inhibitory_schur_modulation_script/` contains the older inhibitory output folder from a previous script naming convention. The active notebooks now use `outputs/inhibitory_schur_modulation/`, including its `self_comparison/` subfolder.
- `outputs/moved_csv/` contains non-canonical CSV files moved out of the project root and `matrices/` folder during cleanup. These are retained for provenance but are not canonical inputs.

## Reproducibility notes

The active notebooks were previously executed successfully after module consolidation. The project assumes a Python environment with NumPy, pandas, SciPy, matplotlib, seaborn, Jupyter, and IPython display support.

To rerun the project in order, open the notebooks from `01_...` through `10_...`.

## Interpretation cautions

- Schur modes are mathematical coordinates. Biological interpretation should focus on robust loadings, region summaries, and cross-analysis consistency rather than a single complex mode in isolation.
- Spectral-radius normalization makes dynamic comparisons stable and comparable, but it changes absolute weight scale.
- With-self and no-self matrices are normalized independently, so comparisons describe normalized dynamics after diagonal removal, not raw diagonal mass alone.
- The no-self variant repeatedly shows higher non-normality, stronger propagated signal, larger linear-system gain, and larger Gramian trace after matched normalization. This is a strong pattern, but it should be interpreted as a structural/dynamical consequence of redistributing influence into off-diagonal pathways under normalization.
