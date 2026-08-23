# Inhibitory Modulation Analysis

This project contains a staged notebook workflow for analyzing inhibitory modulation in the signed hippocampal connectivity matrix.

## Notebook Run Order

Run the notebooks in this order:

1. `notebooks/01_inhibitory_modulation_backbone_workflow.ipynb`

   Foundational workflow. Loads and validates the signed connectome, confirms orientation and E/I structure, removes self-connections for the backbone analysis, normalizes connection strengths, defines backbone geometry, simulates propagation, and measures baseline stability, transient amplification, inhibitory strength effects, neuron ablations, edge ablations, motif geometry, and randomized inhibitory nulls.

2. `notebooks/02_inhibitory_modulation_analysis.ipynb`

   Main mechanistic fracture analysis. Uses the same matrix conventions to compare `with_self` and `no_self` variants, then runs EE/EI/IE/II block removals, Schur reduction, resolvent responses, motif-removal stability, Neumann chain-depth selection, candidate disinhibitory relays, and explicit `E-I-E` / `E-I-I-E` path rankings.

3. `notebooks/03_inhibitory_two_edge_triad_sensitivity.ipynb`

   Local motif companion and sensitivity analysis. Consolidates the previous self-free and self-inclusive triad notebooks into one workflow. Use the `self_free` variant as the primary two-edge triad census, and use `with_self_connections` to test how diagonal terms affect weight rankings, edge participation, ablation, aggregation, and Schur-style reductions.

## Analysis Flow

The intended flow is conceptual rather than file-dependent: each notebook reloads the source matrix and calls shared helper functions instead of requiring outputs from the previous notebook. The recommended narrative is:

1. Establish the data, orientation, E/I labels, normalization, and baseline dynamics.
2. Apply coarse global fractures by removing EE, EI, IE, and II blocks.
3. Interpret inhibitory effects through Schur reduction, resolvent response, Neumann expansion, and explicit paths.
4. Test targeted fractures through inhibitory neuron, inhibitory edge, and motif-class ablations.
5. Use the consolidated triad notebook as a downstream local motif explanation and self-connection sensitivity check.

## Inputs And Outputs

Primary inputs:

- `matrices/mij_matrix.csv`
- `matrices/mij_netlist.csv`

Primary output folders:

- `outputs/inhibitory_modulation_backbone_workflow/`
- `outputs/inhibitory_modulation/`
- `outputs/inhibitory_two_edge_triad_sensitivity/`

The notebooks are intended to be run top-to-bottom after any change to the input matrix or helper functions.
