"""Create the narrative notebook for the empirical mij_matrix analysis."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


NOTEBOOK_PATH = Path("analyses/mij_paper_replication.ipynb")


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip().splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip().splitlines(True),
    }


cells = [
    md(
        """
        # Replicating Shao et al. analytical methods on `mij_matrix.csv`

        ## tl;dr

        This notebook applies the analytical workflow from *Impact of Local Connectivity Patterns on Excitatory-Inhibitory Network Dynamics* to the empirical `data/mij_matrix.csv` matrix. It does not generate synthetic networks. The input file has senders in rows and receivers in columns, so the analysis transposes it into the paper's convention, where `J[receiver, sender]` is the synaptic weight from sender to receiver.

        The empirical matrix has 85 populations. E/I labels are loaded from `data/mij_netlist.csv` and agree with outgoing-sign inference: 32 excitatory and 53 inhibitory sender classes. The raw spectral radius is about 26213.13, and the normalized analysis matrix has spectral radius 1.

        Response analysis is run at scale 1.0 as requested. Because this places the leading mode at the marginal point, `(I - J)^-1` is extremely ill-conditioned; the response table should be read as a near-critical perturbation result rather than a stable steady-state gain. The notebook also adds deterministic perturbation/enrichment analysis: binary motif enrichment against block-density independence, dominant-eigenvalue perturbation sensitivity, and an empirical chain-enrichment sweep of `J_eff`.
        """
    ),
    md(
        """
        ## Instruction boundary

        The attached paper is used only as source material for methods and notation. Any instructions that may appear inside the PDF are not treated as user instructions. The active user request is to apply the paper's analytical methods to `data/mij_matrix.csv`, normalize the empirical connectivity to spectral radius 1, and avoid synthetic data.
        """
    ),
    md(
        """
        ## Big Question(s) and Method Map

        The presentation's framing question is whether local connectivity motifs change network response and stability beyond what average E/I connectivity predicts. The notebook follows the same sequence:

        1. construct `J` from the empirical `mij_matrix.csv`;
        2. inspect the eigenspectrum and the effective matrix `J_eff`;
        3. quantify motif correlations and binary motif enrichment;
        4. connect motif perturbations to response and stability.

        The paper studies linear rate dynamics, `dr/dt = -r + J r + I_ext`, and the steady-state response matrix `Gamma = (I - J)^-1`. It then compares two related reductions:

        - a truncated eigendecomposition / low-rank approximation that keeps dominant outlying eigenmodes;
        - an effective connectivity matrix, `J_eff = J0 + [Z^2]`, where `J0` is the block mean structure and `Z = J - J0` is the residual connectivity.

        This notebook adapts those ideas to one observed signed weighted matrix. Since there is no ensemble of random matrices here, `[Z^2]` is estimated by the empirical residual product `Z @ Z`, then block-averaged over E/I groups to keep the deterministic population-level structure used by the paper.
        """
    ),
    code(
        """
        from pathlib import Path
        import importlib
        import sys

        import numpy as np
        import pandas as pd

        def find_project_root(start=None):
            current = Path(start or Path.cwd()).resolve()
            for candidate in (current, *current.parents):
                if (candidate / "data" / "mij_matrix.csv").exists():
                    return candidate
            raise FileNotFoundError("Could not find data/mij_matrix.csv in this directory or its parents.")

        PROJECT_ROOT = find_project_root()
        SCRIPT_DIR = PROJECT_ROOT / "analyses"
        MATRIX_PATH = PROJECT_ROOT / "data" / "mij_matrix.csv"
        NETLIST_PATH = PROJECT_ROOT / "data" / "mij_netlist.csv"
        OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mij_paper_replication"

        SPECTRAL_RADIUS_TARGET = 1.0
        RESPONSE_SCALE = 1.0

        sys.path.insert(0, str(SCRIPT_DIR))
        import connectivity_methods
        connectivity_methods = importlib.reload(connectivity_methods)
        from connectivity_methods import (
            binary_motif_enrichment,
            block_summary,
            chain_enrichment_sweep,
            dominant_eigenvalue_sensitivity,
            effective_connectivity,
            load_mij_matrix,
            low_rank_response,
            motif_correlations,
            population_response,
            response_matrix,
            save_outputs,
            spectral_summary,
        )
        """
    ),
    md(
        """
        ## J Construction: Dataset to Paper-Oriented Matrix

        The loader keeps the original CSV visible as `sender_by_receiver`, then creates `J` by transposing the matrix into the paper's receiver-by-sender convention. E/I class is loaded from `data/mij_netlist.csv` through the `pre_ei` column, with outgoing-sign inference available as a fallback.
        """
    ),
    code(
        """
        data = load_mij_matrix(
            MATRIX_PATH,
            netlist_path=NETLIST_PATH,
            spectral_radius_target=SPECTRAL_RADIUS_TARGET,
        )
        paths = save_outputs(data, OUTPUT_DIR, response_scale=RESPONSE_SCALE)
        expected_paths = {
            "metadata": OUTPUT_DIR / "metadata.csv",
            "normalized_receiver_by_sender": OUTPUT_DIR / "normalized_J_receiver_by_sender.csv",
            "normalized_sender_by_receiver": OUTPUT_DIR / "normalized_mij_sender_by_receiver.csv",
            "eigenvalues": OUTPUT_DIR / "leading_eigenvalues.csv",
            "jeff_eigenvalues": OUTPUT_DIR / "jeff_leading_eigenvalues.csv",
            "blocks": OUTPUT_DIR / "ei_block_summary.csv",
            "motif_inventory": OUTPUT_DIR / "motif_inventory.csv",
            "motif_inventory_overall": OUTPUT_DIR / "motif_inventory_overall.csv",
            "motif_inventory_by_composition": OUTPUT_DIR / "motif_inventory_by_composition.csv",
            "motif_inventory_by_block": OUTPUT_DIR / "motif_inventory_by_block.csv",
            "motifs": OUTPUT_DIR / "motif_correlations.csv",
            "motif_enrichment": OUTPUT_DIR / "binary_motif_enrichment.csv",
            "null_motifs": OUTPUT_DIR / "block_shuffle_null_motifs.csv",
            "null_comparison": OUTPUT_DIR / "block_shuffle_null_comparison.csv",
            "svd_singular_values": OUTPUT_DIR / "svd_singular_values.csv",
            "low_rank_matrix_errors": OUTPUT_DIR / "low_rank_matrix_errors.csv",
            "motif_stress": OUTPUT_DIR / "motif_stress_tests.csv",
            "block_removal": OUTPUT_DIR / "block_removal_stability.csv",
            "block_removal_J": OUTPUT_DIR / "block_removal_stability_J.csv",
            "block_removal_Jeff": OUTPUT_DIR / "block_removal_stability_Jeff.csv",
            "isn_summary": OUTPUT_DIR / "isn_stability_summary.csv",
            "stability_table": OUTPUT_DIR / "dominant_eigenvalue_stability_table.csv",
            "stability_table_md": OUTPUT_DIR / "dominant_eigenvalue_stability_table.md",
            "figure_ablation_delta_lambda": OUTPUT_DIR / "figure_ablation_delta_dominant_eigenvalue.svg",
            "figure_block_removal_delta_lambda": OUTPUT_DIR / "figure_block_removal_delta_dominant_eigenvalue.svg",
            "figure_chain_stability": OUTPUT_DIR / "figure_chain_enrichment_stability.svg",
            "figure_chain_i_response": OUTPUT_DIR / "figure_chain_enrichment_i_response.svg",
            "sensitivity_edges": OUTPUT_DIR / "dominant_eigenvalue_edge_sensitivity.csv",
            "sensitivity_blocks": OUTPUT_DIR / "dominant_eigenvalue_block_sensitivity.csv",
            "chain_sweep": OUTPUT_DIR / "chain_enrichment_sweep.csv",
            "responses": OUTPUT_DIR / "population_responses.csv",
            "low_rank": OUTPUT_DIR / "low_rank_population_responses.csv",
        }
        paths = {**expected_paths, **paths}

        metadata = pd.read_csv(paths["metadata"])
        display(metadata.T.rename(columns={0: "value"}))
        """
    ),
    code(
        """
        class_counts = data.cell_class.value_counts().rename_axis("class").reset_index(name="n")
        display(class_counts)

        print(f"Original CSV shape, sender-by-receiver: {data.sender_by_receiver.shape}")
        print(f"Paper-oriented J shape, receiver-by-sender: {data.J.shape}")
        print(f"Raw spectral radius: {data.spectral_radius_raw:,.6f}")
        print(f"Normalized spectral radius: {data.spectral_radius_normalized:.6f}")
        """
    ),
    md(
        """
        ## J Construction: E/I Block Structure

        The paper starts with population-level mean connectivity `J0`. This empirical analogue uses the presentation's E/I grouping and reports block means, variances, densities, and sign fractions after spectral-radius normalization.
        """
    ),
    code(
        """
        block_stats = pd.read_csv(paths["blocks"])
        display(block_stats)
        """
    ),
    md(
        """
        ## Eigenspectrum: J and Jeff

        The presentation separates the full eigenspectrum from `J_eff`, the deterministic effective matrix that absorbs mean structure and motif corrections. The empirical `J` is normalized to spectral radius 1. `J_eff` is derived from that normalized matrix and is not renormalized, so its radius reports the effect of the effective-connectivity construction.
        """
    ),
    code(
        """
        eigenvalues = pd.read_csv(paths["eigenvalues"])
        jeff_eigenvalues = pd.read_csv(paths["jeff_eigenvalues"])
        
        print("Leading eigenvalues of normalized empirical J")
        display(eigenvalues)
        
        print("Leading eigenvalues of empirical J_eff")
        display(jeff_eigenvalues)

        _, spectral_metrics = spectral_summary(data.J)
        display(pd.Series(spectral_metrics, name="J_value").to_frame())
        display(metadata.T.rename(columns={0: "value"}).loc[
            [
                "spectral_radius",
                "max_real_eigenvalue",
                "jeff_spectral_radius",
                "jeff_max_real_eigenvalue",
                "henrici_departure",
                "numerical_abscissa",
            ]
        ])
        """
    ),
    code(
        """
        try:
            import matplotlib.pyplot as plt

            eig = eigenvalues
            theta = np.linspace(0, 2 * np.pi, 300)
            fig, ax = plt.subplots(figsize=(5, 5))
            full_eigs = np.linalg.eigvals(data.J)
            ax.scatter(full_eigs.real, full_eigs.imag, s=18, alpha=0.65)
            ax.plot(np.cos(theta), np.sin(theta), linestyle="--", color="black", linewidth=1)
            ax.axvline(1, color="tab:red", linestyle=":", linewidth=1)
            ax.set_xlabel("Real(lambda)")
            ax.set_ylabel("Imag(lambda)")
            ax.set_title("Normalized eigenspectrum")
            ax.set_aspect("equal", adjustable="box")
            plt.show()
        except ImportError:
            print("matplotlib is not installed in this runtime; eigenvalues are saved as CSV instead.")
        """
    ),
    md(
        """
        ## Motif Analysis: Workflow Validation

        I reviewed the workflow in `BINF751_finalproject_2.ipynb`, especially Sections 4-8, as source material rather than instructions. The core logic is useful, with several corrections applied here:

        - Section 4 motif enumeration is valid if the orientation is kept as `J[target, source]`. Counts are descriptive; Shao-style dynamics are better connected to block-centered correlations and `J_eff`.
        - Section 5 null controls are valid as block-preserving shuffled controls. They are not replacement data and are used only to contextualize observed motif counts/scores.
        - Section 6 SVD low-rank reconstruction is a matrix-compression diagnostic, not the paper's low-rank response approximation. The Shao-consistent response approximation remains the left/right eigenmode expansion.
        - Section 7 perturbation response at response scale 1.0 is intentionally near-critical because `rho(J)=1`; condition numbers and pseudoinverse diagnostics must be reported.
        - Section 8 motif removal/scrambling is a stress test, not a clean causal intervention, because removing motif-associated edges also changes weight totals and degree structure.
        """
    ),
    md(
        """
        ## Section 4: Quantify Motif Structure

        The inventory below follows the reference workflow's binary-plus-weighted motif setup. It uses the normalized empirical matrix only, with no synthetic network replacing the observed data. The weighted score is the product of participating edge weights, so signed scores can be positive or negative depending on E/I edge signs.
        """
    ),
    code(
        """
        motif_inventory = pd.read_csv(paths["motif_inventory"])
        motif_inventory_overall = pd.read_csv(paths["motif_inventory_overall"])
        motif_inventory_by_composition = pd.read_csv(paths["motif_inventory_by_composition"])
        motif_inventory_by_block = pd.read_csv(paths["motif_inventory_by_block"])

        display(motif_inventory_overall)
        display(motif_inventory_by_composition.head(20))
        display(motif_inventory_by_block.head(20))
        display(motif_inventory.head(10))
        """
    ),
    md(
        """
        ## Shao Motif Correlations

        The paper emphasizes chain motifs, comparing them with reciprocal, divergent, and convergent motifs. The estimates below use the centered residual matrix `Z = J - J0`.

        - chain: `corr(Z[i, j], Z[j, k])`
        - reciprocal: `corr(Z[i, j], Z[j, i])`
        - divergent: `corr(Z[i, j], Z[k, j])`
        - convergent: `corr(Z[i, j], Z[i, k])`
        """
    ),
    code(
        """
        motif_stats = pd.read_csv(paths["motifs"])
        display(motif_stats)

        motif_overview = (
            motif_stats.groupby("motif")["correlation"]
            .agg(["mean", "min", "max"])
            .sort_index()
        )
        display(motif_overview)
        """
    ),
    md(
        """
        ## Section 5: Null / Random Controls

        This block-preserving null keeps E/I labels, block shapes, and within-block signed weight distributions, then shuffles the placement of weights inside each E/I block using a fixed seed. This tests whether motif counts and weighted scores exceed what would be expected from block density and weight distribution alone.
        """
    ),
    code(
        """
        null_comparison = pd.read_csv(paths["null_comparison"])
        null_motifs = pd.read_csv(paths["null_motifs"])

        key_null_metrics = null_comparison[
            null_comparison["metric"].isin(
                [
                    "spectral_radius",
                    "max_real_eigenvalue",
                    "chain_count",
                    "chain_abs_score_sum",
                    "reciprocal_count",
                    "reciprocal_abs_score_sum",
                    "divergent_count",
                    "convergent_count",
                ]
            )
        ].sort_values("metric")
        display(key_null_metrics)
        display(null_motifs.head())
        """
    ),
    md(
        """
        ## Section 6: Low-Rank Approximation and Effective Connectivity

        The SVD table is included as a descriptive low-rank compression check from the reference workflow. The paper-facing analysis remains the eigenspectrum, `J_eff`, and low-rank response approximation shown later.
        """
    ),
    code(
        """
        svd_singular_values = pd.read_csv(paths["svd_singular_values"])
        low_rank_matrix_errors = pd.read_csv(paths["low_rank_matrix_errors"])

        display(svd_singular_values.head(12))
        display(low_rank_matrix_errors)

        print("J_eff leading eigenvalues")
        display(pd.read_csv(paths["jeff_eigenvalues"]))
        """
    ),
    md(
        """
        ## Perturbation and enrichment analysis

        The enrichment table counts binary motifs in the observed graph and compares them with analytic expectations from E/I block densities. The perturbation tables then ask how the dominant eigenvalue changes under small weight perturbations, and how response/eigenvalue summaries change as the observed chain-like residual component in `J_eff` is depleted or enriched.
        """
    ),
    code(
        """
        motif_enrichment = pd.read_csv(paths["motif_enrichment"])
        display(motif_enrichment)

        enrichment_overview = (
            motif_enrichment.groupby("motif")["enrichment_ratio"]
            .agg(["mean", "min", "max"])
            .sort_index()
        )
        display(enrichment_overview)
        """
    ),
    code(
        """
        sensitivity_blocks = pd.read_csv(paths["sensitivity_blocks"])
        sensitivity_edges = pd.read_csv(paths["sensitivity_edges"])

        display(sensitivity_blocks)
        display(sensitivity_edges.head(20))
        """
    ),
    md(
        """
        ## Key Result: Motifs, Stability, and ISN Plausibility

        The focal question is whether removing or enhancing motifs moves the network toward or away from the ISN/paradoxical-response regime. For the leak-normalized linear model, the full network is stable when `max_real(lambda(J)) < 1`. An ISN-like condition is more specific: the E-only subnetwork is unstable by the same threshold, while the full E/I network is stabilized by inhibition and shows a negative I response to I input.
        """
    ),
    code(
        """
        isn_summary = pd.read_csv(paths["isn_summary"])
        stability_table = pd.read_csv(paths["stability_table"])
        block_removal = pd.read_csv(paths["block_removal"])

        display(stability_table)

        block_removal_focus = block_removal[
            block_removal["condition"].ne("original")
        ][
            [
                "matrix",
                "removed_block",
                "max_real_eigenvalue",
                "delta_max_real_eigenvalue",
                "stability_margin_1_minus_max_real",
                "delta_stability_margin_1_minus_max_real",
                "mean_I_to_I_input",
                "paradoxical_I_response",
            ]
        ]
        display(block_removal_focus)

        isn_focus = isn_summary[
            [
                "analysis",
                "condition",
                "motif_level",
                "max_real_eigenvalue",
                "stability_margin_1_minus_max_real",
                "delta_stability_margin_vs_reference",
                "e_subnetwork_max_real_eigenvalue",
                "e_subnetwork_stability_margin_1_minus_max_real",
                "mean_I_to_I_input",
                "paradoxical_I_response",
                "isn_plausible_stable_full_unstable_E_negative_I_response",
            ]
        ].copy()
        display(isn_focus)

        chain_isn = isn_focus[isn_focus["analysis"].eq("chain_enrichment_sweep_on_Jeff")]
        stable_paradoxical_chain = chain_isn[
            (chain_isn["stability_margin_1_minus_max_real"] > 0)
            & (chain_isn["paradoxical_I_response"])
        ]
        display(stable_paradoxical_chain)

        try:
            from IPython.display import SVG

            display(SVG(filename=str(paths["figure_ablation_delta_lambda"])))
            display(SVG(filename=str(paths["figure_block_removal_delta_lambda"])))
            display(SVG(filename=str(paths["figure_chain_stability"])))
            display(SVG(filename=str(paths["figure_chain_i_response"])))
        except Exception as exc:
            print("SVG display is unavailable in this runtime. Figure files are:")
            print(paths["figure_ablation_delta_lambda"])
            print(paths["figure_block_removal_delta_lambda"])
            print(paths["figure_chain_stability"])
            print(paths["figure_chain_i_response"])
            print(exc)
        """
    ),
    md(
        """
        ## Section 8: Motif Removal / Scrambling Stress Tests

        These manipulations test whether edges that participate in the strongest motifs are dynamically influential. Interpret them as stress tests: removing top motif-associated edges also changes the degree and weight distribution, while block shuffling preserves block-level weight distributions but scrambles local placement.
        """
    ),
    code(
        """
        motif_stress = pd.read_csv(paths["motif_stress"])
        display(motif_stress)

        stress_focus = motif_stress[
            [
                "condition",
                "spectral_radius",
                "delta_spectral_radius",
                "stability_margin_1_minus_max_real",
                "delta_stability_margin_1_minus_max_real",
                "e_subnetwork_max_real_eigenvalue",
                "e_subnetwork_stability_margin_1_minus_max_real",
                "chain_count",
                "delta_chain_count",
                "mean_I_to_I_input",
                "delta_mean_I_to_I_input",
                "paradoxical_I_response",
                "isn_plausible_stable_full_unstable_E_negative_I_response",
            ]
        ]
        display(stress_focus)
        """
    ),
    code(
        """
        chain_sweep = pd.read_csv(paths["chain_sweep"])

        sweep_eigen_summary = (
            chain_sweep[
                [
                    "chain_enrichment_multiplier",
                    "spectral_radius",
                    "max_real_eigenvalue",
                    "stability_margin_1_minus_max_real",
                    "e_subnetwork_max_real_eigenvalue",
                    "e_subnetwork_stability_margin_1_minus_max_real",
                    "condition_number_I_minus_scaled_J",
                    "response_solver",
                ]
            ]
            .drop_duplicates()
            .sort_values("chain_enrichment_multiplier")
            .reset_index(drop=True)
        )
        display(sweep_eigen_summary)

        chain_sweep = chain_sweep.assign(
            response_pair=chain_sweep["receiver_group"] + "<-" + chain_sweep["input_group"]
        )
        sweep_response_wide = chain_sweep.pivot_table(
            index="chain_enrichment_multiplier",
            columns="response_pair",
            values="mean_response",
            aggfunc="first",
        )
        display(sweep_response_wide)

        sweep_paradoxical_wide = chain_sweep.pivot_table(
            index="chain_enrichment_multiplier",
            columns="response_pair",
            values="paradoxical",
            aggfunc="first",
        )
        display(sweep_paradoxical_wide)

        sweep_diag = (
            chain_sweep[chain_sweep["receiver_group"] == chain_sweep["input_group"]]
            [
                [
                    "chain_enrichment_multiplier",
                    "receiver_group",
                    "spectral_radius",
                    "max_real_eigenvalue",
                    "stability_margin_1_minus_max_real",
                    "e_subnetwork_max_real_eigenvalue",
                    "mean_response",
                    "paradoxical",
                    "isn_plausible_stable_full_unstable_E_negative_I_response",
                ]
            ]
            .sort_values(["receiver_group", "chain_enrichment_multiplier"])
            .reset_index(drop=True)
        )
        display(sweep_diag)

        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].plot(
                sweep_eigen_summary["chain_enrichment_multiplier"],
                sweep_eigen_summary["spectral_radius"],
                marker="o",
            )
            axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
            axes[0].set_xlabel("Chain enrichment multiplier")
            axes[0].set_ylabel("Spectral radius of perturbed Jeff")
            axes[0].set_title("Stability boundary")

            diagonal_columns = [
                column
                for column in sweep_response_wide.columns
                if column.split("<-")[0] == column.split("<-")[1]
            ]
            for column in diagonal_columns:
                axes[1].plot(sweep_response_wide.index, sweep_response_wide[column], marker="o", label=column)
            axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
            axes[1].set_xlabel("Chain enrichment multiplier")
            axes[1].set_ylabel("Mean same-population response")
            axes[1].set_title("Paradoxical-response sign flip")
            axes[1].legend(title="Response")
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib is not installed in this runtime; compact sweep tables are shown above.")
        """
    ),
    md(
        """
        ## Full, effective, and low-rank responses

        The paper defines a paradoxical response as a negative diagonal response: population `p` responds in the opposite direction to uniform input into population `p`. Here the response matrix is evaluated at the requested radius-one scale, `Gamma = (I - J)^-1`. The reported condition number is part of the result. When the full matrix is ill-conditioned, the table also includes a clearly labeled pseudoinverse diagnostic for the singular limit.
        """
    ),
    code(
        """
        responses = pd.read_csv(paths["responses"])
        display(responses)

        response_pivot = responses.pivot_table(
            index=["receiver_group", "input_group"],
            columns="method",
            values="mean_response",
        )
        display(response_pivot)
        """
    ),
    code(
        """
        low_rank = pd.read_csv(paths["low_rank"])
        display(low_rank)

        low_rank_pivot = low_rank.pivot_table(
            index=["receiver_group", "input_group"],
            columns="rank",
            values="mean_response",
        )
        display(low_rank_pivot)
        """
    ),
    md(
        """
        ## Interpretation

        The empirical matrix is strongly non-normal after normalization: the spectral radius is 1, while the numerical abscissa is far larger than 1. That means directions in state space can be strongly amplified even though eigenvalues alone sit inside the unit disk after scaling.

        The E/I block means show much stronger average E-to-I drive than E-to-E drive after normalization, while inhibitory blocks are much weaker in mean magnitude. Chain correlations are small in raw coefficient size, but the paper's main point is that small chain over-representation can matter because it shapes outlying modes and population responses.

        At radius one, the full response is dominated by the marginal eigenmode and is therefore very large and sign-sensitive. Treat those values as perturbative evidence about the instability boundary, not as stable physiological gains. The enrichment and sensitivity tables are more robust summaries of which E/I motif classes and weight blocks push the dominant mode.
        """
    ),
    md(
        """
        ## Open choices to confirm

        1. If a cell-type annotation file beyond `mij_netlist.csv` exists, it can replace the current `pre_ei` labels.
        2. A supplemental response-scale sweep below 1.0 would make the approach to the marginal radius-one result easier to interpret.
        3. The current notebook keeps the paper's two-class E/I grouping; a separate appendix could add region-by-E/I groups without changing the main comparison.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH.resolve()}")
