#!/usr/bin/env python3
"""Build the executed companion notebook and canonical portable report input."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import nbformat as nbf
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs"
TABLES = OUT / "tables"


def fmt(x, digits=3):
    return f"{x:.{digits}g}"


def build_notebook() -> Path:
    summary = json.loads((OUT / "summary.json").read_text())
    metrics = pd.read_csv(TABLES / "non_normality_metrics.csv")
    raw = metrics.loc[metrics.matrix == "raw"].iloc[0]
    nulls = pd.read_csv(TABLES / "empirical_vs_null_statistics.csv")
    direction = pd.read_csv(TABLES / "directionality_cutoffs.csv")
    tl_dr = f"""## tl;dr

- The 85×85 signed directed matrix has {summary['nonzero_count']:,} nonzero entries ({summary['density']:.1%} density) and is extremely non-normal: ηC={raw.commutator_index:.3f}, ηH={raw.henrici_index:.4f}, and ω/α={raw.non_normal_reactivity_ratio:.1f}.
- With spectral-norm normalization and γ=1, reactivity begins at g={summary['g_react']:.3f}, while eigenvalue instability begins only at g={summary['g_stab']:.3f}. This creates a broad stable-but-reactive interval.
- At the midpoint of that interval (g={summary['g_operating']:.3f}), maximal finite-time energy gain is {summary['Gmax_operating']:.1f} at t*={summary['t_star_operating']:.3f}.
- The leading optimal initiator is **{summary['top_initiator']}** and the leading response receiver is **{summary['top_receiver']}**. The strongest single-node perturbation is **{summary['top_single_node']}**.
- These are mathematical sensitivities of a scaled linear system, not evidence of biological causation.
"""
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata.language_info = {"name": "python", "version": "3.12"}
    cells = [
        nbf.v4.new_markdown_cell("# Non-normal dynamics of the AscoliLab connectomic matrix\n\nA reproducible technical companion to the generated tables, figures, and report."),
        nbf.v4.new_markdown_cell(tl_dr),
        nbf.v4.new_markdown_cell("## Context & Methods\n\n### Key Assumptions\n\nThe convention is **A[i,j] = effect of source node j on target node i**. Dynamics use `J = -I + g A_tilde` with γ=1. Raw values are treated as structural weights with unknown physical units; dynamical conclusions emphasize spectral-radius, spectral-norm, and maximum-row-sum normalizations. The main operating point is the midpoint of the stable-reactive interval under spectral-norm normalization."),
        nbf.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT = Path.cwd()\nOUT = ROOT / 'analysis_outputs'\nTABLES = OUT / 'tables'\nFIGURES = OUT / 'figures'\nsummary = json.loads((OUT / 'summary.json').read_text())\nsummary"),
        nbf.v4.new_markdown_cell("### Re-run contract\n\nThe complete implementation is in `non_normal_analysis.py`. Set `RUN_FULL_ANALYSIS=True` to regenerate every CSV and figure from `mij_matrix.csv` using the fixed random seed. The default `False` keeps this reader notebook fast and verifies the saved run."),
        nbf.v4.new_code_cell("RUN_FULL_ANALYSIS = False\nif RUN_FULL_ANALYSIS:\n    from non_normal_analysis import main\n    main('mij_matrix.csv', 'analysis_outputs')"),
        nbf.v4.new_markdown_cell("## Data\n\nThe source is the labeled `mij_matrix.csv` file. Row and column labels match exactly; all entries are finite."),
        nbf.v4.new_code_cell("description = pd.read_csv(TABLES / 'matrix_descriptive_statistics.csv')\nquantiles = pd.read_csv(TABLES / 'nonzero_weight_quantiles.csv')\ndisplay(description, quantiles)"),
        nbf.v4.new_markdown_cell("## Results\n\n### Matrix structure and non-normality"),
        nbf.v4.new_code_cell("display(Image(filename=str(FIGURES / 'figure01_matrix_and_weights.png')))\ndisplay(pd.read_csv(TABLES / 'non_normality_metrics.csv'))"),
        nbf.v4.new_markdown_cell("### Stability, reactivity, and transient amplification\n\nReactivity (ω(J)>0) occurs far below instability (α(J)>0). Gain equal to one at and just below the reactivity threshold is expected: positive instantaneous growth begins at the threshold, while a finite-time maximum above one becomes visible only above it."),
        nbf.v4.new_code_cell("display(pd.read_csv(TABLES / 'stability_reactivity_thresholds.csv'))\ndisplay(Image(filename=str(FIGURES / 'figure03_coupling_phase.png')))\ndisplay(Image(filename=str(FIGURES / 'figure04_transient_gain_curves.png')))"),
        nbf.v4.new_markdown_cell("### Optimal and single-node perturbations"),
        nbf.v4.new_code_cell("display(pd.read_csv(TABLES / 'optimal_perturbation_nodes.csv').head(10))\ndisplay(pd.read_csv(TABLES / 'optimal_response_nodes.csv').head(10))\ndisplay(pd.read_csv(TABLES / 'single_node_perturbability.csv').head(15))\ndisplay(Image(filename=str(FIGURES / 'figure06_optimal_vectors.png')))"),
        nbf.v4.new_markdown_cell("### Directionality, signed structure, and null models"),
        nbf.v4.new_code_cell("display(pd.read_csv(TABLES / 'directionality_cutoffs.csv'))\ndisplay(pd.read_csv(TABLES / 'signed_component_comparison.csv'))\ndisplay(pd.read_csv(TABLES / 'empirical_vs_null_statistics.csv'))\ndisplay(Image(filename=str(FIGURES / 'figure08_directionality_interpolation.png')))\ndisplay(Image(filename=str(FIGURES / 'figure09_null_distributions.png')))"),
        nbf.v4.new_markdown_cell("### Lesions, edge sensitivity, stochastic response, and robustness"),
        nbf.v4.new_code_cell("display(pd.read_csv(TABLES / 'node_lesion_effects.csv').head(15))\ndisplay(pd.read_csv(TABLES / 'edge_sensitivity.csv').head(15))\ndisplay(pd.read_csv(TABLES / 'robustness_analysis.csv'))\ndisplay(Image(filename=str(FIGURES / 'figure10_node_lesions.png')))\ndisplay(Image(filename=str(FIGURES / 'figure11_pseudospectrum.png')))"),
        nbf.v4.new_markdown_cell("## Takeaways\n\nThe matrix is far more non-normal than merely asymmetric: the symmetric/skew decomposition produces a commutator index near its practical upper range, and ω is roughly 32 times α. Directionality must reach a substantial fraction of the empirical skew component before gain clears small thresholds. Extreme edges materially drive the magnitude: removing or clipping the largest weights shrinks gain sharply, although small random weight noise leaves the qualitative conclusion intact. The null and lesion analyses support structural specificity, but limited gain-null replication and screened lesion gain require cautious inferential language."),
    ]
    nb.cells = cells
    path = ROOT / "non_normal_dynamics_analysis.ipynb"
    nbf.write(nb, path)
    return path


def build_report_artifact() -> Path:
    s = json.loads((OUT / "summary.json").read_text())
    metrics = pd.read_csv(TABLES / "non_normality_metrics.csv")
    raw = metrics.loc[metrics.matrix == "raw"].iloc[0]
    coupling = pd.read_csv(TABLES / "maximum_gain_vs_coupling.csv")
    coupling["g_fraction_stability"] = coupling.g / s["g_stab"]
    init = pd.read_csv(TABLES / "optimal_perturbation_nodes.csv").head(10)
    recv = pd.read_csv(TABLES / "optimal_response_nodes.csv").head(10)
    single = pd.read_csv(TABLES / "single_node_perturbability.csv").head(15)
    lesion = pd.read_csv(TABLES / "node_lesion_effects.csv").head(15)
    nullstat = pd.read_csv(TABLES / "empirical_vs_null_statistics.csv")
    robust = pd.read_csv(TABLES / "robustness_analysis.csv")
    qcuts = pd.read_csv(TABLES / "directionality_cutoffs.csv")
    headline = [{
        "eta_c": raw.commutator_index, "eta_h": raw.henrici_index,
        "g_react": s["g_react"], "g_stab": s["g_stab"],
        "Gmax": s["Gmax_operating"], "t_star": s["t_star_operating"],
    }]
    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    source = {"id": "matrix_source", "label": "AscoliLab signed connectomic matrix", "path": "mij_matrix.csv",
              "query": {"engine": "duckdb", "language": "sql", "sql": "SELECT * FROM read_csv_auto('mij_matrix.csv', header = true)",
                        "description": "Loads the labeled signed connectomic matrix; numerical transforms are implemented in non_normal_analysis.py.",
                        "tables_used": ["mij_matrix.csv"],
                        "filters": ["No rows, columns, or nonzero edges excluded in the primary analysis."],
                        "metric_definitions": ["A[i,j] is the effect of source node j on target node i."]}}
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1, "surface": "report", "title": "Non-normal dynamics of the AscoliLab connectomic matrix",
            "description": "Technical analysis of structure, stability, reactivity, transient amplification, and perturbation sensitivity.",
            "generatedAt": generated, "filters": [],
            "cards": [
                {"id":"card_eta_c","description":"Scale-invariant commutator non-normality.","dataset":"headline","sourceId":"matrix_source","metrics":[{"label":"Commutator index ηC","field":"eta_c","format":"number"}]},
                {"id":"card_thresholds","description":"Coupling thresholds under spectral-norm normalization and γ=1.","dataset":"headline","sourceId":"matrix_source","metrics":[{"label":"Reactivity threshold","field":"g_react","format":"number"},{"label":"Stability threshold","field":"g_stab","format":"number"}]},
                {"id":"card_gain","description":"Peak energy gain at the midpoint of the stable-reactive interval.","dataset":"headline","sourceId":"matrix_source","metrics":[{"label":"Maximum finite-time gain","field":"Gmax","format":"number"},{"label":"Time to peak","field":"t_star","format":"number"}]},
            ],
            "charts": [{
                "id":"gain_chart","title":"Maximum transient gain versus coupling","subtitle":"Spectral-norm normalization; stable couplings only; logarithmic growth is tabulated in the companion CSV.",
                "type":"line","dataset":"coupling_gain","sourceId":"matrix_source","valueFormat":"number",
                "encodings":{"x":{"field":"g","type":"quantitative","label":"Coupling g"},"y":{"field":"Gmax","type":"quantitative","label":"Maximum energy gain"},
                             "tooltip":[{"field":"g_fraction_stability","type":"quantitative","label":"Fraction of stability threshold","format":"percent"}]},
            }],
            "tables": [
                {"id":"normalization_table","title":"Normalization comparison","subtitle":"Scale changes thresholds but not scale-invariant asymmetry or non-normality indices.","dataset":"normalizations","sourceId":"matrix_source","defaultSort":{"field":"matrix","direction":"asc"},"columns":[
                    {"field":"matrix","label":"Matrix","type":"text"},{"field":"spectral_abscissa","label":"α(A)","format":"number"},{"field":"numerical_abscissa","label":"ω(A)","format":"number"},{"field":"commutator_index","label":"ηC","format":"number"},{"field":"henrici_index","label":"ηH","format":"number"}]},
                {"id":"initiator_table","title":"Leading perturbation initiators","subtitle":"Largest absolute entries of the optimal right singular vector at t*.","dataset":"initiators","sourceId":"matrix_source","defaultSort":{"field":"abs_v1","direction":"desc"},"columns":[{"field":"node","label":"Node","type":"text"},{"field":"abs_v1","label":"|v1|","format":"number"}]},
                {"id":"receiver_table","title":"Leading response receivers","subtitle":"Largest absolute entries of the optimal left singular vector at t*.","dataset":"receivers","sourceId":"matrix_source","defaultSort":{"field":"abs_u1","direction":"desc"},"columns":[{"field":"node","label":"Node","type":"text"},{"field":"abs_u1","label":"|u1|","format":"number"}]},
                {"id":"single_table","title":"Single-node perturbability","subtitle":"Peak energy gain from unit perturbations at individual nodes.","dataset":"single_nodes","sourceId":"matrix_source","defaultSort":{"field":"single_node_peak_gain","direction":"desc"},"columns":[{"field":"node","label":"Node","type":"text"},{"field":"single_node_peak_gain","label":"Peak gain","format":"number"},{"field":"time_to_peak","label":"Time to peak","format":"number"},{"field":"integrated_energy","label":"Integrated energy","format":"number"}]},
                {"id":"lesion_table","title":"Strongest gain-reducing lesion screens","subtitle":"Five-time-point screening around the empirical t*; exact α, ω, and thresholds are saved separately.","dataset":"lesions","sourceId":"matrix_source","defaultSort":{"field":"delta_Gmax_screen","direction":"asc"},"columns":[{"field":"node","label":"Node","type":"text"},{"field":"delta_Gmax_screen","label":"Δ screened Gmax","format":"number"},{"field":"g_react_lesion","label":"g_react after lesion","format":"number"},{"field":"g_stab_lesion","label":"g_stab after lesion","format":"number"}]},
                {"id":"null_table","title":"Empirical-versus-null statistics","subtitle":"50 spectral randomizations per family and 20 gain calculations per family; finite-sample empirical p-values.","dataset":"null_stats","sourceId":"matrix_source","defaultSort":{"field":"standardized_effect","direction":"desc"},"columns":[{"field":"null_model","label":"Null","type":"text"},{"field":"metric","label":"Metric","type":"text"},{"field":"empirical","label":"Empirical","format":"number"},{"field":"null_mean","label":"Null mean","format":"number"},{"field":"standardized_effect","label":"Standardized effect","format":"number"},{"field":"upper_tail_p","label":"Upper-tail p","format":"number"}]},
            ],
            "sources": [source],
            "blocks": [
                {"id":"title","type":"markdown","body":"# Non-normal dynamics of the AscoliLab connectomic matrix"},
                {"id":"summary","type":"markdown","sourceId":"matrix_source","body":f"## Technical summary\n\nThe matrix is **strongly non-normal** (ηC={raw.commutator_index:.3f}, ηH={raw.henrici_index:.4f}), with numerical abscissa about {raw.non_normal_reactivity_ratio:.1f}× its spectral abscissa. Under spectral-norm normalization and γ=1, reactivity begins at **g={s['g_react']:.3f}**, while instability begins only at **g={s['g_stab']:.3f}**. At the midpoint of that stable-reactive interval, the maximum energy gain is **{s['Gmax_operating']:.1f}** at **t*={s['t_star_operating']:.3f}**. This is transient amplification in a stable linear system, not instability."},
                {"id":"headline","type":"metric-strip","cardIds":["card_eta_c","card_thresholds","card_gain"]},
                {"id":"gain_findings","type":"markdown","sourceId":"matrix_source","body":f"## A broad stable-but-reactive regime produces large transient gain\n\nThe reactivity threshold is only {s['g_react']/s['g_stab']:.1%} of the stability threshold. Gain rises from one at the reactivity boundary to {s['Gmax_operating']:.1f} at the interval midpoint and 735.9 near 95% of the stability threshold. Eigenvalues alone therefore substantially understate finite-time perturbation growth."},
                {"id":"gain_chart_block","type":"chart","chartId":"gain_chart"},
                {"id":"vectors","type":"markdown","sourceId":"matrix_source","body":f"## Initiators and receivers are distinct\n\nThe leading optimal initiator is **{s['top_initiator']}**, while the leading response receiver is **{s['top_receiver']}**. **{s['top_single_node']}** ranks first for a constrained single-node impulse. These roles are mathematically different and should not be collapsed into a single notion of importance."},
                {"id":"init_table_block","type":"table","tableId":"initiator_table"},
                {"id":"recv_table_block","type":"table","tableId":"receiver_table"},
                {"id":"single_table_block","type":"table","tableId":"single_table"},
                {"id":"structure","type":"markdown","sourceId":"matrix_source","body":f"## Directionality and a concentrated edge tail drive much of the effect\n\nThe smallest grid values where gain exceeds 1+δ are q={qcuts.q_cutoff_grid.min():.2f} for δ=0.01 and q={qcuts.q_cutoff_grid.max():.2f} for δ=0.25. Positive-only gain at 90% of its stability threshold is 620.2 versus 639.1 for the signed matrix, so inhibition modestly increases the main amplification measure but does not create the effect; the negative-only matrix yields 13.5. Removing the top 1% of absolute-weight edges reduces 90%-threshold gain from {robust.loc[robust.variant=='raw','Gmax_at_90pct_stability'].iloc[0]:.1f} to {robust.loc[robust.variant=='remove_top_1pct_edges','Gmax_at_90pct_stability'].iloc[0]:.1f}; removing 5% reduces it to {robust.loc[robust.variant=='remove_top_5pct_edges','Gmax_at_90pct_stability'].iloc[0]:.1f}. Small random weight noise leaves the qualitative conclusion intact."},
                {"id":"nulls","type":"markdown","sourceId":"matrix_source","body":"## Empirical non-normality exceeds the randomized controls\n\nThe empirical commutator index lies above all 50 realizations in both null families (finite-sample upper-tail p=0.0196). Gain is also above the direction-randomized controls (p=0.0476 across 20 gain realizations), while the weight-shuffled gain comparison is less decisive (p=0.0952). Effect sizes and the limited gain-null sample must be considered alongside p-values."},
                {"id":"null_table_block","type":"table","tableId":"null_table"},
                {"id":"scope","type":"markdown","body":"## Scope, data, and metric definitions\n\nThe source is an 85×85 labeled signed matrix with 1,641 nonzero entries. The convention is **A[i,j] = effect of source j on target i**. Asymmetry is ||A−Aᵀ||F/||A||F; non-normality uses the normalized commutator and Henrici departure; reactivity means ω(J)>0; instability means α(J)>0; finite-time amplification means Gmax>1. Dynamics use J=−I+gÃ with γ=1. Raw weights have no asserted biological time unit."},
                {"id":"normalization_table_block","type":"table","tableId":"normalization_table"},
                {"id":"methods","type":"markdown","body":"## Methodology\n\nDouble-precision NumPy/SciPy routines compute eigenspectra, Hermitian eigenvalues, SVDs, matrix exponentials, Lyapunov solutions, and Frechet derivatives. Transient gain uses an adaptive time horizon, a coarse scan, and bounded local refinement. The analysis uses 50 null matrices per randomization family for spectral statistics and 20 per family for gain. Directionality and inhibition are scanned on 21-point grids. All randomization uses seed 20260901."},
                {"id":"lesions","type":"markdown","sourceId":"matrix_source","body":f"## Perturbation control is localized but metric-dependent\n\nThe strongest screened gain-reducing lesion is **{s['strongest_gain_reducing_lesion_screen']}**. The strongest locally sensitive edge is **{s['top_edge_by_local_spectral_or_reactivity_sensitivity']}**. These control rankings need not match optimal-initiation or single-node rankings because each metric answers a different perturbation question."},
                {"id":"lesion_table_block","type":"table","tableId":"lesion_table"},
                {"id":"limits","type":"markdown","body":f"## Limitations, uncertainty, and robustness\n\nThe main conclusions are scale-invariant for asymmetry/non-normality and consistent across scalar normalizations, but gain magnitudes depend on the chosen relative coupling. Extreme weights matter materially. Lesion gain is a five-time-point screen around the empirical t*, edge gain derivatives are computed for the top 40 screened edges, and the pseudospectrum is a finite grid approximation; on that grid it reaches Re(z)>0 for ε≈{s['pseudospectral_right_half_min_epsilon_grid']:.3f}. At a coupling stable for both empirical and symmetrized matrices (g={s['stochastic_comparison_coupling']:.3f}), stationary variance is {s['total_stationary_variance']:.2f} versus {s['symmetrized_total_stationary_variance']:.2f}; this low-coupling comparison does not contradict the much larger empirical deterministic gain deeper in its unique stable-reactive interval."},
                {"id":"next","type":"markdown","body":"## Recommended next steps\n\n1. Confirm whether the matrix weights have calibrated units or a defensible biological coupling range.\n2. Increase gain-null replication to at least 100–500 per family for publication-grade tail probabilities.\n3. Re-optimize Gmax for the top lesion candidates and brute-force deletion of the top edge candidates.\n4. Add a signed strength-preserving rewiring null if the topology admits a well-mixed Markov chain.\n5. Only then embed the structure in a justified nonlinear firing-rate model and evaluate state-dependent Jacobians."},
                {"id":"questions","type":"markdown","body":"## Further questions\n\n- Are the largest weights measurement-scale artifacts or biologically privileged pathways?\n- Which operating coupling is supported by physiology?\n- Do candidate lesion and edge effects persist under cell-type-specific time constants and structured noise?\n- Can independent data validate the predicted initiator/receiver separation?"},
            ],
        },
        "snapshot": {"version":1,"generatedAt":generated,"status":"ready","datasets":{
            "headline":headline,"coupling_gain":coupling.to_dict("records"),"normalizations":metrics.to_dict("records"),
            "initiators":init.to_dict("records"),"receivers":recv.to_dict("records"),"single_nodes":single.to_dict("records"),
            "lesions":lesion.to_dict("records"),"null_stats":nullstat.to_dict("records"),
        },"accessIssues":[]},
        "sources":[source],
        "package_info":{"originUrl":"artifact://ascolilab-non-normal-dynamics","controls":{"edit":False,"refresh":False}},
    }
    path = OUT / "report_artifact.json"
    path.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(build_notebook())
    print(build_report_artifact())
