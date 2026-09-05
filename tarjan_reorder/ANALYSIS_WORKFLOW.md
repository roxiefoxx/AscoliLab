# Analysis Workflow

Use the notebooks in this order, from foundational to more specialized:

1. `01_matrix_reordering_paper_analysis.ipynb`
   - Foundational notebook and best first read.
   - Establishes the paper convention for `Mij`, validates matrix orientation, compares ordering methods, runs Tarjan threshold sweeps, performs degree-preserving null tests, and connects ordering to Schur/eigenmode analyses.
   - This is the most comprehensive single notebook for the current repository because it covers data validation, global ordering, benchmarks, null tests, modal analysis, and a trisynaptic-pathway check in one workflow.

2. `02_modified_tarjan_pathway_reconstruction.ipynb`
   - Specialized follow-up on pathway reconstruction after the global ordering work.
   - Extends Modified Tarjan into threshold stability, condensation-DAG path enumeration, pathway null tests, and downstream Schur/eigenmode relevance.
   - Partly overlaps with the foundational notebook's Modified Tarjan, threshold, and trisynaptic sections, but keeps a deeper pathway-focused analysis.

3. `03_motif_schur_decomposition_analysis.ipynb`
   - Motif-level decomposition notebook.
   - Consumes Schur-ready 3x3 motif block exports, decomposes each motif with and without self-weights, ranks motif Schur modes, tests enhancement/ablation effects, summarizes superpatterns, and exports motif/cell summaries.
   - This is the most comprehensive notebook for local three-node motif Schur analysis.

4. `04_motif_signed_overlap_operator_analysis.ipynb`
   - Most specialized and technically layered motif notebook.
   - Reuses motif block metadata, builds a sparse motif-index by motif-index signed overlap operator, decomposes distributed motif-neighborhood modes, and adds signed Laplacian analyses.
   - It depends conceptually on the motif Schur workflow and should be read after it.

## Redundancy Notes

- None of the four current notebooks is an exact duplicate.
- `02_modified_tarjan_pathway_reconstruction.ipynb` is partially redundant with `01_matrix_reordering_paper_analysis.ipynb` for Modified Tarjan ordering, threshold sweeps, trisynaptic paths, and modal relevance. Keep it when pathway stability and DAG path enumeration matter; otherwise the foundational notebook is enough for a compact global view.
- `04_motif_signed_overlap_operator_analysis.ipynb` overlaps with `03_motif_schur_decomposition_analysis.ipynb` in loading motif blocks and using dominant motif-mode information. Keep it when the question is about relationships among motif occurrences; otherwise the Schur notebook is the simpler motif-level endpoint.
- `build_matrix_reordering_notebook.py` is not an analysis notebook. It is a notebook builder for `01_matrix_reordering_paper_analysis.ipynb` and should be treated as generation support.

## Companion Python Scripts

Notebook-facing reusable code is intentionally named with a `_script.py` suffix so the script modules are distinct from reader-facing notebooks:

- `matrix_reordering_paper_analysis_script.py`
- `modified_tarjan_pathway_analysis_script.py`
- `motif_schur_decomposition_analysis_script.py`
- `motif_signed_overlap_analysis_script.py`

The notebooks import these `_script.py` modules directly. Keep future notebook support modules on the same naming convention to avoid ambiguity between executable notebooks and reusable Python code.

## Orientation Audit

See `ORIENTATION_AUDIT.md` for the source/target versus row/column convention used by each notebook, helper script, and library call. In short: notebooks `01` and `02` expose runtime controls for matrix path, optional netlist path, explicit matrix orientation, and normalization mode. Use `MATRIX_ORIENTATION = "pre_rows_post_columns"` when the CSV table is source-to-target; the loader will transpose it once for analysis as `M[post, pre]`.
