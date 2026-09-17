# Pre-sync backup — 2026-09-05

Copies of the files that were overwritten or displaced when this repository was
synced to `GitHub/EE_backbone`. Nothing here is referenced by the analysis; the
folder exists only so the sync is reversible. Delete it once you are satisfied.

| File | Why it was replaced |
| --- | --- |
| `ee_backbone_analysis.py` | Was a 42-byte shim, `from ee_backbone_analysis_script import *`. Replaced by the full canonical module (1,011 lines). |
| `ee_backbone_analysis_script.py.superseded` | The pre-refactor implementation, moved out of the top level. A byte-identical copy now lives in `archive/ee_backbone_analysis_script.py`, matching the GitHub layout. |
| `ee_backbone_method_comparison.ipynb` | Pre-refactor version, superseded by the corrected notebook with the feedback-arc ordering method added. |
| `ei_backbone_analysis.ipynb` | Self-contained pre-refactor version **with saved cell outputs** (414 KB). The replacement imports `ei_backbone_analysis_helpers.py` and has outputs stripped, so it needs re-running. This backup is the only copy of those executed outputs. |
| `update_advisor_notebook_brief.py` | Older revision (2026-08-21); the GitHub copy (2026-08-24) adds "Limitations" and "Recommended next steps" sections. |
| `docs/ee_backbone_method_brief_for_advisor.docx` | Older build of the advisor brief, regenerated from the newer script. |

Note: `docs/~$_backbone_method_brief_for_advisor.docx` in the parent folder is a
Microsoft Word lock file, left in place. It means the .docx was open in Word at some
point; it can be deleted.
