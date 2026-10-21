"""Re-export shim -- the implementation lives in `tarjan_reorder/motif_signed_overlap_analysis_script.py`.

Corrections pass 2026-09-17.  This file and its counterpart in `tarjan_reorder/`
were byte-identical copies, so a fix to one silently left the other stale and
running both notebooks duplicated every output for no gain.  `tarjan_reorder/`
is the canonical location; this module now imports from it by path so there is
exactly one implementation to maintain.

Anything importing `motif_signed_overlap_analysis_script` from this
directory keeps working unchanged.
"""
import importlib.util as _ilu
import pathlib as _pl
import sys as _sys

_SOURCE = _pl.Path(__file__).resolve().parent.parent / "tarjan_reorder" / "motif_signed_overlap_analysis_script.py"
if not _SOURCE.exists():
    raise ImportError(f"canonical implementation not found at {_SOURCE}")

_spec = _ilu.spec_from_file_location("_canonical_motif_signed_overlap_analysis_script", _SOURCE)
_module = _ilu.module_from_spec(_spec)
_sys.modules["_canonical_motif_signed_overlap_analysis_script"] = _module
_spec.loader.exec_module(_module)

globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
__all__ = [k for k in vars(_module) if not k.startswith("_")]
CANONICAL_SOURCE = str(_SOURCE)
