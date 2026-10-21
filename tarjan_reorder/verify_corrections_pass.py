#!/usr/bin/env python3
"""Re-derive every headline number the 2026-09-17 corrections pass changed.

Checks the artifacts, not the prose. Run from the repository root:

    python3 tarjan_reorder/verify_corrections_pass.py

Exits non-zero on any mismatch. This is what caught "all 100 surrogates certified" in a draft
of docs/CORRECTIONS_PASS_RESULTS.md when the real figure was 96 — which is why the reported
argument is now the dual-bound one, which does not depend on solver time at all.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tarjan_reorder"))
from optimal_ordering_lop import LOP, load

OK = FAIL = 0


def check(name, got, want, tol=None):
    global OK, FAIL
    if isinstance(want, (int, float)) and not isinstance(want, bool):
        good = abs(got - want) <= (tol if tol is not None else abs(want) * 1e-6)
    else:
        good = got == want
    print(("  OK   " if good else "  FAIL ") + f"{name}: {got}" + ("" if good else f" (expected {want})"))
    OK += good; FAIL += not good


print("B1 / C1 — 85-type ordering null and stationary covariance")
t = pd.read_csv(ROOT / "tarjan_reorder/outputs/corrections_pass/null_tests.csv").set_index("metric")
e = pd.read_csv(ROOT / "tarjan_reorder/outputs/corrections_pass/null_ensemble.csv")
m = json.loads((ROOT / "tarjan_reorder/outputs/corrections_pass/null_meta.json").read_text())
check("weighted feedforward z", round(float(t.loc["weighted_feedforward_fraction", "z_score"]), 2), 3.15, 5e-3)
check("weighted feedforward p", round(float(t.loc["weighted_feedforward_fraction", "empirical_p_one_sided"]), 4), 0.0099, 1e-4)
check("binary feedforward z", round(float(t.loc["feedforward_edge_fraction", "z_score"]), 2), 1.98, 5e-3)
check("trace(P) z", round(float(t.loc["trace_P", "z_score"]), 2), -0.39, 5e-3)
check("observed solved to certified optimality", m["observed_lop_certified"], True)
check("every surrogate preserved the dyad census", m["all_nulls_dyad_preserved"], True)
# The decisive check: gap-proof, independent of how many surrogates certified.
check("observed exceeds EVERY surrogate dual bound", bool((e.lop_upper_bound < m["observed_lop_forward"]).all()), True)

print("\nB2 — 85-node certificate")
lab, cls, W = load(str(ROOT / "tarjan_reorder/mij_netlist.csv"))
A = np.abs(W); lop = LOP(len(A))
rw, rb = lop.solve(A), lop.solve((A > 0).astype(float))
check("weighted forward fraction", round(rw["forward"], 6), 0.995185, 1e-6)
check("weighted certified", rw["certified"], True)
check("binary forward fraction", round(rb["forward"], 6), 0.717834, 1e-6)
check("binary backward edges", int(round((1 - rb["forward"]) * 1570)), 443)
order = pd.read_csv(ROOT / "tarjan_reorder/optimal_order_85.csv")
check("exported ordering == solver output", list(order.cell_type) == [lab[i] for i in rw["order"]], True)

print("\nB3 — inhibitory feedforward/feedback under the 85-type ordering")
M = pd.read_csv(ROOT / "tarjan_reorder/mij_matrix.csv", index_col=0).astype(float)
labels = list(M.columns)
Wi = np.where(M.to_numpy().T < 0, -M.to_numpy().T, 0.0)
post, pre = np.nonzero(Wi)
pos = {l: k for k, l in enumerate(order.cell_type)}
d = np.array([pos[labels[i]] - pos[labels[j]] for i, j in zip(post, pre)])
ei = order.set_index("cell_type")["ei"].to_dict()
isE = np.array([ei[labels[i]] == "e" for i in post])
check("feedforward inhibitory edges", int((d > 0).sum()), 497)
check("feedback inhibitory edges", int((d < 0).sum()), 366)
check("I->E edges counted forward", int(((d > 0) & isE).sum()), 28)

print("\nC3 — weight concentration")
w = json.loads((ROOT / "tarjan_reorder/outputs/corrections_pass/weight_concentration.json").read_text())
check("top-12 share of all |m_ij|", round(w["top12_edge_share_of_total_abs_weight"], 3), 0.489, 5e-4)
check("top 12 all principal -> interneuron", w["top12_all_principal_to_interneuron"], True)
check("DG local / mossy fibre ratio", round(w["DG_local_over_mossy_fibre_ratio"]), 955, 1)
check("median excitatory out-weight onto interneurons", round(w["median_excitatory_out_weight_onto_interneurons"], 3), 0.714, 5e-4)
check("DG Granule onto interneurons", round(w["DG_Granule_out_weight_onto_interneurons"], 3), 0.916, 5e-4)
check("E->E share of matrix norm", round(w["EE_share_of_matrix_norm"], 3), 0.063, 5e-4)
check("E->E Frobenius at rho=0.95", round(w["frobenius_EE_rho1"], 2), 4.91, 5e-3)
check("E->I Frobenius at rho=0.95", round(w["frobenius_EI_rho1"], 2), 77.02, 5e-3)

print("\nC4 — chain signature and the participation degeneracy")
c = pd.read_csv(ROOT / "motif_analysis/outputs/corrections_pass/chain_signature_table.csv").set_index("signature")
cm = json.loads((ROOT / "motif_analysis/outputs/corrections_pass/weighted_motif_null_meta.json").read_text())
check("E->I->I share", round(float(c.loc["EII", "observed_share"]), 3), 0.363, 5e-4)
check("E->E->E share", round(float(c.loc["EEE", "observed_share"]), 3), 0.034, 5e-4)
check("E->I->I z", round(float(c.loc["EII", "z_share"]), 2), 5.07, 5e-3)
check("E->E->E z (depleted)", bool(float(c.loc["EEE", "z_share"]) < 0), True)
check("participation correlates 1.0 with strength", cm["participation_correlation_with_strength"], 1.0, 1e-9)
check("closed-form degeneracy error < 1e-12", cm["participation_degeneracy_max_relative_error"] < 1e-12, True)

print("\nB7 — triad census orientation")
cen = pd.read_csv(ROOT / "motif_analysis/outputs/triad_census/triad_census.csv").set_index("triad_class")["count"]
for k, v in (("300", 978), ("210", 1532), ("120D", 1666), ("030T", 2249), ("030C", 16), ("201", 405)):
    check(f"triad class {k}", int(cen[k]), v)

print(f"\n{OK} checks passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
