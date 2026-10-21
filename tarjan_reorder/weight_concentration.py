#!/usr/bin/env python3
"""C3 of the corrections pass -- the weight-concentration table.

The corrected reading of the ordering result ("the optimal E->E ordering does
not recover the trisynaptic direction; DG sorts to the end, which localizes the
trisynaptic signal outside the E->E block") rests on four facts about where the
weight in m_ij actually sits.  They were quoted in prose before they existed in
any notebook.  This module is the single reproducible source for them.

    python3 weight_concentration.py            # prints the table
    from weight_concentration import weight_concentration_table
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

# Normalization convention for the block-norm table; matches
# jacobian_analysis / schur_decomp / inhib_modulation.
NORM_TARGET = 0.95

HERE = Path(__file__).resolve().parent


def load(matrix_csv=HERE / "mij_matrix.csv", netlist_csv=HERE / "mij_netlist.csv"):
    df = pd.read_csv(matrix_csv, index_col=0).astype(float)
    labels = list(df.columns)
    W_full = df.to_numpy().T.copy()            # [post, pre], self-weights intact
    W = W_full.copy()
    np.fill_diagonal(W, 0.0)
    net = pd.read_csv(netlist_csv)
    ei = {}
    for n, e in (("pre_neuron", "pre_ei"), ("post_neuron", "post_ei")):
        for a, b in net[[n, e]].drop_duplicates().itertuples(index=False):
            ei[str(a)] = str(b).strip().lower()
    cls = np.array([ei[l] for l in labels])
    return W, labels, cls, W_full


def weight_concentration_table(W, labels, cls, W_with_diagonal=None):
    A = np.abs(W)                              # [post, pre], diagonal already zeroed
    if W_with_diagonal is None:
        W_with_diagonal = W
    total = A.sum()
    isE, isI = cls == "e", cls == "i"

    post, pre = np.nonzero(A)
    edges = pd.DataFrame({
        "pre": [labels[j] for j in pre], "post": [labels[i] for i in post],
        "pre_ei": cls[pre], "post_ei": cls[post], "abs_m_ij": A[post, pre],
    }).sort_values("abs_m_ij", ascending=False).reset_index(drop=True)

    top12 = edges.head(12)
    top12_share = float(top12.abs_m_ij.sum() / total)
    top12_all_E_to_I = bool(((top12.pre_ei == "e") & (top12.post_ei == "i")).all())

    # excitatory out-weight landing on interneurons, per source type
    frac = []
    for j in np.flatnonzero(isE):
        out = A[:, j]
        if out.sum() > 0:
            frac.append((labels[j], float(out[isI].sum() / out.sum())))
    frac = pd.DataFrame(frac, columns=["excitatory_type", "fraction_onto_interneurons"])

    def pair(a, b):
        i, j = labels.index(b), labels.index(a)     # W[post, pre] = a -> b
        return float(abs(W[i, j]))

    dg_local = pair("DG Granule", "DG AIPRIM")
    dg_ca3 = pair("DG Granule", "CA3 Pyramidal")

    fro = lambda M, m: float(np.linalg.norm(M[np.ix_(*m)], ord="fro"))
    ee, ei_, ie, ii = (fro(A, (isE, isE)), fro(A, (isI, isE)),
                       fro(A, (isE, isI)), fro(A, (isI, isI)))
    total_fro = float(np.linalg.norm(A, ord="fro"))

    # Same block norms on the normalized operator used by jacobian_analysis,
    # schur_decomp and inhib_modulation: self-weights RETAINED, spectral radius
    # scaled to 0.95.  This is the scale the E->E 4.91 vs E->I 77.02 figures in
    # the writeup are on.  The ratio is scale-free (6.3% either way); only the
    # absolute numbers move, and they move by a factor of ~1.8 depending on
    # whether the diagonal is kept, which is why the convention is stated here.
    Wd = np.asarray(W_with_diagonal, dtype=float)
    rho = float(np.max(np.abs(np.linalg.eigvals(Wd))))
    An = np.abs(Wd) * (NORM_TARGET / rho)
    ee_n, ei_n = fro(An, (isE, isE)), fro(An, (isI, isE))
    ie_n, ii_n = fro(An, (isE, isI)), fro(An, (isI, isI))

    return {
        "n_types": len(labels), "n_excitatory": int(isE.sum()), "n_inhibitory": int(isI.sum()),
        "n_edges": int(len(edges)), "total_abs_m_ij": float(total),
        "top12_edge_share_of_total_abs_weight": top12_share,
        "top12_all_principal_to_interneuron": top12_all_E_to_I,
        "top12_edges": top12,
        "DG_Granule_to_DG_AIPRIM": dg_local,
        "DG_Granule_to_CA3_Pyramidal": dg_ca3,
        "DG_local_over_mossy_fibre_ratio": dg_local / dg_ca3 if dg_ca3 else np.inf,
        "median_excitatory_out_weight_onto_interneurons":
            float(frac.fraction_onto_interneurons.median()),
        "DG_Granule_out_weight_onto_interneurons":
            float(frac.loc[frac.excitatory_type == "DG Granule",
                           "fraction_onto_interneurons"].iloc[0]),
        "frobenius_EE": ee, "frobenius_EI": ei_, "frobenius_IE": ie, "frobenius_II": ii,
        "frobenius_total": total_fro,
        "spectral_radius": rho,
        "frobenius_EE_rho1": ee_n, "frobenius_EI_rho1": ei_n,
        "frobenius_IE_rho1": ie_n, "frobenius_II_rho1": ii_n,
        "EE_share_of_matrix_norm": ee / total_fro,
        "excitatory_out_weight_fractions": frac.sort_values(
            "fraction_onto_interneurons", ascending=False).reset_index(drop=True),
    }


def main():
    W, labels, cls, W_full = load()
    r = weight_concentration_table(W, labels, cls, W_full)
    print(f"{r['n_types']} types ({r['n_excitatory']} E / {r['n_inhibitory']} I), "
          f"{r['n_edges']} edges, total |m_ij| = {r['total_abs_m_ij']:,.1f}\n")
    print("Top 12 edges by |m_ij|")
    print(r["top12_edges"].to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
    print(f"\n  share of all |m_ij|            {r['top12_edge_share_of_total_abs_weight']:.1%}")
    print(f"  all principal -> interneuron   {r['top12_all_principal_to_interneuron']}")
    print(f"\nDG Granule -> DG AIPRIM        {r['DG_Granule_to_DG_AIPRIM']:,.1f}")
    print(f"DG Granule -> CA3 Pyramidal    {r['DG_Granule_to_CA3_Pyramidal']:,.1f}")
    print(f"  ratio                        {r['DG_local_over_mossy_fibre_ratio']:,.0f}x")
    print(f"\nexcitatory out-weight onto interneurons")
    print(f"  median over excitatory types {r['median_excitatory_out_weight_onto_interneurons']:.1%}")
    print(f"  DG Granule                   {r['DG_Granule_out_weight_onto_interneurons']:.1%}")
    print(f"\nFrobenius norm by E/I block (rows = post, columns = pre)")
    for k in ("EE", "EI", "IE", "II"):
        print(f"  {k}  {r['frobenius_'+k]:>12,.2f}")
    print(f"  total {r['frobenius_total']:>11,.2f}")
    print(f"  E->E as a share of the matrix norm  {r['EE_share_of_matrix_norm']:.1%}")
    print(f"\n  on the normalized operator (self-weights retained, "
          f"rho scaled {r['spectral_radius']:,.1f} -> {NORM_TARGET}):")
    print(f"    E->E {r['frobenius_EE_rho1']:.2f}   E->I {r['frobenius_EI_rho1']:.2f}   "
          f"I->E {r['frobenius_IE_rho1']:.2f}   I->I {r['frobenius_II_rho1']:.2f}")

    out = HERE / "outputs" / "corrections_pass"
    out.mkdir(parents=True, exist_ok=True)
    r["top12_edges"].to_csv(out / "weight_concentration_top12_edges.csv", index=False)
    r["excitatory_out_weight_fractions"].to_csv(
        out / "weight_concentration_excitatory_out_fractions.csv", index=False)
    scalars = {k: v for k, v in r.items() if not isinstance(v, pd.DataFrame)}
    (out / "weight_concentration.json").write_text(json.dumps(scalars, indent=2, default=float))
    print(f"\nwrote {out}/weight_concentration.json and two CSVs")


if __name__ == "__main__":
    main()
