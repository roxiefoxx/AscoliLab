#!/usr/bin/env python3
"""B1 + C1 of the corrections pass, on one shared surrogate ensemble.

B1  exact-LOP ordering statistics, observed vs degree- and dyad-preserving nulls
C1  stationary covariance trace(P) from JP + PJ^T + I = 0, same null ensemble

Both tests used to run against a plain directed edge swap, which destroys about
60% of the 422 mutual dyads.  A mutual dyad contributes one backward edge under
ANY ordering, so that surrogate was not a null for the quantity a
feedforward-ordering statistic measures.  Both tests also need the *same*
solver on the observed and the null side: the gap between a heuristic and the
optimum is not the same for the two ensembles, which is what inflated z = +3.15.

Sign convention.  The rewire moves edge targets and keeps sources fixed, so
every surrogate edge keeps the sign of its presynaptic type: Dale's law is
preserved by construction.

    python3 corrections_pass_nulls.py --nulls 100 --outdir outputs/corrections_pass
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.linalg import solve_continuous_lyapunov

from optimal_ordering_lop import (LOP, rewire_degree_and_dyad_preserving,
                                  dyad_census)

RHO_TARGET = 0.95          # matches jacobian_analysis / schur_decomp / inhib_modulation


# ------------------------------------------------------------------ helpers
def load_post_pre(matrix_csv="mij_matrix.csv"):
    df = pd.read_csv(matrix_csv, index_col=0).astype(float)
    labels = list(df.columns)
    M = df.to_numpy().T.copy()             # [post, pre]
    return M, labels


def ordering_metrics(M, order):
    X = M[np.ix_(order, order)]
    B = np.abs(X) > 0
    up_c = int(np.count_nonzero(np.triu(B, 1)))
    lo_c = int(np.count_nonzero(np.tril(B, -1)))
    up_w = float(np.abs(np.triu(X, 1)).sum())
    lo_w = float(np.abs(np.tril(X, -1)).sum())
    return {"backward_edges": up_c, "forward_edges": lo_c,
            "feedforward_edge_fraction": lo_c / max(up_c + lo_c, 1),
            "backward_abs_weight": up_w, "forward_abs_weight": lo_w,
            "weighted_feedforward_fraction": lo_w / max(up_w + lo_w, np.finfo(float).eps)}


def trace_P(M_post_pre, target=RHO_TARGET):
    """trace of the stationary covariance, and the same-spectrum normal control."""
    W = np.asarray(M_post_pre, dtype=float)
    rho = float(np.max(np.abs(np.linalg.eigvals(W))))
    W = W * (target / rho)
    J = -np.eye(len(W)) + W
    ev = np.linalg.eigvals(J)
    if np.max(ev.real) >= 0:
        return np.nan, np.nan
    P = solve_continuous_lyapunov(J, -np.eye(len(J)))
    return float(np.trace(P)), float(np.sum(-1.0 / (2.0 * ev.real)).real)


def signed_from_magnitudes(A_pre_post, sign_of_source):
    """Re-apply Dale's law: every edge takes the sign of its presynaptic type."""
    S = A_pre_post * sign_of_source[:, None]
    return S.T.copy()                      # back to [post, pre]


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="mij_matrix.csv")
    ap.add_argument("--nulls", type=int, default=100)
    ap.add_argument("--null-time-limit", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--outdir", default="outputs/corrections_pass")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    M, labels = load_post_pre(a.matrix)
    np.fill_diagonal_view = None
    n = len(M)
    A = np.abs(M).T.copy()                 # [pre, post], non-negative
    np.fill_diagonal(A, 0.0)
    # sign of each presynaptic type, from the sign of its outgoing row
    Mpp = M.T.copy(); np.fill_diagonal(Mpp, 0.0)
    sign_src = np.where((Mpp < 0).sum(1) > (Mpp > 0).sum(1), -1.0, 1.0)

    obs_mut, obs_asym = dyad_census(A)
    lop = LOP(n)
    t0 = time.time()
    obs_res = lop.solve(A)
    obs_order = [int(i) for i in obs_res["order"]]
    obs_met = ordering_metrics(M, obs_order)
    obs_tr, obs_norm = trace_P(M)
    print(f"n={n}  dyad census {obs_mut} mutual / {obs_asym} asymmetric")
    print(f"observed LOP forward {obs_res['forward']:.6f}  "
          f"{'CERTIFIED' if obs_res['certified'] else 'gap %.1e' % obs_res['gap']}  "
          f"{time.time()-t0:.1f}s")
    print(f"observed trace(P) {obs_tr:.1f}  normal control {obs_norm:.1f}  "
          f"excess {obs_tr/obs_norm:.0f}x", flush=True)

    rng = np.random.default_rng(a.seed)
    rows = []
    for k in range(a.nulls):
        R = rewire_degree_and_dyad_preserving(A, rng)
        mut, asym = dyad_census(R)
        res = lop.solve(R, time_limit=a.null_time_limit)
        Rm = signed_from_magnitudes(R, sign_src)
        met = ordering_metrics(Rm, [int(i) for i in res["order"]])
        tr, nrm = trace_P(Rm)
        rows.append({"null": k, "mutual_dyads": mut, "asymmetric_edges": asym,
                     "dyad_census_preserved": bool(mut == obs_mut and asym == obs_asym),
                     "lop_forward": float(res["forward"]),
                     "lop_upper_bound": float(res["upper_bound"]),
                     "lop_certified": bool(res["certified"]),
                     "trace_P": tr, "trace_P_normal_control": nrm,
                     **met})
        print(f"  null {k+1:3d}/{a.nulls}  lop {res['forward']:.5f} "
              f"(ub {res['upper_bound']:.5f}{'' if res['certified'] else ', not certified'})  "
              f"traceP {tr:.1f}", flush=True)
        pd.DataFrame(rows).to_csv(out / "null_ensemble.csv", index=False)

    raw = pd.DataFrame(rows)

    def ztest(name, observed, vals, direction="higher"):
        vals = np.asarray(vals, dtype=float); vals = vals[np.isfinite(vals)]
        sd = vals.std(ddof=1)
        extreme = np.sum(vals >= observed) if direction == "higher" else np.sum(vals <= observed)
        return {"metric": name, "observed": float(observed),
                "null_mean": float(vals.mean()), "null_sd": float(sd),
                "z_score": float((observed - vals.mean()) / sd) if sd > 0 else np.nan,
                "empirical_p_one_sided": float((extreme + 1) / (len(vals) + 1)),
                "n_null": int(len(vals))}

    tests = [
        ztest("weighted_feedforward_fraction", obs_met["weighted_feedforward_fraction"],
              raw["weighted_feedforward_fraction"]),
        ztest("feedforward_edge_fraction", obs_met["feedforward_edge_fraction"],
              raw["feedforward_edge_fraction"]),
        ztest("lop_forward_fraction", obs_res["forward"], raw["lop_forward"]),
        # conservative: certified observed optimum vs time-limited null dual bounds
        ztest("lop_forward_fraction_vs_null_upper_bound", obs_res["forward"],
              raw["lop_upper_bound"]),
        ztest("trace_P", obs_tr, raw["trace_P"]),
    ]
    tests = pd.DataFrame(tests)
    tests.to_csv(out / "null_tests.csv", index=False)

    meta = {"n_nodes": n, "n_nulls": a.nulls, "seed": a.seed,
            "rho_target": RHO_TARGET,
            "observed_dyad_census": [obs_mut, obs_asym],
            "observed_lop_forward": obs_res["forward"],
            "observed_lop_certified": bool(obs_res["certified"]),
            "observed_trace_P": obs_tr,
            "observed_trace_P_normal_control": obs_norm,
            "observed_ordering_metrics": obs_met,
            "null_time_limit_s": a.null_time_limit,
            "all_nulls_dyad_preserved": bool(raw["dyad_census_preserved"].all()),
            "n_nulls_certified": int(raw["lop_certified"].sum())}
    (out / "null_meta.json").write_text(json.dumps(meta, indent=2))
    print("\n" + tests.to_string(index=False))
    print(f"\nwrote {out}/null_ensemble.csv, null_tests.csv, null_meta.json")


if __name__ == "__main__":
    main()
