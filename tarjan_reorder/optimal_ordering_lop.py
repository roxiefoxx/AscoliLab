"""
Exact linear ordering (minimum feedback arc set) for the 85-type connectome.

The ordering problem behind every "feedforward" claim is the Linear Ordering
Problem. It is NP-hard in general, but at n = 85 it solves to certified
optimality in seconds with HiGHS (scipy.optimize.milp), so no heuristic and no
"upper bound on the true minimum" caveat is needed for the observed graph.

Formulation
-----------
    x_ij = 1 if type i precedes type j            (one binary per unordered pair)
    maximise  sum over edges (pre -> post) of  w * [pos(pre) < pos(post)]
    subject to  0 <= x_ij + x_jk - x_ik <= 1      for every triple i < j < k

The triangle constraints do not depend on the weights, so the constraint matrix
is built once and reused across the observed graph and every null draw.

Usage
-----
    python optimal_ordering_lop.py --netlist mij_netlist.csv --out optimal_order_85.csv
    python optimal_ordering_lop.py --netlist mij_netlist.csv --nulls 100
"""
from __future__ import annotations
import argparse, itertools, time
import numpy as np, pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix


# ---------------------------------------------------------------- data
def load(netlist_path):
    net = pd.read_csv(netlist_path)
    lab = sorted(set(net.pre_neuron) | set(net.post_neuron))
    idx = {l: i for i, l in enumerate(lab)}
    n = len(lab)
    ei = {}
    for cn, ce in [("pre_neuron", "pre_ei"), ("post_neuron", "post_ei")]:
        for a, b in net[[cn, ce]].drop_duplicates().itertuples(index=False):
            ei[a] = str(b).lower()
    W = np.zeros((n, n))                       # signed, indexed [pre, post]
    for r in net.itertuples(index=False):
        W[idx[r.pre_neuron], idx[r.post_neuron]] += r.m_ij
    np.fill_diagonal(W, 0.0)
    return lab, np.array([ei[l] for l in lab]), W


# ------------------------------------------------- LOP constraint matrix
class LOP:
    def __init__(self, n):
        self.n = n
        self.pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        self.pid = {p: k for k, p in enumerate(self.pairs)}
        self.IJ = np.array(self.pairs)
        rows, cols, vals, r = [], [], [], 0
        for i, j, k in itertools.combinations(range(n), 3):
            rows += [r, r, r]
            cols += [self.pid[(i, j)], self.pid[(j, k)], self.pid[(i, k)]]
            vals += [1, 1, -1]
            r += 1
        self.con = LinearConstraint(
            coo_matrix((vals, (rows, cols)), shape=(r, len(self.pairs))),
            np.zeros(r), np.ones(r))

    def solve(self, Aw, time_limit=300):
        """Aw[i, j] = non-negative weight on edge i -> j. Returns dict."""
        I, J = self.IJ[:, 0], self.IJ[:, 1]
        c = -(Aw[I, J] - Aw[J, I])
        const, total = Aw[J, I].sum(), Aw.sum()
        res = milp(c=c, constraints=[self.con], integrality=np.ones(len(self.pairs)),
                   bounds=Bounds(0, 1),
                   options={"time_limit": time_limit, "mip_rel_gap": 0.0})
        x = np.round(res.x).astype(int)
        score = np.zeros(self.n)
        for (i, j), k in self.pid.items():
            score[j if x[k] == 1 else i] += 1
        db = res.mip_dual_bound
        return dict(order=np.argsort(score),
                    forward=(const - res.fun) / total,
                    upper_bound=(const - db) / total if db is not None else np.nan,
                    gap=res.mip_gap, certified=(res.mip_gap is not None and res.mip_gap <= 1e-9))


# ------------------------------------------------------------- nulls
def rewire_degree_and_dyad_preserving(Aw, rng, mult=8):
    """Preserve in-degree, out-degree AND the mutual/asymmetric dyad census.

    Mutual dyads are swapped only against mutual dyads and asymmetric edges only
    against asymmetric edges, so no swap converts one into the other. A plain
    directed edge swap (networkx.directed_edge_swap and the like) destroys about
    60% of the 422 mutual dyads on this graph. A mutual pair contributes one
    backward edge under ANY ordering, so losing them makes the null artificially
    easy to make feedforward and biases the test against significance.
    """
    Bn = Aw > 0
    mutual = np.argwhere(Bn & Bn.T & (np.arange(len(Aw))[:, None] < np.arange(len(Aw))))
    asym = np.argwhere(Bn & ~Bn.T)
    out = np.zeros_like(Aw)

    # --- asymmetric edges: standard degree-preserving swap within the class
    src, tgt = asym[:, 0].copy(), asym[:, 1].copy()
    w = Aw[src, tgt].copy()
    pres = {(int(a), int(b)) for a, b in zip(src, tgt)}
    mut_set = {(int(a), int(b)) for a, b in mutual} | {(int(b), int(a)) for a, b in mutual}
    m = len(w)
    for _ in range(mult * max(m, 1)):
        e1, e2 = rng.integers(0, m, 2)
        if e1 == e2:
            continue
        a, b = src[e1], tgt[e1]
        c, d = src[e2], tgt[e2]
        if a == d or c == b:
            continue
        new = [(int(a), int(d)), (int(c), int(b))]
        if any(e in pres or e in mut_set for e in new):
            continue
        if any((e[1], e[0]) in pres or (e[1], e[0]) in mut_set for e in new):
            continue                                    # would create a new mutual dyad
        pres.discard((int(a), int(b))); pres.discard((int(c), int(d)))
        pres.update(new)
        tgt[e1], tgt[e2] = d, b
    out[src, tgt] = w

    # --- mutual dyads: swap partners among themselves, keeping them mutual
    mu = mutual.copy()
    mw = np.array([[Aw[a, b], Aw[b, a]] for a, b in mutual])
    k = len(mu)
    for _ in range(mult * max(k, 1)):
        e1, e2 = rng.integers(0, k, 2)
        if e1 == e2:
            continue
        a, b = mu[e1]; c, d = mu[e2]
        if len({a, b, c, d}) < 4:
            continue
        cand = [(a, d), (c, b)]
        occupied = {(int(p), int(q)) for p, q in mu} | {(int(q), int(p)) for p, q in mu}
        if any((int(p), int(q)) in occupied or (int(q), int(p)) in occupied for p, q in cand):
            continue
        if any(out[p, q] > 0 or out[q, p] > 0 for p, q in cand):
            continue
        mu[e1], mu[e2] = [a, d], [c, b]
    for (a, b), (wab, wba) in zip(mu, mw):
        out[a, b] = wab
        out[b, a] = wba
    return out


def dyad_census(Aw):
    B = Aw > 0
    return int((B & B.T).sum() // 2), int((B ^ (B & B.T)).sum())


# -------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--netlist", default="mij_netlist.csv")
    ap.add_argument("--out", default=None, help="write the optimal ordering to this CSV")
    ap.add_argument("--nulls", type=int, default=0)
    ap.add_argument("--null-time-limit", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    lab, cls, W = load(args.netlist)
    n = len(lab)
    A, B = np.abs(W), (np.abs(W) > 0).astype(float)
    lop = LOP(n)
    print(f"n={n}  pair variables={len(lop.pairs)}  triangle constraints={n*(n-1)*(n-2)//6}")
    print(f"dyad census: {dyad_census(A)[0]} mutual, {dyad_census(A)[1]} asymmetric\n")

    for tag, mat in (("weighted  (all |m_ij|)", A), ("binary    (edge count)", B)):
        t0 = time.time()
        r = lop.solve(mat)
        flag = "CERTIFIED OPTIMAL" if r["certified"] else f"gap {r['gap']:.2e}"
        print(f"{tag}: forward fraction {r['forward']:.6f}   {flag}   {time.time()-t0:.1f}s")
        if tag.startswith("weighted") and args.out:
            o = r["order"]
            pd.DataFrame({"position": np.arange(n),
                          "cell_type": [lab[i] for i in o],
                          "ei": [cls[i] for i in o]}).to_csv(args.out, index=False)
            print(f"   -> {args.out}")

    if args.nulls:
        obs = lop.solve(A)["forward"]
        rng = np.random.default_rng(args.seed)
        inc, ub = [], []
        for s in range(args.nulls):
            R = rewire_degree_and_dyad_preserving(A, rng)
            if s == 0:
                print(f"\n   null dyad census check: {dyad_census(R)} (should match the observed census)")
            r = lop.solve(R, time_limit=args.null_time_limit)
            inc.append(r["forward"]); ub.append(r["upper_bound"])
            print(f"   null {s+1:3d}/{args.nulls}  incumbent {r['forward']:.5f}  upper {r['upper_bound']:.5f}", flush=True)
        inc, ub = np.array(inc), np.array(ub)
        print(f"\nobserved (certified) {obs:.6f}")
        print(f"null incumbents {inc.mean():.5f} ± {inc.std(ddof=1):.5f}")
        print(f"null upper bnds {ub.mean():.5f} ± {ub.std(ddof=1):.5f}")
        print(f"conservative z (vs upper bounds) = {(obs-ub.mean())/ub.std(ddof=1):+.2f}   "
              f"p = {(1+np.sum(ub>=obs))/(1+len(ub)):.4f}")
        print("Report the conservative figure: comparing a certified observed optimum against")
        print("time-limited null incumbents would overstate significance.")


if __name__ == "__main__":
    main()
