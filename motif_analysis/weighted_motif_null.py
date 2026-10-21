#!/usr/bin/env python3
"""C4 of the corrections pass -- weighted motif census against a proper null.

Two results, one of them negative and load-bearing.

1. THE NODE-LEVEL WEIGHTED TRIAD MEASURE IS DEGENERATE.  The quantity in use --
   for each connected triad, sum the weight on its edges and credit all three
   members -- is an exact affine function of node strength and contains no triad
   information whatsoever.  With S = |M| + |M|^T (zero diagonal),
   s_a = sum_b S[a, b] and T = S.sum() / 2,

       participation(a) = (n - 3) * s_a + T                         [exact]

   because each of a's incident weights is counted once per choice of the third
   node (n - 2 times), and every triad not containing a contributes T - s_a.
   Triads with no edges contribute zero either way, so the "connected" filter is
   a no-op.  Verified numerically below to 3e-15 relative, with corr = 1.0
   against strength.  A z-score on this quantity is a z-score on node strength.
   It cannot distinguish DG Granule, or anything else, on motif grounds.

2. WHAT REPLACES IT.  Weighted participation resolved BY TRIAD CLASS (the
   Davis-Holland-Leinhardt 16), as a share of the node's total triad weight.
   Shares are strength-normalized by construction -- the degenerate total
   divides out -- so a class share is a statement about which motif
   configurations a node's weight sits in, which is the question the measure was
   supposed to answer.  Each share is z-scored against the null ensemble.

Null model.  Degree- and dyad-census-preserving rewire (the same surrogate as
the ordering test).  Preserved exactly: every node's in-degree and out-degree,
the mutual/asymmetric dyad census, the global multiset of edge weights and hence
total strength.  NOT preserved: node strength, since reassigning targets is what
makes it a null.  The realized deviation is measured and reported rather than
waved at -- and because the class measure is a share, a strength shift that
rescales a node's whole profile cancels out of it.

The chain-signature table (E/I signature of X -> Y -> Z, weighted) aggregates
over the graph rather than crediting a node, so it never had the
single-heavy-edge problem and is reportable on the observed graph alone.  It is
z-scored against the same ensemble here since the ensemble is already built.

    python3 weighted_motif_null.py --nulls 100
"""
from __future__ import annotations
import argparse, itertools, json, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

TRIAD_CLASSES = ['003', '012', '102', '021D', '021U', '021C', '111D', '111U',
                 '030T', '030C', '201', '120D', '120U', '120C', '210', '300']
PAIR_ORDER = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]


def rewire_degree_and_dyad_preserving(Aw, rng, mult=8):
    """Preserve directed degree and the mutual/asymmetric dyad census."""
    Bn = Aw > 0
    n = len(Aw)
    upper = np.arange(n)[:, None] < np.arange(n)
    mutual = np.argwhere(Bn & Bn.T & upper)
    asym = np.argwhere(Bn & ~Bn.T)
    out = np.zeros_like(Aw)

    src, tgt = asym[:, 0].copy(), asym[:, 1].copy()
    weights = Aw[src, tgt].copy()
    present = {(int(a), int(b)) for a, b in zip(src, tgt)}
    mutual_edges = {(int(a), int(b)) for a, b in mutual} | {(int(b), int(a)) for a, b in mutual}
    m = len(weights)
    for _ in range(mult * max(m, 1)):
        e1, e2 = rng.integers(0, m, 2)
        if e1 == e2:
            continue
        a, b = src[e1], tgt[e1]
        c, d = src[e2], tgt[e2]
        if a == d or c == b:
            continue
        new_edges = [(int(a), int(d)), (int(c), int(b))]
        if any(edge in present or edge in mutual_edges for edge in new_edges):
            continue
        if any((edge[1], edge[0]) in present or (edge[1], edge[0]) in mutual_edges for edge in new_edges):
            continue
        present.discard((int(a), int(b)))
        present.discard((int(c), int(d)))
        present.update(new_edges)
        tgt[e1], tgt[e2] = d, b
    out[src, tgt] = weights

    swapped_mutual = mutual.copy()
    mutual_weights = np.array([[Aw[a, b], Aw[b, a]] for a, b in mutual])
    k = len(swapped_mutual)
    for _ in range(mult * max(k, 1)):
        e1, e2 = rng.integers(0, k, 2)
        if e1 == e2:
            continue
        a, b = swapped_mutual[e1]
        c, d = swapped_mutual[e2]
        if len({a, b, c, d}) < 4:
            continue
        candidates = [(a, d), (c, b)]
        occupied = {(int(p), int(q)) for p, q in swapped_mutual} | {
            (int(q), int(p)) for p, q in swapped_mutual
        }
        if any((int(p), int(q)) in occupied or (int(q), int(p)) in occupied for p, q in candidates):
            continue
        if any(out[p, q] > 0 or out[q, p] > 0 for p, q in candidates):
            continue
        swapped_mutual[e1], swapped_mutual[e2] = [a, d], [c, b]
    for (a, b), (wab, wba) in zip(swapped_mutual, mutual_weights):
        out[a, b] = wab
        out[b, a] = wba
    return out


def dyad_census(Aw):
    B = Aw > 0
    mutual = B & B.T
    return int(mutual.sum() // 2), int((B ^ mutual).sum())


# ----------------------------------------------------------------- triad codes
def _classify(mat):
    mutual = sum(1 for i in range(3) for j in range(i + 1, 3) if mat[i][j] and mat[j][i])
    asym = sum(1 for i in range(3) for j in range(3) if i != j and mat[i][j] and not mat[j][i])
    null = 3 - mutual - asym
    key = (mutual, asym, null)
    base = {(0, 0, 3): '003', (0, 1, 2): '012', (1, 0, 2): '102',
            (2, 0, 1): '201', (2, 1, 0): '210', (3, 0, 0): '300'}
    if key in base:
        return base[key]
    if key == (0, 2, 1):
        outd = [sum(mat[i]) for i in range(3)]
        ind = [sum(mat[j][i] for j in range(3)) for i in range(3)]
        return '021D' if 2 in outd else '021U' if 2 in ind else '021C'
    if key == (1, 1, 1):
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            if mat[i][j] and mat[j][i]:
                k = 3 - i - j
                return '111D' if (mat[k][i] or mat[k][j]) else '111U'
    if key == (0, 3, 0):
        return '030T' if 2 in [sum(mat[i]) for i in range(3)] else '030C'
    if key == (1, 2, 0):
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            if mat[i][j] and mat[j][i]:
                k = 3 - i - j
                if mat[k][i] + mat[k][j] == 2:
                    return '120D'
                if mat[i][k] + mat[j][k] == 2:
                    return '120U'
                return '120C'
    raise ValueError(f"unclassified triad {key}")


def _code_to_class_index():
    out = np.empty(64, dtype=np.int8)
    for bits in range(64):
        mat = [[0] * 3 for _ in range(3)]
        for i, (x, y) in enumerate(PAIR_ORDER):
            mat[x][y] = (bits >> i) & 1
        out[bits] = TRIAD_CLASSES.index(_classify(mat))
    return out


CODE2CLASS = _code_to_class_index()


# ---------------------------------------------------------------------- data
def _default_input_path(local_name, fallback):
    local_path = HERE / local_name
    return local_path if local_path.exists() else fallback


def load(matrix_csv=None, netlist_csv=None):
    matrix_csv = Path(matrix_csv or _default_input_path("mij_matrix.csv", ROOT / "non-normal_matrices" / "data" / "mij_matrix.csv"))
    netlist_csv = Path(netlist_csv or _default_input_path("mij_netlist.csv", ROOT / "tarjan_reorder" / "mij_netlist.csv"))
    M = pd.read_csv(matrix_csv, index_col=0).astype(float)
    names = list(M.index)
    if list(M.columns) != names:
        raise ValueError("Row and column labels must match in the same order.")
    A = M.to_numpy()                       # A[pre, post] -- asserted below
    net = pd.read_csv(netlist_csv)
    index = {label: i for i, label in enumerate(names)}
    bad = sum(abs(A[index[r.pre_neuron], index[r.post_neuron]] - r.m_ij)
              > 1e-6 * max(1.0, abs(r.m_ij)) for r in net.itertuples(index=False))
    if bad:
        raise ValueError(f"{matrix_csv} is not [pre, post]: {bad} netlist rows disagree.")
    ei = {}
    for nc, ec in (("pre_neuron", "pre_ei"), ("post_neuron", "post_ei")):
        for a_, b_ in net[[nc, ec]].dropna().drop_duplicates().itertuples(index=False):
            ei[str(a_)] = str(b_).strip().upper()
    W = np.abs(A).copy()
    np.fill_diagonal(W, 0.0)
    return names, W, np.array([ei[x] for x in names])


# ------------------------------------------------------------------ measures
class TriadIndex:
    """Precomputed triad index; the combination list is reused across surrogates."""

    def __init__(self, n):
        self.n = n
        combos = np.fromiter(itertools.chain.from_iterable(
            itertools.combinations(range(n), 3)), dtype=np.int32)
        self.abc = combos.reshape(-1, 3)
        self.a, self.b, self.c = self.abc.T

    def class_weight(self, W):
        """(16, n) matrix: weight of triads of each class that each node sits in."""
        B = (W > 0)
        a, b, c = self.a, self.b, self.c
        bits = (B[a, b].astype(np.int64)
                | (B[b, a] << 1) | (B[a, c] << 2)
                | (B[c, a] << 3) | (B[b, c] << 4) | (B[c, b] << 5))
        cls = CODE2CLASS[bits].astype(np.int64)
        S = W + W.T
        w = S[a, b] + S[a, c] + S[b, c]
        out = np.zeros((len(TRIAD_CLASSES), self.n))
        for node in (a, b, c):
            np.add.at(out, (cls, node), w)
        return out


def participation_closed_form(W):
    n = len(W)
    S = W + W.T
    return (n - 3) * S.sum(1) + S.sum() / 2.0


def chain_signature_weight(W, ei):
    """Weight carried by directed open 3-chains X -> Y -> Z, keyed by E/I signature."""
    B = W > 0
    n = len(W)
    out = {}
    for y in range(n):
        xs = np.flatnonzero(B[:, y])
        zs = np.flatnonzero(B[y, :])
        if not len(xs) or not len(zs):
            continue
        wx, wz = W[xs, y], W[y, zs]
        for ii, x in enumerate(xs):
            keep = zs != x
            if not keep.any():
                continue
            vals = wx[ii] + wz[keep]
            for z, v in zip(zs[keep], vals):
                key = ei[x] + ei[y] + ei[z]
                out[key] = out.get(key, 0.0) + float(v)
    return out


# ---------------------------------------------------------------------- main
def run_weighted_motif_null_analysis(
    nulls=100,
    seed=11,
    matrix=None,
    netlist=None,
    outdir=None,
    progress=True,
):
    out = Path(outdir or HERE / "outputs" / "corrections_pass")
    out.mkdir(parents=True, exist_ok=True)

    names, W, ei = load(matrix, netlist)
    n = len(names)
    idx = TriadIndex(n)
    obs_census = dyad_census(W)
    if progress:
        print(f"{n} types, {int((W>0).sum())} edges, "
              f"dyad census {obs_census[0]} mutual / {obs_census[1]} asymmetric")

    # --- result 1: the degeneracy, demonstrated rather than asserted ---------
    S = W + W.T
    closed = participation_closed_form(W)
    brute = np.zeros(n)
    for x, y, z in idx.abc:
        s = S[x, y] + S[x, z] + S[y, z]
        if s > 0:
            brute[x] += s; brute[y] += s; brute[z] += s
    rel = float(np.abs(brute - closed).max() / brute.max())
    corr = float(np.corrcoef(brute, S.sum(1))[0, 1])
    if progress:
        print(f"\nnode-level weighted triad participation == (n-3)*strength + const: "
              f"max relative error {rel:.1e}, corr with strength {corr:.6f}")
        print("  -> the measure carries no triad information; reported by class share below")

    # --- result 2: class-resolved shares vs the null -------------------------
    obs_cw = idx.class_weight(W)
    obs_tot = obs_cw.sum(0)
    obs_share = np.divide(obs_cw, obs_tot, out=np.zeros_like(obs_cw), where=obs_tot > 0)
    obs_chain = chain_signature_weight(W, ei)

    rng = np.random.default_rng(seed)
    null_share = np.empty((nulls, len(TRIAD_CLASSES), n))
    null_chain, preserved, out_dev, in_dev = [], [], [], []
    t0 = time.time()
    for k in range(nulls):
        R = rewire_degree_and_dyad_preserving(W, rng)
        preserved.append(dyad_census(R) == obs_census)
        if abs(R.sum() - W.sum()) > 1e-6 * W.sum():
            raise AssertionError("total strength not preserved by the rewire")
        Rs = R + R.T
        out_dev.append(np.abs(R.sum(1) - W.sum(1)) / np.maximum(W.sum(1), 1e-12))
        in_dev.append(np.abs(R.sum(0) - W.sum(0)) / np.maximum(W.sum(0), 1e-12))
        cw = idx.class_weight(R)
        tot = cw.sum(0)
        null_share[k] = np.divide(cw, tot, out=np.zeros_like(cw), where=tot > 0)
        null_chain.append(chain_signature_weight(R, ei))
        if progress and (k + 1) % 10 == 0:
            print(f"  null {k+1:3d}/{nulls}  ({time.time()-t0:.0f}s)", flush=True)

    mean, sd = null_share.mean(0), null_share.std(0, ddof=1)
    z = np.where(sd > 0, (obs_share - mean) / np.where(sd > 0, sd, 1.0), np.nan)
    p = (1 + (null_share >= obs_share[None]).sum(0)) / (1 + nulls)

    rows = []
    for ci, cname in enumerate(TRIAD_CLASSES):
        for j in range(n):
            rows.append({"cell_type": names[j], "ei": ei[j], "triad_class": cname,
                         "observed_share": obs_share[ci, j],
                         "null_mean_share": mean[ci, j], "null_sd_share": sd[ci, j],
                         "z_score": z[ci, j], "empirical_p_one_sided": p[ci, j]})
    nodes = pd.DataFrame(rows)
    nodes["strength"] = nodes.cell_type.map(dict(zip(names, S.sum(1))))
    nodes.to_csv(out / "weighted_triad_class_share_z.csv", index=False)

    degenerate = pd.DataFrame({
        "cell_type": names, "ei": ei, "strength": S.sum(1),
        "weighted_triad_participation": brute,
        "closed_form_(n-3)*strength+T": closed,
    })
    degenerate.to_csv(out / "weighted_triad_participation_degenerate.csv", index=False)

    keys = sorted(obs_chain, key=lambda k: -obs_chain[k])
    total = sum(obs_chain.values())
    crows = []
    for key in keys:
        vals = np.array([c.get(key, 0.0) for c in null_chain])
        totals = np.array([sum(c.values()) for c in null_chain])
        obs_s = obs_chain[key] / total
        shares = vals / totals
        sd_s = shares.std(ddof=1)
        crows.append({"signature": key, "observed_weight": obs_chain[key],
                      "observed_share": obs_s,
                      "null_mean_share": float(shares.mean()),
                      "null_sd_share": float(sd_s),
                      "z_share": float((obs_s - shares.mean()) / sd_s) if sd_s > 0 else np.nan,
        "empirical_p_one_sided": float((1 + (shares >= obs_s).sum()) / (1 + nulls))})
    chains = pd.DataFrame(crows)
    chains.to_csv(out / "chain_signature_table.csv", index=False)

    (out / "weighted_motif_null_meta.json").write_text(json.dumps({
        "n_nulls": nulls, "seed": seed, "n_types": n,
        "all_nulls_dyad_preserved": bool(np.all(preserved)),
        "observed_dyad_census": list(obs_census),
        "participation_degeneracy_max_relative_error": rel,
        "participation_correlation_with_strength": corr,
        "null_model": "degree- and dyad-census-preserving rewire; total strength "
                      "and the weight multiset preserved exactly, node strength not",
        "relative_out_strength_deviation_median": float(np.median(out_dev)),
        "relative_out_strength_deviation_max": float(np.max(out_dev)),
        "relative_in_strength_deviation_median": float(np.median(in_dev)),
        "relative_in_strength_deviation_max": float(np.max(in_dev)),
    }, indent=2))

    if progress:
        print(f"\nnode-strength deviation across surrogates (relative): "
              f"out median {np.median(out_dev):.3f} max {np.max(out_dev):.2f}; "
              f"in median {np.median(in_dev):.3f} max {np.max(in_dev):.2f}")
        print("\n=== chain signature X -> Y -> Z, weighted ===")
        print(chains.to_string(index=False, float_format=lambda x: f"{x:,.4g}"))
        print("\n=== strongest class-share deviations (|z|, p <= 0.05) ===")
    strong = nodes[(nodes.empirical_p_one_sided <= 0.05) & nodes.z_score.notna()]
    if progress:
        print(strong.reindex(strong.z_score.abs().sort_values(ascending=False).index)
              .head(20).to_string(index=False, float_format=lambda x: f"{x:,.4g}"))
        print("\n=== DG Granule by class ===")
        print(nodes[nodes.cell_type == "DG Granule"].to_string(
            index=False, float_format=lambda x: f"{x:,.4g}"))
        print(f"\nwrote {out}/weighted_triad_class_share_z.csv, chain_signature_table.csv, "
              f"weighted_triad_participation_degenerate.csv")

    return {
        "output_dir": out,
        "node_class_share_z": nodes,
        "weighted_participation_degenerate": degenerate,
        "chain_signature_table": chains,
        "strong_class_share_deviations": strong,
        "metadata": {
            "n_nulls": nulls,
            "seed": seed,
            "observed_dyad_census": obs_census,
            "participation_degeneracy_max_relative_error": rel,
            "participation_correlation_with_strength": corr,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nulls", type=int, default=100)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--matrix", default=None)
    ap.add_argument("--netlist", default=None)
    ap.add_argument("--outdir", default=str(HERE / "outputs" / "corrections_pass"))
    a = ap.parse_args()
    run_weighted_motif_null_analysis(
        nulls=a.nulls,
        seed=a.seed,
        matrix=a.matrix,
        netlist=a.netlist,
        outdir=a.outdir,
        progress=True,
    )


if __name__ == "__main__":
    main()
