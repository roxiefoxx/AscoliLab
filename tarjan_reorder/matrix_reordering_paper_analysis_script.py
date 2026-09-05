#!/usr/bin/env python3
"""Paper-inspired matrix reordering and linear-mode analysis for Mij.

Implements the workflow reviewed in Borst & Leibold (2023), J Neurosci
43:3599-3610.  Matrix convention throughout is the paper convention:
M[post, pre] is the directed weight pre -> post, so feedforward edges lie
below the diagonal after a successful ordering.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import networkx as nx
import numpy as np
import pandas as pd
from scipy.linalg import schur, eigh, subspace_angles
from scipy.stats import kendalltau
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

DEFAULT_PERCENTILES = (0, 10, 25, 50, 75)
DEFAULT_BENCHMARK_SIZES = (30, 60, 120)
REGION_ORDER = {"EC": 0, "DG": 1, "CA3": 2, "CA2": 3, "CA1": 4, "SUB": 5}
MATRIX_ORIENTATIONS = {
    "auto_from_netlist",
    "pre_rows_post_columns",
    "post_rows_pre_columns",
}
NORMALIZATION_MODES = {"spectral_radius", "none"}


def region(label: str) -> str:
    p = label.split()[0]
    if p.startswith("CA3"): return "CA3"
    if p in {"EC", "MEC", "LEC"}: return "EC"
    return p if p in REGION_ORDER else "Other"


def _optional_path(path):
    if path is None:
        return None
    path = Path(path)
    if str(path).strip().lower() in {"", "none", "null"}:
        return None
    return path


def _normalize_matrix_orientation(matrix_orientation):
    value = str(matrix_orientation).strip().lower()
    aliases = {
        "auto": "auto_from_netlist",
        "netlist": "auto_from_netlist",
        "source_to_target": "pre_rows_post_columns",
        "pre_to_post": "pre_rows_post_columns",
        "pre_post": "pre_rows_post_columns",
        "rows_pre_cols_post": "pre_rows_post_columns",
        "pre_rows_post_cols": "pre_rows_post_columns",
        "target_from_source": "post_rows_pre_columns",
        "post_from_pre": "post_rows_pre_columns",
        "post_pre": "post_rows_pre_columns",
        "rows_post_cols_pre": "post_rows_pre_columns",
        "post_rows_pre_cols": "post_rows_pre_columns",
    }
    value = aliases.get(value, value)
    if value not in MATRIX_ORIENTATIONS:
        raise ValueError(
            "matrix_orientation must be one of "
            f"{sorted(MATRIX_ORIENTATIONS)}; got {matrix_orientation!r}"
        )
    return value


def load_inputs(
    matrix_path: Path,
    netlist_path: Path | None = None,
    matrix_orientation="auto_from_netlist",
    netlist_orientation_tolerance=1e-10,
):
    """Load inputs and return M in paper/dynamics form: rows=post, columns=pre."""
    raw = pd.read_csv(matrix_path, index_col=0)
    if raw.shape[0] != raw.shape[1] or set(raw.index) != set(raw.columns):
        raise ValueError("mij_matrix must be square with matching row/column labels")
    raw = raw.loc[raw.index, raw.index].astype(float)
    labels = list(raw.index)
    requested_orientation = _normalize_matrix_orientation(matrix_orientation)
    netlist_path = _optional_path(netlist_path)
    edges = pd.DataFrame(columns=["pre_neuron", "post_neuron", "m_ij"])
    audit = {
        "matrix_path": str(matrix_path),
        "netlist_path": str(netlist_path) if netlist_path is not None else None,
        "n_nodes": len(labels),
        "matrix_orientation_requested": requested_orientation,
        "analysis_orientation": "rows=post, columns=pre",
        "matrix_entry_definition": "M[post, pre] is the weight from presynaptic source pre to postsynaptic target post",
    }

    if netlist_path is not None:
        edges = pd.read_csv(netlist_path)
        required = {"pre_neuron", "post_neuron", "m_ij"}
        if not required <= set(edges.columns):
            raise ValueError(f"mij_netlist is missing {sorted(required-set(edges.columns))}")
        pos = {x: i for i, x in enumerate(labels)}
        recon = np.zeros(raw.shape, float)  # paper/dynamics convention
        unknown = 0
        for row in edges.itertuples(index=False):
            if row.pre_neuron in pos and row.post_neuron in pos:
                recon[pos[row.post_neuron], pos[row.pre_neuron]] += float(row.m_ij)
            else:
                unknown += 1
        as_is = raw.to_numpy()
        transposed = raw.to_numpy().T
        denom = max(np.linalg.norm(recon), np.finfo(float).eps)
        err_as_is = float(np.linalg.norm(as_is - recon) / denom)
        err_transposed = float(np.linalg.norm(transposed - recon) / denom)
        audit.update({
            "netlist_rows": int(len(edges)),
            "unknown_netlist_rows": int(unknown),
            "relative_error_matrix_as_is_vs_netlist_post_pre": err_as_is,
            "relative_error_matrix_transposed_vs_netlist_post_pre": err_transposed,
        })
    else:
        err_as_is = None
        err_transposed = None
        audit.update({
            "netlist_rows": None,
            "unknown_netlist_rows": None,
            "relative_error_matrix_as_is_vs_netlist_post_pre": None,
            "relative_error_matrix_transposed_vs_netlist_post_pre": None,
        })

    if requested_orientation == "auto_from_netlist":
        if netlist_path is None:
            raise ValueError(
                "matrix_orientation='auto_from_netlist' requires netlist_path. "
                "Use matrix_orientation='pre_rows_post_columns' when rows are presynaptic sources "
                "and columns are postsynaptic targets, or 'post_rows_pre_columns' when rows are "
                "postsynaptic targets and columns are presynaptic sources."
            )
        input_orientation = "post_rows_pre_columns" if err_as_is <= err_transposed else "pre_rows_post_columns"
    else:
        input_orientation = requested_orientation

    if input_orientation == "post_rows_pre_columns":
        M = raw.to_numpy()
        transposed_input = False
        selected_residual = err_as_is
    else:
        M = raw.to_numpy().T
        transposed_input = True
        selected_residual = err_transposed

    audit.update({
        "input_matrix_orientation": input_orientation,
        "input_matrix_transposed_for_analysis": bool(transposed_input),
        "selected_orientation_residual_vs_netlist_post_pre": selected_residual,
        "selected_orientation": (
            "input already rows=post, columns=pre"
            if not transposed_input
            else "input rows=pre, columns=post; transposed to rows=post, columns=pre"
        ),
    })
    if (
        netlist_path is not None
        and netlist_orientation_tolerance is not None
        and selected_residual is not None
        and selected_residual > netlist_orientation_tolerance
    ):
        raise ValueError(
            f"Selected matrix_orientation={matrix_orientation!r} gives residual "
            f"{selected_residual:.3g} against the netlist in M[post, pre] convention. "
            "Use matrix_orientation='auto_from_netlist' to select automatically, "
            "or choose the opposite explicit orientation if the netlist is authoritative."
        )
    return M, labels, edges, audit


def graph_from_matrix(M, labels, theta=0.0):
    G = nx.DiGraph(); G.add_nodes_from(labels)
    for post, pre in zip(*np.nonzero(np.abs(M) > theta)):
        if post != pre:
            G.add_edge(labels[pre], labels[post], weight=float(abs(M[post, pre])), signed_weight=float(M[post, pre]))
    return G


def apply_order(M, order):
    return M[np.ix_(order, order)]


def rcm_order(M):
    B = (np.abs(M) > 0).astype(int); B = np.maximum(B, B.T)
    return list(map(int, reverse_cuthill_mckee(csr_matrix(B), symmetric_mode=True)))


def pagerank_order(M, labels):
    G = graph_from_matrix(M, labels)
    scores = nx.pagerank(G, weight="weight") if G.number_of_edges() else {x: 0 for x in labels}
    # Sources/low importance first, consistent with the paper's description.
    return sorted(range(len(labels)), key=lambda i: (scores[labels[i]], labels[i]))


def dfs_finish_order(M, labels):
    """Reverse DFS finishing order: the flow-graph traversal ordering.

    Unlike ``tarjan_order``, this assigns an order to every node even inside a
    single strongly connected component, because a depth-first traversal visits
    all of them.  Borst & Leibold's Fig. 7 "Tarjan" panel reaches an order index
    near 1.0 on dense single-SCC benchmark graphs, which condensation plus
    topological sort cannot do; this is the ordering that reproduces it.
    Deterministic: roots are taken in label order and successors in edge
    insertion order.
    """
    G = graph_from_matrix(M, labels)
    seen = set()
    finished = []
    for root in labels:
        if root in seen:
            continue
        seen.add(root)
        stack = [(root, iter(G.successors(root)))]
        while stack:
            node, children = stack[-1]
            for child in children:
                if child not in seen:
                    seen.add(child)
                    stack.append((child, iter(G.successors(child))))
                    break
            else:
                finished.append(node)
                stack.pop()
    index = {x: i for i, x in enumerate(labels)}
    # Finishing order lists sinks first; reverse it so sources lead.
    return [index[x] for x in reversed(finished)]


def anatomical_order(labels):
    return sorted(range(len(labels)), key=lambda i: (REGION_ORDER.get(region(labels[i]), 99), labels[i]))


def normalized_laplacian(M):
    """Absolute, symmetrized normalized Laplacian and its ordered eigensystem."""
    W=np.abs(M)+np.abs(M.T); np.fill_diagonal(W,0)
    degree=W.sum(axis=1); inv=np.zeros_like(degree); keep=degree>0; inv[keep]=1/np.sqrt(degree[keep])
    L=np.eye(len(M))-inv[:,None]*W*inv[None,:]
    # Standard convention gives zero rows/columns for isolates.
    L[~keep,:]=0; L[:,~keep]=0
    vals,vecs=eigh(L)
    return L,vals,vecs


def fiedler_order(M):
    L,vals,vecs=normalized_laplacian(M)
    positive=np.flatnonzero(vals>1e-10)
    if not len(positive): return list(range(len(M)))
    order=list(map(int,np.argsort(vecs[:,positive[0]])))
    # Sign is arbitrary; orient by the paper's lower-triangle direction.
    base=int(np.count_nonzero(np.triu(np.abs(M)>0,1)))
    a=ordering_metrics(M,order,base,0)["upper_triangle_edges"]
    b=ordering_metrics(M,order[::-1],base,0)["upper_triangle_edges"]
    return order if a<=b else order[::-1]


def _local_feedback_cost(M, order):
    X = apply_order(M, order)
    r, c = np.indices(X.shape)
    return (int(np.count_nonzero(np.triu(np.abs(X) > 0, 1))),
            float(np.sum(np.abs(X)[r < c] * (c-r)[r < c])))


def _improve_scc_order(M, nodes, max_passes=12):
    """Modified-Tarjan step: adjacent swaps minimize upper entries, then length."""
    order = list(nodes)
    best = _local_feedback_cost(M, order)
    for _ in range(max_passes):
        changed = False
        for k in range(len(order)-1):
            trial = order.copy(); trial[k], trial[k+1] = trial[k+1], trial[k]
            score = _local_feedback_cost(M, trial)
            if score < best:
                order, best, changed = trial, score, True
        if not changed: break
    return order


def tarjan_order(M, labels, modified=False):
    G = graph_from_matrix(M, labels)
    sccs = list(nx.strongly_connected_components(G))
    C = nx.condensation(G, sccs)
    topo = list(nx.topological_sort(C))
    label_to_i = {x: i for i, x in enumerate(labels)}
    order = []
    for cid in topo:
        members = [label_to_i[x] for x in C.nodes[cid]["members"]]
        members.sort(key=lambda i: labels[i])
        if modified and len(members) > 1:
            sub = M[np.ix_(members, members)]
            local = _improve_scc_order(sub, range(len(members)))
            members = [members[i] for i in local]
        order.extend(members)
    return order, sccs


def ordering_metrics(M, order, baseline_upper, runtime_s):
    X = apply_order(M, order); B = np.abs(X) > 0
    upper_count = int(np.count_nonzero(np.triu(B, 1)))
    lower_count = int(np.count_nonzero(np.tril(B, -1)))
    upper_weight = float(np.sum(np.abs(np.triu(X, 1))))
    lower_weight = float(np.sum(np.abs(np.tril(X, -1))))
    r, c = np.nonzero(B & ~np.eye(len(X), dtype=bool))
    bandwidth = int(np.max(np.abs(r-c))) if len(r) else 0
    return {"upper_triangle_edges": upper_count,
            "recurrency_excess_ratio": upper_count/max(baseline_upper, 1),
            "feedforward_edge_fraction": lower_count/max(upper_count+lower_count, 1),
            "order_index": float(np.count_nonzero(np.diag(B, -1))/max(len(X)-1, 1)),
            "upper_abs_weight": upper_weight, "lower_abs_weight": lower_weight,
            "weighted_feedforward_fraction": lower_weight/max(upper_weight+lower_weight, np.finfo(float).eps),
            "bandwidth": bandwidth, "runtime_s": runtime_s}


def compare_methods(M, labels):
    baseline = int(np.count_nonzero(np.triu(np.abs(M) > 0, 1)))
    makers = {
        "Original": lambda: list(range(len(labels))),
        "Anatomical": lambda: anatomical_order(labels),
        "Laplacian Fiedler": lambda: fiedler_order(M),
        "Reverse Cuthill-McKee": lambda: rcm_order(M),
        "PageRank": lambda: pagerank_order(M, labels),
        "Tarjan SCC": lambda: tarjan_order(M, labels, False)[0],
        "Tarjan DFS order": lambda: dfs_finish_order(M, labels),
        "Modified Tarjan": lambda: tarjan_order(M, labels, True)[0],
    }
    rows, orders = [], {}
    for name, maker in makers.items():
        t0 = time.perf_counter(); order = maker(); elapsed = time.perf_counter()-t0
        orders[name] = order
        rows.append({"method": name, **ordering_metrics(M, order, baseline, elapsed)})
    scores = pd.DataFrame(rows)
    # Paper keeps objectives separate; these ranks deliberately are not collapsed.
    scores["recurrency_rank"] = scores["recurrency_excess_ratio"].rank(method="min")
    scores["order_index_rank"] = scores["order_index"].rank(ascending=False, method="min")
    scores["runtime_rank"] = scores["runtime_s"].rank(method="min")
    return scores.sort_values(["recurrency_rank", "order_index_rank"]), orders


def laplacian_bridge_analysis(M, labels, threshold_details, U, paths, k_subspace=6):
    """Connect SCC topology and graph geometry to Schur/eigenmode dynamics."""
    L,lev,F=normalized_laplacian(M)
    positive=np.flatnonzero(lev>1e-10); fidx=positive[0] if len(positive) else 0
    f=F[:,fidx]
    threshold_rows=[]
    for q,(Mt,torder,_) in threshold_details.items():
        Lt,vals,_=normalized_laplacian(Mt)
        ncomp=int(np.count_nonzero(vals<1e-10))
        pos=np.flatnonzero(vals>1e-10)
        lambda2=float(vals[1]) if len(vals)>1 else 0.0
        first_positive=float(vals[pos[0]]) if len(pos) else 0.0
        forder=fiedler_order(Mt)
        rank_t=np.empty(len(M),int); rank_f=np.empty(len(M),int)
        rank_t[torder]=np.arange(len(M)); rank_f[forder]=np.arange(len(M))
        tau=float(abs(kendalltau(rank_t,rank_f).statistic))
        threshold_rows.append({"percentile":q,"laplacian_components":ncomp,
                               "algebraic_connectivity":lambda2,
                               "first_positive_laplacian_eigenvalue":first_positive,
                               "abs_kendall_tarjan_fiedler":tau})
    A=M-np.eye(len(M)); eigvals,eigvecs=np.linalg.eig(A); idx=np.argsort(eigvals.real)[::-1]
    low_indices=positive[:min(k_subspace,len(positive))]
    Flow=F[:,low_indices]
    mode_rows=[]
    for rank_,j in enumerate(idx[:12],1):
        v=eigvecs[:,j]; denom=float(np.real(v.conj().T@v))
        energy=float(np.real(v.conj().T@L@v)/denom)
        participation=float(np.linalg.norm(Flow.T.conj()@v)**2/denom) if Flow.size else np.nan
        mode_rows.append({"rank":rank_,"laplacian_energy":energy,
                          f"low_frequency_participation_k{k_subspace}":participation})
    # Slow invariant Schur subspace, selected by real part of eigenvalues.
    k=min(k_subspace,len(M)-1); cutoff=np.sort(eigvals.real)[::-1][k-1]-1e-12
    _,Qslow,sdim=schur(A,output="complex",sort=lambda x:x.real>=cutoff)
    Qslow=Qslow[:,:sdim]
    angles=np.degrees(subspace_angles(Flow,Qslow)) if Flow.size and Qslow.size else np.array([])
    overlap=pd.DataFrame({"principal_angle_index":np.arange(1,len(angles)+1),
                          "angle_degrees":angles,"cosine_squared":np.cos(np.radians(angles))**2})
    leakage=[]
    for kk in (2,4,6,8,12):
        inds=positive[:min(kk,len(positive))]; Fk=F[:,inds]
        if not Fk.size: continue
        Ak=Fk.T@A@Fk; denom=np.linalg.norm(A@Fk,"fro")
        leakage.append({"k_laplacian_modes":len(inds),"galerkin_leakage":float(np.linalg.norm(A@Fk-Fk@Ak,"fro")/max(denom,np.finfo(float).eps))})
    path_rows=[]
    if not paths.empty:
        pos={x:i for i,x in enumerate(labels)}
        for row in paths.itertuples(index=False):
            coords=np.array([f[pos[getattr(row,c)]] for c in ("EC","DG","CA3","CA1")])
            path_rows.append({"EC":row.EC,"DG":row.DG,"CA3":row.CA3,"CA1":row.CA1,
                              "fiedler_monotonic":bool(np.all(np.diff(coords)>0) or np.all(np.diff(coords)<0))})
    return {"L":L,"eigenvalues":lev,"fiedler":f,"thresholds":pd.DataFrame(threshold_rows),
            "mode_metrics":pd.DataFrame(mode_rows),"subspace_overlap":overlap,
            "leakage":pd.DataFrame(leakage),"path_monotonicity":pd.DataFrame(path_rows)}


def _binary_metrics(X, reference_upper):
    B = np.abs(X) > 0
    upper = int(np.count_nonzero(np.triu(B, 1)))
    return upper / max(reference_upper, 1), float(np.count_nonzero(np.diag(B, -1)) / max(len(X)-1, 1))


def paper_synthetic_benchmark(sizes=DEFAULT_BENCHMARK_SIZES, repeats=15, seed=20230817,
                              capture=(60, 0)):
    """Known-order scramble benchmark modeled on Fig. 7 (median and 90% interval).

    ``capture=(n, repeat)`` additionally retains the matrices of that one
    realization so the Fig. 7 top row can be drawn from exactly the realization
    the summary curves measure, rather than from a separately seeded draw.
    """
    rng = np.random.default_rng(seed); rows=[]
    capture_n, capture_rep = (capture if capture else (None, None))
    if capture_n is not None and capture_n not in sizes:
        capture_n = sizes[len(sizes)//2]
    panels = {}
    for n in sizes:
        labels=[f"n{i}" for i in range(n)]
        for rep in range(repeats):
            base=np.zeros((n,n),float)
            lower=np.tril_indices(n,-1); upper=np.triu_indices(n,1)
            lm=rng.random(len(lower[0])) < 0.50
            um=rng.random(len(upper[0])) < 0.25
            base[lower[0][lm],lower[1][lm]]=1
            base[upper[0][um],upper[1][um]]=1
            ref_upper=int(np.count_nonzero(np.triu(base,1)))
            perm=rng.permutation(n); scrambled=apply_order(base,perm)
            makers={"Scrambled":lambda:list(range(n)),
                    "Reverse Cuthill-McKee":lambda:rcm_order(scrambled),
                    "Laplacian Fiedler":lambda:fiedler_order(scrambled),
                    "PageRank":lambda:pagerank_order(scrambled,labels),
                    "Tarjan SCC":lambda:tarjan_order(scrambled,labels,False)[0],
                    "Tarjan DFS order":lambda:dfs_finish_order(scrambled,labels),
                    "Modified Tarjan":lambda:tarjan_order(scrambled,labels,True)[0]}
            capturing = (n==capture_n and rep==capture_rep)
            if capturing:
                G=graph_from_matrix(scrambled,labels)
                sccs=list(nx.strongly_connected_components(G))
                ref_metrics=_binary_metrics(base,ref_upper)
                panels={"n_nodes":n,"repeat":rep,"reference_upper_edges":ref_upper,
                        "n_scc":len(sccs),"largest_scc":max(map(len,sccs)),
                        "edges":G.number_of_edges(),
                        "matrices":{"M original":base.copy(),"M scrambled":scrambled.copy()},
                        "annotations":{"M original":{"excess":ref_metrics[0]-1,
                                                     "order_index":ref_metrics[1],"runtime_s":np.nan}},
                        "order":["M original","M scrambled"]}
            for method,maker in makers.items():
                t0=time.perf_counter(); order=maker(); runtime=time.perf_counter()-t0
                X=apply_order(scrambled,order)
                rec,oi=_binary_metrics(X,ref_upper)
                rows.append({"n_nodes":n,"repeat":rep,"method":method,
                             "recurrency_excess_ratio":rec,"order_index":oi,"runtime_s":runtime})
                if capturing and method!="Scrambled":
                    panels["matrices"][method]=X
                    # Paper reports the INCREASE relative to the reference, i.e. ratio - 1.
                    panels["annotations"][method]={"excess":rec-1,"order_index":oi,"runtime_s":runtime}
                    panels["order"].append(method)
    raw=pd.DataFrame(rows)
    summary=(raw.groupby(["n_nodes","method"])[["recurrency_excess_ratio","order_index","runtime_s"]]
             .agg(["median",lambda x:np.quantile(x,.05),lambda x:np.quantile(x,.95)]).reset_index())
    summary.columns=["n_nodes","method"]+[f"{metric}_{stat}" for metric,stat in summary.columns[2:]]
    summary=summary.rename(columns={c:c.replace("<lambda_0>","q05").replace("<lambda_1>","q95") for c in summary.columns})
    return raw,summary,panels


FIG7_PANELS = ("M original", "M scrambled", "Tarjan SCC", "Tarjan DFS order",
               "Modified Tarjan", "PageRank")


def plot_fig7_top_row(panels, out, shown=FIG7_PANELS):
    """Recreate the Fig. 7 top row from the captured benchmark realization."""
    if not panels:
        return None
    figdir = Path(out)/"figures"; figdir.mkdir(parents=True, exist_ok=True)
    names = [x for x in shown if x in panels["matrices"]]
    ncol = 3; nrow = int(np.ceil(len(names)/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2*ncol, 4.8*nrow), constrained_layout=True)
    for ax, name in zip(np.atleast_1d(axes).flat, names):
        ax.imshow(np.abs(panels["matrices"][name]) > 0, cmap="Greens",
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(name, fontsize=11)
        ax.tick_params(labelsize=7)
        ax.set_ylabel("postsynaptic", fontsize=8)
        a = panels["annotations"].get(name)
        note = "presynaptic"
        if a is not None:
            note += f"\nupper: {a['excess']:+.2f}    order: {a['order_index']:.2f}"
            if np.isfinite(a["runtime_s"]):
                note += f"    time: {a['runtime_s']:.3f} s"
        ax.set_xlabel(note, fontsize=8)
    for ax in np.atleast_1d(axes).flat[len(names):]:
        ax.axis("off")
    fig.suptitle(
        f"Fig. 7 top row recreated: N={panels['n_nodes']}, repeat {panels['repeat']}, "
        f"{panels['edges']} edges, {panels['n_scc']} SCC "
        f"(largest {panels['largest_scc']}); 'upper' is the excess over the reference (ratio - 1)")
    fig.savefig(figdir/"paper_fig7_top_row.png", dpi=180); plt.close(fig)
    return figdir/"paper_fig7_top_row.png"


# Values printed on the Fig. 7 top row of Borst & Leibold (2023), N = 60.
# "upper" there is the INCREASE in upper-triangle entries relative to the reference.
PAPER_FIG7_REFERENCE = {
    "Tarjan":     {"paper_upper_excess": 0.48, "paper_order_index": 1.00, "paper_runtime_s": 0.02},
    "Tarjan mod": {"paper_upper_excess": 0.39, "paper_order_index": 0.98, "paper_runtime_s": 0.10},
    "BFA":        {"paper_upper_excess": 0.10, "paper_order_index": 0.71, "paper_runtime_s": 31.12},
}
# Which local method stands opposite which published panel.  PageRank is a
# stand-in for BFA on the recurrency objective only; the brute-force code was
# never published, so this is a comparison of role, not of algorithm.
FIG7_PANEL_TO_PAPER = {
    "Tarjan DFS order": "Tarjan",
    "Modified Tarjan": "Tarjan mod",
    "PageRank": "BFA (stand-in)",
}


def compare_to_published_fig7(panel_metrics):
    """Line up this run's captured panels against the published Fig. 7 values."""
    if panel_metrics is None or panel_metrics.empty:
        return pd.DataFrame()
    rows = []
    for row in panel_metrics.itertuples(index=False):
        paper_name = FIG7_PANEL_TO_PAPER.get(row.panel)
        if paper_name is None:
            continue
        published = PAPER_FIG7_REFERENCE[paper_name.replace(" (stand-in)", "")]
        rows.append({
            "this_run_method": row.panel,
            "published_panel": paper_name,
            "upper_excess_here": row.upper_triangle_excess_vs_reference,
            "upper_excess_paper": published["paper_upper_excess"],
            "order_index_here": row.order_index,
            "order_index_paper": published["paper_order_index"],
            "runtime_s_here": row.runtime_s,
            "runtime_s_paper": published["paper_runtime_s"],
        })
    return pd.DataFrame(rows)


def fig7_panel_metrics(panels):
    """Tabulate the annotations drawn on the recreated Fig. 7 top row."""
    if not panels:
        return pd.DataFrame()
    rows = []
    for name in panels["order"]:
        a = panels["annotations"].get(name)
        if a is None:
            continue
        rows.append({"panel": name, "upper_triangle_excess_vs_reference": a["excess"],
                     "recurrency_excess_ratio": a["excess"]+1,
                     "order_index": a["order_index"], "runtime_s": a["runtime_s"]})
    frame = pd.DataFrame(rows)
    frame.insert(0, "n_nodes", panels["n_nodes"])
    frame.insert(1, "largest_scc", panels["largest_scc"])
    return frame


def _degree_preserving_rewire(M, rng, swaps_per_edge=5):
    """Directed double-edge swaps preserving every node's in/out degree and edge count."""
    B=np.abs(M)>0; np.fill_diagonal(B,False)
    edges=[tuple(x) for x in np.argwhere(B)]  # (post, pre)
    edge_set=set(edges); target=swaps_per_edge*len(edges); accepted=0
    for _ in range(max(target*20,1)):
        if accepted>=target or len(edges)<2: break
        a,b=rng.choice(len(edges),2,replace=False); e1=edges[a]; e2=edges[b]
        post1,pre1=e1; post2,pre2=e2
        n1=(post1,pre2); n2=(post2,pre1)
        if post1==pre2 or post2==pre1 or n1==n2 or n1 in edge_set or n2 in edge_set: continue
        edge_set.remove(e1); edge_set.remove(e2); edge_set.add(n1); edge_set.add(n2)
        edges[a]=n1; edges[b]=n2; accepted+=1
    weights=M[B].copy(); rng.shuffle(weights)
    X=np.zeros_like(M); 
    for edge,w in zip(edges,weights): X[edge]=w
    return X, accepted


def ordering_null_test(M, labels, n_null=100, seed=20230818):
    """Compare observed optimized structure with directed degree-preserving rewires."""
    rng=np.random.default_rng(seed)
    observed_order=tarjan_order(M,labels,True)[0]
    obs=ordering_metrics(M,observed_order,int(np.count_nonzero(np.triu(np.abs(M)>0,1))),0)
    rows=[]
    for k in range(n_null):
        X,accepted=_degree_preserving_rewire(M,rng)
        order=tarjan_order(X,labels,True)[0]
        met=ordering_metrics(X,order,int(np.count_nonzero(np.triu(np.abs(X)>0,1))),0)
        rows.append({"null_repeat":k,"accepted_edge_swaps":accepted,**met})
    raw=pd.DataFrame(rows); tests=[]
    for metric,direction in [("recurrency_excess_ratio","lower"),("order_index","higher"),
                             ("feedforward_edge_fraction","higher"),("weighted_feedforward_fraction","higher")]:
        vals=raw[metric].to_numpy(); value=float(obs[metric])
        extreme=np.sum(vals<=value) if direction=="lower" else np.sum(vals>=value)
        tests.append({"metric":metric,"better_direction":direction,"observed":value,
                      "null_mean":float(vals.mean()),"null_sd":float(vals.std(ddof=1)),
                      "z_score":float((value-vals.mean())/vals.std(ddof=1)) if vals.std(ddof=1)>0 else np.nan,
                      "empirical_p_one_sided":float((extreme+1)/(n_null+1)),"n_null":n_null})
    return raw,pd.DataFrame(tests)


def threshold_sweep(M, labels, percentiles):
    nz = np.abs(M[np.nonzero(M)]); rows=[]; details={}
    for q in percentiles:
        theta = 0.0 if q == 0 else float(np.percentile(nz, q))
        Mt = M.copy(); Mt[np.abs(Mt) < theta] = 0; np.fill_diagonal(Mt, 0)
        order, sccs = tarjan_order(Mt, labels, True)
        met = ordering_metrics(Mt, order, int(np.count_nonzero(np.triu(np.abs(Mt)>0, 1))), 0.0)
        G = graph_from_matrix(Mt, labels)
        rows.append({"percentile": q, "theta": theta, "edges_retained": G.number_of_edges(),
                     "retained_abs_weight_fraction": float(np.sum(np.abs(Mt))/np.sum(np.abs(M))),
                     "n_scc": len(sccs), "largest_scc": max(map(len, sccs)),
                     "is_dag": nx.is_directed_acyclic_graph(G), **met})
        details[q] = (Mt, order, sccs)
    return pd.DataFrame(rows), details


def normalize_spectral_radius(M, target=1.0):
    """Return a positive scalar normalization with spectral radius ``target``."""
    radius = float(np.max(np.abs(np.linalg.eigvals(M))))
    if not np.isfinite(radius) or radius <= np.finfo(float).eps:
        raise ValueError(f"Cannot normalize matrix with spectral radius {radius}")
    return (target / radius) * M, radius


def apply_normalization(M, normalization="spectral_radius", spectral_radius_target=1.0):
    """Apply the requested matrix normalization and report spectral-radius metadata."""
    mode = "none" if normalization is None else str(normalization).strip().lower()
    aliases = {"false": "none", "no": "none", "off": "none", "raw": "none", "rho": "spectral_radius"}
    mode = aliases.get(mode, mode)
    if mode not in NORMALIZATION_MODES:
        raise ValueError(f"normalization must be one of {sorted(NORMALIZATION_MODES)}; got {normalization!r}")
    raw_radius = float(np.max(np.abs(np.linalg.eigvals(M))))
    if mode == "none":
        return M.copy(), {
            "normalization": "none",
            "raw_spectral_radius": raw_radius,
            "spectral_radius_target": None,
            "analysis_spectral_radius": raw_radius,
        }
    M_normalized, _ = normalize_spectral_radius(M, spectral_radius_target)
    return M_normalized, {
        "normalization": "spectral_radius",
        "raw_spectral_radius": raw_radius,
        "spectral_radius_target": float(spectral_radius_target),
        "analysis_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(M_normalized)))),
    }


def export_paper_convention_inputs(M_raw, M_analysis, labels, edges, audit, output_dir):
    """Write explicit canonical inputs without modifying the user's source files."""
    canonical=Path(output_dir)/"canonical_inputs"; canonical.mkdir(parents=True,exist_ok=True)
    # The CSV top-left header explicitly states both axes; the JSON manifest
    # separately records them as machine-readable fields.
    index=pd.Index(labels,name="post_neuron \\ pre_neuron")
    columns=pd.Index(labels,name="pre_neuron")
    raw_path=canonical/"mij_matrix_paper_convention_raw.csv"
    analysis_path=canonical/"mij_matrix_paper_convention_analysis.csv"
    netlist_path=canonical/"mij_netlist_explicit_pre_to_post.csv"
    pd.DataFrame(M_raw,index=index,columns=columns).to_csv(raw_path)
    pd.DataFrame(M_analysis,index=index,columns=columns).to_csv(analysis_path)
    if edges is not None and len(edges):
        edges.to_csv(netlist_path,index=False)
        netlist_output = str(netlist_path)
    else:
        netlist_output = None
    manifest={**audit,
              "matrix_entry_definition":"M_ij is the weight from presynaptic cell j to postsynaptic cell i",
              "row_axis":"post_neuron","column_axis":"pre_neuron",
              "graph_edge_rule":"column j -> row i",
              "feedforward_triangle_after_ordering":"lower",
              "raw_matrix":str(raw_path),"analysis_matrix":str(analysis_path),
              "netlist":netlist_output}
    with open(canonical/"paper_convention_manifest.json","w") as f: json.dump(manifest,f,indent=2)
    return manifest


def schur_and_modes(M, labels):
    """Analyze A=M-I; caller supplies the already normalized dimensionless M."""
    A = M - np.eye(len(M))
    R, U = schur(A, output="real")  # A = U R U^T; standard SciPy upper triangular
    eigvals, eigvecs = np.linalg.eig(A)
    idx = np.argsort(eigvals.real)[::-1]
    mode_rows=[]
    for rank_, k in enumerate(idx[:12], 1):
        v=eigvecs[:, k]; participation=np.abs(v)**2; participation/=participation.sum()
        top=np.argsort(participation)[::-1][:5]
        mode_rows.append({"rank": rank_, "eigenvalue_real": eigvals[k].real,
                          "eigenvalue_imag": eigvals[k].imag,
                          "decay_time_tau_units": -1/eigvals[k].real if eigvals[k].real < 0 else np.inf,
                          "top_cell_types": " | ".join(labels[i] for i in top),
                          "top_participation": " | ".join(f"{participation[i]:.3f}" for i in top)})
    schur_rows=[]
    for k in range(min(12, len(labels))):
        p=U[:, k]**2; p/=p.sum(); top=np.argsort(p)[::-1][:5]
        schur_rows.append({"schur_coordinate": k+1, "R_diagonal": R[k,k],
                           "top_cell_types": " | ".join(labels[i] for i in top),
                           "top_basis_weights_sq": " | ".join(f"{p[i]:.3f}" for i in top)})
    return A, R, U, pd.DataFrame(mode_rows), pd.DataFrame(schur_rows)


def trisynaptic_paths(M, labels, order, top_n=30):
    """Rank EC -> DG -> CA3 -> CA1 paths by geometric mean absolute M weight."""
    groups={r:[i for i,x in enumerate(labels) if region(x)==r] for r in ("EC","DG","CA3","CA1")}
    pos={node:k for k,node in enumerate(order)}; rows=[]
    for ec in groups["EC"]:
      for dg in groups["DG"]:
       a=abs(M[dg,ec])
       if not a: continue
       for ca3 in groups["CA3"]:
        b=abs(M[ca3,dg])
        if not b: continue
        for ca1 in groups["CA1"]:
         c=abs(M[ca1,ca3])
         if not c: continue
         seq=[ec,dg,ca3,ca1]
         rows.append({"EC":labels[ec],"DG":labels[dg],"CA3":labels[ca3],"CA1":labels[ca1],
                      "path_score_geomean":float((a*b*c)**(1/3)),
                      "sign_product":int(np.sign(M[dg,ec]*M[ca3,dg]*M[ca1,ca3])),
                      "respects_directional_order":all(pos[seq[k]]<pos[seq[k+1]] for k in range(3))})
    return pd.DataFrame(rows).sort_values("path_score_geomean", ascending=False).head(top_n) if rows else pd.DataFrame()


def plot_outputs(M, labels, scores, orders, sweep, details, R, U, benchmark_summary, null_raw, null_tests, lapbridge, out, benchmark_panels=None):
    figdir=out/"figures"; figdir.mkdir(parents=True, exist_ok=True)
    shown=["Original","Reverse Cuthill-McKee","PageRank","Tarjan SCC","Modified Tarjan","Anatomical"]
    fig,axes=plt.subplots(2,3,figsize=(15,10))
    vmax=np.max(np.log1p(np.abs(M)))
    for ax,name in zip(axes.flat,shown):
        X=apply_order(M,orders[name]); Z=np.sign(X)*np.log1p(np.abs(X))
        ax.imshow(Z,cmap="RdBu_r",norm=TwoSlopeNorm(vcenter=0,vmin=-vmax,vmax=vmax),aspect="auto")
        ax.set_title(name); ax.set_xlabel("pre"); ax.set_ylabel("post")
    fig.suptitle("Signed log Mij under shared row/column permutations"); fig.tight_layout(); fig.savefig(figdir/"method_matrix_comparison.png",dpi=180); plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(14,4))
    s=scores.set_index("method")
    s["recurrency_excess_ratio"].sort_values().plot.barh(ax=axes[0],title="Recurrency (lower better)")
    s["order_index"].sort_values().plot.barh(ax=axes[1],title="Order index (higher better)")
    s["runtime_s"].sort_values().plot.barh(ax=axes[2],title="Runtime (s; lower better)")
    fig.tight_layout(); fig.savefig(figdir/"paper_style_method_ranking.png",dpi=180); plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(14,4))
    axes[0].plot(sweep.percentile,sweep.largest_scc,"o-"); axes[0].set(title="Largest recurrent block",xlabel="|Mij| percentile",ylabel="nodes")
    axes[1].plot(sweep.percentile,sweep.weighted_feedforward_fraction,"o-"); axes[1].set(title="Weight below diagonal",xlabel="|Mij| percentile",ylabel="fraction")
    axes[2].plot(sweep.percentile,sweep.edges_retained,"o-"); axes[2].set(title="Edges retained",xlabel="|Mij| percentile",ylabel="edges")
    fig.tight_layout(); fig.savefig(figdir/"tarjan_threshold_sweep.png",dpi=180); plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].imshow(np.sign(R)*np.log1p(np.abs(R)),cmap="RdBu_r",aspect="auto"); axes[0].set_title("Real Schur form R (orthogonal basis)")
    axes[1].imshow(U,cmap="RdBu_r",aspect="auto"); axes[1].set_title("Orthogonal population basis U")
    fig.tight_layout(); fig.savefig(figdir/"schur_operator_and_basis.png",dpi=180); plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for method,g in benchmark_summary.groupby("method"):
        g=g.sort_values("n_nodes")
        for ax,metric,title in zip(axes,["recurrency_excess_ratio","order_index","runtime_s"],
                                   ["Known-order recurrency","First-subdiagonal order index","Runtime scaling"]):
            ax.plot(g.n_nodes,g[f"{metric}_median"],"o-",label=method)
            ax.fill_between(g.n_nodes,g[f"{metric}_q05"],g[f"{metric}_q95"],alpha=.12)
            ax.set(xlabel="nodes",title=title)
    for ax in axes: ax.set_xscale("log")
    axes[2].set_yscale("log"); axes[0].set_ylabel("ratio (lower better)"); axes[1].set_ylabel("fraction (higher better)")
    axes[2].set_ylabel("seconds (log scale)")
    # Paper Fig. 7 draws a dashed N^2 ln N reference, scaled by an arbitrary constant.
    sizes=np.sort(benchmark_summary.n_nodes.unique()).astype(float)
    if len(sizes)>1:
        theory=sizes**2*np.log(sizes)
        anchor_row=benchmark_summary.loc[benchmark_summary.n_nodes==sizes[0],"runtime_s_median"]
        if len(anchor_row):
            theory=theory/theory[0]*float(anchor_row.max())
            axes[2].plot(sizes,theory,"k--",lw=1,label=r"$\propto N^2\ln N$")
    axes[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figdir/"paper_synthetic_benchmark.png",dpi=180); plt.close(fig)

    metrics=["recurrency_excess_ratio","order_index","feedforward_edge_fraction","weighted_feedforward_fraction"]
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    test_index=null_tests.set_index("metric")
    for ax,metric in zip(axes.flat,metrics):
        ax.hist(null_raw[metric],bins=18,color="#9aa7b2",edgecolor="white")
        ax.axvline(test_index.loc[metric,"observed"],color="#b22222",lw=2,label="observed")
        ax.set_title(f"{metric}\np={test_index.loc[metric,'empirical_p_one_sided']:.3g}")
        ax.legend()
    fig.suptitle("Degree-preserving topology null: Modified Tarjan"); fig.tight_layout(); fig.savefig(figdir/"ordering_null_test.png",dpi=180); plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(12,8))
    lt=lapbridge["thresholds"]
    axes[0,0].plot(lt.percentile,lt.algebraic_connectivity,"o-"); axes[0,0].set(title="Algebraic connectivity",xlabel="|Mij| percentile",ylabel=r"$\lambda_2$")
    axes[0,1].plot(lt.percentile,lt.abs_kendall_tarjan_fiedler,"o-"); axes[0,1].set(title="Tarjan-Fiedler agreement",xlabel="|Mij| percentile",ylabel=r"$|\tau|$")
    mm=lapbridge["mode_metrics"]
    axes[1,0].scatter(mm.laplacian_energy,mm.iloc[:,-1],c=mm["rank"],cmap="viridis"); axes[1,0].set(title="Dynamical-mode graph structure",xlabel="Laplacian energy",ylabel="low-frequency participation")
    leak=lapbridge["leakage"]; axes[1,1].plot(leak.k_laplacian_modes,leak.galerkin_leakage,"o-"); axes[1,1].set(title="Laplacian reduced-model closure",xlabel="retained modes k",ylabel="Galerkin leakage")
    fig.suptitle("Integrated Tarjan-Laplacian-Schur bridge"); fig.tight_layout(); fig.savefig(figdir/"laplacian_bridge.png",dpi=180); plt.close(fig)

    plot_fig7_top_row(benchmark_panels, out)


def run_analysis(matrix_path="mij_matrix.csv", netlist_path="mij_netlist.csv", output_dir="outputs/paper_matrix_reordering",
                 percentiles=DEFAULT_PERCENTILES, matrix_orientation="auto_from_netlist",
                 normalization="spectral_radius", spectral_radius_target=1.0,
                 benchmark_repeats=15, n_null=100, netlist_orientation_tolerance=1e-10):
    out=Path(output_dir); (out/"tables").mkdir(parents=True,exist_ok=True)
    M_raw,labels,edges,audit=load_inputs(
        Path(matrix_path),
        _optional_path(netlist_path),
        matrix_orientation,
        netlist_orientation_tolerance,
    )
    M, normalization_audit = apply_normalization(M_raw, normalization, spectral_radius_target)
    audit.update(normalization_audit)
    canonical_manifest=export_paper_convention_inputs(M_raw,M,labels,edges,audit,out)
    scores,orders=compare_methods(M,labels)
    sweep,details=threshold_sweep(M,labels,percentiles)
    A,R,U,modes,schur_modes=schur_and_modes(M,labels)
    benchmark_raw,benchmark_summary,benchmark_panels=paper_synthetic_benchmark(repeats=benchmark_repeats)
    benchmark_panel_metrics=fig7_panel_metrics(benchmark_panels)
    fig7_vs_paper=compare_to_published_fig7(benchmark_panel_metrics)
    null_raw,null_tests=ordering_null_test(M,labels,n_null=n_null)
    best=scores.sort_values(["recurrency_rank","order_index_rank"]).iloc[0].method
    paths=trisynaptic_paths(M,labels,orders[best])
    lapbridge=laplacian_bridge_analysis(M,labels,details,U,paths)
    modes=modes.merge(lapbridge["mode_metrics"],on="rank",how="left")
    membership=[]
    for q,(_,order,sccs) in details.items():
        omap={i:k for k,i in enumerate(order)}
        for sid,comp in enumerate(sorted(sccs,key=lambda x:-len(x)),1):
            for cell in comp: membership.append({"percentile":q,"scc_rank_by_size":sid,"scc_size":len(comp),"cell_type":cell,"order_position":omap[labels.index(cell)]})
    tables={"method_rankings":scores,"threshold_sweep":sweep,"scc_membership":pd.DataFrame(membership),
            "dominant_eigenmodes":modes,"schur_population_basis":schur_modes,"trisynaptic_paths":paths,
            "synthetic_benchmark_raw":benchmark_raw,"synthetic_benchmark_summary":benchmark_summary,
            "fig7_top_row_panel_metrics":benchmark_panel_metrics,
            "fig7_this_run_vs_published":fig7_vs_paper,
            "ordering_null_raw":null_raw,"ordering_null_tests":null_tests}
    tables.update({"laplacian_threshold_bridge":lapbridge["thresholds"],
                   "laplacian_schur_subspace_overlap":lapbridge["subspace_overlap"],
                   "laplacian_reduced_model_leakage":lapbridge["leakage"],
                   "trisynaptic_fiedler_monotonicity":lapbridge["path_monotonicity"]})
    for name,df in tables.items(): df.to_csv(out/"tables"/f"{name}.csv",index=False)
    with open(out/"input_audit.json","w") as f: json.dump(audit,f,indent=2)
    plot_outputs(M,labels,scores,orders,sweep,details,R,U,benchmark_summary,null_raw,null_tests,lapbridge,out,benchmark_panels)
    summary={"audit":audit,"paper_convention_inputs":canonical_manifest,"best_paper_style_method":best,
             "best_recurrency_method":scores.loc[scores.recurrency_rank.idxmin(),"method"],
             "best_order_index_method":scores.loc[scores.order_index_rank.idxmin(),"method"],
             "threshold_percentiles":list(percentiles),"benchmark_repeats":benchmark_repeats,
             "n_degree_preserving_nulls":n_null,"output_dir":str(out)}
    with open(out/"summary.json","w") as f: json.dump(summary,f,indent=2)
    return {"M":M,"M_paper_raw":M_raw,"M_paper_analysis":M,"M_paper_rho1":M,"labels":labels,"scores":scores,"orders":orders,"sweep":sweep,"details":details,
            "A":A,"R":R,"U":U,"modes":modes,"schur_modes":schur_modes,"paths":paths,
            "benchmark_raw":benchmark_raw,"benchmark_summary":benchmark_summary,
            "benchmark_panels":benchmark_panels,"benchmark_panel_metrics":benchmark_panel_metrics,
            "fig7_vs_paper":fig7_vs_paper,
            "null_raw":null_raw,"null_tests":null_tests,"laplacian_bridge":lapbridge,"summary":summary}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix",default="mij_matrix.csv"); p.add_argument("--netlist",default="mij_netlist.csv")
    p.add_argument("--output",default="outputs/paper_matrix_reordering")
    p.add_argument("--percentiles",nargs="+",type=float,default=list(DEFAULT_PERCENTILES))
    p.add_argument("--matrix-orientation",default="auto_from_netlist", choices=sorted(MATRIX_ORIENTATIONS),
                   help="orientation of the input matrix before analysis")
    p.add_argument("--normalization",default="spectral_radius", choices=sorted(NORMALIZATION_MODES),
                   help="use 'none' to analyze raw weights without spectral-radius scaling")
    p.add_argument("--spectral-radius",type=float,default=1.0,
                   help="target spectral radius for M before all analyses (default: 1)")
    p.add_argument("--benchmark-repeats",type=int,default=15)
    p.add_argument("--n-null",type=int,default=100)
    p.add_argument("--netlist-orientation-tolerance",type=float,default=1e-10,
                   help="maximum allowed residual for the selected matrix orientation when a netlist is provided")
    a=p.parse_args(); result=run_analysis(a.matrix,a.netlist,a.output,a.percentiles,
                                           matrix_orientation=a.matrix_orientation,
                                           normalization=a.normalization,
                                           spectral_radius_target=a.spectral_radius,
                                           benchmark_repeats=a.benchmark_repeats,
                                           n_null=a.n_null,
                                           netlist_orientation_tolerance=a.netlist_orientation_tolerance)
    print(json.dumps(result["summary"],indent=2))


if __name__ == "__main__": main()
