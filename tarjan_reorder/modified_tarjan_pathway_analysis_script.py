#!/usr/bin/env python3
"""Modified-Tarjan pathway reconstruction with stability, null, and modal tests."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.linalg import schur, subspace_angles

from matrix_reordering_paper_analysis_script import (
    _degree_preserving_rewire,
    _optional_path,
    apply_normalization,
    graph_from_matrix,
    load_inputs,
    MATRIX_ORIENTATIONS,
    NORMALIZATION_MODES,
    region,
    tarjan_order,
)

DEFAULT_THRESHOLDS = tuple(range(0, 91, 5))
NULL_THRESHOLDS = (25, 50, 75)


def threshold_matrix(M, q):
    nz=np.abs(M[M!=0]); theta=0.0 if q==0 else float(np.percentile(nz,q))
    X=M.copy(); X[np.abs(X)<theta]=0; np.fill_diagonal(X,0)
    return X,theta


def condensation_model(M, labels):
    """Build a topologically indexed SCC condensation DAG with aggregated edges."""
    G=graph_from_matrix(M,labels)
    sccs=list(nx.strongly_connected_components(G)); raw=nx.condensation(G,sccs)
    topo=list(nx.topological_sort(raw)); relabel={old:new for new,old in enumerate(topo)}
    C=nx.relabel_nodes(raw,relabel,copy=True)
    membership={cell:relabel[cid] for cell,cid in raw.graph["mapping"].items()}
    for cid in C.nodes:
        members=sorted(C.nodes[cid]["members"])
        C.nodes[cid]["members"]=members
        C.nodes[cid]["size"]=len(members)
        C.nodes[cid]["regions"]="|".join(sorted(set(region(x) for x in members)))
    for u,v in C.edges:
        candidates=[]
        for pre in C.nodes[u]["members"]:
            for post in C.nodes[v]["members"]:
                i=labels.index(post); j=labels.index(pre); w=M[i,j]
                if w!=0: candidates.append((abs(w),w,pre,post))
        candidates.sort(reverse=True)
        total=sum(x[0] for x in candidates); best=candidates[0]
        C.edges[u,v].update(abs_weight=total,max_abs_weight=best[0],signed_weight=best[1],
                            representative_pre=best[2],representative_post=best[3],n_cell_edges=len(candidates))
    return G,C,membership


def _canonical_progression(C,path):
    targets=["EC","DG","CA3","CA1"]; k=0
    for node in path:
        regions=set(C.nodes[node]["regions"].split("|"))
        if targets[k] in regions:
            k+=1
            if k==len(targets): return True
    return False


def enumerate_top_dag_paths(C, max_modules=8, top_k=100, beam_per_state=50):
    """Dynamic-programming beam enumeration of weighted paths in the DAG."""
    states=defaultdict(list); completed=[]
    for node in nx.topological_sort(C):
        states[(node,1)].append(([node],0.0,[]))
        incoming=[]
        for pred in C.predecessors(node):
            edge=C.edges[pred,node]; logw=np.log(max(edge["abs_weight"],np.finfo(float).tiny))
            for length in range(1,max_modules):
                for path,score,edges in states.get((pred,length),[]):
                    incoming.append((path+[node],score+logw,edges+[(pred,node)]))
        for path,score,edges in sorted(incoming,key=lambda x:x[1],reverse=True)[:beam_per_state]:
            states[(node,len(path))].append((path,score,edges)); completed.append((path,score,edges))
    rows=[]
    for path,logsum,edges in completed:
        nedge=len(edges); score=float(np.exp(logsum/nedge))
        rep=[f"{C.edges[e]['representative_pre']} -> {C.edges[e]['representative_post']}" for e in edges]
        rows.append({"n_modules":len(path),"module_path":" -> ".join(map(str,path)),
                     "module_sizes":" -> ".join(str(C.nodes[x]["size"]) for x in path),
                     "module_regions":" -> ".join(C.nodes[x]["regions"] for x in path),
                     "representative_cell_edges":" | ".join(rep),"path_score_geomean":score,
                     "min_aggregate_edge_weight":min(C.edges[e]["abs_weight"] for e in edges),
                     "canonical_region_progression":_canonical_progression(C,path)})
    return pd.DataFrame(rows).sort_values("path_score_geomean",ascending=False).head(top_k) if rows else pd.DataFrame()


def canonical_cell_paths(M,labels,top_k=100,modified_order=None):
    groups={r:[i for i,x in enumerate(labels) if region(x)==r] for r in ("EC","DG","CA3","CA1")}
    rows=[]; order_pos={node:k for k,node in enumerate(modified_order)} if modified_order is not None else None
    for ec in groups["EC"]:
      for dg in groups["DG"]:
       w1=M[dg,ec]
       if w1==0: continue
       for ca3 in groups["CA3"]:
        w2=M[ca3,dg]
        if w2==0: continue
        for ca1 in groups["CA1"]:
         w3=M[ca1,ca3]
         if w3==0: continue
         rows.append({"EC":labels[ec],"DG":labels[dg],"CA3":labels[ca3],"CA1":labels[ca1],
                      "path_signature":" -> ".join(labels[x] for x in (ec,dg,ca3,ca1)),
                      "path_score_geomean":float(abs(w1*w2*w3)**(1/3)),
                      "min_edge_weight":float(min(abs(w1),abs(w2),abs(w3))),
                      "sign_product":int(np.sign(w1*w2*w3)),
                      "respects_modified_tarjan_order":(
                          all(order_pos[a]<order_pos[b] for a,b in zip((ec,dg,ca3),(dg,ca3,ca1)))
                          if order_pos is not None else np.nan)})
    return pd.DataFrame(rows).sort_values("path_score_geomean",ascending=False).head(top_k) if rows else pd.DataFrame()


def threshold_pathway_sweep(M,labels,percentiles=DEFAULT_THRESHOLDS,top_k=100):
    summary=[]; dag_tables=[]; canonical_tables=[]; models={}
    for q in percentiles:
        X,theta=threshold_matrix(M,q); G,C,mapping=condensation_model(X,labels)
        modified_order,_=tarjan_order(X,labels,modified=True)
        dag=enumerate_top_dag_paths(C,top_k=top_k); canonical=canonical_cell_paths(X,labels,top_k=top_k,modified_order=modified_order)
        if not dag.empty: dag.insert(0,"percentile",q); dag_tables.append(dag)
        if not canonical.empty: canonical.insert(0,"percentile",q); canonical_tables.append(canonical)
        summary.append({"percentile":q,"theta":theta,"edges":G.number_of_edges(),"n_scc":C.number_of_nodes(),
                        "largest_scc":max((C.nodes[x]["size"] for x in C),default=0),
                        "condensation_edges":C.number_of_edges(),"enumerated_top_paths":len(dag),
                        "canonical_paths_retained":len(canonical),
                        "best_dag_path_score":dag.path_score_geomean.max() if len(dag) else np.nan,
                        "best_canonical_score":canonical.path_score_geomean.max() if len(canonical) else np.nan,
                        "canonical_fraction_top_dag":dag.canonical_region_progression.mean() if len(dag) else np.nan,
                        "canonical_fraction_modified_order":canonical.respects_modified_tarjan_order.mean() if len(canonical) else np.nan})
        models[q]=(X,G,C,mapping)
    dag_all=pd.concat(dag_tables,ignore_index=True) if dag_tables else pd.DataFrame()
    can_all=pd.concat(canonical_tables,ignore_index=True) if canonical_tables else pd.DataFrame()
    return pd.DataFrame(summary),dag_all,can_all,models


def stability_tables(dag_all,canonical_all,n_thresholds):
    if dag_all.empty: dag_stability=pd.DataFrame()
    else:
        dag_stability=(dag_all.groupby("representative_cell_edges")
                       .agg(threshold_count=("percentile","nunique"),first_percentile=("percentile","min"),
                            last_percentile=("percentile","max"),median_score=("path_score_geomean","median"),
                            canonical_region_progression=("canonical_region_progression","max"))
                       .reset_index())
        dag_stability["threshold_stability"]=dag_stability.threshold_count/n_thresholds
        dag_stability=dag_stability.sort_values(["threshold_stability","median_score"],ascending=False)
    if canonical_all.empty: canonical_stability=pd.DataFrame()
    else:
        canonical_stability=(canonical_all.groupby(["path_signature","EC","DG","CA3","CA1"])
                             .agg(threshold_count=("percentile","nunique"),first_percentile=("percentile","min"),
                                  last_percentile=("percentile","max"),median_score=("path_score_geomean","median"),
                                  min_score=("path_score_geomean","min"),sign_product=("sign_product","first"),
                                  modified_order_fraction=("respects_modified_tarjan_order","mean"))
                             .reset_index())
        canonical_stability["threshold_stability"]=canonical_stability.threshold_count/n_thresholds
        canonical_stability=canonical_stability.sort_values(["threshold_stability","median_score"],ascending=False)
    return dag_stability,canonical_stability


def _null_statistics(X,labels,top_k=50):
    _,C,membership=condensation_model(X,labels); dag=enumerate_top_dag_paths(C,top_k=top_k)
    modified_order,_=tarjan_order(X,labels,modified=True)
    can=canonical_cell_paths(X,labels,top_k=top_k,modified_order=modified_order)
    cross_steps=0
    if len(can):
        best=can.iloc[0]; cells=[best[x] for x in ("EC","DG","CA3","CA1")]
        cross_steps=sum(membership[a]!=membership[b] for a,b in zip(cells[:-1],cells[1:]))
    return {"largest_scc":max((C.nodes[x]["size"] for x in C),default=0),
            "n_scc":C.number_of_nodes(),"best_dag_path_score":dag.path_score_geomean.max() if len(dag) else 0,
            "canonical_fraction_top_dag":dag.canonical_region_progression.mean() if len(dag) else 0,
            "best_canonical_score":can.path_score_geomean.max() if len(can) else 0,
            "best_canonical_cross_scc_steps":cross_steps,
            "canonical_fraction_modified_order":can.respects_modified_tarjan_order.mean() if len(can) else 0}


def pathway_null_tests(M,labels,percentiles=NULL_THRESHOLDS,n_null=100,seed=20230819):
    rng=np.random.default_rng(seed); raw=[]; tests=[]
    directions={"largest_scc":"lower","n_scc":"higher","best_dag_path_score":"higher",
                "canonical_fraction_top_dag":"higher","best_canonical_score":"higher",
                "best_canonical_cross_scc_steps":"higher","canonical_fraction_modified_order":"higher"}
    for q in percentiles:
        X,_=threshold_matrix(M,q); observed=_null_statistics(X,labels)
        null_rows=[]
        for b in range(n_null):
            Xn,accepted=_degree_preserving_rewire(X,rng); row=_null_statistics(Xn,labels)
            row.update(percentile=q,null_repeat=b,accepted_edge_swaps=accepted); raw.append(row); null_rows.append(row)
        frame=pd.DataFrame(null_rows)
        for metric,direction in directions.items():
            vals=frame[metric].to_numpy(float); obs=float(observed[metric])
            extreme=np.sum(vals<=obs) if direction=="lower" else np.sum(vals>=obs)
            tests.append({"percentile":q,"metric":metric,"better_direction":direction,"observed":obs,
                          "null_mean":vals.mean(),"null_sd":vals.std(ddof=1),
                          "z_score":((obs-vals.mean())/vals.std(ddof=1)) if vals.std(ddof=1)>0 else np.nan,
                          "empirical_p_one_sided":(extreme+1)/(n_null+1),"n_null":n_null})
    return pd.DataFrame(raw),pd.DataFrame(tests)


def modal_followup(M,labels,canonical_stability,n_paths=20,n_modes=12):
    A=M-np.eye(len(M)); evals,evecs=np.linalg.eig(A); order=np.argsort(evals.real)[::-1]
    R,U=schur(A,output="real"); label_i={x:i for i,x in enumerate(labels)}
    participation=[]; lesions=[]
    for path_rank,row in enumerate(canonical_stability.head(n_paths).itertuples(index=False),1):
        nodes=[label_i[getattr(row,x)] for x in ("EC","DG","CA3","CA1")]; node_set=set(nodes)
        for mode_rank,j in enumerate(order[:n_modes],1):
            v=evecs[:,j]; p=np.abs(v)**2; p/=p.sum()
            participation.append({"path_rank":path_rank,"path_signature":row.path_signature,
                                  "basis":"eigen","mode":mode_rank,"eigenvalue_real":evals[j].real,
                                  "participation":float(p[nodes].sum())})
        for k in range(min(n_modes,U.shape[1])):
            p=U[:,k]**2; p/=p.sum()
            participation.append({"path_rank":path_rank,"path_signature":row.path_signature,
                                  "basis":"schur","mode":k+1,"eigenvalue_real":R[k,k],
                                  "participation":float(p[nodes].sum())})
        Ml=M.copy()
        for pre,post in zip(nodes[:-1],nodes[1:]): Ml[post,pre]=0
        Al=Ml-np.eye(len(M)); lesion_e=np.linalg.eigvals(Al)
        k=min(6,len(M)-1); cutoff=np.sort(evals.real)[::-1][k-1]-1e-12
        _,Q0,s0=schur(A,output="complex",sort=lambda x:x.real>=cutoff)
        cutoff_l=np.sort(lesion_e.real)[::-1][k-1]-1e-12
        _,Q1,s1=schur(Al,output="complex",sort=lambda x:x.real>=cutoff_l)
        angles=np.degrees(subspace_angles(Q0[:,:s0],Q1[:,:s1]))
        lesions.append({"path_rank":path_rank,"path_signature":row.path_signature,
                        "baseline_spectral_abscissa":float(evals.real.max()),
                        "lesioned_spectral_abscissa":float(lesion_e.real.max()),
                        "spectral_abscissa_shift":float(lesion_e.real.max()-evals.real.max()),
                        "max_slow_subspace_angle_deg":float(angles.max()) if len(angles) else np.nan})
    return pd.DataFrame(participation),pd.DataFrame(lesions)


def plot_results(summary,dag_stability,canonical_stability,null_raw,null_tests,participation,models,out):
    figdir=out/"figures"; figdir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    axes[0,0].plot(summary.percentile,summary.largest_scc,"o-"); axes[0,0].set(title="Largest recurrent module",xlabel="percentile",ylabel="cells")
    axes[0,1].plot(summary.percentile,summary.n_scc,"o-"); axes[0,1].set(title="SCC fragmentation",xlabel="percentile",ylabel="SCCs")
    axes[1,0].plot(summary.percentile,summary.best_canonical_score,"o-"); axes[1,0].set(title="Best canonical path strength",xlabel="percentile",ylabel="geometric mean")
    axes[1,1].plot(summary.percentile,summary.canonical_fraction_top_dag,"o-"); axes[1,1].set(title="Canonical progression among top DAG paths",xlabel="percentile",ylabel="fraction")
    fig.tight_layout(); fig.savefig(figdir/"threshold_pathway_stability.png",dpi=180); plt.close(fig)

    metrics=["largest_scc","best_dag_path_score","canonical_fraction_top_dag","best_canonical_score"]
    fig,axes=plt.subplots(len(NULL_THRESHOLDS),len(metrics),figsize=(15,9))
    tests=null_tests.set_index(["percentile","metric"])
    for r,q in enumerate(NULL_THRESHOLDS):
      for c,metric in enumerate(metrics):
        ax=axes[r,c]; vals=null_raw.loc[null_raw.percentile==q,metric]
        ax.hist(vals,bins=15,color="#aab4bd",edgecolor="white")
        obs=tests.loc[(q,metric),"observed"]; p=tests.loc[(q,metric),"empirical_p_one_sided"]
        ax.axvline(obs,color="#b22222",lw=2); ax.set_title(f"Q{q} {metric}\np={p:.3g}",fontsize=9)
    fig.tight_layout(); fig.savefig(figdir/"pathway_null_tests.png",dpi=180); plt.close(fig)

    if not participation.empty:
        e=participation[participation.basis=="eigen"].pivot(index="path_rank",columns="mode",values="participation")
        s=participation[participation.basis=="schur"].pivot(index="path_rank",columns="mode",values="participation")
        fig,axes=plt.subplots(1,2,figsize=(14,6))
        for ax,x,title in [(axes[0],e,"Eigenmode participation"),(axes[1],s,"Schur-coordinate participation")]:
            im=ax.imshow(x,aspect="auto",cmap="magma"); ax.set(title=title,xlabel="mode",ylabel="stable-path rank"); fig.colorbar(im,ax=ax,shrink=.7)
        fig.tight_layout(); fig.savefig(figdir/"pathway_modal_participation.png",dpi=180); plt.close(fig)

    # Q50 condensation DAG: node size=SCC size, red edges=top enumerated paths.
    q=50; _,_,C,_=models[q]; pos={n:(n,-i) for i,n in enumerate(nx.topological_sort(C))}
    fig,ax=plt.subplots(figsize=(15,6)); sizes=[40+25*C.nodes[n]["size"] for n in C]
    nx.draw_networkx(C,pos,ax=ax,node_size=sizes,node_color=[C.nodes[n]["size"] for n in C],cmap="viridis",font_size=7,arrowsize=10,width=.6)
    ax.set_title("Q50 condensation DAG (node size/color = SCC size; labels = topological module index)"); ax.axis("off")
    fig.tight_layout(); fig.savefig(figdir/"q50_condensation_dag.png",dpi=180); plt.close(fig)


def run_analysis(matrix_path="mij_matrix.csv",netlist_path="mij_netlist.csv",output_dir="outputs/modified_tarjan_pathways",
                 percentiles=DEFAULT_THRESHOLDS,null_percentiles=NULL_THRESHOLDS,n_null=100,
                 matrix_orientation="auto_from_netlist",normalization="spectral_radius",
                 spectral_radius_target=1.0,netlist_orientation_tolerance=1e-10):
    out=Path(output_dir); (out/"tables").mkdir(parents=True,exist_ok=True)
    Mraw,labels,_,audit=load_inputs(
        Path(matrix_path),
        _optional_path(netlist_path),
        matrix_orientation,
        netlist_orientation_tolerance,
    )
    M,normalization_audit=apply_normalization(Mraw,normalization,spectral_radius_target)
    audit.update(normalization_audit)
    summary,dag_all,can_all,models=threshold_pathway_sweep(M,labels,percentiles)
    dag_stability,can_stability=stability_tables(dag_all,can_all,len(percentiles))
    null_raw,null_tests=pathway_null_tests(M,labels,null_percentiles,n_null)
    participation,lesions=modal_followup(M,labels,can_stability)
    tables={"threshold_summary":summary,"condensation_paths_all_thresholds":dag_all,
            "canonical_paths_all_thresholds":can_all,"condensation_path_stability":dag_stability,
            "canonical_path_stability":can_stability,"pathway_null_raw":null_raw,
            "pathway_null_tests":null_tests,"pathway_modal_participation":participation,
            "pathway_edge_lesion_modal_shifts":lesions}
    for name,df in tables.items(): df.to_csv(out/"tables"/f"{name}.csv",index=False)
    plot_results(summary,dag_stability,can_stability,null_raw,null_tests,participation,models,out)
    result={"audit":audit,
            "percentiles":list(percentiles),"null_percentiles":list(null_percentiles),"n_null":n_null,
            "output_dir":str(out)}
    with open(out/"summary.json","w") as f: json.dump(result,f,indent=2)
    return {"M":M,"labels":labels,"summary":summary,"dag_paths":dag_all,"canonical_paths":can_all,
            "dag_stability":dag_stability,"canonical_stability":can_stability,"null_raw":null_raw,
            "null_tests":null_tests,"participation":participation,"lesions":lesions,"models":models,"run_summary":result}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--matrix",default="mij_matrix.csv")
    p.add_argument("--netlist",default="mij_netlist.csv"); p.add_argument("--output",default="outputs/modified_tarjan_pathways")
    p.add_argument("--matrix-orientation",default="auto_from_netlist",choices=sorted(MATRIX_ORIENTATIONS),
                   help="orientation of the input matrix before analysis")
    p.add_argument("--normalization",default="spectral_radius",choices=sorted(NORMALIZATION_MODES),
                   help="use 'none' to analyze raw weights without spectral-radius scaling")
    p.add_argument("--spectral-radius",type=float,default=1.0)
    p.add_argument("--netlist-orientation-tolerance",type=float,default=1e-10)
    p.add_argument("--n-null",type=int,default=100); a=p.parse_args()
    print(json.dumps(run_analysis(a.matrix,a.netlist,a.output,n_null=a.n_null,
                                  matrix_orientation=a.matrix_orientation,
                                  normalization=a.normalization,
                                  spectral_radius_target=a.spectral_radius,
                                  netlist_orientation_tolerance=a.netlist_orientation_tolerance)["run_summary"],indent=2))


if __name__=="__main__": main()
