#!/usr/bin/env python3
"""Regenerate the Schur-ready 3x3 motif block exports that notebooks 03 and 04 read.

Why this file exists
--------------------
`03_motif_schur_decomposition_analysis.ipynb` and
`04_motif_signed_overlap_operator_analysis.ipynb` read four artifacts:

    outputs/motif_pathway_analysis/schur_ready_3x3_subnetworks_{with,without}_self.{csv,npz}

Nothing in the repository wrote them.  The producer, `motif_pathway_analysis.ipynb`, is not
committed anywhere, and a grep across the tree including archives finds those four notebooks
as the only mentions, all reads (`docs/RUN_ORDER.md`, "Blocked — cannot run").  This module
is a replacement producer, written 2026-09-17 as part of the corrections pass.

WHAT IS RECONSTRUCTED AND WHAT IS NOT
-------------------------------------
The file format is recovered exactly -- it is pinned by what the consuming scripts read, and
every column below is required by `motif_schur_decomposition_analysis_script.py` or
`motif_signed_overlap_analysis_script.py`.

The *selection* is not recovered.  Which triads the original export contained is unknowable
from what survives, and it matters: the connectome has 68,686 connected triads, and the
motif-by-motif overlap operator in notebook 04 is dense enough that all of them would need
tens of gigabytes.  So the selection here is a NEW, EXPLICIT DECISION, exposed as parameters
rather than buried:

    --classes   which Davis-Holland-Leinhardt triad classes to export.
                Default: the closed classes, those with a cycle or a reciprocal pair --
                030T 030C 201 120D 120U 120C 210 300 -- 8,431 triads.  These are the ones a
                local Schur decomposition has anything to say about; an open 012 or 021D
                triad is nilpotent and its spectrum is zero.
    --max-motifs  cap for the 04 operator, applied by descending total |m_ij| WITHIN each
                class so no class is dropped by the cap.  Default 2000.
                `membership @ membership.T` has roughly 85 * (N/85)^2 nonzeros; at N = 2000
                that is about 5e6, at N = 12000 it is 1.7e8.

Anyone rerunning 03/04 should set these deliberately and record what they used.  Results are
comparable across runs only at the same selection.

ORIENTATION
-----------
Blocks are written SOURCE ROWS / TARGET COLUMNS -- `block[i, j]` is the weight from node i to
node j -- because that is what `TRANSPOSE_SOURCE_TARGET_BLOCKS = True` in both notebooks
expects, and the notebooks transpose once on load to reach the `[post, pre]` convention the
rest of the pipeline uses.  The flat `w_11..w_33` metadata columns carry the same untransposed
block, so audit item A3's check

    w = meta[[f"w_{i}{j}" ...]].to_numpy().reshape(-1, 3, 3)
    assert np.allclose(w, blocks_raw)

passes by construction, and is asserted here before writing.

    python3 motif_pathway_export.py
    python3 motif_pathway_export.py --classes 300 210 120D --max-motifs 500
"""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# DHL class order, and the Rees et al. (2016) superpattern letters A-M, which map
# letter-for-letter onto the 13 connected classes (docs/KICKOFF_rees_es.md).
TRIAD_CLASSES = ['003', '012', '102', '021D', '021U', '021C', '111D', '111U',
                 '030T', '030C', '201', '120D', '120U', '120C', '210', '300']
SUPERPATTERN_LETTER = dict(zip(
    ['012', '102', '021D', '021U', '021C', '111D', '111U',
     '030T', '030C', '201', '120D', '120U', '120C', '210', '300'],
    list("ABCDEFGHIJKLMNO")))
CLOSED_CLASSES = ('030T', '030C', '201', '120D', '120U', '120C', '210', '300')
PAIR_ORDER = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
MATRIX_COLUMNS = [f"w_{i}{j}" for i in range(1, 4) for j in range(1, 4)]


def _classify(mat):
    mutual = sum(1 for i in range(3) for j in range(i + 1, 3) if mat[i][j] and mat[j][i])
    asym = sum(1 for i in range(3) for j in range(3) if i != j and mat[i][j] and not mat[j][i])
    key = (mutual, asym, 3 - mutual - asym)
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


def _tables():
    """code -> (class name, canonical code under relabeling)."""
    cls = np.empty(64, dtype=np.int8)
    canon = np.empty(64, dtype=np.int8)
    for bits in range(64):
        mat = [[0] * 3 for _ in range(3)]
        for i, (x, y) in enumerate(PAIR_ORDER):
            mat[x][y] = (bits >> i) & 1
        cls[bits] = TRIAD_CLASSES.index(_classify(mat))
        best = 64
        for perm in itertools.permutations(range(3)):
            code = 0
            for i, (x, y) in enumerate(PAIR_ORDER):
                if mat[perm[x]][perm[y]]:
                    code |= 1 << i
            best = min(best, code)
        canon[bits] = best
    return cls, canon


CODE_CLASS, CODE_CANONICAL = _tables()


def load(matrix_csv=None, netlist_csv=None):
    matrix_csv = Path(matrix_csv or HERE / "mij_matrix.csv")
    netlist_csv = Path(netlist_csv or HERE / "mij_netlist.csv")
    M = pd.read_csv(matrix_csv, index_col=0).astype(float)
    names = list(M.index)
    if list(M.columns) != names:
        raise ValueError("Row and column labels must match in the same order.")
    A = M.to_numpy()                                    # A[pre, post]
    net = pd.read_csv(netlist_csv)
    index = {label: i for i, label in enumerate(names)}
    bad = sum(abs(A[index[r.pre_neuron], index[r.post_neuron]] - r.m_ij)
              > 1e-6 * max(1.0, abs(r.m_ij)) for r in net.itertuples(index=False))
    if bad:
        raise ValueError(f"{matrix_csv} is not [pre, post]: {bad} netlist rows disagree.")
    return names, A


def export(names, A, classes, max_motifs, outdir):
    n = len(names)
    A_self = A.copy()                                   # self-weights retained
    A_no = A.copy(); np.fill_diagonal(A_no, 0.0)        # self-weights removed
    B = (np.abs(A_no) > 0)                              # adjacency ignores self-loops

    wanted = {TRIAD_CLASSES.index(c) for c in classes}
    rows, keep = [], []
    for a, b, c in itertools.combinations(range(n), 3):
        bits = (int(B[a, b]) | (int(B[b, a]) << 1) | (int(B[a, c]) << 2)
                | (int(B[c, a]) << 3) | (int(B[b, c]) << 4) | (int(B[c, b]) << 5))
        ci = int(CODE_CLASS[bits])
        if ci not in wanted:
            continue
        idx = (a, b, c)
        total = float(np.abs(A_no[np.ix_(idx, idx)]).sum())
        keep.append(idx)
        rows.append({"class_index": ci, "triad_class": TRIAD_CLASSES[ci],
                     "superpattern": TRIAD_CLASSES[ci],
                     "superpattern_letter": SUPERPATTERN_LETTER[TRIAD_CLASSES[ci]],
                     "observed_motif_code": bits,
                     "canonical_motif_id": int(CODE_CANONICAL[bits]),
                     "node_1": names[a], "node_2": names[b], "node_3": names[c],
                     "nodes": " | ".join(names[i] for i in idx),
                     "total_abs_weight": total})
    meta = pd.DataFrame(rows)
    print(f"{len(meta)} triads in the requested classes")
    print(meta.triad_class.value_counts().reindex(classes).to_string())

    if max_motifs and len(meta) > max_motifs:
        # proportional per class, so the cap never drops a class entirely
        share = (meta.triad_class.value_counts() / len(meta) * max_motifs).round().astype(int)
        share = share.clip(lower=1)
        pick = (meta.sort_values("total_abs_weight", ascending=False)
                    .groupby("triad_class", group_keys=False)
                    .apply(lambda g: g.head(int(share[g.name])), include_groups=True))
        sel = sorted(pick.index)
        print(f"capped to {len(sel)} motifs (--max-motifs {max_motifs}), "
              f"ranked by total |m_ij| within each class")
    else:
        sel = list(range(len(meta)))

    meta = meta.loc[sel].reset_index(drop=True)
    idxs = [keep[i] for i in sel]
    meta.insert(0, "motif_index", np.arange(len(meta)))

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for tag, source in (("without_self", A_no), ("with_self", A_self)):
        blocks = np.stack([source[np.ix_(i, i)] for i in idxs]).astype(float)
        flat = pd.DataFrame(blocks.reshape(len(blocks), 9), columns=MATRIX_COLUMNS)
        frame = pd.concat([meta, flat], axis=1)
        # audit item A3: the flat columns must reproduce the npz blocks untransposed
        check = frame[MATRIX_COLUMNS].to_numpy().reshape(-1, 3, 3)
        assert np.allclose(check, blocks), "w_ij columns disagree with analysis_blocks"
        csv_path = outdir / f"schur_ready_3x3_subnetworks_{tag}.csv"
        npz_path = outdir / f"schur_ready_3x3_subnetworks_{tag}.npz"
        frame.to_csv(csv_path, index=False)
        np.savez_compressed(npz_path, analysis_blocks=blocks)
        written += [csv_path, npz_path]
        print(f"  {tag}: blocks {blocks.shape}  -> {csv_path.name}, {npz_path.name}")

    (outdir / "export_provenance.json").write_text(json.dumps({
        "produced_by": "tarjan_reorder/motif_pathway_export.py",
        "date": "2026-09-17",
        "reason": "original producer motif_pathway_analysis.ipynb is not in the repository",
        "classes": list(classes), "max_motifs": max_motifs,
        "n_motifs": int(len(meta)),
        "block_orientation": "source rows / target columns (notebooks transpose on load)",
        "selection_is_a_new_decision": True,
    }, indent=2))
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=None)
    ap.add_argument("--netlist", default=None)
    ap.add_argument("--classes", nargs="+", default=list(CLOSED_CLASSES))
    ap.add_argument("--max-motifs", type=int, default=2000)
    ap.add_argument("--outdir", default=str(HERE / "outputs" / "motif_pathway_analysis"))
    a = ap.parse_args()
    for c in a.classes:
        if c not in TRIAD_CLASSES:
            raise SystemExit(f"unknown triad class {c!r}; choose from {TRIAD_CLASSES}")
    names, A = load(a.matrix, a.netlist)
    export(names, A, a.classes, a.max_motifs, a.outdir)


if __name__ == "__main__":
    main()
