"""Directed triad census (Davis-Holland-Leinhardt 16) with per-node participation.

Orientation (audit item A3, verified 2026-09-17).  ``mij_matrix.csv`` is stored
``A[pre, post]``, so ``Bn[x, y]`` below is the edge ``x -> y`` and the triad
classification reads edge directions correctly as stored.  This is the one place
in the pipeline that does NOT transpose to ``M[post, pre]``, and it is right not
to: the dynamics convention exists so that ``(A x)_i`` is drive into ``i``,
which is a statement about matrix-vector products, not about graph edges.  The
D/U triad labels are not transpose-invariant -- reading the matrix the other way
would swap 021D/021U, 111D/111U and 120D/120U -- so the assertion below pins it
against the netlist rather than leaving it to a comment.

Usage:  python3 triad_census_participation.py [matrix_csv] [netlist_csv]
"""
import sys, pathlib
import pandas as pd, numpy as np, itertools, collections

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
MATRIX_CSV = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "non-normal_matrices" / "data" / "mij_matrix.csv"
NETLIST_CSV = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "tarjan_reorder" / "mij_netlist.csv"
OUTDIR = HERE / "outputs" / "triad_census"
OUTDIR.mkdir(parents=True, exist_ok=True)

M = pd.read_csv(MATRIX_CSV, index_col=0)
names = list(M.index); n = len(names)
if list(M.columns) != names:
    raise ValueError("Row and column labels must match in the same order.")
A = M.values.astype(float)            # A[pre, post]  -- asserted below

netlist = pd.read_csv(NETLIST_CSV)
index = {label: i for i, label in enumerate(names)}

# --- orientation assertion: A[pre, post] must reproduce the netlist ----------
def _residual(transposed):
    B = A.T if transposed else A
    bad = 0
    for row in netlist.itertuples(index=False):
        i, j = index[row.pre_neuron], index[row.post_neuron]
        if abs(B[i, j] - row.m_ij) > 1e-6 * max(1.0, abs(row.m_ij)):
            bad += 1
    return bad

as_stored, transposed = _residual(False), _residual(True)
if as_stored != 0:
    raise ValueError(
        f"{MATRIX_CSV} is not [pre, post]: {as_stored} of {len(netlist)} netlist "
        f"rows disagree as stored ({transposed} disagree transposed). The triad "
        "D/U labels depend on this; fix the input rather than the labels."
    )
print(f"orientation check: A[pre, post] reproduces all {len(netlist)} netlist rows "
      f"(transposed reading fails {transposed})")

# --- E/I from the netlist, with the row-sign rule as a cross-check -----------
ei_map = {}
for neuron_col, ei_col in (("pre_neuron", "pre_ei"), ("post_neuron", "post_ei")):
    for a_, b_ in netlist[[neuron_col, ei_col]].dropna().drop_duplicates().itertuples(index=False):
        value = str(b_).strip().lower()
        if ei_map.setdefault(str(a_), value) != value:
            raise ValueError(f"Conflicting E/I metadata for {a_}")
missing = [label for label in names if label not in ei_map]
if missing:
    raise ValueError(f"Missing E/I metadata for {missing[:5]}")
ei = np.array([ei_map[label].upper() for label in names])

row_sign = np.array(["E" if (len(v) and v[0] > 0) else "I"
                     for v in (A[k][A[k] != 0] for k in range(n))])
disagree = [names[k] for k in range(n) if row_sign[k] != ei[k]]
print(f"E/I: netlist labels used; row-sign rule agrees on {n - len(disagree)}/{n}"
      + (f", disagrees on {disagree[:5]}" if disagree else ""))
B = (A != 0).astype(np.int8)          # binary adjacency, B[pre,post]
np.fill_diagonal(B, B.diagonal())     # keep self-loops as-is for reporting
Bn = B.copy(); np.fill_diagonal(Bn, 0)  # triads ignore self-loops

print("nodes", n, "E", (ei=='E').sum(), "I", (ei=='I').sum())
print("edges (no self)", Bn.sum(), "self-loops", int(B.diagonal().sum()))

# ---- DHL 13 triad census with per-node participation ----
LABELS = ['003','012','102','021D','021U','021C','111D','111U','030T','030C','120D','120U','120C','210','300']
# canonical classification via edge pattern
def tri_class(a,b,c):
    # a,b,c indices; build 6-bit
    e = {}
    pairs = [(a,b),(b,a),(a,c),(c,a),(b,c),(c,b)]
    m = sum(int(Bn[x,y])<<i for i,(x,y) in enumerate(pairs))
    return m
# Precompute mapping from 6-bit code -> triad type by brute force isomorphism
from itertools import permutations
def code_from_mat(mat):
    idx=[(0,1),(1,0),(0,2),(2,0),(1,2),(2,1)]
    return sum(int(mat[x][y])<<i for i,(x,y) in enumerate(idx))
def classify_mat(mat):
    md = sum(1 for i in range(3) for j in range(3) if i<j and mat[i][j] and mat[j][i])
    asym = sum(1 for i in range(3) for j in range(3) if i!=j and mat[i][j] and not mat[j][i])
    null = 3 - md - asym
    key=(md,asym,null)
    base={(0,0,3):'003',(0,1,2):'012',(1,0,2):'102',(0,2,1):None,(1,1,1):None,
          (2,0,1):'201',(0,3,0):None,(1,2,0):None,(2,1,0):'210',(3,0,0):'300'}
    if key in base and base[key]: return base[key]
    if key==(0,2,1):
        outd=[sum(mat[i]) for i in range(3)]; ind=[sum(mat[j][i] for j in range(3)) for i in range(3)]
        if 2 in outd: return '021D'
        if 2 in ind: return '021U'
        return '021C'
    if key==(1,1,1):
        # find the mutual pair
        for i,j in [(0,1),(0,2),(1,2)]:
            if mat[i][j] and mat[j][i]:
                k=3-i-j
                if mat[k][i] or mat[k][j]: return '111D'   # asym points INTO the dyad
                return '111U'
    if key==(0,3,0):
        outd=[sum(mat[i]) for i in range(3)]
        if 2 in outd: return '030T'
        return '030C'
    if key==(1,2,0):
        for i,j in [(0,1),(0,2),(1,2)]:
            if mat[i][j] and mat[j][i]:
                k=3-i-j
                o=mat[k][i]+mat[k][j]; inn=mat[i][k]+mat[j][k]
                if o==2: return '120D'
                if inn==2: return '120U'
                return '120C'
    return '?'
CODE2CLASS={}
for bits in range(64):
    idx=[(0,1),(1,0),(0,2),(2,0),(1,2),(2,1)]
    mat=[[0]*3 for _ in range(3)]
    for i,(x,y) in enumerate(idx):
        mat[x][y]=(bits>>i)&1
    CODE2CLASS[bits]=classify_mat(mat)

census=collections.Counter()
part=collections.defaultdict(lambda: np.zeros(n, dtype=np.int64))
for a,b,c in itertools.combinations(range(n),3):
    cl = CODE2CLASS[tri_class(a,b,c)]
    census[cl]+=1
    if cl!='003':
        v=part[cl]; v[a]+=1; v[b]+=1; v[c]+=1
tot=sum(census.values())
print("\n=== Triad census (85 nodes, self-loops excluded) ===")
for k in ['003','012','102','021D','021U','021C','111D','111U','030T','030C','201','120D','120U','120C','210','300']:
    if census.get(k): print(f"  {k:5s} {census[k]:8d}  {100*census[k]/tot:6.2f}%")
pd.DataFrame(census.most_common(), columns=["triad_class", "count"]).to_csv(
    OUTDIR / "triad_census.csv", index=False)
participation = pd.DataFrame({k: v for k, v in sorted(part.items())}, index=names)
participation.index.name = "cell_type"
participation.insert(0, "ei", ei)
participation.to_csv(OUTDIR / "node_triad_participation.csv")
np.save(OUTDIR / "part.npy", {k: v for k, v in part.items()}, allow_pickle=True)
pd.Series(ei, index=names, name="ei").to_csv(OUTDIR / "ei.csv")
print(f"\nwrote {OUTDIR}/triad_census.csv, node_triad_participation.csv, ei.csv, part.npy")
