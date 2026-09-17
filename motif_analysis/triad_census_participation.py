import pandas as pd, numpy as np, itertools, collections
M = pd.read_csv('/mnt/user-data/uploads/AscoliLab/non-normal_matrices/data/mij_matrix.csv', index_col=0)
names = list(M.index); n = len(names)
A = M.values.astype(float)            # A[pre, post]
# E/I from row sign
ei = []
for k in range(n):
    v = A[k][A[k]!=0]
    ei.append('E' if (len(v) and v[0] > 0) else 'I')
ei = np.array(ei)
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
np.save('/home/claude/part.npy', {k:v for k,v in part.items()}, allow_pickle=True)
pd.Series(ei, index=names).to_csv('/home/claude/ei.csv')
