import pandas as pd, numpy as np, itertools, collections
M=pd.read_csv('/mnt/user-data/uploads/AscoliLab/non-normal_matrices/data/mij_matrix.csv',index_col=0)
names=list(M.index); n=len(names); A=M.values.astype(float)
ei=pd.read_csv('/home/claude/ei.csv',index_col=0).iloc[:,0].values
W=np.abs(A).copy(); np.fill_diagonal(W,0)
Bn=(W>0).astype(int)
tot=W.sum()

# weighted triad participation: for each connected triad, sum |m| on its 3-6 edges; credit each member
wpart=np.zeros(n); tri_w=[]
for a,b,c in itertools.combinations(range(n),3):
    s=W[a,b]+W[b,a]+W[a,c]+W[c,a]+W[b,c]+W[c,b]
    if s>0:
        wpart[a]+=s; wpart[b]+=s; wpart[c]+=s
df=pd.DataFrame({'cell':names,'ei':ei,'wpart':wpart}).set_index('cell')
df['rank']=df['wpart'].rank(ascending=False,method='min').astype(int)
print("=== Top 12 by WEIGHTED triad participation ===")
for c,r in df.sort_values('wpart',ascending=False).head(12).iterrows():
    print(f"  {r['rank']:3d}. {c:36s} {r['ei']}  {r['wpart']:14,.0f}")
print("  ---- DG ----")
for c in ['DG Granule','DG AIPRIM','DG Axo Axonic','DG HICAP','DG MOLAX','DG Mossy']:
    r=df.loc[c]; print(f"  {r['rank']:3d}. {c:36s} {r['ei']}  {r['wpart']:14,.0f}")

# 3-cycles (030C) membership
cyc=[]
for a,b,c in itertools.combinations(range(n),3):
    for p in [(a,b,c),(a,c,b)]:
        x,y,z=p
        if Bn[x,y] and Bn[y,z] and Bn[z,x] and not Bn[y,x] and not Bn[z,y] and not Bn[x,z]:
            cyc.append((names[x],names[y],names[z]))
print(f"\n=== 030C 3-cycles: {len(cyc)} total ===")
for t in cyc: print("   ",' -> '.join(t))

# open 3-chains X->Y->Z by E/I type, weighted
chain=collections.Counter(); chainw=collections.Counter()
for x in range(n):
    for y in range(n):
        if x==y or not Bn[x,y]: continue
        for z in range(n):
            if z in (x,y) or not Bn[y,z]: continue
            k=ei[x]+ei[y]+ei[z]
            chain[k]+=1; chainw[k]+= W[x,y]+W[y,z]
print("\n=== Directed 3-chains X->Y->Z by E/I signature ===")
for k in sorted(chain,key=lambda k:-chainw[k]):
    print(f"  {k}  count={chain[k]:6d}  weight={chainw[k]:16,.0f}  ({100*chainw[k]/sum(chainw.values()):5.1f}% of chain weight)")

# DG Granule as source of E->I->? chains
print("\n=== DG Granule (source) 3-chain weight by downstream signature ===")
g=names.index('DG Granule'); loc=collections.Counter()
for y in range(n):
    if not Bn[g,y]: continue
    for z in range(n):
        if z in (g,y) or not Bn[y,z]: continue
        loc[ei[g]+ei[y]+ei[z]] += W[g,y]+W[y,z]
for k,v in loc.most_common(): print(f"  {k}  {v:16,.0f}  ({100*v/sum(loc.values()):5.1f}%)")

# EII disinhibition: which second-order interneuron targets does DG Granule reach
print("\n=== DG Granule -> I -> I  top disinhibitory paths (by min edge weight) ===")
paths=[]
for y in range(n):
    if not Bn[g,y] or ei[y]!='I': continue
    for z in range(n):
        if z in (g,y) or not Bn[y,z] or ei[z]!='I': continue
        paths.append((min(W[g,y],W[y,z]), names[y], names[z], W[g,y], W[y,z]))
paths.sort(reverse=True)
for p in paths[:10]:
    print(f"   DG Granule -> {p[1]:26s} -> {p[2]:26s}  ({p[3]:12,.0f} , {p[4]:12,.0f})")
print(f"   ...{len(paths)} such paths total")
