import numpy as np, pandas as pd
from scipy.linalg import expm

net = pd.read_csv('mij_netlist.csv')
lab = sorted(set(net.pre_neuron) | set(net.post_neuron))
idx = {l:i for i,l in enumerate(lab)}; n=len(lab)
ei = {}
for cn,ce in [('pre_neuron','pre_ei'),('post_neuron','post_ei')]:
    for a,b in net[[cn,ce]].drop_duplicates().itertuples(index=False): ei[a]=b.lower()
cls = np.array([ei[l] for l in lab]); E=np.flatnonzero(cls=='e'); I=np.flatnonzero(cls=='i')

# M[post,pre] straight from the netlist -- authoritative
M = np.zeros((n,n))
for r in net.itertuples(index=False):
    M[idx[r.post_neuron], idx[r.pre_neuron]] += r.m_ij
kap = np.zeros(n)
for r in net[['post_neuron','kappa_j']].drop_duplicates().itertuples(index=False):
    kap[idx[r.post_neuron]] = r.kappa_j
q = pd.read_csv('ml_qi_vector.csv').set_index('neuron_type')['q_i'].reindex(lab).to_numpy()
print(f'n={n}  E={len(E)} I={len(I)}  kappa zeros={np.sum(kap==0)}  q nan={np.isnan(q).sum()}')
print(f'q_i  median E {np.median(q[E]):.0f}  median I {np.median(q[I]):.0f}  ratio {np.median(q[E])/np.median(q[I]):.2f}x')

def norm_rho(A, target=0.95):
    r = max(abs(np.linalg.eigvals(A)));  return A*(target/r)

def henrici(A):
    fro2 = np.linalg.norm(A,'fro')**2
    ev = np.linalg.eigvals(A)
    return np.sqrt(max(fro2 - np.sum(np.abs(ev)**2), 0.0))

def metrics(W):
    W = norm_rho(W)
    J = -np.eye(len(W)) + W
    ts = np.linspace(0, 8, 161)[1:]
    peak = max(np.linalg.norm(expm(J*t), 2) for t in ts)
    S = (J + J.T)/2
    return dict(alpha=max(np.linalg.eigvals(J).real),
                omega=max(np.linalg.eigvalsh(S)),
                henrici_frac=henrici(J)/np.linalg.norm(J,'fro'),
                peak=peak)

# ---- weighted feedback-arc ordering: GR heuristic + best-insertion sweeps ----
def fwd_fraction(A, order):
    pos = np.empty(len(order), int); pos[order] = np.arange(len(order))
    W = np.abs(A); np.fill_diagonal(W, 0)
    r, c = np.nonzero(W)               # r=post(target), c=pre(source); edge c->r
    fwd = W[r, c][pos[c] < pos[r]].sum()
    return fwd / W[r, c].sum()

def order_search(A, sweeps=6, seed=0):
    W = np.abs(A).copy(); np.fill_diagonal(W, 0)
    m = len(W); rng = np.random.default_rng(seed)
    out = W.sum(0); inn = W.sum(1)          # col sums = outgoing, row sums = incoming
    order = list(np.argsort(-(out - inn)))  # greedy net-outflow start
    best = fwd_fraction(A, np.array(order))
    for _ in range(sweeps):
        improved = False
        for v in rng.permutation(m):
            cur = list(order); cur.remove(v)
            cand, cb = None, best
            for p in range(m):
                trial = cur[:p] + [v] + cur[p:]
                f = fwd_fraction(A, np.array(trial))
                if f > cb + 1e-12: cb, cand = f, trial
            if cand: order, best, improved = cand, cb, True
        if not improved: break
    return best

def blocks(A):
    o={}
    for k,(r,c) in {'E->E':(E,E),'E->I':(I,E),'I->E':(E,I),'I->I':(I,I)}.items():
        o[k]=np.linalg.norm(A[np.ix_(r,c)])
    return o

kr = kap.copy(); kr[kr==0]=1.0
surrogates = {
 'M  (real operator)'              : M.copy(),
 'binary  sign(M) only'            : np.sign(M),
 'M / kappa_j   (no postsyn gain)' : M / kr[:,None],
 'M / q_i       (no census)'       : M / q[None,:],
 'M / (kappa_j q_i)  (neither)'    : M / kr[:,None] / q[None,:],
}
rows=[]
for name,A in surrogates.items():
    mt = metrics(A); b = blocks(norm_rho(A))
    ff = order_search(A)
    rows.append(dict(surrogate=name, henrici_pct=100*mt['henrici_frac'], omega=mt['omega'],
                     peak=mt['peak'], EI_over_IE=b['E->I']/b['I->E'], fwd_weight_frac=ff))
df=pd.DataFrame(rows)
pd.set_option('display.width',200,'display.max_colwidth',34)
print()
print(df.to_string(index=False, float_format=lambda v: f'{v:,.3f}'))
import numpy as np, pandas as pd
from scipy.linalg import expm
net = pd.read_csv('mij_netlist.csv')
lab = sorted(set(net.pre_neuron)|set(net.post_neuron)); idx={l:i for i,l in enumerate(lab)}; n=len(lab)
M=np.zeros((n,n))
for r in net.itertuples(index=False): M[idx[r.post_neuron], idx[r.pre_neuron]] += r.m_ij
q = pd.read_csv('ml_qi_vector.csv').set_index('neuron_type')['q_i'].reindex(lab).to_numpy()

def norm_rho(A,t=0.95): return A*(t/max(abs(np.linalg.eigvals(A))))
def henrici(A):
    ev=np.linalg.eigvals(A); return np.sqrt(max(np.linalg.norm(A,'fro')**2-np.sum(np.abs(ev)**2),0))
def metrics(W,label):
    W=norm_rho(W); J=-np.eye(n)+W
    peak=max(np.linalg.norm(expm(J*t),2) for t in np.linspace(0,8,161)[1:])
    print(f'{label:52s} alpha={max(np.linalg.eigvals(J).real):+.4f}  '
          f'omega={max(np.linalg.eigvalsh((J+J.T)/2)):7.3f}  '
          f'henrici={100*henrici(J)/np.linalg.norm(J,"fro"):5.1f}%  peak={peak:7.3f}')

D=np.diag(q); Di=np.diag(1/q)
print('SAME OPERATOR, DIFFERENT ACTIVITY VARIABLE  (similarity transform -- spectrum identical)')
metrics(M,                'extensive: A = population total output (poster)')
metrics(Di @ M @ D,       'intensive: r = per-cell firing rate')
metrics(np.diag(1/np.sqrt(q)) @ M @ np.diag(np.sqrt(q)), 'symmetric split: A / sqrt(q)')
ev1=np.sort_complex(np.linalg.eigvals(norm_rho(M))); ev2=np.sort_complex(np.linalg.eigvals(norm_rho(Di@M@D)))
print(f'\nspectra identical? max|diff| = {np.max(np.abs(ev1-ev2)):.2e}   <-- confirms similarity, not a model change')
