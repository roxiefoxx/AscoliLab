# archive/wij_structural

Archived Sept 2026. These are the raw `w_ij` matrices -- the **structural**
connectome, before the postsynaptic gain term `kappa_j` is applied.

All active analysis uses `m_ij = w_ij * kappa_j`, the **functional** operator.

## Why they were pulled out

Having both live in the tree makes it possible to load the wrong one and get a
plausible-looking result. There is no orientation oracle for this mistake the
way there is for a transpose: `w_ij` and `m_ij` have the same shape, the same
labels, the same sign pattern and the same E/I block structure. Only the
magnitudes differ, and they differ in a way that matters.

## The measurement that motivated the split

Block Frobenius norms, `[post, pre]` convention, E->I means E source onto I
receiver:

| Operator | E->I / I->E |
|---|---:|
| `m_ij = w_ij * kappa_j` | 912x |
| `w_ij` alone | 86x |

So roughly one order of magnitude (10.6x) of the feedforward asymmetry that the
non-normality argument rests on is contributed by postsynaptic gain, not by
wiring. `kappa_j` is indexed by the postsynaptic type, so the entire E->I block
is multiplied by interneuron kappa and the entire I->E block by principal-cell
kappa (median 0.0935 vs 0.0345 Hz/pA, ~9x at the tails).

That is a real result, not a confound -- but it belongs to a separate question:
how much of this circuit's feedforward geometry is wiring and how much is
excitability? That comparison is deferred, not abandoned.

## Caveat on these specific files

`wij_matrix.csv` was not verified to be exactly the `w_ij` column of
`mij_netlist.csv`. Before quoting the 86x figure anywhere, reconstruct `w_ij`
from the netlist and confirm. The netlist is the authoritative source.

## Component factors

`ml_cij`, `ml_gij`, `ml_qi`, `ml_taud` and `ml_vij` netlists were left in
`non-normal_matrices/data/`. They are the individual terms of
`m_ij = q_i * c_ij * g_ij * (V_ref - V_rev) * tau_ij * kappa_j` and are needed to
answer *which* factor drives the very small I->E block (0.8% of block-norm total
even before kappa). Move them here too only if that question is closed.
