# jacobian_analysis

Reproducible analysis suite that treats the signed hippocampal-entorhinal
connectivity matrix as the **Jacobian of a continuous-time rate model** rather
than as a discrete-time state-update operator.

It reuses the workflows from `schur_decomp`, `inhib_modulation`, and
`motif_analysis` — ablation tests, eigen/Schur decomposition, block fractures,
matched with-self/no-self comparisons — on the same matrix, in coordinates where
stability is a half-plane instead of a disk.

## Why a separate project

The existing projects analyze

```text
x[t + 1] = M x[t]                    stable iff rho(M) < 1
```

This one analyzes

```text
tau dx_i/dt = -x_i + sum_j W_ij x_j
J = (-I + W) / tau                   stable iff max Re(lambda) < 0
```

`W` is the same oriented, normalized `M[post, pre]`, so nothing about the data
or the ablation machinery changes. What changes is that a continuous-time system
separates three growth rates that the discrete spectral radius collapses into
one:

| Quantity | Definition | Answers |
|---|---|---|
| Spectral abscissa | `alpha(J) = max Re(lambda)` | How fast does it settle, eventually? |
| Numerical abscissa | `omega(J) = lambda_max((J + J^T)/2)` | How fast can it grow at `t = 0`? |
| Pseudospectral abscissa | `alpha_eps(J)` | How far can a perturbation of size `eps` push it? |

These three coincide for a normal matrix. The gap between them **is** transient
amplification, and it is invisible to eigenvalue analysis alone. Given that this
network is strongly non-normal, that gap is the point of the project.

The continuous formulation also makes several questions expressible that the
discrete one cannot state cleanly: the optimal-perturbation SVD of `exp(Jt)`,
the Kreiss lower bound on peak growth, the inhibitory gain at which the spectral
abscissa crosses zero, and the ISN test comparing `J_EE` against full `J`.

## Conventions

- **Orientation.** `mij_matrix.csv` on disk is `M[pre, post]` and is transposed
  on load to `W[post, pre]`, so `W[i, j]` is the influence of `j` on `i`. See
  `../docs/TRANSPOSE_WARNINGS.md`.
- **Orientation is checked, not remembered.** `jc.assert_post_pre` runs the
  Dale's-law sign oracle from `../docs/orientation_convention_proposal.md` at
  every load: E-to-I entries must be positive and I-to-E negative. Measured
  purity on this matrix is 100% / 0%, so a transposed matrix raises rather than
  silently reporting reversed causation.
- **Blocks are named source-to-target.** `EI` is E onto I, which occupies rows
  `I` and columns `E` of a `[post, pre]` matrix. `jc.ablate_block` spells the
  indexing out so the naming collision cannot propagate.
- **Normalization is applied to `W`, never to `J`.** Scaling `J` would rescale
  the leak along with the synaptic weights and silently change the membrane time
  constant. Scaling `W` changes only synaptic gain, which is the quantity being
  varied. Default `rho(W) = 0.95`, matching `schur_decomp` and
  `inhib_modulation` so results stay comparable — and since `lambda_J = lambda_W - 1`,
  any `rho(W) < 1` guarantees a stable Jacobian with margin at least `1 - rho`.
- **Ablations are not renormalized by default.** A cut changes the network's
  gain; renormalizing afterwards scales that change away. Both questions are
  legitimate but they are different questions, so notebook 04 runs the sweep
  both ways.
- **Self-connections are not the leak.** The diagonal of `W` is a genuine
  autapse term. The `-I` leak lives in `J` and is never removed by a `no_self`
  variant.

Canonical inputs: `matrices/mij_matrix.csv`, `matrices/mij_netlist.csv`
(85 cell types: 32 excitatory, 53 inhibitory). Outputs go to
`outputs/<notebook_name>/`.

## Notebook order

| Notebook | Status | Purpose |
|---|---|---|
| `01_jacobian_construction.ipynb` | **written** | Load, verify orientation, normalize, build `J`. Spectrum, stability, non-normality vs a normal surrogate, Schur coupling, transient envelope, pseudospectra and Kreiss bound, DC gain, E/I block baseline. |
| `02_jacobian_schur_modes.ipynb` | scaffold | Which cell types, E/I classes, and regions carry the amplifying modes. Port of `schur_decomp/03` and `/05`. |
| `03_jacobian_transient_amplification.ipynb` | scaffold | Optimal perturbations from the SVD of `exp(Jt)`; what gets amplified and whether it is biologically reachable. No discrete-time equivalent. |
| `04_jacobian_ablation_tests.ipynb` | scaffold | Node, edge, and graded-weakening ablations against block-preserving nulls. Port of `inhib_modulation/02`. |
| `05_jacobian_ei_block_fractures.ipynb` | scaffold | EE/EI/IE/II and self-connection removals, inhibitory gain sweep to the stability crossing, ISN test, inhibitory Schur complement. |
| `06_jacobian_compare_variants.ipynb` | scaffold | Matched with-self vs no-self, gain sweep, `tau` heterogeneity, leak sensitivity. Port of the `_compare` pattern in `schur_decomp`. |

Each notebook reloads the source matrix and calls shared helpers rather than
depending on the previous notebook's outputs, matching the pattern in
`inhib_modulation`. The narrative order is: establish the Jacobian and its
stability structure (01), find where the amplification lives (02, 03), fracture
it (04, 05), then test how much the modelling choices are carrying (06).

## Shared module

`jacobian_core.py` holds everything that must stay identical across notebooks:
loading and orientation assertion, normalization, `build_jacobian`,
`stability_summary`, `non_normality_summary`, `schur_decompose`,
`transient_envelope`, `resolvent_norm_grid`, `pseudospectral_abscissa`,
`kreiss_constant`, `steady_state_response`, the ablation primitives, and
`condition_summary` / `ablation_sweep`, which turn any ablated `W` into one row
of a comparison table.

`build_notebooks.py` regenerates the notebooks. It overwrites, so use
`--skip-existing` or `--only 02` once you have started editing:

```bash
python build_notebooks.py --only 01
python build_notebooks.py --skip-existing
```

## Headline results from notebook 01

At `rho(W) = 0.95`, `tau = 1`, leak `= 1`:

| Quantity | Value | Reading |
|---|---:|---|
| Spectral abscissa | `-0.0500` | Stable; slowest mode decays with time constant `20 tau` |
| Numerical abscissa | `29.58` | Growth rate at `t = 0` is strongly positive |
| Abscissa gap | `29.63` | The entire gap is non-normal transient structure |
| Peak amplification | `27.11x` at `t = 1.2 tau` | Worst-case transient gain |
| Kreiss lower bound | `18.47` (at `eps = 0.05`) | Amplification guaranteed without simulating |
| Henrici departure | `77.17` (99.3% of `‖J‖_F`) | Almost all of the matrix norm is non-normal |
| Eigenvector condition number | `9.9e4` | The modal basis is nowhere near orthogonal |
| Spectral radius of `J` | `1.31` | Not a stability statistic here; the fastest mode has time constant `0.76 tau` |

The network is asymptotically stable and amplifies a worst-case perturbation
27-fold before decaying. A purely spectral reading predicts monotone decay at
rate `0.05` and is wrong about the entire first few `tau` of the response. The
normal surrogate — a diagonal matrix carrying the same eigenvalues — has peak
amplification exactly `1.0`, so the amplification is attributable to
non-normal structure and not to the spectrum.

The block baseline suggests where that structure comes from: at this
normalization the `EI` block (E onto I) has Frobenius norm `77.02`, while `EE`
is `4.91`, `II` is `0.40`, and `IE` is `0.084`. Nearly all of the matrix norm
sits in one off-diagonal block with a very weak return path — the canonical
feed-forward geometry that produces non-normal amplification. Notebooks 04 and
05 are the place to test that reading rather than assume it.

The last spectral-radius row is worth flagging on its own, because it is easy
to misread. `rho(J) = 1.31 > 1` does **not** mean the network would be unstable
under the discrete criterion. The spectral radius measures distance from the
origin, and in continuous time the origin means nothing: the stability boundary
is the imaginary axis, and `|lambda| = sqrt(Re^2 + Im^2)` mixes decay rate with
oscillation frequency. `rho(J)` is a speed statistic, not a stability one.

Arithmetically, `lambda_J = lambda_W - 1`, so the spectrum of `J` is the
spectrum of `W` — a cloud of radius `0.95` — translated to sit at centre `-1`.
The eigenvalue attaining `rho(J)` is `lambda = -1.3103`, the *most negative* one.
Measured from the correct centre, `max |lambda_J + 1| = 0.95` exactly, as it must
be. Only the choice of origin makes it look large.

What the number does say is about timescales, and it follows from the eigenvalue
equation alone. If `J v = lambda v` then `W v = (lambda + 1) v`, so each
eigenvalue of `J` is `mu - 1` for an eigenvalue `mu` of `W`, and the per-mode
effective time constant is `tau / (1 - mu)` — the standard linear-recurrent-network
result (Dayan & Abbott, *Theoretical Neuroscience*, ch. 7). A mode is slower than
`tau` when `Re(mu) > 0` and faster when `Re(mu) < 0`. Here `Re(mu)` spans
`[-0.310, 0.950]`, giving effective time constants from `0.76 tau` to `20 tau`,
with 44 of 85 modes faster than `tau`. That is a 26-fold spread: the system is
stiff.

Two cautions on reading those per-mode time constants as relaxation times. `J` is
strongly non-normal — the eigenvector matrix has condition number `9.9e4` and the
fastest eigenvector has overlap `0.69` with another — so eigen-directions are not
separately excitable and a single mode's time constant is not something that can
be isolated and measured. And since `omega(J) = +29.6`, a worst-case state grows
rather than relaxing at any of these rates. The time constants describe the
spectrum, not the observable response.

The two criteria are not in conflict, and the connection is exact:
`(I + J) - W = 0` to machine precision, so the discrete operator `M` analyzed in
`schur_decomp` **is** the forward-Euler propagator of this Jacobian at
`dt = tau`. The discrete criterion `rho(M) < 1` is the Euler-step condition
`|1 + dt*lambda| < 1`, a disk of radius `1/dt` centred at `-1/dt`; the continuous
criterion is the half-plane. Both are satisfied here. What is meaningless is
comparing `rho(J)` itself to 1.

That equivalence comes with a condition worth stating in the dissertation: the
largest stable forward-Euler step for this Jacobian is `1.53 tau`, so `dt = tau`
is inside the stability region with roughly 50% margin. Had the network's fastest
mode been about twice as fast, the discrete analysis would have been an unstable
discretization of a stable system — an artifact of the time step rather than a
property of the circuit. The exact propagator gives `rho(exp(J*tau)) = 0.9512`,
against `0.95` for the Euler approximation, so at this step size the two agree
closely.

## Environment

Python with `numpy`, `scipy`, `pandas`, `matplotlib`, and `nbformat` (for
`build_notebooks.py`). Notebook 01 runs end to end in under 30 seconds; the
pseudospectra grid and the Kreiss sweep dominate that time and both take a
`resolution` argument if a faster pass is wanted.
