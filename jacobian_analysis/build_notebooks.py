"""Generate the numbered notebooks for the jacobian_analysis project.

Run from the project root:  python build_notebooks.py

Notebook 01 is written in full. Notebooks 02-06 are scaffolds: every section
heading, the method note explaining what the section is for, and the helper
calls sketched out, with the analysis left to be filled in. Regenerating
overwrites an unmodified scaffold, so pass --only 01 (or --skip-existing) once
you have started editing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


def write(name: str, cells: list[nbf.NotebookNode]) -> Path:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    path = ROOT / name
    nbf.write(nb, path)
    return path


PREAMBLE = """
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import jacobian_core as jc

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (7.5, 4.5), "axes.grid": True, "grid.alpha": 0.25})

MATRIX = "matrices/mij_matrix.csv"
NETLIST = "matrices/mij_netlist.csv"

# Synaptic gain. rho(W) < 1 guarantees a stable Jacobian, since eigenvalues of
# J = -I + W are those of W shifted left by one. 0.95 matches the normalization
# used in schur_decomp and inhib_modulation so results stay comparable.
GAIN = 0.95
TAU = 1.0        # membrane time constant; time is measured in units of tau
LEAK = 1.0       # coefficient on -I; leave at 1 unless testing leak sensitivity
"""


# ==========================================================================
# 01 - Jacobian construction and stability foundation
# ==========================================================================


def notebook_01() -> list[nbf.NotebookNode]:
    cells = []
    cells.append(md("""
# 01 - Jacobian construction and stability foundation

Foundation notebook for the `jacobian_analysis` project. Everything downstream
assumes the matrix conventions, normalization, and stability vocabulary
established here.

## What changes relative to the discrete-time projects

`schur_decomp`, `inhib_modulation`, and `motif_analysis` all treat the signed
connectivity matrix as a discrete state-update operator:

$$x[t+1] = M\\,x[t], \\qquad \\text{stable} \\iff \\rho(M) < 1$$

This project treats the same matrix as the Jacobian of a continuous-time linear
rate model linearized about its fixed point:

$$\\tau \\frac{dx_i}{dt} = -x_i + \\sum_j W_{ij} x_j,
\\qquad J = \\frac{-I + W}{\\tau},
\\qquad \\text{stable} \\iff \\max_k \\operatorname{Re}\\lambda_k < 0$$

`W` is the same oriented, normalized `M[post, pre]` used everywhere else, so the
ablation, Schur, and motif workflows port over unchanged. What the change buys
is that a continuous-time system separates three growth rates that the discrete
spectral radius collapses into one:

| Quantity | Symbol | Answers |
|---|---|---|
| Spectral abscissa | $\\alpha(J) = \\max \\operatorname{Re}\\lambda$ | How fast does the network settle, eventually? |
| Numerical abscissa | $\\omega(J) = \\lambda_{\\max}\\!\\big(\\tfrac{J + J^{T}}{2}\\big)$ | How fast can it grow right now, at $t = 0$? |
| Pseudospectral abscissa | $\\alpha_\\varepsilon(J)$ | How far can a perturbation of size $\\varepsilon$ push it? |

For a normal matrix all three agree. The gap between them is transient
amplification, and it is the quantity a purely spectral analysis cannot see.

## Sections

1. Load and verify orientation
2. Normalization and gain
3. Build the Jacobian
4. Eigenvalue spectrum
5. Stability summary
6. Non-normality and the normal surrogate
7. Schur decomposition and modal coupling
8. Transient amplification
9. Pseudospectra and the Kreiss bound
10. Steady-state gain
11. E/I block structure
12. Save outputs and verify
"""))

    cells.append(code(PREAMBLE + """
OUT = jc.output_dir("01_jacobian_construction")
print("outputs ->", OUT)
"""))

    cells.append(md("""
## 1. Load and verify orientation

Every `mij_matrix.csv` in this repository is stored as `M[pre, post]` and must
be transposed before use (`docs/TRANSPOSE_WARNINGS.md`). `load_jacobian_data`
transposes on load and then verifies the result rather than trusting it: E/I
identity here comes from the sign of outgoing weights, so the sign pattern of
the off-diagonal blocks is an orientation oracle. E to I entries must be
positive and I to E entries negative; the two blocks swap completely under
transposition, with 100/0 separation in this dataset. A wrong orientation raises
instead of silently reporting reversed causation.
"""))

    cells.append(code("""
data = jc.load_jacobian_data(MATRIX, NETLIST)

print(f"cell types: {data.n}")
print(f"excitatory: {len(data.excitatory)}   inhibitory: {len(data.inhibitory)}")
print()
print("orientation check (W[post, pre]):")
for key, value in data.orientation_check.items():
    print(f"  {key:22s} {value:.4f}")

masks = jc.ei_masks(data.ei, data.labels)
labels = data.labels
"""))

    cells.append(md("""
## 2. Normalization and gain

Normalization is applied to `W`, never to `J`. Scaling `J` would rescale the
leak along with the synaptic weights, silently changing the membrane time
constant; scaling `W` changes only the synaptic gain, which is the quantity
being varied. The raw matrix has a spectral radius near $2.6 \\times 10^{4}$, so
without normalization the Jacobian is wildly unstable and only structural
quantities would be meaningful.

With $\\rho(W) = g < 1$ the Jacobian is guaranteed stable, because
$\\lambda_J = \\lambda_W - 1$ places every eigenvalue inside a disk of radius $g$
centered at $-1$. The stability margin is then at least $1 - g$.
"""))

    cells.append(code("""
norm_rows = []
for method, target in [("spectral_radius", GAIN), ("spectral_abscissa", GAIN), ("frobenius", 10.0), ("none", 1.0)]:
    _, info = jc.normalize_weights(data.W_raw, method=method, target=target)
    norm_rows.append(info)
normalization_options = pd.DataFrame(norm_rows).set_index("method")
display(normalization_options)

W, norm_info = jc.normalize_weights(data.W_raw, method="spectral_radius", target=GAIN)
print()
print(f"chosen: rho(W) = {norm_info['spectral_radius']:.4f}  (scale factor {norm_info['scale']:.3e})")
print(f"guaranteed stability margin >= {1.0 - GAIN:.3f}")
"""))

    cells.append(md("""
## 3. Build the Jacobian

$J = \\mathrm{diag}(1/\\tau)\\,(-\\,\\text{leak}\\cdot I + W)$. With scalar $\\tau$
this is a pure shift of the spectrum. A non-uniform $\\tau$ is a left diagonal
scaling that adds non-normality on its own, so it stays at the scalar default
here and is varied only where a notebook explicitly asks that question.
"""))

    cells.append(code("""
J = jc.build_jacobian(W, tau=TAU, leak=LEAK)
J_frame = data.frame(J)

print(f"J shape {J.shape}")
print(f"diagonal range: [{np.min(np.diag(J)):.4f}, {np.max(np.diag(J)):.4f}]")
print(f"off-diagonal nonzeros: {np.count_nonzero(J - np.diag(np.diag(J)))}")
J_frame.iloc[:5, :5]
"""))

    cells.append(md("""
## 4. Eigenvalue spectrum

Stability is now a half-plane rather than a disk. Note that the discrete
criterion and the continuous one can disagree on the same matrix: $\\rho(J) > 1$
is unremarkable for a stable Jacobian, because what matters is where the
eigenvalues sit relative to the imaginary axis, not the unit circle.
"""))

    cells.append(code("""
eigvals = np.linalg.eigvals(J)
order = np.argsort(-eigvals.real)
spectrum = pd.DataFrame({
    "rank_by_real": np.arange(1, len(eigvals) + 1),
    "real": eigvals[order].real,
    "imag": eigvals[order].imag,
    "abs": np.abs(eigvals[order]),
    "decay_time_constant": np.where(eigvals[order].real < 0, -1.0 / eigvals[order].real, np.inf),
})
display(spectrum.head(12).round(4))

fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.scatter(eigvals.real, eigvals.imag, s=22, alpha=0.75, edgecolor="none")
ax.axvline(0.0, color="crimson", lw=1.2, label="stability boundary (Re = 0)")
theta = np.linspace(0, 2 * np.pi, 400)
ax.plot(-1 + GAIN * np.cos(theta), GAIN * np.sin(theta), ls="--", lw=1.0, color="grey",
        label=f"disk: centre -1, radius {GAIN}")
ax.set_xlabel("Re(lambda)")
ax.set_ylabel("Im(lambda)")
ax.set_title("Jacobian spectrum")
ax.legend(loc="upper left", fontsize=8)
ax.set_aspect("equal", adjustable="datalim")
fig.tight_layout()
fig.savefig(OUT / "spectrum.png", dpi=150)
plt.show()
"""))

    cells.append(md("""
## 5. Stability summary

`stability_margin` is the slowest decay rate in the system: the network returns
to its fixed point no faster than $e^{-\\text{margin}\\,t}$. `abscissa_gap` is the
first appearance of the central result: a positive numerical abscissa alongside
a negative spectral abscissa means the network is asymptotically stable and yet
grows at $t = 0$.
"""))

    cells.append(code("""
stability = jc.stability_summary(J)
stability_table = pd.Series(stability, name="value").to_frame()
display(stability_table)

print()
if stability["amplifying_but_stable"]:
    print("Stable but amplifying: alpha = "
          f"{stability['spectral_abscissa']:.4f} < 0 < omega = {stability['numerical_abscissa']:.4f}")
    print("Eigenvalues alone predict monotone decay. They are wrong about the transient.")
else:
    print(f"alpha = {stability['spectral_abscissa']:.4f}, omega = {stability['numerical_abscissa']:.4f}")
"""))
    cells.append(md("""
## 6. Non-normality and the normal surrogate

A matrix is normal when $J J^{T} = J^{T} J$; then its eigenvectors are
orthogonal and the spectrum tells the whole story. Henrici's departure measures
what is left over: $\\sqrt{\\|J\\|_F^2 - \\sum_k |\\lambda_k|^2}$, which is exactly
the Frobenius norm of the strictly upper-triangular part of the Schur form.

To make the number concrete, the **normal surrogate** is built here: a diagonal
matrix carrying the same eigenvalues as $J$ and nothing else. It has identical
asymptotic behaviour and zero transient amplification, so every difference
between $J$ and the surrogate downstream is attributable to non-normal
structure rather than to the spectrum.
"""))

    cells.append(code("""
non_normality = jc.non_normality_summary(J)
J_normal = np.real(np.diag(eigvals))  # normal surrogate: same spectrum, no coupling

comparison = pd.DataFrame({
    "jacobian": jc.non_normality_summary(J),
    "normal_surrogate": jc.non_normality_summary(J_normal),
})
display(comparison)

print()
print(f"Henrici departure: {non_normality['henrici_departure']:.3f} "
      f"({100 * non_normality['henrici_departure_relative']:.1f}% of ||J||_F)")
print(f"Eigenvector condition number: {non_normality['eigenvector_condition_number']:.3e}")
print("A large eigenvector condition number means the modal basis is far from")
print("orthogonal, so modal amplitudes can cancel and re-emerge over time.")
"""))

    cells.append(md("""
## 7. Schur decomposition and modal coupling

$J = Q T Q^{*}$ with $Q$ unitary and $T$ upper triangular. The diagonal of $T$
holds the eigenvalues; the strictly upper triangle holds the feed-forward
coupling between modes that an eigendecomposition discards. Schur coordinates
are used throughout this repository because they are numerically stable for a
strongly non-normal directed matrix and keep the cascade structure visible.

Modes are ordered by real part here, so mode 1 is the slowest-decaying one. This
is the continuous-time analogue of ordering by eigenvalue magnitude in
`schur_decomp`.
"""))

    cells.append(code("""
schur = jc.schur_decompose(J, sort="real")
print(f"reconstruction error ||Q T Q* - J||_F = {schur['reconstruction_error']:.3e}")
print(f"modal coupling norm ||triu(T, 1)||_F  = {schur['coupling_norm']:.3f}")
print(f"Henrici departure (should match)      = {non_normality['henrici_departure']:.3f}")

T = schur["T"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
im0 = axes[0].imshow(np.abs(T), cmap="magma", aspect="auto")
axes[0].set_title("|T| - Schur form")
axes[0].set_xlabel("mode")
axes[0].set_ylabel("mode")
fig.colorbar(im0, ax=axes[0], fraction=0.046)

coupling_per_mode = np.abs(schur["coupling"]).sum(axis=0)
axes[1].plot(np.arange(1, len(coupling_per_mode) + 1), coupling_per_mode, lw=1.2)
axes[1].set_title("incoming coupling per mode")
axes[1].set_xlabel("mode index")
axes[1].set_ylabel("sum |T[k, m]|, k < m")
fig.tight_layout()
fig.savefig(OUT / "schur_coupling.png", dpi=150)
plt.show()
"""))

    cells.append(md("""
## 8. Transient amplification

$\\|e^{Jt}\\|_2$ is the largest factor by which any unit initial condition can be
amplified at time $t$. For a stable normal matrix it decays monotonically from
1. Anything above 1 is transient amplification, and it requires non-normality.

Three curves are compared: the true envelope, the eigenvalue prediction
$e^{\\alpha t}$, and the normal surrogate. Where the first departs from the other
two is the part of the dynamics that the spectrum does not explain.
"""))

    cells.append(code("""
times = np.linspace(0.0, 40.0, 401)
envelope = jc.transient_envelope(J, times)
peak = jc.transient_peak(envelope)
surrogate_envelope = jc.transient_envelope(J_normal, times)

print(f"peak amplification {peak['peak_amplification']:.3f}x at t = {peak['time_of_peak']:.2f} tau")
print(f"normal surrogate peak {surrogate_envelope['growth'].max():.3f}x (expected 1.000)")

fig, ax = plt.subplots()
ax.plot(envelope["time"], envelope["growth"], lw=1.8, label="||exp(Jt)||  (true envelope)")
ax.plot(times, np.exp(stability["spectral_abscissa"] * times), lw=1.4, ls="--",
        label="exp(alpha t)  (eigenvalue prediction)")
ax.plot(surrogate_envelope["time"], surrogate_envelope["growth"], lw=1.4, ls=":",
        label="normal surrogate")
ax.axhline(1.0, color="grey", lw=0.8)
ax.axvline(peak["time_of_peak"], color="crimson", lw=0.8, ls="--")
ax.set_yscale("log")
ax.set_xlabel("time (units of tau)")
ax.set_ylabel("amplification")
ax.set_title("Transient response envelope")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "transient_envelope.png", dpi=150)
plt.show()
"""))

    cells.append(md("""
## 9. Pseudospectra and the Kreiss bound

The $\\varepsilon$-pseudospectrum is the set of complex $z$ with
$\\sigma_{\\min}(zI - J) \\le \\varepsilon$: the eigenvalues reachable under a
perturbation of norm $\\varepsilon$. For a normal matrix it is a union of discs
of radius $\\varepsilon$ around the eigenvalues. For a non-normal one it can
bulge far past that, and when it crosses the imaginary axis the network is a
perturbation of size $\\varepsilon$ away from instability even though its
eigenvalues are safely stable.

The Kreiss constant $K = \\sup_\\varepsilon \\alpha_\\varepsilon / \\varepsilon$ turns
that geometry into a certificate: $\\sup_t \\|e^{Jt}\\| \\ge K$, a guaranteed lower
bound on peak amplification that needs no simulation.
"""))

    cells.append(code("""
kreiss = jc.kreiss_constant(J, epsilons=(0.5, 0.2, 0.1, 0.05, 0.02, 0.01))
display(kreiss.round(4))

K = float(np.nanmax(kreiss["kreiss_lower_bound"]))
print(f"Kreiss lower bound on peak growth: {K:.3f}")
print(f"observed peak amplification:       {peak['peak_amplification']:.3f}")
assert peak["peak_amplification"] >= K - 1e-6, "peak growth must be at least the Kreiss bound"

# Contour plot. Reduce `resolution` if this cell is slow.
span = 1.5
re_axis, im_axis, resolvent = jc.resolvent_norm_grid(
    J, real_range=(-2.5, 0.75), imag_range=(-span, span), resolution=90
)
fig, ax = plt.subplots(figsize=(6.8, 5.2))
levels = [1e1, 1e2, 1e3, 1e4, 1e5]
contour = ax.contour(re_axis, im_axis, resolvent, levels=levels, cmap="viridis")
ax.clabel(contour, fmt=lambda v: f"eps=1e-{int(np.log10(v))}", fontsize=7)
ax.scatter(eigvals.real, eigvals.imag, s=12, color="crimson", label="eigenvalues", zorder=3)
ax.axvline(0.0, color="black", lw=1.0)
ax.set_xlabel("Re(z)")
ax.set_ylabel("Im(z)")
ax.set_title("Pseudospectra of J")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "pseudospectra.png", dpi=150)
plt.show()
"""))

    cells.append(md("""
## 10. Steady-state gain

For a stable Jacobian, a constant input $u$ drives the network to
$x = -J^{-1} u$. This is the continuous-time counterpart of the
$(I - M)^{-1}$ resolvent and Neumann-series machinery in `inhib_modulation`:
the same object, in coordinates where stability is a half-plane. A large DC gain
alongside a small stability margin is the standard signature of a network
operating near an instability.
"""))

    cells.append(code("""
gain = jc.steady_state_gain(J)
display(pd.Series(gain, name="value").to_frame())

R = jc.steady_state_response(J)
per_cell_gain = pd.DataFrame({
    "total_output_gain": np.abs(R).sum(axis=0),
    "total_input_gain": np.abs(R).sum(axis=1),
    "ei": [data.ei.loc[label] for label in labels],
}, index=labels)
display(per_cell_gain.sort_values("total_output_gain", ascending=False).head(10).round(3))
"""))

    cells.append(md("""
## 11. E/I block structure

Block-level summaries of $W$, kept here so downstream ablation notebooks have a
baseline to difference against. Blocks are named source-to-target: `EI` is E
onto I, which occupies rows $I$ and columns $E$ of a `[post, pre]` matrix.
"""))

    cells.append(code("""
block_rows = []
for block in ["EE", "EI", "IE", "II"]:
    source, target = block[0].lower(), block[1].lower()
    rows = np.flatnonzero(masks[target])
    cols = np.flatnonzero(masks[source])
    sub = W[np.ix_(rows, cols)]
    nz = sub[sub != 0]
    block_rows.append({
        "block": block,
        "shape": f"{sub.shape[0]}x{sub.shape[1]}",
        "n_connections": int(nz.size),
        "density": float(nz.size / sub.size) if sub.size else np.nan,
        "mean_weight": float(nz.mean()) if nz.size else np.nan,
        "frobenius_norm": float(np.linalg.norm(sub, ord="fro")),
        "fraction_positive": float((nz > 0).mean()) if nz.size else np.nan,
    })
blocks = pd.DataFrame(block_rows).set_index("block")
display(blocks.round(4))

diag = np.diag(W)
print(f"self-connections: {int(np.count_nonzero(diag))} nonzero, "
      f"range [{diag.min():.4f}, {diag.max():.4f}]")
"""))

    cells.append(md("""
## 12. Save outputs and verify

Written to `outputs/01_jacobian_construction/`. The final cell asserts the
invariants the rest of the project relies on, so a broken assumption fails here
rather than surfacing as a plausible-looking wrong number three notebooks later.
"""))

    cells.append(code("""
jc.save_table(J_frame, OUT, "jacobian_post_by_pre")
jc.save_table(data.frame(W), OUT, "weights_normalized_post_by_pre")
jc.save_table(spectrum, OUT, "eigenvalue_spectrum", index=False)
jc.save_table(stability_table, OUT, "stability_summary")
jc.save_table(comparison, OUT, "non_normality_vs_normal_surrogate")
jc.save_table(envelope, OUT, "transient_envelope", index=False)
jc.save_table(kreiss, OUT, "kreiss_bounds", index=False)
jc.save_table(blocks, OUT, "ei_block_summary")
jc.save_table(per_cell_gain, OUT, "steady_state_gain_per_cell")
jc.save_table(normalization_options, OUT, "normalization_options")
print("saved:", sorted(p.name for p in OUT.glob("*.csv")))
"""))

    cells.append(code("""
# Verification
assert data.orientation_check["E_to_I_positive"] > 0.9, "orientation check failed"
assert data.orientation_check["I_to_E_negative"] > 0.9, "orientation check failed"
assert abs(norm_info["spectral_radius"] - GAIN) < 1e-9, "normalization did not hit its target"
assert stability["spectral_abscissa"] < 0, "J should be stable at gain < 1"
assert abs(stability["spectral_abscissa"] - (norm_info["spectral_abscissa"] - LEAK)) < 1e-8, \\
    "spectrum of J must be the spectrum of W shifted left by the leak"
assert schur["reconstruction_error"] < 1e-8, "Schur reconstruction is not accurate"
assert abs(schur["coupling_norm"] - non_normality["henrici_departure"]) < 1e-6, \\
    "Henrici departure must equal the norm of the strict upper triangle of T"
assert abs(envelope["growth"].iloc[0] - 1.0) < 1e-9, "||exp(J*0)|| must be exactly 1"
assert peak["peak_amplification"] >= K - 1e-6, "peak growth violates the Kreiss lower bound"
assert surrogate_envelope["growth"].max() < 1.0 + 1e-8, "the normal surrogate must not amplify"
print("all checks passed")
"""))

    cells.append(md("""
## Result

At `rho(W) = 0.95`, `tau = 1`, leak `= 1`:

| Quantity | Value | Reading |
|---|---:|---|
| Stability margin | 0.0500 | slowest decay rate; time constant 20 tau |
| Numerical abscissa | 29.58 | growth rate at t = 0 |
| Peak amplification | 27.11x at t = 1.2 tau | worst-case transient gain |
| Kreiss lower bound | 18.47 (eps = 0.05) | amplification guaranteed without simulation |
| Henrici departure | 77.17 (99.3% of ||J||_F) | distance from normal |
| Eigenvector condition number | 9.9e4 | modal basis far from orthogonal |

The network is asymptotically stable and amplifies a worst-case perturbation
27-fold before decaying. The normal surrogate, carrying the same eigenvalues and
no coupling, has peak amplification exactly 1.0, so the amplification is
attributable to non-normal structure rather than to the spectrum.

Note also that `rho(J) = 1.31`: the same matrix would be judged unstable by the
discrete-time criterion used in `schur_decomp` and `inhib_modulation`. The two
criteria are not interchangeable.

Next: `02_jacobian_schur_modes.ipynb` asks which cell types and regions carry
the amplifying modes.
"""))
    return cells


SETUP = PREAMBLE + """
data = jc.load_jacobian_data(MATRIX, NETLIST)
labels = data.labels
masks = jc.ei_masks(data.ei, labels)
W, norm_info = jc.normalize_weights(data.W_raw, method="spectral_radius", target=GAIN)
J = jc.build_jacobian(W, tau=TAU, leak=LEAK)
baseline = jc.stability_summary(J)
print(f"alpha = {baseline['spectral_abscissa']:.4f}   omega = {baseline['numerical_abscissa']:.4f}")
"""


# ==========================================================================
# 02-06 - scaffolds
# ==========================================================================


def notebook_02() -> list[nbf.NotebookNode]:
    return [
        md("""
# 02 - Schur modes: which cells carry the amplification

**Status: scaffold.** Sections and helper calls are sketched; the analysis is
not written.

Notebook 01 established that the network is stable but strongly amplifying. This
notebook asks *where* that amplification lives. The Schur basis is the right
coordinate system for the question: its strictly upper triangle is the
feed-forward cascade between modes, and amplification is precisely energy
flowing down that cascade before it decays.

Continuous-time reading of the Schur form: modes are ordered by real part, so
mode 1 is slowest-decaying. Coupling `T[k, m]` with `k < m` means mode `m` feeds
mode `k`, i.e. a faster-decaying mode pumps a slower one. That is the structural
mechanism behind a positive numerical abscissa.

This is the port of `schur_decomp/03_schur_signal.ipynb` and
`05_schur_greedy_trees.ipynb` to the Jacobian.
"""),
        code(SETUP + """
OUT = jc.output_dir("02_jacobian_schur_modes")
schur = jc.schur_decompose(J, sort="real")
T, Q = schur["T"], schur["Q"]
"""),
        md("""
## 1. Mode inventory

One row per mode: eigenvalue, decay time constant, oscillation frequency,
incoming and outgoing coupling weight, and the dominant contributing cell type
from `|Q[:, m]|`.
"""),
        code("""
# TODO: build the mode table.
# eigenvalues = schur["eigenvalues"]; loadings = np.abs(Q)
# dominant_cell = [labels[int(np.argmax(loadings[:, m]))] for m in range(len(labels))]
"""),
        md("""
## 2. Cell-type and region loading

Which cell types load onto the slowest modes, and onto the most strongly coupled
ones? Region is the first token of the label (fold `CA3c` into `CA3`, as in
`schur_core_script.region_from_label`).
"""),
        code("""
# TODO: loading summaries by cell type, by E/I class, and by region.
"""),
        md("""
## 3. Coupling cascade structure

Where in the mode ordering does the coupling concentrate? A cascade that runs
from fast to slow modes amplifies; one confined within a narrow band does not.
"""),
        code("""
# TODO: coupling heatmap by mode distance |k - m|; cumulative coupling profile.
"""),
        md("""
## 4. Greedy contributor trees

Port of `schur_decomp/05_schur_greedy_trees.ipynb`: for each leading mode, take
the top contributors from `|Q[:, m]|` and grow a greedy information-flow tree
over `W` restricted to those cells.
"""),
        code("""
# TODO: greedy tree per leading mode; count components; tabulate tree edges.
"""),
        md("""
## 5. Save and verify
"""),
        code("""
# TODO: save tables to OUT; assert reconstruction error and coupling norm match notebook 01.
"""),
    ]


def notebook_03() -> list[nbf.NotebookNode]:
    return [
        md("""
# 03 - Transient amplification and optimal perturbations

**Status: scaffold.** Sections and helper calls are sketched; the analysis is
not written.

Notebook 01 measured *how much* the network amplifies. This one asks *what gets
amplified*: the initial condition that grows most, what it looks like in cell
types, and whether it is biologically reachable.

The tool is the SVD of the propagator. At each time $t$, the leading right
singular vector of $e^{Jt}$ is the optimal initial condition and the leading left
singular vector is the state it evolves into; the leading singular value is the
amplification. This has no discrete-time equivalent in the existing notebooks -
it is new capability the Jacobian formulation buys.
"""),
        code(SETUP + """
OUT = jc.output_dir("03_jacobian_transient_amplification")
from scipy.linalg import expm
"""),
        md("""
## 1. Optimal perturbation at the peak time

At $t^{*}$ from notebook 01, take `U, s, Vh = svd(expm(J * t_star))`. `Vh[0]` is
the optimal initial condition, `U[:, 0]` the amplified output state, `s[0]` the
gain.
"""),
        code("""
# TODO: SVD of the propagator at the peak; tabulate the top cells of input and output states.
"""),
        md("""
## 2. Input and output structure over time

How does the optimal perturbation change with the horizon? Track the E/I
composition and region composition of `Vh[0]` and `U[:, 0]` across `t`.
"""),
        code("""
# TODO: sweep t; record composition and overlap between successive optimal inputs.
"""),
        md("""
## 3. Biological reachability

The optimal perturbation is a worst case and may be a fine-tuned pattern no
stimulus could produce. Compare it against realistic seeds: single cell types,
whole regions, and E-only or I-only drive.
"""),
        code("""
# TODO: seeded envelopes via jc.transient_envelope(J, times, x0=seed); rank seeds by peak gain.
"""),
        md("""
## 4. Pseudospectra in depth

Extend the notebook 01 contour: resolve the boundary crossing of the imaginary
axis and record the critical epsilon at which the pseudospectrum first becomes
unstable.
"""),
        code("""
# TODO: bisect on epsilon using jc.pseudospectral_abscissa to find the critical value.
"""),
        md("""
## 5. Save and verify
"""),
        code("""
# TODO: save; assert s[0] at the peak matches the notebook 01 peak amplification.
"""),
    ]


def notebook_04() -> list[nbf.NotebookNode]:
    return [
        md("""
# 04 - Ablation tests

**Status: scaffold.** Sections and helper calls are sketched; the analysis is
not written.

Direct port of the ablation workflow in
`inhib_modulation/02_inhibitory_modulation_analysis.ipynb`, with the readouts
replaced by continuous-time ones. Each ablated network is summarized by
`jc.condition_summary`, which returns stability, non-normality, transient, and
DC-gain numbers in a single row, and `jc.ablation_sweep` differences every
condition against the intact network.

**Methodological note.** Ablations are *not* renormalized by default. A cut
changes the network's gain, and renormalizing afterwards would scale that change
away and report only the residual structural effect. Both questions are
legitimate but they are different questions: run the sweep twice, with
`renormalize=None` and `renormalize="spectral_radius"`, and report both.
"""),
        code(SETUP + """
OUT = jc.output_dir("04_jacobian_ablation_tests")
n = len(labels)
"""),
        md("""
## 1. Single-cell-type ablations

Silence one cell type at a time (`jc.ablate_nodes`) and rank by the change in
stability margin, numerical abscissa, and peak amplification. The three
rankings need not agree, and where they disagree is the interesting part: a cell
whose removal barely moves the eigenvalues but collapses the transient is doing
structural, not spectral, work.
"""),
        code("""
# TODO:
# conditions = {labels[i]: jc.ablate_nodes(W, [i]) for i in range(n)}
# table = jc.ablation_sweep(W, conditions, reference="intact", with_transient=True)
"""),
        md("""
## 2. Edge ablations

Rank individual connections. Start with the strongest edges by weight, then by
the eigenvalue-sensitivity criterion $|v_i u_j|$ from the dominant left and right
eigenvectors, which predicts first-order spectral effect without re-solving.
"""),
        code("""
# TODO: top-k edges by weight and by sensitivity; jc.ablate_edges; compare predicted vs actual shift.
"""),
        md("""
## 3. Graded weakening

Ablation is the endpoint of a continuum. Sweep `factor` from 1.0 to 0.0 for the
top candidates and check whether the effect is linear or has a threshold.
"""),
        code("""
# TODO: factor sweep per candidate; plot stability margin and peak amplification vs factor.
"""),
        md("""
## 4. Null comparison

Compare measured ablation effects against a block-preserving weight shuffle, so
"this edge matters" is a claim about this network rather than about any network
with this degree and weight distribution.
"""),
        code("""
# TODO: shuffled nulls; z-scores per condition.
"""),
        md("""
## 5. Save and verify
"""),
        code("""
# TODO: save both renormalized and un-renormalized sweeps; assert the intact row matches notebook 01.
"""),
    ]


def notebook_05() -> list[nbf.NotebookNode]:
    return [
        md("""
# 05 - E/I block fractures and inhibitory structure

**Status: scaffold.** Sections and helper calls are sketched; the analysis is
not written.

Coarse structural fractures: remove or scale each E/I block and the
self-connections, and ask what holds the network's stability together. Port of
the block-removal and Schur-complement sections of `inhib_modulation`.

Blocks are named source-to-target throughout (`EI` = E onto I = rows I, columns
E of a `[post, pre]` matrix). `jc.ablate_block` spells the indexing out so the
naming collision documented in `docs/orientation_convention_proposal.md` cannot
propagate.

The continuous-time framing sharpens the ISN question. In an
inhibition-stabilized network the excitatory subnetwork is unstable on its own
and stability comes from feedback inhibition; here that is testable directly by
comparing $\\max\\operatorname{Re}\\lambda$ of $J_{EE}$ against that of the full $J$.
"""),
        code(SETUP + """
OUT = jc.output_dir("05_jacobian_ei_block_fractures")
"""),
        md("""
## 1. Block removals

`EE`, `EI`, `IE`, `II`, self-connections, and the two composite cuts (all
inhibitory output, all inhibitory input).
"""),
        code("""
# TODO:
# conditions = {b: jc.ablate_block(W, masks, b) for b in ("EE", "EI", "IE", "II")}
# conditions["self"] = jc.ablate_self_connections(W)
# table = jc.ablation_sweep(W, conditions, reference="intact")
"""),
        md("""
## 2. Graded inhibitory gain

Scale `IE` and `II` by a factor sweep. Track where the spectral abscissa crosses
zero: that crossing is the inhibitory gain at which the network loses stability,
and it is the cleanest single number this project can report about the role of
inhibition.
"""),
        code("""
# TODO: factor sweep on inhibitory blocks; locate the zero crossing of the spectral abscissa.
"""),
        md("""
## 3. ISN test

Compare the excitatory submatrix in isolation against the full network. An
unstable $J_{EE}$ with a stable $J$ is the definition of an
inhibition-stabilized network.
"""),
        code("""
# TODO: eigenvalues of J[E, E] vs J; report both abscissas.
"""),
        md("""
## 4. Schur complement of inhibition

Eliminate the inhibitory variables to get the effective excitatory Jacobian
$J_{EE} - J_{EI} J_{II}^{-1} J_{IE}$, the continuous-time version of the
inhibitory Schur complement in `inhib_modulation`. Compare its non-normality
against the raw $J_{EE}$: effective inhibitory feedback can create amplification
that neither block shows alone.
"""),
        code("""
# TODO: partition J by masks; form the Schur complement; run stability and non-normality on it.
"""),
        md("""
## 5. Save and verify
"""),
        code("""
# TODO: save; assert the block partition recovers J exactly when reassembled.
"""),
    ]


def notebook_06() -> list[nbf.NotebookNode]:
    return [
        md("""
# 06 - Matched comparisons: self-connections, gain, and time constants

**Status: scaffold.** Sections and helper calls are sketched; the analysis is
not written.

The comparison notebook, following the `_compare` pattern in `schur_decomp`.
Every result in notebooks 01-05 is conditional on three modelling choices; this
notebook tests how much each of them is carrying.

1. **Self-connections.** With-self versus no-self, each normalized
   independently, so the comparison is between matched normalized dynamics
   rather than raw weights. Note the difference from the discrete-time projects:
   the diagonal of `W` is a genuine autapse term, while the `-I` leak lives in
   `J` and is never removed.
2. **Gain.** The choice of $\\rho(W) = 0.95$ is a convention. Sweep it and check
   whether conclusions are gain-dependent or hold across the stable range.
3. **Time constants.** Uniform $\\tau$ is an assumption. Non-uniform $\\tau$ is a
   left diagonal scaling that introduces non-normality on its own, so a
   sensitivity check matters here more than it does in discrete time.
"""),
        code(SETUP + """
OUT = jc.output_dir("06_jacobian_compare_variants")
variants = jc.matrix_variants(data.W_raw)
"""),
        md("""
## 1. With-self versus no-self

Normalize each variant independently to the same gain, then compare stability,
non-normality, transient peak, dominant eigenvector overlap, and DC gain.
"""),
        code("""
# TODO:
# rows = []
# for name, W_raw_variant in variants.items():
#     W_v, info = jc.normalize_weights(W_raw_variant, "spectral_radius", GAIN)
#     rows.append(jc.condition_summary(W_v, name, tau=TAU, leak=LEAK))
"""),
        md("""
## 2. Gain sweep

Sweep $\\rho(W)$ across the stable range and up to the boundary. Report peak
amplification and the Kreiss bound as functions of gain; expect both to diverge
as gain approaches 1.
"""),
        code("""
# TODO: gain sweep; plot peak amplification and stability margin vs gain.
"""),
        md("""
## 3. Time-constant heterogeneity

Compare uniform $\\tau$ against a plausible E/I split (inhibitory interneurons
are typically faster), and against randomized $\\tau$ as a null.
"""),
        code("""
# TODO: per-cell tau vectors; jc.build_jacobian(W, tau=tau_vector); compare summaries.
"""),
        md("""
## 4. Leak sensitivity
"""),
        code("""
# TODO: sweep the leak coefficient; confirm the spectrum shifts as expected.
"""),
        md("""
## 5. Save and verify
"""),
        code("""
# TODO: save the comparison table; assert the with_self row reproduces notebook 01.
"""),
    ]


NOTEBOOKS = {
    "01": ("01_jacobian_construction.ipynb", notebook_01),
    "02": ("02_jacobian_schur_modes.ipynb", notebook_02),
    "03": ("03_jacobian_transient_amplification.ipynb", notebook_03),
    "04": ("04_jacobian_ablation_tests.ipynb", notebook_04),
    "05": ("05_jacobian_ei_block_fractures.ipynb", notebook_05),
    "06": ("06_jacobian_compare_variants.ipynb", notebook_06),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="notebook numbers to build, e.g. --only 01 04")
    parser.add_argument("--skip-existing", action="store_true", help="do not overwrite notebooks already on disk")
    args = parser.parse_args()

    wanted = args.only or list(NOTEBOOKS)
    for key in wanted:
        if key not in NOTEBOOKS:
            raise SystemExit(f"unknown notebook {key}; choose from {sorted(NOTEBOOKS)}")
        name, builder = NOTEBOOKS[key]
        if args.skip_existing and (ROOT / name).exists():
            print(f"skip   {name}")
            continue
        path = write(name, builder())
        print(f"wrote  {path.name}")


if __name__ == "__main__":
    main()
