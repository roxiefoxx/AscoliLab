"""Shared core utilities for the Jacobian analysis notebooks.

This module is the Jacobian-side counterpart to ``schur_decomp/schur_core_script.py``.
It holds the pieces that must stay identical across every notebook in this
project: loading and orienting the Mij data, asserting orientation, normalizing
the weight matrix, building the Jacobian, and the continuous-time stability,
non-normality, Schur, transient, and ablation primitives.

Model
-----
The existing projects in this repository treat the connectivity matrix as a
discrete-time state-update operator::

    x[t + 1] = M x[t]                     stable iff  rho(M) < 1

This project instead treats it as the Jacobian of a continuous-time linear rate
model linearized about its fixed point::

    tau_i dx_i/dt = -x_i + sum_j W_ij x_j
    J = (-I + W) / tau                    stable iff  max Re(lambda) < 0

``W`` is the same oriented, normalized ``M[post, pre]`` used everywhere else, so
every workflow in ``schur_decomp`` and ``inhib_modulation`` ports over directly.
Only the stability criterion changes, and it changes in a useful way: a discrete
system has one scalar summary (the spectral radius), while a continuous system
separates the *asymptotic* rate (spectral abscissa), the *worst-case initial*
growth rate (numerical abscissa), and the *finite-time* transient envelope
(pseudospectral abscissa / Kreiss constant). Those three coincide for a normal
matrix and diverge for a non-normal one, which is the point of this project.

Orientation
-----------
Every ``mij_matrix.csv`` on disk is ``M[pre, post]`` and must be transposed
before use. See ``docs/TRANSPOSE_WARNINGS.md`` at the repository root. This
module transposes on load and then *verifies* the result with the Dale's-law
sign oracle from ``docs/orientation_convention_proposal.md``, so orientation is
checked rather than remembered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "JacobianData",
    "assert_post_pre",
    "load_jacobian_data",
    "normalize_weights",
    "build_jacobian",
    "matrix_variants",
    "ei_masks",
    "stability_summary",
    "non_normality_summary",
    "schur_decompose",
    "transient_envelope",
    "pseudospectral_abscissa",
    "resolvent_norm_grid",
    "kreiss_constant",
    "ablate_nodes",
    "ablate_edges",
    "ablate_block",
    "ablate_self_connections",
    "condition_summary",
    "ablation_sweep",
    "save_table",
    "output_dir",
]


# --------------------------------------------------------------------------
# Loading and orientation
# --------------------------------------------------------------------------


@dataclass
class JacobianData:
    """Oriented connectivity plus everything needed to build a Jacobian.

    ``W_raw`` is ``W[post, pre]``: rows are postsynaptic receivers, columns are
    presynaptic sources, so ``W[i, j]`` is the influence of ``j`` on ``i``.
    ``df_pre_post`` keeps the untransposed CSV exactly as it sits on disk.
    """

    W_raw: np.ndarray
    labels: list[str]
    ei: pd.Series
    df_pre_post: pd.DataFrame
    netlist: pd.DataFrame | None
    matrix_path: Path
    netlist_path: Path | None
    orientation_check: dict[str, float] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.W_raw.shape[0]

    @property
    def excitatory(self) -> list[str]:
        return self.ei[self.ei == "e"].index.tolist()

    @property
    def inhibitory(self) -> list[str]:
        return self.ei[self.ei == "i"].index.tolist()

    def frame(self, M: np.ndarray) -> pd.DataFrame:
        """Wrap a same-shaped array as a labeled post-by-pre DataFrame."""
        return pd.DataFrame(M, index=self.labels, columns=self.labels)


def assert_post_pre(
    M: np.ndarray,
    classes: Sequence[str],
    *,
    name: str = "matrix",
    min_purity: float = 0.9,
) -> dict[str, float]:
    """Raise unless ``M`` is oriented ``M[post, pre]``.

    E/I identity in this dataset comes from the sign of outgoing weights, so the
    sign pattern of the off-diagonal E/I blocks is itself an orientation oracle:
    E to I entries are positive and I to E entries are negative, and the two
    blocks swap completely under transposition. Separation is 100/0 in this
    dataset, so there is no tolerance to tune.

    Returns the measured purities so a notebook can print them as evidence.
    """
    M = np.asarray(M, dtype=float)
    cls = np.asarray([str(c).lower() for c in classes])
    E = np.flatnonzero(cls == "e")
    I = np.flatnonzero(cls == "i")
    if not len(E) or not len(I):
        return {}

    def purity(rows: np.ndarray, cols: np.ndarray, want_positive: bool) -> float | None:
        block = M[np.ix_(rows, cols)]
        nz = block[block != 0]
        if nz.size == 0:
            return None
        frac = float((nz > 0).mean())
        return frac if want_positive else 1.0 - frac

    e_to_i = purity(I, E, True)   # rows = I receivers, cols = E sources
    i_to_e = purity(E, I, False)  # rows = E receivers, cols = I sources
    measured = {
        "E_to_I_positive": e_to_i if e_to_i is not None else float("nan"),
        "I_to_E_negative": i_to_e if i_to_e is not None else float("nan"),
        "n_excitatory": float(len(E)),
        "n_inhibitory": float(len(I)),
    }
    bad = [
        f"{key}={value:.3f}"
        for key, value in (("E_to_I_positive", e_to_i), ("I_to_E_negative", i_to_e))
        if value is not None and value < min_purity
    ]
    if bad:
        raise ValueError(
            f"{name} does not look like M[post, pre]: {', '.join(bad)}. "
            "Excitatory sources should drive positive columns and inhibitory "
            "sources negative ones. The matrix is probably transposed."
        )
    return measured


def load_jacobian_data(
    matrix_path: str | Path = "matrices/mij_matrix.csv",
    netlist_path: str | Path | None = "matrices/mij_netlist.csv",
    *,
    transpose: bool = True,
    verify_orientation: bool = True,
) -> JacobianData:
    """Load ``mij_matrix.csv``, transpose to ``[post, pre]``, and verify it.

    ``transpose=True`` is correct for every ``mij_matrix.csv`` in this
    repository. Set it to ``False`` only for a file already stored as
    ``[post, pre]`` (for example ``normalized_J_receiver_by_sender.csv``); the
    orientation check will catch the mistake either way.
    """
    matrix_path = Path(matrix_path)
    df = pd.read_csv(matrix_path, index_col=0)
    if df.shape[0] != df.shape[1]:
        raise ValueError(f"Mij must be square; got {df.shape}.")
    if list(df.index) != list(df.columns):
        raise ValueError("Mij row and column labels must match in the same order.")
    if df.index.has_duplicates:
        raise ValueError("Cell-type labels must be unique.")
    df = df.astype(float)

    labels = list(df.columns)
    W_raw = df.to_numpy(dtype=float).T.copy() if transpose else df.to_numpy(dtype=float).copy()

    netlist = None
    netlist_path_obj = Path(netlist_path) if netlist_path is not None else None
    ei = pd.Series(index=labels, dtype="object")
    if netlist_path_obj is not None and netlist_path_obj.exists():
        netlist = pd.read_csv(netlist_path_obj)
        mapping: dict[str, str] = {}
        for column_pair in (("pre_neuron", "pre_ei"), ("post_neuron", "post_ei")):
            neuron_col, ei_col = column_pair
            if neuron_col not in netlist.columns or ei_col not in netlist.columns:
                continue
            pairs = netlist[[neuron_col, ei_col]].dropna().drop_duplicates()
            for row in pairs.itertuples(index=False):
                key = str(getattr(row, neuron_col))
                value = str(getattr(row, ei_col)).strip().lower()
                previous = mapping.get(key)
                if previous is not None and previous != value:
                    raise ValueError(f"Conflicting E/I metadata for {key}: {previous} vs {value}")
                mapping[key] = value
        ei = pd.Series({label: mapping.get(label, np.nan) for label in labels}, dtype="object")

    if ei.isna().any():
        # Fall back to the sign of each source's outgoing column in [post, pre].
        inferred = {}
        for j, label in enumerate(labels):
            column = W_raw[:, j]
            inferred[label] = "i" if np.sum(column < 0) > np.sum(column > 0) else "e"
        ei = ei.combine_first(pd.Series(inferred, dtype="object"))
    if ei.isna().any():
        raise ValueError(f"Missing E/I metadata for {ei[ei.isna()].index.tolist()[:10]}")
    ei = ei.astype(str).str.lower()

    check: dict[str, float] = {}
    if verify_orientation:
        check = assert_post_pre(W_raw, ei.loc[labels].tolist(), name=str(matrix_path))

    return JacobianData(
        W_raw=W_raw,
        labels=labels,
        ei=ei,
        df_pre_post=df,
        netlist=netlist,
        matrix_path=matrix_path,
        netlist_path=netlist_path_obj,
        orientation_check=check,
    )


# --------------------------------------------------------------------------
# Normalization and Jacobian construction
# --------------------------------------------------------------------------


def normalize_weights(
    W: np.ndarray,
    method: str = "spectral_radius",
    target: float = 0.95,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Scale the weight matrix before the leak term is added.

    Normalization is applied to ``W``, never to ``J``. Scaling ``J`` would
    rescale the leak along with the synaptic weights and silently change the
    membrane time constant; scaling ``W`` changes only the synaptic gain, which
    is the quantity the analysis is actually varying.

    Methods
    -------
    ``spectral_radius``
        Set ``rho(W)`` to ``target``. With ``target < 1`` the resulting Jacobian
        is guaranteed stable, since ``lambda_J = lambda_W - 1`` and every
        ``lambda_W`` sits inside a disk of radius ``target``. This is the direct
        analogue of the ``0.95`` and ``0.85`` normalization used in
        ``schur_decomp`` and ``inhib_modulation``, so results are comparable.
    ``spectral_abscissa``
        Set ``max Re(lambda_W)`` to ``target``. Controls the stability margin
        exactly (the margin becomes ``1 - target``) while leaving the rest of the
        spectrum free. Use this when the question is about distance to
        instability rather than about matched overall gain.
    ``frobenius``
        Set ``||W||_F`` to ``target``. Matched total synaptic drive; makes no
        stability guarantee.
    ``none``
        Leave the raw weights alone. The raw matrix has a spectral radius of
        roughly 2.6e4, so the Jacobian is wildly unstable and only structural
        (not dynamical) quantities are meaningful.
    """
    W = np.asarray(W, dtype=float)
    method = method.lower().strip()
    eigvals = np.linalg.eigvals(W)
    rho = float(np.max(np.abs(eigvals)))
    alpha = float(np.max(eigvals.real))

    if method == "none":
        scale = 1.0
    elif method in {"spectral_radius", "rho"}:
        if rho <= np.finfo(float).eps:
            raise ValueError("Cannot spectral-radius normalize a zero matrix.")
        scale = target / rho
    elif method in {"spectral_abscissa", "alpha"}:
        if alpha <= np.finfo(float).eps:
            raise ValueError("Cannot abscissa-normalize a matrix with nonpositive abscissa.")
        scale = target / alpha
    elif method in {"frobenius", "fro"}:
        norm = float(np.linalg.norm(W, ord="fro"))
        if norm <= np.finfo(float).eps:
            raise ValueError("Cannot Frobenius-normalize a zero matrix.")
        scale = target / norm
    else:
        raise ValueError("method must be 'none', 'spectral_radius', 'spectral_abscissa', or 'frobenius'.")

    out = W * scale
    out_eigvals = np.linalg.eigvals(out)
    info: dict[str, float | str] = {
        "method": method,
        "target": float(target),
        "scale": float(scale),
        "raw_spectral_radius": rho,
        "raw_spectral_abscissa": alpha,
        "spectral_radius": float(np.max(np.abs(out_eigvals))),
        "spectral_abscissa": float(np.max(out_eigvals.real)),
        "frobenius_norm": float(np.linalg.norm(out, ord="fro")),
    }
    return out, info


def build_jacobian(
    W: np.ndarray,
    tau: float | np.ndarray = 1.0,
    leak: float = 1.0,
) -> np.ndarray:
    """Return ``J = diag(1/tau) @ (-leak * I + W)``.

    ``tau`` may be a scalar or a length-N vector of per-cell-type membrane time
    constants. A non-uniform ``tau`` acts as a left diagonal scaling, which
    changes the eigenvalues and adds non-normality on its own; keep it at the
    scalar default unless a notebook is explicitly asking that question.
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    if W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square; got {W.shape}.")
    core = -leak * np.eye(n) + W
    tau_arr = np.broadcast_to(np.asarray(tau, dtype=float), (n,)).astype(float)
    if np.any(tau_arr <= 0):
        raise ValueError("tau must be strictly positive.")
    return core / tau_arr[:, np.newaxis]


def matrix_variants(W: np.ndarray) -> dict[str, np.ndarray]:
    """Return ``with_self`` and ``no_self`` versions of a weight matrix.

    Note the asymmetry with the discrete-time projects: here the diagonal of
    ``W`` is a genuine self-connection, while the ``-I`` leak lives in ``J`` and
    is never removed. Zeroing the diagonal of ``W`` removes autapse-like
    recurrence and leaves the leak intact.
    """
    with_self = np.asarray(W, dtype=float).copy()
    no_self = with_self.copy()
    np.fill_diagonal(no_self, 0.0)
    return {"with_self": with_self, "no_self": no_self}


def ei_masks(ei: pd.Series, labels: Sequence[str]) -> dict[str, np.ndarray]:
    """Return boolean index arrays for the excitatory and inhibitory sets."""
    values = np.asarray([str(ei.loc[label]).lower() for label in labels])
    return {"e": values == "e", "i": values == "i"}


# --------------------------------------------------------------------------
# Continuous-time stability and non-normality
# --------------------------------------------------------------------------


def stability_summary(J: np.ndarray) -> dict[str, float | bool]:
    """Continuous-time stability diagnostics for a Jacobian.

    The three growth rates returned here are the reason this project exists.
    For a normal matrix they collapse onto one number; the gap between them is
    exactly the transient amplification that eigenvalues alone cannot see.

    ``spectral_abscissa``
        ``max Re(lambda)``. The asymptotic growth rate: sets the sign of
        stability and the long-time decay rate, nothing else.
    ``numerical_abscissa``
        ``lambda_max((J + J^T) / 2)``. The instantaneous worst-case growth rate
        at ``t = 0``. Strictly greater than the spectral abscissa for a
        non-normal matrix, and a positive value with a negative spectral
        abscissa is the signature of a stable network that still amplifies.
    ``stability_margin``
        ``-max Re(lambda)``. Positive means stable; the value is the slowest
        decay rate in the system.
    """
    J = np.asarray(J, dtype=float)
    eigvals = np.linalg.eigvals(J)
    dominant_idx = int(np.argmax(eigvals.real))
    dominant = complex(eigvals[dominant_idx])
    symmetric_part = (J + J.T) / 2.0
    numerical_abscissa = float(np.max(np.linalg.eigvalsh(symmetric_part)))
    spectral_abscissa = float(np.max(eigvals.real))
    return {
        "spectral_abscissa": spectral_abscissa,
        "stability_margin": -spectral_abscissa,
        "stable": bool(spectral_abscissa < 0.0),
        "numerical_abscissa": numerical_abscissa,
        "abscissa_gap": numerical_abscissa - spectral_abscissa,
        "amplifying_but_stable": bool(spectral_abscissa < 0.0 < numerical_abscissa),
        "spectral_radius": float(np.max(np.abs(eigvals))),
        "dominant_real": float(dominant.real),
        "dominant_imag": float(dominant.imag),
        "dominant_frequency_hz_per_tau": float(abs(dominant.imag) / (2.0 * np.pi)),
        "slowest_time_constant": float(-1.0 / spectral_abscissa) if spectral_abscissa < 0 else float("inf"),
        "n_unstable_modes": int(np.sum(eigvals.real > 0.0)),
        "n_oscillatory_modes": int(np.sum(np.abs(eigvals.imag) > 1e-12)),
    }


def non_normality_summary(J: np.ndarray) -> dict[str, float]:
    """Quantify how far ``J`` is from normal, by several standard measures."""
    J = np.asarray(J, dtype=float)
    eigvals = np.linalg.eigvals(J)
    frob_sq = float(np.linalg.norm(J, ord="fro") ** 2)
    eig_sq = float(np.sum(np.abs(eigvals) ** 2))
    henrici = float(np.sqrt(max(frob_sq - eig_sq, 0.0)))
    commutator = J @ J.T - J.T @ J
    try:
        eigvecs = np.linalg.eig(J)[1]
        eigvec_condition = float(np.linalg.cond(eigvecs))
    except np.linalg.LinAlgError:
        eigvec_condition = float("inf")
    singular_values = np.linalg.svd(J, compute_uv=False)
    return {
        "henrici_departure": henrici,
        "henrici_departure_relative": henrici / np.sqrt(frob_sq) if frob_sq > 0 else 0.0,
        "commutator_norm": float(np.linalg.norm(commutator, ord="fro")),
        "eigenvector_condition_number": eigvec_condition,
        "frobenius_norm": float(np.sqrt(frob_sq)),
        "spectral_norm": float(singular_values[0]),
        "smallest_singular_value": float(singular_values[-1]),
        "condition_number": float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else float("inf"),
    }


def schur_decompose(J: np.ndarray, sort: str = "real") -> dict[str, np.ndarray]:
    """Complex Schur decomposition ``J = Q T Q*``, ordered by mode dominance.

    Ordering by real part puts the slowest-decaying (most dynamically relevant)
    mode first, which is the continuous-time analogue of ordering by eigenvalue
    magnitude in the discrete-time notebooks. The strictly upper-triangular part
    of ``T`` is the feed-forward coupling between modes: it is what carries
    transient amplification, and it is zero exactly when ``J`` is normal.
    """
    from scipy.linalg import schur

    J = np.asarray(J, dtype=float)
    T, Q = schur(J, output="complex")
    diag = np.diag(T)
    if sort == "real":
        order = np.argsort(-diag.real)
    elif sort == "abs":
        order = np.argsort(-np.abs(diag))
    elif sort == "none":
        order = np.arange(len(diag))
    else:
        raise ValueError("sort must be 'real', 'abs', or 'none'.")
    # Reordering the basis without re-triangularizing would break T, so reorder
    # by repeated similarity is avoided here: return the ordering as an index
    # array and let callers index the diagonal and coupling summaries with it.
    coupling = np.triu(T, k=1)
    return {
        "T": T,
        "Q": Q,
        "eigenvalues": diag,
        "order": order,
        "coupling": coupling,
        "coupling_norm": float(np.linalg.norm(coupling, ord="fro")),
        "reconstruction_error": float(np.linalg.norm(Q @ T @ Q.conj().T - J, ord="fro")),
    }


def transient_envelope(
    J: np.ndarray,
    times: Iterable[float] | None = None,
    x0: np.ndarray | None = None,
) -> pd.DataFrame:
    """Worst-case (and optionally seeded) response envelope over time.

    ``growth`` is ``||exp(J t)||_2``: the largest factor by which any unit
    initial condition can be amplified at time ``t``. For a stable normal matrix
    it decays monotonically from 1. Any value above 1 is transient amplification
    and can only happen when ``J`` is non-normal.
    """
    from scipy.linalg import expm

    J = np.asarray(J, dtype=float)
    if times is None:
        times = np.linspace(0.0, 20.0, 201)
    times = np.asarray(list(times), dtype=float)

    rows = []
    for t in times:
        E = expm(J * t)
        row = {
            "time": float(t),
            "growth": float(np.linalg.norm(E, ord=2)),
            "frobenius_growth": float(np.linalg.norm(E, ord="fro")),
        }
        if x0 is not None:
            row["seeded_norm"] = float(np.linalg.norm(E @ np.asarray(x0, dtype=float)))
        rows.append(row)
    return pd.DataFrame(rows)


def transient_peak(envelope: pd.DataFrame, column: str = "growth") -> dict[str, float]:
    """Peak amplification and when it happens, from a ``transient_envelope``."""
    idx = int(envelope[column].idxmax())
    return {
        "peak_amplification": float(envelope.loc[idx, column]),
        "time_of_peak": float(envelope.loc[idx, "time"]),
        "amplifies": bool(envelope[column].max() > 1.0 + 1e-9),
    }


def resolvent_norm_grid(
    J: np.ndarray,
    real_range: tuple[float, float] = (-2.0, 1.0),
    imag_range: tuple[float, float] = (-2.0, 2.0),
    resolution: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid of ``1 / sigma_min(z I - J)`` for pseudospectrum contour plots.

    Returns ``(real_axis, imag_axis, resolvent_norm)``. Contours of this surface
    at level ``1/epsilon`` are the epsilon-pseudospectrum: the set of eigenvalues
    reachable by a perturbation of norm ``epsilon``. A pseudospectrum that
    bulges across the imaginary axis while the true spectrum sits safely to its
    left is the geometric statement of the same fact that the abscissa gap
    reports numerically.
    """
    J = np.asarray(J, dtype=float)
    n = J.shape[0]
    re = np.linspace(real_range[0], real_range[1], resolution)
    im = np.linspace(imag_range[0], imag_range[1], resolution)
    out = np.zeros((len(im), len(re)))
    eye = np.eye(n)
    for a, y in enumerate(im):
        for b, x in enumerate(re):
            z = complex(x, y)
            smin = np.linalg.svd(z * eye - J, compute_uv=False)[-1]
            out[a, b] = 1.0 / smin if smin > 0 else np.inf
    return re, im, out


def pseudospectral_abscissa(
    J: np.ndarray,
    epsilon: float = 0.01,
    resolution: int = 61,
    coarse_steps: int = 40,
    refine_steps: int = 25,
) -> float:
    """Rightmost real part of the epsilon-pseudospectrum.

    Computed as the largest ``x`` for which some ``z = x + iy`` satisfies
    ``sigma_min(z I - J) <= epsilon``. For each ``y`` the search scans inward
    from a point guaranteed to lie outside the set (``||J||_2 + epsilon`` bounds
    the pseudospectrum) and then bisects the bracket it finds. Scanning from
    outside matters: ``sigma_min`` is not monotone in ``x``, so a bisection
    seeded at the spectral abscissa can converge to a point that is not in the
    set at all.

    The bound ``(alpha_eps - alpha) / epsilon`` is the Kreiss constant, which in
    turn lower bounds the peak of ``||exp(J t)||``.
    """
    J = np.asarray(J, dtype=float)
    n = J.shape[0]
    eye = np.eye(n)
    eigvals = np.linalg.eigvals(J)
    norm = float(np.linalg.norm(J, ord=2))

    def sigma_min(x: float, y: float) -> float:
        return float(np.linalg.svd(complex(x, y) * eye - J, compute_uv=False)[-1])

    y_max = max(float(np.max(np.abs(eigvals.imag))) * 1.5, norm * 0.25, 1.0)
    x_right = norm + epsilon                                   # certainly outside
    x_left = float(np.min(eigvals.real)) - epsilon              # certainly inside, at some y
    best = -np.inf
    for y in np.linspace(-y_max, y_max, resolution):
        xs = np.linspace(x_right, x_left, coarse_steps)
        step = xs[0] - xs[1]
        inside = None
        for x in xs:
            if sigma_min(x, y) <= epsilon:
                inside = float(x)
                break
        if inside is None:
            continue
        lo, hi = inside, min(inside + step, x_right)             # lo inside, hi outside
        for _ in range(refine_steps):
            mid = 0.5 * (lo + hi)
            if sigma_min(mid, y) <= epsilon:
                lo = mid
            else:
                hi = mid
        best = max(best, lo)
    if not np.isfinite(best):
        return float(np.max(eigvals.real))
    return float(best)


def kreiss_constant(J: np.ndarray, epsilons: Iterable[float] = (0.5, 0.2, 0.1, 0.05)) -> pd.DataFrame:
    """Kreiss lower bound on peak transient growth, across perturbation sizes.

    ``K = sup_eps (alpha_eps - alpha_max_stable) / eps``. Any value above 1 is a
    certificate that the network must amplify, independent of simulation.
    """
    rows = []
    for eps in epsilons:
        alpha_eps = pseudospectral_abscissa(J, epsilon=float(eps))
        rows.append(
            {
                "epsilon": float(eps),
                "pseudospectral_abscissa": alpha_eps,
                "kreiss_lower_bound": float(alpha_eps / eps) if alpha_eps > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Steady-state response
# --------------------------------------------------------------------------


def steady_state_response(J: np.ndarray) -> np.ndarray:
    """Return ``-J^{-1}``, the DC gain from constant input to steady state.

    For a stable Jacobian, ``dx/dt = J x + u`` settles at ``x = -J^{-1} u``.
    This is the continuous-time counterpart of the ``(I - M)^{-1}`` resolvent /
    Neumann-series machinery in ``inhib_modulation``: same object, expressed in
    the coordinates where stability is a half-plane rather than a disk.
    """
    J = np.asarray(J, dtype=float)
    return -np.linalg.inv(J)


def steady_state_gain(J: np.ndarray) -> dict[str, float]:
    """Scalar summaries of the DC gain matrix ``-J^{-1}``."""
    R = steady_state_response(J)
    singular_values = np.linalg.svd(R, compute_uv=False)
    return {
        "dc_gain_spectral_norm": float(singular_values[0]),
        "dc_gain_frobenius_norm": float(np.linalg.norm(R, ord="fro")),
        "dc_gain_mean_abs": float(np.mean(np.abs(R))),
        "dc_gain_condition_number": float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else float("inf"),
    }


# --------------------------------------------------------------------------
# Ablations
# --------------------------------------------------------------------------
#
# Every ablation acts on the *weight* matrix W and the Jacobian is rebuilt from
# the result, so the leak term is never removed and an ablated network is still
# a legitimate Jacobian. Ablations are not renormalized by default: renormalizing
# after a cut would rescale the network back to the original gain and hide the
# very effect being measured. Pass renormalize=True only to answer the separate,
# narrower question of whether a cut changes network *structure* once overall
# gain is matched.


def ablate_nodes(W: np.ndarray, indices: Iterable[int]) -> np.ndarray:
    """Silence cell types: zero their incoming and outgoing weights."""
    W = np.asarray(W, dtype=float).copy()
    idx = np.asarray(list(indices), dtype=int)
    if idx.size:
        W[idx, :] = 0.0
        W[:, idx] = 0.0
    return W


def ablate_edges(W: np.ndarray, edges: Iterable[tuple[int, int]], factor: float = 0.0) -> np.ndarray:
    """Remove (or scale by ``factor``) specific ``(post, pre)`` connections."""
    W = np.asarray(W, dtype=float).copy()
    for post, pre in edges:
        W[int(post), int(pre)] *= factor
    return W


def ablate_block(
    W: np.ndarray,
    masks: dict[str, np.ndarray],
    block: str,
    factor: float = 0.0,
) -> np.ndarray:
    """Remove or scale one E/I block.

    ``block`` is written source-to-target: ``"EI"`` is E onto I, which occupies
    rows ``I`` and columns ``E`` of a ``[post, pre]`` matrix. This is the naming
    collision flagged in ``docs/orientation_convention_proposal.md``, so the
    indexing is spelled out here rather than left to the caller.
    """
    W = np.asarray(W, dtype=float).copy()
    block = block.upper().strip()
    if len(block) != 2 or any(c not in "EI" for c in block):
        raise ValueError("block must be one of 'EE', 'EI', 'IE', 'II' (source then target).")
    source, target = block[0].lower(), block[1].lower()
    rows = np.flatnonzero(masks[target])
    cols = np.flatnonzero(masks[source])
    if rows.size and cols.size:
        W[np.ix_(rows, cols)] *= factor
    return W


def ablate_self_connections(W: np.ndarray, factor: float = 0.0) -> np.ndarray:
    """Scale the diagonal of ``W``. Note this is the autapse term, not the leak."""
    W = np.asarray(W, dtype=float).copy()
    np.fill_diagonal(W, np.diag(W) * factor)
    return W


def condition_summary(
    W: np.ndarray,
    condition: str,
    *,
    tau: float | np.ndarray = 1.0,
    leak: float = 1.0,
    renormalize: str | None = None,
    renormalize_target: float = 0.95,
    with_transient: bool = True,
    times: Iterable[float] | None = None,
) -> dict[str, float | str | bool]:
    """Build ``J`` from a (possibly ablated) ``W`` and summarize it in one row.

    This is the unit of every ablation table in this project: one condition in,
    one row of stability, non-normality, transient, and DC-gain numbers out.
    """
    W = np.asarray(W, dtype=float)
    info: dict[str, float | str | bool] = {"condition": condition}
    if renormalize is not None:
        W, norm_info = normalize_weights(W, method=renormalize, target=renormalize_target)
        info["renormalize_scale"] = norm_info["scale"]
    J = build_jacobian(W, tau=tau, leak=leak)
    info.update(stability_summary(J))
    info.update(non_normality_summary(J))
    if info.get("stable", False):
        info.update(steady_state_gain(J))
    if with_transient:
        envelope = transient_envelope(J, times=times)
        info.update(transient_peak(envelope))
    info["w_frobenius_norm"] = float(np.linalg.norm(W, ord="fro"))
    info["n_nonzero_weights"] = int(np.count_nonzero(W))
    return info


def ablation_sweep(
    W: np.ndarray,
    conditions: dict[str, np.ndarray],
    *,
    reference: str = "intact",
    delta_columns: Sequence[str] = (
        "spectral_abscissa",
        "numerical_abscissa",
        "abscissa_gap",
        "peak_amplification",
        "henrici_departure",
        "dc_gain_spectral_norm",
    ),
    **kwargs,
) -> pd.DataFrame:
    """Summarize a dict of ablated weight matrices into one comparison table.

    ``conditions`` maps a condition name to an already-ablated ``W``. The
    reference condition is added automatically if it is not present.
    """
    conditions = dict(conditions)
    conditions.setdefault(reference, np.asarray(W, dtype=float))
    rows = [condition_summary(matrix, name, **kwargs) for name, matrix in conditions.items()]
    table = pd.DataFrame(rows).set_index("condition")
    if reference in table.index:
        for column in delta_columns:
            if column in table.columns:
                table[f"delta_{column}"] = table[column] - table.loc[reference, column]
        ordered = [reference] + [name for name in table.index if name != reference]
        table = table.loc[ordered]
    return table


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------


def output_dir(notebook_name: str, root: str | Path = "outputs") -> Path:
    """Return (and create) ``outputs/<notebook_name>/``, the per-notebook folder."""
    path = Path(root) / notebook_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_table(table: pd.DataFrame, directory: str | Path, name: str, index: bool = True) -> Path:
    """Write a DataFrame to ``<directory>/<name>.csv`` and return the path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    table.to_csv(path, index=index)
    return path
