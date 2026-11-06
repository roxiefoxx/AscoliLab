"""Kron reduction of the 85-type operator onto the 72-type subset.

The question
------------
An earlier parameterisation of the circuit had 72 neuron types. The current
operator has 85. The 13 types the 72 lacks are not "zero"; they are cells that
receive input from, and send output to, the other 72. Leaving them out of a
model (truncation) is not the same as accounting for them and then hiding them
(reduction). This module computes both and measures the difference, so the
85 -> 72 case can calibrate how much the 37 types still missing from the
85-type operator (relative to the 122-type Hippocampome circuit) could matter.

Model and conventions
---------------------
Everything follows ``jacobian_analysis``:

    tau dx/dt = -x + W x + u,    J = -I + W,    W = M[post, pre] / rho * 0.95

``W`` is loaded and orientation-checked by ``jacobian_core.load_jacobian_data``
(the Dale's-law sign oracle) and normalised once, on the full 85-type matrix.
Reductions are taken at that same scale; renormalising a reduced operator to
rho = 0.95 is offered separately, because it hides the stability shift that is
one of the main effects being measured.

Partition the types into kept ``K`` (the 72) and eliminated ``R`` (the 13).

Truncation (what a 72-type model *is*)::

    W_trunc = W_KK

Static Kron reduction (Schur complement of ``I - W`` onto ``K``)::

    W_kron = W_KK + W_KR (I - W_RR)^-1 W_RK

It is exact at steady state: ``(I - W_kron)^-1 == [(I - W)^-1]_KK``. The
correction is the sum over every path that leaves ``K``, wanders through ``R``
any number of steps, and returns: ``W_KR (I + W_RR + W_RR^2 + ...) W_RK``.

Dynamic (frequency-dependent) Kron reduction::

    W_kron(s) = W_KK + W_KR ((1 + s) I - W_RR)^-1 W_RK

exact at every Laplace frequency ``s``. The static version is ``s = 0``: it
treats the hidden cells as if they settled instantly.

The same Schur complement already appears in this repository as the
"inhibitory Schur complement" ``M_eff_E`` (eliminate I, keep E). Kron reduction
is that operation with an arbitrary choice of which nodes to hide.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import expm

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "jacobian_analysis"))

import jacobian_core as jc  # noqa: E402

MATRIX = REPO / "jacobian_analysis" / "matrices" / "mij_matrix.csv"
NETLIST = REPO / "jacobian_analysis" / "matrices" / "mij_netlist.csv"
TYPES_72 = HERE / "data" / "types_72.csv"
Q_VECTOR = REPO / "non-normal_matrices" / "data" / "ml_qi_vector.csv"
ORDER_85 = REPO / "tarjan_reorder" / "optimal_order_85.csv"

GAIN = 0.95
TIMES = np.round(np.linspace(0.0, 10.0, 101), 10)
PEAK_TIMES = np.round(np.linspace(0.0, 4.0, 81), 10)  # full-85 peak is at 1.2 tau


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


@dataclass
class Partition:
    """The normalised 85-type operator and the 72 / 13 split."""

    W: np.ndarray            # [post, pre], rho = GAIN
    W_raw: np.ndarray        # [post, pre], raw units
    scale: float
    labels: np.ndarray
    ei: np.ndarray           # 'e' / 'i' per type
    keep: np.ndarray         # indices of the 72
    elim: np.ndarray         # indices of the 13
    kappa: np.ndarray        # postsynaptic gain per type (Hz/pA)
    q: np.ndarray            # census per type
    order_position: np.ndarray  # position of each type in the certified 85-type order
    orientation_check: dict

    @property
    def n(self) -> int:
        return len(self.labels)

    def sub(self, M: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return M[np.ix_(rows, cols)]

    def frame(self, M: np.ndarray, idx: np.ndarray | None = None) -> pd.DataFrame:
        names = self.labels if idx is None else self.labels[idx]
        return pd.DataFrame(M, index=names, columns=names)


def region_of(label: str) -> str:
    head = label.split()[0]
    return "CA3" if head == "CA3c" else head


def load_partition(gain: float = GAIN, types_72: Path = TYPES_72) -> Partition:
    d = jc.load_jacobian_data(MATRIX, NETLIST)
    W, info = jc.normalize_weights(d.W_raw, "spectral_radius", gain)
    labels = np.array(d.labels)
    ei = d.ei.loc[labels].to_numpy()

    keep_names = pd.read_csv(types_72)["neuron_type"].str.strip()
    unknown = sorted(set(keep_names) - set(labels))
    if unknown:
        raise ValueError(f"72-type list has names not in the 85-type matrix: {unknown}")
    if len(keep_names) != 72 or keep_names.duplicated().any():
        raise ValueError("types_72.csv must list 72 distinct types.")
    is_keep = np.isin(labels, keep_names)
    keep = np.flatnonzero(is_keep)
    elim = np.flatnonzero(~is_keep)

    kappa = d.netlist.groupby("post_neuron")["kappa_j"].first().reindex(labels).to_numpy(float)
    q = pd.read_csv(Q_VECTOR).set_index("neuron_type")["q_i"].reindex(labels).to_numpy(float)
    if np.isnan(kappa).any() or np.isnan(q).any():
        raise ValueError("Missing kappa_j or q_i for some types.")

    order = pd.read_csv(ORDER_85).set_index("cell_type")["position"].reindex(labels)
    if order.isna().any():
        raise ValueError("Certified order does not cover all 85 types.")

    return Partition(
        W=W, W_raw=d.W_raw, scale=float(info["scale"]), labels=labels, ei=ei,
        keep=keep, elim=elim, kappa=kappa, q=q,
        order_position=order.to_numpy(int), orientation_check=d.orientation_check,
    )


# --------------------------------------------------------------------------
# Reductions
# --------------------------------------------------------------------------


def complement(n: int, elim: Iterable[int]) -> np.ndarray:
    return np.setdiff1d(np.arange(n), np.asarray(list(elim), dtype=int))


def kron_correction(W: np.ndarray, keep: np.ndarray, elim: np.ndarray, s: complex = 0.0, leak: float = 1.0) -> np.ndarray:
    """``W_KR ((leak + s) I - W_RR)^-1 W_RK``: every K -> R... -> K path, summed."""
    W_KR = W[np.ix_(keep, elim)]
    W_RK = W[np.ix_(elim, keep)]
    W_RR = W[np.ix_(elim, elim)]
    A = (leak + s) * np.eye(len(elim)) - W_RR
    return W_KR @ np.linalg.solve(A, W_RK)


def kron_reduce(W: np.ndarray, keep: np.ndarray, elim: np.ndarray, s: complex = 0.0) -> np.ndarray:
    """Kron-reduced operator on ``keep``. ``s = 0`` is the static reduction."""
    return W[np.ix_(keep, keep)] + kron_correction(W, keep, elim, s)


def truncate(W: np.ndarray, keep: np.ndarray) -> np.ndarray:
    return W[np.ix_(keep, keep)].copy()


def transfer(W: np.ndarray, s: complex = 0.0) -> np.ndarray:
    """``((1 + s) I - W)^-1``: input u -> response x of ``dx/dt = -x + Wx + u``."""
    return np.linalg.inv((1.0 + s) * np.eye(len(W)) - W)


def rel_err(A: np.ndarray, B: np.ndarray) -> float:
    """``||A - B||_F / ||B||_F`` with ``B`` the reference."""
    return float(np.linalg.norm(A - B) / np.linalg.norm(B))


# --------------------------------------------------------------------------
# Operator summaries
# --------------------------------------------------------------------------


def peak_amplification(W: np.ndarray, times: np.ndarray = PEAK_TIMES) -> tuple[float, float]:
    J = W - np.eye(len(W))
    growth = np.array([np.linalg.norm(expm(J * t), 2) for t in times])
    k = int(np.argmax(growth))
    return float(growth[k]), float(times[k])


def block_norms(W: np.ndarray, ei: np.ndarray) -> dict[str, float]:
    """Frobenius norm of each E/I block, named in the arrow sense (source_to_target)."""
    out = {}
    for src in "ei":
        for tgt in "ei":
            out[f"{src.upper()}_to_{tgt.upper()}"] = float(np.linalg.norm(W[np.ix_(ei == tgt, ei == src)]))
    return out


def forward_fraction(W: np.ndarray, position: np.ndarray) -> float:
    """Share of off-diagonal |weight| running from an earlier to a later position."""
    A = np.abs(W).copy()
    np.fill_diagonal(A, 0.0)
    post_pos = position[:, None]
    pre_pos = position[None, :]
    return float(A[post_pos > pre_pos].sum() / A.sum())


def dale_violations(W: np.ndarray, ei: np.ndarray, rel_tol: float = 1e-12) -> dict[str, float]:
    """Entries whose sign disagrees with the source's class.

    ``rel_tol`` is relative to ``max |W|``. Reported as counts and as a share of
    total off-diagonal |weight|, which is the meaningful size.
    """
    sign = np.where(ei == "e", 1.0, -1.0)[None, :]
    floor = rel_tol * np.abs(W).max()
    wrong = (W * sign < 0) & (np.abs(W) > floor)
    mixed_cols = int(sum(((W[:, j] > floor).any() and (W[:, j] < -floor).any()) for j in range(W.shape[1])))
    return {
        "dale_wrong_sign_entries": int(wrong.sum()),
        "dale_wrong_sign_weight_share": float(np.abs(W[wrong]).sum() / np.abs(W).sum()),
        "dale_mixed_sign_columns": mixed_cols,
    }


def operator_summary(W: np.ndarray, ei: np.ndarray, position: np.ndarray, name: str) -> dict:
    J = W - np.eye(len(W))
    eig = np.linalg.eigvals(W)
    dom = eig[np.argmax(eig.real)]
    stab = jc.stability_summary(J)
    nn = jc.non_normality_summary(J)
    peak, tpk = peak_amplification(W)
    blocks = block_norms(W, ei)
    floor = 1e-12 * np.abs(W).max()
    out = {
        "operator": name,
        "n_types": len(W),
        "n_edges": int((np.abs(W) > floor).sum()),
        "rho_W": float(np.max(np.abs(eig))),
        "dominant_eigenvalue_W": float(dom.real) if abs(dom.imag) < 1e-12 else complex(dom),
        "spectral_abscissa_J": stab["spectral_abscissa"],
        "stability_margin": stab["stability_margin"],
        "numerical_abscissa_J": stab["numerical_abscissa"],
        "henrici_relative": nn["henrici_departure_relative"],
        "peak_amplification": peak,
        "time_of_peak": tpk,
        "E_to_I_over_I_to_E": blocks["E_to_I"] / blocks["I_to_E"],
        **{f"block_{k}": v for k, v in blocks.items()},
        "forward_weight_fraction": forward_fraction(W, position),
    }
    out.update(dale_violations(W, ei))
    return out


def renormalise(W: np.ndarray, gain: float = GAIN) -> np.ndarray:
    return W * gain / np.max(np.abs(np.linalg.eigvals(W)))


# --------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------


def dominant_pair(W: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Dominant (largest real part) eigenvalue with right and left eigenvectors."""
    vals, right = np.linalg.eig(W)
    k = int(np.argmax(vals.real))
    lvals, left = np.linalg.eig(W.T)
    kl = int(np.argmin(np.abs(lvals - vals[k])))
    return complex(vals[k]), right[:, k], left[:, kl]


def single_node_table(P: Partition) -> pd.DataFrame:
    """Leave-one-out from the full 85 for each of the 13, plus a first-order screen.

    ``rho_if_deleted`` is the exact spectral radius after truncating that one type.
    ``first_order_shift`` predicts the change in the dominant eigenvalue when the
    type's rank-1 Kron correction ``W_Kr W_rK / (1 - W_rr)`` is added back to the
    truncated 84-type operator: ``u' Delta v / u' v`` with ``u, v`` the left and
    right dominant eigenvectors. It needs only the hidden type's in- and
    out-weights, so it is a screen that can be run on imputed weights.
    """
    W = P.W
    base_rho = float(np.max(np.abs(np.linalg.eigvals(W))))
    rows = []
    for r in P.elim:
        keep = complement(P.n, [r])
        Wt = truncate(W, keep)
        D = kron_correction(W, keep, np.array([r]))
        lam, v, u = dominant_pair(Wt)
        pred = complex(u @ D @ v / (u @ v))
        lam_t = float(np.max(np.abs(np.linalg.eigvals(Wt))))
        rows.append({
            "neuron_type": P.labels[r],
            "ei": P.ei[r],
            "region": region_of(P.labels[r]),
            "in_strength": float(np.abs(W[r, keep]).sum()),     # row r: inputs to r
            "out_strength": float(np.abs(W[keep, r]).sum()),    # column r: outputs of r
            "self_weight": float(W[r, r]),
            "n_in": int((W[r, keep] != 0).sum()),
            "n_out": int((W[keep, r] != 0).sum()),
            "kron_correction_fro": float(np.linalg.norm(D)),
            "kron_correction_rel": float(np.linalg.norm(D) / np.linalg.norm(Wt)),
            "rho_if_deleted": lam_t,
            "rho_shift_if_deleted": lam_t - base_rho,
            "first_order_restore_shift": float(pred.real),
            "exact_restore_shift": base_rho - lam_t,
        })
    return pd.DataFrame(rows).sort_values("rho_shift_if_deleted", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Imputation (what can be done for types with no parameters)
# --------------------------------------------------------------------------


IMPUTATIONS = ("truncation", "topology_blockmedian", "topology_q_kappa", "oracle")


def impute_hidden(P: Partition, elim: np.ndarray, mode: str) -> np.ndarray:
    """Replace every weight touching ``elim`` with an imputed value.

    Pretends the hidden types' synaptic parameters are unknown but their
    *topology* (which edges exist) and E/I class are known, as is the case for
    the Hippocampome types outside the 85. Signs always follow Dale's law.

    ``topology_blockmedian``
        |w| = median |W| of known (kept-kept) edges in the same
        source-class x target-class block.
    ``topology_q_kappa``
        Uses the factorisation ``W[post, pre] = kappa_post * B * q_pre``.
        The base synaptic term ``B`` is imputed by the block median of known
        edges, while census ``q`` and gain ``kappa`` of the hidden types are
        taken as known. Tests whether census + excitability are enough.
    """
    W = P.W
    if mode == "oracle":
        return W.copy()
    keep = complement(P.n, elim)
    is_keep = np.zeros(P.n, bool)
    is_keep[keep] = True
    exists = W != 0
    known = exists & is_keep[:, None] & is_keep[None, :]
    hidden = exists & ~known
    sign = np.where(P.ei == "e", 1.0, -1.0)

    if mode == "topology_blockmedian":
        base = np.abs(W)
        scale_post = np.ones(P.n)
        scale_pre = np.ones(P.n)
    elif mode == "topology_q_kappa":
        base = np.abs(W) / np.outer(P.kappa, P.q)
        scale_post = P.kappa
        scale_pre = P.q
    else:
        raise ValueError(f"unknown imputation mode {mode!r}")

    medians = {}
    for src in "ei":
        for tgt in "ei":
            m = known & (P.ei == tgt)[:, None] & (P.ei == src)[None, :]
            medians[(src, tgt)] = float(np.median(base[m]))

    out = W.copy()
    for i, j in zip(*np.nonzero(hidden)):
        out[i, j] = sign[j] * medians[(P.ei[j], P.ei[i])] * scale_post[i] * scale_pre[j]
    return out


def reduced_operator(P: Partition, elim: np.ndarray, mode: str) -> np.ndarray:
    keep = complement(P.n, elim)
    if mode == "truncation":
        return truncate(P.W, keep)
    return kron_reduce(impute_hidden(P, elim, mode), keep, elim)


# --------------------------------------------------------------------------
# Null ensemble
# --------------------------------------------------------------------------


def matched_random_sets(P: Partition, n_sets: int, seed: int = 0) -> list[np.ndarray]:
    """Random hidden sets with the same E/I make-up as the real 13."""
    rng = np.random.default_rng(seed)
    E = np.flatnonzero(P.ei == "e")
    I = np.flatnonzero(P.ei == "i")
    n_e = int((P.ei[P.elim] == "e").sum())
    n_i = int((P.ei[P.elim] == "i").sum())
    return [np.sort(np.r_[rng.choice(E, n_e, replace=False), rng.choice(I, n_i, replace=False)]) for _ in range(n_sets)]


def hidden_set_metrics(P: Partition, elim: np.ndarray, G_full: np.ndarray, with_peak: bool = True) -> dict:
    keep = complement(P.n, elim)
    G_ref = G_full[np.ix_(keep, keep)]
    Wt = truncate(P.W, keep)
    D = kron_correction(P.W, keep, elim)
    out = {
        "rho_truncated": float(np.max(np.abs(np.linalg.eigvals(Wt)))),
        "kron_correction_rel": float(np.linalg.norm(D) / np.linalg.norm(Wt)),
    }
    for mode in IMPUTATIONS[:-1]:
        Wr = Wt if mode == "truncation" else reduced_operator(P, elim, mode)
        out[f"ss_err_{mode}"] = rel_err(transfer(Wr), G_ref)
        out[f"rho_{mode}"] = float(np.max(np.abs(np.linalg.eigvals(Wr))))
    if with_peak:
        out["peak_truncated"] = peak_amplification(Wt)[0]
    return out


def save(table: pd.DataFrame, name: str, index: bool = False) -> Path:
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    path = out / f"{name}.csv"
    table.to_csv(path, index=index)
    return path
