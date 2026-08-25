
"""
network_linear_systems_tools_script.py

Reusable tools for unmasked linear-system analyses of a directed connectivity matrix.

Matrix convention used throughout:
    raw CSV:
        rows    = outgoing/source nodes
        columns = receiver/target nodes

    state matrix:
        M[receiver, source]
        x[t+1] = M @ x[t]

For structural path analysis:
    (M^k)[receiver, source] is the total weighted contribution of all length-k walks
    from source to receiver under the column-vector convention.

For continuous transfer-function analysis:
    x_dot = A x + B u
    y     = C x
    G(s)  = C (sI - A)^(-1) B

The helper `make_stable_continuous_A` creates a stable continuous-time A from M by shifting
the spectrum left:
    A = M - shift * I
where shift > max(Re(lambda(M))).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from scipy import linalg

from schur_core_script import (
    load_mij_matrix,
    normalize_state_matrix,
    spectral_abscissa,
    spectral_radius,
)


@dataclass
class MatrixPrepResult:
    """Container for prepared matrix data."""
    M: np.ndarray
    raw: np.ndarray
    labels: list[str]
    df_source_receiver: pd.DataFrame
    normalization: str
    spectral_radius: float
    notes: str


def load_source_receiver_csv(
    csv_path: str,
    normalization: str = "none",
    target_spectral_radius: float = 1.0,
) -> MatrixPrepResult:
    """
    Load a source→receiver CSV and convert it into state-update convention.

    Parameters
    ----------
    csv_path:
        Path to CSV with first column as row index.
    normalization:
        "none", "column_l1", or "spectral_radius".
        "column_l1" normalizes each source column after transposition.
    target_spectral_radius:
        Used when normalization == "spectral_radius".

    Returns
    -------
    MatrixPrepResult
        Contains M with convention M[receiver, source].
    """
    df = load_mij_matrix(csv_path)
    raw = df.to_numpy(dtype=float)

    # Convert source rows / receiver columns into state matrix M[receiver, source].
    M_raw = raw.T.copy()
    labels = list(df.columns)

    normalization = normalization.lower().strip()
    if normalization == "none":
        M, _ = normalize_state_matrix(M_raw, method="none", target=target_spectral_radius)
        notes = "No normalization. Absolute weight scale is preserved."
    elif normalization in {"column_l1", "col_l1", "row_l1"}:
        M, _ = normalize_state_matrix(M_raw, method="column_l1", target=target_spectral_radius)
        normalization = "column_l1"
        notes = "Column L1 normalization after orientation correction. Each source column has abs-sum 1."
    elif normalization == "spectral_radius":
        M, _ = normalize_state_matrix(M_raw, method="spectral_radius", target=target_spectral_radius)
        notes = f"Spectral-radius normalization to rho={target_spectral_radius:g}."
    else:
        raise ValueError("normalization must be 'none', 'column_l1', or 'spectral_radius'.")

    return MatrixPrepResult(
        M=M,
        raw=raw,
        labels=labels,
        df_source_receiver=df,
        normalization=normalization,
        spectral_radius=spectral_radius(M),
        notes=notes,
    )

def top_entries(
    X: np.ndarray,
    labels: Sequence[str],
    n: int = 10,
    include_diagonal: bool = True,
    by_abs: bool = True,
) -> pd.DataFrame:
    """
    Return top entries of a square matrix as source→receiver contributions.

    X[receiver, source] is reported as source -> receiver.
    """
    X = np.asarray(X)
    rows = []
    for receiver_idx in range(X.shape[0]):
        for source_idx in range(X.shape[1]):
            if not include_diagonal and receiver_idx == source_idx:
                continue
            value = X[receiver_idx, source_idx]
            rows.append({
                "source": labels[source_idx],
                "receiver": labels[receiver_idx],
                "source_index": source_idx,
                "receiver_index": receiver_idx,
                "value": value,
                "abs_value": abs(value),
            })
    df = pd.DataFrame(rows)
    sort_col = "abs_value" if by_abs else "value"
    return df.sort_values(sort_col, ascending=False).head(n).reset_index(drop=True)


def matrix_power_series(
    M: np.ndarray,
    max_power: int = 6,
    alpha: float = 1.0,
    include_identity: bool = False,
) -> Dict[str, Any]:
    """
    Compute powers M^k and weighted cumulative path matrix.

    For x[t+1] = M x[t], (M^k)[receiver, source] aggregates all length-k walks
    from source to receiver.

    cumulative = sum_{k=start}^{max_power} alpha^k M^k
    where start is 0 if include_identity else 1.
    """
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    powers: Dict[int, np.ndarray] = {}

    current = np.eye(n)
    if include_identity:
        powers[0] = current.copy()

    cumulative = np.zeros_like(M, dtype=float)
    if include_identity:
        cumulative += current

    for k in range(1, max_power + 1):
        current = M @ current if k > 1 else M.copy()
        powers[k] = current.copy()
        cumulative += (alpha ** k) * current

    norms = pd.DataFrame({
        "k": list(powers.keys()),
        "fro_norm": [np.linalg.norm(powers[k], "fro") for k in powers],
        "spectral_norm": [np.linalg.norm(powers[k], 2) for k in powers],
        "max_abs_entry": [np.max(np.abs(powers[k])) for k in powers],
        "sum_abs_entries": [np.sum(np.abs(powers[k])) for k in powers],
    })

    return {
        "powers": powers,
        "cumulative": cumulative,
        "norms": norms,
        "max_power": max_power,
        "alpha": alpha,
        "include_identity": include_identity,
    }


def enumerate_power_top_paths(
    powers: Dict[int, np.ndarray],
    labels: Sequence[str],
    top_n: int = 10,
    include_diagonal: bool = False,
) -> Dict[int, pd.DataFrame]:
    """Return top source→receiver entries for each power."""
    out = {}
    for k, Mk in powers.items():
        out[k] = top_entries(Mk, labels, n=top_n, include_diagonal=include_diagonal)
    return out


def node_path_scores(X: np.ndarray, labels: Sequence[str]) -> pd.DataFrame:
    """
    Summarize outgoing and incoming weighted path scores for X.

    X[receiver, source]:
        outgoing score for source j = sum_i |X[i,j]|
        incoming score for receiver i = sum_j |X[i,j]|
    """
    Xabs = np.abs(np.asarray(X))
    outgoing = Xabs.sum(axis=0)
    incoming = Xabs.sum(axis=1)

    return pd.DataFrame({
        "cell": labels,
        "outgoing_path_score": outgoing,
        "incoming_path_score": incoming,
        "net_out_minus_in": outgoing - incoming,
    }).sort_values("outgoing_path_score", ascending=False).reset_index(drop=True)


def make_B(labels: Sequence[str], driver_nodes: Optional[Sequence[str | int]] = None) -> Tuple[np.ndarray, list[int]]:
    """
    Create an input matrix B.

    If driver_nodes is None, B=I.
    If a sequence is given, B selects those node channels.
    """
    n = len(labels)
    if driver_nodes is None:
        return np.eye(n), list(range(n))

    indices: list[int] = []
    label_to_idx = {str(label): i for i, label in enumerate(labels)}
    for item in driver_nodes:
        if isinstance(item, int):
            idx = item
        else:
            idx = label_to_idx[str(item)]
        if idx < 0 or idx >= n:
            raise IndexError(f"driver index {idx} out of range")
        indices.append(idx)

    B = np.eye(n)[:, indices]
    return B, indices


def make_C(labels: Sequence[str], output_nodes: Optional[Sequence[str | int]] = None) -> Tuple[np.ndarray, list[int]]:
    """
    Create an output matrix C.

    If output_nodes is None, C=I.
    If a sequence is given, C selects those node readouts.
    """
    n = len(labels)
    if output_nodes is None:
        return np.eye(n), list(range(n))

    indices: list[int] = []
    label_to_idx = {str(label): i for i, label in enumerate(labels)}
    for item in output_nodes:
        if isinstance(item, int):
            idx = item
        else:
            idx = label_to_idx[str(item)]
        if idx < 0 or idx >= n:
            raise IndexError(f"output index {idx} out of range")
        indices.append(idx)

    C = np.eye(n)[indices, :]
    return C, indices


def controllability_gramian_discrete(
    A: np.ndarray,
    B: np.ndarray,
    horizon: int = 25,
) -> Dict[str, Any]:
    """
    Finite-horizon discrete controllability Gramian:
        Wc(K) = sum_{k=0}^{K-1} A^k B B^H (A^H)^k

    Works whether or not A is asymptotically stable.
    """
    A = np.asarray(A, dtype=complex)
    B = np.asarray(B, dtype=complex)
    n = A.shape[0]

    W = np.zeros((n, n), dtype=complex)
    Ak = np.eye(n, dtype=complex)
    increments = []

    for k in range(horizon):
        term = Ak @ B @ B.conj().T @ Ak.conj().T
        W += term
        increments.append({
            "k": k,
            "term_trace": float(np.real(np.trace(term))),
            "term_fro_norm": float(np.linalg.norm(term, "fro")),
            "cumulative_trace": float(np.real(np.trace(W))),
            "cumulative_rank_est": int(np.linalg.matrix_rank(W, tol=1e-10)),
        })
        Ak = A @ Ak

    eigvals = np.linalg.eigvalsh((W + W.conj().T) / 2)
    return {
        "Wc": W,
        "increments": pd.DataFrame(increments),
        "eigvals": eigvals,
        "rank_est": int(np.linalg.matrix_rank(W, tol=1e-10)),
        "trace": float(np.real(np.trace(W))),
        "condition_number": safe_condition_number(W),
        "horizon": horizon,
    }


def controllability_gramian_continuous(
    A: np.ndarray,
    B: np.ndarray,
) -> Dict[str, Any]:
    """
    Infinite-horizon continuous controllability Gramian for stable A:
        A W + W A^H + B B^H = 0

    Requires max(real(eig(A))) < 0.
    """
    A = np.asarray(A, dtype=complex)
    B = np.asarray(B, dtype=complex)
    alpha = spectral_abscissa(A)
    if alpha >= 0:
        raise ValueError(f"Continuous-time A is not stable. spectral abscissa={alpha:.6g} >= 0.")
    W = linalg.solve_continuous_lyapunov(A, -(B @ B.conj().T))
    W = (W + W.conj().T) / 2
    eigvals = np.linalg.eigvalsh(W)
    return {
        "Wc": W,
        "eigvals": eigvals,
        "rank_est": int(np.linalg.matrix_rank(W, tol=1e-10)),
        "trace": float(np.real(np.trace(W))),
        "condition_number": safe_condition_number(W),
        "spectral_abscissa": alpha,
    }


def controllability_scores(Wc: np.ndarray, labels: Sequence[str]) -> pd.DataFrame:
    """
    Node-level controllability/state-diffusion scores from the Gramian.

    diagonal_energy:
        how much each state coordinate can be energized by the chosen inputs.
    """
    W = np.asarray(Wc)
    diag_energy = np.real(np.diag(W))
    row_spread = np.sum(np.abs(W), axis=1)
    return pd.DataFrame({
        "cell": labels,
        "diagonal_energy": diag_energy,
        "row_abs_spread": row_spread,
    }).sort_values("diagonal_energy", ascending=False).reset_index(drop=True)


def make_stable_continuous_A(
    M: np.ndarray,
    margin: float = 0.25,
    shift: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Convert a structural matrix M into a stable continuous-time A by left-shifting spectrum.

    A = M - shift * I

    If shift is None:
        shift = max(real(eig(M))) + margin
        with an additional floor so shift is positive.
    """
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    alpha_M = spectral_abscissa(M)
    if shift is None:
        shift = max(alpha_M + margin, margin)
    A = M - shift * np.eye(n)
    return {
        "A": A,
        "shift": float(shift),
        "spectral_abscissa_M": float(alpha_M),
        "spectral_abscissa_A": spectral_abscissa(A),
    }


def frequency_response(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    omega: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute continuous-time frequency response:
        G(iω) = C (iω I - A)^(-1) B

    Returns:
        gains_2norm[ω] = ||G(iω)||_2
        gains_fro[ω]   = ||G(iω)||_F
        responses       = list of response matrices
    """
    A = np.asarray(A, dtype=complex)
    B = np.asarray(B, dtype=complex)
    C = np.asarray(C, dtype=complex)
    n = A.shape[0]
    I = np.eye(n, dtype=complex)

    gains_2 = []
    gains_fro = []
    min_sigma = []
    responses = []

    for w in omega:
        Z = 1j * w * I - A
        X = linalg.solve(Z, B, assume_a="gen")
        G = C @ X
        responses.append(G)
        svals_G = np.linalg.svd(G, compute_uv=False)
        svals_Z = np.linalg.svd(Z, compute_uv=False)
        gains_2.append(float(svals_G[0] if len(svals_G) else 0.0))
        gains_fro.append(float(np.linalg.norm(G, "fro")))
        min_sigma.append(float(np.min(svals_Z)))

    return {
        "omega": np.asarray(omega),
        "gains_2norm": np.asarray(gains_2),
        "gains_fro": np.asarray(gains_fro),
        "min_sigma_sI_minus_A": np.asarray(min_sigma),
        "responses": responses,
    }


def top_frequency_channels(
    G: np.ndarray,
    input_labels: Sequence[str],
    output_labels: Sequence[str],
    n: int = 10,
) -> pd.DataFrame:
    """
    Return top transfer-function channels from input to output for a single G matrix.

    G[output, input].
    """
    G = np.asarray(G)
    rows = []
    for out_idx in range(G.shape[0]):
        for in_idx in range(G.shape[1]):
            val = G[out_idx, in_idx]
            rows.append({
                "input": input_labels[in_idx],
                "output": output_labels[out_idx],
                "value": val,
                "magnitude": abs(val),
                "phase_degrees": float(np.degrees(np.angle(val))),
            })
    return pd.DataFrame(rows).sort_values("magnitude", ascending=False).head(n).reset_index(drop=True)


def safe_condition_number(X: np.ndarray, tol: float = 1e-14) -> float:
    """Robust condition number based on singular values."""
    s = np.linalg.svd(np.asarray(X), compute_uv=False)
    if len(s) == 0:
        return float("nan")
    if s[-1] < tol:
        return float("inf")
    return float(s[0] / s[-1])


def compare_self_linear_systems(
    csv_path: str,
    *,
    target_spectral_radius: float = 0.95,
    max_path_length: int = 6,
    gramian_horizon: int = 25,
    omega: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Compare normalized path, controllability, and frequency analyses."""
    frame = pd.read_csv(csv_path, index_col=0)
    labels = list(frame.columns)
    source_frames = {"with_self": frame.copy()}
    no_self = frame.copy()
    np.fill_diagonal(no_self.values, 0.0)
    source_frames["no_self"] = no_self
    if omega is None:
        omega = np.logspace(-2, 2, 80)

    variants, rows, path_rows, control_rows, frequency_rows = {}, [], [], [], []
    for variant, source_frame in source_frames.items():
        raw_state = source_frame.to_numpy(dtype=float).T
        rho_raw = spectral_radius(raw_state)
        M = raw_state * (target_spectral_radius / rho_raw)
        paths = matrix_power_series(M, max_power=max_path_length)
        path_scores = node_path_scores(paths["cumulative"], labels).head(20).copy()
        path_scores.insert(0, "variant", variant)
        path_rows.append(path_scores)

        B, _ = make_B(labels)
        gramian = controllability_gramian_discrete(M, B, horizon=gramian_horizon)
        control = controllability_scores(gramian["Wc"], labels).head(20).copy()
        control.insert(0, "variant", variant)
        control_rows.append(control)

        stable = make_stable_continuous_A(M)
        C, _ = make_C(labels)
        response = frequency_response(stable["A"], B, C, omega)
        peak_index = int(np.argmax(response["gains_2norm"]))
        frequency_rows.append(
            pd.DataFrame(
                {
                    "variant": variant,
                    "omega": response["omega"],
                    "gain_2norm": response["gains_2norm"],
                    "gain_fro": response["gains_fro"],
                }
            )
        )
        rows.append(
            {
                "variant": variant,
                "raw_spectral_radius": rho_raw,
                "achieved_spectral_radius": spectral_radius(M),
                "matrix_fro_norm": float(np.linalg.norm(M, "fro")),
                "non_normality": float(np.linalg.norm(M @ M.T - M.T @ M, "fro")),
                "cumulative_path_fro_norm": float(np.linalg.norm(paths["cumulative"], "fro")),
                "gramian_trace": gramian["trace"],
                "gramian_condition": gramian["condition_number"],
                "peak_frequency": float(response["omega"][peak_index]),
                "peak_frequency_gain": float(response["gains_2norm"][peak_index]),
            }
        )
        variants[variant] = {
            "M": M,
            "source_frame": source_frame,
            "paths": paths,
            "gramian": gramian,
            "stable_continuous": stable,
            "frequency_response": response,
        }
    return {
        "variants": variants,
        "summary": pd.DataFrame(rows),
        "top_path_scores": pd.concat(path_rows, ignore_index=True),
        "top_controllability_scores": pd.concat(control_rows, ignore_index=True),
        "frequency_curves": pd.concat(frequency_rows, ignore_index=True),
    }
