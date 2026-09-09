#!/usr/bin/env python3
"""Reproducible non-normal dynamics analysis for a signed directed matrix.

Convention: A[i, j] is the effect of source node j on target node i.
The main dynamical model is dx/dt = J x, J = -I + g A_tilde.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import eig, eigvals, norm, svd
from scipy.linalg import expm, expm_frechet, solve_continuous_lyapunov
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata


SEED = 20260901
BLUE, ORANGE, GOLD, PINK, INK, GREY = "#326891", "#E1812C", "#D4A72C", "#C8556D", "#222222", "#8B8B8B"


def load_matrix(path: str | Path) -> tuple[np.ndarray, list[str]]:
    frame = pd.read_csv(path, index_col=0)
    if frame.shape[0] != frame.shape[1]:
        raise ValueError(f"Matrix must be square; found {frame.shape}.")
    if list(frame.index.astype(str)) != list(frame.columns.astype(str)):
        raise ValueError("Row and column labels differ or are ordered differently.")
    matrix = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Matrix contains NaN or infinite values.")
    return matrix, list(frame.index.astype(str))


def matrix_validation(A: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    nz = A[A != 0]
    q = np.quantile(nz, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    rows = {
        "n_rows": A.shape[0], "n_columns": A.shape[1], "nonzero_count": np.count_nonzero(A),
        "nonzero_fraction": np.count_nonzero(A) / A.size, "positive_count": np.sum(A > 0),
        "negative_count": np.sum(A < 0), "zero_count": np.sum(A == 0),
        "minimum_weight": A.min(), "maximum_weight": A.max(), "nonzero_mean": nz.mean(),
        "nonzero_median": np.median(nz), "nonzero_std": nz.std(ddof=0),
        "nonzero_diagonal": np.count_nonzero(np.diag(A)), "contains_nan": np.isnan(A).any(),
        "contains_infinity": np.isinf(A).any(), "all_zero_rows": np.sum(np.all(A == 0, axis=1)),
        "all_zero_columns": np.sum(np.all(A == 0, axis=0)),
    }
    quantiles = pd.DataFrame({"quantile": [0, .01, .05, .25, .5, .75, .95, .99, 1], "nonzero_weight": q})
    return pd.DataFrame({"metric": rows.keys(), "value": rows.values()}), quantiles


def node_strengths(A: np.ndarray, labels: list[str]) -> pd.DataFrame:
    pos, neg = np.maximum(A, 0), np.minimum(A, 0)
    out = pd.DataFrame({
        "node": labels,
        "absolute_in_strength": np.abs(A).sum(axis=1),
        "absolute_out_strength": np.abs(A).sum(axis=0),
        "signed_in_strength": A.sum(axis=1), "signed_out_strength": A.sum(axis=0),
        "positive_in_strength": pos.sum(axis=1), "positive_out_strength": pos.sum(axis=0),
        "negative_in_strength": neg.sum(axis=1), "negative_out_strength": neg.sum(axis=0),
        "in_degree": np.count_nonzero(A, axis=1), "out_degree": np.count_nonzero(A, axis=0),
    })
    return out


def concentration(A: np.ndarray, labels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    absw = np.abs(A)
    total = absw.sum()
    node = node_strengths(A, labels)
    node["out_weight_share"] = node.absolute_out_strength / total
    node["in_weight_share"] = node.absolute_in_strength / total
    ii, jj = np.nonzero(A)
    edges = pd.DataFrame({"target": np.asarray(labels)[ii], "source": np.asarray(labels)[jj], "weight": A[ii, jj]})
    edges["absolute_weight"] = edges.weight.abs()
    edges["weight_share"] = edges.absolute_weight / total
    edges = edges.sort_values("absolute_weight", ascending=False).reset_index(drop=True)
    edges["cumulative_weight_share"] = edges.weight_share.cumsum()
    return node.sort_values("absolute_out_strength", ascending=False), edges


def spectral_metrics(A: np.ndarray, name: str = "matrix") -> dict[str, float | str]:
    nrm_f = norm(A, "fro")
    vals, vecs = eig(A)
    S = (A + A.T) / 2
    asym = norm(A - A.T, "fro") / nrm_f if nrm_f else 0.0
    comm = A @ A.T - A.T @ A
    eta_c = norm(comm, "fro") / nrm_f**2 if nrm_f else 0.0
    h_sq = max(0.0, nrm_f**2 - float(np.sum(np.abs(vals) ** 2)))
    d_h = math.sqrt(h_sq)
    cond_v = float(np.linalg.cond(vecs))
    return {
        "matrix": name, "frobenius_norm": nrm_f, "spectral_norm": norm(A, 2),
        "spectral_radius": np.max(np.abs(vals)), "spectral_abscissa": np.max(vals.real),
        "numerical_abscissa": np.linalg.eigvalsh(S)[-1], "reactivity_gap": np.linalg.eigvalsh(S)[-1] - np.max(vals.real),
        "non_normal_reactivity_ratio": (np.linalg.eigvalsh(S)[-1] / np.max(vals.real)) if np.max(vals.real) != 0 else np.nan,
        "asymmetry_index": asym, "commutator_index": eta_c, "henrici_departure": d_h,
        "henrici_index": d_h / nrm_f if nrm_f else 0.0, "eigenvector_condition": cond_v,
        "diagonalization_warning": "yes" if (not np.isfinite(cond_v) or cond_v > 1e10) else "no",
    }


def normalize_versions(A: np.ndarray) -> dict[str, np.ndarray]:
    rho = np.max(np.abs(eigvals(A)))
    n2 = norm(A, 2)
    ninf = norm(A, np.inf)
    return {"raw": A.copy(), "spectral_radius": A / rho, "spectral_norm": A / n2, "max_row_sum": A / ninf}


def thresholds(A: np.ndarray, gamma: float = 1.0) -> dict[str, float]:
    alpha = float(np.max(eigvals(A).real))
    omega = float(np.linalg.eigvalsh((A + A.T) / 2)[-1])
    gs = gamma / alpha if alpha > 0 else np.inf
    gr = gamma / omega if omega > 0 else np.inf
    return {"alpha_A": alpha, "omega_A": omega, "g_react": gr, "g_stab": gs,
            "reactive_to_stable_ratio": gr / gs if np.isfinite(gr) and np.isfinite(gs) else np.nan,
            "stable_reactive_width": max(0.0, gs - gr) if np.isfinite(gs) and np.isfinite(gr) else np.nan}


def stable_time_grid(J: np.ndarray, n: int = 70) -> np.ndarray:
    alpha = float(np.max(eigvals(J).real))
    if alpha >= 0:
        raise ValueError("Transient gain requires a stable J.")
    decay = max(-alpha, 1e-4)
    horizon = min(500.0, max(8.0, 10.0 / decay))
    positive = np.geomspace(max(1e-5, horizon * 1e-6), horizon, n - 1)
    return np.r_[0.0, positive]


def gain_at(J: np.ndarray, t: float) -> float:
    return float(svd(expm(J * t), compute_uv=False)[0] ** 2)


def transient_gain(J: np.ndarray, n_time: int = 70, refine: bool = True) -> dict[str, object]:
    times = stable_time_grid(J, n_time)
    gains = np.array([gain_at(J, t) for t in times])
    k = int(np.argmax(gains))
    t_star, g_star = float(times[k]), float(gains[k])
    if refine and 0 < k < len(times) - 1:
        result = minimize_scalar(lambda t: -gain_at(J, float(t)), bounds=(times[k-1], times[k+1]), method="bounded",
                                 options={"xatol": 1e-7})
        if result.success:
            t_star, g_star = float(result.x), float(-result.fun)
    P = expm(J * t_star)
    U, s, Vh = svd(P, full_matrices=False)
    return {"times": times, "gains": gains, "Gmax": g_star, "t_star": t_star,
            "v1": Vh[0].copy(), "u1": U[:, 0].copy(), "sigma1": float(s[0])}


def coupling_phase(A: np.ndarray, n: int = 241) -> tuple[pd.DataFrame, dict[str, float]]:
    th = thresholds(A)
    upper = 1.08 * th["g_stab"] if np.isfinite(th["g_stab"]) else 3.0
    g = np.linspace(0, upper, n)
    return pd.DataFrame({"g": g, "spectral_abscissa_J": -1 + g * th["alpha_A"],
                         "numerical_abscissa_J": -1 + g * th["omega_A"]}), th


def representative_couplings(th: dict[str, float]) -> list[tuple[str, float]]:
    gr, gs = th["g_react"], th["g_stab"]
    if np.isfinite(gr) and np.isfinite(gs) and gr < gs:
        return [("below_reactivity", .75 * gr), ("near_reactivity", .99 * gr),
                ("stable_reactive", .5 * (gr + gs)), ("near_stability", .95 * gs)]
    return [("low", .25 * gs), ("middle", .5 * gs), ("high", .8 * gs), ("near_stability", .95 * gs)]


def single_node_perturbations(J: np.ndarray, labels: list[str], times: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if times is None:
        times = stable_time_grid(J, 80)
    n = J.shape[0]
    peak = np.zeros(n); tpeak = np.zeros(n); integral = np.zeros(n); downstream = np.zeros((n, n))
    norms = np.zeros((len(times), n))
    for k, t in enumerate(times):
        P = expm(J * t)
        e = P * P
        norms[k] = e.sum(axis=0)
        downstream = np.maximum(downstream, np.abs(P))
    peak_idx = np.argmax(norms, axis=0)
    peak = norms[peak_idx, np.arange(n)]
    tpeak = times[peak_idx]
    # Observability Gramian: J^T Q + QJ = -I; e_i^T Q e_i is integrated energy.
    Q = solve_continuous_lyapunov(J.T, -np.eye(n))
    integral = np.diag(Q)
    result = pd.DataFrame({"node": labels, "single_node_peak_gain": peak, "time_to_peak": tpeak,
                           "integrated_energy": integral, "final_grid_energy": norms[-1]})
    result["perturbability_rank"] = rankdata(-result.single_node_peak_gain, method="min").astype(int)
    susceptibility = pd.DataFrame({"node": labels, "max_received_amplitude": downstream.max(axis=1),
                                   "sum_max_received_amplitude": downstream.sum(axis=1)})
    susceptibility["susceptibility_rank"] = rankdata(-susceptibility.sum_max_received_amplitude, method="min").astype(int)
    response = pd.DataFrame(downstream, index=labels, columns=labels)
    response.index.name = "response_target"
    return result.sort_values("perturbability_rank"), susceptibility.sort_values("susceptibility_rank"), response


def signed_interpolation(A: np.ndarray, beta_values: np.ndarray, n_time: int = 42) -> pd.DataFrame:
    positive, negative = np.maximum(A, 0), np.minimum(A, 0)
    rows = []
    for beta in beta_values:
        M = positive + beta * negative
        M = M / norm(M, 2)
        met, th = spectral_metrics(M), thresholds(M)
        g = .9 * th["g_stab"] if np.isfinite(th["g_stab"]) else 1.0
        G = transient_gain(-np.eye(len(M)) + g * M, n_time=n_time, refine=False)["Gmax"]
        rows.append({"beta": beta, **{k: met[k] for k in ["spectral_abscissa", "numerical_abscissa", "commutator_index", "henrici_index"]},
                     "g_react": th["g_react"], "g_stab": th["g_stab"], "Gmax_at_90pct_stability": G})
    return pd.DataFrame(rows)


def directionality_interpolation(A: np.ndarray, q_values: np.ndarray, n_time: int = 42) -> pd.DataFrame:
    S, K = (A + A.T) / 2, (A - A.T) / 2
    rows = []
    for q in q_values:
        M = S + q * K
        M = M / norm(M, 2)
        met, th = spectral_metrics(M), thresholds(M)
        g = .9 * th["g_stab"] if np.isfinite(th["g_stab"]) else 1.0
        G = transient_gain(-np.eye(len(M)) + g * M, n_time=n_time, refine=False)["Gmax"]
        rows.append({"q": q, **{k: met[k] for k in ["spectral_abscissa", "numerical_abscissa", "commutator_index", "henrici_index"]},
                     "g_react": th["g_react"], "g_stab": th["g_stab"], "Gmax_at_90pct_stability": G})
    return pd.DataFrame(rows)


def weight_shuffled_null(A: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mask = A != 0
    out = np.zeros_like(A)
    out[mask] = rng.permutation(A[mask])
    return out


def direction_randomized_null(A: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros_like(A)
    n = len(A)
    for i in range(n):
        if A[i, i] != 0:
            out[i, i] = A[i, i]
    for i in range(n):
        for j in range(i + 1, n):
            pair = [A[i, j], A[j, i]]
            if rng.random() < .5:
                pair.reverse()
            out[i, j], out[j, i] = pair
    return out


def null_models(A: np.ndarray, n_null: int = 50, n_gain: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    models = [("empirical", A), ("symmetric", (A + A.T) / 2)]
    models += [("weight_shuffled", weight_shuffled_null(A, rng)) for _ in range(n_null)]
    models += [("direction_randomized", direction_randomized_null(A, rng)) for _ in range(n_null)]
    counts = {"weight_shuffled": 0, "direction_randomized": 0}
    for idx, (kind, M0) in enumerate(models):
        M = M0 / norm(M0, 2)
        met, th = spectral_metrics(M), thresholds(M)
        do_gain = kind in {"empirical", "symmetric"} or counts.get(kind, n_gain) < n_gain
        if kind in counts:
            counts[kind] += 1
        G = np.nan
        if do_gain and np.isfinite(th["g_stab"]):
            J = -np.eye(len(M)) + .9 * th["g_stab"] * M
            G = transient_gain(J, n_time=34, refine=False)["Gmax"]
        rows.append({"model": kind, "replicate": idx, **{k: met[k] for k in ["commutator_index", "henrici_index", "spectral_abscissa", "numerical_abscissa"]},
                     "Gmax_at_90pct_stability": G})
    return pd.DataFrame(rows)


def empirical_null_statistics(nulls: pd.DataFrame) -> pd.DataFrame:
    empirical = nulls.loc[nulls.model == "empirical"].iloc[0]
    rows = []
    for kind in ["weight_shuffled", "direction_randomized"]:
        subset = nulls.loc[nulls.model == kind]
        for metric in ["commutator_index", "henrici_index", "spectral_abscissa", "numerical_abscissa", "Gmax_at_90pct_stability"]:
            vals = subset[metric].dropna().to_numpy()
            if not len(vals):
                continue
            x = float(empirical[metric])
            rows.append({"null_model": kind, "metric": metric, "empirical": x, "null_mean": vals.mean(),
                         "null_sd": vals.std(ddof=1), "standardized_effect": (x - vals.mean()) / vals.std(ddof=1) if vals.std(ddof=1) else np.nan,
                         "empirical_percentile": np.mean(vals <= x), "upper_tail_p": (1 + np.sum(vals >= x)) / (len(vals) + 1), "n": len(vals)})
    return pd.DataFrame(rows)


def lesion_analysis(A: np.ndarray, labels: list[str], g: float, baseline_gain: float, baseline_t: float) -> pd.DataFrame:
    n = len(A); rows = []
    for i, label in enumerate(labels):
        keep = np.arange(n) != i
        M = A[np.ix_(keep, keep)]
        met, th = spectral_metrics(M), thresholds(M)
        J = -np.eye(n - 1) + g * M
        if np.max(eigvals(J).real) < 0:
            # Screen at baseline peak and nearby times; refine the strongest lesions later in interpretation.
            ts = np.unique(np.clip(np.array([0, .5 * baseline_t, baseline_t, 1.5 * baseline_t, 2 * baseline_t]), 0, None))
            G = max(gain_at(J, t) for t in ts)
        else:
            G = np.inf
        rows.append({"node": label, "spectral_abscissa_A_lesion": met["spectral_abscissa"],
                     "numerical_abscissa_A_lesion": met["numerical_abscissa"], "g_react_lesion": th["g_react"],
                     "g_stab_lesion": th["g_stab"], "Gmax_screen_lesion": G, "delta_Gmax_screen": G - baseline_gain})
    return pd.DataFrame(rows).sort_values("delta_Gmax_screen")


def edge_sensitivity(A: np.ndarray, labels: list[str], g: float, t_star: float, top_frechet: int = 40) -> pd.DataFrame:
    vals, vl, vr = None, None, None
    # Dominant eigenvalue left/right vectors for alpha sensitivity.
    evals, right = eig(A)
    k = int(np.argmax(evals.real)); lam = evals[k]; r = right[:, k]
    evals_t, left_raw = eig(A.T)
    l = left_raw[:, int(np.argmin(np.abs(evals_t - lam)))]
    denom = np.vdot(l, r)
    S = (A + A.T) / 2
    omega_vals, omega_vecs = np.linalg.eigh(S); w = omega_vecs[:, -1]
    ii, jj = np.nonzero(A)
    rows = []
    for i, j in zip(ii, jj):
        dalpha = np.real(np.conj(l[i]) * r[j] / denom)
        domega = w[i] * w[j] if i != j else w[i] ** 2
        rows.append({"target": labels[i], "source": labels[j], "i": i, "j": j, "weight": A[i, j],
                     "dalpha_dAij": dalpha, "domega_dAij": domega,
                     "elasticity_alpha": A[i, j] * dalpha, "elasticity_omega": A[i, j] * domega})
    out = pd.DataFrame(rows)
    out["priority"] = np.maximum(out.elasticity_alpha.abs(), out.elasticity_omega.abs())
    J = -np.eye(len(A)) + g * A
    P = expm(J * t_star); U, s, Vh = svd(P, full_matrices=False); u, v = U[:, 0], Vh[0]
    out["dG_dAij"] = np.nan
    for idx in out.nlargest(top_frechet, "priority").index:
        i, j = int(out.at[idx, "i"]), int(out.at[idx, "j"])
        E = np.zeros_like(A); E[i, j] = g * t_star
        dP = expm_frechet(J * t_star, E, compute_expm=False)
        out.at[idx, "dG_dAij"] = 2 * s[0] * np.real(u @ dP @ v)
    out["elasticity_G"] = out.weight * out.dG_dAij
    return out.sort_values("priority", ascending=False)


def pseudospectrum(J: np.ndarray, nx: int = 42, ny: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ev = eigvals(J)
    xr = np.linspace(min(ev.real.min() - .6, -2), max(.4, ev.real.max() + .6), nx)
    yi = np.linspace(ev.imag.min() - .6, ev.imag.max() + .6, ny)
    zlog = np.empty((ny, nx))
    I = np.eye(len(J))
    for iy, y in enumerate(yi):
        for ix, x in enumerate(xr):
            smin = svd((x + 1j*y) * I - J, compute_uv=False)[-1]
            zlog[iy, ix] = -np.log10(max(smin, 1e-16))
    return xr, yi, zlog


def stochastic_covariance(J: np.ndarray, labels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    C = solve_continuous_lyapunov(J, -np.eye(len(J)))
    C = (C + C.T) / 2
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    node = pd.DataFrame({"node": labels, "stationary_variance": np.diag(C)}).sort_values("stationary_variance", ascending=False)
    modes = pd.DataFrame({"mode": np.arange(1, min(11, len(J) + 1)), "eigenvalue": vals[order[:10]],
                          "variance_fraction": vals[order[:10]] / vals.sum()})
    return node, modes, float(np.trace(C))


def robustness_variants(A: np.ndarray) -> dict[str, np.ndarray]:
    variants = {"raw": A.copy(), "no_diagonal": A - np.diag(np.diag(A))}
    abs_nonzero = np.abs(A[A != 0])
    for pct in [1, 5, 10]:
        threshold = np.percentile(abs_nonzero, 100 - pct)
        M = A.copy(); M[np.abs(M) >= threshold] = 0
        variants[f"remove_top_{pct}pct_edges"] = M
    for pct in [95, 99]:
        cap = np.percentile(abs_nonzero, pct)
        variants[f"winsor_{pct}pct"] = np.clip(A, -cap, cap)
    rng = np.random.default_rng(SEED)
    for scale in [.01, .05]:
        M = A.copy(); mask = M != 0; M[mask] *= 1 + rng.normal(0, scale, mask.sum())
        variants[f"weight_noise_{int(scale*100)}pct"] = M
    return variants


def savefig(path: Path):
    plt.tight_layout(); plt.savefig(path, dpi=240, bbox_inches="tight"); plt.close()


def plot_outputs(A, labels, outputs, phase, gain_curves, coupling_gain, direction, nulls, lesions, opt, single, pseudo):
    figdir = outputs / "figures"
    # Figure 1
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.3, 1]})
    vmax = np.quantile(np.abs(A[A != 0]), .99)
    im = ax[0].imshow(A, cmap="PuOr_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax[0].set(title="Signed weighted matrix (99th-percentile color scale)", xlabel="source j", ylabel="target i")
    fig.colorbar(im, ax=ax[0], fraction=.046)
    nz = A[A != 0]
    ax[1].hist(np.sign(nz) * np.log10(1 + np.abs(nz)), bins=50, color=BLUE, edgecolor="white")
    ax[1].set(title="Signed log-weight distribution", xlabel="sign(w) log10(1+|w|)", ylabel="edges")
    savefig(figdir / "figure01_matrix_and_weights.png")
    # Figure 2
    ev = eigvals(A)
    plt.figure(figsize=(7, 6)); plt.scatter(ev.real, ev.imag, s=24, c=BLUE, alpha=.8); plt.axvline(0, color=INK, lw=.8)
    plt.xlabel("real part"); plt.ylabel("imaginary part"); plt.title("Complex eigenvalue spectrum — raw matrix")
    savefig(figdir / "figure02_eigenvalue_spectrum.png")
    # Figure 3
    plt.figure(figsize=(8, 5)); th = thresholds(A)
    plt.axvspan(0, th["g_react"], color="#E8EDF2", alpha=.8)
    plt.axvspan(th["g_react"], th["g_stab"], color="#F7E8D8", alpha=.75)
    plt.axvspan(th["g_stab"], phase.g.max(), color="#F5DDE2", alpha=.75)
    plt.plot(phase.g, phase.spectral_abscissa_J, color=BLUE, label="spectral abscissa α(J)")
    plt.plot(phase.g, phase.numerical_abscissa_J, color=ORANGE, ls="--", label="numerical abscissa ω(J)")
    plt.axhline(0, color=INK, lw=.8)
    plt.axvline(th["g_react"], color=ORANGE, ls=":", label="g_react")
    plt.axvline(th["g_stab"], color=BLUE, ls=":", label="g_stab")
    ymax = max(phase.numerical_abscissa_J.max(), 1)
    plt.text(.5*th["g_react"], .84*ymax, "stable /\nnon-reactive", ha="center", va="top", fontsize=8)
    plt.text(.5*(th["g_react"]+th["g_stab"]), .84*ymax, "stable / reactive", ha="center", va="top", fontsize=8)
    plt.text(.5*(th["g_stab"]+phase.g.max()), .84*ymax, "unstable", ha="center", va="top", fontsize=8, rotation=90)
    plt.xlabel("coupling g"); plt.ylabel("abscissa"); plt.title("Coupling phase diagram — spectral-norm normalization"); plt.legend()
    savefig(figdir / "figure03_coupling_phase.png")
    # Figure 4
    plt.figure(figsize=(8, 5))
    for name, dat in gain_curves.items(): plt.plot(dat["times"], dat["gains"], label=name.replace("_", " "))
    plt.axhline(1, color=INK, lw=.8); plt.xscale("symlog", linthresh=.02)
    plt.xlabel("time"); plt.ylabel("G(t; g)"); plt.title("Finite-time transient gain"); plt.legend()
    savefig(figdir / "figure04_transient_gain_curves.png")
    # Figure 5
    plt.figure(figsize=(8, 5)); plt.plot(coupling_gain.g, coupling_gain.Gmax, marker="o", color=BLUE)
    plt.axvline(th["g_react"], color=ORANGE, ls=":"); plt.axvline(th["g_stab"], color=BLUE, ls=":")
    plt.yscale("log"); plt.xlabel("coupling g"); plt.ylabel("Gmax"); plt.title("Maximum transient gain versus coupling")
    savefig(figdir / "figure05_max_gain_vs_coupling.png")
    # Figure 6
    order_v = np.argsort(np.abs(opt["v1"]))[-15:]; order_u = np.argsort(np.abs(opt["u1"]))[-15:]
    fig, ax = plt.subplots(1, 2, figsize=(13, 7))
    ax[0].barh(np.asarray(labels)[order_v], np.abs(opt["v1"])[order_v], color=BLUE); ax[0].set_title("Optimal initial perturbation |v1|")
    ax[1].barh(np.asarray(labels)[order_u], np.abs(opt["u1"])[order_u], color=ORANGE); ax[1].set_title("Optimal response |u1|")
    savefig(figdir / "figure06_optimal_vectors.png")
    # Figure 7
    top = single.nsmallest(25, "perturbability_rank").sort_values("single_node_peak_gain")
    plt.figure(figsize=(9, 8)); plt.barh(top.node, top.single_node_peak_gain, color=BLUE)
    plt.xlabel("peak energy gain"); plt.title("Top single-node perturbability rankings")
    savefig(figdir / "figure07_single_node_ranking.png")
    # Figure 8
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].plot(direction.q, direction.commutator_index, color=BLUE); ax[0].set(xlabel="directionality q", ylabel="ηC", title="Commutator non-normality")
    ax[1].plot(direction.q, direction.numerical_abscissa, color=ORANGE); ax[1].set(xlabel="directionality q", ylabel="ω(Aq)", title="Numerical abscissa")
    ax[2].plot(direction.q, direction.Gmax_at_90pct_stability, color=PINK); ax[2].set(xlabel="directionality q", ylabel="Gmax", title="Gain at 90% of stability threshold")
    savefig(figdir / "figure08_directionality_interpolation.png")
    # Figure 9
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [("commutator_index", "ηC"), ("henrici_index", "ηH"), ("Gmax_at_90pct_stability", "Gmax")]
    for a, (m, title) in zip(ax, metrics):
        groups = [nulls.loc[nulls.model == k, m].dropna() for k in ["weight_shuffled", "direction_randomized"]]
        a.boxplot(groups, tick_labels=["weight\nshuffle", "direction\nrandomized"], patch_artist=True,
                  boxprops={"facecolor": "#DDE8F0"}); emp = nulls.loc[nulls.model == "empirical", m].iloc[0]
        a.axhline(emp, color=ORANGE, lw=2, label="empirical"); a.set_title(title)
    ax[0].legend(); savefig(figdir / "figure09_null_distributions.png")
    # Figure 10
    top = lesions.reindex(lesions.delta_Gmax_screen.abs().sort_values(ascending=False).index).head(25).sort_values("delta_Gmax_screen")
    plt.figure(figsize=(9, 8)); plt.barh(top.node, top.delta_Gmax_screen, color=np.where(top.delta_Gmax_screen < 0, BLUE, ORANGE))
    plt.axvline(0, color=INK, lw=.8); plt.xlabel("Δ screened Gmax after lesion"); plt.title("Largest node-lesion effects")
    savefig(figdir / "figure10_node_lesions.png")
    # Pseudospectrum
    xr, yi, z = pseudo
    plt.figure(figsize=(8, 6)); c = plt.contourf(xr, yi, z, levels=24, cmap="Blues"); plt.axvline(0, color=ORANGE, lw=1.2)
    plt.colorbar(c, label="log10 resolvent norm"); plt.xlabel("Re z"); plt.ylabel("Im z"); plt.title("Approximate pseudospectral resolvent map")
    savefig(figdir / "figure11_pseudospectrum.png")


def main(csv_path: str = "mij_matrix.csv", output_dir: str = "analysis_outputs") -> dict[str, object]:
    root = Path(output_dir); tables = root / "tables"; figures = root / "figures"
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    Araw, labels = load_matrix(csv_path)
    desc, quantiles = matrix_validation(Araw); desc.to_csv(tables / "matrix_descriptive_statistics.csv", index=False); quantiles.to_csv(tables / "nonzero_weight_quantiles.csv", index=False)
    strengths = node_strengths(Araw, labels); strengths.to_csv(tables / "node_strengths.csv", index=False)
    dominant_nodes, dominant_edges = concentration(Araw, labels); dominant_nodes.to_csv(tables / "weight_concentration_nodes.csv", index=False); dominant_edges.to_csv(tables / "weight_concentration_edges.csv", index=False)
    versions = normalize_versions(Araw)
    metrics = pd.DataFrame([spectral_metrics(M, name) for name, M in versions.items()])
    thresholds_table = pd.DataFrame([{"normalization": name, **thresholds(M)} for name, M in versions.items()])
    metrics.to_csv(tables / "non_normality_metrics.csv", index=False); thresholds_table.to_csv(tables / "stability_reactivity_thresholds.csv", index=False)
    A = versions["spectral_norm"]
    phase, th = coupling_phase(A); phase.to_csv(tables / "coupling_phase.csv", index=False)
    gain_curves = {}; selected_rows = []
    for name, g in representative_couplings(th):
        res = transient_gain(-np.eye(len(A)) + g * A)
        gain_curves[name] = res; selected_rows.append({"coupling_label": name, "g": g, "Gmax": res["Gmax"], "t_star": res["t_star"]})
    selected = pd.DataFrame(selected_rows); selected.to_csv(tables / "selected_coupling_gains.csv", index=False)
    g_oper = .5 * (th["g_react"] + th["g_stab"]) if th["g_react"] < th["g_stab"] else .9 * th["g_stab"]
    J = -np.eye(len(A)) + g_oper * A
    opt = transient_gain(J, n_time=90)
    init = pd.DataFrame({"node": labels, "abs_v1": np.abs(opt["v1"]), "v1": opt["v1"]}).sort_values("abs_v1", ascending=False)
    recv = pd.DataFrame({"node": labels, "abs_u1": np.abs(opt["u1"]), "u1": opt["u1"]}).sort_values("abs_u1", ascending=False)
    init.to_csv(tables / "optimal_perturbation_nodes.csv", index=False); recv.to_csv(tables / "optimal_response_nodes.csv", index=False)
    single, susceptibility, response = single_node_perturbations(J, labels)
    single.to_csv(tables / "single_node_perturbability.csv", index=False); susceptibility.to_csv(tables / "node_susceptibility.csv", index=False); response.to_csv(tables / "downstream_max_response_matrix.csv")
    beta = signed_interpolation(Araw, np.linspace(0, 1, 21)); beta.to_csv(tables / "signed_inhibition_interpolation.csv", index=False)
    direction = directionality_interpolation(Araw, np.linspace(0, 1, 21)); direction.to_csv(tables / "directionality_interpolation.csv", index=False)
    cutoffs = []
    for delta in [.01, .05, .10, .25]:
        hit = direction.loc[direction.Gmax_at_90pct_stability >= 1 + delta, "q"]
        cutoffs.append({"delta": delta, "q_cutoff_grid": hit.min() if len(hit) else np.nan})
    pd.DataFrame(cutoffs).to_csv(tables / "directionality_cutoffs.csv", index=False)
    nulls = null_models(Araw); nulls.to_csv(tables / "null_model_results.csv", index=False)
    null_stats = empirical_null_statistics(nulls); null_stats.to_csv(tables / "empirical_vs_null_statistics.csv", index=False)
    lesions = lesion_analysis(A, labels, g_oper, opt["Gmax"], opt["t_star"]); lesions.to_csv(tables / "node_lesion_effects.csv", index=False)
    edges = edge_sensitivity(A, labels, g_oper, opt["t_star"]); edges.to_csv(tables / "edge_sensitivity.csv", index=False)
    # Identical-noise comparison requires a coupling stable for both matrices.
    Asym = (A + A.T) / 2
    g_noise = .9 * min(th["g_stab"], thresholds(Asym)["g_stab"])
    Jnoise = -np.eye(len(A)) + g_noise * A
    Jnoise_sym = -np.eye(len(A)) + g_noise * Asym
    node_var, cov_modes, total_var = stochastic_covariance(Jnoise, labels); node_var.to_csv(tables / "stochastic_node_variance.csv", index=False); cov_modes.to_csv(tables / "stochastic_covariance_modes.csv", index=False)
    _, _, total_var_sym = stochastic_covariance(Jnoise_sym, labels)
    # Coupling-to-gain curve only on stable range.
    gs = np.linspace(0, .98 * th["g_stab"], 24)
    coupling_gain = pd.DataFrame([{"g": g, "Gmax": transient_gain(-np.eye(len(A)) + g*A, n_time=38, refine=False)["Gmax"]} for g in gs])
    coupling_gain.to_csv(tables / "maximum_gain_vs_coupling.csv", index=False)
    pseudo = pseudospectrum(J)
    np.savez_compressed(root / "pseudospectrum_grid.npz", real=pseudo[0], imag=pseudo[1], log10_resolvent=pseudo[2])
    robust_rows = []
    for name, M0 in robustness_variants(Araw).items():
        M = M0 / norm(M0, 2); met, thr = spectral_metrics(M), thresholds(M)
        gg = .9 * thr["g_stab"] if np.isfinite(thr["g_stab"]) else 1.0
        gain = transient_gain(-np.eye(len(M)) + gg*M, n_time=34, refine=False)["Gmax"]
        robust_rows.append({"variant": name, **{k: met[k] for k in ["commutator_index", "henrici_index", "spectral_abscissa", "numerical_abscissa"]}, **thr, "Gmax_at_90pct_stability": gain})
    robustness = pd.DataFrame(robust_rows); robustness.to_csv(tables / "robustness_analysis.csv", index=False)
    # signed component comparison
    signed_rows = []
    for name, M0 in [("positive", np.maximum(Araw,0)), ("negative", np.minimum(Araw,0)), ("signed", Araw)]:
        M=M0/norm(M0,2); met=spectral_metrics(M,name); thr=thresholds(M); gg=.9*thr["g_stab"] if np.isfinite(thr["g_stab"]) else 1
        gain=transient_gain(-np.eye(len(M))+gg*M,n_time=42,refine=False)["Gmax"] if np.max(eigvals(-np.eye(len(M))+gg*M).real)<0 else np.nan
        signed_rows.append({**met, **thr, "Gmax_at_90pct_stability":gain})
    pd.DataFrame(signed_rows).to_csv(tables / "signed_component_comparison.csv",index=False)
    plot_outputs(A, labels, root, phase, gain_curves, coupling_gain, direction, nulls, lesions, opt, single, pseudo)
    summary = {
        "matrix_size": list(Araw.shape), "nonzero_count": int(np.count_nonzero(Araw)), "density": float(np.count_nonzero(Araw)/Araw.size),
        "convention": "A[i,j] is effect of source j on target i", "normalization_for_dynamics": "spectral norm",
        "alpha_A2": float(th["alpha_A"]), "omega_A2": float(th["omega_A"]), "g_react": float(th["g_react"]), "g_stab": float(th["g_stab"]),
        "g_operating": float(g_oper), "Gmax_operating": float(opt["Gmax"]), "t_star_operating": float(opt["t_star"]),
        "top_initiator": init.iloc[0].node, "top_receiver": recv.iloc[0].node,
        "top_single_node": single.iloc[0].node, "strongest_gain_reducing_lesion_screen": lesions.iloc[0].node,
        "stochastic_comparison_coupling": float(g_noise), "total_stationary_variance": total_var, "symmetrized_total_stationary_variance": total_var_sym,
        "pseudospectral_right_half_min_epsilon_grid": float(10 ** (-pseudo[2][:, pseudo[0] > 0].max())),
        "top_edge_by_local_spectral_or_reactivity_sensitivity": f"{edges.iloc[0].source} -> {edges.iloc[0].target}",
        "null_randomizations_per_family": 50, "null_gain_randomizations_per_family": 20,
        "notes": ["Node-lesion Gmax is a five-time-point screen around the empirical t*; other lesion metrics are exact.",
                  "Edge alpha/omega effects use first-order eigenvalue sensitivities; G sensitivity uses Frechet derivatives for the top 40 screened edges.",
                  "Pseudospectrum is a finite grid approximation, not a contour-certified boundary."]
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
