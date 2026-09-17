from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import eigvals, expm, svdvals
from scipy.stats import kurtosis, skew
from scipy.sparse.linalg import expm_multiply


def load_signed_matrix(
    matrix_path: Path,
    csv_orientation: str = "source_rows",
    zero_diagonal: bool = True,
) -> Tuple[list[str], np.ndarray, np.ndarray, pd.Series]:
    raw = pd.read_csv(matrix_path, header=None)
    row_labels = raw.iloc[1:, 0].astype(str).tolist()
    col_labels = raw.iloc[0, 1:].astype(str).tolist()
    weights_csv = (
        raw.iloc[1:, 1:]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float, copy=True)
    )

    if weights_csv.shape[0] != weights_csv.shape[1]:
        raise ValueError("Matrix must be square.")
    if len(row_labels) != weights_csv.shape[0] or len(col_labels) != weights_csv.shape[1]:
        raise ValueError("Label dimensions do not match matrix.")
    if row_labels != col_labels:
        raise ValueError("Row and column labels differ; inspect ordering before proceeding.")
    if not np.isfinite(weights_csv).all():
        raise ValueError("Matrix contains NaN/Inf values.")

    if zero_diagonal:
        np.fill_diagonal(weights_csv, 0.0)

    if csv_orientation == "source_rows":
        weights_state = weights_csv.T.copy()
    elif csv_orientation == "target_rows":
        weights_state = weights_csv.copy()
    else:
        raise ValueError("csv_orientation must be 'source_rows' or 'target_rows'.")

    n = len(row_labels)
    summary = pd.Series(
        {
            "nodes": n,
            "positive edges": int((weights_csv > 0).sum()),
            "negative edges": int((weights_csv < 0).sum()),
            "zero entries": int((weights_csv == 0).sum()),
            "nonzero density (off-diagonal)": float(
                (weights_csv != 0).sum() / (n * (n - 1))
            ),
            "minimum weight": float(weights_csv.min()),
            "maximum weight": float(weights_csv.max()),
        }
    )
    return row_labels, weights_csv, weights_state, summary


def weight_scale_report(weights: np.ndarray) -> pd.Series:
    nonzero = weights[weights != 0]
    return pd.Series(
        {
            "|weight| median": np.median(np.abs(nonzero)),
            "|weight| 95th percentile": np.quantile(np.abs(nonzero), 0.95),
            "|weight| maximum": np.max(np.abs(nonzero)),
        }
    )


def normalize_and_split(
    weights_state: np.ndarray,
    quantile: float = 0.95,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    abs_nonzero = np.abs(weights_state[weights_state != 0])
    scale = float(np.quantile(abs_nonzero, quantile))
    weights = weights_state / scale
    weights_e = np.maximum(weights, 0.0)
    weights_i = np.maximum(-weights, 0.0)
    return scale, weights, weights_e, weights_i


def resolve_backbone(
    labels: Sequence[str],
    backbone_labels: Optional[Sequence[str]] = None,
) -> Tuple[list[str], dict[str, int], np.ndarray, dict[str, int], bool]:
    if backbone_labels is None:
        backbone = list(labels)
        used_placeholder = True
    else:
        missing = [x for x in backbone_labels if x not in labels]
        if missing:
            raise ValueError(f"Backbone labels not found in matrix: {missing[:10]}")
        if len(set(backbone_labels)) != len(backbone_labels):
            raise ValueError("Backbone contains duplicate nodes.")
        backbone = list(backbone_labels)
        used_placeholder = False

    index = {label: i for i, label in enumerate(labels)}
    backbone_index = np.array([index[x] for x in backbone], dtype=int)
    position = {node: k for k, node in enumerate(backbone)}
    return backbone, index, backbone_index, position, used_placeholder


def system_matrix(
    weights_e: np.ndarray,
    weights_i: np.ndarray,
    g: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.8,
) -> np.ndarray:
    return -alpha * np.eye(weights_e.shape[0]) + beta * (weights_e - g * weights_i)


def dale_sender_summary(labels: Sequence[str], weights: np.ndarray) -> pd.DataFrame:
    rows = []
    for j, label in enumerate(labels):
        out = weights[:, j]
        nonzero = out[out != 0]
        if len(nonzero) == 0:
            frac_pos = frac_neg = np.nan
            classification = "isolated/no outgoing"
        else:
            frac_pos = float(np.mean(nonzero > 0))
            frac_neg = float(np.mean(nonzero < 0))
            if frac_neg >= 0.9:
                classification = "mostly inhibitory"
            elif frac_pos >= 0.9:
                classification = "mostly excitatory"
            else:
                classification = "mixed-sign"
        rows.append((label, len(nonzero), frac_pos, frac_neg, classification))

    return pd.DataFrame(
        rows,
        columns=[
            "node",
            "out_degree",
            "fraction_positive",
            "fraction_negative",
            "classification",
        ],
    )


def impulse_response(
    weights_e: np.ndarray,
    weights_i: np.ndarray,
    x0: np.ndarray,
    times: np.ndarray,
    g: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.8,
) -> np.ndarray:
    matrix = system_matrix(weights_e, weights_i, g=g, alpha=alpha, beta=beta)
    return expm_multiply(matrix, x0, start=times[0], stop=times[-1], num=len(times))


def backbone_metrics(
    response: np.ndarray,
    times: np.ndarray,
    backbone: Sequence[str],
    backbone_index: np.ndarray,
) -> pd.DataFrame:
    values = response[:, backbone_index]
    abs_values = np.abs(values)
    peak = abs_values.max(axis=0)
    auc = np.trapezoid(abs_values, times, axis=0)
    t_peak = times[np.argmax(abs_values, axis=0)]
    return pd.DataFrame(
        {
            "backbone_position": np.arange(len(backbone)),
            "node": list(backbone),
            "peak_abs": peak,
            "auc_abs": auc,
            "t_peak": t_peak,
        }
    )


def inhibition_strength_sweep(
    g_values: Iterable[float],
    weights_e: np.ndarray,
    weights_i: np.ndarray,
    x0: np.ndarray,
    times: np.ndarray,
    terminal_index: int,
    backbone_index: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.8,
) -> pd.DataFrame:
    rows = []
    for g in g_values:
        matrix = system_matrix(weights_e, weights_i, g=g, alpha=alpha, beta=beta)
        eig = eigvals(matrix)
        response = impulse_response(
            weights_e, weights_i, x0, times, g=g, alpha=alpha, beta=beta
        )
        terminal = np.abs(response[:, terminal_index])
        all_backbone = np.abs(response[:, backbone_index])
        rows.append(
            {
                "g": g,
                "spectral_abscissa": np.max(eig.real),
                "terminal_peak": terminal.max(),
                "terminal_auc": np.trapezoid(terminal, times),
                "terminal_t_peak": times[np.argmax(terminal)],
                "backbone_total_auc": np.trapezoid(all_backbone, times, axis=0).sum(),
            }
        )
    return pd.DataFrame(rows)


def transient_amplification(
    g_values: Iterable[float],
    transient_times: np.ndarray,
    weights_e: np.ndarray,
    weights_i: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.8,
) -> pd.DataFrame:
    rows = []
    for g in g_values:
        matrix = system_matrix(weights_e, weights_i, g=g, alpha=alpha, beta=beta)
        gains = np.asarray([svdvals(expm(matrix * t))[0] for t in transient_times])
        rows.append(
            {
                "g": g,
                "G_max": gains.max(),
                "t_at_G_max": transient_times[np.argmax(gains)],
                "spectral_abscissa": np.max(eigvals(matrix).real),
                "numerical_abscissa": np.max(np.linalg.eigvalsh((matrix + matrix.T) / 2)),
            }
        )
    return pd.DataFrame(rows)


def inhibitory_node_ablation(
    labels: Sequence[str],
    weights_e: np.ndarray,
    weights_i: np.ndarray,
    x0: np.ndarray,
    times: np.ndarray,
    terminal_index: int,
    position: dict[str, int],
    baseline_terminal_peak: float,
    g: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.8,
) -> pd.DataFrame:
    rows = []
    for j, label in enumerate(labels):
        if not np.any(weights_i[:, j] > 0):
            continue

        weights_i_ablated = weights_i.copy()
        weights_i_ablated[:, j] = 0.0
        response = impulse_response(
            weights_e,
            weights_i_ablated,
            x0,
            times,
            g=g,
            alpha=alpha,
            beta=beta,
        )
        terminal_peak = np.abs(response[:, terminal_index]).max()
        rows.append(
            {
                "node": label,
                "backbone_position": position.get(label, np.nan),
                "inhibitory_out_degree": int((weights_i[:, j] > 0).sum()),
                "inhibitory_out_strength": weights_i[:, j].sum(),
                "terminal_peak_after_ablation": terminal_peak,
                "delta_terminal_peak": terminal_peak - baseline_terminal_peak,
            }
        )
    return pd.DataFrame(rows).sort_values("delta_terminal_peak", ascending=False)


def inhibitory_edge_ablation(
    labels: Sequence[str],
    weights_e: np.ndarray,
    weights_i: np.ndarray,
    x0: np.ndarray,
    times: np.ndarray,
    terminal_index: int,
    baseline_terminal_peak: float,
    top_k: int = 50,
    g: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.8,
) -> pd.DataFrame:
    targets, sources = np.where(weights_i > 0)
    edge_table = pd.DataFrame(
        {
            "source_idx": sources,
            "target_idx": targets,
            "source": [labels[j] for j in sources],
            "target": [labels[i] for i in targets],
            "inhibitory_magnitude": weights_i[targets, sources],
        }
    ).sort_values("inhibitory_magnitude", ascending=False).head(top_k)

    rows = []
    for row in edge_table.itertuples(index=False):
        weights_i_ablated = weights_i.copy()
        weights_i_ablated[row.target_idx, row.source_idx] = 0.0
        response = impulse_response(
            weights_e,
            weights_i_ablated,
            x0,
            times,
            g=g,
            alpha=alpha,
            beta=beta,
        )
        terminal_peak = np.abs(response[:, terminal_index]).max()
        rows.append(
            {
                "source": row.source,
                "target": row.target,
                "inhibitory_magnitude": row.inhibitory_magnitude,
                "delta_terminal_peak": terminal_peak - baseline_terminal_peak,
            }
        )

    return pd.DataFrame(rows).sort_values("delta_terminal_peak", ascending=False)


def inhibitory_backbone_motifs(
    labels: Sequence[str],
    weights_i: np.ndarray,
    position: dict[str, int],
) -> pd.DataFrame:
    targets, sources = np.where(weights_i > 0)
    rows = []
    for target_idx, source_idx in zip(targets, sources):
        source = labels[source_idx]
        target = labels[target_idx]
        if source not in position or target not in position:
            continue
        distance = position[target] - position[source]
        motif = "feedforward" if distance > 0 else ("feedback" if distance < 0 else "self")
        rows.append(
            {
                "source": source,
                "target": target,
                "source_position": position[source],
                "target_position": position[target],
                "distance": distance,
                "abs_distance": abs(distance),
                "motif": motif,
                "inhibitory_magnitude": weights_i[target_idx, source_idx],
            }
        )
    return pd.DataFrame(rows)


def randomize_inhibitory_targets(
    weights_i: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n = weights_i.shape[0]
    weights_i_null = np.zeros_like(weights_i)
    for source_idx in range(n):
        existing_targets = np.flatnonzero(weights_i[:, source_idx] > 0)
        weights = weights_i[existing_targets, source_idx].copy()
        k = len(weights)
        if k == 0:
            continue
        allowed = np.array([target_idx for target_idx in range(n) if target_idx != source_idx])
        new_targets = rng.choice(allowed, size=k, replace=False)
        rng.shuffle(weights)
        weights_i_null[new_targets, source_idx] = weights
    return weights_i_null


def inhibitory_target_null_model(
    weights_e: np.ndarray,
    weights_i: np.ndarray,
    x0: np.ndarray,
    times: np.ndarray,
    terminal_index: int,
    observed_metric: float,
    n_null: int = 500,
    g: float = 1.0,
    random_seed: int = 2026,
    alpha: float = 1.0,
    beta: float = 0.8,
) -> Tuple[np.ndarray, pd.Series]:
    """Compare the observed terminal peak against randomized inhibitory targets.

    Returns the null draws and a summary Series. Inference is by empirical p-value,
    not by z-score: this null is strongly right-skewed (a minority of randomizations
    produce very large terminal peaks), so its mean sits far above its median and the
    standard deviation is several times the mean. A z-score computed against a
    distribution like that is not interpretable, and at the previous default of
    n_null=25 it was not even reproducible across reruns.

    The estimator used is ``(1 + #{null >= observed}) / (1 + n_null)``, which is the
    convention already prescribed in the null-model section of
    ``ee_backbone_deterministic_stochastic_comparison.ipynb`` and which can never
    return exactly zero.

    ``spectral_abscissa`` of each null draw is also recorded, because a randomization
    that destabilizes the system would inflate the terminal peak for reasons that have
    nothing to do with inhibitory placement.
    """
    rng = np.random.default_rng(random_seed)
    null_metrics = np.empty(n_null)
    null_abscissa = np.empty(n_null)
    for draw in range(n_null):
        weights_i_null = randomize_inhibitory_targets(weights_i, rng)
        response = impulse_response(
            weights_e,
            weights_i_null,
            x0,
            times,
            g=g,
            alpha=alpha,
            beta=beta,
        )
        null_metrics[draw] = np.abs(response[:, terminal_index]).max()
        null_abscissa[draw] = np.max(
            eigvals(
                system_matrix(weights_e, weights_i_null, g=g, alpha=alpha, beta=beta)
            ).real
        )

    n_at_least = int(np.sum(null_metrics >= observed_metric))
    n_at_most = int(np.sum(null_metrics <= observed_metric))
    null_sd = float(null_metrics.std(ddof=1))

    summary = pd.Series(
        {
            "observed": float(observed_metric),
            "n_null": int(n_null),
            "null_mean": float(null_metrics.mean()),
            "null_median": float(np.median(null_metrics)),
            "null_sd": null_sd,
            "null_skew": float(skew(null_metrics)),
            "null_excess_kurtosis": float(kurtosis(null_metrics)),
            "observed_percentile_in_null": float(100.0 * np.mean(null_metrics < observed_metric)),
            "empirical_p_observed_ge_null": (1 + n_at_least) / (1 + n_null),
            "empirical_p_observed_le_null": (1 + n_at_most) / (1 + n_null),
            "z_score_do_not_use": float((observed_metric - null_metrics.mean()) / (null_sd + 1e-12)),
            "n_null_unstable": int(np.sum(null_abscissa > 0)),
            "null_max_spectral_abscissa": float(null_abscissa.max()),
        }
    )
    return null_metrics, summary
