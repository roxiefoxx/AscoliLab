"""Helpers for ``07_isn_paradox.ipynb``: ISN status and paradoxical responses.

Plan: ``docs/ISN_PARADOX_PLAN.md``. The notebook stays short and calls the
functions here, one phase per section.

Model (same as the rest of ``jacobian_analysis``)
--------------------------------------------------
::

    tau dx/dt = -x + g W x + b         W = M[post, pre], spectral radius 1
    J = T^-1 (-I + g W)                T = diag(tau)
    Gamma = (I - g W)^-1               steady-state response to input b

``g`` is the global synaptic gain. Every magnitude in this project is
conditional on a normalisation constant, so ISN status is reported as a
*range of g*, not as a yes/no at one arbitrary value.

Definitions
-----------
ISN
    Full network stable, excitatory subnetwork (inhibition clamped) unstable:
    ``alpha(g W_EE) > 1 > alpha(g W)``. ``alpha`` is the max real part of the
    eigenvalues (the stability boundary of ``-I + gW`` sits at Re = 1).
Self-paradox of type k
    ``Gamma_kk < 0``. By Cramer's rule
    ``Gamma_kk = det(I - g W_-k) / det(I - g W)``, where ``W_-k`` has type k
    removed (clamped). Inside the stable range the denominator is positive, so
    type k is paradoxical  <=>  an odd number of real eigenvalues of
    ``g W_-k`` exceed 1, i.e. the network is unstable with k clamped.
Class paradox
    The summed response of a set S of types to a drive on S is negative. This
    depends on the basis (uniform vs census-weighted drive); ``Gamma_kk`` does
    not, because diagonal similarity transforms leave the diagonal of the
    inverse unchanged.

Naming
------
Block names in this module are always written *source -> target* and spelled
out (``i_to_e``), never as a bare ``EI``. In a ``[post, pre]`` matrix,
``i_to_e = W[E rows, I columns]``.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import expm

import jacobian_core as jc

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_Q_PATH = REPO / "non-normal_matrices" / "data" / "ml_qi_vector.csv"

__all__ = [
    "Network",
    "load_network",
    "cell_metadata",
    "class_sets",
    "abscissa",
    "isn_window",
    "inhibitory_schur_complement",
    "gain_sweep",
    "raw_gain_report",
    "leading_mode",
    "regional_isn",
    "clamp_screen",
    "stabilizer_contributions",
    "response_matrix",
    "self_response_sweep",
    "paradox_onsets",
    "verify_cramer",
    "class_response_table",
    "path_decomposition",
    "two_population_reduction",
    "step_response",
    "step_response_table",
    "tau_ratio_sweep",
    "surrogate_operators",
    "isn_profile",
    "profile_table",
    "jitter_ensemble",
    "null_ensemble",
    "rate_dependent_isn",
    "simulate_rectified",
    "nonlinear_step_table",
    "prediction_table",
    "plot_gain_sweep",
    "plot_onsets",
    "plot_step_responses",
    "plot_tau_ratio",
    "plot_null",
]


# ==========================================================================
# Data
# ==========================================================================


@dataclass
class Network:
    """Oriented operator plus metadata, normalised to unit spectral radius."""

    data: jc.JacobianData
    variants: dict[str, np.ndarray]  # with_self / no_self, each rho = 1
    labels: list[str]
    meta: pd.DataFrame  # label-indexed: ei, region, ec_depth, i_class
    E: np.ndarray  # boolean masks
    I: np.ndarray
    kappa: np.ndarray  # postsynaptic gain per type (Hz/pA)
    q: np.ndarray  # census per type

    @property
    def n(self) -> int:
        return len(self.labels)

    def idx(self, names: Iterable[str]) -> np.ndarray:
        pos = {name: i for i, name in enumerate(self.labels)}
        return np.array([pos[name] for name in names], dtype=int)


def _unit_rho(W: np.ndarray) -> np.ndarray:
    rho = float(np.max(np.abs(np.linalg.eigvals(W))))
    return W / rho


def load_network(
    matrix: str | Path = "matrices/mij_matrix.csv",
    netlist: str | Path = "matrices/mij_netlist.csv",
    q_path: str | Path = DEFAULT_Q_PATH,
) -> Network:
    """Load ``M``, verify orientation, build unit-rho with_self / no_self variants."""
    data = jc.load_jacobian_data(matrix, netlist)
    labels = data.labels
    masks = jc.ei_masks(data.ei, labels)
    variants = {k: _unit_rho(v) for k, v in jc.matrix_variants(data.W_raw).items()}

    kappa = np.ones(len(labels))
    if data.netlist is not None and "kappa_j" in data.netlist.columns:
        kmap = data.netlist.groupby("post_neuron")["kappa_j"].first().to_dict()
        kappa = np.array([kmap.get(name, np.nan) for name in labels], dtype=float)
        kappa[~np.isfinite(kappa) | (kappa <= 0)] = 1.0

    q = np.full(len(labels), np.nan)
    if Path(q_path).exists():
        qmap = pd.read_csv(q_path).set_index("neuron_type")["q_i"]
        q = qmap.reindex(labels).to_numpy(dtype=float)

    return Network(
        data=data,
        variants=variants,
        labels=labels,
        meta=cell_metadata(labels, data.ei),
        E=masks["e"],
        I=masks["i"],
        kappa=kappa,
        q=q,
    )


# ---- cell metadata --------------------------------------------------------

# Interneuron classes from Hippocampome type names. Keyword-based, so check the
# table printed in the notebook and edit here if a type lands in the wrong class.
I_CLASS_RULES: list[tuple[str, str]] = [
    ("axo_axonic", r"Axo Axonic"),
    ("basket_CCK", r"Basket CCK"),
    ("basket", r"Basket|AIPRIM"),
    ("bistratified", r"Bistratified"),
    ("OLM_like", r"\bO LM|O LMR|OR LM|Oriens Alveus|HIPP\b|HIPROM"),
    ("neurogliaform_ivy", r"Neurogliaform|Ivy|MOPP"),
    ("projecting", r"Back Projection|Projecting|Trilaminar|Radial Trilaminar"),
    (
        "dendrite_targeting",
        r"Perforant Path Associated|Schaffer Collateral Associated|Mossy Fiber Associated|HICAP|MOLAX"
        r"|LMR|Apical Targeting|Total Molecular Layer|Radiatum|SO SO|SP SR|Lucidum|Quadrilaminar",
    ),
]

REGION_ALIASES = {"CA3c": "CA3"}


def _region(name: str) -> str:
    head = name.split()[0]
    return REGION_ALIASES.get(head, head)


def _ec_depth(name: str) -> str:
    """Superficial (LI-LIII) vs deep (LIV-LVI) for entorhinal types; '' elsewhere."""
    if _region(name) not in {"EC", "LEC", "MEC"}:
        return ""
    layers = re.findall(r"\bL([IV]+)\b", name)
    if not layers:
        return ""
    deep = {"IV", "V", "VI"}
    return "deep" if layers[0] in deep else "superficial"


def _i_class(name: str) -> str:
    for cls, pattern in I_CLASS_RULES:
        if re.search(pattern, name):
            return cls
    return "other"


def cell_metadata(labels: Sequence[str], ei: pd.Series) -> pd.DataFrame:
    meta = pd.DataFrame(index=pd.Index(labels, name="cell_type"))
    meta["ei"] = [str(ei.loc[x]).lower() for x in labels]
    meta["region"] = [_region(x) for x in labels]
    meta["ec_depth"] = [_ec_depth(x) for x in labels]
    meta["i_class"] = [(_i_class(x) if e == "i" else "") for x, e in zip(labels, meta["ei"])]
    return meta


def class_sets(net: Network) -> dict[str, list[str]]:
    """Named sets of inhibitory types: all-I, by class, by region."""
    m = net.meta
    inh = m[m.ei == "i"]
    sets: dict[str, list[str]] = {"all_I": inh.index.tolist()}
    for cls, grp in inh.groupby("i_class"):
        sets[f"class:{cls}"] = grp.index.tolist()
    for reg, grp in inh.groupby("region"):
        sets[f"region:{reg}"] = grp.index.tolist()
    for (reg, cls), grp in inh.groupby(["region", "i_class"]):
        if len(grp) >= 1 and cls != "other":
            sets[f"{reg}:{cls}"] = grp.index.tolist()
    return sets


# ==========================================================================
# Phase 1 - operating point
# ==========================================================================


def abscissa(A: np.ndarray) -> float:
    """Max real part of the eigenvalues (``-inf`` for an empty matrix)."""
    if A.size == 0:
        return -np.inf
    return float(np.max(np.linalg.eigvals(A).real))


def _blocks(W: np.ndarray, E: np.ndarray, I: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "e_to_e": W[np.ix_(E, E)],
        "i_to_e": W[np.ix_(E, I)],
        "e_to_i": W[np.ix_(I, E)],
        "i_to_i": W[np.ix_(I, I)],
    }


def _crit_gain(alpha: float) -> float:
    return 1.0 / alpha if alpha > 0 else np.inf


def isn_window(W: np.ndarray, E: np.ndarray) -> dict[str, float]:
    """Gain range in which the network is an ISN.

    ``g_E`` is where the E subnetwork loses stability, ``g_full`` where the
    whole network does. ISN for ``g_E < g < g_full``.
    """
    g_e = _crit_gain(abscissa(W[np.ix_(E, E)]))
    g_full = _crit_gain(abscissa(W))
    exists = g_e < g_full
    return {
        "alpha_full": abscissa(W),
        "alpha_EE": abscissa(W[np.ix_(E, E)]),
        "g_E": g_e,
        "g_full": g_full,
        "window_exists": bool(exists),
        "window_width": float(g_full - g_e) if exists else 0.0,
        "window_relative": float((g_full - g_e) / g_full) if exists else 0.0,
    }


def inhibitory_schur_complement(W: np.ndarray, E: np.ndarray, I: np.ndarray, g: float) -> np.ndarray:
    """Effective E->E operator with inhibition eliminated at steady state.

    ``W_eff = gW_e_to_e + gW_i_to_e (I - gW_i_to_i)^-1 gW_e_to_i``.
    """
    b = _blocks(g * W, E, I)
    n_i = int(I.sum())
    return b["e_to_e"] + b["i_to_e"] @ np.linalg.solve(np.eye(n_i) - b["i_to_i"], b["e_to_i"])


def gain_sweep(W: np.ndarray, E: np.ndarray, I: np.ndarray, gains: Iterable[float]) -> pd.DataFrame:
    rows = []
    for g in gains:
        a_full = abscissa(g * W)
        a_ee = abscissa(g * W[np.ix_(E, E)])
        a_ii = abscissa(g * W[np.ix_(I, I)])
        try:
            a_eff = abscissa(inhibitory_schur_complement(W, E, I, g))
        except np.linalg.LinAlgError:
            a_eff = np.nan
        rows.append(
            {
                "g": g,
                "alpha_full": a_full,
                "alpha_EE": a_ee,
                "alpha_E_eff": a_eff,
                "alpha_II": a_ii,
                "stable": a_full < 1,
                "E_unstable": a_ee > 1,
                "ISN": (a_full < 1) and (a_ee > 1),
            }
        )
    return pd.DataFrame(rows)


def raw_gain_report(net: Network) -> pd.DataFrame:
    """What the *unnormalised* operator implies, per variant.

    ``g_raw`` is the gain that turns the unit-rho matrix back into raw ``M``.
    If ``M`` were already in rate-to-rate units this would be the operating
    point; values far above ``g_full`` mean the raw units are not a closed-loop
    gain and an extra scale factor is missing.
    """
    rows = []
    for name, W_raw in jc.matrix_variants(net.data.W_raw).items():
        rho = float(np.max(np.abs(np.linalg.eigvals(W_raw))))
        win = isn_window(W_raw / rho, net.E)
        rows.append({"variant": name, "raw_rho": rho, "g_raw": rho, **win})
    return pd.DataFrame(rows).set_index("variant")


# ==========================================================================
# Phase 2 - ISN test proper
# ==========================================================================


def leading_mode(A: np.ndarray, labels: Sequence[str], top: int | None = 10) -> pd.DataFrame:
    """Right and left eigenvector loadings of the max-real-part eigenvalue."""
    vals, right = np.linalg.eig(A)
    k = int(np.argmax(vals.real))
    lam = vals[k]
    vals_t, left = np.linalg.eig(A.T)
    kl = int(np.argmin(np.abs(vals_t - lam)))
    r = np.abs(right[:, k])
    l = np.abs(left[:, kl])
    out = pd.DataFrame(
        {"right_loading": r / r.sum(), "left_loading": l / l.sum()},
        index=pd.Index(labels, name="cell_type"),
    ).sort_values("right_loading", ascending=False)
    out.attrs["eigenvalue"] = complex(lam)
    return out.head(top) if top else out


def regional_isn(net: Network, W: np.ndarray, g_ref: float | None = None) -> pd.DataFrame:
    """Local ISN windows for region-restricted subnetworks.

    For region R: ``g_E_R`` destabilises R's E types alone; ``g_RR`` destabilises
    R's full (E+I) subnetwork in isolation. A region is a local ISN candidate if
    ``g_E_R < min(g_RR, g_full)``.
    """
    g_full = _crit_gain(abscissa(W))
    groups = dict(net.meta.groupby("region").groups)
    ec = net.meta[net.meta.ec_depth != ""]
    for depth, grp in ec.groupby("ec_depth"):
        groups[f"EC-all {depth}"] = grp.index
    rows = []
    for region, names in groups.items():
        idx = net.idx(names)
        in_r = np.zeros(net.n, bool)
        in_r[idx] = True
        e_r = in_r & net.E
        if e_r.sum() == 0:
            continue
        g_er = _crit_gain(abscissa(W[np.ix_(e_r, e_r)]))
        g_rr = _crit_gain(abscissa(W[np.ix_(in_r, in_r)]))
        row = {
            "region": region,
            "n_E": int(e_r.sum()),
            "n_I": int((in_r & net.I).sum()),
            "g_E_region": g_er,
            "g_region_full": g_rr,
            "local_ISN_window": g_er < min(g_rr, g_full),
        }
        if g_ref is not None:
            row[f"local_ISN_at_g={g_ref}"] = g_er < g_ref < min(g_rr, g_full)
        rows.append(row)
    return pd.DataFrame(rows).set_index("region").sort_values("g_E_region")


def _real_eigs_above(A: np.ndarray, threshold: float = 1.0, tol: float = 1e-10) -> int:
    ev = np.linalg.eigvals(A)
    return int(np.sum((np.abs(ev.imag) < tol) & (ev.real > threshold)))


def clamp_screen(
    net: Network, W: np.ndarray, g: float, sets: Mapping[str, Sequence[str]]
) -> pd.DataFrame:
    """Clamp (remove) each set of types; is the remaining network stable?

    For a single type this is exactly the self-paradox test (module docstring).
    """
    rows = []
    for name, members in sets.items():
        keep = np.ones(net.n, bool)
        keep[net.idx(members)] = False
        sub = g * W[np.ix_(keep, keep)]
        a = abscissa(sub)
        n_real = _real_eigs_above(sub)
        rows.append(
            {
                "clamped": name,
                "n_types": len(members),
                "alpha_after_clamp": a,
                "unstable_after_clamp": a > 1,
                "n_real_eigs_gt_1": n_real,
                "odd_real_crossings": n_real % 2 == 1,
            }
        )
    return pd.DataFrame(rows).set_index("clamped").sort_values("alpha_after_clamp", ascending=False)


def stabilizer_contributions(net: Network, W: np.ndarray, g: float) -> pd.DataFrame:
    """Attribute the stabilising shift of the leading E eigenvalue to I types.

    ``W_eff - gW_e_to_e = sum_k outer(B[:, k], R[k, :])`` with ``B = gW_i_to_e``
    and ``R = (I - gW_i_to_i)^-1 gW_e_to_i``. First-order shift of the leading
    eigenvalue ``lambda`` of ``gW_e_to_e``: ``u^T B_k R_k v / u^T v``.
    Negative = stabilising. Contributions sum to the first-order total.
    """
    E, I = net.E, net.I
    b = _blocks(g * W, E, I)
    R = np.linalg.solve(np.eye(int(I.sum())) - b["i_to_i"], b["e_to_i"])
    B = b["i_to_e"]
    vals, right = np.linalg.eig(b["e_to_e"])
    k = int(np.argmax(vals.real))
    v = right[:, k]
    vals_t, left = np.linalg.eig(b["e_to_e"].T)
    u = left[:, int(np.argmin(np.abs(vals_t - vals[k])))]
    denom = u @ v
    shift = np.array([(u @ B[:, j]) * (R[j, :] @ v) / denom for j in range(B.shape[1])])
    i_labels = [x for x, keep in zip(net.labels, I) if keep]
    out = pd.DataFrame(
        {
            "first_order_shift": shift.real,
            "direct_loop_gain": [((u @ B[:, j]) * (b["e_to_i"][j, :] @ v) / denom).real for j in range(B.shape[1])],
        },
        index=pd.Index(i_labels, name="cell_type"),
    )
    out["i_class"] = net.meta.loc[out.index, "i_class"]
    out["region"] = net.meta.loc[out.index, "region"]
    out.attrs["lambda_EE"] = float(vals[k].real)
    out.attrs["alpha_E_eff_exact"] = abscissa(inhibitory_schur_complement(W, E, I, g))
    out.attrs["first_order_total"] = float(shift.real.sum())
    return out.sort_values("first_order_shift")


# ==========================================================================
# Phase 3 - paradoxical responses
# ==========================================================================


def response_matrix(W: np.ndarray, g: float, check: bool = True) -> np.ndarray:
    """``Gamma = (I - gW)^-1``; refuses unstable or near-singular cases."""
    A = np.eye(len(W)) - g * W
    if check:
        if abscissa(g * W) >= 1:
            raise ValueError(f"g = {g} is outside the stable range; Gamma is not a steady state.")
        cond = np.linalg.cond(A)
        if cond > 1e12:
            raise ValueError(f"I - gW is near-singular (cond = {cond:.2e}).")
    return np.linalg.inv(A)


def self_response_sweep(net: Network, W: np.ndarray, gains: Iterable[float]) -> pd.DataFrame:
    """``Gamma_kk`` for every type at every gain (long format)."""
    rows = []
    for g in gains:
        d = np.diag(response_matrix(W, g))
        for name, val, ei in zip(net.labels, d, net.meta.ei):
            rows.append({"g": g, "cell_type": name, "ei": ei, "Gamma_kk": val})
    return pd.DataFrame(rows)


def paradox_onsets(
    net: Network, W: np.ndarray, types: Sequence[str] | None = None, margin: float = 0.01
) -> pd.DataFrame:
    """Exact gain at which each type turns paradoxical (from the clamped spectrum).

    ``Gamma_kk(g) < 0`` iff an odd number of real eigenvalues of ``W_-k`` exceed
    ``1/g``. So the onset is ``1/lambda_1`` (largest real eigenvalue of
    ``W_-k``) and it counts only if it is below ``g_full``.

    Near criticality ``Gamma ~ v u^T / (1 - g/g_full)``, so *every* type with
    ``u_k v_k < 0`` turns paradoxical in a sliver just below ``g_full``. That
    is a trivial consequence of approaching instability, not a property of the
    type. ``robust_paradox`` therefore requires ``onset < (1 - margin) g_full``.
    """
    g_full = _crit_gain(abscissa(W))
    types = list(types) if types is not None else [x for x, i in zip(net.labels, net.I) if i]
    rows = []
    for name in types:
        keep = np.ones(net.n, bool)
        keep[net.idx([name])] = False
        ev = np.linalg.eigvals(W[np.ix_(keep, keep)])
        real_pos = np.sort(ev.real[(np.abs(ev.imag) < 1e-10) & (ev.real > 0)])[::-1]
        onset = 1.0 / real_pos[0] if real_pos.size else np.inf
        n_before_full = int(np.sum(real_pos > 1.0 / g_full)) if np.isfinite(g_full) else 0
        rows.append(
            {
                "cell_type": name,
                "onset_gain": onset,
                "onset_over_g_full": onset / g_full,
                "paradoxical_before_g_full": onset < g_full,
                "robust_paradox": onset < (1.0 - margin) * g_full,
                "n_crossings_before_g_full": n_before_full,
            }
        )
    out = pd.DataFrame(rows).set_index("cell_type")
    out = out.join(net.meta[["region", "i_class"]])
    return out.sort_values("onset_gain")


def verify_cramer(net: Network, W: np.ndarray, gains: Iterable[float]) -> pd.DataFrame:
    """Check ``Gamma_kk < 0  <=>  odd real crossings of g W_-k`` for every I type."""
    rows = []
    i_idx = np.flatnonzero(net.I)
    for g in gains:
        G = response_matrix(W, g)
        mismatches = 0
        for k in i_idx:
            keep = np.arange(net.n) != k
            pred = _real_eigs_above(g * W[np.ix_(keep, keep)]) % 2 == 1
            mismatches += pred != (G[k, k] < 0)
        rows.append({"g": g, "n_checked": len(i_idx), "mismatches": mismatches})
    out = pd.DataFrame(rows)
    if out.mismatches.sum():
        raise AssertionError(f"Cramer criterion failed:\n{out}")
    return out


def _drive(net: Network, members: Sequence[str], weights: np.ndarray | None = None) -> np.ndarray:
    b = np.zeros(net.n)
    idx = net.idx(members)
    b[idx] = 1.0 if weights is None else weights[idx]
    return b


def class_response_table(
    net: Network,
    W: np.ndarray,
    g: float,
    sets: Mapping[str, Sequence[str]],
    basis: str = "uniform",
) -> pd.DataFrame:
    """Drive each set, report the set's own response.

    ``basis='uniform'`` drives every member equally; ``'census'`` scales the
    drive by ``q``. Class sums are basis-dependent, so always report the basis.
    """
    G = response_matrix(W, g)
    weights = None
    if basis == "census":
        weights = np.nan_to_num(net.q / np.nanmedian(net.q), nan=1.0)
    elif basis != "uniform":
        raise ValueError("basis must be 'uniform' or 'census'")
    rows = []
    for name, members in sets.items():
        idx = net.idx(members)
        r = G @ _drive(net, members, weights)
        own = r[idx]
        rows.append(
            {
                "set": name,
                "n_types": len(members),
                "own_response_sum": own.sum(),
                "own_response_mean": own.mean(),
                "n_members_negative": int((own < 0).sum()),
                "E_response_sum": r[net.E].sum(),
                "class_paradox": own.sum() < 0,
            }
        )
    out = pd.DataFrame(rows).set_index("set").sort_values("own_response_sum")
    out.attrs.update({"g": g, "basis": basis})
    return out


def path_decomposition(
    net: Network, W: np.ndarray, g: float, sets: Mapping[str, Sequence[str]]
) -> pd.DataFrame:
    """Split each set's own response into direct + I->I + E-loop parts.

    - ``direct``: the drive itself (no recurrence).
    - ``via_I_to_I``: inhibitory-only network (E loop cut) minus direct.
    - ``via_E_loop``: full network minus inhibitory-only network.

    A negative total with a non-negative inhibitory-only response is an
    ISN-type paradox (it needs the E loop). A negative inhibitory-only response
    is disinhibition and would be there without any excitatory feedback.
    """
    G_full = response_matrix(W, g)
    W_cut = W.copy()
    W_cut[np.ix_(net.I, net.E)] = 0.0  # cut e_to_i: inhibition no longer sees E
    G_ionly = response_matrix(W_cut, g, check=False)
    rows = []
    for name, members in sets.items():
        idx = net.idx(members)
        b = _drive(net, members)
        direct = b[idx].sum()
        i_only = (G_ionly @ b)[idx].sum()
        total = (G_full @ b)[idx].sum()
        if total >= 0:
            verdict = "not paradoxical"
        elif i_only < 0:
            verdict = "disinhibition (no E loop needed)"
        else:
            verdict = "ISN-type paradox (needs E loop)"
        rows.append(
            {
                "set": name,
                "direct": direct,
                "via_I_to_I": i_only - direct,
                "via_E_loop": total - i_only,
                "total": total,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows).set_index("set").sort_values("total")


def two_population_reduction(net: Network, W: np.ndarray, g: float) -> pd.Series:
    """Collapse to a 2x2 E/I model (uniform weighting) and test the classic conditions.

    ``w_ab`` = mean over targets in a of the summed input from all of b.
    Paradox in 2-pop needs ``g w_EE > 1``; stability then needs
    ``g^2 |w_ItoE| w_EtoI > (g w_EE - 1)(1 + g |w_ItoI|)`` (plus trace < 0).
    """
    E, I = net.E, net.I
    w = {
        "w_EtoE": W[np.ix_(E, E)].sum(1).mean(),
        "w_ItoE": W[np.ix_(E, I)].sum(1).mean(),
        "w_EtoI": W[np.ix_(I, E)].sum(1).mean(),
        "w_ItoI": W[np.ix_(I, I)].sum(1).mean(),
    }
    W2 = g * np.array([[w["w_EtoE"], w["w_ItoE"]], [w["w_EtoI"], w["w_ItoI"]]])
    A = -np.eye(2) + W2
    loop = g * g * abs(w["w_ItoE"]) * w["w_EtoI"]
    need = (g * w["w_EtoE"] - 1) * (1 + g * abs(w["w_ItoI"]))
    out = pd.Series(
        {
            **w,
            "g": g,
            "E_alone_unstable (g w_EE > 1)": g * w["w_EtoE"] > 1,
            "E-I-E loop gain": loop,
            "loop gain needed": need,
            "trace<0": np.trace(A) < 0,
            "det>0": np.linalg.det(A) > 0,
            "2pop stable": abscissa(A) < 0,
            "2pop ISN & paradox": (abscissa(A) < 0) and (g * w["w_EtoE"] > 1),
        }
    )
    return out


def step_response(
    net: Network,
    W: np.ndarray,
    g: float,
    members: Sequence[str],
    tau: float | np.ndarray = 1.0,
    t_max: float | None = None,
    dt: float = 0.05,
) -> pd.DataFrame:
    """Exact linear response to a unit step of drive onto ``members`` at t = 0.

    Returns the summed response of the driven set and of all E types over time.
    ``t_max`` defaults to 6 slowest-mode time constants, ``6 / |alpha(J)|``;
    near ``g_full`` that is long (50 tau per time constant at g = 0.98).
    """
    tau_v = np.broadcast_to(np.asarray(tau, float), (net.n,))
    J = (-np.eye(net.n) + g * W) / tau_v[:, None]
    if t_max is None:
        t_max = 6.0 / abs(abscissa(J))
    x_ss = response_matrix(W, g) @ _drive(net, members)
    P = expm(J * dt)
    idx = net.idx(members)
    x = np.zeros(net.n)
    rows = []
    for step in range(int(round(t_max / dt)) + 1):
        rows.append({"t": step * dt, "driven_set": x[idx].sum(), "all_E": x[net.E].sum()})
        x = x_ss + P @ (x - x_ss)
    return pd.DataFrame(rows)


def _signature(ts: pd.DataFrame, col: str = "driven_set") -> dict[str, float | bool | str]:
    y = ts[col].to_numpy()
    t = ts["t"].to_numpy()
    k_peak = int(np.argmax(y))
    final = y[-1]
    crossing = np.nan
    if y[k_peak] > 0 and final < 0:
        after = np.flatnonzero(y[k_peak:] < 0)
        crossing = float(t[k_peak + after[0]]) if after.size else np.nan
    shape = "rise-then-dip (ISN signature)" if (y[k_peak] > 0 and final < 0) else (
        "monotone negative" if final < 0 else "positive"
    )
    return {"peak": float(y[k_peak]), "t_peak": float(t[k_peak]), "final": float(final),
            "t_zero_cross": crossing, "shape": shape}


def step_response_table(
    net: Network, W: np.ndarray, g: float, sets: Mapping[str, Sequence[str]], **kw
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    traces = {name: step_response(net, W, g, members, **kw) for name, members in sets.items()}
    table = pd.DataFrame({name: _signature(ts) for name, ts in traces.items()}).T
    return table, traces


# ==========================================================================
# Phase 4 - robustness
# ==========================================================================


def tau_ratio_sweep(net: Network, W: np.ndarray, g: float, ratios: Iterable[float]) -> pd.DataFrame:
    """Stability of ``J = T^-1(-I + gW)`` with ``tau_E = 1``, ``tau_I = ratio``.

    Steady-state ``Gamma`` does not depend on tau, but stability can (Hopf):
    slow inhibition can make a paradoxical steady state unreachable.
    """
    rows = []
    for r in ratios:
        tau = np.where(net.I, r, 1.0)
        J = jc.build_jacobian(g * W, tau=tau)
        ev = np.linalg.eigvals(J)
        k = int(np.argmax(ev.real))
        rows.append({"tau_I_over_tau_E": r, "alpha_J": ev.real[k], "imag_at_alpha": abs(ev.imag[k]),
                     "stable": ev.real[k] < 0})
    return pd.DataFrame(rows)


def surrogate_operators(net: Network, variant: str = "with_self") -> dict[str, np.ndarray]:
    """Weights-vs-topology surrogates (docs/WEIGHTS_VS_TOPOLOGY.md), unit rho.

    ``W[post, pre] = diag(kappa) B^T diag(q)``: divide rows by kappa, columns by q.
    """
    W = jc.matrix_variants(net.data.W_raw)[variant]
    kap = net.kappa[:, None]
    q = net.q[None, :]
    out = {"M": W, "binary_sign": np.sign(W), "M/kappa": W / kap}
    if np.all(np.isfinite(net.q)):
        out["M/q"] = W / q
        out["M/(kappa q)"] = W / kap / q
    return {k: _unit_rho(v) for k, v in out.items()}


def isn_profile(net: Network, W: np.ndarray, margin: float = 0.01) -> dict[str, float | int | str]:
    """Normalisation-free summary: window ratio and (robustly) paradoxical set."""
    win = isn_window(W, net.E)
    on = paradox_onsets(net, W, margin=margin)
    para = on[on.robust_paradox]
    return {
        **win,
        "g_E_over_g_full": win["g_E"] / win["g_full"],
        "n_paradoxical_any": int(on.paradoxical_before_g_full.sum()),
        "n_paradoxical_robust": int(len(para)),
        "first_paradoxical": para.index[0] if len(para) else "",
        "first_onset_over_g_full": float(para.onset_over_g_full.iloc[0]) if len(para) else np.nan,
    }


def profile_table(net: Network, operators: Mapping[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame({name: isn_profile(net, W) for name, W in operators.items()}).T


def jitter_ensemble(
    net: Network,
    variant: str = "with_self",
    sigma_kappa: float = 0.3,
    sigma_q: float = 0.0,
    n: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Lognormal jitter of kappa (rows) and/or q (columns); profile per draw."""
    rng = np.random.default_rng(seed)
    W0 = jc.matrix_variants(net.data.W_raw)[variant]
    rows = []
    for _ in range(n):
        rk = np.exp(sigma_kappa * rng.standard_normal(net.n))[:, None]
        rq = np.exp(sigma_q * rng.standard_normal(net.n))[None, :]
        prof = isn_profile(net, _unit_rho(W0 * rk * rq))
        rows.append(prof)
    return pd.DataFrame(rows)


def _rewire_fn() -> Callable:
    path = str(REPO / "motif_analysis")
    if path not in sys.path:
        sys.path.insert(0, path)
    from weighted_motif_null import rewire_degree_and_dyad_preserving  # noqa: E402

    return rewire_degree_and_dyad_preserving


def _null_draw(W: np.ndarray, E: np.ndarray, I: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    n = len(W)
    off = ~np.eye(n, dtype=bool)
    if kind == "block_weight_shuffle":
        # Fixed topology and signs; shuffle magnitudes within each E/I block.
        out = W.copy()
        for rows, cols in [(E, E), (I, E), (E, I), (I, I)]:
            m = np.zeros((n, n), bool)
            m[np.ix_(rows, cols)] = True
            m &= off & (W != 0)
            out[m] = rng.permutation(W[m])
        return out
    if kind == "degree_dyad_rewire":
        # Degree + dyad census preserved (motif_analysis null); Dale's law
        # re-imposed from the source column; diagonal kept as observed.
        absW = np.abs(W) * off
        new = _rewire_fn()(absW, rng)
        sign = np.where(E, 1.0, -1.0)[None, :]
        out = np.abs(new) * sign
        out[np.diag_indices(n)] = np.diag(W)
        return out
    raise ValueError("kind must be 'block_weight_shuffle' or 'degree_dyad_rewire'")


def null_ensemble(
    net: Network, variant: str = "with_self", kind: str = "block_weight_shuffle", n: int = 50, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    W0 = jc.matrix_variants(net.data.W_raw)[variant]
    return pd.DataFrame([isn_profile(net, _unit_rho(_null_draw(W0, net.E, net.I, kind, rng))) for _ in range(n)])


# ==========================================================================
# Phase 5 - nonlinear / state dependence
# ==========================================================================


def rate_dependent_isn(
    net: Network, W: np.ndarray, g: float, rate_scales: Iterable[float], ratio_I_to_E: float = 10.0, power: float = 2.0
) -> pd.DataFrame:
    """ISN status vs baseline activity for a power-law transfer ``f(u) = u^n``.

    The local gain is ``f'(u0) = n r0^((n-1)/n)``, applied per *postsynaptic*
    row. kappa is taken as the gain at a reference rate r_ref = 1 for every
    type, so the effective operator is ``diag((r0 / 1)^((n-1)/n)) gW``.
    Baselines: ``r0_E = s``, ``r0_I = ratio * s``. With heterogeneous rates the
    E/I gain balance moves with activity, which uniform g scaling cannot do.
    """
    rows = []
    expo = (power - 1.0) / power
    for s in rate_scales:
        r0 = np.where(net.I, ratio_I_to_E * s, s)
        Weff = (r0 ** expo)[:, None] * (g * W)
        a_full = abscissa(Weff)
        a_ee = abscissa(Weff[np.ix_(net.E, net.E)])
        rows.append({"rate_scale": s, "alpha_full": a_full, "alpha_EE": a_ee,
                     "stable": a_full < 1, "ISN": a_full < 1 < a_ee})
    return pd.DataFrame(rows)


def simulate_rectified(
    net: Network,
    W: np.ndarray,
    g: float,
    baseline: np.ndarray,
    extra_drive: np.ndarray,
    tau: float | np.ndarray = 1.0,
    t_max: float | None = None,
    dt: float = 0.05,
) -> np.ndarray:
    """Threshold-linear rate model ``tau dr/dt = -r + [gWr + h]_+`` (forward Euler).

    ``h`` is chosen so that ``baseline`` is a fixed point; ``extra_drive`` is
    switched on at t = 0. Returns r(t) as an array (time x type). ``t_max``
    defaults to 8 slowest-mode time constants of the linearisation.
    """
    tau_v = np.broadcast_to(np.asarray(tau, float), (net.n,))
    if t_max is None:
        t_max = 8.0 / abs(abscissa((-np.eye(net.n) + g * W) / tau_v[:, None]))
    h0 = baseline - g * W @ baseline
    r = baseline.copy()
    out = np.empty((int(round(t_max / dt)) + 1, net.n))
    for k in range(out.shape[0]):
        out[k] = r
        r = r + dt / tau_v * (-r + np.maximum(g * W @ r + h0 + extra_drive, 0.0))
    return out


def nonlinear_step_table(
    net: Network,
    W: np.ndarray,
    g: float,
    sets: Mapping[str, Sequence[str]],
    amplitudes: Iterable[float],
    baseline: np.ndarray | None = None,
) -> pd.DataFrame:
    """Steady-state change of each driven set in the rectified model vs linear prediction."""
    base = np.ones(net.n) if baseline is None else baseline
    G = response_matrix(W, g)
    rows = []
    for name, members in sets.items():
        idx = net.idx(members)
        b = _drive(net, members)
        for a in amplitudes:
            traj = simulate_rectified(net, W, g, base, a * b)
            delta = traj[-1] - base
            rows.append(
                {
                    "set": name,
                    "amplitude": a,
                    "nonlinear_delta": delta[idx].sum(),
                    "linear_delta": a * (G @ b)[idx].sum(),
                    "n_types_silenced": int((traj[-1] <= 1e-9).sum()),
                    "max_rate_over_baseline": float(np.max(traj[-1] / base)),
                    # relative: last-stretch drift small compared with the response itself
                    "converged": bool(
                        np.max(np.abs(traj[-1] - traj[-50])) <= 1e-3 * max(1e-12, np.max(np.abs(delta)))
                    ),
                }
            )
    return pd.DataFrame(rows)


# ==========================================================================
# Phase 6 - predictions
# ==========================================================================


def prediction_table(
    onsets: pd.DataFrame, decomposition: pd.DataFrame, steps: pd.DataFrame, g_ref: float
) -> pd.DataFrame:
    """One row per robustly paradoxical type: onset, status at ``g_ref``, mechanism, transient.

    A type with a robust onset can still be non-paradoxical at the reference
    gain (onset above ``g_ref``); only types with ``onset < g_ref`` become predictions.
    """
    para = onsets[onsets.robust_paradox].copy()
    para["paradoxical_at_g_ref"] = para.onset_gain < g_ref
    para["mechanism"] = [decomposition["verdict"].get(x, "") for x in para.index]
    para["step_shape"] = [steps["shape"].get(x, "") for x in para.index]
    para["experiment"] = [
        f"Optogenetic activation of {x}; record {x}: predict net decrease after a transient rise"
        if at_ref else f"Not paradoxical at g = {g_ref}; becomes so above g = {on:.3f}"
        for x, at_ref, on in zip(para.index, para.paradoxical_at_g_ref, para.onset_gain)
    ]
    return para


# ==========================================================================
# Plots
# ==========================================================================


def _plt():
    import matplotlib.pyplot as plt

    return plt


def plot_gain_sweep(sweeps: Mapping[str, pd.DataFrame], marks: Sequence[float] = (0.85, 0.95), ax=None):
    plt = _plt()
    n = len(sweeps)
    if ax is None:
        _, ax = plt.subplots(1, n, figsize=(6 * n, 4), sharey=True, squeeze=False)
        ax = ax[0]
    for a, (name, df) in zip(ax, sweeps.items()):
        a.plot(df.g, df.alpha_full, label=r"$\alpha(gW)$ full")
        a.plot(df.g, df.alpha_EE, label=r"$\alpha(gW_{EE})$ E only")
        a.plot(df.g, df.alpha_E_eff, "--", label=r"$\alpha(W_{eff})$ inhibition eliminated")
        a.axhline(1, color="k", lw=0.8)
        isn = df[df.ISN]
        if len(isn):
            a.axvspan(isn.g.min(), isn.g.max(), color="tab:green", alpha=0.15, label="ISN window")
        for m in marks:
            a.axvline(m, color="grey", ls=":", lw=1)
        a.set_title(name)
        a.set_xlabel("global gain g")
    ax[0].set_ylabel("max Re eigenvalue")
    ax[0].legend(fontsize=8)
    return ax


def plot_onsets(onsets: pd.DataFrame, top: int = 15, ax=None):
    plt = _plt()
    df = onsets[np.isfinite(onsets.onset_over_g_full)].head(top).iloc[::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.3 * len(df) + 1))
    colors = ["tab:red" if p else "tab:grey" for p in df.paradoxical_before_g_full]
    ax.barh(df.index, df.onset_over_g_full, color=colors)
    ax.axvline(1, color="k", lw=0.8)
    ax.set_xlabel(r"paradox onset gain / $g_{full}$  (red: < 1, reachable)")
    return ax


def plot_step_responses(traces: Mapping[str, pd.DataFrame], normalize: bool = True, ax=None):
    """Driven-set response vs log time; ``normalize`` scales each trace to its peak |value|."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for name, ts in traces.items():
        ts = ts[ts.t > 0]
        y = ts.driven_set / ts.driven_set.abs().max() if normalize else ts.driven_set
        ax.plot(ts.t, y, label=name)
    ax.set_xscale("log")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"time ($\tau$)")
    ax.set_ylabel("response of driven set" + (" / peak |response|" if normalize else ""))
    ax.legend(fontsize=7)
    return ax


def plot_tau_ratio(df: pd.DataFrame, ax=None):
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogx(df.tau_I_over_tau_E, df.alpha_J, marker="o")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\tau_I / \tau_E$")
    ax.set_ylabel(r"$\alpha(J)$")
    return ax


def plot_null(null: pd.DataFrame, observed: Mapping[str, float], column: str, ax=None):
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    vals = null[column].replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(vals, bins=20, color="lightgrey", edgecolor="grey")
    ax.axvline(observed[column], color="tab:red", lw=2, label="observed")
    ax.set_xlabel(column)
    ax.legend()
    return ax


def empirical_p(null: pd.Series, observed: float, tail: str = "two") -> float:
    """Empirical p-value with the +1 correction."""
    x = null.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    lo = (np.sum(x <= observed) + 1) / (len(x) + 1)
    hi = (np.sum(x >= observed) + 1) / (len(x) + 1)
    return {"lower": lo, "upper": hi, "two": min(1.0, 2 * min(lo, hi))}[tail]
