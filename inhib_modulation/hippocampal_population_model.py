
"""
hippocampal_population_model.py

Data-driven 85-population hippocampal ODE model.

Model
-----
For postsynaptic population j:

    tau_m,j * dr_j/dt
        = -r_j + F( sum_i m_ij r_i + I_j(t) )

where:
    r_i(t)   = population activity state
    tau_m,j  = membrane/population time constant (ms -> s)
    m_ij     = effective coupling already supplied in mij_netlist.csv
               and defined upstream as k_j * w_ij
    I_j(t)   = external drive

Important
---------
tau_d,ij is already folded into w_ij and therefore into m_ij in the supplied
netlist. It is NOT applied again in this reduced ODE model.

This script:
  1. loads mij_netlist.csv
  2. loads tau_m_85_updated.csv
  3. constructs M[post, pre]
  4. simulates the coupled ODE system
  5. supports pulse and sinusoidal inputs
  6. provides selected-population, heatmap, region, E/I, eigenvalue,
     and coupling-sweep plots

The raw m_ij matrix may be numerically large depending on the original activity
normalization. For exploratory work, this script supports optional spectral
normalization. This normalization is a modeling assumption, not a biological
parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------

@dataclass
class ModelData:
    cell_types: list[str]
    index: dict[str, int]
    mij: pd.DataFrame
    tau: pd.DataFrame
    M_raw: np.ndarray
    tau_ms: np.ndarray
    ei_map: dict[str, str]


# ---------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------

def load_model_data(
    mij_path: str | Path = "mij_netlist.csv",
    tau_path: str | Path = "tau_m_85_updated.csv",
    unresolved_tau: str = "median",
) -> ModelData:
    """
    Load the effective-coupling netlist and cell-type membrane time constants.

    Parameters
    ----------
    mij_path
        Path to mij_netlist.csv.
    tau_path
        Path to tau_m_85_updated.csv.
    unresolved_tau
        How to handle missing tau_m values:
          - "median": fill missing values with median measured tau_m
          - "error": raise an exception if any are missing

    Returns
    -------
    ModelData
    """
    mij = pd.read_csv(mij_path)
    tau = pd.read_csv(tau_path)

    required_m = {
        "pre_neuron", "post_neuron", "pre_ei", "post_ei", "m_ij"
    }
    missing_cols = required_m - set(mij.columns)
    if missing_cols:
        raise ValueError(f"mij netlist missing columns: {sorted(missing_cols)}")

    required_tau = {"neuron_type", "tau_m_ms"}
    missing_cols = required_tau - set(tau.columns)
    if missing_cols:
        raise ValueError(f"tau table missing columns: {sorted(missing_cols)}")

    # Preserve a deterministic ordering.
    cell_types = sorted(set(mij["pre_neuron"]) | set(mij["post_neuron"]))
    index = {name: i for i, name in enumerate(cell_types)}

    # Build M[post, pre].
    n = len(cell_types)
    M_raw = np.zeros((n, n), dtype=float)

    for row in mij.itertuples(index=False):
        i = index[row.pre_neuron]
        j = index[row.post_neuron]
        M_raw[j, i] = float(row.m_ij)

    tau_map = dict(zip(tau["neuron_type"], tau["tau_m_ms"]))
    tau_ms = np.array([tau_map.get(name, np.nan) for name in cell_types], dtype=float)

    if np.isnan(tau_ms).any():
        missing_types = [cell_types[i] for i in np.where(np.isnan(tau_ms))[0]]

        if unresolved_tau == "error":
            raise ValueError(
                "Missing tau_m values for:\n  " + "\n  ".join(missing_types)
            )

        if unresolved_tau == "median":
            median_tau = float(np.nanmedian(tau_ms))
            tau_ms[np.isnan(tau_ms)] = median_tau
            print(
                f"[warning] Filled {len(missing_types)} unresolved tau_m values "
                f"with median tau_m = {median_tau:.3f} ms."
            )
            print("[warning] These values should be sensitivity-tested.")
        else:
            raise ValueError("unresolved_tau must be 'median' or 'error'")

    ei_map = {}
    for _, row in mij[["pre_neuron", "pre_ei"]].drop_duplicates().iterrows():
        ei_map[row["pre_neuron"]] = row["pre_ei"]
    for _, row in mij[["post_neuron", "post_ei"]].drop_duplicates().iterrows():
        ei_map.setdefault(row["post_neuron"], row["post_ei"])

    return ModelData(
        cell_types=cell_types,
        index=index,
        mij=mij,
        tau=tau,
        M_raw=M_raw,
        tau_ms=tau_ms,
        ei_map=ei_map,
    )


# ---------------------------------------------------------------------
# Coupling matrix handling
# ---------------------------------------------------------------------

def spectral_radius(M: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def prepare_coupling(
    M_raw: np.ndarray,
    normalization: Optional[str] = "spectral",
    target_radius: float = 0.85,
    global_scale: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """
    Prepare the effective m_ij matrix for simulation.

    normalization:
      None        -> use raw m_ij
      "spectral"  -> scale so spectral radius == target_radius
      "max_abs"   -> divide by max absolute matrix element

    Returns
    -------
    M, metadata
    """
    M = np.asarray(M_raw, dtype=float).copy()
    rho_raw = spectral_radius(M)

    if normalization is None:
        factor = 1.0
    elif normalization == "spectral":
        factor = target_radius / rho_raw if rho_raw > 0 else 1.0
    elif normalization == "max_abs":
        denom = np.max(np.abs(M))
        factor = 1.0 / denom if denom > 0 else 1.0
    else:
        raise ValueError("normalization must be None, 'spectral', or 'max_abs'")

    M *= factor * global_scale

    info = {
        "raw_spectral_radius": rho_raw,
        "normalization": normalization,
        "normalization_factor": factor,
        "global_scale": global_scale,
        "final_spectral_radius": spectral_radius(M),
    }
    return M, info


# ---------------------------------------------------------------------
# Transfer functions
# ---------------------------------------------------------------------

def transfer_function(
    x: np.ndarray,
    kind: str = "relu",
    max_rate: Optional[float | np.ndarray] = None,
    sigmoid_gain: float = 1.0,
    sigmoid_midpoint: float = 0.0,
) -> np.ndarray:
    """
    Apply the population transfer function.

    Note:
        Because m_ij already contains k_j, k_j is NOT multiplied here.
    """
    x = np.asarray(x, dtype=float)

    if kind == "linear":
        y = x

    elif kind == "relu":
        y = np.maximum(x, 0.0)

    elif kind == "sigmoid":
        cap = 1.0 if max_rate is None else np.asarray(max_rate)
        z = np.clip(sigmoid_gain * (x - sigmoid_midpoint), -60, 60)
        return cap / (1.0 + np.exp(-z))

    else:
        raise ValueError("kind must be 'linear', 'relu', or 'sigmoid'")

    if max_rate is not None:
        y = np.minimum(y, np.asarray(max_rate))

    return y


# ---------------------------------------------------------------------
# External drives
# ---------------------------------------------------------------------

def zero_drive(n: int) -> Callable[[float], np.ndarray]:
    def drive(t: float) -> np.ndarray:
        return np.zeros(n)
    return drive


def pulse_drive(
    cell_types: Sequence[str],
    targets: str | Sequence[str],
    amplitude: float = 1.0,
    start_ms: float = 0.0,
    duration_ms: float = 50.0,
) -> Callable[[float], np.ndarray]:
    if isinstance(targets, str):
        targets = [targets]

    idx = {name: i for i, name in enumerate(cell_types)}
    target_ids = [idx[name] for name in targets]

    n = len(cell_types)
    start_s = start_ms / 1000.0
    stop_s = (start_ms + duration_ms) / 1000.0

    def drive(t: float) -> np.ndarray:
        u = np.zeros(n)
        if start_s <= t <= stop_s:
            u[target_ids] = amplitude
        return u

    return drive


def sinusoidal_drive(
    cell_types: Sequence[str],
    targets: str | Sequence[str],
    amplitude: float = 1.0,
    frequency_hz: float = 8.0,
    baseline: float = 0.0,
    start_ms: float = 0.0,
    stop_ms: Optional[float] = None,
) -> Callable[[float], np.ndarray]:
    if isinstance(targets, str):
        targets = [targets]

    idx = {name: i for i, name in enumerate(cell_types)}
    target_ids = [idx[name] for name in targets]

    n = len(cell_types)
    start_s = start_ms / 1000.0
    stop_s = np.inf if stop_ms is None else stop_ms / 1000.0

    def drive(t: float) -> np.ndarray:
        u = np.zeros(n)
        if start_s <= t <= stop_s:
            val = baseline + amplitude * 0.5 * (
                1.0 + np.sin(2.0 * np.pi * frequency_hz * t)
            )
            u[target_ids] = val
        return u

    return drive


def composite_drive(
    *drives: Callable[[float], np.ndarray],
) -> Callable[[float], np.ndarray]:
    def drive(t: float) -> np.ndarray:
        return np.sum([d(t) for d in drives], axis=0)
    return drive


# ---------------------------------------------------------------------
# ODE simulation
# ---------------------------------------------------------------------

def simulate(
    M: np.ndarray,
    tau_ms: Sequence[float],
    cell_types: Sequence[str],
    drive: Optional[Callable[[float], np.ndarray]] = None,
    t_span_ms: tuple[float, float] = (0.0, 500.0),
    dt_ms: float = 0.25,
    r0: Optional[Sequence[float]] = None,
    transfer_kind: str = "relu",
    max_rate: Optional[float | np.ndarray] = None,
    sigmoid_gain: float = 1.0,
    sigmoid_midpoint: float = 0.0,
    method: str = "RK45",
) -> pd.DataFrame:
    """
    Simulate:

        tau_m,j * dr_j/dt
            = -r_j + F( sum_i m_ij r_i + I_j(t) )

    tau_ms is converted internally to seconds.
    """
    n = len(cell_types)

    M = np.asarray(M, dtype=float)
    tau_s = np.asarray(tau_ms, dtype=float) / 1000.0

    if M.shape != (n, n):
        raise ValueError(f"M shape must be {(n, n)}, got {M.shape}")

    if tau_s.shape != (n,):
        raise ValueError(f"tau_ms must have length {n}")

    if np.any(tau_s <= 0):
        raise ValueError("All tau_m values must be positive")

    if drive is None:
        drive = zero_drive(n)

    if r0 is None:
        r0 = np.zeros(n)
    else:
        r0 = np.asarray(r0, dtype=float)

    def rhs(t, r):
        recurrent_input = M @ r
        total_input = recurrent_input + drive(t)

        target_activity = transfer_function(
            total_input,
            kind=transfer_kind,
            max_rate=max_rate,
            sigmoid_gain=sigmoid_gain,
            sigmoid_midpoint=sigmoid_midpoint,
        )

        return (-r + target_activity) / tau_s

    t0_s, t1_s = np.asarray(t_span_ms) / 1000.0
    t_eval = np.arange(
        t0_s,
        t1_s + 0.5 * dt_ms / 1000.0,
        dt_ms / 1000.0,
    )

    sol = solve_ivp(
        rhs,
        (t0_s, t1_s),
        r0,
        t_eval=t_eval,
        method=method,
        rtol=1e-7,
        atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    out = pd.DataFrame(sol.y.T, columns=list(cell_types))
    out.insert(0, "time_ms", sol.t * 1000.0)

    return out


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def linear_jacobian(
    M: np.ndarray,
    tau_ms: Sequence[float],
) -> np.ndarray:
    """
    Jacobian for the linear active regime:

        tau * dr/dt = -r + M r

    giving:

        J = diag(1/tau) @ (M - I)
    """
    tau_s = np.asarray(tau_ms, dtype=float) / 1000.0
    return np.diag(1.0 / tau_s) @ (M - np.eye(M.shape[0]))


def stability_summary(
    M: np.ndarray,
    tau_ms: Sequence[float],
) -> dict:
    J = linear_jacobian(M, tau_ms)
    eig = np.linalg.eigvals(J)

    return {
        "max_real_eigenvalue": float(np.max(eig.real)),
        "stable_linear_regime": bool(np.max(eig.real) < 0),
        "jacobian_eigenvalues": eig,
    }


def region_of(name: str) -> str:
    if name.startswith("CA3c"):
        return "CA3"

    for prefix in ("DG", "CA1", "CA2", "CA3", "SUB", "MEC", "LEC", "EC"):
        if name.startswith(prefix):
            return prefix

    return "Other"


def aggregate_by_region(
    trajectories: pd.DataFrame,
    cell_types: Sequence[str],
) -> pd.DataFrame:
    groups: dict[str, list[str]] = {}

    for name in cell_types:
        groups.setdefault(region_of(name), []).append(name)

    out = pd.DataFrame({"time_ms": trajectories["time_ms"]})

    for region, names in groups.items():
        out[region] = trajectories[names].mean(axis=1)

    return out


def aggregate_ei(
    trajectories: pd.DataFrame,
    cell_types: Sequence[str],
    ei_map: Mapping[str, str],
) -> pd.DataFrame:
    out = pd.DataFrame({"time_ms": trajectories["time_ms"]})

    exc = [name for name in cell_types if ei_map.get(name, "").lower().startswith("e")]
    inh = [name for name in cell_types if ei_map.get(name, "").lower().startswith("i")]

    if exc:
        out["Excitatory"] = trajectories[exc].mean(axis=1)

    if inh:
        out["Inhibitory"] = trajectories[inh].mean(axis=1)

    return out


def coupling_sweep(
    M_base: np.ndarray,
    tau_ms: Sequence[float],
    cell_types: Sequence[str],
    drive: Callable[[float], np.ndarray],
    scales: Sequence[float],
    t_span_ms: tuple[float, float] = (0.0, 500.0),
    dt_ms: float = 0.5,
) -> pd.DataFrame:
    rows = []

    for scale in scales:
        M = scale * M_base

        traj = simulate(
            M,
            tau_ms,
            cell_types,
            drive=drive,
            t_span_ms=t_span_ms,
            dt_ms=dt_ms,
            transfer_kind="relu",
        )

        X = traj[list(cell_types)].to_numpy()

        stab = stability_summary(M, tau_ms)

        rows.append(
            {
                "scale": float(scale),
                "peak_activity": float(np.max(X)),
                "mean_final_activity": float(np.mean(X[-1])),
                "max_real_eigenvalue": stab["max_real_eigenvalue"],
                "stable_linear_regime": stab["stable_linear_regime"],
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_selected(
    trajectories: pd.DataFrame,
    selected: Sequence[str],
    title: str = "Selected population trajectories",
):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for name in selected:
        if name in trajectories.columns:
            ax.plot(trajectories["time_ms"], trajectories[name], label=name)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Population activity")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_heatmap(
    trajectories: pd.DataFrame,
    cell_types: Sequence[str],
    title: str = "All population activity",
):
    X = trajectories[list(cell_types)].to_numpy().T

    fig, ax = plt.subplots(figsize=(11, 10))

    im = ax.imshow(
        X,
        aspect="auto",
        origin="lower",
        extent=[
            trajectories["time_ms"].iloc[0],
            trajectories["time_ms"].iloc[-1],
            0,
            len(cell_types),
        ],
    )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron-type index")
    ax.set_title(title)

    fig.colorbar(im, ax=ax, label="Population activity")
    fig.tight_layout()
    return fig, ax


def plot_regions(
    region_df: pd.DataFrame,
    title: str = "Regional activity",
):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for col in region_df.columns:
        if col != "time_ms":
            ax.plot(region_df["time_ms"], region_df[col], label=col)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Mean activity")
    ax.set_title(title)
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig, ax


def plot_ei(
    ei_df: pd.DataFrame,
    title: str = "Excitatory vs inhibitory activity",
):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for col in ei_df.columns:
        if col != "time_ms":
            ax.plot(ei_df["time_ms"], ei_df[col], label=col)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Mean activity")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    return fig, ax


def plot_eigenvalues(
    M: np.ndarray,
    title: str = "Effective coupling eigenvalues",
):
    eig = np.linalg.eigvals(M)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(eig.real, eig.imag, s=25, alpha=0.8)

    ax.axvline(1.0, linestyle="--", linewidth=1.0)

    ax.set_xlabel("Real part")
    ax.set_ylabel("Imaginary part")
    ax.set_title(title)

    fig.tight_layout()
    return fig, ax


def plot_sweep(
    sweep_df: pd.DataFrame,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        sweep_df["scale"],
        sweep_df["peak_activity"],
        marker="o",
    )
    ax.set_xlabel("Global coupling scale")
    ax.set_ylabel("Peak activity")
    ax.set_title("Network response vs coupling strength")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------
# Example workflow
# ---------------------------------------------------------------------

def main():
    """
    Example simulation using a short DG Granule pulse.
    """

    data = load_model_data(
        mij_path="mij_netlist.csv",
        tau_path="tau_m_85_updated.csv",
        unresolved_tau="median",
    )

    print(f"Loaded {len(data.cell_types)} neuron types.")
    print(f"Loaded {len(data.mij)} directed edges.")

    M, info = prepare_coupling(
        data.M_raw,
        normalization="spectral",
        target_radius=0.85,
    )

    print("\nCoupling information:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    stab = stability_summary(M, data.tau_ms)

    print("\nLinearized stability:")
    print(f"  max Re(lambda): {stab['max_real_eigenvalue']:.6g}")
    print(f"  stable: {stab['stable_linear_regime']}")

    # -------------------------------------------------------------
    # Experiment 1: DG Granule pulse
    # -------------------------------------------------------------

    drive = pulse_drive(
        data.cell_types,
        targets="DG Granule",
        amplitude=1.0,
        start_ms=0.0,
        duration_ms=50.0,
    )

    traj = simulate(
        M,
        data.tau_ms,
        data.cell_types,
        drive=drive,
        t_span_ms=(0.0, 500.0),
        dt_ms=0.25,
        transfer_kind="relu",
    )

    traj.to_csv("hippocampal_trajectories.csv", index=False)

    selected = [
        "DG Granule",
        "DG Basket",
        "CA3 Pyramidal",
        "CA3 Basket",
        "CA2 Pyramidal",
        "CA1 Pyramidal",
    ]

    plot_selected(
        traj,
        selected,
        title="DG Granule pulse",
    )
    plt.savefig("selected_populations.png", dpi=180, bbox_inches="tight")
    plt.show()

    plot_heatmap(
        traj,
        data.cell_types,
        title="All 85 populations: DG pulse",
    )
    plt.savefig("activity_heatmap.png", dpi=180, bbox_inches="tight")
    plt.show()

    regions = aggregate_by_region(traj, data.cell_types)
    plot_regions(regions, title="Regional activity after DG pulse")
    plt.savefig("regional_activity.png", dpi=180, bbox_inches="tight")
    plt.show()

    ei = aggregate_ei(
        traj,
        data.cell_types,
        data.ei_map,
    )
    plot_ei(ei)
    plt.savefig("ei_activity.png", dpi=180, bbox_inches="tight")
    plt.show()

    plot_eigenvalues(M)
    plt.savefig("coupling_eigenvalues.png", dpi=180, bbox_inches="tight")
    plt.show()

    # -------------------------------------------------------------
    # Experiment 2: 8-Hz entorhinal input
    # -------------------------------------------------------------

    available_targets = [
        x for x in [
            "MEC LII Stellate",
            "EC LII III Pyramidal Tripolar",
        ]
        if x in data.index
    ]

    if available_targets:
        theta_drive = sinusoidal_drive(
            data.cell_types,
            targets=available_targets,
            amplitude=0.8,
            frequency_hz=8.0,
            baseline=0.05,
            start_ms=0.0,
            stop_ms=1500.0,
        )

        theta_traj = simulate(
            M,
            data.tau_ms,
            data.cell_types,
            drive=theta_drive,
            t_span_ms=(0.0, 1500.0),
            dt_ms=0.5,
            transfer_kind="relu",
        )

        theta_traj.to_csv(
            "hippocampal_theta_trajectories.csv",
            index=False,
        )

        plot_selected(
            theta_traj,
            [
                x for x in [
                    "MEC LII Stellate",
                    "DG Granule",
                    "CA3 Pyramidal",
                    "CA1 Pyramidal",
                ]
                if x in data.index
            ],
            title="8-Hz entorhinal drive",
        )
        plt.savefig(
            "theta_drive_selected.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.show()

    # -------------------------------------------------------------
    # Experiment 3: coupling-strength sweep
    # -------------------------------------------------------------

    scales = np.linspace(0.2, 1.3, 12)

    sweep = coupling_sweep(
        M,
        data.tau_ms,
        data.cell_types,
        drive=drive,
        scales=scales,
        t_span_ms=(0.0, 500.0),
        dt_ms=0.5,
    )

    sweep.to_csv("coupling_sweep.csv", index=False)

    plot_sweep(sweep)
    plt.savefig("coupling_sweep.png", dpi=180, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
