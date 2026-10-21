from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "02_inhibitory_modulation_analysis"
MATRIX_PATH = PROJECT_ROOT / "matrices" / "mij_matrix.csv"
NETLIST_PATH = PROJECT_ROOT / "matrices" / "mij_netlist.csv"
HELPER_PATH = PROJECT_ROOT / "notebooks" / "inhibitory_modulation.py"

N_NULL_REMOVAL = 250
RANDOM_STATE = 20260928


MOTIF_LABELS = {
    "self_connections": "SELF",
    "all_excitatory_edges_E_to_E": "E to E edges",
    "all_inhibitory_edges_I_to_I": "I to I edges",
    "all_excitatory_E_E_E_chains": "E E E chains",
    "all_inhibitory_I_I_I_chains": "I I I chains",
    "E_I_E_inhibitory_feedback": "E I E feedback",
    "E_I_I_E_disinhibitory_feedback": "E I I E disinhibition",
    "E_to_I_to_I_to_E_loop": "E I I E loop",
    "disinhibitory_I_to_I_to_E": "I I E disinhibition",
    "directed_two_step_chain": "two step chains",
    "divergent_one_source_two_receivers": "divergent",
    "convergent_two_sources_one_receiver": "convergent",
    "reciprocal_pair": "reciprocal pair",
}


PALETTE = {
    "with_self": "#1f4e79",
    "no_self": "#b35c1e",
}


def load_helper_module():
    spec = importlib.util.spec_from_file_location("inhibitory_modulation", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def clean_label(value: str) -> str:
    return MOTIF_LABELS.get(value, value.replace("_", " "))


def fmt_number(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def compute_removal_null_z() -> pd.DataFrame:
    """Compute matched random-edge null z-scores for motif-removal deltas.

    The notebook saves observed delta dominant-real values for each motif-removal
    edge set, but it does not save a matched null distribution for those deltas.
    This function builds one by removing random nonzero edge sets of the same
    size within each normalized variant.
    """

    im = load_helper_module()
    data = im.load_connectivity(
        MATRIX_PATH,
        matrix_orientation="pre_by_post",
        metadata_netlist_path=NETLIST_PATH,
    )
    variants = im.normalized_matrix_variants(data.matrix, method="spectral_radius", target=1.0)
    observed = pd.read_csv(OUTPUT_DIR / "motif_stability_by_variant.csv")
    rng = np.random.default_rng(RANDOM_STATE)
    rows: list[dict[str, object]] = []

    for variant, variant_rows in observed.groupby("variant", sort=False):
        matrix = variants[variant]["matrix"]
        arr = matrix.to_numpy(dtype=float)
        nonzero_positions = np.argwhere(arr != 0)
        base = im.spectral_summary(arr)["dominant_real"]
        cache: dict[int, np.ndarray] = {}

        for edge_count in sorted(variant_rows["edges_removed"].dropna().astype(int).unique()):
            if edge_count <= 0:
                cache[edge_count] = np.array([], dtype=float)
                continue
            deltas = []
            for _ in range(N_NULL_REMOVAL):
                selected = rng.choice(len(nonzero_positions), size=edge_count, replace=False)
                removed = arr.copy()
                coords = nonzero_positions[selected]
                removed[coords[:, 0], coords[:, 1]] = 0.0
                after = im.spectral_summary(removed)["dominant_real"]
                deltas.append(float(after - base))
            cache[edge_count] = np.array(deltas, dtype=float)

        for row in variant_rows.itertuples(index=False):
            edge_count = int(row.edges_removed)
            null = cache.get(edge_count, np.array([], dtype=float))
            if null.size < 2:
                null_mean = np.nan
                null_sd = np.nan
                null_z = np.nan
                p_more_stabilizing = np.nan
                p_more_destabilizing = np.nan
            else:
                null_mean = float(np.mean(null))
                null_sd = float(np.std(null, ddof=1))
                null_z = (
                    float((row.delta_dominant_real - null_mean) / null_sd)
                    if null_sd > np.finfo(float).eps
                    else np.nan
                )
                p_more_stabilizing = float((1 + np.sum(null <= row.delta_dominant_real)) / (len(null) + 1))
                p_more_destabilizing = float((1 + np.sum(null >= row.delta_dominant_real)) / (len(null) + 1))
            rows.append(
                {
                    "variant": row.variant,
                    "motif": row.motif,
                    "motif_label": clean_label(row.motif),
                    "edges_removed": edge_count,
                    "observed_delta_dominant_real": row.delta_dominant_real,
                    "null_mean_delta_dominant_real": null_mean,
                    "null_sd_delta_dominant_real": null_sd,
                    "null_z_delta_dominant_real": null_z,
                    "empirical_p_more_stabilizing": p_more_stabilizing,
                    "empirical_p_more_destabilizing": p_more_destabilizing,
                    "n_null": N_NULL_REMOVAL,
                    "null_model": "random nonzero edge sets matched on edges_removed within variant",
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "motif_removal_null_z_by_variant.csv", index=False)
    return out


def panel_svg(
    data: pd.DataFrame,
    value_col: str,
    label_col: str,
    title: str,
    x_label: str,
    *,
    x: int,
    y: int,
    width: int,
    label_width: int = 180,
    row_height: int = 34,
) -> tuple[str, int]:
    ordered_labels = list(dict.fromkeys(data[label_col].tolist()))
    values: list[float] = []
    by_variant = {}
    for variant in ("with_self", "no_self"):
        subset = data[data["variant"] == variant].set_index(label_col)
        series = {
            label: float(subset.loc[label, value_col])
            for label in ordered_labels
            if label in subset.index and np.isfinite(float(subset.loc[label, value_col]))
        }
        by_variant[variant] = series
        values.extend(series.values())

    max_abs = max([abs(v) for v in values] + [1.0])
    max_abs *= 1.12
    x_min, x_max = -max_abs, max_abs
    plot_x = x + label_width
    plot_w = width - label_width - 20
    top = y + 52
    panel_h = 86 + row_height * len(ordered_labels)
    zero_x = plot_x + ((0 - x_min) / (x_max - x_min)) * plot_w

    def sx(value: float) -> float:
        return plot_x + ((value - x_min) / (x_max - x_min)) * plot_w

    parts = [
        f'<text x="{x}" y="{y + 18}" font-size="16" font-weight="700" fill="#111111">{escape(title)}</text>',
        f'<text x="{x}" y="{y + 38}" font-size="11" fill="#555555">{escape(x_label)}</text>',
        f'<line x1="{zero_x:.1f}" y1="{top - 12}" x2="{zero_x:.1f}" y2="{top + row_height * len(ordered_labels)}" stroke="#333333" stroke-width="1"/>',
    ]

    for tick in (-max_abs, 0, max_abs):
        tx = sx(tick)
        parts.append(
            f'<line x1="{tx:.1f}" y1="{top - 8}" x2="{tx:.1f}" y2="{top + row_height * len(ordered_labels)}" stroke="#d9d9d9" stroke-width="0.8"/>'
        )
        parts.append(
            f'<text x="{tx:.1f}" y="{top + row_height * len(ordered_labels) + 18}" text-anchor="middle" font-size="10" fill="#555555">{escape(fmt_number(tick))}</text>'
        )

    bar_h = 10
    offsets = {"with_self": -6, "no_self": 7}
    for i, label in enumerate(ordered_labels):
        yy = top + i * row_height + 16
        parts.append(
            f'<text x="{x}" y="{yy + 4}" font-size="11" fill="#222222">{escape(str(label))}</text>'
        )
        for variant in ("with_self", "no_self"):
            if label not in by_variant[variant]:
                continue
            value = by_variant[variant][label]
            x0 = sx(min(0, value))
            x1 = sx(max(0, value))
            bar_y = yy + offsets[variant]
            parts.append(
                f'<rect x="{x0:.1f}" y="{bar_y:.1f}" width="{max(1, x1 - x0):.1f}" height="{bar_h}" fill="{PALETTE[variant]}"/>'
            )
            label_x = x1 + 4 if value >= 0 else x0 - 4
            anchor = "start" if value >= 0 else "end"
            parts.append(
                f'<text x="{label_x:.1f}" y="{bar_y + 8:.1f}" text-anchor="{anchor}" font-size="9.5" fill="#333333">{escape(fmt_number(value))}</text>'
            )

    legend_y = y + panel_h - 2
    legend_x = x + label_width
    for j, variant in enumerate(("with_self", "no_self")):
        lx = legend_x + j * 105
        parts.append(f'<rect x="{lx}" y="{legend_y}" width="12" height="12" fill="{PALETTE[variant]}"/>')
        parts.append(f'<text x="{lx + 18}" y="{legend_y + 10}" font-size="10.5" fill="#333333">{escape(variant)}</text>')
    return "\n".join(parts), panel_h


def write_svg(path: Path, width: int, height: int, body: str, title: str, note: str | None = None) -> Path:
    note_svg = ""
    if note:
        note_svg = f'<text x="24" y="{height - 22}" font-size="11" fill="#555555">{escape(note)}</text>'
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="24" y="34" font-size="20" font-weight="700" fill="#111111">{escape(title)}</text>
{body}
{note_svg}
</svg>
'''
    path.write_text(svg, encoding="utf-8")
    return path


def make_null_z_visual(removal_null: pd.DataFrame) -> Path:
    enrichment = pd.read_csv(OUTPUT_DIR / "motif_enrichment_by_variant.csv")
    enrichment = enrichment[np.isfinite(enrichment["z_score"])].copy()
    enrichment["motif_label"] = enrichment["motif"].map(clean_label)
    enrichment = enrichment.sort_values(["motif_label", "variant"])

    removal = removal_null[np.isfinite(removal_null["null_z_delta_dominant_real"])].copy()
    removal["sort_abs"] = removal["null_z_delta_dominant_real"].abs()
    removal_order = (
        removal.groupby("motif_label")["sort_abs"].max().sort_values(ascending=False).index.tolist()
    )
    removal["motif_label"] = pd.Categorical(removal["motif_label"], categories=removal_order, ordered=True)
    removal = removal.sort_values(["motif_label", "variant"])

    left_svg, left_h = panel_svg(
        enrichment,
        "z_score",
        "motif_label",
        "Motif enrichment null z score",
        "z score of observed count vs shuffled-weight null",
        x=24,
        y=62,
        width=610,
    )
    right_svg, right_h = panel_svg(
        removal,
        "null_z_delta_dominant_real",
        "motif_label",
        "Motif-removal null z score",
        "z score of observed delta real eigenvalue vs matched random edge removals",
        x=650,
        y=62,
        width=720,
    )
    path = OUTPUT_DIR / "motif_null_z_scores_enrichment_vs_removal.svg"
    note = (
        f"Removal null: {N_NULL_REMOVAL} random nonzero edge sets per variant and edge-count match. "
        "Finite enrichment z scores only; zero-variance count nulls omitted."
    )
    write_svg(path, 1400, max(left_h, right_h) + 110, left_svg + "\n" + right_svg, "Null-standardized motif evidence", note)
    return path


def make_delta_visual() -> Path:
    stability = pd.read_csv(OUTPUT_DIR / "motif_stability_by_variant.csv")
    stability["motif_label"] = stability["motif"].map(clean_label)
    stability["sort_abs"] = stability["delta_dominant_real"].abs()
    order = stability.groupby("motif_label")["sort_abs"].max().sort_values(ascending=False).index.tolist()
    stability["motif_label"] = pd.Categorical(stability["motif_label"], categories=order, ordered=True)
    stability = stability.sort_values(["motif_label", "variant"])

    body, panel_h = panel_svg(
        stability,
        "delta_dominant_real",
        "motif_label",
        "Motif-removal delta dominant real eigenvalue",
        "dominant real after removal minus before removal",
        x=24,
        y=62,
        width=940,
        label_width=205,
    )
    path = OUTPUT_DIR / "motif_delta_dominant_real_removal.svg"
    note = (
        "Negative values mean removal lowered the dominant real eigenvalue; "
        "positive values mean removal raised it."
    )
    write_svg(path, 1000, panel_h + 110, body, "Raw eigenvalue effect of motif-removal", note)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    removal_null = compute_removal_null_z()
    z_path = make_null_z_visual(removal_null)
    delta_path = make_delta_visual()
    print(f"Wrote {z_path}")
    print(f"Wrote {delta_path}")
    print(f"Wrote {OUTPUT_DIR / 'motif_removal_null_z_by_variant.csv'}")


if __name__ == "__main__":
    main()
