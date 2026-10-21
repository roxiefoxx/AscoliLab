"""Rees et al. (2016), Figure 7-style analysis of directed induced triads.

Every unordered triple is counted once, using A[pre, post] != 0; signs,
weights and self-loops do not enter this structural analysis. This reproduces
the analysis on the supplied dataset, not the original paper's numerical data.

Panel D is an explicit operational reconstruction: mean motif participation
per cell type in one group divided by that in the other. The paper's prose
does not fully specify that normalization; raw participation ratios are also
exported. Default scope is triples entirely within CA1. Both are configurable.

Source: https://doi.org/10.1523/ENEURO.0205-16.2016 (Figure 7).
Dependencies: numpy, pandas, matplotlib, networkx.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import comb
from pathlib import Path
from typing import Mapping
from importlib.metadata import version
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator
import networkx as nx
from networkx.generators.triads import triad_graph
import numpy as np
import pandas as pd

from weighted_motif_null import CODE2CLASS, PAIR_ORDER, TRIAD_CLASSES, TriadIndex


# Rees lettering follows the arrows in Figure 7A, not DHL class order.
REES_TO_DHL = dict(zip(
    ("-C", "-B", "-A", *"ABCDEFGHIJKLM"),
    ("003", "012", "102", "021U", "021C", "021D", "111D", "030T",
     "111U", "030C", "120D", "201", "120C", "120U", "210", "300"),
))
CONNECTED = list("ABCDEFGHIJKLM")
ORDER = list(REES_TO_DHL)
PST_TYPES = (
    "CA1 Axo Axonic", "CA1 Horizontal Axo Axonic", "CA1 Basket",
    "CA1 Basket CCK+", "CA1 Horizontal Basket",
)
REGION_ALIASES = {"LEC": "EC", "MEC": "EC", "CA3c": "CA3"}
REGION_COLORS = {
    "DG": "#795329", "CA3": "#ad7b54", "CA1": "#cd7726",
    "EC": "#648849", "CA2": "#9675a8", "SUB": "#ba6377",
    "Intermodular": "#bfd1e7",
}
STYLE = {"font.family": "DejaVu Sans", "font.size": 11,
         "axes.spines.top": False, "axes.spines.right": False,
         "axes.titleweight": "bold", "figure.facecolor": "white"}


@dataclass(frozen=True)
class Figure7Config:
    output_dir: Path = Path("outputs/consolidated_01_qc_census/rees_figure7")
    region_aliases: Mapping[str, str] = field(default_factory=lambda: REGION_ALIASES.copy())
    pst_types: tuple[str, ...] = PST_TYPES
    pyramidal_types: tuple[str, ...] = ("CA1 Pyramidal",)
    node_metadata: Path | None = None
    ratio_scope: str = "within_ca1"  # or all_triads containing each CA1 type
    ratio_normalization: str = "per_type"  # or raw participation totals
    top_n: int = 20


def _node_metadata(data: dict, config: Figure7Config) -> pd.DataFrame:
    labels = data["labels"]
    metadata = pd.DataFrame({"cell_type": labels, "ei": data["ei"]})
    metadata["region"] = [config.region_aliases.get(s.split()[0], s.split()[0]) for s in labels]
    metadata["ca1_group"] = "not_ca1"
    metadata.loc[metadata.region == "CA1", "ca1_group"] = "unassigned"
    metadata.loc[(metadata.region == "CA1") & (metadata.ei == "I"), "ca1_group"] = "other_interneuron"
    metadata.loc[metadata.cell_type.isin(config.pyramidal_types), "ca1_group"] = "pyramidal"
    metadata.loc[metadata.cell_type.isin(config.pst_types), "ca1_group"] = "pst"
    if config.node_metadata is not None:
        custom = pd.read_csv(config.node_metadata).set_index("cell_type")
        if not custom.index.is_unique or set(custom.index) != set(labels):
            raise ValueError("Node metadata must contain exactly one row for every matrix cell_type.")
        custom = custom.reindex(labels)
        for column in ("region", "ca1_group"):
            if column in custom:
                if custom[column].isna().any():
                    raise ValueError(f"Missing values in node metadata column {column}.")
                metadata[column] = custom[column].to_numpy()
    allowed = {"not_ca1", "unassigned", "pyramidal", "pst", "other_interneuron"}
    if not set(metadata.ca1_group) <= allowed:
        raise ValueError(f"ca1_group must use {sorted(allowed)}.")
    if metadata.region.eq("Intermodular").any() or metadata.region.astype(str).str.strip().eq("").any():
        raise ValueError("Node regions must be nonempty; Intermodular is reserved for mixed-region triples.")
    ca1 = metadata.region == "CA1"
    if (ca1 & (metadata.ca1_group == "not_ca1")).any() or ((~ca1) & (metadata.ca1_group != "not_ca1")).any():
        raise ValueError("CA1 group assignments must agree with region == CA1.")
    inhibitory = metadata.ca1_group.isin(["pst", "other_interneuron"])
    if (inhibitory & (metadata.ei != "I")).any() or ((metadata.ca1_group == "pyramidal") & (metadata.ei != "E")).any():
        raise ValueError("CA1 group assignments conflict with E/I labels.")
    return metadata


def calculate_figure7_tables(data: dict, config: Figure7Config | None = None) -> dict[str, pd.DataFrame]:
    """Return auditable counts, regional shares, node participation and ratios."""
    config = config or Figure7Config()
    if config.ratio_scope not in {"within_ca1", "all_triads"}:
        raise ValueError("ratio_scope must be within_ca1 or all_triads.")
    if config.ratio_normalization not in {"per_type", "raw"} or config.top_n < 1:
        raise ValueError("Choose ratio_normalization per_type/raw and top_n >= 1.")
    matrix = np.asarray(data["matrix"])
    n = len(data["labels"])
    if matrix.shape != (n, n) or not np.isfinite(matrix).all() or len(set(data["labels"])) != n:
        raise ValueError("A finite square matrix and unique cell labels are required.")
    nodes = _node_metadata(data, config)
    binary = matrix != 0
    triples = TriadIndex(n)
    members = np.column_stack((triples.a, triples.b, triples.c))
    bits = np.zeros(len(members), dtype=np.int64)
    for bit, (pre, post) in enumerate(PAIR_ORDER):
        bits |= binary[members[:, pre], members[:, post]].astype(np.int64) << bit
    classes = np.array(TRIAD_CLASSES)[CODE2CLASS[bits]]
    reverse = {dhl: letter for letter, dhl in REES_TO_DHL.items()}
    letters = np.array([reverse[c] for c in classes])
    counts = pd.Series(letters).value_counts().reindex(ORDER, fill_value=0)
    census = pd.DataFrame({"superpattern": ORDER, "triad_class": list(REES_TO_DHL.values()),
                           "count": counts.to_numpy(), "connected": [s in CONNECTED for s in ORDER]})
    total = comb(n, 3) if n >= 3 else 0
    census["share"] = census["count"] / total if total else np.nan
    region = nodes.region.to_numpy()[members]
    intra = (region[:, 0] == region[:, 1]) & (region[:, 0] == region[:, 2])
    location = np.where(intra, region[:, 0], "Intermodular")
    regions = sorted(set(nodes.region), key=lambda r: (["DG", "CA3", "CA1", "EC"].index(r) if r in {"DG", "CA3", "CA1", "EC"} else 4, r))
    locations = regions + ["Intermodular"]
    region_counts = pd.crosstab(pd.Series(letters, name="superpattern"), pd.Series(location, name="region"))
    region_counts = region_counts.reindex(index=CONNECTED, columns=locations, fill_value=0)
    region_shares = region_counts.div(region_counts.sum(axis=1).replace(0, np.nan), axis=0)
    participation = np.zeros((n, len(ORDER)), dtype=np.int64)
    scoped = np.zeros_like(participation)
    within_ca1 = intra & (region[:, 0] == "CA1")
    for j, letter in enumerate(ORDER):
        selected = letters == letter
        np.add.at(participation[:, j], members[selected].ravel(), 1)
        np.add.at(scoped[:, j], members[selected & within_ca1].ravel(), 1)
    full_participation = nodes.join(pd.DataFrame(participation, columns=ORDER))
    ca1_participation = scoped if config.ratio_scope == "within_ca1" else participation
    groups = nodes.ca1_group
    group_counts = {g: int((groups == g).sum()) for g in ("pyramidal", "pst", "other_interneuron")}
    group_counts["interneuron"] = group_counts["pst"] + group_counts["other_interneuron"]
    usage = {}
    for group in group_counts:
        mask = groups.isin(["pst", "other_interneuron"]) if group == "interneuron" else groups == group
        usage[group] = ca1_participation[mask].sum(axis=0)[3:]
    ratios = pd.DataFrame({"superpattern": CONNECTED})
    for group, values in usage.items():
        ratios[f"{group}_types"] = group_counts[group]
        ratios[f"{group}_participation"] = values
        ratios[f"{group}_mean_per_type"] = values / group_counts[group] if group_counts[group] else np.nan
    for name, numerator, denominator in (("pyr_int", "pyramidal", "interneuron"), ("pst_other", "pst", "other_interneuron")):
        for suffix in ("participation", "mean_per_type"):
            a, b = ratios[f"{numerator}_{suffix}"], ratios[f"{denominator}_{suffix}"]
            ratios[f"{name}_{suffix}_ratio"] = a.div(b.replace(0, np.nan))
    ratios["scope"] = config.ratio_scope
    ratios["normalization"] = config.ratio_normalization
    if census["count"].sum() != total or participation.sum() != 3 * total:
        raise AssertionError("Triad counts or node participation failed reconciliation.")
    if not np.array_equal(region_counts.sum(axis=1).to_numpy(), counts.loc[CONNECTED].to_numpy()):
        raise AssertionError("Regional counts do not reconcile with the connected census.")
    return {"census": census, "regional_counts": region_counts.reset_index(),
            "regional_shares": region_shares.reset_index(), "node_metadata": nodes,
            "node_participation": full_participation, "ca1_ratios": ratios}


def _atlas(figure):
    axes = figure.subplots(4, 4)
    for ax, (letter, dhl) in zip(axes.flat, REES_TO_DHL.items()):
        graph = triad_graph(dhl)
        positions = {"a": (0, 1), "b": (0, 0), "c": (1, .5)}
        nx.draw_networkx_nodes(graph, positions, ax=ax, node_color="#a18db3", edgecolors="#252525", node_size=110)
        nx.draw_networkx_edges(graph, positions, ax=ax, arrows=True, arrowsize=13,
                               node_size=150, width=1.5, connectionstyle="arc3,rad=0.06")
        ax.set(xlim=(-.4, 1.4), ylim=(-.35, 1.4))
        ax.set_title(f"{letter}  ({dhl})", fontsize=10, pad=0)
        ax.axis("off")
    figure.suptitle("A  Directed triad superpatterns", fontsize=15, fontweight="bold")


def _counts(figure, tables):
    pie, bars = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.2]})
    census = tables["census"].set_index("superpattern")
    disconnected = census.loc[["-A", "-B", "-C"], "count"]
    values = [*disconnected, census.loc[CONNECTED, "count"].sum()]
    labels = [*(f"{s}: {v:,}" for s, v in disconnected.items()), f"Connected: {values[-1]:,}"]
    if sum(values):
        pie.pie(values, labels=labels, colors=["#b35b5c", "#8662a2", "#729cc6", "#444444"],
                textprops={"fontsize": 10}, wedgeprops={"edgecolor": "white"}, labeldistance=1.08)
    else:
        pie.text(.5, .5, "No triples", ha="center", transform=pie.transAxes)
    connected_counts = census.loc[CONNECTED, "count"]
    bars.barh(CONNECTED, connected_counts, color="#527ea6")
    bars.invert_yaxis()
    for y, value in enumerate(connected_counts):
        bars.text(value, y, f" {value:,}", va="center", fontsize=9)
    bars.set_xlim(0, max(connected_counts.max() * 1.35, 1))
    bars.set(xlabel="Induced triples (count)", ylabel="Connected superpattern")
    bars.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    figure.suptitle("B  Superpattern frequencies", fontsize=15, fontweight="bold")


def _regional(ax, tables):
    shares = tables["regional_shares"].set_index("superpattern")
    bottom = np.zeros(len(CONNECTED))
    extra_colors = plt.get_cmap("tab20")
    for i, column in enumerate(shares):
        values = shares[column].to_numpy()
        ax.bar(CONNECTED, values, bottom=bottom, label=column, width=.65,
               color=REGION_COLORS.get(column, extra_colors(i % 20)), edgecolor="white", linewidth=.4)
        bottom += np.nan_to_num(values)
    for i in np.flatnonzero(shares.isna().all(axis=1)):
        ax.text(i, .04, "n=0", ha="center", rotation=90, fontsize=9)
    ax.set(ylim=(0, 1), ylabel="Share of motif occurrences", title="C  Regional utilization of connected superpatterns")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.25)


def _ratios(ax, tables, config):
    ratios = tables["ca1_ratios"]
    suffix = "mean_per_type" if config.ratio_normalization == "per_type" else "participation"
    for name, label, color, marker, style in (
        ("pyr_int", "CA1 pyramidal / interneuron", "#527ea6", "^", "-"),
        ("pst_other", "CA1 PST / other interneuron", "#bb4e55", "o", "--"),
    ):
        ax.plot(CONNECTED, ratios[f"{name}_{suffix}_ratio"], label=label, color=color, marker=marker, linestyle=style)
    ax.axhline(1, color="#333333", linewidth=1)
    scope = "CA1-only triples" if config.ratio_scope == "within_ca1" else "all triples"
    normalization = "per-type means" if config.ratio_normalization == "per_type" else "raw totals"
    ax.set(ylabel="Participation ratio", xlabel="Superpattern", ylim=(0, None),
           title=f"D  CA1 group usage: {scope}, {normalization}\nOperational reconstruction")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.25)
    if ratios.filter(like="_ratio").isna().any().any():
        ax.text(0, -.25, "Gaps: missing group or zero denominator; see ca1_ratios.csv.", transform=ax.transAxes, fontsize=9)


def _participation_plots(tables, config):
    frame = tables["node_participation"]
    totals = frame[CONNECTED].sum(axis=1)
    histogram, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    ax.hist(totals, bins=min(20, max(1, len(totals))), color="#527ea6", edgecolor="white")
    ax.set(xlabel="Connected induced triples per cell type", ylabel="Cell types (count)",
           title="Distribution of node participation in connected motifs")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    top = frame.loc[totals.nlargest(config.top_n).index].set_index("cell_type")[CONNECTED]
    heatmap, ax = plt.subplots(figsize=(12, max(4, .32 * len(top) + 1.8)), layout="constrained")
    if len(top):
        image = ax.imshow(np.log1p(top), aspect="auto", cmap="Blues", vmin=0)
        heatmap.colorbar(image, ax=ax, label="log(1 + participation count)")
        ax.set_yticks(range(len(top)), top.index)
    ax.set_xticks(range(len(CONNECTED)), CONNECTED)
    ax.set(xlabel="Superpattern", title=f"Motif participation of the top {len(top)} cell types")
    return {"participation_histogram": histogram, "participation_heatmap": heatmap}


def plot_figure7(tables: dict, config: Figure7Config | None = None) -> dict:
    """Build standalone panels and a composite figure; no implicit display."""
    config = config or Figure7Config()
    with plt.rc_context(STYLE):
        atlas = plt.figure(figsize=(10, 9), layout="constrained")
        _atlas(atlas)
        counts = plt.figure(figsize=(13, 6), layout="constrained")
        _counts(counts, tables)
        regional, ax = plt.subplots(figsize=(13, 4.5), layout="constrained")
        _regional(ax, tables)
        ratios, ax = plt.subplots(figsize=(13, 4.5), layout="constrained")
        _ratios(ax, tables, config)
        combined = plt.figure(figsize=(17, 15), layout="constrained")
        grid = combined.add_gridspec(3, 2, height_ratios=[2.2, 1, 1])
        _atlas(combined.add_subfigure(grid[0, 0]))
        _counts(combined.add_subfigure(grid[0, 1]), tables)
        _regional(combined.add_subplot(grid[1, :]), tables)
        _ratios(combined.add_subplot(grid[2, :]), tables, config)
        combined.suptitle("Figure 7-style structural motif analysis | supplied dataset", fontsize=17, fontweight="bold")
        return {"panel_a_atlas": atlas, "panel_b_counts": counts,
                "panel_c_regions": regional, "panel_d_ca1_ratios": ratios,
                "figure7_composite": combined, **_participation_plots(tables, config)}


def run_figure7_analysis(data: dict, config: Figure7Config | None = None) -> dict:
    """Calculate and export tables, PNG/PDF figures and input provenance.

    Pass the result of motif_workflow.run_qc_and_census (or load_connectome).
    Figures are closed after export to avoid automatic duplicate notebook output.
    Display selected PNG files explicitly in the notebook.
    """
    config = config or Figure7Config()
    tables = calculate_figure7_tables(data, config)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    figures = plot_figure7(tables, config)
    figure_paths = {}
    try:
        for name, figure in figures.items():
            figure_paths[name] = out / f"{name}.png"
            figure.savefig(figure_paths[name], dpi=160)
            figure.savefig(out / f"{name}.pdf")
    finally:
        for figure in figures.values():
            plt.close(figure)
    paths = data.get("paths")
    sources = {}
    for name in ("matrix", "netlist"):
        path = Path(getattr(paths, name)) if paths is not None else None
        if path is not None:
            sources[name] = {"path": str(path.resolve()), "sha256": sha256(path.read_bytes()).hexdigest()}
    if config.node_metadata is not None:
        path = Path(config.node_metadata)
        sources["node_metadata"] = {"path": str(path.resolve()), "sha256": sha256(path.read_bytes()).hexdigest()}
    notes = [
        "Rees Figure 7-style reanalysis; published numerical results are not reference inputs.",
        "Binary A[pre,post] != 0, unordered distinct triples, induced edges, no self-loops.",
        "Panel C uses supplied/inferred anatomical regions, not detected graph communities.",
        "Panel D is an operational reconstruction; scope and normalization are explicit.",
        "Ratios have no null model or significance interpretation. Zero denominators are missing, not zero.",
    ]
    unassigned = tables["node_metadata"].query("ca1_group == 'unassigned'")
    if len(unassigned):
        notes.append(f"{len(unassigned)} CA1 types are unassigned and excluded from group ratios; review node_metadata.csv.")
    manifest = {"source_doi": "10.1523/ENEURO.0205-16.2016", "inputs": sources,
                "nodes": len(data["labels"]), "triads": int(tables["census"]["count"].sum()),
                "region_aliases": dict(config.region_aliases), "ratio_scope": config.ratio_scope,
                "ratio_normalization": config.ratio_normalization,
                "dependencies": {name: version(name) for name in ("numpy", "pandas", "matplotlib", "networkx")},
                "notes": notes}
    (out / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"tables": tables, "figure_paths": figure_paths, "output_dir": out, "notes": notes}
