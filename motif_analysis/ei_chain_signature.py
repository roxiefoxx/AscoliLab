"""Reusable E/I chain-signature analysis.

This module keeps the bulky table-building and figure code out of the
notebook.  The public entry point is ``run_chain_signature_analysis``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


SIGNATURE_ORDER = ["EEE", "EEI", "EIE", "EII", "IEE", "IEI", "IIE", "III"]
ROLES = ("source", "middle", "target")


@dataclass(frozen=True)
class ChainSignatureConfig:
    data_path: Path | str = Path("mij_netlist.csv")
    output_dir: Path | str = Path("outputs") / "ei_chain_signature_eda"
    signature_order: tuple[str, ...] = tuple(SIGNATURE_ORDER)
    selected_signature: str = "EII"
    top_n: int = 20

    @property
    def table_dir(self) -> Path:
        return Path(self.output_dir) / "tables"

    @property
    def figure_dir(self) -> Path:
        return Path(self.output_dir) / "figures"


@dataclass
class ChainSignatureResult:
    edges: pd.DataFrame
    chains: pd.DataFrame
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    table_paths: dict[str, Path] = field(default_factory=dict)
    figure_paths: dict[str, Path] = field(default_factory=dict)


def clean_ei(value: object) -> str:
    value = str(value).strip().upper()
    if value.startswith("E"):
        return "E"
    if value.startswith("I"):
        return "I"
    raise ValueError(f"Unexpected E/I label: {value!r}")


def parse_region(neuron_name: object) -> str:
    parts = str(neuron_name).strip().split()
    return parts[0] if parts else "Unknown"


def parse_cell_type(neuron_name: object) -> str:
    parts = str(neuron_name).strip().split()
    if len(parts) <= 1:
        return str(neuron_name).strip() or "Unknown"
    return " ".join(parts[1:])


def pct(numerator: float, denominator: float) -> float:
    if denominator == 0 or pd.isna(denominator):
        return 0.0
    return 100.0 * numerator / denominator


def safe_name(value: object) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return value.strip("_") or "value"


def ensure_output_dirs(config: ChainSignatureConfig) -> None:
    for directory in [Path(config.output_dir), config.table_dir, config.figure_dir]:
        directory.mkdir(parents=True, exist_ok=True)


def save_table(
    df: pd.DataFrame,
    name: str,
    config: ChainSignatureConfig,
    index: bool = False,
) -> Path:
    path = config.table_dir / f"{safe_name(name)}.csv"
    df.to_csv(path, index=index)
    return path


def display_file(path: Path) -> None:
    try:
        from IPython.display import Image as IPythonImage, display

        display(IPythonImage(filename=str(path)))
    except Exception:
        print(f"saved figure: {path}")


def _font(size: int = 16, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    return ImageFont.load_default()


FONT_TITLE = _font(28, bold=True)
FONT_SUBTITLE = _font(18)
FONT_LABEL = _font(16)
FONT_SMALL = _font(13)
FONT_TINY = _font(11)


def _text_width(draw: ImageDraw.ImageDraw, text: object, fnt: ImageFont.ImageFont) -> int:
    left, _top, right, _bottom = draw.textbbox((0, 0), str(text), font=fnt)
    return right - left


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: object,
    fnt: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    width: int,
    line_spacing: int = 4,
) -> int:
    lines = []
    for raw_line in str(text).splitlines():
        words = raw_line.split()
        current = ""
        for word in words:
            candidate = (current + " " + word).strip()
            if _text_width(draw, candidate, fnt) <= width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)

    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_spacing
    return y


def _color_mix(
    low: tuple[int, int, int],
    high: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(low[i] + (high[i] - low[i]) * t) for i in range(3))


def _maybe_display(path: Path, show: bool) -> Path:
    if show:
        display_file(path)
    return path


def draw_horizontal_bar(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    path: Path,
    subtitle: str | None = None,
    value_format: str = "{:,.1f}",
    color: tuple[int, int, int] = (56, 112, 166),
    show: bool = True,
) -> Path:
    plot_df = df[[label_col, value_col]].copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce").fillna(0)
    plot_df = plot_df.sort_values(value_col, ascending=True)

    width = 1300
    row_h = 30
    height = max(420, 170 + row_h * len(plot_df))
    margin_left = 330
    margin_right = 160
    margin_top = 105 if subtitle else 80
    margin_bottom = 70

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((28, 22), title, font=FONT_TITLE, fill=(32, 37, 42))
    if subtitle:
        _draw_wrapped_text(
            draw,
            (30, 60),
            subtitle,
            FONT_SUBTITLE,
            (83, 91, 99),
            width - 60,
        )

    x0 = margin_left
    x1 = width - margin_right
    y0 = margin_top
    y1 = height - margin_bottom
    min_val = min(float(plot_df[value_col].min()), 0.0)
    max_val = max(float(plot_df[value_col].max()), 0.0)
    if min_val == max_val:
        max_val = min_val + 1.0

    scale = (x1 - x0) / (max_val - min_val)
    zero_x = int(x0 + (0 - min_val) * scale)
    draw.line((x0, y1, x1, y1), fill=(190, 196, 201), width=2)
    if min_val < 0 < max_val:
        draw.line((zero_x, y0 - 8, zero_x, y1), fill=(120, 127, 135), width=2)

    for i, (_, row) in enumerate(plot_df.iterrows()):
        y = y0 + i * row_h
        val = float(row[value_col])
        val_x = int(x0 + (val - min_val) * scale)
        bx0 = min(zero_x, val_x)
        bx1 = max(zero_x, val_x)
        draw.rectangle((bx0, y + 5, bx1, y + row_h - 5), fill=color)
        draw.text((20, y + 4), str(row[label_col])[:48], font=FONT_SMALL, fill=(34, 39, 46))

        value_label = value_format.format(val)
        label_x = bx1 + 8 if val >= 0 else max(4, bx0 - _text_width(draw, value_label, FONT_SMALL) - 8)
        draw.text((label_x, y + 4), value_label, font=FONT_SMALL, fill=(34, 39, 46))

    draw.text((x0, height - 45), value_col.replace("_", " "), font=FONT_LABEL, fill=(83, 91, 99))
    img.save(path)
    return _maybe_display(path, show)


def draw_grouped_signature_bars(
    df: pd.DataFrame,
    title: str,
    path: Path,
    signature_order: tuple[str, ...] = tuple(SIGNATURE_ORDER),
    show: bool = True,
) -> Path:
    plot_df = df.set_index("signature").reindex(signature_order).reset_index()
    width, height = 1250, 650
    margin_left, margin_right, margin_top, margin_bottom = 90, 50, 90, 115

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((30, 22), title, font=FONT_TITLE, fill=(32, 37, 42))

    x0, x1 = margin_left, width - margin_right
    y0, y1 = margin_top, height - margin_bottom
    max_val = max(float(plot_df[["count_pct", "abs_weight_pct"]].max().max()), 1.0)

    for frac in np.linspace(0, 1, 6):
        y = y1 - int((y1 - y0) * frac)
        draw.line((x0, y, x1, y), fill=(229, 232, 235), width=1)
        draw.text((32, y - 8), f"{max_val * frac:.0f}%", font=FONT_TINY, fill=(83, 91, 99))

    group_w = (x1 - x0) / len(plot_df)
    bar_w = group_w * 0.30
    colors = {"count_pct": (56, 112, 166), "abs_weight_pct": (214, 128, 58)}
    for i, row in plot_df.iterrows():
        cx = x0 + i * group_w + group_w / 2
        for j, metric in enumerate(["count_pct", "abs_weight_pct"]):
            val = float(row[metric])
            h = int((y1 - y0) * val / max_val)
            bx0 = cx - bar_w + j * bar_w
            bx1 = bx0 + bar_w - 4
            draw.rectangle((bx0, y1 - h, bx1, y1), fill=colors[metric])
            draw.text((bx0, y1 - h - 18), f"{val:.1f}", font=FONT_TINY, fill=(34, 39, 46))
        draw.text((cx - 18, y1 + 16), row["signature"], font=FONT_LABEL, fill=(34, 39, 46))

    draw.text(
        (x0, height - 45),
        "Blue = chain count share; orange = absolute chain-weight share",
        font=FONT_LABEL,
        fill=(83, 91, 99),
    )
    img.save(path)
    return _maybe_display(path, show)


def draw_heatmap(
    pivot: pd.DataFrame,
    title: str,
    path: Path,
    value_format: str = "{:.1f}",
    max_rows: int = 30,
    max_cols: int = 30,
    show: bool = True,
) -> Path:
    data = pivot.copy()
    if data.shape[0] > max_rows:
        data = data.iloc[:max_rows]
    if data.shape[1] > max_cols:
        data = data.iloc[:, :max_cols]

    rows = [str(x) for x in data.index]
    cols = [str(x) for x in data.columns]
    cell_w = max(54, min(110, int(940 / max(len(cols), 1))))
    cell_h = 30
    margin_left = min(360, max(120, 10 + max((len(r) for r in rows), default=6) * 7))
    margin_top = 145
    width = margin_left + cell_w * len(cols) + 90
    height = margin_top + cell_h * len(rows) + 80

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((28, 22), title, font=FONT_TITLE, fill=(32, 37, 42))
    if data.empty:
        draw.text((30, 90), "No data available.", font=FONT_LABEL, fill=(83, 91, 99))
        img.save(path)
        return _maybe_display(path, show)

    values = data.to_numpy(dtype=float)
    max_val = float(np.nanmax(values)) if np.isfinite(values).any() else 0.0
    max_val = max(max_val, 1.0)

    for j, col in enumerate(cols):
        x = margin_left + j * cell_w
        draw.text((x + 3, 100), col[:12], font=FONT_TINY, fill=(34, 39, 46))

    for i, row in enumerate(rows):
        y = margin_top + i * cell_h
        draw.text((20, y + 7), row[:44], font=FONT_SMALL, fill=(34, 39, 46))
        for j, _col in enumerate(cols):
            val = float(data.iloc[i, j]) if pd.notna(data.iloc[i, j]) else 0.0
            t = math.sqrt(max(val, 0) / max_val)
            fill = _color_mix((245, 248, 250), (38, 99, 150), t)
            x = margin_left + j * cell_w
            draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), fill=fill, outline=(230, 233, 236))
            if cell_w >= 70:
                text = value_format.format(val)
                text_fill = (20, 24, 28) if t < 0.55 else "white"
                draw.text((x + 5, y + 8), text, font=FONT_TINY, fill=text_fill)

    img.save(path)
    return _maybe_display(path, show)


def load_edges(data_path: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    netlist = pd.read_csv(data_path)
    required_columns = {"pre_neuron", "post_neuron", "pre_ei", "post_ei", "m_ij"}
    missing_columns = sorted(required_columns - set(netlist.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    edges = netlist.rename(
        columns={
            "pre_neuron": "source",
            "post_neuron": "target",
            "pre_ei": "source_ei",
            "post_ei": "target_ei",
        }
    ).copy()
    edges["source_ei"] = edges["source_ei"].map(clean_ei)
    edges["target_ei"] = edges["target_ei"].map(clean_ei)
    edges["m_ij"] = pd.to_numeric(edges["m_ij"], errors="coerce")
    edges = edges.dropna(subset=["source", "target", "source_ei", "target_ei", "m_ij"])
    edges = edges[(edges["source"] != edges["target"]) & (edges["m_ij"] != 0)].copy()
    edges["source_region"] = edges["source"].map(parse_region)
    edges["target_region"] = edges["target"].map(parse_region)
    edges["source_cell_type"] = edges["source"].map(parse_cell_type)
    edges["target_cell_type"] = edges["target"].map(parse_cell_type)
    edges["abs_edge_weight"] = edges["m_ij"].abs()

    node_ei_from_source = edges[["source", "source_ei"]].rename(
        columns={"source": "neuron", "source_ei": "ei"}
    )
    node_ei_from_target = edges[["target", "target_ei"]].rename(
        columns={"target": "neuron", "target_ei": "ei"}
    )
    node_ei = pd.concat([node_ei_from_source, node_ei_from_target], ignore_index=True).drop_duplicates()
    conflicting_nodes = node_ei.groupby("neuron")["ei"].nunique()
    conflicting_nodes = conflicting_nodes[conflicting_nodes > 1]
    if not conflicting_nodes.empty:
        raise ValueError(f"Nodes with conflicting E/I labels: {conflicting_nodes.index.tolist()[:10]}")

    return netlist, edges


def data_profile(netlist: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": [
                "raw edge rows",
                "usable nonzero non-self edges",
                "unique neurons",
                "unique source regions",
                "unique target regions",
            ],
            "value": [
                len(netlist),
                len(edges),
                len(set(edges["source"]) | set(edges["target"])),
                edges["source_region"].nunique(),
                edges["target_region"].nunique(),
            ],
        }
    )


def build_chains(edges: pd.DataFrame) -> pd.DataFrame:
    left_edges = edges.add_suffix("_1")
    right_edges = edges.add_suffix("_2")
    chains = left_edges.merge(
        right_edges,
        left_on="target_1",
        right_on="source_2",
        how="inner",
    )
    chains = chains[
        (chains["source_1"] != chains["target_1"])
        & (chains["target_1"] != chains["target_2"])
        & (chains["source_1"] != chains["target_2"])
    ].copy()

    chains = chains.rename(
        columns={
            "source_1": "source",
            "target_1": "middle",
            "target_2": "target",
            "source_ei_1": "source_ei",
            "target_ei_1": "middle_ei",
            "target_ei_2": "target_ei",
            "m_ij_1": "edge1_weight",
            "m_ij_2": "edge2_weight",
            "source_region_1": "source_region",
            "target_region_1": "middle_region",
            "target_region_2": "target_region",
            "source_cell_type_1": "source_cell_type",
            "target_cell_type_1": "middle_cell_type",
            "target_cell_type_2": "target_cell_type",
        }
    )
    chains["signature"] = chains["source_ei"] + chains["middle_ei"] + chains["target_ei"]
    chains["signed_chain_weight"] = chains["edge1_weight"] * chains["edge2_weight"]
    chains["abs_chain_weight"] = chains["signed_chain_weight"].abs()
    chains["chain_intensity"] = np.sqrt(chains["abs_chain_weight"])
    chains["region_pathway"] = (
        chains["source_region"] + " -> " + chains["middle_region"] + " -> " + chains["target_region"]
    )
    chains["cell_type_pathway"] = (
        chains["source_cell_type"] + " -> " + chains["middle_cell_type"] + " -> " + chains["target_cell_type"]
    )
    chains["chain_label"] = chains["source"] + " -> " + chains["middle"] + " -> " + chains["target"]
    return chains


def chain_profile(chains: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": [
                "simple directed 3-node chains",
                "signatures observed",
                "total absolute chain weight",
                "total signed chain weight",
            ],
            "value": [
                len(chains),
                chains["signature"].nunique(),
                chains["abs_chain_weight"].sum(),
                chains["signed_chain_weight"].sum(),
            ],
        }
    )


def signature_summary(chains: pd.DataFrame, signature_order: tuple[str, ...]) -> pd.DataFrame:
    summary = (
        chains.groupby("signature")
        .agg(
            chain_count=("signature", "size"),
            abs_chain_weight=("abs_chain_weight", "sum"),
            signed_chain_weight=("signed_chain_weight", "sum"),
            mean_intensity=("chain_intensity", "mean"),
            median_intensity=("chain_intensity", "median"),
            max_abs_chain_weight=("abs_chain_weight", "max"),
        )
        .reindex(signature_order, fill_value=0)
        .reset_index()
    )
    summary["count_pct"] = summary["chain_count"].map(lambda v: pct(v, summary["chain_count"].sum()))
    summary["abs_weight_pct"] = summary["abs_chain_weight"].map(lambda v: pct(v, summary["abs_chain_weight"].sum()))
    summary["weight_vs_count_pct_point_gap"] = summary["abs_weight_pct"] - summary["count_pct"]
    return summary[
        [
            "signature",
            "chain_count",
            "count_pct",
            "abs_chain_weight",
            "abs_weight_pct",
            "weight_vs_count_pct_point_gap",
            "signed_chain_weight",
            "mean_intensity",
            "median_intensity",
            "max_abs_chain_weight",
        ]
    ]


def role_long_table(chains: pd.DataFrame, field: str) -> pd.DataFrame:
    frames = []
    for role in ROLES:
        frames.append(
            chains[["signature", f"{role}_{field}", "abs_chain_weight"]]
            .rename(columns={f"{role}_{field}": field})
            .assign(role=role)
        )
    return pd.concat(frames, ignore_index=True)


def makeup_by_signature(role_table: pd.DataFrame, field: str) -> pd.DataFrame:
    makeup = (
        role_table.groupby(["signature", field])
        .agg(chain_node_mentions=(field, "size"), abs_chain_weight=("abs_chain_weight", "sum"))
        .reset_index()
    )
    makeup["mention_pct_within_signature"] = makeup.groupby("signature")["chain_node_mentions"].transform(
        lambda s: 100 * s / s.sum()
    )
    makeup["weight_pct_within_signature"] = makeup.groupby("signature")["abs_chain_weight"].transform(
        lambda s: 100 * s / s.sum()
    )
    return makeup


def role_makeup_by_signature(role_table: pd.DataFrame, field: str) -> pd.DataFrame:
    role_makeup = (
        role_table.groupby(["signature", "role", field])
        .agg(chain_count=(field, "size"), abs_chain_weight=("abs_chain_weight", "sum"))
        .reset_index()
    )
    role_makeup["count_pct_within_signature_role"] = role_makeup.groupby(["signature", "role"])[
        "chain_count"
    ].transform(lambda s: 100 * s / s.sum())
    return role_makeup


def pathways_by_signature(chains: pd.DataFrame, pathway_col: str) -> pd.DataFrame:
    return (
        chains.groupby(["signature", pathway_col])
        .agg(
            chain_count=(pathway_col, "size"),
            abs_chain_weight=("abs_chain_weight", "sum"),
            signed_chain_weight=("signed_chain_weight", "sum"),
            mean_intensity=("chain_intensity", "mean"),
        )
        .reset_index()
        .sort_values(["abs_chain_weight", "chain_count"], ascending=False)
    )


def region_two_edge_flow(chains: pd.DataFrame) -> pd.DataFrame:
    flow_edge1 = chains[["source_region", "middle_region", "abs_chain_weight"]].rename(
        columns={"source_region": "from_region", "middle_region": "to_region"}
    ).assign(edge_position="source_to_middle")
    flow_edge2 = chains[["middle_region", "target_region", "abs_chain_weight"]].rename(
        columns={"middle_region": "from_region", "target_region": "to_region"}
    ).assign(edge_position="middle_to_target")
    return (
        pd.concat([flow_edge1, flow_edge2], ignore_index=True)
        .groupby(["from_region", "to_region"])
        .agg(chain_edge_mentions=("to_region", "size"), abs_chain_weight=("abs_chain_weight", "sum"))
        .reset_index()
        .sort_values("abs_chain_weight", ascending=False)
    )


def signature_drilldown(chains: pd.DataFrame, signature: str) -> pd.DataFrame:
    columns = [
        "signature",
        "chain_label",
        "region_pathway",
        "cell_type_pathway",
        "edge1_weight",
        "edge2_weight",
        "signed_chain_weight",
        "abs_chain_weight",
        "chain_intensity",
    ]
    return chains[chains["signature"] == signature].sort_values("abs_chain_weight", ascending=False)[columns]


def _table(result: ChainSignatureResult, name: str, df: pd.DataFrame, config: ChainSignatureConfig) -> None:
    result.tables[name] = df
    result.table_paths[name] = save_table(df, name, config)


def _draw_figures(result: ChainSignatureResult, config: ChainSignatureConfig, show_figures: bool) -> None:
    summary = result.tables["signature_summary"]
    figures = result.figure_paths

    figures["signature_count_vs_weight_share"] = draw_grouped_signature_bars(
        summary,
        "E/I signature share: count vs absolute chain weight",
        config.figure_dir / "signature_count_vs_weight_share.png",
        signature_order=config.signature_order,
        show=show_figures,
    )
    figures["signature_abs_chain_weight"] = draw_horizontal_bar(
        summary,
        "signature",
        "abs_chain_weight",
        "Absolute chain weight by E/I signature",
        config.figure_dir / "signature_abs_chain_weight.png",
        value_format="{:,.0f}",
        color=(214, 128, 58),
        show=show_figures,
    )
    figures["signature_signed_chain_weight"] = draw_horizontal_bar(
        summary,
        "signature",
        "signed_chain_weight",
        "Signed chain weight by E/I signature",
        config.figure_dir / "signature_signed_chain_weight.png",
        value_format="{:,.0f}",
        color=(91, 133, 93),
        show=show_figures,
    )

    region_makeup = result.tables["region_makeup_by_signature"]
    region_pivot = (
        region_makeup.pivot_table(
            index="region",
            columns="signature",
            values="weight_pct_within_signature",
            fill_value=0,
        )
        .reindex(columns=config.signature_order)
        .sort_values(list(config.signature_order), ascending=False)
    )
    figures["region_makeup_by_signature_heatmap"] = draw_heatmap(
        region_pivot,
        "Region makeup by signature (% of absolute chain weight)",
        config.figure_dir / "region_makeup_by_signature_heatmap.png",
        value_format="{:.1f}",
        show=show_figures,
    )

    region_role = result.tables["region_role_makeup_by_signature"]
    for role in ROLES:
        role_pivot = (
            region_role[region_role["role"] == role]
            .pivot_table(
                index="region",
                columns="signature",
                values="count_pct_within_signature_role",
                fill_value=0,
            )
            .reindex(columns=config.signature_order)
            .sort_values(list(config.signature_order), ascending=False)
        )
        figures[f"{role}_region_by_signature_heatmap"] = draw_heatmap(
            role_pivot,
            f"{role.title()} region by signature (% of chains)",
            config.figure_dir / f"{role}_region_by_signature_heatmap.png",
            value_format="{:.1f}",
            show=show_figures,
        )

    cell_makeup = result.tables["cell_type_makeup_by_signature"]
    top_cell_types = (
        cell_makeup.groupby("cell_type")["abs_chain_weight"].sum().sort_values(ascending=False).head(config.top_n).index
    )
    cell_pivot = (
        cell_makeup[cell_makeup["cell_type"].isin(top_cell_types)]
        .pivot_table(
            index="cell_type",
            columns="signature",
            values="weight_pct_within_signature",
            fill_value=0,
        )
        .reindex(columns=config.signature_order)
        .loc[top_cell_types]
    )
    figures["cell_type_makeup_by_signature_heatmap"] = draw_heatmap(
        cell_pivot,
        f"Top {config.top_n} cell-type makeup by signature (% of absolute chain weight)",
        config.figure_dir / "cell_type_makeup_by_signature_heatmap.png",
        value_format="{:.1f}",
        show=show_figures,
    )

    cell_role = result.tables["cell_type_role_makeup_by_signature"]
    for role in ROLES:
        top_for_role = (
            cell_role[cell_role["role"] == role]
            .groupby("cell_type")["abs_chain_weight"]
            .sum()
            .sort_values(ascending=False)
            .head(config.top_n)
            .index
        )
        role_pivot = (
            cell_role[(cell_role["role"] == role) & (cell_role["cell_type"].isin(top_for_role))]
            .pivot_table(
                index="cell_type",
                columns="signature",
                values="count_pct_within_signature_role",
                fill_value=0,
            )
            .reindex(columns=config.signature_order)
            .loc[top_for_role]
        )
        figures[f"{role}_cell_type_by_signature_heatmap"] = draw_heatmap(
            role_pivot,
            f"Top {config.top_n} {role} cell types by signature (% of chains)",
            config.figure_dir / f"{role}_cell_type_by_signature_heatmap.png",
            value_format="{:.1f}",
            show=show_figures,
        )

    region_pathways = result.tables["region_pathways_by_signature"]
    figures["top_region_pathways"] = draw_horizontal_bar(
        region_pathways.head(config.top_n),
        "region_pathway",
        "abs_chain_weight",
        f"Top {config.top_n} region pathways by absolute chain weight",
        config.figure_dir / "top_region_pathways.png",
        value_format="{:,.0f}",
        color=(56, 112, 166),
        show=show_figures,
    )

    cell_pathways = result.tables["cell_type_pathways_by_signature"]
    figures["top_cell_type_pathways"] = draw_horizontal_bar(
        cell_pathways.head(config.top_n),
        "cell_type_pathway",
        "abs_chain_weight",
        f"Top {config.top_n} cell-type pathways by absolute chain weight",
        config.figure_dir / "top_cell_type_pathways.png",
        value_format="{:,.0f}",
        color=(214, 128, 58),
        show=show_figures,
    )

    flow = result.tables["region_to_region_two_edge_flow"]
    flow_pivot = flow.pivot_table(index="from_region", columns="to_region", values="abs_chain_weight", fill_value=0)
    ordered_regions = sorted(set(flow_pivot.index) | set(flow_pivot.columns))
    flow_pivot = flow_pivot.reindex(index=ordered_regions, columns=ordered_regions, fill_value=0)
    figures["region_to_region_two_edge_flow_heatmap"] = draw_heatmap(
        flow_pivot,
        "Region-to-region flow across both chain edges (absolute weight)",
        config.figure_dir / "region_to_region_two_edge_flow_heatmap.png",
        value_format="{:,.0f}",
        show=show_figures,
    )

    drilldown = result.tables[f"high_weight_{config.selected_signature}_chains"]
    figures[f"top_{config.selected_signature}_chains"] = draw_horizontal_bar(
        drilldown.head(config.top_n),
        "chain_label",
        "abs_chain_weight",
        f"Top {config.top_n} {config.selected_signature} chains by absolute chain weight",
        config.figure_dir / f"top_{config.selected_signature}_chains.png",
        value_format="{:,.0f}",
        color=(91, 133, 93),
        show=show_figures,
    )


def run_chain_signature_analysis(
    config: ChainSignatureConfig | None = None,
    *,
    make_figures: bool = True,
    show_figures: bool = True,
) -> ChainSignatureResult:
    config = config or ChainSignatureConfig()
    ensure_output_dirs(config)

    netlist, edges = load_edges(config.data_path)
    chains = build_chains(edges)
    result = ChainSignatureResult(edges=edges, chains=chains)

    _table(result, "data_profile", data_profile(netlist, edges), config)
    _table(result, "chain_profile", chain_profile(chains), config)
    _table(result, "signature_summary", signature_summary(chains, config.signature_order), config)

    region_roles = role_long_table(chains, "region")
    _table(result, "region_makeup_by_signature", makeup_by_signature(region_roles, "region"), config)
    _table(result, "region_role_makeup_by_signature", role_makeup_by_signature(region_roles, "region"), config)

    cell_roles = role_long_table(chains, "cell_type")
    _table(result, "cell_type_makeup_by_signature", makeup_by_signature(cell_roles, "cell_type"), config)
    _table(result, "cell_type_role_makeup_by_signature", role_makeup_by_signature(cell_roles, "cell_type"), config)

    _table(result, "region_pathways_by_signature", pathways_by_signature(chains, "region_pathway"), config)
    _table(result, "cell_type_pathways_by_signature", pathways_by_signature(chains, "cell_type_pathway"), config)
    _table(result, "region_to_region_two_edge_flow", region_two_edge_flow(chains), config)
    _table(result, f"high_weight_{config.selected_signature}_chains", signature_drilldown(chains, config.selected_signature), config)

    if make_figures:
        _draw_figures(result, config, show_figures)

    return result
