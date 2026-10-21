"""Functional queries over the interneuron knowledge base.

    from resolver import KB
    from query import describe, find
    kb = KB("hippocampal_interneuron_kb.sqlite")

    describe(kb, "CA1 O LM", condition="ACh", stimulus_at="distal")
    describe(kb, "CA1 Basket CCK+", condition="eCB", inhibits_at="perisomatic")
    find(kb, inhibits_at="distal", region="CA1", condition="opioid")     # -> DataFrame
    find(kb, stimulus_at="TA")                                            # who is recruited by EC input?

Locations
    inhibits_at : where the cell's OUTPUT lands on its targets (axon layer / postsynaptic compartment)
    stimulus_at : where INPUT arrives on the interneuron (its dendritic layers), or a named pathway
    Accepted: 'distal', 'proximal', 'perisomatic', 'AIS', 'soma', 'apical', 'oblique', 'basal', 'dendritic',
              'volume', any layer code ('SLM', 'SR', 'SO', 'SP', 'SL', 'ALV', 'OML', 'MML', 'IML', 'SG', 'H',
              'LI'..'LVI'), or a pathway code ('SC', 'TA', 'NRe', 'PP', 'PP_CA3', 'MF', 'CA1_rec', 'CA3_rec', 'HC_assoc').

Conditions
    A condition code ('ACh', '5-HT', 'DA', 'eCB', 'opioid', 'CCK', 'OXT', 'GABA_B', 'theta', 'gamma', 'SWR',
    'trains', 'CFC', 'navigation'), a receptor ('M1', 'MOR', 'CB1', ...), or a plain word ('acetylcholine',
    'serotonin', 'repetitive', 'place cells', ...). Several may be given as a list.

What the answers are built from
    * function / circuit-role statements from the literature (with page locators)
    * anatomy claims (dendrite and axon layers, target compartments)
    * overlap of those layers with afferent-pathway termination zones
    * the netlist itself (who actually drives the cell, whom it hits)
    * receptor-effect claims for the condition (sign of the effect, not its size)
The KB does NOT yet model how the interneuron integrates proximal vs distal input (attenuation, dendritic
nonlinearities); 'stimulus_at' answers whether input there reaches the cell's dendrites, not how strongly
the soma feels it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

LOCATIONS = {
    "distal":      ({"distal_dendrite"}, {"hippocampal_CA": {"SLM"}, "dentate": {"OML", "MML"}}),
    "proximal":    ({"proximal_dendrite", "soma"}, {"hippocampal_CA": {"SP"}, "dentate": {"SG", "IML"}}),
    "perisomatic": ({"soma", "proximal_dendrite", "AIS"}, {"hippocampal_CA": {"SP"}, "dentate": {"SG"}}),
    "AIS":         ({"AIS"}, {}),
    "soma":        ({"soma"}, {}),
    "apical":      ({"apical_shaft", "oblique_dendrite"}, {"hippocampal_CA": {"SR"}}),
    "oblique":     ({"oblique_dendrite"}, {"hippocampal_CA": {"SR"}}),
    "basal":       ({"basal_dendrite"}, {"hippocampal_CA": {"SO"}}),
    "dendritic":   ({"basal_dendrite", "oblique_dendrite", "apical_shaft", "distal_dendrite", "dendrite_unspecified",
                     "proximal_dendrite"},
                    {"hippocampal_CA": {"SO", "SR", "SLM"}, "dentate": {"OML", "MML", "IML"}}),
    "volume":      ({"nonsynaptic_volume"}, {}),
}
ALIASES = {
    "acetylcholine": "ACh", "cholinergic": "ACh", "muscarinic": "ACh", "nicotinic": "ACh",
    "serotonin": "5-HT", "serotonergic": "5-HT", "dopamine": "DA", "dopaminergic": "DA",
    "endocannabinoid": "eCB", "cannabinoid": "eCB", "thc": "eCB", "opioids": "opioid", "enkephalin": "opioid",
    "oxytocin": "OXT", "cholecystokinin": "CCK", "gabab": "GABA_B",
    "repetitive": "trains", "high-frequency": "trains", "high frequency": "trains", "burst": "trains",
    "train": "trains", "ripples": "SWR", "sharp wave": "SWR", "fear": "CFC", "fear conditioning": "CFC",
    "place cells": "navigation", "place": "navigation", "spatial": "navigation",
}
COMPARTMENT_LABEL = {
    "AIS": "axon initial segment", "soma": "soma", "proximal_dendrite": "proximal dendrites",
    "basal_dendrite": "basal dendrites", "oblique_dendrite": "oblique dendrites", "apical_shaft": "apical shaft",
    "distal_dendrite": "distal dendrites / tuft", "dendrite_unspecified": "dendrites",
    "nonsynaptic_volume": "extracellular volume (no synaptic target)",
}
EFFECT_PHRASE = {
    ("excitability", "increase"): "raises excitability -> more inhibition of {where}",
    ("excitability", "decrease"): "lowers excitability -> less inhibition of {where} (disinhibition)",
    ("excitability", "biphasic"): "biphasic excitability change",
    ("excitability", "none"):     "no effect on excitability",
    ("release", "decrease"):      "lowers GABA release -> disinhibits {where}",
    ("release", "increase"):      "raises GABA release -> more inhibition of {where}",
    ("release", "none"):          "no effect on release",
    ("adaptation", "decrease"):   "reduces spike adaptation -> more sustained firing",
    ("adaptation", "increase"):   "increases spike adaptation",
    ("ih", "decrease"):           "reduces Ih",
    ("ih", "increase"):           "increases Ih",
}
SIGN = {("excitability", "increase"): 1, ("excitability", "decrease"): -1,
        ("release", "increase"): 1, ("release", "decrease"): -1, ("adaptation", "decrease"): 0.5,
        ("adaptation", "increase"): -0.5}


# ----------------------------------------------------------------------------
def _cite(c):
    return f"[{c['subject']} {c.get('locator') or ''}]".replace(" ]", "]")


def _conditions(kb, condition):
    """-> (condition codes, receptor symbols, labels)"""
    if condition is None:
        return set(), set(), []
    toks = [condition] if isinstance(condition, str) else list(condition)
    conds = {r["code"]: r for r in kb.db.execute(
        "SELECT c.code, c.kind, m.symbol AS modulator FROM condition c LEFT JOIN modulator m USING (modulator_id)")}
    recs = {r["symbol"]: r["mod"] for r in kb.db.execute(
        "SELECT r.symbol, m.symbol AS mod FROM receptor r JOIN modulator m USING (modulator_id)")}
    codes, receptors = set(), set()
    for t in toks:
        key = ALIASES.get(t.lower().strip(), t.strip())
        hit = next((c for c in conds if c.lower() == key.lower()), None)
        if hit:
            codes.add(hit)
            mod = conds[hit]["modulator"]
            receptors |= {s for s, m in recs.items() if m == mod} if mod else set()
            continue
        rhit = next((s for s in recs if s.lower() == key.lower()), None)
        if rhit:
            receptors.add(rhit)
            codes |= {c for c, r in conds.items() if r["modulator"] == recs[rhit]}
            continue
        raise ValueError(f"unknown condition {t!r}. Codes: {sorted(conds)}; receptors: {sorted(recs)}")
    return codes, receptors, toks


def _location(kb, loc, cell_scheme, region=None, parent=None):
    """-> (compartments, layer codes in the cell's scheme, label, pathway or None)"""
    if loc is None:
        return None
    if loc in LOCATIONS:
        comps, layers = LOCATIONS[loc]
        return comps, layers.get(cell_scheme, set()), loc, None
    pw = kb.db.execute("""SELECT p.code, l.code AS layer, l.scheme, pt.weight, r.code AS target
                          FROM pathway p JOIN pathway_termination pt USING (pathway_id) JOIN layer l USING (layer_id)
                          LEFT JOIN region r ON r.region_id = p.target_region_id WHERE p.code = ?""", (loc,)).fetchall()
    if pw:
        layers = {r["layer"] for r in pw if r["scheme"] == cell_scheme
                  and (r["target"] is None or r["target"] in (region, parent))}
        return set(), layers, f"{loc} pathway", loc
    lay = kb.db.execute("SELECT scheme FROM layer WHERE code = ?", (loc,)).fetchall()
    if lay:
        return set(), ({loc} if any(r["scheme"] == cell_scheme for r in lay) else set()), loc, None
    raise ValueError(f"unknown location {loc!r}. Use one of {sorted(LOCATIONS)}, a layer code or a pathway code")


def _fraction(profile, keys):
    if profile is None or not keys:
        return None
    return sum(v for k, v in profile["profile"].items() if k in keys)


def _pathways(kb, profile, cell):
    """Overlap of a laminar profile with each afferent pathway that terminates in the cell's region."""
    if profile is None:
        return []
    out = []
    for p in kb.db.execute("""SELECT p.code, p.name, sr.code AS src, tr.code AS tgt FROM pathway p
                              LEFT JOIN region sr ON sr.region_id = p.source_region_id
                              LEFT JOIN region tr ON tr.region_id = p.target_region_id"""):
        if p["tgt"] not in (cell["region"], cell["parent_region"]):
            continue
        term = {r["code"]: r["weight"] for r in kb.db.execute(
            """SELECT l.code, pt.weight FROM pathway_termination pt JOIN layer l USING (layer_id)
               JOIN pathway USING (pathway_id) WHERE pathway.code = ? AND l.scheme = ?""", (p["code"], cell["scheme"]))}
        tot = sum(term.values())
        if not tot:
            continue
        ov = sum(profile["profile"].get(l, 0.0) * w / tot for l, w in term.items())
        if ov > 0:
            local = p["src"] is not None and p["src"] in (cell["region"], cell["parent_region"])
            out.append({"pathway": p["code"], "name": p["name"], "overlap": ov,
                        "motif": "feedback / local" if local else "feedforward"})
    return sorted(out, key=lambda r: -r["overlap"])


def _netlist(kb, name, k=5):
    cid = kb.cell_id(name)
    q = """SELECT ct.netlist_name AS other, ct.ei, e.m_ij, e.w_ij FROM netlist_edge e JOIN netlist n USING (netlist_id)
           JOIN cell_type ct ON ct.cell_type_id = e.{other} WHERE n.name = ? AND e.{me} = ?"""
    ins = [dict(r) for r in kb.db.execute(q.format(other="pre_id", me="post_id"), (kb.netlist, cid))]
    outs = [dict(r) for r in kb.db.execute(q.format(other="post_id", me="pre_id"), (kb.netlist, cid))]
    tot_out = sum(abs(r["m_ij"]) for r in outs) or 1.0
    tot_w = sum(abs(r["w_ij"]) for r in outs) or 1.0
    exc_in = sorted([r for r in ins if r["ei"] == "E"], key=lambda r: -abs(r["m_ij"]))
    inh_in = sorted([r for r in ins if r["ei"] == "I"], key=lambda r: -abs(r["m_ij"]))
    return {"top_excitatory_inputs": exc_in[:k], "top_inhibitory_inputs": inh_in[:k],
            "top_targets": sorted(outs, key=lambda r: -abs(r["m_ij"]))[:k],
            "output_onto_principal": sum(abs(r["m_ij"]) for r in outs if r["ei"] == "E") / tot_out,
            "output_onto_principal_w": sum(abs(r["w_ij"]) for r in outs if r["ei"] == "E") / tot_w,
            "n_targets": len(outs), "n_inputs": len(ins)}


def _where(out_comp, out_layer):
    for prof, lab in ((out_comp, COMPARTMENT_LABEL), (out_layer, {})):
        if prof:
            items = sorted(prof["profile"].items(), key=lambda kv: -kv[1])
            keep = [k for k, v in items if v >= 0.25 and k != "unassigned"] or [items[0][0]]
            return " / ".join(lab.get(k, k) for k in keep) + " of its targets"
    return "its targets"


# ----------------------------------------------------------------------------
@dataclass
class CellFunction:
    cell: str
    family: str | None
    region: str
    markers: dict
    roles: list = field(default_factory=list)
    functions: list = field(default_factory=list)
    output: dict = field(default_factory=dict)
    input: dict = field(default_factory=dict)
    netlist: dict = field(default_factory=dict)
    condition: dict = field(default_factory=dict)
    location: dict = field(default_factory=dict)
    caveats: list = field(default_factory=list)

    def __str__(self):
        L = [f"{self.cell}  [{self.family or 'unmapped'}, {self.region}]"]
        known = {m: s for m, s in self.markers.items() if s != "unknown"}
        L.append("  markers: " + (", ".join(f"{m}{'+' if s == 'positive' else '-'}" for m, s in known.items())
                                   or "none established"))
        L.append("\nFUNCTION")
        for r in self.roles:
            L.append(f"  role: {r['text']}  {r['cite']}")
        for f in self.functions:
            L.append(f"  - {f['text']}  {f['cite']}")
        if not self.roles and not self.functions:
            L.append("  (no function statement in the KB for this cell or its family)")
        o = self.output
        L.append("\nOUTPUT (where it inhibits)")
        if o.get("compartments"):
            L.append("  compartments: " + _fmt(o["compartments"]) + f"  [{o['compartments_from']}]")
        if o.get("layers"):
            L.append("  layers: " + _fmt(o["layers"]) + f"  [{o['layers_from']}]")
        for p in o.get("gates", []):
            L.append(f"  co-localised with {p['name']} ({p['pathway']}) input: overlap {p['overlap']:.2f}")
        n = self.netlist
        if n:
            L.append(f"  netlist: {n['n_targets']} target populations; share of weight onto principal cells "
                     f"{100 * n['output_onto_principal']:.0f}% (m_ij) / {100 * n['output_onto_principal_w']:.0f}% (w_ij); "
                     f"strongest: " + ", ".join(t["other"] for t in n["top_targets"][:3]))
        i = self.input
        L.append("\nINPUT (what recruits it)")
        if i.get("layers"):
            L.append("  dendrites: " + _fmt(i["layers"]) + f"  [{i['layers_from']}]")
        for p in i.get("recruited_by", []):
            L.append(f"  recruited by {p['name']} ({p['pathway']}, {p['motif']}): overlap {p['overlap']:.2f}")
        if n:
            L.append("  strongest excitatory drivers (netlist): "
                     + ", ".join(t["other"] for t in n["top_excitatory_inputs"][:3]))
            if n["top_inhibitory_inputs"]:
                L.append("  strongest inhibitory inputs (netlist): "
                         + ", ".join(t["other"] for t in n["top_inhibitory_inputs"][:3]))
        if self.location:
            L.append("\nLOCATION")
            for line in self.location.get("lines", []):
                L.append("  " + line)
        if self.condition:
            c = self.condition
            L.append(f"\nUNDER {', '.join(c['labels'])}")
            for e in c["effects"]:
                L.append(f"  - {e['text']}  {e['cite']}")
            if not c["effects"]:
                L.append("  (no condition-specific claims for this cell or its family)")
            L.append(f"  net effect on its output: {c['net']}")
        if self.caveats:
            L.append("\nCAVEATS")
            for cv in self.caveats:
                L.append("  * " + cv)
        return "\n".join(L)

    def _repr_pretty_(self, p, cycle):      # IPython / Jupyter
        p.text(str(self))


def _fmt(d):
    return ", ".join(f"{k} {v:.2f}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]) if v > 0)


# ----------------------------------------------------------------------------
def describe(kb, cell, condition=None, stimulus_at=None, inhibits_at=None, max_functions=6):
    """Functional profile of one netlist population, optionally under a condition and at a location."""
    meta = kb.cell(cell)
    if meta["ei"] != "I":
        raise ValueError(f"{cell} is excitatory in the netlist; describe() covers interneurons")
    pt = kb.primary_taxon(cell) or {}
    codes, receptors, labels = _conditions(kb, condition)
    cond_ids = {r["condition_id"]: r["code"] for r in kb.db.execute("SELECT condition_id, code FROM condition")}

    markers = {m: kb.marker_status(cell, m)[0] for m in ("PV", "SST", "CCK", "CB1", "NPY", "VIP", "nNOS", "5HT3A")}
    res = CellFunction(cell=cell, family=pt.get("code"), region=meta["region"], markers=markers)

    # function statements
    for c in kb.claims(cell, "circuit_role"):
        res.roles.append({"text": c["value_cat"].replace("_", " "), "cite": _cite(c), "rank": c["rank"]})
        break
    seen = set()
    for c in kb.claims(cell, "function_statement"):
        if c["condition_id"] is not None or c["value_text"] in seen:
            continue
        seen.add(c["value_text"])
        res.functions.append({"text": c["value_text"], "cite": _cite(c), "rank": c["rank"],
                              "confidence": c["confidence"] * c["mapping_conf"]})
    res.functions = res.functions[:max_functions]

    # anatomy
    out_layer, out_comp = kb.layer_profile(cell, "axon"), kb.layer_profile(cell, "compartment")
    dend = kb.layer_profile(cell, "dendrite")
    if out_comp:
        res.output.update(compartments=out_comp["profile"], compartments_from=f"{out_comp['subject']} {out_comp['locator']}")
    if out_layer:
        res.output.update(layers=out_layer["profile"], layers_from=f"{out_layer['subject']} {out_layer['locator']}")
    res.output["gates"] = [p for p in _pathways(kb, out_layer, meta) if p["overlap"] >= 0.15]
    if dend:
        res.input.update(layers=dend["profile"], layers_from=f"{dend['subject']} {dend['locator']}")
    res.input["recruited_by"] = [p for p in _pathways(kb, dend, meta) if p["overlap"] >= 0.1]
    res.netlist = _netlist(kb, cell)
    where = _where(out_comp, out_layer)

    # location
    if stimulus_at or inhibits_at:
        lines = []
        if stimulus_at:
            comps, layers, label, pw = _location(kb, stimulus_at, meta["scheme"], meta["region"], meta["parent_region"])
            f = _fraction(dend, layers)
            if dend is None:
                lines.append(f"input at {label}: unknown (no dendritic-layer claim for this cell or family)")
            elif not layers:
                lines.append(f"input at {label}: does not apply to a {meta['region']} cell")
            else:
                verdict = "strongly placed" if f >= 0.4 else "partly placed" if f >= 0.1 else "not placed"
                lines.append(f"input at {label}: {100 * f:.0f}% of dendritic territory -> {verdict} to be recruited")
            res.location["stimulus_fraction"] = f
        if inhibits_at:
            comps, layers, label, _ = _location(kb, inhibits_at, meta["scheme"], meta["region"], meta["parent_region"])
            fc = _fraction(out_comp, comps) if comps else None
            fl = _fraction(out_layer, layers) if layers else None
            f = fc if fc is not None else fl
            if f is None:
                lines.append(f"output at {label}: unknown (no axon claim for this cell or family)")
            else:
                verdict = "main target" if f >= 0.4 else "partial target" if f >= 0.1 else "does not target"
                basis = "compartment" if fc is not None else "layer"
                lines.append(f"output at {label}: {100 * f:.0f}% of output ({basis} basis) -> {verdict}")
            res.location["inhibition_fraction"] = f
        res.location["lines"] = lines

    # condition
    if condition is not None:
        effects, net, signed = [], 0.0, False
        for c in kb.claims(cell, "function_statement"):
            if cond_ids.get(c["condition_id"]) in codes:
                effects.append({"text": c["value_text"], "cite": _cite(c)})
        for rec in sorted(receptors):
            for c in kb.claims(cell, "receptor_effect", receptor=rec):
                key = (c["mechanism"], c["value_cat"])
                phrase = EFFECT_PHRASE.get(key, f"{c['mechanism']}: {c['value_cat']}").format(where=where)
                extra = f" [{c['value_text']}]" if c["value_text"] else ""
                effects.append({"text": f"{rec}: {phrase}{extra}", "cite": _cite(c)})
                net += SIGN.get(key, 0) * c["confidence"] * c["mapping_conf"]
                signed = signed or key in SIGN
            if not kb.claims(cell, "receptor_effect", receptor=rec):
                for c in kb.claims(cell, "receptor_expression", receptor=rec)[:1]:
                    if c["value_cat"] in ("positive", "present", "high", "subset"):
                        effects.append({"text": f"expresses {rec} ({c['value_cat']}); effect not characterised",
                                        "cite": _cite(c)})
        for prop in ("theta_phase_locking", "theta_phase_preference", "excitatory_input_stp", "release_mode"):
            for c in kb.claims(cell, prop)[:1]:
                if cond_ids.get(c["condition_id"]) in codes:
                    effects.append({"text": f"{prop.replace('_', ' ')}: {c['value_cat']}"
                                    + (f" ({c['value_text']})" if c["value_text"] else ""), "cite": _cite(c)})
        if not effects:
            net_txt = "unknown (no claims)"
        elif net > 0.3:
            net_txt = "more inhibition of its targets"
        elif net < -0.3:
            net_txt = "less inhibition (disinhibits its targets)"
        elif signed:
            net_txt = "mixed / weak"
        else:
            net_txt = "no signed receptor effect recorded; see statements"
        res.condition = {"labels": labels, "codes": sorted(codes), "receptors": sorted(receptors),
                         "effects": effects, "net": net_txt, "net_score": round(net, 2)}

    # caveats
    if pt.get("confidence", 1) < 0.6:
        res.caveats.append(f"family mapping to {pt.get('code')} has confidence {pt.get('confidence')}; "
                           "statements are inherited from that family")
    if any(r["rank"] > 1 for r in res.functions):
        res.caveats.append("some statements come from a parent family, not the exact subtype")
    if meta["scheme"] == "hippocampal_CA" and meta["region"] != "CA1" and (dend or out_layer):
        res.caveats.append(f"laminar anatomy comes from CA1 descriptions applied to a {meta['region']} cell")
    if res.netlist.get("n_targets"):
        res.caveats.append("netlist shares are of per-target-population weight, not synapse counts; m_ij also "
                           "carries the target's kappa_j, which inflates interneuron targets")
    if stimulus_at:
        res.caveats.append("stimulus_at reports whether input there reaches the cell's dendrites; "
                           "proximal-vs-distal integration inside the interneuron is not modelled")
    return res


def find(kb, stimulus_at=None, inhibits_at=None, condition=None, region=None, min_fraction=0.1):
    """Table of interneurons matching location / region filters, with their condition response."""
    rows = []
    for cell in kb.interneurons():
        meta = kb.cell(cell)
        if region and region not in (meta["region"], meta["parent_region"]):
            continue
        d = describe(kb, cell, condition=condition, stimulus_at=stimulus_at, inhibits_at=inhibits_at)
        fs, fi = d.location.get("stimulus_fraction"), d.location.get("inhibition_fraction")
        if stimulus_at and (fs is None or fs < min_fraction):
            continue
        if inhibits_at and (fi is None or fi < min_fraction):
            continue
        out = d.output.get("compartments") or d.output.get("layers") or {}
        rows.append({
            "cell": cell, "family": d.family, "region": d.region,
            "input_frac_at_location": None if fs is None else round(fs, 2),
            "output_frac_at_location": None if fi is None else round(fi, 2),
            "main_output": max(out, key=out.get) if out else None,
            "role": d.roles[0]["text"] if d.roles else None,
            "gates": ", ".join(p["pathway"] for p in d.output.get("gates", [])) or None,
            "recruited_by": ", ".join(p["pathway"] for p in d.input.get("recruited_by", [])) or None,
            "condition_net": d.condition.get("net") if condition else None,
            "condition_effects": (" | ".join(e["text"] for e in d.condition.get("effects", [])) or None)
                                 if condition else None,
            "function": d.functions[0]["text"] if d.functions else None,
        })
    key = "output_frac_at_location" if inhibits_at else "input_frac_at_location" if stimulus_at else None
    if pd is None:
        return rows
    df = pd.DataFrame(rows)
    if key and not df.empty:
        df = df.sort_values(key, ascending=False)
    return df.reset_index(drop=True)


# convenience: kb.describe(...), kb.find(...)
def attach(kb_cls):
    kb_cls.describe = lambda self, *a, **k: describe(self, *a, **k)
    kb_cls.find = lambda self, *a, **k: find(self, *a, **k)
    return kb_cls


try:
    from resolver import KB as _KB
    attach(_KB)
except ImportError:  # pragma: no cover
    pass


# ============================================================================
# "What is this cell known to do?"  -- literature / experiment report
# ============================================================================
SECTIONS = [
    ("identity", "Identity", ["marker_status", "embryonic_origin"]),
    ("anatomy", "Anatomy & connectivity",
     ["soma_layer_distribution", "dendrite_layer_distribution", "axon_layer_distribution",
      "axon_compartment_distribution", "output_target_class_distribution", "postsynaptic_cells_per_neuron",
      "synapses_per_connection", "terminal_ratio_PV_to_CCK", "crosses_fissure", "axon_total_length", "bouton_count",
      "bouton_density", "bouton_target_distance", "fraction_of_CA1_interneurons", "cell_count_CA1"]),
    ("physiology", "Intrinsic physiology",
     ["firing_pattern", "V_rest", "V_th", "R_in", "tau_m", "sag_index", "f_2x", "adaptation_ratio", "AP_amplitude",
      "AP_halfwidth", "AHP_amplitude", "AHP_halfwidth", "ih_prominence", "gap_junction_coupling",
      "gap_junction_conductance", ("experimental_finding", "physiology")]),
    ("synaptic", "Synaptic input & output",
     ["excitatory_input_stp", "release_mode", "release_probability", "quantal_release_halfduration",
      "synchronicity_ratio", "conduction_velocity", "gabaa_decay_class", "gabaa_decay_ratio_vs_fast", "gabab_output"]),
    ("function", "Circuit function", ["circuit_role", "function_statement"]),
    ("neuromodulation", "Neuromodulation", ["receptor_effect", "receptor_expression", "cb1_relative_strength"]),
    ("oscillation", "Network oscillations",
     ["theta_phase_locking", "theta_phase_preference", ("experimental_finding", "oscillation")]),
    ("plasticity", "Plasticity", [("experimental_finding", "plasticity")]),
    ("behavior", "Behavior", [("experimental_finding", "behavior")]),
    ("disease", "Disease & injury", [("experimental_finding", "disease")]),
]
EVIDENCE_TAG = {"measurement": "measured", "review_summary": "review", "secondary_citation": "review-cited data",
                "curator_inference": "curator inference", "modeling_assumption": "assumption",
                "name_parse": "name", "model_estimate": "model fit"}


def _refs(kb, cited):
    if not cited:
        return []
    out = []
    for n in re.findall(r"\d+", cited):
        r = kb.db.execute("""SELECT rr.first_author, rr.year, rr.doi, rr.citation FROM review_reference rr
                             JOIN source s USING (source_id) WHERE s.short_ref='Pelkey2017' AND ref_num=?""",
                          (int(n),)).fetchone()
        if r:
            out.append({"ref": int(n), "short": f"{r['first_author']} {r['year'] or ''}".strip(),
                        "doi": r["doi"], "citation": r["citation"]})
    return out


def _claim_text(kb, c, prop):
    name = kb.db.execute("SELECT name, unit FROM property WHERE code=?", (prop,)).fetchone()
    label, unit = name["name"], name["unit"] or ""
    unit = "" if unit in ("ratio", "fraction", "dimensionless") else f" {unit}"
    tail = f" ({c['value_text']})" if c["value_text"] and prop not in (
        "function_statement", "experimental_finding") else ""
    if prop == "marker_status":
        sym = kb.db.execute("SELECT symbol FROM marker WHERE marker_id=?", (c["marker_id"],)).fetchone()[0]
        sign = {"positive": "+", "negative": "-", "low": " low", "mixed": " (subset)", "unknown": " ?"}
        return f"{sym}{sign.get(c['value_cat'], ' ' + str(c['value_cat']))}{tail}"
    if prop in ("receptor_effect", "receptor_expression", "cb1_relative_strength"):
        rec = kb.db.execute("SELECT symbol FROM receptor WHERE receptor_id=?", (c["receptor_id"],)).fetchone()
        rec = rec[0] if rec else "?"
        mech = f" on {c['mechanism']}" if c["mechanism"] else ""
        verb = {"receptor_effect": "effect", "receptor_expression": "expression",
                "cb1_relative_strength": "relative strength"}[prop]
        return f"{rec} {verb}{mech}: {c['value_cat']}{tail}"
    if prop in ("function_statement", "experimental_finding"):
        return c["value_text"]
    if c["value_num"] is not None:
        sd = f" ± {c['value_sd']:g}" if c["value_sd"] is not None else ""
        n = f", n={c['n']}" if c["n"] else ""
        approx = "~" if c["stat"] in ("approx",) else "≤" if c["stat"] == "upper_bound" else \
                 "≥" if c["stat"] == "lower_bound" else ""
        return f"{label}: {approx}{c['value_num']:g}{sd}{unit}{n}{tail}"
    if c["value_lo"] is not None:
        return f"{label}: {c['value_lo']:g}–{c['value_hi']:g}{unit}{tail}"
    if c["value_cat"] is not None:
        return f"{label}: {c['value_cat'].replace('_', ' ')}{tail}"
    return f"{label}: {c['value_text']}"


@dataclass
class KnownReport:
    cell: str
    family: str | None
    sections: dict
    filters: dict
    caveats: list = field(default_factory=list)

    def __str__(self):
        hdr = f"{self.cell}  [{self.family or 'unmapped'}]"
        if any(self.filters.values()):
            hdr += "   filters: " + ", ".join(f"{k}={v}" for k, v in self.filters.items() if v)
        L = [hdr]
        if not any(self.sections.values()):
            L.append("  (nothing in the KB matches)")
        for title, items in self.sections.items():
            if not items:
                continue
            L.append(f"\n{title.upper()}")
            for it in items:
                cond = f" [under {it['condition']}]" if it["condition"] else ""
                refs = "; ".join(r["short"] for r in it["refs"][:4]) + (" ..." if len(it["refs"]) > 4 else "")
                src = f"{it['subject']} {it['locator']}".strip()
                L.append(f"  - {it['text']}{cond}")
                L.append(f"      {src} | {it['evidence']} | conf {it['confidence']:.2f}" + (f" | {refs}" if refs else ""))
        if self.caveats:
            L.append("\nCAVEATS")
            L += ["  * " + c for c in self.caveats]
        return "\n".join(L)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def to_frame(self):
        rows = [dict(section=s, **{k: v for k, v in it.items() if k != "refs"},
                     primary_refs="; ".join(r["short"] for r in it["refs"]),
                     primary_dois="; ".join(r["doi"] or "" for r in it["refs"]))
                for s, items in self.sections.items() for it in items]
        return pd.DataFrame(rows) if pd else rows


def known(kb, cell, topic=None, condition=None, min_confidence=0.0):
    """Everything the KB holds about what this population is known to do, grouped by topic, each line traced to
    the review page and the primary papers it cites.

    topic     : None or one/several of identity, anatomy, physiology, synaptic, function, neuromodulation,
                oscillation, plasticity, behavior, disease
    condition : restrict to findings that hold under a condition (same vocabulary as describe())
    """
    meta = kb.cell(cell)
    pt = kb.primary_taxon(cell) or {}
    topics = None if topic is None else {topic} if isinstance(topic, str) else set(topic)
    bad = (topics or set()) - {s[0] for s in SECTIONS}
    if bad:
        raise ValueError(f"unknown topic(s) {sorted(bad)}; use {[s[0] for s in SECTIONS]}")
    codes, receptors, labels = _conditions(kb, condition)
    cond_name = {r["condition_id"]: r["code"] for r in kb.db.execute("SELECT condition_id, code FROM condition")}
    rec_sym = {r["receptor_id"]: r["symbol"] for r in kb.db.execute("SELECT receptor_id, symbol FROM receptor")}

    sections = {}
    for key, title, props in SECTIONS:
        if topics and key not in topics:
            continue
        items, seen = [], set()
        for p in props:
            prop, sub = (p if isinstance(p, tuple) else (p, None))
            claims = kb.claims(cell, prop)
            if sub:
                claims = [c for c in claims if c["value_cat"] == sub]
            # keep the most specific claim per (property, marker, receptor, mechanism) for scalar properties
            if prop not in ("function_statement", "experimental_finding", "receptor_effect"):
                best = {}
                for c in claims:
                    k = (c["marker_id"], c["receptor_id"], c["mechanism"], c["condition_id"])
                    best.setdefault(k, c)
                claims = list(best.values())
            for c in claims:
                cname = cond_name.get(c["condition_id"])
                if condition is not None:
                    hit = cname in codes or rec_sym.get(c["receptor_id"]) in receptors
                    if not hit:
                        continue
                conf = c["confidence"] * c["mapping_conf"]
                if conf < min_confidence:
                    continue
                text = _claim_text(kb, c, prop)
                if (text, cname) in seen:
                    continue
                seen.add((text, cname))
                items.append({"text": text, "condition": cname, "subject": c["subject"], "rank": c["rank"],
                              "locator": c["locator"] or "", "confidence": conf,
                              "evidence": EVIDENCE_TAG.get(c["evidence_class"], c["evidence_class"]),
                              "refs": _refs(kb, c.get("cited_ref") or kb.db.execute(
                                  "SELECT cited_ref FROM evidence WHERE evidence_id=?",
                                  (c["evidence_id"],)).fetchone()[0])})
        items.sort(key=lambda it: (it["rank"], -it["confidence"]))
        sections[title] = items

    rep = KnownReport(cell=cell, family=pt.get("code"), sections=sections,
                      filters={"topic": topic, "condition": ", ".join(labels) if labels else None})
    if pt.get("confidence", 1) < 0.6:
        rep.caveats.append(f"mapping to {pt.get('code')} has confidence {pt.get('confidence')}; "
                           "family-level findings may not apply to this exact type")
    fam_only = all(it["subject"] != cell for items in sections.values() for it in items)
    if fam_only and any(sections.values()):
        rep.caveats.append("no finding in the KB is about this exact netlist type; everything is inherited from "
                           "its family (mostly CA1 rodent work)")
    if pt.get("code") in ("BASKET",):
        rep.caveats.append("bare 'Basket' is molecularly unresolved: PVBC- and CCKBC-specific findings are NOT "
                           "shown; query those families with findings(kb, family='PVBC') to compare")
    rep.caveats.append("source coverage: Pelkey et al. 2017 only (plus its cited primary papers); absence of a "
                       "finding here is not evidence of absence")
    return rep


def findings(kb, topic=None, condition=None, contains=None, family=None):
    """Literature findings across the whole KB, one row per finding at the level it was reported
    (family or cell), with the netlist populations that inherit it. Good for 'which cells fire during SWRs?'."""
    codes, receptors, labels = _conditions(kb, condition)
    q = """SELECT pc.*, p.code AS prop, e.locator, e.cited_ref, e.evidence_class, t.code AS taxon,
                  ct.netlist_name AS cell, c.code AS cond, r.symbol AS rec
           FROM property_claim pc JOIN property p USING (property_id) JOIN evidence e USING (evidence_id)
           LEFT JOIN taxon t ON t.taxon_id = pc.taxon_id LEFT JOIN cell_type ct ON ct.cell_type_id = pc.cell_type_id
           LEFT JOIN condition c ON c.condition_id = pc.condition_id LEFT JOIN receptor r ON r.receptor_id = pc.receptor_id
           WHERE p.code IN ('experimental_finding','function_statement','circuit_role','receptor_effect',
                            'theta_phase_locking','theta_phase_preference') AND pc.superseded_by IS NULL"""
    rows = []
    for c in kb.db.execute(q):
        c = dict(c)
        t = c["value_cat"] if c["prop"] == "experimental_finding" else \
            "function" if c["prop"] in ("function_statement", "circuit_role") else \
            "neuromodulation" if c["prop"] == "receptor_effect" else "oscillation"
        if topic and t not in ({topic} if isinstance(topic, str) else set(topic)):
            continue
        if condition is not None and not (c["cond"] in codes or c["rec"] in receptors):
            continue
        text = _claim_text(kb, c, c["prop"])
        if contains and contains.lower() not in text.lower():
            continue
        subj = c["taxon"] or c["cell"]
        if family and subj != family:
            continue
        if c["taxon"]:
            inherit = [r[0] for r in kb.db.execute("""
                SELECT DISTINCT ct.netlist_name FROM cell_type_taxon x JOIN cell_type ct USING (cell_type_id)
                JOIN v_taxon_ancestry a ON a.taxon_id = x.taxon_id
                WHERE a.ancestor_id = ? AND x.is_primary = 1 ORDER BY 1""", (c["taxon_id"],))]
        else:
            inherit = [c["cell"]]
        rows.append({"topic": t, "subject": subj, "condition": c["cond"], "finding": text,
                     "locator": c["locator"], "evidence": EVIDENCE_TAG.get(c["evidence_class"]),
                     "confidence": c["confidence"],
                     "primary_refs": "; ".join(r["short"] for r in _refs(kb, c["cited_ref"])),
                     "netlist_cells": ", ".join(inherit) if inherit else "(none in netlist)"})
    df = pd.DataFrame(rows) if pd else rows
    if pd is not None and not df.empty:
        df = df.sort_values(["topic", "subject"]).reset_index(drop=True)
    return df


def _attach_known(kb_cls):
    kb_cls.known = lambda self, *a, **k: known(self, *a, **k)
    kb_cls.findings = lambda self, *a, **k: findings(self, *a, **k)


try:
    _attach_known(_KB)
except NameError:  # pragma: no cover
    pass
