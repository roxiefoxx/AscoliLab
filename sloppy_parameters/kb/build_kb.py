"""Build the hippocampal interneuron knowledge base (SQLite).

    python build_kb.py                       # uses ../mij_netlist.csv, writes hippocampal_interneuron_kb.sqlite
    python build_kb.py --netlist path.csv --out kb.sqlite

Everything scientific lives in seed/*.csv so it can be curated in a spreadsheet
and diffed in git; this script only loads, links and validates.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = HERE / "seed"

# ----------------------------------------------------------------------------
# Controlled vocabularies
# ----------------------------------------------------------------------------
REGIONS = [  # code, name, parent, scheme
    ("DG", "Dentate gyrus", None, "dentate"),
    ("CA3", "Cornu Ammonis 3", None, "hippocampal_CA"),
    ("CA3c", "CA3c (hilar end of CA3)", "CA3", "hippocampal_CA"),
    ("CA2", "Cornu Ammonis 2", None, "hippocampal_CA"),
    ("CA1", "Cornu Ammonis 1", None, "hippocampal_CA"),
    ("SUB", "Subiculum", None, "subiculum"),
    ("EC", "Entorhinal cortex", None, "entorhinal"),
    ("LEC", "Lateral entorhinal cortex", "EC", "entorhinal"),
    ("MEC", "Medial entorhinal cortex", "EC", "entorhinal"),
]
LAYERS = [  # scheme, code, name, ordinal
    ("hippocampal_CA", "ALV", "alveus", 0), ("hippocampal_CA", "SO", "stratum oriens", 1),
    ("hippocampal_CA", "SP", "stratum pyramidale", 2), ("hippocampal_CA", "SL", "stratum lucidum", 3),
    ("hippocampal_CA", "SR", "stratum radiatum", 4), ("hippocampal_CA", "SLM", "stratum lacunosum-moleculare", 5),
    ("dentate", "OML", "outer molecular layer", 0), ("dentate", "MML", "middle molecular layer", 1),
    ("dentate", "IML", "inner molecular layer", 2), ("dentate", "SG", "stratum granulosum", 3),
    ("dentate", "H", "hilus", 4),
    ("subiculum", "ML", "molecular layer", 0), ("subiculum", "PL", "pyramidal layer", 1),
] + [("entorhinal", f"L{r}", f"layer {r}", i) for i, r in enumerate(["I", "II", "III", "IV", "V", "VI"], 1)]
COMPARTMENTS = ["AIS", "soma", "proximal_dendrite", "basal_dendrite", "oblique_dendrite",
                "apical_shaft", "distal_dendrite", "dendrite_unspecified", "nonsynaptic_volume"]
MARKERS = [  # symbol, name, kind
    ("PV", "parvalbumin", "calcium_binding"), ("CB", "calbindin", "calcium_binding"),
    ("CR", "calretinin", "calcium_binding"), ("SST", "somatostatin", "neuropeptide"),
    ("CCK", "cholecystokinin", "neuropeptide"), ("NPY", "neuropeptide Y", "neuropeptide"),
    ("VIP", "vasoactive intestinal peptide", "neuropeptide"), ("nNOS", "neuronal nitric oxide synthase", "enzyme"),
    ("Reelin", "reelin", "other"), ("VGluT3", "vesicular glutamate transporter 3", "transporter"),
    ("COUPTFII", "COUP transcription factor 2", "transcription_factor"),
    ("SATB1", "SATB1", "transcription_factor"), ("mGluR1a", "mGluR1 alpha", "receptor"),
    ("Elfn1", "Elfn1", "adhesion"), ("Syt2", "synaptotagmin 2", "synaptic_protein"),
    ("5HT3A", "5-HT3A receptor", "receptor"), ("CB1", "cannabinoid receptor 1", "receptor"),
]
MODULATORS = [("ACh", "acetylcholine"), ("5-HT", "serotonin"), ("DA", "dopamine"), ("eCB", "endocannabinoids"),
              ("opioid", "opioid peptides"), ("CCK", "cholecystokinin (peptide)"), ("OXT", "oxytocin"),
              ("GABA", "GABA (presynaptic / extrasynaptic)")]
RECEPTORS = [  # symbol, modulator, mechanism, marker
    ("CB1", "eCB", "metabotropic", "CB1"), ("M1", "ACh", "metabotropic", None), ("M2", "ACh", "metabotropic", None),
    ("M3", "ACh", "metabotropic", None), ("M4", "ACh", "metabotropic", None),
    ("nAChR_a7", "ACh", "ionotropic", None), ("5HT3A", "5-HT", "ionotropic", "5HT3A"),
    ("5HT2A", "5-HT", "metabotropic", None), ("MOR", "opioid", "metabotropic", None),
    ("DOR", "opioid", "metabotropic", None), ("KOR", "opioid", "metabotropic", None),
    ("D1", "DA", "metabotropic", None), ("D2", "DA", "metabotropic", None), ("D4", "DA", "metabotropic", None),
    ("CCK2", "CCK", "metabotropic", None), ("OXTR", "OXT", "metabotropic", None),
    ("GABAB", "GABA", "metabotropic", None),
]
PATHWAYS = [  # code, name, source, target, [(scheme, layer, weight)]
    ("SC", "Schaffer collateral", "CA3", "CA1", [("hippocampal_CA", "SR", 1.0), ("hippocampal_CA", "SO", 0.6)]),
    ("TA", "Temporoammonic (EC -> CA1)", "EC", "CA1", [("hippocampal_CA", "SLM", 1.0)]),
    ("NRe", "Nucleus reuniens -> CA1", None, "CA1", [("hippocampal_CA", "SLM", 1.0)]),
    ("PP_CA3", "Perforant path (EC -> CA3 s.l.m.)", "EC", "CA3", [("hippocampal_CA", "SLM", 1.0)]),
    ("PP", "Perforant path (EC -> DG)", "EC", "DG", [("dentate", "OML", 0.6), ("dentate", "MML", 0.6)]),
    ("MF", "Mossy fibers (DG -> CA3)", "DG", "CA3", [("hippocampal_CA", "SL", 1.0), ("dentate", "H", 0.2)]),
    ("CA3_rec", "CA3 recurrent collaterals", "CA3", "CA3", [("hippocampal_CA", "SR", 0.6), ("hippocampal_CA", "SO", 0.6)]),
    ("CA1_rec", "CA1 pyramidal collaterals", "CA1", "CA1", [("hippocampal_CA", "SO", 1.0), ("hippocampal_CA", "ALV", 0.2)]),
    ("HC_assoc", "Hilar commissural-associational", "DG", "DG", [("dentate", "IML", 1.0)]),
]
SOURCES = [  # short_ref, citation, doi, pmid, year, type
    ("Pelkey2017", "Pelkey KA, Chittajallu R, Craig MT, Tricoire L, Wester JC, McBain CJ. Hippocampal GABAergic "
     "Inhibitory Interneurons. Physiol Rev 97:1619-1747, 2017.", "10.1152/physrev.00007.2017", "28954853", 2017, "review"),
    ("mij_netlist", "AscoliLab m_ij netlist (mij_netlist.csv); m_ij = q_i c_ij g_ij (V_ref - V_rev) tau_ij kappa_j",
     None, None, 2026, "netlist"),
    ("curator", "Curator decisions recorded in seed/*.csv", None, None, 2026, "curator"),
    ("assumption", "Modeling defaults chosen to make the ODE runnable; replace with data", None, None, 2026, "assumption"),
]
NETLIST_FACTORS = [  # code, description, embedded, overlaps_param
    ("q_i", "Presynaptic census count (Hippocampome)", 1, None),
    ("c_ij", "Axonal-dendritic connection probability (laminar overlap)", 1, "a_layer,d_layer"),
    ("g_ij", "Unitary conductance (TPM)", 1, None),
    ("driving_force", "(V_ref - V_rev)", 1, None),
    ("tau_ij", "Synaptic decay (TPM): folded into the weight as charge", 1, "tau_A"),
    ("kappa_j", "Target transfer-function slope from the Izhikevich model", 1, "tau,eta,gamma_in"),
    ("U_stp", "Short-term plasticity (U, tau_r, tau_f) - NOT included", 0, "U_A"),
]
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}


def read_csv(name):
    with open(SEED / name, newline="", encoding="utf-8") as fh:
        return [{k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()} for row in csv.DictReader(fh)]


def nz(x, cast=str):
    """'' -> None, otherwise cast."""
    if x is None or x == "":
        return None
    return cast(x)


# ----------------------------------------------------------------------------
# Netlist name parsing
# ----------------------------------------------------------------------------
REGION_CODES = {r[0] for r in REGIONS}


def parse_name(name: str) -> dict:
    toks = name.split()
    region = toks[0]
    if region not in REGION_CODES:
        raise ValueError(f"unknown region token in {name!r}")
    rest = toks[1:]
    layers, proj = [], None
    # Entorhinal layer tokens: 'LI II', 'LIII V', 'LV VI' ... (inclusive range)
    if rest and re.fullmatch(r"L[IV]+", rest[0]):
        first = ROMAN[rest[0][1:]]
        last = first
        rest = rest[1:]
        if rest and re.fullmatch(r"[IV]+", rest[0]) and rest[0] in ROMAN:
            last = ROMAN[rest[0]]
            rest = rest[1:]
        layers = list(range(first, last + 1))
    # 'SUB CA1 Projecting Pyramidal' -> projection target
    if region == "SUB" and rest and rest[0] in REGION_CODES:
        proj, rest = rest[0], rest[1:]
    inv = {v: k for k, v in ROMAN.items()}
    token = None
    if layers:
        token = "L" + "-".join(inv[l] for l in ([layers[0], layers[-1]] if len(layers) > 1 else layers))
    return {"region": region, "layers": [f"L{inv[l]}" for l in layers], "layer_token": token,
            "projection_target": proj, "name_core": " ".join(rest)}


# ----------------------------------------------------------------------------
class Builder:
    def __init__(self, db_path: Path):
        if db_path.exists():
            db_path.unlink()
        self.db = sqlite3.connect(db_path)
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript((HERE / "schema.sql").read_text(encoding="utf-8"))
        self.ids: dict[tuple, int] = {}

    def ins(self, table, **cols):
        cols = {k: v for k, v in cols.items()}
        sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
        return self.db.execute(sql, list(cols.values())).lastrowid

    def id(self, kind, *key):
        try:
            return self.ids[(kind,) + key]
        except KeyError as e:
            raise KeyError(f"unknown {kind}: {key}") from e

    def evidence(self, source, cls, locator=None, cited_ref=None, quote=None):
        return self.ins("evidence", source_id=self.id("source", source), evidence_class=cls,
                        locator=locator, cited_ref=cited_ref, quote=quote)

    # --------------------------------------------------------------- vocab
    def vocab(self):
        for code, name, parent, scheme in REGIONS:
            self.ids[("region", code)] = self.ins("region", code=code, name=name, scheme=scheme,
                                                  parent_region_id=self.ids.get(("region", parent)))
        for scheme, code, name, o in LAYERS:
            self.ids[("layer", scheme, code)] = self.ins("layer", scheme=scheme, code=code, name=name, ordinal=o)
        for o, c in enumerate(COMPARTMENTS):
            self.ids[("compartment", c)] = self.ins("compartment", code=c, name=c.replace("_", " "), ordinal=o)
        for s, n, k in MARKERS:
            self.ids[("marker", s)] = self.ins("marker", symbol=s, name=n, kind=k)
        for s, n in MODULATORS:
            self.ids[("modulator", s)] = self.ins("modulator", symbol=s, name=n)
        for s, m, mech, mk in RECEPTORS:
            self.ids[("receptor", s)] = self.ins("receptor", symbol=s, modulator_id=self.id("modulator", m),
                                                 mechanism=mech, marker_id=self.ids.get(("marker", mk)))
        for short, cit, doi, pmid, yr, typ in SOURCES:
            self.ids[("source", short)] = self.ins("source", short_ref=short, citation=cit, doi=doi, pmid=pmid,
                                                   year=yr, source_type=typ)
        pk = self.id("source", "Pelkey2017")
        for r in read_csv("pelkey2017_references.csv"):
            self.ins("review_reference", source_id=pk, ref_num=int(r["ref_num"]), first_author=nz(r["first_author"]),
                     year=nz(r["year"], int), doi=nz(r["doi"]), citation=r["citation"])
        for code, name, src, tgt, terms in PATHWAYS:
            pid = self.ins("pathway", code=code, name=name, source_region_id=self.ids.get(("region", src)),
                           target_region_id=self.ids.get(("region", tgt)))
            for scheme, layer, w in terms:
                self.ins("pathway_termination", pathway_id=pid, layer_id=self.id("layer", scheme, layer), weight=w)

    # ---------------------------------------------------------- cell types
    def cell_types(self, netlist_csv: Path):
        with open(netlist_csv, newline="", encoding="utf-8") as fh:
            edges = list(csv.DictReader(fh))
        ei = {}
        for e in edges:
            for who, flag in ((e["pre_neuron"], e["pre_ei"]), (e["post_neuron"], e["post_ei"])):
                f = flag.strip().upper()
                if ei.setdefault(who, f) != f:
                    raise ValueError(f"inconsistent E/I flag for {who}")
        ev = self.evidence("curator", "name_parse", locator="build_kb.parse_name",
                           quote="Region / entorhinal layer / projection target parsed from the netlist label; "
                                 "compound layer tokens are read as inclusive ranges with equal weights")
        for name in sorted(ei):
            p = parse_name(name)
            cid = self.ins("cell_type", netlist_name=name, region_id=self.id("region", p["region"]),
                           layer_token=p["layer_token"],
                           projection_target_id=self.ids.get(("region", p["projection_target"])),
                           name_core=p["name_core"], ei=ei[name])
            self.ids[("cell", name)] = cid
            for l in p["layers"]:
                self.ins("cell_type_layer_token", cell_type_id=cid, layer_id=self.id("layer", "entorhinal", l),
                         weight=1.0 / len(p["layers"]), evidence_id=ev)
        return edges, ei

    # ------------------------------------------------------------ taxonomy
    def taxonomy(self):
        for r in read_csv("taxon.csv"):
            src = "Pelkey2017" if r["locator"].startswith("p.") else "curator"
            cls = "review_summary" if src == "Pelkey2017" else "curator_inference"
            ev = self.evidence(src, cls, locator=r["locator"])
            self.ids[("taxon", r["code"])] = self.ins(
                "taxon", code=r["code"], name=r["name"], rank=r["rank"],
                parent_taxon_id=self.ids.get(("taxon", r["parent"])), definition=r["definition"], evidence_id=ev)
        for r in read_csv("cell_type_taxon.csv"):
            ev = self.evidence("curator", "curator_inference", quote=r["rationale"])
            self.ins("cell_type_taxon", cell_type_id=self.id("cell", r["netlist_name"]),
                     taxon_id=self.id("taxon", r["taxon"]), is_primary=int(r["is_primary"]),
                     confidence=float(r["confidence"]), evidence_id=ev, rationale=r["rationale"])

    # -------------------------------------------------------------- claims
    def subjects(self, spec):
        out = []
        for s in spec.split(";"):
            kind, key = s.split(":", 1)
            out.append(("taxon_id", self.id("taxon", key)) if kind == "t" else ("cell_type_id", self.id("cell", key)))
        return out

    def claims(self):
        for r in read_csv("property.csv"):
            self.ids[("property", r["code"])] = self.ins(
                "property", code=r["code"], name=r["name"], category=r["category"], value_type=r["value_type"],
                unit=nz(r["unit"]), allowed_values=nz(r["allowed_values"]), description=nz(r["description"]))
        for r in read_csv("condition.csv"):
            self.ids[("condition", r["code"])] = self.ins(
                "condition", code=r["code"], kind=r["kind"], description=nz(r["description"]),
                modulator_id=self.ids.get(("modulator", r["modulator"])) if r["modulator"] else None)
        for r in read_csv("context.csv"):
            self.ids[("context", r["code"])] = self.ins(
                "context", species=nz(r["species"]), strain_or_line=nz(r["strain_or_line"]), age=nz(r["age"]),
                preparation=nz(r["preparation"]), temperature_c=nz(r["temperature_c"], float),
                region_studied=nz(r["region_studied"]), notes=nz(r["notes"]))

        bins = {}
        for b in read_csv("pelkey2017_distribution_bins.csv"):
            bins.setdefault(b["key"], []).append(b)
        used_bins = set()
        for r in read_csv("pelkey2017_claims.csv"):
            ev = self.evidence("Pelkey2017", r["evidence_class"], locator=r["locator"],
                               cited_ref=nz(r["cited_ref"]), quote=nz(r["quote"]))
            for col, sid in self.subjects(r["subject"]):
                cid = self.ins(
                    "property_claim", **{col: sid}, property_id=self.id("property", r["property"]),
                    marker_id=self.ids.get(("marker", r["marker"])) if r["marker"] else None,
                    receptor_id=self.ids.get(("receptor", r["receptor"])) if r["receptor"] else None,
                    mechanism=nz(r["mechanism"]), value_num=nz(r["value_num"], float),
                    condition_id=self.id("condition", r["condition"]) if r.get("condition") else None,
                    value_sd=nz(r["value_sd"], float), value_lo=nz(r["value_lo"], float),
                    value_hi=nz(r["value_hi"], float), n=nz(r["n"], int), stat=nz(r["stat"]),
                    value_cat=nz(r["value_cat"]), value_text=nz(r["value_text"]),
                    confidence=float(r["confidence"]), inheritable=int(r["inheritable"]),
                    requires_marker_id=self.ids.get(("marker", r["requires_marker"])) if r["requires_marker"] else None,
                    requires_marker_status=nz(r["requires_status"]), evidence_id=ev,
                    context_id=self.ids.get(("context", r["context"])), curator="seed:Pelkey2017")
                for i, b in enumerate(bins.get(r["key"], [])):
                    used_bins.add(r["key"])
                    self.ins("claim_distribution_bin", claim_id=cid, bin=i,
                             layer_id=self.id("layer", b["scheme"], b["layer"]) if b["layer"] else None,
                             compartment_id=self.id("compartment", b["compartment"]) if b["compartment"] else None,
                             target_taxon_id=self.id("taxon", b["target_taxon"]) if b["target_taxon"] else None,
                             fraction=nz(b["fraction"], float), role=nz(b["role"]))
        orphan = set(bins) - used_bins
        if orphan:
            raise ValueError(f"distribution bins without a claim: {sorted(orphan)}")

        # Table 1 (Tricoire et al. 2011 via Pelkey 2017): one claim per cluster x property x taxon
        clusters = {r["cluster"]: r for r in read_csv("pelkey2017_table1_clusters.csv")}
        for row in read_csv("pelkey2017_table1.csv"):
            for c, meta in clusters.items():
                ev = self.evidence("Pelkey2017", "secondary_citation", locator="Table 1 (p.1646)", cited_ref="1138",
                                   quote=f"Cluster {c} (n={meta['n']}): {meta['description']} [{meta['locator']}]")
                for taxon in meta["taxa"].split(";"):
                    self.ins("property_claim", taxon_id=self.id("taxon", taxon),
                             property_id=self.id("property", row["property"]),
                             value_num=float(row[f"c{c}_mean"]), value_sd=float(row[f"c{c}_sd"]),
                             n=int(meta["n"]), stat="mean", value_text=f"Tricoire cluster {c}",
                             confidence=float(meta["confidence"]), inheritable=1, evidence_id=ev,
                             context_id=self.id("context", "TRICOIRE2011"), curator="seed:Pelkey2017_Table1")

    # ------------------------------------------------------------- netlist
    def netlist(self, netlist_csv: Path, edges, ei):
        nid = self.ins("netlist", name="mij_netlist", source_file=netlist_csv.name,
                       weight_formula="m_ij = w_ij * kappa_j = q_i c_ij g_ij (V_ref - V_rev) tau_ij kappa_j",
                       convention="i = pre (row), j = post (column); m_ij signed (negative for GABAergic pre)",
                       source_id=self.id("source", "mij_netlist"))
        self.ids[("netlist", "mij_netlist")] = nid
        for code, desc, emb, ov in NETLIST_FACTORS:
            self.ins("netlist_weight_factor", netlist_id=nid, factor_code=code, description=desc, embedded=emb,
                     overlaps_param=ov)
        kappa = {}
        for e in edges:
            kappa.setdefault(e["post_neuron"], float(e["kappa_j"]))
        for idx, name in enumerate(sorted(ei)):
            self.ins("netlist_population", netlist_id=nid, cell_type_id=self.id("cell", name), matrix_index=idx,
                     kappa=kappa.get(name))
        for e in edges:
            m = float(e["m_ij"])
            sign = -1 if e["pre_ei"].strip().lower() == "i" else 1
            if m != 0 and (m < 0) != (sign < 0):
                raise ValueError(f"sign of m_ij disagrees with pre_ei for {e['pre_neuron']} -> {e['post_neuron']}")
            self.ins("netlist_edge", netlist_id=nid, pre_id=self.id("cell", e["pre_neuron"]),
                     post_id=self.id("cell", e["post_neuron"]), w_ij=float(e["w_ij"]),
                     kappa_j=float(e["kappa_j"]), m_ij=m, sign=sign)

    # ------------------------------------------------------------ ODE layer
    def ode(self):
        mid = self.ins("ode_model", code="MPR_QIF_adapt_GABAAB_v1",
                       description="Montbrio-Pazo-Roxin QIF mean field per population with linear adaptation and "
                                   "GABA-A / GABA-B synaptic states (project knowledge sect. 4)",
                       state_per_population="r, v, a, s_A, s_B",
                       units_note="time in ms, rates in kHz, v dimensionless with V_scale normalisation")
        self.ids[("model",)] = mid
        for r in read_csv("model_parameter.csv"):
            self.ids[("param", r["code"])] = self.ins(
                "model_parameter", model_id=mid, code=r["code"], symbol=r["symbol"], scope=r["scope"], unit=r["unit"],
                lower_bound=nz(r["lower_bound"], float), upper_bound=nz(r["upper_bound"], float),
                required=int(r["required"]), applies_to=r["applies_to"], description=r["description"])
        for r in read_csv("parameter_derivation.csv"):
            ev = self.evidence("curator", "modeling_assumption", locator="parameter_derivation.csv",
                               quote=r["assumptions"])
            self.ids[("derivation", r["code"])] = self.ins(
                "parameter_derivation", param_id=self.id("param", r["param"]), code=r["code"],
                input_properties=r["input_properties"], expression=r["expression"],
                sd_expression=nz(r["sd_expression"]), output_distribution=r["output_distribution"],
                assumptions=r["assumptions"], evidence_id=ev, priority=int(r["priority"]))
        for r in read_csv("parameter_prior.csv"):
            src = "Pelkey2017" if r["locator"].startswith(("p.", "Table")) else "assumption"
            ev = self.evidence(src, r["evidence_class"], locator=nz(r["locator"]), quote=nz(r["notes"]))
            subj = {}
            if r["subject"] != "global":
                kind, key = r["subject"].split(":", 1)
                subj = {"t": {"taxon_id": self.ids.get(("taxon", key))},
                        "c": {"cell_type_id": self.ids.get(("cell", key))},
                        "r": {"region_id": self.ids.get(("region", key))}}[kind]
            self.ins("parameter_prior", param_id=self.id("param", r["param"]), **subj,
                     receptor_id=self.ids.get(("receptor", r["receptor"])) if r["receptor"] else None,
                     distribution=r["distribution"], p1=float(r["p1"]), p2=nz(r["p2"], float),
                     lower=nz(r["lower"], float), upper=nz(r["upper"], float), confidence=float(r["confidence"]),
                     basis=r["basis"],
                     requires_marker_id=self.ids.get(("marker", r["requires_marker"])) if r["requires_marker"] else None,
                     requires_marker_status=nz(r["requires_status"]), evidence_id=ev, notes=nz(r["notes"]))

    # ------------------------------------------------------------ checks
    def validate(self):
        problems = []
        fk = self.db.execute("PRAGMA foreign_key_check").fetchall()
        if fk:
            problems.append(f"foreign key violations: {fk[:5]}")
        missing = self.db.execute("""
            SELECT netlist_name FROM cell_type ct WHERE ei='I' AND NOT EXISTS
            (SELECT 1 FROM cell_type_taxon x WHERE x.cell_type_id = ct.cell_type_id AND x.is_primary = 1)""").fetchall()
        if missing:
            problems.append(f"interneurons without a primary taxon: {[m[0] for m in missing]}")
        mapped_e = self.db.execute("""
            SELECT ct.netlist_name FROM cell_type ct JOIN cell_type_taxon x USING (cell_type_id)
            WHERE ct.ei = 'E'""").fetchall()
        if mapped_e:
            problems.append(f"excitatory populations mapped to interneuron taxa: {mapped_e}")
        bad_dist = self.db.execute("""
            SELECT pc.claim_id, SUM(b.fraction) s FROM property_claim pc JOIN claim_distribution_bin b USING (claim_id)
            GROUP BY pc.claim_id HAVING s > 1.0001""").fetchall()
        if bad_dist:
            problems.append(f"distribution claims with fractions summing > 1: {bad_dist}")
        if problems:
            raise SystemExit("VALIDATION FAILED\n  " + "\n  ".join(problems))

    def summary(self):
        q = lambda s: self.db.execute(s).fetchone()[0]
        return {
            "cell_types": q("SELECT COUNT(*) FROM cell_type"),
            "interneurons": q("SELECT COUNT(*) FROM cell_type WHERE ei='I'"),
            "taxa": q("SELECT COUNT(*) FROM taxon"),
            "claims": q("SELECT COUNT(*) FROM property_claim"),
            "distribution_bins": q("SELECT COUNT(*) FROM claim_distribution_bin"),
            "edges": q("SELECT COUNT(*) FROM netlist_edge"),
            "review_references": q("SELECT COUNT(*) FROM review_reference"),
            "model_parameters": q("SELECT COUNT(*) FROM model_parameter"),
            "priors": q("SELECT COUNT(*) FROM parameter_prior"),
            "derivations": q("SELECT COUNT(*) FROM parameter_derivation"),
        }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--netlist", type=Path, default=HERE.parent / "mij_netlist.csv")
    ap.add_argument("--out", type=Path, default=HERE / "hippocampal_interneuron_kb.sqlite")
    a = ap.parse_args(argv)
    b = Builder(a.out)
    b.vocab()
    edges, ei = b.cell_types(a.netlist)
    b.taxonomy()
    b.claims()
    b.netlist(a.netlist, edges, ei)
    b.ode()
    b.validate()
    b.db.commit()
    for k, v in b.summary().items():
        print(f"{k:>18}: {v}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
