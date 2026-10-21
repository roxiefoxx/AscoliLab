"""Hierarchical parameter-prior resolver for the hippocampal interneuron KB.

    from resolver import KB
    kb = KB("hippocampal_interneuron_kb.sqlite")
    kb.resolve("CA1 O LM", "tau")              # one Prior with provenance
    kb.bundle("CA1 Basket CCK+")               # Theta_i : every ODE parameter for one population
    kb.parameter_table()                       # long DataFrame, all 53 interneurons
    kb.coverage()                              # which level of evidence each parameter came from
    ode = kb.ode_inputs()                      # arrays in netlist order + W0 for the MPR model

Resolution order for (cell, parameter):
    rank 0   claims / priors attached to the exact netlist population
    rank 1+d claims / priors on the population's taxon and its ancestors (d = depth above the mapped taxon)
    rank 50  child mixture: the population's family is unresolved (e.g. bare 'Basket'), so the prior is the
             moment-matched mixture of its subtypes' priors -- never used for marker-conditional parameters
    rank 60  region-level priors
    rank 99  global modeling defaults
Within a rank, literature-derived candidates beat modeling assumptions, then derivation priority, then confidence.
The chosen prior is widened by  var = var_min + (1 - c_eff) var_unc,  c_eff = confidence x mapping confidence
x 0.9^(depth) (project knowledge sect. 21).
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None
try:
    from scipy import stats as _st
except ImportError:  # pragma: no cover
    _st = None

GLOBAL_RANK, REGION_RANK, MIXTURE_RANK, SIBLING_RANK = 99, 60, 50, 55
MARKER_MIN_SUPPORT = 0.5   # claim confidence x mapping confidence needed before a marker counts as known
SIGMA_UNC_LOG = 0.5        # extra log-sd added at zero confidence (lognormal)
REL_UNC = 0.5              # extra relative sd added at zero confidence (normal family)
DEPTH_DECAY = 0.9
BASIS_ORDER = {"fitted": 0, "literature_value": 1, "derived_from_claims": 1, "child_mixture": 2,
               "modeling_assumption": 3}
SIGN_PRIORS = {  # receptor_effect -> (dist, p1, p2, lower, upper)
    "increase": ("truncnormal", 1.0, 0.5, 0.0, None),
    "decrease": ("truncnormal", -1.0, 0.5, None, 0.0),
    "biphasic": ("normal", 0.0, 1.0, None, None),
    "none": ("normal", 0.0, 0.1, None, None),
}
MARKER_MAP = {"positive": "positive", "low": "positive", "negative": "negative",
              "mixed": "unknown", "subset": "unknown", "unknown": "unknown"}
SAFE = {"math": math, "sqrt": math.sqrt, "abs": abs, "min": min, "max": max, "pi": math.pi, "exp": math.exp,
        "log": math.log}


# ----------------------------------------------------------------------------
@dataclass
class Prior:
    param: str
    dist: str
    p1: float
    p2: float | None = None
    lower: float | None = None
    upper: float | None = None
    confidence: float = 0.0
    rank: int = GLOBAL_RANK
    specificity: str = "global"
    basis: str = "modeling_assumption"
    source: str = ""
    receptor: str | None = None
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    # -- summary statistics ---------------------------------------------------
    def _frozen(self):
        if _st is None:
            return None
        d, a, b = self.dist, self.p1, self.p2
        if d == "beta":
            k = max(b, 1e-6)
            return _st.beta(max(a * k, 1e-3), max((1 - a) * k, 1e-3))
        if d == "truncnormal":
            lo = -np.inf if self.lower is None else (self.lower - a) / b
            hi = np.inf if self.upper is None else (self.upper - a) / b
            return _st.truncnorm(lo, hi, loc=a, scale=b)
        return None

    def median(self):
        f = self._frozen()
        if f is not None:
            return float(f.median())
        d, a, b = self.dist, self.p1, self.p2
        if d in ("fixed", "lognormal"):
            return a
        if d == "normal":
            return a
        if d in ("truncnormal", "beta"):          # no scipy: Monte Carlo
            return float(np.median(self.sample(200_000, np.random.default_rng(0))))
        if d == "halfnormal":
            return 0.6745 * a
        if d == "uniform":
            return 0.5 * (a + b)
        raise ValueError(d)

    def mean(self):
        f = self._frozen()
        if f is not None:
            return float(f.mean())
        d, a, b = self.dist, self.p1, self.p2
        if d == "fixed":
            return a
        if d == "lognormal":
            return a * math.exp(0.5 * b * b)
        if d == "normal":
            return a
        if d == "halfnormal":
            return a * math.sqrt(2 / math.pi)
        if d == "uniform":
            return 0.5 * (a + b)
        return float(np.mean(self.sample(200_000, np.random.default_rng(0))))

    def sd(self):
        f = self._frozen()
        if f is not None:
            return float(f.std())
        d, a, b = self.dist, self.p1, self.p2
        if d == "fixed":
            return 0.0
        if d == "lognormal":
            return a * math.exp(0.5 * b * b) * math.sqrt(math.exp(b * b) - 1)
        if d == "normal":
            return b
        if d == "halfnormal":
            return a * math.sqrt(1 - 2 / math.pi)
        if d == "uniform":
            return (b - a) / math.sqrt(12)
        return float(np.std(self.sample(200_000, np.random.default_rng(0))))

    def sample(self, n, rng=None):
        rng = rng or np.random.default_rng()
        d, a, b = self.dist, self.p1, self.p2
        if d == "fixed":
            x = np.full(n, a, float)
        elif d == "lognormal":
            x = a * np.exp(b * rng.standard_normal(n))
        elif d == "normal":
            x = a + b * rng.standard_normal(n)
        elif d == "truncnormal":
            lo = -np.inf if self.lower is None else self.lower
            hi = np.inf if self.upper is None else self.upper
            x = np.empty(0)
            while x.size < n:
                y = a + b * rng.standard_normal(4 * n)
                x = np.concatenate([x, y[(y >= lo) & (y <= hi)]])
            x = x[:n]
        elif d == "beta":
            k = max(b, 1e-6)
            x = rng.beta(max(a * k, 1e-3), max((1 - a) * k, 1e-3), n)
        elif d == "halfnormal":
            x = np.abs(a * rng.standard_normal(n))
        elif d == "uniform":
            x = rng.uniform(a, b, n)
        else:
            raise ValueError(d)
        if d not in ("truncnormal",):
            if self.lower is not None:
                x = np.maximum(x, self.lower)
            if self.upper is not None:
                x = np.minimum(x, self.upper)
        return x

    def row(self):
        r = asdict(self)
        r["median"], r["mean"], r["sd"] = self.median(), self.mean(), self.sd()
        r["evidence"] = "; ".join(f"{e.get('subject','')}:{e.get('locator','')}" for e in self.evidence)
        r["warnings"] = " | ".join(self.warnings)
        return r


# ----------------------------------------------------------------------------
class KB:
    def __init__(self, path, netlist="mij_netlist"):
        self.path = Path(path)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.netlist = netlist
        self.model_id = self.db.execute("SELECT model_id FROM ode_model ORDER BY model_id LIMIT 1").fetchone()[0]
        self.params = {r["code"]: dict(r) for r in self.db.execute(
            "SELECT * FROM model_parameter WHERE model_id = ?", (self.model_id,))}
        self.globals = {r["code"]: r["p1"] for r in self.db.execute("""
            SELECT mp.code, pp.p1 FROM parameter_prior pp JOIN model_parameter mp USING (param_id)
            WHERE mp.scope = 'global' AND pp.distribution = 'fixed'
              AND pp.cell_type_id IS NULL AND pp.taxon_id IS NULL AND pp.region_id IS NULL""")}
        self._nl_warn = self._netlist_warnings()

    # ------------------------------------------------------------ lookups
    @lru_cache(maxsize=None)
    def cell_id(self, name):
        r = self.db.execute("SELECT cell_type_id FROM cell_type WHERE netlist_name = ?", (name,)).fetchone()
        if r is None:
            raise KeyError(f"no netlist population named {name!r}")
        return r[0]

    @lru_cache(maxsize=None)
    def cell(self, name):
        return dict(self.db.execute("""
            SELECT ct.*, r.code AS region, r.scheme, pr.code AS parent_region
            FROM cell_type ct JOIN region r USING (region_id)
            LEFT JOIN region pr ON pr.region_id = r.parent_region_id
            WHERE ct.netlist_name = ?""", (name,)).fetchone())

    @lru_cache(maxsize=None)
    def taxon_code(self, tid):
        return self.db.execute("SELECT code FROM taxon WHERE taxon_id = ?", (tid,)).fetchone()[0]

    @lru_cache(maxsize=None)
    def _ancestors(self, tid):
        out, d = [], 0
        while tid is not None:
            out.append((tid, d))
            tid = self.db.execute("SELECT parent_taxon_id FROM taxon WHERE taxon_id = ?", (tid,)).fetchone()[0]
            d += 1
        return tuple(out)

    @lru_cache(maxsize=None)
    def chain(self, name):
        """[(kind, id, rank, depth_above_mapped, mapping_conf)] from most to least specific."""
        cid = self.cell_id(name)
        out = [("cell", cid, 0, 0, 1.0)]
        best = {}
        for m in self.db.execute("SELECT taxon_id, confidence, is_primary FROM cell_type_taxon WHERE cell_type_id=?",
                                 (cid,)):
            for anc, d in self._ancestors(m["taxon_id"]):
                key = anc
                cand = (1 + d + (0 if m["is_primary"] else 1), d, m["confidence"])
                if key not in best or (cand[0], -cand[2]) < (best[key][0], -best[key][2]):
                    best[key] = cand
        out += sorted((("taxon", t, r, d, c) for t, (r, d, c) in best.items()), key=lambda x: (x[2], -x[4]))
        return tuple(out)

    @lru_cache(maxsize=None)
    def taxon_chain(self, tid, mapping_conf=1.0, base_rank=1):
        return tuple(("taxon", t, base_rank + d, d, mapping_conf) for t, d in self._ancestors(tid))

    def primary_taxon(self, name):
        r = self.db.execute("""SELECT t.taxon_id, t.code, x.confidence FROM cell_type_taxon x JOIN taxon t USING (taxon_id)
                               WHERE x.cell_type_id = ? AND x.is_primary = 1""", (self.cell_id(name),)).fetchone()
        return dict(r) if r else None

    def interneurons(self):
        return [r[0] for r in self.db.execute("""
            SELECT ct.netlist_name FROM netlist_population np JOIN cell_type ct USING (cell_type_id)
            JOIN netlist n USING (netlist_id) WHERE n.name = ? AND ct.ei = 'I' ORDER BY np.matrix_index""",
                                              (self.netlist,))]

    def populations(self):
        return [dict(r) for r in self.db.execute("""
            SELECT ct.netlist_name, ct.ei, np.matrix_index, np.kappa FROM netlist_population np
            JOIN cell_type ct USING (cell_type_id) JOIN netlist n USING (netlist_id)
            WHERE n.name = ? ORDER BY np.matrix_index""", (self.netlist,))]

    # ------------------------------------------------------------ claims
    def claims(self, name, prop, chain=None, marker=None, receptor=None, mechanism=None, check_markers=True):
        """All live claims for `prop` reachable from the population, most specific first."""
        chain = chain or self.chain(name)
        q = """SELECT pc.*, e.locator, e.evidence_class, s.short_ref, m.symbol AS req_marker
               FROM property_claim pc JOIN property p USING (property_id)
               JOIN evidence e USING (evidence_id) JOIN source s USING (source_id)
               LEFT JOIN marker m ON m.marker_id = pc.requires_marker_id
               WHERE p.code = ? AND pc.superseded_by IS NULL AND {subj} = ?"""
        extra, args = "", []
        if marker:
            extra += " AND pc.marker_id = (SELECT marker_id FROM marker WHERE symbol = ?)"; args.append(marker)
        if receptor:
            extra += " AND pc.receptor_id = (SELECT receptor_id FROM receptor WHERE symbol = ?)"; args.append(receptor)
        if mechanism:
            extra += " AND pc.mechanism = ?"; args.append(mechanism)
        out = []
        for kind, sid, rank, depth, mconf in chain:
            col = "pc.cell_type_id" if kind == "cell" else "pc.taxon_id"
            for r in self.db.execute(q.format(subj=col) + extra, [prop, sid] + args):
                r = dict(r)
                if kind == "taxon" and depth > 0 and not r["inheritable"]:
                    continue
                if check_markers and r["req_marker"]:
                    if self.marker_status(name, r["req_marker"], chain=chain)[0] != r["requires_marker_status"]:
                        continue
                r.update(rank=rank, depth=depth, mapping_conf=mconf,
                         subject=self.cell_name(sid) if kind == "cell" else self.taxon_code(sid))
                out.append(r)
        out.sort(key=lambda r: (r["rank"], -r["confidence"] * r["mapping_conf"]))
        return out

    @lru_cache(maxsize=None)
    def cell_name(self, cid):
        return self.db.execute("SELECT netlist_name FROM cell_type WHERE cell_type_id=?", (cid,)).fetchone()[0]

    def marker_status(self, name, marker, chain=None):
        """Tri-state marker status ('positive'/'negative'/'unknown') and the claim behind it."""
        cs = self.claims(name, "marker_status", chain=chain, marker=marker, check_markers=False)
        if not cs:
            # receptor-as-marker: a CB1 / 5HT3A receptor_expression claim also counts
            cs = [c for c in self.claims(name, "receptor_expression", chain=chain, receptor=marker,
                                         check_markers=False)]
        cs = [c for c in cs if c["confidence"] * c["mapping_conf"] >= MARKER_MIN_SUPPORT]
        if not cs:
            return "unknown", None
        best = cs[0]
        return MARKER_MAP.get(best["value_cat"], "unknown"), best

    def best_numeric(self, name, prop, chain=None):
        for c in self.claims(name, prop, chain=chain):
            if c["value_num"] is not None or c["value_lo"] is not None:
                return c
        return None

    # ------------------------------------------------------------ candidates
    def _explicit(self, name, param, receptor, chain, cell):
        pid = self.params[param]["param_id"]
        q = """SELECT pp.*, m.symbol AS req_marker, e.locator FROM parameter_prior pp
               JOIN evidence e USING (evidence_id)
               LEFT JOIN marker m ON m.marker_id = pp.requires_marker_id
               WHERE pp.param_id = ? AND {cond}"""
        rec = "" if receptor is None else " AND pp.receptor_id = (SELECT receptor_id FROM receptor WHERE symbol = ?)"
        rargs = [] if receptor is None else [receptor]
        found = []
        for kind, sid, rank, depth, mconf in chain:
            col = "pp.cell_type_id" if kind == "cell" else "pp.taxon_id"
            for r in self.db.execute(q.format(cond=f"{col} = ?") + rec, [pid, sid] + rargs):
                found.append((dict(r), rank, depth, mconf,
                              self.cell_name(sid) if kind == "cell" else self.taxon_code(sid)))
        if cell is not None:
            for reg in (cell["region"], cell["parent_region"]):
                if reg:
                    for r in self.db.execute(q.format(cond="pp.region_id = (SELECT region_id FROM region WHERE code=?)")
                                             + rec, [pid, reg] + rargs):
                        found.append((dict(r), REGION_RANK, 0, 1.0, f"region:{reg}"))
        for r in self.db.execute(q.format(cond="pp.cell_type_id IS NULL AND pp.taxon_id IS NULL AND pp.region_id IS NULL")
                                 + rec, [pid] + rargs):
            found.append((dict(r), GLOBAL_RANK, 0, 1.0, "global"))
        out = []
        for r, rank, depth, mconf, subj in found:
            if r["req_marker"] and name is not None:
                if self.marker_status(name, r["req_marker"], chain=chain)[0] != r["requires_marker_status"]:
                    continue
            out.append(Prior(param=param, dist=r["distribution"], p1=r["p1"], p2=r["p2"], lower=r["lower"],
                             upper=r["upper"], confidence=r["confidence"] * mconf, rank=rank,
                             specificity="global" if rank == GLOBAL_RANK else subj, basis=r["basis"],
                             source=f"prior:{r['prior_id']}", receptor=receptor,
                             evidence=[{"subject": subj, "locator": r["locator"] or "", "prior_id": r["prior_id"],
                                        "requires_marker": r["req_marker"]}],
                             warnings=[] if not r["req_marker"] else
                             [f"conditional on {r['req_marker']} {r['requires_marker_status']}"]))
        return out

    def _derived(self, name, param, chain):
        pid = self.params[param]["param_id"]
        out = []
        for d in self.db.execute("""SELECT * FROM parameter_derivation WHERE param_id = ? ORDER BY priority""", (pid,)):
            d = dict(d)
            if d["output_distribution"] == "sign_rule":
                continue
            ns, used = dict(SAFE), []
            ns.update(self.globals)
            ok = True
            for prop in [s.strip() for s in d["input_properties"].split(",")]:
                c = self.best_numeric(name, prop, chain=chain)
                if c is None:
                    ok = False
                    break
                ns[prop] = c["value_num"] if c["value_num"] is not None else 0.5 * (c["value_lo"] + c["value_hi"])
                ns[prop + "_sd"] = c["value_sd"] or 0.0
                ns[prop + "_n"] = c["n"] or 1
                ns[prop + "_lo"] = c["value_lo"] if c["value_lo"] is not None else ns[prop]
                ns[prop + "_hi"] = c["value_hi"] if c["value_hi"] is not None else ns[prop]
                used.append(c)
            if not ok:
                continue
            try:
                value = float(eval(d["expression"], {"__builtins__": {}}, ns))
                ns["value"] = value
                sd = float(eval(d["sd_expression"], {"__builtins__": {}}, ns)) if d["sd_expression"] else 0.0
            except (ZeroDivisionError, ValueError, TypeError):
                continue
            dist = d["output_distribution"]
            if dist == "lognormal":
                if value <= 0:
                    continue
                p1, p2, lo, hi = value, math.sqrt(math.log(1 + (sd / value) ** 2)) if sd > 0 else 0.05, None, None
            elif dist == "beta":
                p1 = min(max(value, 1e-3), 1 - 1e-3)
                var = max(sd, 1e-3) ** 2
                p2 = max(p1 * (1 - p1) / var - 1, 1.0)
                lo, hi = 0.0, 1.0
            elif dist == "uniform":
                p1, p2, lo, hi = value - 2 * sd, value + 2 * sd, None, None
            else:
                p1, p2, lo, hi = value, max(sd, 1e-9), None, None
            rank = max(c["rank"] for c in used)
            depth = max(c["depth"] for c in used)
            mconf = min(c["mapping_conf"] for c in used)
            conf = min(c["confidence"] for c in used)
            out.append(Prior(param=param, dist=dist, p1=p1, p2=p2, lower=lo, upper=hi, confidence=conf * mconf,
                             rank=rank, specificity=" + ".join(sorted({c["subject"] for c in used})),
                             basis="derived_from_claims", source=f"derivation:{d['code']}",
                             evidence=[{"subject": c["subject"], "property": p, "claim_id": c["claim_id"],
                                        "locator": c["locator"], "value": c["value_num"]}
                                       for p, c in zip([s.strip() for s in d["input_properties"].split(",")], used)],
                             warnings=[]))
            out[-1]._depth = depth
            out[-1]._priority = d["priority"]
        return out

    def _sign_rule(self, name, param, receptor, chain):
        mech = {"alpha_eta": "excitability", "alpha_rel": "release"}[param]
        cs = self.claims(name, "receptor_effect", chain=chain, receptor=receptor, mechanism=mech)
        if not cs:
            return []
        best = cs[0]
        dist, p1, p2, lo, hi = SIGN_PRIORS[best["value_cat"]]
        warn = []
        rivals = {c["value_cat"] for c in cs if c["rank"] == best["rank"]}
        if len(rivals) > 1:
            warn.append(f"conflicting {receptor} effects at the same level: {sorted(rivals)}")
        return [Prior(param=param, dist=dist, p1=p1, p2=p2, lower=lo, upper=hi,
                      confidence=best["confidence"] * best["mapping_conf"], rank=best["rank"],
                      specificity=best["subject"], basis="derived_from_claims", source="derivation:sign_rule",
                      receptor=receptor,
                      evidence=[{"subject": best["subject"], "claim_id": best["claim_id"], "locator": best["locator"],
                                 "value": best["value_cat"]}],
                      warnings=warn + ["sign from literature; magnitude is a modeling assumption"])]

    def _candidates(self, name, param, receptor, chain):
        cell = self.cell(name) if name is not None else None
        cands = self._explicit(name, param, receptor, chain, cell)
        if param in ("alpha_eta", "alpha_rel") and receptor:
            cands += self._sign_rule(name, param, receptor, chain)
        else:
            cands += self._derived(name, param, chain)
        return cands

    @staticmethod
    def _pick(cands):
        def key(p):
            return (p.rank, BASIS_ORDER.get(p.basis, 9), getattr(p, "_priority", 10), -p.confidence)
        return sorted(cands, key=key)[0] if cands else None

    # ------------------------------------------------------------ public API
    def resolve(self, name, param, receptor=None, widen=True, allow_mixture=True):
        if param not in self.params:
            raise KeyError(f"unknown model parameter {param!r}; known: {sorted(self.params)}")
        chain = self.chain(name)
        best = self._pick(self._candidates(name, param, receptor, chain))
        if best is None:
            return None
        if best.rank == GLOBAL_RANK and allow_mixture and receptor is None:
            mix = self._child_mixture(name, param)
            if mix is not None:
                best = mix
        if widen and best.rank < GLOBAL_RANK:
            best = self._widen(best)
        best.warnings = list(best.warnings) + self._nl_warn.get(param, [])
        return best

    def _child_mixture(self, name, param):
        """Unresolved family. (a) bare family with subtypes (e.g. 'Basket'): every subtype must contribute.
        (b) leaf placeholder (e.g. DT_UNRESOLVED, TRILAMINAR): mixture over sibling families that have data."""
        pt = self.primary_taxon(name)
        if pt is None:
            return None
        kids = self.db.execute("SELECT taxon_id, code FROM taxon WHERE parent_taxon_id = ?",
                               (pt["taxon_id"],)).fetchall()
        if len(kids) >= 2:
            comps = self._components(name, param, kids, pt["confidence"], require_all=True)
            if comps:
                return self._mix(param, comps, MIXTURE_RANK, 0.5, pt["code"],
                                 f"{pt['code']} is molecularly unresolved; prior spans its subtypes")
            return None
        if not kids:
            parent = self.db.execute("SELECT parent_taxon_id FROM taxon WHERE taxon_id=?",
                                     (pt["taxon_id"],)).fetchone()[0]
            sibs = [k for k in self.db.execute("SELECT taxon_id, code FROM taxon WHERE parent_taxon_id = ?",
                                               (parent,)).fetchall() if k["taxon_id"] != pt["taxon_id"]]
            comps = self._components(name, param, sibs, pt["confidence"], require_all=False)
            if comps and len(comps) >= 2:
                return self._mix(param, comps, SIBLING_RANK, 0.35, pt["code"],
                                 f"{pt['code']} has no family-level data; prior spans sibling families "
                                 f"under {self.taxon_code(parent)}")
        return None

    def _components(self, name, param, taxa, mapping_conf, require_all):
        comps = []
        for k in taxa:
            ch = self.taxon_chain(k["taxon_id"], mapping_conf, 1)
            c = self._pick([p for p in self._candidates(name, param, None, ch)
                            if p.rank < GLOBAL_RANK and p.basis != "modeling_assumption"
                            and not any(e.get("requires_marker") for e in p.evidence)])
            if c is None:
                if require_all:
                    return None      # every subtype must contribute, otherwise stay broad
                continue
            comps.append((k["code"], c))
        return comps

    def _mix(self, param, comps, rank, conf_factor, parent_code, warning):
        dists = {c.dist for _, c in comps}
        if dists == {"lognormal"}:
            mus = np.array([math.log(c.p1) for _, c in comps]); s2 = np.array([c.p2 ** 2 for _, c in comps])
            mu = mus.mean(); var = (s2 + mus ** 2).mean() - mu ** 2
            p1, p2, dist = math.exp(mu), math.sqrt(var), "lognormal"
        elif dists <= {"normal", "truncnormal"}:
            m = np.array([c.p1 for _, c in comps]); s2 = np.array([c.p2 ** 2 for _, c in comps])
            mu = m.mean(); var = (s2 + m ** 2).mean() - mu ** 2
            p1, p2, dist = mu, math.sqrt(var), "normal"
        elif dists == {"beta"}:
            p1, p2, dist = float(np.mean([c.p1 for _, c in comps])), min(c.p2 for _, c in comps) / 2, "beta"
        elif dists == {"halfnormal"}:
            p1, p2, dist = float(np.sqrt(np.mean([c.p1 ** 2 for _, c in comps]))), None, "halfnormal"
        else:
            return None
        conf = conf_factor * float(np.mean([c.confidence for _, c in comps]))
        kind = "mixture" if rank == MIXTURE_RANK else "sibling_mixture"
        return Prior(param=param, dist=dist, p1=p1, p2=p2, lower=comps[0][1].lower, upper=comps[0][1].upper,
                     confidence=conf, rank=rank, specificity=f"{kind}[{','.join(k for k, _ in comps)}]",
                     basis="child_mixture", source=f"{kind}:{parent_code}",
                     evidence=[e for _, c in comps for e in c.evidence], warnings=[warning])

    @staticmethod
    def _widen(p: Prior):
        depth = max(p.rank - 1, 0) if p.rank < MIXTURE_RANK else 0   # mixtures are already discounted
        c = max(min(p.confidence * DEPTH_DECAY ** depth, 1.0), 0.0)
        q = Prior(**{k: v for k, v in asdict(p).items()})
        if p.dist == "lognormal":
            q.p2 = math.sqrt(p.p2 ** 2 + (1 - c) * SIGMA_UNC_LOG ** 2)
        elif p.dist in ("normal", "truncnormal"):
            q.p2 = math.sqrt(p.p2 ** 2 + (1 - c) * (REL_UNC * max(abs(p.p1), p.p2)) ** 2)
        elif p.dist == "beta":
            q.p2 = max(2.0, p.p2 * (0.25 + 0.75 * c))
        elif p.dist == "halfnormal":
            q.p1 = p.p1 * (1 + (1 - c) * REL_UNC)
        q.confidence = c
        return q

    def receptors_for(self, name):
        chain = self.chain(name)
        recs = set()
        for prop in ("receptor_effect", "receptor_expression"):
            for c in self.claims(name, prop, chain=chain):
                if c["receptor_id"]:
                    recs.add(self.db.execute("SELECT symbol FROM receptor WHERE receptor_id=?",
                                             (c["receptor_id"],)).fetchone()[0])
        return sorted(recs)

    def bundle(self, name, include_optional=True):
        """Theta_i: {param or (param, receptor): Prior} for one population."""
        out = {}
        for code, p in self.params.items():
            if p["scope"] == "global" or p["applies_to"] == "E":
                continue
            if not include_optional and not p["required"]:
                continue
            if p["scope"] == "population":
                r = self.resolve(name, code)
                if r is not None:
                    out[code] = r
            elif p["scope"] == "population_receptor":
                recs = ["CB1"] if code == "beta_CB1" else [
                    r for r in self.receptors_for(name)
                    if not (code == "alpha_rel" and r == "CB1")      # CB1 on release is carried by beta_CB1
                    and self.claims(name, "receptor_effect", receptor=r,
                                   mechanism={"alpha_eta": "excitability", "alpha_rel": "release"}[code])]
                for rec in recs:
                    r = self.resolve(name, code, receptor=rec)
                    if r is not None:
                        out[(code, rec)] = r
        return out

    # ------------------------------------------------------------ morphology
    def layer_profile(self, name, which="axon"):
        """Normalised laminar / compartment vector from the most specific applicable distribution claim.
        which: 'soma' | 'dendrite' | 'axon' | 'compartment' | 'target'"""
        prop = {"soma": "soma_layer_distribution", "dendrite": "dendrite_layer_distribution",
                "axon": "axon_layer_distribution", "compartment": "axon_compartment_distribution",
                "target": "output_target_class_distribution"}[which]
        scheme = self.cell(name)["scheme"]
        for c in self.claims(name, prop):
            bins = self.db.execute("""
                SELECT b.*, l.code AS layer, l.scheme, cp.code AS comp, t.code AS tgt
                FROM claim_distribution_bin b LEFT JOIN layer l USING (layer_id)
                LEFT JOIN compartment cp USING (compartment_id) LEFT JOIN taxon t ON t.taxon_id = b.target_taxon_id
                WHERE b.claim_id = ? ORDER BY b.bin""", (c["claim_id"],)).fetchall()
            if which in ("soma", "dendrite", "axon"):
                home = [b for b in bins if b["scheme"] == scheme]
                if not home:
                    continue                      # CA1 strata do not apply to a DG / EC population
                bins = home                       # off-scheme bins (e.g. axon crossing the fissure) are dropped
            key = lambda b: b["layer"] or b["comp"] or b["tgt"]
            fixed = {key(b): b["fraction"] for b in bins if b["fraction"] is not None}
            roles = {key(b): self.globals.get(f"role_w_{b['role']}", 0.0) for b in bins if b["fraction"] is None}
            resid = max(1.0 - sum(fixed.values()), 0.0)
            vec = dict(fixed)
            if roles and sum(roles.values()) > 0:
                tot = sum(roles.values())
                mass = resid if fixed else 1.0
                vec.update({k: mass * w / tot for k, w in roles.items()})
            elif resid > 1e-9:
                vec["unassigned"] = resid
            s = sum(vec.values())
            vec = {k: v / s for k, v in vec.items()} if s > 0 else vec
            return {"profile": vec, "claim_id": c["claim_id"], "subject": c["subject"], "rank": c["rank"],
                    "locator": c["locator"], "role_converted": bool(roles), "text": c["value_text"]}
        return None

    # ------------------------------------------------------------ tables
    def parameter_table(self, names=None, include_optional=True):
        rows = []
        for n in names or self.interneurons():
            pt = self.primary_taxon(n) or {}
            for k, p in self.bundle(n, include_optional).items():
                r = p.row()
                r.update(cell=n, taxon=pt.get("code"), param=k if isinstance(k, str) else k[0],
                         receptor=None if isinstance(k, str) else k[1])
                rows.append(r)
        cols = ["cell", "taxon", "param", "receptor", "dist", "p1", "p2", "lower", "upper", "median", "mean", "sd",
                "confidence", "rank", "specificity", "basis", "source", "evidence", "warnings"]
        return pd.DataFrame(rows)[cols] if pd else rows

    def coverage(self, names=None):
        """Cell x required-parameter grid of evidence level: cell / taxon / mixture / region / global."""
        names = names or self.interneurons()
        req = [c for c, p in self.params.items() if p["scope"] == "population" and p["required"]]
        def lvl(p):
            if p is None:
                return "missing"
            if p.rank == 0:
                return "cell"
            if p.rank < MIXTURE_RANK:
                return f"taxon(d{p.rank - 1})"
            return {MIXTURE_RANK: "subtype_mixture", SIBLING_RANK: "sibling_mixture",
                    REGION_RANK: "region"}.get(p.rank, "global")
        grid = {n: {c: lvl(self.resolve(n, c)) for c in req} for n in names}
        return pd.DataFrame(grid).T if pd else grid

    def sample(self, n_samples=100, names=None, params=None, seed=0):
        """Joint (independent) prior draws: {param: array (n_samples, n_cells)}."""
        rng = np.random.default_rng(seed)
        names = names or self.interneurons()
        params = params or [c for c, p in self.params.items() if p["scope"] == "population" and p["required"]]
        out = {}
        for c in params:
            cols = []
            for nme in names:
                pr = self.resolve(nme, c)
                cols.append(pr.sample(n_samples, rng) if pr else np.full(n_samples, np.nan))
            out[c] = np.stack(cols, axis=1)
        return out

    # ------------------------------------------------------------ ODE handoff
    def connectivity(self, weight="m_ij"):
        """Signed coupling in state convention M[post, pre] (rows = target, columns = source)."""
        pops = self.populations()
        idx = {p["netlist_name"]: p["matrix_index"] for p in pops}
        M = np.zeros((len(pops), len(pops)))
        col = {"m_ij": "e.m_ij", "w_ij": "e.w_ij"}[weight]          # both are stored signed
        for r in self.db.execute(f"""
            SELECT a.netlist_name AS pre, b.netlist_name AS post, {col} AS w FROM netlist_edge e
            JOIN netlist n USING (netlist_id) JOIN cell_type a ON a.cell_type_id = e.pre_id
            JOIN cell_type b ON b.cell_type_id = e.post_id WHERE n.name = ?""", (self.netlist,)):
            M[idx[r["post"]], idx[r["pre"]]] = r["w"]
        return [p["netlist_name"] for p in pops], M

    def ode_inputs(self, point="median", weight="m_ij", include_optional=False):
        """Everything the MPR notebook needs, in netlist matrix order.
        Returns dict(names, is_inhibitory, M_raw, W0, rho, params{code: array}, table, warnings).
        Parameters are filled for interneurons only; excitatory rows are NaN."""
        names, M = self.connectivity(weight)
        rho = float(np.max(np.abs(np.linalg.eigvals(np.abs(M)))))
        pops = {p["netlist_name"]: p for p in self.populations()}
        inh = np.array([pops[n]["ei"] == "I" for n in names])
        table = self.parameter_table([n for n in names if pops[n]["ei"] == "I"], include_optional)
        params = {}
        pick = {"median": "median", "mean": "mean"}[point]
        for code in sorted({p for p in table["param"]} if pd else []):
            sub = table[(table["param"] == code) & table["receptor"].isna()]
            if sub.empty:
                continue
            arr = np.full(len(names), np.nan)
            for _, r in sub.iterrows():
                arr[names.index(r["cell"])] = r[pick]
            params[code] = arr
        warns = sorted({w for ws in self._nl_warn.values() for w in ws})
        return {"names": names, "is_inhibitory": inh, "M_raw": M, "W0": M / rho, "rho": rho,
                "params": params, "globals": dict(self.globals), "table": table, "warnings": warns}

    def _netlist_warnings(self):
        out = {}
        for r in self.db.execute("""SELECT f.* FROM netlist_weight_factor f JOIN netlist n USING (netlist_id)
                                    WHERE n.name = ? AND f.overlaps_param IS NOT NULL""", (self.netlist,)):
            for p in r["overlaps_param"].split(","):
                if r["embedded"]:
                    msg = {
                        "tau_ij": "m_ij already integrates synaptic decay (charge). Use tau_A for kinetics only: with "
                                  "tau_A s' = -s + U r the steady-state gain is tau-independent; s' = -s/tau_A + U r "
                                  "would count tau twice.",
                        "kappa_j": "m_ij includes the target's Izhikevich transfer slope kappa_j while the MPR mass "
                                   "supplies its own transfer function; use weight='w_ij' (kappa removed) or fix the "
                                   "MPR gain, not both.",
                        "c_ij": "m_ij already contains axonal-dendritic overlap c_ij; do not multiply Omega_morph "
                                "(a_layer x d_layer) into J again.",
                    }.get(r["factor_code"], f"{r['factor_code']} is embedded in m_ij")
                else:
                    msg = f"{r['factor_code']} is NOT in m_ij; {p} is free to carry it."
                out.setdefault(p.strip(), []).append(msg)
        return out

    # ------------------------------------------------------------ curation helpers
    def add_claim(self, subject, prop, *, source, evidence_class, locator=None, quote=None, confidence=0.5,
                  context=None, **values):
        """subject='t:OLM' or 'c:CA1 O LM'. values: value_num, value_sd, n, stat, value_cat, marker, receptor, ..."""
        kind, key = subject.split(":", 1)
        subj = ({"taxon_id": self.db.execute("SELECT taxon_id FROM taxon WHERE code=?", (key,)).fetchone()[0]}
                if kind == "t" else {"cell_type_id": self.cell_id(key)})
        sid = self.db.execute("SELECT source_id FROM source WHERE short_ref=?", (source,)).fetchone()
        if sid is None:
            raise KeyError(f"add the source {source!r} first (INSERT INTO source ...)")
        ev = self.db.execute("INSERT INTO evidence (source_id, evidence_class, locator, quote) VALUES (?,?,?,?)",
                             (sid[0], evidence_class, locator, quote)).lastrowid
        for k, tbl, col in (("marker", "marker", "symbol"), ("receptor", "receptor", "symbol")):
            if k in values:
                values[f"{k}_id"] = self.db.execute(f"SELECT {k}_id FROM {tbl} WHERE {col}=?",
                                                    (values.pop(k),)).fetchone()[0]
        cols = dict(subj, property_id=self.db.execute("SELECT property_id FROM property WHERE code=?",
                                                       (prop,)).fetchone()[0],
                    evidence_id=ev, confidence=confidence, **values)
        if context:
            cols["context_id"] = context
        cid = self.db.execute(f"INSERT INTO property_claim ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                              list(cols.values())).lastrowid
        self.db.commit()
        self._clear()
        return cid

    def record_estimates(self, description, estimates, observables=None, code_ref=None, method="MAP"):
        """Store fitted values as model_estimate rows (never as claims). estimates: [(cell, param, value, sd)]"""
        run = self.db.execute("INSERT INTO fit_run (model_id, description, observables, code_ref) VALUES (?,?,?,?)",
                              (self.model_id, description, observables, code_ref)).lastrowid
        for cell, param, value, sd in estimates:
            self.db.execute("""INSERT INTO model_estimate (run_id, param_id, cell_type_id, value, sd, method)
                               VALUES (?,?,?,?,?,?)""",
                            (run, self.params[param]["param_id"], self.cell_id(cell), value, sd, method))
        self.db.commit()
        return run

    def _clear(self):
        for f in (self.chain, self.taxon_chain):
            f.cache_clear()


if __name__ == "__main__":  # quick smoke test
    import sys
    kb = KB(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("hippocampal_interneuron_kb.sqlite"))
    for cell in ("CA1 O LM", "CA1 Basket", "CA1 Basket CCK+", "MEC LIII Superficial Multipolar Interneuron"):
        print(f"\n== {cell}  [{(kb.primary_taxon(cell) or {}).get('code')}]")
        for k, p in kb.bundle(cell, include_optional=False).items():
            print(f"  {str(k):<22} {p.dist:<11} median={p.median():9.4g}  conf={p.confidence:.2f}  "
                  f"from {p.specificity} ({p.basis})")
