-- =====================================================================
-- Hippocampal interneuron knowledge base  ->  neural-mass ODE priors
-- SQLite >= 3.37   (PRAGMA foreign_keys = ON is required at connect time)
--
-- Design rules (from hippocampal_interneuron_neural_mass_project_knowledge.md):
--   * Exact netlist populations are the primary key space (cell_type).
--   * Labels are lookup keys, not parameters: every number lives in a
--     property_claim (what a source says) or a parameter_prior (what the
--     model will use), never in cell_type.
--   * Markers are tri-state (positive / negative / unknown); absence of a
--     token in a name is never evidence of negativity.
--   * Measured facts, curator inferences, modeling assumptions and fitted
--     estimates are distinct evidence classes and never share a table row.
--   * Less annotation -> broader prior, not an invented subtype.
--   * Factors already embedded in the netlist weight m_ij are declared in
--     netlist_weight_factor so the resolver can flag double counting.
-- =====================================================================

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------
-- 1. Controlled vocabularies
-- ---------------------------------------------------------------------

CREATE TABLE region (
    region_id        INTEGER PRIMARY KEY,
    code             TEXT NOT NULL UNIQUE,          -- 'CA1', 'CA3c', 'MEC'
    name             TEXT NOT NULL,
    parent_region_id INTEGER REFERENCES region(region_id),  -- CA3c -> CA3, MEC -> EC
    scheme           TEXT NOT NULL CHECK (scheme IN ('hippocampal_CA','dentate','subiculum','entorhinal'))
);

-- Laminae. Hippocampal strata and entorhinal layers live in one table,
-- partitioned by scheme so that d_il and a_ilc have a well-defined layer axis.
CREATE TABLE layer (
    layer_id   INTEGER PRIMARY KEY,
    scheme     TEXT NOT NULL CHECK (scheme IN ('hippocampal_CA','dentate','subiculum','entorhinal')),
    code       TEXT NOT NULL,                       -- 'SO','SP','SR','SL','SLM','ALV' | 'OML','MML','IML','SG','H' | 'LI'..'LVI'
    name       TEXT NOT NULL,
    ordinal    INTEGER NOT NULL,                    -- superficial-to-deep (or alveus-to-fissure) order
    UNIQUE (scheme, code)
);

-- Postsynaptic compartments (index c in a_{j,l,c} and Gamma_{i,c}).
CREATE TABLE compartment (
    compartment_id INTEGER PRIMARY KEY,
    code           TEXT NOT NULL UNIQUE,            -- 'AIS','soma','proximal_dendrite','basal_dendrite','oblique_dendrite','apical_shaft','distal_apical','dendrite_unspecified','nonsynaptic_volume'
    name           TEXT NOT NULL,
    ordinal        INTEGER NOT NULL                 -- electrotonic distance rank from the spike initiation zone
);

CREATE TABLE marker (
    marker_id INTEGER PRIMARY KEY,
    symbol    TEXT NOT NULL UNIQUE,                 -- 'PV','SST','CCK','NPY','VIP','CR','CB','nNOS','Reelin','VGluT3','COUPTFII','SATB1','mGluR1a','Elfn1','Syt2','5HT3A','CB1'
    name      TEXT NOT NULL,
    kind      TEXT NOT NULL CHECK (kind IN ('calcium_binding','neuropeptide','receptor','transcription_factor','enzyme','transporter','adhesion','synaptic_protein','other'))
);

CREATE TABLE modulator (
    modulator_id INTEGER PRIMARY KEY,
    symbol       TEXT NOT NULL UNIQUE,              -- 'ACh','5-HT','DA','eCB','opioid','CCK','OXT','GABA_ambient'
    name         TEXT NOT NULL
);

CREATE TABLE receptor (
    receptor_id  INTEGER PRIMARY KEY,
    symbol       TEXT NOT NULL UNIQUE,              -- 'CB1','M1','M2','M3','M4','nAChR_a7','5HT3A','5HT2A','MOR','DOR','KOR','D1','D2','D4','CCK2','OXTR','GABAB'
    modulator_id INTEGER NOT NULL REFERENCES modulator(modulator_id),
    mechanism    TEXT NOT NULL CHECK (mechanism IN ('ionotropic','metabotropic')),
    marker_id    INTEGER REFERENCES marker(marker_id)   -- when the receptor is also used as an identity marker (CB1, 5HT3A)
);

-- Conditions under which a claim holds or a function is expressed
-- (neuromodulatory state, oscillation, activity pattern, behavior).
CREATE TABLE condition (
    condition_id INTEGER PRIMARY KEY,
    code         TEXT NOT NULL UNIQUE,              -- 'ACh','opioid','theta','trains','CFC'
    kind         TEXT NOT NULL CHECK (kind IN ('neuromodulator','oscillation','activity','behavior')),
    modulator_id INTEGER REFERENCES modulator(modulator_id),
    description  TEXT
);

-- Afferent pathways (index k in p_{k,l} and Omega^{in}_{k,i}).
CREATE TABLE pathway (
    pathway_id       INTEGER PRIMARY KEY,
    code             TEXT NOT NULL UNIQUE,          -- 'PP','TA','SC','MF','CA3_rec','CA1_rec','HC_assoc','NRe'
    name             TEXT NOT NULL,
    source_region_id INTEGER REFERENCES region(region_id),
    target_region_id INTEGER REFERENCES region(region_id)
);

CREATE TABLE pathway_termination (
    pathway_id INTEGER NOT NULL REFERENCES pathway(pathway_id),
    layer_id   INTEGER NOT NULL REFERENCES layer(layer_id),
    weight     REAL    NOT NULL CHECK (weight >= 0),   -- p_{k,l}; normalised per pathway by the resolver
    claim_id   INTEGER REFERENCES property_claim(claim_id),
    PRIMARY KEY (pathway_id, layer_id)
);

-- ---------------------------------------------------------------------
-- 2. Provenance
-- ---------------------------------------------------------------------

CREATE TABLE source (
    source_id   INTEGER PRIMARY KEY,
    short_ref   TEXT NOT NULL UNIQUE,               -- 'Pelkey2017'
    citation    TEXT NOT NULL,
    doi         TEXT,
    pmid        TEXT,
    year        INTEGER,
    source_type TEXT NOT NULL CHECK (source_type IN ('review','primary','database','netlist','curator','model_fit','assumption'))
);

-- Bibliography of a review, so evidence.cited_ref numbers resolve to the primary experiments.
CREATE TABLE review_reference (
    source_id    INTEGER NOT NULL REFERENCES source(source_id),
    ref_num      INTEGER NOT NULL,
    first_author TEXT,
    year         INTEGER,
    doi          TEXT,
    citation     TEXT NOT NULL,
    PRIMARY KEY (source_id, ref_num)
);

-- One row per assertion-level provenance: where in the source, and what kind of knowledge it is.
CREATE TABLE evidence (
    evidence_id    INTEGER PRIMARY KEY,
    source_id      INTEGER NOT NULL REFERENCES source(source_id),
    evidence_class TEXT NOT NULL CHECK (evidence_class IN (
                        'measurement',          -- a number measured in the cited work
                        'review_summary',       -- a review's synthesis of primary work
                        'secondary_citation',   -- review reporting another paper's number (cited_ref says which)
                        'name_parse',           -- deterministic parse of the netlist label
                        'curator_inference',    -- a curator's mapping or reading, not stated verbatim
                        'modeling_assumption',  -- a default chosen to make the ODE runnable
                        'model_estimate')),     -- fitted value fed back from inference
    locator        TEXT,                        -- 'p.1624', 'Table 1', 'Fig. 22G'
    cited_ref      TEXT,                        -- reference number(s) inside a review, e.g. '102' (Bezaire & Soltesz 2013)
    quote          TEXT                         -- short supporting excerpt
);

-- Experimental context (species, age, prep). Claims without context are allowed but penalised.
CREATE TABLE context (
    context_id     INTEGER PRIMARY KEY,
    species        TEXT,                        -- 'mouse','rat','rodent'
    strain_or_line TEXT,                        -- 'Nkx2-1Cre:RCE', '5HT3aR-eGFP'
    age            TEXT,                        -- 'P15-P30', 'adult'
    preparation    TEXT CHECK (preparation IN ('acute_slice','in_vivo','culture','anatomy','mixed') OR preparation IS NULL),
    temperature_c  REAL,
    region_studied TEXT,                        -- may differ from the cell_type region the claim is applied to
    notes          TEXT
);

-- ---------------------------------------------------------------------
-- 3. Cell types and taxonomy
-- ---------------------------------------------------------------------

-- Exactly one row per netlist population. Name-derived fields only.
CREATE TABLE cell_type (
    cell_type_id            INTEGER PRIMARY KEY,
    netlist_name            TEXT NOT NULL UNIQUE,   -- exact string from the netlist, e.g. 'CA1 Basket CCK+'
    region_id               INTEGER NOT NULL REFERENCES region(region_id),
    layer_token             TEXT,                   -- raw Roman-numeral token for EC/MEC/LEC names, e.g. 'LI-II'
    projection_target_id    INTEGER REFERENCES region(region_id),   -- 'SUB CA1 Projecting Pyramidal' -> CA1
    name_core               TEXT NOT NULL,          -- label with region / layer tokens removed
    ei                      TEXT NOT NULL CHECK (ei IN ('E','I')),  -- from netlist pre_ei/post_ei
    hippocampome_id         TEXT,                   -- to be filled from Hippocampome.org
    notes                   TEXT
);

-- Compound / weighted layer membership parsed from names (LI-II -> LI 0.5, LII 0.5).
CREATE TABLE cell_type_layer_token (
    cell_type_id INTEGER NOT NULL REFERENCES cell_type(cell_type_id),
    layer_id     INTEGER NOT NULL REFERENCES layer(layer_id),
    weight       REAL NOT NULL CHECK (weight > 0 AND weight <= 1),
    evidence_id  INTEGER NOT NULL REFERENCES evidence(evidence_id),
    PRIMARY KEY (cell_type_id, layer_id)
);

-- Family / subtype hierarchy. Claims and priors attach here and are inherited downward.
CREATE TABLE taxon (
    taxon_id        INTEGER PRIMARY KEY,
    code            TEXT NOT NULL UNIQUE,           -- 'BASKET', 'CCKBC', 'OLM_FAMILY'
    name            TEXT NOT NULL,
    rank            TEXT NOT NULL CHECK (rank IN ('root','class','family','subtype')),
    parent_taxon_id INTEGER REFERENCES taxon(taxon_id),
    definition      TEXT,
    evidence_id     INTEGER REFERENCES evidence(evidence_id)
);

-- Which taxon each netlist population belongs to. Many-to-many with confidence,
-- because several Hippocampome types sit ambiguously between families.
CREATE TABLE cell_type_taxon (
    cell_type_id INTEGER NOT NULL REFERENCES cell_type(cell_type_id),
    taxon_id     INTEGER NOT NULL REFERENCES taxon(taxon_id),
    is_primary   INTEGER NOT NULL DEFAULT 1 CHECK (is_primary IN (0,1)),
    confidence   REAL NOT NULL CHECK (confidence > 0 AND confidence <= 1),
    evidence_id  INTEGER NOT NULL REFERENCES evidence(evidence_id),
    rationale    TEXT,
    PRIMARY KEY (cell_type_id, taxon_id)
);
CREATE UNIQUE INDEX ux_one_primary_taxon ON cell_type_taxon(cell_type_id) WHERE is_primary = 1;

-- ---------------------------------------------------------------------
-- 4. Claims: (subject, property, value, qualifiers, evidence, context, confidence)
-- ---------------------------------------------------------------------

CREATE TABLE property (
    property_id INTEGER PRIMARY KEY,
    code        TEXT NOT NULL UNIQUE,               -- 'tau_m','R_in','V_rest','marker_status','axon_layer_role'
    name        TEXT NOT NULL,
    category    TEXT NOT NULL CHECK (category IN ('census','anatomy','connectivity','intrinsic','synaptic','release','marker','neuromodulation','oscillation','development')),
    value_type  TEXT NOT NULL CHECK (value_type IN ('numeric','categorical','distribution','boolean')),
    unit        TEXT,                               -- canonical unit; numeric claims must be stored in it
    allowed_values TEXT,                            -- '|'-separated for categorical properties
    description TEXT
);

CREATE TABLE property_claim (
    claim_id        INTEGER PRIMARY KEY,
    -- subject: exactly one of cell_type / taxon
    cell_type_id    INTEGER REFERENCES cell_type(cell_type_id),
    taxon_id        INTEGER REFERENCES taxon(taxon_id),
    property_id     INTEGER NOT NULL REFERENCES property(property_id),
    -- optional qualifiers (which marker, receptor, layer, compartment, target ...)
    marker_id       INTEGER REFERENCES marker(marker_id),
    receptor_id     INTEGER REFERENCES receptor(receptor_id),
    layer_id        INTEGER REFERENCES layer(layer_id),
    compartment_id  INTEGER REFERENCES compartment(compartment_id),
    target_taxon_id INTEGER REFERENCES taxon(taxon_id),       -- postsynaptic class, for connectivity claims
    mechanism       TEXT CHECK (mechanism IN ('excitability','release','ih','adaptation','im','postsynaptic','presynaptic','gap_junction') OR mechanism IS NULL),
    condition_id    INTEGER REFERENCES condition(condition_id),              -- claim holds under this condition
    -- value
    value_num       REAL,
    value_sd        REAL,
    value_lo        REAL,
    value_hi        REAL,
    n               INTEGER,
    stat            TEXT CHECK (stat IN ('mean','median','point','range','approx','lower_bound','upper_bound') OR stat IS NULL),
    value_cat       TEXT,                           -- categorical value (checked against property.allowed_values by trigger)
    value_text      TEXT,                           -- free-text detail
    -- weight of evidence
    confidence      REAL NOT NULL DEFAULT 0.5 CHECK (confidence > 0 AND confidence <= 1),
    inheritable     INTEGER NOT NULL DEFAULT 1 CHECK (inheritable IN (0,1)),  -- may descendants of taxon_id use it?
    requires_marker_id     INTEGER REFERENCES marker(marker_id),               -- inherit only if this marker...
    requires_marker_status TEXT CHECK (requires_marker_status IN ('positive','negative') OR requires_marker_status IS NULL), -- ...has this status
    evidence_id     INTEGER NOT NULL REFERENCES evidence(evidence_id),
    context_id      INTEGER REFERENCES context(context_id),
    superseded_by   INTEGER REFERENCES property_claim(claim_id),
    curator         TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    CHECK ((cell_type_id IS NULL) <> (taxon_id IS NULL)),
    CHECK (value_sd IS NULL OR value_sd >= 0),
    CHECK (value_lo IS NULL OR value_hi IS NULL OR value_lo <= value_hi)
);
CREATE INDEX ix_claim_subject_ct    ON property_claim(cell_type_id, property_id);
CREATE INDEX ix_claim_subject_taxon ON property_claim(taxon_id, property_id);

-- Categorical values must be in the property's allowed list.
CREATE TRIGGER trg_claim_cat_check
BEFORE INSERT ON property_claim
WHEN NEW.value_cat IS NOT NULL
 AND (SELECT allowed_values FROM property WHERE property_id = NEW.property_id) IS NOT NULL
 AND instr('|' || (SELECT allowed_values FROM property WHERE property_id = NEW.property_id) || '|',
           '|' || NEW.value_cat || '|') = 0
BEGIN
    SELECT RAISE(ABORT, 'value_cat not in property.allowed_values');
END;

-- Vector-valued claims: laminar / compartment distributions
-- (soma stratum membership, d_{i,l}, a_{j,l,c}, target-class fractions).
-- Fractions need not sum to 1 in the source; the resolver normalises and
-- reports the residual mass as 'unassigned'.
CREATE TABLE claim_distribution_bin (
    claim_id        INTEGER NOT NULL REFERENCES property_claim(claim_id) ON DELETE CASCADE,
    bin             INTEGER NOT NULL,
    layer_id        INTEGER REFERENCES layer(layer_id),
    compartment_id  INTEGER REFERENCES compartment(compartment_id),
    target_taxon_id INTEGER REFERENCES taxon(taxon_id),
    fraction        REAL CHECK (fraction IS NULL OR (fraction >= 0 AND fraction <= 1)),
    role            TEXT CHECK (role IN ('dominant','major','minor','sparse','avoids','present') OR role IS NULL),
    PRIMARY KEY (claim_id, bin),
    CHECK (fraction IS NOT NULL OR role IS NOT NULL)
);

-- ---------------------------------------------------------------------
-- 5. Netlist (the coupling backbone)
-- ---------------------------------------------------------------------

CREATE TABLE netlist (
    netlist_id     INTEGER PRIMARY KEY,
    name           TEXT NOT NULL UNIQUE,
    source_file    TEXT NOT NULL,
    weight_formula TEXT NOT NULL,                   -- 'm_ij = q_i c_ij g_ij (V_ref - V_rev) tau_ij kappa_j'
    convention     TEXT NOT NULL,                   -- 'row = pre (i), column = post (j)'
    source_id      INTEGER REFERENCES source(source_id),
    notes          TEXT
);

-- Which biophysical factors are already folded into the edge weight.
-- The resolver refuses to multiply an ODE parameter into J_ij if it is listed
-- here with embedded = 1, and warns when an ODE time constant duplicates one.
CREATE TABLE netlist_weight_factor (
    netlist_id     INTEGER NOT NULL REFERENCES netlist(netlist_id),
    factor_code    TEXT NOT NULL,                   -- 'q_i','c_ij','g_ij','driving_force','tau_ij','kappa_j','U_stp'
    description    TEXT NOT NULL,
    embedded       INTEGER NOT NULL CHECK (embedded IN (0,1)),
    overlaps_param TEXT,                            -- model_parameter.code it would double-count, e.g. 'tau_A'
    PRIMARY KEY (netlist_id, factor_code)
);

CREATE TABLE netlist_population (
    netlist_id     INTEGER NOT NULL REFERENCES netlist(netlist_id),
    cell_type_id   INTEGER NOT NULL REFERENCES cell_type(cell_type_id),
    matrix_index   INTEGER NOT NULL,                -- row/column index in the exported 85x85 matrix
    kappa          REAL,                            -- kappa_j (Izhikevich transfer slope) when the population is a target
    PRIMARY KEY (netlist_id, cell_type_id),
    UNIQUE (netlist_id, matrix_index)
);

CREATE TABLE netlist_edge (
    netlist_id   INTEGER NOT NULL REFERENCES netlist(netlist_id),
    pre_id       INTEGER NOT NULL REFERENCES cell_type(cell_type_id),
    post_id      INTEGER NOT NULL REFERENCES cell_type(cell_type_id),
    w_ij         REAL NOT NULL,
    kappa_j      REAL NOT NULL,
    m_ij         REAL NOT NULL,
    sign         INTEGER NOT NULL CHECK (sign IN (-1, 1)),   -- -1 for GABAergic presynaptic population
    PRIMARY KEY (netlist_id, pre_id, post_id)
);
CREATE INDEX ix_edge_post ON netlist_edge(netlist_id, post_id);

-- ---------------------------------------------------------------------
-- 6. ODE model layer
-- ---------------------------------------------------------------------

CREATE TABLE ode_model (
    model_id    INTEGER PRIMARY KEY,
    code        TEXT NOT NULL UNIQUE,               -- 'MPR_QIF_adapt_GABAAB_v1'
    description TEXT NOT NULL,
    state_per_population TEXT NOT NULL,             -- 'r, v, a, s_A, s_B'
    units_note  TEXT
);

CREATE TABLE model_parameter (
    param_id    INTEGER PRIMARY KEY,
    model_id    INTEGER NOT NULL REFERENCES ode_model(model_id),
    code        TEXT NOT NULL,                      -- 'tau','Delta','eta','b','tau_a','tau_A','tau_B','U_A','U_B','beta_CB1', ...
    symbol      TEXT NOT NULL,                      -- LaTeX
    scope       TEXT NOT NULL CHECK (scope IN ('population','population_layer','population_layer_compartment','population_receptor','edge','global')),
    unit        TEXT NOT NULL,
    lower_bound REAL,
    upper_bound REAL,
    required    INTEGER NOT NULL DEFAULT 1 CHECK (required IN (0,1)),   -- needed to integrate the ODE
    applies_to  TEXT NOT NULL DEFAULT 'I' CHECK (applies_to IN ('I','E','both')),
    description TEXT,
    UNIQUE (model_id, code)
);

-- Declarative rules that turn claims into parameter values,
-- e.g. tau <- tau_m ;  Delta <- 0.674 * sigma_eta.
CREATE TABLE parameter_derivation (
    derivation_id INTEGER PRIMARY KEY,
    param_id      INTEGER NOT NULL REFERENCES model_parameter(param_id),
    code          TEXT NOT NULL UNIQUE,
    input_properties TEXT NOT NULL,                 -- comma-separated property.code list
    expression    TEXT NOT NULL,                    -- python expression over input codes (+ '<code>_sd')
    sd_expression TEXT,                             -- python expression for the propagated SD
    output_distribution TEXT NOT NULL CHECK (output_distribution IN ('normal','lognormal','truncnormal','beta','halfnormal','uniform','fixed','sign_rule')),
    assumptions   TEXT NOT NULL,
    evidence_id   INTEGER NOT NULL REFERENCES evidence(evidence_id),
    priority      INTEGER NOT NULL DEFAULT 10       -- lower = tried first
);

-- Priors: explicit distributions attached at any specificity level.
-- Exactly one of (cell_type_id, taxon_id, region_id) or none (= global default).
CREATE TABLE parameter_prior (
    prior_id      INTEGER PRIMARY KEY,
    param_id      INTEGER NOT NULL REFERENCES model_parameter(param_id),
    cell_type_id  INTEGER REFERENCES cell_type(cell_type_id),
    taxon_id      INTEGER REFERENCES taxon(taxon_id),
    region_id     INTEGER REFERENCES region(region_id),
    receptor_id   INTEGER REFERENCES receptor(receptor_id),
    layer_id      INTEGER REFERENCES layer(layer_id),
    compartment_id INTEGER REFERENCES compartment(compartment_id),
    distribution  TEXT NOT NULL CHECK (distribution IN ('normal','lognormal','truncnormal','beta','halfnormal','uniform','fixed')),
    p1            REAL NOT NULL,                    -- normal/truncnormal: mean; lognormal: median; beta: mean; halfnormal: scale; uniform: low; fixed: value
    p2            REAL,                             -- normal/truncnormal: sd;   lognormal: sigma (log space); beta: concentration; uniform: high
    lower         REAL,
    upper         REAL,
    confidence    REAL NOT NULL CHECK (confidence > 0 AND confidence <= 1),
    basis         TEXT NOT NULL CHECK (basis IN ('derived_from_claims','literature_value','modeling_assumption','fitted')),
    requires_marker_id     INTEGER REFERENCES marker(marker_id),
    requires_marker_status TEXT CHECK (requires_marker_status IN ('positive','negative') OR requires_marker_status IS NULL),
    derivation_id INTEGER REFERENCES parameter_derivation(derivation_id),
    evidence_id   INTEGER NOT NULL REFERENCES evidence(evidence_id),
    notes         TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    CHECK ((cell_type_id IS NOT NULL) + (taxon_id IS NOT NULL) + (region_id IS NOT NULL) <= 1)
);
CREATE INDEX ix_prior_param ON parameter_prior(param_id);

CREATE TABLE fit_run (
    run_id      INTEGER PRIMARY KEY,
    model_id    INTEGER NOT NULL REFERENCES ode_model(model_id),
    netlist_id  INTEGER REFERENCES netlist(netlist_id),
    description TEXT NOT NULL,
    observables TEXT,                               -- 'theta freq, gamma power, ...'
    code_ref    TEXT,                               -- notebook / commit
    run_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Posterior / fitted values. Kept apart from claims and priors by construction.
CREATE TABLE model_estimate (
    estimate_id  INTEGER PRIMARY KEY,
    run_id       INTEGER NOT NULL REFERENCES fit_run(run_id),
    param_id     INTEGER NOT NULL REFERENCES model_parameter(param_id),
    cell_type_id INTEGER REFERENCES cell_type(cell_type_id),
    receptor_id  INTEGER REFERENCES receptor(receptor_id),
    value        REAL NOT NULL,
    sd           REAL,
    ci_lo        REAL,
    ci_hi        REAL,
    method       TEXT NOT NULL                      -- 'MAP','posterior_mean','continuation_fit'
);

-- ---------------------------------------------------------------------
-- 7. Views
-- ---------------------------------------------------------------------

-- Recursive taxon ancestry with depth (0 = the taxon itself).
CREATE VIEW v_taxon_ancestry AS
WITH RECURSIVE anc(taxon_id, ancestor_id, depth) AS (
    SELECT taxon_id, taxon_id, 0 FROM taxon
    UNION ALL
    SELECT anc.taxon_id, t.parent_taxon_id, anc.depth + 1
    FROM anc JOIN taxon t ON t.taxon_id = anc.ancestor_id
    WHERE t.parent_taxon_id IS NOT NULL
)
SELECT * FROM anc;

-- Every claim reachable by a cell type: direct (depth -1 -> rank 0) or via its taxa.
-- inherit_rank: 0 = direct on the cell type, 1 = its taxon, 2 = parent taxon, ...
CREATE VIEW v_cell_claim AS
SELECT ct.cell_type_id, ct.netlist_name, pc.claim_id, p.code AS property_code,
       0 AS inherit_rank, 1.0 AS mapping_confidence, pc.confidence
FROM cell_type ct
JOIN property_claim pc ON pc.cell_type_id = ct.cell_type_id
JOIN property p        ON p.property_id = pc.property_id
WHERE pc.superseded_by IS NULL
UNION ALL
SELECT ct.cell_type_id, ct.netlist_name, pc.claim_id, p.code,
       1 + a.depth, ctt.confidence, pc.confidence
FROM cell_type ct
JOIN cell_type_taxon ctt ON ctt.cell_type_id = ct.cell_type_id
JOIN v_taxon_ancestry a  ON a.taxon_id = ctt.taxon_id
JOIN property_claim pc   ON pc.taxon_id = a.ancestor_id
JOIN property p          ON p.property_id = pc.property_id
WHERE pc.superseded_by IS NULL
  AND (a.depth = 0 OR pc.inheritable = 1);

CREATE VIEW v_interneuron AS
SELECT ct.cell_type_id, ct.netlist_name, r.code AS region, ct.name_core,
       t.code AS primary_taxon, ctt.confidence AS taxon_confidence
FROM cell_type ct
JOIN region r ON r.region_id = ct.region_id
LEFT JOIN cell_type_taxon ctt ON ctt.cell_type_id = ct.cell_type_id AND ctt.is_primary = 1
LEFT JOIN taxon t ON t.taxon_id = ctt.taxon_id
WHERE ct.ei = 'I';
