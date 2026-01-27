# Edge-Centric Hippocampal Network Analysis

## Project Overview

This project applies **edge-centric network analysis** to hippocampal connectivity data, transforming traditional node-centric networks into edge-level networks to reveal higher-order organizational principles.

**Based on:** Betzel, R. F., Faskowitz, J., & Sporns, O. (2023). "Living on the edge: network neuroscience beyond nodes." *Trends in Cognitive Sciences*, 27(11), 1068-1084.

---

## Directory Structure

```
project/
├── code/                           # Working directory (run all code from here)
│   ├── 02a_edge_list.ipynb        # Main: Edge network creation
│   ├── centrality_hub_community_analysis.ipynb  # Hub and community detection
│   ├── paths.py                   # Path configuration (import this!)
│   ├── PROJECT_CONFIG.md          # This file
│   └── analysis_scripts/          # Additional analysis scripts
│
├── data/                          # Input data (read-only)
│   └── w_ij_gaa.csv              # Original connectivity matrix (72×72 nodes)
│
└── outputs/                       # All analysis outputs
    ├── edgelist/                 # Edge lists, statistics, matrices
    │   ├── edge_statistics.csv
    │   ├── edge_fc_matrix.npy
    │   ├── node_community_participation.csv
    │   └── pathway_*.csv
    │
    ├── netlist/                  # Network files for external tools
    │   ├── edge_network.graphml
    │   └── edge_network_analyzed.graphml
    │
    └── edgematrix/               # Visualizations
        ├── edge_network_summary.png
        ├── centrality_distributions.png
        ├── community_*.png
        └── pathway_*.png
```

---

## File Paths (from /code/ working directory)

### Input
```python
'../data/w_ij_gaa.csv'  # Original connectivity matrix
```

### Outputs
```python
'../outputs/edgelist/'   # CSV files, NPY matrices, edge lists
'../outputs/netlist/'    # GraphML and other network formats
'../outputs/edgematrix/' # All PNG visualizations and figures
```

---

## Data Description

### Input Data: `w_ij_gaa.csv`
- **Size:** 72 nodes × 72 nodes
- **Type:** Directed, weighted connectivity matrix
- **Content:** Connection weights between hippocampal cell types
- **Regions:** DG, CA3, CA2, CA1, EC (Entorhinal Cortex)
- **Ordering:** Anatomically organized by region and cell type
- **Weights:** Both positive (excitatory) and negative (inhibitory) connections

### Network Statistics
- **Nodes:** 72 cell types
- **Edges:** 1,091 non-zero connections (~22% density)
- **Edge nodes:** 1,091 (in edge-centric network)
- **Edge-edge connections:** ~35,344

---

## Analysis Workflow

### 1. Edge Network Creation (`02a_edge_list.ipynb`)
- Load original connectivity matrix
- Build node-level network
- **Transform to edge-level network** (line graph)
- Calculate edge functional connectivity (eFC)
- Export network files and statistics

**Outputs:**
- `edge_statistics.csv` → `edgelist/`
- `edge_fc_matrix.npy` → `edgelist/`
- `edge_network.graphml` → `netlist/`
- Summary figures → `edgematrix/`

### 2. Centrality & Community Analysis (`centrality_hub_community_analysis.ipynb`)
- Load edge network from GraphML
- Compute centrality measures (degree, betweenness, closeness, eigenvector, PageRank)
- Identify hub edges
- Detect communities
- Analyze community structure

**Outputs:**
- `centrality_hub_community_analysis.csv` → `edgelist/`
- `edge_community_assignments.csv` → `edgelist/`
- Centrality figures → `edgematrix/`
- Community visualizations → `edgematrix/`

### 3. Pathway-Specific Analysis
- Extract edges by anatomical pathway
- Analyze trisynaptic circuit (EC→DG→CA3→CA1)
- Compare feedforward vs recurrent connections
- Cell type-specific analysis

**Outputs:**
- `pathway_*.csv` → `edgelist/`
- `pathway_comparison.csv` → `edgelist/`
- Pathway visualizations → `edgematrix/`

---

## Key Concepts

### Edge-Centric Network
- **Traditional:** Nodes = neurons/regions, Edges = connections
- **Edge-centric:** Edges become nodes, Connected if they share a node
- **Benefit:** Reveals higher-order structure, overlapping communities

### Edge Functional Connectivity (eFC)
- Binary matrix showing which edges share nodes
- Size: n_edges × n_edges (1091 × 1091)
- Black pixel = edges share a node
- Reveals modular organization

### Edge Communities
- Communities at the edge level
- Project onto nodes → **overlapping** node communities
- More realistic than traditional non-overlapping partitions

---

## Anatomical Regions

### Hippocampal Regions (in data order)
1. **DG (Dentate Gyrus)** - 13 cell types (nodes 0-12)
2. **CA3** - 10 cell types (nodes 13-22)
3. **CA2** - 4 cell types (nodes 23-26)
4. **CA1** - 24 cell types (nodes 27-50)
5. **EC (Entorhinal Cortex)** - 21 cell types (nodes 51-71)

### Major Pathways
- **Perforant Path:** EC → DG
- **Mossy Fibers:** DG → CA3
- **Schaffer Collaterals:** CA3 → CA1
- **Back Projection:** CA1 → EC
- **Recurrent:** CA3 → CA3, CA1 → CA1

---

## Usage Instructions

### Quick Start
```python
# In any notebook/script in /code/
from paths import CONNECTIVITY_FILE, OUTPUT_DIRS

# Load data
df = pd.read_csv(CONNECTIVITY_FILE)

# Save outputs
plt.savefig(f"{OUTPUT_DIRS['edgematrix']}/my_figure.png")
stats.to_csv(f"{OUTPUT_DIRS['edgelist']}/my_stats.csv")
```

### Running Analyses
1. Start in `/code/` directory
2. Run `02a_edge_list.ipynb` first (creates edge network)
3. Then run `centrality_hub_community_analysis.ipynb`
4. Use pathway analysis scripts for specific circuits

### File Naming Conventions
- **Edge lists:** `edge_statistics.csv`, `pathway_REGION_to_REGION_edges.csv`
- **Matrices:** `edge_fc_matrix.npy`
- **Networks:** `edge_network.graphml`, `edge_network_analyzed.graphml`
- **Figures:** `descriptive_name.png` (e.g., `pathway_eFC_CA3_to_CA1.png`)

---

## Software Requirements

### Python Packages
```bash
pip install numpy pandas matplotlib seaborn scipy igraph
```

### Required Libraries
- `numpy` - Array operations
- `pandas` - Data manipulation
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `scipy` - Scientific computing
- `igraph` (python-igraph) - Network analysis

---

## Common Tasks

### Load Edge Network
```python
import igraph as ig
G_edges = ig.Graph.Read_GraphML('../outputs/netlist/edge_network.graphml')
```

### Load Edge Statistics
```python
stats = pd.read_csv('../outputs/edgelist/edge_statistics.csv')
```

### Load eFC Matrix
```python
eFC = np.load('../outputs/edgelist/edge_fc_matrix.npy')
```

### Analyze Specific Pathway
```python
from analyze_pathways_organized import get_edges_by_pathway, visualize_pathway_efc

# Get CA3 → CA1 edges
edge_indices, pathway_edges = get_edges_by_pathway('CA3', 'CA1')

# Visualize
visualize_pathway_efc('CA3', 'CA1')
```

---

## Troubleshooting

### Issue: "File not found"
- Check you're in `/code/` directory
- Use paths from `paths.py`
- Verify file exists in correct output folder

### Issue: "Module not found"
- Run notebooks from `/code/` directory
- Check Python packages installed
- Use `from paths import ...` for path configs

### Issue: "Empty edge network"
- Make sure `02a_edge_list.ipynb` ran successfully
- Check `edge_statistics.csv` has data
- Verify `edge_network.graphml` exists in `netlist/`

---

## Notes

### Data Properties
- Original matrix is anatomically ordered
- Edges inherit this organization
- Negative weights represent inhibitory connections
- Network is moderately sparse (~22% density)
- No missing data (0% NaN values)

### Analysis Notes
- Edge network is 15× larger than node network (1091 vs 72)
- eFC matrix is ~1.2M elements (1091²)
- Hub edges have degree > 90
- 6 major edge communities detected
- Modularity: ~0.52 (high)

---

## References

1. Betzel, R. F., Faskowitz, J., & Sporns, O. (2023). Living on the edge: network neuroscience beyond nodes. *Trends in Cognitive Sciences*, 27(11), 1068-1084.

2. Rubinov, M., & Sporns, O. (2010). Complex network measures of brain connectivity: Uses and interpretations. *NeuroImage*, 52(3), 1059-1069.

3. Bullmore, E., & Sporns, O. (2009). Complex brain networks: graph theoretical analysis of structural and functional systems. *Nature Reviews Neuroscience*, 10(3), 186-198.

---

## Contact & Updates

**Last Updated:** January 2025
**Analysis Version:** 1.0
**Dataset:** Hippocampal connectivity (72 cell types)

For questions or issues, refer to notebook comments or analysis scripts.
