"""
Pathway-Specific Edge Network Analysis

Output Organization:
- ../outputs/edgelist/   - Edge lists and statistics
- ../outputs/netlist/    - Network files (graphml, etc)
- ../outputs/edgematrix/ - Edge matrix visualizations and analyses
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================================
# SETUP OUTPUT DIRECTORIES
# ============================================================================

# Define output folders
OUTPUT_DIRS = {
    'edgelist': '../outputs/edgelist/',
    'netlist': '../outputs/netlist/',
    'edgematrix': '../outputs/edgematrix/'
}

# Create directories if they don't exist
for dir_path in OUTPUT_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)
    print(f"✓ Output directory ready: {dir_path}")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

# Load original connectivity matrix
df = pd.read_csv('w_ij_gaa.csv', index_col=0)
W = df.values
node_labels = df.columns.tolist()

# Load edge statistics
stats = pd.read_csv(OUTPUT_DIRS['edgelist'] + 'edge_statistics.csv')
print(f"✓ Loaded edge statistics: {len(stats)} edges")

# Load edge FC matrix
eFC = np.load(OUTPUT_DIRS['edgelist'] + 'edge_fc_matrix.npy')
print(f"✓ Loaded edge FC matrix: {eFC.shape}")

# ============================================================================
# DEFINE ANATOMICAL REGIONS
# ============================================================================

regions = {
    'DG': [i for i, name in enumerate(node_labels) if name.startswith('DG')],
    'CA3': [i for i, name in enumerate(node_labels) if name.startswith('CA3')],
    'CA2': [i for i, name in enumerate(node_labels) if name.startswith('CA2')],
    'CA1': [i for i, name in enumerate(node_labels) if name.startswith('CA1')],
    'EC': [i for i, name in enumerate(node_labels) if 'EC' in name or 'MEC' in name or 'LEC' in name]
}

print("\n" + "="*70)
print("ANATOMICAL REGIONS")
print("="*70)
for region, indices in regions.items():
    print(f"  {region}: {len(indices)} cell types (node indices {min(indices)}-{max(indices)})")

# ============================================================================
# PATHWAY EXTRACTION FUNCTIONS
# ============================================================================

def get_edges_by_pathway(source_region=None, target_region=None):
    """Extract edges for a specific pathway"""
    
    if source_region and source_region in regions:
        source_indices = regions[source_region]
    else:
        source_indices = range(len(node_labels))
    
    if target_region and target_region in regions:
        target_indices = regions[target_region]
    else:
        target_indices = range(len(node_labels))
    
    # Find matching edges
    edge_indices = []
    for idx, row in stats.iterrows():
        source_idx = row['source_idx']
        target_idx = row['target_idx']
        
        if source_idx in source_indices and target_idx in target_indices:
            edge_indices.append(idx)
    
    return edge_indices, stats.loc[edge_indices]


def summarize_pathway(source_region, target_region, save_csv=True):
    """Get summary statistics for a pathway"""
    
    edge_indices, pathway_edges = get_edges_by_pathway(source_region, target_region)
    
    print(f"\n{'='*70}")
    print(f"PATHWAY: {source_region} → {target_region}")
    print(f"{'='*70}")
    print(f"Number of edges: {len(edge_indices)}")
    
    if len(edge_indices) == 0:
        print("No edges found for this pathway!")
        return None, None
    
    # Statistics
    if 'degree' in pathway_edges.columns:
        print(f"\nDegree statistics:")
        print(f"  Mean: {pathway_edges['degree'].mean():.2f}")
        print(f"  Max: {pathway_edges['degree'].max():.0f}")
        print(f"  Min: {pathway_edges['degree'].min():.0f}")
    
    if 'betweenness' in pathway_edges.columns:
        print(f"\nBetweenness statistics:")
        print(f"  Mean: {pathway_edges['betweenness'].mean():.2f}")
        print(f"  Max: {pathway_edges['betweenness'].max():.2f}")
    
    if 'original_weight' in pathway_edges.columns:
        print(f"\nOriginal weight statistics:")
        print(f"  Mean: {pathway_edges['original_weight'].mean():.2f}")
        print(f"  Sum: {pathway_edges['original_weight'].sum():.2f}")
    
    # Top edges
    print(f"\nTop 5 edges by betweenness:")
    top_edges = pathway_edges.nlargest(5, 'betweenness') if 'betweenness' in pathway_edges.columns else pathway_edges.head(5)
    for idx, edge in top_edges.iterrows():
        print(f"  • {edge['edge_label'][:60]}")
        if 'betweenness' in edge:
            print(f"    Betweenness: {edge['betweenness']:.2f}")
    
    # Save pathway edges to CSV
    if save_csv:
        filename = OUTPUT_DIRS['edgelist'] + f'pathway_{source_region}_to_{target_region}_edges.csv'
        pathway_edges.to_csv(filename, index=False)
        print(f"\n✓ Saved edge list: {filename}")
    
    return edge_indices, pathway_edges


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_pathway_efc(source_region, target_region, save=True):
    """Visualize edge FC matrix for a specific pathway"""
    
    edge_indices, pathway_edges = get_edges_by_pathway(source_region, target_region)
    
    if len(edge_indices) < 2:
        print(f"Not enough edges for {source_region}→{target_region} visualization")
        return None
    
    # Extract submatrix
    pathway_efc = eFC[np.ix_(edge_indices, edge_indices)]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Panel 1: Pathway eFC matrix
    ax = axes[0]
    im = ax.imshow(pathway_efc, cmap='binary', aspect='auto', interpolation='nearest')
    ax.set_title(f'Edge FC Matrix: {source_region} → {target_region}\\n({len(edge_indices)} edges)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Edge Index (within pathway)')
    ax.set_ylabel('Edge Index (within pathway)')
    plt.colorbar(im, ax=ax, label='Connected')
    
    # Panel 2: Degree distribution
    ax = axes[1]
    degrees = pathway_efc.sum(axis=1)
    ax.hist(degrees, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(degrees.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean={degrees.mean():.1f}')
    ax.set_xlabel('Degree (within pathway)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Degree Distribution: {source_region} → {target_region}',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filename = OUTPUT_DIRS['edgematrix'] + f'pathway_eFC_{source_region}_to_{target_region}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    else:
        plt.show()
    
    return pathway_efc


def compare_pathways(pathway_list, save=True):
    """Compare statistics across multiple pathways"""
    
    results = []
    
    for source, target in pathway_list:
        edge_indices, pathway_edges = get_edges_by_pathway(source, target)
        
        if len(edge_indices) == 0:
            continue
        
        # Extract submatrix for internal connectivity
        pathway_efc = eFC[np.ix_(edge_indices, edge_indices)]
        internal_density = pathway_efc.sum() / (len(edge_indices) ** 2)
        
        results.append({
            'pathway': f'{source}→{target}',
            'n_edges': len(edge_indices),
            'mean_degree': pathway_edges['degree'].mean() if 'degree' in pathway_edges else 0,
            'mean_betweenness': pathway_edges['betweenness'].mean() if 'betweenness' in pathway_edges else 0,
            'internal_density': internal_density,
            'total_weight': pathway_edges['original_weight'].sum() if 'original_weight' in pathway_edges else 0
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    
    print(f"\n{'='*70}")
    print("PATHWAY COMPARISON")
    print(f"{'='*70}")
    print(comparison_df.to_string(index=False))
    
    # Save comparison to CSV
    if save:
        csv_filename = OUTPUT_DIRS['edgelist'] + 'pathway_comparison.csv'
        comparison_df.to_csv(csv_filename, index=False)
        print(f"\n✓ Saved comparison table: {csv_filename}")
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Number of edges
    ax = axes[0, 0]
    ax.bar(range(len(comparison_df)), comparison_df['n_edges'], color='steelblue')
    ax.set_xticks(range(len(comparison_df)))
    ax.set_xticklabels(comparison_df['pathway'], rotation=45, ha='right')
    ax.set_ylabel('Number of Edges')
    ax.set_title('Edge Count by Pathway', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Mean degree
    ax = axes[0, 1]
    ax.bar(range(len(comparison_df)), comparison_df['mean_degree'], color='coral')
    ax.set_xticks(range(len(comparison_df)))
    ax.set_xticklabels(comparison_df['pathway'], rotation=45, ha='right')
    ax.set_ylabel('Mean Degree')
    ax.set_title('Average Degree by Pathway', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Internal density
    ax = axes[1, 0]
    ax.bar(range(len(comparison_df)), comparison_df['internal_density'], color='green')
    ax.set_xticks(range(len(comparison_df)))
    ax.set_xticklabels(comparison_df['pathway'], rotation=45, ha='right')
    ax.set_ylabel('Internal Density')
    ax.set_title('Internal Connectivity Density', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Total weight
    ax = axes[1, 1]
    ax.bar(range(len(comparison_df)), comparison_df['total_weight'], color='purple')
    ax.set_xticks(range(len(comparison_df)))
    ax.set_xticklabels(comparison_df['pathway'], rotation=45, ha='right')
    ax.set_ylabel('Total Weight')
    ax.set_title('Total Connection Weight', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save:
        filename = OUTPUT_DIRS['edgematrix'] + 'pathway_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization: {filename}")
        plt.close()
    else:
        plt.show()
    
    return comparison_df


def analyze_cell_type(cell_pattern, as_source=True, as_target=True, save_csv=True):
    """Analyze edges involving a specific cell type"""
    
    print(f"\n{'='*70}")
    print(f"CELL TYPE ANALYSIS: '{cell_pattern}'")
    print(f"{'='*70}")
    
    results = {'as_source': None, 'as_target': None}
    
    if as_source:
        source_edges = stats[stats['source_node'].str.contains(cell_pattern, case=False, na=False)]
        print(f"\nAs SOURCE: {len(source_edges)} edges")
        
        if len(source_edges) > 0:
            print(f"  Targets most connected to:")
            target_counts = source_edges['target_node'].value_counts().head(5)
            for target, count in target_counts.items():
                print(f"    • {target}: {count} connections")
            results['as_source'] = source_edges
            
            if save_csv:
                filename = OUTPUT_DIRS['edgelist'] + f'celltype_{cell_pattern.replace(" ", "_")}_as_source.csv'
                source_edges.to_csv(filename, index=False)
                print(f"  ✓ Saved: {filename}")
    
    if as_target:
        target_edges = stats[stats['target_node'].str.contains(cell_pattern, case=False, na=False)]
        print(f"\nAs TARGET: {len(target_edges)} edges")
        
        if len(target_edges) > 0:
            print(f"  Sources most connected from:")
            source_counts = target_edges['source_node'].value_counts().head(5)
            for source, count in source_counts.items():
                print(f"    • {source}: {count} connections")
            results['as_target'] = target_edges
            
            if save_csv:
                filename = OUTPUT_DIRS['edgelist'] + f'celltype_{cell_pattern.replace(" ", "_")}_as_target.csv'
                target_edges.to_csv(filename, index=False)
                print(f"  ✓ Saved: {filename}")
    
    return results


def visualize_pathway_in_full_matrix(source_region, target_region, save=True):
    """Show where a pathway appears in the full eFC matrix"""
    
    edge_indices, _ = get_edges_by_pathway(source_region, target_region)
    
    if len(edge_indices) == 0:
        print("No edges found for this pathway")
        return
    
    # Create highlight mask
    highlight = np.zeros_like(eFC)
    highlight[np.ix_(edge_indices, edge_indices)] = 1
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot full matrix in grayscale
    ax.imshow(eFC, cmap='Greys', aspect='auto', alpha=0.3)
    
    # Overlay highlighted region
    masked = np.ma.masked_where(highlight == 0, highlight)
    ax.imshow(masked, cmap='Reds', aspect='auto', alpha=0.8)
    
    # Mark boundaries
    if len(edge_indices) > 1:
        min_idx = min(edge_indices)
        max_idx = max(edge_indices)
        ax.axhline(min_idx, color='red', linewidth=2, linestyle='--', alpha=0.7)
        ax.axhline(max_idx, color='red', linewidth=2, linestyle='--', alpha=0.7)
        ax.axvline(min_idx, color='red', linewidth=2, linestyle='--', alpha=0.7)
        ax.axvline(max_idx, color='red', linewidth=2, linestyle='--', alpha=0.7)
    
    ax.set_title(f'Full eFC Matrix with {source_region}→{target_region} Highlighted\\n'
                 f'(Red region: {len(edge_indices)} edges, indices {min(edge_indices)}-{max(edge_indices)})',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Edge Index')
    ax.set_ylabel('Edge Index')
    
    plt.tight_layout()
    
    if save:
        filename = OUTPUT_DIRS['edgematrix'] + f'full_matrix_with_{source_region}_to_{target_region}_highlighted.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    else:
        plt.show()


# ============================================================================
# EXAMPLE ANALYSES
# ============================================================================

print("\n" + "="*70)
print("RUNNING EXAMPLE ANALYSES")
print("="*70)

# Example 1: Trisynaptic pathway
print("\n" + "="*70)
print("1. TRISYNAPTIC PATHWAY")
print("="*70)

summarize_pathway('EC', 'DG')  # Perforant path
visualize_pathway_efc('EC', 'DG')

summarize_pathway('DG', 'CA3')  # Mossy fibers
visualize_pathway_efc('DG', 'CA3')

summarize_pathway('CA3', 'CA1')  # Schaffer collaterals
visualize_pathway_efc('CA3', 'CA1')

# Example 2: Compare major pathways
print("\n" + "="*70)
print("2. COMPARE MAJOR HIPPOCAMPAL PATHWAYS")
print("="*70)

major_pathways = [
    ('EC', 'DG'),
    ('DG', 'CA3'),
    ('CA3', 'CA1'),
    ('CA1', 'EC'),
    ('CA3', 'CA3'),  # Recurrent
    ('CA1', 'CA1')   # Recurrent
]

comparison_df = compare_pathways(major_pathways)

# Example 3: Cell type analysis
print("\n" + "="*70)
print("3. PYRAMIDAL CELL ANALYSIS")
print("="*70)

ca1_pyr = analyze_cell_type('CA1 Pyramidal')
ca3_pyr = analyze_cell_type('CA3 Pyramidal')

# Example 4: Highlight in full matrix
print("\n" + "="*70)
print("4. VISUALIZE PATHWAYS IN FULL MATRIX")
print("="*70)

visualize_pathway_in_full_matrix('CA3', 'CA1')
visualize_pathway_in_full_matrix('DG', 'CA3')

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nOutput locations:")
print(f"  Edge lists: {OUTPUT_DIRS['edgelist']}")
print(f"  Network files: {OUTPUT_DIRS['netlist']}")
print(f"  Visualizations: {OUTPUT_DIRS['edgematrix']}")
