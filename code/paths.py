"""
Project Path Configuration

This module centralizes all file paths for the edge-centric network analysis project.
Import this in all notebooks and scripts to ensure consistent file locations.

Usage:
    from paths import CONNECTIVITY_FILE, OUTPUT_DIRS
    
    # Load data
    df = pd.read_csv(CONNECTIVITY_FILE)
    
    # Save outputs
    fig.savefig(f"{OUTPUT_DIRS['edgematrix']}/my_figure.png")

Directory Structure:
    project/
    ├── code/          # Working directory (run scripts from here)
    ├── data/          # Input data
    └── outputs/       # All outputs
        ├── edgelist/
        ├── netlist/
        └── edgematrix/
"""

import os

# ============================================================================
# BASE DIRECTORIES
# ============================================================================

# Get the directory containing this file (should be /code/)
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

# Parent directory (project root)
PROJECT_ROOT = os.path.dirname(CODE_DIR)

# Main directories
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUTS_BASE = os.path.join(PROJECT_ROOT, 'outputs')

# ============================================================================
# INPUT DATA PATHS
# ============================================================================

# Original connectivity matrix
CONNECTIVITY_FILE = os.path.join(DATA_DIR, 'w_ij_gaa.csv')

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================

OUTPUT_DIRS = {
    'edgelist': os.path.join(OUTPUTS_BASE, 'edgelist'),
    'netlist': os.path.join(OUTPUTS_BASE, 'netlist'),
    'edgematrix': os.path.join(OUTPUTS_BASE, 'edgematrix')
}

# Individual output paths (for convenience)
EDGELIST_DIR = OUTPUT_DIRS['edgelist']
NETLIST_DIR = OUTPUT_DIRS['netlist']
EDGEMATRIX_DIR = OUTPUT_DIRS['edgematrix']

# ============================================================================
# COMMON OUTPUT FILES
# ============================================================================

# Edge network files
EDGE_STATS_FILE = os.path.join(EDGELIST_DIR, 'edge_statistics.csv')
EDGE_FC_MATRIX_FILE = os.path.join(EDGELIST_DIR, 'edge_fc_matrix.npy')
EDGE_NETWORK_FILE = os.path.join(NETLIST_DIR, 'edge_network.graphml')
EDGE_NETWORK_ANALYZED_FILE = os.path.join(NETLIST_DIR, 'edge_network_analyzed.graphml')

# Community files
NODE_PARTICIPATION_FILE = os.path.join(EDGELIST_DIR, 'node_community_participation.csv')
EDGE_COMMUNITIES_FILE = os.path.join(EDGELIST_DIR, 'edge_community_assignments.csv')

# Analysis results
CENTRALITY_RESULTS_FILE = os.path.join(EDGELIST_DIR, 'centrality_hub_community_analysis.csv')
PATHWAY_COMPARISON_FILE = os.path.join(EDGELIST_DIR, 'pathway_comparison.csv')
ANALYSIS_SUMMARY_FILE = os.path.join(EDGELIST_DIR, 'analysis_summary.txt')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def initialize_output_dirs():
    """
    Create all output directories if they don't exist.
    Call this at the start of any analysis script.
    """
    for dir_name, dir_path in OUTPUT_DIRS.items():
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Output directory ready: {dir_path}")


def get_pathway_edge_file(source_region, target_region):
    """
    Get the standard filename for pathway-specific edge lists.
    
    Args:
        source_region: Source anatomical region (e.g., 'CA3')
        target_region: Target anatomical region (e.g., 'CA1')
    
    Returns:
        str: Full path to pathway edge list file
    """
    filename = f'pathway_{source_region}_to_{target_region}_edges.csv'
    return os.path.join(EDGELIST_DIR, filename)


def get_pathway_figure_file(source_region, target_region, figure_type='eFC'):
    """
    Get the standard filename for pathway-specific figures.
    
    Args:
        source_region: Source anatomical region
        target_region: Target anatomical region
        figure_type: Type of figure (e.g., 'eFC', 'comparison', 'highlighted')
    
    Returns:
        str: Full path to pathway figure file
    """
    filename = f'pathway_{figure_type}_{source_region}_to_{target_region}.png'
    return os.path.join(EDGEMATRIX_DIR, filename)


def get_celltype_file(cell_pattern, role='source'):
    """
    Get the standard filename for cell type-specific edge lists.
    
    Args:
        cell_pattern: Cell type name or pattern
        role: 'source' or 'target'
    
    Returns:
        str: Full path to cell type edge list file
    """
    safe_name = cell_pattern.replace(' ', '_').replace('/', '_')
    filename = f'celltype_{safe_name}_as_{role}.csv'
    return os.path.join(EDGELIST_DIR, filename)


def verify_paths():
    """
    Verify that all expected paths exist and are accessible.
    Useful for debugging path issues.
    """
    print("="*70)
    print("PATH VERIFICATION")
    print("="*70)
    
    print(f"\nWorking directory: {os.getcwd()}")
    print(f"Code directory: {CODE_DIR}")
    print(f"Project root: {PROJECT_ROOT}")
    
    print(f"\nData directory: {DATA_DIR}")
    print(f"  Exists: {os.path.exists(DATA_DIR)}")
    print(f"  Connectivity file: {CONNECTIVITY_FILE}")
    print(f"    Exists: {os.path.exists(CONNECTIVITY_FILE)}")
    
    print(f"\nOutput directories:")
    for name, path in OUTPUT_DIRS.items():
        exists = os.path.exists(path)
        print(f"  {name}: {path}")
        print(f"    Exists: {exists}")
        if exists:
            files = os.listdir(path)
            print(f"    Files: {len(files)}")


def list_output_files(output_type='all'):
    """
    List all files in output directories.
    
    Args:
        output_type: 'all', 'edgelist', 'netlist', or 'edgematrix'
    """
    if output_type == 'all':
        dirs_to_check = OUTPUT_DIRS.items()
    else:
        dirs_to_check = [(output_type, OUTPUT_DIRS[output_type])]
    
    print("="*70)
    print("OUTPUT FILES")
    print("="*70)
    
    for dir_name, dir_path in dirs_to_check:
        print(f"\n{dir_name.upper()}: {dir_path}")
        if os.path.exists(dir_path):
            files = sorted(os.listdir(dir_path))
            if files:
                for f in files:
                    file_path = os.path.join(dir_path, f)
                    size = os.path.getsize(file_path)
                    size_str = f"{size:,} bytes" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
                    print(f"  • {f} ({size_str})")
            else:
                print("  (empty)")
        else:
            print("  (directory does not exist)")


# ============================================================================
# INITIALIZATION
# ============================================================================

# Automatically create output directories when this module is imported
initialize_output_dirs()

# ============================================================================
# MODULE INFORMATION
# ============================================================================

__version__ = '1.0.0'
__author__ = 'Edge-Centric Network Analysis Project'

if __name__ == '__main__':
    # When run directly, perform verification
    print(__doc__)
    verify_paths()
    print()
    list_output_files()
