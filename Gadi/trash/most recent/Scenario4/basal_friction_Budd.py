# Visualization of Budd (1970) Sliding Law implementation on different profiles
# Ana Fabela Hinojosa

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from model import model
from scipy.interpolate import interp1d
from bamgflowband import bamgflowband
from configf9_synthetic import ModelConfig
import argparse
import os

def setup_test_model(config):
    """Create a test model with the specified configuration"""
    # Create a higher-resolution, uniform grid for more accurate representation
    x_transformed = np.linspace(config.x_params['start'], 
                        config.x_params['end'],
                        int((config.x_params['end'] - config.x_params['start'])/config.x_params['step']) + 1)

    # Get bedrock elevation with exact wavelength
    b_transformed = config.get_bedrock_elevation(x_transformed)

    # Use linear interpolation with 'linear' method explicitly to preserve features
    mean_thickness = config.ice_params['mean_thickness']
    h_transformed = mean_thickness * np.ones_like(x_transformed)

    # Create model with improved mesh settings
    md = bamgflowband(model(), x_transformed, b_transformed + h_transformed, b_transformed, 'hmax', config.mesh_hmax)

    # Ensure the mesh points are within the interpolation domain
    mesh_x = np.clip(md.mesh.x, 
                    config.x_params['start'], 
                    config.x_params['end'] - 1e-10)  # Small buffer to avoid edge issues

    # Create interpolation functions. Use linear interpolation to ensure wavelength is preserved
    surface_interpolant = interp1d(x_transformed, b_transformed + h_transformed, kind='linear')
    base_interpolant = interp1d(x_transformed, b_transformed, kind='linear')

    # Apply interpolation to mesh points
    md.geometry.surface = surface_interpolant(mesh_x)
    md.geometry.base = base_interpolant(mesh_x)
    md.geometry.thickness = md.geometry.surface - md.geometry.base
    md.geometry.bed = md.geometry.base
    
    # Initialize friction parameters
    md.friction.coefficient = np.ones((md.mesh.numberofvertices))
    md.friction.p = np.ones((md.mesh.numberofelements))
    md.friction.q = np.ones((md.mesh.numberofelements))
    
    # Initialize material properties for viscosity calculation
    # Use the B value from config instead of calculating from temperature
    md.materials.rheology_B = config.B * np.ones(md.mesh.numberofvertices)
    
    return md

def analyze_sliding_coefficient(profile_id=1, output_dir="basal_friction_analysis", save_plots=True):
    """
    Analyze and visualize the sliding coefficient for a specific bedrock profile
    
    Args:
        profile_id: ID of the profile to analyze
        output_dir: Directory to save the output plots
        save_plots: Whether to save plots (True) or display them (False)
    """
    # Create output directory if it doesn't exist
    if save_plots and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create the configuration with the specified profile
    config = ModelConfig(profile_id=profile_id)
    
    print(f"\nAnalyzing basal friction for profile {profile_id}")
    print(f"Profile parameters:")
    print(f"  Amplitude: {config.bedrock_params['amplitude']:.5f} km")
    print(f"  Wavelength: {config.bedrock_params['lambda']:.5f} km")
    
    # Create the test model
    md = setup_test_model(config)

    # Get basal nodes
    basal_nodes = np.where(md.mesh.vertexflags(1))[0]
    x_coords = md.mesh.x
    print(f"Number of basal nodes: {len(basal_nodes)}")

    # Use the calculate_sliding_coefficient method directly from config
    # This will automatically apply the right formula and flat region value from configf9.py
    sliding_coefficient = config.calculate_sliding_coefficient(md, is_initial=True)

    # Count nodes in flat regions for reporting - ONLY FOR BASAL NODES
    straight_section_1 = config.x_params['start'] + 15.0  # First 15km is flat
    straight_section_2 = config.x_params['end'] - 15.0    # Last 15km is flat

    # Create mask for flat regions ONLY for basal nodes
    basal_x = x_coords[basal_nodes]
    flat_mask_basal = (basal_x < straight_section_1) | (basal_x >= straight_section_2)
    num_flat_nodes = np.sum(flat_mask_basal)

    # For reporting purposes, get a sample flat region value if available
    flat_region_value = "N/A"
    if num_flat_nodes > 0:
        # Find indices of basal nodes in flat regions
        flat_basal_indices = [i for i, is_flat in enumerate(flat_mask_basal) if is_flat]
        if flat_basal_indices:
            # Get the first flat basal node index
            flat_node_idx = basal_nodes[flat_basal_indices[0]]
            flat_region_value = sliding_coefficient[flat_node_idx]

    print(f"Nodes with flat-region friction value: {num_flat_nodes}")
    print(f"Flat region friction value: {flat_region_value}")

    # Generate spatial visualization of the mesh with coefficient values
    plt.figure(figsize=(15, 5))

    # Plot full mesh first
    plt.scatter(md.mesh.x, md.mesh.y, s=2, color='lightgrey', alpha=0.5, label='mesh')

    # Plot basal nodes with sliding coefficient values
    basal_x = md.mesh.x[basal_nodes]
    basal_y = md.mesh.y[basal_nodes]
    sc = plt.scatter(basal_x, basal_y, s=40, c=sliding_coefficient[basal_nodes], 
                    cmap='viridis', label='Basal nodes')
    plt.colorbar(sc, label="Budd's sliding coefficient")
    plt.title(f"Mesh with Budd's Sliding Coefficient - Profile {profile_id}")
    plt.xlabel('X coordinate (km)')
    plt.ylabel('Y coordinate (km)')
    plt.legend()
    plt.grid(True, linestyle=":", color='k', alpha=0.4)
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f"{output_dir}/profile_{profile_id}_budd_sliding_coefficient_mesh.png")
        plt.close()
    else:
        plt.show()

    # Generate plot of coefficient along the x-axis
    plt.figure(figsize=(15, 5))
    
    # Sort by x-coordinate for a clean line plot
    sort_idx = np.argsort(basal_x)
    sorted_x = basal_x[sort_idx]
    sorted_coeff = sliding_coefficient[basal_nodes][sort_idx]
    
    # Get the bed profile for reference
    bed_values = [config.get_bedrock_elevation(x) for x in sorted_x]
    
    # Scale bed profile to fit in the same plot
    bed_min = min(bed_values)
    bed_max = max(bed_values)
    bed_range = bed_max - bed_min
    coeff_min = min(sorted_coeff)
    coeff_max = max(sorted_coeff)
    coeff_range = coeff_max - coeff_min
    
    # Scale bed to match coefficient range
    scaled_bed = [(b - bed_min) / bed_range * coeff_range * 0.3 + coeff_min for b in bed_values]
    
    # Plot coefficient
    plt.plot(sorted_x, sorted_coeff, 'b-', linewidth=2, label="Sliding coefficient")
    
    # Plot scaled bed profile
    plt.plot(sorted_x, scaled_bed, 'k--', linewidth=1, alpha=0.6, label="Bed profile (scaled)")
    
    # # Add vertical lines at the boundary transitions
    # plt.axvline(x=straight_section_1, color='r', linestyle='--', alpha=0.5, label='Transition zones')
    # plt.axvline(x=straight_section_2, color='r', linestyle='--', alpha=0.5)
    
    plt.title(f"Budd's Sliding Coefficient along X-axis - Profile {profile_id}")
    plt.xlabel('X coordinate (km)')
    plt.ylabel('Coefficient value')
    plt.legend()
    plt.grid(True, linestyle=":", color='k', alpha=0.4)
    
    if save_plots:
        plt.savefig(f"{output_dir}/profile_{profile_id}_budd_sliding_coefficient_line.png")
        plt.close()
    else:
        plt.show()

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Coefficient range: {sliding_coefficient[basal_nodes].min():.3f} to {sliding_coefficient[basal_nodes].max():.3f}")
    print(f"Mean coefficient: {np.mean(sliding_coefficient[basal_nodes]):.3f}")
    print(f"Formula parameters: η={0.5 * config.B:.3e}, ω={config.omega:.3f}, β={config.bedrock_params['amplitude']:.3e}")
    print(f"Number of nodes with flat-region friction: {num_flat_nodes}")
    
    return {
        'profile_id': profile_id,
        'min_coefficient': sliding_coefficient[basal_nodes].min(),
        'max_coefficient': sliding_coefficient[basal_nodes].max(),
        'mean_coefficient': np.mean(sliding_coefficient[basal_nodes]),
        'flat_region_value': flat_region_value if flat_region_value != "N/A" else 0,
        'num_flat_nodes': num_flat_nodes,
        'amplitude': config.bedrock_params['amplitude'],
        'wavelength': config.bedrock_params['lambda'],
    }

def batch_analyze_profiles(profile_ids, output_dir="basal_friction_analysis"):
    """
    Analyze a batch of profiles and generate a comparison report
    
    Args:
        profile_ids: List of profile IDs to analyze
        output_dir: Directory to save the output plots and report
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = []
    for profile_id in profile_ids:
        print(f"\n{'='*50}")
        result = analyze_sliding_coefficient(profile_id, output_dir)
        results.append(result)
    
    # Generate a comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot mean coefficient vs wavelength
    wavelengths = [r['wavelength'] for r in results]
    mean_coeffs = [r['mean_coefficient'] for r in results]
    max_coeffs = [r['max_coefficient'] for r in results]
    min_coeffs = [r['min_coefficient'] for r in results]
    
    plt.plot(wavelengths, mean_coeffs, 'bo-', label='Mean coefficient')
    plt.fill_between(wavelengths, min_coeffs, max_coeffs, alpha=0.2, color='blue', label='Min-Max range')
    
    plt.xlabel('Wavelength (km)')
    plt.ylabel('Sliding Coefficient')
    plt.title('Sliding Coefficient vs Wavelength')
    plt.grid(True, linestyle=":", color='k', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/coefficient_vs_wavelength.png")
    plt.close()
    
    # Generate a comparison report
    with open(f"{output_dir}/comparison_report.txt", 'w') as f:
        f.write("Budd's Sliding Coefficient Analysis Report\n")
        f.write("="*50 + "\n\n")
        
        f.write("Profile Summary:\n")
        f.write("-"*50 + "\n")
        f.write(f"{'Profile ID':^10}{'Amplitude':^12}{'Wavelength':^12}{'Min Coeff':^12}{'Max Coeff':^12}{'Mean Coeff':^12}{'Flat Value':^12}\n")
        f.write("-"*70 + "\n")
        
        for r in results:
            f.write(f"{r['profile_id']:^10}{r['amplitude']:.5f}{r['wavelength']:^12.2f}{r['min_coefficient']:^12.2f}"
                    f"{r['max_coefficient']:^12.2f}{r['mean_coefficient']:^12.2f}{r['flat_region_value']:^12.2f}\n")
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")
    print(f"Comparison report: {output_dir}/comparison_report.txt")
    print(f"Comparison plot: {output_dir}/coefficient_vs_wavelength.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze Budd sliding law implementation for bedrock profiles')
    parser.add_argument('-p', '--profile', type=int, default=1, help='Profile ID to analyze')
    parser.add_argument('-b', '--batch', type=str, help='Batch analyze multiple profiles (comma-separated IDs or range)')
    parser.add_argument('-o', '--output', type=str, default="basal_friction_analysis", help='Output directory for plots and reports')
    parser.add_argument('--show', action='store_true', help='Show plots instead of saving them')
    
    args = parser.parse_args()
    
    if args.batch:
        # Parse batch specification
        if '-' in args.batch:
            # Range of profiles
            start, end = map(int, args.batch.split('-'))
            profile_ids = list(range(start, end + 1))
        else:
            # Comma-separated list
            profile_ids = [int(pid.strip()) for pid in args.batch.split(',')]
        
        batch_analyze_profiles(profile_ids, args.output)
    else:
        # Single profile analysis
        analyze_sliding_coefficient(args.profile, args.output, save_plots=not args.show)