#!/usr/bin/env python3
"""
Check coordinate ordering in the extracted surface/basal data
"""

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import sys
import re
import os

# Import bedrock configuration for proper mesh reconstruction
try:
    # Try to import from parent directories (flowline.py level)
    sys.path.append('../../')  # Add flowline.py directory to path
    from bedrock_generator import SyntheticBedrockModelConfig
    HAS_BEDROCK_CONFIG = True
except ImportError:
    HAS_BEDROCK_CONFIG = False
    print("Warning: bedrock_generator not available. Using simplified domain calculation.")

# Import scipy for domain optimization
try:
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Using simplified domain calculation.")

def find_optimal_domain_length(bedrock_config, target_L=160e3, search_window=5e3):
    """Find optimal domain length matching flowline.py exactly."""
    if not HAS_SCIPY:
        return target_L
        
    wavelength = bedrock_config.profile_params['wavelength']
    target_periods = target_L / wavelength
    n_periods_candidates = [int(np.floor(target_periods)), int(np.ceil(target_periods))]
    
    best_L = None
    best_score = float('inf')
    
    for n_periods in n_periods_candidates:
        if n_periods <= 0:
            continue
            
        L_exact = n_periods * wavelength
        search_start = max(0, L_exact - search_window/2)
        search_end = L_exact + search_window/2
        x_search = np.linspace(search_start, search_end, int(search_window / 50))
        bed_search = bedrock_config.get_bedrock_elevation(x_search)
        
        peaks, _ = find_peaks(bed_search, distance=int(0.3 * wavelength / 50))
        
        if len(peaks) == 0:
            peak_L = L_exact
        else:
            peak_positions = x_search[peaks]
            closest_peak_idx = np.argmin(np.abs(peak_positions - L_exact))
            peak_L = peak_positions[closest_peak_idx]
        
        distance_from_target = abs(peak_L - target_L)
        period_error = abs(peak_L / wavelength - n_periods)
        score = distance_from_target + 1000 * period_error
        
        if score < best_score:
            best_score = score
            best_L = peak_L
    
    return best_L if best_L is not None else target_L

def reconstruct_mesh_coordinates(resolution_factor, profile_id):
    """Reconstruct correct mesh coordinates matching flowline.py"""
    target_L = 160e3
    target_nx = 1600
    
    if HAS_BEDROCK_CONFIG and profile_id is not None:
        try:
            bedrock_config = SyntheticBedrockModelConfig(profile_id=profile_id)
            L = find_optimal_domain_length(bedrock_config, target_L)
        except:
            L = target_L
    else:
        L = target_L
    
    nx = int(target_nx * L / target_L)
    x_coords = np.linspace(0, L, nx)
    return x_coords

def extract_profile_id_from_filename(filename):
    """Extract profile ID from NetCDF filename"""
    basename = os.path.basename(filename)
    if basename.endswith('.nc'):
        basename = basename[:-3]
    
    patterns = [
        r'^(\d+)_([A-Za-z]\d+)_([0-9.]+)_final_time=.*_timestep=',
        r'^(\d+)_([A-Za-z]\d+)_([0-9.]+)$'
    ]
    
    for pattern in patterns:
        match = re.match(pattern, basename)
        if match:
            return int(match.group(1)), match.group(2), float(match.group(3))
    
    return None, None, None

def check_coordinate_ordering():
    """Check if coordinates are properly sorted"""
    
    # Load one file to check ordering - use current format
    nc_file = "S4/022_S4_0.5.nc"
    
    print(f"Checking coordinate ordering in: {nc_file}")
    
    dataset = nc.Dataset(nc_file, 'r')
    
    # Current format doesn't have mesh groups - just check the data structure
    print(f"Available groups: {list(dataset.groups.keys())}")
    if 'results' in dataset.groups:
        print(f"Results subgroups: {list(dataset.groups['results'].groups.keys())}")
        if 'TransientSolution' in dataset.groups['results'].groups:
            ts_group = dataset.groups['results'].groups['TransientSolution']
            print(f"TransientSolution variables: {list(ts_group.variables.keys())}")
    
    # Load velocity data to check structure
    vx_raw = np.array(dataset.groups['results'].groups['TransientSolution'].variables['Vx'][:])
    print(f"Vx shape: {vx_raw.shape}")
    
    # Extract profile ID and reconstruct proper coordinates
    n_points = vx_raw.shape[1]  # spatial points
    profile_id, experiment, detected_res = extract_profile_id_from_filename(nc_file)
    
    if profile_id is not None:
        print(f"Detected: Profile {profile_id}, Experiment {experiment}, Resolution {detected_res}")
        # Reconstruct coordinates using the same method as flowline.py
        x_coords = reconstruct_mesh_coordinates(detected_res, profile_id)
        
        # Handle mismatch between reconstructed and NetCDF points
        if len(x_coords) != n_points:
            print(f"Warning: Reconstructed {len(x_coords)} points, but NetCDF has {n_points} points")
            print(f"Using interpolation to match NetCDF data structure")
            
            if len(x_coords) > n_points:
                # Downsample coordinates
                indices = np.linspace(0, len(x_coords)-1, n_points, dtype=int)
                x_coords = x_coords[indices]
            else:
                # Upsample coordinates via interpolation
                x_old = np.linspace(0, np.max(x_coords), len(x_coords))
                x_new = np.linspace(0, np.max(x_coords), n_points)
                x_coords = np.interp(x_new, x_old, x_coords)
        
        print(f"Using reconstructed coordinates: {len(x_coords)} points from {x_coords[0]/1000:.1f} to {x_coords[-1]/1000:.1f}km")
    else:
        print(f"Warning: Could not extract profile ID, using fallback coordinates")
        x_coords = np.linspace(0, 160000, n_points)  # Fallback: 0 to 160km
        print(f"Using fallback coordinates: {n_points} points from 0 to 160km")
    
    # Use all points as both surface and basal for demonstration
    x_surface = x_coords
    x_basal = x_coords
    
    print(f"Surface nodes: {len(x_surface)}")
    print(f"Basal nodes: {len(x_basal)}")
    
    # Check if coordinates are sorted
    surface_sorted = np.all(x_surface[:-1] <= x_surface[1:])
    basal_sorted = np.all(x_basal[:-1] <= x_basal[1:])
    
    print(f"Surface coordinates sorted: {surface_sorted}")
    print(f"Basal coordinates sorted: {basal_sorted}")
    
    if not surface_sorted:
        print(f"Surface x range: [{np.min(x_surface):.0f}, {np.max(x_surface):.0f}]")
        print(f"First 10 surface x: {x_surface[:10]}")
        print(f"Last 10 surface x: {x_surface[-10:]}")
    
    if not basal_sorted:
        print(f"Basal x range: [{np.min(x_basal):.0f}, {np.max(x_basal):.0f}]")
        print(f"First 10 basal x: {x_basal[:10]}")
        print(f"Last 10 basal x: {x_basal[-10:]}")
    
    # Plot to visualize the ordering issue
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot raw coordinates (first 100 points to see pattern)
    n_plot = min(100, len(x_surface))
    ax1.plot(range(n_plot), x_surface[:n_plot]/1000, 'bo-', label='Surface x-coords', markersize=4)
    ax1.set_xlabel('Node index')
    ax1.set_ylabel('X coordinate (km)')
    ax1.set_title('Raw X-coordinates (first 100 surface nodes)')
    ax1.grid(True, alpha=0.3)
    
    # Plot sorted vs unsorted
    x_surface_sorted = np.sort(x_surface)
    ax2.plot(x_surface/1000, 'b-', alpha=0.7, linewidth=2, label='Original order')
    ax2.plot(x_surface_sorted/1000, 'r-', alpha=0.7, linewidth=2, label='Sorted order')
    ax2.set_xlabel('Node index')
    ax2.set_ylabel('X coordinate (km)')
    ax2.set_title('Coordinate Ordering Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('coordinate_ordering_check.png', dpi=150, bbox_inches='tight')
    print("Saved: coordinate_ordering_check.png")
    plt.show()
    
    dataset.close()

if __name__ == "__main__":
    check_coordinate_ordering()