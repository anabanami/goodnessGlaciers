#!/usr/bin/env python3
"""
Debug the exact loading process for NetCDF files without mesh groups
"""

import numpy as np
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
    """
    Find domain length that guarantees an integer number of bedrock periods
    and ends at/near a bedrock peak for optimal boundary conditions.
    Matches find_optimal_domain_length() from flowline.py exactly.
    """
    if not HAS_SCIPY:
        print(f"Warning: scipy not available for optimal domain calculation")
        return target_L
        
    wavelength = bedrock_config.profile_params['wavelength']
    
    # Step 1: Find the closest integer number of periods to target
    target_periods = target_L / wavelength
    n_periods_candidates = [int(np.floor(target_periods)), int(np.ceil(target_periods))]
    
    print(f"Domain optimization for integer periods:")
    print(f"  Target L: {target_L/1000:.1f} km")
    print(f"  Wavelength: {wavelength/1000:.1f} km")
    print(f"  Target periods: {target_periods:.2f}")
    print(f"  Candidate periods: {n_periods_candidates}")
    
    best_L = None
    best_score = float('inf')
    best_n_periods = None
    
    for n_periods in n_periods_candidates:
        if n_periods <= 0:
            continue
            
        # Exact domain length for integer periods
        L_exact = n_periods * wavelength
        
        # Search for peaks near this exact length
        search_start = max(0, L_exact - search_window/2)
        search_end = L_exact + search_window/2
        x_search = np.linspace(search_start, search_end, int(search_window / 50))  # 50m resolution
        bed_search = bedrock_config.get_bedrock_elevation(x_search)
        
        # Find peaks in this region
        peaks, properties = find_peaks(bed_search, 
                                       distance=int(0.3 * wavelength / 50))
        
        if len(peaks) == 0:
            # No peaks found, use exact integer period length
            peak_L = L_exact
            distance_from_target = abs(L_exact - target_L)
        else:
            # Find peak closest to exact integer period length
            peak_positions = x_search[peaks]
            closest_peak_idx = np.argmin(np.abs(peak_positions - L_exact))
            peak_L = peak_positions[closest_peak_idx]
            distance_from_target = abs(peak_L - target_L)
        
        # Score based on: 1) distance from target, 2) how close to exact integer periods
        period_error = abs(peak_L / wavelength - n_periods)
        score = distance_from_target + 1000 * period_error  # Heavily weight period accuracy
        
        print(f"  n={n_periods}: L={peak_L/1000:.3f} km, periods={peak_L/wavelength:.4f}, score={score:.1f}")
        
        if score < best_score:
            best_score = score
            best_L = peak_L
            best_n_periods = n_periods
    
    if best_L is None:
        # Fallback: use exact integer periods closest to target
        n_periods = round(target_periods)
        best_L = n_periods * wavelength
        best_n_periods = n_periods
        print(f"  Fallback: Using exact {n_periods} periods")
    
    # Final validation
    actual_periods = best_L / wavelength
    period_error = abs(actual_periods - best_n_periods)
    
    print(f"Final domain configuration:")
    print(f"  Optimal L: {best_L/1000:.3f} km")
    print(f"  Integer periods: {best_n_periods}")
    print(f"  Actual periods: {actual_periods:.6f}")
    print(f"  Period error: {period_error:.6f}")
    print(f"  Adjustment from target: {(best_L - target_L)/1000:.3f} km")
    
    # Ensure period error is negligible (< 0.1% of a period)
    if period_error > 0.001:
        print(f"  WARNING: Period error {period_error:.6f} exceeds tolerance!")
    else:
        print(f"  ✓ Period precision verified")
    
    return best_L

def reconstruct_mesh_coordinates(resolution_factor, profile_id):
    """Reconstruct correct mesh coordinates matching flowline.py"""
    # Domain parameters - must match flowline.py exactly
    target_L = 160e3  # Target domain length (m)
    target_nx = 1600  # Target resolution
    
    # Try to load bedrock configuration
    bedrock_config = None
    if HAS_BEDROCK_CONFIG and profile_id is not None:
        try:
            bedrock_config = SyntheticBedrockModelConfig(profile_id=profile_id)
            print(f"Loaded bedrock config for profile {profile_id}")
        except Exception as e:
            print(f"Warning: Could not load bedrock config for profile {profile_id}: {e}")
    
    # Calculate optimal domain length
    if bedrock_config is not None:
        try:
            L = find_optimal_domain_length(bedrock_config, target_L)
            print(f"Using optimal domain length: {L/1000:.3f} km")
        except:
            L = target_L
            print(f"Warning: Could not calculate optimal domain length, using target: {target_L/1000:.0f}km")
    else:
        L = target_L
        print(f"Using target domain length: {target_L/1000:.0f}km")
    
    # Adjust nx proportionally to maintain resolution (matching flowline.py line 675)
    nx = int(target_nx * L / target_L)
    
    # Create 1D profile coordinates
    x_coords = np.linspace(0, L, nx)
    
    print(f"Reconstructed mesh: {nx} points from 0 to {L/1000:.3f}km")
    return x_coords

def extract_profile_id_from_filename(filename):
    """Extract profile ID from NetCDF filename"""
    # Try multiple patterns:
    # 1. XXX_SY_Z.Z_final_time=*_timestep=* (original format)
    # 2. XXX_SY_Z.Z (simplified format)
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
            profile_id = int(match.group(1))
            experiment = match.group(2)
            resolution_factor = float(match.group(3))
            return profile_id, experiment, resolution_factor
    
    return None, None, None

def debug_loading_process():
    """Debug step-by-step what happens in the NetCDF loading process"""
    
    filename = "S4/022_S4_0.5.nc"
    print(f"=== Debugging NetCDF loading process for {filename} ===")
    
    # Load exactly as the main script does
    dataset = nc.Dataset(filename, 'r')
    
    # Load time and velocity data
    time = np.array(dataset.groups['results'].groups['TransientSolution'].variables['time'][:])
    vx_raw = np.array(dataset.groups['results'].groups['TransientSolution'].variables['Vx'][:])
    vy_raw = np.array(dataset.groups['results'].groups['TransientSolution'].variables['Vy'][:])
    
    print(f"Time shape: {time.shape}, values: {time}")
    print(f"Vx raw shape: {vx_raw.shape}")
    print(f"Vy raw shape: {vy_raw.shape}")
    
    # Flatten spatial dimension as the main script does
    vx = vx_raw.squeeze(axis=-1) if vx_raw.ndim == 3 else vx_raw
    vy = vy_raw.squeeze(axis=-1) if vy_raw.ndim == 3 else vy_raw
    
    print(f"Vx shape after squeeze: {vx.shape}")
    print(f"Vy shape after squeeze: {vy.shape}")
    
    # Extract final timestep data
    final_idx = -1
    vx_final = vx[final_idx, :]
    vy_final = vy[final_idx, :]
    
    # Extract profile ID and reconstruct proper coordinates
    n_points = len(vx_final)
    profile_id, experiment, detected_res = extract_profile_id_from_filename(filename)
    
    if profile_id is not None:
        print(f"Detected: Profile {profile_id}, Experiment {experiment}, Resolution {detected_res}")
        # Reconstruct coordinates using the same method as flowline.py
        x_coords = reconstruct_mesh_coordinates(0.5, profile_id)  # Use detected resolution
        
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
        
        # Convert to normalized coordinates for comparison with static analysis
        x_hat = x_coords / np.max(x_coords)  # Normalize to 0-1 range
    else:
        print(f"Warning: Could not extract profile ID from {filename}, using fallback coordinates")
        x_hat = np.arange(n_points, dtype=float) / n_points  # Fallback: 0 to 1 normalized
    
    print(f"Final timestep data:")
    print(f"  Vx final range: [{np.min(vx_final):.2e}, {np.max(vx_final):.2e}]")
    print(f"  Vy final range: [{np.min(vy_final):.2e}, {np.max(vy_final):.2e}]")
    print(f"  Non-zero Vx: {np.count_nonzero(vx_final)}/{len(vx_final)}")
    print(f"  Non-zero Vy: {np.count_nonzero(vy_final)}/{len(vy_final)}")
    
    print(f"Placeholder coordinate range: [{np.min(x_hat):.6f}, {np.max(x_hat):.6f}]")
    print(f"Placeholder coords first 5: {x_hat[:5]}")
    print(f"Placeholder coords last 5: {x_hat[-5:]}")
    
    # Check monotonicity of placeholder coordinates (should be monotonic increasing)
    is_monotonic_increasing = np.all(np.diff(x_hat) >= 0)
    print(f"Placeholder coordinates monotonic increasing: {is_monotonic_increasing}")
    
    # Test sorting by decreasing coordinate (as would be done for real coordinates)
    sort_indices = np.argsort(-x_hat)  # Sort by decreasing x_hat
    x_hat_sorted = x_hat[sort_indices]
    vx_final_sorted = vx_final[sort_indices]
    vy_final_sorted = vy_final[sort_indices]
    
    # Check if sorting was needed
    was_already_sorted = np.array_equal(sort_indices, np.arange(len(x_hat)))
    print(f"Was already sorted (desc): {was_already_sorted}")
    
    if not was_already_sorted:
        print(f"Applied sorting to {len(x_hat)} points")
        print(f"Sort indices first 10: {sort_indices[:10]}")
        print(f"Sort indices last 10: {sort_indices[-10:]}")
    
    print(f"Sorted coords range: [{np.min(x_hat_sorted):.6f}, {np.max(x_hat_sorted):.6f}]")
    print(f"Sorted coords first 5: {x_hat_sorted[:5]}")
    print(f"Sorted coords last 5: {x_hat_sorted[-5:]}")
    
    # Check monotonicity after sorting (should be decreasing)
    is_monotonic_decreasing = np.all(np.diff(x_hat_sorted) <= 0)
    print(f"Monotonic decreasing after sorting: {is_monotonic_decreasing}")
    
    # Show velocity statistics after sorting
    print(f"Velocity statistics after sorting:")
    print(f"  Vx sorted range: [{np.min(vx_final_sorted):.2e}, {np.max(vx_final_sorted):.2e}]")
    print(f"  Vy sorted range: [{np.min(vy_final_sorted):.2e}, {np.max(vy_final_sorted):.2e}]")
    
    dataset.close()

if __name__ == "__main__":
    debug_loading_process()