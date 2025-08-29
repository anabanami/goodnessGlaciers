#!/usr/bin/env python3
"""
Debug script to investigate velocity data extraction issues
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

def debug_velocity_data():
    """Debug velocity data extraction from current NetCDF format (without mesh groups)"""
    
    resolutions = [0.5, 0.75, 1.0, 1.5]
    results = {}
    
    for res_factor in resolutions:
        # Current NetCDF file format
        nc_file = f"S4/022_S4_{res_factor}.nc"
        
        print(f"\n=== Resolution {res_factor} ===")
        print(f"File: {nc_file}")
        
        try:
            dataset = nc.Dataset(nc_file, 'r')
            
            # Check file structure
            print(f"Groups: {list(dataset.groups.keys())}")
            if 'results' in dataset.groups:
                print(f"Results subgroups: {list(dataset.groups['results'].groups.keys())}")
                if 'TransientSolution' in dataset.groups['results'].groups:
                    ts_group = dataset.groups['results'].groups['TransientSolution']
                    print(f"TransientSolution variables: {list(ts_group.variables.keys())}")
            
            # Load time and velocity data
            time = np.array(dataset.groups['results'].groups['TransientSolution'].variables['time'][:])
            vx_raw = np.array(dataset.groups['results'].groups['TransientSolution'].variables['Vx'][:])
            vy_raw = np.array(dataset.groups['results'].groups['TransientSolution'].variables['Vy'][:])
            
            print(f"Time array: {time}")
            print(f"Vx shape: {vx_raw.shape}")
            print(f"Vy shape: {vy_raw.shape}")
            
            # Flatten spatial dimension if needed
            vx = vx_raw.squeeze(axis=-1) if vx_raw.ndim == 3 else vx_raw
            vy = vy_raw.squeeze(axis=-1) if vy_raw.ndim == 3 else vy_raw
            
            print(f"Vx shape after squeeze: {vx.shape}")
            print(f"Vy shape after squeeze: {vy.shape}")
            
            # Extract final time step
            final_idx = -1
            vx_final = vx[final_idx, :]
            vy_final = vy[final_idx, :]
            
            print(f"Final time index: {final_idx}, time value: {time[final_idx]}")
            print(f"Final Vx shape: {vx_final.shape}")
            
            # Velocity statistics
            print(f"\nVelocity Statistics (final time):")
            print(f"  Vx: min={np.min(vx_final):.2e}, max={np.max(vx_final):.2e}, mean={np.mean(vx_final):.2e}")
            print(f"  Vy: min={np.min(vy_final):.2e}, max={np.max(vy_final):.2e}, mean={np.mean(vy_final):.2e}")
            print(f"  |V|: min={np.min(np.sqrt(vx_final**2 + vy_final**2)):.2e}, max={np.max(np.sqrt(vx_final**2 + vy_final**2)):.2e}")
            
            # Check for non-zero values
            nonzero_vx = np.count_nonzero(vx_final)
            nonzero_vy = np.count_nonzero(vy_final)
            print(f"  Non-zero Vx: {nonzero_vx}/{len(vx_final)} ({nonzero_vx/len(vx_final)*100:.1f}%)")
            print(f"  Non-zero Vy: {nonzero_vy}/{len(vy_final)} ({nonzero_vy/len(vy_final)*100:.1f}%)")
            
            # Check time evolution
            if len(time) > 1:
                print(f"\nTime Evolution:")
                max_vx_evolution = [np.max(np.abs(vx[i, :])) for i in range(len(time))]
                max_vy_evolution = [np.max(np.abs(vy[i, :])) for i in range(len(time))]
                print(f"  Max |Vx| over time: {max_vx_evolution}")
                print(f"  Max |Vy| over time: {max_vy_evolution}")
            
            # Extract profile ID from filename and reconstruct proper coordinates
            n_points = len(vx_final)
            profile_id, experiment, detected_res = extract_profile_id_from_filename(nc_file)
            
            if profile_id is not None:
                print(f"Detected: Profile {profile_id}, Experiment {experiment}, Resolution {detected_res}")
                # Reconstruct coordinates using the same method as flowline.py
                x_coords = reconstruct_mesh_coordinates(res_factor, profile_id)
                
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
            else:
                print(f"Warning: Could not extract profile ID from {nc_file}, using fallback coordinates")
                x_coords = np.linspace(0, 160000, n_points)  # Fallback: 0 to 160km
            
            results[res_factor] = {
                'x': x_coords,
                'vx': vx_final,
                'vy': vy_final,
                'n_points': n_points,
                'time_final': time[final_idx]
            }
            
            dataset.close()
            
        except Exception as e:
            print(f"Error loading {nc_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare velocities between resolutions
    if len(results) > 1:
        print(f"\n=== Velocity Comparison ===")
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot Vx
        for res_factor in sorted(results.keys()):
            result = results[res_factor]
            ax1.plot(result['x']/1000, result['vx'], 
                    'o-', label=f'res={res_factor}', alpha=0.7, markersize=2)
        
        ax1.set_xlabel('Distance (km)')
        ax1.set_ylabel('Vx (m/a)')
        ax1.set_title('Vx Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot Vy
        for res_factor in sorted(results.keys()):
            result = results[res_factor]
            ax2.plot(result['x']/1000, result['vy'], 
                    'o-', label=f'res={res_factor}', alpha=0.7, markersize=2)
        
        ax2.set_xlabel('Distance (km)')
        ax2.set_ylabel('Vy (m/a)')
        ax2.set_title('Vy Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('debug_velocity_comparison.png', dpi=150, bbox_inches='tight')
        print(f"Saved plot: debug_velocity_comparison.png")
        plt.show()
        
        # Quick interpolation test
        print(f"\n=== Quick Interpolation Test ===")
        ref_res = min(results.keys())
        ref_result = results[ref_res]
        x_ref = ref_result['x']
        
        for res_factor in results.keys():
            if res_factor == ref_res:
                continue
                
            result = results[res_factor]
            
            # Simple interpolation test for Vx
            vx_interp = np.interp(x_ref, result['x'], result['vx'])
            
            # Calculate difference
            diff = ref_result['vx'] - vx_interp
            ref_norm = np.linalg.norm(ref_result['vx'])
            l2_error = np.linalg.norm(diff) / ref_norm if ref_norm > 0 else float('nan')
            
            print(f"  Vx L2 error (res {ref_res} vs {res_factor}): {l2_error*100:.2f}%")
            print(f"  Max difference: {np.max(np.abs(diff)):.2e} m/a")
            print(f"  Mean difference: {np.mean(np.abs(diff)):.2e} m/a")

if __name__ == "__main__":
    debug_velocity_data()