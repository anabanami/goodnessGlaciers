#!/usr/bin/env python3
"""
Transient Grid Independence Study Analysis Script

Compares final steady states from transient ice flow simulations (.nc files)
across different mesh resolutions to determine optimal computational settings.

Usage:
    python analyse_transient_convergence.py --auto                          # Auto-detect everything
    python analyse_transient_convergence.py --auto --final_time_only        # Auto-detect, final states only
    python analyse_transient_convergence.py                                 # Auto-detect if no profile/experiment specified
    python analyse_transient_convergence.py --profile_id XXX --experiment SX # Manual specification
    python analyse_transient_convergence.py --profile_id XXX --experiment SX --final_time_only
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
from pathlib import Path
import sys
import re

# Additional imports for mesh reconstruction
try:
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Install with: pip install scipy")

# Import bedrock configuration - needed for proper mesh reconstruction
try:
    # Try to import from parent directories (flowline.py level)
    import sys
    sys.path.append('../../')  # Add flowline.py directory to path
    from bedrock_generator import SyntheticBedrockModelConfig
    HAS_BEDROCK_CONFIG = True
except ImportError:
    HAS_BEDROCK_CONFIG = False
    print("Warning: bedrock_generator not available. Mesh reconstruction will use simplified domain calculation.")

# Try to import ISSM/bamg functionality
try:
    from bamgflowband import bamgflowband
    HAS_BAMG = True
except ImportError:
    HAS_BAMG = False
    print("Warning: bamgflowband not available. Mesh reconstruction will be limited.")

# Check for netCDF4 availability
try:
    import netCDF4 as nc
    HAS_NETCDF = True
except ImportError:
    HAS_NETCDF = False
    print("Warning: netCDF4 not available. Install with: pip install netCDF4")

class TransientConvergenceAnalyzer:
    """Analyzes grid convergence for transient ice flow simulations"""
    
    def __init__(self, profile_id=None, experiment=None):
        self.profile_id = profile_id
        self.experiment = experiment
        self.results = {}
        self.convergence_metrics = {}
        self.time_series = {}
    
    def reconstruct_mesh_coordinates(self, resolution_factor, config=None):
        """Reconstruct mesh coordinates for a given resolution factor"""
        print(f"  Reconstructing mesh coordinates for resolution factor {resolution_factor}")
        
        if not HAS_SCIPY:
            print(f"  Warning: scipy not available, using simplified reconstruction")
            return self._simple_mesh_reconstruction(resolution_factor, config)
        
        if not HAS_BAMG:
            print(f"  Warning: bamgflowband not available, using simplified reconstruction")
            return self._simple_mesh_reconstruction(resolution_factor, config)
        
        if config is None:
            print(f"  Warning: No config provided, using simplified reconstruction")
            return self._simple_mesh_reconstruction(resolution_factor, config)
        
        try:
            return self._full_mesh_reconstruction(resolution_factor, config)
        except Exception as e:
            print(f"  Warning: Full mesh reconstruction failed ({e}), using simplified reconstruction")
            return self._simple_mesh_reconstruction(resolution_factor, config)
    
    def _find_optimal_domain_length(self, bedrock_config, target_L=160e3, search_window=5e3):
        """
        Find domain length that guarantees an integer number of bedrock periods
        and ends at/near a bedrock peak for optimal boundary conditions.
        Matches find_optimal_domain_length() from flowline.py exactly.
        """
        if not HAS_SCIPY:
            print(f"  Warning: scipy not available for optimal domain calculation")
            return target_L
            
        wavelength = bedrock_config.profile_params['wavelength']
        
        # Step 1: Find the closest integer number of periods to target
        target_periods = target_L / wavelength
        n_periods_candidates = [int(np.floor(target_periods)), int(np.ceil(target_periods))]
        
        print(f"  Domain optimization for integer periods:")
        print(f"    Target L: {target_L/1000:.1f} km")
        print(f"    Wavelength: {wavelength/1000:.1f} km")
        print(f"    Target periods: {target_periods:.2f}")
        print(f"    Candidate periods: {n_periods_candidates}")
        
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
            
            print(f"    n={n_periods}: L={peak_L/1000:.3f} km, periods={peak_L/wavelength:.4f}, score={score:.1f}")
            
            if score < best_score:
                best_score = score
                best_L = peak_L
                best_n_periods = n_periods
        
        if best_L is None:
            # Fallback: use exact integer periods closest to target
            n_periods = round(target_periods)
            best_L = n_periods * wavelength
            best_n_periods = n_periods
            print(f"    Fallback: Using exact {n_periods} periods")
        
        # Final validation
        actual_periods = best_L / wavelength
        period_error = abs(actual_periods - best_n_periods)
        
        print(f"  Final domain configuration:")
        print(f"    Optimal L: {best_L/1000:.3f} km")
        print(f"    Integer periods: {best_n_periods}")
        print(f"    Actual periods: {actual_periods:.6f}")
        print(f"    Period error: {period_error:.6f}")
        print(f"    Adjustment from target: {(best_L - target_L)/1000:.3f} km")
        
        # Ensure period error is negligible (< 0.1% of a period)
        if period_error > 0.001:
            print(f"    WARNING: Period error {period_error:.6f} exceeds tolerance!")
        else:
            print(f"    ✓ Period precision verified")
        
        return best_L

    def _simple_mesh_reconstruction(self, resolution_factor, config=None):
        """Simple mesh reconstruction without dependencies"""
        # Domain parameters - must match flowline.py exactly
        target_L = 160e3  # Target domain length (m)
        target_nx = 1600  # Target resolution
        
        # If config is available, use the optimal domain length calculation from flowline.py
        if config is not None:
            try:
                L = self._find_optimal_domain_length(config, target_L)
            except:
                L = target_L
                print(f"  Warning: Could not calculate optimal domain length, using target: {target_L/1000:.0f}km")
        else:
            L = target_L
        
        # Adjust nx proportionally to maintain resolution (matching flowline.py line 675)
        nx = int(target_nx * L / target_L)
        
        # Create 1D profile coordinates
        x_coords = np.linspace(0, L, nx)
        
        print(f"  Simple reconstruction: {nx} points from 0 to {L/1000:.3f}km")
        return x_coords
    
    def _full_mesh_reconstruction(self, resolution_factor, config):
        """Full mesh reconstruction matching flowline.py exactly"""
        # Domain parameters - must match flowline.py exactly
        target_L = 160e3  # Target domain length (m)
        target_nx = 1600  # Target resolution
        
        # Use the exact optimal domain length calculation from flowline.py
        L = self._find_optimal_domain_length(config, target_L)
        
        # Adjust nx proportionally to maintain resolution (matching flowline.py line 675)
        nx = int(target_nx * L / target_L)
        
        # Create 1D profile
        x_1D = np.linspace(0, L, nx)
        b0 = config.get_bedrock_elevation(x_1D)
        s0 = b0 + config.ice_thickness
        
        # Get bedrock wavelength from config for adaptive meshing
        bed_wavelength = config.profile_params['wavelength']
        ice_thickness = config.ice_thickness
        
        # Match exactly the adaptive_bamg function from flowline.py
        wavelength_thickness_ratio = bed_wavelength / ice_thickness  # unitless
        
        if bed_wavelength < 15000:
            refinement_factor = 50
        else:
            refinement_factor = 200

        hmax = (bed_wavelength / refinement_factor) * resolution_factor
        
        # Use adaptive bamg meshing with exact parameters from flowline.py  
        md = bamgflowband(None, x_1D, s0, b0,
                          'hmax', hmax,
                          'anisomax', 3,
                          'vertical', 1)
        
        x_coords = md.mesh.x  # Keep in meters
        print(f"  Full reconstruction: {len(x_coords)} points from 0 to {L/1000:.1f}km")
        return x_coords
    
    def detect_available_datasets(self):
        """Automatically detect profile ID, experiment, and resolution factors from NetCDF files"""
        # Search for NetCDF files in root directory and experiment folders (S1/, S2/, S3/, S4/)
        patterns = [
            "*.nc",       # Current directory
            "S1/*.nc",
            "S2/*.nc", 
            "S3/*.nc",
            "S4/*.nc"
        ]
        
        all_files = []
        for pattern in patterns:
            files = glob.glob(pattern, recursive=False)
            all_files.extend(files)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for f in all_files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)
        
        if not unique_files:
            raise FileNotFoundError("No NetCDF files found matching patterns")
        
        datasets = {}  # {(profile_id, experiment): [resolution_factors]}
        
        for filepath in unique_files:
            try:
                # Extract filename from path
                filename = os.path.basename(filepath)
                
                # Accept any .nc file and try to parse profile_experiment_resolution pattern
                if not filename.endswith('.nc'):
                    continue
                
                # Remove .nc extension
                basename = filename[:-3]
                
                # Try multiple patterns:
                # 1. XXX_SY_Z.Z_final_time=*_timestep=* (original format)
                # 2. XXX_SY_Z.Z (simplified format)
                patterns = [
                    r'^(\d+)_([A-Za-z]\d+)_([0-9.]+)_final_time=.*_timestep=',
                    r'^(\d+)_([A-Za-z]\d+)_([0-9.]+)$'
                ]
                
                match = None
                for pattern in patterns:
                    match = re.match(pattern, basename)
                    if match:
                        break
                
                if match:
                    profile_str = match.group(1)
                    experiment = match.group(2)
                    resolution_str = match.group(3)
                    
                    profile_id = int(profile_str)
                    resolution_factor = float(resolution_str)
                    
                    key = (profile_id, experiment)
                    if key not in datasets:
                        datasets[key] = []
                    datasets[key].append((resolution_factor, filepath))
                    
                else:
                    print(f"Warning: Could not parse NetCDF filename pattern: {filename}")
                    
            except (ValueError, AttributeError) as e:
                print(f"Warning: Could not parse components from {filename}: {e}")
        
        # Sort resolution factors for each dataset and extract just the factors
        for key in datasets:
            # Sort by resolution factor
            datasets[key].sort(key=lambda x: x[0])
            # Store both factors and file paths
            resolution_data = datasets[key]
            datasets[key] = {
                'resolution_factors': [item[0] for item in resolution_data],
                'file_paths': {item[0]: item[1] for item in resolution_data}
            }
        
        print(f"Auto-detected transient datasets:")
        for (profile_id, experiment), data in datasets.items():
            resolutions = data['resolution_factors']
            print(f"  Profile {profile_id:03d} - Experiment {experiment}: {resolutions}")
            for res_factor in resolutions:
                print(f"    res_factor={res_factor}: {data['file_paths'][res_factor]}")
        
        return datasets
    
    def auto_configure(self):
        """Automatically configure analyser with detected datasets"""
        datasets = self.detect_available_datasets()
        
        if len(datasets) == 0:
            raise FileNotFoundError("No valid transient datasets found")
        elif len(datasets) == 1:
            # Single dataset - use it automatically
            (self.profile_id, self.experiment), data = next(iter(datasets.items()))
            print(f"Auto-selected: Profile {self.profile_id:03d}, Experiment {self.experiment}")
            return data['resolution_factors'], data['file_paths']
        else:
            # Multiple datasets - show options
            print(f"\nMultiple datasets found:")
            dataset_list = list(datasets.items())
            for i, ((profile_id, experiment), data) in enumerate(dataset_list):
                resolutions = data['resolution_factors']
                print(f"  [{i}] Profile {profile_id:03d} - Experiment {experiment}: {resolutions}")
            
            # For now, use the first one (could be enhanced with user input)
            (self.profile_id, self.experiment), data = dataset_list[0]
            print(f"Auto-selected first dataset: Profile {self.profile_id:03d}, Experiment {self.experiment}")
            return data['resolution_factors'], data['file_paths']
        
    def convert_velocity_units(self, var, data):
        """Convert velocity from m/s to m/year if needed."""
        units = getattr(var, 'units', '').lower()
        if ('m/s' in units) or ('m s-1' in units) or ('m s^-1' in units) or units == '':
            converted = data * 31556926  # Convert m/s to m/year
            # Clean up any extreme values that might be from NaN/inf conversion
            converted = np.where(np.abs(converted) > 1e6, 0, converted)  # Cap at 1000 km/year
            return converted
        # Already in m/year or similar
        return data
    
    def convert_time_units(self, var, data):
        """Convert time from seconds to years if needed."""
        units = getattr(var, 'units', '').lower()
        if 'sec' in units or units == '':  # Default to seconds if no units
            return data / 31556926  # Convert seconds to years
        elif 'yr' in units or 'year' in units:
            return data  # Already in years
        else:
            return data / 31556926  # Safe fallback to seconds

    def load_transient_results(self, resolution_factors=None, file_paths=None, final_time_only=True):
        """Load results from transient simulation NetCDF files (without mesh groups)"""
        print(f"\n=== Loading Transient Results for Profile {self.profile_id:03d} {self.experiment} ===")
        
        if not HAS_NETCDF:
            raise ImportError("netCDF4 package is required. Install with: pip install netCDF4")
        
        # Use provided resolution factors or default
        if resolution_factors is None:
            resolution_factors = [0.5, 0.75, 1.0]
        
        for res_factor in resolution_factors:
            nc_file = None
            
            # If file_paths provided (from auto-detection), use them directly
            if file_paths and res_factor in file_paths:
                nc_file = file_paths[res_factor]
            else:
                # Fallback to original search patterns
                patterns = [
                    f"{res_factor}/{self.profile_id:03d}_{self.experiment}_{res_factor}_final_time=*_yrs_timestep=*_yrs.nc",
                    f"{self.profile_id:03d}_{self.experiment}_{res_factor}_final_time=*_yrs_timestep=*_yrs.nc",
                    f"*{self.profile_id:03d}*{self.experiment}*{res_factor}*.nc"
                ]
                
                for pattern in patterns:
                    matching_files = glob.glob(pattern)
                    if matching_files:
                        nc_file = matching_files[0]
                        break
            
            if nc_file:
                print(f"Loading {nc_file} for resolution factor {res_factor}")
                try:
                    dataset = nc.Dataset(nc_file, 'r')
                    
                    # Load time series from TransientSolution group (structure: results/TransientSolution/)
                    time_var = dataset.groups['results'].groups['TransientSolution'].variables['time']
                    time_raw = np.array(time_var[:])
                    time = self.convert_time_units(time_var, time_raw)
                    print(f"  Time conversion: {time_raw[0]:.2e} -> {time[0]:.3f} years (first timestep)")
                    
                    # Load velocity fields - shape is (time, spatial_points, 1)
                    vx_var = dataset.groups['results'].groups['TransientSolution'].variables['Vx']
                    vy_var = dataset.groups['results'].groups['TransientSolution'].variables['Vy']
                    vx_raw_data = np.array(vx_var[:])
                    vy_raw_data = np.array(vy_var[:])
                    
                    # Convert velocity units
                    vx_raw = self.convert_velocity_units(vx_var, vx_raw_data)
                    vy_raw = self.convert_velocity_units(vy_var, vy_raw_data)
                    print(f"  Velocity conversion: {vx_raw_data.max():.2e} -> {vx_raw.max():.3f} m/year (max vx)")
                    
                    # Flatten spatial dimension: (time, spatial_points, 1) -> (time, spatial_points)
                    vx = vx_raw.squeeze(axis=-1) if vx_raw.ndim == 3 else vx_raw
                    vy = vy_raw.squeeze(axis=-1) if vy_raw.ndim == 3 else vy_raw
                    
                    n_points = vx.shape[1]  # Number of spatial points
                    n_timesteps = len(time)
                    
                    print(f"  Note: NetCDF contains {n_points} spatial points, {n_timesteps} timesteps")
                    
                    # Create bedrock configuration if available
                    bedrock_config = None
                    if HAS_BEDROCK_CONFIG and self.profile_id is not None:
                        try:
                            bedrock_config = SyntheticBedrockModelConfig(profile_id=self.profile_id)
                            print(f"  Loaded bedrock config for profile {self.profile_id}")
                        except Exception as e:
                            print(f"  Warning: Could not load bedrock config for profile {self.profile_id}: {e}")
                    
                    # Reconstruct mesh coordinates for this resolution factor
                    x_coords = self.reconstruct_mesh_coordinates(res_factor, bedrock_config)
                    
                    # Check if the number of points matches
                    if len(x_coords) != n_points:
                        print(f"  Warning: Mesh reconstruction gave {len(x_coords)} points, but NetCDF has {n_points} points")
                        print(f"  Using interpolation to match NetCDF data structure")
                        
                        # Interpolate mesh coordinates to match NetCDF data points
                        if len(x_coords) > n_points:
                            # Downsample mesh coordinates
                            indices = np.linspace(0, len(x_coords)-1, n_points, dtype=int)
                            x_coords = x_coords[indices]
                        else:
                            # Upsample mesh coordinates via interpolation
                            x_old = np.linspace(0, np.max(x_coords), len(x_coords))
                            x_new = np.linspace(0, np.max(x_coords), n_points)
                            x_coords = np.interp(x_new, x_old, x_coords)
                    
                    print(f"  Using reconstructed coordinates: {len(x_coords)} points from {x_coords[0]/1000:.1f} to {x_coords[-1]/1000:.1f}km")
                    
                    # Use the same data for both surface and basal (will be refined later)
                    vx_surface = vx  # Shape: (time, points)
                    vx_basal = vx    # Shape: (time, points)
                    vz_surface = np.zeros_like(vx_surface)  # Initialize vz (2D flowline typically has vz=0)
                    
                    if final_time_only:
                        # Use only the final time step
                        final_idx = -1
                        self.results[res_factor] = {
                            'filename': nc_file,
                            'x': x_coords,
                            'x_surface': x_coords,
                            'x_basal': x_coords,
                            'time_final': time[final_idx],
                            'vx_surface': vx_surface[final_idx, :],
                            'vz_surface': vz_surface[final_idx, :],
                            'vx_basal': vx_basal[final_idx, :],
                            'n_points': n_points,
                            'n_points_total': n_points,
                            'n_surface_points': n_points,
                            'n_basal_points': n_points,
                            'n_timesteps': n_timesteps,
                            'simulation_time': time[-1] - time[0] if len(time) > 1 else 0,
                            'needs_mesh_reconstruction': False  # Coordinates already reconstructed
                        }
                        print(f"  ✓ Loaded final state: {n_points} points, t={time[final_idx]:.1f} time units")
                    else:
                        # Store full time series
                        self.results[res_factor] = {
                            'filename': nc_file,
                            'x': x_coords,
                            'x_surface': x_coords,
                            'x_basal': x_coords,
                            'time': time,
                            'vx_surface': vx_surface,
                            'vz_surface': vz_surface,
                            'vx_basal': vx_basal,
                            'n_points': n_points,
                            'n_points_total': n_points,
                            'n_surface_points': n_points,
                            'n_basal_points': n_points,
                            'n_timesteps': n_timesteps,
                            'simulation_time': time[-1] - time[0] if len(time) > 1 else 0,
                            'needs_mesh_reconstruction': False  # Coordinates already reconstructed
                        }
                        self.time_series[res_factor] = {
                            'time': time,
                            'max_vx_surface': np.max(np.abs(vx_surface), axis=1),
                            'max_vx_basal': np.max(np.abs(vx_basal), axis=1)
                        }
                        print(f"  ✓ Loaded time series: {n_points} points, {n_timesteps} timesteps, t={time[-1]:.1f} time units")
                    
                    dataset.close()
                    
                except Exception as e:
                    print(f"  ✗ Error loading {nc_file}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"  ✗ No NetCDF results found for resolution factor {res_factor}")
                if 'patterns' in locals():
                    print(f"    Searched patterns: {patterns}")
        
        if not self.results:
            raise FileNotFoundError(f"No transient results found for profile {self.profile_id} experiment {self.experiment}")
        
        print(f"Successfully loaded {len(self.results)} resolution datasets")
        print(f"Note: Mesh reconstruction will be needed for spatial analysis")
        return self.results
    
    def validate_transient_data_quality(self):
        """Validate transient simulation data quality and consistency"""
        print(f"\n=== Transient Data Quality Validation ===")
        
        issues = []
        
        for res_factor, result in self.results.items():
            x_data = result['x']
            vx_surf = result['vx_surface']
            vz_surf = result['vz_surface']
            vx_basal = result['vx_basal']
            
            # Check coordinate range and ordering
            x_min, x_max = np.min(x_data), np.max(x_data)
            print(f"  Resolution {res_factor}: x range [{x_min:.1f}, {x_max:.1f}], {len(x_data)} points")
            print(f"    Simulation time: {result['simulation_time']:.1f} years, {result['n_timesteps']} timesteps")
            
            # Check for NaN or infinite values in final state
            for var_name, var_data in [('vx_surface', vx_surf), ('vz_surface', vz_surf), ('vx_basal', vx_basal)]:
                nan_count = np.sum(np.isnan(var_data))
                inf_count = np.sum(np.isinf(var_data))
                if nan_count > 0:
                    issues.append(f"Resolution {res_factor}, {var_name}: {nan_count} NaN values")
                if inf_count > 0:
                    issues.append(f"Resolution {res_factor}, {var_name}: {inf_count} infinite values")
            
            # Check velocity magnitudes
            max_vx_surf = np.max(np.abs(vx_surf))
            max_vx_basal = np.max(np.abs(vx_basal))
            print(f"    Final velocities: max |vx_surface| = {max_vx_surf:.2e} m/a, max |vx_basal| = {max_vx_basal:.2e} m/a")
            
            # Check if simulation reached steady state (if time series available)
            if res_factor in self.time_series:
                ts = self.time_series[res_factor]
                final_velocity_change = np.abs(ts['max_vx_surface'][-1] - ts['max_vx_surface'][-10]) if len(ts['max_vx_surface']) > 10 else 0
                velocity_trend = (ts['max_vx_surface'][-1] - ts['max_vx_surface'][0]) / len(ts['max_vx_surface'])
                print(f"    Steady state check: final velocity change = {final_velocity_change:.3f} m/a")
                if final_velocity_change > 0.1:  # Arbitrary threshold
                    issues.append(f"Resolution {res_factor}: May not have reached steady state (velocity still changing)")
        
        # Check coordinate overlap between resolutions
        if len(self.results) > 1:
            x_mins = [np.min(result['x']) for result in self.results.values()]
            x_maxs = [np.max(result['x']) for result in self.results.values()]
            
            overlap_min = np.max(x_mins)
            overlap_max = np.min(x_maxs)
            
            if overlap_min >= overlap_max:
                issues.append(f"No coordinate overlap between resolutions")
            else:
                overlap_fraction = (overlap_max - overlap_min) / (np.max(x_maxs) - np.min(x_mins))
                print(f"  Coordinate overlap: [{overlap_min:.1f}, {overlap_max:.1f}] ({overlap_fraction*100:.1f}% of total range)")
        
        if issues:
            print(f"  ⚠️  Found {len(issues)} data quality issues:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"  ✓ All data quality checks passed")
        
        return issues
    
    def interpolate_to_common_grid(self):
        """Interpolate all results to a common spatial grid for comparison"""
        if len(self.results) < 2:
            print("Need at least 2 resolution datasets for comparison")
            return
        
        # Find the common domain (intersection of all coordinate ranges)
        x_mins = []
        x_maxs = []
        for res_factor, result in self.results.items():
            x_data = result['x']
            x_mins.append(np.min(x_data))
            x_maxs.append(np.max(x_data))
        
        # Use conservative bounds (intersection of all domains)
        x_min_common = np.max(x_mins)
        x_max_common = np.min(x_maxs)
        
        print(f"\nCommon domain: x = [{x_min_common:.1f}, {x_max_common:.1f}]")
        
        # Use finest resolution for reference grid density, but restrict to common domain
        finest_res = min(self.results.keys())
        x_finest = self.results[finest_res]['x']
        
        # Create reference grid within common domain
        mask_ref = (x_finest >= x_min_common) & (x_finest <= x_max_common)
        x_ref = x_finest[mask_ref]
        
        if len(x_ref) < 10:
            print(f"Warning: Very small overlap region ({len(x_ref)} points). Results may be unreliable.")
        
        print(f"Reference grid has {len(x_ref)} points in common domain")
        
        for res_factor in self.results:
            result = self.results[res_factor]
            x_orig = result['x']
            
            if res_factor == finest_res:
                # For finest resolution, just crop to common domain
                # Check if we have time series or final state only
                if 'time' in result:
                    # Time series - use final time step
                    result['vx_surface_interp'] = result['vx_surface'][-1, mask_ref]
                    result['vz_surface_interp'] = result['vz_surface'][-1, mask_ref]
                    result['vx_basal_interp'] = result['vx_basal'][-1, mask_ref]
                else:
                    # Final state only
                    result['vx_surface_interp'] = result['vx_surface'][mask_ref]
                    result['vz_surface_interp'] = result['vz_surface'][mask_ref]
                    result['vx_basal_interp'] = result['vx_basal'][mask_ref]
            else:
                # For other resolutions, interpolate within their valid range
                x_min_orig = np.min(x_orig)
                x_max_orig = np.max(x_orig)
                
                # Only interpolate where we have data
                mask_valid = (x_ref >= x_min_orig) & (x_ref <= x_max_orig)
                
                # Initialize with NaN
                vx_surf_interp = np.full(len(x_ref), np.nan)
                vz_surf_interp = np.full(len(x_ref), np.nan)
                vx_basal_interp = np.full(len(x_ref), np.nan)
                
                if np.any(mask_valid):
                    # Interpolate only valid points
                    # Check if we have time series or final state only
                    if 'time' in result:
                        # Full time series - use final time step
                        vx_surf_interp[mask_valid] = np.interp(x_ref[mask_valid], x_orig, result['vx_surface'][-1, :])
                        vz_surf_interp[mask_valid] = np.interp(x_ref[mask_valid], x_orig, result['vz_surface'][-1, :])
                        vx_basal_interp[mask_valid] = np.interp(x_ref[mask_valid], x_orig, result['vx_basal'][-1, :])
                    else:
                        # Final state only - data is already 1D arrays
                        vx_surf_interp[mask_valid] = np.interp(x_ref[mask_valid], x_orig, result['vx_surface'])
                        vz_surf_interp[mask_valid] = np.interp(x_ref[mask_valid], x_orig, result['vz_surface'])
                        vx_basal_interp[mask_valid] = np.interp(x_ref[mask_valid], x_orig, result['vx_basal'])
                
                result['vx_surface_interp'] = vx_surf_interp
                result['vz_surface_interp'] = vz_surf_interp
                result['vx_basal_interp'] = vx_basal_interp
                
                valid_points = np.sum(mask_valid)
                print(f"  Interpolated res_factor={res_factor}: {valid_points}/{len(x_ref)} points valid")
        
        # Store reference grid and common domain info
        self.x_ref = x_ref
        self.x_min_common = x_min_common
        self.x_max_common = x_max_common
        return x_ref
    
    def calculate_convergence_metrics(self):
        """Calculate convergence metrics between different resolutions"""
        if len(self.results) < 2:
            return
        
        print(f"\n=== Calculating Transient Convergence Metrics ===")
        
        # Use finest resolution as reference
        finest_res = min(self.results.keys())
        ref_result = self.results[finest_res]
        
        variables = ['vx_surface', 'vz_surface', 'vx_basal']
        
        for res_factor in self.results:
            if res_factor == finest_res:
                continue
            
            result = self.results[res_factor]
            metrics = {}
            
            for var in variables:
                # Get reference and comparison data (both interpolated to common grid)
                ref_data = ref_result[f'{var}_interp']
                comp_data = result[f'{var}_interp']
                
                # Remove NaN values (points outside common domain)
                valid_mask = ~(np.isnan(ref_data) | np.isnan(comp_data))
                
                if not np.any(valid_mask):
                    print(f"  {var} (res={res_factor}): No valid comparison points")
                    metrics[var] = {
                        'l2_relative_error': np.nan,
                        'max_relative_error': np.nan,
                        'rmse': np.nan,
                        'valid_points': 0
                    }
                    continue
                
                ref_valid = ref_data[valid_mask]
                comp_valid = comp_data[valid_mask]
                
                # Handle near-zero reference values
                ref_norm = np.linalg.norm(ref_valid)
                ref_max = np.max(np.abs(ref_valid))
                
                if ref_norm < 1e-10:
                    print(f"  {var} (res={res_factor}): Near-zero reference values")
                    l2_error = np.nan
                    max_error = np.nan
                else:
                    # Calculate L2 relative error
                    l2_error = np.linalg.norm(ref_valid - comp_valid) / ref_norm
                    
                    # Maximum pointwise relative error  
                    max_error = np.max(np.abs(ref_valid - comp_valid)) / ref_max
                
                # Root mean square error (absolute, not relative)
                rmse = np.sqrt(np.mean((ref_valid - comp_valid)**2))
                
                metrics[var] = {
                    'l2_relative_error': l2_error,
                    'max_relative_error': max_error,
                    'rmse': rmse,
                    'max_ref_value': ref_max,
                    'max_comp_value': np.max(np.abs(comp_valid)) if len(comp_valid) > 0 else np.nan,
                    'valid_points': np.sum(valid_mask)
                }
                
                print(f"  {var} (res={res_factor}):")
                print(f"    L2 relative error: {l2_error:.4f} ({l2_error*100:.2f}%)")
                print(f"    Max relative error: {max_error:.4f} ({max_error*100:.2f}%)")
                print(f"    RMSE: {rmse:.2f} m/a")
                print(f"    Valid points: {np.sum(valid_mask)}/{len(ref_data)}")
            
            self.convergence_metrics[res_factor] = metrics
        
        return self.convergence_metrics
    
    def assess_convergence(self, tolerance=0.01):
        """Assess whether transient solutions are converged within tolerance"""
        print(f"\n=== Transient Convergence Assessment (tolerance={tolerance*100:.1f}%) ===")
        
        convergence_summary = {}
        
        for res_factor, metrics in self.convergence_metrics.items():
            converged_vars = []
            
            for var, var_metrics in metrics.items():
                l2_error = var_metrics['l2_relative_error']
                
                # Handle NaN values
                if np.isnan(l2_error):
                    is_converged = False
                    status = "✗ NO DATA (NaN)"
                    error_str = "nan"
                else:
                    is_converged = l2_error < tolerance
                    status = "✓ CONVERGED" if is_converged else "✗ NOT CONVERGED"
                    error_str = f"{l2_error*100:.2f}"
                
                converged_vars.append(is_converged)
                print(f"  {var} (res={res_factor}): {error_str}% - {status}")
            
            overall_converged = all(converged_vars)
            convergence_summary[res_factor] = {
                'overall_converged': overall_converged,
                'converged_variables': sum(converged_vars),
                'total_variables': len(converged_vars)
            }
            
            status = "✓ CONVERGED" if overall_converged else "✗ NOT CONVERGED"
            print(f"  Overall (res={res_factor}): {status}")
        
        return convergence_summary
    
    def create_transient_comparison_plots(self, save_plots=True):
        """Create comprehensive comparison plots for transient results"""
        print(f"\n=== Creating Transient Comparison Plots ===")
        
        if not hasattr(self, 'x_ref'):
            self.interpolate_to_common_grid()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Transient Grid Convergence Study - Profile {self.profile_id:03d} {self.experiment}', fontsize=14)
        
        # Plot 1: Surface velocity comparison (final states)
        ax1 = axes[0, 0]
        for res_factor in sorted(self.results.keys()):
            result = self.results[res_factor]
            if res_factor == min(self.results.keys()):
                x_data = result['x']
                # Handle time series vs final state data
                if 'time' in result:
                    y_data = result['vx_surface'][-1, :]  # Final time step
                else:
                    y_data = result['vx_surface']  # Already final state
            else:
                x_data = self.x_ref
                y_data = result['vx_surface_interp']  # Already interpolated to final state
            
            ax1.plot(x_data/1000, y_data, label=f'res_factor={res_factor}', linewidth=2)
        
        ax1.set_xlabel('Distance (km)')
        ax1.set_ylabel('Surface velocity (m/a)')
        ax1.set_title('Final Surface Velocity Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Basal velocity comparison (final states)
        ax2 = axes[0, 1]
        for res_factor in sorted(self.results.keys()):
            result = self.results[res_factor]
            if res_factor == min(self.results.keys()):
                x_data = result['x']
                # Handle time series vs final state data
                if 'time' in result:
                    y_data = result['vx_basal'][-1, :]  # Final time step
                else:
                    y_data = result['vx_basal']  # Already final state
            else:
                x_data = self.x_ref
                y_data = result['vx_basal_interp']  # Already interpolated to final state
            
            ax2.plot(x_data/1000, y_data, label=f'res_factor={res_factor}', linewidth=2)
        
        ax2.set_xlabel('Distance (km)')
        ax2.set_ylabel('Basal velocity (m/a)')
        ax2.set_title('Final Basal Velocity Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Convergence metrics
        ax3 = axes[1, 0]
        if self.convergence_metrics:
            res_factors = []
            l2_errors_vx_surf = []
            l2_errors_vx_basal = []
            
            for res_factor in sorted(self.convergence_metrics.keys()):
                metrics = self.convergence_metrics[res_factor]
                res_factors.append(res_factor)
                
                vx_surf_error = metrics['vx_surface']['l2_relative_error']
                vx_basal_error = metrics['vx_basal']['l2_relative_error']
                
                l2_errors_vx_surf.append(vx_surf_error * 100 if not np.isnan(vx_surf_error) else 0)
                l2_errors_vx_basal.append(vx_basal_error * 100 if not np.isnan(vx_basal_error) else 0)
            
            x_pos = np.arange(len(res_factors))
            width = 0.35
            
            ax3.bar(x_pos - width/2, l2_errors_vx_surf, width, label='Surface vx', alpha=0.8)
            ax3.bar(x_pos + width/2, l2_errors_vx_basal, width, label='Basal vx', alpha=0.8)
            ax3.axhline(y=1.0, color='red', linestyle='--', label='1% tolerance')
            
            ax3.set_xlabel('Resolution Factor')
            ax3.set_ylabel('L2 Relative Error (%)')
            ax3.set_title('Transient Convergence Metrics')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels([f'{rf}' for rf in res_factors])
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Time evolution (if available) or computational scaling
        ax4 = axes[1, 1]
        if self.time_series:
            # Plot velocity evolution over time
            for res_factor in sorted(self.time_series.keys()):
                ts = self.time_series[res_factor]
                ax4.plot(ts['time'], ts['max_vx_surface'], label=f'res_factor={res_factor}', linewidth=2)
            
            ax4.set_xlabel('Time (years)')
            ax4.set_ylabel('Max Surface Velocity (m/a)')
            ax4.set_title('Velocity Evolution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            # Computational scaling plot (based on node counts and timesteps)
            res_factors = sorted(self.results.keys())
            node_counts = [self.results[rf]['n_points'] for rf in res_factors]
            timesteps = [self.results[rf]['n_timesteps'] for rf in res_factors]
            relative_costs = [n * t / (min(node_counts) * min(timesteps)) for n, t in zip(node_counts, timesteps)]
            
            ax4.semilogy(res_factors, relative_costs, 'o-', linewidth=2, markersize=8, color='steelblue')
            ax4.set_xlabel('Resolution Factor')
            ax4.set_ylabel('Relative Computational Cost')
            ax4.set_title('Computational Scaling')
            ax4.grid(True, alpha=0.3)
            ax4.invert_xaxis()  # Lower resolution factor = higher cost
        
        plt.tight_layout()
        
        if save_plots:
            filename = f"{self.profile_id:03d}_{self.experiment}_transient_convergence_analysis.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"  Saved plot: {filename}")
        
        plt.show()
        return fig
    
    def generate_transient_report(self):
        """Generate a comprehensive transient convergence analysis report"""
        print(f"\n=== Generating Transient Analysis Report ===")
        
        report_lines = []
        report_lines.append(f"# Transient Grid Independence Study Report")
        report_lines.append(f"## Profile {self.profile_id:03d} - Experiment {self.experiment}")
        report_lines.append(f"")
        
        # Results summary
        report_lines.append(f"## Results Summary")
        report_lines.append(f"- Loaded {len(self.results)} resolution datasets")
        for res_factor in sorted(self.results.keys()):
            result = self.results[res_factor]
            report_lines.append(f"  - Resolution factor {res_factor}: {result['n_points']} points, {result['simulation_time']:.1f} years")
        report_lines.append(f"")
        
        # Convergence analysis
        if self.convergence_metrics:
            report_lines.append(f"## Transient Convergence Analysis")
            convergence_summary = self.assess_convergence(tolerance=0.01)
            
            for res_factor in sorted(self.convergence_metrics.keys()):
                metrics = self.convergence_metrics[res_factor]
                summary = convergence_summary[res_factor]
                
                report_lines.append(f"### Resolution Factor {res_factor}")
                report_lines.append(f"- Overall convergence: {'✓ PASSED' if summary['overall_converged'] else '✗ FAILED'}")
                
                for var in ['vx_surface', 'vx_basal', 'vz_surface']:
                    if var in metrics:
                        l2_error = metrics[var]['l2_relative_error']
                        max_error = metrics[var]['max_relative_error']
                        if not np.isnan(l2_error):
                            report_lines.append(f"- {var}: L2={l2_error*100:.2f}%, Max={max_error*100:.2f}%")
                        else:
                            report_lines.append(f"- {var}: No valid data (NaN)")
                report_lines.append(f"")
        
        # Recommendations
        report_lines.append(f"## Recommendations")
        if self.convergence_metrics:
            # Find most efficient converged solution
            converged_solutions = []
            for res_factor, summary in convergence_summary.items():
                if summary['overall_converged']:
                    # Simple cost estimate based on points and simulation time
                    result = self.results[res_factor]
                    cost = result['n_points'] * result['n_timesteps'] / 1000  # Relative cost
                    converged_solutions.append((res_factor, cost))
            
            if converged_solutions:
                # Sort by cost (ascending)
                converged_solutions.sort(key=lambda x: x[1])
                optimal_res, optimal_cost = converged_solutions[0]
                report_lines.append(f"- **Optimal resolution factor**: {optimal_res} (relative cost: {optimal_cost:.2f})")
                report_lines.append(f"- Based on transient simulation convergence to steady state")
            else:
                report_lines.append(f"- **No fully converged solutions found** - consider finer resolution or longer simulation time")
        
        # Save report
        report_filename = f"{self.profile_id:03d}_{self.experiment}_transient_convergence_report.md"
        with open(report_filename, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"  Saved report: {report_filename}")
        return report_lines

def main():
    parser = argparse.ArgumentParser(description='Analyze transient grid convergence for ice flow simulations')
    parser.add_argument('--profile_id', type=int, 
                       help='Bedrock profile ID (if not provided, auto-detect)')
    parser.add_argument('--experiment', choices=['S1', 'S2', 'S3', 'S4'],
                       help='Experiment type (if not provided, auto-detect)')
    parser.add_argument('--resolution_factors', nargs='+', type=float,
                       help='Resolution factors to analyse (if not provided, use all detected)')
    parser.add_argument('--tolerance', type=float, default=0.01,
                       help='Convergence tolerance (default: 0.01 = 1%)')
    parser.add_argument('--final_time_only', action='store_true',
                       help='Only analyse final time states, not full time series')
    parser.add_argument('--plot_only', action='store_true',
                       help='Only create plots, skip analysis')
    parser.add_argument('--auto', action='store_true',
                       help='Automatically detect profile, experiment, and resolution factors')
    
    args = parser.parse_args()
    
    # Determine if we should use automatic detection
    use_auto = args.auto or (args.profile_id is None and args.experiment is None)
    
    try:
        if use_auto:
            # Initialize analyser with automatic detection
            analyser = TransientConvergenceAnalyzer()
            
            # Auto-detect profile, experiment, and resolution factors
            resolution_factors, file_paths = analyser.auto_configure()
            
            # Use detected resolution factors if not specified by user
            if args.resolution_factors is None:
                args.resolution_factors = resolution_factors
            
            # Load transient simulation results using detected file paths
            analyser.load_transient_results(args.resolution_factors, file_paths, args.final_time_only)
        else:
            # Use provided or default values
            profile_id = args.profile_id or 22
            experiment = args.experiment or 'S4'
            resolution_factors = args.resolution_factors or [0.5, 0.75, 1.0]
            
            # Initialize analyser with specified parameters
            analyser = TransientConvergenceAnalyzer(profile_id, experiment)
            
            # Load transient simulation results
            analyser.load_transient_results(resolution_factors, None, args.final_time_only)
        
        # Validate data quality
        analyser.validate_transient_data_quality()
        
        if not args.plot_only:
            # Interpolate to common grid
            analyser.interpolate_to_common_grid()
            
            # Calculate convergence metrics
            analyser.calculate_convergence_metrics()
            
            # Assess convergence
            analyser.assess_convergence(args.tolerance)
            
            # Generate report
            analyser.generate_transient_report()
        
        # Create plots
        analyser.create_transient_comparison_plots()
        
        print(f"\n✓ Transient grid convergence analysis complete!")
        
    except Exception as e:
        print(f"✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())