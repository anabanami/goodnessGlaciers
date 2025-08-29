"""
Phase Analysis - Analyses phase relationship between bed and surface over time
"""
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from scipy import signal
from scipy.signal import correlate
from scipy.signal import sosfreqz
import os
import sys
import glob
import re

# Add parent directory (2 levels up) to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, parent_dir)
from bedrock_generator import SyntheticBedrockModelConfig
from bamgflowband import bamgflowband


def parse_filename(ncfile):
    """Extract profile ID and experiment from filename"""
    if len(sys.argv) < 2:
        print("Usage: python phase_analysis.py <file.nc>")
        sys.exit(1)
    
    bed = os.path.splitext(os.path.basename(ncfile))[0]
    profile_str, exp = bed.split('_')[:2]
    profile_id = int(profile_str)
    print(f"Profile ID: {profile_id:03d}, Experiment: {exp}")
    return profile_id, exp


def load_config(ncfile, bedrock_file):
    """Load bedrock configuration"""
    name = os.path.splitext(os.path.basename(ncfile))[0]
    profile_str, exp = name.split('_')[:2]
    profile_id = int(profile_str)

    # derive the bedrock_profiles folder directly from the passed-in .npz path
    output_dir = os.path.dirname(bedrock_file)
    config = SyntheticBedrockModelConfig(profile_id, output_dir=output_dir)
    wavelength_km = config.profile_params['wavelength'] / 1000
    return config, wavelength_km


def build_mesh_coordinates_from_netcdf(dataset, config):
    """Extract mesh coordinates directly from the NetCDF file if available"""
    try:
        # Try to get mesh from NetCDF first
        mesh_group = dataset.groups.get('Mesh') or dataset.groups.get('mesh')
        if mesh_group and 'x' in mesh_group.variables and 'y' in mesh_group.variables:
            x_mesh = mesh_group.variables['x'][:] / 1000  # Convert to km
            y_mesh = mesh_group.variables['y'][:] / 1000  # Convert to km
            print("Using mesh coordinates from NetCDF file")
            return x_mesh, y_mesh
    except:
        pass
    
    # Fallback: reconstruct mesh using same algorithm as flowline.py
    print("Reconstructing mesh coordinates using flowline.py algorithm")
    return build_mesh_coordinates_flowline_exact(config)

def build_mesh_coordinates_flowline_exact(config, resolution_factor=0.5):
    """Build mesh coordinates exactly matching the flowline.py adaptive_bamg function"""
    from scipy.signal import find_peaks
    
    # Domain parameters - must match flowline.py exactly
    target_L = 160e3  # Target domain length (m)
    target_nx = 1600  # Target resolution
    
    # Find optimal L that ends at a peak (matching flowline.py logic)
    search_window = 5e3
    x_search = np.linspace(target_L - search_window, target_L + search_window, 
                          int(2 * search_window / 100))  # 100m resolution
    bed_search = config.get_bedrock_elevation(x_search)
    
    # Find peaks
    peaks, _ = find_peaks(bed_search, 
                         distance=int(0.5 * config.profile_params['wavelength'] / 100))
    
    if len(peaks) > 0:
        peak_positions = x_search[peaks]
        closest_idx = np.argmin(np.abs(peak_positions - target_L))
        L = peak_positions[closest_idx]
    else:
        L = target_L
    
    # Adjust nx proportionally to maintain resolution
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
    
    return md.mesh.x / 1000, md.mesh.y / 1000  # Convert to km

# Keep the old function for backward compatibility
def build_mesh_coordinates(config, resolution_factor=0.5):
    """Build mesh coordinates from simulation parameters matching flowline.py exactly"""
    return build_mesh_coordinates_flowline_exact(config, resolution_factor)


def phase_shift_analysis(dx, bed, surface, wavelength, time_val=None):
    """Calculate phase shift between bed and surface signals - original algorithm"""
    if time_val is not None:
        print(f"~~~Comparing bed and surface at t={time_val:.1f} years~~~")

    # Normalize signals before correlation
    bed_norm = (bed - np.mean(bed)) / np.std(bed)
    surface_norm = (surface - np.mean(surface)) / np.std(surface)
    
    # Calculate cross-correlation
    xcorr = correlate(bed_norm, surface_norm, mode='full')
    nsamples = bed.size
    
    # Find the lag with maximum correlation
    max_corr_idx = np.argmax(xcorr)
    center_idx = len(xcorr) // 2
    shift_idx = max_corr_idx - center_idx
    
    # Convert lag to distance
    lag_distance = shift_idx * dx
    
    # Limit to ±half wavelength to avoid jumping to next peak
    if abs(lag_distance) > wavelength/2:
        lag_distance = lag_distance % wavelength
        if lag_distance > wavelength/2:
            lag_distance -= wavelength
    
    # Calculate phase shift in radians
    phase_shift = (2 * np.pi * lag_distance) / wavelength
    phase_shift_deg = (phase_shift * 180) / np.pi
    
    # Calculate correlation coefficient
    max_corr = xcorr[max_corr_idx] / (nsamples * np.std(bed) * np.std(surface))
    
    return phase_shift, lag_distance, max_corr


def load_time_step_data(dataset, time_step, config):
    """Load data for specific time step"""
    print(f"Loading data for time step {time_step}")
    
    # Get time
    tsol = dataset.groups['results'].groups['TransientSolution']
    times = tsol.variables['time'][:]
    time_val = float(times[time_step])  # Times are already in years

    # Load data (use initial bed, current surface)
    surface_data = tsol.variables['Surface'][time_step].squeeze()
    
    # Get mesh coordinates - try NetCDF first, then reconstruct
    x_mesh, y_mesh = build_mesh_coordinates_from_netcdf(dataset, config)
    
    print(f"Surface data shape: {surface_data.shape}")
    print(f"Mesh coordinates shape: x={x_mesh.shape}, y={y_mesh.shape}")
    
    # Check dimensions match
    if surface_data.shape[0] != x_mesh.shape[0]:
        raise ValueError(f"Dimension mismatch: surface_data has {surface_data.shape[0]} elements, "
                        f"but mesh has {x_mesh.shape[0]} vertices")
    
    # 3) mask to centre‐line
    tol = 0.05  # km tolerance
    mask = np.abs(y_mesh) <= tol
    if mask.sum() < 10:
        mask = y_mesh <= np.percentile(y_mesh, 10)

    print(f"Centerline mask: {mask.sum()} vertices selected from {len(mask)} total")

    # 4) apply mask to x & surface
    x       = x_mesh[mask]
    surface = surface_data[mask]

    # 5) now compute true bedrock only at those x (km→m!)
    b0 = config.get_bedrock_elevation(x * 1e3)
    
    # 6) sort into ascending x and remove duplicates
    idx = np.argsort(x)
    x, b0, surface = x[idx], b0[idx], surface[idx]
    
    # Remove duplicate x-coordinates to prevent gradient calculation issues
    unique_mask = np.concatenate(([True], np.diff(x) > 1e-10))  # 1e-10 km = 0.1mm tolerance
    x, b0, surface = x[unique_mask], b0[unique_mask], surface[unique_mask]
    
    print(f"Extracted profile with {len(x)} points, X range: {x.min():.1f} to {x.max():.1f} km")
    return x, b0, surface, time_val


def plot_signals(x, bed, surface, time_val, config, profile_id, output_dir, ylim_main=None, ylim=None):
    """Plot bed & surface + signals"""
    b = (bed - np.mean(bed)) / 1e3  # Convert to km, remove mean
    s = (surface - np.mean(surface)) / 1e3
    h = config.ice_thickness / 1e3
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios':[3,2]})
    
    # Top panel: bed + surface + fill
    ax1.plot(x, b, 'k-', label='Bed')
    ax1.plot(x, s + h, 'b-', label='Surface')
    ax1.fill_between(x, b, s + h, alpha=0.3, label='Ice thickness')
    ax1.set_ylabel('Elevation (km)')
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.set_title(f'Bed vs Surface at t={time_val:.1f}yr')
    ax1.legend(loc='upper right')
    
    # Bottom panel: signals
    # Detrend bed and surface

    bed_detrended = signal.detrend(bed)
    surface_detrended = signal.detrend(surface)

    ax2.plot(x, bed_detrended, 'k-', label='Bed detrended')
    ax2.plot(x, surface_detrended, 'b-', label='Surface detrended')
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Filtered elev. (m)')
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='upper right')
    ax2.set_title('mean corrected signals')
    
    plt.tight_layout()
    fname = f"signals_t{time_val:.1f}_{profile_id:03d}.png"
    # plt.show()
    plt.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close()


def visualise_cross_correlation(x, bed_slope, surface_slope, wavelength, time_val, profile_id, output_dir):
    """Create cross-correlation visualization"""
    plt.figure(figsize=(12, 6))
    
    # Normalize signals for correlation
    bed_norm = (bed_slope - np.mean(bed_slope)) / np.std(bed_slope)
    surf_norm = (surface_slope - np.mean(surface_slope)) / np.std(surface_slope)
    
    # Calculate cross-correlation
    xcorr = correlate(bed_norm, surf_norm, mode='full')
    nsamples = bed_norm.size
    lags = np.arange(-(nsamples-1), nsamples)
    dx = np.mean(np.diff(x))
    lag_distances = lags * dx
    
    # Find maximum correlation
    max_corr_idx = np.argmax(xcorr)
    center_idx = len(xcorr) // 2
    shift_idx = max_corr_idx - center_idx
    lag_distance = shift_idx * dx
    
    # Wrap phase calculation
    if abs(lag_distance) > wavelength/2:
        lag_distance = lag_distance % wavelength
        if lag_distance > wavelength/2:
            lag_distance -= wavelength
    
    phase_shift = (2 * np.pi * lag_distance) / wavelength
    phase_deg = phase_shift * 180 / np.pi
    max_corr = xcorr[max_corr_idx] / (nsamples * np.std(bed_slope) * np.std(surface_slope))
    
    # Plot correlation (only near center for clarity)
    xlim = 1.5 * wavelength
    visible_lags = np.where((lag_distances >= -xlim) & (lag_distances <= xlim))[0]
    
    plt.plot(lag_distances[visible_lags], xcorr[visible_lags], 'b-', linewidth=2)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Zero lag')
    plt.axvline(x=lag_distance, color='purple', linestyle='--', linewidth=2,
               label=f'Maximum correlation: {lag_distance:.2f} km')
    
    # Add theoretical phase lines
    theoretical_lag = (90/360) * wavelength
    plt.axvline(x=theoretical_lag, color='g', linestyle='--', linewidth=1.5,
               label=f'Theoretical (90°): {theoretical_lag:.2f} km')
    plt.axvline(x=-theoretical_lag, color='r', linestyle='--', linewidth=1.5,
               label=f'Opposite (-90°): {-theoretical_lag:.2f} km')
    
    plt.xlabel('Lag distance (km)')
    plt.ylabel('Cross-correlation')
    plt.title(f'Cross-correlation: bed and surface at t={time_val:.1f} years')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', c="grey")
    
    plt.annotate(f'Measured phase shift: {phase_deg:.1f}° (lag: {lag_distance:.3f} km)',
                xy=(0.02, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f"bed_surface_correlation_t{time_val:.1f}_{profile_id:03d}.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    return phase_deg, lag_distance


def visualise_phase_relationship(x, bed_slope, surface_slope, wavelength, time_val, profile_id, output_dir):
    """Create phase relationship visualization"""
    plt.figure(figsize=(14, 8))
    
    # Normalize signals for comparison
    bed_norm = (bed_slope - np.mean(bed_slope)) / np.std(bed_slope)
    surf_norm = (surface_slope - np.mean(surface_slope)) / np.std(surface_slope)
    
    # Plot normalized signals
    plt.plot(x, bed_norm, 'k-', label='Bed (normalised)', linewidth=2)
    plt.plot(x, surf_norm, 'b-', label='Surface (normalised)', linewidth=2)
    
    # Find and mark peaks
    bed_peaks = signal.find_peaks(bed_norm)[0]
    surf_peaks = signal.find_peaks(surf_norm)[0]
    plt.plot(x[bed_peaks], bed_norm[bed_peaks], 'ko', markersize=8)
    plt.plot(x[surf_peaks], surf_norm[surf_peaks], 'bo', markersize=8)
    
    # Draw vertical guides at bed peaks
    for bp in bed_peaks:
        plt.axvline(x=x[bp], color='gray', linestyle='--', alpha=0.3)
    
    # Calculate phase shift information
    dx = np.mean(np.diff(x))
    xcorr = correlate(bed_norm, surf_norm, mode='full')
    max_corr_idx = np.argmax(xcorr)
    center_idx = len(xcorr) // 2
    shift_idx = max_corr_idx - center_idx
    lag_distance = shift_idx * dx
    
    phase_shift = (2 * np.pi * lag_distance) / wavelength
    # Modulus wrap via complex exponential
    phase_deg = np.degrees(np.angle(np.exp(1j * phase_shift)))
    
    plt.annotate(f'Measured phase shift: {phase_deg:.1f}°\nLag distance: {lag_distance:.3f} km\n'
                f'Theoretical phase: 90°\nWavelength: {wavelength:.2f} km',
                xy=(0.02, 0.02), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8), fontsize=10)
    
    plt.title(f'Phase relationship between bed and surface at t={time_val:.1f} years', fontsize=14)
    plt.xlabel('Distance (km)', fontsize=12)
    plt.ylabel('Normalised amplitude', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12, loc='upper right')
    
    # Set limits to show 3 wavelengths
    xlim_start = x.min()
    xlim_end = min(x.min() + 3*wavelength, x.max())
    plt.xlim(xlim_start, xlim_end)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f"phase_relationship_t{time_val:.1f}_{profile_id:03d}.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    return phase_deg, lag_distance


def analyse_phase_evolution(dataset, wavelength, config, profile_id, output_dir):
    """Main analysis function - preserves original two-pass approach"""
    # Setup
    tsol = dataset.groups['results'].groups['TransientSolution']
    times = tsol.variables['time'][:]
    num_steps = len(times)

    print(f"Analysing {num_steps} time steps")
    
    # Create output directories
    for subdir in ['signals', 'correlations', 'phase_relationship']:
        os.makedirs(f"{output_dir}/{subdir}", exist_ok=True)
    
    # First pass: determine global y-axis limits
    print("\n=== First pass: determining global y-axis limits ===")
    all_bed_vals = []
    all_surface_vals = []
    all_bed_slope_vals = []
    all_surface_slope_vals = []
    
    for step in range(1, num_steps):
        print(f"Analysing {num_steps} time steps (skipping first time step due to masked data)")
        
        x, bed, surface, time_val = load_time_step_data(dataset, step, config)
        dx = np.mean(np.diff(x))
        
        # Calculate gradients with edge handling
        bed_slope = np.gradient(bed, x, edge_order=1)
        surface_slope = np.gradient(surface, x, edge_order=1)
        
        # Replace any NaN or inf values
        bed_slope = np.where(np.isfinite(bed_slope), bed_slope, 0)
        surface_slope = np.where(np.isfinite(surface_slope), surface_slope, 0)

        # Store values for limit calculation (after mean removal)
        all_bed_vals.extend((bed - np.mean(bed)).tolist())
        all_surface_vals.extend((surface - np.mean(surface)).tolist())
        all_bed_slope_vals.extend((bed_slope - np.mean(bed_slope)).tolist())
        all_surface_slope_vals.extend((surface_slope - np.mean(surface_slope)).tolist())
    
    # Calculate global y-limits with padding
    main_min = min(min(all_bed_vals), min(all_surface_vals))
    main_max = max(max(all_bed_vals), max(all_surface_vals))
    slope_min = min(min(all_bed_slope_vals), min(all_surface_slope_vals))
    slope_max = max(max(all_bed_slope_vals), max(all_surface_slope_vals))
    
    padding_main = 0.1 * (main_max - main_min)
    padding = 0.1 * (slope_max - slope_min)
    
    ylim_main = (main_min - padding_main, main_max + padding_main)
    ylim = (slope_min - padding, slope_max + padding)
    
    print(f"Global y-limits for main plots: {ylim_main}")
    print(f"Global y-limits for slope plots: {ylim}")
    
    # Second pass: actual analysis with consistent y-limits
    print("\n=== Second pass: processing with consistent y-axis limits ===")
    
    # Storage for results
    time_values = []
    phase_shifts = []
    lag_distances = []
    correlations = []
    
    # Process each time step
    for step in range(1, num_steps):
        print(f"\n=== Processing time step {step+1}/{num_steps} ===")
        
        x, bed, surface, time_val = load_time_step_data(dataset, step, config)
        dx = np.mean(np.diff(x))

        # Calculate gradients with edge handling
        bed_slope = np.gradient(bed, x, edge_order=1)
        surface_slope = np.gradient(surface, x, edge_order=1)
        
        # Replace any NaN or inf values
        bed_slope = np.where(np.isfinite(bed_slope), bed_slope, 0)
        surface_slope = np.where(np.isfinite(surface_slope), surface_slope, 0)
        
        # Generate plots
        plot_signals(x, bed, surface,
                    time_val, config, profile_id, f"{output_dir}/signals", ylim_main, ylim)
        visualise_phase_relationship(x, bed_slope, surface_slope, 
                                    wavelength, time_val, profile_id, f"{output_dir}/phase_relationship")
        visualise_cross_correlation(x, bed_slope, surface_slope,
                                   wavelength, time_val, profile_id, f"{output_dir}/correlations")
        
        # Analyse phase shift using original algorithm
        phase_shift, lag, corr = phase_shift_analysis(dx, bed_slope, surface_slope, wavelength, time_val)
        
        # Store results
        time_values.append(time_val)
        phase_shifts.append(phase_shift)
        lag_distances.append(lag)
        correlations.append(corr)
    
    # Plot evolution over time
    plt.figure(figsize=(14, 10))
    phase_shifts_deg = np.array(phase_shifts) * 180 / np.pi
    
    # Phase shift evolution
    plt.subplot(2, 1, 1)
    plt.plot(time_values, phase_shifts_deg, 'bo-', label='Phase shift', linewidth=2)
    plt.axhline(y=90, color='g', linestyle='--', label='Theoretical (90°)', linewidth=2)
    plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    plt.axhline(y=-90, color='r', linestyle='--', label='Opposite phase (-90°)', linewidth=1.5)
    plt.fill_between(time_values, 0, 90, color='green', alpha=0.1, label='Positive phase region')
    plt.fill_between(time_values, 0, -90, color='red', alpha=0.1, label='Negative phase region')
    plt.xlabel('Time (years)')
    plt.ylabel('Phase shift (degrees)')
    plt.title('Evolution of phase shift between bed and surface over time')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='upper right')
    
    # Lag distance evolution
    plt.subplot(2, 1, 2)
    plt.plot(time_values, lag_distances, 'ro-', label='Lag distance', linewidth=2)
    plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    theoretical_lag = wavelength/4
    plt.axhline(y=theoretical_lag, color='g', linestyle='--', 
                label=f'Theoretical (λ/4 = {theoretical_lag:.2f} km)', linewidth=2)
    plt.axhline(y=-theoretical_lag, color='r', linestyle='--', 
                label=f'Opposite (–λ/4 = {-theoretical_lag:.2f} km)', linewidth=1.5)
    plt.xlabel('Time (years)')
    plt.ylabel('Lag distance (km)')
    plt.title('Evolution of spatial lag between bed and surface')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f"phase_evolution_{profile_id:03d}.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    
    # Create results dictionary
    results = {
        'time': time_values,
        'phase_shift': phase_shifts,
        'phase_shift_deg': phase_shifts_deg,
        'lag_distance': lag_distances,
        'correlation': correlations,
        'ylim_main': ylim_main,
        'ylim': ylim,
        'wavelength': wavelength
    }
    
    return results

def main():

    # Get bedrock profiles - pointing to the correct location
    bedrock_files = glob.glob("../../bedrock_profiles/bedrock_profile_*.npz")
    bedrock_profiles = {re.search(r'(\d+)\.npz$', f).group(1): f for f in bedrock_files}

    
    ncfile = sys.argv[1]
    global profile_id  # Need this for plot filenames

    profile_id, exp = parse_filename(ncfile)
    config, wavelength_km = load_config(ncfile, bedrock_profiles[f"{profile_id:03d}"])
    
    output_dir = f"{profile_id:03d}_{exp}_phase_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    with nc.Dataset(ncfile, 'r') as dataset:
        results = analyse_phase_evolution(dataset, wavelength_km, config, profile_id, output_dir)
    
    # Write summary results to file
    summary_file = os.path.join(output_dir, "phase_analysis_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Phase Analysis Summary\n")
        f.write("=====================\n\n")
        f.write(f"Bedrock wavelength: {wavelength_km} km\n\n")
        f.write(f"Coordinates: Slope-parallel\n\n")
        f.write("Time-dependent phase shift:\n")
        f.write("-------------------------\n")
        for i in range(len(results['time'])):
            f.write(f"Time: {results['time'][i]:.1f} years\n")
            f.write(f"  Phase shift: {results['phase_shift'][i]/np.pi:.2f}π radians ({results['phase_shift_deg'][i]:.1f} degrees)\n")
            f.write(f"  Lag distance: {results['lag_distance'][i]:.3f} km\n")
            f.write(f"  Correlation: {results['correlation'][i]:.3f}\n")
            f.write(f"  Difference from π/2: {abs(results['phase_shift'][i] - np.pi/2)/np.pi:.2f}π radians\n\n")
        
        f.write("\nPlotting Information:\n")
        f.write("--------------------\n")
        f.write(f"Y-axis limits (main plots): {results['ylim_main']}\n")
        f.write(f"Y-axis limits (plots): {results['ylim']}\n")
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")
    print(f"Using consistent y-axis limits for all plots:")
    print(f"  Main plots: {results['ylim_main']}")
    print(f"  Filtered plots: {results['ylim']}")


YTS = 31556926  # ISSM seconds per year

if __name__ == '__main__':
    main()