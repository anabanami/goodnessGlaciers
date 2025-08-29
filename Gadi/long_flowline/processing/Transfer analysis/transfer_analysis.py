import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from scipy import signal
from scipy.signal import hilbert
from scipy.signal import butter, sosfiltfilt
from scipy.signal import correlate
from scipy.signal import sosfreqz
import pandas as pd
from bamgflowband import bamgflowband
import glob
import re
import os
import sys

# Add parent directory (2 levels up) to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, parent_dir)
from bedrock_generator import SyntheticBedrockModelConfig
from domain_utils import find_optimal_domain_length


def load_config(ncfile, bedrock_file):
    """Load bedrock configuration"""
    name = os.path.splitext(os.path.basename(ncfile))[0]
    profile_str, exp = name.split('_')[:2]
    profile_id = int(profile_str)

    # derive the bedrock_profiles folder directly from the passed-in .npz path
    output_dir = os.path.dirname(bedrock_file)
    config = SyntheticBedrockModelConfig(profile_id, output_dir=output_dir)
    λb_km = config.profile_params['wavelength'] / 1000
    return config, λb_km


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
    
    # Domain parameters - must match flowline.py exactly (updated to 135km)
    target_L = 135e3  # Updated to match flowline.py
    L = find_optimal_domain_length(config, target_L)
    
    # Adjust nx proportionally to maintain resolution (updated to match flowline.py)
    target_nx = 1350  # Updated to match flowline.py
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


def bandpass_filter(x, dx, data, λ, scenario_key):
    """Apply bandpass filter centered on λb"""
   
    fs = 1.0 / dx
    nyquist = fs / 2
    target_freq = 1.0 / λ # f0 in cycles per km

    w0 = 2 * np.pi * target_freq / fs  # convert to angular frequency in radians/sample

    low_freq = target_freq * 0.5
    high_freq = target_freq * 1.5

    sos = butter(4, [low_freq, high_freq], btype='band', fs=fs, output='sos')
    # print(f"low = {low_freq:.4f}, high = {high_freq:.4f}, nyquist = {nyquist:.4f}")

    # 1) compute frequency response
    w, h = sosfreqz(sos, worN=[w0]) 
    gain_f0 = np.abs(h[0])
    # print(f"{gain_f0 =}")
    return sosfiltfilt(sos, data), w, gain_f0, low_freq, high_freq


def phase_shift_analysis(dx, bed, surface, λb):
    """Calculate phase shift between bed and surface signals - original algorithm"""

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
    
    # Limit to ±half λb to avoid jumping to next peak
    if abs(lag_distance) > λb/2:
        lag_distance = lag_distance % λb
        if lag_distance > λb/2:
            lag_distance -= λb
    
    # Calculate phase shift in radians
    phase_shift = (2 * np.pi * lag_distance) / λb
    phase_shift_deg = (phase_shift * 180) / np.pi
    
    # Calculate correlation coefficient
    max_corr = xcorr[max_corr_idx] / (nsamples * np.std(bed) * np.std(surface))
    
    return phase_shift, lag_distance, max_corr


def load_data(dataset, config):
    # 1) pull in the last surface
    surface_data = dataset.groups['results'].groups['TransientSolution'].variables['Surface'][-1].squeeze()

    # 2) build mesh coords - try NetCDF first, then reconstruct
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
    x = x_mesh[mask]
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
    return x, surface, b0


def process_profile(ncfile, bedrock_file,
                    scenario_key, scenario_name, profile_id,
                    do_plot: bool = True):
    """Process a single profile - NO FILTERING but with proper λs estimation"""

    # --- load config and data ---
    config, λb = load_config(ncfile, bedrock_file)
    with nc.Dataset(ncfile) as dataset:
        x, surface, b0 = load_data(dataset, config)
    dx = np.mean(np.diff(x))

    # --- compute raw slopes ---
    b0_slope = np.gradient(b0, x)
    surface_slope = np.gradient(surface, x)

    # --- estimate surface wavelength from raw slope WITH VALIDATION ---
    try:
        analytic_surface_raw = hilbert(surface_slope)
        inst_phase_surface = np.unwrap(np.angle(analytic_surface_raw))
        k_surface_inst = np.gradient(inst_phase_surface, x)
        
        # Filter out bad wavenumber values
        k_valid = k_surface_inst[np.isfinite(k_surface_inst) & (k_surface_inst > 0)]
        
        if len(k_valid) > 10:  # Need enough valid points
            λs_inst = 2 * np.pi / k_valid
            
            # Remove outliers - keep reasonable wavelengths
            λs_median = np.median(λs_inst)
            λs_valid = λs_inst[(λs_inst > λs_median/5) & (λs_inst < λs_median*5)]
            
            if len(λs_valid) > 5:
                λs = np.mean(λs_valid)
                
                # Final sanity check
                if np.isfinite(λs) and 0.5*λb < λs < 10*λb:  # Reasonable range relative to λb
                    print(f"Estimated λs = {λs:.2f} km (λb = {λb:.2f} km)")
                else:
                    print(f"WARNING: λs = {λs:.2f} km outside reasonable range, using λb")
                    λs = λb
            else:
                print("WARNING: Too many λs outliers, using λb")
                λs = λb
        else:
            print("WARNING: Insufficient valid wavenumbers, using λb")
            λs = λb
            
    except Exception as e:
        print(f"WARNING: λs estimation failed ({e}), using λb")
        λs = λb

    # --- NO FILTERING - use raw slopes directly ---
    # --- compute envelopes ---
    env_b0 = np.abs(hilbert(b0_slope))  # Raw slope envelope
    env_surf = np.abs(hilbert(surface_slope))  # Raw slope envelope

    Ab_slope = env_b0.mean()
    As_slope = env_surf.mean()

    # wavenumbers (using the validated λs)
    k_b = 2*np.pi/λb
    k_s = 2*np.pi/λs

    # true elevation amplitudes:
    Ab = Ab_slope / k_b
    As = As_slope / k_s

    # amplitude ratio
    amp_ratio = Ab / As if As > 0 else np.inf

    # wavelength ratio
    wavelength_ratio = λb / λs

    # --- phase shift analysis using raw slopes ---
    phase_shift, lag_distance, max_corr = phase_shift_analysis(dx, b0_slope, surface_slope, λb)
    phase_shift_deg = np.degrees(phase_shift)

    # --- theoretical Budd damping for comparison ---
    Z_km = config.ice_thickness / 1e3
    k = 2 * np.pi / λb
    ωZ = k * Z_km
    psi = (np.exp(ωZ) - np.exp(-ωZ)) / (ωZ)**2

    # --- print summary ---
    print(f"Maximum correlation: {max_corr:.3f}")
    print(f"Spatial shift: {lag_distance:.3f} km")
    print(f"Phase shift: {phase_shift:.2f}π radians or {phase_shift_deg:.1f}°")
    print(f"→ measured Amp ratio = {amp_ratio:6.2f}")
    print(f"→ theoretical ψ      =   {psi:.2f}")
    print(f"→ λ ratio            = {wavelength_ratio:6.2f}")

    # --- optional quick plot ---
    if do_plot:
        # For plotting, use raw elevation data (no filtering)
        plt.figure(figsize=(16,5))
        plt.plot(x, b0, 'k-', label='Raw bedrock', linewidth=2)
        plt.plot(x, surface, 'C0-', label='Raw surface', linewidth=2)
        plt.fill_between(x, b0, surface, alpha=0.3)
        plt.grid(True, linestyle=':')
        plt.title(f'{scenario_name} – Profile {profile_id} (Raw Signals)')
        plt.xlabel('Distance (km)')
        plt.ylabel('Elevation (km)')
        plt.legend()
        
        # Add text box with key metrics
        textstr = f'λb = {λb:.1f} km\nλs = {λs:.1f} km\nRatio = {amp_ratio:.1f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.savefig(f"transfer_plots/surf_vs_bed/{profile_id}_{scenario_name}_raw.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()

    # --- return results dict (same structure as before) ---
    return {
        'scenario'         : scenario_key,
        'scenario_name'    : scenario_name,
        'profile_id'       : profile_id,
        'wavelength_b'     : λb,
        'wavelength_s'     : λs,
        'amplitude_b'      : Ab,
        'amplitude_s'      : As,
        'amplitude_ratio'  : amp_ratio,
        'theory_amp_ratio' : psi,
        'wavelength_ratio' : wavelength_ratio,
        'phase_shift_rad'  : phase_shift,
        'phase_shift_deg'  : phase_shift_deg,
        'lag_distance'     : lag_distance,
        'max_correlation'  : max_corr
    }


def analyse_bed_surface_transfer():
    """Analyzes bed-to-surface transfer using NetCDF4 data"""
    
    # Setup
    scenarios = {
        "S1": "Linear rheology + frozen bed",
        "S2": "Non-linear rheology + frozen bed", 
        "S3": "Linear rheology + sliding law",
        "S4": "Non-linear rheology + sliding law"
    }
    
    results = []
    os.makedirs("transfer_plots", exist_ok=True)
    os.makedirs("transfer_plots/surf_vs_bed", exist_ok=True)
    
    # Get bedrock profiles - pointing to the correct location
    bedrock_files = glob.glob("../../bedrock_profiles/bedrock_profile_*.npz")
    bedrock_profiles = {re.search(r'(\d+)\.npz$', f).group(1): f for f in bedrock_files}
    
    print(f"Found {len(bedrock_profiles)} bedrock profiles")
    print(f"Bedrock profile IDs: {sorted(bedrock_profiles.keys())}")
    
    # Process each scenario
    for scenario_key, scenario_name in scenarios.items():
        scenario_dir = scenario_key  # S1, S2, etc. in current directory
        if not os.path.exists(scenario_dir):
            print(f"Warning: Directory {scenario_dir} not found")
            continue
            
        print(f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"\nProcessing {scenario_key}: {scenario_name}")
        
        # Find NC files directly in scenario directory
        ncfiles = glob.glob(f"{scenario_dir}/*_{scenario_key}_0.5.nc")
        print(f"  Found {len(ncfiles)} NC files")
        
        for ncfile in ncfiles:
            # Extract profile ID from filename (e.g., "001" from "001_S1_0.5.nc")
            filename = os.path.basename(ncfile)
            profile_match = re.match(r'(\d+)_S\d+_0\.5\.nc$', filename)
            if not profile_match:
                print(f"  Skipping {filename} - couldn't extract profile ID")
                continue
                
            profile_id = profile_match.group(1)
            
            # Check if we have bedrock data for this profile
            if profile_id not in bedrock_profiles:
                print(f"  Skipping profile {profile_id} - no bedrock data found")
                continue
                
            print(f"\nProcessing profile {profile_id}")
            result = process_profile(ncfile, bedrock_profiles[profile_id], 
                                   scenario_key, scenario_name, profile_id)
            if result:
                results.append(result)
    
    # Save and plot results
    if results:
        df = pd.DataFrame(results)
        df.to_csv("transfer_function_results.csv", index=False)
        generate_plots(df)
        print(f"\nAnalysis complete! Processed {len(results)} profiles.")
        print(f"Results saved to transfer_function_results.csv")
        print(f"Plots saved in transfer_plots/ directory")
    else:
        print("No data processed.")
    
    return pd.DataFrame(results)


def generate_plots(df):
    """Generate summary plots including matrix visualizations"""

    scenarios = df['scenario'].unique()
    markers = ['s', 'o', '^', 'd']
    colors = ['C0', 'C2', 'C3', 'C4']
    
    # Plot 1: Amplitude ratio
    plt.figure(figsize=(7.5, 5))
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label= "Perfect transfer")
    for i, scenario in enumerate(scenarios):
        subset = df[df['scenario'] == scenario].sort_values('wavelength_b')
        if not subset.empty:
            plt.plot(subset['wavelength_b'], subset['amplitude_ratio'], 
                    marker=markers[i], color=colors[i], linestyle='', 
                    label=subset['scenario_name'].iloc[0], markersize=8)
    plt.xlabel('Bedrock Wavelength (km)')
    plt.ylabel('Amplitude Ratio (Ab/As)')
    plt.title('Transfer of Bedrock Topography to Surface')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("transfer_plots/amplitude_transfer.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Phase shift
    plt.figure(figsize=(7.5, 5))
    plt.axhline(y=90.0, color='green', linestyle='--', alpha=0.7, label="theoretical phase shift")
    plt.axhline(y=0.0, color='grey', linestyle='--', alpha=0.7)
    for i, scenario in enumerate(scenarios):
        subset = df[df['scenario'] == scenario].sort_values('wavelength_b')
        if not subset.empty:
            plt.plot(subset['wavelength_b'], subset['phase_shift_deg'], 
                    marker=markers[i], color=colors[i], linestyle='', 
                    label=subset['scenario_name'].iloc[0], markersize=8)
    
    plt.xlabel('Bedrock Wavelength (km)')
    plt.ylabel('Phase Shift (degrees)')
    plt.title('Phase Shift Between Bed and Surface')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("transfer_plots/phase_shift.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Wavelength ratio
    plt.figure(figsize=(7.5, 5))
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label= "Perfect transfer")
    for i, scenario in enumerate(scenarios):
        subset = df[df['scenario'] == scenario].sort_values('wavelength_b')
        if not subset.empty:
            plt.plot(subset['wavelength_b'], subset['wavelength_ratio'], 
                    marker=markers[i], color=colors[i], linestyle='', 
                    label=subset['scenario_name'].iloc[0], markersize=8)
    
    plt.xlabel('Bedrock Wavelength (km)')
    plt.ylabel('Wavelength Ratio (λb/λs)')
    plt.title('Wavelength Transfer from Bed to Surface')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("transfer_plots/wavelength_transfer.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate matrix plots
    generate_matrix_plots(df)


def generate_matrix_plots(df):
    """Generate matrix/heatmap visualizations"""
    
    # Get unique scenarios and wavelengths
    scenarios = sorted(df['scenario'].unique())
    wavelengths = sorted(df['wavelength_b'].unique())
    
    # Create scenario labels with proper names
    scenario_labels = []
    for scenario in scenarios:
        scenario_name = df[df['scenario'] == scenario]['scenario_name'].iloc[0]
        scenario_labels.append(scenario_name)
    
    # Initialize matrices
    amplitude_matrix = np.full((len(scenarios), len(wavelengths)), np.nan)
    phase_matrix = np.full((len(scenarios), len(wavelengths)), np.nan)
    wavelength_ratio_matrix = np.full((len(scenarios), len(wavelengths)), np.nan)
    
    # Fill matrices with data
    for _, row in df.iterrows():
        scenario_idx = scenarios.index(row['scenario'])
        wavelength_idx = wavelengths.index(row['wavelength_b'])
        
        amplitude_matrix[scenario_idx, wavelength_idx] = row['amplitude_ratio']
        phase_matrix[scenario_idx, wavelength_idx] = row['phase_shift_deg']
        wavelength_ratio_matrix[scenario_idx, wavelength_idx] = row['wavelength_ratio']
    
    
    # Create separate detailed matrices with better formatting
    create_detailed_matrices(amplitude_matrix, phase_matrix, wavelength_ratio_matrix, 
                            scenarios, wavelengths, scenario_labels)


def create_detailed_matrices(amplitude_matrix, phase_matrix, wavelength_ratio_matrix, 
                           scenarios, wavelengths, scenario_labels):
    """Create detailed individual matrix plots"""
    
    # Amplitude Ratio Matrix
    plt.figure(figsize=(7.5, 5))
    im = plt.imshow(amplitude_matrix, cmap='inferno', aspect='auto')
    
    plt.title('Amplitude Ratio (Ab/As)', fontsize=16, pad=20)
    plt.ylabel('Scenario', fontsize=14)
    plt.xlabel('Bedrock Wavelength (km)', fontsize=14)
    
    # Custom ticks and labels
    plt.yticks(range(len(scenarios)), scenarios, fontsize=12)
    plt.xticks(range(len(wavelengths)), [f"{w:.1f}" for w in wavelengths], 
               rotation=45, fontsize=10)
    
    # Add colorbar with better formatting
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Amplitude Ratio', rotation=270, labelpad=20, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Add text annotations for values
    for i in range(len(scenarios)):
        for j in range(len(wavelengths)):
            if not np.isnan(amplitude_matrix[i, j]):
                text = plt.text(j, i, f'{amplitude_matrix[i, j]:.2f}',
                               ha="center", va="center", color="white" if amplitude_matrix[i, j] < 400 else "black",
                               fontsize=12)
    plt.tight_layout()
    plt.savefig("transfer_plots/amplitude_ratio_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Phase Shift Matrix
    plt.figure(figsize=(7.5, 5))
    im = plt.imshow(phase_matrix, cmap='coolwarm', aspect='auto', vmin=-180, vmax=180)
    
    plt.title('Phase Shift (degrees)', fontsize=16, pad=20)
    plt.ylabel('Scenario', fontsize=14)
    plt.xlabel('Bedrock Wavelength (km)', fontsize=14)
    
    # Custom ticks and labels
    plt.yticks(range(len(scenarios)), scenarios, fontsize=12)
    plt.xticks(range(len(wavelengths)), [f"{w:.1f}" for w in wavelengths], 
               rotation=45, fontsize=10)
    
    # Add colorbar with better formatting
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Phase Shift (degrees)', rotation=270, labelpad=20, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Add text annotations for values
    for i in range(len(scenarios)):
        for j in range(len(wavelengths)):
            if not np.isnan(phase_matrix[i, j]):
                text = plt.text(j, i, f'{phase_matrix[i, j]:.0f}°',
                               ha="center", va="center", 
                               color="white" if abs(phase_matrix[i, j]) > 90 else "black",
                               fontsize=12)
    
    plt.tight_layout()
    plt.savefig("transfer_plots/phase_shift_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Matrix plots saved:")
    print("  - transfer_plots/amplitude_ratio_matrix.png (detailed)")
    print("  - transfer_plots/phase_shift_matrix.png (detailed)")


if __name__ == "__main__":

    results = analyse_bed_surface_transfer()
    print("Analysis complete!")
