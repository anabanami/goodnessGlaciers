"""
Phase Analysis Script for ISSM Simulation Output

Usage:
    python phase_analysis.py <path_to_transient_results.nc> [--axis {x,y}] [--position <value>]

Description:
    This script performs a phase relationship analysis between the bedrock
    topography and the ice surface elevation over time, based on the output
    of an ISSM transient simulation.

    It extracts a 1D data profile along a specified line and calculates the
    phase shift between the bed and surface signals for each time step.

    Arguments:
      <path_to_transient_results.nc>  : Required. Path to the input NetCDF file.
      --axis {x,y}                    : Optional. The axis along which to extract the
                                        data profile. Defaults to 'x'.
      --position <value>              : Optional. The coordinate on the other axis
                                        where the profile is taken. Defaults to the
                                        center of the domain.

Example Usage:
    # Analyze the default centerline (x-axis profile at the domain's y-center)
    python phase_analysis.py IsmipF_S1_30-Transient.nc

    # Analyze a specific off-center profile along the x-axis
    python phase_analysis.py IsmipF_S1_30-Transient.nc --axis x --position 25000

    # Analyze a profile along the y-axis
    python phase_analysis.py IsmipF_S1_30-Transient.nc --axis y --position 40000
"""
import os
import sys
import time
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from scipy import signal
from scipy.signal import correlate

# Add ISSM/pyISSM to the Python path.
# Please ensure this path is correct for your system.
sys.path.append('/home/ana/pyISSM/src')
from model import model
from squaremesh import squaremesh
from parameterize import parameterize
from SetIceSheetBC import SetIceSheetBC


def parse_filename(filename):
    """
    Parses the input NetCDF filename to extract simulation parameters.
    Example: 'IsmipF_S1_2_2-Transient.nc' -> ('IsmipF', 'S1', 2.0, 2.0)
    """
    base_name = os.path.splitext(os.path.basename(filename))[0]
    parts = base_name.split('_')
    
    if len(parts) < 4:
        raise ValueError(f"Filename '{filename}' does not match the expected format 'NAME_SCENARIO_HRES_VRES-TYPE.nc'")

    param_profile = parts[0]
    scenario = parts[1]
    try:
        h_resolution_factor = float(parts[2])
        # The vertical resolution factor is the first part of the last segment
        v_resolution_factor = float(parts[3].split('-')[0])
    except (ValueError, IndexError):
        print(f"Warning: Could not determine resolution factors from '{base_name}'. Defaulting to 1.0.")
        h_resolution_factor = 2
        v_resolution_factor = 2
        
    print(f"✅ Parsed Filename: Profile='{param_profile}', Scenario='{scenario}', H-Res='{h_resolution_factor}', V-Res='{v_resolution_factor}'")
    return param_profile, scenario, h_resolution_factor, v_resolution_factor


def reconstruct_mesh(param_profile, scenario, h_res_factor, v_res_factor):
    """
    Reconstructs the 3D model mesh based on the filename parameters
    by replicating the setup process from the 'runme.py' script.
    """
    print("\nReconstructing model mesh...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    param_filename = f"{param_profile}.py"
    param_file_path = os.path.join(parent_dir, param_filename)
    
    print(f"  Using Parameter File: '{param_file_path}'")
    
    if not os.path.exists(param_file_path):
        raise FileNotFoundError(f"Could not find parameter file at '{param_file_path}'!")

    # Replicate the meshing and extrusion steps from runme.py
    md = model()
    x_max = 100000
    y_max = 100000
    
    # Calculate nodes based on horizontal resolution factor
    x_nodes = int(30 * h_res_factor)
    y_nodes = int(30 * h_res_factor)
    
    md = squaremesh(md, x_max, y_max, x_nodes, y_nodes)
    
    # Set the required miscellaneous attributes before parameterization
    md.miscellaneous.filename = param_profile
    md.miscellaneous.scenario = scenario
    md.miscellaneous.h_resolution_factor = h_res_factor
    md.miscellaneous.v_resolution_factor = v_res_factor
    
    md = parameterize(md, param_file_path)
    
    # Calculate layers based on vertical resolution factor
    base_vertical_layers = 5
    num_layers = int(base_vertical_layers * v_res_factor)
    md = md.extrude(num_layers, 1)

    print(f"✅ Mesh reconstructed successfully ({md.mesh.numberofvertices} vertices, {num_layers} layers).")
    return md


def load_config_by_parsing(param_profile):
    """
    Loads key parameters by parsing the parameter file as plain text.
    This is a robust method that avoids executing the file in a complex
    mock environment, thus preventing AttributeError issues.
    """
    print(f"\nLoading configuration by parsing '{param_profile}.py'...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    param_filename = f"{param_profile}.py"
    param_file_path = os.path.join(parent_dir, param_filename)
    
    if not os.path.exists(param_file_path):
        raise FileNotFoundError(f"Could not find parameter file at '{param_file_path}'!")
        
    local_vars = {}
    # Regex to find lines like 'variable = value'
    assignment_re = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.*)")
    
    with open(param_file_path, 'r') as f:
        lines = f.readlines()

    # We now need alpha and H_0 to reconstruct the baseline
    target_vars = ['H_0', 'sigma', 'amplitude_0', 'alpha']
    
    for line in lines:
        if line.strip().startswith('#'):
            continue
        
        match = assignment_re.match(line)
        if match:
            var_name = match.group(1).strip()
            expression = match.group(2).split('#')[0].strip()
            
            if var_name in target_vars:
                try:
                    local_vars[var_name] = eval(expression, {"np": np}, local_vars)
                except (NameError, TypeError):
                    pass


    config = {
        'H_0': local_vars.get('H_0', 1000),
        'sigma': local_vars.get('sigma', 10000),
        'amplitude_0': local_vars.get('amplitude_0', 100),
        'alpha': local_vars.get('alpha', -3.0) # degrees
    }
    
    if 'sigma' not in local_vars:
        raise ValueError(f"Could not parse 'sigma' from '{param_file_path}'. Ensure it is defined as a simple numerical assignment.")

    # The characteristic "wavelength" is assumed to be the Gaussian width 'sigma'.
    wavelength = config['sigma']
    
    print(f"✅ Config loaded: Wavelength (sigma) = {wavelength} m, Slope (alpha) = {config['alpha']} deg")
    return config, wavelength


def get_profile_data(md, dataset, time_step_index, axis, position, config):
    """
    Extracts data for a specific time step along a specified profile line
    and calculates the analytical unperturbed baseline.
    """
    tsol = dataset['results/TransientSolution']
    time_val = tsol['time'][time_step_index]
    
    surface_full = np.squeeze(tsol['Surface'][time_step_index, :])
    base_full = np.squeeze(tsol['Base'][time_step_index, :])

    if axis == 'x':
        profile_coords = md.mesh.x
        slice_coords = md.mesh.y
    else:
        profile_coords = md.mesh.y
        slice_coords = md.mesh.x

    tolerance = 1e-5
    profile_indices = np.where(np.abs(slice_coords - position) < tolerance)[0]
    
    if len(profile_indices) == 0:
        return np.array([]), np.array([]), np.array([]), time_val, np.array([]), np.array([])

    coords_on_profile = profile_coords[profile_indices]
    sorted_order = np.argsort(coords_on_profile)
    sorted_indices = profile_indices[sorted_order]
    
    x_profile = profile_coords[sorted_indices]
    surface_profile = surface_full[sorted_indices]
    base_profile = base_full[sorted_indices]

    alpha_rad = config['alpha'] * np.pi / 180.0
    H_0 = config['H_0']
    
    if axis == 'x':
        unperturbed_surface = x_profile * np.tan(alpha_rad)
    else:
        unperturbed_surface = np.zeros_like(x_profile)
        
    unperturbed_base = unperturbed_surface - H_0
    
    return x_profile, base_profile, surface_profile, time_val, unperturbed_base, unperturbed_surface


def phase_shift_analysis(x, base_signal, surface_signal, wavelength, time_val=None):
    """
    Calculates the phase shift between the bed and surface signals using
    cross-correlation.
    """
    if len(x) < 2 or np.std(base_signal) == 0 or np.std(surface_signal) == 0:
        return 0, 0
    
    dx = np.mean(np.diff(x))
    
    bed_norm = (base_signal - np.mean(base_signal)) / np.std(base_signal)
    surface_norm = (surface_signal - np.mean(surface_signal)) / np.std(surface_signal)
    
    xcorr = correlate(bed_norm, surface_norm, mode='full')
    
    max_corr_idx = np.argmax(xcorr)
    center_idx = len(xcorr) // 2
    shift_idx = max_corr_idx - center_idx
    
    lag_distance = shift_idx * dx
    
    phase_shift_rad = (2 * np.pi * lag_distance) / wavelength
    phase_shift_rad = np.angle(np.exp(1j * phase_shift_rad))
    
    phase_shift_deg = np.degrees(phase_shift_rad)

    return phase_shift_deg, lag_distance


def plot_signals(x, base_signal, surface_signal, time_val, wavelength, output_path, axis, position, config):
    """
    Plots the isolated bed and surface signals, with the surface vertically
    offset by the reference ice thickness H_0 for clarity.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    other_axis = 'Y' if axis == 'x' else 'X'
    
    # Get the reference ice thickness from the config dictionary
    H_0 = config.get('H_0', 1000) # Use .get() for safety, with a default value

    # Plot the bed signal, and the surface signal offset vertically by H_0
    ax.plot(x / 1000, base_signal, 'k-', label='Bed Signal')
    ax.plot(x / 1000, surface_signal + H_0, 'c-', label=f'Surface Signal (Offset by H = {H_0} m)')
    ax.fill_between(x / 1000, base_signal, surface_signal + H_0, alpha=0.3, label='Ice thickness')
    ax.set_xlabel(f'Distance along {axis.upper()}-axis (km)')
    ax.set_ylabel('Relative Elevation (m)')
    ax.set_title(f'Isolated Signals at t={time_val:.2f}yr (Profile at {other_axis}={position/1000:.1f}km)')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend()
    
    # Set x-limits to show the entire profile domain.
    if len(x) > 0:
        ax.set_xlim(x.min() / 1000, x.max() / 1000)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_cross_correlation(x, base_signal, surface_signal, wavelength, time_val, output_path, axis, position):
    """Visualizes the cross-correlation between the two signals."""
    if len(x) < 2 or np.std(base_signal) == 0 or np.std(surface_signal) == 0:
        return
        
    dx = np.mean(np.diff(x))
    other_axis = 'Y' if axis == 'x' else 'X'
    
    bed_norm = (base_signal - np.mean(base_signal)) / np.std(base_signal)
    surface_norm = (surface_signal - np.mean(surface_signal)) / np.std(surface_signal)
    
    xcorr = correlate(bed_norm, surface_norm, mode='full')
    lags = np.arange(-(bed_norm.size - 1), bed_norm.size)
    lag_distances = lags * dx
    
    max_corr_idx = np.argmax(xcorr)
    lag_at_max = lag_distances[max_corr_idx]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(lag_distances / 1000, xcorr, 'b-')
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.6, label='Zero Lag')
    ax.axvline(x=lag_at_max / 1000, color='r', linestyle='--', label=f'Max Correlation Lag ({lag_at_max/1000:.2f} km)')

    ax.set_title(f'Cross-Correlation at t={time_val:.2f}yr (Profile at {other_axis}={position/1000:.1f}km)')
    ax.set_xlabel('Lag Distance (km)')
    ax.set_ylabel('Cross-Correlation Coefficient')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend()
    
    ax.set_xlim(-1.5 * wavelength / 1000, 1.5 * wavelength / 1000)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    

def analyse_phase_evolution(md, dataset, wavelength, output_dir, axis, position, config):
    """
    Main analysis function. Iterates through all time steps, performs
    phase analysis, generates plots, and returns summary statistics.
    """
    tsol = dataset['results/TransientSolution']
    times_sec = tsol.variables['time'][:]
    num_steps = len(times_sec)

    print(f"\nAnalysing phase evolution across {num_steps} time steps...")
    
    plot_dirs = {'signals': '1_Signals', 'correlation': '2_Cross_Correlation', 'evolution': '3_Evolution_Plots'}
    for subdir in plot_dirs.values():
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    results = {
        'times': [],
        'phase_shifts_deg': [],
        'lag_distances': []
    }

    for i in range(num_steps):
        progress = (i + 1) / num_steps * 100
        print(f"\r  Processing time step {i+1}/{num_steps} [{progress:3.0f}%]", end="")
        
        x, base, surface, time_val, unperturbed_base, unperturbed_surface = get_profile_data(md, dataset, i, axis, position, config)
        
        if len(x) < 2:
            print(f"\nWarning: Not enough data points on profile for step {i}. Skipping.")
            results['times'].append(time_val)
            results['phase_shifts_deg'].append(np.nan)
            results['lag_distances'].append(np.nan)
            continue
        
        base_signal = base - unperturbed_base
        surface_signal = surface - unperturbed_surface
            
        phase_deg, lag_dist = phase_shift_analysis(x, base_signal, surface_signal, wavelength, time_val)
        
        results['times'].append(time_val)
        results['phase_shifts_deg'].append(phase_deg)
        results['lag_distances'].append(lag_dist)

        signal_plot_path = os.path.join(output_dir, plot_dirs['signals'], f'signals_t_{i:04d}.png')
        plot_signals(x, base_signal, surface_signal, time_val, wavelength, signal_plot_path, axis, position, config)

        corr_plot_path = os.path.join(output_dir, plot_dirs['correlation'], f'correlation_t_{i:04d}.png')
        plot_cross_correlation(x, base_signal, surface_signal, wavelength, time_val, corr_plot_path, axis, position)

    print("\n✅ Analysis of all time steps complete.")

    print("Generating evolution plots...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    other_axis = 'Y' if axis == 'x' else 'X'
    
    ax1.plot(results['times'], results['phase_shifts_deg'], 'b.-', label='Measured Phase Shift')
    ax1.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Phase Shift (degrees)')
    ax1.set_title(f'Evolution of Phase Shift (Profile at {other_axis}={position/1000:.1f}km)')
    ax1.grid(True, linestyle=':')
    ax1.legend()

    ax2.plot(results['times'], np.array(results['lag_distances']) / 1000, 'r.-', label='Measured Lag')
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (years)')
    ax2.set_ylabel('Lag Distance (km)')
    ax2.set_title(f'Evolution of Spatial Lag (Profile at {other_axis}={position/1000:.1f}km)')
    ax2.grid(True, linestyle=':')
    ax2.legend()
    
    plt.tight_layout()
    evo_plot_path = os.path.join(output_dir, plot_dirs['evolution'], 'phase_evolution_summary.png')
    fig.savefig(evo_plot_path, dpi=300)
    plt.close(fig)
    print("✅ Evolution plots saved.")
    
    return results
    

def write_summary_file(output_dir, nc_file, param_profile, scenario, h_resolution, v_resolution, wavelength, results, axis, position):
    """Writes a text file summarizing the analysis results."""
    summary_path = os.path.join(output_dir, 'summary.txt')
    other_axis = 'Y' if axis == 'x' else 'X'
    
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("            Phase Analysis Summary Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Timestamp: {time.ctime()}\n")
        f.write(f"Source File: {nc_file}\n\n")
        f.write("-- Simulation Parameters --\n")
        f.write(f"  Parameter Profile: {param_profile}\n")
        f.write(f"  Scenario: {scenario}\n")
        f.write(f"  Horizontal Resolution Factor: {h_resolution}\n")
        f.write(f"  Vertical Resolution Factor: {v_resolution}\n")
        f.write(f"  Bedrock Wavelength (sigma): {wavelength/1000:.2f} km\n\n")
        f.write("-- Analysis Profile --\n")
        f.write(f"  Profile Axis: {axis.upper()}\n")
        f.write(f"  Profile Position: {other_axis} = {position/1000:.2f} km\n\n")
        f.write("-- Analysis Results --\n")
        f.write("Time-dependent phase shift and spatial lag:\n")
        f.write("-------------------------------------------\n")
        f.write(f"{'Time (yr)':>12} | {'Phase (deg)':>14} | {'Lag (km)':>12}\n")
        f.write(f"{'-'*12} | {'-'*14} | {'-'*12}\n")

        for i in range(len(results['times'])):
            time_val = results['times'][i]
            phase_val = results['phase_shifts_deg'][i]
            lag_val = results['lag_distances'][i] / 1000
            f.write(f"{time_val:12.3f} | {phase_val:14.2f} | {lag_val:12.3f}\n")
    
    print(f"✅ Summary report written to '{summary_path}'")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Phase Analysis Script for ISSM Simulation Output.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('nc_file', type=str, help="Path to the input NetCDF file.")
    parser.add_argument('--axis', type=str, choices=['x', 'y'], default='x',
                        help="The axis along which to extract the data profile. Defaults to 'x'.")
    parser.add_argument('--position', type=float, default=None,
                        help="Coordinate on the other axis for the profile. Defaults to the domain center.")
    
    args = parser.parse_args()

    if not os.path.exists(args.nc_file):
        print(f"Error: Input file not found at '{args.nc_file}'")
        sys.exit(1)

    start_time = time.time()
    print(f"\n{'='*60}\nStarting Phase Analysis for: {os.path.basename(args.nc_file)}\n{'='*60}")

    try:
        param_profile, scenario, h_res, v_res = parse_filename(args.nc_file)

        output_dir_name = f"{os.path.splitext(os.path.basename(args.nc_file))[0]}_phase_analysis"
        os.makedirs(output_dir_name, exist_ok=True)
        print(f"Results will be saved in: '{output_dir_name}'")

        md = reconstruct_mesh(param_profile, scenario, h_res, v_res)
        config, wavelength = load_config_by_parsing(param_profile)
        
        requested_position = args.position
        if requested_position is None:
            if args.axis == 'x':
                requested_position = np.nanmax(md.mesh.y) / 2.0
            else:
                requested_position = np.nanmax(md.mesh.x) / 2.0
            print(f"No position specified, defaulting to center: {requested_position:.1f} m")

        if args.axis == 'x':
            all_coords_on_slice_axis = md.mesh.y
        else:
            all_coords_on_slice_axis = md.mesh.x
            
        unique_coords = np.unique(all_coords_on_slice_axis)
        closest_coord_index = np.argmin(np.abs(unique_coords - requested_position))
        analysis_position = unique_coords[closest_coord_index]

        if not np.isclose(requested_position, analysis_position, atol=1e-3):
            print(f"  -> Requested position was {requested_position:.1f} m.")
            print(f"  -> Using closest available mesh line at {analysis_position:.1f} m.")
        else:
            print(f"Using profile at position: {analysis_position:.1f} m")

        with nc.Dataset(args.nc_file, 'r') as dataset:
            results = analyse_phase_evolution(md, dataset, wavelength, output_dir_name, args.axis, analysis_position, config)

        write_summary_file(output_dir_name, args.nc_file, param_profile, scenario, h_res, v_res, wavelength, results, args.axis, analysis_position)
        
    except Exception as e:
        print(f"\n❌ An error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    end_time = time.time()
    print(f"\n{'='*60}\n🎉 Analysis complete in {end_time - start_time:.2f} seconds.\n{'='*60}")


if __name__ == '__main__':
    main()