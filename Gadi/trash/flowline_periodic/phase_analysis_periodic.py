#!/usr/bin/env python
"""
Phase Analysis for Periodic Boundary Conditions - Direct NetCDF Access
This script analyzes the phase relationship between bed topography and surface elevation
over time in an ISSM NetCDF file with periodic boundary conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from scipy import signal
from scipy.signal import correlate
import os
import sys


def bandpass_filter(x, dx, signal_to_filter, wavelength):
    """Apply bandpass filter based on the bedrock wavelength from config"""
    print(f"\n=== Filtering frequencies ===")
    nyquist = 1 / (2 * dx)  # cycles/km
    
    # Set filter window around the bedrock wavelength
    window_fraction = 0.5  # Allow ±50% around main wavelength
    wavelength_min = wavelength * (1 - window_fraction)
    wavelength_max = wavelength * (1 + window_fraction)
    
    freq_min = 1 / wavelength_max  # cycles/km
    freq_max = 1 / wavelength_min  # cycles/km
    
    # Normalise by Nyquist
    low = freq_min / nyquist
    high = freq_max / nyquist
    
    print(f"Bedrock wavelength: {wavelength} km")
    print(f"Filter window: {wavelength_min:.2f} km to {wavelength_max:.2f} km")
    print(f"Frequencies: {freq_min:.3f} to {freq_max:.3f} cycles/km")
    
    # Apply Butterworth filter
    order = 4
    b, a = signal.butter(order, [low, high], btype='band')
    signal_filtered = signal.filtfilt(b, a, signal_to_filter)
    
    return signal_filtered


def phase_shift_analysis(dx, base, surface, wavelength, time_val=None):
    """Calculate phase shift between base and surface signals with improved precision"""
    
    if time_val is not None:
        print(f"~~~Comparing base and surface at t={time_val:.1f} years~~~")
    else:
        print(f"~~~Comparing base and surface~~~")

    # Normalise signals before correlation
    base_norm = (base - np.mean(base)) / np.std(base)
    surface_norm = (surface - np.mean(surface)) / np.std(surface)
    
    # Calculate cross-correlation
    xcorr = correlate(base_norm, surface_norm, mode='full')
    nsamples = base.size
    
    # Create lag array
    lags = np.arange(-(nsamples-1), nsamples)
    
    # Find the lag with maximum correlation
    max_corr_idx = np.argmax(xcorr)
    
    # Calculate the center index (zero lag position)
    center_idx = len(xcorr) // 2
    
    # Calculate the shift relative to center (positive = surface lags behind bed)
    shift_idx = max_corr_idx - center_idx
    
    # Convert lag to distance
    lag_distance = shift_idx * dx
    
    # Use a more precise phase calculation
    # Limit to ±half wavelength to avoid jumping to next peak
    if abs(lag_distance) > wavelength/2:
        # Find the remainder when divided by wavelength
        lag_distance = lag_distance % wavelength
        # Ensure it's in the range [-wavelength/2, wavelength/2]
        if lag_distance > wavelength/2:
            lag_distance -= wavelength
    
    # Calculate phase shift in radians (positive = surface lags behind bed)
    phase_shift = (2 * np.pi * lag_distance) / wavelength
    
    # Convert to degrees
    phase_shift_deg = (phase_shift * 180) / np.pi
    
    # Calculate correlation coefficient
    max_corr = xcorr[max_corr_idx] / (nsamples * np.std(base) * np.std(surface))
    
    print(f"Maximum correlation at raw index: {max_corr_idx}, center at: {center_idx}")
    print(f"Shift from center: {shift_idx} indices")
    print(f"Maximum correlation: {max_corr:.3f}")
    print(f"Spatial shift: {lag_distance:.3f} km")
    print(f"Phase shift: {phase_shift/np.pi:.2f}π radians or {phase_shift_deg:.1f} degrees")
    
    return phase_shift, lag_distance, max_corr


def visualise_cross_correlation(x, base_filtered, surface_filtered, wavelength, time_val, output_dir=None):
    """Create a dedicated visualization of cross-correlation between bed and surface"""
    plt.figure(figsize=(12, 6))
    
    # Normalise signals for correlation
    base_norm = (base_filtered - np.mean(base_filtered)) / np.std(base_filtered)
    surf_norm = (surface_filtered - np.mean(surface_filtered)) / np.std(surface_filtered)
    
    # Calculate cross-correlation
    xcorr = correlate(base_norm, surf_norm, mode='full')
    nsamples = base_norm.size
    
    # Create lag array
    lags = np.arange(-(nsamples-1), nsamples)
    
    # Calculate spacing
    dx = np.mean(np.diff(x))
    
    # Convert lags to distances
    lag_distances = lags * dx
    
    # Find the lag with maximum correlation (global maximum)
    max_corr_idx = np.argmax(xcorr)
    
    # Calculate the center index (zero lag position)
    center_idx = len(xcorr) // 2
    
    # Calculate the shift from center
    shift_idx = max_corr_idx - center_idx
    
    # Convert lag to distance
    lag_distance = shift_idx * dx
    
    # Use a more precise phase calculation
    # Limit to ±half wavelength to avoid jumping to next peak
    if abs(lag_distance) > wavelength/2:
        # Find the remainder when divided by wavelength
        lag_distance = lag_distance % wavelength
        # Ensure it's in the range [-wavelength/2, wavelength/2]
        if lag_distance > wavelength/2:
            lag_distance -= wavelength
    
    # Calculate phase shift
    phase_shift = (2 * np.pi * lag_distance) / wavelength
    phase_deg = phase_shift * 180 / np.pi
    
    # Calculate correlation coefficient
    max_corr = xcorr[max_corr_idx] / (nsamples * np.std(base_filtered) * np.std(surface_filtered))
    
    # Only show correlation near center for clarity
    xlim = 1.5 * wavelength  # Show +/- 1.5 wavelengths
    visible_lags = np.where((lag_distances >= -xlim) & (lag_distances <= xlim))[0]
    
    # Plot correlation
    plt.plot(lag_distances[visible_lags], xcorr[visible_lags], 'b-', linewidth=2)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Zero lag')
    
    # Plot the normalised lag distance that's within ±half wavelength
    plt.axvline(x=lag_distance, color='purple', linestyle='--', linewidth=2,
               label=f'Maximum correlation: {lag_distance:.2f} km')
    
    # Add theoretical phase lines
    theoretical_lag = (90/360) * wavelength  # Positive 90° lag in km
    plt.axvline(x=theoretical_lag, color='g', linestyle='--', linewidth=1.5,
               label=f'Theoretical (90°): {theoretical_lag:.2f} km')
    
    # Add negative theoretical phase line
    plt.axvline(x=-theoretical_lag, color='r', linestyle='--', linewidth=1.5,
               label=f'Opposite (-90°): {-theoretical_lag:.2f} km')
    
    plt.xlabel('Lag distance (km)')
    plt.ylabel('Cross-correlation')
    plt.title(f'Cross-correlation: bed and surface at t={time_val:.1f} years')
    
    # Position the legend in the upper right corner
    plt.legend(loc='upper right')
    
    plt.grid(True, linestyle=':', c="grey")
    
    # Add annotation about the phase in the upper left
    plt.annotate(f'Measured phase shift: {phase_deg:.1f}° (lag: {lag_distance:.3f} km)',
                xy=(0.02, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                fontsize=10)
    
    plt.tight_layout()
    
    if output_dir:
        filename = os.path.join(output_dir, f"bed_surface_correlation_t{time_val:.1f}.png")
        plt.savefig(filename, dpi=300)
        print(f"Saved bed-surface correlation plot to {filename}")
    
    return phase_deg, lag_distance


def plot_signals(x, base, surface, base_filtered, surface_filtered, time_val, output_dir=None, ylim_main=None, ylim_filtered=None):
    """Plot the base and surface elevation signals with consistent y-axis limits"""
    # Remove mean for comparison
    base_prof = base - np.mean(base)
    surface_prof = surface - np.mean(surface)
    
    plt.figure(figsize=(12, 8))
    
    # Plot unfiltered data
    plt.subplot(2, 1, 1)
    plt.plot(x, base_prof, 'k-', label='Bed topography')
    plt.plot(x, surface_prof, 'b-', label='Surface elevation')
    plt.ylabel('Elevation (km)')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='upper right')
    
    # Apply consistent y-limits if provided
    if ylim_main is not None:
        plt.ylim(ylim_main)
    
    title = 'Bed vs Surface (mean removed)'
    if time_val is not None:
        title += f' at t={time_val:.1f} years'
    plt.title(title)
    
    # Plot filtered data
    base_filtered_prof = base_filtered - np.mean(base_filtered)
    surface_filtered_prof = surface_filtered - np.mean(surface_filtered)
    
    plt.subplot(2, 1, 2)
    plt.plot(x, base_filtered_prof, 'k-', label='Filtered bed')
    plt.plot(x, surface_filtered_prof, 'b-', label='Filtered surface')
    plt.xlabel('Distance (km)')
    plt.ylabel('Filtered elevation (km)')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='upper right')
    plt.title('Bandpass filtered signals (mean removed)')
    
    # Apply consistent y-limits to filtered plot if provided
    if ylim_filtered is not None:
        plt.ylim(ylim_filtered)
    
    plt.tight_layout()
    
    if output_dir:
        if time_val is not None:
            filename = os.path.join(output_dir, f"signals_t{time_val:.1f}.png")
        else:
            filename = os.path.join(output_dir, "signals.png")
        plt.savefig(filename, dpi=300)
        print(f"Saved signals plot to {filename}")


def visualise_phase_relationship(x, base_filtered, surface_filtered, wavelength, time_val, output_dir=None):
    """Create a dedicated visualization of phase relationship between signals"""
    plt.figure(figsize=(14, 8))
    
    # Normalise signals for comparison
    base_norm = (base_filtered - np.mean(base_filtered)) / np.std(base_filtered)
    surf_norm = (surface_filtered - np.mean(surface_filtered)) / np.std(surface_filtered)
    
    # Plot normalised signals
    plt.plot(x, base_norm, 'k-', label='Bed (normalised)', linewidth=2)
    plt.plot(x, surf_norm, 'b-', label='Surface (normalised)', linewidth=2)
    
    # Find peaks for visual reference
    bed_peaks = signal.find_peaks(base_norm)[0]
    surf_peaks = signal.find_peaks(surf_norm)[0]
    
    # Draw peak markers
    plt.plot(x[bed_peaks], base_norm[bed_peaks], 'ko', markersize=8)
    plt.plot(x[surf_peaks], surf_norm[surf_peaks], 'bo', markersize=8)
    
    # Draw vertical guides at bed peaks
    for bp in bed_peaks:
        plt.axvline(x=x[bp], color='gray', linestyle='--', alpha=0.3)
    
    # Calculate phase shift information
    dx = np.mean(np.diff(x))
    xcorr = correlate(base_norm, surf_norm, mode='full')
    max_corr_idx = np.argmax(xcorr)
    center_idx = len(xcorr) // 2
    shift_idx = max_corr_idx - center_idx
    lag_distance = shift_idx * dx
    phase_shift = (2 * np.pi * lag_distance) / wavelength
    phase_deg = phase_shift * 180 / np.pi
    
    # Calculate theoretical values for comparison
    quarter_wavelength = wavelength / 4
    theoretical_phase = 90  # degrees
    
    # Add annotation showing the phase information
    plt.annotate(f'Measured phase shift: {phase_deg:.1f}°\nLag distance: {lag_distance:.3f} km\n'
                f'Theoretical phase: {theoretical_phase}°\nWavelength: {wavelength:.2f} km',
                xy=(0.02, 0.02), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                fontsize=10)
    
    # Add a title with time information
    plt.title(f'Phase relationship between bed and surface at t={time_val:.1f} years', fontsize=14)
    plt.xlabel('Distance (km)', fontsize=12)
    plt.ylabel('Normalised amplitude', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12)
    
    # Set limits to show 3 wavelengths
    xlim_start = x.min()
    xlim_end = min(x.min() + 3*wavelength, x.max())
    plt.xlim(xlim_start, xlim_end)
    
    plt.tight_layout()
    
    if output_dir:
        filename = os.path.join(output_dir, f"phase_relationship_t{time_val:.1f}.png")
        plt.savefig(filename, dpi=300)
        print(f"Saved phase relationship plot to {filename}")
    
    return phase_deg, lag_distance


def load_time_dependent_data_from_netcdf(dataset, time_step):
    """Load base and surface data for a specific time step directly from NetCDF"""
    print(f"Loading data for time step {time_step}")
    
    # Get coordinates from mesh group
    x = dataset.groups['mesh'].variables['x'][:]
    
    # Get base and surface vertices
    base_indices = np.where(dataset.groups['mesh'].variables['vertexonbase'][:] == 1)[0]
    surf_indices = np.where(dataset.groups['mesh'].variables['vertexonsurface'][:] == 1)[0]
    
    x_base = x[base_indices]
    x_surf = x[surf_indices]
    
    # Get elevations from TransientSolution
    transient = dataset.groups['results'].groups['TransientSolution']
    base = transient.variables['Base'][time_step, base_indices]
    surface = transient.variables['Surface'][time_step, surf_indices]
    
    # Get time value
    time_val = float(transient.variables['time'][time_step])
    
    # Sort arrays by x coordinate
    base_sort_idx = np.argsort(x_base)
    surf_sort_idx = np.argsort(x_surf)
    
    x_base = x_base[base_sort_idx]
    x_surf = x_surf[surf_sort_idx]
    base = base[base_sort_idx]
    surface = surface[surf_sort_idx]
    
    # Interpolate to ensure matching x coordinates
    if len(x_base) != len(x_surf):
        print(f"Interpolating to match coordinates (base: {len(x_base)}, surface: {len(x_surf)} points)")
        if len(x_base) < len(x_surf):
            surface = np.interp(x_base, x_surf, surface)
            x = x_base
        else:
            base = np.interp(x_surf, x_base, base)
            x = x_surf
    else:
        x = x_base
    
    # Remove edges to avoid boundary effects
    buffer = 2.0  # km from each edge
    valid_idx = (x >= min(x) + buffer) & (x <= max(x) - buffer)
    if sum(valid_idx) > 10:  # Ensure we have enough points left
        print(f"Removing edge points within {buffer} km of boundaries")
        x = x[valid_idx]
        base = base[valid_idx]
        surface = surface[valid_idx]
    
    return x, base, surface, time_val


def analyze_phase_over_time_from_netcdf(dataset, wavelength, output_dir=None):
    """Analyze phase shift between bed and surface for all time steps with improved visualizations"""
    # Get number of time steps
    transient = dataset.groups['results'].groups['TransientSolution']
    num_steps = transient.variables['time'].shape[0]
    print(f"Analyzing {num_steps} time steps")
    
    # Create arrays to store results
    time_values = []
    phase_shifts = []
    lag_distances = []
    correlations = []
    
    # Create output directories if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        signals_dir = os.path.join(output_dir, "signals")
        corr_dir = os.path.join(output_dir, "correlations")
        phase_dir = os.path.join(output_dir, "phase_relationship")
        os.makedirs(signals_dir, exist_ok=True)
        os.makedirs(corr_dir, exist_ok=True)
        os.makedirs(phase_dir, exist_ok=True)
    else:
        signals_dir = None
        corr_dir = None
        phase_dir = None
    
    # First pass: determine global min/max values
    print("\n=== First pass: determining global y-axis limits ===")
    all_base_vals = []
    all_surface_vals = []
    all_base_filtered_vals = []
    all_surface_filtered_vals = []
    
    for step in range(num_steps):
        print(f"Analyzing limits for time step {step+1}/{num_steps}")
        
        # Load data for this time step
        x, base, surface, time_val = load_time_dependent_data_from_netcdf(dataset, step)
        
        # Calculate dx (spacing between points)
        dx = np.mean(np.diff(x))
        
        # Apply bandpass filter
        base_filtered = bandpass_filter(x, dx, base, wavelength)
        surface_filtered = bandpass_filter(x, dx, surface, wavelength)
        
        # Store values for limit calculation (after mean removal)
        all_base_vals.extend((base - np.mean(base)).tolist())
        all_surface_vals.extend((surface - np.mean(surface)).tolist())
        all_base_filtered_vals.extend((base_filtered - np.mean(base_filtered)).tolist())
        all_surface_filtered_vals.extend((surface_filtered - np.mean(surface_filtered)).tolist())
    
    # Calculate global y-limits with some padding
    main_min = min(min(all_base_vals), min(all_surface_vals))
    main_max = max(max(all_base_vals), max(all_surface_vals))
    filtered_min = min(min(all_base_filtered_vals), min(all_surface_filtered_vals))
    filtered_max = max(max(all_base_filtered_vals), max(all_surface_filtered_vals))
    
    # Add 10% padding
    padding_main = 0.1 * (main_max - main_min)
    padding_filtered = 0.1 * (filtered_max - filtered_min)
    
    ylim_main = (main_min - padding_main, main_max + padding_main)
    ylim_filtered = (filtered_min - padding_filtered, filtered_max + padding_filtered)
    
    print(f"Global y-limits for main plots: {ylim_main}")
    print(f"Global y-limits for filtered plots: {ylim_filtered}")
    
    # Second pass: actual analysis with consistent y-limits
    print("\n=== Second pass: processing with consistent y-axis limits ===")
    
    # Loop over all time steps
    for step in range(num_steps):
        print(f"\n=== Processing time step {step+1}/{num_steps} ===")
        
        # Load data for this time step
        x, base, surface, time_val = load_time_dependent_data_from_netcdf(dataset, step)
        
        # Calculate dx (spacing between points)
        dx = np.mean(np.diff(x))
        
        # Apply bandpass filter
        base_filtered = bandpass_filter(x, dx, base, wavelength)
        surface_filtered = bandpass_filter(x, dx, surface, wavelength)
        
        # Plot signals with consistent y-limits
        if signals_dir:
            plot_signals(x, base, surface, base_filtered, surface_filtered, 
                       time_val, signals_dir, ylim_main, ylim_filtered)
        
        # Generate enhanced phase relationship visualization
        if phase_dir:
            visualise_phase_relationship(x, base_filtered, surface_filtered, 
                                        wavelength, time_val, phase_dir)
        
        # Create cross-correlation visualization
        if corr_dir:
            visualise_cross_correlation(x, base_filtered, surface_filtered,
                                       wavelength, time_val, corr_dir)
        
        # Analyze phase shift
        phase_shift, lag, corr = phase_shift_analysis(dx, base_filtered, surface_filtered, wavelength, time_val)
        
        # Store results
        time_values.append(time_val)
        phase_shifts.append(phase_shift)
        lag_distances.append(lag)
        correlations.append(corr)
    
    # Plot enhanced evolution of phase shift over time
    plt.figure(figsize=(14, 10))
    
    # Convert phase shift to degrees for plotting
    phase_shifts_deg = np.array(phase_shifts) * 180 / np.pi
    
    # Plot phase shift evolution
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
    plt.legend(loc='best')
    
    # Plot lag distance evolution
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
    plt.legend(loc='best')
    
    plt.tight_layout()
    
    if output_dir:
        filename = os.path.join(output_dir, "phase_evolution.png")
        plt.savefig(filename, dpi=300)
        print(f"Saved enhanced phase evolution plot to {filename}")
    
    # Create a summary dictionary
    results = {
        'time': time_values,
        'phase_shift': phase_shifts,
        'phase_shift_deg': phase_shifts_deg,
        'lag_distance': lag_distances,
        'correlation': correlations,
        'ylim_main': ylim_main,
        'ylim_filtered': ylim_filtered
    }
    
    return results


def main():
    print(f"\n=== Loading NetCDF file directly ===")
    
    # Load NetCDF file directly
    filename = "ice_flow_results.nc"
    try:
        dataset = nc.Dataset(filename, 'r')
        print(f"Successfully opened {filename}")
    except Exception as e:
        print(f"Error opening file: {e}")
        return
    
    # Check for slope angle in miscellaneous
    slope_angle = None
    if 'miscellaneous' in dataset.groups:
        misc = dataset.groups['miscellaneous']
        if 'slope_angle' in misc.variables:
            slope_angle = misc.variables['slope_angle'][...]
            print(f"Found slope angle in model: {slope_angle} radians")
    
    # Try to get wavelength from config files
    try:
        # Try to import from various possible config modules
        wavelength = None
        for config_name in ['configp', 'config', 'configf8']:
            try:
                config_module = __import__(config_name)
                # Use brackets to access 'lambda' since it's a Python keyword
                wavelength = config_module.config.bedrock_params['lambda']
                print(f"Using wavelength from {config_name}: {wavelength} km")
                break
            except (ImportError, AttributeError):
                pass
        
        if wavelength is None:
            # Default wavelength if not found
            wavelength = 9.72
            print(f"Using default wavelength: {wavelength} km")
    except Exception as e:
        # Default wavelength if any error occurs
        wavelength = 9.72
        print(f"Using default wavelength: {wavelength} km (error: {e})")
    
    # Create output directory
    output_dir = "phase_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze phase relationship over time with consistent y-axis limits
    results = analyze_phase_over_time_from_netcdf(dataset, wavelength, output_dir)
    
    # Close the dataset
    dataset.close()
    
    # Write summary results to file
    summary_file = os.path.join(output_dir, "phase_analysis_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Phase Analysis Summary\n")
        f.write("=====================\n\n")
        f.write(f"Bedrock wavelength: {wavelength} km\n\n")
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
        f.write(f"Y-axis limits (filtered plots): {results['ylim_filtered']}\n")
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")
    print(f"Using consistent y-axis limits for all plots:")
    print(f"  Main plots: {results['ylim_main']}")
    print(f"  Filtered plots: {results['ylim_filtered']}")


if __name__ == '__main__':
    main()