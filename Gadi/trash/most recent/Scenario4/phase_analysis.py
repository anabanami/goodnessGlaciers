"""
Phase Analysis - Direct NetCDF Access
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
import re
import glob


def get_wavelength_from_profile(profile_id, profiles_dir="bedrock_profiles"):
    """Load the wavelength from the saved profile data"""
    try:
        filename = f"{profiles_dir}/bedrock_profile_{profile_id:03d}.npz"
        data = np.load(filename)
        wavelength = float(data['wavelength'])
        print(f"Loaded wavelength {wavelength} km from profile {profile_id}")
        return wavelength
    except (FileNotFoundError, KeyError):
        print(f"Could not load wavelength from profile {profile_id}, using default")
        return None


def extract_profile_from_filename(filename):
    """Extract profile number from the filename pattern like flowline9_profile_002.nc
    
    Args:
        filename: Path to the NetCDF file
    
    Returns:
        Profile number as an integer or None if not found
    """
    # Extract just the filename without the path
    base_filename = os.path.basename(filename)
    
    # Try to extract profile number using regex
    match = re.search(r'profile_(\d+)', base_filename)
    if match:
        return int(match.group(1))
    
    # Try other patterns
    match = re.search(r'_(\d{2,3})\.nc$', base_filename)
    if match:
        return int(match.group(1))
        
    # Default to None if no pattern found
    return None


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
    """Create a dedicated visualization of cross-correlation between bed and surface with improved resolution"""
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
    
    # Create a smoothed version of the correlation curve using interpolation
    from scipy import interpolate
    
    # Get visible portion of the correlation and lag distances
    visible_xcorr = xcorr[visible_lags]
    visible_lag_distances = lag_distances[visible_lags]
    
    # Create interpolation function (cubic spline for smoothness)
    f_interp = interpolate.interp1d(visible_lag_distances, visible_xcorr, kind='cubic', 
                                   bounds_error=False, fill_value=0)
    
    # Create more densely sampled lag distances for smoother curve
    smooth_lags = np.linspace(visible_lag_distances[0], visible_lag_distances[-1], 1000)
    
    # Generate smoothed correlation values
    smooth_xcorr = f_interp(smooth_lags)
    
    # Plot the smoothed correlation curve
    plt.plot(smooth_lags, smooth_xcorr, 'b-', linewidth=2, label='Cross-correlation')
    
    # Add vertical lines
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Zero lag')
    plt.axvline(x=lag_distance, color='purple', linestyle='--', linewidth=2,
               label=f'Maximum correlation: {lag_distance:.2f} km')
    
    theoretical_lag = (90/360) * wavelength  # Positive 90° lag in km
    plt.axvline(x=theoretical_lag, color='g', linestyle='--', linewidth=1.5,
               label=f'Theoretical (90°): {theoretical_lag:.2f} km')
    plt.axvline(x=-theoretical_lag, color='r', linestyle='--', linewidth=1.5,
               label=f'Opposite (-90°): {-theoretical_lag:.2f} km')
    
    plt.xlabel('Lag distance (km)')
    plt.ylabel('Cross-correlation')
    plt.title(f'Cross-correlation: bed and surface at t={time_val:.1f} years')
    
    # Position the legend in the upper right corner
    plt.legend(loc='upper right')
    
    plt.grid(True, linestyle=":", color='k', alpha=0.4)
    
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
    plt.grid(True, linestyle=":", color='k', alpha=0.4)
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
    plt.grid(True, linestyle=":", color='k', alpha=0.4)
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
    plt.grid(True, linestyle=":", color='k', alpha=0.4)
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


def load_surface_data_from_netcdf(dataset, time_step, x_base):
    """Load ONLY surface data for a specific time step and interpolate to match x_base"""
    print(f"Loading surface data for time step {time_step}")
    
    # Get coordinates from mesh group
    x = dataset.groups['mesh'].variables['x'][:]
    
    # Get surface vertices
    surf_indices = np.where(dataset.groups['mesh'].variables['vertexonsurface'][:] == 1)[0]
    x_surf = x[surf_indices]
    
    # Get surface elevation from TransientSolution
    transient = dataset.groups['results'].groups['TransientSolution']
    surface = transient.variables['Surface'][time_step, surf_indices]
    
    # Get time value
    time_val = float(transient.variables['time'][time_step])
    
    # Sort arrays by x coordinate
    surf_sort_idx = np.argsort(x_surf)
    x_surf = x_surf[surf_sort_idx]
    surface = surface[surf_sort_idx]
    
    # Interpolate surface onto the same x-coordinates as base
    surface_interp = np.interp(x_base, x_surf, surface)
    
    print(f"Interpolated surface data onto {len(x_base)} fixed bedrock points")
    
    # Diagnostic check to see how much interpolation affects the data
    if time_step == 0:  # Only do this for the first time step to avoid clutter
        plt.figure(figsize=(12, 6))
        plt.plot(x_surf, surface, 'r-', label='Original surface data')
        plt.plot(x_base, surface_interp, 'b-', label='Interpolated surface data')
        plt.xlabel('Distance (km)')
        plt.ylabel('Surface elevation (km)')
        plt.title(f'Effect of interpolation on surface data at t={time_val:.1f} years')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('surface_interpolation_check.png', dpi=300)
        plt.close()
        
        print("Saved diagnostic plot of interpolation effect to 'surface_interpolation_check.png'")
    
    return surface_interp, time_val

def analyze_phase_over_time_from_netcdf(dataset, wavelength, output_dir=None, profile_id=None):
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
        # Include profile ID in directory names if available
        profile_suffix = f"_profile{profile_id:03d}" if profile_id is not None else ""
        
        os.makedirs(output_dir, exist_ok=True)
        signals_dir = os.path.join(output_dir, f"signals{profile_suffix}")
        corr_dir = os.path.join(output_dir, f"correlations{profile_suffix}")
        phase_dir = os.path.join(output_dir, f"phase_relationship{profile_suffix}")
        os.makedirs(signals_dir, exist_ok=True)
        os.makedirs(corr_dir, exist_ok=True)
        os.makedirs(phase_dir, exist_ok=True)
    else:
        signals_dir = None
        corr_dir = None
        phase_dir = None
    
    # CRITICAL CHANGE: Load the bedrock profile just once at the beginning
    print("\n=== Loading bedrock profile once ===")
    # Get coordinates from mesh group
    x = dataset.groups['mesh'].variables['x'][:]
    
    # Get base vertices
    base_indices = np.where(dataset.groups['mesh'].variables['vertexonbase'][:] == 1)[0]
    x_base = x[base_indices]
    
    # Get initial bedrock elevation from time step 0
    base = dataset.groups['results'].groups['TransientSolution'].variables['Base'][0, base_indices]
    
    # Sort arrays by x coordinate
    base_sort_idx = np.argsort(x_base)
    x_base = x_base[base_sort_idx]
    base = base[base_sort_idx]
    
    # Remove edges to avoid boundary effects
    buffer = 2.0  # km from each edge
    valid_idx = (x_base >= min(x_base) + buffer) & (x_base <= max(x_base) - buffer)
    if sum(valid_idx) > 10:  # Ensure we have enough points left
        print(f"Removing edge points within {buffer} km of boundaries")
        x_base = x_base[valid_idx]
        base = base[valid_idx]
    
    # Calculate dx (spacing between points)
    dx = np.mean(np.diff(x_base))
    
    # Apply bandpass filter to bedrock once
    base_filtered = bandpass_filter(x_base, dx, base, wavelength)
    
    print(f"Loaded bedrock profile with {len(x_base)} points, dx={dx:.4f} km")
    
    # First pass: determine global min/max values for surface only
    print("\n=== First pass: determining global y-axis limits ===")
    all_surface_vals = []
    all_surface_filtered_vals = []
    
    for step in range(num_steps):
        print(f"Analyzing limits for time step {step+1}/{num_steps}")
        
        # Load ONLY the surface data for this time step
        surface, time_val = load_surface_data_from_netcdf(dataset, step, x_base)
        
        # Apply bandpass filter to surface
        surface_filtered = bandpass_filter(x_base, dx, surface, wavelength)
        
        # Store values for limit calculation (after mean removal)
        all_surface_vals.extend((surface - np.mean(surface)).tolist())
        all_surface_filtered_vals.extend((surface_filtered - np.mean(surface_filtered)).tolist())
    
    # Include bedrock in the y-limits calculation
    all_base_vals = (base - np.mean(base)).tolist()
    all_base_filtered_vals = (base_filtered - np.mean(base_filtered)).tolist()
    
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
        
        # Load ONLY the surface data for this time step
        surface, time_val = load_surface_data_from_netcdf(dataset, step, x_base)
        
        # Apply bandpass filter to surface
        surface_filtered = bandpass_filter(x_base, dx, surface, wavelength)
        
        # Plot signals with consistent y-limits
        if signals_dir:
            plot_signals(x_base, base, surface, base_filtered, surface_filtered, 
                       time_val, signals_dir, ylim_main, ylim_filtered)
        
        # Generate enhanced phase relationship visualization
        if phase_dir:
            visualise_phase_relationship(x_base, base_filtered, surface_filtered, 
                                        wavelength, time_val, phase_dir)
        
        # Create cross-correlation visualization
        if corr_dir:
            visualise_cross_correlation(x_base, base_filtered, surface_filtered,
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

    # Add profile ID to title if available
    profile_info = f" (Profile {profile_id})" if profile_id is not None else ""
    plt.xlabel('Time (years)')
    plt.ylabel('Phase shift (degrees)')
    plt.title(f'Evolution of phase shift between bed and surface over time{profile_info}')
    plt.grid(True, linestyle=":", color='k', alpha=0.4)
    plt.legend(loc='best')
    
    # Plot lag distance evolution
    plt.subplot(2, 1, 2)
    plt.plot(time_values, lag_distances, 'ro-', label='Lag distance', linewidth=2)
    plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    theoretical_lag = wavelength/4
    # plt.axhline(y=theoretical_lag, color='g', linestyle='--', 
    #             label=f'Theoretical (λ/4 = {theoretical_lag:.2f} km)', linewidth=2)
    # plt.axhline(y=-theoretical_lag, color='r', linestyle='--', 
    #             label=f'Opposite (–λ/4 = {-theoretical_lag:.2f} km)', linewidth=1.5)

    plt.xlabel('Time (years)')
    plt.ylabel('Lag distance (km)')
    plt.title(f'Evolution of spatial lag between bed and surface{profile_info}')
    plt.grid(True, linestyle=":", color='k', alpha=0.4)
    plt.legend(loc='best')
    
    plt.tight_layout()
    
    if output_dir:
        # Include profile ID in filename if available
        profile_suffix = f"_profile{profile_id:03d}" if profile_id is not None else ""
        filename = os.path.join(output_dir, f"phase_evolution{profile_suffix}.png")
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
        'ylim_filtered': ylim_filtered,
        'profile_id': profile_id
    }
    
    return results


def process_multiple_files(file_pattern, output_dir="phase_analysis_results", default_wavelength=9.72):
    """Process multiple NetCDF files matching a pattern
    
    Args:
        file_pattern: Glob pattern for files to process
        output_dir: Base directory to save output
        default_wavelength: Default wavelength if not found in files
    """
    # Find all files matching the pattern
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return
    
    print(f"Found {len(files)} files to process")
    
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file
    results = {}
    for file in files:
        profile_id = extract_profile_from_filename(file)
        result = process_single_file(file, output_dir, default_wavelength)
        if result is not None and profile_id is not None:
            results[profile_id] = result
    
    # Create a comparative plot if we have multiple profiles
    if len(results) > 1:
        create_comparative_plot(results, output_dir)
        
    return results

def compare_fixed_vs_evolving_bed(dataset, wavelength, output_dir=None, profile_id=None):
    """Compare phase analysis using fixed bedrock vs. evolving bedrock"""
    # Get number of time steps
    transient = dataset.groups['results'].groups['TransientSolution']
    num_steps = transient.variables['time'].shape[0]
    
    # Create arrays for both methods
    time_values = []
    phase_shifts_fixed = []
    phase_shifts_evolving = []
    
    # Load initial bedrock once
    x, base_initial, _, _ = load_time_dependent_data_from_netcdf(dataset, 0)
    dx = np.mean(np.diff(x))
    base_initial_filtered = bandpass_filter(x, dx, base_initial, wavelength)
    
    # Process each time step
    for step in range(num_steps):
        # Load time-specific data
        x, base_evolving, surface, time_val = load_time_dependent_data_from_netcdf(dataset, step)
        
        # Apply filters
        base_evolving_filtered = bandpass_filter(x, dx, base_evolving, wavelength)
        surface_filtered = bandpass_filter(x, dx, surface, wavelength)
        
        # Calculate phase shifts with both methods
        phase_fixed, _, _ = phase_shift_analysis(
            dx, base_initial_filtered, surface_filtered, wavelength, time_val)
        phase_evolving, _, _ = phase_shift_analysis(
            dx, base_evolving_filtered, surface_filtered, wavelength, time_val)
        
        # Store results
        time_values.append(time_val)
        phase_shifts_fixed.append(phase_fixed * 180 / np.pi)  # Convert to degrees
        phase_shifts_evolving.append(phase_evolving * 180 / np.pi)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    plt.plot(time_values, phase_shifts_fixed, 'b-', label='Fixed Initial Bed', linewidth=2)
    plt.plot(time_values, phase_shifts_evolving, 'r--', label='Evolving Bed', linewidth=2)
    plt.axhline(y=90, color='g', linestyle=':', label='Theoretical (90°)')
    plt.xlabel('Time (years)')
    plt.ylabel('Phase Shift (degrees)')
    plt.title(f'Comparison of Phase Analysis Methods - Profile {profile_id}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'method_comparison_profile_{profile_id}.png'), dpi=300)
        
    return {
        'time': time_values,
        'phase_fixed': phase_shifts_fixed,
        'phase_evolving': phase_shifts_evolving
    }


def analyze_method_differences(results_comparison, output_dir=None, profile_id=None):
    """Analyze differences between fixed and evolving bed analysis"""
    time_values = results_comparison['time']
    phase_fixed = np.array(results_comparison['phase_fixed'])
    phase_evolving = np.array(results_comparison['phase_evolving'])
    
    # Calculate difference
    phase_difference = phase_fixed - phase_evolving
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_values, phase_difference, 'k-', linewidth=2)
    plt.axhline(y=0, color='r', linestyle=':')
    plt.xlabel('Time (years)')
    plt.ylabel('Phase Difference (degrees)')
    plt.title(f'Difference in Phase Calculation Methods - Profile {profile_id}')
    plt.grid(True, alpha=0.3)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'method_difference_profile_{profile_id}.png'), dpi=300)
    
    return {
        'time': time_values,
        'phase_difference': phase_difference,
        'mean_difference': np.mean(phase_difference),
        'max_difference': np.max(np.abs(phase_difference))
    }


def create_comparative_plot(results_dict, output_dir):
    """Create a comparative plot of phase shifts across multiple profiles
    
    Args:
        results_dict: Dictionary of results keyed by profile_id
        output_dir: Directory to save output
    """
    if not results_dict:
        return
        
    plt.figure(figsize=(14, 10))
    
    # Plot phase shift evolution for each profile
    plt.subplot(2, 1, 1)
    for profile_id, results in sorted(results_dict.items()):
        plt.plot(results['time'], results['phase_shift_deg'], 'o-', 
                label=f'Profile {profile_id}', linewidth=2)
    
    plt.axhline(y=90, color='g', linestyle='--', label='Theoretical (90°)', linewidth=2)
    plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    plt.axhline(y=-90, color='r', linestyle='--', label='Opposite phase (-90°)', linewidth=1.5)
    
    plt.xlabel('Time (years)')
    plt.ylabel('Phase shift (degrees)')
    plt.title('Comparison of phase shift evolution across profiles')
    plt.grid(True, linestyle=":", color='k', alpha=0.4)
    plt.legend(loc='best')
    
    # Plot lag distance evolution for each profile
    plt.subplot(2, 1, 2)
    for profile_id, results in sorted(results_dict.items()):
        plt.plot(results['time'], results['lag_distance'], 'o-', 
                label=f'Profile {profile_id}', linewidth=2)
    
    plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    
    # Use wavelength from first profile for theoretical lines
    first_profile = next(iter(results_dict.values()))
    wavelength = first_profile['lag_distance'][0] * 4  # Estimate wavelength from lag distance
    theoretical_lag = wavelength/4
    
    plt.axhline(y=theoretical_lag, color='g', linestyle='--', 
                label=f'Theoretical (λ/4 ≈ {theoretical_lag:.2f} km)', linewidth=2)
    plt.axhline(y=-theoretical_lag, color='r', linestyle='--', 
                label=f'Opposite (–λ/4 ≈ {-theoretical_lag:.2f} km)', linewidth=1.5)
    
    plt.xlabel('Time (years)')
    plt.ylabel('Lag distance (km)')
    plt.title('Comparison of spatial lag evolution across profiles')
    plt.grid(True, linestyle=":", color='k', alpha=0.4)
    plt.legend(loc='best')
    
    plt.tight_layout()
    
    # Save figure
    filename = os.path.join(output_dir, "comparative_phase_evolution.png")
    plt.savefig(filename, dpi=300)
    print(f"Saved comparative phase evolution plot to {filename}")


def visualize_bedrock_evolution(dataset, output_dir=None, profile_id=None):
    """Visualize how the bedrock evolves over time"""
    # Get number of time steps
    transient = dataset.groups['results'].groups['TransientSolution']
    num_steps = transient.variables['time'].shape[0]
    
    # Get initial bedrock
    x, base_initial, _, time_initial = load_time_dependent_data_from_netcdf(dataset, 0)
    
    # Sample time steps to show evolution (e.g., beginning, middle, end)
    steps_to_show = [0, num_steps//4, num_steps//2, 3*num_steps//4, num_steps-1]
    
    plt.figure(figsize=(12, 8))
    
    # Plot each selected time step
    for step in steps_to_show:
        x, base, surface, time_val = load_time_dependent_data_from_netcdf(dataset, step)
        plt.plot(x, base, label=f't = {time_val:.1f} years')
    
    plt.xlabel('Distance (km)')
    plt.ylabel('Elevation (km)')
    plt.title(f'Bedrock Evolution Over Time - Profile {profile_id}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'bedrock_evolution_profile_{profile_id}.png'), dpi=300)


def get_wavelength_from_profile(profile_id, profiles_dir="bedrock_profiles"):
    """Load the wavelength from the saved profile data"""
    try:
        filename = f"{profiles_dir}/bedrock_profile_{profile_id:03d}.npz"
        data = np.load(filename)
        wavelength = float(data['wavelength'])
        print(f"Loaded wavelength {wavelength} km from profile {profile_id}")
        return wavelength
    except (FileNotFoundError, KeyError):
        print(f"Could not load wavelength from profile {profile_id}, using default")
        return None

def process_single_file(filename, output_dir="phase_analysis_results", default_wavelength=9.72):
    """Process a single NetCDF file for phase analysis
    
    Args:
        filename: Path to the NetCDF file
        output_dir: Directory to save output
        default_wavelength: Default wavelength if not found in the file
    """
    print(f"\n=== Loading NetCDF file directly: {filename} ===")
    
    # Extract profile number from filename - this needs to come FIRST
    profile_id = extract_profile_from_filename(filename)
    if profile_id is not None:
        print(f"Detected profile ID: {profile_id}")
        # Create profile-specific output directory if profile_id is found
        profile_dir = os.path.join(output_dir, f"profile_{profile_id:03d}")
        os.makedirs(profile_dir, exist_ok=True)
        
        # Try to get wavelength from profile database AFTER getting profile_id
        profile_wavelength = get_wavelength_from_profile(profile_id)
        if profile_wavelength is not None:
            wavelength = profile_wavelength
            print(f"Using wavelength from profile database: {wavelength} km")
    else:
        print("No profile ID detected in filename")
        profile_dir = output_dir
    
    # Load NetCDF file
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
    
    # Only try to get wavelength from config or NetCDF if we didn't already get it from profile
    if 'wavelength' not in locals():
        try:
            # Try to import the config module
            try:
                from configf9_synthetic import config
                wavelength = config.bedrock_params['lambda']
                print(f"Using wavelength from configf9: {wavelength} km")
            except (ImportError, AttributeError, KeyError):
                # If config import fails, try to get wavelength from dataset
                if 'miscellaneous' in dataset.groups:
                    misc = dataset.groups['miscellaneous']
                    if 'bedrock_wavelength' in misc.variables:
                        wavelength = misc.variables['bedrock_wavelength'][...]
                        print(f"Using wavelength from netCDF file: {wavelength} km")
                    else:
                        wavelength = default_wavelength
                        print(f"Using default wavelength: {wavelength} km")
                else:
                    wavelength = default_wavelength
                    print(f"Using default wavelength: {wavelength} km")
        except Exception as e:
            wavelength = default_wavelength
            print(f"Error getting wavelength, using default: {wavelength} km")
            print(f"Error was: {e}")
    
    # Create output directory
    os.makedirs(profile_dir, exist_ok=True)
    
    # First do the comparison analyses that need to be done before closing the dataset
    print("Comparing fixed vs. evolving bed approaches...")
    comparison_results = compare_fixed_vs_evolving_bed(dataset, wavelength, profile_dir, profile_id)
    print("Visualizing bedrock evolution...")
    visualize_bedrock_evolution(dataset, profile_dir, profile_id)
    
    # Analyze phase relationship over time with consistent y-axis limits
    results = analyze_phase_over_time_from_netcdf(dataset, wavelength, profile_dir, profile_id)
    
    # Close the dataset
    dataset.close()
    
    # Now do analyses that don't need the open dataset
    print("Analyzing method differences...")
    difference_analysis = analyze_method_differences(comparison_results, profile_dir, profile_id)
    
    # Write summary results to file
    profile_suffix = f"_profile{profile_id:03d}" if profile_id is not None else ""
    summary_file = os.path.join(profile_dir, f"phase_analysis_summary{profile_suffix}.txt")
    with open(summary_file, 'w') as f:
        f.write("Phase Analysis Summary\n")
        f.write("=====================\n\n")
        f.write(f"Bedrock wavelength: {wavelength} km\n\n")
        f.write(f"Coordinates: Slope-parallel\n\n")
        
        # Add profile information if available
        if profile_id is not None:
            f.write(f"Profile ID: {profile_id}\n\n")
            
        f.write("Time-dependent phase shift:\n")
        f.write("-------------------------\n")
        for i in range(len(results['time'])):
            f.write(f"Time: {results['time'][i]:.1f} years\n")
            f.write(f"  Phase shift: {results['phase_shift'][i]/np.pi:.2f}π radians ({results['phase_shift_deg'][i]:.1f} degrees)\n")
            f.write(f"  Lag distance: {results['lag_distance'][i]:.3f} km\n")
            f.write(f"  Correlation: {results['correlation'][i]:.3f}\n")
            f.write(f"  Difference from π/2: {abs(results['phase_shift'][i] - np.pi/2)/np.pi:.2f}π radians\n\n")
        
        # Add method comparison information
        f.write("\nMethod Comparison:\n")
        f.write("----------------\n")
        f.write(f"Mean difference between fixed and evolving bed: {difference_analysis['mean_difference']:.2f} degrees\n")
        f.write(f"Maximum absolute difference: {difference_analysis['max_difference']:.2f} degrees\n\n")
        
        f.write("\nPlotting Information:\n")
        f.write("--------------------\n")
        f.write(f"Y-axis limits (main plots): {results['ylim_main']}\n")
        f.write(f"Y-axis limits (filtered plots): {results['ylim_filtered']}\n")
    
    print(f"\nAnalysis complete. Results saved to {profile_dir}")
    print(f"Phase analysis summary written to {summary_file}")
    
    return results

def main():
    """Main function with command-line argument handling"""
    import argparse
    
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Phase analysis for ISSM NetCDF files')
    parser.add_argument('file', nargs='?', default='ice_flow_results.nc',
                        help='NetCDF file to analyze or file pattern with wildcard')
    parser.add_argument('--pattern', help='File pattern for batch processing (e.g., "flowline*.nc")')
    parser.add_argument('--output-dir', default='phase_analysis_results',
                        help='Directory to save output files')
    parser.add_argument('--wavelength', type=float, default=9.72,
                        help='Default bedrock wavelength in km if not found in file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process files based on arguments
    if args.pattern:
        # Process multiple files matching a pattern
        print(f"\n=== Processing multiple files matching pattern: {args.pattern} ===")
        process_multiple_files(args.pattern, args.output_dir, args.wavelength)
    else:
        # Process a single file
        print(f"\n=== Processing single file: {args.file} ===")
        process_single_file(args.file, args.output_dir, args.wavelength)


if __name__ == "__main__":
    main()