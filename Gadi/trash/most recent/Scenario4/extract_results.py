#!/usr/bin/env python
"""
ISSM Flow Visualization
This script properly visualizes ISSM flow results using triangular mesh plotting.
"""

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os
import sys
import re


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


def create_mask(triang):
    """Create a mask for triangulation to remove flat triangles
    
    Args:
        triang: Matplotlib triangulation object
    
    Returns:
        Mask array for the triangulation
    """
    analyzer = tri.TriAnalyzer(triang)
    return analyzer.get_flat_tri_mask(0.001)


def extract_visualize_issm_results(filename, output_dir="."):
    """
    Extract and visualize ISSM flow results.
    
    Args:
        filename: Path to the NetCDF file
        output_dir: Directory to save output
    """
    print(f"\n{'='*80}")
    print(f"EXTRACTING AND VISUALIZING ISSM RESULTS FROM: {filename}")
    print(f"{'='*80}\n")
    
    # Extract profile number from filename
    profile_id = extract_profile_from_filename(filename)
    if profile_id is not None:
        print(f"Detected profile ID: {profile_id}")
        # Create profile-specific output directory if profile_id is found
        profile_dir = os.path.join(output_dir, f"profile_{profile_id:03d}")
        os.makedirs(profile_dir, exist_ok=True)
    else:
        print("No profile ID detected in filename")
        profile_dir = output_dir
    
    os.makedirs(profile_dir, exist_ok=True)
    
    # Open the NetCDF file
    dataset = nc.Dataset(filename, 'r')
    
    # Dictionary to store extracted data
    data = {
        'mesh': {},
        'geometry': {},
        'friction': {},
        'time': {},
        'transient': {}
    }
    
    # Store profile information
    data['profile_id'] = profile_id
    
    # Get time information
    if 'timestepping' in dataset.groups:
        time_group = dataset.groups['timestepping']
        data['time']['time_step'] = time_group.variables['time_step'][...]
        data['time']['start_time'] = time_group.variables['start_time'][...]
        data['time']['final_time'] = time_group.variables['final_time'][...]
        
        # Calculate time steps
        num_steps = int((data['time']['final_time'] - data['time']['start_time']) / 
                      data['time']['time_step']) + 1
        data['time']['num_steps'] = num_steps
        data['time']['times'] = np.linspace(data['time']['start_time'], 
                                         data['time']['final_time'], 
                                         num_steps)
    else:
        print("No time information found")
    
    # Extract mesh data
    if 'mesh' in dataset.groups:
        mesh_group = dataset.groups['mesh']
        data['mesh']['x'] = mesh_group.variables['x'][...]
        data['mesh']['y'] = mesh_group.variables['y'][...]
        
        # Elements are typically 1-indexed in ISSM
        elements = mesh_group.variables['elements'][...]
        if np.min(elements) == 1:  # Check if 1-indexed
            elements = elements - 1  # Convert to 0-indexed for matplotlib
        
        data['mesh']['elements'] = elements
        
        if 'vertexonbase' in mesh_group.variables:
            data['mesh']['vertexonbase'] = mesh_group.variables['vertexonbase'][...]
        
        if 'vertexonsurface' in mesh_group.variables:
            data['mesh']['vertexonsurface'] = mesh_group.variables['vertexonsurface'][...]
        
        print(f"Mesh: {len(data['mesh']['x'])} vertices, {len(data['mesh']['elements'])} elements")
    else:
        print("No mesh data found")
    
    # Extract geometry data
    if 'geometry' in dataset.groups:
        geom_group = dataset.groups['geometry']
        for var in ['surface', 'base', 'thickness', 'bed']:
            if var in geom_group.variables:
                data['geometry'][var] = geom_group.variables[var][...]
        
        print("Geometry data extracted")
    else:
        print("No geometry data found")
    
    # Extract friction data
    if 'friction' in dataset.groups:
        friction_group = dataset.groups['friction']
        if 'coefficient' in friction_group.variables:
            data['friction']['coefficient'] = friction_group.variables['coefficient'][...]
            print("Friction data extracted")
    else:
        print("No friction data found")
    
    # Extract slope angle if available
    if 'miscellaneous' in dataset.groups:
        misc_group = dataset.groups['miscellaneous']
        if 'slope_angle' in misc_group.variables:
            data['slope_angle'] = misc_group.variables['slope_angle'][...]
            print(f"Slope angle: {data['slope_angle']} radians")
    
    # Extract TransientSolution data
    if 'results' in dataset.groups:
        results_group = dataset.groups['results']
        
        # Check for TransientSolution
        if hasattr(results_group, 'groups') and 'TransientSolution' in results_group.groups:
            transient = results_group.groups['TransientSolution']
            print("Found TransientSolution group")
            
            # Extract time information from TransientSolution
            if 'time' in transient.variables:
                data['transient']['time'] = transient.variables['time'][...]
                print(f"  Extracted time with shape {data['transient']['time'].shape}")
                # Update num_steps based on actual time data
                data['time']['actual_steps'] = len(data['transient']['time'])
            elif 'step' in transient.variables:
                data['transient']['step'] = transient.variables['step'][...]
                print(f"  Extracted step with shape {data['transient']['step'].shape}")
                # Update num_steps based on actual step data
                data['time']['actual_steps'] = len(data['transient']['step'])
            
            # Extract key variables
            key_vars = ['Vx', 'Vy', 'Vel', 'Pressure', 'Surface', 'Base', 'Thickness']
            for var_name in transient.variables:
                if var_name in key_vars:
                    data['transient'][var_name] = transient.variables[var_name][...]
                    print(f"  Extracted {var_name} with shape {data['transient'][var_name].shape}")
    
    # Close the dataset
    dataset.close()
    
    if 'geometry' in data and 'base' in data['geometry']:
        print(f"Using static bedrock topography with min={np.min(data['geometry']['base']):.3f}, "
              f"max={np.max(data['geometry']['base']):.3f} km")
    
    # Visualize the data
    if 'mesh' in data and 'transient' in data:
        # Determine the number of time steps
        num_steps = data['time']['actual_steps'] if 'actual_steps' in data['time'] else 5
        
        print(f"Creating visualizations for {num_steps} time steps...")
        
        # Create folders for different plot types
        field_dirs = {}
        for field in ['Pressure', 'Vx', 'Vel', 'Thickness']:
            if field in data['transient']:
                field_dir = os.path.join(profile_dir, field)
                os.makedirs(field_dir, exist_ok=True)
                field_dirs[field] = field_dir
        
        # Create visualizations for each time step
        for step in range(num_steps):
            time_val = data['transient']['time'][step] if 'time' in data['transient'] else step
            
            for field, field_dir in field_dirs.items():
                if field in data['transient'] and step < data['transient'][field].shape[0]:
                    # Include profile ID in filename if available
                    profile_suffix = f"_profile{profile_id:03d}" if profile_id is not None else ""
                    file_path = os.path.join(field_dir, f"{field}_step{step:02d}{profile_suffix}.png")
                    
                    # Get field data for this time step
                    field_data = data['transient'][field][step]
                    
                    # Default vmin/vmax values
                    vmin, vmax = None, None
                    
                    # Special handling for velocity
                    if field in ['Vx', 'Vel']:
                        vmax = max(abs(field_data.min()), abs(field_data.max()))
                        vmin = -vmax if field == 'Vx' else 0
                    
                    # Create and save visualization
                    create_field_plot(data, field, field_data, step, time_val, file_path, vmin, vmax)
        
        # Create a combined visualization showing geometry and velocity
        if 'Vx' in data['transient'] and 'Surface' in data['transient'] and 'Base' in data['transient']:
            create_combined_plots(data, profile_dir)
    
        # Add pressure gradient analysis
        if 'Pressure' in data['transient']:
            print("\nPerforming pressure gradient analysis...")
            # Analyze pressure gradient evolution
            analyze_pressure_gradient_evolution(data, profile_dir)
    
    return data
    

def transform_coordinates(x, z, alpha, inverse=False):
    """Transform coordinates between original and slope-parallel systems
    
    Args:
        x, z: Coordinates to transform
        alpha: Slope angle in radians
        inverse: If True, transform from (x,z) to (x',z'). If False, (x',z') to (x,z)
    
    Returns:
        tuple: (transformed_x, transformed_z)
    """
    if inverse:
        # From global to slope-parallel (x,z) -> (x',z')
        return (x * np.cos(alpha) + z * np.sin(alpha),
               -x * np.sin(alpha) + z * np.cos(alpha))
    else:
        # From slope-parallel to global (x',z') -> (x,z)
        return (x * np.cos(alpha) - z * np.sin(alpha),
               x * np.sin(alpha) + z * np.cos(alpha))

def get_transformed_coordinates(data, step):
    """Get coordinates transformed by ice evolution
    
    Args:
        data: Dictionary containing mesh and transient data
        step: Time step
    
    Returns:
        tuple: (x, y) coordinates
    """
    # Get base mesh coordinates
    x = data['mesh']['x']
    y = data['mesh']['y']
    
    # Use ONLY the initial geometry for the base, while allowing surface to evolve
    if 'transient' in data and 'Surface' in data['transient'] and 'geometry' in data and 'base' in data['geometry']:
        try:
            # Always use initial bedrock geometry
            initial_base = data['geometry']['base']
            
            # Get initial thickness
            if 'thickness' in data['geometry']:
                initial_thickness = data['geometry']['thickness']
            else:
                # Calculate from initial surface and base if available
                if 'surface' in data['geometry']:
                    initial_thickness = data['geometry']['surface'] - initial_base
                else:
                    # Use evolved thickness as fallback
                    initial_thickness = data['transient']['Thickness'][step]
            
            # Get evolved surface if available, otherwise calculate from evolved thickness
            if 'Surface' in data['transient']:
                evolved_surface = data['transient']['Surface'][step]
                # Calculate evolved thickness from evolved surface and INITIAL base
                evolved_thickness = evolved_surface - initial_base
            elif 'Thickness' in data['transient']:
                # Use evolved thickness to calculate evolved surface
                evolved_thickness = data['transient']['Thickness'][step]
                evolved_surface = initial_base + evolved_thickness
            
            # Calculate relative height in the initial geometry
            rel_height = (y - initial_base) / initial_thickness
            rel_height = np.clip(rel_height, 0, 1)  # Ensure values are between 0 and 1
            
            # Transform to new geometry while preserving initial base
            y = initial_base + rel_height * evolved_thickness
            
        except (KeyError, IndexError) as e:
            print(f"Warning: Could not transform coordinates: {e}")
    
    # Apply slope-parallel transformation if needed
    if 'slope_angle' in data and data['slope_angle'] != 0:
        alpha = data['slope_angle']
        x, y = transform_coordinates(x, y, alpha, inverse=False)
    
    return x, y


def create_field_plot(data, field_name, field_data, step, time_val, file_path, vmin=None, vmax=None):
    """Create a visualization of a field on the mesh
    
    Args:
        data: Dictionary containing mesh and field data
        field_name: Name of the field being plotted
        field_data: Array of field values
        step: Time step
        time_val: Time value
        file_path: Path to save the plot
        vmin, vmax: Optional min/max values for color scale
    """
    # Get mesh coordinates
    x, y = get_transformed_coordinates(data, step)
    
    # Create triangulation
    triang = tri.Triangulation(x, y, triangles=data['mesh']['elements'])
    
    # Create mask for poor quality triangles
    mask = create_mask(triang)
    triang.set_mask(mask)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Choose appropriate colormap
    cmap = 'viridis'
    if field_name == 'Vx':
        cmap = 'coolwarm'
    elif field_name == 'Pressure':
        cmap = 'plasma'
    
    # Create pseudocolor plot
    tc = ax.tripcolor(triang, field_data, cmap=cmap, shading='gouraud', 
                     vmin=vmin, vmax=vmax)
    
    # Add a light mesh outline
    ax.triplot(triang, 'k-', lw=0.1, alpha=0.3)
    
    # Include profile number in title if available
    profile_info = f" (Profile {data['profile_id']})" if 'profile_id' in data and data['profile_id'] is not None else ""
    
    # Add title and labels
    ax.set_title(f'{field_name} at t={time_val:.1f} years{profile_info} (Static Bedrock)')
    ax.set_xlabel('Distance along flowline (km)')
    ax.set_ylabel('Elevation (km)')
    ax.grid(True, linestyle=":", color='k', alpha=0.4)
    
    # Add colorbar
    cbar = plt.colorbar(tc, ax=ax)
    if field_name == 'Vx' or field_name == 'Vel':
        cbar.set_label('Velocity (km/yr)')
    elif field_name == 'Pressure':
        cbar.set_label('Pressure')
    elif field_name == 'Thickness':
        cbar.set_label('Thickness (km)')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close(fig)
    
    print(f"Created {field_name} plot for time step {step}: {file_path}")

def create_combined_plots(data, output_dir):
    """Create combined plots showing geometry and velocity together
    
    Args:
        data: Dictionary containing mesh and field data
        output_dir: Directory to save output
    """
    # Create output directory
    combined_dir = os.path.join(output_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)
    
    # Determine number of time steps
    num_steps = data['time']['actual_steps'] if 'actual_steps' in data['time'] else 5
    
    # Get profile ID for filenames
    profile_id = data.get('profile_id')
    profile_suffix = f"_profile{profile_id:03d}" if profile_id is not None else ""
    
    # Create plots for first, middle, and last time steps
    step_indices = [0, num_steps//2, num_steps-1]
    
    for step in step_indices:
        if step >= num_steps:
            continue
            
        # Get time value
        time_val = data['transient']['time'][step] if 'time' in data['transient'] else step
        
        # Get mesh coordinates
        x, y = get_transformed_coordinates(data, step)
        
        # Create triangulation
        triang = tri.Triangulation(x, y, triangles=data['mesh']['elements'])
        
        # Create mask for poor quality triangles
        mask = create_mask(triang)
        triang.set_mask(mask)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
        
        # Include profile number in title if available
        profile_info = f" (Profile {profile_id})" if profile_id is not None else ""
        
        # Plot 1: Velocity field
        if 'Vx' in data['transient']:
            velocity = data['transient']['Vx'][step]
            vmax = max(abs(velocity.min()), abs(velocity.max()))
            tc1 = ax1.tripcolor(triang, velocity, cmap='coolwarm', shading='gouraud', 
                               vmin=-vmax, vmax=vmax)
            ax1.triplot(triang, 'k-', lw=0.1, alpha=0.2)
            ax1.set_title(f'Ice Velocity (Vx) at t={time_val:.1f} years{profile_info}')
            ax1.set_ylabel('Elevation (km)')
            cbar1 = fig.colorbar(tc1, ax=ax1)
            cbar1.set_label('Velocity (km/yr)')
            ax1.grid(True, linestyle=":", color='k', alpha=0.4)

        
        # Plot 2: Pressure field
        if 'Pressure' in data['transient']:
            pressure = data['transient']['Pressure'][step]
            tc2 = ax2.tripcolor(triang, pressure, cmap='plasma', shading='gouraud')
            ax2.triplot(triang, 'k-', lw=0.1, alpha=0.2)
            ax2.set_title(f'Ice Pressure at t={time_val:.1f} years{profile_info}')
            ax2.set_xlabel('Distance along flowline (km)')
            ax2.set_ylabel('Elevation (km)')
            cbar2 = fig.colorbar(tc2, ax=ax2)
            cbar2.set_label('Pressure')
            ax2.grid(True, linestyle=":", color='k', alpha=0.4)
        
        # Save figure with profile ID in filename
        plt.tight_layout()
        plt.savefig(os.path.join(combined_dir, f'combined_step{step:02d}{profile_suffix}.png'), dpi=300)
        plt.close(fig)
        
        print(f"Created combined plot for time step {step}")
    
    # Create summary plot showing velocity evolution
    if 'Vx' in data['transient']:
        create_velocity_evolution_plot(data, combined_dir)

def create_velocity_evolution_plot(data, output_dir):
    """Create a plot showing the evolution of maximum velocity over time
    
    Args:
        data: Dictionary containing mesh and field data
        output_dir: Directory to save output
    """
    # Determine number of time steps
    num_steps = data['time']['actual_steps'] if 'actual_steps' in data['time'] else 5
    times = data['transient']['time'] if 'time' in data['transient'] else np.arange(num_steps)
    
    # Get profile ID for filenames
    profile_id = data.get('profile_id')
    profile_suffix = f"_profile{profile_id:03d}" if profile_id is not None else ""
    profile_info = f" (Profile {profile_id})" if profile_id is not None else ""
    
    # Calculate maximum velocity at each time step
    max_velocity = []
    min_velocity = []
    avg_velocity = []
    
    for step in range(num_steps):
        if 'Vx' in data['transient'] and step < data['transient']['Vx'].shape[0]:
            velocity = data['transient']['Vx'][step]
            max_velocity.append(velocity.max())
            min_velocity.append(velocity.min())
            avg_velocity.append(np.mean(velocity))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot velocity evolution
    ax.plot(times[:len(max_velocity)], max_velocity, 'r-o', label='Maximum Vx')
    ax.plot(times[:len(min_velocity)], min_velocity, 'b-o', label='Minimum Vx')
    ax.plot(times[:len(avg_velocity)], avg_velocity, 'g-o', label='Average Vx')
    
    # Add title and labels
    ax.set_title(f'Velocity Evolution Over Time{profile_info}')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Velocity (km/yr)')
    ax.grid(True, linestyle=":", color='k', alpha=0.4)
    ax.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'velocity_evolution{profile_suffix}.png'), dpi=300)
    plt.close(fig)
    
    print(f"Created velocity evolution plot")

def process_multiple_files(file_pattern, output_dir="."):
    """Process multiple files matching a pattern
    
    Args:
        file_pattern: Glob pattern for files to process
        output_dir: Base directory to save output
    """
    import glob
    
    # Find all files matching the pattern
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return
    
    print(f"Found {len(files)} files to process")
    
    # Process each file
    for file in files:
        extract_visualize_issm_results(file, output_dir)


def analyze_terminus_pressure_gradient(data, output_dir, step_index=None):
    """Analyze pressure gradients at the terminus boundary
    
    Args:
        data: Dictionary containing mesh and field data
        output_dir: Directory to save output
        step_index: Time step index to analyze (default: last available step)
    
    Returns:
        Dictionary with pressure gradient analysis results
    """
    if 'transient' not in data or 'Pressure' not in data['transient']:
        print("No pressure data available for analysis")
        return None
    
    # Select appropriate time step
    num_steps = data['transient']['Pressure'].shape[0]
    if step_index is None or step_index >= num_steps:
        step_index = num_steps - 1
    
    # Get time value
    time_val = data['transient']['time'][step_index] if 'time' in data['transient'] else step_index
    
    # Get coordinates and mesh information
    x, y = get_transformed_coordinates(data, step_index)
    
    # Define terminus region (rightmost 5% of domain)
    terminus_x = np.max(x)
    domain_width = np.max(x) - np.min(x)
    near_terminus = x > (terminus_x - 0.05 * domain_width)
    
    # Get pressure field
    pressure = data['transient']['Pressure'][step_index]
    
    # Sort by x-coordinate for gradient calculation
    sorted_indices = np.argsort(x[near_terminus])
    sorted_x = x[near_terminus][sorted_indices]
    sorted_p = pressure[near_terminus][sorted_indices]
    
    # Calculate pressure gradient using finite differences
    dx = np.diff(sorted_x)
    dp = np.diff(sorted_p)
    pressure_gradient = dp / dx  # Pa/km
    gradient_positions = (sorted_x[:-1] + sorted_x[1:]) / 2  # Midpoints
    
    # Calculate statistics
    max_gradient = np.max(np.abs(pressure_gradient))
    mean_gradient = np.mean(pressure_gradient)
    
    # Get terminus gradient (last ~5 points)
    terminus_points = min(5, len(pressure_gradient))
    terminus_gradient = pressure_gradient[-terminus_points:]
    
    # Create output directory
    analysis_dir = os.path.join(output_dir, "pressure_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Get profile ID for filename
    profile_id = data.get('profile_id')
    profile_suffix = f"_profile{profile_id:03d}" if profile_id is not None else ""
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Pressure vs. distance
    ax1.plot(sorted_x, sorted_p, 'b-o', markersize=4)
    ax1.axvline(x=terminus_x, color='r', linestyle='--', label='Terminus')
    ax1.set_title(f'Ice Pressure Near Terminus at t={time_val:.1f} years{" (Profile " + str(profile_id) + ")" if profile_id else ""}')
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Pressure (Pa)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Pressure gradient vs. distance
    ax2.plot(gradient_positions, pressure_gradient, 'g-o', markersize=4)
    ax2.axvline(x=terminus_x, color='r', linestyle='--', label='Terminus')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_title('Pressure Gradient Near Terminus')
    ax2.set_xlabel('Distance (km)')
    ax2.set_ylabel('Pressure Gradient (Pa/km)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, f'terminus_pressure_analysis_step{step_index:02d}{profile_suffix}.png'), dpi=300)
    plt.close(fig)
    
    # Create additional visualization: 2D pressure field
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create triangulation
    triang = tri.Triangulation(x, y, triangles=data['mesh']['elements'])
    
    # Create mask for poor quality triangles
    mask = create_mask(triang)
    triang.set_mask(mask)
    
    # Create pressure plot
    tc = ax.tripcolor(triang, pressure, cmap='plasma', shading='gouraud')
    
    # Highlight terminus region
    ax.triplot(triang, 'k-', lw=0.1, alpha=0.2)
    
    # Add a vertical line at terminus
    ax.axvline(x=terminus_x, color='r', linestyle='--', linewidth=2, label='Terminus')
    
    # Add title and labels
    ax.set_title(f'Pressure Field at t={time_val:.1f} years{" (Profile " + str(profile_id) + ")" if profile_id else ""}')
    ax.set_xlabel('Distance along flowline (km)')
    ax.set_ylabel('Elevation (km)')
    ax.grid(True, linestyle=":", color='k', alpha=0.4)
    ax.legend()
    
    # Add colorbar
    cbar = plt.colorbar(tc, ax=ax)
    cbar.set_label('Pressure (Pa)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, f'pressure_field_step{step_index:02d}{profile_suffix}.png'), dpi=300)
    plt.close(fig)
    
    # Print analysis results
    print("\nTerminus Pressure Gradient Analysis:")
    print(f"Time step: {step_index}, t = {time_val:.1f} years")
    print(f"Maximum gradient magnitude: {max_gradient:.2e} Pa/km")
    print(f"Mean gradient: {mean_gradient:.2e} Pa/km")
    print(f"Mean gradient at terminus: {np.mean(terminus_gradient):.2e} Pa/km")
    
    # Check for artificial pressure buildup
    if np.mean(terminus_gradient) > 0:
        print("WARNING: Positive pressure gradient at terminus indicates potential artificial pressure buildup")
    elif np.mean(terminus_gradient) < -1e6:  # Example threshold
        print("WARNING: Very negative pressure gradient at terminus may indicate numerical issues")
    
    # Return analysis results
    return {
        'max_gradient': max_gradient,
        'mean_gradient': mean_gradient,
        'terminus_gradient': np.mean(terminus_gradient),
        'step_index': step_index,
        'time': time_val
    }


def analyze_pressure_gradient_evolution(data, output_dir):
    """Analyze how pressure gradients evolve throughout the simulation
    
    Args:
        data: Dictionary containing mesh and field data
        output_dir: Directory to save output
    """
    if 'transient' not in data or 'Pressure' not in data['transient']:
        print("No pressure data available for time series analysis")
        return
    
    # Determine number of time steps
    num_steps = data['transient']['Pressure'].shape[0]
    times = data['transient']['time'] if 'time' in data['transient'] else np.arange(num_steps)
    
    # Get profile ID for filename
    profile_id = data.get('profile_id')
    profile_suffix = f"_profile{profile_id:03d}" if profile_id is not None else ""
    
    # Create output directory
    analysis_dir = os.path.join(output_dir, "pressure_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Analyze each time step
    max_gradients = []
    mean_gradients = []
    terminus_gradients = []
    
    # Use a subset of time steps if there are many
    step_indices = list(range(0, num_steps, max(1, num_steps // 10)))
    if (num_steps - 1) not in step_indices:
        step_indices.append(num_steps - 1)
    
    for step_index in step_indices:
        print(f"Analyzing pressure gradient at step {step_index}/{num_steps-1}...")
        result = analyze_terminus_pressure_gradient(data, output_dir, step_index)
        if result:
            max_gradients.append(result['max_gradient'])
            mean_gradients.append(result['mean_gradient'])
            terminus_gradients.append(result['terminus_gradient'])
    
    # Create evolution plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot gradient evolution
    ax.plot(times[step_indices], max_gradients, 'r-o', label='Maximum Gradient Magnitude')
    ax.plot(times[step_indices], mean_gradients, 'g-o', label='Mean Gradient')
    ax.plot(times[step_indices], terminus_gradients, 'b-o', label='Terminus Gradient')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Add title and labels
    ax.set_title(f'Pressure Gradient Evolution{" (Profile " + str(profile_id) + ")" if profile_id else ""}')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Pressure Gradient (Pa/km)')
    ax.grid(True)
    ax.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, f'pressure_gradient_evolution{profile_suffix}.png'), dpi=300)
    plt.close(fig)
    
    print(f"Pressure gradient evolution analysis complete")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_results.py <netcdf_file> [--output-dir=<directory>]")
        print("   or: python extract_results.py --pattern=<file_pattern> [--output-dir=<directory>]")
        print("\nExamples:")
        print("  python extract_results.py flowline9_profile_001.nc")
        print("  python extract_results.py --pattern='flowline9_profile_*.nc' --output-dir=results")
        sys.exit(1)
    
    # Get output directory
    output_dir = "."
    pattern = None
    
    # Parse command line arguments
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--output-dir="):
            output_dir = arg.split("=")[1]
        elif arg.startswith("--pattern="):
            pattern = arg.split("=")[1]
    
    if pattern:
        # Process multiple files
        process_multiple_files(pattern, output_dir)
    else:
        # Process a single file
        filename = sys.argv[1]
        extract_visualize_issm_results(filename, output_dir)