#!/usr/bin/env python
"""
Improved ISSM Flow Visualization
This script properly visualizes ISSM flow results using triangular mesh plotting.
"""

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os
import sys

def create_mask(triang):
    """Create a mask for triangulation to remove flat triangles
    
    Args:
        triang: Matplotlib triangulation object
    
    Returns:
        Mask array for the triangulation
    """
    analyzer = tri.TriAnalyzer(triang)
    return analyzer.get_flat_tri_mask(0.02)

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
    
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Visualize the data
    if 'mesh' in data and 'transient' in data:
        # Determine the number of time steps
        num_steps = data['time']['actual_steps'] if 'actual_steps' in data['time'] else 5
        
        print(f"Creating visualizations for {num_steps} time steps...")
        
        # Create folders for different plot types
        field_dirs = {}
        for field in ['Pressure', 'Vx', 'Vel', 'Thickness']:
            if field in data['transient']:
                field_dir = os.path.join(output_dir, field)
                os.makedirs(field_dir, exist_ok=True)
                field_dirs[field] = field_dir
        
        # Create visualizations for each time step
        for step in range(num_steps):
            time_val = data['transient']['time'][step] if 'time' in data['transient'] else step
            
            for field, field_dir in field_dirs.items():
                if field in data['transient'] and step < data['transient'][field].shape[0]:
                    file_path = os.path.join(field_dir, f"{field}_step{step:02d}.png")
                    
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
            create_combined_plots(data, output_dir)
    
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
    
    # If we have evolving geometry, transform the coordinates
    if 'transient' in data and 'Base' in data['transient'] and 'Thickness' in data['transient']:
        try:
            # Get initial geometry
            if 'geometry' in data:
                initial_base = data['geometry']['base']
                initial_thickness = data['geometry']['thickness']
                
                # Get evolved geometry
                evolved_base = data['transient']['Base'][step]
                evolved_thickness = data['transient']['Thickness'][step]
                
                # Calculate relative height in the initial geometry
                rel_height = (y - initial_base) / initial_thickness
                rel_height = np.clip(rel_height, 0, 1)  # Ensure values are between 0 and 1
                
                # Transform to new geometry
                y = evolved_base + rel_height * evolved_thickness
            
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
    
    # Add title and labels
    ax.set_title(f'{field_name} at t={time_val:.1f} years')
    ax.set_xlabel('Distance along flowline (km)')
    ax.set_ylabel('Elevation (km)')
    
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
        
        # Plot 1: Velocity field
        if 'Vx' in data['transient']:
            velocity = data['transient']['Vx'][step]
            vmax = max(abs(velocity.min()), abs(velocity.max()))
            tc1 = ax1.tripcolor(triang, velocity, cmap='coolwarm', shading='gouraud', 
                               vmin=-vmax, vmax=vmax)
            ax1.triplot(triang, 'k-', lw=0.1, alpha=0.2)
            ax1.set_title(f'Ice Velocity (Vx) at t={time_val:.1f} years')
            ax1.set_ylabel('Elevation (km)')
            cbar1 = fig.colorbar(tc1, ax=ax1)
            cbar1.set_label('Velocity (km/yr)')
        
        # Plot 2: Pressure field
        if 'Pressure' in data['transient']:
            pressure = data['transient']['Pressure'][step]
            tc2 = ax2.tripcolor(triang, pressure, cmap='plasma', shading='gouraud')
            ax2.triplot(triang, 'k-', lw=0.1, alpha=0.2)
            ax2.set_title(f'Ice Pressure at t={time_val:.1f} years')
            ax2.set_xlabel('Distance along flowline (km)')
            ax2.set_ylabel('Elevation (km)')
            cbar2 = fig.colorbar(tc2, ax=ax2)
            cbar2.set_label('Pressure')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(combined_dir, f'combined_step{step:02d}.png'), dpi=300)
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
    ax.set_title('Velocity Evolution Over Time')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Velocity (km/yr)')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'velocity_evolution.png'), dpi=300)
    plt.close(fig)
    
    print(f"Created velocity evolution plot")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_issm_flow.py <netcdf_file> [--output-dir=<directory>]")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # Get output directory
    output_dir = "."
    for arg in sys.argv[2:]:
        if arg.startswith("--output-dir="):
            output_dir = arg.split("=")[1]
    
    # Extract and visualize the results
    extract_visualize_issm_results(filename, output_dir)