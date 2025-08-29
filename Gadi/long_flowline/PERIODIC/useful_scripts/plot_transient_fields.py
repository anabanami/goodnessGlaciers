import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('/home/ana/pyISSM/src')
import pyissm as issm
from pyissm import plot as iplt


def plot_transient_fields(md):                                                            
    """                                                                                                    
    Plot fields from the last time step of the transient evolution                                                                                                    
    """                                                                                                    
    print("\n=== PLOTTING TRANSIENT FIELDS (LAST TIME STEP) ===")                                          
    # Check if transient results exist
    if not hasattr(md.results, 'TransientSolution'):
        print("No TransientSolution found in md.results")
        return

    transient_solution = md.results.TransientSolution

    # velocity magnitude
    iplt.plot_model_field(md, transient_solution.Vel[:, -1], show_cbar=True)
    plt.title('Velocity Magnitude (m/yr)')
    plt.savefig(f"velocity_final_{BEDROCK_PROFILE_ID:03d}_{exp}.png", dpi=300, bbox_inches='tight')

    # Pressure magnitude
    iplt.plot_model_field(md, transient_solution.Pressure[:, -1], show_cbar=True)
    plt.title('Pressure Magnitude (Pa)')
    plt.savefig(f"pressure_final_{BEDROCK_PROFILE_ID:03d}_{exp}.png", dpi=300, bbox_inches='tight')

    # Thickness
    iplt.plot_model_field(md, transient_solution.Thickness[:,-1], show_cbar=True)
    plt.title('Thickness (m)')
    plt.savefig(f"thickness_final_{BEDROCK_PROFILE_ID:03d}_{exp}.png", dpi=300, bbox_inches='tight')

    # Bed, base and surface elevations
    plt.figure(figsize=(16, 5))
    # Use md.mesh.x directly to ensure correct dimensions
    mesh_x_full = md.mesh.x
    # Get surface nodes only for proper 1D profile plotting
    surface_idx = np.where(md.mesh.vertexonsurface == 1)[0]
    x_surface = md.mesh.x[surface_idx]
    # Sort by x position for proper line plotting
    sort_idx = np.argsort(x_surface)
    x_sorted = x_surface[sort_idx]
    surface_sorted = transient_solution.Surface[surface_idx, -1][sort_idx]
    base_sorted = transient_solution.Base[surface_idx, -1][sort_idx]
    bed_sorted = md.geometry.bed[surface_idx][sort_idx]

    plt.plot(x_sorted, surface_sorted, label='Surface')
    plt.plot(x_sorted, base_sorted, label='Base')
    plt.fill_between(x_sorted, base_sorted, surface_sorted, alpha=0.3)
    plt.plot(x_sorted, bed_sorted, color="brown", label='Bed')
    plt.legend()
    plt.title('Base & Surface Elevation (m)')
    plt.savefig(f"signals_elevations_final_{BEDROCK_PROFILE_ID:03d}_{exp}.png", dpi=300, bbox_inches='tight')

    plt.show()

    # IS THIS THE MESH AT THE LAST TIME STEP ?
    visual_mesh_check(md)


import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

def plot_max_velocity_from_netcdf(filename):
    """
    Reads a NetCDF output file from ISSM and plots the evolution of maximum velocity.
    """
    print(f"\n=== PLOTTING MAX VELOCITY EVOLUTION FROM {filename} ===")
    
    # Define the path to the results group within the NetCDF file
    group_path = 'results/TransientSolution'
    
    try:
        # 1. Open the specific group within the NetCDF dataset
        ds = xr.open_dataset(filename, group=group_path)
        
        # 2. Extract the time and velocity data
        time_steps = ds['time'].values
        velocity_data = ds['Vel']
        
        # 3. Calculate the maximum velocity at each time step
        #    From your ncdump, the spatial dimension is named 'VertNum'.
        max_velocities = velocity_data.max(dim='VertNum').values
        
        # 4. Create the plot
        plt.figure(figsize=(10, 6))
        plt.scatter(time_steps, max_velocities, alpha=0.8, s=30)
        plt.xlabel('Time (years)')
        plt.ylabel('Maximum Velocity (m/yr)')
        plt.title('Evolution of Maximum Velocity (from NetCDF)')
        plt.grid(linestyle=':', alpha=0.3)
        
        # Save the figure with a new name
        base_name = filename.rsplit('.', 1)[0]
        plt.savefig(f"{base_name}_max_vel_evolution.png", dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
    
        print(f"✓ Plot saved for {filename}")
        print(f"  Max velocity range: {np.min(max_velocities):.2f} to {np.max(max_velocities):.2f} m/yr")

    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nDEBUGGING INFO:")
        print("Please ensure the group path and dimension names are correct for your NetCDF file.")
        print(f"Attempted to open group: '{group_path}'")

