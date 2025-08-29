import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from types import SimpleNamespace

# --- ISSM Imports ---
# Add the specific path to pyISSM source, as done in the simulation script
import sys
sys.path.append('/home/ana/pyISSM/src')
import pyissm as issm
from pyissm import plot as iplt
from model import model
from bamgflowband import bamgflowband

# --- Bedrock Generator Import ---
# This assumes 'bedrock_generator.py' is in the same directory
# or in your PYTHONPATH.
try:
    from bedrock_generator import SyntheticBedrockModelConfig
except ImportError:
    print("Error: Could not import SyntheticBedrockModelConfig.")
    print("Please ensure 'bedrock_generator.py' is accessible.")
    sys.exit(1)


def adaptive_bamg(md, x, s0, b, bed_wavelength, ice_thickness, resolution_factor=1.0):
    """
    Recreates the adaptive mesh. This function is copied from the simulation script
    but modified to accept necessary parameters directly.
    """
    print("\n============ RECREATING MESH ============")
    # hmax: Resolution based on wavelength
    wavelength_thickness_ratio = bed_wavelength / ice_thickness

    if bed_wavelength < 15000:
        refinement_factor = 50
    else:
        refinement_factor = 200

    hmax = (bed_wavelength / refinement_factor) * resolution_factor

    # Generate the mesh
    md = bamgflowband(md, x, s0, b,
                      'hmax', hmax,
                      'anisomax', 3,
                      'vertical', 1)

    print(f"\n[ADAPTIVE_BAMG] Mesh statistics:")
    print(f"  wavelength_thickness_ratio: {wavelength_thickness_ratio:.2f}")
    print(f"  hmax: {hmax:.2f} m")
    print(f"  resolution_factor: {resolution_factor}")
    print(f"  refinement_factor: {refinement_factor}")
    print(f"  Total vertices: {md.mesh.numberofvertices}")
    print(f"  Elements: {md.mesh.numberofelements}")
    print(f"========================================")

    return md

def recreate_model_and_mesh(profile_id, resolution_factor):
    """
    Sets up the initial model object and generates the mesh based on the
    profile ID and resolution factor.
    """
    print(f"--- Initializing model for Profile ID: {profile_id}, Resolution: {resolution_factor} ---")
    # --- 1. Recreate Bedrock and Geometry Configuration ---
    bedrock_config = SyntheticBedrockModelConfig(profile_id=profile_id)
    bed_wavelength = bedrock_config.profile_params['wavelength']
    ice_thickness = bedrock_config.ice_thickness

    # --- 2. Recreate Domain ---
    L_buffer_inlet = 25e3
    L_interest = 160e3
    L_buffer_terminus = 25e3
    L = L_buffer_inlet + L_interest + L_buffer_terminus
    nx = int(L * 0.01)
    x_1D = np.linspace(0, L, nx)
    b = bedrock_config.get_bedrock_elevation(x_1D)
    s0 = b + ice_thickness

    # --- 3. Initialize a new ISSM model ---
    md = model()

    # --- 4. Generate the mesh ---
    md = adaptive_bamg(md, x_1D, s0, b, bed_wavelength, ice_thickness, resolution_factor)

    # --- 5. Set geometry on the mesh ---
    mesh_x = md.mesh.x
    bed_2d = np.interp(mesh_x, x_1D, b)
    surface_2d = np.interp(mesh_x, x_1D, s0)
    md.geometry.surface = surface_2d
    md.geometry.bed = bed_2d
    md.geometry.thickness = surface_2d - bed_2d
    md.geometry.base = md.geometry.bed

    return md

def plot_transient_fields(md, nc_filename, profile_id, exp):
    """
    Loads transient results from a NetCDF file, attaches them to the model object,
    and plots key fields from the last time step.
    """
    print(f"\n=== PLOTTING TRANSIENT FIELDS FROM {nc_filename} ===")

    try:
        # --- 1. Load data from the NetCDF file ---
        group_path = 'results/TransientSolution'
        ds = xr.open_dataset(nc_filename, group=group_path)

        # --- 2. Extract data for the FINAL time step and flatten it to 1D ---
        # The .values call returns a numpy array. We select the last time step [-1]
        # and then .flatten() to convert from shape (N, 1) to (N,).
        vel_final = ds['Vel'].values[-1, :].flatten()
        pressure_final = ds['Pressure'].values[-1, :].flatten()
        thickness_final = ds['Thickness'].values[-1, :].flatten()
        surface_final = ds['Surface'].values[-1, :].flatten()
        base_final = ds['Base'].values[-1, :].flatten()

        # --- 3. Now, we can call the plotting functions with the correctly shaped 1D arrays ---
        # Velocity magnitude
        plt.figure()
        iplt.plot_model_field(md, vel_final, show_cbar=True)
        plt.title('Velocity Magnitude (m/yr) - Final Step')
        plt.savefig(f"velocity_final_{profile_id:03d}_{exp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved velocity_final_{profile_id:03d}_{exp}.png")

        # Pressure magnitude
        plt.figure()
        iplt.plot_model_field(md, pressure_final, show_cbar=True)
        plt.title('Pressure (Pa) - Final Step')
        plt.savefig(f"pressure_final_{profile_id:03d}_{exp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved pressure_final_{profile_id:03d}_{exp}.png")

        # Thickness
        plt.figure()
        iplt.plot_model_field(md, thickness_final, show_cbar=True)
        plt.title('Thickness (m) - Final Step')
        plt.savefig(f"thickness_final_{profile_id:03d}_{exp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved thickness_final_{profile_id:03d}_{exp}.png")

        # Bed, base and surface elevations
        plt.figure(figsize=(16, 5))
        surface_idx = np.where(md.mesh.vertexonsurface == 1)[0]
        x_surface = md.mesh.x[surface_idx]
        sort_idx = np.argsort(x_surface)
        x_sorted = x_surface[sort_idx]

        # Use the flattened final-step arrays we created above
        surface_sorted = surface_final[surface_idx][sort_idx]
        base_sorted = base_final[surface_idx][sort_idx]
        bed_sorted = md.geometry.bed[surface_idx][sort_idx]

        plt.plot(x_sorted / 1000, surface_sorted, label='Surface')
        plt.plot(x_sorted / 1000, base_sorted, label='Base')
        plt.fill_between(x_sorted / 1000, base_sorted, surface_sorted, alpha=0.3)
        plt.plot(x_sorted / 1000, bed_sorted, color="brown", label='Bed')
        plt.legend()
        plt.xlabel("Distance (km)")
        plt.ylabel("Elevation (m)")
        plt.title('Elevations - Final Step')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(f"elevations_final_{profile_id:03d}_{exp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved elevations_final_{profile_id:03d}_{exp}.png")

    except Exception as e:
        print(f"An error occurred in plot_transient_fields: {e}")
        import traceback
        traceback.print_exc()


def plot_max_velocity_from_netcdf(filename):
    """
    Reads a NetCDF output file from ISSM and plots the evolution of maximum velocity.
    """
    print(f"\n=== PLOTTING MAX VELOCITY EVOLUTION FROM {filename} ===")
    group_path = 'results/TransientSolution'
    try:
        ds = xr.open_dataset(filename, group=group_path)
        time_steps = ds['time'].values
        velocity_data = ds['Vel']
        
        # --- FIX: Use the correct dimension name 'Vel_dim1' instead of 'VertNum' ---
        max_velocities = velocity_data.max(dim='Vel_dim1').values

        plt.figure(figsize=(10, 6))
        plt.scatter(time_steps, max_velocities, alpha=0.8, s=30)
        plt.xlabel('Time (years)')
        plt.ylabel('Maximum Velocity (m/yr)')
        plt.title('Evolution of Maximum Velocity')
        plt.grid(linestyle=':', alpha=0.7)

        base_name = filename.rsplit('.nc', 1)[0]
        output_plot_filename = f"{base_name}_max_vel_evolution.png"
        plt.savefig(output_plot_filename, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Plot saved: {output_plot_filename}")
        print(f"  Max velocity range: {np.min(max_velocities):.2f} to {np.max(max_velocities):.2f} m/yr")

    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
    except Exception as e:
        print(f"An error occurred in plot_max_velocity_from_netcdf: {e}")
        import traceback
        traceback.print_exc()


def find_and_process_netcdf():
    """
    Main function to find the NetCDF file, parse its name, and run the analysis.
    """
    # --- 1. Find the target NetCDF file in the current directory ---
    # Regex to match the pattern: (number)_(S followed by a number)_(number).nc
    # This pattern is simplified to be more robust
    pattern = re.compile(r"(\d+)_S(\d)_([\d.]+).*?\.nc")

    nc_files = glob.glob('*.nc')
    target_file = None

    for f in nc_files:
        if pattern.match(f):
            target_file = f
            break # Found a match

    if not target_file:
        print("Error: No matching NetCDF file found in the current directory.")
        print("Expected format: XXX_SY_Z... .nc (e.g., 165_S3_0.875.nc)")
        return

    print(f"--- Found target file: {target_file} ---")

    # --- 2. Parse the filename to get parameters ---
    match = pattern.match(target_file)
    if not match:
        print(f"Error: Filename '{target_file}' does not match expected format.")
        return

    profile_id = int(match.group(1))
    exp_num = int(match.group(2))
    resolution_factor = float(match.group(3))
    exp = f'S{exp_num}'

    print(f"Parsed parameters: Profile ID={profile_id}, Exp={exp}, Resolution={resolution_factor}")

    # --- 3. Recreate the model and mesh ---
    md = recreate_model_and_mesh(profile_id, resolution_factor)

    # --- 4. Run the analysis and plotting functions ---
    plot_transient_fields(md, target_file, profile_id, exp)
    plot_max_velocity_from_netcdf(target_file)

    print("\n--- Analysis complete. ---")


if __name__ == "__main__":
    find_and_process_netcdf()