"""
Usage:

python extract_final_step.py <transient_results.nc>

This script processes a single transient NetCDF output file from ISSM. It:
1.  Infers the Parameter File (e.g., 'IsmipF.py') and resolution_factor
    from the input filename.
2.  Reconstructs the original 3D model mesh by replicating the meshing
    and extrusion steps from the 'runme.py' script.
3.  Opens the .nc file and processes ONLY THE FINAL TIME STEP.
4.  Plots both surface and basal layers for velocity fields and basal for pressure.
5.  Saves final-state plots to a directory structure separated by layer.
6.  Additionally, it creates a summary plot of max velocity over time.

Example:
python extract_final_step.py IsmipF_S1_1-Transient.nc
"""
import os
import sys
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

# Add ISSM/pyISSM to path
sys.path.append('/home/ana/pyISSM/src')
from model import model
from squaremesh import squaremesh
from parameterize import parameterize
from pyissm import plot as iplt

def reconstruct_mesh(filename):
    """
    Reconstructs the 3D model and mesh based on the filename conventions
    and the process outlined in runme.py.
    """
    print("Reconstructing mesh from filename parameters...")
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split('_')
    
    param_filename = parts[0] + ".py"
    try:
        # This logic assumes the name will always have 4 parts.
        # parts[2] is the X factor, parts[3] is the Z factor.
        h_resolution_factor = float(parts[2])
        v_resolution_factor = float(parts[3].split('-')[0])
    except (ValueError, IndexError):
        print(f"Warning: Could not determine resolution factors from '{base}'.")
        print("         This script expects 4 parts in the name (e.g., IsmipF_S3_0.5_1.5-Transient).")
        print("         Defaulting to 1.0 for both factors.")
        h_resolution_factor = 1.0
        v_resolution_factor = 1.0

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    param_file_path = os.path.join(parent_dir, param_filename)
    
    print(f"  Using Parameter File: '{param_file_path}'")
    print(f"  Using Resolution Factor X: {h_resolution_factor}")
    print(f"  Using Resolution Factor Z: {v_resolution_factor}")
    
    if not os.path.exists(param_file_path):
        raise FileNotFoundError(f"parameterize error message: file '{param_filename}' not found at expected location '{param_file_path}'!")

    # Replicate steps from runme.py
    md = model()
    x_max = 100000
    y_max = 100000

    # The number of nodes must be an integer
    x_nodes = int(30 * h_resolution_factor)
    y_nodes = int(30 * h_resolution_factor)

    base_vertical_layers = 5
    num_layers = int(base_vertical_layers * v_resolution_factor)

    md = squaremesh(md, x_max, y_max, x_nodes, y_nodes)

    # Set the required miscellaneous attributes before parameterization
    md.miscellaneous.filename = parts[0]  # "IsmipF" from filename
    md.miscellaneous.scenario = parts[1]   # "S?" from filename
    md.miscellaneous.h_resolution_factor = h_resolution_factor
    md.miscellaneous.v_resolution_factor = v_resolution_factor

    md = parameterize(md, param_file_path)
    md = md.extrude(num_layers, 1)

    print(f"✅ Mesh reconstructed successfully ({md.mesh.numberofvertices} vertices, {md.mesh.numberofelements} elements).")
    return md


def plot_velocity_evolution(times_in_years, vel_data, out_dir):
    """Plots the evolution of maximum velocity over the whole simulation."""
    print("  Generating max velocity evolution plot...")
    try:
        # Calculate max velocity at each time step (axis=1 targets the spatial dimension)
        max_velocities = np.max(vel_data, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(times_in_years, max_velocities)
        plt.xlabel('Time (years)')
        plt.ylabel('Maximum Velocity (m/yr)')
        plt.title('Evolution of Maximum Velocity')
        plt.grid(linestyle=':', alpha=0.7)
        plt.savefig(os.path.join(out_dir, "velocity_evolution.png"), dpi=200, bbox_inches='tight')
        plt.close()
        print("    ✓ Max velocity evolution plot saved.")

    except Exception as e:
        print(f"    ✗ An error occurred while plotting max velocity: {e}")

def visualise_final_step(results_file):
    """
    Reconstructs a model mesh and plots the final transient step from a file.
    """
    start_time = time.time()
    print(f"\n{'='*60}\n📂 Processing Results File: {results_file}\n{'='*60}")

    try:
        md = reconstruct_mesh(results_file)
    except Exception as e:
        print(f"❌ Critical Error: Could not reconstruct the mesh.")
        print(f"   Error details: {e}")
        return

    try:
        print("Reading transient solution data...")
        ds = nc.Dataset(results_file, 'r')
        tsol_group = ds['results/TransientSolution']
        times_in_years = tsol_group.variables['time'][:]
        n_steps = len(times_in_years)
        print(f"✅ Found transient solution with {n_steps} time steps.")
    except Exception as e:
        print(f"❌ Critical Error: Could not read transient data from '{results_file}'.")
        print(f"   Error details: {e}")
        return
        
    if n_steps <= 1:
        print("ℹ️  File has only one time step. Skipping.")
        ds.close()
        return

    # --- PLOT VELOCITY EVOLUTION (SUMMARY PLOT) ---
    out_dir_base = os.path.splitext(os.path.basename(results_file))[0] + '_FINAL'
    os.makedirs(out_dir_base, exist_ok=True)
    if 'Vel' in tsol_group.variables:
        vel_all_steps = tsol_group.variables['Vel'][:]
        plot_velocity_evolution(times_in_years, vel_all_steps, out_dir_base)

    # --- PLOT FINAL STEP FOR ALL FIELDS AND LAYERS ---
    fields_and_layers_to_plot = [
        ('Vx', 'Surface'), ('Vx', 'Basal'),
        ('Vy', 'Surface'), ('Vy', 'Basal'),
        ('Vz', 'Surface'), ('Vz', 'Basal'),
        ('Vel', 'Surface'), ('Vel', 'Basal'),
        ('Pressure', 'Basal')
    ]
    
    last_step_index = n_steps - 1
    final_time_in_years = times_in_years[last_step_index]

    print(f"🚀 Generating plots for final time step (t={final_time_in_years:.2f} years)...")
    
    for field_name, layer_name in fields_and_layers_to_plot:
        if field_name not in tsol_group.variables:
            print(f"⚠️ Warning: Field '{field_name}' not found. Skipping.")
            continue

        print(f"  Plotting final state for {field_name} ({layer_name})...")
        
        data_for_final_step = np.squeeze(tsol_group.variables[field_name][last_step_index, :])

        # Convert data from meters/second to meters/year for plotting
        if field_name in ['Vx', 'Vy', 'Vz', 'Vel']:
            data_for_final_step = data_for_final_step
        
        # Additional check for shape mismatch before plotting
        if data_for_final_step.shape[0] != md.mesh.numberofvertices:
            print(f"\n❌ Mismatch Error: Data for '{field_name}' has {data_for_final_step.shape[0]} points, but reconstructed mesh has {md.mesh.numberofvertices} vertices. Skipping file.")
            ds.close()
            return

        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
        
        plot_kwargs = {'show_cbar': True}
        if field_name in ['Vx', 'Vy', 'Vz']: plot_kwargs['cmap'] = 'coolwarm'
        elif field_name == 'Vel': plot_kwargs['cmap'] = 'viridis'
        elif field_name == 'Pressure': plot_kwargs['cmap'] = 'plasma'
        
        if layer_name == 'Basal':
            plot_kwargs['layer'] = 1
            layer_label = "Base"
        else:
            layer_label = "Surface"
            
        title = f"{field_name} at {final_time_in_years:.4f} years ({layer_label}, Final Step)"

        iplt.plot_model_field(md, data_for_final_step, ax=ax, **plot_kwargs)
        ax.set_title(title, fontsize=14)
        
        out_dir_final = os.path.join(out_dir_base, layer_label, field_name)
        os.makedirs(out_dir_final, exist_ok=True)
        out_path = os.path.join(out_dir_final, f"final_{field_name}.png")
        
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    ds.close()
    print(f"\n🎉 All plots for final step completed.")
    print(f"⏱️  Total time for {results_file}: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_final_step.py <transient_results.nc> [another.nc ...]")
        print("   or: python extract_final_step.py *.nc")
        sys.exit(1)

    files_to_process = []
    for arg in sys.argv[1:]:
        files_to_process.extend(glob.glob(arg))
    
    if not files_to_process:
        print(f"Error: No files found matching: {sys.argv[1:]}")
        sys.exit(1)

    for results_file in files_to_process:
        visualise_final_step(results_file)

    print("\nAll files processed.")