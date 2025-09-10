"""
Usage:

python extract_results.py <transient_results.nc>

This script processes a single transient NetCDF output file from ISSM. It:
1.  Infers the Parameter File (e.g., 'IsmipF.py') and resolution_factor
    from the input filename.
2.  Reconstructs the original 3D model mesh by replicating the meshing
    and extrusion steps from the 'runme.py' script.
3.  Opens the <transient_results.nc> file to read the solution data.
4.  Plots the surface and basal layers for all specified fields.
5.  Saves the plots into a directory structure separated by layer
    (e.g., ./RunName/Surface/ and ./RunName/Base/).

Example:
python extract_results.py IsmipF_S1_1-Transient.nc
"""
import os
import sys
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

# Add ISSM/pyISSM to path
# Please ensure this path is correct for your system
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
        # parts[2] is the X factor, parts[3] is the Y factor.
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
    print(f"  Using Resolution Factor Y: {v_resolution_factor}")
    
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

    md = parameterize(md, param_file_path)
    md = md.extrude(num_layers, 1)

    print(f"✅ Mesh reconstructed successfully ({md.mesh.numberofvertices} vertices, {md.mesh.numberofelements} elements).")
    return md

def visualise_file(results_file):
    """
    Reconstructs a model mesh and plots transient results from a file.
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

    fields_and_layers_to_plot = [
        ('Vx', 'Surface'), ('Vx', 'Basal'),
        ('Vy', 'Surface'), ('Vy', 'Basal'),
        ('Vz', 'Surface'), ('Vz', 'Basal'),
        ('Vel', 'Surface'), ('Vel', 'Basal'),
        ('Pressure', 'Basal')
    ]
    
    out_dir_base = os.path.splitext(os.path.basename(results_file))[0]

    plot_start = time.time()
    total_plots = (n_steps - 1) * len(fields_and_layers_to_plot)
    plot_count = 0
    print(f"🚀 Starting to generate {total_plots} plots...")

    for i in range(1, n_steps):
        current_time_in_years = times_in_years[i]
        
        for field_name, layer_name in fields_and_layers_to_plot:
            plot_count += 1
            progress = (plot_count / total_plots) * 100
            print(f"\r  Plotting... [{progress:3.0f}%] - Time step {i}/{n_steps-1}, Field: {field_name} ({layer_name})", end="")

            if field_name not in tsol_group.variables:
                if i == 1: print(f"\n⚠️ Warning: Field '{field_name}' not found. Skipping.")
                continue

            data_for_step = np.squeeze(tsol_group.variables[field_name][i, :])

            if field_name in ['Vx', 'Vy', 'Vz', 'Vel']:
                data_for_step = data_for_step
            
            # Additional check for shape mismatch before plotting
            if data_for_step.shape[0] != md.mesh.numberofvertices:
                print(f"\n❌ Mismatch Error: Data for '{field_name}' has {data_for_step.shape[0]} points, but reconstructed mesh has {md.mesh.numberofvertices} vertices. Skipping file.")
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
            
            title = f"{field_name} at {current_time_in_years:.4f} years ({layer_label})"
            iplt.plot_model_field(md, data_for_step, ax=ax, **plot_kwargs)
            ax.set_title(title, fontsize=14)
            
            out_dir_final = os.path.join(out_dir_base, layer_label, field_name)
            os.makedirs(out_dir_final, exist_ok=True)
            
            out_path = os.path.join(out_dir_final, f"{field_name}_step{i:03d}.png")
            
            fig.savefig(out_path, dpi=200)
            plt.close(fig)

    ds.close()
    print(f"\n\n🎉 All {plot_count} plots completed.")
    print(f"⏱️  Plotting time: {time.time() - plot_start:.2f}s")
    print(f"⏱️  Total time for {results_file}: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_results.py <transient_results.nc>")
        sys.exit(1)

    results_file = sys.argv[1]
    
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at '{results_file}'")
        sys.exit(1)
        
    visualise_file(results_file)