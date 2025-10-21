import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

# Set font sizes
plt.rcParams.update({
    'font.size': 12,          # Default font size
    'axes.titlesize': 16,     # Title font size
    'axes.labelsize': 14,     # Axis label font size
    'xtick.labelsize': 12,    # X-axis tick label size
    'ytick.labelsize': 12,    # Y-axis tick label size
    'legend.fontsize': 12,    # Legend font size
    'figure.titlesize': 16    # Figure title font size
})

# Attempt to import pyISSM modules, with silent placeholders on failure
try:
    sys.path.append('/home/ana/pyISSM/src')
    from model import model
    from squaremesh import squaremesh
    from parameterize import parameterize
except ImportError:
    class model:
        def __init__(self):
            self.miscellaneous = type('miscellaneous', (object,), {})()
        def extrude(self, *args): return self
    def squaremesh(md, *args): return md
    def parameterize(md, *args): return md

def reconstruct_mesh(filename):
    """Placeholder for mesh reconstruction logic if needed by the parameterize script."""
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split('_')
    param_filename = parts[0] + ".py"
    try:
        h_res = float(parts[2])
        v_res = float(parts[3].split('-')[0])
    except (ValueError, IndexError):
        h_res, v_res = 1.0, 1.0

    param_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), param_filename)
    if not os.path.exists(param_file_path): return None
    
    md = model()
    md = squaremesh(md, 100000, 100000, int(30 * h_res), int(30 * h_res))
    md.miscellaneous.filename, md.miscellaneous.scenario = parts[0], parts[1]
    md.miscellaneous.h_resolution_factor, md.miscellaneous.v_resolution_factor = h_res, v_res
    md = parameterize(md, param_file_path)
    md = md.extrude(int(5 * v_res), 1)
    return md

def extract_velocity_data(filepath):
    """Extracts time and max velocity from a single NetCDF file."""
    reconstruct_mesh(filepath) # Keep this step for workflow consistency
    try:
        with nc.Dataset(filepath, 'r') as ds:
            tsol = ds['results/TransientSolution']
            times = tsol.variables['time'][:]
            if len(times) > 1 and 'Vel' in tsol.variables:
                max_vels = np.max(tsol.variables['Vel'][:], axis=1)
                label = os.path.basename(filepath).replace('.nc', '')
                return times, max_vels, label
    except Exception as e:
        print(f"Warning: Could not process {os.path.basename(filepath)}. Reason: {e}", file=sys.stderr)
    return None

def main():
    """Processes all .nc files given and generates a combined plot."""
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} *.nc", file=sys.stderr)
        sys.exit(1)

    files = sorted([f for arg in sys.argv[1:] for f in glob.glob(arg)])
    if not files:
        print("Error: No files found.", file=sys.stderr)
        sys.exit(1)

    # Process files and filter out failures in one line
    plot_data = [data for data in (extract_velocity_data(f) for f in files) if data]

    if not plot_data:
        print("Error: No valid data could be extracted for plotting.", file=sys.stderr)
        sys.exit(1)
        
    # Create the plot
    plt.figure(figsize=(12, 8))
    for times, max_vels, label in plot_data:
        plt.plot(times, max_vels, label=label, alpha=0.8)

    plt.xlabel('Time (years)')
    plt.ylabel('Maximum Velocity (m/yr)')
    plt.title('Evolution of Maximum Velocity')
    plt.grid(linestyle=':', alpha=0.7)
    
    legend_opts = {'title': 'Simulations'}
    if len(plot_data) > 5:
        legend_opts.update({'bbox_to_anchor': (1.04, 1), 'loc': 'upper left'})
    plt.legend(**legend_opts)

    output_filename = "combined_velocity_evolution.png"
    plt.savefig(output_filename, dpi=200)
    plt.tight_layout()
    plt.show()
    plt.close()
    print(f"✓ Plot saved to {output_filename}")

if __name__ == "__main__":
    main()