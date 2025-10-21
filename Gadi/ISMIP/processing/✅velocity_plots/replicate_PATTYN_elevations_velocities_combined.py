import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from scipy import signal

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
    """Extracts surface velocity centreline profile and surface elevation from a single NetCDF file."""
    md = reconstruct_mesh(filepath) # Keep this step for workflow consistency
    try:
        with nc.Dataset(filepath, 'r') as ds:
            tsol = ds['results/TransientSolution']
            last_step_idx = len(tsol.variables['time'][:]) - 1
            
            # Get velocity magnitude and surface elevation at last timestep
            vel_magnitude = tsol.variables['Vel'][last_step_idx, :]
            surface_elevation = tsol.variables['Surface'][last_step_idx, :]
            
            surface_indices = np.where(md.mesh.vertexonsurface)[0]

            # --- CENTERLINE EXTRACTION ---
            y_centre = 50000  # Define the geometric centre of the domain

            # 1. Get all unique y-coordinates from the surface mesh nodes
            unique_y_coords = np.unique(md.mesh.y[surface_indices])
            
            # 2. Find which of these unique y-coordinates is closest to the centre
            closest_y_to_centre = unique_y_coords[np.argmin(np.abs(unique_y_coords - y_centre))]
            print(f"    Found mesh centreline at y={closest_y_to_centre:.1f}m")

            # 3. Select all nodes that lie on this identified centreline
            # Use a very small tolerance for float comparison
            surface_centreline_indices = surface_indices[
                np.abs(md.mesh.y[surface_indices] - closest_y_to_centre) < 1e-6
            ]
            
            if len(surface_centreline_indices) == 0:
                print(f"    FATAL: Could not find any centreline nodes. Check mesh generation.")
                return None
        
            print(f"    Found {len(surface_centreline_indices)} surface nodes along centreline.")

            # Extract scenario identifier from filename (e.g., S3, S4)
            filename = os.path.basename(filepath)
            scenario = "Unknown"
            try:
                # Look for pattern like IsmipF_S3_... or flat_bed_S4_...
                parts = filename.split('_')
                for part in parts:
                    if part.startswith('S') and len(part) == 2 and part[1].isdigit():
                        scenario = part
                        break
            except:
                pass

            return {
                'x_surf': md.mesh.x[surface_centreline_indices],
                'vel_surf': vel_magnitude[surface_centreline_indices],
                'surface_elev': surface_elevation[surface_centreline_indices],
                'filename': os.path.basename(filepath),
                'scenario': scenario
            }
    
    except Exception as e:
        print(f"Warning: Could not process {os.path.basename(filepath)}. Reason: {e}", file=sys.stderr)
        return None

def detrend_elevation(x, elevation):
    """Remove linear trend from elevation data."""
    # Sort by x to ensure proper ordering
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    elev_sorted = elevation[sorted_indices]
    
    # Fit linear trend and remove it
    detrended = signal.detrend(elev_sorted, type='linear')
    
    # Return in original order
    original_order = np.argsort(sorted_indices)
    return detrended[original_order]

def main():
    """Processes all .nc files given and generates a combined plot."""
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} *.nc", file=sys.stderr)
        sys.exit(1)

    files = sorted([f for arg in sys.argv[1:] for f in glob.glob(arg)])
    if not files:
        print("Error: No files found.", file=sys.stderr)
        sys.exit(1)

    plot_data = [data for data in (extract_velocity_data(f) for f in files) if data]

    if not plot_data:
        print("Error: No valid data could be extracted for plotting.", file=sys.stderr)
        sys.exit(1)
        
    # Find minimum x value for tick placement
    min_x = min(np.min(data['x_surf'])/1000 - 50 for data in plot_data)
    
    # --- PLOTTING SETUP ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- PLOT 1: DETRENDED ELEVATION ---
    for data in plot_data:
        sort_surf = np.argsort(data['x_surf'])
        x_sorted = data['x_surf'][sort_surf]
        elev_detrended = detrend_elevation(data['x_surf'], data['surface_elev'])
        ax1.plot(x_sorted/1000 - 50, elev_detrended[sort_surf], label=data['filename'])
        
        # y-axes limits for replication of Pattyn's plot ONLY:
        elevation_y_limits = (-30, 50)

        if data['scenario'] == "S1":
            velocity_y_limits = (91, 100)
        elif data['scenario'] == "S3":
            velocity_y_limits = (186, 200)

    # --- PLOT 2: SURFACE VELOCITY ---
    
    # Store plot lines and labels for a combined legend later
    lines_left, labels_left = [], []
    lines_right, labels_right = [], []

    # Single loop to plot velocities, alternating between left and right axes
    for i, data in enumerate(plot_data):
        sort_surf = np.argsort(data['x_surf'])
        x_sorted = data['x_surf'][sort_surf]
        vel_sorted = data['vel_surf'][sort_surf]

        # Use the index 'i' to decide which axis to plot on
        if i % 2 == 0:  # Even index (0, 2, 4...) -> plot on left axis (ax2)
            # The comma after 'line' is important, as plot returns a list of lines
            line, = ax2.plot(x_sorted/1000 - 50, vel_sorted, label=data['filename'])
            lines_left.append(line)
            labels_left.append(data['filename'])
        else:  # Odd index (1, 3, 5...) -> plot on right axis (ax2_right)
            line, = ax2_right.plot(x_sorted/1000, vel_sorted, color="C1", label=data['filename'])
            lines_right.append(line)
            labels_right.append(data['filename'])

    # --- FORMATTING AND LABELS ---
    ax1.set(title='Final Surface Elevation (Detrended)', xlabel='Distance from centre (km)', ylabel='Elevation(m)')
    # to replicate Pattyn's Plot (ONLY)
    ax1.set_ylim(elevation_y_limits[0], elevation_y_limits[1])
    ax1.set_xlim(-50, 50)

    ax1.legend(loc='best')
    # Custom grid - horizontal lines plus vertical line at x=0
    ax1.grid(True, axis='y', linestyle=':')
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.7)
    # Custom x-axis ticks
    ax1.set_xticks([min_x, -50, 0, 50])

    # Define colors for each axis to match the plot lines
    left_axis_color = 'k'  # Default color for the first plot on ax2

    # Collect scenarios for axis labels
    scenarios_left = [plot_data[i]['scenario'] for i in range(len(plot_data)) if i % 2 == 0]
    
    ax2.set(title='Final Surface Velocity', xlabel='Distance from centre (km)')
    
    # Create y-axis labels with scenario identifiers
    left_label = f"Vel Mag (m/yr) - {', '.join(scenarios_left)}" if scenarios_left else "Vel Mag (m/yr)"
    
    ax2.set_ylabel(left_label, color=left_axis_color)
    ax2.tick_params(axis='y', labelcolor=left_axis_color)
    ax2.set_ylim(velocity_y_limits[0], velocity_y_limits[1])
    ax2.set_xlim(-50, 50)
    
    # Custom grid - horizontal lines plus vertical line at x=0
    ax2.grid(True, axis='y', linestyle=':')
    ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.7)
    # Custom x-axis ticks
    ax2.set_xticks([min_x, -50, 0, 50])
    
    ax2.legend()

    output_filename = f"combined_elevation_detrended_surface_velocity_{scenarios_left}.png"#_{scenarios_right}.png"
    plt.tight_layout()
    plt.savefig(output_filename, dpi=500)
    plt.show()
    plt.close()
    print(f"✓ Plot saved to {output_filename}")

if __name__ == "__main__":
    main()