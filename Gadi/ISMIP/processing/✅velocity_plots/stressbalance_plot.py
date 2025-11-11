import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.signal import savgol_filter
from loadmodel import loadmodel

# Set font sizes to match transfer_analysis.py
plt.rcParams.update({
    'font.size': 12,          # Default font size
    'axes.titlesize': 16,     # Title font size
    'axes.labelsize': 14,     # Axis label font size
    'xtick.labelsize': 12,    # X-axis tick label size
    'ytick.labelsize': 12,    # Y-axis tick label size
    'legend.fontsize': 12,    # Legend font size
    'figure.titlesize': 16    # Figure title font size
})

# --- CONFIGURATION ---
SLICE_Y_COORDINATE = 50000  # meters
TOLERANCE = 500             # meters

def extract_slice(md, y_coord, tol):
    """Extracts and sorts a 2D data slice from the ISSM model."""
    s_idx = np.where((md.mesh.vertexonsurface == 1) & (abs(md.mesh.y - y_coord) < tol))[0]
    b_idx = np.where((md.mesh.vertexonbase == 1) & (abs(md.mesh.y - y_coord) < tol))[0]

    if s_idx.size == 0:
        raise ValueError(f"No surface nodes found for slice at y={y_coord}")

    x = md.mesh.x[s_idx]
    x_hat = (x - x.min()) / (x.max() - x.min())
    
    sort_order = np.argsort(x_hat)
    
    return {
        'x_hat': x_hat[sort_order],
        'vx_s': md.results.StressbalanceSolution.Vx[s_idx][sort_order],
        'vx_b': md.results.StressbalanceSolution.Vx[b_idx][sort_order],
        'bed': md.geometry.base[b_idx][sort_order]
    }

def main(filepath):
    """Loads a model, extracts a slice, and generates two plots."""
    base_filename = os.path.splitext(os.path.basename(filepath))[0]
    md = loadmodel(filepath)

    if not hasattr(md.results, 'StressbalanceSolution'):
        raise ValueError("StressbalanceSolution not found in the model.")

    data = extract_slice(md, SLICE_Y_COORDINATE, TOLERANCE)
    x_hat, vx_s, vx_b, bed = data['x_hat'], data['vx_s'], data['vx_b'], data['bed']

    # --- Plot 1: Velocity Profile ---
    plt.figure(figsize=(16, 5))
    plt.plot(x_hat, vx_s, lw=2, label='Surface Velocity (Vx)')
    plt.plot(x_hat, vx_b, lw=2, label='Basal Velocity (Vx)')
    plt.xlim(0, 1)
    plt.xlabel("Normalized x-distance")
    plt.ylabel("Velocity (m a⁻¹)")
    plt.title(f"Velocity Profile for {base_filename}")
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{base_filename}_velocity.png", dpi=300)
    plt.close()
    print(f"Saved figure: {base_filename}_velocity.png")

    # --- Plot 2: Velocity Oscillations on Bed Topography ---
    win_len = min(101, len(vx_b) - (1 if len(vx_b) % 2 == 0 else 0))
    vx_b_osc = vx_b - savgol_filter(vx_b, window_length=win_len, polyorder=3)
    norm = colors.Normalize(vmin=-abs(vx_b_osc).max(), vmax=abs(vx_b_osc).max())

    plt.figure(figsize=(16, 5))
    sc = plt.scatter(x_hat, bed, s=10, c=vx_b_osc, cmap='viridis', norm=norm)
    plt.colorbar(sc, label='Basal Velocity Oscillations (m a⁻¹)')
    plt.xlim(0, 1)
    plt.xlabel("Normalized x-distance")
    plt.ylabel("Bed Elevation (m)")
    plt.title(f"Basal Velocity Oscillations for {base_filename}")
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{base_filename}_velocity_color.png", dpi=300)
    plt.close()
    print(f"Saved figure: {base_filename}_velocity_color.png")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(f"Usage: python {sys.argv[0]} <path_to_your_netcdf_file.nc>")
    main(sys.argv[1])