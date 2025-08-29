import glob
import os
import sys
# Get parent directory and construct profile path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from bedrock_generator import SyntheticBedrockModelConfig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.signal import savgol_filter


if len(sys.argv) < 2:
    print("Usage: python basal_shear_stress_plots.py <profile_file>")
    sys.exit(1)

fname = sys.argv[1]
profile_id = int(os.path.basename(fname).split('_')[0])
exp = str(os.path.basename(fname).split('_')[1])


def read_slice(fname):
    d = np.loadtxt(fname, comments='#')
    xhat, vx_s, vz_s, vx_b = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    
    # Sort by x_hat to ensure proper ordering for plotting
    order = np.argsort(xhat)
    xhat_sorted = xhat[order]
    vx_s_sorted = vx_s[order]
    vx_b_sorted = vx_b[order]

    return xhat_sorted, vx_s_sorted, vx_b_sorted

# Main plotting
L = 175  # km
author = 'AFH'

tag   = f"{author}{L:03d}.txt"
files = glob.glob(tag)

# Read velocity slice
xhat, vx_s, vx_b = read_slice(f'{fname}')

plt.figure(figsize=(16, 5))
plt.plot(xhat, vx_s, lw=2, label='surface velocity')
plt.plot(xhat, vx_b, lw=2, label='basal velocity')
# plt.title(f"{L} km", loc='left', fontsize=10)
# plt.xlim(0, 1)
plt.xlabel("Normalized x")
plt.ylabel("Velocity (m a⁻¹)")
plt.grid(True, linestyle=":", color='k', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(f"{profile_id:03d}_{exp}_velocity.png", dpi=300)
# plt.show()

########################################################################################################################################################
# over bedrock
PROFILE_DIR = os.path.join(parent_dir, "bedrock_profiles")

bedrock_config = SyntheticBedrockModelConfig(profile_id=profile_id, output_dir=PROFILE_DIR)

# Bed elevation from profile
bed_elevation = bedrock_config.get_bedrock_elevation(xhat * L * 1e3)  # xhat → meters

# Smooth version (large-scale trend)
vx_b_smooth = savgol_filter(vx_b, window_length=101, polyorder=3)

# Subtract trend to isolate oscillations
vx_b_oscillations = vx_b - vx_b_smooth


# Normalize to the range of oscillations
max_dev = np.max(np.abs(vx_b_oscillations))
norm = colors.Normalize(vmin=-max_dev, vmax=+max_dev)

# --- Plot ---

# Plot with color representing oscillations
plt.figure(figsize=(16, 5))
sc = plt.scatter(xhat, bed_elevation, s=10, c=vx_b_oscillations, cmap='viridis', norm=norm)
cbar = plt.colorbar(sc)
cbar.set_label('basal velocity oscillations (m/y)')
plt.grid(True, linestyle=":", color='k', alpha=0.4)
# plt.xlim(0, 1)
plt.xlabel("Normalized x")
plt.ylabel("Bed elevation (m)")
plt.title("Basal velocity over bed topography", loc='left', fontsize=10)
plt.tight_layout()
plt.savefig(f"{profile_id:03d}_{exp}_velocity_color.png", dpi=300)

# plt.show()
