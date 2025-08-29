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

# --- Setup ---
L = 210  # km
author = 'AFH'

PROFILE_DIR = os.path.join(parent_dir, "bedrock_profiles")

bedrock_config = SyntheticBedrockModelConfig(profile_id=profile_id, output_dir=PROFILE_DIR)

# Read velocity slice
xhat, vx_s, vx_b = read_slice(f'{fname}')

# constants & material properties
yts = 31556926 # s/yr
A = 1e-16 / yts # from table 1 in Pattyn 2008
rheology_n = 3
rheology_B = A ** (-1/rheology_n)
eta = rheology_B / 2

# Bed elevation from profile
bed_elevation = bedrock_config.get_bedrock_elevation(xhat * L * 1e3)  # xhat → meters

# bedrock parameters
wavelength = bedrock_config.profile_params['wavelength']  # meters
omega = 2 * np.pi / wavelength # 1/meters
amplitude = bedrock_config.profile_params['amplitude'] # meters
beta_1 = amplitude  # bed slope amplitude in meters

V_avg = np.mean(vx_b)

# Budd's basal shear stress
x_coords = xhat * L * 1e3 # in meters

dx = np.diff(x_coords) # spacing in meters
dbed = np.diff(bed_elevation)
bed_slope = dbed / dx

tau_xz = 2 * eta * V_avg * omega * beta_1 * np.cos(omega * x_coords)

# Smooth version (large-scale trend)
tau_xz_smooth = savgol_filter(tau_xz, window_length=101, polyorder=3)
# Subtract trend to isolate oscillations
tau_xz_oscillations = tau_xz - tau_xz_smooth

# Normalize to the range of oscillations
max_dev = np.max(np.abs(tau_xz_oscillations))
norm = colors.Normalize(vmin=-max_dev, vmax=+max_dev)

# --- Plot ---

# Plot with color representing oscillations
plt.figure(figsize=(16, 5))
sc = plt.scatter(xhat, bed_elevation, s=10, c=tau_xz_oscillations, cmap='coolwarm', norm=norm)
cbar = plt.colorbar(sc)
cbar.set_label('Shear stress oscillations (Pa)')
plt.grid(True, linestyle=":", color='k', alpha=0.4)
# plt.xlim(0, 1)
plt.xlabel("Normalized x")
plt.ylabel("Bed elevation (m)")
plt.title("internal shear stress (at z=0) over bed topography", loc='left', fontsize=10)
plt.tight_layout()
plt.savefig(f"{profile_id:03d}_{exp}_internal_SS_at_z=0_color.png", dpi=300)
# plt.show()


