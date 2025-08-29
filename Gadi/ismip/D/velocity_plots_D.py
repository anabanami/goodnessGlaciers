import numpy as np
import glob
import matplotlib.pyplot as plt

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
lengths = [5, 10, 20, 40, 80, 160]  # km
author = 'ana'
model = 1
fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey=False)
axs = axs.ravel()

for ax, L in zip(axs, lengths):
    tag   = f"{author}{model}d{L:03d}.txt"
    files = glob.glob(tag)
    if not files:
        ax.set_visible(False);  continue

    xhat, vx_s, vx_b = read_slice(files[0])

    ax.plot(xhat, vx_s, lw=2, label='surface velocity')
    # ax.plot(xhat, vx_b, lw=2, label='basal velocity')

    ax.set_title(f"{L} km", loc='left', fontsize=10)
    ax.set_xlim(0, 1);  ax.set_xlabel("Normalized x")
    ax.set_ylabel("Velocity (m a⁻¹)")
    ax.grid(True, lw=0.3, alpha=0.6)
    ax.legend()

fig.tight_layout()
fig.savefig("ExpD_velocity_panels.png", dpi=300)
plt.show()
