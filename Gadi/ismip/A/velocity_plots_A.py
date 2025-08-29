import numpy as np
import glob
import matplotlib.pyplot as plt

def read_slice(fname, yhat_target=0.25, yband=0.04):
    d = np.loadtxt(fname, comments='#')
    xhat, yhat, vx, vy = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    speed = np.hypot(vx, vy)

    xs, vs = [], []
    for x in np.unique(xhat):
        rows = np.where((xhat == x) & (np.abs(yhat - yhat_target) < yband))[0]
        if len(rows) == 0:
            continue
        xs.append(x)
        vs.append(np.median(speed[rows]))

    order = np.argsort(xs)
    return np.array(xs)[order], np.array(vs)[order]


lengths = [5, 10, 20, 40, 80, 160] # km
author  = 'ana'; model = 1
fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharey=False)
axs = axs.ravel()

for ax, L in zip(axs, lengths):
    tag   = f"{author}{model}a{L:03d}.txt"
    files = glob.glob(tag)
    if not files:
        ax.set_visible(False);  continue

    # with one model:
    xhat, mu = read_slice(files[0])

    ax.plot(xhat, mu, lw=2, label='surface velocity')
    ax.set_title(f"{L} km", loc='left', fontsize=10)
    ax.set_xlim(0, 1);  ax.set_xlabel("Normalized x")
    ax.set_ylabel("Velocity (m a⁻¹)")
    ax.grid(True, lw=0.3, alpha=0.6)
    ax.legend()

fig.tight_layout()
fig.savefig("ExpA_velocity_panels.png", dpi=300)
plt.show()
