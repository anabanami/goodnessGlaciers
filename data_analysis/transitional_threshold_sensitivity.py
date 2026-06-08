"""
Threshold sensitivity analysis for the β = 2.0 hard/transitional boundary.

Step 1: Load all window-level beta values from both region directories.
"""
import pandas as pd
from pathlib import Path

# Gather all window-level CSVs from both region directories
base = Path("/home/ana/Desktop/code/Data/ODSA/v23")
csv_dirs = [
    base / "Ockenden-regions" / "window_csvs",
    base / "SMUG-regions" / "window_csvs",
]

frames = []
for d in csv_dirs:
    for f in sorted(d.glob("*_window_stats.csv")):
        df = pd.read_csv(f, usecols=["beta"])
        # tag with the region name (filename minus the suffix)
        df["region"] = f.stem.replace("_window_stats", "")
        frames.append(df)

all_beta = pd.concat(frames, ignore_index=True)

print(f"Total windows loaded: {len(all_beta)}")
print(f"Beta range: [{all_beta['beta'].min():.3f}, {all_beta['beta'].max():.3f}]")
print(f"\nPer-region counts:")
print(all_beta.groupby("region").size().to_string())

# ── Step 2: KDE of the pooled beta distribution ──────────────────────
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

beta = all_beta["beta"].values
x_grid = np.linspace(0.5, 3.0, 500)       # evaluation points
kde = gaussian_kde(beta, bw_method=0.08)   # bandwidth 0.08 — tight enough
                                            # to see structure, not so tight
                                            # it's just noise
density = kde(x_grid)

fig, ax = plt.subplots(figsize=(9, 4))
ax.fill_between(x_grid, density, alpha=0.3, color="steelblue")
ax.plot(x_grid, density, color="steelblue", lw=2)
ax.axvline(2.0, color="red", ls="--", lw=1.5, label="current threshold (2.0)")
ax.axvline(1.5, color="grey", ls=":", lw=1, alpha=0.6, label="chaotic/hard (1.5)")
ax.axvline(2.5, color="grey", ls=":", lw=1, alpha=0.6, label="transitional/soft (2.5)")
ax.set_xlabel("β (spectral slope)")
ax.set_ylabel("density")
ax.set_title("KDE of pooled window-level β  (n = {})".format(len(beta)))
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("threshold_kde.png", dpi=150)
print("\n✓ Saved threshold_kde.png")

# ── Step 3: Threshold sweep ──────────────────────────────────────────
# Slide the hard/transitional boundary from 1.8 to 2.6.
# The chaotic/hard (1.5) and transitional/soft (2.5) boundaries stay fixed,
# EXCEPT: the transitional/soft boundary must always be above the swept
# threshold, so when threshold > 2.5 the "transitional" bin vanishes.

thresholds = np.arange(1.8, 2.61, 0.01)
n = len(beta)

pct_chaotic      = np.array([(beta < 1.5).sum() / n * 100] * len(thresholds))
pct_hard         = np.array([((beta >= 1.5) & (beta < t)).sum() / n * 100 for t in thresholds])
pct_transitional = np.array([((beta >= t) & (beta < 2.5)).sum() / n * 100 for t in thresholds])
pct_soft         = np.array([(beta >= 2.5).sum() / n * 100] * len(thresholds))

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(thresholds, pct_hard, label="hard", color="sienna", lw=2)
ax.plot(thresholds, pct_transitional, label="transitional", color="goldenrod", lw=2)
ax.plot(thresholds, pct_chaotic, label="chaotic (fixed)", color="grey", ls=":", lw=1.5)
ax.plot(thresholds, pct_soft, label="soft (fixed)", color="grey", ls="--", lw=1.5)
ax.axvline(2.0, color="red", ls="--", lw=1.5, alpha=0.7, label="current (2.0)")
ax.set_xlabel("hard / transitional threshold")
ax.set_ylabel("% of windows")
ax.set_title("Classification sensitivity to threshold position")
ax.legend(fontsize=8, loc="center right")
fig.tight_layout()
fig.savefig("threshold_sweep.png", dpi=150)
print("✓ Saved threshold_sweep.png")

# Print a few key values around 2.1
print("\n threshold   % hard     % trans     Δhard/0.1")
print("  ──────────┼──────────┼───────────┼───────────")
for t in [1.9, 2.0, 2.1, 2.2, 2.3, 2.4]:
    h = ((beta >= 1.5) & (beta < t)).sum() / n * 100
    tr = ((beta >= t) & (beta < 2.5)).sum() / n * 100
    # rate of change: how much does %hard change per 0.1 shift?
    t2 = t + 0.1
    h2 = ((beta >= 1.5) & (beta < t2)).sum() / n * 100
    dh = h2 - h
    print(f"    {t:.1f}     │  {h:5.1f}   │  {tr:5.1f}    │  {dh:+.1f}%")

# ── Step 4: Find local minima in the KDE between 1.8 and 2.5 ─────────
from scipy.signal import argrelmin

mask = (x_grid >= 1.8) & (x_grid <= 2.5)
x_sub = x_grid[mask]
d_sub = density[mask]

# order=15: point must be lower than 15 neighbours on each side
minima_idx = argrelmin(d_sub, order=15)[0]
maxima_idx = argrelmin(-d_sub, order=15)[0]  # local maxima

print("\n── KDE structure between β=1.8 and β=2.5 ──")
print("Local maxima (peaks):")
for i in maxima_idx:
    print(f"  β = {x_sub[i]:.3f}   density = {d_sub[i]:.4f}")
print("Local minima (saddles):")
for i in minima_idx:
    print(f"  β = {x_sub[i]:.3f}   density = {d_sub[i]:.4f}")

# ── Step 5: Per-region KDEs ─────────────────────────────────────────
# Check whether the ~2.2 peak is consistent across regions or driven
# by a few dominant ones.

regions = all_beta.groupby("region")
n_regions = regions.ngroups
cols = 3
rows = int(np.ceil(n_regions / cols))

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows),
                         sharex=True, sharey=False)
axes = axes.flatten()

print("\n── Per-region KDE peaks (β = 1.8–2.5) ──")
print(f"  {'region':<55s}   n    peak β   density")
print("  " + "─" * 85)

for ax, (name, grp) in zip(axes, regions):
    b = grp["beta"].values
    if len(b) < 10:
        ax.set_title(f"{name}\n(n={len(b)}, too few)", fontsize=7)
        ax.set_visible(True)
        continue
    k = gaussian_kde(b, bw_method=0.08)
    d = k(x_grid)
    ax.fill_between(x_grid, d, alpha=0.3, color="steelblue")
    ax.plot(x_grid, d, color="steelblue", lw=1.5)
    ax.axvline(2.0, color="red", ls="--", lw=1, alpha=0.7)
    ax.axvline(1.5, color="grey", ls=":", lw=0.8, alpha=0.4)
    ax.axvline(2.5, color="grey", ls=":", lw=0.8, alpha=0.4)
    ax.set_title(f"{name}\n(n={len(b)})", fontsize=7)
    ax.tick_params(labelsize=6)

    # find peaks in the 1.8–2.5 range for this region
    d_sub_r = d[mask]
    max_idx_r = argrelmin(-d_sub_r, order=15)[0]
    if len(max_idx_r) > 0:
        # report the highest peak in the range
        best = max_idx_r[np.argmax(d_sub_r[max_idx_r])]
        print(f"  {name:<55s} {len(b):>4d}    {x_sub[best]:.3f}   {d_sub_r[best]:.4f}")
    else:
        # no local peak found — report the max of the range
        imax = np.argmax(d_sub_r)
        print(f"  {name:<55s} {len(b):>4d}    {x_sub[imax]:.3f}*  {d_sub_r[imax]:.4f}  (*boundary max)")

# hide unused subplots
for i in range(n_regions, len(axes)):
    axes[i].set_visible(False)

fig.suptitle("Per-region KDE of β  (red dashed = 2.0 threshold)", fontsize=10, y=1.01)
fig.tight_layout()
fig.savefig("threshold_kde_per_region.png", dpi=150, bbox_inches="tight")
print("\n✓ Saved threshold_kde_per_region.png")
