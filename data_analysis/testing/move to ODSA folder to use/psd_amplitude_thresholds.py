"""
Derive psd_amplitude_1km threshold from window-level spectral data
using Jenks natural breaks (2-class: low vs high amplitude).

Sibling of psd_intercept_thresholds.py — same structure so the two are
directly comparable. psd_amplitude_1km = psd_intercept + 3*beta, i.e. the
fit evaluated in-band (lambda=1km) rather than extrapolated to lambda=1m.

Outputs:
  - Pooled histogram with Jenks 2-class break
  - Per-region histograms with pooled break overlaid (generalisability check)
  - Class-conditional distributions (amp within each bed_class)
  - Survey-group breakdown: does the bimodal gap track survey or geology?
  - Summary table to console
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jenkspy
from pathlib import Path
from loading import OUTPUT_BASE_PATH
from config import Tee

VAR = 'psd_amplitude_1km'
LABEL = 'PSD amplitude @ 1 km'

BED_COLORS = {
    'chaotic': '#d62728', 'hard': '#ff7f0e',
    'transitional': '#9467bd', 'soft': '#1f77b4',
}

OUT = Path(OUTPUT_BASE_PATH, "bed_character", "psd_amplitude_thresholds")
OUT.mkdir(parents=True, exist_ok=True)
sys.stdout = Tee(OUT / "psd_amplitude_thresholds_log.txt")

# ── Load all windows ──
csvs = sorted(Path(OUTPUT_BASE_PATH, "window_csvs").glob("*_window_stats.csv"))
all_df = pd.concat([
    pd.read_csv(f).assign(region=f.stem.replace("_w50km_window_stats", ""))
    for f in csvs
], ignore_index=True)
all_df = all_df.dropna(subset=[VAR])
all_df['survey'] = all_df['region'].str.split('_Fig').str[0]
A = all_df[VAR].values

# ── Compute Jenks 2-class break ──
jenks_break = jenkspy.jenks_breaks(A, n_classes=2)[1]

print(f"Pooled: n={len(A)}, median={np.median(A):.2f}, "
      f"range=[{A.min():.2f}, {A.max():.2f}]")
print(f"Jenks 2-class break: {jenks_break:.2f}")

# ── Per-region breaks ──
regions = sorted(all_df['region'].unique())
print(f"\n{'Region':<45} {'n':>4}  {'median':>6}  {'jenks-2':>8}")
print("-" * 70)
for reg in regions:
    rc = all_df.loc[all_df.region == reg, VAR].values
    j = jenkspy.jenks_breaks(rc, n_classes=2)[1] if len(rc) >= 4 else np.nan
    print(f"{reg:<45} {len(rc):>4}  {np.median(rc):>6.2f}  {j:>8.2f}")

# ── Per-survey breaks (geology vs acquisition check) ──
surveys = sorted(all_df['survey'].unique())
print(f"\n{'Survey':<25} {'n':>4}  {'median':>6}  {'jenks-2':>8}")
print("-" * 50)
for sv in surveys:
    sc = all_df.loc[all_df.survey == sv, VAR].values
    j = jenkspy.jenks_breaks(sc, n_classes=2)[1] if len(sc) >= 4 else np.nan
    print(f"{sv:<25} {len(sc):>4}  {np.median(sc):>6.2f}  {j:>8.2f}")

# ── Figure 1: Pooled histogram with Jenks break ──
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(A, bins=40, color='0.7', edgecolor='k', linewidth=0.3, label='all windows')
ax.axvline(jenks_break, color='tab:red', ls='-', lw=2, label=f'Jenks 2-class ({jenks_break:.2f})')
ax.set_xlabel(LABEL)
ax.set_ylabel('count')
ax.set_title(f'Pooled {VAR} distribution (n={len(A)})')
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "psd_amplitude_pooled_break.png", dpi=150)
plt.close(fig)

# ── Figure 2: Per-region histograms with pooled break ──
ncols = min(len(regions) + 1, 4)
nrows = int(np.ceil((len(regions) + 1) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
axes = axes.flatten()

for i, reg in enumerate(regions):
    ax = axes[i]
    rc = all_df.loc[all_df.region == reg, VAR].values
    ax.hist(rc, bins=20, color='0.7', edgecolor='k', linewidth=0.3)
    ax.axvline(jenks_break, color='tab:red', ls='-', lw=1.5)
    ax.set_title(reg.replace("ASB_ICECAP_2010_", "").replace("POLARGAP_2015_", ""),
                 fontsize=9)
    ax.set_xlabel('amp', fontsize=8)

ax = axes[len(regions)]
ax.hist(A, bins=40, color='0.7', edgecolor='k', linewidth=0.3)
ax.axvline(jenks_break, color='tab:red', ls='-', lw=2, label=f'Jenks 2-class ({jenks_break:.2f})')
ax.set_title(f'ALL REGIONS (n={len(A)})', fontsize=9, fontweight='bold')
ax.legend(fontsize=7)

for j in range(len(regions) + 1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle('Per-region psd_amplitude_1km with pooled Jenks break overlaid', fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / "psd_amplitude_per_region_break.png", dpi=150)
plt.close(fig)

# ── Figure 3: Class-conditional amplitude distributions ──
# NB: bed_class = classify_beta(beta), so any separation here is partly built in.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
classes = ['chaotic', 'hard', 'transitional', 'soft']

ax = axes[0]
for cls in classes:
    vals = all_df.loc[all_df.bed_class == cls, VAR].values
    if len(vals):
        ax.hist(vals, bins=30, alpha=0.5, color=BED_COLORS[cls],
                label=f'{cls} (n={len(vals)})', edgecolor='k', linewidth=0.2)
ax.axvline(jenks_break, color='tab:red', ls='-', lw=1.5, label=f'Jenks ({jenks_break:.2f})')
ax.set_xlabel(LABEL)
ax.set_ylabel('count')
ax.set_title('amplitude distribution by bed_class')
ax.legend(fontsize=8)

ax = axes[1]
data = [all_df.loc[all_df.bed_class == c, VAR].dropna().values for c in classes]
bp = ax.boxplot(data, labels=classes, patch_artist=True, widths=0.6)
for patch, cls in zip(bp['boxes'], classes):
    patch.set_facecolor(BED_COLORS[cls])
    patch.set_alpha(0.6)
ax.axhline(jenks_break, color='tab:red', ls='-', lw=1.5, label=f'Jenks ({jenks_break:.2f})')
ax.set_ylabel(LABEL)
ax.set_title('amplitude by bed_class (box plots)')
ax.legend(fontsize=8)

fig.suptitle('Does psd_amplitude_1km separate bed classes? (bed_class is beta-derived)',
             fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / "psd_amplitude_class_conditional.png", dpi=150)
plt.close(fig)

# ── Figure 4: Survey-group breakdown (geology vs acquisition) ──
fig, ax = plt.subplots(figsize=(9, 5))
for sv in surveys:
    sc = all_df.loc[all_df.survey == sv, VAR].values
    ax.hist(sc, bins=30, alpha=0.5, label=f'{sv} (n={len(sc)}, med={np.median(sc):.2f})',
            edgecolor='k', linewidth=0.2)
ax.axvline(jenks_break, color='tab:red', ls='-', lw=2, label=f'pooled Jenks ({jenks_break:.2f})')
ax.set_xlabel(LABEL)
ax.set_ylabel('count')
ax.set_title('psd_amplitude_1km by survey — does the pooled break sit in a survey gap?')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "psd_amplitude_by_survey.png", dpi=150)
plt.close(fig)

print(f"\nPlots saved to {OUT.relative_to(Path.cwd())}")
