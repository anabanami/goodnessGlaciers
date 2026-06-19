import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from loading import OUTPUT_BASE_PATH
from config import Tee

  # On interpreting these crosstabs for geological character:

  # The crosstab gives the joint distribution of spectral roughness character (beta-based)
  #  and relief amplitude. Here's how to read the dominant combinations:

# ┌──────────────────┬──────────────────────────────────────────────────────────────────┐
# │    Dominant      │                          Interpretation                          │
# │   combination    │                                                                  │
# ├──────────────────┼──────────────────────────────────────────────────────────────────┤
# │ hard +           │ Crystalline basement / shield terrain — rough at short           │
# │ subdued/flat     │ wavelengths but low total relief. Think cratonic bedrock.        │
# ├──────────────────┼──────────────────────────────────────────────────────────────────┤
# │ hard +           │ Active/recent orogen or volcanic terrain — rough texture and big │
# │ mountainous      │  topography.                                                     │
# ├──────────────────┼──────────────────────────────────────────────────────────────────┤
# │ soft +           │ Sediment-draped mountains — high relief but smoothed at short    │
# │ mountainous      │ wavelengths (marine sediments over buried topography, or glacial │
# │                  │  erosion smoothing peaks).                                       │
# ├──────────────────┼──────────────────────────────────────────────────────────────────┤
# │ soft +           │ Sedimentary basin — smooth at all scales. Classic soft-sediment  │
# │ flat/subdued     │ signature.                                                       │
# ├──────────────────┼──────────────────────────────────────────────────────────────────┤
# │ transitional +   │ Mixed — could be partially eroded mountains, or heterogeneous    │
# │ mountainous      │ geology.                                                         │
# ├──────────────────┼──────────────────────────────────────────────────────────────────┤
# │ chaotic +        │ Noisy/disordered — possibly heavily fractured, or data quality   │
# │ anything         │ issues.                                                          │
# └──────────────────┴──────────────────────────────────────────────────────────────────┘

### --------------------------------------------------------------------------------------
# # test
# path = "v23/Ockenden-regions/window_csvs/ASB_ICECAP_2010_Fig4_Aurora_SB_w50km_window_stats.csv"
# df = pd.read_csv(path)
# ct = pd.crosstab(df.bed_class, df.relief_class, normalize='all') * 100
# print(ct.round(1))


# Loop over regions:
# NOTE THAT: if a bed_class category (e.g. "soft") has zero windows, its row is omitted entirely.

# for f in sorted(Path(OUTPUT_BASE_PATH, "window_csvs").glob("*_window_stats.csv")):
#     df = pd.read_csv(f)
#     name = f.stem.replace("_w50km_window_stats", "")
#     print(f"\n=== {name} ===")
#     print("bed_class × relief_class (%):")
#     print(pd.crosstab(df.bed_class, df.relief_class, normalize='all').mul(100).round(2))
#
#     if 'psd_amplitude_1km' in df.columns:
#         tmp = df.dropna(subset=['psd_amplitude_1km'])
#         tmp = tmp.assign(amp_bin=pd.qcut(tmp['psd_amplitude_1km'], q=4, precision=1))
#         print("\nbed_class × psd_amplitude_1km bin (%):")
#         print(pd.crosstab(tmp.bed_class, tmp.amp_bin, normalize='all').mul(100).round(2))

### --------------------------------------------------------------------------------------

OUT = Path(OUTPUT_BASE_PATH, "bed_character", "beta_intercept_check")
OUT.mkdir(parents=True, exist_ok=True)
sys.stdout = Tee(OUT / "beta_intercept_check_log.txt")

for f in sorted(Path(OUTPUT_BASE_PATH, "window_csvs").glob("*_window_stats.csv")):
    df = pd.read_csv(f)
    name = f.stem.replace("_w50km_window_stats", "")
    print(f"\n=== {name} ===")
    if 'psd_intercept' in df.columns:
        tmp = df.dropna(subset=['psd_intercept'])
        tmp = tmp.assign(C_bin=pd.qcut(tmp['psd_intercept'], q=4, precision=1))
        print("bed_class × psd_intercept bin (%):")
        print(pd.crosstab(tmp.bed_class, tmp.C_bin, normalize='all').mul(100).round(2))


# ── Diagnostic: relief_m vs psd_amplitude_1km, colored by bed_class ──

BED_COLORS = {
    'chaotic': '#d62728', 'hard': '#ff7f0e',
    'transitional': '#9467bd', 'soft': '#1f77b4',
}

csvs = sorted(Path(OUTPUT_BASE_PATH, "window_csvs").glob("*_window_stats.csv"))
all_df = pd.concat([pd.read_csv(f).assign(
    region=f.stem.replace("_w50km_window_stats", "")) for f in csvs], ignore_index=True)
all_df = all_df.dropna(subset=['relief_m', 'psd_amplitude_1km', 'bed_class'])

# Per-region panels + one combined
regions = all_df['region'].unique()
ncols = min(len(regions) + 1, 4)
nrows = int(np.ceil((len(regions) + 1) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
axes_flat = axes.flatten()

def scatter_panel(ax, df, title):
    for cls in ['chaotic', 'hard', 'transitional', 'soft']:
        sub = df[df['bed_class'] == cls]
        if len(sub):
            ax.scatter(sub['relief_m'], sub['psd_amplitude_1km'],
                       c=BED_COLORS[cls], label=cls, s=15, alpha=0.6, edgecolors='none')
    from scipy.stats import pearsonr
    valid = df[['relief_m', 'psd_amplitude_1km']].dropna()
    r, p = pearsonr(valid['relief_m'], valid['psd_amplitude_1km'])
    n = len(valid)
    ax.set_title(f'{title}  (r={r:.2f}, p={p:.1e}, n={n})', fontsize=10)
    ax.set_xlabel('Relief (m)')
    ax.set_ylabel('PSD amplitude @ 1 km')
    ax.grid(True, alpha=0.3)

for i, reg in enumerate(regions):
    scatter_panel(axes_flat[i], all_df[all_df['region'] == reg], reg)
scatter_panel(axes_flat[len(regions)], all_df, 'ALL REGIONS')
axes_flat[len(regions)].legend(fontsize=8)

for j in range(len(regions) + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle('Relief vs PSD amplitude @ 1 km — does psd_intercept add info beyond relief?', fontsize=13)
plt.tight_layout()
BIC_OUT = Path(OUTPUT_BASE_PATH, "bed_character", "beta_intercept_check")
BIC_OUT.mkdir(parents=True, exist_ok=True)
out = BIC_OUT / "relief_vs_psd_amplitude_diagnostic.png"
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f"\nDiagnostic saved: {out}")


# ── Diagnostic: β vs psd_intercept (purely spectral 2D classification) ──

all_df2 = pd.concat([pd.read_csv(f).assign(
    region=f.stem.replace("_w50km_window_stats", "")) for f in csvs], ignore_index=True)
all_df2 = all_df2.dropna(subset=['beta', 'psd_intercept', 'bed_class'])

regions2 = all_df2['region'].unique()
ncols2 = min(len(regions2) + 1, 4)
nrows2 = int(np.ceil((len(regions2) + 1) / ncols2))
fig2, axes2 = plt.subplots(nrows2, ncols2, figsize=(5 * ncols2, 4.5 * nrows2), squeeze=False)
axes2_flat = axes2.flatten()

def scatter_panel_spectral(ax, df, title):
    from scipy.stats import pearsonr
    for cls in ['chaotic', 'hard', 'transitional', 'soft']:
        sub = df[df['bed_class'] == cls]
        if len(sub):
            ax.scatter(sub['psd_intercept'], sub['beta'],
                       c=BED_COLORS[cls], label=cls, s=15, alpha=0.6, edgecolors='none')
    valid = df[['psd_intercept', 'beta']].dropna()
    r, p = pearsonr(valid['psd_intercept'], valid['beta'])
    n = len(valid)
    ax.set_title(f'{title}  (r={r:.2f}, p={p:.1e}, n={n})', fontsize=10)
    ax.set_xlabel('PSD intercept (C)')
    ax.set_ylabel('β')
    ax.grid(True, alpha=0.3)

for i, reg in enumerate(regions2):
    scatter_panel_spectral(axes2_flat[i], all_df2[all_df2['region'] == reg], reg)
scatter_panel_spectral(axes2_flat[len(regions2)], all_df2, 'ALL REGIONS')
axes2_flat[len(regions2)].legend(fontsize=8)

for j in range(len(regions2) + 1, len(axes2_flat)):
    axes2_flat[j].set_visible(False)

fig2.suptitle('β vs PSD intercept (C) — purely spectral 2D roughness classification', fontsize=13)
plt.tight_layout()
out2 = BIC_OUT / "beta_vs_psd_intercept_diagnostic.png"
plt.savefig(out2, dpi=200, bbox_inches='tight')
plt.close()
print(f"Diagnostic saved: {out2}")


