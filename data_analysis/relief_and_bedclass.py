import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# # test
# path = "v23/Ockenden-regions/window_csvs/ASB_ICECAP_2010_Fig4_Aurora_SB_w50km_window_stats.csv"
# df = pd.read_csv(path)
# ct = pd.crosstab(df.bed_class, df.relief_class, normalize='all') * 100
# print(ct.round(1))


# Loop over regions:

# NOTE THAT: if a bed_class category (e.g. "soft") has zero windows, its row is omitted entirely.
from pathlib import Path
from loading import OUTPUT_BASE_PATH

for f in sorted(Path(OUTPUT_BASE_PATH, "window_csvs").glob("*_window_stats.csv")):
    df = pd.read_csv(f)
    name = f.stem.replace("_w50km_window_stats", "")
    print(f"\n=== {name} ===")
    print("bed_class × relief_class (%):")
    print(pd.crosstab(df.bed_class, df.relief_class, normalize='all').mul(100).round(2))

    if 'psd_amplitude_1km' in df.columns:
        tmp = df.dropna(subset=['psd_amplitude_1km'])
        tmp = tmp.assign(amp_bin=pd.cut(tmp['psd_amplitude_1km'],
                                        bins=[-np.inf, 2, 3, 4, 5, np.inf],
                                        labels=['<2', '2–3', '3–4', '4–5', '>5']))
        print("\nbed_class × psd_amplitude_1km bin (%):")
        print(pd.crosstab(tmp.bed_class, tmp.amp_bin, normalize='all').mul(100).round(2))


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
    r = df[['relief_m', 'psd_amplitude_1km']].corr().iloc[0, 1]
    ax.set_title(f'{title}  (r={r:.2f})', fontsize=10)
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
out = Path(OUTPUT_BASE_PATH, "bed_character", "relief_vs_psd_amplitude_diagnostic.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f"\nDiagnostic saved: {out}")

  # On interpreting these crosstabs for geological character:

  # The crosstab gives you the joint distribution of spectral roughness character (beta-based)
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

#   Looking at your regions:

#   - Recovery SB (both datasets): dominated by transitional + mountainous (~30-34%) with some
#    soft — consistent with a subglacial basin margin with mixed geology.
#   - Resolution SH, Highland A, Golicyna SH, Aurora SB: all dominated by hard + subdued
#   (35-44%) — classic East Antarctic cratonic shield signature.
#   - Pensacola/Pole: transitional + subdued (26%) and hard + subdued (25%) — mixed character,
#    makes sense for the transition zone.
#   - Hercules Dome: transitional + subdued/mountainous dominant — transitional character
#   consistent with subglacial highlands.

#   A quick way to collapse this into a single "dominant terrain type" label per region: just
#   take the cell with the highest percentage. You could also compute a weighted summary
#   metric if you want something more continuous — e.g., a "hardness index" (fraction
#   hard+chaotic vs soft) and a "relief index" (fraction mountainous vs flat) as two
#   orthogonal axes, then plot regions in that 2D space.
