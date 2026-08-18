"""Retired, superseded by vector_independence.py. Kept for the figures the two docs embed.

    ODSA_OUTPUT_BASE=individual_region_TEST/RSL python "testing the vector/beta_intercept_check/beta_intercept_check.py"

Globs <ODSA_OUTPUT_BASE>/window_csvs/, which is a flat tree, so it runs one region at a time.
The multi-region root has no top-level window_csvs and fails on an empty concat. Writes four
diagnostics and a log into <ODSA_OUTPUT_BASE>/bed_character/beta_intercept_check/. The env
value is used as given, so pass an absolute path if you want absolute paths in the log.

Window set and retirement note: papers/sensitivity testing/beta_intercept_check🧪/beta_intercept_check.md
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from loading import OUTPUT_BASE_PATH
from config import Tee, PROCESSING_FLAG_NOTE, processing_flag_of
from plotting import flag_title

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
# path = "v23/Ockenden-regions/window_csvs/ASB_ICECAP_2010_Fig4C_Aurora_SB_w50km_window_stats.csv"
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
    pflag = processing_flag_of(df)
    if pflag:
        print(f"  processing: {PROCESSING_FLAG_NOTE.get(pflag, pflag)}")
    if 'psd_intercept' in df.columns:
        tmp = df.dropna(subset=['psd_intercept'])
        tmp = tmp.assign(C_bin=pd.qcut(tmp['psd_intercept'], q=4, precision=1))
        print("bed_class × psd_intercept bin (%):")
        print(pd.crosstab(tmp.bed_class, tmp.C_bin, normalize='all').mul(100).round(2))


# ── Scatter diagnostics: per-region panels + one combined, colored by bed_class ──

BED_COLORS = {
    'chaotic': '#d62728', 'hard': '#ff7f0e',
    'transitional': '#9467bd', 'soft': '#1f77b4',
}
BED_ORDER = ['chaotic', 'hard', 'transitional', 'soft']


def _panel_flag(df):
    """Migration flag for a panel — only if uniform, else None (e.g. ALL REGIONS)."""
    flags = df['processing_flag'].dropna().unique() if 'processing_flag' in df.columns else []
    return flags[0] if len(flags) == 1 else None


def diag_grid(data, xcol, ycol, xlabel, ylabel, suptitle, fname, vlines=()):
    df_all = data.dropna(subset=[xcol, ycol, 'bed_class'])
    regions = df_all['region'].unique()
    ncols = min(len(regions) + 1, 4)
    nrows = int(np.ceil((len(regions) + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    def panel(ax, df, title):
        for cls in BED_ORDER:
            sub = df[df['bed_class'] == cls]
            if len(sub):
                ax.scatter(sub[xcol], sub[ycol], c=BED_COLORS[cls], label=cls,
                           s=15, alpha=0.6, edgecolors='none')
        valid = df[[xcol, ycol]].dropna()
        r, p = pearsonr(valid[xcol], valid[ycol])
        flag_title(ax, f'{title}\n(r={r:.2f}, p={p:.1e}, n={len(valid)})',
                   _panel_flag(df), fontsize=10)
        for x in vlines:
            ax.axvline(x, color='0.5', ls='--', lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    for i, reg in enumerate(regions):
        panel(axes_flat[i], df_all[df_all['region'] == reg], reg)
    panel(axes_flat[len(regions)], df_all, 'ALL REGIONS')
    axes_flat[len(regions)].legend(fontsize=8)

    for j in range(len(regions) + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(suptitle, fontsize=13)
    plt.tight_layout()
    out = OUT / fname
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Diagnostic saved: {out}")


csvs = sorted(Path(OUTPUT_BASE_PATH, "window_csvs").glob("*_window_stats.csv"))
all_df = pd.concat([pd.read_csv(f).assign(
    region=f.stem.replace("_w50km_window_stats", "")) for f in csvs], ignore_index=True)

print()
diag_grid(all_df, 'relief_m', 'psd_amplitude_1km', 'Relief (m)', 'PSD amplitude @ 1 km',
          'Relief vs PSD amplitude @ 1 km — does psd_intercept add info beyond relief?',
          'relief_vs_psd_amplitude_diagnostic.png')

diag_grid(all_df, 'psd_intercept', 'beta', 'PSD intercept', 'β',
          'β vs PSD intercept — Mechanical coupling diagnostic',
          'beta_vs_psd_intercept_diagnostic.png')

diag_grid(all_df, 'psd_amplitude_1km', 'beta', 'PSD amplitude @ 1 km', 'β',
          'β vs PSD amplitude @ 1 km — 2D roughness classification (in-band amplitude)',
          'beta_vs_psd_amplitude_diagnostic.png')

# vlines are the relief_class breaks, for reading the scatter against the categorical bins
diag_grid(all_df, 'relief_m', 'beta', 'Relief (m)', 'β',
          'β vs relief — is spectral slope independent of total relief?',
          'beta_vs_relief_diagnostic.png', vlines=(350, 800))

