"""Sensitivity of segment spectral parameters to the relief gradient threshold (m/km).

Compares β (slope) and psd_intercept (amplitude) across threshold values
(10, 15, 20 m/km) to test whether the choice of gradient threshold
materially affects spectral results.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

BASE = Path(__file__).parent
THRESHOLDS = [10, 15, 20]
colors = ['#66c2a5', '#fc8d62', '#8da0cb']


REGIONS = sorted(
    p.stem.replace('_window_stats', '')
    for p in (BASE / '15m_per_km/SMUG-regions/window_csvs').glob('*_window_stats.csv')
)


def load_data(thresh):
    """Load window_stats and segment_stats from all SMUG regions."""
    smug = BASE / f"{thresh}m_per_km/SMUG-regions"
    ws_frames, ss_frames = [], []
    for region in REGIONS:
        ws = pd.read_csv(smug / 'window_csvs' / f'{region}_window_stats.csv')
        ss = pd.read_csv(smug / 'segment_csvs' / f'{region}_segment_stats.csv')
        ws['region'] = region
        ss['region'] = region
        ws_frames.append(ws)
        ss_frames.append(ss)
    return pd.concat(ws_frames, ignore_index=True), pd.concat(ss_frames, ignore_index=True)


# --- Gather data ---
summary = []
within_seg_vars = {}
within_seg_vars_intercept = {}
beta_uncs = {}
intercept_uncs = {}

for thresh in THRESHOLDS:
    ws, ss = load_data(thresh)
    grp = ws.groupby(['region', 'trajectory', 'segment'])
    seg_var_beta = grp['beta'].var().dropna()
    seg_var_intercept = grp['psd_intercept'].var().dropna()
    within_seg_vars[thresh] = seg_var_beta
    within_seg_vars_intercept[thresh] = seg_var_intercept
    beta_uncs[thresh] = ss['beta_uncertainty'].dropna()
    intercept_uncs[thresh] = ss['psd_intercept_uncertainty'].dropna()

    summary.append({
        'threshold': thresh,
        'n_segments': len(ss),
        'n_windows': len(ws),
        'n_segs_multwin': len(seg_var_beta),
        'beta_unc_median': ss['beta_uncertainty'].median(),
        'intercept_unc_median': ss['psd_intercept_uncertainty'].median(),
        'within_seg_beta_var_median': seg_var_beta.median(),
        'within_seg_intercept_var_median': seg_var_intercept.median(),
        'beta_std_all_segs': ss['beta'].std(),
        'intercept_std_all_segs': ss['psd_intercept'].std(),
    })

# --- Figure 1: β (top row) and psd_intercept (bottom row) ---
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
tick_labels_unc = [f'{t} m/km\n(n={len(beta_uncs[t])})' for t in THRESHOLDS]
tick_labels_var = [f'{t} m/km\n(n={len(within_seg_vars[t])})' for t in THRESHOLDS]

def _boxplot(ax, data, tick_labels, ylabel, title):
    bp = ax.boxplot(data, tick_labels=tick_labels, patch_artist=True, widths=0.5)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

# Top row: β
_boxplot(axes[0, 0], [beta_uncs[t].values for t in THRESHOLDS],
         tick_labels_unc, 'β uncertainty', 'A) β uncertainty by threshold')
_boxplot(axes[0, 1], [within_seg_vars[t].values for t in THRESHOLDS],
         tick_labels_var, 'within-segment β variance',
         'B) Within-segment β variance by threshold')
ax = axes[0, 2]
for i, thresh in enumerate(THRESHOLDS):
    _, ss = load_data(thresh)
    ax.hist(ss['beta'], bins=20, alpha=0.4, color=colors[i],
            label=f'{thresh} m/km (n={len(ss)})', edgecolor='k', linewidth=0.3)
ax.set_xlabel('β')
ax.set_ylabel('count')
ax.set_title('C) Segment β distributions by threshold')
ax.legend(fontsize=8)

# Bottom row: psd_intercept
_boxplot(axes[1, 0], [intercept_uncs[t].values for t in THRESHOLDS],
         tick_labels_unc, 'intercept uncertainty',
         'D) PSD intercept uncertainty by threshold')
_boxplot(axes[1, 1], [within_seg_vars_intercept[t].values for t in THRESHOLDS],
         tick_labels_var, 'within-segment intercept variance',
         'E) Within-segment intercept variance by threshold')
ax = axes[1, 2]
for i, thresh in enumerate(THRESHOLDS):
    _, ss = load_data(thresh)
    ax.hist(ss['psd_intercept'], bins=20, alpha=0.4, color=colors[i],
            label=f'{thresh} m/km (n={len(ss)})', edgecolor='k', linewidth=0.3)
ax.set_xlabel('PSD intercept')
ax.set_ylabel('count')
ax.set_title('F) Segment intercept distributions by threshold')
ax.legend(fontsize=8)

plt.suptitle('Sensitivity of spectral parameters to relief gradient threshold', fontweight='bold')
plt.tight_layout()
plt.savefig(BASE / 'gradient_threshold_sensitivity.png', dpi=150)
plt.close()

# --- Figure 2: β_uncertainty vs segment length to check confound ---
fig, ax = plt.subplots(figsize=(7, 5))
for i, thresh in enumerate(THRESHOLDS):
    _, ss = load_data(thresh)
    ax.scatter(ss['n_windows'], ss['beta_uncertainty'],
               alpha=0.4, color=colors[i], s=20, label=f'{thresh} m/km')
ax.set_xlabel('windows per segment (proxy for length)')
ax.set_ylabel('β uncertainty')
ax.set_title('β uncertainty vs segment length by threshold')
ax.legend()
plt.tight_layout()
plt.savefig(BASE / 'beta_unc_vs_length.png', dpi=150)
plt.close()

# --- Print results ---
print("\n=== Gradient threshold sensitivity ===\n")
df = pd.DataFrame(summary)
print(df.to_string(index=False))

if len(THRESHOLDS) >= 2:
    for name, seg_vars, uncs in [
        ('β', within_seg_vars, beta_uncs),
        ('PSD intercept', within_seg_vars_intercept, intercept_uncs),
    ]:
        print(f"\n\n--- {name}: within-segment variance (segments with ≥2 windows) ---")
        print(f"    H0: gradient threshold has no effect on within-segment {name} homogeneity")
        h_stat, p_val = stats.kruskal(*[seg_vars[t].values for t in THRESHOLDS])
        print(f"    Kruskal-Wallis: H = {h_stat:.3f}, p = {p_val:.4f}")
        print(f"\n    Pairwise Mann-Whitney U (alternative: lower threshold < higher):")
        for i in range(len(THRESHOLDS)):
            for j in range(i+1, len(THRESHOLDS)):
                u, p = stats.mannwhitneyu(seg_vars[THRESHOLDS[i]],
                                          seg_vars[THRESHOLDS[j]], alternative='less')
                print(f"      {THRESHOLDS[i]} vs {THRESHOLDS[j]} m/km: U={u:.0f}, p={p:.4f}"
                      f" ({'*' if p < 0.05 else 'ns'})")

        print(f"\n--- {name}: fit uncertainty (all segments) ---")
        print("    NOTE: may be confounded by segment length differences across thresholds")
        h_stat, p_val = stats.kruskal(*[uncs[t].values for t in THRESHOLDS])
        print(f"    Kruskal-Wallis: H = {h_stat:.3f}, p = {p_val:.4f}")
else:
    print("\n    (statistical tests skipped — need ≥2 thresholds)")

print("\n\n--- Effect sizes ---")
for name, seg_vars, uncs in [
    ('β', within_seg_vars, beta_uncs),
    ('PSD intercept', within_seg_vars_intercept, intercept_uncs),
]:
    print(f"    Within-segment {name} variance (median):")
    for t in THRESHOLDS:
        print(f"      {t} m/km: {seg_vars[t].median():.5f}")
    print(f"    {name} uncertainty (median, all segments):")
    for t in THRESHOLDS:
        print(f"      {t} m/km: {uncs[t].median():.5f}")
