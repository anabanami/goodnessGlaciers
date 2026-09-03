"""Sensitivity of spectral parameters to GRADIENT_THRESHOLD (m/km).

Post-fix pipeline: threshold now gates is_transition (which windows contribute
to results), not just where segments split. Key questions:
  1. What fraction of windows are flagged as transition at each threshold?
  2. Is β of the surviving (non-transition) population stable across thresholds?
  3. Are within-segment variance and fit uncertainty stable?

Also tests TZ-merge distance (2, 5, 10 km) if data directories exist.

Run from v23/; reads from and writes results (log + figures) to
v23/TESTING_LANDSCAPE_SPLITTING/gradient_threshold_sensitivity/.
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

HERE = Path(__file__).resolve().parent                           # .../v23
OUT_ROOT = HERE / 'TESTING_LANDSCAPE_SPLITTING'
BASE = OUT_ROOT / 'gradient_threshold_sensitivity'  # this script's own data + output tree
THRESHOLDS = [10, 15, 20]
MERGE_DISTANCES = [2, 5, 10]  # km
colors = ['#66c2a5', '#fc8d62', '#8da0cb']

DATA_DIR = 'Ockenden-regions-sensitivityTEST'

# Tee stdout to log file
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, msg):
        for s in self.streams:
            s.write(msg)
    def flush(self):
        for s in self.streams:
            s.flush()

BASE.mkdir(parents=True, exist_ok=True)
_log = open(BASE / 'gradient_threshold_sensitivity.log', 'w')
sys.stdout = _Tee(sys.__stdout__, _log)

# Use 15 m/km as reference to discover region filenames (all three dirs have same structure)
REGIONS = sorted(
    p.stem.replace('_window_stats', '')
    for p in (BASE / f'15m_per_km/{DATA_DIR}/window_csvs').glob('*_window_stats.csv')
)


def load_data(thresh):
    """Load window_stats and segment_stats for a gradient threshold run."""
    root = BASE / f"{thresh}m_per_km/{DATA_DIR}"
    ws_frames, ss_frames = [], []
    for region in REGIONS:
        wf = root / 'window_csvs' / f'{region}_window_stats.csv'
        sf = root / 'segment_csvs' / f'{region}_segment_stats.csv'
        if not wf.exists() or not sf.exists():
            continue
        ws = pd.read_csv(wf)
        ss = pd.read_csv(sf)
        ws['region'] = region
        ss['region'] = region
        ws_frames.append(ws)
        ss_frames.append(ss)
    return pd.concat(ws_frames, ignore_index=True), pd.concat(ss_frames, ignore_index=True)


def load_merge_data(merge_km):
    """Load window_stats and segment_stats for a TZ-merge distance run."""
    root = BASE / f"merge_{merge_km}km/{DATA_DIR}"
    if not root.exists():
        return None, None
    ws_frames, ss_frames = [], []
    for region in REGIONS:
        wf = root / 'window_csvs' / f'{region}_window_stats.csv'
        sf = root / 'segment_csvs' / f'{region}_segment_stats.csv'
        if not wf.exists() or not sf.exists():
            continue
        ws = pd.read_csv(wf)
        ss = pd.read_csv(sf)
        ws['region'] = region
        ss['region'] = region
        ws_frames.append(ws)
        ss_frames.append(ss)
    if not ws_frames:
        return None, None
    return pd.concat(ws_frames, ignore_index=True), pd.concat(ss_frames, ignore_index=True)


# =============================================================================
# GRADIENT THRESHOLD SENSITIVITY
# =============================================================================
print("\n" + "="*70)
print("GRADIENT THRESHOLD SENSITIVITY (post-fix pipeline)")
print("="*70)

summary = []
within_seg_vars = {}
within_seg_vars_intercept = {}
beta_uncs = {}
intercept_uncs = {}
transition_fracs = {}
homog_betas = {}

for thresh in THRESHOLDS:
    ws, ss = load_data(thresh)

    # Transition fraction — the new headline sensitivity
    n_total = len(ws)
    n_trans = ws['is_transition'].sum()
    frac = n_trans / n_total if n_total > 0 else 0
    transition_fracs[thresh] = frac

    # Filter to homogeneous windows only (what the pipeline reports)
    ws_h = ws[~ws['is_transition']].copy()
    homog_betas[thresh] = ws_h['beta'].dropna()

    # Within-segment variance on homogeneous windows only
    grp = ws_h.groupby(['region', 'trajectory', 'segment'])
    seg_var_beta = grp['beta'].var().dropna()
    seg_var_intercept = grp['psd_intercept'].var().dropna()
    within_seg_vars[thresh] = seg_var_beta
    within_seg_vars_intercept[thresh] = seg_var_intercept
    beta_uncs[thresh] = ss['beta_uncertainty'].dropna()
    intercept_uncs[thresh] = ss['psd_intercept_uncertainty'].dropna()

    summary.append({
        'threshold': thresh,
        'n_windows_total': n_total,
        'n_windows_homog': len(ws_h),
        'n_transition': n_trans,
        'pct_transition': f"{frac*100:.1f}%",
        'n_segments': len(ss),
        'beta_median_homog': ws_h['beta'].median(),
        'beta_std_homog': ws_h['beta'].std(),
        'beta_unc_median': ss['beta_uncertainty'].median(),
        'intercept_unc_median': ss['psd_intercept_uncertainty'].median(),
        'within_seg_beta_var_median': seg_var_beta.median() if len(seg_var_beta) else np.nan,
    })

# --- Print summary ---
print("\n--- Summary table ---\n")
df = pd.DataFrame(summary)
print(df.to_string(index=False))

# --- Transition fraction by region ---
print("\n\n--- Transition fraction by region ---\n")
print(f"{'Region':<55} {'10 m/km':>8} {'15 m/km':>8} {'20 m/km':>8}")
for region in REGIONS:
    row = f"{region:<55}"
    for thresh in THRESHOLDS:
        ws, _ = load_data(thresh)
        wr = ws[ws['region'] == region]
        frac = wr['is_transition'].sum() / len(wr) if len(wr) else 0
        row += f" {frac*100:>6.1f}%"
    print(row)

# --- Statistical tests on homogeneous β ---
print("\n\n--- Homogeneous-window β: stability across thresholds ---")
print("    H0: threshold choice does not affect β of retained windows")
if all(len(homog_betas[t]) > 0 for t in THRESHOLDS):
    h_stat, p_val = stats.kruskal(*[homog_betas[t].values for t in THRESHOLDS])
    print(f"    Kruskal-Wallis: H = {h_stat:.3f}, p = {p_val:.4f}")
    for i in range(len(THRESHOLDS)):
        for j in range(i+1, len(THRESHOLDS)):
            u, p = stats.mannwhitneyu(homog_betas[THRESHOLDS[i]],
                                      homog_betas[THRESHOLDS[j]])
            print(f"      {THRESHOLDS[i]} vs {THRESHOLDS[j]} m/km: U={u:.0f}, p={p:.4f}"
                  f" ({'*' if p < 0.05 else 'ns'})")
    print(f"\n    Medians: " + ", ".join(
        f"{t} m/km = {homog_betas[t].median():.3f}" for t in THRESHOLDS))

# --- Within-segment variance tests ---
if len(THRESHOLDS) >= 2:
    for name, seg_vars, uncs in [
        ('β', within_seg_vars, beta_uncs),
        ('PSD intercept', within_seg_vars_intercept, intercept_uncs),
    ]:
        print(f"\n\n--- {name}: within-segment variance (homogeneous windows, ≥2 per segment) ---")
        valid = [t for t in THRESHOLDS if len(seg_vars[t]) > 0]
        if len(valid) >= 2:
            h_stat, p_val = stats.kruskal(*[seg_vars[t].values for t in valid])
            print(f"    Kruskal-Wallis: H = {h_stat:.3f}, p = {p_val:.4f}")
            print(f"    Medians: " + ", ".join(
                f"{t} m/km = {seg_vars[t].median():.5f}" for t in valid))

        print(f"\n--- {name}: fit uncertainty (all segments) ---")
        valid_u = [t for t in THRESHOLDS if len(uncs[t]) > 0]
        if len(valid_u) >= 2:
            h_stat, p_val = stats.kruskal(*[uncs[t].values for t in valid_u])
            print(f"    Kruskal-Wallis: H = {h_stat:.3f}, p = {p_val:.4f}")


# =============================================================================
# FIGURE 1: Main sensitivity figure (3 rows)
# =============================================================================
fig, axes = plt.subplots(3, 3, figsize=(14, 11))

# Row 1: Transition gating
ax = axes[0, 0]
bars = ax.bar(range(len(THRESHOLDS)),
              [transition_fracs[t]*100 for t in THRESHOLDS],
              color=colors, edgecolor='k', linewidth=0.5)
ax.set_xticks(range(len(THRESHOLDS)))
ax.set_xticklabels([f'{t} m/km' for t in THRESHOLDS])
ax.set_ylabel('% windows flagged transition')
ax.set_title('A) Transition fraction vs threshold')

ax = axes[0, 1]
for i, thresh in enumerate(THRESHOLDS):
    ax.hist(homog_betas[thresh], bins=25, alpha=0.4, color=colors[i],
            label=f'{thresh} m/km (n={len(homog_betas[thresh])})',
            edgecolor='k', linewidth=0.3)
ax.set_xlabel('β')
ax.set_ylabel('count')
ax.set_title('B) Homogeneous-window β distributions')
ax.legend(fontsize=8)

ax = axes[0, 2]
medians = [homog_betas[t].median() for t in THRESHOLDS]
iqr_lo = [homog_betas[t].quantile(0.25) for t in THRESHOLDS]
iqr_hi = [homog_betas[t].quantile(0.75) for t in THRESHOLDS]
ax.errorbar(THRESHOLDS, medians,
            yerr=[np.array(medians)-np.array(iqr_lo),
                  np.array(iqr_hi)-np.array(medians)],
            fmt='o-', color='k', capsize=5, markersize=8)
ax.set_xlabel('gradient threshold (m/km)')
ax.set_ylabel('β (median ± IQR)')
ax.set_title('C) Homogeneous β stability')
ax.set_xticks(THRESHOLDS)

# Row 2: β uncertainty and within-segment variance
tick_labels = [f'{t} m/km\n(n={len(beta_uncs[t])})' for t in THRESHOLDS]

def _boxplot(ax, data, tick_labels, ylabel, title):
    bp = ax.boxplot(data, tick_labels=tick_labels, patch_artist=True, widths=0.5)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

_boxplot(axes[1, 0], [beta_uncs[t].values for t in THRESHOLDS],
         tick_labels, 'β uncertainty', 'D) Segment β uncertainty')

tick_labels_var = [f'{t} m/km\n(n={len(within_seg_vars[t])})' for t in THRESHOLDS]
_boxplot(axes[1, 1], [within_seg_vars[t].values for t in THRESHOLDS],
         tick_labels_var, 'within-segment β variance',
         'E) Within-segment β variance (homog.)')

ax = axes[1, 2]
for i, thresh in enumerate(THRESHOLDS):
    _, ss = load_data(thresh)
    ax.scatter(ss['n_windows'], ss['beta_uncertainty'],
               alpha=0.4, color=colors[i], s=20, label=f'{thresh} m/km')
ax.set_xlabel('windows per segment')
ax.set_ylabel('β uncertainty')
ax.set_title('F) β uncertainty vs segment length')
ax.legend(fontsize=8)

# Row 3: PSD intercept
_boxplot(axes[2, 0], [intercept_uncs[t].values for t in THRESHOLDS],
         [f'{t} m/km\n(n={len(intercept_uncs[t])})' for t in THRESHOLDS],
         'intercept uncertainty', 'G) PSD intercept uncertainty')
_boxplot(axes[2, 1], [within_seg_vars_intercept[t].values for t in THRESHOLDS],
         tick_labels_var, 'within-segment intercept var.',
         'H) Within-segment intercept variance (homog.)')

ax = axes[2, 2]
for i, thresh in enumerate(THRESHOLDS):
    ws, _ = load_data(thresh)
    ws_h = ws[~ws['is_transition']]
    ax.hist(ws_h['psd_intercept'].dropna(), bins=25, alpha=0.4, color=colors[i],
            label=f'{thresh} m/km', edgecolor='k', linewidth=0.3)
ax.set_xlabel('PSD intercept')
ax.set_ylabel('count')
ax.set_title('I) Homogeneous-window intercept distributions')
ax.legend(fontsize=8)

plt.suptitle('Gradient threshold sensitivity — post-fix pipeline\n'
             '(transition windows excluded from β analysis)',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(BASE / 'gradient_threshold_sensitivity.png', dpi=150)
plt.close()


# =============================================================================
# TZ-MERGE DISTANCE SENSITIVITY
# =============================================================================
merge_data_available = any(
    (BASE / f"merge_{d}km/{DATA_DIR}").exists() for d in MERGE_DISTANCES
)

if merge_data_available:
    print("\n\n" + "="*70)
    print("TZ-MERGE DISTANCE SENSITIVITY")
    print("="*70)

    merge_summary = []
    merge_betas = {}

    for d in MERGE_DISTANCES:
        ws, ss = load_merge_data(d)
        if ws is None:
            print(f"  {d} km: no data found, skipping")
            continue

        n_total = len(ws)
        n_trans = ws['is_transition'].sum()
        ws_h = ws[~ws['is_transition']]
        merge_betas[d] = ws_h['beta'].dropna()

        merge_summary.append({
            'merge_km': d,
            'n_windows_total': n_total,
            'n_transition': n_trans,
            'pct_transition': f"{n_trans/n_total*100:.1f}%",
            'n_homog': len(ws_h),
            'beta_median_homog': ws_h['beta'].median(),
            'beta_std_homog': ws_h['beta'].std(),
        })

    if merge_summary:
        print("\n--- Summary table ---\n")
        print(pd.DataFrame(merge_summary).to_string(index=False))

        valid_d = [d for d in MERGE_DISTANCES if d in merge_betas and len(merge_betas[d]) > 0]
        if len(valid_d) >= 2:
            print("\n--- Homogeneous β stability across merge distances ---")
            h_stat, p_val = stats.kruskal(*[merge_betas[d].values for d in valid_d])
            print(f"    Kruskal-Wallis: H = {h_stat:.3f}, p = {p_val:.4f}")
            print(f"    Medians: " + ", ".join(
                f"{d} km = {merge_betas[d].median():.3f}" for d in valid_d))

        # Figure 2: merge distance sensitivity
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        valid_d = [d for d in MERGE_DISTANCES if d in merge_betas]

        ax = axes[0]
        fracs = [ws['is_transition'].sum()/len(ws)*100
                 for d in valid_d
                 for ws, _ in [load_merge_data(d)] if ws is not None]
        ax.bar(range(len(valid_d)), fracs, color=colors[:len(valid_d)],
               edgecolor='k', linewidth=0.5)
        ax.set_xticks(range(len(valid_d)))
        ax.set_xticklabels([f'{d} km' for d in valid_d])
        ax.set_ylabel('% windows flagged transition')
        ax.set_title('A) Transition fraction vs merge distance')

        ax = axes[1]
        for i, d in enumerate(valid_d):
            ax.hist(merge_betas[d], bins=25, alpha=0.4, color=colors[i],
                    label=f'{d} km (n={len(merge_betas[d])})',
                    edgecolor='k', linewidth=0.3)
        ax.set_xlabel('β')
        ax.set_ylabel('count')
        ax.set_title('B) Homogeneous β by merge distance')
        ax.legend(fontsize=8)

        ax = axes[2]
        meds = [merge_betas[d].median() for d in valid_d]
        lo = [merge_betas[d].quantile(0.25) for d in valid_d]
        hi = [merge_betas[d].quantile(0.75) for d in valid_d]
        ax.errorbar(valid_d, meds,
                    yerr=[np.array(meds)-np.array(lo), np.array(hi)-np.array(meds)],
                    fmt='o-', color='k', capsize=5, markersize=8)
        ax.set_xlabel('TZ-merge distance (km)')
        ax.set_ylabel('β (median ± IQR)')
        ax.set_title('C) Homogeneous β stability')
        ax.set_xticks(valid_d)

        plt.suptitle('TZ-merge distance sensitivity — post-fix pipeline',
                     fontweight='bold')
        plt.tight_layout()
        plt.savefig(BASE / 'merge_distance_sensitivity.png', dpi=150)
        plt.close()
        print(f"\n  Saved: merge_distance_sensitivity.png")

else:
    print("\n\n--- TZ-merge distance: no data directories found ---")
    print(f"    Expected: {BASE}/merge_{{2,5,10}}km/{DATA_DIR}/")
    print("    Run bed_analysis.py with merge_gap_km = 2, 5, 10 to generate.")

print("\n\nDone.")

sys.stdout = sys.__stdout__
_log.close()
print(f"Log saved: {BASE / 'gradient_threshold_sensitivity.log'}")
