"""Distribution analysis of relief, beta, and RMS roughness across Ockenden reference regions."""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent          # .../v23
ODSA = HERE.parent                               # .../ODSA
OCKENDEN = ODSA / 'Ockenden-regions'
# THRESHOLDS = [200, 600]  # initial values — to be revised from this analysis
# THRESHOLDS = [300, 800]
# THRESHOLDS = [350, 900]
# Production. Must match RELIEF_CLASSES in bed_character.py, which sets relief_class
# in the window CSVs. 350/800 sit on the relief P25 (368) and P75 (815).
THRESHOLDS = [350, 800]

# Cross-check the sweep value against the adopted production value. THRESHOLDS is
# deliberately free to vary (it is the derivation knob), so a mismatch is a
# non-fatal reminder, not an error: if a swept value is adopted, RELIEF_CLASSES in
# bed_character.py has to be updated to match. The message is emitted after the Tee
# is installed below, so it lands in relief_distribution.log alongside the results.
import sys as _sys
_sys.path.insert(0, str(ODSA))
try:
    from bed_character import RELIEF_THRESHOLDS as _PROD_THRESHOLDS
except Exception as _e:
    THRESHOLD_CHECK = [f"NOTE: could not import RELIEF_THRESHOLDS from bed_character.py to cross-check ({_e})."]
else:
    if list(THRESHOLDS) == list(_PROD_THRESHOLDS):
        THRESHOLD_CHECK = [f"Relief-threshold check: THRESHOLDS {THRESHOLDS} match production "
                           f"RELIEF_CLASSES {_PROD_THRESHOLDS} in bed_character.py."]
    else:
        THRESHOLD_CHECK = [f"WARNING: THRESHOLDS {THRESHOLDS} differ from production "
                           f"RELIEF_CLASSES {_PROD_THRESHOLDS} in bed_character.py.",
                           "         Expected during a sweep. If you adopt this value, update "
                           "RELIEF_CLASSES in bed_character.py to match."]


# Results for this run go to v23/relief_thresholds/<thresholds joined by '-'>/
OUT = HERE / 'relief_thresholds' / '-'.join(str(t) for t in THRESHOLDS)
OUT.mkdir(parents=True, exist_ok=True)


LANDSCAPE_CLASS = {
    'Fig4C_Aurora_SB_lowrelief': 'low-relief',
    'Fig2A_Maud_SB': 'low-relief',
    'Fig2D_Recovery_SL': 'low-relief',
    'Fig2G_Highland_A': 'alpine',
    'Fig2H_Golicyna_SM': 'alpine',
    'Fig2C_Hercules_Dome': 'alpine',
    'Pensacola_Pole': 'selective erosion',
}

CLASS_COLORS = {
    'low-relief': '#2196F3',
    'alpine': '#E65100',
    'selective erosion': '#7B1FA2',
}

def load_windows(folder):
    dfs = {}
    for f in sorted((folder / 'window_csvs').glob('*window_stats*.csv')):
        name = f.stem.replace('_w50km_window_stats', '')
        for pfx in ['ASB_ICECAP_2010_', 'POLARGAP_2015_', 'Rec_Catch_']:
            name = name.replace(pfx, '')
        dfs[name] = pd.read_csv(f)
    return dfs

import sys, io

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
    def flush(self):
        for st in self.streams:
            st.flush()

_tee_buf = io.StringIO()
sys.stdout = Tee(sys.__stdout__, _tee_buf)

for _line in THRESHOLD_CHECK:
    print(_line)

regions = load_windows(OCKENDEN)

# tag each region with its landscape class
for name in list(regions):
    cls = LANDSCAPE_CLASS.get(name, 'unknown')
    regions[name]['_class'] = cls

all_relief = np.concatenate([df['relief_m'].dropna().values for df in regions.values()])
all_beta = np.concatenate([df['beta'].dropna().values for df in regions.values()])
all_rms = np.concatenate([df['rms_roughness'].dropna().values for df in regions.values()])

print(f"Total windows: {len(all_relief)}")
print(f"\nRELIEF (m):")
for p in [5, 10, 25, 50, 75, 90, 95]:
    print(f"  P{p:02d}: {np.percentile(all_relief, p):.0f}")
print(f"  mean: {all_relief.mean():.0f}, std: {all_relief.std():.0f}")

flat = np.sum(all_relief < THRESHOLDS[0])
subdued = np.sum((all_relief >= THRESHOLDS[0]) & (all_relief < THRESHOLDS[1]))
mount = np.sum(all_relief >= THRESHOLDS[1])
print(f"\nCurrent classification (thresholds {THRESHOLDS}):")
print(f"  flat:        {flat:4d} ({100*flat/len(all_relief):.1f}%)")
print(f"  subdued:     {subdued:4d} ({100*subdued/len(all_relief):.1f}%)")
print(f"  mountainous: {mount:4d} ({100*mount/len(all_relief):.1f}%)")

# per-class summary
print(f"\n{'Class':<20s} {'N':>5s} {'med':>6s} {'IQR':>12s}")
print('-'*50)
for cls in ['low-relief', 'alpine', 'selective erosion']:
    r = np.concatenate([df['relief_m'].dropna().values for df in regions.values() if df['_class'].iloc[0] == cls])
    q25, q50, q75 = np.percentile(r, [25, 50, 75])
    print(f"{cls:<20s} {len(r):5d} {q50:6.0f} [{q25:5.0f}-{q75:5.0f}]")

# per-region summary
print(f"\n{'Region':<35s} {'Class':<18s} {'N':>5s} {'med':>6s} {'IQR':>12s} {'B_med':>6s} {'RMS_med':>8s}")
print('-'*95)
for name, df in sorted(regions.items()):
    r = df['relief_m'].dropna()
    b = df['beta'].dropna()
    rms = df['rms_roughness'].dropna()
    cls = LANDSCAPE_CLASS.get(name, '?')
    q25, q50, q75 = np.percentile(r, [25, 50, 75])
    print(f"{name:<35s} {cls:<18s} {len(r):5d} {q50:6.0f} [{q25:5.0f}-{q75:5.0f}] {b.median():6.2f} {rms.median():8.1f}")

# ── FIGURE 1: Relief histograms ──
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

ax = axes[0]
bins = np.arange(0, 2200, 50)
for cls in ['low-relief', 'alpine', 'selective erosion']:
    vals = np.concatenate([df['relief_m'].dropna().values for df in regions.values() if df['_class'].iloc[0] == cls])
    ax.hist(vals, bins=bins, alpha=0.6, label=cls, color=CLASS_COLORS[cls], edgecolor='white', linewidth=0.3)
for t in THRESHOLDS:
    ax.axvline(t, color='red', ls='--', lw=1.5, label=f'threshold {t}m')
ax.set_xlabel('Relief (m)')
ax.set_ylabel('Window count')
ax.set_title('Relief distribution by landscape class')
ax.legend()

ax = axes[1]
data, labels, colors = [], [], []
for name, df in sorted(regions.items(), key=lambda x: LANDSCAPE_CLASS.get(x[0], '')):
    data.append(df['relief_m'].dropna().values)
    labels.append(name)
    colors.append(CLASS_COLORS.get(LANDSCAPE_CLASS.get(name, ''), 'gray'))
bp = ax.boxplot(data, vert=True, patch_artist=True, showfliers=False, widths=0.6,
                medianprops=dict(color='black', linewidth=2))
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
for t in THRESHOLDS:
    ax.axhline(t, color='red', ls='--', lw=1.5)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Relief (m)')
ax.set_title('Relief by region (boxes = IQR, whiskers = 1.5xIQR)')
fig.tight_layout()
fig.savefig(OUT / 'relief_distribution.png', dpi=150)
print(f"\nSaved: relief_distribution.png")

# ── FIGURE 2: Beta vs Relief, RMS vs Relief ──
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

ax = axes2[0]
for name, df in sorted(regions.items()):
    c = CLASS_COLORS.get(LANDSCAPE_CLASS.get(name, ''), 'gray')
    ax.scatter(df['relief_m'], df['beta'], s=8, alpha=0.4, c=c, edgecolors='none')
for t in THRESHOLDS:
    ax.axvline(t, color='red', ls='--', lw=1.2)
ax.set_xlabel('Relief (m)')
ax.set_ylabel('Beta')
ax.set_title('Beta vs Relief')
ax.legend(handles=[Line2D([0],[0],marker='o',color=c,ls='',ms=6,label=cls)
                   for cls, c in CLASS_COLORS.items()])

ax = axes2[1]
for name, df in sorted(regions.items()):
    c = CLASS_COLORS.get(LANDSCAPE_CLASS.get(name, ''), 'gray')
    ax.scatter(df['relief_m'], df['rms_roughness'], s=8, alpha=0.4, c=c, edgecolors='none')
for t in THRESHOLDS:
    ax.axvline(t, color='red', ls='--', lw=1.2)
ax.set_xlabel('Relief (m)')
ax.set_ylabel('RMS roughness (m)')
ax.set_title('RMS roughness vs Relief')
fig2.tight_layout()
fig2.savefig(OUT / 'relief_vs_metrics.png', dpi=150)
print(f"Saved: relief_vs_metrics.png")

# ── FIGURE 3: Relief CDF ──
fig3, ax3 = plt.subplots(figsize=(8, 5))
sorted_r = np.sort(all_relief)
cdf = np.arange(1, len(sorted_r)+1) / len(sorted_r)
ax3.plot(sorted_r, cdf, 'k-', lw=2)
# per-class CDFs
for cls in ['low-relief', 'alpine', 'selective erosion']:
    vals = np.sort(np.concatenate([df['relief_m'].dropna().values for df in regions.values() if df['_class'].iloc[0] == cls]))
    ax3.plot(vals, np.arange(1, len(vals)+1)/len(vals), color=CLASS_COLORS[cls], lw=1.5, alpha=0.7, label=cls)
for t in THRESHOLDS:
    frac = np.searchsorted(sorted_r, t) / len(sorted_r)
    ax3.axvline(t, color='red', ls='--', lw=1.5)
    ax3.annotate(f'{t}m -> {frac:.1%}', xy=(t, frac), xytext=(t+50, frac-0.08),
                fontsize=10, color='red')
ax3.set_xlabel('Relief (m)')
ax3.set_ylabel('Cumulative fraction')
ax3.set_title('Relief CDF — all windows + per class')
ax3.legend()
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig(OUT / 'relief_cdf.png', dpi=150)
print(f"Saved: relief_cdf.png")

sys.stdout = sys.__stdout__
with open(OUT / 'relief_distribution.log', 'w') as f:
    f.write(_tee_buf.getvalue())
print(f"Log written to {OUT / 'relief_distribution.log'}")

# plt.show()
