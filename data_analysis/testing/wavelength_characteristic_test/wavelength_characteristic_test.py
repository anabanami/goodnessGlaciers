"""Does a per-window characteristic wavelength behave like a vector element?

`wavelength_detections.csv` carries every bump more than 2x above the fitted power law,
which is a list and not a variable. Reducing it to one number per window needs two
choices: which band counts as a bedform, and which peak in that band is the one. This
scores the candidate reductions the same way the hill count was scored, so the definition
is chosen on evidence rather than asserted.

Candidates, all per window:
  wl_band      strongest peak by residual height inside the bedform band
  wl_all       strongest peak by residual height at any wavelength in the fit band
  wl_n_band    how many detections fall in the bedform band
  wl_height    residual height of the strongest in-band peak, an amplitude-like quantity
  wl_eta       Li_2010's eta over the same band, computed in the pipeline from the whole
               band rather than one peak, with the amplitude divided out

The band matters. Without it the strongest peak is often tens of km, which is segment
scale topography rather than a bedform, and Spagnolo_2017's MSGL range is what item 7's
literature case rests on.

Everything joins on the window, so hill count enters exactly rather than through a segment
median, and the controls are the same four the hill count residual was tested against.

Usage:  python wavelength_characteristic_test.py [run folder ...]   (from v23/ or ODSA/)
"""
import sys, io, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kruskal, rankdata

HERE = Path(__file__).resolve().parent


def _odsa_root(start):
    p = start
    while p != p.parent:
        if (p / 'all_data').is_dir():
            return p
        p = p.parent
    return start.parent


ODSA = _odsa_root(HERE)
sys.path.insert(0, str(HERE if (HERE / 'config.py').exists() else HERE.parent))
from config import WINDOW_SIZE, BEDFORM_BAND_M  # noqa: E402


def _resolve(p):
    q = Path(p)
    return q if q.exists() else ODSA / p


SOURCES = [_resolve(a) for a in sys.argv[1:]] or [ODSA / 'Ockenden-regions']
OUT = HERE / 'wavelength_characteristic'
OUT.mkdir(parents=True, exist_ok=True)

# Spagnolo_2017 MSGL range, the published bedform anchor item 7 rests on. Read from
# config so the peak reductions here and the eta the pipeline writes share one band.
BAND_M = BEDFORM_BAND_M

# Elements the reduction has to be independent of, all read from the window CSV except
# beta_iqr, which is a segment property broadcast onto its windows. hill_count is the
# overlap this test exists to measure.
AGAINST = ['beta', 'beta_iqr', 'relief_m', 'rms_roughness', 'psd_amplitude_1km',
           'skewness', 'kurtosis', 'hill_count', 'measures_speed_mean', 'incidence_deg']
# wl_eta is Li_2010's eta, computed in the pipeline over the same band and read from the
# window CSV rather than reduced from the peak list. The others pick one peak; it uses
# the whole band and divides the amplitude out.
CANDIDATES = ['wl_band', 'wl_all', 'wl_n_band', 'wl_height', 'wl_eta']
KEY = ['trajectory', 'segment', 'window_id']
MIN_RELIABLE_N = 30


class Tee:
    def __init__(self, *s): self.s = s
    def write(self, m): [x.write(m) for x in self.s]
    def flush(self): [x.flush() for x in self.s]


_buf = io.StringIO()
sys.stdout = Tee(sys.__stdout__, _buf)


def _short(name):
    for pfx in ['ASB_ICECAP_2010_', 'POLARGAP_2015_', 'Rec_Catch_', 'BAS_2012_ICEGRAV_']:
        name = name.replace(pfx, '')
    return name


def _spacing(src):
    """Median sample spacing per trajectory, parsed from the run's own log. The pipeline
    prints it per trajectory and does not export it, and it is worth controlling for
    because survey design tracks region almost perfectly."""
    f = src / 'bed_analysis_log.txt'
    if not f.exists():
        return {}
    out = {}
    for ln in f.read_text().splitlines():
        m = re.search(r'Trajectory (\S+): \d+ segments, combined median spacing = ([\d.]+)m', ln)
        if m:
            out.setdefault(m.group(1), []).append(float(m.group(2)))
    return {k: float(np.median(v)) for k, v in out.items()}


regions, skipped = {}, []
for src in SOURCES:
    spacing = _spacing(src)
    dets = sorted(src.glob('*/*_wavelength_detections.csv')) or \
           sorted(src.glob('*/*/*_wavelength_detections.csv'))
    for f in dets:
        stem = f.name.replace('_wavelength_detections.csv', '')
        seg_f = src / 'segment_csvs' / f'{stem}_segment_stats.csv'
        win_f = src / 'window_csvs' / f'{stem}_window_stats.csv'
        if not win_f.exists():
            skipped.append(f'{_short(stem)} (no window CSV)')
            continue
        d = pd.read_csv(f)
        if 'window_id' not in d.columns or 'residual_height' not in d.columns:
            skipped.append(f'{_short(stem)} (detections predate the window-level export)')
            continue
        win = pd.read_csv(win_f)
        win['dx_m'] = win.trajectory.map(spacing)
        # Accepts the single production column or a swept one at the adopted gate.
        if 'hill_count' not in win.columns and 'hill_count_20' in win.columns:
            win = win.rename(columns={'hill_count_20': 'hill_count'})
        win = win.rename(columns={'eta_wavelength_m': 'wl_eta'})

        # One row per window, from the detections that window produced.
        keep = KEY + ['wavelength_m', 'residual_height']
        in_band = d[d.wavelength_m.between(*BAND_M)]
        best_band = (in_band.sort_values('residual_height', ascending=False)
                     .groupby(KEY, as_index=False).first())
        best_all = (d.sort_values('residual_height', ascending=False)
                    .groupby(KEY, as_index=False).first())
        n_band = in_band.groupby(KEY, as_index=False).size().rename(columns={'size': 'wl_n_band'})

        win = win.merge(best_band[keep].rename(columns={'wavelength_m': 'wl_band',
                                                        'residual_height': 'wl_height'}),
                        on=KEY, how='left')
        win = win.merge(best_all[KEY + ['wavelength_m']].rename(columns={'wavelength_m': 'wl_all'}),
                        on=KEY, how='left')
        win = win.merge(n_band, on=KEY, how='left')
        win['wl_n_band'] = win['wl_n_band'].fillna(0)

        # beta_iqr is defined per segment and only where the segment holds two or more
        # windows, so it stays thin however it is joined.
        if seg_f.exists():
            s = pd.read_csv(seg_f)
            if 'beta_iqr' in s.columns:
                win = win.merge(s[['trajectory', 'segment', 'beta_iqr']],
                                on=['trajectory', 'segment'], how='left')
        regions[_short(stem)] = win

if skipped:
    print('skipped: ' + '; '.join(skipped) + '\n')
if not regions:
    sys.exit('no usable detections. They must carry window_id and residual_height, which '
             'means a run made after per-window detection was added to bed_analysis.py')

print('source: ' + ', '.join(str(s) for s in SOURCES))
print(f'bedform band {BAND_M[0]}-{BAND_M[1]} m, window {WINDOW_SIZE/1000:.0f} km')
print(f'{len(regions)} region(s), {sum(len(d) for d in regions.values())} windows\n')

# ── Coverage: can the reduction even be computed? ─────────────────────────────
print('COVERAGE (a reduction undefined on most windows cannot be a vector element)')
print(f"{'region':<34s} {'wins':>5s} {'wl_band':>9s} {'wl_all':>9s} "
      f"{'med band':>9s} {'IQR band':>9s} {'med all':>9s} "
      f"{'med eta':>9s} {'IQR eta':>9s} {'med n':>6s}")
print('-' * 116)
for name, d in sorted(regions.items()):
    iqr = lambda s: s.quantile(.75) - s.quantile(.25)
    print(f'{name:<34s} {len(d):5d} '
          f'{100*d.wl_band.notna().mean():8.1f}% {100*d.wl_all.notna().mean():8.1f}% '
          f'{d.wl_band.median():9.0f} {iqr(d.wl_band):9.0f} {d.wl_all.median():9.0f} '
          f'{d.wl_eta.median() if "wl_eta" in d else np.nan:9.0f} '
          f'{iqr(d.wl_eta) if "wl_eta" in d else np.nan:9.0f} {d.wl_n_band.median():6.0f}')
print()

# ── Independence ──────────────────────────────────────────────────────────────
rows = []
for name, d in sorted(regions.items()):
    for c in CANDIDATES:
        cells, ns = [], []
        for a in AGAINST:
            if a not in d.columns:
                cells.append(np.nan); ns.append(0); continue
            m = d[c].notna() & d[a].notna()
            ns.append(int(m.sum()))
            cells.append(spearmanr(d.loc[m, c], d.loc[m, a]).statistic
                         if m.sum() >= 8 and d.loc[m, c].nunique() > 1 else np.nan)
        rows.append({'region': name, 'candidate': c,
                     **dict(zip(AGAINST, cells)), **{f'n_{a}': n for a, n in zip(AGAINST, ns)}})
corr = pd.DataFrame(rows)
corr.to_csv(OUT / 'wavelength_correlations.csv', index=False)

print('SPEARMAN vs WINDOW-LEVEL ELEMENTS (high = redundant)')
print(f"{'region':<34s} {'candidate':<10s} " + ''.join(f'{a[:11]:>12s}' for a in AGAINST))
print('-' * (46 + 12 * len(AGAINST)))
for _, r in corr.iterrows():
    print(f'{r["region"]:<34s} {r["candidate"]:<10s} ' +
          ''.join(f'{r[a]:12.2f}' if np.isfinite(r[a]) else f"{'-':>12s}" for a in AGAINST))
print()

reliable = corr[AGAINST].where(corr[[f'n_{a}' for a in AGAINST]].to_numpy() >= MIN_RELIABLE_N)
print(f'WORST |rho| PER CANDIDATE (n >= {MIN_RELIABLE_N})')
for c in CANDIDATES:
    sub = reliable[corr.candidate == c].abs()
    if sub.notna().any().any():
        flat = sub.stack(); i, j = flat.idxmax()
        print(f'  {c:<10s} {flat.max():.2f}  ({j}, {corr.loc[i, "region"]}, n={corr.loc[i, f"n_{j}"]})')
    else:
        print(f'  {c:<10s} no cell reaches n >= {MIN_RELIABLE_N}')
print()

# ── Is Li's index decoupled the way it claims? ────────────────────────────────
# eta is xi over the slope-spectrum integral, so it should be free of xi, and xi itself
# should be the band-limited amplitude that rms_slope turned out to duplicate.
if any('xi_band' in d for d in regions.values()):
    print('LI TWO-PARAMETER CHECK (xi is diagnostic here, not a proposed element)')
    print(f"{'region':<34s} {'rho(eta,xi)':>12s} {'rho(xi,amp1km)':>15s} {'n':>6s}")
    for name, d in sorted(regions.items()):
        if 'xi_band' not in d or 'wl_eta' not in d:
            continue
        m = d.wl_eta.notna() & d.xi_band.notna() & d.psd_amplitude_1km.notna()
        if m.sum() < 8:
            continue
        print(f'{name:<34s} {spearmanr(d.loc[m, "wl_eta"], d.loc[m, "xi_band"]).statistic:12.2f} '
              f'{spearmanr(d.loc[m, "xi_band"], d.loc[m, "psd_amplitude_1km"]).statistic:15.2f} '
              f'{int(m.sum()):6d}')
    print()

# ── Region separation beyond the existing elements ────────────────────────────
# The same four the hill count residual was tested against, so the two are comparable.
CONTROLS = ['beta', 'relief_m', 'rms_roughness', 'psd_amplitude_1km']


def _eps2(groups):
    groups = [np.asarray(g) for g in groups if len(g) > 1]
    if len(groups) < 2:
        return np.nan
    n, k = sum(len(g) for g in groups), len(groups)
    return (kruskal(*groups).statistic - k + 1) / (n - k)


def _rank_resid(frame, target, controls):
    X = np.column_stack([rankdata(frame[c]) for c in controls] + [np.ones(len(frame))])
    y = rankdata(frame[target])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


pool = pd.concat([d.assign(_region=k) for k, d in regions.items()], ignore_index=True)
print('REGION SEPARATION, POOLED (Kruskal-Wallis epsilon^2)')
print(f'Controls: {", ".join(CONTROLS)}. A candidate that collapses once they are removed')
print('is those elements re-expressed and adds no degree of freedom.')
base = pool.dropna(subset=CONTROLS + ['_region'])
gb = [base[base._region == r] for r in sorted(base._region.unique())]
print('  baseline: ' + '   '.join(f'{c} {_eps2([x[c] for x in gb]):.3f}' for c in CONTROLS))
dxb = pool.dropna(subset=['dx_m', '_region'])
if len(dxb):
    print('  sampling spacing separates the regions on its own at '
          f"{_eps2([dxb[dxb._region == r].dx_m for r in sorted(dxb._region.unique())]):.3f}, "
          'so it is carried as a fifth control')
print(f"{'candidate':<10s} {'n':>5s} {'raw':>8s} {'| controls':>12s} {'| + spacing':>13s}")
print('-' * 54)
resid_by = {}
for c in CANDIDATES:
    sub = pool[['_region', c] + CONTROLS].dropna()
    if sub[c].nunique() < 2 or sub._region.nunique() < 2:
        print(f'{c:<10s} {len(sub):5d}   too few distinct values')
        continue
    sub = sub.assign(_r=_rank_resid(sub, c, CONTROLS))
    g = [sub[sub._region == r] for r in sorted(sub._region.unique())]
    resid_by[c] = (sorted(sub._region.unique()), [x['_r'].to_numpy() for x in g])
    sub5 = pool[['_region', c] + CONTROLS + ['dx_m']].dropna()
    if len(sub5) and sub5[c].nunique() > 1:
        sub5 = sub5.assign(_r=_rank_resid(sub5, c, CONTROLS + ['dx_m']))
        g5 = [sub5[sub5._region == r] for r in sorted(sub5._region.unique())]
        extra = f'{_eps2([x["_r"] for x in g5]):13.3f}'
    else:
        extra = f'{"-":>13s}'
    print(f'{c:<10s} {len(sub):5d} {_eps2([x[c] for x in g]):8.3f} '
          f'{_eps2([x["_r"] for x in g]):12.3f}{extra}')

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ax = axes[0]
allb = pool.wl_band.dropna()
alla = pool.wl_all.dropna()
bins = np.geomspace(200, 60000, 40)
ax.hist(alla, bins=bins, histtype='step', lw=1.8, label='strongest, any scale')
ax.hist(allb, bins=bins, histtype='step', lw=1.8, label=f'strongest in {BAND_M[0]}-{BAND_M[1]} m')
for b in BAND_M:
    ax.axvline(b, color='red', ls='--', lw=1.0)
ax.set_xscale('log')
ax.set_xlabel('wavelength (m)')
ax.set_ylabel('windows')
ax.set_title('Characteristic wavelength by reduction')
ax.legend(fontsize=8)

ax = axes[1]
if resid_by:
    ks = [c for c in CANDIDATES if c in resid_by]
    ax.boxplot([np.concatenate(resid_by[c][1]) for c in ks], patch_artist=True,
               showfliers=False, widths=0.6, medianprops=dict(color='black', lw=2))
    ax.set_xticklabels(ks, rotation=30, ha='right', fontsize=9)
    ax.axhline(0, color='red', ls='--', lw=1.0)
    ax.set_ylabel('rank, existing elements removed')
    ax.set_title('Residual spread by candidate')
fig.tight_layout()
fig.savefig(OUT / 'wavelength_characteristic.png', dpi=150)
print(f'\nSaved: {OUT / "wavelength_characteristic.png"}')
print(f'Saved: {OUT / "wavelength_correlations.csv"}')

sys.stdout = sys.__stdout__
(OUT / 'wavelength_characteristic_test.log').write_text(_buf.getvalue())
print(f'Log written to {OUT / "wavelength_characteristic_test.log"}')
