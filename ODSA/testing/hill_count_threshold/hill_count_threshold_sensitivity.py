"""Which hill-count relief threshold, if any, earns a vector slot.

Reads hill_count_{20,50,100,250} from the window CSVs (bed_analysis emits all four
in one pass) and reports, per region and threshold, the resolving power of the count and
its Spearman correlation with the elements already in the vector. A threshold is useless
if the count is degenerate (most windows at 0-1, or saturated at the WINDOW_SIZE/5 km
ceiling) and redundant if it tracks an existing element. incidence_deg is included
because a transect count is a geometry-dependent read of an areal quantity.

Usage:  python hill_count_threshold_sensitivity.py [run folder ...]   (from v23/ or ODSA/)
        default is Ockenden-regions/; pass several when the regions were run
        one per folder, e.g. individual_region_TEST/{RSL,HD,PPB}
"""
import sys, io
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kruskal, rankdata

HERE = Path(__file__).resolve().parent


def _odsa_root(start):
    """Nearest ancestor holding all_data/, so this file works unchanged from v23/ and from
    the frozen snapshot several folders deeper."""
    p = start
    while p != p.parent:
        if (p / 'all_data').is_dir():
            return p
        p = p.parent
    return start.parent


ODSA = _odsa_root(HERE)
# config.py sits beside this script inside the snapshot and one level up in v23/. Taking
# the adjacent one first is what makes a snapshot copy read its own frozen constants.
sys.path.insert(0, str(HERE if (HERE / 'config.py').exists() else HERE.parent))
from config import HILL_THRESHOLD_M, HILL_BOX_M, WINDOW_SIZE

# The swept values are this test's parameter, not production's, so they live here.
# Production carries only the adopted gate. Reading them from config instead would
# silently reduce the sweep to that one column. Ockenden_2026 publish these four.
HILL_SWEEP_THRESHOLDS = (20, 50, 100, 250)

# Cross-check against the adopted value, the same way relief_distribution.py checks its
# THRESHOLDS against bed_character.RELIEF_CLASSES. A mismatch is a non-fatal reminder:
# an adopted gate outside the swept set has no evidence in this log. Emitted after the
# Tee is installed below so it lands in the log with the results.
ADOPTED_CHECK = (
    [f'Adopted gate check: HILL_THRESHOLD_M = {HILL_THRESHOLD_M} m is in the swept set '
     f'{HILL_SWEEP_THRESHOLDS}.']
    if HILL_THRESHOLD_M in HILL_SWEEP_THRESHOLDS else
    [f'WARNING: HILL_THRESHOLD_M = {HILL_THRESHOLD_M} m is not in the swept set '
     f'{HILL_SWEEP_THRESHOLDS}, so nothing below is evidence for it.',
     '         Either sweep that value or revisit the adopted gate in config.py.'])

def _resolve(p):
    """Accept a path relative to the cwd or to the ODSA root, so this runs from either."""
    q = Path(p)
    return q if q.exists() else ODSA / p


SOURCES = [_resolve(a) for a in sys.argv[1:]] or [ODSA / 'Ockenden-regions']
OUT = HERE / 'hill_count_threshold'
OUT.mkdir(parents=True, exist_ok=True)

# The vector elements the count has to be independent of to be worth adding, plus the
# geometry term. skewness and kurtosis are per window as of 2026-08-02 and matter most
# here: both descend from Sugden's hills-versus-troughs idea, so a hill count that merely
# restates phase asymmetry would be the cheapest way for this variable to fail. They read
# as '-' against any CSV written before that date. beta_iqr is exported but is defined
# only for segments holding two or more windows, which is a minority everywhere and
# near-absent at Hercules Dome, so it is joined on from the segment CSV and will be blank
# where the segment could not support it.
AGAINST = ['beta', 'beta_iqr', 'relief_m', 'rms_roughness', 'psd_amplitude_1km',
           'skewness', 'kurtosis', 'bed_elev_mean', 'measures_speed_mean', 'incidence_deg']
CEILING = WINDOW_SIZE / HILL_BOX_M


class Tee:
    def __init__(self, *s): self.s = s
    def write(self, m): [x.write(m) for x in self.s]
    def flush(self): [x.flush() for x in self.s]


_buf = io.StringIO()
sys.stdout = Tee(sys.__stdout__, _buf)

for _line in ADOPTED_CHECK:
    print(_line)
print()

# A source is either a run folder holding window_csvs/ (Ockenden-regions) or a parent of
# run folders (individual_region_TEST). Take both, and label by run folder as well as
# region when more than one run is in play, so a baseline and a re-run cannot collide.
found = []
for src in SOURCES:
    hits = sorted(src.glob('window_csvs/*window_stats*.csv')) or \
           sorted(src.glob('*/window_csvs/*window_stats*.csv'))
    for f in hits:
        name = f.stem.replace('_window_stats', '')
        for pfx in ['ASB_ICECAP_2010_', 'POLARGAP_2015_', 'Rec_Catch_', 'BAS_2012_ICEGRAV_']:
            name = name.replace(pfx, '')
        found.append((f.parent.parent.name, name, f))

if not found:
    sys.exit(f'no window CSVs under {", ".join(str(s) for s in SOURCES)} '
             f'(looked in <source>/window_csvs and <source>/*/window_csvs)')

# Prefix with the run folder only where a region name occurs in more than one run, so
# labels stay short in the normal case and two runs of the same region cannot silently
# overwrite each other. Pooling two runs of one region double counts it, hence the warning.
_name_runs = {}
for run, name, _ in found:
    _name_runs.setdefault(name, set()).add(run)
_dupes = sorted(n for n, r in _name_runs.items() if len(r) > 1)
if _dupes:
    print(f'WARNING: {len(_dupes)} region(s) appear in more than one run folder '
          f'({", ".join(_dupes[:3])}{"..." if len(_dupes) > 3 else ""}). They are kept '
          f'separate, but the pooled statistics below will count them twice. Pass a '
          f'single run folder unless that is what you want.\n')

cols = [f'hill_count_{t}' for t in HILL_SWEEP_THRESHOLDS]
regions, skipped, iqr_note = {}, [], []
for run, name, f in found:
    df = pd.read_csv(f)
    if any(c not in df.columns for c in cols):
        skipped.append(f'{run}/{name}')
        continue
    # beta_iqr lives on the segment, so bring it down to the windows of that segment. The
    # segment CSV has to be this region's: a folder holding several regions would
    # otherwise merge every window frame against whichever file globbed first.
    seg = f.parent.parent / 'segment_csvs' / f.name.replace('_window_stats.csv',
                                                            '_segment_stats.csv')
    if seg.exists() and 'beta_iqr' not in df.columns:
        s = pd.read_csv(seg)
        if {'trajectory', 'segment', 'beta_iqr'} <= set(s.columns):
            n_before = len(df)
            df = df.merge(s[['trajectory', 'segment', 'beta_iqr']].drop_duplicates(
                subset=['trajectory', 'segment']), on=['trajectory', 'segment'], how='left')
            assert len(df) == n_before, f'{name}: beta_iqr merge changed the row count'
            iqr_note.append(f'{name}: beta_iqr on {df.beta_iqr.notna().sum()}/{len(df)} windows')
    elif not seg.exists():
        iqr_note.append(f'{name}: no matching segment CSV, beta_iqr unavailable')
    regions[f'{run}/{name}' if len(_name_runs[name]) > 1 else name] = df

if skipped:
    print(f'skipped {len(skipped)} run(s) predating the hill-count column: '
          f'{", ".join(skipped)}\n')
if not regions:
    sys.exit('no window CSVs carry the per-threshold hill_count_<t> columns. Production '
             'emits a single hill_count at the adopted gate, so the sweep reads a run made '
             'by the snapshot pipeline in v23/hill_count_threshold/hill_count_pipeline/')

if iqr_note:
    print('beta_iqr is undefined for single-window segments: ' + '; '.join(iqr_note) + '\n')

print('source: ' + ', '.join(str(s) for s in SOURCES))
print(f'window {WINDOW_SIZE/1000:.0f} km, box {HILL_BOX_M/1000:.0f} km -> count ceiling ~{CEILING:.0f}')
print(f'{len(regions)} region(s), {sum(len(d) for d in regions.values())} windows\n')

# ── Resolving power ───────────────────────────────────────────────────────────
print('RESOLVING POWER (a count pinned at 0-1 or at the ceiling separates nothing)')
print(f"{'region':<42s} {'thr':>5s} {'n':>5s} {'med':>4s} {'IQR':>9s} {'max':>4s} "
      f"{'%at 0-1':>8s} {'%at ceil':>9s} {'distinct':>9s}")
print('-' * 110)
for name, df in sorted(regions.items()):
    for t in HILL_SWEEP_THRESHOLDS:
        c = df[f'hill_count_{t}'].dropna()
        if c.empty:
            print(f'{name:<42s} {t:5d}     - all NaN (windows shorter than the box?)')
            continue
        q25, q50, q75 = np.percentile(c, [25, 50, 75])
        print(f'{name:<42s} {t:5d} {len(c):5d} {q50:4.0f} [{q25:3.0f}-{q75:3.0f}] {c.max():4.0f} '
              f'{100*(c <= 1).mean():7.1f}% {100*(c >= CEILING).mean():8.1f}% {c.nunique():9d}')
    print()

# ── Independence ──────────────────────────────────────────────────────────────
print('SPEARMAN vs EXISTING VECTOR ELEMENTS (high = redundant; the whole case for the')
print('variable is that no column is consistently high across regions)')
print(f"{'region':<42s} {'thr':>5s} " + ''.join(f'{a[:11]:>12s}' for a in AGAINST))
print('-' * (48 + 12 * len(AGAINST)))
# A rank correlation on a handful of windows is not evidence of redundancy. At n=30 the
# 5% critical |rho| is about 0.36, so a cell below that n cannot distinguish a headline
# 0.8 from noise. Such cells are still printed, because a real coupling in a small region
# is worth seeing, but they are barred from setting the worst-case number.
MIN_RELIABLE_N = 30
rows = []
for name, df in sorted(regions.items()):
    for t in HILL_SWEEP_THRESHOLDS:
        cells, ns = [], []
        for a in AGAINST:
            if a not in df.columns:
                cells.append(np.nan), ns.append(0)
                continue
            m = df[f'hill_count_{t}'].notna() & df[a].notna()
            ns.append(int(m.sum()))
            cells.append(spearmanr(df.loc[m, f'hill_count_{t}'], df.loc[m, a]).statistic
                         if m.sum() >= 8 and df.loc[m, f'hill_count_{t}'].nunique() > 1 else np.nan)
        rows.append({'region': name, 'threshold': t,
                     **{a: c for a, c in zip(AGAINST, cells)},
                     **{f'n_{a}': n for a, n in zip(AGAINST, ns)}})
        print(f'{name:<42s} {t:5d} ' +
              ''.join(f'{c:12.2f}' if np.isfinite(c) else f"{'-':>12s}" for c in cells))
    print()

corr = pd.DataFrame(rows)
corr.to_csv(OUT / 'hill_count_correlations.csv', index=False)

# Mask every cell that rests on too few paired windows before taking the maximum.
reliable = corr[AGAINST].where(corr[[f'n_{a}' for a in AGAINST]].to_numpy() >= MIN_RELIABLE_N)
thin_cells = sorted({f'{corr.loc[i, "region"].split("/")[-1]}/{a} (n={corr.loc[i, f"n_{a}"]})'
                     for a in AGAINST for i in corr.index
                     if np.isfinite(corr.loc[i, a]) and corr.loc[i, f'n_{a}'] < MIN_RELIABLE_N})

print(f'WORST |rho| PER THRESHOLD, ACROSS ALL REGIONS AND ELEMENTS (n >= {MIN_RELIABLE_N})')
print('(the number that decides redundancy; the earlier 3-region test put thr=50 at 0.63)')
for t in HILL_SWEEP_THRESHOLDS:
    sub = reliable[corr.threshold == t].abs()
    if sub.notna().any().any():
        flat = sub.stack()
        i, j = flat.idxmax()
        print(f'  thr {t:3d} m: {flat.max():.2f}  ({j}, {corr.loc[i, "region"]}, '
              f'n={corr.loc[i, f"n_{j}"]})')
    else:
        print(f'  thr {t:3d} m: no cell reaches n >= {MIN_RELIABLE_N}')
if thin_cells:
    print(f'  excluded as too small: {", ".join(thin_cells)}')

# ── Independence beyond beta ──────────────────────────────────────────────────
# The sweep shows the count tracking beta at a loose gate and the amplitude family at a
# tight one, so the open question is whether anything is left once beta is removed. Two
# things have to hold for the count to earn a vector slot. Its remaining correlations
# must survive beta being partialled out, which says it is not a beta proxy. And its
# beta-residual must still separate the regions, which says the leftover is landscape
# signal rather than the noise of a coarse integer.


def _sp(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return spearmanr(x[m], y[m]).statistic if m.sum() >= 8 else np.nan


PARTIAL = [a for a in AGAINST if a != 'beta']
print('\nPARTIAL SPEARMAN CONTROLLING FOR BETA (what survives once beta is removed)')
print(f"{'region':<42s} {'thr':>5s} " + ''.join(f'{a[:11]:>12s}' for a in PARTIAL))
print('-' * (48 + 12 * len(PARTIAL)))
for name, df in sorted(regions.items()):
    for t in HILL_SWEEP_THRESHOLDS:
        h = df[f'hill_count_{t}'].to_numpy(float)
        b = df['beta'].to_numpy(float)
        r_hb = _sp(h, b)
        cells = []
        for a in PARTIAL:
            r_hx, r_bx = (_sp(h, df[a].to_numpy(float)), _sp(b, df[a].to_numpy(float))) \
                if a in df.columns else (np.nan, np.nan)
            den = np.sqrt(max(0.0, (1 - r_hb ** 2) * (1 - r_bx ** 2)))
            cells.append((r_hx - r_hb * r_bx) / den if den > 1e-12 else np.nan)
        print(f'{name:<42s} {t:5d} ' +
              ''.join(f'{c:12.2f}' if np.isfinite(c) else f"{'-':>12s}" for c in cells))
    print()


def _eps2(groups):
    """Kruskal-Wallis effect size: 0 = regions indistinguishable, 1 = fully separated."""
    groups = [np.asarray(g) for g in groups if len(g) > 1]
    if len(groups) < 2:
        return np.nan
    n, k = sum(len(g) for g in groups), len(groups)
    return (kruskal(*groups).statistic - k + 1) / (n - k)


# Controlling for beta alone is not enough: at a tight gate the count tracks the
# amplitude family, so a beta-residual that still separates the regions could be relief
# re-expressed as an integer. CONTROLS is every existing element the count could be
# hiding inside, removed together by rank regression.
CONTROLS = [c for c in ['beta', 'relief_m', 'rms_roughness', 'psd_amplitude_1km']]
pool = pd.concat([d.assign(_region=k) for k, d in regions.items()], ignore_index=True)


def _rank_resid(frame, target, controls):
    """Residual of target's ranks after least-squares removal of the controls' ranks."""
    X = np.column_stack([rankdata(frame[c]) for c in controls] + [np.ones(len(frame))])
    y = rankdata(frame[target])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


print('REGION SEPARATION, POOLED (Kruskal-Wallis epsilon^2)')
print('Each existing element on its own is the bar. hill|beta removes beta; hill|all')
print('removes beta, relief, rms and amplitude together. If hill|all collapses toward 0,')
print('the count is those elements re-expressed and adds no degree of freedom.')
base = pool.dropna(subset=CONTROLS + ['_region'])
gb = [base[base._region == r] for r in sorted(base._region.unique())]
print('  baseline: ' + '   '.join(f'{c} {_eps2([x[c] for x in gb]):.3f}' for c in CONTROLS))
print(f"{'thr':>5s} {'n':>5s} {'hill':>10s} {'hill|beta':>11s} {'hill|all':>10s}")
print('-' * 45)
resid_by_thr = {}
for t in HILL_SWEEP_THRESHOLDS:
    sub = pool[['_region', f'hill_count_{t}'] + CONTROLS].dropna()
    sub = sub.assign(_rb=_rank_resid(sub, f'hill_count_{t}', ['beta']),
                     _ra=_rank_resid(sub, f'hill_count_{t}', CONTROLS))
    g = [sub[sub._region == r] for r in sorted(sub._region.unique())]
    resid_by_thr[t] = (sorted(sub._region.unique()), [x['_ra'].to_numpy() for x in g])
    print(f'{t:5d} {len(sub):5d} {_eps2([x[f"hill_count_{t}"] for x in g]):10.3f} '
          f'{_eps2([x["_rb"] for x in g]):11.3f} {_eps2([x["_ra"] for x in g]):10.3f}')

# beta_iqr cannot join CONTROLS above without dropping every single-window segment, which
# removes Hercules Dome outright and turns a three-group test into a two-group one. So it
# gets its own pass on the subset where it exists, reported separately and read with the
# smaller n and the missing region in mind.
if 'beta_iqr' in pool.columns and pool.beta_iqr.notna().any():
    WITH_IQR = CONTROLS + ['beta_iqr']
    MIN_GROUP = 10  # a region reduced to a handful of windows cannot carry a rank test
    iqr_pool = pool.dropna(subset=WITH_IQR)
    sizes = iqr_pool._region.value_counts()
    kept = sorted(sizes[sizes >= MIN_GROUP].index)
    thin = sorted(f'{r} ({sizes[r]})' for r in sizes[sizes < MIN_GROUP].index)
    iqr_pool = iqr_pool[iqr_pool._region.isin(kept)]
    print(f'\nSUPPLEMENTARY: same test with beta_iqr added to the controls')
    print(f'  {len(iqr_pool)} of {len(pool)} windows have beta_iqr; '
          f'{len(kept)} of {len(regions)} regions survive: {", ".join(kept)}')
    if thin:
        print(f'  dropped below {MIN_GROUP} windows: {", ".join(thin)}')
    if len(kept) >= 2:
        print(f"{'thr':>5s} {'n':>5s} {'hill':>10s} {'hill|all+iqr':>14s}")
        print('-' * 38)
        for t in HILL_SWEEP_THRESHOLDS:
            sub = iqr_pool[['_region', f'hill_count_{t}'] + WITH_IQR].dropna()
            sub = sub.assign(_r=_rank_resid(sub, f'hill_count_{t}', WITH_IQR))
            g = [sub[sub._region == r] for r in sorted(sub._region.unique())]
            print(f'{t:5d} {len(sub):5d} '
                  f'{_eps2([x[f"hill_count_{t}"] for x in g]):10.3f} '
                  f'{_eps2([x["_r"] for x in g]):14.3f}')
    else:
        print('  too few regions left to compare')

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
bins = np.arange(-0.5, CEILING + 2.5)
for t in HILL_SWEEP_THRESHOLDS:
    allc = np.concatenate([d[f'hill_count_{t}'].dropna().values for d in regions.values()])
    ax.hist(allc, bins=bins, histtype='step', lw=1.8, label=f'{t} m')
ax.axvline(CEILING, color='red', ls='--', lw=1.2)
ax.annotate('ceiling', xy=(CEILING, 0), xytext=(CEILING - 1.6, ax.get_ylim()[1] * 0.9),
            fontsize=9, color='red')
ax.set_xlabel('hill count per window')
ax.set_ylabel('windows')
ax.set_title('Count distribution by relief threshold')
ax.legend(title='threshold')

ax = axes[1]
grid = corr.groupby('threshold')[AGAINST].apply(lambda g: g.abs().max())
im = ax.imshow(grid.values, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(len(AGAINST)))
ax.set_xticklabels(AGAINST, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(grid)))
ax.set_yticklabels([f'{t} m' for t in grid.index])
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        v = grid.values[i, j]
        if np.isfinite(v):
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8)
ax.set_title('worst |Spearman| across regions')
fig.colorbar(im, ax=ax, label='|rho|')
fig.tight_layout()
fig.savefig(OUT / 'hill_count_thresholds.png', dpi=150)

# Beta-residual by region. Boxes sitting on top of each other mean the count says nothing
# about landscape that beta has not already said.
fig2, axes2 = plt.subplots(1, len(HILL_SWEEP_THRESHOLDS),
                           figsize=(4 * len(HILL_SWEEP_THRESHOLDS), 4.5), sharey=True)
axes2 = np.atleast_1d(axes2)
for ax2, t in zip(axes2, HILL_SWEEP_THRESHOLDS):
    labels, groups = resid_by_thr[t]
    ax2.boxplot(groups, patch_artist=True, showfliers=False, widths=0.6,
                medianprops=dict(color='black', lw=2))
    ax2.axhline(0, color='red', ls='--', lw=1.2)
    ax2.set_xticklabels([l.split('/')[-1][:14] for l in labels], rotation=45,
                        ha='right', fontsize=8)
    ax2.set_title(f'{t} m')
axes2[0].set_ylabel('hill count rank, existing elements removed')
fig2.suptitle('Hill count with beta, relief, rms and amplitude removed, by region')
fig2.tight_layout()
fig2.savefig(OUT / 'hill_count_beta_residual.png', dpi=150)

print(f'\nSaved: {OUT / "hill_count_thresholds.png"}')
print(f'Saved: {OUT / "hill_count_beta_residual.png"}')
print(f'Saved: {OUT / "hill_count_correlations.csv"}')

sys.stdout = sys.__stdout__
(OUT / 'hill_count_threshold_sensitivity.log').write_text(_buf.getvalue())
print(f'Log written to {OUT / "hill_count_threshold_sensitivity.log"}')
