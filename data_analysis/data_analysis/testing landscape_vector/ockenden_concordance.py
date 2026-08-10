"""Queue item 5b Test A: how does the vector relate to [Ockenden_2026]'s classification?

NOT a test of whether the vector recovers her classes. It should not — different method,
different data, and if a transect on point measurements did reproduce an areal classification
built by interpolation, the co-located result would be in trouble. Three questions instead:

  1. OVERLAP     do her classes separate at all on my elements? (z >= 2 AND d >= 1, the
                 two-scale rule, spread from every window and n_eff only shrinking the error)
  2. EXPLAINED   how much of each element's variance does her classification account for?
                 Kruskal-Wallis epsilon-squared, the same statistic as the hill_count doc.
  3. RESOLVED    how much of it sits INSIDE one 50 km cell, where her method has exactly one
                 value by construction? This is what a transect adds to an areal map.

Read 3 against the borrowed axes as a control: bed elevation and velocity are smooth gridded
products and should hold little within-cell variance, while the fast-decorrelating elements
ODSA measures should hold a lot. That contrast is the claim.

    python ockenden_concordance.py [root]

Needs ockenden_window_class.csv (run ockenden_class.py first). Writes
ockenden_concordance.csv and ockenden_variance.csv to the tree root.
"""
import glob, os, sys
import numpy as np, pandas as pd
from config import Tee
from scipy.stats import kruskal
from landscape_vector import _independent_subset, COMPOSITION_DECIMATE_KM

ROOT = sys.argv[1] if len(sys.argv) > 1 else 'individual_region_TEST'
SNAP = 'v23/hill_count_threshold/ODSA Ockenden-regions_skew/window_csvs'
KEY = ['trajectory', 'segment', 'window_id']
MIN_N = 10          # classes thinner than this are reported but not tested
Z_MIN, D_MIN = 2.0, 1.0

# hill_count_50 is her published gate; 20 m is production. Both are hers, so report both.
MEASURED = ['beta', 'psd_amplitude_1km', 'rms_roughness', 'eta_wavelength_m', 'hill_count',
            'hill_count_50', 'skewness', 'kurtosis', 'xi_band', 'relief_m']
BORROWED = ['bed_elev_mean', 'measures_speed_mean']

# Two of her five classes are not positively defined. Say so wherever they are quoted.
SOFT = {'sel_erosion_relict': 'RESIDUAL — none-of-the-above in her mask chain',
        'low_relief': 'poordetail mask — low curvature OR few hills OR low slope'}


def load(root):
    # 16 flight lines cross more than one region box, so trajectory/segment/window is NOT
    # unique on its own. Region keys the class join; source file keys the snapshot join,
    # the same alignment the hill_count additive check uses.
    frames = []
    for f in sorted(glob.glob(os.path.join(root, '*', 'window_csvs', '*_window_stats.csv'))):
        x = pd.read_csv(f)
        x['region'] = os.path.basename(os.path.dirname(os.path.dirname(f)))
        x['src'] = os.path.basename(f)
        frames.append(x)
    d = pd.concat(frames, ignore_index=True)
    n = len(d)

    cls = pd.read_csv('ockenden_window_class.csv')
    need = ['ockenden_class', 'alt_agrees', 'cell_id']
    assert 'cell_id' in cls, 'stale ockenden_window_class.csv — re-run ockenden_class.py'
    d = d.merge(cls[['region'] + KEY + need], on=['region'] + KEY, how='left')
    assert len(d) == n, f'class join fanned {n} rows to {len(d)}'

    # The frozen snapshot is the only run carrying the other three gates.
    snap_files = sorted(glob.glob(os.path.join(SNAP, '*_window_stats.csv')))
    assert snap_files, f'frozen hill-count snapshot not found at {SNAP}'
    snap = pd.concat([pd.read_csv(f)[KEY + ['hill_count_50']].assign(src=os.path.basename(f))
                      for f in snap_files], ignore_index=True)
    d = d.merge(snap, on=['src'] + KEY, how='left')
    assert len(d) == n, f'snapshot join fanned {n} rows to {len(d)}'

    d = d[~d.is_transition & d.ockenden_class.notna() & (d.ockenden_class != 'invalid_dunes')]
    print(f"{n} windows in, {len(d)} classified non-transition, "
          f"hill_count_50 present on {d.hill_count_50.notna().sum()}")
    return d.copy()


def cols(d):
    return [c for c in MEASURED + BORROWED if c in d]


def n_eff(g):
    """Independent windows in this class, greedy at the tuple decorrelation distance."""
    return max(len(_independent_subset(g[['center_x', 'center_y']].values, COMPOSITION_DECIMATE_KM)), 1)


# --- 1. OVERLAP ------------------------------------------------------------------------
def pairs(d, col):
    """Every class pair on one element, z shrunk by n_eff and d on the pooled spread."""
    out, stat = [], {}
    for c, g in d.groupby('ockenden_class'):
        v = g[col].dropna()
        if len(v) < MIN_N:
            continue
        stat[c] = (v.median(), v.std(ddof=1), len(v), n_eff(g.loc[v.index]))
    for a, b in [(a, b) for i, a in enumerate(stat) for b in list(stat)[i + 1:]]:
        (ma, sa, na, ea), (mb, sb, nb, eb) = stat[a], stat[b]
        se = np.hypot(sa / np.sqrt(ea), sb / np.sqrt(eb))
        pooled = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
        z = (ma - mb) / se if se else np.nan
        dd = (ma - mb) / pooled if pooled else np.nan
        out.append({'element': col, 'a': a, 'b': b, 'median_a': ma, 'median_b': mb,
                    'n_a': na, 'n_b': nb, 'n_eff_a': ea, 'n_eff_b': eb,
                    'z': z, 'd': dd, 'separates': abs(z) >= Z_MIN and abs(dd) >= D_MIN})
    return out


# --- 2. EXPLAINED / 3. RESOLVED --------------------------------------------------------
def eps2(d, col, by='ockenden_class'):
    """Share of the element's rank variance the grouping explains. Descriptive, not a test."""
    g = [v[col].dropna().values for _, v in d.groupby(by) if v[col].notna().sum() >= MIN_N]
    if len(g) < 2:
        return np.nan
    n, k = sum(len(x) for x in g), len(g)
    return (kruskal(*g).statistic - k + 1) / (n - k)


def within_cell(d, col):
    """Variance share sitting inside one 50 km cell, where her classification has one value."""
    x = d.dropna(subset=[col, 'cell_id'])
    x = x[x.groupby('cell_id')[col].transform('size') >= 2]
    if len(x) < MIN_N or x[col].var(ddof=1) == 0:
        return np.nan, 0, np.nan
    within = x.groupby('cell_id')[col].var(ddof=1).mean()
    return within / x[col].var(ddof=1), x.cell_id.nunique(), len(x) / x.cell_id.nunique()


def variance_table(d):
    rows = []
    for c in cols(d):
        frac, ncell, per = within_cell(d, c)
        rows.append({'element': c, 'family': 'borrowed' if c in BORROWED else 'measured',
                     'eps2_class': eps2(d, c), 'within_cell_frac': frac,
                     'n_cells': ncell, 'windows_per_cell': per})
    return pd.DataFrame(rows).sort_values('within_cell_frac', ascending=False)


def run(d, label):
    print(f"\n{'=' * 78}\n{label}: {len(d)} windows, {d.ockenden_class.nunique()} classes\n{'=' * 78}")
    print(d.ockenden_class.value_counts().to_string())
    for c, why in SOFT.items():
        if c in d.ockenden_class.values:
            print(f"  ! {c}: {why}")
    print("\nmedians by class:")
    print(d.groupby('ockenden_class')[cols(d)].median().T.round(2).to_string())

    res = pd.DataFrame([r for c in cols(d) for r in pairs(d, c)])
    rank = res[res.separates].groupby('element').size().rename('pairs_separated')
    tot = res.groupby('element').size().rename('pairs_tested')
    print("\n1. OVERLAP — class pairs separated (z>=2 AND d>=1):")
    print(pd.concat([rank, tot], axis=1).fillna(0).astype(int)
          .sort_values('pairs_separated', ascending=False).to_string())
    fail = res[~res.separates]
    print(f"   of {len(fail)} non-separating pairs, {((fail.z.abs() < Z_MIN) & (fail.d.abs() < D_MIN)).sum()}"
          f" fail BOTH criteria — overlap, not precision")

    var = variance_table(d)
    print("\n2. EXPLAINED by her classes / 3. RESOLVED inside one cell:")
    print(var.round(3).to_string(index=False))
    return res, var


if __name__ == '__main__':
    sys.stdout = Tee('ockenden_concordance_log.txt')
    d = load(ROOT)
    res, var = run(d, 'ALL SNAPPED WINDOWS')
    res2, var2 = run(d[d.alt_agrees], 'NON-STRADDLING ONLY')
    res['subset'], res2['subset'] = 'all', 'non_straddling'
    var['subset'], var2['subset'] = 'all', 'non_straddling'
    pd.concat([res, res2], ignore_index=True).to_csv('ockenden_concordance.csv', index=False)
    pd.concat([var, var2], ignore_index=True).to_csv('ockenden_variance.csv', index=False)
    print("\nRead the within-cell column against family: borrowed axes are smooth gridded")
    print("products and should sit low; the elements ODSA measures should sit high.")
    print("Wrote ockenden_concordance.csv, ockenden_variance.csv")
