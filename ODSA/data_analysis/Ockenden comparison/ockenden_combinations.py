"""Do [Ockenden_2026]'s classes separate on combinations of vector elements?

Extends the OVERLAP test of ockenden_concordance.py from one element to every subset of the
four classifying axes (beta, relief, velocity, elevation), plus the seven descriptors
together and all eleven elements together. For a class pair, the separation uses the vector
of median differences, measured in units of the pooled covariance (d) and in units of the
covariance of the medians at the independent count (z). A pair separates when
z >= z_min(k) AND d >= D_MIN, for k elements. z_min(k) is the chi-square quantile with k
degrees of freedom at the coverage of Z_MIN on one element, so z_min(1) = Z_MIN and the
test on one element is the concordance test exactly. z_min grows with k, so adding an
element can lower the count of separated pairs.

Like the concordance, this is not a test of whether the axes recover her classes.

    python ockenden_combinations.py [root]

Needs ockenden_window_class.csv (run ockenden_class.py first). Writes
ockenden_combinations.csv and ockenden_combinations_log.txt into the run tree (ROOT), which
defaults to OUTPUT_BASE_PATH.
"""
import glob, itertools, os, sys
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import chi2, norm
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Tee
from loading import OUTPUT_BASE_PATH as _REGION_BASE
from ockenden_concordance import KEY, MIN_N, Z_MIN, D_MIN, n_independent

ROOT = sys.argv[1] if len(sys.argv) > 1 else _REGION_BASE

AXES = ['beta', 'relief_m', 'measures_speed_mean', 'bed_elev_mean']
SYMBOL = {'beta': 'B', 'relief_m': 'R', 'measures_speed_mean': 'V', 'bed_elev_mean': 'E'}
# Tested as one block, labelled D. hill_count_50 is her published gate, so it is left out.
DESCRIPTORS = ['A_1km', 'rms_roughness', 'eta_wavelength_m', 'hill_count', 'skewness',
               'kurtosis', 'xi_band']
SETS = ([list(c) for k in range(1, len(AXES) + 1) for c in itertools.combinations(AXES, k)]
        + [DESCRIPTORS, AXES + DESCRIPTORS])
Z_COVERAGE = 2 * norm.cdf(Z_MIN) - 1


def z_min(k):
    return float(np.sqrt(chi2.ppf(Z_COVERAGE, k)))


def load_windows(root):
    """Non-transition windows with an Ockenden class, the 650 of the concordance."""
    files = sorted(glob.glob(os.path.join(root, '*', 'window_csvs', '*_window_stats.csv')))
    assert files, f'no window CSVs under {root}'
    # Region keys the class join, since 16 flight lines cross more than one region box.
    d = pd.concat([pd.read_csv(f).assign(region=os.path.basename(os.path.dirname(os.path.dirname(f))))
                   for f in files], ignore_index=True)
    n = len(d)
    cls = pd.read_csv(os.path.join(root, 'ockenden_window_class.csv'))
    d = d.merge(cls[['region'] + KEY + ['ockenden_class', 'alt_agrees']],
                on=['region'] + KEY, how='left')
    assert len(d) == n, f'class join fanned {n} rows to {len(d)}'
    d = d[~d.is_transition.astype(bool) & d.ockenden_class.notna()
          & (d.ockenden_class != 'invalid_dunes')]
    print(f'{len(d)} windows from {root}')
    return d.copy()


def label(cols):
    return '+'.join([SYMBOL[a] for a in cols if a in SYMBOL]
                    + (['D'] if set(DESCRIPTORS) <= set(cols) else []))


def pairs(d, axes):
    """Every class pair on one subset of axes, over windows that carry all of them."""
    x = d.dropna(subset=list(axes))
    stat = {}
    for c, g in x.groupby('ockenden_class'):
        if len(g) < MIN_N:
            continue
        v = g[list(axes)].values
        stat[c] = (np.median(v, axis=0), np.atleast_2d(np.cov(v, rowvar=False)), len(g),
                   n_independent(g))
    out = []
    for a, b in [(a, b) for i, a in enumerate(stat) for b in list(stat)[i + 1:]]:
        (ma, ca, na, ea), (mb, cb, nb, eb) = stat[a], stat[b]
        diff = ma - mb
        se = ca / ea + cb / eb
        pooled = ((na - 1) * ca + (nb - 1) * cb) / (na + nb - 2)
        z = np.sqrt(diff @ np.linalg.solve(se, diff))
        dd = np.sqrt(diff @ np.linalg.solve(pooled, diff))
        out.append({'axes': label(axes), 'n_axes': len(axes), 'a': a, 'b': b,
                    'n_a': na, 'n_b': nb, 'n_independent_a': ea, 'n_independent_b': eb,
                    'z': z, 'z_min': z_min(len(axes)), 'd': dd,
                    'separates': z >= z_min(len(axes)) and dd >= D_MIN})
    return out


def run(d, subset):
    res = pd.DataFrame([r for cols in SETS for r in pairs(d, cols)])
    res['subset'] = subset
    summary = (res.groupby(['n_axes', 'axes'])
               .agg(pairs_separated=('separates', 'sum'), pairs_tested=('separates', 'size'))
               .reset_index().sort_values(['n_axes', 'pairs_separated'],
                                          ascending=[True, False]))
    print(f"\n{'=' * 78}\n{subset}: {len(d)} windows\n{'=' * 78}")
    print("pairs separated per combination (B beta, R relief, V velocity, E elevation, "
          "D all seven descriptors):")
    print(summary.to_string(index=False))
    best = summary.groupby('n_axes').head(1)
    print("\nbest combination at each size:")
    print(best.to_string(index=False))
    return res


def check_single_axes(res, root):
    """The one-axis rows must reproduce ockenden_concordance.csv."""
    path = os.path.join(root, 'ockenden_concordance.csv')
    if not os.path.exists(path):
        print(f"\n{path} not found, single-axis check skipped")
        return
    c = pd.read_csv(path)
    c['axes'] = c.element.map(SYMBOL)
    one = res[res.n_axes == 1]
    m = one.merge(c.dropna(subset=['axes']), on=['subset', 'axes', 'a', 'b'],
                  suffixes=('', '_conc'))
    bad = int((m.separates != m.separates_conc).sum())
    print(f"\nsingle-axis check against ockenden_concordance.csv: {len(m)} of {len(one)} "
          f"rows matched, {bad} disagree on separates, max|dz| "
          f"{(m.z - m.z_conc.abs()).abs().max():.2e}")


if __name__ == '__main__':
    sys.stdout = Tee(os.path.join(ROOT, 'ockenden_combinations_log.txt'))
    d = load_windows(ROOT)
    res = pd.concat([run(d, 'all'), run(d[d.alt_agrees.astype(bool)], 'non_straddling')],
                    ignore_index=True)
    check_single_axes(res, ROOT)
    out = os.path.join(ROOT, 'ockenden_combinations.csv')
    res.to_csv(out, index=False)
    print(f"Wrote {out}")
