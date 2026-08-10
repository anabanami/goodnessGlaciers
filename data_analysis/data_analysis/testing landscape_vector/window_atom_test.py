"""Queue item 4: is a 50%-overlapping window a legitimate classification atom?

Measures how fast the class tuple (beta, relief, elevation, velocity) decorrelates with
distance. Agreement at chance = independent. Gives the decimation distance for item 5.
"""
import glob, os, sys
import numpy as np, pandas as pd
from config import Tee
from landscape_vector import VELOCITY_CLASSES

ROOT = sys.argv[1] if len(sys.argv) > 1 else 'individual_region_TEST'
AXES = ['bed_class', 'relief_class', 'elevation_class', 'velocity_band']
BINS = [0, 1, 25, 50, 75, 100, 150, 200, 300, 1e9]


def band(v):
    for n, lo, hi in VELOCITY_CLASSES:
        if lo <= v < hi:
            return n
    return 'na'


def load(root):
    out = []
    for f in sorted(glob.glob(os.path.join(root, '*', 'window_csvs', '*_window_stats.csv'))):
        d = pd.read_csv(f)
        d['region'] = os.path.basename(os.path.dirname(os.path.dirname(f)))
        out.append(d)
    d = pd.concat(out, ignore_index=True)
    d = d[~d.is_transition].copy()
    d['velocity_band'] = d.measures_speed_mean.map(band)
    return d.dropna(subset=AXES)


def pairs(d, within_traj):
    """Agreement per axis and on the full tuple, binned by centre separation."""
    rows = []
    for _, g in d.groupby(['region', 'trajectory'] if within_traj else 'region'):
        x, y = g.center_x.values, g.center_y.values
        i, j = np.triu_indices(len(g), k=1)
        dist = np.hypot(x[i] - x[j], y[i] - y[j]) / 1e3
        rec = {'dist': dist}
        for a in AXES:
            v = g[a].values
            rec[a] = (v[i] == v[j])
        rec['tuple'] = np.all([rec[a] for a in AXES], axis=0)
        rows.append(pd.DataFrame(rec))
    p = pd.concat(rows, ignore_index=True)
    p['bin'] = pd.cut(p.dist, BINS, right=False)
    return p.groupby('bin', observed=True).agg(
        n=('dist', 'size'), **{a: (a, 'mean') for a in AXES + ['tuple']}).round(3)


if __name__ == '__main__':
    sys.stdout = Tee(os.path.join(ROOT, 'window_atom_test_log.txt'))
    d = load(ROOT)
    print(f"{len(d)} non-transition windows, {d.region.nunique()} regions\n")
    # Chance level: tuple agreement for randomly paired windows.
    s = d.sample(frac=1, random_state=0)
    hit = np.all([(d[a].values == s[a].values) for a in AXES], axis=0)
    print(f"chance tuple agreement (shuffled): {hit.mean():.3f}\n")
    for label, wt in [('WITHIN TRAJECTORY', True), ('ALL PAIRS IN REGION', False)]:
        print(f"=== {label} (agreement by centre separation, km) ===")
        print(pairs(d, wt).to_string(), "\n")
