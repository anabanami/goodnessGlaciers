"""Per-window Ockenden landscape class, from the published metrics rather than from prose.

Reproduces the [Ockenden_2026] classification masks off the Zenodo NetCDFs
[Ockenden_2025_data] — the same logic as map_flightlines.py, itself a port of
Antarctic_FIGURES.ipynb cell 45 — and snaps each production window to its cell.
Her grid is 50 km and the window is 50 km, so this is a matched-scale join.

Supersedes the region-level strings in all_data/Ockenden/scan_all_datasets.py and the
eyeball reading in all_data/Ockenden/regions.md. Neither is authoritative; this is.

    python ockenden_class.py [root] [--metrics DIR]

Writes ockenden_window_class.csv to the tree root. Reads only.
"""
import glob, os, sys
import numpy as np, pandas as pd
from config import Tee
from netCDF4 import Dataset
from scipy.spatial import cKDTree

ROOT = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else 'individual_region_TEST'
METRICS = 'all_data/Ockenden/Data_Science_Zenodo/Data_Science_Zenodo/Metrics/'
if '--metrics' in sys.argv:
    METRICS = sys.argv[sys.argv.index('--metrics') + 1]

CELL_M = 50_000                      # Ockenden metric grid spacing, PS71 metres
SNAP_MAX_M = CELL_M * np.sqrt(2) / 2  # half-diagonal: the furthest an in-grid point can sit


def masks(metrics_dir=METRICS):
    """Cell centres and class label, verbatim from Antarctic_FIGURES.ipynb cell 45."""
    def m(name):
        ds = Dataset(os.path.join(metrics_dir, name + '.nc'))
        d = ds.variables['data'][:].data
        ds.close()
        return d

    x, y = m('X_ifpa'), m('Y_ifpa')
    i_rms_slope_h, ifpa_count_250, ifpa_mean = m('i_rms_slope_h'), m('ifpa_count_max_250'), m('ifpa_mean')
    i_std_l, ifpa_b1_thick, ifpa_rms_slope = m('i_std_l'), m('ifpa_b1_thickness'), m('ifpa_rms_slope')
    ifpa_rms_curv, ifpa_count_20 = m('ifpa_rms_curvature'), m('ifpa_count_max_20')
    ifpa_count_100, ifpa_wav_max = m('ifpa_count_max_100'), m('ifpa_wav_max_power')

    mountain = (i_rms_slope_h > 2) | (ifpa_count_250 > 10)
    sgm = (~mountain) & ((ifpa_mean > 1000) | (i_std_l > 19))
    sgm2 = (~mountain) & (~sgm) & (ifpa_b1_thick > -5.0) & (ifpa_rms_slope < 1.1) & (ifpa_mean > 500)
    poor = (~mountain) & (~sgm) & (~sgm2) & \
           ((ifpa_rms_curv < 0.025) | (ifpa_count_20 < 15) | (i_rms_slope_h < 0.07))
    dunes = (~poor) & (~mountain) & (~sgm) & (~sgm2) & \
            ((ifpa_rms_slope / ifpa_rms_curv) < 14.75) & (ifpa_rms_slope < 0.9) & \
            (ifpa_wav_max < 5000) & (ifpa_count_100 == 0)
    ice = (~mountain) & (~poor) & (~dunes) & (~sgm) & (~sgm2) & \
          (ifpa_b1_thick < -5.5) & (ifpa_rms_slope > 1.0)
    ice2 = (~mountain) & (~poor) & (~dunes) & (~sgm) & (~sgm2) & (~ice)

    named = [(poor, 'low_relief'), (sgm | sgm2, 'alpine_subglacial'), (mountain, 'alpine_subaerial'),
             (ice, 'sel_erosion_icestreams'), (ice2, 'sel_erosion_relict'), (dunes, 'invalid_dunes')]
    stack = np.array([k for k, _ in named])
    assert stack.sum(0).max() <= 1, "masks overlap; draw order would decide the class"
    lab = np.full(len(x), 'unclassified', dtype=object)
    for k, name in named:
        lab[k] = name
    return x, y, lab


def windows(root):
    out = []
    for f in sorted(glob.glob(os.path.join(root, '*', 'window_csvs', '*_window_stats.csv'))):
        d = pd.read_csv(f)
        d['region'] = os.path.basename(os.path.dirname(os.path.dirname(f)))
        out.append(d[['region', 'trajectory', 'segment', 'window_id',
                      'center_x', 'center_y', 'is_transition']])
    return pd.concat(out, ignore_index=True)


def classify(root=ROOT, metrics_dir=METRICS):
    x, y, lab = masks(metrics_dir)
    w = windows(root)
    dist, idx = cKDTree(np.c_[x, y]).query(np.c_[w.center_x, w.center_y], k=2)
    w['ockenden_class'] = np.where(dist[:, 0] <= SNAP_MAX_M, lab[idx[:, 0]], None)
    w['cell_id'] = idx[:, 0]   # her classification has ONE value per cell; mine can vary inside it
    w['snap_km'] = (dist[:, 0] / 1e3).round(2)
    # The window is one cell wide, so it can straddle. Second cell says whether it does.
    w['alt_class'] = lab[idx[:, 1]]
    w['alt_agrees'] = w.alt_class == w.ockenden_class
    return w


if __name__ == '__main__':
    sys.stdout = Tee('ockenden_class_log.txt')
    w = classify()
    w.to_csv('ockenden_window_class.csv', index=False)
    live = w[~w.is_transition]
    print(f"{len(w)} windows, {len(live)} non-transition | grid {len(masks()[0])} cells @ {CELL_M/1e3:.0f} km")
    print(f"unsnapped (>{SNAP_MAX_M/1e3:.1f} km from any cell): {w.ockenden_class.isna().sum()}")
    print(f"snap distance km: median {w.snap_km.median():.1f}, p95 {w.snap_km.quantile(.95):.1f}")
    print(f"straddling (2nd cell disagrees): {(~live.alt_agrees).mean():.1%}\n")
    print(pd.crosstab(live.region, live.ockenden_class).to_string(), "\n")
    print("PPB, the active/relict question:")
    print(live[live.region == 'PPB'].ockenden_class.value_counts().to_string())
