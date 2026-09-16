"""Data-intrinsic RES coverage tags per region.

No region boxes/polygons: every metric derives from the track geometry itself,
referenced to the survey's own footprint (convex hull of the track points).
Generalises to any RES track bundle keyed on `trajectory`.

Inputs (written by bed_analysis_23.py):
  coverage_csvs/<region>_track_points.csv  -> x, y, trajectory_id (~1 km cloud)
  window_csvs/<region>_window_stats.csv    -> center_x, center_y, azimuth_deg, is_transition, beta

Output:
  coverage_csvs/coverage_summary.csv  (one row per region) + log table.
  gap_* is the footprint-cell distance to the nearest track point; analysed_gap_* is
  the same to the nearest beta-bearing window centre, so analysed_gap_excess_p90_km
  (analysed minus track p90) flags area that was flown but yielded no usable beta.

Usage:
  python coverage_tags.py            # all regions found
  python coverage_tags.py Hercules   # partial-match filter
"""
import os, sys, glob
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from config import Tee, processing_flag_of
from loading import OUTPUT_BASE_PATH

# --- tier thresholds (placed in the empirical gaps of the region set) ---
GAP_GOOD_KM, GAP_POOR_KM = 20.0, 45.0   # 90th-pct interpolation distance; breaks at 17.5|24 and 29|66
R_GOOD, R_POOR           = 0.40, 0.80   # axial azimuth concentration (0=isotropic,1=uni-directional); breaks at 0.35|0.46 and 0.63|0.96
N_GOOD, N_POOR           = 80, 25       # homogeneous windows entering beta; breaks at 21|30 and 52|110
GRID_KM                  = 5.0          # interpolation-distance sampling grid


def _axial_R(az_deg):
    """Axial (0-180) directional concentration: |mean exp(i*2theta)|. ~1 uni-directional."""
    a = np.asarray(az_deg, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    return float(np.abs(np.mean(np.exp(1j * np.deg2rad(2 * a)))))


def _hull_area_km2(pts):
    if len(pts) < 3:
        return np.nan
    try:
        return ConvexHull(pts).volume / 1e6  # 2-D: .volume is the area
    except Exception:
        return np.nan


def _footprint_cells(x, y, grid_km=GRID_KM):
    """Grid cells inside the track-point convex hull (the survey footprint)."""
    pts = np.column_stack([x, y])
    if len(pts) < 4:
        return None
    try:
        tri = Delaunay(pts)
    except Exception:
        return None
    g = grid_km * 1000
    gx, gy = np.meshgrid(np.arange(x.min(), x.max() + g, g),
                         np.arange(y.min(), y.max() + g, g))
    cells = np.column_stack([gx.ravel(), gy.ravel()])
    cells = cells[tri.find_simplex(cells) >= 0]   # keep inside footprint hull
    return cells if len(cells) else None


def _dist_stats(cells, targets, prefix):
    """Median/p90/max distance (km) from each footprint cell to its nearest target."""
    if cells is None or len(targets) < 1:
        return {}
    d = cKDTree(np.asarray(targets)).query(cells)[0] / 1000
    return {f'{prefix}median_km': float(np.median(d)),
            f'{prefix}p90_km': float(np.percentile(d, 90)),
            f'{prefix}max_km': float(d.max())}


def _line_spacing_km(df):
    """Median over trajectories of each track's typical distance to the nearest other track."""
    trajs = df['trajectory_id'].unique()
    if len(trajs) < 2:
        return np.nan
    pts = df[['x', 'y']].values
    lab = df['trajectory_id'].values
    meds = []
    for t in trajs:
        m = lab == t
        other = pts[~m]
        if len(other) == 0 or m.sum() == 0:
            continue
        meds.append(np.median(cKDTree(other).query(pts[m])[0]) / 1000)
    return float(np.median(meds)) if meds else np.nan


def _grade(val, good, poor, higher_is_worse=True):
    if not np.isfinite(val):
        return 'NA'
    if higher_is_worse:
        return 'A' if val < good else ('C' if val > poor else 'B')
    return 'A' if val > good else ('C' if val < poor else 'B')


def coverage_for_region(track_csv, window_csv):
    pc = pd.read_csv(track_csv)
    x, y = pc['x'].values, pc['y'].values
    foot_area = _hull_area_km2(np.column_stack([x, y]))

    cells = _footprint_cells(x, y)
    res = {'n_track_points': len(pc),
           'n_trajectories': pc['trajectory_id'].nunique(),
           'footprint_km2': foot_area,
           'line_spacing_km': _line_spacing_km(pc)}
    res.update(_dist_stats(cells, np.column_stack([x, y]), 'gap_'))

    win = pd.read_csv(window_csv) if window_csv and os.path.exists(window_csv) else pd.DataFrame()
    res['processing_flag'] = processing_flag_of(win) if len(win) else None
    az_R = _axial_R(win['azimuth_deg']) if 'azimuth_deg' in win else np.nan
    res['azimuth_R'] = az_R

    homog = win[win.get('is_transition', False) == False] if len(win) else win
    homog = homog[np.isfinite(homog['beta'])] if 'beta' in homog else homog
    res['n_homog'] = len(homog)
    c = (homog[['center_x', 'center_y']].dropna().values
         if {'center_x', 'center_y'}.issubset(homog.columns) else np.empty((0, 2)))
    # fraction of footprint spanned by the homogeneous (beta-bearing) windows
    if len(c) >= 3 and np.isfinite(foot_area):
        res['beta_footprint_frac'] = _hull_area_km2(c) / foot_area if foot_area else np.nan
    else:
        res['beta_footprint_frac'] = np.nan
    # analysed-coverage gap: same footprint cells as gap_*, but distance to the nearest
    # beta-bearing window centre instead of any track point. Where analysed_gap_p90 far
    # exceeds gap_p90 the area was flown but no usable beta survived (transition-heavy or
    # fragmented terrain); the excess is that difference.
    if len(c) >= 1:
        res.update(_dist_stats(cells, c, 'analysed_gap_'))
        if 'gap_p90_km' in res and 'analysed_gap_p90_km' in res:
            res['analysed_gap_excess_p90_km'] = res['analysed_gap_p90_km'] - res['gap_p90_km']

    # --- tier: worst of the three axes, components reported ---
    g_gap = _grade(res.get('gap_p90_km', np.nan), GAP_GOOD_KM, GAP_POOR_KM, True)
    g_dir = _grade(az_R, R_GOOD, R_POOR, True)
    g_n   = _grade(res['n_homog'], N_GOOD, N_POOR, False)
    res['grade_gap'], res['grade_dir'], res['grade_n'] = g_gap, g_dir, g_n
    order = {'A': 0, 'B': 1, 'C': 2, 'NA': -1}
    grades = {'sparsity': g_gap, 'directionality': g_dir, 'sample': g_n}
    worst = max((order[g] for g in grades.values()), default=-1)
    res['tier'] = {0: 'A', 1: 'B', 2: 'C'}.get(worst, 'NA')
    res['tier_driver'] = ','.join(k for k, g in grades.items() if order[g] == worst and worst >= 1) or '-'
    return res


def discover(directory='.'):
    out = {}
    for f in glob.glob(os.path.join(directory, 'coverage_csvs', '*_track_points.csv')):
        region = os.path.basename(f).replace('_track_points.csv', '')
        win = os.path.join(directory, 'window_csvs', f'{region}_window_stats.csv')
        out[region] = {'track': f, 'window': win if os.path.exists(win) else None}
    return out


def run_all(directory=None, region_filter=None):
    directory = directory or OUTPUT_BASE_PATH
    regions = discover(directory)
    if region_filter:
        regions = {r: v for r, v in regions.items() if region_filter.lower() in r.lower()}
    if not regions:
        print(f"No *_track_points.csv found under {os.path.join(directory, 'coverage_csvs')} "
              f"(re-run bed_analysis_23.py to emit them).")
        return

    rows = []
    for region in sorted(regions):
        r = coverage_for_region(regions[region]['track'], regions[region]['window'])
        r['region'] = region
        rows.append(r)
        print(f"{region}: tier {r['tier']} ({r['tier_driver']}) | "
              f"gap_p90 {r.get('gap_p90_km', float('nan')):.0f} km | "
              f"analysed_gap_p90 {r.get('analysed_gap_p90_km', float('nan')):.0f} km "
              f"(+{r.get('analysed_gap_excess_p90_km', float('nan')):.0f}) | R {r['azimuth_R']:.2f} | "
              f"n_homog {r['n_homog']} | spacing {r['line_spacing_km']:.0f} km")

    cols = ['region', 'tier', 'tier_driver', 'processing_flag', 'n_homog', 'n_trajectories',
            'gap_median_km', 'gap_p90_km', 'gap_max_km',
            'analysed_gap_median_km', 'analysed_gap_p90_km', 'analysed_gap_max_km',
            'analysed_gap_excess_p90_km', 'line_spacing_km',
            'azimuth_R', 'footprint_km2', 'beta_footprint_frac',
            'grade_gap', 'grade_dir', 'grade_n', 'n_track_points']
    df = pd.DataFrame(rows).reindex(columns=cols)
    out = os.path.join(directory, 'coverage_csvs', 'coverage_summary.csv')
    df.to_csv(out, index=False)
    print(f"\nWrote {out} ({len(df)} regions)")
    return df


if __name__ == '__main__':
    rf = sys.argv[1] if len(sys.argv) > 1 else None
    sys.stdout = Tee(os.path.join(OUTPUT_BASE_PATH, 'coverage_tags_log.txt'))
    run_all(region_filter=rf)
