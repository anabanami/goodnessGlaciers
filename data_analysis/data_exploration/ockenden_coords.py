"""
Subset Bedmap radar data to match Ockenden et al. (2025) figure regions.

Uses the EXACT PS71 bounding boxes from the Zenodo code repository
(Antarctic_FIGURES.ipynb, bounds2[0..8]) instead of approximate lat/lon
guesses. Filtering is done in PS71 space to avoid polar lat/lon distortion.

Usage:
    python ockenden_coords.py
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
from pyproj import Transformer

BASE_DIR = '/home/ana/Desktop/code/Data/ODSA/all_data/bedmap3_data/bedmap3/Results/'
METRICS_DIR = '/home/ana/Desktop/code/Data/ODSA/all_data/Ockenden/Data_Science_Zenodo/Data_Science_Zenodo/Metrics_v2/'

# WGS84 <-> EPSG:3031 (Antarctic Polar Stereographic, PS71)
to_ps71 = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)

# --- PPB core selection: single source of truth. Referenced by the region dict
# ('ps71') and by BOTH subset twins (_ppb_core_subset, ppb_core_subset). Change
# the box or the spur trim here, once. ---
PPB_CORE_BOX = [-0.247e6, 0.340e6, -0.398e6, 0.280e6]
PPB_SPUR_LEGS = ['P33.1', 'P33.3']   # the two lone southern rays (POLARGAP flight P33)
PPB_SPUR_LAT_CUT = -88.5             # drop spur points with lat > this (|lat| < 88.5)

# ---------------------------------------------------------------------------
# Exact PS71 bounds from Antarctic_FIGURES.ipynb  (bounds2, second block)
#   format: [x_min, x_max, y_min, y_max] in metres
# ---------------------------------------------------------------------------
# Ordered to match loading.py's target_files. Each region carries its source
# 'file' + 'dataset_label' (the label prefix), so the loop can run in this order
# while reading each CSV only once (see file_cache in main()).
# Dropped vs the original 9: Fig2F_Resolution_SH and Fig2B_Wilhelm_II — both sit
# in ~zero alpine-class cells (badly classified).
OCKENDEN_REGIONS = {
    'Pensacola_Pole': {
        'file': 'BAS_2015_POLARGAP_AIR_BM3.csv',
        'dataset_label': 'POLARGAP_2015',
        'ps71': PPB_CORE_BOX,
        'description': 'Pensacola-Pole Basin -- core square + P33 spur trim (see Ockenden/ppb/)',
        'ockenden_class': 'selective erosion',
        'fig': 'Fig 1B-D',
        'core_subset': True,  # bespoke core-square + P33 trim, not a plain box
        'loading_subset_repr': "'subset': _ppb_core_subset,",  # matches loading.py
    },
    'Fig4C_Aurora_SB_lowrelief': {
        'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
        'dataset_label': 'ASB_ICECAP_2010',
        'ps71': [1.05e6, 2.20e6, -0.80e6, 0.20e6],
        'description': 'Aurora SB -- low-relief cells only (Ockenden metrics: hill50<=5, relief<=500m)',
        'ockenden_class': 'low-relief',
        'fig': 'Fig 4 classification region (filtered)',
        'cell_mask': True,  # flag: use Ockenden metric grid instead of simple box
        'loading_subset_repr': "'subset': lambda df: _ps71_lowrelief_subset(df, [1.05e6, 2.20e6, -0.80e6, 0.20e6]),",
    },
    'Fig4C_Aurora_SB_square': {
        'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
        'dataset_label': 'ASB_ICECAP_2010',
        'ps71': [1.10e6, 1.40e6, -0.78e6, -0.48e6],
        'description': ('Aurora SB -- unmasked 300x300 km square (same box size as '
                        'Golicyna/Highland A), sited on the SW low-relief cluster of the '
                        'Fig4C bounds. NOT filtered to low-relief cells: ~60% of the 50 km '
                        'cells are low-relief, the rest are whatever else is there.'),
        'ockenden_class': 'low-relief (mixed)',
        'fig': 'Fig 4 classification region (unfiltered square)',
    },
    'Fig2A_Maud_SB': {
        'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
        'dataset_label': 'ASB_ICECAP_2010',
        'ps71': [0.15e6, 0.45e6, 1.025e6, 1.325e6],
        'description': 'Maud Subglacial Basin -- 400 km incised channel',
        'ockenden_class': 'low-relief / selective erosion',
        'fig': 'Fig 2A',
    },
    'Fig2D_Recovery_SL': {
        'file': 'BAS_2012_ICEGRAV_AIR_BM3.csv',
        'dataset_label': 'Rec_Catch',
        'ps71': [0.0e6, 0.30e6, 0.6e6, 0.9e6],
        'description': 'Recovery Subglacial Lakes -- geological boundary',
        'ockenden_class': 'low-relief / selective erosion',
        'fig': 'Fig 2D',
    },
    'Fig2G_Highland_A': {
        'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
        'dataset_label': 'ASB_ICECAP_2010',
        'ps71': [1.90e6, 2.20e6, -0.725e6, -0.425e6],
        'description': 'Highland A -- paleo-river landscape',
        'ockenden_class': 'alpine / selective erosion / low-relief',
        'fig': 'Fig 2G',
    },
    'Fig2H_Golicyna_SM': {
        'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
        'dataset_label': 'ASB_ICECAP_2010',
        'ps71': [2.15e6, 2.45e6, -0.5e6, -0.2e6],
        'description': 'Golicyna Subglacial Mountains -- dendritic valleys',
        'ockenden_class': 'alpine / selective erosion / low-relief',
        'fig': 'Fig 2H',
    },
    'Fig2C_Hercules_Dome': {
        'file': 'BAS_2015_POLARGAP_AIR_BM3.csv',
        'dataset_label': 'POLARGAP_2015',
        'ps71': [-0.6e6, -0.3e6, -0.23e6, 0.07e6],
        'description': 'Hercules Dome -- U-shaped valleys',
        'ockenden_class': 'alpine / selective erosion',
        'fig': 'Fig 2C',
    },
}


def load_lowrelief_cells(ps71_bounds, hill_thresh=5, relief_thresh=500):
    """Return (x, y) arrays of Ockenden grid cell centers classified as low-relief
    within a PS71 bounding box. Uses Ockenden Metrics_v2 grids (50 km cells)."""
    x_grid = nc.Dataset(METRICS_DIR + 'x_ifpa.nc')['data'][:].data
    y_grid = nc.Dataset(METRICS_DIR + 'y_ifpa.nc')['data'][:].data
    relief = nc.Dataset(METRICS_DIR + 'ifpa_relief.nc')['data'][:].data
    hill50 = nc.Dataset(METRICS_DIR + 'ifpa_count_max_50.nc')['data'][:].data

    xmin, xmax, ymin, ymax = ps71_bounds
    mask = ((x_grid >= xmin) & (x_grid <= xmax) &
            (y_grid >= ymin) & (y_grid <= ymax) &
            (hill50 <= hill_thresh) & (relief <= relief_thresh))
    return x_grid[mask], y_grid[mask]


def cell_mask_subset(df, ps71_bounds, half_cell=25000):
    """Filter RES points to those falling within low-relief Ockenden grid cells."""
    lr_x, lr_y = load_lowrelief_cells(ps71_bounds)
    hit = np.zeros(len(df), dtype=bool)
    for cx, cy in zip(lr_x, lr_y):
        hit |= (np.abs(df['x_ps71'].values - cx) <= half_cell) & \
               (np.abs(df['y_ps71'].values - cy) <= half_cell)
    return df[hit].copy()


def load_bedmap_csv(filepath):
    df = pd.read_csv(filepath, comment='#', header=0, low_memory=False)
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if 'latitude' in cl:
            col_map[c] = 'lat'
        elif 'longitude' in cl:
            col_map[c] = 'lon'
        elif 'bedrock' in cl:
            col_map[c] = 'bed'
        elif 'trajectory' in cl:
            col_map[c] = 'traj_id'
    return df.rename(columns=col_map)


def add_ps71(df):
    """Add x_ps71, y_ps71 columns (metres) from lat/lon."""
    x, y = to_ps71.transform(df['lon'].values, df['lat'].values)
    df['x_ps71'] = x
    df['y_ps71'] = y
    return df


def spatial_subset_ps71(df, ps71_bounds):
    """Filter in PS71 space. bounds = [x_min, x_max, y_min, y_max]."""
    xmin, xmax, ymin, ymax = ps71_bounds
    return df[
        (df['x_ps71'] >= xmin) & (df['x_ps71'] <= xmax) &
        (df['y_ps71'] >= ymin) & (df['y_ps71'] <= ymax)
    ].copy()


def ppb_core_subset(df, ps71_bounds):
    """PPB core square + trim of the two lone southern rays (POLARGAP P33.1/P33.3)
    below 88.5 S. Report-side twin of _ppb_core_subset (below), using this
    script's renamed columns."""
    sub = spatial_subset_ps71(df, ps71_bounds)
    spur = sub['traj_id'].astype(str).isin(PPB_SPUR_LEGS)
    south = sub['lat'] > PPB_SPUR_LAT_CUT
    return sub[~(spur & south)].copy()


# ---------------------------------------------------------------------------
# Subset functions EXPORTED to loading.py.
# These operate on RAW Bedmap columns ('longitude (degree_east)' etc.) — i.e.
# the dataframe exactly as loading.load_datasets() reads it, BEFORE any column
# renaming. loading.py imports the three names below and its dataset entries
# reference them; the "COPY FOR loading.py" snippets print these names. They are
# the source of truth for the actual pipeline's subsetting (distinct from the
# report helpers above, which run on this script's renamed lat/lon/traj_id cols).
# ---------------------------------------------------------------------------
def _ps71_subset(df, ps71_bounds):
    """Subset a Bedmap dataframe to a PS71 bounding box [xmin, xmax, ymin, ymax]."""
    x, y = to_ps71.transform(
        df['longitude (degree_east)'].values,
        df['latitude (degree_north)'].values,
    )
    xmin, xmax, ymin, ymax = ps71_bounds
    mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    return df[mask].copy()


def _ppb_core_subset(df):
    """PPB core square, with the two lone southern rays (POLARGAP flight P33,
    legs P33.1/P33.3) trimmed below 88.5 S — they extend out of the dense fan
    as an isolated spur. Inner part near the pole node (|lat| >= 88.5) is kept."""
    sub = _ps71_subset(df, PPB_CORE_BOX)
    spur = sub['trajectory_id'].astype(str).isin(PPB_SPUR_LEGS)
    south = sub['latitude (degree_north)'] > PPB_SPUR_LAT_CUT
    return sub[~(spur & south)].copy()


def _ps71_lowrelief_subset(df, ps71_bounds,
                           metrics_dir=None, hill_thresh=5, relief_thresh=500):
    """Subset to RES points falling within Ockenden low-relief grid cells (50 km)."""
    if metrics_dir is None:
        metrics_dir = METRICS_DIR
    x_grid = nc.Dataset(metrics_dir + 'x_ifpa.nc')['data'][:].data
    y_grid = nc.Dataset(metrics_dir + 'y_ifpa.nc')['data'][:].data
    relief = nc.Dataset(metrics_dir + 'ifpa_relief.nc')['data'][:].data
    hill50 = nc.Dataset(metrics_dir + 'ifpa_count_max_50.nc')['data'][:].data

    xmin, xmax, ymin, ymax = ps71_bounds
    cell_mask = ((x_grid >= xmin) & (x_grid <= xmax) &
                 (y_grid >= ymin) & (y_grid <= ymax) &
                 (hill50 <= hill_thresh) & (relief <= relief_thresh))
    lr_x, lr_y = x_grid[cell_mask], y_grid[cell_mask]

    px, py = to_ps71.transform(
        df['longitude (degree_east)'].values,
        df['latitude (degree_north)'].values,
    )
    hit = np.zeros(len(df), dtype=bool)
    for cx, cy in zip(lr_x, lr_y):
        hit |= (np.abs(px - cx) <= 25000) & (np.abs(py - cy) <= 25000)
    return df[hit].copy()


def ps71_to_latlon_corners(ps71_bounds):
    """Convert PS71 box corners to lat/lon for reference."""
    from_ps71 = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    xmin, xmax, ymin, ymax = ps71_bounds
    corners_x = [xmin, xmax, xmin, xmax]
    corners_y = [ymin, ymin, ymax, ymax]
    lons, lats = from_ps71.transform(corners_x, corners_y)
    return {
        'lat_min': min(lats), 'lat_max': max(lats),
        'lon_min': min(lons), 'lon_max': max(lons),
    }


def main():
    print("=" * 80)
    print("SUBSET BEDMAP DATA FOR OCKENDEN et al. (2025) -- PS71 BOUNDS")
    print("=" * 80)

    # Print region summary
    print("\nOckenden regions (from Zenodo PS71 bounds):")
    for rkey, r in OCKENDEN_REGIONS.items():
        ll = ps71_to_latlon_corners(r['ps71'])
        print(f"  {rkey:30s}  [{r['ockenden_class']:25s}]  "
              f"lat [{ll['lat_min']:7.2f}, {ll['lat_max']:7.2f}]  "
              f"lon [{ll['lon_min']:8.2f}, {ll['lon_max']:8.2f}]")

    found_overlaps = []
    file_cache = {}  # filepath -> (df_with_ps71, has_bed) | None; read each CSV once

    # Iterate regions in loading.py order. Regions interleave source files, so we
    # cache each loaded CSV to avoid re-reading the large files.
    for rkey, region in OCKENDEN_REGIONS.items():
        filepath = BASE_DIR + region['file']
        label = region['dataset_label']

        if filepath not in file_cache:
            print(f"\n{'~'*70}")
            print(f"  Reading {region['file']}")
            print(f"{'~'*70}")
            try:
                df = load_bedmap_csv(filepath)
            except FileNotFoundError:
                print(f"  *** FILE NOT FOUND ***")
                file_cache[filepath] = None
                continue
            has_bed = 'bed' in df.columns
            if has_bed:
                df = df[df['bed'] != -9999]
            df = add_ps71(df)
            print(f"  Valid rows: {len(df):,}")
            print(f"  PS71 x: [{df['x_ps71'].min():.0f}, {df['x_ps71'].max():.0f}]")
            print(f"  PS71 y: [{df['y_ps71'].min():.0f}, {df['y_ps71'].max():.0f}]")
            file_cache[filepath] = (df, has_bed)

        if file_cache[filepath] is None:
            continue
        df, has_bed = file_cache[filepath]

        if region.get('cell_mask'):
            sub = cell_mask_subset(df, region['ps71'])
        elif region.get('core_subset'):
            sub = ppb_core_subset(df, region['ps71'])
        else:
            sub = spatial_subset_ps71(df, region['ps71'])

        if len(sub) == 0:
            print(f"\n  x {rkey}: no overlap")
            continue

        ll = ps71_to_latlon_corners(region['ps71'])
        print(f"\n  >> {rkey}: {len(sub):,} pts  [{region['ockenden_class']}]")
        print(f"     {region['description']}")
        print(f"     PS71 box: {region['ps71']}")
        if region.get('cell_mask'):
            lr_x, lr_y = load_lowrelief_cells(region['ps71'])
            print(f"     Cell mask: {len(lr_x)} low-relief cells (50 km grid)")
        print(f"     ~lat [{ll['lat_min']:.2f}, {ll['lat_max']:.2f}]  "
              f"~lon [{ll['lon_min']:.2f}, {ll['lon_max']:.2f}]")

        if has_bed:
            print(f"     Bed elev: [{sub['bed'].min():.0f}, {sub['bed'].max():.0f}] m")
        if 'traj_id' in sub.columns:
            trajs = sub[sub['traj_id'] != -9999]['traj_id'].unique()
            print(f"     Trajectories: {len(trajs)}")

        found_overlaps.append({
            'dataset': label, 'file': region['file'],
            'region': rkey, 'n_points': len(sub),
            'class': region['ockenden_class'],
        })

        # Ready-to-use subset for loading.py (regions with bespoke subsetting
        # override the default plain-box lambda via 'loading_subset_repr').
        b = region['ps71']
        subset_line = region.get(
            'loading_subset_repr',
            f"'subset': lambda df, _b={b}: _ps71_subset(df, _b),")
        print(f"\n***~~~~******~~~~***COPY FOR loading.py:***~~~~******~~~~***")
        print(f"     {{")
        print(f"         'file': '{region['file']}',")
        print(f"         'label': '{label}_{rkey}',")
        print(f"         {subset_line}")
        print(f"     }},")
        print(f"\n~~~~******~~~~******~~~~******~~~~******~~~~******~~~~******~~~~")

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    if found_overlaps:
        for ov in found_overlaps:
            print(f"  {ov['dataset']:20s} x {ov['region']:30s} "
                  f"[{ov['class']:25s}] -> {ov['n_points']:>8,} pts")
    else:
        print("  No overlaps found.")


if __name__ == '__main__':
    import sys, io, os

    log_path = os.path.join(os.path.dirname(__file__), 'ockenden_coords-results.log')
    tee = io.StringIO()

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, s):
            for st in self.streams:
                st.write(s)
        def flush(self):
            for st in self.streams:
                st.flush()

    sys.stdout = Tee(sys.__stdout__, tee)
    main()
    sys.stdout = sys.__stdout__

    with open(log_path, 'w') as f:
        f.write(tee.getvalue())
    print(f"\nLog written to {log_path}")
