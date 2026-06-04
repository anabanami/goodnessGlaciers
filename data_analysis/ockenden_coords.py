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
METRICS_DIR = '/home/ana/Desktop/code/Data/ODSA/Ockenden/Data_Science_Zenodo/Data_Science_Zenodo/Metrics_v2/'

# WGS84 <-> EPSG:3031 (Antarctic Polar Stereographic, PS71)
to_ps71 = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)

# ---------------------------------------------------------------------------
# Exact PS71 bounds from Antarctic_FIGURES.ipynb  (bounds2, second block)
#   format: [x_min, x_max, y_min, y_max] in metres
# ---------------------------------------------------------------------------
OCKENDEN_REGIONS = {
    'Fig2C_Hercules_Dome': {
        'ps71': [-0.6e6, -0.3e6, -0.23e6, 0.07e6],
        'description': 'Hercules Dome -- U-shaped valleys',
        'ockenden_class': 'alpine',
        'fig': 'Fig 2C',
    },
    'Fig2A_Maud_SB': {
        'ps71': [0.15e6, 0.45e6, 1.025e6, 1.325e6],
        'description': 'Maud Subglacial Basin -- 400 km incised channel',
        'ockenden_class': 'low-relief / selective erosion',
        'fig': 'Fig 2A',
    },
    'Fig2B_Wilhelm_II': {
        'ps71': [2.02e6, 2.32e6, 0.05e6, 0.35e6],
        'description': 'Wilhelm II Land',
        'ockenden_class': 'alpine',
        'fig': 'Fig 2B',
    },
    'Fig2D_Recovery_SB': {
        'ps71': [0.0e6, 0.30e6, 0.6e6, 0.9e6],
        'description': 'Recovery Subglacial Basin -- geological boundary',
        'ockenden_class': 'low-relief / selective erosion',
        'fig': 'Fig 2D',
    },
    'Fig2G_Highland_A': {
        'ps71': [1.90e6, 2.20e6, -0.725e6, -0.425e6],
        'description': 'Highland A -- paleo-river landscape',
        'ockenden_class': 'alpine',
        'fig': 'Fig 2G',
    },
    # Zhigalov (Fig 2E) omitted — no RES data overlap
    'Gamburtsev_N': {
        'ps71': [1.0e6, 1.25e6, 0.28e6, 0.50e6],
        'description': 'Northern Gamburtsev Subglacial Mountains',
        'ockenden_class': 'alpine (subaerial)',
        'fig': 'not in Fig 2',
    },
    'Fig2H_Golicyna_SM': {
        'ps71': [2.15e6, 2.45e6, -0.5e6, -0.2e6],
        'description': 'Golicyna Subglacial Mountains -- dendritic valleys',
        'ockenden_class': 'alpine',
        'fig': 'Fig 2H',
    },
    'Fig2F_Resolution_SH': {
        'ps71': [1.05e6, 1.35e6, -1.575e6, -1.275e6],
        'description': 'Resolution Subglacial Highlands -- alpine valleys',
        'ockenden_class': 'alpine',
        'fig': 'Fig 2F',
    },
    'Fig4_Aurora_SB_lowrelief': {
        'ps71': [1.05e6, 2.20e6, -0.80e6, 0.20e6],
        'description': 'Aurora SB -- low-relief cells only (Ockenden metrics: hill50<=5, relief<=500m)',
        'ockenden_class': 'low-relief',
        'fig': 'Fig 4 classification region (filtered)',
        'cell_mask': True,  # flag: use Ockenden metric grid instead of simple box
    },
    'Fig1_Pensacola_Pole': {
        'ps71': [-0.9e6, 0.3e6, -0.6e6, 0.3e6],
        'description': 'Pensacola-Pole Basin -- selective erosion',
        'ockenden_class': 'selective erosion',
        'fig': 'Fig 1B-D',
    },
}


DATASETS = [
    {
        'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
        'label': 'ASB_ICECAP_2010',
        'candidate_regions': [
            'Fig4_Aurora_SB_lowrelief', 'Fig2A_Maud_SB',
            'Fig2F_Resolution_SH',
            'Fig2G_Highland_A', 'Fig2H_Golicyna_SM',
            'Fig2B_Wilhelm_II',
        ],
    },
    {
        'file': 'BAS_2012_ICEGRAV_AIR_BM3.csv',
        'label': 'Rec_Catch',
        'candidate_regions': ['Fig2D_Recovery_SB'],
    },
    {
        'file': 'BAS_2015_POLARGAP_AIR_BM3.csv',
        'label': 'POLARGAP_2015',
        'candidate_regions': [
            'Fig1_Pensacola_Pole', 'Fig2C_Hercules_Dome',
        ],
    },
]


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

    for ds in DATASETS:
        filepath = BASE_DIR + ds['file']
        label = ds['label']
        print(f"\n{'~'*70}")
        print(f"  {label}  ({ds['file']})")
        print(f"{'~'*70}")

        try:
            df = load_bedmap_csv(filepath)
        except FileNotFoundError:
            print(f"  *** FILE NOT FOUND ***")
            continue

        has_bed = 'bed' in df.columns
        if has_bed:
            df = df[df['bed'] != -9999]

        df = add_ps71(df)
        print(f"  Valid rows: {len(df):,}")
        print(f"  PS71 x: [{df['x_ps71'].min():.0f}, {df['x_ps71'].max():.0f}]")
        print(f"  PS71 y: [{df['y_ps71'].min():.0f}, {df['y_ps71'].max():.0f}]")

        for rkey in ds['candidate_regions']:
            region = OCKENDEN_REGIONS[rkey]

            if region.get('cell_mask'):
                sub = cell_mask_subset(df, region['ps71'])
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
                'dataset': label, 'file': ds['file'],
                'region': rkey, 'n_points': len(sub),
                'class': region['ockenden_class'],
            })

            # Ready-to-use subset for loading.py
            b = region['ps71']
            print(f"\n     COPY FOR loading.py:")
            print(f"     {{")
            print(f"         'file': '{ds['file']}',")
            print(f"         'label': '{label}_{rkey}',")
            print(f"         'subset': lambda df, _b={b}: _ps71_subset(df, _b),")
            print(f"     }},")

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
