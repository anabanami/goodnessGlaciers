"""
Subset Bedmap radar data to match [Ockenden_2026] figure regions.

Uses the EXACT PS71 bounding boxes from the Zenodo code repository [Ockenden_2025_data]
(Antarctic_FIGURES.ipynb, bounds2[0..8]) instead of approximate lat/lon
guesses. Filtering is done in PS71 space to avoid polar lat/lon distortion.

Usage:
    python ockenden_coords.py
"""

import glob
import pandas as pd
import numpy as np
import netCDF4 as nc
from pyproj import Transformer

# Glob over every release, not just bedmap3: the Dome C SW square pulls two
# Bedmap2 campaigns. Mirrors loading.py's base_path, so a region's 'file' is
# just the CSV name and the release is read off the _BM<n> suffix.
RESULTS_GLOB = '/home/ana/Desktop/code/Data/ODSA/all_data/bedmap3_data/bedmap*/Results/'
METRICS_DIR = '/home/ana/Desktop/code/Data/ODSA/all_data/Ockenden/Data_Science_Zenodo/Data_Science_Zenodo/Metrics_v2/'


def resolve(filename):
    """Absolute path of a Bedmap Results CSV, whichever release it lives in."""
    m = glob.glob(RESULTS_GLOB + filename)
    if not m:
        raise FileNotFoundError(filename)
    return m[0]

# WGS84 <-> EPSG:3031 (Antarctic Polar Stereographic, PS71)
to_ps71 = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)

# --- PPB core selection: single source of truth. Referenced by the region dict
# ('ps71') and by BOTH subset twins (_ppb_core_subset, ppb_core_subset). Change
# the box or the spur trim here, once. ---
PPB_CORE_BOX = [-0.247e6, 0.340e6, -0.398e6, 0.280e6]
PPB_SPUR_LEGS = ['P33.1', 'P33.3']   # the two lone southern rays (POLARGAP flight P33)
PPB_SPUR_LAT_CUT = -88.5             # drop spur points with lat > this (|lat| < 88.5)

# --- SOAR line reconstruction (UTIG_1999_SOAR-LVS-WLK, Bedmap2), used by the
# second PPB entry. That compilation numbers trajectory_id 1..N per ROW
# (153,103 ids / 153,103 rows), so it has no flight lines to load; the ids are
# rebuilt by cutting the file's row order wherever consecutive points jump more
# than SOAR_GAP_M. Nominal along-track spacing in the PPB box is 1.04 km
# (p90 1.18 km), so 5 km is ~5x nominal and does not cut inside a line. Result:
# 126 segments >= 5 pts, 9,053 of 9,113 in-box points, median 42 pts (~44 km),
# median end-to-end/path-length 0.999 -- i.e. straight lines, and they plot as
# the orthogonal SOAR grid. ---
SOAR_GAP_M = 5000        # split row order on jumps larger than this
SOAR_MIN_PTS = 5         # drop reconstructed segments shorter than this

# --- Dome C SW square: single source of truth, shared by the four campaign
# entries below (two Bedmap2, two Bedmap3). 300x300 km, same size as the
# Golicyna/Highland A/Aurora squares. Centred on the hand-drawn overview box
# (see the georeferencing note on the region entries). ---
DOMEC_SW_BOX = [1.02e6, 1.32e6, -1.237e6, -0.937e6]

# --- Dronning Maud Land 3E square: same deal, 300x300 km concentric with a
# second hand-drawn overview box. ---
DML_3E_BOX = [-0.05e6, 0.25e6, 1.62e6, 1.92e6]

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
    # Second campaign in the SAME PPB core square: the Bedmap2 SOAR grid that
    # crosses the black overview box NE-SW (the green track under the POLARGAP
    # fan). Same box as 'Pensacola_Pole', different platform and era, so the two
    # are directly comparable over identical ground.
    # CAVEAT, and it is not small: along-track spacing is 1.04 km here vs 31 m
    # for POLARGAP in the same box. Any along-track roughness/relief metric is
    # sampled ~34x coarser and is NOT comparable to the rest of the catalogue at
    # face value -- treat this region as a bed-elevation / long-wavelength check
    # (and as an independent 1999 sounding of ground POLARGAP re-flew in 2015),
    # not as a roughness sample. Trajectories are reconstructed (see SOAR_GAP_M).
    'Pensacola_Pole_SOAR_BM2': {
        'file': 'UTIG_1999_SOAR-LVS-WLK_AIR_BM2.csv',
        'dataset_label': 'SOAR_1999',
        'ps71': PPB_CORE_BOX,
        'description': ('Pensacola-Pole Basin -- Bedmap2 SOAR TAM/South Pole grid over the '
                        'same core square as Pensacola_Pole. 9,053 bed points on 126 '
                        'reconstructed lines; 1.04 km along-track spacing (POLARGAP: 31 m).'),
        'ockenden_class': 'selective erosion',
        'fig': 'n/a -- same box as Fig 1B-D, second campaign',
        'soar_subset': True,  # box + gap-split line reconstruction, not a plain box
        'loading_subset_repr': "'subset': _soar_ppb_subset,",  # matches loading.py
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
    # --- Dome C SW square -------------------------------------------------
    # NOT an Ockenden figure region: added from a hand-drawn box on the
    # Bedmap1/2/3 overview map (map_bedmap3_all.py --generations 1 2 3
    # --by-release). The drawn box was PS71 x[941, 1399] km, y[-1311, -862] km
    # (~457 km wide, centre 132.89E / 75.38S); DOMEC_SW_BOX is the concentric
    # 300x300 km crop of it. Georeferenced off the 70S/60S graticule circles of
    # the overview PNG and cross-checked against the flight-track caches.
    # 'DomeC_SW' is positional (the box sits SW of the Dome C survey hub), not a
    # gazetteer name -- rename if you want the actual subglacial feature.
    # Ockenden Metrics_v2 over the box: 36 cells, median relief 960 m, median
    # hill50 10.5, only 1/36 cells low-relief -> high-relief terrain, so no
    # low-relief cell mask here (unlike Fig4C_Aurora_SB_lowrelief).
    # Thirteen campaigns have tracks in the box; only the three below survive.
    # Deliberately excluded:
    #   - INGV Talos-Dome 1997/1999/2001/2003 (the green fan out of Dome C),
    #     AWI_2007_ANTR, INGV_1997_ITASE -- bedrock_altitude is -9999 for every
    #     row, so loading.py's bed filter empties them.
    #   - UTIG_1999_SOAR-LVS-WLK (the dense orange grid the box was drawn
    #     around) -- bed IS valid (4,505 pts) but this legacy compilation numbers
    #     trajectory_id 1..N per ROW (153,103 ids for 153,103 rows), so it loads
    #     as 4,505 one-point tracks and every along-track metric downstream is
    #     meaningless. Along-track spacing is ~1.1 km vs ~25 m for the three
    #     below. Reconstruct flight lines from point order/geometry to use it.
    #   - PRIC_2018_CHA4 -- misses the 300 km box entirely.
    # Each campaign gets its own entry because loading.py is one file per entry;
    # they share DOMEC_SW_BOX.
    'DomeC_SW_sq_WISE_ISODYN': {
        'file': 'BAS_2005_WISE-ISODYN_AIR_BM2.csv',
        'dataset_label': 'BM2',
        'ps71': DOMEC_SW_BOX,
        'description': 'Dome C SW 300x300 km square -- BAS 2005 WISE-ISODYN box transects',
        'ockenden_class': 'unclassified (high-relief)',
        'fig': 'n/a -- hand-drawn overview box',
    },
    'DomeC_SW_sq_ICECAP': {
        'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
        'dataset_label': 'BM3',
        'ps71': DOMEC_SW_BOX,
        'description': 'Dome C SW 300x300 km square -- UTIG 2010 ICECAP long NE-SW lines',
        'ockenden_class': 'unclassified (high-relief)',
        'fig': 'n/a -- hand-drawn overview box',
    },
    'DomeC_SW_sq_ICEBRIDGE': {
        'file': 'NASA_2013_ICEBRIDGE_AIR_BM3.csv',
        'dataset_label': 'BM3',
        'ps71': DOMEC_SW_BOX,
        'description': ('Dome C SW 300x300 km square -- NASA 2013 IceBridge. Stands in for '
                        'the INGV Talos-Dome lines, which look like the dominant BM3 coverage '
                        'on the overview map but carry no bed.'),
        'ockenden_class': 'unclassified (high-relief)',
        'fig': 'n/a -- hand-drawn overview box',
    },
    # --- Dronning Maud Land 3E square -------------------------------------
    # Second hand-drawn overview box, same treatment as DomeC_SW. Drawn box was
    # PS71 x[-119, 318] km, y[1547, 1991] km (~440 km wide, centre 3.23E /
    # 73.79S); DML_3E_BOX is the concentric 300x300 km crop.
    # Eighteen campaigns have tracks in the box and exactly ONE is usable. The
    # solid orange on the overview map is the AWI DML1-10 / ANTR series plus
    # AWI_2018_JURAS and BAS_2001_MAMOG -- all bedrock_altitude = -9999
    # throughout. BEDMAP1 has valid bed but the row-counter trajectory_id.
    # So this region is UTIG_2010_ICECAP alone: the green fan on the overview
    # map, hub at the box's north edge, rays running south across it.
    # Ockenden Metrics_v2 over the box: 42 cells, median relief 1120 m, median
    # hill50 18.0, 0/42 low-relief -- the most rugged of the ODSA regions
    # (DomeC_SW is 960 m / 10.5), consistent with the DML mountain ranges.
    'DML_3E_sq_ICECAP': {
        'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
        'dataset_label': 'BM3',
        'ps71': DML_3E_BOX,
        'description': ('Dronning Maud Land 3E 300x300 km square -- UTIG 2010 ICECAP. '
                        '101,492 bed points over 25 trajectories (median 3,586 pts/track, '
                        '~23 m spacing), bed -853 to 2567 m. Disjoint from Fig2A_Maud_SB, '
                        'which is the same campaign file ~500 km to the south.'),
        'ockenden_class': 'unclassified (very high relief)',
        'fig': 'n/a -- hand-drawn overview box',
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


def _gap_split_ids(x, y, gap=SOAR_GAP_M, prefix='SOAR_L'):
    """Label runs of consecutive rows as lines, cutting where the point-to-point
    step exceeds `gap`. Relies on file row order being acquisition order."""
    cut = np.hypot(np.diff(x), np.diff(y)) > gap
    return np.array([f'{prefix}{i:03d}' for i in np.concatenate([[0], np.cumsum(cut)])])


def soar_ppb_subset(df, ps71_bounds):
    """PPB core square over the Bedmap2 SOAR file, with trajectory_id rebuilt from
    row-order gaps. Report-side twin of _soar_ppb_subset (below)."""
    sub = spatial_subset_ps71(df, ps71_bounds)
    sub = sub[sub['bed'] != -9999].copy()      # split on real soundings only
    sub['traj_id'] = _gap_split_ids(sub['x_ps71'].values, sub['y_ps71'].values)
    keep = sub.groupby('traj_id')['traj_id'].transform('size') >= SOAR_MIN_PTS
    return sub[keep].copy()


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


def _soar_ppb_subset(df):
    """PPB core square over UTIG_1999_SOAR-LVS-WLK, with trajectory_id REPLACED by
    lines reconstructed from row-order gaps (the file's own ids are a row counter,
    see _warn_degenerate_trajectories in loading.py). Bed nulls are dropped here so
    the split runs on real soundings; loading.py's own null filter then finds none.
    Segments shorter than SOAR_MIN_PTS points are dropped."""
    sub = _ps71_subset(df, PPB_CORE_BOX)
    sub = sub[sub['bedrock_altitude (m)'] != -9999].copy()
    x, y = to_ps71.transform(
        sub['longitude (degree_east)'].values,
        sub['latitude (degree_north)'].values,
    )
    sub['trajectory_id'] = _gap_split_ids(x, y)
    keep = sub.groupby('trajectory_id')['trajectory_id'].transform('size') >= SOAR_MIN_PTS
    return sub[keep].copy()


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
    print("SUBSET BEDMAP DATA FOR OCKENDEN_2026 -- PS71 BOUNDS")
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
        try:
            filepath = resolve(region['file'])
        except FileNotFoundError:
            print(f"\n  x {rkey}: {region['file']} *** NOT FOUND ***")
            continue
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
        elif region.get('soar_subset'):
            sub = soar_ppb_subset(df, region['ps71'])
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
