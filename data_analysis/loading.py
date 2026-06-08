import glob
import os
import numpy as np
import pandas as pd
import netCDF4 as nc
from pyproj import Transformer


_MIGRATED = {'2-D migration processing', '2-D Synthetic Aperture Radar processing',
             '2-D Synthetic Aperture Radar focused processing'}
_PARTIAL  = {'1-D Synthetic Aperture Radar processing',
             'Synthetic Aperture Radar unfocused processing',
             'pik1 (short coherent) processing',
             'MUSIC (Swath) Processing'}

_to_ps71 = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)

# Output configuration
OUTPUT_BASE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    # 'TEST-ONE-SMUG-region/',
    # 'SMUG-regions/',
    'Ockenden-regions/',
)


def _ps71_subset(df, ps71_bounds):
    """Subset a Bedmap dataframe to a PS71 bounding box [xmin, xmax, ymin, ymax]."""
    x, y = _to_ps71.transform(
        df['longitude (degree_east)'].values,
        df['latitude (degree_north)'].values,
    )
    xmin, xmax, ymin, ymax = ps71_bounds
    mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    return df[mask].copy()


def _ps71_lowrelief_subset(df, ps71_bounds,
                           metrics_dir=None, hill_thresh=5, relief_thresh=500):
    """Subset to RES points falling within Ockenden low-relief grid cells (50 km)."""
    if metrics_dir is None:
        metrics_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'Ockenden/Data_Science_Zenodo/Data_Science_Zenodo/Metrics_v2/')
    x_grid = nc.Dataset(metrics_dir + 'x_ifpa.nc')['data'][:].data
    y_grid = nc.Dataset(metrics_dir + 'y_ifpa.nc')['data'][:].data
    relief = nc.Dataset(metrics_dir + 'ifpa_relief.nc')['data'][:].data
    hill50 = nc.Dataset(metrics_dir + 'ifpa_count_max_50.nc')['data'][:].data

    xmin, xmax, ymin, ymax = ps71_bounds
    cell_mask = ((x_grid >= xmin) & (x_grid <= xmax) &
                 (y_grid >= ymin) & (y_grid <= ymax) &
                 (hill50 <= hill_thresh) & (relief <= relief_thresh))
    lr_x, lr_y = x_grid[cell_mask], y_grid[cell_mask]

    px, py = _to_ps71.transform(
        df['longitude (degree_east)'].values,
        df['latitude (degree_north)'].values,
    )
    hit = np.zeros(len(df), dtype=bool)
    for cx, cy in zip(lr_x, lr_y):
        hit |= (np.abs(px - cx) <= 25000) & (np.abs(py - cy) <= 25000)
    return df[hit].copy()

def _parse_processing_flag(filepath):
    with open(filepath) as f:
        for line in f:
            if not line.startswith('#'):
                break
            if line.startswith('#history:'):
                hist = line.split(':', 1)[1].strip()
                if hist in _MIGRATED:
                    return 'migrated'
                if hist in _PARTIAL:
                    return 'partial'
                return 'unmigrated_or_unknown'
    return 'unmigrated_or_unknown'


def load_datasets():
    base_path = 'all_data/bedmap3_data/bedmap*/Results/'
    all_dfs = []

    target_files = [
        # =================================================================
        # Ockenden et al. (2025) regions — PS71 bounds from Zenodo
        # =================================================================

        # LOW-RELIEF: Aurora SB filtered to Ockenden low-relief cells
        {
            'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
            'label': 'ASB_ICECAP_2010_Fig4_Aurora_SB_lowrelief',
            'subset': lambda df: _ps71_lowrelief_subset(
                df, [1.05e6, 2.20e6, -0.80e6, 0.20e6]),
        },

        # LOW-RELIEF / SELECTIVE EROSION: Maud Subglacial Basin
        {
            'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
            'label': 'ASB_ICECAP_2010_Fig2A_Maud_SB',
            'subset': lambda df: _ps71_subset(
                df, [0.15e6, 0.45e6, 1.025e6, 1.325e6]),
        },

        # LOW-RELIEF / SELECTIVE EROSION: Recovery Subglacial Basin
        {
            'file': 'BAS_2012_ICEGRAV_AIR_BM3.csv',
            'label': 'Rec_Catch_Fig2D_Recovery_SB',
            'subset': lambda df: _ps71_subset(
                df, [0.0e6, 0.30e6, 0.6e6, 0.9e6]),
        },

        # ALPINE: Resolution Subglacial Highlands
        {
            'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
            'label': 'ASB_ICECAP_2010_Fig2F_Resolution_SH',
            'subset': lambda df: _ps71_subset(
                df, [1.05e6, 1.35e6, -1.575e6, -1.275e6]),
        },

        # ALPINE: Highland A
        {
            'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
            'label': 'ASB_ICECAP_2010_Fig2G_Highland_A',
            'subset': lambda df: _ps71_subset(
                df, [1.90e6, 2.20e6, -0.725e6, -0.425e6]),
        },

        # ALPINE: Golicyna Subglacial Mountains
        {
            'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
            'label': 'ASB_ICECAP_2010_Fig2H_Golicyna_SM',
            'subset': lambda df: _ps71_subset(
                df, [2.15e6, 2.45e6, -0.5e6, -0.2e6]),
        },

        # ALPINE: Wilhelm II Land
        {
            'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
            'label': 'ASB_ICECAP_2010_Fig2B_Wilhelm_II',
            'subset': lambda df: _ps71_subset(
                df, [2.02e6, 2.32e6, 0.05e6, 0.35e6]),
        },

        # ALPINE: Hercules Dome
        {
            'file': 'BAS_2015_POLARGAP_AIR_BM3.csv',
            'label': 'POLARGAP_2015_Fig2C_Hercules_Dome',
            'subset': lambda df: _ps71_subset(
                df, [-0.6e6, -0.3e6, -0.23e6, 0.07e6]),
        },

        # SELECTIVE EROSION: Pensacola-Pole Basin
        {
            'file': 'BAS_2015_POLARGAP_AIR_BM3.csv',
            'label': 'POLARGAP_2015_Fig1_Pensacola_Pole',
            'subset': lambda df: _ps71_subset(
                df, [-0.9e6, 0.3e6, -0.6e6, 0.3e6]),
        },
    ]

    file_cache = {}

    for item in target_files:
        filename = item['file']
        label = item['label']
        matches = glob.glob(os.path.join(base_path, filename))
        if not matches:
            print(f"⚠️ Warning: {filename} not found. Skipping.")
            continue

        filepath = matches[0]

        try:
            if filepath not in file_cache:
                print(f"  Reading {filename}...")
                file_cache[filepath] = pd.read_csv(filepath, comment='#', low_memory=False)
                file_cache[filepath]['processing_flag'] = _parse_processing_flag(filepath)

            df = file_cache[filepath].copy()

            if 'subset' in item:
                df = item['subset'](df)

            if 'force_id' in item:
                df['trajectory_id'] = item['force_id']

            initial_len = len(df)
            has_valid_bed = df['bedrock_altitude (m)'] != -9999
            has_valid_traj = (df['trajectory_id'] != -9999) | ('force_id' in item)
            df = df[has_valid_bed & has_valid_traj].copy()

            df['trajectory_id'] = df['trajectory_id'].astype(str)

            if len(df) > 0:
                pflag = df['processing_flag'].iloc[0]
                print(f"✓ {label} loaded: {len(df)} rows (Filtered {initial_len - len(df)} nulls) [{pflag}]")
                all_dfs.append({'name': label, 'data': df})
            else:
                print(f"---{label} resulted in 0 rows.---")

        except Exception as e:
            print(f"---Error loading {label}: {e}---")

    del file_cache
    return all_dfs
