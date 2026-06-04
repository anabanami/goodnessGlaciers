"""
Scan all 84 Bedmap CSVs against Ockenden PS71 bounding boxes.
Finds which datasets have RES data in each region.
"""

import pandas as pd
import numpy as np
from pyproj import Transformer
from pathlib import Path
import sys, io, os, time

RESULTS_DIR = Path('/home/ana/Desktop/code/Data/ODSA/all_data/bedmap3_data/bedmap3/Results/')
LOG_PATH = os.path.join(os.path.dirname(__file__), 'scan_all_datasets.log')

to_ps71 = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)

OCKENDEN_REGIONS = {
    'Fig2C_Hercules_Dome': {
        'ps71': [-0.6e6, -0.3e6, -0.23e6, 0.07e6],
        'ockenden_class': 'alpine',
    },
    'Fig2A_Maud_SB': {
        'ps71': [0.15e6, 0.45e6, 1.025e6, 1.325e6],
        'ockenden_class': 'low-relief / selective erosion',
    },
    'Fig2B_Wilhelm_II': {
        'ps71': [2.02e6, 2.32e6, 0.05e6, 0.35e6],
        'ockenden_class': 'alpine',
    },
    'Fig2D_Recovery_SB': {
        'ps71': [0.0e6, 0.30e6, 0.6e6, 0.9e6],
        'ockenden_class': 'low-relief / selective erosion',
    },
    'Fig2G_Highland_A': {
        'ps71': [1.90e6, 2.20e6, -0.725e6, -0.425e6],
        'ockenden_class': 'alpine',
    },
    'Gamburtsev_N': {
        'ps71': [1.0e6, 1.25e6, 0.28e6, 0.50e6],
        'ockenden_class': 'alpine (subaerial)',
    },
    'Fig2H_Golicyna_SM': {
        'ps71': [2.15e6, 2.45e6, -0.5e6, -0.2e6],
        'ockenden_class': 'alpine',
    },
    'Fig2F_Resolution_SH': {
        'ps71': [1.05e6, 1.35e6, -1.575e6, -1.275e6],
        'ockenden_class': 'alpine',
    },
    'Fig4_Aurora_SB': {
        'ps71': [1.05e6, 2.20e6, -0.80e6, 0.20e6],
        'ockenden_class': 'low-relief',
    },
    'Fig1_Pensacola_Pole': {
        'ps71': [-0.9e6, 0.3e6, -0.6e6, 0.3e6],
        'ockenden_class': 'selective erosion',
    },
}


def load_and_project(filepath):
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
    df = df.rename(columns=col_map)
    if 'bed' in df.columns:
        df = df[df['bed'] != -9999]
    x, y = to_ps71.transform(df['lon'].values, df['lat'].values)
    df['x_ps71'] = x
    df['y_ps71'] = y
    return df


def count_in_box(df, ps71):
    xmin, xmax, ymin, ymax = ps71
    return int(((df['x_ps71'] >= xmin) & (df['x_ps71'] <= xmax) &
                (df['y_ps71'] >= ymin) & (df['y_ps71'] <= ymax)).sum())


def main():
    csvs = sorted(RESULTS_DIR.glob('*.csv'))
    print(f"Scanning {len(csvs)} CSVs against {len(OCKENDEN_REGIONS)} Ockenden regions\n")

    # results[region][file] = n_points
    results = {r: {} for r in OCKENDEN_REGIONS}
    t0 = time.time()

    for i, csv_path in enumerate(csvs):
        fname = csv_path.name
        print(f"[{i+1:2d}/{len(csvs)}] {fname}...", end=' ', flush=True)
        try:
            df = load_and_project(csv_path)
        except Exception as e:
            print(f"SKIP ({e})")
            continue

        hits = []
        for rkey, region in OCKENDEN_REGIONS.items():
            n = count_in_box(df, region['ps71'])
            if n > 0:
                results[rkey][fname] = n
                hits.append(f"{rkey}={n:,}")

        if hits:
            print(f"{len(df):>9,} rows -> {', '.join(hits)}")
        else:
            print(f"{len(df):>9,} rows -> no overlap")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s\n")

    # Summary per region
    print("=" * 90)
    print("OVERLAP SUMMARY BY REGION")
    print("=" * 90)
    for rkey, region in OCKENDEN_REGIONS.items():
        datasets = results[rkey]
        cls = region['ockenden_class']
        print(f"\n  {rkey}  [{cls}]")
        if not datasets:
            print(f"    ** NO DATASETS FOUND **")
        else:
            for fname, n in sorted(datasets.items(), key=lambda x: -x[1]):
                print(f"    {fname:50s} {n:>10,} pts")


if __name__ == '__main__':
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

    with open(LOG_PATH, 'w') as f:
        f.write(tee.getvalue())
    print(f"\nLog written to {LOG_PATH}")
