"""
Subset Bedmap radar data to match Ockenden et al. (2025) figure regions.

Uses lat/lon bounding boxes estimated from named features in the paper.
These are approximate — refine by checking against the Metrics_v2/*.nc grids
if needed (those contain x_ifpa.nc and y_ifpa.nc with PS71 coordinates).

Usage:
    1. Adjust BASE_DIR to the folder containing the datasets
    2. Run: python subset_for_ockenden.py
    3. Check which regions have data overlap
    4. Copy the suggested 'subset' lambdas into the analysis pipeline
"""

import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = '/home/ana/Desktop/code/Data/Bedmap/all_data/'

# ============================================================================
# OCKENDEN FIGURE REGIONS — lat/lon bounding boxes
#
# Approximate centers from named Antarctic features.
# At ~75°S: 1° lat ≈ 111 km, 1° lon ≈ 29 km
# At ~85°S: 1° lat ≈ 111 km, 1° lon ≈ 10 km
# Paper states panels are 100x100 to 300x300 km.
# ============================================================================

OCKENDEN_REGIONS = {
    # Fig. 2A — Maud Subglacial Basin (300x300 km)
    'Fig2A_Maud_SB': {
        'lat_min': -76.5, 'lat_max': -73.5,
        'lon_min': 5.0,   'lon_max': 25.0,
        'description': 'Maud Subglacial Basin — 400km incised channel',
    },
    # Fig. 2D — Recovery Subglacial Basin (300x300 km)
    'Fig2D_Recovery_SB': {
        'lat_min': -83.5, 'lat_max': -80.5,
        'lon_min': -35.0, 'lon_max': -15.0,
        'description': 'Recovery Subglacial Basin — geological boundary',
    },
    # Fig. 2C — Hercules Dome (300x300 km)
    'Fig2C_Hercules_Dome': {
        'lat_min': -87.5, 'lat_max': -85.0,
        'lon_min': -120.0, 'lon_max': -100.0,
        'description': 'Hercules Dome — U-shaped valleys',
    },
    # Fig. 1B-D — Pensacola-Pole Basin (main showcase, larger region)
    'Fig1_Pensacola_Pole': {
        'lat_min': -88.0, 'lat_max': -82.0,
        'lon_min': -60.0, 'lon_max': -20.0,
        'description': 'Pensacola-Pole Basin — main comparison region (Fig 1)',
    },
    # Fig. 2F — Resolution Subglacial Highlands (300x300 km)
    'Fig2F_Resolution_SH': {
        'lat_min': -76.0, 'lat_max': -73.0,
        'lon_min': 135.0, 'lon_max': 150.0,
        'description': 'Resolution Subglacial Highlands — alpine valleys',
    },
    # Fig. 2G — Highland A (300x300 km)
    'Fig2G_Highland_A': {
        'lat_min': -76.0, 'lat_max': -73.0,
        'lon_min': 118.0, 'lon_max': 132.0,
        'description': 'Highland A — paleo-river landscape',
    },
    # Fig. 2H — Golicyna Subglacial Highlands (300x300 km)
    'Fig2H_Golicyna_SH': {
        'lat_min': -75.0, 'lat_max': -72.0,
        'lon_min': 103.0, 'lon_max': 117.0,
        'description': 'Golicyna Subglacial Highlands — dendritic valleys',
    },
    # Aurora Subglacial Basin — low-relief, sedimentary (from Fig 4 classification)
    'Fig4_Aurora_SB': {
        'lat_min': -76.0, 'lat_max': -71.0,
        'lon_min': 105.0, 'lon_max': 125.0,
        'description': 'Aurora Subglacial Basin — classified as low-relief',
    },
}

# ============================================================================
# MY DATASETS → candidate Ockenden regions
# ============================================================================

DATASETS = [
    # {
    #     'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
    #     'label': 'ASB_ICECAP_2010',
    #     'candidate_regions': [
    #         'Fig4_Aurora_SB', 'Fig2F_Resolution_SH',
    #         'Fig2G_Highland_A', 'Fig2H_Golicyna_SH',
    #     ],
    # },
    # {
    #     'file': 'UTIG_2008_ICECAP_AIR_BM2.csv',
    #     'label': 'ASB_ICECAP_2008',
    #     'candidate_regions': [
    #         'Fig4_Aurora_SB', 'Fig2F_Resolution_SH',
    #         'Fig2G_Highland_A', 'Fig2H_Golicyna_SH',
    #     ],
    # },
    ############################################################################
    {
        'file': 'BAS_2012_ICEGRAV_AIR_BM3.csv',
        'label': 'Rec_Catch',
        'candidate_regions': [
            'Fig2D_Recovery_SB', 'Fig1_Pensacola_Pole',
        ],
    },

    {
        'file': 'BAS_2010_IMAFI_AIR_BM3.csv',
        'label': 'Rec__SB',
        'candidate_regions': [
            'Fig2D_Recovery_SB',
        ],
    },

    {
        'file': 'NASA_2014_ICEBRIDGE_AIR_BM3.csv',
        'label': 'Rec__SB',
        'candidate_regions': [
            'Fig2D_Recovery_SB',
        ],
    },

    {
        'file': 'NASA_2016_ICEBRIDGE_AIR_BM3.csv',
        'label': 'Rec__SB',
        'candidate_regions': [
            'Fig2D_Recovery_SB',
        ],
    },
    
    {
        'file': 'NASA_2017_ICEBRIDGE_AIR_BM3.csv',
        'label': 'Rec__SB',
        'candidate_regions': [
            'Fig2D_Recovery_SB',
        ],
    },
    
    {
        'file': 'NASA_2018_ICEBRIDGE_AIR_BM3.csv',
        'label': 'Rec__SB',
        'candidate_regions': [
            'Fig2D_Recovery_SB',
        ],
    },

    {
        'file': 'NASA_2019_ICEBRIDGE_AIR_BM3.csv',
        'label': 'Rec__SB',
        'candidate_regions': [
            'Fig2D_Recovery_SB',
        ],
    },
    
    ############################################################################

    # {
    #     'file': 'AWI_2018_ANIRES_AIR_BM3.csv',
    #     'label': 'DML_AniRES',
    #     'candidate_regions': [
    #         'Fig2A_Maud_SB',
    #     ],
    # },
    # {
    #     'file': 'BAS_2015_POLARGAP_AIR_BM3.csv',
    #     'label': 'POLARGAP_2015',
    #     'candidate_regions': [
    #         'Fig1_Pensacola_Pole', 'Fig2C_Hercules_Dome',
    #     ],
    # },
]


def load_bedmap_csv(filepath):
    """Load a Bedmap CSV with standardized column names."""
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
    return df


def spatial_subset(df, region):
    """Filter dataframe to a lat/lon bounding box."""
    mask = (
        (df['lat'] >= region['lat_min']) &
        (df['lat'] <= region['lat_max']) &
        (df['lon'] >= region['lon_min']) &
        (df['lon'] <= region['lon_max'])
    )
    return df[mask].copy()


def main():
    print("=" * 80)
    print("SUBSET BEDMAP DATA FOR OCKENDEN et al. (2025) COMPARISON")
    print("=" * 80)

    found_overlaps = []

    for ds in DATASETS:
        filepath = BASE_DIR + ds['file']
        label = ds['label']
        print(f"\n{'━' * 70}")
        print(f"  {label}  ({ds['file']})")
        print(f"{'━' * 70}")

        try:
            df = load_bedmap_csv(filepath)
        except FileNotFoundError:
            print(f"  *** FILE NOT FOUND ***")
            continue

        n_total = len(df)
        has_bed = 'bed' in df.columns
        if has_bed:
            df_valid = df[df['bed'] != -9999]
            n_valid = len(df_valid)
        else:
            df_valid = df
            n_valid = n_total

        print(f"  Rows: {n_total:,}  |  Valid bed picks: {n_valid:,}")
        print(f"  Lat:  [{df['lat'].min():.2f}, {df['lat'].max():.2f}]")
        print(f"  Lon:  [{df['lon'].min():.2f}, {df['lon'].max():.2f}]")

        for rkey in ds['candidate_regions']:
            region = OCKENDEN_REGIONS[rkey]
            sub = spatial_subset(df_valid, region) if has_bed else spatial_subset(df, region)

            if len(sub) == 0:
                print(f"\n  ✗ {rkey}: no overlap")
                continue

            n_sub = len(sub)
            print(f"\n  ✓ {rkey}: {n_sub:,} valid points")
            print(f"    {region['description']}")
            print(f"    Subset lat: [{sub['lat'].min():.3f}, {sub['lat'].max():.3f}]")
            print(f"    Subset lon: [{sub['lon'].min():.3f}, {sub['lon'].max():.3f}]")

            if has_bed:
                print(f"    Bed elev:   [{sub['bed'].min():.0f}, {sub['bed'].max():.0f}] m")

            if 'traj_id' in sub.columns:
                trajs = sub[sub['traj_id'] != -9999]['traj_id'].unique()
                print(f"    Trajectories: {len(trajs)}")

            found_overlaps.append({
                'dataset': label,
                'file': ds['file'],
                'region': rkey,
                'n_points': n_sub,
            })

            # Print ready-to-use subset definition
            r = region
            print(f"\n    ┌─ COPY INTO YOUR ANALYSIS PIPELINE ───────────────┐")
            print(f"    {{")
            print(f"        'file': '{ds['file']}',")
            print(f"        'label': '{label}_{rkey}',")
            print(f"        'subset': lambda df, _r={{")
            print(f"            'lat_min': {r['lat_min']}, 'lat_max': {r['lat_max']},")
            print(f"            'lon_min': {r['lon_min']}, 'lon_max': {r['lon_max']},")
            print(f"        }}: df[")
            print(f"            (df['latitude (degree_north)'] >= _r['lat_min']) &")
            print(f"            (df['latitude (degree_north)'] <= _r['lat_max']) &")
            print(f"            (df['longitude (degree_east)']  >= _r['lon_min']) &")
            print(f"            (df['longitude (degree_east)']  <= _r['lon_max'])")
            print(f"        ].copy(),")
            print(f"    }},")
            print(f"    └────────────────────────────────────────────────────┘")

    # Summary
    print(f"\n\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    if found_overlaps:
        for ov in found_overlaps:
            print(f"  {ov['dataset']:20s} × {ov['region']:25s} → {ov['n_points']:>8,} pts")
    else:
        print("  No overlaps found.")

    print(f"""
NEXT STEPS:
  1. If a region shows 0 overlap, try expanding the box by 1-2 degrees.
""")


if __name__ == '__main__':
    main()