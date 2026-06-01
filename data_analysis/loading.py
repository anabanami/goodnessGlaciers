import glob
import os
import pandas as pd


_MIGRATED = {'2-D migration processing', '2-D Synthetic Aperture Radar processing',
             '2-D Synthetic Aperture Radar focused processing'}
_PARTIAL  = {'1-D Synthetic Aperture Radar processing',
             'Synthetic Aperture Radar unfocused processing',
             'pik1 (short coherent) processing',
             'MUSIC (Swath) Processing'}


# Output configuration
OUTPUT_BASE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    # 'TEST-ONE-SMUG-region/',
    'SMUG-regions/',
    # 'Ockenden-regions/',
)

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
        # {
        #     'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
        #     'label': 'TEST_Aurora_SB',
        #     'subset': lambda df: df.iloc[8508112:8508112+17528].copy(),
        # },

        {
            'file': 'PRIC_2016_CHA2_AIR_BM3.csv',
            'label': 'PEL_CHA2',
            'subset': lambda df: df.iloc[410823 : 410823 + 54566].copy(),
            'force_id': 'PRIC_2016_CHA2',
        },

        {
            'file': 'BAS_2010_IMAFI_AIR_BM3.csv',
            'label': 'Moller_Stream'
        },

        {
            'file': 'BAS_2018_Thwaites_AIR_BM3.csv',
            'label':'Thwaites_BAS'
        },

        {
          'file': 'AWI_2018_ANIRES_AIR_BM3.csv',
          'label': 'DML_AniRES'
         },
    ###########################################################################
        # {
        #     'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
        #     'label': 'ASB_ICECAP_2010_Fig4_Aurora_SB',
        #     'subset': lambda df, _r={
        #         'lat_min': -76.0, 'lat_max': -71.0,
        #         'lon_min': 105.0, 'lon_max': 125.0,
        #     }: df[
        #         (df['latitude (degree_north)'] >= _r['lat_min']) &
        #         (df['latitude (degree_north)'] <= _r['lat_max']) &
        #         (df['longitude (degree_east)']  >= _r['lon_min']) &
        #         (df['longitude (degree_east)']  <= _r['lon_max'])
        #     ].copy(),
        # },

        # {
        #     'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
        #     'label': 'ASB_ICECAP_2010_Fig2F_Resolution_SH',
        #     'subset': lambda df, _r={
        #         'lat_min': -76.0, 'lat_max': -73.0,
        #         'lon_min': 135.0, 'lon_max': 150.0,
        #     }: df[
        #         (df['latitude (degree_north)'] >= _r['lat_min']) &
        #         (df['latitude (degree_north)'] <= _r['lat_max']) &
        #         (df['longitude (degree_east)']  >= _r['lon_min']) &
        #         (df['longitude (degree_east)']  <= _r['lon_max'])
        #     ].copy(),
        # },

        # {
        #     'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
        #     'label': 'ASB_ICECAP_2010_Fig2G_Highland_A',
        #     'subset': lambda df, _r={
        #         'lat_min': -76.0, 'lat_max': -73.0,
        #         'lon_min': 118.0, 'lon_max': 132.0,
        #     }: df[
        #         (df['latitude (degree_north)'] >= _r['lat_min']) &
        #         (df['latitude (degree_north)'] <= _r['lat_max']) &
        #         (df['longitude (degree_east)']  >= _r['lon_min']) &
        #         (df['longitude (degree_east)']  <= _r['lon_max'])
        #     ].copy(),
        # },

        # {
        #     'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
        #     'label': 'ASB_ICECAP_2010_Fig2H_Golicyna_SH',
        #     'subset': lambda df, _r={
        #         'lat_min': -75.0, 'lat_max': -72.0,
        #         'lon_min': 103.0, 'lon_max': 117.0,
        #     }: df[
        #         (df['latitude (degree_north)'] >= _r['lat_min']) &
        #         (df['latitude (degree_north)'] <= _r['lat_max']) &
        #         (df['longitude (degree_east)']  >= _r['lon_min']) &
        #         (df['longitude (degree_east)']  <= _r['lon_max'])
        #     ].copy(),
        # },

        # {
        #     'file': 'BAS_2012_ICEGRAV_AIR_BM3.csv',
        #     'label': 'Rec_Catch_Fig2D_Recovery_SB',
        #     'subset': lambda df, _r={
        #         'lat_min': -83.5, 'lat_max': -80.5,
        #         'lon_min': -35.0, 'lon_max': -15.0,
        #     }: df[
        #         (df['latitude (degree_north)'] >= _r['lat_min']) &
        #         (df['latitude (degree_north)'] <= _r['lat_max']) &
        #         (df['longitude (degree_east)']  >= _r['lon_min']) &
        #         (df['longitude (degree_east)']  <= _r['lon_max'])
        #     ].copy(),
        # },

        # {
        #     'file': 'NASA_2018_ICEBRIDGE_AIR_BM3.csv',
        #     'label': '2018_Rec_SB_Fig2D_Recovery_SB',
        #     'subset': lambda df, _r={
        #         'lat_min': -83.5, 'lat_max': -80.5,
        #         'lon_min': -35.0, 'lon_max': -15.0,
        #     }: df[
        #         (df['latitude (degree_north)'] >= _r['lat_min']) &
        #         (df['latitude (degree_north)'] <= _r['lat_max']) &
        #         (df['longitude (degree_east)']  >= _r['lon_min']) &
        #         (df['longitude (degree_east)']  <= _r['lon_max'])
        #     ].copy(),
        # },

        # {
        #     'file': 'BAS_2015_POLARGAP_AIR_BM3.csv',
        #     'label': 'POLARGAP_2015_Fig1_Pensacola_Pole',
        #     'subset': lambda df, _r={
        #         'lat_min': -88.0, 'lat_max': -82.0,
        #         'lon_min': -60.0, 'lon_max': -20.0,
        #     }: df[
        #         (df['latitude (degree_north)'] >= _r['lat_min']) &
        #         (df['latitude (degree_north)'] <= _r['lat_max']) &
        #         (df['longitude (degree_east)']  >= _r['lon_min']) &
        #         (df['longitude (degree_east)']  <= _r['lon_max'])
        #     ].copy(),
        # },

        # {
        #     'file': 'BAS_2015_POLARGAP_AIR_BM3.csv',
        #     'label': 'POLARGAP_2015_Fig2C_Hercules_Dome',
        #     'subset': lambda df, _r={
        #         'lat_min': -87.5, 'lat_max': -85.0,
        #         'lon_min': -120.0, 'lon_max': -100.0,
        #     }: df[
        #         (df['latitude (degree_north)'] >= _r['lat_min']) &
        #         (df['latitude (degree_north)'] <= _r['lat_max']) &
        #         (df['longitude (degree_east)']  >= _r['lon_min']) &
        #         (df['longitude (degree_east)']  <= _r['lon_max'])
        #     ].copy(),
        # },
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
