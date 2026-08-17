import glob
import os
import sys
import pandas as pd

# ===== Ockenden-regions ONLY =====================================================
# These subset functions exist FOR THE EXCLUSIVE USE of the Ockenden-region
# target_files below. They are dataset-specific and are defined in
# ockenden_coords.py (the front of the pipeline: ockenden_coords.py -> loading.py
# -> everything else), which also prints the "COPY FOR loading.py" entries that
# reference them. If you switch to other datasets, this import (and the entries
# that use it) can be removed.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'all_data', 'Ockenden'))
from ockenden_coords import (  # noqa: E402
    _ps71_subset, _ppb_core_subset, _ps71_lowrelief_subset, _soar_ppb_subset)
# =================================================================================


# Bedmap #history strings verbatim. The same processing chain is spelled several ways:
# 2D and 2-D, focused/focussed/focusing, and one entry doubles the trailing word.
_MIGRATED = {'2-D migration processing', '2-D Synthetic Aperture Radar processing',
             '2-D Synthetic Aperture Radar focused processing',
             '2-D Synthetic Aperture Radar focusing processing',
             '2D Synthetic Aperture Radar processing',
             '2D Synthetic Aperture Radar focussed processing'}
_PARTIAL  = {'1-D Synthetic Aperture Radar processing',
             'Synthetic Aperture Radar unfocused processing',
             'pik1 (short coherent) processing',
             'MUSIC (Swath) Processing', 'MUSIC (Swath) Processing processing'}
_WARNED = set()

# Output configuration (ODSA_OUTPUT_BASE env override isolates sweep runs)
OUTPUT_BASE_PATH = os.environ.get('ODSA_OUTPUT_BASE') or os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    # 'Ockenden-regions/',
    'individual_region_TEST/'
    # 'v23/peak-masking_threshold/threshold_10.0/Ockenden-regions-sensitivityTEST'
    # 'new/PPB_soar'
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
                # A SAR or migration string that matches neither set states no dimension,
                # so it stays unknown and says so rather than passing silently.
                low = hist.lower()
                if ('synthetic aperture radar' in low or 'migration' in low) \
                        and hist not in _WARNED:
                    _WARNED.add(hist)
                    print(f"  unlisted SAR/migration history, read as unknown: {hist!r}")
                return 'unmigrated_or_unknown'
    return 'unmigrated_or_unknown'


def _warn_degenerate_trajectories(label, df, min_pts_per_traj=5, max_singleton_frac=0.5):
    """Flag datasets whose trajectory_id names rows rather than flight lines.

    15 Bedmap1/2 campaigns (BEDMAP1_1966-2000, UTIG_2004_AGASEA, UTIG_2008_ICECAP,
    NASA_2004/2010_ICEBRIDGE, UTIG_1999_SOAR-LVS-WLK, BGR_2002_PCMEGA, ...) number
    trajectory_id 1..N per ROW. The bed elevations are valid so they sail through
    the -9999 filters above, but every "track" is one point and every along-track
    metric downstream is meaningless. No Bedmap3 campaign does this.
    See all_data/Ockenden/scan_bed_validity.{py,log,csv} for the full audit.

    Warns rather than drops: a small region box legitimately clips long lines to a
    few points each, and that is the user's call, not the loader's.
    """
    sizes = df.groupby('trajectory_id').size()
    per_traj, singleton_frac = sizes.median(), (sizes == 1).mean()
    if per_traj >= min_pts_per_traj and singleton_frac <= max_singleton_frac:
        return
    print(f"  {label}: {len(sizes)} trajectories for {len(df)} points "
          f"(median {per_traj:.0f} pts/track, {singleton_frac:.0%} single-point). "
          f"trajectory_id looks like a row counter, not a flight line — "
          f"along-track metrics on this dataset will be meaningless.")


def load_datasets():
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'all_data/bedmap3_data/bedmap*/Results/')
    all_dfs = []

    target_files = [
        # =================================================================
        # [Ockenden_2026] regions — PS71 bounds from Zenodo [Ockenden_2025_data]
        # =================================================================
        
        # SELECTIVE EROSION: Pensacola-Pole Basin — core square (right portion of
        # the POLARGAP fan incl. the radial convergence node). Matches the
        # hand-drawn black square on the overview map; subset of the full PPB box.
        {
            'file': 'BAS_2015_POLARGAP_AIR_BM3.csv',
            'label': 'POLARGAP_2015_Pensacola_Pole',
            'subset': _ppb_core_subset,
        },
        # {
        #   'file': 'UTIG_1999_SOAR-LVS-WLK_AIR_BM2.csv',
        #   'label': 'SOAR_1999_Pensacola_Pole_SOAR_BM2',
        #   'subset': _soar_ppb_subset,
        # },

        # LOW-RELIEF: Aurora SB filtered to Ockenden low-relief cells
        {
            'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
            'label': 'ASB_ICECAP_2010_Fig4C_Aurora_SB_lowrelief',
            'subset': lambda df: _ps71_lowrelief_subset(
                df, [1.05e6, 2.20e6, -0.80e6, 0.20e6]),
        },

        # # SELECTIVE EROSION/LOW RELIEF: Aurora Subglacial Basin
        # {
        #  'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
        #  'label': 'ASB_ICECAP_2010_Fig4C_Aurora_SB_square',
        #  'subset': lambda df, _b=[1100000.0, 1400000.0, -780000.0, -480000.0]: _ps71_subset(df, _b),
        # },

        # LOW-RELIEF / SELECTIVE EROSION: Maud Subglacial Basin
        {
            'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
            'label': 'ASB_ICECAP_2010_Fig2A_Maud_SB',
            'subset': lambda df: _ps71_subset(
                df, [0.15e6, 0.45e6, 1.025e6, 1.325e6]),
        },

        # LOW-RELIEF / SELECTIVE EROSION: Recovery Subglacial Lakes
        {
            'file': 'BAS_2012_ICEGRAV_AIR_BM3.csv',
            'label': 'Rec_Catch_Fig2D_Recovery_SL',
            'subset': lambda df: _ps71_subset(
                df, [0.0e6, 0.30e6, 0.6e6, 0.9e6]),
        },

        # ALPINE/SELECTIVE EROSION/LOW RELIEF: Highland A
        {
            'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
            'label': 'ASB_ICECAP_2010_Fig2G_Highland_A',
            'subset': lambda df: _ps71_subset(
                df, [1.90e6, 2.20e6, -0.725e6, -0.425e6]),
        },

        # ALPINE/SELECTIVE EROSION/LOW RELIEF: Golicyna Subglacial Mountains
        {
            'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
            'label': 'ASB_ICECAP_2010_Fig2H_Golicyna_SM',
            'subset': lambda df: _ps71_subset(
                df, [2.15e6, 2.45e6, -0.5e6, -0.2e6]),
        },

        # ALPINE/SELECTIVE EROSION: Hercules Dome
        {
            'file': 'BAS_2015_POLARGAP_AIR_BM3.csv',
            'label': 'POLARGAP_2015_Fig2C_Hercules_Dome',
            'subset': lambda df: _ps71_subset(
                df, [-0.6e6, -0.3e6, -0.23e6, 0.07e6]),
        },

        # # ALPINE/LOW RELIEF: Dome C
        # {
        # 'file': 'BAS_2005_WISE-ISODYN_AIR_BM2.csv',
        # 'label': 'BM2_DomeC_SW_sq_WISE_ISODYN',
        # 'subset': lambda df, _b=[1020000.0, 1320000.0, -1237000.0, -937000.0]:_ps71_subset(df, _b),
        # },
        # {
        # 'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
        # 'label': 'BM3_DomeC_SW_sq_ICECAP',
        # 'subset': lambda df, _b=[1020000.0, 1320000.0, -1237000.0, -937000.0]:_ps71_subset(df, _b),
        # },
        # {
        # 'file': 'NASA_2013_ICEBRIDGE_AIR_BM3.csv',
        # 'label': 'BM3_DomeC_SW_sq_ICEBRIDGE',
        # 'subset': lambda df, _b=[1020000.0, 1320000.0, -1237000.0, -937000.0]:_ps71_subset(df, _b),
        # },

        # # ALPINE/SELECTIVE EROSION: Dronning Maud Land
        # {
        # 'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
        # 'label': 'BM3_DML_3E_sq_ICECAP',
        # 'subset': lambda df, _b=[-50000.0, 250000.0, 1620000.0, 1920000.0]: _ps71_subset(df, _b),
        # },

    ]

    # ODSA_REGION_FILTER: comma-separated label substrings (used by the window-size sweep)
    _rf = os.environ.get('ODSA_REGION_FILTER')
    if _rf:
        keys = [k.strip().lower() for k in _rf.split(',') if k.strip()]
        target_files = [t for t in target_files if any(k in t['label'].lower() for k in keys)]
        print(f"  [ODSA_REGION_FILTER] {len(target_files)} region(s): {[t['label'] for t in target_files]}")

    file_cache = {}

    for item in target_files:
        filename = item['file']
        label = item['label']
        matches = glob.glob(os.path.join(base_path, filename))
        if not matches:
            print(f"Warning: {filename} not found. Skipping.")
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
                print(f"{label} loaded: {len(df)} rows (Filtered {initial_len - len(df)} nulls) [{pflag}]")
                _warn_degenerate_trajectories(label, df)
                all_dfs.append({'name': label, 'data': df})
            else:
                print(f"---{label} resulted in 0 rows.---")

        except Exception as e:
            print(f"---Error loading {label}: {e}---")

    del file_cache
    return all_dfs
