import os
import re
import sys


# Window parameters (env-overridable for the window-size sweep; defaults unchanged)
WINDOW_SIZE = int(os.environ.get('ODSA_WINDOW_SIZE', 50000))  # metres
STEP_SIZE = WINDOW_SIZE // 2  # 50% overlap
WINDOW_TYPE = os.environ.get('ODSA_WINDOW_TYPE', 'rectangular')

# Peak masking parameters
peak_masking_height_threshold = 2.0
bin_buffer = 5

# Radar migration / processing-status palette (set in loading._parse_processing_flag).
# Migration affects bed-geometry fidelity, so any spectral metric (beta, PSD amplitude,
# roughness) carries this caveat; shared by plotting.py and bed_character.py.
PROCESSING_FLAG_COLORS = {
    'migrated':             'C2',
    'partial':              'C1',
    'unmigrated_or_unknown': 'C3',
}

# Spectral-reliability caveat keyed on migration status. Shared by the scripts that
# report beta-derived quantities (bed_character, beta_intercept_check, weighted_anisotropy).
PROCESSING_FLAG_NOTE = {
    'migrated':              'migrated — β reliable',
    'partial':               'PARTIAL migration — β may be biased by residual diffraction',
    'unmigrated_or_unknown': 'UNMIGRATED/unknown — β classification suspect (diffraction tails)',
}


def processing_flag_of(df):
    """Modal processing flag of a window/segment frame (None for pre-flag CSVs)."""
    if 'processing_flag' in df.columns and df['processing_flag'].notna().any():
        return df['processing_flag'].dropna().mode().iloc[0]
    return None

# Landscape splitting parameters
SMOOTHING_LENGTH = WINDOW_SIZE  # metres
GRADIENT_THRESHOLD = 15 # m/km

class Tee:
    """Write to both stdout and a log file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w')
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
    def flush(self):
        self.terminal.flush()
        self.log.flush()


def get_region_folder(dataset_name):
    is_2008 = '_2008_' in dataset_name
    match = re.search(r'(Fig\w+_\w+)$', dataset_name)
    if match:
        region = match.group(1)
        return f'2008_{region}' if is_2008 else region
    region = dataset_name.split('_')[-1]
    return f'2008_{region}' if is_2008 else dataset_name


def ensure_output_dirs(base_path, region_folder):
    region_path = os.path.join(base_path, region_folder)
    trajectories_path = os.path.join(region_path, 'trajectories')
    psd_path = os.path.join(region_path, 'psd')
    os.makedirs(trajectories_path, exist_ok=True)
    os.makedirs(psd_path, exist_ok=True)
    return {'region': region_path, 'trajectories': trajectories_path, 'psd': psd_path}
