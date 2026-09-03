import os
import re
import sys


# Frozen snapshot of the ODSA/ pipeline (taken 2026-07-14), carrying BOTH 2026-06-11 fixes
# (is_transition flag, single-window degeneracy bypass). It exists solely to produce the
# NO-landscape-splitting control arm, which ODSA/ itself cannot: splitting is unconditional
# there and ODSA/ is kept unmodified. Unlike the sibling pre-fix legacy codebase, this arm's
# segment-level beta is clean, so it is the one valid at BOTH window and segment scale.
#
# Anchors: everything resolves from this file's location, so it runs from any cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
ODSA_ROOT = os.environ.get('ODSA_ROOT') or os.path.abspath(
    os.path.join(_HERE, os.pardir, os.pardir, os.pardir, os.pardir))     # .../ODSA
DATA_ROOT = os.path.join(ODSA_ROOT, 'all_data')                          # shared with ODSA/
RESULTS_ROOT = os.environ.get('ODSA_RESULTS_ROOT') or os.path.abspath(
    os.path.join(_HERE, os.pardir, os.pardir))                           # .../TESTING_LANDSCAPE_SPLITTING

if not os.path.isdir(DATA_ROOT):
    raise RuntimeError(
        f'all_data not found at {DATA_ROOT}. Set ODSA_ROOT to the ODSA checkout root.')


# Window parameters (env-overridable for the sensitivity sweeps)
WINDOW_SIZE = int(os.environ.get('ODSA_WINDOW_SIZE', 50000))  # metres
STEP_SIZE = WINDOW_SIZE // 2  # 50% overlap

# The production taper. Hann suppresses the spectral leakage that biases beta low in
# proportion to beta (see 'Sensitivity analysis - window type'); 50% overlap satisfies
# COLA, so adjacent Hann tapers sum to unity. Output names carry a suffix only for
# non-standard windows, so the standard run keeps stable filenames.
WINDOW_TYPES = ('rectangular', 'hann', 'tukey')
STANDARD_WINDOW = 'hann'
WINDOW_TYPE = os.environ.get('ODSA_WINDOW_TYPE', STANDARD_WINDOW)
if WINDOW_TYPE not in WINDOW_TYPES:
    raise ValueError(f'unknown WINDOW_TYPE {WINDOW_TYPE!r}; expected one of {WINDOW_TYPES}')

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

# OFF by default — the whole point of this snapshot. ODSA_SPLIT=1 restores the production
# (splitting-ON) behaviour, which must reproduce ODSA/Ockenden-regions/ exactly; that is the
# fidelity check that this snapshot has not drifted from the pipeline it was copied from.
ENABLE_LANDSCAPE_SPLITTING = os.environ.get('ODSA_SPLIT', '0').lower() in ('1', 'true', 'yes')

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
