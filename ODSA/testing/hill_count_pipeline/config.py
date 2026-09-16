import os
import re
import sys


# This is a frozen copy of the pipeline that produced the hill-count threshold sweep, and
# it sits several folders below the ODSA root rather than in it. Data and output paths in
# the live pipeline resolve against the file's own directory or the working directory,
# neither of which is right here, so they resolve against ODSA_ROOT instead: the nearest
# ancestor holding all_data/. Path resolution is the only thing that differs from the
# live pipeline at the time of the sweep; nothing that touches a number was changed.
def _find_odsa_root(start):
    p = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(p, 'all_data')):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            return os.path.abspath(start)
        p = parent


ODSA_ROOT = _find_odsa_root(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOT_DIR = os.path.dirname(os.path.abspath(__file__))


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
peak_masking_height_threshold = 2.0 # 2.0 is standard
bin_buffer = 5
# False bypasses window-level peak masking (segment two-pass unaffected); used by
# the window-type taper-isolation ladder. True is production — restore after any test.
WINDOW_MASK = True

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

# Hill count (Ockenden_2026 bumpiness, transect form): a point is a hill if it is the
# max of a box this long and that box's relief clears the threshold. All four published
# thresholds are emitted per window; the sensitivity sweep decides which enters the vector.
HILL_BOX_M = 5000.0
HILL_RELIEF_THRESHOLDS = (20, 50, 100, 250)
# The gate adopted from this sweep. Added after the fact so the sensitivity script can
# make its adopted-gate cross-check without reaching into the live pipeline. Nothing in
# this snapshot reads it, so it changes no computed value.
HILL_THRESHOLD_M = 20

# Landscape splitting parameters
SMOOTHING_LENGTH = WINDOW_SIZE  # metres
GRADIENT_THRESHOLD = 15 # 15 m/km is standard

# Flow-direction stencil half-width, as a multiple of local ice thickness (used by
# REMA_extractor.extract_rema_flow_vector to measure the surface slope that sets the
# modelled flow bearing, hence the incidence angle theta). Production is 5; McCormack
# et al. (2019) recommend ~10. Env-overridable for the stencil sensitivity sweep.
STENCIL_FACTOR = float(os.environ.get('ODSA_STENCIL_FACTOR', 5))

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
