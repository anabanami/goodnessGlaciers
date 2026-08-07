import os
import re
import sys


# Anchors. This is the pre-fix (unfixed landscape splitting) snapshot, kept in place
# under v23/TESTING_LANDSCAPE_SPLITTING/legacy.../ rather than copied to ODSA/ to run.
# Everything resolves from this file's location, so the codebase runs from any cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
ODSA_ROOT = os.environ.get('ODSA_ROOT') or os.path.abspath(
    os.path.join(_HERE, os.pardir, os.pardir, os.pardir, os.pardir))     # .../ODSA
DATA_ROOT = os.path.join(ODSA_ROOT, 'all_data')                          # shared with ODSA/
RESULTS_ROOT = os.environ.get('ODSA_RESULTS_ROOT') or os.path.abspath(
    os.path.join(_HERE, os.pardir, os.pardir))                           # .../TESTING_LANDSCAPE_SPLITTING

if not os.path.isdir(DATA_ROOT):
    raise RuntimeError(
        f'all_data not found at {DATA_ROOT}. Set ODSA_ROOT to the ODSA checkout root.')


# Window parameters
WINDOW_SIZE = 50000  # metres
STEP_SIZE = WINDOW_SIZE // 2  # 50% overlap
# Hann, to match the current pipeline. The taper itself needed no porting: bed_analysis_22.py
# already carried the hann/tukey branches, only this constant was pinned to rectangular.
# Output names carry a suffix only for non-standard windows, so a Hann run keeps the plain
# _w50km filenames the existing folders and the current Ockenden-regions/ both use.
WINDOW_TYPES = ('rectangular', 'hann', 'tukey')
STANDARD_WINDOW = 'hann'
WINDOW_TYPE = os.environ.get('ODSA_WINDOW_TYPE', STANDARD_WINDOW)
if WINDOW_TYPE not in WINDOW_TYPES:
    raise ValueError(f'unknown WINDOW_TYPE {WINDOW_TYPE!r}; expected one of {WINDOW_TYPES}')

# Peak masking parameters
peak_masking_height_threshold = 2.0
bin_buffer = 5

# Landscape splitting parameters
# MUST be kept in step with OUTPUT_BASE_PATH in loading.py — the two together select the arm:
#   False -> Ockenden-regions-No_Landscape_splitting-TEST   (NO-split arm)
#   True  -> Ockenden-regions-prefix-version                (pre-fix, splitting-ON arm)
# Mismatching them writes one arm's results into the other arm's folder, silently.
ENABLE_LANDSCAPE_SPLITTING = False
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
