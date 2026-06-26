"""
Scatter plot of peak detection statistics vs peak_masking_height_threshold.

Run from v23/; reads from and writes results to v23/peak-masking_threshold/.
Wavelength-detection CSVs are read from each
threshold_*/<region-group>/<region>/ folder, and the matching segment_stats
CSV from the sibling <region-group>/segment_csvs/ folder.

Panel 3 computes masked fraction properly by reconstructing the frequency
grid, mapping wavelengths to bin indices, and counting unique masked bins
(accounting for buffer overlap).
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent          # .../v23
ODSA = HERE.parent                              # .../ODSA
sys.path.insert(0, str(ODSA))
from config import Tee                          # noqa: E402

BASE_DIR = HERE / "peak-masking_threshold"      # this script's data/results folder

# Tee all console output into the results folder alongside the figures
BASE_DIR.mkdir(parents=True, exist_ok=True)
_real_stdout = sys.stdout
sys.stdout = Tee(str(BASE_DIR / "peak-masking_threshold_log.txt"))

# --- Frequency grid parameters (must match bed_analysis_19.py) ---
WINDOW_SIZE = 50000  # metres
N_BINS = 500
DX_MEDIAN_DEFAULT = 15.0  # conservative fallback (smallest typical spacing)
MIN_FREQ = 1 / WINDOW_SIZE
MAX_FREQ = 1 / (2 * DX_MEDIAN_DEFAULT)

# Reference frequency grid (same geomspace as the analysis)
REF_FREQS = np.geomspace(MIN_FREQ, MAX_FREQ, num=N_BINS)
REF_WAVELENGTHS = 1 / REF_FREQS  # descending (long → short)

# Fit range mask: 250m to 50km
FIT_MASK = (REF_WAVELENGTHS >= 250) & (REF_WAVELENGTHS <= 50000)
FIT_INDICES = np.where(FIT_MASK)[0]
N_FIT_BINS = len(FIT_INDICES)


def wavelengths_to_bin_indices(wavelengths):
    """Map detected wavelengths to nearest bin index in the reference grid."""
    indices = []
    for wl in wavelengths:
        idx = np.argmin(np.abs(REF_WAVELENGTHS - wl))
        indices.append(idx)
    return np.array(indices)


def compute_masked_fraction(wavelengths_per_trajectory, n_segments, buffer):
    """
    Compute average masked fraction across segments for a region.

    Since the CSV doesn't track which segment each detection belongs to,
    we compute per-trajectory masked bins (unique, with overlap) and
    average across the trajectory's segments. This is an upper bound
    because cross-segment peaks don't actually share a spectrum.
    """
    total_masked_frac = 0
    total_segments = 0

    for traj_id, wls in wavelengths_per_trajectory.items():
        n_seg = n_segments.get(traj_id, 1)

        # Map wavelengths to bin indices
        peak_bins = wavelengths_to_bin_indices(wls)

        # Apply buffer and collect unique masked bins within fit range
        masked_bins = set()
        for p in peak_bins:
            for offset in range(-buffer, buffer + 1):
                b = p + offset
                if b in FIT_INDICES:
                    masked_bins.add(b)

        # This trajectory's masked fraction (averaged over its segments)
        # Upper bound: all peaks treated as one spectrum
        frac = len(masked_bins) / N_FIT_BINS if N_FIT_BINS > 0 else 0
        # Scale down by segments (peaks are spread across segments, not all in one)
        frac_per_seg = min(frac / n_seg, 1.0) if n_seg > 0 else frac

        total_masked_frac += frac_per_seg * n_seg
        total_segments += n_seg

    return total_masked_frac / total_segments if total_segments > 0 else 0


# --- Collect data from all threshold folders ---
records = []
# Store raw wavelengths per trajectory for proper masked fraction calc
raw_data = {}  # key: (threshold, region) -> {traj_id: [wavelengths]}

for threshold_dir in sorted(BASE_DIR.glob('threshold_*')):
    match = re.search(r'threshold_([\d.]+)', threshold_dir.name)
    if not match:
        continue
    threshold = float(match.group(1))

    for csv_path in threshold_dir.glob('*/*/*_wavelength_detections.csv'):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        region_folder = csv_path.parent.name
        region_name = re.sub(r'_w\d+km.*$', '', region_folder)

        confirmed = df[df['type'] == 'confirmed']['wavelength_m']
        candidate = df[df['type'] == 'candidate']['wavelength_m']
        all_wavelengths = df['wavelength_m']

        # Count segments per trajectory from segment_stats CSV. Segment stats
        # live in the sibling segment_csvs/ folder, one file per region, named
        # with the same prefix as the wavelength-detections CSV.
        seg_name = csv_path.name.replace('_wavelength_detections.csv',
                                         '_segment_stats.csv')
        seg_csv = list(csv_path.parent.parent.glob(f'segment_csvs/{seg_name}'))
        n_segments_total = 1
        seg_per_traj = {}
        if seg_csv:
            seg_df = pd.read_csv(seg_csv[0])
            n_segments_total = seg_df.shape[0]
            seg_per_traj = seg_df.groupby('trajectory').size().to_dict()

        # Store wavelengths per trajectory for masked fraction calc
        wl_per_traj = {}
        for traj_id, group in df.groupby('trajectory'):
            wl_per_traj[traj_id] = group['wavelength_m'].values

        raw_data[(threshold, region_name)] = {
            'wl_per_traj': wl_per_traj,
            'seg_per_traj': seg_per_traj,
        }

        records.append({
            'threshold': threshold,
            'region': region_name,
            'n_confirmed': len(confirmed),
            'n_candidate': len(candidate),
            'n_total': len(all_wavelengths),
            'max_confirmed_wl': confirmed.max() if len(confirmed) > 0 else np.nan,
            'median_confirmed_wl': confirmed.median() if len(confirmed) > 0 else np.nan,
            'max_all_wl': all_wavelengths.max(),
            'n_segments': n_segments_total,
        })

data = pd.DataFrame(records)
if data.empty:
    print("No wavelength detection CSVs found.")
    sys.stdout.flush()
    sys.stdout = _real_stdout
    exit()

print(data.to_string(index=False))
print(f"\nReference grid: {N_BINS} bins, fit range: {N_FIT_BINS} bins (250m–50km)")

# --- Compute masked fractions for multiple buffer sizes ---
BUFFER_SIZES = [1, 2, 3, 5, 7]

masked_frac_results = []
for (threshold, region), rd in raw_data.items():
    for buf in BUFFER_SIZES:
        frac = compute_masked_fraction(rd['wl_per_traj'], rd['seg_per_traj'], buf)
        masked_frac_results.append({
            'threshold': threshold,
            'region': region,
            'buffer': buf,
            'masked_frac': frac,
        })

mf_data = pd.DataFrame(masked_frac_results)

# --- Plotting ---
regions = sorted(data['region'].unique())
colours = plt.cm.tab10(np.linspace(0, 1, len(regions)))
markers = ['o', 's', '^', 'D', 'v', 'P', 'X']

fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)

# Panel 1: Max confirmed wavelength vs threshold
ax1 = axes[0]
for i, region in enumerate(regions):
    subset = data[data['region'] == region].sort_values('threshold')
    ax1.plot(subset['threshold'], subset['max_confirmed_wl'] / 1000,
             marker=markers[i % len(markers)], color=colours[i],
             label=region, linewidth=1.5, markersize=7)

ax1.set_ylabel('Max Confirmed Wavelength (km)')
ax1.set_title('Peak Masking Threshold Sensitivity')
ax1.legend(fontsize='small')
ax1.grid(True, alpha=0.3)

# Panel 2: Number of confirmed detections vs threshold
ax2 = axes[1]
for i, region in enumerate(regions):
    subset = data[data['region'] == region].sort_values('threshold')
    ax2.plot(subset['threshold'], subset['n_confirmed'],
             marker=markers[i % len(markers)], color=colours[i],
             label=region, linewidth=1.5, markersize=7)

ax2.set_ylabel('Number of Confirmed Detections')
ax2.legend(fontsize='small')
ax2.grid(True, alpha=0.3)

# Panel 3: Masked fraction vs threshold, one line per (region, buffer)
ax3 = axes[2]
# Use line style to distinguish buffer sizes
linestyles = {1: ':', 2: '--', 3: '-.', 5: '-', 7: (0, (3, 1, 1, 1))}

for i, region in enumerate(regions):
    for buf in BUFFER_SIZES:
        subset = mf_data[(mf_data['region'] == region) & (mf_data['buffer'] == buf)].sort_values('threshold')
        label = f'{region} (±{buf})' if i == 0 or buf == 5 else None
        # Only label buffer=5 (current default) for all regions, others just for first region
        if buf == 5:
            label = f'{region} (±{buf}, current)'
        elif i == 0:
            label = f'±{buf} buffer'
        else:
            label = None

        alpha = 1.0 if buf == 5 else 0.4
        ax3.plot(subset['threshold'], subset['masked_frac'] * 100,
                 marker=markers[i % len(markers)], color=colours[i],
                 linestyle=linestyles.get(buf, '-'),
                 linewidth=1.5 if buf == 5 else 1.0,
                 markersize=5, alpha=alpha, label=label)

ax3.set_xlabel('Peak Masking Height Threshold')
ax3.set_ylabel('Est. Masked Fraction of Fit Range (%)')
ax3.axhline(y=30, color='grey', linestyle='--', alpha=0.5, linewidth=0.8)
ax3.text(10.2, 31, '30%', color='grey', fontsize=8, va='bottom')
ax3.legend(fontsize='x-small', ncol=2)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(BASE_DIR / 'threshold_sensitivity.png', dpi=600, bbox_inches='tight')
print(f"\nSaved: {BASE_DIR / 'threshold_sensitivity.png'}")

sys.stdout.flush()
sys.stdout = _real_stdout
plt.show()
