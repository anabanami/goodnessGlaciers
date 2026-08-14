"""
Scatter plot of peak detection statistics vs peak_masking_height_threshold.

Run from v23/; reads from and writes results to v23/peak-masking_threshold/.
Wavelength-detection CSVs are read from each
threshold_*/<region-group>/<region>/ folder, and the matching segment_stats
CSV from the sibling <region-group>/segment_csvs/ folder.

Panel 3 estimates what fraction of the fit range peak masking removes, by
reconstructing the frequency grid, mapping wavelengths to bin indices, and
counting unique masked bins (buffer overlap merged). The split of a
trajectory's peaks across its segments is a modelling assumption rather than
a measurement, and it sets the absolute level of the panel — read the curves
as an estimate and compare them to each other, not to the 30% line. See
compute_masked_fraction.
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
from config import Tee, WINDOW_SIZE, bin_buffer, FIT_BAND_M  # noqa: E402

BASE_DIR = HERE / "peak-masking_threshold"      # this script's data/results folder

# Tee all console output into the results folder alongside the figures
BASE_DIR.mkdir(parents=True, exist_ok=True)
_real_stdout = sys.stdout
sys.stdout = Tee(str(BASE_DIR / "peak-masking_threshold_log.txt"))

# --- Frequency grid: mirrors analyse_sliding_windows (bed_analysis.py).
# WINDOW_SIZE, bin_buffer and the fit band come from config, so an env-overridden or
# retuned run is picked up automatically. N_BINS and the dx floor are still literals in
# that function, so they are duplicated here and guarded below.
N_BINS = 500
DX_MEDIAN_DEFAULT = 15.0  # pipeline dx floor: max_freq = 1/(2*max(dx_median, 15.0))
FIT_MIN_WL, FIT_MAX_WL = FIT_BAND_M  # metres, from config
MIN_FREQ = 1 / WINDOW_SIZE
MAX_FREQ = 1 / (2 * DX_MEDIAN_DEFAULT)

# Reference frequency grid (same geomspace as the analysis). The pipeline builds this
# per segment from that segment's own dx_median; pinning it to the 15 m floor gives the
# finest grid any segment can have, which puts the fewest bins in the fit range and so
# maximises the masked fraction — but only along this axis, and the n_seg split in
# compute_masked_fraction pushes the other way, so the result is not a bound overall.
REF_FREQS = np.geomspace(MIN_FREQ, MAX_FREQ, num=N_BINS)
REF_WAVELENGTHS = 1 / REF_FREQS  # descending (long → short)

FIT_MASK = (REF_WAVELENGTHS >= FIT_MIN_WL) & (REF_WAVELENGTHS <= FIT_MAX_WL)
FIT_INDICES = set(np.where(FIT_MASK)[0].tolist())
N_FIT_BINS = len(FIT_INDICES)

# --- Drift guard for the two values still mirrored. The fit band is imported above, so
# it needs no guard; N_BINS and the dx floor are inline literals in
# analyse_sliding_windows (bed_analysis.py), so read them from that source and warn
# (non-fatal) if this copy has gone stale.
def _prodval(pat, cast=float):
    m = re.search(pat, (ODSA / "bed_analysis.py").read_text())
    return cast(m.group(1)) if m else None
for _label, _mine, _prod in [
    ("grid bins (num=)", N_BINS,            _prodval(r"geomspace\([^)]*num=(\d+)", int)),
    ("dx floor",         DX_MEDIAN_DEFAULT, _prodval(r"max\(dx_median,\s*([\d.]+)\)")),
]:
    if _prod is None:
        print(f"NOTE: could not locate {_label} in bed_analysis.py to cross-check.")
    elif _mine != _prod:
        print(f"WARNING: {_label} = {_mine} here but {_prod} in bed_analysis.py — this mirror is STALE.")


def wavelengths_to_bin_indices(wavelengths):
    """Map detected wavelengths to nearest bin index in the reference grid."""
    indices = []
    for wl in wavelengths:
        idx = np.argmin(np.abs(REF_WAVELENGTHS - wl))
        indices.append(idx)
    return np.array(indices)


def compute_masked_fraction(wavelengths_per_trajectory, n_segments, buffer):
    """
    Segment-weighted mean masked fraction of the fit range for a region.

    The CSV doesn't track which segment each detection belongs to, so per
    trajectory we count unique masked bins (buffered, overlap-merged) over all
    its detections, then divide by its segment count to spread those peaks
    across segments. Trajectories are weighted by segment count, making the
    result the mean of that per-segment estimate over the region's segments.

    Not a bound in either direction: pooling a trajectory's peaks into one
    spectrum inflates the bin count, the 1/n_seg split deflates it, and neither
    is calibrated against the real per-segment spectra. The two are the same
    order of magnitude, so they do not cancel to anything principled — read the
    output as an estimate, not a ceiling.

    Superseded once detections carry a segment column; see the schema note below.
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

        # All of this trajectory's peaks pooled into a single spectrum
        frac = len(masked_bins) / N_FIT_BINS if N_FIT_BINS > 0 else 0
        # Spread back over its segments
        frac_per_seg = frac / n_seg if n_seg > 0 else frac

        # Weighting by n_seg makes the running totals a segment-weighted mean
        total_masked_frac += frac_per_seg * n_seg
        total_segments += n_seg

    return total_masked_frac / total_segments if total_segments > 0 else 0


# --- Collect data from all threshold folders ---
records = []
# Store raw wavelengths per trajectory for proper masked fraction calc
raw_data = {}  # key: (threshold, region) -> {traj_id: [wavelengths]}
# Detections-CSV schema seen per threshold folder; drives the note printed below.
schemas = set()

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

        schemas.add('per-window' if 'segment' in df.columns else 'trajectory-level')

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

# The stored sweep read trajectory-level detections; the pipeline now emits them per
# window. Neither panel is ported, so a re-run needs the patch this note spells out.
if 'per-window' in schemas:
    print("=" * 78)
    print("PER-WINDOW DETECTIONS FOUND — PANELS 2 AND 3 NOT YET PORTED. PATCH BEFORE USE.")
    print("=" * 78)
    if 'trajectory-level' in schemas:
        print("MIXED SWEEP: some threshold_* folders are legacy trajectory-level, some are")
        print("per-window. Curves from the two are not comparable at any threshold. Re-run")
        print("every threshold_* folder on one pipeline version before reading the figure.\n")
    print("What changed: detections moved from the segment fit to per window")
    print("(bed_analysis.py, 'wavelength_detections' built from window_stats). The CSV")
    print("gained segment/window_id/residual_height; the stored sweep had only")
    print("trajectory,wavelength_m,type.\n")
    print("Panel 2 (raw counts): overlapping windows re-count the same physical wavelength,")
    print("so counts inflate by a region-dependent factor — 1.0x to 3.6x across the seven")
    print("regions at threshold 2.0 when this was checked. The threshold response still")
    print("has the right sign, but region-to-region ordering does not survive. Either")
    print("de-duplicate within a segment or relabel the axis as a per-window count.\n")
    print("Panel 3 (masked fraction): compute_masked_fraction splits a trajectory's pooled")
    print("peaks across its segments by 1/n_seg only because the old CSV had no segment")
    print("column. It does now. Replace that with: group detections by")
    print("(trajectory, segment), count unique buffered bins per segment, and divide the")
    print("summed fractions by the segment_stats ROW COUNT, not by the number of segments")
    print("appearing in the detections CSV — segments with no detections contribute a")
    print("real zero and must stay in the denominator. (Hercules had 73 segment rows but")
    print("detections in only 72.) That makes the panel a measurement rather than an")
    print("estimate, so the 30% line becomes meaningful and the 'not a bound in either")
    print("direction' caveat in the sensitivity doc can be dropped.\n")
    print("Not the problem: segment counts. They were identical between the stored sweep")
    print("and the 2026-07-30 production run (130/86/14/25/73/115/68), so the denominator")
    print("is stable — the numerator is what moved.")
    print("=" * 78 + "\n")
elif 'trajectory-level' in schemas:
    print("Legacy trajectory-level detections throughout — stored sweep, self-consistent.\n")

print(data.to_string(index=False))
print(f"\nReference grid: {N_BINS} bins, fit range: {N_FIT_BINS} bins "
      f"({FIT_MIN_WL:.0f}m–{FIT_MAX_WL/1000:.0f}km)")

# --- Compute masked fractions for multiple buffer sizes ---
# bin_buffer (config) is the production value and is always drawn, highlighted.
BUFFER_SIZES = sorted({1, 2, 3, 5, 7, bin_buffer})

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
        # Label the production buffer for every region; the others once, on the first.
        if buf == bin_buffer:
            label = f'{region} (±{buf}, current)'
        elif i == 0:
            label = f'±{buf} buffer'
        else:
            label = None

        current = buf == bin_buffer
        ax3.plot(subset['threshold'], subset['masked_frac'] * 100,
                 marker=markers[i % len(markers)], color=colours[i],
                 linestyle=linestyles.get(buf, '-'),
                 linewidth=1.5 if current else 1.0,
                 markersize=5, alpha=1.0 if current else 0.4, label=label)

ax3.set_xlabel('Peak Masking Height Threshold')
ax3.set_ylabel('Est. Masked Fraction of Fit Range (%)')
ax3.axhline(y=30, color='grey', linestyle='--', alpha=0.5, linewidth=0.8)
ax3.text(0.99, 30, '30%', color='grey', fontsize=8, va='bottom', ha='right',
         transform=ax3.get_yaxis_transform())
ax3.legend(fontsize='x-small', ncol=2)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(BASE_DIR / 'threshold_sensitivity.png', dpi=600, bbox_inches='tight')
print(f"\nSaved: {BASE_DIR / 'threshold_sensitivity.png'}")

sys.stdout.flush()
sys.stdout = _real_stdout
plt.show()
