import os, sys, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from scipy.stats import norm, gaussian_kde
from config import (Tee, WINDOW_SIZE, PROCESSING_FLAG_NOTE as _FLAG_NOTE,
                    processing_flag_of as region_flag)
from plotting import flag_suptitle as _flag_suptitle, flag_title as _flag_title

"""
Bed character classification from window-level spectral data.

Classifies each sliding window by its power-law exponent (beta) into
bed character classes, summarises per segment and per region, and
produces a diagnostic visualisation.

Usage:
  python bed_character.py                          # interactive, discovers from window_csvs/
  python bed_character.py Pensacola                # partial match
  python bed_character.py window_csvs/some.csv     # direct path
  python bed_character.py individual_region_TEST   # walk a tree, then compare
  python bed_character.py --compare individual_region_TEST   # compare only, no reprocessing
"""

# Beta thresholds for bed character classification
BED_CLASSES = [
    ('chaotic',       -np.inf, 1.5),
    ('hard',          1.5,     2.0),
    ('transitional',  2.0,     2.5),
    ('soft',          2.5,     np.inf),
]

# Relief thresholds — anchored to Ockenden et al. (2026) reference regions
RELIEF_CLASSES = [
    ('flat',        -np.inf, 350),
    ('subdued',     350,     800),
    ('mountainous', 800,     np.inf),
]

# The two finite class boundaries, exposed so the derivation tool
# (v23/relief_distribution.py) can cross-check its sweep value against the
# adopted production value. Kept in step by that check, not by a shared source.
RELIEF_THRESHOLDS = [hi for _, _, hi in RELIEF_CLASSES if np.isfinite(hi)]

# Elevation thresholds — absolute bed elevation (m a.s.l.) [Siegert_2004, Frederick_2016]
ELEVATION_CLASSES = [
    ('submerged',  -np.inf, 0),
    ('emerged',   0,       1000),
    ('elevated',   1000,    np.inf),
]

BED_COLORS = {
    'chaotic':      '#d62728',
    'hard':         '#ff7f0e',
    'transitional': '#9467bd',
    'soft':         '#1f77b4',
}

# Output configuration - nested inside region output from loading.py
from loading import OUTPUT_BASE_PATH as _REGION_BASE
OUTPUT_DIR = os.path.join(_REGION_BASE, 'bed_character/')


CLASS_ORDER = [name for name, _, _ in BED_CLASSES]
BED_EDGES = np.array([BED_CLASSES[0][1]] + [hi for _, _, hi in BED_CLASSES])  # [-inf, 1.5, 2.0, 2.5, inf]
P_COLS = [f'p_{name}' for name in CLASS_ORDER]

# Excess beta uncertainty beyond the formal PSD-fit error, added in quadrature.
# Stays 0.0: beta_uncertainty is a lower bound (the geomspace frequency grid
# oversamples the window's Fourier resolution and the Hann taper correlates
# adjacent bins), sigma_fit = 0.051 on the masked production fit; how much
# higher the true sigma runs is not measured. Class composition is
# insensitive to sigma across the measured bracket and any plausible
# value. Per-window class_confidence is NOT, so don't quote it as measured.
# See v23/beta_sigma_calibration.py and "Sensitivity test - beta_sigma".
SIGMA_EXTRA = 0.0

# Band-truncation offset: segments shorter than WINDOW_SIZE are fit over a narrower
# band and read this much steeper. Measured at Pensacola, relief-matched (v23 §9);
# NOT applied to any exported beta. Used here only to show, per panel, where the
# offline debias would land the region median relative to a class break.
TRUNC_OFFSET = 0.30
ON_BREAK_MARGIN = 0.10   # debiased median this close to a break -> no bare class label
# The label is the class of the MEDIAN, so it can differ from the class holding the
# most windows, and neither test implies the other: a region can sit far from every
# break and still be a near-tie between two classes.
TIE_MARGIN = 0.10        # top two class shares this close -> plurality is not stable


def add_soft_membership(df, sigma_extra=None):
    """Per-window class membership P(class | beta, sigma).

    The class boundaries are conventions on a continuous variable, and beta is a
    fit with a standard error, so a hard label discards real information near a
    threshold. Treating beta as Normal(beta, sigma) gives each window a fractional
    membership in every class; sigma -> 0 recovers the hard threshold assignment.

    sigma = sqrt(beta_uncertainty**2 + sigma_extra**2); see SIGMA_EXTRA.
    """
    sigma_extra = SIGMA_EXTRA if sigma_extra is None else sigma_extra
    b = df['beta'].to_numpy(float)
    s = (df['beta_uncertainty'].to_numpy(float) if 'beta_uncertainty' in df.columns
         else np.zeros(len(df)))
    s = np.sqrt(np.nan_to_num(s) ** 2 + sigma_extra ** 2)
    ok = np.isfinite(s) & (s > 0)

    P = np.zeros((len(df), len(CLASS_ORDER)))
    if ok.any():
        P[ok] = np.diff(norm.cdf(BED_EDGES[None, :], b[ok, None], s[ok, None]), axis=1)
    if (~ok).any():  # no usable sigma -> degenerate one-hot (hard assignment)
        P[~ok, np.searchsorted(BED_EDGES[1:-1], b[~ok], side='right')] = 1.0

    df[P_COLS] = P
    df['bed_class'] = [CLASS_ORDER[i] for i in P.argmax(1)]  # MAP label, for description
    df['class_confidence'] = P.max(1)                        # 1.0 = unambiguous, 0.5 = coin-flip
    return df


def expected_fractions(g):
    """Expected class fractions for a group of windows, with Poisson-binomial SE."""
    P = g[P_COLS].to_numpy()
    n = len(g)
    return P.mean(0), np.sqrt((P * (1 - P)).sum(0)) / n


def classify_relief(relief):
    for name, lo, hi in RELIEF_CLASSES:
        if lo <= relief < hi:
            return name
    return 'unknown'


def classify_elevation(elev):
    for name, lo, hi in ELEVATION_CLASSES:
        if lo <= elev < hi:
            return name
    return 'unknown'


def discover_window_csvs(directory='window_csvs'):
    csvs = {}
    for f in sorted(glob.glob(os.path.join(directory, '*_window_stats.csv'))):
        region = os.path.basename(f).replace('_window_stats.csv', '')
        csvs[region] = f
    return csvs


def select_region(csvs):
    if not csvs:
        print("No window_stats CSVs found in window_csvs/")
        return None
    if len(csvs) == 1:
        region = list(csvs.keys())[0]
        print(f"Found 1 region: {region}")
        return region

    sorted_regions = sorted(csvs.keys())
    print(f"\nFound {len(csvs)} regions:")
    for i, r in enumerate(sorted_regions, 1):
        print(f"  {i}. {r}")
    print(f"  0. Process ALL regions")

    while True:
        try:
            choice = int(input("\nSelect region number (or 0 for all): ").strip())
            if choice == 0:
                return 'ALL'
            if 1 <= choice <= len(sorted_regions):
                return sorted_regions[choice - 1]
            print("Invalid choice.")
        except ValueError:
            print("Please enter a number.")


def segment_summary(df):
    """Per-segment bed character summary.

    Class composition is the expectation over soft memberships, not a hard count:
    a window straddling a threshold contributes fractionally to both classes.
    """
    rows = []
    for (traj, seg), g in df.groupby(['trajectory', 'segment']):
        n = len(g)
        frac, frac_se = expected_fractions(g)
        dominant = CLASS_ORDER[int(frac.argmax())]
        class_str = ', '.join(f"{c} {f * n:.1f}" for c, f in zip(CLASS_ORDER, frac)
                              if f * n >= 0.05)

        row = {
            'trajectory': traj,
            'segment': seg,
            'n_windows': n,
            'beta_median': g['beta'].median(),
            'beta_iqr': g['beta'].quantile(0.75) - g['beta'].quantile(0.25) if n > 1 else np.nan,
            'relief_median': g['relief_m'].median(),
            'bed_class': dominant,
            'agreement': float(frac.max()),                     # expected fraction in dominant class
            'class_confidence': float(g['class_confidence'].mean()),
            'class_detail': class_str,
            'relief_class': classify_relief(g['relief_m'].median()),
        }
        row.update({f'frac_{c}': f for c, f in zip(CLASS_ORDER, frac)})
        row.update({f'frac_{c}_se': e for c, e in zip(CLASS_ORDER, frac_se)})
        if 'psd_amplitude_1km' in g.columns:
            row['psd_amp_1km_median'] = g['psd_amplitude_1km'].median()
            row['psd_amp_1km_iqr'] = (g['psd_amplitude_1km'].quantile(0.75)
                                      - g['psd_amplitude_1km'].quantile(0.25)) if n > 1 else np.nan
        if 'bed_elev_mean' in g.columns:
            row['bed_elev_median'] = g['bed_elev_mean'].median()
            row['elevation_class'] = classify_elevation(g['bed_elev_mean'].median())
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(summary, region_name, df, pflag=None):
    """Print terminal table."""
    print(f"\n{'='*80}")
    print(f"  BED CHARACTER: {region_name}")
    if pflag:
        print(f"  Processing: {_FLAG_NOTE.get(pflag, pflag)}")
    print(f"{'='*80}")
    print(f"{'Traj':>8s} {'Seg':>4s} {'n':>3s} {'β_med':>6s} {'β_IQR':>6s} "
          f"{'Relief':>7s} {'A_1km':>6s} {'Bed Class':>14s} {'Agree':>6s} {'Conf':>5s}  Detail")
    print(f"{'-'*98}")
    for _, r in summary.iterrows():
        iqr = f"{r['beta_iqr']:.2f}" if np.isfinite(r['beta_iqr']) else '—'
        agree = f"{r['agreement']:.0%}" if r['n_windows'] > 1 else '(1win)'
        conf = f"{r['class_confidence']:.0%}"
        amp = f"{r['psd_amp_1km_median']:.1f}" if 'psd_amp_1km_median' in r and np.isfinite(r['psd_amp_1km_median']) else '—'
        print(f"{r['trajectory']:>8} {r['segment']:>4.0f} {r['n_windows']:>3.0f} "
              f"{r['beta_median']:>6.2f} {iqr:>6s} {r['relief_median']:>7.0f} "
              f"{amp:>6s} {r['bed_class']:>14s} {agree:>6s} {conf:>5s}  {r['class_detail']}")

    # Region totals — expected composition over soft memberships, not hard counts
    n = len(df)
    frac, frac_se = expected_fractions(df)
    parts = ' | '.join(f"{c} {f:.0%}±{e:.0%}" for c, f, e in zip(CLASS_ORDER, frac, frac_se)
                       if f >= 0.005)
    print(f"{'-'*98}")
    print(f"  Region: {n} windows | {parts}")
    print(f"  β: median {df['beta'].median():.2f}, "
          f"IQR [{df['beta'].quantile(0.25):.2f} – {df['beta'].quantile(0.75):.2f}]")
    amb = (df['class_confidence'] < 0.9).mean()
    print(f"  Class confidence: median {df['class_confidence'].median():.0%}, "
          f"{amb:.0%} of windows ambiguous (<90%)")
    if 'psd_amplitude_1km' in df.columns:
        print(f"  PSD amp @ 1 km: median {df['psd_amplitude_1km'].median():.2f}, "
              f"IQR [{df['psd_amplitude_1km'].quantile(0.25):.2f} – {df['psd_amplitude_1km'].quantile(0.75):.2f}]")
    print(f"{'='*90}")


def plot_bed_character(df, summary, region_name, pflag=None, out_dir=None):
    """Two-panel diagnostic: beta histogram + per-segment stacked bars.

    QC only. The publication figure is the cross-region comparison built from the
    left panel alone (plot_region_comparison); the right panel is not published.
    """
    out_dir = out_dir or OUTPUT_DIR
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                    gridspec_kw={'width_ratios': [1, 1.2]})

    # --- Left: beta density (KDE) shaded by class region ---
    # A continuous density decouples the distribution shape from the discrete
    # class labels: color follows the β-axis thresholds, not per-window class,
    # so there is no seam where windows straddle a boundary. If a real density
    # peak sits on a threshold, the smooth curve still shows it (diagnostic).
    boundaries = [1.5, 2.0, 2.5]
    beta = df['beta'].dropna().values
    beta_min, beta_max = beta.min() - 0.1, beta.max() + 0.1
    grid = np.linspace(beta_min, beta_max, 512)
    dens = gaussian_kde(beta)(grid)
    # Fill under the curve, colored by which class region each grid point is in
    for name, lo, hi in BED_CLASSES:
        seg = (grid >= lo) & (grid < hi)
        if seg.any():
            ax1.fill_between(grid, 0, dens, where=seg, color=BED_COLORS[name],
                             alpha=0.55, label=name, linewidth=0)
    ax1.plot(grid, dens, color='0.25', lw=1.2)
    for b in boundaries:
        ax1.axvline(b, color='k', ls='--', lw=1, alpha=0.6)
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel(r'$\beta$', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Window β distribution', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Right: per-segment stacked horizontal bars ---
    # Build labels and fractions
    seg_labels = []
    class_fractions = {name: [] for name, _, _ in BED_CLASSES}

    for _, r in summary.iterrows():
        seg_labels.append(f"{r['trajectory']} s{r['segment']:.0f}")
        for name, _, _ in BED_CLASSES:
            class_fractions[name].append(r[f'frac_{name}'])  # expected, not hard count

    y_pos = np.arange(len(seg_labels))

    # Only show segments — if too many, limit to those with >1 window or cap at 40
    if len(seg_labels) > 40:
        # Show only multi-window segments
        multi = summary[summary['n_windows'] > 1].index
        if len(multi) > 0:
            seg_labels = [seg_labels[i] for i in multi]
            y_pos = np.arange(len(seg_labels))
            for name in class_fractions:
                class_fractions[name] = [class_fractions[name][i] for i in multi]
            ax2.set_title(f'Bed class by segment (multi-window only, n={len(seg_labels)})', fontsize=12)
        else:
            ax2.set_title('Bed class by segment', fontsize=12)
    else:
        ax2.set_title('Bed class by segment', fontsize=12)

    left = np.zeros(len(seg_labels))
    for name, _, _ in BED_CLASSES:
        vals = np.array(class_fractions[name])
        if vals.sum() > 0:
            ax2.barh(y_pos, vals, left=left, color=BED_COLORS[name], label=name, edgecolor='white', linewidth=0.5)
            left += vals

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(seg_labels, fontsize=7)
    ax2.set_xlabel('Expected fraction', fontsize=12)
    ax2.set_xlim(0, 1)
    ax2.invert_yaxis()
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3, axis='x')

    _flag_suptitle(fig, f'Bed Character — {region_name}', pflag)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{region_name}_bed_character.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {out_path}")


def parse_window_km(csv_path):
    """Extract window size in km from filename like '*_w50km_window_stats.csv'."""
    m = re.search(r'_w(\d+)km_', os.path.basename(csv_path))
    return int(m.group(1)) if m else 50  # fallback


def plot_beta_along_track(df, region_name, csv_path, pflag=None, out_dir=None):
    """Beta vs along-track distance per segment, colored by bed class."""
    out_dir = out_dir or OUTPUT_DIR
    step_km = parse_window_km(csv_path) / 2  # 50% overlap

    # Only plot segments with >3 windows
    groups = [(k, g) for k, g in df.groupby(['trajectory', 'segment']) if len(g) > 3]
    if not groups:
        return

    n = len(groups)
    fig, axes = plt.subplots(n, 1, figsize=(14, max(2.5 * n, 4)), squeeze=False, sharex=False)

    boundaries = [1.5, 2.0, 2.5]

    for ax, ((traj, seg), g) in zip(axes[:, 0], groups):
        g = g.sort_values('window_id')
        dist = g['window_id'].values * step_km
        beta = g['beta'].values

        # Plot colored scatter + connecting line. Marker opacity tracks class
        # confidence, so windows sitting near a threshold read as uncertain.
        conf = g['class_confidence'].values
        ax.plot(dist, beta, color='0.6', lw=0.8, zorder=1)
        if 'beta_uncertainty' in g.columns:
            ax.errorbar(dist, beta, yerr=g['beta_uncertainty'].values, fmt='none',
                        ecolor='0.5', elinewidth=0.7, capsize=1.5, zorder=1)
        for name, lo, hi in BED_CLASSES:
            mask = g['bed_class'].values == name
            if mask.any():
                rgba = np.tile(to_rgba(BED_COLORS[name]), (mask.sum(), 1))
                rgba[:, 3] = 0.25 + 0.75 * conf[mask]  # 0.5 conf -> faint, 1.0 -> solid
                ax.scatter(dist[mask], beta[mask], c=rgba,
                           s=30, label=name, zorder=2, edgecolors='k', linewidths=0.3)

        for b in boundaries:
            ax.axhline(b, color='k', ls='--', lw=0.7, alpha=0.5)

        ax.set_ylabel(r'$\beta$', fontsize=10)
        ax.set_title(f'{traj} seg {seg:.0f}  (n={len(g)})', fontsize=10, loc='left')
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('Along-track distance (km)', fontsize=11)

    # Single legend from first axes
    handles, labels = axes[0, 0].get_legend_handles_labels()
    seen = {}
    unique = [(h, l) for h, l in zip(handles, labels) if l not in seen and not seen.update({l: 1})]
    fig.legend(*zip(*unique), loc='upper right', fontsize=9, framealpha=0.9)

    _flag_suptitle(fig, f'β along track — {region_name}', pflag, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{region_name}_beta_along_track.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Along-track plot saved: {out_path}")


def plot_bed_elevation_heatmap(df, region_name, pflag=None, out_dir=None):
    """Contingency heatmap of bed class vs elevation_class (expected window counts).

    Cells are sums of soft memberships, so a boundary-straddling window is split
    across bed classes rather than being forced into one cell.
    """
    out_dir = out_dir or OUTPUT_DIR
    if 'elevation_class' not in df.columns:
        return

    elev_order = [name for name, _, _ in ELEVATION_CLASSES]

    # Expected counts: sum memberships within each elevation class
    counts = df.groupby('elevation_class')[P_COLS].sum().T
    counts.index = CLASS_ORDER
    counts = counts.reindex(columns=elev_order, fill_value=0.0)

    # Drop empty rows/columns
    counts = counts.loc[counts.sum(axis=1) > 0.05, counts.sum(axis=0) > 0.05]
    if counts.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(counts.values, cmap='YlOrRd', aspect='auto')

    # Annotate cells with count and percentage
    total = counts.values.sum()
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            val = counts.values[i, j]
            pct = val / total * 100
            color = 'white' if val > total * 0.3 else 'black'
            ax.text(j, i, f'{val:.1f}\n({pct:.0f}%)', ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')

    ax.set_xticks(range(counts.shape[1]))
    ax.set_xticklabels(counts.columns, fontsize=11)
    ax.set_yticks(range(counts.shape[0]))
    ax.set_yticklabels(counts.index, fontsize=11)
    ax.set_xlabel('Elevation class', fontsize=12)
    ax.set_ylabel('Bed class (β)', fontsize=12)
    ax.set_title(f'n={total:.0f} windows (expected counts)', fontsize=10)
    _flag_suptitle(fig, f'Bed class × Elevation — {region_name}', pflag, fontsize=13)
    fig.colorbar(im, ax=ax, label='Expected window count', shrink=0.8)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{region_name}_bed_elevation_heatmap.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Elevation heatmap saved: {out_path}")


def process_region(region_name, csv_path, out_dir=None):
    out_dir = out_dir or OUTPUT_DIR
    print(f"\nProcessing: {region_name}")
    df = pd.read_csv(csv_path).dropna(subset=['beta'])

    if len(df) == 0:
        print("  No valid data.")
        return

    pflag = region_flag(df)

    # Classify windows. bed_class is the MAP label (descriptive); the p_* columns
    # carry the soft membership used for all quantitative summaries.
    df = add_soft_membership(df)
    df['relief_class'] = df['relief_m'].apply(classify_relief)
    if 'bed_elev_mean' in df.columns:
        df['elevation_class'] = df['bed_elev_mean'].apply(classify_elevation)

    # Write updated window CSV with classification columns (all windows)
    df.to_csv(csv_path, index=False)
    print(f"  Updated {csv_path} with bed_class, class_confidence, {', '.join(P_COLS)}, "
          f"relief_class, elevation_class columns")

    # Exclude transition windows from β analysis
    n_trans = df['is_transition'].sum() if 'is_transition' in df.columns else 0
    if n_trans:
        df = df[~df['is_transition']].copy()
        print(f"  Excluded {n_trans} transition windows from β analysis ({len(df)} remain)")

    # Segment summary
    summary = segment_summary(df)
    summary['processing_flag'] = pflag
    print_summary(summary, region_name, df, pflag)

    # Save summary CSV
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, f'{region_name}_bed_character_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"  Summary saved: {summary_path}")

    # Plots
    plot_bed_character(df, summary, region_name, pflag, out_dir)
    plot_beta_along_track(df, region_name, csv_path, pflag, out_dir)
    plot_bed_elevation_heatmap(df, region_name, pflag, out_dir)


# ---------------------------------------------------------------------------
# Cross-region comparison. One region's beta distribution is a single number
# dressed up; the comparison across regions is the result. Walks a tree of
# region folders and writes to the top of it, not into any one region.
# ---------------------------------------------------------------------------

def walk_tree(root):
    """Every *_window_stats.csv under a tree of region folders."""
    return sorted(glob.glob(os.path.join(root, '**', '*_window_stats.csv'), recursive=True))


def tree_region_name(csv_path):
    """Short folder name (HD, GSM) when the CSV sits in <region>/window_csvs/."""
    parent = os.path.dirname(csv_path)
    if os.path.basename(parent) == 'window_csvs':
        return os.path.basename(os.path.dirname(parent))
    return re.sub(r'_w\d+km_window_stats\.csv$', '', os.path.basename(csv_path))


def tree_out_dir(csv_path):
    """Per-region bed_character/ folder, sibling of window_csvs/."""
    parent = os.path.dirname(csv_path)
    base = os.path.dirname(parent) if os.path.basename(parent) == 'window_csvs' else parent
    return os.path.join(base, 'bed_character')


def dataset_name(csv_path):
    """The loading.py label a window CSV came from."""
    return re.sub(r'_w\d+km_window_stats\.csv$', '', os.path.basename(csv_path))


def _tkey(t):
    """Trajectory id as a stable string; pandas reads a numeric id back as a float."""
    s = str(t)
    return s[:-2] if s.endswith('.0') else s


def segment_lengths(root, refresh=False):
    """(dataset, trajectory, segment) -> segment length in m, cached at <root>/segment_lengths.csv.

    Truncation is a segment-LENGTH property: bed_analysis falls back to a single
    window when a segment is shorter than WINDOW_SIZE. No exported CSV carries the
    length, so it is re-derived from the raw data by replaying the same
    segmentation. The single-window window-count proxy is deliberately not used:
    it misclassifies 13 of 30 windows at ASB-LR (v23 §8), which is the size of the
    effect being reported. Returns None if the raw data is not reachable.
    """
    path = os.path.join(root, 'segment_lengths.csv')
    if os.path.exists(path) and not refresh:
        d = pd.read_csv(path, dtype={'trajectory': str})
        print(f"  Segment lengths: {len(d)} cached from {path}")
        return {(r.dataset, _tkey(r.trajectory), int(r.segment)): float(r.length_m)
                for r in d.itertuples()}

    try:
        from pyproj import Transformer
        from loading import load_datasets
        from segmentation import split_into_segments, split_by_landscape
    except ImportError as e:
        print(f"  Segment lengths unavailable ({e}); truncation left blank on the figure.")
        return None

    print("  Deriving segment lengths from raw data (cached after the first run)...")
    tf = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    rows = []
    try:
        datasets = load_datasets()
    except Exception as e:
        print(f"  Raw data not reachable ({e}); truncation left blank on the figure.")
        return None

    for d in datasets:
        name, df = d['name'], d['data']
        valid = df[(df['bedrock_altitude (m)'] != -9999) & (df['trajectory_id'] != -9999)]
        for traj_id in valid['trajectory_id'].unique():
            line = valid[valid['trajectory_id'] == traj_id].copy()
            if len(line) < 20:
                continue
            x, y = tf.transform(line['longitude (degree_east)'].values,
                                line['latitude (degree_north)'].values)
            dist = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))])
            gap_segments = split_into_segments(line.copy(), dist)
            if not gap_segments:
                continue
            segs = []
            for sd, sdist in gap_segments:
                segs.extend(split_by_landscape(sd, sdist))
            for i, (_, sdist, _) in enumerate(segs):
                rows.append({'dataset': name, 'trajectory': _tkey(traj_id), 'segment': i + 1,
                             'length_m': float(sdist.max() - sdist.min())})

    if not rows:
        print("  No segments derived; truncation left blank on the figure.")
        return None
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Segment lengths: {len(rows)} derived, cached to {path}")
    return {(r['dataset'], r['trajectory'], r['segment']): r['length_m'] for r in rows}


def class_of(beta):
    return CLASS_ORDER[int(np.searchsorted(BED_EDGES[1:-1], beta, side='right'))]


def region_panel(csv_path, lengths=None):
    """Everything one comparison panel needs, read off an already-written window CSV."""
    df = pd.read_csv(csv_path).dropna(subset=['beta'])
    if len(df) == 0:
        return None
    df = add_soft_membership(df)
    n_total = len(df)

    trans = df['is_transition'].astype(bool) if 'is_transition' in df.columns \
        else pd.Series(False, index=df.index)
    kept = df[~trans].copy()
    if len(kept) < 2:
        return None

    beta = kept['beta'].to_numpy(float)
    counts = kept[P_COLS].to_numpy().sum(0)   # expected windows per class, soft memberships

    # Spatially independent window count, over exactly the windows this panel counts.
    # _independent_subset is greedy and walks rows in order, keeping the first that
    # clears the separation from everything already kept, so this is a LOWER BOUND on
    # independent units and not a canonical count: the same points in a different row
    # order can return a different number. It will also differ from the value in
    # landscape_vector's composition CSV, which decimates that module's own kept set
    # rather than this one. Neither difference is a bug.
    # Imported, not mirrored; the import is lazy because landscape_vector imports this
    # module at its own top level.
    n_indep, decim = np.nan, np.nan
    if {'center_x', 'center_y'} <= set(kept.columns):
        from landscape_vector import _independent_subset, COMPOSITION_DECIMATE_KM
        xy = kept[['center_x', 'center_y']].to_numpy(float)
        xy = xy[np.isfinite(xy).all(1)]
        if len(xy):
            n_indep, decim = len(_independent_subset(xy, COMPOSITION_DECIMATE_KM)), \
                COMPOSITION_DECIMATE_KM

    # Truncation over the retained windows only, matching the '88% of homogeneous
    # windows' basis in ODSA-Documentation.md §4.4.
    n_trunc = n_len = np.nan
    debias = np.nan
    if lengths:
        dset = dataset_name(csv_path)
        L = np.array([lengths.get((dset, _tkey(t), int(s)), np.nan)
                      for t, s in zip(kept['trajectory'], kept['segment'])], float)
        known = np.isfinite(L)
        if not known.any():
            # Dataset not in loading.py's target list — no length, so no claim.
            print(f"  {tree_region_name(csv_path)}: no segment lengths matched "
                  f"{dataset_name(csv_path)!r}; truncation left blank.")
        else:
            if not known.all():
                print(f"  {tree_region_name(csv_path)}: {(~known).sum()} of {len(kept)} "
                      f"retained windows have no segment length; truncation is over the rest.")
            n_len, n_trunc = int(known.sum()), int((L[known] < WINDOW_SIZE).sum())
            # Offline debias: shift the truncated windows, re-take the median. Never
            # written back to any beta; it exists to show how far the median can move.
            debias = float(np.median(np.where(known & (L < WINDOW_SIZE),
                                              beta - TRUNC_OFFSET, beta)))

    med = float(np.median(beta))
    breaks = list(BED_EDGES[1:-1])
    near = min(breaks, key=lambda b: abs(debias - b)) if np.isfinite(debias) else np.nan
    gap = abs(debias - near) if np.isfinite(debias) else np.nan
    on_break = np.isfinite(gap) and gap < ON_BREAK_MARGIN
    moved = np.isfinite(debias) and class_of(debias) != class_of(med)

    # Composition, on the same expected-window basis as the shading. Independent of
    # the break test above: the label follows the median, so it can name a class that
    # is not the one holding the most windows.
    share = counts / max(counts.sum(), 1e-12)
    rank = np.argsort(share)[::-1]
    plur, second = CLASS_ORDER[rank[0]], CLASS_ORDER[rank[1]]
    margin = float(share[rank[0]] - share[rank[1]])

    return {'region': tree_region_name(csv_path), 'dataset': dataset_name(csv_path),
            'csv': csv_path, 'pflag': region_flag(kept),
            'beta': beta, 'counts': counts,
            'n_total': n_total, 'n_kept': len(kept), 'n_excluded': int(trans.sum()),
            'n_truncated': n_trunc, 'n_length_known': n_len,
            'n_independent': n_indep, 'decimate_km': decim,
            'beta_median': med, 'beta_median_debiased': debias,
            'bed_class': class_of(med), 'break_nearest': near,
            'break_gap_debiased': gap, 'on_break': bool(on_break), 'class_moves': bool(moved),
            'class_plurality': plur, 'class_second': second,
            'share_plurality': float(share[rank[0]]), 'share_second': float(share[rank[1]]),
            'share_margin': margin, 'near_tie': bool(margin < TIE_MARGIN),
            'share_label': float(share[CLASS_ORDER.index(class_of(med))]),
            'label_is_plurality': bool(class_of(med) == plur)}


def plot_region_comparison(panels, out_path, alpha_min=0.10, alpha_max=0.60,
                           panel_height=2.4, width=9.5, label_counts=True,
                           caveat_color='#b22222', pad=0.15, curve_frac=0.55,
                           y_counts=0.995, y_caveat=0.80, note_gap=8, fs=8, median_ymax=None,
                           share_y=True, legend_alpha=0.45):
    """Stacked beta KDEs, one row per region, on a shared beta axis.

    Shading opacity tracks each class's share of the region's WINDOWS, not the KDE
    area under the band: the smoother pushes mass across a break, so a class can
    look occupied on the curve while holding almost no windows. The counts are
    printed in-band for the same reason. Both channels are expected windows summed
    over the soft memberships, hence non-integer; the footnote says so, because a
    fractional count reads as a bug otherwise.

    Both §4.4 caveats are printed per panel rather than left to a caption, because
    both vary by region: the retained-of-total window count (transition-zone
    exclusion, §1) and the truncated fraction (band truncation, §4.4).
    """
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(width, panel_height * n + 1.2),
                             squeeze=False, sharex=True)
    axes = axes[:, 0]

    lo = min(p['beta'].min() for p in panels) - pad
    hi = max(p['beta'].max() for p in panels) + pad
    grid = np.linspace(lo, hi, 512)

    # The point of the figure is the comparison, so the density axis is shared by
    # default: an independent axis makes a narrow distribution look peakier.
    dens_all = [gaussian_kde(p['beta'])(grid) for p in panels]
    top_shared = max(d.max() for d in dens_all) / curve_frac

    for ax, p, dens in zip(axes, panels, dens_all):
        share = p['counts'] / max(p['counts'].sum(), 1e-12)
        # The curve is held to the lower curve_frac of the panel so the annotation
        # strip above it never lands on the density.
        top = top_shared if share_y else dens.max() / curve_frac
        for (name, cl, ch), sh, cnt in zip(BED_CLASSES, share, p['counts']):
            seg = (grid >= cl) & (grid < ch)
            if not seg.any():
                continue
            a = alpha_min + (alpha_max - alpha_min) * (sh / max(share.max(), 1e-12))
            ax.fill_between(grid, 0, dens, where=seg, color=BED_COLORS[name],
                            alpha=a, linewidth=0)
            if label_counts:
                # Every visible class is labelled, zero included, so panels line up
                # and an empty class is stated rather than silently absent.
                # The label class is starred and bolded on every panel: the label is the
                # class of the median, which need not be the class holding the most
                # windows, and that disagreement has no threshold to hide behind.
                is_label = name == p['bed_class']
                xs = grid[seg]
                ax.text(0.5 * (max(xs[0], lo) + min(xs[-1], hi)), y_counts * top,
                        f"{name}{'*' if is_label else ''}\n{cnt:.1f} ({sh:.0%})",
                        ha='center', va='top', fontsize=fs - 0.5,
                        color='0.1' if is_label else '0.25',
                        fontweight='bold' if is_label else 'normal')

        ax.plot(grid, dens, color='0.25', lw=1.2)
        for b in BED_EDGES[1:-1]:
            ax.axvline(b, color='k', ls='--', lw=1, alpha=0.6)
        # Stems to the density at the median rather than spanning the axis: the count
        # strip and the annotation boxes sit above the curve, and a full-height rule
        # reads straight through them.
        stem = min(curve_frac, 1.08 * float(np.interp(p['beta_median'], grid, dens)) / top)
        ax.axvline(p['beta_median'], ymax=stem if median_ymax is None else median_ymax,
                   color='k', lw=1.6, alpha=0.85)
        ax.set_ylim(0, top)
        ax.set_ylabel('Density', fontsize=10)
        ax.grid(True, alpha=0.3)
        _flag_title(ax, f"{p['region']}   median β = {p['beta_median']:.2f}", p['pflag'],
                    fontsize=11)

        # Caveat block: both numbers vary by region, so neither belongs in a caption.
        trunc = ('truncated: not derivable' if not np.isfinite(p['n_truncated'])
                 else f"truncated: {p['n_truncated'] / max(p['n_length_known'], 1):.0%} "
                      f"({p['n_truncated']} of {p['n_length_known']}), uncorrected "
                      f"+{TRUNC_OFFSET:.2f} on those")
        # Not called n_eff: that name is already the Kish weighted ESS elsewhere in
        # this project, and this is a greedy spatial-decimation count.
        indep = (f", {int(p['n_independent'])} independent at "
                 f"{p['decimate_km']:.0f} km" if np.isfinite(p['n_independent']) else '')
        lines = [f"{p['n_kept']} of {p['n_total']} windows{indep} "
                 f"({p['n_excluded']} transition-zone excluded)", trunc]
        # The debias distance is printed on every panel, not only the flagged ones,
        # so the margin below only sets emphasis and never hides a narrow hold.
        if np.isfinite(p['beta_median_debiased']):
            b = p['break_nearest']
            lines.append(f"debias → {p['beta_median_debiased']:.2f}, "
                         f"{p['break_gap_debiased']:.2f} from the "
                         f"{class_of(b - 1e-9)}/{class_of(b + 1e-9)} break")
        ax.text(0.985, y_caveat, '\n'.join(lines), transform=ax.transAxes,
                ha='right', va='top', fontsize=fs, color='0.2',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85,
                          edgecolor='0.8'))

        # Two independent ways a bare class label misleads: the median sits near a
        # break, or the composition is a near-tie. Neither implies the other, so both
        # are tested and a panel can carry both lines.
        notes = []
        if p['on_break'] or p['class_moves']:
            b = p['break_nearest']
            notes.append(f"crosses into {class_of(p['beta_median_debiased'])}"
                         if p['class_moves'] else
                         f"within {ON_BREAK_MARGIN:.2f} of the "
                         f"{class_of(b - 1e-9)}/{class_of(b + 1e-9)} break")
        if not p['label_is_plurality']:
            notes.append(f"{p['class_plurality']} holds more windows "
                         f"({p['share_plurality']:.0%}) than the label "
                         f"{p['bed_class']} ({p['share_label']:.0%})")
        elif p['near_tie']:
            notes.append(f"{p['class_plurality']} {p['share_plurality']:.0%} vs "
                         f"{p['class_second']} {p['share_second']:.0%}, "
                         f"{100 * p['share_margin']:.0f} pt margin")
        if notes:
            note = (f"{notes[0]} — do not quote a bare class label" if len(notes) == 1
                    else '\n'.join(notes) + '\n— do not quote a bare class label')
            # Offset in points, not axes fraction: the panels shrink in fraction terms
            # as the fixed-height footer takes a bigger share at small n, and a fixed
            # fraction puts this box through the caveat box at n=2.
            drop = len(lines) * fs * 1.35 + note_gap
            ax.annotate(note, xy=(0.985, y_caveat), xycoords='axes fraction',
                        xytext=(0, -drop), textcoords='offset points',
                        ha='right', va='top',
                        fontsize=fs, color=caveat_color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85,
                                  edgecolor=caveat_color, linewidth=0.8))

    axes[-1].set_xlabel(r'$\beta$', fontsize=12)
    axes[-1].set_xlim(lo, hi)

    # Swatches are drawn at one fixed alpha, not harvested off a panel: opacity is a
    # data channel here, so a legend key at a panel's alpha would assert a value.
    keys = [Patch(facecolor=BED_COLORS[c], alpha=legend_alpha, label=c) for c in CLASS_ORDER]
    # Footer heights are inches converted to figure fraction: the figure grows with
    # the region count, so a fixed fraction would drift.
    fig_h = fig.get_size_inches()[1]
    fig.legend(handles=keys, loc='lower center', ncol=len(keys), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.52 / fig_h))
    fig.text(0.5, 0.10 / fig_h,
             'Class counts are expected windows, summed over the per-window class '
             'memberships, so they are not integers. Fill opacity scales with each '
             "class's share of the region's windows, not with the area under the curve. "
             '*marks the class of the region median, which is the label; it is not '
             'always the class holding the most windows.',
             ha='center', va='bottom', fontsize=fs - 0.5, color='0.35', wrap=True)

    _flag_suptitle(fig, f'Bed character across {n} regions — window β distributions', None)
    plt.tight_layout(rect=[0, 0.85 / fig_h, 1, 0.985])
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Comparison figure saved: {out_path}")


def compare_regions(root, order='beta', **plot_kw):
    """Build the n-region comparison from already-written window CSVs under `root`."""
    csvs = walk_tree(root)
    lengths = segment_lengths(root)

    panels = []
    for f in csvs:
        p = region_panel(f, lengths)
        if p:
            panels.append(p)
        else:
            print(f"  Skipped (no usable windows): {f}")
    if len(panels) < 2:
        print(f"\nCross-region: only {len(panels)} usable region(s) under {root}, "
              f"nothing to compare.")
        return None

    panels.sort(key=(lambda p: p['beta_median']) if order == 'beta'
                else (lambda p: p['region']))

    print(f"\n{'='*104}\n  BED CHARACTER ACROSS {len(panels)} REGIONS\n{'='*104}")
    # 'indep' rather than n_eff: n_eff is the Kish weighted ESS in this project.
    print(f"{'Region':>10s} {'flag':>10s} {'kept/tot':>10s} {'indep':>6s} {'trunc':>14s} "
          f"{'β_med':>6s} {'class':>13s} {'share':>6s} {'plurality':>14s} {'margin':>7s} "
          f"{'debias':>7s} {'gap':>6s}  note")
    rows = []
    for p in panels:
        tr = ('—' if not np.isfinite(p['n_truncated'])
              else f"{p['n_truncated']}/{p['n_length_known']} "
                   f"{p['n_truncated']/max(p['n_length_known'],1):.0%}")
        deb = f"{p['beta_median_debiased']:.2f}" if np.isfinite(p['beta_median_debiased']) else '—'
        gap = f"{p['break_gap_debiased']:.2f}" if np.isfinite(p['break_gap_debiased']) else '—'
        tie = ('plurality is not the label' if not p['label_is_plurality']
               else 'near-tie plurality' if p['near_tie'] else '')
        brk = ('class moves' if p['class_moves'] else
               'on a break' if p['on_break'] else '')
        note = ('; '.join(x for x in (brk, tie) if x)
                + (' — no bare class label' if brk or tie else ''))
        ind = f"{int(p['n_independent'])}" if np.isfinite(p['n_independent']) else '—'
        print(f"{p['region']:>10s} {str(p['pflag']):>10s} "
              f"{p['n_kept']:>4d}/{p['n_total']:<5d} {ind:>6s} {tr:>14s} "
              f"{p['beta_median']:>6.2f} {p['bed_class']:>13s} {p['share_label']:>6.0%} "
              f"{p['class_plurality']:>14s} {p['share_margin']:>7.0%} "
              f"{deb:>7s} {gap:>6s}  {note}")
        rows.append({k: v for k, v in p.items() if k not in ('beta', 'counts', 'csv')}
                    | {f'n_{c}': v for c, v in zip(CLASS_ORDER, p['counts'])})
    print(f"{'='*104}")
    km = next((p['decimate_km'] for p in panels if np.isfinite(p['decimate_km'])), np.nan)
    print(f"  indep = spatially independent retained windows at {km:.0f} km "
          f"(greedy, so a lower bound); not the Kish n_eff.")
    print(f"  Class counts are expected windows over soft memberships; the shading "
          f"opacity follows them, not the KDE area.")
    print(f"  class = the class of the median, which is the label. share = the label "
          f"class's window share; plurality/margin = the largest share and its lead "
          f"over the runner-up. A label below its plurality, or a margin under "
          f"{TIE_MARGIN:.0%}, is not a stable class.")
    print(f"  Truncation is uncorrected in every exported β. debias = median after "
          f"subtracting {TRUNC_OFFSET} from truncated windows (offline check only).")

    out = pd.DataFrame(rows)
    csv_path = os.path.join(root, 'bed_character_comparison.csv')
    out.to_csv(csv_path, index=False)
    print(f"\n  Table saved: {csv_path}")

    plot_region_comparison(panels, os.path.join(root, 'bed_character_comparison.png'),
                           **plot_kw)
    return out


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    # Compare regions already processed, without reprocessing them.
    if arg == '--compare':
        root = sys.argv[2] if len(sys.argv) > 2 else _REGION_BASE
        sys.stdout = Tee(os.path.join(root, 'bed_character_comparison_log.txt'))
        compare_regions(root)
        sys.exit(0)

    # Walk a tree of region folders: per-region QC outputs stay in each region's
    # own bed_character/, the comparison goes to the top of the tree.
    if arg and os.path.isdir(arg):
        found = walk_tree(arg)
        sys.stdout = Tee(os.path.join(arg, 'bed_character_comparison_log.txt'))
        print(f"Walking {arg}: {len(found)} region CSVs")
        for f in found:
            process_region(tree_region_name(f), f, tree_out_dir(f))
        compare_regions(arg)
        sys.exit(0)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, 'bed_character_log.txt')
    sys.stdout = Tee(log_path)

    # Discover CSVs
    csvs = discover_window_csvs(os.path.join(_REGION_BASE, 'window_csvs'))
    if not csvs:
        # Try from region base directory in case of flat layout
        csvs = discover_window_csvs(_REGION_BASE)

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.endswith('.csv'):
            region = os.path.basename(arg).replace('_window_stats.csv', '')
            process_region(region, arg)
        elif arg in csvs:
            process_region(arg, csvs[arg])
        else:
            matches = [r for r in csvs if arg.lower() in r.lower()]
            if len(matches) == 1:
                process_region(matches[0], csvs[matches[0]])
            elif matches:
                print(f"Multiple matches for '{arg}':")
                for m in matches:
                    print(f"  - {m}")
            else:
                print(f"Region '{arg}' not found. Available:")
                for r in sorted(csvs):
                    print(f"  - {r}")
    else:
        selection = select_region(csvs)
        if selection == 'ALL':
            for r in sorted(csvs):
                process_region(r, csvs[r])
        elif selection:
            process_region(selection, csvs[selection])
