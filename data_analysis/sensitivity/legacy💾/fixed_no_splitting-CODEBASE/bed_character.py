import os, sys, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy.stats import norm, gaussian_kde
from config import Tee, PROCESSING_FLAG_NOTE as _FLAG_NOTE, processing_flag_of as region_flag
from plotting import flag_suptitle as _flag_suptitle

"""
Bed character classification from window-level spectral data.

Classifies each sliding window by its power-law exponent (beta) into
bed character classes, summarises per segment and per region, and
produces a diagnostic visualisation.

Usage:
  python bed_character.py                          # interactive, discovers from window_csvs/
  python bed_character.py Pensacola                # partial match
  python bed_character.py window_csvs/some.csv     # direct path
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
# Stays 0.0: beta_uncertainty is a mild underestimate (the geomspace frequency
# grid oversamples the window's Fourier resolution, so true sigma ~0.06 not
# 0.043), but class composition is insensitive to sigma across any plausible
# value. Per-window class_confidence is NOT, so don't quote it as measured.
# See v23/beta_sigma_calibration.py and "Sensitivity test - beta_sigma".
SIGMA_EXTRA = 0.0


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


def plot_bed_character(df, summary, region_name, pflag=None):
    """Two-panel diagnostic: beta histogram + per-segment stacked bars."""
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f'{region_name}_bed_character.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {out_path}")


def parse_window_km(csv_path):
    """Extract window size in km from filename like '*_w50km_window_stats.csv'."""
    m = re.search(r'_w(\d+)km_', os.path.basename(csv_path))
    return int(m.group(1)) if m else 50  # fallback


def plot_beta_along_track(df, region_name, csv_path, pflag=None):
    """Beta vs along-track distance per segment, colored by bed class."""
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

    out_path = os.path.join(OUTPUT_DIR, f'{region_name}_beta_along_track.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Along-track plot saved: {out_path}")


def plot_bed_elevation_heatmap(df, region_name, pflag=None):
    """Contingency heatmap of bed class vs elevation_class (expected window counts).

    Cells are sums of soft memberships, so a boundary-straddling window is split
    across bed classes rather than being forced into one cell.
    """
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

    out_path = os.path.join(OUTPUT_DIR, f'{region_name}_bed_elevation_heatmap.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Elevation heatmap saved: {out_path}")


def process_region(region_name, csv_path):
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, f'{region_name}_bed_character_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"  Summary saved: {summary_path}")

    # Plots
    plot_bed_character(df, summary, region_name, pflag)
    plot_beta_along_track(df, region_name, csv_path, pflag)
    plot_bed_elevation_heatmap(df, region_name, pflag)


if __name__ == "__main__":
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
