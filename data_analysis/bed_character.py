import os, sys, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from bed_analysis_22 import Tee

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
    ('hard',          1.5,     2.1),
    ('transitional',  2.1,     2.5),
    ('soft',          2.5,     np.inf),
]

# Relief thresholds — anchored to reference landscapes (DML_AniRES, Aurora/Golicyna, Moller/Recovery)
RELIEF_CLASSES = [
    ('flat',        -np.inf, 350),
    ('subdued',     350,     800),
    ('mountainous', 800,     np.inf),
]

# Elevation thresholds — absolute bed elevation (m a.s.l.) [Siegert_2004, Frederick_2016]
ELEVATION_CLASSES = [
    ('submerged',  -np.inf, 0),
    ('emergent',   0,       1000),
    ('elevated',   1000,    np.inf),
]

BED_COLORS = {
    'chaotic':      '#d62728',
    'hard':         '#ff7f0e',
    'transitional': '#9467bd',
    'soft':         '#1f77b4',
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bed_character/')


def classify_beta(beta):
    for name, lo, hi in BED_CLASSES:
        if lo <= beta < hi:
            return name
    return 'unknown'


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
    """Per-segment bed character summary."""
    rows = []
    for (traj, seg), g in df.groupby(['trajectory', 'segment']):
        n = len(g)
        dominant = g['bed_class'].mode().iloc[0]
        agreement = (g['bed_class'] == dominant).mean()
        class_counts = g['bed_class'].value_counts()
        class_str = ', '.join(f"{c} {v}" for c, v in class_counts.items())

        row = {
            'trajectory': traj,
            'segment': seg,
            'n_windows': n,
            'beta_median': g['beta'].median(),
            'beta_iqr': g['beta'].quantile(0.75) - g['beta'].quantile(0.25) if n > 1 else np.nan,
            'relief_median': g['relief_m'].median(),
            'bed_class': dominant,
            'agreement': agreement,
            'class_detail': class_str,
            'relief_class': classify_relief(g['relief_m'].median()),
        }
        if 'bed_elev_mean' in g.columns:
            row['bed_elev_median'] = g['bed_elev_mean'].median()
            row['elevation_class'] = classify_elevation(g['bed_elev_mean'].median())
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(summary, region_name, df):
    """Print terminal table."""
    print(f"\n{'='*80}")
    print(f"  BED CHARACTER: {region_name}")
    print(f"{'='*80}")
    print(f"{'Traj':>8s} {'Seg':>4s} {'n':>3s} {'β_med':>6s} {'β_IQR':>6s} "
          f"{'Relief':>7s} {'Bed Class':>14s} {'Agree':>6s}  Detail")
    print(f"{'-'*80}")
    for _, r in summary.iterrows():
        iqr = f"{r['beta_iqr']:.2f}" if np.isfinite(r['beta_iqr']) else '—'
        agree = f"{r['agreement']:.0%}" if r['n_windows'] > 1 else '(1win)'
        print(f"{r['trajectory']:>8} {r['segment']:>4.0f} {r['n_windows']:>3.0f} "
              f"{r['beta_median']:>6.2f} {iqr:>6s} {r['relief_median']:>7.0f} "
              f"{r['bed_class']:>14s} {agree:>6s}  {r['class_detail']}")

    # Region totals
    n = len(df)
    counts = df['bed_class'].value_counts()
    parts = ' | '.join(f"{c} {v/n:.0%}" for c, v in counts.items())
    print(f"{'-'*80}")
    print(f"  Region: {n} windows | {parts}")
    print(f"  β: median {df['beta'].median():.2f}, "
          f"IQR [{df['beta'].quantile(0.25):.2f} – {df['beta'].quantile(0.75):.2f}]")
    print(f"{'='*80}")


def plot_bed_character(df, summary, region_name):
    """Two-panel diagnostic: beta histogram + per-segment stacked bars."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                    gridspec_kw={'width_ratios': [1, 1.2]})

    # --- Left: beta histogram colored by class ---
    # Build bin edges that always include the class boundaries
    boundaries = [1.5, 2.1, 2.5]
    beta_min, beta_max = df['beta'].min() - 0.1, df['beta'].max() + 0.1
    bin_width = 0.1
    bin_edges = np.arange(beta_min, beta_max + bin_width, bin_width)
    # Insert class boundaries and re-sort so no bin straddles two classes
    bin_edges = np.unique(np.sort(np.concatenate([bin_edges, boundaries])))
    # Single stacked histogram with pre-separated data
    class_order = [name for name, _, _ in BED_CLASSES]
    class_data = [df.loc[df['bed_class'] == name, 'beta'].values for name in class_order]
    class_colors = [BED_COLORS[name] for name in class_order]
    # Filter empties
    keep = [len(d) > 0 for d in class_data]
    ax1.hist([d for d, k in zip(class_data, keep) if k],
             bins=bin_edges, stacked=True,
             color=[c for c, k in zip(class_colors, keep) if k],
             label=[n for n, k in zip(class_order, keep) if k],
             edgecolor='white', linewidth=0.5)
    for b in boundaries:
        ax1.axvline(b, color='k', ls='--', lw=1, alpha=0.6)
    ax1.set_xlabel(r'$\beta$', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Window β distribution', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Right: per-segment stacked horizontal bars ---
    # Build labels and fractions
    seg_labels = []
    class_fractions = {name: [] for name, _, _ in BED_CLASSES}

    for _, r in summary.iterrows():
        label = f"{r['trajectory']} s{r['segment']:.0f}"
        seg_labels.append(label)
        detail = r['class_detail']
        # Parse class counts from detail string
        total = r['n_windows']
        for name, _, _ in BED_CLASSES:
            # Count from the original df
            mask = (df['trajectory'] == r['trajectory']) & (df['segment'] == r['segment']) & (df['bed_class'] == name)
            class_fractions[name].append(mask.sum() / total)

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
    ax2.set_xlabel('Fraction', fontsize=12)
    ax2.set_xlim(0, 1)
    ax2.invert_yaxis()
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3, axis='x')

    fig.suptitle(f'Bed Character — {region_name}', fontsize=14, y=1.01)
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


def plot_beta_along_track(df, region_name, csv_path):
    """Beta vs along-track distance per segment, colored by bed class."""
    step_km = parse_window_km(csv_path) / 2  # 50% overlap

    # Only plot segments with >3 windows
    groups = [(k, g) for k, g in df.groupby(['trajectory', 'segment']) if len(g) > 3]
    if not groups:
        return

    n = len(groups)
    fig, axes = plt.subplots(n, 1, figsize=(14, max(2.5 * n, 4)), squeeze=False, sharex=False)

    boundaries = [1.5, 2.1, 2.5]

    for ax, ((traj, seg), g) in zip(axes[:, 0], groups):
        g = g.sort_values('window_id')
        dist = g['window_id'].values * step_km
        beta = g['beta'].values

        # Plot colored scatter + connecting line
        ax.plot(dist, beta, color='0.6', lw=0.8, zorder=1)
        for name, lo, hi in BED_CLASSES:
            mask = g['bed_class'].values == name
            if mask.any():
                ax.scatter(dist[mask], beta[mask], c=BED_COLORS[name],
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

    fig.suptitle(f'β along track — {region_name}', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(OUTPUT_DIR, f'{region_name}_beta_along_track.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Along-track plot saved: {out_path}")


ELEV_COLORS = {
    'submerged': '#2166ac',
    'emergent':  '#b2182b',
    'elevated':  '#762a83',
}


def plot_bed_elevation_heatmap(df, region_name):
    """Contingency heatmap of bed_class vs elevation_class (window counts)."""
    if 'elevation_class' not in df.columns:
        return

    bed_order = [name for name, _, _ in BED_CLASSES]
    elev_order = [name for name, _, _ in ELEVATION_CLASSES]

    # Build counts matrix
    counts = pd.crosstab(df['bed_class'], df['elevation_class'])
    counts = counts.reindex(index=bed_order, columns=elev_order, fill_value=0)

    # Drop empty rows/columns
    counts = counts.loc[counts.sum(axis=1) > 0, counts.sum(axis=0) > 0]
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
            ax.text(j, i, f'{val}\n({pct:.0f}%)', ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')

    ax.set_xticks(range(counts.shape[1]))
    ax.set_xticklabels(counts.columns, fontsize=11)
    ax.set_yticks(range(counts.shape[0]))
    ax.set_yticklabels(counts.index, fontsize=11)
    ax.set_xlabel('Elevation class', fontsize=12)
    ax.set_ylabel('Bed class (β)', fontsize=12)
    ax.set_title(f'Bed class × Elevation — {region_name}\n(n={total} windows)', fontsize=13)
    fig.colorbar(im, ax=ax, label='Window count', shrink=0.8)
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

    # Classify windows
    df['bed_class'] = df['beta'].apply(classify_beta)
    df['relief_class'] = df['relief_m'].apply(classify_relief)
    if 'bed_elev_mean' in df.columns:
        df['elevation_class'] = df['bed_elev_mean'].apply(classify_elevation)

    # Write updated window CSV with classification columns
    df.to_csv(csv_path, index=False)
    print(f"  Updated {csv_path} with bed_class, relief_class columns")

    # Segment summary
    summary = segment_summary(df)
    print_summary(summary, region_name, df)

    # Save summary CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, f'{region_name}_bed_character_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"  Summary saved: {summary_path}")

    # Plots
    plot_bed_character(df, summary, region_name)
    plot_beta_along_track(df, region_name, csv_path)
    plot_bed_elevation_heatmap(df, region_name)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, 'bed_character_log.txt')
    sys.stdout = Tee(log_path)

    # Discover CSVs
    csvs = discover_window_csvs('window_csvs')
    if not csvs:
        # Try from current directory in case user is inside the results folder
        csvs = discover_window_csvs('.')

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
