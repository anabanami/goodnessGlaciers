import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize


"""
Compares cos²(θ) anisotropy fits with and without MEaSUREs-based weighting.

Windows where REMA and MEaSUREs flow directions disagree strongly have
unreliable incidence angles. This script down-weights those windows
in the cos²(θ) fit and shows the effect on the anisotropy signal.

Usage:
  python weighted_anisotropy.py                    # interactive region menu
  python weighted_anisotropy.py Aurora              # partial match
  python weighted_anisotropy.py some_window_stats.csv  # direct path
"""


def discover_regions(directory='.'):
    """
    Find all region datasets in directory by looking for *_segment_stats.csv or *_window_stats.csv files.
    Returns dict of region_name -> {'segment': path, 'window': path}
    """
    regions = {}

    # Find all stats CSV files
    segment_files = glob.glob(os.path.join(directory, '*_segment_stats.csv'))
    window_files = glob.glob(os.path.join(directory, '*_window_stats.csv'))

    # Extract region names from segment files
    for f in segment_files:
        basename = os.path.basename(f)
        region = basename.replace('_segment_stats.csv', '')
        if region not in regions:
            regions[region] = {}
        regions[region]['segment'] = f

    # Extract region names from window files
    for f in window_files:
        basename = os.path.basename(f)
        region = basename.replace('_window_stats.csv', '')
        if region not in regions:
            regions[region] = {}
        regions[region]['window'] = f

    return regions


def select_region(regions):
    """
    Interactive region selection if multiple regions available.
    """
    if not regions:
        print("No region datasets found (*_segment_stats.csv or *_window_stats.csv)")
        return None

    if len(regions) == 1:
        region = list(regions.keys())[0]
        print(f"Found 1 region: {region}")
        return region

    # Multiple regions - show menu
    print(f"\nFound {len(regions)} regions:")
    sorted_regions = sorted(regions.keys())
    for i, region in enumerate(sorted_regions, 1):
        files = regions[region]
        has_seg = 'segment' in files
        has_win = 'window' in files
        status = f"[seg: {'Y' if has_seg else 'N'}, win: {'Y' if has_win else 'N'}]"
        print(f"  {i}. {region} {status}")

    print(f"  0. Process ALL regions")

    while True:
        try:
            choice = input("\nSelect region number (or 0 for all): ").strip()
            choice = int(choice)
            if choice == 0:
                return 'ALL'
            if 1 <= choice <= len(sorted_regions):
                return sorted_regions[choice - 1]
            print("Invalid choice.")
        except ValueError:
            print("Please enter a number.")


def flow_weight(flow_error, cutoff=60.0):
    """
    Linearly decaying weight: 1.0 at 0° error, 0.0 at cutoff.
    Points beyond cutoff get zero weight.
    """
    w = np.clip(1.0 - flow_error / cutoff, 0.0, 1.0)
    w[np.isnan(flow_error)] = 0.0
    return w


def cos2_model(theta_deg, beta_perp, beta_parallel):
    """
    β(θ) = β⊥ + (β∥ - β⊥) cos²(θ)
    """
    theta_rad = np.radians(theta_deg)
    return beta_perp + (beta_parallel - beta_perp) * np.cos(theta_rad)**2


def bootstrap_cos2_uncertainty(theta, beta, weights=None, n_boot=2000, block_length=3):
    """Block bootstrap for cos²θ fit, optionally weighted."""
    n = len(theta)
    boot_params = []

    for _ in range(n_boot):
        # Draw random block start indices
        n_blocks = int(np.ceil(n / block_length))
        starts = np.random.randint(0, n, size=n_blocks)

        # Build bootstrap sample from contiguous blocks
        indices = np.concatenate([np.arange(s, min(s + block_length, n)) for s in starts])[:n]

        t_boot = theta[indices]
        b_boot = beta[indices]

        try:
            if weights is not None:
                w_boot = weights[indices]
                sigma = np.where(w_boot > 0, 1.0 / w_boot, 1e10)
                popt, _ = optimize.curve_fit(cos2_model, t_boot, b_boot,
                                             p0=[np.mean(b_boot), np.mean(b_boot)],
                                             sigma=sigma, absolute_sigma=False,
                                             maxfev=5000)
            else:
                popt, _ = optimize.curve_fit(cos2_model, t_boot, b_boot,
                                             p0=[np.mean(b_boot), np.mean(b_boot)],
                                             maxfev=5000)
            boot_params.append(popt)
        except (RuntimeError, ValueError):
            continue

    boot_params = np.array(boot_params)
    param_se = np.std(boot_params, axis=0)  # [se_beta_perp, se_beta_parallel]
    delta_se = np.std(boot_params[:, 1] - boot_params[:, 0])  # se of (beta_par - beta_perp)
    return param_se, delta_se


def fit_cos2(theta, beta, weights=None):
    """Fit cos²θ model, return dict with fit results or None on failure."""
    low = theta < 30
    high = theta > 60
    p0_par = np.mean(beta[low]) if np.any(low) else np.mean(beta)
    p0_perp = np.mean(beta[high]) if np.any(high) else np.mean(beta)

    try:
        if weights is not None:
            sigma = np.where(weights > 0, 1.0 / weights, 1e10)
            popt, _ = optimize.curve_fit(cos2_model, theta, beta,
                                         p0=[p0_perp, p0_par],
                                         sigma=sigma, absolute_sigma=False)
        else:
            popt, _ = optimize.curve_fit(cos2_model, theta, beta,
                                         p0=[p0_perp, p0_par])

        beta_perp, beta_par = popt
        perr, delta_se = bootstrap_cos2_uncertainty(theta, beta, weights=weights, block_length=3)

        pred = cos2_model(theta, *popt)
        if weights is not None:
            ss_res = np.sum(weights * (beta - pred)**2)
            ss_tot = np.sum(weights * (beta - np.average(beta, weights=weights))**2)
        else:
            ss_res = np.sum((beta - pred)**2)
            ss_tot = np.sum((beta - np.mean(beta))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            'beta_par': beta_par, 'beta_perp': beta_perp,
            'delta': beta_par - beta_perp, 'delta_se': delta_se,
            'perr': perr, 'r2': r2, 'popt': popt,
        }
    except (RuntimeError, ValueError) as e:
        print(f"  Fit failed: {e}")
        return None


def plot_comparison(csv_path):
    df = pd.read_csv(csv_path)

    # Need incidence, beta, and flow error columns
    required = ['incidence_deg', 'beta']
    for col in required:
        if col not in df.columns:
            print(f"Missing column: {col}")
            return

    has_flow_error = 'flow_error_mean' in df.columns
    if not has_flow_error:
        print(f"No flow_error_mean column in {csv_path} — cannot compute weighted fit.")
        print("Run bed_analysis_19.py with MEaSUREs validation enabled first.")
        return

    df_clean = df.dropna(subset=['incidence_deg', 'beta']).copy()
    if len(df_clean) == 0:
        print("No valid data.")
        return

    theta = df_clean['incidence_deg'].values
    beta = df_clean['beta'].values
    beta_err = df_clean['beta_uncertainty'].values if 'beta_uncertainty' in df_clean.columns else None
    ferr = df_clean['flow_error_mean'].values
    weights = flow_weight(ferr)

    n_total = len(theta)
    n_nonzero_weight = np.sum(weights > 0)
    print(f"Loaded {n_total} windows, {n_nonzero_weight} with non-zero weight")

    # --- Fits ---
    fit_unw = fit_cos2(theta, beta)
    fit_w = fit_cos2(theta, beta, weights=weights)

    if fit_unw is None and fit_w is None:
        print("Both fits failed.")
        return

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    x_fit = np.linspace(0, 90, 200)

    # Left panel: unweighted (original)
    ax = axes[0]
    if beta_err is not None and np.any(np.isfinite(beta_err)):
        ax.errorbar(theta, beta, yerr=beta_err, fmt='o', alpha=0.5, ms=3,
                    color='steelblue', ecolor='gray', elinewidth=0.5, capsize=1.5)
    else:
        ax.scatter(theta, beta, alpha=0.5, s=20, c='steelblue')
    ax.set_title('Unweighted (original)', fontsize=12)
    if fit_unw:
        ax.plot(x_fit, cos2_model(x_fit, *fit_unw['popt']), 'k-', linewidth=2,
                label=(f"$\\beta_\\parallel$={fit_unw['beta_par']:.2f}$\\pm${fit_unw['perr'][1]:.2f}\n"
                       f"$\\beta_\\perp$={fit_unw['beta_perp']:.2f}$\\pm${fit_unw['perr'][0]:.2f}\n"
                       f"$\\Delta\\beta$={fit_unw['delta']:+.2f}$\\pm${fit_unw['delta_se']:.2f}, R²={fit_unw['r2']:.3f}"))
        ax.legend(fontsize=9)

    # Right panel: weighted by flow confidence
    ax = axes[1]
    if beta_err is not None and np.any(np.isfinite(beta_err)):
        ax.errorbar(theta, beta, yerr=beta_err, fmt='none', ecolor='gray',
                    elinewidth=0.5, capsize=1.5, alpha=0.5)
    sc = ax.scatter(theta, beta, alpha=0.6, s=20, c=weights, cmap='viridis',
                    vmin=0, vmax=1, edgecolors='none')
    ax.set_title('Weighted by flow confidence', fontsize=12)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Weight (1=agree, 0=disagree)', fontsize=9)
    if fit_w:
        ax.plot(x_fit, cos2_model(x_fit, *fit_w['popt']), 'k-', linewidth=2,
                label=(f"$\\beta_\\parallel$={fit_w['beta_par']:.2f}$\\pm${fit_w['perr'][1]:.2f}\n"
                       f"$\\beta_\\perp$={fit_w['beta_perp']:.2f}$\\pm${fit_w['perr'][0]:.2f}\n"
                       f"$\\Delta\\beta$={fit_w['delta']:+.2f}$\\pm${fit_w['delta_se']:.2f}, R²={fit_w['r2']:.3f}"))
        ax.legend(fontsize=9)

    for ax in axes:
        ax.set_xlabel('Incidence Angle (°)')
        ax.set_xlim(-2, 92)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r'Power Law Exponent ($\beta$)')

    # Suptitle with comparison summary
    if fit_unw and fit_w:
        fig.suptitle(
            f'MEaSUREs Flow-Weighted Anisotropy Comparison (n={n_total} windows)\n'
            f'$\\Delta\\beta$ unweighted: {fit_unw["delta"]:+.3f}  |  '
            f'$\\Delta\\beta$ weighted: {fit_w["delta"]:+.3f}',
            fontsize=13, y=1.02)

    plt.tight_layout()
    output_path = csv_path.replace('_window_stats.csv', '_weighted_anisotropy.png')
    if output_path == csv_path:
        output_path = csv_path.replace('.csv', '_weighted_anisotropy.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

    # Print comparison table
    print(f"\n{'='*55}")
    print(f"{'':20s} {'Unweighted':>15s} {'Weighted':>15s}")
    print(f"{'-'*55}")
    if fit_unw and fit_w:
        print(f"{'beta_parallel':20s} {fit_unw['beta_par']:>8.3f}±{fit_unw['perr'][1]:<5.3f} {fit_w['beta_par']:>8.3f}±{fit_w['perr'][1]:<.3f}")
        print(f"{'beta_perp':20s} {fit_unw['beta_perp']:>8.3f}±{fit_unw['perr'][0]:<5.3f} {fit_w['beta_perp']:>8.3f}±{fit_w['perr'][0]:<.3f}")
        print(f"{'delta_beta':20s} {fit_unw['delta']:>+8.3f}±{fit_unw['delta_se']:<5.3f} {fit_w['delta']:>+8.3f}±{fit_w['delta_se']:<.3f}")
        print(f"{'R²':20s} {fit_unw['r2']:>14.4f} {fit_w['r2']:>14.4f}")
    print(f"{'='*55}")

    print(f"\nSaved to {output_path}")


def plot_segment_scatter(csv_path):
    """
    Segment-level scatter with cos²θ anisotropy fit.
    Auto-detects segment CSV from window CSV path (for backwards compatibility).
    """
    seg_path = csv_path.replace('_window_stats.csv', '_segment_stats.csv')
    plot_segment_scatter_direct(seg_path)


def process_region(region_name, files):
    """Process a single region's data."""
    print(f"\n{'='*60}")
    print(f"Processing: {region_name}")
    print('='*60)

    if 'window' in files:
        plot_comparison(files['window'])
    else:
        print(f"  No window stats file for {region_name}")

    if 'segment' in files:
        plot_segment_scatter_direct(files['segment'])
    else:
        print(f"  No segment stats file for {region_name}")


def plot_segment_scatter_direct(seg_path):
    """
    Segment-level scatter with cos²θ anisotropy fit.
    Takes segment CSV path directly.
    """
    if not os.path.exists(seg_path):
        print(f"No segment CSV found at {seg_path}")
        return

    df = pd.read_csv(seg_path).dropna(subset=['incidence_deg', 'beta'])
    if len(df) == 0:
        print("No valid segment incidence/beta pairs found.")
        return

    has_flow_error = 'flow_error_mean' in df.columns
    if not has_flow_error:
        print(f"No flow_error_mean column in {seg_path} — cannot compute weighted fit.")
        print("Run bed_analysis_19.py with MEaSUREs validation enabled first.")
        return

    print(f"Loaded {len(df)} valid segments from {seg_path}")

    theta = df['incidence_deg'].values
    beta = df['beta'].values
    beta_err = df['beta_uncertainty'].values if 'beta_uncertainty' in df.columns else None
    ferr = df['flow_error_mean'].values
    weights = flow_weight(ferr)

    n_total = len(theta)
    n_nonzero_weight = np.sum(weights > 0)
    print(f"  {n_total} segments, {n_nonzero_weight} with non-zero weight")

    # --- Fits ---
    fit_unw = fit_cos2(theta, beta)
    fit_w = fit_cos2(theta, beta, weights=weights)

    if fit_unw is None and fit_w is None:
        print("Both fits failed.")
        return

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    x_fit = np.linspace(0, 90, 200)

    # Left panel: unweighted (original)
    ax = axes[0]
    if beta_err is not None:
        ax.errorbar(theta, beta, yerr=beta_err, fmt='o', alpha=0.6, ms=5,
                    color='darkorange', ecolor='gray', elinewidth=0.8, capsize=2)
    else:
        ax.scatter(theta, beta, alpha=0.6, s=40, c='darkorange')
    ax.set_title('Unweighted (original)', fontsize=12)
    if fit_unw:
        ax.plot(x_fit, cos2_model(x_fit, *fit_unw['popt']), 'k-', linewidth=2,
                label=(f"$\\beta_\\parallel$={fit_unw['beta_par']:.2f}$\\pm${fit_unw['perr'][1]:.2f}\n"
                       f"$\\beta_\\perp$={fit_unw['beta_perp']:.2f}$\\pm${fit_unw['perr'][0]:.2f}\n"
                       f"$\\Delta\\beta$={fit_unw['delta']:+.2f}$\\pm${fit_unw['delta_se']:.2f}, R²={fit_unw['r2']:.3f}"))
        ax.legend(fontsize=9)

    # Right panel: weighted by flow confidence
    ax = axes[1]
    if beta_err is not None:
        ax.errorbar(theta, beta, yerr=beta_err, fmt='none', ecolor='gray',
                    elinewidth=0.8, capsize=2, alpha=0.5)
    sc = ax.scatter(theta, beta, alpha=0.6, s=40, c=weights, cmap='viridis',
                    vmin=0, vmax=1, edgecolors='none')
    ax.set_title('Weighted by flow confidence', fontsize=12)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Weight (1=agree, 0=disagree)', fontsize=9)
    if fit_w:
        ax.plot(x_fit, cos2_model(x_fit, *fit_w['popt']), 'k-', linewidth=2,
                label=(f"$\\beta_\\parallel$={fit_w['beta_par']:.2f}$\\pm${fit_w['perr'][1]:.2f}\n"
                       f"$\\beta_\\perp$={fit_w['beta_perp']:.2f}$\\pm${fit_w['perr'][0]:.2f}\n"
                       f"$\\Delta\\beta$={fit_w['delta']:+.2f}$\\pm${fit_w['delta_se']:.2f}, R²={fit_w['r2']:.3f}"))
        ax.legend(fontsize=9)

    for ax in axes:
        ax.set_xlabel('Incidence Angle (°)')
        ax.set_xlim(-2, 92)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r'Power Law Exponent ($\beta$)')

    # Suptitle with comparison summary
    if fit_unw and fit_w:
        fig.suptitle(
            f'Segment-Level Weighted Anisotropy Comparison (n={n_total} segments)\n'
            f'$\\Delta\\beta$ unweighted: {fit_unw["delta"]:+.3f}  |  '
            f'$\\Delta\\beta$ weighted: {fit_w["delta"]:+.3f}',
            fontsize=13, y=1.02)

    plt.tight_layout()
    output_path = seg_path.replace('_segment_stats.csv', '_seg_weighted_anisotropy.png')
    if output_path == seg_path:
        output_path = seg_path.replace('.csv', '_seg_weighted_anisotropy.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
    # plt.show()
    plt.close()

    # Print comparison table
    print(f"\n{'='*55}")
    print(f"{'':20s} {'Unweighted':>15s} {'Weighted':>15s}")
    print(f"{'-'*55}")
    if fit_unw and fit_w:
        print(f"{'beta_parallel':20s} {fit_unw['beta_par']:>8.3f}±{fit_unw['perr'][1]:<5.3f} {fit_w['beta_par']:>8.3f}±{fit_w['perr'][1]:<.3f}")
        print(f"{'beta_perp':20s} {fit_unw['beta_perp']:>8.3f}±{fit_unw['perr'][0]:<5.3f} {fit_w['beta_perp']:>8.3f}±{fit_w['perr'][0]:<.3f}")
        print(f"{'delta_beta':20s} {fit_unw['delta']:>+8.3f}±{fit_unw['delta_se']:<5.3f} {fit_w['delta']:>+8.3f}±{fit_w['delta_se']:<.3f}")
        print(f"{'R²':20s} {fit_unw['r2']:>14.4f} {fit_w['r2']:>14.4f}")
    print(f"{'='*55}")

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    # Discover available regions
    regions = discover_regions('.')

    if len(sys.argv) > 1:
        # Region specified on command line
        region_arg = sys.argv[1]

        # Check if it's a direct file path
        if region_arg.endswith('.csv'):
            if 'segment' in region_arg:
                plot_segment_scatter_direct(region_arg)
            else:
                plot_comparison(region_arg)
        else:
            # Treat as region name
            if region_arg in regions:
                process_region(region_arg, regions[region_arg])
            else:
                # Try partial match
                matches = [r for r in regions if region_arg.lower() in r.lower()]
                if len(matches) == 1:
                    process_region(matches[0], regions[matches[0]])
                elif len(matches) > 1:
                    print(f"Multiple matches for '{region_arg}':")
                    for m in matches:
                        print(f"  - {m}")
                else:
                    print(f"Region '{region_arg}' not found. Available regions:")
                    for r in sorted(regions.keys()):
                        print(f"  - {r}")
    else:
        # Interactive selection
        selection = select_region(regions)

        if selection == 'ALL':
            for region_name in sorted(regions.keys()):
                process_region(region_name, regions[region_name])
        elif selection:
            process_region(selection, regions[selection])

