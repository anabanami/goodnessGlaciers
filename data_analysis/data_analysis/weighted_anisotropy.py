import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from config import Tee, PROCESSING_FLAG_NOTE, processing_flag_of
from plotting import flag_suptitle

# Output configuration - nested inside region output from loading.py
from loading import OUTPUT_BASE_PATH as _REGION_BASE
OUTPUT_BASE_PATH = os.path.join(_REGION_BASE, 'anisotropy/')

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
    """Find all region datasets: window CSVs in window_csvs/, segment CSVs in region subfolders."""
    regions = {}
    # Window CSVs in window_csvs/
    for f in glob.glob(os.path.join(directory, 'window_csvs', '*_window_stats.csv')):
        region = os.path.basename(f).replace('_window_stats.csv', '')
        regions.setdefault(region, {})['window'] = f
    # Segment CSVs in segment_csvs/
    for f in glob.glob(os.path.join(directory, 'segment_csvs', '*_segment_stats.csv')):
        region = os.path.basename(f).replace('_segment_stats.csv', '')
        regions.setdefault(region, {})['segment'] = f
    # Fallback: flat directory (legacy layout)
    if not regions:
        for kind, pattern in [('segment', '*_segment_stats.csv'), ('window', '*_window_stats.csv')]:
            for f in glob.glob(os.path.join(directory, pattern)):
                region = os.path.basename(f).replace(f'_{kind}_stats.csv', '')
                regions.setdefault(region, {})[kind] = f
    return regions


def select_region(regions):
    """Interactive region selection if multiple regions available."""
    if not regions:
        print("No region datasets found (*_segment_stats.csv or *_window_stats.csv)")
        return None
    if len(regions) == 1:
        region = list(regions.keys())[0]
        print(f"Found 1 region: {region}")
        return region

    sorted_regions = sorted(regions.keys())
    print(f"\nFound {len(regions)} regions:")
    for i, r in enumerate(sorted_regions, 1):
        f = regions[r]
        print(f"  {i}. {r} [seg: {'Y' if 'segment' in f else 'N'}, win: {'Y' if 'window' in f else 'N'}]")
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


def flow_weight(flow_error, speed=None, angle_cutoff=60.0, speed_cutoff=5.0):
    """
    Combined weight from flow direction agreement and velocity magnitude.
    - Angle component: linear decay from 1.0 at 0° to 0.0 at angle_cutoff.
    - Speed component: 0.0 below speed_cutoff, linear ramp to 1.0 at 2*speed_cutoff.
    Final weight is the product of both components.
    """
    w = np.clip(1.0 - flow_error / angle_cutoff, 0.0, 1.0)
    w[np.isnan(flow_error)] = 0.0
    if speed is not None:
        w_speed = np.clip((speed - speed_cutoff) / speed_cutoff, 0.0, 1.0)
        w_speed[np.isnan(speed)] = 0.0
        w *= w_speed
    return w


def cos2_model(theta_deg, beta_perp, beta_parallel):
    """β(θ) = β⊥ + (β∥ - β⊥) cos²(θ)"""
    return beta_perp + (beta_parallel - beta_perp) * np.cos(np.radians(theta_deg))**2


def _do_curve_fit(theta, beta, weights, p0):
    if weights is not None:
        sigma = np.where(weights > 0, 1.0 / weights, 1e10)
        return optimize.curve_fit(cos2_model, theta, beta, p0=p0,
                                  sigma=sigma, absolute_sigma=False, maxfev=5000)
    return optimize.curve_fit(cos2_model, theta, beta, p0=p0, maxfev=5000)


def bootstrap_cos2_uncertainty(theta, beta, weights=None, n_boot=2000, block_length=3):
    """Block bootstrap for cos²θ fit, optionally weighted."""
    n = len(theta)
    boot_params = []
    for _ in range(n_boot):
        n_blocks = int(np.ceil(n / block_length))
        starts = np.random.randint(0, n, size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block_length, n)) for s in starts])[:n]
        try:
            w_boot = weights[idx] if weights is not None else None
            popt, _ = _do_curve_fit(theta[idx], beta[idx], w_boot, p0=[np.mean(beta), np.mean(beta)])
            boot_params.append(popt)
        except (RuntimeError, ValueError):
            continue

    boot_params = np.array(boot_params)
    return np.std(boot_params, axis=0), np.std(boot_params[:, 1] - boot_params[:, 0])


def fit_cos2(theta, beta, weights=None):
    """Fit cos²θ model, return dict with fit results or None on failure."""
    low, high = theta < 30, theta > 60
    p0_par = np.mean(beta[low]) if np.any(low) else np.mean(beta)
    p0_perp = np.mean(beta[high]) if np.any(high) else np.mean(beta)

    try:
        popt, _ = _do_curve_fit(theta, beta, weights, p0=[p0_perp, p0_par])
        beta_perp, beta_par = popt
        perr, delta_se = bootstrap_cos2_uncertainty(theta, beta, weights=weights)

        pred = cos2_model(theta, *popt)
        if weights is not None:
            ss_res = np.sum(weights * (beta - pred)**2)
            ss_tot = np.sum(weights * (beta - np.average(beta, weights=weights))**2)
        else:
            ss_res = np.sum((beta - pred)**2)
            ss_tot = np.sum((beta - np.mean(beta))**2)

        return dict(beta_par=beta_par, beta_perp=beta_perp,
                    delta=beta_par - beta_perp, delta_se=delta_se,
                    perr=perr, r2=1 - ss_res / ss_tot if ss_tot > 0 else 0, popt=popt)
    except (RuntimeError, ValueError) as e:
        print(f"  Fit failed: {e}")
        return None


def _fit_label(fit):
    return (f"$\\beta_\\parallel$={fit['beta_par']:.2f}$\\pm${fit['perr'][1]:.2f}\n"
            f"$\\beta_\\perp$={fit['beta_perp']:.2f}$\\pm${fit['perr'][0]:.2f}\n"
            f"$\\Delta\\beta$={fit['delta']:+.2f}$\\pm${fit['delta_se']:.2f}, R²={fit['r2']:.3f}")


def _print_comparison(fit_unw, fit_w):
    print(f"\n{'='*55}")
    print(f"{'':20s} {'Unweighted':>15s} {'Weighted':>15s}")
    print(f"{'-'*55}")
    if fit_unw and fit_w:
        for label, key, idx in [('beta_parallel', 'beta_par', 1), ('beta_perp', 'beta_perp', 0)]:
            print(f"{label:20s} {fit_unw[key]:>8.3f}±{fit_unw['perr'][idx]:<5.3f} {fit_w[key]:>8.3f}±{fit_w['perr'][idx]:<.3f}")
        print(f"{'delta_beta':20s} {fit_unw['delta']:>+8.3f}±{fit_unw['delta_se']:<5.3f} {fit_w['delta']:>+8.3f}±{fit_w['delta_se']:<.3f}")
        print(f"{'R²':20s} {fit_unw['r2']:>14.4f} {fit_w['r2']:>14.4f}")
    print(f"{'='*55}")


def plot_anisotropy(csv_path, level='window'):
    """Unified anisotropy comparison plot for window or segment level data."""
    df = pd.read_csv(csv_path).dropna(subset=['incidence_deg', 'beta'])
    if 'is_transition' in df.columns:
        n_tz = int(df['is_transition'].sum())
        if n_tz:
            df = df[~df['is_transition']].copy()
            print(f"  Excluded {n_tz} transition windows from anisotropy fit ({len(df)} remain)")
    if len(df) == 0:
        print("No valid data."); return

    pflag = processing_flag_of(df)
    if pflag:
        print(f"  processing: {PROCESSING_FLAG_NOTE.get(pflag, pflag)}")

    if 'flow_error_mean' not in df.columns:
        print(f"No flow_error_mean column in {csv_path} — cannot compute weighted fit.")
        print("Run bed_analysis_20.py with MEaSUREs validation enabled first.")
        return

    theta = df['incidence_deg'].values
    beta = df['beta'].values
    beta_err = df['beta_uncertainty'].values if 'beta_uncertainty' in df.columns else None
    speed = df['measures_speed_mean'].values if 'measures_speed_mean' in df.columns else None
    weights = flow_weight(df['flow_error_mean'].values, speed=speed)
    if speed is not None:
        n_slow = np.sum(speed < 5.0)
        print(f"  {n_slow} {level}s with MEaSUREs speed < 5 m/yr (down-weighted)")

    n_total = len(theta)
    n_valid = np.sum(weights > 0)
    print(f"Loaded {n_total} {level}s, {n_valid} with non-zero weight")

    if n_valid == 0:
        print(f"  FLOW-AMBIGUOUS: all {level}s have zero weight (ice speed too low).")
        print(f"  Incidence angles unreliable — skipping anisotropy fit.")
        return

    fit_unw = fit_cos2(theta, beta)
    fit_w = fit_cos2(theta, beta, weights=weights)
    if fit_unw is None and fit_w is None:
        print("Both fits failed."); return

    # Style per level
    is_seg = level == 'segment'
    color, ms, s = ('darkorange', 5, 40) if is_seg else ('steelblue', 3, 20)
    elw, cap = (0.8, 2) if is_seg else (0.5, 1.5)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    x_fit = np.linspace(0, 90, 200)

    # Left: unweighted
    ax = axes[0]
    if beta_err is not None and np.any(np.isfinite(beta_err)):
        ax.errorbar(theta, beta, yerr=beta_err, fmt='o', alpha=0.5 + 0.1*is_seg,
                    ms=ms, color=color, ecolor='gray', elinewidth=elw, capsize=cap)
    else:
        ax.scatter(theta, beta, alpha=0.5 + 0.1*is_seg, s=s, c=color)
    ax.set_title('Unweighted (original)', fontsize=12)
    if fit_unw:
        ax.plot(x_fit, cos2_model(x_fit, *fit_unw['popt']), 'k-', lw=2, label=_fit_label(fit_unw))
        ax.legend(fontsize=9)

    # Right: weighted
    ax = axes[1]
    if beta_err is not None and np.any(np.isfinite(beta_err)):
        ax.errorbar(theta, beta, yerr=beta_err, fmt='none', ecolor='gray',
                    elinewidth=elw, capsize=cap, alpha=0.5)
    sc = ax.scatter(theta, beta, alpha=0.6, s=s, c=weights, cmap='viridis',
                    vmin=0, vmax=1, edgecolors='none')
    ax.set_title('Weighted by flow confidence', fontsize=12)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Weight (1=agree, 0=disagree)', fontsize=9)
    if fit_w:
        ax.plot(x_fit, cos2_model(x_fit, *fit_w['popt']), 'k-', lw=2, label=_fit_label(fit_w))
        ax.legend(fontsize=9)

    for ax in axes:
        ax.set_xlabel('Incidence Angle (°)')
        ax.set_xlim(-2, 92)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r'Power Law Exponent ($\beta$)')

    if fit_unw and fit_w:
        lvl = 'Segment' if is_seg else 'Window'
        flag_suptitle(
            fig,
            f'{lvl}-Level Weighted Anisotropy Comparison (n={n_total} {level}s)\n'
            f'$\\Delta\\beta$ unweighted: {fit_unw["delta"]:+.3f}  |  '
            f'$\\Delta\\beta$ weighted: {fit_w["delta"]:+.3f}',
            pflag, fontsize=13)

    plt.tight_layout()
    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
    basename = os.path.basename(csv_path)
    suffix = '_seg_weighted_anisotropy.png' if is_seg else '_weighted_anisotropy.png'
    out_name = basename.replace(f'_{level}_stats.csv', suffix)
    if out_name == basename:
        out_name = basename.replace('.csv', suffix)
    output_path = os.path.join(OUTPUT_BASE_PATH, out_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    _print_comparison(fit_unw, fit_w)
    print(f"\nSaved to {output_path}")
    return {'unweighted': fit_unw, 'weighted': fit_w, 'n': n_total, 'n_valid': int(n_valid)}


def _cross_scale_comparison(win_fits, seg_fits, n_win=0, n_seg=0,
                            n_win_valid=0, n_seg_valid=0, min_n=20):
    """Compare Δβ between window and segment scales via z-score."""
    print(f"\n{'='*55}")
    print("CROSS-SCALE COMPARISON  (window vs segment Δβ)")
    print(f"{'-'*55}")
    low_n = []
    if n_win < min_n:
        low_n.append(f"windows (n={n_win})")
    if n_seg < min_n:
        low_n.append(f"segments (n={n_seg})")
    if low_n:
        print(f"  ** LOW SAMPLE SIZE: {', '.join(low_n)} < {min_n} — "
              f"bootstrap SEs unreliable, interpret with caution **")
    low_n_eff = []
    if n_win_valid < min_n:
        low_n_eff.append(f"windows (n_eff={n_win_valid})")
    if n_seg_valid < min_n:
        low_n_eff.append(f"segments (n_eff={n_seg_valid})")
    for label, key in [('Unweighted', 'unweighted'), ('Weighted', 'weighted')]:
        fw, fs = win_fits.get(key), seg_fits.get(key)
        if fw is None or fs is None:
            print(f"  {label}: fit missing — skipped")
            continue
        if key == 'weighted' and low_n_eff:
            print(f"  ** LOW EFFECTIVE SAMPLE SIZE: {', '.join(low_n_eff)} < {min_n} — "
                  f"weighted bootstrap SEs unreliable, interpret with caution **")
        diff = fw['delta'] - fs['delta']
        se = np.sqrt(fw['delta_se']**2 + fs['delta_se']**2)
        z = diff / se if se > 0 else np.inf
        verdict = 'CONSISTENT' if abs(z) < 2 else 'INCONSISTENT'
        print(f"  {label}:")
        print(f"    Window  Δβ = {fw['delta']:+.3f} ± {fw['delta_se']:.3f}")
        print(f"    Segment Δβ = {fs['delta']:+.3f} ± {fs['delta_se']:.3f}")
        print(f"    Difference  = {diff:+.3f},  z = {abs(z):.2f}  →  {verdict} (|z|<2)")
    print(f"{'='*55}")


def process_region(region_name, files):
    print(f"\n{'='*60}\nProcessing: {region_name}\n{'='*60}")
    fits = {}
    for level in ['window', 'segment']:
        if level in files:
            fits[level] = plot_anisotropy(files[level], level=level)
        else:
            print(f"  No {level} stats file for {region_name}")
    if 'window' in fits and 'segment' in fits and fits['window'] and fits['segment']:
        _cross_scale_comparison(fits['window'], fits['segment'],
                                n_win=fits['window']['n'],
                                n_seg=fits['segment']['n'],
                                n_win_valid=fits['window']['n_valid'],
                                n_seg_valid=fits['segment']['n_valid'])


if __name__ == "__main__":
    regions = discover_regions(_REGION_BASE)
    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
    log_path = os.path.join(OUTPUT_BASE_PATH, 'weighted_anisotropy_log.txt')
    sys.stdout = Tee(log_path)

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.endswith('.csv'):
            level = 'segment' if 'segment' in arg else 'window'
            plot_anisotropy(arg, level=level)
        elif arg in regions:
            process_region(arg, regions[arg])
        else:
            matches = [r for r in regions if arg.lower() in r.lower()]
            if len(matches) == 1:
                process_region(matches[0], regions[matches[0]])
            elif matches:
                print(f"Multiple matches for '{arg}':"); [print(f"  - {m}") for m in matches]
            else:
                print(f"Region '{arg}' not found. Available:"); [print(f"  - {r}") for r in sorted(regions)]
    else:
        selection = select_region(regions)
        if selection == 'ALL':
            for r in sorted(regions):
                process_region(r, regions[r])
        elif selection:
            process_region(selection, regions[selection])
