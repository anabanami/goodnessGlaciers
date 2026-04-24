import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
import sys

# CSV columns:
#   - trajectory: which flight line
#   - window_id: which window within that line
#   - incidence_deg: mean incidence angle for that window
#   - beta: power law exponent (roughness metric)
#   - relief_m, rms_roughness: morphometrics


"""
  Usage:                                                                                    
  # Interactive - shows menu of all available regions                                       
  python incidence_scatter.py                                                               
                                                                                            
  # Specify region name directly                                                            
  python incidence_scatter.py ASB_ICECAP_2010_Fig4_Aurora_SB                                
                                                                                            
  # Partial match works too                                                                 
  python incidence_scatter.py Aurora                                                        
                                                                                            
  # Direct CSV path still works                                                             
  python incidence_scatter.py ASB_ICECAP_2010_Fig4_Aurora_SB_segment_stats.csv              
                                                                                            
  Features:                                                                                 
  - Auto-discovers all *_segment_stats.csv and *_window_stats.csv files in current directory
  - Interactive menu when multiple regions exist (shows which files each region has)        
  - Process ALL option to batch-process every region                                        
  - Partial matching - typing "Aurora" will match "ASB_ICECAP_2010_Fig4_Aurora_SB"          
  - Backwards compatible - direct CSV paths still work                                      
                                                                                            
  When you add new regions later, just drop the CSV files in the directory and they'll      
  automatically appear in the menu.                                                         
                                    
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


def cos2_model(theta_deg, beta_perp, beta_parallel):
    """
    β(θ) = β⊥ + (β∥ - β⊥) cos²(θ)
    """
    theta_rad = np.radians(theta_deg)
    return beta_perp + (beta_parallel - beta_perp) * np.cos(theta_rad)**2


def bootstrap_cos2_uncertainty(theta, beta, n_boot=2000, block_length=3):
    """Block bootstrap for cos²θ fit with correlated overlapping windows."""
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
            popt_b, _ = optimize.curve_fit(cos2_model, t_boot, b_boot,
                                            p0=[np.mean(b_boot), np.mean(b_boot)],
                                            maxfev=5000)
            boot_params.append(popt_b)
        except (RuntimeError, ValueError):
            continue

    boot_params = np.array(boot_params)
    return np.std(boot_params, axis=0)  # bootstrap standard errors


def plot_window_scatter(csv_path):
    """
    Creates a scatter plot of incidence angle vs beta from exported CSV.
    """
    df = pd.read_csv(csv_path)

    # Drop rows with NaN in either column
    df_clean = df.dropna(subset=['incidence_deg', 'beta'])

    if len(df_clean) == 0:
        print("No valid incidence/beta pairs found.")
        return

    theta = df_clean['incidence_deg'].values
    beta = df_clean['beta'].values
    beta_err = df_clean['beta_uncertainty'].values if 'beta_uncertainty' in df_clean.columns else None

    print(f"Loaded {len(df_clean)} valid windows from {csv_path}")

    fig, ax = plt.subplots(figsize=(10, 8))

    if beta_err is not None and np.any(np.isfinite(beta_err)):
        ax.errorbar(theta, beta, yerr=beta_err, fmt='o', alpha=0.5, ms=3,
                    color='steelblue', ecolor='gray', elinewidth=0.5, capsize=1.5)
    else:
        ax.scatter(theta, beta, alpha=0.5, s=20, c='steelblue')

    ax.set_xlabel('Incidence Angle (degrees)')
    ax.set_ylabel(r'Power Law Exponent ($\beta$)')
    ax.set_title(f'Incidence Angle vs Roughness (n={len(df_clean)} windows)')
    ax.grid(True, alpha=0.3)

    x_fit = np.linspace(0, 90, 200)

    # Linear regression with stats
    if len(df_clean) > 2:
        # Linear fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(theta, beta)
        ax.plot(x_fit, intercept + slope * x_fit, 'r--', alpha=0.7, linewidth=1.5,
                label=f'Linear: slope={slope:.4f}, R²={r_value**2:.3f}, p={p_value:.2e}')
        print(f"  Linear: slope={slope:.4f}, R²={r_value**2:.3f}, p={p_value:.2e}")

        # cos²θ fit
        try:
            low = theta < 30
            high = theta > 60
            p0_par = np.mean(beta[low]) if np.any(low) else np.mean(beta)
            p0_perp = np.mean(beta[high]) if np.any(high) else np.mean(beta)

            popt, pcov = optimize.curve_fit(cos2_model, theta, beta, p0=[p0_perp, p0_par])
            beta_perp, beta_par = popt
            # Bootstrap uncertainty (accounts for window overlap correlation)
            perr = bootstrap_cos2_uncertainty(theta, beta, block_length=3)

            delta = beta_par - beta_perp

            pred = cos2_model(theta, *popt)
            ss_res = np.sum((beta - pred)**2)
            ss_tot = np.sum((beta - np.mean(beta))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            ax.plot(x_fit, cos2_model(x_fit, *popt), 'k-', alpha=0.8, linewidth=2,
                    label=(f'cos²θ: β∥={beta_par:.2f}±{perr[1]:.2f}, '
                           f'β⊥={beta_perp:.2f}±{perr[0]:.2f}, '
                           f'Δβ={delta:+.2f}, R²={r2:.3f}'))

            print(f"  cos²θ fit: β∥={beta_par:.3f}±{perr[1]:.3f}, β⊥={beta_perp:.3f}±{perr[0]:.3f}, Δβ={delta:+.3f}, R²={r2:.4f}")
        except (RuntimeError, ValueError) as e:
            print(f"  cos²θ fit failed: {e}")

        ax.legend(fontsize=8)

    plt.tight_layout()

    # Save next to the CSV
    output_path = csv_path.replace('.csv', '_scatter.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    # plt.show()
    plt.close()


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
        plot_window_scatter(files['window'])
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

    print(f"Loaded {len(df)} valid segments from {seg_path}")

    theta = df['incidence_deg'].values
    beta = df['beta'].values
    beta_err = df['beta_uncertainty'].values if 'beta_uncertainty' in df.columns else None

    fig, ax = plt.subplots(figsize=(10, 8))
    if beta_err is not None:
        ax.errorbar(theta, beta, yerr=beta_err, fmt='o', alpha=0.6, ms=5,
                    color='darkorange', ecolor='gray', elinewidth=0.8, capsize=2)
    else:
        ax.scatter(theta, beta, alpha=0.6, s=40, c='darkorange')
    ax.set_xlabel('Incidence Angle (degrees)')
    ax.set_ylabel(r'Power Law Exponent ($\beta$)')
    ax.set_xlim(-2, 92)
    ax.grid(True, alpha=0.3)

    x_fit = np.linspace(0, 90, 200)

    if len(df) > 2:
        # Linear fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(theta, beta)
        ax.plot(x_fit, intercept + slope * x_fit, 'r--', alpha=0.7, linewidth=1.5,
                label=f'Linear: slope={slope:.4f}, R²={r_value**2:.3f}, p={p_value:.2e}')
        print(f"  Linear: slope={slope:.4f}, R²={r_value**2:.3f}, p={p_value:.2e}")

        # cos²θ fit
        try:
            low = theta < 30
            high = theta > 60
            p0_par = np.mean(beta[low]) if np.any(low) else np.mean(beta)
            p0_perp = np.mean(beta[high]) if np.any(high) else np.mean(beta)

            popt, pcov = optimize.curve_fit(cos2_model, theta, beta, p0=[p0_perp, p0_par])
            beta_perp, beta_par = popt
            perr = np.sqrt(np.diag(pcov))
            delta = beta_par - beta_perp

            pred = cos2_model(theta, *popt)
            ss_res = np.sum((beta - pred)**2)
            ss_tot = np.sum((beta - np.mean(beta))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            ax.plot(x_fit, cos2_model(x_fit, *popt), 'k-', alpha=0.8, linewidth=2,
                    label=(f'cos²θ: β∥={beta_par:.2f}±{perr[1]:.2f}, '
                           f'β⊥={beta_perp:.2f}±{perr[0]:.2f}, '
                           f'Δβ={delta:+.2f}, R²={r2:.3f}'))

            print(f"  cos²θ fit: β∥={beta_par:.3f}±{perr[1]:.3f}, β⊥={beta_perp:.3f}±{perr[0]:.3f}, Δβ={delta:+.3f}, R²={r2:.4f}")
        except (RuntimeError, ValueError) as e:
            print(f"  cos²θ fit failed: {e}")

        ax.legend(fontsize=8)

    ax.set_title(f'Segment-Level Roughness Anisotropy (n={len(df)} segments)')
    plt.tight_layout()

    output_path = seg_path.replace('.csv', '_scatter.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
    # plt.show()
    plt.close()


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
                plot_window_scatter(region_arg)
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

