#!/usr/bin/env python3
"""
Batch plotting script that combines velocity and shear stress plot functionality.

Automatically discovers all .txt files in the current directory and generates:
1. Velocity plot (surface vs basal velocity)
2. Velocity colored by oscillations over bed topography
3. Shear stress oscillations over bed topography

Usage:
    python batch_plots.py [--parallel] [--skip-existing]
"""

import glob
import os
import sys
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# Get parent directory and add to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from bedrock_generator import SyntheticBedrockModelConfig

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.signal import savgol_filter


def read_slice(fname):
    """Read velocity data from text file."""
    d = np.loadtxt(fname, comments='#')
    xhat, vx_s, vz_s, vx_b = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    
    # Sort by x_hat to ensure proper ordering for plotting
    order = np.argsort(xhat)
    xhat_sorted = xhat[order]
    vx_s_sorted = vx_s[order]
    vx_b_sorted = vx_b[order]

    return xhat_sorted, vx_s_sorted, vx_b_sorted


def generate_velocity_plot(fname, profile_id, exp, xhat, vx_s, vx_b):
    """Generate basic velocity plot."""
    plt.figure(figsize=(16, 5))
    plt.plot(xhat, vx_s, lw=2, label='surface velocity')
    plt.plot(xhat, vx_b, lw=2, label='basal velocity')
    # plt.xlim(0, 1)
    plt.xlabel("Normalized x")
    plt.ylabel("Velocity (m a⁻¹)")
    plt.grid(True, linestyle=":", color='k', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    
    output_file = f"{profile_id:03d}_{exp}_velocity.png"
    plt.savefig(output_file, dpi=150)
    plt.close()
    return output_file


def generate_velocity_colored_plot(fname, profile_id, exp, xhat, vx_b, bed_elevation):
    """Generate velocity plot colored by oscillations over bed topography."""
    # Smooth version (large-scale trend)
    vx_b_smooth = savgol_filter(vx_b, window_length=101, polyorder=3)
    
    # Subtract trend to isolate oscillations
    vx_b_oscillations = vx_b - vx_b_smooth
    
    # Normalize to the range of oscillations
    max_dev = np.max(np.abs(vx_b_oscillations))
    norm = colors.Normalize(vmin=-max_dev, vmax=+max_dev)
    
    # Plot with color representing oscillations
    plt.figure(figsize=(16, 5))
    sc = plt.scatter(xhat, bed_elevation, s=10, c=vx_b_oscillations, cmap='viridis', norm=norm)
    cbar = plt.colorbar(sc)
    cbar.set_label('basal velocity oscillations (m/y)')
    plt.grid(True, linestyle=":", color='k', alpha=0.4)
    # plt.xlim(0, 1)
    plt.xlabel("Normalized x")
    plt.ylabel("Bed elevation (m)")
    plt.title("Basal velocity over bed topography", loc='left', fontsize=10)
    plt.tight_layout()
    
    output_file = f"{profile_id:03d}_{exp}_velocity_color.png"
    plt.savefig(output_file, dpi=150)
    plt.close()
    return output_file


def generate_shear_stress_plot(fname, profile_id, exp, xhat, vx_b, bed_elevation, bedrock_config):
    """Generate shear stress plot over bed topography."""
    L = 160  # km
    
    # Constants & material properties
    yts = 31556926  # s/yr
    A = 1e-16 / yts  # from table 1 in Pattyn 2008
    rheology_n = 3
    rheology_B = A ** (-1/rheology_n)
    eta = rheology_B / 2
    
    # Bedrock parameters
    wavelength = bedrock_config.profile_params['wavelength']  # meters
    omega = 2 * np.pi / wavelength  # 1/meters
    amplitude = bedrock_config.profile_params['amplitude']  # meters
    beta_1 = amplitude  # bed slope amplitude in meters
    
    V_avg = np.mean(vx_b)
    
    # Budd's basal shear stress
    x_coords = xhat * L * 1e3  # in meters
    tau_xz = 2 * eta * V_avg * omega * beta_1 * np.cos(omega * x_coords)
    
    # Smooth version (large-scale trend)
    tau_xz_smooth = savgol_filter(tau_xz, window_length=101, polyorder=3)
    # Subtract trend to isolate oscillations
    tau_xz_oscillations = tau_xz - tau_xz_smooth
    
    # Normalize to the range of oscillations
    max_dev = np.max(np.abs(tau_xz_oscillations))
    norm = colors.Normalize(vmin=-max_dev, vmax=+max_dev)
    
    # Plot with color representing oscillations
    plt.figure(figsize=(16, 5))
    sc = plt.scatter(xhat, bed_elevation, s=10, c=tau_xz_oscillations, cmap='coolwarm', norm=norm)
    cbar = plt.colorbar(sc)
    cbar.set_label('Shear stress oscillations (Pa)')
    plt.grid(True, linestyle=":", color='k', alpha=0.4)
    # plt.xlim(0, 1)
    plt.xlabel("Normalized x")
    plt.ylabel("Bed elevation (m)")
    plt.title("Internal shear stress (at z=0) over bed topography", loc='left', fontsize=10)
    plt.tight_layout()
    
    output_file = f"{profile_id:03d}_{exp}_internal_SS_at_z=0_color.png"
    plt.savefig(output_file, dpi=150)
    plt.close()
    return output_file


def process_single_file(fname, skip_existing=False):
    """Process a single .txt file and generate all three plots."""
    try:
        # Extract profile info from filename
        basename = os.path.basename(fname)
        profile_id = int(basename.split('_')[0])
        exp = str(basename.split('_')[1])
        
        # Check if outputs already exist
        expected_outputs = [
            f"{profile_id:03d}_{exp}_velocity.png",
            f"{profile_id:03d}_{exp}_velocity_color.png",
            f"{profile_id:03d}_{exp}_internal_SS_at_z=0_color.png"
        ]
        
        if skip_existing and all(os.path.exists(f) for f in expected_outputs):
            return {
                "file": fname,
                "status": "skipped",
                "outputs": expected_outputs,
                "message": "All outputs already exist"
            }
        
        # Read velocity data
        xhat, vx_s, vx_b = read_slice(fname)
        
        # Get bedrock configuration
        PROFILE_DIR = os.path.join(parent_dir, "bedrock_profiles")
        bedrock_config = SyntheticBedrockModelConfig(profile_id=profile_id, output_dir=PROFILE_DIR)
        
        # Bed elevation from profile
        L = 175  # km
        bed_elevation = bedrock_config.get_bedrock_elevation(xhat * L * 1e3)  # xhat → meters
        
        # Generate all three plots
        outputs = []
        
        # 1. Basic velocity plot
        output1 = generate_velocity_plot(fname, profile_id, exp, xhat, vx_s, vx_b)
        outputs.append(output1)
        
        # 2. Velocity colored plot
        output2 = generate_velocity_colored_plot(fname, profile_id, exp, xhat, vx_b, bed_elevation)
        outputs.append(output2)
        
        # 3. Shear stress plot
        output3 = generate_shear_stress_plot(fname, profile_id, exp, xhat, vx_b, bed_elevation, bedrock_config)
        outputs.append(output3)
        
        return {
            "file": fname,
            "status": "success",
            "outputs": outputs,
            "profile_id": profile_id,
            "experiment": exp
        }
        
    except Exception as e:
        return {
            "file": fname,
            "status": "error",
            "error": str(e),
            "outputs": []
        }


def find_txt_files(directory="."):
    """Find all .txt files in the directory."""
    pattern = os.path.join(directory, "*.txt")
    txt_files = glob.glob(pattern)
    # Filter to only include files that look like simulation results
    # (have numeric prefix and experiment code)
    valid_files = []
    for f in txt_files:
        basename = os.path.basename(f)
        parts = basename.split('_')
        try:
            # Check if first part is numeric (profile ID)
            int(parts[0])
            if len(parts) >= 2:  # Has experiment code
                valid_files.append(f)
        except (ValueError, IndexError):
            continue
    
    return sorted(valid_files)


def main():
    parser = argparse.ArgumentParser(description="Batch generate velocity and shear stress plots")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files with existing outputs")
    parser.add_argument("--max-workers", type=int, default=None, help="Max parallel workers")
    
    args = parser.parse_args()
    
    # Find all .txt files
    txt_files = find_txt_files()
    
    if not txt_files:
        print("No .txt files found in current directory")
        return
    
    print(f"Found {len(txt_files)} .txt files to process:")
    for f in txt_files:
        print(f"  - {f}")
    
    print(f"\nProcessing options:")
    print(f"  Parallel: {args.parallel}")
    print(f"  Skip existing: {args.skip_existing}")
    
    # Process files
    start_time = time.time()
    results = []
    
    if args.parallel:
        # Parallel processing
        max_workers = args.max_workers or min(cpu_count(), len(txt_files))
        print(f"\nUsing {max_workers} workers for parallel processing...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_file = {
                executor.submit(process_single_file, f, args.skip_existing): f 
                for f in txt_files
            }
            
            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_file), 1):
                result = future.result()
                results.append(result)
                
                status_icon = "✅" if result["status"] == "success" else "⏭️" if result["status"] == "skipped" else "❌"
                print(f"{status_icon} [{i}/{len(txt_files)}] {result['file']}")
    else:
        # Sequential processing
        print(f"\nProcessing sequentially...")
        
        for i, txt_file in enumerate(txt_files, 1):
            print(f"Processing [{i}/{len(txt_files)}]: {txt_file}")
            result = process_single_file(txt_file, args.skip_existing)
            results.append(result)
            
            status_icon = "✅" if result["status"] == "success" else "⏭️" if result["status"] == "skipped" else "❌"
            print(f"{status_icon} {result['file']}")
            if result["status"] == "error":
                print(f"   Error: {result['error']}")
    
    # Summary
    total_time = time.time() - start_time
    successful = [r for r in results if r["status"] == "success"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors = [r for r in results if r["status"] == "error"]
    
    print(f"\n" + "="*60)
    print(f"BATCH PLOTTING COMPLETE")
    print(f"="*60)
    print(f"📊 Total files: {len(txt_files)}")
    print(f"✅ Successfully processed: {len(successful)}")
    print(f"⏭️  Skipped (existing): {len(skipped)}")
    print(f"❌ Errors: {len(errors)}")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"⚡ Avg time/file: {total_time/len(results):.1f}s")
    
    # Count total plots generated
    total_plots = sum(len(r["outputs"]) for r in successful)
    print(f"🎨 Total plots generated: {total_plots}")
    
    if errors:
        print(f"\n❌ Files with errors:")
        for r in errors:
            print(f"  - {r['file']}: {r['error']}")
    
    if successful:
        print(f"\n✅ Successfully processed files:")
        for r in successful:
            profile_id = r.get('profile_id', 'Unknown')
            experiment = r.get('experiment', 'Unknown')
            print(f"  - Profile {profile_id}, Experiment {experiment}: {len(r['outputs'])} plots")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⛔ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)