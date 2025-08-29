"""
Batch Phase Analysis - Wrapper to run phase_analysis.py iteratively for multiple NC files
"""
import os
import sys
import glob
import re
import netCDF4 as nc

# Import functions from phase_analysis.py
from phase_analysis import (
    load_config, 
    analyse_phase_evolution
)

# Add parent directory to sys.path and set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
# From processing/Transfer analysis, go up to PERIODIC directory
processing_dir = os.path.dirname(script_dir)  # Go up to processing
periodic_dir = os.path.dirname(processing_dir)  # Go up to PERIODIC
parent_dir = os.path.dirname(os.path.dirname(periodic_dir))  # Go up to reach where bedrock_profiles might be
sys.path.insert(0, parent_dir)

# Look for bedrock profiles in multiple possible locations
possible_bedrock_dirs = [
    os.path.join(parent_dir, 'bedrock_profiles'),
    os.path.join(periodic_dir, 'bedrock_profiles'), 
    os.path.join(periodic_dir, 'useful_scripts', 'old_bedrock_database', 'bedrock_profiles')
]

BEDROCK_DIR = None
for bd in possible_bedrock_dirs:
    if os.path.exists(bd):
        BEDROCK_DIR = bd
        print(f"Found bedrock directory: {BEDROCK_DIR}")
        break

if BEDROCK_DIR is None:
    print("Warning: No bedrock profiles directory found!")

print(f"Script directory: {script_dir}")
print(f"Processing directory: {processing_dir}")
print(f"PERIODIC directory: {periodic_dir}")
print(f"Parent directory: {parent_dir}")

def parse_filename(ncfile):
    """Extract profile ID and experiment from filename - batch version without sys.argv check"""
    bed = os.path.splitext(os.path.basename(ncfile))[0]
    profile_str, exp = bed.split('_')[:2]
    profile_id = int(profile_str)
    print(f"Profile ID: {profile_id:03d}, Experiment: {exp}")
    return profile_id, exp

def process_single_file(nc_path, bedrock_file, scenario_out):
    """Process a single NetCDF file using phase_analysis functions"""
    # Extract profile_id and experiment tag
    profile_id, exp = parse_filename(nc_path)
    config, wavelength_km = load_config(nc_path, bedrock_file)

    # Create output directory for this file
    out_dir = os.path.join(scenario_out, f"{profile_id:03d}_{exp}_phase_analysis")
    os.makedirs(out_dir, exist_ok=True)

    # Open dataset and analyse using imported function
    with nc.Dataset(nc_path, 'r') as ds:
        # Check how many time steps are in the dataset
        tsol = ds.groups['results'].groups['TransientSolution']
        times = tsol.variables['time'][:]
        print(f"Dataset contains {len(times)} time steps: {times}")
        
        try:
            results = analyse_phase_evolution(ds, wavelength_km, config, profile_id, out_dir)
        except Exception as e:
            print(f"Error during phase analysis: {e}")
            import traceback
            traceback.print_exc()
            raise

    # Write summary results to file
    summary_file = os.path.join(out_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"# Phase Evolution Summary for {profile_id:03d}_{exp}\n")
        f.write(f"Wavelength (km): {results['wavelength']:.3f}\n")
        f.write("Time (yr), Phase (deg), Lag (km), Corr\n")
        for t, deg, lag, corr in zip(
            results['time'],
            results['phase_shift_deg'],
            results['lag_distance'],
            results['correlation']
        ):
            f.write(f"{t:.2f}, {deg:.1f}, {lag:.3f}, {corr:.3f}\n")
    
    print(f"Completed analysis for {profile_id:03d}_{exp}")
    return results


def main():
    """Main batch processing function"""
    if BEDROCK_DIR is None:
        print("Error: No bedrock profiles directory found! Cannot proceed.")
        return
        
    # Get bedrock profiles - pointing to the correct location
    bedrock_files = glob.glob(os.path.join(BEDROCK_DIR, "bedrock_profile_*.npz"))
    bedrock_profiles = {re.search(r'(\d{3})\.npz$', f).group(1): f for f in bedrock_files}

    # Configuration - look for scenario directories in the PERIODIC directory
    SCENARIOS = ['S1', 'S2', 'S3', 'S4']
    OUTPUT_ROOT = os.path.join(script_dir, 'phase_analysis_results')
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print(f"Starting batch phase analysis for scenarios: {SCENARIOS}")
    print(f"Found {len(bedrock_profiles)} bedrock profiles")
    print(f"Output directory: {OUTPUT_ROOT}")

    for scenario in SCENARIOS:
        print(f"\n=== Processing scenario {scenario} ===")
        scen_path = os.path.join(periodic_dir, scenario)
        
        if not os.path.exists(scen_path):
            print(f"Scenario directory {scen_path} not found, skipping.")
            continue
            
        nc_files = sorted(f for f in os.listdir(scen_path) if f.endswith('.nc'))
        scenario_out = os.path.join(OUTPUT_ROOT, scenario)
        os.makedirs(scenario_out, exist_ok=True)

        print(f"Found {len(nc_files)} NetCDF files in {scenario}")

        for nc_name in nc_files:
            nc_path = os.path.join(scen_path, nc_name)
            print(f"\nProcessing: {nc_name}")
            
            try:
                # Extract profile_id and experiment tag
                profile_id, exp = parse_filename(nc_path)

                # Locate corresponding bedrock file
                bedrock_file = glob.glob(
                                 os.path.join(BEDROCK_DIR, f"*{profile_id:03d}.npz")
                             )

                if not bedrock_file:
                    print(f"No bedrock .npz found for profile {profile_id:03d}, skipping.")
                    continue

                # Process the file
                results = process_single_file(nc_path, bedrock_file[0], scenario_out)
                
            except Exception as e:
                print(f"Error processing {nc_name}: {e}")
                continue

    print(f"\nBatch processing complete! Results saved to: {OUTPUT_ROOT}")


if __name__ == '__main__':
    main()
