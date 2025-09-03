#!/usr/bin/env python3
"""
Batch processing script for extract_results.py

This script has been updated to be fully compatible with the provided
extract_results.py script by incorporating robust post-run checks.
It automatically discovers and processes NetCDF files, generating
visualizations for each result.

Key Improvements for Compatibility:
-   **Silent Failure Detection:** Actively checks for the creation of output
    .png files. If a worker process exits with a "success" code but fails
    to produce any plots, it is correctly flagged as a "Silent Failure".
-   **Smarter Status Reporting:** Differentiates between true successes
    (plots generated), valid skips (e.g., file had only one time step),
    and various failure modes.
-   **Enhanced Logging:** Provides more detailed summary reports to diagnose
    any issues that arise during the batch run.

Usage:
    python batch_extract_results.py                    # Process all .nc files
    python batch_extract_results.py --pattern="*_S4_*" # Process only S4 experiments
    python batch_extract_results.py --parallel         # Process multiple files in parallel
    python batch_extract_results.py --skip-existing    # Skip files with existing output directories
    python batch_extract_results.py --resume           # Resume from last processed file
"""

import os
import glob
import subprocess
import sys
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

def find_nc_files(directory=".", pattern="*.nc"):
    """Find all NetCDF files matching the pattern."""
    search_pattern = os.path.join(directory, pattern)
    files = glob.glob(search_pattern)
    # Filter to only include files (not directories) with .nc extension
    nc_files = [f for f in files if os.path.isfile(f) and f.endswith('.nc')]
    return sorted(nc_files)

def get_output_directory(nc_file):
    """Get the expected output directory name for a NetCDF file."""
    return os.path.splitext(os.path.basename(nc_file))[0]

def check_for_output(nc_file):
    """Check if the file has any generated plots."""
    output_dir = get_output_directory(nc_file)
    if not os.path.exists(output_dir):
        return False
    
    # Check if directory has at least one PNG file (indicating successful processing)
    # Using Path.rglob is efficient for recursively finding files.
    if next(Path(output_dir).rglob("*.png"), None):
        return True
    return False

def process_single_file(nc_file):
    """
    Process a single NetCDF file using extract_results.py.

    This function includes enhanced checks to detect "silent failures" where the
    worker script exits with a success code (0) but fails to produce output plots.
    """
    print(f"\n🔄 Processing: {nc_file}")
    start_time = time.time()
    
    try:
        # Run extract_results.py as a subprocess.
        result = subprocess.run(
            [sys.executable, "extract_results.py", nc_file],
            capture_output=True,
            text=True,
            timeout=900  # 15 minute timeout per file
        )
        
        processing_time = time.time() - start_time
        
        # --- Enhanced Success/Failure Logic ---

        # 1. Check for an explicit failure code from the subprocess. This is a clear error.
        if result.returncode != 0:
            print(f"❌ Failed (Exit Code {result.returncode}): {nc_file}")
            error_output = result.stderr if result.stderr else result.stdout
            print(f"Error Details: {error_output.strip()}")
            return {"file": nc_file, "status": "error", "time": processing_time, "error": error_output}

        # 2. Check for the actual expected output (plots). This is the true measure of success.
        if check_for_output(nc_file):
             print(f"✅ Completed: {nc_file} ({processing_time:.1f}s)")
             return {"file": nc_file, "status": "success", "time": processing_time, "output": result.stdout}

        # 3. Handle cases where no output is valid. The worker script prints "Skipping."
        #    for files with only one time step, which is a valid reason for no plots.
        if "Skipping." in result.stdout:
            print(f"✅ Skipped (by worker): {nc_file} ({processing_time:.1f}s)")
            return {"file": nc_file, "status": "skipped", "time": processing_time, "output": result.stdout}

        # 4. If exit code was 0 but no plots were produced and it wasn't a valid skip,
        #    then we have detected a SILENT FAILURE.
        print(f"❌ Failed (Silent): {nc_file} - Script finished but produced no plots.")
        error_output = result.stdout if result.stdout.strip() else "No output from worker script."
        return {"file": nc_file, "status": "silent_failure", "time": processing_time, "error": error_output}

    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout: {nc_file} (exceeded 15 minutes)")
        return {"file": nc_file, "status": "timeout", "time": 900, "error": "Process timed out after 15 minutes."}
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"💥 Exception: {nc_file} - {str(e)}")
        return {"file": nc_file, "status": "exception", "time": processing_time, "error": str(e)}

def save_resume_state(processed_files, resume_file="batch_resume.txt"):
    """Save the list of processed files for resume functionality."""
    with open(resume_file, 'w') as f:
        for file in processed_files:
            f.write(f"{file}\n")

def load_resume_state(resume_file="batch_resume.txt"):
    """Load the list of already processed files."""
    if not os.path.exists(resume_file):
        return set()
    
    with open(resume_file, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def write_summary_report(results, output_file="batch_processing_summary.txt"):
    """Write a detailed summary report of the batch processing results."""
    successful = [r for r in results if r['status'] == 'success']
    skipped = [r for r in results if r['status'] == 'skipped']
    failed = [r for r in results if r['status'] not in ['success', 'skipped']]
    
    total_time = sum(r['time'] for r in results)
    avg_time = total_time / len(results) if results else 0
    
    with open(output_file, 'w') as f:
        f.write("Batch Processing Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total files processed: {len(results)}\n")
        f.write(f"  - Successful (plots generated): {len(successful)}\n")
        f.write(f"  - Skipped (by worker script):   {len(skipped)}\n")
        f.write(f"  - Failed:                       {len(failed)}\n\n")
        f.write(f"Total processing time: {total_time:.1f}s ({total_time/60:.1f} minutes)\n")
        f.write(f"Average time per file: {avg_time:.1f}s\n\n")
        
        if successful or skipped:
            f.write("Completed Files:\n")
            f.write("-" * 20 + "\n")
            for r in successful:
                f.write(f"  [SUCCESS] {r['file']} ({r['time']:.1f}s)\n")
            for r in skipped:
                f.write(f"  [SKIPPED] {r['file']} ({r['time']:.1f}s)\n")
            f.write("\n")
        
        if failed:
            f.write("Failed Files & Detailed Errors:\n")
            f.write("-" * 20 + "\n")
            for r in failed:
                error_msg = r.get('error', 'Unknown error').strip().replace('\n', ' | ')
                f.write(f"  - FILE: {r['file']}\n")
                f.write(f"    STATUS: {r['status'].upper()}\n")
                f.write(f"    DETAILS: {error_msg}\n\n")

def main():
    parser = argparse.ArgumentParser(description="Batch process NetCDF files with extract_results.py")
    parser.add_argument("--pattern", default="*.nc", help="File pattern to match (default: *.nc)")
    parser.add_argument("--directory", default=".", help="Directory to search for files (default: current)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files with existing output plots")
    parser.add_argument("--parallel", action="store_true", help="Process multiple files in parallel")
    parser.add_argument("--max-workers", type=int, default=None, help="Max parallel workers (default: all CPU cores)")
    parser.add_argument("--resume", action="store_true", help="Resume from previous batch run")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without actually processing")
    
    args = parser.parse_args()
    
    nc_files = find_nc_files(args.directory, args.pattern)
    
    if not nc_files:
        print(f"No NetCDF files found matching pattern '{args.pattern}' in '{args.directory}'")
        return
    
    print(f"Found {len(nc_files)} NetCDF files matching pattern '{args.pattern}'")
    
    processed_files = set()
    if args.resume:
        processed_files = load_resume_state()
        print(f"Resume mode: Found {len(processed_files)} previously processed files")
    
    files_to_process = []
    for nc_file in nc_files:
        if args.resume and nc_file in processed_files:
            print(f"🔄 Skipping (resume): {nc_file}")
            continue
        if args.skip_existing and check_for_output(nc_file):
            print(f"📁 Skipping (exists): {nc_file}")
            continue
        files_to_process.append(nc_file)
    
    if not files_to_process:
        print("No new files to process after filtering.")
        return
    
    print(f"\n📋 Files to process: {len(files_to_process)}")
    for i, file in enumerate(files_to_process, 1):
        print(f"  {i:2d}. {file}")
    
    if args.dry_run:
        print("\n🔍 Dry run complete - no files were processed.")
        return
    
    try:
        input("\nPress Enter to continue or Ctrl+C to cancel...")
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    
    start_time = time.time()
    results = []
    
    if args.parallel:
        max_workers = args.max_workers or cpu_count()
        print(f"\n🚀 Starting parallel processing with up to {max_workers} workers...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(process_single_file, nc_file): nc_file 
                             for nc_file in files_to_process}
            
            for i, future in enumerate(as_completed(future_to_file), 1):
                result = future.result()
                results.append(result)
                processed_files.add(result['file'])
                if i % 5 == 0 or i == len(files_to_process):
                    save_resume_state(processed_files)
                print(f"Progress: {i}/{len(files_to_process)} files completed")
    else:
        print(f"\n🔄 Starting sequential processing...")
        for i, nc_file in enumerate(files_to_process, 1):
            print(f"\nProgress: {i}/{len(files_to_process)}")
            result = process_single_file(nc_file)
            results.append(result)
            processed_files.add(nc_file)
            if i % 5 == 0 or i == len(files_to_process):
                save_resume_state(processed_files)
    
    total_time = time.time() - start_time
    successful = [r for r in results if r['status'] == 'success']
    skipped = [r for r in results if r['status'] == 'skipped']
    failed = [r for r in results if r['status'] not in ['success', 'skipped']]
    
    print(f"\n" + "="*60)
    print(f"BATCH PROCESSING COMPLETE")
    print(f"="*60)
    print(f"📊 Total files processed: {len(results)}")
    print(f"✅ Successful (plots): {len(successful)}")
    print(f"ℹ️  Skipped (by worker): {len(skipped)}")
    print(f"❌ Failed: {len(failed)}")
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    if results:
        print(f"⚡ Avg time/file: {total_time/len(results):.1f}s")
    
    if failed:
        print(f"\n❌ Failed files (see report for details):")
        for r in failed:
            print(f"  - {r['file']} (Reason: {r['status']})")
    
    write_summary_report(results)
    print(f"\n📝 Detailed report written to: batch_processing_summary.txt")
    
    if not failed and os.path.exists("batch_resume.txt"):
        os.remove("batch_resume.txt")
        print("🧹 Cleaned up resume file.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⛔ Process interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 An unexpected error occurred: {e}")
        sys.exit(1)