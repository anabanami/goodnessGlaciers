"""
Batch processing script for phase_analysis.py

This script automatically discovers and processes multiple NetCDF files using
the phase_analysis.py script. It is designed to be robust, allowing for
parallel execution, resuming interrupted runs, and detailed logging.

It includes features like:
-   **Silent Failure Detection:** Checks for the creation of the output
    'summary.txt' file to confirm that the analysis ran successfully,
    preventing cases where the worker script exits cleanly but produces no output.
-   **Pass-through Arguments:** Allows you to specify analysis parameters like
    '--axis' and '--position' that are passed directly to each call of
    phase_analysis.py.
-   **Comprehensive Reporting:** Generates a detailed summary of successful,
    failed, and skipped files.

Usage:
    # Process all .nc files with default analysis (x-axis, center)
    python batch_phase_analysis.py

    # Process S3 experiments in parallel
    python batch_phase_analysis.py --pattern="*_S3_*" --parallel

    # Analyze a specific profile on all files, skipping existing results
    python batch_phase_analysis.py --axis y --position 25000 --skip-existing

    # Resume a previous batch run
    python batch_phase_analysis.py --resume
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
    """Find all NetCDF files matching the pattern in the given directory."""
    search_pattern = os.path.join(directory, pattern)
    files = glob.glob(search_pattern)
    # Ensure we only get files, not directories, ending with .nc
    nc_files = [f for f in files if os.path.isfile(f) and f.endswith('.nc')]
    return sorted(nc_files)

def get_output_directory(nc_file):
    """Determine the expected output directory name for a given NetCDF file."""
    base_name = os.path.splitext(os.path.basename(nc_file))[0]
    return f"{base_name}_phase_analysis"

def check_for_output(nc_file):
    """
    Check if the analysis output (summary.txt) already exists for a file.
    This is the key indicator of a successful prior run.
    """
    output_dir = get_output_directory(nc_file)
    summary_file = os.path.join(output_dir, 'summary.txt')
    return os.path.exists(summary_file)

def process_single_file(nc_file, axis, position):
    """
    Process one NetCDF file using phase_analysis.py and check for success.
    """
    print(f"\n🔄 Processing: {nc_file}")
    start_time = time.time()

    try:
        # Construct the command to run the worker script
        command = [sys.executable, "phase_analysis.py", nc_file, "--axis", axis]
        if position is not None:
            command.extend(["--position", str(position)])

        # Execute the worker script as a subprocess
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=1200  # 20 minute timeout per file
        )

        processing_time = time.time() - start_time

        # --- Enhanced Success/Failure Logic ---

        # 1. Check for a non-zero exit code, which indicates a clear error.
        if result.returncode != 0:
            print(f"❌ Failed (Exit Code {result.returncode}): {nc_file}")
            error_output = result.stderr if result.stderr else result.stdout
            print(f"   Error Details: {error_output.strip()}")
            return {"file": nc_file, "status": "error", "time": processing_time, "error": error_output}

        # 2. The true measure of success: check if the 'summary.txt' was created.
        if check_for_output(nc_file):
            print(f"✅ Completed: {nc_file} ({processing_time:.1f}s)")
            return {"file": nc_file, "status": "success", "time": processing_time, "output": result.stdout}

        # 3. If the script exited cleanly (code 0) but produced no summary, it's a silent failure.
        print(f"❌ Failed (Silent): {nc_file} - Script finished but produced no summary.txt.")
        error_output = result.stdout if result.stdout.strip() else "No output from worker script."
        return {"file": nc_file, "status": "silent_failure", "time": processing_time, "error": error_output}

    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout: {nc_file} (exceeded 20 minutes)")
        return {"file": nc_file, "status": "timeout", "time": 1200, "error": "Process timed out after 20 minutes."}

    except Exception as e:
        processing_time = time.time() - start_time
        print(f"💥 Exception: {nc_file} - {str(e)}")
        return {"file": nc_file, "status": "exception", "time": processing_time, "error": str(e)}

def save_resume_state(processed_files, resume_file="batch_phase_resume.txt"):
    """Save the list of processed files to enable resuming."""
    with open(resume_file, 'w') as f:
        for file in processed_files:
            f.write(f"{file}\n")

def load_resume_state(resume_file="batch_phase_resume.txt"):
    """Load the set of already processed files."""
    if not os.path.exists(resume_file):
        return set()
    with open(resume_file, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def write_summary_report(results, output_file="batch_phase_summary.txt"):
    """Write a detailed summary of the batch processing run."""
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    total_time = sum(r['time'] for r in results)
    avg_time = total_time / len(results) if results else 0

    with open(output_file, 'w') as f:
        f.write("Batch Phase Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total files processed: {len(results)}\n")
        f.write(f"  - Successful: {len(successful)}\n")
        f.write(f"  - Failed:     {len(failed)}\n\n")
        f.write(f"Total processing time: {total_time:.1f}s ({total_time/60:.1f} minutes)\n")
        f.write(f"Average time per file: {avg_time:.1f}s\n\n")

        if successful:
            f.write("Successful Files:\n")
            f.write("-" * 20 + "\n")
            for r in successful:
                f.write(f"  [SUCCESS] {r['file']} ({r['time']:.1f}s)\n")
            f.write("\n")

        if failed:
            f.write("Failed Files & Detailed Errors:\n")
            f.write("-" * 20 + "\n")
            for r in failed:
                error_msg = r.get('error', 'Unknown error').strip().replace('\n', ' | ')
                f.write(f"  - FILE:    {r['file']}\n")
                f.write(f"    STATUS:  {r['status'].upper()}\n")
                f.write(f"    DETAILS: {error_msg}\n\n")

def main():
    parser = argparse.ArgumentParser(description="Batch process NetCDF files with phase_analysis.py")
    parser.add_argument("--pattern", default="*.nc", help="File pattern to match (default: *.nc)")
    parser.add_argument("--directory", default=".", help="Directory to search for files (default: current)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files with existing output directories")
    parser.add_argument("--parallel", action="store_true", help="Process multiple files in parallel")
    parser.add_argument("--max-workers", type=int, default=None, help="Max parallel workers (default: all CPU cores)")
    parser.add_argument("--resume", action="store_true", help="Resume from previous batch run")
    parser.add_argument("--dry-run", action="store_true", help="Show files to be processed without running")
    # Arguments to pass through to the worker script
    parser.add_argument('--axis', type=str, choices=['x', 'y'], default='x',
                        help="The analysis axis to pass to phase_analysis.py. Defaults to 'x'.")
    parser.add_argument('--position', type=float, default=None,
                        help="The analysis position to pass to phase_analysis.py. Defaults to the domain center.")

    args = parser.parse_args()

    nc_files = find_nc_files(args.directory, args.pattern)

    if not nc_files:
        print(f"No NetCDF files found matching pattern '{args.pattern}' in '{args.directory}'")
        return

    print(f"Found {len(nc_files)} NetCDF files matching pattern '{args.pattern}'")

    processed_files = set()
    if args.resume:
        processed_files = load_resume_state()
        print(f"Resume mode: Loaded {len(processed_files)} previously processed files.")

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
        print("No new files to process after applying filters.")
        return

    print(f"\n📋 Files to process: {len(files_to_process)}")
    for i, file in enumerate(files_to_process, 1):
        print(f"  {i:2d}. {file}")
    
    print("\nAnalysis Parameters:")
    print(f"  - Axis: {args.axis.upper()}")
    print(f"  - Position: {'Domain Center (default)' if args.position is None else f'{args.position} m'}")


    if args.dry_run:
        print("\n🔍 Dry run complete. No files were processed.")
        return

    try:
        input("\nPress Enter to begin processing or Ctrl+C to cancel...")
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)

    start_time = time.time()
    results = []
    resume_file = "batch_phase_resume.txt"

    if args.parallel:
        max_workers = args.max_workers or cpu_count()
        print(f"\n🚀 Starting parallel processing with up to {max_workers} workers...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(process_single_file, nc_file, args.axis, args.position): nc_file
                              for nc_file in files_to_process}

            for i, future in enumerate(as_completed(future_to_file), 1):
                result = future.result()
                results.append(result)
                processed_files.add(result['file'])
                if i % 5 == 0 or i == len(files_to_process):
                    save_resume_state(processed_files, resume_file)
                print(f"Progress: {i}/{len(files_to_process)} files completed.")
    else:
        print(f"\n🔄 Starting sequential processing...")
        for i, nc_file in enumerate(files_to_process, 1):
            print(f"\nProgress: {i}/{len(files_to_process)}")
            result = process_single_file(nc_file, args.axis, args.position)
            results.append(result)
            processed_files.add(nc_file)
            if i % 5 == 0 or i == len(files_to_process):
                save_resume_state(processed_files, resume_file)

    total_time = time.time() - start_time
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']

    print(f"\n" + "="*60)
    print(f"BATCH PROCESSING COMPLETE")
    print(f"="*60)
    print(f"📊 Total files processed: {len(results)}")
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    if results:
        print(f"⚡ Avg time/file: {total_time/len(results):.1f}s")

    if failed:
        print(f"\n❌ Failed files (see report for details):")
        for r in failed:
            print(f"  - {r['file']} (Reason: {r['status']})")

    write_summary_report(results)
    print(f"\n📝 Detailed report written to: batch_phase_summary.txt")

    if not failed and os.path.exists(resume_file):
        os.remove(resume_file)
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

