#!/usr/bin/env python3
"""
Batch processing script for extract_results.py

Automatically discovers and processes all NetCDF files in the current directory,
generating visualizations for each simulation result.

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

def is_already_processed(nc_file):
    """Check if the file has already been processed (output directory exists with content)."""
    output_dir = get_output_directory(nc_file)
    if not os.path.exists(output_dir):
        return False
    
    # Check if directory has some PNG files (indicating successful processing)
    png_files = glob.glob(os.path.join(output_dir, "*/*.png"))
    return len(png_files) > 0

def process_single_file(nc_file):
    """Process a single NetCDF file using extract_results.py."""
    print(f"\n🔄 Processing: {nc_file}")
    start_time = time.time()
    
    try:
        # Run extract_results.py as subprocess
        result = subprocess.run(
            [sys.executable, "extract_results.py", nc_file],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per file
        )
        
        processing_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Completed: {nc_file} ({processing_time:.1f}s)")
            return {"file": nc_file, "status": "success", "time": processing_time, "output": result.stdout}
        else:
            print(f"❌ Failed: {nc_file}")
            print(f"Error: {result.stderr}")
            return {"file": nc_file, "status": "error", "time": processing_time, "error": result.stderr}
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout: {nc_file} (exceeded 10 minutes)")
        return {"file": nc_file, "status": "timeout", "time": 600, "error": "Process timeout"}
        
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
    """Write a summary report of the batch processing results."""
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    
    total_time = sum(r['time'] for r in results)
    avg_time = total_time / len(results) if results else 0
    
    with open(output_file, 'w') as f:
        f.write("Batch Processing Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total files processed: {len(results)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n")
        f.write(f"Total processing time: {total_time:.1f}s ({total_time/60:.1f} minutes)\n")
        f.write(f"Average time per file: {avg_time:.1f}s\n\n")
        
        if successful:
            f.write("Successful Files:\n")
            f.write("-" * 20 + "\n")
            for r in successful:
                f.write(f"  {r['file']} ({r['time']:.1f}s)\n")
            f.write("\n")
        
        if failed:
            f.write("Failed Files:\n")
            f.write("-" * 20 + "\n")
            for r in failed:
                f.write(f"  {r['file']} ({r['status']}): {r.get('error', 'Unknown error')}\n")

def main():
    parser = argparse.ArgumentParser(description="Batch process NetCDF files with extract_results.py")
    parser.add_argument("--pattern", default="*.nc", help="File pattern to match (default: *.nc)")
    parser.add_argument("--directory", default=".", help="Directory to search for files (default: current)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files with existing output directories")
    parser.add_argument("--parallel", action="store_true", help="Process multiple files in parallel")
    parser.add_argument("--max-workers", type=int, default=None, help="Max parallel workers (default: CPU count)")
    parser.add_argument("--resume", action="store_true", help="Resume from previous batch run")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without actually processing")
    
    args = parser.parse_args()
    
    # Find all NetCDF files
    nc_files = find_nc_files(args.directory, args.pattern)
    
    if not nc_files:
        print(f"No NetCDF files found matching pattern '{args.pattern}' in '{args.directory}'")
        return
    
    print(f"Found {len(nc_files)} NetCDF files matching pattern '{args.pattern}'")
    
    # Load resume state if requested
    processed_files = set()
    if args.resume:
        processed_files = load_resume_state()
        print(f"Resume mode: {len(processed_files)} files already processed")
    
    # Filter files based on options
    files_to_process = []
    for nc_file in nc_files:
        if args.resume and nc_file in processed_files:
            print(f"🔄 Skipping (resume): {nc_file}")
            continue
            
        if args.skip_existing and is_already_processed(nc_file):
            print(f"📁 Skipping (exists): {nc_file}")
            continue
            
        files_to_process.append(nc_file)
    
    if not files_to_process:
        print("No files to process after filtering")
        return
    
    print(f"\n📋 Files to process: {len(files_to_process)}")
    for i, file in enumerate(files_to_process, 1):
        print(f"  {i:2d}. {file}")
    
    if args.dry_run:
        print("\n🔍 Dry run complete - no files were processed")
        return
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    # Process files
    start_time = time.time()
    results = []
    
    if args.parallel:
        # Parallel processing
        max_workers = args.max_workers or min(cpu_count(), 4)  # Limit to 4 by default
        print(f"\n🚀 Starting parallel processing with {max_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_file = {executor.submit(process_single_file, nc_file): nc_file 
                             for nc_file in files_to_process}
            
            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_file), 1):
                result = future.result()
                results.append(result)
                processed_files.add(result['file'])
                
                # Save progress periodically
                if i % 5 == 0 or i == len(files_to_process):
                    save_resume_state(processed_files)
                
                print(f"Progress: {i}/{len(files_to_process)} files completed")
    
    else:
        # Sequential processing
        print(f"\n🔄 Starting sequential processing...")
        
        for i, nc_file in enumerate(files_to_process, 1):
            print(f"\nProgress: {i}/{len(files_to_process)}")
            result = process_single_file(nc_file)
            results.append(result)
            processed_files.add(nc_file)
            
            # Save progress periodically
            if i % 5 == 0 or i == len(files_to_process):
                save_resume_state(processed_files)
    
    # Final summary
    total_time = time.time() - start_time
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    
    print(f"\n" + "="*60)
    print(f"BATCH PROCESSING COMPLETE")
    print(f"="*60)
    print(f"📊 Total files: {len(results)}")
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"⚡ Avg time/file: {total_time/len(results):.1f}s")
    
    if failed:
        print(f"\n❌ Failed files:")
        for r in failed:
            print(f"  - {r['file']} ({r['status']})")
    
    # Write detailed summary report
    write_summary_report(results)
    print(f"\n📝 Detailed report written to: batch_processing_summary.txt")
    
    # Clean up resume file if everything completed successfully
    if not failed and os.path.exists("batch_resume.txt"):
        os.remove("batch_resume.txt")
        print("🧹 Cleaned up resume file")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⛔ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)