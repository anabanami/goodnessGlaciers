#!/usr/bin/env python3
"""
Batch converter for ISSM .outbin files to NetCDF format.

This script automatically finds all .outbin files in the current directory
and converts them using the convert_to_nc.py functionality.

Usage:
    python batch_convert.py [--layout grouped|flat] [--skip-existing] [--parallel]
"""

import os
import argparse
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from convert_to_nc import create_netcdf_from_outbin


def find_outbin_files(directory="."):
    """Find all .outbin files in the directory"""
    pattern = os.path.join(directory, "*.outbin")
    return glob.glob(pattern)


def convert_single_file(input_file, layout="grouped", skip_existing=False):
    """Convert a single .outbin file to .nc"""
    output_file = os.path.splitext(input_file)[0] + ".nc"
    
    if skip_existing and os.path.exists(output_file):
        print(f"Skipping {input_file} - {output_file} already exists")
        return {"status": "skipped", "input": input_file, "output": output_file}
    
    try:
        create_netcdf_from_outbin(input_file, output_file, layout=layout)
        return {"status": "success", "input": input_file, "output": output_file}
    except Exception as e:
        print(f"Error converting {input_file}: {e}")
        return {"status": "error", "input": input_file, "error": str(e)}


def batch_convert(layout="grouped", skip_existing=False, parallel=False, max_workers=None):
    """Convert all .outbin files in the current directory"""
    outbin_files = find_outbin_files()
    
    if not outbin_files:
        print("No .outbin files found in current directory")
        return
    
    print(f"Found {len(outbin_files)} .outbin files to convert:")
    for f in outbin_files:
        print(f"  - {f}")
    
    print(f"\nStarting conversion with layout='{layout}'...")
    print(f"Skip existing: {skip_existing}")
    print(f"Parallel processing: {parallel}")
    
    results = []
    
    if parallel:
        # Use ThreadPoolExecutor for parallel processing
        if max_workers is None:
            max_workers = min(len(outbin_files), os.cpu_count())
        
        print(f"Using {max_workers} workers for parallel processing")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_file = {
                executor.submit(convert_single_file, f, layout, skip_existing): f 
                for f in outbin_files
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                result = future.result()
                results.append(result)
    else:
        # Sequential processing
        for i, input_file in enumerate(outbin_files, 1):
            print(f"\nProcessing {i}/{len(outbin_files)}: {input_file}")
            result = convert_single_file(input_file, layout, skip_existing)
            results.append(result)
    
    # Print summary
    print(f"\n{'='*60}")
    print("CONVERSION SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in results if r["status"] == "success"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors = [r for r in results if r["status"] == "error"]
    
    print(f"Total files: {len(outbin_files)}")
    print(f"Successfully converted: {len(successful)}")
    print(f"Skipped (already exist): {len(skipped)}")
    print(f"Errors: {len(errors)}")
    
    if successful:
        print(f"\nSuccessfully converted:")
        for r in successful:
            print(f"  ✓ {r['input']} → {r['output']}")
    
    if skipped:
        print(f"\nSkipped files:")
        for r in skipped:
            print(f"  - {r['input']}")
    
    if errors:
        print(f"\nErrors:")
        for r in errors:
            print(f"  ✗ {r['input']}: {r['error']}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert ISSM .outbin files to NetCDF format"
    )
    parser.add_argument(
        '--layout', 
        choices=['grouped', 'flat'], 
        default='grouped',
        help='NetCDF layout (default: grouped)'
    )
    parser.add_argument(
        '--skip-existing', 
        action='store_true',
        help='Skip files where .nc already exists'
    )
    parser.add_argument(
        '--parallel', 
        action='store_true',
        help='Use parallel processing for faster conversion'
    )
    parser.add_argument(
        '--max-workers', 
        type=int,
        help='Maximum number of parallel workers (default: number of CPUs)'
    )
    
    args = parser.parse_args()
    
    batch_convert(
        layout=args.layout,
        skip_existing=args.skip_existing,
        parallel=args.parallel,
        max_workers=args.max_workers
    )


if __name__ == "__main__":
    main()