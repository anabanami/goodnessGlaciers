#!/usr/bin/env python3
"""
Batch converter for ISSM .outbin files to NetCDF format using ISSM tools.

This script automatically finds all .outbin files matching the expected
ISSM naming pattern in the current directory and converts them using the
`convert_outbin_to_nc` function from the `convert_to_nc_issm.py` script.

It supports parallel processing for faster conversion on multi-core systems.

Usage:
    python batch_convert_issm.py [--skip-existing] [--parallel] [--from-cluster]
"""

import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the necessary functions from your provided script
# This assumes 'convert_to_nc_issm.py' is in the same directory or in the PYTHONPATH
try:
    from convert_to_nc_issm import convert_outbin_to_nc, find_matching_outbin_files
except ImportError:
    print("Error: Could not import from 'convert_to_nc_issm.py'.")
    print("Please ensure the script is in the same directory or your PYTHONPATH.")
    exit(1)


def convert_single_file(input_file, use_cluster=False, skip_existing=False):
    """
    Worker function to convert a single .outbin file to .nc.

    Args:
        input_file (str): The path to the input .outbin file.
        use_cluster (bool): Flag to determine if loadresultsfromcluster should be used.
        skip_existing (bool): If True, skip conversion if the .nc file already exists.

    Returns:
        dict: A dictionary containing the status of the conversion.
    """
    output_file = os.path.splitext(input_file)[0] + ".nc"
    
    if skip_existing and os.path.exists(output_file):
        print(f"Skipping {input_file} -> {output_file} already exists.")
        return {"status": "skipped", "input": input_file, "output": output_file}
    
    print(f"Processing: {input_file}")
    try:
        # Call the conversion function from the ISSM-specific script
        success = convert_outbin_to_nc(input_file, output_file, use_cluster=use_cluster)
        
        if success:
            return {"status": "success", "input": input_file, "output": output_file}
        else:
            # The called function should print its own detailed errors
            return {"status": "error", "input": input_file, "error": "Conversion failed. See logs above."}
            
    except Exception as e:
        print(f"An unexpected error occurred while converting {input_file}: {e}")
        return {"status": "error", "input": input_file, "error": str(e)}


def batch_convert(from_cluster=False, skip_existing=False, parallel=False, max_workers=None):
    """
    Finds and converts all matching .outbin files in the current directory.

    Args:
        from_cluster (bool): Passed to the conversion function.
        skip_existing (bool): Skip files if their .nc output already exists.
        parallel (bool): Run conversions in parallel using a ThreadPoolExecutor.
        max_workers (int, optional): The maximum number of threads to use in parallel mode.
    """
    outbin_files = find_matching_outbin_files()
    
    if not outbin_files:
        print("No matching .outbin files found in current directory.")
        print("Expected pattern: IsmipF_S[1-4]_*_*-Transient.outbin")
        return
    
    print(f"Found {len(outbin_files)} matching .outbin files to convert:")
    for f in outbin_files:
        print(f"  - {f}")
    
    print(f"\nStarting conversion...")
    print(f"Skip existing: {skip_existing}")
    print(f"Parallel processing: {parallel}")
    print(f"Load from cluster: {from_cluster}")
    
    results = []
    
    if parallel:
        if max_workers is None:
            # Default to the number of CPUs, but don't exceed the number of files
            max_workers = min(len(outbin_files), os.cpu_count() or 1)
        
        print(f"Using {max_workers} workers for parallel processing.")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(convert_single_file, f, from_cluster, skip_existing): f 
                for f in outbin_files
            }
            
            for future in as_completed(future_to_file):
                results.append(future.result())
    else:
        # Sequential processing
        for i, input_file in enumerate(outbin_files, 1):
            print(f"\n--- Processing {i}/{len(outbin_files)} ---")
            result = convert_single_file(input_file, from_cluster, skip_existing)
            results.append(result)
    
    # --- Print Summary ---
    print(f"\n{'='*60}")
    print("CONVERSION SUMMARY")
    print(f"{'='*60}")
    
    successful = sorted([r for r in results if r["status"] == "success"], key=lambda x: x['input'])
    skipped = sorted([r for r in results if r["status"] == "skipped"], key=lambda x: x['input'])
    errors = sorted([r for r in results if r["status"] == "error"], key=lambda x: x['input'])
    
    print(f"Total files found: {len(outbin_files)}")
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
        print(f"\nFiles with errors:")
        for r in errors:
            print(f"  ✗ {r['input']}: {r['error']}")


def main():
    """Parses command-line arguments and starts the batch conversion."""
    parser = argparse.ArgumentParser(
        description="Batch convert ISSM .outbin files to NetCDF using the ISSM python tools.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--skip-existing', 
        action='store_true',
        help='Skip files where the output .nc file already exists.'
    )
    parser.add_argument(
        '--parallel', 
        action='store_true',
        help='Use parallel processing for faster conversion on multi-core systems.'
    )
    parser.add_argument(
        '--max-workers', 
        type=int,
        default=None,
        help='Maximum number of parallel workers (default: number of CPU cores).'
    )
    parser.add_argument(
        '--from-cluster',
        action='store_true',
        help="Use 'loadresultsfromcluster' instead of 'loadresultsfromdisk'."
    )
    
    args = parser.parse_args()
    
    batch_convert(
        from_cluster=args.from_cluster,
        skip_existing=args.skip_existing,
        parallel=args.parallel,
        max_workers=args.max_workers
    )


if __name__ == "__main__":
    main()