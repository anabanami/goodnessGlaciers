#!/usr/bin/env python3
"""
A simple batch converter for ISSM .outbin files.

This script finds all files with the '.outbin' extension in the current
directory and calls the 'convert_to_nc_issm.py' script for each one sequentially.
"""

import glob
import subprocess
import sys

def run_batch_conversion():
    """
    Finds and converts all .outbin files in the current directory.
    """
    # 1. Find all .outbin files in the current directory.
    outbin_files = glob.glob('*.outbin')

    if not outbin_files:
        print("No .outbin files found in the current directory.")
        return

    print(f"Found {len(outbin_files)} files to convert:")
    for f in outbin_files:
        print(f"  - {f}")
    print("-" * 30)

    # 2. Loop through the files and call the conversion script for each.
    for i, input_file in enumerate(outbin_files, 1):
        print(f"--> Processing file {i}/{len(outbin_files)}: {input_file}")
        
        command = ["python", "convert_to_nc_issm.py", input_file]
        
        try:
            # Run the external script, capturing its output.
            # check=True will raise an error if the script fails.
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                check=True
            )
            # Print the output from the successful conversion
            print(result.stdout)

        except subprocess.CalledProcessError as e:
            # If the script returns an error, print the error message
            print(f"--- ERROR converting {input_file} ---")
            print(e.stderr)
        except FileNotFoundError:
            # This error occurs if 'convert_to_nc_issm.py' isn't found
            print("FATAL ERROR: Could not find 'convert_to_nc_issm.py'.")
            print("Please ensure it is in the same directory as this script.")
            sys.exit(1) # Exit the script immediately

    print("-" * 30)
    print("Batch conversion complete.")


if __name__ == "__main__":
    run_batch_conversion()