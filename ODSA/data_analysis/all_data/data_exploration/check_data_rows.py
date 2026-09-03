import numpy as np
import pandas as pd
import sys
import os


"""
# Single file                                                                             
python check_data_rows.py                                                                 
JR_data+REMA/bedmap3_data/bedmap3/Results/PRIC_2016_CHA2_AIR_BM3.csv                      
                                                                                        
# Multiple specific files                                                                 
python check_data_rows.py                                                                 
JR_data+REMA/bedmap3_data/bedmap3/Results/PRIC_2016_CHA2_AIR_BM3.csv                      
JR_data+REMA/bedmap3_data/bedmap3/Results/BAS_2015_POLARGAP_AIR_BM3.csv                   
                                                                                        
# All CSVs in a directory at once                                                         
python check_data_rows.py JR_data+REMA/bedmap3_data/bedmap3/Results/                      
                                                                                        
All three invocation styles work — individual files, multiple files, or a whole directory.

"""



COLUMNS_TO_CHECK = [
    'trajectory_id',
    'bedrock_altitude (m)',
    'longitude (degree_east)',
    'latitude (degree_north)',
]

INVALID_VALUE = -9999


def check_file(filepath):
    """Validate key columns in a single Bedmap CSV file."""
    print(f"\n{'=' * 80}")
    print(f"Checking: {filepath}")
    print(f"{'=' * 80}")

    if not os.path.isfile(filepath):
        print(f"  ERROR: file not found")
        return

    df = pd.read_csv(filepath, comment='#', header=0)

    print(f"  Total rows: {len(df)}")

    # Check each required column exists and has valid values
    for col in COLUMNS_TO_CHECK:
        if col not in df.columns:
            print(f"  WARNING: column '{col}' not found in file")
            continue

        total = len(df)
        null_count = df[col].isna().sum()
        invalid_count = (df[col] == INVALID_VALUE).sum()
        valid_count = total - null_count - invalid_count

        print(f"\n  [{col}]")
        print(f"    Valid:   {valid_count}/{total}")
        if null_count > 0:
            print(f"    NaN:     {null_count}")
        if invalid_count > 0:
            print(f"    Invalid ({INVALID_VALUE}): {invalid_count}")

    # Trajectory ID summary
    if 'trajectory_id' in df.columns:
        num_rows = len(df)
        num_unique = df['trajectory_id'].nunique()
        print(f"\n  Trajectory IDs: {num_unique} unique out of {num_rows} rows")
        if num_rows == num_unique:
            print(f"    Every trajectory_id is unique")
        else:
            avg_points = num_rows / num_unique
            print(f"    Data is grouped. Average of {avg_points:.2f} points per trajectory")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_data_rows.py <file1.csv> [file2.csv] ...")
        print("       python check_data_rows.py <directory>")
        sys.exit(1)

    paths = sys.argv[1:]

    # If a single directory is provided, check all CSVs in it
    if len(paths) == 1 and os.path.isdir(paths[0]):
        directory = paths[0]
        paths = sorted(
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith('.csv')
        )
        if not paths:
            print(f"No CSV files found in {directory}")
            sys.exit(1)
        print(f"Found {len(paths)} CSV files in {directory}")

    for filepath in paths:
        check_file(filepath)

    print(f"\n{'=' * 80}")
    print(f"Done. Checked {len(paths)} file(s).")
