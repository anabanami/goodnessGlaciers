# batch_extract_results.py Documentation

## Overview

`batch_extract_results.py` is a robust batch processing script designed to automatically discover and process NetCDF files using the `extract_results.py` script. It provides enhanced failure detection, parallel processing capabilities, and comprehensive progress tracking for large-scale data processing workflows.

## Key Features

- **Automatic File Discovery**: Finds and processes all NetCDF files matching specified patterns
- **Robust Failure Detection**: Detects "silent failures" where scripts exit successfully but produce no output
- **Parallel Processing**: Supports multi-core processing for faster batch operations
- **Resume Functionality**: Can resume interrupted batch runs from where they left off
- **Smart Output Checking**: Verifies actual plot generation rather than just exit codes
- **Comprehensive Reporting**: Generates detailed summary reports of processing results

## Command Line Usage

### Basic Usage
```bash
python batch_extract_results.py                    # Process all .nc files in current directory
```

### Pattern Filtering
```bash
python batch_extract_results.py --pattern="*_S4_*" # Process only files matching S4 experiment pattern
python batch_extract_results.py --pattern="exp_*.nc" --directory="/path/to/data"
```

### Processing Options
```bash
python batch_extract_results.py --parallel         # Enable parallel processing
python batch_extract_results.py --max-workers=4    # Limit to 4 parallel workers
python batch_extract_results.py --skip-existing    # Skip files with existing output
python batch_extract_results.py --resume           # Resume from last processed file
python batch_extract_results.py --dry-run          # Preview what would be processed
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--pattern` | `*.nc` | File pattern to match (supports glob patterns) |
| `--directory` | `.` | Directory to search for NetCDF files |
| `--skip-existing` | `False` | Skip files that already have output directories with PNG files |
| `--parallel` | `False` | Enable parallel processing using multiple CPU cores |
| `--max-workers` | CPU count | Maximum number of parallel workers |
| `--resume` | `False` | Resume from previously interrupted batch run |
| `--dry-run` | `False` | Show what would be processed without actually processing |

## Processing Logic

### Success Detection
The script uses a multi-layered approach to determine processing success:

1. **Exit Code Check**: Verifies the subprocess exits with code 0
2. **Output Verification**: Confirms that PNG files are actually generated in the output directory
3. **Valid Skip Detection**: Recognizes when the worker script legitimately skips processing (e.g., files with only one time step)
4. **Silent Failure Detection**: Identifies cases where the script exits successfully but produces no output

### Status Categories

- **Success**: Script completed and generated visualization plots
- **Skipped**: Worker script legitimately skipped the file (e.g., insufficient time steps)
- **Error**: Script failed with non-zero exit code
- **Silent Failure**: Script completed successfully but produced no plots
- **Timeout**: Processing exceeded 15-minute time limit
- **Exception**: Unexpected error during processing

## Output and Reporting

### Console Output
Real-time progress updates with emoji indicators:
- 🔄 Processing file
- ✅ Successful completion
- ❌ Processing failure
- ⏰ Timeout
- 💥 Exception

### Summary Report
Generates `batch_processing_summary.txt` containing:
- Overall statistics (success/failure counts)
- Processing times and averages
- Detailed error information for failed files
- Complete list of processed files with their status

### Resume State
Maintains `batch_resume.txt` to track processed files for resume functionality.

## Functions

### Core Functions

#### `find_nc_files(directory=".", pattern="*.nc")`
Discovers NetCDF files matching the specified pattern in the given directory.

#### `process_single_file(nc_file)`
Processes a single NetCDF file with comprehensive error handling and status detection.

#### `check_for_output(nc_file)`
Verifies that processing actually generated visualization plots by checking for PNG files in the output directory.

### Utility Functions

#### `get_output_directory(nc_file)`
Determines the expected output directory name based on the NetCDF filename.

#### `save_resume_state(processed_files, resume_file="batch_resume.txt")`
Saves the list of processed files for resume functionality.

#### `load_resume_state(resume_file="batch_resume.txt")`
Loads previously processed files when resuming a batch run.

#### `write_summary_report(results, output_file="batch_processing_summary.txt")`
Generates a comprehensive summary report of the batch processing session.

## Error Handling

The script includes robust error handling for:
- Subprocess failures and non-zero exit codes
- Processing timeouts (15-minute limit per file)
- Silent failures (successful exit but no output)
- File system errors and exceptions
- User interruption (Ctrl+C)

## Dependencies

- Python 3.x standard library modules:
  - `os`, `glob`, `subprocess`, `sys`, `argparse`, `time`
  - `pathlib.Path`
  - `concurrent.futures.ProcessPoolExecutor`
  - `multiprocessing.cpu_count`

## Integration Requirements

- Requires `extract_results.py` to be present in the same directory
- NetCDF files to process must be accessible in the specified directory
- Sufficient disk space for output visualization files

## Performance Considerations

- Default timeout of 15 minutes per file
- Parallel processing uses all available CPU cores by default
- Resume state is saved every 5 processed files
- Memory usage scales with the number of parallel workers