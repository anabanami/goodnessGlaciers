# batch_convert.py

Batch converter for ISSM `.outbin` files to NetCDF format.

## Overview

This script automatically finds all `.outbin` files in the current directory and converts them to NetCDF format using the `convert_to_nc.py` functionality. It supports both sequential and parallel processing modes.

## Usage

```bash
python batch_convert.py [--layout grouped|flat] [--skip-existing] [--parallel] [--max-workers N]
```

### Command Line Options

- `--layout {grouped|flat}` - NetCDF layout format (default: grouped)
- `--skip-existing` - Skip files where corresponding .nc file already exists
- `--parallel` - Enable parallel processing for faster conversion
- `--max-workers N` - Maximum number of parallel workers (default: number of CPUs)

### Examples

```bash
# Convert all .outbin files with default settings
python batch_convert.py

# Convert with flat layout and skip existing files
python batch_convert.py --layout flat --skip-existing

# Use parallel processing with 4 workers
python batch_convert.py --parallel --max-workers 4
```

## Functionality

### Core Functions

- **`find_outbin_files(directory=".")`** - Discovers all `.outbin` files in the specified directory
- **`convert_single_file(input_file, layout="grouped", skip_existing=False)`** - Converts a single `.outbin` file to `.nc`
- **`batch_convert(layout="grouped", skip_existing=False, parallel=False, max_workers=None)`** - Main conversion function

### Features

- **Automatic file discovery** - Finds all `.outbin` files in current directory
- **Skip existing files** - Optional flag to avoid re-converting files
- **Parallel processing** - Multi-threaded conversion for improved performance
- **Progress tracking** - Shows conversion progress and detailed summary
- **Error handling** - Captures and reports conversion errors per file

### Output

The script provides:
- List of files to be converted
- Real-time conversion progress
- Comprehensive summary with success/skip/error counts
- Detailed results for each file

### Dependencies

Requires `convert_to_nc.py` module with `create_netcdf_from_outbin` function.