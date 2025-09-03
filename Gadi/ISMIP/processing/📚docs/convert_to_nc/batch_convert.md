# batch_convert.py

Batch converter for ISSM .outbin files to NetCDF format.

## Overview

This script automatically finds all `.outbin` files in the current directory and converts them to NetCDF format using the `convert_to_nc.py` functionality. It supports both sequential and parallel processing modes.

## Usage

```bash
python batch_convert.py [--layout grouped|flat] [--skip-existing] [--parallel] [--max-workers N]
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--layout` | choice | `grouped` | NetCDF layout format (`grouped` or `flat`) |
| `--skip-existing` | flag | `false` | Skip files where `.nc` already exists |
| `--parallel` | flag | `false` | Use parallel processing for faster conversion |
| `--max-workers` | int | CPU count | Maximum number of parallel workers |

### Examples

```bash
# Convert all .outbin files with default settings
python batch_convert.py

# Convert with flat layout and skip existing files
python batch_convert.py --layout flat --skip-existing

# Use parallel processing with 4 workers
python batch_convert.py --parallel --max-workers 4

# Full options example
python batch_convert.py --layout grouped --skip-existing --parallel
```

## Functionality

### Core Features

1. **Automatic File Discovery**: Finds all `.outbin` files in the current directory
2. **Batch Processing**: Converts multiple files in sequence or parallel
3. **Flexible Layouts**: Supports both grouped and flat NetCDF layouts
4. **Skip Existing**: Option to skip files that have already been converted
5. **Parallel Processing**: Multi-threaded conversion for improved performance
6. **Comprehensive Reporting**: Detailed summary of conversion results

### Processing Modes

#### Sequential Processing (Default)
- Processes files one at a time
- Shows progress for each file
- Safer for large files or limited memory

#### Parallel Processing (`--parallel`)
- Uses ThreadPoolExecutor for concurrent conversion
- Default worker count equals number of CPU cores
- Significantly faster for multiple small-to-medium files

### Output

The script provides detailed feedback including:
- List of files found for conversion
- Progress updates during processing
- Comprehensive summary with:
  - Total files processed
  - Successfully converted files
  - Skipped files (if using `--skip-existing`)
  - Error details for failed conversions

### Dependencies

- Requires `convert_to_nc.py` module in the same directory
- Uses standard Python libraries: `os`, `argparse`, `glob`, `concurrent.futures`

### Error Handling

- Gracefully handles conversion errors
- Continues processing remaining files if individual conversions fail
- Reports all errors in the final summary
- Returns detailed error information for debugging

## Function Reference

### `find_outbin_files(directory=".")`
Finds all .outbin files in the specified directory.

### `convert_single_file(input_file, layout="grouped", skip_existing=False)`
Converts a single .outbin file to .nc format with error handling.

### `batch_convert(layout="grouped", skip_existing=False, parallel=False, max_workers=None)`
Main function that orchestrates the batch conversion process.