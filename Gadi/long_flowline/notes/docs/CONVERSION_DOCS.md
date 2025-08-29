# ISSM .outbin to NetCDF Conversion Tools

This directory contains tools for converting ISSM (Ice Sheet System Model) `.outbin` binary files to NetCDF format.

## Files Overview

- **`convert_to_nc.py`** - Single file converter with detailed analysis capabilities
- **`batch_convert.py`** - Batch processor for multiple files with parallel processing support
- **`CONVERSION_DOCS.md`** - This documentation file

## convert_to_nc.py

### Purpose
Converts individual ISSM `.outbin` files to NetCDF format with support for both flat and grouped (ISSM-style) layouts.

### Usage

#### Basic Conversion
```bash
python convert_to_nc.py input.outbin
```
This creates `input.nc` with default grouped layout.

#### Specify Output File
```bash
python convert_to_nc.py input.outbin output.nc
```

#### Choose Layout
```bash
# Grouped layout (default) - creates results/TransientSolution groups
python convert_to_nc.py input.outbin --layout grouped

# Flat layout - all variables at root level
python convert_to_nc.py input.outbin --layout flat
```

#### Analysis Only (no conversion)
```bash
python convert_to_nc.py input.outbin --analyse-only
```
Reads and displays file contents without creating NetCDF output.

### Features
- **Data Type Support**: Handles scalars, strings, arrays (double/int), and complex arrays
- **Metadata Preservation**: Maintains time steps, variable names, and shapes
- **Two Layout Options**:
  - `grouped`: Variables under `results/TransientSolution` groups (ISSM-compatible)
  - `flat`: All variables at root level
- **Robust Error Handling**: Detailed error messages for corrupted files
- **Progress Tracking**: Shows each variable as it's processed

### Output Format
The NetCDF files include:
- **Global Attributes**: Creation date, source info, original filename
- **Time Dimension**: All time steps from the simulation
- **Variables**: All data from the .outbin file with proper dimensions
- **Metadata**: Original ISSM data types and shapes preserved

## batch_convert.py

### Purpose
Automatically finds and converts all `.outbin` files in the current directory using the same conversion logic as `convert_to_nc.py`.

### Usage

#### Basic Batch Conversion
```bash
python batch_convert.py
```
Converts all `.outbin` files in current directory with grouped layout.

#### Skip Existing Files
```bash
python batch_convert.py --skip-existing
```
Only converts files where `.nc` doesn't already exist.

#### Parallel Processing
```bash
python batch_convert.py --parallel
```
Uses multiple CPU cores for faster processing of many files.

#### Combined Options
```bash
python batch_convert.py --parallel --skip-existing --layout flat
```

#### Control Parallel Workers
```bash
python batch_convert.py --parallel --max-workers 4
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--layout {grouped,flat}` | NetCDF structure layout | `grouped` |
| `--skip-existing` | Skip files where .nc already exists | `False` |
| `--parallel` | Use parallel processing | `False` |
| `--max-workers N` | Max parallel workers | CPU count |

### Features
- **Automatic Discovery**: Finds all `.outbin` files in current directory
- **Progress Tracking**: Shows current file being processed
- **Parallel Processing**: Optional multi-threading for speed
- **Error Handling**: Continues processing other files if one fails
- **Comprehensive Summary**: Shows success/skip/error counts
- **Identical Output**: Uses same conversion function as single converter

### Example Output
```
Found 45 .outbin files to convert:
  - 002_S1_0.5.outbin
  - 002_S2_0.5.outbin
  - 003_S1_0.5.outbin
  ...

Starting conversion with layout='grouped'...
Skip existing: True
Parallel processing: True
Using 8 workers for parallel processing

============================================================
CONVERSION SUMMARY
============================================================
Total files: 45
Successfully converted: 43
Skipped (already exist): 2
Errors: 0

Successfully converted:
  ✓ 002_S1_0.5.outbin → 002_S1_0.5.nc
  ✓ 002_S2_0.5.outbin → 002_S2_0.5.nc
  ...
```

## Data Types Supported

Both tools handle these ISSM data types:

| Type | Description | NetCDF Storage |
|------|-------------|----------------|
| 1 | Double scalar | `f8` variable |
| 2 | String | `S1` character array |
| 3 | Double array | `f8` with dimensions |
| 4 | Integer array | `i4` with dimensions |
| 5 | Complex array | `c16` with dimensions |

## File Structure

### Input (.outbin format)
Binary files with:
- Variable headers (name, time, step)
- Data type indicators
- Array dimensions
- Raw data values

### Output (.nc format)
NetCDF4 files with:
- Global metadata
- Time dimension
- Variable dimensions (auto-created)
- All original data with proper types

## Performance Tips

1. **Use `--parallel`** for many files (>10)
2. **Use `--skip-existing`** for incremental processing
3. **Use `--analyse-only`** to check file integrity before conversion
4. **Monitor disk space** - NetCDF files can be larger than binary

## Troubleshooting

### Common Issues

**"Invalid name length" error**
- File may be corrupted or not a valid .outbin file
- Try `--analyse-only` first to check file structure

**"Could not read time/step" error**
- Incomplete file or unexpected format
- Check if file was fully written by ISSM

**Memory issues with large files**
- Process files individually rather than in parallel
- Use `--max-workers 1` to limit memory usage

### Verification
Compare original and converted data:
```python
# Check conversion worked correctly
import netCDF4 as nc
with nc.Dataset('output.nc', 'r') as f:
    print(f.variables.keys())
    print(f['results/TransientSolution/Vel'][:])
```

## Dependencies

Both scripts require:
- `numpy`
- `netCDF4` 
- `struct` (built-in)
- `os` (built-in)
- `argparse` (built-in)
- `datetime` (built-in)
- `glob` (built-in)
- `concurrent.futures` (built-in)

Install with: `pip install numpy netCDF4`