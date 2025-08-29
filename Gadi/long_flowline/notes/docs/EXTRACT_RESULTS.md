# Processing Scripts Documentation

This directory contains scripts for extracting and visualizing results from ISSM flowline simulation NetCDF files.

## Overview

The processing pipeline consists of two main scripts:
- **`extract_results.py`** - Core visualization script for individual NetCDF files
- **`batch_extract_results.py`** - Batch processing wrapper for multiple files

## extract_results.py

### Purpose
Extracts field data (velocity, pressure, etc.) from ISSM simulation NetCDF files and generates publication-ready visualizations with proper mesh reconstruction.

### Key Features
- **Smart time unit detection**: Automatically handles time values in seconds or years
- **Adaptive mesh reconstruction**: Rebuilds triangular mesh from bedrock profile configurations
- **Parallel plotting**: Uses multiprocessing for fast visualization generation
- **Unit conversion**: Automatically converts velocities from m/s to m/yr when appropriate
- **Multiple field support**: Handles Vx, Vel, Pressure, Surface, Base fields
- **Both static and transient data**: Works with single timestep or time series data

### Usage

#### Basic Usage
```bash
# Process a single NetCDF file
python extract_results.py simulation_results.nc

# Process multiple files with pattern matching
python extract_results.py --pattern='flowline_27_S4_*.nc'
```

#### Input Requirements
- NetCDF files from ISSM simulations
- Corresponding bedrock profile configuration (automatically detected from filename)
- Files should follow naming convention: `{profile_id}_{experiment}_{resolution_factor}_*.nc`

#### Output Structure
Creates a directory named after the input file (without .nc extension):
```
simulation_results/
├── Vx/
│   ├── Vx_step00.png
│   ├── Vx_step01.png
│   └── ...
├── Vel/
│   ├── Vel_step00.png
│   └── ...
└── Pressure/
    ├── Pressure_step00.png
    └── ...
```

### Technical Details

#### Time Unit Handling
The script intelligently detects time units:
1. **Smart range detection**: If time values are between 0.001-10000, assumes they're already in years
2. **Unit attribute checking**: Reads NetCDF time variable units
3. **Fallback logic**: Large values (>100k) assumed to be in seconds and converted

#### Mesh Reconstruction
- Extracts profile ID from filename
- Loads bedrock configuration from `../bedrock_profiles/`
- Uses adaptive bamg meshing with wavelength-based refinement
- Matches exact parameters from original flowline.py simulation

#### Field Configuration
Each field has customized visualization settings:
- **Vx**: Coolwarm colormap with zero-centered normalization
- **Vel**: Viridis colormap for velocity magnitude
- **Pressure**: Plasma colormap for pressure fields

## batch_extract_results.py

### Purpose
Batch processing wrapper that automatically discovers and processes multiple NetCDF files in parallel or sequence.

### Key Features
- **Automatic file discovery**: Finds all .nc files matching patterns
- **Parallel processing**: Process multiple files simultaneously
- **Resume functionality**: Continue from where previous run stopped
- **Skip existing**: Avoid reprocessing files with existing output
- **Progress tracking**: Real-time progress updates and timing
- **Error handling**: Robust timeout and exception handling
- **Summary reporting**: Generates detailed processing reports

### Usage

#### Basic Usage
```bash
# Process all .nc files in current directory
python batch_extract_results.py

# Process only specific experiment files
python batch_extract_results.py --pattern="*_S4_*.nc"

# Parallel processing with 4 workers
python batch_extract_results.py --parallel --max-workers=4

# Skip files that already have output directories
python batch_extract_results.py --skip-existing

# Resume interrupted batch run
python batch_extract_results.py --resume

# Dry run to see what would be processed
python batch_extract_results.py --dry-run
```

#### Advanced Options
```bash
# Process files in specific directory
python batch_extract_results.py --directory=/path/to/netcdf/files

# Combine multiple options
python batch_extract_results.py --pattern="*periodic*" --parallel --skip-existing
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--pattern` | File pattern to match | `*.nc` |
| `--directory` | Directory to search for files | `.` (current) |
| `--skip-existing` | Skip files with existing output directories | False |
| `--parallel` | Process multiple files in parallel | False |
| `--max-workers` | Maximum parallel workers | CPU count, max 4 |
| `--resume` | Resume from previous batch run | False |
| `--dry-run` | Show what would be processed without processing | False |

### Output Files

#### Progress Tracking
- **`batch_resume.txt`**: List of processed files for resume functionality
- **`batch_processing_summary.txt`**: Detailed processing report

#### Summary Report Contents
- Total files processed, successful, and failed
- Processing times (total and average per file)
- List of successful files with timing
- List of failed files with error messages

### Error Handling

The batch processor handles several error conditions:
- **Timeouts**: 10-minute limit per file
- **Process errors**: Captures stderr from extract_results.py
- **Exceptions**: Catches and logs Python exceptions
- **Interruption**: Graceful handling of Ctrl+C

## Performance Considerations

### Single File Processing
- **Mesh building**: 1-5 seconds depending on resolution
- **Data loading**: 2-10 seconds depending on file size and timesteps
- **Plotting**: 0.1-0.5 seconds per plot (parallelized)
- **Total time**: Typically 10-30 seconds per file

### Batch Processing
- **Sequential**: Processes one file at a time
- **Parallel**: Recommended for large batches, uses 4 workers by default
- **Memory usage**: Each worker loads full dataset into memory
- **I/O considerations**: SSD storage recommended for large datasets

## File Naming Conventions

### Input Files
Expected format: `{profile_id}_{experiment}_{resolution_factor}_*.nc`

Examples:
- `flowline_27_S4_1.0_periodic_transient.nc`
- `165_S3_1.25_non_periodic_static.nc`

### Output Directories
Named after input file without extension:
- `flowline_27_S4_1.0_periodic_transient/`
- `165_S3_1.25_non_periodic_static/`

## Troubleshooting

### Common Issues

#### Dimension Mismatch
**Problem**: "DIMENSION MISMATCH" errors
**Cause**: Mesh vertices don't match data points in NetCDF
**Solution**: Check bedrock profile configuration and ensure correct profile_id

#### Time Unit Issues
**Problem**: Plot titles show very small or very large time values
**Cause**: Incorrect time unit detection
**Solution**: Check NetCDF time variable units or modify smart detection ranges

#### Memory Errors
**Problem**: Out of memory during parallel processing
**Cause**: Too many parallel workers for available RAM
**Solution**: Reduce `--max-workers` or use sequential processing

#### Missing Bedrock Profiles
**Problem**: Cannot find bedrock configuration
**Cause**: Missing profile files in `../bedrock_profiles/`
**Solution**: Ensure bedrock profile files exist and are accessible

### Performance Tips

1. **Use SSD storage** for faster I/O with large NetCDF files
2. **Limit parallel workers** to available RAM (each worker loads full dataset)
3. **Skip existing files** when rerunning batches: `--skip-existing`
4. **Use resume functionality** for long-running batches: `--resume`
5. **Process by experiment type** using patterns: `--pattern="*_S4_*"`

## Dependencies

### Required Python Packages
- `numpy` - Numerical computations
- `netCDF4` - NetCDF file reading
- `matplotlib` - Plotting and visualization
- `bamgflowband` - Adaptive mesh generation
- Custom modules: `bedrock_generator`, domain utilities

### System Requirements
- Python 3.6+
- Sufficient RAM for dataset size (typically 1-8GB per file)
- Multi-core CPU recommended for parallel processing

## Integration with ISSM Workflow

These processing scripts are designed to work with:
1. **ISSM simulation output**: NetCDF files from flowline simulations
2. **Bedrock configurations**: Synthetic bedrock profile definitions  
3. **Domain utilities**: Mesh generation and optimization tools
4. **Analysis pipelines**: Downstream analysis of generated visualizations