# convert_to_nc.py

## Overview

A Python script that converts ISSM (Ice Sheet System Model) binary output files (.outbin) to NetCDF format with support for both flat and grouped layouts compatible with downstream ISSM tools.

## Functionality

The script provides the following capabilities:

1. **Binary Format Reading**: Parses ISSM .outbin binary files using the `ISSMOutbinReader` class
2. **Data Type Support**: Handles multiple ISSM data types:
   - Type 1: Scalar double values
   - Type 2: String values
   - Type 3: Double precision arrays
   - Type 4: Integer arrays
   - Type 5: Complex number arrays
3. **Layout Options**: Creates NetCDF files in either flat or grouped structure
4. **Analysis Mode**: Option to analyze .outbin files without conversion

## Usage

```bash
python convert_to_nc.py input.outbin [output.nc] [--layout grouped|flat] [--analyse-only]
```

### Arguments

- `input_file`: Path to the ISSM .outbin file (required)
- `output_file`: Output NetCDF filename (optional, defaults to input filename with .nc extension)
- `--layout`: Output layout format (optional, choices: `grouped` or `flat`, default: `grouped`)
- `--analyse-only`: Only analyze the .outbin file without creating NetCDF output (optional)

### Examples

```bash
# Convert with default grouped layout
python convert_to_nc.py simulation.outbin

# Convert with custom output filename and flat layout
python convert_to_nc.py simulation.outbin results.nc --layout flat

# Analyze file contents without conversion
python convert_to_nc.py simulation.outbin --analyse-only
```

## Output Formats

### Grouped Layout (Default)
Creates a NetCDF file with ISSM-style hierarchy:
```
results/
  TransientSolution/
    time[time]
    step[time]
    <variable_name>[time, dim1, dim2]
```

### Flat Layout
Creates a NetCDF file with all variables at the root level:
```
time[time]
step[time]
<variable_name>[time, dim1, dim2]
```

## NetCDF Metadata

The output file includes comprehensive metadata:
- `title`: "ISSM Simulation Output"
- `description`: Conversion details and original filename
- `creation_date`: ISO format timestamp
- `source`: "ISSM"
- `converter`: Script name
- `original_file`: Path to source .outbin file

## Variable Attributes

Each converted variable includes:
- `issm_type`: Original ISSM data type (1-5)
- `original_shape`: Original array dimensions

## Dependencies

- numpy
- netCDF4
- argparse (standard library)
- struct (standard library)
- os (standard library)
- datetime (standard library)

## Error Handling

- Validates input file existence and format
- Handles corrupted or incomplete binary data
- Provides detailed error messages for debugging
- Supports graceful handling of unknown data types

## Data Type Mapping

| ISSM Type | Description | NetCDF Type |
|-----------|-------------|-------------|
| 1 | Scalar double | f8 |
| 2 | String | S1 (char array) |
| 3 | Double array | f8 |
| 4 | Integer array | i4 |
| 5 | Complex array | c16 |