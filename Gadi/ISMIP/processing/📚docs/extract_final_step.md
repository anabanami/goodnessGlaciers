# extract_final_step.py

## Overview

A Python script that processes ISSM (Ice Sheet System Model) transient NetCDF output files to generate visualizations of the final simulation state and velocity evolution summary. This is an optimized version of `extract_results.py` that focuses on the final time step rather than all time steps.

## Functionality

The script performs the following operations:

1. **Mesh Reconstruction**: Infers the original parameter file and resolution factor from the input filename, then reconstructs the 3D model mesh by replicating the meshing and extrusion steps
2. **Data Extraction**: Opens the transient NetCDF file and reads solution data, focusing on the final time step
3. **Velocity Evolution Plot**: Creates a summary plot showing maximum velocity evolution over the entire simulation timeline
4. **Final State Visualization**: Creates plots for the final time step showing:
   - Surface layer: Velocity fields (Vx, Vy, Vz, Vel)
   - Basal layer: Velocity fields (Vx, Vy, Vz, Vel) and Pressure field
5. **Batch Processing**: Supports processing multiple files using glob patterns

## Usage

```bash
# Single file
python extract_final_step.py <transient_results.nc>

# Multiple files
python extract_final_step.py file1.nc file2.nc file3.nc

# Wildcard pattern
python extract_final_step.py *.nc
```

### Examples

```bash
# Process single file
python extract_final_step.py IsmipF_S1_1-Transient.nc

# Process all NetCDF files in directory
python extract_final_step.py *.nc

# Process specific pattern
python extract_final_step.py IsmipF_*-Transient.nc
```

## Input Requirements

- **Input File(s)**: One or more transient NetCDF files from ISSM simulation
- **Parameter File**: The script expects corresponding parameter files (e.g., `IsmipF.py`) to exist in the parent directory
- **Filename Convention**: Input filenames should follow the pattern `<ParamName>_<Scenario>_<Resolution>-Transient.nc`

## Output

### Directory Structure
- Creates a directory named after each input file (without extension) with `_FINAL` suffix
- Within each file directory, creates subdirectories for each layer:
  - `Surface/` - Contains surface layer plots
  - `Base/` - Contains basal layer plots
- Evolution summary plot is saved in the root file directory

### Generated Files
- **Final State Plots**: `final_<field>.png` for each field, organized by layer:
  - `Surface/Vx/final_Vx.png`, `Surface/Vy/final_Vy.png`, etc.
  - `Base/Vx/final_Vx.png`, `Base/Vy/final_Vy.png`, `Base/Pressure/final_Pressure.png`, etc.
- **Evolution Summary**: `velocity_evolution.png` showing maximum velocity over time (in root directory)

### Plot Characteristics
- **Resolution**: 120 DPI for final state plots, 150 DPI for evolution plot
- **Size**: 12x6 inches for final state plots, 10x6 inches for evolution plot
- **Colormaps**: 
  - Velocity components (Vx, Vy, Vz): coolwarm
  - Velocity magnitude (Vel): viridis
  - Pressure: plasma

## Key Features

- **Efficient Processing**: Only processes the final time step instead of all steps
- **Batch Support**: Can process multiple files in a single command
- **Wildcard Support**: Uses glob patterns for flexible file selection
- **Progress Feedback**: Provides detailed status messages and timing information
- **Error Handling**: Graceful handling of missing files, corrupted data, and mesh reconstruction errors

## Dependencies

- numpy
- matplotlib
- netCDF4
- glob (standard library)
- ISSM Python libraries (model, squaremesh, parameterize, pyissm)

## Configuration

- **Time Conversion**: Uses `SECONDS_PER_YEAR = 31556926.0` for time unit conversion
- **Mesh Parameters**: Fixed domain size (100km x 100km) with configurable resolution
- **Layer Selection**: Both surface and basal layers for velocity fields, basal layer (layer 1) for pressure

## Error Handling

- Validates input file existence
- Checks for parameter file availability
- Handles mesh reconstruction failures
- Skips files with insufficient time steps (≤1)
- Continues processing remaining files if one fails in batch mode
- Provides detailed error messages and warnings

## Performance

Significantly faster than `extract_results.py` for large simulations since it only processes the final time step rather than generating plots for every time step in the simulation.