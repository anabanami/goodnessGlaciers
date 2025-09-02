# extract_results.py

## Overview

A Python script that processes ISSM (Ice Sheet System Model) transient NetCDF output files to generate visualizations of simulation results over time.

## Functionality

The script performs the following operations:

1. **Mesh Reconstruction**: Infers the original parameter file and resolution factor from the input filename, then reconstructs the 3D model mesh by replicating the meshing and extrusion steps from the original simulation
2. **Data Extraction**: Opens the transient NetCDF file and reads solution data including time steps and field variables
3. **Visualization**: Creates plots for each time step showing:
   - Surface layer: Velocity fields (Vx, Vy, Vz, Vel)
   - Basal layer: Pressure field
4. **Output Management**: Saves all plots to organized directories named after the input file

## Usage

```bash
python extract_results.py <transient_results.nc>
```

### Example

```bash
python extract_results.py IsmipF_S1_1-Transient.nc
```

## Input Requirements

- **Input File**: A transient NetCDF file from ISSM simulation
- **Parameter File**: The script expects a corresponding parameter file (e.g., `IsmipF.py`) to exist in the parent directory
- **Filename Convention**: The input filename should follow the pattern `<ParamName>_<Scenario>_<Resolution>-Transient.nc`

## Output

- **Directory Structure**: Creates a directory named after the input file (without extension)
- **Subdirectories**: One subdirectory per field (Vx, Vy, Vz, Vel, Pressure)
- **Plot Files**: PNG images for each time step, named `<field>_step<XXX>.png`

## Dependencies

- numpy
- matplotlib
- netCDF4
- ISSM Python libraries (model, squaremesh, parameterize, pyissm)

## Configuration

- **Time Conversion**: Uses `SECONDS_PER_YEAR = 31556926.0` for time unit conversion
- **Plot Settings**: Configurable colormaps per field type (coolwarm, viridis, plasma)
- **Output Resolution**: 120 DPI for saved plots

## Error Handling

- Validates input file existence
- Checks for parameter file availability
- Handles mesh reconstruction failures
- Skips files with insufficient time steps (≤1)
- Provides progress feedback during plotting operations