# plot_transient_fields.py

## Overview
A Python script for analyzing and visualizing transient ice sheet model results from ISSM (Ice Sheet System Model) simulations. The script automatically finds NetCDF output files, recreates the corresponding model geometry, and generates comprehensive plots of the final simulation state.

## Dependencies
- `numpy`, `matplotlib`, `xarray` - Scientific computing and visualization
- `pyISSM` - Ice Sheet System Model Python interface
- `bedrock_generator.py` - Custom bedrock generation module (required in same directory)

## Core Functions

### `adaptive_bamg(md, x, s0, b, bed_wavelength, ice_thickness, resolution_factor)`
Recreates adaptive mesh based on bedrock wavelength and ice thickness parameters.

**Parameters:**
- `md` - ISSM model object
- `x` - 1D coordinate array
- `s0` - Surface elevation array
- `b` - Bed elevation array
- `bed_wavelength` - Wavelength of bedrock topography
- `ice_thickness` - Ice thickness value
- `resolution_factor` - Mesh resolution multiplier (default: 1.0)

### `recreate_model_and_mesh(profile_id, resolution_factor)`
Sets up complete ISSM model with geometry based on profile ID and resolution factor.

**Parameters:**
- `profile_id` - Integer identifier for bedrock profile configuration
- `resolution_factor` - Mesh resolution scaling factor

**Returns:** Configured ISSM model object with mesh and geometry

### `plot_transient_fields(md, nc_filename, profile_id, exp)`
Main visualization function that generates plots from final time step of transient simulation.

**Generated Plots:**
- `velocity_final_{profile_id:03d}_{exp}.png` - Velocity magnitude field
- `pressure_final_{profile_id:03d}_{exp}.png` - Pressure field  
- `thickness_final_{profile_id:03d}_{exp}.png` - Ice thickness field
- `elevations_final_{profile_id:03d}_{exp}.png` - Surface, base, and bed elevation profiles

### `plot_max_velocity_from_netcdf(filename)`
Creates time evolution plot of maximum velocity throughout the simulation.

**Output:** `{filename_base}_max_vel_evolution.png`

### `find_and_process_netcdf()`
Main entry point that automatically discovers and processes NetCDF files in current directory.

## Usage

### Automatic Processing
```bash
python plot_transient_fields.py
```
The script automatically:
1. Searches for NetCDF files matching pattern `XXX_SY_Z.nc` (e.g., `165_S3_0.875.nc`)
2. Parses filename to extract profile ID, experiment number, and resolution factor
3. Recreates model geometry
4. Generates all visualization plots

### File Naming Convention
Expected NetCDF filename format: `{profile_id}_S{exp_num}_{resolution_factor}.nc`

**Example:** `165_S3_0.875.nc`
- Profile ID: 165
- Experiment: S3
- Resolution factor: 0.875

## Output Files
All plots are saved as high-resolution PNG files (300 DPI) with descriptive filenames including profile ID and experiment identifier.

## Configuration
- Domain length: 210 km (25 km inlet buffer + 160 km interest region + 25 km terminus buffer)
- Mesh refinement: Adaptive based on bedrock wavelength (<15 km: factor 50, ≥15 km: factor 200)
- Anisotropy maximum: 3
- Vertical mesh layers: 1 (flowband model)

## Error Handling
The script includes comprehensive error handling for:
- Missing NetCDF files
- Incorrect filename formats
- NetCDF data loading failures
- Missing dependencies

## Requirements
- `bedrock_generator.py` must be accessible in the same directory or Python path
- pyISSM installation with correct path configuration
- NetCDF files must contain `results/TransientSolution` group with required variables: `Vel`, `Pressure`, `Thickness`, `Surface`, `Base`, `time`