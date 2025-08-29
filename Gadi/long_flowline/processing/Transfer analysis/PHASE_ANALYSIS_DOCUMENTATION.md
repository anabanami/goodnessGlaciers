# Phase Analysis Tools Documentation

This document provides comprehensive documentation for the phase analysis tools used to analyze the phase relationship between bed topography and ice surface evolution in glacier flowline models.

## Table of Contents

1. [Overview](#overview)
2. [Scripts](#scripts)
3. [Installation & Dependencies](#installation--dependencies)
4. [Usage](#usage)
5. [Input Requirements](#input-requirements)
6. [Output Description](#output-description)
7. [API Reference](#api-reference)
8. [Analysis Methods](#analysis-methods)
9. [Configuration](#configuration)
10. [Examples](#examples)

---

## Overview

The phase analysis tools quantify the spatial and temporal relationship between bed topography and ice surface response in glacier flowline simulations. The tools analyze ISSM (Ice-Sheet and Sea-Level System Model) outputs to determine:

- **Phase shifts** between bed undulations and surface response
- **Spatial lag distances** between bed and surface features
- **Correlation strength** between bed and surface signals
- **Time evolution** of the phase relationship

### Scientific Context

In glacier dynamics, the relationship between bed topography and surface topography provides insights into:
- Ice flow mechanics and stress transmission
- Response times to basal perturbations  
- Filtering effects of ice thickness and flow
- Validation of theoretical transfer functions

---

## Scripts

### `phase_analysis.py`
Single-file processor for detailed analysis of individual simulations.

**Purpose:** Comprehensive analysis of one netCDF file with detailed visualizations and statistics.

**Best for:** 
- Detailed investigation of specific cases
- Quality control and validation
- Generating publication-ready figures

### `batch_phase_analysis.py`
Multi-file processor for large-scale parameter studies.

**Purpose:** Automated processing of multiple scenarios and parameter combinations.

**Best for:**
- Parameter sweeps and sensitivity studies
- Comparative analysis across scenarios
- High-throughput data processing

---

## Installation & Dependencies

### Required Python Packages
```bash
pip install numpy matplotlib scipy netcdf4
```

### ISSM Dependencies
Requires ISSM installation with Python bindings:
- `bamgflowband` - Mesh generation
- `bedrock_generator.SyntheticBedrockModelConfig` - Configuration management

### Directory Structure
```
project_root/
├── bedrock_profiles/
│   └── bedrock_profile_*.npz        # Bedrock configuration files
├── flowline/
│   └── processing/
│       └── Transfer analysis/
│           ├── S1/, S2/, S3/, S4/   # Scenario directories
│           ├── phase_analysis.py
│           └── batch_phase_analysis.py
```

---

## Usage

### Single File Analysis
```bash
python phase_analysis.py <filename.nc>
```

**Example:**
```bash
python phase_analysis.py 002_S1_0.5.nc
```

### Batch Processing
```bash
python batch_phase_analysis.py
```

Automatically processes all `.nc` files in scenario subdirectories (`S1/`, `S2/`, `S3/`, `S4/`).

---

## Input Requirements

### NetCDF File Structure
Required netCDF structure from ISSM simulations:
```
dataset.groups['results'].groups['TransientSolution']
├── variables['time'][:]              # Time steps (years)
├── variables['Surface'][time, space] # Surface elevation (m)
└── [other ISSM output variables]
```

### Bedrock Configuration Files
Required `.npz` files in `../../bedrock_profiles/`:
- Naming: `bedrock_profile_XXX.npz` (where XXX is 3-digit profile ID)
- Contains: Bedrock parameters (wavelength, amplitude, etc.)

### File Naming Convention
NetCDF files: `{profile_id}_{scenario}_{resolution}.nc`
- `profile_id`: 3-digit bedrock profile identifier
- `scenario`: Experiment type (S1, S2, S3, S4)  
- `resolution`: Mesh resolution factor

**Examples:**
- `002_S1_0.5.nc` - Profile 2, Scenario S1, 0.5x resolution
- `156_S3_0.5.nc` - Profile 156, Scenario S3, 0.5x resolution

---

## Output Description

### Directory Structure
```
{profile_id}_{scenario}_phase_analysis/
├── signals/                    # Bed-surface comparison plots
├── correlations/              # Cross-correlation analysis
├── phase_relationship/        # Normalized signal comparisons  
├── phase_evolution_{id}.png   # Time series summary
└── phase_analysis_summary.txt # Numerical results
```

### Plot Types

#### 1. Signal Plots (`signals/`)
Two-panel plots showing:
- **Top panel:** Bed and surface geometry with ice thickness
- **Bottom panel:** Detrended bed and surface signals

#### 2. Correlation Plots (`correlations/`)
Cross-correlation analysis showing:
- Cross-correlation function vs spatial lag
- Maximum correlation location
- Theoretical phase markers (±90°)

#### 3. Phase Relationship Plots (`phase_relationship/`)
Normalized signal comparison showing:
- Bed and surface signals (normalized)
- Peak identification and alignment
- Phase shift quantification

#### 4. Evolution Summary (`phase_evolution_{id}.png`)
Time series showing:
- Phase shift evolution (degrees)
- Spatial lag evolution (km)
- Reference lines for theoretical values

### Summary File Format
Tab-separated text file with:
```
Time (years) | Phase (degrees) | Lag (km) | Correlation
0.20         | 81.2           | 0.864    | 0.892
12.50        | 78.9           | 0.841    | 0.876
...
```

---

## API Reference

### Core Functions

#### `analyse_phase_evolution(dataset, wavelength, config, profile_id, output_dir)`
Main analysis function performing complete phase evolution analysis.

**Parameters:**
- `dataset`: Open netCDF4.Dataset object
- `wavelength`: Bedrock wavelength (km)  
- `config`: SyntheticBedrockModelConfig object
- `profile_id`: Profile identifier (integer) for plot filenames
- `output_dir`: Output directory path

**Returns:**
```python
{
    'time': [time_values],           # Years
    'phase_shift': [phase_radians],  # Radians
    'phase_shift_deg': [phase_deg],  # Degrees  
    'lag_distance': [lag_km],        # Kilometers
    'correlation': [correlations],   # Correlation coefficients
    'wavelength': wavelength         # Bedrock wavelength (km)
}
```

#### `load_time_step_data(dataset, time_step, config)`
Extract and process data for specific time step with intelligent mesh handling.

**Parameters:**
- `dataset`: netCDF4.Dataset object
- `time_step`: Time step index (integer)
- `config`: SyntheticBedrockModelConfig object

**Features:**
- Automatic mesh coordinate extraction from NetCDF or reconstruction
- Dimension validation between surface data and mesh coordinates
- Debug output for troubleshooting

**Returns:**
- `x`: Spatial coordinates (km)
- `bed`: Bed elevation (m)
- `surface`: Surface elevation (m)
- `time_val`: Time value (years)

#### `phase_shift_analysis(dx, bed, surface, wavelength, time_val=None)`
Calculate phase shift between bed and surface signals.

**Parameters:**
- `dx`: Spatial resolution (km)
- `bed`: Bed elevation array (m)
- `surface`: Surface elevation array (m)  
- `wavelength`: Bedrock wavelength (km)
- `time_val`: Time for logging (years, optional)

**Returns:**
- `phase_shift`: Phase shift (radians)
- `lag_distance`: Spatial lag (km)
- `max_corr`: Maximum correlation coefficient

### Plotting Functions

#### `plot_signals(x, bed, surface, time_val, config, profile_id, output_dir, ylim_main=None, ylim=None)`
Generate two-panel bed vs surface comparison plots.

#### `visualise_cross_correlation(x, bed_slope, surface_slope, wavelength, time_val, profile_id, output_dir)`
Create cross-correlation visualization plots.

#### `visualise_phase_relationship(x, bed_slope, surface_slope, wavelength, time_val, profile_id, output_dir)`
Generate normalized signal comparison plots.

**Note:** All plotting functions now require `profile_id` parameter for unique filename generation.

### Utility Functions

#### `parse_filename(ncfile)`
Extract profile ID and experiment from filename.

#### `load_config(ncfile, bedrock_file)`
Load bedrock configuration from .npz file.

#### `build_mesh_coordinates_from_netcdf(dataset, config)`
Extract mesh coordinates from NetCDF file or reconstruct using flowline.py algorithm.

#### `build_mesh_coordinates_flowline_exact(config, resolution_factor=0.5)`
Reconstruct mesh coordinates exactly matching the flowline.py adaptive_bamg function.

#### `build_mesh_coordinates(config, resolution_factor=0.5)`
Legacy function maintained for backward compatibility.

---

## Analysis Methods

### Phase Shift Calculation

1. **Signal Preprocessing:**
   ```python
   bed_norm = (bed - np.mean(bed)) / np.std(bed)
   surface_norm = (surface - np.mean(surface)) / np.std(surface)
   ```

2. **Cross-Correlation:**
   ```python
   xcorr = correlate(bed_norm, surface_norm, mode='full')
   max_corr_idx = np.argmax(xcorr)
   lag_distance = (max_corr_idx - center_idx) * dx
   ```

3. **Phase Wrapping:**
   Constrains lag to ±λ/2 to avoid phase ambiguity:
   ```python
   if abs(lag_distance) > wavelength/2:
       lag_distance = lag_distance % wavelength
       if lag_distance > wavelength/2:
           lag_distance -= wavelength
   ```

4. **Phase Conversion:**
   ```python
   phase_shift = (2π × lag_distance) / wavelength
   ```

### Mesh Generation
The scripts now feature intelligent mesh coordinate handling:

1. **Primary Method:** Extract mesh coordinates directly from NetCDF file if available
2. **Fallback Method:** Reconstruct mesh using exact flowline.py algorithm

**Reconstruction Parameters:**
- **Domain optimization:** Finds optimal domain length ending at bedrock peak
- **Short wavelengths** (<15 km): `refinement_factor = 50`
- **Long wavelengths** (≥15 km): `refinement_factor = 200`  
- **Resolution:** `hmax = (wavelength / refinement_factor) × resolution_factor`

**Dimension Validation:**
- Automatic verification that surface data and mesh coordinates have matching dimensions
- Clear error messages for dimension mismatches

### Center-line Extraction
- Primary: `|y| ≤ 0.05 km` tolerance
- Fallback: Bottom 10% of y-coordinates
- Duplicate removal: 0.1 mm tolerance

---

## Configuration

### Scenario Types
- **S1:** Linear rheology + frozen bed (no-slip)
- **S2:** Non-linear rheology + frozen bed  
- **S3:** Linear rheology + sliding law
- **S4:** Non-linear rheology + sliding law

### Time Conversion
**Important Update:** ISSM outputs time values that are already in years, so no conversion is needed:
```python
# Current implementation (fixed)
time_val = float(times[time_step])  # Times are already in years

# Old implementation (incorrect)
# YTS = 31556926  # Seconds per year
# time_years = time_seconds / YTS  # This was causing t=0.0 for all steps
```

**Note:** The filename `final_time=300_yrs_timestep=0.08333333333333333_yrs.nc` indicates 300-year simulations, and the time array in the NetCDF file contains values like [0.104, 13.021, 26.042, ...] which are already in years.

### Resolution Factor
Default: `0.5` (fine resolution matching typical simulation setup)

---

## Examples

### Example 1: Single Analysis
```python
import netCDF4 as nc
from phase_analysis import analyse_phase_evolution, load_config

# Load configuration
config, wavelength_km = load_config('002_S1_0.5.nc', 
                                   '../../bedrock_profiles/bedrock_profile_002.npz')

# Run analysis  
with nc.Dataset('002_S1_0.5.nc', 'r') as dataset:
    results = analyse_phase_evolution(dataset, wavelength_km, config, 2, 'output_dir')

# Extract results
print(f"Wavelength: {results['wavelength']:.2f} km")
print(f"Final phase: {results['phase_shift_deg'][-1]:.1f}°")
print(f"Final correlation: {results['correlation'][-1]:.3f}")
```

### Example 2: Custom Processing
```python
from phase_analysis import load_time_step_data, phase_shift_analysis

# Process specific time step
x, bed, surface, time_val = load_time_step_data(dataset, time_step=5, config=config)

# Calculate phase shift
dx = np.mean(np.diff(x))
bed_slope = np.gradient(bed, x, edge_order=1)
surface_slope = np.gradient(surface, x, edge_order=1)

phase, lag, corr = phase_shift_analysis(dx, bed_slope, surface_slope, 
                                       wavelength_km, time_val)

print(f"Time {time_val:.1f} yr: Phase = {np.degrees(phase):.1f}°, "
      f"Lag = {lag:.3f} km, Correlation = {corr:.3f}")
```

### Example 3: Batch Results Analysis
```python
import glob
import pandas as pd

# Collect all summary files
summary_files = glob.glob('phase_analysis_results/S1/*/summary.txt')

data = []
for file in summary_files:
    # Parse profile ID from path
    profile_id = file.split('/')[-2].split('_')[0]
    
    # Load summary data
    df = pd.read_csv(file, skiprows=2, names=['Time', 'Phase', 'Lag', 'Corr'])
    df['Profile'] = profile_id
    data.append(df)

# Combine all results
all_results = pd.concat(data, ignore_index=True)

# Analyze final phase shifts
final_phases = all_results.groupby('Profile')['Phase'].last()
print(f"Mean final phase shift: {final_phases.mean():.1f}°")
print(f"Phase shift std dev: {final_phases.std():.1f}°")
```

---

## Performance Notes

- **Processing time:** ~30 seconds per file (reduced if mesh in NetCDF)
- **Memory usage:** ~100 MB per analysis
- **Bottleneck:** Mesh reconstruction (bamgflowband calls) when NetCDF mesh unavailable
- **Optimization:** Intelligent mesh handling reduces processing time when mesh coordinates are stored in NetCDF
- **Batch processing:** Several hours for complete dataset

## Recent Fixes (v1.1)

### Time Conversion Bug Fix
**Issue:** All plots were being overwritten with `t=0.0` in filenames because time values were incorrectly being divided by `YTS`.

**Solution:** Removed time conversion since ISSM NetCDF files already store time in years.

**Impact:** 
- Plot filenames now correctly show distinct time values (e.g., `signals_t0.1_165.png`, `signals_t13.0_165.png`)
- Time evolution plots now show proper temporal progression
- No more plot overwriting issues

### Profile ID Parameter Addition
**Issue:** Plotting functions referenced undefined `profile_id` variable causing runtime errors.

**Solution:** Added `profile_id` parameter to all plotting functions and updated function calls.

**Impact:**
- All plotting functions now work correctly
- Unique plot filenames include profile identifier
- Both `phase_analysis.py` and `batch_phase_analysis.py` are functional

---

*Documentation for phase analysis tools v1.1*
*Compatible with ISSM flowline simulation outputs*