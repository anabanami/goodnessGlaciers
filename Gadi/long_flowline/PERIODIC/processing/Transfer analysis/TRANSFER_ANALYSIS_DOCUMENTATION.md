# Transfer Analysis Tool Documentation

This document provides comprehensive documentation for the `transfer_analysis.py` script, which analyzes the transfer function between bed topography and ice surface elevation in glacier flowline simulations.

## Table of Contents

1. [Overview](#overview)
2. [Scientific Purpose](#scientific-purpose)
3. [Installation & Dependencies](#installation--dependencies)
4. [Usage](#usage)
5. [Input Requirements](#input-requirements)
6. [Output Description](#output-description)
7. [Analysis Methods](#analysis-methods)
8. [API Reference](#api-reference)
9. [Configuration](#configuration)
10. [Examples](#examples)
11. [Performance Notes](#performance-notes)

---

## Overview

The `transfer_analysis.py` script performs comprehensive analysis of the bed-to-surface transfer function in glacier dynamics simulations. It quantifies how bedrock topography influences ice surface elevation across different scenarios and bedrock wavelengths.

### Key Features

- **Automated batch processing** of multiple NetCDF simulation files
- **Transfer function analysis** using amplitude ratios and phase shifts
- **Theoretical comparison** with Budd damping theory
- **Comprehensive visualization** including scatter plots and matrix heatmaps
- **Intelligent mesh handling** with dimension validation
- **CSV output** for further analysis

---

## Scientific Purpose

### Transfer Function Theory

In glacier dynamics, the relationship between bed topography and surface response is characterized by:

1. **Amplitude Ratio** (A_s/A_b): How much surface amplitude is reduced compared to bed amplitude
2. **Phase Shift**: Spatial lag between bed and surface features
3. **Wavelength Dependence**: How transfer characteristics vary with bedrock wavelength

### Theoretical Framework

The analysis compares measured values against **Budd's analytical theory**:

```
ψ = 1 / (1 + (k·Z)²)^0.5
```

Where:
- `ψ` = theoretical amplitude ratio
- `k` = wavenumber (2π/wavelength)
- `Z` = ice thickness

### Scenario Types

- **S1**: Linear rheology + frozen bed (no-slip)
- **S2**: Non-linear rheology + frozen bed  
- **S3**: Linear rheology + sliding law
- **S4**: Non-linear rheology + sliding law

---

## Installation & Dependencies

### Required Python Packages
```bash
pip install numpy matplotlib scipy netcdf4 pandas
```

### ISSM Dependencies
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
│           ├── S1/, S2/, S3/, S4/   # Scenario directories with .nc files
│           ├── transfer_analysis.py
│           └── transfer_plots/      # Generated output directory
```

---

## Usage

### Basic Execution
```bash
python transfer_analysis.py
```

The script automatically:
1. Discovers all bedrock profile configurations
2. Searches for NetCDF files in scenario subdirectories
3. Processes each profile-scenario combination
4. Generates comprehensive analysis and plots

### Expected Runtime
- **Small dataset**: ~30 minutes for 50 profiles
- **Large dataset**: Several hours for 875 profiles
- **Bottleneck**: Mesh reconstruction for files without embedded mesh data

---

## Input Requirements

### NetCDF File Structure
Required structure from ISSM simulations:
```
dataset.groups['results'].groups['TransientSolution']
└── variables['Surface'][-1]  # Final surface elevation (m)
```

### Bedrock Configuration Files
- **Location**: `../../bedrock_profiles/bedrock_profile_XXX.npz`
- **Contains**: Wavelength, amplitude, ice thickness parameters
- **Format**: NumPy compressed archive

### File Naming Convention
NetCDF files: `{profile_id}_{scenario}_{resolution}.nc`

**Examples:**
- `165_S1_0.5.nc` - Profile 165, Scenario S1, 0.5x resolution
- `002_S3_0.5.nc` - Profile 2, Scenario S3, 0.5x resolution

---

## Output Description

### Primary Outputs

#### 1. CSV Results File: `transfer_function_results.csv`
Contains quantitative results for all processed profiles:

| Column | Description | Units |
|--------|-------------|-------|
| `profile_id` | Bedrock profile identifier | - |
| `scenario` | Experiment scenario (S1-S4) | - |
| `scenario_name` | Human-readable scenario description | - |
| `wavelength_b` | Bedrock wavelength | km |
| `ice_thickness` | Ice thickness | km |
| `amplitude_ratio` | Measured surface/bed amplitude ratio | - |
| `theory_amp_ratio` | Theoretical amplitude ratio (Budd) | - |
| `wavelength_ratio` | Surface/bed wavelength ratio | - |
| `phase_shift_rad` | Phase shift | radians |
| `phase_shift_deg` | Phase shift | degrees |
| `lag_distance` | Spatial lag distance | km |
| `max_correlation` | Cross-correlation coefficient | - |

#### 2. Visualization Directory: `transfer_plots/`
```
transfer_plots/
├── amplitude_ratio.png           # Scatter plot: wavelength vs amplitude ratio
├── phase_shift.png              # Scatter plot: wavelength vs phase shift
├── wavelength_ratio.png         # Scatter plot: wavelength vs wavelength ratio
├── amplitude_ratio_matrix.png   # Heatmap: scenario vs wavelength (amplitude)
└── phase_shift_matrix.png       # Heatmap: scenario vs wavelength (phase)
```

### Plot Descriptions

#### Scatter Plots
- **X-axis**: Bedrock wavelength (km)
- **Y-axis**: Transfer function metrics
- **Color/Symbol**: Different scenarios (S1-S4)
- **Reference lines**: Theoretical predictions where applicable

#### Matrix Heatmaps
- **Rows**: Scenarios (S1, S2, S3, S4)
- **Columns**: Bedrock wavelengths (sorted)
- **Colors**: Transfer function values (amplitude ratio or phase shift)
- **Format**: High-resolution PNG (300 DPI)

---

## Analysis Methods

### 1. Data Loading and Mesh Handling

The script features intelligent mesh coordinate extraction:

```python
def build_mesh_coordinates_from_netcdf(dataset, config):
    # Try NetCDF embedded mesh first
    # Fallback to flowline.py reconstruction algorithm
```

**Key Features:**
- Automatic dimension validation
- Clear error messages for mismatches
- Debug output for troubleshooting

### 2. Transfer Function Calculation

#### Phase Shift Analysis
```python
def phase_shift_analysis(dx, bed, surface, λb):
    # 1. Normalize signals
    bed_norm = (bed - np.mean(bed)) / np.std(bed)
    surface_norm = (surface - np.mean(surface)) / np.std(surface)
    
    # 2. Cross-correlation
    xcorr = correlate(bed_norm, surface_norm, mode='full')
    
    # 3. Find maximum correlation lag
    max_corr_idx = np.argmax(xcorr)
    lag_distance = (max_corr_idx - center_idx) * dx
    
    # 4. Phase wrapping (±λ/2 constraint)
    if abs(lag_distance) > λb/2:
        lag_distance = lag_distance % λb
        if lag_distance > λb/2:
            lag_distance -= λb
    
    # 5. Convert to phase shift
    phase_shift = (2 * π * lag_distance) / λb
```

#### Amplitude Ratio Calculation
```python
# Calculate RMS amplitudes
bed_slope = np.gradient(bed, x)
surface_slope = np.gradient(surface, x)

bed_amplitude = np.sqrt(np.mean(bed_slope**2))
surface_amplitude = np.sqrt(np.mean(surface_slope**2))

amplitude_ratio = surface_amplitude / bed_amplitude
```

### 3. Theoretical Comparison

**Budd Damping Theory:**
```python
Z_km = ice_thickness / 1e3
k = 2 * π / wavelength_b
ωZ = k * Z_km
ψ = 1 / np.sqrt(1 + ωZ**2)  # Theoretical amplitude ratio
```

### 4. Mesh Reconstruction

Exactly replicates the `flowline.py` adaptive meshing algorithm:

- **Domain optimization**: Finds optimal length ending at bedrock peak
- **Adaptive resolution**: 
  - Short wavelengths (<15 km): `refinement_factor = 50`
  - Long wavelengths (≥15 km): `refinement_factor = 200`
- **Mesh parameters**: `hmax = (wavelength / refinement_factor) × resolution_factor`

---

## API Reference

### Core Functions

#### `analyse_bed_surface_transfer()`
Main analysis function that orchestrates the entire workflow.

**Returns:**
```python
pandas.DataFrame  # Complete results with all metrics
```

#### `process_profile(ncfile, bedrock_file, scenario_key, scenario_name, profile_id, do_plot=True)`
Process a single profile-scenario combination.

**Parameters:**
- `ncfile`: Path to NetCDF simulation file
- `bedrock_file`: Path to bedrock configuration (.npz)
- `scenario_key`: Scenario identifier (S1, S2, S3, S4)
- `scenario_name`: Human-readable scenario description
- `profile_id`: Profile ID number
- `do_plot`: Whether to generate individual plots

**Returns:**
```python
dict  # Single result record with all metrics
```

#### `load_data(dataset, config)`
Extract and process surface data with intelligent mesh handling.

**Parameters:**
- `dataset`: Open netCDF4.Dataset object
- `config`: SyntheticBedrockModelConfig object

**Features:**
- Automatic mesh coordinate extraction or reconstruction
- Dimension validation with clear error messages
- Center-line extraction with tolerance fallback

**Returns:**
- `x`: Spatial coordinates (km)
- `surface`: Final surface elevation (m)
- `b0`: Bedrock elevation (m)

#### `phase_shift_analysis(dx, bed, surface, λb)`
Calculate phase shift and correlation between bed and surface signals.

**Parameters:**
- `dx`: Spatial resolution (km)
- `bed`: Bed elevation array (m)
- `surface`: Surface elevation array (m)
- `λb`: Bedrock wavelength (km)

**Returns:**
- `phase_shift`: Phase shift (radians)
- `lag_distance`: Spatial lag (km)
- `max_corr`: Maximum correlation coefficient

### Utility Functions

#### `load_config(ncfile, bedrock_file)`
Load bedrock configuration from .npz file and extract wavelength.

#### `build_mesh_coordinates_from_netcdf(dataset, config)`
Intelligent mesh coordinate extraction with NetCDF preference.

#### `generate_plots(df)`
Generate comprehensive visualization suite from results DataFrame.

---

## Configuration

### Scenario Definitions
```python
SCENARIO_CONFIGS = {
    'S1': 'Linear rheology + frozen bed',
    'S2': 'Non-linear rheology + frozen bed', 
    'S3': 'Linear rheology + sliding law',
    'S4': 'Non-linear rheology + sliding law'
}
```

### Plot Styling
- **DPI**: 300 (high resolution)
- **Color scheme**: Distinct colors for each scenario
- **Markers**: Different symbols for visual distinction
- **Format**: PNG with tight bounding boxes

### Processing Parameters
- **Center-line tolerance**: 0.05 km
- **Duplicate removal**: 0.1 mm spatial tolerance
- **Phase wrapping**: ±λ/2 constraint
- **Mesh resolution**: 0.5x factor (fine resolution)

---

## Examples

### Example 1: Basic Usage
```bash
# Run complete analysis
python transfer_analysis.py

# Check outputs
ls transfer_plots/
cat transfer_function_results.csv | head -10
```

### Example 2: Results Analysis
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('transfer_function_results.csv')

# Analyze S1 scenario
s1_data = df[df['scenario'] == 'S1']

# Plot wavelength vs amplitude ratio
plt.figure(figsize=(8, 6))
plt.scatter(s1_data['wavelength_b'], s1_data['amplitude_ratio'], 
           alpha=0.7, label='Measured')
plt.scatter(s1_data['wavelength_b'], s1_data['theory_amp_ratio'], 
           alpha=0.7, label='Theory (Budd)')
plt.xlabel('Bedrock Wavelength (km)')
plt.ylabel('Amplitude Ratio')
plt.legend()
plt.title('S1: Linear Rheology + Frozen Bed')
plt.show()

# Summary statistics
print(f"Mean amplitude ratio: {s1_data['amplitude_ratio'].mean():.3f}")
print(f"Mean phase shift: {s1_data['phase_shift_deg'].mean():.1f}°")
print(f"Correlation with theory: {s1_data['amplitude_ratio'].corr(s1_data['theory_amp_ratio']):.3f}")
```

### Example 3: Scenario Comparison
```python
# Compare scenarios by wavelength range
short_wavelengths = df[df['wavelength_b'] < 10]
long_wavelengths = df[df['wavelength_b'] > 20]

print("Short wavelengths (<10 km):")
print(short_wavelengths.groupby('scenario')['amplitude_ratio'].mean())

print("\nLong wavelengths (>20 km):")
print(long_wavelengths.groupby('scenario')['amplitude_ratio'].mean())
```

---

## Performance Notes

- **Processing time**: ~30-60 seconds per profile
- **Memory usage**: ~100-200 MB per analysis
- **Bottleneck**: Mesh reconstruction when NetCDF mesh unavailable
- **Optimization**: Embedded mesh coordinates significantly reduce processing time
- **Parallelization**: Could be implemented for large-scale studies

### Troubleshooting Common Issues

#### Dimension Mismatch Errors
```
IndexError: boolean index did not match indexed array along dimension 0
```
**Solution**: The improved mesh handling automatically resolves this by validating dimensions and using appropriate mesh coordinates.

#### Missing NetCDF Structure
```
KeyError: 'TransientSolution'
```
**Solution**: Ensure NetCDF files contain the required ISSM output structure with TransientSolution group.

#### Memory Issues
**Solution**: Process smaller batches or increase available RAM. The script loads one file at a time to minimize memory usage.

---

## Output Interpretation

### Amplitude Ratio
- **Values < 1**: Surface amplitude smaller than bed amplitude (typical)
- **Values ≈ 1**: Perfect transfer (rare)
- **Values > 1**: Surface amplification (unusual, may indicate numerical issues)

### Phase Shift
- **0°**: Perfect in-phase alignment
- **±90°**: Quarter-wavelength lag
- **±180°**: Half-wavelength lag (opposite phase)

### Wavelength Ratio
- **Values ≈ 1**: Surface wavelength matches bed wavelength
- **Values > 1**: Surface wavelength longer than bed
- **Values < 1**: Surface wavelength shorter than bed

### Correlation Coefficient
- **Values > 0.8**: Strong correlation between bed and surface
- **Values 0.5-0.8**: Moderate correlation
- **Values < 0.5**: Weak correlation (may indicate noise or complex dynamics)

---

*Documentation for transfer_analysis.py v1.0*  
*Compatible with ISSM flowline simulation outputs*