# LONG_FLOWLINE.PY Documentation

## Overview

`long_flowline.py` is a comprehensive Python script for ice sheet modeling using the Ice Sheet System Model (ISSM). It simulates ice flow along synthetic bedrock profiles using full-Stokes equations, with support for various ISMIP-HOM (Ice Sheet Model Intercomparison Project - Higher Order Models) experiments.

## Key Features

- **Ice sheet modeling** with ISSM framework
- **Synthetic bedrock generation** using configurable profiles
- **Multiple experiment types** (S1-S4) with different boundary conditions
- **Adaptive mesh generation** with resolution control
- **Full-Stokes flow equations** for accurate physics
- **Transient and diagnostic solutions**
- **ISMIP-HOM compatible output** formatting
- **Comprehensive diagnostics** and visualization

## Dependencies

```python
from bedrock_generator import SyntheticBedrockModelConfig
from socket import gethostname
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pyissm as issm
from pyissm import plot as iplt
```

### ISSM-specific modules:
- `model` - Core ISSM model class
- `cuffey` - Ice rheology calculations
- `bamgflowband` - Adaptive mesh generation
- `setflowequation` - Flow equation setup
- `friction` - Basal friction models
- `export_netCDF` - NetCDF output export
- `solve` - ISSM solver interface

## Main Functions

### Output and Results Processing

#### `format_output(md, L, filename)`
Formats simulation results according to ISMIP-HOM specifications.

**Parameters:**
- `md`: ISSM model object containing results
- `L`: Domain length (m)
- `filename`: Output filename prefix

**Returns:**
- `filename`: Path to saved output file
- `output_data`: Formatted numpy array with columns [x_hat, vx_surface, vz_surface, vx_basal]

**Functionality:**
- Extracts velocity fields (Vx, Vy, Vz) from stress balance solution
- Normalizes coordinates to 0-1 range
- Filters surface and basal vertices
- Saves tab-delimited text file with velocity data

#### `save_results(md, L, filename)`
High-level wrapper for saving simulation results with comprehensive diagnostics.

**Parameters:**
- `md`: ISSM model object
- `L`: Domain length (m)  
- `filename`: Output filename prefix

**Returns:**
- `filename`: Path to saved file (or None if error)

**Features:**
- Diagnostic output of available results
- Error handling and reporting
- Automatic result structure inspection

### Mesh Generation

#### `adaptive_bamg(md, x, s0, b, resolution_factor=1.0)`
Generates adaptive mesh using BAMG (Bidimensional Anisotropic Mesh Generator).

**Parameters:**
- `md`: ISSM model object
- `x`: 1D coordinate array (m)
- `s0`: Surface elevation array (m)
- `b`: Bed elevation array (m)
- `resolution_factor`: Mesh refinement multiplier (default: 1.0)

**Returns:**
- `md`: Updated model with mesh
- `nv`: Number of vertices
- `ne`: Number of elements
- `resolution_factor`: Applied resolution factor

**Algorithm:**
- Calculates wavelength-to-thickness ratio for adaptive refinement
- Sets maximum element size (`hmax`) based on bed wavelength
- Uses anisotropic mesh adaptation with vertical layers
- Provides detailed mesh statistics

### Boundary Conditions

#### `setup_non_periodic_boundary_conditions(md)`
Applies non-periodic boundary conditions for flowline simulation.

**Parameters:**
- `md`: ISSM model object

**Returns:**
- `md`: Model with boundary conditions applied

**Boundary conditions:**
- **Inlet (x=0)**: Depth-dependent velocity profile (calls `setup_depth_dependent_inlet_bc`)
- **Terminus (x=L)**: Neumann condition with free outflow (vx=NaN)

#### `setup_depth_dependent_inlet_bc(md, exp, u_b_sliding=20.0)`
Replaces simple Dirichlet BC with physically-based, depth-dependent velocity profile.

**Parameters:**
- `md`: ISSM model object
- `exp`: Experiment identifier ('S1', 'S2', 'S3', 'S4')
- `u_b_sliding`: Basal sliding velocity for sliding experiments (m/a)

**Returns:**
- `md`: Model with depth-dependent inlet velocity applied

**Physics:**
- Calculates surface slope at inlet
- Applies analytical velocity profile: u(z) = u_b + deformation velocity
- Accounts for sliding vs. no-slip scenarios
- Uses proper rheological parameters (n=1 for S1/S3, n=3 for S2/S4)

#### `plot_inlet_velocity_profile(md, exp)`
Generates visualization of the prescribed inlet velocity profile.

**Parameters:**
- `md`: ISSM model object
- `exp`: Experiment identifier

**Functionality:**
- Creates scatter plot of velocity vs. elevation
- Color-coded by velocity magnitude
- Saves as PNG file for diagnostic purposes

#### `diagnose_boundary_conditions(md)`
Diagnostic function to verify boundary condition setup.

**Features:**
- Counts prescribed velocity vertices
- Checks for over-constrained systems
- Validates terminus conditions
- Provides detailed boundary condition statistics

### Physics Setup

#### `setup_friction(md, exp)`
Configures basal friction according to experiment type.

**Parameters:**
- `md`: ISSM model object
- `exp`: Experiment identifier ('S1', 'S2', 'S3', 'S4')

**Returns:**
- `md`: Model with friction setup

**Friction laws:**
- **S1, S2**: No-slip boundary (frozen bed)
- **S3, S4**: Pattyn's linear sliding law with β² = 1500 Pa·a·m⁻¹

#### `debug_friction_setup(md)`
Diagnostic function for friction parameter verification.

### Analysis and Diagnostics

#### `analyse_driving_stress(md, L)`
Analyzes driving stress distribution and geometric trends.

**Parameters:**
- `md`: ISSM model object with solution
- `L`: Domain length (m)

**Returns:**
- `surface_slope`: Surface slope array
- `bed_slope`: Bed slope array  
- `thickness_sorted`: Ice thickness sorted by x-position

**Analysis includes:**
- Surface and bed elevation trends
- Ice thickness variations
- Driving stress at domain boundaries
- Systematic trend detection

#### `diagnose_acceleration_onset(md, L)`
Identifies locations where velocity acceleration becomes excessive.

**Parameters:**
- `md`: ISSM model with stress balance solution
- `L`: Domain length (m)

**Returns:**
- `x_sorted`: Sorted x-coordinates
- `vx_sorted`: Sorted x-velocity values
- `dvx_dx`: Velocity gradient array

### Simulation Control

#### `Solve(md)`
Configures and executes transient full-Stokes simulation.

**Parameters:**
- `md`: ISSM model object

**Returns:**
- `md`: Model with transient solution

**Configuration:**
- Stress balance coupling frequency: 1 (every timestep)
- Active physics: stress balance + mass transport
- Inactive: SMB, thermal evolution
- Requested outputs: Vx, Vy, Surface, Base

## Global Configuration

### Bedrock Profiles
```python
BEDROCK_PROFILE_ID = 1   # Selects synthetic bedrock configuration
```

Multiple predefined profiles available with different slope (S) and curvature (K) parameters.

### Experiment Types
- **S1**: No-slip boundary, linear rheology (n=1)
- **S2**: No-slip boundary, nonlinear rheology (n=3)  
- **S3**: Sliding boundary, linear rheology (n=1)
- **S4**: Sliding boundary, nonlinear rheology (n=3)

### Domain Parameters
```python
L_buffer_inlet = 25e3    # Inlet buffer zone (25 km)
L_interest = 160e3       # Region of interest (160 km)  
L_buffer_terminus = 25e3 # Terminus buffer zone (25 km)
L = L_buffer_inlet + L_interest + L_buffer_terminus  # Total domain (210 km)
```

### Material Properties
```python
rho_ice = 910           # Ice density (kg/m³)
ice_temperature = 213.15 # Cold ice temperature (K)
rheology_B = 7.37e13    # Ice rheology parameter (Pa·s^(1/n))
```

### Time Integration
```python
final_time = 1      # Simulation duration (years)
timestep = 1/73     # Time step (≈5 days)
```

## Simulation Workflow

1. **Initialization**
   - Load bedrock configuration
   - Set experiment parameters
   - Define three-part domain geometry (inlet buffer + region of interest + terminus buffer)

2. **Mesh Generation**
   - Create adaptive mesh using BAMG
   - Apply resolution factors for convergence studies
   - Visualize mesh quality

3. **Physics Setup**
   - Configure flow equations (Full-Stokes)
   - Set material properties and rheology
   - Apply boundary conditions
   - Setup friction laws

4. **Diagnostic Solution**
   - Solve steady-state stress balance
   - Analyze velocity fields
   - Check for acceleration issues

5. **Transient Solution**
   - Configure time stepping
   - Solve time-dependent problem
   - Export results in NetCDF format

6. **Post-processing**
   - Format output for ISMIP-HOM
   - Generate inlet velocity profile plots
   - Generate diagnostic plots
   - Analyze driving stress patterns

## Output Files

### Text Output
- **Format**: Tab-delimited ASCII
- **Filename**: `{profile_id}_{experiment}_{resolution}_static.txt`
- **Columns**: x_hat, vx_surface, vz_surface, vx_basal
- **Coordinates**: Normalized to [0,1] range

### NetCDF Output
- **Format**: CF-compliant NetCDF
- **Filename**: `{profile_id}_{experiment}_{resolution}_{final_time=}_yrs_timestep={timestep:.5f}_yrs.nc`
- **Contents**: Full model state and time series

### Visualization
- **Mesh plots**: `profile_{id}_{exp}_{resolution}_mesh.png`
- **Inlet velocity profiles**: `inlet_velocity_profile_{id}_{exp}_{resolution}_{exp}.png`
- **Acceleration diagnostics**: `acceleration_diagnostic_{id}_{exp}.png`
- **Diagnostic plots**: Various analysis figures

## Usage Example

```python
# Configure bedrock profile
BEDROCK_PROFILE_ID = 1
bedrock_config = SyntheticBedrockModelConfig(profile_id=BEDROCK_PROFILE_ID)

# Set experiment type
exp = 'S2'  # No-slip with nonlinear rheology

# Run simulation
python long_flowline.py
```

## Error Handling

The script includes comprehensive error handling for:
- Missing velocity components
- Mesh generation failures
- Solver convergence issues
- Output formatting errors
- File I/O problems

## Performance Considerations

- **Mesh resolution**: Controlled by `resolution_factor` parameter
- **Adaptive refinement**: Based on wavelength-to-thickness ratio  
- **Solver tolerance**: Configurable convergence criteria
- **Time stepping**: Adaptive based on resolution

## Dependencies and Requirements

- **ISSM**: Ice Sheet System Model framework
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization
- **NetCDF**: Data export
- **Custom modules**: Bedrock generation, ISSM utilities

This script provides a complete framework for ice sheet flowline modeling with ISSM, suitable for research applications and numerical experiments.