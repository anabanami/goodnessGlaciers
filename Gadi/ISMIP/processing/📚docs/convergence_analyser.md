# Grid Convergence Analyser Documentation

## Overview

The `convergence_analyser.py` script is a specialized tool for performing grid convergence studies on ISSM (Ice Sheet System Model) transient simulations. It automates the process of analyzing multiple simulation results at different mesh resolutions to assess numerical convergence and determine appropriate grid resolution for ice flow modeling.

## Table of Contents

1. [Theory and Background](#theory-and-background)
2. [Usage](#usage)
3. [Input Requirements](#input-requirements)
4. [Output Products](#output-products)
5. [Functionality Breakdown](#functionality-breakdown)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)
8. [Technical Details](#technical-details)

## Theory and Background

### Grid Convergence Studies

Grid convergence analysis is a fundamental verification technique in computational modeling that ensures numerical solutions are approaching the true solution as mesh resolution increases. The key principle is that as the grid is refined (smaller elements, more nodes), the numerical error should decrease systematically.

### Mathematical Foundation

The script computes **L2 norm errors** between solutions at different resolutions:

- **Relative L2 Error** (when reference norm > 10⁻⁶):
  ```
  L2_relative = ||u_ref - u_coarse|| / ||u_ref|| × 100%
  ```

- **Absolute L2 Error** (when reference norm ≈ 0):
  ```
  L2_absolute = ||u_ref - u_coarse|| (in m/year)
  ```

Where:
- `u_ref` is the velocity field at reference resolution (finest mesh)
- `u_coarse` is the velocity field at coarser resolution
- `||·||` denotes the L2 norm

### Convergence Criteria

A simulation is considered converged when the relative L2 error falls below a specified tolerance (default: 1%). This indicates that further mesh refinement yields negligible improvement in solution accuracy.


### ISSM Environment
The script expects:
- pyISSM installed at `/home/ana/pyISSM/src` (configurable in script)
- Access to ISSM model generation functions (`squaremesh`, `parameterize`)
- Parameter files in the parent directory

## Place the script:
   Put `convergence_analyser.py` in a subdirectory of your simulation outputs:
   ```
   project/
   ├── parameter_file.py
   ├── results/
   │   ├── convergence_analyser.py
   │   └── *.nc files
   ```

## Usage

### Command Line Interface

```bash
python convergence_analyser.py <file1.nc> <file2.nc> ... <fileN.nc>
```

Or using wildcards:
```bash
python convergence_analyser.py IsmipF_S1_*-Transient.nc
```

### File Naming Convention

Input files must follow this strict naming pattern:
```
<ParameterFile>_<Scenario>_<ResolutionFactor>-Transient.nc
```

Examples:
- `IsmipF_S1_1.0-Transient.nc` (reference resolution)
- `IsmipF_S1_0.5-Transient.nc` (coarser mesh)
- `IsmipF_S1_2.0-Transient.nc` (finer mesh)

Where:
- **ParameterFile**: Base name of the parameter script (e.g., `IsmipF`)
- **Scenario**: Scenario identifier (e.g., `S1`, `S2`)
- **ResolutionFactor**: Mesh resolution multiplier (1.0 = reference)

## Input Requirements

### NetCDF File Structure

The script expects ISSM-generated NetCDF files with:
```
results/
├── TransientSolution/
│   ├── time[:]         # Time steps in seconds
│   ├── Vel[time, nodes] # Velocity magnitude array
│   └── ... other variables
```

### Parameter Files

A corresponding parameter file (`<ParameterFile>.py`) must exist in the parent directory containing model configuration parameters.

### Mesh Configuration

The script reconstructs 3D meshes assuming:
- Square domain: 100km × 100km
- Base resolution: 30 × 30 nodes (at resolution_factor=1.0)
- 5 vertical layers with 1:1 extrusion ratio

## Output Products

The script generates three types of outputs:

### 1. Summary Plot (`*_convergence_summary.png`)

A 2×2 subplot figure containing:
- **Top Left**: Surface velocity profiles along centerline
- **Top Right**: Basal velocity profiles along centerline
- **Bottom Left**: L2 error bar chart comparing resolutions
- **Bottom Right**: Maximum velocity evolution over time

### 2. Markdown Report (`*_convergence_report.md`)

A formatted table showing:
- Resolution factors tested
- L2 errors for surface and basal velocities
- Convergence status (✓/✗) based on tolerance

Example output:
```markdown
| Resolution Factor | Surface Vel L2 Error | Basal Vel L2 Error | Overall Status |
|:---:|:---:|:---:|:---:|
| 0.5 | 4.52% | 3.21% | **✗ NOT CONVERGED** |
| 2.0 | 0.87% | 0.62% | **✓ CONVERGED** |
```

### 3. Console Output

Real-time progress updates and error messages during analysis.

## Functionality Breakdown

### Core Components

#### 1. `reconstruct_mesh(filename, resolution_factor)`
Reconstructs the 3D finite element mesh based on resolution factor:
```python
x_nodes = int(30 * resolution_factor)
y_nodes = int(30 * resolution_factor)
```

#### 2. `ConvergenceAnalyzer` Class

Main analysis workflow with methods:

##### `_load_results()`
- Parses filenames using regex pattern
- Loads NetCDF data for final time step
- Extracts centerline velocities (at y=50km)
- Handles multiple resolution factors

##### `_interpolate_to_common_grid()`
- Interpolates all solutions onto reference grid
- Ensures consistent comparison points
- Uses 1D linear interpolation along centerline

##### `_calculate_convergence_metrics()`
- Computes L2 norms for velocity fields
- Intelligently switches between relative/absolute errors
- Stores metrics for each resolution

##### `_create_comparison_plots()`
- Generates comprehensive 4-panel figure
- Visualizes spatial profiles and temporal evolution
- Includes error bars with convergence threshold

##### `_generate_report()`
- Creates markdown-formatted convergence table
- Applies convergence criteria (default: 1% tolerance)
- Provides clear pass/fail assessment

### Centerline Extraction Algorithm

The script uses a robust method to extract centerline data:

1. Identifies all unique y-coordinates in the mesh
2. Finds the y-coordinate closest to domain center (50km)
3. Selects all nodes lying on this centerline
4. Applies to both surface and basal layers

This approach handles irregular meshes and ensures consistent centerline selection across resolutions.

## Examples

### Example 1: Basic Convergence Study

```bash
# Run analysis on three resolutions
python convergence_analyser.py IsmipF_S1_0.5-Transient.nc \
                               IsmipF_S1_1.0-Transient.nc \
                               IsmipF_S1_2.0-Transient.nc
```

Expected output:
```
Found 3 result files to process...
Detected configuration: IsmipF, Scenario: S1
  - Loading IsmipF_S1_0.5-Transient.nc (resolution factor: 0.5)
    Found mesh centerline at y=50000.0m
    Found 15 surface nodes and 15 basal nodes along centerline.
  - Loading IsmipF_S1_1.0-Transient.nc (resolution factor: 1.0)
    Found mesh centerline at y=50000.0m
    Found 30 surface nodes and 30 basal nodes along centerline.
  - Loading IsmipF_S1_2.0-Transient.nc (resolution factor: 2.0)
    Found mesh centerline at y=50000.0m
    Found 60 surface nodes and 60 basal nodes along centerline.

Successfully loaded 3 datasets.
Interpolating results onto reference grid (res=1.0)...
Calculating Convergence Metrics (vs. res=1.0)...
  res=0.5: Surface Vel L2 (relative)=4.523, Basal Vel L2 (relative)=3.214
  res=2.0: Surface Vel L2 (relative)=0.872, Basal Vel L2 (relative)=0.621
Creating summary plot...
  Saved plot: IsmipF_S1_convergence_summary.png
Generating analysis report...
  Saved report: IsmipF_S1_convergence_report.md

Analysis complete.
```

### Example 2: Using Wildcards

```bash
# Analyze all transient files in directory
python convergence_analyser.py *-Transient.nc
```

### Example 3: Custom Tolerance

To modify convergence tolerance, edit line in `_generate_report()`:
```python
def _generate_report(self, tolerance=0.5):  # 0.5% instead of 1%
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "Parameter file not found"
**Problem**: Script cannot locate the parameter file.
**Solution**: Ensure parameter file exists in parent directory with correct naming.

#### 2. "Could not find any centerline nodes"
**Problem**: Mesh doesn't have nodes at expected centerline location.
**Solution**: Check mesh generation parameters or adjust centerline detection tolerance.

#### 3. "Reference solution (res=1.0) not found"
**Problem**: No file with resolution_factor=1.0 provided.
**Solution**: Include reference resolution file in input list.

#### 4. Import errors for pyISSM
**Problem**: ISSM Python modules not found.
**Solution**: Update the path in line 21 to your pyISSM installation.

#### 5. Empty interpolation arrays
**Problem**: Centerline extraction failed for some resolutions.
**Solution**: Check mesh consistency across resolutions; verify domain dimensions.

### Debug Mode

To enable verbose debugging, add print statements in key methods:
```python
print(f"Debug: x_surf shape = {data['x_surf'].shape}")
print(f"Debug: vel_surf range = [{np.min(vel_surf)}, {np.max(vel_surf)}]")
```

## Technical Details

### Performance Considerations

- **Memory Usage**: Scales with mesh size and time steps
- **Computation Time**: O(n log n) for sorting, O(n) for interpolation
- **File I/O**: NetCDF4 lazy loading minimizes memory footprint

### Numerical Precision

- Centerline tolerance: 1e-6 meters
- Near-zero threshold: 1e-6 for norm calculations
- Interpolation: Linear (1st order accurate)

### Assumptions and Limitations

1. **Structured Mesh**: Assumes regular quadrilateral elements
2. **Domain Geometry**: Fixed 100km × 100km square domain
3. **Centerline Location**: Fixed at y=50km (domain center)
4. **Time Step**: Uses only final time step for spatial analysis
5. **2D Interpolation**: Performs 1D interpolation along centerline only

### Extending the Script

To adapt for different use cases:

#### Custom Domain Size
```python
# In reconstruct_mesh()
x_max, y_max = 200000, 150000  # 200km × 150km
```

#### Different Base Resolution
```python
# In reconstruct_mesh()
x_nodes = int(50 * resolution_factor)  # Base: 50×50 instead of 30×30
```

#### Multiple Centerlines
```python
# Add multiple y-coordinates for analysis
centerlines = [25000, 50000, 75000]  # Quarter points
for y_center in centerlines:
    # Extract and analyze each centerline
```

#### Alternative Error Metrics
```python
# Add L-infinity norm
l_inf_error = np.max(np.abs(ref_data - comp_data))
```

## References

- ISSM Documentation: [https://issm.jpl.nasa.gov/](https://issm.jpl.nasa.gov/)
- Grid Convergence Theory: Roache, P.J. (1998). *Verification and Validation in Computational Science and Engineering*
- L2 Norm: Burden, R.L. & Faires, J.D. (2010). *Numerical Analysis*



