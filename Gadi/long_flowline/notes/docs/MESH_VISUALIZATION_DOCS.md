# Mesh Visualization on Full Bedrock Domain

## Overview

The `plot_mesh_on_full_bedrock.py` script provides a comprehensive visualization that shows how your ice sheet simulation mesh relates to the complete bedrock topography. This is essential for understanding the spatial coverage and context of your simulation within the broader bedrock domain.

## Purpose

This script addresses the key question: **"How much of the available bedrock topography does my simulation actually cover?"**

By overlaying the finite element mesh on the full bedrock domain, you can:
- Visualize the spatial extent of your simulation
- Understand how the domain optimization affects coverage
- See the relationship between bedrock wavelength and simulation domain
- Validate that your mesh captures the intended number of bedrock periods

## Usage

### Basic Usage
```bash
python plot_mesh_on_full_bedrock.py
```
Uses default parameters:
- Profile ID: 165
- Resolution factor: 1.25
- Full bedrock extent: 175 km

### Custom Parameters
```bash
python plot_mesh_on_full_bedrock.py <profile_id> <resolution_factor> <max_extent_km>
```

**Examples:**
```bash
# Profile 165, resolution 1.25, 175km extent
python plot_mesh_on_full_bedrock.py 165 1.25 175

# Profile 270, finer resolution, 200km extent
python plot_mesh_on_full_bedrock.py 270 0.5 200

# Profile 639, coarser resolution, 150km extent
python plot_mesh_on_full_bedrock.py 639 1.5 150
```

## Script Functionality

### 1. Domain Generation
- **Full Bedrock Domain**: Generates high-resolution bedrock topography up to specified extent (default: 175km)
- **Simulation Domain**: Recreates the exact domain used in `flowline.py` with optimal length for integer periods
- **Domain Optimization**: Uses the same `find_optimal_domain_length()` function as the simulation

### 2. Mesh Generation
- **Identical Parameters**: Uses exact same parameters as `flowline.py`:
  - Target domain length: 135km (with optimization)
  - Adaptive mesh refinement based on bedrock wavelength
  - Same resolution factors and bamgflowband settings
- **Exact Reconstruction**: Generates the same finite element mesh that would be used in simulation

### 3. Visualization Elements

The output plot includes:

#### Background
- **Full Bedrock Profile** (black line): Complete bedrock topography up to maximum extent
- **Grid and Axis Labels**: Professional formatting with distance in km

#### Simulation Domain
- **Simulation Bedrock** (red line): The actual bedrock profile used in mesh generation
- **Ice Surface** (blue line): Surface elevation with ice thickness
- **Ice Thickness** (light blue fill): Visual representation of ice volume

#### Mesh Overlay
- **Finite Element Mesh** (green lines): The actual triangular mesh elements
- **Domain Boundaries** (purple dashed lines): Start (0km) and end points of simulation
- **Wavelength Markers** (orange dotted lines): Marks each complete bedrock period

#### Information Panel
- **Statistics Box**: Shows key metrics including:
  - Simulation vs. full domain lengths
  - Percentage coverage
  - Number of mesh vertices
  - Resolution factor used

## Output Information

### Console Output
```
=== PLOTTING MESH ON FULL BEDROCK ===
Profile ID: 165
Resolution factor: 1.25
Max bedrock extent: 175 km

Generating simulation mesh for profile 165...
Simulation domain: 133.056 km (1330 points)
Mesh parameters: wavelength=6.3km, hmax=158.4m
Generated mesh: 26962 vertices, 51240 elements

Generating full bedrock domain up to 175 km...

=== SUMMARY ===
Full bedrock domain: 0 → 175 km
Simulation domain: 0 → 133.056 km
Coverage: 76.0% of full bedrock
Bedrock wavelength: 6.3 km
Periods in simulation: 21.000
Mesh vertices: 26,962
```

### Generated File
- **Filename Format**: `mesh_on_full_bedrock_profile_{ID:03d}_res_{factor}.png`
- **Example**: `mesh_on_full_bedrock_profile_165_res_1.25.png`
- **Resolution**: 300 DPI for publication quality

## Technical Details

### Dependencies
```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from bedrock_generator import SyntheticBedrockModelConfig
from domain_utils import find_optimal_domain_length
from bamgflowband import bamgflowband
```

### Key Functions

#### `get_full_bedrock_domain(bedrock_config, max_extent_km=175)`
- Generates high-resolution bedrock profile for full domain
- Uses 100m spacing for smooth visualization
- Returns coordinates in km for plotting

#### `generate_simulation_mesh(profile_id, resolution_factor=1.25)`
- Recreates exact mesh from `flowline.py`
- Uses identical domain optimization and mesh parameters
- Returns mesh object and associated geometric data

#### `plot_mesh_on_full_bedrock(...)`
- Main plotting function that combines all elements
- Handles coordinate transformations and scaling
- Generates professional publication-ready figures

## Integration with Workflow

### Before Simulation
Use this script to:
- **Validate domain coverage** before running expensive simulations
- **Understand spatial context** of your chosen parameters
- **Check period optimization** is working correctly

### After Simulation
Use this script to:
- **Document simulation setup** for papers/reports
- **Compare different resolution factors** visually
- **Understand mesh density** in relation to bedrock features

### With Processing Scripts
This script complements:
- `extract_results.py`: Shows the mesh that results are plotted on
- `transfer_analysis.py`: Provides context for bed-to-surface transfer
- `phase_analysis.py`: Shows spatial domain for phase calculations

## Interpretation Guide

### Coverage Analysis
- **High Coverage (>80%)**: Good representation of bedrock variability
- **Medium Coverage (60-80%)**: Adequate for most studies
- **Low Coverage (<60%)**: May miss important bedrock features

### Mesh Quality Indicators
- **Mesh Density**: More triangles in areas of high curvature
- **Boundary Conditions**: Mesh should end near bedrock peaks
- **Wavelength Coverage**: Should include integer number of periods

### Domain Optimization Success
- **Clean Period Boundaries**: Domain should end at natural wavelength boundaries
- **Consistent Coverage**: Similar patterns across different profiles
- **Efficient Use**: Not excessive domain for computational resources

## Troubleshooting

### Common Issues

#### Import Errors
```bash
ModuleNotFoundError: No module named 'bamgflowband'
```
**Solution**: Ensure ISSM is properly installed and Python path includes ISSM modules

#### Memory Issues
```bash
MemoryError: Unable to allocate array
```
**Solution**: Reduce `max_extent_km` or increase system memory

#### Mesh Generation Fails
```bash
Error in bamgflowband: mesh generation failed
```
**Solution**: Check bedrock profile validity and adjust resolution factor

### Performance Tips
- Use smaller `max_extent_km` for faster generation
- Increase resolution factor for coarser (faster) meshes
- Run in background for multiple profiles: `nohup python script.py &`

## Example Applications

### Research Use Cases
1. **Method Validation**: Show mesh covers sufficient bedrock periods
2. **Parameter Studies**: Compare different resolution factors
3. **Domain Sensitivity**: Assess impact of domain length choices
4. **Publication Figures**: Professional visualization for papers

### Workflow Integration
```bash
# Generate mesh visualization for all test profiles
for profile in 165 270 639; do
    python plot_mesh_on_full_bedrock.py $profile 1.25 175
done

# Compare different resolutions for same profile
for res in 0.5 1.0 1.25 1.5; do
    python plot_mesh_on_full_bedrock.py 165 $res 175
done
```

## Related Documentation
- `flowline.py`: Main simulation script
- `domain_utils.py`: Domain optimization functions
- `extract_results.py`: Results visualization
- `ISMIP-HOM` specifications: Benchmark requirements