# Plotting Scripts Documentation

This document describes the velocity and shear stress plotting scripts for analyzing ice flow simulation results.

## Overview

The plotting scripts process text files containing velocity data from ice flow simulations and generate visualizations showing:
- Surface and basal velocities
- Velocity variations over bed topography
- Shear stress distributions

## Script Types

### Single File Scripts

#### `velocity_plots.py`

**Purpose:** Creates velocity plots for a single simulation result file.

**Usage:**
```bash
python velocity_plots.py <profile_file>
```

**Example:**
```bash
python velocity_plots.py 027_S4_0.5_static.txt
```

**Outputs:**
1. **Basic velocity plot:** `{profile_id}_{experiment}_velocity_non_periodic.png`
   - Line plot showing surface velocity vs basal velocity
   - X-axis: Normalized distance (0-1)
   - Y-axis: Velocity (m a⁻¹)

2. **Velocity over topography:** `{profile_id}_{experiment}_velocity_color_non_periodic.png`
   - Scatter plot of bed elevation colored by basal velocity oscillations
   - Uses Savitzky-Golay filter to isolate velocity oscillations from trend
   - Colormap: 'viridis'

**Key Features:**
- Sorts data by x-coordinate to ensure proper plotting order
- Calculates velocity oscillations by removing smooth trend
- Integrates with bedrock profile configuration

---

#### `shear_stress_plots.py`

**Purpose:** Calculates and visualizes basal shear stress for a single simulation.

**Usage:**
```bash
python shear_stress_plots.py <profile_file>
```

**Example:**
```bash
python shear_stress_plots.py 027_S4_0.5_static.txt
```

**Output:**
- **Shear stress over topography:** `{profile_id}_{experiment}_internal_SS_at_z=0_color_non_periodic.png`
  - Scatter plot of bed elevation colored by shear stress oscillations
  - Uses Budd's basal shear stress formulation
  - Colormap: 'coolwarm' (diverging)

**Shear Stress Calculation:**
- Uses Budd's formulation: `τ_xz = 2η V̄ ω β₁ cos(ωx)`
- Material properties from Pattyn 2008
- Parameters:
  - `A = 1×10⁻¹⁶ s⁻¹ Pa⁻³` (flow law parameter)
  - `n = 3` (rheology exponent)
  - `η = B/2` (effective viscosity)

---

### Batch Processing Scripts

#### `batch_plots.py` ⭐ *Recommended*

**Purpose:** Combines functionality of both single scripts, processes all `.txt` files automatically.

**Usage:**
```bash
python batch_plots.py [OPTIONS]
```

**Options:**
- `--parallel`: Use multiprocessing for faster processing
- `--skip-existing`: Skip files with existing output plots
- `--max-workers N`: Limit parallel workers (default: CPU count)

**Examples:**
```bash
# Process all files sequentially
python batch_plots.py

# Use parallel processing
python batch_plots.py --parallel

# Skip existing outputs and use 4 workers
python batch_plots.py --parallel --skip-existing --max-workers 4
```

**Outputs for each file:**
1. Basic velocity plot
2. Velocity over topography (colored by oscillations)
3. Shear stress over topography (colored by oscillations)

**Features:**
- **Automatic file discovery:** Finds all valid `.txt` files
- **Parallel processing:** Significantly faster for large datasets
- **Progress tracking:** Real-time status updates
- **Error handling:** Continues processing if individual files fail
- **Skip existing:** Avoids regenerating existing plots
- **Comprehensive summary:** Reports success/failure statistics

---

## File Format Requirements

All scripts expect input files with the following format:

```
# Optional header lines starting with #
x_normalized  vx_surface  vz_surface  vx_basal
0.000         100.5       2.1         95.2
0.001         101.2       2.0         96.1
...
```

**Columns:**
1. `x_normalized`: Normalized x-coordinate (0-1)
2. `vx_surface`: Surface velocity in x-direction (m/yr)
3. `vz_surface`: Surface velocity in z-direction (m/yr) 
4. `vx_basal`: Basal velocity in x-direction (m/yr)

**Filename Convention:**
- Format: `{profile_id}_{experiment}_{additional_info}.txt`
- Example: `027_S4_0.5_static.txt`
  - Profile ID: 027
  - Experiment: S4
  - Additional: 0.5_static

---

## Dependencies

### Required Python Packages
```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.signal import savgol_filter
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor  # batch_plots.py only
```

### Custom Modules
- `bedrock_generator.SyntheticBedrockModelConfig`: Provides bedrock profile parameters
- Requires bedrock profile data in `../bedrock_profiles/` directory

---

## Configuration Parameters

### Physical Constants
- `L = 160e3` m: Domain length
- `YTS = 31556926` s: Seconds per year

### Material Properties (Shear Stress)
- `A = 1e-16 / YTS` Pa⁻³ s⁻¹: Flow law parameter
- `n = 3`: Rheology exponent  
- `B = A^(-1/n)`: Flow law constant
- `η = B/2`: Effective viscosity

### Plotting Parameters
- **Figure size:** 16×5 inches
- **DPI:** 300 (high quality)
- **Smoothing:** Savitzky-Golay filter (window=101, polynomial order=3)
- **Grid:** Dotted style, 40% transparency

---

## Output File Naming

### Velocity Plots
- Basic: `{profile_id:03d}_{experiment}_velocity_non_periodic.png`
- Colored: `{profile_id:03d}_{experiment}_velocity_color_non_periodic.png`

### Shear Stress Plots
- `{profile_id:03d}_{experiment}_internal_SS_at_z=0_color_non_periodic.png`

**Examples:**
- `027_S4_velocity_non_periodic.png`
- `027_S4_velocity_color_non_periodic.png`
- `027_S4_internal_SS_at_z=0_color_non_periodic.png`

---

## Performance Considerations

### Single File Scripts
- **Processing time:** ~2-5 seconds per file
- **Memory usage:** Low (processes one file at a time)
- **Best for:** Individual file analysis, debugging

### Batch Script
- **Sequential mode:** Same as single scripts, but automated
- **Parallel mode:** 3-5× faster with 4+ cores
- **Memory usage:** Higher (multiple processes)
- **Best for:** Large datasets, production runs

### Optimization Tips
1. Use `--parallel` for >5 files
2. Use `--skip-existing` for incremental processing
3. Limit `--max-workers` if memory constrained
4. Process files in batches if dealing with hundreds of files

---

## Error Handling

### Common Issues
1. **File format errors:** Invalid column count or non-numeric data
2. **Missing bedrock profiles:** Profile ID not found in bedrock configuration
3. **Plotting errors:** Memory issues with large datasets

### Error Recovery
- **Single scripts:** Stop on first error
- **Batch script:** Continues processing, reports errors in summary

### Troubleshooting
```bash
# Test single file first
python velocity_plots.py problem_file.txt

# Check file format
head -10 problem_file.txt

# Verify bedrock profile exists
ls ../bedrock_profiles/profile_XXX_*
```

---

## Integration with Other Tools

### Workflow Integration
```bash
# Complete processing pipeline
python batch_convert.py --parallel          # Convert .outbin to .nc
python batch_extract_results.py --parallel  # Extract results to .txt
python batch_plots.py --parallel           # Generate plots
```

### Data Sources
- Input files typically generated by `extract_results.py`
- Compatible with ISSM simulation outputs
- Supports various experiment types (S1, S2, S3, S4)

---

## Best Practices

### File Organization
```
processing/
├── *.txt                    # Input data files
├── *_velocity_*.png         # Velocity plots
├── *_internal_SS_*.png      # Shear stress plots
├── batch_plots.py           # Batch processing script
├── velocity_plots.py        # Single file velocity
└── shear_stress_plots.py    # Single file shear stress
```

### Batch Processing Workflow
1. **Preview:** Use single scripts to test on representative files
2. **Batch run:** Use `batch_plots.py --parallel --skip-existing`
3. **Quality check:** Review error summary and spot-check outputs
4. **Iterate:** Re-run failed files individually if needed

### Performance Monitoring
- Monitor CPU usage during parallel processing
- Check available disk space before large batch runs
- Use `time python batch_plots.py` to measure performance

---

## Version History

- **v1.0:** Individual `velocity_plots.py` and `shear_stress_plots.py`
- **v2.0:** Added `batch_plots.py` with parallel processing
- **Current:** Integrated error handling and progress reporting