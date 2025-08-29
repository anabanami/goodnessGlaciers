# Grid Convergence Analysis Tool Documentation

## Overview

`analyse_grid_convergence.py` is a Python script designed to analyze grid convergence for computational fluid dynamics simulations, specifically for ice flow modeling. The tool automatically detects available datasets, loads results from different mesh resolutions, and performs convergence analysis to determine if the numerical solution is grid-independent.

## Class: GridConvergenceAnalyzer

### Purpose
The `GridConvergenceAnalyzer` class provides a comprehensive framework for assessing whether simulation results have converged with respect to grid resolution. It compares solutions obtained using different mesh densities and quantifies the differences using both relative and absolute error metrics.

### Key Features

- **Automatic Dataset Detection**: Scans for result files matching the pattern `*_*_*_static.txt`
- **Multi-Resolution Analysis**: Compares solutions across different mesh resolutions
- **Convergence Assessment**: Uses configurable tolerance thresholds to determine convergence
- **Automated Reporting**: Generates detailed markdown reports and visualization plots
- **Error Handling**: Robust error handling for missing files and invalid data

## Usage

### Basic Usage
```python
from analyse_grid_convergence import GridConvergenceAnalyzer

# Auto-configure and run analysis
analyzer = GridConvergenceAnalyzer()
resolution_factors = analyzer.auto_configure()
analyzer.load_static_results(resolution_factors)
analyzer.calculate_convergence_metrics()
converged = analyzer.assess_convergence()
analyzer.generate_report(converged)
analyzer.create_comparison_plots()
```

### Command Line Usage
```bash
python analyse_grid_convergence.py
```

## Input Data Format

The tool expects static result files with the naming convention:
```
{profile_id:03d}_{experiment}_{resolution_factor}_static.txt
```

Each file should contain space-separated columns:
1. `x_hat`: Normalized x-coordinate (x/L)
2. `vx_surface`: Surface velocity in x-direction (m/a)
3. `vy_surface`: Surface velocity in y-direction (m/a) 
4. `vx_basal`: Basal velocity in x-direction (m/a)

## Configuration Parameters

### Tolerance Settings
- **Relative Tolerance**: 1% (0.01) - Used for non-zero velocity solutions
- **Absolute Tolerance**: 0.01 m/a - Used for near-zero velocity solutions
- **Near-Zero Threshold**: 0.1 m/a - Boundary between relative/absolute error metrics

## Core Methods

### `detect_available_datasets()`
- Scans current directory for static result files
- Parses filename components to extract profile ID, experiment, and resolution factor
- Returns dictionary of available datasets grouped by profile and experiment

### `load_static_results(resolution_factors)`
- Loads numerical results from static output files
- Sorts data by x-coordinate for consistent interpolation
- Stores results in internal dictionary structure

### `calculate_convergence_metrics()`
- Uses finest resolution as reference solution
- Interpolates coarser solutions onto reference grid
- Calculates L2 relative error for non-zero solutions
- Calculates RMSE for near-zero solutions

### `assess_convergence()`
- Compares error metrics against tolerance thresholds
- Determines convergence status for each resolution
- Returns overall convergence boolean

### `generate_report(converged)`
- Creates detailed markdown report with:
  - Analysis metadata and parameters
  - Key solution metrics
  - Convergence results table
  - Recommendations

### `create_comparison_plots()`
- Generates 4-panel visualization:
  - Surface velocity comparison across resolutions
  - Basal velocity comparison across resolutions  
  - Convergence metrics bar chart
  - Computational scaling (mesh complexity)

## Output Files

### Report File
- **Filename**: `{profile_id:03d}_{experiment}_convergence_report.md`
- **Content**: Comprehensive analysis report in markdown format

### Plot File  
- **Filename**: `{profile_id:03d}_{experiment}_convergence_analysis.png`
- **Content**: Multi-panel comparison plots

## Error Handling

The tool includes robust error handling for common issues:
- Missing or malformed input files
- Invalid filename formats
- Data loading errors
- Numerical computation issues

## Dependencies

- `numpy`: Numerical computations and array operations
- `matplotlib.pyplot`: Plotting and visualization
- `os`: File system operations
- `glob`: File pattern matching
- `datetime`: Timestamp generation

## Example Output

When run successfully, the tool provides console output showing:
- Auto-detected datasets
- Loading progress for each resolution
- Convergence metrics calculations
- Overall convergence assessment
- File generation confirmation

The analysis concludes with a clear indication of whether the solution has converged within the specified tolerances.