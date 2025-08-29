# Bedrock Profile Generator Documentation

## Overview

The Bedrock Profile Generator is a Python toolkit for creating synthetic 1D bedrock profiles for ice sheet modeling applications. It generates periodic bedrock topographies with configurable undulations, extensions, and various statistical properties suitable for use with ice flow models like ISSM.

## Files

- `bedrock_generator.py` - Main generator class and functionality
- `bedrock_settings.py` - Configuration parameters and ranges
- `bedrock_generator_documentation.md` - This documentation file

## Key Features

- **Three-region structure**: Pre-extension (flat) → Undulated domain → Post-extension (flat)
- **Periodic boundary compatibility**: Pre-extension starts at peak amplitude height
- **Configurable undulations**: Amplitude, wavelength, skewness, kurtosis, and noise
- **Smooth transitions**: Automated smoothing between regions
- **Batch generation**: Create multiple profiles with systematic parameter combinations
- **Visualization**: Automatic plotting with region boundaries marked
- **Profile management**: Save/load profiles with metadata

## Domain Structure

The generated profiles have a three-region structure optimized for periodic boundary conditions:

```
[25km flat pre-extension] → [160km undulated domain] → [25km flat post-extension]
```

### Region Details

1. **Pre-extension (25km)**:
   - Flat slope at peak amplitude height
   - Elevation: `base_slope × x + amplitude`
   - Ensures smooth connection to first undulation peak

2. **Undulated Domain (160km)**:
   - Cosine-based undulations with configurable properties
   - Optional skewness, kurtosis, and noise transformations
   - Undulation pattern: `amplitude × cos(2π × x / wavelength)`

3. **Post-extension (25km)**:
   - Flat slope continuation
   - Smooth transition from last undulation
   - Elevation follows base slope trend

## Configuration Parameters

### Domain Settings (`bedrock_settings.py`)

```python
DOMAIN_LENGTH = 160e3      # Length of undulated region (m)
RESOLUTION = 0.1e3         # Spatial resolution (m)
ICE_THICKNESS = 1.92e3     # Reference ice thickness (m)
BASE_SLOPE = -np.deg2rad(0.1)  # Base bed slope (radians)
```

### Parameter Ranges

The script generates profiles across these parameter ranges:

```python
PARAM_RANGES = {
    'amplitude': np.linspace(0.01, 0.020, 7) * ICE_THICKNESS,    # 19.2-38.4m
    'wavelength': np.array([2.0, 3.3, 5.0, 8.0, 10.0]) * ICE_THICKNESS,  # 3.84-19.2km
    'skewness': np.linspace(-0.2, 0.2, 5),                      # Asymmetry
    'kurtosis': np.linspace(-0.2, 0.2, 5)                       # Peak sharpness
}
```

## Usage

### Basic Profile Generation

```python
from bedrock_generator import BedrockGenerator

# Initialize generator
generator = BedrockGenerator()

# Generate single profile
x, bed = generator.generate_bedrock_profile(
    amplitude=38.4,      # meters
    wavelength=6336,     # meters (3.3 × ice thickness)
    skewness=0.0,
    kurtosis=0.0,
    noise_level=0.0,
    initial_elevation=1.0
)
```

### Batch Profile Generation

```python
# Generate complete suite of profiles
generator.generate_all_profiles(n_profiles=875)
```

This creates:
- `bedrock_profiles/` directory
- Individual `.npz` files for each profile
- Corresponding `.png` visualization plots
- `bedrock_metadata.txt` with all parameters

### Loading Existing Profiles

```python
from bedrock_generator import SyntheticBedrockModelConfig

# Load specific profile
config = SyntheticBedrockModelConfig(profile_id=42)

# Access profile data
x = config.x_profile
bed = config.bed_profile
params = config.profile_params
```

## Profile Types

### Profile 001 - Reference Flat Profile
- Zero amplitude undulations
- Flat slope throughout all regions  
- Used as baseline/control case

### Standard Profiles (002+)
- Various amplitude/wavelength combinations
- Systematic coverage of parameter space
- Optional skewness, kurtosis, and noise variations

## Key Methods

### BedrockGenerator Class

#### `generate_bedrock_profile()`
Main method for creating individual profiles.

**Parameters:**
- `amplitude` (float): Undulation amplitude in meters
- `wavelength` (float): Undulation wavelength in meters  
- `skewness` (float): Asymmetry parameter (-0.2 to 0.2)
- `kurtosis` (float): Peak sharpness parameter (-0.2 to 0.2)
- `noise_level` (float): White noise level (0.0 to 0.2)
- `initial_elevation` (float): Base elevation in meters
- `pre_extension_length` (float): Pre-extension length (default: 25km)
- `post_extension_length` (float): Post-extension length (default: 25km)
- `profile_id` (int): Profile identifier for special cases
- `smoothing_length` (float): Smoothing zone length (default: 2km)

**Returns:**
- `x` (array): x-coordinates in meters
- `bed` (array): Bedrock elevations in meters

#### `generate_all_profiles(n_profiles=100)`
Generates systematic parameter combinations.

#### `save_profile(x, bed, params, profile_id)`  
Saves profile data and visualization.

### Transformation Methods

#### `apply_skewness(bed, skewness)`
Applies asymmetric transformation to undulations.

#### `apply_kurtosis(bed, kurtosis)`
Adjusts peak sharpness and distribution shape.

#### `add_noise(bed, noise_level)`
Adds white noise to bedrock profile.

#### `smoothing()`
Handles smooth transitions between regions.

## Output Files

### Profile Data (`.npz` files)
Contains:
- `x`: x-coordinates 
- `bed`: bedrock elevations
- `amplitude`, `wavelength`, `skewness`, `kurtosis`, `noise_level`, `initial_elevation`: Parameters

### Visualization (`.png` files)
Shows:
- Bedrock profile (blue line)
- Base slope reference (dashed black line)
- Region boundaries (vertical dashed lines)
- Parameter information in title

### Metadata (`bedrock_metadata.txt`)
Lists all profiles with their parameters.

## Integration with Ice Flow Models

### ISSM Integration
Use `SyntheticBedrockModelConfig` class:

```python
config = SyntheticBedrockModelConfig(profile_id=123)
bed_elevation = config.get_bedrock_elevation(x_coordinates)
```

### Periodic Boundary Conditions
The three-region structure ensures:
- Smooth periodic continuation
- Consistent elevation at boundaries
- Proper derivative matching

## Advanced Features

### Custom Parameter Combinations
Modify `create_parameter_combinations()` method to specify custom parameter sets.

### Special Profile Handling
Profile 001 receives special treatment as a flat reference case.

### Peak Detection and Smoothing
Automatic detection of undulation peaks for optimal smoothing transitions.

## Mathematical Details

### Base Elevation
```
base_bed(x) = initial_elevation + base_slope × x
```

### Undulated Region
```
bed(x) = base_bed(x) + amplitude × cos(2π × (x - undulated_start) / wavelength)
```

### Pre-extension
```
bed(x) = base_bed(x) + amplitude  [constant peak height]
```

### Transformations
- **Skewness**: Power transformation for asymmetry
- **Kurtosis**: Peak sharpness adjustment
- **Noise**: Gaussian white noise addition

## Example Workflow

1. **Configure parameters** in `bedrock_settings.py`
2. **Generate profiles**: `python bedrock_generator.py`
3. **Review outputs** in `bedrock_profiles/` directory
4. **Select profiles** based on visualization and metadata
5. **Load into model** using `SyntheticBedrockModelConfig`

## Tips and Best Practices

- **Wavelength scaling**: Use multiples of ice thickness (Budd optimal: 3.3×)
- **Amplitude limits**: Keep realistic relative to ice thickness
- **Resolution**: Match your ice flow model requirements
- **Profile 001**: Always include as flat reference case
- **Parameter space**: Use systematic sampling for comprehensive coverage
- **Visualization**: Always check plots before using in models

## Dependencies

- `numpy`: Array operations and mathematical functions
- `matplotlib.pyplot`: Plotting and visualization  
- `os`: File system operations
- `datetime`: Timestamp generation

## File Naming Convention

- Profiles: `bedrock_profile_XXX.npz` (XXX = zero-padded profile ID)
- Plots: `bedrock_profile_XXX.png`
- Metadata: `bedrock_metadata.txt`