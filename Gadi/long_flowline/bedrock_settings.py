import numpy as np

DOMAIN_LENGTH = 160e3  # m
RESOLUTION = 0.1e3 # m
ICE_THICKNESS = 1.92e3 # m
BASE_SLOPE = -np.deg2rad(0.1) # tan(0.1°) × 100 = 0.175%

# Parameter ranges for bedrock generation - can adjust as needed
PARAM_RANGES = {
    # Amplitude as fraction of ice thickness
    'amplitude': np.linspace(0.01, 0.020, 7) * ICE_THICKNESS,
    # Wavelength as multiple of ice thickness (Budd found ~3.3Z is optimal)
    'wavelength': np.array([2.0, 3.3, 5.0, 8.0, 10.0]) * ICE_THICKNESS,
    # Skewness parameter
    'skewness': np.linspace(-0.2, 0.2, 5),
    # Kurtosis parameter
    'kurtosis': np.linspace(-0.2, 0.2, 5)
}
