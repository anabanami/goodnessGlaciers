import numpy as np
import rasterio
from pyproj import Transformer
from scipy.ndimage import map_coordinates
import os


class REMACache:
    """Cache for REMA DEM data to avoid repeated file I/O."""

    def __init__(self):
        self._data = None
        self._transform = None
        self._inv_transform = None
        self._path = None

    def load(self, dem_path):
        """Load DEM data if not already cached for this path."""
        if self._path == dem_path and self._data is not None:
            return  # Already loaded

        with rasterio.open(dem_path) as src:
            self._data = src.read(1)
            self._data = np.where(self._data == src.nodata, np.nan, self._data)
            self._transform = src.transform
            self._inv_transform = ~src.transform
            self._path = dem_path

    def sample(self, x_coords, y_coords):
        """Sample elevations at given coordinates using bilinear interpolation."""
        rows, cols = self._inv_transform * (x_coords, y_coords)
        return map_coordinates(self._data, [rows, cols], order=1, mode='nearest')

    def clear(self):
        """Clear the cache to free memory."""
        self._data = None
        self._transform = None
        self._inv_transform = None
        self._path = None


# Global cache instance
_rema_cache = REMACache()


def get_rema_cache():
    """Get the global REMA cache instance."""
    return _rema_cache


def extract_rema_elevation(x_coords, y_coords, dem_path, cache=None):
    """
    Extracts REMA elevation using Bilinear Interpolation (order=1).
    This matches the accuracy of JR's Fortran bilinear method.

    Args:
        x_coords, y_coords: Coordinate arrays
        dem_path: Path to DEM file
        cache: Optional REMACache instance. If None, uses global cache.
    """
    if cache is None:
        cache = _rema_cache
    cache.load(dem_path)
    return cache.sample(x_coords, y_coords)


def calculate_ice_thickness(surface_elevs, bedrock_elevs):
    ice_thickness = surface_elevs - bedrock_elevs
    # set negative values to NaN
    ice_thickness[ice_thickness < 0] = np.nan
    return ice_thickness


def extract_rema_flow_vector(x, y, dem_path, ice_thickness, cache=None):
    """
    Estimates the regional ice flow vector (-dS/dx, -dS/dy) from REMA.

    Args:
        x, y: Coordinates arrays (EPSG:3031)
        dem_path: Path to the REMA mosaic
        ice_thickness: The baseline for the gradient calculation (meters).
                       McCormack et al. (2019) recommend ~10x ice thickness.
        cache: Optional REMACache instance. If None, uses global cache.
    Returns:
        flow_x, flow_y: Normalized vector components of the flow direction.
    """
    if cache is None:
        cache = _rema_cache
    cache.load(dem_path)

    delta = ice_thickness * 5

    # Create stencil coordinates (Central Difference)
    x_left, x_right = x - delta, x + delta
    y_down, y_up    = y - delta, y + delta

    # Sample all 4 neighbors using cached data (no file I/O)
    z_left  = cache.sample(x_left, y)
    z_right = cache.sample(x_right, y)
    z_down  = cache.sample(x, y_down)
    z_up    = cache.sample(x, y_up)

    # Calculate Gradient (Slope)
    # dz/dx = (z_right - z_left) / (2*delta)
    # Flow drives 'downhill', so Flow = -Gradient
    dx_slope = (z_right - z_left) / (2 * delta)
    dy_slope = (z_up - z_down) / (2 * delta)

    flow_x = -dx_slope
    flow_y = -dy_slope

    # Normalize vectors (we only care about direction, not speed)
    magnitude = np.sqrt(flow_x**2 + flow_y**2)

    # Handle flat areas (avoid divide by zero)
    magnitude[magnitude == 0] = 1.0

    return flow_x / magnitude, flow_y / magnitude