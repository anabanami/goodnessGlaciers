import numpy as np
import rasterio
from pyproj import Transformer
from scipy.ndimage import map_coordinates
import os


def extract_rema_elevation(x_coords, y_coords, dem_path):
    """
    Extracts REMA elevation using Bilinear Interpolation (order=1).
    This matches the accuracy of JR's Fortran bilinear method.
    """
    with rasterio.open(dem_path) as src:
        # Read the first band (elevation)
        data = src.read(1)
        
        # Replace nodata with NaN to prevent it from contaminating the average
        data = np.where(data == src.nodata, np.nan, data)

        # Convert World Coords (meters) -> Image Indices (row, col)
        # ~src.transform is the inverse transform (World -> Image)
        rows, cols = ~src.transform * (x_coords, y_coords)

        # Interpolate:
        # order=1 is Bilinear (linear in x, linear in y)
        # mode='nearest' handles edges gracefully
        elevations = map_coordinates(data, [rows, cols], order=1, mode='nearest')

    return elevations


def calculate_ice_thickness(surface_elevs, bedrock_elevs):
	ice_thickness = surface_elevs - bedrock_elevs
	# set negative values to NaN 
	ice_thickness[ice_thickness < 0] = np.nan

	return ice_thickness

    
def extract_rema_flow_vector(x, y, dem_path, ice_thickness):
    """
    Estimates the regional ice flow vector (-dS/dx, -dS/dy) from REMA.
    
    Args:
        x, y: Coordinates arrays (EPSG:3031)
        dem_path: Path to the REMA mosaic
        ice thickness: The baseline for the gradient calculation (meters).
                         McCormack et al. (2019) recommend ~10x ice thickness.
    Returns:
        flow_x, flow_y: Normalized vector components of the flow direction.
    """
    delta = ice_thickness * 5
    
    # Create stencil coordinates (Central Difference)
    x_left, x_right = x - delta, x + delta
    y_down, y_up    = y - delta, y + delta
    
    # Re-use your existing function to query the 4 neighbors
    z_left  = extract_rema_elevation(x_left, y, dem_path)
    z_right = extract_rema_elevation(x_right, y, dem_path)
    z_down  = extract_rema_elevation(x, y_down, dem_path)
    z_up    = extract_rema_elevation(x, y_up, dem_path)
    
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