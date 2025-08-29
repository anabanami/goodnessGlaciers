import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from model import model
from scipy.interpolate import interp1d
from bamgflowband import bamgflowband
from configp import config
from paterson import paterson

""" Compare the old and new Budd (1970) Sliding Law implementations """

# Create mesh
def setup_test_model():
    # Create a higher-resolution, uniform grid for more accurate representation
    x_transformed = np.linspace(config.x_params['start'], 
                        config.x_params['end'],
                        int((config.x_params['end'] - config.x_params['start'])/config.x_params['step']) + 1)

    # Get bedrock elevation with exact wavelength
    b_transformed = config.get_bedrock_elevation(x_transformed)

    # Use linear interpolation with 'linear' method explicitly to preserve features
    mean_thickness = config.ice_params['mean_thickness']
    h_transformed = mean_thickness * np.ones_like(x_transformed)

    # Create model with improved mesh settings
    md = bamgflowband(model(), x_transformed, b_transformed + h_transformed, b_transformed, 'hmax', config.mesh_hmax)

    # Ensure the mesh points are within the interpolation domain
    mesh_x = np.clip(md.mesh.x, 
                    config.x_params['start'], 
                    config.x_params['end'] - 1e-10)  # Small buffer to avoid edge issues

    # Create interpolation functions. Use linear interpolation to ensure wavelength is preserved
    surface_interpolant = interp1d(x_transformed, b_transformed + h_transformed, kind='linear')
    base_interpolant = interp1d(x_transformed, b_transformed, kind='linear')

    # Apply interpolation to mesh points
    md.geometry.surface = surface_interpolant(mesh_x)
    md.geometry.base = base_interpolant(mesh_x)
    md.geometry.thickness = md.geometry.surface - md.geometry.base
    md.geometry.bed = md.geometry.base
    
    # Initialize friction parameters
    md.friction.coefficient = np.ones((md.mesh.numberofvertices))
    md.friction.p = np.ones((md.mesh.numberofelements))
    md.friction.q = np.ones((md.mesh.numberofelements))
    
    # Initialize material properties for viscosity calculation
    # Use the B value from config instead of calculating from temperature
    md.materials.rheology_B = config.B * np.ones(md.mesh.numberofvertices)
    
    return md


# Create the test model
md = setup_test_model()

# Get basal nodes
basal_nodes = np.where(md.mesh.vertexflags(1))[0]
x_coords = md.mesh.x
print(f"Number of basal nodes: {len(basal_nodes)}")

# Calculate old-style sliding coefficient
old_sliding_coefficient = np.ones_like(md.friction.coefficient)
old_sliding_coefficient[basal_nodes] = 1.0 + 10 * np.cos(config.omega * x_coords[basal_nodes])

# ----

# Calculate the individual components of the physics-based formula
omega = config.omega
Beta_1 = config.bedrock_params['amplitude']
# Use the hardness parameter B from config directly
eta = 0.5 * config.B

# Calculate coefficient terms for basal nodes
omega_x = omega * x_coords[basal_nodes]
cos_term = np.cos(omega_x)

# Calculate the base coefficient
base_coefficient = 2 * eta * omega * Beta_1 * cos_term

# Calculate the offset to ensure non-negative values
# Add a small buffer (2.1 factor) for numerical safety
offset = 2.1 * eta * omega * Beta_1

# Apply the offset to ensure non-negative values
new_sliding_coefficient = np.ones_like(md.friction.coefficient)
new_sliding_coefficient[basal_nodes] = base_coefficient + offset

# Set friction to zero in flat regions (using domain boundaries from get_bedrock_elevation)
straight_section_1 = config.x_params['start'] + 5.0  # First 5km is flat
straight_section_2 = config.x_params['end'] - 5.0    # Last 5km is flat

# Create mask for flat regions at the domain edges
flat_regions = []
num_modified = 0

for i, idx in enumerate(basal_nodes):
    if x_coords[idx] < straight_section_1 or x_coords[idx] >= straight_section_2:
        flat_regions.append(idx)
        num_modified += 1

# Set friction to a very small value in the flat regions
if num_modified > 0:
    print(f"Setting friction to near-zero for {num_modified} nodes at domain edges")
    for idx in flat_regions:
        new_sliding_coefficient[idx] = 3000  # Use minimum value in pattern
else:
    print("WARNING: No edge nodes identified for friction modification!")

# Optional: Apply clipping if needed for numerical stability
new_coefficient_clipped = np.clip(new_sliding_coefficient, 0.001, 5.0)

# Generate spatial visualization of the mesh with coefficient values
plt.figure(figsize=(15, 5))

# Plot full mesh first
plt.scatter(md.mesh.x, md.mesh.y, s=2, color='lightgrey', alpha=0.5, label='mesh')

# Plot basal nodes with new coefficient values
basal_x = md.mesh.x[basal_nodes]
basal_y = md.mesh.y[basal_nodes]
sc = plt.scatter(basal_x, basal_y, s=40, c=new_sliding_coefficient[basal_nodes], 
                 cmap='viridis', label='Basal nodes')
plt.colorbar(sc, label="Budd's sliding coefficient")
plt.title("Mesh with Budd's Sliding Coefficient")
plt.xlabel('X coordinate (km)')
plt.ylabel('Y coordinate (km)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("budd_sliding_coefficient_mesh.png")
plt.show()
# plt.close()

# Print summary statistics
print("\nSummary Statistics:")
print(f"Old coefficient range: {old_sliding_coefficient[basal_nodes].min():.3f} to {old_sliding_coefficient[basal_nodes].max():.3f}")
print(f"New coefficient range: {new_sliding_coefficient[basal_nodes].min():.3f} to {new_sliding_coefficient[basal_nodes].max():.3f}")
print(f"Mean old coefficient: {np.mean(old_sliding_coefficient[basal_nodes]):.3f}")
print(f"Mean new coefficient: {np.mean(new_sliding_coefficient[basal_nodes]):.3f}")
print(f"Formula parameters: η={eta:.3e}, ω={omega:.3f}, β={Beta_1:.3e}")
print(f"Number of nodes with near-zero friction at domain edges: {num_modified}")