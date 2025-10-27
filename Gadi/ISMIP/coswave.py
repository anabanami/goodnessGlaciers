import numpy as np
from plotmodel import plotmodel
from SetIceSheetBC import SetIceSheetBC
import matplotlib.pyplot as plt

# Parameterization for ISMIP F experiment

#Set the Simulation generic name #md.miscellaneous
filename = md.miscellaneous.filename
Scenario = md.miscellaneous.scenario
h_res = md.miscellaneous.h_resolution_factor
v_res = md.miscellaneous.v_resolution_factor

A = 2.140373 * 1e-7 # ice-flow parameter, units: Pa⁻¹ a⁻¹
alpha = - 3 # mean surface slope (max in x zero in y), units: ◦

H_0 = 1e3 # ice thickness, units: m

# perturbation parameters
amplitude_0 = 0.05 * H_0 # amplitude(100), units: m

# 1. Define TARGET wavelength
# target_wavelength = 2 * H_0 # (2e3) units: m
# target_wavelength = 3.3 * H_0 # (3.3e3) units: m
target_wavelength = 10 * H_0 # (10e3) units: m

# Extract wavelength scaling factor for naming
wavelength_factor = target_wavelength / H_0

# Construct the file_prefix string including wavelength factor
file_prefix = f"{filename}_{Scenario}_{h_res}_{v_res}_w{wavelength_factor:g}"

md.miscellaneous.name = file_prefix + '-Transient'

# 2. Get the length of the domain from the mesh
domain_length_x = np.max(md.mesh.x) - np.min(md.mesh.x)
# Fraction of the domain should have the undulating bed.
undulation_fraction = 0.8
print('   Defining undulation zone...')
# length of the undulating central section
undulation_length = domain_length_x * undulation_fraction
print(f'   Total domain length: {domain_length_x:.2f} m')
print(f'   Undulation zone length: {undulation_length:.2f} m')

# start and end coordinates of the undulation zone
buffer_length = (domain_length_x - undulation_length) / 2
x_min = np.min(md.mesh.x)
x_max = np.max(md.mesh.x)
x_start = x_min + buffer_length
x_end = x_max - buffer_length
print(f'   Wavy bed from x={x_start:.2f} m to x={x_end:.2f} m')

# 3. Calculate how many target waves fit and round to the nearest integer
num_waves_integer = np.round(undulation_length / target_wavelength)
print(f'   Adjusting to fit {num_waves_integer} waves into the domain.')

# 4. Calculate the NEW, adjusted wavelength that fits perfectly in the domain
adjusted_wavelength = undulation_length / num_waves_integer
print(f'   New adjusted wavelength: {adjusted_wavelength:.2f} m')

# 5. Use this adjusted wavelength for your calculations
omega = 2 * np.pi / adjusted_wavelength

# --- Construct Geometry ---
print('   Constructing Geometry')
md.geometry.surface = md.mesh.x * np.tan(alpha * np.pi / 180.0) # -0.0524 slope

# --- Construct the Bed Perturbation ---
print('   Constructing Bed Perturbation')
# We use a modified cosine function: A * (cos(k*x) - 1).
# This ensures the wave starts and ends at a value of 0 with a slope of 0,
# creating a smooth (C1 continuous) join with the flat sections.
# The wave will oscillate downwards, between 0 and -2 * amplitude_0.

if num_waves_integer > 0:
    wave_component = amplitude_0 * (np.cos((md.mesh.x - x_start) * omega) - 1)
else:
    wave_component = 0.0 # No waves if zone is too small

bed_perturbation = np.where(
    (md.mesh.x >= x_start) & (md.mesh.x <= x_end), # Condition
    wave_component,                               # Value if True
    0.0                                           # Value if False
)

# --- Construct Geometry ---
print('   Constructing Geometry')
md.geometry.surface = md.mesh.x * np.tan(alpha * np.pi / 180.0)

## XY pattern (egg carton)
# bed_perturbation = amplitude_0 * np.cos(md.mesh.x * omega) * np.cos(md.mesh.y * omega)

# Add the localized 2D perturbation to the base
md.geometry.base = md.geometry.surface - H_0 + bed_perturbation
md.geometry.thickness = md.geometry.surface - md.geometry.base



##########################################################################################################
# After calculating bed_perturbation, add validation:
print(f'   Bed perturbation range: [{bed_perturbation.min():.2f}, {bed_perturbation.max():.2f}] m')

# Ensure thickness is always positive
min_thickness = md.geometry.thickness.min()
if min_thickness <= 0:
    print(f'   WARNING: Negative or zero thickness detected: {min_thickness:.2f} m')
    # Add a small buffer to ensure positive thickness everywhere
    md.geometry.thickness = np.maximum(md.geometry.thickness, 10.0)  # Minimum 10m thickness
    md.geometry.base = md.geometry.surface - md.geometry.thickness
    print(f'   Adjusted minimum thickness to 10 m')

else:
	print(f"    {min_thickness = }")

##########################################################################################################




print('   Defining friction parameters')

# convert to ISSM units:
A_seconds =  A / md.constants.yts
c = 1.0
beta_squared = 1.0 / (c * A_seconds * H_0)
md.friction.coefficient = np.sqrt(beta_squared) * np.ones((md.mesh.numberofvertices))

# one friction exponent (p,q) per element
md.friction.p = np.ones((md.mesh.numberofelements))
md.friction.q = np.zeros((md.mesh.numberofelements))

print('   Construct ice rheological properties')

# The rheology parameters sit in the material section #md.materials
# B_1 has one value per vertex
rheology_B_1 = (1 / A_seconds) * np.ones((md.mesh.numberofvertices))
# n has one value per element
linear_rheology_n = np.ones((md.mesh.numberofelements))

md.materials.rheology_B = rheology_B_1
# n has one value per element
md.materials.rheology_n = linear_rheology_n

# SCALING B following Getraer and Morlihem (2025).
non_linear_rheology_n = 4 * np.ones((md.mesh.numberofelements))

# Experimental Scenario
if Scenario in ("S1", "S3"):
	md.materials.rheology_B = rheology_B_1
	# n has one value per element
	md.materials.rheology_n = linear_rheology_n

elif Scenario == "S2":
	epsilon_S1 = 0.10275 # units: a⁻¹
	# for internal unit consistency
	epsilon_S1_seconds = epsilon_S1 / md.constants.yts  # units: s⁻¹
    # SCALING B following Getraer and Morlihem (2025).
	md.materials.rheology_B = rheology_B_1 * epsilon_S1_seconds**(3/4) # units: Pa a⁽¹/⁴⁾
	md.materials.rheology_n = non_linear_rheology_n

else: # Scenario == "S4"
	epsilon_S3 = 0.20509 # units: a⁻¹
	# for internal unit consistency
	epsilon_S3_seconds = epsilon_S3 / md.constants.yts # units: s⁻¹
	md.materials.rheology_B = rheology_B_1 * epsilon_S3_seconds**(3/4) # units: Pa a⁽¹/⁴⁾
	md.materials.rheology_n = non_linear_rheology_n

print('   Set boundary conditions')

# Set the default boundary conditions for an ice-sheet
# # help SetIceSheetBC
md = SetIceSheetBC(md)

print('   Initializing velocity and pressure')

#initialize the velocity and pressurefields of #md.initialization
md.initialization.vx = np.zeros((md.mesh.numberofvertices))
md.initialization.vy = np.zeros((md.mesh.numberofvertices))
md.initialization.vz = np.zeros((md.mesh.numberofvertices))
md.initialization.pressure = np.zeros((md.mesh.numberofvertices))
	
