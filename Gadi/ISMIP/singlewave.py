import numpy as np
from plotmodel import plotmodel
from SetIceSheetBC import SetIceSheetBC
#Parameterization for ISMIP F experiment

#Set the Simulation generic name #md.miscellaneous
filename = md.miscellaneous.filename
Scenario = md.miscellaneous.scenario
h_res = md.miscellaneous.h_resolution_factor
v_res = md.miscellaneous.v_resolution_factor

# Construct the file_prefix string
file_prefix = f"{filename}_{Scenario}_{h_res}_{v_res}"

md.miscellaneous.name = file_prefix + "-Transient"


A = 2.140373 * 1e-7 # ice-flow parameter, units: Pa⁻¹ a⁻¹
alpha = - 3 # mean surface slope (max in x zero in y), units: ◦

H_0 = 1e3 # ice thickness, units: m

# perturbation parameters
sigma = 10 * H_0 # gaussian bump width, units: m
amplitude_0 = 0.1 * H_0 # gaussian bump amplitude(100m), units: m

# --- Define the 2D Feature Parameters ---
# Define the center of the domain for the feature
x_min = np.nanmin(md.mesh.x)
x_max = np.nanmax(md.mesh.x)
x_center = (x_max + x_min) / 2.0

y_min = np.nanmin(md.mesh.y)
y_max = np.nanmax(md.mesh.y)
y_center = (y_max + y_min) / 2.0

# For a circular feature, we'll set them to be the same
sigma_x = sigma
sigma_y = sigma

# --- Construct Geometry ---
print('   Constructing Geometry')
md.geometry.surface = md.mesh.x * np.tan(alpha * np.pi / 180.0)

# --- Create the 2D heap-and-trough perturbation ---
# 1. Calculate shifted coordinates
x_shifted = md.mesh.x - x_center
y_shifted = md.mesh.y - y_center

# 2. Calculate the 2D Gaussian envelope
gaussian_2d = np.exp( -((x_shifted**2) / (2 * sigma_x**2) + (y_shifted**2) / (2 * sigma_y**2)) )

# 3. Calculate the un-normalized derivative with respect to x
raw_perturbation = -x_shifted * gaussian_2d

# 4. Normalize so the peak magnitude is 1.0
if np.nanmax(np.abs(raw_perturbation)) > 0:
	normalized_perturbation = raw_perturbation / np.nanmax(np.abs(raw_perturbation))
else:
	normalized_perturbation = np.zeros_like(raw_perturbation)

# 5. Scale by the desired amplitude
bed_perturbation = amplitude_0 * normalized_perturbation

# Add the localized 2D perturbation to the base
md.geometry.base = md.geometry.surface - 1000.0 + bed_perturbation
md.geometry.thickness = md.geometry.surface - md.geometry.base

#plot the geometry to check it out
# plotmodel(md, 'data', md.geometry.thickness)

print('   Defining friction parameters')
# convert to ISSM units:
A_seconds =  A / md.constants.yts
c = 1.0
beta_squared = 1.0 / (c * A_seconds * H_0)
md.friction.coefficient = np.sqrt(beta_squared) * np.ones((md.mesh.numberofvertices))

#These parameters will not be used but need to be fixed #md.friction
#one friction exponent (p,q) per element
md.friction.p = np.ones((md.mesh.numberofelements))
md.friction.q = np.zeros((md.mesh.numberofelements))

print('   Construct ice rheological properties')

#The rheology parameters sit in the material section #md.materials
# B_1 has one value per vertex
rheology_B_1 = (1 / A_seconds) * np.ones((md.mesh.numberofvertices))
#n has one value per element
linear_rheology_n = np.ones((md.mesh.numberofelements))

md.materials.rheology_B = rheology_B_1
#n has one value per element
md.materials.rheology_n = linear_rheology_n

# SCALING B following Getraer and Morlihem (2025).
non_linear_rheology_n = 4 * np.ones((md.mesh.numberofelements))

# Experimental Scenario
if Scenario in ("S1", "S3"):
	md.materials.rheology_B = rheology_B_1
	#n has one value per element
	md.materials.rheology_n = linear_rheology_n

elif Scenario == "S2":
	epsilon_S1 = 0.10275 # units: a⁻¹
	# for internal unit consistency
	epsilon_S1_seconds = epsilon_S1 / md.constants.yts  # units: s⁻¹
    # SCALING B following Getraer and Morlihem (2025).
	md.materials.rheology_B = rheology_B_1 * epsilon_S1_seconds**(3/4) # units: Pa a⁽¹/⁴⁾
	md.materials.rheology_n = non_linear_rheology_n

else: #Scenario == "S4"
	epsilon_S3 = 0.20509 # units: a⁻¹
	# for internal unit consistency
	epsilon_S3_seconds = epsilon_S3 / md.constants.yts # units: s⁻¹
	md.materials.rheology_B = rheology_B_1 * epsilon_S3_seconds**(3/4) # units: Pa a⁽¹/⁴⁾
	md.materials.rheology_n = non_linear_rheology_n

print('   Set boundary conditions')

#Set the default boundary conditions for an ice-sheet
# #help SetIceSheetBC
md = SetIceSheetBC(md)

print('   Initializing velocity and pressure')

#initialize the velocity and pressurefields of #md.initialization
md.initialization.vx = np.zeros((md.mesh.numberofvertices))
md.initialization.vy = np.zeros((md.mesh.numberofvertices))
md.initialization.vz = np.zeros((md.mesh.numberofvertices))
md.initialization.pressure = np.zeros((md.mesh.numberofvertices))
	
