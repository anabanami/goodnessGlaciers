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
alpha = - 3.0 # mean surface slope (max in x zero in y), units: ◦

H_0 = 1e3 # ice thickness, units: m

# perturbation parameters
sigma = 10 * H_0 # gaussian bump width, units: m
amplitude_0 = 0.1 * H_0 # gaussian bump amplitude(500m), units: m

#Geometry
print('   Constructing Geometry')

#Define the geometry of the simulation #md.geometry
#surface is [-x*tan(3.0*pi/180)] #md.mesh
md.geometry.surface = md.mesh.x * np.tan(alpha * np.pi / 180.0)
#base is [surface-1000+100*exp(-((x-L/2).^2+(y-L/2).^2)/(10000.^2))]
#L is the size of the side of the square #max(md.mesh.x)-min(md.mesh.x)
L = np.nanmax(md.mesh.x) - np.nanmin(md.mesh.x)

# should this be the bed not the base???? if using bed then model is not consistent
md.geometry.base = md.geometry.surface - 1000.0

md.geometry.thickness = md.geometry.surface - md.geometry.base

#plot the geometry to check it out
# plotmodel(md, 'data', md.geometry.thickness)

print('   Defining friction parameters')

#conversion form year to seconds with #md.constants.yts
# md.friction.coefficient = np.sqrt(md.constants.yts / (1000 * A)) * np.ones((md.mesh.numberofvertices)) << this is the website version?

# I think that this is the correct version based on Pattyn 2008
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
	
