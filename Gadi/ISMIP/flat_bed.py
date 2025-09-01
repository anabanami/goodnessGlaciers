import numpy as np
from plotmodel import plotmodel
from SetIceSheetBC import SetIceSheetBC
#Parameterization for ISMIP F experiment

#Set the Simulation generic name #md.miscellaneous
md.miscellaneous.name = 'flat_bed'

A = 2.140373 * 1e-7 # ice-flow parameter, units: Pa⁻¹ a⁻¹
n = 1 # flow law exponent
alpha = - 0.1 # mean surface slope (max in x zero in y), units: ◦

H_0 =1e3 # ice thickness, units: m

# perturbation parameters
sigma = 10 * H_0 # gaussian bump width, units: m
amplitude_0 = 0.1 * H_0 # gaussian bump amplitude(100m), units: m

#Geometry
print('   Constructing Geometry')

#Define the geometry of the simulation #md.geometry
md.geometry.surface = md.mesh.x * np.tan(alpha * np.pi / 180.0)

#L is the size of the side of the square #max(md.mesh.x)-min(md.mesh.x)
L = np.nanmax(md.mesh.x) - np.nanmin(md.mesh.x)

# should this be the bed not the base????
md.geometry.base = md.geometry.surface - 1000.0 

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
#B has one value per vertex
md.materials.rheology_B = (1 / A_seconds) * np.ones((md.mesh.numberofvertices))
#n has one value per element
md.materials.rheology_n = np.ones((md.mesh.numberofelements))

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
	
