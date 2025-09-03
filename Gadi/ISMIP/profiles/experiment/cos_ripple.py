import numpy as np
from SetIceSheetBC import SetIceSheetBC
from plotmodel import plotmodel
import matplotlib.pyplot as plt


# Parameterization for cos ripple experiment

# Set the Simulation generic name # md.miscellaneous
md.miscellaneous.name = 'cos_ripple'

initial_elevation = 100 # m
ice_thickness = 1.92e3 # m
base_slope = np.tan(0.1*np.pi/180.) # -np.deg2rad(0.1) = tan(0.1°) × 100 = 0.175%
amplitude = 100 # amplitude = 0.02 * ice_thickness #= 38.4 m 
wavelength = 3.3 * ice_thickness
omega = 2 * np.pi / wavelength


# Geometry
print('   Constructing Geometry')

L = max(md.mesh.x) - min(md.mesh.x)


md.geometry.base = - (md.mesh.x * base_slope) + amplitude * np.cos(md.mesh.x * omega)
md.geometry.surface = md.geometry.base + ice_thickness
md.geometry.thickness = md.geometry.surface - md.geometry.base


print('   Defining friction parameters')
# These parameters will not be used but need to be fixed # md.friction
# one friction coefficient per node (md.mesh.numberofvertices,1)
# md.friction.coefficient=200.0*ones(md.mesh.numberofvertices,1);

md.friction.coefficient = 200.0 * np.ones(md.mesh.numberofvertices)

# one friciton exponent (p,q) per element
# md.friction.p=ones(md.mesh.numberofelements,1);
# md.friction.q=ones(md.mesh.numberofelements,1);

md.friction.p = np.ones(md.mesh.numberofelements)
md.friction.q = np.ones(md.mesh.numberofelements)


print('   Construct ice rheological properties')
# The rheology parameters sit in the material section # md.materials
# B has one value per vertex
# md.materials.rheology_B=6.8067e7*ones(md.mesh.numberofvertices,1);

md.materials.rheology_B = 6.8067e7 * np.ones(md.mesh.numberofvertices)

# n has one value per element
# md.materials.rheology_n=3*ones(md.mesh.numberofelements,1);

md.materials.rheology_n = 3 * np.ones(md.mesh.numberofelements)

# Rheology law
md.materials.rheology_law = "BuddJacka"

print('   Set boundary conditions')
# Set the default boundary conditions for an ice-sheet
# help SetIceSheetBC
# md=SetIceSheetBC(md);

md=SetIceSheetBC(md)
