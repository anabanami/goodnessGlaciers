import numpy as np
from SetIceSheetBC import SetIceSheetBC
from plotmodel import plotmodel
import matplotlib.pyplot as plt


# Parameterization for ISMIP A experiment

# Set the Simulation generic name # md.miscellaneous
md.miscellaneous.name = 'ISMIP_A'

# Geometry
print('   Constructing Geometry')
# Define the geometry of the simulation # md.geometry
# surface is [-x*tan(0.5*pi/180)] # md.mesh
# md.geometry.surface=-md.mesh.x*tan(0.5*pi/180.);

md.geometry.surface = - md.mesh.x * np.tan(0.5*np.pi/180.)

# base is [surface-1000+500*sin(x*2*pi/L).*sin(y*2*pi/L)]
# L is the size of the side of the square # max(md.mesh.x)-min(md.mesh.x)
# L=max(md.mesh.x)-min(md.mesh.x);
# md.geometry.base=md.geometry.surface-1000.0+500.0*sin(md.mesh.x*2.0*pi/L).*sin(md.mesh.y*2.0*pi/L);

L = max(md.mesh.x) - min(md.mesh.x)
md.geometry.base = md.geometry.surface - 1000.0 + 500.0 * np.sin(md.mesh.x * 2.0 * np.pi / L) * np.sin(md.mesh.y * 2.0 * np.pi / L)

# thickness is the difference between surface and base # md.geometry
# md.geometry.thickness=md.geometry.surface-md.geometry.base;

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
