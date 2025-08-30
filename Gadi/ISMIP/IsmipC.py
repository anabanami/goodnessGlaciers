import numpy as np
from SetIceSheetBC import SetIceSheetBC
from plotmodel import plotmodel
import matplotlib.pyplot as plt


# Parameterization for ISMIP C experiment

# Set the Simulation generic name # md.miscellaneous
md.miscellaneous.name = 'ISMIP_C'

# Geometry
print('   Constructing Geometry')
# Define the geometry of the simulation # md.geometry
# surface is [-x*tan(0.1*pi/180)] # md.mesh
# md.geometry.surface=-md.mesh.x*tan(0.1*pi/180.);

md.geometry.surface = - md.mesh.x * np.tan(0.1*np.pi/180.)

# base is [surface-1000]
# L is the size of the side of the square # max(md.mesh.x)-min(md.mesh.x)
# L=max(md.mesh.x)-min(md.mesh.x);
# md.geometry.base=md.geometry.surface-1000.0 

L = max(md.mesh.x) - min(md.mesh.x)
md.geometry.base = md.geometry.surface - 1000.0 

# thickness is the difference between surface and base # md.geometry
# md.geometry.thickness=md.geometry.surface-md.geometry.base;

md.geometry.thickness = md.geometry.surface - md.geometry.base

print('   Defining friction parameters')
# one friction coefficient per node (md.mesh.numberofvertices,1)
# md.friction.coefficient=1000.0+ 1000.0 * sin(omega * md.mesh.x) * sin(omega * md.mesh.y)
omega = 2 * np.pi / L
beta2_raw = 1000.0 + 1000.0 * np.sin(omega * md.mesh.x) * np.sin(omega * md.mesh.y)
#REGULARISATION TO AVOID velocity singularity v_x(z_b)=(ρ g H tanα)/β2 where β2 = 0
md.friction.coefficient = np.maximum(beta2_raw, 1.0)

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
