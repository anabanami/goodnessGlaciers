import numpy as np
from plotmodel import plotmodel
from SetIceSheetBC import SetIceSheetBC
#Parameterization for ISMIP F experiment

#Set the Simulation generic name #md.miscellaneous
md.miscellaneous.name = 'IsmipF_cor.py'
#Geometry
print('   Constructing Geometry')
#Define the geometry of the simulation #md.geometry
#surface is [-x*tan(3.0*pi/180)] #md.mesh
md.geometry.surface = md.mesh.x * np.tan(3.0 * np.pi / 180.0)

#base is [surface-1000+100*exp(-((x-L/2).^2+(y-L/2).^2)/(10000.^2))]
#L is the size of the side of the square #max(md.mesh.x)-min(md.mesh.x)
L = np.nanmax(md.mesh.x)-np.nanmin(md.mesh.x)
md.geometry.base = md.geometry.surface - 1000 + 100 * np.exp(-((md.mesh.x - L / 2)**2 + (md.mesh.y - L / 2)**2) / (10000**2))

#thickness is the difference between surface and base #md.geometry
md.geometry.thickness = md.geometry.surface - md.geometry.base
# Add bed geometry - for grounded ice, bed equals base
md.geometry.bed = md.geometry.base.copy()  # Copy base to bed

# Ensure no NaN values in bed
if np.any(np.isnan(md.geometry.bed)):
    print("Warning: NaN values found in bed geometry, filling with nearest valid values")
    # Replace NaN with valid values (you might want to adjust this based on your needs)
    valid_mask = ~np.isnan(md.geometry.bed)
    invalid_mask = np.isnan(md.geometry.bed)
    md.geometry.bed[invalid_mask] = np.interp(np.flatnonzero(invalid_mask),
                                             np.flatnonzero(valid_mask),
                                             md.geometry.bed[valid_mask])
    
#plot the geometry to check it out
plotmodel(md, 'data', md.geometry.thickness)


print('   Defining friction parameters')
#These parameters will not be used but need to be fixed #md.friction
#one friciton coefficient per node (md.mesh.numberofvertices,1)
#conversion from year to seconds with #md.constants.yts
md.friction.coefficient = np.sqrt(md.constants.yts / (1000 * 2.140373 * 1e-7)) * np.ones((md.mesh.numberofvertices))
#one friction exponent (p,q) per element
md.friction.p = np.ones((md.mesh.numberofelements))
md.friction.q = np.zeros((md.mesh.numberofelements))


print('   Construct ice rheological properties')
#The rheology parameters sit in the material section #md.materials
#B has one value per vertex
md.materials.rheology_B = (1 / (2.140373 * 1e-7 / md.constants.yts)) * np.ones((md.mesh.numberofvertices))
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

# Initialize additional fields needed for transient simulation
md.initialization.temperature = 273.15 * np.ones((md.mesh.numberofvertices))  # Initialize temperature field
md.initialization.watercolumn = np.zeros((md.mesh.numberofvertices))  # Initialize water column
md.initialization.groundedice = np.ones((md.mesh.numberofvertices))   # Initialize grounded ice mask

# Set up transient simulation parameters
md.timestepping.time_step = 4.0  # 4 year timestep
md.timestepping.final_time = 80.0  # 80 year simulation
md.timestepping.start_time = 0.0
md.timestepping.cycle_forcing = 1    # Enable forcing cycles if needed
md.settings.output_frequency = 1    # Output every timestep

# Configure solver parameters
md.stressbalance.maxiter = 100
md.stressbalance.stabilization = 1
md.stressbalance.convergence = 1e-4
md.stressbalance.reltol = 0.1

# Set up needed results fields
md.results.requested_outputs = ['default', 'GeometryEvolution']

# Disable thermal model as specified
md.transient.isthermal = 0
md.transient.isgroundingline = 1  # Enable grounding line evolution
md.transient.ismasstransport = 1  # Enable mass transport

# Set up mass balance forcing
smb = np.zeros((md.mesh.numberofvertices))  # Replace with actual SMB if needed
md.smb.mass_balance = smb