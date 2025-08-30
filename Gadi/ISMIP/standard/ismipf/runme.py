import os
from pathlib import Path
from generic import generic
from socket import gethostname

import numpy as np
import matplotlib.pyplot as plt

from model import model
from squaremesh import squaremesh
from plotmodel import plotmodel
from export_netCDF import export_netCDF
from loadmodel import loadmodel
from setmask import setmask
from parameterize import parameterize
from setflowequation import setflowequation
from verbose import verbose
from solve import solve

#step 7 is specific to ISMIPA
ParamFile='IsmipA.par'
steps = [1, 2, 3, 4, 5, 6, 7]

## step 8 is specific to ISMIPF
# ParamFile = 'IsmipF.py'
# steps = [1, 2, 3, 4, 5, 6, 8]


# 7 steps:
# Mesh Generation #1/7
print("Generating the mesh")
#initialize md as a new model
md = model() 
md = squaremesh(md, 100000, 100000, 30, 30) #(md, x-meters, y-meters, x-nodes, y-nodes)
# plot the given mesh 
plotmodel(md, 'data', 'mesh','figure', 1)
plt.show()
# save the given model
export_netCDF(md, 'mesh.nc')


# Masks #2/7
print("Setting the masks")
# load the preceding step
md = loadmodel('mesh.nc')
# all ISMIP nodes are grounded # where we are solving
md = setmask(md, '', '') 
# plot the given mask #md.mask to locate the field
plotmodel(md, 'data', md.mask.ocean_levelset, 'figure', 2) #<<<md.mask.ocean_levelset???
plt.show()
# save the given model
export_netCDF(md, 'masked.nc')


# Parameterisation #3/7
print("Parameterizing")
# load the preceding step
md = loadmodel('masked.nc')
# parametrize the model
md = parameterize(md, ParamFile)
# save the given model
export_netCDF(md, 'parameterized.nc')


# Extrusion #4/7
print("Extruding")
# load the preceding step
md = loadmodel('parameterized.nc')
# vertically extrude the preceding mesh
# only 5 layers exponent 1
md = md.extrude(5, 1)
# plot the 3D geometry
plotmodel(md, 'data', md.geometry.base, 'figure', 3)
plt.show()
# save the given model
export_netCDF(md, 'extrusion.nc')


#Set the flow computing method #5/7
print("setting flow approximation")
# load the preceding step
md = loadmodel('extrusion.nc')
# set the approximation for the flow computation
# Higher Order Model (HO) TRY SIA TOO
md = setflowequation(md, 'HO', 'all')
# md = setflowequation(md, 'SIA', 'all') <<<< inconsistent for ice shelf?
# save the given model
export_netCDF(md, 'setflow.nc')


#Set Boundary Conditions #6/7
print("setting boundary conditions")
# load the preceding step
md = loadmodel('setflow.nc')

# DIRICHLET boundary condition are known as SPCs
# ice frozen to the base, no velocity   #md.stressbalance
# SPCs are initialized at NaN one value per vertex
# (SPCs values that the solution takes along the boundary of the domain are fixed)
md.stressbalance.spcvx = np.nan * np.ones((md.mesh.numberofvertices))
md.stressbalance.spcvy = np.nan * np.ones((md.mesh.numberofvertices))
md.stressbalance.spcvz = np.nan * np.ones((md.mesh.numberofvertices))

# extract the nodenumbers at the base #md.mesh.vertexonbase
basalnodes = np.nonzero(md.mesh.vertexonbase)
# set the sliding to zero on the bed (Vx and Vy)
md.stressbalance.spcvx[basalnodes] = 0.0
md.stressbalance.spcvy[basalnodes] = 0.0

# periodic boundaries have to be fixed on the sides
# Find the indices of the sides of the domain, for x and then for y
# for x
# create maxX, list of indices where x is equal to max of x
maxX = np.squeeze(np.nonzero(md.mesh.x == np.nanmax(md.mesh.x)))
# create minX, list of indices where x is equal to min of x
minX = np.squeeze(np.nonzero(md.mesh.x == np.nanmin(md.mesh.x)))
# for y
# create maxY, list of indices where y is equal to max of y
# but not where x is equal to max or min of x
# (i.e, indices in maxX and minX should be excluded from maxY and minY)
maxY = np.squeeze(np.nonzero(np.logical_and.reduce((md.mesh.y == np.nanmax(md.mesh.y), md.mesh.x != np.nanmin(md.mesh.x), md.mesh.x != np.nanmax(md.mesh.x)))))
# create minY, list of indices where y is equal to max of y
# but not where x is equal to max or min of x
minY = np.squeeze(np.nonzero(np.logical_and.reduce((md.mesh.y == np.nanmin(md.mesh.y), md.mesh.x != np.nanmin(md.mesh.x), md.mesh.x != np.nanmax(md.mesh.x)))))
# set the node that should be paired together, minX with maxX and minY with maxY
# #md.stressbalance.vertex_pairing
md.stressbalance.vertex_pairing = np.hstack((np.vstack((minX + 1, maxX + 1)), np.vstack((minY + 1, maxY + 1)))).T
md.masstransport.vertex_pairing = md.stressbalance.vertex_pairing
# save the given model
export_netCDF(md, 'BCS.nc')


#Solving 7/7
print("running the solver")
md = loadmodel("./BCS.nc")
# Set cluster #md.cluster
# set only the name and number of process
md.cluster = generic('name', gethostname(), 'np', 2)
# Set which control message you want to see #help verbose
md.verbose = verbose('convergence', True)
# set the transient model to ignore the thermal model
# #md.transient
md.transient.isthermal = 0
# define the timestepping scheme
# everything here should be provided in years #md.timestepping
# give the length of the time_step (4 years)
md.timestepping.time_step = 4
# give final_time (20*4 years time_steps)
md.timestepping.final_time = 20 * 4
# we are solving a TransientSolution
md = solve(md, 'Transient')
# save the given model
export_netCDF(md, "./Solution.nc")


###############################################################################

# Plotting BEV
def plot_simulation_frames(md, output_folder):
    """
    Plot and save each timestep of the simulation with proper colorbar handling
    """
    num_timesteps = len(md.results.TransientSolution)
    
    # Create a figure with subplots for velocity and thickness
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Keep track of colorbar axes to remove them
    cbar1 = None
    cbar2 = None
    
    for timestep in range(num_timesteps):
        # Clear previous colorbars if they exist
        if cbar1 is not None:
            cbar1.remove()
        if cbar2 is not None:
            cbar2.remove()
            
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        
        # Get current solution
        solution = md.results.TransientSolution[timestep]
        
        # Plot velocity
        vel = solution.Vel
        im1 = ax1.tricontourf(md.mesh.x, md.mesh.y, vel)
        ax1.set_title(f'Velocity at timestep {timestep} (year {timestep * 4})')
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Velocity (m/year)')
        
        # Plot thickness
        thickness = solution.Thickness
        im2 = ax2.tricontourf(md.mesh.x, md.mesh.y, thickness)
        ax2.set_title(f'Thickness at timestep {timestep} (year {timestep * 4})')
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Thickness (m)')
        
        # Add grid and axes labels
        for ax in [ax1, ax2]:
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_xlabel('Distance (m)')
            ax.set_ylabel('Distance (m)')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save the current frame
        plt.savefig(output_folder / f'timestep_{timestep:03d}.png', dpi=300, bbox_inches='tight')
        
        # Display the plot (comment this out for faster batch processing)
        plt.pause(0.5)
    
    plt.close()
# Usage:
folder1 = Path('sim_BEV')
folder1.mkdir(exist_ok=True)

# Clear existing files
for f in folder1.glob('*.png'):
    f.unlink()

md = loadmodel("./Solution.nc")
plot_simulation_frames(md, folder1)


# def plot_simulation_frames_side_view(md, output_folder):
#     """
#     Plot zoomed side view of ice thickness evolution with maximized plot area
#     """
#     num_timesteps = len(md.results.TransientSolution)
    
#     # Create figure with tight layout from the start
#     fig = plt.figure(figsize=(15, 8))
#     # Create axes that fills most of the figure
#     ax = fig.add_axes([0.1, 0.1, 0.75, 0.8])  # [left, bottom, width, height]
    
#     # Get coordinates at middle y-value
#     mid_y = np.mean(md.mesh.y)
#     tolerance = (md.mesh.y[1] - md.mesh.y[0]) / 2
#     mid_y_indices = np.where(np.abs(md.mesh.y - mid_y) < tolerance)[0]
    
#     # Sort points by x-coordinate
#     x_coords = md.mesh.x[mid_y_indices]
#     sort_idx = np.argsort(x_coords)
#     x_coords = x_coords[sort_idx]
    
#     # Get bedrock profile
#     bed = md.geometry.base[mid_y_indices][sort_idx]
    
#     # Keep track of colorbar
#     cbar = None
    
#     # Set zoom limits
#     x_min = 15000      # Start of x-axis (m)
#     x_max = 25000      # End of x-axis (m)
#     y_min = 15000      # Bottom of y-axis (m)
#     y_max = 25000      # Top of y-axis (m)
    
#     for timestep in range(num_timesteps):
#         # Clear previous plot and colorbar
#         ax.clear()
#         if cbar is not None:
#             cbar.remove()
        
#         # Get current solution
#         solution = md.results.TransientSolution[timestep]
        
#         # Get thickness along the profile
#         thickness = solution.Thickness[mid_y_indices][sort_idx]
#         surface = bed + thickness
        
#         # Plot bed and surface
#         ax.plot(x_coords, bed, 'k-', label='Bedrock', linewidth=2)
#         ax.fill_between(x_coords, bed, surface, color='lightblue', alpha=0.3)
#         ax.plot(x_coords, surface, 'b-', label='Ice Surface', linewidth=2)
        
#         # Add velocity information
#         vel = solution.Vel[mid_y_indices][sort_idx]
#         scatter = ax.scatter(x_coords, surface, c=vel, cmap='viridis', 
#                            s=50, label='Velocity', zorder=5)
        
#         # Add colorbar in a good position
#         cbar_ax = fig.add_axes([0.87, 0.1, 0.03, 0.8])  # [left, bottom, width, height]
#         cbar = plt.colorbar(scatter, cax=cbar_ax)
#         cbar.set_label('Velocity (m/year)', fontsize=10)
        
#         # Set proper aspect ratio
#         ax.set_aspect('equal')
        
#         # Customize plot
#         ax.set_xlabel('Distance (m)', fontsize=12)
#         ax.set_ylabel('Elevation (m)', fontsize=12)
#         ax.set_title(f'Ice Profile at timestep {timestep} (year {timestep * 4})', 
#                     fontsize=14, pad=20)  # Added pad to prevent title overlap
#         ax.grid(True, linestyle='--', alpha=0.3)
#         ax.legend(loc='upper left', fontsize=10)
        
#         # Set zoom limits
#         ax.set_xlim(x_min, x_max)
#         ax.set_ylim(y_min, y_max)
        
#         # Save the current frame
#         plt.savefig(output_folder / f'profile_timestep_{timestep:03d}.png', 
#                    dpi=300, bbox_inches='tight')
        
#         # Display the plot
#         plt.pause(0.5)
    
#     plt.close()

# # Usage:
# folder2 = Path('sim_SideV')
# folder2.mkdir(exist_ok=True)

# # Clear existing files
# for f in folder2.glob('*.png'):
#     f.unlink()

# md = loadmodel("./Solution.nc")
# plot_simulation_frames_side_view(md, folder2)