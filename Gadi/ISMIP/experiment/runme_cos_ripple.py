import numpy as np
from model import model
from squaremesh import squaremesh
from plotmodel import plotmodel
from export_netCDF import export_netCDF
from loadmodel import loadmodel
from setmask import setmask
from parameterize import parameterize
from setflowequation import setflowequation
from solve import solve
from generic import generic
from socket import gethostname
from verbose import verbose

import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/ana/pyISSM/src')
import pyissm as issm
from pyissm import plot as iplt



# ====================================================================================================
# THINGS I NEED?
# md.initialization.temperature = 
# ====================================================================================================

ParamFile = 'cos_ripple.py'

x_max_length = 80000 # metres
y_max_length = 80000 # metres

x_nodes = 40
y_nodes = 40

#Run Steps

# Mesh Generation #1
print("\n===== Generating the mesh =====")
#initialize md as a new model
md = model()
# Side is 80 km long with 20 points
md = squaremesh(md, x_max_length, y_max_length, x_nodes, y_nodes)

md_mesh, md_x, md_y, md_elements, md_is3d = issm.model.mesh.process_mesh(md)
print("Plotting mesh")
iplt.plot_mesh2d(md_mesh, show_nodes = True)
plt.title("Full mesh") 
plt.savefig(f"cos_ripple_mesh.png")
# plt.show()

# convert the vertex on boundary array into boolean
boundary_mask = md.mesh.vertexonboundary.astype(bool)
x_boundaries = md.mesh.x[boundary_mask]
y_boundaries = md.mesh.y[boundary_mask]

print("\n===== Plotting mesh and highlighting vertex boundaries =====")
iplt.plot_mesh2d(md_mesh, show_nodes = True)
plt.scatter(x_boundaries, y_boundaries, label='boundaries')
plt.legend()
plt.title("Mesh boundaries") 
plt.savefig(f"cos_ripple_mesh_boundaries.png")
# plt.show()

# ====================================================================================================

# Masks #2
print("\n===== Setting the masks =====")
# # mask out areas that are not ice
nv, ne = md.mesh.numberofvertices, md.mesh.numberofelements
# print(nv)
# print(ne)

md.mask.ice_levelset = - np.ones(nv) #Ice is present if md.mask.ice_levelset is negative
md.mask.ocean_levelset = np.ones(nv) # grounded ice if positive (OR floating ice if negative)

# md = setmask(md, '', '') << ISSM SETUP WHY?

# plot the given mask #md.mask to locate the field
print("\n===== Plotting mesh and highlighting elements")
fig, ax = plt.subplots(figsize = (7, 7))
iplt.plot_model_elements(md,
                         md.mask.ice_levelset,
                         md.mask.ocean_levelset, 
                         type='grounded_ice_elements', 
                         ax = ax
)

# md.mask.ice_levelset[boundary_mask] = 0 #(icefront at boundary)
# iplt.plot_model_elements(md, 
#                          md.mask.ice_levelset, 
#                          md.mask.ocean_levelset, 
#                          type='ice_front_elements', 
#                          ax = ax, 
#                          color='tab:orange'
# )

plt.title("Set mask - grounded ice elements") 
plt.savefig("cos_ripple_grounded_ice_elements.png")
# plt.show()

# ====================================================================================================

# Parameterization #3
print("\n===== Parameterizing =====")
md = parameterize(md, ParamFile)

# THICKNESS plot
iplt.plot_model_field(md, md.geometry.thickness, show_cbar = True)
plt.title("Ice thickness") 
plt.savefig("cos_ripple_thickness_geometry_2D.png")
# plt.show()


# ====================================================================================================

# Extrusion #4
print("\n===== Extruding =====")

# vertically extrude the preceding mesh
# only 5 layers exponent 1
md = md.extrude(5, 1)
# plot the 3D geometry
print("\n===== Plotting base geometry =====")

## 3D plot
plotmodel(md, 'data', md.geometry.base, 'figure', 4,
        'figsize', [12, 12],
        'xlabel', 'x (m)',
        'ylabel', 'y (m)',
)

plt.tight_layout()
plt.title("3D model geometry") 
plt.savefig("cos_ripple_geometry_3D.png")
# plt.show()

# 2D plot
iplt.plot_model_field(md, md.geometry.base, layer=1, show_cbar = True)
plt.title("Base geometry") 
plt.savefig("cos_ripple_base_geometry_2D.png")
# plt.show()

# ====================================================================================================

# Set the flow computing method #5
print("\n===== setting flow approximation: HO =====")
# Higher Order Model (HO) TRY FS TOO
md = setflowequation(md, 'HO', 'all')
flowequation_type = 'High Order'

# print("\n===== setting flow approximation: FS =====")
# md = setflowequation(md, 'FS', 'all')
# md.stressbalance.FSreconditioning = 1
# flowequation_type = 'Full Stokes'

# ====================================================================================================

# Set Boundary Conditions #6
print("\n===== setting boundary conditions =====")

# DIRICHLET boundary condition are known as SPCs
# ice frozen to the base, no velocity
# SPCs are initialized at NaN one value per vertex
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
# for x:
# create maxX, list of indices where x is equal to max of x
maxX = np.squeeze(np.nonzero(md.mesh.x == np.nanmax(md.mesh.x)))
# create minX, list of indices where x is equal to min of x
minX = np.squeeze(np.nonzero(md.mesh.x == np.nanmin(md.mesh.x)))
# for y:
# create maxY, list of indices where y is equal to max of y
# but not where x is equal to max or min of x
# (i.e, indices in maxX and minX should be excluded from maxY and minY)
maxY = np.squeeze(np.nonzero(np.logical_and.reduce((md.mesh.y == np.nanmax(md.mesh.y), md.mesh.x != np.nanmin(md.mesh.x), md.mesh.x != np.nanmax(md.mesh.x)))))
# create minY, list of indices where y is equal to max of y
# but not where x is equal to max or min of x
minY = np.squeeze(np.nonzero(np.logical_and.reduce((md.mesh.y == np.nanmin(md.mesh.y), md.mesh.x != np.nanmin(md.mesh.x), md.mesh.x != np.nanmax(md.mesh.x)))))

# # set the node that should be paired together, minX with maxX and minY with maxY
md.stressbalance.vertex_pairing = np.hstack((np.vstack((minX + 1, maxX + 1)), np.vstack((minY + 1, maxY + 1)))).T

# ====================================================================================================

# Solving STRESSBALANCE #7
print("\n===== Running Stressbalance Solver =====")

# Set which control message to see
# md.verbose = verbose('convergence', True)

# misc
# md.masstransport.isfreesurface = 1
md.stressbalance.abstol = np.nan
# md.groundingline.migration = 'None'

# solver settings    
md.settings.solver_residue_threshold = 1e-4 
md.stressbalance.restol = 1e-4
md.stressbalance.reltol = 1e-4
md.stressbalance.maxiter = 100

# diagnostic solve
md.stressbalance.requested_outputs = ['default', # default='Vx', 'Vy', 'Vz', 'Vel', 'Pressure',
                                    'Thickness',
                                    'Surface',
                                    'Base', 
]

md = solve(md, 'Stressbalance')

print("=======================================================================================")
print("\nAvailable results in md.results.StressbalanceSolution:")
stress_solution = md.results.StressbalanceSolution
# print(stress_solution)

# Full solution arrays
vx_full = stress_solution.Vx
vy_full = stress_solution.Vy
vz_full = stress_solution.Vz
vel_full = stress_solution.Vel
Pressure_full = stress_solution.Pressure
Thickness_full = stress_solution.Thickness
Surface_full = stress_solution.Surface
Base_full = stress_solution.Base

basal_idx = np.where(md.mesh.vertexonbase == 1)[0]
vx_basal = vx_full[basal_idx]
vy_basal = vy_full[basal_idx]
vz_basal = vz_full[basal_idx]

surface_idx = np.where(md.mesh.vertexonsurface == 1)[0]
vx_surface = vx_full[surface_idx]
vy_surface = vy_full[surface_idx]
vz_surface = vz_full[surface_idx]

# ------------------------------------------------------------------
# Quick print‑outs
# ------------------------------------------------------------------
print(
    f"Surface velocity ranges (m a⁻¹):\n"
    f"  vx_surface: [{vx_surface.min():.5f}, {vx_surface.max():.5f}]\n"
    f"  vy_surface: [{vy_surface.min():.5f}, {vy_surface.max():.5f}]\n"
    f"  vz_surface: [{vz_surface.min():.5f}, {vz_surface.max():.5f}]"
)
print(
    "Basal velocity ranges (m a⁻¹):\n"
    f"  vx_basal: [{vx_basal.min():.5f}, {vx_basal.max():.5f}]\n"
    f"  vy_basal: [{vy_basal.min():.5f}, {vy_basal.max():.5f}]\n"
    f"  vz_basal: [{vz_basal.min():.5f}, {vz_basal.max():.5f}]"
)

print(
    "Vel ranges (m a⁻¹):\n"
    f"  vel: [{vel_full.min():.5f}, {vel_full.max():.5f}]\n"
)

print(
    "Pressure ranges (Pa):"f"[{Pressure_full.min():.5f}, {Pressure_full.max():.5f}]\n"
)

print(
    "Thickness ranges (m):"f"[{Thickness_full.min():.5f}, {Thickness_full.max():.5f}]\n"
)

# export_netCDF(md, "sb_solution.nc")

breakpoint()
# ======================================================================================= 

# Solving TRANSIENT #8
print("\n===== Transient solver for the A case =====")

time_step = 1
final_time = 20

# # Initialise Transient model from zero
md.initialization.vx = np.zeros_like(vx_full)
md.initialization.vy = np.zeros_like(vy_full)
md.initialization.vz = np.zeros_like(vz_full)
md.initialization.vel = np.zeros_like(vel_full)
md.initialization.pressure =  np.zeros_like(Pressure_full)
md.initialization.thickness =  np.zeros_like(Thickness_full)

# # Initialise from stress balance
# md.initialization.vx = stress_solution.Vx
# md.initialization.vy = stress_solution.Vy
# md.initialization.vz =s tress_solution.Vz
# md.initialization.vel = stress_solution.Vel
# md.initialization.pressure = stress_solution.Pressure
# md.initialization.thickness = stress_solution.Thickness

# Set which control message to see
# md.verbose = verbose('convergence', True)
# SET active Physics
md.settings.sb_coupling_frequency = 1 # run stress balance every timestep
md.transient.isstressbalance = 1
md.transient.ismasstransport = 1
md.transient.isthermal = 0
md.transient.issmb = 0

# define the timestepping scheme (YEARS)
md.timestepping.time_step = time_step
md.timestepping.start_time = 0.0
md.timestepping.final_time = time_step * final_time

md.settings.output_frequency = time_step # <<<


# Set up output settings
md.transient.requested_outputs = ['default',
                                    'Thickness',
                                    'Surface',
                                    'Base',
]

print("sb_coupling_frequency:", md.settings.sb_coupling_frequency)
print("output_frequency:", md.settings.output_frequency)
print("isstressbalance:", md.transient.isstressbalance)
print("ismasstransport:", md.transient.ismasstransport)

print(f"\nΔt (yr): {md.timestepping.time_step}\nTfinal (yr): {md.timestepping.final_time}\n≈nsteps: {int(md.timestepping.final_time/md.timestepping.time_step)}")

print(f"\n===== Solving Transient {flowequation_type} =====")
md = solve(md, 'Transient')
# plot the surface velocities #plotdoc
plotmodel(md, 'data', md.results.TransientSolution[19].Vel, 'layer', 5, 'figure', 5)

# # # Check transient fields   
# plot_transient_fields(md)

# save the given model
print(f"\n===== Saving Transient Solution =====")
output_filename = f"Transient_{md.miscellaneous.name}_{final_time=}_yrs_timestep={time_step:.5f}_yrs.nc"

# export_netCDF(md, output_filename)
print(f"✓ Full results saved to {output_filename}")

# ## check max velocity evolution
# plot_max_velocity_from_netcdf(output_filename)

print("=======================================================================================")

print("\nAvailable results in (last time step) md.results.TransientSolution:")
transient_solution = md.results.TransientSolution
# print(stress_solution)

# Full solution arrays for last time step
vx_full_transient = transient_solution.Vx[:,-1]
vy_full_transient = transient_solution.Vy[:,-1]
vz_full_transient = transient_solution.Vz[:,-1]
vel_full_transient = transient_solution.Vel[:,-1]
Pressure_full_transient = transient_solution.Pressure[:,-1]
Thickness_full_transient = transient_solution.Thickness[:,-1]
Surface_full_transient = transient_solution.Surface[:,-1]
Base_full_transient = transient_solution.Base[:,-1]

basal_idx = np.where(md.mesh.vertexonbase == 1)[0]
vx_basal_transient = vx_full_transient[basal_idx]
vy_basal_transient = vy_full_transient[basal_idx]
vz_basal_transient = vz_full_transient[basal_idx]

surface_idx = np.where(md.mesh.vertexonsurface == 1)[0]
vx_surface_transient = vx_full_transient[surface_idx]
vy_surface_transient = vy_full_transient[surface_idx]
vz_surface_transient = vz_full_transient[surface_idx]

# ------------------------------------------------------------------
# Quick print‑outs
# ------------------------------------------------------------------
print(
    f"Surface velocity ranges (m a⁻¹):\n"
    f"  vx_surface: [{vx_surface_transient.min():.5f}, {vx_surface_transient.max():.5f}]\n"
    f"  vy_surface: [{vy_surface_transient.min():.5f}, {vy_surface_transient.max():.5f}]\n"
    f"  vz_surface: [{vz_surface_transient.min():.5f}, {vz_surface_transient.max():.5f}]"
)
print(
    "Basal velocity ranges (m a⁻¹):\n"
    f"  vx_basal: [{vx_basal_transient.min():.5f}, {vx_basal_transient.max():.5f}]\n"
    f"  vy_basal: [{vy_basal_transient.min():.5f}, {vy_basal_transient.max():.5f}]\n"
    f"  vz_basal: [{vz_basal_transient.min():.5f}, {vz_basal_transient.max():.5f}]"
)

print(
    "Vel ranges (m a⁻¹):\n"
    f"  vel: [{vel_full_transient.min():.5f}, {vel_full_transient.max():.5f}]\n"
)

print(
    "Pressure ranges (Pa):"f"[{Pressure_full_transient.min():.5f}, {Pressure_full_transient.max():.5f}]\n"
)

print(
    "Thickness ranges (m):"f"[{Thickness_full_transient.min():.5f}, {Thickness_full_transient.max():.5f}]\n"
)
print("=======================================================================================")

# breakpoint()
