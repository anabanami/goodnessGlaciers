import numpy as np
from model import model
from squaremesh import squaremesh
from plotmodel import plotmodel
from export_netCDF import export_netCDF
from loadmodel import loadmodel
from setmask import setmask
from parameterize import parameterize
from setflowequation import setflowequation
from socket import gethostname
from generic import generic
from solve import solve
from plotdoc import plotdoc

import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/ana/pyISSM/src')
import pyissm as issm
from pyissm import plot as iplt


# ParamFile = 'IsmipF.py'
# ParamFile = 'flat.py'
# ParamFile = 'singlewave.py'
ParamFile = 'coswave.py'

filename = os.path.splitext(ParamFile)[0] 
# print(f"{filename = }")

steps = [1, 2, 3, 4, 5, 6, 7, 8]
x_max = 100000
y_max = 100000

if filename == 'coswave':
    h_resolution_factor = 5
    v_resolution_factor = 5
else:
    h_resolution_factor = 2
    v_resolution_factor = 2


# Baseline number of layers
base_vertical_layers = 5

x_nodes = int(30 * h_resolution_factor)
y_nodes = int(30 * h_resolution_factor)

num_layers = int(base_vertical_layers * v_resolution_factor)


## EXPERIMENT
# No sliding + linear rheology
Scenario = "S1"
# # No sliding + non-linear rheology
# Scenario = "S2"
# # sliding + linear rheology
# Scenario = "S3"
# # sliding + non-linear rheology
# Scenario = "S4"


# TIME
timestep = 1/12
final_time = 300
output_frequency = 100

print("\n============================================================")
print(f"\nRunning {Scenario} with {filename}")
print(f"\n{h_resolution_factor = } and {v_resolution_factor = }")
print(f"\nNumber of nodes is {x_nodes} × {y_nodes} = {x_nodes * y_nodes}")
print(f"\nNumber of vertical layers: {num_layers}")
print("\n============================================================")


# Construct the new filename string
# Example: IsmipF_S1_1.0_1.0-Mesh_generation.nc
file_prefix = f"{filename}_{Scenario}_{h_resolution_factor}_{v_resolution_factor}"


#Mesh Generation #1
if 1 in steps:
    print("\n===== Generating the mesh =====")
    #initialize md as a new model help(model)
    md = model()

    if ParamFile == 'IsmipF.py':
        md = squaremesh(md, x_max, y_max, x_nodes, y_nodes)
        print(f"{md.mesh.numberofelements = }")
        print(f" Total bed area: {x_max} × {y_max} = {x_max * y_max}")
        print(f" mean element area = {(x_max * y_max) / md.mesh.numberofelements}")

    elif ParamFile == 'flat.py':
        md = squaremesh(md, x_max, y_max, x_nodes, y_nodes)
        print(f"{md.mesh.numberofelements = }")
        print(f" Total bed area: {x_max} × {y_max} = {x_max * y_max}")
        print(f" mean element area = {(x_max * y_max) / md.mesh.numberofelements}")

    elif ParamFile == 'singlewave.py':
        md = squaremesh(md, x_max, y_max, x_nodes, y_nodes)
        print(f"{md.mesh.numberofelements = }")
        print(f" Total bed area: {x_max} × {y_max} = {x_max * y_max}")
        print(f" mean element area = {(x_max * y_max) / md.mesh.numberofelements}")

    elif ParamFile == 'coswave.py':
        md = squaremesh(md, x_max, y_max, x_nodes, y_nodes)
        print(f"{md.mesh.numberofelements = }")
        print(f" Total bed area: {x_max} × {y_max} = {x_max * y_max}")
        print(f" mean element area = {(x_max * y_max) / md.mesh.numberofelements}")



    print("\n===== Plotting mesh =====")
    md_mesh, md_x, md_y, md_elements, md_is3d = issm.model.mesh.process_mesh(md)
    # iplt.plot_mesh2d(md_mesh, show_nodes = True)
    # plt.title("Full mesh") 
    # plt.savefig(f"{file_prefix}_mesh.png")
    # plt.close()
    # # plt.show()

    # convert the vertex on boundary array into boolean
    boundary_mask = md.mesh.vertexonboundary.astype(bool)
    x_boundaries = md.mesh.x[boundary_mask]
    y_boundaries = md.mesh.y[boundary_mask]

    print("\n===== Plotting mesh and highlighting vertex boundaries =====")
    # iplt.plot_mesh2d(md_mesh, show_nodes = True)
    # plt.scatter(x_boundaries, y_boundaries, label='boundaries')
    # plt.legend()
    # plt.title("Mesh boundaries") 
    # plt.savefig(f"{file_prefix}_mesh_boundaries.png")
    # plt.close()
    # # plt.show()

    # Path(f"{file_prefix}-Mesh_generation.nc").unlink(missing_ok=True)
    # export_netCDF(md, f"{file_prefix}-Mesh_generation.nc")


#Masks #2
if 2 in steps:
    print("\n===== Setting the masks =====")
    # md = loadmodel(f"{file_prefix}-Mesh_generation.nc")
    
    nv, ne = md.mesh.numberofvertices, md.mesh.numberofelements

    # # all nodes are grounded
    # md = setmask(md, '', '')

    print("\n===== Plotting mask=====")
    md.mask.ice_levelset = - np.ones(nv) #Ice is present if md.mask.ice_levelset is negative
    md.mask.ocean_levelset = np.ones(nv) # grounded ice if positive (OR floating ice if negative)

    # # plot the given mask #md.mask to locate the field
    # fig, ax = plt.subplots(figsize = (7, 7))
    # iplt.plot_model_elements(md,
    #                          md.mask.ice_levelset,
    #                          md.mask.ocean_levelset, 
    #                          type='grounded_ice_elements', 
    #                          ax = ax
    # )

    # plt.title("Set mask - grounded ice elements") 
    # plt.savefig(f"{file_prefix}_grounded_ice_elements.png")
    # plt.close()
    # # plt.show()

    # Path(f"{file_prefix}-SetMask.nc").unlink(missing_ok=True)
    # export_netCDF(md, f"{file_prefix}-SetMask.nc")


#Parameterisation #3
if 3 in steps:
    print("\n===== Parameterising =====")
    # md = loadmodel(f"{file_prefix}-SetMask.nc")

    # make naming variables available to Paramfile
    md.miscellaneous.filename = filename
    md.miscellaneous.h_resolution_factor = h_resolution_factor
    md.miscellaneous.v_resolution_factor = v_resolution_factor
    md.miscellaneous.scenario = Scenario

    md = parameterize(md, ParamFile)
    
    print(f"\n{md.miscellaneous.filename = }")
    print(f"\n{md.miscellaneous.h_resolution_factor = }")
    print(f"\n{md.miscellaneous.v_resolution_factor = }")
    print(f"\n{md.miscellaneous.scenario = }")
    print(f"\nn = {md.materials.rheology_n[0]}")

    # iplt.plot_model_field(md, md.geometry.thickness, show_cbar = True)
    # plt.title("Ice thickness") 
    # plt.savefig(f"{file_prefix}_thickness_geometry_2D.png")
    # plt.close()
    # # plt.show()
    
    # Path(f"{file_prefix}-Parameterisation.nc").unlink(missing_ok=True)
    # export_netCDF(md, f"{file_prefix}-Parameterisation.nc")


#Extrusion #4
if 4 in steps:
    print("\n===== Extruding =====")
    # md = loadmodel(f"{file_prefix}-Parameterisation.nc")
    # vertically extrude the preceding mesh #help extrude
    # only 5 layers exponent 1
    md = md.extrude(num_layers, 1)

    print("\n===== Plotting base geometry =====")
    ## 3D plot
    # plotmodel(md, 'data', md.geometry.base, 'figure', 4,
    #         'figsize', [12, 12],
    #         'xlabel', 'x (m)',
    #         'ylabel', 'y (m)',
    # )

    # plt.title("3D model geometry") 
    # plt.savefig(f"{file_prefix}_geometry_3D.png")
    # plt.close()
    # # plt.show()

    # # 2D plot
    # iplt.plot_model_field(md, md.geometry.base, layer=1, show_cbar = True)
    # plt.title("Base geometry") 
    # plt.savefig(f"{file_prefix}_base_geometry_2D.png")
    # plt.close()
    # # plt.show()

    # Path(f"{file_prefix}-Extrusion.nc").unlink(missing_ok=True)
    # export_netCDF(md, f"{file_prefix}-Extrusion.nc")


#Set the flow computing method #5
if 5 in steps:
    print("\n===== Setting flow approximation: HO =====")
    # md = loadmodel(f"{file_prefix}-Extrusion.nc")

    md = setflowequation(md, 'HO', 'all')

    # Path(f"{file_prefix}-SetFlow.nc").unlink(missing_ok=True)
    # export_netCDF(md, f"{file_prefix}-SetFlow.nc")


#Set Boundary Conditions #6
if 6 in steps:
    print("\n===== Setting boundary conditions =====")
    # md = loadmodel(f"{file_prefix}-SetFlow.nc")

    # ice frozen to the base, no velocity
    # SPCs are initialized at NaN one value per vertex
    md.stressbalance.spcvx = np.nan * np.ones((md.mesh.numberofvertices))
    md.stressbalance.spcvy = np.nan * np.ones((md.mesh.numberofvertices))
    md.stressbalance.spcvz = np.nan * np.ones((md.mesh.numberofvertices))

    if Scenario in ("S1", "S2"):
        # extract the nodenumbers at the base #md.mesh.vertexonbase
        basalnodes = np.nonzero(md.mesh.vertexonbase)
        # set the sliding to zero on the bed (Vx and Vy)
        md.stressbalance.spcvx[basalnodes] = 0.0
        md.stressbalance.spcvy[basalnodes] = 0.0

    # periodic boundaries have to be fixed on the sides
    # Find the indices of the sides of the domain, for x and then for y
    # for x
    # create maxX, list of indices where x is equal to max of x (use >> help find)
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

    # set mass transport pairing for periodic boundaries
    md.masstransport.vertex_pairing = md.stressbalance.vertex_pairing

    # save the given model
    Path(f"{file_prefix}-BoundaryCondition.nc").unlink(missing_ok=True)
    export_netCDF(md, f"{file_prefix}-BoundaryCondition.nc")

# # Solving #7
# if 7 in steps:
#     print("\n===== Running Stressbalance Solver =====")
#     md = loadmodel(f"{file_prefix}-BoundaryCondition.nc")
#     ## Set which control message you want to see #help verbose
#     ## md.verbose = verbose('convergence', True)
#     md.cluster=generic('name', gethostname(), 'np', 4)
#     md = solve(md, 'Stressbalance')

#     print("\n============================================================")

#     print("\nAvailable results in md.results.StressbalanceSolution:")
#     stress_solution = md.results.StressbalanceSolution
#     # print(stress_solution)

#     vx_full = stress_solution.Vx
#     vy_full = stress_solution.Vy
#     vz_full = stress_solution.Vz
    
#     vel_full = stress_solution.Vel

#     pressure = stress_solution.Pressure

#     basal_idx = np.where(md.mesh.vertexonbase == 1)[0]
#     vx_basal = vx_full[basal_idx]
#     vy_basal = vy_full[basal_idx]
#     vz_basal = vz_full[basal_idx]

#     surface_idx = np.where(md.mesh.vertexonsurface == 1)[0]
#     vx_surface = vx_full[surface_idx]
#     vy_surface = vy_full[surface_idx]
#     vz_surface = vz_full[surface_idx]

#     # ------------------------------------------------------------------
#     # Quick print‑outs
#     # ------------------------------------------------------------------
#     print(
#         f"Surface velocity ranges (m a⁻¹):\n"
#         f"  vx_surface: [{vx_surface.min():.5f}, {vx_surface.max():.5f}]\n"
#         f"  vy_surface: [{vy_surface.min():.5f}, {vy_surface.max():.5f}]\n"
#         f"  vz_surface: [{vz_surface.min():.5f}, {vz_surface.max():.5f}]"
#     )
#     print(
#         "Basal velocity ranges (m a⁻¹):\n"
#         f"  vx_basal: [{vx_basal.min():.5f}, {vx_basal.max():.5f}]\n"
#         f"  vy_basal: [{vy_basal.min():.5f}, {vy_basal.max():.5f}]\n"
#         f"  vz_basal: [{vz_basal.min():.5f}, {vz_basal.max():.5f}]"
#     )

#     print(
#         "Vel ranges (m a⁻¹):\n"
#         f"  vel: [{vel_full.min():.5f}, {vel_full.max():.5f}]"
#     )
#     print(f"Pressure ranges (Pa): [{pressure.min():.5f}, {pressure.max():.5f}]\n")

#     print("\n============================================================")

#     # # save the given model
#     # Path(f"{file_prefix}-StressBalance.nc").unlink(missing_ok=True)
#     # export_netCDF(md, f"{file_prefix}-StressBalance.nc")
#     # plot the surface velocities #plotdoc
#     plotmodel(md, 'data', md.results.StressbalanceSolution.Vel, 'figure', 4)
#     plt.savefig(f"{file_prefix}_stress_solution_Vel.png")
#     plt.close()
#     # plt.show()

# #     plt.quiver(md.mesh.x, md.mesh.y, md.results.StressbalanceSolution.Vx, md.results.StressbalanceSolution.Vy)
# #     plt.savefig("quiver_Vx_Vy.png")
# #     plt.close()
# #     # plt.show()

# # breakpoint()

#Solving #8
if 8 in steps:
    print("\n===== Running Transient Solver =====")
    md = loadmodel(f"{file_prefix}-BoundaryCondition.nc")
    ## Set which control message you want to see #help verbose
    # md.verbose = verbose('convergence', True)
    md.cluster=generic('name', gethostname(), 'np', 4)

    md.transient.deactivateall()
    md.settings.sb_coupling_frequency = 1 # run stress balance every timestep #????
    md.transient.isstressbalance = 1
    md.transient.ismasstransport = 1
    md.transient.isthermal = 0
    md.transient.issmb = 0

    # breakpoint()

    ####################################################################
    # Scale timestep by the most restrictive (largest) resolution factor
    combined_res_factor = max(h_resolution_factor, v_resolution_factor)

    ##  define the timestepping scheme
    md.timestepping.time_step = timestep / combined_res_factor # length of a single step

    # ~* MY LOGIC *~
    # If combined_res_factor doubles (finer mesh), timestep must halve.
    # combined_res_factor = 2 => timestep / 2 = 0.5 * timestep
    # If combined_res_factor halves (coarser mesh), timestep can double.
    # combined_res_factor = 0.5 => timestep / 0.5 = 2 * timestep

    md.timestepping.final_time = final_time
    md.settings.output_frequency = int(output_frequency * combined_res_factor)
    ####################################################################


    md = solve(md, 'Transient')

    Path(f"{file_prefix}-Transient.nc").unlink(missing_ok=True)
    export_netCDF(md, f"{file_prefix}-Transient.nc")
    # plot the surface velocities #plotdoc
    plotmodel(md, 'data', md.results.TransientSolution[-1].Vel, 'layer', 5, 'figure', 5)
    plt.savefig(f"{file_prefix}_transient_solution_Vel_layer5_last_timestep.png")
    plt.close()
    # plt.show()

    print("\n============================================================")
    
    print("\nAvailable results in md.results.TransientSolution:")
    transient_solution = md.results.TransientSolution
    # print(stress_solution)

    vx_full_transient = transient_solution.Vx
    vy_full_transient = transient_solution.Vy
    vz_full_transient = transient_solution.Vz

    vel_full_transient = transient_solution.Vel

    pressure_transient = transient_solution.Pressure

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
        f"  vel: [{vel_full_transient.min():.5f}, {vel_full_transient.max():.5f}]"
    )
    print(f"Pressure ranges (Pa): [{pressure_transient.min():.5f}, {pressure_transient.max():.5f}]\n")

    print("\n============================================================")

    plt.quiver(md.mesh.x, md.mesh.y, md.results.TransientSolution[-1].Vx, md.results.TransientSolution[-1].Vy)
    plt.savefig("quiver_Vx_Vy.png")
    plt.close()
    # plt.show()

    print("\n============================================================")
    print(f"\nFINISHED {Scenario} with {file_prefix} and {h_resolution_factor = }")
    print(f"\nNumber of nodes is {x_nodes} × {y_nodes} = {x_nodes * y_nodes}")
    print(f"\n{final_time = } and {timestep = }")
    print("\n============================================================")

    breakpoint()


