import numpy as np
from model import *
from squaremesh import squaremesh
from plotmodel import plotmodel
from export_netCDF import export_netCDF
from loadmodel import loadmodel
from setmask import setmask
from parameterize import parameterize
from setflowequation import setflowequation
from socket import gethostname
from solve import solve
from plotdoc import plotdoc
import os

ParamFile = 'IsmipF.py'
# ParamFile = 'IsmipG.py'

# ParamFile = 'IsmipA_ISSM_sol.py'


steps = [1, 2, 3, 4, 5, 6, 7, 8]
x_max = 100000
y_max = 100000

x_nodes = 30
y_nodes = 30

# No sliding
Scenario = "S1"
# # sliding
# Scenario = "S3"


#Mesh Generation #1
if 1 in steps:
    print("\n===== Generating the mesh =====")
    #initialize md as a new model help(model)
    md = model()

    # generate a squaremesh help(squaremesh)
    if ParamFile == 'IsmipA_ISSM_sol':
        md = squaremesh(md, 80000, 80000, 20, 20)
    elif ParamFile == 'IsmipF.py':
        md = squaremesh(md, x_max, y_max, x_nodes, y_nodes)

    # print("\n===== Plotting mesh =====")
    # plot the given mesh plotdoc()
    # plotmodel(md, 'data', 'mesh', 'figure', 1)

    os.remove("ISMIP-Mesh_generation.nc") 
    export_netCDF(md, "ISMIP-Mesh_generation.nc")


#Masks #2
if 2 in steps:
    print("\n===== Setting the masks =====")
    md = loadmodel("ISMIP-Mesh_generation.nc")
    # all nodes are grounded
    md = setmask(md, '', '')

    # print("\n===== Plotting mask=====")
    # plotmodel(md, 'data', md.mask.ocean_levelset, 'figure', 2)

    os.remove("ISMIP-SetMask.nc") 
    export_netCDF(md, "ISMIP-SetMask.nc")


#Parameterization #3
if 3 in steps:
    print("\n===== Parameterizing =====")
    md = loadmodel("ISMIP-SetMask.nc")

    md = parameterize(md, ParamFile)

    os.remove("ISMIP-Parameterization.nc")
    export_netCDF(md, "ISMIP-Parameterization.nc")


#Extrusion #4
if 4 in steps:
    print("\n===== Extruding =====")
    md = loadmodel("ISMIP-Parameterization.nc")
    # vertically extrude the preceding mesh #help extrude
    # only 5 layers exponent 1
    md = md.extrude(5, 1)

    # print("\n===== Plotting base geometry =====")
    # plot the 3D geometry #plotdoc
    # plotmodel(md, 'data', md.geometry.base, 'figure', 3)

    os.remove("ISMIP-Extrusion.nc")
    export_netCDF(md, "ISMIP-Extrusion.nc")


#Set the flow computing method #5
if 5 in steps:
    print("\n===== Setting flow approximation: HO =====")
    md = loadmodel("ISMIP-Extrusion.nc")

    md = setflowequation(md, 'HO', 'all')

    os.remove("ISMIP-SetFlow.nc")
    export_netCDF(md, "ISMIP-SetFlow.nc")


#Set Boundary Conditions #6
if 6 in steps:
    print("\n===== Setting boundary conditions =====")
    md = loadmodel("ISMIP-SetFlow.nc")

    # ice frozen to the base, no velocity
    # SPCs are initialized at NaN one value per vertex
    md.stressbalance.spcvx = np.nan * np.ones((md.mesh.numberofvertices))
    md.stressbalance.spcvy = np.nan * np.ones((md.mesh.numberofvertices))
    md.stressbalance.spcvz = np.nan * np.ones((md.mesh.numberofvertices))

    if Scenario == "S1":
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
    
    if ParamFile == 'IsmipF.py':
        # if we are dealing with IsmipF the solution is in masstransport
        md.masstransport.vertex_pairing = md.stressbalance.vertex_pairing

    # save the given model
    os.remove("ISMIP-BoundaryCondition.nc")
    export_netCDF(md, "ISMIP-BoundaryCondition.nc")

#Solving #7
if 7 in steps:
    print("\n===== Running Stressbalance Solver =====")
    md = loadmodel("ISMIP-BoundaryCondition.nc")
    ## Set which control message you want to see #help verbose
    ## md.verbose = verbose('convergence', True)
    md = solve(md, 'Stressbalance')

    print("\n============================================================")

    print("\nAvailable results in md.results.StressbalanceSolution:")
    stress_solution = md.results.StressbalanceSolution
    # print(stress_solution)

    vx_full = stress_solution.Vx
    vy_full = stress_solution.Vy
    vz_full = stress_solution.Vz
    vel_full = stress_solution.Vel

    pressure = stress_solution.Pressure

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
        f"  vel: [{vel_full.min():.5f}, {vel_full.max():.5f}]"
    )
    print(f"Pressure ranges (Pa): [{pressure.min():.5f}, {pressure.max():.5f}]\n")

    print("\n============================================================")

    # save the given model
    os.remove("ISMIP-StressBalance.nc")
    export_netCDF(md, "ISMIP-StressBalance.nc")
    # plot the surface velocities #plotdoc
    plotmodel(md, 'data', md.results.StressbalanceSolution.Vel, 'figure', 4)


## Initialising velocity and pressure
# md = loadmodel("ISMIP-StressBalance.nc")

# # md.initialization.vx = md.results.StressbalanceSolution.Vx
# # md.initialization.vy = md.results.StressbalanceSolution.Vy
# # md.initialization.vz = md.results.StressbalanceSolution.Vz

# # md.initialization.pressure = md.results.StressbalanceSolution.Pressure


#Solving #8
if 8 in steps:
    print("\n===== Running Transient Solver =====")
    md = loadmodel("ISMIP-BoundaryCondition.nc")
    ## Set which control message you want to see #help verbose
    # md.verbose = verbose('convergence', True)

    md.transient.deactivateall()

    md.transient.isstressbalance = 1
    md.transient.ismasstransport = 1
    md.transient.isthermal = 0
    md.transient.issmb = 0

    # define the timestepping scheme
    md.timestepping.time_step = 1/12
    # give final_time (20 years * time_step)
    md.timestepping.final_time = 1 #* 3#0#0


    md = solve(md, 'Transient')

    os.remove("ISMIP-Transient.nc")
    export_netCDF(md, "ISMIP-Transient.nc")
    # plot the surface velocities #plotdoc
    plotmodel(md, 'data', md.results.TransientSolution[-1].Vel, 'layer', 5, 'figure', 5)


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
    vx_basal_transient = vx_full[basal_idx]
    vy_basal_transient = vy_full[basal_idx]
    vz_basal_transient = vz_full[basal_idx]

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


    breakpoint()


