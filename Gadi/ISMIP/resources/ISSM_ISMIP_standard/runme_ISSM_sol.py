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


ParamFile = 'IsmipA_ISSM_sol.py'
# ParamFile = 'IsmipA.py'

steps = [1, 2, 3, 4, 5, 6, 7]


#Run Steps

#Mesh Generation #1
if 1 in steps:
    print("Now generating the mesh")
    #initialize md as a new model help(model)
    md = model()
    # generate a squaremesh help(squaremesh)
    # Side is 80 km long with 20 points
    if ParamFile == 'IsmipA_ISSM_sol.py':
        md = squaremesh(md, 80000, 80000, 20, 20)
    elif ParamFile == 'IsmipF_cor.py':
        md = squaremesh(md, 100000, 100000, 30, 30)

    # plot the given mesh plotdoc()
    plotmodel(md, 'data', 'mesh', 'figure', 1)
    # save the given model
    # export_netCDF(md, "ISMIP-Mesh_generation.nc")

#Masks #2
if 2 in steps:
    print("Setting the masks")
    # load the preceding step help(loadmodel)
    # path is given by the organizer with the name of the given step
    # md = loadmodel("ISMIP-Mesh_generation.nc")
    # set the mask help(setmask)
    # all MISMIP nodes are grounded
    md = setmask(md, '', '')
    # plot the given mask #md.mask to locate the field
    plotmodel(md, 'data', md.mask.ocean_levelset, 'figure', 2)
    # save the given model
    # export_netCDF(md, "ISMIP-SetMask.nc")

#Parameterization #3
if 3 in steps:
    print("Parameterizing")
    # load the preceding step #help loadmodel
    # path is given by the organizer with the name of the given step
    # md = loadmodel("ISMIP-SetMask.nc")
    # parametrize the model # help parameterize
    # you will need to fil-up the parameter file (given by the
    # ParamFile variable)
    md = parameterize(md, ParamFile)
    # save the given model
    # export_netCDF(md, "ISMIP-Parameterization.nc")

#Extrusion #4
if 4 in steps:
    print("Extruding")
    # load the preceding step #help loadmodel
    # path is given by the organizer with the name of the given step
    # md = loadmodel("ISMIP-Parameterization.nc")
    # vertically extrude the preceding mesh #help extrude
    # only 5 layers exponent 1
    md = md.extrude(5, 1)
    # plot the 3D geometry #plotdoc
    plotmodel(md, 'data', md.geometry.base, 'figure', 3)
    # save the given model
    # export_netCDF(md, "ISMIP-Extrusion.nc")

#Set the flow computing method #5
if 5 in steps:
    print("setting flow approximation")
    # load the preceding step #help loadmodel
    # path is given by the organizer with the name of the given step
    # md = loadmodel("ISMIP-Extrusion.nc")
    # set the approximation for the flow computation #help setflowequation
    # We will be using the Higher Order Model (HO)
    md = setflowequation(md, 'HO', 'all')
    # save the given model
    # export_netCDF(md, "ISMIP-SetFlow.nc")

#Set Boundary Conditions #6
if 6 in steps:
    print("setting boundary conditions")
    # load the preceding step #help loadmodel
    # path is given by the organizer with the name of the given step
    # md = loadmodel("ISMIP-SetFlow.nc")
    # dirichlet boundary condition are known as SPCs
    # ice frozen to the base, no velocity   #md.stressbalance
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
    if ParamFile == 'IsmipF_cor.py':
        # if we are dealing with IsmipF the solution is in masstransport
        md.masstransport.vertex_pairing = md.stressbalance.vertex_pairing

    # save the given model
    # export_netCDF(md, "ISMIP-BoundaryCondition.nc")

#Solving #7
if 7 in steps:
    print("running the solver for the A case")
    # load the preceding step #help loadmodel
    # path is given by the organizer with the name of the given step
    # md = loadmodel("ISMIP-BoundaryCondition.nc")
    # Set cluster #md.cluster
    # generic parameters #help generic
    # set only the name and number of process
    md.cluster = generic('name', gethostname(), 'np', 2)
    # Set which control message you want to see #help verbose
    md.verbose = verbose('convergence', True)
    # Solve #help solve
    # we are solving a StressBalance
    md = solve(md, 'Stressbalance')

    #=======================================================================================
    print("\nAvailable results in md.results.StressbalanceSolution:")
    stress_solution = md.results.StressbalanceSolution
    # print(stress_solution)

    vx_full = stress_solution.Vx
    vy_full = stress_solution.Vy
    vz_full = stress_solution.Vz
    vel_full = stress_solution.Vel

    basal_idx = np.where(md.mesh.vertexonbase == 1)[0]
    vx_basal = vx_full[basal_idx]
    vy_basal = vy_full[basal_idx]
    vz_basal = vz_full[basal_idx]

    surface_idx = np.where(md.mesh.vertexonsurface == 1)[0]
    vx_surface = vx_full[surface_idx]
    vy_surface = vy_full[surface_idx]
    vz_surface = vz_full[surface_idx]

    # ------------------------------------------------------------------
    # Quick velocity print‑outs
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
    #=======================================================================================


    # save the given model
    # export_netCDF(md, "ISMIP-StressBalance.nc")
    # plot the surface velocities #plotdoc
    plotmodel(md, 'data', md.results.StressbalanceSolution.Vel, 'figure', 4)




#Solving #8
if 8 in steps:
    print("running the solver for the F case")
    # load the preceding step #help loadmodel
    # path is given by the organizer with the name of the given step
    # md = loadmodel("ISMIP-BoundaryCondition.nc")
    # Set cluster #md.cluster
    # generic parameters #help generic
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
    md.timestepping.time_step = 1
    # give final_time (20*4 years time_steps)
    md.timestepping.final_time = 1 * 20
    # Solve #help solve
    # we are solving a TransientSolution
    md = solve(md, 'Transient')
    # save the given model
    # export_netCDF(md, "ISMIP-Transient.nc")
    # plot the surface velocities #plotdoc
    plotmodel(md, 'data', md.results.TransientSolution[19].Vel, 'layer', 5, 'figure', 5)