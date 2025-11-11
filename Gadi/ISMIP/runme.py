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
from verbose import verbose

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
    h_resolution_factor = 9
    v_resolution_factor = 2
else:
    h_resolution_factor = 2
    v_resolution_factor = 2


# Baseline number of layers
base_vertical_layers = 5

x_nodes = int(30 * h_resolution_factor)
y_nodes = int(30 * h_resolution_factor)

num_layers = int(base_vertical_layers * v_resolution_factor)


# EXPERIMENT
# # No sliding + linear rheology
Scenario = "S1"
# # # No sliding + non-linear rheology
# Scenario = "S2"
# # # # Sliding + linear rheology
# Scenario = "S3"
# # # # Sliding + non-linear rheology
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
if filename == 'coswave':
    # For coswave, we'll get the wavelength factor from the parameter file
    # This will be updated after parameterization step
    file_prefix = f"{filename}_{Scenario}_{h_resolution_factor}_{v_resolution_factor}"
else:
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


    md_mesh, md_x, md_y, md_elements, md_is3d = issm.model.mesh.process_mesh(md)

    # convert the vertex on boundary array into boolean
    boundary_mask = md.mesh.vertexonboundary.astype(bool)
    x_boundaries = md.mesh.x[boundary_mask]
    y_boundaries = md.mesh.y[boundary_mask]

    # Path(f"{file_prefix}-Mesh_generation.nc").unlink(missing_ok=True)
    # export_netCDF(md, f"{file_prefix}-Mesh_generation.nc")


#Masks #2
if 2 in steps:
    print("\n===== Setting the masks =====")
    # md = loadmodel(f"{file_prefix}-Mesh_generation.nc")
    
    nv, ne = md.mesh.numberofvertices, md.mesh.numberofelements

    md.mask.ice_levelset = - np.ones(nv) #Ice is present if md.mask.ice_levelset is negative
    md.mask.ocean_levelset = np.ones(nv) # grounded ice if positive (OR floating ice if negative)

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
    
    # Update file_prefix with wavelength info for coswave
    if filename == 'coswave':
        # Extract wavelength factor from the parameterized model
        # The coswave.py file constructs the name with wavelength info
        transient_name = md.miscellaneous.name
        # Get the base name (everything before '-Transient')
        base_name = transient_name.split('-Transient')[0]
        file_prefix = base_name
        print(f"\nUpdated file_prefix for coswave: {file_prefix}")
    
    print(f"\n{md.miscellaneous.filename = }")
    print(f"\n{md.miscellaneous.h_resolution_factor = }")
    print(f"\n{md.miscellaneous.v_resolution_factor = }")
    print(f"\n{md.miscellaneous.scenario = }")
    print(f"\nn = {md.materials.rheology_n[0]}")

    # Path(f"{file_prefix}-Parameterisation.nc").unlink(missing_ok=True)
    # export_netCDF(md, f"{file_prefix}-Parameterisation.nc")


#Extrusion #4
if 4 in steps:
    print("\n===== Extruding =====")
    # md = loadmodel(f"{file_prefix}-Parameterisation.nc")
    md = md.extrude(num_layers, 1)

    # Path(f"{file_prefix}-Extrusion.nc").unlink(missing_ok=True)
    # export_netCDF(md, f"{file_prefix}-Extrusion.nc")


    ##########################################################################################################

    # Check mesh quality
    print(f"\n===== Mesh Quality Check =====")
    print(f"Number of 3D elements: {md.mesh.numberofelements}")
    print(f"Number of 3D vertices: {md.mesh.numberofvertices}")

    # Calculate element aspect ratios
    dx = x_max / x_nodes
    dy = y_max / y_nodes  
    dz = 1000 / num_layers  # Assuming 1000m total thickness
    aspect_ratio_xy = max(dx, dy) / min(dx, dy)
    aspect_ratio_xz = dx / dz
    print(f"Aspect ratios: xy={aspect_ratio_xy:.2f}, xz={aspect_ratio_xz:.2f}")

    if aspect_ratio_xz > 10:
        print("WARNING: Very high aspect ratio detected! Consider adjusting mesh resolution.")

    ##########################################################################################################
    # Extract and plot bed profile along centerline (like phase_analysis.py) 
    center_y = np.nanmax(md.mesh.y) / 2.0
    # Find the closest actual mesh line to center
    unique_y = np.unique(md.mesh.y)
    closest_y_idx = np.argmin(np.abs(unique_y - center_y))
    actual_y = unique_y[closest_y_idx]
    tolerance = 1e-5
    profile_indices = np.where(np.abs(md.mesh.y - actual_y) < tolerance)[0]
    
    print(f"Center Y: {center_y}, Actual Y: {actual_y}, Found {len(profile_indices)} profile points")
    if len(profile_indices) == 0:
        print(f"Y range: [{np.min(md.mesh.y)}, {np.max(md.mesh.y)}]")

    if len(profile_indices) > 0:
        print("Extracting and plotting bed profile...")
        x_coords = md.mesh.x[profile_indices]
        bed_coords = md.geometry.base[profile_indices]

        sorted_order = np.argsort(x_coords)
        x_profile = x_coords[sorted_order]
        bed_profile = bed_coords[sorted_order]

        plt.figure(figsize=(12, 6))
        plt.plot(x_profile / 1000, bed_profile, 'k-', linewidth=2)
        plt.xlabel('Distance along X-axis (km)')
        plt.ylabel('Bed Elevation (m)')
        plt.title(f'Bed Profile at Y = {center_y/1000:.1f} km')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{file_prefix}_bed_profile.png",dpi=500, bbox_inches='tight')
        plt.show()
        plt.close()


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

    ##########################################################################################################

    # verify the pairing:
    print(f"Vertex pairing shape: {md.stressbalance.vertex_pairing.shape}")
    print(f"First few pairs: {md.stressbalance.vertex_pairing[:10,:]}")

    # Ensure pairing indices are within bounds
    max_idx = md.stressbalance.vertex_pairing.max()
    if max_idx > md.mesh.numberofvertices:
        print(f"ERROR: Vertex pairing has invalid indices! Max: {max_idx}, vertices: {md.mesh.numberofvertices}")

    ##########################################################################################################


# # Solving #7
# if 7 in steps:
#     print("\n===== Running Stressbalance Solver =====")
#     md = loadmodel(f"{file_prefix}-BoundaryCondition.nc")
#     ## Set which control message you want to see #help verbose
#     ## md.verbose = verbose('convergence', True)
#     md.cluster=generic('name', gethostname(), 'np', 8)
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
# breakpoint()

#Solving #8
if 8 in steps:
    print("\n===== Running Transient Solver =====")
    md = loadmodel(f"{file_prefix}-BoundaryCondition.nc")

    ##########################################################################################################

    import warnings
    warnings.filterwarnings('error')  # Turn warnings into errors

    # Check for any NaN or Inf values
    for field in ['thickness', 'surface', 'base']:
        data = getattr(md.geometry, field)
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            print(f"WARNING: NaN or Inf in geometry.{field}")

    ##########################################################################################################


    # md.verbose = verbose('all') 
    # md.verbose = verbose('convergence', True)
    md.cluster=generic('name', gethostname(), 'np', 8)

    md.transient.deactivateall()
    md.settings.sb_coupling_frequency = 1 # run stress balance every timestep #????
    md.transient.isstressbalance = 1
    md.transient.ismasstransport = 1
    md.transient.isthermal = 0
    md.transient.issmb = 0

    # breakpoint()

    md.transient.requested_outputs = [
    'default','Vx','Vy','Vz', 'Vel','Pressure','Thickness','Surface','Base','StrainRatexx','StrainRateyy','StrainRatexy'
    ]

    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
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
    # TO BE ABLE TO RESTART THE MODEL IF REACHING WALLTIME:
    md.settings.checkpoint_frequency = md.settings.output_frequency
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    # breakpoint()

    ##########################################################################################################
    print("--- Running final sanity checks ---")
    fields_to_check = {
        'geometry.base': md.geometry.base,
        'geometry.surface': md.geometry.surface,
        'geometry.thickness': md.geometry.thickness,
        'friction.coefficient': md.friction.coefficient,
        'materials.rheology_B': md.materials.rheology_B
    }

    for name, field in fields_to_check.items():
        if np.any(np.isnan(field)):
            print(f"ERROR: NaN detected in {name}")
        if np.any(np.isinf(field)):
            print(f"ERROR: Inf detected in {name}")
    print("--- Sanity checks complete ---")

    ##########################################################################################################

    md = solve(md, 'Transient')

    Path(f"{file_prefix}-Transient.nc").unlink(missing_ok=True)
    export_netCDF(md, f"{file_prefix}-Transient.nc")
    # # plot the surface velocities #plotdoc
    # plotmodel(md, 'data', md.results.TransientSolution[-1].Vel, 'layer', 5, 'figure', 5)
    # plt.savefig(f"{file_prefix}_transient_solution_Vel_layer5_last_timestep.png")
    # plt.show()
    # plt.close()

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
    print(f"\nFINISHED {Scenario} with {file_prefix} and {h_resolution_factor = }")
    print(f"\nNumber of nodes is {x_nodes} × {y_nodes} = {x_nodes * y_nodes}")
    print(f"\n{final_time = } and {timestep = }")
    print("\n============================================================")

    # breakpoint()


