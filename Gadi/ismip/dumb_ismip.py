import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import os
import sys
sys.path.append('/home/ana/pyISSM/src')
import pyissm as issm
from pyissm import plot as iplt
from model import model 
from bamgflowband import bamgflowband
from setflowequation import setflowequation
from socket import gethostname
from frictionweertman import frictionweertman
from export_netCDF import export_netCDF
from solve import solve

hostname = gethostname()
# Check if we're on the Gadi cluster
if 'gadi' in hostname.lower():
    # We're on Gadi
    if 'PBS_NCPUS' in os.environ:
        num_processors = int(os.environ.get('PBS_NCPUS'))
        print(f"\nGadi cluster detected ({hostname}). Using {num_processors} processors from PBS_NCPUS.")
else:
    # We're on local machine - always use 1 processor
    num_processors = 1
    print(f"\nLocal environment detected ({hostname}). Using 1 processor.")

# create mesh
md = model()
L = 5e3
x = np.linspace(0, L, 50)
# print(f"{np.shape(x)=}")
omega = 2 * np.pi / L
alpha_deg = 0.5 # as per pattyn 2008
alpha = np.deg2rad(alpha_deg) # radians

s = -x * np.tan(alpha)
# print(f"{np.shape(s)=}")
b = s - 1000 + 500 * np.sin(omega * x)
# print(f"{np.shape(b)=}")
h = s - b
# print(f"{np.shape(h)=}")

md = bamgflowband(md, x, s, b, 'hmax', 80, 'nvert', 10)
nv = md.mesh.numberofvertices
ne = md.mesh.numberofelements

# visual check
md_mesh, md_x, md_y, md_elements, md_is3d = issm.model.mesh.process_mesh(md)
iplt.plot_mesh2d(md_mesh, show_nodes = True)
plt.axis('equal')
plt.show()

md.geometry.surface = -md.mesh.x * np.tan(alpha)
md.geometry.bed = md.geometry.surface - (1000 + 500 * np.sin(omega * md.mesh.x))
md.geometry.base = md.geometry.bed
md.geometry.thickness = md.geometry.surface - md.geometry.bed

# # constants and material properties
md.constants.yts = 31556926 # s/y

# # rheology
rheology_n = 3
A = 1e-16 / md.constants.yts # from table 1 in Pattyn 2008
md.materials.rheology_B = A ** (-1/rheology_n) * np.ones(nv)
md.materials.rheology_n = rheology_n * np.ones(ne)
md.materials.rheology_law = "BuddJacka"
md.materials.rho_ice = 910  # kg/m^3

# Friction parameters
# BUDD
md.friction.p = np.ones(ne)
md.friction.q = np.ones(ne)
# C_realistic = 1e-3
C_realistic = 1e5
md.friction.coefficient = np.full(nv, C_realistic)

## WEERTMAN
# beta2 = 2000 # Pa a m^-1 
# beta2_sec = beta2 * md.constants.yts # Pa s m^-1 

# md.friction = frictionweertman()
# md.friction.C = np.full(nv, beta2_sec**(-1))
# md.friction.m = np.ones(ne)

# print(f"{md.friction.C.shape = }")
# print(f"{md.friction.C.min() = }, {md.friction.C.max()}")
# print(f"{md.friction.m.shape = }")

# flow equation
md = setflowequation(md, 'FS', 'all')


# velocity initialisation
md.initialization.vx = np.zeros(nv)
md.initialization.vy = np.zeros(nv)
md.initialization.vel = np.zeros(nv)
# hydrostatic seed
md.initialization.pressure = (md.constants.g * md.materials.rho_ice * 
							(md.geometry.surface - md.mesh.y))
# temperature
md.initialization.temperature = (273.15 - 20) * np.ones(nv)

# what kind of system?
md.mask.ocean_levelset = np.ones(nv)
md.mask.ice_levelset = -np.ones(nv)

# basal forcing
md.basalforcings.floatingice_melting_rate = np.zeros(nv)
md.basalforcings.groundedice_melting_rate = np.zeros(nv)

# mass balance
md.smb.mass_balance = 0.5 * np.ones(nv)

# Boundary conditions 
# ( temperature, thickness, velocity, 
# referential (tells ISSM “here’s the local 2D basis” for the stress‐balance ), loadingforce)
md.thermal.spctemperature = np.full(nv, np.nan)
md.masstransport.spcthickness = np.full(nv, np.nan)
md.stressbalance.spcvx = np.full(nv, np.nan)
md.stressbalance.spcvy = np.full(nv, np.nan)
md.stressbalance.spcvz = np.full(nv, np.nan)
md.stressbalance.referential = np.full((nv, 6), np.nan)
md.stressbalance.loadingforce = np.zeros((nv, 3))

# basal boundary is frozen
pos = np.where(md.mesh.vertexonbase == 1)[0]
md.stressbalance.spcvx[pos] = 0
md.stressbalance.spcvy[pos] = 0

# print("\n=== MESH LEFT AND RIGHT BOUNDARIES INVESTIGATION ===")
# # vertex pairing for periodic boundary conditions
# Find boundary vertices automatically
x_min, x_max = np.min(md.mesh.x), np.max(md.mesh.x)
tolerance = 1e-6

# Find left boundary vertices (inlet)
left_vertices = np.where(np.abs(md.mesh.x - x_min) < tolerance)[0]
# print(f"pre-sort{left_vertices}, {len(left_vertices) = }")
left_y = md.mesh.y[left_vertices]
left_sorted_idx = np.argsort(left_y)
left_vertices = left_vertices[left_sorted_idx] + 1  # Sort by y-coordinate and correct index
# print(f"post-sort{left_vertices}, {len(left_vertices) = }")

# Find right boundary vertices (terminus)
right_vertices = np.where(np.abs(md.mesh.x - x_max) < tolerance)[0]
# print(f"pre-sort{right_vertices}, {len(right_vertices)=}")
right_y = md.mesh.y[right_vertices]
right_sorted_idx = np.argsort(right_y)
right_vertices = right_vertices[right_sorted_idx] + 1 # Sort by y-coordinate correct index
# print(f"post-sort{right_vertices}, {len(right_vertices)=}")

# print(f"Found {len(left_vertices)} left boundary vertices")
# print(f"Found {len(right_vertices)} right boundary vertices")
# print(f"Left boundary vertices: {left_vertices}")
# print(f"Right boundary vertices: {right_vertices}")

# Create periodic pairing
n_pairs = min(len(left_vertices), len(right_vertices))
posx = left_vertices[:n_pairs]  # First n left vertices 
posx2 = right_vertices[:n_pairs] # First n right vertices

for i in range(n_pairs):
    left_y = md.mesh.y[posx[i]]
    right_y = md.mesh.y[posx2[i]]
    # print(f"Pair {i}: left vertex {posx[i]} (y={left_y:.2f}) ↔ right vertex {posx2[i]} (y={right_y:.2f})")

md.stressbalance.vertex_pairing = np.column_stack([posx, posx2])
# md.masstransport.vertex_pairing = np.column_stack([posx, posx2])

# Handle unpaired vertex - let it be free!
unpaired_vertex = right_vertices[-1]
md.stressbalance.spcvx[unpaired_vertex] = float('nan')  # Free to move
md.stressbalance.spcvy[unpaired_vertex] = float('nan')  # Free to move
# print(f"{n_pairs} periodic pairs + 1 free surface vertex")
# print("===========================\n")

# solver tolerances
md.stressbalance.abstol = np.nan
md.stressbalance.FSreconditioning = 1
md.stressbalance.shelf_dampening = 1
md.masstransport.isfreesurface = 1
md.stressbalance.maxiter = 20

# # visual check
# mesh, x, y, elems, _ = issm.model.mesh.process_mesh(md)
# # start a fresh figure and draw the mesh
# fig, ax = plt.subplots(figsize=(10,5))
# iplt.plot_mesh2d(mesh, ax=ax, color='0.7', linewidth=0.3)
# # define BC‐sets and colours
# bc_map = {
#     'bed'      : (1, 'tab:blue'),
#     'terminus' : (2, 'tab:orange'),
#     'surface'  : (3, 'tab:red'),
#     'inlet'    : (4, 'tab:purple'),
# }
# # scatter each group of vertices
# for label, (flag, col) in bc_map.items():
#     idx = np.where(md.mesh.vertexflags(flag))[0]
#     ax.scatter(x[idx], y[idx],
#                s=20,
#                color=col,
#                label=label,
#                edgecolor='k', linewidth=0.2)
# # tidy up
# ax.set_xlabel('X (m)')
# ax.set_ylabel('Y (m)')
# ax.legend(title='BC type', loc='upper right')
# plt.tight_layout()
# plt.show()

md.miscellaneous.name = 'ismipB'

# md.transient = deactivateall(md.transient)
md.transient.isstressbalance = 1
md.transient.ismasstransport = 0

md.stressbalance.restol = 1e-2
md.stressbalance.reltol = 1e-1

md.timestepping.start_time = 0
md.timestepping.final_time = 1
md.timestepping.time_step = 1/12
md.transient.requested_outputs = ['default', 'Surface', 'Base']

# Before solve:
print(f"Max friction coefficient: {np.max(md.friction.coefficient)}")
print(f"Max driving stress estimate: {np.max(md.materials.rho_ice * md.constants.g * md.geometry.thickness * np.tan(alpha))}")

md = solve(md, 'Transient')

print("Saving results")
export_netCDF(md, md.miscellaneous.name+'.nc')