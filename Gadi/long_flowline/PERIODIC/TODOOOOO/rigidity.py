from bedrock_generator import SyntheticBedrockModelConfig
from socket import gethostname
import numpy as np
from numpy import gradient
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('/home/ana/pyISSM/src')
import pyissm as issm
from pyissm import plot as iplt
from model import model
from cuffey import cuffey
from bamgflowband import bamgflowband
from setflowequation import setflowequation
from friction import friction
from export_netCDF import export_netCDF
from solve import solve


def format_output(md, L, filename):
    """
    Format output according to ISMIP‑HOM specifications (velocity only for now)
    """    
    # ------------------------------------------------------------------
    # 0️⃣  Scaled coordinates (always 0–1)
    # ------------------------------------------------------------------
    x_hat = md.mesh.x.flatten() / L
    y_hat = md.mesh.y.flatten() / L
    print(f"Coordinate ranges: x_hat=[{x_hat.min():.3f}, {x_hat.max():.3f}],"
          f"y_hat=[{y_hat.min():.3f}, {y_hat.max():.3f}]"
    )

    # ------------------------------------------------------------------
    # 1️⃣  Extract velocity fields (already in **m a⁻¹** on return from ISSM)
    # ------------------------------------------------------------------
    sol = md.results.StressbalanceSolution
    sol = sol[0] if hasattr(sol, "__len__") and len(sol) else sol

    vx_surface = sol.Vx.flatten()
    vy_surface = sol.Vy.flatten() if hasattr(sol, "Vy") else np.zeros_like(vx_surface)
    
    if hasattr(sol, "Vz"):
        vz_surface = sol.Vz.flatten()
    else:
        print("Warning: No Vz field found, using zeros")
        vz_surface = np.zeros_like(vx_surface)



    # Keep *copies* of the full (unfiltered) fields – needed for basal output
    vx_full, vy_full = vx_surface.copy(), vy_surface.copy()

    # ------------------------------------------------------------------
    # 2️⃣  Vertex masks
    # ------------------------------------------------------------------
    surface_idx = np.where(md.mesh.vertexonsurface == 1)[0]

    # surface rows for top‑of‑ice columns
    x_hat = x_hat[surface_idx]
    vx_surface, vz_surface = (
        vx_surface[surface_idx],
        vz_surface[surface_idx],
        )
    # matching basal nodes share the same column index ordering in ISSM
    basal_idx = np.where(md.mesh.vertexonbase == 1)[0]
    vx_basal = vx_full[basal_idx]

    # ------------------------------------------------------------------
    # 3️⃣  Quick sanity print‑outs
    # ------------------------------------------------------------------
    print(
        f"Surface velocity ranges (m a⁻¹):\n"
        f"  vx: [{vx_surface.min():.5f}, {vx_surface.max():.5f}]\n"
        f"  vy: [{vy_surface.min():.5f}, {vy_surface.max():.5f}]\n"
        f"  vz: [{vz_surface.min():.5f}, {vz_surface.max():.5f}]"
    )
    print(
        "Basal velocity ranges (m a⁻¹):\n"
        f"  vx_basal: [{vx_basal.min():.5f}, {vx_basal.max():.5f}]")

    y_range = md.mesh.y.max() - md.mesh.y.min()
    if md.mesh.dimension == 2 or y_range < L / 2:
        output_data = np.column_stack((x_hat, vx_surface, vz_surface, vx_basal))
    else:
        centre = np.isclose(y_hat, 0.5, atol=0.01)
        output_data = np.column_stack(
            (x_hat[centre], vx_surface[centre], vz_surface[centre], vx_basal[centre])
        )
    header = "x_hat vx_surface vz_surface vx_basal" # 4 cols: x̂ vx(zs) vz(zs) vx(zb)
    # ------------------------------------------------------------------
    # 5️⃣  Write to disk
    # ------------------------------------------------------------------
    filename = f"{filename}_static.txt"
    np.savetxt(filename, output_data, fmt="%.6f", delimiter="\t", header=header, comments="# ")

    print(f"✓ Saved {filename} with shape {output_data.shape}")
    print("First 5 rows:")
    print(output_data[:5])

    return filename, output_data


def save_results(md, L, filename):
    print(f"\n=== SAVING STATIC OUTPUT ===")
    
    # Check what results are available
    print("Available results in md.results:")
    for attr in dir(md.results):
        if not attr.startswith('_'):
            print(f"  md.results.{attr}")
    
    print("\nAvailable results in md.results.StressbalanceSolution:")
    stress_solution = md.results.StressbalanceSolution
    for attr in dir(stress_solution):
        if not attr.startswith('_'):
            try:
                value = getattr(stress_solution, attr)
                if hasattr(value, 'shape'):
                    print(f"  {attr}: shape {value.shape}")
                elif hasattr(value, '__len__'):
                    print(f"  {attr}: length {len(value)}, type {type(value)}")
                else:
                    print(f"  {attr}: {type(value)}")
            except Exception as e:
                print(f"  {attr}: (error: {e})")
    
    # Check if it's a list/array of solution steps
    if hasattr(stress_solution, '__len__') and len(stress_solution) > 0:
        print(f"\nStressbalanceSolution appears to be a sequence with {len(stress_solution)} elements")
        print("Checking first element:")
        first_element = stress_solution[0] if len(stress_solution) > 0 else stress_solution
        for attr in dir(first_element):
            if not attr.startswith('_'):
                try:
                    value = getattr(first_element, attr)
                    if hasattr(value, 'shape'):
                        print(f"  [0].{attr}: shape {value.shape}")
                    else:
                        print(f"  [0].{attr}: {type(value)}")
                except Exception as e:
                    print(f"  [0].{attr}: (error: {e})")
    
    # Try to format and save
    try:
        filename, data = format_output(md, L, filename)
        return filename
    
    except Exception as e:
        print(f"Error in format_output: {e}")
        return None


def adaptive_bamg(md, x, s0, b, resolution_factor=1.0):
    print("\n============ SETTING MESH==============")
    # hmax: Resolution based on wavelength

    # Scale refinement with wavelength/thickness ratio
    wavelength_thickness_ratio = bed_wavelength / ice_thickness  # unitless
    # refinement_factor = min(50, 10 * wavelength_thickness_ratio)

    if bed_wavelength < 15000:
        refinement_factor = 50
    else:
        refinement_factor = 200

    hmax = (bed_wavelength / refinement_factor) * resolution_factor

    # Now do horizontal/adaptive meshing without asking for nvert
    md = bamgflowband(md, x, s0, b,
                      # 'hmin', hmin,
                      'hmax', hmax,
                      'anisomax', 3,
                      'vertical', 1)
    
    nv, ne = md.mesh.numberofvertices, md.mesh.numberofelements

    print(f"\n[ADAPTIVE_BAMG] FINAL Mesh statistics:")
    print(f"  wavelength_thickness_ratio: {wavelength_thickness_ratio}")
    print(f"  hmax: {hmax}")
    print(f"  resolution_factor: {resolution_factor}")
    print(f"  refinement_factor: {refinement_factor}")
    print(f"  Total vertices: {md.mesh.numberofvertices}")
    print(f"  Elements: {md.mesh.numberofelements}")
    print(f"  inlet vertices: {len(np.where(md.mesh.vertexflags(4))[0])}")
    print(f"  terminus vertices: {len(np.where(md.mesh.vertexflags(2))[0])}")
    print(f"========================================")

    return md, nv, ne, resolution_factor


def calculate_relative_depth(md, vertices):
    surface_z = md.geometry.surface[vertices]
    bed_z = md.geometry.bed[vertices]
    vertex_z =  md.mesh.y[vertices] 

    thickness = surface_z - bed_z
    relative_depth = (vertex_z - bed_z) / thickness

    return relative_depth


def setup_periodic_boundary_conditions(md):
    tolerance = 1e-6
    # find left and right vertices
    x_min, x_max = np.min(md.mesh.x), np.max(md.mesh.x)
    left_vertices = np.where(np.abs(md.mesh.x - x_min) < tolerance)[0]
    right_vertices = np.where(np.abs(md.mesh.x - x_max) < tolerance)[0]

    print(f"Found {len(left_vertices)} left, {len(right_vertices)} right vertices")
    # pairing logic requires matching relative depths instead of coordinates
    pairs = []
    used_right = set()

    left_relative_depths = calculate_relative_depth(md, left_vertices)
    right_relative_depths = calculate_relative_depth(md, right_vertices)

    for i, left_idx in enumerate(left_vertices):
        left_depth = left_relative_depths[i]

        best_match = None
        best_depth_diff = float('inf')

        for j, right_idx in enumerate(right_vertices):
            if j in used_right:
                continue

            right_depth = right_relative_depths[j]
            depth_diff = abs(left_depth - right_depth)

            if depth_diff < 1e-1 and depth_diff < best_depth_diff:
                best_depth_diff = depth_diff
                best_match = j

        if best_match is not None:
            pairs.append([left_idx, right_vertices[best_match]])
            used_right.add(best_match)

    # convert to issm 1-based indexing 
    pairs_1based = np.array(pairs) + 1
    md.stressbalance.vertex_pairing = pairs_1based
    md.masstransport.vertex_pairing = pairs_1based
    print(f"✅ Created {len(pairs)} flowband pairs using relative depth matching")

    return md


def analyse_driving_stress(md, L):
    """
    Analyze why U-shape persists despite correct geometric periodic BCs
    """
    print(f"\n=== DRIVING STRESS DIAGNOSTIC {exp} ===")
    
    # Get surface vertices
    surface_idx = np.where(md.mesh.vertexonsurface == 1)[0]
    x = md.mesh.x[surface_idx]
    
    # Surface elevation and slope
    surface_z = md.geometry.surface[surface_idx]
    bed_z = md.geometry.bed[surface_idx]
    thickness = surface_z - bed_z

    # Sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    surface_sorted = surface_z[sort_idx]
    bed_sorted = bed_z[sort_idx]
    thickness_sorted = thickness[sort_idx]
    
    # Calculate surface slope (driving stress proxy)
    dx = np.diff(x_sorted)
    dz_surface = np.diff(surface_sorted)
    surface_slope = dz_surface / dx
    
    # Calculate bed slope  
    dz_bed = np.diff(bed_sorted)
    bed_slope = dz_bed / dx
    
    print(f"Surface elevation: {surface_sorted[0]:.3f} to {surface_sorted[-1]:.3f} m")
    print(f"Bed elevation: {bed_sorted[0]:.3f} to {bed_sorted[-1]:.3f} m")
    print(f"Ice thickness: {thickness_sorted[0]:.3f} to {thickness_sorted[-1]:.3f} m")
    
    # Check for systematic trends
    surface_trend = surface_sorted[-1] - surface_sorted[0]
    bed_trend = bed_sorted[-1] - bed_sorted[0]
    thickness_trend = thickness_sorted[-1] - thickness_sorted[0]
    
    print(f"\nSystematic trends over {L/1000:.0f} km:")
    print(f"  Surface: {surface_trend:.3f} m ({surface_trend/(L/1000):.3f} m/km)")
    print(f"  Bed: {bed_trend:.3f} m ({bed_trend/(L/1000):.3f} m/km)")
    print(f"  Thickness: {thickness_trend:.3f} m ({thickness_trend/(L/1000):.3f} m/km)")
    
    # Driving stress calculation
    g = md.constants.g # m/s²
    
    # At boundaries
    slope_left = surface_slope[0] if len(surface_slope) > 0 else 0
    slope_right = surface_slope[-1] if len(surface_slope) > 0 else 0
    thickness_left = thickness_sorted[0]
    thickness_right = thickness_sorted[-1]
    
    driving_stress_left = rho_ice * g * thickness_left * abs(slope_left)
    driving_stress_right = rho_ice * g * thickness_right * abs(slope_right)
    
    print(f"\nDriving stress at boundaries:")
    print(f"  Left (x=0): {driving_stress_left:.1f} Pa")
    print(f"  Right (x=L): {driving_stress_right:.1f} Pa")
    print(f"  Difference: {abs(driving_stress_right - driving_stress_left):.1f} Pa")
    print("\n======================================")
    
    return surface_slope, bed_slope, thickness_sorted


def setup_friction(md, exp):
    """
    Define basal friction field using Pattyn's law, 
    but with spatial variation inspired by Budd's formulation.
    """
    print("\n============ SETUP FRICTION ==============")    
    md.friction = friction()
    md.friction.p = np.ones(ne)
    md.friction.q = np.zeros(ne) # pattyn 2008 linear sliding means no N_eff

    basal_nodes = np.where(md.mesh.vertexonbase == 1)[0]
    x_basal = md.mesh.x[basal_nodes]

    if exp in ('S1', 'S2'):
        print(f"\nEXPERIMENT {exp}: basal boundary is frozen (no slip)")
        md.stressbalance.spcvx[basal_nodes] = 0
        md.stressbalance.spcvy[basal_nodes] = 0
        md.friction.coefficient = np.ones(nv)

    else:
        omega = 2 * np.pi / bed_wavelength  # m⁻¹

        # uniform friction to suit Pattyn's β²:
        beta2 = 1500 # (Pa·a·m⁻¹)
        print(f"\nExperiment {exp}: β² = {beta2} field")
        
        # Avoid negative or zero
        beta2 = np.clip(beta2, 1e-4, None)

        print(f"β² field statistics:")
        print(f"  Array size: {beta2.size}")
        print(f"  Range: [{np.min(beta2):.1f}, {np.max(beta2):.1f}] Pa·a·m⁻¹")
        print(f"  Mean: {np.mean(beta2):.1f} Pa·a·m⁻¹\n")

        # Convert to ISSM units and set friction coefficient
        beta2_issm = beta2 * md.constants.yts  # Pa·a·m⁻¹ → Pa·s·m⁻¹
        md.friction.coefficient = np.zeros(md.mesh.numberofvertices)
        md.friction.coefficient[basal_nodes] = np.sqrt(beta2_issm)

        print(f"Final friction coefficient:")
        print(f"  Array size: {md.friction.coefficient.size}")
        print(f"  Range: [{np.min(md.friction.coefficient):.1f}, {np.max(md.friction.coefficient):.1f}] Pa·s·m⁻¹")

    return md


def debug_friction_setup(md):
    """Debug what's actually happening with friction"""
    print("\n============ FRICTION DIAGNOSTIC ==============")
    basal_nodes = np.where(md.mesh.vertexonbase == 1)[0]
    
    if exp in ('S3', 'S4'):
        # intended values
        beta2_intended = 1500  # Pa·a·m⁻¹
        beta2_issm_intended = beta2_intended * md.constants.yts
        friction_coeff_intended = np.sqrt(beta2_issm_intended)

        print(f"INTENDED FRICTION:")
        print(f"  β² = {beta2_intended} Pa·a·m⁻¹")
        print(f"  β² (ISSM) = {beta2_issm_intended:.2e} Pa·s·m⁻¹") 
        print(f"  friction coeff = {friction_coeff_intended:.0f} Pa·s·m⁻¹")
    
    actual_friction_coeff = md.friction.coefficient[basal_nodes]
    actual_beta2_issm = actual_friction_coeff**2
    actual_beta2_annual = actual_beta2_issm / md.constants.yts
    
    print(f"\nACTUAL FRICTION:")
    print(f"  friction coeff range = [{np.min(actual_friction_coeff):.0f}, {np.max(actual_friction_coeff):.0f}]")
    print(f"  β² (ISSM) range = [{np.min(actual_beta2_issm):.2e}, {np.max(actual_beta2_issm):.2e}]")
    print(f"  β² (annual) range = [{np.min(actual_beta2_annual):.0f}, {np.max(actual_beta2_annual):.0f}]")


def visual_mesh_check(md):
    # visual MESH check
    md_mesh, md_x, md_y, md_elements, md_is3d = issm.model.mesh.process_mesh(md)
    iplt.plot_mesh2d(md_mesh, show_nodes = True)
    plt.savefig(f"profile_{BEDROCK_PROFILE_ID:03d}_{exp}_{resolution_factor}_mesh.png")
    plt.show()


def compare_results_summary(sol_n3, sol_n1, field_name):
    """Print statistical comparison of two solution fields"""
    if not hasattr(sol_n3, field_name) or not hasattr(sol_n1, field_name):
        print(f"⚠ Field {field_name} not available in one or both solutions")
        return
        
    field_n3 = getattr(sol_n3, field_name)
    field_n1 = getattr(sol_n1, field_name)
    
    # Handle potential array shape differences
    if field_n3.shape != field_n1.shape:
        print(f"⚠ Shape mismatch for {field_name}: n3={field_n3.shape}, n1={field_n1.shape}")
        return
    
    diff = field_n1 - field_n3
    rel_diff = np.abs(diff) / (np.abs(field_n3) + 1e-12)
    
    print(f"\n=== {field_name} COMPARISON ===")
    print(f"  n=3 range: [{field_n3.min():.3e}, {field_n3.max():.3e}]")
    print(f"  n=1 range: [{field_n1.min():.3e}, {field_n1.max():.3e}]")
    print(f"  Max abs diff: {np.abs(diff).max():.3e}")
    print(f"  Mean rel diff: {np.mean(rel_diff):.3f}")
    print(f"  RMS rel diff: {np.sqrt(np.mean(rel_diff**2)):.3f}")

def plot_field_comparison(sol_n3, sol_n1, field_name, md, L, save_prefix=""):
    """Plot side-by-side comparison of a field from both solutions"""
    if not hasattr(sol_n3, field_name) or not hasattr(sol_n1, field_name):
        print(f"⚠ Cannot plot {field_name} - not available in solutions")
        return
        
    field_n3 = getattr(sol_n3, field_name)
    field_n1 = getattr(sol_n1, field_name)
    
    if field_n3.shape != field_n1.shape:
        print(f"⚠ Cannot plot {field_name} - shape mismatch")
        return
    
    # Get surface vertices for plotting
    surface_idx = np.where(md.mesh.vertexonsurface == 1)[0]
    x_surface = md.mesh.x[surface_idx] / 1000  # Convert to km
    
    # Extract surface values
    field_n3_surface = field_n3.flatten()[surface_idx]
    field_n1_surface = field_n1.flatten()[surface_idx]
    
    # Sort by x-coordinate
    sort_idx = np.argsort(x_surface)
    x_sorted = x_surface[sort_idx]
    field_n3_sorted = field_n3_surface[sort_idx]
    field_n1_sorted = field_n1_surface[sort_idx]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot n=3 results
    ax1.plot(x_sorted, field_n3_sorted, 'b-', linewidth=2, label=f'{field_name} (n=3)')
    ax1.set_ylabel(f'{field_name}')
    ax1.set_title(f'Non-linear (n=3) - {field_name}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot n=1 results  
    ax2.plot(x_sorted, field_n1_sorted, 'r-', linewidth=2, label=f'{field_name} (n=1)')
    ax2.set_ylabel(f'{field_name}')
    ax2.set_title(f'Linear (n=1) - {field_name}')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot difference
    diff = field_n1_sorted - field_n3_sorted
    ax3.plot(x_sorted, diff, 'g-', linewidth=2, label=f'Difference (n=1 - n=3)')
    ax3.set_xlabel('Distance (km)')
    ax3.set_ylabel(f'{field_name} Difference')
    ax3.set_title(f'Difference: Linear - Non-linear')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}comparison_{field_name}_{BEDROCK_PROFILE_ID:03d}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()


################################################################################
# ~~~~~~~~~~~~~~~ 1. SINGLE SETUP AND MESH GENERATION ~~~~~~~~~~~~~~~
################################################################################

# --- Globals and Geometry (as before) ---
BEDROCK_PROFILE_ID = 165
bedrock_config = SyntheticBedrockModelConfig(profile_id=BEDROCK_PROFILE_ID)
ice_thickness = bedrock_config.ice_thickness
bed_wavelength = bedrock_config.profile_params['wavelength']

###############~~~~~~ SIMULATION  ~~~~~~~~############################

# GEOMETRY
L = 210e3
nx = int(L * 0.01) # Adjusted proportionally for ~100m resolution
md = model()
x_1D = np.linspace(0, L, nx)
b = bedrock_config.get_bedrock_elevation(x_1D)
s0 = b + ice_thickness

# mesh
md, nv, ne, resolution_factor = adaptive_bamg(md, x_1D, s0, b, 1.0)

# constants & material properties
yts = 31556926 # s/yr
rho_ice = 910 # kg/m^3. From table 1 in Pattyn 2008
ice_temperature = (273.15 - 10) # for initialisation

mesh_x = md.mesh.x
bed_2d = np.interp(mesh_x, x_1D, b) # Bedrock at all mesh points
surface_2d = np.interp(mesh_x, x_1D, s0) # Surface at all mesh points

# GEOMETRY SETTINGS  
md.geometry.surface = surface_2d
md.geometry.bed = bed_2d
md.geometry.thickness = surface_2d - bed_2d
md.geometry.base = md.geometry.bed

# constants and material properties
md.constants.yts = yts
md.materials.rho_ice = rho_ice

# where the ice is grounded and floating in the domain
md.mask.ocean_levelset = np.ones(nv) # grounded ice if positive floating ice if negative, grounding line lies where its zero
md.mask.ice_levelset = -np.ones(nv) #Ice is present if md.mask.ice_levelset is negative, icefront lies where its zero
# find nodes on the terminus vertical face and set icefront there
terminus_nodes = np.where(md.mesh.vertexflags(2))[0]
md.mask.ice_levelset[terminus_nodes] = 0          # ice front at terminus

md = setflowequation(md, 'FS', 'all')

# velocity initialisation
md.initialization.vx = np.zeros(nv)
md.initialization.vy = np.zeros(nv)
md.initialization.vz = np.zeros(nv)
md.initialization.vel = np.zeros(nv)
#hydrostatic seed
md.initialization.pressure = (md.constants.g * md.materials.rho_ice *
                            (md.geometry.surface - md.geometry.base))
# temperature
md.initialization.temperature = ice_temperature * np.ones(nv)

# basal forcing
md.basalforcings.floatingice_melting_rate = np.zeros(nv)
md.basalforcings.groundedice_melting_rate = np.zeros(nv)

# SMB initialisation
md.smb.mass_balance = np.zeros(nv)

# BCS
#temperature, thickness, velocity #loadingforce
# referential local 2D basis for stress-balance
md.thermal.spctemperature = np.full(nv, np.nan)
md.masstransport.spcthickness = np.full(nv, np.nan)
md.stressbalance.loadingforce = np.zeros((nv, 3))
md.stressbalance.referential = np.full((nv, 6), np.nan)

#spatial BCS
md.stressbalance.spcvx = np.full(nv, np.nan)
md.stressbalance.spcvy = np.full(nv, np.nan)
md.stressbalance.spcvz = np.full(nv, np.nan)

# periodic BCS
md = setup_periodic_boundary_conditions(md)

# misc
md.stressbalance.abstol = np.nan
md.stressbalance.FSreconditioning = 1
md.stressbalance.shelf_dampening = 1
md.masstransport.isfreesurface = 1
md.groundingline.migration = 'None'

# solver settings    
md.settings.solver_residue_threshold = 1e-3 #<<<<
md.stressbalance.restol = 1e-3 #<<<<
md.stressbalance.reltol = 1e-3 #<<<<
md.stressbalance.maxiter = 200 #<<< 50


################################################################################
# ~~~~~~~~~~~~~~~ 2. RUN 1: NON-LINEAR REFERENCE (n=3) ~~~~~~~~~~~~~~~
################################################################################


print("\n\n===== STARTING: Non-Linear (n=3) Reference Simulation =====")

# --- Configure for n=3 ---
exp_n3 = 'S4'
rheology_B_n3 = cuffey(ice_temperature)
md.materials.rheology_B = rheology_B_n3 * np.ones(nv)
md.materials.rheology_n = 3 * np.ones(ne)
md.materials.rheology_law = "BuddJacka"

md.miscellaneous.name = f'{BEDROCK_PROFILE_ID:03d}_{exp_n3}_{resolution_factor}'

# --- Friction and Solve ---
md = setup_friction(md, exp_n3)

md.stressbalance.requested_outputs = [
    'Vx', 'Vy', 'Vel',                    
    'StrainRatexx', 'StrainRatexy',              
    'StressTensorxx', 'StressTensorxy', 'StressTensoryy',          
    'BasalDragx', 'BasalDragy'                   
]
md = solve(md,'Stressbalance')

# Store n=3 results before they get overwritten
print("--- Storing n=3 results for comparison ---")
results_n3 = md.results.StressbalanceSolution

# --- Save results from non-linear run ---
print("\n--- Saving results for non-linear run ---")
try:
    save_results(md, L, md.miscellaneous.name)
except Exception as e:
    print(f"⚠ Error saving n=3 output: {e}")

# --- Extract reference state directly into variables ---
print("\n--- Extracting reference state for equivalent linear run ---")
stress_xx_n3 = md.results.StressbalanceSolution.StressTensorxx
stress_yy_n3 = md.results.StressbalanceSolution.StressTensoryy
stress_xy_n3 = md.results.StressbalanceSolution.StressTensorxy

################################################################################
# ~~~~~~~~~~~~~~~ 3. RUN 2: EQUIVALENT LINEAR (n=1) ~~~~~~~~~~~~~~~
################################################################################

print("\n\n===== STARTING: Equivalent Linear (n=1) Simulation =====")

# --- Calculate Spatially-Varying Rheology for n=1 ---
print("--- Calculating equivalent linear rheology (B for n=1) ---")

# 1. Convert B_n3 to A_n3. For n=3, B = A^(-1/n) => A = B^(-n)
A_n3 = rheology_B_n3**(-3)

# 2. Calculate the stress-dependent term from derivation
# Formula: A1 = A3 * 0.25 * ((σxx-σyy)² + 4*σxy²)
stress_factor = 0.25 * ((stress_xx_n3 - stress_yy_n3)**2 + 4 * stress_xy_n3**2)

# 3. Calculate the spatially-varying A_n1 field
A_n1_field = A_n3 * stress_factor

# 4. Convert A_n1 to rheology_B for the linear (n=1) case
# For n=1, ISSM expects B to be the dynamic viscosity η, where η = (2*A)^-1
# Use a very small floor value to prevent division by zero
A_n1_field[A_n1_field == 0] = 1e-6
rheology_B_n1 = (2 * A_n1_field)**(-1)
rheology_B_n1 = np.clip(rheology_B_n1, 1e9, 1e15)

# 5. Apply tuning factor
tuning_factor = 1.5  # Start with 50% more viscosity
print(f"Applying empirical tuning factor of {tuning_factor}")
rheology_B_n1 *= tuning_factor

print(f"✓ Calculated spatially-varying rheology_B field.")
print(f"  New rheology_B range: [{np.min(rheology_B_n1):.2e}, {np.max(rheology_B_n1):.2e}]")

# --- Re-configure the SAME model object for n=1 ---
exp_n1 = 'S3_equivalent'
md.materials.rheology_B = rheology_B_n1
md.materials.rheology_n = 1 * np.ones(ne)
md.materials.rheology_law = "BuddJacka"

md.miscellaneous.name = f'{BEDROCK_PROFILE_ID:03d}_{exp_n1}_{resolution_factor}'

# --- Friction and Solve ---
# The friction setup for 'S3' and 'S4' is identical in script, so we can reuse it.
# If it were different, you would call setup_friction(md, 'S3') here.
md.stressbalance.requested_outputs = [
    'Vx', 'Vy', 'Vel',                    
    'StrainRatexx', 'StrainRatexy',              
    'StressTensorxx', 'StressTensorxy', 'StressTensoryy',          
    'BasalDragx', 'BasalDragy'                   
]
md = solve(md, 'Stressbalance')
# --- Save results from equivalent linear run ---
print("\n--- Saving results for equivalent linear run ---")

print("\n--- Extracting reference state for equivalent linear run ---")
stress_xx_n1 = md.results.StressbalanceSolution.StressTensorxx
stress_yy_n1 = md.results.StressbalanceSolution.StressTensoryy
stress_xy_n1 = md.results.StressbalanceSolution.StressTensorxy


try:
    save_results(md, L, md.miscellaneous.name)
except Exception as e:
    print(f"⚠ Error saving n=1 output: {e}")

print("\n\n===== COMPARING S4 (n=3) vs S3 (n=1) RESULTS =====")

# Store solution references
sol_n3 = results_n3  # from your stored n=3 results
sol_n1 = md.results.StressbalanceSolution  # from your n=1 solve

# Statistical comparisons
print("\n--- STATISTICAL COMPARISONS ---")
fields_to_compare = ['Vx', 'Vel', 'StrainRatexx', 'StrainRatexy', 
                     'StressTensorxx', 'StressTensorxy', 'BasalDragx']

for field in fields_to_compare:
    compare_results_summary(sol_n3, sol_n1, field)

# Visual comparisons - create plots for key fields
print("\n--- GENERATING COMPARISON PLOTS ---")
key_fields_to_plot = ['Vx', 'Vel', 'StressTensorxx', 'StrainRatexx']

for field in key_fields_to_plot:
    plot_field_comparison(sol_n3, sol_n1, field, md, L, 
                         save_prefix=f"profile_{BEDROCK_PROFILE_ID:03d}_")

# Summary velocity comparison plot
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Get surface data for plotting
surface_idx = np.where(md.mesh.vertexonsurface == 1)[0]
x_surface = md.mesh.x[surface_idx] / 1000
sort_idx = np.argsort(x_surface)
x_sorted = x_surface[sort_idx]

# Plot Vx comparison
if hasattr(sol_n3, 'Vx') and hasattr(sol_n1, 'Vx'):
    vx_n3 = sol_n3.Vx.flatten()[surface_idx][sort_idx]
    vx_n1 = sol_n1.Vx.flatten()[surface_idx][sort_idx]
    
    axes[0,0].plot(x_sorted, vx_n3, 'b-', label='n=3', linewidth=2)
    axes[0,0].plot(x_sorted, vx_n1, 'r--', label='n=1', linewidth=2)
    axes[0,0].set_ylabel('Vx (m/yr)')
    axes[0,0].set_title('Surface Velocity X-Component')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

# Plot Vel comparison  
if hasattr(sol_n3, 'Vel') and hasattr(sol_n1, 'Vel'):
    vel_n3 = sol_n3.Vel.flatten()[surface_idx][sort_idx]
    vel_n1 = sol_n1.Vel.flatten()[surface_idx][sort_idx]
    
    axes[0,1].plot(x_sorted, vel_n3, 'b-', label='n=3', linewidth=2)
    axes[0,1].plot(x_sorted, vel_n1, 'r--', label='n=1', linewidth=2)
    axes[0,1].set_ylabel('Vel (m/yr)')
    axes[0,1].set_title('Surface Speed')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

# Plot velocity difference
if hasattr(sol_n3, 'Vx') and hasattr(sol_n1, 'Vx'):
    vx_diff = vx_n1 - vx_n3
    axes[1,0].plot(x_sorted, vx_diff, 'g-', linewidth=2)
    axes[1,0].set_xlabel('Distance (km)')
    axes[1,0].set_ylabel('Vx Difference (m/yr)')
    axes[1,0].set_title('Velocity Difference (Linear - Non-linear)')
    axes[1,0].grid(True, alpha=0.3)

# Plot relative error
if hasattr(sol_n3, 'Vx') and hasattr(sol_n1, 'Vx'):
    rel_error = np.abs(vx_diff) / (np.abs(vx_n3) + 1e-6) * 100
    axes[1,1].plot(x_sorted, rel_error, 'm-', linewidth=2)
    axes[1,1].set_xlabel('Distance (km)')
    axes[1,1].set_ylabel('Relative Error (%)')
    axes[1,1].set_title('Relative Velocity Error')
    axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'velocity_summary_comparison_{BEDROCK_PROFILE_ID:03d}.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✓ Comparison complete! Check generated plots and statistics above.")
print(f"✓ Plots saved with prefix: profile_{BEDROCK_PROFILE_ID:03d}_")
print("\n\n===== Workflow Complete =====")
