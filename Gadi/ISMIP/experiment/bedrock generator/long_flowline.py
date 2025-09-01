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
    vx_surface, vy_surface, vz_surface = (
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
        output_data = np.column_stack((x_hat, vx_surface, vy_surface, vx_basal))
    else:
        centre = np.isclose(y_hat, 0.5, atol=0.01)
        output_data = np.column_stack(
            (x_hat[centre], vx_surface[centre], vy_surface[centre], vx_basal[centre])
        )

    sort_indices = np.argsort(output_data[:, 0])
    output_data = output_data[sort_indices]

    header = "x_hat vx_surface vy_surface vx_basal" # 4 cols: x̂ vx(zs) vy(zs) vx(zb) - flowband model
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


def setup_non_periodic_boundary_conditions(md):
    """
    Apply non-periodic BCs with:
    depth dependent velocity (vx=) at inlet
    Neumman free outflow at the terminus (vx=NaN)
    """
    print("\n============ SETTING NON-PERIODIC BCS ==============")
    # Boundary conditions
    # Get boundary node sets
    inlet_nodes = np.where(md.mesh.vertexflags(4))[0]  
    terminus_nodes = np.where(md.mesh.vertexflags(2))[0]

    print(f" inlet nodes shape: {inlet_nodes.shape}")
    print(f" terminus nodes shape: {terminus_nodes.shape}")

    # OLD: Dirichlet inlet boundary
    md.stressbalance.spcvx[inlet_nodes] = 0.0
    
    # --- NEW DEPTH-DEPENDENT INLET BOUNDARY ---
    # md = setup_depth_dependent_inlet_bc(md, exp, u_b_sliding=20.0)

    # Neumman terminus
    md.stressbalance.spcvx[terminus_nodes] = np.nan
    md.stressbalance.spcvy[terminus_nodes] = np.nan

    return md


def diagnose_boundary_conditions(md):  
    """Diagnose if velocities are zero"""
    print("\n============ BCS DIAGNOSTIC ==============")    
    # Check how many vertices have prescribed velocities
    prescribed_vx = ~np.isnan(md.stressbalance.spcvx)
    prescribed_vy = ~np.isnan(md.stressbalance.spcvy)
    
    print(f"Vertices with prescribed vx: {np.sum(prescribed_vx)}")
    print(f"Vertices with prescribed vy: {np.sum(prescribed_vy)}")
    print(f"Total vertices: {len(md.stressbalance.spcvx)}")
    
    if np.sum(prescribed_vx) > len(md.stressbalance.spcvx) * 0.1:
        print("⚠️  Warning: Too many vertices have prescribed velocities!")
        print("This might over-constrain the system")
    
    # Check terminus conditions specifically
    terminus_nodes = np.where(md.mesh.vertexflags(2))[0]
    terminus_vx = md.stressbalance.spcvx[terminus_nodes]
    
    print(f"\nTerminus nodes: {len(terminus_nodes)}")
    if np.all(np.isnan(terminus_vx)):
        print("Terminus vx range: [NaN, NaN] m/s (all values are NaN)")
    else:
        print(f"Terminus vx range: [{np.nanmin(terminus_vx):.2e}, {np.nanmax(terminus_vx):.2e}] m/s")

    return md

def diagnose_acceleration_onset(md, L):
    """Find where runaway acceleration starts"""
    surface_idx = np.where(md.mesh.vertexonsurface == 1)[0]
    x = md.mesh.x[surface_idx]

    # Get velocity from the diagnostic stress balance solution
    sol = md.results.StressbalanceSolution
    vx = sol.Vx[surface_idx]

    print(f"Debug info:")
    print(f"  Surface nodes: {len(surface_idx)}")
    print(f"  x shape: {x.shape}")
    print(f"  vx shape: {vx.shape}")
    print(f"  x range: [{x.min():.1f}, {x.max():.1f}] m")
    print(f"  vx range: [{vx.min():.3f}, {vx.max():.3f}] m/yr")

    # Sort by x position
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    vx_sorted = vx[sort_idx]

    # Simple velocity gradient calculation
    if len(x_sorted) < 2:
        print("Not enough points for gradient calculation")
        return x_sorted, vx_sorted, np.zeros_like(vx_sorted)
    
    # Calculate gradient using simple finite differences
    dvx_dx = np.zeros_like(vx_sorted)
    
    # Forward difference for all points except last
    for i in range(len(x_sorted) - 1):
        dx = x_sorted[i+1] - x_sorted[i]
        dvx = vx_sorted[i+1] - vx_sorted[i]
        if dx != 0:
            dvx_dx[i] = dvx / dx
    
    # Use the second-to-last gradient for the last point
    if len(dvx_dx) > 1:
        dvx_dx[-1] = dvx_dx[-2]

    print(f"  dvx_dx range: [{dvx_dx.min():.2e}, {dvx_dx.max():.2e}] 1/yr")

    # Find where acceleration becomes excessive (define threshold)
    threshold = 1e-3  # m/yr per m, adjust as needed
    problem_idx = np.where(dvx_dx > threshold)[0]

    if len(problem_idx) > 0:
        problem_distance = x_sorted[problem_idx[0]]
        distance_from_terminus = L - problem_distance
        print(f"Acceleration problem starts at {distance_from_terminus/1000:.1f} km")

    return x_sorted, vx_sorted, dvx_dx


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


def plot_transient_fields(md):                                                            
    """                                                                                                    
    Plot fields from the last time step of the transient evolution                                                                                                    
    """                                                                                                    
    print("\n=== PLOTTING TRANSIENT FIELDS (LAST TIME STEP) ===")                                          
    # Check if transient results exist
    if not hasattr(md.results, 'TransientSolution'):
        print("No TransientSolution found in md.results")
        return

    transient_solution = md.results.TransientSolution

    # velocity magnitude
    iplt.plot_model_field(md, transient_solution.Vel[:, -1], show_cbar=True)
    plt.title('Velocity Magnitude (m/yr)')
    plt.savefig(f"velocity_final_{BEDROCK_PROFILE_ID:03d}_{exp}.png", dpi=300, bbox_inches='tight')

    # Pressure magnitude
    iplt.plot_model_field(md, transient_solution.Pressure[:, -1], show_cbar=True)
    plt.title('Pressure Magnitude (Pa)')
    plt.savefig(f"pressure_final_{BEDROCK_PROFILE_ID:03d}_{exp}.png", dpi=300, bbox_inches='tight')

    # Thickness
    iplt.plot_model_field(md, transient_solution.Thickness[:,-1], show_cbar=True)
    plt.title('Thickness (m)')
    plt.savefig(f"thickness_final_{BEDROCK_PROFILE_ID:03d}_{exp}.png", dpi=300, bbox_inches='tight')

    # Bed, base and surface elevations
    plt.figure(figsize=(16, 5))
    # Use md.mesh.x directly to ensure correct dimensions
    mesh_x_full = md.mesh.x
    # Get surface nodes only for proper 1D profile plotting
    surface_idx = np.where(md.mesh.vertexonsurface == 1)[0]
    x_surface = md.mesh.x[surface_idx]
    # Sort by x position for proper line plotting
    sort_idx = np.argsort(x_surface)
    x_sorted = x_surface[sort_idx]
    surface_sorted = transient_solution.Surface[surface_idx, -1][sort_idx]
    base_sorted = transient_solution.Base[surface_idx, -1][sort_idx]
    bed_sorted = md.geometry.bed[surface_idx][sort_idx]

    plt.plot(x_sorted, surface_sorted, label='Surface')
    plt.plot(x_sorted, base_sorted, label='Base')
    plt.fill_between(x_sorted, base_sorted, surface_sorted, alpha=0.3)
    plt.plot(x_sorted, bed_sorted, color="brown", label='Bed')
    plt.legend()
    plt.title('Base & Surface Elevation (m)')
    plt.savefig(f"signals_elevations_final_{BEDROCK_PROFILE_ID:03d}_{exp}.png", dpi=300, bbox_inches='tight')

    plt.show()

    # IS THIS THE MESH AT THE LAST TIME STEP ?
    visual_mesh_check(md)


import xarray as xr

def plot_max_velocity_from_netcdf(filename):
    """
    Reads a NetCDF output file from ISSM and plots the evolution of maximum velocity.
    """
    print(f"\n=== PLOTTING MAX VELOCITY EVOLUTION FROM {filename} ===")
    
    # Define the path to the results group within the NetCDF file
    group_path = 'results/TransientSolution'
    
    try:
        # 1. Open the specific group within the NetCDF dataset
        ds = xr.open_dataset(filename, group=group_path)
        
        # 2. Extract the time and velocity data
        time_steps = ds['time'].values
        velocity_data = ds['Vel']
        
        # 3. Calculate the maximum velocity at each time step
        #    From ncdump, the spatial dimension is named 'VertNum'.
        max_velocities = velocity_data.max(dim='VertNum').values
        
        # 4. Create the plot
        plt.figure(figsize=(10, 6))
        plt.scatter(time_steps, max_velocities, alpha=0.8, s=30)
        plt.xlabel('Time (years)')
        plt.ylabel('Maximum Velocity (m/yr)')
        plt.title('Evolution of Maximum Velocity (from NetCDF)')
        plt.grid(linestyle=':', alpha=0.3)
        
        # Save the figure with a new name
        base_name = filename.rsplit('.', 1)[0]
        plt.savefig(f"{base_name}_max_vel_evolution.png", dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
    
        print(f"✓ Plot saved for {filename}")
        print(f"  Max velocity range: {np.min(max_velocities):.2f} to {np.max(max_velocities):.2f} m/yr")

    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nDEBUGGING INFO:")
        print("Please ensure the group path and dimension names are correct for NetCDF file.")
        print(f"Attempted to open group: '{group_path}'")


def Solve(md):
    # TRANSIENT - ACTIVE PHYSICS
    md.settings.sb_coupling_frequency = 1 # run stress balance every timestep
    md.transient.isstressbalance = 1
    md.transient.ismasstransport = 1
    md.transient.issmb = 0
    md.transient.isthermal = 0

    # TIMESTEPPING SETUP
    md.timestepping.time_step = timestep * resolution_factor # years

    md.timestepping.start_time = 0.0  # years
    md.timestepping.final_time = final_time  # years
    md.settings.output_frequency = 100 * resolution_factor #1

    # Set up output settings
    md.transient.requested_outputs = [
    'default','Vx','Vy','Vel','Pressure','Thickness','Surface','Base', 'SurfaceSlopeX','BedSlopeX',
    ]

    print("sb_coupling_frequency:", md.settings.sb_coupling_frequency)
    print("output_frequency:", md.settings.output_frequency)
    print("isstressbalance:", md.transient.isstressbalance)
    print("Δt (yr):", md.timestepping.time_step, 
    "Tfinal (yr):", md.timestepping.final_time, 
    "≈nsteps:", int(md.timestepping.final_time/md.timestepping.time_step))


    return solve(md, 'Transient')

######################################################################################################################################


##################~~~~~~ GLOBALS  ~~~~##################
################# THINGS TO PLAY WITH ##################
########################################################

# BEDROCK_PROFILE_ID = 0   # S=-0, K=0
BEDROCK_PROFILE_ID = 1   # S=-0, K=0

# BEDROCK_PROFILE_ID = 5   # S=-0.2, K=0.1
# BEDROCK_PROFILE_ID = 25  # S=0.2, K=0.1
# BEDROCK_PROFILE_ID = 127 # S=-0.2, K=-0.2
# BEDROCK_PROFILE_ID = 133 # S=-0.1, K=-0.1
# BEDROCK_PROFILE_ID = 270 # S=0.1, K=0.1
# BEDROCK_PROFILE_ID = 271 # S=0.1, K=0.2
# BEDROCK_PROFILE_ID = 396 # S=0.1, K=0.2
# BEDROCK_PROFILE_ID = 399 # S=0.2, K=0.0
# BEDROCK_PROFILE_ID = 519 # S=0.1, K=0.0
# BEDROCK_PROFILE_ID = 520 # S=0.1, K=0.1
# BEDROCK_PROFILE_ID = 636 # S=-0.1, K=0.2
# BEDROCK_PROFILE_ID = 639 # S=0.0, K=0.0
# BEDROCK_PROFILE_ID = 768 # S=0.1, K=-0.1
# BEDROCK_PROFILE_ID = 769 # S=0.1, K=0.0

# BEDROCK_PROFILE_ID = 33  # S=-0.1, K=-0.1
# BEDROCK_PROFILE_ID = 39  # S=0.0, K=0.0
# BEDROCK_PROFILE_ID = 165 # S=0.0, K=0.1 <<<
# BEDROCK_PROFILE_ID = 172 # S=0.2, K=-0.2
# BEDROCK_PROFILE_ID = 287 # S=0.0, K=-0.2
# BEDROCK_PROFILE_ID = 288 # S=0.0, K=-0.1
# BEDROCK_PROFILE_ID = 412 # S=0.0, K=-0.2
# BEDROCK_PROFILE_ID = 420 # S=0.1, K=0.1
# BEDROCK_PROFILE_ID = 546 # S=0.1, K=0.2
# BEDROCK_PROFILE_ID = 547 # S=0.2, K=-0.2
# BEDROCK_PROFILE_ID = 667 # S=0.1, K=-0.2
# BEDROCK_PROFILE_ID = 674 # S=0.2, K=0.0
# BEDROCK_PROFILE_ID = 783 # S=-0.1, K=-0.1
# BEDROCK_PROFILE_ID = 794 # S=0.1, K=0.0

# BEDROCK_PROFILE_ID = 63  # S=0.0, K=-0.1
# BEDROCK_PROFILE_ID = 64  # S=0.0, K=0.0
# BEDROCK_PROFILE_ID = 177 # S=-0.2, K=-0.2
# BEDROCK_PROFILE_ID = 178 # S=-0.2, K=-0.1
# BEDROCK_PROFILE_ID = 318 # S=0.1, K=-0.1
# BEDROCK_PROFILE_ID = 319 # S=0.1, K=0.0
# BEDROCK_PROFILE_ID = 436 # S=-0.1, K=0.2
# BEDROCK_PROFILE_ID = 437 # S=0.0, K=-0.2
# BEDROCK_PROFILE_ID = 552 # S=-0.2, K=-0.2
# BEDROCK_PROFILE_ID = 574 # S=0.2, K=0.0
# BEDROCK_PROFILE_ID = 694 # S=0.1, K=0.0
# BEDROCK_PROFILE_ID = 696 # S=0.1, K=0.2
# BEDROCK_PROFILE_ID = 807 # S=-0.1, K=-0.2
# BEDROCK_PROFILE_ID = 826 # S=0.2, K=0.2

# BEDROCK_PROFILE_ID = 88  # S=0.0, K=-0.1
# BEDROCK_PROFILE_ID = 89  # S=0.0, K=0.0
# BEDROCK_PROFILE_ID = 213 # S=0.0, K=-0.1
# BEDROCK_PROFILE_ID = 214 # S=0.0, K=0.0
# BEDROCK_PROFILE_ID = 345 # S=0.1, K=0.1
# BEDROCK_PROFILE_ID = 346 # S=0.1, K=0.2
# BEDROCK_PROFILE_ID = 463 # S=0.0, K=-0.1
# BEDROCK_PROFILE_ID = 464 # S=0.0, K=0.0
# BEDROCK_PROFILE_ID = 590 # S=0.0, K=0.1
# BEDROCK_PROFILE_ID = 591 # S=0.0, K=0.2
# BEDROCK_PROFILE_ID = 725 # S=0.2, K=0.1
# BEDROCK_PROFILE_ID = 726 # S=0.2, K=0.2
# BEDROCK_PROFILE_ID = 828 # S=-0.2, K=-0.1
# BEDROCK_PROFILE_ID = 843 # S=0.1, K=-0.1

########################################################

bedrock_config = SyntheticBedrockModelConfig(profile_id=BEDROCK_PROFILE_ID)
alpha = bedrock_config.alpha
bed_amplitude = bedrock_config.profile_params['amplitude']
bed_wavelength = bedrock_config.profile_params['wavelength']
ice_thickness = bedrock_config.ice_thickness

########################################################
# Choose experiment

# exp = 'S1' # no slip
# rheology_n = 1

# exp = 'S2' # no slip + ice rheology
# rheology_n = 3

exp = 'S3' # slip only
rheology_n = 1

# exp = 'S4' # slip + ice rheology
# rheology_n = 3

########################################################

final_time = 300#0.15# 1 # years

if rheology_n == 1:  # S1, S3 cases
    timestep = 2/1460  # 0.5 days (2x smaller)
else:  # S2, S4 cases
    timestep = 1/365   # 1 day (current stable value)


########################################################

# constants & material properties
yts = 31556926 # s/yr

rho_ice = 910 # kg/m^3. From table 1 in Pattyn 2008
ice_temperature = (273.15 - 60) # for initialisation

# rheology
if rheology_n==1:
    # if n=1: ISSM interprets the rheology_B parameter as the dynamic viscosity (η)
    # in Pattyn for (n=1), the effective viscosity (η) is defined as η=(2A)^−1
    A = 2.140373e-7 / yts
    rheology_B = (2 * A)**(-1) # = 7.37e13
else:
    rheology_B = cuffey(ice_temperature)

###############~~~~~~ SIMULATION  ~~~~~~~~############################

# GEOMETRY
L_buffer_inlet = 25e3    # The length of sacrificial buffer zone
L_interest = 160e3 # region of interest
L_buffer_terminus = 25e3    # The length of sacrificial buffer zone
L = L_buffer_inlet + L_interest + L_buffer_terminus # Total domain length is now 210 km

nx = int(L * 0.01) # Adjusted proportionally for ~100m resolution
md = model()
x_1D = np.linspace(0, L, nx)
b = bedrock_config.get_bedrock_elevation(x_1D)
s0 = b + ice_thickness

# mesh
## grid convergence test:
# md, nv, ne, resolution_factor = adaptive_bamg(md, x_1D, s0, b, 1.125) # coarsest resolution
md, nv, ne, resolution_factor = adaptive_bamg(md, x_1D, s0, b, 1.0) # coarse resolution
# md, nv, ne, resolution_factor = adaptive_bamg(md, x_1D, s0, b, 0.875) # Medium resolution
# md, nv, ne, resolution_factor = adaptive_bamg(md, x_1D, s0, b, 0.75) # fine resolution


mesh_x = md.mesh.x
bed_2d = np.interp(mesh_x, x_1D, b) # Bedrock at all mesh points
surface_2d = np.interp(mesh_x, x_1D, s0) # Surface at all mesh points

# GEOMETRY SETTINGS  
md.geometry.surface = surface_2d
md.geometry.bed = bed_2d
md.geometry.thickness = surface_2d - bed_2d
md.geometry.base = md.geometry.bed

# visual_mesh_check(md) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# constants and material properties
md.constants.yts = yts
# rheology
md.materials.rheology_B = rheology_B * np.ones(nv)
md.materials.rheology_n = rheology_n * np.ones(ne)
md.materials.rheology_law = "BuddJacka"
md.materials.rho_ice = rho_ice

# where the ice is grounded and floating in the domain
md.mask.ocean_levelset = np.ones(nv) # grounded ice if positive floating ice if negative, grounding line lies where its zero
md.mask.ice_levelset = -np.ones(nv) #Ice is present if md.mask.ice_levelset is negative, icefront lies where its zero
# find nodes on the terminus vertical face and set icefront there
terminus_nodes = np.where(md.mesh.vertexflags(2))[0]
md.mask.ice_levelset[terminus_nodes] = 0          # ice front at terminus
md.mask.ocean_levelset[terminus_nodes] = -1       # hydrostatic pressure at terminus
######################################################################################################################################

md = setflowequation(md, 'FS', 'all')

# velocity initialisation
md.initialization.vx = np.zeros(nv)
md.initialization.vy = np.zeros(nv)
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

md = setup_friction(md, exp)
# diagnostic
debug_friction_setup(md)

## non-periodic BCS
md = setup_non_periodic_boundary_conditions(md)
# diagnostic
md = diagnose_boundary_conditions(md)

# misc
md.stressbalance.abstol = np.nan
md.stressbalance.FSreconditioning = 1
md.stressbalance.shelf_dampening = 1
md.masstransport.isfreesurface = 1
md.groundingline.migration = 'None'

md.miscellaneous.name = f'{BEDROCK_PROFILE_ID:03d}_{exp}_{resolution_factor}'

# solver settings    
md.settings.solver_residue_threshold = 1e-3 #<<<<
md.stressbalance.restol = 1e-3 #<<<<
md.stressbalance.reltol = 1e-3 #<<<<
md.stressbalance.maxiter = 200 #<<< 50


print("\n===== Solving Diagnostic Stressbalance =====")
# diagnostic solve
md = solve(md,'Stressbalance')

##################################################################################################################
print("\n===== DIAGNOSING ACCELERATION =====")
x_sorted, vx_sorted, dvx_dx = diagnose_acceleration_onset(md, L)

# Optional: Plot the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(x_sorted/1000, vx_sorted, 'b-', linewidth=2)
ax1.set_xlabel('Distance from inlet (km)')
ax1.set_ylabel('Velocity (m/yr)')
ax1.set_title('Velocity Profile')
ax1.grid(True)

ax2.plot(x_sorted/1000, dvx_dx, 'r-', linewidth=2)
ax2.set_xlabel('Distance from inlet (km)')
ax2.set_ylabel('dVx/dx (1/yr)')
ax2.set_title('Acceleration Profile')
ax2.grid(True)
ax2.axhline(y=1e-3, color='k', linestyle='--', label='Threshold')
ax2.legend()

plt.tight_layout()
plt.savefig(f"acceleration_diagnostic_{BEDROCK_PROFILE_ID:03d}_{exp}.png")
# plt.show()
plt.close()
################################################################################################################

# Save in ISMIP-HOM format text file (velocity only for now)
try:
    filename = save_results(md, L, md.miscellaneous.name)
    print(f"✓ output saved: {filename}")
except Exception as e:
    print(f"⚠ Error saving output: {e}")
    import traceback
    traceback.print_exc()

# TEST driving stress difference between boundaries:
surface_slope, bed_slope, thickness = analyse_driving_stress(md, L)


print("\n===== Solving Transient Full-Stokes =====")
# transient solve
md = Solve(md) 

# # Check transient fields   
plot_transient_fields(md)

print("Solving complete - saving results")
# Keep NetCDF export for analysis
output_filename = f"{md.miscellaneous.name}_{final_time=}_yrs_timestep={timestep:.5f}_yrs.nc"
export_netCDF(md, output_filename)

print(f"✓ Full results saved to {output_filename}")

## check max velocity evolution
plot_max_velocity_from_netcdf(output_filename)
