from socket import gethostname
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import os
import sys
sys.path.append('/home/ana/pyISSM/src')
import pyissm as issm
from pyissm import plot as iplt
from model import model 
from bamg import bamg
from bamgflowband import bamgflowband
from setflowequation import setflowequation
from friction import friction
from export_netCDF import export_netCDF
from solve import solve

def format_ismip_output(md, exp, L, author_code="ana", model_num=1):
    """
    Format output according to ISMIP‑HOM specifications (velocity only for now)

    Parameters
    ----------
    md : ISSM model object with results
    exp : str            – experiment letter ('A', 'B', 'C', 'D')
    L   : float          – domain length in **metres**
    author_code : str    – 3‑letter author code (default 'ana')
    model_num   : int    – model identifier (default 1)
    """
    print(f"\n=== FORMATTING ISMIP OUTPUT FOR EXP {exp} ===")

    # ------------------------------------------------------------------
    # 0️⃣  Scaled coordinates (always 0–1)
    # ------------------------------------------------------------------
    x_hat = md.mesh.x.flatten() / L
    y_hat = md.mesh.y.flatten() / L

    print(
        f"Coordinate ranges: x_hat=[{x_hat.min():.3f}, {x_hat.max():.3f}], "
        f"y_hat=[{y_hat.min():.3f}, {y_hat.max():.3f}]"
    )

    # ------------------------------------------------------------------
    # 1️⃣  Extract velocity fields (already in **m a⁻¹** on return from ISSM)
    # ------------------------------------------------------------------
    sol = md.results.StressbalanceSolution
    sol = sol[0] if hasattr(sol, "__len__") and len(sol) else sol

    vx_surface = sol.Vx.flatten()
    vy_surface = sol.Vy.flatten() if hasattr(sol, "Vy") else np.zeros_like(vx_surface)
    vz_surface = sol.Vz.flatten() if hasattr(sol, "Vz") else np.zeros_like(vx_surface)

    # Keep *copies* of the full (unfiltered) fields – needed for basal output
    vx_full, vy_full = vx_surface.copy(), vy_surface.copy()

    # ------------------------------------------------------------------
    # 2️⃣  Vertex masks
    # ------------------------------------------------------------------
    surface_idx = np.where(md.mesh.vertexonsurface == 1)[0]

    if exp in {"A", "B"}:
        # surface‑only benchmark
        x_hat, y_hat = x_hat[surface_idx], y_hat[surface_idx]
        vx_surface, vy_surface, vz_surface = (
            vx_surface[surface_idx],
            vy_surface[surface_idx],
            vz_surface[surface_idx],
        )

    elif exp in {"C", "D"}:
        # surface rows for top‑of‑ice columns
        x_hat, y_hat = x_hat[surface_idx], y_hat[surface_idx]
        vx_surface, vy_surface, vz_surface = (
            vx_surface[surface_idx],
            vy_surface[surface_idx],
            vz_surface[surface_idx],
        )

        # matching basal nodes share the same column index ordering in ISSM
        basal_idx = np.where(md.mesh.vertexonbase == 1)[0]
        vx_basal, vy_basal = vx_full[basal_idx], vy_full[basal_idx]

    # ------------------------------------------------------------------
    # 3️⃣  Quick sanity print‑outs
    # ------------------------------------------------------------------
    print(
        f"Surface velocity ranges (m a⁻¹):\n"
        f"  vx: [{vx_surface.min():.2f}, {vx_surface.max():.2f}]\n"
        f"  vy: [{vy_surface.min():.2f}, {vy_surface.max():.2f}]\n"
        f"  vz: [{vz_surface.min():.2f}, {vz_surface.max():.2f}]"
    )

    if exp in {"C", "D"}:
        print(
            "Basal velocity ranges (m a⁻¹):\n"
            f"  vx_basal: [{vx_basal.min():.2f}, {vx_basal.max():.2f}]\n"
            f"  vy_basal: [{vy_basal.min():.2f}, {vy_basal.max():.2f}]"
        )

    # ------------------------------------------------------------------
    # 4️⃣  Assemble ISMIP output arrays
    # ------------------------------------------------------------------
    if exp == "A":   # 5 cols: x̂ ŷ vx(zs) vy(zs) vz(zs)
        output_data = np.column_stack((x_hat, y_hat, vx_surface, vy_surface, vz_surface))
        header = "x_hat y_hat vx_surface vy_surface vz_surface"

    elif exp == "B":  # 3 cols: x̂ vx(zs) vz(zs)  (flow‑band variant handled later)
        y_range = md.mesh.y.max() - md.mesh.y.min()
        if md.mesh.dimension == 2 or y_range < L / 2:
            output_data = np.column_stack((x_hat, vx_surface, vz_surface))
        else:
            centre = np.isclose(y_hat, 0.5, atol=0.01)
            output_data = np.column_stack((x_hat[centre], vx_surface[centre], vz_surface[centre]))
        header = "x_hat vx_surface vz_surface"

    elif exp == "C":  # 7 cols: x̂ ŷ vx(zs) vy(zs) vz(zs) vx(zb) vy(zb)
        output_data = np.column_stack(
            (
                x_hat,
                y_hat,
                vx_surface,
                vy_surface,
                vz_surface,
                vx_basal,
                vy_basal,
            )
        )
        header = "x_hat y_hat vx_surface vy_surface vz_surface vx_basal vy_basal"

    elif exp == "D":  # 4 cols: x̂ vx(zs) vz(zs) vx(zb)
        y_range = md.mesh.y.max() - md.mesh.y.min()
        if md.mesh.dimension == 2 or y_range < L / 2:
            output_data = np.column_stack((x_hat, vx_surface, vz_surface, vx_basal))
        else:
            centre = np.isclose(y_hat, 0.5, atol=0.01)
            output_data = np.column_stack(
                (x_hat[centre], vx_surface[centre], vz_surface[centre], vx_basal[centre])
            )
        header = "x_hat vx_surface vz_surface vx_basal"

    # ------------------------------------------------------------------
    # 5️⃣  Write to disk
    # ------------------------------------------------------------------
    filename = f"{author_code}{model_num}{exp.lower()}{int(L/1000):03d}.txt"
    np.savetxt(filename, output_data, fmt="%.6f", delimiter="\t", header=header, comments="# ")

    print(f"✓ Saved {filename} with shape {output_data.shape}")
    print("First 5 rows:")
    print(output_data[:5])

    return filename, output_data


def save_ismip_results(md, exp, L):
    
    print(f"\n=== SAVING ISMIP-HOM OUTPUT FOR EXP {exp}, L={L/1000}km ===")
    
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
    
    # Try to format and save - this will help us debug further
    try:
        filename, data = format_ismip_output(md, exp, L, 
                                           author_code="ana",  # Your initials
                                           model_num=1)
        return filename
    except Exception as e:
        print(f"Error in format_ismip_output: {e}")
        return None


def choose_mesh_spacings(L, exp):
    """
    Rules:
    - Base: 5km domain with hmax=80m, nvert=10, nx=50  
    - hmax scaling: 25% for each doubling
    - TAR = hmax/nvert = 8
    - Geometry: spacing = hmax/1.6
    """
    if exp in ('B', 'D'):
        print(f"\n=== Flowband EXPERIMENT {exp} ===")
        # ===== base case: 5km =====
        base_L = 5000
        base_hmax = 80
        nx_base = 80
        nvert = 20

        if L <= base_L:
            # 5km or smaller
            hmax_xy = int(base_hmax/ 2)
            nx = int(nx_base / 2)

        else:
            # 25% for each increase in domain length
            doublings = np.log2(L / base_L) # example: np.log2(40000 / base_L) = 3.0
            hmax_xy = base_hmax * (1.25 ** doublings)
            # round
            hmax_xy = round(hmax_xy, 1) if hmax_xy >= 100 else round(hmax_xy)
            # ===== GEOMETRY SAMPLING =====
            geometry_spacing = hmax_xy / 1.6 # 
            nx = int(max(nx_base, geometry_spacing))  # establish minimum as nx_base

    else:
        # ===== 3D CASES - COARSER RESOLUTION =====
        print(f"\n=== 3D EXPERIMENT {exp} ===")

        if L == 10000:
            hmax_xy = 300   # m

        elif L == 20000:
            hmax_xy = 600   # m
        
        elif L == 40000:
            hmax_xy = 2000   # m

        elif L > 40000:
            # fall back for largest domain
            hmax_xy = 6000   # m

        else: 
            # coarser base resolution for 3D
            hmax_xy = 150   # m
        
        nvert = 8   # vertical layers
        nx = int(L / hmax_xy) + 1  # geometry points

    # ===== COMMON CALCULATIONS =====
    elements_along_flow = L / hmax_xy
    aspect_ratio = hmax_xy / nvert
    L_km = L / 1000
    
    # ===== QUALITY CHECKS =====
    print(f"\nMesh for L = {L_km:.0f} km")
    print(f"   hmax_xy  = {hmax_xy:.0f} m")
    print(f"   nvert    = {nvert}")
    print(f"   nx       = {nx}")
    print(f"   Elements along-flow = {elements_along_flow:.0f}")
    
    return nvert, hmax_xy, nx


def calculate_relative_depth(md, vertices, mesh_type='3d'):
    """Unified relative depth for both 3D and flowband"""
    surface_z = md.geometry.surface[vertices]
    bed_z = md.geometry.bed[vertices]
    
    if mesh_type == 'flowband':
        vertex_z = md.mesh.y[vertices]  # flowband: y is actually z
    else:  # 3D
        vertex_z = md.mesh.z[vertices]  # 3D: z is z
    
    thickness = surface_z - bed_z
    relative_depth = (vertex_z - bed_z) / thickness
    return relative_depth


def setup_3d_periodic_boundaries(md, tolerance=1e-6):
    """
    Final corrected version for ISMIP-HOM with sloping geometry.
    Matches relative positions within ice column, not absolute z-coordinates.
    """
    print("=== SETTING UP 3D PERIODIC BOUNDARY CONDITIONS (FINAL FIX) ===")
    
    # Get mesh boundaries
    x_min, x_max = np.min(md.mesh.x), np.max(md.mesh.x)
    y_min, y_max = np.min(md.mesh.y), np.max(md.mesh.y)
    
    print(f"Domain bounds: x=[{x_min:.1f}, {x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}]")
    
    # Find ALL vertices on each boundary
    left_vertices = np.where(np.abs(md.mesh.x - x_min) < tolerance)[0]
    right_vertices = np.where(np.abs(md.mesh.x - x_max) < tolerance)[0]
    bottom_vertices = np.where(np.abs(md.mesh.y - y_min) < tolerance)[0]
    top_vertices = np.where(np.abs(md.mesh.y - y_max) < tolerance)[0]
    
    print(f"Found boundary vertices:")
    print(f"  Left (x={x_min:.1f}): {len(left_vertices)} vertices")
    print(f"  Right (x={x_max:.1f}): {len(right_vertices)} vertices") 
    print(f"  Bottom (y={y_min:.1f}): {len(bottom_vertices)} vertices")
    print(f"  Top (y={y_max:.1f}): {len(top_vertices)} vertices")
    
    def find_periodic_pairs_with_geometry(vertices1, vertices2, match_coord_idx, name):
        """
        Match vertices based on one coordinate plus relative depth in ice column.
        match_coord_idx: 0 for x, 1 for y
        """
        print(f"\n--- Pairing {name} boundaries (GEOMETRY-AWARE) ---")
        
        if match_coord_idx == 0:  # Match x-coordinates (for bottom-top)
            coord1 = md.mesh.x[vertices1]
            coord2 = md.mesh.x[vertices2]
            coord_name = "x"
        else:  # Match y-coordinates (for left-right)
            coord1 = md.mesh.y[vertices1] 
            coord2 = md.mesh.y[vertices2]
            coord_name = "y"
        
        # Calculate relative depths
        rel_depth1 = calculate_relative_depth(md, vertices1, mesh_type='3d')
        rel_depth2 = calculate_relative_depth(md, vertices2, mesh_type='3d')
        
        print(f"  Matching {coord_name}-coordinate + relative depth")
        print(f"  Boundary 1: {coord_name}=[{coord1.min():.1f}, {coord1.max():.1f}], rel_depth=[{rel_depth1.min():.3f}, {rel_depth1.max():.3f}]")
        print(f"  Boundary 2: {coord_name}=[{coord2.min():.1f}, {coord2.max():.1f}], rel_depth=[{rel_depth2.min():.3f}, {rel_depth2.max():.3f}]")
        
        pairs = []
        used_indices2 = set()
        
        # Tolerances
        coord_tolerance = max(1e-3, tolerance * 100)
        depth_tolerance = 1e-6  # Very tight tolerance for relative depth
        
        for i, (v1, c1, d1) in enumerate(zip(vertices1, coord1, rel_depth1)):
            best_match = None
            best_distance = float('inf')
            
            for j, (v2, c2, d2) in enumerate(zip(vertices2, coord2, rel_depth2)):
                if j in used_indices2:
                    continue
                
                # Calculate combined distance: coordinate + relative depth
                coord_dist = abs(c1 - c2)
                depth_dist = abs(d1 - d2)
                
                if coord_dist < coord_tolerance and depth_dist < depth_tolerance:
                    total_distance = coord_dist + 1000 * depth_dist  # Weight depth matching heavily
                    if total_distance < best_distance:
                        best_distance = total_distance
                        best_match = j
            
            if best_match is not None:
                pairs.append([v1, vertices2[best_match]])
                used_indices2.add(best_match)
        
        print(f"  Found {len(pairs)} pairs")
        print(f"  Unmatched from boundary 1: {len(vertices1) - len(pairs)}")
        print(f"  Unmatched from boundary 2: {len(vertices2) - len(pairs)}")
        
        # Show first few pairs for validation
        if len(pairs) > 0:
            print(f"  First 3 pairs:")
            for i in range(min(3, len(pairs))):
                v1, v2 = pairs[i]
                coord1_val = md.mesh.x[v1] if match_coord_idx == 0 else md.mesh.y[v1]
                coord2_val = md.mesh.x[v2] if match_coord_idx == 0 else md.mesh.y[v2]
                depth1 = calculate_relative_depth(md, [v1], mesh_type='3d')[0]
                depth2 = calculate_relative_depth(md, [v2], mesh_type='3d')[0]
                print(f"    v{v1+1} ({coord_name}={coord1_val:.1f}, rel_depth={depth1:.3f}) ↔ v{v2+1} ({coord_name}={coord2_val:.1f}, rel_depth={depth2:.3f})")
        
        return np.array(pairs) if pairs else np.empty((0, 2), dtype=int)
    
    # Left-right pairing: match y-coordinates + relative depth
    print("\n=== LEFT-RIGHT PAIRING (GEOMETRY-AWARE) ===")
    lr_pairs = find_periodic_pairs_with_geometry(left_vertices, right_vertices, 1, "left-right")
    
    # Bottom-top pairing: match x-coordinates + relative depth
    print("\n=== BOTTOM-TOP PAIRING (GEOMETRY-AWARE) ===")
    bt_pairs = find_periodic_pairs_with_geometry(bottom_vertices, top_vertices, 0, "bottom-top")
    
    # Combine pairs and convert to 1-based indexing for ISSM
    all_pairs = []
    if len(lr_pairs) > 0:
        all_pairs.append(lr_pairs + 1)  # Convert to 1-based
    if len(bt_pairs) > 0:
        all_pairs.append(bt_pairs + 1)  # Convert to 1-based
    
    if len(all_pairs) > 0:
        all_pairs = np.vstack(all_pairs)
        
        # Assign to model
        md.stressbalance.vertex_pairing = all_pairs
        md.masstransport.vertex_pairing = all_pairs
        
        print(f"\n=== FINAL PAIRING SUMMARY ===")
        print(f"Total vertex pairs: {len(all_pairs)}")
        print(f"Left-right pairs: {len(lr_pairs)}")
        print(f"Bottom-top pairs: {len(bt_pairs)}")
        
        # Final validation with geometry info
        print(f"\nFinal validation (first 3 pairs):")
        for i in range(min(3, len(all_pairs))):
            v1_idx, v2_idx = all_pairs[i] - 1  # Convert back to 0-based
            v1_coords = [md.mesh.x[v1_idx], md.mesh.y[v1_idx], md.mesh.z[v1_idx]]
            v2_coords = [md.mesh.x[v2_idx], md.mesh.y[v2_idx], md.mesh.z[v2_idx]]
            
            # Show relative depths
            v1_rel_depth = calculate_relative_depth(md, [v1_idx], mesh_type='3d')[0]
            v2_rel_depth = calculate_relative_depth(md, [v2_idx], mesh_type='3d')[0]
            
            distance = np.linalg.norm(np.array(v1_coords) - np.array(v2_coords))
            print(f"  Pair {i+1}: v{all_pairs[i,0]} {v1_coords} (rel_depth={v1_rel_depth:.3f}) ↔ v{all_pairs[i,1]} {v2_coords} (rel_depth={v2_rel_depth:.3f})")
    else:
        print("⚠ ERROR: No vertex pairs found!")
        return md
    
    # Success checks
    if len(lr_pairs) > 0:
        print(f"\n✅ SUCCESS: Found {len(lr_pairs)} left-right pairs using geometry-aware matching!")
    else:
        print(f"\n⚠ CRITICAL ERROR: Still no left-right pairs found!")
    
    if len(bt_pairs) > 0:
        print(f"✅ SUCCESS: Found {len(bt_pairs)} bottom-top pairs")
    else:
        print(f"⚠ CRITICAL ERROR: No bottom-top pairs found!")
    
    return md





# ===================================================================================================================================================

### THINGS TO PLAY WITH ####
# # Choose experiment
# exp = 'A'
# exp = 'B' # FLOWBAND
# exp = 'C'
exp = 'D' # FLOWBAND

# spatial domain
domain_length_km = [5, 10, 20, 40, 80, 160]
domain_length_m = [L * 1e3 for L in domain_length_km] # m


# ===================================================================================================================================================

# slope
alpha_deg  = 0.5 if exp in ('A', 'B') else 0.1     # °  → 0.5° for A,B, 0.1° for C,D
alpha = np.deg2rad(alpha_deg) # radians
print(f"{alpha_deg = }")

# RUN EXPERIMENTS
for L in domain_length_m:
    print(f"\n{L = }")
    omega = 2 * np.pi / L

    # initialise model
    md = model()
    nvert, hmax_xy, nx = choose_mesh_spacings(L, exp)
    x_1D = np.linspace(0, L, nx)
    s0 = -x_1D * np.tan(alpha)
    
    # rectangular mesh for A, C
    rect = {
        'x': np.array([0, L, L, 0, 0], dtype=float),
        'y': np.array([0, 0, L, L, 0], dtype=float),
        'nods': 5 # 5 points because the contour is closed
    }

    if exp == 'A':
        # generate 2D mesh with dummy variables
        md = bamg(md,'domain', [rect],  'hmax', hmax_xy, 'nvert', nvert)  
        x = md.mesh.x
        y = md.mesh.y
        s = - x * np.tan(alpha)
        b = s - 1000 + 500 * np.sin(omega * x) * np.sin(omega * y)

    elif exp == 'B':
        # generate 1D mesh with dummy variables
        b0 = s0 - 1000 + 500 * np.sin(omega * x_1D)
        md = bamgflowband(md, x_1D, s0, b0, 'hmax', hmax_xy, 'nvert', nvert)
        x = md.mesh.x
        print(f"{np.size(x)}")
        s = - x * np.tan(alpha)
        b = s - 1000 + 500 * np.sin(omega * x)
    
    elif exp == 'C':
        # generate 2D mesh with dummy variables
        md = bamg(md,'domain', [rect],  'hmax', hmax_xy, 'nvert', nvert)  
        x = md.mesh.x
        y = md.mesh.y
        s = - x * np.tan(alpha)
        b = s - 1000

    else:
        # generate 1D mesh with dummy variables
        b0 = s0 - 1000
        md = bamgflowband(md, x_1D, s0, b0, 'hmax', hmax_xy, 'nvert', nvert)
        x = md.mesh.x
        print(f"{np.size(x)}")
        s = - x * np.tan(alpha)
        b = s - 1000
    
    # GEOMETRY SETTINGS
    md.geometry.surface = s
    md.geometry.bed = b
    md.geometry.thickness = s - b
    md.geometry.base = md.geometry.bed

    if exp in ('A', 'C'):
        md = md.extrude(nvert, 1)  # nvert layers, extrusion method 1
        print(f"\nAfter extrusion - Vertices: {md.mesh.numberofvertices}, Elements: {md.mesh.numberofelements}")
        print(f"Mesh dimension: {md.mesh.dimension}")  # Should be 3

    # # visual MESH check
    # md_mesh, md_x, md_y, md_elements, md_is3d = issm.model.mesh.process_mesh(md)
    # iplt.plot_mesh2d(md_mesh, show_nodes = True)    
    # plt.axis('equal')
    # plt.show()

    nv, ne  = md.mesh.numberofvertices, md.mesh.numberofelements

    # constants and material properties
    md.constants.yts = 31556926 # s/y

    # rheology
    rheology_n = 3
    A = 1e-16 / md.constants.yts # from table 1 in Pattyn 2008
    md.materials.rheology_B = A ** (-1/rheology_n) * np.ones(nv)
    md.materials.rheology_n = rheology_n * np.ones(ne)
    md.materials.rheology_law = "BuddJacka"
    md.materials.rho_ice = 910  # kg/m^3

    # flow equation
    md = setflowequation(md, 'FS', 'all')

    # velocity initialisation
    md.initialization.vx = np.zeros(nv)
    md.initialization.vy = np.zeros(nv)
    md.initialization.vel = np.zeros(nv)
    # hydrostatic seed
    md.initialization.pressure = (md.constants.g * md.materials.rho_ice * 
                                (md.geometry.surface - md.geometry.base))
    # temperature
    md.initialization.temperature = (273.15 - 10) * np.ones(nv) # as per pattyn 2008

    # what kind of system? # CHECK THIS IS CORRECT
    md.mask.ocean_levelset = np.ones(nv)
    md.mask.ice_levelset = -np.ones(nv)

    # basal forcing
    md.basalforcings.floatingice_melting_rate = np.zeros(nv)
    md.basalforcings.groundedice_melting_rate = np.zeros(nv)

    # mass balance
    md.smb.mass_balance = np.zeros(nv)

    # Boundary conditions 
    # (temperature, thickness, velocity, # loadingforce, 
    # referential (tells ISSM “here’s the local 2D basis” for the stress‐balance))
    md.thermal.spctemperature = np.full(nv, np.nan)
    md.masstransport.spcthickness = np.full(nv, np.nan)
    md.stressbalance.loadingforce = np.zeros((nv, 3))
    md.stressbalance.referential = np.full((nv, 6), np.nan)

    # SPATIAL BCS 
    md.stressbalance.spcvx = np.full(nv, np.nan)
    md.stressbalance.spcvy = np.full(nv, np.nan)
    md.stressbalance.spcvz = np.full(nv, np.nan)

    ##===========================================================================
    # FRICTION SETUP - HANDLES 3D EXTRUDED MESHES CORRECTLY
    ##===========================================================================

    # Initialize friction object ONCE
    md.friction = friction()
    md.friction.p = np.ones(ne)
    md.friction.q = np.zeros(ne) # pattyn linear sliding means no N_eff

    if exp in ('A', 'B'):
        print(f"\nEXPERIMENT {exp}: basal boundary is frozen (no slip)")
        basal = np.where(md.mesh.vertexonbase == 1)[0]
        md.stressbalance.spcvx[basal] = 0
        md.stressbalance.spcvy[basal] = 0
        md.friction.coefficient = np.ones(nv)

    else:
        print(f"\nExperiment {exp}: sinusoidal Budd β² field")
        
        # Get coordinates - these are for ALL vertices in the mesh (including 3D)
        x = np.asarray(md.mesh.x)
        y = np.asarray(md.mesh.y)
        
        # Ensure 1D arrays
        if x.ndim > 1:
            x = x.flatten()
        if y.ndim > 1:
            y = y.flatten()
        
        print(f"Mesh coordinates:")
        print(f"  Total vertices (nv): {nv}")
        print(f"  x array size: {x.size}")
        print(f"  y array size: {y.size}")
        print(f"  x range: [{np.min(x):.1f}, {np.max(x):.1f}]")
        print(f"  y range: [{np.min(y):.1f}, {np.max(y):.1f}]")
        
        # Verify coordinate array sizes match total vertices
        if x.size != nv:
            raise ValueError(f"x coordinate array size ({x.size}) != nv ({nv})")
        if y.size != nv:
            raise ValueError(f"y coordinate array size ({y.size}) != nv ({nv})")
        
        # Calculate friction coefficient for ALL vertices based on their x,y position
        # For 3D meshes, vertices at different z levels but same x,y get the same friction
        if exp == "C":
            # 2D case: β²(x,y) = 1000 + 1000 sin(ωx) sin(ωy)
            beta2 = 1000.0 + 1000.0 * np.sin(omega * x) * np.sin(omega * y)
            print(f"Applied 2D friction field β²(x,y) = 1000 + 1000*sin(ωx)*sin(ωy)")
        
        elif exp == "D":
            # 1D case: β²(x) = 1000 + 1000 sin(ωx)
            beta2 = 1000.0 + 1000.0 * np.sin(omega * x)
            print(f"Applied 1D friction field β²(x) = 1000 + 1000*sin(ωx)")
        
        else:
            raise ValueError("exp must be A, B, C, or D")
        
        # Verify result array size
        if beta2.size != nv:
            raise ValueError(f"β² array size ({beta2.size}) != nv ({nv})")
        
        print(f"β² field statistics:")
        print(f"  Array size: {beta2.size}")
        print(f"  Range: [{np.min(beta2):.1f}, {np.max(beta2):.1f}] Pa·a·m⁻¹")
        print(f"  Mean: {np.mean(beta2):.1f} Pa·a·m⁻¹")
        
        # Convert to ISSM units and set friction coefficient
        beta2_issm = beta2 * md.constants.yts  # Pa·a·m⁻¹ → Pa·s·m⁻¹
        md.friction.coefficient = np.sqrt(beta2_issm)
        
        print(f"Final friction coefficient:")
        print(f"  Array size: {md.friction.coefficient.size}")
        print(f"  Range: [{np.min(md.friction.coefficient):.1f}, {np.max(md.friction.coefficient):.1f}] Pa·s·m⁻¹")

    ##===========================================================================

    if exp in ('B', 'D'):
        # === 2D PERIODIC BOUNDARIES FOR EXPERIMENTS B & D ===
        tolerance = 1e-6  # Add this line
        # Find left/right boundary vertices (same logic as 3D function)
        x_min, x_max = np.min(md.mesh.x), np.max(md.mesh.x)
        left_vertices = np.where(np.abs(md.mesh.x - x_min) < tolerance)[0]
        right_vertices = np.where(np.abs(md.mesh.x - x_max) < tolerance)[0]

        print(f"Found {len(left_vertices)} left, {len(right_vertices)} right vertices")  # Add diagnostic
        
        # Use the same pairing logic as find_periodic_pairs_with_geometry
        # but match relative depths instead of coordinates
        pairs = []
        used_right = set()
        
        left_rel_depths = calculate_relative_depth(md, left_vertices, mesh_type='flowband')
        right_rel_depths = calculate_relative_depth(md, right_vertices, mesh_type='flowband')
        
        for i, left_idx in enumerate(left_vertices):
            left_depth = left_rel_depths[i]
            
            best_match = None
            best_depth_diff = float('inf')
            
            for j, right_idx in enumerate(right_vertices):
                if j in used_right:
                    continue
                
                right_depth = right_rel_depths[j]
                depth_diff = abs(left_depth - right_depth)
                
                if depth_diff < 1e-1 and depth_diff < best_depth_diff:
                    best_depth_diff = depth_diff
                    best_match = j
            
            if best_match is not None:
                pairs.append([left_idx, right_vertices[best_match]])
                used_right.add(best_match)
        
        # Convert to 1-based and assign (same as 3D function)
        if len(pairs) > 0:
            pairs_1based = np.array(pairs) + 1
            md.stressbalance.vertex_pairing = pairs_1based
            md.masstransport.vertex_pairing = pairs_1based
            print(f"✅ Created {len(pairs)} flowband pairs using relative depth matching")

    else:
        # === 3D PERIODIC BOUNDARIES FOR EXPERIMENTS A & C ===
        md = setup_3d_periodic_boundaries(md, tolerance=1e-6)
        ##===========================================================================


    md.stressbalance.abstol = np.nan
    md.stressbalance.FSreconditioning = 1
    md.stressbalance.shelf_dampening = 1
    md.masstransport.isfreesurface = 1
    md.stressbalance.maxiter = 100

    md.miscellaneous.name = 'Pattyn'

    md.transient.isstressbalance = 1
    md.transient.ismasstransport = 0
    md.transient.issmb = 0
    md.transient.isthermal = 0

    ## Experiment A, B
    # md.settings.solver_residue_threshold = 1e-4
    # md.stressbalance.restol = 1e-5
    # md.stressbalance.reltol = 1e-4

    # Experiment C
    md.settings.solver_residue_threshold = 2e-1
    md.stressbalance.restol = 2e-2
    md.stressbalance.reltol = 2e-1

    md.transient.requested_outputs = ['default', 'Surface', 'Base']

    if hasattr(md, 'friction') and hasattr(md.friction, 'coefficient'):
        print(f"Max friction coefficient: {np.max(md.friction.coefficient)}")
    else:
        print("No friction law applied (no-slip condition)")

    print(f"Max driving stress estimate: {np.max(md.materials.rho_ice * md.constants.g * md.geometry.thickness * np.tan(alpha))}")

    md = solve(md,'Stressbalance')
    print("Solving complete - saving results")

    # Save in ISMIP-HOM format (velocity only for now)
    try:
        ismip_filename = save_ismip_results(md, exp, L)
        print(f"✓ ISMIP output saved: {ismip_filename}")
    except Exception as e:
        print(f"⚠ Error saving ISMIP output: {e}")
        import traceback
        traceback.print_exc()

    # Keep NetCDF export for analysis
    export_netCDF(md, f"{md.miscellaneous.name}_{exp}_{L*1e-3}km.nc")
