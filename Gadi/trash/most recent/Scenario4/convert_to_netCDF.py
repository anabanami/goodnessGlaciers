import sys
import numpy as np
import matplotlib.pyplot as plt
from model import model
from scipy.interpolate import interp1d
from bamgflowband import bamgflowband
from verbose import verbose
from generic import generic
from solve import solve
from export_netCDF import export_netCDF
import argparse
import os
import re
from configf9_synthetic import ModelConfig
from loadresultsfromdisk import loadresultsfromdisk


def parse_arguments():
    """Parse command line arguments for profile selection and input file"""
    parser = argparse.ArgumentParser(description='Run ice flow simulation with synthetic bedrock profile')
    parser.add_argument('--profile', type=int, default=1, 
                        help='Profile ID to use (based on existing profiles)')
    parser.add_argument('--input', type=str, default='flowline9_synthetic.outbin',
                        help='Input .outbin file to convert')
    parser.add_argument('--output', type=str, default='',
                        help='Output filename (without extension). If not specified, derives from input filename.')
    parser.add_argument('--output-dir', type=str, default='',
                        help='Output directory for results. If not specified, uses same directory as input file.')
    args = parser.parse_args()
    
    return args


def extract_profile_from_filename(filename):
    """Extract profile number from the filename pattern like flowline9_profile_002.outbin"""
    # Try to extract profile number using regex
    match = re.search(r'profile_(\d+)', filename)
    if match:
        return int(match.group(1))
    
    # Try other patterns
    match = re.search(r'_(\d{2,3})\.outbin$', filename)
    if match:
        return int(match.group(1))
        
    # Default to None if no pattern found
    return None



def adjust_loading_force(md, alpha, config):
    """Adjust loading force with gradual reduction near terminus for stability"""
    # In slope-parallel coordinates, gravity components are:
    g_parallel = config.g * np.sin(alpha)      # Along x' (down-slope)
    g_perpendicular = config.g * np.cos(alpha) # Along z' (perpendicular to slope)

    # Initialize loadingforce array
    num_vertices = md.mesh.numberofvertices
    md.stressbalance.loadingforce = np.zeros((num_vertices, 3))
    
    # Get coordinates for transition zone
    x_coords = md.mesh.x
    terminus_x = np.max(x_coords)
    transition_start = terminus_x - 3.0  # Start transition 3km from terminus
    
    # Create smooth transition factor (1 = full force, 0 = no force at terminus)
    transition_factor = np.ones(num_vertices)
    transition_mask = (x_coords > transition_start)
    if np.any(transition_mask):
        # Calculate normalized position in transition zone (1 at start, 0 at terminus)
        normalized_pos = (terminus_x - x_coords[transition_mask]) / (terminus_x - transition_start)
        transition_factor[transition_mask] = normalized_pos
    
    # Set forces with transition factor
    md.stressbalance.loadingforce[:, 0] = g_parallel * transition_factor     # x' component
    md.stressbalance.loadingforce[:, 1] = -g_perpendicular * transition_factor  # z' component

    return md


def step_callback(md_step):
    """Update sliding coefficient after each time step"""
    # Get current step number
    step = md_step.timestepping.step_counter
    current_time = md_step.timestepping.start_time + (step * md_step.timestepping.time_step)
    
    # Only print status every 10 steps to reduce console output
    if step % 10 == 0:
        print(f"Time step {step}: t = {current_time} yr - Updating sliding coefficient")
    
    # Update sliding coefficient based on new velocity field
    md_step = config.update_sliding_coefficient(md_step)
    
    return md_step


# Set callback - NOTE: This needs to be defined after config is created
def wrapped_callback(md_step):
    return config.step_callback(md_step, save_checkpoint)


def Solve(md, config):
    """Solve the model using TaylorHood Transient Full-Stokes with optimized solver settings"""
    print('===== Solving TaylorHood Transient Full-Stokes with Enhanced Convergence =====')   
    
    # Apply configuration settings
    md.flowequation.fe_FS = config.solver_settings['flowequation_fe_FS']
    # md.flowequation.p1p1_stabilization = 'SUPG'  # Add stabilization specific to P1P1

    # Set up time steps
    md.timestepping.time_step = config.time_settings['time_step']
    md.timestepping.start_time = config.time_settings['start_time']
    md.timestepping.final_time = config.time_settings['final_time']
    md.settings.output_frequency = config.time_settings['output_frequency']
    
    # Apply solver settings
    md.stressbalance.convergence = config.solver_settings['convergence']
    md.stressbalance.restol = config.solver_settings['restol']
    md.stressbalance.reltol = config.solver_settings['reltol']
    md.stressbalance.abstol = config.solver_settings['abstol']
    md.stressbalance.maxiter = config.solver_settings['maxiter']
    md.stressbalance.min_iterations = config.solver_settings['min_iterations']
    md.stressbalance.stabilization = config.solver_settings['stabilization']
    md.flowequation.augmented_lagrangian_r = config.solver_settings['augmented_lagrangian_r']
    md.settings.solver_residue_threshold = 5e-3
    
    # Set up output settings
    # md.verbose = verbose('convergence', config.output_settings['verbose'])
    md.transient.requested_outputs = config.output_settings['requested_outputs']
    
    # Set callback
    md.timestepping.step_callback = wrapped_callback
    
    # Print solver configuration
    print("\nSolver Configuration:")
    print(f"  Timestep: {md.timestepping.time_step} years")
    print(f"  Duration: {md.timestepping.final_time - md.timestepping.start_time} years")
    print(f"  Max nonlinear iterations: {md.stressbalance.maxiter}")
    print(f"  Convergence criteria: {md.stressbalance.convergence}")
    print(f"  Relative tolerance: {md.stressbalance.reltol}")

    return solve(md, 'Transient')


def setup_model_for_loading(config, terminus_refinement_km=0.5, refinement_factor=4.0):
    """
    Set up the model EXACTLY like the original simulation but skip solving
    This ensures perfect compatibility for loading results
    """
    print(f"\n=== Setting up model structure for loading (no solving) ===")
    print(f"Terminus refinement zone: {terminus_refinement_km} km")
    print(f"Refinement factor: {refinement_factor}x finer")
    
    # Steps 1-12: EXACTLY the same as your original setup_model function
    
    # Step 1: Create variable resolution x-coordinates
    x_start = config.x_params['start']
    x_end = config.x_params['end']
    base_resolution = config.x_params['step']
    fine_resolution = base_resolution / refinement_factor
    
    # Define zones
    terminus_start = x_end - terminus_refinement_km
    transition_width = 0 # km - smooth transition zone
    transition_start = terminus_start - transition_width
    
    # Build x-coordinates with variable resolution
    x_coords = []
    
    # 1. Coarse resolution from start to transition zone
    if transition_start > x_start:
        x_coarse = np.arange(x_start, transition_start, base_resolution)
        x_coords.extend(x_coarse)
    
    # 2. Transition zone with gradually increasing resolution
    if transition_width > 0:
        n_transition_points = int(transition_width / fine_resolution)
        x_transition = np.linspace(transition_start, terminus_start, n_transition_points)
        x_coords.extend(x_transition)
    
    # 3. Fine resolution in terminus zone
    x_fine = np.arange(terminus_start, x_end + fine_resolution/2, fine_resolution)
    x_coords.extend(x_fine)
    
    # Remove duplicates and sort
    x_transformed = np.unique(np.array(x_coords))
    
    # Ensure we end exactly at x_end
    if x_transformed[-1] < x_end:
        x_transformed = np.append(x_transformed, x_end)
    
    print(f"Mesh statistics:")
    print(f"  Total points: {len(x_transformed)}")
    print(f"  Coarse resolution: {base_resolution:.3f} km")
    print(f"  Fine resolution: {fine_resolution:.3f} km")
    print(f"  Points in terminus zone: {np.sum(x_transformed >= terminus_start)}")
    
    # Step 2: Get bedrock elevation with exact wavelength from the profile
    b_transformed = config.get_bedrock_elevation(x_transformed)
    
    # Step 3: Set ice thickness
    mean_thickness = config.ice_params['mean_thickness']
    h_transformed = mean_thickness * np.ones_like(x_transformed)
    
    # Step 4: Use fine mesh size for initial mesh generation
    initial_mesh_size = config.mesh_hmax / refinement_factor
    print(f"Using uniform fine mesh size: {initial_mesh_size:.3f} km for initial generation")
    
    # Step 5: Create mesh using bamgflowband with uniform fine mesh size
    x_meters = x_transformed * 1000
    surface_meters = (b_transformed + h_transformed) * 1000  
    bed_meters = b_transformed * 1000
    hmax_meters = initial_mesh_size * 1000

    md = bamgflowband(model(), x_meters, surface_meters, bed_meters, 'hmax', hmax_meters)

    # Convert mesh coordinates back to km for internal consistency
    md.mesh.x = md.mesh.x / 1000
    md.mesh.y = md.mesh.y / 1000

    # Step 6: Apply variable mesh refinement after mesh generation
    terminus_x = np.max(md.mesh.x)
    
    # Create variable mesh size based on actual mesh coordinates
    mesh_x_coords = md.mesh.x
    variable_mesh_sizes = np.full(md.mesh.numberofvertices, initial_mesh_size)
    
    # Make regions outside terminus coarser
    coarse_region = mesh_x_coords < (terminus_x - terminus_refinement_km)
    transition_region = ((mesh_x_coords >= (terminus_x - terminus_refinement_km - transition_width)) & 
                        (mesh_x_coords < (terminus_x - terminus_refinement_km)))
    
    # Apply coarse mesh to non-terminus regions
    variable_mesh_sizes[coarse_region] = config.mesh_hmax
    
    # Apply transition smoothing
    if np.any(transition_region):
        transition_x = mesh_x_coords[transition_region]
        transition_start_x = terminus_x - terminus_refinement_km - transition_width
        terminus_start_x = terminus_x - terminus_refinement_km
        
        # Normalize transition progress (0 = coarse, 1 = fine)
        progress = (transition_x - transition_start_x) / transition_width
        fine_size = config.mesh_hmax / refinement_factor
        coarse_size = config.mesh_hmax
        variable_mesh_sizes[transition_region] = coarse_size - progress * (coarse_size - fine_size)
    
    # Apply the variable mesh sizes to bamg if available
    if hasattr(md, 'bamg'):
        md.bamg.hmax = variable_mesh_sizes
        print(f"Applied variable mesh refinement: {variable_mesh_sizes.min():.3f} to {variable_mesh_sizes.max():.3f} km")
    
    # Store in mesh object as well
    md.mesh.hmax = variable_mesh_sizes
        
    # Step 7: Ensure the mesh points are within the interpolation domain
    mesh_x = np.clip(md.mesh.x, 
                    config.x_params['start'], 
                    config.x_params['end'] - 1e-10)
    
    # Step 8: Use linear interpolation to preserve wavelength
    from scipy.interpolate import interp1d
    surface_interpolant = interp1d(x_transformed, b_transformed + h_transformed, kind='linear')
    base_interpolant = interp1d(x_transformed, b_transformed, kind='linear')

    # Apply interpolation to mesh points
    md.geometry.surface = surface_interpolant(mesh_x)
    md.geometry.base = base_interpolant(mesh_x)
    md.geometry.thickness = md.geometry.surface - md.geometry.base
    md.geometry.bed = md.geometry.base

    # Step 9: Store coordinate system information
    md.miscellaneous.slope_angle = config.alpha
    md.miscellaneous.is_transformed_coordinates = True

    # Step 10: Adjust loading force for slope-parallel coordinates
    md = adjust_loading_force(md, config.alpha, config)

    # Step 11: Apply all settings from config
    md = config.setup_model_settings(md)
    
    # Step 12: Misc settings
    md.miscellaneous.name = config.name
    md.cluster = generic('np', config.num_processors)
    md.groundingline.migration = 'None'
    
    # Step 13: SKIP THE SOLVING PART! Just initialize without solving
    print("Setting up initial fields (no solving)...")
    
    # Just set up basic initialization without solving
    md.initialization.vx = np.maximum(md.initialization.vx, 0.0)
    
    # Initialize pressure field manually (simple hydrostatic)
    md.initialization.pressure = config.rho_ice * config.g * md.geometry.thickness
    
    # Step 14: SKIP pressure smoothing (not needed for loading)
    
    # Step 15: Print final mesh statistics
    print(f"\nFinal mesh statistics:")
    print(f"  Total vertices: {md.mesh.numberofvertices}")
    print(f"  Total elements: {md.mesh.numberofelements}")
    
    # Count vertices in different zones
    terminus_vertices = np.sum(md.mesh.x >= (terminus_x - terminus_refinement_km))
    transition_vertices = np.sum(((md.mesh.x >= (terminus_x - terminus_refinement_km - transition_width)) & 
                                 (md.mesh.x < (terminus_x - terminus_refinement_km))))
    coarse_vertices = np.sum(md.mesh.x < (terminus_x - terminus_refinement_km - transition_width))
    
    print(f"  Vertices in coarse zone: {coarse_vertices}")
    print(f"  Vertices in transition zone: {transition_vertices}")
    print(f"  Vertices in terminus zone: {terminus_vertices}")
    print(f"=== Model structure ready for loading ===\n")
    
    return md


if __name__== '__main__':
    # Parse command line arguments (your existing code)
    args = parse_arguments()
    
    # Get input file information (your existing code)
    input_file = args.input
    input_basename = os.path.basename(input_file)
    input_dirname = os.path.dirname(input_file) or '.'
    
    # Try to extract profile from filename (your existing code)
    profile_from_file = extract_profile_from_filename(input_basename)
    profile_id = args.profile if profile_from_file is None else profile_from_file
    
    # Create output filename (your existing code)
    if args.output:
        output_basename = args.output
    else:
        input_name = os.path.splitext(input_basename)[0]
        output_basename = f"{input_name}"
        
    if not re.search(r'profile_\d+', output_basename) and not re.search(r'_\d{2,3}$', output_basename):
        output_basename = f"{output_basename}_profile_{profile_id:03d}"
    
    output_dir = args.output_dir or input_dirname
    output_file = os.path.join(output_dir, f"{output_basename}.nc")
    
    # Check if input file exists and backup it
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)
    
    # Create backup BEFORE any model operations
    import shutil
    backup_file = input_file + ".backup"
    print(f"Creating backup: {backup_file}")
    shutil.copy2(input_file, backup_file)
    
    # Create a new config with specified profile
    config = ModelConfig(profile_id=profile_id)

    print(f"\nSetting up model structure for profile {profile_id}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Set up the model EXACTLY like your original code but without solving
    md = setup_model_for_loading(config, terminus_refinement_km=5.0, refinement_factor=4.0)

    # Set what physics are active (for consistency with original)
    md.transient.issmb = 0
    md.transient.ismasstransport = 1
    md.transient.isthermal = 1

    # Show basic mesh information
    basal_nodes = np.where(md.mesh.vertexflags(1))[0]
    print(f"Mesh has {md.mesh.numberofvertices} vertices, {len(basal_nodes)} are basal nodes")

    # Load results from backup file (preserves original)
    print(f"\nLoading results from {backup_file}...")
    md = loadresultsfromdisk(md, backup_file)
    
    # Check if loading was successful
    if md is None or not hasattr(md, 'results'):
        print("ERROR: Failed to load results from file.")
        # Still restore the original file
        if os.path.exists(backup_file):
            shutil.move(backup_file, input_file)
        sys.exit(1)

    print("✅ Results loaded successfully!")
    
    # Restore original file
    if os.path.exists(backup_file):
        shutil.move(backup_file, input_file)
        print("✅ Original file restored")

    # Export to NetCDF
    print(f"\nSaving results to {output_file}...")
    export_netCDF(md, output_file)
    
    print("✅ Conversion completed successfully!")