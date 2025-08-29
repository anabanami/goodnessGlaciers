# flowline_synthetic.py - Flowband model with synthetic bedrock profiles
# Ana Fabela Hinojosa

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from model import model
from scipy.interpolate import interp1d
from bamgflowband import bamgflowband
from verbose import verbose
from generic import generic
from solve import solve
from export_netCDF import export_netCDF
from config_synthetic import ModelConfig
import pickle
import time
from datetime import datetime
import shutil

def parse_arguments():
    """Parse command line arguments for profile selection"""
    parser = argparse.ArgumentParser(description='Run ice flow simulation with synthetic bedrock profile')
    parser.add_argument('--profile', type=int, default=1, 
                        help='Profile ID to use (based on existing profiles)')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    args = parser.parse_args()
    
    return args


def setup_model(config):
    """Set up the model with bedrock undulations in slope-parallel coordinates"""
    x_transformed = np.linspace(config.x_params['start'], 
                        config.x_params['end'],
                        int((config.x_params['end'] - config.x_params['start'])/config.x_params['step']) + 1)
    
    # Get bedrock elevation with exact wavelength from the profile
    b_transformed = config.get_bedrock_elevation(x_transformed)
    
    # Set ice thickness
    mean_thickness = config.ice_params['mean_thickness']
    h_transformed = mean_thickness * np.ones_like(x_transformed)

    # Pass scalar hmax to bamgflowband
    md = bamgflowband(model(), x_transformed, b_transformed + h_transformed, b_transformed, 'hmax', config.mesh_hmax)

    # Show basic mesh information
    basal_nodes = np.where(md.mesh.vertexflags(1))[0]
    print("\n============================")
    print(f"Mesh has {md.mesh.numberofvertices} vertices, {len(basal_nodes)} are basal nodes")
    print(f"Mesh has {md.mesh.numberofelements} elements")
    print("=========================\n")


    # Ensure the mesh points are within the interpolation domain
    mesh_x = np.clip(md.mesh.x, 
                    config.x_params['start'], 
                    config.x_params['end'] - 1e-10)
    
    # Use linear interpolation to preserve wavelength
    surface_interpolant = interp1d(x_transformed, b_transformed + h_transformed, kind='linear')
    base_interpolant = interp1d(x_transformed, b_transformed, kind='linear')

    # Apply interpolation to mesh points
    md.geometry.surface = surface_interpolant(mesh_x)
    md.geometry.base = base_interpolant(mesh_x)
    md.geometry.thickness = md.geometry.surface - md.geometry.base
    md.geometry.bed = md.geometry.base

    # Store coordinate system information
    md.miscellaneous.slope_angle = config.alpha
    md.miscellaneous.is_transformed_coordinates = True

    # Apply all settings from config
    md = config.setup_model_settings(md)

    # Adjust loading force for slope-parallel coordinates
    md = adjust_loading_force(md, config.alpha, config)
    
    # Misc
    md.miscellaneous.name = config.name
    md.cluster = generic('np', config.num_processors)
    md.groundingline.migration = 'None'
    
    # Mesh check - LB code
    print("\n===Checking mesh===")
    temp_filename = f"temp_mesh_check_{config.profile_id}.nc"
    export_netCDF(md, temp_filename)
    import sys
    sys.path.append('/home/ana/pyISSM/src')
    import pyissm as issm
    ana = issm.model.io.load_model(temp_filename)
    ana_mesh, ana_x, ana_y, ana_elements, ana_is3d = issm.model.mesh.process_mesh(ana)
    # issm.plot.plot_mesh2d(ana_mesh, show_nodes = True)
    issm.plot.plot_model_nodes(ana, ana.mask.ice_levelset, ana.mask.ocean_levelset, s=4, type='ice_front_nodes')
    plt.show()
    # # Field check - LB code
    # print("\n===Checking bed===")
    # issm.plot.plot_model_field(md, md.geometry.bed, show_mesh=True, show_cbar=True)
    # plt.show()

    # # Clean up
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    
    return md


def adjust_loading_force(md, alpha, config):
    """Adjust loading force with gradual reduction near terminus for stability"""
    # In slope-parallel coordinates, gravity components are:
    g_parallel = config.g * np.sin(alpha)      # Along x' (down-slope)
    g_perpendicular = config.g * np.cos(alpha) # Along y' (perpendicular to slope)
    
    # Get mesh coordinates
    x_coords = md.mesh.x
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    domain_length = x_max - x_min
    
    # Define transition zone near terminus (last 10% of domain)
    transition_length = 0.1 * domain_length
    transition_start = x_max - transition_length
    
    # Create smooth transition factor (1 = full force, 0 = no force at terminus)
    transition_factor = np.ones(len(x_coords))
    transition_mask = (x_coords > transition_start)
    
    if np.any(transition_mask):
        # Smooth cubic transition from 1 to 0.1 (not zero to avoid complete singularity)
        normalized_pos = (x_coords[transition_mask] - transition_start) / transition_length
        # Cubic polynomial: starts at 1, ends at 0.1, with smooth derivatives
        cubic_factor = 1.0 - 0.9 * (3 * normalized_pos**2 - 2 * normalized_pos**3)
        transition_factor[transition_mask] = cubic_factor
    
    print(f"Loading force transition: {np.min(transition_factor):.3f} to {np.max(transition_factor):.3f}")
    
    # Set forces with transition factor
    md.stressbalance.loadingforce[:, 0] = g_parallel * config.rho_ice * transition_factor       # x' component
    md.stressbalance.loadingforce[:, 1] = -g_perpendicular * config.rho_ice * transition_factor   # y' component
    md.stressbalance.loadingforce[:, 2] = 0.0

    return md


def Solve(md, config):
    """Solve the model using TaylorHood Transient Full-Stokes with optimized solver settings"""
    print('\n===== Solving TaylorHood Transient Full-Stokes =====')   
    
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
    md.settings.solver_residue_threshold = config.solver_settings['solver_residue_threshold']
    md.stressbalance.maxiter = config.solver_settings['maxiter']
    md.stressbalance.min_iterations = config.solver_settings['min_iterations']
    md.stressbalance.stabilization = config.solver_settings['stabilization']
    md.flowequation.augmented_lagrangian_r = config.solver_settings['augmented_lagrangian_r']
    
    # Set up output settings
    # md.verbose = verbose('convergence', config.output_settings['verbose'])
    md.transient.requested_outputs = config.output_settings['requested_outputs']
    

    # Set callback
    def wrapped_callback(md_step):
        return config.step_callback(md_step, save_checkpoint)

    # Set callback
    md.timestepping.step_callback = wrapped_callback
    # Debug: verify callback is set
    print(f"DEBUG: Callback set to: {md.timestepping.step_callback}")

    # Print solver configuration
    print("\nSolver Configuration:")
    print(f"  Timestep: {md.timestepping.time_step} years")
    print(f"  Duration: {md.timestepping.final_time - md.timestepping.start_time} years")
    print(f"  Max nonlinear iterations: {md.stressbalance.maxiter}")
    print(f"  Convergence criteria: {md.stressbalance.convergence}")
    print(f"  Relative tolerance: {md.stressbalance.reltol}")

    print("\n=========================")
    print(f"THICKNESS  {np.min(md.geometry.thickness), np.max(md.geometry.thickness)}")
    print(f"BASE  {np.min(md.geometry.base), np.max(md.geometry.base)}")
    print(f"SURFACE  {np.min(md.geometry.surface), np.max(md.geometry.surface)}")
    print("=========================\n")

    print("vx init range:", np.min(md.initialization.vx), np.max(md.initialization.vx))
    print("friction range:", np.min(md.friction.C), np.max(md.friction.C))
    print("loadingforce x-range:", np.min(md.stressbalance.loadingforce[:, 0]), np.max(md.stressbalance.loadingforce[:, 0]))

    return solve(md, 'Transient')


def save_checkpoint(md, config, output_dir='checkpoints'):
    """Save a checkpoint of the current model state"""
    # Create checkpoints directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate checkpoint filename with timestamp and current timestep
    step = md.timestepping.step_counter
    current_time = md.timestepping.start_time + (step * md.timestepping.time_step)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_filename = f"{output_dir}/checkpoint_{config.name}_step{step}_t{current_time:.1f}_{timestamp}.pkl"
    
    # Extract essential restart data
    restart_data = {
        'step_counter': md.timestepping.step_counter,
        'start_time': current_time,
        'final_time': md.timestepping.final_time,
        'results': md.results if hasattr(md, 'results') else None,
        'profile_id': config.profile_id,
        'timestamp': timestamp
    }
    
    # Save checkpoint data
    try:
        with open(checkpoint_filename, 'wb') as f:
            pickle.dump(restart_data, f)
        
        # Also save a "latest" checkpoint that will be overwritten each time
        latest_checkpoint = f"{output_dir}/latest_{config.name}.pkl"
        shutil.copy(checkpoint_filename, latest_checkpoint)
        
        print(f"Checkpoint saved: {checkpoint_filename}")
        return True
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        return False


def load_checkpoint(checkpoint_file):
    """Load a checkpoint file"""
    try:
        with open(checkpoint_file, 'rb') as f:
            restart_data = pickle.load(f)
        return restart_data
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


def check_for_restart():
    """Check if there's a checkpoint to restart from"""
    parser = argparse.ArgumentParser(description='Run ice flow simulation with synthetic bedrock profile')
    parser.add_argument('--profile', type=int, default=1, 
                        help='Profile ID to use (based on existing profiles)')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--restart', type=str, default=None,
                        help='Restart from checkpoint file (or use "latest" for latest checkpoint)')
    args = parser.parse_args()
    
    if args.restart:
        checkpoint_file = args.restart
        # If "latest" specified, find the latest checkpoint for this profile
        if checkpoint_file.lower() == 'latest':
            checkpoint_dir = 'checkpoints'
            if not os.path.exists(checkpoint_dir):
                print("No checkpoints directory found. Starting a new simulation.")
                return args, None
            
            # Look for latest checkpoint with this profile ID
            pattern = f"latest_flowline9_profile_{args.profile:03d}.pkl"
            latest_file = os.path.join(checkpoint_dir, pattern)
            
            if os.path.exists(latest_file):
                checkpoint_file = latest_file
                print(f"Found latest checkpoint: {checkpoint_file}")
            else:
                print(f"No latest checkpoint found for profile {args.profile}. Starting a new simulation.")
                return args, None
                
        # Load the checkpoint
        if os.path.exists(checkpoint_file):
            restart_data = load_checkpoint(checkpoint_file)
            if restart_data:
                print(f"Restarting from checkpoint: {checkpoint_file}")
                print(f"Resuming from time step {restart_data['step_counter']} at t={restart_data['start_time']:.1f} yr")
                return args, restart_data
        
        print(f"Checkpoint file not found or invalid: {checkpoint_file}")
        print("Starting a new simulation.")
    
    return args, None


def diagnose_boundary_conditions(md):
    """Diagnose boundary condition setup to find NaN issues"""
    print("\n=== BOUNDARY CONDITION DIAGNOSTICS ===")
    
    # Check all vertex flags
    for flag in range(1, 10):
        nodes = np.where(md.mesh.vertexflags(flag))[0]
        if len(nodes) > 0:
            x_coords = md.mesh.x[nodes]
            y_coords = md.mesh.y[nodes]
            print(f"Flag {flag}: {len(nodes)} nodes")
            print(f"  X range: {np.min(x_coords):.0f} to {np.max(x_coords):.0f}")
            print(f"  Y range: {np.min(y_coords):.0f} to {np.max(y_coords):.0f}")
            
            # Check what these flags typically mean:
            if flag == 1:
                print("  (Flag 1 = bed/base)")
            elif flag == 2:
                print("  (Flag 2 = terminus/ice front)")
            elif flag == 3:
                print("  (Flag 3 = surface)")
            elif flag == 4:
                print("  (Flag 4 = inlet)")
    
    # Check velocity constraints
    vx_constrained = ~np.isnan(md.stressbalance.spcvx)
    vy_constrained = ~np.isnan(md.stressbalance.spcvy)
    
    print(f"\nVelocity constraints:")
    print(f"  Vx constrained: {np.sum(vx_constrained)} nodes")
    print(f"  Vy constrained: {np.sum(vy_constrained)} nodes")
    
    # Check for over-constrained nodes (both vx and vy constrained)
    over_constrained = vx_constrained & vy_constrained
    print(f"  Over-constrained nodes (both vx and vy): {np.sum(over_constrained)}")
    
    if np.any(vx_constrained):
        vx_vals = md.stressbalance.spcvx[vx_constrained] # Convert to m/yr
        print(f"  Vx constraint values: {np.min(vx_vals):.1f} to {np.max(vx_vals):.1f} m/yr")
    
    if np.any(vy_constrained):
        vy_vals = md.stressbalance.spcvy[vy_constrained] # Convert to m/yr
        print(f"  Vy constraint values: {np.min(vy_vals):.3f} to {np.max(vy_vals):.3f} m/yr")
    
    # Check loading force
    lf = md.stressbalance.loadingforce
    print(f"\nLoading force:")
    print(f"  Array shape: {lf.shape}")
    print(f"  X component: min={np.min(lf[:,0]):.1f}, max={np.max(lf[:,0]):.1f} N/m³")
    if lf.shape[1] > 1:
        print(f"  Y component: min={np.min(lf[:,1]):.1f}, max={np.max(lf[:,1]):.1f} N/m³")
    if lf.shape[1] > 2:
        print(f"  Z component: min={np.min(lf[:,2]):.1f}, max={np.max(lf[:,2]):.1f} N/m³")
    
    # Check pressure
    pressure = md.initialization.pressure

    print(f"\nInitial pressure:")
    print(f"  Min: {np.min(pressure)/1e6:.2f} MPa, Max: {np.max(pressure)/1e6:.2f} MPa")
    print(f"  Mean: {np.mean(pressure)/1e6:.2f} MPa")
    
    # Check for potential issues
    print(f"\nPotential issues:")
    if np.sum(over_constrained) > 0:
        print(f"  ⚠ {np.sum(over_constrained)} over-constrained nodes detected!")
        over_constrained_nodes = identify_over_constrained_nodes(md)
        visualize_constrained_nodes(md)
    
    if np.sum(vx_constrained) == 0 and np.sum(vy_constrained) == 0:
        print(f"  ⚠ No velocity constraints - system may be under-constrained!")

    else:
        print("¯\_(ツ)_/¯ ")    
    print("=====================================\n")


def identify_over_constrained_nodes(md):
    """Identify and analyze over-constrained nodes in detail"""
    print("\n=== DETAILED OVER-CONSTRAINED NODE ANALYSIS ===")
    
    # Find over-constrained nodes
    vx_constrained = ~np.isnan(md.stressbalance.spcvx)
    vy_constrained = ~np.isnan(md.stressbalance.spcvy)
    over_constrained = vx_constrained & vy_constrained
    
    over_constrained_indices = np.where(over_constrained)[0]
    
    print(f"Found {len(over_constrained_indices)} over-constrained nodes:")
    
    for i, node_idx in enumerate(over_constrained_indices):
        print(f"\nNode {i+1}: Index {node_idx}")
        print(f"  Coordinates: x={md.mesh.x[node_idx]:.1f}, y={md.mesh.y[node_idx]:.1f}")
        print(f"  Vx constraint: {md.stressbalance.spcvx[node_idx]:.3f} m/yr")
        print(f"  Vy constraint: {md.stressbalance.spcvy[node_idx]:.3f} m/yr")
        
        # Check which vertex flags this node has
        print(f"  Vertex flags:")
        for flag in range(1, 10):
            flag_nodes = np.where(md.mesh.vertexflags(flag))[0]
            if node_idx in flag_nodes:
                flag_meaning = {
                    1: "bed/base", 2: "terminus/ice front", 3: "surface", 
                    4: "inlet", 5: "outlet", 6: "lateral boundary"
                }
                meaning = flag_meaning.get(flag, "unknown")
                print(f"    Flag {flag}: {meaning}")
        
        # Check if it's at domain boundaries
        x_coord = md.mesh.x[node_idx]
        y_coord = md.mesh.y[node_idx]
        
        x_min, x_max = np.min(md.mesh.x), np.max(md.mesh.x)
        y_min, y_max = np.min(md.mesh.y), np.max(md.mesh.y)
        
        boundary_info = []
        if abs(x_coord - x_min) < 1e-6:
            boundary_info.append("left boundary (inlet)")
        if abs(x_coord - x_max) < 1e-6:
            boundary_info.append("right boundary (terminus)")
        if abs(y_coord - y_min) < 1e-6:
            boundary_info.append("bottom boundary (bed)")
        if abs(y_coord - y_max) < 1e-6:
            boundary_info.append("top boundary (surface)")
            
        if boundary_info:
            print(f"  Located at: {', '.join(boundary_info)}")
    
    return over_constrained_indices

# Also add a visualization function
def visualize_constrained_nodes(md):
    """Create a plot showing all constrained nodes"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    
    # Plot all mesh nodes
    plt.scatter(md.mesh.x, md.mesh.y, c='gray', s=1, alpha=1, label='All nodes')
    
    # Plot nodes with vx constraints
    vx_constrained = ~np.isnan(md.stressbalance.spcvx)
    if np.any(vx_constrained):
        plt.scatter(md.mesh.x[vx_constrained], md.mesh.y[vx_constrained], 
                   c='blue', s=20, marker='>', label='Vx constrained', alpha=0.7)
    
    # Plot nodes with vy constraints  
    vy_constrained = ~np.isnan(md.stressbalance.spcvy)
    if np.any(vy_constrained):
        plt.scatter(md.mesh.x[vy_constrained], md.mesh.y[vy_constrained],
                   c='green', s=20, marker='^', label='Vy constrained', alpha=0.7)
    
    # Highlight over-constrained nodes
    over_constrained = vx_constrained & vy_constrained
    if np.any(over_constrained):
        plt.scatter(md.mesh.x[over_constrained], md.mesh.y[over_constrained],
                   c='red', s=100, marker='X', label='OVER-CONSTRAINED', 
                   edgecolors='black', linewidth=2)
        
        # Annotate over-constrained nodes
        over_indices = np.where(over_constrained)[0]
        for idx in over_indices:
            plt.annotate(f'Node {idx}', 
                        (md.mesh.x[idx], md.mesh.y[idx]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    # plt.xlabel('X coordinate (m)')
    # plt.ylabel('Y coordinate (m)')
    # plt.title('Boundary Condition Constraints Visualization')
    # plt.legend()
    # plt.grid(True, linestyle=":", alpha=0.3)
    # plt.axis('equal')
    # plt.tight_layout()
    # plt.savefig('constrained_nodes_visualization.png', dpi=150, bbox_inches='tight')
    # plt.show()
    
    # print("Constraint visualization saved as 'constrained_nodes_visualization.png'")


if __name__== '__main__':
    
    # Check for restart and parse arguments
    args, restart_data = check_for_restart()
    
    # Create output directory if it doesn't exist
    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a new config with specified profile
    config = ModelConfig(profile_id=args.profile)
    
    config.verify_units()

    # If restarting from checkpoint
    if restart_data:
        # Set up the model using configuration
        print("1. Setting up model for restart")
        md = setup_model(config)
        
        # Set what physics are active
        md.transient.issmb = 0
        md.transient.ismasstransport = 1
        md.transient.isthermal = 1
        
        # Update timestepping settings from checkpoint
        md.timestepping.step_counter = restart_data['step_counter']
        md.timestepping.start_time = restart_data['start_time']
        md.timestepping.final_time = restart_data['final_time']
        
        # Copy results from checkpoint if available
        if restart_data['results']:
            md.results = restart_data['results']
        
        print(f"Restarting from time {md.timestepping.start_time} yr (step {md.timestepping.step_counter})")
        print(f"Will continue to final time {md.timestepping.final_time} yr")
        
    else:
        # Set up a new model
        print("1. Setting up new model")
        md = setup_model(config)
        
        # Add boundary condition diagnostics
        print("\n1.5. Diagnosing boundary conditions")
        diagnose_boundary_conditions(md)

        # Set what physics are active
        md.transient.issmb = 0
        md.transient.ismasstransport = 1
        md.transient.isthermal = 1
    
    # Generate the initial friction coefficient plot if not restarting
    if not restart_data:
        print("2. Plotting initial friction coefficient")
        config.plot_friction_coefficient(md, is_first_step=True)

    print("\n3. Trying steady-state solve")
    try:
        print("Solving steady-state stress balance")
        
        md_steady = solve(md, 'Stressbalance')

        print("\n=== STEADY-STATE DIAGNOSTICS ===")
        vx = md_steady.results.StressbalanceSolution.Vx
        vy = md_steady.results.StressbalanceSolution.Vy
        pressure = md_steady.results.StressbalanceSolution.Pressure

        print(f"Velocity X: min={np.min(vx):.1f}, max={np.max(vx):.1f}, mean={np.mean(vx):.1f} m/yr")
        print(f"Velocity Y: min={np.min(vy):.1f}, max={np.max(vy):.1f}, mean={np.mean(vy):.1f} m/yr") 
        print(f"Pressure: min={np.min(pressure):.0f}, max={np.max(pressure):.0f}, mean={np.mean(pressure):.0f} Pa")
        print("================================")

        md = md_steady  # Use steady result as initial condition

        print("\n✓ Steady-state successful! Now trying transient...")
        
        md = Solve(md, config)
        
    except Exception as e:
        print(f"\nSteady-state solve failed: {e}")
        print("The problem is in the fundamental model setup, not time stepping.")
        raise e
    
    # Calculate average velocity for the final plot
    basal_nodes = np.where(md.mesh.vertexflags(1))[0]
    if hasattr(md.results, 'StressbalanceSolution') and hasattr(md.results.StressbalanceSolution, 'Vx'):
        vx_basal = md.results.StressbalanceSolution.Vx[basal_nodes]
    elif hasattr(md.results, 'TransientSolution') and hasattr(md.results.TransientSolution, 'Vx'):
        vx_basal = md.results.TransientSolution.Vx[basal_nodes]
    elif hasattr(md.results, 'Vx'):
        vx_basal = md.results.Vx[basal_nodes]
    else:
        vx_basal = md.initialization.vx[basal_nodes]
    
    avg_velocity = np.mean(np.abs(vx_basal))
    
    # # Generate the final friction coefficient plot
    # print("\n4. Plotting final friction coefficient")
    # config.plot_friction_coefficient(md, is_final_step=True, velocity=avg_velocity)
    
    print("\n5. Saving results")
    # Create unique filename with profile ID
    output_file = os.path.join(output_dir, f'ice_flow_results_profile_{args.profile:03d}.nc')
    export_netCDF(md, output_file)
    
    # Also save a plot of the bed profile used
    plt.figure(figsize=(10, 5))
    x_domain = np.linspace(config.x_params['start'], config.x_params['end'], 500)
    bed = config.get_bedrock_elevation(x_domain)
    
    plt.plot(x_domain, bed, 'k-', linewidth=2)
    plt.plot(x_domain, (bed + config.ice_params['mean_thickness']), 'b-', linewidth=1.5)
    plt.fill_between(x_domain, bed, bed + config.ice_params['mean_thickness'], color='lightblue', alpha=0.5)
    
    plt.xlabel('Distance (km)')
    plt.ylabel('Elevation (km)')
    plt.title(f'Profile {args.profile}: λ={config.profile_params["wavelength"]:.2f}km, '
              f'A={config.profile_params["amplitude"]:.4f}m')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'bed_profile_{args.profile:03d}.png'), dpi=150)
    
    print(f"\nSimulation complete! Results saved to: {output_file}")