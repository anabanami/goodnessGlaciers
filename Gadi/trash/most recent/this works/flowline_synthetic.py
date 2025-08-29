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
    
    # Create model with mesh settings
    md = bamgflowband(model(), x_transformed, b_transformed + h_transformed, b_transformed, 'hmax', config.mesh_hmax)

    # Show basic mesh information
    basal_nodes = np.where(md.mesh.vertexflags(1))[0]
    print("=========================================================")
    print(f"Mesh has {md.mesh.numberofvertices} vertices, {len(basal_nodes)} are basal nodes")
    print(f"Mesh has {md.mesh.numberofelements} elements")
    print("=========================================================")


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

    # # Clean up
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    
    return md


def Solve(md, config):
    """Solve the model using TaylorHood Transient Full-Stokes"""
    print('===== Solving TaylorHood Transient Full-Stokes =====')   
    
    # Apply configuration settings
    md.flowequation.fe_FS = config.solver_settings['flowequation_fe_FS']
    # md.verbose = verbose('solution', True, 'solver', True, 'convergence', True)

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

    # # Set callback
    # def wrapped_callback(md_step):
    #     return config.step_callback(md_step, save_checkpoint)

    # # Set callback
    # md.timestepping.step_callback = wrapped_callback
    # # Debug: verify callback is set
    # print(f"DEBUG: Callback set to: {md.timestepping.step_callback}")

    # Print solver configuration
    print("\nSolver Configuration:")
    print(f"  Timestep: {md.timestepping.time_step} years")
    print(f"  Duration: {md.timestepping.final_time - md.timestepping.start_time} years")
    print(f"  Max nonlinear iterations: {md.stressbalance.maxiter}")
    print(f"  Convergence criteria: {md.stressbalance.convergence}")
    print(f"  Relative tolerance: {md.stressbalance.reltol}")

    print("vx init range:", np.min(md.initialization.vx), np.max(md.initialization.vx))
    print("vy init range:", np.min(md.initialization.vy), np.max(md.initialization.vy))

    # print("friction range:", np.min(md.friction.C), np.max(md.friction.C)) # Weertman
    print("friction range:", np.min(md.friction.coefficient), np.max(md.friction.coefficient)) #Budd


    return solve(md, 'Transient')


def check_for_args():
    """Check if there's a checkpoint to restart from"""
    parser = argparse.ArgumentParser(description='Run ice flow simulation with synthetic bedrock profile')
    parser.add_argument('--profile', type=int, default=1, 
                        help='Profile ID to use (based on existing profiles)')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--restart', type=str, default=None,
                        help='Restart from checkpoint file (or use "latest" for latest checkpoint)')
    args = parser.parse_args()
    
    return args


def diagnose_boundary_conditions(md):
    """Diagnose boundary condition setup to find NaN issues"""
    print("\n=========== BOUNDARY CONDITION DIAGNOSTICS ===========")
    
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
    print(f"  Vx constrained: {np.sum(~np.isnan(md.stressbalance.spcvx))} nodes")
    print(f"  Vy constrained: {np.sum(~np.isnan(md.stressbalance.spcvy))} nodes")

    
    # Check for over-constrained nodes (both vx and vy constrained)
    over_constrained = vx_constrained & vy_constrained
    print(f"  Over-constrained nodes (both vx and vy): {np.sum(over_constrained)}")
    
    if np.any(vx_constrained):
        vx_vals = md.stressbalance.spcvx[vx_constrained] # Convert to m/yr
        print(f"  Vx constraint values: {np.min(vx_vals):.1f} to {np.max(vx_vals):.1f} m/yr")
    
    if np.any(vy_constrained):
        vy_vals = md.stressbalance.spcvy[vy_constrained] # Convert to m/yr
        print(f"  Vy constraint values: {np.min(vy_vals):.3f} to {np.max(vy_vals):.3f} m/yr")
    
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

    nan_surface = np.isnan(md.geometry.surface).any()
    nan_base    = np.isnan(md.geometry.base).any()
    if nan_surface or nan_base:
        print("⚠ NaNs detected in geometry:")
        if nan_surface: print("  - md.geometry.surface has NaNs")
        if nan_base:    print("  - md.geometry.base has NaNs")
    
    # 5. Check other critical fields
    # # Weertman
    # if hasattr(md.friction, 'C') and np.isnan(md.friction.C).any():
    #     print("⚠ NaNs in friction.C")

    # Budd
    if hasattr(md.friction, 'coefficient') and np.isnan(md.friction.coefficient).any():
        print("⚠ NaNs in friction.coefficient")
    
    if hasattr(md, 'temperature') and np.isnan(md.temperature).any():
        print("⚠ NaNs in temperature field")
    
    # 6. Final assertion to halt if any NaNs remain
    assert not (nan_surface or nan_base), "Geometry still contains NaNs!"
    print("=========================================================")



if __name__== '__main__':
    
    # Check for restart and parse arguments
    args = check_for_args()
    
    # Create output directory if it doesn't exist
    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a new config with specified profile
    config = ModelConfig(profile_id=args.profile)
    
    config.verify_units()

    # config.debug_coordinate_system()

    # Set up a new model
    print("1. Setting up new model")
    md = setup_model(config)
    
    # Add boundary condition diagnostics
    print("\n1.5. Diagnosing boundary conditions")
    diagnose_boundary_conditions(md)

    # Set what physics are active
    md.transient.issmb = 0
    md.transient.ismasstransport = 0
    md.transient.isthermal = 0


    print("2. Plotting initial friction coefficient")
    # config.plot_friction_coefficient(md, is_first_step=True)
    
    # Call the solver
    print("\n3. Trying steady-state solve\n")
    md_steady = solve(md, 'Stressbalance')

    print("\n=========== STEADY-STATE DIAGNOSTICS ===========")
    vx = md_steady.results.StressbalanceSolution.Vx
    vy = md_steady.results.StressbalanceSolution.Vy
    vel = md_steady.results.StressbalanceSolution.Vel
    pressure = md_steady.results.StressbalanceSolution.Pressure

    print(f"Velocity X: min={np.min(vx):.1f}, max={np.max(vx):.1f}, mean={np.mean(vx):.1f} m/yr")
    print(f"Velocity Y: min={np.min(vy):.1f}, max={np.max(vy):.1f}, mean={np.mean(vy):.1f} m/yr") 
    print(f"Velocity VEL: min={np.min(vel):.1f}, max={np.max(vel):.1f}, mean={np.mean(vel):.1f} m/yr") 

    print(f"Pressure: min={np.min(pressure):.0f}, max={np.max(pressure):.0f}, mean={np.mean(pressure):.0f} Pa")
    
    # C = md_steady.friction.C #Weertman
    C = md_steady.friction.coefficient #Budd
    print(f"Friction C:  min={C.min():.1e}, max={C.max():.1e}, mean={C.mean():.1e} Pa·a·m⁻¹")

    print("Mesh elements:", md_steady.mesh.numberofelements)
    print("Solver residue threshold:", md_steady.settings.solver_residue_threshold)
    print("===============================================\n")

    md = md_steady  # Use steady result as initial condition
    #md.initialization.vx = md_steady.results.StressbalanceSolution.Vx


    print("\n✓ Steady-state successful! Now trying transient...")
    md = Solve(md, config)
    
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
    
    # Generate the final friction coefficient plot
    print("\n4. Plotting final friction coefficient")
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
    plt.plot(x_domain, bed + config.ice_params['mean_thickness'], 'b-', linewidth=1.5)
    plt.fill_between(x_domain, bed, bed + config.ice_params['mean_thickness'], color='lightblue', alpha=0.5)
    
    plt.xlabel('Distance (km)')
    plt.ylabel('Elevation (km)')
    plt.title(f'Profile {args.profile}: λ={config.profile_params["wavelength"]:.2f}km, '
              f'A={config.profile_params["amplitude"]:.4f}m')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'bed_profile_{args.profile:03d}.png'), dpi=150)
    
    print(f"\nSimulation complete! Results saved to: {output_file}")
