# flowline_periodic.py - Flowband model with cosine undulations, flat in and out flow bedrock and periodic BCs
# Ana Fabela Hinojosa

import os
import numpy as np
import matplotlib.pyplot as plt
from model import model
from scipy.interpolate import interp1d
from bamgflowband import bamgflowband
from verbose import verbose
from generic import generic
from solve import solve
from export_netCDF import export_netCDF
# Import from new config file
from configp import config


def setup_model():
    """Set up the model with bedrock undulations in slope-parallel coordinates"""
    # Create a higher-resolution, uniform grid for more accurate representation
    x_transformed = np.linspace(config.x_params['start'], 
                        config.x_params['end'],
                        int((config.x_params['end'] - config.x_params['start'])/config.x_params['step']) + 1)
    
    # Get bedrock elevation with exact wavelength
    b_transformed = config.get_bedrock_elevation(x_transformed)
    
    # Set ice thickness
    mean_thickness = config.ice_params['mean_thickness']
    h_transformed = mean_thickness * np.ones_like(x_transformed)
    
    # Create model with mesh settings
    md = bamgflowband(model(), x_transformed, b_transformed + h_transformed, b_transformed, 'hmax', config.mesh_hmax)
    
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

    # Adjust loading force for slope-parallel coordinates
    md = adjust_loading_force(md, config.alpha)

    # Apply all settings from config
    md = config.setup_model_settings(md)
    
    # Misc
    md.miscellaneous.name = config.name
    md.cluster = generic('np', config.num_processors)
    md.groundingline.migration = 'None'
    
    # Add explicit temporary folder to avoid conflicts
    md.settings.temporary_folder = '/tmp/issm_flowline_periodic'
    
    return md


def adjust_loading_force(md, alpha):
    """Adjust loading force for slope-parallel coordinate system"""
    # In slope-parallel coordinates, gravity components are:
    g_parallel = config.g * np.sin(alpha)      # Along x' (down-slope)
    g_perpendicular = config.g * np.cos(alpha) # Along z' (perpendicular to slope)

    # Initialize loadingforce array
    num_vertices = md.mesh.numberofvertices
    md.stressbalance.loadingforce = np.zeros((num_vertices, 3))

    # Set forces in slope-parallel coordinates
    md.stressbalance.loadingforce[:, 0] = g_parallel     # x' component
    md.stressbalance.loadingforce[:, 1] = -g_perpendicular  # z' component

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


def Solve(md):
    """Solve the model using Elmer Transient Full-Stokes with optimized solver settings"""
    print('===== Solving Elmer Transient Full-Stokes with Enhanced Convergence =====')   
    
    # Apply configuration settings
    md.flowequation.fe_FS = config.solver_settings['flowequation_fe_FS']
    
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
    
    # Set up output settings
    # md.verbose = verbose('convergence', config.output_settings['verbose'])
    md.transient.requested_outputs = config.output_settings['requested_outputs']
    
    # Set callback
    md.timestepping.step_callback = step_callback
    
    # Print solver configuration
    print("\nSolver Configuration:")
    print(f"  Timestep: {md.timestepping.time_step} years")
    print(f"  Duration: {md.timestepping.final_time - md.timestepping.start_time} years")
    print(f"  Max nonlinear iterations: {md.stressbalance.maxiter}")
    print(f"  Convergence criteria: {md.stressbalance.convergence}")
    print(f"  Relative tolerance: {md.stressbalance.reltol}")
    
    return solve(md, 'Transient')


if __name__== '__main__':
    # Set up the model using configuration
    print("\n1. Setting up model...")
    md = setup_model()

    # Set what physics are active
    md.transient.issmb = 0 # no surface mass balance
    md.transient.ismasstransport = 1
    md.transient.isthermal = 1
    # md.transient.isstressbalance = 1
    # md.transient.isgroundingline = 0 # no GL migration
    # # md.transient.isgia = 0  # no postglacial
    # # md.transient.isdamageevolution = 0 # no damage evolution
    # # md.transient.ishydrology = 0: # no hydrology solution

    # Debug: Show basic mesh information
    basal_nodes = np.where(md.mesh.vertexflags(1))[0]
    print(f"Mesh has {md.mesh.numberofvertices} vertices, {len(basal_nodes)} are basal nodes")


    # Debug
    print(f"Before solve, flow equation type: {md.flowequation.fe_FS}")
    md.flowequation.fe_FS = 'P1P1'  # Force it here
    # Call the solver
    md = Solve(md)
    print(f"After explicit set, flow equation type: {md.flowequation.fe_FS}")
    
    
    print("\n3. Saving results...")
    export_netCDF(md, 'ice_flow_results.nc')
    
    # Plot final friction coefficient
    md = config.plot_friction_coefficient(md, is_final_step=True)
    
    print("\nSimulation complete!")