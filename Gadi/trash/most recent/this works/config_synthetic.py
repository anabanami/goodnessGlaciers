# config_synthetic.py - Flowband model with bedrock from synthetic profiles
# Ana Fabela Hinojosa

import numpy as np
import matplotlib.pyplot as plt
from setflowequation import setflowequation
from bedrock_generator import SyntheticBedrockModelConfig
# Import the fixed Budd theory implementation
# from Budd_theory import BuddTheory
from Friction_theory import FrictionTheory
# Detect running environment and set processor count accordingly
import os
import socket
from cuffey import cuffey


class ModelConfig(SyntheticBedrockModelConfig):
    def __init__(self, profile_id=1, output_dir="bedrock_profiles"):
        # Initialize the base SyntheticBedrockModelConfig
        super().__init__(profile_id, output_dir)
        
        # Initialize physical parameters
        self.init_physical_parameters()
        
        # Initialize solver parameters
        self.init_solver_parameters()
        
        # Model identifier - CHANGED FOR NEW VERSION
        self.name = f'flowline_profile_{profile_id:03d}'


    def verify_units(self):
        """
        Optional diagnostic function to verify unit consistency
        """
        print("\n=== Unit Verification ===")
        
        # Check spatial units
        print(f"\nDomain length: {(self.x_params['end'] - self.x_params['start'])/1e3:.1f} km")
        print(f"Resolution: {self.x_params['step']:.1f} m")
        
        # Check time units
        # print(f"Time conversion factor: {self.yts:.0f} s/year")
        print(f"\nSimulation duration: {self.time_settings['final_time']} years")
        print(f"Simulation time step: {self.time_settings['time_step']} years")            
        print("=========================\n")

    def debug_coordinate_system(self):
        """Debug the coordinate system and loading force calculation"""
        print(f"=== COORDINATE SYSTEM DEBUG ===")
        print(f"Base slope: {self.base_slope}")
        print(f"Alpha (radians): {self.alpha}")
        print(f"Alpha (degrees): {np.degrees(self.alpha)}")
        print(f"sin(alpha): {np.sin(self.alpha)}")
        print(f"cos(alpha): {np.cos(self.alpha)}")
        
        # Expected forces
        g_parallel_expected = self.g * np.sin(self.alpha)
        g_perpendicular_expected = self.g * np.cos(self.alpha)
        force_parallel_expected = g_parallel_expected * self.rho_ice
        force_perpendicular_expected = -g_perpendicular_expected * self.rho_ice
        
        print(f"g_parallel expected: {g_parallel_expected:.6f} m/s²")
        print(f"g_perpendicular expected: {g_perpendicular_expected:.6f} m/s²")
        print(f"Force parallel expected: {force_parallel_expected:.1f} N/m³")
        print(f"Force perpendicular expected: {force_perpendicular_expected:.1f} N/m³")
        print("================================\n")


    def init_physical_parameters(self):
        """Initialize physical parameters"""
        # Ice physics
        self.rheology_n = 3.0 # Glen's flow law exponent
        self.rho_ice = 917.0  # kg/m^3
        self.friction_m = 3.0  # Weertman exponent (can be 1/3, 1, etc.)
        
        # # Constants
        self.yts = 31556926.0  # seconds per year
        self.g = 9.80665  # gravity acceleration m/s^2

        # Thermal parameters
        self.ice_temperature = 253.15  # -20°C
        self.surface_temperature = 263.15  # -10°C
        self.basal_temperature = 273.15  # Pressure melting point
        self.thermal_conductivity = 2.1  # W/(m·K)
        self.heat_capacity = 2009  # J/(kg·K)


    def init_solver_parameters(self):
        """Initialize solver and numerical parameters"""
        # Time settings
        self.time_settings = {
            'time_step': 1,   # years
            'start_time': 0,
            'final_time': 1,  # years
            'output_frequency': 1 # save every n years
        }
        
        # Solver settings
        self.solver_settings = {
            'flowequation_fe_FS': 'TaylorHood',
            'convergence': 'relative',
            'restol': 0.001,
            'reltol': 0.01,
            'abstol': 10,
            'maxiter': 100,
            'solver_residue_threshold': 1e-2,
            'min_iterations': 3,
            'stabilization': 3,
            'augmented_lagrangian_r': 100.0
        }
        
        # Output settings
        self.output_settings = {
            'requested_outputs': ['default', 'Vx', 'Vy'],
            'verbose': True
        }
        
        # Mass transport settings
        self.mass_transport_params = {
            'hydrostatic_adjustment': 'Incremental',
            'min_thickness': 0.01
        }
        
        # Other parameters
        # Processor selection using hostname to determine if we're on Gadi
        hostname = socket.gethostname()
        
        # Check if we're on the Gadi cluster
        if 'gadi' in hostname.lower():
            # We're on Gadi
            if 'PBS_NCPUS' in os.environ:
                self.num_processors = int(os.environ.get('PBS_NCPUS'))
                print(f"\nGadi cluster detected ({hostname}). Using {self.num_processors} processors from PBS_NCPUS.")
        else:
            # We're on local machine - always use 1 processor
            self.num_processors = 1
            print(f"\nLocal environment detected ({hostname}). Using 1 processor.")
    
        # self.fs_reconditioning = 1.2

    @property
    def omega(self):
        """Calculate angular frequency from wavelength"""
        return 2 * np.pi / self.bedrock_params['lambda']


    def setup_model_settings(self, md):
        """Configure core model settings"""
        # Set up masks
        md.mask.ice_levelset = -np.ones((md.mesh.numberofvertices))
        md.mask.ice_levelset[np.where(md.mesh.vertexflags(2))] = 0.
        md.mask.ocean_levelset = np.ones((md.mesh.numberofvertices))

        # Apply other settings
        md = self.setup_thermal_model(md)
        md = self.setup_stress_balance(md)
        md = self.setup_mass_balance(md)
        # md = self.setup_budd_sliding(md)
        md = self.setup_friction(md)  
        return md


    def setup_stress_balance(self, md):
        """Configure stress balance settings with improved and robust initial conditions"""
        # Get domain information
        x_coords = md.mesh.x
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        domain_length = x_max - x_min
        
        # Velocity initialisation
        md.initialization.vx = np.zeros(md.mesh.numberofvertices)
        md.initialization.vy = np.zeros(md.mesh.numberofvertices)
        md.initialization.vz = np.zeros(md.mesh.numberofvertices)
        md.initialization.vel = np.zeros(md.mesh.numberofvertices)
        
        # Material properties
        md.materials.rheology_n = self.rheology_n * np.ones(md.mesh.numberofelements)

        # Initialize pressure
        md.initialization.pressure = np.zeros(md.mesh.numberofvertices)
        # hydrostatic seed (ρ g h)
        H = md.geometry.surface - md.geometry.base
        md.initialization.pressure = self.rho_ice * self.g * H
        
        # Initialize stress balance arrays
        md.stressbalance.referential = np.nan * np.ones((md.mesh.numberofvertices, 6))
        md.stressbalance.loadingforce = np.zeros((md.mesh.numberofvertices, 3))
        md.stressbalance.spcvx = np.nan * np.ones((md.mesh.numberofvertices,))
        md.stressbalance.spcvy = np.nan * np.ones((md.mesh.numberofvertices,))
        md.stressbalance.spcvz = np.nan * np.ones((md.mesh.numberofvertices,))

        # Boundary conditions
        # Get boundary node sets
        bed_nodes = np.where(md.mesh.vertexflags(1))[0]
        terminus_nodes = np.where(md.mesh.vertexflags(2))[0] 
        surface_nodes = np.where(md.mesh.vertexflags(3))[0]
        inlet_nodes = np.where(md.mesh.vertexflags(4))[0]  

        # Free boundary at base (no constraints)
        # If spcnormalstress already exists, make sure it's not applied at base
        if hasattr(md.stressbalance, 'spcnormalstress'):
            md.stressbalance.spcnormalstress[terminus_nodes] = np.nan

        # Free boundary at terminus (no constraints)
        if hasattr(md.stressbalance, 'spcnormalstress'):
            md.stressbalance.spcnormalstress[terminus_nodes] = np.nan

        # Free boundary at surface (no constraints)
        if hasattr(md.stressbalance, 'spcnormalstress'):
            md.stressbalance.spcnormalstress[surface_nodes] = np.nan

        # # Dirichlet inlet boundary
        md.stressbalance.spcvx[inlet_nodes] = 0.0  

        # Set up solver parameters
        md.stressbalance.abstol = np.nan
        md.stressbalance.reltol = self.solver_settings['reltol']
        md.stressbalance.maxiter = self.solver_settings['maxiter']
        md.flowequation.augmented_lagrangian_r = self.solver_settings['augmented_lagrangian_r']
        # md.stressbalance.FSreconditioning = self.fs_reconditioning
        md.stressbalance.convergence = 'relative'

        md = setflowequation(md, 'FS', 'all')
        
        return md


    def setup_mass_balance(self, md):
        """Configure mass balance settings"""
        # Initialize surface mass balance
        md.smb.mass_balance = np.zeros(md.mesh.numberofvertices)
        
        # Set up mass transport configuration
        md.masstransport.spcthickness = np.nan * np.ones((md.mesh.numberofvertices,))
        md.masstransport.min_thickness = self.mass_transport_params['min_thickness']
        md.masstransport.hydrostatic_adjustment = self.mass_transport_params['hydrostatic_adjustment']
        md.masstransport.isfreesurface = 1
        
        md.basalforcings.groundedice_melting_rate = np.zeros(md.mesh.numberofvertices)
        md.basalforcings.floatingice_melting_rate = np.zeros(md.mesh.numberofvertices)
        md.basalforcings.geothermalflux = np.zeros(md.mesh.numberofvertices)

        return md


    def setup_thermal_model(self, md):
        """Configure thermal model settings"""
        # Basic thermal fields
        md.thermal.spctemperature = np.nan * np.ones((md.mesh.numberofvertices))
        md.initialization.temperature = self.ice_temperature * np.ones(md.mesh.numberofvertices)
        md.thermal.temperature = self.ice_temperature * np.ones(md.mesh.numberofvertices)

        # # # Material properties - use temperature-dependent rheology
        md.materials.rheology_B = cuffey(md.initialization.temperature) 
        md.materials.rheology_law = 'BuddJacka'
        print(f"Cuffey B values - min: {np.min(md.materials.rheology_B):.2e}, max: {np.max(md.materials.rheology_B):.2e}, mean: {np.mean(md.materials.rheology_B):.2e} Pa·s^(1/n)")
        print(f"Temperature input to cuffey: {np.mean(md.initialization.temperature):.1f} K")
        print(f"rheology law: {md.materials.rheology_law}")
    
        md.thermal.conductivity = self.thermal_conductivity * np.ones(md.mesh.numberofvertices)
        md.thermal.capacity = self.heat_capacity * np.ones(md.mesh.numberofvertices)
        
        # Heat sources
        md.thermal.internal_heating = np.zeros(md.mesh.numberofvertices)
        md.thermal.friction_heating = np.zeros(md.mesh.numberofvertices)
        
        return md


    def setup_friction(self, md):
        # create friction implementation instance
        fric_impl = FrictionTheory(self)
        md = fric_impl.calculate_basal_friction(md)
        print("Successfully applied friction")
        return md


    def plot_friction_coefficient(self, md, is_first_step=False, is_final_step=False, velocity=None):
        """Plot friction coefficient"""
        basal_nodes = np.where(md.mesh.vertexflags(1))[0]
        x_coords = md.mesh.x[basal_nodes]
        friction_vals = md.friction.coefficient[basal_nodes]
        
        # Get the bed elevation for basal nodes (scaled to fit in the same plot)
        bed_elevation = md.geometry.bed[basal_nodes]
        
        # Determine plot type and filename
        if is_first_step:
            title = "Initial Friction Coefficient"
            filename = "initial_friction.png"
        elif is_final_step:
            title = f"Final Friction Coefficient (Avg Velocity: {velocity:.4f} m/yr)" if velocity else "Final Friction Coefficient"
            filename = "final_friction.png"
        else:
            return md  # No plot for intermediate steps
        
        # Sort by x-coordinate for a clean plot
        sort_idx = np.argsort(x_coords)
        x_sorted = x_coords[sort_idx]
        bed_sorted = bed_elevation[sort_idx]

        # Scatter plot
        if is_first_step or is_final_step:
            plt.figure(figsize=(12, 6))
            
            # First, plot the bed profile (scaled down and shifted up to fit in same view)
            # Find min/max of bed
            bed_min = bed_sorted.min()
            bed_max = bed_sorted.max()
            # Scale bed profile to stay within plot bounds
            bed_range = bed_max - bed_min
            if bed_range > 0:
                scaled_bed = 0.03 * (bed_sorted - bed_min) / bed_range - 0.03  # Scale to fit within -0.03 to 0
            else:
                scaled_bed = np.zeros_like(bed_sorted) - 0.03
            
            # Then add the friction coefficient scatter plot on top
            sc = plt.scatter(x_sorted, scaled_bed, 
                      s=40, c=friction_vals, cmap='viridis')
            plt.colorbar(sc, label="Friction coefficient")
            plt.title(title)
            plt.grid(True, linestyle=":", color='k', alpha=0.4)
            plt.xlabel("Distance along flowline (m)")
            plt.ylim([-0.05, 0.05])
            
            if is_first_step:
                plt.savefig(f"{filename.replace('.png', '_scatter.png')}")
            else:  # final step
                plt.savefig(f"{filename.replace('.png', '_scatter.png')}")
            plt.close()
        
        return md


    def step_callback(self, md_step, save_checkpoint_func):
        """Update sliding coefficient"""
        step = md_step.timestepping.step_counter
        current_time = md_step.timestepping.start_time + (step * md_step.timestepping.time_step)

        if step % 2 == 0 or step % 2 == 1:  # Check both step 8 and step 7
            print(f"DEBUG: Step modulo check: step={step}, step%2={step%2}")
            if step % 2 == 0:
                budd_impl = BuddTheoryImplementation(self)
                current_velocity = budd_impl.estimate_velocity_magnitude(md_step)
                print(f"Step {step}: Mean surface velocity = {current_velocity:.2f} m/yr")
        # # Existing sliding coefficient update logic
        # if step % 8 == 0:
        #     print(f"Time step {step}: t = {current_time} yr - Updating sliding coefficient")
        #     md_step = self.update_sliding_coefficient(md_step)
        
        return md_step
