# configp.py - Flowband model with bedrock and cosine undulations and periodic boundary conditions
# Ana Fabela Hinojosa

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from setflowequation import setflowequation

class ModelConfig:
    def __init__(self):
        # Initialize coordinate system parameters
        self.init_coordinate_system()
        
        # Initialize domain parameters
        self.init_domain_parameters()
        
        # Initialize physical parameters
        self.init_physical_parameters()
        
        # Initialize solver parameters
        self.init_solver_parameters()
        
        # Model identifier - CHANGED FOR NEW VERSION
        self.name = 'flowline_periodic'


    def init_coordinate_system(self):
        """Setup coordinate system and transformations"""
        # Base slope angle
        self.base_slope = -0.1  # Negative for downhill slope
        self.alpha = np.arctan(self.base_slope)  # Angle in radians


    def init_domain_parameters(self):
        """Initialize domain and geometry parameters"""
        # Domain extents and resolution
        self.x_params = {
            'start': 0.001,
            'end': 77.76,  # Exactly 8 wavelengths
            'step': 0.2
        }
        
        # Mesh resolution
        self.mesh_hmax = self.x_params['step'] * 4 #/2 # km

        # Ice parameters
        self.ice_params = {
            'mean_thickness': 2.7,  # km
        }
        
        # Bedrock parameters
        self.bedrock_params = {
            'initial_elevation': 1.0,
            'slope': self.base_slope,
            'lambda': 9.72,  # 3.6 times Z
            'amplitude': self.ice_params['mean_thickness']/50
        }

    def init_physical_parameters(self):
        """Initialize physical parameters"""
        # Ice physics
        self.rheology_n = 4.0
        # 917 kg/m³ converted to kg/km³
        self.rho_ice = 917.0 * (10**9)  # 9.17e11 kg/km³
        
        # Constants
        self.yts = 31556926.0  # seconds per year
        self.g = 9.80665e-3 * self.yts**2  # gravity acceleration km/year^2
        
        # Rheology parameters
        self.A = 1e-16  # flow parameter
        self.B = (self.A / self.yts)**(-1/self.rheology_n)  # hardness parameter

        # Thermal parameters
        self.ice_temperature = 253.0  # -20°C
        self.surface_temperature = 263.15  # -10°C
        self.basal_temperature = 273.15  # Pressure melting point
        self.thermal_conductivity = 2100  # W/(km·K)
        self.heat_capacity = 2009  # J/(kg·K)


    def init_solver_parameters(self):
        """Initialize solver and numerical parameters"""
        # Time settings
        self.time_settings = {
            'time_step': 1,        # years
            'start_time': 0,
            'final_time': 600,     # years
            'output_frequency': 10 # save every n years
        }
        
        # Solver settings
        self.solver_settings = {
            'flowequation_fe_FS': 'P1P1',
            'convergence': 'relative',
            'restol': 1e-3,
            'reltol': 1e-3,
            'abstol': 1e-3,
            'maxiter': 300,
            'min_iterations': 2,
            'stabilization': 1,
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
        self.num_processors = 1
        self.fs_reconditioning = 1.0


    @property
    def omega(self):
        """Calculate angular frequency from wavelength"""
        return 2 * np.pi / self.bedrock_params['lambda']
     
   
    def get_bedrock_elevation(self, x_prime):
        """Calculate bedrock elevation directly in slope-parallel coordinates
        
        Args:
            x_prime: x-coordinate(s) in slope-parallel system
                    
        Returns:
            z_prime: elevation in slope-parallel coordinates
        """
        # Define where the non-undulated sections begin and end
        straight_section_1 = self.x_params['start'] + 5.0  # First 5km is flat
        straight_section_2 = self.x_params['end'] - 5.0    # Last 5km is flat
        
        # Calculate the standard undulated bedrock elevation for all points
        undulated_elevation = (self.bedrock_params['initial_elevation'] + 
                              self.bedrock_params['amplitude'] * np.cos(self.omega * x_prime))
        
        # Calculate fixed elevations for the flat sections
        start_transition_elevation = (self.bedrock_params['initial_elevation'] + 
                                     self.bedrock_params['amplitude'] * np.cos(self.omega * straight_section_1))
        
        end_transition_elevation = (self.bedrock_params['initial_elevation'] + 
                                   self.bedrock_params['amplitude'] * np.cos(self.omega * straight_section_2))
        
        # Create a copy of the input array to avoid modifying the original
        result = np.copy(undulated_elevation)
        
        # Apply the flat section at the beginning
        result = np.where(x_prime < straight_section_1, start_transition_elevation, result)
        
        # Apply the flat section at the end
        result = np.where(x_prime >= straight_section_2, end_transition_elevation, result)
        
        return result


    def setup_periodic_boundary(self, md):
        """Configure periodic boundary conditions between inflow and terminus"""
        
        # Find vertices at the inflow (minimum x) and terminus (maximum x)
        inflow_vertices = np.where(np.isclose(md.mesh.x, self.x_params['start']))[0]
        terminus_vertices = np.where(np.isclose(md.mesh.x, self.x_params['end']))[0]
        
        # Ensure we have matching numbers of vertices to pair
        min_vertices = min(len(inflow_vertices), len(terminus_vertices))
        
        if min_vertices == 0:
            print("Warning: Could not find boundary vertices for periodic conditions")
            return md
        
        inflow_sorted = inflow_vertices
        terminus_sorted = terminus_vertices
        
        # Pair the vertices (add 1 because ISSM uses 1-based indexing)
        paired_vertices = np.vstack((inflow_sorted[:min_vertices] + 1, 
                                    terminus_sorted[:min_vertices] + 1)).T
        
        # Set the vertex pairing for both stress balance and mass transport(?)
        md.stressbalance.vertex_pairing = paired_vertices
        md.masstransport.vertex_pairing = paired_vertices
        
        print(f"Set up periodic boundary conditions with {min_vertices} paired vertices")
        
        return md


    def setup_model_settings(self, md):
        """Configure core model settings"""

        print(f"Number of mesh elements: {md.mesh.numberofelements}")

        # Set up masks
        md.mask.ice_levelset = -np.ones((md.mesh.numberofvertices))
        md.mask.ice_levelset[np.where(md.mesh.vertexflags(2))] = 0.
        md.mask.ocean_levelset = np.ones((md.mesh.numberofvertices))

        # Apply other settings
        md = self.Budd_sliding(md)
        md = self.setup_thermal_model(md)
        md = self.setup_stress_balance(md)
        md = self.setup_mass_balance(md)
        
        # Set up periodic boundary conditions
        md = self.setup_periodic_boundary(md)
        
        return md


    def setup_stress_balance(self, md):
        """Configure stress balance settings"""
        # Initialize velocity components
        md.initialization.vx = np.zeros((md.mesh.numberofvertices))
        md.initialization.vy = np.zeros((md.mesh.numberofvertices))

        # Material properties
        md.materials.rheology_n = self.rheology_n * np.ones((md.mesh.numberofelements))

        # initialize pressure
        md.initialization.pressure = np.zeros((md.mesh.numberofvertices))
        
        # Initialize stress balance arrays
        md.stressbalance.referential = np.nan * np.ones((md.mesh.numberofvertices, 6))
        md.stressbalance.loadingforce = np.zeros((md.mesh.numberofvertices, 3))
        md.stressbalance.spcvx = np.nan * np.ones((md.mesh.numberofvertices,))
        md.stressbalance.spcvy = np.nan * np.ones((md.mesh.numberofvertices,))

        # # - Fixed upstream boundary
        # flag4 = np.where(md.mesh.vertexflags(4))
        # md.stressbalance.spcvx[flag4] = 0.
        # md.stressbalance.spcvy[flag4] = 0.

        # Set up solver parameters
        md.stressbalance.abstol = self.solver_settings['abstol']
        md.stressbalance.reltol = self.solver_settings['reltol']
        md.stressbalance.convergence = self.solver_settings['convergence']
        md.stressbalance.maxiter = self.solver_settings['maxiter']
        md.flowequation.augmented_lagrangian_r = self.solver_settings['augmented_lagrangian_r']
        md.stressbalance.FSreconditioning = self.fs_reconditioning
        md.settings.solver_residue_threshold = 1e-4  # More relaxed threshold
        md.flowequation.fe_FS = 'P1P1'
        md = setflowequation(md, 'FS', 'all')
        
        return md


    def setup_mass_balance(self, md):
        """Configure mass balance settings"""

        # Initialize surface mass balance
        md.smb.mass_balance = np.zeros((md.mesh.numberofvertices)) # this was missing a set of parentheses
        
        # Set up mass transport configuration
        md.masstransport.spcthickness = np.nan * np.ones((md.mesh.numberofvertices,))
        md.masstransport.min_thickness = self.mass_transport_params['min_thickness']
        md.masstransport.hydrostatic_adjustment = self.mass_transport_params['hydrostatic_adjustment']
        md.masstransport.isfreesurface = 1
        
        md.basalforcings.groundedice_melting_rate = np.zeros((md.mesh.numberofvertices))
        md.basalforcings.floatingice_melting_rate = np.zeros((md.mesh.numberofvertices))
        md.basalforcings.geothermalflux = np.zeros((md.mesh.numberofvertices))

        return md


    def setup_thermal_model(self, md):
        """Configure thermal model settings"""
        # Basic thermal fields
        md.thermal.spctemperature = np.nan * np.ones((md.mesh.numberofvertices))
        md.initialization.temperature = self.ice_temperature * np.ones(md.mesh.numberofvertices)
        md.thermal.temperature = self.ice_temperature * np.ones(md.mesh.numberofvertices)
        
        # Material properties - use constant B instead of temperature-dependent rheology
        md.materials.rheology_B = self.B * np.ones(md.mesh.numberofvertices)
        md.thermal.conductivity = self.thermal_conductivity * np.ones(md.mesh.numberofvertices)
        md.thermal.capacity = self.heat_capacity * np.ones(md.mesh.numberofvertices)
        
        # Heat sources
        md.thermal.internal_heating = np.zeros(md.mesh.numberofvertices)
        md.thermal.friction_heating = np.zeros(md.mesh.numberofvertices)
        
        return md



    def Budd_sliding(self, md):
        """Budd (1970) Sliding Law with updated shear stress"""
        # Set up friction
        md.friction.coefficient = np.ones((md.mesh.numberofvertices))
        md.friction.p = np.ones((md.mesh.numberofelements))
        md.friction.q = np.ones((md.mesh.numberofelements))

        # Get basal nodes
        basal_nodes = np.where(md.mesh.vertexflags(1))[0]
        
        # Calculate sliding coefficient using physics-based formula
        sliding_coefficient = self.calculate_sliding_coefficient(md)

        # Debug output
        print(f"Friction range: {sliding_coefficient[basal_nodes].min():.3f} to {sliding_coefficient[basal_nodes].max():.3f}")

        # Apply to model at basal nodes only
        md.friction.coefficient[basal_nodes] = sliding_coefficient[basal_nodes]

        return md


    def calculate_sliding_coefficient(self, md, is_initial=True):
        """Calculate sliding coefficient based on the physical shear stress model
        
        Args:
            md: Model object
            is_initial: Whether this is the initial calculation or an update
        """
        # Get required parameters
        omega = self.omega
        Beta_1 = self.bedrock_params['amplitude']
        
        # Get basal nodes
        basal_nodes = np.where(md.mesh.vertexflags(1))[0]
        x_coords = md.mesh.x[basal_nodes]
        
        # Get velocity data for diagnostic purposes only, not for coefficient calculation
        velocity_source = "N/A"
        V = 0.0
        
        if not is_initial:
            # Try to find velocity data in different possible locations
            vx_basal = None
            
            # Check common locations for velocity data
            if hasattr(md.results, 'StressbalanceSolution') and hasattr(md.results.StressbalanceSolution, 'Vx'):
                vx_basal = md.results.StressbalanceSolution.Vx[basal_nodes]
                velocity_source = "StressbalanceSolution"
            elif hasattr(md.results, 'TransientSolution') and hasattr(md.results.TransientSolution, 'Vx'):
                vx_basal = md.results.TransientSolution.Vx[basal_nodes]
                velocity_source = "TransientSolution"
            elif hasattr(md.results, 'Vx'):
                vx_basal = md.results.Vx[basal_nodes]
                velocity_source = "results.Vx"
            else:
                # Fallback to initialization
                vx_basal = md.initialization.vx[basal_nodes]
                velocity_source = "initialization"
            
            # Calculate average velocity for diagnostic output only
            V = np.mean(np.abs(vx_basal))
            V = max(V, 0.01)  # Ensure minimum velocity for stability
        
        # Calculate viscosity using constant B instead of temperature-dependent value
        eta = 0.5 * self.B
        
        # Calculate coefficient terms
        omega_x = omega * x_coords
        cos_term = np.cos(omega_x)

        # Calculate the base coefficient
        base_coefficient = 2 * eta * omega * Beta_1 * cos_term

        # Calculate the offset - the minimum possible value is -2*eta*omega*Beta_1
        # Add a small buffer (1.05 factor) for numerical safety
        offset = 2.1 * eta * omega * Beta_1

        # Apply the offset to ensure non-negative values
        sliding_coefficient = base_coefficient + offset
        
        # For updates, take absolute value and apply scaling
        scaling = 0.2 if not is_initial else 1.0
        if not is_initial:
            sliding_coefficient = np.abs(sliding_coefficient * scaling)
        
        # Set friction to zero in flat regions (using domain boundaries from get_bedrock_elevation)
        straight_section_1 = self.x_params['start'] + 5.0  # First 5km is flat
        straight_section_2 = self.x_params['end'] - 5.0    # Last 5km is flat
        
        # Create mask for flat regions
        flat_mask = (x_coords < straight_section_1) | (x_coords >= straight_section_2)
        
        # Set friction to zero or a very small value in flat regions
        sliding_coefficient[flat_mask] = 3000  # Use minimum value in pattern
        
        # Print diagnostic info if needed
        if not is_initial:
            print_step = True
            if hasattr(md, 'timestepping') and md.timestepping.step_counter > 0:
                print_step = (md.timestepping.step_counter % 5 == 0)
            
            if print_step:
                current_time = 0
                if hasattr(md, 'timestepping'):
                    current_time = md.timestepping.start_time
                    if md.timestepping.step_counter > 0:
                        current_time += md.timestepping.time_step
                        
                print(f"\nFriction at t={current_time:.1f} yr (velocity from {velocity_source}):")
                print(f"  Velocity: {V:.4f} km/yr")
                print(f"  Friction range: {sliding_coefficient.min():.3f} to {sliding_coefficient.max():.3f}")
                print(f"  Flat regions with near-zero friction: {np.sum(flat_mask)} nodes")
        
        return sliding_coefficient


    def update_sliding_coefficient(self, md):
        """Update sliding coefficient based on current velocity solution"""
        # Get basal nodes
        basal_nodes = np.where(md.mesh.vertexflags(1))[0]
        
        # Calculate sliding coefficient using physics-based formula
        sliding_coefficient = self.calculate_sliding_coefficient(md, is_initial=False)
        
        # Apply to model at basal nodes only
        md.friction.coefficient[basal_nodes] = sliding_coefficient
        
        # Create plots at significant steps
        is_first_step = not hasattr(md, 'timestepping') or md.timestepping.step_counter == 0
        is_final_step = hasattr(md, 'timestepping') and \
                       (md.timestepping.start_time + md.timestepping.time_step >= md.timestepping.final_time)
        
        if is_first_step or is_final_step:
            self.plot_friction_coefficient(md, is_first_step=is_first_step, is_final_step=is_final_step)
        
        return md


    def plot_friction_coefficient(self, md, is_first_step=False, is_final_step=False, velocity=None):
        """Plot friction coefficient"""
        basal_nodes = np.where(md.mesh.vertexflags(1))[0]
        x_coords = md.mesh.x[basal_nodes]
        friction_vals = md.friction.coefficient[basal_nodes]
        
        # Determine plot type and filename
        if is_first_step:
            title = "Initial Friction Coefficient"
            filename = "initial_friction.png"
        elif is_final_step:
            title = f"Final Friction Coefficient (Avg Velocity: {velocity:.4f} km/yr)" if velocity else "Final Friction Coefficient"
            filename = "final_friction.png"
        else:
            return md  # No plot for intermediate steps
        
        # Sort by x-coordinate for a clean plot
        sort_idx = np.argsort(x_coords)

        
        # Scatter plot
        if is_final_step:
            plt.figure(figsize=(12, 6))
            sc = plt.scatter(x_coords, np.zeros_like(x_coords), 
                      s=40, c=friction_vals, cmap='viridis')
            plt.colorbar(sc, label="Friction coefficient")
            plt.title(title)
            plt.xlabel("Distance along flowline (km)")
            plt.ylim([-0.05, 0.05])
            plt.savefig(f"{filename.replace('.png', '_scatter.png')}")
            plt.close()
        
        return md

# Create default configuration instance
config = ModelConfig()

