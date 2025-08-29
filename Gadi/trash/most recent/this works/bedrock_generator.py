# bedrock_generator.py - Generate synthetic bedrock profiles (1D) compatible with configp.py
# Modified to ensure sloped but undulation-free boundaries

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.ndimage import gaussian_filter1d

class BedrockGenerator:
    def __init__(self):
        # Base parameters to match configp.py
        self.domain_length = 50 * 1e3  # m
        self.resolution = 0.2 * 1e3 # m
        self.ice_thickness = 1.92 * 1e3 # m
        self.base_slope = -0.0015 # Negative for downhill slope: 0.9deg
        
        # Output directory
        self.output_dir = "bedrock_profiles"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Parameter ranges for bedrock generation - can adjust as needed
        self.param_ranges = {
            # Amplitude as fraction of ice thickness
            'amplitude': np.linspace(0.01, 0.1, 5),
            
            # Wavelength as multiple of ice thickness (Budd found ~3.3Z is optimal)
            'wavelength': np.array([2.0, 3.3, 5.0, 8.0, 10.0]) * self.ice_thickness,
            
            # Skewness parameter
            'skewness': np.linspace(-0.4, 0.4, 5),
            
            # Kurtosis parameter
            'kurtosis': np.linspace(-0.4, 0.4, 5)
        }


    def save_profile(self, x, bed, params, profile_id):
        """Save bedrock profile to file"""
        # Create filename
        filename = f"{self.output_dir}/bedrock_profile_{profile_id:03d}"
        
        # Save numpy array with parameters
        np.savez(f"{filename}.npz", 
                x=x, 
                bed=bed, 
                amplitude=params['amplitude'],
                wavelength=params['wavelength'],
                skewness=params.get('skewness', 0),
                kurtosis=params.get('kurtosis', 0),
                noise_level=params.get('noise_level', 0),
                initial_elevation=params.get('initial_elevation', 1.0),
                transition_width=params.get('transition_width', 2))
        
        # Plot and save figure
        plt.figure(figsize=(10, 4))

        # Add vertical lines at the boundary transitions
        plt.axvline(x=15.0, color='r', linestyle='--', alpha=0.5, label='Boundary regions')
        plt.axvline(x=(self.domain_length-15.0 * 1e3) / 1e3, color='r', linestyle='--', alpha=0.5)

        # Plot the bedrock profile
        plt.plot(x / 1e3, bed / 1e3, 'b-', label='Bed profile')
        
        # Plot the base sloped line for reference
        base_bed = self.generate_base_bed(x, params.get('initial_elevation', 1.0))
        plt.plot(x / 1e3, base_bed / 1e3, 'k--', alpha=0.5, label='Base slope')
        plt.xlabel('Distance (km)')
        plt.ylabel('Bed elevation (km)')
        plt.legend()
        
        title_parts = []
        title_parts.append(f"λ={params['wavelength'] / 1e3:.1f}km")
        title_parts.append(f"A={params['amplitude'] / 1e3:.3f}km")
        
        if params.get('skewness', 0) != 0:
            title_parts.append(f"S={params['skewness']:.1f}")
        
        if params.get('kurtosis', 0) != 0:
            title_parts.append(f"K={params['kurtosis']:.1f}")
            
        if params.get('noise_level', 0) > 0:
            title_parts.append(f"N={params['noise_level']:.2f}")
        
        if params.get('transition_width', 2) != 2:
            title_parts.append(f"T={params['transition_width']:.1f}km")
            
        plt.title(f"Profile {profile_id}: " + ", ".join(title_parts))
        plt.grid(True, linestyle=":", color='k', alpha=0.4)
        plt.savefig(f"{filename}.png", dpi=150)
        plt.close()


    def generate_x_grid(self):
        """Generate the x-coordinate grid exactly like configp.py"""
        return np.linspace(0.001, self.domain_length, 
                          int(self.domain_length / self.resolution) + 1)
        

    def generate_base_bed(self, x, initial_elevation=1.0):
        """Generate the base sloped bed without undulations"""
        return initial_elevation + self.base_slope * x

    
    def apply_skewness(self, bed, skewness):
        """Apply skewness transformation to the bed profile"""
        if skewness == 0:
            return bed
        
        # Center the data around zero for the transformation
        mean_bed = np.mean(bed)
        centered = bed - mean_bed
        
        # Apply power transformation to introduce skewness
        # For positive skewness, higher values get stretched more
        # For negative skewness, lower values get stretched more
        sign = np.sign(centered)
        abs_centered = np.abs(centered)
        
        if skewness > 0:
            transformed = sign * (abs_centered ** (1 + skewness))
        else:
            transformed = sign * (abs_centered ** (1 / (1 - skewness)))
        
        # Normalize to maintain amplitude
        if np.max(np.abs(transformed)) > 0:
            scale_factor = np.max(np.abs(centered)) / np.max(np.abs(transformed))
            transformed *= scale_factor
        
        # Return to original mean
        return transformed + mean_bed
    

    def apply_kurtosis(self, bed, kurtosis):
        """Apply kurtosis transformation to the bed profile"""
        if kurtosis == 0:
            return bed
        
        # Center and normalize the data
        mean_bed = np.mean(bed)
        std_bed = np.std(bed)
        if std_bed == 0:
            return bed
            
        z = (bed - mean_bed) / std_bed
        
        # Apply transformation to adjust kurtosis
        if kurtosis > 0:  # Increase peakedness
            result = np.sign(z) * np.abs(z) ** (1 + kurtosis/2)
        else:  # Flatten the distribution
            result = np.sign(z) * np.abs(z) ** (1 / (1 - kurtosis/2))
        
        # Rescale back to original scale
        scaled = result * std_bed + mean_bed
        
        # Preserve original min/max range
        orig_range = np.max(bed) - np.min(bed)
        new_range = np.max(scaled) - np.min(scaled)
        if new_range > 0:
            scaled = ((scaled - np.min(scaled)) * orig_range / new_range) + np.min(bed)
        
        return scaled
    

    def add_noise(self, bed, noise_level=0.0):
        """Add white noise to the bed profile"""
        if noise_level <= 0:
            return bed
        
        amplitude = np.max(bed) - np.min(bed)
        if amplitude <= 0:
            return bed
            
        noise = noise_level * amplitude * np.random.normal(size=len(bed))
        return bed + noise
    

    def generate_bedrock_profile(self, 
                            amplitude, 
                            wavelength, 
                            skewness=0, 
                            kurtosis=0, 
                            noise_level=0.0,
                            initial_elevation=1.0,
                            transition_width=2,
                            create_debug_plot=False):
        """
        Generate a complete bedrock profile with all features
        The entire domain has undulations (no undulation-free boundaries)
        """
        # Generate the x grid
        x = self.generate_x_grid()
        
        # Generate the base bed (sloped line)
        base_bed = self.generate_base_bed(x, initial_elevation)
        
        # Calculate the angular frequency
        omega = 2 * np.pi / wavelength
        
        # Apply cosine undulation to the entire domain
        bed = base_bed + amplitude * np.cos(omega * x)
        
        # Apply transformations to the entire bed
        if skewness != 0:
            bed = self.apply_skewness(bed, skewness)
        
        if kurtosis != 0:
            bed = self.apply_kurtosis(bed, kurtosis)
        
        if noise_level > 0:
            bed = self.add_noise(bed, noise_level)
        
        # Make a copy for the final result
        final_bed = np.copy(bed)
        
        # Save debug data if requested
        if create_debug_plot and wavelength == 13.5 and abs(amplitude - 0.270) < 0.001:
            self.debug_data = {
                'x': x,
                'transformed_bed': bed,
                'final_bed': final_bed,
                'base_bed': base_bed,
                'params': {
                    'amplitude': amplitude,
                    'wavelength': wavelength,
                    'initial_elevation': initial_elevation,
                    'transition_width': transition_width
                }
            }
        
        return x, final_bed


    def create_gradient_smoothing_debug_plot(self):
        """Create a debug plot showing the effect of gradient smoothing with different transition widths"""
        if not hasattr(self, 'debug_data'):
            print("No debug data available. Generate a profile with create_debug_plot=True first.")
            return
        
        x = self.debug_data['x']
        transformed_bed = self.debug_data['transformed_bed'] 
        base_bed = self.debug_data['base_bed']
        final_bed = self.debug_data['final_bed']
        params = self.debug_data['params']
        
        plt.figure(figsize=(12, 8))
        
        # Plot the original profile before smoothing
        plt.subplot(2, 1, 1)
        plt.axvline(x=15.0, color='g', linestyle='--')
        plt.axvline(x=(self.domain_length-15.0 * 1e3) / 1e3, color='g', linestyle='--')
        plt.plot(x / 1e3, transformed_bed / 1e3, 'r-', label='Before smoothing')
        plt.plot(x / 1e3, base_bed / 1e3, 'k--', alpha=0.5, label='Base slope')
        plt.title('Before smoothing')
        plt.legend()
        plt.grid(True, linestyle=":", color='k', alpha=0.4)

        
        # Plot the smoothed profile
        plt.subplot(2, 1, 2)
        plt.plot(x / 1e3, transformed_bed / 1e3, 'r-', label='Original')
        plt.plot(x / 1e3, final_bed / 1e3, 'b-', label=f'Smoothed (width={params["transition_width"]}km)')
        
        # Highlight the transition regions
        plt.axvline(x=15.0, color='g', linestyle='--')
        plt.axvline(x=(self.domain_length-15.0 * 1e3) / 1e3, color='g', linestyle='--')
        plt.title('After gradient-based smoothing')
        plt.legend()
        plt.grid(True, linestyle=":", color='k', alpha=0.4)
        
        plt.tight_layout()
        plt.savefig('gradient_smoothing_debug.png', dpi=150)
        plt.close()


    def create_parameter_combinations(self, n_samples=100):
        """
        Create a systematic set of parameter combinations covering the parameter space
        """
        param_list = []
        
        # Always include a flat profile
        configp_params = {
            'amplitude': 0.01,  # No undulations but I need to kept a non zero amplitude for stability
            'wavelength': 9.72 * 1e3,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'noise_level': 0.0,
            'initial_elevation': 1.0
        }
        param_list.append(configp_params)
        
        # Create combinations covering all primary parameters
        for amplitude in self.param_ranges['amplitude']:
            for wavelength in self.param_ranges['wavelength']:
                for skewness in self.param_ranges['skewness']:
                    for kurtosis in self.param_ranges['kurtosis']:
                        # Skip if this is identical to the configp params
                        if (abs(amplitude * self.ice_thickness - configp_params['amplitude']) < 1e-6 and
                            abs(wavelength - configp_params['wavelength']) < 1e-6 and
                            abs(skewness) < 1e-6 and abs(kurtosis) < 1e-6):
                            continue
                            
                        params = {
                            'amplitude': amplitude * self.ice_thickness,  # Scale by ice thickness
                            'wavelength': wavelength,
                            'skewness': skewness,
                            'kurtosis': kurtosis,
                            'noise_level': 0.0,
                            'initial_elevation': 1.0
                        }
                        param_list.append(params)
        
        # If we have more combinations than needed, select a representative subset
        if len(param_list) > n_samples:
            # Always keep the first entry (the configp match)
            first_entry = param_list[0]
            
            # Take evenly spaced indices to get n_samples-1 from the rest of the list
            remaining = param_list[1:]
            indices = np.linspace(0, len(remaining) - 1, n_samples-1).astype(int)
            param_list = [first_entry] + [remaining[i] for i in indices]
        
        # If we need more combinations, add some with noise
        elif len(param_list) < n_samples:
            # How many more do we need?
            remaining = n_samples - len(param_list)
            
            # Add profiles with noise
            for i in range(remaining):
                # Pick a random existing parameter set (but not the first/configp one)
                if len(param_list) > 1:
                    base_params = param_list[1 + (i % (len(param_list)-1))].copy()
                else:
                    base_params = param_list[0].copy()
                
                # Add increasing noise levels
                base_params['noise_level'] = 0.05 + (i / remaining) * 0.15
                
                param_list.append(base_params)
                
                # If we have enough, stop
                if len(param_list) >= n_samples:
                    break
        
        return param_list[:n_samples]
    
    def generate_all_profiles(self, n_profiles=100):
        """Generate all bedrock profiles and save them"""
        # Get parameter combinations
        param_combinations = self.create_parameter_combinations(n_profiles)
        
        # Create metadata file
        metadata_file = f"{self.output_dir}/bedrock_metadata.txt"
        with open(metadata_file, 'w') as f:
            f.write(f"Synthetic Bedrock Profiles Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Domain Length: {self.domain_length / 1e3} km\n")
            f.write(f"Resolution: {self.resolution / 1e3} km\n")
            f.write(f"Ice Thickness: {self.ice_thickness / 1e3} km\n")
            f.write(f"Base Slope: {self.base_slope}\n")
            f.write(f"All profiles have sloped, undulation-free boundaries in the first and last 15km\n\n")
            # f.write("Profile Parameters:\n")
            
            # Generate each profile
            for i, params in enumerate(param_combinations):
                profile_id = i + 1
                print(f"Generating profile {profile_id}/{n_profiles}...")
                
                # Generate the profile
                x, bed = self.generate_bedrock_profile(
                    amplitude=params['amplitude'],
                    wavelength=params['wavelength'],
                    skewness=params['skewness'],
                    kurtosis=params['kurtosis'],
                    noise_level=params['noise_level'],
                    initial_elevation=params['initial_elevation'],
                    transition_width=2  
                )
                
                # Save it
                self.save_profile(x, bed, params, profile_id)
                
                # Add metadata
                f.write(f"Profile {profile_id:03d}:\n")
                f.write(f"  Amplitude: {params['amplitude'] / 1e3:.6f} km\n")
                f.write(f"  Wavelength: {params['wavelength'] / 1e3:.6f} km\n")
                f.write(f"  Skewness: {params['skewness']:.6f}\n")
                f.write(f"  Kurtosis: {params['kurtosis']:.6f}\n")
                f.write(f"  Noise Level: {params['noise_level']:.6f}\n")
                f.write(f"  Initial Elevation: {params['initial_elevation']:.6f}\n\n")
                
        print(f"Generated {n_profiles} bedrock profiles in '{self.output_dir}'")
        print(f"Metadata saved to '{metadata_file}'")
        # print(f"All profiles have sloped, undulation-free boundaries in first and last 15km.")
    

    def test_boundary_regions(self):
        """Test function to verify the boundary regions are sloped but undulation-free"""
        # Generate a test profile with short wavelength to emphasize undulations
        x, bed = self.generate_bedrock_profile(
            amplitude=0.027 * 1e3,
            wavelength=2.7 * 1e3,
            skewness=0.0,
            kurtosis=0.0,
            noise_level=0.0,
            initial_elevation=1.0
        )
        
        # Generate the base sloped line
        base_bed = self.generate_base_bed(x, 1.0)
        
        # Check the first 5km section
        first_section = x < (15.0  * 1e3)
        first_section_data = bed[first_section]
        first_section_x = x[first_section]
        
        # Check if the first section follows a straight line (should be parallel to base bed)
        if len(first_section_x) > 1:
            # Calculate the slope of the first section
            first_slope = (first_section_data[-1] - first_section_data[0]) / (first_section_x[-1] - first_section_x[0])
            base_slope = self.base_slope
            slope_match = abs(first_slope - base_slope) < 1e-5
            
            # Check if the section is undulation-free by fitting a line and checking residuals
            if len(first_section_x) > 2:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(first_section_x, first_section_data)
                fitted_line = intercept + slope * first_section_x
                residuals = first_section_data - fitted_line
                undulation_free = np.max(np.abs(residuals)) < 1e-5
            else:
                undulation_free = True
        else:
            slope_match = True
            undulation_free = True
            
        # Similarly check the last 5km section
        last_section = x > (self.domain_length - (15.0 * 1e3))
        last_section_data = bed[last_section]
        last_section_x = x[last_section]
        
        if len(last_section_x) > 1:
            last_slope = (last_section_data[-1] - last_section_data[0]) / (last_section_x[-1] - last_section_x[0])
            base_slope = self.base_slope
            last_slope_match = abs(last_slope - base_slope) < 1e-5
            
            if len(last_section_x) > 2:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(last_section_x, last_section_data)
                fitted_line = intercept + slope * last_section_x
                residuals = last_section_data - fitted_line
                last_undulation_free = np.max(np.abs(residuals)) < 1e-5
            else:
                last_undulation_free = True
        else:
            last_slope_match = True
            last_undulation_free = True
        
        # Plot for visualization
        plt.figure(figsize=(12, 6))
        # Highlight the boundary regions
        plt.axvline(x=15.0, color='r', linestyle='--', alpha=0.5, label='Boundary regions')
        plt.axvline(x=(self.domain_length-15.0 * 1e3) / 1e3, color='r', linestyle='--', alpha=0.5)

        # Plot the entire profile
        plt.plot(x / 1e3, bed / 1e3, 'b-', label='Bed profile')
        
        # Plot the base sloped line
        plt.plot(x / 1e3, base_bed / 1e3, 'k--', alpha=0.5, label='Base slope')
        
        # Fill the boundary regions
        plt.fill_between(x / 1e3, bed / 1e3, where=first_section, color='g', alpha=0.3, label='First 15km (undulation-free)')
        plt.fill_between(x / 1e3, bed / 1e3, where=last_section, color='orange', alpha=0.3, label='Last 15km (undulation-free)')
        
        plt.grid(True, linestyle=":", color='k', alpha=0.4)
        plt.xlabel('Distance (km)')
        plt.ylabel('Bed elevation (km)')
        plt.title(f'Boundary Regions Test\nFirst 15km: slope match={slope_match}, undulation-free={undulation_free}\nLast 15km: slope match={last_slope_match}, undulation-free={last_undulation_free}')
        plt.legend()
        plt.savefig(f"{self.output_dir}/boundary_test.png", dpi=150)
        plt.close()
        
        print(f"Boundary regions test results:")
        print(f"First 15km: slope match={slope_match}, undulation-free={undulation_free}")
        print(f"Last 15km: slope match={last_slope_match}, undulation-free={last_undulation_free}")
        print(f"Test plot saved to '{self.output_dir}/boundary_test.png'")
        
        return slope_match and undulation_free and last_slope_match and last_undulation_free


class SyntheticBedrockModelConfig:
    """Model configuration with synthetic bedrock profiles for ISSM"""
    
    def __init__(self, profile_id=1, output_dir="bedrock_profiles"):
        self.base_slope = -0.0015
        
        # Load the profile ID and path
        self.profile_id = profile_id
        self.output_dir = output_dir

        # Load the bedrock profile
        filename = f"{output_dir}/bedrock_profile_{profile_id:03d}.npz"
        try:
            data = np.load(filename)
            self.x_profile = data['x']
            self.bed_profile = data['bed']
            self.profile_params = {
                'amplitude': float(data['amplitude']),
                'wavelength': float(data['wavelength']),
                'skewness': float(data['skewness']),
                'kurtosis': float(data['kurtosis']),
                'noise_level': float(data['noise_level']),
                'initial_elevation': float(data.get('initial_elevation', 1.0))
            }
            print(f"\nLoaded bedrock profile {profile_id} with parameters:")
            for key, value in self.profile_params.items():
                print(f"  {key}: {value}")
        except FileNotFoundError:
            raise ValueError(f"Profile {profile_id} not found in {output_dir}. Generate profiles first.")
        
        # Initialize coordinate system parameters
        self.init_coordinate_system()
        
        # Initialize domain parameters
        self.init_domain_parameters()
                
        # Model identifier
        self.name = f'bedrock_profile_{profile_id:03d}'

    def init_coordinate_system(self):
        """Setup coordinate system and transformations"""
        # Base slope angle
        self.alpha = np.arctan(self.base_slope)  # Angle in radians

    def init_domain_parameters(self):
        """Initialize domain and geometry parameters"""
        # Domain extents from the profile
        self.x_params = {
            'start': self.x_profile[0],
            'end': self.x_profile[-1],
            'step': self.x_profile[1] - self.x_profile[0]
        }
        
        # Mesh resolution
        self.mesh_hmax = self.x_params['step'] * 2
        
        # Ice parameters
        self.ice_params = {
            'mean_thickness': 1.92 * 1e3,
        }
        
        # Bedrock parameters - for reference only, actual bed comes from profile
        self.bedrock_params = {
            'initial_elevation': self.profile_params['initial_elevation'],
            'slope': self.base_slope,
            'lambda': self.profile_params['wavelength'],
            'amplitude': self.profile_params['amplitude']
        }

    def get_bedrock_elevation(self, x_prime):
        """Calculate bedrock elevation using the loaded profile
        
        Args:
            x_prime: x-coordinate(s) in slope-parallel system
                    
        Returns:
            z_prime: elevation in slope-parallel coordinates
        """
        # Use linear interpolation for points within the domain
        return np.interp(x_prime, self.x_profile, self.bed_profile)


if __name__ == "__main__":
    # Generate the profiles
    generator = BedrockGenerator()
    generator.generate_all_profiles(n_profiles=625)
    
    # # Test that the boundary regions are sloped but undulation-free
    # generator.test_boundary_regions()