# bedrock_generator.py - Generate synthetic bedrock profiles (1D)
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from bedrock_settings import (
    DOMAIN_LENGTH,
    RESOLUTION,
    ICE_THICKNESS,
    BASE_SLOPE,
    PARAM_RANGES,
)

class BedrockGenerator:
    def __init__(self):
        self.domain_length = DOMAIN_LENGTH
        self.resolution = RESOLUTION
        self.ice_thickness = ICE_THICKNESS
        self.base_slope = BASE_SLOPE
        self.param_ranges = PARAM_RANGES

        # Output directory
        self.output_dir = "bedrock_profiles"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


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
                initial_elevation=params.get('initial_elevation', 1.0)) # 1 meter
        
        # Plot and save figure
        plt.figure(figsize=(12, 4))

        # Add vertical lines for the region boundaries
        pre_extension_end = 25e3  # 25km pre-extension
        undulated_end = pre_extension_end + self.domain_length  # End of undulated region
        
        plt.axvline(x=pre_extension_end / 1e3, color='r', linestyle='--', alpha=0.7, label='Undulated start')
        plt.axvline(x=undulated_end / 1e3, color='g', linestyle='--', alpha=0.7, label='Undulated end')

        # Plot the bedrock profile
        plt.plot(x / 1e3, bed / 1e3, 'b-', label='Bed profile')
        
        # Plot the base sloped line for reference
        base_bed = self.generate_base_bed(x, params.get('initial_elevation', 1.0))
        plt.plot(x / 1e3, base_bed / 1e3, 'k--', alpha=0.5, label='Base slope')
        plt.xlabel('Distance (km)')
        plt.ylabel('Bed elevation (km)')
        plt.legend()
        
        title_parts = []
        # MODIFICATION: Added check for profile 000 title
        if profile_id == 0:
            title_parts.append("Gaussian Bump")
        title_parts.append(f"λ={params['wavelength'] / 1e3:.1f}km")
        title_parts.append(f"A={params['amplitude'] / 1e3:.3f}km")
        
        if params.get('skewness', 0) != 0:
            title_parts.append(f"S={params['skewness']:.1f}")
        
        if params.get('kurtosis', 0) != 0:
            title_parts.append(f"K={params['kurtosis']:.1f}")
            
        if params.get('noise_level', 0) > 0:
            title_parts.append(f"N={params['noise_level']:.2f}")
            
        plt.title(f"Profile {profile_id:03d}: " + ", ".join(title_parts))
        plt.grid(True, linestyle=":", color='k', alpha=0.4)
        plt.savefig(f"{filename}.png", dpi=150)
        plt.close()


    def generate_x_grid(self, pre_extension_length=25e3, post_extension_length=25e3):
        """Generate the x-coordinate grid with extensions before and after undulated region"""
        extended_domain = pre_extension_length + self.domain_length + post_extension_length
        n_points = int(extended_domain/self.resolution)
        return np.linspace(0, extended_domain, n_points)
        

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
    

    def find_nearest_peak(self, x, bed, wavelength, target_x, search_start=None, search_end=None):
        """
        Find the nearest peak (local maximum) to the target position
        where the derivative is approximately zero.
        
        Args:
            x: x-coordinate array
            bed: bedrock elevation array  
            wavelength: wavelength of the undulations
            target_x: target position to find peak near
            search_start: optional start of search region
            search_end: optional end of search region
            
        Returns:
            peak_x: x-coordinate of nearest peak
        """
        # Determine search region
        if search_start is None:
            search_start = 0
        if search_end is None:
            search_end = np.inf
            
        # Only look in the specified search region
        search_mask = (x >= search_start) & (x <= search_end)
        x_search = x[search_mask]
        bed_search = bed[search_mask]
        
        if len(x_search) < 3:
            return target_x
        
        # Calculate derivatives to find where slope is near zero
        dx = x_search[1] - x_search[0]
        derivatives = np.gradient(bed_search, dx)
        
        # Find potential peaks (where derivative changes from + to -)
        # Look for points where derivative is close to zero and bed value is high
        derivative_threshold = np.max(np.abs(derivatives)) * 0.1  # 10% of max derivative
        near_zero_derivative = np.abs(derivatives) < derivative_threshold
        
        # Also check that it's actually a local maximum (not minimum)
        local_maxima = []
        for i in range(1, len(bed_search) - 1):
            if (bed_search[i] > bed_search[i-1] and bed_search[i] > bed_search[i+1] and 
                near_zero_derivative[i]):
                local_maxima.append(i)
        
        if not local_maxima:
            # Fallback: just find the point with minimum derivative magnitude
            min_deriv_idx = np.argmin(np.abs(derivatives))
            return x_search[min_deriv_idx]
        
        # Find the local maximum closest to target_x
        distances = [abs(x_search[i] - target_x) for i in local_maxima]
        closest_peak_idx = local_maxima[np.argmin(distances)]
        
        return x_search[closest_peak_idx]

    def smoothing(self, x, bed, wavelength, initial_elevation, profile_id=None, smoothing_length=2e3, 
                 pre_extension_length=25e3, post_extension_length=25e3):
        """
        Apply smoothing at both transitions: pre-extension to undulated and undulated to post-extension.
        Exception: Profiles 000 and 001 use flat linear trend instead of smoothing.
        
        Args:
            x: x-coordinate array
            bed: bedrock elevation array
            wavelength: wavelength of undulations (needed to find peaks)
            initial_elevation: initial elevation parameter
            profile_id: profile identifier for special cases (e.g., profile 001)
            smoothing_length: length of smoothing zone in meters (default: 2000m)
            pre_extension_length: length of pre-extension region
            post_extension_length: length of post-extension region
            
        Returns:
            smoothed_bed: bedrock with smoothing applied at both transitions
        """
        smoothed_bed = bed.copy()
        
        # Find transition points
        undulated_start = pre_extension_length
        undulated_end = pre_extension_length + self.domain_length
        
        # --- MODIFICATION: Added profile 0 to special case ---
        # Special case for profile 000 (Gaussian) and 001 (flat): use flat linear trend
        if profile_id in [0, 1]:
            # For these profiles, set both extensions to follow base slope without undulations
            pre_extension_mask = x < undulated_start
            post_extension_mask = x >= undulated_end
            
            if np.any(pre_extension_mask):
                # Pre-extension should follow base slope level (no amplitude)
                pre_extension_bed = initial_elevation + self.base_slope * x[pre_extension_mask]
                smoothed_bed[pre_extension_mask] = pre_extension_bed
                
            if np.any(post_extension_mask):
                post_extension_bed = initial_elevation + self.base_slope * x[post_extension_mask]
                smoothed_bed[post_extension_mask] = post_extension_bed
            return smoothed_bed
        
        # Normal smoothing for all other profiles
        
        # For normal profiles, pre-extension is already at peak height, so minimal smoothing needed
        # The transition from pre-extension (at peak height) to undulated region should be smooth
        # since the undulation starts at a peak (cos(0) = 1)
        
        # Smooth transition from undulated region to post-extension
        if smoothing_length > 0:
            # Find the nearest peak before undulated_end to start the post-smoothing
            ideal_post_smooth_start = undulated_end - smoothing_length
            actual_post_smooth_start = self.find_nearest_peak(x, bed, wavelength, ideal_post_smooth_start,
                                                            search_end=undulated_end)
            
            post_smooth_end = x[-1]
            post_smooth_mask = (x >= actual_post_smooth_start) & (x <= post_smooth_end)
            
            if np.any(post_smooth_mask):
                x_post_smooth = x[post_smooth_mask]
                
                # Start value: value at the peak in undulated region
                start_idx_post = np.where(x >= actual_post_smooth_start)[0][0]
                y0_post = bed[start_idx_post]
                dy0_dx_post = 0  # Peak has zero derivative
                
                # End value: base slope at post_smooth_end
                y1_post = initial_elevation + self.base_slope * post_smooth_end
                dy1_dx_post = self.base_slope  # Base slope derivative
                
                # Linear interpolation for smooth transition
                smoothed_values_post = y0_post + self.base_slope * (x_post_smooth - actual_post_smooth_start)
                smoothed_bed[post_smooth_mask] = smoothed_values_post
        
        return smoothed_bed
    

    def generate_bedrock_profile(self, 
                            amplitude, 
                            wavelength, 
                            skewness=0, 
                            kurtosis=0, 
                            noise_level=0.0,
                            initial_elevation=1.0,
                            pre_extension_length=25e3,
                            post_extension_length=25e3,
                            profile_id=None,
                            smoothing_length=2e3):
        """
        Generate a complete bedrock profile with all features
        Pre-extension (flat) -> undulated domain -> post-extension (flat)
        with smoothing for smooth transitions (except profile 000/001 uses linear trend)
        """
        # Generate the extended x grid
        x = self.generate_x_grid(pre_extension_length, post_extension_length)
        
        # Identify which points are in different regions
        undulated_start = pre_extension_length
        undulated_end = pre_extension_length + self.domain_length
        
        pre_extension_mask = x < undulated_start
        undulated_domain_mask = (x >= undulated_start) & (x < undulated_end)
        post_extension_mask = x >= undulated_end
        
        # Generate the base bed (sloped line) for the entire domain
        base_bed = self.generate_base_bed(x, initial_elevation)
        
        # Apply cosine undulation or special perturbation only to the undulated domain
        bed = base_bed.copy()
        # Shift the undulation to start from x=0 within the undulated region
        x_undulated = x[undulated_domain_mask] - undulated_start
        
        # --- MODIFICATION: Added logic for Profile 000 ---
        if amplitude > 0:
            if profile_id == 0:  # Special case for single Gaussian bump
                # Center of the undulated domain
                mu = self.domain_length / 2.0
                # The 'wavelength' parameter is interpreted as the standard deviation (sigma)
                sigma = wavelength
                
                if sigma > 0:
                    gaussian_bump = amplitude * np.exp(-((x_undulated - mu)**2) / (2 * sigma**2))
                    bed[undulated_domain_mask] += gaussian_bump

            else:  # Standard cosine undulations for all other profiles
                if wavelength > 0:
                    omega = 2 * np.pi / wavelength
                    bed[undulated_domain_mask] += amplitude * np.cos(omega * x_undulated)
        # --- END MODIFICATION ---

        # Set pre-extension to start at peak height (amplitude above base slope)
        # NOTE: This will be overridden for profiles 0 and 1 in the smoothing function
        bed[pre_extension_mask] = initial_elevation + self.base_slope * x[pre_extension_mask] + amplitude
        
        # Apply transformations only to the undulated domain portion
        if skewness != 0:
            undulated_bed = bed[undulated_domain_mask]
            transformed_bed = self.apply_skewness(undulated_bed, skewness)
            bed[undulated_domain_mask] = transformed_bed
        
        if kurtosis != 0:
            undulated_bed = bed[undulated_domain_mask]
            transformed_bed = self.apply_kurtosis(undulated_bed, kurtosis)
            bed[undulated_domain_mask] = transformed_bed
        
        if noise_level > 0:
            undulated_bed = bed[undulated_domain_mask]
            noisy_bed = self.add_noise(undulated_bed, noise_level)
            bed[undulated_domain_mask] = noisy_bed
        
        # Apply smoothing for transitions (both sides of undulated region)
        bed = self.smoothing(x, bed, wavelength, initial_elevation, profile_id, smoothing_length, 
                           pre_extension_length, post_extension_length)
        
        # Make a copy for the final result
        final_bed = np.copy(bed)
        
        return x, final_bed


    def create_parameter_combinations(self, n_samples=100):
        """
        Create a systematic set of parameter combinations covering the parameter space
        """
        param_list = []
        
        # --- ADDITION: Define and add Profile 000 ---
        # Profile 000: A single Gaussian bump on a flat slope
        profile000_params = {
            'amplitude': np.median(self.param_ranges['amplitude']),
            'wavelength': 3.3 * self.ice_thickness, # For Gaussian, this is treated as sigma
            'skewness': 0.0,
            'kurtosis': 0.0,
            'noise_level': 0.0,
            'initial_elevation': 1.0
        }
        param_list.append(profile000_params)
        
        # --- Profile 001 (formerly first profile): A completely flat profile ---
        configp_params = {
            'amplitude': 0.00,  # No undulations amplitude
            'wavelength': 3.3 * ICE_THICKNESS, # A non-zero wavelength is kept for consistency
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
                        # Skip if this is identical to the configp params (flat profile)
                        if (abs(amplitude) < 1e-9 and abs(skewness) < 1e-9 and abs(kurtosis) < 1e-9):
                            continue
                            
                        params = {
                            'amplitude': amplitude,
                            'wavelength': wavelength,
                            'skewness': skewness,
                            'kurtosis': kurtosis,
                            'noise_level': 0.0,
                            'initial_elevation': 1.0
                        }
                        # Avoid duplicates
                        if params not in param_list:
                            param_list.append(params)
        
        # If we have more combinations than needed, select a representative subset
        if len(param_list) > n_samples:
            # Always keep the first two entries (Gaussian and flat)
            first_entries = param_list[:2]
            
            # Take evenly spaced indices to get n_samples-2 from the rest of the list
            remaining = param_list[2:]
            indices = np.linspace(0, len(remaining) - 1, n_samples-2).astype(int)
            param_list = first_entries + [remaining[i] for i in indices]
        
        # If we need more combinations, add some with noise
        elif len(param_list) < n_samples:
            # How many more do we need?
            remaining_needed = n_samples - len(param_list)
            
            # Add profiles with noise
            for i in range(remaining_needed):
                # Pick a random existing parameter set (but not the first two special cases)
                if len(param_list) > 2:
                    base_params = param_list[2 + (i % (len(param_list)-2))].copy()
                else: # Fallback if only special cases exist
                    base_params = param_list[1].copy()
                
                # Add increasing noise levels
                base_params['noise_level'] = 0.05 + (i / remaining_needed) * 0.15
                
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
            f.write(f"Base Slope: {self.base_slope}\n\n")
            
            # Generate each profile
            for i, params in enumerate(param_combinations):
                # --- MODIFICATION: Use 0-based indexing for profile IDs ---
                profile_id = i 
                print(f"Generating profile {profile_id:03d}/{len(param_combinations)-1:03d}...")
                
                # Generate the profile
                x, bed = self.generate_bedrock_profile(
                    amplitude=params['amplitude'],
                    wavelength=params['wavelength'],
                    skewness=params['skewness'],
                    kurtosis=params['kurtosis'],
                    noise_level=params['noise_level'],
                    initial_elevation=params['initial_elevation'],
                    profile_id=profile_id,
                )
                
                # Save it
                self.save_profile(x, bed, params, profile_id)
                
                # Add metadata
                f.write(f"Profile {profile_id:03d}:\n")
                if profile_id == 0:
                    f.write(f"  Type: Single Gaussian Bump\n")
                elif params['amplitude'] == 0:
                    f.write(f"  Type: Flat Reference\n")

                f.write(f"  Amplitude: {params['amplitude'] / 1e3:.6f} km\n")
                f.write(f"  Wavelength: {params['wavelength'] / 1e3:.6f} km\n")
                f.write(f"  Skewness: {params['skewness']:.6f}\n")
                f.write(f"  Kurtosis: {params['kurtosis']:.6f}\n")
                f.write(f"  Noise Level: {params['noise_level']:.6f}\n")
                f.write(f"  Initial Elevation: {params['initial_elevation']:.6f}\n\n")
                
        print(f"\nGenerated {len(param_combinations)} bedrock profiles in '{self.output_dir}'")
        print(f"Metadata saved to '{metadata_file}'")
    

class SyntheticBedrockModelConfig:
    """Model configuration with synthetic bedrock profiles for ISSM"""
    
    def __init__(self, profile_id=1, output_dir="bedrock_profiles"):
        self.domain_length = DOMAIN_LENGTH
        self.resolution = RESOLUTION
        self.ice_thickness = ICE_THICKNESS
        self.base_slope = BASE_SLOPE
        self.param_ranges = PARAM_RANGES
        
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
        # Domain extents from the profile (now includes 25km pre + 25km post extensions)
        self.x_params = {
            'start': self.x_profile[0],
            'end': self.x_profile[-1],
            'step': self.resolution
        }
        
        # Store original domain length for reference
        self.original_domain_length = DOMAIN_LENGTH
        
        # Ice parameters
        self.ice_params = {
            'mean_thickness': self.ice_thickness,
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
    # Note: The total number of unique profiles is 1 (Gaussian) + 1 (Flat) + 7*5*5*5 (Combinations) = 877
    # The create_parameter_combinations function handles sampling if n_profiles is smaller.
    generator.generate_all_profiles(n_profiles=877)