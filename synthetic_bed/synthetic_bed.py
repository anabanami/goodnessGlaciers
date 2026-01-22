import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import linregress
from scipy.signal import welch

def generate_production_bed(size_km, resolution_m, beta_flow, beta_cross, target_relief_m, max_slope_threshold=15.0):

	# Spatial Setup
	L = size_km * 1000
	N = int(L / resolution_m)

	# Frequency Setup
	white_noise = np.random.standard_normal((N, N))
	fft_coeffs = np.fft.fftn(white_noise)
	kx = np.fft.fftfreq(N, d=resolution_m)
	ky = np.fft.fftfreq(N, d=resolution_m)
	kx_grid, ky_grid = np.meshgrid(kx, ky)

	# Anisotropy logic
	freq_radial = np.sqrt(kx_grid**2 + ky_grid**2)
	freq_radial[0, 0] = 1 

	theta = np.arctan2(ky_grid, kx_grid)
	# interpolate beta
	beta_map = beta_flow * (np.cos(theta))**2 + beta_cross * (np.sin(theta)**2)

	# Spectral filter
	scaling_factor = 1/ (freq_radial ** (beta_map / 2))
	filtered_coeffs = fft_coeffs * scaling_factor
	raw_bed = np.fft.ifftn(filtered_coeffs).real

	# ADAPTIVE PHYSICS: GAUSSIAN SMOOTHING (ACT AS SEDIMENT DRAPE)
	sigma_m = 0
	sigma_step = 50
	valid_bed = None

	print(f"Optimizing Bed for Max Slope < {max_slope_threshold}°")

	while True:
		if sigma_m == 0:
			current_bed = raw_bed.copy()
		else:
			sigma_pixels = sigma_m / resolution_m
			current_bed = gaussian_filter(raw_bed, sigma=sigma_pixels)

		# ROBUST normalisation to prevent rogue pizels from shrinking everything
		p_min, p_max = np.percentile(raw_bed, [1, 99])
		current_relief = p_max - p_min
		relief_scale_factor = target_relief_m / current_relief

		current_bed = current_bed * relief_scale_factor

		# Check slope
		dy, dx = np.gradient(current_bed, resolution_m) # 100m spacing
		slope = np.sqrt(dx**2 + dy**2)
		max_slope_deg = np.degrees(np.arctan(slope.max()))
		print(f"sigma={sigma_m}m -> Max Slope: {max_slope_deg:.2f}°")

		if max_slope_deg <= max_slope_threshold:
			valid_bed = current_bed
			print("✅Slopes are physically reasonable for ice flow.")
			break
		if sigma_m > 2000:
			print("Could not satisfy slope constraint even with massive smoothing.")
			valid_bed = current_bed
			break

		# Increase smoothing for next iteration
		sigma_m += sigma_step

	# Center bed at 0 for ISSM convenience
	valid_bed -= valid_bed.mean()


	return valid_bed, sigma_m


def validate_anisotropy(bed, resolution_m):
    """
    Slices the 2D bed into 1D profiles (Rows vs Cols) to verify anisotropy.
    Comparing Beta_Flow (X) vs Beta_Cross (Y).
    """
    print("\n=== 1D ANISOTROPY CHECK ===")
    
    # 1. EXTRACT PROFILES
    # Middle Row (Flow / X-direction)
    mid_row_idx = bed.shape[0] // 2
    profile_flow = bed[mid_row_idx, :]
    
    # Middle Column (Cross / Y-direction)
    mid_col_idx = bed.shape[1] // 2
    profile_cross = bed[:, mid_col_idx]
    
    # Distance vector
    dist = np.arange(len(profile_flow)) * resolution_m / 1000 # km
    
    # 2. CALCULATE SPECTRA (Welch's Method)
    # We use Welch instead of raw FFT for cleaner 1D spectra (less noise)
    freq_x, psd_x = welch(profile_flow, fs=1/resolution_m, nperseg=len(profile_flow)//4)
    freq_y, psd_y = welch(profile_cross, fs=1/resolution_m, nperseg=len(profile_cross)//4)
    
    # 3. FIT POWER LAWS
    def fit_beta(freqs, psd):
        # Mask frequencies (avoid DC and Nyquist edge)
        mask = (freqs > 0) & (freqs < 0.5 * np.max(freqs))
        log_f = np.log10(freqs[mask])
        log_p = np.log10(psd[mask])
        slope, intercept, _, _, _ = linregress(log_f, log_p)
        return -slope

    beta_x_meas = fit_beta(freq_x, psd_x)
    beta_y_meas = fit_beta(freq_y, psd_y)
    
    print(f"X-Direction (Flow) Beta: {beta_x_meas:.2f} (Target ~2.5)")
    print(f"Y-Direction (Cross) Beta: {beta_y_meas:.2f} (Target ~1.8)")
    
    # 4. PLOTTING
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Top Left: Spatial Profiles
    ax = axes[0,0]
    ax.plot(dist, profile_flow, label=f'Flow (X) - Beta={beta_x_meas:.2f}', color='C0')
    ax.plot(dist, profile_cross + 500, label=f'Cross (Y) - Beta={beta_y_meas:.2f} (+500m offset)', color='C1')
    ax.set_title("1D Spatial Profiles")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Elevation (m)")
    ax.legend()
    
    # Top Right: Spectral Comparison
    ax = axes[0,1]
    ax.loglog(freq_x, psd_x, color='C0', label='Flow PSD')
    ax.loglog(freq_y, psd_y, color='C1', label='Cross PSD')
    ax.set_title("Power Spectral Density")
    ax.set_xlabel("Frequency (1/m)")
    ax.legend()
    
    # Bottom: The Bed with cut lines
    ax = axes[1,0]
    im = ax.imshow(bed, cmap='gist_earth', origin='lower', extent=[0, dist.max(), 0, dist.max()])
    ax.axhline(mid_row_idx * resolution_m / 1000, color='C0', linestyle='--', label='Flow Slice')
    ax.axvline(mid_col_idx * resolution_m / 1000, color='C1', linestyle='--', label='Cross Slice')
    ax.set_title("Slicing Locations")
    ax.legend()
    
    # Cleanup empty plot
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.show()


# SOFT
final_bed, used_sigma = generate_production_bed(
	size_km=1500,
	resolution_m=500,
	beta_flow=2.5,	 # Smooth stream-lined
	beta_cross=1.8,	# Cross-roughness
	target_relief_m=1100, 
	max_slope_threshold=15.0
)

# Plotting
plt.figure(figsize=(12, 8))
plt.imshow(final_bed, cmap='gist_earth', origin='lower', extent=[0, 50, 50, 0])
plt.colorbar(label='Elevation (m)')
plt.title("Synthetic Ross Ice Shelf Bed (Streamlined X)")
plt.xlabel("X (Flow Direction) [km]")
plt.ylabel("Y (Cross Flow) [km]")
plt.title(f"Adaptive Bed Generation\n(Required Sigma: {used_sigma}m)")
plt.savefig('ROSS.png', dpi=500, bbox_inches='tight')
plt.show()

validate_anisotropy(final_bed, resolution_m=500)


# # HARD:
# final_hard_bed, used_sigma = generate_production_bed(
# 	size_km=1500,
# 	resolution_m=500,
# 	beta_flow=1.5,	# Bumpy hard bed
# 	beta_cross=1.8,	# Cross-roughness
# 	target_relief_m=2000, 
# 	max_slope_threshold=15.0
# )

# # Plotting
# plt.figure(figsize=(12, 8))
# plt.imshow(final_hard_bed, cmap='gist_earth', origin='lower', extent=[0, 50, 50, 0])
# plt.colorbar(label='Elevation (m)')
# plt.title("Synthetic PEL Ice Shelf Bed (hard bed)")
# plt.xlabel("X (Flow Direction) [km]")
# plt.ylabel("Y (Cross Flow) [km]")
# plt.title(f"Adaptive Bed Generation\n(Required Sigma: {used_sigma}m)")
# plt.savefig('PEL.png', dpi=500, bbox_inches='tight')
# plt.show()

# validate_anisotropy(final_hard_bed, resolution_m=500)