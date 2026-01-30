import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import welch


def generate_anisotropic_bed(
    N: int,
    resolution_m: float,
    beta_flow: float,
    beta_cross: float,
    target_relief_m: float,
    max_slope_deg: float
):    
    # Frequency grids
    kx = np.fft.fftfreq(N, d=resolution_m)
    ky = np.fft.fftfreq(N, d=resolution_m)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')

    # Avoid division by zero
    kx_safe = np.where(kx_grid == 0, 1e-10, np.abs(kx_grid))
    ky_safe = np.where(ky_grid == 0, 1e-10, np.abs(ky_grid))

    # White noise canvas
    white_noise = np.random.standard_normal((N, N))
    fft_coeffs = np.fft.fftn(white_noise)
    
    # SEPARABLE spectral filter:
    # scaling_factor(kx, ky) ∝ |kx|^(-β_flow/2) × |ky|^(-β_cross/2)
    # => PSD(kx, ky) = |H|² ∝ |kx|^(-β_flow) × |ky|^(-β_cross)
    # => Marginal PSD(kx) = ∫ PSD dky ∝ |kx|^(-β_flow)
    scaling_factor = 1.0 / (kx_safe ** (beta_flow / 2) * ky_safe ** (beta_cross / 2))
    
    # Zero DC and axes to avoid singularities
    scaling_factor[0, 0] = 0
    scaling_factor[0, :] = 0
    scaling_factor[:, 0] = 0
    
    # Apply filter
    filtered_coeffs = fft_coeffs * scaling_factor
    
    # Back to spatial domain
    bed = np.fft.ifftn(filtered_coeffs).real
    
    # Normalize to target relief
    bed = (bed - bed.mean()) / (bed.std() + 1e-10)
    p_min, p_max = np.percentile(bed, [1, 99])
    current_relief = p_max - p_min
    bed *= (target_relief_m / current_relief)
    
    # Check slopes
    dy, dx = np.gradient(bed, resolution_m)
    current_max_slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2).max()))
    
    # Track what happens
    slope_limited = False
    scale_applied = 1.0
    
    if current_max_slope > max_slope_deg:
        slope_limited = True
        # Binary search for max acceptable scale
        # This preserves β because it's only amplitude scaling!
        scale_lo, scale_hi = 0.0, 1.0

        for _ in range(30):
            scale_mid = (scale_lo + scale_hi) / 2
            test_bed = bed * scale_mid
            dy, dx = np.gradient(test_bed, resolution_m)
            test_slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2).max()))
                        
            if test_slope > max_slope_deg:
                scale_hi = scale_mid
            else:
                scale_lo = scale_mid
                
            if abs(test_slope - max_slope_deg) < 0.5:
                break
        
        scale_applied = scale_lo
        bed *= scale_applied
    
    # Final diagnostics
    p_min, p_max = np.percentile(bed, [1, 99])
    final_relief = p_max - p_min
    
    dy, dx = np.gradient(bed, resolution_m)
    final_max_slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2).max()))
    
    diagnostics = {
        'slope_limited': slope_limited,
        'scale_applied': scale_applied,
        'final_relief_m': final_relief,
        'relief_pct': 100 * final_relief / target_relief_m,
        'final_max_slope_deg': final_max_slope
    }
    
    return bed, diagnostics


def inject_bedforms(bed, resolution_m, bedform_configs, max_slope_deg):
    """
    Add geologically informed bedform patterns (drumlins, MSGLs, etc.)
    """
    N = bed.shape[0]
    x = np.arange(N) * resolution_m
    y = np.arange(N) * resolution_m
    x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
    
    bedform_layer = np.zeros_like(bed)
    
    for wavelength_m, amplitude_m, angle_deg in bedform_configs:
        angle_rad = np.deg2rad(angle_deg)
        # 2D oriented wave formula
        phase = (2 * np.pi / wavelength_m) * (
            x_grid * np.cos(angle_rad) + y_grid * np.sin(angle_rad)
        )
        bedform_layer += amplitude_m * np.sin(phase)
    
    # Combined slope check
    combined = bed + bedform_layer
    dy, dx = np.gradient(combined, resolution_m)
    max_slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2).max()))
    
    if max_slope > max_slope_deg:
        # Binary search for safe bedform amplitude 
        scale_lo, scale_hi = 0.0, 1.0
        for _ in range(20):
            scale_mid = (scale_lo + scale_hi) / 2
            test_bed = bed + bedform_layer * scale_mid
            dy, dx = np.gradient(test_bed, resolution_m)
            test_slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2).max()))
            if test_slope > max_slope_deg:
                scale_hi = scale_mid
            else:
                scale_lo = scale_mid
        
        print(f"Bedforms were scaled to {scale_lo:.0%} to satisfy {max_slope_deg}° limit.")
        bedform_layer *= scale_lo
    
    return bed + bedform_layer


def compute_marginal_beta(bed, resolution_m, direction='x'):
    """
    Compute β from 2D FFT marginal spectrum.
    
    This measures β the same way analyse_bedrock.py would measure it
    from a flight line along the given direction.
    """
    N = bed.shape[0]
    fft2 = np.fft.fft2(bed)
    power2d = np.abs(fft2)**2
    
    kx = np.fft.fftfreq(N, d=resolution_m)
    ky = np.fft.fftfreq(N, d=resolution_m)
    
    if direction == 'x':
        # Marginal P(kx) = average over ky
        power_1d = np.mean(power2d, axis=1)
        freqs = np.abs(kx)
    else:
        # Marginal P(ky) = average over kx
        power_1d = np.mean(power2d, axis=0)
        freqs = np.abs(ky)
    
    # Sort by frequency
    idx = np.argsort(freqs)
    freqs = freqs[idx]
    power_1d = power_1d[idx]
    
    # Fit power law (skip DC)
    mask = (freqs > freqs[1]) & (freqs < freqs.max() * 0.5) & (power_1d > 0)
    if mask.sum() < 5:
        return np.nan, 0
    
    log_f = np.log10(freqs[mask])
    log_p = np.log10(power_1d[mask])
    slope, _, r2, _, _ = linregress(log_f, log_p)
    
    return -slope, r2**2


def validate_anisotropy(bed, name, resolution_m, target_beta_flow, target_beta_cross):
    """Validate that generated bed has correct β values."""
    
    print("\n=== ANISOTROPY VALIDATION ===")
    
    # 1. EXTRACT PROFILES
    mid_row_idx = bed.shape[0] // 2
    profile_flow = bed[mid_row_idx, :]
    
    mid_col_idx = bed.shape[1] // 2
    profile_cross = bed[:, mid_col_idx]
    
    # DEFINE DISTANCE
    dist = np.arange(len(profile_flow)) * resolution_m / 1000 # km
    
    # 2. CALCULATE SPECTRA (Welch's Method for clean 1D profiles)
    freq_x, psd_x = welch(profile_flow, fs=1/resolution_m, nperseg=len(profile_flow)//4)
    freq_y, psd_y = welch(profile_cross, fs=1/resolution_m, nperseg=len(profile_cross)//4)
    
    # 3. MEASURE BETAS (Marginal spectra)
    beta_x, r2_x = compute_marginal_beta(bed, resolution_m, 'x')
    beta_y, r2_y = compute_marginal_beta(bed, resolution_m, 'y')
    
    print(f"X (Flow):  β = {beta_x:.2f} (target {target_beta_flow}, R²={r2_x:.2f})")
    print(f"Y (Cross): β = {beta_y:.2f} (target {target_beta_cross}, R²={r2_y:.2f})")
    
    # 4. PLOTTING
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Top Left: Spatial Profiles
    ax = axes[0,0]
    ax.plot(dist, profile_flow, label=f'Flow (X) - β={beta_x:.2f}', color='C0')
    ax.plot(dist, profile_cross + 500, label=f'Cross (Y) - β={beta_y:.2f} (+500m off)', color='C1')
    ax.set_title("1D Spatial Profiles (Slices)")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Elevation (m)")
    ax.legend()
    
    # Top Right: Spectral Comparison (Log-Log)
    ax = axes[0,1]
    ax.loglog(freq_x, psd_x, color='C0', label='Flow PSD')
    ax.loglog(freq_y, psd_y, color='C1', label='Cross PSD')
    ax.set_title("Spectral Signature (Welch)")
    ax.set_xlabel("Frequency (1/m)")
    ax.set_ylabel("Power")
    ax.legend()
    
    # Bottom Left: Map with Cut Lines
    ax = axes[1,0]
    im = ax.imshow(bed.T, cmap='terrain', origin='lower', extent=[0, dist.max(), 0, dist.max()])
    ax.axhline(mid_col_idx * resolution_m / 1000, color='C0', linestyle='--', label='Cross Slice')
    ax.axvline(mid_row_idx * resolution_m / 1000, color='C1', linestyle='--', label='Flow Slice')
    ax.set_xlabel("X - Flow (km)")
    ax.set_ylabel("Y - Cross (km)")
    ax.set_title("Slice Locations")
    plt.colorbar(im, ax=ax, label='Elevation (m)')
    ax.legend()
    
    # Bottom Right: Slope distribution [cite: 34, 35]
    ax = axes[1, 1]
    dy, dx = np.gradient(bed, resolution_m)
    slopes = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    ax.hist(slopes.ravel(), bins=50, density=True, alpha=0.7, color='C2')
    ax.axvline(slopes.max(), color='r', ls='--', label=f'Max: {slopes.max():.1f}°')
    ax.set_xlabel('Slope (°)')
    ax.set_title('Slope Distribution (Constraint Check)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{name}_anisotropy_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return beta_x, beta_y

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    
    configs = {
        'ROSS': {
            'size_km': 400,
            'resolution_m': 500,
            'target_relief_m': 1100,
            'max_slope_deg': 15.0,
            'beta_flow': 2.5,
            'beta_cross': 1.8,
            'bedforms': [(10000, 60, 30), (800, 20, 90)]
        },
        'PEL': {
            'size_km': 200,
            'resolution_m': 150,
            'target_relief_m': 2000,
            'max_slope_deg': 40.0,
            'beta_flow': 2.0,
            'beta_cross': 1.8,
            'bedforms': [(500, 30, 0)]
        }
    }
    
    for name, cfg in configs.items():
        print(f"\n{'='*50}")
        print(f"GENERATING: {name}")
        print(f"Resolution: {cfg['resolution_m']}m | Max allowed slope: {cfg['max_slope_deg']}°")
        print(f"Target β: flow={cfg['beta_flow']}, cross={cfg['beta_cross']}")
        print('='*50)
        
        N = int(cfg['size_km'] * 1000 / cfg['resolution_m'])
        
        bed, diag = generate_anisotropic_bed(
            N=N,
            resolution_m=cfg['resolution_m'],
            beta_flow=cfg['beta_flow'],
            beta_cross=cfg['beta_cross'],
            target_relief_m=cfg['target_relief_m'],
            max_slope_deg=cfg['max_slope_deg'],
        )
        
        print(f"\nBase bed generated:")
        print(f"  Final Relief: {diag['final_relief_m']:.0f}m ({diag['relief_pct']:.0f}% of target)")
        print(f"  Max slope: {diag['final_max_slope_deg']:.1f}°")
        if diag['slope_limited']:
            print(f"Relief was reduced to satisfy slope constraint")
        
        # Add bedforms
        if cfg.get('bedforms'):
            print(f"\nInjecting {len(cfg['bedforms'])} bedform layers...")
            bed = inject_bedforms(
                bed, 
                cfg['resolution_m'], 
                cfg['bedforms'], 
                cfg['max_slope_deg']
            )
        
        # Validate
        beta_x, beta_y = validate_anisotropy(
            bed, name, cfg['resolution_m'],
            cfg['beta_flow'], cfg['beta_cross']
        )
        
        print(f"\n{'*~'*25}")