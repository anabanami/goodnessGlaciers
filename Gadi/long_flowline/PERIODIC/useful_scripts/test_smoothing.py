#!/usr/bin/env python3
"""Test script for smoothing in bedrock_generator.py"""

import numpy as np
import matplotlib.pyplot as plt
from bedrock_generator import BedrockGenerator

def test_smoothing():
    """Test the smoothing functionality"""
    
    # Create generator instance
    generator = BedrockGenerator()
    
    # Generate test profile (165)
    test_params = {
        'amplitude': 22.400000000000002,
        'wavelength': 6336.0,
        'skewness': 0,
        'kurtosis': 0.10,
        'noise_level': 0,
        'initial_elevation': 1000,  # 1km initial elevation
        'extension_length': 50e3,  # 50km extension
        'smoothing_length': 2e3   # 2km smoothing zone
    }
    
    print("Generating test profile...")
    x, bed = generator.generate_bedrock_profile(**test_params)
    
    # Find where the smoothing actually starts (at the peak)
    ideal_smooth_start = generator.domain_length - test_params['smoothing_length']
    actual_smooth_start = generator.find_nearest_peak(x, bed, test_params['wavelength'], ideal_smooth_start)
    
    # Generate profile without smoothing for comparison
    test_params_no_smooth = test_params.copy()
    test_params_no_smooth['smoothing_length'] = 0
    
    # Temporarily disable smoothing by creating profile manually
    x_no_smooth = generator.generate_x_grid(test_params['extension_length'])
    original_domain_mask = x_no_smooth <= generator.domain_length
    base_bed = generator.generate_base_bed(x_no_smooth, test_params['initial_elevation'])
    omega = 2 * np.pi / test_params['wavelength']
    bed_no_smooth = base_bed.copy()
    bed_no_smooth[original_domain_mask] += test_params['amplitude'] * np.cos(omega * x_no_smooth[original_domain_mask])
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot both profiles
    plt.subplot(2, 1, 1)
    plt.plot(x / 1e3, bed, 'b-', label='With smoothing', linewidth=2)
    plt.plot(x_no_smooth / 1e3, bed_no_smooth, 'r--', label='Without smoothing', linewidth=1, alpha=0.7)
    
    # Mark the domain boundary and smoothing zone
    domain_end = generator.domain_length / 1e3
    ideal_smooth_start = (generator.domain_length - test_params['smoothing_length']) / 1e3
    actual_smooth_start_km = actual_smooth_start / 1e3
    smooth_end = (generator.domain_length + test_params['smoothing_length']) / 1e3
    
    plt.axvline(x=domain_end, color='g', linestyle='-', alpha=0.7, label='Domain end')
    plt.axvline(x=ideal_smooth_start, color='gray', linestyle='--', alpha=0.5, label='Ideal smooth start')
    plt.axvline(x=actual_smooth_start_km, color='orange', linestyle=':', alpha=0.7, label='Actual smooth start (peak)')
    plt.axvline(x=smooth_end, color='red', linestyle=':', alpha=0.7, label='Smoothing end')
    plt.fill_betweenx(plt.ylim(), actual_smooth_start_km, smooth_end, alpha=0.2, color='orange', label='Smoothing zone')
    
    plt.xlabel('Distance (km)')
    plt.ylabel('Bed elevation (m)')
    plt.title('Bedrock Profile with Smoothing')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zoom in on transition region
    plt.subplot(2, 1, 2)
    
    # Focus on transition region
    transition_mask = (x / 1e3 >= actual_smooth_start_km - 1) & (x / 1e3 <= smooth_end + 1)
    transition_mask_no_smooth = (x_no_smooth / 1e3 >= actual_smooth_start_km - 1) & (x_no_smooth / 1e3 <= smooth_end + 1)
    
    plt.plot(x[transition_mask] / 1e3, bed[transition_mask], 'b-', label='With smoothing', linewidth=2)
    plt.plot(x_no_smooth[transition_mask_no_smooth] / 1e3, bed_no_smooth[transition_mask_no_smooth], 'r--', 
             label='Without smoothing', linewidth=1, alpha=0.7)
    
    plt.axvline(x=domain_end, color='g', linestyle='-', alpha=0.7, label='Domain end')
    plt.axvline(x=ideal_smooth_start, color='gray', linestyle='--', alpha=0.5, label='Ideal smooth start')
    plt.axvline(x=actual_smooth_start_km, color='orange', linestyle=':', alpha=0.7, label='Actual smooth start (peak)')
    plt.axvline(x=smooth_end, color='red', linestyle=':', alpha=0.7, label='Smoothing end')
    plt.fill_betweenx(plt.ylim(), actual_smooth_start_km, smooth_end, alpha=0.2, color='orange', label='Smoothing zone')
    
    plt.xlabel('Distance (km)')
    plt.ylabel('Bed elevation (m)')
    plt.title('Transition Region (Zoomed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('smoothing_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Calculate and print some metrics
    print(f"\nTest Results:")
    print(f"Domain length: {generator.domain_length / 1e3:.1f} km")
    print(f"Ideal smoothing zone: {ideal_smooth_start:.1f} - {smooth_end:.1f} km")
    print(f"Actual smoothing zone: {actual_smooth_start_km:.1f} - {smooth_end:.1f} km (total: {smooth_end - actual_smooth_start_km:.1f} km)")
    print(f"Peak found at: {actual_smooth_start_km:.1f} km (shift: {actual_smooth_start_km - ideal_smooth_start:.2f} km)")
    print(f"Total profile length: {x[-1] / 1e3:.1f} km")
    
    # Check continuity at smoothing boundaries
    actual_smooth_start_idx = np.argmin(np.abs(x - actual_smooth_start))
    domain_end_idx = np.argmin(np.abs(x - generator.domain_length))
    smooth_end_idx = np.argmin(np.abs(x - (generator.domain_length + test_params['smoothing_length'])))
    
    print(f"Bed elevation at smoothing start (peak): {bed[actual_smooth_start_idx]:.2f} m")
    print(f"Bed elevation at domain end: {bed[domain_end_idx]:.2f} m")  
    print(f"Bed elevation at smoothing end: {bed[smooth_end_idx]:.2f} m")
    
    # Calculate derivatives to check smoothness
    dx = x[1] - x[0]
    derivatives = np.gradient(bed, dx)
    
    print(f"Derivative at smoothing start (peak): {derivatives[actual_smooth_start_idx]:.6f}")
    print(f"Derivative at domain end: {derivatives[domain_end_idx]:.6f}")
    print(f"Derivative at smoothing end: {derivatives[smooth_end_idx]:.6f}")
    
    # Check derivatives in a small region around smoothing end
    end_region_mask = (x >= (generator.domain_length + test_params['smoothing_length'] - 200)) & \
                      (x <= (generator.domain_length + test_params['smoothing_length'] + 200))
    print(f"Max derivative magnitude near smoothing end: {np.max(np.abs(derivatives[end_region_mask])):.6f}")
    
    # Check if the extension beyond smoothing follows base slope
    extension_mask = x > (generator.domain_length + test_params['smoothing_length'])
    if np.any(extension_mask):
        # Calculate expected base bed values for the extension
        expected_extension = generator.generate_base_bed(x[extension_mask], test_params['initial_elevation'])
        actual_extension = bed[extension_mask]
        
        extension_error = np.abs(actual_extension - expected_extension)
        max_error = np.max(extension_error)
        print(f"Max error from base slope in extension region: {max_error:.6f} m")
        
        if max_error < 1e-10:
            print("✓ Extension region properly follows base slope trend")
        else:
            print("⚠ Extension region deviates from base slope trend")
    
    print("Test completed. Check _smoothing_test.png' for visual verification.")

if __name__ == "__main__":
    test_smoothing()