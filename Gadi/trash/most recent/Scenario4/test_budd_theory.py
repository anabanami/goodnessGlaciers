# test_budd_theory.py - Script to test Budd theory predictions
# Run this to validate your implementation

import numpy as np
import matplotlib.pyplot as plt
from configf9_synthetic import ModelConfig
from bedrock_generator import BedrockGenerator

def test_wavelength_response():
    """Test how different wavelengths affect ice flow according to Budd theory"""
    
    # Test wavelengths around the critical ratio
    ice_thickness = 1.92e3  # m
    test_wavelengths = np.array([2.0, 3.3, 5.0, 8.0]) * ice_thickness
    
    results = []
    
    print("Testing Budd theory predictions...")
    print("=" * 50)
    
    for i, wavelength in enumerate(test_wavelengths):
        print(f"\nTesting λ = {wavelength/1e3:.1f} km (λ/H = {wavelength/ice_thickness:.1f})")
        
        # Create a test configuration
        config = ModelConfig(profile_id=27)  # ADJUST
        
        # Calculate predicted damping factor
        from Budd_theory import BuddTheoryImplementation
        budd_impl = BuddTheoryImplementation(config)
        
        psi = budd_impl.calculate_damping_factor(wavelength, ice_thickness)
        
        # Theoretical predictions
        w_Z = 2 * np.pi * ice_thickness / wavelength
        
        results.append({
            'wavelength': wavelength,
            'lambda_over_H': wavelength / ice_thickness,
            'w_Z': w_Z,
            'damping_factor': psi,
            'relative_amplitude': 1/psi if psi > 0 else float('inf')
        })
        
        print(f"  w*Z = {w_Z:.2f}")
        print(f"  Damping factor ψ = {psi:.4f}")
        print(f"  Surface/bed amplitude ratio = {1/psi if psi > 0 else 'inf':.3f}")
        
        # Energy dissipation factor (peaks at λ/H ≈ 3.3)
        optimal_ratio = 3.3
        current_ratio = wavelength / ice_thickness
        energy_factor = np.exp(-0.5 * ((current_ratio - optimal_ratio) / 2.0)**2)
        print(f"  Energy dissipation factor = {energy_factor:.4f}")
    
    # Create summary plot
    plot_budd_predictions(results)
    
    return results

def plot_budd_predictions(results):
    """Plot Budd theory predictions"""
    
    lambda_over_H = [r['lambda_over_H'] for r in results]
    damping_factors = [r['damping_factor'] for r in results]
    relative_amplitudes = [r['relative_amplitude'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Damping factor vs wavelength ratio
    ax1.semilogy(lambda_over_H, damping_factors, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=3.3, color='r', linestyle='--', alpha=0.7, label='Optimal λ/H = 3.3')
    ax1.set_xlabel('Wavelength/Thickness (λ/H)')
    ax1.set_ylabel('Damping Factor ψ')
    ax1.set_title('Budd Theory: Damping vs Wavelength')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Surface undulation amplitude relative to bed
    finite_amplitudes = [ra for ra in relative_amplitudes if ra != float('inf')]
    finite_ratios = [lambda_over_H[i] for i, ra in enumerate(relative_amplitudes) if ra != float('inf')]
    
    ax2.plot(finite_ratios, finite_amplitudes, 'go-', linewidth=2, markersize=8)
    ax2.axvline(x=3.3, color='r', linestyle='--', alpha=0.7, label='Optimal λ/H = 3.3')
    ax2.set_xlabel('Wavelength/Thickness (λ/H)')
    ax2.set_ylabel('Surface/Bed Amplitude Ratio')
    ax2.set_title('Surface Undulation Response')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('budd_theory_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nBudd theory predictions saved to 'budd_theory_predictions.png'")

def analyze_profile_wavelengths():
    """Analyze the wavelengths in your generated profiles"""
    
    print("\nAnalyzing generated bedrock profiles...")
    print("=" * 50)
    
    # Load metadata to see what wavelengths you have
    import os
    metadata_file = "bedrock_profiles/bedrock_metadata.txt"
    
    if os.path.exists(metadata_file):
        print("Available profiles:")
        
        # Simple parsing - you might need to adjust this
        with open(metadata_file, 'r') as f:
            lines = f.readlines()
            
        ice_thickness = 1.92e3  # m
        
        for line in lines:
            if "Profile" in line and ":" in line:
                profile_id = line.split()[1].replace(':', '')
            elif "Wavelength:" in line:
                wavelength = float(line.split()[1])
                lambda_over_H = wavelength / ice_thickness
                
                # Identify interesting cases
                note = ""
                if abs(lambda_over_H - 3.3) <= 0.3:
                    note = " ← OPTIMAL for Budd theory!"
                elif lambda_over_H <= 2.5:
                    note = " ← Short wave (high damping)"
                elif 3.6 < lambda_over_H <= 6.0:
                    note = " ← Intermediate wave (moderate damping)"
                elif lambda_over_H > 6.0:
                    note = " ← Long wave (low damping)"
                
                print(f"  Profile {profile_id}: λ = {wavelength/1e3:.1f} km, λ/H = {lambda_over_H:.2f}{note}")
    else:
        print("Metadata file not found. Generate profiles first.")

def quick_convergence_test():
    """Quick test to see if convergence improved"""
    
    print("\nTesting convergence with new Budd implementation...")
    print("=" * 50)
    
    try:
        # Test with a profile near the optimal wavelength
        config = ModelConfig(profile_id=152)  # ADJUST
        
        print(f"Testing profile {config.profile_id}:")
        print(f"  Wavelength: {config.bedrock_params['lambda']/1e3:.1f} km")
        print(f"  λ/H ratio: {config.bedrock_params['lambda']/config.ice_params['mean_thickness']:.2f}")
        print(f"  Amplitude: {config.bedrock_params['amplitude']:.1f} m")
        
        # You would run a short ISSM simulation here to test convergence
        print("\n  → Run your ISSM simulation to test convergence!")
        print("  → Check for:")
        print("    - Faster convergence")
        print("    - More stable solutions") 
        print("    - Physically reasonable sliding coefficients")
        
    except Exception as e:
        print(f"Error loading profile: {e}")

if __name__ == "__main__":
    print("Budd Theory Implementation Test")
    print("=" * 50)
    
    # Run tests
    results = test_wavelength_response()
    analyze_profile_wavelengths()
    quick_convergence_test()
    
    print("\n" + "=" * 50)
    print("Test completed! Check the diagnostic plots and run ISSM simulations.")
    print("Key things to verify:")
    print("1. Surface undulations should be minimal for λ/H ≈ 3.3")
    print("2. Energy dissipation should peak around λ/H ≈ 3.3") 
    print("3. Convergence should be improved")
    print("4. Sliding coefficients should be physically reasonable")