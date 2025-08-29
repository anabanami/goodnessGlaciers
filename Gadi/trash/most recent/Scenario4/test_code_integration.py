#!/usr/bin/env python3
"""
test_integration.py - Test script to verify Budd theory integration
Run this to check if all components work together properly
"""

import numpy as np
import sys
import os

def test_imports():
    """Test that all modules can be imported correctly"""
    print("=== Testing Imports ===")
    
    try:
        from bedrock_generator import BedrockGenerator, SyntheticBedrockModelConfig
        print("✓ bedrock_generator imported successfully")
    except ImportError as e:
        print(f"✗ bedrock_generator import failed: {e}")
        return False
    
    try:
        from Budd_theory import BuddTheoryImplementation
        print("✓ Budd_theory imported successfully")
    except ImportError as e:
        print(f"✗ Budd_theory import failed: {e}")
        return False
    
    try:
        from config_synthetic import ModelConfig
        print("✓ config_synthetic imported successfully")
    except ImportError as e:
        print(f"✗ config_synthetic import failed: {e}")
        return False
    
    return True


def test_bedrock_generation():
    """Test bedrock profile generation"""
    print("\n=== Testing Bedrock Generation ===")
    
    try:
        from bedrock_generator import BedrockGenerator
        
        # Create generator
        generator = BedrockGenerator()
        print("✓ BedrockGenerator created")
        
        # Test single profile generation
        x, bed = generator.generate_bedrock_profile(
            amplitude=50.0,  # 50m amplitude
            wavelength=10000.0,  # 10km wavelength
            skewness=0.0,
            kurtosis=0.0,
            noise_level=0.0,
            initial_elevation=1000.0
        )
        
        print(f"✓ Profile generated: {len(x)} points")
        print(f"  X range: {x.min()/1000:.1f} to {x.max()/1000:.1f} km")
        print(f"  Bed range: {bed.min():.1f} to {bed.max():.1f} m")
        print(f"  Bed amplitude: {(bed.max()-bed.min())/2:.1f} m")
        
        return True
        
    except Exception as e:
        print(f"✗ Bedrock generation failed: {e}")
        return False


def test_config_creation():
    """Test configuration creation"""
    print("\n=== Testing Configuration Creation ===")
    
    try:
        # First, create a simple bedrock profile for testing
        from bedrock_generator import BedrockGenerator
        generator = BedrockGenerator()
        
        # Generate a single test profile
        print("Generating test bedrock profile...")
        x, bed = generator.generate_bedrock_profile(
            amplitude=50.0,
            wavelength=10000.0,
            skewness=0.0,
            kurtosis=0.0,
            noise_level=0.0,
            initial_elevation=1000.0
        )
        
        # Save it manually for testing
        import os
        output_dir = "test_bedrock_profiles"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save test profile
        params = {
            'amplitude': 50.0,
            'wavelength': 10000.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'noise_level': 0.0,
            'initial_elevation': 1000.0
        }
        
        np.savez(f"{output_dir}/bedrock_profile_001.npz", 
                x=x, bed=bed, **params)
        
        # Create metadata file
        with open(f"{output_dir}/bedrock_metadata.txt", 'w') as f:
            f.write("Test bedrock profile\n")
            f.write("Profile 001: Test profile for integration testing\n")
        
        print("✓ Test bedrock profile created")
        
        # Now test config creation
        from config_synthetic import ModelConfig
        config = ModelConfig(profile_id=1, output_dir=output_dir)
        
        print("✓ ModelConfig created successfully")
        print(f"  Profile ID: {config.profile_id}")
        print(f"  Domain length: {(config.x_params['end']-config.x_params['start'])/1000:.1f} km")
        print(f"  Wavelength: {config.bedrock_params['lambda']/1000:.1f} km")
        print(f"  Amplitude: {config.bedrock_params['amplitude']:.1f} m")
        print(f"  Omega: {config.omega:.6f} rad/m")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration creation failed: {e}")
        return False


def test_budd_theory():
    """Test Budd theory implementation"""
    print("\n=== Testing Budd Theory Implementation ===")
    
    try:
        # Create a mock model object for testing
        class MockMesh:
            def __init__(self):
                self.numberofvertices = 100
                self.numberofelements = 90
                self.x = np.linspace(0, 150000, 100)  # 150 km domain
                self.y = np.zeros(100)
                
            def vertexflags(self, flag):
                if flag == 1:  # Basal nodes
                    return np.arange(10)  # First 10 nodes are basal
                elif flag == 2:  # Surface nodes  
                    return np.arange(90, 100)  # Last 10 nodes are surface
                else:
                    return np.array([])
        
        class MockGeometry:
            def __init__(self):
                x = np.linspace(0, 150000, 100)
                self.bed = 1000 + 50 * np.cos(2*np.pi*x/10000)  # 50m amplitude, 10km wavelength
                self.thickness = np.full(100, 1920)  # 1920m thick ice
                self.surface = self.bed + self.thickness
                self.base = self.bed
        
        class MockInitialization:
            def __init__(self):
                self.vx = np.full(100, 10.0)  # 10 m/yr velocity
                self.vy = np.zeros(100)
                self.temperature = np.full(100, 253.15)  # -20°C
                self.pressure = np.full(100, 1e7)  # 10 MPa
        
        class MockMaterials:
            def __init__(self):
                self.rheology_B = np.full(90, 1.6e8)  # Standard B value
                self.rheology_n = np.full(90, 3.0)
        
        class MockFriction:
            def __init__(self):
                self.coefficient = np.ones(100) * 1000
                self.p = np.ones(90)
                self.q = np.ones(90)
        
        class MockModel:
            def __init__(self):
                self.mesh = MockMesh()
                self.geometry = MockGeometry()
                self.initialization = MockInitialization()
                self.materials = MockMaterials()
                self.friction = MockFriction()
        
        # Create mock configuration
        from config_synthetic import ModelConfig
        
        # Use the test profile created in previous test
        output_dir = "test_bedrock_profiles"
        config = ModelConfig(profile_id=1, output_dir=output_dir)
        
        # Create Budd implementation
        from Budd_theory import BuddTheoryImplementation
        budd_impl = BuddTheoryImplementation(config)
        print("✓ BuddTheoryImplementation created")
        
        # Create mock model
        md = MockModel()
        print("✓ Mock model created")
        
        # Test effective viscosity calculation
        eta_eff = budd_impl.calculate_effective_viscosity(md)
        print(f"✓ Effective viscosity calculated: {np.min(eta_eff):.2e} to {np.max(eta_eff):.2e} Pa·s")
        
        # Test velocity estimation
        vel_mag = budd_impl.estimate_velocity_magnitude(md)
        print(f"✓ Velocity magnitude estimated: {vel_mag:.1f} m/yr")
        
        # Test sliding coefficient calculation
        sliding_coeff, basal_nodes = budd_impl.calculate_Budd_sliding_coefficient(md, eta_eff)
        print(f"✓ Sliding coefficients calculated for {len(basal_nodes)} basal nodes")
        print(f"  Range: {np.min(sliding_coeff):.0f} to {np.max(sliding_coeff):.0f}")
        
        # Test full application
        md = budd_impl.apply_budd_sliding(md, eta_eff, update=False)
        print("✓ Budd sliding applied successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Budd theory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test the full integration"""
    print("\n=== Testing Full Integration ===")
    
    try:
        from config_synthetic import ModelConfig
        from Budd_theory import BuddTheoryImplementation
        
        # Use test profile
        output_dir = "test_bedrock_profiles"
        config = ModelConfig(profile_id=1, output_dir=output_dir)
        
        # Test unit verification
        config.verify_units()
        print("✓ Unit verification completed")
        
        # Test omega calculation
        omega = config.omega
        print(f"✓ Omega calculated: {omega:.6f} rad/m")
        
        # Test that all required attributes exist
        required_attrs = ['ice_params', 'bedrock_params', 'solver_settings', 
                         'time_settings', 'output_settings']
        
        for attr in required_attrs:
            if hasattr(config, attr):
                print(f"✓ {attr} exists")
            else:
                print(f"✗ {attr} missing")
                return False
        
        print("✓ All integration tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def cleanup_test_files():
    """Clean up test files"""
    print("\n=== Cleaning Up Test Files ===")
    
    import shutil
    test_dirs = ["test_bedrock_profiles"]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"✓ Removed {test_dir}")
    
    print("✓ Cleanup completed")


def main():
    """Run all integration tests"""
    print("Starting Budd Theory Integration Tests\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Bedrock Generation Test", test_bedrock_generation), 
        ("Configuration Creation Test", test_config_creation),
        ("Budd Theory Test", test_budd_theory),
        ("Full Integration Test", test_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Integration is working correctly.")
        print("\nYou can now run:")
        print("  python flowline_synthetic.py --profile 1")
    else:
        print("❌ SOME TESTS FAILED! Check the errors above.")
        print("\nCommon fixes:")
        print("  - Ensure ISSM is properly installed and in Python path")
        print("  - Check that all required dependencies are available")
        print("  - Verify file permissions for writing test files")
    
    # Cleanup
    cleanup_test_files()


if __name__ == "__main__":
    main()