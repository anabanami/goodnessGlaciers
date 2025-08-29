#!/usr/bin/env python3
"""
Calculate theoretical node counts for different resolution factors
"""

# Parameters from bedrock_metadata.txt and bedrock_settings.py
bed_wavelength = 6.336e3  # m (6.336 km from bedrock_metadata.txt)
ice_thickness = 1.92e3    # m (from bedrock_settings.py)

# Domain length (approximate from flowline.py)
L = 160e3  # m (target domain length)

print("=== Bedrock and Grid Parameters ===")
print(f"Bed wavelength: {bed_wavelength/1000:.3f} km")
print(f"Ice thickness: {ice_thickness/1000:.3f} km")
print(f"Domain length: {L/1000:.3f} km")
print(f"Wavelength/thickness ratio: {bed_wavelength/ice_thickness:.2f}")

# Calculate refinement factor (from adaptive_bamg function)
if bed_wavelength < 15000:
    refinement_factor = 50
else:
    refinement_factor = 200

print(f"Refinement factor: {refinement_factor}")

# Calculate hmax for different resolution factors
resolution_factors = [0.5, 0.75, 1.0, 1.25]

print("\n=== Theoretical Mesh Spacing (hmax) ===")
for rf in resolution_factors:
    hmax = (bed_wavelength / refinement_factor) * rf
    print(f"Resolution factor {rf}: hmax = {hmax:.2f} m")
    
    # Rough estimate of nodes based on domain length / hmax
    # This is very approximate since bamg uses adaptive meshing
    approx_nodes_1d = L / hmax
    print(f"  -> Approximate 1D nodes: {approx_nodes_1d:.0f}")
    
    # For 2D flowband with vertical layers, multiply by some factor
    # This is very rough since vertical discretization also matters
    print(f"  -> Very rough 2D estimate: {approx_nodes_1d * 10:.0f} - {approx_nodes_1d * 20:.0f} nodes")

print("\n=== Actual Node Counts from Output Files ===")
actual_counts = {
    0.5: 3165,   # minus header line
    0.75: 1583,
    1.0: 1583,
    1.25: 1583
}

for rf, count in actual_counts.items():
    print(f"Resolution factor {rf}: {count} surface nodes")

print("\n=== Analysis ===")
print("The issue is clear: resolutions 0.75, 1.0, and 1.25 all have")
print("exactly the same number of nodes (1583), while 0.5 has ~2x more (3165).")
print("\nThis suggests there might be:")
print("1. A minimum mesh size constraint that's being hit")
print("2. The bamg mesher is reaching some internal limit")
print("3. There's a bug in how resolution_factor is being applied")
print("\nThe hmax values should be different, so if bamg is working correctly,")
print("we should see different node counts for each resolution.")