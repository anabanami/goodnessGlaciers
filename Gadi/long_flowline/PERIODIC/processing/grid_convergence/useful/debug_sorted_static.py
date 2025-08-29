#!/usr/bin/env python3
"""
Debug the sorted static data to verify sorting worked
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def debug_sorted_static():
    """Debug sorted static diagnostic files"""
    
    # Try to find static files in current directory or subdirectories
    import glob
    
    # Search for static files
    static_files = glob.glob("*_static.txt") + glob.glob("*/static/*_static.txt") + glob.glob("*/*_static.txt")
    
    files = []
    for filepath in static_files:
        # Extract resolution factor from filename
        basename = os.path.basename(filepath)
        if "022_S4_" in basename:  # Focus on profile 022 experiment S4
            try:
                # Extract resolution from filename like "022_S4_0.5_static.txt"
                parts = basename.replace("_static.txt", "").split("_")
                if len(parts) >= 3:
                    res_factor = float(parts[2])
                    files.append((filepath, res_factor))
            except (ValueError, IndexError):
                continue
    
    # Sort by resolution factor
    files.sort(key=lambda x: x[1])
    
    if not files:
        print("No static files found. Looking for pattern: '*022_S4_*_static.txt'")
        print("Available files:", glob.glob("*.txt"))
        return
    
    results = {}
    
    for filename, res_factor in files:
        print(f"\n=== Loading and sorting {filename} ===")
        
        data = np.loadtxt(filename)
        x_hat = data[:, 0]
        vx_surface = data[:, 1]
        
        # Sort by x_hat in decreasing order (1.0 to 0.0)
        sort_indices = np.argsort(-x_hat)
        x_hat_sorted = x_hat[sort_indices]
        vx_surface_sorted = vx_surface[sort_indices]
        
        # Check if sorting was needed
        was_sorted = np.array_equal(sort_indices, np.arange(len(x_hat)))
        print(f"Was already sorted: {was_sorted}")
        
        if not was_sorted:
            print(f"Applied sorting to {len(x_hat)} points")
            # Show the problematic region
            non_sorted_idx = np.where(sort_indices != np.arange(len(x_hat)))[0][:5]
            print(f"First 5 indices that needed reordering: {non_sorted_idx}")
        
        # Check final sorting
        is_monotonic = np.all(np.diff(x_hat_sorted) <= 0)
        print(f"After sorting - monotonic decreasing: {is_monotonic}")
        print(f"X range: [{np.min(x_hat_sorted):.6f}, {np.max(x_hat_sorted):.6f}]")
        
        results[res_factor] = {
            'x_hat': x_hat_sorted,
            'vx_surface': vx_surface_sorted
        }
    
    # Now test convergence between sorted data
    print(f"\n=== Quick Convergence Test on Sorted Data ===")
    
    # Use 0.5 as reference
    ref_result = results[0.5]
    x_ref = ref_result['x_hat']
    vx_ref = ref_result['vx_surface']
    
    for res_factor in [0.75, 1.0]:
        result = results[res_factor]
        
        # Interpolate to common grid
        vx_interp = np.interp(x_ref, result['x_hat'], result['vx_surface'])
        
        # Calculate L2 error
        diff = vx_ref - vx_interp
        l2_error = np.linalg.norm(diff) / np.linalg.norm(vx_ref)
        
        print(f"Surface velocity L2 error (0.5 vs {res_factor}): {l2_error*100:.2f}%")
        print(f"Max difference: {np.max(np.abs(diff)):.2f} m/a")
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for res_factor in sorted(results.keys()):
        result = results[res_factor]
        ax.plot(result['x_hat'], result['vx_surface'], 
               label=f'res={res_factor}', linewidth=2)
    
    ax.set_xlabel('Normalized distance (x/L)')
    ax.set_ylabel('Surface velocity (m/a)')
    ax.set_title('Static Diagnostic Velocity Comparison (After Sorting)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig('debug_sorted_static_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: debug_sorted_static_comparison.png")
    plt.show()

if __name__ == "__main__":
    debug_sorted_static()