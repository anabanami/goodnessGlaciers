#!/usr/bin/env python3
"""
Grid Convergence Analysis for Ice Flow Modeling - Diagnostic Solutions
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime

class GridConvergenceAnalyzer:
    def __init__(self, profile_id=None, experiment=None):
        self.profile_id = profile_id
        self.experiment = experiment
        self.results = {}
    
    def detect_available_datasets(self):
        """Automatically detect profile ID, experiment, and resolution factors from static files"""
        pattern = "*_*_*_static.txt"
        files = glob.glob(pattern)
        
        if not files:
            raise FileNotFoundError("No static result files found matching pattern '*_*_*_static.txt'")
        
        datasets = {}  # {(profile_id, experiment): [resolution_factors]}
        
        for filename in files:
            try:
                # Extract components from filename
                # Format: 022_S4_0.75_static.txt -> profile=022, experiment=S4, resolution=0.75
                basename = filename.replace('_static.txt', '')
                parts = basename.split('_')
                
                if len(parts) >= 3:
                    profile_str = parts[0]
                    experiment = parts[1]  
                    resolution_str = parts[2]
                    
                    profile_id = int(profile_str)
                    resolution_factor = float(resolution_str)
                    
                    key = (profile_id, experiment)
                    if key not in datasets:
                        datasets[key] = []
                    datasets[key].append(resolution_factor)
                    
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse components from {filename}: {e}")
        
        # Sort resolution factors for each dataset
        for key in datasets:
            datasets[key].sort()
        
        print(f"Auto-detected datasets:")
        for (profile_id, experiment), resolutions in datasets.items():
            print(f"  Profile {profile_id:03d} - Experiment {experiment}: {resolutions}")
        
        return datasets
    
    def auto_configure(self):
        """Automatically configure analyser with detected datasets"""
        datasets = self.detect_available_datasets()
        
        if len(datasets) == 0:
            raise FileNotFoundError("No valid datasets found")
        elif len(datasets) == 1:
            # Single dataset - use it automatically
            (self.profile_id, self.experiment), resolution_factors = next(iter(datasets.items()))
            print(f"Auto-selected: Profile {self.profile_id:03d}, Experiment {self.experiment}")
            return resolution_factors
        else:
            # Multiple datasets - show options
            print(f"\nMultiple datasets found:")
            dataset_list = list(datasets.items())
            for i, ((profile_id, experiment), resolutions) in enumerate(dataset_list):
                print(f"  [{i}] Profile {profile_id:03d} - Experiment {experiment}: {resolutions}")
            
            # For now, use the first one (could be enhanced with user input)
            (self.profile_id, self.experiment), resolution_factors = dataset_list[0]
            print(f"Auto-selected first dataset: Profile {self.profile_id:03d}, Experiment {self.experiment}")
            return resolution_factors
        
    def load_static_results(self, resolution_factors):
        """Load static diagnostic results using exact same method as debug_sorted_static.py"""
        
        print(f"=== Loading Results for Profile {self.profile_id:03d} {self.experiment} ===")
        
        for res_factor in resolution_factors:
            # Build filename exactly like debug script
            filename = f"{self.profile_id:03d}_{self.experiment}_{res_factor}_static.txt"
            
            if os.path.exists(filename):
                print(f"Loading {filename} for resolution factor {res_factor}")
                
                try:
                    # Load data exactly like debug_sorted_static.py
                    data = np.loadtxt(filename)
                    x_hat = data[:, 0]
                    vx_surface = data[:, 1]
                    vz_surface = data[:, 2] 
                    vx_basal = data[:, 3]
                    
                    # Sort by x_hat in decreasing order (1.0 to 0.0) - same as debug script
                    sort_indices = np.argsort(-x_hat)
                    x_hat_sorted = x_hat[sort_indices]
                    vx_surface_sorted = vx_surface[sort_indices]
                    vz_surface_sorted = vz_surface[sort_indices]
                    vx_basal_sorted = vx_basal[sort_indices]
                    
                    # Check if sorting was needed - same logic as debug script
                    was_sorted = np.array_equal(sort_indices, np.arange(len(x_hat)))
                    if not was_sorted:
                        print(f"    ⚠️  Applied coordinate sorting (data was not monotonic)")
                    else:
                        print(f"    ✅ Data already properly sorted")
                    
                    # Store results
                    self.results[res_factor] = {
                        'filename': filename,
                        'x_hat': x_hat_sorted,
                        'vx_surface': vx_surface_sorted,
                        'vz_surface': vz_surface_sorted,
                        'vx_basal': vx_basal_sorted,
                        'n_points': len(data)
                    }
                    print(f"  ✓ Loaded {len(data)} data points")
                    
                except Exception as e:
                    print(f"  ✗ Error loading {filename}: {e}")
            else:
                print(f"  ✗ File not found: {filename}")
        
        if not self.results:
            raise FileNotFoundError(f"No static results found for profile {self.profile_id} experiment {self.experiment}")
        
        print(f"Successfully loaded {len(self.results)} resolution datasets")
        return True
    
    def validate_data_quality(self):
        """Validate the loaded data quality"""
        
        print(f"\n=== Data Quality Validation ===")
        issues = []
        
        for res_factor in sorted(self.results.keys()):
            result = self.results[res_factor]
            x_data = result['x_hat']
            vx_surf = result['vx_surface'] 
            vx_basal = result['vx_basal']
            
            # Check coordinate range
            x_min, x_max = np.min(x_data), np.max(x_data)
            print(f"  Resolution {res_factor}: x range [{x_min:.6f}, {x_max:.6f}], {len(x_data)} points")
            print(f"  Resolution {res_factor}: max |vx_surface| = {np.max(np.abs(vx_surf)):.2f} m/a, max |vx_basal| = {np.max(np.abs(vx_basal)):.2f} m/a")
            
            # Check for monotonic coordinates (should be decreasing from 1 to 0)
            if not np.all(np.diff(x_data) <= 0):
                issues.append(f"Resolution {res_factor}: Non-monotonic x coordinates")
            
            # Check for NaN/infinite values
            if np.any(~np.isfinite(vx_surf)) or np.any(~np.isfinite(vx_basal)):
                issues.append(f"Resolution {res_factor}: Contains NaN or infinite values")
        
        # Check coordinate overlap
        all_x_min = max(np.min(self.results[rf]['x_hat']) for rf in self.results.keys())
        all_x_max = min(np.max(self.results[rf]['x_hat']) for rf in self.results.keys())
        
        if all_x_max <= all_x_min:
            issues.append("No coordinate overlap between resolutions")
        else:
            overlap_percent = 100 * (all_x_max - all_x_min) / (1.0 - 0.0)
            print(f"  Coordinate overlap: [{all_x_min:.6f}, {all_x_max:.6f}] ({overlap_percent:.1f}% of total range)")
        
        if issues:
            print(f"  ⚠️  Found {len(issues)} data quality issues:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"  ✓ All data quality checks passed")
        
        return len(issues) == 0
    
    def interpolate_to_common_grid(self):
        """Interpolate all solutions to finest resolution grid"""
        
        print(f"\n=== Interpolating to Common Grid ===")
        
        # Find finest resolution (most points) as reference
        finest_res = min(self.results.keys(), key=lambda x: x)  # Finest has smallest res_factor
        ref_result = self.results[finest_res]
        self.x_ref = ref_result['x_hat']
        
        print(f"Using resolution {finest_res} as reference")
        print(f"Common domain: x = [{np.min(self.x_ref):.6f}, {np.max(self.x_ref):.6f}]")
        print(f"Reference grid has {len(self.x_ref)} points in common domain")
        
        # Interpolate other resolutions to reference grid
        for res_factor in self.results.keys():
            result = self.results[res_factor]
            
            if res_factor == finest_res:
                # Reference resolution - use original data
                result['vx_surface_interp'] = result['vx_surface']
                result['vz_surface_interp'] = result['vz_surface']
                result['vx_basal_interp'] = result['vx_basal']
                print(f"  Reference resolution {finest_res}: using original data")
            else:
                # Other resolutions - interpolate to reference grid
                x_orig = result['x_hat']
                
                # Find valid interpolation range
                x_min_orig = np.min(x_orig)
                x_max_orig = np.max(x_orig) 
                mask_valid = (self.x_ref >= x_min_orig) & (self.x_ref <= x_max_orig)
                
                # Initialize with NaN
                vx_surf_interp = np.full(len(self.x_ref), np.nan)
                vz_surf_interp = np.full(len(self.x_ref), np.nan)
                vx_basal_interp = np.full(len(self.x_ref), np.nan)
                
                # Interpolate only valid points
                if np.any(mask_valid):
                    vx_surf_interp[mask_valid] = np.interp(self.x_ref[mask_valid], x_orig, result['vx_surface'])
                    vz_surf_interp[mask_valid] = np.interp(self.x_ref[mask_valid], x_orig, result['vz_surface'])
                    vx_basal_interp[mask_valid] = np.interp(self.x_ref[mask_valid], x_orig, result['vx_basal'])
                
                result['vx_surface_interp'] = vx_surf_interp
                result['vz_surface_interp'] = vz_surf_interp 
                result['vx_basal_interp'] = vx_basal_interp
                
                n_valid = np.sum(mask_valid)
                print(f"  Interpolated res_factor={res_factor}: {n_valid}/{len(self.x_ref)} points valid")
    
    def calculate_convergence_metrics(self, tolerance=0.01):
        """Calculate convergence metrics using same approach as debug script"""
        
        print(f"\n=== Calculating Convergence Metrics ===")
        
        # Find reference resolution (finest)
        finest_res = min(self.results.keys(), key=lambda x: x)
        ref_result = self.results[finest_res]
        
        convergence_results = {}
        
        for res_factor in sorted(self.results.keys()):
            if res_factor == finest_res:
                continue  # Skip reference
                
            result = self.results[res_factor]
            convergence_results[res_factor] = {}
            
            # Calculate errors using direct interpolation like debug script
            # Use reference grid and interpolate comparison data to it
            x_ref = ref_result['x_hat']
            x_comp = result['x_hat']
            
            for field_name, ref_data, comp_data in [
                ('vx_surface', ref_result['vx_surface'], result['vx_surface']),
                ('vz_surface', ref_result['vz_surface'], result['vz_surface']),  
                ('vx_basal', ref_result['vx_basal'], result['vx_basal'])
            ]:
                
                # Interpolate comparison data to reference grid
                # Fix: np.interp requires increasing x coordinates, but our data is sorted decreasing (1.0 to 0.0)
                # Reverse arrays for interpolation, then reverse result back
                comp_interp = np.interp(x_ref[::-1], x_comp[::-1], comp_data[::-1])[::-1]
                
                # Check for near-zero reference values
                ref_norm = np.linalg.norm(ref_data)
                if ref_norm < 1e-10:
                    print(f"  {field_name} (res={res_factor}): Near-zero reference values")
                    convergence_results[res_factor][field_name] = {
                        'l2_error': np.nan, 'max_error': np.nan, 'rmse': 0.0, 'n_valid': len(ref_data)
                    }
                    continue
                
                # Calculate errors - exactly like debug script
                diff = ref_data - comp_interp
                l2_error = np.linalg.norm(diff) / ref_norm
                max_error = np.max(np.abs(diff)) / np.max(np.abs(ref_data))
                rmse = np.sqrt(np.mean(diff**2))
                
                n_valid = len(ref_data)
                
                convergence_results[res_factor][field_name] = {
                    'l2_error': l2_error,
                    'max_error': max_error, 
                    'rmse': rmse,
                    'n_valid': n_valid
                }
                
                print(f"  {field_name} (res={res_factor}):")
                print(f"    L2 relative error: {l2_error:.4f} ({l2_error*100:.2f}%)")
                print(f"    Max relative error: {max_error:.4f} ({max_error*100:.2f}%)")
                print(f"    RMSE: {rmse:.2f} m/a")
                print(f"    Valid points: {n_valid}/{len(ref_data)}")
        
        # Store results
        self.convergence_results = convergence_results
        return convergence_results
    
    def assess_convergence(self, tolerance=0.01):
        """Assess convergence based on tolerance threshold"""
        
        print(f"\n=== Convergence Assessment (tolerance={tolerance*100:.1f}%) ===")
        
        overall_converged = True
        
        for res_factor in sorted(self.convergence_results.keys()):
            results = self.convergence_results[res_factor]
            res_converged = True
            
            for field_name in ['vx_surface', 'vz_surface', 'vx_basal']:
                field_result = results[field_name]
                l2_error = field_result['l2_error']
                
                if np.isnan(l2_error):
                    status = "✗ NO DATA (NaN)"
                    res_converged = False
                elif l2_error <= tolerance:
                    status = "✓ CONVERGED"
                else:
                    status = "✗ NOT CONVERGED" 
                    res_converged = False
                
                print(f"  {field_name} (res={res_factor}): {l2_error*100:.2f}% - {status}")
            
            overall_status = "✓ CONVERGED" if res_converged else "✗ NOT CONVERGED"
            print(f"  Overall (res={res_factor}): {overall_status}")
            
            if not res_converged:
                overall_converged = False
        
        return overall_converged
    
    def create_comparison_plots(self):
        """Create comparison plots using same approach as debug script"""
        
        print(f"\n=== Creating Comparison Plots ===")
        
        # Create 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Grid Convergence Study - Profile {self.profile_id:03d} {self.experiment}', fontsize=14)
        
        # Plot 1: Surface velocity comparison
        ax1 = axes[0, 0]
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        for i, res_factor in enumerate(sorted(self.results.keys())):
            result = self.results[res_factor]
            ax1.plot(result['x_hat'], result['vx_surface'], 
                    label=f'res_factor={res_factor}', color=colors[i % len(colors)], linewidth=2)
        
        ax1.set_xlabel('Normalized distance (x/L)')
        ax1.set_ylabel('Surface velocity (m/a)')
        ax1.set_title('Surface Velocity Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Basal velocity comparison
        ax2 = axes[0, 1]
        for i, res_factor in enumerate(sorted(self.results.keys())):
            result = self.results[res_factor]
            ax2.plot(result['x_hat'], result['vx_basal'],
                    label=f'res_factor={res_factor}', color=colors[i % len(colors)], linewidth=2)
        
        ax2.set_xlabel('Normalized distance (x/L)')
        ax2.set_ylabel('Basal velocity (m/a)')
        ax2.set_title('Basal Velocity Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Convergence metrics
        ax3 = axes[1, 0]
        if hasattr(self, 'convergence_results') and self.convergence_results:
            res_factors = sorted(self.convergence_results.keys())
            surface_errors = [self.convergence_results[rf]['vx_surface']['l2_error']*100 
                             for rf in res_factors if not np.isnan(self.convergence_results[rf]['vx_surface']['l2_error'])]
            basal_errors = [self.convergence_results[rf]['vx_basal']['l2_error']*100
                           for rf in res_factors if not np.isnan(self.convergence_results[rf]['vx_basal']['l2_error'])]
            
            x_pos = np.arange(len(res_factors))
            if surface_errors:
                ax3.bar(x_pos - 0.2, surface_errors, 0.4, label='Surface vx', color='steelblue')
            if basal_errors:
                ax3.bar(x_pos + 0.2, basal_errors, 0.4, label='Basal vx', color='orange')
            
            ax3.axhline(y=1.0, color='red', linestyle='--', label='1% threshold')
            ax3.set_xlabel('Resolution factor')
            ax3.set_ylabel('L2 Relative Error (%)')
            ax3.set_title('Convergence Metrics')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(res_factors)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
        
        # Plot 4: Computational scaling - using actual mesh vertices, not output points
        ax4 = axes[1, 1]
        res_factors = sorted(self.results.keys())
        
        # Manual mesh vertex counts from outlog_S4.md (actual simulation mesh)
        # These should ideally be extracted from simulation data automatically
        mesh_vertices = {
            0.5: 118235,   # from outlog: "Total vertices: 118235"
            0.75: 50246,   # from outlog: "Total vertices: 50246" 
            1.0: 43109,    # from outlog: "Total vertices: 43109"
            1.25: 42554    # from outlog: "Total vertices: 42554"
        }
        
        node_counts = [mesh_vertices.get(rf, self.results[rf]['n_points']) for rf in res_factors]
        
        ax4.plot(res_factors, node_counts, 'bo-', linewidth=2, markersize=8)
        ax4.set_xlabel('Resolution factor')
        ax4.set_ylabel('Mesh vertices (actual)')
        ax4.set_title('Computational Scaling (Mesh Complexity)')
        ax4.grid(True, alpha=0.3)
        
        # Add annotation about data source
        ax4.text(0.02, 0.98, 'Note: Mesh vertex counts\nfrom simulation logs', 
                transform=ax4.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{self.profile_id:03d}_{self.experiment}_convergence_analysis.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"  Saved plot: {plot_filename}")
        plt.show()
        
        return plot_filename
    
    def generate_report(self):
        """Generate convergence analysis report"""
        
        print(f"\n=== Generating Analysis Report ===")
        
        report_filename = f"{self.profile_id:03d}_{self.experiment}_convergence_report.md"
        
        with open(report_filename, 'w') as f:
            f.write(f"# Grid Convergence Analysis Report\n")
            f.write(f"**Profile:** {self.profile_id:03d}  \n")
            f.write(f"**Experiment:** {self.experiment}  \n") 
            f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n")
            
            f.write(f"## Summary\n")
            f.write(f"- Loaded {len(self.results)} resolution datasets\n")
            for res_factor in sorted(self.results.keys()):
                result = self.results[res_factor]
                f.write(f"  - Resolution factor {res_factor}: {result['n_points']} points\n")
            
            if hasattr(self, 'convergence_results'):
                f.write(f"\n## Convergence Analysis\n")
                for res_factor in sorted(self.convergence_results.keys()):
                    results = self.convergence_results[res_factor]
                    f.write(f"### Resolution Factor {res_factor}\n")
                    
                    # Determine overall convergence
                    surface_converged = results['vx_surface']['l2_error'] <= 0.01 if not np.isnan(results['vx_surface']['l2_error']) else False
                    basal_converged = results['vx_basal']['l2_error'] <= 0.01 if not np.isnan(results['vx_basal']['l2_error']) else False
                    overall_converged = surface_converged and basal_converged
                    
                    status = "✓ PASSED" if overall_converged else "✗ FAILED"
                    f.write(f"- Overall convergence: {status}\n")
                    
                    for field in ['vx_surface', 'vx_basal']:
                        l2_err = results[field]['l2_error']
                        max_err = results[field]['max_error']
                        if not np.isnan(l2_err):
                            f.write(f"- {field}: L2={l2_err*100:.2f}%, Max={max_err*100:.2f}%\n")
                        else:
                            f.write(f"- {field}: No valid data (NaN)\n")
                    f.write(f"\n")
            
            f.write(f"## Recommendations\n")
            f.write(f"- **No fully converged solutions found** - consider finer resolution or longer simulation time\n")
        
        print(f"  Saved report: {report_filename}")
        return report_filename

def main():
    """Main analysis function"""
    
    tolerance = 0.01  # 1% tolerance
    
    try:
        # Create analyser with automatic detection
        analyser = GridConvergenceAnalyzer()
        
        # Auto-detect profile, experiment, and resolution factors
        resolution_factors = analyser.auto_configure()
        
        # Load results with exact same method as debug script
        analyser.load_static_results(resolution_factors)
        
        # Validate data quality
        analyser.validate_data_quality()
        
        # Interpolate to common grid
        analyser.interpolate_to_common_grid()
        
        # Calculate convergence metrics
        analyser.calculate_convergence_metrics(tolerance)
        
        # Assess convergence
        converged = analyser.assess_convergence(tolerance)
        
        # Generate report
        analyser.generate_report()
        
        # Create plots
        analyser.create_comparison_plots()
        
        print(f"\n✓ Grid convergence analysis complete!")
        
        if converged:
            print(f"🎉 All resolutions converged within {tolerance*100:.1f}% tolerance")
        else:
            print(f"⚠️  Some resolutions did not converge within {tolerance*100:.1f}% tolerance")
        
    except Exception as e:
        print(f"✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()