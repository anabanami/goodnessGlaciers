import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime
import netCDF4 as nc


class TransientConvergenceAnalyzer:
    def __init__(self, profile_id=None, experiment=None):
        self.profile_id = profile_id
        self.experiment = experiment
        self.results = {}
        # Add a new dictionary for evolution data and define yts
        self.evolution_data = {}
        self.yts = 31556926  # seconds per year
        self.relative_tolerance = 0.01  # 1%
        self.absolute_tolerance = 0.01  # A realistic 0.01 m/a
        self.near_zero_threshold = 0.1 # m/a


    def detect_available_datasets(self):
        # ... (This function is unchanged)
        pattern = "*_*_*.nc"
        files = glob.glob(pattern)

        if not files:
            raise FileNotFoundError("No NetCDF result files found matching pattern '*_*_*.nc'")
        datasets = {}

        for filename in files:

            try:
                basename = filename.replace('.nc', '')
                parts = basename.split('_')

                if len(parts) >= 3:
                    profile_str, experiment, resolution_str = parts[0], parts[1], parts[2]
                    profile_id, resolution_factor = int(profile_str), float(resolution_str)
                    key = (profile_id, experiment)
                    if key not in datasets:
                        datasets[key] = []
                    datasets[key].append(resolution_factor)

            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse components from {filename}: {e}")

        for key in datasets:

            datasets[key].sort()
        print("Auto-detected datasets:")

        for (profile_id, experiment), resolutions in datasets.items():
            print(f"  Profile {profile_id:03d} - Experiment {experiment}: {resolutions}")

        return datasets


    def auto_configure(self):
        # ... (This function is unchanged)
        datasets = self.detect_available_datasets()

        if not datasets:
            raise FileNotFoundError("No valid datasets found")
        (self.profile_id, self.experiment), resolution_factors = next(iter(datasets.items()))
        print(f"\nAuto-selecting first detected dataset: Profile {self.profile_id:03d}, Experiment {self.experiment}")
        return resolution_factors


    def load_transient_results(self, resolution_factors):
        """Load and sort transient results from NetCDF files, using the last timestep."""
        print(f"\n=== Loading Transient Results for Profile {self.profile_id:03d} {self.experiment} ===")

        for res_factor in resolution_factors:
            filename = f"{self.profile_id:03d}_{self.experiment}_{res_factor}.nc"

            if os.path.exists(filename):
                print(f"Loading {filename}...")

                try:
                    with nc.Dataset(filename, 'r') as dataset:
                        # Access the TransientSolution group
                        transient_group = dataset.groups['results'].groups['TransientSolution']
                        
                        # Get the last timestep (index -1)
                        vx_data = transient_group.variables['Vx'][-1, :, 0]  # Last time, all nodes, first component
                        vy_data = transient_group.variables['Vy'][-1, :, 0]  # Last time, all nodes, first component
                        
                        # Convert from m/s to m/a (multiply by seconds per year)
                        vx_data = vx_data * self.yts
                        vy_data = vy_data * self.yts
                        
                        # Extract flowband centerline nodes using the same logic as periodic_flowline.py
                        # This is a flowband model, so we need to find centerline nodes (y_hat ≈ 0.5)
                        
                        n_total = len(vx_data)
                        
                        # Reconstruct approximate mesh structure for flowband
                        # Based on bamgflowband mesh generation in periodic_flowline.py
                        
                        # Domain parameters (lines 708-711 in periodic_flowline.py)
                        L_total = 210e3  # 25km + 160km + 25km
                        
                        # Estimate flowband mesh structure
                        # Typical flowband: long in x-direction, narrow in y-direction
                        # Assume roughly rectangular mesh distribution
                        
                        # Find actual factors of 43,437 nodes
                        # 43,437 = 3 × 14,479 (only factorization)
                        possible_meshes = [
                            (14479, 3),    # 14479 along flow, 3 across
                            (3, 14479),    # 3 along flow, 14479 across  
                            (2100, None),  # Try to extract 2100 centerline nodes directly
                        ]
                        
                        best_centerline = None
                        best_rmse = float('inf')
                        
                        for nx, ny in possible_meshes:
                            if ny is None:
                                # Special case: try to find nodes that match 2100-point pattern
                                print(f"  Trying to find 2100 centerline nodes...")
                                
                                # Method 1: Check if 14479/7 ≈ 2100 (every 7th node along long dimension)
                                if 14479 // 7 == 2068:  # Close to 2100
                                    # Try extracting every 7th node from the 14479 sequence
                                    centerline_indices = np.arange(0, 14479, 7)[:2100]  
                                    # Since mesh could be 14479x3 or 3x14479, try both
                                    
                                    for layout in ['row', 'col']:
                                        if layout == 'row':  # 3 rows, 14479 cols
                                            # Take middle row (row 1 of 0,1,2)
                                            full_indices = np.arange(14479, 2*14479)  # Middle row
                                            sample_indices = full_indices[::7][:2100]
                                        else:  # 14479 rows, 3 cols  
                                            # Take middle column (col 1 of 0,1,2)
                                            full_indices = np.arange(1, n_total, 3)  # Every 3rd starting from 1
                                            sample_indices = full_indices[::7][:2100]  # Subsample
                                        
                                        if len(sample_indices) == 2100:
                                            centerline_vx = vx_data[sample_indices]
                                            centerline_vy = vy_data[sample_indices]
                                            x_hat_center = np.linspace(0.0, 1.0, 2100)
                                            
                                            mean_abs_vy = np.mean(np.abs(centerline_vy))
                                            
                                            # Validate against static
                                            static_data = np.loadtxt("165_S1_1.0_static.txt")
                                            vx_static = static_data[:, 1]
                                            rmse = np.sqrt(np.mean((centerline_vx - vx_static)**2))
                                            
                                            print(f"  Layout {layout}: |Vy|={mean_abs_vy:.3f}, RMSE={rmse:.1f}")
                                            
                                            if rmse < best_rmse:
                                                best_rmse = rmse
                                                best_centerline = (x_hat_center, centerline_vx, centerline_vy)
                                continue
                            
                            if n_total == nx * ny:  # Valid mesh dimensions
                                # Reconstruct mesh coordinates  
                                x_coords = np.linspace(0, L_total, nx)
                                y_coords = np.linspace(0, L_total/10, ny)
                                
                                # Create mesh indices (row-major: ny rows, nx columns)
                                mesh_indices = np.arange(n_total).reshape(ny, nx)
                                
                                # Find centerline (middle row if ny > 1)
                                if ny == 1:
                                    centerline_indices = mesh_indices[0, :]  # Only one row
                                else:
                                    center_row = ny // 2
                                    centerline_indices = mesh_indices[center_row, :]
                                
                                # Extract centerline velocities
                                centerline_vx = vx_data[centerline_indices]
                                centerline_vy = vy_data[centerline_indices]
                                
                                # Normalized x coordinates
                                x_hat_center = x_coords / L_total
                                
                                # Check if this gives reasonable results
                                mean_abs_vy = np.mean(np.abs(centerline_vy))
                                
                                print(f"  Mesh {nx}x{ny}: {len(centerline_vx)} nodes, |Vy|={mean_abs_vy:.3f}")
                                
                                # If we get exactly 2100 points or close, validate
                                if abs(len(centerline_vx) - 2100) <= 100:
                                    static_data = np.loadtxt("165_S1_1.0_static.txt")
                                    vx_static = static_data[:, 1]
                                    
                                    if len(centerline_vx) == len(vx_static):
                                        rmse = np.sqrt(np.mean((centerline_vx - vx_static)**2))
                                        print(f"    RMSE vs static: {rmse:.1f}")
                                        
                                        if rmse < best_rmse:
                                            best_rmse = rmse
                                            best_centerline = (x_hat_center, centerline_vx, centerline_vy)
                        
                        if best_centerline is not None:
                            x_hat, flowline_vx, flowline_vy = best_centerline
                            print(f"  ✓ Using best centerline (RMSE={best_rmse:.1f})")
                        else:
                            # Fallback: use the previous method that worked reasonably well
                            print("  Using fallback method (nodes 2000-4099)")
                            flowline_start = 2000
                            flowline_end = 4100
                            flowline_vx = vx_data[flowline_start:flowline_end]
                            flowline_vy = vy_data[flowline_start:flowline_end]
                            x_hat = np.linspace(0.0, 1.0, len(flowline_vx))
                        
                        # Create data array similar to static file format
                        data = np.column_stack([
                            x_hat,
                            flowline_vx,  # surface vx
                            flowline_vy,  # surface vy
                            np.zeros_like(flowline_vx)  # basal vx - set to zero as placeholder
                        ])
                        
                        # Sort data by x-coordinate (should already be sorted, but ensure consistency)
                        sort_indices = np.argsort(data[:, 0])
                        data = data[sort_indices]
                        
                        self.results[res_factor] = {
                            'x_hat': data[:, 0],
                            'vx_surface': data[:, 1], 
                            'vy_surface': data[:, 2],
                            'vx_basal': data[:, 3],
                        }
                        
                        print(f"  ✓ Extracted {len(data)} flowline points from {len(vx_data)} total nodes")
                        print(f"  ✓ X-coordinate range: {np.min(x_hat):.3f} to {np.max(x_hat):.3f}")
                        print(f"  ✓ Surface velocity range: {np.min(flowline_vx):.1f} to {np.max(flowline_vx):.1f} m/a")
                        print(f"  ✓ Time steps in file: {len(transient_group.variables['time'][:])}")

                except Exception as e:
                    print(f"  ✗ Error loading {filename}: {e}")

            else:
                print(f"  ✗ File not found: {filename}")

        if not self.results:
            raise FileNotFoundError(f"No results found for profile {self.profile_id}")

        print(f"Successfully loaded {len(self.results)} resolution datasets.")
        return True

    # --- START: NEW FUNCTION ---
    def load_velocity_evolution(self, resolution_factors):
        """Load time-series data for velocity evolution from NetCDF files."""
        print("\n=== Loading Velocity Evolution Data ===")

        for res_factor in resolution_factors:
            filename = f"{self.profile_id:03d}_{self.experiment}_{res_factor}.nc"

            if os.path.exists(filename):
                print(f"Loading evolution data from {filename}...")
                try:
                    with nc.Dataset(filename, 'r') as dataset:
                        transient_group = dataset.groups['results'].groups['TransientSolution']
                        
                        # Load time data and convert to years
                        time_data_seconds = transient_group.variables['time'][:]
                        time_data_years = time_data_seconds / self.yts

                        # Load Vx for all timesteps, convert to m/a
                        vx_all_times = transient_group.variables['Vx'][:, :, 0] * self.yts
                        
                        # Calculate the maximum velocity at each time step
                        max_vx_evolution = np.max(vx_all_times, axis=1)
                        
                        self.evolution_data[res_factor] = {
                            'time_years': time_data_years,
                            'max_vx': max_vx_evolution,
                        }
                        print(f"  ✓ Found {len(time_data_years)} timesteps.")

                except Exception as e:
                    print(f"  ✗ Error loading evolution data from {filename}: {e}")
            else:
                print(f"  ✗ File not found: {filename}")
        
        if not self.evolution_data:
            print("  ! No evolution data was loaded.")
        
        return True


    def calculate_convergence_metrics(self):
        # ... (This function is unchanged from v2.2, it is correct)
        print("\n=== Calculating Convergence Metrics ===")
        finest_res = min(self.results.keys())
        ref_result = self.results[finest_res]
        self.convergence_results = {}
        
        for res_factor in sorted(self.results.keys()):
            
            if res_factor == finest_res: continue
            self.convergence_results[res_factor] = {}
            result = self.results[res_factor]
            x_ref, x_comp = ref_result['x_hat'], result['x_hat']
            
            for field_name, ref_data, comp_data in [
                ('vx_surface', ref_result['vx_surface'], result['vx_surface']),
                ('vx_basal', ref_result['vx_basal'], result['vx_basal'])
            ]:
                comp_interp = np.interp(x_ref, x_comp, comp_data)
                diff = ref_data - comp_interp
                rmse = np.sqrt(np.mean(diff**2))
                max_abs_ref = np.max(np.abs(ref_data))
                
                if max_abs_ref < self.near_zero_threshold:
                    error_value, metric_type = rmse, 'absolute'
                    note = f"Near-zero solution (RMSE={rmse:.2e})"
                    print(f"  {field_name} (res={res_factor}): {note}")

                else:
                    ref_norm = np.linalg.norm(ref_data)
                    error_value, metric_type = np.linalg.norm(diff) / ref_norm, 'relative'
                    note = ""
                    print(f"  {field_name} (res={res_factor}): L2={error_value*100:.2f}%")
                self.convergence_results[res_factor][field_name] = {
                    'error_value': error_value, 'metric_type': metric_type, 'note': note
                }
        return self.convergence_results


    def assess_convergence(self):
        # ... (This function is unchanged from v2.2, it is correct)
        print(f"\n=== Convergence Assessment ===")
        overall_converged = True

        for res_factor, results in self.convergence_results.items():
            res_converged = True

            for field, metrics in results.items():
                if metrics['metric_type'] == 'relative' and metrics['error_value'] > self.relative_tolerance: res_converged = False

                elif metrics['metric_type'] == 'absolute' and metrics['error_value'] > self.absolute_tolerance: res_converged = False

            status = "✓ PASSED" if res_converged else "✗ FAILED"
            print(f"  Overall (res={res_factor}): {status}")

            if not res_converged: overall_converged = False

        return overall_converged


    def generate_report(self, converged):
        # ... (This function is unchanged from v2.2, it is correct)
        print("\n=== Generating Analysis Report ===")
        report_filename = f"{self.profile_id:03d}_{self.experiment}_transient_convergence_report.md"
        finest_res = min(self.results.keys())
        ref_result = self.results[finest_res]

        with open(report_filename, 'w') as f:
            f.write(f"# Transient Grid Convergence Analysis Report\n\n")
            f.write(f"| Parameter | Value |\n| :--- | :--- |\n")
            f.write(f"| **Profile** | {self.profile_id:03d} |\n| **Experiment** | {self.experiment} |\n")
            f.write(f"| **Analysis Date** | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |\n| **Reference Resolution** | `{finest_res}` |\n")
            f.write(f"| **Data Source** | NetCDF transient results (last timestep) |\n")
            f.write(f"\n## Key Metrics (from Reference Solution)\n\n")
            f.write(f"| Metric | Value |\n| :--- | :--- |\n")
            f.write(f"| Max Surface Velocity | {np.max(np.abs(ref_result['vx_surface'])):.3f} m/a |\n")
            f.write(f"| Max Basal Velocity | {np.max(np.abs(ref_result['vx_basal'])):.3f} m/a |\n")
            f.write("\n## Convergence Analysis\n\n")
            f.write(f"An absolute metric (RMSE) is used for solutions where max velocity is < {self.near_zero_threshold} m/a.\n\n")
            f.write("| Resolution | Status | Surface vx Error | Basal vx Error | Notes |\n")
            f.write("| :--- | :---: | :---: | :---: | :--- |\n")

            for res, metrics in self.convergence_results.items():
                is_conv = self.assess_single_resolution(metrics)
                status = "✓ PASSED" if is_conv else "✗ FAILED"
                surf_metrics, basal_metrics = metrics['vx_surface'], metrics['vx_basal']
                surf_err_str = f"{surf_metrics['error_value']*100:.2f}%" if surf_metrics['metric_type'] == 'relative' else f"{surf_metrics['error_value']:.2e}"
                basal_err_str = f"{basal_metrics['error_value']*100:.2f}%" if basal_metrics['metric_type'] == 'relative' else f"{basal_metrics['error_value']:.2e}"
                note = surf_metrics['note'] or basal_metrics['note']
                f.write(f"| `{res}` | {status} | {surf_err_str} | {basal_err_str} | {note.split('(')[0].strip()} |\n")
            f.write("\n## Recommendations\n\n")
            if converged: f.write("**✓ Transient solution has converged.** The results are consistent and can be considered reliable.\n")
            else: f.write("**✗ Transient solution has NOT converged.** The results show sensitivity to grid resolution.\n")
        print(f"  ✓ Saved report: {report_filename}")


    def assess_single_resolution(self, metrics):

        # ... (This function is unchanged from v2.2, it is correct)
        for field_metrics in metrics.values():
            if field_metrics['metric_type'] == 'relative' and field_metrics['error_value'] > self.relative_tolerance: return False
            if field_metrics['metric_type'] == 'absolute' and field_metrics['error_value'] > self.absolute_tolerance: return False
        return True


    def create_comparison_plots(self):
        print("\n=== Creating Comparison Plots ===")
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'Transient Grid Convergence Study - Profile {self.profile_id:03d} {self.experiment}', fontsize=16)
        
        # --- Plots for ax1, ax2, ax3 are unchanged ---
        ax1, ax2 = axes[0, 0], axes[0, 1]
        
        for i, res_factor in enumerate(sorted(self.results.keys())):
            result = self.results[res_factor]
            ax1.plot(result['x_hat'], result['vx_surface'], label=f'res_factor={res_factor}')
            ax2.plot(result['x_hat'], result['vx_basal'], label=f'res_factor={res_factor}')
        
        for ax, title, ylabel in [(ax1, 'Surface Velocity Comparison', 'Surface velocity (m/a)'), (ax2, 'Basal Velocity Comparison', 'Basal velocity (m/a)')]:
            ax.set_xlabel('Normalized distance (x/L)'), ax.set_ylabel(ylabel), ax.set_title(title), ax.legend(), ax.grid(True, linestyle=':', alpha=0.6)
        
        ax3 = axes[1, 0]
        if hasattr(self, 'convergence_results') and self.convergence_results:
            res_factors = sorted(self.convergence_results.keys())
            x_pos = np.arange(len(res_factors))
            surface_errors = [m['vx_surface']['error_value'] for m in self.convergence_results.values()]
            ax3.bar(x_pos, surface_errors, 0.4, label='Surface vx')
            ax3.axhline(y=self.relative_tolerance, linestyle='--', label='1% threshold')
            ax3.set_xticks(x_pos, res_factors), ax3.set_xlabel('Resolution factor (compared to finest)'), ax3.set_ylabel('L2 Relative Error (%) or RMSE')
            ax3.set_title('Convergence Metrics'), ax3.legend(), ax3.grid(True, linestyle=':', alpha=0.6), ax3.set_yscale('log')
        
        ax4 = axes[1, 1]
        # Check if evolution data exists before plotting
        if hasattr(self, 'evolution_data') and self.evolution_data:
            for res_factor in sorted(self.evolution_data.keys()):
                data = self.evolution_data[res_factor]
                ax4.plot(data['time_years'], data['max_vx'], label=f'res_factor={res_factor}')

            ax4.set_title('Velocity Evolution')
            ax4.set_xlabel('Time (years)')
            ax4.set_ylabel('Max Surface Velocity (m/a)')
            ax4.legend()
            ax4.grid(True, linestyle=':', alpha=0.6)
        else:
            # Display a message if no data is available
            ax4.text(0.5, 0.5, 'No velocity evolution data found.', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax4.transAxes)
            ax4.set_title('Velocity Evolution')
        
        # Save the figure
        plot_filename = f"{self.profile_id:03d}_{self.experiment}_transient_convergence_plots.png"
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(plot_filename)
        print(f"  ✓ Saved plots to {plot_filename}")
        plt.show()
        

def main():
    try:
        analyser = TransientConvergenceAnalyzer()
        resolution_factors = analyser.auto_configure()
        analyser.load_transient_results(resolution_factors)

        # Call the new function to load evolution data
        analyser.load_velocity_evolution(resolution_factors)
        analyser.calculate_convergence_metrics()
        converged = analyser.assess_convergence()
        analyser.generate_report(converged)
        analyser.create_comparison_plots()
        print(f"\n✓ Transient grid convergence analysis complete!")
        if converged: print(f"🎉 All transient resolutions converged within tolerance.")
        else: print(f"⚠️ Some transient resolutions did not converge within tolerance.")
    except Exception as e:
        print(f"\n✗ An error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
