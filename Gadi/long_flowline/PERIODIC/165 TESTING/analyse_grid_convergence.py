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
        self.relative_tolerance = 0.01  # 1%
        self.absolute_tolerance = 0.01  # A realistic 0.01 m/a
        self.near_zero_threshold = 0.1 # m/a


    def detect_available_datasets(self):
        # ... (This function is unchanged)
        pattern = "*_*_*_static.txt"
        files = glob.glob(pattern)

        if not files:
            raise FileNotFoundError("No static result files found matching pattern '*_*_*_static.txt'")
        datasets = {}

        for filename in files:

            try:
                basename = filename.replace('_static.txt', '')
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


    def load_static_results(self, resolution_factors):
        """Load and sort static diagnostic results."""
        print(f"\n=== Loading Results for Profile {self.profile_id:03d} {self.experiment} ===")

        for res_factor in resolution_factors:
            filename = f"{self.profile_id:03d}_{self.experiment}_{res_factor}_static.txt"

            if os.path.exists(filename):
                print(f"Loading {filename}...")

                try:
                    data = np.loadtxt(filename)
                    # --- SAFEGUARD: Sort data by x-coordinate after loading ---
                    sort_indices = np.argsort(data[:, 0])
                    data = data[sort_indices]
                    # --- END SAFEGUARD ---
                    self.results[res_factor] = {
                        'x_hat': data[:, 0],
                        'vx_surface': data[:, 1],
                        'vy_surface': data[:, 2],
                        'vx_basal': data[:, 3],
                    }
                    print(f"  ✓ Loaded and sorted {len(data)} data points")

                except Exception as e:
                    print(f"  ✗ Error loading {filename}: {e}")

            else:
                print(f"  ✗ File not found: {filename}")

        if not self.results:
            raise FileNotFoundError(f"No results found for profile {self.profile_id}")

        print(f"Successfully loaded {len(self.results)} resolution datasets.")
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
        report_filename = f"{self.profile_id:03d}_{self.experiment}_convergence_report.md"
        finest_res = min(self.results.keys())
        ref_result = self.results[finest_res]

        with open(report_filename, 'w') as f:
            f.write(f"# Grid Convergence Analysis Report\n\n")
            f.write(f"| Parameter | Value |\n| :--- | :--- |\n")
            f.write(f"| **Profile** | {self.profile_id:03d} |\n| **Experiment** | {self.experiment} |\n")
            f.write(f"| **Analysis Date** | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |\n| **Reference Resolution** | `{finest_res}` |\n")
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
            if converged: f.write("**✓ Solution has converged.** The results are consistent and can be considered reliable.\n")
            else: f.write("**✗ Solution has NOT converged.** The results show sensitivity to grid resolution.\n")
        print(f"  ✓ Saved report: {report_filename}")


    def assess_single_resolution(self, metrics):

        # ... (This function is unchanged from v2.2, it is correct)
        for field_metrics in metrics.values():
            if field_metrics['metric_type'] == 'relative' and field_metrics['error_value'] > self.relative_tolerance: return False
            if field_metrics['metric_type'] == 'absolute' and field_metrics['error_value'] > self.absolute_tolerance: return False
        return True


    def create_comparison_plots(self):

        # ... (This function is unchanged from v2.2, it is correct)
        print("\n=== Creating Comparison Plots ===")
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'Grid Convergence Study - Profile {self.profile_id:03d} {self.experiment}', fontsize=16)
        ax1, ax2 = axes[0, 0], axes[0, 1]
        
        for i, res_factor in enumerate(sorted(self.results.keys())):
            result = self.results[res_factor]
            ax1.plot(result['x_hat'], result['vx_surface'], label=f'res={res_factor}')
            ax2.plot(result['x_hat'], result['vx_basal'], label=f'res={res_factor}')
        
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
        res_factors, mesh_vertices = sorted(self.results.keys()), {0.75: 51103, 0.875: 43479, 1.0: 43437, 1.125: 44039}
        node_counts = [mesh_vertices.get(rf, 0) for rf in res_factors]
        ax4.plot(res_factors, node_counts, 'o-', markersize=8)
        ax4.set_xlabel('Resolution factor'), ax4.set_ylabel('Mesh vertices'), ax4.set_title('Computational Scaling (Mesh Complexity)'), ax4.grid(True, linestyle=':', alpha=0.6)
        ax4.text(0.02, 0.98, 'Note: Mesh vertex counts\nmanually sourced from logs', transform=ax4.transAxes, fontsize=9, va='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))
        plt.tight_layout(rect=[0, 0, 1, 0.96]), plt.savefig(f"{self.profile_id:03d}_{self.experiment}_convergence_analysis.png", dpi=150)
        print(f"  ✓ Saved plot: {self.profile_id:03d}_{self.experiment}_convergence_analysis.png"), plt.show()

def main():
    try:
        analyser = GridConvergenceAnalyzer()
        resolution_factors = analyser.auto_configure()
        analyser.load_static_results(resolution_factors)
        analyser.calculate_convergence_metrics()
        converged = analyser.assess_convergence()
        analyser.generate_report(converged)
        analyser.create_comparison_plots()
        print(f"\n✓ Grid convergence analysis complete!")
        if converged: print(f"🎉 All resolutions converged within tolerance.")
        else: print(f"⚠️ Some resolutions did not converge within tolerance.")
    except Exception as e:
        print(f"\n✗ An error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()