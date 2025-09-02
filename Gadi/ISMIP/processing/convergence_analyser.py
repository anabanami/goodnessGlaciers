#!/usr/bin/env python3
"""
Grid Convergence Study Analysis Script for ISSM Models

Takes multiple NetCDF files from transient simulations at different resolutions,
reconstructs the mesh for each, and performs a convergence analysis.

The script assumes a reference simulation with resolution_factor=1.0 exists
among the provided files.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
import re
import sys
import netCDF4 as nc

# Add ISSM/pyISSM to path
sys.path.append('/home/ana/pyISSM/src')
from model import model
from squaremesh import squaremesh
from parameterize import parameterize

# Define constant for time conversion
SECONDS_PER_YEAR = 31556926.0

def reconstruct_mesh(filename, resolution_factor):
    """
    Reconstructs the 3D model and mesh based on filename conventions.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    param_filename = base.split('_')[0] + ".py"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    param_file_path = os.path.join(parent_dir, param_filename)
    
    if not os.path.exists(param_file_path):
        raise FileNotFoundError(f"Parameter file not found: '{param_file_path}'")

    md = model()
    x_max, y_max = 100000, 100000
    x_nodes = int(30 * resolution_factor)
    y_nodes = int(30 * resolution_factor)
    md = squaremesh(md, x_max, y_max, x_nodes, y_nodes)
    md = parameterize(md, param_file_path)
    md = md.extrude(5, 1)
    return md

class ConvergenceAnalyzer:
    """Analyzes grid convergence for transient ISSM ice flow simulations."""

    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.results = {}
        self.config = {}
        self.ref_resolution = 1.0  # As requested, use res=1.0 as the standard

    def run(self):
        """Main workflow to load, process, and analyze results."""
        if not self._load_results():
            return
        
        self._interpolate_to_common_grid()
        self._calculate_convergence_metrics()
        self._create_comparison_plots()
        self._generate_report()
        print("\nAnalysis complete.")

    def _load_results(self):
        """Detects, parses, and loads results from NetCDF files."""
        if not self.file_paths:
            print("Error: No files provided for analysis.")
            return False

        print(f"Found {len(self.file_paths)} result files to process...")

        file_regex = re.compile(r"([a-zA-Z0-9_]+?)_([a-zA-Z0-9]+)_(\d+\.?\d*)-Transient\.nc")

        for file_path in self.file_paths:
            filename = os.path.basename(file_path)
            match = file_regex.match(filename)
            if not match:
                print(f"  - Skipping '{filename}': does not match expected name format.")
                continue

            param_file, scenario, res_factor_str = match.groups()
            res_factor = float(res_factor_str)

            if not self.config:
                self.config['ParamFile'] = param_file
                self.config['Scenario'] = scenario
                print(f"Detected configuration: {self.config['ParamFile']}, Scenario: {self.config['Scenario']}")

            print(f"  - Loading {filename} (resolution factor: {res_factor})")
            
            try:
                # --- THIS IS THE CRITICAL FIX: Reconstruct mesh for each file ---
                md = reconstruct_mesh(file_path, res_factor)
                
                with nc.Dataset(file_path, 'r') as ds:
                    tsol = ds['results']['TransientSolution']
                    n_steps = len(tsol.variables['time'][:])
                    last_step_idx = n_steps - 1
                    
                    surface_indices = np.where(md.mesh.vertexonsurface)[0]
                    basal_indices = np.where(md.mesh.vertexonbase)[0]
                    
                    vx_full = np.squeeze(tsol.variables['Vx'][last_step_idx, :])
                    vel_series = tsol.variables['Vel'][:]
                    
                    self.results[res_factor] = {
                        'x_surf': md.mesh.x[surface_indices],
                        'vx_surf': vx_full[surface_indices],
                        'x_base': md.mesh.x[basal_indices],
                        'vx_base': vx_full[basal_indices],
                        'times': tsol.variables['time'][:] / SECONDS_PER_YEAR,
                        'max_vel_series': np.max(vel_series, axis=1),
                        'n_nodes': md.mesh.numberofvertices
                    }
            except Exception as e:
                print(f"    Error processing file {filename}: {e}")

        if self.ref_resolution not in self.results:
            print(f"\nError: Reference solution with resolution factor {self.ref_resolution} not found. Aborting.")
            return False
            
        print(f"\nSuccessfully loaded {len(self.results)} datasets.")
        return True

    def _interpolate_to_common_grid(self):
        """Interpolates all results onto the grid of the reference resolution."""
        print(f"\nInterpolating results onto reference grid (res={self.ref_resolution})...")
        ref_result = self.results[self.ref_resolution]
        ref_x_surf = ref_result['x_surf']
        ref_x_base = ref_result['x_base']

        for res_factor, data in self.results.items():
            if res_factor == self.ref_resolution:
                data['vx_surf_interp'] = data['vx_surf']
                data['vx_base_interp'] = data['vx_base']
                continue
            
            # Sort data before 1D interpolation
            sort_surf = np.argsort(data['x_surf'])
            sort_base = np.argsort(data['x_base'])
            
            data['vx_surf_interp'] = np.interp(ref_x_surf, data['x_surf'][sort_surf], data['vx_surf'][sort_surf])
            data['vx_base_interp'] = np.interp(ref_x_base, data['x_base'][sort_base], data['vx_base'][sort_base])

    def _calculate_convergence_metrics(self):
        """Calculate L2 errors for surface and basal velocities."""
        print(f"Calculating Convergence Metrics (vs. res={self.ref_resolution})...")
        ref_result = self.results[self.ref_resolution]
        
        for res_factor in sorted(self.results.keys()):
            if res_factor == self.ref_resolution:
                continue
            
            metrics = {}
            for var in ['vx_surf', 'vx_base']:
                ref_data = ref_result[f'{var}_interp']
                comp_data = self.results[res_factor][f'{var}_interp']
                
                ref_norm = np.linalg.norm(ref_data)
                if ref_norm < 1e-10:
                    l2_error = np.nan
                else:
                    l2_error = np.linalg.norm(ref_data - comp_data) / ref_norm
                metrics[var] = {'l2_relative_error': l2_error * 100} # Store as %
            self.results[res_factor]['metrics'] = metrics
            print(f"  res={res_factor}: Surface Vx L2 Error={metrics['vx_surf']['l2_relative_error']:.2f}%, "
                  f"Basal Vx L2 Error={metrics['vx_base']['l2_relative_error']:.2f}%")

    def _create_comparison_plots(self):
        """Create the 2x2 summary plot."""
        print("Creating summary plot...")
        fig, axs = plt.subplots(2, 2, figsize=(18, 14))
        title = f"Grid Convergence: {self.config['ParamFile']} - {self.config['Scenario']}"
        fig.suptitle(title, fontsize=20)
        
        # Plot 1,1: Surface Velocity
        for res, data in sorted(self.results.items()):
            axs[0, 0].plot(data['x_surf'] / 1000, data['vx_surf'], label=f'Res = {res}')
        axs[0, 0].set(title='Final Surface Velocity Comparison', xlabel='Distance (km)', ylabel='Velocity (m/yr)')
        axs[0, 0].legend(); axs[0, 0].grid(True, linestyle=':')

        # Plot 1,2: Basal Velocity
        for res, data in sorted(self.results.items()):
            axs[0, 1].plot(data['x_base'] / 1000, data['vx_base'], label=f'Res = {res}')
        axs[0, 1].set(title='Final Basal Velocity Comparison', xlabel='Distance (km)', ylabel='Velocity (m/yr)')
        axs[0, 1].legend(); axs[0, 1].grid(True, linestyle=':')
        
        # Plot 2,1: L2 Error
        errors = {r: d['metrics'] for r, d in self.results.items() if r != self.ref_resolution}
        res_labels = [str(r) for r in sorted(errors.keys())]
        surf_errors = [errors[r]['vx_surf']['l2_relative_error'] for r in sorted(errors.keys())]
        base_errors = [errors[r]['vx_base']['l2_relative_error'] for r in sorted(errors.keys())]
        x_pos = np.arange(len(res_labels))
        width = 0.35
        axs[1, 0].bar(x_pos - width/2, surf_errors, width, label='Surface Vx')
        axs[1, 0].bar(x_pos + width/2, base_errors, width, label='Basal Vx')
        axs[1, 0].set(title=f'Relative L2 Error (vs. Res={self.ref_resolution})', xlabel='Resolution Factor', ylabel='L2 Error (%)')
        axs[1, 0].set_xticks(x_pos); axs[1, 0].set_xticklabels(res_labels)
        axs[1, 0].axhline(1.0, color='red', linestyle='--', label='1% Threshold')
        axs[1, 0].legend()
        
        # Plot 2,2: Velocity Convergence
        for res, data in sorted(self.results.items()):
            axs[1, 1].plot(data['times'], data['max_vel_series'], label=f'Res = {res}')
        axs[1, 1].set(title='Maximum Velocity Convergence Over Time', xlabel='Time (years)', ylabel='Max Velocity (m/yr)')
        axs[1, 1].legend(); axs[1, 1].grid(True, linestyle=':')

        filename = f"{self.config['ParamFile']}_{self.config['Scenario']}_convergence_summary.png"
        plt.savefig(filename, dpi=200)
        print(f"  Saved plot: {filename}")
        plt.tight_layout()
        plt.show()

    def _generate_report(self, tolerance=1.0):
        """Generate a markdown convergence analysis report."""
        print("Generating analysis report...")
        report_lines = [
            f"# Grid Convergence Study Report",
            f"**Setup:** {self.config['ParamFile']} | **Scenario:** {self.config['Scenario']}",
            f"**Reference Resolution Factor:** {self.ref_resolution}\n"
        ]
        report_lines.append("| Resolution Factor | Surface Vx L2 Error | Basal Vx L2 Error | Overall Status |")
        report_lines.append("|:---:|:---:|:---:|:---:|")

        optimal_res = None
        for res in sorted(self.results.keys()):
            if res == self.ref_resolution: continue
            metrics = self.results[res]['metrics']
            err_vx_s = metrics['vx_surf']['l2_relative_error']
            err_vx_b = metrics['vx_base']['l2_relative_error']
            is_converged = (err_vx_s < tolerance) and (err_vx_b < tolerance)
            status = "✓ CONVERGED" if is_converged else "✗ NOT CONVERGED"
            if is_converged and optimal_res is None: optimal_res = res
            report_lines.append(f"| {res} | {err_vx_s:.2f}% | {err_vx_b:.2f}% | **{status}** |")

        report_lines.append("\n## Recommendation")
        if optimal_res is not None:
            recommendation = (f"The solution is converged within the {tolerance}% tolerance at a "
                              f"**resolution factor of {optimal_res}**.\nThis is the most "
                              f"computationally efficient setting that meets the criteria.")
        else:
            recommendation = (f"No tested resolution met the {tolerance}% tolerance criteria.\n"
                              f"Consider running a finer simulation to act as a more accurate reference.")
        report_lines.append(recommendation)
        
        report_filename = f"{self.config['ParamFile']}_{self.config['Scenario']}_convergence_report.md"
        with open(report_filename, 'w') as f:
            f.write('\n'.join(report_lines))
        print(f"  Saved report: {report_filename}\n")
        print("--- REPORT PREVIEW ---")
        print('\n'.join(report_lines))
        print("----------------------")

def main():
    """Main function to parse arguments and run the convergence analysis."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('files', nargs='+', help='List of NetCDF files to analyze (e.g., IsmipF_S1_*-Transient.nc).')
    args = parser.parse_args()

    files_to_process = []
    for pattern in args.files:
        files_to_process.extend(glob.glob(pattern))
    
    if len(files_to_process) < 2:
        print("Error: At least two NetCDF files are required for a comparison.", file=sys.stderr)
        return 1
    
    analyzer = ConvergenceAnalyzer(files_to_process)
    analyzer.run()
    return 0

if __name__ == "__main__":
    sys.exit(main())