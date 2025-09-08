#!/usr/bin/env python3
"""
Grid Convergence Study Analysis Script for ISSM Models

Takes multiple NetCDF files from transient simulations at different resolutions,
reconstructs the mesh for each, and performs a convergence analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.cm as cm
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


def reconstruct_mesh(filename, h_resolution_factor, v_resolution_factor):
    """
    Reconstructs the 3D model and mesh based on the filename conventions.
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
    x_nodes = int(30 * h_resolution_factor)
    y_nodes = int(30 * h_resolution_factor)

    base_vertical_layers = 5
    num_layers = int(base_vertical_layers * v_resolution_factor)

    md = squaremesh(md, x_max, y_max, x_nodes, y_nodes)
    md = parameterize(md, param_file_path)
    md = md.extrude(num_layers, 1)
    return md

class ConvergenceAnalyzer:
    """Analyzes grid convergence for transient ISSM ice flow simulations."""

    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.results = {}
        self.config = {}
        self.ref_resolution = (2, 2) # Example: ref is h_res=2, v_res=2


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
        print(f"Found {len(self.file_paths)} result files to process...")
        file_regex = re.compile(r"([a-zA-Z0-9_]+?)_([a-zA-Z0-9]+)_(\d+\.?\d*)_(\d+\.?\d*)-Transient\.nc")
        
        for file_path in self.file_paths:
            filename = os.path.basename(file_path)
            match = file_regex.match(filename)
            
            if not match:
                print(f"  - Skipping '{filename}': does not match name format.")
                continue
            
            param_file, scenario, h_resolution_factor_str, v_resolution_factor_str = match.groups()
            h_res_factor = float(h_resolution_factor_str)
            v_res_factor = float(v_resolution_factor_str)

            res_key = (h_res_factor, v_res_factor)

            if not self.config:
                self.config['ParamFile'], self.config['Scenario'] = param_file, scenario
                print(f"Detected configuration: {param_file}, Scenario: {scenario}")
            print(f"  - Loading {filename} (h_res={h_res_factor}, v_res={v_res_factor})")
            
            try:
                md = reconstruct_mesh(file_path, h_res_factor, v_res_factor)
                with nc.Dataset(file_path, 'r') as ds:
                    tsol = ds['results']['TransientSolution']
                    last_step_idx = len(tsol.variables['time'][:]) - 1
                    
                    surface_indices = np.where(md.mesh.vertexonsurface)[0]
                    basal_indices = np.where(md.mesh.vertexonbase)[0]
                    
                    vel_full = np.squeeze(tsol.variables['Vel'][last_step_idx, :])
                    vel_series = tsol.variables['Vel'][:]
                    
                    # --- CENTERLINE EXTRACTION ---
                    y_centre = 50000  # Define the geometric centre of the domain

                    # 1. Get all unique y-coordinates from the surface mesh nodes
                    unique_y_coords = np.unique(md.mesh.y[surface_indices])
                    
                    # 2. Find which of these unique y-coordinates is closest to the centre
                    closest_y_to_centre = unique_y_coords[np.argmin(np.abs(unique_y_coords - y_centre))]
                    print(f"    Found mesh centreline at y={closest_y_to_centre:.1f}m")

                    # 3. Select all nodes that lie on this identified centreline
                    # Use a very small tolerance for float comparison
                    surface_centreline_indices = surface_indices[
                        np.abs(md.mesh.y[surface_indices] - closest_y_to_centre) < 1e-6
                    ]
                    basal_centreline_indices = basal_indices[
                        np.abs(md.mesh.y[basal_indices] - closest_y_to_centre) < 1e-6
                    ]
                    # --- --- ---

                    if len(surface_centreline_indices) == 0:
                        print(f"    FATAL: Could not find any centreline nodes for res={h_res_factor}. Check mesh generation.")
                        continue
                    
                    print(f"    Found {len(surface_centreline_indices)} surface nodes and {len(basal_centreline_indices)} basal nodes along centreline.")

                    self.results[res_key] = {
                        'x_surf': md.mesh.x[surface_centreline_indices],
                        'vel_surf': vel_full[surface_centreline_indices],
                        'x_base': md.mesh.x[basal_centreline_indices],
                        'vel_base': vel_full[basal_centreline_indices],
                        'times': tsol.variables['time'][:],
                        'max_vel_series': np.max(vel_series, axis=1)
                    }
                    
            except Exception as e:
                print(f"    Error processing file {filename}: {e}")
                
        if self.ref_resolution not in self.results:
            print(f"\nError: Reference solution (res={self.ref_resolution}) not found. Aborting.")
            return False
        print(f"\nSuccessfully loaded {len(self.results)} datasets.")
        return True

    def _interpolate_to_common_grid(self):
        """Interpolates all results onto the grid of the reference resolution."""
        print(f"\nInterpolating results onto reference grid (res={self.ref_resolution})...")
        ref = self.results[self.ref_resolution]
        # Ensure reference grid is sorted before use
        ref_sort_idx = np.argsort(ref['x_surf'])
        ref_x_sorted = ref['x_surf'][ref_sort_idx]

        for res, data in self.results.items():
            # If data is empty for this resolution (due to centreline failure), skip it
            if len(data['x_surf']) == 0:
                print(f"  Skipping interpolation for res={res} as no data was loaded.")
                data['vel_surf_interp'] = []
                data['vel_base_interp'] = []
                continue

            # Sort data before 1D interpolation
            sort_surf = np.argsort(data['x_surf'])
            sort_base = np.argsort(data['x_base'])
            
            # Interpolate onto the sorted reference grid
            data['vel_surf_interp'] = np.interp(ref_x_sorted, data['x_surf'][sort_surf], data['vel_surf'][sort_surf])
            data['vel_base_interp'] = np.interp(ref_x_sorted, data['x_base'][sort_base], data['vel_base'][sort_base])
            
            # Store the common sorted reference grid for L2 calculation
            if res == self.ref_resolution:
                data['vel_surf_interp'] = data['vel_surf'][ref_sort_idx]
                data['vel_base_interp'] = data['vel_base'][ref_sort_idx]

    def _calculate_convergence_metrics(self):
        """Calculates L2 errors, intelligently handling near-zero fields."""
        print(f"Calculating Convergence Metrics (vs. res={self.ref_resolution})...")
        ref = self.results[self.ref_resolution]
        
        for res in sorted(self.results.keys()):
            if res == self.ref_resolution: continue
            
            metrics, data = {}, self.results[res]

            # Skip if data is empty for this resolution
            if 'vel_surf_interp' not in data or len(data['vel_surf_interp']) == 0:
                print(f"  Skipping metrics for res={res} as no data was interpolated.")
                continue

            for var in ['vel_surf', 'vel_base']:
                ref_data, comp_data = ref[f'{var}_interp'], data[f'{var}_interp']
                ref_norm = np.linalg.norm(ref_data)
                diff_norm = np.linalg.norm(ref_data - comp_data)
                
                if ref_norm < 0.1:
                    l2_error, err_type = diff_norm, 'absolute'
                else:
                    l2_error, err_type = (diff_norm / ref_norm) * 100, 'relative'
                metrics[var] = {'l2_error': l2_error, 'type': err_type}

            self.results[res]['metrics'] = metrics
            print(f"  res={res}: Surface Vel L2 ({metrics['vel_surf']['type']})={metrics['vel_surf']['l2_error']:.3f}, "
                  f"Basal Vel L2 ({metrics['vel_base']['type']})={metrics['vel_base']['l2_error']:.3f}")

     
    def _create_comparison_plots(self):
        """Create the 2x2 summary plot."""
        print("Creating summary plot...")

        # Generate a list of unique colors for the plots ---
        num_plots = len(self.results)
        # Options: 'tab20', 'viridis', 'plasma', 'inferno'
        # 'tab20b' is designed for this and extends the style you like.
        colors = cm.tab20b(np.linspace(0.5 / 20, 1- (0.5 / 20), 20))
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Grid Convergence: {self.config['ParamFile']} - {self.config['Scenario']}", fontsize=15)
        
        # Use enumerate to access colors by index ---
        for i, (res, data) in enumerate(sorted(self.results.items())):
            if len(data['x_surf']) == 0: continue # Don't plot if no data

            sort_surf = np.argsort(data['x_surf'])
            axs[0, 0].plot(data['x_surf'][sort_surf]/1000, data['vel_surf'][sort_surf], label=f'{res[0]}x{res[1]}', color=colors[i])
            
            if len(data['x_base']) > 0:
                sort_base = np.argsort(data['x_base'])
                axs[0, 1].plot(data['x_base'][sort_base]/1000, data['vel_base'][sort_base], label=f'{res[0]}x{res[1]}', color=colors[i])

            axs[1, 1].plot(data['times'], data['max_vel_series'], label=f'{res[0]}x{res[1]}', color=colors[i])

        axs[0, 0].set(title='Final Surface Velocity', xlabel='Distance (km)', ylabel='Vel Mag (m/yr)')
        axs[0, 0].legend(ncol=2)
        axs[0, 0].grid(True, ls=':')
        
        axs[0, 1].set(title='Final Basal Velocity', xlabel='Distance (km)', ylabel='Vel Mag (m/yr)')
        axs[0, 1].legend(ncol=2)
        axs[0, 1].grid(True, ls=':')
        
        axs[1, 1].set(title='Maximum Velocity Convergence', xlabel='Time (years)', ylabel='Max Vel (m/yr)')
        axs[1, 1].legend(ncol=2)
        axs[1, 1].grid(True, ls=':')
        
        ax3 = axs[1, 0]
        # Filter out resolutions that might have failed to load
        errors = {r: d['metrics'] for r, d in self.results.items() if r != self.ref_resolution and 'metrics' in d}
        if errors:
            res_labels = [f'{r[0]}x{r[1]}' for r in sorted(errors.keys())]
            x_pos = np.arange(len(res_labels))
            width = 0.35
            
            for i, res in enumerate(sorted(errors.keys())):
                surf_err = errors[res]['vel_surf']
                base_err = errors[res]['vel_base']
                
                ax3.bar(x_pos[i] - width/2, surf_err['l2_error'], width, color='C0')
                ax3.bar(x_pos[i] + width/2, base_err['l2_error'], width, color='C1')

            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(res_labels, rotation=45, ha='right', fontsize=9)
        
        ax3.set(title=f'L2 Error (vs. Res={self.ref_resolution})', xlabel='Resolution Factor (h x v)', ylabel='Error (Relative % or Absolute m/a)')
        ax3.axhline(1.0, color='red', linestyle='--', label='1% Rel. Threshold')
        
        legend_elements = [
            Patch(facecolor='C0', label='Surface Vel L2'),
            Patch(facecolor='C1', label='Basal Vel L2'),
            ax3.lines[0]
        ]
        ax3.legend(handles=legend_elements)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        filename = f"{self.config['ParamFile']}_{self.config['Scenario']}_convergence_summary.png"
        plt.savefig(filename, dpi=200); print(f"  Saved plot: {filename}")
        plt.show()


    def _generate_report(self, tolerance=1.0):
        """Generate a markdown convergence analysis report."""
        print("Generating analysis report...")
        report = [f"# Grid Convergence Study Report", f"**Setup:** {self.config['ParamFile']} | **Scenario:** {self.config['Scenario']}", f"**Reference Resolution Factor:** {self.ref_resolution}\n"]
        report.append("| Resolution Factor | Surface Vel L2 Error | Basal Vel L2 Error | Overall Status |")
        report.append("|:---:|:---:|:---:|:---:|")

        for res in sorted(self.results.keys()):
            if res == self.ref_resolution or 'metrics' not in self.results[res]: continue
            metrics = self.results[res]['metrics']
            surf, base = metrics['vel_surf'], metrics['vel_base']
            
            surf_str = f"{surf['l2_error']:.2f}%" if surf['type'] == 'relative' else f"{surf['l2_error']:.2e} m/a"
            base_str = f"{base['l2_error']:.2f}%" if base['type'] == 'relative' else f"{base['l2_error']:.2e} m/a"
            
            is_converged = (surf['type'] == 'relative' and surf['l2_error'] < tolerance)
            status = "✓ CONVERGED" if is_converged else "✗ NOT CONVERGED"
            report.append(f"| {res} | {surf_str} | {base_str} | **{status}** |")

        report.append("\n*Absolute error (m/a) is reported for fields where the reference solution norm is near zero.*")
        
        report_filename = f"{self.config['ParamFile']}_{self.config['Scenario']}_convergence_report.md"
        with open(report_filename, 'w') as f: f.write('\n'.join(report))
        print(f"  Saved report: {report_filename}\n"); print("--- REPORT PREVIEW ---"); print('\n'.join(report)); print("----------------------")

def main():
    """Main function to parse arguments and run the convergence analysis."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('files', nargs='+', help='List of NetCDF files (e.g., IsmipF_S1_*-Transient.nc).')
    args = parser.parse_args()
    analyzer = ConvergenceAnalyzer(args.files)
    analyzer.run()
    return 0

if __name__ == "__main__":
    sys.exit(main())