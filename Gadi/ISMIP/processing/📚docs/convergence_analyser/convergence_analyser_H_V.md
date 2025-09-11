1. Overall Purpose

This script automates the process of analyzing the results from multiple transient ice flow simulations that were run at different grid resolutions. Its primary goal is to determine if the model's solution (specifically ice velocity) "converges," meaning the results become stable and stop changing significantly as the mesh resolution increases. This version of the script is enhanced to handle convergence studies where both horizontal and vertical resolutions are varied independently.

The script performs the following main tasks:

    Parses a list of NetCDF result files, identifying the horizontal (H) and vertical (V) resolution of each.

    Reconstructs the model mesh for each simulation based on filename conventions.

    Extracts final-state velocity data along a central flowline for both the surface and the base of the ice.

    Interpolates all results onto a common high-resolution grid for comparison.

    Calculates the L2 error norm for each lower-resolution result against a high-resolution reference solution.

    Generates a 2x2 summary plot and a Markdown report summarizing the convergence results.

2. How to Use

This is a command-line script. You would run it from the terminal, providing the paths to the NetCDF output files as arguments. The script expects the files to be in the same directory.

Example Usage:

python convergence_analyser_H_V.py MyExperiment_Scenario1_*_*-Transient.nc

or

python convergence_analyser_H_V.py MyExperiment_Scenario1_1.0_1.0-Transient.nc MyExperiment_Scenario1_1.5_1.0-Transient.nc MyExperiment_Scenario1_2.0_2.0-Transient.nc

3. Key Components and Workflow
a. File Parsing and Setup (_load_results)

    The script uses a regular expression to parse the filenames. It expects a specific format:
    [ParamFile]_[Scenario]_[H_ResolutionFactor]_[V_ResolutionFactor]-Transient.nc

        ParamFile: The base name of the ISSM parameter file (e.g., IsmipF).

        Scenario: The experiment name (e.g., S1).

        H_ResolutionFactor: A float representing the horizontal mesh resolution multiplier (e.g., 1.0, 1.5).

        V_ResolutionFactor: A float representing the vertical mesh resolution multiplier (e.g., 1.0, 2.0).

    It identifies one simulation as the reference solution. This is hardcoded as a tuple, for instance (2, 2), representing the highest resolution run for both horizontal and vertical factors (note: now uses integer values).

b. Mesh Reconstruction (reconstruct_mesh)

    This is a crucial and assumption-heavy step. The script does not read the mesh from the output files. Instead, it recreates it from scratch.

    It assumes the model domain is a square of size 100km×100km.

    It creates a 2D squaremesh where the number of horizontal nodes is scaled by the h_resolution_factor (e.g., 30 * h_resolution_factor).

    It sets required miscellaneous attributes (filename, scenario, h_resolution_factor, v_resolution_factor) before parameterization.

    It parameterizes the model using the corresponding .py parameter file (e.g., IsmipF.py), which it assumes is in the parent directory.

    Finally, it extrudes the 2D mesh into a 3D mesh where the number of vertical layers is scaled by the v_resolution_factor (e.g., 5 * v_resolution_factor).

c. Data Extraction (_load_results)

    This process is largely the same as the original script. It opens each NetCDF file, focuses on the last time step, and extracts velocity data along a calculated centerline for the ice surface and base.

d. Interpolation (_interpolate_to_common_grid)

    To compare results from different grids, they must be evaluated at the same points. This function interpolates all velocity profiles onto the horizontal grid of the high-resolution reference solution.

e. Error Calculation (_calculate_convergence_metrics)

    The script calculates the L2 error norm to measure the difference between each result and the reference solution. For a reference solution vector u_ref and a comparison solution vector u_comp, the relative L2 error is:

    E_L2 = (||u_ref - u_comp||_2 / ||u_ref||_2) × 100

    where ||·||_2 is the Euclidean norm.

    The script intelligently reports the absolute error if the norm of the reference solution is near zero (< 0.1) to avoid division errors.

f. Outputs (_create_comparison_plots, _generate_report)

The script generates two primary output files:

    A PNG Image (*_convergence_summary.png): A 2x2 plot containing:

        Top-Left: Surface velocity profiles for all resolutions along the centerline.

        Top-Right: Basal velocity profiles for all resolutions along the centerline.

        Bottom-Left: A bar chart showing the calculated L2 errors for each resolution combination, with separate bars for surface and basal velocity errors.

        Bottom-Right: The time evolution of the maximum velocity in the domain for each simulation.

    A Markdown Report (*_convergence_report.md): A text file with a formatted table summarizing the L2 errors for both surface and basal velocities, stating whether each resolution has "CONVERGED" based on a 1% relative error tolerance. The resolution is reported as a tuple (H, V).

4. Dependencies and Limitations

    Dependencies: The script requires numpy, matplotlib, and netCDF4. Most importantly, it has a critical dependency on pyISSM, the Python interface for ISSM.

    Hardcoded pyISSM Path: The line sys.path.append('/home/ana/pyISSM/src') is a hardcoded path. To run this script, you would need to change this to the location of the pyISSM/src directory.

    Hardcoded Assumptions: The script is not general-purpose. It is tailored for a specific experimental setup and makes several hardcoded assumptions:

        Domain size is 100km×100km.

        The base horizontal mesh resolution is 30×30 elements.

        The base number of vertical layers is 5.

        The reference resolution is the tuple (2, 2).

        The centerline is at y=50000 m (automatically detected from mesh).

    Filename Convention: The script will fail if the NetCDF files do not strictly adhere to the ParamFile_Scenario_H-Res_V-Res-Transient.nc naming convention. It now uses a more robust regular expression for parsing filenames.