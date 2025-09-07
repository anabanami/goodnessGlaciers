The script automates the process of analyzing the results from multiple transient ice flow simulations that were run at different grid resolutions. Its goal is to determine if the model's solution (specifically ice velocity) "converges," meaning the results become stable and stop changing significantly as the mesh resolution increases. This is a critical step in verifying the numerical accuracy of a model setup.

The script performs the following main tasks:

    Parses a list of NetCDF result files, identifying the resolution of each.

    Reconstructs the model mesh for each simulation based on filename conventions.

    Extracts final-state velocity data along a central flowline for both the surface and the base of the ice.

    Interpolates all results onto a common high-resolution grid for comparison.

    Calculates the L2 error norm for each lower-resolution result against a high-resolution reference solution.

    Generates a 2x2 summary plot and a Markdown report summarizing the convergence results.

2. How to Use

This is a command-line script. You would run it from the terminal, providing the paths to the NetCDF output files as arguments. The script expects the files to be in the same directory.

Example Usage:
Bash

python convergence_analyzer.py MyExperiment_Scenario1_*.nc

or
Bash

python convergence_analyzer.py MyExperiment_Scenario1_1.0-Transient.nc MyExperiment_Scenario1_1.5-Transient.nc MyExperiment_Scenario1_2.0-Transient.nc

3. Key Components and Workflow

a. File Parsing and Setup (_load_results)

    The script uses a regular expression to parse the filenames. It expects a specific format:
    [ParamFile]_[Scenario]_[ResolutionFactor]-Transient.nc

        ParamFile: The base name of the ISSM parameter file (e.g., IsmipF).

        Scenario: The experiment name (e.g., S1).

        ResolutionFactor: A float representing the mesh resolution multiplier (e.g., 1.0, 1.5, 2.0).

    It identifies one simulation as the reference solution. This is hardcoded to be the one with resolution_factor=2.0. All other simulations will be compared against this one.

b. Mesh Reconstruction (reconstruct_mesh)

    This is a crucial and assumption-heavy step. The script does not read the mesh from the output files. Instead, it recreates it from scratch.

    It assumes the model domain is a square of size 100textkmtimes100textkm.

    It creates a 2D squaremesh where the number of nodes is scaled by the resolution_factor (e.g., 30 * resolution_factor).

    It parameterizes the model using the corresponding .py parameter file (e.g., IsmipF.py), which it assumes is in the parent directory.

    Finally, it extrudes the 2D mesh into a 3D mesh with 5 layers.

c. Data Extraction (_load_results)

    For each file, it opens the NetCDF dataset.

    It focuses on the last time step of the transient solution.

    Centerline Extraction: It extracts data along a central profile of the model domain. It does this by:

        Defining a geometric centerline at y=50000 m.

        Finding the actual y-coordinate in the mesh that is closest to this geometric center.

        Selecting all surface and basal nodes that lie on this identified mesh centerline.

    It stores the x-coordinates and corresponding velocities (converted to m/yr) for the surface and base, along with the time series of the maximum velocity across the whole domain.

d. Interpolation (_interpolate_to_common_grid)

    To compare results from different grids, they must be evaluated at the same points.

    This function takes the sorted x-coordinates from the high-resolution reference grid.

    It then uses 1D linear interpolation (numpy.interp) to resample the velocity profiles from all other resolutions onto this common reference grid.

e. Error Calculation (_calculate_convergence_metrics)

    The script calculates the L2 error norm, which is a standard way to measure the difference between two datasets. For a reference solution vector u_ref and a comparison solution vector u_comp (both on the same grid), the relative L2 error is:
    EL2​=∥uref​∥2​∥uref​−ucomp​∥2​​


    where ∣cdot∣_2 is the Euclidean norm.

    The script includes a smart feature: if the norm of the reference solution is very small (less than 0.1, e.g., for basal velocity in a frozen-bed region), it reports the absolute error (∣u_ref−u_comp∣_2) instead of the relative error to avoid division by zero or near-zero.

f. Outputs (_create_comparison_plots, _generate_report)

The script generates two primary output files:

    A PNG Image (*_convergence_summary.png): A 2x2 plot containing:

        Top-Left: Surface velocity profiles for all resolutions.

        Top-Right: Basal velocity profiles for all resolutions.

        Bottom-Left: A bar chart showing the calculated L2 errors.

        Bottom-Right: The time evolution of the maximum velocity in the domain for each simulation, showing how the solutions converge over time.

    A Markdown Report (*_convergence_report.md): A text file with a formatted table summarizing the L2 errors and stating whether each resolution has "CONVERGED" based on a 1% relative error tolerance.

4. Dependencies and Limitations

    Dependencies: The script requires numpy, matplotlib, and netCDF4. Most importantly, it has a critical dependency on pyISSM, the Python interface for ISSM.

    Hardcoded pyISSM Path: The line sys.path.append('/home/ana/pyISSM/src') is a hardcoded path specific to a user named "ana". To run this script, you would need to change this to the location of the pyISSM/src directory.

    Hardcoded Assumptions: The script is not general-purpose. It is tailored for a specific experimental setup and makes several hardcoded assumptions:

        Domain size is 100textkmtimes100textkm.

        The base mesh resolution is 30times30 elements.

        The mesh is extruded to 5 layers.

        The reference resolution factor is 2.0.

        The centerline is at y=50000 m.

    Filename Convention: The script will fail if the NetCDF files do not strictly adhere to the ParamFile_Scenario_ResolutionFactor-Transient.nc naming convention.