The script is a post-processing tool designed to analyze the output of a transient ice flow simulation from the Ice Sheet System Model (ISSM). Its primary goal is to quantify the phase relationship between the bedrock topography and the responding ice surface topography over time.

In simpler terms, if there is a bump on the bedrock, the ice flowing over it will also form a bump on its surface. This surface bump is often shifted slightly downstream. The script measures this downstream shift (spatial lag) and expresses it as a phase angle, tracking how it changes throughout the simulation.

2. How It Works (Methodology)

The script follows a sophisticated workflow to perform the analysis:

    Parameter Inference: It intelligently parses the input NetCDF filename (e.g., IsmipF_S1_30-Transient.nc) to automatically deduce key simulation parameters like the parameter profile (IsmipF), scenario (S1), and mesh resolution (30).

    Mesh & Configuration Reconstruction: Instead of just reading coordinates, the script reconstructs the original model setup.

        It uses pyISSM library functions (squaremesh, parameterize) to rebuild the model mesh exactly as it was during the simulation run. This requires access to the original ISSM parameter file (e.g., IsmipF.py).

        It safely parses this parameter file as text to extract key physical constants like the background ice thickness (H_0), the bedrock slope (alpha), and the characteristic wavelength of the bedrock perturbation (sigma), without executing the file.

    Profile Extraction: For each time step recorded in the NetCDF file, it extracts a 1D profile of the ice base and surface elevations along a user-defined line (e.g., along the x-axis at the domain's center).

    Signal Isolation: The core of the analysis is to separate the "signal" (the topographic bump) from the "background" (the unperturbed, sloping ice sheet). It does this by calculating the theoretical baseline for the bed and surface and subtracting it from the actual elevations.

    Cross-Correlation Analysis: To find the shift between the bedrock signal and the surface signal, it uses a standard signal processing technique called cross-correlation.

        It computes the cross-correlation function between the two signals using scipy.signal.correlate.

        The peak of this function corresponds to the spatial lag (d_lag) where the two signals are most similar.

    Phase Shift Calculation: The calculated spatial lag is converted into a phase shift (phi) in degrees using the bedrock's characteristic wavelength (lambda, which is assumed to be the parameter sigma). The formula used is:
    ϕdeg​=(λ2π⋅dlag​​)⋅π180​

    Time Evolution Analysis: Steps 3-6 are repeated for every time step in the simulation. The script stores the phase shift and lag distance for each point in time, allowing it to generate plots that show how the phase relationship evolves.

3. Usage

The script is run from the command line.

Syntax:
Bash

python phase_analysis.py <path_to_netcdf_file> [options]

Arguments:

    <path_to_transient_results.nc>: (Required) The full path to the NetCDF output file from the ISSM transient simulation.

    --axis {x,y}: (Optional) The axis along which to take the 1D profile. Defaults to 'x'.

    --position <value>: (Optional) The coordinate on the other axis where the profile line is located. For example, if --axis x, this is the y-coordinate. If not provided, it defaults to the center of the domain.

Examples:
Bash

# Analyze the centerline profile along the x-axis
python phase_analysis.py IsmipF_S1_30-Transient.nc

# Analyze a profile along the x-axis at y = 25000 m
python phase_analysis.py IsmipF_S1_30-Transient.nc --axis x --position 25000

# Analyze a profile along the y-axis at x = 40000 m
python phase_analysis.py IsmipF_S1_30-Transient.nc --axis y --position 40000

4. Dependencies & File Structure

To run the script successfully, you need:

Python Libraries:

    numpy

    matplotlib

    netCDF4

    scipy

ISSM Library:

    pyISSM: The script relies heavily on the Python interface for ISSM. You must have pyISSM installed and accessible.

    Path Configuration: The script contains a hardcoded path: sys.path.append('/home/ana/pyISSM/src'). You will need to change this path to match the location of the pyISSM/src directory on the system.

File Structure:

    The script assumes that the ISSM parameter file (e.g., IsmipF.py) that was used to generate the data is located in the parent directory of the script itself.

5. Outputs

The script generates a new directory named after the input file, with _phase_analysis appended (e.g., IsmipF_S1_30-Transient_phase_analysis/). Inside this directory, you will find:

    summary.txt: A detailed text report containing:

        A timestamp of the analysis.

        Information on the source file and simulation parameters.

        The specific profile that was analyzed.

        A table listing the calculated phase shift (degrees) and lag distance (km) for every time step.

    A 1_Signals/ directory: Contains PNG plots for each time step, visualizing the isolated bedrock and ice surface signals.

    A 2_Cross_Correlation/ directory: Contains PNG plots for each time step, showing the cross-correlation function and the detected lag of maximum correlation.

    A 3_Evolution_Plots/ directory: Contains summary PNG plots showing the evolution of the phase shift and spatial lag over the entire simulation time.