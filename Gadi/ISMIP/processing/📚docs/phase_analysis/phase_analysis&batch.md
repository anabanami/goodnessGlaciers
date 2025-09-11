This suite consists of two Python scripts designed to work together to perform and manage a phase relationship analysis of ice sheet dynamics.

    phase_analysis.py: This is the core scientific tool. It is designed to operate on a single NetCDF file from an ISSM simulation. It extracts data profiles, performs a cross-correlation analysis to determine the phase shift between the bedrock and ice surface, and generates detailed plots and a summary report.

    batch_phase_analysis.py: This is a powerful automation and management tool. It is designed to run phase_analysis.py on multiple NetCDF files efficiently. It handles file discovery, parallel processing, error handling, and provides high-level summary reports for large-scale analysis campaigns.

You will typically interact with the batch_phase_analysis.py script for most tasks, which in turn calls the core script for each data file.

phase_analysis.py: The Core Analysis Script
Purpose

To perform a time-dependent phase shift analysis between bedrock topography and ice surface elevation for a single ISSM transient simulation output file.
Key Features

    Automatic Parameter Detection: Parses the input filename (e.g., IsmipF_S1_2_2-Transient.nc) to extract simulation parameters like the experiment profile, scenario, horizontal resolution factor, and vertical resolution factor.

    Mesh Reconstruction: Reconstructs the original ISSM model mesh using the pyISSM library, properly setting miscellaneous attributes before parameterization.

    Profile Extraction: Extracts 1D data profiles for bedrock and ice surface elevation along a specified x or y coordinate, with automatic calculation of analytical unperturbed baselines.

    Signal Processing: Uses SciPy to perform a cross-correlation between the isolated signals (after removing unperturbed baselines), accurately calculating the spatial lag and phase shift for each time step.

    Comprehensive Output: Generates detailed plots and a text-based summary of the results, including offset surface plots for better visualization.

Usage

The script is called from the command line, with the path to the NetCDF file as the main argument.

python phase_analysis.py <path_to_the_netcdf_file.nc> [OPTIONS]

Command-Line Arguments

Argument
	

Description
	

Default

nc_file
	

(Required) The full path to the input transient results NetCDF file.
	


--axis {x,y}
	

The axis along which the 1D data profile will be extracted.
	

x

--position <value>
	

The coordinate (in meters) on the other axis where the profile line is located.
	

Domain Center

Example:

# Analyze a profile along the y-axis at x = 40000 meters
python phase_analysis.py IsmipF_S1_2_2-Transient.nc --axis y --position 40000

Input Requirements

    Python Environment: Must have numpy, matplotlib, netCDF4, and scipy installed.

    pyISSM: The path to the pyISSM source directory must be correctly set within the script.

    Filename Convention: The script expects filenames to follow the pattern Profile_Scenario_Hres_Vres-Transient.nc. For example: IsmipF_S1_2_2-Transient.nc. The parsing logic now properly handles both horizontal and vertical resolution factors with improved error handling.

Output Structure

For an input file named MyExperiment.nc, the script will create a directory named MyExperiment_phase_analysis/. This directory will contain:

    summary.txt: A detailed report including simulation parameters and a table of phase shifts and lag distances for every time step.

    1_Signals/: A folder containing plots of the isolated bedrock and surface signals for each time step (signals_t_XXXX.png). The surface signal is vertically offset by the reference ice thickness H_0 for clarity.

    2_Cross_Correlation/: A folder containing plots of the cross-correlation function for each time step (correlation_t_XXXX.png).

    3_Evolution_Plots/: A folder containing a final summary plot showing the evolution of the phase shift and spatial lag over the entire simulation (phase_evolution_summary.png).

batch_phase_analysis.py: The Automation Script
Purpose

To automate the execution of phase_analysis.py on multiple NetCDF files, providing features for parallel processing, error handling, and job management.
Key Features

    File Discovery: Automatically finds .nc files in a directory matching a specified pattern.

    Parallel Execution: Can run multiple analyses simultaneously to dramatically reduce processing time on multi-core systems.

    Robust Error Handling: Detects "silent failures" where the analysis script runs but produces no output, and handles timeouts gracefully.

    Job Management: Can skip already processed files and resume an interrupted batch run.

    Centralized Reporting: Generates a single summary report for the entire batch job.

Usage Examples

Basic Run (Sequential):
Process all .nc files in the current directory, one by one.

python batch_phase_analysis.py

Parallel Run with a Specific Pattern:
Process all files containing "S3" in their name, using all available CPU cores.

python batch_phase_analysis.py --pattern="*_S3_*.nc" --parallel

Passing Analysis Parameters:
Analyze a specific off-center profile (y=25000) for all files. Skip any that already have results.

python batch_phase_analysis.py --axis y --position 25000 --skip-existing

Resume an Interrupted Run:
If a previous run was stopped, this command will continue from where it left off.

python batch_phase_analysis.py --resume

Dry Run:
List all files that would be processed without actually running the analysis.

python batch_phase_analysis.py --dry-run

Command-Line Arguments

Argument
	

Description
	

Default

--pattern
	

The glob pattern to match for input files.
	

"*.nc"

--directory
	

The directory to search for files.
	

Current directory

--skip-existing
	

If set, skips any file that already has an output directory with a summary.txt.
	

False

--parallel
	

If set, processes files in parallel using all available CPU cores.
	

False

--max-workers <int>
	

Specify the maximum number of parallel processes.
	

All CPU cores

--resume
	

If set, resumes a batch run from a previous state.
	

False

--dry-run
	

If set, lists files to be processed and exits without running.
	

False

--axis
	

Passed directly to phase_analysis.py. See above.
	

x

--position
	

Passed directly to phase_analysis.py. See above.
	

None (center)
Output Files

In addition to the output directories created by phase_analysis.py, the batch script creates:

    batch_phase_summary.txt: A high-level report detailing which files succeeded, which failed (and why), and total processing time.

    batch_phase_resume.txt: A temporary file created during a run that tracks completed files. It is used for the --resume functionality and is automatically deleted upon successful completion of the entire batch.

Typical Workflow

    Preparation:

        Place all the .nc simulation output files into a single directory.

        Ensure the Python environment is correctly set up.

    Initial Test (Optional but Recommended):

        Run the core analysis script on a single file to ensure everything is working correctly.

    python phase_analysis.py MyExperiment_S1_2_2-Transient.nc

    Batch Processing:

        Navigate to the directory containing the .nc files.

        Use the batch_phase_analysis.py script to process all files. For the fastest results, use the --parallel flag.

    # Run analysis on all files in parallel, using the default centerline profile
    python batch_phase_analysis.py --parallel

    Review Results:

        Once the batch is complete, check the batch_phase_summary.txt for a high-level overview.

        For any failed files, the summary will provide error details.

        Dive into the individual output directories (e.g., MyExperiment_S1_2_2-Transient_phase_analysis/) to view the detailed plots and summary.txt reports.