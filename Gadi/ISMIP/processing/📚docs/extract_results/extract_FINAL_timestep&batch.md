Executive Summary

These two scripts automate the visualization of the final results from Ice Sheet System Model (ISSM) simulations. They are designed to efficiently process large datasets and extract key summary information.

    extract_final_step.py: This is the "worker" script. Its job is to process a single NetCDF (.nc) simulation output file. It intelligently reconstructs the model's 3D mesh, reads the time-series data, and generates .png image plots for various physical fields (like velocity and pressure) for only the final time step. As a bonus, it also creates a summary plot showing the evolution of the maximum velocity over the entire simulation.

    batch_extract_final_step.py: This is the "manager" or "orchestrator" script. It finds multiple .nc files and manages the process of running extract_final_step.py on each one. It is designed to be highly robust, with features for parallel processing, resuming interrupted jobs, and sophisticated error checking to ensure that every file is handled correctly.

In-Depth Analysis
1. extract_final_step.py (The Worker)

This script is the core engine for plotting the final-state results. When run, it performs the following steps for a single input file (e.g., IsmipF_S1_1-Transient.nc):

    Mesh Reconstruction: The script cannot plot the data without the model's grid (the mesh). It cleverly reconstructs this mesh by:

        Parsing the input filename to infer the original parameter file (e.g., IsmipF.py) and the mesh resolution.

        Using the pyISSM library to programmatically repeat the steps that created the original mesh (squaremesh, parameterize, extrude).

    Data Loading: It opens the NetCDF file and reads the full transient solution data, including all time steps and results for different variables.

    Plot Generation:

        Final Step Plots: It jumps directly to the last time step in the results. For this final step, it plots a predefined list of fields: Vx, Vy, Vz (velocity components), Vel (velocity magnitude), and Pressure. It generates separate plots for the ice 'Surface' and 'Basal' layers.

        Summary Plot: Before finishing, it processes the entire time series for the Vel field to generate a 2D plot of maximum velocity vs. time, providing a concise summary of the simulation's stability and evolution.

        A critical safety check ensures that the number of data points matches the number of mesh vertices before plotting.

    Output: The generated plots are saved into a structured directory hierarchy with a _FINAL suffix. For an input file named IsmipF_S1_1-Transient.nc, the output would look like this:

    IsmipF_S1_1-Transient_FINAL/
    ├── velocity_evolution.png  <-- Summary Plot
    ├── Base/
    │   ├── Pressure/
    │   │   └── final_Pressure.png
    │   ├── Vel/
    │   │   └── final_Vel.png
    │   └── ...
    └── Surface/
        ├── Vel/
        │   └── final_Vel.png
        └── ...

2. batch_extract_final_step.py (The Manager)

This script makes the visualization process scalable and reliable. It does not perform any plotting itself; instead, it intelligently calls extract_final_step.py for you.

    File Discovery: It uses glob to find all .nc files in a directory that match a given pattern.

    Parallel Execution (--parallel): To significantly speed up processing a large number of files, it can run multiple instances of extract_final_step.py simultaneously, taking advantage of multiple CPU cores.

    Robust Job Management: This is the script's strongest feature.

        Skip Existing (--skip-existing): It can check if an output directory (with the _FINAL suffix) already exists and skip it, saving time on reruns.

        Resume (--resume): If a large batch job is interrupted, this feature allows it to resume from where it left off.

        Silent Failure Detection: It doesn't just trust that the worker script "succeeded". After the worker finishes, the manager checks if any .png files were actually created. If not, it's flagged as a "Silent Failure" for easy diagnosis.

        Detailed Reporting: At the end of the batch run, it generates a batch_final_processing_summary.txt file. This report neatly summarizes which files succeeded, were skipped, or failed, providing detailed error messages.

How to Use the Scripts

    Prerequisites:

        Ensure you have Python installed with the necessary libraries: numpy, matplotlib, and netCDF4. You can install them with pip:

        pip install numpy matplotlib netCDF4

        The pyISSM library must be installed and accessible. Crucially, the path in extract_final_step.py is hardcoded:

        sys.path.append('/home/ana/pyISSM/src')

        You must change /home/ana/pyISSM/src to the correct path on your system.

    File Placement:

        Place both extract_final_step.py and batch_extract_final_step.py in the same directory.

        Place your .nc data files in that same directory (or specify the directory with the --directory argument).

        Ensure your ISSM parameter files (e.g., IsmipF.py) are located in the parent directory of where the scripts are, as expected by the path logic in reconstruct_mesh.

    Execution:
    Open your terminal in the directory containing the scripts.

        To process all .nc files sequentially:

        python batch_extract_final_step.py

        To run in parallel for maximum speed:

        python batch_extract_final_step.py --parallel

        To process only a specific subset of experiments (e.g., all S4 runs):

        python batch_extract_final_step.py --parallel --pattern="*_S4_*.nc"

        To resume an interrupted job:

        python batch_extract_final_step.py --parallel --resume

