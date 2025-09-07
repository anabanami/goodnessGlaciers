### Executive Summary

These two scripts automate the visualization of simulation results from the Ice Sheet System Model (ISSM).

1.  **`extract_results.py`**: This is the "worker" script. Its job is to process a *single* NetCDF (`.nc`) simulation output file. It intelligently reconstructs the model's 3D mesh from information in the filename, reads the time-series data, and generates a series of `.png` image plots for various physical fields (like velocity and pressure).

2.  **`batch_extract_results.py`**: This is the "manager" or "orchestrator" script. It finds multiple `.nc` files and manages the process of running `extract_results.py` on each one. It is designed to be highly robust, with features for parallel processing, resuming interrupted jobs, and sophisticated error checking to ensure that every file is handled correctly.

-----

### In-Depth Analysis

#### 1\. `extract_results.py` (The Worker)

This script is the core engine for plotting. When run, it performs the following steps for a single input file (e.g., `IsmipF_S1_1-Transient.nc`):

  * **Mesh Reconstruction**: The script cannot plot the data without the model's grid (the mesh). It cleverly reconstructs this mesh by:

      * Parsing the input filename to infer the original parameter file (e.g., `IsmipF.py`) and the mesh resolution.
      * Using the `pyISSM` library to programmatically repeat the steps that created the original mesh (`squaremesh`, `parameterize`, `extrude`).
      * **Note**: It contains a fix `float(parts[2].split('-')[0])` to correctly handle non-integer resolution factors.

  * **Data Loading**: It opens the NetCDF file and reads the transient solution data, including the time steps and the results for different variables.

  * **Plot Generation**:

      * It iterates through each time step in the results.
      * For each step, it plots a predefined list of fields: `Vx`, `Vy`, `Vz` (velocity components), `Vel` (velocity magnitude), and `Pressure`.
      * It generates separate plots for the ice 'Surface' and 'Basal' layers.
      * It converts velocity units from meters per second ($m/s$) to a more intuitive meters per year ($m/yr$) for the plots.
      * A critical safety check ensures that the number of data points in the results file matches the number of vertices in the reconstructed mesh before attempting to plot.

  * **Output**: The generated plots are saved into a structured directory hierarchy. For an input file named `IsmipF_S1_1-Transient.nc`, the output would look like this:

    ```
    IsmipF_S1_1-Transient/
    ├── Base/
    │   ├── Pressure/
    │   │   ├── Pressure_step001.png
    │   │   └── ...
    │   ├── Vel/
    │   │   └── ...
    │   └── ...
    └── Surface/
        ├── Vel/
        │   ├── Vel_step001.png
        │   └── ...
        └── ...
    ```

#### 2\. `batch_extract_results.py` (The Manager)

This script makes the visualization process scalable and reliable. It does not perform any plotting itself; instead, it intelligently calls `extract_results.py` for you.

  * **File Discovery**: It uses `glob` to find all `.nc` files in a directory that match a given pattern.

  * **Parallel Execution (`--parallel`)**: To significantly speed up processing a large number of files, it can run multiple instances of `extract_results.py` simultaneously using Python's `ProcessPoolExecutor`, taking advantage of multiple CPU cores.

  * **Robust Job Management**: This is the script's strongest feature.

      * **Skip Existing (`--skip-existing`)**: It can check if an output directory with plots already exists for a file and skip it, saving time on reruns.
      * **Resume (`--resume`)**: If a large batch job is interrupted, this feature allows it to resume from where it left off by reading a `batch_resume.txt` file that tracks completed files.
      * **Silent Failure Detection**: It doesn't just trust that the worker script "succeeded". After `extract_results.py` finishes, the manager checks if any `.png` files were actually created. If the worker script ran without crashing but produced no output, it's flagged as a "Silent Failure"—a very powerful diagnostic tool.
      * **Detailed Reporting**: At the end of the batch run, it generates a `batch_processing_summary.txt` file. This report neatly summarizes which files succeeded, which were skipped, and which failed, providing detailed error messages for each failure.

### How to Use the Scripts

1.  **Prerequisites**:

      * Ensure you have Python installed with the necessary libraries: `numpy`, `matplotlib`, and `netCDF4`. You can install them with pip:
        ```bash
        pip install numpy matplotlib netCDF4
        ```
      * The `pyISSM` library must be installed and accessible. **Crucially**, the path in `extract_results.py` is hardcoded:
        ```python
        sys.path.append('/home/ana/pyISSM/src')
        ```
        You **must** change `/home/ana/pyISSM/src` to the correct path on the system.

2.  **File Placement**:

      * Place both `extract_results.py` and `batch_extract_results.py` in the same directory.
      * Place the `.nc` data files in that same directory (or specify the directory with the `--directory` argument).
      * Ensure the ISSM parameter files (e.g., `IsmipF.py`) are located in the parent directory of where the scripts are, as expected by the path logic in `reconstruct_mesh`.

3.  **Execution**:
    In a terminal in the directory containing the scripts.

      * **To process all `.nc` files sequentially**:

        ```bash
        python batch_extract_results.py
        ```

      * **To run in parallel for maximum speed**:

        ```bash
        python batch_extract_results.py --parallel
        ```

      * **To process only a specific subset of experiments (e.g., all S4 runs)**:

        ```bash
        python batch_extract_results.py --parallel --pattern="*_S4_*.nc"
        ```

      * **To resume an interrupted job**:

        ```bash
        python batch_extract_results.py --parallel --resume
        ```