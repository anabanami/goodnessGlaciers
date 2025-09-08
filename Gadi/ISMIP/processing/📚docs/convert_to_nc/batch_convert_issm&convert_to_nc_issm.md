The primary goal of these scripts is to leverage ISSM's own internal functions to ensure the data structure is perfectly preserved, while providing a powerful and user-friendly interface for batch processing large numbers of files efficiently.
Scripts Overview

There are two main scripts that work in tandem:

    convert_to_nc_issm.py: The core conversion engine. This script handles the logic for converting a single .outbin file. It directly imports and uses ISSM's Python tools to load the binary data and then meticulously reconstructs the nested data structure within a NetCDF file.

    batch_convert_issm.py: The high-level batch processor. This script acts as a wrapper around the core engine, providing advanced features like automatic file discovery, parallel processing to speed up conversions on multi-core machines, and the ability to skip already converted files.

Core Philosophy: Using ISSM's Built-in Functions

The reliability of this conversion process hinges on its use of official ISSM Python functions. Instead of trying to manually parse the proprietary .outbin format, the convert_to_nc_issm.py script uses the following key functions from the ISSM environment:

    from loadresultsfromdisk import loadresultsfromdisk: This is the standard ISSM function for loading simulation results from a .outbin file on your local machine. It reads the binary data and populates an ISSM model object with the results.

    from loadresultsfromcluster import loadresultsfromcluster: This is an alternative function used for loading results directly from a remote cluster where the simulation was run.

    from model import model: The script creates an instance of the main ISSM model class. This object acts as a container that the loadresults... functions populate with the simulation data, including mesh details, parameters, and time-dependent solution steps.

    from results import results: The results of the simulation are stored within the model.results object, which this script uses to access the data for writing to NetCDF.

By using these functions, we guarantee that the data is interpreted exactly as ISSM intended, preserving the integrity of complex, nested structures like md.results.TransientSolution.steps.
Script 1: convert_to_nc_issm.py (The Engine)

This script is the foundation of the conversion process.
Key Functionality

    ISSM Function Integration: As described above, it directly calls loadresultsfromdisk or loadresultsfromcluster to load data into an ISSM model object.

    Structure Preservation: After loading, it inspects the model object to identify the solution type (e.g., TransientSolution) and iterates through each time step.

    NetCDF Grouping: It creates a NetCDF file that mirrors the ISSM results structure. The data is organized into nested groups, typically results/TransientSolution/, making the final file intuitive to navigate.

    Dimension and Variable Handling: It intelligently detects the dimensions of the data (like time and vertices) and creates corresponding NetCDF variables for each field (e.g., Vel, MaskIceLevelset, Pressure).

    File Name Recognition: It is specifically designed to work with files matching the pattern IsmipF_S[1-4]_*_*-Transient.outbin, ensuring it only processes relevant simulation outputs.

Direct Usage (for a single file)

While it's designed to be called by the batch script, you can use it directly to convert a single file:

# Convert a single file using results from disk
python convert_to_nc_issm.py IsmipF_S1_0.5_2-Transient.outbin

# Convert a single file by loading results from a cluster
python convert_to_nc_issm.py IsmipF_S1_0.5_2-Transient.outbin --from-cluster

Script 2: batch_convert_issm.py (The Batch Processor)

This script makes the conversion process practical for large datasets. It automates the process of finding and converting files using the engine script.
Key Functionality

    Automatic File Discovery: When run, it automatically scans the current directory for all files that match the IsmipF_S[1-4]_*_*-Transient.outbin pattern.

    Parallel Processing: This is a major feature for efficiency. By using the --parallel flag, the script will distribute the conversion tasks across multiple CPU cores. This can dramatically reduce the total time required to process dozens or hundreds of files. You can even control the number of simultaneous jobs with --max-workers.

    Skip Existing: If you run the script multiple times, the --skip-existing flag tells it not to re-convert any files that already have a corresponding .nc output. This is useful for resuming an interrupted batch job.

    Clear Summaries: After the process completes, it prints a clean, color-coded summary detailing which files were successfully converted, which were skipped, and which encountered errors.

Usage Examples

Place both scripts in the same directory as your .outbin files.

Basic Batch Conversion:
This command will find all matching .outbin files and convert them one by one.

python batch_convert_issm.py

Parallel Conversion (Recommended):
This is the most powerful command. It uses all available CPU cores to convert files simultaneously and skips any that have already been done.

python batch_convert_issm.py --parallel --skip-existing

Controlled Parallel Conversion:
This command uses a maximum of 4 CPU cores.

python batch_convert_issm.py --parallel --max-workers 4

Batch Conversion from a Cluster:
If your results are on a cluster, use the --from-cluster flag. This is passed down to the core conversion script.

python batch_convert_issm.py --parallel --from-cluster

Requirements

    A working Python environment.

    The ISSM Python libraries must be correctly installed and included in your PYTHONPATH. The scripts will fail if they cannot import loadresultsfromdisk, etc.

    The netCDF4 Python library (pip install netCDF4).