#!/bin/bash

# This script generates PBS queue files for each simulation subdirectory.
# It dynamically determines the model name from the existing '*-Transient.bin' file.
# It should be run from the parent 'execution' directory that contains
# all the 'key_res_factor=(...)' folders.

# Get the absolute path of the current directory where the script is being run.
BASE_DIR=$(pwd)

echo "Starting queue file generation in: $BASE_DIR"
echo "-------------------------------------------------"

# Loop through all top-level directories that match the 'key_res_factor' pattern.
for res_dir in "${BASE_DIR}"/'key_res_factor='*','*'/'; do
    # Check if the found path is actually a directory.
    if [[ ! -d "${res_dir}" ]]; then
        continue
    fi

    # Now, loop through the subdirectories within each resolution folder (e.g., "S1_F").
    for model_dir in "${res_dir}"*/; do
        if [[ ! -d "${model_dir}" ]]; then
            continue
        fi

        # --- DYNAMIC NAME DETECTION ---
        # Find the first file ending in '-Transient.bin' to use as the base name.
        # The 'find...-quit' combination is efficient and stops after the first match.
        bin_file=$(find "${model_dir}" -maxdepth 1 -name "*-Transient.bin" -print -quit)

        # If no .bin file is found, skip this directory.
        if [[ -z "$bin_file" ]]; then
            echo "-> Warning: No '*-Transient.bin' file found in ${model_dir}. Skipping."
            continue
        fi

        # Extract the base name by removing the path and the suffix.
        # e.g., /path/to/IsmipF_S1_0.5_2-Transient.bin -> IsmipF_S1_0.5_2
        MODEL_NAME_BASE=$(basename "$bin_file" "-Transient.bin")
        
        # The model name needed by issm.exe requires the '-Transient' suffix.
        MODEL_NAME_FOR_ISSM="${MODEL_NAME_BASE}-Transient"

        # Define all file paths using the full, descriptive model name.
        QUEUE_FILE_PATH="${model_dir}${MODEL_NAME_FOR_ISSM}.queue"
        OUT_LOG_NAME="${MODEL_NAME_FOR_ISSM}.outlog"
        ERR_LOG_NAME="${MODEL_NAME_FOR_ISSM}.errlog"

        echo "-> Creating queue file for model '${MODEL_NAME_FOR_ISSM}' at: ${QUEUE_FILE_PATH}"

        # Create the queue file using a 'here document'.
        # We REMOVE quotes from 'EOF' to allow variable expansion for our new variables
        # (like $MODEL_NAME_FOR_ISSM), but we must ESCAPE PBS variables (like \$PBS_O_WORKDIR)
        # so they are expanded by the job scheduler, not by this script.
        cat > "${QUEUE_FILE_PATH}" << EOF
#PBS -S /bin/bash
#PBS -P au88
#PBS -q normal
#PBS -l ncpus=12
#PBS -l walltime=48:00:00
#PBS -l mem=48gb
#PBS -M ana.fabelahinojosa@monash.edu
#PBS -o ${OUT_LOG_NAME}
#PBS -e ${ERR_LOG_NAME}
#PBS -l wd

# Source bashrc to set up environment
source \$HOME/.bashrc

# Load needed modules
module load openmpi/4.1.7

# Enable spack
source /home/565/ah3716/spack/0.22/spack-config/spack-enable.bash

# Load spack issm module
spack load issm@ana-local-version-allocation-bugfix %gcc@13

# Run simulation for transient solve
# The job's working directory is set to the submission directory by "#PBS -l wd".
# \$PBS_O_WORKDIR holds the path to this submission directory.
mpiexec -np 12 issm.exe TransientSolution \$PBS_O_WORKDIR ${MODEL_NAME_FOR_ISSM} \$PBS_O_WORKDIR
EOF
    done
done

echo "-------------------------------------------------"
echo "Finished creating all queue files."