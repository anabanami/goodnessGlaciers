#!/bin/bash

# This script generates PBS queue files for each simulation subdirectory.
# It should be run from the parent 'execution' directory that contains
# all the 'key_res_factor=(...)' folders.

# Get the absolute path of the current directory where the script is being run.
BASE_DIR=$(pwd)

echo "Starting queue file generation in: $BASE_DIR"
echo "-------------------------------------------------"

# Loop through all top-level directories that match the 'key_res_factor' pattern.
# The trailing slash on the pattern ensures we only match directories.
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

        # Define the full path for the new queue file to be created.
        QUEUE_FILE_PATH="${model_dir}IsmipF.queue"

        echo "-> Creating queue file at: ${QUEUE_FILE_PATH}"

        # Create the queue file using a 'here document' (cat << 'EOF').
        # Quoting 'EOF' is important: it prevents the shell from expanding
        # variables like $HOME or $PBS_O_WORKDIR *now*. We want them to be
        # expanded later, when the PBS job actually runs.
        cat > "${QUEUE_FILE_PATH}" << 'EOF'
#PBS -S /bin/bash
#PBS -P su58
#PBS -q normal
#PBS -l ncpus=12
#PBS -l walltime=48:00:00
#PBS -l mem=48gb
#PBS -M ana.fabelahinojosa@monash.edu
#PBS -o IsmipF.outlog
#PBS -e IsmipF.errlog
#PBS -l wd

# Source bashrc to set up environment
source $HOME/.bashrc

# Load needed modules
module load openmpi/4.1.7

# Enable spack
source /home/565/ah3716/spack/0.22/spack-config/spack-enable.bash

# Load spack issm module
spack load issm@ana-local-version-allocation-bugfix %gcc@13

# Run simulation for IsmipF transient solve
# ISSM syntax: TransientSolution <model_dir> <model_name> <execution_dir>
# - <model_dir>: where .bin/.toolkits files are located AND where output files will be saved
# - <model_name>: base name of model files
# - <execution_dir>: where job executes and temporary files go

# The job's working directory is set to the submission directory by "#PBS -l wd".
# $PBS_O_WORKDIR is a PBS environment variable that holds the path to this submission directory.
# This makes the script generic and ensures the right paths are used for each job.
mpiexec -np 12 issm.exe TransientSolution $PBS_O_WORKDIR IsmipF $PBS_O_WORKDIR
EOF
    done
done

echo "-------------------------------------------------"
echo "Finished creating all queue files."
