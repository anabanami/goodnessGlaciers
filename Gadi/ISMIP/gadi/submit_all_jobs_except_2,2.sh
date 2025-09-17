#!/bin/bash

# This script finds and submits all '.queue' files located within
# the nested subdirectories of 'key_res_factor=(...)', skipping a specific folder.
# It should be run from the main 'execution' directory.

echo "Starting job submission process..."
echo "-------------------------------------------------"

# Get the absolute path of the directory where the script is running.
BASE_DIR=$(pwd)
SKIP_DIR_NAME="key_res_factor=(2,2)"

# 'nullglob' ensures that the loop doesn't run at all if no *.queue files are found,
# preventing errors.
shopt -s nullglob

# Loop through all top-level directories that match the 'key_res_factor' pattern.
for res_dir in "${BASE_DIR}"/'key_res_factor='*','*'/'; do
    # Check if the current item is a directory before proceeding
    if [[ ! -d "${res_dir}" ]]; then
        continue
    fi

    # Extract the base name of the directory to check if it should be skipped
    res_dir_basename=$(basename "${res_dir}")
    if [[ "${res_dir_basename}" == "${SKIP_DIR_NAME}" ]]; then
        echo "Skipping directory: ${res_dir}"
        echo "-------------------------------------------------"
        continue
    fi
    
    # Loop through the second-level model directories (e.g., "S1_F").
    for model_dir in "${res_dir}"*/; do
        if [[ ! -d "${model_dir}" ]]; then
            continue
        fi

        # Find and loop through all files ending with .queue in the directory
        found_jobs=0
        for queue_file_path in "${model_dir}"*.queue; do
            # Extract just the filename for the qsub command
            queue_filename=$(basename "$queue_file_path")
            
            echo "Found queue file: $queue_file_path"
            echo "Submitting job from directory: ${model_dir}"
            
            # Submit the job from within its directory.
            # Using a subshell (...) changes directory only for this command.
            (cd "${model_dir}" && qsub "${queue_filename}")
            
            echo "-------------------------------------------------"
            found_jobs=$((found_jobs + 1))
        done

        if [ "$found_jobs" -eq 0 ]; then
             echo "Info: No .queue files found in ${model_dir}"
             echo "-------------------------------------------------"
        fi
    done
done

echo "All jobs have been submitted."