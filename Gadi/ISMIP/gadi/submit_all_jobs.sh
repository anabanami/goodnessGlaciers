#!/bin/bash

# This script finds and submits all 'IsmipF.queue' files located within
# the nested subdirectories of 'key_res_factor=(...)'.
# It should be run from the main 'execution' directory.

echo "Starting job submission process..."
echo "-------------------------------------------------"

# Get the absolute path of the directory where the script is running.
BASE_DIR=$(pwd)

# Loop through all top-level directories that match the 'key_res_factor' pattern.
# The trailing slash ensures we only match directories.
for res_dir in "${BASE_DIR}"/'key_res_factor='*','*'/'; do
    # Check if the found path is actually a directory.
    if [[ ! -d "${res_dir}" ]]; then
        continue
    fi
    
    # folder (e.g., "S1_F"), which is where the queue files are actually located.
    for model_dir in "${res_dir}"*/; do
        if [[ ! -d "${model_dir}" ]]; then
            continue
        fi

        QUEUE_FILE_PATH="${model_dir}IsmipF.queue"

        # Check if the queue file actually exists in this subdirectory.
        if [ -f "$QUEUE_FILE_PATH" ]; then
            echo "Found queue file: $QUEUE_FILE_PATH"
            echo "Submitting job from directory: ${model_dir}"

            # Submit the job from within its directory.
            # Using a subshell (...) is a clean way to change directory
            # for a single command without affecting the script's main location.
            (cd "${model_dir}" && qsub IsmipF.queue)

            echo "-------------------------------------------------"
        else
            # This warning is now more specific to the subdirectory.
            echo "Info: No queue file found in ${model_dir}"
            echo "-------------------------------------------------"
        fi
    done
done

echo "All jobs have been submitted."