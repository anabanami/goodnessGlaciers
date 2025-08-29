#!/usr/bin/env bash
set -euo pipefail

# Base directories for GADI structure
MODEL_BASE="/home/565/ah3716/ice_models/flowline"
EXEC_BASE="/scratch/su58/ah3716/execution/flowline"

echo "Auto-detecting folders and profile IDs from ${MODEL_BASE}"
echo "Model base: ${MODEL_BASE}"
echo "Execution base: ${EXEC_BASE}"
echo

# Find all S folders (S1, S2, S3, S4, etc.)
S_FOLDERS=($(find "${MODEL_BASE}" -mindepth 1 -maxdepth 1 -type d -name "S*" | sort))

if [ ${#S_FOLDERS[@]} -eq 0 ]; then
    echo "No S folders found in ${MODEL_BASE}"
    exit 1
fi

total_queues=0
created_queues=0

for S_FOLDER in "${S_FOLDERS[@]}"; do
    S_NAME=$(basename "${S_FOLDER}")
    echo "Processing experiment: ${S_NAME}"
    
    # Find all .bin files in this S folder to extract profile IDs and generate queue files
    BIN_FILES=($(find "${S_FOLDER}" -name "*.bin" 2>/dev/null))
    
    if [ ${#BIN_FILES[@]} -eq 0 ]; then
        echo "  No .bin files found in ${S_FOLDER}"
        continue
    fi
    
    # Extract unique profile IDs from .bin filenames
    declare -A profile_ids
    for bin_file in "${BIN_FILES[@]}"; do
        filename=$(basename "${bin_file}" .bin)
        # Extract profile ID - assuming format like "164_S4_0.5.bin" where 164 is the profile ID
        if [[ $filename =~ ^([0-9]+)_ ]]; then
            profile_id="${BASH_REMATCH[1]}"
            # Force base-10 to avoid octal interpretation
            profile_id_padded=$(printf "%03d" "$((10#$profile_id))")
            profile_ids["$profile_id_padded"]=1
        fi
    done
    
    # Log detected profile IDs
    echo "  Detected profile IDs: ${!profile_ids[@]}"
    
    # Generate queue files for each .bin file found
    for bin_file in "${BIN_FILES[@]}"; do
        MODEL_DIR=$(dirname "${bin_file}")
        MODEL_NAME=$(basename "${bin_file}" .bin)
        
        # Extract profile ID from filename for directory structure
        if [[ $MODEL_NAME =~ ^([0-9]+)_ ]]; then
            profile_id="${BASH_REMATCH[1]}"
            profile_id_padded=$(printf "%03d" "$((10#$profile_id))")
            
            # Create execution directory
            EXEC_DIR="${EXEC_BASE}/${S_NAME}/${profile_id_padded}"
            mkdir -p "${EXEC_DIR}"
            
            # Generate queue file in execution directory
            QUEUE_FILE="${EXEC_DIR}/${MODEL_NAME}.queue"
            
            total_queues=$((total_queues + 1))

            cat > "${QUEUE_FILE}" << EOF
#PBS -S /bin/bash
#PBS -P su58
#PBS -q normal
#PBS -l ncpus=12
#PBS -l walltime=48:00:00
#PBS -l mem=48gb
#PBS -M ana.fabelahinojosa@monash.edu
#PBS -o ${MODEL_NAME}.outlog
#PBS -e ${MODEL_NAME}.errlog
#PBS -l wd

# Source bashrc to set up environment
source \$HOME/.bashrc

# Load needed modules
module load openmpi/4.1.7
module load python3/3.9.2

# Enable spack
source /home/565/ah3716/spack/0.22/spack-config/spack-enable.bash

# Load spack issm module
spack load issm@ana-local-version %gcc@13

# Run simulation for experiment ${S_NAME}, profile ${profile_id_padded}
# ISSM syntax: TransientSolution <model_dir> <model_name> <execution_dir>
# - <model_dir>: where .bin/.toolkits files are located AND where output files will be saved
# - <model_name>: base name of model files  
# - <execution_dir>: where job executes and temporary files go
mpiexec -np 12 \$ISSM_DIR/bin/issm.exe TransientSolution ${MODEL_DIR} ${MODEL_NAME} ${EXEC_DIR}
EOF

            created_queues=$((created_queues + 1))
            echo "  [${S_NAME}/${profile_id_padded}] created queue: ${QUEUE_FILE}"
        fi
    done
    
    echo
done

echo "Summary:"
echo "  Total queue files created: ${created_queues}"
echo "  Model structure: ${MODEL_BASE}/{EXP}/{PROFILE}/"
echo "  Execution structure: ${EXEC_BASE}/{EXP}/{PROFILE}/"
echo
echo "Next step: Use submit_all_jobs.sh to submit the queue files"