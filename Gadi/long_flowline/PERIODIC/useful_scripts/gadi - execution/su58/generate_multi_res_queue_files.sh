#!/bin/bash

# Base directories for GADI structure
MODEL_BASE="/home/565/ah3716/ice_models/flowline"
EXEC_BASE="/scratch/su58/ah3716/execution/flowline"

# Experiments
EXPERIMENTS=("S1" "S2" "S3" "S4")
# EXPERIMENTS=("S3" "S4")

# Resolution factors
RESOLUTION_FACTORS=("0.75" "0.875" "1.0" "1.125")

# Fixed parameters
FINAL_TIME=300     # years (matching periodic_flowline.py)
TIMESTEP=0.0833333 # years (base timestep = 1/12, actual timestep = base/resolution_factor)

echo "Generating queue files for grid resolution testing..."
echo "Model base: ${MODEL_BASE}"
echo "Execution base: ${EXEC_BASE}"

total_queues=0
created_queues=0

for EXP in "${EXPERIMENTS[@]}"; do
    EXP_DIR="${MODEL_BASE}/${EXP}"
    
    if [ ! -d "${EXP_DIR}" ]; then
        echo "Warning: Experiment directory ${EXP_DIR} does not exist, skipping."
        continue
    fi
    
    echo "Processing experiment: ${EXP}"
    
    # Find all profile directories
    PROFILE_DIRS=($(find "${EXP_DIR}" -mindepth 1 -maxdepth 1 -type d | sort))
    
    if [ ${#PROFILE_DIRS[@]} -eq 0 ]; then
        echo "  No profile directories found in ${EXP_DIR}"
        continue
    fi
    
    for PROFILE_DIR in "${PROFILE_DIRS[@]}"; do
        PROFILE=$(basename "${PROFILE_DIR}")
        
        # Find resolution factor directories within this profile
        RES_DIRS=($(find "${PROFILE_DIR}" -mindepth 1 -maxdepth 1 -type d | sort))
        
        for RES_DIR in "${RES_DIRS[@]}"; do
            RES_FACTOR=$(basename "${RES_DIR}")
            
            # Check if this is a valid resolution factor
            if [[ ! " ${RESOLUTION_FACTORS[@]} " =~ " ${RES_FACTOR} " ]]; then
                continue
            fi
            
            # Look for model files in this directory
            MODEL_FILES=($(find "${RES_DIR}" -maxdepth 1 -name "*.bin" 2>/dev/null))
            
            for MODEL_FILE in "${MODEL_FILES[@]}"; do
                MODEL_NAME=$(basename "${MODEL_FILE}" .bin)
                
                # Create execution directory
                EXEC_DIR="${EXEC_BASE}/${EXP}/${PROFILE}/${RES_FACTOR}"
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

# Enable spack
source /home/565/ah3716/spack/0.22/spack-config/spack-enable.bash

# Load spack issm module
spack load issm@ana-local-version-allocation-bugfix %gcc@13

# Run simulation for experiment ${EXP}, profile ${PROFILE}, resolution ${RES_FACTOR}
# ISSM syntax: TransientSolution <model_dir> <model_name> <execution_dir>
# - <model_dir>: where .bin/.toolkits files are located AND where output files will be saved
# - <model_name>: base name of model files  
# - <execution_dir>: where job executes and temporary files go
mpiexec -np 12 issm.exe TransientSolution ${RES_DIR} ${MODEL_NAME} ${EXEC_DIR}
EOF

                created_queues=$((created_queues + 1))
                echo "  [${EXP}/${PROFILE}/${RES_FACTOR}] created queue: ${QUEUE_FILE}"
            done
        done
    done
done

echo ""
echo "Queue file generation summary:"
echo "  Total queue files created: ${created_queues}"
echo "  Model structure: ${MODEL_BASE}/{EXP}/{PROFILE}/{RES_FACTOR}/"
echo "  Execution structure: ${EXEC_BASE}/{EXP}/{PROFILE}/{RES_FACTOR}/"
echo ""
echo "Next step: Use submit_all_jobs.sh to submit the queue files"
