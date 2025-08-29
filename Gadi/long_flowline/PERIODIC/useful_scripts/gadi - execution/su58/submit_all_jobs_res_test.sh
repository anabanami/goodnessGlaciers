#!/bin/bash

# Base directory for GADI structure
EXEC_BASE="/scratch/su58/ah3716/execution/flowline"

# Experiments and resolution factors
# add other EXP to list if needed
EXPERIMENTS=("S1" "S2" "S3" "S4")
# EXPERIMENTS=("S3" "S4")
RESOLUTION_FACTORS=("0.75" "0.875" "1.0"  "1.125")

echo "Submitting all queue files from grid resolution structure..."
echo "Model base: ${EXEC_BASE}"

total_submitted=0
total_found=0

for EXP in "${EXPERIMENTS[@]}"; do
    EXP_DIR="${EXEC_BASE}/${EXP}"
    
    if [ ! -d "${EXP_DIR}" ]; then
        echo "Warning: Experiment directory ${EXP_DIR} does not exist, skipping."
        continue
    fi
    
    echo "Processing experiment: ${EXP}"
    
    # Find all profile directories
    PROFILE_DIRS=($(find "${EXP_DIR}" -mindepth 1 -maxdepth 1 -type d | sort))
    
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
            
            # Find queue files in this directory
            QUEUE_FILES=($(find "${RES_DIR}" -maxdepth 1 -name "*.queue" 2>/dev/null))
            
            for QUEUE_FILE in "${QUEUE_FILES[@]}"; do
                queue_basename=$(basename "${QUEUE_FILE}")
                model_name=$(basename "${QUEUE_FILE}" .queue)
                
                total_found=$((total_found + 1))
                echo "  [${EXP}/${PROFILE}/${RES_FACTOR}] Submitting: ${queue_basename}"
                
                # Change to the directory containing the queue file
                cd "${RES_DIR}" || continue
                
                # Submit the job
                if command -v qsub &> /dev/null; then
                    if qsub "${QUEUE_FILE}"; then
                        total_submitted=$((total_submitted + 1))
                        echo "    ✓ Successfully submitted ${queue_basename}"
                    else
                        echo "    ✗ Failed to submit ${queue_basename}"
                    fi
                else
                    echo "    ⚠ qsub command not found - would submit: ${queue_basename}"
                    total_submitted=$((total_submitted + 1))  # Count as submitted for demo
                fi
            done
        done
    done
done

echo ""
echo "Job submission summary:"
echo "  Queue files found: ${total_found}"
echo "  Jobs submitted: ${total_submitted}"
echo "  Failed submissions: $((total_found - total_submitted))"
echo ""
echo "Use 'qstat' to check job status"
echo "Use 'qstat -u \$USER' to see only your jobs"
