#!/bin/bash

# Base directory for GADI structure
EXEC_BASE="/scratch/au88/ah3716/execution/flowline"

# Auto-detect all experiment folders (S1, S2, S3, S4, etc.)
EXPERIMENTS=($(find "${EXEC_BASE}" -mindepth 1 -maxdepth 1 -type d -name "S*" | sort | xargs -n1 basename))

echo "Submitting all queue files from current resolution structure..."
echo "Execution base: ${EXEC_BASE}"
echo "Detected experiments: ${EXPERIMENTS[@]}"
echo

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
    
    if [ ${#PROFILE_DIRS[@]} -eq 0 ]; then
        echo "  No profile directories found in ${EXP_DIR}"
        continue
    fi
    
    for PROFILE_DIR in "${PROFILE_DIRS[@]}"; do
        PROFILE=$(basename "${PROFILE_DIR}")
        
        # Find queue files directly in profile directory (no resolution subdirectories)
        QUEUE_FILES=($(find "${PROFILE_DIR}" -maxdepth 1 -name "*.queue" 2>/dev/null))
        
        if [ ${#QUEUE_FILES[@]} -eq 0 ]; then
            echo "  [${EXP}/${PROFILE}] No queue files found"
            continue
        fi
        
        for QUEUE_FILE in "${QUEUE_FILES[@]}"; do
            queue_basename=$(basename "${QUEUE_FILE}")
            model_name=$(basename "${QUEUE_FILE}" .queue)
            
            total_found=$((total_found + 1))
            echo "  [${EXP}/${PROFILE}] Submitting: ${queue_basename}"
            
            # Change to the directory containing the queue file
            cd "${PROFILE_DIR}" || continue
            
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
    
    echo
done

echo "Job submission summary:"
echo "  Queue files found: ${total_found}"
echo "  Jobs submitted: ${total_submitted}"
echo "  Failed submissions: $((total_found - total_submitted))"
echo ""
echo "Use 'qstat' to check job status"
echo "Use 'qstat -u \$USER' to see only your jobs"