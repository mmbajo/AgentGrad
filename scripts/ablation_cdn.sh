#!/bin/bash

# Enable error handling
set -e

# List of environments (excluding humanoid)
ENVS=(hopper walker pendulum)
MAX_EPS=(10000 10000 2000)
AGENTS=(td3 sac)
SEEDS=(0 1 2 3 4)  # 10 seeds

# Save frequencies
SAVE_FREQ=1000

# Maximum number of parallel processes
MAX_PARALLEL=3

# Create logs directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="outputs/ablation_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# Log file
LOG_FILE="${LOG_DIR}/ablation.log"

# Progress tracking
TOTAL_RUNS=$((${#ENVS[@]} * ${#AGENTS[@]} * ${#SEEDS[@]} * 2))  # *2 for double_q true/false
CURRENT_RUN=0

log() {
    local message="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $message" | tee -a "$LOG_FILE"
}

# Function to run training
run_training() {
    local env=$1
    local agent=$2
    local seed=$3
    local max_eps=$4
    local use_double_q=$5

    CURRENT_RUN=$((CURRENT_RUN + 1))
    log "Progress: ${CURRENT_RUN}/${TOTAL_RUNS} - Running: env=$env, agent=$agent, seed=$seed, max_eps=$max_eps, use_double_q=$use_double_q"
    
    # Run training
    if ! uv run main.py \
        --config-name cdq_ablation \
        env=$env \
        env.max_episodes=$max_eps \
        agent=$agent \
        seed=$seed \
        agent.use_double_q=$use_double_q \
        save.metrics_freq=$SAVE_FREQ \
        save.model_freq=$SAVE_FREQ \
        2>&1 | tee -a "${LOG_DIR}/${env}_${agent}_${seed}_double_q_${use_double_q}.log"; then
        
        log "ERROR: Failed run with env=$env agent=$agent seed=$seed use_double_q=$use_double_q"
        echo "$env,$agent,$seed,$use_double_q" >> "${LOG_DIR}/failed_runs.txt"
        return 1
    fi
}

# Function to wait for available slot
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 1
    done
}

# Function to run training in background
run_training_parallel() {
    local env=$1
    local agent=$2
    local seed=$3
    local max_eps=$4
    local use_double_q=$5
    
    # Wait for available slot
    wait_for_slot
    
    # Run training in background
    run_training "$env" "$agent" "$seed" "$max_eps" "$use_double_q" &
}

# Print experiment info
log "Starting ablation study with $MAX_PARALLEL parallel processes"
log "Total runs: $TOTAL_RUNS"
log "Environments: ${ENVS[*]}"
log "Agents: ${AGENTS[*]}"
log "Seeds: ${SEEDS[*]}"
log "Save frequency: $SAVE_FREQ"
log "Log directory: $LOG_DIR"

# Track start time
START_TIME=$(date +%s)

# Main execution loop
for i in "${!ENVS[@]}"; do
    env=${ENVS[$i]}
    max_eps=${MAX_EPS[$i]}
    for seed in "${SEEDS[@]}"; do
        for agent in "${AGENTS[@]}"; do
            for use_double_q in false true; do
                run_training_parallel "$env" "$agent" "$seed" "$max_eps" "$use_double_q"
            done
        done
    done
done

# Wait for all remaining jobs to complete
log "Waiting for remaining jobs to complete..."
wait

# Calculate and log duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

log "Ablation study completed!"
log "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
log "Results saved in: $LOG_DIR"

# Check for failed runs
if [[ -f "${LOG_DIR}/failed_runs.txt" ]]; then
    log "WARNING: Some runs failed. Check ${LOG_DIR}/failed_runs.txt for details"
    exit 1
fi

exit 0
