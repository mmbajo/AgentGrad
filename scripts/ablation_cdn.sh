#!/bin/bash

# List of environments (excluding humanoid)
ENVS=("lunar_lander" "ant" "hopper" "walker" "pendulum")
MAX_EPS=(10000 10000 10000 10000 2000)
AGENTS=("td3" "sac")
SEEDS=(42 43 44 45 46 47 48 49 50 51)  # 10 seeds

# Save frequencies
SAVE_FREQ=1000

# Function to run evaluation
run_eval() {
    local env=$1
    local agent=$2
    local seed=$3
    local max_eps=$4
    
    echo "Running evaluation: env=$env, agent=$agent, seed=$seed, double Q=false"
    
    # Run training with evaluation
    ./scripts/eval.sh \
        env=$env \
        env.max_episodes=$max_eps \
        agent=$agent \
        seed=$seed \
        agent.use_double_q=false \
        save.metrics_freq=$SAVE_FREQ \
        save.model_freq=$SAVE_FREQ
}

# Main execution loop
for i in "${!ENVS[@]}"; do
    env=${ENVS[$i]}
    max_eps=${MAX_EPS[$i]}
    for seed in "${SEEDS[@]}"; do
        for agent in "${AGENTS[@]}"; do
            run_eval "$env" "$agent" "$seed" "$max_eps"
        done
    done
done

echo "Ablation study completed!"
