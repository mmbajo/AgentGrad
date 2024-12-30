#!/bin/bash

# Check if at least one argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: ./scripts/eval.sh env=environment_name [seed=value] [eval.model_name=model_name.pt] [other_hydra_args...]"
    echo "Example: ./scripts/eval.sh env=humanoid seed=43 eval.model_name=best_model.pt"
    exit 1
fi

# Start virtual display
Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
export DISPLAY=:1

# Wait a moment for Xvfb to start
sleep 2

# Run evaluation script with all provided arguments
echo "Running evaluation with arguments: $@"
uv run eval.py "$@"

# Clean up
killall Xvfb