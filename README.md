# AgentGrad

A modular reinforcement learning framework for continuous control tasks.

## TODO
- [ ] Add VPG Agent
- [ ] Add SAC Agent
- [ ] Add DDPG Agent
- [ ] Add TD3 Agent
- [ ] Add PPO Agent
- [ ] Add DQN Agent
- [ ] Init JAX environment
- [ ] TD3 in JAX
- [ ] PPO in JAX
- [ ] DQN in JAX

## Installation

First, install system dependencies:
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install xvfb mesa-utils xorg-dev swig

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then set up the Python environment:
```bash
# Create and activate conda environment
conda create -n agentgrad python=3.12
conda activate agentgrad

# Clone the repository
git clone https://github.com/mmbajo/AgentGrad.git
cd AgentGrad

# Install Python dependencies
uv install
```

## Training

```bash
# Train DDPG on LunarLander
python main.py env=lunar_lander agent=ddpg seed=42

# Train on different environments
python main.py env=humanoid agent=ddpg seed=42
python main.py env=ant agent=ddpg seed=42
```

## Evaluation

The evaluation script supports rendering and recording videos of trained agents. For headless servers, we provide a script that sets up a virtual display.

### Using the Evaluation Script

```bash
# Basic evaluation
./scripts/eval.sh env=lunar_lander seed=42

# Evaluate specific model checkpoint
./scripts/eval.sh env=humanoid seed=43 eval.model_name=best_model.pt

# Customize evaluation
./scripts/eval.sh env=ant seed=42 eval.episodes=20 eval.record_video=true
```

The evaluation results will be saved in the experiment directory:
```
experiment_dir/
├── eval/
│   ├── metrics_best_model.json    # Evaluation metrics
│   ├── summary_best_model.json    # Summary statistics
│   └── videos/                    # Recorded videos if enabled
│       └── eval_best_model-*.mp4
├── models/                        # Saved model checkpoints
└── metrics/                       # Training metrics
```

### Configuration

The framework uses Hydra for configuration. Key configuration files:
- `config/config.yaml`: Main configuration
- `config/env/*.yaml`: Environment-specific settings
- `config/agent/*.yaml`: Agent-specific settings

Common configuration options:
```yaml
# Training
seed: 42                  # Random seed
save.metrics_freq: 200    # Save metrics every N episodes
save.model_freq: 200      # Save model checkpoints every N episodes

# Evaluation
eval.episodes: 10         # Number of evaluation episodes
eval.record_video: true   # Whether to record videos
eval.model_name: "best_model.pt"  # Model to evaluate
```

### Other Citation Formats

**APA**
```
Bajo, M. (2024). AgentGrad: A Modular Reinforcement Learning Framework [Computer software]. https://github.com/mmbajo/AgentGrad
```

**IEEE**
```
M. Bajo, "AgentGrad: A Modular Reinforcement Learning Framework," GitHub repository, 2024. [Online]. Available: https://github.com/mmbajo/AgentGrad
```

## Project Structure

```
AgentGrad/
├── agents/               # Agent implementations
│   └── torch_ddpg/      # DDPG agent
├── config/              # Configuration files
├── scripts/             # Utility scripts
└── utils/               # Helper utilities
    └── metrics/         # Metrics tracking system
```

## Features

- Modular agent implementations
- Flexible configuration system
- Comprehensive metrics tracking
- Video recording of trained agents
- Support for multiple environments:
  - LunarLander
  - Humanoid
  - Ant
- Automatic checkpointing
- WandB integration for experiment tracking

## Citation

If you use AgentGrad in your research, please cite it using:

```bibtex
@software{agentgrad2024,
  author       = {Mark Bajo},
  title        = {AgentGrad: A Modular Reinforcement Learning Framework},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/mmbajo/AgentGrad}},
}
```

## License

MIT License