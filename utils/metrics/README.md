# Metrics System

This directory contains a modular metrics tracking system for reinforcement learning environments. The system is designed to be easily extensible for new environments while providing robust default functionality.

## Overview

The metrics system consists of:
- A base abstract class (`BaseMetricAccumulator`) that defines the interface
- Environment-specific implementations (e.g., `LunarLanderMetrics`)
- A minimal implementation (`MinimalMetrics`) for new/unsupported environments
- A factory class (`MetricsFactory`) for creating appropriate metrics instances

## Basic Usage

```python
from utils.metrics.factory import MetricsFactory
from torch.utils.tensorboard import SummaryWriter

# Create metrics for a supported environment
metrics = MetricsFactory.create(
    env_id="LunarLanderContinuous-v3",
    writer=SummaryWriter(...),  # Optional
    use_wandb=True,  # Optional
)

# Use in training loop
metrics.push_back(
    reward=episode_reward,
    critic_loss=total_critic_loss,
    actor_loss=total_actor_loss,
    length=episode_steps,
    action=action,
    state=state,
    info=info,
)

# Save and summarize
metrics.save(log_dir)
metrics.save_summary(log_dir)
metrics.print_summary()
```

## Core Features

### Basic Metrics (All Environments)
- Episode rewards
- Episode lengths
- Actor/Critic losses
- State/Action history

### Logging Support
- Tensorboard integration
- Weights & Biases (wandb) integration
- JSON file saving
- Pretty-printed summaries

### Environment-Specific Metrics
Environment-specific implementations can track additional metrics. For example, LunarLander tracks:
- Distance from goal
- Velocity magnitude
- Tilt and angular velocity
- Engine usage
- Landing success

## Adding Support for New Environments

1. Create a new class inheriting from `BaseMetricAccumulator`:
```python
class NewEnvMetrics(BaseMetricAccumulator):
    def _initialize_env_metrics(self) -> None:
        self._custom_metric1: List[float] = []
        self._custom_metric2: List[float] = []
    
    def _update_env_metrics(
        self, state: np.ndarray, action: np.ndarray, info: Dict[str, Any]
    ) -> None:
        # Update your custom metrics
        self._custom_metric1.append(...)
        self._custom_metric2.append(...)
    
    def _write_env_tensorboard(self) -> None:
        if not self._writer:
            return
        self._writer.add_scalar("Custom/Metric1", self._custom_metric1[-1], self._episode)
        self._writer.add_scalar("Custom/Metric2", self._custom_metric2[-1], self._episode)
    
    def _get_env_wandb_metrics(self) -> Dict[str, float]:
        return {
            "custom_metric1": self._custom_metric1[-1],
            "custom_metric2": self._custom_metric2[-1],
        }
    
    def _get_env_metric_names(self) -> List[str]:
        return ["custom_metric1", "custom_metric2"]
    
    def _get_env_metrics_dict(self) -> Dict[str, List[Any]]:
        return {
            "custom_metric1": self._custom_metric1,
            "custom_metric2": self._custom_metric2,
        }
```

2. Register your metrics class:
```python
MetricsFactory.register("NewEnv-v1", NewEnvMetrics)
```

## Fallback Behavior

For environments without specific metric implementations, the system automatically falls back to `MinimalMetrics`, which:
- Tracks only the basic metrics (rewards, losses, episode lengths)
- Works with any environment without modification
- Provides all the saving/loading/summary functionality

## File Structure

```
metrics/
├── README.md
├── __init__.py
├── base_metrics.py       # Abstract base class
├── factory.py           # Factory for creating metrics
├── minimal_metrics.py   # Minimal implementation
└── lunar_lander_metrics.py  # Example environment-specific implementation
```

## Configuration

The metrics system can be configured through your Hydra config:

```yaml
# config.yaml
tensorboard:
  enabled: false  # Whether to use tensorboard logging

wandb:
  mode: online  # Set to disabled to turn off wandb logging
``` 