from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np
import json
from pathlib import Path
from tabulate import tabulate
import wandb


class BaseMetricAccumulator(ABC):
    """Base class for metric accumulation and tracking.

    This class provides the basic structure for collecting and analyzing metrics
    during training or evaluation. Specific environment metrics should inherit
    from this class and implement the abstract methods.
    """

    def __init__(
        self,
        writer: Optional[Any] = None,
        use_wandb: bool = True,
    ):
        """Initialize the metric accumulator.

        Args:
            writer: Optional tensorboard writer for logging
            use_wandb: Whether to use wandb logging
        """
        self._writer = writer
        self._use_wandb = use_wandb
        self._episode = 0

        # Basic metrics that all environments should track
        self._rewards: List[float] = []
        self._episode_length: List[int] = []
        self._actor_losses: List[float] = []
        self._critic_losses: List[float] = []
        self._states: List[np.ndarray] = []
        self._actions: List[np.ndarray] = []

        # Initialize environment-specific metrics
        self._initialize_env_metrics()

    @abstractmethod
    def _initialize_env_metrics(self) -> None:
        """Initialize environment-specific metrics.

        This method should be implemented by subclasses to initialize
        any additional metrics specific to their environment.
        """
        pass

    @abstractmethod
    def _update_env_metrics(
        self, state: np.ndarray, action: np.ndarray, info: Dict[str, Any]
    ) -> None:
        """Update environment-specific metrics.

        Args:
            state: Current environment state
            action: Action taken in the environment
            info: Environment info dictionary
        """
        pass

    @abstractmethod
    def _write_env_tensorboard(self) -> None:
        """Write environment-specific metrics to tensorboard."""
        pass

    @abstractmethod
    def _get_env_wandb_metrics(self) -> Dict[str, float]:
        """Get environment-specific metrics for wandb logging.

        Returns:
            Dictionary of metric name to current value
        """
        pass

    def push_back(
        self,
        reward: float,
        critic_loss: float,
        actor_loss: float,
        length: int,
        action: np.ndarray,
        state: np.ndarray,
        info: Dict[str, Any],
    ) -> None:
        """Add a new set of metrics from the current episode.

        Args:
            reward: Episode reward
            critic_loss: Critic loss value
            actor_loss: Actor loss value
            length: Episode length
            action: Action taken
            state: Environment state
            info: Environment info dictionary
        """
        # Update basic metrics
        self._rewards.append(reward)
        self._critic_losses.append(critic_loss)
        self._actor_losses.append(actor_loss)
        self._episode_length.append(length)
        self._states.append(state)
        self._actions.append(action)

        # Update environment-specific metrics
        self._update_env_metrics(state, action, info)

        # Log metrics
        if self._writer:
            self._write_to_tensorboard()
        if self._use_wandb:
            self._write_to_wandb()

        self._episode += 1

    def _write_to_tensorboard(self) -> None:
        """Write all metrics to tensorboard."""
        if not self._writer:
            return

        # Write basic metrics
        self._writer.add_scalar("Metrics/Reward", self._rewards[-1], self._episode)
        self._writer.add_scalar(
            "Metrics/Episode Length", self._episode_length[-1], self._episode
        )
        self._writer.add_scalar(
            "Metrics/Total Critic Loss", self._critic_losses[-1], self._episode
        )
        self._writer.add_scalar(
            "Metrics/Total Actor Loss", self._actor_losses[-1], self._episode
        )

        # Write environment-specific metrics
        self._write_env_tensorboard()

    def _write_to_wandb(self) -> None:
        """Write all metrics to wandb."""
        if not self._use_wandb:
            return

        metrics = {
            "reward": self._rewards[-1],
            "episode_length": self._episode_length[-1],
            "total_critic_loss": self._critic_losses[-1],
            "total_actor_loss": self._actor_losses[-1],
            "episode": self._episode,
        }

        # Add environment-specific metrics
        metrics.update(self._get_env_wandb_metrics())

        wandb.log(metrics)

    def get_metric_names(self) -> List[str]:
        """Get list of all metric names."""
        basic_metrics = ["rewards", "episode_length", "actor_losses", "critic_losses"]
        env_metrics = self._get_env_metric_names()
        return basic_metrics + env_metrics

    @abstractmethod
    def _get_env_metric_names(self) -> List[str]:
        """Get list of environment-specific metric names."""
        pass

    def save(self, path: Path, is_training: bool = True) -> None:
        """Save metrics to file.

        Args:
            path: Path to save metrics file
            is_training: Whether these are training metrics
        """
        metrics = {
            "rewards": self._rewards,
            "episode_length": self._episode_length,
            "actor_losses": self._actor_losses,
            "critic_losses": self._critic_losses,
        }

        # Add environment-specific metrics
        metrics.update(self._get_env_metrics_dict())

        # Save metrics
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)

    @abstractmethod
    def _get_env_metrics_dict(self) -> Dict[str, List[Any]]:
        """Get dictionary of environment-specific metrics."""
        pass

    def summarize_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate summary statistics for all metrics."""
        summary = {}
        for metric_name in self.get_metric_names():
            metric_data = getattr(self, f"_{metric_name}")
            if metric_data:
                summary[metric_name] = {
                    "mean": float(np.mean(metric_data)),
                    "std": float(np.std(metric_data)),
                    "min": float(np.min(metric_data)),
                    "max": float(np.max(metric_data)),
                }
        return summary

    def save_summary(self, path: Path, is_training: bool = True) -> None:
        """Save summary statistics to file.

        Args:
            path: Path to save summary file
            is_training: Whether these are training metrics
        """
        summary = self.summarize_metrics()
        with open(path, "w") as f:
            json.dump(summary, f, indent=4)

    @classmethod
    def load(cls, path: Path, is_training: bool = True) -> "BaseMetricAccumulator":
        """Load metrics from files."""
        _t = "training" if is_training else "eval"
        with open(path / f"{_t}_metrics.json", "r") as f:
            metrics = json.load(f)

        states = np.load(path / f"{_t}_states.npy")
        actions = np.load(path / f"{_t}_actions.npy")

        accumulator = cls(writer=None, use_wandb=False)
        for attr, values in metrics.items():
            setattr(accumulator, f"_{attr}", values)
        accumulator._states = list(states)
        accumulator._actions = list(actions)
        accumulator._episode = len(accumulator._rewards)

        return accumulator

    def print_summary(self) -> None:
        """Print a formatted summary of all metrics."""
        summary = self.summarize_metrics()
        table_data = []
        for metric, stats in summary.items():
            table_data.append(
                [
                    metric.replace("_", " ").title(),
                    f"{stats['mean']:.4f}",
                    f"{stats['std']:.4f}",
                    f"{stats['min']:.4f}",
                    f"{stats['max']:.4f}",
                ]
            )
        print(
            tabulate(
                table_data,
                headers=["Metric", "Mean", "Std Dev", "Min", "Max"],
                tablefmt="grid",
                numalign="right",
                stralign="left",
            )
        )

    def clear_metrics(self) -> None:
        """Clear all accumulated metrics to free memory."""
        self._rewards.clear()
        self._episode_length.clear()
        self._actor_losses.clear()
        self._critic_losses.clear()
        self._states.clear()
        self._actions.clear()
        self._clear_env_metrics()

    @abstractmethod
    def _clear_env_metrics(self) -> None:
        """Clear environment-specific metrics.

        This method should be implemented by subclasses to clear
        any additional metrics specific to their environment.
        """
        pass
