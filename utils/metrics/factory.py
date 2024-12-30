from typing import Optional, Type, Any
from utils.metrics.base_metrics import BaseMetricAccumulator
from utils.metrics.lunar_lander_metrics import LunarLanderMetrics
from utils.metrics.minimal_metrics import MinimalMetrics
from loguru import logger


class MetricsFactory:
    """Factory class to create appropriate metrics for different environments."""

    _registry = {
        "LunarLanderContinuous-v3": LunarLanderMetrics,
        # Add more environments here
    }

    @classmethod
    def register(cls, env_id: str, metrics_class: Type[BaseMetricAccumulator]) -> None:
        """Register a new environment-specific metrics class.

        Args:
            env_id: Environment ID (e.g., "LunarLanderContinuous-v3")
            metrics_class: Metrics class for this environment
        """
        cls._registry[env_id] = metrics_class

    @classmethod
    def create(
        cls,
        env_id: str,
        writer: Optional[Any] = None,
        use_wandb: bool = True,
    ) -> BaseMetricAccumulator:
        """Create metrics accumulator for the specified environment.

        Args:
            env_id: Environment ID (e.g., "LunarLanderContinuous-v3")
            writer: Optional tensorboard writer
            use_wandb: Whether to use wandb logging

        Returns:
            Appropriate metrics accumulator for the environment.
            If no specific metrics class is registered for the environment,
            returns a MinimalMetrics instance that tracks only basic metrics.
        """
        if env_id not in cls._registry:
            logger.warning(
                f"No metrics class registered for environment {env_id}. "
                f"Using minimal metrics. Available environments: {list(cls._registry.keys())}"
            )
            return MinimalMetrics(writer=writer, use_wandb=use_wandb)

        metrics_class = cls._registry[env_id]
        return metrics_class(writer=writer, use_wandb=use_wandb)
