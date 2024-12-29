import numpy as np
from typing import Dict, List, Any
from utils.metrics.base_metrics import BaseMetricAccumulator


class MinimalMetrics(BaseMetricAccumulator):
    """Minimal metric accumulator that works with any environment.
    
    This class provides basic metrics tracking without any environment-specific metrics.
    It can be used as a fallback when no specific metrics class is available for an environment.
    """
    
    def _initialize_env_metrics(self) -> None:
        """Initialize minimal metrics (none needed)."""
        pass
    
    def _update_env_metrics(
        self, state: np.ndarray, action: np.ndarray, info: Dict[str, Any]
    ) -> None:
        """Update minimal metrics (none needed).
        
        Args:
            state: Environment state
            action: Action taken
            info: Environment info dictionary
        """
        pass
    
    def _write_env_tensorboard(self) -> None:
        """Write minimal metrics to tensorboard (none needed)."""
        pass
    
    def _get_env_wandb_metrics(self) -> Dict[str, float]:
        """Get minimal metrics for wandb logging (none needed)."""
        return {}
    
    def _get_env_metric_names(self) -> List[str]:
        """Get list of minimal metric names (none)."""
        return []
    
    def _get_env_metrics_dict(self) -> Dict[str, List[Any]]:
        """Get dictionary of minimal metrics (none)."""
        return {}
    
    def _clear_env_metrics(self) -> None:
        """Clear minimal metrics (none needed)."""
        pass 