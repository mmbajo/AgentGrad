import numpy as np
from typing import Dict, List, Any
from utils.metrics.base_metrics import BaseMetricAccumulator


class LunarLanderMetrics(BaseMetricAccumulator):
    """Metric accumulator specific to the LunarLander environment."""

    def _initialize_env_metrics(self) -> None:
        """Initialize LunarLander-specific metrics."""
        self._main_engine_fuel_used: List[float] = []
        self._left_right_engine_fuel_used: List[float] = []
        self._dist_from_goal: List[float] = []
        self._velocity_magnitude: List[float] = []
        self._tilt: List[float] = []
        self._angular_velocity_magnitude: List[float] = []
        self._success: List[int] = []

    def _update_env_metrics(
        self, state: np.ndarray, action: np.ndarray, info: Dict[str, Any]
    ) -> None:
        """Update LunarLander-specific metrics.

        Args:
            state: LunarLander state [x, y, vx, vy, angle, v_angle, left_leg, right_leg]
            action: Action [main_engine, left_right_engine]
            info: Environment info dictionary
        """
        # Update action-based metrics
        self._main_engine_fuel_used.append(float(action[0]))
        self._left_right_engine_fuel_used.append(float(action[1]))

        # Update state-based metrics
        x, y, vx, vy, angle, v_angle, left_leg, right_leg = state.tolist()

        def _mag(x: float, y: float) -> float:
            return np.sqrt(x * x + y * y)

        def _is_success(reward: float, left_leg: int, right_leg: int) -> int:
            return 1 if (left_leg + right_leg == 2) and reward >= 200 else 0

        self._dist_from_goal.append(_mag(x, y))
        self._velocity_magnitude.append(_mag(vx, vy))
        self._tilt.append(abs(angle))
        self._angular_velocity_magnitude.append(abs(v_angle))
        self._success.append(_is_success(self._rewards[-1], left_leg, right_leg))

    def _write_env_tensorboard(self) -> None:
        """Write LunarLander-specific metrics to tensorboard."""
        if not self._writer:
            return

        self._writer.add_scalar(
            "Metrics/Distance from Goal", self._dist_from_goal[-1], self._episode
        )
        self._writer.add_scalar(
            "Metrics/Velocity Magnitude", self._velocity_magnitude[-1], self._episode
        )
        self._writer.add_scalar("Metrics/Tilt", self._tilt[-1], self._episode)
        self._writer.add_scalar(
            "Metrics/Angular Velocity Magnitude",
            self._angular_velocity_magnitude[-1],
            self._episode,
        )
        self._writer.add_scalar(
            "Metrics/Main Engine Usage",
            self._main_engine_fuel_used[-1],
            self._episode,
        )
        self._writer.add_scalar(
            "Metrics/Left-Right Engine Usage",
            self._left_right_engine_fuel_used[-1],
            self._episode,
        )
        self._writer.add_scalar(
            "Metrics/Success",
            self._success[-1],
            self._episode,
        )

    def _get_env_wandb_metrics(self) -> Dict[str, float]:
        """Get environment-specific metrics for wandb logging."""
        return {
            "distance_from_goal": self._dist_from_goal[-1],
            "velocity_magnitude": self._velocity_magnitude[-1],
            "tilt": self._tilt[-1],
            "angular_velocity_magnitude": self._angular_velocity_magnitude[-1],
            "main_engine_usage": self._main_engine_fuel_used[-1],
            "left_right_engine_usage": self._left_right_engine_fuel_used[-1],
            "success": self._success[-1],
        }

    def _get_env_metric_names(self) -> List[str]:
        """Get list of LunarLander-specific metric names."""
        return [
            "main_engine_fuel_used",
            "left_right_engine_fuel_used",
            "dist_from_goal",
            "velocity_magnitude",
            "tilt",
            "angular_velocity_magnitude",
            "success",
        ]

    def _get_env_metrics_dict(self) -> Dict[str, List[Any]]:
        """Get dictionary of LunarLander-specific metrics."""
        return {
            "main_engine_fuel_used": self._main_engine_fuel_used,
            "left_right_engine_fuel_used": self._left_right_engine_fuel_used,
            "dist_from_goal": self._dist_from_goal,
            "velocity_magnitude": self._velocity_magnitude,
            "tilt": self._tilt,
            "angular_velocity_magnitude": self._angular_velocity_magnitude,
            "success": self._success,
        }

    def _clear_env_metrics(self) -> None:
        """Clear LunarLander-specific metrics."""
        self._main_engine_fuel_used.clear()
        self._left_right_engine_fuel_used.clear()
        self._dist_from_goal.clear()
        self._velocity_magnitude.clear()
        self._tilt.clear()
        self._angular_velocity_magnitude.clear()
        self._success.clear()
